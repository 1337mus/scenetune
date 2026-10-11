"""CPU-only planning and validation for the opt-in Controlled3D study."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from . import controlled3d, controlled3d_assets, controlled3d_schema
from . import controlled3d_study_payload as study_payload
from .matched_study import MAX_PAYLOAD_BYTES, TRAINING_EPOCHS, TRAINING_STEPS

APPROVED_CONTROLLED3D_MANIFEST_SHA256: str | None = (
    "b6c4e7ea4c20fafbded78feaef68b1b8f8f24fdc132a3a29c25570edb4bb099b"
)
APPROVED_CONTROLLED3D_REGISTRY_SHA256: str | None = (
    "1bfa7bf4604d8ef0bb06c5b5da44f7be8bc1297b4cdba1e1baa23d42849d4253"
)
RUN_MARKER_NAME = ".controlled3d-study-run.json"
PAYLOAD_NAME = "controlled3d-study-input.zip"
PROOF_NAME = "controlled3d-validation-proof.json"
validate_controlled3d_study_payload_bytes = study_payload.validate_controlled3d_study_payload_bytes
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_STATES = ("cream", "sage", "clay")
_PRODUCT_COUNTS = {"train": 6, "dev": 3}


@dataclass(frozen=True)
class Controlled3DStudyPlan:
    """A write-once local input bundle and immutable provenance marker."""

    output_dir: Path
    payload_path: Path
    run_id: str
    payload_bytes: int
    payload_sha256: str
    source_dataset_manifest_sha256: str
    source_registry_sha256: str
    validation_proof_sha256: str
    train_export_completion_sha256: str
    training_pair_ids: tuple[str, ...]
    development_pair_ids: tuple[str, ...]


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_pin(value: object, label: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ValueError(f"Controlled3D {label} digest pin is unset")
    return value


def _read_regular(path: str | Path, max_bytes: int) -> bytes:
    target = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(target, flags)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
                raise ValueError(f"unsafe or oversized file: {target.name}")
            raw = stream.read(max_bytes + 1)
    except OSError as exc:
        raise ValueError(f"file is missing or unsafe: {target.name}") from exc
    if len(raw) > max_bytes:
        raise ValueError(f"file exceeds the configured size limit: {target.name}")
    return raw


def approve_manifest_pin(manifest_path: str | Path) -> str:
    """Hash the manifest only after its reviewed digest has been bound in source."""
    expected = _require_pin(APPROVED_CONTROLLED3D_MANIFEST_SHA256, "manifest")
    raw = _read_regular(manifest_path, controlled3d_schema._MAX_JSON_BYTES)
    digest = _digest(raw)
    if digest != expected:
        raise ValueError("Controlled3D manifest digest does not match the reviewed pin")
    return digest


def _tree_files(root: Path) -> set[str]:
    found: set[str] = set()
    for current, directories, filenames in os.walk(root, followlinks=False):
        base = Path(current)
        for name in directories:
            if stat.S_ISLNK(os.lstat(base / name).st_mode):
                raise ValueError("train export contains a symlink directory")
        for name in filenames:
            path = base / name
            mode = os.lstat(path).st_mode
            if not stat.S_ISREG(mode):
                raise ValueError("train export contains a non-regular file")
            found.add(path.relative_to(root).as_posix())
    return found


def _train_export_expected(manifest: dict[str, Any]) -> tuple[dict[str, str], list[dict[str, str]]]:
    pairs = [row for row in manifest["pairs"] if row["split"] == "train"]
    exported = {
        "schema_version": 1,
        "provenance": manifest["provenance"],
        "assignments": [row for row in manifest["assignments"] if row["split"] == "train"],
        "pairs": pairs,
    }
    expected: dict[str, str] = {}
    metadata: list[dict[str, str]] = []
    for pair in pairs:
        renders = {row["name"]: row for row in pair["renders"]}
        for ref in (
            pair["snapshot_bundle"],
            *[item for render in pair["renders"] for item in (render["image"], render["mask"])],
            *pair["pins"]["local_files"],
        ):
            if ref["path"] in expected:
                raise ValueError("train export references a reused asset path")
            expected[ref["path"]] = ref["sha256"]
        source = renders["source"]["image"]["path"]
        for state in _STATES:
            metadata.append(
                {
                    "image": renders[state]["image"]["path"],
                    "edit_image": source,
                    "prompt": (
                        "Keep the product, label, and geometry unchanged; "
                        f"change only the background to the {state} variant."
                    ),
                }
            )
    expected["manifest.json"] = _digest(controlled3d_schema._dump(exported).encode("utf-8"))
    expected["metadata.json"] = _digest(controlled3d_schema._dump(metadata).encode("utf-8"))
    return expected, metadata


def _validate_train_export(
    export_dir: Path, manifest: dict[str, Any]
) -> tuple[dict[str, bytes], str]:
    if export_dir.is_symlink() or not export_dir.is_dir():
        raise ValueError("Controlled3D train export is missing or unsafe")
    expected, _metadata = _train_export_expected(manifest)
    completion_raw = _read_regular(export_dir / "completion.json", 8 * 1024 * 1024)
    completion = controlled3d_schema._loads_any(completion_raw)
    if (
        type(completion) is not dict
        or set(completion) != {"schema_version", "files"}
        or type(completion["schema_version"]) is not int
        or completion["schema_version"] != 1
        or type(completion["files"]) is not list
    ):
        raise ValueError("Controlled3D train export completion receipt is invalid")
    rows = completion["files"]
    if rows != [{"path": name, "sha256": expected[name]} for name in sorted(expected)]:
        raise ValueError(
            "Controlled3D train export file index differs from the validated train split"
        )
    if _tree_files(export_dir) != set(expected) | {"completion.json"}:
        raise ValueError("Controlled3D train export contains missing or unexpected files")
    contents: dict[str, bytes] = {}
    for name, digest in expected.items():
        raw = _read_regular(export_dir / Path(*PurePosixPath(name).parts), 32 * 1024 * 1024)
        if _digest(raw) != digest:
            raise ValueError(f"Controlled3D train export file changed: {name}")
        contents[name] = raw
    canonical_completion = controlled3d_schema._dump(completion).encode("utf-8")
    if completion_raw != canonical_completion:
        raise ValueError("Controlled3D train export completion is not canonical JSON")
    return contents, _digest(completion_raw)


def _validation_proof(
    manifest: dict[str, Any], validated: Any, manifest_sha256: str, registry_sha256: str
) -> bytes:
    pairs = []
    for pair in sorted(
        (row for row in manifest["pairs"] if row["split"] in {"train", "dev"}),
        key=lambda row: row["pair_id"],
    ):
        renders = {row["name"]: row for row in pair["renders"]}
        pairs.append(
            {
                "pair_id": pair["pair_id"],
                "product_id": pair["product_id"],
                "split": pair["split"],
                "expected_words": pair["expected_words"],
                "font": pair["pins"]["font"],
                "scene_sha256": pair["pins"]["scene_sha256"],
                "snapshot_bundle_sha256": pair["snapshot_bundle"]["sha256"],
                "masks": [
                    {
                        "state": state,
                        "file_sha256": renders[state]["mask"]["sha256"],
                        "decoded_sha256": renders[state]["mask"]["decoded_sha256"],
                        "bbox": renders[state]["bbox"],
                    }
                    for state in ("source", *_STATES)
                ],
            }
        )
    return _json_bytes(
        {
            "schema_version": 1,
            "source_dataset_manifest_sha256": manifest_sha256,
            "source_registry_sha256": registry_sha256,
            "validated_metrics": controlled3d_schema._loads_any(validated.metrics_json),
            "pairs": pairs,
        }
    )


def _verify_validation_proof(raw: bytes, payload: Any) -> None:
    proof = controlled3d_schema._loads_any(raw)
    if (
        type(proof) is not dict
        or set(proof)
        != {
            "schema_version",
            "source_dataset_manifest_sha256",
            "source_registry_sha256",
            "validated_metrics",
            "pairs",
        }
        or proof["schema_version"] != 1
        or proof["source_dataset_manifest_sha256"] != payload.source_dataset_manifest_sha256
        or proof["source_registry_sha256"] != payload.source_registry_sha256
        or raw != _json_bytes(proof)
    ):
        raise ValueError("Controlled3D validation proof does not bind the reviewed inputs")
    rows = {
        (row["split"], row["scene_pair_id"], row["product_id"])
        for row in (*payload.train_rows, *payload.dev_rows)
    }
    expected = {(split, pair, product) for split, pair, product in rows}
    actual = {
        (row.get("split"), row.get("pair_id"), row.get("product_id")) for row in proof["pairs"]
    }
    if actual != expected or len(actual) != len(proof["pairs"]):
        raise ValueError("Controlled3D validation proof pair set differs from the payload")
    for row in proof["pairs"]:
        masks = row.get("masks")
        if (
            not isinstance(row.get("expected_words"), list)
            or not isinstance(row.get("font"), dict)
            or not isinstance(row.get("scene_sha256"), str)
            or not _DIGEST.fullmatch(row["scene_sha256"])
            or not isinstance(row.get("snapshot_bundle_sha256"), str)
            or not _DIGEST.fullmatch(row["snapshot_bundle_sha256"])
            or not isinstance(masks, list)
            or [mask.get("state") for mask in masks] != ["source", *_STATES]
            or any(
                not isinstance(mask.get("file_sha256"), str)
                or not _DIGEST.fullmatch(mask["file_sha256"])
                or not isinstance(mask.get("decoded_sha256"), str)
                or not _DIGEST.fullmatch(mask["decoded_sha256"])
                or not isinstance(mask.get("bbox"), list)
                for mask in masks
            )
        ):
            raise ValueError("Controlled3D validation proof has invalid captured evidence")


def _check_counts(manifest: dict[str, Any]) -> None:
    counts = Counter(pair["split"] for pair in manifest["pairs"])
    products = {
        split: {row["product_id"] for row in manifest["pairs"] if row["split"] == split}
        for split in _PRODUCT_COUNTS
    }
    if (
        dict(counts) != _PRODUCT_COUNTS
        or {key: len(value) for key, value in products.items()} != _PRODUCT_COUNTS
    ):
        raise ValueError("Controlled3D study requires six train and three development products")
    if any(
        count != 1 for count in Counter(row["product_id"] for row in manifest["pairs"]).values()
    ):
        raise ValueError("Controlled3D study requires exactly one reviewed pair per product")


def plan_controlled3d_study(
    dataset_root: str | Path,
    manifest_path: str | Path,
    registry_path: str | Path,
    train_export_dir: str | Path,
    output_dir: str | Path,
    *,
    run_id: str,
) -> Controlled3DStudyPlan:
    """Validate reviewed inputs, build a train/dev-only bundle, and bind it write-once."""
    manifest_pin = _require_pin(APPROVED_CONTROLLED3D_MANIFEST_SHA256, "manifest")
    registry_pin = _require_pin(APPROVED_CONTROLLED3D_REGISTRY_SHA256, "registry")
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id) or ".." in run_id:
        raise ValueError("invalid Controlled3D run identifier")
    root = Path(dataset_root)
    root_identity = controlled3d_assets.root_identity(root)
    manifest_raw = _read_regular(manifest_path, controlled3d_schema._MAX_JSON_BYTES)
    manifest_sha256 = _digest(manifest_raw)
    if manifest_sha256 != manifest_pin:
        raise ValueError("Controlled3D manifest digest does not match the reviewed pin")
    registry_raw = _read_regular(registry_path, controlled3d_schema._MAX_JSON_BYTES)
    registry_sha256 = _digest(registry_raw)
    if registry_sha256 != registry_pin:
        raise ValueError("Controlled3D registry digest does not match the reviewed pin")
    manifest = controlled3d_schema._manifest_shape(controlled3d_schema._loads_any(manifest_raw))
    _check_counts(manifest)
    validated = controlled3d.validate_controlled3d(
        root,
        manifest,
        reviewed_registry=registry_raw,
        registry_sha256=registry_sha256,
    )
    if validated.root_identity != root_identity:
        raise ValueError("Controlled3D dataset root changed during validation")
    export_dir = Path(train_export_dir)
    export_files, export_sha256 = _validate_train_export(export_dir, manifest)
    proof_raw = _validation_proof(manifest, validated, manifest_sha256, registry_sha256)
    proof_sha256 = _digest(proof_raw)
    study_files: dict[str, bytes] = {}
    train_rows: list[dict[str, Any]] = []
    dev_rows: list[dict[str, Any]] = []
    for pair in sorted(manifest["pairs"], key=lambda row: row["pair_id"]):
        split = pair["split"]
        if split == "sealed_test":
            continue
        renders = {row["name"]: row for row in pair["renders"]}
        render_data: dict[str, dict[str, bytes]] = {}
        for state, render in renders.items():
            image_ref, mask_ref = render["image"], render["mask"]
            if split == "train":
                image_raw = export_files[image_ref["path"]]
                mask_raw = export_files[mask_ref["path"]]
            else:
                image_raw = controlled3d_assets.read_asset(root, image_ref["path"])
                mask_raw = controlled3d_assets.read_asset(root, mask_ref["path"])
            if _digest(image_raw) != image_ref["sha256"] or _digest(mask_raw) != mask_ref["sha256"]:
                raise ValueError(
                    f"Controlled3D {split} asset changed after validation: {pair['pair_id']}"
                )
            render_data[state] = {"image": image_raw, "mask": mask_raw}
        rows = study_payload._build_rows(split, pair, render_data, study_files)
        (train_rows if split == "train" else dev_rows).extend(rows)
    train_rows.sort(key=lambda row: (row["pair_id"], row["state"]))
    dev_rows.sort(key=lambda row: (row["pair_id"], row["state"]))
    payload_raw = study_payload._archive_bytes(
        run_id,
        manifest_sha256,
        registry_sha256,
        proof_sha256,
        export_sha256,
        train_rows,
        dev_rows,
        study_files,
    )
    if len(payload_raw) > MAX_PAYLOAD_BYTES:
        raise ValueError("Controlled3D study payload exceeds the byte limit")
    # Validate exactly the bytes that the remote one-shot worker will receive.
    study_payload.validate_controlled3d_study_payload_bytes(payload_raw, run_id)
    target = Path(output_dir)
    if target.exists() or target.is_symlink() or not target.parent.is_dir():
        raise FileExistsError(
            "Controlled3D plan destination must be new and have an existing parent"
        )
    controlled3d_assets._reject_destination_path(str(target))
    payload_sha256 = _digest(payload_raw)
    marker = {
        "schema_version": 1,
        "run_id": run_id,
        "payload_file": PAYLOAD_NAME,
        "payload_bytes": len(payload_raw),
        "payload_sha256": payload_sha256,
        "source_dataset_manifest_sha256": manifest_sha256,
        "source_registry_sha256": registry_sha256,
        "validation_proof_sha256": proof_sha256,
        "train_export_completion_sha256": export_sha256,
        "training_pair_ids": sorted({row["pair_id"] for row in train_rows}),
        "development_pair_ids": sorted({row["pair_id"] for row in dev_rows}),
        "training_epochs": TRAINING_EPOCHS,
        "training_steps": TRAINING_STEPS,
        "dataset_root_identity": list(root_identity),
    }
    _publish_once(target, payload_raw, proof_raw, _json_bytes(marker))
    return load_controlled3d_study_plan(target)


def _publish_once(target: Path, payload: bytes, proof: bytes, marker: bytes) -> None:
    stage = Path(tempfile.mkdtemp(prefix=".controlled3d-study-stage-", dir=target.parent))
    claimed = False
    try:
        (stage / PAYLOAD_NAME).write_bytes(payload)
        (stage / PROOF_NAME).write_bytes(proof)
        (stage / RUN_MARKER_NAME).write_bytes(marker)
        os.mkdir(target)
        claimed = True
        for name in (PAYLOAD_NAME, PROOF_NAME, RUN_MARKER_NAME):
            shutil.copyfile(stage / name, target / name)
        if _digest(_read_regular(target / PAYLOAD_NAME, MAX_PAYLOAD_BYTES)) != _digest(payload):
            raise ValueError("Controlled3D payload changed while publishing")
    except Exception:
        if claimed:
            shutil.rmtree(target, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def load_controlled3d_study_plan(output_dir: str | Path) -> Controlled3DStudyPlan:
    """Load and recheck a previously published write-once study plan."""
    directory = Path(output_dir)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Controlled3D plan directory is missing or unsafe")
    marker_raw = _read_regular(directory / RUN_MARKER_NAME, 1024 * 1024)
    marker = controlled3d_schema._loads_any(marker_raw)
    expected_fields = {
        "schema_version",
        "run_id",
        "payload_file",
        "payload_bytes",
        "payload_sha256",
        "source_dataset_manifest_sha256",
        "source_registry_sha256",
        "validation_proof_sha256",
        "train_export_completion_sha256",
        "training_pair_ids",
        "development_pair_ids",
        "training_epochs",
        "training_steps",
        "dataset_root_identity",
    }
    if (
        type(marker) is not dict
        or set(marker) != expected_fields
        or marker["payload_file"] != PAYLOAD_NAME
    ):
        raise ValueError("Controlled3D plan marker fields are invalid")
    if (
        marker["schema_version"] != 1
        or marker["training_epochs"] != TRAINING_EPOCHS
        or marker["training_steps"] != TRAINING_STEPS
    ):
        raise ValueError("Controlled3D plan marker differs from the fixed schedule")
    for field, pin in (
        ("source_dataset_manifest_sha256", APPROVED_CONTROLLED3D_MANIFEST_SHA256),
        ("source_registry_sha256", APPROVED_CONTROLLED3D_REGISTRY_SHA256),
    ):
        if marker[field] != _require_pin(
            pin, field.removeprefix("source_").removesuffix("_sha256")
        ):
            raise ValueError("Controlled3D plan source pin differs from the reviewed source")
    payload_raw = _read_regular(directory / PAYLOAD_NAME, MAX_PAYLOAD_BYTES)
    proof_raw = _read_regular(directory / PROOF_NAME, 16 * 1024 * 1024)
    if (
        marker["payload_bytes"] != len(payload_raw)
        or marker["payload_sha256"] != _digest(payload_raw)
        or marker["validation_proof_sha256"] != _digest(proof_raw)
    ):
        raise ValueError("Controlled3D plan payload or validation proof changed")
    payload = study_payload.validate_controlled3d_study_payload_bytes(payload_raw, marker["run_id"])
    _verify_validation_proof(proof_raw, payload)
    if (
        marker["validation_proof_sha256"] != payload.validation_proof_sha256
        or marker["train_export_completion_sha256"] != payload.train_export_completion_sha256
        or marker["training_pair_ids"] != sorted({row["pair_id"] for row in payload.train_rows})
        or marker["development_pair_ids"] != sorted({row["pair_id"] for row in payload.dev_rows})
    ):
        raise ValueError("Controlled3D plan marker does not bind its payload")
    return Controlled3DStudyPlan(
        output_dir=directory,
        payload_path=directory / PAYLOAD_NAME,
        run_id=payload.run_id,
        payload_bytes=len(payload_raw),
        payload_sha256=_digest(payload_raw),
        source_dataset_manifest_sha256=payload.source_dataset_manifest_sha256,
        source_registry_sha256=payload.source_registry_sha256,
        validation_proof_sha256=payload.validation_proof_sha256,
        train_export_completion_sha256=payload.train_export_completion_sha256,
        training_pair_ids=tuple(marker["training_pair_ids"]),
        development_pair_ids=tuple(marker["development_pair_ids"]),
    )
