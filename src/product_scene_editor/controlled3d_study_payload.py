"""Strict CPU validation and assembly for Controlled3D worker payloads."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath
from typing import Any

from PIL import Image

from . import controlled3d_schema
from .matched_study import MAX_PAYLOAD_BYTES, TRAINING_EPOCHS, TRAINING_STEPS

_STATES = ("cream", "sage", "clay")
_PRODUCT_COUNTS = {"train": 6, "dev": 3}
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_ROW_FIELDS = {
    "pair_id",
    "scene_pair_id",
    "product_id",
    "split",
    "state",
    "prompt",
    "image",
    "edit_image",
    "mask",
    "bbox",
    "source_sha256",
    "target_sha256",
    "mask_sha256",
}
_STUDY_FIELDS = {
    "schema_version",
    "run_id",
    "source_dataset_manifest_sha256",
    "source_registry_sha256",
    "validation_proof_sha256",
    "train_export_completion_sha256",
    "train_rows",
    "dev_rows",
    "product_counts",
    "training_epochs",
    "training_steps",
}


@dataclass(frozen=True)
class Controlled3DStudyPayload:
    """Validated CPU study rows and image bytes unpacked from an archive."""

    run_id: str
    source_dataset_manifest_sha256: str
    source_registry_sha256: str
    validation_proof_sha256: str
    train_export_completion_sha256: str
    train_rows: tuple[dict[str, Any], ...]
    dev_rows: tuple[dict[str, Any], ...]
    files: dict[str, bytes]


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


def _open_png(raw: bytes, mode: str, label: str) -> Image.Image:
    try:
        with Image.open(BytesIO(raw)) as image:
            allowed_modes = ("RGB", "RGBA") if mode == "RGB" else (mode,)
            if image.format != "PNG" or image.mode not in allowed_modes or image.size != (512, 512):
                raise ValueError(f"{label} must be a 512x512 {mode} PNG")
            if image.mode == "RGBA":
                if image.getchannel("A").getextrema() != (255, 255):
                    raise ValueError(f"{label} RGBA alpha must be fully opaque")
            image.load()
            return image.convert("RGB") if image.mode == "RGBA" else image.copy()
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"{label} is not a valid PNG") from exc


def _prompt(state: str) -> str:
    return (
        "Change only the background to the "
        f"{state} variant; preserve the product, packaging, exact label text, shape, and position."
    )


def _build_rows(
    split: str,
    pair: dict[str, Any],
    render_data: dict[str, dict[str, bytes]],
    files: dict[str, bytes],
) -> list[dict[str, Any]]:
    pair_id, product_id = pair["pair_id"], pair["product_id"]
    source = render_data["source"]["image"]
    mask = render_data["source"]["mask"]
    source_name = f"{split}/source/{pair_id}.png"
    mask_name = f"{split}/masks/{pair_id}.png"
    files[source_name], files[mask_name] = source, mask
    source_image = _open_png(source, "RGB", f"{pair_id} source")
    mask_image = _open_png(mask, "L", f"{pair_id} mask")
    mask_pixels = mask_image.tobytes()
    if any(value not in (0, 255) for value in mask_pixels) or mask_image.getbbox() is None:
        raise ValueError(f"Controlled3D mask is not binary and non-empty: {pair_id}")
    rows: list[dict[str, Any]] = []
    for state in _STATES:
        target = render_data[state]["image"]
        target_name = f"{split}/target/{pair_id}-{state}.png"
        files[target_name] = target
        target_image = _open_png(target, "RGB", f"{pair_id} {state} target")
        if target_image.size != source_image.size:
            raise ValueError(f"Controlled3D target dimensions differ: {pair_id}")
        row = {
            "pair_id": f"{pair_id}-{state}",
            "scene_pair_id": pair_id,
            "product_id": product_id,
            "split": split,
            "state": state,
            "prompt": _prompt(state),
            "image": target_name,
            "edit_image": source_name,
            "mask": mask_name,
            "bbox": list(
                next(render["bbox"] for render in pair["renders"] if render["name"] == "source")
            ),
            "source_sha256": _digest(source),
            "target_sha256": _digest(target),
            "mask_sha256": _digest(mask),
        }
        rows.append(row)
    return rows


def _archive_bytes(
    run_id: str,
    manifest_sha256: str,
    registry_sha256: str,
    proof_sha256: str,
    export_sha256: str,
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    files: dict[str, bytes],
) -> bytes:
    study = {
        "schema_version": 1,
        "run_id": run_id,
        "source_dataset_manifest_sha256": manifest_sha256,
        "source_registry_sha256": registry_sha256,
        "validation_proof_sha256": proof_sha256,
        "train_export_completion_sha256": export_sha256,
        "train_rows": train_rows,
        "dev_rows": dev_rows,
        "product_counts": _PRODUCT_COUNTS,
        "training_epochs": TRAINING_EPOCHS,
        "training_steps": TRAINING_STEPS,
    }
    archive_files = {
        **files,
        "metadata.json": _json_bytes(
            [{key: row[key] for key in ("image", "edit_image", "prompt")} for row in train_rows]
        ),
        "controlled3d_manifest.json": _json_bytes(study),
    }
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, raw in sorted(archive_files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, raw)
    return buffer.getvalue()


def _image_paths(row: dict[str, Any], files: dict[str, bytes], seen: set[str]) -> None:
    pair_id = row["scene_pair_id"]
    expected = {
        "edit_image": f"{row['split']}/source/{pair_id}.png",
        "mask": f"{row['split']}/masks/{pair_id}.png",
        "image": f"{row['split']}/target/{pair_id}-{row['state']}.png",
    }
    if any(row[key] != path for key, path in expected.items()):
        raise ValueError("Controlled3D row contains an unsafe or unexpected image path")
    for path in expected.values():
        if path not in files:
            raise ValueError(f"Controlled3D payload asset is missing: {path}")
        seen.add(path)
    for field, path_key, mode in (
        ("source_sha256", "edit_image", "RGB"),
        ("target_sha256", "image", "RGB"),
        ("mask_sha256", "mask", "L"),
    ):
        path = row[path_key]
        if row[field] != _digest(files[path]):
            raise ValueError(f"Controlled3D payload checksum mismatch: {path}")
        image = _open_png(files[path], mode, path)
        if mode == "L":
            if any(value not in (0, 255) for value in image.tobytes()):
                raise ValueError("Controlled3D mask must be binary")
            bbox = image.getbbox()
            if bbox is None or list(bbox) != row["bbox"]:
                raise ValueError("Controlled3D mask bbox differs from row evidence")


def _validate_payload_files(
    study: object,
    metadata: object,
    files: dict[str, bytes],
    run_id: str,
    manifest_pin: str,
    registry_pin: str,
) -> Controlled3DStudyPayload:
    if type(study) is not dict or set(study) != _STUDY_FIELDS:
        raise ValueError("Controlled3D study manifest fields are missing or unknown")
    if (
        type(study["schema_version"]) is not int
        or study["schema_version"] != 1
        or study["run_id"] != run_id
        or study["source_dataset_manifest_sha256"] != manifest_pin
        or study["source_registry_sha256"] != registry_pin
        or study["product_counts"] != _PRODUCT_COUNTS
        or study["training_epochs"] != TRAINING_EPOCHS
        or study["training_steps"] != TRAINING_STEPS
        or any(
            not isinstance(study[key], str) or not _DIGEST.fullmatch(study[key])
            for key in ("validation_proof_sha256", "train_export_completion_sha256")
        )
    ):
        raise ValueError("Controlled3D study manifest differs from the reviewed profile")
    train_rows, dev_rows = study["train_rows"], study["dev_rows"]
    if type(train_rows) is not list or len(train_rows) != 18:
        raise ValueError("Controlled3D profile requires exactly 18 training rows")
    if type(dev_rows) is not list or len(dev_rows) != 9:
        raise ValueError("Controlled3D profile requires exactly nine development rows")
    if type(metadata) is not list or len(metadata) != 18:
        raise ValueError("DiffSynth metadata must contain exactly 18 training rows")
    if train_rows != sorted(
        train_rows, key=lambda row: (row.get("pair_id", ""), row.get("state", ""))
    ) or dev_rows != sorted(
        dev_rows, key=lambda row: (row.get("pair_id", ""), row.get("state", ""))
    ):
        raise ValueError("Controlled3D rows must be in stable pair/state order")
    expected_files = {"metadata.json", "controlled3d_manifest.json"}
    product_ids: dict[str, set[str]] = {"train": set(), "dev": set()}
    rows_by_pair: dict[str, set[str]] = {}
    product_by_pair: dict[str, str] = {}
    seen_paths: set[str] = set()
    row_ids: set[tuple[str, str]] = set()
    for split, rows in (("train", train_rows), ("dev", dev_rows)):
        for row in rows:
            if type(row) is not dict or set(row) != _ROW_FIELDS or row["split"] != split:
                raise ValueError("Controlled3D row fields or split are invalid")
            pair_id = controlled3d_schema._id(row["pair_id"], "pair ID")
            scene_pair_id = controlled3d_schema._id(row["scene_pair_id"], "scene pair ID")
            product_id = controlled3d_schema._id(row["product_id"], "product ID")
            state = row["state"]
            if (
                type(state) is not str
                or state not in _STATES
                or pair_id != f"{scene_pair_id}-{state}"
                or (split, pair_id) in row_ids
            ):
                raise ValueError("Controlled3D pair/state row is invalid or duplicated")
            row_ids.add((split, pair_id))
            if row["prompt"] != _prompt(state):
                raise ValueError("Controlled3D prompt differs from the reviewed edit instruction")
            bbox = row["bbox"]
            if (
                type(bbox) is not list
                or len(bbox) != 4
                or any(type(value) is not int or value < 0 or value > 512 for value in bbox)
            ):
                raise ValueError("Controlled3D bbox is invalid")
            for digest_field in ("source_sha256", "target_sha256", "mask_sha256"):
                if type(row[digest_field]) is not str or not _DIGEST.fullmatch(row[digest_field]):
                    raise ValueError("Controlled3D row checksum is invalid")
            _image_paths(row, files, seen_paths)
            key = split + ":" + scene_pair_id
            rows_by_pair.setdefault(key, set()).add(state)
            if product_by_pair.setdefault(key, product_id) != product_id:
                raise ValueError("Controlled3D scene pair changes product identity")
            product_ids[split].add(product_id)
    if product_ids["train"] & product_ids["dev"]:
        raise ValueError("Controlled3D product identity crosses train/dev split")
    if len(product_ids["train"]) != 6 or len(product_ids["dev"]) != 3:
        raise ValueError("Controlled3D rows must cover six train and three dev products")
    if any(states != set(_STATES) for states in rows_by_pair.values()):
        raise ValueError("each Controlled3D product pair needs cream, sage, and clay rows")
    for split, count in (("train", 6), ("dev", 3)):
        pairs = [pair for pair in rows_by_pair if pair.startswith(split + ":")]
        if len(pairs) != count:
            raise ValueError("Controlled3D profile requires one reviewed pair per product")
    for row in train_rows:
        expected_files.update(
            {
                row["image"],
                row["edit_image"],
                row["mask"],
            }
        )
    for row in dev_rows:
        expected_files.update({row["image"], row["edit_image"], row["mask"]})
    if set(files) != expected_files:
        raise ValueError("Controlled3D archive contains missing or unexpected files")
    expected_metadata = [
        {key: row[key] for key in ("image", "edit_image", "prompt")} for row in train_rows
    ]
    if metadata != expected_metadata:
        raise ValueError("DiffSynth metadata does not match training rows")
    return Controlled3DStudyPayload(
        run_id=run_id,
        source_dataset_manifest_sha256=study["source_dataset_manifest_sha256"],
        source_registry_sha256=study["source_registry_sha256"],
        validation_proof_sha256=study["validation_proof_sha256"],
        train_export_completion_sha256=study["train_export_completion_sha256"],
        train_rows=tuple(train_rows),
        dev_rows=tuple(dev_rows),
        files=files,
    )


def validate_controlled3d_study_payload_bytes(
    payload_bytes: bytes, run_id: str
) -> Controlled3DStudyPayload:
    """Validate archive bytes against both unset-or-reviewed digest pins and row contracts."""
    from . import controlled3d_study as planner

    manifest_pin = _require_pin(planner.APPROVED_CONTROLLED3D_MANIFEST_SHA256, "manifest")
    registry_pin = _require_pin(planner.APPROVED_CONTROLLED3D_REGISTRY_SHA256, "registry")
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id) or ".." in run_id:
        raise ValueError("invalid Controlled3D run identifier")
    if (
        not isinstance(payload_bytes, bytes)
        or not payload_bytes
        or len(payload_bytes) > MAX_PAYLOAD_BYTES
    ):
        raise ValueError("Controlled3D payload is empty or exceeds the byte limit")
    try:
        with zipfile.ZipFile(BytesIO(payload_bytes)) as archive:
            infos = archive.infolist()
            names = [item.filename for item in infos]
            if (
                len(names) != len(set(names))
                or sum(item.file_size for item in infos) > 64 * 1024 * 1024
            ):
                raise ValueError("Controlled3D archive has duplicate paths or expands too large")
            for info in infos:
                path = PurePosixPath(info.filename)
                if (
                    "\\" in info.filename
                    or path.is_absolute()
                    or ".." in path.parts
                    or "sealed_test" in path.parts
                    or info.filename not in {"metadata.json", "controlled3d_manifest.json"}
                    and (
                        len(path.parts) != 3
                        or path.parts[0] not in {"train", "dev"}
                        or path.suffix != ".png"
                    )
                ):
                    raise ValueError("Controlled3D archive contains an unsafe or unexpected path")
            files = {item.filename: archive.read(item) for item in infos}
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise ValueError("Controlled3D payload is not a valid archive") from exc
    if files.get("metadata.json") is None or files.get("controlled3d_manifest.json") is None:
        raise ValueError("Controlled3D payload metadata is missing")
    study = controlled3d_schema._loads_any(files["controlled3d_manifest.json"])
    metadata = controlled3d_schema._loads_any(files["metadata.json"])
    if files["controlled3d_manifest.json"] != _json_bytes(study) or files[
        "metadata.json"
    ] != _json_bytes(metadata):
        raise ValueError("Controlled3D payload JSON must be canonical")
    return _validate_payload_files(study, metadata, files, run_id, manifest_pin, registry_pin)
