"""CPU-side planning and validation for the bounded eight-pair edit pilot."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

from .dataset import verify_dataset

RUN_MARKER_NAME = ".pilot-run.json"
PAYLOAD_NAME = "pilot-input.zip"
MAX_PAYLOAD_BYTES = 16 * 1024 * 1024
TRAINING_PAIR_COUNT = 8
DEVELOPMENT_PAIR_COUNT = 1
APPROVED_DATASET_MANIFEST_SHA256 = (
    "206b07f99b2fd1c001b94898f3cacfa7010bacb7e08aa731b972135000e9a8f9"
)
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class PilotPlan:
    """A write-once local input bundle and its immutable run marker."""

    output_dir: Path
    payload_path: Path
    run_id: str
    payload_bytes: int
    payload_sha256: str
    source_dataset_manifest_sha256: str
    training_pair_ids: tuple[str, ...]
    development_pair_id: str


@dataclass(frozen=True)
class PilotPayload:
    """Validated rows and image bytes unpacked from a pilot archive."""

    run_id: str
    source_dataset_manifest_sha256: str
    train_rows: tuple[dict[str, Any], ...]
    dev_rows: tuple[dict[str, Any], ...]
    files: dict[str, bytes]


def _validate_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id) or ".." in run_id:
        raise ValueError("run_id must be a short path-safe identifier")


def _stable_selection(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for pair in pairs:
        grouped.setdefault(pair["product_id"], []).append(pair)
    for product_pairs in grouped.values():
        product_pairs.sort(key=lambda pair: pair["pair_id"])

    selected: list[dict[str, Any]] = []
    product_ids = sorted(grouped)
    while len(selected) < TRAINING_PAIR_COUNT:
        advanced = False
        for product_id in product_ids:
            if grouped[product_id]:
                selected.append(grouped[product_id].pop(0))
                advanced = True
                if len(selected) == TRAINING_PAIR_COUNT:
                    break
        if not advanced:
            break
    if len(selected) != TRAINING_PAIR_COUNT:
        raise ValueError("dataset must contain at least eight train pairs")
    return selected


def _archive_bytes(
    dataset_root: Path,
    run_id: str,
    source_dataset_manifest_sha256: str,
    train_pairs: list[dict[str, Any]],
    dev_pair: dict[str, Any],
) -> bytes:
    files: dict[str, bytes] = {}
    train_rows: list[dict[str, str]] = []
    manifest_train: list[dict[str, Any]] = []
    for pair in train_pairs:
        pair_id = pair["pair_id"]
        source_name = f"train/source/{pair_id}.png"
        target_name = f"train/target/{pair_id}.png"
        files[source_name] = (dataset_root / pair["source_path"]).read_bytes()
        files[target_name] = (dataset_root / pair["target_path"]).read_bytes()
        row = {"image": target_name, "edit_image": source_name, "prompt": pair["prompt"]}
        train_rows.append(row)
        manifest_train.append(
            {
                "pair_id": pair_id,
                "product_id": pair["product_id"],
                "split": "train",
                **row,
                "source_sha256": hashlib.sha256(files[source_name]).hexdigest(),
                "target_sha256": hashlib.sha256(files[target_name]).hexdigest(),
            }
        )

    dev_source = f"dev/source/{dev_pair['pair_id']}.png"
    dev_target = f"dev/target/{dev_pair['pair_id']}.png"
    files[dev_source] = (dataset_root / dev_pair["source_path"]).read_bytes()
    files[dev_target] = (dataset_root / dev_pair["target_path"]).read_bytes()
    dev_row: dict[str, Any] = {
        "pair_id": dev_pair["pair_id"],
        "product_id": dev_pair["product_id"],
        "split": "dev",
        "image": dev_target,
        "edit_image": dev_source,
        "prompt": dev_pair["prompt"],
        "source_sha256": hashlib.sha256(files[dev_source]).hexdigest(),
        "target_sha256": hashlib.sha256(files[dev_target]).hexdigest(),
    }
    files["metadata.json"] = (
        json.dumps(train_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    files["pilot_manifest.json"] = (
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "source_dataset_manifest_sha256": source_dataset_manifest_sha256,
                "training_pairs": manifest_train,
                "development_pair": dev_row,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")

    buffer = BytesIO()
    with zipfile.ZipFile(
        buffer, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, data)
    return buffer.getvalue()


def plan_pilot(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    run_id: str,
    max_payload_bytes: int = MAX_PAYLOAD_BYTES,
) -> PilotPlan:
    """Create a fresh, bounded archive containing eight train and one dev pair."""
    _validate_run_id(run_id)
    if (
        isinstance(max_payload_bytes, bool)
        or not isinstance(max_payload_bytes, int)
        or max_payload_bytes <= 0
    ):
        raise ValueError("max_payload_bytes must be a positive integer")

    dataset_root = Path(dataset_dir)
    manifest = verify_dataset(dataset_root)
    source_dataset_manifest_sha256 = hashlib.sha256(
        (dataset_root / "manifest.json").read_bytes()
    ).hexdigest()
    train_pairs = _stable_selection(
        [pair for pair in manifest["pairs"] if pair["split"] == "train"]
    )
    dev_pairs = sorted(
        (pair for pair in manifest["pairs"] if pair["split"] == "dev"),
        key=lambda pair: (pair["product_id"], pair["pair_id"]),
    )
    if not dev_pairs:
        raise ValueError("dataset must contain at least one dev pair")
    dev_pair = dev_pairs[0]
    if any(pair["product_id"] == dev_pair["product_id"] for pair in train_pairs):
        raise ValueError("training and development product identities overlap")

    archive_bytes = _archive_bytes(
        dataset_root, run_id, source_dataset_manifest_sha256, train_pairs, dev_pair
    )
    if len(archive_bytes) > max_payload_bytes:
        raise ValueError("pilot payload exceeds the configured byte limit")

    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    try:
        payload_path = output / PAYLOAD_NAME
        payload_path.write_bytes(archive_bytes)
        digest = hashlib.sha256(archive_bytes).hexdigest()
        marker = {
            "schema_version": 1,
            "run_id": run_id,
            "payload_file": PAYLOAD_NAME,
            "payload_bytes": len(archive_bytes),
            "payload_sha256": digest,
            "source_dataset_manifest_sha256": source_dataset_manifest_sha256,
            "training_pair_ids": [pair["pair_id"] for pair in train_pairs],
            "development_pair_id": dev_pair["pair_id"],
        }
        (output / RUN_MARKER_NAME).write_text(
            json.dumps(marker, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise

    return PilotPlan(
        output_dir=output,
        payload_path=payload_path,
        run_id=run_id,
        payload_bytes=len(archive_bytes),
        payload_sha256=digest,
        source_dataset_manifest_sha256=source_dataset_manifest_sha256,
        training_pair_ids=tuple(pair["pair_id"] for pair in train_pairs),
        development_pair_id=dev_pair["pair_id"],
    )


def read_pilot_payload(plan: PilotPlan) -> PilotPayload:
    """Revalidate a planned archive immediately before local inspection or upload."""
    marker_path = plan.output_dir / RUN_MARKER_NAME
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        payload_bytes = plan.payload_path.read_bytes()
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("pilot run marker or payload is missing or invalid") from exc
    actual_digest = hashlib.sha256(payload_bytes).hexdigest()
    if (
        marker.get("run_id") != plan.run_id
        or marker.get("payload_file") != PAYLOAD_NAME
        or marker.get("payload_bytes") != len(payload_bytes)
        or plan.payload_bytes != len(payload_bytes)
        or marker.get("payload_sha256") != actual_digest
        or plan.payload_sha256 != actual_digest
        or marker.get("source_dataset_manifest_sha256") != plan.source_dataset_manifest_sha256
    ):
        raise ValueError("pilot payload checksum or run marker mismatch")
    if len(payload_bytes) > MAX_PAYLOAD_BYTES:
        raise ValueError("pilot payload exceeds the configured byte limit")

    try:
        with zipfile.ZipFile(BytesIO(payload_bytes)) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("pilot archive contains duplicate paths")
            files: dict[str, bytes] = {}
            for name in names:
                path = PurePosixPath(name)
                if path.is_absolute() or ".." in path.parts or "sealed_test" in path.parts:
                    raise ValueError("pilot archive contains an unsafe or sealed-test path")
                if not (
                    name == "metadata.json"
                    or name == "pilot_manifest.json"
                    or path.parts[0] in {"train", "dev"}
                ):
                    raise ValueError("pilot archive contains an unexpected path")
                files[name] = archive.read(name)
            manifest = json.loads(files["pilot_manifest.json"])
            train_rows = manifest["training_pairs"]
            dev_row = manifest["development_pair"]
            metadata = json.loads(files["metadata.json"])
    except (KeyError, OSError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise ValueError("pilot payload is not a valid archive") from exc

    if (
        manifest.get("schema_version") != 1
        or manifest.get("run_id") != plan.run_id
        or manifest.get("source_dataset_manifest_sha256") != plan.source_dataset_manifest_sha256
        or not isinstance(plan.source_dataset_manifest_sha256, str)
        or not _DIGEST_PATTERN.fullmatch(plan.source_dataset_manifest_sha256)
        or not isinstance(train_rows, list)
        or len(train_rows) != TRAINING_PAIR_COUNT
        or not isinstance(dev_row, dict)
        or dev_row.get("split") != "dev"
        or not isinstance(metadata, list)
        or len(metadata) != TRAINING_PAIR_COUNT
        or marker.get("training_pair_ids") != [row.get("pair_id") for row in train_rows]
        or marker.get("development_pair_id") != dev_row.get("pair_id")
    ):
        raise ValueError("pilot archive does not match the eight-train/one-dev contract")
    expected_names = {"metadata.json", "pilot_manifest.json"}
    train_product_ids: set[str] = set()
    for row in (*train_rows, dev_row):
        if not isinstance(row, dict) or row.get("split") not in {"train", "dev"}:
            raise ValueError("pilot archive contains a row outside the allowed splits")
        for key in ("image", "edit_image"):
            filename = row.get(key)
            if not isinstance(filename, str) or filename not in files:
                raise ValueError("pilot archive row references a missing image")
            if PurePosixPath(filename).parts[0] != row["split"]:
                raise ValueError("pilot image path does not match its split")
            expected_names.add(filename)
        source_name, target_name = row["edit_image"], row["image"]
        if (
            row.get("source_sha256") != hashlib.sha256(files[source_name]).hexdigest()
            or row.get("target_sha256") != hashlib.sha256(files[target_name]).hexdigest()
        ):
            raise ValueError("pilot image checksum does not match its row")
        if row["split"] == "train":
            product_id = row.get("product_id")
            if not isinstance(product_id, str) or not product_id:
                raise ValueError("pilot row is missing its product identity")
            train_product_ids.add(product_id)
    dev_product_id = dev_row.get("product_id")
    if not isinstance(dev_product_id, str) or not dev_product_id:
        raise ValueError("development row is missing its product identity")
    if dev_product_id in train_product_ids or set(files) != expected_names:
        raise ValueError("pilot archive has product leakage or unexpected files")
    if any(row.get("split") != "train" for row in train_rows):
        raise ValueError("pilot training archive contains a non-train row")
    if metadata != [
        {key: row[key] for key in ("image", "edit_image", "prompt")} for row in train_rows
    ]:
        raise ValueError("DiffSynth metadata rows do not match the pilot manifest")

    return PilotPayload(
        run_id=plan.run_id,
        source_dataset_manifest_sha256=plan.source_dataset_manifest_sha256,
        train_rows=tuple(train_rows),
        dev_rows=(dev_row,),
        files=files,
    )


def load_pilot_plan(output_dir: str | Path) -> PilotPlan:
    """Load a previously planned run without permitting a new payload or overwrite."""
    output = Path(output_dir)
    try:
        marker = json.loads((output / RUN_MARKER_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("pilot run marker is missing or invalid") from exc
    if not isinstance(marker, dict) or marker.get("schema_version") != 1:
        raise ValueError("pilot run marker schema is invalid")
    run_id = marker.get("run_id")
    if not isinstance(run_id, str):
        raise ValueError("pilot run marker is missing its run_id")
    _validate_run_id(run_id)
    payload_bytes = marker.get("payload_bytes")
    payload_sha256 = marker.get("payload_sha256")
    dataset_sha256 = marker.get("source_dataset_manifest_sha256")
    pair_ids = marker.get("training_pair_ids")
    dev_pair_id = marker.get("development_pair_id")
    if (
        isinstance(payload_bytes, bool)
        or not isinstance(payload_bytes, int)
        or payload_bytes <= 0
        or not isinstance(payload_sha256, str)
        or not _DIGEST_PATTERN.fullmatch(payload_sha256)
        or not isinstance(dataset_sha256, str)
        or not _DIGEST_PATTERN.fullmatch(dataset_sha256)
        or not isinstance(pair_ids, list)
        or any(not isinstance(pair_id, str) for pair_id in pair_ids)
        or not isinstance(dev_pair_id, str)
    ):
        raise ValueError("pilot run marker fields are invalid")
    payload_path = output / PAYLOAD_NAME
    plan = PilotPlan(
        output_dir=output,
        payload_path=payload_path,
        run_id=run_id,
        payload_bytes=payload_bytes,
        payload_sha256=payload_sha256,
        source_dataset_manifest_sha256=dataset_sha256,
        training_pair_ids=tuple(pair_ids),
        development_pair_id=dev_pair_id,
    )
    payload = read_pilot_payload(plan)
    if (
        len(payload.train_rows) != TRAINING_PAIR_COUNT
        or len(payload.dev_rows) != DEVELOPMENT_PAIR_COUNT
    ):
        raise ValueError("saved pilot plan does not match the fixed pair count")
    return plan
