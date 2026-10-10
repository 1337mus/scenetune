"""CPU-only planning, payload validation, and scoring for the matched study."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, ImageChops, ImageStat

RUN_MARKER_NAME = ".matched-study-run.json"
PAYLOAD_NAME = "matched-study-input.zip"
MAX_PAYLOAD_BYTES = 16 * 1024 * 1024
TRAINING_PAIR_COUNT = 18
DEVELOPMENT_PAIR_COUNT = 9
TRAINING_EPOCHS = 8
TRAINING_STEPS = 144
APPROVED_DATASET_MANIFEST_SHA256 = (
    "206b07f99b2fd1c001b94898f3cacfa7010bacb7e08aa731b972135000e9a8f9"
)

_IMAGE_SIZE = (512, 512)
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_SCHEMA_FIELDS = {"schema_version", "provenance", "pairs"}
_PROVENANCE_FIELDS = {"image_source", "external_assets", "employer_input", "license"}
_PROVENANCE = {
    "image_source": "original_synthetic",
    "external_assets": [],
    "employer_input": "none",
    "license": "CC0-1.0",
}
_SOURCE_PAIR_FIELDS = {
    "product_id",
    "pair_id",
    "split",
    "prompt",
    "source_path",
    "target_path",
    "mask_path",
    "bbox",
    "generator_seed",
    "data_license",
    "sha256",
}
_ROW_FIELDS = {
    "pair_id",
    "product_id",
    "split",
    "prompt",
    "image",
    "edit_image",
    "mask",
    "bbox",
    "source_sha256",
    "target_sha256",
    "mask_sha256",
}
_MANIFEST_FIELDS = {
    "schema_version",
    "run_id",
    "source_dataset_manifest_sha256",
    "train_rows",
    "dev_rows",
    "product_counts",
    "training_epochs",
    "training_steps",
}
_MARKER_FIELDS = {
    "schema_version",
    "run_id",
    "payload_file",
    "payload_bytes",
    "payload_sha256",
    "source_dataset_manifest_sha256",
    "training_pair_ids",
    "development_pair_ids",
    "training_epochs",
    "training_steps",
}
_PRODUCT_COUNTS = {"train": 6, "dev": 3, "sealed_test": 3}
_SPLIT_PAIR_COUNTS = {"train": 18, "dev": 9, "sealed_test": 9}
_ASSET_FIELDS = {
    "source": "source_path",
    "target": "target_path",
    "mask": "mask_path",
}


@dataclass(frozen=True)
class MatchedStudyPlan:
    """A write-once local input bundle and its immutable marker."""

    output_dir: Path
    payload_path: Path
    run_id: str
    payload_bytes: int
    payload_sha256: str
    source_dataset_manifest_sha256: str
    training_pair_ids: tuple[str, ...]
    development_pair_ids: tuple[str, ...]


@dataclass(frozen=True)
class MatchedStudyPayload:
    """Validated manifest rows and image bytes unpacked from a matched archive."""

    run_id: str
    source_dataset_manifest_sha256: str
    train_rows: tuple[dict[str, Any], ...]
    dev_rows: tuple[dict[str, Any], ...]
    files: dict[str, bytes]


def _unique_keys(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _decode_json(data: bytes, label: str) -> Any:
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_keys,
            parse_constant=_invalid_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc


def _validate_identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value) or ".." in value:
        raise ValueError(f"{name} must be a path-safe identifier")
    return value


def _validate_run_id(run_id: object) -> str:
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id) or ".." in run_id:
        raise ValueError("run_id must be a short path-safe identifier")
    return run_id


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_bbox(value: object, pair_id: str) -> list[int]:
    if not isinstance(value, list) or len(value) != 4 or any(not _is_int(item) for item in value):
        raise ValueError(f"pair {pair_id!r} bbox must contain four integers")
    return value


def _validate_source_manifest_metadata(manifest: object) -> list[dict[str, Any]]:
    if not isinstance(manifest, dict) or set(manifest) != _SOURCE_SCHEMA_FIELDS:
        raise ValueError("manifest fields are missing or unknown")
    if not _is_int(manifest["schema_version"]) or manifest["schema_version"] != 1:
        raise ValueError("unsupported manifest schema_version")
    provenance = manifest["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != _PROVENANCE_FIELDS:
        raise ValueError("provenance fields are missing or unknown")
    if provenance != _PROVENANCE:
        raise ValueError("manifest provenance must describe original synthetic CC0 data")
    pairs = manifest["pairs"]
    if not isinstance(pairs, list) or len(pairs) != sum(_SPLIT_PAIR_COUNTS.values()):
        raise ValueError("approved manifest must contain exactly 36 pair rows")

    pair_ids: set[str] = set()
    product_splits: dict[str, str] = {}
    pair_counts: Counter[str] = Counter()
    products_by_split: dict[str, set[str]] = {split: set() for split in _PRODUCT_COUNTS}
    checked: list[dict[str, Any]] = []
    for pair in pairs:
        if not isinstance(pair, dict) or set(pair) != _SOURCE_PAIR_FIELDS:
            raise ValueError("pair fields are missing or unknown")
        product_id = _validate_identifier(pair["product_id"], "product_id")
        pair_id = _validate_identifier(pair["pair_id"], "pair_id")
        split = pair["split"]
        if not isinstance(split, str) or split not in _SPLIT_PAIR_COUNTS:
            raise ValueError(f"unknown split: {split!r}")
        if pair_id in pair_ids:
            raise ValueError(f"duplicate pair_id: {pair_id!r}")
        pair_ids.add(pair_id)
        previous_split = product_splits.setdefault(product_id, split)
        if previous_split != split:
            raise ValueError(f"product identity {product_id!r} crosses splits")
        products_by_split[split].add(product_id)
        pair_counts[split] += 1
        prompt = pair["prompt"]
        if not isinstance(prompt, str) or "keep the package unchanged" not in prompt.lower():
            raise ValueError(f"pair {pair_id!r} prompt must preserve the package")
        _validate_bbox(pair["bbox"], pair_id)
        seed = pair["generator_seed"]
        if not _is_int(seed):
            raise ValueError(f"pair {pair_id!r} generator_seed must be an integer")
        if pair["data_license"] != "CC0-1.0":
            raise ValueError(f"pair {pair_id!r} data license is not CC0-1.0")
        digests = pair["sha256"]
        if not isinstance(digests, dict) or set(digests) != set(_ASSET_FIELDS):
            raise ValueError(f"pair {pair_id!r} sha256 fields are invalid")

        # Sealed rows are metadata-only. In particular, do not validate their asset paths or
        # checksum values by resolving, hashing, copying, or opening those assets.
        for field in _ASSET_FIELDS.values():
            if not isinstance(pair[field], str):
                raise ValueError(f"pair {pair_id!r} {field} must be a string")
        checked.append(pair)

    if dict(pair_counts) != _SPLIT_PAIR_COUNTS:
        raise ValueError("approved manifest pair counts do not match the frozen split counts")
    if {split: len(products) for split, products in products_by_split.items()} != _PRODUCT_COUNTS:
        raise ValueError("approved manifest product counts do not match the frozen split counts")
    if any(
        count != 3
        for split in _PRODUCT_COUNTS
        for count in Counter(
            pair["product_id"] for pair in checked if pair["split"] == split
        ).values()
    ):
        raise ValueError("each product identity must have exactly three scene pairs")
    return checked


def _resolve_asset(root: Path, relative: str, split: str, role: str) -> Path:
    path = PurePosixPath(relative)
    folder = "masks" if role == "mask" else role
    if (
        "\\" in relative
        or path.is_absolute()
        or ".." in path.parts
        or path.parts[:2] != (split, folder)
        or len(path.parts) != 3
        or path.suffix.lower() != ".png"
        or path.as_posix() != relative
    ):
        raise ValueError(f"unsafe asset path: {relative!r}")
    resolved = (root / Path(*path.parts)).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"unsafe asset path: {relative!r}") from exc
    return resolved


def _read_checked_asset(
    dataset_root: Path, pair: dict[str, Any], role: str
) -> tuple[str, bytes, Image.Image]:
    relative = pair[_ASSET_FIELDS[role]]
    if PurePosixPath(relative).name != f"{pair['pair_id']}.png":
        raise ValueError(f"unsafe asset path: {relative!r}")
    path = _resolve_asset(dataset_root, relative, pair["split"], role)
    digest = pair["sha256"][role]
    if not isinstance(digest, str) or not _DIGEST_PATTERN.fullmatch(digest):
        raise ValueError(f"pair {pair['pair_id']!r} {role} checksum is invalid")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"asset is missing for pair {pair['pair_id']!r} {role}") from exc
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"checksum mismatch for pair {pair['pair_id']!r} {role}")
    try:
        with Image.open(BytesIO(data)) as opened:
            if opened.format != "PNG":
                raise ValueError(f"pair {pair['pair_id']!r} {role} asset must be PNG")
            expected_mode = "L" if role == "mask" else "RGB"
            if opened.mode != expected_mode or opened.size != _IMAGE_SIZE:
                raise ValueError(
                    f"pair {pair['pair_id']!r} {role} must be a 512x512 {expected_mode} PNG"
                )
            opened.load()
            image = opened.copy()
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"pair {pair['pair_id']!r} {role} PNG is invalid") from exc
    return relative, data, image


def _verify_pair_images(pair: dict[str, Any], assets: dict[str, Image.Image]) -> str:
    source, target, mask = assets["source"], assets["target"], assets["mask"]
    if any(value not in (0, 255) for value in mask.tobytes()):
        raise ValueError(f"pair {pair['pair_id']!r} mask must be binary")
    if tuple(pair["bbox"]) != mask.getbbox():
        raise ValueError(f"pair {pair['pair_id']!r} mask bbox does not match manifest")
    histogram = mask.histogram()
    if not histogram[0] or not histogram[255]:
        raise ValueError(f"pair {pair['pair_id']!r} mask must contain foreground and background")
    diff = ImageChops.difference(source, target)
    black = Image.new("RGB", _IMAGE_SIZE)
    if ImageChops.composite(diff, black, mask).getbbox():
        raise ValueError(f"pair {pair['pair_id']!r} product foreground differs")
    if not ImageChops.composite(diff, black, ImageChops.invert(mask)).getbbox():
        raise ValueError(f"pair {pair['pair_id']!r} has no background scene change")
    foreground = ImageChops.composite(source, black, mask)
    return hashlib.sha256(mask.tobytes() + foreground.tobytes()).hexdigest()


def _matched_row(pair: dict[str, Any], asset_data: dict[str, bytes]) -> dict[str, Any]:
    pair_id, split = pair["pair_id"], pair["split"]
    source_name = f"{split}/source/{pair_id}.png"
    target_name = f"{split}/target/{pair_id}.png"
    mask_name = f"{split}/mask/{pair_id}.png"
    return {
        "pair_id": pair_id,
        "product_id": pair["product_id"],
        "split": split,
        "prompt": pair["prompt"],
        "image": target_name,
        "edit_image": source_name,
        "mask": mask_name,
        "bbox": list(pair["bbox"]),
        "source_sha256": hashlib.sha256(asset_data["source"]).hexdigest(),
        "target_sha256": hashlib.sha256(asset_data["target"]).hexdigest(),
        "mask_sha256": hashlib.sha256(asset_data["mask"]).hexdigest(),
    }


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _archive_bytes(
    run_id: str,
    source_dataset_manifest_sha256: str,
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    files: dict[str, bytes],
) -> bytes:
    metadata = [{key: row[key] for key in ("image", "edit_image", "prompt")} for row in train_rows]
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "source_dataset_manifest_sha256": source_dataset_manifest_sha256,
        "train_rows": train_rows,
        "dev_rows": dev_rows,
        "product_counts": _PRODUCT_COUNTS,
        "training_epochs": TRAINING_EPOCHS,
        "training_steps": TRAINING_STEPS,
    }
    archive_files = {
        **files,
        "metadata.json": _json_bytes(metadata),
        "matched_manifest.json": _json_bytes(manifest),
    }
    buffer = BytesIO()
    with zipfile.ZipFile(
        buffer, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name, data in sorted(archive_files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, data)
    return buffer.getvalue()


def _validate_payload_rows(
    manifest: object,
    metadata: object,
    files: dict[str, bytes],
    run_id: str,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], str]:
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("matched manifest fields are missing or unknown")
    source_digest = manifest["source_dataset_manifest_sha256"]
    if (
        not _is_int(manifest["schema_version"])
        or manifest["schema_version"] != 1
        or manifest["run_id"] != run_id
        or source_digest != APPROVED_DATASET_MANIFEST_SHA256
        or manifest["product_counts"] != _PRODUCT_COUNTS
        or manifest["training_epochs"] != TRAINING_EPOCHS
        or manifest["training_steps"] != TRAINING_STEPS
    ):
        raise ValueError("matched manifest does not match the frozen study contract")
    if (
        not isinstance(manifest["train_rows"], list)
        or len(manifest["train_rows"]) != TRAINING_PAIR_COUNT
    ):
        raise ValueError("matched manifest must contain exactly 18 train rows")
    if (
        not isinstance(manifest["dev_rows"], list)
        or len(manifest["dev_rows"]) != DEVELOPMENT_PAIR_COUNT
    ):
        raise ValueError("matched manifest must contain exactly nine dev rows")
    if not isinstance(metadata, list) or len(metadata) != TRAINING_PAIR_COUNT:
        raise ValueError("DiffSynth metadata must contain exactly 18 train rows")

    train_rows = manifest["train_rows"]
    dev_rows = manifest["dev_rows"]
    if any(not isinstance(row, dict) for row in (*train_rows, *dev_rows)):
        raise ValueError("matched pair rows must be objects")
    if train_rows != sorted(train_rows, key=lambda row: row.get("pair_id", "")):
        raise ValueError("matched train rows must be in stable pair_id order")
    if dev_rows != sorted(dev_rows, key=lambda row: row.get("pair_id", "")):
        raise ValueError("matched dev rows must be in stable pair_id order")

    expected_names = {"metadata.json", "matched_manifest.json"}
    product_ids: dict[str, set[str]] = {"train": set(), "dev": set()}
    pair_ids: set[str] = set()
    fingerprints_by_split: dict[str, set[str]] = {"train": set(), "dev": set()}
    for split, rows in (("train", train_rows), ("dev", dev_rows)):
        rows_product_counts: Counter[str] = Counter()
        for row in rows:
            if not isinstance(row, dict) or set(row) != _ROW_FIELDS:
                raise ValueError("matched pair row fields are missing or unknown")
            pair_id = _validate_identifier(row["pair_id"], "pair_id")
            product_id = _validate_identifier(row["product_id"], "product_id")
            if pair_id in pair_ids:
                raise ValueError(f"duplicate matched pair_id: {pair_id!r}")
            pair_ids.add(pair_id)
            if row["split"] != split:
                raise ValueError("matched pair row is in the wrong split")
            if (
                not isinstance(row["prompt"], str)
                or "keep the package unchanged" not in row["prompt"].lower()
            ):
                raise ValueError(f"pair {pair_id!r} prompt must preserve the package")
            bbox = _validate_bbox(row["bbox"], pair_id)
            if any(value < 0 or value > _IMAGE_SIZE[0] for value in bbox):
                raise ValueError(f"pair {pair_id!r} bbox is outside the image")
            product_ids[split].add(product_id)
            rows_product_counts[product_id] += 1
            names = {
                "source": f"{split}/source/{pair_id}.png",
                "target": f"{split}/target/{pair_id}.png",
                "mask": f"{split}/mask/{pair_id}.png",
            }
            if (
                row["edit_image"] != names["source"]
                or row["image"] != names["target"]
                or row["mask"] != names["mask"]
            ):
                raise ValueError("matched row contains an unsafe or unexpected asset path")
            if any(name not in files for name in names.values()):
                raise ValueError(f"matched pair {pair_id!r} is missing an image asset")
            asset_data = {
                "source": files[names["source"]],
                "target": files[names["target"]],
                "mask": files[names["mask"]],
            }
            for role, digest_field in (
                ("source", "source_sha256"),
                ("target", "target_sha256"),
                ("mask", "mask_sha256"),
            ):
                digest = row[digest_field]
                if (
                    not isinstance(digest, str)
                    or not _DIGEST_PATTERN.fullmatch(digest)
                    or digest != hashlib.sha256(asset_data[role]).hexdigest()
                ):
                    raise ValueError(f"pair {pair_id!r} {role} checksum mismatch")
            images = _open_payload_images(pair_id, asset_data)
            fingerprint = _verify_pair_images({"pair_id": pair_id, "bbox": bbox}, images)
            fingerprints_by_split[split].add(fingerprint)
            expected_names.update(names.values())
        if len(rows_product_counts) != _PRODUCT_COUNTS[split] or any(
            count != 3 for count in rows_product_counts.values()
        ):
            raise ValueError(f"matched {split} rows must cover three pairs for each product")
    if product_ids["train"] & product_ids["dev"]:
        raise ValueError("training and development product identities overlap")
    if fingerprints_by_split["train"] & fingerprints_by_split["dev"]:
        raise ValueError("product foreground fingerprint crosses train/dev splits")
    if set(files) != expected_names:
        raise ValueError("matched archive has missing or unexpected files")
    if metadata != [
        {key: row[key] for key in ("image", "edit_image", "prompt")} for row in train_rows
    ]:
        raise ValueError("DiffSynth metadata rows do not match matched train rows")
    return tuple(train_rows), tuple(dev_rows), source_digest


def _open_payload_images(pair_id: str, assets: dict[str, bytes]) -> dict[str, Image.Image]:
    images: dict[str, Image.Image] = {}
    for role, data in assets.items():
        try:
            with Image.open(BytesIO(data)) as opened:
                expected_mode = "L" if role == "mask" else "RGB"
                if (
                    opened.format != "PNG"
                    or opened.mode != expected_mode
                    or opened.size != _IMAGE_SIZE
                ):
                    raise ValueError(
                        f"pair {pair_id!r} {role} must be a 512x512 {expected_mode} PNG"
                    )
                opened.load()
                images[role] = opened.copy()
        except (OSError, Image.DecompressionBombError) as exc:
            raise ValueError(f"pair {pair_id!r} {role} PNG is invalid") from exc
    return images


def validate_matched_study_payload_bytes(payload_bytes: bytes, run_id: str) -> MatchedStudyPayload:
    """Validate worker input bytes against the frozen study contract and expected run ID."""
    expected_run_id = _validate_run_id(run_id)
    if not isinstance(payload_bytes, bytes) or not payload_bytes:
        raise ValueError("matched payload must be non-empty bytes")
    if len(payload_bytes) > MAX_PAYLOAD_BYTES:
        raise ValueError("matched payload exceeds the configured byte limit")
    try:
        with zipfile.ZipFile(BytesIO(payload_bytes)) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ValueError("matched archive contains duplicate paths")
            if sum(info.file_size for info in infos) > 64 * 1024 * 1024:
                raise ValueError("matched archive expands beyond the configured byte limit")
            for info in infos:
                path = PurePosixPath(info.filename)
                if (
                    "\\" in info.filename
                    or path.is_absolute()
                    or ".." in path.parts
                    or "sealed_test" in path.parts
                    or not info.filename.endswith(".png")
                    and info.filename not in {"metadata.json", "matched_manifest.json"}
                ):
                    raise ValueError("matched archive contains an unsafe or unexpected path")
            files = {info.filename: archive.read(info) for info in infos}
    except (KeyError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise ValueError("matched payload is not a valid archive") from exc
    if "matched_manifest.json" not in files or "metadata.json" not in files:
        raise ValueError("matched archive is missing its manifest or metadata")
    manifest = _decode_json(files["matched_manifest.json"], "matched_manifest.json")
    metadata = _decode_json(files["metadata.json"], "metadata.json")
    train_rows, dev_rows, source_digest = _validate_payload_rows(
        manifest, metadata, files, expected_run_id
    )
    return MatchedStudyPayload(
        run_id=expected_run_id,
        source_dataset_manifest_sha256=source_digest,
        train_rows=train_rows,
        dev_rows=dev_rows,
        files=files,
    )


def plan_matched_study(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    run_id: str,
    max_payload_bytes: int = MAX_PAYLOAD_BYTES,
) -> MatchedStudyPlan:
    """Create a fresh bounded payload with all 18 train and nine dev pairs only."""
    _validate_run_id(run_id)
    if (
        isinstance(max_payload_bytes, bool)
        or not isinstance(max_payload_bytes, int)
        or max_payload_bytes <= 0
    ):
        raise ValueError("max_payload_bytes must be a positive integer")

    dataset_root = Path(dataset_dir)
    try:
        manifest_bytes = (dataset_root / "manifest.json").read_bytes()
    except OSError as exc:
        raise ValueError("manifest.json is missing or unreadable") from exc
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_digest != APPROVED_DATASET_MANIFEST_SHA256:
        raise ValueError("dataset manifest does not match the approved manifest digest")
    manifest = _decode_json(manifest_bytes, "manifest.json")
    pairs = _validate_source_manifest_metadata(manifest)

    selected_by_split = {
        split: sorted(
            (pair for pair in pairs if pair["split"] == split),
            key=lambda pair: pair["pair_id"],
        )
        for split in ("train", "dev")
    }
    if len(selected_by_split["train"]) != TRAINING_PAIR_COUNT:
        raise ValueError("dataset must contain exactly 18 train pairs")
    if len(selected_by_split["dev"]) != DEVELOPMENT_PAIR_COUNT:
        raise ValueError("dataset must contain exactly nine dev pairs")
    train_products = {pair["product_id"] for pair in selected_by_split["train"]}
    dev_products = {pair["product_id"] for pair in selected_by_split["dev"]}
    if train_products & dev_products:
        raise ValueError("training and development product identities overlap")

    archive_files: dict[str, bytes] = {}
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    fingerprints_by_split: dict[str, set[str]] = {"train": set(), "dev": set()}
    for split in ("train", "dev"):
        for pair in selected_by_split[split]:
            assets: dict[str, bytes] = {}
            images: dict[str, Image.Image] = {}
            for role in _ASSET_FIELDS:
                _, data, image = _read_checked_asset(dataset_root, pair, role)
                assets[role] = data
                images[role] = image
            fingerprint = _verify_pair_images(pair, images)
            fingerprints_by_split[split].add(fingerprint)
            row = _matched_row(pair, assets)
            rows_by_split[split].append(row)
            archive_files[row["edit_image"]] = assets["source"]
            archive_files[row["image"]] = assets["target"]
            archive_files[row["mask"]] = assets["mask"]
    if fingerprints_by_split["train"] & fingerprints_by_split["dev"]:
        raise ValueError("product foreground fingerprint crosses train/dev splits")

    archive_bytes = _archive_bytes(
        run_id,
        manifest_digest,
        rows_by_split["train"],
        rows_by_split["dev"],
        archive_files,
    )
    if len(archive_bytes) > max_payload_bytes:
        raise ValueError("matched payload exceeds the configured byte limit")
    # Validate the exact bytes we are about to persist as a second, independent contract check.
    validate_matched_study_payload_bytes(archive_bytes, run_id)

    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    payload_path = output / PAYLOAD_NAME
    digest = hashlib.sha256(archive_bytes).hexdigest()
    try:
        payload_path.write_bytes(archive_bytes)
        marker = {
            "schema_version": 1,
            "run_id": run_id,
            "payload_file": PAYLOAD_NAME,
            "payload_bytes": len(archive_bytes),
            "payload_sha256": digest,
            "source_dataset_manifest_sha256": manifest_digest,
            "training_pair_ids": [row["pair_id"] for row in rows_by_split["train"]],
            "development_pair_ids": [row["pair_id"] for row in rows_by_split["dev"]],
            "training_epochs": TRAINING_EPOCHS,
            "training_steps": TRAINING_STEPS,
        }
        with (output / RUN_MARKER_NAME).open("x", encoding="utf-8") as marker_file:
            json.dump(marker, marker_file, sort_keys=True, indent=2)
            marker_file.write("\n")
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise

    return MatchedStudyPlan(
        output_dir=output,
        payload_path=payload_path,
        run_id=run_id,
        payload_bytes=len(archive_bytes),
        payload_sha256=digest,
        source_dataset_manifest_sha256=manifest_digest,
        training_pair_ids=tuple(row["pair_id"] for row in rows_by_split["train"]),
        development_pair_ids=tuple(row["pair_id"] for row in rows_by_split["dev"]),
    )


def read_matched_study_payload(plan: MatchedStudyPlan) -> MatchedStudyPayload:
    """Recheck the immutable marker, payload digest, archive, and all payload images."""
    marker_path = plan.output_dir / RUN_MARKER_NAME
    try:
        marker = _decode_json(marker_path.read_bytes(), "matched study marker")
        payload_bytes = plan.payload_path.read_bytes()
    except OSError as exc:
        raise ValueError("matched study marker or payload is missing") from exc
    actual_digest = hashlib.sha256(payload_bytes).hexdigest()
    if (
        not isinstance(marker, dict)
        or set(marker) != _MARKER_FIELDS
        or not _is_int(marker.get("schema_version"))
        or marker.get("schema_version") != 1
        or marker.get("run_id") != plan.run_id
        or marker.get("payload_file") != PAYLOAD_NAME
        or marker.get("payload_bytes") != len(payload_bytes)
        or plan.payload_bytes != len(payload_bytes)
        or marker.get("payload_sha256") != actual_digest
        or plan.payload_sha256 != actual_digest
        or marker.get("source_dataset_manifest_sha256") != plan.source_dataset_manifest_sha256
        or marker.get("training_pair_ids") != list(plan.training_pair_ids)
        or marker.get("development_pair_ids") != list(plan.development_pair_ids)
        or marker.get("training_epochs") != TRAINING_EPOCHS
        or marker.get("training_steps") != TRAINING_STEPS
    ):
        raise ValueError("matched payload checksum or run marker mismatch")
    if len(payload_bytes) > MAX_PAYLOAD_BYTES:
        raise ValueError("matched payload exceeds the configured byte limit")
    payload = validate_matched_study_payload_bytes(payload_bytes, plan.run_id)
    if (
        tuple(row["pair_id"] for row in payload.train_rows) != plan.training_pair_ids
        or tuple(row["pair_id"] for row in payload.dev_rows) != plan.development_pair_ids
    ):
        raise ValueError("matched payload pair IDs do not match the run marker")
    return payload


def load_matched_study_plan(output_dir: str | Path) -> MatchedStudyPlan:
    """Load an existing write-once plan and verify its marker and archived payload."""
    output = Path(output_dir)
    try:
        marker = _decode_json((output / RUN_MARKER_NAME).read_bytes(), "matched study marker")
    except OSError as exc:
        raise ValueError("matched study marker is missing or invalid") from exc
    if not isinstance(marker, dict) or set(marker) != _MARKER_FIELDS:
        raise ValueError("matched study marker fields are missing or unknown")
    run_id = _validate_run_id(marker.get("run_id"))
    payload_bytes = marker.get("payload_bytes")
    payload_digest = marker.get("payload_sha256")
    source_digest = marker.get("source_dataset_manifest_sha256")
    train_ids = marker.get("training_pair_ids")
    dev_ids = marker.get("development_pair_ids")
    if (
        not _is_int(marker.get("schema_version"))
        or marker.get("schema_version") != 1
        or marker.get("payload_file") != PAYLOAD_NAME
        or isinstance(payload_bytes, bool)
        or not isinstance(payload_bytes, int)
        or payload_bytes <= 0
        or not isinstance(payload_digest, str)
        or not _DIGEST_PATTERN.fullmatch(payload_digest)
        or source_digest != APPROVED_DATASET_MANIFEST_SHA256
        or not isinstance(train_ids, list)
        or len(train_ids) != TRAINING_PAIR_COUNT
        or any(not isinstance(pair_id, str) for pair_id in train_ids)
        or not isinstance(dev_ids, list)
        or len(dev_ids) != DEVELOPMENT_PAIR_COUNT
        or any(not isinstance(pair_id, str) for pair_id in dev_ids)
        or marker.get("training_epochs") != TRAINING_EPOCHS
        or marker.get("training_steps") != TRAINING_STEPS
    ):
        raise ValueError("matched study marker fields are invalid")
    plan = MatchedStudyPlan(
        output_dir=output,
        payload_path=output / PAYLOAD_NAME,
        run_id=run_id,
        payload_bytes=payload_bytes,
        payload_sha256=payload_digest,
        source_dataset_manifest_sha256=source_digest,
        training_pair_ids=tuple(train_ids),
        development_pair_ids=tuple(dev_ids),
    )
    payload = read_matched_study_payload(plan)
    if (
        len(payload.train_rows) != TRAINING_PAIR_COUNT
        or len(payload.dev_rows) != DEVELOPMENT_PAIR_COUNT
    ):
        raise ValueError("saved matched plan does not match fixed pair counts")
    return plan


def _validate_score_image(image: Image.Image, name: str, mode: str) -> None:
    if not isinstance(image, Image.Image) or image.mode != mode or image.size != _IMAGE_SIZE:
        raise ValueError(f"{name} must be a {_IMAGE_SIZE[0]}x{_IMAGE_SIZE[1]} {mode} image")


def _max_channel_difference(left: Image.Image, right: Image.Image) -> Image.Image:
    differences = ImageChops.difference(left, right).split()
    return ImageChops.lighter(ImageChops.lighter(differences[0], differences[1]), differences[2])


def _mean_absolute_error(left: Image.Image, right: Image.Image, region: Image.Image) -> float:
    means = ImageStat.Stat(ImageChops.difference(left, right), mask=region).mean
    return sum(means) / 3


def _fraction_over_8(
    left: Image.Image, right: Image.Image, region: Image.Image, region_pixels: int
) -> float:
    over_8 = _max_channel_difference(left, right).point(lambda value: 255 if value > 8 else 0)
    selected = ImageChops.multiply(over_8, region)
    changed_pixels = selected.histogram()[255]
    return changed_pixels / region_pixels


def score_pair(
    source: Image.Image,
    target: Image.Image,
    mask: Image.Image,
    generated: Image.Image,
) -> dict[str, float | int]:
    """Score fixed-position product pixels and background changes against the fixture.

    MAE averages RGB channel differences over each full region. Threshold fractions count a pixel
    only when its largest channel difference is greater than 8. Foreground error measures strict
    pixel preservation at fixed positions; it does not measure OCR, shape, or general image quality.
    """
    for name, image in (("source", source), ("target", target), ("generated", generated)):
        _validate_score_image(image, name, "RGB")
    _validate_score_image(mask, "mask", "L")
    histogram = mask.histogram()
    if any(histogram[1:255]):
        raise ValueError("mask must contain only 0 and 255")
    foreground_pixels, background_pixels = histogram[255], histogram[0]
    if not foreground_pixels or not background_pixels:
        raise ValueError("mask must contain both foreground and background pixels")
    background = ImageChops.invert(mask)
    return {
        "foreground_mean_absolute_error": _mean_absolute_error(source, generated, mask),
        "foreground_fraction_pixels_over_8": _fraction_over_8(
            source, generated, mask, foreground_pixels
        ),
        "background_target_mean_absolute_error": _mean_absolute_error(
            target, generated, background
        ),
        "background_fraction_changed_from_source_over_8": _fraction_over_8(
            source, generated, background, background_pixels
        ),
        "foreground_pixels": foreground_pixels,
        "background_pixels": background_pixels,
    }
