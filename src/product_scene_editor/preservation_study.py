"""CPU-only planning and validation for the opt-in preservation study."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import struct
import tempfile
import zipfile
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from . import controlled3d_assets, controlled3d_schema, controlled3d_study_payload
from .controlled3d_study_payload import Controlled3DStudyPayload
from .matched_study import TRAINING_EPOCHS, TRAINING_STEPS

RUN_MARKER_NAME = ".preservation-study-run.json"
PAYLOAD_NAME = "preservation-study-input.zip"
PARENT_RUN_ID = "scenetune-controlled3d-2026-10-11-r1"
PARENT_PAYLOAD_SHA256 = "c1facb0534d77c63385ed5ef42b4763d4fe89b6cbf89c11f9097ecaf80b2c925"
APPROVED_PRESERVATION_MANIFEST_SHA256: str | None = (
    "4cf9dfeaa9cf5a9edd9587c04bcc93e82d611d25b364cbcf1486e7ca3f2c5aeb"
)

_PARENT_MEMBER = "parent.zip"
_MANIFEST_MEMBER = "preservation_manifest.json"
_METADATA_MEMBER = "metadata.json"
_STATES = ("cream", "sage", "clay")
_SPLITS = {"train": 6, "dev": 3}
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_MAX_PARENT_BYTES = 16 * 1024 * 1024
_MAX_PAYLOAD_BYTES = 24 * 1024 * 1024
_MAX_MEMBERS = 128
_MAX_EXPANDED_BYTES = 64 * 1024 * 1024
_MARKER_FIELDS = {
    "schema_version",
    "run_id",
    "payload_file",
    "payload_bytes",
    "payload_sha256",
    "source_dataset_manifest_sha256",
    "preservation_manifest_sha256",
    "parent_payload_sha256",
    "training_pair_ids",
    "development_pair_ids",
    "training_epochs",
    "training_steps",
}


@dataclass(frozen=True)
class PreservationStudyPlan:
    """A write-once local payload plan compatible with MatchedStudyPlan."""

    output_dir: Path
    payload_path: Path
    run_id: str
    payload_bytes: int
    payload_sha256: str
    source_dataset_manifest_sha256: str
    training_pair_ids: tuple[str, ...]
    development_pair_ids: tuple[str, ...]
    parent_payload_sha256: str
    preservation_manifest_sha256: str


@dataclass(frozen=True)
class PreservationStudyPayload:
    """Validated rows and virtual image files derived from the pinned parent."""

    run_id: str
    source_dataset_manifest_sha256: str
    train_rows: tuple[dict[str, Any], ...]
    dev_rows: tuple[dict[str, Any], ...]
    files: dict[str, bytes]
    parent_payload_sha256: str
    preservation_manifest_sha256: str


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ValueError(f"Preservation {label} digest pin is unset or invalid")
    return value


def _validate_run_id(run_id: object) -> str:
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id) or ".." in run_id:
        raise ValueError("preservation run ID is invalid")
    return run_id


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


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def _png_bytes(image: Image.Image) -> bytes:
    if image.mode != "RGB" or image.size != (512, 512):
        raise ValueError("derived targets must be 512x512 RGB images")
    pixels = image.tobytes()
    scanlines = b"".join(b"\x00" + pixels[row * 1536 : (row + 1) * 1536] for row in range(512))
    deflate = bytearray(b"\x78\x01")
    for offset in range(0, len(scanlines), 65535):
        block = scanlines[offset : offset + 65535]
        final = offset + len(block) == len(scanlines)
        deflate.append(1 if final else 0)
        deflate.extend(struct.pack("<HH", len(block), len(block) ^ 0xFFFF))
        deflate.extend(block)
    zlib_stream = bytes(deflate) + struct.pack(">I", zlib.adler32(scanlines))
    ihdr = struct.pack(">IIBBBBB", 512, 512, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib_stream)
        + _png_chunk(b"IEND", b"")
    )


def _parent_payload(parent_bytes: bytes) -> Controlled3DStudyPayload:
    expected = _require_digest(PARENT_PAYLOAD_SHA256, "parent payload")
    if (
        not isinstance(parent_bytes, bytes)
        or not parent_bytes
        or len(parent_bytes) > _MAX_PARENT_BYTES
    ):
        raise ValueError("parent Controlled3D payload is empty or oversized")
    if _digest(parent_bytes) != expected:
        raise ValueError("parent Controlled3D payload does not match the immutable pin")
    parent = controlled3d_study_payload.validate_controlled3d_study_payload_bytes(
        parent_bytes, PARENT_RUN_ID
    )
    return parent


def _target_bytes(parent: Controlled3DStudyPayload, row: dict[str, Any]) -> bytes:
    source = controlled3d_study_payload._open_png(
        parent.files[row["edit_image"]], "RGB", "parent source"
    )
    target = controlled3d_study_payload._open_png(
        parent.files[row["image"]], "RGB", "parent target"
    )
    mask = controlled3d_study_payload._open_png(
        parent.files[row["mask"]], "L", "parent product mask"
    )
    return _png_bytes(Image.composite(source, target, mask))


def _make_row(
    parent: Controlled3DStudyPayload,
    source_row: dict[str, Any],
    condition: str,
    target_assets: dict[str, bytes],
) -> dict[str, Any]:
    split = source_row["split"]
    scene_pair_id = source_row["scene_pair_id"]
    pair_id = f"{scene_pair_id}-{condition}"
    source_raw = parent.files[source_row["edit_image"]]
    mask_raw = parent.files[source_row["mask"]]
    source_path = f"{split}/source/{pair_id}.png"
    mask_path = f"{split}/mask/{pair_id}.png"
    if condition == "identity":
        image_path = source_path
        target_raw = source_raw
        prompt = (
            "Keep the package unchanged; preserve the source image exactly without editing "
            "the background, product, packaging, label, shape, or position."
        )
    else:
        image_path = f"{split}/target/{pair_id}.png"
        target_raw = _target_bytes(parent, source_row)
        prompt = source_row["prompt"]
        target_assets[image_path] = target_raw
    return {
        "pair_id": pair_id,
        "product_id": source_row["product_id"],
        "split": split,
        "prompt": prompt,
        "image": image_path,
        "edit_image": source_path,
        "mask": mask_path,
        "bbox": list(source_row["bbox"]),
        "source_sha256": _digest(source_raw),
        "target_sha256": _digest(target_raw),
        "mask_sha256": _digest(mask_raw),
    }


def _derive(
    parent: Controlled3DStudyPayload, parent_sha256: str
) -> tuple[dict[str, Any], bytes, bytes, dict[str, bytes]]:
    pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    product_pair: dict[tuple[str, str], str] = {}
    for row in (*parent.train_rows, *parent.dev_rows):
        key = (row["split"], row["product_id"])
        scene_pair_id = row["scene_pair_id"]
        if key in product_pair and product_pair[key] != scene_pair_id:
            raise ValueError("parent must contain exactly one pair per product")
        product_pair[key] = scene_pair_id
        pairs[scene_pair_id][row["state"]] = row
    if (
        len({key[1] for key in product_pair if key[0] == "train"}) != 6
        or len({key[1] for key in product_pair if key[0] == "dev"}) != 3
    ):
        raise ValueError("parent must contain six train and three dev products")

    targets: dict[str, bytes] = {}
    train_rows: list[dict[str, Any]] = []
    train_products = sorted(
        (product_id, pair_id)
        for (split, product_id), pair_id in product_pair.items()
        if split == "train"
    )
    for product_index, (_product_id, scene_pair_id) in enumerate(train_products):
        states = pairs[scene_pair_id]
        train_rows.append(_make_row(parent, states[_STATES[0]], "identity", targets))
        omitted_state = _STATES[product_index % len(_STATES)]
        for state in _STATES:
            if state != omitted_state:
                train_rows.append(_make_row(parent, states[state], state, targets))
    dev_rows: list[dict[str, Any]] = []
    dev_products = sorted(
        (product_id, pair_id)
        for (split, product_id), pair_id in product_pair.items()
        if split == "dev"
    )
    for _product_id, scene_pair_id in dev_products:
        for state in _STATES:
            dev_rows.append(_make_row(parent, pairs[scene_pair_id][state], state, targets))
    train_rows.sort(key=lambda row: row["pair_id"])
    dev_rows.sort(key=lambda row: row["pair_id"])
    conditions = Counter(row["pair_id"].rsplit("-", 1)[-1] for row in train_rows)
    manifest = {
        "schema_version": 1,
        "parent_run_id": PARENT_RUN_ID,
        "parent_payload_sha256": parent_sha256,
        "source_dataset_manifest_sha256": parent.source_dataset_manifest_sha256,
        "product_counts": _SPLITS,
        "training_condition_counts": dict(conditions),
        "training_epochs": TRAINING_EPOCHS,
        "training_steps": TRAINING_STEPS,
        "train_rows": train_rows,
        "dev_rows": dev_rows,
    }
    manifest_raw = _json_bytes(manifest)
    metadata_raw = _json_bytes(
        [{key: row[key] for key in ("image", "edit_image", "prompt")} for row in train_rows]
    )
    return manifest, manifest_raw, metadata_raw, targets


def _zip_bytes(
    parent_bytes: bytes, manifest_raw: bytes, metadata_raw: bytes, targets: dict[str, bytes]
) -> bytes:
    members = {
        _PARENT_MEMBER: parent_bytes,
        _MANIFEST_MEMBER: manifest_raw,
        _METADATA_MEMBER: metadata_raw,
        **targets,
    }
    if len(members) > _MAX_MEMBERS:
        raise ValueError("preservation payload has too many archive members")
    if sum(map(len, members.values())) > _MAX_EXPANDED_BYTES:
        raise ValueError("preservation payload expands beyond the configured byte limit")
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, raw in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, raw)
    result = buffer.getvalue()
    if len(result) > _MAX_PAYLOAD_BYTES:
        raise ValueError("preservation payload exceeds the configured byte limit")
    return result


def build_preservation_payload(parent_bytes: bytes, run_id: str) -> tuple[bytes, str]:
    """Build deterministic CPU-derived bytes before root binds the new manifest pin."""
    _validate_run_id(run_id)
    parent = _parent_payload(parent_bytes)
    parent_sha256 = _digest(parent_bytes)
    _manifest, manifest_raw, metadata_raw, targets = _derive(parent, parent_sha256)
    return (
        _zip_bytes(parent_bytes, manifest_raw, metadata_raw, targets),
        _digest(manifest_raw),
    )


def _virtual_files(
    parent: Controlled3DStudyPayload,
    manifest: dict[str, Any],
    archive_files: dict[str, bytes],
) -> dict[str, bytes]:
    files = {
        _METADATA_MEMBER: archive_files[_METADATA_MEMBER],
        _MANIFEST_MEMBER: archive_files[_MANIFEST_MEMBER],
    }
    parent_rows = {
        (row["split"], row["scene_pair_id"], row["state"]): row
        for row in (*parent.train_rows, *parent.dev_rows)
    }
    for row in (*manifest["train_rows"], *manifest["dev_rows"]):
        condition = row["pair_id"].rsplit("-", 1)[-1]
        scene_pair_id = row["pair_id"][: -(len(condition) + 1)]
        state = _STATES[0] if condition == "identity" else condition
        source_row = parent_rows[(row["split"], scene_pair_id, state)]
        files[row["edit_image"]] = parent.files[source_row["edit_image"]]
        files[row["mask"]] = parent.files[source_row["mask"]]
        if row["image"] == row["edit_image"]:
            files[row["image"]] = parent.files[source_row["edit_image"]]
        else:
            files[row["image"]] = archive_files[row["image"]]
    return files


def validate_preservation_study_payload_bytes(raw: bytes, run_id: str) -> PreservationStudyPayload:
    """Validate outer bytes, pinned parent, deterministic derivation, and every row asset."""
    manifest_pin = _require_digest(APPROVED_PRESERVATION_MANIFEST_SHA256, "manifest")
    expected_run_id = _validate_run_id(run_id)
    if not isinstance(raw, bytes) or not raw or len(raw) > _MAX_PAYLOAD_BYTES:
        raise ValueError("preservation payload is empty or exceeds the byte limit")
    try:
        with zipfile.ZipFile(BytesIO(raw)) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or len(names) > _MAX_MEMBERS:
                raise ValueError("preservation archive has duplicate paths or too many members")
            if sum(info.file_size for info in infos) > _MAX_EXPANDED_BYTES:
                raise ValueError("preservation archive expands beyond the byte limit")
            for info in infos:
                archive_path = PurePosixPath(info.filename)
                if (
                    "\\" in info.filename
                    or archive_path.is_absolute()
                    or ".." in archive_path.parts
                ):
                    raise ValueError("preservation archive contains an unsafe path")
            archive_files = {info.filename: archive.read(info) for info in infos}
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise ValueError("preservation payload is not a valid archive") from exc
    for member in (_PARENT_MEMBER, _MANIFEST_MEMBER, _METADATA_MEMBER):
        if member not in archive_files:
            raise ValueError("preservation payload is missing required metadata")
    parent_bytes = archive_files[_PARENT_MEMBER]
    parent = _parent_payload(parent_bytes)
    parent_sha256 = _digest(parent_bytes)
    manifest = controlled3d_schema._loads_any(archive_files[_MANIFEST_MEMBER])
    metadata = controlled3d_schema._loads_any(archive_files[_METADATA_MEMBER])
    if archive_files[_MANIFEST_MEMBER] != _json_bytes(manifest) or archive_files[
        _METADATA_MEMBER
    ] != _json_bytes(metadata):
        raise ValueError("preservation payload JSON must be canonical")
    expected_manifest, manifest_raw, metadata_raw, target_assets = _derive(parent, parent_sha256)
    manifest_sha256 = _digest(manifest_raw)
    if manifest_sha256 != manifest_pin:
        raise ValueError("preservation manifest digest differs from the reviewed pin")
    if (
        archive_files[_MANIFEST_MEMBER] != manifest_raw
        or archive_files[_METADATA_MEMBER] != metadata_raw
    ):
        raise ValueError("preservation manifest or metadata differs from deterministic derivation")
    expected_names = {
        _PARENT_MEMBER,
        _MANIFEST_MEMBER,
        _METADATA_MEMBER,
        *target_assets,
    }
    if set(archive_files) != expected_names:
        raise ValueError("preservation archive contains missing or unexpected files")
    for path, expected_raw in target_assets.items():
        if archive_files[path] != expected_raw:
            raise ValueError(f"preservation target differs from exact CPU derivation: {path}")
    files = _virtual_files(parent, expected_manifest, archive_files)
    return PreservationStudyPayload(
        run_id=expected_run_id,
        source_dataset_manifest_sha256=parent.source_dataset_manifest_sha256,
        train_rows=tuple(expected_manifest["train_rows"]),
        dev_rows=tuple(expected_manifest["dev_rows"]),
        files=files,
        parent_payload_sha256=parent_sha256,
        preservation_manifest_sha256=manifest_sha256,
    )


def _publish_once(target: Path, payload: bytes, marker: bytes) -> None:
    stage = Path(tempfile.mkdtemp(prefix=".preservation-study-stage-", dir=target.parent))
    claimed = False
    try:
        (stage / PAYLOAD_NAME).write_bytes(payload)
        (stage / RUN_MARKER_NAME).write_bytes(marker)
        os.mkdir(target)
        claimed = True
        shutil.copyfile(stage / PAYLOAD_NAME, target / PAYLOAD_NAME)
        shutil.copyfile(stage / RUN_MARKER_NAME, target / RUN_MARKER_NAME)
        if _digest(_read_regular(target / PAYLOAD_NAME, _MAX_PAYLOAD_BYTES)) != _digest(payload):
            raise ValueError("preservation payload changed while publishing")
    except Exception:
        if claimed:
            shutil.rmtree(target, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def plan_preservation_study(
    parent_payload_path: str | Path,
    output_dir: str | Path,
    *,
    run_id: str,
) -> PreservationStudyPlan:
    """Derive and publish a pinned preservation payload into a new directory."""
    manifest_pin = _require_digest(APPROVED_PRESERVATION_MANIFEST_SHA256, "manifest")
    run_id = _validate_run_id(run_id)
    parent_bytes = _read_regular(parent_payload_path, _MAX_PARENT_BYTES)
    payload_raw, derived_sha256 = build_preservation_payload(parent_bytes, run_id)
    if derived_sha256 != manifest_pin:
        raise ValueError("derived preservation manifest does not match the reviewed pin")
    payload = validate_preservation_study_payload_bytes(payload_raw, run_id)
    target = Path(output_dir)
    if target.exists() or target.is_symlink() or not target.parent.is_dir():
        raise FileExistsError("preservation plan destination must be new with an existing parent")
    controlled3d_assets._reject_destination_path(str(target))
    payload_sha256 = _digest(payload_raw)
    marker = {
        "schema_version": 1,
        "run_id": run_id,
        "payload_file": PAYLOAD_NAME,
        "payload_bytes": len(payload_raw),
        "payload_sha256": payload_sha256,
        "source_dataset_manifest_sha256": payload.source_dataset_manifest_sha256,
        "preservation_manifest_sha256": payload.preservation_manifest_sha256,
        "parent_payload_sha256": payload.parent_payload_sha256,
        "training_pair_ids": sorted(row["pair_id"] for row in payload.train_rows),
        "development_pair_ids": sorted(row["pair_id"] for row in payload.dev_rows),
        "training_epochs": TRAINING_EPOCHS,
        "training_steps": TRAINING_STEPS,
    }
    _publish_once(target, payload_raw, _json_bytes(marker))
    return load_preservation_study_plan(target)


def read_preservation_study_payload(plan: PreservationStudyPlan) -> PreservationStudyPayload:
    """Re-read a plan payload and verify all pins and bytes before worker use."""
    if not isinstance(plan, PreservationStudyPlan):
        raise ValueError("preservation plan type is invalid")
    raw = _read_regular(plan.payload_path, _MAX_PAYLOAD_BYTES)
    if len(raw) != plan.payload_bytes or _digest(raw) != plan.payload_sha256:
        raise ValueError("preservation plan payload bytes changed")
    payload = validate_preservation_study_payload_bytes(raw, plan.run_id)
    if (
        payload.source_dataset_manifest_sha256 != plan.source_dataset_manifest_sha256
        or payload.parent_payload_sha256 != plan.parent_payload_sha256
        or payload.preservation_manifest_sha256 != plan.preservation_manifest_sha256
        or tuple(row["pair_id"] for row in payload.train_rows) != plan.training_pair_ids
        or tuple(row["pair_id"] for row in payload.dev_rows) != plan.development_pair_ids
    ):
        raise ValueError("preservation plan metadata or pair IDs do not bind its payload")
    return payload


def load_preservation_study_plan(output_dir: str | Path) -> PreservationStudyPlan:
    """Load a saved write-once plan and verify its marker against the payload bytes."""
    _require_digest(APPROVED_PRESERVATION_MANIFEST_SHA256, "manifest")
    directory = Path(output_dir)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("preservation plan directory is missing or unsafe")
    marker_raw = _read_regular(directory / RUN_MARKER_NAME, 1024 * 1024)
    marker = controlled3d_schema._loads_any(marker_raw)
    if type(marker) is not dict or set(marker) != _MARKER_FIELDS:
        raise ValueError("preservation plan marker fields are invalid")
    if marker_raw != _json_bytes(marker):
        raise ValueError("preservation plan marker is not canonical JSON")
    run_id = _validate_run_id(marker["run_id"])
    if (
        type(marker["schema_version"]) is not int
        or marker["schema_version"] != 1
        or marker["payload_file"] != PAYLOAD_NAME
        or type(marker["payload_bytes"]) is not int
        or type(marker["training_epochs"]) is not int
        or marker["training_epochs"] != TRAINING_EPOCHS
        or type(marker["training_steps"]) is not int
        or marker["training_steps"] != TRAINING_STEPS
        or marker["parent_payload_sha256"]
        != _require_digest(PARENT_PAYLOAD_SHA256, "parent payload")
        or marker["preservation_manifest_sha256"] != APPROVED_PRESERVATION_MANIFEST_SHA256
    ):
        raise ValueError("preservation plan marker differs from the reviewed profile")
    for field in ("payload_sha256", "source_dataset_manifest_sha256"):
        _require_digest(marker[field], field)
    for field in ("training_pair_ids", "development_pair_ids"):
        values = marker[field]
        if type(values) is not list or any(type(item) is not str for item in values):
            raise ValueError("preservation plan pair IDs are invalid")
        if values != sorted(set(values)):
            raise ValueError("preservation plan pair IDs are not unique and ordered")
    payload_path = directory / PAYLOAD_NAME
    payload_raw = _read_regular(payload_path, _MAX_PAYLOAD_BYTES)
    if (
        len(payload_raw) != marker["payload_bytes"]
        or _digest(payload_raw) != marker["payload_sha256"]
    ):
        raise ValueError("preservation plan payload differs from its marker")
    payload = validate_preservation_study_payload_bytes(payload_raw, run_id)
    if (
        payload.source_dataset_manifest_sha256 != marker["source_dataset_manifest_sha256"]
        or payload.parent_payload_sha256 != marker["parent_payload_sha256"]
        or payload.preservation_manifest_sha256 != marker["preservation_manifest_sha256"]
        or sorted(row["pair_id"] for row in payload.train_rows) != marker["training_pair_ids"]
        or sorted(row["pair_id"] for row in payload.dev_rows) != marker["development_pair_ids"]
    ):
        raise ValueError("preservation plan marker does not bind its payload rows")
    return PreservationStudyPlan(
        output_dir=directory,
        payload_path=payload_path,
        run_id=run_id,
        payload_bytes=len(payload_raw),
        payload_sha256=_digest(payload_raw),
        source_dataset_manifest_sha256=payload.source_dataset_manifest_sha256,
        training_pair_ids=tuple(marker["training_pair_ids"]),
        development_pair_ids=tuple(marker["development_pair_ids"]),
        parent_payload_sha256=payload.parent_payload_sha256,
        preservation_manifest_sha256=payload.preservation_manifest_sha256,
    )
