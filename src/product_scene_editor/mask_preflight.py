"""Canonical, CPU-only input contract for the pinned BiRefNet mask preflight."""

from __future__ import annotations

import hashlib
import json
import re
import stat
import warnings
import zipfile
import zlib
from io import BytesIO
from pathlib import PurePosixPath
from typing import Any

from PIL import Image

MODEL = {
    "repo_id": "ZhengPeng7/BiRefNet_lite",
    "revision": "aa62cd87eafb9cc43056d08ef3615a14628b831d",
}
MODEL_FILES = {
    "BiRefNet_config.py": {
        "sha256": "e7b8c2a74f6cea6a59553d517f71d47f2c1d90e670a13416af17c25fe2f3dc52",
        "size": 298,
    },
    "birefnet.py": {
        "sha256": "af8568b5be406bf4d2a68a7ed6d72e40f73b37a1fb6fc9ebd71b5b3cbcd069c9",
        "size": 92134,
    },
    "config.json": {
        "sha256": "9dc8614fccddd40c601aeadc69b9db6dd820598179b2a2198492e6ffa016a824",
        "size": 410,
    },
    "model.safetensors": {
        "sha256": "4417d89795250e698c3cb0ae8df15743810065f646f48a694fdfa7ca052d0815",
        "size": 177634392,
    },
}
RUNTIME_PINS = {
    "torch": "2.5.1+cu124",
    "torchvision": "0.20.1+cu124",
    "numpy": "1.26.4",
    "transformers": "4.48.3",
    "timm": "1.0.15",
    "kornia": "0.8.0",
    "einops": "0.8.1",
    "huggingface-hub": "0.28.1",
    "safetensors": "0.5.3",
    "Pillow": "12.0.0",
}
CONTEXT = {"profile": "reflex-personal", "workspace": "rajath-61258", "environment": "main"}
SETTINGS = {
    "source_size": [512, 512],
    "input_size": [1024, 1024],
    "input_resampling": "PIL_BILINEAR",
    "input_color_mode": "RGB",
    "input_dtype": "float32",
    "normalization_mean": [0.485, 0.456, 0.406],
    "normalization_std": [0.229, 0.224, 0.225],
    "output_index": -1,
    "activation": "sigmoid",
    "output_resampling": "bilinear_align_corners_false",
    "output_size": [512, 512],
    "alpha_threshold": 0.5,
    "protected_core_erosion_px": 2,
    "protected_core_filter": "PIL_MINFILTER_5X5",
}
MAX_PAYLOAD_BYTES = 16 * 1024 * 1024
MAX_MANIFEST_BYTES = 256 * 1024
MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_UNCOMPRESSED_BYTES = 16 * 1024 * 1024
MAX_RESULT_BYTES = 16 * 1024 * 1024

_MANIFEST_FIELDS = {
    "schema_version",
    "profile",
    "source_sha",
    "context",
    "model",
    "model_files",
    "runtime_pins",
    "settings",
    "cases",
}
_CASE_FIELDS = {"case_id", "source_rgb"}
_ASSET_FIELDS = {"path", "sha256"}
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_PATH = re.compile(r"^cases/[A-Za-z0-9][A-Za-z0-9._-]{0,63}/source\.png$")


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _require_object(value: object, fields: set[str], role: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{role} fields are missing or unknown")
    return value


def _validate_manifest(manifest: object) -> dict[str, Any]:
    data = _require_object(manifest, _MANIFEST_FIELDS, "manifest")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("mask preflight schema version must be 1")
    if data["profile"] != "mask-preflight":
        raise ValueError("mask preflight profile is unsupported")
    if not isinstance(data["source_sha"], str) or _SHA1.fullmatch(data["source_sha"]) is None:
        raise ValueError("source_sha must be a lowercase 40-character digest")
    if data["context"] != CONTEXT or data["model"] != MODEL:
        raise ValueError("mask preflight target or model pin differs from the reviewed contract")
    if data["model_files"] != MODEL_FILES:
        raise ValueError("BiRefNet pinned file digests differ from the reviewed contract")
    if data["runtime_pins"] != RUNTIME_PINS:
        raise ValueError("mask preflight runtime pins differ from the reviewed contract")
    if _canonical_json(data["settings"]) != _canonical_json(SETTINGS):
        raise ValueError("mask preflight settings differ from the frozen contract")
    cases = data["cases"]
    if not isinstance(cases, list) or len(cases) != 24:
        raise ValueError("mask preflight requires exactly 24 source cases")
    return data


def _safe_source_path(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("source asset path must be a string")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or _SOURCE_PATH.fullmatch(value) is None
    ):
        raise ValueError("source asset path is unsafe or does not match its case")
    return value


def _validate_source_png(content: bytes, case_id: str) -> None:
    if not isinstance(content, bytes) or not content:
        raise ValueError(f"case {case_id} source PNG is empty or invalid")
    if len(content) > MAX_SOURCE_BYTES:
        raise ValueError(f"case {case_id} source PNG exceeds the 2 MiB limit")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as opened:
                if (
                    opened.format != "PNG"
                    or opened.mode != "RGB"
                    or opened.size != (512, 512)
                    or getattr(opened, "n_frames", 1) != 1
                ):
                    raise ValueError(f"case {case_id} source must be one 512x512 RGB PNG")
                opened.load()
    except (
        OSError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError(f"case {case_id} source is not a readable 512x512 RGB PNG") from exc


def _validate_cases(
    manifest: dict[str, Any], files: dict[str, bytes], *, require_sorted: bool
) -> None:
    cases = manifest["cases"]
    ids: list[str] = []
    source_hashes: set[str] = set()
    expected_files: set[str] = set()
    for value in cases:
        case = _require_object(value, _CASE_FIELDS, "case")
        case_id = case["case_id"]
        if not isinstance(case_id, str) or _SAFE_ID.fullmatch(case_id) is None:
            raise ValueError("case_id must be a safe identifier")
        ids.append(case_id)
        asset = _require_object(case["source_rgb"], _ASSET_FIELDS, "source_rgb")
        path = _safe_source_path(asset["path"])
        if path != f"cases/{case_id}/source.png" or path in expected_files:
            raise ValueError("source asset path must be unique and match its case ID")
        expected_files.add(path)
        digest = asset["sha256"]
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError("source_rgb SHA-256 must be lowercase hexadecimal")
        if digest in source_hashes:
            raise ValueError("source_rgb SHA-256 values must be unique")
        source_hashes.add(digest)
        content = files.get(path)
        if not isinstance(content, bytes):
            raise ValueError(f"case {case_id} source PNG is missing or invalid")
        _validate_source_png(content, case_id)
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"case {case_id} source SHA-256 mismatch")
    if len(set(ids)) != 24:
        raise ValueError("mask preflight case IDs must be unique")
    if require_sorted and ids != sorted(ids):
        raise ValueError("mask preflight cases must be sorted by case_id")
    if set(files) != expected_files:
        raise ValueError("mask preflight source members are missing or unexpected")


def _serialize_payload(manifest: dict[str, Any], files: dict[str, bytes]) -> bytes:
    contents = {**files, "manifest.json": _canonical_json(manifest)}
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, content in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            archive.writestr(info, content)
    return output.getvalue()


def build_payload(manifest: dict[str, Any], files: dict[str, bytes]) -> bytes:
    """Validate and encode the frozen 24-case payload as deterministic ZIP_STORED bytes."""
    data = _validate_manifest(manifest)
    _validate_cases(data, files, require_sorted=True)
    manifest_bytes = _canonical_json(data)
    if len(manifest_bytes) > MAX_MANIFEST_BYTES:
        raise ValueError("mask preflight manifest exceeds the 256 KiB limit")
    if sum(map(len, files.values())) + len(manifest_bytes) > MAX_TOTAL_UNCOMPRESSED_BYTES:
        raise ValueError("mask preflight payload expands beyond the 16 MiB limit")
    payload = _serialize_payload(data, files)
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("mask preflight payload exceeds the 16 MiB limit")
    return payload


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _safe_archive_member(name: object) -> str:
    if not isinstance(name, str) or (
        name != "manifest.json" and _SOURCE_PATH.fullmatch(name) is None
    ):
        raise ValueError("mask preflight ZIP contains an unsafe member")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or path.as_posix() != name
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("mask preflight ZIP contains an unsafe member")
    return name


def validate_payload_bytes(payload: bytes) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Validate archive shape, frozen pins, canonical bytes, and every source PNG."""
    if not isinstance(payload, bytes) or not payload or len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("mask preflight payload is empty or exceeds the 16 MiB limit")
    try:
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if not infos or len(set(names)) != len(names) or names != sorted(names):
                raise ValueError("mask preflight ZIP members must be unique and sorted")
            if archive.comment:
                raise ValueError("mask preflight ZIP comment is not allowed")
            total_size = 0
            for info in infos:
                name = _safe_archive_member(info.filename)
                mode = info.external_attr >> 16
                if (
                    info.is_dir()
                    or stat.S_IFMT(mode) != stat.S_IFREG
                    or info.flag_bits & 1
                    or info.compress_type != zipfile.ZIP_STORED
                ):
                    raise ValueError(
                        "mask preflight ZIP contains a non-regular or compressed member"
                    )
                limit = MAX_MANIFEST_BYTES if name == "manifest.json" else MAX_SOURCE_BYTES
                if info.file_size > limit:
                    raise ValueError("mask preflight ZIP member exceeds its size limit")
                total_size += info.file_size
                if total_size > MAX_TOTAL_UNCOMPRESSED_BYTES:
                    raise ValueError("mask preflight ZIP expands beyond the 16 MiB limit")
            if names.count("manifest.json") != 1:
                raise ValueError("mask preflight ZIP must contain one manifest.json")
            files = {
                info.filename: archive.read(info)
                for info in infos
                if info.filename != "manifest.json"
            }
            manifest_bytes = archive.read("manifest.json")
            manifest = json.loads(
                manifest_bytes.decode("utf-8"),
                object_pairs_hook=_unique_pairs,
                parse_constant=_reject_constant,
            )
    except (
        KeyError,
        OSError,
        RuntimeError,
        EOFError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
        zlib.error,
    ) as exc:
        raise ValueError("mask preflight payload is not a readable ZIP") from exc
    data = _validate_manifest(manifest)
    _validate_cases(data, files, require_sorted=True)
    if _serialize_payload(data, files) != payload:
        raise ValueError("mask preflight ZIP is not in canonical deterministic form")
    return data, files
