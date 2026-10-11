"""Strict JSON, schema, provenance-pin, and identity exclusion rules."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any
from urllib.parse import urlsplit

_MAX_JSON_BYTES = 8 * 1024 * 1024
_MAX_JSON_DEPTH = 32
_MAX_JSON_COLLECTION = 100_000
_SPLITS = ("train", "dev", "sealed_test")
_STATES = ("source", "cream", "sage", "clay")
_HEX = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:._-]{0,127}$")
_DENIED_PRODUCTS = frozenset(
    {
        "studio-preview-carton-001",
        "studio-preview-carton-002",
        "studio-preview-carton-003",
        "studio-preview-carton-004",
        "studio-preview-carton-005",
        "studio-preview-dev-carton-001",
        "studio-preview-dev-jar-001",
        "studio-preview-dev-pouch-001",
        "8076800195057",
        "8000500037560",
        "3168930010883",
        "OFF-R1-003",
        "OFF-R1-005",
        "OFF-R1-006",
    }
)
_DENIED_FAMILIES = frozenset(
    {
        "legacy-studio-preview-family-001",
        "controlled-studio-family-002",
        "controlled-studio-family-003",
        "controlled-studio-family-004",
        "controlled-studio-family-005",
        "varied-studio-sweep-dev-001",
        "varied-window-tabletop-dev-001",
        "varied-kitchen-shelf-dev-001",
        "controlled3d-neutral-wall-support-001",
        "controlled3d-neutral-wall-support-shelf-001",
        "off-brand:barilla",
        "off-brand:kinder",
        "off-brand:cruesli",
    }
)

_DENIED_PAIR_IDS = frozenset({"OFF-R1-003", "OFF-R1-005", "OFF-R1-006"})


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _check_json(value: Any, depth: int = 0) -> None:
    if depth > _MAX_JSON_DEPTH:
        raise ValueError("JSON exceeds maximum nesting depth")
    if type(value) is dict:
        if len(value) > _MAX_JSON_COLLECTION:
            raise ValueError("JSON object exceeds maximum entry count")
        for key, child in value.items():
            if type(key) is not str:
                raise ValueError("JSON keys must be strings")
            _check_json(child, depth + 1)
    elif type(value) is list:
        if len(value) > _MAX_JSON_COLLECTION:
            raise ValueError("JSON array exceeds maximum entry count")
        for child in value:
            _check_json(child, depth + 1)
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError("non-finite JSON number")
    elif value is None or type(value) in (str, bool, int):
        return
    else:
        raise ValueError("unsupported JSON value")


def _loads_any(raw: bytes | str) -> Any:
    if len(raw) > _MAX_JSON_BYTES:
        raise ValueError("JSON exceeds maximum size")
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid JSON") from exc
    _check_json(value)
    return value


def _dump(value: Any) -> str:
    _check_json(value)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _fields(value: Any, names: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != names:
        raise ValueError(f"{label} fields are missing or unknown")
    return value


def _id(value: Any, label: str) -> str:
    if type(value) is not str or not _ID.fullmatch(value):
        raise ValueError(f"{label} must be a valid ID")
    return value


def _split(value: Any) -> str:
    if type(value) is not str or value not in _SPLITS:
        raise ValueError("split must be train, dev, or sealed_test")
    return value


def _digest(value: Any, label: str = "SHA-256") -> str:
    if type(value) is not str or not _HEX.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _safe_path(value: Any, split: str) -> str:
    if type(value) is not str or not value or len(value) > 512 or "\\" in value or "\x00" in value:
        raise ValueError("unsafe asset path")
    parts = value.split("/")
    if parts[0] != split or any(part in ("", ".", "..") for part in parts) or value.startswith("/"):
        raise ValueError("unsafe asset path")
    return value


def _ref(value: Any, split: str, names: set[str], role: str, used: set[str]) -> dict[str, Any]:
    item = _fields(value, names, role)
    path = _safe_path(item["path"], split)
    if path in used:
        raise ValueError(f"asset path is reused: {path}")
    used.add(path)
    _digest(item["sha256"], f"{role} sha256")
    return item


def _pins(value: Any, split: str, used: set[str]) -> list[dict[str, str]]:
    names = {
        "recipe_sha256",
        "renderer_sha256",
        "observer_sha256",
        "validator_source_sha256",
        "capture_contract_sha256",
        "blender_build",
        "scene_sha256",
        "font",
        "local_files",
    }
    pins = _fields(value, names, "provenance pins")
    for key in (
        "recipe_sha256",
        "renderer_sha256",
        "observer_sha256",
        "validator_source_sha256",
        "capture_contract_sha256",
        "scene_sha256",
    ):
        _digest(pins[key], key)
    build = _fields(pins["blender_build"], {"version", "sha256"}, "Blender build pin")
    if type(build["version"]) is not str or not build["version"]:
        raise ValueError("Blender build version must be non-empty")
    _digest(build["sha256"], "Blender build sha256")
    font = _fields(
        pins["font"], {"source_url", "file_sha256", "license_id", "license_sha256"}, "font pin"
    )
    url = urlsplit(font["source_url"]) if type(font["source_url"]) is str else None
    if url is None or url.scheme != "https" or not url.netloc or font["license_id"] != "OFL-1.1":
        raise ValueError("font source and license pin are invalid")
    _digest(font["file_sha256"], "font file sha256")
    _digest(font["license_sha256"], "font license sha256")
    if type(pins["local_files"]) is not list:
        raise ValueError("local pin files must be a list")
    expected = {
        "recipe_sha256": pins["recipe_sha256"],
        "renderer_sha256": pins["renderer_sha256"],
        "observer_sha256": pins["observer_sha256"],
        "capture_contract_sha256": pins["capture_contract_sha256"],
        "scene_sha256": pins["scene_sha256"],
        "font_file": font["file_sha256"],
        "font_license": font["license_sha256"],
    }
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in pins["local_files"]:
        row = _fields(raw, {"pin", "path", "sha256"}, "local pin file")
        pin = row["pin"]
        if type(pin) is not str or pin not in expected or pin in seen:
            raise ValueError("local pin file has an unknown or duplicate provenance pin")
        if row["sha256"] != expected[pin]:
            raise ValueError(f"local {pin} pin does not match declared provenance")
        seen.add(pin)
        _ref(
            {"path": row["path"], "sha256": row["sha256"]},
            split,
            {"path", "sha256"},
            "local pin file",
            used,
        )
        rows.append({"path": row["path"], "sha256": row["sha256"]})
    missing = expected.keys() - seen
    if missing:
        raise ValueError("missing required local provenance pins: " + ", ".join(sorted(missing)))
    return rows


def _manifest_shape(value: Any) -> dict[str, Any]:
    manifest = _fields(value, {"schema_version", "provenance", "assignments", "pairs"}, "manifest")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("manifest schema_version must be integer 1")
    provenance = _fields(
        manifest["provenance"],
        {"synthetic_source", "license", "external_inputs", "employer_inputs"},
        "provenance",
    )
    if provenance != {
        "synthetic_source": "original_synthetic",
        "license": "CC0-1.0",
        "external_inputs": [],
        "employer_inputs": [],
    }:
        raise ValueError(
            "manifest provenance must be original synthetic CC0-1.0 with no "
            "external or employer inputs"
        )
    if (
        type(manifest["assignments"]) is not list
        or not manifest["assignments"]
        or type(manifest["pairs"]) is not list
        or not manifest["pairs"]
    ):
        raise ValueError("manifest assignments and pairs must be non-empty lists")
    for row in manifest["assignments"]:
        item = _fields(row, {"kind", "id", "split"}, "assignment")
        if item["kind"] not in ("product", "family"):
            raise ValueError("assignment kind must be product or family")
        _id(item["id"], "assignment ID")
        _split(item["split"])
    for row in manifest["pairs"]:
        item = _fields(
            row,
            {
                "pair_id",
                "product_id",
                "background_family_id",
                "split",
                "expected_words",
                "pins",
                "snapshot_bundle",
                "renders",
            },
            "pair",
        )
        for key in ("pair_id", "product_id", "background_family_id"):
            _id(item[key], key)
        _split(item["split"])
        if (
            type(item["expected_words"]) is not list
            or not item["expected_words"]
            or any(type(word) is not str or not word for word in item["expected_words"])
        ):
            raise ValueError("expected_words must be non-empty strings")
        if type(item["renders"]) is not list or len(item["renders"]) != 4:
            raise ValueError("each pair needs exactly four renders")
    return manifest
