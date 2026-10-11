"""Validation for captured controlled-studio scene state."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any
from urllib.parse import urlsplit

_MAX_JSON_DEPTH = 32
_MAX_JSON_COLLECTION = 100_000
_STATE_NAMES = ("source", "cream", "sage", "clay")
_DOMAINS = (
    "geometry",
    "materials",
    "camera",
    "lights",
    "world",
    "compositor",
    "settings",
    "label",
    "wall",
)
_REQUIRED_SETTINGS = {
    "engine": "CYCLES",
    "device": "CPU",
    "compositor_device": "CPU",
    "threads": 2,
    "threads_mode": "FIXED",
    "resolution": [512, 512],
    "samples": 96,
    "seed": 7321,
    "denoise": False,
    "adaptive": False,
    "color_depth": "8-bit PNG",
    "view_transform": "Standard",
}
_INTEGER_SETTINGS = {"threads", "samples", "seed"}
_BOOLEAN_SETTINGS = {"denoise", "adaptive"}
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _validate_json_value(value: Any, depth: int = 0, active: set[int] | None = None) -> None:
    if depth > _MAX_JSON_DEPTH:
        raise ValueError("snapshot exceeds JSON limits: maximum depth is 32")
    if active is None:
        active = set()
    if type(value) is dict:
        if id(value) in active:
            raise ValueError("snapshot contains a cycle")
        if len(value) > _MAX_JSON_COLLECTION:
            raise ValueError("snapshot exceeds JSON limits: object has too many entries")
        active.add(id(value))
        try:
            for key, child in value.items():
                if type(key) is not str:
                    raise ValueError("JSON object keys must be strings")
                _validate_json_value(child, depth + 1, active)
        finally:
            active.remove(id(value))
    elif type(value) is list:
        if id(value) in active:
            raise ValueError("snapshot contains a cycle")
        if len(value) > _MAX_JSON_COLLECTION:
            raise ValueError("snapshot exceeds JSON limits: array has too many entries")
        active.add(id(value))
        try:
            for child in value:
                _validate_json_value(child, depth + 1, active)
        finally:
            active.remove(id(value))
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError("snapshot contains an invalid JSON value: non-finite number")
    elif value is None or type(value) in (str, bool, int):
        return
    else:
        raise ValueError("snapshot contains an invalid JSON value")


def snapshot_sha256(snapshot: dict[str, Any]) -> str:
    """Return a canonical SHA-256 digest for a finite JSON snapshot."""
    if type(snapshot) is not dict:
        raise ValueError("snapshot must be a JSON object")
    _validate_json_value(snapshot)
    return hashlib.sha256(_canonical_bytes(snapshot)).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("snapshot strings must be valid UTF-8") from exc


def _without_wall_color(snapshot: dict[str, Any]) -> dict[str, Any]:
    invariant = dict(snapshot)
    wall = invariant.get("wall")
    if type(wall) is dict:
        invariant_wall = dict(wall)
        invariant_wall.pop("base_color_rgba", None)
        invariant["wall"] = invariant_wall
    return invariant


def _require_fields(value: Any, fields: tuple[str, ...], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError(f"{label} fields are missing or unknown")
    return value


def _validate_vector(value: Any, size: int, label: str) -> None:
    if (
        type(value) is not list
        or len(value) != size
        or any(type(item) not in (int, float) or not math.isfinite(item) for item in value)
    ):
        raise ValueError(f"{label} must contain {size} finite numbers")


def _validate_geometry(snapshot: dict[str, Any], name: str) -> None:
    geometry = snapshot["geometry"]
    if type(geometry) is not dict or not geometry:
        raise ValueError(f"{name} geometry must be a non-empty object map")
    for object_name, raw_object in geometry.items():
        label = f"{name} geometry object {object_name!r}"
        if type(object_name) is not str or not object_name:
            raise ValueError(f"{name} geometry object names must be non-empty")
        item = _require_fields(
            raw_object,
            ("mesh_sha256", "matrix_world", "material_names", "pass_index", "visibility"),
            label,
        )
        if type(item["mesh_sha256"]) is not str or not _DIGEST_PATTERN.fullmatch(
            item["mesh_sha256"]
        ):
            raise ValueError(f"{label} mesh_sha256 must be a lowercase SHA-256 digest")
        _validate_vector(item["matrix_world"], 16, f"{label} matrix_world")
        material_names = item["material_names"]
        if (
            type(material_names) is not list
            or not material_names
            or any(type(material) is not str or not material for material in material_names)
        ):
            raise ValueError(f"{label} material_names must be a non-empty list of names")
        pass_index = item["pass_index"]
        if type(pass_index) is not int or pass_index < 0:
            raise ValueError(f"{label} pass_index must be a non-negative integer")
        visibility = _require_fields(
            item["visibility"],
            ("hide_render", "visible_camera", "visible_shadow"),
            f"{label} visibility",
        )
        if any(type(value) is not bool for value in visibility.values()):
            raise ValueError(f"{label} visibility values must be booleans")


def _validate_node_tree(value: Any, label: str) -> None:
    tree = _require_fields(value, ("nodes", "links"), label)
    nodes, links = tree["nodes"], tree["links"]
    if type(nodes) is not list or not nodes or any(type(node) is not dict for node in nodes):
        raise ValueError(f"{label} nodes must be a non-empty list of objects")
    if type(links) is not list or any(type(link) is not dict for link in links):
        raise ValueError(f"{label} links must be a list of objects")


def _marker_count(value: Any) -> int:
    if value == "wall_base_color_parameter":
        return 1
    if type(value) is dict:
        return sum(_marker_count(child) for child in value.values())
    if type(value) is list:
        return sum(_marker_count(child) for child in value)
    return 0


def _validate_materials(snapshot: dict[str, Any], name: str) -> None:
    materials = snapshot["materials"]
    if type(materials) is not dict or not materials:
        raise ValueError(f"{name} materials must be a non-empty object map")
    for material_name, raw_tree in materials.items():
        if type(material_name) is not str or not material_name:
            raise ValueError(f"{name} materials names must be non-empty")
        _validate_node_tree(raw_tree, f"{name} material {material_name}")


def _validate_parameters(value: Any, label: str) -> None:
    if type(value) is not dict or not value:
        raise ValueError(f"{label} parameters must be a non-empty object")


def _validate_camera_and_lights(snapshot: dict[str, Any], name: str) -> None:
    camera = _require_fields(
        snapshot["camera"], ("name", "matrix_world", "parameters"), f"{name} camera"
    )
    if type(camera["name"]) is not str or not camera["name"]:
        raise ValueError(f"{name} camera name must be non-empty")
    _validate_vector(camera["matrix_world"], 16, f"{name} camera matrix_world")
    _validate_parameters(camera["parameters"], f"{name} camera")
    lights = snapshot["lights"]
    if type(lights) is not dict or not lights:
        raise ValueError(f"{name} lights must be a non-empty object map")
    for light_name, raw_light in lights.items():
        label = f"{name} light {light_name!r}"
        if type(light_name) is not str or not light_name:
            raise ValueError(f"{name} light names must be non-empty")
        light = _require_fields(raw_light, ("matrix_world", "parameters"), label)
        _validate_vector(light["matrix_world"], 16, f"{label} matrix_world")
        _validate_parameters(light["parameters"], label)


def _validate_settings(snapshot: dict[str, Any], name: str) -> None:
    settings = snapshot["settings"]
    if type(settings) is not dict or not settings:
        raise ValueError(f"{name} settings must be a non-empty object")
    missing = set(_REQUIRED_SETTINGS) - set(settings)
    if missing:
        raise ValueError(f"{name} settings missing required fields: {', '.join(sorted(missing))}")
    for field, expected in _REQUIRED_SETTINGS.items():
        value = settings[field]
        if field in _INTEGER_SETTINGS:
            if type(value) is not int:
                raise ValueError(f"{name} settings {field} must be an integer")
        elif field in _BOOLEAN_SETTINGS:
            if type(value) is not bool:
                raise ValueError(f"{name} settings {field} must be a boolean")
        elif field == "resolution":
            if (
                type(value) is not list
                or len(value) != 2
                or any(type(side) is not int for side in value)
            ):
                raise ValueError(f"{name} settings resolution must contain two integers")
        elif type(value) is not str:
            raise ValueError(f"{name} settings {field} must be a string")
        if value != expected:
            raise ValueError(f"{name} settings {field} must equal {expected!r}")


def _validate_label(snapshot: dict[str, Any], name: str) -> None:
    label = _require_fields(snapshot["label"], ("words", "font"), f"{name} label")
    words = label["words"]
    if (
        type(words) is not list
        or not words
        or any(type(word) is not str or not word for word in words)
    ):
        raise ValueError(f"{name} label words must be a non-empty list of non-empty strings")
    font = _require_fields(
        label["font"],
        ("source_url", "file_sha256", "license_id", "license_sha256"),
        f"{name} label font",
    )
    source_url = font["source_url"]
    try:
        secure_source = (
            type(source_url) is str
            and urlsplit(source_url).scheme == "https"
            and bool(urlsplit(source_url).hostname)
        )
    except ValueError:
        secure_source = False
    if not secure_source:
        raise ValueError(f"{name} label font source_url must be an HTTPS URL")
    for field in ("file_sha256", "license_sha256"):
        if type(font[field]) is not str or not _DIGEST_PATTERN.fullmatch(font[field]):
            raise ValueError(f"{name} label font {field} must be a lowercase SHA-256 digest")
    if font["license_id"] != "OFL-1.1":
        raise ValueError(f"{name} label font license_id must be OFL-1.1")


def _validate_wall_marker(snapshot: dict[str, Any], name: str) -> None:
    wall_material = snapshot["wall"]["material_name"]
    for material_name, tree in snapshot["materials"].items():
        count = _marker_count(tree)
        if material_name == wall_material:
            if count != 1:
                raise ValueError(
                    f"{name} materials wall Base Color marker must appear exactly once"
                )
        elif count:
            raise ValueError(f"{name} materials wall Base Color marker appears in another material")


def _validate_wall(snapshot: dict[str, Any], name: str) -> None:
    wall = _require_fields(
        snapshot["wall"],
        ("material_name", "node_name", "base_color_rgba"),
        f"{name} wall",
    )
    material_name = wall["material_name"]
    node_name = wall["node_name"]
    materials = snapshot["materials"]
    if type(material_name) is not str or not material_name:
        raise ValueError(f"{name} wall material_name must be non-empty")
    if type(node_name) is not str or not node_name:
        raise ValueError(f"{name} wall node_name must be non-empty")
    if type(materials) is not dict or material_name not in materials:
        raise ValueError(f"{name} wall material_name must name a declared material")
    rgba = wall["base_color_rgba"]
    valid_numbers = (
        type(rgba) is list
        and len(rgba) == 4
        and all(type(value) in (int, float) and math.isfinite(value) for value in rgba)
    )
    if not valid_numbers or any(not 0.0 <= value <= 1.0 for value in rgba[:3]) or rgba[3] != 1.0:
        raise ValueError(f"{name} wall base_color_rgba must have RGB in [0, 1] and alpha 1.0")


def validate_studio_states(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate four captured variants without approving rendering or training."""
    if type(records) is not list or len(records) != len(_STATE_NAMES):
        raise ValueError("records must contain exactly source, cream, sage, and clay")
    rows: dict[str, dict[str, Any]] = {}
    for row in records:
        if type(row) is not dict or set(row) != {"name", "snapshot", "snapshot_sha256"}:
            raise ValueError("studio-state row fields are missing or unknown")
        name = row["name"]
        if type(name) is not str or name not in _STATE_NAMES:
            raise ValueError(f"unknown studio-state name: {name!r}")
        if name in rows:
            raise ValueError(f"duplicate studio-state name: {name!r}")
        declared = row["snapshot_sha256"]
        if type(declared) is not str or not _DIGEST_PATTERN.fullmatch(declared):
            raise ValueError(f"{name} snapshot_sha256 must be a lowercase SHA-256 digest")
        snapshot = row["snapshot"]
        if type(snapshot) is not dict:
            raise ValueError(f"{name} snapshot must be an object")
        if set(snapshot) != set(_DOMAINS):
            raise ValueError(f"{name} snapshot fields are missing or unknown")
        if snapshot_sha256(snapshot) != declared:
            raise ValueError(f"{name} snapshot_sha256 does not match its snapshot")
        _validate_geometry(snapshot, name)
        _validate_materials(snapshot, name)
        _validate_camera_and_lights(snapshot, name)
        _validate_node_tree(snapshot["world"], f"{name} world")
        _validate_node_tree(snapshot["compositor"], f"{name} compositor")
        _validate_settings(snapshot, name)
        _validate_label(snapshot, name)
        _validate_wall(snapshot, name)
        _validate_wall_marker(snapshot, name)
        rows[name] = row
    if set(rows) != set(_STATE_NAMES):
        raise ValueError("studio-state rows must contain source, cream, sage, and clay")

    source = rows["source"]["snapshot"]
    for name in _STATE_NAMES[1:]:
        target = rows[name]["snapshot"]
        if target["wall"]["base_color_rgba"][:3] == source["wall"]["base_color_rgba"][:3]:
            raise ValueError(f"{name} wall RGB must differ from source")
        for domain in _DOMAINS:
            left = _without_wall_color(source)[domain] if domain == "wall" else source[domain]
            right = _without_wall_color(target)[domain] if domain == "wall" else target[domain]
            if _canonical_bytes(left) != _canonical_bytes(right):
                raise ValueError(f"{domain} changed in {name}")
    invariant = _without_wall_color(source)
    return {
        "status": "validated_captured_state",
        "source_snapshot_sha256": rows["source"]["snapshot_sha256"],
        "names": list(_STATE_NAMES),
        "invariant_sha256": hashlib.sha256(_canonical_bytes(invariant)).hexdigest(),
    }
