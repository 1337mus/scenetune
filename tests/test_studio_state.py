from __future__ import annotations

import hashlib
import math
from copy import deepcopy
from typing import Any

import pytest

from product_scene_editor.studio_state import snapshot_sha256, validate_studio_states


def test_snapshot_sha256_is_stable_across_mapping_insertion_order() -> None:
    first = {"z": [True, None], "a": 1}
    second = {"a": 1, "z": [True, None]}

    assert snapshot_sha256(first) == snapshot_sha256(second)
    assert snapshot_sha256(first) == hashlib.sha256(b'{"a":1,"z":[true,null]}').hexdigest()


def test_snapshot_sha256_rejects_non_string_mapping_keys() -> None:
    with pytest.raises(ValueError, match="JSON object keys must be strings"):
        snapshot_sha256({"child": {1: "value"}})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid",
    [object(), ("tuple",), math.nan, math.inf, -math.inf],
    ids=["object", "tuple", "nan", "positive-infinity", "negative-infinity"],
)
def test_snapshot_sha256_rejects_non_finite_or_unsupported_values(invalid: Any) -> None:
    with pytest.raises(ValueError, match="snapshot contains an invalid JSON value"):
        snapshot_sha256({"value": invalid})


def _nested_snapshot(value_depth: int) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"leaf": None}
    for _ in range(value_depth - 1):
        snapshot = {"child": snapshot}
    return snapshot


@pytest.mark.parametrize(
    "snapshot",
    [
        _nested_snapshot(33),
        {"items": [None] * 100_001},
        {f"key-{index}": None for index in range(100_001)},
    ],
    ids=["depth", "list-size", "mapping-size"],
)
def test_snapshot_sha256_rejects_json_structures_over_limits(snapshot: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="snapshot exceeds JSON limits"):
        snapshot_sha256(snapshot)


def test_snapshot_sha256_accepts_maximum_supported_json_depth() -> None:
    assert len(snapshot_sha256(_nested_snapshot(32))) == 64


def test_snapshot_sha256_rejects_cyclic_python_containers() -> None:
    snapshot: dict[str, Any] = {}
    snapshot["self"] = snapshot

    with pytest.raises(ValueError, match="snapshot contains a cycle"):
        snapshot_sha256(snapshot)


def test_snapshot_sha256_requires_object_root() -> None:
    with pytest.raises(ValueError, match="snapshot must be a JSON object"):
        snapshot_sha256(["not-an-object"])  # type: ignore[arg-type]


def test_snapshot_sha256_rejects_strings_that_are_not_valid_utf8() -> None:
    with pytest.raises(ValueError, match="snapshot strings must be valid UTF-8"):
        snapshot_sha256({"text": "\ud800"})


def _valid_snapshot(wall_rgb: list[float]) -> dict[str, Any]:
    identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    return {
        "geometry": {
            "Carton": {
                "mesh_sha256": "a" * 64,
                "matrix_world": identity.copy(),
                "material_names": ["carton_material"],
                "pass_index": 1,
                "visibility": {
                    "hide_render": False,
                    "visible_camera": True,
                    "visible_shadow": True,
                },
            }
        },
        "materials": {
            "carton_material": {
                "nodes": [{"type": "Principled", "base_color": [0.2, 0.3, 0.4]}],
                "links": [],
            },
            "wall_material": {
                "nodes": [
                    {
                        "type": "Principled",
                        "base_color": "wall_base_color_parameter",
                        "roughness": 0.8,
                    }
                ],
                "links": [],
            },
        },
        "camera": {
            "name": "Camera",
            "matrix_world": identity.copy(),
            "parameters": {"type": "ORTHO", "ortho_scale": 5.0},
        },
        "lights": {
            "Key": {
                "matrix_world": identity.copy(),
                "parameters": {"type": "AREA", "energy": 1.0},
            }
        },
        "world": {"nodes": [{"type": "Background", "strength": 0.5}], "links": []},
        "compositor": {
            "nodes": [{"type": "Composite", "mask_output": {"destination": "fixed"}}],
            "links": [],
        },
        "settings": {
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
        },
        "label": {
            "words": ["SORA", "BOTANICAL", "SMALL BATCH"],
            "font": {
                "source_url": "https://example.test/font.otf",
                "file_sha256": "b" * 64,
                "license_id": "OFL-1.1",
                "license_sha256": "c" * 64,
            },
        },
        "wall": {
            "material_name": "wall_material",
            "node_name": "Principled",
            "base_color_rgba": [*wall_rgb, 1.0],
        },
    }


def _valid_records() -> list[dict[str, Any]]:
    variants = [
        ("source", [0.80, 0.80, 0.80]),
        ("cream", [0.72, 0.67, 0.57]),
        ("sage", [0.45, 0.58, 0.42]),
        ("clay", [0.68, 0.40, 0.32]),
    ]
    records = []
    for name, rgb in variants:
        snapshot = _valid_snapshot(rgb)
        records.append(
            {"name": name, "snapshot": snapshot, "snapshot_sha256": snapshot_sha256(snapshot)}
        )
    return records


def _rehash_records(records: list[dict[str, Any]]) -> None:
    for row in records:
        row["snapshot_sha256"] = snapshot_sha256(row["snapshot"])


_DELETE = object()


def _set_path(root: Any, path: tuple[str | int, ...], value: Any) -> None:
    parent = root
    for key in path[:-1]:
        parent = parent[key]
    if value is _DELETE:
        del parent[path[-1]]
    else:
        parent[path[-1]] = value


def test_validate_studio_states_accepts_four_captured_wall_variants() -> None:
    records = _valid_records()

    result = validate_studio_states(records)

    assert result["status"] == "validated_captured_state"
    assert result["source_snapshot_sha256"] == records[0]["snapshot_sha256"]
    assert result["names"] == ["source", "cream", "sage", "clay"]
    assert len(result["invariant_sha256"]) == 64


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "missing",
        "duplicate",
        "unknown",
        "row-extra-field",
        "row-missing-field",
        "bad-hash",
    ],
)
def test_validate_studio_states_rejects_invalid_comparison_rows(case: str) -> None:
    records = _valid_records()
    if case == "empty":
        records = []
        expected = "exactly"
    elif case == "missing":
        records.pop()
        expected = "exactly"
    elif case == "duplicate":
        records[3]["name"] = "cream"
        expected = "duplicate"
    elif case == "unknown":
        records[3]["name"] = "plum"
        expected = "unknown"
    elif case == "row-extra-field":
        records[0]["approval"] = True
        expected = "row fields"
    elif case == "row-missing-field":
        del records[0]["snapshot_sha256"]
        expected = "row fields"
    else:
        records[0]["snapshot_sha256"] = "A" * 64
        expected = "lowercase SHA-256"

    with pytest.raises(ValueError, match=expected):
        validate_studio_states(records)


def test_validate_studio_states_accepts_reordered_rows_without_mutating_them() -> None:
    records = _valid_records()[::-1]
    original = deepcopy(records)

    result = validate_studio_states(records)

    assert result["names"] == ["source", "cream", "sage", "clay"]
    assert records == original


@pytest.mark.parametrize("extra", [False, True], ids=["missing-domain", "unknown-domain"])
def test_validate_studio_states_requires_exact_snapshot_top_level_fields(extra: bool) -> None:
    records = _valid_records()
    for row in records:
        if extra:
            row["snapshot"]["unexpected"] = {}
        else:
            del row["snapshot"]["world"]
    _rehash_records(records)

    with pytest.raises(ValueError, match="snapshot fields are missing or unknown"):
        validate_studio_states(records)


def test_validate_studio_states_rejects_mismatched_snapshot_digest() -> None:
    records = _valid_records()
    records[0]["snapshot_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="source snapshot_sha256 does not match its snapshot"):
        validate_studio_states(records)


def test_validate_studio_states_rejects_rehashed_geometry_change() -> None:
    records = _valid_records()
    records[1]["snapshot"]["geometry"]["Carton"]["mesh_sha256"] = "d" * 64
    _rehash_records(records)

    with pytest.raises(ValueError, match="geometry changed in cream"):
        validate_studio_states(records)


@pytest.mark.parametrize(
    ("domain", "path", "replacement"),
    [
        ("geometry", ("geometry", "Carton", "mesh_sha256"), "d" * 64),
        ("materials", ("materials", "carton_material", "nodes", 0, "base_color", 0), 0.25),
        ("camera", ("camera", "parameters", "ortho_scale"), 5.25),
        ("lights", ("lights", "Key", "parameters", "energy"), 1.25),
        ("world", ("world", "nodes", 0, "strength"), 0.75),
        ("compositor", ("compositor", "nodes", 0, "type"), "DifferentComposite"),
        ("settings", ("settings", "extra_render_setting"), 1),
        ("label", ("label", "words", 0), "SORA LAB"),
        ("wall", ("wall", "node_name"), "Principled.001"),
    ],
)
def test_validate_studio_states_rejects_each_rehashed_invariant_domain_change(
    domain: str, path: tuple[str | int, ...], replacement: Any
) -> None:
    records = _valid_records()
    changed: Any = records[1]["snapshot"]
    for key in path[:-1]:
        changed = changed[key]
    changed[path[-1]] = replacement
    _rehash_records(records)

    with pytest.raises(ValueError, match=f"{domain} changed in cream"):
        validate_studio_states(records)


@pytest.mark.parametrize(
    "rgba",
    [
        [True, 0.67, 0.57, 1.0],
        [1.01, 0.67, 0.57, 1.0],
        [0.72, 0.67, 0.57, 0.9],
        [0.72, 0.67, 0.57],
    ],
    ids=["boolean-channel", "out-of-range", "alpha-not-one", "wrong-length"],
)
def test_validate_studio_states_rejects_invalid_wall_rgba(rgba: list[Any]) -> None:
    records = _valid_records()
    records[1]["snapshot"]["wall"]["base_color_rgba"] = rgba
    _rehash_records(records)

    with pytest.raises(ValueError, match="cream wall base_color_rgba"):
        validate_studio_states(records)


def test_validate_studio_states_requires_each_wall_color_to_differ_from_source() -> None:
    records = _valid_records()
    records[1]["snapshot"]["wall"]["base_color_rgba"] = [0.8, 0.8, 0.8, 1.0]
    _rehash_records(records)

    with pytest.raises(ValueError, match="cream wall RGB must differ from source"):
        validate_studio_states(records)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("geometry", "Carton", "mesh_sha256"), "not-a-digest"),
        (("geometry", "Carton", "matrix_world"), [0.0] * 15),
        (("geometry", "Carton", "material_names"), []),
        (("geometry", "Carton", "pass_index"), True),
        (("geometry", "Carton", "visibility", "hide_render"), _DELETE),
        (("geometry", "Carton", "visibility", "visible_shadow"), "yes"),
        (("geometry",), {}),
    ],
)
def test_validate_studio_states_rejects_malformed_shared_geometry(
    path: tuple[str | int, ...], replacement: Any
) -> None:
    records = _valid_records()
    for row in records:
        _set_path(row["snapshot"], path, replacement)
    _rehash_records(records)

    with pytest.raises(ValueError, match="geometry"):
        validate_studio_states(records)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("materials", "carton_material", "nodes", 0, "base_color"), "wall_base_color_parameter"),
        (("materials", "wall_material", "nodes", 0, "base_color"), "white"),
        (
            ("materials", "wall_material", "nodes", 0, "second_base_color"),
            "wall_base_color_parameter",
        ),
        (
            ("materials", "carton_material", "links"),
            [{"value": "wall_base_color_parameter"}],
        ),
    ],
    ids=[
        "marker-in-non-wall-material",
        "missing-wall-marker",
        "multiple-wall-markers",
        "marker-in-non-wall-link",
    ],
)
def test_validate_studio_states_rejects_bad_wall_base_color_marker_binding(
    path: tuple[str | int, ...], replacement: Any
) -> None:
    records = _valid_records()
    for row in records:
        _set_path(row["snapshot"], path, replacement)
    _rehash_records(records)

    with pytest.raises(ValueError, match="materials wall Base Color marker"):
        validate_studio_states(records)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("materials",), {}),
        (("materials", "wall_material", "nodes"), []),
        (("materials", "carton_material", "links"), ["not-a-link"]),
        (("materials", "wall_material", "unknown"), []),
    ],
)
def test_validate_studio_states_rejects_malformed_shared_material_trees(
    path: tuple[str | int, ...], replacement: Any
) -> None:
    records = _valid_records()
    for row in records:
        _set_path(row["snapshot"], path, replacement)
    _rehash_records(records)

    with pytest.raises(ValueError, match="material"):
        validate_studio_states(records)


@pytest.mark.parametrize(
    ("domain", "path", "replacement"),
    [
        ("camera", ("camera", "name"), ""),
        ("camera", ("camera", "matrix_world"), [0.0] * 15),
        ("camera", ("camera", "parameters"), {}),
        ("light", ("lights",), {}),
        ("light", ("lights", "Key", "matrix_world"), [0.0] * 15),
        ("light", ("lights", "Key", "parameters"), {}),
        ("world", ("world", "nodes"), []),
        ("world", ("world", "links"), ["not-a-link"]),
        ("compositor", ("compositor", "nodes"), []),
        ("compositor", ("compositor", "links"), ["not-a-link"]),
    ],
)
def test_validate_studio_states_rejects_malformed_camera_lighting_and_node_trees(
    domain: str, path: tuple[str | int, ...], replacement: Any
) -> None:
    records = _valid_records()
    for row in records:
        _set_path(row["snapshot"], path, replacement)
    _rehash_records(records)

    with pytest.raises(ValueError, match=domain):
        validate_studio_states(records)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("engine", "BLENDER_EEVEE_NEXT"),
        ("device", "GPU"),
        ("compositor_device", "GPU"),
        ("threads", True),
        ("threads_mode", "AUTO"),
        ("resolution", [512, 256]),
        ("samples", 96.0),
        ("seed", True),
        ("denoise", 1),
        ("adaptive", 1),
        ("color_depth", "16-bit PNG"),
        ("view_transform", "AgX"),
    ],
)
def test_validate_studio_states_requires_exact_render_settings(
    field: str, replacement: Any
) -> None:
    records = _valid_records()
    for row in records:
        row["snapshot"]["settings"][field] = replacement
    _rehash_records(records)

    with pytest.raises(ValueError, match=f"settings {field}"):
        validate_studio_states(records)


def test_validate_studio_states_rejects_missing_required_render_setting() -> None:
    records = _valid_records()
    for row in records:
        del row["snapshot"]["settings"]["seed"]
    _rehash_records(records)

    with pytest.raises(ValueError, match="settings.*seed"):
        validate_studio_states(records)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("label", "words"), []),
        (("label", "words", 0), ""),
        (("label", "font", "source_url"), "http://example.test/font.otf"),
        (("label", "font", "file_sha256"), "B" * 64),
        (("label", "font", "license_id"), "CC0-1.0"),
        (("label", "font", "license_sha256"), "short"),
        (("label", "font", "unexpected"), "value"),
        (("label", "extra"), "value"),
    ],
    ids=[
        "empty-words",
        "empty-word",
        "http-font",
        "uppercase-font-hash",
        "wrong-license",
        "bad-license-hash",
        "extra-font-field",
        "extra-label-field",
    ],
)
def test_validate_studio_states_rejects_malformed_shared_label(
    path: tuple[str | int, ...], replacement: Any
) -> None:
    records = _valid_records()
    for row in records:
        _set_path(row["snapshot"], path, replacement)
    _rehash_records(records)

    with pytest.raises(ValueError, match="label"):
        validate_studio_states(records)
