"""Focused tests for the opt-in Controlled3D validator."""

import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from itertools import permutations
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw

from product_scene_editor import controlled3d_assets as assets
from product_scene_editor.controlled3d import (
    ValidatedControlled3D,
    _pins,
    draft_manifest,
    export_train_only,
    validate_controlled3d,
)
from product_scene_editor.studio_state import snapshot_sha256


def test_draft_manifest_rejects_duplicate_json_keys(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        draft_manifest(manifest)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ('{"schema_version":1,"value":NaN}', "non-finite JSON number"),
        ("[]", "manifest root must be an object"),
    ],
)
def test_draft_manifest_rejects_non_strict_or_wrong_root_json(tmp_path, raw, message) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        draft_manifest(manifest)


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _snapshot(wall_rgb: list[float]) -> dict[str, Any]:
    identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
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
            "Key": {"matrix_world": identity.copy(), "parameters": {"type": "AREA", "energy": 1.0}}
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
            "words": ["LABEL", "FOR", "TEST"],
            "font": {
                "source_url": "https://example.test/font.otf",
                "file_sha256": "a" * 64,
                "license_id": "OFL-1.1",
                "license_sha256": "b" * 64,
            },
        },
        "wall": {
            "material_name": "wall_material",
            "node_name": "Principled",
            "base_color_rgba": [*wall_rgb, 1.0],
        },
    }


def _png(image: Image.Image) -> bytes:
    from io import BytesIO

    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def _write(root: Path, relative: str, raw: bytes) -> dict[str, str]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {"path": relative, "sha256": _digest(raw)}


def _make_case(
    tmp_path: Path, splits: tuple[str, ...] = ("train",)
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    root = tmp_path / "dataset"
    root.mkdir(parents=True)
    products: list[dict[str, str]] = []
    families: list[dict[str, Any]] = []
    registry_pairs: list[dict[str, Any]] = []
    assignments: list[dict[str, str]] = []
    pairs: list[dict[str, Any]] = []
    wall_colors = ([0.8, 0.8, 0.8], [0.72, 0.67, 0.57], [0.45, 0.58, 0.42], [0.68, 0.4, 0.32])
    variant_backgrounds = [(220, 220, 220), (238, 225, 205), (105, 140, 105), (175, 105, 80)]
    for split in splits:
        product_id = f"demo-product-{split}"
        family_id = f"demo-family-{split}"
        alias_id = f"demo-family-alias-{split}"
        pair_id = f"demo-pair-{split}"
        words = [f"{split.upper()}", "LABEL", "TEST"]
        products.append({"product_id": product_id, "split": split})
        families.append({"canonical_id": family_id, "split": split, "aliases": [alias_id]})
        assignments.extend(
            [
                {"kind": "product", "id": product_id, "split": split},
                {"kind": "family", "id": family_id, "split": split},
            ]
        )
        recipe_raw = f"recipe for {split}".encode()
        recipe_path = f"{split}/pins/recipe.bin"
        recipe_ref = _write(root, recipe_path, recipe_raw)
        local_pin_refs: dict[str, dict[str, str]] = {}
        for pin in (
            "renderer_sha256",
            "observer_sha256",
            "capture_contract_sha256",
            "scene_sha256",
            "font_file",
            "font_license",
        ):
            local_pin_refs[pin] = _write(root, f"{split}/pins/{pin}.bin", f"{split} {pin}".encode())
        studio_source = Path(__file__).parents[1] / "src/product_scene_editor/studio_state.py"
        studio_sha = _digest(studio_source.read_bytes())
        pins = {
            "recipe_sha256": recipe_ref["sha256"],
            "renderer_sha256": local_pin_refs["renderer_sha256"]["sha256"],
            "observer_sha256": local_pin_refs["observer_sha256"]["sha256"],
            "validator_source_sha256": studio_sha,
            "capture_contract_sha256": local_pin_refs["capture_contract_sha256"]["sha256"],
            "blender_build": {"version": "4.3.2", "sha256": "f" * 64},
            "scene_sha256": local_pin_refs["scene_sha256"]["sha256"],
            "font": {
                "source_url": "https://example.test/font.otf",
                "file_sha256": local_pin_refs["font_file"]["sha256"],
                "license_id": "OFL-1.1",
                "license_sha256": local_pin_refs["font_license"]["sha256"],
            },
            "local_files": [
                {"pin": "recipe_sha256", "path": recipe_path, "sha256": recipe_ref["sha256"]},
                *[
                    {"pin": pin, "path": f"{split}/pins/{pin}.bin", **ref}
                    for pin, ref in local_pin_refs.items()
                ],
            ],
        }
        registry_pairs.append(
            {
                "pair_id": pair_id,
                "product_id": product_id,
                "background_family_id": family_id,
                "split": split,
                "expected_words": words,
                "pins": pins,
            }
        )
        records: list[dict[str, Any]] = []
        for name, wall_rgb in zip(("source", "cream", "sage", "clay"), wall_colors, strict=True):
            snapshot = _snapshot(wall_rgb)
            snapshot["label"]["words"] = words
            snapshot["label"]["font"] = deepcopy(pins["font"])
            records.append(
                {"name": name, "snapshot": snapshot, "snapshot_sha256": snapshot_sha256(snapshot)}
            )
        snapshot_raw = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
        snapshot_ref = _write(root, f"{split}/states/{pair_id}.json", snapshot_raw)
        renders: list[dict[str, Any]] = []
        product_colors = {
            "train": (190, 35, 45),
            "dev": (40, 180, 80),
            "sealed_test": (45, 40, 200),
        }
        background_shift = {"train": 0, "dev": 18, "sealed_test": -18}[split]
        for name, background in zip(
            ("source", "cream", "sage", "clay"), variant_backgrounds, strict=True
        ):
            product_color = product_colors[split]
            background = tuple(
                max(0, min(255, channel + background_shift)) for channel in background
            )
            image = Image.new("RGB", (512, 512), background)
            ImageDraw.Draw(image).rectangle((180, 170, 329, 349), fill=product_color)
            mask = Image.new("L", (512, 512), 0)
            ImageDraw.Draw(mask).rectangle((180, 170, 329, 349), fill=255)
            image_ref = _write(root, f"{split}/images/{pair_id}-{name}.png", _png(image))
            mask_raw = _png(mask)
            mask_ref = _write(root, f"{split}/masks/{pair_id}-{name}.png", mask_raw)
            renders.append(
                {
                    "name": name,
                    "image": image_ref,
                    "mask": {**mask_ref, "decoded_sha256": _digest(mask.tobytes())},
                    "bbox": [180, 170, 330, 350],
                }
            )
        pairs.append(
            {
                "pair_id": pair_id,
                "product_id": product_id,
                "background_family_id": family_id,
                "split": split,
                "expected_words": words,
                "pins": pins,
                "snapshot_bundle": snapshot_ref,
                "renders": renders,
            }
        )
    registry = {
        "schema_version": 1,
        "products": products,
        "families": families,
        "pairs": registry_pairs,
    }
    manifest = {
        "schema_version": 1,
        "provenance": {
            "synthetic_source": "original_synthetic",
            "license": "CC0-1.0",
            "external_inputs": [],
            "employer_inputs": [],
        },
        "assignments": assignments,
        "pairs": pairs,
    }
    return root, manifest, registry


def _registry_bytes(registry: dict[str, Any]) -> bytes:
    return json.dumps(registry, sort_keys=True, separators=(",", ":")).encode()


def _validate(
    root: Path, manifest: dict[str, Any], registry: dict[str, Any]
) -> ValidatedControlled3D:
    registry_bytes = _registry_bytes(registry)
    return validate_controlled3d(
        root, manifest, reviewed_registry=registry_bytes, registry_sha256=_digest(registry_bytes)
    )


def test_validate_controlled3d_accepts_pinned_cpu_pair_and_reports_rgb_changes(tmp_path) -> None:
    root, manifest, registry = _make_case(tmp_path)

    result = _validate(root, manifest, registry)

    metrics = json.loads(result.metrics_json)
    assert isinstance(result, ValidatedControlled3D)
    assert len(result.registry_sha256) == 64
    assert metrics["pairs"]["demo-pair-train"]["rgb_changes"]["cream"]["background"] > 0
    assert metrics["pairs"]["demo-pair-train"]["rgb_changes"]["cream"]["product"] == 0


_DENIED_PRODUCTS = [
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
]
_DENIED_PAIR_IDS = ["OFF-R1-003", "OFF-R1-005", "OFF-R1-006"]
_DENIED_FAMILIES = [
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
]


@pytest.mark.parametrize("split", ["train", "dev", "sealed_test"])
@pytest.mark.parametrize("identifier", _DENIED_PRODUCTS)
def test_fixed_product_exclusions_apply_to_every_pool(tmp_path, split, identifier) -> None:
    root, manifest, registry = _make_case(tmp_path, (split,))
    registry["products"][0]["product_id"] = identifier
    registry["pairs"][0]["product_id"] = identifier
    manifest["assignments"][0]["id"] = identifier
    manifest["pairs"][0]["product_id"] = identifier

    with pytest.raises(ValueError, match="excluded product"):
        _validate(root, manifest, registry)


@pytest.mark.parametrize("split", ["train", "dev", "sealed_test"])
@pytest.mark.parametrize("identifier", _DENIED_PAIR_IDS)
def test_fixed_pair_id_exclusions_apply_to_every_pool(tmp_path, split, identifier) -> None:
    root, manifest, registry = _make_case(tmp_path, (split,))
    registry["pairs"][0]["pair_id"] = identifier
    manifest["pairs"][0]["pair_id"] = identifier

    with pytest.raises(ValueError, match="excluded pair"):
        _validate(root, manifest, registry)


def test_provenance_pins_require_all_local_files(tmp_path) -> None:
    _, _, registry = _make_case(tmp_path)
    pins = deepcopy(registry["pairs"][0]["pins"])
    pins["local_files"] = [row for row in pins["local_files"] if row["pin"] != "font_license"]

    with pytest.raises(ValueError, match="missing required local provenance pins:.*font_license"):
        _pins(pins, "train", set())


def test_provenance_pins_accept_and_verify_font_license_file(tmp_path) -> None:
    _, _, registry = _make_case(tmp_path)
    pins = deepcopy(registry["pairs"][0]["pins"])
    rows = _pins(pins, "train", set())
    assert len(rows) == 7
    license_path = next(row["path"] for row in pins["local_files"] if row["pin"] == "font_license")
    assert any(row["path"] == license_path for row in rows)

    mismatch = deepcopy(pins)
    license_row = next(row for row in mismatch["local_files"] if row["pin"] == "font_license")
    license_row["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="font_license pin does not match"):
        _pins(mismatch, "train", set())


@pytest.mark.parametrize("split", ["train", "dev", "sealed_test"])
@pytest.mark.parametrize("identifier", _DENIED_FAMILIES)
def test_fixed_family_and_alias_exclusions_apply_to_every_pool(tmp_path, split, identifier) -> None:
    root, manifest, registry = _make_case(tmp_path, (split,))
    if identifier == "controlled3d-neutral-wall-support-001":
        registry["families"][0]["canonical_id"] = identifier
        registry["pairs"][0]["background_family_id"] = identifier
        manifest["assignments"][1]["id"] = identifier
        manifest["pairs"][0]["background_family_id"] = identifier
    else:
        registry["families"][0]["aliases"] = [identifier]
        manifest["assignments"][1]["id"] = identifier

    with pytest.raises(ValueError, match="excluded family"):
        _validate(root, manifest, registry)


def test_registry_digest_and_exact_manifest_schema_are_required(tmp_path) -> None:
    root, manifest, registry = _make_case(tmp_path)
    raw = _registry_bytes(registry)
    with pytest.raises(ValueError, match="registry digest"):
        validate_controlled3d(root, manifest, reviewed_registry=raw, registry_sha256="0" * 64)

    extra = deepcopy(manifest)
    extra["unexpected"] = True
    with pytest.raises(ValueError, match="manifest fields"):
        _validate(root, extra, registry)


def test_alias_assignments_cannot_split_or_duplicate_the_canonical_family(tmp_path) -> None:
    root, manifest, registry = _make_case(tmp_path)
    manifest["assignments"][1]["id"] = "demo-family-alias-train"
    manifest["assignments"].append({"kind": "family", "id": "demo-family-train", "split": "train"})

    with pytest.raises(ValueError, match="duplicate canonical family assignment"):
        _validate(root, manifest, registry)


def test_manifest_rejects_pair_endpoint_split_leakage(tmp_path) -> None:
    root, manifest, registry = _make_case(tmp_path)
    manifest["pairs"][0]["split"] = "dev"

    with pytest.raises(ValueError, match="split"):
        _validate(root, manifest, registry)


@pytest.mark.parametrize(
    "path",
    ["../train/escape.png", "/train/absolute.png", "train\\assets\\image.png", "train/a//b.png"],
)
def test_nonsealed_asset_paths_are_lexically_safe(tmp_path, path) -> None:
    root, manifest, registry = _make_case(tmp_path)
    manifest["pairs"][0]["renders"][0]["image"]["path"] = path

    with pytest.raises(ValueError, match="unsafe asset path"):
        _validate(root, manifest, registry)


def test_nonsealed_path_components_cannot_be_symlinks(tmp_path) -> None:
    root, manifest, registry = _make_case(tmp_path)
    original = manifest["pairs"][0]["renders"][0]["image"]
    link = root / "train/images/linked.png"
    link.symlink_to(root / original["path"])
    original["path"] = "train/images/linked.png"

    with pytest.raises(ValueError, match="symlink"):
        _validate(root, manifest, registry)


def test_shared_state_checker_rejects_actual_rehashed_geometry_change(tmp_path) -> None:
    root, manifest, registry = _make_case(tmp_path)
    pair = manifest["pairs"][0]
    bundle = root / pair["snapshot_bundle"]["path"]
    records = json.loads(bundle.read_text(encoding="utf-8"))
    records[1]["snapshot"]["geometry"]["Carton"]["mesh_sha256"] = "9" * 64
    records[1]["snapshot_sha256"] = snapshot_sha256(records[1]["snapshot"])
    raw = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    bundle.write_bytes(raw)
    pair["snapshot_bundle"]["sha256"] = _digest(raw)

    with pytest.raises(ValueError, match="geometry changed in cream"):
        _validate(root, manifest, registry)


def _replace_mask(
    root: Path, manifest: dict[str, Any], mask: Image.Image, *, bbox: list[int] | None = None
) -> None:
    pair = manifest["pairs"][0]
    mask_bytes = _png(mask)
    for render in pair["renders"]:
        ref = render["mask"]
        (root / ref["path"]).write_bytes(mask_bytes)
        ref["sha256"] = _digest(mask_bytes)
        ref["decoded_sha256"] = _digest(mask.tobytes())
        if bbox is not None:
            render["bbox"] = bbox


@pytest.mark.parametrize("kind", ["nonbinary", "empty", "clipped", "no-core"])
def test_mask_contract_rejects_invalid_masks(tmp_path, kind) -> None:
    root, manifest, registry = _make_case(tmp_path)
    mask = Image.new("L", (512, 512), 0)
    draw = ImageDraw.Draw(mask)
    if kind == "nonbinary":
        draw.rectangle((180, 170, 329, 349), fill=255)
        mask.putpixel((220, 220), 128)
        bbox = [180, 170, 330, 350]
    elif kind == "empty":
        bbox = [0, 0, 0, 0]
    elif kind == "clipped":
        draw.rectangle((0, 170, 100, 349), fill=255)
        bbox = [0, 170, 101, 350]
    else:
        draw.rectangle((180, 170, 182, 172), fill=255)
        bbox = [180, 170, 183, 173]
    _replace_mask(root, manifest, mask, bbox=bbox)

    with pytest.raises(ValueError, match="mask"):
        _validate(root, manifest, registry)


def test_product_pixel_changes_are_reported_without_a_quality_threshold(tmp_path) -> None:
    root, manifest, registry = _make_case(tmp_path)
    image_ref = manifest["pairs"][0]["renders"][1]["image"]
    image_path = root / image_ref["path"]
    image = Image.open(image_path).convert("RGB")
    image.putpixel((250, 250), (1, 2, 3))
    image_ref["sha256"] = _digest(_write_png := _png(image))
    image_path.write_bytes(_write_png)

    metrics = json.loads(_validate(root, manifest, registry).metrics_json)
    cream = metrics["pairs"]["demo-pair-train"]["rgb_changes"]["cream"]
    assert cream["product"] > 0
    assert cream["core"] > 0


def test_changed_asset_after_validation_blocks_export_before_publication(tmp_path) -> None:
    root, manifest, registry = _make_case(tmp_path)
    validated = _validate(root, manifest, registry)
    image_path = root / manifest["pairs"][0]["renders"][0]["image"]["path"]
    image_path.write_bytes(b"changed")
    destination = tmp_path / "output"

    with pytest.raises(ValueError, match="asset hash mismatch"):
        export_train_only(root, validated, destination)
    assert not destination.exists()


def test_train_only_export_omits_dev_and_sealed_data_and_never_reads_sealed_paths(
    tmp_path, monkeypatch
) -> None:
    import builtins
    import io
    import os
    import shutil

    root, manifest, registry = _make_case(tmp_path, ("train", "dev", "sealed_test"))
    shutil.rmtree(root / "sealed_test")
    real_open, real_io_open = builtins.open, io.open
    real_os_open, real_os_stat, real_os_lstat = os.open, os.stat, os.lstat
    real_path_methods = {
        "resolve": Path.resolve,
        "stat": Path.stat,
        "lstat": Path.lstat,
        "is_file": Path.is_file,
        "exists": Path.exists,
        "open": Path.open,
        "read_bytes": Path.read_bytes,
    }
    train_reads: list[str] = []
    rejected_seams: list[str] = []
    sealed_root = str(root / "sealed_test").replace("\\", "/")

    def _has_sealed_path(value: Any) -> bool:
        normalized = str(value).replace("\\", "/")
        return (
            normalized == "sealed_test"
            or normalized.startswith(sealed_root + "/")
            or normalized == sealed_root
        )

    def reject_sealed(seam: str, path: Any) -> None:
        if _has_sealed_path(path):
            rejected_seams.append(seam)
            raise AssertionError(f"sealed path {seam}: {path}")

    def checked_open(path, *args, **kwargs):
        reject_sealed("builtins.open", path)
        return real_open(path, *args, **kwargs)

    def checked_io_open(path, *args, **kwargs):
        reject_sealed("io.open", path)
        return real_io_open(path, *args, **kwargs)

    def checked_os_open(path, *args, **kwargs):
        reject_sealed("os.open", path)
        if kwargs.get("dir_fd") is not None and "train" in str(path):
            train_reads.append(str(path))
        return real_os_open(path, *args, **kwargs)

    def checked_os_stat(path, *args, **kwargs):
        reject_sealed("os.stat", path)
        return real_os_stat(path, *args, **kwargs)

    def checked_os_lstat(path, *args, **kwargs):
        reject_sealed("os.lstat", path)
        return real_os_lstat(path, *args, **kwargs)

    def checked_path_method(name):
        original = real_path_methods[name]

        def checked(path, *args, **kwargs):
            reject_sealed(f"Path.{name}", path)
            return original(path, *args, **kwargs)

        return checked

    monkeypatch.setattr(builtins, "open", checked_open)
    monkeypatch.setattr(io, "open", checked_io_open)
    monkeypatch.setattr(os, "open", checked_os_open)
    monkeypatch.setattr(os, "stat", checked_os_stat)
    monkeypatch.setattr(os, "lstat", checked_os_lstat)
    for method in real_path_methods:
        monkeypatch.setattr(Path, method, checked_path_method(method))

    sealed_file = root / "sealed_test" / "never-created.png"
    patched_io_open = io.open
    positive_controls = {
        "builtins.open": lambda: open(sealed_file, "rb"),
        "io.open": lambda: patched_io_open(sealed_file, "rb"),
        "os.open": lambda: os.open(sealed_file, os.O_RDONLY),
        "os.stat": lambda: os.stat(sealed_file),
        "os.lstat": lambda: os.lstat(sealed_file),
        **{
            f"Path.{method}": lambda method=method: getattr(sealed_file, method)()
            for method in real_path_methods
        },
    }
    for seam, access in positive_controls.items():
        with pytest.raises(AssertionError, match=f"sealed path {seam}"):
            access()
        assert seam in rejected_seams

    validated = _validate(root, manifest, registry)
    destination = tmp_path / "export"
    export_train_only(root, validated, destination)

    assert train_reads
    assert any(path.startswith("demo-pair-train-") for path in train_reads)
    assert (destination / "completion.json").is_file()
    exported = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
    assert {row["split"] for row in exported["assignments"]} == {"train"}
    assert len(exported["pairs"]) == 1
    assert len(metadata) == 3
    assert all(set(row) == {"image", "edit_image", "prompt"} for row in metadata)
    output_text = json.dumps(exported) + json.dumps(metadata)
    assert "DEV" not in output_text and "SEALED_TEST" not in output_text
    assert not (destination / "dev").exists()
    assert not (destination / "sealed_test").exists()


def test_draft_manifest_validates_ids_but_never_opens_referenced_assets(
    tmp_path, monkeypatch
) -> None:
    import product_scene_editor.controlled3d_assets as assets
    from product_scene_editor.controlled3d import draft_manifest

    _, manifest, _ = _make_case(tmp_path)
    manifest["pairs"][0]["renders"][0]["image"]["path"] = "train/../not-read.png"
    path = tmp_path / "draft.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        assets, "read_asset", lambda *_args: pytest.fail("draft inspected an asset")
    )

    result = draft_manifest(path)

    assert result["pairs"][0]["renders"][0]["image"]["path"] == "train/../not-read.png"


def test_schema_rejects_boolean_version_and_nonempty_external_inputs(tmp_path) -> None:
    root, manifest, registry = _make_case(tmp_path)
    manifest["schema_version"] = True
    with pytest.raises(ValueError, match="schema_version"):
        _validate(root, manifest, registry)

    _, manifest, registry = _make_case(tmp_path / "external")
    manifest["provenance"]["external_inputs"] = ["unapproved-image.png"]
    with pytest.raises(ValueError, match="provenance"):
        _validate(root, manifest, registry)


def test_installed_shared_validator_source_pin_is_checked(tmp_path) -> None:
    root, manifest, registry = _make_case(tmp_path)
    registry["pairs"][0]["pins"]["validator_source_sha256"] = "0" * 64
    manifest["pairs"][0]["pins"]["validator_source_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="validator source pin"):
        _validate(root, manifest, registry)


def test_decoded_product_copy_cannot_cross_train_and_dev(tmp_path) -> None:
    root, manifest, registry = _make_case(tmp_path, ("train", "dev"))
    train = manifest["pairs"][0]["renders"][0]
    dev_pair = manifest["pairs"][1]
    dev = dev_pair["renders"][0]
    mask = Image.open(root / train["mask"]["path"])
    source = Image.open(root / train["image"]["path"]).convert("RGB")
    duplicate = Image.open(root / dev["image"]["path"]).convert("RGB")
    mask_pixels = mask.tobytes()
    for index, value in enumerate(mask_pixels):
        if value:
            duplicate.putpixel(
                (index % 512, index // 512), source.getpixel((index % 512, index // 512))
            )
    raw = _png(duplicate)
    (root / dev["image"]["path"]).write_bytes(raw)
    dev["image"]["sha256"] = _digest(raw)

    with pytest.raises(ValueError, match="decoded product pixels cross"):
        _validate(root, manifest, registry)


def test_rgba_target_must_have_opaque_alpha(tmp_path) -> None:
    root, manifest, registry = _make_case(tmp_path)
    row = manifest["pairs"][0]["renders"][1]
    source = Image.open(root / row["image"]["path"]).convert("RGBA")
    source.putpixel((10, 10), (238, 225, 205, 128))
    raw = _png(source)
    (root / row["image"]["path"]).write_bytes(raw)
    row["image"]["sha256"] = _digest(raw)

    with pytest.raises(ValueError, match="alpha must be opaque"):
        _validate(root, manifest, registry)


def test_train_export_cleans_owned_stage_when_staging_fails(tmp_path, monkeypatch) -> None:
    root, manifest, registry = _make_case(tmp_path)
    validated = _validate(root, manifest, registry)
    destination = tmp_path / "failed-output"
    original = Path.write_bytes

    def fail_metadata(path, data):
        if path.name == "metadata.json":
            raise OSError("simulated staging write failure")
        return original(path, data)

    monkeypatch.setattr(Path, "write_bytes", fail_metadata)
    with pytest.raises(OSError, match="simulated staging write failure"):
        export_train_only(root, validated, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".controlled3d-stage-*"))


def test_export_rejects_existing_destination_and_symlink_destination(tmp_path) -> None:
    root, manifest, registry = _make_case(tmp_path)
    validated = _validate(root, manifest, registry)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        export_train_only(root, validated, existing)

    target = tmp_path / "symlinked"
    target.symlink_to(existing, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        export_train_only(root, validated, target)


@pytest.mark.parametrize("render_names", list(permutations(("source", "cream", "sage", "clay"))))
def test_export_maps_reordered_renders_by_name(tmp_path, render_names) -> None:
    root, manifest, registry = _make_case(tmp_path)
    pair = manifest["pairs"][0]
    rows = {row["name"]: row for row in pair["renders"]}
    pair["renders"] = [rows[name] for name in render_names]
    target_names = ("cream", "sage", "clay")
    expected_targets = [rows[name]["image"]["path"] for name in target_names]
    expected_source = rows["source"]["image"]["path"]
    expected_prompts = [
        "Keep the product, label, and geometry unchanged; "
        f"change only the background to the {name} variant."
        for name in target_names
    ]

    validated = _validate(root, manifest, registry)
    destination = tmp_path / "ordered-export"
    export_train_only(root, validated, destination)
    metadata = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))

    assert [row["image"] for row in metadata] == expected_targets
    assert [row["edit_image"] for row in metadata] == [expected_source] * 3
    assert [row["prompt"] for row in metadata] == expected_prompts


def test_export_metadata_matches_remote_development_consumer_contract(tmp_path) -> None:
    from product_scene_editor.remote import _prepare_matched_development_assets

    root, manifest, registry = _make_case(tmp_path)
    pair = manifest["pairs"][0]
    renders = {row["name"]: row for row in pair["renders"]}
    validated = _validate(root, manifest, registry)
    destination = tmp_path / "consumer-contract-export"
    export_train_only(root, validated, destination)
    metadata = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))

    consumer_rows = []
    for index, name in enumerate(("cream", "sage", "clay")):
        exported_row = metadata[index]
        for duplicate in range(3):
            consumer_rows.append(
                {
                    **exported_row,
                    "pair_id": f"synthetic-{name}-{duplicate}",
                    "mask": renders[name]["mask"]["path"],
                }
            )

    assets = _prepare_matched_development_assets(consumer_rows, destination, Image)
    expected_source = Image.open(destination / renders["source"]["image"]["path"]).convert("RGB")
    for index, name in enumerate(("cream", "sage", "clay")):
        expected_target = Image.open(destination / renders[name]["image"]["path"]).convert("RGB")
        for duplicate in range(3):
            asset = assets[f"synthetic-{name}-{duplicate}"]
            assert asset["source"].tobytes() == expected_source.tobytes()
            assert asset["target"].tobytes() == expected_target.tobytes()
            assert asset["prompt"] == metadata[index]["prompt"]


def test_read_asset_rejects_fifo_without_blocking(tmp_path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO special files are unavailable")
    root, _, _ = _make_case(tmp_path)
    relative = "train/pins/pipe"
    fifo = root / relative
    fifo.parent.mkdir(parents=True, exist_ok=True)
    os.mkfifo(fifo)
    code = """import sys
from product_scene_editor.controlled3d_assets import read_asset
try:
    read_asset(sys.argv[1], sys.argv[2])
except ValueError as error:
    print(f"rejected:{error}")
else:
    raise SystemExit("FIFO was accepted")
"""

    result = subprocess.run(
        [sys.executable, "-c", code, str(root), relative],
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "rejected:asset is not a regular file" in result.stdout


def test_export_removes_only_owned_output_after_publish_failure(tmp_path, monkeypatch) -> None:
    root, manifest, registry = _make_case(tmp_path)
    validated = _validate(root, manifest, registry)
    destination = tmp_path / "partial-output"
    neighbor = tmp_path / "neighbor"
    neighbor.mkdir()
    (neighbor / "keep.txt").write_text("keep", encoding="utf-8")
    original_copyfile = assets.shutil.copyfile
    copies = 0

    def fail_after_copy(source, target, *args, **kwargs):
        nonlocal copies
        result = original_copyfile(source, target, *args, **kwargs)
        copies += 1
        if copies == 1:
            raise OSError("simulated publish failure")
        return result

    monkeypatch.setattr(assets.shutil, "copyfile", fail_after_copy)
    with pytest.raises(OSError, match="simulated publish failure"):
        export_train_only(root, validated, destination)

    assert copies == 1
    assert not destination.exists()
    assert not (destination / "completion.json").exists()
    assert (neighbor / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert not list(tmp_path.glob(".controlled3d-stage-*"))


def test_export_preserves_competing_destination_created_before_claim(tmp_path, monkeypatch) -> None:
    root, manifest, registry = _make_case(tmp_path)
    validated = _validate(root, manifest, registry)
    destination = tmp_path / "raced-output"
    original_mkdir = assets.os.mkdir

    def create_competing_destination(path, *args, **kwargs):
        if os.path.abspath(os.fspath(path)) == str(destination):
            original_mkdir(path, *args, **kwargs)
            (destination / "competitor.txt").write_text("keep", encoding="utf-8")
            raise FileExistsError("competing destination won")
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(assets.os, "mkdir", create_competing_destination)
    with pytest.raises(FileExistsError, match="competing destination won"):
        export_train_only(root, validated, destination)

    assert (destination / "competitor.txt").read_text(encoding="utf-8") == "keep"
    assert not (destination / "completion.json").exists()
    assert not list(tmp_path.glob(".controlled3d-stage-*"))
