from __future__ import annotations

import hashlib
from io import BytesIO
from zipfile import ZIP_STORED, ZipFile

import pytest
from PIL import Image

from product_scene_editor import mask_preflight

_MODEL = {
    "repo_id": "ZhengPeng7/BiRefNet_lite",
    "revision": "aa62cd87eafb9cc43056d08ef3615a14628b831d",
}
_MODEL_FILES = {
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
_RUNTIME_PINS = {
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
_SETTINGS = {
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
_CONTEXT = {"profile": "reflex-personal", "workspace": "rajath-61258", "environment": "main"}


def _png(color: tuple[int, int, int]) -> bytes:
    stream = BytesIO()
    Image.new("RGB", (512, 512), color).save(stream, format="PNG")
    return stream.getvalue()


def _fixture() -> tuple[dict[str, object], dict[str, bytes]]:
    files: dict[str, bytes] = {}
    cases: list[dict[str, object]] = []
    for index in range(24):
        case_id = f"case-{index:02d}"
        path = f"cases/{case_id}/source.png"
        source = _png((index + 1, 40, 80))
        files[path] = source
        cases.append(
            {
                "case_id": case_id,
                "source_rgb": {"path": path, "sha256": hashlib.sha256(source).hexdigest()},
            }
        )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "profile": "mask-preflight",
        "source_sha": "a" * 40,
        "context": _CONTEXT,
        "model": _MODEL,
        "model_files": _MODEL_FILES,
        "runtime_pins": _RUNTIME_PINS,
        "settings": _SETTINGS,
        "cases": cases,
    }
    return manifest, files


def test_payload_builder_emits_canonical_stored_zip_and_round_trips() -> None:
    manifest, files = _fixture()

    payload = mask_preflight.build_payload(manifest, files)
    parsed_manifest, parsed_files = mask_preflight.validate_payload_bytes(payload)
    second_payload = mask_preflight.build_payload(manifest, files)

    assert payload == second_payload
    assert parsed_manifest == manifest
    assert parsed_files == files
    with ZipFile(BytesIO(payload)) as archive:
        assert archive.namelist() == sorted(archive.namelist())
        assert all(member.compress_type == ZIP_STORED for member in archive.infolist())
        assert all(member.date_time == (1980, 1, 1, 0, 0, 0) for member in archive.infolist())


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("model", {"repo_id": _MODEL["repo_id"], "revision": "0" * 40}, "model pin"),
        (
            "model_files",
            {**_MODEL_FILES, "birefnet.py": {"sha256": "0" * 64, "size": 1}},
            "file digests",
        ),
        ("runtime_pins", {**_RUNTIME_PINS, "torch": "0.0"}, "runtime pins"),
        ("settings", {**_SETTINGS, "alpha_threshold": 0.51}, "settings"),
    ],
)
def test_payload_builder_rejects_changed_frozen_pins(field, replacement, message) -> None:
    manifest, files = _fixture()
    manifest[field] = replacement

    with pytest.raises(ValueError, match=message):
        mask_preflight.build_payload(manifest, files)


def test_payload_builder_rejects_source_hash_and_member_mismatch() -> None:
    manifest, files = _fixture()
    first_path = manifest["cases"][0]["source_rgb"]["path"]
    files[first_path] = _png((255, 0, 0))

    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        mask_preflight.build_payload(manifest, files)


def test_payload_builder_rejects_unknown_or_unreferenced_assets() -> None:
    manifest, files = _fixture()
    manifest["cases"][0]["unexpected"] = "reviewed"

    with pytest.raises(ValueError, match="case fields"):
        mask_preflight.build_payload(manifest, files)

    clean_manifest, clean_files = _fixture()
    clean_files["cases/extra/source.png"] = _png((1, 2, 3))
    with pytest.raises(ValueError, match="missing or unexpected"):
        mask_preflight.build_payload(clean_manifest, clean_files)


def test_payload_validator_rejects_noncanonical_archive_bytes() -> None:
    manifest, files = _fixture()
    payload = mask_preflight.build_payload(manifest, files)
    with ZipFile(BytesIO(payload)) as archive:
        content = [(item.filename, archive.read(item)) for item in archive.infolist()]
    reordered = BytesIO()
    with ZipFile(reordered, "w", compression=ZIP_STORED) as archive:
        for name, data in reversed(content):
            archive.writestr(name, data)

    with pytest.raises(ValueError, match="sorted"):
        mask_preflight.validate_payload_bytes(reordered.getvalue())
