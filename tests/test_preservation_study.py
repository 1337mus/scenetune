from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter
from io import BytesIO

import pytest

from product_scene_editor import preservation_study


def _parent_payload(monkeypatch):
    from test_controlled3d_study import _controlled3d_payload

    parent_bytes, _train, _dev, _files = _controlled3d_payload(monkeypatch)
    monkeypatch.setattr(
        preservation_study,
        "PARENT_RUN_ID",
        "controlled3d-test-001",
        raising=False,
    )
    monkeypatch.setattr(
        preservation_study,
        "PARENT_PAYLOAD_SHA256",
        hashlib.sha256(parent_bytes).hexdigest(),
        raising=False,
    )
    return parent_bytes


def test_build_preservation_payload_uses_balanced_identity_and_edit_rows(monkeypatch) -> None:
    parent_bytes = _parent_payload(monkeypatch)

    candidate, manifest_sha256 = preservation_study.build_preservation_payload(
        parent_bytes, "preservation-test-001"
    )

    with zipfile.ZipFile(BytesIO(candidate)) as archive:
        manifest_raw = archive.read("preservation_manifest.json")
    manifest = json.loads(manifest_raw)
    train_rows = manifest["train_rows"]
    dev_rows = manifest["dev_rows"]
    assert hashlib.sha256(manifest_raw).hexdigest() == manifest_sha256
    assert len(train_rows) == 18
    assert len(dev_rows) == 9
    assert Counter(row["product_id"] for row in train_rows) == {
        f"train-product-{index}": 3 for index in range(6)
    }
    assert Counter(row["pair_id"].rsplit("-", 1)[-1] for row in train_rows) == {
        "identity": 6,
        "cream": 4,
        "sage": 4,
        "clay": 4,
    }
    assert Counter(row["product_id"] for row in dev_rows) == {
        f"dev-product-{index}": 3 for index in range(3)
    }
    expected_states = ("cream", "sage", "clay")
    for product_index in range(6):
        product_rows = [
            row for row in train_rows if row["product_id"] == f"train-product-{product_index}"
        ]
        assert {row["pair_id"].rsplit("-", 1)[-1] for row in product_rows} == {
            "identity",
            *set(expected_states) - {expected_states[product_index % 3]},
        }
    repeated_candidate, repeated_manifest_sha256 = preservation_study.build_preservation_payload(
        parent_bytes, "preservation-test-other-run"
    )
    assert repeated_candidate == candidate
    assert repeated_manifest_sha256 == manifest_sha256


def test_validator_recomputes_exact_masked_targets_and_identity_rows(monkeypatch) -> None:
    from PIL import Image

    from product_scene_editor import controlled3d_study_payload

    parent_bytes = _parent_payload(monkeypatch)
    candidate, manifest_sha256 = preservation_study.build_preservation_payload(
        parent_bytes, "preservation-test-001"
    )
    monkeypatch.setattr(
        preservation_study, "APPROVED_PRESERVATION_MANIFEST_SHA256", manifest_sha256
    )
    parsed_parent = controlled3d_study_payload.validate_controlled3d_study_payload_bytes(
        parent_bytes, "controlled3d-test-001"
    )
    parsed = preservation_study.validate_preservation_study_payload_bytes(
        candidate, "preservation-test-001"
    )

    identity_rows = [row for row in parsed.train_rows if row["pair_id"].endswith("-identity")]
    assert len(identity_rows) == 6
    assert all(row["image"] == row["edit_image"] for row in identity_rows)
    for row in (*parsed.train_rows, *parsed.dev_rows):
        if row["pair_id"].endswith("-identity"):
            continue
        condition = row["pair_id"].rsplit("-", 1)[-1]
        scene_pair_id = row["pair_id"][: -(len(condition) + 1)]
        parent_row = next(
            parent_row
            for parent_row in (*parsed_parent.train_rows, *parsed_parent.dev_rows)
            if parent_row["scene_pair_id"] == scene_pair_id and parent_row["state"] == condition
        )
        source = Image.open(BytesIO(parsed.files[row["edit_image"]])).convert("RGB")
        target = Image.open(BytesIO(parsed.files[row["image"]])).convert("RGB")
        mask = Image.open(BytesIO(parsed.files[row["mask"]])).convert("L")
        original_target = Image.open(BytesIO(parsed_parent.files[parent_row["image"]])).convert(
            "RGB"
        )
        source_pixels = source.tobytes()
        target_pixels = target.tobytes()
        background_pixels = original_target.tobytes()
        for index, mask_pixel in enumerate(mask.tobytes()):
            start = index * 3
            expected = source_pixels if mask_pixel else background_pixels
            assert target_pixels[start : start + 3] == expected[start : start + 3]


def test_read_rejects_plan_with_changed_pair_ids(tmp_path, monkeypatch) -> None:
    from dataclasses import replace

    parent_bytes = _parent_payload(monkeypatch)
    _candidate, manifest_sha256 = preservation_study.build_preservation_payload(
        parent_bytes, "preservation-test-002"
    )
    monkeypatch.setattr(
        preservation_study, "APPROVED_PRESERVATION_MANIFEST_SHA256", manifest_sha256
    )
    parent_path = tmp_path / "parent.zip"
    parent_path.write_bytes(parent_bytes)
    plan = preservation_study.plan_preservation_study(
        parent_path,
        tmp_path / "planned",
        run_id="preservation-test-002",
    )

    altered_plan = replace(plan, training_pair_ids=tuple(reversed(plan.training_pair_ids)))
    with pytest.raises(ValueError, match="pair IDs|marker"):
        preservation_study.read_preservation_study_payload(altered_plan)


def test_validator_fails_closed_without_derived_pin(monkeypatch) -> None:
    monkeypatch.setattr(preservation_study, "APPROVED_PRESERVATION_MANIFEST_SHA256", None)

    with pytest.raises(ValueError, match="manifest.*pin"):
        preservation_study.validate_preservation_study_payload_bytes(b"not a zip", "run-1")


def test_validator_uses_derived_manifest_pin_not_parent_manifest_digest(monkeypatch) -> None:
    parent_bytes = _parent_payload(monkeypatch)
    candidate, derived_manifest_sha256 = preservation_study.build_preservation_payload(
        parent_bytes, "preservation-test-003"
    )
    parent = preservation_study._parent_payload(parent_bytes)
    monkeypatch.setattr(
        preservation_study,
        "APPROVED_PRESERVATION_MANIFEST_SHA256",
        parent.source_dataset_manifest_sha256,
    )

    with pytest.raises(ValueError, match="manifest digest differs"):
        preservation_study.validate_preservation_study_payload_bytes(
            candidate, "preservation-test-003"
        )

    monkeypatch.setattr(
        preservation_study, "APPROVED_PRESERVATION_MANIFEST_SHA256", derived_manifest_sha256
    )
    parsed = preservation_study.validate_preservation_study_payload_bytes(
        candidate, "preservation-test-003"
    )
    assert parsed.source_dataset_manifest_sha256 == parent.source_dataset_manifest_sha256
    assert parsed.preservation_manifest_sha256 == derived_manifest_sha256


def test_validator_rejects_tampered_derived_target(tmp_path, monkeypatch) -> None:
    parent_bytes = _parent_payload(monkeypatch)
    candidate, manifest_sha256 = preservation_study.build_preservation_payload(
        parent_bytes, "preservation-test-004"
    )
    monkeypatch.setattr(
        preservation_study, "APPROVED_PRESERVATION_MANIFEST_SHA256", manifest_sha256
    )
    with zipfile.ZipFile(BytesIO(candidate)) as archive:
        members = {info.filename: archive.read(info) for info in archive.infolist()}
    target_name = next(name for name in members if name.startswith("train/target/"))
    members[target_name] += b"tampered"
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(members.items()):
            archive.writestr(name, raw)

    with pytest.raises(ValueError, match="exact CPU derivation"):
        preservation_study.validate_preservation_study_payload_bytes(
            buffer.getvalue(), "preservation-test-004"
        )


def test_plan_round_trip_binds_parent_and_payload_and_is_write_once(tmp_path, monkeypatch) -> None:
    parent_bytes = _parent_payload(monkeypatch)
    _candidate, manifest_sha256 = preservation_study.build_preservation_payload(
        parent_bytes, "preservation-test-005"
    )
    monkeypatch.setattr(
        preservation_study, "APPROVED_PRESERVATION_MANIFEST_SHA256", manifest_sha256
    )
    parent_path = tmp_path / "parent.zip"
    parent_path.write_bytes(parent_bytes)
    output_dir = tmp_path / "plan"
    plan = preservation_study.plan_preservation_study(
        parent_path, output_dir, run_id="preservation-test-005"
    )
    loaded = preservation_study.load_preservation_study_plan(output_dir)
    parsed = preservation_study.read_preservation_study_payload(loaded)

    assert loaded == plan
    assert len(parsed.train_rows) == 18
    assert len(parsed.dev_rows) == 9
    assert parsed.parent_payload_sha256 == preservation_study.PARENT_PAYLOAD_SHA256
    with pytest.raises(FileExistsError, match="must be new"):
        preservation_study.plan_preservation_study(
            parent_path, output_dir, run_id="preservation-test-005"
        )

    plan.payload_path.write_bytes(plan.payload_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="bytes changed"):
        preservation_study.read_preservation_study_payload(plan)


def test_validator_rejects_parent_digest_substitution_and_extra_assets(monkeypatch) -> None:
    parent_bytes = _parent_payload(monkeypatch)
    candidate, manifest_sha256 = preservation_study.build_preservation_payload(
        parent_bytes, "preservation-test-006"
    )
    monkeypatch.setattr(
        preservation_study, "APPROVED_PRESERVATION_MANIFEST_SHA256", manifest_sha256
    )
    with zipfile.ZipFile(BytesIO(candidate)) as archive:
        members = {info.filename: archive.read(info) for info in archive.infolist()}

    original_manifest = members["preservation_manifest.json"]
    manifest = json.loads(original_manifest)
    manifest["parent_payload_sha256"] = "9" * 64
    members["preservation_manifest.json"] = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    substituted = BytesIO()
    with zipfile.ZipFile(substituted, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(members.items()):
            archive.writestr(name, raw)
    with pytest.raises(ValueError, match="deterministic derivation"):
        preservation_study.validate_preservation_study_payload_bytes(
            substituted.getvalue(), "preservation-test-006"
        )

    members["preservation_manifest.json"] = original_manifest
    members["sealed_test/extra.png"] = b"not an image"
    extra_member = BytesIO()
    with zipfile.ZipFile(extra_member, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(members.items()):
            archive.writestr(name, raw)
    with pytest.raises(ValueError, match="missing or unexpected files"):
        preservation_study.validate_preservation_study_payload_bytes(
            extra_member.getvalue(), "preservation-test-006"
        )


def test_validator_rejects_metadata_change(monkeypatch) -> None:
    parent_bytes = _parent_payload(monkeypatch)
    candidate, manifest_sha256 = preservation_study.build_preservation_payload(
        parent_bytes, "preservation-test-007"
    )
    monkeypatch.setattr(
        preservation_study, "APPROVED_PRESERVATION_MANIFEST_SHA256", manifest_sha256
    )
    with zipfile.ZipFile(BytesIO(candidate)) as archive:
        members = {info.filename: archive.read(info) for info in archive.infolist()}
    metadata = json.loads(members["metadata.json"])
    metadata[0]["prompt"] = "Change the product and label."
    members["metadata.json"] = (
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    altered = BytesIO()
    with zipfile.ZipFile(altered, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(members.items()):
            archive.writestr(name, raw)

    with pytest.raises(ValueError, match="metadata differs"):
        preservation_study.validate_preservation_study_payload_bytes(
            altered.getvalue(), "preservation-test-007"
        )


def test_portable_target_encoder_avoids_pillow_save_and_checksums(monkeypatch) -> None:
    import struct
    import zlib

    from PIL import Image

    parent_bytes = _parent_payload(monkeypatch)

    def reject_pillow_save(*_args, **_kwargs):
        raise AssertionError("Pillow image encoding is not portable")

    monkeypatch.setattr(Image.Image, "save", reject_pillow_save)
    candidate, manifest_sha256 = preservation_study.build_preservation_payload(
        parent_bytes, "preservation-portable-001"
    )
    repeated, repeated_manifest_sha256 = preservation_study.build_preservation_payload(
        parent_bytes, "preservation-portable-002"
    )
    assert repeated == candidate
    assert repeated_manifest_sha256 == manifest_sha256
    monkeypatch.setattr(
        preservation_study, "APPROVED_PRESERVATION_MANIFEST_SHA256", manifest_sha256
    )
    parsed = preservation_study.validate_preservation_study_payload_bytes(
        candidate, "preservation-portable-001"
    )
    row = next(row for row in parsed.train_rows if row["image"] != row["edit_image"])
    png = parsed.files[row["image"]]
    with Image.open(BytesIO(png)) as decoded:
        decoded.load()
        assert decoded.format == "PNG"
        assert decoded.mode == "RGB"
        assert decoded.size == (512, 512)
        decoded_pixels = decoded.tobytes()

    chunks: list[tuple[bytes, bytes]] = []
    offset = 8
    while offset < len(png):
        length = struct.unpack(">I", png[offset : offset + 4])[0]
        kind = png[offset + 4 : offset + 8]
        data = png[offset + 8 : offset + 8 + length]
        crc = struct.unpack(">I", png[offset + 8 + length : offset + 12 + length])[0]
        assert crc == zlib.crc32(kind + data) & 0xFFFFFFFF
        chunks.append((kind, data))
        offset += length + 12
    assert offset == len(png)
    idat = b"".join(data for kind, data in chunks if kind == b"IDAT")
    assert chunks[0][0] == b"IHDR"
    assert chunks[-1] == (b"IEND", b"")
    assert idat[:2] == b"\x78\x01"
    raw_scanlines = zlib.decompress(idat)
    expected_scanlines = b"".join(
        b"\x00" + decoded_pixels[row_index * 1536 : (row_index + 1) * 1536]
        for row_index in range(512)
    )
    assert raw_scanlines == expected_scanlines

    deflate = idat[2:-4]
    block_offset = 0
    final = False
    while not final:
        header = deflate[block_offset]
        final = bool(header & 1)
        assert header & 0x06 == 0
        length, inverse = struct.unpack("<HH", deflate[block_offset + 1 : block_offset + 5])
        assert length ^ inverse == 0xFFFF
        block_offset += 5 + length
    assert block_offset == len(deflate)
