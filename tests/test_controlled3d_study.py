from __future__ import annotations

import json

import pytest

from product_scene_editor import controlled3d_study


def test_unset_controlled3d_manifest_pin_fails_before_reading_manifest(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(controlled3d_study, "APPROVED_CONTROLLED3D_MANIFEST_SHA256", None)
    missing_manifest = tmp_path / "manifest.json"

    with pytest.raises(ValueError, match="manifest.*pin|pin.*manifest"):
        controlled3d_study.approve_manifest_pin(missing_manifest)


def test_plan_refuses_unset_reviewed_digests_before_reading_any_input(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(controlled3d_study, "APPROVED_CONTROLLED3D_MANIFEST_SHA256", None)
    monkeypatch.setattr(controlled3d_study, "APPROVED_CONTROLLED3D_REGISTRY_SHA256", None)
    with pytest.raises(ValueError, match="manifest.*pin|pin.*manifest"):
        controlled3d_study.plan_controlled3d_study(
            tmp_path / "dataset",
            tmp_path / "manifest.json",
            tmp_path / "registry.json",
            tmp_path / "train-export",
            tmp_path / "plan",
            run_id="controlled3d-test-001",
        )


def _png(image):
    from io import BytesIO

    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def _controlled3d_payload(monkeypatch):
    import hashlib

    from PIL import Image, ImageDraw

    from product_scene_editor import controlled3d_study_payload as payload

    manifest_sha = "1" * 64
    registry_sha = "2" * 64
    monkeypatch.setattr(controlled3d_study, "APPROVED_CONTROLLED3D_MANIFEST_SHA256", manifest_sha)
    monkeypatch.setattr(controlled3d_study, "APPROVED_CONTROLLED3D_REGISTRY_SHA256", registry_sha)
    files = {}
    by_split = {"train": [], "dev": []}
    for split, count in (("train", 6), ("dev", 3)):
        for index in range(count):
            pair_id = f"{split}-pair-{index}"
            product_id = f"{split}-product-{index}"
            source = Image.new("RGB", (512, 512), (20, 30, 40))
            ImageDraw.Draw(source).rectangle((100, 100, 199, 199), fill=(20, 80, 120))
            mask = Image.new("L", (512, 512), 0)
            ImageDraw.Draw(mask).rectangle((100, 100, 199, 199), fill=255)
            source_path = f"{split}/source/{pair_id}.png"
            mask_path = f"{split}/masks/{pair_id}.png"
            source_raw, mask_raw = _png(source), _png(mask)
            files[source_path], files[mask_path] = source_raw, mask_raw
            for state, offset in zip(payload._STATES, (1, 2, 3), strict=True):
                target = source.copy()
                ImageDraw.Draw(target).rectangle((100, 100, 199, 199), fill=(20 + offset, 80, 120))
                target_path = f"{split}/target/{pair_id}-{state}.png"
                target_raw = _png(target)
                files[target_path] = target_raw
                by_split[split].append(
                    {
                        "pair_id": f"{pair_id}-{state}",
                        "scene_pair_id": pair_id,
                        "product_id": product_id,
                        "split": split,
                        "state": state,
                        "prompt": payload._prompt(state),
                        "image": target_path,
                        "edit_image": source_path,
                        "mask": mask_path,
                        "bbox": [100, 100, 200, 200],
                        "source_sha256": hashlib.sha256(source_raw).hexdigest(),
                        "target_sha256": hashlib.sha256(target_raw).hexdigest(),
                        "mask_sha256": hashlib.sha256(mask_raw).hexdigest(),
                    }
                )
    train = sorted(by_split["train"], key=lambda row: (row["pair_id"], row["state"]))
    dev = sorted(by_split["dev"], key=lambda row: (row["pair_id"], row["state"]))
    raw = payload._archive_bytes(
        "controlled3d-test-001",
        manifest_sha,
        registry_sha,
        "3" * 64,
        "4" * 64,
        train,
        dev,
        files,
    )
    return raw, train, dev, files


def test_controlled3d_payload_uses_train_only_metadata_and_allows_profile_relighting(
    monkeypatch,
) -> None:
    from product_scene_editor import controlled3d_study_payload as payload

    raw, train, dev, _files = _controlled3d_payload(monkeypatch)
    parsed = payload.validate_controlled3d_study_payload_bytes(raw, "controlled3d-test-001")

    assert len(parsed.train_rows) == 18
    assert len(parsed.dev_rows) == 9
    assert [row["pair_id"] for row in parsed.train_rows] == [row["pair_id"] for row in train]
    assert [row["pair_id"] for row in parsed.dev_rows] == [row["pair_id"] for row in dev]
    metadata = json.loads(parsed.files["metadata.json"])
    assert len(metadata) == 18
    assert all(row["split"] == "train" for row in parsed.train_rows)
    assert not any("dev" in json.dumps(row) or "sealed_test" in json.dumps(row) for row in metadata)
    assert all(row["scene_pair_id"] != row["pair_id"] for row in (*train, *dev))


def test_payload_accepts_opaque_rgba_and_rejects_transparent_rgb_assets(monkeypatch) -> None:
    import hashlib
    from io import BytesIO

    from PIL import Image

    from product_scene_editor import controlled3d_study_payload as payload

    raw, train, dev, files = _controlled3d_payload(monkeypatch)
    source_path = train[0]["edit_image"]
    target_path = train[0]["image"]

    def rgba_bytes(path, alpha=255):
        with Image.open(BytesIO(files[path])) as source:
            image = source.convert("RGBA")
        if alpha != 255:
            image.putpixel((0, 0), (1, 2, 3, alpha))
        stream = BytesIO()
        image.save(stream, format="PNG")
        return stream.getvalue()

    files[source_path] = rgba_bytes(source_path)
    for row in train:
        if row["edit_image"] == source_path:
            row["source_sha256"] = hashlib.sha256(files[source_path]).hexdigest()
    files[target_path] = rgba_bytes(target_path)
    train[0]["target_sha256"] = hashlib.sha256(files[target_path]).hexdigest()
    valid = payload._archive_bytes(
        "controlled3d-test-001", "1" * 64, "2" * 64, "3" * 64, "4" * 64, train, dev, files
    )

    parsed = payload.validate_controlled3d_study_payload_bytes(valid, "controlled3d-test-001")
    assert parsed.files[source_path] == files[source_path]
    assert parsed.files[target_path] == files[target_path]

    files[target_path] = rgba_bytes(target_path, alpha=254)
    train[0]["target_sha256"] = hashlib.sha256(files[target_path]).hexdigest()
    transparent = payload._archive_bytes(
        "controlled3d-test-001", "1" * 64, "2" * 64, "3" * 64, "4" * 64, train, dev, files
    )
    with pytest.raises(ValueError, match="opaque"):
        payload.validate_controlled3d_study_payload_bytes(transparent, "controlled3d-test-001")


def test_controlled3d_payload_rejects_changed_asset_bytes(monkeypatch) -> None:
    from product_scene_editor import controlled3d_study_payload as payload

    raw, _train, _dev, files = _controlled3d_payload(monkeypatch)
    parsed = payload.validate_controlled3d_study_payload_bytes(raw, "controlled3d-test-001")
    files["dev/target/dev-pair-0-cream.png"] += b"tampered"
    altered = payload._archive_bytes(
        "controlled3d-test-001",
        "1" * 64,
        "2" * 64,
        "3" * 64,
        "4" * 64,
        list(parsed.train_rows),
        list(parsed.dev_rows),
        files,
    )
    with pytest.raises(ValueError, match="checksum"):
        payload.validate_controlled3d_study_payload_bytes(altered, "controlled3d-test-001")


def test_plan_and_reload_accept_a_generated_nine_product_candidate(tmp_path, monkeypatch) -> None:
    import hashlib

    from test_controlled3d import _make_case, _registry_bytes

    from product_scene_editor import controlled3d, controlled3d_schema

    root, manifest, registry = _make_case(tmp_path, ("train",) * 6 + ("dev",) * 3)
    manifest_raw = (controlled3d_schema._dump(manifest) + "\n").encode()
    registry_raw = _registry_bytes(registry)
    manifest_path = tmp_path / "manifest.json"
    registry_path = tmp_path / "registry.json"
    manifest_path.write_bytes(manifest_raw)
    registry_path.write_bytes(registry_raw)
    monkeypatch.setattr(
        controlled3d_study,
        "APPROVED_CONTROLLED3D_MANIFEST_SHA256",
        hashlib.sha256(manifest_raw).hexdigest(),
    )
    monkeypatch.setattr(
        controlled3d_study,
        "APPROVED_CONTROLLED3D_REGISTRY_SHA256",
        hashlib.sha256(registry_raw).hexdigest(),
    )
    validated = controlled3d.validate_controlled3d(
        root,
        manifest,
        reviewed_registry=registry_raw,
        registry_sha256=hashlib.sha256(registry_raw).hexdigest(),
    )
    train_export = tmp_path / "train-export"
    controlled3d.export_train_only(root, validated, train_export)

    plan = controlled3d_study.plan_controlled3d_study(
        root,
        manifest_path,
        registry_path,
        train_export,
        tmp_path / "plan",
        run_id="controlled3d-plan-001",
    )
    loaded = controlled3d_study.load_controlled3d_study_plan(plan.output_dir)

    assert loaded.source_dataset_manifest_sha256 == hashlib.sha256(manifest_raw).hexdigest()
    assert loaded.source_registry_sha256 == hashlib.sha256(registry_raw).hexdigest()
    assert loaded.validation_proof_sha256 == plan.validation_proof_sha256
    assert loaded.train_export_completion_sha256 == plan.train_export_completion_sha256
    assert len(loaded.training_pair_ids) == 18
    assert len(loaded.development_pair_ids) == 9
