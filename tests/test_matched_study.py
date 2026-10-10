from __future__ import annotations

import hashlib
import json
import math
import zipfile
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from product_scene_editor import matched_study


def test_score_pair_uses_mask_regions_and_max_channel_thresholds() -> None:
    source = Image.new("RGB", (512, 512), (10, 20, 30))
    target = source.copy()
    generated = source.copy()
    mask = Image.new("L", source.size, 0)
    mask.putpixel((0, 0), 255)
    mask.putpixel((1, 0), 255)

    generated.putpixel((0, 0), (19, 20, 30))
    generated.putpixel((1, 0), (13, 26, 30))
    generated.putpixel((2, 0), (19, 20, 30))
    target.putpixel((3, 0), (13, 23, 35))

    metrics = matched_study.score_pair(source, target, mask, generated)
    background_pixels = 512 * 512 - 2

    assert set(metrics) == {
        "foreground_mean_absolute_error",
        "foreground_fraction_pixels_over_8",
        "background_target_mean_absolute_error",
        "background_fraction_changed_from_source_over_8",
        "foreground_pixels",
        "background_pixels",
    }
    assert metrics["foreground_mean_absolute_error"] == 3.0
    assert metrics["foreground_fraction_pixels_over_8"] == 0.5
    assert math.isclose(
        metrics["background_target_mean_absolute_error"],
        20 / (3 * background_pixels),
    )
    assert metrics["background_fraction_changed_from_source_over_8"] == (1 / background_pixels)
    assert metrics["foreground_pixels"] == 2
    assert metrics["background_pixels"] == background_pixels


_MANIFEST_PROVENANCE = {
    "image_source": "original_synthetic",
    "external_assets": [],
    "employer_input": "none",
    "license": "CC0-1.0",
}


def _write_matched_dataset(root: Path) -> str:
    pairs: list[dict[str, object]] = []
    for split, product_count in (("train", 6), ("dev", 3), ("sealed_test", 3)):
        for product_number in range(product_count):
            product_id = f"{split}-{product_number:03d}"
            for scene in range(3):
                pair_id = f"{product_id}-scene-{scene:02d}"
                if split == "sealed_test":
                    paths = {
                        role: f"../poison/{pair_id}.png" for role in ("source", "target", "mask")
                    }
                    digests = {role: "not-a-digest" for role in paths}
                else:
                    paths = {
                        "source": f"{split}/source/{pair_id}.png",
                        "target": f"{split}/target/{pair_id}.png",
                        "mask": f"{split}/masks/{pair_id}.png",
                    }
                    source = Image.new("RGB", (512, 512), (198, 207, 215))
                    target = Image.new("RGB", (512, 512), (224, 201, 178))
                    mask = Image.new("L", (512, 512), 0)
                    bbox = (190, 130, 322, 366)
                    split_offset = 72 if split == "dev" else 0
                    color = (52 + split_offset + product_number * 12, 112, 120)
                    for image in (source, target):
                        ImageDraw.Draw(image).rectangle(
                            (bbox[0], bbox[1], bbox[2] - 1, bbox[3] - 1), fill=color
                        )
                    ImageDraw.Draw(mask).rectangle(
                        (bbox[0], bbox[1], bbox[2] - 1, bbox[3] - 1), fill=255
                    )
                    digests: dict[str, str] = {}
                    for role, image in (("source", source), ("target", target), ("mask", mask)):
                        asset_path = root / paths[role]
                        asset_path.parent.mkdir(parents=True, exist_ok=True)
                        image.save(asset_path, format="PNG")
                        digests[role] = hashlib.sha256(asset_path.read_bytes()).hexdigest()
                pairs.append(
                    {
                        "product_id": product_id,
                        "pair_id": pair_id,
                        "split": split,
                        "prompt": "Move the package and keep the package unchanged.",
                        "source_path": paths["source"],
                        "target_path": paths["target"],
                        "mask_path": paths["mask"],
                        "bbox": [190, 130, 322, 366],
                        "generator_seed": 17,
                        "data_license": "CC0-1.0",
                        "sha256": digests,
                    }
                )
    root.mkdir(parents=True, exist_ok=True)
    manifest_bytes = (
        json.dumps(
            {"schema_version": 1, "provenance": _MANIFEST_PROVENANCE, "pairs": pairs},
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    (root / "manifest.json").write_bytes(manifest_bytes)
    return hashlib.sha256(manifest_bytes).hexdigest()


def test_plan_builds_stable_isolated_18_train_9_dev_payload_without_reading_sealed_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    manifest_sha256 = _write_matched_dataset(dataset_dir)
    monkeypatch.setattr(
        matched_study,
        "APPROVED_DATASET_MANIFEST_SHA256",
        manifest_sha256,
        raising=False,
    )

    plan = matched_study.plan_matched_study(dataset_dir, tmp_path / "run-001", run_id="matched-001")

    assert len(plan.training_pair_ids) == 18
    assert len(plan.development_pair_ids) == 9
    assert plan.training_pair_ids == tuple(sorted(plan.training_pair_ids))
    assert plan.development_pair_ids == tuple(sorted(plan.development_pair_ids))
    assert plan.source_dataset_manifest_sha256 == manifest_sha256
    assert plan.payload_bytes == plan.payload_path.stat().st_size
    payload_bytes = plan.payload_path.read_bytes()
    assert plan.payload_sha256 == hashlib.sha256(payload_bytes).hexdigest()
    with zipfile.ZipFile(plan.payload_path) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)) == 83
        assert set(names) == {
            "metadata.json",
            "matched_manifest.json",
            *(
                f"{split}/{kind}/{pair_id}.png"
                for split, pair_ids in (
                    ("train", plan.training_pair_ids),
                    ("dev", plan.development_pair_ids),
                )
                for pair_id in pair_ids
                for kind in ("source", "target", "mask")
            ),
        }
        assert all("sealed_test" not in name for name in names)
        study_manifest = json.loads(archive.read("matched_manifest.json"))
        metadata = json.loads(archive.read("metadata.json"))
    assert set(study_manifest) == {
        "schema_version",
        "run_id",
        "source_dataset_manifest_sha256",
        "train_rows",
        "dev_rows",
        "product_counts",
        "training_epochs",
        "training_steps",
    }
    assert study_manifest["training_epochs"] == 8
    assert study_manifest["training_steps"] == 144
    assert not any(
        "sealed_test" in str(row)
        for key in ("train_rows", "dev_rows")
        for row in study_manifest[key]
    )
    train_rows = study_manifest["train_rows"]
    dev_rows = study_manifest["dev_rows"]
    expected_row_fields = {
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
    assert all(set(row) == expected_row_fields for row in (*train_rows, *dev_rows))
    assert study_manifest["product_counts"] == {"dev": 3, "sealed_test": 3, "train": 6}
    assert [row["pair_id"] for row in train_rows] == list(plan.training_pair_ids)
    assert [row["pair_id"] for row in dev_rows] == list(plan.development_pair_ids)
    assert len(metadata) == 18
    assert all(set(row) == {"image", "edit_image", "prompt"} for row in metadata)
    assert metadata == [
        {key: row[key] for key in ("image", "edit_image", "prompt")} for row in train_rows
    ]
    assert {row["product_id"] for row in train_rows}.isdisjoint(
        row["product_id"] for row in dev_rows
    )
    assert (plan.output_dir / matched_study.RUN_MARKER_NAME).is_file()
    loaded_plan = matched_study.load_matched_study_plan(plan.output_dir)
    assert loaded_plan == plan
    payload = matched_study.read_matched_study_payload(loaded_plan)
    assert len(payload.train_rows) == 18
    assert len(payload.dev_rows) == 9
    assert len(payload.files) == 83
    assert sum(name.endswith(".png") for name in payload.files) == 81
    with pytest.raises(FileExistsError):
        matched_study.plan_matched_study(dataset_dir, plan.output_dir, run_id="matched-001")


def test_manifest_digest_pin_is_the_approved_dataset_manifest() -> None:
    assert matched_study.APPROVED_DATASET_MANIFEST_SHA256 == (
        "206b07f99b2fd1c001b94898f3cacfa7010bacb7e08aa731b972135000e9a8f9"
    )


def test_read_payload_rejects_payload_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    manifest_sha256 = _write_matched_dataset(dataset_dir)
    monkeypatch.setattr(matched_study, "APPROVED_DATASET_MANIFEST_SHA256", manifest_sha256)
    plan = matched_study.plan_matched_study(
        dataset_dir, tmp_path / "run-tampered", run_id="matched-tampered"
    )

    payload_bytes = plan.payload_path.read_bytes()
    plan.payload_path.write_bytes(payload_bytes[:-1] + bytes([payload_bytes[-1] ^ 1]))

    with pytest.raises(ValueError, match="checksum|marker"):
        matched_study.read_matched_study_payload(plan)


def test_planning_never_opens_sealed_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    manifest_sha256 = _write_matched_dataset(dataset_dir)
    monkeypatch.setattr(matched_study, "APPROVED_DATASET_MANIFEST_SHA256", manifest_sha256)

    plan = matched_study.plan_matched_study(
        dataset_dir, tmp_path / "run-withheld", run_id="matched-withheld"
    )

    assert len(plan.development_pair_ids) == 9
    assert not (dataset_dir / "sealed_test").exists()


def test_planner_rejects_unsafe_train_asset_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_matched_dataset(dataset_dir)
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first_train = next(pair for pair in manifest["pairs"] if pair["split"] == "train")
    first_train["source_path"] = "train/source/../../sealed_test/poison.png"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    monkeypatch.setattr(matched_study, "APPROVED_DATASET_MANIFEST_SHA256", manifest_sha256)

    with pytest.raises(ValueError, match="unsafe asset path"):
        matched_study.plan_matched_study(
            dataset_dir, tmp_path / "run-unsafe", run_id="matched-unsafe"
        )


def test_planner_rejects_foreground_changes_even_with_updated_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_matched_dataset(dataset_dir)
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pair = next(pair for pair in manifest["pairs"] if pair["split"] == "train")
    target_path = dataset_dir / pair["target_path"]
    with Image.open(target_path) as opened:
        target = opened.copy()
    target.putpixel((200, 140), (1, 2, 3))
    target.save(target_path, format="PNG")
    pair["sha256"]["target"] = hashlib.sha256(target_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    monkeypatch.setattr(matched_study, "APPROVED_DATASET_MANIFEST_SHA256", manifest_sha256)

    with pytest.raises(ValueError, match="foreground differs"):
        matched_study.plan_matched_study(
            dataset_dir, tmp_path / "run-changed-foreground", run_id="matched-changed"
        )


def test_planner_rejects_non_binary_mask_even_with_updated_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_matched_dataset(dataset_dir)
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pair = next(pair for pair in manifest["pairs"] if pair["split"] == "train")
    mask_path = dataset_dir / pair["mask_path"]
    with Image.open(mask_path) as opened:
        mask = opened.copy()
    mask.putpixel((1, 1), 128)
    mask.save(mask_path, format="PNG")
    pair["sha256"]["mask"] = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    monkeypatch.setattr(matched_study, "APPROVED_DATASET_MANIFEST_SHA256", manifest_sha256)

    with pytest.raises(ValueError, match="mask must be binary"):
        matched_study.plan_matched_study(
            dataset_dir, tmp_path / "run-non-binary", run_id="matched-non-binary"
        )
