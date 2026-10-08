import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from product_scene_editor import dataset


def _write_fixture(
    root: Path, assignments: list[tuple[str, str, str]], same_foreground: bool = False
) -> None:
    pairs: list[dict[str, object]] = []
    for pair_id, product_id, split in assignments:
        source_rel = f"{split}/source/{pair_id}.png"
        target_rel = f"{split}/target/{pair_id}.png"
        mask_rel = f"{split}/masks/{pair_id}.png"
        source = Image.new("RGB", (512, 512), (198, 207, 215))
        target = Image.new("RGB", (512, 512), (224, 201, 178))
        mask = Image.new("L", (512, 512), 0)
        bbox = (190, 130, 322, 366)
        suffix = product_id.rsplit("-", 1)[-1]
        variation = int(suffix) if suffix.isdigit() else sum(product_id.encode()) % 180
        product_color = (52 if same_foreground else 52 + variation, 112, 120)
        for image, color in ((source, product_color), (target, product_color)):
            ImageDraw.Draw(image).rectangle(
                (bbox[0], bbox[1], bbox[2] - 1, bbox[3] - 1), fill=color
            )
        ImageDraw.Draw(mask).rectangle((bbox[0], bbox[1], bbox[2] - 1, bbox[3] - 1), fill=255)
        files = {"source": source, "target": target, "mask": mask}
        paths = {"source": source_rel, "target": target_rel, "mask": mask_rel}
        digests: dict[str, str] = {}
        for key, image in files.items():
            path = root / paths[key]
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
            digests[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        pairs.append(
            {
                "product_id": product_id,
                "pair_id": pair_id,
                "split": split,
                "prompt": (
                    "Place the unchanged package in another studio; keep the package unchanged."
                ),
                "source_path": source_rel,
                "target_path": target_rel,
                "mask_path": mask_rel,
                "bbox": list(bbox),
                "generator_seed": 17,
                "data_license": "CC0-1.0",
                "sha256": digests,
            }
        )
    manifest = {
        "schema_version": 1,
        "provenance": {
            "image_source": "original_synthetic",
            "external_assets": [],
            "employer_input": "none",
            "license": "CC0-1.0",
        },
        "pairs": pairs,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_verify_dataset_rejects_product_identity_leaking_across_splits(tmp_path: Path) -> None:
    _write_fixture(
        tmp_path,
        [("leaked-train", "pack-007", "train"), ("leaked-test", "pack-007", "sealed_test")],
    )

    with pytest.raises(ValueError, match="product identity.*crosses splits"):
        dataset.verify_dataset(tmp_path)


def test_verify_dataset_rejects_mismatched_product_foreground(tmp_path: Path) -> None:
    _write_fixture(tmp_path, [("pair-0", "pack-000", "train")])
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pair = manifest["pairs"][0]
    target_path = tmp_path / pair["target_path"]
    with Image.open(target_path) as opened:
        target = opened.convert("RGB")
    target.putpixel((220, 200), (250, 3, 9))
    target.save(target_path)
    pair["sha256"]["target"] = hashlib.sha256(target_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="product foreground differs"):
        dataset.verify_dataset(tmp_path)


def test_verify_dataset_rejects_asset_checksum_mismatch(tmp_path: Path) -> None:
    _write_fixture(tmp_path, [("pair-0", "pack-000", "train")])
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    target_path = tmp_path / manifest["pairs"][0]["target_path"]
    with Image.open(target_path) as opened:
        target = opened.convert("RGB")
    target.putpixel((0, 0), (2, 4, 6))
    target.save(target_path)

    with pytest.raises(ValueError, match="checksum mismatch"):
        dataset.verify_dataset(tmp_path)


def test_verify_dataset_rejects_unknown_manifest_fields(tmp_path: Path) -> None:
    _write_fixture(tmp_path, [("pair-0", "pack-000", "train")])
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pairs"][0]["unexpected"] = "ignored fields break the manifest contract"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="pair fields are missing or unknown"):
        dataset.verify_dataset(tmp_path)


def test_export_training_contains_only_train_images_and_metadata(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_fixture(
        dataset_root,
        [
            ("pair-train", "pack-000", "train"),
            ("pair-dev", "pack-001", "dev"),
            ("pair-test", "pack-002", "sealed_test"),
        ],
    )
    output = tmp_path / "export"

    dataset.export_training(dataset_root, output)

    rows = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    exported_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert len(rows) == len(exported_manifest["pairs"]) == 1
    assert set(rows[0]) == {"image", "edit_image", "prompt"}
    assert rows[0]["image"] == "train/target/pair-train.png"
    assert rows[0]["edit_image"] == "train/source/pair-train.png"
    assert exported_manifest["pairs"][0]["split"] == "train"
    assert (output / rows[0]["image"]).is_file()
    assert (output / rows[0]["edit_image"]).is_file()
    assert not (output / "dev").exists()
    assert not (output / "sealed_test").exists()


def test_verify_dataset_rejects_asset_path_traversal(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_fixture(dataset_root, [("pair-0", "pack-000", "train")])
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    escaped = tmp_path / "escaped.png"
    escaped.write_bytes((dataset_root / "train/source/pair-0.png").read_bytes())
    manifest["pairs"][0]["source_path"] = "../escaped.png"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe asset path"):
        dataset.verify_dataset(dataset_root)


def test_export_training_refuses_to_overwrite_existing_destination(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_fixture(dataset_root, [("pair-0", "pack-000", "train")])
    output = tmp_path / "export"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        dataset.export_training(dataset_root, output)

    assert marker.read_text(encoding="utf-8") == "existing"


def test_verify_dataset_rejects_product_pixels_reassigned_to_another_id_across_splits(
    tmp_path: Path,
) -> None:
    _write_fixture(
        tmp_path,
        [("pixel-train", "pack-000", "train"), ("pixel-test", "renamed-999", "sealed_test")],
        same_foreground=True,
    )

    with pytest.raises(ValueError, match="product foreground.*crosses splits"):
        dataset.verify_dataset(tmp_path)
