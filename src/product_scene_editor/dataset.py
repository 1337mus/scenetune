"""Strict manifests and safe exports for original synthetic scene pairs."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, ImageChops

_SCHEMA_FIELDS = {"schema_version", "provenance", "pairs"}
_PROVENANCE_FIELDS = {"image_source", "external_assets", "employer_input", "license"}
_PAIR_FIELDS = {
    "product_id",
    "pair_id",
    "split",
    "prompt",
    "source_path",
    "target_path",
    "mask_path",
    "bbox",
    "generator_seed",
    "data_license",
    "sha256",
}
_PROVENANCE = {
    "image_source": "original_synthetic",
    "external_assets": [],
    "employer_input": "none",
    "license": "CC0-1.0",
}
_SPLITS = {"train", "dev", "sealed_test"}
_ASSET_FIELDS = {"source": "source_path", "target": "target_path", "mask": "mask_path"}
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _unique_keys(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _resolve_asset(root: Path, relative: str, split: str, role: str) -> Path:
    path = PurePosixPath(relative)
    folder = "masks" if role == "mask" else role
    if path.is_absolute() or ".." in path.parts or path.parts[:2] != (split, folder):
        raise ValueError(f"unsafe asset path: {relative!r}")
    resolved = (root / Path(*path.parts)).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"unsafe asset path: {relative!r}") from exc
    return resolved


def _load_manifest(root: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(
            (root / "manifest.json").read_text(encoding="utf-8"),
            object_pairs_hook=_unique_keys,
            parse_constant=_invalid_constant,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("manifest.json is missing or invalid JSON") from exc
    if not isinstance(manifest, dict) or set(manifest) != _SCHEMA_FIELDS:
        raise ValueError("manifest fields are missing or unknown")
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported manifest schema_version")
    provenance = manifest["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != _PROVENANCE_FIELDS:
        raise ValueError("provenance fields are missing or unknown")
    if provenance != _PROVENANCE:
        raise ValueError("manifest provenance must describe original synthetic CC0 data")
    if not isinstance(manifest["pairs"], list) or not manifest["pairs"]:
        raise ValueError("manifest pairs must be a non-empty array")
    return manifest


def _verify_pixels(paths: dict[str, Path], pair: dict[str, Any]) -> str:
    with Image.open(paths["source"]) as image:
        source = image.convert("RGB")
    with Image.open(paths["target"]) as image:
        target = image.convert("RGB")
    with Image.open(paths["mask"]) as image:
        mask = image.convert("L")
    if source.size != (512, 512) or target.size != source.size or mask.size != source.size:
        raise ValueError(f"pair {pair['pair_id']!r} images must all be 512x512")
    if any(value not in (0, 255) for value in mask.tobytes()):
        raise ValueError(f"pair {pair['pair_id']!r} mask must be binary")
    if tuple(pair["bbox"]) != mask.getbbox():
        raise ValueError(f"pair {pair['pair_id']!r} mask bbox does not match manifest")
    diff = ImageChops.difference(source, target)
    black = Image.new("RGB", source.size)
    if ImageChops.composite(diff, black, mask).getbbox():
        raise ValueError(f"pair {pair['pair_id']!r} product foreground differs")
    if not ImageChops.composite(diff, black, ImageChops.invert(mask)).getbbox():
        raise ValueError(f"pair {pair['pair_id']!r} has no background scene change")
    foreground = ImageChops.composite(source, black, mask)
    return hashlib.sha256(mask.tobytes() + foreground.tobytes()).hexdigest()


def verify_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    """Validate strict JSON, split isolation, asset integrity, and foreground pairing."""
    root, manifest = Path(dataset_dir), _load_manifest(Path(dataset_dir))
    product_splits: dict[str, str] = {}
    pair_ids: set[str] = set()
    foreground_splits: dict[str, str] = {}
    for pair in manifest["pairs"]:
        if not isinstance(pair, dict) or set(pair) != _PAIR_FIELDS:
            raise ValueError("pair fields are missing or unknown")
        product_id, pair_id, split = pair["product_id"], pair["pair_id"], pair["split"]
        if not all(isinstance(item, str) and item for item in (product_id, pair_id)):
            raise ValueError("product_id and pair_id must be non-empty strings")
        if not isinstance(split, str) or split not in _SPLITS:
            raise ValueError(f"unknown split: {split!r}")
        if product_id in product_splits and product_splits[product_id] != split:
            raise ValueError(f"product identity {product_id!r} crosses splits")
        product_splits[product_id] = split
        if pair_id in pair_ids:
            raise ValueError(f"duplicate pair_id: {pair_id!r}")
        pair_ids.add(pair_id)
        if (
            not isinstance(pair["prompt"], str)
            or "keep the package unchanged" not in pair["prompt"].lower()
        ):
            raise ValueError(f"pair {pair_id!r} prompt must preserve the package")
        seed, bbox = pair["generator_seed"], pair["bbox"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"pair {pair_id!r} generator_seed must be an integer")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(isinstance(value, bool) or not isinstance(value, int) for value in bbox)
        ):
            raise ValueError(f"pair {pair_id!r} bbox must contain four integers")
        if pair["data_license"] != "CC0-1.0":
            raise ValueError(f"pair {pair_id!r} data license is not CC0-1.0")
        digests, paths = pair["sha256"], {}
        if not isinstance(digests, dict) or set(digests) != set(_ASSET_FIELDS):
            raise ValueError(f"pair {pair_id!r} sha256 fields are invalid")
        for role, field in _ASSET_FIELDS.items():
            relative = pair[field]
            if not isinstance(relative, str):
                raise ValueError(f"pair {pair_id!r} {field} must be a string")
            path, digest = _resolve_asset(root, relative, split, role), digests[role]
            if not isinstance(digest, str) or not _DIGEST_PATTERN.fullmatch(digest):
                raise ValueError(f"pair {pair_id!r} {role} checksum is invalid")
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"checksum mismatch for pair {pair_id!r} {role}")
            paths[role] = path
        fingerprint = _verify_pixels(paths, pair)
        previous = foreground_splits.setdefault(fingerprint, split)
        if previous != split:
            raise ValueError("product foreground fingerprint crosses splits")
    return manifest


def build_dataset(
    output_dir: str | Path, seed: int = 20261008, products: int = 12
) -> dict[str, Any]:
    """Create identity-separated 50/25/25 pools with three target scenes per product."""
    from .render import SCENES, render_pair

    root = Path(output_dir)
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(products, bool) or not isinstance(products, int) or products < 4 or products % 4:
        raise ValueError("products must be a positive multiple of four (at least four)")
    root.mkdir(parents=True)
    pairs: list[dict[str, Any]] = []
    try:
        for index in range(products):
            split = (
                "train"
                if index < products // 2
                else "dev"
                if index < products * 3 // 4
                else "sealed_test"
            )
            product_id = f"pack-{index:03d}"
            for scene in SCENES:
                pair_id, product_seed = f"{product_id}-{scene}", seed + index
                rendered = render_pair(product_id, product_seed, scene)
                assets = {
                    "source": (rendered.source, f"{split}/source/{pair_id}.png"),
                    "target": (rendered.target, f"{split}/target/{pair_id}.png"),
                    "mask": (rendered.mask, f"{split}/masks/{pair_id}.png"),
                }
                digests: dict[str, str] = {}
                for role, (image, relative) in assets.items():
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(path)
                    digests[role] = hashlib.sha256(path.read_bytes()).hexdigest()
                pairs.append(
                    {
                        "product_id": product_id,
                        "pair_id": pair_id,
                        "split": split,
                        "prompt": rendered.prompt,
                        "source_path": assets["source"][1],
                        "target_path": assets["target"][1],
                        "mask_path": assets["mask"][1],
                        "bbox": list(rendered.bbox),
                        "generator_seed": product_seed,
                        "data_license": "CC0-1.0",
                        "sha256": digests,
                    }
                )
        manifest = {"schema_version": 1, "provenance": _PROVENANCE, "pairs": pairs}
        (root / "manifest.json").write_text(
            json.dumps(manifest, allow_nan=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        return verify_dataset(root)
    except BaseException:
        shutil.rmtree(root)
        raise


def export_training(dataset_dir: str | Path, destination: str | Path) -> Path:
    """Copy only train assets and write DiffSynth image/edit_image/prompt metadata."""
    source, manifest = Path(dataset_dir), verify_dataset(dataset_dir)
    train_pairs = [pair for pair in manifest["pairs"] if pair["split"] == "train"]
    if not train_pairs:
        raise ValueError("dataset has no train pairs to export")
    output = Path(destination)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    try:
        rows: list[dict[str, str]] = []
        for pair in train_pairs:
            for field in _ASSET_FIELDS.values():
                relative = pair[field]
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / relative, target)
            rows.append(
                {
                    "image": pair["target_path"],
                    "edit_image": pair["source_path"],
                    "prompt": pair["prompt"],
                }
            )
        training_manifest = {**manifest, "pairs": train_pairs}
        for name, value in (("manifest.json", training_manifest), ("metadata.json", rows)):
            (output / name).write_text(
                json.dumps(value, allow_nan=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
        verify_dataset(output)
    except BaseException:
        shutil.rmtree(output)
        raise
    return output
