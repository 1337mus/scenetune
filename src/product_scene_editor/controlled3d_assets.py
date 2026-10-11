"""Bounded CPU asset checks and explicit train-only publication helpers."""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
import stat
import tempfile
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter

_MAX_ASSET_BYTES = 32 * 1024 * 1024
_STATES = ("source", "cream", "sage", "clay")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def root_identity(root: str | Path) -> tuple[str, int, int]:
    absolute = os.path.abspath(os.fspath(root))
    info = os.lstat(absolute)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError("dataset root must be a real directory")
    return absolute, info.st_dev, info.st_ino


def read_asset(root: str | Path, relative: str) -> bytes:
    base = os.path.abspath(os.fspath(root))
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    root_fd = os.open(base, os.O_RDONLY | directory | nofollow)
    current_fd = root_fd
    try:
        parts = relative.split("/")
        for index, part in enumerate(parts):
            flags = os.O_RDONLY | nofollow
            if index < len(parts) - 1:
                flags |= directory
            else:
                flags |= getattr(os, "O_NONBLOCK", 0)
            child_fd = os.open(part, flags, dir_fd=current_fd)
            if current_fd != root_fd:
                os.close(current_fd)
            current_fd = child_fd
        info = os.fstat(current_fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"asset is not a regular file: {relative}")
        if info.st_size > _MAX_ASSET_BYTES:
            raise ValueError(f"asset exceeds size limit: {relative}")
        chunks: list[bytes] = []
        remaining = _MAX_ASSET_BYTES + 1
        while remaining:
            chunk = os.read(current_fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > _MAX_ASSET_BYTES:
            raise ValueError(f"asset exceeds size limit: {relative}")
        return raw
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(f"symlink in asset path: {relative}") from exc
        raise ValueError(f"cannot safely read asset: {relative}") from exc
    finally:
        if current_fd != root_fd:
            os.close(current_fd)
        os.close(root_fd)


def _decode_image(raw: bytes, label: str) -> bytes:
    try:
        image = Image.open(BytesIO(raw))
        if (
            image.format != "PNG"
            or image.size != (512, 512)
            or image.mode not in ("RGB", "RGBA")
            or getattr(image, "n_frames", 1) != 1
        ):
            raise ValueError(f"{label} must be a single-frame 512x512 RGB/RGBA PNG")
        image.load()
        if image.mode == "RGBA" and image.getchannel("A").getextrema() != (255, 255):
            raise ValueError(f"{label} alpha must be opaque")
        return image.convert("RGB").tobytes()
    except (OSError, SyntaxError) as exc:
        raise ValueError(f"invalid PNG: {label}") from exc


def _decode_mask(raw: bytes, row: dict[str, Any], label: str) -> tuple[bytes, bytes]:
    try:
        mask = Image.open(BytesIO(raw))
        if (
            mask.format != "PNG"
            or mask.size != (512, 512)
            or mask.mode != "L"
            or getattr(mask, "n_frames", 1) != 1
        ):
            raise ValueError(f"{label} must be a single-frame 512x512 grayscale PNG mask")
        mask.load()
        pixels = mask.tobytes()
    except (OSError, SyntaxError) as exc:
        raise ValueError(f"invalid mask PNG: {label}") from exc
    if any(value not in (0, 255) for value in pixels):
        raise ValueError(f"mask must contain only 0/255 values: {label}")
    if _sha(pixels) != row["mask"]["decoded_sha256"]:
        raise ValueError(f"decoded mask hash mismatch: {label}")
    coords = [index for index, value in enumerate(pixels) if value]
    if not coords:
        raise ValueError(f"mask is empty: {label}")
    xs, ys = [i % 512 for i in coords], [i // 512 for i in coords]
    if [min(xs), min(ys), max(xs) + 1, max(ys) + 1] != row["bbox"]:
        raise ValueError(f"mask bbox mismatch: {label}")
    if any(pixels[x] or pixels[511 * 512 + x] for x in range(512)) or any(
        pixels[y * 512] or pixels[y * 512 + 511] for y in range(512)
    ):
        raise ValueError(f"mask outline is clipped: {label}")
    eroded = Image.frombytes("L", (512, 512), pixels).filter(ImageFilter.MinFilter(5)).tobytes()
    core = bytes(255 if pixels[i] and eroded[i] == 255 else 0 for i in range(len(pixels)))
    if not any(core):
        raise ValueError(f"mask has no 5x5 product core: {label}")
    return pixels, core


def _snapshot(raw: bytes, pair: dict[str, Any], loads_any: Callable[[bytes | str], Any]) -> None:
    from product_scene_editor.studio_state import validate_studio_states

    records = loads_any(raw)
    validate_studio_states(records)
    pins = pair["pins"]["font"]
    expected_font = {
        "source_url": pins["source_url"],
        "file_sha256": pins["file_sha256"],
        "license_id": pins["license_id"],
        "license_sha256": pins["license_sha256"],
    }
    for row in records:
        snapshot = row["snapshot"]
        if (
            snapshot["label"]["words"] != pair["expected_words"]
            or snapshot["label"]["font"] != expected_font
        ):
            raise ValueError(
                f"captured label words or font do not match reviewed pins: {pair['pair_id']}"
            )


def validate_assets(
    root: str | Path,
    manifest: dict[str, Any],
    refs: list[tuple[str, str, str, str]],
    pair_rows: dict[str, list[dict[str, Any]]],
    loads_any: Callable[[bytes | str], Any],
) -> dict[str, Any]:
    loaded: dict[str, bytes] = {}
    image_hash_pool: dict[str, str] = {}
    for split, path, expected, role in refs:
        if split == "sealed_test":
            continue
        raw = read_asset(root, path)
        if _sha(raw) != expected:
            raise ValueError(f"asset hash mismatch: {path}")
        loaded[path] = raw
        if role == "image":
            if expected in image_hash_pool and image_hash_pool[expected] != split:
                raise ValueError("byte-identical image crosses nonsealed pools")
            image_hash_pool[expected] = split
    source_sha = _sha(Path(__file__).with_name("studio_state.py").read_bytes())
    metrics: dict[str, Any] = {"pairs": {}}
    product_pixels: dict[str, str] = {}
    for pair in manifest["pairs"]:
        if pair["split"] == "sealed_test":
            continue
        if pair["pins"]["validator_source_sha256"] != source_sha:
            raise ValueError("validator source pin differs from installed shared validator")
        for row in pair["pins"]["local_files"]:
            if _sha(loaded[row["path"]]) != row["sha256"]:
                raise ValueError(f"local pin file hash mismatch: {row['path']}")
        _snapshot(loaded[pair["snapshot_bundle"]["path"]], pair, loads_any)
        rows = pair_rows[pair["pair_id"]]
        mask_pixels: bytes | None = None
        core_pixels: bytes | None = None
        rgb: dict[str, bytes] = {}
        for row in rows:
            name = row["name"]
            mask, core = _decode_mask(loaded[row["mask"]["path"]], row, f"{pair['pair_id']}:{name}")
            if mask_pixels is not None and mask != mask_pixels:
                raise ValueError(f"masks differ across states: {pair['pair_id']}")
            mask_pixels, core_pixels = mask, core
            rgb[name] = _decode_image(loaded[row["image"]["path"]], f"{pair['pair_id']}:{name}")
        assert mask_pixels is not None and core_pixels is not None
        product_digest = _sha(
            bytes(
                channel
                for i, value in enumerate(mask_pixels)
                if value
                for channel in rgb["source"][i * 3 : i * 3 + 3]
            )
        )
        if product_digest in product_pixels and product_pixels[product_digest] != pair["split"]:
            raise ValueError("decoded product pixels cross nonsealed pools")
        product_pixels[product_digest] = pair["split"]
        regions = {
            "product": [bool(value) for value in mask_pixels],
            "core": [bool(value) for value in core_pixels],
            "edge": [bool(mask_pixels[i] and not core_pixels[i]) for i in range(len(mask_pixels))],
            "background": [not bool(value) for value in mask_pixels],
        }
        changes: dict[str, dict[str, int]] = {}
        source = rgb["source"]
        for name in ("cream", "sage", "clay"):
            target = rgb[name]
            diff = [
                source[i] != target[i]
                or source[i + 1] != target[i + 1]
                or source[i + 2] != target[i + 2]
                for i in range(0, len(source), 3)
            ]
            changes[name] = {
                region: sum(
                    changed and inside for changed, inside in zip(diff, membership, strict=True)
                )
                for region, membership in regions.items()
            }
        metrics["pairs"][pair["pair_id"]] = {"split": pair["split"], "rgb_changes": changes}
    return metrics


def _reject_destination_path(path: str) -> None:
    path_obj = Path(path)
    current = Path(path_obj.anchor)
    for part in path_obj.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError("destination path has a symlink component")


def export_train_only(
    root: str | Path,
    validated: Any,
    destination: str | Path,
    manifest: dict[str, Any],
    revalidate: Callable[..., Any],
    dump: Callable[[Any], str],
) -> Path:
    """Revalidate, copy train assets, and publish a completion receipt last."""
    rechecked = revalidate(
        root,
        manifest,
        reviewed_registry=validated.canonical_registry,
        registry_sha256=validated.registry_sha256,
    )
    if rechecked.root_identity != validated.root_identity:
        raise ValueError("dataset root changed after validation")
    root_path, target = os.path.abspath(os.fspath(root)), os.path.abspath(os.fspath(destination))
    if os.path.commonpath((root_path, target)) in (root_path, target):
        raise ValueError("export destination must not overlap dataset root")
    _reject_destination_path(target)
    parent = os.path.dirname(target)
    if not os.path.isdir(parent):
        raise ValueError("export destination parent must exist")
    if os.path.lexists(target):
        raise ValueError("export destination already exists")
    stage = tempfile.mkdtemp(prefix=".controlled3d-stage-", dir=parent)
    claimed = False
    try:
        train_pairs = [pair for pair in manifest["pairs"] if pair["split"] == "train"]
        if not train_pairs:
            raise ValueError("no train pairs to export")
        files: dict[str, str] = {}
        for pair in train_pairs:
            refs = [
                pair["snapshot_bundle"],
                *[x for row in pair["renders"] for x in (row["image"], row["mask"])],
                *pair["pins"]["local_files"],
            ]
            for ref in refs:
                raw = read_asset(root, ref["path"])
                if _sha(raw) != ref["sha256"]:
                    raise ValueError(f"asset changed while exporting: {ref['path']}")
                output = Path(stage, ref["path"])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(raw)
                if _sha(output.read_bytes()) != ref["sha256"]:
                    raise ValueError(f"staged asset verification failed: {ref['path']}")
                files[ref["path"]] = ref["sha256"]
        exported = {
            "schema_version": 1,
            "provenance": manifest["provenance"],
            "assignments": [row for row in manifest["assignments"] if row["split"] == "train"],
            "pairs": train_pairs,
        }
        metadata: list[dict[str, str]] = []
        for pair in train_pairs:
            renders_by_name = {row["name"]: row for row in pair["renders"]}
            source_image = renders_by_name["source"]["image"]["path"]
            for name in _STATES[1:]:
                metadata.append(
                    {
                        "image": renders_by_name[name]["image"]["path"],
                        "edit_image": source_image,
                        "prompt": (
                            "Keep the product, label, and geometry unchanged; "
                            f"change only the background to the {name} variant."
                        ),
                    }
                )
        for name, value in (("manifest.json", exported), ("metadata.json", metadata)):
            raw = dump(value).encode("utf-8")
            Path(stage, name).write_bytes(raw)
            files[name] = _sha(raw)
        receipt = dump(
            {
                "schema_version": 1,
                "files": [
                    {"path": name, "sha256": digest} for name, digest in sorted(files.items())
                ],
            }
        ).encode()
        os.mkdir(target)
        claimed = True
        for current, directories, filenames in os.walk(stage):
            relative = os.path.relpath(current, stage)
            out_dir = target if relative == "." else os.path.join(target, relative)
            os.makedirs(out_dir, exist_ok=True)
            for directory in directories:
                os.makedirs(os.path.join(out_dir, directory), exist_ok=True)
            for filename in filenames:
                source = os.path.join(current, filename)
                rel = os.path.relpath(source, stage)
                out = os.path.join(target, rel)
                shutil.copyfile(source, out)
                key = rel.replace(os.sep, "/")
                if _sha(Path(out).read_bytes()) != files[key]:
                    raise ValueError(f"published file verification failed: {rel}")
        temp_receipt = os.path.join(target, ".completion.tmp")
        Path(temp_receipt).write_bytes(receipt)
        os.replace(temp_receipt, os.path.join(target, "completion.json"))
        return Path(target)
    except Exception:
        if claimed:
            shutil.rmtree(target, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
