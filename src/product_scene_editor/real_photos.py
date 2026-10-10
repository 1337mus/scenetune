"""Private CPU-only intake checks for externally sourced product photos."""

import hashlib
import json
import re
import warnings
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn
from urllib.parse import urlsplit

from PIL import Image

_MAX_METADATA_BYTES = 5 * 1024 * 1024
_MAX_PHOTO_BYTES = 5 * 1024 * 1024
_MAX_TOTAL_PHOTO_BYTES = 50 * 1024 * 1024
_PHOTO_FIELDS = set(
    """case_id product_id family_id split source_path sha256 source_url license license_url
    creator attribution rights_review""".split()
)
_PHOTO_LICENSES = {"CC0-1.0", "CC-BY-4.0", "CC-BY-SA-3.0", "CC-BY-SA-4.0", "owned"}
_IMAGE_HASH = re.compile(r"^[0-9a-f]{64}$")


def _unique_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _https_url(value: object, field: str) -> str:
    url = _text(value, field)
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{field} must be an HTTPS URL with a hostname") from exc
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError(f"{field} must be an HTTPS URL with a hostname")
    if port is not None and not 1 <= port <= 65535:
        raise ValueError(f"{field} has an invalid port")
    return url


def _inside_root(path: Path, root: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
        if path.absolute() != resolved:
            raise ValueError(f"{label} may not pass through a symlink")
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{label} is missing or escapes the intake root") from exc
    return resolved


def _source_file(root: Path, split: str, value: object) -> Path:
    relative = _text(value, "source_path")
    posix = PurePosixPath(relative)
    if (
        "\\" in relative
        or posix.is_absolute()
        or posix.as_posix() != relative
        or ".." in posix.parts
        or posix.parts[:2] != (split, "source")
        or len(posix.parts) < 3
    ):
        raise ValueError(f"source_path must be a safe path below {split}/source/")
    path = _inside_root(root / Path(*posix.parts), root, "source_path")
    if not path.is_file():
        raise ValueError("source_path must identify a file")
    return path


def _image_dimensions(content: bytes, case_id: str) -> list[int]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as image:
                dimensions = [int(image.width), int(image.height)]
                image.verify()
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        OSError,
        ValueError,
    ) as exc:
        raise ValueError(f"photo {case_id!r} is not a readable image") from exc
    if min(dimensions) < 1:
        raise ValueError(f"photo {case_id!r} has invalid dimensions")
    return dimensions


def _load_intake(root: Path | str) -> tuple[Path, dict[str, Any]]:
    base = Path(root)
    try:
        resolved_root = base.resolve(strict=True)
        if not resolved_root.is_dir():
            raise ValueError("intake root must be a directory")
        intake_path = _inside_root(resolved_root / "intake.json", resolved_root, "intake.json")
        if not intake_path.is_file() or intake_path.stat().st_size > _MAX_METADATA_BYTES:
            raise ValueError("intake.json is missing or exceeds the metadata size limit")
        content = intake_path.read_bytes()
        if len(content) > _MAX_METADATA_BYTES:
            raise ValueError("intake.json exceeds the metadata size limit")
        intake = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_unique_object_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("intake.json is missing or invalid JSON") from exc
    if not isinstance(intake, dict) or set(intake) != {
        "schema_version",
        "photos",
        "reserved_groups",
    }:
        raise ValueError("intake fields are missing or unknown")
    if type(intake["schema_version"]) is not int or intake["schema_version"] != 1:
        raise ValueError("unsupported intake schema_version")
    if not isinstance(intake["photos"], list) or not isinstance(intake["reserved_groups"], list):
        raise ValueError("photos and reserved_groups must be arrays")
    return resolved_root, intake


def verify_intake(root: Path | str) -> dict[str, Any]:
    """Check local image bytes; this does not establish source rights or evaluation readiness."""
    resolved_root, intake = _load_intake(root)
    reserved_products: set[str] = set()
    reserved_families: set[str] = set()
    for group in intake["reserved_groups"]:
        if not isinstance(group, dict) or set(group) != {"product_id", "family_id"}:
            raise ValueError("reserved_groups may contain only product_id and family_id")
        reserved_products.add(_text(group["product_id"], "reserved product_id"))
        reserved_families.add(_text(group["family_id"], "reserved family_id"))

    cases: set[str] = set()
    hashes: set[str] = set()
    group_splits: dict[tuple[str, str], str] = {}
    split_counts = {"train": 0, "dev": 0}
    dimensions: dict[str, list[int]] = {}
    pending_rights: list[str] = []
    unresolved_creators: list[str] = []
    total_bytes = 0

    for photo in intake["photos"]:
        if not isinstance(photo, dict) or set(photo) != _PHOTO_FIELDS:
            raise ValueError("photo fields are missing or unknown")
        case_id, product_id, family_id = (
            _text(photo[field], field) for field in ("case_id", "product_id", "family_id")
        )
        if case_id in cases:
            raise ValueError(f"duplicate case_id: {case_id}")
        cases.add(case_id)
        split = photo["split"]
        if not isinstance(split, str) or split not in split_counts:
            raise ValueError(f"photo {case_id!r} split must be train or dev")
        for field, value in (("product_id", product_id), ("family_id", family_id)):
            previous = group_splits.setdefault((field, value), split)
            if previous != split:
                raise ValueError(f"{field} {value!r} crosses train and dev")
        if product_id in reserved_products or family_id in reserved_families:
            raise ValueError(f"photo {case_id!r} overlaps a reserved product or family")

        license_name = photo["license"]
        if not isinstance(license_name, str) or license_name not in _PHOTO_LICENSES:
            raise ValueError(f"photo {case_id!r} has an unsupported license label")
        source_url = photo["source_url"]
        if source_url is None and license_name != "owned":
            raise ValueError("source_url may be null only for owned photos")
        if source_url is not None:
            _https_url(source_url, "source_url")
        license_url = photo["license_url"]
        if license_url is None and license_name != "owned":
            raise ValueError("license_url may be null only for owned photos")
        if license_url is not None:
            _https_url(license_url, "license_url")
        creator = photo["creator"]
        if creator is not None:
            _text(creator, "creator")
        _text(photo["attribution"], "attribution")
        rights_review = photo["rights_review"]
        if not isinstance(rights_review, str) or rights_review not in {"pending", "approved"}:
            raise ValueError(f"photo {case_id!r} rights_review must be pending or approved")
        if rights_review == "pending":
            pending_rights.append(case_id)
        if creator is None:
            unresolved_creators.append(case_id)

        digest = photo["sha256"]
        if not isinstance(digest, str) or not _IMAGE_HASH.fullmatch(digest):
            raise ValueError(f"photo {case_id!r} sha256 must be a lowercase SHA-256 digest")
        source = _source_file(resolved_root, split, photo["source_path"])
        size = source.stat().st_size
        if size > _MAX_PHOTO_BYTES or total_bytes + size > _MAX_TOTAL_PHOTO_BYTES:
            raise ValueError("photo inputs exceed the per-photo or total size limit")
        with source.open("rb") as stream:
            content = stream.read(_MAX_PHOTO_BYTES + 1)
        if len(content) != size or len(content) > _MAX_PHOTO_BYTES:
            raise ValueError("photo input changed or exceeds the per-photo size limit")
        total_bytes += len(content)
        actual_hash = hashlib.sha256(content).hexdigest()
        if actual_hash != digest:
            raise ValueError(f"photo {case_id!r} source hash does not match")
        if actual_hash in hashes:
            raise ValueError(f"duplicate photo hash for case {case_id!r}")
        hashes.add(actual_hash)
        dimensions[case_id] = _image_dimensions(content, case_id)
        split_counts[split] += 1

    return {
        "photo_count": len(cases),
        "split_counts": split_counts,
        "dimensions": dimensions,
        "pending_rights_case_ids": sorted(pending_rights),
        "unresolved_creator_case_ids": sorted(unresolved_creators),
        "ready_for_model_evaluation": False,
        "not_ready_reasons": [
            "License labels and rights_review do not establish legal clearance or human review.",
            (
                "Masks and text/label preservation are not verified, "
                "even when rights_review is approved."
            ),
            "Exact byte hashes detect identical files only, not near-duplicate photos or products.",
            "Creator and attribution are not verified; a null creator remains unresolved.",
        ],
    }
