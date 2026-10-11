"""CPU-only contract for a selected ABO sample with clean backgrounds.

Protected pixels measure mechanical preservation only; they do not guarantee image quality.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
import zipfile
import zlib
from io import BytesIO
from pathlib import PurePosixPath
from typing import TypedDict
from urllib.parse import urlsplit

from PIL import Image

from . import remote
from .protected_product import compose_protected_product

_MAX_PAYLOAD_BYTES = 32 * 1024 * 1024
_MAX_MEMBER_BYTES = 4 * 1024 * 1024
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_TOTAL_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
_MANIFEST_FIELDS = {
    "schema_version",
    "profile",
    "source_sha",
    "context",
    "modal_sdk",
    "diffsynth_commit",
    "runtime_pins",
    "model_components",
    "generation",
    "prompts",
    "cases",
}
_CASE_FIELDS = {
    "case_id",
    "pool",
    "brand_id",
    "family_id",
    "product_id",
    "source_image_id",
    "source_rgb",
    "alpha_l",
    "protected_l",
    "source_provenance",
    "mask_review",
}
_ASSET_FIELDS = {"path", "sha256"}
_PROVENANCE_FIELDS = {"url", "raw_source_sha256", "metadata_sha256"}
_MASK_REVIEW_FIELDS = {"status", "sha256"}
_ASSET_ROLES = ("source_rgb", "alpha_l", "protected_l")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_ABO_PATH = re.compile(r"^/images/original/[0-9a-f]{2}/[0-9a-f]{8}\.(?:jpg|png)$")
_ABO_HOSTS = {
    "amazon-berkeley-objects.s3.amazonaws.com",
    "amazon-berkeley-objects.s3.us-east-1.amazonaws.com",
}
_CONTEXT = {
    "profile": remote.PROFILE,
    "workspace": remote.WORKSPACE,
    "environment": remote.ENVIRONMENT,
}
_GENERATION = {
    "steps": 16,
    "cfg": 4,
    "width": 512,
    "height": 512,
    "seed_start": 7321,
    "seed_rule": "sorted_case_id_index",
}
_PROMPTS = {
    "full_image_edit": (
        "Replace only the background with a warm beige studio backdrop and a light stone display "
        "surface. Keep the product, text, logos, shape, size, orientation and position unchanged. "
        "Do not put any object in front of the product."
    ),
    "empty_scene": (
        "An empty light stone display surface against a warm beige studio backdrop. Soft studio "
        "light from the upper left. Straight-on product photograph. Leave the center empty. No "
        "products, objects, text or logos."
    ),
}
_SOURCE_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_keys(value: object, expected: set[str], role: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{role} fields are missing or unknown")
    return value


def _validate_manifest_header(manifest: object) -> dict[str, object]:
    data = _require_keys(manifest, _MANIFEST_FIELDS, "manifest")
    if (
        type(data["schema_version"]) is not int
        or data["schema_version"] != 1
        or data["profile"] != "protected-product-comparison"
    ):
        raise ValueError("unsupported protected comparison schema or profile")
    source_sha = data["source_sha"]
    if not isinstance(source_sha, str) or _SOURCE_SHA.fullmatch(source_sha) is None:
        raise ValueError("source_sha must be the frozen lowercase 40-character source digest")
    if data["context"] != _CONTEXT:
        raise ValueError("protected comparison context differs from its approved brief")
    if data["modal_sdk"] != "1.6.1":
        raise ValueError("Modal SDK pin differs from the approved runtime")
    if data["diffsynth_commit"] != remote.DIFFSYNTH_COMMIT:
        raise ValueError("DiffSynth source pin differs from the approved runtime")
    if data["runtime_pins"] != remote.RUNTIME_PINS:
        raise ValueError("runtime pins differ from the approved environment")
    if data["model_components"] != remote.MODEL_COMPONENTS:
        raise ValueError("model component pins differ from the approved models")
    generation = data["generation"]
    if not isinstance(generation, dict) or any(
        type(generation.get(field)) is not int
        for field in ("steps", "cfg", "width", "height", "seed_start")
    ):
        raise ValueError("generation numeric settings must be exact integers")
    if not isinstance(generation.get("seed_rule"), str):
        raise ValueError("generation seed_rule must be a string")
    if generation != _GENERATION:
        raise ValueError("generation settings differ from the frozen comparison")
    if data["prompts"] != _PROMPTS:
        raise ValueError("frozen prompt text differs from the protected comparison")
    return data


def _digest(value: object, role: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{role} must be a lowercase SHA-256 digest")
    return value


def _safe_member_name(value: object, role: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{role} path must be a string")
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{role} path is unsafe")
    return value


def _validate_source_url(value: object) -> None:
    if not isinstance(value, str):
        raise ValueError("source provenance URL must be an official ABO object URL")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("source provenance URL must be an official ABO object URL") from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname not in _ABO_HOSTS
        or parsed.netloc != parsed.hostname
        or port is not None
        or parsed.username is not None
        or parsed.password is not None
        or "?" in value
        or "#" in value
        or parsed.query
        or parsed.fragment
        or _ABO_PATH.fullmatch(parsed.path) is None
    ):
        raise ValueError("source provenance URL must use the official ABO image path")


def _validate_case_rows(
    manifest: dict[str, object], files: dict[str, bytes], *, require_sorted: bool
) -> None:
    generation_schedule(manifest)
    cases = manifest["cases"]
    assert isinstance(cases, list)
    case_ids = [str(case["case_id"]) for case in cases]
    if require_sorted and case_ids != sorted(case_ids):
        raise ValueError("payload cases must be sorted by case_id")

    expected_files: set[str] = set()
    source_shas: set[str] = set()
    raw_shas: set[str] = set()
    for case in cases:
        row = _require_keys(case, _CASE_FIELDS, "case")
        case_id = row["case_id"]
        if not isinstance(case_id, str) or _SAFE_ID.fullmatch(case_id) is None:
            raise ValueError("case_id must be a safe identifier")
        for field in ("brand_id", "family_id", "product_id", "source_image_id"):
            value = row[field]
            if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
                raise ValueError(f"{field} must be a safe identifier")

        provenance = _require_keys(
            row["source_provenance"], _PROVENANCE_FIELDS, "source provenance"
        )
        _validate_source_url(provenance["url"])
        raw_sha = _digest(provenance["raw_source_sha256"], "raw source SHA-256")
        _digest(provenance["metadata_sha256"], "metadata SHA-256")
        if raw_sha in raw_shas:
            raise ValueError("raw source SHA-256 values must be unique")
        raw_shas.add(raw_sha)

        review = _require_keys(row["mask_review"], _MASK_REVIEW_FIELDS, "mask review")
        if review["status"] != "accepted_by_root_visual_review":
            raise ValueError("mask review must be accepted by root visual review")
        _digest(review["sha256"], "mask review SHA-256")

        paths: dict[str, str] = {}
        expected_shas: dict[str, str] = {}
        for role in _ASSET_ROLES:
            asset = _require_keys(row[role], _ASSET_FIELDS, role)
            path = _safe_member_name(asset["path"], role)
            expected_path = f"cases/{case_id}/{role.replace('_', '-')}.png"
            if path != expected_path:
                raise ValueError(f"{role} path does not match its case and role")
            if path in expected_files:
                raise ValueError("asset paths must be unique")
            expected_files.add(path)
            paths[role] = path
            asset_sha = _digest(asset["sha256"], f"{role} SHA-256")
            expected_shas[path] = asset_sha
            if role == "source_rgb":
                if asset_sha in source_shas:
                    raise ValueError("source_rgb SHA-256 values must be unique")
                source_shas.add(asset_sha)

        for role, path in paths.items():
            content = files.get(path)
            if not isinstance(content, bytes) or not content:
                raise ValueError(f"{role} file is missing or invalid")
            if len(content) > _MAX_MEMBER_BYTES:
                raise ValueError(f"{role} file exceeds the per-member byte limit")
            actual_sha = hashlib.sha256(content).hexdigest()
            if actual_sha != expected_shas[path]:
                raise ValueError(f"{role} SHA-256 mismatch")

    if set(files) != expected_files:
        raise ValueError("payload image members are missing or unexpected")
    background_stream = BytesIO()
    Image.new("RGB", (512, 512), (128, 128, 128)).save(background_stream, format="PNG")
    for case in cases:
        row = case
        source = files[row["source_rgb"]["path"]]
        alpha = files[row["alpha_l"]["path"]]
        protected = files[row["protected_l"]["path"]]
        try:
            for role, content in (
                ("source_rgb", source),
                ("alpha_l", alpha),
                ("protected_l", protected),
            ):
                with Image.open(BytesIO(content)) as image:
                    if image.size != (512, 512):
                        raise ValueError(f"{role} must be 512x512")
            _, report = compose_protected_product(
                source, background_stream.getvalue(), alpha, protected
            )
        except (OSError, ValueError, Image.DecompressionBombError, zlib.error) as exc:
            raise ValueError(f"case {row['case_id']} has invalid RGB/L PNG assets: {exc}") from exc
        if int(report["fully_background_count"]) * 10 < 512 * 512:
            raise ValueError("alpha mask must leave at least 10% fully background")


class GenerationTask(TypedDict):
    purpose: str
    pool: str
    arm: str
    case_id: str
    index: int
    seed: int
    replay_of: int | None


def _serialize_payload(manifest: dict[str, object], files: dict[str, bytes]) -> bytes:
    contents = dict(files)
    contents["manifest.json"] = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, content in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, content)
    return output.getvalue()


def build_payload(manifest: dict[str, object], files: dict[str, bytes]) -> bytes:
    """Validate and package JSON and image bytes in a deterministic ZIP payload."""
    _validate_manifest_header(manifest)
    generation_schedule(manifest)
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("protected comparison cases must be an array")
    normalized = dict(manifest)
    normalized["cases"] = sorted(cases, key=lambda case: str(case.get("case_id")))
    _validate_case_rows(normalized, files, require_sorted=True)
    payload = _serialize_payload(normalized, files)
    if len(payload) > _MAX_PAYLOAD_BYTES:
        raise ValueError("protected comparison payload exceeds the 32 MiB byte limit")
    return payload


def validate_payload_bytes(payload: bytes) -> tuple[dict[str, object], dict[str, bytes]]:
    """Validate archive safety, frozen manifest pins, hashes, image masks, and ZIP bytes."""
    if not isinstance(payload, bytes) or not payload or len(payload) > _MAX_PAYLOAD_BYTES:
        raise ValueError("protected comparison payload is empty or exceeds the 32 MiB byte limit")
    try:
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if not infos or len(set(names)) != len(names):
                raise ValueError("protected comparison ZIP has duplicate or missing members")
            if archive.comment:
                raise ValueError("protected comparison ZIP comment is not allowed")
            total_uncompressed = 0
            for info in infos:
                name = _safe_member_name(info.filename, "ZIP member")
                mode = info.external_attr >> 16
                if (
                    name != "manifest.json"
                    and not name.startswith("cases/")
                    or info.is_dir()
                    or stat.S_ISLNK(mode)
                    or info.flag_bits & 1
                    or info.compress_type != zipfile.ZIP_STORED
                ):
                    raise ValueError("protected comparison ZIP contains an unsafe member")
                limit = _MAX_MANIFEST_BYTES if name == "manifest.json" else _MAX_MEMBER_BYTES
                if info.file_size > limit:
                    raise ValueError("protected comparison ZIP member exceeds its size limit")
                total_uncompressed += info.file_size
                if total_uncompressed > _MAX_TOTAL_UNCOMPRESSED_BYTES:
                    raise ValueError("protected comparison ZIP expands beyond its byte limit")
            if names.count("manifest.json") != 1:
                raise ValueError("protected comparison ZIP must contain one manifest.json")
            members = {info.filename: archive.read(info) for info in infos}
        manifest_raw = members.pop("manifest.json")
        manifest = json.loads(
            manifest_raw.decode("utf-8"),
            object_pairs_hook=_unique_json_pairs,
            parse_constant=_reject_json_constant,
        )
    except (
        KeyError,
        OSError,
        RuntimeError,
        EOFError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
        zlib.error,
    ) as exc:
        raise ValueError("protected comparison payload is not a readable ZIP") from exc
    data = _validate_manifest_header(manifest)
    _validate_case_rows(data, members, require_sorted=True)
    if _serialize_payload(data, members) != payload:
        raise ValueError("protected comparison ZIP is not in canonical deterministic form")
    return data, members


def _unique_json_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def generation_schedule(manifest: dict[str, object]) -> tuple[GenerationTask, ...]:
    """Return 48 paired generations and four fresh-load replays in stable order."""
    cases = manifest.get("cases")
    if (
        not isinstance(cases, list)
        or len(cases) != 24
        or any(
            not isinstance(case, dict)
            or not isinstance(case.get("case_id"), str)
            or not isinstance(case.get("pool"), str)
            or case.get("pool") not in ("development", "sealed_test")
            for case in cases
        )
    ):
        raise ValueError("protected comparison requires exactly 24 identified cases")
    ordered = sorted(cases, key=lambda case: str(case["case_id"]))
    case_ids = [str(case["case_id"]) for case in ordered]
    if len(set(case_ids)) != 24:
        raise ValueError("protected comparison case IDs must be unique")
    unique_values: dict[str, set[str]] = {
        field: set()
        for field in ("product_id", "source_image_id", "source_rgb SHA", "raw source SHA")
    }
    for case in ordered:
        scalar_values = {
            "product_id": case.get("product_id"),
            "source_image_id": case.get("source_image_id"),
        }
        source_rgb = case.get("source_rgb")
        provenance = case.get("source_provenance")
        if not isinstance(source_rgb, dict) or not isinstance(provenance, dict):
            raise ValueError("source identity objects are missing or invalid")
        scalar_values["source_rgb SHA"] = source_rgb.get("sha256")
        scalar_values["raw source SHA"] = provenance.get("raw_source_sha256")
        for field, value in scalar_values.items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field} must be a non-empty identifier")
            if value in unique_values[field]:
                raise ValueError(f"{field} values must be unique")
            unique_values[field].add(value)
    development = [case for case in ordered if case["pool"] == "development"]
    sealed_test = [case for case in ordered if case["pool"] == "sealed_test"]
    if len(development) != 6 or len(sealed_test) != 18:
        raise ValueError("protected comparison requires six development and 18 sealed test cases")
    group_pools: dict[tuple[str, str], str] = {}
    for case in ordered:
        pool = str(case["pool"])
        for field in ("brand_id", "family_id"):
            group_id = case.get(field)
            if not isinstance(group_id, str) or not group_id.strip():
                raise ValueError(f"{field} must be a non-empty group identifier")
            previous = group_pools.setdefault((field, group_id), pool)
            if previous != pool:
                raise ValueError(f"{field} crosses development and sealed test")

    tasks: list[GenerationTask] = []
    primary_indices: dict[tuple[str, str], int] = {}
    arms = ("base_full_image_edit", "empty_scene_composite")
    case_seed_indices = {case_id: index for index, case_id in enumerate(case_ids)}
    for case in ordered:
        case_id = str(case["case_id"])
        for arm in arms:
            index = len(tasks)
            primary_indices[(case_id, arm)] = index
            tasks.append(
                {
                    "purpose": "primary",
                    "pool": str(case["pool"]),
                    "arm": arm,
                    "case_id": case_id,
                    "index": index,
                    "seed": 7321 + case_seed_indices[case_id],
                    "replay_of": None,
                }
            )
    for case in development[:2]:
        case_id = str(case["case_id"])
        for arm in arms:
            tasks.append(
                {
                    "purpose": "fresh_load_replay",
                    "pool": "development",
                    "arm": arm,
                    "case_id": case_id,
                    "index": len(tasks),
                    "seed": 7321 + case_seed_indices[case_id],
                    "replay_of": primary_indices[(case_id, arm)],
                }
            )
    return tuple(tasks)
