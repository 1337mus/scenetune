"""CPU-only validation and payload helpers for private real-photo inference."""

from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, TypedDict

from PIL import Image, ImageOps

CASE_IDS = ("OFF-R1-003", "OFF-R1-005", "OFF-R1-006")
INTAKE_RELATIVE_PATH = "data/real-photos-off-r1/intake.json"
INTAKE_ASSET_PREFIX = PurePosixPath(INTAKE_RELATIVE_PATH).parent
PROMPT = (
    "Replace only the background with a light stone counter and a softly blurred kitchen. "
    "Keep the visible package, text, logos, shape, size, orientation and position unchanged. "
    "Do not put any object in front of the package."
)
GENERATION_SETTINGS = {
    "seed": 7321,
    "steps": 16,
    "cfg_scale": 4,
    "width": 512,
    "height": 512,
}
ADAPTER_RUN_ID = "klein-edit-matched-2026-10-09-r1"
ADAPTER_ARCHIVE_SHA256 = "6e124ffcb51ff4a8606320c384c309ce107c2030cf17a18e46565924741e4e58"
ADAPTER_MEMBER_NAME = "adapter/step-144.safetensors"
ADAPTER_MEMBER_SHA256 = "bf8494f24f2ba7ac40d0af765793923416387c8ce2736091b2cbdf9309e99581"
ADAPTER_MEMBER_BYTES = 96_364_672
MAX_INPUT_BYTES = 120 * 1024 * 1024
MAX_RESULT_BYTES = 16 * 1024 * 1024
RUN_MARKER_NAME = ".real-photo-run.json"
PAYLOAD_NAME = "real-photo-input.zip"
RESULT_NAME = "real-photo-result.zip"
ATTEMPT_MARKER_NAME = "real-photo-modal-attempt.json"
IMAGE_SIZE = (512, 512)
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class GenerationTask(TypedDict):
    case_id: str
    mode: str
    model_session: str


def generation_schedule() -> tuple[GenerationTask, ...]:
    """Return three base jobs, three adapter jobs, then one fresh-load replay."""
    tasks: list[GenerationTask] = []
    for case_id in CASE_IDS:
        tasks.append({"case_id": case_id, "mode": "base", "model_session": "base"})
    for case_id in CASE_IDS:
        tasks.append({"case_id": case_id, "mode": "strict_adapter", "model_session": "strict-1"})
    tasks.append({"case_id": CASE_IDS[0], "mode": "strict_repeat", "model_session": "strict-2"})
    return tuple(tasks)


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical JSON used for preparation, annotation, plan, and receipt bindings."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _case_hash(case: dict[str, Any], flat_name: str, nested_name: str) -> str:
    value = case.get(flat_name)
    if value is None:
        nested = case.get(nested_name)
        value = nested.get("sha256") if isinstance(nested, dict) else None
    if not _is_digest(value):
        raise ValueError(f"case {case.get('case_id')!r} has an invalid {flat_name}")
    return str(value)


def validate_source_geometry_review(preparation: dict[str, Any], review: dict[str, Any]) -> None:
    """Require a human confirmation that binds every prepared source/mask/box byte digest."""
    if not isinstance(preparation, dict) or not isinstance(review, dict):
        raise ValueError("preparation and source geometry review must be JSON objects")
    if (
        review.get("schema_version") != 1
        or review.get("status") != "approved"
        or review.get("reviewed_by") != "Raj"
    ):
        raise ValueError("source geometry review is not approved by Raj")
    statement = review.get("confirmation_statement")
    if not isinstance(statement, str) or not statement.strip():
        raise ValueError("source geometry review requires the actual human confirmation")
    reviewed_at = review.get("reviewed_at")
    if not isinstance(reviewed_at, str):
        raise ValueError("source geometry review timestamp is missing")
    try:
        parsed_at = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("source geometry review timestamp is invalid") from exc
    if parsed_at.tzinfo is None or parsed_at.utcoffset() is None:
        raise ValueError("source geometry review timestamp must include a timezone")

    raw_cases = preparation.get("cases")
    review_cases = review.get("cases")
    if (
        not isinstance(raw_cases, list)
        or not isinstance(review_cases, list)
        or len(raw_cases) != len(CASE_IDS)
        or len(review_cases) != len(CASE_IDS)
        or review.get("case_ids") != list(CASE_IDS)
    ):
        raise ValueError("source geometry review must cover the exact three approved cases")
    prep_by_id: dict[str, dict[str, Any]] = {}
    for case in raw_cases:
        if not isinstance(case, dict) or case.get("case_id") not in CASE_IDS:
            raise ValueError("preparation contains an unexpected case")
        if case["case_id"] in prep_by_id:
            raise ValueError("preparation contains duplicate cases")
        prep_by_id[case["case_id"]] = case
    if tuple(prep_by_id) != CASE_IDS:
        raise ValueError("preparation cases are missing or not in the frozen order")

    expected_review_keys = {
        "case_id",
        "prep_sha256",
        "source_sha256",
        "derived_sha256",
        "mask_sha256",
        "label_boxes_sha256",
    }
    for case_id, reviewed in zip(CASE_IDS, review_cases, strict=True):
        if not isinstance(reviewed, dict) or set(reviewed) != expected_review_keys:
            raise ValueError("source geometry review case fields are missing or unknown")
        case = prep_by_id[case_id]
        boxes = case.get("label_boxes")
        if not isinstance(boxes, list):
            raise ValueError(f"case {case_id} label boxes are missing")
        computed_boxes = canonical_sha256(boxes)
        recorded_boxes = case.get("label_boxes_sha256")
        if recorded_boxes != computed_boxes:
            raise ValueError(f"case {case_id} label-box digest does not match its boxes")
        expected = {
            "case_id": case_id,
            "prep_sha256": canonical_sha256(case),
            "source_sha256": _case_hash(case, "source_sha256", "source"),
            "derived_sha256": _case_hash(case, "derived_sha256", "derived"),
            "mask_sha256": _case_hash(case, "mask_sha256", "mask"),
            "label_boxes_sha256": computed_boxes,
        }
        if reviewed != expected:
            raise ValueError(f"source geometry review does not bind case {case_id}")


def validate_adapter_bytes(
    data: bytes,
    *,
    expected_digest: str = ADAPTER_MEMBER_SHA256,
    expected_bytes: int = ADAPTER_MEMBER_BYTES,
) -> str:
    if not isinstance(data, bytes) or not data or len(data) != expected_bytes:
        raise ValueError("adapter member has the wrong byte size")
    actual = sha256_bytes(data)
    if not _is_digest(expected_digest) or actual != expected_digest:
        raise ValueError("adapter member SHA-256 does not match the approved artifact")
    return actual


def prepare_rgb_input(source_bytes: bytes) -> tuple[bytes, dict[str, int]]:
    """Aspect-fit RGB source pixels on a fixed white 512 square using Pillow LANCZOS."""
    if not isinstance(source_bytes, bytes) or not source_bytes:
        raise ValueError("source photo bytes are empty")
    try:
        with Image.open(BytesIO(source_bytes)) as opened:
            oriented = ImageOps.exif_transpose(opened).convert("RGB")
            original_width, original_height = oriented.size
            scale = min(512 / original_width, 512 / original_height)
            width = max(1, min(512, math.floor(original_width * scale + 0.5)))
            height = max(1, min(512, math.floor(original_height * scale + 0.5)))
            resized = oriented.resize((width, height), Image.Resampling.LANCZOS)
            left = (512 - width) // 2
            top = (512 - height) // 2
            canvas = Image.new("RGB", IMAGE_SIZE, (255, 255, 255))
            canvas.paste(resized, (left, top))
            details = {
                "source_width": original_width,
                "source_height": original_height,
                "resized_width": width,
                "resized_height": height,
                "pad_left": left,
                "pad_top": top,
                "pad_right": 512 - width - left,
                "pad_bottom": 512 - height - top,
            }
            stream = BytesIO()
            canvas.save(stream, format="PNG", compress_level=9, optimize=False)
            return stream.getvalue(), details
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError("source photo is not a readable image") from exc


def validate_preparation_and_intake(
    preparation: dict[str, Any],
    intake: dict[str, Any],
    *,
    active_intake_bytes: bytes | None = None,
) -> None:
    """Fail closed unless active intake and the prepared derivatives describe only dev cases."""
    if not isinstance(preparation, dict) or not isinstance(intake, dict):
        raise ValueError("preparation and active intake must be JSON objects")
    if preparation.get("schema_version") != 1 or intake.get("schema_version") != 1:
        raise ValueError("unsupported real-photo preparation or intake schema")
    aggregate = preparation.get("aggregate_sha256")
    unsigned_prep = dict(preparation)
    unsigned_prep.pop("aggregate_sha256", None)
    if not _is_digest(aggregate) or canonical_sha256(unsigned_prep) != aggregate:
        raise ValueError("real-photo preparation aggregate digest does not match")
    intake_digest = (
        sha256_bytes(active_intake_bytes)
        if active_intake_bytes is not None
        else sha256_bytes(canonical_json_bytes(intake))
    )
    input_intake = preparation.get("input_intake")
    if not isinstance(input_intake, dict) or input_intake.get("active_sha256") != intake_digest:
        raise ValueError("preparation is not bound to the active intake manifest")
    photos = intake.get("photos")
    prep_cases = preparation.get("cases")
    if not isinstance(photos, list) or not isinstance(prep_cases, list):
        raise ValueError("intake photos and preparation cases must be arrays")
    selected_rows = [
        row for row in photos if isinstance(row, dict) and row.get("case_id") in CASE_IDS
    ]
    selected_photos = {row.get("case_id"): row for row in selected_rows}
    by_id = {row.get("case_id"): row for row in prep_cases if isinstance(row, dict)}
    if (
        len(selected_rows) != len(CASE_IDS)
        or set(selected_photos) != set(CASE_IDS)
        or set(by_id) != set(CASE_IDS)
    ):
        raise ValueError("active intake and preparation must contain the exact approved cases")
    if len(prep_cases) != len(CASE_IDS):
        raise ValueError("preparation contains duplicate or unexpected cases")
    for case_id in CASE_IDS:
        photo = selected_photos[case_id]
        case = by_id[case_id]
        if photo.get("split") != "dev" or case.get("split") != "dev":
            raise ValueError(f"case {case_id} is not explicitly development-only")
        if (
            case.get("source", {}).get("sha256") != photo.get("sha256")
            or case.get("source", {}).get("path") != photo.get("source_path")
            or case.get("product_id") != photo.get("product_id")
            or case.get("family_id") != photo.get("family_id")
            or case.get("role") != "development_only"
        ):
            raise ValueError(f"case {case_id} identity/source does not match active dev intake")
        if photo.get("rights_review") != "pending":
            raise ValueError("private inference runner expects rights review to remain pending")
        if case.get("label_boxes_review_status") != "unreviewed":
            raise ValueError("preparation annotation state is unexpected")


def _safe_archive(archive: zipfile.ZipFile, *, total_limit: int) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    total = 0
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if (
            path.is_absolute()
            or not info.filename
            or "\\" in info.filename
            or ".." in path.parts
            or info.filename in files
            or info.is_dir()
        ):
            raise ValueError("archive contains an unsafe or duplicate path")
        total += info.file_size
        if total > total_limit:
            raise ValueError("archive exceeds its byte limit")
        files[info.filename] = archive.read(info)
    return files


def canonical_archive(files: dict[str, bytes]) -> bytes:
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(files.items()):
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError("archive path is unsafe")
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, data)
    result = stream.getvalue()
    if len(result) > MAX_INPUT_BYTES:
        raise ValueError("real-photo payload exceeds the input byte limit")
    return result


def _open_checked_image(data: bytes, *, mode: str, role: str) -> Image.Image:
    try:
        with Image.open(BytesIO(data)) as opened:
            if opened.format != "PNG" or opened.mode != mode or opened.size != IMAGE_SIZE:
                raise ValueError(f"{role} must be a {IMAGE_SIZE[0]}px {mode} PNG")
            opened.load()
            return opened.copy()
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"{role} is not a readable PNG") from exc


def validate_payload_bytes(
    data: bytes,
    expected_run_id: str,
    *,
    expected_adapter_digest: str = ADAPTER_MEMBER_SHA256,
    expected_adapter_bytes: int = ADAPTER_MEMBER_BYTES,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Validate exact, bounded inference inputs without importing Modal or model packages."""
    if not isinstance(data, bytes) or not data or len(data) > MAX_INPUT_BYTES:
        raise ValueError("real-photo payload is empty or exceeds the input limit")
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            files = _safe_archive(archive, total_limit=MAX_INPUT_BYTES)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("real-photo payload is not a valid ZIP archive") from exc
    try:
        manifest = json.loads(files["manifest.json"].decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("real-photo payload is missing a valid manifest") from exc
    manifest_fields = {
        "schema_version",
        "run_id",
        "case_ids",
        "cases",
        "prompt",
        "generation_settings",
        "adapter_sha256",
        "training_function_called",
        "optimizer_updates",
        "preparation_sha256",
        "geometry_review_sha256",
        "geometry_review_status",
        "active_intake_sha256",
        "label_confirmation_sha256",
        "private_source_review_sha256",
        "quality_review_status",
        "adapter_archive_sha256",
        "adapter_member_sha256",
        "modal_sdk",
        "diffsynth_commit",
        "runtime_pins",
        "model_components",
    }
    from . import remote

    if (
        not isinstance(manifest, dict)
        or set(manifest) != manifest_fields
        or manifest.get("schema_version") != 1
        or manifest.get("run_id") != expected_run_id
        or manifest.get("case_ids") != list(CASE_IDS)
        or manifest.get("prompt") != PROMPT
        or manifest.get("generation_settings") != GENERATION_SETTINGS
        or manifest.get("training_function_called") is not False
        or manifest.get("optimizer_updates") != 0
        or not all(
            _is_digest(manifest.get(key))
            for key in (
                "preparation_sha256",
                "active_intake_sha256",
                "label_confirmation_sha256",
                "private_source_review_sha256",
            )
        )
        or manifest.get("geometry_review_status")
        not in {"approved", "awaiting_human_review", "not_present"}
        or manifest.get("quality_review_status")
        != (
            "approved"
            if manifest.get("geometry_review_status") == "approved"
            else "pending_human_geometry_review"
        )
        or (
            manifest.get("geometry_review_status") in {"approved", "awaiting_human_review"}
            and not _is_digest(manifest.get("geometry_review_sha256"))
        )
        or (
            manifest.get("geometry_review_status") == "not_present"
            and manifest.get("geometry_review_sha256") is not None
        )
        or manifest.get("adapter_archive_sha256") != ADAPTER_ARCHIVE_SHA256
        or manifest.get("adapter_member_sha256") != ADAPTER_MEMBER_SHA256
        or manifest.get("modal_sdk") != "1.6.1"
        or manifest.get("diffsynth_commit") != remote.DIFFSYNTH_COMMIT
        or manifest.get("runtime_pins") != remote.RUNTIME_PINS
        or manifest.get("model_components") != remote.MODEL_COMPONENTS
    ):
        raise ValueError("real-photo payload does not match the fixed inference contract")
    try:
        adapter_digest = validate_adapter_bytes(
            files[ADAPTER_MEMBER_NAME],
            expected_digest=expected_adapter_digest,
            expected_bytes=expected_adapter_bytes,
        )
    except KeyError as exc:
        raise ValueError("real-photo payload is missing its adapter member") from exc
    if manifest.get("adapter_sha256") != adapter_digest:
        raise ValueError("real-photo manifest adapter digest does not match its member")

    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != len(CASE_IDS):
        raise ValueError("real-photo manifest must contain exactly three cases")
    expected_files = {"manifest.json", ADAPTER_MEMBER_NAME}
    case_fields = {
        "case_id",
        "split",
        "source_path",
        "source_sha256",
        "input_path",
        "input_sha256",
        "mask_path",
        "mask_sha256",
        "label_boxes",
        "label_boxes_sha256",
    }
    for case_id, case in zip(CASE_IDS, cases, strict=True):
        if (
            not isinstance(case, dict)
            or set(case) != case_fields
            or case.get("case_id") != case_id
            or case.get("split") != "dev"
        ):
            raise ValueError("real-photo manifest case rows are invalid")
        boxes = case.get("label_boxes")
        if not isinstance(boxes, list) or case.get("label_boxes_sha256") != canonical_sha256(boxes):
            raise ValueError(f"case {case_id} label boxes do not match their digest")
        for role, path_key, hash_key in (
            ("source", "source_path", "source_sha256"),
            ("input", "input_path", "input_sha256"),
            ("mask", "mask_path", "mask_sha256"),
        ):
            path_value = case.get(path_key)
            if not isinstance(path_value, str):
                raise ValueError(f"case {case_id} {role} path is missing")
            path = PurePosixPath(path_value)
            expected_parent = ("photos", case_id)
            if (
                path.is_absolute()
                or path.as_posix() != path_value
                or path.parts[:2] != expected_parent
                or len(path.parts) != 3
                or ".." in path.parts
                or "\\" in path_value
                or path_value not in files
            ):
                raise ValueError(f"case {case_id} {role} path is unsafe or missing")
            content = files[path_value]
            digest = case.get(hash_key)
            if not _is_digest(digest) or sha256_bytes(content) != digest:
                raise ValueError(f"case {case_id} {role} digest does not match")
            expected_files.add(path_value)
            if role == "input":
                _open_checked_image(content, mode="RGB", role=f"case {case_id} input")
            elif role == "mask":
                mask = _open_checked_image(content, mode="L", role=f"case {case_id} mask")
                values = set(mask.tobytes())
                if not values <= {0, 255} or values != {0, 255}:
                    raise ValueError(
                        f"case {case_id} mask must be binary with foreground/background"
                    )
    if set(files) != expected_files:
        raise ValueError("real-photo payload has unexpected or unreferenced members")
    return manifest, files


def _validate_result_bindings(
    receipt: dict[str, Any],
    expected_run_id: str,
    *,
    expected_plan: dict[str, Any] | None,
    expected_payload_sha256: str | None,
) -> None:
    from . import remote

    if (
        not _is_digest(receipt.get("payload_sha256"))
        or receipt.get("preparation_sha256") is None
        or not _is_digest(receipt.get("preparation_sha256"))
        or not _is_digest(receipt.get("active_intake_sha256"))
        or not _is_digest(receipt.get("label_confirmation_sha256"))
        or not _is_digest(receipt.get("private_source_review_sha256"))
        or receipt.get("geometry_review_status")
        not in {"approved", "awaiting_human_review", "not_present"}
        or receipt.get("quality_review_status")
        != (
            "approved"
            if receipt.get("geometry_review_status") == "approved"
            else "pending_human_geometry_review"
        )
        or (
            receipt.get("geometry_review_status") in {"approved", "awaiting_human_review"}
            and not _is_digest(receipt.get("geometry_review_sha256"))
        )
        or (
            receipt.get("geometry_review_status") == "not_present"
            and receipt.get("geometry_review_sha256") is not None
        )
        or receipt.get("adapter_archive_sha256") != ADAPTER_ARCHIVE_SHA256
        or receipt.get("adapter_member_sha256") != ADAPTER_MEMBER_SHA256
        or receipt.get("modal_sdk") != "1.6.1"
        or receipt.get("diffsynth_commit") != remote.DIFFSYNTH_COMMIT
        or receipt.get("runtime_pins") != remote.RUNTIME_PINS
        or receipt.get("model_components") != remote.MODEL_COMPONENTS
    ):
        raise ValueError("real-photo receipt is missing fixed plan and source bindings")

    runtime_versions = receipt.get("runtime_versions")
    expected_runtime_versions = {"modal": "1.6.1", **remote.RUNTIME_PINS}
    if (
        not isinstance(runtime_versions, dict)
        or any(
            key not in expected_runtime_versions or not isinstance(value, str) or not value
            for key, value in runtime_versions.items()
        )
        or (receipt.get("status") == "passed" and runtime_versions != expected_runtime_versions)
        or (
            receipt.get("status") == "passed"
            and receipt.get("observed_diffsynth_commit") != remote.DIFFSYNTH_COMMIT
        )
    ):
        raise ValueError("real-photo receipt runtime evidence is incomplete or mismatched")
    observed_commit = receipt.get("observed_diffsynth_commit")
    if observed_commit is not None and not isinstance(observed_commit, str):
        raise ValueError("real-photo receipt has invalid observed DiffSynth identity")

    strict_loads = receipt.get("strict_load_validations")
    if not isinstance(strict_loads, list) or len(strict_loads) > 2:
        raise ValueError("real-photo receipt strict-load evidence must be a bounded array")
    observed_modes: dict[str, Any] | None = None
    for record in strict_loads:
        if not isinstance(record, dict) or set(record) != {
            "observed",
            "mode_changed",
            "adapter_validation",
        }:
            raise ValueError("real-photo strict-load evidence has an invalid shape")
        observed = record.get("observed")
        adapter_validation = record.get("adapter_validation")
        if (
            not isinstance(observed, dict)
            or set(observed) != {"pipe_training", "lora_dropout_training"}
            or not isinstance(observed.get("pipe_training"), bool)
            or not isinstance(observed.get("lora_dropout_training"), list)
            or any(not isinstance(mode, bool) for mode in observed["lora_dropout_training"])
            or record.get("mode_changed") is not False
            or not isinstance(adapter_validation, dict)
            or adapter_validation
            != {"validated": True, "restored_tensor_count": 200, "module_count": 100}
            or (observed_modes is not None and observed != observed_modes)
        ):
            raise ValueError("real-photo strict-load validation or module modes differ")
        observed_modes = observed
    if receipt.get("status") == "passed" and len(strict_loads) != 2:
        raise ValueError("passed real-photo receipt requires two strict-load validations")
    peak = receipt.get("peak_allocated_bytes")
    if peak is not None and (type(peak) is not int or peak < 0):
        raise ValueError("real-photo receipt has invalid CUDA peak-memory evidence")
    elapsed = receipt.get("elapsed_seconds")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or elapsed < 0
    ):
        raise ValueError("real-photo receipt has invalid elapsed-time evidence")
    if receipt.get("status") == "passed" and (type(peak) is not int or peak <= 0):
        raise ValueError("passed real-photo receipt requires positive CUDA peak-memory evidence")

    if (expected_plan is None) != (expected_payload_sha256 is None):
        raise ValueError("result validation requires both expected plan and payload digest")
    if expected_plan is not None:
        if (
            not _is_digest(expected_payload_sha256)
            or expected_plan.get("run_id") != expected_run_id
            or expected_plan.get("payload_sha256") != expected_payload_sha256
        ):
            raise ValueError("expected real-photo plan and payload digest are inconsistent")
        bindings = {
            "payload_sha256": expected_payload_sha256,
            "preparation_sha256": expected_plan.get("preparation_sha256"),
            "geometry_review_sha256": expected_plan.get("geometry_review_sha256"),
            "geometry_review_status": expected_plan.get("geometry_review_status"),
            "active_intake_sha256": expected_plan.get("active_intake_sha256"),
            "label_confirmation_sha256": expected_plan.get("label_confirmation_sha256"),
            "private_source_review_sha256": expected_plan.get("private_source_review_sha256"),
            "quality_review_status": expected_plan.get("quality_review_status"),
            "adapter_archive_sha256": expected_plan.get("adapter_archive_sha256"),
            "adapter_member_sha256": expected_plan.get("adapter_member_sha256"),
            "modal_sdk": expected_plan.get("modal_sdk"),
            "diffsynth_commit": expected_plan.get("diffsynth_commit"),
            "runtime_pins": expected_plan.get("runtime_pins"),
            "model_components": expected_plan.get("model_components"),
        }
        if any(receipt.get(key) != value for key, value in bindings.items()):
            raise ValueError("real-photo result does not match its immutable plan or payload")


def validate_result_bundle(
    data: bytes,
    expected_run_id: str,
    *,
    expected_plan: dict[str, Any] | None = None,
    expected_payload_sha256: str | None = None,
) -> tuple[dict[str, Any], set[str]]:
    """Validate a bounded worker receipt and every claimed 512px output before local storage."""
    if not isinstance(data, bytes) or not data or len(data) > MAX_RESULT_BYTES:
        raise ValueError("real-photo result is empty or exceeds the output limit")
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            files = _safe_archive(archive, total_limit=MAX_RESULT_BYTES)
        receipt = json.loads(files["receipt.json"].decode("utf-8"))
    except (OSError, KeyError, UnicodeDecodeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise ValueError("real-photo result is not a valid receipt archive") from exc
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != 1
        or receipt.get("run_id") != expected_run_id
        or receipt.get("status") not in {"passed", "failed"}
        or receipt.get("generation_settings") != GENERATION_SETTINGS
        or receipt.get("schedule") != list(generation_schedule())
        or receipt.get("training_function_called") is not False
        or receipt.get("optimizer_updates") != 0
        or receipt.get("quality_claim") is not False
        or receipt.get("worker_failure_confirmed") is not (receipt.get("status") == "failed")
    ):
        raise ValueError("real-photo result receipt does not match the fixed inference contract")
    _validate_result_bindings(
        receipt,
        expected_run_id,
        expected_plan=expected_plan,
        expected_payload_sha256=expected_payload_sha256,
    )
    artifacts = receipt.get("output_artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("real-photo receipt output_artifacts must be an array")
    expected_paths = {
        f"outputs/{task['case_id']}/{task['mode']}.png" for task in generation_schedule()
    }
    by_path: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        if (
            not isinstance(artifact, dict)
            or set(artifact) != {"path", "sha256", "width", "height"}
            or artifact.get("path") not in expected_paths
            or artifact["path"] in by_path
            or not _is_digest(artifact.get("sha256"))
            or artifact.get("width") != 512
            or artifact.get("height") != 512
        ):
            raise ValueError("real-photo receipt contains invalid output metadata")
        by_path[artifact["path"]] = artifact
    image_paths = set(files) - {"receipt.json"}
    if image_paths != set(by_path) or not image_paths <= expected_paths:
        raise ValueError("real-photo result members do not match the receipt output list")
    opened: dict[str, Image.Image] = {}
    for path, artifact in by_path.items():
        content = files[path]
        if sha256_bytes(content) != artifact["sha256"]:
            raise ValueError(f"real-photo output digest does not match for {path}")
        opened[path] = _open_checked_image(content, mode="RGB", role=path)
    replay_first = f"outputs/{CASE_IDS[0]}/strict_adapter.png"
    replay_second = f"outputs/{CASE_IDS[0]}/strict_repeat.png"
    if replay_first in opened and replay_second in opened:
        replay_matches = opened[replay_first].tobytes() == opened[replay_second].tobytes()
        if receipt.get("replay_pixel_equal") is not replay_matches:
            raise ValueError("real-photo replay equality claim does not match the output pixels")
    elif receipt.get("status") == "passed":
        raise ValueError("passed real-photo result is missing the strict replay pair")
    if receipt["status"] == "passed" and (
        set(by_path) != expected_paths or receipt.get("replay_pixel_equal") is not True
    ):
        raise ValueError("passed real-photo result is incomplete or replay was not pixel-exact")
    if receipt["status"] == "failed":
        failure = receipt.get("failure")
        if not isinstance(failure, dict) or not all(
            isinstance(failure.get(key), str) and failure[key]
            for key in ("stage", "type", "message")
        ):
            raise ValueError("failed real-photo result is missing bounded failure evidence")
    return receipt, image_paths


def validate_visible_word_confirmation(
    preparation: dict[str, Any], confirmation: dict[str, Any]
) -> None:
    expected_words = {
        "OFF-R1-003": ["Barilla", "SPAGHETTI"],
        "OFF-R1-005": ["kinder", "bueno"],
        "OFF-R1-006": ["QUAKER", "Cruesli", "MÉLANGE DE NOIX"],
    }
    if (
        confirmation.get("status") != "confirmed_visible_word_spellings_only"
        or confirmation.get("confirmed_by") != "Raj"
        or confirmation.get("confirmation_statement_exact") != "Yes, the words match"
        or confirmation.get("source_intake_sha256")
        != preparation.get("input_intake", {}).get("collection_original_sha256")
    ):
        raise ValueError("visible-word confirmation does not match the preserved collection intake")
    annotations = confirmation.get("annotations")
    cases = preparation.get("cases")
    if not isinstance(annotations, list) or not isinstance(cases, list):
        raise ValueError("visible-word confirmation records are missing")
    annotation_by_id = {row.get("case_id"): row for row in annotations if isinstance(row, dict)}
    prep_by_id = {row.get("case_id"): row for row in cases if isinstance(row, dict)}
    if set(annotation_by_id) != set(CASE_IDS) or set(prep_by_id) != set(CASE_IDS):
        raise ValueError("visible-word confirmation must cover exactly the selected cases")
    for case_id, words in expected_words.items():
        annotation = annotation_by_id[case_id]
        case = prep_by_id[case_id]
        confirmed = annotation.get("confirmed_words")
        if (
            confirmed != words
            or [item.get("text") for item in case.get("label_boxes", [])] != words
            or annotation.get("word_boxes_reviewed") is not False
            or annotation.get("mask_created_or_reviewed") is not False
            or case.get("label_boxes_review_status") != "unreviewed"
        ):
            raise ValueError(f"visible words or review boundary changed for case {case_id}")


def _read_json_file(
    path: Path, *, max_bytes: int = 10 * 1024 * 1024
) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError(f"required JSON evidence is missing or unsafe: {path.name}")
    data = path.read_bytes()
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path.name} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value, data


def _read_private_file(root: Path, relative: object, *, from_intake_parent: bool = False) -> bytes:
    if not isinstance(relative, str):
        raise ValueError("preparation asset path must be a relative string")
    posix = PurePosixPath(relative)
    if (
        not relative
        or "\\" in relative
        or posix.is_absolute()
        or posix.as_posix() != relative
        or ".." in posix.parts
    ):
        raise ValueError("preparation contains an unsafe private asset path")
    asset_path = INTAKE_ASSET_PREFIX / posix if from_intake_parent else posix
    candidate = root.joinpath(*asset_path.parts)
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError("preparation asset is missing or escapes the project root") from exc
    if candidate.absolute() != resolved or not resolved.is_file():
        raise ValueError("preparation asset may not pass through symlinks")
    return resolved.read_bytes()


def _archive_adapter(archive_path: Path) -> bytes:
    if archive_path.is_symlink() or not archive_path.is_file():
        raise ValueError("approved matched-study result archive is missing or unsafe")
    archive_bytes = archive_path.read_bytes()
    if sha256_bytes(archive_bytes) != ADAPTER_ARCHIVE_SHA256:
        raise ValueError("matched-study result archive digest does not match the approved artifact")
    try:
        with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
            if archive.namelist().count(ADAPTER_MEMBER_NAME) != 1:
                raise ValueError("matched-study archive must contain one step-144 adapter")
            info = archive.getinfo(ADAPTER_MEMBER_NAME)
            if info.file_size != ADAPTER_MEMBER_BYTES:
                raise ValueError("matched-study adapter member has the wrong byte size")
            adapter = archive.read(info)
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise ValueError("matched-study result archive is invalid") from exc
    validate_adapter_bytes(adapter)
    from .reload_diagnostic import validate_adapter_header

    validate_adapter_header(adapter)
    return adapter


def _write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()


def create_real_photo_plan(
    project_root: str | Path,
    output_dir: str | Path,
    *,
    run_id: str,
    adapter_archive: str | Path,
) -> dict[str, Any]:
    """Create a private, write-once inference plan after the source/intake gates pass."""
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", run_id):
        raise ValueError("run_id must be a short path-safe identifier")
    root = Path(project_root).resolve(strict=True)
    output = Path(output_dir)
    output_parent = output.parent.resolve(strict=True)
    try:
        output_parent.relative_to(root)
    except ValueError as exc:
        raise ValueError("private real-photo plan must be inside the project root") from exc
    if output.exists() or output.is_symlink():
        raise FileExistsError("real-photo plan directory already exists")
    intake_path = root / INTAKE_RELATIVE_PATH
    preparation_path = root / ".context/real-photo-preparation.json"
    review_path = root / ".context/real-photo-source-geometry-review.json"
    label_path = root / ".context/real-photo-label-confirmation.json"
    source_review_path = root / ".context/real-photo-private-inference-source-review.json"
    exclusions_path = root / ".context/real-photo-development-exclusions.json"
    intake, intake_bytes = _read_json_file(intake_path)
    preparation, _prep_bytes = _read_json_file(preparation_path)
    label_confirmation, label_bytes = _read_json_file(label_path)
    source_review, source_review_bytes = _read_json_file(source_review_path)
    exclusions, _exclusions_bytes = _read_json_file(exclusions_path)
    input_intake = preparation.get("input_intake")
    if not isinstance(input_intake, dict) or input_intake.get("path") != INTAKE_RELATIVE_PATH:
        raise ValueError("preparation intake path does not match the fixed active intake")
    if (
        source_review.get("approved_for_private_inference") is not True
        or source_review.get("human_rights_review") is not False
        or source_review.get("legal_clearance_claim") is not False
        or source_review.get("public_redistribution_clearance") is not False
        or source_review.get("training_clearance") is not False
        or source_review.get("compute_clearance") is not False
        or source_review.get("retain_attribution_and_keep_all_images_private") is not True
        or source_review.get("case_ids") != list(CASE_IDS)
    ):
        raise ValueError("source review does not approve only the scoped private inference use")
    validate_visible_word_confirmation(preparation, label_confirmation)
    validate_preparation_and_intake(preparation, intake, active_intake_bytes=intake_bytes)
    if preparation.get("label_confirmation", {}).get("sha256") != sha256_bytes(label_bytes):
        raise ValueError("preparation does not bind the confirmed visible-word spellings")
    validate_development_exclusions(preparation, exclusions)
    if exclusions.get("source_intake_sha256") != sha256_bytes(intake_bytes):
        raise ValueError("development exclusions do not bind the active intake snapshot")
    from . import remote

    geometry_review_sha256: str | None = None
    geometry_review_status = "not_present"
    if review_path.is_file() and not review_path.is_symlink():
        geometry_review, geometry_review_bytes = _read_json_file(review_path)
        if geometry_review.get("status") == "approved":
            validate_source_geometry_review(preparation, geometry_review)
        else:
            validate_pending_geometry_review(preparation, geometry_review)
        geometry_review_sha256 = sha256_bytes(geometry_review_bytes)
        geometry_review_status = str(geometry_review["status"])
    adapter = _archive_adapter(Path(adapter_archive))
    files: dict[str, bytes] = {ADAPTER_MEMBER_NAME: adapter}
    intake_photos = {
        row["case_id"]: row for row in intake.get("photos", []) if isinstance(row, dict)
    }
    prepared_cases: list[dict[str, Any]] = []
    for case in preparation["cases"]:
        case_id = case["case_id"]
        photo = intake_photos[case_id]
        source = _read_private_file(root, case["source"]["path"], from_intake_parent=True)
        prepared = _read_private_file(root, case["derived"]["path"], from_intake_parent=True)
        mask = _read_private_file(root, case["mask"]["path"], from_intake_parent=True)
        for content, expected, label in (
            (source, case["source"]["sha256"], "source"),
            (prepared, case["derived"]["sha256"], "derived"),
            (mask, case["mask"]["sha256"], "mask"),
        ):
            if sha256_bytes(content) != expected:
                raise ValueError(f"case {case_id} {label} bytes no longer match preparation")
        _open_checked_image(prepared, mode="RGB", role=f"case {case_id} prepared image")
        opened_mask = _open_checked_image(mask, mode="L", role=f"case {case_id} mask")
        mask_values = set(opened_mask.tobytes())
        if not mask_values <= {0, 255} or mask_values != {0, 255}:
            raise ValueError(f"case {case_id} mask must be binary with foreground/background")
        prefix = f"photos/{case_id}"
        source_name = f"{prefix}/source.bin"
        input_name = f"{prefix}/input.png"
        mask_name = f"{prefix}/mask.png"
        files.update({source_name: source, input_name: prepared, mask_name: mask})
        boxes = case["label_boxes"]
        prepared_cases.append(
            {
                "case_id": case_id,
                "split": "dev",
                "source_path": source_name,
                "source_sha256": case["source"]["sha256"],
                "input_path": input_name,
                "input_sha256": case["derived"]["sha256"],
                "mask_path": mask_name,
                "mask_sha256": case["mask"]["sha256"],
                "label_boxes": boxes,
                "label_boxes_sha256": case["label_boxes_sha256"],
            }
        )
        if photo.get("split") != "dev":
            raise ValueError(f"active intake case {case_id} is not development-only")
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "case_ids": list(CASE_IDS),
        "cases": prepared_cases,
        "prompt": PROMPT,
        "generation_settings": GENERATION_SETTINGS,
        "adapter_sha256": sha256_bytes(adapter),
        "training_function_called": False,
        "optimizer_updates": 0,
        "preparation_sha256": preparation["aggregate_sha256"],
        "geometry_review_sha256": geometry_review_sha256,
        "geometry_review_status": geometry_review_status,
        "active_intake_sha256": sha256_bytes(intake_bytes),
        "label_confirmation_sha256": sha256_bytes(label_bytes),
        "private_source_review_sha256": sha256_bytes(source_review_bytes),
        "quality_review_status": (
            "approved" if geometry_review_status == "approved" else "pending_human_geometry_review"
        ),
        "adapter_archive_sha256": ADAPTER_ARCHIVE_SHA256,
        "adapter_member_sha256": ADAPTER_MEMBER_SHA256,
        "modal_sdk": "1.6.1",
        "diffsynth_commit": remote.DIFFSYNTH_COMMIT,
        "runtime_pins": dict(remote.RUNTIME_PINS),
        "model_components": remote.MODEL_COMPONENTS,
    }
    files["manifest.json"] = canonical_json_bytes(manifest)
    payload = canonical_archive(files)
    validate_payload_bytes(payload, run_id)
    output.mkdir(parents=True, exist_ok=False)
    payload_path = output / PAYLOAD_NAME
    _write_once(payload_path, payload)
    plan = {
        "schema_version": 1,
        "run_id": run_id,
        "payload_file": PAYLOAD_NAME,
        "payload_bytes": len(payload),
        "payload_sha256": sha256_bytes(payload),
        "preparation_sha256": preparation["aggregate_sha256"],
        "geometry_review_sha256": geometry_review_sha256,
        "geometry_review_status": geometry_review_status,
        "quality_review_status": manifest["quality_review_status"],
        "active_intake_sha256": manifest["active_intake_sha256"],
        "label_confirmation_sha256": manifest["label_confirmation_sha256"],
        "private_source_review_sha256": manifest["private_source_review_sha256"],
        "case_ids": list(CASE_IDS),
        "adapter_archive_sha256": ADAPTER_ARCHIVE_SHA256,
        "adapter_member_sha256": ADAPTER_MEMBER_SHA256,
        "generation_settings": GENERATION_SETTINGS,
        "schedule": list(generation_schedule()),
        "modal_sdk": "1.6.1",
        "diffsynth_commit": remote.DIFFSYNTH_COMMIT,
        "runtime_pins": dict(remote.RUNTIME_PINS),
        "model_components": remote.MODEL_COMPONENTS,
    }
    marker_bytes = (json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    _write_once(output / RUN_MARKER_NAME, marker_bytes)
    plan["plan_sha256"] = sha256_bytes(marker_bytes)
    return plan


def validate_pending_geometry_review(preparation: dict[str, Any], review: dict[str, Any]) -> None:
    """Bind an unreviewed human-review sidecar without treating it as annotation approval."""
    if (
        review.get("schema_version") != 1
        or review.get("status") != "awaiting_human_review"
        or review.get("preparation_manifest_aggregate_sha256")
        != preparation.get("aggregate_sha256")
    ):
        raise ValueError("pending geometry review does not bind this preparation manifest")
    prep_cases = preparation.get("cases")
    review_cases = review.get("cases")
    if not isinstance(prep_cases, list) or not isinstance(review_cases, list):
        raise ValueError("pending geometry review cases must be arrays")
    prep_by_id = {case.get("case_id"): case for case in prep_cases if isinstance(case, dict)}
    review_by_id = {case.get("case_id"): case for case in review_cases if isinstance(case, dict)}
    if set(prep_by_id) != set(CASE_IDS) or set(review_by_id) != set(CASE_IDS):
        raise ValueError("pending geometry review must cover exactly the selected cases")
    for case_id in CASE_IDS:
        prep_case = prep_by_id[case_id]
        review_case = review_by_id[case_id]
        expected = {
            "source_sha256": prep_case["source"]["sha256"],
            "derived_sha256": prep_case["derived"]["sha256"],
            "mask_sha256": prep_case["mask"]["sha256"],
            "label_boxes_sha256": prep_case["label_boxes_sha256"],
        }
        if (
            any(review_case.get(key) != value for key, value in expected.items())
            or review_case.get("mask_review") != "pending"
            or review_case.get("label_box_review") != "pending"
            or prep_case.get("mask", {}).get("human_approved") is not False
            or prep_case.get("label_boxes_review_status") != "unreviewed"
        ):
            raise ValueError(f"pending geometry review does not match case {case_id}")


def validate_development_exclusions(
    preparation: dict[str, Any], exclusions: dict[str, Any]
) -> None:
    if (
        exclusions.get("schema_version") != 1
        or exclusions.get("status") != "frozen_development_only"
    ):
        raise ValueError("real-photo cases are not frozen as development-only")
    groups = exclusions.get("groups")
    cases = preparation.get("cases")
    if not isinstance(groups, list) or not isinstance(cases, list):
        raise ValueError("development exclusions and preparation cases must be arrays")
    group_by_id = {row.get("case_id"): row for row in groups if isinstance(row, dict)}
    prep_by_id = {row.get("case_id"): row for row in cases if isinstance(row, dict)}
    if set(group_by_id) != set(CASE_IDS) or set(prep_by_id) != set(CASE_IDS):
        raise ValueError("development exclusions must cover exactly the selected cases")
    for case_id in CASE_IDS:
        group = group_by_id[case_id]
        case = prep_by_id[case_id]
        if (
            group.get("development_role") != "dev"
            or group.get("excluded_from") != ["future_train", "final_test"]
            or group.get("product_id") != case.get("product_id")
            or group.get("family_id") != case.get("family_id")
            or case.get("role") != "development_only"
            or case.get("split") != "dev"
        ):
            raise ValueError(f"case {case_id} is not explicitly excluded as development-only")
