"""CPU-only data contract for the paired real-photo preservation comparison."""

from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import TypedDict

from PIL import Image

from . import real_photo_runner as legacy
from . import remote

PROFILE = "preservation-pair"
MAX_INPUT_BYTES = 240 * 1024 * 1024
MAX_RESULT_BYTES = 24 * 1024 * 1024
CASE_IDS = legacy.CASE_IDS
RUN_MARKER_NAME = legacy.RUN_MARKER_NAME
PAYLOAD_NAME = legacy.PAYLOAD_NAME
RESULT_NAME = legacy.RESULT_NAME
LEGACY_INPUT_SHA256 = "e7e2ae360ea698b94dedad4598ca358fd5680d3185f724c36075a2ad384a21d2"
LEGACY_RUN_ID = "real-photo-off-r1-2026-10-10-r2"
CONFIRMATION_SHA256 = "45c18d485994ee017424c58e81abc2ade5ee54a904a2940c9bed1fd1e65e5981"
CONFIRMATION_PATH = ".context/real-photo-current-confirmation-20261011.json"
PREPARATION_SHA256 = "8133eacd8cbe4cb3781b0693bfe55d5f430ad5ddafe8302bef5d1b804fd01c3f"
GEOMETRY_REVIEW_SHA256 = "9ccdf08e8d4bb1ef5403b01b74eec11584e43d300041fa246a20e3e567d4f7ad"
ACTIVE_INTAKE_SHA256 = "ad918745583f1fd07716b2bf253c705d41e36613107b3941726f32cdb31d033e"
LABEL_CONFIRMATION_SHA256 = "7b8e346430a64939a6e9448dd5be09b4864be7bb7aa7f5f469accc1192a78ddb"
PRIVATE_SOURCE_REVIEW_SHA256 = "649b68d11552b9a618deb6beaffddae326535876f29d6f25bda2145653d85065"
LEGACY_CASE_PINS = {
    "OFF-R1-003": {
        "source_sha256": "92f7102fceecbe14b840e1cd4e70fcad7cc48107346aad2180383980552873ff",
        "input_sha256": "63ea88399bfe48ee5b16a3e69d8c10fc288727e546b2680efa127df0a8f7a987",
        "mask_sha256": "05a549ca1c51f59214c89668b9919a95313ce828e6a32d8eb18bc8bbf339a5b7",
    },
    "OFF-R1-005": {
        "source_sha256": "e6b8cd88dd3e0ce0d3b81ff6386d0e4fdf75eefb1a980569575eb4df05df9182",
        "input_sha256": "f33f569e3a2e50ac74e3f81bd91867c66b36d57ee367f7d660aec08911f2aee3",
        "mask_sha256": "67f1aeab8898632a44f438f85ac44aba63dafc2550104f99ff934e13c9117c45",
    },
    "OFF-R1-006": {
        "source_sha256": "88cb731a5f974c3685d3bb401ed330088bf10d6a5209e305f494414182faf867",
        "input_sha256": "76240c40afca6151982f9fdd2b8ae7895200ad7afdf8ab2b1856af492f13e33c",
        "mask_sha256": "9f5e9316478bdbdbf3db09a5e06512c665929c79bbd02ba57006687ce250ee32",
    },
}
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

ADAPTER_PINS: dict[str, dict[str, object]] = {
    "prior": {
        "run_id": "scenetune-controlled3d-2026-10-11-r1",
        "archive_path": (
            "artifacts/controlled3d-studies/scenetune-controlled3d-2026-10-11-r1/"
            "matched-study-result.zip"
        ),
        "archive_sha256": "433c3e71aee4227ea12c3ffa9d136325ac771437da85764f1d8a12c7a385cc37",
        "receipt_member": "receipt.json",
        "receipt_sha256": "872867c587cb032d819f71c12c0204658b24fae800ef12fe83ea720b0768d5e5",
        "study_type": "controlled3d144",
        "member_name": "adapter/step-144.safetensors",
        "member_sha256": "7a58eb3ec53f90835f14a657d646582c4c353ab6b341e948d0ce1e4bcf3637aa",
        "member_bytes": 96_364_672,
        "payload_path": "adapters/prior/step-144.safetensors",
    },
    "new": {
        "run_id": "scenetune-preservation-2026-10-11-r2",
        "archive_path": (
            "artifacts/preservation-studies/scenetune-preservation-2026-10-11-r2/"
            "matched-study-result.zip"
        ),
        "archive_sha256": "3ecfd6ab71f2e0097baa56942f01138d28584c4d2383574d48e0bcb4fdc305ec",
        "receipt_member": "receipt.json",
        "receipt_sha256": "ab05189c62554ff41ae73bb30345886eafd3582a9a8aa3b551f305ccab73bc11",
        "study_type": "preservation144",
        "member_name": "adapter/step-144.safetensors",
        "member_sha256": "40e5314a2259dd46b477f38b319073266c65749fd623954b095f9925b1ebe42e",
        "member_bytes": 96_364_672,
        "payload_path": "adapters/new/step-144.safetensors",
    },
}


class GenerationTask(TypedDict):
    case_id: str
    arm: str
    session: str


def generation_schedule() -> tuple[GenerationTask, ...]:
    """Return five fixed arms, each in the frozen case order."""
    sessions = (
        ("base", "base"),
        ("prior", "prior-strict-1"),
        ("prior_reloaded", "prior-strict-2"),
        ("new", "new-strict-1"),
        ("new_reloaded", "new-strict-2"),
    )
    return tuple(
        {"case_id": case_id, "arm": arm, "session": session}
        for arm, session in sessions
        for case_id in CASE_IDS
    )


_BINDING_FIELDS = (
    "profile",
    "run_id",
    "case_ids",
    "cases",
    "legacy_input_sha256",
    "legacy_run_id",
    "preparation_sha256",
    "label_confirmation_sha256",
    "confirmation",
    "adapters",
    "prompt",
    "generation_settings",
    "schedule",
    "modal_sdk",
    "diffsynth_commit",
    "runtime_pins",
    "model_components",
    "trainer",
    "training_function_called",
    "optimizer_updates",
)
_PROVENANCE_FIELDS = (
    "geometry_review_sha256",
    "geometry_review_status",
    "quality_review_status",
    "active_intake_sha256",
    "private_source_review_sha256",
)


def clearance_bindings(marker: dict[str, object]) -> dict[str, object]:
    """Return the immutable science and provenance fields bound through every stage."""
    missing = [field for field in _BINDING_FIELDS if field not in marker]
    if missing or marker.get("profile") != PROFILE:
        raise ValueError("preservation-pair marker is missing fixed clearance bindings")
    return {field: marker[field] for field in _BINDING_FIELDS}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _read_bytes(value: str | Path | bytes, name: str) -> bytes:
    if isinstance(value, bytes):
        return value
    try:
        return Path(value).read_bytes()
    except OSError as exc:
        raise ValueError(f"{name} cannot be read") from exc


def _read_archive(data: bytes, *, limit: int, role: str) -> dict[str, bytes]:
    if not data or len(data) > limit:
        raise ValueError(f"{role} is empty or exceeds its byte limit")
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            return legacy._safe_archive(archive, total_limit=limit)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"{role} is not a valid ZIP archive") from exc


def _canonical_pair_archive(files: dict[str, bytes]) -> bytes:
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, content in sorted(files.items()):
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError("preservation-pair archive path is unsafe")
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, content)
    result = stream.getvalue()
    if len(result) > MAX_INPUT_BYTES:
        raise ValueError("preservation-pair payload exceeds the input byte limit")
    return result


def _checked_json(data: bytes, role: str) -> dict[str, object]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{role} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object")
    return value


def _validate_confirmation(
    data: bytes, cases: list[dict[str, object]], *, expected_sha256: str
) -> dict[str, object]:
    if not _is_digest(expected_sha256) or _sha256(data) != expected_sha256:
        raise ValueError("confirmation sidecar SHA-256 does not match its fixed pin")
    confirmation = _checked_json(data, "confirmation sidecar")
    if (
        confirmation.get("schema_version") != 1
        or confirmation.get("status") != "current_source_confirmation_provenance_verified"
        or confirmation.get("reviewed_by") != "Raj"
        or confirmation.get("new_user_confirmation") is not False
        or confirmation.get("annotations_changed") is not False
        or confirmation.get("new_compute") is not False
    ):
        raise ValueError("confirmation sidecar does not match the reviewed provenance")
    confirmed_cases = confirmation.get("cases")
    if not isinstance(confirmed_cases, list) or len(confirmed_cases) != len(CASE_IDS):
        raise ValueError("confirmation sidecar must cover the exact three cases")
    for case_id, confirmed, case in zip(CASE_IDS, confirmed_cases, cases, strict=True):
        if not isinstance(confirmed, dict) or confirmed.get("case_id") != case_id:
            raise ValueError("confirmation sidecar cases are missing or out of order")
        for field in (
            "product_id",
            "family_id",
            "role",
            "label_boxes",
            "label_boxes_canonical_sha256",
            "label_box_geometry",
            "blue_outline_review",
            "whole_package_shape_scoring_in_scope",
            "masked_product_pixel_scoring_in_scope",
            "scoring_applicability",
            "exclusion_reason",
        ):
            case_field = {
                "label_boxes": "label_boxes",
                "label_boxes_canonical_sha256": "label_boxes_sha256",
            }.get(field, field)
            confirmation_field = "labels_and_boxes" if field == "label_boxes" else field
            if confirmed.get(confirmation_field) != case.get(case_field):
                raise ValueError(f"confirmation sidecar does not bind case {case_id} {field}")
    rights = confirmation.get("rights_and_use")
    if not isinstance(rights, dict) or (
        rights.get("rights_review") != "pending"
        or rights.get("authorized_scope") != "private inference development only"
        or rights.get("training_authorized") is not False
        or rights.get("public_redistribution_authorized") is not False
    ):
        raise ValueError("confirmation sidecar rights and use must remain restricted")
    return confirmation


def _validate_cases(
    manifest: dict[str, object], files: dict[str, bytes]
) -> list[dict[str, object]]:
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != len(CASE_IDS):
        raise ValueError("preservation-pair manifest must contain exactly three cases")
    case_fields = {
        "case_id",
        "product_id",
        "family_id",
        "role",
        "source_sha256",
        "input_path",
        "input_sha256",
        "mask_path",
        "mask_sha256",
        "label_boxes",
        "label_boxes_sha256",
        "label_box_geometry",
        "blue_outline_review",
        "whole_package_shape_scoring_in_scope",
        "masked_product_pixel_scoring_in_scope",
        "scoring_applicability",
        "exclusion_reason",
    }
    expected_files = {"manifest.json"}
    for case_id, case in zip(CASE_IDS, cases, strict=True):
        if not isinstance(case, dict) or set(case) != case_fields or case.get("case_id") != case_id:
            raise ValueError("preservation-pair case rows are invalid or out of order")
        if case.get("role") != "development_only" or not _is_digest(case.get("source_sha256")):
            raise ValueError(f"case {case_id} is not bound to a development-only source")
        pins = LEGACY_CASE_PINS[case_id]
        if any(case.get(field) != digest for field, digest in pins.items()):
            raise ValueError(f"case {case_id} does not match its immutable legacy input hashes")
        expected_outline = "confirmed_by_Raj" if case_id != "OFF-R1-006" else "not_confirmed_by_Raj"
        scoreable = case_id != "OFF-R1-006"
        if (
            case.get("label_box_geometry") != "confirmed_by_Raj"
            or case.get("blue_outline_review") != expected_outline
            or case.get("whole_package_shape_scoring_in_scope") is not scoreable
            or case.get("masked_product_pixel_scoring_in_scope") is not scoreable
            or not isinstance(case.get("scoring_applicability"), str)
            or not case["scoring_applicability"]
            or (not scoreable and not isinstance(case.get("exclusion_reason"), str))
            or (scoreable and case.get("exclusion_reason") is not None)
        ):
            raise ValueError(f"case {case_id} has an invalid geometry or scoring scope")
        boxes = case.get("label_boxes")
        if not isinstance(boxes, list) or case.get("label_boxes_sha256") != legacy.canonical_sha256(
            boxes
        ):
            raise ValueError(f"case {case_id} label boxes do not match their canonical digest")
        for role in ("input", "mask"):
            path_field = f"{role}_path"
            digest_field = f"{role}_sha256"
            path_value = case.get(path_field)
            if not isinstance(path_value, str):
                raise ValueError(f"case {case_id} {role} path is missing")
            path = PurePosixPath(path_value)
            if (
                path.is_absolute()
                or path.as_posix() != path_value
                or path.parts != ("photos", case_id, f"{role}.png")
                or path_value not in files
                or not _is_digest(case.get(digest_field))
                or _sha256(files[path_value]) != case[digest_field]
            ):
                raise ValueError(f"case {case_id} {role} path or digest is invalid")
            expected_files.add(path_value)
            mode = "RGB" if role == "input" else "L"
            legacy._open_checked_image(files[path_value], mode=mode, role=f"case {case_id} {role}")
            if role == "mask":
                with Image.open(BytesIO(files[path_value])) as mask_image:
                    pixels = set(mask_image.tobytes())
                if pixels != {0, 255}:
                    raise ValueError(
                        f"case {case_id} mask must be binary with foreground/background"
                    )
    return cases


def validate_payload_bytes(data: bytes, run_id: str) -> tuple[dict[str, object], dict[str, bytes]]:
    """Validate the exact bounded pair payload and all immutable provenance bindings."""
    if not isinstance(data, bytes) or not data or len(data) > MAX_INPUT_BYTES:
        raise ValueError("preservation-pair payload is empty or exceeds the input limit")
    files = _read_archive(data, limit=MAX_INPUT_BYTES, role="preservation-pair payload")
    manifest = _checked_json(files.get("manifest.json", b""), "preservation-pair manifest")
    expected_manifest_fields = {
        "schema_version",
        *_BINDING_FIELDS,
        *_PROVENANCE_FIELDS,
    }
    if set(manifest) != expected_manifest_fields:
        raise ValueError("preservation-pair manifest has missing or unknown fields")
    bindings = clearance_bindings(manifest)
    if (
        manifest.get("schema_version") != 1
        or bindings["run_id"] != run_id
        or bindings["case_ids"] != list(CASE_IDS)
        or bindings["legacy_input_sha256"] != LEGACY_INPUT_SHA256
        or bindings["legacy_run_id"] != LEGACY_RUN_ID
        or bindings["prompt"] != legacy.PROMPT
        or bindings["generation_settings"] != legacy.GENERATION_SETTINGS
        or bindings["schedule"] != list(generation_schedule())
        or bindings["modal_sdk"] != "1.6.1"
        or bindings["diffsynth_commit"] != remote.DIFFSYNTH_COMMIT
        or bindings["runtime_pins"] != remote.RUNTIME_PINS
        or bindings["model_components"] != remote.MODEL_COMPONENTS
        or bindings["trainer"]
        != {
            "repo": "https://github.com/modelscope/DiffSynth-Studio",
            "commit": remote.DIFFSYNTH_COMMIT,
        }
        or bindings["training_function_called"] is not False
        or bindings["optimizer_updates"] != 0
        or manifest.get("geometry_review_status") != "awaiting_human_review"
        or manifest.get("quality_review_status") != "pending_human_geometry_review"
        or not all(
            _is_digest(manifest.get(key))
            for key in (
                "preparation_sha256",
                "geometry_review_sha256",
                "active_intake_sha256",
                "label_confirmation_sha256",
                "private_source_review_sha256",
            )
        )
    ):
        raise ValueError("preservation-pair payload does not match the fixed inference contract")
    cases = _validate_cases(manifest, files)
    adapters = bindings.get("adapters")
    if not isinstance(adapters, dict) or set(adapters) != {"prior", "new"}:
        raise ValueError("preservation-pair adapters must bind the fixed prior and new artifacts")
    expected_files = {"manifest.json"}
    for arm in ("prior", "new"):
        pin = ADAPTER_PINS[arm]
        entry = adapters.get(arm)
        if not isinstance(entry, dict) or entry != pin:
            raise ValueError(f"{arm} adapter metadata does not match its fixed archive pin")
        member_path = pin["payload_path"]
        member = files.get(str(member_path))
        if (
            not isinstance(member, bytes)
            or len(member) != pin["member_bytes"]
            or _sha256(member) != pin["member_sha256"]
        ):
            raise ValueError(f"{arm} adapter payload does not match its pinned digest")
        expected_files.add(str(member_path))
    confirmation = bindings.get("confirmation")
    if not isinstance(confirmation, dict) or set(confirmation) != {"path", "member", "sha256"}:
        raise ValueError("preservation-pair confirmation binding is invalid")
    if (
        confirmation.get("path") != CONFIRMATION_PATH
        or confirmation.get("sha256") != CONFIRMATION_SHA256
    ):
        raise ValueError("preservation-pair confirmation does not match its fixed pin")
    confirmation_member = confirmation.get("member")
    if not isinstance(confirmation_member, str) or confirmation_member not in files:
        raise ValueError("preservation-pair confirmation member is missing")
    confirmation_bytes = files[confirmation_member]
    case_rows = cases
    _validate_confirmation(confirmation_bytes, case_rows, expected_sha256=CONFIRMATION_SHA256)
    expected_files.update(
        (
            confirmation_member,
            *(str(case["input_path"]) for case in cases),
            *(str(case["mask_path"]) for case in cases),
        )
    )
    if set(files) != expected_files:
        raise ValueError("preservation-pair payload contains unexpected or unreferenced members")
    return manifest, files


def _validate_adapter_archive(value: str | Path | bytes, arm: str) -> tuple[bytes, bytes]:
    archive_bytes = _read_bytes(value, f"{arm} adapter archive")
    pin = ADAPTER_PINS[arm]
    if _sha256(archive_bytes) != pin["archive_sha256"]:
        raise ValueError(f"{arm} adapter archive SHA-256 does not match its fixed pin")
    members = _read_archive(archive_bytes, limit=MAX_INPUT_BYTES, role=f"{arm} adapter archive")
    receipt_name = str(pin["receipt_member"])
    member_name = str(pin["member_name"])
    receipt = members.get(receipt_name)
    adapter = members.get(member_name)
    if (
        not isinstance(receipt, bytes)
        or _sha256(receipt) != pin["receipt_sha256"]
        or not isinstance(adapter, bytes)
        or len(adapter) != pin["member_bytes"]
        or _sha256(adapter) != pin["member_sha256"]
    ):
        raise ValueError(f"{arm} adapter archive receipt or model member does not match its pin")
    receipt_value = _checked_json(receipt, f"{arm} adapter receipt")
    if receipt_value.get("run_id") != pin["run_id"] or receipt_value.get("status") != "passed":
        raise ValueError(f"{arm} adapter receipt does not identify the approved successful run")
    return archive_bytes, adapter


def _write_once(path: Path, data: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(data)
    except FileExistsError as exc:
        raise ValueError(f"refusing to overwrite existing {path.name}") from exc


def create_comparison_plan(
    root: str | Path,
    output_dir: str | Path,
    *,
    run_id: str,
    legacy_input: str | Path | bytes,
    prior_archive: str | Path | bytes,
    new_archive: str | Path | bytes,
    confirmation: str | Path | bytes,
) -> dict[str, object]:
    """Create a write-once, local plan from the three pinned inputs and reviewed sidecar."""
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("run_id must be a short path-safe identifier")
    root_path = Path(root).resolve(strict=True)
    legacy_bytes = _read_bytes(legacy_input, "legacy input archive")
    if _sha256(legacy_bytes) != LEGACY_INPUT_SHA256:
        raise ValueError("legacy input archive SHA-256 does not match its fixed pin")
    legacy_manifest, legacy_files = legacy.validate_payload_bytes(legacy_bytes, LEGACY_RUN_ID)
    _prior_archive_bytes, prior_adapter = _validate_adapter_archive(prior_archive, "prior")
    _new_archive_bytes, new_adapter = _validate_adapter_archive(new_archive, "new")
    confirmation_bytes = _read_bytes(confirmation, "confirmation sidecar")
    if _sha256(confirmation_bytes) != CONFIRMATION_SHA256:
        raise ValueError("confirmation sidecar SHA-256 does not match its fixed pin")
    confirmation_value = _checked_json(confirmation_bytes, "confirmation sidecar")
    confirmation_cases = confirmation_value.get("cases")
    if not isinstance(confirmation_cases, list) or len(confirmation_cases) != len(CASE_IDS):
        raise ValueError("confirmation sidecar must contain the exact three cases")
    source_cases = legacy_manifest.get("cases")
    if not isinstance(source_cases, list) or len(source_cases) != len(CASE_IDS):
        raise ValueError("legacy real-photo payload does not contain the exact three cases")
    prepared_cases: list[dict[str, object]] = []
    files: dict[str, bytes] = {}
    for case_id, legacy_case, confirmed in zip(
        CASE_IDS, source_cases, confirmation_cases, strict=True
    ):
        if (
            not isinstance(legacy_case, dict)
            or not isinstance(confirmed, dict)
            or confirmed.get("case_id") != case_id
        ):
            raise ValueError("legacy and confirmation cases are missing or out of order")
        input_path = str(legacy_case.get("input_path", ""))
        mask_path = str(legacy_case.get("mask_path", ""))
        input_name, mask_name = f"photos/{case_id}/input.png", f"photos/{case_id}/mask.png"
        input_bytes, mask_bytes = legacy_files.get(input_path), legacy_files.get(mask_path)
        if not isinstance(input_bytes, bytes) or not isinstance(mask_bytes, bytes):
            raise ValueError(f"legacy payload is missing fixed input or mask for {case_id}")
        source_pin = confirmed.get("source")
        prepared_pin = confirmed.get("prepared_image")
        mask_pin = confirmed.get("mask")
        if (
            not isinstance(source_pin, dict)
            or source_pin.get("sha256") != legacy_case.get("source_sha256")
            or not isinstance(prepared_pin, dict)
            or prepared_pin.get("sha256") != _sha256(input_bytes)
            or not isinstance(mask_pin, dict)
            or mask_pin.get("sha256") != _sha256(mask_bytes)
        ):
            raise ValueError(
                f"legacy input bytes do not match the current confirmation for {case_id}"
            )
        files[input_name], files[mask_name] = input_bytes, mask_bytes
        expected_fields = (
            "product_id",
            "family_id",
            "role",
            "label_boxes_canonical_sha256",
            "label_box_geometry",
            "blue_outline_review",
            "whole_package_shape_scoring_in_scope",
            "masked_product_pixel_scoring_in_scope",
            "scoring_applicability",
            "exclusion_reason",
        )
        row: dict[str, object] = {
            "case_id": case_id,
            "product_id": confirmed.get("product_id"),
            "family_id": confirmed.get("family_id"),
            "role": confirmed.get("role"),
            "source_sha256": legacy_case.get("source_sha256"),
            "input_path": input_name,
            "input_sha256": _sha256(input_bytes),
            "mask_path": mask_name,
            "mask_sha256": _sha256(mask_bytes),
            "label_boxes": confirmed.get("labels_and_boxes"),
            "label_boxes_sha256": confirmed.get("label_boxes_canonical_sha256"),
            "label_box_geometry": confirmed.get("label_box_geometry"),
            "blue_outline_review": confirmed.get("blue_outline_review"),
            "whole_package_shape_scoring_in_scope": confirmed.get(
                "whole_package_shape_scoring_in_scope"
            ),
            "masked_product_pixel_scoring_in_scope": confirmed.get(
                "masked_product_pixel_scoring_in_scope"
            ),
            "scoring_applicability": confirmed.get("scoring_applicability"),
            "exclusion_reason": confirmed.get("exclusion_reason"),
        }
        if any(key not in confirmed for key in expected_fields):
            raise ValueError(f"confirmation sidecar case {case_id} is incomplete")
        prepared_cases.append(row)
    _validate_confirmation(confirmation_bytes, prepared_cases, expected_sha256=CONFIRMATION_SHA256)
    files.update(
        {
            str(ADAPTER_PINS["prior"]["payload_path"]): prior_adapter,
            str(ADAPTER_PINS["new"]["payload_path"]): new_adapter,
            CONFIRMATION_PATH: confirmation_bytes,
        }
    )
    from . import real_photo_runner as legacy_module

    generation_settings = dict(legacy_module.GENERATION_SETTINGS)
    bindings: dict[str, object] = {
        "profile": PROFILE,
        "run_id": run_id,
        "case_ids": list(CASE_IDS),
        "cases": prepared_cases,
        "legacy_input_sha256": LEGACY_INPUT_SHA256,
        "legacy_run_id": LEGACY_RUN_ID,
        "preparation_sha256": PREPARATION_SHA256,
        "geometry_review_sha256": GEOMETRY_REVIEW_SHA256,
        "geometry_review_status": "awaiting_human_review",
        "quality_review_status": "pending_human_geometry_review",
        "active_intake_sha256": ACTIVE_INTAKE_SHA256,
        "label_confirmation_sha256": LABEL_CONFIRMATION_SHA256,
        "private_source_review_sha256": PRIVATE_SOURCE_REVIEW_SHA256,
        "confirmation": {
            "path": CONFIRMATION_PATH,
            "member": CONFIRMATION_PATH,
            "sha256": CONFIRMATION_SHA256,
        },
        "adapters": ADAPTER_PINS,
        "prompt": legacy.PROMPT,
        "generation_settings": generation_settings,
        "schedule": list(generation_schedule()),
        "modal_sdk": "1.6.1",
        "diffsynth_commit": remote.DIFFSYNTH_COMMIT,
        "runtime_pins": dict(remote.RUNTIME_PINS),
        "model_components": remote.MODEL_COMPONENTS,
        "trainer": {
            "repo": "https://github.com/modelscope/DiffSynth-Studio",
            "commit": remote.DIFFSYNTH_COMMIT,
        },
        "training_function_called": False,
        "optimizer_updates": 0,
    }
    manifest = {"schema_version": 1, **bindings}
    files["manifest.json"] = legacy.canonical_json_bytes(manifest)
    payload = _canonical_pair_archive(files)
    if len(payload) > MAX_INPUT_BYTES:
        raise ValueError("preservation-pair payload exceeds the input byte limit")
    validate_payload_bytes(payload, run_id)
    plan: dict[str, object] = {
        **bindings,
        "schema_version": 1,
        "payload_file": PAYLOAD_NAME,
        "payload_bytes": len(payload),
        "payload_sha256": _sha256(payload),
    }
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = root_path / output_path
    try:
        output_path.parent.resolve(strict=True).relative_to(root_path)
    except (OSError, ValueError) as exc:
        raise ValueError("comparison plan must be inside the project root") from exc
    if output_path.exists() or output_path.is_symlink():
        raise ValueError("refusing to replace an existing comparison plan directory")
    try:
        output_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ValueError("refusing to replace an existing comparison plan directory") from exc
    _write_once(output_path / PAYLOAD_NAME, payload)
    marker_bytes = (json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    _write_once(output_path / RUN_MARKER_NAME, marker_bytes)
    plan["plan_sha256"] = _sha256(marker_bytes)
    return plan


def load_plan_record(output_dir: str | Path) -> dict[str, object]:
    """Load the immutable marker and verify its payload before returning it."""
    directory = Path(output_dir)
    try:
        marker_bytes = (directory / RUN_MARKER_NAME).read_bytes()
        plan = _checked_json(marker_bytes, "comparison plan marker")
        payload = (directory / PAYLOAD_NAME).read_bytes()
    except OSError as exc:
        raise ValueError("comparison plan marker or payload is missing") from exc
    expected_fields = {
        "schema_version",
        *_BINDING_FIELDS,
        *_PROVENANCE_FIELDS,
        "payload_file",
        "payload_bytes",
        "payload_sha256",
    }
    if set(plan) != expected_fields or plan.get("schema_version") != 1:
        raise ValueError("comparison plan marker has missing or unknown fields")
    if plan.get("profile") != PROFILE:
        raise ValueError("comparison plan marker profile is invalid")
    if (
        plan.get("payload_file") != PAYLOAD_NAME
        or plan.get("payload_bytes") != len(payload)
        or plan.get("payload_sha256") != _sha256(payload)
        or not _is_digest(plan.get("payload_sha256"))
    ):
        raise ValueError("comparison plan marker does not bind its payload")
    manifest, _ = validate_payload_bytes(payload, str(plan.get("run_id", "")))
    if clearance_bindings(plan) != clearance_bindings(manifest) or any(
        plan.get(key) != manifest.get(key) for key in _PROVENANCE_FIELDS
    ):
        raise ValueError("comparison plan marker does not match its payload manifest")
    plan["plan_sha256"] = _sha256(marker_bytes)
    return plan


def validate_result_bundle(
    data: bytes,
    run_id: str,
    *,
    expected_plan: dict[str, object],
    expected_payload_sha256: str,
) -> tuple[dict[str, object], set[str]]:
    """Validate a bounded pair receipt, hashes, images, strict loads, and six exact replays."""
    if not isinstance(data, bytes) or not data or len(data) > MAX_RESULT_BYTES:
        raise ValueError("preservation-pair result is empty or exceeds the output limit")
    files = _read_archive(data, limit=MAX_RESULT_BYTES, role="preservation-pair result")
    receipt = _checked_json(files.get("receipt.json", b""), "preservation-pair receipt")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("profile") != PROFILE
        or receipt.get("run_id") != run_id
        or receipt.get("status") not in {"passed", "failed"}
        or receipt.get("generation_settings") != legacy.GENERATION_SETTINGS
        or receipt.get("schedule") != list(generation_schedule())
        or receipt.get("training_function_called") is not False
        or receipt.get("optimizer_updates") != 0
        or receipt.get("quality_claim") is not False
        or receipt.get("worker_failure_confirmed") is not (receipt.get("status") == "failed")
    ):
        raise ValueError("preservation-pair receipt does not match the fixed inference contract")
    if (
        expected_plan.get("run_id") != run_id
        or expected_plan.get("payload_sha256") != expected_payload_sha256
    ):
        raise ValueError("expected plan and payload digest are inconsistent")
    expected_bindings = clearance_bindings(expected_plan)
    try:
        actual_bindings = clearance_bindings(receipt)
    except ValueError as exc:
        raise ValueError("preservation-pair receipt is missing clearance bindings") from exc
    if (
        actual_bindings != expected_bindings
        or receipt.get("payload_sha256") != expected_payload_sha256
    ):
        raise ValueError("preservation-pair receipt does not match its immutable plan or payload")
    for key in (
        "geometry_review_sha256",
        "geometry_review_status",
        "quality_review_status",
        "active_intake_sha256",
        "private_source_review_sha256",
    ):
        if receipt.get(key) != expected_plan.get(key):
            raise ValueError(
                "preservation-pair receipt source-review bindings do not match its plan"
            )
    artifacts = receipt.get("output_artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("preservation-pair receipt output_artifacts must be an array")
    expected_paths = {
        f"outputs/{task['case_id']}/{task['arm']}.png" for task in generation_schedule()
    }
    by_path: dict[str, dict[str, object]] = {}
    for artifact in artifacts:
        if (
            not isinstance(artifact, dict)
            or set(artifact) != {"path", "sha256", "width", "height"}
            or artifact.get("path") not in expected_paths
            or artifact.get("path") in by_path
            or not _is_digest(artifact.get("sha256"))
            or artifact.get("width") != 512
            or artifact.get("height") != 512
        ):
            raise ValueError("preservation-pair receipt contains invalid output metadata")
        by_path[str(artifact["path"])] = artifact
    image_paths = set(files) - {"receipt.json"}
    if image_paths != set(by_path) or not image_paths <= expected_paths:
        raise ValueError("preservation-pair result members do not match the receipt output list")
    opened: dict[str, Image.Image] = {}
    for path, artifact in by_path.items():
        content = files[path]
        if _sha256(content) != artifact["sha256"]:
            raise ValueError(f"preservation-pair output digest does not match for {path}")
        opened[path] = legacy._open_checked_image(content, mode="RGB", role=path)
    replay_claims = receipt.get("replay_pixel_equal")
    replay_ok = True
    if replay_claims is None and receipt.get("status") == "passed":
        raise ValueError("passed preservation-pair result is missing its replay claims")
    if replay_claims is not None:
        if not isinstance(replay_claims, dict) or set(replay_claims) != set(CASE_IDS):
            raise ValueError("preservation-pair receipt must report all six replay checks")
        for case_id in CASE_IDS:
            case_claims = replay_claims.get(case_id)
            if not isinstance(case_claims, dict) or set(case_claims) != {"prior", "new"}:
                raise ValueError("preservation-pair replay claim shape is invalid")
            for arm in ("prior", "new"):
                claim = case_claims.get(arm)
                if claim is not None and not isinstance(claim, bool):
                    raise ValueError("preservation-pair replay claims must be booleans or null")
                first = opened.get(f"outputs/{case_id}/{arm}.png")
                replay = opened.get(f"outputs/{case_id}/{arm}_reloaded.png")
                if first is not None and replay is not None:
                    equal = first.tobytes() == replay.tobytes()
                    if claim is not equal:
                        raise ValueError(
                            "preservation-pair replay equality claim does not match output pixels"
                        )
                    replay_ok = replay_ok and equal
                elif claim is not None:
                    raise ValueError("preservation-pair replay claim has no matching image pair")
                elif receipt.get("status") == "passed":
                    raise ValueError("passed preservation-pair result is missing a replay image")
    strict_expected = [
        ("prior", "prior-strict-1"),
        ("prior", "prior-strict-2"),
        ("new", "new-strict-1"),
        ("new", "new-strict-2"),
    ]
    strict = receipt.get("strict_load_validations")
    if (
        not isinstance(strict, list)
        or len(strict) > 4
        or (receipt.get("status") == "passed" and len(strict) != 4)
    ):
        raise ValueError("preservation-pair receipt has an invalid strict-load validation count")
    for row, (arm, session) in zip(strict, strict_expected, strict=False):
        validation = row.get("validation") if isinstance(row, dict) else None
        observed = validation.get("observed") if isinstance(validation, dict) else None
        adapter_validation = (
            validation.get("adapter_validation") if isinstance(validation, dict) else None
        )
        if (
            not isinstance(row, dict)
            or set(row) != {"arm", "session", "validation"}
            or row.get("arm") != arm
            or row.get("session") != session
            or not isinstance(validation, dict)
            or set(validation) != {"observed", "mode_changed", "adapter_validation"}
            or validation.get("mode_changed") is not False
            or not isinstance(observed, dict)
            or set(observed) != {"pipe_training", "lora_dropout_training"}
            or not isinstance(observed.get("pipe_training"), bool)
            or not isinstance(observed.get("lora_dropout_training"), list)
            or any(not isinstance(mode, bool) for mode in observed["lora_dropout_training"])
            or adapter_validation
            != {
                "validated": True,
                "restored_tensor_count": 200,
                "module_count": 100,
            }
        ):
            raise ValueError("preservation-pair strict-load validation order is invalid")
    strict_observations = [row["validation"]["observed"] for row in strict]
    if strict_observations and any(
        observed != strict_observations[0] for observed in strict_observations[1:]
    ):
        raise ValueError("preservation-pair strict-load observed modes differ between sessions")
    if receipt.get("status") == "passed":
        if (
            set(by_path) != expected_paths
            or not replay_ok
            or any(row["validation"].get("mode_changed") is not False for row in strict)
        ):
            raise ValueError(
                "passed preservation-pair result is incomplete or failed replay/load checks"
            )
    else:
        failure = receipt.get("failure")
        if not isinstance(failure, dict) or not all(
            isinstance(failure.get(key), str) and failure[key]
            for key in ("stage", "type", "message")
        ):
            raise ValueError("failed preservation-pair result is missing bounded failure evidence")
    elapsed = receipt.get("elapsed_seconds")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or elapsed < 0
    ):
        raise ValueError("preservation-pair receipt has invalid elapsed-time evidence")
    return receipt, image_paths
