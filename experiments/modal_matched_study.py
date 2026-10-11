"""Default to CPU-only matched-study planning; bound any explicit Modal run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import zipfile
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

from experiments import modal_pilot
from product_scene_editor import controlled3d_study, matched_study, preservation_study, remote

APP_NAME = "product-scene-editor-matched-study"
RESULT_NAME = "matched-study-result.zip"
MODAL_FAILURE_NAME = "matched-study-modal-failure-receipt.json"
UNVERIFIED_RESULT_NAME = "matched-study-unverified-result.bin"
ATTEMPT_MARKER_NAME = "matched-study-modal-attempt.json"
REMOTE_TIMEOUT_SECONDS = 2700
REMOTE_STARTUP_TIMEOUT_SECONDS = 300
DEFAULT_RUN_ID = "klein-edit-matched-2026-10-09-r1"
_write_json_once = modal_pilot._write_json_once
_build_image = modal_pilot._build_image


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("artifacts/pilot-data-2026-10-08-r2")
    )
    parser.add_argument("--plan-dir", type=Path)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--study-profile",
        choices=("matched144", "controlled3d144", "preservation144"),
        default="matched144",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--train-export-dir", type=Path)
    parser.add_argument("--parent-payload", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--launch", action="store_true", help="run one explicitly cleared study")
    mode.add_argument("--recover", action="store_true", help="observe only the recorded Modal call")
    parser.add_argument("--clearance", type=Path)
    parser.add_argument("--profile")
    parser.add_argument("--workspace")
    parser.add_argument("--environment")
    return parser


def _remote_worker(payload: bytes, run_id: str, study_profile: str = "matched144") -> bytes:
    if study_profile == "controlled3d144":
        return remote.run_remote_controlled3d_study(payload, run_id)
    if study_profile == "preservation144":
        return remote.run_remote_preservation_study(payload, run_id)
    if study_profile != "matched144":
        raise ValueError("unknown study profile")
    return remote.run_remote_matched_study(payload, run_id)


def _study_spec(study_profile: str) -> tuple[Any, str, int, int]:
    if study_profile == "controlled3d144":
        return controlled3d_study, "controlled3d144", 18, 9
    if study_profile == "preservation144":
        return preservation_study, "preservation144", 18, 9
    if study_profile == "matched144":
        return (
            matched_study,
            "matched144",
            matched_study.TRAINING_PAIR_COUNT,
            matched_study.DEVELOPMENT_PAIR_COUNT,
        )
    raise ValueError("unknown study profile")


def _plan_binding(plan: Any, study_profile: str) -> dict[str, Any]:
    spec, study_type, _train_count, _dev_count = _study_spec(study_profile)
    marker = plan.output_dir / spec.RUN_MARKER_NAME
    fields = {
        "study_type": study_type,
        "source_dataset_manifest_sha256": plan.source_dataset_manifest_sha256,
        "training_pair_ids": list(plan.training_pair_ids),
        "development_pair_ids": list(plan.development_pair_ids),
    }
    if study_profile == "controlled3d144":
        fields.update(
            source_registry_sha256=plan.source_registry_sha256,
            validation_proof_sha256=plan.validation_proof_sha256,
            train_export_completion_sha256=plan.train_export_completion_sha256,
        )
    elif study_profile == "preservation144":
        fields.update(
            parent_payload_sha256=plan.parent_payload_sha256,
            preservation_manifest_sha256=plan.preservation_manifest_sha256,
        )
    return {"plan_marker_path": marker, **fields}


def _reserve_attempt_marker(
    plan: matched_study.MatchedStudyPlan,
    marker_path: Path,
    *,
    source_commit: str,
    context: dict[str, str],
    study_profile: str = "matched144",
) -> dict[str, Any]:
    binding = _plan_binding(plan, study_profile)
    payload = plan.payload_path.read_bytes()
    plan_marker = binding["plan_marker_path"].read_bytes()
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    if payload_sha256 != plan.payload_sha256 or len(payload) != plan.payload_bytes:
        raise ValueError("matched-study payload changed before attempt reservation")
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("matched-study source commit must be a full Git SHA")
    if context != {
        "profile": remote.PROFILE,
        "workspace": remote.WORKSPACE,
        "environment": remote.ENVIRONMENT,
    }:
        raise ValueError("matched-study launch context must match the reviewed Modal target")
    marker = {
        "schema_version": 1,
        "run_id": plan.run_id,
        **{key: value for key, value in binding.items() if key != "plan_marker_path"},
        "status": "reserved",
        "source_commit": source_commit,
        "payload_sha256": payload_sha256,
        "plan_sha256": hashlib.sha256(plan_marker).hexdigest(),
        "gpu": "A100-80GB",
        "gpu_count": 1,
        "cores": 4,
        "memory_gib": 64,
        "startup_timeout_seconds": REMOTE_STARTUP_TIMEOUT_SECONDS,
        "timeout_seconds": REMOTE_TIMEOUT_SECONDS,
        "max_calls": 1,
        "retries": 0,
        "allowance_usd": 15,
        **context,
        "reserved_at_utc": datetime.now(UTC).isoformat(),
        "app_id": None,
        "call_id": None,
        "shutdown_check": (
            "Verify the detached matched-study app is stopped after observing this call."
        ),
    }
    _write_json_once(marker_path, marker)
    return marker


def _update_attempt_marker(marker_path: Path, **updates: Any) -> dict[str, Any]:
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if not isinstance(marker, dict):
        raise ValueError("matched-study attempt marker is invalid")
    for key in ("app_id", "call_id"):
        if key not in updates:
            continue
        new_value = updates.get(key)
        old_value = marker.get(key)
        if old_value is not None and new_value != old_value:
            raise ValueError(f"matched-study attempt {key} is immutable once recorded")
        if new_value is not None and (not isinstance(new_value, str) or not new_value):
            raise ValueError(f"matched-study attempt {key} must be a non-empty string")
    marker.update(updates)
    temp_path = marker_path.with_name(f".{marker_path.name}.{os.getpid()}.tmp")
    with temp_path.open("x", encoding="utf-8") as stream:
        json.dump(marker, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, marker_path)
    return marker


def _run_local_git(project_root: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["git", "-C", str(project_root), *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        raise ValueError("matched-study clearance could not verify the local Git state")
    return completed


def _validate_launch_clearance(
    plan: Any,
    *,
    project_root: Path,
    clearance_path: Path,
    context: dict[str, str],
    now: datetime | None = None,
    study_profile: str = "matched144",
) -> dict[str, Any]:
    if clearance_path.is_symlink() or not clearance_path.is_file():
        raise ValueError("matched-study launch clearance is missing or unsafe")
    try:
        clearance = json.loads(clearance_path.read_text(encoding="utf-8"))
        payload = plan.payload_path.read_bytes()
        binding = _plan_binding(plan, study_profile)
        plan_marker_path = binding["plan_marker_path"]
        plan_marker = plan_marker_path.read_bytes()
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("matched-study clearance or bound plan is missing or invalid") from exc
    if not isinstance(clearance, dict):
        raise ValueError("matched-study launch clearance must be a JSON object")
    spec, study_type, train_count, dev_count = _study_spec(study_profile)
    binding = _plan_binding(plan, study_profile)
    expected = {
        "schema_version": 1,
        "run_id": plan.run_id,
        **{key: value for key, value in binding.items() if key != "plan_marker_path"},
        "status": "approved",
        "review_status": "approved",
        "ci_status": "success",
        "payload_sha256": plan.payload_sha256,
        "plan_sha256": hashlib.sha256(plan_marker).hexdigest(),
        "training_pairs": train_count,
        "development_pairs": dev_count,
        "epochs": matched_study.TRAINING_EPOCHS,
        "training_steps": matched_study.TRAINING_STEPS,
        "allowance_usd": 15,
        "gpu": "A100-80GB",
        "gpu_count": 1,
        "cores": 4,
        "memory_gib": 64,
        "startup_timeout_seconds": REMOTE_STARTUP_TIMEOUT_SECONDS,
        "timeout_seconds": REMOTE_TIMEOUT_SECONDS,
        "max_calls": 1,
        "retries": 0,
        **context,
    }
    if study_profile == "matched144":
        pinned_manifest = matched_study.APPROVED_DATASET_MANIFEST_SHA256
    elif study_profile == "controlled3d144":
        pinned_manifest = controlled3d_study.APPROVED_CONTROLLED3D_MANIFEST_SHA256
    else:
        pinned_manifest = preservation_study.APPROVED_PRESERVATION_MANIFEST_SHA256
        if pinned_manifest is None:
            raise ValueError("preservation manifest digest pin is unset")
    pinned_registry = controlled3d_study.APPROVED_CONTROLLED3D_REGISTRY_SHA256
    if (
        hashlib.sha256(payload).hexdigest() != plan.payload_sha256
        or len(payload) != plan.payload_bytes
        or plan.payload_path.is_symlink()
        or plan_marker_path.is_symlink()
        or (
            study_profile != "preservation144"
            and plan.source_dataset_manifest_sha256 != pinned_manifest
        )
        or (study_profile == "controlled3d144" and plan.source_registry_sha256 != pinned_registry)
        or (
            study_profile == "preservation144"
            and (
                plan.parent_payload_sha256 != preservation_study.PARENT_PAYLOAD_SHA256
                or plan.preservation_manifest_sha256 != pinned_manifest
            )
        )
        or context
        != {
            "profile": remote.PROFILE,
            "workspace": remote.WORKSPACE,
            "environment": remote.ENVIRONMENT,
        }
        or set(clearance) != set(expected) | {"source_commit", "checked_at"}
        or any(clearance.get(key) != value for key, value in expected.items())
    ):
        raise ValueError(
            "matched-study launch clearance does not match the approved plan and account"
        )
    source_commit = clearance.get("source_commit")
    if not isinstance(source_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("matched-study clearance source commit must be a full Git SHA")
    checked_value = clearance.get("checked_at")
    if not isinstance(checked_value, str):
        raise ValueError("matched-study clearance timestamp is missing")
    try:
        checked_at = datetime.fromisoformat(checked_value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("matched-study clearance timestamp is invalid") from exc
    current_time = now or datetime.now(UTC)
    if checked_at.tzinfo is None or current_time.tzinfo is None:
        raise ValueError("matched-study clearance timestamps must include a timezone")
    age = current_time.astimezone(UTC) - checked_at.astimezone(UTC)
    if age < timedelta(0) or age > timedelta(minutes=30):
        raise ValueError("matched-study launch clearance is stale or from the future")
    if _run_local_git(project_root, ["status", "--porcelain", "--untracked-files=all"]).stdout:
        raise ValueError("matched-study launch requires a clean current Git worktree")
    head = _run_local_git(project_root, ["rev-parse", "HEAD"]).stdout.strip()
    if source_commit != head:
        raise ValueError("matched-study clearance source commit does not match current HEAD")
    _run_local_git(project_root, ["verify-commit", "HEAD"])
    return clearance


def _failure_receipt(
    run_id: str,
    stage: str,
    error: BaseException,
    *,
    app_id: str | None = None,
    call_id: str | None = None,
    status: str = "observation_unknown",
    study_profile: str = "matched144",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "study_type": _study_spec(study_profile)[1],
        "status": status,
        "failure": {
            "stage": stage,
            "type": type(error).__name__,
            "message": remote.sanitize_failure(error),
        },
        "worker_failure_confirmed": False,
        "app_id": app_id,
        "call_id": call_id,
        "same_call_recovery": (
            "Use --recover to observe only this recorded Modal call; never spawn a replacement. "
            "Verify the detached app is stopped after resolution."
        ),
        "shutdown_check": (
            "Verify the detached matched-study app is stopped after resolving this call."
        ),
    }


def _validate_attempt_binding(
    plan: matched_study.MatchedStudyPlan,
    attempt_path: Path,
    context: dict[str, str],
    study_profile: str = "matched144",
) -> dict[str, Any]:
    if attempt_path.is_symlink() or not attempt_path.is_file():
        raise ValueError("matched-study attempt marker is missing or unsafe")
    try:
        marker = json.loads(attempt_path.read_text(encoding="utf-8"))
        payload = plan.payload_path.read_bytes()
        binding = _plan_binding(plan, study_profile)
        plan_marker = binding["plan_marker_path"].read_bytes()
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("matched-study attempt or plan is missing or invalid") from exc
    expected = {
        "schema_version": 1,
        "run_id": plan.run_id,
        **{key: value for key, value in binding.items() if key != "plan_marker_path"},
        "payload_sha256": plan.payload_sha256,
        "plan_sha256": hashlib.sha256(plan_marker).hexdigest(),
        "gpu": "A100-80GB",
        "gpu_count": 1,
        "cores": 4,
        "memory_gib": 64,
        "startup_timeout_seconds": REMOTE_STARTUP_TIMEOUT_SECONDS,
        "timeout_seconds": REMOTE_TIMEOUT_SECONDS,
        "max_calls": 1,
        "retries": 0,
        "allowance_usd": 15,
        **context,
    }
    if (
        not isinstance(marker, dict)
        or marker.get("status") not in {"observing", "modal_error", "observation_unknown"}
        or set(marker)
        - set(expected)
        - {
            "status",
            "source_commit",
            "reserved_at_utc",
            "app_id",
            "call_id",
            "shutdown_check",
            "spawned_at_utc",
            "failure_stage",
            "last_error",
            "last_observation_error",
            "last_observed_at_utc",
            "worker_status",
            "completed_at_utc",
            "shutdown_status",
            "shutdown_error",
            "shutdown_failed_at_utc",
            "shutdown_completed_at_utc",
            "unverified_result",
        }
        or any(marker.get(key) != value for key, value in expected.items())
        or hashlib.sha256(payload).hexdigest() != plan.payload_sha256
        or len(payload) != plan.payload_bytes
        or plan.payload_path.is_symlink()
        or binding["plan_marker_path"].is_symlink()
        or not isinstance(marker.get("source_commit"), str)
        or not re.fullmatch(r"[0-9a-f]{40}", marker["source_commit"])
        or not isinstance(marker.get("call_id"), str)
        or not marker["call_id"]
        or not isinstance(marker.get("app_id"), str)
        or not marker["app_id"]
    ):
        raise ValueError("matched-study attempt marker does not bind this plan and Modal target")
    evidence = marker.get("unverified_result")
    if evidence is not None:
        evidence_path = plan.output_dir / UNVERIFIED_RESULT_NAME
        if (
            not isinstance(evidence, dict)
            or set(evidence) != {"file", "bytes", "sha256"}
            or evidence.get("file") != UNVERIFIED_RESULT_NAME
            or isinstance(evidence.get("bytes"), bool)
            or not isinstance(evidence.get("bytes"), int)
            or not 0 < evidence["bytes"] <= remote.MAX_RESULT_BYTES
            or not isinstance(evidence.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", evidence["sha256"])
            or evidence_path.is_symlink()
            or not evidence_path.is_file()
            or evidence_path.stat().st_size != evidence["bytes"]
            or hashlib.sha256(evidence_path.read_bytes()).hexdigest() != evidence["sha256"]
        ):
            raise ValueError("matched-study unverified result evidence is missing or changed")
    return marker


def _write_failure_once(path: Path, receipt: dict[str, Any]) -> None:
    if not path.exists() and not path.is_symlink():
        _write_json_once(path, receipt)


def _preserve_unverified_result(plan: Any, data: object) -> dict[str, Any] | None:
    if not isinstance(data, bytes) or not data or len(data) > remote.MAX_RESULT_BYTES:
        return None
    path = plan.output_dir / UNVERIFIED_RESULT_NAME
    with path.open("xb") as stream:
        stream.write(data)
    return {
        "file": UNVERIFIED_RESULT_NAME,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _record_result_validation_failure(
    plan: matched_study.MatchedStudyPlan,
    attempt_path: Path,
    failure_path: Path,
    result_bytes: object,
    error: BaseException,
    *,
    app_id: str,
    call_id: str,
    study_profile: str = "matched144",
) -> None:
    failure = _failure_receipt(
        plan.run_id,
        "modal_result_validation",
        error,
        app_id=app_id,
        call_id=call_id,
        status="modal_error",
        study_profile=study_profile,
    )
    updates: dict[str, Any] = {
        "status": "modal_error",
        "failure_stage": "modal_result_validation",
        "last_error": remote.sanitize_failure(error),
    }
    try:
        evidence = _preserve_unverified_result(plan, result_bytes)
    except Exception as preserve_error:
        failure["unverified_result_preservation_error"] = remote.sanitize_failure(preserve_error)
    else:
        if evidence is not None:
            failure["unverified_result"] = evidence
            updates["unverified_result"] = evidence
    _update_attempt_marker(attempt_path, **updates)
    _write_failure_once(failure_path, failure)
    print(json.dumps(failure, sort_keys=True))


def _stop_detached_app(app_id: str, *, context: dict[str, str]) -> None:
    environment = dict(os.environ)
    environment["MODAL_PROFILE"] = context["profile"]
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "modal",
                "app",
                "stop",
                "--yes",
                "--env",
                context["environment"],
                app_id,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
            env=environment,
        )
    except Exception as exc:
        raise RuntimeError("could not stop the detached matched-study Modal app") from exc
    if completed.returncode == 0:
        return
    try:
        observation = subprocess.run(
            [
                sys.executable,
                "-m",
                "modal",
                "app",
                "list",
                "--json",
                "--env",
                context["environment"],
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
            env=environment,
        )
    except Exception as exc:
        raise RuntimeError("could not stop the detached matched-study Modal app") from exc
    if observation.returncode != 0 or not _app_list_confirms_stopped(observation.stdout, app_id):
        raise RuntimeError("Modal did not confirm that the matched-study app stopped")


def _app_list_confirms_stopped(output: str, app_id: str) -> bool:
    try:
        apps = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(apps, list):
        return False
    expected_fields = {"app_id", "description", "state", "tasks", "created_at", "stopped_at"}
    matching_apps: list[dict[str, Any]] = []
    for app in apps:
        if not isinstance(app, dict) or set(app) != expected_fields:
            return False
        if (
            not isinstance(app["app_id"], str)
            or not app["app_id"]
            or not isinstance(app["state"], str)
            or not isinstance(app["tasks"], str)
            or not app["tasks"].isdigit()
        ):
            return False
        if any(
            app[field] is not None and not isinstance(app[field], str)
            for field in ("description", "created_at", "stopped_at")
        ):
            return False
        if app["app_id"] == app_id:
            matching_apps.append(app)
    return (
        len(matching_apps) == 1
        and matching_apps[0]["state"] == "stopped"
        and matching_apps[0]["tasks"] == "0"
    )


def _record_shutdown_failure(
    plan: matched_study.MatchedStudyPlan,
    attempt_path: Path,
    failure_path: Path,
    error: BaseException,
    *,
    app_id: str,
    call_id: str,
    study_profile: str = "matched144",
) -> None:
    failure = _failure_receipt(
        plan.run_id,
        "modal_app_stop",
        error,
        app_id=app_id,
        call_id=call_id,
        status="shutdown_error",
        study_profile=study_profile,
    )
    _update_attempt_marker(
        attempt_path,
        status="shutdown_error",
        shutdown_status="error",
        shutdown_error=remote.sanitize_failure(error),
        shutdown_failed_at_utc=datetime.now(UTC).isoformat(),
    )
    _write_failure_once(failure_path, failure)


def _recover(
    plan: Any,
    *,
    context: dict[str, str],
    study_profile: str = "matched144",
) -> int:
    """Poll only the call ID recorded by the original launch; never create a new call."""
    result_path = plan.output_dir / RESULT_NAME
    failure_path = plan.output_dir / MODAL_FAILURE_NAME
    attempt_path = plan.output_dir / ATTEMPT_MARKER_NAME
    if result_path.exists() or result_path.is_symlink():
        raise FileExistsError("matched-study result already exists; refusing to overwrite it")
    marker = _validate_attempt_binding(plan, attempt_path, context, study_profile)
    remote.validate_launch_environment(context)
    remote.verify_modal_identity(**context)
    os.environ["MODAL_PROFILE"] = context["profile"]
    if importlib.metadata.version("modal") != "1.6.1":
        raise RuntimeError("Modal SDK 1.6.1 is required")
    import modal

    call_id = marker["call_id"]
    app_id = marker["app_id"]
    try:
        result_bytes = modal.FunctionCall.from_id(call_id).get(timeout=0)
    except Exception as exc:
        failure = _failure_receipt(
            plan.run_id,
            "modal_recovery_observation",
            exc,
            app_id=app_id,
            call_id=call_id,
            study_profile=study_profile,
        )
        _update_attempt_marker(
            attempt_path,
            status="observation_unknown",
            failure_stage="modal_recovery_observation",
            last_observation_error=remote.sanitize_failure(exc),
            last_observed_at_utc=datetime.now(UTC).isoformat(),
        )
        _write_failure_once(failure_path, failure)
        print(json.dumps(failure, sort_keys=True))
        return 1
    result_status = 1
    try:
        receipt, _names = _validate_result_bundle(result_bytes, plan, study_profile)
        with result_path.open("xb") as stream:
            stream.write(result_bytes)
    except Exception as exc:
        _record_result_validation_failure(
            plan,
            attempt_path,
            failure_path,
            result_bytes,
            exc,
            app_id=app_id,
            call_id=call_id,
            study_profile=study_profile,
        )
    else:
        _update_attempt_marker(
            attempt_path,
            status="worker_returned",
            worker_status=receipt["status"],
            completed_at_utc=datetime.now(UTC).isoformat(),
        )
        result_status = 0 if receipt["status"] == "passed" else 1
        print(
            json.dumps(
                {"run_id": plan.run_id, "status": receipt["status"], "result": str(result_path)},
                sort_keys=True,
            )
        )
    try:
        _stop_detached_app(app_id, context=context)
    except Exception as exc:
        _record_shutdown_failure(
            plan,
            attempt_path,
            failure_path,
            exc,
            app_id=app_id,
            call_id=call_id,
            study_profile=study_profile,
        )
        print(
            json.dumps(
                _failure_receipt(
                    plan.run_id,
                    "modal_app_stop",
                    exc,
                    app_id=app_id,
                    call_id=call_id,
                    status="shutdown_error",
                    study_profile=study_profile,
                ),
                sort_keys=True,
            )
        )
        return 1
    _update_attempt_marker(
        attempt_path,
        shutdown_status="stopped",
        shutdown_completed_at_utc=datetime.now(UTC).isoformat(),
    )
    return result_status


def _launch(
    plan: Any,
    *,
    context: dict[str, str],
    project_root: Path | None = None,
    clearance_path: Path | None = None,
    study_profile: str = "matched144",
) -> int:
    result_path = plan.output_dir / RESULT_NAME
    failure_path = plan.output_dir / MODAL_FAILURE_NAME
    attempt_path = plan.output_dir / ATTEMPT_MARKER_NAME
    if any(
        path.exists() or path.is_symlink() for path in (result_path, failure_path, attempt_path)
    ):
        raise FileExistsError(
            "matched-study output already has a result, failure, or attempt marker"
        )
    train_count, dev_count = _study_spec(study_profile)[2:]
    payload: Any
    if study_profile == "controlled3d144":
        payload = controlled3d_study.validate_controlled3d_study_payload_bytes(
            plan.payload_path.read_bytes(), plan.run_id
        )
        manifest_pin = controlled3d_study.APPROVED_CONTROLLED3D_MANIFEST_SHA256
        registry_pin = controlled3d_study.APPROVED_CONTROLLED3D_REGISTRY_SHA256
        valid = (
            payload.source_registry_sha256 == registry_pin
            and payload.validation_proof_sha256 == plan.validation_proof_sha256
            and payload.train_export_completion_sha256 == plan.train_export_completion_sha256
        )
    elif study_profile == "preservation144":
        payload = preservation_study.read_preservation_study_payload(plan)
        manifest_pin = plan.source_dataset_manifest_sha256
        derived_manifest_pin = preservation_study.APPROVED_PRESERVATION_MANIFEST_SHA256
        valid = (
            derived_manifest_pin is not None
            and plan.parent_payload_sha256 == preservation_study.PARENT_PAYLOAD_SHA256
            and plan.preservation_manifest_sha256 == derived_manifest_pin
            and payload.preservation_manifest_sha256 == derived_manifest_pin
        )
    else:
        payload = matched_study.read_matched_study_payload(plan)
        manifest_pin = matched_study.APPROVED_DATASET_MANIFEST_SHA256
        valid = True
    if (
        len(payload.train_rows) != train_count
        or len(payload.dev_rows) != dev_count
        or tuple(row["pair_id"] for row in payload.train_rows) != plan.training_pair_ids
        or tuple(row["pair_id"] for row in payload.dev_rows) != plan.development_pair_ids
        or payload.source_dataset_manifest_sha256 != manifest_pin
        or not valid
    ):
        raise ValueError("study launch requires the exact reviewed train/dev plan")
    root = project_root or Path(__file__).resolve().parents[1]
    default_clearance = (
        ".context/controlled3d-study-launch-clearance.json"
        if study_profile == "controlled3d144"
        else ".context/preservation-study-launch-clearance.json"
        if study_profile == "preservation144"
        else ".context/matched-study-launch-clearance.json"
    )
    clearance = _validate_launch_clearance(
        plan,
        project_root=root,
        clearance_path=clearance_path or root / default_clearance,
        context=context,
        study_profile=study_profile,
    )
    remote.validate_launch_environment(context)
    _reserve_attempt_marker(
        plan,
        attempt_path,
        source_commit=clearance["source_commit"],
        context=context,
        study_profile=study_profile,
    )
    phase = "modal_identity"
    app: Any | None = None
    app_id: str | None = None
    call_id: str | None = None
    try:
        remote.verify_modal_identity(**context)
        os.environ["MODAL_PROFILE"] = context["profile"]
        phase = "modal_import"
        import modal

        if importlib.metadata.version("modal") != "1.6.1":
            raise RuntimeError("Modal SDK 1.6.1 is required")
        phase = "modal_build"
        if not (root / "src/product_scene_editor/remote.py").is_file():
            raise FileNotFoundError("matched-study remote worker source is missing")
        image = _build_image(modal, root)
        app = modal.App(APP_NAME, image=image)
        run_once = app.function(
            gpu="A100-80GB",
            cpu=(4.0, 4.0),
            memory=(65536, 65536),
            ephemeral_disk=524288,
            min_containers=0,
            max_containers=1,
            buffer_containers=0,
            scaledown_window=2,
            retries=0,
            single_use_containers=True,
            serialized=True,
            include_source=False,
            startup_timeout=REMOTE_STARTUP_TIMEOUT_SECONDS,
            timeout=REMOTE_TIMEOUT_SECONDS,
        )(_remote_worker)
        phase = "modal_app_startup"
        with modal.enable_output(), app.run(environment_name=context["environment"], detach=True):
            app_id_value = getattr(app, "app_id", None) or getattr(app, "object_id", None)
            if not isinstance(app_id_value, str) or not app_id_value:
                raise RuntimeError("Modal app did not expose an app ID before spawn")
            app_id = app_id_value
            _update_attempt_marker(attempt_path, status="app_started", app_id=app_id)
            phase = "modal_spawn"
            call = run_once.spawn(plan.payload_path.read_bytes(), plan.run_id, study_profile)
            call_id_value = getattr(call, "object_id", None) or getattr(call, "call_id", None)
            if not isinstance(call_id_value, str) or not call_id_value:
                raise RuntimeError("Modal call did not expose a call ID after spawn")
            call_id = call_id_value
            _update_attempt_marker(
                attempt_path,
                status="observing",
                call_id=call_id,
                spawned_at_utc=datetime.now(UTC).isoformat(),
            )
            phase = "modal_observation"
            result_bytes = call.get()
    except Exception as exc:
        if app_id is None and app is not None:
            app_id_value = getattr(app, "app_id", None)
            if isinstance(app_id_value, str) and app_id_value:
                app_id = app_id_value
        unknown_observation = phase == "modal_observation" and call_id is not None
        failure = _failure_receipt(
            plan.run_id,
            phase,
            exc,
            app_id=app_id,
            call_id=call_id,
            status="observation_unknown" if unknown_observation else "modal_error",
            study_profile=study_profile,
        )
        _update_attempt_marker(
            attempt_path,
            status="observation_unknown" if unknown_observation else "modal_error",
            app_id=app_id,
            call_id=call_id,
            failure_stage=phase,
            last_error=remote.sanitize_failure(exc),
        )
        _write_failure_once(failure_path, failure)
        print(json.dumps(failure, sort_keys=True))
        return 1
    result_status = 1
    try:
        receipt, _names = _validate_result_bundle(result_bytes, plan, study_profile)
        with result_path.open("xb") as stream:
            stream.write(result_bytes)
    except Exception as exc:
        _record_result_validation_failure(
            plan,
            attempt_path,
            failure_path,
            result_bytes,
            exc,
            app_id=app_id,
            call_id=call_id,
            study_profile=study_profile,
        )
    else:
        _update_attempt_marker(
            attempt_path,
            status="worker_returned",
            worker_status=receipt["status"],
            completed_at_utc=datetime.now(UTC).isoformat(),
        )
        result_status = 0 if receipt["status"] == "passed" else 1
        print(
            json.dumps(
                {"run_id": plan.run_id, "status": receipt["status"], "result": str(result_path)},
                sort_keys=True,
            )
        )
    try:
        _stop_detached_app(app_id, context=context)
    except Exception as exc:
        _record_shutdown_failure(
            plan,
            attempt_path,
            failure_path,
            exc,
            app_id=app_id,
            call_id=call_id,
            study_profile=study_profile,
        )
        print(
            json.dumps(
                _failure_receipt(
                    plan.run_id,
                    "modal_app_stop",
                    exc,
                    app_id=app_id,
                    call_id=call_id,
                    status="shutdown_error",
                    study_profile=study_profile,
                ),
                sort_keys=True,
            )
        )
        return 1
    _update_attempt_marker(
        attempt_path,
        shutdown_status="stopped",
        shutdown_completed_at_utc=datetime.now(UTC).isoformat(),
    )
    return result_status


def _load_plan(output_dir: Path, study_profile: str = "matched144") -> Any:
    spec = _study_spec(study_profile)[0]
    if study_profile == "matched144":
        return spec.load_matched_study_plan(output_dir)
    if study_profile == "controlled3d144":
        return spec.load_controlled3d_study_plan(output_dir)
    return spec.load_preservation_study_plan(output_dir)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    study_profile = args.study_profile
    study_dir = {
        "controlled3d144": "controlled3d-studies",
        "preservation144": "preservation-studies",
    }.get(study_profile, "matched-studies")
    plan_dir = args.plan_dir or Path(f"artifacts/{study_dir}/{args.run_id}")
    if args.launch or args.recover:
        plan = _load_plan(plan_dir, study_profile)
        if args.run_id != plan.run_id:
            raise ValueError("--run-id does not match the write-once study plan")
        context = {
            "profile": args.profile or "",
            "workspace": args.workspace or "",
            "environment": args.environment or "",
        }
        if not all(context.values()):
            mode = "--recover" if args.recover else "--launch"
            raise ValueError(f"{mode} requires explicit --profile, --workspace, and --environment")
        if args.recover:
            return _recover(plan, context=context, study_profile=study_profile)
        return _launch(
            plan,
            context=context,
            clearance_path=args.clearance,
            study_profile=study_profile,
        )

    if study_profile == "preservation144":
        if args.parent_payload is None:
            raise ValueError("preservation planning requires --parent-payload")
        plan = preservation_study.plan_preservation_study(
            args.parent_payload, plan_dir, run_id=args.run_id
        )
        payload = preservation_study.read_preservation_study_payload(plan)
    elif not args.dataset_dir.is_dir():
        raise FileNotFoundError("the reviewed study dataset must be present before planning")
    elif study_profile == "controlled3d144":
        if args.manifest is None or args.registry is None or args.train_export_dir is None:
            raise ValueError(
                "Controlled3D planning requires --manifest, --registry, and --train-export-dir"
            )
        plan = controlled3d_study.plan_controlled3d_study(
            args.dataset_dir,
            args.manifest,
            args.registry,
            args.train_export_dir,
            plan_dir,
            run_id=args.run_id,
        )
        payload = controlled3d_study.validate_controlled3d_study_payload_bytes(
            plan.payload_path.read_bytes(), plan.run_id
        )
    else:
        plan = matched_study.plan_matched_study(args.dataset_dir, plan_dir, run_id=args.run_id)
        payload = matched_study.read_matched_study_payload(plan)
    print(
        json.dumps(
            {
                "status": "planned",
                "study_type": _study_spec(study_profile)[1],
                "run_id": plan.run_id,
                "plan_dir": str(plan.output_dir),
                "payload_bytes": plan.payload_bytes,
                "payload_sha256": plan.payload_sha256,
                "source_dataset_manifest_sha256": plan.source_dataset_manifest_sha256,
                "source_registry_sha256": getattr(plan, "source_registry_sha256", None),
                "validation_proof_sha256": getattr(plan, "validation_proof_sha256", None),
                "train_export_completion_sha256": getattr(
                    plan, "train_export_completion_sha256", None
                ),
                "training_pairs": len(payload.train_rows),
                "development_pairs": len(payload.dev_rows),
                "training_steps": matched_study.TRAINING_STEPS,
                "modal_called": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _validate_result_bundle(
    data: bytes, plan: Any, study_profile: str = "matched144"
) -> tuple[dict[str, Any], set[str]]:
    """Validate a worker archive against the exact plan-specific artifact contract."""
    if not isinstance(data, bytes) or not data or len(data) > remote.MAX_RESULT_BYTES:
        raise ValueError("Modal returned an empty or oversized matched-study result")
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            raw_names = archive.namelist()
            names = set(raw_names)
            if len(raw_names) != len(names):
                raise ValueError("Modal returned duplicate matched-study artifact paths")
            if any(
                PurePosixPath(name).is_absolute()
                or ".." in PurePosixPath(name).parts
                or "\\" in name
                for name in names
            ):
                raise ValueError("Modal returned an unsafe matched-study artifact path")
            receipt = json.loads(archive.read("receipt.json"))
    except (OSError, KeyError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise ValueError("Modal result is not a valid matched-study receipt archive") from exc
    if (
        not isinstance(receipt, dict)
        or receipt.get("run_id") != plan.run_id
        or receipt.get("study_type") != _study_spec(study_profile)[1]
        or receipt.get("status") not in {"passed", "failed"}
    ):
        raise ValueError("Modal result receipt does not match this study run")
    if study_profile in {"controlled3d144", "preservation144"} and receipt["status"] == "passed":
        training = receipt.get("training")
        source_manifest = (
            training.get("source_dataset_manifest_sha256") if isinstance(training, dict) else None
        )
        if source_manifest != plan.source_dataset_manifest_sha256:
            raise ValueError("Modal study receipt source manifest does not bind this plan")
        if study_profile == "controlled3d144":
            expected_provenance = {
                "source_registry_sha256": plan.source_registry_sha256,
                "validation_proof_sha256": plan.validation_proof_sha256,
                "train_export_completion_sha256": plan.train_export_completion_sha256,
            }
            if (
                receipt.get("controlled3d_provenance") != expected_provenance
                or receipt.get("metrics_reference") != remote.CONTROLLED3D_METRIC_REFERENCE
            ):
                raise ValueError("Modal Controlled3D receipt provenance does not bind this plan")
        else:
            expected_provenance = {
                "parent_payload_sha256": plan.parent_payload_sha256,
                "preservation_manifest_sha256": plan.preservation_manifest_sha256,
            }
            if receipt.get("preservation_provenance") != expected_provenance:
                raise ValueError("Modal preservation receipt provenance does not bind this plan")
    required = {
        "receipt.json",
        "adapter/step-144.safetensors",
        *(
            f"images/{pair_id}/{mode}.png"
            for pair_id in plan.development_pair_ids
            for mode in ("base", "trained", "reloaded")
        ),
    }
    if receipt["status"] == "passed" and names != required:
        raise ValueError(
            "Modal matched-study result artifact set does not match its 29-file contract"
        )
    if receipt["status"] == "failed" and ("receipt.json" not in names or not names <= required):
        raise ValueError("Modal matched-study failure artifact set contains unexpected files")
    return receipt, names


if __name__ == "__main__":
    raise SystemExit(main())
