"""CPU planning and guarded one-shot Modal inference for real-photo development cases."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from experiments import modal_matched_study, modal_pilot
from product_scene_editor import real_photo_runner as runner
from product_scene_editor import remote

APP_NAME = "product-scene-editor-real-photo-inference"
ATTEMPT_NAME = "real-photo-modal-attempt.json"
FAILURE_NAME = "real-photo-modal-failure-receipt.json"
SHUTDOWN_FAILURE_NAME = "real-photo-modal-shutdown-failure-receipt.json"
RECOVERY_FAILURE_NAME = "real-photo-modal-recovery-failure-receipt.json"
UNVERIFIED_NAME = "real-photo-unverified-result.bin"
REMOTE_STARTUP_TIMEOUT_SECONDS = 300
REMOTE_TIMEOUT_SECONDS = 900
GPU = "A100-80GB"
CORES = 4
MEMORY_GIB = 64
ALLOWANCE_USD = 15
CONTEXT = {
    "profile": remote.PROFILE,
    "workspace": remote.WORKSPACE,
    "environment": remote.ENVIRONMENT,
}
_build_image = modal_pilot._build_image
_app_list_confirms_stopped = modal_matched_study._app_list_confirms_stopped
_stop_detached_app = modal_matched_study._stop_detached_app


@dataclass(frozen=True)
class RealPhotoPlan:
    output_dir: Path
    payload_path: Path
    run_id: str
    payload_bytes: int
    payload_sha256: str
    plan_sha256: str
    marker: dict[str, Any]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="real-photo-off-r1-2026-10-10-r1")
    parser.add_argument(
        "--plan-dir",
        type=Path,
        default=Path("artifacts/real-photo-runs/real-photo-off-r1-2026-10-10-r1"),
    )
    parser.add_argument(
        "--adapter-archive",
        type=Path,
        default=Path(
            "artifacts/matched-studies/klein-edit-matched-2026-10-09-r1/matched-study-result.zip"
        ),
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--launch", action="store_true", help="run the explicitly cleared plan")
    action.add_argument("--recover", action="store_true", help="observe only the recorded call")
    parser.add_argument(
        "--clearance",
        type=Path,
        default=Path(".context/real-photo-launch-clearance.json"),
    )
    parser.add_argument("--profile")
    parser.add_argument("--workspace")
    parser.add_argument("--environment")
    return parser


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _write_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def _update_attempt_marker(path: Path, **updates: Any) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("real-photo attempt marker is missing or unsafe")
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("real-photo attempt marker is invalid") from exc
    if not isinstance(marker, dict):
        raise ValueError("real-photo attempt marker must be a JSON object")
    for key in ("app_id", "call_id"):
        if key not in updates:
            continue
        new_value = updates[key]
        old_value = marker.get(key)
        if old_value is not None and new_value != old_value:
            raise ValueError(f"real-photo attempt {key} is immutable once recorded")
        if new_value is not None and (not isinstance(new_value, str) or not new_value):
            raise ValueError(f"real-photo attempt {key} must be a non-empty string")
    if "call_count" in updates and updates["call_count"] != 1:
        raise ValueError("real-photo attempt is restricted to one Modal call")
    if "spawn_count" in updates and updates["spawn_count"] != 1:
        raise ValueError("real-photo attempt is restricted to one Modal spawn")
    marker.update(updates)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(marker, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return marker


def _load_plan(output_dir: Path) -> RealPhotoPlan:
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError("real-photo plan directory is missing or unsafe")
    marker_path = output_dir / runner.RUN_MARKER_NAME
    payload_path = output_dir / runner.PAYLOAD_NAME
    if marker_path.is_symlink() or payload_path.is_symlink():
        raise ValueError("real-photo plan files may not be symlinks")
    try:
        marker_bytes = marker_path.read_bytes()
        marker = json.loads(marker_bytes.decode("utf-8"))
        payload = payload_path.read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("real-photo plan marker or payload is missing or invalid") from exc
    if not isinstance(marker, dict):
        raise ValueError("real-photo plan marker must be a JSON object")
    digest = hashlib.sha256(payload).hexdigest()
    if (
        marker.get("schema_version") != 1
        or not isinstance(marker.get("run_id"), str)
        or marker.get("payload_file") != runner.PAYLOAD_NAME
        or marker.get("payload_bytes") != len(payload)
        or marker.get("payload_sha256") != digest
        or len(payload) > runner.MAX_INPUT_BYTES
        or marker.get("case_ids") != list(runner.CASE_IDS)
        or marker.get("generation_settings") != runner.GENERATION_SETTINGS
        or marker.get("schedule") != list(runner.generation_schedule())
        or marker.get("adapter_archive_sha256") != runner.ADAPTER_ARCHIVE_SHA256
        or marker.get("adapter_member_sha256") != runner.ADAPTER_MEMBER_SHA256
        or marker.get("modal_sdk") != "1.6.1"
        or marker.get("diffsynth_commit") != remote.DIFFSYNTH_COMMIT
        or marker.get("runtime_pins") != remote.RUNTIME_PINS
        or marker.get("model_components") != remote.MODEL_COMPONENTS
        or set(marker)
        != {
            "schema_version",
            "run_id",
            "payload_file",
            "payload_bytes",
            "payload_sha256",
            "preparation_sha256",
            "geometry_review_sha256",
            "geometry_review_status",
            "quality_review_status",
            "active_intake_sha256",
            "label_confirmation_sha256",
            "private_source_review_sha256",
            "case_ids",
            "adapter_archive_sha256",
            "adapter_member_sha256",
            "generation_settings",
            "schedule",
            "modal_sdk",
            "diffsynth_commit",
            "runtime_pins",
            "model_components",
        }
    ):
        raise ValueError("real-photo plan marker does not bind its payload and fixed contract")
    manifest, _files = runner.validate_payload_bytes(payload, marker["run_id"])
    for marker_key, manifest_key in (
        ("preparation_sha256", "preparation_sha256"),
        ("geometry_review_sha256", "geometry_review_sha256"),
        ("geometry_review_status", "geometry_review_status"),
        ("active_intake_sha256", "active_intake_sha256"),
        ("label_confirmation_sha256", "label_confirmation_sha256"),
        ("private_source_review_sha256", "private_source_review_sha256"),
        ("quality_review_status", "quality_review_status"),
    ):
        if marker.get(marker_key) != manifest.get(manifest_key):
            raise ValueError("real-photo plan binding does not match the payload manifest")
    return RealPhotoPlan(
        output_dir=output_dir,
        payload_path=payload_path,
        run_id=marker["run_id"],
        payload_bytes=len(payload),
        payload_sha256=digest,
        plan_sha256=hashlib.sha256(marker_bytes).hexdigest(),
        marker=marker,
    )


def _run_git(project_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(project_root), *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if completed.returncode:
        raise ValueError("real-photo clearance could not verify local Git state")
    return completed.stdout.strip()


def _validate_clearance(
    plan: RealPhotoPlan,
    *,
    project_root: Path,
    clearance_path: Path,
    context: dict[str, str],
    now: datetime | None = None,
) -> dict[str, Any]:
    if clearance_path.is_symlink() or not clearance_path.is_file():
        raise ValueError("real-photo launch clearance is missing or unsafe")
    try:
        clearance = json.loads(clearance_path.read_text(encoding="utf-8"))
        marker_bytes = (plan.output_dir / runner.RUN_MARKER_NAME).read_bytes()
        payload = plan.payload_path.read_bytes()
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("real-photo clearance or bound plan is missing or invalid") from exc
    if not isinstance(clearance, dict):
        raise ValueError("real-photo launch clearance must be a JSON object")
    expected = {
        "schema_version": 1,
        "run_id": plan.run_id,
        "status": "approved",
        "review_status": "approved",
        "ci_status": "success",
        "payload_sha256": plan.payload_sha256,
        "plan_sha256": hashlib.sha256(marker_bytes).hexdigest(),
        "preparation_sha256": plan.marker["preparation_sha256"],
        "geometry_review_sha256": plan.marker["geometry_review_sha256"],
        "geometry_review_status": plan.marker["geometry_review_status"],
        "active_intake_sha256": plan.marker["active_intake_sha256"],
        "label_confirmation_sha256": plan.marker["label_confirmation_sha256"],
        "private_source_review_sha256": plan.marker["private_source_review_sha256"],
        "quality_review_status": plan.marker["quality_review_status"],
        "case_ids": list(runner.CASE_IDS),
        "adapter_archive_sha256": runner.ADAPTER_ARCHIVE_SHA256,
        "adapter_member_sha256": runner.ADAPTER_MEMBER_SHA256,
        "generation_settings": runner.GENERATION_SETTINGS,
        "schedule": list(runner.generation_schedule()),
        "gpu": GPU,
        "cores": CORES,
        "memory_gib": MEMORY_GIB,
        "startup_timeout_seconds": REMOTE_STARTUP_TIMEOUT_SECONDS,
        "timeout_seconds": REMOTE_TIMEOUT_SECONDS,
        "max_calls": 1,
        "retries": 0,
        "allowance_usd": ALLOWANCE_USD,
        "modal_sdk": "1.6.1",
        "diffsynth_commit": remote.DIFFSYNTH_COMMIT,
        "runtime_pins": remote.RUNTIME_PINS,
        "model_components": remote.MODEL_COMPONENTS,
        **context,
    }
    if (
        set(clearance) != set(expected) | {"source_commit", "checked_at"}
        or any(clearance.get(key) != value for key, value in expected.items())
        or hashlib.sha256(payload).hexdigest() != plan.payload_sha256
        or len(payload) != plan.payload_bytes
        or plan.payload_path.is_symlink()
        or (plan.output_dir / runner.RUN_MARKER_NAME).is_symlink()
        or context != CONTEXT
    ):
        raise ValueError("real-photo clearance does not match the immutable plan and account")
    source_commit = clearance.get("source_commit")
    if not isinstance(source_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("real-photo clearance source commit must be a full Git SHA")
    timestamp = clearance.get("checked_at")
    if not isinstance(timestamp, str):
        raise ValueError("real-photo clearance timestamp is missing")
    try:
        checked_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("real-photo clearance timestamp is invalid") from exc
    current = now or datetime.now(UTC)
    if checked_at.tzinfo is None or current.tzinfo is None:
        raise ValueError("real-photo clearance timestamps must include a timezone")
    age = current.astimezone(UTC) - checked_at.astimezone(UTC)
    if age < timedelta(0) or age > timedelta(minutes=30):
        raise ValueError("real-photo clearance is stale or from the future")
    if _run_git(project_root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("real-photo launch requires a clean current Git worktree")
    head = _run_git(project_root, "rev-parse", "HEAD")
    if head != source_commit:
        raise ValueError("real-photo clearance commit does not match current HEAD")
    _run_git(project_root, "verify-commit", "HEAD")
    return clearance


def _failure_receipt(
    run_id: str,
    stage: str,
    error: BaseException,
    *,
    app_id: str | None,
    call_id: str | None,
    status: str,
    worker_failure_confirmed: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "status": status,
        "failure": {
            "stage": stage,
            "type": type(error).__name__,
            "message": remote.sanitize_failure(error),
        },
        "worker_failure_confirmed": worker_failure_confirmed,
        "app_id": app_id,
        "call_id": call_id,
        "same_call_recovery": (
            "Use --recover to observe only this recorded call ID. Never create another call."
        ),
        "shutdown_check": "Require Modal app state stopped with zero tasks before closure.",
    }


def _attempt_path(plan: RealPhotoPlan) -> Path:
    return plan.output_dir / ATTEMPT_NAME


def _reserve_attempt(plan: RealPhotoPlan, source_commit: str, context: dict[str, str]) -> Path:
    path = _attempt_path(plan)
    _write_once(
        path,
        {
            "schema_version": 1,
            "run_id": plan.run_id,
            "status": "reserved",
            "source_commit": source_commit,
            "payload_sha256": plan.payload_sha256,
            "plan_sha256": plan.plan_sha256,
            "case_ids": list(runner.CASE_IDS),
            "generation_settings": runner.GENERATION_SETTINGS,
            "gpu": GPU,
            "cores": CORES,
            "memory_gib": MEMORY_GIB,
            "startup_timeout_seconds": REMOTE_STARTUP_TIMEOUT_SECONDS,
            "timeout_seconds": REMOTE_TIMEOUT_SECONDS,
            "max_calls": 1,
            "call_count": 1,
            "spawn_count": 0,
            "retries": 0,
            "allowance_usd": ALLOWANCE_USD,
            **context,
            "reserved_at_utc": datetime.now(UTC).isoformat(),
            "app_id": None,
            "call_id": None,
            "shutdown_status": "pending",
        },
    )
    return path


def _confirm_app_stopped(app_id: str, context: dict[str, str]) -> None:
    environment = dict(os.environ)
    environment["MODAL_PROFILE"] = context["profile"]
    completed = subprocess.run(
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
    if completed.returncode != 0 or not _app_list_confirms_stopped(completed.stdout, app_id):
        raise RuntimeError("Modal did not confirm the app is stopped with zero tasks")


def _stop_and_confirm(app_id: str, context: dict[str, str]) -> None:
    _stop_detached_app(app_id, context=context)
    _confirm_app_stopped(app_id, context)


def _is_function_timeout(error: BaseException, modal_module: Any | None) -> bool:
    timeout_type = getattr(getattr(modal_module, "exception", None), "FunctionTimeoutError", None)
    return isinstance(timeout_type, type) and isinstance(error, timeout_type)


def _preserve_unverified_result(plan: RealPhotoPlan, result: object) -> dict[str, Any] | None:
    if not isinstance(result, bytes) or not result or len(result) > runner.MAX_RESULT_BYTES:
        return None
    path = plan.output_dir / UNVERIFIED_NAME
    with path.open("xb") as stream:
        stream.write(result)
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "file": UNVERIFIED_NAME,
        "bytes": len(result),
        "sha256": hashlib.sha256(result).hexdigest(),
    }


def _finish_result(
    plan: RealPhotoPlan,
    attempt_path: Path,
    result_bytes: object,
    *,
    app_id: str | None,
    call_id: str | None,
    receipt_path: Path,
) -> int:
    result_path = plan.output_dir / runner.RESULT_NAME
    try:
        if not isinstance(result_bytes, bytes):
            raise ValueError("Modal returned a non-byte result")
        receipt, _names = runner.validate_result_bundle(
            result_bytes,
            plan.run_id,
            expected_plan=plan.marker,
            expected_payload_sha256=plan.payload_sha256,
        )
        with result_path.open("xb") as stream:
            stream.write(result_bytes)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception as exc:
        failure = _failure_receipt(
            plan.run_id,
            "result_validation",
            exc,
            app_id=app_id,
            call_id=call_id,
            status="modal_error",
            worker_failure_confirmed=False,
        )
        try:
            evidence = _preserve_unverified_result(plan, result_bytes)
        except Exception as preserve_error:
            failure["unverified_result_preservation_error"] = remote.sanitize_failure(
                preserve_error
            )
        else:
            if evidence is not None:
                failure["unverified_result"] = evidence
        _update_attempt_marker(
            attempt_path,
            status="modal_error",
            failure_stage="result_validation",
            app_id=app_id,
            call_id=call_id,
            last_error=remote.sanitize_failure(exc),
        )
        _write_once(receipt_path, failure)
        return 1
    _update_attempt_marker(
        attempt_path,
        status="worker_returned",
        worker_failure_confirmed=receipt["status"] == "failed",
        worker_status=receipt["status"],
        app_id=app_id,
        call_id=call_id,
        completed_at_utc=datetime.now(UTC).isoformat(),
    )
    return 0 if receipt["status"] == "passed" else 1


def _record_shutdown_failure(
    plan: RealPhotoPlan,
    attempt_path: Path,
    error: BaseException,
    *,
    app_id: str,
    call_id: str | None,
) -> None:
    _update_attempt_marker(
        attempt_path,
        status="shutdown_error",
        shutdown_status="error",
        shutdown_error=remote.sanitize_failure(error),
        app_id=app_id,
        call_id=call_id,
        shutdown_failed_at_utc=datetime.now(UTC).isoformat(),
    )
    failure_path = plan.output_dir / SHUTDOWN_FAILURE_NAME
    if not failure_path.exists():
        _write_once(
            failure_path,
            _failure_receipt(
                plan.run_id,
                "modal_app_stop",
                error,
                app_id=app_id,
                call_id=call_id,
                status="shutdown_error",
                worker_failure_confirmed=False,
            ),
        )


def _launch(
    plan: RealPhotoPlan,
    *,
    context: dict[str, str],
    project_root: Path,
    clearance_path: Path,
) -> int:
    output_conflicts = (
        plan.output_dir / runner.RESULT_NAME,
        plan.output_dir / FAILURE_NAME,
        plan.output_dir / ATTEMPT_NAME,
        plan.output_dir / UNVERIFIED_NAME,
    )
    if any(path.exists() or path.is_symlink() for path in output_conflicts):
        raise FileExistsError("real-photo plan already has output or an attempt marker")
    clearance = _validate_clearance(
        plan,
        project_root=project_root,
        clearance_path=clearance_path,
        context=context,
    )
    remote.validate_launch_environment(context)
    remote.verify_modal_identity(**context)
    attempt_path = _reserve_attempt(plan, clearance["source_commit"], context)
    phase = "modal_import"
    app: Any | None = None
    modal_module: Any | None = None
    app_id: str | None = None
    call_id: str | None = None
    observation_unknown = False
    no_result = object()
    result_bytes: object = no_result
    inner_error: Exception | None = None
    try:
        os.environ["MODAL_PROFILE"] = context["profile"]
        import modal

        modal_module = modal

        if importlib.metadata.version("modal") != "1.6.1":
            raise RuntimeError("Modal SDK 1.6.1 is required")
        image = _build_image(modal, project_root)
        app = modal.App(APP_NAME, image=image)
        app_id_value = getattr(app, "app_id", None) or getattr(app, "object_id", None)
        if isinstance(app_id_value, str) and app_id_value:
            app_id = app_id_value
            _update_attempt_marker(attempt_path, app_id=app_id)
        run_once = app.function(
            gpu=GPU,
            cpu=(float(CORES), float(CORES)),
            memory=(MEMORY_GIB * 1024, MEMORY_GIB * 1024),
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
            if app_id is None:
                app_id_value = getattr(app, "app_id", None) or getattr(app, "object_id", None)
                if isinstance(app_id_value, str) and app_id_value:
                    app_id = app_id_value
                    _update_attempt_marker(attempt_path, app_id=app_id)
            if app_id is None:
                raise RuntimeError("Modal app ID was unavailable before the call")
            _update_attempt_marker(attempt_path, status="app_started", app_id=app_id)
            phase = "modal_spawn"
            _update_attempt_marker(attempt_path, status="spawn_reserved", spawn_count=1)
            try:
                call = run_once.spawn(plan.payload_path.read_bytes(), plan.run_id)
                call_id_value = getattr(call, "object_id", None) or getattr(call, "call_id", None)
                if not isinstance(call_id_value, str) or not call_id_value:
                    raise RuntimeError("Modal call ID was unavailable after spawn")
                call_id = call_id_value
                _update_attempt_marker(
                    attempt_path,
                    status="observing",
                    call_id=call_id,
                    spawn_count=1,
                    spawned_at_utc=datetime.now(UTC).isoformat(),
                )
                phase = "modal_observation"
                result_bytes = call.get()
            except Exception as exc:
                # Detached Modal app.run suppresses ConnectionError raised from
                # inside its context body; carry it past __exit__ explicitly.
                inner_error = exc
        if inner_error is not None:
            raise inner_error
        if result_bytes is no_result:
            raise RuntimeError("Modal call exited without returning or reporting an error")
    except Exception as exc:
        if app_id is None and app is not None:
            app_id_value = getattr(app, "app_id", None) or getattr(app, "object_id", None)
            if isinstance(app_id_value, str) and app_id_value:
                app_id = app_id_value
                _update_attempt_marker(attempt_path, app_id=app_id)
        function_timeout = phase == "modal_observation" and _is_function_timeout(exc, modal_module)
        attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
        spawn_may_have_started = phase == "modal_spawn" and attempt.get("spawn_count") == 1
        unknown = (phase == "modal_observation" and call_id is not None) or spawn_may_have_started
        observation_unknown = unknown and not function_timeout
        if function_timeout:
            status = "worker_failed"
        elif unknown:
            status = "observation_unknown"
        else:
            status = "modal_error"
        failure = _failure_receipt(
            plan.run_id,
            phase,
            exc,
            app_id=app_id,
            call_id=call_id,
            status=status,
            worker_failure_confirmed=function_timeout,
        )
        _update_attempt_marker(
            attempt_path,
            status=status,
            worker_failure_confirmed=function_timeout,
            **({"worker_status": "failed"} if function_timeout else {}),
            app_id=app_id,
            call_id=call_id,
            failure_stage=phase,
            last_error=remote.sanitize_failure(exc),
        )
        _write_once(plan.output_dir / FAILURE_NAME, failure)
        result_status = 1
    else:
        result_status = _finish_result(
            plan,
            attempt_path,
            result_bytes,
            app_id=app_id,
            call_id=call_id,
            receipt_path=plan.output_dir / FAILURE_NAME,
        )
    if app_id is not None and not observation_unknown:
        try:
            _stop_and_confirm(app_id, context=context)
        except Exception as exc:
            _record_shutdown_failure(plan, attempt_path, exc, app_id=app_id, call_id=call_id)
            return 1
        _update_attempt_marker(
            attempt_path,
            shutdown_status="stopped",
            shutdown_completed_at_utc=datetime.now(UTC).isoformat(),
        )
    return result_status


def _validate_attempt(plan: RealPhotoPlan) -> dict[str, Any]:
    path = _attempt_path(plan)
    if path.is_symlink() or not path.is_file():
        raise ValueError("real-photo attempt marker is missing or unsafe")
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("real-photo attempt marker is invalid") from exc
    if (
        not isinstance(marker, dict)
        or marker.get("run_id") != plan.run_id
        or marker.get("payload_sha256") != plan.payload_sha256
        or marker.get("plan_sha256") != plan.plan_sha256
        or marker.get("status") not in {"observation_unknown", "shutdown_error"}
        or marker.get("call_count") != 1
        or marker.get("spawn_count") not in {0, 1}
        or not isinstance(marker.get("app_id"), str)
        or not marker["app_id"]
        or marker.get("call_id") is not None
        and (
            not isinstance(marker.get("call_id"), str)
            or not marker["call_id"]
            or marker.get("spawn_count") != 1
        )
    ):
        raise ValueError("only the originally recorded unknown call can be recovered")
    return marker


def _recover(plan: RealPhotoPlan, *, context: dict[str, str]) -> int:
    result_path = plan.output_dir / runner.RESULT_NAME
    marker = _validate_attempt(plan)
    if result_path.exists() or result_path.is_symlink():
        if result_path.is_symlink() or not result_path.is_file():
            raise ValueError("real-photo result path is unsafe")
        runner.validate_result_bundle(
            result_path.read_bytes(),
            plan.run_id,
            expected_plan=plan.marker,
            expected_payload_sha256=plan.payload_sha256,
        )
        if marker.get("status") != "shutdown_error":
            raise FileExistsError("real-photo result already exists; refusing to overwrite it")
        remote.validate_launch_environment(context)
        remote.verify_modal_identity(**context)
        os.environ["MODAL_PROFILE"] = context["profile"]
        if importlib.metadata.version("modal") != "1.6.1":
            raise RuntimeError("Modal SDK 1.6.1 is required")
        try:
            _stop_and_confirm(marker["app_id"], context=context)
        except Exception as exc:
            _record_shutdown_failure(
                plan, _attempt_path(plan), exc, app_id=marker["app_id"], call_id=marker["call_id"]
            )
            return 1
        _update_attempt_marker(
            _attempt_path(plan),
            status="worker_returned",
            shutdown_status="stopped",
            shutdown_completed_at_utc=datetime.now(UTC).isoformat(),
        )
        return 0 if marker.get("worker_status") == "passed" else 1
    if marker.get("status") == "shutdown_error" and (
        marker.get("worker_failure_confirmed") is True
        or marker.get("failure_stage") == "result_validation"
    ):
        remote.validate_launch_environment(context)
        remote.verify_modal_identity(**context)
        os.environ["MODAL_PROFILE"] = context["profile"]
        if importlib.metadata.version("modal") != "1.6.1":
            raise RuntimeError("Modal SDK 1.6.1 is required")
        try:
            _stop_and_confirm(marker["app_id"], context=context)
        except Exception as exc:
            _record_shutdown_failure(
                plan,
                _attempt_path(plan),
                exc,
                app_id=marker["app_id"],
                call_id=marker["call_id"],
            )
            return 1
        final_status = (
            "worker_failed" if marker.get("worker_failure_confirmed") is True else "modal_error"
        )
        _update_attempt_marker(
            _attempt_path(plan),
            status=final_status,
            shutdown_status="stopped",
            shutdown_completed_at_utc=datetime.now(UTC).isoformat(),
        )
        return 1
    app_id = marker["app_id"]
    call_id = marker["call_id"]
    if call_id is None:
        # A missing call ID means spawn may have succeeded. Ordinary recovery
        # must preserve that uncertainty; closure is a separate operator action.
        return 1
    remote.validate_launch_environment(context)
    remote.verify_modal_identity(**context)
    os.environ["MODAL_PROFILE"] = context["profile"]
    if importlib.metadata.version("modal") != "1.6.1":
        raise RuntimeError("Modal SDK 1.6.1 is required")
    import modal

    recovery_observation_unknown = False
    try:
        result_bytes = modal.FunctionCall.from_id(call_id).get(timeout=0)
    except Exception as exc:
        if _is_function_timeout(exc, modal):
            _update_attempt_marker(
                _attempt_path(plan),
                status="worker_failed",
                worker_failure_confirmed=True,
                worker_status="failed",
                failure_stage="same_call_recovery",
                last_error=remote.sanitize_failure(exc),
                last_observed_at_utc=datetime.now(UTC).isoformat(),
            )
            _write_once(
                plan.output_dir / RECOVERY_FAILURE_NAME,
                _failure_receipt(
                    plan.run_id,
                    "same_call_recovery",
                    exc,
                    app_id=app_id,
                    call_id=call_id,
                    status="worker_failed",
                    worker_failure_confirmed=True,
                ),
            )
        else:
            recovery_observation_unknown = True
            _update_attempt_marker(
                _attempt_path(plan),
                status="observation_unknown",
                worker_failure_confirmed=False,
                failure_stage="same_call_recovery",
                last_observation_error=remote.sanitize_failure(exc),
                last_observed_at_utc=datetime.now(UTC).isoformat(),
            )
        result_status = 1
    else:
        result_status = _finish_result(
            plan,
            _attempt_path(plan),
            result_bytes,
            app_id=app_id,
            call_id=call_id,
            receipt_path=plan.output_dir / "real-photo-recovery-receipt.json",
        )
    if recovery_observation_unknown:
        return result_status
    try:
        _stop_and_confirm(app_id, context=context)
    except Exception as exc:
        _record_shutdown_failure(plan, _attempt_path(plan), exc, app_id=app_id, call_id=call_id)
        return 1
    _update_attempt_marker(
        _attempt_path(plan),
        shutdown_status="stopped",
        shutdown_completed_at_utc=datetime.now(UTC).isoformat(),
    )
    return result_status


def _remote_worker(payload: bytes, run_id: str) -> bytes:
    from product_scene_editor.real_photo_inference import run_remote_real_photo_inference

    return run_remote_real_photo_inference(payload, run_id)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if not args.launch and not args.recover:
        if not args.plan_dir.parent.exists():
            args.plan_dir.parent.mkdir(parents=True, exist_ok=True)
        plan_record = runner.create_real_photo_plan(
            root,
            args.plan_dir,
            run_id=args.run_id,
            adapter_archive=args.adapter_archive,
        )
        print(json.dumps(plan_record, ensure_ascii=False, sort_keys=True))
        return 0
    plan = _load_plan(args.plan_dir)
    if args.launch:
        supplied = {
            "profile": args.profile,
            "workspace": args.workspace,
            "environment": args.environment,
        }
        if not all(isinstance(value, str) and value for value in supplied.values()):
            raise ValueError("--launch requires explicit Modal profile, workspace, and environment")
        context = {key: str(value) for key, value in supplied.items()}
        return _launch(
            plan,
            context=context,
            project_root=root,
            clearance_path=args.clearance,
        )
    return _recover(plan, context=CONTEXT)


if __name__ == "__main__":
    raise SystemExit(main())
