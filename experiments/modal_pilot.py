"""Default to a CPU-only plan; require --launch for one bounded Modal call."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import zipfile
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

from product_scene_editor import dataset, pilot, remote

APP_NAME = "product-scene-editor-pilot"
RESULT_NAME = "pilot-result.zip"
MODAL_FAILURE_NAME = "modal-failure-receipt.json"
REMOTE_TIMEOUT_SECONDS = 2700
REMOTE_STARTUP_TIMEOUT_SECONDS = 300
ATTEMPT_MARKER_NAME = "modal-attempt.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("artifacts/pilot-data-2026-10-08-r2")
    )
    parser.add_argument("--plan-dir", type=Path, default=Path("artifacts/pilots/pilot-001"))
    parser.add_argument("--run-id", default="pilot-001")
    parser.add_argument("--launch", action="store_true", help="run the explicit one-shot GPU pilot")
    parser.add_argument("--profile")
    parser.add_argument("--workspace")
    parser.add_argument("--environment")
    return parser


def _write_json_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def _reserve_launch_attempt(
    plan: pilot.PilotPlan,
    marker_path: Path,
    *,
    source_commit: str,
    context: dict[str, str],
) -> dict[str, Any]:
    payload = plan.payload_path.read_bytes()
    plan_marker = (plan.output_dir / pilot.RUN_MARKER_NAME).read_bytes()
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    if payload_sha256 != plan.payload_sha256:
        raise ValueError("pilot payload changed before launch reservation")
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("launch source commit must be a full Git SHA")
    if context != {
        "profile": remote.PROFILE,
        "workspace": remote.WORKSPACE,
        "environment": remote.ENVIRONMENT,
    }:
        raise ValueError("launch context must match the reviewed Modal target")
    marker = {
        "schema_version": 1,
        "run_id": plan.run_id,
        "status": "reserved",
        "source_commit": source_commit,
        "payload_sha256": payload_sha256,
        "plan_sha256": hashlib.sha256(plan_marker).hexdigest(),
        "source_dataset_manifest_sha256": plan.source_dataset_manifest_sha256,
        **context,
        "reserved_at_utc": datetime.now(UTC).isoformat(),
        "app_id": None,
        "call_id": None,
    }
    _write_json_once(marker_path, marker)
    return marker


def _update_attempt_marker(marker_path: Path, **updates: Any) -> dict[str, Any]:
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if not isinstance(marker, dict):
        raise ValueError("Modal attempt marker is invalid")
    for key in ("app_id", "call_id"):
        if key not in updates:
            continue
        new_value = updates.get(key)
        old_value = marker.get(key)
        if old_value is not None and new_value != old_value:
            raise ValueError(f"Modal attempt {key} is immutable once recorded")
        if new_value is not None and (not isinstance(new_value, str) or not new_value):
            raise ValueError(f"Modal attempt {key} must be a non-empty string")
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
        raise ValueError("launch clearance could not verify the local Git state")
    return completed


def _validate_launch_clearance(
    plan: pilot.PilotPlan,
    *,
    project_root: Path,
    clearance_path: Path,
    context: dict[str, str],
    now: datetime | None = None,
) -> dict[str, Any]:
    if clearance_path.is_symlink() or not clearance_path.is_file():
        raise ValueError("launch clearance is missing or unsafe")
    try:
        clearance = json.loads(clearance_path.read_text(encoding="utf-8"))
        payload = plan.payload_path.read_bytes()
        plan_marker = (plan.output_dir / pilot.RUN_MARKER_NAME).read_bytes()
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("launch clearance or bound plan is missing or invalid") from exc
    if not isinstance(clearance, dict):
        raise ValueError("launch clearance must be a JSON object")

    payload_sha256 = hashlib.sha256(payload).hexdigest()
    plan_sha256 = hashlib.sha256(plan_marker).hexdigest()
    if (
        payload_sha256 != plan.payload_sha256
        or clearance.get("payload_sha256") != payload_sha256
        or clearance.get("plan_sha256") != plan_sha256
        or clearance.get("source_dataset_manifest_sha256") != plan.source_dataset_manifest_sha256
        or clearance.get("run_id") != plan.run_id
        or clearance.get("status") != "approved"
        or clearance.get("review_status") != "approved"
        or clearance.get("ci_status") != "success"
        or clearance.get("allowance_usd") != 15
        or clearance.get("profile") != context.get("profile")
        or clearance.get("workspace") != context.get("workspace")
        or clearance.get("environment") != context.get("environment")
        or context
        != {
            "profile": remote.PROFILE,
            "workspace": remote.WORKSPACE,
            "environment": remote.ENVIRONMENT,
        }
    ):
        raise ValueError("launch clearance does not match the approved plan and account")

    checked_at_value = clearance.get("checked_at")
    if not isinstance(checked_at_value, str):
        raise ValueError("launch clearance timestamp is missing")
    try:
        checked_at = datetime.fromisoformat(checked_at_value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("launch clearance timestamp is invalid") from exc
    current_time = now or datetime.now(UTC)
    if checked_at.tzinfo is None or current_time.tzinfo is None:
        raise ValueError("launch clearance timestamps must include a timezone")
    age = current_time.astimezone(UTC) - checked_at.astimezone(UTC)
    if age < timedelta(0) or age > timedelta(minutes=30):
        raise ValueError("launch clearance is stale or from the future")

    if _run_local_git(project_root, ["status", "--porcelain", "--untracked-files=all"]).stdout:
        raise ValueError("launch requires a clean current Git worktree")
    head = _run_local_git(project_root, ["rev-parse", "HEAD"]).stdout.strip()
    if clearance.get("source_commit") != head or not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ValueError("launch clearance source commit does not match current HEAD")
    _run_local_git(project_root, ["verify-commit", "HEAD"])
    return clearance


def _remote_worker(payload: bytes, run_id: str) -> bytes:
    from product_scene_editor.remote import run_remote_training

    return run_remote_training(payload, run_id)


def _build_image(modal: Any, project_root: Path) -> Any:
    packages = [
        f"{name}=={version}" for name, version in remote.RUNTIME_PINS.items() if name != "imageio"
    ]
    packages.append(f"imageio[ffmpeg]=={remote.RUNTIME_PINS['imageio']}")
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .env({"USE_HUB_KERNELS": "NO"})
        .apt_install("git")
        .pip_install(*packages, extra_index_url=remote.TORCH_INDEX)
        .run_commands(
            "git clone --filter=blob:none --no-checkout "
            "https://github.com/modelscope/DiffSynth-Studio.git /opt/DiffSynth-Studio",
            f"git -C /opt/DiffSynth-Studio checkout --detach {remote.DIFFSYNTH_COMMIT} "
            f'&& test "$(git -C /opt/DiffSynth-Studio rev-parse HEAD)" = {remote.DIFFSYNTH_COMMIT} '
            "&& pip install --no-deps /opt/DiffSynth-Studio",
        )
        .add_local_python_source("product_scene_editor", copy=True)
    )
    if not (project_root / "src/product_scene_editor/remote.py").is_file():
        raise FileNotFoundError("product_scene_editor remote worker source is missing")
    return image


def _validate_result_bundle(data: bytes, run_id: str) -> tuple[dict[str, Any], set[str]]:
    if not isinstance(data, bytes) or not data or len(data) > remote.MAX_RESULT_BYTES:
        raise ValueError("Modal returned an empty or oversized result")
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            names = set(archive.namelist())
            if any(
                PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
                for name in names
            ):
                raise ValueError("Modal returned an unsafe artifact path")
            receipt = json.loads(archive.read("receipt.json"))
    except (OSError, KeyError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise ValueError("Modal result is not a valid receipt archive") from exc
    if receipt.get("run_id") != run_id or receipt.get("status") not in {"passed", "failed"}:
        raise ValueError("Modal result receipt does not match this run")
    required_success = {
        "receipt.json",
        "adapter/step-8.safetensors",
        "images/base.png",
        "images/trained.png",
        "images/reloaded.png",
    }
    allowed_failure = {
        "receipt.json",
        "adapter/step-8.safetensors",
        "images/base.png",
        "images/trained.png",
        "images/reloaded.png",
    }
    if receipt["status"] == "passed" and names != required_success:
        raise ValueError("Modal result artifact set does not match its status")
    if receipt["status"] == "failed" and not names <= allowed_failure:
        raise ValueError("Modal result artifact set does not match its status")
    return receipt, names


def _failure_receipt(
    run_id: str,
    stage: str,
    error: BaseException,
    *,
    app_id: str | None = None,
    call_id: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "status": "modal_error",
        "failure": {
            "stage": stage,
            "type": type(error).__name__,
            "message": remote.sanitize_failure(error),
        },
        "worker_failure_confirmed": False,
        "app_id": app_id,
        "call_id": call_id,
        "same_call_recovery": (
            "Inspect the recorded Modal call and observe that same call only; never spawn a "
            "replacement automatically. After resolution, verify the detached app is stopped."
        ),
        "shutdown_check": "Root must verify this detached app is stopped after resolving the call.",
    }


def _launch(plan: pilot.PilotPlan, *, profile: str, workspace: str, environment: str) -> int:
    if plan.source_dataset_manifest_sha256 != pilot.APPROVED_DATASET_MANIFEST_SHA256:
        raise ValueError("pilot plan does not use the reviewed dataset manifest")
    context = {"profile": profile, "workspace": workspace, "environment": environment}
    result_path = plan.output_dir / RESULT_NAME
    failure_path = plan.output_dir / MODAL_FAILURE_NAME
    attempt_path = plan.output_dir / ATTEMPT_MARKER_NAME
    if (
        result_path.exists()
        or result_path.is_symlink()
        or failure_path.exists()
        or failure_path.is_symlink()
        or attempt_path.exists()
        or attempt_path.is_symlink()
    ):
        raise FileExistsError("pilot output already has a result, failure, or attempt marker")
    validated = pilot.read_pilot_payload(plan)
    payload = plan.payload_path.read_bytes()
    if len(validated.train_rows) != remote.TRAINING_STEPS or len(validated.dev_rows) != 1:
        raise ValueError("pilot launch requires exactly eight train pairs and one dev pair")

    project_root = Path(__file__).resolve().parents[1]
    clearance = _validate_launch_clearance(
        plan,
        project_root=project_root,
        clearance_path=project_root / ".context" / "launch-clearance.json",
        context=context,
    )
    _reserve_launch_attempt(
        plan,
        attempt_path,
        source_commit=clearance["source_commit"],
        context=context,
    )

    app_id: str | None = None
    call_id: str | None = None
    phase = "modal_identity"
    try:
        remote.verify_modal_identity(profile=profile, workspace=workspace, environment=environment)
        os.environ["MODAL_PROFILE"] = profile
        phase = "modal_import"
        import modal

        modal_version = importlib.metadata.version("modal")
        if modal_version != "1.6.1":
            raise RuntimeError("Modal SDK 1.6.1 is required")
        phase = "modal_build"
        image = _build_image(modal, project_root)
        app = modal.App(APP_NAME, image=image)
        run_once = app.function(
            gpu="A100-80GB",
            cpu=(4.0, 4.0),
            memory=(65536, 65536),
            ephemeral_disk=30000,
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
            max_inputs=1,
        )(_remote_worker)
        phase = "modal_app_startup"
        with modal.enable_output(), app.run(environment_name=environment, detach=True):
            app_id_value = getattr(app, "app_id", None) or getattr(app, "object_id", None)
            if not isinstance(app_id_value, str) or not app_id_value:
                raise RuntimeError("Modal app did not expose an app ID before spawn")
            app_id = app_id_value
            _update_attempt_marker(attempt_path, status="app_started", app_id=app_id)
            phase = "modal_spawn"
            call = run_once.spawn(payload, plan.run_id)
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
        failure = _failure_receipt(plan.run_id, phase, exc, app_id=app_id, call_id=call_id)
        _update_attempt_marker(
            attempt_path,
            status="modal_error",
            failure_stage=phase,
            last_error=remote.sanitize_failure(exc),
        )
        _write_json_once(failure_path, failure)
        print(json.dumps(failure, sort_keys=True))
        return 1

    try:
        receipt, _names = _validate_result_bundle(result_bytes, plan.run_id)
        with result_path.open("xb") as stream:
            stream.write(result_bytes)
    except Exception as exc:
        failure = _failure_receipt(
            plan.run_id,
            "modal_result_validation",
            exc,
            app_id=app_id,
            call_id=call_id,
        )
        _update_attempt_marker(
            attempt_path,
            status="modal_error",
            failure_stage="modal_result_validation",
            last_error=remote.sanitize_failure(exc),
        )
        _write_json_once(failure_path, failure)
        print(json.dumps(failure, sort_keys=True))
        return 1
    _update_attempt_marker(
        attempt_path,
        status="worker_returned",
        worker_status=receipt["status"],
        completed_at_utc=datetime.now(UTC).isoformat(),
    )
    print(
        json.dumps(
            {"status": receipt["status"], "run_id": plan.run_id, "result": str(result_path)},
            sort_keys=True,
        )
    )
    return 0 if receipt["status"] == "passed" else 1


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.launch:
        plan = pilot.load_pilot_plan(args.plan_dir)
        if args.run_id != plan.run_id:
            raise ValueError("--run-id does not match the write-once pilot plan")
        if not all((args.profile, args.workspace, args.environment)):
            raise ValueError("--launch requires explicit --profile, --workspace, and --environment")
        return _launch(
            plan,
            profile=args.profile,
            workspace=args.workspace,
            environment=args.environment,
        )

    if not args.dataset_dir.is_dir():
        raise FileNotFoundError("the reviewed pilot dataset must be present before planning")
    dataset.verify_dataset(args.dataset_dir)
    plan = pilot.plan_pilot(args.dataset_dir, args.plan_dir, run_id=args.run_id)
    validated = pilot.read_pilot_payload(plan)
    print(
        json.dumps(
            {
                "status": "planned",
                "run_id": plan.run_id,
                "plan_dir": str(plan.output_dir),
                "payload_bytes": plan.payload_bytes,
                "payload_sha256": plan.payload_sha256,
                "training_pairs": len(validated.train_rows),
                "development_pairs": len(validated.dev_rows),
                "modal_called": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
