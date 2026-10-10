"""Default to a CPU-only plan; require fresh clearance for one inference-only Modal call."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from experiments import modal_pilot
from product_scene_editor import reload_diagnostic as diagnostic
from product_scene_editor import remote

_write_json_once = modal_pilot._write_json_once
_update_attempt = modal_pilot._update_attempt_marker
_build_image = modal_pilot._build_image


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-archive",
        type=Path,
        default=Path(f"artifacts/pilots/{diagnostic.R3_RUN_ID}/pilot-input.zip"),
    )
    parser.add_argument(
        "--result-archive",
        type=Path,
        default=Path(f"artifacts/pilots/{diagnostic.R3_RUN_ID}/pilot-result.zip"),
    )
    parser.add_argument("--plan-dir", type=Path, default=Path("artifacts/reload-diagnostics/r3"))
    parser.add_argument(
        "--clearance", type=Path, default=Path(".context/reload-diagnostic-clearance.json")
    )
    parser.add_argument("--run-id", default=diagnostic.R3_RUN_ID)
    parser.add_argument(
        "--launch",
        action="store_true",
        help="run the explicitly cleared one-shot inference diagnostic",
    )
    parser.add_argument("--profile")
    parser.add_argument("--workspace")
    parser.add_argument("--environment")
    return parser


def _run_local_git(project_root: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(project_root), *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode:
        raise ValueError("clearance could not verify the local Git state")
    return result


def _validate_clearance(
    plan: diagnostic.ReplayPlan,
    *,
    project_root: Path,
    clearance_path: Path,
    context: dict[str, str],
    now: datetime | None = None,
) -> dict[str, Any]:
    if clearance_path.is_symlink() or not clearance_path.is_file():
        raise ValueError("diagnostic launch clearance is missing or unsafe")
    try:
        clearance = json.loads(clearance_path.read_text(encoding="utf-8"))
        plan_marker = (plan.output_dir / diagnostic.RUN_MARKER_NAME).read_bytes()
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("diagnostic clearance or plan is missing or invalid") from exc
    diagnostic.read_reload_payload(plan)
    if not isinstance(clearance, dict) or (
        clearance.get("schema_version") != 1
        or clearance.get("run_id") != plan.run_id
        or clearance.get("payload_sha256") != plan.payload_sha256
        or clearance.get("plan_sha256") != hashlib.sha256(plan_marker).hexdigest()
        or clearance.get("source_input_sha256") != diagnostic.R3_INPUT_SHA256
        or clearance.get("source_result_sha256") != diagnostic.R3_RESULT_SHA256
        or clearance.get("status") != "approved"
        or clearance.get("review_status") != "approved"
        or clearance.get("ci_status") != "success"
        or clearance.get("allowance_usd") != 15
        or clearance.get("gpu") != "A100-80GB"
        or clearance.get("max_runtime_seconds") != 1200
        or clearance.get("max_calls") != 1
        or context
        != {
            "profile": remote.PROFILE,
            "workspace": remote.WORKSPACE,
            "environment": remote.ENVIRONMENT,
        }
        or any(clearance.get(key) != value for key, value in context.items())
    ):
        raise ValueError("diagnostic clearance does not match the approved payload and account")
    checked = clearance.get("checked_at")
    if not isinstance(checked, str):
        raise ValueError("diagnostic clearance timestamp is missing")
    try:
        checked_at = datetime.fromisoformat(checked.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("diagnostic clearance timestamp is invalid") from exc
    current = now or datetime.now(UTC)
    if checked_at.tzinfo is None or current.tzinfo is None:
        raise ValueError("diagnostic clearance timestamps must include a timezone")
    age = current.astimezone(UTC) - checked_at.astimezone(UTC)
    if age < timedelta(0) or age > timedelta(minutes=30):
        raise ValueError("diagnostic clearance is stale or from the future")
    if _run_local_git(project_root, ["status", "--porcelain", "--untracked-files=all"]).stdout:
        raise ValueError("diagnostic launch requires a clean worktree")
    head = _run_local_git(project_root, ["rev-parse", "HEAD"]).stdout.strip()
    if clearance.get("source_commit") != head or not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ValueError("diagnostic clearance source commit does not match current HEAD")
    _run_local_git(project_root, ["verify-commit", "HEAD"])
    return clearance


def _reserve_attempt(
    plan: diagnostic.ReplayPlan, path: Path, *, source_commit: str, context: dict[str, str]
) -> None:
    plan_marker = (plan.output_dir / diagnostic.RUN_MARKER_NAME).read_bytes()
    payload = plan.payload_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != plan.payload_sha256 or not re.fullmatch(
        r"[0-9a-f]{40}", source_commit
    ):
        raise ValueError("diagnostic plan changed before attempt reservation")
    _write_json_once(
        path,
        {
            "schema_version": 1,
            "run_id": plan.run_id,
            "status": "reserved",
            "source_commit": source_commit,
            "payload_sha256": plan.payload_sha256,
            "plan_sha256": hashlib.sha256(plan_marker).hexdigest(),
            "gpu": "A100-80GB",
            "max_runtime_seconds": 1200,
            "max_calls": 1,
            "project_allowance_usd": 15,
            **context,
            "reserved_at_utc": datetime.now(UTC).isoformat(),
            "app_id": None,
            "call_id": None,
            "shutdown_check": (
                "Verify the detached diagnostic app is stopped after observing this call."
            ),
        },
    )


def _remote_worker(payload: bytes, run_id: str) -> bytes:
    from product_scene_editor.reload_diagnostic import run_remote_reload_diagnostic

    return run_remote_reload_diagnostic(payload, run_id)


def _launch(
    plan: diagnostic.ReplayPlan,
    *,
    context: dict[str, str],
    project_root: Path | None = None,
    clearance_path: Path | None = None,
) -> int:
    diagnostic.read_reload_payload(plan)
    result_path = plan.output_dir / diagnostic.RESULT_NAME
    failure_path = plan.output_dir / "modal-failure-receipt.json"
    attempt_path = plan.output_dir / diagnostic.ATTEMPT_MARKER_NAME
    if any(
        path.exists() or path.is_symlink() for path in (result_path, failure_path, attempt_path)
    ):
        raise FileExistsError("diagnostic output already has a result, failure, or attempt marker")
    if not all(context.values()):
        raise ValueError("--launch requires explicit profile, workspace, and environment")
    root = project_root or Path(__file__).resolve().parents[1]
    clearance = _validate_clearance(
        plan,
        project_root=root,
        clearance_path=clearance_path or root / ".context/reload-diagnostic-clearance.json",
        context=context,
    )
    remote.validate_launch_environment(context)
    _reserve_attempt(plan, attempt_path, source_commit=clearance["source_commit"], context=context)
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
        if not (root / "src/product_scene_editor/reload_diagnostic.py").is_file():
            raise FileNotFoundError("reload diagnostic worker source is missing")
        image = _build_image(modal, root)
        app = modal.App("product-scene-editor-reload-diagnostic", image=image)
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
            startup_timeout=300,
            timeout=900,
        )(_remote_worker)
        phase = "modal_app_startup"
        with modal.enable_output(), app.run(environment_name=context["environment"], detach=True):
            app_id = getattr(app, "app_id", None) or getattr(app, "object_id", None)
            if not isinstance(app_id, str) or not app_id:
                raise RuntimeError("Modal app ID was unavailable before spawn")
            _update_attempt(attempt_path, status="app_started", app_id=app_id)
            phase = "modal_spawn"
            call = run_once.spawn(plan.payload_path.read_bytes(), plan.run_id)
            call_id = getattr(call, "object_id", None) or getattr(call, "call_id", None)
            if not isinstance(call_id, str) or not call_id:
                raise RuntimeError("Modal call ID was unavailable after spawn")
            _update_attempt(attempt_path, status="observing", call_id=call_id)
            phase = "modal_observation"
            result = call.get()
    except Exception as exc:
        failure = modal_pilot._failure_receipt(
            plan.run_id, phase, exc, app_id=app_id, call_id=call_id
        )
        _update_attempt(
            attempt_path, status="modal_error", app_id=app_id, call_id=call_id, failure_stage=phase
        )
        _write_json_once(failure_path, failure)
        return 1
    try:
        receipt, _names = diagnostic.validate_result_bundle(result, plan.run_id)
        with result_path.open("xb") as stream:
            stream.write(result)
    except Exception as exc:
        _update_attempt(attempt_path, status="modal_error", failure_stage="result_validation")
        _write_json_once(
            failure_path,
            modal_pilot._failure_receipt(
                plan.run_id, "modal_result_validation", exc, app_id=app_id, call_id=call_id
            ),
        )
        return 1
    _update_attempt(
        attempt_path,
        status="worker_returned",
        worker_status=receipt["status"],
        completed_at_utc=datetime.now(UTC).isoformat(),
    )
    print(
        json.dumps(
            {"run_id": plan.run_id, "status": receipt["status"], "result": str(result_path)},
            sort_keys=True,
        )
    )
    return 0 if receipt["status"] == "measured" else 1


def _load_plan(output_dir: Path) -> diagnostic.ReplayPlan:
    try:
        marker = json.loads((output_dir / diagnostic.RUN_MARKER_NAME).read_text(encoding="utf-8"))
        payload_path = output_dir / diagnostic.PAYLOAD_NAME
        data = payload_path.read_bytes()
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("diagnostic plan is missing or invalid") from exc
    digest = hashlib.sha256(data).hexdigest()
    if (
        not isinstance(marker, dict)
        or marker.get("run_id") != diagnostic.R3_RUN_ID
        or marker.get("payload_file") != diagnostic.PAYLOAD_NAME
        or marker.get("payload_sha256") != digest
        or marker.get("payload_bytes") != len(data)
    ):
        raise ValueError("diagnostic plan marker does not match its payload")
    plan = diagnostic.ReplayPlan(output_dir, payload_path, diagnostic.R3_RUN_ID, len(data), digest)
    diagnostic.read_reload_payload(plan)
    return plan


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.run_id != diagnostic.R3_RUN_ID:
        raise ValueError("--run-id must identify the pinned R3 run")
    if args.launch:
        plan = _load_plan(args.plan_dir)
        context = {
            "profile": args.profile or "",
            "workspace": args.workspace or "",
            "environment": args.environment or "",
        }
        return _launch(plan, context=context, clearance_path=args.clearance)
    plan = diagnostic.plan_reload_diagnostic(args.input_archive, args.result_archive, args.plan_dir)
    diagnostic.read_reload_payload(plan)
    print(
        json.dumps(
            {
                "status": "planned",
                "run_id": plan.run_id,
                "plan_dir": str(plan.output_dir),
                "payload_bytes": plan.payload_bytes,
                "payload_sha256": plan.payload_sha256,
                "modal_called": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
