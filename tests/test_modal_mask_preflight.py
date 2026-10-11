from __future__ import annotations

import hashlib
import json
import stat
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from experiments import modal_mask_preflight as controller

_IMAGE = (
    "python:3.12.12-slim-bookworm@sha256:"
    "593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c"
)


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> controller.Plan:
    contract: Any = ModuleType("product_scene_editor.mask_preflight")
    contract.CONTEXT = {
        "profile": "reflex-personal",
        "workspace": "rajath-61258",
        "environment": "main",
    }
    contract.MODEL = {"repo_id": "model", "revision": "a" * 40}
    contract.MODEL_FILES = {"model.safetensors": {"sha256": "b" * 64, "size": 7}}
    contract.RUNTIME_PINS = {"torch": "2.5.1+cu124"}
    contract.SETTINGS = {"source_size": [512, 512]}

    def validate(payload: bytes) -> tuple[dict[str, Any], dict[str, bytes]]:
        if payload != b"payload":
            raise ValueError("payload differs")
        return (
            {
                "schema_version": 1,
                "profile": "mask-preflight",
                "source_sha": "c" * 40,
                "context": contract.CONTEXT,
                "model": contract.MODEL,
                "model_files": contract.MODEL_FILES,
                "runtime_pins": contract.RUNTIME_PINS,
                "settings": contract.SETTINGS,
                "cases": [{"case_id": f"case-{index:02d}"} for index in range(24)],
            },
            {},
        )

    contract.validate_payload_bytes = validate
    monkeypatch.setitem(sys.modules, "product_scene_editor.mask_preflight", contract)
    monkeypatch.setattr(controller, "PROJECT_ROOT", tmp_path)
    (tmp_path / "payload.zip").write_bytes(b"payload")
    plan_value = {
        "schema_version": 1,
        "source_commit": "c" * 40,
        "run_id": "mask-r1",
        "payload_path": "payload.zip",
        "payload_sha256": hashlib.sha256(b"payload").hexdigest(),
        "output_dir": "out",
        "context": contract.CONTEXT,
        "resources": {
            "gpu": "A10",
            "cpu_cores": 2,
            "memory_mib": 8192,
            "startup_timeout_seconds": 180,
            "execution_timeout_seconds": 600,
            "retries": 0,
        },
        "runtime_pins": contract.RUNTIME_PINS,
        "model": contract.MODEL,
        "model_files": contract.MODEL_FILES,
        "settings": contract.SETTINGS,
        "image": _IMAGE,
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan_value), encoding="utf-8")
    return controller._validate_plan(plan_path, root=tmp_path)


def _clearance(plan: controller.Plan, tmp_path: Path, *, checked: str) -> Path:
    evidence = {
        "checked_utc": checked,
        "context": plan.value["context"],
        "active_apps": [],
        "rates": {
            "a10_usd_per_second": 0.000306,
            "cpu_core_usd_per_second": 0.0000131,
            "memory_gib_usd_per_second": 0.00000222,
        },
        "billing": {
            "scope": "workspace-delayed-not-project-invoice",
            "workspace_gross_usd": "8.12345678",
            "workspace_net_usd": "2.00000000",
        },
    }
    evidence_path = tmp_path / "usage-rates.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    value = {
        "schema_version": 1,
        "status": "approved",
        "plan_sha256": plan.plan_sha256,
        "source_commit": plan.value["source_commit"],
        "payload_sha256": plan.payload_sha256,
        "run_id": plan.value["run_id"],
        "context": plan.value["context"],
        "ci_status": "success",
        "ci_run_id": 123,
        "ci_source_commit": plan.value["source_commit"],
        "checked_utc": checked,
        "usage_rates_evidence_path": "usage-rates.json",
        "usage_rates_evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        "estimated_seconds": 780,
        "estimated_compute_usd": 0.2729688,
        "step_reserve_usd": 1,
        "allowance_usd": 8,
        "cost_exclusions": ["image_build", "storage", "transfer", "shutdown_lag"],
    }
    path = tmp_path / "clearance.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _result(plan: controller.Plan, *, status: str = "failed", members=None) -> bytes:
    contract = sys.modules["product_scene_editor.mask_preflight"]
    files = members or {}
    artifacts = [
        {"path": name, "sha256": hashlib.sha256(content).hexdigest(), "width": 512, "height": 512}
        for name, content in files.items()
    ]
    receipt = {
        "schema_version": 1,
        "run_id": plan.value["run_id"],
        "status": status,
        "source_sha": plan.value["source_commit"],
        "payload_sha256": plan.payload_sha256,
        "context": plan.value["context"],
        "model": contract.MODEL,
        "model_files": contract.MODEL_FILES,
        "runtime_pins": contract.RUNTIME_PINS,
        "settings": contract.SETTINGS,
        "mask_status": "draft_unaccepted",
        "training_function_called": False,
        "optimizer_updates": 0,
        "output_artifacts": artifacts,
    }
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in [
            ("receipt.json", json.dumps(receipt, sort_keys=True)),
            *files.items(),
        ]:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            archive.writestr(info, content)
    return output.getvalue()


def test_default_cli_only_validates_frozen_plan_and_payload(tmp_path, monkeypatch, capsys) -> None:
    plan = _fixture(tmp_path, monkeypatch)
    calls = []
    imports = []
    monkeypatch.setattr(controller, "_launch", lambda *_args, **_kwargs: calls.append("launch"))
    monkeypatch.setattr(controller, "_recover", lambda *_args, **_kwargs: calls.append("recover"))
    original_import = controller.importlib.import_module

    def record_import(name, *args, **kwargs):
        imports.append(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(controller.importlib, "import_module", record_import)

    assert controller.main(["--plan", str(plan.path)]) == 0
    assert calls == []
    assert "modal" not in imports
    assert json.loads(capsys.readouterr().out)["status"] == "validated"


def test_plan_pin_mismatch_fails_before_payload_clearance_or_identity(tmp_path, monkeypatch):
    plan = _fixture(tmp_path, monkeypatch)
    value = json.loads(plan.path.read_text())
    value["resources"]["gpu"] = "A100"
    plan.path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="frozen runtime contract"):
        controller.main(["--plan", str(plan.path), "--launch", "--clearance", "missing.json"])
    assert not (plan.output_dir / controller.ATTEMPT_NAME).exists()


def test_source_requires_head_cached_main_and_clean_source(tmp_path, monkeypatch):
    plan = _fixture(tmp_path, monkeypatch)
    calls = []

    def git(_root, *args):
        calls.append(args)
        return "d" * 40 if args == ("rev-parse", "HEAD") else "c" * 40

    monkeypatch.setattr(controller, "_git", git)
    with pytest.raises(ValueError, match="HEAD"):
        controller._verify_source(plan)
    assert calls == [("rev-parse", "HEAD")]


def test_clearance_binds_fresh_evidence_hash_context_and_empty_app_list(tmp_path, monkeypatch):
    plan = _fixture(tmp_path, monkeypatch)
    now = controller.datetime.now(controller.UTC)
    path = _clearance(plan, tmp_path, checked=now.isoformat())
    assert controller._validate_clearance(plan, path, now=now)["ci_run_id"] == 123
    evidence_path = tmp_path / "usage-rates.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["active_apps"] = [{"app_id": "ap-running"}]
    evidence_path.write_text(json.dumps(evidence))
    clearance = json.loads(path.read_text())
    clearance["usage_rates_evidence_sha256"] = hashlib.sha256(
        evidence_path.read_bytes()
    ).hexdigest()
    path.write_text(json.dumps(clearance))
    with pytest.raises(ValueError, match="active apps"):
        controller._validate_clearance(plan, path, now=now)


def test_clearance_rejects_stale_timestamp_and_mismatched_evidence_clock(tmp_path, monkeypatch):
    plan = _fixture(tmp_path, monkeypatch)
    now = controller.datetime.now(controller.UTC)
    path = _clearance(plan, tmp_path, checked=(now - controller.timedelta(minutes=31)).isoformat())
    with pytest.raises(ValueError, match="stale"):
        controller._validate_clearance(plan, path, now=now)
    path = _clearance(plan, tmp_path, checked=now.isoformat())
    evidence_path = tmp_path / "usage-rates.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["checked_utc"] = (now - controller.timedelta(minutes=1)).isoformat()
    evidence_path.write_text(json.dumps(evidence))
    clearance = json.loads(path.read_text())
    clearance["usage_rates_evidence_sha256"] = hashlib.sha256(
        evidence_path.read_bytes()
    ).hexdigest()
    path.write_text(json.dumps(clearance))
    with pytest.raises(ValueError, match="stale.*account|targets another"):
        controller._validate_clearance(plan, path, now=now)


def test_clearance_failure_precedes_modal_identity_and_attempt(tmp_path, monkeypatch):
    plan = _fixture(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(controller, "_verify_source", lambda _plan: calls.append("source"))
    monkeypatch.setattr(controller, "_verify_identity", lambda _ctx: calls.append("identity"))
    with pytest.raises(ValueError, match="clearance is missing"):
        controller._launch(plan, clearance_path=tmp_path / "missing.json")
    assert calls == ["source"]
    assert not (plan.output_dir / controller.ATTEMPT_NAME).exists()


def test_attempt_marker_is_exclusive_and_modal_ids_are_immutable(tmp_path, monkeypatch):
    plan = _fixture(tmp_path, monkeypatch)
    path = controller._reserve_attempt(plan)
    with pytest.raises(FileExistsError):
        controller._reserve_attempt(plan)
    controller._update_attempt(path, status="app_started", app_id="ap-original")
    with pytest.raises(ValueError, match="immutable"):
        controller._update_attempt(path, app_id="ap-other")
    controller._update_attempt(path, status="spawn_reserved", spawn_count=1)
    with pytest.raises(ValueError, match="one reserved spawn"):
        controller._update_attempt(path, spawn_count=2)


def test_recovery_requires_manual_stop_when_spawn_has_no_call_id(tmp_path, monkeypatch, capsys):
    plan = _fixture(tmp_path, monkeypatch)
    path = controller._reserve_attempt(plan)
    controller._update_attempt(path, status="spawn_reserved", app_id="ap-same", spawn_count=1)
    monkeypatch.setattr(controller, "_verify_identity", lambda _ctx: pytest.fail("unexpected API"))
    assert controller._recover(plan) == 1
    assert json.loads(capsys.readouterr().out)["app_id"] == "ap-same"


def test_result_validation_preserves_draft_only_receipt_and_rejects_unsafe_members(
    tmp_path, monkeypatch
):
    plan = _fixture(tmp_path, monkeypatch)
    assert controller._validate_result(_result(plan), plan)["status"] == "failed"
    with pytest.raises(ValueError, match="safe receipt ZIP"):
        controller._validate_result(_result(plan, members={"../escape.png": b"opaque"}), plan)


def test_shutdown_calls_stop_once_and_caps_list_checks_at_four(tmp_path, monkeypatch):
    plan = _fixture(tmp_path, monkeypatch)
    marker = controller._reserve_attempt(plan)
    controller._update_attempt(
        marker, status="observing", app_id="ap-exact", call_id="fc-exact", spawn_count=1
    )
    running = '[{"app_id":"ap-exact","state":"running","tasks":"1"}]'
    responses = [SimpleNamespace(returncode=0, stdout="")]
    responses.extend(SimpleNamespace(returncode=0, stdout=running) for _ in range(4))
    calls, waits = [], []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return responses.pop(0)

    monkeypatch.setattr(controller.subprocess, "run", run)
    monkeypatch.setattr(controller.time, "sleep", waits.append)
    with pytest.raises(RuntimeError, match="exact app stopped"):
        controller._stop_and_confirm(plan, "ap-exact")
    assert len(calls) == 5
    assert sum("stop" in call[0] for call in calls) == 1
    assert all(call[1]["timeout"] == 5 for call in calls[1:])
    assert waits == [2, 2, 2]
    events = [
        json.loads(path.read_text()) for path in plan.output_dir.glob("mask-preflight-*.json")
    ]
    actions = [event["action"] for event in events if "action" in event]
    assert actions.count("stop_request") == 1
    assert actions.count("stop_response") == 1
    assert actions.count("app_list") == 4
    assert all(
        event["confirmed_stopped"] is False for event in events if event.get("action") == "app_list"
    )


def test_observer_lock_excludes_second_owner_and_releases(tmp_path, monkeypatch):
    plan = _fixture(tmp_path, monkeypatch)
    with controller._observer(plan, "launch"):
        with pytest.raises(RuntimeError, match="another mask preflight observer"):
            with controller._observer(plan, "recover"):
                pytest.fail("second observer entered")
    with controller._observer(plan, "recover"):
        pass
    owner = json.loads((plan.output_dir / controller.OBSERVER_NAME).read_text())
    assert owner["action"] == "recover"


def test_client_timeout_is_unknown_but_remote_function_timeout_is_definitive():
    class FunctionTimeoutError(Exception):
        pass

    class InternalFailure(Exception):
        pass

    modal = SimpleNamespace(
        exception=SimpleNamespace(
            FunctionTimeoutError=FunctionTimeoutError,
            InternalFailure=InternalFailure,
        )
    )
    assert not controller._is_terminal_call_error(TimeoutError("observer deadline"), modal)
    assert controller._is_terminal_call_error(FunctionTimeoutError("remote execution limit"), modal)
    assert controller._is_terminal_call_error(InternalFailure("remote worker failed"), modal)


def test_launch_bounds_observer_wait_and_keeps_client_timeout_pending(tmp_path, monkeypatch):
    plan = _fixture(tmp_path, monkeypatch)
    clearance = _clearance(
        plan, tmp_path, checked=controller.datetime.now(controller.UTC).isoformat()
    )
    observed_timeouts = []

    class ObserverTimeout(Exception):
        pass

    class FunctionTimeoutError(Exception):
        pass

    class Call:
        object_id = "fc-observer-timeout"

        def get(self, *, timeout):
            observed_timeouts.append(timeout)
            raise ObserverTimeout("client observer timed out")

    class Function:
        def __call__(self, _worker):
            return self

        def spawn(self, *_args):
            return Call()

    class AppContext:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class App:
        app_id = "ap-observer-timeout"

        def __init__(self, *_args, **_kwargs):
            pass

        def function(self, **kwargs):
            assert kwargs["cpu"] == (2, 2)
            assert kwargs["memory"] == (8192, 8192)
            assert kwargs["max_containers"] == 1
            return Function()

        def run(self, **_kwargs):
            return AppContext()

    modal = SimpleNamespace(
        App=App,
        enable_output=AppContext,
        exception=SimpleNamespace(
            TimeoutError=ObserverTimeout,
            FunctionTimeoutError=FunctionTimeoutError,
        ),
    )
    original_import = controller.importlib.import_module
    monkeypatch.setattr(
        controller.importlib,
        "import_module",
        lambda name: modal if name == "modal" else original_import(name),
    )
    monkeypatch.setattr(controller.importlib.metadata, "version", lambda _name: "1.6.1")
    monkeypatch.setattr(controller, "_build_image", lambda *_args: object())
    monkeypatch.setattr(controller, "_verify_source", lambda _plan: None)
    monkeypatch.setattr(controller, "_verify_identity", lambda _context: None)
    monkeypatch.setattr(
        controller.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("unknown observation must not stop the app"),
    )

    assert controller._launch(plan, clearance_path=clearance) == 1
    assert observed_timeouts == [
        controller.STARTUP_TIMEOUT_SECONDS + controller.EXECUTION_TIMEOUT_SECONDS
    ]
    attempt = controller._read_attempt(plan)
    assert attempt["status"] == "observation_pending"
    assert attempt["call_id"] == "fc-observer-timeout"


def test_recovery_uses_cached_result_and_resumes_exact_app_shutdown(tmp_path, monkeypatch):
    plan = _fixture(tmp_path, monkeypatch)
    marker = controller._reserve_attempt(plan)
    controller._update_attempt(
        marker,
        status="shutdown_pending",
        app_id="ap-same",
        call_id="fc-same",
        spawn_count=1,
        stop_requested_at_utc=controller.datetime.now(controller.UTC).isoformat(),
    )
    result = _result(plan)
    result_path = plan.output_dir / controller.RESULT_NAME
    result_path.write_bytes(result)
    calls = []

    def no_modal_import(name):
        if name == "modal":
            pytest.fail("cached result recovery must not fetch the provider call")
        return original_import(name)

    original_import = controller.importlib.import_module
    monkeypatch.setattr(controller.importlib, "import_module", no_modal_import)
    monkeypatch.setattr(controller, "_verify_identity", lambda _context: None)

    responses = [
        SimpleNamespace(returncode=0, stdout=""),
        SimpleNamespace(
            returncode=0,
            stdout='[{"app_id":"ap-same","state":"stopped","tasks":"0"}]',
        ),
    ]

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return responses.pop(0)

    monkeypatch.setattr(controller.subprocess, "run", run)
    assert controller._recover(plan) == 1
    assert result_path.read_bytes() == result
    assert len(calls) == 2
    assert calls[0][0][-1] == "ap-same"
    assert "stop" in calls[0][0]
    assert "list" in calls[1][0]
    attempt = controller._read_attempt(plan)
    assert attempt["status"] == "stopped"
    assert attempt["worker_status"] == "failed"


def test_invalid_cached_result_is_preserved_and_still_stops_recorded_app(tmp_path, monkeypatch):
    plan = _fixture(tmp_path, monkeypatch)
    marker = controller._reserve_attempt(plan)
    controller._update_attempt(
        marker,
        status="shutdown_pending",
        app_id="ap-invalid",
        call_id="fc-invalid",
        spawn_count=1,
    )
    invalid_result = b"preserve these invalid bytes"
    result_path = plan.output_dir / controller.RESULT_NAME
    result_path.write_bytes(invalid_result)
    monkeypatch.setattr(controller, "_verify_identity", lambda _context: None)
    calls = []
    responses = [
        SimpleNamespace(returncode=0, stdout=""),
        SimpleNamespace(
            returncode=0,
            stdout='[{"app_id":"ap-invalid","state":"stopped","tasks":"0"}]',
        ),
    ]

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return responses.pop(0)

    monkeypatch.setattr(controller.subprocess, "run", run)
    assert controller._recover(plan) == 1
    assert result_path.read_bytes() == invalid_result
    assert calls[0][0][-1] == "ap-invalid"
    failure = next(plan.output_dir.glob("mask-preflight-failure-*.json"))
    assert json.loads(failure.read_text())["stage"] == "cached_result_validation"
    assert controller._read_attempt(plan)["status"] == "stopped"


def test_failure_events_are_unique_durable_and_sanitized(tmp_path, monkeypatch):
    plan = _fixture(tmp_path, monkeypatch)
    marker = controller._reserve_attempt(plan)
    controller._update_attempt(
        marker, status="observing", app_id="ap-fail", call_id="fc-fail", spawn_count=1
    )
    error = RuntimeError("authorization=secret-value")
    controller._failure(plan, "observe", error, pending=True)
    controller._failure(plan, "observe", error, pending=True)
    paths = list(plan.output_dir.glob("mask-preflight-failure-*.json"))
    assert len(paths) == 2
    assert len({path.name for path in paths}) == 2
    assert all(
        "authorization=[REDACTED]" in json.loads(path.read_text())["message"] for path in paths
    )


def test_successful_shutdown_persists_exact_stopped_zero_observation(tmp_path, monkeypatch):
    plan = _fixture(tmp_path, monkeypatch)
    marker = controller._reserve_attempt(plan)
    controller._update_attempt(marker, status="app_started", app_id="ap-stopped")
    stopped = '[{"app_id":"ap-stopped","state":"stopped","tasks":"0"}]'
    responses = [
        SimpleNamespace(returncode=0, stdout=""),
        SimpleNamespace(returncode=0, stdout=stopped),
    ]
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return responses.pop(0)

    monkeypatch.setattr(controller.subprocess, "run", run)
    controller._stop_and_confirm(plan, "ap-stopped")
    observations = [
        json.loads(path.read_text())
        for path in plan.output_dir.glob("mask-preflight-app_list-*.json")
    ]
    assert len(observations) == 1
    assert observations[0]["confirmed_stopped"] is True
    assert observations[0]["matching_apps"] == [
        {"app_id": "ap-stopped", "state": "stopped", "tasks": "0"}
    ]
    assert controller._read_attempt(plan)["status"] == "stopped"
