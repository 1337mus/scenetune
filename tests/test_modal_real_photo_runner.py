from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from experiments import modal_real_photo_runner as runner


def test_stopped_app_requires_strict_identity_and_zero_tasks() -> None:
    app = {
        "app_id": "ap-123",
        "description": "real-photo",
        "state": "stopped",
        "tasks": "0",
        "created_at": None,
        "stopped_at": None,
    }
    assert runner._app_list_confirms_stopped(json.dumps([app]), "ap-123")
    app["tasks"] = "1"
    assert not runner._app_list_confirms_stopped(json.dumps([app]), "ap-123")
    app["tasks"] = "0"
    app["state"] = "running"
    assert not runner._app_list_confirms_stopped(json.dumps([app]), "ap-123")
    assert not runner._app_list_confirms_stopped("[]", "ap-123")


def _app_record(app_id: str = "ap-123", state: str = "stopped", tasks: str = "0"):
    return {
        "app_id": app_id,
        "description": "real-photo",
        "state": state,
        "tasks": tasks,
        "created_at": None,
        "stopped_at": None,
    }


def _completed(returncode: int, stdout: str = ""):
    from types import SimpleNamespace

    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")


def test_bounded_shutdown_waits_for_stopped_and_records_both_observations(
    tmp_path, monkeypatch
) -> None:
    import sys
    import time

    context = {"profile": "personal-profile", "workspace": "personal", "environment": "main"}
    stopping = json.dumps([_app_record(state="stopping")])
    stopped = json.dumps([_app_record()])
    results = [_completed(0), _completed(0, stopping), _completed(0, stopped)]
    calls = []
    sleeps = []

    def run(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return results.pop(0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(time, "sleep", sleeps.append)

    runner._stop_and_confirm("ap-123", context=context, output_dir=tmp_path)

    assert len(calls) == 3
    assert calls[0][0] == [
        sys.executable,
        "-m",
        "modal",
        "app",
        "stop",
        "--yes",
        "--env",
        "main",
        "ap-123",
    ]
    assert calls[0][1]["timeout"] == 30
    assert calls[0][1]["env"]["MODAL_PROFILE"] == "personal-profile"
    assert calls[1][0][3:6] == ["app", "list", "--json"]
    assert calls[1][1]["timeout"] == calls[2][1]["timeout"] == 10
    assert calls[1][1]["env"]["MODAL_PROFILE"] == "personal-profile"
    assert [call[0][5] for call in calls[1:]] == ["--json", "--json"]
    assert sleeps == [2]

    evidence = json.loads((tmp_path / runner.SHUTDOWN_OBSERVATION_NAME).read_text())
    assert evidence["app_id"] == "ap-123"
    assert evidence["status"] == "stopped"
    assert evidence["stop"]["returncode"] == 0
    assert [item["app_list_json"] for item in evidence["observations"]] == [stopping, stopped]
    assert not any(key in evidence for key in ("token", "secret", "password"))


def test_bounded_shutdown_accepts_nonzero_stop_when_later_observation_confirms_closed(
    tmp_path, monkeypatch
) -> None:
    context = {"profile": "personal-profile", "workspace": "personal", "environment": "main"}
    responses = [_completed(7), _completed(0, json.dumps([_app_record()]))]
    calls = []

    def run(arguments, **kwargs):
        calls.append(arguments)
        return responses.pop(0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    runner._stop_and_confirm("ap-123", context=context, output_dir=tmp_path)
    evidence = json.loads((tmp_path / runner.SHUTDOWN_OBSERVATION_NAME).read_text())
    assert sum("stop" in arguments for arguments in calls) == 1
    assert evidence["stop"]["returncode"] == 7
    assert evidence["status"] == "stopped"


@pytest.mark.parametrize(
    "apps",
    [
        [],
        [_app_record(app_id="ap-other")],
        [_app_record(tasks="1")],
    ],
    ids=["absent-app", "wrong-app", "positive-tasks"],
)
def test_bounded_shutdown_never_closes_without_exact_stopped_zero_task_app(
    tmp_path, monkeypatch, apps
) -> None:
    import time

    context = {"profile": "personal-profile", "workspace": "personal", "environment": "main"}
    calls = []
    sleeps = []

    def run(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return _completed(0) if len(calls) == 1 else _completed(0, json.dumps(apps))

    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(time, "sleep", sleeps.append)

    with pytest.raises(RuntimeError, match="stopped with zero tasks"):
        runner._stop_and_confirm("ap-123", context=context, output_dir=tmp_path)

    evidence = json.loads((tmp_path / runner.SHUTDOWN_OBSERVATION_NAME).read_text())
    assert len(calls) == 7
    assert len(evidence["observations"]) == 6
    assert evidence["status"] == "unconfirmed"
    assert sleeps == [2, 2, 2, 2, 2]
    assert all(item["app_list_json"] == json.dumps(apps) for item in evidence["observations"])


def test_bounded_shutdown_preserves_timeout_and_malformed_observation_diagnostics(
    tmp_path, monkeypatch
) -> None:
    import subprocess
    import time

    context = {"profile": "personal-profile", "workspace": "personal", "environment": "main"}
    calls = []

    def run(arguments, **kwargs):
        calls.append(arguments)
        if len(calls) in {1, 2}:
            raise subprocess.TimeoutExpired(
                arguments, timeout=kwargs["timeout"], output="secret=do-not-save"
            )
        if len(calls) == 3:
            return _completed(0, "not-json")
        return _completed(0, json.dumps([_app_record()]))

    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    runner._stop_and_confirm("ap-123", context=context, output_dir=tmp_path)

    evidence = json.loads((tmp_path / runner.SHUTDOWN_OBSERVATION_NAME).read_text())
    assert evidence["status"] == "stopped"
    assert evidence["stop"]["returncode"] is None
    assert evidence["stop"]["error_type"] == "TimeoutExpired"
    assert evidence["observations"][0]["returncode"] is None
    assert evidence["observations"][0]["error_type"] == "TimeoutExpired"
    assert evidence["observations"][0]["app_list_json"] is None
    assert evidence["observations"][1]["app_list_json"] is None
    assert evidence["observations"][1]["error"] == "Modal app list returned malformed JSON"
    assert "secret=do-not-save" not in (tmp_path / runner.SHUTDOWN_OBSERVATION_NAME).read_text()


def test_attempt_update_preserves_modal_ids_and_never_allows_a_second_call(
    tmp_path,
) -> None:
    marker = tmp_path / "attempt.json"
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "fixture",
                "status": "reserved",
                "app_id": None,
                "call_id": None,
                "call_count": 0,
            }
        )
    )
    runner._update_attempt_marker(marker, status="observing", app_id="ap-123", call_id="fc-123")
    with pytest.raises(ValueError):
        runner._update_attempt_marker(marker, call_id="fc-456")
    with pytest.raises(ValueError):
        runner._update_attempt_marker(marker, call_count=2)


def test_unknown_launch_observation_preserves_ids_and_leaves_shutdown_pending(
    tmp_path, monkeypatch
) -> None:
    from contextlib import nullcontext
    from types import SimpleNamespace

    class ConnectionError(Exception):
        pass

    plan = runner.RealPhotoPlan(
        output_dir=tmp_path,
        payload_path=tmp_path / "payload.zip",
        run_id="fixture",
        payload_bytes=7,
        payload_sha256="a" * 64,
        plan_sha256="b" * 64,
        marker={"preparation_sha256": "c" * 64},
    )
    plan.payload_path.write_bytes(b"payload")
    context = dict(runner.CONTEXT)
    observation_error = ConnectionError("provider observation unavailable")
    stopped = []
    spawned = []

    class FakeCall:
        object_id = "fc-original"

        def get(self):
            raise observation_error

    class FakeFunction:
        def spawn(self, *_args):
            spawned.append(1)
            return FakeCall()

    class FakeApp:
        app_id = "ap-original"

        def __init__(self, *_args, **_kwargs):
            pass

        def function(self, **_kwargs):
            return lambda _function: FakeFunction()

        def run(self, **_kwargs):
            class SuppressingRun:
                def __enter__(self):
                    return self

                @staticmethod
                def __exit__(_self, _type, error, _traceback):
                    return isinstance(error, ConnectionError)

            return SuppressingRun()

    fake_modal = SimpleNamespace(
        App=FakeApp,
        enable_output=nullcontext,
        exception=SimpleNamespace(ConnectionError=ConnectionError),
    )
    monkeypatch.setitem(__import__("sys").modules, "modal", fake_modal)
    monkeypatch.setattr(runner, "_build_image", lambda *_args: object())
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _name: "1.6.1")
    monkeypatch.setattr(
        runner, "_validate_clearance", lambda *_args, **_kwargs: {"source_commit": "f" * 40}
    )
    monkeypatch.setattr(runner.remote, "validate_launch_environment", lambda _context: None)
    monkeypatch.setattr(runner.remote, "verify_modal_identity", lambda **_context: None)
    monkeypatch.setattr(
        runner, "_stop_and_confirm", lambda *args, **kwargs: stopped.append(args[0])
    )

    assert (
        runner._launch(
            plan,
            context=context,
            project_root=tmp_path,
            clearance_path=tmp_path / "clearance.json",
        )
        == 1
    )

    marker = json.loads((tmp_path / runner.ATTEMPT_NAME).read_text())
    failure = json.loads((tmp_path / runner.FAILURE_NAME).read_text())
    assert marker["status"] == "observation_unknown"
    assert marker["app_id"] == "ap-original"
    assert marker["call_id"] == "fc-original"
    assert marker["shutdown_status"] == "pending"
    assert not (tmp_path / runner.SHUTDOWN_OBSERVATION_NAME).exists()
    assert failure["worker_failure_confirmed"] is False
    assert spawned == [1]
    assert stopped == []


def _fixture_plan(tmp_path):
    plan = runner.RealPhotoPlan(
        output_dir=tmp_path,
        payload_path=tmp_path / "payload.zip",
        run_id="fixture",
        payload_bytes=7,
        payload_sha256="a" * 64,
        plan_sha256="b" * 64,
        marker={"preparation_sha256": "c" * 64},
    )
    plan.payload_path.write_bytes(b"payload")
    return plan


@pytest.mark.parametrize(
    "missing_field",
    ["modal_sdk", "diffsynth_commit", "runtime_pins", "model_components"],
)
def test_legacy_clearance_keeps_all_frozen_runtime_pins_required(
    tmp_path, monkeypatch, missing_field
) -> None:
    plan = _fixture_plan(tmp_path)
    payload_bytes = plan.payload_path.read_bytes()
    plan = replace(
        plan,
        payload_bytes=len(payload_bytes),
        payload_sha256=runner.hashlib.sha256(payload_bytes).hexdigest(),
    )
    marker_fields = {
        "preparation_sha256": "c" * 64,
        "geometry_review_sha256": "d" * 64,
        "geometry_review_status": "not_present",
        "active_intake_sha256": "e" * 64,
        "label_confirmation_sha256": "f" * 64,
        "private_source_review_sha256": "a" * 64,
        "quality_review_status": "pending_human_geometry_review",
    }
    plan.marker.update(marker_fields)
    marker_path = plan.output_dir / runner.runner.RUN_MARKER_NAME
    marker_path.write_bytes(b"legacy marker fixture")
    context = dict(runner.CONTEXT)
    now = datetime.now(UTC)
    source_commit = "b" * 40
    clearance = {
        "schema_version": 1,
        "run_id": plan.run_id,
        "status": "approved",
        "review_status": "approved",
        "ci_status": "success",
        "payload_sha256": plan.payload_sha256,
        "plan_sha256": runner.hashlib.sha256(marker_path.read_bytes()).hexdigest(),
        **marker_fields,
        "case_ids": list(runner.runner.CASE_IDS),
        "adapter_archive_sha256": runner.runner.ADAPTER_ARCHIVE_SHA256,
        "adapter_member_sha256": runner.runner.ADAPTER_MEMBER_SHA256,
        "generation_settings": runner.runner.GENERATION_SETTINGS,
        "schedule": list(runner.runner.generation_schedule()),
        "gpu": runner.GPU,
        "cores": runner.CORES,
        "memory_gib": runner.MEMORY_GIB,
        "startup_timeout_seconds": runner.REMOTE_STARTUP_TIMEOUT_SECONDS,
        "timeout_seconds": runner.REMOTE_TIMEOUT_SECONDS,
        "max_calls": 1,
        "retries": 0,
        "allowance_usd": runner.ALLOWANCE_USD,
        "modal_sdk": "1.6.1",
        "diffsynth_commit": runner.remote.DIFFSYNTH_COMMIT,
        "runtime_pins": runner.remote.RUNTIME_PINS,
        "model_components": runner.remote.MODEL_COMPONENTS,
        **context,
        "source_commit": source_commit,
        "checked_at": now.isoformat(),
    }

    def run_git(_root: Path, *arguments: str) -> str:
        if arguments[0] == "status":
            return ""
        if arguments[0] == "rev-parse":
            return source_commit
        assert arguments == ("verify-commit", "HEAD")
        return ""

    monkeypatch.setattr(runner, "_run_git", run_git)
    clearance_path = tmp_path / "clearance.json"
    clearance_path.write_text(json.dumps(clearance), encoding="utf-8")
    assert (
        runner._validate_clearance(
            plan,
            project_root=tmp_path,
            clearance_path=clearance_path,
            context=context,
            now=now,
        )
        == clearance
    )

    clearance.pop(missing_field)
    clearance_path.write_text(json.dumps(clearance), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the immutable plan"):
        runner._validate_clearance(
            plan,
            project_root=tmp_path,
            clearance_path=clearance_path,
            context=context,
            now=now,
        )


def test_recovery_of_unknown_same_call_preserves_work_and_ids(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    plan = _fixture_plan(tmp_path)
    context = dict(runner.CONTEXT)
    attempt_path = runner._reserve_attempt(plan, "f" * 40, context)
    runner._update_attempt_marker(
        attempt_path,
        status="observation_unknown",
        app_id="ap-original",
        call_id="fc-original",
        spawn_count=1,
    )
    observation_error = RuntimeError("provider still cannot observe call")
    looked_up = []
    stopped = []

    class Call:
        def get(self, *, timeout):
            assert timeout == 0
            raise observation_error

    fake_modal = SimpleNamespace(
        FunctionCall=SimpleNamespace(from_id=lambda call_id: looked_up.append(call_id) or Call())
    )
    monkeypatch.setitem(__import__("sys").modules, "modal", fake_modal)
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _name: "1.6.1")
    monkeypatch.setattr(runner.remote, "validate_launch_environment", lambda _context: None)
    monkeypatch.setattr(runner.remote, "verify_modal_identity", lambda **_context: None)
    monkeypatch.setattr(
        runner, "_stop_and_confirm", lambda app_id, **_kwargs: stopped.append(app_id)
    )

    assert runner._recover(plan, context=context) == 1
    marker = json.loads(attempt_path.read_text())
    assert looked_up == ["fc-original"]
    assert stopped == []
    assert marker["status"] == "observation_unknown"
    assert marker["app_id"] == "ap-original"
    assert marker["call_id"] == "fc-original"
    assert marker["shutdown_status"] == "pending"


def test_recovery_timeout_preserves_original_unknown_receipt(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    class FunctionTimeoutError(Exception):
        pass

    plan = _fixture_plan(tmp_path)
    context = dict(runner.CONTEXT)
    attempt_path = runner._reserve_attempt(plan, "f" * 40, context)
    runner._update_attempt_marker(
        attempt_path,
        status="observation_unknown",
        app_id="ap-original",
        call_id="fc-original",
        spawn_count=1,
    )
    original = runner._failure_receipt(
        plan.run_id,
        "modal_observation",
        RuntimeError("provider observation unavailable"),
        app_id="ap-original",
        call_id="fc-original",
        status="observation_unknown",
    )
    runner._write_once(tmp_path / runner.FAILURE_NAME, original)
    original_bytes = (tmp_path / runner.FAILURE_NAME).read_bytes()
    shutdown_commands = []

    class Call:
        @staticmethod
        def get(*, timeout):
            assert timeout == 0
            raise FunctionTimeoutError("function execution timed out")

    fake_modal = SimpleNamespace(
        exception=SimpleNamespace(FunctionTimeoutError=FunctionTimeoutError),
        FunctionCall=SimpleNamespace(from_id=lambda _call_id: Call()),
    )
    monkeypatch.setitem(__import__("sys").modules, "modal", fake_modal)
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _name: "1.6.1")
    monkeypatch.setattr(runner.remote, "validate_launch_environment", lambda _context: None)
    monkeypatch.setattr(runner.remote, "verify_modal_identity", lambda **_context: None)

    def shutdown_run(arguments, **kwargs):
        shutdown_commands.append((arguments, kwargs))
        if "stop" in arguments:
            return _completed(0)
        return _completed(0, json.dumps([_app_record(app_id="ap-original")]))

    monkeypatch.setattr(runner.subprocess, "run", shutdown_run)

    assert runner._recover(plan, context=context) == 1
    marker = json.loads(attempt_path.read_text())
    recovery = json.loads((tmp_path / runner.RECOVERY_FAILURE_NAME).read_text())
    assert (tmp_path / runner.FAILURE_NAME).read_bytes() == original_bytes
    assert recovery["status"] == "worker_failed"
    assert recovery["worker_failure_confirmed"] is True
    assert marker["worker_status"] == "failed"
    assert marker["worker_failure_confirmed"] is True
    assert marker["shutdown_status"] == "stopped"
    assert len(shutdown_commands) == 2
    assert json.loads((tmp_path / runner.SHUTDOWN_OBSERVATION_NAME).read_text())["status"] == (
        "stopped"
    )


@pytest.mark.parametrize(
    ("failure_stage", "worker_failure_confirmed"),
    [("result_validation", False), ("modal_observation", True)],
)
def test_shutdown_error_recovery_does_not_reobserve_terminal_work(
    tmp_path, monkeypatch, failure_stage, worker_failure_confirmed
) -> None:
    from types import SimpleNamespace

    plan = _fixture_plan(tmp_path)
    context = dict(runner.CONTEXT)
    attempt_path = runner._reserve_attempt(plan, "f" * 40, context)
    runner._update_attempt_marker(
        attempt_path,
        status="shutdown_error",
        app_id="ap-terminal",
        call_id="fc-terminal",
        spawn_count=1,
        failure_stage=failure_stage,
        worker_failure_confirmed=worker_failure_confirmed,
    )
    stopped = []
    looked_up = []
    monkeypatch.setitem(
        __import__("sys").modules,
        "modal",
        SimpleNamespace(
            FunctionCall=SimpleNamespace(from_id=lambda call_id: looked_up.append(call_id))
        ),
    )
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _name: "1.6.1")
    monkeypatch.setattr(runner.remote, "validate_launch_environment", lambda _context: None)
    monkeypatch.setattr(runner.remote, "verify_modal_identity", lambda **_context: None)
    monkeypatch.setattr(
        runner, "_stop_and_confirm", lambda app_id, **_kwargs: stopped.append(app_id)
    )

    assert runner._recover(plan, context=context) == 1
    marker = json.loads(attempt_path.read_text())
    assert marker["shutdown_status"] == "stopped"
    assert marker["status"] == ("worker_failed" if worker_failure_confirmed else "modal_error")
    assert stopped == ["ap-terminal"]
    assert looked_up == []


def test_unverified_result_and_shutdown_failures_are_both_preserved(tmp_path, monkeypatch) -> None:
    plan = _fixture_plan(tmp_path)
    attempt_path = runner._reserve_attempt(plan, "f" * 40, dict(runner.CONTEXT))
    runner._update_attempt_marker(
        attempt_path,
        status="observing",
        app_id="ap-original",
        call_id="fc-original",
        spawn_count=1,
    )
    monkeypatch.setattr(
        runner,
        "_validate_clearance",
        lambda *_args, **_kwargs: {"source_commit": "f" * 40},
    )
    monkeypatch.setattr(
        runner, "_preserve_unverified_result", lambda _plan, _result: {"file": "unverified.bin"}
    )
    monkeypatch.setattr(
        runner.runner,
        "validate_result_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("payload binding mismatch")),
    )

    assert (
        runner._finish_result(
            plan,
            attempt_path,
            b"untrusted bytes",
            app_id="ap-original",
            call_id="fc-original",
            receipt_path=tmp_path / runner.FAILURE_NAME,
        )
        == 1
    )
    prior_failure_bytes = (tmp_path / runner.FAILURE_NAME).read_bytes()
    failure = json.loads(prior_failure_bytes)
    assert failure["worker_failure_confirmed"] is False
    assert failure["unverified_result"] == {"file": "unverified.bin"}

    runner._record_shutdown_failure(
        plan,
        attempt_path,
        RuntimeError("stop API unavailable"),
        app_id="ap-original",
        call_id="fc-original",
    )
    assert (tmp_path / runner.FAILURE_NAME).read_bytes() == prior_failure_bytes
    shutdown_path = tmp_path / runner.SHUTDOWN_FAILURE_NAME
    shutdown = json.loads(shutdown_path.read_text())
    assert shutdown["failure"]["stage"] == "modal_app_stop"
    assert shutdown["failure"]["message"] == "stop API unavailable"
    marker = json.loads(attempt_path.read_text())
    assert marker["shutdown_status"] == "error"


def test_ambiguous_spawn_preserves_unknown_work(tmp_path, monkeypatch) -> None:
    from contextlib import nullcontext
    from types import SimpleNamespace

    class ConnectionError(Exception):
        pass

    plan = _fixture_plan(tmp_path)
    context = dict(runner.CONTEXT)
    spawned = []
    stopped = []
    looked_up = []

    class FakeFunction:
        def spawn(self, *_args):
            spawned.append(1)
            raise ConnectionError("spawn response unavailable")

    class FakeApp:
        app_id = "ap-ambiguous"

        def __init__(self, *_args, **_kwargs):
            pass

        def function(self, **_kwargs):
            return lambda _function: FakeFunction()

        def run(self, **_kwargs):
            class SuppressingRun:
                def __enter__(self):
                    return self

                @staticmethod
                def __exit__(_self, _type, error, _traceback):
                    return isinstance(error, ConnectionError)

            return SuppressingRun()

    fake_modal = SimpleNamespace(
        App=FakeApp,
        enable_output=nullcontext,
        exception=SimpleNamespace(ConnectionError=ConnectionError),
        FunctionCall=SimpleNamespace(from_id=lambda call_id: looked_up.append(call_id)),
    )
    monkeypatch.setitem(__import__("sys").modules, "modal", fake_modal)
    monkeypatch.setattr(runner, "_build_image", lambda *_args: object())
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _name: "1.6.1")
    monkeypatch.setattr(
        runner, "_validate_clearance", lambda *_args, **_kwargs: {"source_commit": "f" * 40}
    )
    monkeypatch.setattr(runner.remote, "validate_launch_environment", lambda _context: None)
    monkeypatch.setattr(runner.remote, "verify_modal_identity", lambda **_context: None)
    monkeypatch.setattr(
        runner, "_stop_and_confirm", lambda app_id, **_kwargs: stopped.append(app_id)
    )

    assert (
        runner._launch(
            plan,
            context=context,
            project_root=tmp_path,
            clearance_path=tmp_path / "clearance.json",
        )
        == 1
    )
    marker = json.loads((tmp_path / runner.ATTEMPT_NAME).read_text())
    failure = json.loads((tmp_path / runner.FAILURE_NAME).read_text())
    assert marker["status"] == "observation_unknown"
    assert marker["app_id"] == "ap-ambiguous"
    assert marker["call_id"] is None
    assert failure["worker_failure_confirmed"] is False
    assert failure["same_call_recovery"]
    assert spawned == [1]
    assert stopped == []

    with pytest.raises(FileExistsError):
        runner._launch(
            plan,
            context=context,
            project_root=tmp_path,
            clearance_path=tmp_path / "clearance.json",
        )
    assert spawned == [1]
    assert runner._recover(plan, context=context) == 1
    marker = json.loads((tmp_path / runner.ATTEMPT_NAME).read_text())
    assert marker["status"] == "observation_unknown"
    assert marker["shutdown_status"] == "pending"
    assert marker["call_id"] is None
    assert stopped == []
    assert looked_up == []
    assert not (tmp_path / runner.SHUTDOWN_OBSERVATION_NAME).exists()


def test_non_byte_result_is_recorded_and_shutdown_is_confirmed(tmp_path, monkeypatch) -> None:
    from contextlib import nullcontext
    from types import SimpleNamespace

    plan = _fixture_plan(tmp_path)
    context = dict(runner.CONTEXT)
    shutdown_commands = []

    class FakeCall:
        object_id = "fc-nonbyte"

        @staticmethod
        def get():
            return {"not": "bytes"}

    class FakeFunction:
        @staticmethod
        def spawn(*_args):
            return FakeCall()

    class FakeApp:
        app_id = "ap-nonbyte"

        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def function(**_kwargs):
            return lambda _function: FakeFunction()

        @staticmethod
        def run(**_kwargs):
            return nullcontext()

    monkeypatch.setitem(
        __import__("sys").modules,
        "modal",
        SimpleNamespace(App=FakeApp, enable_output=nullcontext),
    )
    monkeypatch.setattr(runner, "_build_image", lambda *_args: object())
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _name: "1.6.1")
    monkeypatch.setattr(
        runner, "_validate_clearance", lambda *_args, **_kwargs: {"source_commit": "f" * 40}
    )
    monkeypatch.setattr(runner.remote, "validate_launch_environment", lambda _context: None)
    monkeypatch.setattr(runner.remote, "verify_modal_identity", lambda **_context: None)

    def shutdown_run(arguments, **kwargs):
        shutdown_commands.append((arguments, kwargs))
        if "stop" in arguments:
            return _completed(0)
        return _completed(0, json.dumps([_app_record(app_id="ap-nonbyte")]))

    monkeypatch.setattr(runner.subprocess, "run", shutdown_run)

    assert (
        runner._launch(
            plan,
            context=context,
            project_root=tmp_path,
            clearance_path=tmp_path / "clearance.json",
        )
        == 1
    )
    failure = json.loads((tmp_path / runner.FAILURE_NAME).read_text())
    marker = json.loads((tmp_path / runner.ATTEMPT_NAME).read_text())
    assert failure["worker_failure_confirmed"] is False
    assert failure["failure"]["stage"] == "result_validation"
    assert marker["shutdown_status"] == "stopped"
    assert len(shutdown_commands) == 2
    evidence = json.loads((tmp_path / runner.SHUTDOWN_OBSERVATION_NAME).read_text())
    assert evidence["app_id"] == "ap-nonbyte"
    assert evidence["status"] == "stopped"


def test_app_id_survives_app_run_context_startup_error(tmp_path, monkeypatch) -> None:
    from contextlib import nullcontext
    from types import SimpleNamespace

    plan = _fixture_plan(tmp_path)
    context = dict(runner.CONTEXT)
    stopped = []

    class FailingRun:
        def __init__(self, app):
            self.app = app

        def __enter__(self):
            self.app.app_id = "ap-startup"
            raise RuntimeError("app run context failed")

        @staticmethod
        def __exit__(*_args):
            return False

    class FakeApp:
        def __init__(self, *_args, **_kwargs):
            self.app_id = None

        @staticmethod
        def function(**_kwargs):
            return lambda _function: SimpleNamespace(spawn=lambda *_args: None)

        def run(self, **_kwargs):
            return FailingRun(self)

    monkeypatch.setitem(
        __import__("sys").modules,
        "modal",
        SimpleNamespace(App=FakeApp, enable_output=nullcontext),
    )
    monkeypatch.setattr(runner, "_build_image", lambda *_args: object())
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _name: "1.6.1")
    monkeypatch.setattr(
        runner, "_validate_clearance", lambda *_args, **_kwargs: {"source_commit": "f" * 40}
    )
    monkeypatch.setattr(runner.remote, "validate_launch_environment", lambda _context: None)
    monkeypatch.setattr(runner.remote, "verify_modal_identity", lambda **_context: None)
    monkeypatch.setattr(
        runner, "_stop_and_confirm", lambda app_id, **_kwargs: stopped.append(app_id)
    )

    assert (
        runner._launch(
            plan,
            context=context,
            project_root=tmp_path,
            clearance_path=tmp_path / "clearance.json",
        )
        == 1
    )
    failure = json.loads((tmp_path / runner.FAILURE_NAME).read_text())
    assert failure["app_id"] == "ap-startup"
    assert failure["worker_failure_confirmed"] is False
    assert failure["failure"]["stage"] == "modal_app_startup"
    assert failure["failure"]["message"] == "app run context failed"
    assert stopped == ["ap-startup"]


def test_only_modal_function_timeout_confirms_worker_failure(tmp_path, monkeypatch) -> None:
    from contextlib import nullcontext
    from types import SimpleNamespace

    class FunctionTimeoutError(Exception):
        pass

    plan = _fixture_plan(tmp_path)
    context = dict(runner.CONTEXT)
    stopped = []

    class FakeCall:
        object_id = "fc-terminal-timeout"

        @staticmethod
        def get():
            raise FunctionTimeoutError("function execution timed out")

    class FakeFunction:
        @staticmethod
        def spawn(*_args):
            return FakeCall()

    class FakeApp:
        app_id = "ap-terminal-timeout"

        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def function(**_kwargs):
            return lambda _function: FakeFunction()

        @staticmethod
        def run(**_kwargs):
            return nullcontext()

    fake_modal = SimpleNamespace(
        App=FakeApp,
        enable_output=nullcontext,
        exception=SimpleNamespace(FunctionTimeoutError=FunctionTimeoutError),
    )
    monkeypatch.setitem(__import__("sys").modules, "modal", fake_modal)
    monkeypatch.setattr(runner, "_build_image", lambda *_args: object())
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _name: "1.6.1")
    monkeypatch.setattr(
        runner, "_validate_clearance", lambda *_args, **_kwargs: {"source_commit": "f" * 40}
    )
    monkeypatch.setattr(runner.remote, "validate_launch_environment", lambda _context: None)
    monkeypatch.setattr(runner.remote, "verify_modal_identity", lambda **_context: None)
    monkeypatch.setattr(
        runner, "_stop_and_confirm", lambda app_id, **_kwargs: stopped.append(app_id)
    )

    assert (
        runner._launch(
            plan,
            context=context,
            project_root=tmp_path,
            clearance_path=tmp_path / "clearance.json",
        )
        == 1
    )
    failure = json.loads((tmp_path / runner.FAILURE_NAME).read_text())
    marker = json.loads((tmp_path / runner.ATTEMPT_NAME).read_text())
    assert failure["worker_failure_confirmed"] is True
    assert marker["status"] == "worker_failed"
    assert marker["shutdown_status"] == "stopped"
    assert stopped == ["ap-terminal-timeout"]


def test_parser_exposes_explicit_preservation_pair_inputs() -> None:
    args = runner._parser().parse_args(
        [
            "--comparison-profile",
            "preservation-pair",
            "--run-id",
            "paired-run",
            "--plan-dir",
            "artifacts/paired-run",
            "--legacy-input",
            "legacy.zip",
            "--prior-adapter-archive",
            "prior.zip",
            "--new-adapter-archive",
            "new.zip",
            "--confirmation",
            "confirmation.json",
        ]
    )
    assert args.comparison_profile == "preservation-pair"
    assert args.run_id == "paired-run"
    assert str(args.plan_dir) == "artifacts/paired-run"
    assert str(args.legacy_input) == "legacy.zip"
    assert str(args.prior_adapter_archive) == "prior.zip"
    assert str(args.new_adapter_archive) == "new.zip"
    assert str(args.confirmation) == "confirmation.json"


def test_pair_profile_rejects_missing_explicit_plan_inputs_before_io() -> None:
    with pytest.raises(ValueError, match="preservation-pair requires explicit"):
        runner.main(["--comparison-profile", "preservation-pair"])


def _pair_fixture_plan(output_dir):
    case_ids = ("OFF-R1-003", "OFF-R1-005", "OFF-R1-006")
    schedule = [
        {"case_id": case_id, "arm": arm, "session": session}
        for arm, session in (
            ("base", "base"),
            ("prior", "prior-strict-1"),
            ("prior_reloaded", "prior-strict-2"),
            ("new", "new-strict-1"),
            ("new_reloaded", "new-strict-2"),
        )
        for case_id in case_ids
    ]
    plan = runner.RealPhotoPlan(
        output_dir=output_dir,
        payload_path=output_dir / "payload.zip",
        run_id="paired-fixture",
        payload_bytes=7,
        payload_sha256="a" * 64,
        plan_sha256="b" * 64,
        marker={"profile": "preservation-pair", "schedule": schedule},
    )
    plan.payload_path.write_bytes(b"payload")
    return plan, schedule


def test_pair_recovery_observes_only_the_recorded_call_from_observing_state(
    tmp_path, monkeypatch
) -> None:
    from types import SimpleNamespace

    plan, schedule = _pair_fixture_plan(tmp_path)
    context = dict(runner.CONTEXT)
    attempt_path = runner._reserve_attempt(plan, "f" * 40, context)
    runner._update_attempt_marker(
        attempt_path,
        status="observing",
        app_id="ap-pair-original",
        call_id="fc-pair-original",
        spawn_count=1,
    )
    looked_up = []
    finished = []
    stopped = []

    class Call:
        def get(self, *, timeout):
            assert timeout == 0
            return b"same-call-result"

    fake_modal = SimpleNamespace(
        FunctionCall=SimpleNamespace(from_id=lambda call_id: looked_up.append(call_id) or Call())
    )
    monkeypatch.setitem(__import__("sys").modules, "modal", fake_modal)
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _name: "1.6.1")
    monkeypatch.setattr(runner.remote, "validate_launch_environment", lambda _context: None)
    monkeypatch.setattr(runner.remote, "verify_modal_identity", lambda **_context: None)

    def finish(_plan, marker_path, result, *, app_id, call_id, receipt_path):
        finished.append((result, app_id, call_id, receipt_path))
        runner._update_attempt_marker(
            marker_path,
            status="worker_returned",
            worker_status="passed",
            app_id=app_id,
            call_id=call_id,
        )
        return 0

    monkeypatch.setattr(runner, "_finish_result", finish)
    monkeypatch.setattr(
        runner, "_stop_and_confirm", lambda app_id, **_kwargs: stopped.append(app_id)
    )

    assert runner._recover(plan, context=context) == 0

    marker = json.loads(attempt_path.read_text())
    assert looked_up == ["fc-pair-original"]
    assert finished[0][:3] == (b"same-call-result", "ap-pair-original", "fc-pair-original")
    assert stopped == ["ap-pair-original"]
    assert marker["status"] == "worker_returned"
    assert marker["app_id"] == "ap-pair-original"
    assert marker["call_id"] == "fc-pair-original"
    assert marker["comparison_profile"] == "preservation-pair"
    assert marker["payload_sha256"] == plan.payload_sha256
    assert marker["plan_sha256"] == plan.plan_sha256
    assert marker["source_commit"] == "f" * 40
    assert marker["timeout_seconds"] == 1800
    assert marker["schedule"] == schedule
    assert all(marker[key] == value for key, value in context.items())
    with pytest.raises(ValueError, match="immutable"):
        runner._update_attempt_marker(attempt_path, call_id="fc-not-the-original")


def test_pair_recovery_observation_error_stays_unknown_and_never_spawns_again(
    tmp_path, monkeypatch
) -> None:
    from types import SimpleNamespace

    plan, _schedule = _pair_fixture_plan(tmp_path)
    context = dict(runner.CONTEXT)
    attempt_path = runner._reserve_attempt(plan, "f" * 40, context)
    runner._update_attempt_marker(
        attempt_path,
        status="observing",
        app_id="ap-pair-original",
        call_id="fc-pair-original",
        spawn_count=1,
    )
    looked_up = []
    stopped = []
    observation_error = RuntimeError("provider observation unavailable")

    class Call:
        def get(self, *, timeout):
            assert timeout == 0
            raise observation_error

    fake_modal = SimpleNamespace(
        FunctionCall=SimpleNamespace(from_id=lambda call_id: looked_up.append(call_id) or Call())
    )
    monkeypatch.setitem(__import__("sys").modules, "modal", fake_modal)
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _name: "1.6.1")
    monkeypatch.setattr(runner.remote, "validate_launch_environment", lambda _context: None)
    monkeypatch.setattr(runner.remote, "verify_modal_identity", lambda **_context: None)
    monkeypatch.setattr(
        runner, "_stop_and_confirm", lambda app_id, **_kwargs: stopped.append(app_id)
    )

    assert runner._recover(plan, context=context) == 1
    marker = json.loads(attempt_path.read_text())
    assert looked_up == ["fc-pair-original"]
    assert stopped == []
    assert marker["status"] == "observation_unknown"
    assert marker["worker_failure_confirmed"] is False
    assert marker["app_id"] == "ap-pair-original"
    assert marker["call_id"] == "fc-pair-original"
    assert marker["shutdown_status"] == "pending"


@pytest.mark.parametrize("action", ["--launch", "--recover"])
def test_requested_pair_profile_must_match_plan_marker_before_action(
    tmp_path, monkeypatch, action
) -> None:
    mismatched_plan = runner.RealPhotoPlan(
        output_dir=tmp_path,
        payload_path=tmp_path / "payload.zip",
        run_id="paired-fixture",
        payload_bytes=0,
        payload_sha256="a" * 64,
        plan_sha256="b" * 64,
        marker={"profile": "legacy"},
    )
    monkeypatch.setattr(runner, "_load_plan", lambda _output: mismatched_plan)
    monkeypatch.setattr(runner, "_launch", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(runner, "_recover", lambda *_args, **_kwargs: 0)
    with pytest.raises(ValueError, match="requested comparison profile"):
        runner.main(
            [
                "--comparison-profile",
                "preservation-pair",
                "--run-id",
                "paired-fixture",
                "--plan-dir",
                str(tmp_path),
                "--legacy-input",
                "legacy.zip",
                "--prior-adapter-archive",
                "prior.zip",
                "--new-adapter-archive",
                "new.zip",
                "--confirmation",
                "confirmation.json",
                action,
                "--profile",
                runner.remote.PROFILE,
                "--workspace",
                runner.remote.WORKSPACE,
                "--environment",
                runner.remote.ENVIRONMENT,
            ]
        )


def test_pair_plan_creation_uses_comparison_builder_and_explicit_sources(tmp_path, monkeypatch):
    from product_scene_editor import real_photo_comparison as comparison

    plan_dir = tmp_path / "paired-plan"
    sources = {
        "legacy_input": tmp_path / "legacy.zip",
        "prior_adapter_archive": tmp_path / "prior.zip",
        "new_adapter_archive": tmp_path / "new.zip",
        "confirmation": tmp_path / "confirmation.json",
    }
    calls = []
    record = {"profile": "preservation-pair", "run_id": "paired-fixture"}

    def create(root, output, *, run_id, **kwargs):
        calls.append((root, output, run_id, kwargs))
        return record

    monkeypatch.setattr(comparison, "create_comparison_plan", create, raising=False)
    monkeypatch.setattr(
        runner.runner,
        "create_real_photo_plan",
        lambda *_args, **_kwargs: pytest.fail("paired profile used the legacy plan builder"),
    )

    assert (
        runner.main(
            [
                "--comparison-profile",
                "preservation-pair",
                "--run-id",
                "paired-fixture",
                "--plan-dir",
                str(plan_dir),
                "--legacy-input",
                str(sources["legacy_input"]),
                "--prior-adapter-archive",
                str(sources["prior_adapter_archive"]),
                "--new-adapter-archive",
                str(sources["new_adapter_archive"]),
                "--confirmation",
                str(sources["confirmation"]),
            ]
        )
        == 0
    )
    assert calls == [
        (
            Path(runner.__file__).resolve().parents[1],
            plan_dir,
            "paired-fixture",
            {
                "legacy_input": sources["legacy_input"],
                "prior_archive": sources["prior_adapter_archive"],
                "new_archive": sources["new_adapter_archive"],
                "confirmation": sources["confirmation"],
            },
        )
    ]


def test_pair_worker_dispatches_payload_to_paired_inference(monkeypatch) -> None:
    import sys
    from types import ModuleType

    import product_scene_editor

    calls = []
    fake_module = ModuleType("product_scene_editor.real_photo_comparison_inference")

    def infer(payload, run_id):
        calls.append((payload, run_id))
        return b"paired-result"

    fake_module.run_remote_real_photo_comparison = infer
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    monkeypatch.setattr(
        product_scene_editor, "real_photo_comparison_inference", fake_module, raising=False
    )

    assert runner._remote_comparison_worker(b"paired-payload", "paired-run") == b"paired-result"
    assert calls == [(b"paired-payload", "paired-run")]
