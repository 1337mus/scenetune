from __future__ import annotations

import json

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
    stopped = []

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
    monkeypatch.setattr(
        runner, "_stop_and_confirm", lambda app_id, **_kwargs: stopped.append(app_id)
    )

    assert runner._recover(plan, context=context) == 1
    marker = json.loads(attempt_path.read_text())
    recovery = json.loads((tmp_path / runner.RECOVERY_FAILURE_NAME).read_text())
    assert (tmp_path / runner.FAILURE_NAME).read_bytes() == original_bytes
    assert recovery["status"] == "worker_failed"
    assert recovery["worker_failure_confirmed"] is True
    assert marker["worker_status"] == "failed"
    assert marker["worker_failure_confirmed"] is True
    assert marker["shutdown_status"] == "stopped"
    assert stopped == ["ap-original"]


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


def test_non_byte_result_is_recorded_and_shutdown_is_confirmed(tmp_path, monkeypatch) -> None:
    from contextlib import nullcontext
    from types import SimpleNamespace

    plan = _fixture_plan(tmp_path)
    context = dict(runner.CONTEXT)
    stopped = []

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
    assert failure["worker_failure_confirmed"] is False
    assert failure["failure"]["stage"] == "result_validation"
    assert marker["shutdown_status"] == "stopped"
    assert stopped == ["ap-nonbyte"]


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
