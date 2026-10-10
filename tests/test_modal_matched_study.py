from __future__ import annotations

import hashlib
import io
import json
import runpy
import subprocess
import sys
import zipfile
from datetime import UTC, datetime, timedelta
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import modal_matched_study
from product_scene_editor import matched_study, remote


def test_successful_result_must_contain_exact_29_file_contract() -> None:
    plan = SimpleNamespace(
        run_id="matched-run",
        development_pair_ids=tuple(f"dev-{index}" for index in range(9)),
    )
    expected_names = {
        "receipt.json",
        "adapter/step-144.safetensors",
        *(
            f"images/{pair_id}/{mode}.png"
            for pair_id in plan.development_pair_ids
            for mode in ("base", "trained", "reloaded")
        ),
    }
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr(
            "receipt.json",
            json.dumps({"run_id": plan.run_id, "study_type": "matched144", "status": "passed"}),
        )
        for name in expected_names - {"receipt.json"}:
            archive.writestr(name, b"artifact")

    receipt, names = modal_matched_study._validate_result_bundle(bundle.getvalue(), plan)

    assert receipt["study_type"] == "matched144"
    assert len(names) == 29
    assert names == expected_names

    with zipfile.ZipFile(bundle) as original:
        missing_one = io.BytesIO()
        with zipfile.ZipFile(missing_one, "w") as altered:
            for name in expected_names - {"images/dev-0/base.png"}:
                altered.writestr(name, original.read(name))
    with pytest.raises(ValueError, match="artifact set"):
        modal_matched_study._validate_result_bundle(missing_one.getvalue(), plan)


def test_attempt_marker_is_separate_and_binds_the_matched_plan(tmp_path: Path) -> None:
    output_dir = tmp_path / "matched-run"
    output_dir.mkdir()
    payload_path = output_dir / matched_study.PAYLOAD_NAME
    payload_path.write_bytes(b"approved-payload")
    plan_path = output_dir / matched_study.RUN_MARKER_NAME
    plan_path.write_bytes(b"immutable-plan-marker")
    payload = payload_path.read_bytes()
    plan = matched_study.MatchedStudyPlan(
        output_dir=output_dir,
        payload_path=payload_path,
        run_id="matched-run",
        payload_bytes=len(payload),
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        source_dataset_manifest_sha256=matched_study.APPROVED_DATASET_MANIFEST_SHA256,
        training_pair_ids=tuple(f"train-{index}" for index in range(18)),
        development_pair_ids=tuple(f"dev-{index}" for index in range(9)),
    )
    context = {
        "profile": remote.PROFILE,
        "workspace": remote.WORKSPACE,
        "environment": remote.ENVIRONMENT,
    }

    marker = modal_matched_study._reserve_attempt_marker(
        plan,
        output_dir / modal_matched_study.ATTEMPT_MARKER_NAME,
        source_commit="a" * 40,
        context=context,
    )

    assert modal_matched_study.ATTEMPT_MARKER_NAME != "modal-attempt.json"
    assert marker["run_id"] == plan.run_id
    assert marker["payload_sha256"] == plan.payload_sha256
    assert marker["plan_sha256"] == hashlib.sha256(plan_path.read_bytes()).hexdigest()
    assert marker["development_pair_ids"] == list(plan.development_pair_ids)
    assert marker["status"] == "reserved"
    with pytest.raises(FileExistsError):
        modal_matched_study._reserve_attempt_marker(
            plan,
            output_dir / modal_matched_study.ATTEMPT_MARKER_NAME,
            source_commit="a" * 40,
            context=context,
        )


def test_launch_clearance_must_be_fresh_and_bound_to_exact_matched_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "matched-run"
    output_dir.mkdir()
    payload_path = output_dir / matched_study.PAYLOAD_NAME
    payload_path.write_bytes(b"approved-payload")
    plan_marker = output_dir / matched_study.RUN_MARKER_NAME
    plan_marker.write_bytes(b"immutable-plan-marker")
    payload = payload_path.read_bytes()
    plan = matched_study.MatchedStudyPlan(
        output_dir=output_dir,
        payload_path=payload_path,
        run_id="matched-run",
        payload_bytes=len(payload),
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        source_dataset_manifest_sha256=matched_study.APPROVED_DATASET_MANIFEST_SHA256,
        training_pair_ids=tuple(f"train-{index}" for index in range(18)),
        development_pair_ids=tuple(f"dev-{index}" for index in range(9)),
    )
    context = {
        "profile": remote.PROFILE,
        "workspace": remote.WORKSPACE,
        "environment": remote.ENVIRONMENT,
    }
    now = datetime(2026, 10, 10, 4, 0, tzinfo=UTC)
    commit = "a" * 40
    clearance = {
        "schema_version": 1,
        "run_id": plan.run_id,
        "study_type": "matched144",
        "status": "approved",
        "review_status": "approved",
        "ci_status": "success",
        "source_commit": commit,
        "payload_sha256": plan.payload_sha256,
        "plan_sha256": hashlib.sha256(plan_marker.read_bytes()).hexdigest(),
        "source_dataset_manifest_sha256": plan.source_dataset_manifest_sha256,
        "training_pair_ids": list(plan.training_pair_ids),
        "development_pair_ids": list(plan.development_pair_ids),
        "training_pairs": 18,
        "development_pairs": 9,
        "epochs": 8,
        "training_steps": 144,
        "checked_at": (now - timedelta(minutes=5)).isoformat(),
        "allowance_usd": 15,
        "gpu": "A100-80GB",
        "gpu_count": 1,
        "cores": 4,
        "memory_gib": 64,
        "startup_timeout_seconds": 300,
        "timeout_seconds": 2700,
        "max_calls": 1,
        "retries": 0,
        **context,
    }
    clearance_path = tmp_path / "matched-study-launch-clearance.json"
    clearance_path.write_text(json.dumps(clearance), encoding="utf-8")
    monkeypatch.setattr(
        modal_matched_study,
        "_run_local_git",
        lambda _root, args: subprocess.CompletedProcess(
            args, 0, commit if args[-2:] == ["rev-parse", "HEAD"] else "", ""
        ),
        raising=False,
    )

    validated = modal_matched_study._validate_launch_clearance(
        plan,
        project_root=tmp_path,
        clearance_path=clearance_path,
        context=context,
        now=now,
    )
    assert validated["run_id"] == plan.run_id

    clearance["development_pair_count"] = 8
    clearance_path.write_text(json.dumps(clearance), encoding="utf-8")
    with pytest.raises(ValueError, match="clearance"):
        modal_matched_study._validate_launch_clearance(
            plan,
            project_root=tmp_path,
            clearance_path=clearance_path,
            context=context,
            now=now,
        )


def test_cpu_plan_cli_entrypoint_runs_without_modal_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from test_matched_study import _write_matched_dataset

    repository = Path(__file__).resolve().parents[1]
    dataset_dir = tmp_path / "synthetic-dataset"
    manifest_sha256 = _write_matched_dataset(dataset_dir)
    monkeypatch.setattr(
        matched_study,
        "APPROVED_DATASET_MANIFEST_SHA256",
        manifest_sha256,
        raising=False,
    )
    plan_dir = tmp_path / "cpu-cli-plan"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "modal_matched_study.py",
            "--dataset-dir",
            str(dataset_dir),
            "--plan-dir",
            str(plan_dir),
            "--run-id",
            "cpu-cli",
        ],
    )

    with pytest.raises(SystemExit) as exited:
        runpy.run_path(str(repository / "experiments/modal_matched_study.py"), run_name="__main__")

    output = json.loads(capsys.readouterr().out)
    assert exited.value.code == 0
    assert output["status"] == "planned"
    assert output["run_id"] == "cpu-cli"
    assert output["training_pairs"] == 18
    assert output["development_pairs"] == 9
    assert output["training_steps"] == 144
    assert output["modal_called"] is False
    assert matched_study.load_matched_study_plan(plan_dir).run_id == "cpu-cli"
    assert not (plan_dir / modal_matched_study.ATTEMPT_MARKER_NAME).exists()


def test_invalid_bounded_result_is_preserved_as_unverified_bytes(tmp_path: Path) -> None:
    output_dir = tmp_path / "matched-run"
    output_dir.mkdir()
    payload_path = output_dir / matched_study.PAYLOAD_NAME
    payload_path.write_bytes(b"approved-payload")
    (output_dir / matched_study.RUN_MARKER_NAME).write_bytes(b"immutable-plan-marker")
    payload = payload_path.read_bytes()
    plan = matched_study.MatchedStudyPlan(
        output_dir=output_dir,
        payload_path=payload_path,
        run_id="matched-run",
        payload_bytes=len(payload),
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        source_dataset_manifest_sha256=matched_study.APPROVED_DATASET_MANIFEST_SHA256,
        training_pair_ids=tuple(f"train-{index}" for index in range(18)),
        development_pair_ids=tuple(f"dev-{index}" for index in range(9)),
    )
    context = {
        "profile": remote.PROFILE,
        "workspace": remote.WORKSPACE,
        "environment": remote.ENVIRONMENT,
    }
    attempt_path = output_dir / modal_matched_study.ATTEMPT_MARKER_NAME
    modal_matched_study._reserve_attempt_marker(
        plan, attempt_path, source_commit="a" * 40, context=context
    )
    modal_matched_study._update_attempt_marker(
        attempt_path, status="observing", app_id="ap-evidence", call_id="fc-evidence"
    )
    returned = b"bounded but not a valid result archive"

    modal_matched_study._record_result_validation_failure(
        plan,
        attempt_path,
        output_dir / modal_matched_study.MODAL_FAILURE_NAME,
        returned,
        ValueError("invalid archive"),
        app_id="ap-evidence",
        call_id="fc-evidence",
    )

    evidence_path = output_dir / modal_matched_study.UNVERIFIED_RESULT_NAME
    evidence = json.loads((output_dir / modal_matched_study.MODAL_FAILURE_NAME).read_text())
    marker = json.loads(attempt_path.read_text())
    assert evidence_path.read_bytes() == returned
    assert evidence["status"] == "modal_error"
    assert evidence["worker_failure_confirmed"] is False
    assert evidence["unverified_result"]["file"] == modal_matched_study.UNVERIFIED_RESULT_NAME
    assert evidence["unverified_result"]["bytes"] == len(returned)
    assert evidence["unverified_result"]["sha256"] == hashlib.sha256(returned).hexdigest()
    assert marker["unverified_result"] == evidence["unverified_result"]
    assert (
        modal_matched_study._validate_attempt_binding(plan, attempt_path, context)["call_id"]
        == "fc-evidence"
    )

    modal_matched_study._record_result_validation_failure(
        plan,
        attempt_path,
        output_dir / modal_matched_study.MODAL_FAILURE_NAME,
        b"different unverified bytes",
        ValueError("another invalid archive"),
        app_id="ap-evidence",
        call_id="fc-evidence",
    )
    assert evidence_path.read_bytes() == returned


@pytest.mark.parametrize("worker_status", ["passed", "failed"])
def test_recovery_observes_only_the_recorded_modal_call_id_and_stops_completed_app(
    worker_status: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "matched-run"
    output_dir.mkdir()
    payload_path = output_dir / matched_study.PAYLOAD_NAME
    payload_path.write_bytes(b"approved-payload")
    plan_marker = output_dir / matched_study.RUN_MARKER_NAME
    plan_marker.write_bytes(b"immutable-plan-marker")
    payload = payload_path.read_bytes()
    plan = matched_study.MatchedStudyPlan(
        output_dir=output_dir,
        payload_path=payload_path,
        run_id="matched-run",
        payload_bytes=len(payload),
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        source_dataset_manifest_sha256=matched_study.APPROVED_DATASET_MANIFEST_SHA256,
        training_pair_ids=tuple(f"train-{index}" for index in range(18)),
        development_pair_ids=tuple(f"dev-{index}" for index in range(9)),
    )
    context = {
        "profile": remote.PROFILE,
        "workspace": remote.WORKSPACE,
        "environment": remote.ENVIRONMENT,
    }
    attempt_path = output_dir / modal_matched_study.ATTEMPT_MARKER_NAME
    modal_matched_study._reserve_attempt_marker(
        plan, attempt_path, source_commit="a" * 40, context=context
    )
    modal_matched_study._update_attempt_marker(
        attempt_path, status="observing", app_id="ap-recorded", call_id="fc-recorded"
    )
    result_names = {
        "receipt.json",
        "adapter/step-144.safetensors",
        *(
            f"images/{pair_id}/{mode}.png"
            for pair_id in plan.development_pair_ids
            for mode in ("base", "trained", "reloaded")
        ),
    }
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr(
            "receipt.json",
            json.dumps(
                {"run_id": plan.run_id, "study_type": "matched144", "status": worker_status}
            ),
        )
        for name in result_names - {"receipt.json"}:
            archive.writestr(name, b"artifact")
    observations: list[tuple[str, float | None]] = []
    recovered_ids: list[str] = []
    outcomes: list[BaseException | bytes] = [
        TimeoutError("call is still running"),
        bundle.getvalue(),
    ]

    class FakeFunctionCall:
        @classmethod
        def from_id(cls, call_id: str) -> FakeFunctionCall:
            recovered_ids.append(call_id)
            return cls()

        def get(self, *, timeout: float | None = None) -> bytes:
            observations.append(("get", timeout))
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    fake_modal = type("FakeModal", (), {"FunctionCall": FakeFunctionCall})
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setattr(remote, "verify_modal_identity", lambda **_kwargs: None)
    monkeypatch.setattr(metadata, "version", lambda _name: "1.6.1")
    monkeypatch.delenv("MODAL_PROFILE", raising=False)
    stop_commands: list[tuple[list[str], dict[str, str]]] = []

    def stop_app(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        stop_commands.append((command, kwargs["env"]))  # type: ignore[arg-type]
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(modal_matched_study.subprocess, "run", stop_app)

    pending_status = modal_matched_study._recover(plan, context=context)
    unknown_marker = json.loads(attempt_path.read_text(encoding="utf-8"))
    failure = json.loads((output_dir / modal_matched_study.MODAL_FAILURE_NAME).read_text())
    assert pending_status == 1
    assert unknown_marker["status"] == "observation_unknown"
    assert unknown_marker["call_id"] == "fc-recorded"
    assert failure["worker_failure_confirmed"] is False
    assert failure["call_id"] == "fc-recorded"
    assert stop_commands == []

    status = modal_matched_study._recover(plan, context=context)

    assert status == (0 if worker_status == "passed" else 1)
    assert recovered_ids == ["fc-recorded", "fc-recorded"]
    assert observations == [("get", 0), ("get", 0)]
    marker = json.loads(attempt_path.read_text(encoding="utf-8"))
    assert marker["call_id"] == "fc-recorded"
    assert marker["status"] == "worker_returned"
    assert marker["worker_status"] == worker_status
    assert marker["shutdown_status"] == "stopped"
    assert (output_dir / modal_matched_study.RESULT_NAME).is_file()
    assert len(stop_commands) == 1
    stop_command, stop_environment = stop_commands[0]
    assert stop_command[-1] == "ap-recorded"
    assert stop_command[2:5] == ["modal", "app", "stop"]
    assert stop_environment["MODAL_PROFILE"] == remote.PROFILE
    assert failure["worker_failure_confirmed"] is False


def test_stop_detached_app_confirms_already_stopped_from_exact_app_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = {"profile": remote.PROFILE, "environment": "main"}
    other_app = {
        "app_id": "ap-other",
        "description": "other app",
        "state": "deployed",
        "tasks": "2",
        "created_at": "2026-10-10 00:00:00",
        "stopped_at": None,
    }
    stopped_app = {
        "app_id": "ap-stopped",
        "description": "matched study",
        "state": "stopped",
        "tasks": "0",
        "created_at": "2026-10-10 00:00:00",
        "stopped_at": "2026-10-10 00:01:00",
    }
    responses = [
        subprocess.CompletedProcess([], 1, "", "unclassified stop error"),
        subprocess.CompletedProcess(
            [],
            0,
            json.dumps([other_app, stopped_app]),
            "",
        ),
    ]
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return responses.pop(0)

    monkeypatch.setattr(modal_matched_study.subprocess, "run", run)

    modal_matched_study._stop_detached_app("ap-stopped", context=context)

    assert len(calls) == 2
    stop_command, stop_kwargs = calls[0]
    list_command, list_kwargs = calls[1]
    assert stop_command[2:5] == ["modal", "app", "stop"]
    assert list_command == [
        sys.executable,
        "-m",
        "modal",
        "app",
        "list",
        "--json",
        "--env",
        "main",
    ]
    assert stop_kwargs["env"]["MODAL_PROFILE"] == remote.PROFILE  # type: ignore[index]
    assert list_kwargs["env"]["MODAL_PROFILE"] == remote.PROFILE  # type: ignore[index]
    assert stop_kwargs["timeout"] == 60
    assert list_kwargs["timeout"] == 30


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "duplicate",
        "running",
        "nonzero_tasks",
        "malformed_row",
        "malformed_json",
        "malformed_root",
        "observation_exit",
        "observation_timeout",
    ],
)
def test_stop_detached_app_rejects_unconfirmed_list_observations(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = {"profile": remote.PROFILE, "environment": "main"}
    stopped_app: dict[str, str | None] = {
        "app_id": "ap-stopped",
        "description": "matched study",
        "state": "stopped",
        "tasks": "0",
        "created_at": "2026-10-10 00:00:00",
        "stopped_at": "2026-10-10 00:01:00",
    }
    rows: list[dict[str, str | None]] = [stopped_app]
    list_output: str | None = None
    list_returncode = 0
    if case == "missing":
        rows = []
    elif case == "duplicate":
        rows = [stopped_app, stopped_app.copy()]
    elif case == "running":
        stopped_app["state"] = "deployed"
    elif case == "nonzero_tasks":
        stopped_app["tasks"] = "1"
    elif case == "malformed_row":
        del stopped_app["stopped_at"]
    elif case == "malformed_json":
        list_output = "{"
    elif case == "malformed_root":
        list_output = json.dumps({"apps": []})
    elif case == "observation_exit":
        list_returncode = 1
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        if len(calls) == 1:
            return subprocess.CompletedProcess(command, 1, "", "stop did not confirm")
        if case == "observation_timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        output = list_output if list_output is not None else json.dumps(rows)
        return subprocess.CompletedProcess(command, list_returncode, output, "")

    monkeypatch.setattr(modal_matched_study.subprocess, "run", run)

    error_message = "could not stop" if case == "observation_timeout" else "Modal did not confirm"
    with pytest.raises(RuntimeError, match=error_message):
        modal_matched_study._stop_detached_app("ap-stopped", context=context)

    assert len(calls) == 2
    list_command, list_kwargs = calls[1]
    assert list_command[2:5] == ["modal", "app", "list"]
    assert list_kwargs["timeout"] == 30
    assert list_kwargs["env"]["MODAL_PROFILE"] == remote.PROFILE  # type: ignore[index]
