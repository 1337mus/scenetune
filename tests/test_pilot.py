from __future__ import annotations

import contextlib
import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import UTC, datetime, timedelta
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

from product_scene_editor import dataset, pilot, remote


def _dataset(path: Path) -> dict[str, object]:
    return dataset.build_dataset(path, seed=20261008, products=12)


def test_plan_pilot_contains_eight_train_pairs_and_one_dev_pair_only(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    _dataset(source)

    plan = pilot.plan_pilot(source, tmp_path / "run-001", run_id="pilot-001")
    payload = pilot.read_pilot_payload(plan)

    assert len(payload.train_rows) == 8
    assert len(payload.dev_rows) == 1
    assert len(plan.training_pair_ids) == 8
    assert plan.source_dataset_manifest_sha256 == payload.source_dataset_manifest_sha256
    assert plan.development_pair_id == payload.dev_rows[0]["pair_id"]
    assert all(row["split"] == "train" for row in payload.train_rows)
    assert payload.dev_rows[0]["split"] == "dev"
    assert len({row["product_id"] for row in payload.train_rows}) >= 4
    with zipfile.ZipFile(plan.payload_path) as archive:
        names = set(archive.namelist())
    assert not any("sealed_test" in name for name in names)
    assert all(
        name.startswith(("train/", "dev/")) or name in {"metadata.json", "pilot_manifest.json"}
        for name in names
    )
    assert (plan.output_dir / pilot.RUN_MARKER_NAME).is_file()


def test_plan_pilot_refuses_to_overwrite_a_run_directory(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    _dataset(source)
    output = tmp_path / "run-001"
    output.mkdir()
    marker = output / "existing.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        pilot.plan_pilot(source, output, run_id="pilot-001")

    assert marker.read_text(encoding="utf-8") == "keep"


def test_plan_pilot_rejects_path_like_run_ids(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    _dataset(source)

    with pytest.raises(ValueError, match="run_id"):
        pilot.plan_pilot(source, tmp_path / "run", run_id="../overwrite")


def test_plan_pilot_enforces_payload_size_before_returning(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    _dataset(source)
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="payload exceeds"):
        pilot.plan_pilot(source, output, run_id="pilot-001", max_payload_bytes=1)

    assert not output.exists()


def test_read_pilot_payload_detects_payload_tampering(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    _dataset(source)
    plan = pilot.plan_pilot(source, tmp_path / "run", run_id="pilot-001")
    plan.payload_path.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="payload checksum"):
        pilot.read_pilot_payload(plan)


def test_modal_contract_uses_pinned_sources_and_safe_workspace() -> None:
    assert remote.MODEL_COMPONENTS["transformer"]["revision"] == (
        "a3b4f4849157f664bdbc776fd7453c2783562f4d"
    )
    assert remote.MODEL_COMPONENTS["text_encoder"]["revision"] == (
        "e7b7dc27f91deacad38e78976d1f2b499d76a294"
    )
    assert remote.TRAINING_STEPS == 8
    assert remote.PROFILE == "reflex-personal"
    assert remote.WORKSPACE == "rajath-61258"
    assert remote.ENVIRONMENT == "main"


def test_model_download_requests_only_required_files_anonymously() -> None:
    calls: list[dict[str, object]] = []

    def snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return f"/snapshots/{kwargs['revision']}"

    snapshots = remote._download_model_snapshots(snapshot_download)

    assert snapshots["transformer"] == Path("/snapshots/a3b4f4849157f664bdbc776fd7453c2783562f4d")
    assert (
        snapshots["text_encoder"]
        == snapshots["vae"]
        == snapshots["tokenizer"]
        == Path("/snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294")
    )
    assert calls == [
        {
            "repo_id": "black-forest-labs/FLUX.2-klein-base-4B",
            "revision": "a3b4f4849157f664bdbc776fd7453c2783562f4d",
            "allow_patterns": ["transformer/*.safetensors"],
            "token": False,
        },
        {
            "repo_id": "black-forest-labs/FLUX.2-klein-4B",
            "revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
            "allow_patterns": [
                "text_encoder/*.safetensors",
                "tokenizer/**",
                "vae/diffusion_pytorch_model.safetensors",
            ],
            "token": False,
        },
    ]


def test_trainable_parameters_must_be_lora_only() -> None:
    remote._validate_trainable_parameter_names(
        ["pipe.dit.blocks.0.attn.to_q.lora_A.default.weight"]
    )

    with pytest.raises(RuntimeError, match="only LoRA"):
        remote._validate_trainable_parameter_names(
            ["pipe.dit.blocks.0.attn.to_q.weight", "pipe.dit.blocks.0.attn.to_q.lora_B.weight"]
        )


def test_gradient_norm_must_be_finite() -> None:
    with pytest.raises(FloatingPointError, match="gradient norm"):
        remote._validate_gradient_norm(float("inf"))


def test_final_adapter_weights_must_all_be_finite() -> None:
    with pytest.raises(FloatingPointError, match="adapter weights"):
        remote._validate_finite_adapter_weights([True, False, True])


def test_optimizer_rejects_update_nine_before_step() -> None:
    remote._guard_optimizer_update(7)

    with pytest.raises(RuntimeError, match="eight-update limit"):
        remote._guard_optimizer_update(8)


def test_failed_worker_bundle_preserves_bounded_progress_and_saved_artifact_subset(
    tmp_path: Path,
) -> None:
    from experiments import modal_pilot

    (tmp_path / "base.png").write_bytes(b"base-image")
    bundle = remote._failure_bundle(
        "pilot-001",
        "trained_generation",
        RuntimeError("render failed"),
        {
            "optimizer_updates": 8,
            "optimizer_steps_attempted": 8,
            "losses": [0.5] * 8,
            "gradient_steps": [{"norm": 1.0}] * 8,
            "elapsed_seconds": 42.0,
        },
        tmp_path,
    )

    receipt, names = modal_pilot._validate_result_bundle(bundle, "pilot-001")

    assert receipt["status"] == "failed"
    assert names == {"receipt.json", "images/base.png"}
    assert receipt["progress"]["optimizer_updates"] == 8
    assert receipt["progress"]["optimizer_steps_attempted"] == 8
    assert receipt["progress"]["saved_artifacts"] == ["images/base.png"]


def test_modal_attempt_marker_is_exclusive_and_binds_plan(tmp_path: Path) -> None:
    from experiments import modal_pilot

    source = tmp_path / "dataset"
    _dataset(source)
    plan = pilot.plan_pilot(source, tmp_path / "plan", run_id="pilot-001")
    marker_path = plan.output_dir / modal_pilot.ATTEMPT_MARKER_NAME
    context = {
        "profile": remote.PROFILE,
        "workspace": remote.WORKSPACE,
        "environment": remote.ENVIRONMENT,
    }

    marker = modal_pilot._reserve_launch_attempt(
        plan, marker_path, source_commit="a" * 40, context=context
    )

    assert marker["run_id"] == plan.run_id
    assert marker["payload_sha256"] == plan.payload_sha256
    assert marker["plan_sha256"]
    with pytest.raises(FileExistsError):
        modal_pilot._reserve_launch_attempt(
            plan, marker_path, source_commit="a" * 40, context=context
        )


def test_modal_observation_errors_are_unknown_and_keep_same_call_recovery_ids() -> None:
    from experiments import modal_pilot

    receipt = modal_pilot._failure_receipt(
        "pilot-001",
        "modal_observation",
        TimeoutError("observation timed out"),
        app_id="ap-test-app",
        call_id="fc-test-call",
    )

    assert receipt["status"] == "modal_error"
    assert receipt["worker_failure_confirmed"] is False
    assert receipt["app_id"] == "ap-test-app"
    assert receipt["call_id"] == "fc-test-call"
    assert "never spawn a replacement" in receipt["same_call_recovery"]
    assert "verify" in receipt["shutdown_check"]


def test_launch_requires_fresh_clearance_for_signed_clean_head_and_exact_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments import modal_pilot

    source = tmp_path / "dataset"
    _dataset(source)
    plan = pilot.plan_pilot(source, tmp_path / "plan", run_id="pilot-001")
    now = datetime(2026, 10, 8, 23, 0, tzinfo=UTC)
    source_commit = "a" * 40
    context = {
        "profile": remote.PROFILE,
        "workspace": remote.WORKSPACE,
        "environment": remote.ENVIRONMENT,
    }
    clearance_path = tmp_path / ".context" / "launch-clearance.json"
    clearance_path.parent.mkdir()
    clearance = {
        "status": "approved",
        "review_status": "approved",
        "ci_status": "success",
        "source_commit": source_commit,
        "payload_sha256": hashlib.sha256(plan.payload_path.read_bytes()).hexdigest(),
        "plan_sha256": hashlib.sha256(
            (plan.output_dir / pilot.RUN_MARKER_NAME).read_bytes()
        ).hexdigest(),
        "source_dataset_manifest_sha256": plan.source_dataset_manifest_sha256,
        "checked_at": (now - timedelta(minutes=5)).isoformat(),
        "run_id": plan.run_id,
        **context,
        "allowance_usd": 15,
    }
    clearance_path.write_text(json.dumps(clearance), encoding="utf-8")
    commands: list[list[str]] = []
    git_state = {"dirty": False, "signed": True}

    def fake_git(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[-3:] == ["status", "--porcelain", "--untracked-files=all"]:
            stdout = " M tracked.py" if git_state["dirty"] else ""
            return subprocess.CompletedProcess(command, 0, stdout, "")
        if command[-2:] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, source_commit, "")
        if command[-2:] == ["verify-commit", "HEAD"]:
            return subprocess.CompletedProcess(command, int(not git_state["signed"]), "", "")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(subprocess, "run", fake_git)

    modal_pilot._validate_launch_clearance(
        plan,
        project_root=tmp_path,
        clearance_path=clearance_path,
        context=context,
        now=now,
    )
    assert all(command[0] == "git" for command in commands)

    clearance["payload_sha256"] = "0" * 64
    clearance_path.write_text(json.dumps(clearance), encoding="utf-8")
    with pytest.raises(ValueError, match="clearance"):
        modal_pilot._validate_launch_clearance(
            plan,
            project_root=tmp_path,
            clearance_path=clearance_path,
            context=context,
            now=now,
        )
    clearance["payload_sha256"] = hashlib.sha256(plan.payload_path.read_bytes()).hexdigest()
    clearance["checked_at"] = (now - timedelta(minutes=31)).isoformat()
    clearance_path.write_text(json.dumps(clearance), encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        modal_pilot._validate_launch_clearance(
            plan,
            project_root=tmp_path,
            clearance_path=clearance_path,
            context=context,
            now=now,
        )
    clearance["checked_at"] = now.isoformat()
    clearance_path.write_text(json.dumps(clearance), encoding="utf-8")
    git_state["dirty"] = True
    with pytest.raises(ValueError, match="clean current Git"):
        modal_pilot._validate_launch_clearance(
            plan,
            project_root=tmp_path,
            clearance_path=clearance_path,
            context=context,
            now=now,
        )
    git_state["dirty"] = False
    git_state["signed"] = False
    with pytest.raises(ValueError, match="local Git state"):
        modal_pilot._validate_launch_clearance(
            plan,
            project_root=tmp_path,
            clearance_path=clearance_path,
            context=context,
            now=now,
        )


@pytest.mark.parametrize("startup_failure", [False, True], ids=["worker-returned", "startup-error"])
def test_launch_reserves_before_build_and_records_call_before_get(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, startup_failure: bool
) -> None:
    from experiments import modal_pilot

    source = tmp_path / "dataset"
    _dataset(source)
    plan = pilot.plan_pilot(source, tmp_path / "plan", run_id="pilot-001")
    monkeypatch.setattr(
        pilot, "APPROVED_DATASET_MANIFEST_SHA256", plan.source_dataset_manifest_sha256
    )
    monkeypatch.setattr(
        modal_pilot,
        "_validate_launch_clearance",
        lambda *_args, **_kwargs: {"source_commit": "a" * 40},
    )
    monkeypatch.setattr(remote, "verify_modal_identity", lambda **_kwargs: None)
    monkeypatch.setattr(metadata, "version", lambda _name: "1.6.1")
    monkeypatch.delenv("MODAL_PROFILE", raising=False)
    marker_path = plan.output_dir / modal_pilot.ATTEMPT_MARKER_NAME
    options: dict[str, object] = {}
    required_names = {
        "receipt.json",
        "adapter/step-8.safetensors",
        "images/base.png",
        "images/trained.png",
        "images/reloaded.png",
    }
    result = remote._bundle(
        {
            "receipt.json": json.dumps({"run_id": plan.run_id, "status": "passed"}).encode(),
            **{name: b"x" for name in required_names - {"receipt.json"}},
        }
    )

    class FakeCall:
        object_id = "fc-test-call"

        def get(self) -> bytes:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            assert marker["app_id"] == "ap-test-app"
            assert marker["call_id"] == "fc-test-call"
            return result

    class FakeFunction:
        def spawn(self, _payload: bytes, _run_id: str) -> FakeCall:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            assert marker["app_id"] == "ap-test-app"
            assert marker["call_id"] is None
            return FakeCall()

    class StartupFailureContext:
        def __enter__(self) -> None:
            raise RuntimeError("app startup failed")

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeApp:
        app_id = "ap-test-app"

        def __init__(self, _name: str, *, image: object) -> None:
            assert image == "built-image"

        def function(self, **kwargs: object) -> Any:
            options.update(kwargs)
            return lambda _function: FakeFunction()

        def run(self, *, environment_name: str, detach: bool) -> Any:
            assert environment_name == remote.ENVIRONMENT
            assert detach is True
            return StartupFailureContext() if startup_failure else contextlib.nullcontext(self)

    fake_modal = type(
        "FakeModal",
        (),
        {"App": FakeApp, "enable_output": staticmethod(contextlib.nullcontext)},
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)

    def build_image(_modal: object, _project_root: Path) -> str:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        assert marker["status"] == "reserved"
        assert marker["app_id"] is None
        return "built-image"

    monkeypatch.setattr(modal_pilot, "_build_image", build_image)

    status = modal_pilot._launch(
        plan,
        profile=remote.PROFILE,
        workspace=remote.WORKSPACE,
        environment=remote.ENVIRONMENT,
    )

    assert options["memory"] == (65536, 65536)
    assert options["ephemeral_disk"] == 524288
    assert options["startup_timeout"] == 300
    assert options["timeout"] == 2700
    assert options["scaledown_window"] == 2
    assert options["single_use_containers"] is True
    assert "max_inputs" not in options
    assert options["retries"] == 0
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if startup_failure:
        assert status == 1
        failure = json.loads((plan.output_dir / modal_pilot.MODAL_FAILURE_NAME).read_text())
        assert marker["app_id"] == failure["app_id"] == "ap-test-app"
        assert marker["call_id"] is failure["call_id"] is None
        assert marker["status"] == "modal_error"
        assert failure["failure"]["stage"] == "modal_app_startup"
        return
    assert status == 0
    assert marker["app_id"] == "ap-test-app"
    assert marker["call_id"] == "fc-test-call"
    assert marker["status"] == "worker_returned"


def test_modal_launch_guard_rejects_credential_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "never-print-this")

    with pytest.raises(ValueError, match="credential environment overrides"):
        remote.validate_launch_environment(
            {"profile": "reflex-personal", "workspace": "rajath-61258", "environment": "main"}
        )


def test_modal_launch_guard_rejects_wrong_profile_workspace_or_environment() -> None:
    for key, value in (("profile", "default"), ("workspace", "other"), ("environment", "dev")):
        expected = {
            "profile": "reflex-personal",
            "workspace": "rajath-61258",
            "environment": "main",
        }
        expected[key] = value
        with pytest.raises(ValueError):
            remote.validate_launch_environment(expected)


def test_worker_schedule_rejects_anything_other_than_eight_single_sample_updates() -> None:
    assert (
        remote.validate_training_contract(
            train_pair_count=8,
            batch_size=1,
            dataset_repeat=1,
            num_epochs=1,
            gradient_accumulation_steps=1,
            process_count=1,
            save_steps=8,
        )
        == 8
    )
    for changes in (
        {"train_pair_count": 9},
        {"batch_size": 2},
        {"dataset_repeat": 2},
        {"num_epochs": 2},
        {"gradient_accumulation_steps": 2},
        {"process_count": 2},
        {"save_steps": 7},
    ):
        config = {
            "train_pair_count": 8,
            "batch_size": 1,
            "dataset_repeat": 1,
            "num_epochs": 1,
            "gradient_accumulation_steps": 1,
            "process_count": 1,
            "save_steps": 8,
        }
        config.update(changes)
        with pytest.raises(ValueError):
            remote.validate_training_contract(**config)


def test_remote_input_validation_uses_only_referenced_train_and_dev_assets(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    _dataset(source)
    plan = pilot.plan_pilot(source, tmp_path / "run", run_id="pilot-001")
    work = tmp_path / "remote-input"

    train_rows, dev_row, source_hash = remote._load_payload(
        plan.payload_path.read_bytes(), plan.run_id, work
    )

    assert len(train_rows) == 8
    assert dev_row["split"] == "dev"
    assert source_hash == plan.source_dataset_manifest_sha256
    assert len(list(work.rglob("*.png"))) == 18
    assert len(list((work / "training-data").rglob("*.png"))) == 16
    assert not list((work / "training-data" / "dev").rglob("*.png"))
    assert not any("sealed_test" in path.parts for path in work.rglob("*"))


def test_remote_input_validation_rejects_unreferenced_sealed_test_bytes(tmp_path: Path) -> None:
    import io
    import zipfile

    source = tmp_path / "dataset"
    _dataset(source)
    plan = pilot.plan_pilot(source, tmp_path / "run", run_id="pilot-001")
    buffer = io.BytesIO()
    with zipfile.ZipFile(plan.payload_path) as original, zipfile.ZipFile(buffer, "w") as altered:
        for name in original.namelist():
            altered.writestr(name, original.read(name))
        altered.writestr("sealed_test/source/leak.png", b"not-mounted")

    with pytest.raises(ValueError, match="sealed-test path"):
        remote._load_payload(buffer.getvalue(), plan.run_id, tmp_path / "remote-input")


def test_default_experiment_mode_builds_a_plan_without_calling_modal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from experiments import modal_pilot

    dataset_dir = tmp_path / "dataset"
    plan_dir = tmp_path / "plan"
    _dataset(dataset_dir)
    status = modal_pilot.main(
        ["--dataset-dir", str(dataset_dir), "--plan-dir", str(plan_dir), "--run-id", "cpu-plan"]
    )

    output = json.loads(capsys.readouterr().out)
    assert status == 0
    assert output["status"] == "planned"
    assert output["modal_called"] is False
    assert output["training_pairs"] == 8
    assert output["development_pairs"] == 1
    assert pilot.load_pilot_plan(plan_dir).run_id == "cpu-plan"


def test_launch_refuses_a_different_data_manifest_before_modal_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments import modal_pilot

    source = tmp_path / "dataset"
    dataset.build_dataset(source, seed=20261009, products=12)
    plan = pilot.plan_pilot(source, tmp_path / "plan", run_id="custom-data")
    monkeypatch.setattr(
        remote,
        "verify_modal_identity",
        lambda **_kwargs: pytest.fail("unreviewed data must stop before Modal identity checks"),
    )

    with pytest.raises(ValueError, match="reviewed dataset manifest"):
        modal_pilot._launch(
            plan,
            profile="reflex-personal",
            workspace="rajath-61258",
            environment="main",
        )
