from __future__ import annotations

import hashlib
import json
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from product_scene_editor import reload_diagnostic as diagnostic


def _adapter(*, unpaired: bool = False, nonfinite: bool = False, max_finite: bool = False) -> bytes:
    header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    chunks: list[bytes] = []
    offset = 0
    for index in range(diagnostic.EXPECTED_MODULES):
        for side, shape in (("A", [32, 2]), ("B", [2, 32])):
            if unpaired and index == 0 and side == "B":
                continue
            content = bytearray(32 * 2 * 2)
            if nonfinite and index == 0 and side == "A":
                content[:2] = b"\x80\x7f"
            if max_finite and index == 0 and side == "A":
                content[:2] = b"\x7f\x7f"
            name = f"block{index}.lora_{side}.default.weight"
            header[name] = {
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [offset, offset + len(content)],
            }
            chunks.append(bytes(content))
            offset += len(content)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    return len(encoded).to_bytes(8, "little") + encoded + b"".join(chunks)


def _archive(path: Path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in sorted(files.items()):
            archive.writestr(name, content)


def _r3_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    adapter = _adapter()
    reference = b"reference-png-fixture"
    trained = b"trained-png-fixture"
    source_manifest = {
        "run_id": diagnostic.R3_RUN_ID,
        "development_pair": {
            "pair_id": "pack-006-garden_studio",
            "split": "dev",
            "edit_image": "dev/source/pack-006-garden_studio.png",
            "source_sha256": hashlib.sha256(reference).hexdigest(),
            "target_sha256": "target-hash",
            "prompt": diagnostic.R3_PROMPT,
        },
    }
    source = tmp_path / "pilot-input.zip"
    _archive(
        source,
        {
            "pilot_manifest.json": json.dumps(source_manifest).encode(),
            "dev/source/pack-006-garden_studio.png": reference,
        },
    )
    result = tmp_path / "pilot-result.zip"
    receipt = {
        "run_id": diagnostic.R3_RUN_ID,
        "status": "failed",
        "failure": {"stage": "adapter_reload"},
        "progress": {"optimizer_updates": 8},
    }
    _archive(
        result,
        {
            "adapter/step-8.safetensors": adapter,
            "images/trained.png": trained,
            "receipt.json": json.dumps(receipt).encode(),
        },
    )
    monkeypatch.setattr(
        diagnostic, "R3_INPUT_SHA256", hashlib.sha256(source.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        diagnostic, "R3_RESULT_SHA256", hashlib.sha256(result.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(diagnostic, "R3_ADAPTER_SHA256", hashlib.sha256(adapter).hexdigest())
    monkeypatch.setattr(diagnostic, "R3_REFERENCE_SHA256", hashlib.sha256(reference).hexdigest())
    monkeypatch.setattr(diagnostic, "R3_TARGET_SHA256", "target-hash")
    monkeypatch.setattr(diagnostic, "R3_TRAINED_SHA256", hashlib.sha256(trained).hexdigest())
    return source, result


def test_adapter_header_checks_tensor_pairs_and_rank() -> None:
    tensors = diagnostic.validate_adapter_header(_adapter())

    assert len(tensors) == 200
    assert len({name.rsplit(".lora_", 1)[0] for name in tensors}) == 100
    assert all(tensor.dtype == "BF16" for tensor in tensors.values())
    assert all(32 in tensor.shape for tensor in tensors.values())


def test_adapter_header_rejects_unpaired_and_nonfinite_tensors() -> None:
    assert len(diagnostic.validate_adapter_header(_adapter(max_finite=True))) == 200
    with pytest.raises(ValueError, match="complete A/B pairs"):
        diagnostic.validate_adapter_header(_adapter(unpaired=True))
    with pytest.raises(ValueError, match="non-finite"):
        diagnostic.validate_adapter_header(_adapter(nonfinite=True))


def test_plan_contains_only_pinned_replay_inputs_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, result = _r3_sources(tmp_path, monkeypatch)
    plan = diagnostic.plan_reload_diagnostic(source, result, tmp_path / "run")
    payload = diagnostic.read_reload_payload(plan)

    expected = {
        "replay_manifest.json",
        "adapter/step-8.safetensors",
        "images/reference.png",
        "images/original-trained.png",
    }
    assert set(payload.files) == expected
    with zipfile.ZipFile(plan.payload_path) as archive:
        assert set(archive.namelist()) == expected
    with pytest.raises(FileExistsError):
        diagnostic.plan_reload_diagnostic(source, result, tmp_path / "run")


def test_reload_payload_detects_adapter_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, result = _r3_sources(tmp_path, monkeypatch)
    plan = diagnostic.plan_reload_diagnostic(source, result, tmp_path / "run")
    plan.payload_path.write_bytes(b"changed")

    with pytest.raises(ValueError, match="checksum"):
        diagnostic.read_reload_payload(plan)


def test_failed_result_bundle_accepts_bounded_partial_outputs() -> None:
    receipt = {
        "run_id": diagnostic.R3_RUN_ID,
        "status": "failed",
        "failure": {"stage": "input_validation"},
    }
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("receipt.json", json.dumps(receipt))
        archive.writestr("images/original-trained.png", b"png")

    parsed, names = diagnostic.validate_result_bundle(buffer.getvalue())

    assert parsed["status"] == "failed"
    assert names == {"receipt.json", "images/original-trained.png"}


def test_remote_worker_rejects_invalid_input_before_model_import() -> None:
    result = diagnostic.run_remote_reload_diagnostic(b"bad", diagnostic.R3_RUN_ID)

    receipt, names = diagnostic.validate_result_bundle(result)

    assert receipt["status"] == "failed"
    assert receipt["failure"]["stage"] == "input_validation"
    assert receipt["optimizer_updates"] == 0
    assert names == {"receipt.json"}


def test_restored_adapter_values_require_exact_keys_shapes_dtypes_and_values() -> None:
    source = {
        "module.lora_A.default.weight": ((32, 2), "BF16", b"a"),
        "module.lora_B.default.weight": ((2, 32), "BF16", b"b"),
    }
    expected = {name: (value[0], value[1]) for name, value in source.items()}

    diagnostic.validate_restored_adapter(
        expected, source, source, lambda left, right: left[2] == right[2], expected_module_count=1
    )
    negatives = [
        (
            {key: value for key, value in source.items() if key.endswith("lora_A.default.weight")},
            source,
            "keys",
        ),
        ({**expected, "extra": ((1,), "BF16")}, source, "keys"),
        ({**expected, "module.lora_A.default.weight": ((16, 2), "BF16")}, source, "shape"),
        ({**expected, "module.lora_A.default.weight": ((32, 2), "FP32")}, source, "dtype"),
        (expected, {**source, "module.lora_B.default.weight": ((2, 32), "BF16", b"x")}, "value"),
    ]
    for specs, values, message in negatives:
        with pytest.raises(ValueError, match=message):
            diagnostic.validate_restored_adapter(
                specs,
                source,
                values,
                lambda left, right: left[2] == right[2],
                expected_module_count=1,
            )


def test_plan_only_cli_never_uses_modal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from experiments import modal_reload_diagnostic as launcher

    source, result = _r3_sources(tmp_path, monkeypatch)
    monkeypatch.setattr(
        launcher.remote,
        "verify_modal_identity",
        lambda **_kwargs: pytest.fail("Modal identity queried"),
    )

    status = launcher.main(
        [
            "--input-archive",
            str(source),
            "--result-archive",
            str(result),
            "--plan-dir",
            str(tmp_path / "plan"),
        ]
    )

    assert status == 0
    assert '"modal_called": false' in capsys.readouterr().out


def test_clearance_requires_fresh_matching_head_and_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import UTC, datetime, timedelta
    from types import SimpleNamespace

    from experiments import modal_reload_diagnostic as launcher

    source, result = _r3_sources(tmp_path, monkeypatch)
    plan = diagnostic.plan_reload_diagnostic(source, result, tmp_path / "plan")
    context = {
        "profile": diagnostic.remote.PROFILE,
        "workspace": diagnostic.remote.WORKSPACE,
        "environment": diagnostic.remote.ENVIRONMENT,
    }
    head = "a" * 40
    dirty = [False]
    verify_failure = [False]

    def git_result(_root: Path, args: list[str]) -> Any:
        if args[0] == "status":
            return SimpleNamespace(returncode=0, stdout=" M file" if dirty[0] else "")
        if args[0] == "verify-commit" and verify_failure[0]:
            raise ValueError("clearance could not verify the local Git state")
        return SimpleNamespace(returncode=0, stdout=head)

    monkeypatch.setattr(launcher, "_run_local_git", git_result)
    now = datetime.now(UTC)
    clearance = {
        "schema_version": 1,
        "run_id": plan.run_id,
        "payload_sha256": plan.payload_sha256,
        "plan_sha256": hashlib.sha256(
            (plan.output_dir / diagnostic.RUN_MARKER_NAME).read_bytes()
        ).hexdigest(),
        "source_input_sha256": diagnostic.R3_INPUT_SHA256,
        "source_result_sha256": diagnostic.R3_RESULT_SHA256,
        "status": "approved",
        "review_status": "approved",
        "ci_status": "success",
        "allowance_usd": 15,
        "gpu": "A100-80GB",
        "max_runtime_seconds": 1200,
        "max_calls": 1,
        **context,
        "source_commit": head,
        "checked_at": now.isoformat(),
    }
    path = tmp_path / "clearance.json"
    path.write_text(json.dumps(clearance), encoding="utf-8")

    assert launcher._validate_clearance(
        plan, project_root=tmp_path, clearance_path=path, context=context, now=now
    )
    with pytest.raises(ValueError, match="missing or unsafe"):
        launcher._validate_clearance(
            plan,
            project_root=tmp_path,
            clearance_path=tmp_path / "missing.json",
            context=context,
            now=now,
        )
    clearance["checked_at"] = (now - timedelta(minutes=31)).isoformat()
    path.write_text(json.dumps(clearance), encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        launcher._validate_clearance(
            plan, project_root=tmp_path, clearance_path=path, context=context, now=now
        )
    clearance["checked_at"] = now.isoformat()
    clearance["source_commit"] = "b" * 40
    path.write_text(json.dumps(clearance), encoding="utf-8")
    with pytest.raises(ValueError, match="HEAD"):
        launcher._validate_clearance(
            plan, project_root=tmp_path, clearance_path=path, context=context, now=now
        )
    wrong_context = {**context, "environment": "wrong"}
    with pytest.raises(ValueError, match="clearance"):
        launcher._validate_clearance(
            plan, project_root=tmp_path, clearance_path=path, context=wrong_context, now=now
        )
    clearance["source_commit"] = head
    path.write_text(json.dumps(clearance), encoding="utf-8")
    dirty[0] = True
    with pytest.raises(ValueError, match="clean worktree"):
        launcher._validate_clearance(
            plan, project_root=tmp_path, clearance_path=path, context=context, now=now
        )
    dirty[0] = False
    verify_failure[0] = True
    with pytest.raises(ValueError, match="Git state"):
        launcher._validate_clearance(
            plan, project_root=tmp_path, clearance_path=path, context=context, now=now
        )


def test_attempt_marker_is_reserved_before_provider_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments import modal_reload_diagnostic as launcher

    source, result = _r3_sources(tmp_path, monkeypatch)
    plan = diagnostic.plan_reload_diagnostic(source, result, tmp_path / "plan")
    context = {
        "profile": diagnostic.remote.PROFILE,
        "workspace": diagnostic.remote.WORKSPACE,
        "environment": diagnostic.remote.ENVIRONMENT,
    }
    monkeypatch.setattr(
        launcher, "_validate_clearance", lambda *_args, **_kwargs: {"source_commit": "a" * 40}
    )

    def identity(**_kwargs: Any) -> None:
        assert (plan.output_dir / diagnostic.ATTEMPT_MARKER_NAME).is_file()
        raise RuntimeError("stop before provider work")

    monkeypatch.setattr(launcher.remote, "verify_modal_identity", identity)

    status = launcher._launch(plan, context=context, project_root=tmp_path)

    marker = json.loads((plan.output_dir / diagnostic.ATTEMPT_MARKER_NAME).read_text())
    assert status == 1
    assert marker["status"] == "modal_error"
