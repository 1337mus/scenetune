from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pytest
from PIL import Image

from product_scene_editor import protected_comparison as comparison
from product_scene_editor import protected_comparison_inference as worker
from product_scene_editor import remote


def _png(mode: str, size: tuple[int, int], color: int | tuple[int, int, int]) -> bytes:
    stream = BytesIO()
    Image.new(mode, size, color).save(stream, format="PNG")
    return stream.getvalue()


def _fixture() -> tuple[dict[str, Any], dict[str, bytes]]:
    alpha = Image.new("L", (512, 512), 0)
    alpha.paste(255, (80, 80, 432, 432))
    protected = Image.new("L", (512, 512), 0)
    protected.paste(255, (100, 100, 412, 412))
    alpha_stream = BytesIO()
    protected_stream = BytesIO()
    alpha.save(alpha_stream, format="PNG")
    protected.save(protected_stream, format="PNG")
    alpha_bytes, protected_bytes = alpha_stream.getvalue(), protected_stream.getvalue()
    files: dict[str, bytes] = {}
    cases: list[dict[str, Any]] = []
    for index in range(24):
        case_id = f"case-{index:02d}"
        prefix = f"cases/{case_id}"
        source_path = f"{prefix}/source-rgb.png"
        alpha_path = f"{prefix}/alpha-l.png"
        protected_path = f"{prefix}/protected-l.png"
        source = _png("RGB", (512, 512), (index + 1, 40, 80))
        files.update(
            {source_path: source, alpha_path: alpha_bytes, protected_path: protected_bytes}
        )
        cases.append(
            {
                "case_id": case_id,
                "pool": "development" if index < 6 else "sealed_test",
                "brand_id": f"brand-{index}",
                "family_id": f"family-{index}",
                "product_id": f"product-{index}",
                "source_image_id": f"image-{index}",
                "source_rgb": {
                    "path": source_path,
                    "sha256": hashlib.sha256(source).hexdigest(),
                },
                "alpha_l": {
                    "path": alpha_path,
                    "sha256": hashlib.sha256(alpha_bytes).hexdigest(),
                },
                "protected_l": {
                    "path": protected_path,
                    "sha256": hashlib.sha256(protected_bytes).hexdigest(),
                },
                "source_provenance": {
                    "url": (
                        "https://amazon-berkeley-objects.s3.amazonaws.com/images/original/"
                        f"{index + 1:02x}/{index + 1:08x}.jpg"
                    ),
                    "raw_source_sha256": f"{index + 1:064x}",
                    "metadata_sha256": "c" * 64,
                },
                "mask_review": {
                    "status": "accepted_by_root_visual_review",
                    "sha256": f"{index + 100:064x}",
                },
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "profile": "protected-product-comparison",
        "source_sha": "e" * 40,
        "context": {
            "profile": remote.PROFILE,
            "workspace": remote.WORKSPACE,
            "environment": remote.ENVIRONMENT,
        },
        "modal_sdk": "1.6.1",
        "diffsynth_commit": remote.DIFFSYNTH_COMMIT,
        "runtime_pins": dict(remote.RUNTIME_PINS),
        "model_components": remote.MODEL_COMPONENTS,
        "generation": {
            "steps": 16,
            "cfg": 4,
            "width": 512,
            "height": 512,
            "seed_start": 7321,
            "seed_rule": "sorted_case_id_index",
        },
        "prompts": {
            "full_image_edit": (
                "Replace only the background with a warm beige studio backdrop and a light stone "
                "display surface. Keep the product, text, logos, shape, size, orientation and "
                "position unchanged. Do not put any object in front of the product."
            ),
            "empty_scene": (
                "An empty light stone display surface against a warm beige studio backdrop. Soft "
                "studio light from the upper left. Straight-on product photograph. Leave the "
                "center empty. No products, objects, text or logos."
            ),
        },
        "cases": cases,
    }
    payload = comparison.build_payload(manifest, files)
    return comparison.validate_payload_bytes(payload)


class _FakeCuda:
    def __init__(self) -> None:
        self.empty_cache_calls = 0
        self.reset_peak_memory_stats_calls = 0

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def reset_peak_memory_stats(self) -> None:
        self.reset_peak_memory_stats_calls += 1

    def max_memory_allocated(self) -> int:
        return 12345

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _FakeRuntime:
    def __init__(self) -> None:
        self.cuda = _FakeCuda()

    @contextmanager
    def inference_mode(self):
        yield


class _FakePipe:
    device = "cuda:0"

    def __init__(self, delta: int = 0, delta_arm: str | None = None) -> None:
        self.delta = delta
        self.delta_arm = delta_arm
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Image.Image:
        self.calls.append(kwargs)
        seed = int(kwargs["seed"])
        arm = "base_full_image_edit" if "edit_image" in kwargs else "empty_scene_composite"
        arm_offset = 0 if arm == "base_full_image_edit" else 40
        delta = self.delta if self.delta_arm is None or self.delta_arm == arm else 0
        color = (seed + arm_offset + delta) % 256
        return Image.new("RGB", (512, 512), (color, 90, 120))


def _execute(
    delta_for_replay: int = 0, delta_arm: str | None = None
) -> tuple[dict[str, Any], list[_FakePipe], _FakeRuntime]:
    manifest, files = _fixture()
    schedule = comparison.generation_schedule(manifest)
    runtime = _FakeRuntime()
    pipes: list[_FakePipe] = []

    def load_pipe(_snapshots: dict[str, Path], device: str) -> _FakePipe:
        assert device == "cuda"
        pipe = _FakePipe(delta_for_replay if pipes else 0, delta_arm)
        pipes.append(pipe)
        return pipe

    state = worker._new_execution_state()
    worker._execute_schedule(manifest, files, schedule, runtime, {}, load_pipe, state)
    return state, pipes, runtime


def test_fixed_schedule_loads_twice_and_runs_48_primaries_plus_four_replays() -> None:
    state, pipes, runtime = _execute()

    assert state["status"] == "passed"
    assert state["generation_call_count"] == 52
    assert len(pipes) == 2
    assert len(pipes[0].calls) == 48
    assert len(pipes[1].calls) == 4
    assert len(state["generation_records"]) == 52
    assert len(state["generation_call_timings"]) == 52
    assert all(record["generation_seconds"] >= 0 for record in state["generation_call_timings"])
    assert len(state["replay_checks"]) == 4
    assert all(record["final_pixel_equal"] for record in state["replay_checks"])
    assert all(record["raw_background_pixel_equal"] for record in state["replay_checks"])
    assert len(state["composed_integrity_records"]) == 26
    assert len(state["outputs"]) == 78
    assert runtime.cuda.empty_cache_calls == 1


def test_generation_uses_fixed_settings_and_empty_scene_has_no_image_references() -> None:
    _state, pipes, _runtime = _execute()
    calls = pipes[0].calls
    edit_call = next(call for call in calls if "edit_image" in call)
    empty_scene_call = next(call for call in calls if "edit_image" not in call)

    assert edit_call["edit_image"] and "input_image" not in edit_call
    assert "edit_image" not in empty_scene_call
    assert "input_image" not in empty_scene_call
    empty_scene_calls = [call for call in calls if "edit_image" not in call]
    assert len(empty_scene_calls) == 24
    assert all("input_image" not in call for call in empty_scene_calls)
    setting_keys = ("num_inference_steps", "cfg_scale", "width", "height")
    assert {key: edit_call[key] for key in setting_keys} == {
        "num_inference_steps": 16,
        "cfg_scale": 4,
        "width": 512,
        "height": 512,
    }
    assert {key: empty_scene_call[key] for key in setting_keys} == {
        "num_inference_steps": 16,
        "cfg_scale": 4,
        "width": 512,
        "height": 512,
    }


def test_changed_replay_fails_with_partial_evidence() -> None:
    state, pipes, _runtime = _execute(delta_for_replay=1)

    assert state["status"] == "failed"
    assert len(pipes[0].calls) == 48
    assert len(pipes[1].calls) == 1
    assert state["generation_call_count"] == 49
    assert len(state["generation_records"]) == 49
    assert state["replay_checks"][-1]["final_pixel_equal"] is False
    assert state["replay_checks"][-1]["raw_background_pixel_equal"] is False
    assert len(state["composed_integrity_records"]) == 24
    assert state["outputs"]
    assert state["failure"]["stage"].startswith("replay_generation_")


def test_changed_empty_scene_replay_fails_after_recording_composite_evidence() -> None:
    state, pipes, _runtime = _execute(delta_for_replay=1, delta_arm="empty_scene_composite")

    assert state["status"] == "failed"
    assert len(pipes[0].calls) == 48
    assert len(pipes[1].calls) == 2
    assert state["generation_call_count"] == 50
    assert [check["final_pixel_equal"] for check in state["replay_checks"]] == [True, False]
    assert len(state["composed_integrity_records"]) == 25
    assert "replays/case-00/empty_scene_composite_final.png" in state["outputs"]


def test_invalid_payload_fails_before_runtime_or_model_loading(monkeypatch) -> None:
    def forbidden_runtime_load() -> tuple[Any, Any]:
        raise AssertionError("runtime loading happened before payload validation")

    monkeypatch.setattr(worker, "_runtime_dependencies", forbidden_runtime_load)
    result = worker.run_remote_protected_comparison_inference(b"invalid payload", "run-1")

    with ZipFile(BytesIO(result)) as archive:
        receipt = json.loads(archive.read("receipt.json"))
    assert receipt["status"] == "failed"
    assert receipt["run_id"] == "run-1"
    assert receipt["generation_call_count"] == 0
    assert receipt["failure"]["stage"] == "input_validation"


def test_result_packaging_rejects_unsafe_members_and_over_limit_results(monkeypatch) -> None:
    with pytest.raises(ValueError, match="unsafe member"):
        worker._pack_result({"outputs/case-01/../escaped.png": b"data"})

    monkeypatch.setattr(worker, "MAX_RESULT_BYTES", 10)
    with pytest.raises(ValueError, match="64 MiB result limit"):
        worker._pack_result({"outputs/case-01/image.png": bytes(range(256))})


def test_receipt_binds_run_source_context_and_canonical_manifest() -> None:
    manifest, _files = _fixture()
    canonical_manifest = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    receipt = worker._receipt(
        "passed",
        worker._new_execution_state(),
        run_id="run-24",
        started=time.monotonic(),
        stage="done",
        payload_sha256="a" * 64,
        manifest=manifest,
        runtime_versions={},
        observed_commit=None,
        peak_memory_bytes=None,
        failure=None,
    )

    assert receipt["run_id"] == "run-24"
    assert receipt["source_sha"] == manifest["source_sha"]
    assert receipt["context"] == manifest["context"]
    assert receipt["manifest_sha256"] == hashlib.sha256(canonical_manifest).hexdigest()


def _configure_remote_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    *,
    runtime_versions: dict[str, str] | None = None,
    fail_download: bool = False,
    pipe_factory: Any | None = None,
) -> tuple[dict[str, Any], bytes, _FakeRuntime, list[_FakePipe], list[dict[str, Any]]]:
    from product_scene_editor import real_photo_inference

    manifest, files = _fixture()
    payload = comparison.build_payload(manifest, files)
    runtime = _FakeRuntime()
    versions = runtime_versions or {"modal": "1.6.1", **remote.RUNTIME_PINS}
    download_calls: list[dict[str, Any]] = []
    pipes: list[_FakePipe] = []

    def snapshot_download(**kwargs: Any) -> str:
        download_calls.append(kwargs)
        if fail_download:
            raise RuntimeError("snapshot fetch failed: Bearer sentinel-secret-value")
        return str(Path("/fake/snapshots") / kwargs["revision"])

    def load_pipe(_snapshots: dict[str, Path], device: str) -> _FakePipe:
        assert device == "cuda"
        pipe = pipe_factory() if pipe_factory is not None else _FakePipe()
        pipes.append(pipe)
        return pipe

    monkeypatch.setattr(worker, "_runtime_dependencies", lambda: (runtime, snapshot_download))
    monkeypatch.setattr(real_photo_inference, "_runtime_versions", lambda: dict(versions))
    monkeypatch.setattr(
        real_photo_inference, "_observed_diffsynth_commit", lambda: remote.DIFFSYNTH_COMMIT
    )
    monkeypatch.setattr(remote, "_load_pipe", load_pipe)
    return manifest, payload, runtime, pipes, download_calls


def _unpack_receipt(result: bytes) -> tuple[dict[str, Any], list[str]]:
    with ZipFile(BytesIO(result)) as archive:
        receipt = json.loads(archive.read("receipt.json"))
        names = archive.namelist()
    return receipt, names


class _FailOnThirdCallPipe(_FakePipe):
    def __call__(self, **kwargs: Any) -> Image.Image:
        if len(self.calls) == 2:
            self.calls.append(kwargs)
            raise RuntimeError("generation failed: Bearer sentinel-generation-value")
        return super().__call__(**kwargs)


def test_remote_endpoint_runs_valid_fixture_and_returns_bound_success_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, payload, runtime, pipes, download_calls = _configure_remote_endpoint(monkeypatch)

    result = worker.run_remote_protected_comparison_inference(payload, "endpoint-success")

    receipt, names = _unpack_receipt(result)
    assert receipt["status"] == "passed"
    assert receipt["run_id"] == "endpoint-success"
    assert receipt["source_sha"] == manifest["source_sha"]
    assert receipt["context"] == manifest["context"]
    assert receipt["payload_sha256"] == hashlib.sha256(payload).hexdigest()
    assert receipt["runtime_versions"] == {"modal": "1.6.1", **remote.RUNTIME_PINS}
    assert receipt["observed_diffsynth_commit"] == remote.DIFFSYNTH_COMMIT
    assert receipt["generation_call_count"] == 52
    assert len(pipes) == 2
    assert len(pipes[0].calls) == 48
    assert len(pipes[1].calls) == 4
    assert len(download_calls) == 2
    assert all(call["token"] is False for call in download_calls)
    assert runtime.cuda.reset_peak_memory_stats_calls == 1
    assert receipt["peak_allocated_bytes"] == 12345
    assert "outputs/case-00/full_image_edit.png" in names
    assert "receipt.json" in names


def test_runtime_pin_failure_returns_failed_receipt_before_download_or_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = {"modal": "1.6.1", **remote.RUNTIME_PINS, "torch": "wrong-version"}
    manifest, payload, _runtime, pipes, download_calls = _configure_remote_endpoint(
        monkeypatch, runtime_versions=versions
    )

    result = worker.run_remote_protected_comparison_inference(payload, "runtime-failure")

    receipt, names = _unpack_receipt(result)
    assert receipt["status"] == "failed"
    assert receipt["run_id"] == "runtime-failure"
    assert receipt["source_sha"] == manifest["source_sha"]
    assert receipt["generation_call_count"] == 0
    assert receipt["failure"]["stage"] == "runtime_import"
    assert receipt["failure_stage"] == "runtime_import"
    assert (
        receipt["failure"]["message"]
        == "worker package versions differ from the frozen runtime pins"
    )
    assert download_calls == []
    assert pipes == []
    assert names == ["receipt.json"]


def test_model_snapshot_failure_is_sanitized_and_precedes_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, payload, _runtime, pipes, download_calls = _configure_remote_endpoint(
        monkeypatch, fail_download=True
    )

    result = worker.run_remote_protected_comparison_inference(payload, "download-failure")

    receipt, names = _unpack_receipt(result)
    assert receipt["status"] == "failed"
    assert receipt["run_id"] == "download-failure"
    assert receipt["source_sha"] == manifest["source_sha"]
    assert receipt["generation_call_count"] == 0
    assert receipt["failure"]["stage"] == "model_download"
    assert receipt["failure_stage"] == "model_download"
    assert receipt["failure"]["message"] == "snapshot fetch failed: Bearer [REDACTED]"
    assert "sentinel-secret-value" not in receipt["failure"]["message"]
    assert len(download_calls) == 1
    assert pipes == []
    assert names == ["receipt.json"]


def test_remote_endpoint_reports_partial_generation_evidence_and_sanitized_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, payload, _runtime, pipes, _download_calls = _configure_remote_endpoint(
        monkeypatch, pipe_factory=_FailOnThirdCallPipe
    )

    result = worker.run_remote_protected_comparison_inference(payload, "partial-generation")

    receipt, names = _unpack_receipt(result)
    assert receipt["status"] == "failed"
    assert receipt["run_id"] == "partial-generation"
    assert receipt["source_sha"] == manifest["source_sha"]
    assert receipt["generation_call_count"] == 3
    assert len(receipt["generation_call_timings"]) == 3
    assert len(receipt["generation_records"]) == 2
    assert len(receipt["output_artifacts"]) == 3
    assert receipt["failure"]["stage"] == "primary_generation_case-01_base_full_image_edit"
    assert receipt["failure_stage"] == receipt["failure"]["stage"]
    assert receipt["failure"]["message"] == "generation failed: Bearer [REDACTED]"
    assert "sentinel-generation-value" not in receipt["failure"]["message"]
    assert len(pipes) == 1
    assert len(pipes[0].calls) == 3
    assert sum(name.startswith("outputs/") for name in names) == 3
    assert "receipt.json" in names


def test_packaging_failure_returns_receipt_only_with_original_run_source_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, payload, _runtime, _pipes, _download_calls = _configure_remote_endpoint(monkeypatch)
    original_pack = worker._pack_result
    pack_calls = 0

    def fail_first_pack(files: dict[str, bytes]) -> bytes:
        nonlocal pack_calls
        pack_calls += 1
        if pack_calls == 1:
            assert any(path.startswith("outputs/") for path in files)
            raise RuntimeError("result packaging failed: Bearer sentinel-package-value")
        return original_pack(files)

    monkeypatch.setattr(worker, "_pack_result", fail_first_pack)
    result = worker.run_remote_protected_comparison_inference(payload, "package-failure")

    receipt, names = _unpack_receipt(result)
    canonical_manifest = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert names == ["receipt.json"]
    assert pack_calls == 2
    assert receipt["status"] == "failed"
    assert receipt["run_id"] == "package-failure"
    assert receipt["source_sha"] == manifest["source_sha"]
    assert receipt["context"] == manifest["context"]
    assert receipt["manifest_sha256"] == hashlib.sha256(canonical_manifest).hexdigest()
    assert receipt["payload_sha256"] == hashlib.sha256(payload).hexdigest()
    assert receipt["generation_call_count"] == 52
    assert receipt["output_artifacts"] == []
    assert receipt["failure"]["stage"] == "result_packaging"
    assert receipt["failure_stage"] == "result_packaging"
    assert receipt["failure"]["message"] == "result packaging failed: Bearer [REDACTED]"
    assert "sentinel-package-value" not in receipt["failure"]["message"]
