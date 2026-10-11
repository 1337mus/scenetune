from __future__ import annotations

import builtins
import hashlib
import json
import sys
import zipfile
from contextlib import nullcontext
from io import BytesIO
from types import ModuleType, SimpleNamespace

from PIL import Image

from product_scene_editor import real_photo_comparison as comparison
from product_scene_editor import real_photo_comparison_inference as inference
from product_scene_editor import real_photo_inference as legacy_inference
from product_scene_editor import remote


def test_invalid_payload_emits_bounded_failure_before_runtime_import(monkeypatch) -> None:
    imports: list[str] = []
    original_import = builtins.__import__

    def tracked_import(name, *args, **kwargs):
        if name in {"torch", "huggingface_hub"}:
            imports.append(name)
            raise AssertionError("runtime import must follow payload validation")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", tracked_import)
    monkeypatch.setattr(
        comparison,
        "validate_payload_bytes",
        lambda *_args: (_ for _ in ()).throw(ValueError("invalid fixture payload")),
    )

    result = inference.run_remote_real_photo_comparison(b"invalid payload", "invalid-fixture")
    with zipfile.ZipFile(BytesIO(result)) as archive:
        receipt = json.loads(archive.read("receipt.json"))

    assert imports == []
    assert receipt["status"] == "failed"
    assert receipt["worker_failure_confirmed"] is True
    assert receipt["failure"]["stage"] == "input_validation"
    assert receipt["failure"]["type"] == "ValueError"
    assert receipt["failure"]["message"] == "invalid fixture payload"
    assert "geometry_review_sha256" not in receipt


def test_paired_worker_runs_four_strict_loads_and_six_pixel_exact_replays(monkeypatch) -> None:
    run_id = "paired-cpu-fixture"
    calls: list[tuple[str, tuple[int, int, int]]] = []
    strict_arms: list[str] = []
    cases = [
        {"case_id": case_id, "input_path": f"photos/{case_id}/input.png"}
        for case_id in comparison.CASE_IDS
    ]
    manifest = {field: {} for field in comparison._BINDING_FIELDS}
    manifest.update(
        profile=comparison.PROFILE,
        run_id=run_id,
        case_ids=list(comparison.CASE_IDS),
        cases=cases,
        generation_settings=legacy_inference.runner.GENERATION_SETTINGS,
        schedule=list(comparison.generation_schedule()),
        training_function_called=False,
        optimizer_updates=0,
        adapters={
            "prior": {
                "member_name": "adapter/step-144.safetensors",
                "payload_path": "adapters/prior/fake.safetensors",
            },
            "new": {
                "member_name": "adapter/step-144.safetensors",
                "payload_path": "adapters/new/fake.safetensors",
            },
        },
        prompt="fixture prompt",
    )
    for field in (
        "geometry_review_sha256",
        "geometry_review_status",
        "quality_review_status",
        "active_intake_sha256",
        "private_source_review_sha256",
    ):
        manifest[field] = "fixture"
    files = {
        "adapters/prior/fake.safetensors": b"prior adapter fixture",
        "adapters/new/fake.safetensors": b"new adapter fixture",
    }
    for index, case in enumerate(cases, start=1):
        image = Image.new("RGB", (512, 512), (index, index + 10, index + 20))
        stream = BytesIO()
        image.save(stream, format="PNG")
        files[case["input_path"]] = stream.getvalue()

    monkeypatch.setattr(
        comparison, "validate_payload_bytes", lambda payload, _run: (manifest, files)
    )
    monkeypatch.setattr(legacy_inference, "_runtime_versions", lambda: {"modal": "1.6.1"})
    monkeypatch.setattr(legacy_inference, "_validate_runtime_versions", lambda _versions: None)
    monkeypatch.setattr(
        legacy_inference, "_observed_diffsynth_commit", lambda: remote.DIFFSYNTH_COMMIT
    )
    monkeypatch.setattr(remote, "_download_model_snapshots", lambda _download: {})
    monkeypatch.setattr(
        remote, "_load_pipe", lambda _snapshots, _device: SimpleNamespace(arm="base")
    )
    monkeypatch.setattr(
        remote, "_generate", lambda pipe, _prompt, image, _seed: _generate(pipe, image, calls)
    )
    monkeypatch.setattr(
        "product_scene_editor.reload_diagnostic.validate_adapter_header", lambda _data: {}
    )

    validation = {
        "mode_changed": False,
        "adapter_validation": {
            "validated": True,
            "restored_tensor_count": 200,
            "module_count": 100,
        },
        "observed": {"pipe_training": False, "lora_dropout_training": [False, False]},
    }

    def load_replay(_snapshots, adapter_path, _torch):
        arm = adapter_path.stem
        strict_arms.append(arm)
        return object(), SimpleNamespace(arm=arm), validation

    monkeypatch.setattr("product_scene_editor.reload_diagnostic._load_replay_model", load_replay)

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        manual_seed_all=lambda _seed: None,
        reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 123,
        empty_cache=lambda: None,
    )
    torch.manual_seed = lambda _seed: None
    torch.inference_mode = nullcontext
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "huggingface_hub", ModuleType("huggingface_hub"))
    sys.modules["huggingface_hub"].snapshot_download = object()

    payload = b"paired cpu fixture payload"
    result = inference.run_remote_real_photo_comparison(payload, run_id)
    with zipfile.ZipFile(BytesIO(result)) as archive:
        result_files = {name: archive.read(name) for name in archive.namelist()}
    receipt = json.loads(result_files.pop("receipt.json"))

    assert receipt["status"] == "passed"
    assert strict_arms == ["prior", "prior", "new", "new"]
    assert [call[0] for call in calls] == [
        "base",
        "base",
        "base",
        "prior",
        "prior",
        "prior",
        "prior",
        "prior",
        "prior",
        "new",
        "new",
        "new",
        "new",
        "new",
        "new",
    ]
    assert len(result_files) == 15
    assert all(
        receipt["replay_pixel_equal"][case_id] == {"prior": True, "new": True}
        for case_id in comparison.CASE_IDS
    )
    assert receipt["training_function_called"] is False
    assert receipt["optimizer_updates"] == 0
    assert all(
        receipt[key] == manifest[key]
        for key in (
            "geometry_review_sha256",
            "geometry_review_status",
            "quality_review_status",
            "active_intake_sha256",
            "private_source_review_sha256",
        )
    )
    expected_plan = {**manifest, "payload_sha256": hashlib.sha256(payload).hexdigest()}
    assert comparison.clearance_bindings(receipt) == comparison.clearance_bindings(expected_plan)
    _checked_receipt, output_paths = comparison.validate_result_bundle(
        result,
        run_id,
        expected_plan=expected_plan,
        expected_payload_sha256=hashlib.sha256(payload).hexdigest(),
    )
    assert len(output_paths) == 15


def _generate(
    pipe, image: Image.Image, calls: list[tuple[str, tuple[int, int, int]]]
) -> Image.Image:
    source = image.getpixel((0, 0))
    calls.append((pipe.arm, source))
    offset = {"base": 0, "prior": 30, "new": 60}[pipe.arm]
    return Image.new("RGB", (512, 512), (source[0] + offset, source[1], source[2]))
