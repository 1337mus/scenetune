from __future__ import annotations

import hashlib
import json
import time
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from PIL import Image

from product_scene_editor import mask_preflight_inference as worker


def test_schedule_runs_24_sorted_primaries_and_replays_first_four_sources() -> None:
    manifest = {"cases": [{"case_id": f"case-{index:02d}"} for index in reversed(range(24))]}

    schedule = worker.generation_schedule(manifest)

    assert len(schedule) == 28
    assert [task["case_id"] for task in schedule[:24]] == [
        f"case-{index:02d}" for index in range(24)
    ]
    assert all(task["purpose"] == "primary" for task in schedule[:24])
    assert [task["case_id"] for task in schedule[24:]] == [
        "case-00",
        "case-01",
        "case-02",
        "case-03",
    ]
    assert [task["replay_of"] for task in schedule[24:]] == [0, 1, 2, 3]


def test_invalid_payload_returns_failure_before_runtime_import(monkeypatch) -> None:
    runtime_imports: list[bool] = []
    monkeypatch.setattr(
        worker,
        "_runtime_dependencies",
        lambda: runtime_imports.append(True),
        raising=False,
    )

    result = worker.run_remote_mask_preflight(b"invalid payload", "run-1")

    assert result
    assert runtime_imports == []


def test_packaging_only_failure_is_primary_failed_receipt_failure() -> None:
    packaging_failure = {
        "stage": "result_packaging",
        "type": "ValueError",
        "message": "result is too large",
    }

    receipt = worker._receipt(
        "failed",
        worker._new_execution_state(),
        run_id="run-1",
        started=time.monotonic(),
        payload_sha256="a" * 64,
        manifest={},
        runtime_versions={},
        peak_memory_bytes=None,
        failure=None,
        packaging_failure=packaging_failure,
    )

    assert receipt["failure"] == packaging_failure
    assert receipt["failure_stage"] == "result_packaging"
    assert receipt["packaging_failure"] == packaging_failure


def test_minimal_receipt_preserves_contract_and_original_failure() -> None:
    manifest = {
        "source_sha": "a" * 40,
        "context": {
            "profile": "reflex-personal",
            "workspace": "rajath-61258",
            "environment": "main",
        },
    }
    original_failure = {"stage": "runtime_import", "type": "RuntimeError", "message": "failed"}
    receipt_error = {
        "stage": "receipt_construction",
        "type": "ValueError",
        "message": "bad receipt",
    }

    raw = worker._minimal_receipt(
        "run-1",
        "b" * 64,
        time.monotonic(),
        "failed",
        original_failure,
        manifest=manifest,
        receipt_construction_failure=receipt_error,
    )
    receipt = json.loads(raw)

    assert receipt["source_sha"] == manifest["source_sha"]
    assert receipt["context"] == manifest["context"]
    assert (
        receipt["manifest_sha256"]
        == hashlib.sha256(worker.contract._canonical_json(manifest)).hexdigest()
    )
    assert receipt["model"] == worker.contract.MODEL
    assert receipt["model_files"] == worker.contract.MODEL_FILES
    assert receipt["settings"] == worker.contract.SETTINGS
    assert receipt["failure"] == original_failure
    assert receipt["receipt_construction_failure"] == receipt_error


def test_strict_load_requires_exact_pinned_tensor_value_count() -> None:
    record = {
        "validated": True,
        "strict": True,
        "tensor_count": 28_812,
        "tensor_value_count": 44_362_660,
        "missing_keys": [],
        "unexpected_keys": [],
        "bb_pretrained": False,
    }

    worker._validate_strict_load(record)
    record["tensor_value_count"] -= 1

    try:
        worker._validate_strict_load(record)
    except ValueError as exc:
        assert "strict safetensors load" in str(exc)
    else:
        raise AssertionError("strict load accepted a mismatched tensor-value count")


def test_replay_mismatch_keeps_partial_outputs_and_replay_evidence() -> None:
    cases = [
        {"case_id": f"case-{index:02d}", "source_rgb": {"path": f"source-{index}"}}
        for index in range(24)
    ]
    manifest = {"cases": cases}
    files = {f"source-{index}": b"unused" for index in range(24)}
    state = worker._new_execution_state()
    calls = 0

    def infer(_model, _source):
        nonlocal calls
        calls += 1
        alpha = Image.new("L", (512, 512), 255)
        if calls == 25:
            alpha.putpixel((0, 0), 0)
        return alpha, alpha.filter(worker.ImageFilter.MinFilter(5))

    worker._execute_schedule(
        manifest,
        files,
        worker.generation_schedule(manifest),
        SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
        lambda: (
            object(),
            {
                "validated": True,
                "strict": True,
                "tensor_count": 28_812,
                "tensor_value_count": 44_362_660,
                "missing_keys": [],
                "unexpected_keys": [],
                "bb_pretrained": False,
            },
        ),
        infer,
        state,
    )

    assert state["status"] == "failed"
    assert state["failure"]["stage"] == "replay_inference_case-00"
    assert len(state["outputs"]) == 50
    assert state["inference_call_count"] == 25
    assert state["replay_checks"] == [
        {
            "case_id": "case-00",
            "primary_index": 0,
            "replay_index": 24,
            "alpha_pixel_equal": False,
            "core_pixel_equal": False,
        }
    ]


def test_snapshot_verification_checks_every_hash_and_false_backbone_config(
    tmp_path: Path, monkeypatch
) -> None:
    contents = {
        "BiRefNet_config.py": b"class BiRefNetConfig: pass\n",
        "birefnet.py": b"class BiRefNet: pass\n",
        "config.json": b'{"bb_pretrained": false}',
        "model.safetensors": b"fixture weights",
    }
    pins = {
        name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in contents.items()
    }
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name, data in contents.items():
        (snapshot / name).write_bytes(data)
    monkeypatch.setattr(worker.contract, "MODEL_FILES", pins)

    assert worker._verify_model_snapshot(snapshot) == {"bb_pretrained": False}

    (snapshot / "birefnet.py").write_bytes(b"tampered source")
    try:
        worker._verify_model_snapshot(snapshot)
    except ValueError as exc:
        assert "size or SHA-256" in str(exc)
    else:
        raise AssertionError("snapshot verification accepted a tampered pinned file")

    true_config = b'{"bb_pretrained": true}'
    (snapshot / "config.json").write_bytes(true_config)
    tampered_source = b"tampered source"
    pins["birefnet.py"] = {
        "size": len(tampered_source),
        "sha256": hashlib.sha256(tampered_source).hexdigest(),
    }
    pins["config.json"] = {
        "size": len(true_config),
        "sha256": hashlib.sha256(true_config).hexdigest(),
    }
    try:
        worker._verify_model_snapshot(snapshot)
    except ValueError as exc:
        assert "bb_pretrained" in str(exc)
    else:
        raise AssertionError("snapshot verification accepted bb_pretrained=true")


def test_verified_loader_records_strict_cpu_tensor_value_count(tmp_path: Path, monkeypatch) -> None:
    contents = {
        "BiRefNet_config.py": (
            b"class BiRefNetConfig:\n"
            b"    def __init__(self, bb_pretrained=False):\n"
            b"        self.bb_pretrained = bb_pretrained\n"
        ),
        "birefnet.py": (
            b"class BiRefNet:\n"
            b"    def __init__(self, config):\n"
            b"        self.config = config\n"
            b"    def load_state_dict(self, state, strict):\n"
            b"        self.state, self.strict = state, strict\n"
            b"        return type('LoadResult', (), {\n"
            b"            'missing_keys': [], 'unexpected_keys': []})()\n"
            b"    def to(self, device, dtype):\n"
            b"        self.device, self.dtype = device, dtype\n"
            b"        return self\n"
            b"    def eval(self):\n"
            b"        self.is_eval = True\n"
        ),
        "config.json": b'{"bb_pretrained": false}',
        "model.safetensors": b"fixture weights",
    }
    pins = {
        name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in contents.items()
    }
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name, data in contents.items():
        (snapshot / name).write_bytes(data)
    monkeypatch.setattr(worker.contract, "MODEL_FILES", pins)
    loaded: list[tuple[str, str]] = []

    class Weight:
        def numel(self) -> int:
            return worker.EXPECTED_MODEL_TENSOR_VALUE_COUNT

    def load_file(_path: str, *, device: str):
        loaded.append(("model.safetensors", device))
        return {"weight": Weight()}

    model, record = worker._load_verified_model(
        snapshot, SimpleNamespace(float32="float32"), load_file
    )

    assert record["validated"] is True
    assert record["strict"] is True
    assert record["tensor_count"] == 1
    assert record["tensor_value_count"] == worker.EXPECTED_MODEL_TENSOR_VALUE_COUNT
    assert record["bb_pretrained"] is False
    assert loaded == [("model.safetensors", "cpu")]
    assert model.strict is True
    assert model.config.bb_pretrained is False
    assert model.device == "cuda"
    assert model.is_eval is True


def test_probability_tensor_requires_pinned_shape_and_finite_values() -> None:
    class Tensor:
        shape = (1, 1, 1024, 1024)

    with pytest.raises(ValueError, match="1x1x1024x1024"):
        worker._validate_probability_tensor(
            SimpleNamespace(shape=(1, 1, 512, 512)), is_finite=lambda _tensor: True
        )
    with pytest.raises(ValueError, match="non-finite"):
        worker._validate_probability_tensor(Tensor(), is_finite=lambda _tensor: False)


def test_packaging_failure_returns_failed_receipt_with_runtime_evidence(monkeypatch) -> None:
    manifest = {
        "source_sha": "a" * 40,
        "context": worker.contract.CONTEXT,
        "cases": [
            {
                "case_id": f"case-{index:02d}",
                "source_rgb": {"path": f"cases/case-{index:02d}/source.png"},
            }
            for index in range(24)
        ],
    }
    files = {f"cases/case-{index:02d}/source.png": b"unused" for index in range(24)}
    monkeypatch.setattr(
        worker.contract, "validate_payload_bytes", lambda _payload: (manifest, files)
    )
    runtime = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_name=lambda _index: "Fake CUDA GPU",
            reset_peak_memory_stats=lambda: None,
            max_memory_allocated=lambda: 2048,
            empty_cache=lambda: None,
        ),
        inference_mode=nullcontext,
    )
    monkeypatch.setattr(worker, "_runtime_dependencies", lambda: (runtime, None, None, None))
    monkeypatch.setattr(worker, "_runtime_versions", lambda: worker.contract.RUNTIME_PINS)
    monkeypatch.setattr(worker, "_load_snapshot", lambda _download: Path("/fake-snapshot"))
    load_record = {
        "validated": True,
        "strict": True,
        "tensor_count": 28_812,
        "tensor_value_count": 44_362_660,
        "missing_keys": [],
        "unexpected_keys": [],
        "bb_pretrained": False,
    }
    monkeypatch.setattr(
        worker, "_load_verified_model", lambda *_args: (object(), load_record.copy())
    )
    alpha = Image.new("L", (512, 512), 255)
    core = alpha.copy()
    monkeypatch.setattr(worker, "_predict_masks", lambda *_args: (alpha, core))
    original_pack = worker._pack_result
    pack_calls = 0

    def fail_first_pack(result_files):
        nonlocal pack_calls
        pack_calls += 1
        if pack_calls == 1:
            raise ValueError("result exceeded the packaging limit")
        return original_pack(result_files)

    monkeypatch.setattr(worker, "_pack_result", fail_first_pack)

    result = worker.run_remote_mask_preflight(b"payload", "run-1")
    with ZipFile(BytesIO(result)) as archive:
        receipt = json.loads(archive.read("receipt.json"))

    assert pack_calls == 2
    assert receipt["status"] == "failed"
    assert receipt["failure"]["stage"] == "result_packaging"
    assert receipt["failure_stage"] == "result_packaging"
    assert receipt["packaging_failure"] == receipt["failure"]
    assert receipt["gpu"] == {"device_count": 1, "name": "Fake CUDA GPU"}
    assert [load["tensor_value_count"] for load in receipt["strict_loads"]] == [
        44_362_660,
        44_362_660,
    ]


def test_receipt_construction_fallback_preserves_original_failure_and_pins(monkeypatch) -> None:
    manifest = {
        "source_sha": "c" * 40,
        "context": worker.contract.CONTEXT,
        "cases": [],
    }
    monkeypatch.setattr(worker.contract, "validate_payload_bytes", lambda _payload: (manifest, {}))

    def fail_runtime_import():
        raise RuntimeError("runtime unavailable")

    monkeypatch.setattr(worker, "_runtime_dependencies", fail_runtime_import)
    monkeypatch.setattr(
        worker,
        "_receipt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("receipt construction failed")),
    )

    result = worker.run_remote_mask_preflight(b"payload", "run-2")
    with ZipFile(BytesIO(result)) as archive:
        receipt = json.loads(archive.read("receipt.json"))

    assert receipt["status"] == "failed"
    assert receipt["source_sha"] == manifest["source_sha"]
    assert receipt["context"] == manifest["context"]
    assert (
        receipt["manifest_sha256"]
        == hashlib.sha256(worker.contract._canonical_json(manifest)).hexdigest()
    )
    assert receipt["model"] == worker.contract.MODEL
    assert receipt["model_files"] == worker.contract.MODEL_FILES
    assert receipt["settings"] == worker.contract.SETTINGS
    assert receipt["failure"]["stage"] == "runtime_import"
    assert receipt["receipt_construction_failure"]["stage"] == "receipt_construction"
