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
