"""Inference-only worker for draft BiRefNet masks; no model dependency loads on import."""

from __future__ import annotations

import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import stat
import sys
import time
import types
import zipfile
from collections.abc import Callable
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from typing import Any, TypedDict

from PIL import Image, ImageFilter

from . import mask_preflight as contract
from . import remote

MAX_RESULT_BYTES = contract.MAX_RESULT_BYTES
EXPECTED_MODEL_TENSOR_VALUE_COUNT = 44_362_703
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SAFE_RESULT_MEMBER = re.compile(
    r"^(?:outputs|replays)/[A-Za-z0-9][A-Za-z0-9._-]{0,63}/(?:alpha|core)\.png$"
)


class GenerationTask(TypedDict):
    purpose: str
    case_id: str
    index: int
    replay_of: int | None


def generation_schedule(manifest: dict[str, Any]) -> tuple[GenerationTask, ...]:
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != 24:
        raise ValueError("mask preflight schedule requires exactly 24 cases")
    case_ids = [case.get("case_id") if isinstance(case, dict) else None for case in cases]
    if any(
        not isinstance(case_id, str) or _SAFE_ID.fullmatch(case_id) is None for case_id in case_ids
    ):
        raise ValueError("mask preflight schedule contains an invalid case ID")
    ordered = sorted(str(case_id) for case_id in case_ids)
    if len(set(ordered)) != 24:
        raise ValueError("mask preflight case IDs must be unique")
    tasks: list[GenerationTask] = [
        {"purpose": "primary", "case_id": case_id, "index": index, "replay_of": None}
        for index, case_id in enumerate(ordered)
    ]
    tasks.extend(
        {
            "purpose": "fresh_load_replay",
            "case_id": case_id,
            "index": len(tasks),
            "replay_of": index,
        }
        for index, case_id in enumerate(ordered[:4])
    )
    return tuple(tasks)


def _new_execution_state() -> dict[str, Any]:
    return {
        "status": "pending",
        "inference_call_count": 0,
        "generation_call_timings": [],
        "strict_loads": [],
        "gpu": None,
        "replay_checks": [],
        "generation_records": [],
        "outputs": {},
        "failure": None,
    }


def _validate_probability_tensor(probability: Any, *, is_finite: Callable[[Any], bool]) -> None:
    if tuple(getattr(probability, "shape", ())) != (1, 1, 1024, 1024):
        raise ValueError("BiRefNet final probability tensor must be 1x1x1024x1024")
    if not is_finite(probability):
        raise ValueError("BiRefNet probability tensor contains non-finite values")


def _validate_masks(alpha: Image.Image, core: Image.Image) -> None:
    if alpha.mode != "L" or core.mode != "L" or alpha.size != (512, 512) or core.size != (512, 512):
        raise ValueError("draft alpha and protected core must be 512x512 L images")
    alpha_pixels, core_pixels = alpha.tobytes(), core.tobytes()
    if any(value not in (0, 255) for value in alpha_pixels + core_pixels):
        raise ValueError("draft alpha and protected core must be binary masks")
    if 255 not in alpha_pixels or 255 not in core_pixels:
        raise ValueError("draft alpha and protected core must both be nonempty")
    if any(
        core_value and not alpha_value
        for core_value, alpha_value in zip(core_pixels, alpha_pixels, strict=True)
    ):
        raise ValueError("protected core must be a subset of the draft alpha mask")


def _png_bytes(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG", compress_level=6, optimize=False)
    return output.getvalue()


def _predict_masks(
    model: Any, source_png: bytes, torch: Any, numpy: Any
) -> tuple[Image.Image, Image.Image]:
    with Image.open(BytesIO(source_png)) as opened:
        source = opened.convert("RGB")
    resized = source.resize((1024, 1024), resample=Image.Resampling.BILINEAR)
    pixels = numpy.asarray(resized, dtype=numpy.float32) / numpy.float32(255.0)
    tensor = (
        torch.from_numpy(pixels)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device="cuda", dtype=torch.float32)
    )
    mean = torch.tensor(contract.SETTINGS["normalization_mean"], dtype=torch.float32, device="cuda")
    std = torch.tensor(contract.SETTINGS["normalization_std"], dtype=torch.float32, device="cuda")
    tensor = (tensor - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
    outputs = model(tensor)
    if not isinstance(outputs, (list, tuple)) or not outputs:
        raise ValueError("BiRefNet must return a nonempty sequence of predictions")
    probability = outputs[-1].sigmoid()
    _validate_probability_tensor(
        probability, is_finite=lambda value: bool(torch.isfinite(value).all().item())
    )
    probability = torch.nn.functional.interpolate(
        probability, size=(512, 512), mode="bilinear", align_corners=False
    )[0, 0]
    if not bool(torch.isfinite(probability).all().item()):
        raise ValueError("resized BiRefNet probability contains non-finite values")
    binary = (probability >= contract.SETTINGS["alpha_threshold"]).to(torch.uint8) * 255
    mask_values = binary.detach().cpu().numpy().astype(numpy.uint8, copy=False)
    alpha = Image.fromarray(mask_values, mode="L")
    core = alpha.filter(ImageFilter.MinFilter(5))
    _validate_masks(alpha, core)
    return alpha, core


def _validate_strict_load(record: dict[str, Any]) -> None:
    if (
        record.get("validated") is not True
        or record.get("strict") is not True
        or type(record.get("tensor_count")) is not int
        or record["tensor_count"] <= 0
        or type(record.get("tensor_value_count")) is not int
        or record["tensor_value_count"] != EXPECTED_MODEL_TENSOR_VALUE_COUNT
        or record.get("missing_keys") != []
        or record.get("unexpected_keys") != []
        or record.get("bb_pretrained") is not False
    ):
        raise ValueError("BiRefNet strict safetensors load did not satisfy the pinned contract")


def _load_observed_model(
    model_loader: Callable[[], tuple[Any, dict[str, Any]]], state: dict[str, Any]
) -> Any:
    observed_count = len(state["strict_loads"])
    model, record = model_loader()
    if len(state["strict_loads"]) == observed_count or state["strict_loads"][-1] is not record:
        state["strict_loads"].append(record)
    _validate_strict_load(record)
    return model


def _execute_schedule(
    manifest: dict[str, Any],
    files: dict[str, bytes],
    schedule: tuple[GenerationTask, ...],
    runtime: Any,
    model_loader: Callable[[], tuple[Any, dict[str, Any]]],
    infer_masks: Callable[[Any, bytes], tuple[Image.Image, Image.Image]],
    state: dict[str, Any],
) -> None:
    cases = {case["case_id"]: case for case in manifest["cases"]}
    primary = [task for task in schedule if task["purpose"] == "primary"]
    replays = [task for task in schedule if task["purpose"] == "fresh_load_replay"]
    if len(schedule) != 28 or len(primary) != 24 or len(replays) != 4:
        raise ValueError("mask preflight schedule must have 24 primaries and four replays")
    primary_masks: dict[str, tuple[bytes, bytes]] = {}
    model: Any | None = None
    stage = "strict_model_load"
    try:
        model = _load_observed_model(model_loader, state)
        for task in primary:
            case = cases[task["case_id"]]
            path = case["source_rgb"]["path"]
            stage = f"primary_inference_{task['case_id']}"
            started = time.monotonic()
            state["inference_call_count"] += 1
            inference_context = (
                runtime.inference_mode() if hasattr(runtime, "inference_mode") else nullcontext()
            )
            try:
                with inference_context:
                    alpha, core = infer_masks(model, files[path])
            finally:
                state["generation_call_timings"].append(
                    {
                        "purpose": task["purpose"],
                        "case_id": task["case_id"],
                        "index": task["index"],
                        "elapsed_seconds": round(time.monotonic() - started, 6),
                    }
                )
            _validate_masks(alpha, core)
            alpha_bytes, core_bytes = _png_bytes(alpha), _png_bytes(core)
            alpha_path = f"outputs/{task['case_id']}/alpha.png"
            core_path = f"outputs/{task['case_id']}/core.png"
            state["outputs"][alpha_path], state["outputs"][core_path] = alpha_bytes, core_bytes
            state["generation_records"].append(
                {
                    "purpose": task["purpose"],
                    "case_id": task["case_id"],
                    "index": task["index"],
                    "alpha_path": alpha_path,
                    "alpha_sha256": hashlib.sha256(alpha_bytes).hexdigest(),
                    "core_path": core_path,
                    "core_sha256": hashlib.sha256(core_bytes).hexdigest(),
                }
            )
            primary_masks[task["case_id"]] = (alpha.tobytes(), core.tobytes())
        del model
        model = None
        gc.collect()
        runtime.cuda.empty_cache()

        stage = "fresh_strict_model_load"
        model = _load_observed_model(model_loader, state)
        for task in replays:
            case = cases[task["case_id"]]
            path = case["source_rgb"]["path"]
            stage = f"replay_inference_{task['case_id']}"
            started = time.monotonic()
            state["inference_call_count"] += 1
            inference_context = (
                runtime.inference_mode() if hasattr(runtime, "inference_mode") else nullcontext()
            )
            try:
                with inference_context:
                    alpha, core = infer_masks(model, files[path])
            finally:
                state["generation_call_timings"].append(
                    {
                        "purpose": task["purpose"],
                        "case_id": task["case_id"],
                        "index": task["index"],
                        "elapsed_seconds": round(time.monotonic() - started, 6),
                    }
                )
            _validate_masks(alpha, core)
            alpha_bytes, core_bytes = _png_bytes(alpha), _png_bytes(core)
            alpha_path = f"replays/{task['case_id']}/alpha.png"
            core_path = f"replays/{task['case_id']}/core.png"
            state["outputs"][alpha_path], state["outputs"][core_path] = alpha_bytes, core_bytes
            expected_alpha, expected_core = primary_masks[task["case_id"]]
            alpha_equal = alpha.tobytes() == expected_alpha
            core_equal = core.tobytes() == expected_core
            state["replay_checks"].append(
                {
                    "case_id": task["case_id"],
                    "primary_index": task["replay_of"],
                    "replay_index": task["index"],
                    "alpha_pixel_equal": alpha_equal,
                    "core_pixel_equal": core_equal,
                }
            )
            state["generation_records"].append(
                {
                    "purpose": task["purpose"],
                    "case_id": task["case_id"],
                    "index": task["index"],
                    "alpha_path": alpha_path,
                    "alpha_sha256": hashlib.sha256(alpha_bytes).hexdigest(),
                    "core_path": core_path,
                    "core_sha256": hashlib.sha256(core_bytes).hexdigest(),
                }
            )
            if not alpha_equal or not core_equal:
                raise ValueError("fresh-model BiRefNet replay masks differ from primary masks")
        state["status"] = "passed"
    except Exception as exc:
        state["status"] = "failed"
        state["failure"] = {
            "stage": stage,
            "type": type(exc).__name__,
            "message": remote.sanitize_failure(exc),
        }
    finally:
        if model is not None:
            del model
        gc.collect()


def _runtime_dependencies() -> tuple[Any, Any, Any, Any]:
    import numpy  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    from safetensors.torch import load_file  # type: ignore[import-not-found]

    return torch, numpy, snapshot_download, load_file


def _runtime_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in contract.RUNTIME_PINS}


def _validate_runtime_versions(versions: dict[str, str]) -> None:
    if versions != contract.RUNTIME_PINS:
        raise RuntimeError("mask preflight worker versions differ from frozen runtime pins")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_model_snapshot(snapshot: Path) -> dict[str, Any]:
    if set(path.name for path in snapshot.iterdir() if path.is_file()) != set(contract.MODEL_FILES):
        raise ValueError("BiRefNet snapshot contains unexpected or missing files")
    for name, expected in contract.MODEL_FILES.items():
        path = snapshot / name
        if (
            not path.is_file()
            or path.stat().st_size != expected["size"]
            or _hash_file(path) != expected["sha256"]
        ):
            raise ValueError(f"BiRefNet pinned file failed size or SHA-256 verification: {name}")
    try:
        config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("BiRefNet config.json is not valid UTF-8 JSON") from exc
    if not isinstance(config, dict) or config.get("bb_pretrained") is not False:
        raise ValueError("BiRefNet config must set bb_pretrained to the boolean false")
    return config


def _load_verified_model(
    snapshot: Path,
    torch: Any,
    load_file: Any,
    *,
    record_observer: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Any, dict[str, Any]]:
    _verify_model_snapshot(snapshot)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    package_name = f"_mask_preflight_birefnet_{time.monotonic_ns()}"
    package = types.ModuleType(package_name)
    package.__path__ = [str(snapshot)]
    package.__package__ = package_name
    sys.modules[package_name] = package
    try:
        config_module = importlib.import_module(f"{package_name}.BiRefNet_config")
        model_module = importlib.import_module(f"{package_name}.birefnet")
        config = config_module.BiRefNetConfig(bb_pretrained=False)
        model = model_module.BiRefNet(config=config)
        tensor_state = load_file(str(snapshot / "model.safetensors"), device="cpu")
        tensor_count = len(tensor_state)
        tensor_value_count = sum(tensor.numel() for tensor in tensor_state.values())
        try:
            loaded = model.load_state_dict(tensor_state, strict=True)
        except RuntimeError as exc:
            record = {
                "validated": False,
                "strict": True,
                "tensor_count": tensor_count,
                "tensor_value_count": tensor_value_count,
                "missing_keys": None,
                "unexpected_keys": None,
                "bb_pretrained": False,
                "load_state_dict_failure": remote.sanitize_failure(exc),
            }
            if record_observer is not None:
                record_observer(record)
            raise
        record = {
            "validated": True,
            "strict": True,
            "tensor_count": tensor_count,
            "tensor_value_count": tensor_value_count,
            "missing_keys": list(loaded.missing_keys),
            "unexpected_keys": list(loaded.unexpected_keys),
            "bb_pretrained": False,
        }
        if record_observer is not None:
            record_observer(record)
        _validate_strict_load(record)
        model = model.to(device="cuda", dtype=torch.float32)
        model.eval()
        return model, record
    finally:
        for name in list(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                sys.modules.pop(name, None)


def _load_snapshot(snapshot_download: Any) -> Path:
    snapshot = Path(
        snapshot_download(
            repo_id=contract.MODEL["repo_id"],
            revision=contract.MODEL["revision"],
            allow_patterns=list(contract.MODEL_FILES),
            token=False,
        )
    ).resolve()
    if snapshot.name != contract.MODEL["revision"]:
        raise RuntimeError("BiRefNet snapshot did not resolve to the pinned revision")
    _verify_model_snapshot(snapshot)
    return snapshot


def _pack_result(files: dict[str, bytes]) -> bytes:
    for name, content in files.items():
        if name != "receipt.json" and _SAFE_RESULT_MEMBER.fullmatch(name) is None:
            raise ValueError("mask preflight result contains an unsafe member")
        if not isinstance(content, bytes):
            raise ValueError("mask preflight result members must be bytes")
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            archive.writestr(info, content)
    result = output.getvalue()
    if len(result) > MAX_RESULT_BYTES:
        raise ValueError("mask preflight result exceeds the 16 MiB limit")
    return result


def _receipt(
    status: str,
    state: dict[str, Any],
    *,
    run_id: str,
    started: float,
    payload_sha256: str,
    manifest: dict[str, Any],
    runtime_versions: dict[str, str],
    peak_memory_bytes: int | None,
    failure: dict[str, str] | None,
    packaging_failure: dict[str, str] | None = None,
) -> dict[str, Any]:
    effective_failure = failure or (state.get("failure") if status == "failed" else None)
    if status == "failed" and effective_failure is None:
        effective_failure = packaging_failure
    artifacts: list[dict[str, Any]] = []
    for path, content in sorted(state["outputs"].items()):
        with Image.open(BytesIO(content)) as opened:
            if opened.format != "PNG" or opened.mode != "L" or opened.size != (512, 512):
                raise ValueError(f"mask output {path} is not a 512x512 L PNG")
        artifacts.append(
            {
                "path": path,
                "sha256": hashlib.sha256(content).hexdigest(),
                "width": 512,
                "height": 512,
            }
        )
    receipt = {
        "schema_version": 1,
        "run_id": run_id,
        "status": status,
        "source_sha": manifest.get("source_sha"),
        "payload_sha256": payload_sha256,
        "manifest_sha256": hashlib.sha256(contract._canonical_json(manifest)).hexdigest()
        if manifest
        else None,
        "context": manifest.get("context"),
        "model": contract.MODEL,
        "model_files": contract.MODEL_FILES,
        "runtime_pins": contract.RUNTIME_PINS,
        "runtime_versions": runtime_versions,
        "settings": contract.SETTINGS,
        "generation_schedule": list(generation_schedule(manifest)) if manifest else [],
        "inference_call_count": state["inference_call_count"],
        "generation_call_timings": state["generation_call_timings"],
        "strict_loads": state["strict_loads"],
        "gpu": state.get("gpu"),
        "replay_checks": state["replay_checks"],
        "generation_records": state["generation_records"],
        "output_artifacts": artifacts,
        "mask_status": "draft_unaccepted",
        "training_function_called": False,
        "optimizer_updates": 0,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "peak_allocated_bytes": peak_memory_bytes,
        "failure": effective_failure,
        "failure_stage": effective_failure.get("stage") if effective_failure else None,
    }
    if packaging_failure is not None:
        receipt["packaging_failure"] = packaging_failure
    return receipt


def _safe_failure(stage: str, exc: BaseException) -> dict[str, str]:
    return {"stage": stage, "type": type(exc).__name__, "message": remote.sanitize_failure(exc)}


def _minimal_receipt(
    run_id: str,
    payload_sha256: str,
    started: float,
    status: str,
    failure: dict[str, str],
    *,
    manifest: dict[str, Any] | None = None,
    runtime_versions: dict[str, str] | None = None,
    peak_memory_bytes: int | None = None,
    strict_loads: list[dict[str, Any]] | None = None,
    gpu: dict[str, Any] | None = None,
    receipt_construction_failure: dict[str, str] | None = None,
) -> bytes:
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "status": status,
        "source_sha": manifest.get("source_sha") if manifest else None,
        "payload_sha256": payload_sha256,
        "manifest_sha256": hashlib.sha256(contract._canonical_json(manifest)).hexdigest()
        if manifest
        else None,
        "context": manifest.get("context") if manifest else None,
        "model": contract.MODEL,
        "model_files": contract.MODEL_FILES,
        "runtime_pins": contract.RUNTIME_PINS,
        "runtime_versions": runtime_versions or {},
        "settings": contract.SETTINGS,
        "strict_loads": strict_loads or [],
        "gpu": gpu,
        "mask_status": "draft_unaccepted",
        "training_function_called": False,
        "optimizer_updates": 0,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "peak_allocated_bytes": peak_memory_bytes,
        "failure": failure,
        "failure_stage": failure.get("stage"),
    }
    if receipt_construction_failure is not None:
        receipt["receipt_construction_failure"] = receipt_construction_failure
    return json.dumps(receipt, sort_keys=True).encode("utf-8")


def run_remote_mask_preflight(payload_bytes: bytes, run_id: str) -> bytes:
    """Validate first, generate draft masks, replay four sources after a strict fresh load."""
    if not isinstance(run_id, str) or _SAFE_ID.fullmatch(run_id) is None:
        raise ValueError("run_id must be a safe identifier")
    started = time.monotonic()
    stage = "input_validation"
    payload_sha256 = (
        hashlib.sha256(payload_bytes).hexdigest()
        if isinstance(payload_bytes, bytes)
        else hashlib.sha256(b"").hexdigest()
    )
    manifest: dict[str, Any] = {}
    runtime_versions: dict[str, str] = {}
    state = _new_execution_state()
    runtime: Any | None = None
    peak_memory: int | None = None
    failure: dict[str, str] | None = None
    status = "failed"
    try:
        manifest, files = contract.validate_payload_bytes(payload_bytes)
        stage = "runtime_import"
        runtime, numpy, snapshot_download, load_file = _runtime_dependencies()
        runtime_versions = _runtime_versions()
        _validate_runtime_versions(runtime_versions)
        stage = "gpu_validation"
        if not runtime.cuda.is_available():
            raise RuntimeError("mask preflight requires exactly one available CUDA GPU")
        device_count = int(runtime.cuda.device_count())
        if device_count != 1:
            raise RuntimeError("mask preflight requires exactly one available CUDA GPU")
        state["gpu"] = {
            "device_count": device_count,
            "name": str(runtime.cuda.get_device_name(0)),
        }
        runtime.cuda.reset_peak_memory_stats()
        stage = "model_snapshot"
        snapshot = _load_snapshot(snapshot_download)
        stage = "inference"

        def model_loader() -> tuple[Any, dict[str, Any]]:
            return _load_verified_model(
                snapshot, runtime, load_file, record_observer=state["strict_loads"].append
            )

        def infer_masks(model: Any, source: bytes) -> tuple[Image.Image, Image.Image]:
            return _predict_masks(model, source, runtime, numpy)

        _execute_schedule(
            manifest,
            files,
            generation_schedule(manifest),
            runtime,
            model_loader,
            infer_masks,
            state,
        )
        status = state["status"]
        if status == "failed":
            failure = state["failure"]
            stage = failure["stage"]
    except Exception as exc:
        failure = _safe_failure(stage, exc)
        state["status"] = "failed"
    finally:
        if runtime is not None:
            try:
                peak_memory = int(runtime.cuda.max_memory_allocated())
            except Exception as exc:
                if failure is None:
                    stage = "peak_memory_capture"
                    failure = _safe_failure(stage, exc)
                    status = "failed"
            gc.collect()
            try:
                runtime.cuda.empty_cache()
            except Exception:
                pass
    try:
        receipt = _receipt(
            status,
            state,
            run_id=run_id,
            started=started,
            payload_sha256=payload_sha256,
            manifest=manifest,
            runtime_versions=runtime_versions,
            peak_memory_bytes=peak_memory,
            failure=failure,
        )
        receipt_bytes = json.dumps(receipt, sort_keys=True).encode()
        result_files = {**state["outputs"], "receipt.json": receipt_bytes}
    except Exception as exc:
        receipt_error = _safe_failure("receipt_construction", exc)
        preserved_failure = failure or state.get("failure") or receipt_error
        minimal = _minimal_receipt(
            run_id,
            payload_sha256,
            started,
            "failed",
            preserved_failure,
            manifest=manifest,
            runtime_versions=runtime_versions,
            peak_memory_bytes=peak_memory,
            strict_loads=state["strict_loads"],
            gpu=state["gpu"],
            receipt_construction_failure=receipt_error,
        )
        return _pack_result({"receipt.json": minimal})
    try:
        return _pack_result(result_files)
    except Exception as exc:
        packaging_failure = _safe_failure("result_packaging", exc)
        preserved_failure = failure or state.get("failure")
        fallback = _receipt(
            "failed",
            {**state, "outputs": {}},
            run_id=run_id,
            started=started,
            payload_sha256=payload_sha256,
            manifest=manifest,
            runtime_versions=runtime_versions,
            peak_memory_bytes=peak_memory,
            failure=preserved_failure,
            packaging_failure=packaging_failure,
        )
        return _pack_result({"receipt.json": json.dumps(fallback, sort_keys=True).encode()})
