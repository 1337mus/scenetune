"""Inference-only worker for the private real-photo comparison.

Modal and CUDA packages are imported only inside the remote entry point. This module has no
trainer, optimizer, or training-loop invocation.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import shutil
import subprocess
import tempfile
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from . import real_photo_runner as runner
from . import remote


def _png_bytes(image: Any) -> bytes:
    output = BytesIO()
    image.convert("RGB").save(output, format="PNG", compress_level=6, optimize=False)
    return output.getvalue()


def _pack_result(files: dict[str, bytes]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, content)
    payload = buffer.getvalue()
    if len(payload) > runner.MAX_RESULT_BYTES:
        raise ValueError("real-photo result exceeds the 16 MiB result limit")
    return payload


def _observed_diffsynth_commit() -> str:
    completed = subprocess.run(
        ["git", "-C", "/opt/DiffSynth-Studio", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    commit = completed.stdout.strip()
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise RuntimeError("DiffSynth checkout returned an invalid commit identity")
    return commit


def _runtime_versions() -> dict[str, str]:
    packages = {"modal": "modal", **{name: name for name in remote.RUNTIME_PINS}}
    return {key: importlib.metadata.version(package) for key, package in packages.items()}


def _validate_runtime_versions(versions: dict[str, str]) -> None:
    expected = {"modal": "1.6.1", **remote.RUNTIME_PINS}
    if versions != expected:
        raise RuntimeError("worker package versions differ from the frozen runtime pins")


def _validate_strict_load(record: dict[str, Any]) -> None:
    if (
        record.get("mode_changed") is not False
        or record.get("adapter_validation")
        != {"validated": True, "restored_tensor_count": 200, "module_count": 100}
        or not isinstance(record.get("observed"), dict)
        or set(record["observed"]) != {"pipe_training", "lora_dropout_training"}
        or not isinstance(record["observed"].get("pipe_training"), bool)
        or not isinstance(record["observed"].get("lora_dropout_training"), list)
        or any(not isinstance(mode, bool) for mode in record["observed"]["lora_dropout_training"])
    ):
        raise ValueError("strict adapter load did not satisfy the audited frozen validation")


def _receipt(
    run_id: str,
    status: str,
    outputs: dict[str, bytes],
    *,
    started: float,
    stage: str,
    replay_equal: bool | None,
    manifest: dict[str, Any],
    payload_sha256: str,
    runtime_versions: dict[str, str],
    observed_diffsynth_commit: str | None,
    strict_load_validations: list[dict[str, Any]],
    failure: BaseException | None = None,
    peak_memory_bytes: int | None = None,
) -> dict[str, Any]:
    output_artifacts = []
    for path, content in sorted(outputs.items()):
        with Image.open(BytesIO(content)) as image:
            width, height = image.size
            if image.mode != "RGB" or image.size != (512, 512):
                raise ValueError(f"worker output {path} is not a 512x512 RGB PNG")
        output_artifacts.append(
            {
                "path": path,
                "sha256": hashlib.sha256(content).hexdigest(),
                "width": width,
                "height": height,
            }
        )
    value: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "status": status,
        "generation_settings": runner.GENERATION_SETTINGS,
        "schedule": list(runner.generation_schedule()),
        "training_function_called": False,
        "optimizer_updates": 0,
        "output_artifacts": output_artifacts,
        "replay_pixel_equal": replay_equal,
        "quality_claim": False,
        "worker_failure_confirmed": status == "failed",
        "payload_sha256": payload_sha256,
        "preparation_sha256": manifest.get("preparation_sha256"),
        "geometry_review_sha256": manifest.get("geometry_review_sha256"),
        "geometry_review_status": manifest.get("geometry_review_status"),
        "active_intake_sha256": manifest.get("active_intake_sha256"),
        "label_confirmation_sha256": manifest.get("label_confirmation_sha256"),
        "private_source_review_sha256": manifest.get("private_source_review_sha256"),
        "quality_review_status": manifest.get("quality_review_status"),
        "adapter_archive_sha256": manifest.get("adapter_archive_sha256"),
        "adapter_member_sha256": manifest.get("adapter_member_sha256"),
        "modal_sdk": manifest.get("modal_sdk"),
        "diffsynth_commit": manifest.get("diffsynth_commit"),
        "observed_diffsynth_commit": observed_diffsynth_commit,
        "runtime_pins": manifest.get("runtime_pins"),
        "runtime_versions": dict(runtime_versions),
        "model_components": manifest.get("model_components"),
        "strict_load_validations": list(strict_load_validations),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "peak_allocated_bytes": peak_memory_bytes,
        "failure": (
            {
                "stage": stage,
                "type": type(failure).__name__,
                "message": remote.sanitize_failure(failure),
            }
            if failure is not None
            else None
        ),
    }
    return value


def run_remote_real_photo_inference(payload_bytes: bytes, run_id: str) -> bytes:
    """Generate exactly seven images, never training, and return bounded partial evidence."""
    started = time.monotonic()
    stage = "input_validation"
    outputs: dict[str, bytes] = {}
    replay_equal: bool | None = None
    work_root: Path | None = None
    peak_memory: int | None = None
    failure: BaseException | None = None
    status = "failed"
    manifest: dict[str, Any] = {}
    runtime_versions: dict[str, str] = {}
    observed_commit: str | None = None
    strict_loads: list[dict[str, Any]] = []
    torch_runtime: Any | None = None
    try:
        manifest, files = runner.validate_payload_bytes(payload_bytes, run_id)
        stage = "runtime_import"
        import torch  # type: ignore[import-not-found]
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

        torch_runtime = torch
        runtime_versions = _runtime_versions()
        _validate_runtime_versions(runtime_versions)
        stage = "diffsynth_identity"
        observed_commit = _observed_diffsynth_commit()
        if observed_commit != remote.DIFFSYNTH_COMMIT:
            raise RuntimeError("DiffSynth checkout differs from the frozen commit")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("real-photo inference requires exactly one available CUDA GPU")
        torch.manual_seed(7321)
        if hasattr(torch.cuda, "manual_seed_all"):
            torch.cuda.manual_seed_all(7321)
        torch.cuda.reset_peak_memory_stats()
        stage = "adapter_validation"
        adapter = files[runner.ADAPTER_MEMBER_NAME]
        from .reload_diagnostic import _load_replay_model, validate_adapter_header

        validate_adapter_header(adapter)
        work_root = Path(tempfile.mkdtemp(prefix=f"real-photo-{run_id}-"))
        adapter_path = work_root / "step-144.safetensors"
        adapter_path.write_bytes(adapter)
        images: dict[str, Image.Image] = {}
        for case in manifest["cases"]:
            with Image.open(BytesIO(files[case["input_path"]])) as opened:
                images[case["case_id"]] = opened.convert("RGB")

        stage = "model_download"
        snapshots = remote._download_model_snapshots(snapshot_download)
        task_schedule = runner.generation_schedule()

        stage = "base_load"
        base_pipe = remote._load_pipe(snapshots, "cuda")
        for task in task_schedule[:3]:
            stage = f"base_generation_{task['case_id']}"
            with torch.inference_mode():
                generated = remote._generate(
                    base_pipe, runner.PROMPT, images[task["case_id"]].copy(), seed=7321
                ).convert("RGB")
            outputs[f"outputs/{task['case_id']}/base.png"] = _png_bytes(generated)
            del generated
        del base_pipe
        gc.collect()
        torch.cuda.empty_cache()

        stage = "strict_adapter_load"
        strict_model, strict_pipe, strict_validation = _load_replay_model(
            snapshots, adapter_path, torch
        )
        _validate_strict_load(strict_validation)
        strict_loads.append(strict_validation)
        for task in task_schedule[3:6]:
            stage = f"strict_adapter_generation_{task['case_id']}"
            with torch.inference_mode():
                generated = remote._generate(
                    strict_pipe, runner.PROMPT, images[task["case_id"]].copy(), seed=7321
                ).convert("RGB")
            outputs[f"outputs/{task['case_id']}/strict_adapter.png"] = _png_bytes(generated)
            del generated
        del strict_pipe, strict_model
        gc.collect()
        torch.cuda.empty_cache()

        stage = "strict_repeat_load"
        repeat_model, repeat_pipe, repeat_validation = _load_replay_model(
            snapshots, adapter_path, torch
        )
        _validate_strict_load(repeat_validation)
        strict_loads.append(repeat_validation)
        if strict_validation["observed"] != repeat_validation["observed"]:
            raise ValueError("observed strict adapter module modes changed after fresh reload")
        task = task_schedule[-1]
        stage = "strict_repeat_generation"
        with torch.inference_mode():
            repeated = remote._generate(
                repeat_pipe, runner.PROMPT, images[task["case_id"]].copy(), seed=7321
            ).convert("RGB")
        path = f"outputs/{task['case_id']}/strict_repeat.png"
        outputs[path] = _png_bytes(repeated)
        strict_path = f"outputs/{task['case_id']}/strict_adapter.png"
        with Image.open(BytesIO(outputs[strict_path])) as first_image:
            first_pixels = first_image.convert("RGB").tobytes()
        replay_equal = first_pixels == repeated.tobytes()
        del repeated, repeat_pipe, repeat_model, snapshots, images
        gc.collect()
        torch.cuda.empty_cache()
        if not replay_equal:
            raise ValueError("fresh strict adapter reload repeat was not pixel-exact")
        status = "passed"
    except BaseException as exc:
        failure = exc
        status = "failed"
    finally:
        if torch_runtime is not None:
            try:
                peak_memory = int(torch_runtime.cuda.max_memory_allocated())
            except Exception as exc:
                if failure is None:
                    failure = exc
                    status = "failed"
                    stage = "peak_memory_capture"
        if work_root is not None:
            shutil.rmtree(work_root, ignore_errors=True)

    try:
        receipt = _receipt(
            run_id,
            status,
            outputs,
            started=started,
            stage=stage,
            replay_equal=replay_equal,
            manifest=manifest,
            payload_sha256=hashlib.sha256(payload_bytes).hexdigest(),
            runtime_versions=runtime_versions,
            observed_diffsynth_commit=observed_commit,
            strict_load_validations=strict_loads,
            failure=failure,
            peak_memory_bytes=peak_memory,
        )
        return _pack_result({**outputs, "receipt.json": runner.canonical_json_bytes(receipt)})
    except Exception as pack_error:
        receipt = _receipt(
            run_id,
            "failed",
            {},
            started=started,
            stage="result_packaging",
            replay_equal=replay_equal,
            manifest=manifest,
            payload_sha256=hashlib.sha256(payload_bytes).hexdigest(),
            runtime_versions=runtime_versions,
            observed_diffsynth_commit=observed_commit,
            strict_load_validations=strict_loads,
            failure=pack_error,
            peak_memory_bytes=peak_memory,
        )
        return _pack_result({"receipt.json": runner.canonical_json_bytes(receipt)})
