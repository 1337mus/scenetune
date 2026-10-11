"""Inference-only worker for the frozen protected-product comparison.

Runtime and model dependencies stay inside the remote entry point so payload validation and
the injected CPU schedule helper remain usable on the laptop.
"""

from __future__ import annotations

import gc
import hashlib
import json
import re
import time
import zipfile
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from . import protected_comparison as comparison
from . import remote
from .protected_product import compose_protected_product

MAX_RESULT_BYTES = 64 * 1024 * 1024
_SAFE_RESULT_MEMBER = re.compile(
    r"^(?:outputs|replays)/[A-Za-z0-9][A-Za-z0-9._-]{0,63}/"
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\.png$"
)


def _png_bytes(image: Image.Image) -> bytes:
    output = BytesIO()
    image.convert("RGB").save(output, format="PNG", compress_level=6, optimize=False)
    return output.getvalue()


def _pack_result(files: dict[str, bytes]) -> bytes:
    for name, content in files.items():
        if name != "receipt.json" and (
            _SAFE_RESULT_MEMBER.fullmatch(name) is None or ".." in name.split("/")
        ):
            raise ValueError("protected comparison result contains an unsafe member")
        if not isinstance(content, bytes):
            raise ValueError("protected comparison result members must be bytes")
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, content)
    result = output.getvalue()
    if len(result) > MAX_RESULT_BYTES:
        raise ValueError("protected comparison result exceeds the 64 MiB result limit")
    return result


def _new_execution_state() -> dict[str, Any]:
    return {
        "status": "pending",
        "generation_call_count": 0,
        "generation_records": [],
        "generation_call_timings": [],
        "replay_checks": [],
        "composed_integrity_records": [],
        "outputs": {},
        "failure": None,
    }


def _load_case_image(
    files: dict[str, bytes], case: dict[str, Any], role: str, mode: str
) -> Image.Image:
    asset = case[role]
    if not isinstance(asset, dict):
        raise ValueError(f"case {case['case_id']} has an invalid {role} asset reference")
    with Image.open(BytesIO(files[asset["path"]])) as opened:
        return opened.convert(mode)


def _output_path(task: comparison.GenerationTask) -> str:
    case_id = task["case_id"]
    if task["purpose"] == "fresh_load_replay":
        return f"replays/{case_id}/{task['arm']}.png"
    if task["arm"] == "base_full_image_edit":
        return f"outputs/{case_id}/full_image_edit.png"
    return f"outputs/{case_id}/empty_scene_raw.png"


def _generation_kwargs(
    pipe: Any,
    manifest: dict[str, Any],
    task: comparison.GenerationTask,
    source: Image.Image,
) -> dict[str, Any]:
    generation = manifest["generation"]
    kwargs: dict[str, Any] = {
        "prompt": manifest["prompts"][
            "full_image_edit" if task["arm"] == "base_full_image_edit" else "empty_scene"
        ],
        "seed": task["seed"],
        "rand_device": pipe.device,
        "num_inference_steps": generation["steps"],
        "cfg_scale": generation["cfg"],
        "height": generation["height"],
        "width": generation["width"],
    }
    if task["arm"] == "base_full_image_edit":
        kwargs["edit_image"] = [source.copy()]
    return kwargs


def _run_task(
    pipe: Any,
    runtime: Any,
    manifest: dict[str, Any],
    case: dict[str, Any],
    files: dict[str, bytes],
    task: comparison.GenerationTask,
    state: dict[str, Any],
) -> tuple[Image.Image, bytes]:
    source = _load_case_image(files, case, "source_rgb", "RGB")
    kwargs = _generation_kwargs(pipe, manifest, task, source)
    inference_context = (
        runtime.inference_mode() if hasattr(runtime, "inference_mode") else nullcontext()
    )
    with inference_context:
        state["generation_call_count"] += 1
        generation_started = time.monotonic()
        try:
            generated = pipe(**kwargs)
        finally:
            state["generation_call_timings"].append(
                {
                    "index": task["index"],
                    "purpose": task["purpose"],
                    "case_id": task["case_id"],
                    "arm": task["arm"],
                    "generation_seconds": round(time.monotonic() - generation_started, 6),
                }
            )
    if not isinstance(generated, Image.Image):
        raise ValueError("inference pipeline did not return a PIL image")
    generated = generated.convert("RGB")
    if generated.size != (512, 512):
        raise ValueError("inference pipeline output is not 512x512")
    raw_png = _png_bytes(generated)
    relative_path = _output_path(task)
    state["outputs"][relative_path] = raw_png
    record: dict[str, Any] = {
        "index": task["index"],
        "purpose": task["purpose"],
        "case_id": task["case_id"],
        "arm": task["arm"],
        "seed": task["seed"],
        "generation_seconds": state["generation_call_timings"][-1]["generation_seconds"],
        "output_path": relative_path,
        "output_sha256": hashlib.sha256(raw_png).hexdigest(),
    }
    state["generation_records"].append(record)

    if task["purpose"] == "primary" and task["arm"] == "empty_scene_composite":
        composite_png, integrity = _compose_task(files, case, task, raw_png, state)
        return generated, composite_png
    if task["purpose"] == "fresh_load_replay" and task["arm"] == "empty_scene_composite":
        composite_png, _integrity = _compose_task(files, case, task, raw_png, state)
        return generated, composite_png
    return generated, raw_png


def _compose_task(
    files: dict[str, bytes],
    case: dict[str, Any],
    task: comparison.GenerationTask,
    raw_png: bytes,
    state: dict[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    alpha = case["alpha_l"]
    protected = case["protected_l"]
    compose_started = time.monotonic()
    composite_png, integrity = compose_protected_product(
        files[case["source_rgb"]["path"]],
        raw_png,
        files[alpha["path"]],
        files[protected["path"]],
    )
    composite_path = (
        f"outputs/{task['case_id']}/empty_scene_composite.png"
        if task["purpose"] == "primary"
        else f"replays/{task['case_id']}/empty_scene_composite_final.png"
    )
    state["outputs"][composite_path] = composite_png
    state["composed_integrity_records"].append(
        {
            "purpose": task["purpose"],
            "index": task["index"],
            "case_id": task["case_id"],
            "output_path": composite_path,
            "compositing_seconds": round(time.monotonic() - compose_started, 6),
            **integrity,
        }
    )
    return composite_png, integrity


def _execute_schedule(
    manifest: dict[str, Any],
    files: dict[str, bytes],
    schedule: tuple[comparison.GenerationTask, ...],
    runtime: Any,
    snapshots: dict[str, Path],
    load_pipe: Any,
    state: dict[str, Any],
) -> None:
    """Execute primaries, release the first pipeline, then replay the four frozen tasks."""
    cases = {case["case_id"]: case for case in manifest["cases"]}
    primary = [task for task in schedule if task["purpose"] == "primary"]
    replay = [task for task in schedule if task["purpose"] == "fresh_load_replay"]
    if len(primary) != 48 or len(replay) != 4 or len(schedule) != 52:
        raise ValueError("protected comparison schedule must contain 48 primaries and four replays")
    primary_pixels: dict[tuple[str, str], bytes] = {}
    primary_background_pixels: dict[tuple[str, str], bytes] = {}
    stage = "base_load"
    pipe: Any | None = None
    try:
        pipe = load_pipe(snapshots, "cuda")
        for task in primary:
            stage = f"primary_generation_{task['case_id']}_{task['arm']}"
            generated, final_png = _run_task(
                pipe, runtime, manifest, cases[task["case_id"]], files, task, state
            )
            primary_background_pixels[(task["case_id"], task["arm"])] = generated.tobytes()
            with Image.open(BytesIO(final_png)) as final_image:
                primary_pixels[(task["case_id"], task["arm"])] = final_image.tobytes()
        del pipe
        pipe = None
        gc.collect()
        runtime.cuda.empty_cache()

        stage = "fresh_replay_load"
        pipe = load_pipe(snapshots, "cuda")
        for task in replay:
            stage = f"replay_generation_{task['case_id']}_{task['arm']}"
            generated, final_png = _run_task(
                pipe, runtime, manifest, cases[task["case_id"]], files, task, state
            )
            expected_pixels = primary_pixels[(task["case_id"], task["arm"])]
            with Image.open(BytesIO(final_png)) as final_image:
                final_pixels = final_image.tobytes()
            final_equal = final_pixels == expected_pixels
            background_equal = (
                generated.tobytes() == primary_background_pixels[(task["case_id"], task["arm"])]
            )
            state["replay_checks"].append(
                {
                    "case_id": task["case_id"],
                    "arm": task["arm"],
                    "primary_index": task["replay_of"],
                    "replay_index": task["index"],
                    "final_pixel_equal": final_equal,
                    "raw_background_pixel_equal": background_equal,
                }
            )
            if not final_equal or not background_equal:
                raise ValueError("fresh-load replay pixels differ from the primary output")
        state["status"] = "passed"
    except Exception as exc:
        state["status"] = "failed"
        state["failure"] = {
            "stage": stage,
            "type": type(exc).__name__,
            "message": remote.sanitize_failure(exc),
        }
    finally:
        if pipe is not None:
            del pipe
        gc.collect()


def _runtime_dependencies() -> tuple[Any, Any]:
    import torch  # type: ignore[import-not-found]
    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

    return torch, snapshot_download


def _receipt(
    status: str,
    state: dict[str, Any],
    *,
    run_id: str,
    started: float,
    stage: str,
    payload_sha256: str,
    manifest: dict[str, Any],
    runtime_versions: dict[str, str],
    observed_commit: str | None,
    peak_memory_bytes: int | None,
    failure: dict[str, str] | None,
) -> dict[str, Any]:
    output_artifacts: list[dict[str, Any]] = []
    for path, content in sorted(state["outputs"].items()):
        with Image.open(BytesIO(content)) as opened:
            if opened.format != "PNG" or opened.mode != "RGB" or opened.size != (512, 512):
                raise ValueError(f"worker output {path} is not a 512x512 RGB PNG")
        output_artifacts.append(
            {
                "path": path,
                "sha256": hashlib.sha256(content).hexdigest(),
                "width": 512,
                "height": 512,
            }
        )
    arm_generation_seconds: dict[str, float] = {}
    for timing in state["generation_call_timings"]:
        key = f"{timing['purpose']}/{timing['arm']}"
        arm_generation_seconds[key] = round(
            arm_generation_seconds.get(key, 0.0) + timing["generation_seconds"], 6
        )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "status": status,
        "source_sha": manifest.get("source_sha"),
        "context": manifest.get("context"),
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        if manifest
        else None,
        "generation_settings": manifest.get("generation"),
        "schedule": list(comparison.generation_schedule(manifest)) if manifest else [],
        "generation_call_count": state["generation_call_count"],
        "generation_call_timings": state["generation_call_timings"],
        "arm_generation_seconds": arm_generation_seconds,
        "training_function_called": False,
        "optimizer_updates": 0,
        "output_artifacts": output_artifacts,
        "generation_records": state["generation_records"],
        "replay_checks": state["replay_checks"],
        "composed_integrity_records": state["composed_integrity_records"],
        "payload_sha256": payload_sha256,
        "modal_sdk": manifest.get("modal_sdk"),
        "diffsynth_commit": manifest.get("diffsynth_commit"),
        "observed_diffsynth_commit": observed_commit,
        "runtime_pins": manifest.get("runtime_pins"),
        "runtime_versions": runtime_versions,
        "model_components": manifest.get("model_components"),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "peak_allocated_bytes": peak_memory_bytes,
        "failure": failure or (state.get("failure") if status == "failed" else None),
        "failure_stage": stage if status == "failed" else None,
    }


def run_remote_protected_comparison_inference(payload_bytes: bytes, run_id: str) -> bytes:
    """Validate first, run 52 inference calls without training, and return partial evidence."""
    if (
        not isinstance(run_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", run_id) is None
    ):
        raise ValueError("run_id must be a safe identifier")
    started = time.monotonic()
    stage = "input_validation"
    manifest: dict[str, Any] = {}
    runtime_versions: dict[str, str] = {}
    observed_commit: str | None = None
    peak_memory: int | None = None
    runtime: Any | None = None
    failure: dict[str, str] | None = None
    state = _new_execution_state()
    status = "failed"
    try:
        manifest, files = comparison.validate_payload_bytes(payload_bytes)
        stage = "runtime_import"
        runtime, snapshot_download = _runtime_dependencies()
        from . import real_photo_inference

        runtime_versions = real_photo_inference._runtime_versions()
        real_photo_inference._validate_runtime_versions(runtime_versions)
        stage = "diffsynth_identity"
        observed_commit = real_photo_inference._observed_diffsynth_commit()
        if observed_commit != remote.DIFFSYNTH_COMMIT:
            raise RuntimeError("DiffSynth checkout differs from the frozen commit")
        if not runtime.cuda.is_available() or runtime.cuda.device_count() != 1:
            raise RuntimeError("protected comparison requires exactly one available CUDA GPU")
        runtime.cuda.reset_peak_memory_stats()
        stage = "model_download"
        snapshots = remote._download_model_snapshots(snapshot_download)
        stage = "inference"
        _execute_schedule(
            manifest,
            files,
            comparison.generation_schedule(manifest),
            runtime,
            snapshots,
            remote._load_pipe,
            state,
        )
        status = state["status"]
        if status == "failed":
            failure = state["failure"]
            stage = failure["stage"]
        else:
            status = "passed"
    except Exception as exc:
        failure = {
            "stage": stage,
            "type": type(exc).__name__,
            "message": remote.sanitize_failure(exc),
        }
        state["status"] = "failed"
    finally:
        if runtime is not None:
            try:
                peak_memory = int(runtime.cuda.max_memory_allocated())
            except Exception as exc:
                if failure is None:
                    stage = "peak_memory_capture"
                    failure = {
                        "stage": "peak_memory_capture",
                        "type": type(exc).__name__,
                        "message": remote.sanitize_failure(exc),
                    }
                    status = "failed"
            gc.collect()
            try:
                runtime.cuda.empty_cache()
            except Exception:
                pass

    receipt = _receipt(
        status,
        state,
        run_id=run_id,
        started=started,
        stage=stage,
        payload_sha256=hashlib.sha256(payload_bytes).hexdigest()
        if isinstance(payload_bytes, bytes)
        else hashlib.sha256(b"").hexdigest(),
        manifest=manifest,
        runtime_versions=runtime_versions,
        observed_commit=observed_commit,
        peak_memory_bytes=peak_memory,
        failure=failure,
    )
    receipt_bytes = json.dumps(receipt, sort_keys=True).encode()
    try:
        return _pack_result({**state["outputs"], "receipt.json": receipt_bytes})
    except Exception as exc:
        state = {**state, "outputs": {}, "status": "failed"}
        failure = {
            "stage": "result_packaging",
            "type": type(exc).__name__,
            "message": remote.sanitize_failure(exc),
        }
        receipt = _receipt(
            "failed",
            state,
            run_id=run_id,
            started=started,
            stage="result_packaging",
            payload_sha256=hashlib.sha256(payload_bytes).hexdigest()
            if isinstance(payload_bytes, bytes)
            else hashlib.sha256(b"").hexdigest(),
            manifest=manifest,
            runtime_versions=runtime_versions,
            observed_commit=observed_commit,
            peak_memory_bytes=peak_memory,
            failure=failure,
        )
        return _pack_result({"receipt.json": json.dumps(receipt, sort_keys=True).encode()})
