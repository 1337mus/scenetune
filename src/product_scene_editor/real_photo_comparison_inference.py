"""Inference-only paired real-photo comparison worker."""

from __future__ import annotations

import gc
import hashlib
import shutil
import tempfile
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, cast

from PIL import Image

from . import real_photo_comparison as comparison
from . import real_photo_inference as legacy_inference
from . import real_photo_runner as legacy
from . import remote


def _png_bytes(image: Any) -> bytes:
    stream = BytesIO()
    image.convert("RGB").save(stream, format="PNG", compress_level=6, optimize=False)
    return stream.getvalue()


def _pack_result(files: dict[str, bytes]) -> bytes:
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, data)
    payload = stream.getvalue()
    if len(payload) > comparison.MAX_RESULT_BYTES:
        raise ValueError("preservation-pair result exceeds the 24 MiB result limit")
    return payload


def _output_records(outputs: dict[str, bytes]) -> list[dict[str, Any]]:
    records = []
    for path, data in sorted(outputs.items()):
        with Image.open(BytesIO(data)) as image:
            if image.mode != "RGB" or image.size != (512, 512):
                raise ValueError(f"paired output {path} is not a 512x512 RGB PNG")
        records.append(
            {"path": path, "sha256": hashlib.sha256(data).hexdigest(), "width": 512, "height": 512}
        )
    return records


def _receipt(
    run_id: str,
    payload_sha256: str,
    manifest: dict[str, Any],
    outputs: dict[str, bytes],
    replay_pixel_equal: dict[str, dict[str, bool | None]],
    strict_load_validations: list[dict[str, Any]],
    *,
    status: str,
    started: float,
    stage: str,
    runtime_versions: dict[str, str],
    observed_commit: str | None,
    peak_memory: int | None,
    failure: BaseException | None,
) -> dict[str, Any]:
    try:
        bindings = comparison.clearance_bindings(manifest)
    except ValueError:
        bindings = {"profile": comparison.PROFILE, "run_id": run_id}
    return {
        "schema_version": 1,
        **bindings,
        **{
            key: manifest[key]
            for key in (
                "geometry_review_sha256",
                "geometry_review_status",
                "quality_review_status",
                "active_intake_sha256",
                "private_source_review_sha256",
            )
            if key in manifest
        },
        "status": status,
        "payload_sha256": payload_sha256,
        "generation_settings": legacy.GENERATION_SETTINGS,
        "schedule": list(comparison.generation_schedule()),
        "training_function_called": False,
        "optimizer_updates": 0,
        "quality_claim": False,
        "worker_failure_confirmed": status == "failed",
        "output_artifacts": _output_records(outputs),
        "replay_pixel_equal": replay_pixel_equal,
        "strict_load_validations": list(strict_load_validations),
        "runtime_versions": dict(runtime_versions),
        "observed_diffsynth_commit": observed_commit,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "peak_allocated_bytes": peak_memory,
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


def _same_pixels(first: bytes, second: bytes) -> bool:
    with Image.open(BytesIO(first)) as left, Image.open(BytesIO(second)) as right:
        return left.convert("RGB").tobytes() == right.convert("RGB").tobytes()


def run_remote_real_photo_comparison(payload_bytes: bytes, run_id: str) -> bytes:
    """Run three base and twelve sequential strict-adapter generations; never train."""
    started = time.monotonic()
    stage = "input_validation"
    payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    outputs: dict[str, bytes] = {}
    replays: dict[str, dict[str, Any]] = {
        case_id: {"prior": None, "new": None} for case_id in comparison.CASE_IDS
    }
    strict_loads: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {}
    versions: dict[str, str] = {}
    observed_commit: str | None = None
    peak_memory: int | None = None
    failure: BaseException | None = None
    work_root: Path | None = None
    torch: Any | None = None
    status = "failed"
    try:
        manifest, files = comparison.validate_payload_bytes(payload_bytes, run_id)
        stage = "runtime_import"
        import torch as torch_runtime  # type: ignore[import-not-found]
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

        torch = torch_runtime
        versions = legacy_inference._runtime_versions()
        legacy_inference._validate_runtime_versions(versions)
        observed_commit = legacy_inference._observed_diffsynth_commit()
        if observed_commit != remote.DIFFSYNTH_COMMIT:
            raise RuntimeError("DiffSynth checkout differs from the frozen commit")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("preservation-pair inference requires exactly one CUDA GPU")
        torch.manual_seed(7321)
        torch.cuda.manual_seed_all(7321)
        torch.cuda.reset_peak_memory_stats()
        work_root = Path(tempfile.mkdtemp(prefix=f"real-photo-pair-{run_id}-"))
        case_rows = cast(list[dict[str, Any]], manifest["cases"])
        cases = {case["case_id"]: case for case in case_rows}
        adapters = cast(dict[str, dict[str, str]], manifest["adapters"])
        prompt = cast(str, manifest["prompt"])
        images = {
            case_id: Image.open(BytesIO(files[cases[case_id]["input_path"]])).convert("RGB")
            for case_id in comparison.CASE_IDS
        }
        adapter_paths: dict[str, Path] = {}
        from .reload_diagnostic import _load_replay_model, validate_adapter_header

        for key in ("prior", "new"):
            member = adapters[key]["payload_path"]
            adapter = files[member]
            validate_adapter_header(adapter)
            adapter_paths[key] = work_root / f"{key}.safetensors"
            adapter_paths[key].write_bytes(adapter)
        stage = "model_download"
        snapshots = remote._download_model_snapshots(snapshot_download)
        stage = "base_load"
        base_pipe = remote._load_pipe(snapshots, "cuda")
        for case_id in comparison.CASE_IDS:
            stage = f"base_generation_{case_id}"
            with torch.inference_mode():
                image = remote._generate(base_pipe, prompt, images[case_id].copy(), 7321)
            outputs[f"outputs/{case_id}/base.png"] = _png_bytes(image)
            del image
        del base_pipe
        gc.collect()
        torch.cuda.empty_cache()

        observed_modes: dict[str, Any] | None = None
        for arm, session, adapter_key in (
            ("prior", "prior-strict-1", "prior"),
            ("prior_reloaded", "prior-strict-2", "prior"),
            ("new", "new-strict-1", "new"),
            ("new_reloaded", "new-strict-2", "new"),
        ):
            stage = f"{arm}_strict_load"
            model, pipe, validation = _load_replay_model(
                snapshots, adapter_paths[adapter_key], torch
            )
            legacy_inference._validate_strict_load(validation)
            observed = validation["observed"]
            if observed_modes is not None and observed != observed_modes:
                raise ValueError("observed adapter module modes changed between strict sessions")
            observed_modes = observed
            strict_loads.append({"arm": adapter_key, "session": session, "validation": validation})
            for case_id in comparison.CASE_IDS:
                stage = f"{arm}_generation_{case_id}"
                with torch.inference_mode():
                    image = remote._generate(pipe, prompt, images[case_id].copy(), 7321)
                outputs[f"outputs/{case_id}/{arm}.png"] = _png_bytes(image)
                del image
                base_arm = "prior" if arm == "prior_reloaded" else "new"
                if arm.endswith("_reloaded"):
                    replays[case_id][base_arm] = _same_pixels(
                        outputs[f"outputs/{case_id}/{base_arm}.png"],
                        outputs[f"outputs/{case_id}/{arm}.png"],
                    )
                    if not replays[case_id][base_arm]:
                        raise ValueError(
                            f"{base_arm} fresh reload was not pixel-exact for {case_id}"
                        )
            del pipe, model
            gc.collect()
            torch.cuda.empty_cache()
        if (
            len(outputs) != 15
            or len(strict_loads) != 4
            or not all(pair[arm] for pair in replays.values() for arm in ("prior", "new"))
        ):
            raise ValueError("preservation-pair schedule or six reload checks were incomplete")
        status = "passed"
    except BaseException as exc:
        failure = exc
    finally:
        if torch is not None:
            try:
                peak_memory = int(torch.cuda.max_memory_allocated())
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
            payload_sha256,
            manifest,
            outputs,
            replays,
            strict_loads,
            status=status,
            started=started,
            stage=stage,
            runtime_versions=versions,
            observed_commit=observed_commit,
            peak_memory=peak_memory,
            failure=failure,
        )
        return _pack_result({**outputs, "receipt.json": legacy.canonical_json_bytes(receipt)})
    except Exception as exc:
        for checks in replays.values():
            checks.update(prior=None, new=None)
        receipt = _receipt(
            run_id,
            payload_sha256,
            manifest,
            {},
            replays,
            strict_loads,
            status="failed",
            started=started,
            stage="result_packaging",
            runtime_versions=versions,
            observed_commit=observed_commit,
            peak_memory=peak_memory,
            failure=exc,
        )
        return _pack_result({"receipt.json": legacy.canonical_json_bytes(receipt)})
