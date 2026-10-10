"""CPU validation and inference-only replay for the failed R3 adapter reload."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import struct
import time
import zipfile
from dataclasses import dataclass
from io import BytesIO
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any

from . import remote

R3_RUN_ID = "klein-edit-pilot-2026-10-08-r3"
R3_INPUT_SHA256 = "79524d082f377cb236bfa4c0ce782f494c60108907b51220f3396789db8d0409"
R3_RESULT_SHA256 = "e509f43c88f961642fbfd76b5149c35ea6d981417ec0382821729e77effe8e35"
R3_ADAPTER_SHA256 = "873cf13aab10f7a72b65391a547af56a37e1afba1fcbffeb83be6dc2539573a3"
R3_REFERENCE_SHA256 = "a8d873933795f9e4dd8a42482c0707fe6b562c36b47c30562d1d939e4a738958"
R3_TARGET_SHA256 = "71fe2f2269fb01255b3ca18bb26ab59e7a9a500a6cc575076c4d72088ccb353a"
R3_TRAINED_SHA256 = "71610a9c23cdd687e06d5d04280a886a897bb3c3ab062cdc6be7b90accc7db39"
R3_PROMPT = (
    "Move the unchanged package into a soft green garden studio with soft diffuse light; "
    "keep the package unchanged."
)
RUN_MARKER_NAME = ".reload-diagnostic.json"
PAYLOAD_NAME = "reload-diagnostic-input.zip"
RESULT_NAME = "reload-diagnostic-result.zip"
ATTEMPT_MARKER_NAME = "reload-diagnostic-attempt.json"
MAX_INPUT_BYTES = 120 * 1024 * 1024
MAX_RESULT_BYTES = 8 * 1024 * 1024
MAX_HEADER_BYTES = 8 * 1024 * 1024
MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAE_LIMIT = 2.0
PIXELS_OVER_8_LIMIT = 0.01
EXPECTED_MODULES = 100
EXPECTED_RANK = 32
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


@dataclass(frozen=True)
class TensorSpec:
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int


@dataclass(frozen=True)
class ReplayPlan:
    output_dir: Path
    payload_path: Path
    run_id: str
    payload_bytes: int
    payload_sha256: str


@dataclass(frozen=True)
class ReplayPayload:
    run_id: str
    manifest: dict[str, Any]
    files: dict[str, bytes]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _value_schema(value: Any) -> tuple[tuple[int, ...], Any]:
    if isinstance(value, tuple) and len(value) >= 2:
        return tuple(value[0]), value[1]
    return tuple(value.shape), value.dtype


def _validate_adapter_schema(
    expected: dict[str, tuple[tuple[int, ...], Any]],
    values: dict[str, Any],
    expected_module_count: int = EXPECTED_MODULES,
) -> None:
    if not expected or set(values) != set(expected):
        raise ValueError("converted adapter keys do not exactly match injected LoRA parameters")
    if len({name.rsplit(".lora_", 1)[0] for name in expected}) != expected_module_count:
        raise ValueError(
            f"adapter schema must contain exactly {expected_module_count} LoRA modules"
        )
    for name, value in values.items():
        shape, dtype = _value_schema(value)
        if (shape, dtype) != expected[name]:
            raise ValueError(f"adapter shape or dtype mismatch for {name}")


def validate_restored_adapter(
    expected: dict[str, tuple[tuple[int, ...], Any]],
    source: dict[str, Any],
    restored: dict[str, Any],
    equal: Any,
    *,
    expected_module_count: int = EXPECTED_MODULES,
) -> None:
    """Require exact LoRA keys, tensor schemas, and equality after state restoration."""
    _validate_adapter_schema(expected, source, expected_module_count)
    _validate_adapter_schema(expected, restored, expected_module_count)
    if any(not equal(source[name], restored[name]) for name in expected):
        raise ValueError("restored adapter value differs from its validated source value")


def _safe_archive(archive: zipfile.ZipFile, *, total_limit: int) -> dict[str, bytes]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise ValueError("archive contains duplicate paths")
    if sum(info.file_size for info in infos) > total_limit:
        raise ValueError("archive exceeds its uncompressed byte limit")
    files: dict[str, bytes] = {}
    for info in infos:
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or "\\" in info.filename or info.is_dir():
            raise ValueError("archive contains an unsafe path")
        files[info.filename] = archive.read(info)
    return files


def validate_adapter_header(data: bytes) -> dict[str, TensorSpec]:
    """Validate the R3 BF16 LoRA header, tensor extents, and finite stored values."""
    if not isinstance(data, bytes) or len(data) < 8:
        raise ValueError("adapter is truncated")
    header_size = struct.unpack("<Q", data[:8])[0]
    if header_size <= 0 or header_size > MAX_HEADER_BYTES or 8 + header_size > len(data):
        raise ValueError("adapter header size is invalid")
    try:
        header = json.loads(data[8 : 8 + header_size])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("adapter header is invalid JSON") from exc
    if not isinstance(header, dict):
        raise ValueError("adapter header must be an object")
    tensor_data = data[8 + header_size :]
    specs: dict[str, TensorSpec] = {}
    intervals: list[tuple[int, int]] = []
    pairs: dict[str, set[str]] = {}
    for name, record in header.items():
        if name == "__metadata__":
            if not isinstance(record, dict) or record.get("format") != "pt":
                raise ValueError("adapter metadata is invalid")
            continue
        if not isinstance(name, str) or not isinstance(record, dict):
            raise ValueError("adapter tensor record is invalid")
        match = re.fullmatch(r"(.+)\.lora_([AB])\.default\.weight", name)
        if match is None or match.group(1).startswith("pipe.dit."):
            raise ValueError("adapter contains an unexpected LoRA key")
        dtype = record.get("dtype")
        shape = record.get("shape")
        offsets = record.get("data_offsets")
        if (
            dtype != "BF16"
            or not isinstance(shape, list)
            or not shape
            or any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(offset, bool) or not isinstance(offset, int) for offset in offsets)
        ):
            raise ValueError("adapter tensor dtype, shape, or offsets are invalid")
        start, end = offsets
        element_count = math.prod(shape)
        if start < 0 or end - start != element_count * 2 or end > len(tensor_data):
            raise ValueError("adapter tensor extent does not match its shape")
        if (match.group(2) == "A" and shape[0] != EXPECTED_RANK) or (
            match.group(2) == "B" and shape[-1] != EXPECTED_RANK
        ):
            raise ValueError("adapter tensor rank is not 32")
        for offset in range(start, end, 2):
            high = tensor_data[offset + 1] & 0x7F
            if high == 0x7F and tensor_data[offset] & 0x80:
                raise ValueError("adapter contains a non-finite BF16 value")
        module, side = match.group(1), match.group(2)
        pairs.setdefault(module, set()).add(side)
        specs[name] = TensorSpec(dtype, tuple(shape), start, end)
        intervals.append((start, end))
    intervals.sort()
    if not specs or intervals[0][0] != 0 or intervals[-1][1] != len(tensor_data):
        raise ValueError("adapter tensor data has gaps or trailing bytes")
    if any(left[1] != right[0] for left, right in pairwise(intervals)):
        raise ValueError("adapter tensor data overlaps or has gaps")
    if len(pairs) != EXPECTED_MODULES or any(sides != {"A", "B"} for sides in pairs.values()):
        raise ValueError("adapter LoRA keys must be complete A/B pairs for 100 modules")
    return specs


def _source_files(input_path: Path, result_path: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    input_bytes = input_path.read_bytes()
    result_bytes = result_path.read_bytes()
    if _sha256(input_bytes) != R3_INPUT_SHA256 or _sha256(result_bytes) != R3_RESULT_SHA256:
        raise ValueError("R3 source archive checksum does not match the reviewed run")
    with zipfile.ZipFile(BytesIO(input_bytes)) as source_archive:
        source_files = _safe_archive(source_archive, total_limit=MAX_INPUT_BYTES)
    with zipfile.ZipFile(BytesIO(result_bytes)) as result_archive:
        result_files = _safe_archive(result_archive, total_limit=MAX_INPUT_BYTES)
    try:
        source_manifest = json.loads(source_files["pilot_manifest.json"])
        receipt = json.loads(result_files["receipt.json"])
        dev = source_manifest["development_pair"]
        reference = source_files[dev["edit_image"]]
        trained = result_files["images/trained.png"]
        adapter = result_files["adapter/step-8.safetensors"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("R3 source archives are missing required diagnostic artifacts") from exc
    if (
        source_manifest.get("run_id") != R3_RUN_ID
        or dev.get("pair_id") != "pack-006-garden_studio"
        or dev.get("split") != "dev"
        or dev.get("source_sha256") != R3_REFERENCE_SHA256
        or dev.get("target_sha256") != R3_TARGET_SHA256
        or _sha256(reference) != R3_REFERENCE_SHA256
        or receipt.get("run_id") != R3_RUN_ID
        or receipt.get("status") != "failed"
        or receipt.get("failure", {}).get("stage") != "adapter_reload"
        or receipt.get("progress", {}).get("optimizer_updates") != 8
        or _sha256(trained) != R3_TRAINED_SHA256
        or _sha256(adapter) != R3_ADAPTER_SHA256
    ):
        raise ValueError("R3 artifacts do not match the failed adapter-reload run")
    validate_adapter_header(adapter)
    manifest = {
        "schema_version": 1,
        "run_id": R3_RUN_ID,
        "source_input_sha256": R3_INPUT_SHA256,
        "source_result_sha256": R3_RESULT_SHA256,
        "adapter_sha256": R3_ADAPTER_SHA256,
        "reference_sha256": R3_REFERENCE_SHA256,
        "original_trained_sha256": R3_TRAINED_SHA256,
        "prompt": dev["prompt"],
        "pair_id": dev["pair_id"],
        "seed": 7321,
        "inference_steps": 16,
        "cfg_scale": 4,
        "width": 512,
        "height": 512,
        "gpu": "A100-80GB",
        "max_runtime_seconds": 1200,
        "max_calls": 1,
    }
    return {
        "adapter/step-8.safetensors": adapter,
        "images/reference.png": reference,
        "images/original-trained.png": trained,
    }, manifest


def plan_reload_diagnostic(
    input_archive: str | Path,
    result_archive: str | Path,
    output_dir: str | Path,
) -> ReplayPlan:
    """Build a fresh, write-once payload containing only the R3 replay inputs."""
    files, manifest = _source_files(Path(input_archive), Path(result_archive))
    files["replay_manifest.json"] = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, content)
    payload = buffer.getvalue()
    if len(payload) > MAX_INPUT_BYTES:
        raise ValueError("diagnostic payload exceeds its configured byte limit")
    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    try:
        payload_path = output / PAYLOAD_NAME
        payload_path.write_bytes(payload)
        digest = _sha256(payload)
        marker = {
            "schema_version": 1,
            "run_id": R3_RUN_ID,
            "payload_file": PAYLOAD_NAME,
            "payload_bytes": len(payload),
            "payload_sha256": digest,
            "source_input_sha256": R3_INPUT_SHA256,
            "source_result_sha256": R3_RESULT_SHA256,
        }
        (output / RUN_MARKER_NAME).write_text(
            json.dumps(marker, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return ReplayPlan(output, payload_path, R3_RUN_ID, len(payload), digest)


def read_reload_payload(plan: ReplayPlan) -> ReplayPayload:
    """Recheck write-once payload, paths, pinned artifacts, and adapter before upload."""
    try:
        marker = json.loads((plan.output_dir / RUN_MARKER_NAME).read_text(encoding="utf-8"))
        payload_bytes = plan.payload_path.read_bytes()
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("diagnostic payload or marker is missing or invalid") from exc
    digest = _sha256(payload_bytes)
    if (
        not isinstance(marker, dict)
        or marker.get("run_id") != R3_RUN_ID
        or marker.get("payload_file") != PAYLOAD_NAME
        or marker.get("payload_bytes") != len(payload_bytes)
        or marker.get("payload_sha256") != digest
        or marker.get("source_input_sha256") != R3_INPUT_SHA256
        or marker.get("source_result_sha256") != R3_RESULT_SHA256
        or plan.run_id != R3_RUN_ID
        or plan.payload_sha256 != digest
        or len(payload_bytes) > MAX_INPUT_BYTES
    ):
        raise ValueError("diagnostic payload checksum or marker mismatch")
    return read_reload_payload_from_bytes(payload_bytes, plan)


def validate_result_bundle(data: bytes, run_id: str = R3_RUN_ID) -> tuple[dict[str, Any], set[str]]:
    """Accept bounded complete measurements or a failed receipt with partial outputs."""
    if not isinstance(data, bytes) or not data or len(data) > MAX_RESULT_BYTES:
        raise ValueError("diagnostic returned an empty or oversized result")
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            files = _safe_archive(archive, total_limit=MAX_RESULT_BYTES)
        receipt = json.loads(files["receipt.json"])
    except (OSError, KeyError, UnicodeDecodeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise ValueError("diagnostic result is not a valid receipt archive") from exc
    names = set(files)
    allowed = {
        "receipt.json",
        "images/original-trained.png",
        "images/unmerged-1.png",
        "images/unmerged-2.png",
        "images/fused.png",
    }
    if (
        not isinstance(receipt, dict)
        or receipt.get("run_id") != run_id
        or receipt.get("status") not in {"measured", "failed"}
        or not names <= allowed
        or any(
            len(content) > MAX_IMAGE_BYTES
            for name, content in files.items()
            if name != "receipt.json"
        )
    ):
        raise ValueError("diagnostic result receipt or artifact set is invalid")
    required = {
        "receipt.json",
        *(
            f"images/{name}.png"
            for name in ("original-trained", "unmerged-1", "unmerged-2", "fused")
        ),
    }
    if receipt["status"] == "measured" and names != required:
        raise ValueError("complete diagnostic result is missing a comparison image")
    return receipt, names


def _write_result(files: dict[str, bytes], receipt: dict[str, Any]) -> bytes:
    files = {**files, "receipt.json": (json.dumps(receipt, sort_keys=True) + "\n").encode()}
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, content)
    result = buffer.getvalue()
    if len(result) > MAX_RESULT_BYTES:
        raise ValueError("diagnostic result exceeds its 8 MiB byte limit")
    return result


def _image_metrics(reference: Any, generated: Any) -> dict[str, float | int | bool]:
    from PIL import ImageChops

    if reference.size != generated.size or reference.mode != "RGB" or generated.mode != "RGB":
        raise ValueError("diagnostic comparison images must be matching RGB images")
    difference = ImageChops.difference(reference, generated)
    pixels = list(difference.getdata())
    mae = sum(sum(pixel) for pixel in pixels) / (len(pixels) * 3)
    over = sum(max(pixel) > 8 for pixel in pixels) / len(pixels)
    return {
        "mae": mae,
        "fraction_pixels_any_channel_over_8": over,
        "within_existing_thresholds": mae <= MAE_LIMIT and over <= PIXELS_OVER_8_LIMIT,
    }


def _read_image(data: bytes) -> Any:
    from PIL import Image

    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("diagnostic PNG is empty or oversized")
    with Image.open(BytesIO(data)) as opened:
        opened.verify()
    with Image.open(BytesIO(data)) as opened:
        image = opened.convert("RGB")
    if image.size != (512, 512):
        raise ValueError("R3 replay images must be 512x512")
    return image


def _image_bytes(image: Any) -> bytes:
    stream = BytesIO()
    image.save(stream, format="PNG", optimize=True)
    return stream.getvalue()


def _load_replay_model(
    snapshot_dirs: dict[str, Path], adapter_path: Path, torch: Any
) -> tuple[Any, Any, dict[str, Any]]:
    import sys

    sys.path.insert(0, "/opt/DiffSynth-Studio")
    from diffsynth.core import load_state_dict  # type: ignore[import-not-found]
    from examples.flux2.model_training.train import (  # type: ignore[import-not-found]
        Flux2ImageTrainingModule,
    )

    model_paths = [
        sorted(
            str(path)
            for path in (snapshot_dirs["text_encoder"] / "text_encoder").glob("*.safetensors")
        ),
        sorted(
            str(path)
            for path in (snapshot_dirs["transformer"] / "transformer").glob("*.safetensors")
        ),
        [str(snapshot_dirs["vae"] / "vae" / "diffusion_pytorch_model.safetensors")],
    ]
    model = Flux2ImageTrainingModule(
        model_paths=json.dumps(model_paths),
        tokenizer_path=str(snapshot_dirs["tokenizer"] / "tokenizer"),
        lora_base_model="dit",
        lora_target_modules=remote.LORA_TARGET_MODULES,
        lora_rank=EXPECTED_RANK,
        use_gradient_checkpointing=False,
        extra_inputs="edit_image",
        remove_prefix_in_ckpt="pipe.dit.",
        device="cuda",
    )
    pipe = model.pipe
    raw = load_state_dict(str(adapter_path))
    converted = pipe.lora_loader(
        torch_dtype=pipe.torch_dtype, device=pipe.device
    ).convert_state_dict(raw)
    mapped = model.mapping_lora_state_dict(converted)
    parameters = dict(pipe.dit.named_parameters())
    expected = {
        name: (tuple(parameter.shape), parameter.dtype)
        for name, parameter in parameters.items()
        if ".lora_A.default.weight" in name or ".lora_B.default.weight" in name
    }
    _validate_adapter_schema(expected, mapped)
    for name in expected:
        value = mapped[name]
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"converted adapter contains non-finite values for {name}")
    mapped = {name: value.to(device=parameters[name].device) for name, value in mapped.items()}
    load_result = pipe.dit.load_state_dict(mapped, strict=False)
    if load_result.unexpected_keys or any("lora_" in name for name in load_result.missing_keys):
        raise ValueError("adapter load reported missing or unexpected LoRA keys")
    validate_restored_adapter(
        expected,
        mapped,
        {name: parameters[name].detach() for name in expected},
        torch.equal,
    )
    for parameter in pipe.parameters():
        parameter.requires_grad_(False)
    observed_flags = {
        "pipe_training": bool(pipe.training),
        "lora_dropout_training": [
            bool(module.training) for name, module in pipe.named_modules() if "lora_dropout" in name
        ],
    }
    return (
        model,
        pipe,
        {
            "observed": observed_flags,
            "mode_changed": False,
            "adapter_validation": {
                "validated": True,
                "restored_tensor_count": len(expected),
                "module_count": EXPECTED_MODULES,
            },
        },
    )


def run_remote_reload_diagnostic(payload_bytes: bytes, run_id: str) -> bytes:
    """Run two fresh unmerged loads and one fused control; perform no training."""
    started = time.monotonic()
    stage = "input_validation"
    outputs: dict[str, bytes] = {}
    comparisons: dict[str, Any] = {}
    modes: dict[str, Any] = {}
    unmerged_images: list[Any] = []
    work_root: Path | None = None
    try:
        if run_id != R3_RUN_ID:
            raise ValueError("diagnostic run ID does not match the pinned R3 experiment")
        plan = ReplayPlan(
            Path("."), Path("payload"), run_id, len(payload_bytes), _sha256(payload_bytes)
        )
        payload = read_reload_payload_from_bytes(payload_bytes, plan)
        for name in ("images/reference.png", "images/original-trained.png"):
            _read_image(payload.files[name])
        adapter = payload.files["adapter/step-8.safetensors"]
        work_root = Path(__import__("tempfile").mkdtemp(prefix=f"reload-{run_id}-"))
        adapter_path = work_root / "step-8.safetensors"
        adapter_path.write_bytes(adapter)
        reference = _read_image(payload.files["images/reference.png"])
        original = _read_image(payload.files["images/original-trained.png"])

        stage = "runtime_import"
        import gc

        import torch  # type: ignore[import-not-found]
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("diagnostic requires exactly one available CUDA GPU")
        torch.manual_seed(20261008)
        if hasattr(torch.cuda, "manual_seed_all"):
            torch.cuda.manual_seed_all(20261008)
        stage = "model_download"
        snapshot_dirs = remote._download_model_snapshots(snapshot_download)
        outputs["images/original-trained.png"] = payload.files["images/original-trained.png"]
        for index in (1, 2):
            stage = f"unmerged_load_{index}"
            model, pipe, modes[f"unmerged_{index}"] = _load_replay_model(
                snapshot_dirs, adapter_path, torch
            )
            stage = f"unmerged_generation_{index}"
            with torch.inference_mode():
                image = remote._generate(
                    pipe, payload.manifest["prompt"], reference.copy(), seed=7321
                ).convert("RGB")
            outputs[f"images/unmerged-{index}.png"] = _image_bytes(image)
            comparisons[f"unmerged_{index}"] = _image_metrics(original, image)
            unmerged_images.append(image.copy())
            del image, pipe, model
            gc.collect()
            torch.cuda.empty_cache()

        stage = "fused_load"
        pipe = remote._load_pipe(snapshot_dirs, "cuda")
        pipe.load_lora(pipe.dit, str(adapter_path))
        for parameter in pipe.parameters():
            parameter.requires_grad_(False)
        modes["fused"] = {
            "observed": {"pipe_training": bool(pipe.training), "lora_dropout_training": []},
            "mode_changed": False,
        }
        stage = "fused_generation"
        with torch.inference_mode():
            fused = remote._generate(
                pipe, payload.manifest["prompt"], reference.copy(), seed=7321
            ).convert("RGB")
        outputs["images/fused.png"] = _image_bytes(fused)
        comparisons["fused"] = _image_metrics(original, fused)
        comparisons["unmerged_1_vs_unmerged_2"] = _image_metrics(
            unmerged_images[0], unmerged_images[1]
        )
        del fused, pipe
        gc.collect()
        torch.cuda.empty_cache()
        receipt = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "measured",
            "diagnostic": "inference-only saved-adapter replay",
            "optimizer_updates": 0,
            "attempted_optimizer_updates": 0,
            "training_function_called": False,
            "generation": {"seed": 7321, "steps": 16, "cfg_scale": 4, "width": 512, "height": 512},
            "resource_bound": {"gpu": "A100-80GB", "max_runtime_seconds": 1200, "max_calls": 1},
            "project_allowance_usd": 15,
            "comparisons": comparisons,
            "module_modes": modes,
            "adapter_validation": modes["unmerged_1"]["adapter_validation"],
            "saved_artifacts": sorted(outputs),
            "elapsed_seconds": time.monotonic() - started,
            "cause_established": False,
            "quality_claim": False,
        }
        result = _write_result(outputs, receipt)
        if work_root is not None:
            shutil.rmtree(work_root, ignore_errors=True)
        return result
    except Exception as exc:
        receipt = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "failed",
            "failure": {
                "stage": stage,
                "type": type(exc).__name__,
                "message": remote.sanitize_failure(exc),
            },
            "optimizer_updates": 0,
            "attempted_optimizer_updates": 0,
            "training_function_called": False,
            "comparisons": comparisons,
            "module_modes": modes,
            "saved_artifacts": sorted(outputs),
            "elapsed_seconds": time.monotonic() - started,
            "cause_established": False,
            "quality_claim": False,
        }
        if work_root is not None:
            shutil.rmtree(work_root, ignore_errors=True)
        try:
            return _write_result(outputs, receipt)
        except ValueError:
            receipt["failure"] = {
                "stage": "result_packaging",
                "type": "ValueError",
                "message": "bounded partial receipt exceeded the diagnostic result limit",
            }
            receipt["saved_artifacts"] = []
            return _write_result({}, receipt)


def read_reload_payload_from_bytes(data: bytes, plan: ReplayPlan) -> ReplayPayload:
    """Validate remote function bytes without importing a model runtime."""
    if not data or len(data) > MAX_INPUT_BYTES or _sha256(data) != plan.payload_sha256:
        raise ValueError("diagnostic payload checksum or byte limit is invalid")
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            files = _safe_archive(archive, total_limit=MAX_INPUT_BYTES)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("diagnostic payload is not a valid archive") from exc
    manifest = json.loads(files.get("replay_manifest.json", b"null"))
    required = {
        "replay_manifest.json",
        "adapter/step-8.safetensors",
        "images/reference.png",
        "images/original-trained.png",
    }
    if set(files) != required or not isinstance(manifest, dict):
        raise ValueError("diagnostic payload entries do not match the approved schema")
    if (
        manifest.get("run_id") != R3_RUN_ID
        or manifest.get("source_input_sha256") != R3_INPUT_SHA256
        or manifest.get("source_result_sha256") != R3_RESULT_SHA256
        or _sha256(files["adapter/step-8.safetensors"]) != R3_ADAPTER_SHA256
        or _sha256(files["images/reference.png"]) != R3_REFERENCE_SHA256
        or _sha256(files["images/original-trained.png"]) != R3_TRAINED_SHA256
        or manifest.get("pair_id") != "pack-006-garden_studio"
        or manifest.get("prompt") != R3_PROMPT
        or manifest.get("seed") != 7321
        or manifest.get("inference_steps") != 16
        or manifest.get("cfg_scale") != 4
        or manifest.get("width") != 512
        or manifest.get("height") != 512
        or manifest.get("gpu") != "A100-80GB"
        or manifest.get("max_runtime_seconds") != 1200
        or manifest.get("max_calls") != 1
    ):
        raise ValueError("diagnostic payload does not contain the pinned R3 artifacts")
    validate_adapter_header(files["adapter/step-8.safetensors"])
    return ReplayPayload(R3_RUN_ID, manifest, files)
