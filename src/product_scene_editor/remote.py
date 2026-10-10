"""Pinned, bounded Modal execution for the FLUX.2 Klein pilot.

This module imports no GPU or Modal packages on the CPU path. The remote worker loads
those dependencies only after Modal has started the approved one-shot A100 function.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .pilot import APPROVED_DATASET_MANIFEST_SHA256, MAX_PAYLOAD_BYTES, TRAINING_PAIR_COUNT

PROFILE = "reflex-personal"
WORKSPACE = "rajath-61258"
ENVIRONMENT = "main"
TRAINING_STEPS = 8
MAX_RESULT_BYTES = 128 * 1024 * 1024
DIFFSYNTH_COMMIT = "ee631d4907d8d6fc3d39063a2a65c00a954c9bf5"
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
RUNTIME_PINS = {
    "torch": "2.14.1+cu130",
    "torchvision": "0.29.1+cu130",
    "transformers": "5.18.0",
    "peft": "0.21.0",
    "Pillow": "12.0.0",
    "pydantic": "2.13.5",
    "huggingface-hub": "1.33.0",
    "tokenizers": "0.23.2",
    "safetensors": "0.8.0",
    "accelerate": "1.12.0",
    "imageio": "2.37.0",
    "einops": "0.8.1",
    "modelscope": "1.32.0",
    "ftfy": "6.3.1",
    "pandas": "2.3.3",
    "tqdm": "4.67.1",
}
MODEL_COMPONENTS = {
    "transformer": {
        "repo_id": "black-forest-labs/FLUX.2-klein-base-4B",
        "revision": "a3b4f4849157f664bdbc776fd7453c2783562f4d",
    },
    "text_encoder": {
        "repo_id": "black-forest-labs/FLUX.2-klein-4B",
        "revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
    },
    "vae": {
        "repo_id": "black-forest-labs/FLUX.2-klein-4B",
        "revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
    },
    "tokenizer": {
        "repo_id": "black-forest-labs/FLUX.2-klein-4B",
        "revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
    },
}
CREDENTIAL_ENVIRONMENT_KEYS = (
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    "MODAL_OAUTH_REFRESH_TOKEN",
    "MODAL_OAUTH_CLIENT_ID",
    "MODAL_OAUTH_CLIENT_SECRET",
)
LORA_TARGET_MODULES = ",".join(
    [
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "add_q_proj",
        "add_k_proj",
        "add_v_proj",
        "to_add_out",
        "linear_in",
        "linear_out",
        "to_qkv_mlp_proj",
        *(f"single_transformer_blocks.{index}.attn.to_out" for index in range(20)),
    ]
)


class _WorkerFailure(Exception):
    def __init__(self, stage: str, cause: BaseException) -> None:
        super().__init__(str(cause))
        self.stage = stage
        self.cause = cause


def validate_launch_environment(context: dict[str, Any]) -> None:
    """Reject credential overrides and require the exact approved Modal target."""
    if not isinstance(context, dict) or set(context) != {"profile", "workspace", "environment"}:
        raise ValueError("Modal launch context must specify profile, workspace, and environment")
    if context != {"profile": PROFILE, "workspace": WORKSPACE, "environment": ENVIRONMENT}:
        raise ValueError("Modal launch target must be reflex-personal / rajath-61258 / main")
    if any(os.environ.get(key, "") for key in CREDENTIAL_ENVIRONMENT_KEYS):
        raise ValueError("Modal credential environment overrides must be unset")
    configured_profile = os.environ.get("MODAL_PROFILE")
    if configured_profile and configured_profile != PROFILE:
        raise ValueError("MODAL_PROFILE must select reflex-personal")


def validate_training_contract(
    *,
    train_pair_count: int,
    batch_size: int,
    dataset_repeat: int,
    num_epochs: int,
    gradient_accumulation_steps: int,
    process_count: int,
    save_steps: int,
) -> int:
    """Match launcher settings to the pinned-source eight-update schedule."""
    values = (
        train_pair_count,
        batch_size,
        dataset_repeat,
        num_epochs,
        gradient_accumulation_steps,
        process_count,
        save_steps,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("training schedule settings must be integers")
    expected = (TRAINING_PAIR_COUNT, 1, 1, 1, 1, 1, TRAINING_STEPS)
    if values != expected:
        raise ValueError("pilot schedule must make and save exactly eight single-sample updates")
    return TRAINING_STEPS


def _parse_workspace(token_output: str) -> str | None:
    matches = [
        line.split(":", 1)[1].strip().split(" ", 1)[0]
        for line in token_output.splitlines()
        if line.startswith("Workspace:")
    ]
    return matches[0] if len(matches) == 1 else None


def verify_modal_identity(
    *,
    profile: str,
    workspace: str,
    environment: str,
    run_token_info: Any = subprocess.run,
) -> dict[str, str]:
    """Verify the local Modal profile and workspace without exposing token output."""
    validate_launch_environment(
        {"profile": profile, "workspace": workspace, "environment": environment}
    )
    try:
        sdk_version = importlib.metadata.version("modal")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("Modal SDK 1.6.1 is required") from exc
    if sdk_version != "1.6.1":
        raise RuntimeError("Modal SDK 1.6.1 is required")
    child_env = dict(os.environ)
    child_env["MODAL_PROFILE"] = PROFILE
    try:
        completed = run_token_info(
            [sys.executable, "-m", "modal", "token", "info"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
            env=child_env,
        )
    except Exception as exc:
        raise RuntimeError("Modal profile verification failed") from exc
    actual_workspace = _parse_workspace(
        completed.stdout if isinstance(completed.stdout, str) else ""
    )
    if completed.returncode != 0 or actual_workspace != WORKSPACE:
        raise RuntimeError("Modal token info did not identify the approved personal workspace")
    return {
        "profile": PROFILE,
        "workspace": WORKSPACE,
        "environment": ENVIRONMENT,
        "modal_sdk": sdk_version,
    }


def sanitize_failure(error: BaseException) -> str:
    """Keep exception context while redacting common token, URL, and credential forms."""
    message = str(error)[:2000]
    message = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [REDACTED]", message)
    message = re.sub(r"(?i)(hf_[A-Za-z0-9]{12,})", "[REDACTED_TOKEN]", message)
    message = re.sub(
        r"(?i)(token|secret|password|api[_-]?key|authorization)(\s*[=:]\s*)([^\s,;]+)",
        r"\1\2[REDACTED]",
        message,
    )
    message = re.sub(r"(https?://)[^/@\s]+:[^/@\s]+@", r"\1[REDACTED]@", message)
    message = re.sub(r"(https?://[^\s?#]+)[?#][^\s]*", r"\1[REDACTED]", message)
    return message


def _bundle(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, content)
    result = buffer.getvalue()
    if len(result) > MAX_RESULT_BYTES:
        raise ValueError("pilot result exceeds the configured byte limit")
    return result


def _failure_bundle(
    run_id: str,
    stage: str,
    error: BaseException,
    progress: dict[str, Any],
    work_root: Path | None,
) -> bytes:
    saved_artifacts: dict[str, bytes] = {}
    if work_root is not None:
        artifact_paths = (
            ("adapter/step-8.safetensors", work_root / "training-output" / "step-8.safetensors"),
            ("images/base.png", work_root / "base.png"),
            ("images/trained.png", work_root / "trained.png"),
            ("images/reloaded.png", work_root / "reloaded.png"),
        )
        for name, path in artifact_paths:
            if path.is_file() and not path.is_symlink() and path.stat().st_size:
                content = path.read_bytes()
                try:
                    _bundle({**saved_artifacts, name: content})
                except ValueError:
                    continue
                saved_artifacts[name] = content
    failure_progress: dict[str, Any] = {
        "optimizer_updates": min(int(progress.get("optimizer_updates", 0)), TRAINING_STEPS),
        "optimizer_steps_attempted": min(
            int(progress.get("optimizer_steps_attempted", 0)), TRAINING_STEPS
        ),
        "losses": list(progress.get("losses", []))[:TRAINING_STEPS],
        "gradient_steps": list(progress.get("gradient_steps", []))[:TRAINING_STEPS],
        "saved_artifacts": list(saved_artifacts),
        "worker_elapsed_seconds": max(0.0, float(progress.get("elapsed_seconds", 0.0))),
    }
    for key in (
        "training_seconds",
        "step_seconds",
        "max_allocated_gpu_bytes",
        "changed_adapter_tensors",
        "max_parameter_delta",
    ):
        if key in progress and progress[key] is not None:
            value = progress[key]
            failure_progress[key] = list(value)[:TRAINING_STEPS] if key == "step_seconds" else value

    receipt = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "failed",
        "failure": {
            "stage": stage,
            "type": type(error).__name__,
            "message": sanitize_failure(error),
        },
        "progress": failure_progress,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
    }
    artifacts = {
        **saved_artifacts,
        "receipt.json": (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode(),
    }
    return _bundle(artifacts)


def _load_payload(
    payload_bytes: bytes, run_id: str, work_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    if (
        not isinstance(payload_bytes, bytes)
        or not payload_bytes
        or len(payload_bytes) > MAX_PAYLOAD_BYTES
    ):
        raise ValueError("pilot input payload is empty or exceeds the input limit")
    try:
        with zipfile.ZipFile(io.BytesIO(payload_bytes)) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("duplicate archive paths")
            allowed = {"metadata.json", "pilot_manifest.json"}
            for name in names:
                path = PurePosixPath(name)
                if path.is_absolute() or ".." in path.parts or "sealed_test" in path.parts:
                    raise ValueError("unsafe or sealed-test path in pilot archive")
                if name in allowed:
                    continue
                if len(path.parts) != 3 or path.parts[0] not in {"train", "dev"}:
                    raise ValueError("unexpected path in pilot archive")
                if path.parts[1] not in {"source", "target"} or not path.name.endswith(".png"):
                    raise ValueError("unexpected image path in pilot archive")
            files = {name: archive.read(name) for name in names}
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("pilot input is not a valid archive") from exc
    try:
        metadata = json.loads(files["metadata.json"])
        manifest = json.loads(files["pilot_manifest.json"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError("pilot archive is missing metadata") from exc
    train_rows = manifest.get("training_pairs")
    dev_pair = manifest.get("development_pair")
    source_manifest_sha256 = manifest.get("source_dataset_manifest_sha256")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("run_id") != run_id
        or not isinstance(source_manifest_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", source_manifest_sha256)
        or not isinstance(train_rows, list)
        or len(train_rows) != TRAINING_PAIR_COUNT
        or not isinstance(dev_pair, dict)
        or dev_pair.get("split") != "dev"
        or not isinstance(metadata, list)
        or len(metadata) != TRAINING_PAIR_COUNT
    ):
        raise ValueError("pilot input must contain exactly eight train pairs and one dev pair")
    if metadata != [
        {key: row.get(key) for key in ("image", "edit_image", "prompt")} for row in train_rows
    ]:
        raise ValueError("DiffSynth metadata does not match the input manifest")
    for row in [*train_rows, dev_pair]:
        if not isinstance(row, dict) or row.get("split") not in {"train", "dev"}:
            raise ValueError("pilot archive contains a row outside train/dev")
        for key in ("image", "edit_image"):
            filename = row.get(key)
            if not isinstance(filename, str) or filename not in files:
                raise ValueError("pilot archive references a missing image")
            expected_split = "train" if row["split"] == "train" else "dev"
            if PurePosixPath(filename).parts[0] != expected_split:
                raise ValueError("pilot image path does not match its split")
    if any(row.get("split") != "train" for row in train_rows):
        raise ValueError("training rows include a non-train pair")
    if any(row.get("product_id") == dev_pair.get("product_id") for row in train_rows):
        raise ValueError("development product identity overlaps training")
    expected_names = {"metadata.json", "pilot_manifest.json"}
    for row in [*train_rows, dev_pair]:
        source_name, target_name = row.get("edit_image"), row.get("image")
        if not isinstance(source_name, str) or not isinstance(target_name, str):
            raise ValueError("pilot archive has invalid image paths")
        expected_names.update((source_name, target_name))
        if not files[source_name] or not files[target_name]:
            raise ValueError("pilot archive has empty image content")
        if (
            row.get("source_sha256") != hashlib.sha256(files[source_name]).hexdigest()
            or row.get("target_sha256") != hashlib.sha256(files[target_name]).hexdigest()
        ):
            raise ValueError("pilot image checksum does not match its row")
    if set(names) != expected_names:
        raise ValueError("pilot archive contains unreferenced or missing files")
    for name, content in files.items():
        if name not in {"metadata.json", "pilot_manifest.json"}:
            base = work_dir / "training-data" if name.startswith("train/") else work_dir
            target = base / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    training_root = work_dir / "training-data"
    training_root.mkdir(parents=True, exist_ok=True)
    (training_root / "metadata.json").write_bytes(files["metadata.json"])
    return train_rows, dev_pair, source_manifest_sha256


def _component_paths(snapshot_dirs: dict[str, Path]) -> tuple[list[list[str]], str]:
    text_files = sorted(
        str(path) for path in (snapshot_dirs["text_encoder"] / "text_encoder").glob("*.safetensors")
    )
    transformer_files = sorted(
        str(path) for path in (snapshot_dirs["transformer"] / "transformer").glob("*.safetensors")
    )
    vae_file = snapshot_dirs["vae"] / "vae" / "diffusion_pytorch_model.safetensors"
    tokenizer_path = snapshot_dirs["tokenizer"] / "tokenizer"
    if (
        not text_files
        or not transformer_files
        or not vae_file.is_file()
        or not tokenizer_path.is_dir()
    ):
        raise RuntimeError("pinned Hugging Face snapshot is missing a required model component")
    return [text_files, transformer_files, [str(vae_file)]], str(tokenizer_path)


_SNAPSHOT_ALLOW_PATTERNS = {
    "transformer": ("transformer/*.safetensors",),
    "text_encoder": ("text_encoder/*.safetensors",),
    "vae": ("vae/diffusion_pytorch_model.safetensors",),
    "tokenizer": ("tokenizer/**",),
}


def _download_model_snapshots(snapshot_download: Any) -> dict[str, Path]:
    """Fetch only the pinned files required by the local pipeline loader."""
    grouped: dict[tuple[str, str], list[str]] = {}
    for role, spec in MODEL_COMPONENTS.items():
        grouped.setdefault((spec["repo_id"], spec["revision"]), []).append(role)

    downloaded: dict[str, Path] = {}
    for (repo_id, revision), roles in grouped.items():
        allow_patterns = sorted(
            {pattern for role in roles for pattern in _SNAPSHOT_ALLOW_PATTERNS[role]}
        )
        snapshot_path = Path(
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                allow_patterns=allow_patterns,
                token=False,
            )
        ).resolve()
        if snapshot_path.name != revision:
            raise RuntimeError("pinned Hugging Face snapshot did not resolve to its revision")
        downloaded.update(dict.fromkeys(roles, snapshot_path))
    return downloaded


def _validate_trainable_parameter_names(names: list[str]) -> None:
    if not names or any("lora_" not in name for name in names):
        raise RuntimeError("pilot may train only LoRA parameters")


def _validate_gradient_norm(norm: float) -> None:
    if not math.isfinite(norm):
        raise FloatingPointError("LoRA gradient norm must be finite")


def _validate_finite_adapter_weights(finite_flags: list[bool]) -> None:
    if not finite_flags or not all(finite_flags):
        raise FloatingPointError("final adapter weights must be finite")


def _guard_optimizer_update(completed_updates: int) -> None:
    if completed_updates >= TRAINING_STEPS:
        raise RuntimeError("pilot eight-update limit reached before optimizer step")


def _load_pipe(snapshot_dirs: dict[str, Path], device: str) -> Any:
    import torch  # type: ignore[import-not-found]
    from diffsynth.pipelines.flux2_image import (  # type: ignore[import-not-found]
        Flux2ImagePipeline,
        ModelConfig,
    )

    model_paths, tokenizer_path = _component_paths(snapshot_dirs)
    configs = [ModelConfig(path=paths) for paths in model_paths]
    return Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=configs,
        tokenizer_config=ModelConfig(path=tokenizer_path),
    )


def _generate(pipe: Any, prompt: str, reference: Any, seed: int) -> Any:
    return pipe(
        prompt=prompt,
        edit_image=[reference],
        seed=seed,
        rand_device=pipe.device,
        num_inference_steps=16,
        cfg_scale=4,
        height=512,
        width=512,
    )


RELOAD_LOAD_METHOD = (
    "Flux2ImageTrainingModule state conversion and mapping followed by "
    "pipe.dit.load_state_dict(strict=False), with exact restored LoRA "
    "tensor and module validation"
)


def _reload_and_generate(
    snapshot_dirs: dict[str, Path],
    adapter_path: Path,
    prompt: str,
    reference: Any,
    seed: int,
    torch: Any,
) -> tuple[Any, dict[str, Any]]:
    """Load through the strict diagnostic path and keep its model alive through inference."""
    from .reload_diagnostic import _load_replay_model

    reload_model, reloaded_pipe, validation = _load_replay_model(snapshot_dirs, adapter_path, torch)
    try:
        reloaded_image = _generate(reloaded_pipe, prompt, reference, seed).convert("RGB")
        receipt_fields = {
            "adapter_validation": validation["adapter_validation"],
            "load_method": RELOAD_LOAD_METHOD,
        }
        return reloaded_image, receipt_fields
    finally:
        del reloaded_pipe, reload_model
        gc.collect()
        torch.cuda.empty_cache()


def run_remote_training(payload_bytes: bytes, run_id: str) -> bytes:
    """Train and verify one adapter, returning a bounded self-contained result archive."""
    stage = "input_validation"
    started = time.monotonic()
    work_root: Path | None = None
    progress: dict[str, Any] = {
        "optimizer_updates": 0,
        "optimizer_steps_attempted": 0,
        "losses": [],
        "gradient_steps": [],
        "elapsed_seconds": 0.0,
    }
    try:
        if (
            not isinstance(run_id, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", run_id)
            or ".." in run_id
        ):
            raise ValueError("invalid run identifier")
        work_root = Path(tempfile.mkdtemp(prefix=f"pilot-{run_id}-"))
        train_rows, dev_row, source_manifest_sha256 = _load_payload(
            payload_bytes, run_id, work_root
        )
        if source_manifest_sha256 != APPROVED_DATASET_MANIFEST_SHA256:
            raise ValueError("pilot data manifest does not match the reviewed dataset")
        validate_training_contract(
            train_pair_count=len(train_rows),
            batch_size=1,
            dataset_repeat=1,
            num_epochs=1,
            gradient_accumulation_steps=1,
            process_count=1,
            save_steps=TRAINING_STEPS,
        )

        stage = "runtime_import"
        import torch
        from accelerate import Accelerator  # type: ignore[import-not-found]
        from diffsynth.core import UnifiedDataset  # type: ignore[import-not-found]
        from diffsynth.diffusion import (  # type: ignore[import-not-found]
            ModelLogger,
            launch_training_task,
        )
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
        from PIL import Image, ImageChops

        sys.path.insert(0, "/opt/DiffSynth-Studio")
        from examples.flux2.model_training.train import (  # type: ignore[import-not-found]
            Flux2ImageTrainingModule,
        )

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("pilot requires exactly one available CUDA GPU")
        torch.manual_seed(20261008)
        if hasattr(torch.cuda, "manual_seed_all"):
            torch.cuda.manual_seed_all(20261008)
        torch.cuda.reset_peak_memory_stats()

        stage = "model_download"
        downloaded = _download_model_snapshots(snapshot_download)

        stage = "base_generation"
        dev_source_path = work_root / dev_row["edit_image"]
        dev_target_path = work_root / dev_row["image"]
        with Image.open(dev_source_path) as opened:
            reference = opened.convert("RGB")
        with Image.open(dev_target_path) as opened:
            dev_target = opened.convert("RGB")
        with Image.open(dev_source_path) as opened:
            source_check = opened.convert("RGB")
        if (
            reference.size != (512, 512)
            or dev_target.size != reference.size
            or source_check.size != reference.size
        ):
            raise ValueError("pilot development images must be 512x512")
        base_pipe = _load_pipe(downloaded, "cuda")
        base_image = _generate(base_pipe, dev_row["prompt"], reference, seed=7321).convert("RGB")
        base_image.save(work_root / "base.png", format="PNG", optimize=True)
        del base_pipe
        torch.cuda.empty_cache()

        stage = "training_setup"
        accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision="bf16")
        if accelerator.num_processes != 1:
            raise RuntimeError("pilot requires a single Accelerate process")
        data_root = work_root / "training-data"
        training_dataset = UnifiedDataset(
            base_path=str(data_root),
            metadata_path=str(data_root / "metadata.json"),
            repeat=1,
            data_file_keys=["image", "edit_image"],
            main_data_operator=UnifiedDataset.default_image_operator(
                base_path=str(data_root),
                max_pixels=262144,
                height=512,
                width=512,
                height_division_factor=16,
                width_division_factor=16,
            ),
        )
        if len(training_dataset) != TRAINING_PAIR_COUNT:
            raise RuntimeError("DiffSynth dataset loader did not produce exactly eight samples")
        model = Flux2ImageTrainingModule(
            model_paths=json.dumps(
                [
                    sorted(
                        str(path)
                        for path in (downloaded["text_encoder"] / "text_encoder").glob(
                            "*.safetensors"
                        )
                    ),
                    sorted(
                        str(path)
                        for path in (downloaded["transformer"] / "transformer").glob(
                            "*.safetensors"
                        )
                    ),
                    [str(downloaded["vae"] / "vae" / "diffusion_pytorch_model.safetensors")],
                ]
            ),
            tokenizer_path=str(downloaded["tokenizer"] / "tokenizer"),
            lora_base_model="dit",
            lora_target_modules=LORA_TARGET_MODULES,
            lora_rank=32,
            use_gradient_checkpointing=True,
            extra_inputs="edit_image",
            remove_prefix_in_ckpt="pipe.dit.",
            device="cuda",
        )
        unwrapped = model
        trainable = [
            (name, parameter)
            for name, parameter in unwrapped.named_parameters()
            if parameter.requires_grad
        ]
        if not trainable:
            raise RuntimeError("DiffSynth did not expose any trainable LoRA parameters")
        _validate_trainable_parameter_names([name for name, _parameter in trainable])
        initial = {name: parameter.detach().float().cpu().clone() for name, parameter in trainable}
        grad_steps: list[dict[str, float | int | bool]] = []

        class _PilotAdamW(torch.optim.AdamW):  # type: ignore[misc]
            def step(self, closure: Any = None) -> Any:
                _guard_optimizer_update(len(grad_steps))
                grads = [
                    parameter.grad.detach().float().reshape(-1)
                    for _, parameter in trainable
                    if parameter.grad is not None
                ]
                if not grads:
                    raise RuntimeError("no LoRA gradients were produced")
                flattened = torch.cat(grads)
                finite = bool(torch.isfinite(flattened).all().item())
                norm = float(torch.linalg.vector_norm(flattened).item())
                _validate_gradient_norm(norm)
                nonzero = bool(torch.count_nonzero(flattened).item())
                if not finite or not nonzero:
                    raise FloatingPointError("LoRA gradients must be finite and nonzero")
                progress["optimizer_steps_attempted"] = len(grad_steps) + 1
                result = super().step(closure)
                grad_steps.append(
                    {"finite": finite, "norm": norm, "nonzero": nonzero, "tensors": len(grads)}
                )
                progress["gradient_steps"] = grad_steps
                progress["optimizer_updates"] = len(grad_steps)
                return result

        # DiffSynth imports the configured optimizer by module path from its upstream loop.
        optimizer_symbol = "_PilotAdamW"
        setattr(sys.modules[__name__], optimizer_symbol, _PilotAdamW)

        output_dir = work_root / "training-output"
        if output_dir.exists():
            raise FileExistsError("pilot training output already exists")

        class _CheckedLogger(ModelLogger):  # type: ignore[misc]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self.steps = 0
                self.losses: list[float] = []
                self.step_seconds: list[float] = []
                self.last_step_time = time.monotonic()

            def on_step_end(
                self, accelerator: Any, model: Any, save_steps: int | None = None, **kwargs: Any
            ) -> None:
                loss = kwargs.get("loss")
                if loss is None or not bool(torch.isfinite(loss).all().item()):
                    raise FloatingPointError("training loss is missing or non-finite")
                self.losses.append(float(loss.detach().float().mean().item()))
                self.steps += 1
                progress["optimizer_updates"] = self.steps
                progress["losses"] = self.losses
                now = time.monotonic()
                self.step_seconds.append(now - self.last_step_time)
                self.last_step_time = now
                progress["step_seconds"] = self.step_seconds
                progress["training_seconds"] = max(0.0, now - training_started)
                progress["max_allocated_gpu_bytes"] = int(torch.cuda.max_memory_allocated())
                super().on_step_end(accelerator, model, save_steps, **kwargs)

        logger = _CheckedLogger(str(output_dir), remove_prefix_in_ckpt="pipe.dit.")
        training_args = type(
            "PilotArgs",
            (),
            {
                "learning_rate": 1e-4,
                "weight_decay": 0.0,
                "dataset_num_workers": 0,
                "save_steps": TRAINING_STEPS,
                "num_epochs": 1,
                "enable_model_cpu_offload": False,
                "enable_optimizer_cpu_offload": False,
                "cpu_offload_split_threshold": None,
                "customized_optimizer": "product_scene_editor.remote._PilotAdamW",
                "output_path": str(output_dir),
            },
        )()

        stage = "training"
        torch.cuda.synchronize()
        training_started = time.monotonic()
        training_finished = False
        launch_training_task(
            accelerator,
            training_dataset,
            model,
            logger,
            args=training_args,
        )
        torch.cuda.synchronize()
        training_seconds = time.monotonic() - training_started
        progress["training_seconds"] = training_seconds
        progress["step_seconds"] = logger.step_seconds
        progress["max_allocated_gpu_bytes"] = int(torch.cuda.max_memory_allocated())
        training_finished = True
        if logger.steps != TRAINING_STEPS or len(grad_steps) != TRAINING_STEPS:
            raise RuntimeError("DiffSynth completed an unexpected number of optimizer updates")
        adapter_path = output_dir / "step-8.safetensors"
        if not adapter_path.is_file() or adapter_path.stat().st_size == 0:
            raise RuntimeError("DiffSynth did not save the step-8 LoRA adapter")
        progress["saved_artifacts"] = ["adapter/step-8.safetensors", "images/base.png"]
        trained_model = accelerator.unwrap_model(model)
        final: dict[str, Any] = {}
        final_finite_flags: list[bool] = []
        for name, parameter in trained_model.named_parameters():
            if parameter.requires_grad:
                final_value = parameter.detach().float().cpu()
                final[name] = final_value
                final_finite_flags.append(bool(torch.isfinite(final_value).all().item()))
        _validate_finite_adapter_weights(final_finite_flags)
        if set(initial) != set(final):
            raise RuntimeError("trainable adapter parameter set changed during training")
        changed = [name for name in initial if not torch.equal(initial[name], final[name])]
        progress["changed_adapter_tensors"] = len(changed)
        if not changed:
            raise RuntimeError("LoRA parameters did not change during the eight updates")
        max_parameter_delta = max(
            float((final[name] - initial[name]).abs().max().item()) for name in changed
        )
        progress["max_parameter_delta"] = max_parameter_delta

        stage = "trained_generation"
        trained_pipe = trained_model.pipe
        trained_image = _generate(trained_pipe, dev_row["prompt"], reference, seed=7321).convert(
            "RGB"
        )
        trained_image.save(work_root / "trained.png", format="PNG", optimize=True)
        progress["saved_artifacts"] = [
            "adapter/step-8.safetensors",
            "images/base.png",
            "images/trained.png",
        ]

        delattr(sys.modules[__name__], optimizer_symbol)
        accelerator.free_memory()
        trainable.clear()
        del (
            trained_pipe,
            trained_model,
            unwrapped,
            model,
            training_dataset,
            parameter,
            _PilotAdamW,
            accelerator,
        )
        gc.collect()
        torch.cuda.empty_cache()

        stage = "adapter_reload"
        reloaded_image, reload_receipt_fields = _reload_and_generate(
            downloaded, adapter_path, dev_row["prompt"], reference, 7321, torch
        )
        reloaded_image.save(work_root / "reloaded.png", format="PNG", optimize=True)
        progress["saved_artifacts"] = [
            "adapter/step-8.safetensors",
            "images/base.png",
            "images/trained.png",
            "images/reloaded.png",
        ]
        torch.cuda.empty_cache()
        trained_again = Image.open(work_root / "trained.png").convert("RGB")
        difference = ImageChops.difference(trained_again, reloaded_image)
        diff_values = list(difference.getdata())
        mean_abs_error = sum(sum(pixel) for pixel in diff_values) / (len(diff_values) * 3)
        max_abs_error = max(max(pixel) for pixel in diff_values)
        pixels_over_8 = sum(max(pixel) > 8 for pixel in diff_values)
        fraction_over_8 = pixels_over_8 / len(diff_values)
        if mean_abs_error > 2.0 or fraction_over_8 > 0.01:
            raise RuntimeError("reloaded adapter generation exceeded the explicit pixel tolerance")
        stage = "result_packaging"
        package_versions: dict[str, str] = {}
        for package in (
            *RUNTIME_PINS,
            "accelerate",
            "diffsynth",
            "modelscope",
            "einops",
            "imageio",
            "ftfy",
            "pandas",
        ):
            try:
                package_versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                package_versions[package] = "not-installed"
        receipt = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "passed",
            "trainer": {
                "repo": "https://github.com/modelscope/DiffSynth-Studio",
                "commit": DIFFSYNTH_COMMIT,
            },
            "components": MODEL_COMPONENTS,
            "training": {
                "training_pairs": len(train_rows),
                "development_pairs": 1,
                "source_dataset_manifest_sha256": source_manifest_sha256,
                "optimizer_updates": logger.steps,
                "gradient_accumulation_steps": 1,
                "dataset_repeat": 1,
                "epochs": 1,
                "save_steps": TRAINING_STEPS,
                "losses": logger.losses,
                "gradient_steps": grad_steps,
                "changed_adapter_tensors": len(changed),
                "max_parameter_delta": max_parameter_delta,
                "training_seconds": training_seconds,
                "step_seconds": logger.step_seconds,
            },
            "generation_reload_check": {
                "seed": 7321,
                "steps": 16,
                "mean_absolute_pixel_error": mean_abs_error,
                "max_absolute_pixel_error": max_abs_error,
                "fraction_of_pixels_over_8": fraction_over_8,
                "tolerance": {"mean_absolute_pixel_error_max": 2.0, "fraction_over_8_max": 0.01},
                **reload_receipt_fields,
            },
            "resources": {
                "gpu": torch.cuda.get_device_name(0),
                "max_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
                "worker_elapsed_seconds": time.monotonic() - started,
            },
            "runtime_versions": package_versions,
            "recorded_at_utc": datetime.now(UTC).isoformat(),
        }
        artifacts = {
            "adapter/step-8.safetensors": adapter_path.read_bytes(),
            "images/base.png": (work_root / "base.png").read_bytes(),
            "images/trained.png": (work_root / "trained.png").read_bytes(),
            "images/reloaded.png": (work_root / "reloaded.png").read_bytes(),
            "receipt.json": (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode("utf-8"),
        }
        return _bundle(artifacts)
    except Exception as exc:
        progress["elapsed_seconds"] = time.monotonic() - started
        if "training_started" in locals() and not training_finished:
            progress["training_seconds"] = max(0.0, time.monotonic() - training_started)
        if "torch" in locals():
            try:
                torch.cuda.synchronize()
                progress["max_allocated_gpu_bytes"] = int(torch.cuda.max_memory_allocated())
            except Exception:
                pass
        if isinstance(exc, _WorkerFailure):
            stage = exc.stage
            error = exc.cause
        else:
            error = exc
        return _failure_bundle(
            run_id if isinstance(run_id, str) else "invalid", stage, error, progress, work_root
        )
    finally:
        if work_root is not None:
            shutil.rmtree(work_root, ignore_errors=True)
