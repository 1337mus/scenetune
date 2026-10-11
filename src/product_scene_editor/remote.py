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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .pilot import APPROVED_DATASET_MANIFEST_SHA256, MAX_PAYLOAD_BYTES, TRAINING_PAIR_COUNT

PROFILE = "reflex-personal"
WORKSPACE = "rajath-61258"
ENVIRONMENT = "main"
TRAINING_STEPS = 8
_MATCHED_TRAINING_STEPS = 144
_MATCHED_DATASET_MANIFEST_SHA256 = (
    "206b07f99b2fd1c001b94898f3cacfa7010bacb7e08aa731b972135000e9a8f9"
)
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


@dataclass(frozen=True, slots=True)
class _TrainingProfile:
    name: str
    train_pair_count: int
    dev_pair_count: int
    epochs: int
    updates: int
    dataset_manifest_sha256: str | None


_PILOT_PROFILE = _TrainingProfile(
    name="pilot",
    train_pair_count=TRAINING_PAIR_COUNT,
    dev_pair_count=1,
    epochs=1,
    updates=TRAINING_STEPS,
    dataset_manifest_sha256=APPROVED_DATASET_MANIFEST_SHA256,
)
_MATCHED_PROFILE = _TrainingProfile(
    name="matched-study",
    train_pair_count=18,
    dev_pair_count=9,
    epochs=8,
    updates=_MATCHED_TRAINING_STEPS,
    dataset_manifest_sha256=_MATCHED_DATASET_MANIFEST_SHA256,
)
_CONTROLLED3D_PROFILE = _TrainingProfile(
    name="controlled3d-study",
    train_pair_count=18,
    dev_pair_count=9,
    epochs=8,
    updates=_MATCHED_TRAINING_STEPS,
    dataset_manifest_sha256=None,
)
CONTROLLED3D_METRIC_REFERENCE = {
    "foreground_mean_absolute_error": "source",
    "target_foreground_mean_absolute_error": "target",
    "source_to_target_foreground_mean_absolute_error": "source_to_target_baseline",
    "limitation": (
        "Pixel metrics measure reference similarity; they do not prove label text or shape quality."
    ),
}


def _require_training_profile(profile: object) -> _TrainingProfile:
    """Accept only the two closed internal profiles, by object identity."""
    if profile is _PILOT_PROFILE:
        return _PILOT_PROFILE
    if profile is _MATCHED_PROFILE:
        return _MATCHED_PROFILE
    if profile is _CONTROLLED3D_PROFILE:
        return _CONTROLLED3D_PROFILE
    raise ValueError("training profile must be one of the three canonical private profiles")


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


def _validate_training_profile_contract(
    profile: object,
    *,
    train_pair_count: int,
    batch_size: int,
    dataset_repeat: int,
    num_epochs: int,
    gradient_accumulation_steps: int,
    process_count: int,
    save_steps: int,
) -> int:
    selected = _require_training_profile(profile)
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
    expected = (
        selected.train_pair_count,
        1,
        1,
        selected.epochs,
        1,
        1,
        selected.updates,
    )
    if values != expected:
        raise ValueError(
            f"{selected.name} schedule must make and save exactly "
            f"{selected.updates} single-sample updates"
        )
    return selected.updates


def _validate_matched_training_contract(
    *,
    train_pair_count: int,
    batch_size: int,
    dataset_repeat: int,
    num_epochs: int,
    gradient_accumulation_steps: int,
    process_count: int,
    save_steps: int,
) -> int:
    """Match the frozen 18-pair, eight-epoch, 144-update study schedule."""
    return _validate_training_profile_contract(
        _MATCHED_PROFILE,
        train_pair_count=train_pair_count,
        batch_size=batch_size,
        dataset_repeat=dataset_repeat,
        num_epochs=num_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        process_count=process_count,
        save_steps=save_steps,
    )


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
    profile: object = _PILOT_PROFILE,
) -> bytes:
    selected = _require_training_profile(profile)
    saved_artifacts: dict[str, bytes] = {}
    if work_root is not None and selected is _PILOT_PROFILE:
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
    elif work_root is not None:
        matched_paths: list[tuple[str, Path]] = [
            (
                "adapter/step-144.safetensors",
                work_root / "training-output" / "step-144.safetensors",
            )
        ]
        for relative_name in progress.get("saved_artifacts", []):
            if not isinstance(relative_name, str) or not relative_name.startswith("images/"):
                continue
            path = work_root / relative_name
            matched_paths.append((relative_name, path))
        for name, path in matched_paths:
            if path.is_file() and not path.is_symlink() and path.stat().st_size:
                content = path.read_bytes()
                try:
                    _bundle({**saved_artifacts, name: content})
                except ValueError:
                    continue
                saved_artifacts[name] = content
    failure_progress: dict[str, Any] = {
        "optimizer_updates": min(int(progress.get("optimizer_updates", 0)), selected.updates),
        "optimizer_steps_attempted": min(
            int(progress.get("optimizer_steps_attempted", 0)), selected.updates
        ),
        "losses": list(progress.get("losses", []))[: selected.updates],
        "gradient_steps": list(progress.get("gradient_steps", []))[: selected.updates],
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
            failure_progress[key] = (
                list(value)[: selected.updates] if key == "step_seconds" else value
            )

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
    if selected is not _PILOT_PROFILE:
        receipt["study_type"] = (
            "controlled3d144" if selected is _CONTROLLED3D_PROFILE else "matched144"
        )
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


def _load_matched_payload(
    payload_bytes: bytes, run_id: str, work_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    from .matched_study import validate_matched_study_payload_bytes

    payload = validate_matched_study_payload_bytes(payload_bytes, run_id)
    if payload.source_dataset_manifest_sha256 != _MATCHED_PROFILE.dataset_manifest_sha256:
        raise ValueError("matched-study data manifest does not match the reviewed dataset")
    training_root = work_dir / "training-data"
    for name, content in payload.files.items():
        if name in {"metadata.json", "matched_manifest.json"}:
            continue
        relative = PurePosixPath(name)
        if relative.parts[0] not in {"train", "dev"}:
            raise ValueError("matched-study payload contains an unexpected asset path")
        base = training_root if relative.parts[0] == "train" else work_dir
        target = base.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    training_root.mkdir(parents=True, exist_ok=True)
    training_root.joinpath("metadata.json").write_bytes(payload.files["metadata.json"])
    return list(payload.train_rows), list(payload.dev_rows), payload.source_dataset_manifest_sha256


def _load_controlled3d_payload(
    payload_bytes: bytes, run_id: str, work_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, dict[str, str]]:
    from .controlled3d_study import validate_controlled3d_study_payload_bytes

    payload = validate_controlled3d_study_payload_bytes(payload_bytes, run_id)
    training_root = work_dir / "training-data"
    for name, content in payload.files.items():
        if name in {"metadata.json", "controlled3d_manifest.json"}:
            continue
        relative = PurePosixPath(name)
        if relative.parts[0] not in {"train", "dev"}:
            raise ValueError("Controlled3D payload contains an unexpected asset path")
        base = training_root if relative.parts[0] == "train" else work_dir
        target = base.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    training_root.mkdir(parents=True, exist_ok=True)
    training_root.joinpath("metadata.json").write_bytes(payload.files["metadata.json"])
    bindings = {
        "source_registry_sha256": payload.source_registry_sha256,
        "validation_proof_sha256": payload.validation_proof_sha256,
        "train_export_completion_sha256": payload.train_export_completion_sha256,
    }
    return (
        list(payload.train_rows),
        list(payload.dev_rows),
        payload.source_dataset_manifest_sha256,
        bindings,
    )


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


def _guard_profile_optimizer_update(profile: object, completed_updates: int) -> None:
    selected = _require_training_profile(profile)
    if completed_updates >= selected.updates:
        if selected is _PILOT_PROFILE:
            raise RuntimeError("pilot eight-update limit reached before optimizer step")
        raise RuntimeError("matched-study 144-update limit reached before optimizer step")


def _guard_optimizer_update(completed_updates: int) -> None:
    _guard_profile_optimizer_update(_PILOT_PROFILE, completed_updates)


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


def _matched_reload_pixel_metrics(trained: Any, reloaded: Any) -> dict[str, float | int]:
    from PIL import ImageChops, ImageStat

    if (
        getattr(trained, "mode", None) != "RGB"
        or getattr(reloaded, "mode", None) != "RGB"
        or trained.size != (512, 512)
        or reloaded.size != trained.size
    ):
        raise ValueError("matched reload images must both be 512x512 RGB")
    difference = ImageChops.difference(trained, reloaded)
    channels = difference.split()
    max_channel = ImageChops.lighter(ImageChops.lighter(channels[0], channels[1]), channels[2])
    mean_absolute_error = sum(ImageStat.Stat(difference).mean) / 3
    max_absolute_error = max(difference.tobytes())
    over_8 = max_channel.point(lambda value: 255 if value > 8 else 0)
    fraction_over_8 = over_8.histogram()[255] / (512 * 512)
    return {
        "mean_absolute_pixel_error": mean_absolute_error,
        "fraction_of_pixels_over_8": fraction_over_8,
        "max_absolute_pixel_error": max_absolute_error,
    }


def _matched_image_name(pair_id: str, mode: str) -> str:
    if (
        not isinstance(pair_id, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", pair_id)
        or ".." in pair_id
        or mode not in {"base", "trained", "reloaded"}
    ):
        raise ValueError("matched development pair ID or image mode is invalid")
    return f"images/{pair_id}/{mode}.png"


def _prepare_matched_development_assets(
    rows: list[dict[str, Any]],
    work_root: Path,
    image_module: Any,
    *,
    allow_relighting: bool = False,
) -> dict[str, dict[str, Any]]:
    assets: dict[str, dict[str, Any]] = {}
    for row in rows:
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str):
            raise ValueError("matched development pair IDs must be strings")
        _matched_image_name(pair_id, "base")
        if pair_id in assets:
            raise ValueError("matched development pair IDs must be unique")
        with image_module.open(work_root / row["edit_image"]) as opened:
            source = opened.convert("RGB")
        with image_module.open(work_root / row["image"]) as opened:
            target = opened.convert("RGB")
        with image_module.open(work_root / row["mask"]) as opened:
            mask = opened.convert("L")
        if allow_relighting:
            _validate_controlled3d_foreground_inputs(source, target, mask)
        else:
            _validate_matched_foreground_identity(source, target, mask)
        assets[pair_id] = {
            "row": row,
            "prompt": row["prompt"],
            "source": source,
            "target": target,
            "mask": mask,
        }
    if len(assets) != _MATCHED_PROFILE.dev_pair_count:
        raise ValueError("matched study must include exactly nine development pairs")
    return assets


def _generate_matched_pair_images(
    pipe: Any,
    development_assets: dict[str, dict[str, Any]],
    work_root: Path,
    mode: str,
    seed: int,
    progress: dict[str, Any],
) -> dict[str, Any]:
    images: dict[str, Any] = {}
    saved = list(progress.get("saved_artifacts", []))
    for pair_id, assets in development_assets.items():
        image = _generate(pipe, assets["prompt"], assets["source"], seed=seed).convert("RGB")
        relative_name = _matched_image_name(pair_id, mode)
        image_path = work_root / relative_name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path, format="PNG", optimize=True)
        images[pair_id] = image
        if relative_name not in saved:
            saved.append(relative_name)
        progress["saved_artifacts"] = saved
    return images


def _score_matched_generations(
    development_assets: dict[str, dict[str, Any]], generated_by_mode: dict[str, dict[str, Any]]
) -> dict[str, dict[str, dict[str, float | int]]]:
    from .matched_study import score_pair

    metrics: dict[str, dict[str, dict[str, float | int]]] = {}
    for pair_id, assets in development_assets.items():
        metrics[pair_id] = {
            mode: score_pair(
                assets["source"], assets["target"], assets["mask"], generated_by_mode[mode][pair_id]
            )
            for mode in ("base", "trained", "reloaded")
        }
    return metrics


def _score_controlled3d_generations(
    development_assets: dict[str, dict[str, Any]], generated_by_mode: dict[str, dict[str, Any]]
) -> dict[str, dict[str, dict[str, float | int]]]:
    from PIL import ImageChops, ImageStat

    metrics = _score_matched_generations(development_assets, generated_by_mode)
    for pair_id, assets in development_assets.items():
        source_target = ImageChops.difference(assets["source"], assets["target"])
        baseline = sum(ImageStat.Stat(source_target, mask=assets["mask"]).mean) / 3
        for mode in ("base", "trained", "reloaded"):
            difference = ImageChops.difference(assets["target"], generated_by_mode[mode][pair_id])
            metrics[pair_id][mode]["target_foreground_mean_absolute_error"] = (
                sum(ImageStat.Stat(difference, mask=assets["mask"]).mean) / 3
            )
            metrics[pair_id][mode]["source_to_target_foreground_mean_absolute_error"] = baseline
    return metrics


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


def _validate_controlled3d_foreground_inputs(source: Any, target: Any, mask: Any) -> None:
    if (
        getattr(source, "mode", None) != "RGB"
        or getattr(target, "mode", None) != "RGB"
        or getattr(mask, "mode", None) != "L"
        or source.size != (512, 512)
        or target.size != source.size
        or mask.size != source.size
    ):
        raise ValueError("Controlled3D source, target, and mask must be aligned 512x512 images")
    histogram = mask.histogram()
    if any(histogram[1:255]) or not histogram[0] or not histogram[255]:
        raise ValueError("Controlled3D mask must contain binary foreground and background")


def _validate_matched_foreground_identity(source: Any, target: Any, mask: Any) -> None:
    from PIL import ImageChops

    if (
        getattr(source, "mode", None) != "RGB"
        or getattr(target, "mode", None) != "RGB"
        or getattr(mask, "mode", None) != "L"
        or source.size != (512, 512)
        or target.size != source.size
        or mask.size != source.size
    ):
        raise ValueError("matched source, target, and mask must be aligned 512x512 images")
    histogram = mask.histogram()
    if any(histogram[1:255]) or not histogram[0] or not histogram[255]:
        raise ValueError("matched mask must contain both binary foreground and background")
    for channel in ImageChops.difference(source, target).split():
        if ImageChops.multiply(channel, mask).getbbox() is not None:
            raise ValueError("matched source and target foreground pixels must be identical")


def _reload_and_generate_pairs(
    snapshot_dirs: dict[str, Path],
    adapter_path: Path,
    development_assets: dict[str, dict[str, Any]],
    seed: int,
    torch: Any,
    *,
    work_root: Path | None = None,
    progress: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Strictly reload once and retain it while each result is generated and saved."""
    from .reload_diagnostic import _load_replay_model

    if (work_root is None) != (progress is None):
        raise ValueError("matched reload persistence requires both work_root and progress")
    reload_model, reloaded_pipe, validation = _load_replay_model(snapshot_dirs, adapter_path, torch)
    try:
        images: dict[str, Any] = {}
        for pair_id, assets in development_assets.items():
            image = _generate(reloaded_pipe, assets["prompt"], assets["source"], seed=seed).convert(
                "RGB"
            )
            images[pair_id] = image
            if work_root is not None and progress is not None:
                relative_name = _matched_image_name(pair_id, "reloaded")
                image_path = work_root / relative_name
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(image_path, format="PNG", optimize=True)
                saved = list(progress.get("saved_artifacts", []))
                if relative_name not in saved:
                    saved.append(relative_name)
                progress["saved_artifacts"] = saved
        receipt_fields = {
            "adapter_validation": validation["adapter_validation"],
            "load_method": RELOAD_LOAD_METHOD,
        }
        return images, receipt_fields
    finally:
        del reloaded_pipe, reload_model
        gc.collect()
        torch.cuda.empty_cache()


def run_remote_training(payload_bytes: bytes, run_id: str) -> bytes:
    """Train the fixed eight-update pilot and preserve its R4 result contract."""
    return _run_remote_training(payload_bytes, run_id, _PILOT_PROFILE)


def run_remote_matched_study(payload_bytes: bytes, run_id: str) -> bytes:
    """Train the one fixed 144-update adapter and compare it across nine dev pairs."""
    return _run_remote_training(payload_bytes, run_id, _MATCHED_PROFILE)


def run_remote_controlled3d_study(payload_bytes: bytes, run_id: str) -> bytes:
    """Train the fixed Controlled3D 144-update profile against nine held-out edits."""
    return _run_remote_training(payload_bytes, run_id, _CONTROLLED3D_PROFILE)


def _run_remote_training(payload_bytes: bytes, run_id: str, profile: object) -> bytes:
    """Run one of the three closed training profiles through the shared worker body."""
    selected = _require_training_profile(profile)
    stage = "input_validation"
    started = time.monotonic()
    work_root: Path | None = None
    controlled3d_bindings: dict[str, str] = {}
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
        work_root = Path(tempfile.mkdtemp(prefix=f"{selected.name}-{run_id}-"))
        if selected is _PILOT_PROFILE:
            train_rows, dev_row, source_manifest_sha256 = _load_payload(
                payload_bytes, run_id, work_root
            )
            dev_rows = [dev_row]
        elif selected is _MATCHED_PROFILE:
            train_rows, dev_rows, source_manifest_sha256 = _load_matched_payload(
                payload_bytes, run_id, work_root
            )
            dev_row = dev_rows[0]
        else:
            (train_rows, dev_rows, source_manifest_sha256, controlled3d_bindings) = (
                _load_controlled3d_payload(payload_bytes, run_id, work_root)
            )
            dev_row = dev_rows[0]
        expected_manifest = selected.dataset_manifest_sha256
        if selected is _CONTROLLED3D_PROFILE:
            from .controlled3d_study import APPROVED_CONTROLLED3D_MANIFEST_SHA256

            expected_manifest = APPROVED_CONTROLLED3D_MANIFEST_SHA256
        if expected_manifest != source_manifest_sha256:
            raise ValueError(f"{selected.name} data manifest does not match the reviewed dataset")
        _validate_training_profile_contract(
            selected,
            train_pair_count=len(train_rows),
            batch_size=1,
            dataset_repeat=1,
            num_epochs=selected.epochs,
            gradient_accumulation_steps=1,
            process_count=1,
            save_steps=selected.updates,
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
        development_assets: dict[str, dict[str, Any]] = {}
        base_images: dict[str, Any] = {}
        if selected is _PILOT_PROFILE:
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
            base_image = _generate(base_pipe, dev_row["prompt"], reference, seed=7321).convert(
                "RGB"
            )
            base_image.save(work_root / "base.png", format="PNG", optimize=True)
            del base_pipe
        else:
            development_assets = _prepare_matched_development_assets(
                dev_rows,
                work_root,
                Image,
                allow_relighting=selected is _CONTROLLED3D_PROFILE,
            )
            base_pipe = _load_pipe(downloaded, "cuda")
            base_images = _generate_matched_pair_images(
                base_pipe, development_assets, work_root, "base", 7321, progress
            )
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
        if len(training_dataset) != selected.train_pair_count:
            if selected is _PILOT_PROFILE:
                raise RuntimeError("DiffSynth dataset loader did not produce exactly eight samples")
            raise RuntimeError("DiffSynth dataset loader did not produce exactly 18 samples")
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
                _guard_profile_optimizer_update(selected, len(grad_steps))
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
                "save_steps": selected.updates,
                "num_epochs": selected.epochs,
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
        if logger.steps != selected.updates or len(grad_steps) != selected.updates:
            raise RuntimeError("DiffSynth completed an unexpected number of optimizer updates")
        adapter_filename = f"step-{selected.updates}.safetensors"
        adapter_path = output_dir / adapter_filename
        if not adapter_path.is_file() or adapter_path.stat().st_size == 0:
            raise RuntimeError(f"DiffSynth did not save the {adapter_filename} LoRA adapter")
        if selected is _PILOT_PROFILE:
            progress["saved_artifacts"] = ["adapter/step-8.safetensors", "images/base.png"]
        else:
            progress["saved_artifacts"] = [
                f"adapter/{adapter_filename}",
                *list(progress.get("saved_artifacts", [])),
            ]
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
            if selected is _PILOT_PROFILE:
                raise RuntimeError("LoRA parameters did not change during the eight updates")
            raise RuntimeError("LoRA parameters did not change during the 144 updates")
        max_parameter_delta = max(
            float((final[name] - initial[name]).abs().max().item()) for name in changed
        )
        progress["max_parameter_delta"] = max_parameter_delta

        stage = "trained_generation"
        trained_pipe = trained_model.pipe
        if selected is _PILOT_PROFILE:
            trained_image = _generate(
                trained_pipe, dev_row["prompt"], reference, seed=7321
            ).convert("RGB")
            trained_image.save(work_root / "trained.png", format="PNG", optimize=True)
            progress["saved_artifacts"] = [
                "adapter/step-8.safetensors",
                "images/base.png",
                "images/trained.png",
            ]
        else:
            trained_images = _generate_matched_pair_images(
                trained_pipe, development_assets, work_root, "trained", 7321, progress
            )

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
        matched_reload_per_pair: dict[str, dict[str, float | int]] = {}
        matched_metrics: dict[str, dict[str, dict[str, float | int]]] = {}
        if selected is _PILOT_PROFILE:
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
                raise RuntimeError(
                    "reloaded adapter generation exceeded the explicit pixel tolerance"
                )
        else:
            reloaded_images, reload_receipt_fields = _reload_and_generate_pairs(
                downloaded,
                adapter_path,
                development_assets,
                7321,
                torch,
                work_root=work_root,
                progress=progress,
            )
            for pair_id, reloaded_image in reloaded_images.items():
                per_pair = _matched_reload_pixel_metrics(trained_images[pair_id], reloaded_image)
                matched_reload_per_pair[pair_id] = per_pair
                if (
                    per_pair["mean_absolute_pixel_error"] > 2.0
                    or per_pair["fraction_of_pixels_over_8"] > 0.01
                ):
                    raise RuntimeError(
                        f"reloaded adapter generation exceeded the explicit pixel tolerance "
                        f"for {pair_id}"
                    )
            score_generations = {
                "base": base_images,
                "trained": trained_images,
                "reloaded": reloaded_images,
            }
            if selected is _CONTROLLED3D_PROFILE:
                matched_metrics = _score_controlled3d_generations(
                    development_assets, score_generations
                )
            else:
                matched_metrics = _score_matched_generations(development_assets, score_generations)
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
                "development_pairs": len(dev_rows),
                "source_dataset_manifest_sha256": source_manifest_sha256,
                "optimizer_updates": logger.steps,
                "gradient_accumulation_steps": 1,
                "dataset_repeat": 1,
                "epochs": selected.epochs,
                "save_steps": selected.updates,
                "losses": logger.losses,
                "gradient_steps": grad_steps,
                "changed_adapter_tensors": len(changed),
                "max_parameter_delta": max_parameter_delta,
                "training_seconds": training_seconds,
                "step_seconds": logger.step_seconds,
            },
            "generation_reload_check": (
                {
                    "seed": 7321,
                    "steps": 16,
                    "mean_absolute_pixel_error": mean_abs_error,
                    "max_absolute_pixel_error": max_abs_error,
                    "fraction_of_pixels_over_8": fraction_over_8,
                    "tolerance": {
                        "mean_absolute_pixel_error_max": 2.0,
                        "fraction_over_8_max": 0.01,
                    },
                    **reload_receipt_fields,
                }
                if selected is _PILOT_PROFILE
                else {
                    "seed": 7321,
                    "steps": 16,
                    "tolerance": {
                        "mean_absolute_pixel_error_max": 2.0,
                        "fraction_over_8_max": 0.01,
                    },
                    **reload_receipt_fields,
                    "per_pair": matched_reload_per_pair,
                }
            ),
            "resources": {
                "gpu": torch.cuda.get_device_name(0),
                "max_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
                "worker_elapsed_seconds": time.monotonic() - started,
            },
            "runtime_versions": package_versions,
            "recorded_at_utc": datetime.now(UTC).isoformat(),
        }
        if selected is _PILOT_PROFILE:
            artifacts = {
                "adapter/step-8.safetensors": adapter_path.read_bytes(),
                "images/base.png": (work_root / "base.png").read_bytes(),
                "images/trained.png": (work_root / "trained.png").read_bytes(),
                "images/reloaded.png": (work_root / "reloaded.png").read_bytes(),
                "receipt.json": (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode(
                    "utf-8"
                ),
            }
        else:
            receipt["study_type"] = (
                "controlled3d144" if selected is _CONTROLLED3D_PROFILE else "matched144"
            )
            receipt["metrics"] = {"per_pair": matched_metrics}
            if selected is _CONTROLLED3D_PROFILE:
                receipt["controlled3d_provenance"] = controlled3d_bindings
                receipt["metrics_reference"] = CONTROLLED3D_METRIC_REFERENCE
            artifacts = {
                f"adapter/{adapter_filename}": adapter_path.read_bytes(),
                **{
                    _matched_image_name(pair_id, mode): (
                        work_root / _matched_image_name(pair_id, mode)
                    ).read_bytes()
                    for pair_id in development_assets
                    for mode in ("base", "trained", "reloaded")
                },
                "receipt.json": (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode(
                    "utf-8"
                ),
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
            run_id if isinstance(run_id, str) else "invalid",
            stage,
            error,
            progress,
            work_root,
            selected,
        )
    finally:
        if work_root is not None:
            shutil.rmtree(work_root, ignore_errors=True)
