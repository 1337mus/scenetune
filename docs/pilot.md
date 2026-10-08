# Eight-pair FLUX.2 Klein scene-edit pilot

The pilot checks that the pinned DiffSynth edit-training path can consume the project's owned procedural package renders, make eight LoRA updates, save and reload the adapter, and return actual before/after images. Synthetic packaging tests the pipeline only; it is not evidence about real product photography or image quality.

## CPU plan

The default command only verifies the reviewed dataset and creates a write-once payload. It does not import Modal, download weights, or start compute:

```sh
uv run python -m experiments.modal_pilot \
  --dataset-dir artifacts/pilot-data-2026-10-08-r2 \
  --plan-dir artifacts/pilots/pilot-001 \
  --run-id pilot-001
```

The reviewed manifest SHA256 is `206b07f99b2fd1c001b94898f3cacfa7010bacb7e08aa731b972135000e9a8f9`. Planning verifies the whole local manifest and all image hashes, picks eight train pairs with round-robin product coverage, and adds one development pair from a separate product identity. The archive contains only those eighteen source/target images and the paired DiffSynth metadata. No masks, other development pairs, or sealed-test files are included. The `.pilot-run.json` marker binds the run ID, manifest hash, pair IDs, and payload checksum. Existing run directories are never overwritten.

The remote worker checks the archive paths, row counts, image hashes, split labels, product separation, source-manifest hash, and upload size again before unpacking it. The input limit is 16 MiB. It receives the training data and one development example as function arguments; it mounts no dataset, volume, secret, or local cache.

## Training and checks

The remote environment uses Python 3.12, Torch `2.14.1+cu130`, torchvision `0.29.1+cu130`, and the pinned packages in `product_scene_editor.remote.RUNTIME_PINS`. DiffSynth-Studio is cloned at `ee631d4907d8d6fc3d39063a2a65c00a954c9bf5`; its source tree is checked against that commit and installed without resolving its dependency ranges. HF snapshots are downloaded only inside the remote function at the exact revisions in `source-lock-candidate.json`, with anonymous `token=False` access and allow-patterns limited to `transformer/*.safetensors`, `text_encoder/*.safetensors`, `vae/diffusion_pytorch_model.safetensors`, and `tokenizer/**`. The worker loads the local component file paths.

The pinned `diffsynth/diffusion/runner.py` builds a PyTorch `DataLoader` without `batch_size`, so PyTorch uses its default of one sample per batch. `launch_training_task` runs one epoch over eight rows and calls `optimizer.step()` once per batch. `ModelLogger.on_step_end` runs after that call, increments its counter, and saves `step-8.safetensors` when `save_steps=8`.

This pinned-source calculation predicts eight updates; the actual GPU count remains unobserved until a cleared run. The worker uses DiffSynth's `Flux2ImageTrainingModule`, `launch_training_task`, scheduler, and `ModelLogger`. It enforces these checks:

- Every trainable parameter name must identify a LoRA weight.
- Losses, gradients, gradient norms, and final adapter weights must be finite; gradients must also be nonzero.
- Optimizer step nine is rejected before parameter mutation.
- The worker requires eight successful optimizer calls, eight logger callbacks, a changed adapter tensor, and the saved step-8 adapter.

The upstream implementation paths are [`diffsynth/diffusion/runner.py`](https://github.com/modelscope/DiffSynth-Studio/blob/ee631d4907d8d6fc3d39063a2a65c00a954c9bf5/diffsynth/diffusion/runner.py), [`diffsynth/diffusion/logger.py`](https://github.com/modelscope/DiffSynth-Studio/blob/ee631d4907d8d6fc3d39063a2a65c00a954c9bf5/diffsynth/diffusion/logger.py), and [`examples/flux2/model_training/train.py`](https://github.com/modelscope/DiffSynth-Studio/blob/ee631d4907d8d6fc3d39063a2a65c00a954c9bf5/examples/flux2/model_training/train.py).

The worker saves the base PNG before training. After training, it saves the trained image and releases model and Accelerator-held references before loading a fresh pipeline. It reloads the saved adapter and generates again with the same seed. The reloaded PNG is saved before the pixel check so a parity failure still returns the image. Reload parity requires mean absolute RGB pixel error at most 2 and no more than 1% of pixels above an 8-level channel error.

The result receipt records losses, gradient norms, adapter deltas, per-step and total training time, max allocated GPU memory, package versions, model revisions, and pixel differences. A successful result archive contains the adapter, base/trained/reloaded PNGs, and receipt; the returned archive limit is 128 MiB. A worker failure returns bounded progress and any adapter or PNGs already saved, with a sanitized failure receipt. Modal image/build/spawn/observation errors use a separate `modal-failure-receipt.json`, include the app/call IDs when known, and never claim a confirmed worker failure.

## Explicit remote launch

The command requires a previously created plan plus explicit account context. It rejects any `MODAL_TOKEN_*` or Modal OAuth credential environment override, selects profile `reflex-personal`, verifies workspace `rajath-61258`, and uses environment `main`:

```sh
uv run --extra control python -m experiments.modal_pilot \
  --launch \
  --dataset-dir artifacts/pilot-data-2026-10-08-r2 \
  --plan-dir artifacts/pilots/pilot-001 \
  --run-id pilot-001 \
  --profile reflex-personal \
  --workspace rajath-61258 \
  --environment main
```

Before importing Modal or starting image work, launch requires `.context/launch-clearance.json` for the current clean, signed HEAD. It binds the source commit, exact `.pilot-run.json` SHA256, payload SHA256, reviewed dataset manifest, run ID, Modal target, approved review/CI statuses, `$15` allowance, and a `checked_at` no more than 30 minutes old. The launcher makes only local Git checks; it does not query GitHub.

The launcher then reserves `modal-attempt.json` exclusively before Modal identity or image/build work. After spawn, it records the Modal app and call IDs before observing that same call. The reserved marker is never removed, and no automatic replacement call is started.

| Resource | Bound |
|---|---:|
| GPU | A100-80GB, one |
| CPU | 4 physical cores |
| Host memory | 64 GiB |
| Ephemeral disk | 30 GB |
| Startup timeout | 300 seconds |
| Function timeout | 2,700 seconds |
| Containers | One maximum, single-use, no warm or buffer containers |
| Retries / inputs | 0 retries, one input |

The app runs detached so its ID remains observable if the client disconnects. After the result or same-call recovery, root must explicitly stop the app and record the shutdown check. An unknown observation error requires inspecting the recorded call only; do not spawn another call.

At the refreshed rates, using all 300 seconds of startup and all 2,700 seconds of function time estimates about $2.70 of compute, excluding image build and storage. The $15 allocation is not a hard provider dollar cap. No Modal app or GPU call was made while implementing or running CPU checks.

## Evidence status

The pinned trainer files and loader/loop/logger helpers were inspected at the stated source commit. `runner.py` omits `batch_size` from `DataLoader`, iterates once per epoch, and calls the optimizer before the logger callback. `logger.py` increments its counter in `on_step_end` and saves SafeTensors from `save_model`; `train.py` defines `Flux2ImageTrainingModule`. Modal SDK 1.6.1 function/image/app-run signatures were checked locally. CPU tests verify eight/one selection, run-marker overwrite refusal, payload limits and integrity, split isolation, credential/profile checks, finite-value and optimizer-limit guards, clearance matching, attempt-marker ordering, failed artifact subsets, sealed-test rejection, and the default no-Modal path. The real GPU worker, model downloads, package installation inside Modal, A100 memory fit, training time, adapter reload parity, detached-app shutdown, and actual charges remain unverified until the reviewed one-shot run. Report those measurements from the returned receipt and attempt marker; do not infer quality improvement from this pilot.
