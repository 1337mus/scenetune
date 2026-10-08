# Trainer contract: FLUX.2 Klein Base 4B paired edits

Status: **trainer and component pair verified; the eight-update boundary is supported by the pinned loader/loop/logger source and enforced in the worker. Runtime counts remain unobserved until the cleared pilot runs.** DiffSynth-Studio is the route. Musubi remains an alternative only after someone verifies a compatible single-file Klein AE conversion; its current documented AE format does not directly match the pinned Klein VAE.

## Locked sources and component licenses

Pin DiffSynth-Studio to commit `ee631d4907d8d6fc3d39063a2a65c00a954c9bf5`.

The trainer's base4B LoRA and inference examples intentionally combine these two Apache-2.0 Hugging Face snapshots:

| Role | Repository | Revision | Files used |
|---|---|---|---|
| Trainable transformer | `black-forest-labs/FLUX.2-klein-base-4B` | `a3b4f4849157f664bdbc776fd7453c2783562f4d` | `transformer/*.safetensors` |
| Text encoder | `black-forest-labs/FLUX.2-klein-4B` | `e7b7dc27f91deacad38e78976d1f2b499d76a294` | `text_encoder/*.safetensors` |
| VAE | `black-forest-labs/FLUX.2-klein-4B` | `e7b7dc27f91deacad38e78976d1f2b499d76a294` | `vae/diffusion_pytorch_model.safetensors` |
| Tokenizer | `black-forest-labs/FLUX.2-klein-4B` | `e7b7dc27f91deacad38e78976d1f2b499d76a294` | `tokenizer/**` |

The full Klein 4B revision is recorded in the launch worker's [`MODEL_COMPONENTS`](../src/product_scene_editor/remote.py) pins. The pinned model repos carry Apache-2.0 licenses; the base model README explicitly permits commercial use. Use the Klein VAE and TE that the base4B trainer examples select; do not substitute FLUX.2-dev components. Materialize these snapshots at the pinned revisions and pass local component paths at runtime so no unrevisioned model-ID resolution can occur.

## Paired-edit data contract

The base4B LoRA example's edit recipe trains the base4B transformer with `--data_file_keys "image,edit_image"` and `--extra_inputs "edit_image"`. Export each record with `image` pointing to the changed target and `edit_image` pointing to its starting/reference image. Keep the synthetic images under stable names such as `targets/<product>__<scene>.png` and `references/<product>__<scene>.png`; metadata rows pair those paths explicitly. Captions/prompts stay on the same metadata row. The model inference API receives the reference as `edit_image=[PIL.Image]`. Masks are not part of this documented training path.

Use only rows from the training identity split. Development and sealed test identities stay out of the trainer dataset and prompt cache. The commented recipe uses Qwen Edit data as an example data source, but its component mapping uses the FLUX.2 Klein base4B transformer; data source and model are separate.

## Minimal pilot candidate

The pinned base4B LoRA example supplies the model component mapping, LoRA target modules, `--lora_base_model dit`, rank 32, gradient checkpointing, and learning rate `1e-4`. For an eight-example smoke pass at the generator's 512 px size, use an eight-row training-only metadata file, `--max_pixels 262144`, `--dataset_repeat 1`, `--num_epochs 1`, and `--save_steps 8`. In the pinned `diffsynth/diffusion/runner.py`, `launch_training_task` creates `DataLoader(dataset, shuffle=True, collate_fn=..., num_workers=...)` without `batch_size`, so PyTorch's default batch size is one. The loop runs one pass over the eight rows and calls `optimizer.step()` once per batch; `ModelLogger.on_step_end` follows that call, increments its count, and writes `step-8.safetensors` at `save_steps=8`. This source evidence predicts eight updates for the validated eight-row dataset; it is not an observed GPU result. The worker independently counts successful optimizer calls, refuses call nine before updating parameters, checks the logger count, and requires the saved step-8 adapter. The detailed source paths and remaining runtime evidence boundary are recorded in `.context/runtime-worker-evidence.md`.

The validation example reloads the saved adapter with `pipe.load_lora(pipe.dit, adapter_path)`. The native base4B inference call accepts `pipe(prompt, edit_image=[reference], seed=0, num_inference_steps=50, cfg_scale=4)` and saves the returned image. The one-epoch worker loads `step-8.safetensors`, matching the pinned logger's save rule; the remote write itself remains unobserved until the cleared run.

## Runtime dependencies and limits

DiffSynth's pinned `pyproject.toml` requires Python >=3.10.1 and lists Torch >=2.0, TorchVision, Transformers, ImageIO/FFmpeg, SafeTensors, Einops, ModelScope, ftfy, pandas, Accelerate and PEFT; its training extra adds DeepSpeed. Most dependencies are ranges, not fully pinned by upstream. The current project remains CPU-only, so install GPU dependencies only in the remote worker. Upstream's base model card says about 13 GB VRAM for inference; that is not a training requirement. No verified training VRAM, step-time or dollar estimate is available, so the $15 pilot remains an allocation rather than a forecast.

## Sources

- [DiffSynth commit](https://github.com/modelscope/DiffSynth-Studio/commit/ee631d4907d8d6fc3d39063a2a65c00a954c9bf5), [base4B LoRA example](https://github.com/modelscope/DiffSynth-Studio/blob/ee631d4907d8d6fc3d39063a2a65c00a954c9bf5/examples/flux2/model_training/lora/FLUX.2-klein-base-4B.sh), [base4B validation](https://github.com/modelscope/DiffSynth-Studio/blob/ee631d4907d8d6fc3d39063a2a65c00a954c9bf5/examples/flux2/model_training/validate_lora/FLUX.2-klein-base-4B.py), [base4B edit inference](https://github.com/modelscope/DiffSynth-Studio/blob/ee631d4907d8d6fc3d39063a2a65c00a954c9bf5/examples/flux2/model_inference/FLUX.2-klein-base-4B.py), [shared trainer](https://github.com/modelscope/DiffSynth-Studio/blob/ee631d4907d8d6fc3d39063a2a65c00a954c9bf5/examples/flux2/model_training/train.py), [FLUX2 docs](https://github.com/modelscope/DiffSynth-Studio/blob/ee631d4907d8d6fc3d39063a2a65c00a954c9bf5/docs/en/Model_Details/FLUX2.md), [project dependencies](https://github.com/modelscope/DiffSynth-Studio/blob/ee631d4907d8d6fc3d39063a2a65c00a954c9bf5/pyproject.toml).
- [Pinned base4B model card and license](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B/tree/a3b4f4849157f664bdbc776fd7453c2783562f4d); [pinned Klein 4B model card and license](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/tree/e7b7dc27f91deacad38e78976d1f2b499d76a294).
- [Musubi FLUX.2 docs](https://github.com/kohya-ss/musubi-tuner/blob/f8a1b03794a49239a3539015075f5123d6c07d66/docs/flux_2.md), [Musubi AE loader](https://github.com/kohya-ss/musubi-tuner/blob/f8a1b03794a49239a3539015075f5123d6c07d66/src/musubi_tuner/flux_2/flux2_utils.py).
