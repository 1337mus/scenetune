# Product Scene Editor

A small experiment in reference-image editing with **FLUX.2 Klein Base 4B**.

The goal: change a product photo's background while keeping its shape, color and label intact.

## Status

**Pilot R3 failed at saved-adapter reload after eight updates. No image-quality improvement is established.**

| Evidence | Result |
|---|---|
| Reload parity | Failed: average channel difference 7.82 /255 (limit 2); 25.8% of pixels differ by >8 (limit 1%) |
| Runtime | 129 seconds including downloads, generations, and training; no peak-memory or isolated training-time result |
| Review | [R3 report](docs/pilot-result-2026-10-08.md) · [contact sheet](docs/assets/pilot-r3/contact-sheet.png) |
| Next step | Diagnose saved-adapter reload mismatch before any further training; R3 was not retried |

This run checks the synthetic training pipeline only. It does not establish performance on real product photography.

## What we will compare

| Version | Purpose |
|---|---|
| Original model | Establish what the released model can already do |
| Model with a trained LoRA adapter | Measure what the training changes |

Both versions will use the same prompts, reference images and random seeds.
We will show failures alongside successful edits.

## Planned evidence

- Unseen product identities kept out of training.
- Product shape, label and requested edit checks.
- Saved-adapter reload and real parameter-update checks.
- Generation time, memory use and compute cost.

## Budget

The first pilot has a $15 compute allowance. The proposed full experiment allowance is $100–$150.
These are allocations, not measured bills or provider-enforced caps.

## Models and data

- [FLUX.2 Klein Base 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B).
- Original procedural packaging data. No employer data or paid LLM-generated captions.
- [Pinned DiffSynth training software and model components](docs/trainer-contract.md).

## Development

The laptop tools run on the CPU. Model weights and training stay on Modal.

```sh
uv sync --locked
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy src
```

Code: MIT. Model components keep their upstream licenses.
Original generated dataset images: CC0-1.0.
