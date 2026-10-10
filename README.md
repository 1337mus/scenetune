# Product Scene Editor

A small experiment in reference-image editing with **FLUX.2 Klein Base 4B**.

The goal: change a product photo's background while keeping its shape, color and label intact.

## Status

**R3 completed eight training updates, then failed at adapter reload. A follow-up check found a working reload method. The complete training-and-reload run still needs verification.**

| Evidence | Result |
|---|---|
| R3 training | Failed after eight updates at adapter reload; see the [original R3 report](docs/pilot-result-2026-10-08.md) |
| Reload diagnostic | Separate-weight loads matched exactly; combined weights still fail the same parity limits, with zero training updates |
| Quality | No improvement or real-photo performance is established |
| Reports | [Reload diagnostic](docs/reload-result-2026-10-09.md) · [R3 contact sheet](docs/assets/pilot-r3/contact-sheet.png) |
| Next step | Use the working separate-weight reload in a fresh eight-update training pilot |

The diagnostic used the synthetic pipeline only. It does not validate image quality.

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
