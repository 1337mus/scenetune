# Product Scene Editor

A small experiment in reference-image editing with **FLUX.2 Klein Base 4B**.

The goal: change a product photo's background while keeping its shape, color and label intact.

## Status

**R4 passed a fresh eight-update synthetic training run and exact saved-adapter reload check.** It repeats R3’s eight-update schedule from the base model; it is not eight additional updates to R3. Pipeline behavior is verified, while image quality remains unvalidated.

| Evidence | Result |
|---|---|
| R3 reload | Original R3 training stopped at adapter reload. A separate-weight diagnostic matched exactly; combined weights still fail the frozen parity limits. See the [R3 report](docs/pilot-result-2026-10-08.md) and [reload diagnostic](docs/reload-result-2026-10-09.md). |
| R4 training and reload | 8 optimizer updates; 200 saved tensors across 100 modules restored exactly. Trained and reloaded outputs match pixel-for-pixel. See the [R4 report](docs/pilot-result-2026-10-09-r4.md) and [contact sheet](docs/assets/pilot-r4/contact-sheet.png). |
| Quality | Base and trained outputs differ (average color change 8.60 on a 0–255 scale; 27.64% of pixels differ by more than 8 on that scale). That shows output change only; no quality gain or real-photo performance is established. |
| Runtime and cost | App stopped with 0 tasks and 0 active apps. Actual R4 charge is unknown; the workspace meter is delayed and revisable. |
| Next | Train one new adapter on the larger 18-pair training split, then compare the base and adapter on all 9 development pairs with source, prompt, and seed matched. No new run has been launched. |

R3 and R4 used the synthetic pipeline only. Neither run establishes image quality on real product photography.

## What we will compare

| Version | Purpose |
|---|---|
| Original model | Establish what the released model can already do |
| Model with a trained LoRA adapter | Measure what the training changes |

Both versions will use the same prompts, reference images and random seeds.
We will show failures alongside successful edits.

## Next bounded study

- Train one new adapter on the larger 18-pair training split.
- Compare the base model and this adapter on all 9 development pairs with the same source, prompt, and seed for each pair.
- Predefine checks for product shape, label readability, and the requested background; report all cases and failures.
- The brief remains under review. No new run has been launched.

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
