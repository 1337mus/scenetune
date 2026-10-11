# SceneTune

Fine-tuning FLUX for product photography.

This personal project is separate from [Reflex](https://github.com/1337mus/reflex).

A reference-image editing experiment with **FLUX.2 Klein Base 4B** and a trained LoRA adapter.

The goal is to change product-image backgrounds while keeping products intact.
The first study used original synthetic packaging images. A bounded real-photo comparison has now generated three development cases; the original model was preferred in all three cases, while detailed quality checks remain pending.

## Synthetic training result

**Training reduced pixel error on all nine development examples.**
The original model and trained adapter used the same source images, prompts and random seed.

| Measure | Original model | Trained adapter | Change |
|---|---:|---:|---:|
| Average product-pixel error | 58.00 | 11.93 | 79.4% lower; better in 9/9 cases |
| Average background error against the target | 64.80 | 17.19 | 73.5% lower; better in 9/9 cases |
| Product pixels changed by more than 8 color levels | 99.21% | 91.90% | Lower in 8/9 cases; higher in 1/9 |
| Saved-adapter replay | — | Exact match in 9/9 cases | Zero pixel difference after reload |

Error is the average absolute RGB channel difference on a 0–255 scale. Lower is better.
Product error compares with the source; background error compares with the synthetic target.
The figures are equal-weight averages across nine examples from three products kept out of training.

The adapter learned our flat, synthetic target style. The original model produced more photographic scenes.
Lower pixel error here does not establish better photos. The development products were new, but the background styles appeared in training.

**The product is still not preserved exactly.** Many pixels change even after training.
These checks do not measure label readability, image realism or performance on real photos.
There is no claim of beating other image models.

[Read the full study and see all nine examples](docs/matched-study-result-2026-10-09.md).

![Three scenes for one development product: source, target, original model, trained adapter](docs/assets/matched-study-2026-10-09/pack-006.png)

## Real-photo R2 comparison

R2 ran seven 512×512 generations on three development products (3 base, 3 saved step-144 adapter, 1 strict-load repeat), with zero training; the repeat matched pixel-for-pixel. [Run details](docs/real-photo-comparison-result-2026-10-10.md).

Raj selected the original model in all three cases (Barilla A, Kinder Bueno B, Cruesli A). This records overall preference only; geometry, label correctness and background ratings remain pending. Synthetic pixel-error gains did not translate into adapter preference here; three cases do not establish a broad ranking.

## Scene-data checks

The checker is in the [signed source commit](https://github.com/1337mus/scenetune/commit/ed9e7f73f18d82874e2b7ae80c2a1722b0d22f98); [exact CI run 38110439968, attempt 1](https://github.com/1337mus/scenetune/actions/runs/38110439968) passed on it.

The checker compares recorded product shape, label and font, camera, lighting, materials and render settings. Only wall color may change. It checks the captured records, not whether every scene setting was captured.

| Check | Result |
| --- | --- |
| Project tests | 230 passed |
| New scene tests | 87 passed |
| Code checks | lint, format, types passed |
| R4 | Failed before images: Blender Vector type stopped capture. |
| R5 probe | Failed before snapshots or images: the Render Layers current-scene link was rejected. |
| R6 probe | Passed with 4 snapshots and 0 images. |
| R6 preview | Failed after 1 image: recorded scene fields no longer matched. |
| R7 probe | Passed with 4 snapshots and 0 images. |
| R7 preview | Passed: 4 CPU images in 55.0 s total, 2 threads, 96 samples. |

R4, R5, and R6 preview evidence was saved. Their processes stopped and mounts were detached. There were no GPU runs or training updates.

R7 passed the strict checks, but did not explain the earlier R6 failure or prove a general fix. We saw stable box and label positions, visible wall shades, grain, and low contrast. This preview does not approve image quality or data use.

Next is varied development-scene work. [Full results and contact sheet](docs/controlled-studio-preview-2026-10-10.md).

## What we built

- A paired dataset with source images, target images and product masks.
- Separate product identities for training, development and reserved tests.
- A bounded Modal training job with pinned code, data and model components.
- Adapter saving and strict loading into a fresh model.
- Matched image generation and an independent check of the weights, training records and scores.

The laptop runs CPU checks only. Model downloads, generation and training run on Modal.
No employer data or paid LLM-generated labels are used.

## Training run

| Item | Result |
|---|---|
| Training data | 18 pairs from 6 products |
| Development data | 9 pairs from 3 different products |
| Reserved tests | 9 pairs from 3 more products; still closed |
| Training | 8 passes over the data; 144 updates; one adapter |
| LoRA | Rank 32; 48,168,960 learned values across 100 modules |
| Training checks | All 144 losses finite; all 144 gradient records finite and nonzero |
| Saved weights | 200 tensors restored; all 200 changed during training |
| Image settings | 512×512; seed 7321; 16 steps; guidance 4; 27 total generations |
| Training time | 176.13 seconds |
| Worker time | 489.91 seconds through receipt creation; excludes archive packing |
| Peak memory allocated by Torch | 17.55 GiB; not total GPU memory |
| Compute | One A100-80GB; one call; zero retries |
| Shutdown | Direct check confirmed stopped, zero tasks, no active apps |

[Training design](docs/matched-study.md) · [Pinned trainer and model components](docs/trainer-contract.md)

## Cost: synthetic 144-update study

These figures describe the synthetic study, not the R2 run. The pilot allowance is **$15**. The larger $100–$150 experiment budget has not been used.

| Cost evidence | Amount |
|---|---:|
| Estimate at the full configured time limit | $2.66544 |
| Delayed gross workspace meter after the study | $5.35417914 |
| Net workspace bill after credits at that check | $0 |
| Final charge for this run | Unknown |

The workspace meter includes earlier work. It can lag and change.
The estimate excludes build, storage, transfer, idle grace and tax. It is not a provider spending cap.
Assistant-token costs are separate and are not included here.

## Earlier runs and failures

| Run | What it established |
|---|---|
| Pilot R1 / R2 | Provider configuration failures before GPU inputs; evidence retained privately |
| Real-photo R1 | Runtime import failed before model loading; no outputs or scores; failure evidence retained |
| Real-photo R2 | Seven generations, zero training; original model preferred 3/3 by one reviewer; geometry, label and background ratings pending |
| [R3](docs/pilot-result-2026-10-08.md) | Eight training updates worked, but the saved-adapter replay failed |
| [Reload diagnostic](docs/reload-result-2026-10-09.md) | Separate-weight loading matched exactly; combined-weight loading failed the fixed limits |
| [R4](docs/pilot-result-2026-10-09-r4.md) | Fresh eight-update training and exact replay passed; no quality improvement established |
| Latest 144-update study | Better synthetic pixel scores; exact replay; all nine cases reported |

In the synthetic 144-update study, the latest launcher reported a shutdown-confirmation error after the worker passed.
The first independent closure check for that study also failed on the CLI’s workspace-name format.
A corrected read-only check then confirmed the app was stopped with zero tasks. Both errors remain recorded.

## Next

1. **Prepare reviewed training targets.** The photographic-background composite preserved 43,704 product pixels, but looked oversized and pasted into the room; it stays out of training. Controlled-studio R2/R3 provide one measurable 3D scene recipe. R3 remains visibly grainy, and training remains held.

   [Preview results](docs/controlled-studio-preview-2026-10-10.md) · [Separate 3D data proposal](docs/controlled-studio-data-plan.md) · [Photographic-target findings](docs/photographic-target-plan.md).
2. **Check the data before training.** Use cleared sources, realistic placement and shadows, and separate product and background families in each data pool. Keep the current reserved tests closed.
3. **Run a small matched study once the data passes review.** Compare the original model and a fresh adapter on the same inputs. Check saved-adapter reload, product preservation and photo quality separately.

No further GPU run is cleared. Future compute needs a frozen plan, current prices and usage, pinned inputs, exact-source CI, and root review. Barilla, Kinder Bueno and Quaker Cruesli remain development-only. Their detailed output-label and geometry checks are still pending.

## Development

```sh
uv sync --locked
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy src
```

Code: MIT. Model components keep their upstream licenses.
Original generated dataset images: CC0-1.0.

Model: [FLUX.2 Klein Base 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B).
