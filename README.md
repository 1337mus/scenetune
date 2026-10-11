# Product Scene Editor

A reference-image editing experiment with **FLUX.2 Klein Base 4B** and a trained LoRA adapter.

The goal is to change product-image backgrounds while keeping products intact.
The first study used original synthetic packaging images. A bounded real-photo comparison has now generated three development cases; the original model was preferred in all three cases, while detailed quality checks remain pending.

## Latest result

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

- The six OFF candidates and all prepared assets stay private and out of Git. The frozen development set is Barilla, Kinder Bueno, and Quaker Cruesli; keep all three out of training and the final test set. Improve realistic training targets before another training run.
- Source review is complete for seven word boxes across the three cases and the first two package outlines. Cruesli is a cropped front: exclude whole-package shape and masked-pixel scoring. Output geometry, labels and background ratings remain pending.
- Completed R2: seven 512×512 generations (3 base, 3 saved step-144 adapter, 1 fresh strict-load repeat), zero training, seed 7321, 16 steps, guidance 4.
- The first remote attempt (R1) failed during runtime import before model loading and produced no images or scores. The separately authorized R2 comparison completed; see the [R2 result](docs/real-photo-comparison-result-2026-10-10.md). The [R1 attempt record](docs/real-photo-comparison-attempt-2026-10-10.md) and [source check](docs/real-photo-source-check-2026-10-10.md) preserve its evidence; the [evaluation plan](docs/real-photo-evaluation.md) describes the frozen development set. No further launch is cleared.

The shared-runner shutdown fix from the synthetic study landed in `0862be8`; exact-source [CI run 38089275451 passed](https://github.com/1337mus/product-scene-editor/actions/runs/38089275451). The real-photo shutdown fix landed in `b39fec4`; exact-source [CI run 38097654443 passed](https://github.com/1337mus/product-scene-editor/actions/runs/38097654443). Any future launch still needs fresh pins, CI, usage checks, and root review.

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
