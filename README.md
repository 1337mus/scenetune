# Product Scene Editor

A reference-image editing experiment with **FLUX.2 Klein Base 4B** and a trained LoRA adapter.

The goal is to change a product image’s background while keeping the product intact.
The first study uses original synthetic packaging images. Real-photo performance is untested.

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

## Cost

The pilot allowance is **$15**. The larger $100–$150 experiment budget has not been used.

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
| R1 / R2 | Provider configuration failures before GPU inputs; evidence retained privately |
| [R3](docs/pilot-result-2026-10-08.md) | Eight training updates worked, but the saved-adapter replay failed |
| [Reload diagnostic](docs/reload-result-2026-10-09.md) | Separate-weight loading matched exactly; combined-weight loading failed the fixed limits |
| [R4](docs/pilot-result-2026-10-09-r4.md) | Fresh eight-update training and exact replay passed; no quality improvement established |
| Latest 144-update study | Better synthetic pixel scores; exact replay; all nine cases reported |

The latest launcher reported a shutdown-confirmation error after the worker passed.
The first independent closure check also failed on the CLI’s workspace-name format.
A corrected read-only check then confirmed the app was stopped with zero tasks. Both errors remain recorded.

## Next

Open Food Facts is a candidate photo source. Register API use with the [OFF form](https://docs.google.com/forms/d/e/1FAIpQLSdIE3D8qvjC_zRJw1W8OmuHhsWJ_NSckiiniAHlfaVwUZCziQ/viewform) before API collection, or use Raj-owned photos.
See the [real-photo evaluation plan](docs/real-photo-evaluation.md).
The launcher shutdown fix landed in `0862be8`; exact-source [CI run 38089275451 passed](https://github.com/1337mus/product-scene-editor/actions/runs/38089275451). Any future launch still needs fresh pins, CI, usage checks, and root review. No launch is approved.

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
