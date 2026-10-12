# Product cutout check

12 October 2026 · 24 selected Amazon Berkeley Objects photos

## Result

**18 cutouts passed visual review. Six need work.**
The planned background comparison remains pending.

This step used pretrained [BiRefNet Lite](https://huggingface.co/ZhengPeng7/BiRefNet_lite)
to find each product's outline. It did not train FLUX or a new adapter.

| Check | Result |
| --- | ---: |
| Source photos | 24 |
| First predictions | 24 |
| Repeat predictions after a fresh model load | 4 |
| Exact repeat matches | 4 / 4 |
| Successful model loads | 2 / 2 |
| Cutouts that passed assistant visual review | 18 / 24 |
| Cutouts that need correction | 6 / 24 |
| Training updates | 0 |
| Worker time | 16.9 seconds |
| Peak memory allocated by Torch | 1.65 GiB |

Memory is the allocation measured by Torch, not total GPU memory.
One A10 GPU ran the check. An independent account check confirmed that the app
is stopped, with zero tasks and no other active apps.

## What needs work

| Problem | Photos |
| --- | ---: |
| White package regions or an interior region were removed | 4 |
| Part of a solid USB-hub body was removed | 1 |
| Drop shadows remained around a USB drive and its cap | 1 |

These are cutout failures. They are not results from the planned scene comparison.
The original photos, draft masks and private review panels stay outside Git.

## The first run failed

The first attempt stopped before inference. It produced no masks.
Our load check used Hugging Face's aggregate count of 44,362,660 stored values.
The pinned file header contains 44,362,703 values, including 43 scalar integer buffers.
An independent header check confirmed the difference without downloading weights locally.

We corrected the count and kept the full-file hash and strict loading checks.
The new run used the same 24 photo bytes, model revision and image settings.
Both loads then restored 586 tensors and 44,362,703 stored values without missing
or unexpected keys. Stored values include buffers; this is not a trainable-parameter count.

The failed attempt remains preserved. Failed loads now also keep their observed
counts and error details.

## What this establishes

- The pinned model loads and produces masks on the selected photos.
- Four predictions repeat exactly after a fresh model load.
- Assistant review found 18 usable outlines at 512 × 512 pixels.

There are no hand-labeled reference masks or numeric segmentation-accuracy scores.
The photos were screened first. This is not a test on an unfiltered catalog.
Visual review is an assistant judgment, not a human rating study.
There is no new label-reading score, scene-quality score or training gain.

## Cost and verification

The new run had an estimated compute cost of **$0.27**, with a **$1 total mask-step
reserve** inside the approved $8 comparison allowance. Builds, storage, transfer
and shutdown lag are excluded. These are planning limits, not a provider dollar cap.

The delayed personal-workspace meter showed **$6.95 gross and $0 net** at
00:14:22 UTC. It had not changed from the prelaunch reading. The final charge for
this run is not yet known. Assistant-token costs are separate.

[Signed source `0eb957b`](https://github.com/1337mus/scenetune/commit/0eb957ba8911fe069c54cad2bcc540a3aecf0463)
passed [CI run 38186722626, attempt 1](https://github.com/1337mus/scenetune/actions/runs/38186722626):
534 CPU tests, lint, formatting and strict types. An independent CPU check also covered 40 affected tests
and 19 fixtures for the result checker. The actual run passed that checker.

A local proof helper used the wrong Python environment after writing clearance.
The launcher independently checked source, plan, account and clearance before
its single launch. The helper error and later recheck are preserved privately.
No second launch occurred.

## Next step

Prepare separate CPU corrections for the six rejected masks. Review each correction.
Keep every original mask and failure. Accept the full input set before freezing
the two-method scene comparison. More FLUX training depends on that comparison.

[Comparison design](protected-product-comparison.md) ·
[Official ABO source and attribution](https://amazon-berkeley-objects.s3.amazonaws.com/README.md)
