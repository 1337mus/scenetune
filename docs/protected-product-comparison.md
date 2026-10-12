# Keeping the product while changing its scene

## Why test a different method?

Our earlier edits changed some product colors, shapes and small text.
More training did not give a clear gain on the three real development photos.

This comparison tests a different image pipeline:

```mermaid
flowchart LR
    A[Product photo] --> B[Checked product cutout]
    C[Fixed scene prompt] --> D[FLUX generates an empty scene]
    B --> E[Place original product pixels into the scene]
    D --> E
    E --> F[Check product pixels, edges and scene quality]
```

The product stays in place. FLUX supplies the background.
This can preserve the product without asking FLUX to redraw its label.
A CPU check on two old development photos found a visible duplicate package
when we pasted the source over a full-image edit. Both cases kept product pixels
but failed visually. We will generate an empty scene before compositing.
It does not guarantee a clean cutout, natural shadows or good lighting.

## Current stage

R1, the first comparison attempt, stopped on its first generation because its remote image had no C compiler. It produced no images or scores; its failure report remains historical and its evidence is preserved.

R2 ran the frozen 24-case comparison after a separate CPU-only remote image-build gate verified C, C++ and make. The worker completed all 52 generation calls, 78 PNGs, two successful model loads, four exact fresh-load replays and 26 composites, then stopped with zero tasks. No training ran. The 24 cases were 6 development and 18 reserved, held out by whole brand and product family.

The original local receipt validator rejected the successful worker receipt because its exact-field check omitted the pinned Modal SDK version. All 17 runtime pins matched. A narrow offline validator correction passed a test that reproduced the rejection and then passed with the fix, 26 affected-module CPU tests and 27 independent checker tests. The official and independent audits passed the same unchanged R2 result bundle. The original R2 marker, pending result, source and plan remain preserved; there was no recovery run, relaunch or new GPU generation.

Review found 0/6 development and 0/18 reserved cases usable in either method. Protected pixels were exact in all 24 composites, but all had poor ground contact and 14 added an unrequested stone-like object. Every full-image edit changed product size, shape or color, so the frozen gate failed. The reserved cases have now been reviewed; future confirmation needs new untouched products.

[Full R2 result, audit limits and all case outcomes](protected-product-comparison-result-2026-10-12.md) · [R1 failure report](protected-product-comparison-failure-2026-10-12.md).

| Item | Frozen plan | R1 attempt 1 |
|---|---|---|
| Input set | 24 accepted ABO catalog cutouts | Same frozen payload; all 24 source photos and masks reviewed |
| Pool split | 6 development / 18 reserved | Unchanged; reserved generated outputs unopened |
| Cutout review | 18 original, 3 repaired, 3 replacements | All 24 accepted before launch |
| Methods | Full-image edit; empty-scene generation plus original product | First full-image edit failed before output |
| Model calls | 48 main calls plus 4 fresh-load replays (52 total) | 1 attempted; 0 succeeded |
| PNG artifacts | 78 planned across outputs, composites and replays | 0 produced |
| Training updates | 0 | 0 |
| Model and compute | FLUX.2 Klein Base 4B; one A100 80 GB, 4 CPU cores, 64 GiB RAM | Base pipeline loaded once on CUDA in 21.88 seconds |
| Worker and shutdown | 300-second startup; 1,800-second execution; one call; zero retries | 115.77 seconds; one spawn; stopped with zero tasks |
| Allowance | $8, including the $1 mask reserve | Not a per-run charge or a spending cap |

R1's signed [source commit ab7a429](https://github.com/1337mus/scenetune/commit/ab7a429023893dee1e1d0b3bde19fe31c0fac7b3) passed exact [CI run 38190964224, attempt 1](https://github.com/1337mus/scenetune/actions/runs/38190964224). The separate R2 result and audits are described above. The earlier $15 pilot remains closed.

## How we prepare the photos

1. Download individual photos with their official product records.
2. Keep the download URL, license, attribution and file hashes.
3. Reject logos, transparent products, clipped objects and unclear boundaries.
4. Resize and pad each accepted photo to 512 × 512. Do not crop the product.
5. Create a draft product cutout. Review its outline against the source.
6. Accept a mask only after a recorded visual check.
7. Group whole brands and product families into separate pools.

ABO photos do not come with ready product masks.
A simple cutout works only on selected, clean backgrounds.
A failed mask must stay rejected; we will not count it as a model failure.

The first six-photo screen found two possible cutout candidates.
Four failed: transparent glass, faint edges, a clipped light fixture and a logo.
The sandal candidate also has a reflection that needs a separate check.
Both possible candidates failed the first actual mask check.
The planter produced separate large regions. The sandal retained its reflection.
At that stage, no mask had passed. We kept those failures and did not relax the rules.

We then checked 30 more catalog photos. Six were unsuitable sources.
The simple mask method returned 20 drafts from the 24 remaining photos.
Four failed its checks. Visual review also found missing package regions and
retained background patches. A returned draft does not mean a good cutout.
None of these simple-method masks has been accepted.

The completed mask prerequisite used pretrained [BiRefNet Lite](https://huggingface.co/ZhengPeng7/BiRefNet_lite)
on the same 24 processed photos. It made 24 first-output calls and four repeat
calls, with two strict model loads. Training updates: zero.

| Mask prerequisite | Fixed limit |
| --- | --- |
| Compute | One A10, 2 CPU cores, 8 GiB RAM |
| Time | 180 seconds startup; 600 seconds execution; zero retries |
| Cost | Estimated compute $0.27; total step reserve $1 within the $8 allowance |
| Estimate excludes | Image builds, storage, transfer and shutdown lag |

The estimate was not a provider dollar cap. The mask check used no new photos or
training. The source, weights, input hashes and runtime versions were fixed before
that launch. Root reviewed the resulting cutouts, checked that the mask app had
stopped with zero tasks, and later froze the accepted set and 6/18 split for the
comparison. The mask standard was not lowered to fill the sample.

### Repaired mask check: initial review

The new run completed 24 first predictions and four exact repeat predictions.
Both strict model loads passed. The worker took 16.9 seconds on one A10.
The independent checker passed, and the app is stopped with zero tasks.

The initial assistant review of all 24 unscaled sources and cutouts accepted 18
outlines. Six needed correction: four lost package regions, one lost part of a
USB hub, and one retained a drop shadow. Those drafts were preserved; the final
accepted set is listed below.

[Full cutout result and limits](product-cutout-check-2026-10-12.md).

### Final input checks

| Source of accepted cutouts | Products |
| --- | ---: |
| Original GPU drafts | 18 |
| Original drafts with checked CPU repairs | 3 |
| Replacement catalog photos with checked CPU masks | 3 |
| **Total** | **24** |

We kept the rejected drafts and excluded photos. One replacement needed a repair:
we removed white background trapped between a suitcase’s handle rails.
The source photo stayed unchanged.

Official product and image records check out. Whole brands and product families
were kept in separate pools. The final split and payload were frozen before
launch; R1 produced no images, while R2 later completed the comparison described
above. These are selected, clean catalog photos. Mask acceptance is an assistant
judgment, not a ground-truth segmentation score.


### First mask check: stopped before inference

The first run failed a model-file count check. It made zero predictions.
The app is stopped and has zero running tasks. We keep that failed run unchanged.

Hugging Face's summary lists 44,362,660 stored values. The pinned file header
lists 44,362,703. An independent header check confirmed the 43-value difference.
The float values match; the difference is in integer values.
This is a loading-check defect, not a mask-quality result.

The correction used the pinned file count and kept the full-file hash check.
Failed loads retain their observed details. A separately reviewed run used the
same photos, model and settings; the failed run was not restarted.

All earlier SceneTune product identities and development brands stay excluded.
Rendered ABO views do not count as real catalog photos.
Source photos, masks, private reviews and receipts stay outside Git initially.

## Keep the comparison fixed

Both methods use the same processed source photo and case seed.
Seeds start at 7,321 and increase in sorted case-ID order.
Both use 16 steps and guidance scale 4.
Prompts and settings stay fixed once the plan is frozen.

The empty-scene method received no reference product image. It generated an
empty background before the original product was composited. The R2 output still
contained unrequested stone-like objects and poor product-to-ground contact;
compositing alone did not make the scenes usable.

The planned fresh model load and four replay checks ran in R2 and matched exactly.
The root reviewed all 24 case sheets. The frozen reserved gate failed, and the
reserved outputs have now been reviewed; they are not a fresh held-out set. The [R2 result report](protected-product-comparison-result-2026-10-12.md)
records every case and the limits of these assistant judgments. Any future
confirmatory study requires new untouched product identities, a separate
root-reviewed plan and compute clearance. No GPU launch happens automatically.

## What the numbers can tell us

| Check | What it establishes |
| --- | --- |
| Changed protected pixels | Whether the processed source product pixels stayed exact |
| Source and mask hashes | Which inputs the comparison used |
| Replay pixel agreement | Whether fresh loading reproduced the fixed cases |
| Visual product review | Whether the mask kept the intended product |
| Visual scene review | Whether edges, shadows, lighting and placement look usable |

The protected-pixel check is a property of compositing.
It is not evidence that the model learned better labels or shapes.
Resizing changes the original high-resolution photo before either method runs.
Our exact-pixel claim applies to the reviewed 512 × 512 source.

A usable result must keep the whole intended product, follow the scene request,
and have no obvious cutout gap, duplicate object or floating placement.
The R2 report counts each failure type separately.

The prespecified exploratory gate was at least 14 of 18 usable reserved-test
composites and at least four more usable cases than the ordinary edit, with all
18 passing protected-pixel integrity. R2 preserved protected pixels but had zero
usable reserved composites and zero usable full-image edits, so the observed gain
was zero and the gate failed. These small counts do not establish broad
superiority or performance on difficult, unscreened photos.

We will report all cases, failures and ties.
We will separate automated integrity checks from root visual judgments.
Root visual judgments are not a new human preference study.
We will not call pixel error a text-reading or realism score.

## Decide what happens next

R2 did not produce a usable method or authorize extra epochs or new training.
Address the observed failures: make empty backgrounds genuinely empty, improve
product-to-ground contact and shadows, and keep product interiors unchanged. Any
future study needs new untouched product identities, a separate root-reviewed
plan and compute clearance. No GPU work launches automatically.

## Data source

[Amazon Berkeley Objects](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)
provides real catalog photos under CC BY 4.0.
The [official README](https://amazon-berkeley-objects.s3.amazonaws.com/README.md)
sets the attribution requirements.

Credit: Amazon.com; Matthieu Guillaumin, Thomas Dideriksen, Kenan Deng,
Himanshu Arora, Jasmine Collins and Jitendra Malik.
This work is not endorsed by the dataset authors or product brands.
