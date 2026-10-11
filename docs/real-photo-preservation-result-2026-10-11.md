# Real-photo preservation comparison

Development-only pilot · 11 October 2026

## Scope and execution

The approved comparison tested the original FLUX.2 Klein Base 4B model, the prior 3D-scene adapter, and the fresh preservation adapter on the same three development photos: Barilla, Kinder Bueno, and Quaker Cruesli.

It ran 15 generations: three first-pass outputs per model and six fresh-load replay outputs. Training updates: zero.

| Run check | Result |
|---|---|
| Source and CI | [Commit `b81e93d`](https://github.com/1337mus/scenetune/commit/b81e93d069f42cbb6006db20c776ab1dc5e51fbe); [CI run 38179029569, attempt 1](https://github.com/1337mus/scenetune/actions/runs/38179029569) passed |
| Generation | 512×512; 16 steps; seed 7321; guidance 4 |
| Adapter loading | Four strict loads; each restored 200 tensors across 100 modules |
| Replay | Six of six independently decoded replay pairs matched exactly |
| Worker | 296.788 seconds (about 4.95 minutes); 16.549 GiB peak Torch allocation, not total GPU memory |

An initial audit setup error was preserved. The corrected audit passed without rerunning the GPU job. A read-only closure check at 22:18:21Z confirmed the app was stopped, with zero tasks and no other active apps. The CLI stop command returned 1; its first closure observation still showed one task, before the later zero-task check.

## Visual observations

These are assistant visual observations from review of all nine first-pass outputs against the three sources. All nine show a counter and a blurred kitchen background. Every output changes product position and/or size. The two adapters look close to one another, with no clear overall quality improvement. No new Raj preference was recorded.

| Photo | Observations |
|---|---|
| Barilla | The blue carton remains recognizable and “Barilla” is readable in all three outputs. “SPAGHETTI” is readable in the original output; both adapters visibly distort its letters, so exact word correctness is unconfirmed. |
| Kinder Bueno | “kinder” and “bueno” remain readable in all three. Wrapper size, angle, or location changes; small badges and copy change as well. |
| Quaker Cruesli | “QUAKER” and “Cruesli” are readable in all three. The French subtitle looks readable in the original; the adapters visibly distort small letters and spacing. Exact accent and letter correctness is unconfirmed. |

Seven source-word regions had been confirmed before this comparison. This review assigns no numeric text score and does not transcribe uncertain output strings. The earlier 3/3 preference for the original model applies only to the first, flat-background adapter comparison; it is not a preference result for these two adapters.

## Scoring limits

- Position or size changed in both full-package cases, so fixed-source-mask product-pixel scores are inapplicable.
- Quaker Cruesli is a cropped package front; exclude whole-package shape and masked-pixel scoring entirely.
- No matched real-photo target exists, so there is no background pixel-error score. Background and overall appearance remain visual observations only.
- These observations do not establish a shape score, text-quality percentage, or broad model winner.

## Data, cost, and next steps

The three photos and product families stayed outside training and final tests. Redistribution rights remain unresolved. Photos, generated outputs, weights, and private annotations remain outside Git.

The delayed shared-workspace meter showed $7.06040431 gross and $0 net after credits, versus $6.64223795 before launch (a $0.41816636 increase). This delayed meter is not a per-run invoice. The $1.86736667 estimate excludes extras and is not a provider cap; the $15 pilot allowance remains the limit. Assistant-token costs are separate.

Close the approved pilot here. Future improvement work needs broader realistic paired targets and untouched test photos; these results do not justify more epochs or authorize another training launch.
