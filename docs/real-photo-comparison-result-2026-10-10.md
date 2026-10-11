# Real-photo comparison R2

## What ran

R2 compared three development-only product cases using seven 512×512 generations: three base, three with the saved step-144 adapter, and one fresh strict-load repeat. Training updates: zero. Two strict loads each restored 200 tensors across 100 modules; the independent repeat matched at every pixel.

| Measure | Result |
|---|---|
| Worker time | 203.30 seconds, excluding app setup and shutdown |
| Peak allocation | 16.55 GiB from Torch; not total GPU memory |
| Shutdown | App stopped, zero tasks; no other active apps |
| Exact CI | [38097654443](https://github.com/1337mus/product-scene-editor/actions/runs/38097654443): 143 CPU tests, lint, format and types passed |
| Source | `b39fec4267561472c389c98de5f25bdb34da49ad` |

## What we can say

Assistant visual observations, not human ratings: both approaches produced kitchen scenes with recognizable products. Some package position or size changes appeared; adapter backgrounds looked smoother and less textured than base in these three cases. Source review covers seven word boxes across the cases and the first two package outlines. Cruesli is a cropped front; exclude whole-package shape and masked-pixel scoring.

Output geometry review, label transcripts and background ratings remain pending. These are development cases, not a quality benchmark. No pixel-quality score, label percentage, combined score or broad superiority claim is made.

## Cost

At 2026-10-11 00:18:41 UTC, the delayed shared workspace meter showed $5.46 gross and $0 net after credits. It is not a final invoice for R2, and assistant-token spending is separate. The prelaunch estimate was $1.07, excluding extras. The original $15 allowance is unchanged.

## Next

Review output geometry and human ratings, then choose a data direction before further training. Keep these three development products out of training and final testing.
