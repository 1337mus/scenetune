# 3D-scene preservation study R2

The 2026-10-11 follow-up trained a fresh LoRA from the frozen base to preserve products while changing backgrounds. It compared that adapter with the earlier 3D-scene adapter across nine development edits from three synthetic products. The edits reuse development data; they are not an independent test set.

The export used six training products. It had 18 rows: 12 background-changing pairs and six unchanged controls. The nine evaluated edits reused three development products.

| Training and verification | Result |
|---|---|
| Signed training source | [Commit e82ae3c](https://github.com/1337mus/scenetune/commit/e82ae3c765342e02025b1f0d695ea8f44df4b34c) |
| Exact CI | [Run 38176002659](https://github.com/1337mus/scenetune/actions/runs/38176002659) |
| Training | Eight epochs, 144 updates, one A100 80 GB GPU; 187.674 seconds |
| Gradient checks | 144 finite, nonzero gradient checks; all 200 adapter tensors changed |
| Adapter and replay | 48,168,960 parameters; 200 tensors across 100 modules validated; 9/9 reloads pixel-exact |
| Worker | 595.949 seconds; 17.55 GiB peak allocation |
| Closure | Adapter audit passed; app stopped with zero active tasks |

## Goals and results

| Development goal | Threshold | Observed | Result |
|---|---:|---:|---|
| Lower mean product-source pixel error than the prior adapter | Below 23.36 | 20.93, 10.39% lower | Met |
| Improve product-source error in at least six of nine edits | At least 6/9 | 4/9 | Not met |
| Keep mean background-target error within 20% of the prior adapter | At most 17.69 | 17.71, 20.17% higher | Not met |

| Mean absolute RGB error (0–255) | Original model | Prior 3D adapter | New adapter |
|---|---:|---:|---:|
| Product pixels vs. source | 32.64 | 23.36 | 20.93 |
| Background pixels vs. target | 87.69 | 14.74 | 17.71 |

The new adapter reduced the average product-source pixel error, but improved that measure on only four edits. Its mean background-target error rose just beyond the 20% limit. The study did not meet its combined goals.

| Development edit | Product-source error, prior → new | Background-target error, prior → new |
|---|---:|---:|
| Carton 201 clay | 20.5 → 21.5 | 9.9 → 12.2 |
| Carton 201 cream | 34.8 → 23.3 | 20.0 → 22.8 |
| Carton 201 sage | 16.9 → 18.6 | 12.2 → 14.3 |
| Carton 202 clay | 12.2 → 18.1 | 11.4 → 13.7 |
| Carton 202 cream | 32.1 → 22.0 | 21.4 → 23.8 |
| Carton 202 sage | 25.7 → 25.9 | 12.8 → 15.4 |
| Carton 203 clay | 16.6 → 13.3 | 10.4 → 12.9 |
| Carton 203 cream | 34.4 → 27.2 | 21.3 → 28.7 |
| Carton 203 sage | 17.1 → 18.4 | 13.2 → 15.6 |

All three cream edits and one clay edit improved product-source error; no sage edit improved.

Each pixel error is mean absolute RGB-channel difference on a 0–255 scale. Lower values mean greater pixel similarity for that comparison; they do not establish better image quality, readable text, or exact product preservation.

The background limit was 17.68644816; the observed mean was 17.71199512, a 20.17% rise from the prior adapter. The rounded table values do not hide that the result missed the limit.

## Visual review

The cream variants appear closer to the source product color. The 201 clay edit still shifts green toward brown, while the 202 sage edit stays teal instead of moving toward the brown target. Small words and counts remain distorted or unclear. The study did not score text correctness.

The contact-sheet tiles are unscaled 512×512 RGB images.

- [Carton 201 contact sheet](assets/preservation-study-2026-10-11/product-1.png)
- [Carton 202 contact sheet](assets/preservation-study-2026-10-11/product-2.png)
- [Carton 203 contact sheet](assets/preservation-study-2026-10-11/product-3.png)

## Runtime and cost

All nine fresh-load replays were pixel-exact. The worker validated 200 adapter tensors across 100 target modules. Delayed shared-workspace usage later showed $6.67920668 gross and $0 net after credits. This is a delayed workspace meter reading, not a per-run final invoice; assistant-token costs are separate.

These are development results on synthetic products. This report contains no real photos. The results do not establish a broad model ranking or evaluate the adapters on real photos. The separate real-photo R2 result remains about the prior flat-background adapter only.
