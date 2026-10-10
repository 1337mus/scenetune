# R3 adapter reload diagnostic — 2026-10-09

**The separate-weight replay matched exactly. Combining the adapter weights reproduced the old R3 mismatch and failed the existing parity limits.** This diagnostic made no training updates and establishes no image-quality gain or real-photo performance.

## Results

| Replay path | Average channel difference | Pixels with any channel error >8 | Result |
|---|---:|---:|---|
| Separate-weight load 1 | 0.0 | 0% | Pass |
| Separate-weight load 2 | 0.0 | 0% | Pass |
| Load 1 vs. load 2 | 0.0 | 0% | Identical |
| Combined weights | 7.81574249 | 25.80299% | Fail; identical to the old R3 reload |

The existing limits remain 2 average channel levels and 1% of pixels above the 8-level threshold. The saved adapter was restored and validated as 200 BF16 tensors across 100 modules. The diagnostic made zero optimizer updates and generated three images.

## What the check establishes

In this test, two separate-weight loads reproduced the trained-image reference exactly. The combined-weight path changed the output to match the failed R3 reload. This narrows the mismatch to the combined path under these settings; it does not prove the exact cause.

The separate pipeline used `pipe_training=false`; the fused pipeline used `pipe_training=true`. LoRA modules reported training mode, while configured dropout was 0. Precision rounding is possible, but is not proven as the cause. The mode flags also differed.

## Runtime and operations

The receipt reports 178.68 seconds, including downloads, loading, and three generations. The app is stopped with zero tasks. The estimated resource cost was $1.07; the actual run charge is unknown. The delayed workspace meter showed $4.72154667 metered and $0 billed, not a per-run invoice.

The original R3 failure evidence remains unchanged. There were no retries, no training updates, and no sealed-test use. Next, use the working separate-weight reload in a fresh eight-update training pilot. This will check the complete training-and-reload process.

## Source and execution pins

- Executed source commit: [4b9c7c6fab3f3fe08ba3febe6806d2882e31e301](https://github.com/1337mus/product-scene-editor/commit/4b9c7c6fab3f3fe08ba3febe6806d2882e31e301)
- Exact CI run: [38020475190](https://github.com/1337mus/product-scene-editor/actions/runs/38020475190)
- DiffSynth-Studio revision: [ee631d4907d8d6fc3d39063a2a65c00a954c9bf5](https://github.com/modelscope/DiffSynth-Studio/commit/ee631d4907d8d6fc3d39063a2a65c00a954c9bf5)
