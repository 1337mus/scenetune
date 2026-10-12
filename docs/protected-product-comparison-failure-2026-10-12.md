# Protected-product comparison: first attempt stopped

Run `scenetune-protected-comparison-2026-10-12-r1` made one remote call and stopped during its first image generation. The app was confirmed stopped with zero tasks. All 24 source photos and masks had been reviewed; reserved generated outputs remain unopened. No image-quality conclusion is available.

## Planned and observed

| Measure | Frozen plan | Attempt 1 |
| --- | --- | --- |
| Inputs | 24 accepted catalog cutouts | Same frozen payload; all 24 source photos and masks reviewed |
| Split | 6 development / 18 reserved | Unchanged; reserved generated outputs unopened |
| Model calls | 48 main calls plus 4 fresh-load replays (52 total) | 1 attempted; 0 succeeded |
| PNG artifacts | 78 planned across outputs, composites and replays | 0 produced |
| Pipeline | Load pinned FLUX.2 Klein Base 4B, then generate | Base pipeline loaded on CUDA in 21.88 seconds |
| Training | 0 optimizer updates | 0 |
| Runtime | One bounded call, zero retries | 115.77 seconds; one spawn; app stopped with zero tasks |

The 24 accepted cutouts consist of 18 original masks, three checked repairs and three replacement photos with checked masks. Whole brands and product families were kept separate between the six development cases and 18 reserved cases. Reserved generated outputs remain unopened.

## Failure

The worker failed on the first full-image edit:

```text
RuntimeError: Failed to find C compiler. Please specify via CC environment variable or set triton.knobs.build.impl.
```

The model loaded, but the first generation needed compiler tools that were missing. The official failed-receipt check passed. It recorded one attempted generation, zero successful generation records, zero output artifacts and zero optimizer updates. The reserved image and quality outputs were not opened. The run remains closed and immutable.

The local image builder now installs `build-essential` and checks `cc`, `c++` and `make` during image construction. CPU tests check the image-builder configuration. They do not test a real remote build. This change was not in the signed source used by this run, and no remote image build has verified it.

## Cost and limits

The comparison allowance is $8, including a $1 mask reserve. It is not a provider spending cap. At the 2026-10-12 01:31:55Z check, delayed workspace usage was $7.22932210 gross and $0 after credits. This is an aggregate workspace meter, not this run’s invoice. The final per-run charge is unknown, so the full allowance should not be treated as spent. Assistant-token costs are separate and unavailable. The earlier $15 pilot remains closed.

## Next boundary

There is no automatic retry. Any future attempt needs a separate reviewed plan and fresh source, data and usage checks, and must retain the same six development and 18 reserved cases. Keep reserved results closed. Verify the remote image build before GPU inference. Do not add epochs to the old datasets.

This failed run produced no images, scores or human ratings. It does not establish that either method is better or worse.

## Evidence

The signed [execution source](https://github.com/1337mus/scenetune/commit/ab7a429023893dee1e1d0b3bde19fe31c0fac7b3) passed exact [CI run 38190964224, attempt 1](https://github.com/1337mus/scenetune/actions/runs/38190964224).

| Record | Value |
| --- | --- |
| Plan SHA-256 | `8e41445d176b88ceea3a0ec6b488936edb259b70c27d30744eda0a48fae88b24` |
| Frozen payload SHA-256 | `6539907bac5f87be4aa9c5de54ec44c3daaa507dd42541fd1cd2da3976cf42a2` |
| Failed result SHA-256 | `d4db9d87188865e7608f0e6015a67825d16d3205d266814cf24c38b56d53e532` |
| Shutdown closure SHA-256 | `81ecd8a1de1d0cbc4131129e73b4913da220a0e0774545b18fc093607b457ca2` |
| Closure check | Stopped, 0 tasks, no other active personal apps at `2026-10-12T01:31:55.219610Z` |

The root evidence record and app-list closure are retained privately in ignored workspace paths: `.context/protected-comparison-root-failure-closure-r1.json` and `artifacts/protected-comparisons/scenetune-protected-comparison-2026-10-12-r1/`. These local records are not part of the GitHub report.
