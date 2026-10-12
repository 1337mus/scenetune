# Protected-product comparison: R2 result

## Outcome

Neither method produced a usable image. Composites preserved source pixels exactly, but all 24 had poor ground contact and 14 added an unrequested upright stone; every full-image edit changed product size, shape or color. The reserved gate failed: 0/18 composites were usable, with no gain over full-image edits.

The 24 selected Amazon Berkeley Objects photos were split by whole brand and product family into six development and 18 reserved cases. This clean catalog sample and assistant visual judgments do not establish a broad model ranking or human preference. No OCR or word-accuracy score was collected.

## Frozen comparison and execution

| Measure | R2 result |
|---|---|
| Run ID | `scenetune-protected-comparison-2026-10-12-r2` |
| Source commit | [e05bb07b449be968e2c54f373815dcd3ebcbaa86](https://github.com/1337mus/scenetune/commit/e05bb07b449be968e2c54f373815dcd3ebcbaa86) |
| CI | [Run 38193542749, attempt 1](https://github.com/1337mus/scenetune/actions/runs/38193542749) |
| Inputs | 24 reviewed real catalog photos; 15 distinct brands |
| Split | 6 development / 18 reserved; whole brands and product families kept in one pool |
| Methods | FLUX.2 Klein Base 4B full-image edit; empty-scene generation followed by compositing the original product |
| Image settings | 512 × 512, 16 steps, guidance 4; matched case seeds |
| Generation | 52 calls: 48 main calls plus 4 fresh-load replay calls |
| Artifacts | 78 PNGs: 72 main outputs plus 6 replay outputs |
| Compositing and replay | 26 composites, including 2 replay composites; 4 exact fresh-load replays |
| Model loads | 2 successful model loads |
| Worker resources and limits | One A100 80 GB, 4 CPU cores, 64 GiB RAM; 300-second startup and 1,800-second execution limits |
| Training | 0 optimizer updates |
| Worker time | 394.674 seconds (about 6 minutes 35 seconds) |
| Peak memory | 16.46 GiB allocated by Torch; this is not total GPU memory |
| Result SHA-256 | `ab0d44f2f72d094298bdbace4d436c3a9ef0be97dbd7e284ff6d26d2e13b55f5` |

The 24 primary full-image-edit calls used 226.13 seconds of cumulative generation time; the 24 primary empty-scene calls used 57.26 seconds. These generation-call totals exclude model loading, compositing and other worker work. They do not establish a broader speed advantage.

## Validation and preserved failure history

R1 stopped on its first generation because the remote image lacked a C compiler. That original failed run remains unchanged and produced no images. A separate CPU-only remote image-build check later verified that C and C++ programs could run and that make was available; it passed without starting inference.

R2 then completed its planned calls, replays and shutdown. Its original local receipt validator rejected the successful worker receipt because its exact-field check did not allow the pinned Modal SDK version. All 17 pinned runtime entries—the Modal SDK version and 16 package versions—matched. This was a validator defect, not a generation failure or a pin mismatch.

A narrow offline validator correction passed a test that reproduced the rejection and then passed with the fix, 26 affected-module CPU tests and 27 independent checker tests. The official audit and separate independent audit then passed the same unchanged R2 result bundle, identified by the result hash above. The original R2 attempt marker, pending result, source and frozen plan were preserved. There was no recovery run, relaunch, new GPU generation or training update.

## Integrity and visual results

On the processed 512 × 512 images, the empty-scene compositor preserved all protected source-product pixels exactly in all 24 cases, including all 18 reserved cases. This verifies the compositing operation; it does not mean the generated scene was usable. The full-image edit changed protected product pixels in every case. Root visual review found size, shape or color changes in all 24 full-image edits; some labels also looked distorted. No word score was measured.

All 24 composites failed the ground-contact check because contact shadows were missing or the product appeared to overlap the front of its pedestal. Counts below overlap: the unrequested upright stone-like object appeared in 14 cases (5 development and 9 reserved). Every case was judged unusable in both methods under the frozen rule: preserve the whole product, follow the fixed scene request, and avoid obvious gaps, extra objects or floating placement.

| Case | Pool | Upright stone |
|---|---|---|
| B00EHZYWGM | Reserved | Yes |
| B00OAJ2MRG | Development | Yes |
| B01FI9CGVS | Reserved | No |
| B01GKJ27YG | Reserved | Yes |
| B01H64JL6Q | Reserved | Yes |
| B01LZHHCIT | Development | Yes |
| B072MQ5B53 | Development | Yes |
| B074H7J3NG | Reserved | No |
| B074MFBVX4 | Reserved | No |
| B074MGVXP7 | Reserved | Yes |
| B074MHFSXJ | Reserved | Yes |
| B078ZML314 | Development | Yes |
| B07BL6RDW7 | Reserved | No |
| B07CX8VMX8 | Reserved | Yes |
| B07FFHHG1Q | Reserved | No |
| B07GKDZM4J | Reserved | Yes |
| B07NJPQLFZ | Reserved | No |
| B07Q381TZ4 | Reserved | No |
| B07QZ5G7G5 | Reserved | No |
| B07RGQT3G1 | Reserved | Yes |
| B07RJYB22P | Reserved | Yes |
| B07T1LGMF5 | Development | Yes |
| B07TXPFTHG | Development | No |
| B083RBSL56 | Reserved | No |

The borderline case B074MFBVX4, a cotton-swab product, overlaps the pedestal front without convincing contact. Even counting it as usable would raise the reserved composite count only to 1 of 18, still far below the 14-of-18 threshold.

The frozen gate required at least 14 usable reserved composites and at least four more usable cases than the full-image edit. Observed results were 0 of 18 versus 0 of 18, a gain of zero. The gate failed.

## Runtime and cost limits

| Measure | Evidence |
|---|---|
| R2 compute envelope estimate | $1.865808 |
| Conservative reserve | $5.731616 for the R1 and R2 compute bounds plus $1 mask and $1 CPU-check reserves, within the existing $8 allowance |
| Workspace meter at 2026-10-12 03:36:19 UTC | $7.48773380 gross; $0 net after credits |
| Billing limit | The delayed meter is a workspace total, not a per-run invoice; it is not a provider spending cap |
| Exclusions | Build, storage, transfer and shutdown lag; assistant-token costs are separate and unavailable |

The earlier $15 pilot remains closed. No additional training or GPU work was authorized by this comparison.

## What to do next

Do not add epochs or launch GPU work automatically from this result. First address the failures: make empty-scene backgrounds genuinely empty, improve product-to-ground contact and shadows, and keep product interiors unchanged. Any future study requires new untouched product identities, a separate root-reviewed plan and compute clearance.
