# Photographic target preview

Updated: 2026-10-10. Status: CPU proof-of-fit complete; training held.

## Decision

Use Poly Haven as the one low-friction CC0 source for a private CPU proof-of-fit. The acquired **Photo Studio Loft Hall** backplate is 5,568×3,712 and shows a full room, floor, and seating, with no tabletop. Its public file URL is [DSC_8539.jpg](https://dl.polyhaven.org/file/ph-assets/HDRIs/extra/Backplates/photo_studio_loft_hall/jpg_pretty/DSC_8539.jpg). The file is 8,533,025 bytes, provider MD5 `0b87504bf9922fe70bc0b88d3504765c`, and acquired-file SHA-256 `bab678ba0449f56c61544703338b2fb55f60b9f4ba55c25eedac88f5977f1ecc`. The source and any crop are not yet proven suitable for product placement.

Poly Haven's [license page](https://polyhaven.com/license) says its assets are CC0. Its [API page](https://polyhaven.com/our-api) says research and ML bulk downloads are supported, requires a unique `User-Agent`, asks for Poly Haven credit when the live API is used in a user-facing product, and describes the API as best-effort. Recheck its terms before any later integration.

| Candidate | Use | Limitation |
| --- | --- | --- |
| [Photo Studio Loft Hall](https://polyhaven.com/a/photo_studio_loft_hall) | Acquired flat LDR backplate for the first CPU illustration. | The visible room has no tabletop. It is one background family. |
| [Ferndale Studio 12](https://polyhaven.com/a/ferndale_studio_12) | Optional later candidate. It is an indoor, colored-light studio. | It has a 2:1 HDRI, not a flat backplate. Its tonemapped JPG is 17.6 MB and needs a fixed camera projection. |

## One-case CPU preview

The reviewed CPU helper generated one fresh procedural subject and placed it on a 512×512 photo crop. It read no existing training or test images. The original dataset manifest remains unchanged. The private preview and source files stay out of Git.

An independent pixel recount passed:

| Check | Result |
| --- | ---: |
| Product pixels checked | 43,704 |
| Product pixels changed | 0 |
| Background pixels matching the supplied photo | 218,440 / 218,440 |
| Training updates / new GPU runs | 0 / 0 |

The root visual review found an oversized, pasted-looking package with no credible contact shadow or matching light. This is an assistant observation, not a user rating. The preview is **not ready for training**. Exact pixel copying does not establish realism or model performance. The composite also requires a supplied background; it is not a matched competitor to a model that receives only an editing instruction.

## Later study gate

Do not freeze training data until a better target process addresses scale, a visible support surface, perspective, and contact shadows. Fresh procedural subjects are illustrative only. One background family, including all crops from it, cannot span train, development, and final-test splits. Any later study needs separate product identities and complete background families per split. Keep Barilla, Kinder Bueno, and Quaker Cruesli out of training and final testing; do not open the old sealed test. These are synthetic pairs on photographic backgrounds, not real product-photo training data. R2 base preference was 3/3 on three development cases; this small result is not a target-quality claim.

Any future launch still needs fresh usage and rate checks, frozen source/data/model/dependency pins, exact-source CI, and root clearance. The existing $15 pilot allowance remains the only current allocation. No new compute is assumed, and no training-cost estimate is claimed.
