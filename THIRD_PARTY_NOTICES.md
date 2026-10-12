# Licenses and third-party sources

## Project code and examples

Project code and documentation are covered by the [MIT license](LICENSE).

The original procedural product images and scene targets described in
[the data contract](docs/data.md) are dedicated to the public domain under
[CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/).
The same dedication covers the project's original 3D render inputs and targets.

Comparison sheets also contain FLUX-generated outputs. Captions identify the
source, target, base-model output, and adapter output. These are reported examples;
the procedural-data dedication does not relicense upstream models or software.

Real catalog photos, model checkpoints, and trained adapter weights are not
included in this repository. Packaging brands are referenced only to describe
evaluation cases; no affiliation or endorsement is implied.

## Model and software sources

| Source | Terms and use |
|---|---|
| [FLUX.2 Klein Base 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) and [Klein 4B components](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | Apache-2.0 at the pinned revisions. Used for training and inference; weights are not bundled. See [component pins](docs/trainer-contract.md). |
| [BiRefNet Lite](https://huggingface.co/ZhengPeng7/BiRefNet_lite) | MIT at the pinned revision. Used in the separate cutout check; weights are not bundled. |
| [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio/tree/ee631d4907d8d6fc3d39063a2a65c00a954c9bf5) | External training and inference dependency. Its code keeps its upstream license. |
| [Aileron](https://dotcolon.net/fonts/aileron/) | CC0 according to its author. Pillow's default font uses Aileron Regular when FreeType is available. No font file is bundled here. |

Other dependencies retain their upstream licenses. Installing a dependency or
using its API does not change its license to the project's MIT license.

## Referenced photo datasets

- **Amazon Berkeley Objects (ABO):** used for the 24-product private evaluation.
  The [official dataset README](https://amazon-berkeley-objects.s3.amazonaws.com/README.md)
  specifies CC BY 4.0. Credit for image and 3D data belongs to Amazon.com.
  Credit for constructing the dataset, archives, and benchmark sets belongs to
  Matthieu Guillaumin, Thomas Dideriksen, Kenan Deng, Himanshu Arora,
  Jasmine Collins, and Jitendra Malik. Photos were resized and padded for the
  evaluation. They and their generated edits are not redistributed here.
- **Open Food Facts:** used for the earlier three-photo development checks.
  Source terms and unresolved packaging-rights limits are recorded in the
  [source review](docs/real-photo-source-check-2026-10-10.md). Photos, crops,
  annotations, and generated edits are not redistributed here.
