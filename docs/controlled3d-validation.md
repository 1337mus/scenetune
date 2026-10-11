# Controlled3D validation API

This is an opt-in CPU checker. It does not render, train, change the dataset
validator, or add a CLI route. The reviewed R1 exclusion registry is pinned at
`e34a6e9ee0c9529daff383fa2d685aa5fb97e58716bcc68289369460e72902b5`; its
preview identities, aliases, canonical neutral backdrop, shelf alias, and prior
real-photo development identities are denied from every pool. The separate
fresh-family review approved planning concepts only. It registered no fresh
family or pool and authorized no rendering.

`draft_manifest(path)` reads at most 8 MiB of strict JSON, rejects duplicate
keys and non-finite numbers, and returns an object. It never follows asset
references. `validate_controlled3d(root, manifest, reviewed_registry=...,
registry_sha256=...)` checks the frozen manifest and registry, then checks
train/dev assets. The caller must supply the digest from a root-reviewed
registry; matching a digest proves byte identity only. The checker cannot
decide whether a new family is semantically fresh or whether the registry
received the required human review.

The manifest has exactly four fields:

```json
{
  "schema_version": 1,
  "provenance": {
    "synthetic_source": "original_synthetic",
    "license": "CC0-1.0",
    "external_inputs": [],
    "employer_inputs": []
  },
  "assignments": [
    {"kind":"product","id":"fresh-product-001","split":"train"},
    {"kind":"family","id":"fresh-family-001","split":"train"}
  ],
  "pairs": []
}
```

Assignment rows have exactly `kind`, `id`, and `split`. Splits are `train`,
`dev`, or `sealed_test`; each product and canonical family appears once.
Family assignments may name the canonical ID or one directly bound alias.
Aliases cannot chain, collide with a canonical ID, or split from the canonical
family.

Each pair row has exactly `pair_id`, `product_id`, `background_family_id`,
`split`, `expected_words`, `pins`, `snapshot_bundle`, and `renders`. It must
exactly match one registry pair. A snapshot bundle has `path` and `sha256`.
There are exactly four render rows, named `source`, `cream`, `sage`, and
`clay`. Each row has `name`, `image`, `mask`, and `bbox`; image references have
`path` and encoded-file `sha256`; mask references add `decoded_sha256`. Paths
are unique POSIX relative paths prefixed by their split. Empty components,
`.`/`..`, absolute paths, backslashes, symlinks, and non-regular files are
rejected. `bbox` is the exact half-open `[left, top, right, bottom]` mask box.

The reviewed registry has exactly `schema_version`, `products`, `families`,
and `pairs`. Product rows bind `product_id` to one split. Family rows bind a
`canonical_id` and a list of direct `aliases` to one split. Pair rows bind
`pair_id`, `product_id`, `background_family_id`, and `split`, plus
`expected_words` and `pins`. Every assignment and pair must be present in the
registry, and connected product/family components must remain in one split.
The exclusion list is applied independently of the supplied fresh registry,
so a custom registry cannot reopen a denied identity.

Every pair's `pins` object has exactly these fields: `recipe_sha256`,
`renderer_sha256`, `observer_sha256`, `validator_source_sha256`,
`capture_contract_sha256`, `blender_build`, `scene_sha256`, `font`, and
`local_files`. `blender_build` is `{version, sha256}`. `font` is
`{source_url, file_sha256, license_id, license_sha256}` and requires HTTPS plus
`OFL-1.1`. Each local pin row is `{pin, path, sha256}` and binds a file to
a declared pin. Train and dev pairs require exactly one local file for each of
`recipe_sha256`, `renderer_sha256`, `observer_sha256`, `capture_contract_sha256`,
`scene_sha256`, `font_file`, and `font_license`; the validator source is checked
against the installed `studio_state.py` directly. The Blender version/build is
registry-bound metadata checked by the root launch gate, so no Blender
installer/binary is copied into each pair. Sealed pin paths and digests remain
metadata-only. Manifest pair words and pins must equal the reviewed registry
pair.

Snapshot JSON is loaded from disk and passed to the existing
`studio_state.validate_studio_states`; a claimed receipt is not trusted. The
checker also compares captured label words and font metadata to pair pins. This
validates captured fields only: it does not prove that capture included every
relevant scene object or setting.

Images must be one-frame 512×512 PNGs in explicit RGB or RGBA mode, with opaque
alpha. Masks must be one-frame 512×512 grayscale (`L`) PNGs containing only 0
and 255, with equal decoded pixels across the four states, a nonempty unclipped
outline, the declared bounding box, and a nonempty 5×5-eroded product core.
`decoded_sha256` hashes the 262,144 decoded grayscale bytes in row-major order.
The report counts changed RGB pixels in the product, core, edge, and background
regions for each target versus source. Product changes are reported, not judged
by a quality threshold.

Sealed rows receive JSON, assignment, path, and hash-syntax checks only. Their
paths are never resolved, statted, opened, or hashed, including during export;
their files may be absent. Train/dev reads walk each component without
following symlinks and verify declared hashes.

`export_train_only(root, validated, destination)` revalidates the canonical
manifest, reviewed registry, root identity, and assets before copying. It
includes train assets, train assignments/pairs, and three target metadata rows
per train pair with `image`, `edit_image`, and `prompt`: `image` is the changed
target, `edit_image` is the source/reference, and each prompt asks to change only
the background. It does not include dev or sealed rows or assets. Destination
must be new, non-symlinked, and outside
the dataset root. Files are copied to an owned staging directory, checked, and
published under an exclusive destination claim. Consumers must treat an
export as unavailable until `completion.json` is present; that receipt is
written last. A failure removes the destination created by that export.
Publication assumes the destination parent is trusted and quiescent for the
operation. The exclusive directory creation claims the target, and cleanup
removes only a target created by this export. Hostile concurrent replacement of
the destination or an ancestor is outside the no-clobber guarantee.
