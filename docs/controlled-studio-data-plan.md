# Controlled 3D data plan

Status: proposal. No dataset or export is enabled yet.

## In simple terms

The current synthetic dataset requires exact package pixels between source and target. Keep that rule in `src/product_scene_editor/dataset.py` unchanged.

A 3D render can change product colors when a wall reflects light, even if the product shape and printed words stay fixed. That needs a separate schema and validator.

A small CPU preview can check whether one scene recipe is repeatable and measurable. One product and one studio family cannot establish real-photo quality or broad generalization.

## Separate data lane

Use a separately reviewed `controlled3d` schema and module. Do not weaken or reinterpret the existing verifier or default exporter.

Accept original synthetic geometry, materials, labels, scenes, and renders. Record the source, hash, and verified license for every font, including bundled fonts. Exclude Open Food Facts (OFF) photos and other external product photos.

## Pair record

| Record group | Proposed fields |
| --- | --- |
| Identity | `pair_id`, `product_id`, complete `background_family_id`, `split` |
| Provenance and files | owned-synthetic source; recipe SHA-256; renderer/version/build; saved scene SHA-256; encoded source, target, and mask file SHA-256 values |
| Per-render settings | seed, engine/device, resolution, samples, color management, output format; capture for each pass |
| Per-render scene snapshot | actual subject mesh geometry/materials; original label words, font file/hash, and converted label-mesh geometry; camera; lights; world; material nodes and values; canonical snapshot and SHA-256 for each pass |
| Allowed pair edit | existing wall material’s Principled Base Color RGB input only; save before/after values |
| Mask and measurements | decoded binary silhouette digest, separate from encoded mask-file hash (file metadata may differ); nonempty product core; clipping and RGB-change summaries for product/core/edge/background |

Before each render, capture a canonical snapshot from the live scene and put both the snapshot and its digest in that pass record. An independent validator must compare the captured fields with a root-reviewed allowlist; a renderer-supplied digest alone cannot prove that no scene change was omitted.

Keep the wall material identity and node graph fixed; only its Base Color RGB input may change. Keep product geometry, dimensions, orientation, label text, font, camera, lights, world, seed, and render settings fixed.

The wall can change bounced light on the product, so exact source/target product pixels are not required. Measure pixel counts and RGB differences for product, core, edge, and background; report geometry or word changes separately from light-driven color changes.

Require a binary mask with the same nonempty decoded silhouette in both renders. Require its whole-product bounding box and the product core to remain unclipped, with the core nonempty. Record clipping and color-change measurements, but choose no numeric pass thresholds or dataset counts until a root-reviewed study plan.

## Pool isolation and later gates

Preassign product IDs and complete background families to `train`, `dev`, or `sealed_test` before rendering. Keep every connected product/background group in one pool, exclude all preview subjects and families from new pools, and keep existing sealed tests closed.

Do not train or export this schema by default. A later export needs its own route and validator, a fixed root-reviewed data plan, deterministic checks, exact-source CI, and fresh usage and cost review before any GPU work.
