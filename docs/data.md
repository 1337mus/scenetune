# Synthetic product scene data

`product_scene_editor.render.render_pair(product_id, seed, target_scene)` creates an original Pillow render with a 512×512 neutral source, a target scene, a binary product mask, and a half-open `(left, top, right, bottom)` mask bounding box. The product geometry, label, and pixels are deterministic for a product ID and seed. The three target scenes change only the background; each prompt says to keep the package unchanged.

`product_scene_editor.dataset.build_dataset(output_dir, seed=20261008, products=12)` creates three target pairs per product. Its default split has six training identities, three development identities, and three sealed-test identities. Product IDs are assigned to one pool for all scenes. Asset directories are `train/`, `dev/`, and `sealed_test/`, each with `source/`, `target/`, and `masks/` subfolders.

The strict `manifest.json` records the product and pair IDs, split, prompt, relative asset paths, mask bbox, generator seed, SHA-256 digests, and CC0-1.0 provenance. It declares that every image is an original synthetic render, with no external assets or employer input. `verify_dataset(dataset_dir)` rejects unknown or duplicate JSON fields, unsafe paths, content checksum mismatches, product IDs or identical foregrounds crossing pools, malformed masks, incorrect bboxes, and source/target pairs whose product pixels differ or whose background is unchanged.

`export_training(dataset_dir, destination)` validates the complete dataset, refuses an existing destination, and copies only training pairs and masks. Its filtered `manifest.json` contains no development or sealed-test pairs. DiffSynth-compatible `metadata.json` is a JSON array with `image` pointing to each target, `edit_image` pointing to its source, and the background-only `prompt`.

These are simple procedural packaging illustrations, not photorealistic product photos. The pipeline checks identity grouping, pairing, masks, and export behavior; it does not establish commercial editing quality or evaluate product relighting.
