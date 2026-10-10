# Matched synthetic study input

The CPU planner packages all 18 approved training pairs and nine development pairs from the pinned
dataset manifest. It checks the manifest digest, train/dev identities, source/target/mask checksums,
binary masks, foreground bounds, unchanged product pixels, and a changed background. Sealed rows are
read only as manifest metadata; their paths are never resolved and their assets are never opened.

The bounded `matched-study-input.zip` contains 81 PNGs, 18 DiffSynth metadata rows, and a
`matched_manifest.json` with the exact `train_rows` and `dev_rows` contract, six/three/three product
counts, eight epochs, and 144 optimizer updates. A write-once `.matched-study-run.json` marker pins
the run ID, source manifest digest, payload digest, and selected pair IDs. Loaders recheck the marker,
archive schema, asset digests, and image invariants before returning bytes to the worker.

`score_pair` reports RGB foreground MAE and the fraction of foreground pixels whose largest channel
error exceeds 8, plus background MAE against the synthetic target and the fraction changed from the
source by that threshold. Foreground error measures fixed-position pixel preservation. It does not
measure OCR, shape, or general image quality.
