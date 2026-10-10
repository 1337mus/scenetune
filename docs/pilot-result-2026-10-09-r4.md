# R4 pilot result — 2026-10-09

**Outcome:** R4 passed the eight-update synthetic training and saved-adapter reload check. It started fresh from the base model and repeated R3’s eight-update schedule with the separate-weight reload path; it did not add eight updates to R3.

This confirms pipeline behavior only. It establishes no image-quality improvement or real-photo performance.

## Pins and inputs

- Source: [commit 5e3c47ed2e6738003157e77b01aa44f14f1afaef](https://github.com/1337mus/product-scene-editor/commit/5e3c47ed2e6738003157e77b01aa44f14f1afaef). [CI run 38021402616 passed](https://github.com/1337mus/product-scene-editor/actions/runs/38021402616).
- Trainer: DiffSynth revision ee631d4907d8d6fc3d39063a2a65c00a954c9bf5.
- Data: 8 training pairs across pack-000–005; one development pair from held-out product pack-006.
- Generation: same development source and prompt, seed 7321, 16 steps for base, trained, and reloaded outputs.
- Input ZIP SHA-256: 9e10a34851c15f5fef210730ad7201b2a353cbc459f5dfaca9f3f9880c856a5f.
- Result ZIP SHA-256: 8de556ec3b1719c27b78af6dabed528c7b11a5b04d51709ff8dacbaa3e7cacf8.
- [Contact sheet](assets/pilot-r4/contact-sheet.png). Its five 512×512 image panels are pixel-identical to the ZIP images; only labels and layout were added.

## Results

Image error is mean absolute 8-bit RGB channel difference on a 0–255 scale.

| Check | Result |
|---|---|
| Training | 8 optimizer updates with finite loss and gradient records. Worker receipt reports 200 changed tensors; max recorded parameter delta was 0.00048828125. |
| Saved adapter | 200 finite BF16 tensors across 100 modules and 48,168,960 trainable scalars. Independent audit confirmed exact restoration; it did not compare every tensor with its pre-training value. |
| Reload parity | Trained and reloaded images matched exactly: MAE 0; 0% of pixels above absolute error 8. Both frozen limits passed: MAE ≤2 and pixels above 8 ≤1%. |
| Base vs. trained | Average color change 8.59896/255; 27.6367% of pixels differ by more than 8/255. This measures output change, not improvement. |
| Time and memory | Training 10.5364 s; mean logged step 1.29735 s; peak Torch-allocated memory 17.5459 GiB, not total VRAM. Worker time to receipt was 145.561 s and excludes archive packing. |
| Shutdown | App stopped, 0 tasks, 0 active apps. |
| Cost | Delayed workspace meter at check: $4.72154667 gross, $0 net; revisable and not a final invoice. Actual R4 charge is unknown. Full-allocation estimate: $2.66544, excluding other charges. |

## Next step

Train one new adapter on the larger 18-pair training split, then compare the base and adapter on all 9 development pairs with source, prompt, and seed matched. Predefine checks for product shape, label readability, and the requested background; report every case. The brief remains under review, and no new run has been launched. The R3 report and reload diagnostic remain unchanged.
