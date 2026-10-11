# Keeping the product unchanged

Training study · 11 October 2026

## Why run this study?

The [previous 3D study](controlled3d-study-result-2026-10-11.md) improved background edits. It also changed packet colors and distorted some small words.

This study tests one change to the training data. The model should learn to edit the background while leaving the product intact.

## What changes?

- **Background edits:** copy every product pixel from the source into the target image. Keep the edited background outside the product mask.
- **Unchanged examples:** include source–target pairs where the entire image stays the same.
- **Fresh training:** train a new LoRA from the same frozen original model.

| Item | Fixed amount |
| --- | ---: |
| Training products | 6 |
| Background-edit examples | 12 |
| Unchanged-image examples | 6 |
| Training updates | 144 |
| Passes through the training set | 8 |
| Development products, excluded from training | 3 |
| Development edits | 9 |

Each training product contributes two background edits and one unchanged example. The cream, sage and clay backgrounds each appear in four training edits.

## What counts as success?

Compare the new adapter with the previous 3D adapter.

| Measure | Goal |
| --- | --- |
| Average packet pixel error against the source | Lower |
| Edits with lower packet error | At least 6 of 9 |
| Average background error against the target | At most 20% higher |

Pixel error measures color differences on a 0–255 scale. Lower values mean a closer match. It does **not** measure readable words or overall appearance.

## How do we compare them?

Compare the original model, the previous adapter and the new adapter on the same nine inputs. Recalculate every score against the same new target images.

Keep the prompts, seed, 16 generation steps, guidance setting and 512 × 512 size fixed. Keep all model and trainer versions fixed too.

Check all 144 training updates. Save the adapter, load it into a fresh model, and require all nine replay images to match exactly. Review every result image and its label text.

Use one A100-80GB. Allow at most 45 minutes of execution and 5 minutes of startup. Allow one call and zero retries. These time limits are not a provider dollar cap.

## Limits

- These nine edits reuse development data. They are not a fresh independent test set.
- The synthetic data covers a small set of cartons. It does not prove performance on real product photos.
- Mark unreadable words as unclear. Do not guess them from the expected text.
- Test the saved adapters on the three confirmed real-photo cases after the synthetic run passes its execution checks.

The input archive and target pixels are checked before training. Portable PNG encoding keeps the target bytes independent of the image-library encoder version. Results will be reported after the run, saved-adapter checks and GPU shutdown are verified.
