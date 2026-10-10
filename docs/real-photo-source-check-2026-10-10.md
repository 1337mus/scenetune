# Real-photo source check

## What we collected

We downloaded a small Open Food Facts sample after Raj reported API-use registration. Raw images, crops, full metadata and label drafts stay outside Git.

| Check | Result |
|---|---:|
| Image candidates in the input list | 6 |
| Camera-photo candidates after visual review | 3 |
| Packshots with uncertain camera origin | 3 |
| Photos with source versions, hashes and uploader records checked | 6 |
| Additional downloaded candidates kept excluded | 2 |
| Total image download bytes, including exclusions | 216,349 |
| Product lookups | 15: 7 found, 8 returned 404 |
| Test identities queried | 0 |
| New GPU runs | 0 |

Every request ran once. We kept the eight not-found responses and both exclusions. We did not expand collection to fill the gaps.

## What the images show

| Product | Source observation | Possible use |
|---|---|---|
| Barilla spaghetti | Product box on a photographed surface | Label and product checks, after review |
| Kinder Bueno | Wrapper on a textured surface | Label and product checks, after review |
| Quaker Cruesli | Camera-like photo of a cropped package front | Label check; no whole-product shape claim |
| Nutella, Prince, Coca-Cola | Polished, isolated packshots | Separate stress cases; camera origin unknown |

These are assistant visual observations. They do not prove ownership, camera origin or human review. An uploader name alone is not proof that the person took the photo. The Open Food Facts image version and raw-image uploader mapping were checked for every input.

## What remains

- Raj confirmed the seven listed text labels; boxes around text and product outlines still need review.
- Check product masks before reporting product-pixel changes. No masks are approved yet.
- Compare three development candidates with the saved step-144 adapter and zero training.
- Exclude all three comparison products from future training and final test; no test pool exists yet.

The original 3/3 train-candidate/development grouping is provisional. No real-photo training has happened. Any input used for the next comparison must become a development case and stay out of future training or final tests.

## License and limits

Open Food Facts lists product photos under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). Packaging artwork, logos and other third-party rights remain separate. We retain uploader attribution and [source terms](https://world.openfoodfacts.org/terms-of-use); this report does not grant public image reuse or training rights.

The intake checker passed. Passing it verifies files and grouping rules, not labels, masks or legal clearance. There are no real-photo model results or quality gains to report. The [evaluation plan](real-photo-evaluation.md) keeps label, product and background results separate.
