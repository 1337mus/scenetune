# Real-photo comparison attempt — 2026-10-10

## Result

The first remote attempt failed during runtime import, before model loading. The worker could not find `modal` package metadata (`PackageNotFoundError`) and produced no images. This was not a completed comparison.

| Item | Evidence |
|---|---|
| Run | `real-photo-off-r1-2026-10-10-r1` |
| Signed source commit | `9daa555fb380c90ec4649f78c69b1167d3ecf378` |
| Exact-source CI | Run `38094811214`, attempt 1 passed |
| Modal app / call | `ap-7JKPnxyjX6zbaFhLgSzbR5` / `fc-01M4M23YWXW429QTFXD5SRV56E` |
| Worker result | Failed at `runtime_import` in 3.824 seconds; zero outputs, optimizer updates, and strict-load validations |
| Shutdown | The helper's observation failed; a separate read-only check confirmed the app stopped with zero tasks |

## Cost

- Configured-time estimate: `$1.07`; this excludes extra charges and is not actual spend.
- Delayed workspace meter at closure: `$5.33` gross and `$0` net; it includes earlier work and can lag.
- Actual charge for this call: unknown.

## Scope and next step

The comparison remains three private development photos and seven planned generations at frozen settings, with zero training. Raj confirmed the seven source labels; product masks and text boxes remain pending geometry review. Inputs and any future outputs stay private. No output was opened and no score was produced.

The setup check now reads Modal’s version from its running module. CPU tests cover missing installation metadata and reject version mismatches. The correction has not been tested remotely.

No retry or replacement launch has been cleared.
