# R3 saved-adapter reload diagnostic

This is a bounded, inference-only check of the adapter from `klein-edit-pilot-2026-10-08-r3`. It packages only the pinned R3 adapter, its development reference image, the saved trained PNG, and a manifest. Planning verifies the full original input/result archive hashes, the failed `adapter_reload` receipt, image hashes, and the adapter's 200 BF16 rank-32 tensors. It does not modify the R3 archives.

The default CLI path is CPU-only and never imports Modal or checks provider identity:

```sh
uv run python -m experiments.modal_reload_diagnostic
```

Planning writes a fresh directory under `artifacts/reload-diagnostics/r3`. Existing plan directories are never overwritten. The input bundle is limited to 120 MiB and contains exactly the manifest, adapter, reference, and original trained image. The worker validates the same checks again before importing Torch or downloading model components.

An explicitly cleared one-shot call is available for a later operator run:

```sh
uv run --extra control python -m experiments.modal_reload_diagnostic \
  --launch --profile reflex-personal --workspace rajath-61258 --environment main
```

Before Modal identity, image build, or spawn, launch requires `.context/reload-diagnostic-clearance.json` for the exact payload and plan. The JSON clearance must contain `schema_version: 1`, the run ID, `payload_sha256`, `plan_sha256`, both pinned source archive hashes, `status: "approved"`, `review_status: "approved"`, `ci_status: "success"`, `allowance_usd: 15`, the exact profile/workspace/environment, `gpu: "A100-80GB"`, `max_runtime_seconds: 1200`, `max_calls: 1`, a full `source_commit`, and a timezone-aware `checked_at` no more than 30 minutes old. Launch also requires a clean worktree and a locally verifiable signed HEAD. The $15 value is the project pilot ceiling; preflight estimated about $1.07 for this single A100-80GB call, with a 1,200-second total bound. Record actual usage after the run.

The worker performs two fresh unmerged `Flux2ImageTrainingModule` loads and one separate DiffSynth-fused control. Each path uses the same pinned components, prompt, reference, seed 7321, 16 inference steps, CFG 4, and 512×512 output. It preserves the constructed module/dropout training modes and records them; generation runs under `torch.inference_mode()`. Each unmerged load must match the adapter's exact live LoRA key set, shapes, dtypes, finite values, and restored tensor values. The worker never creates an optimizer or calls a training loop and reports zero optimizer updates.

The receipt compares each output with the original trained PNG and compares the two independent unmerged outputs with one another. It records the existing R3 limits (MAE ≤2 and fraction of pixels with any channel error >8 ≤0.01) without changing them. A completed measurement is reported as `measured` even if a comparison exceeds a limit; this is a diagnostic, not a cause finding or image-quality claim. Failure receipts keep any outputs already produced. The result archive is capped at 8 MiB and does not include the large adapter.

The attempt marker is exclusive and is reserved before provider identity checks. It records the same app/call IDs and instructs the operator to observe only that call and verify the detached app has stopped. No automatic retry or replacement call is started.

CPU tests use small generated Safetensors and ZIP fixtures so they run in a clean checkout. They verify schema and payload tampering, exact LoRA restoration checks, partial receipts, default plan-only behavior, clearance age/account/HEAD guards, and marker ordering. They do not establish live GPU loading, numerical parity, image quality, or actual cost; those observations belong in the returned diagnostic receipt and provider usage record.
