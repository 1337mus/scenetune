from __future__ import annotations

import pytest

from product_scene_editor import remote


def test_matched_schedule_accepts_only_the_exact_144_update_contract() -> None:
    validator = getattr(remote, "_validate_matched_training_contract", None)
    assert callable(validator), "matched schedule validator is missing"

    assert (
        validator(
            train_pair_count=18,
            batch_size=1,
            dataset_repeat=1,
            num_epochs=8,
            gradient_accumulation_steps=1,
            process_count=1,
            save_steps=144,
        )
        == 144
    )
    with pytest.raises(ValueError, match="144"):
        validator(
            train_pair_count=18,
            batch_size=1,
            dataset_repeat=1,
            num_epochs=8,
            gradient_accumulation_steps=1,
            process_count=1,
            save_steps=143,
        )


def test_training_profile_must_be_one_of_the_private_canonical_instances() -> None:
    from dataclasses import replace

    resolver = getattr(remote, "_require_training_profile", None)
    assert callable(resolver), "canonical profile identity guard is missing"
    profile = getattr(remote, "_MATCHED_PROFILE", None)
    assert profile is not None, "matched training profile is missing"

    assert resolver(profile) is profile
    with pytest.raises(ValueError, match="profile"):
        resolver(replace(profile))


def test_matched_optimizer_guard_allows_update_144_and_rejects_update_145() -> None:
    guard = getattr(remote, "_guard_profile_optimizer_update", None)
    assert callable(guard), "profile-scoped optimizer update guard is missing"

    guard(remote._MATCHED_PROFILE, 143)
    with pytest.raises(RuntimeError, match="144"):
        guard(remote._MATCHED_PROFILE, 144)


def test_matched_runner_rejects_invalid_payload_on_cpu() -> None:
    import io
    import json
    import zipfile

    runner = getattr(remote, "run_remote_matched_study", None)
    assert callable(runner), "matched remote entrypoint is missing"

    result = runner(b"not a matched-study archive", "matched-test-001")
    with zipfile.ZipFile(io.BytesIO(result)) as archive:
        receipt = json.loads(archive.read("receipt.json"))
    assert receipt["run_id"] == "matched-test-001"
    assert receipt["status"] == "failed"
    assert receipt["failure"]["stage"] == "input_validation"
    assert receipt["failure"]["type"] == "ValueError"


def test_matched_reload_uses_one_live_replay_model_for_every_pair(monkeypatch, tmp_path) -> None:
    from types import SimpleNamespace

    from product_scene_editor import reload_diagnostic

    load_calls: list[object] = []
    generated: list[tuple[str, object]] = []
    replay_pipe = SimpleNamespace(device="cuda")
    replay_model = object()
    torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None))

    def load_replay_model(snapshot_dirs, adapter_path, torch_arg):
        load_calls.append(torch_arg)
        return replay_model, replay_pipe, {"adapter_validation": {"validated": True}}

    def generate(pipe, prompt, reference, seed):
        generated.append((prompt, pipe))
        assert pipe is replay_pipe
        assert replay_model is not None
        return SimpleNamespace(convert=lambda mode: (prompt, mode, seed))

    monkeypatch.setattr(reload_diagnostic, "_load_replay_model", load_replay_model)
    monkeypatch.setattr(remote, "_generate", generate)
    assets = {
        f"pair-{index}": {"prompt": f"prompt {index}", "source": object()} for index in range(9)
    }

    helper = getattr(remote, "_reload_and_generate_pairs", None)
    assert callable(helper), "matched strict-reload batch helper is missing"
    images, fields = helper({}, tmp_path / "adapter.safetensors", assets, 7321, torch)

    assert len(load_calls) == 1
    assert len(generated) == len(assets)
    assert set(images) == set(assets)
    assert all(pipe is replay_pipe for _prompt, pipe in generated)
    assert fields["adapter_validation"] == {"validated": True}


def test_source_and_target_must_match_inside_foreground_before_scoring() -> None:
    from PIL import Image

    validator = getattr(remote, "_validate_matched_foreground_identity", None)
    assert callable(validator), "matched foreground identity guard is missing"
    source = Image.new("RGB", (512, 512), (10, 20, 30))
    target = source.copy()
    mask = Image.new("L", (512, 512), 0)
    mask.putpixel((0, 0), 255)
    target.putpixel((1, 1), (200, 210, 220))

    validator(source, target, mask)
    target.putpixel((0, 0), (200, 210, 220))
    with pytest.raises(ValueError, match="foreground"):
        validator(source, target, mask)


def test_matched_reload_pixel_metrics_count_changed_pixels_per_pair() -> None:
    from PIL import Image

    calculator = getattr(remote, "_matched_reload_pixel_metrics", None)
    assert callable(calculator), "per-pair reload pixel metric calculator is missing"
    trained = Image.new("RGB", (512, 512), (0, 0, 0))
    reloaded = trained.copy()
    reloaded.putpixel((0, 0), (9, 0, 0))

    assert calculator(trained, trained) == {
        "mean_absolute_pixel_error": 0.0,
        "fraction_of_pixels_over_8": 0.0,
        "max_absolute_pixel_error": 0,
    }
    metrics = calculator(trained, reloaded)
    assert metrics["mean_absolute_pixel_error"] == pytest.approx(9 / (512 * 512 * 3))
    assert metrics["fraction_of_pixels_over_8"] == pytest.approx(1 / (512 * 512))
    assert metrics["max_absolute_pixel_error"] == 9


def test_matched_failure_receipt_keeps_full_144_step_progress(tmp_path) -> None:
    import io
    import json
    import zipfile

    progress = {
        "optimizer_updates": 145,
        "optimizer_steps_attempted": 145,
        "losses": [float(index) for index in range(145)],
        "gradient_steps": [{"finite": True} for _ in range(145)],
        "elapsed_seconds": 1.0,
        "step_seconds": [0.1] * 145,
    }
    result = remote._failure_bundle(
        "matched-001",
        "training",
        RuntimeError("failed"),
        progress,
        tmp_path,
        remote._MATCHED_PROFILE,
    )
    with zipfile.ZipFile(io.BytesIO(result)) as archive:
        receipt = json.loads(archive.read("receipt.json"))

    assert receipt["study_type"] == "matched144"
    assert receipt["progress"]["optimizer_updates"] == 144
    assert receipt["progress"]["optimizer_steps_attempted"] == 144
    assert len(receipt["progress"]["losses"]) == 144
    assert len(receipt["progress"]["gradient_steps"]) == 144
    assert len(receipt["progress"]["step_seconds"]) == 144


def test_partial_matched_reload_keeps_earlier_images_in_failure_bundle(
    monkeypatch, tmp_path
) -> None:
    import io
    import zipfile

    from PIL import Image

    from product_scene_editor import reload_diagnostic

    replay_pipe = object()
    torch = type(
        "Torch", (), {"cuda": type("Cuda", (), {"empty_cache": staticmethod(lambda: None)})}
    )()
    calls = 0

    def load_replay_model(snapshot_dirs, adapter_path, torch_arg):
        return object(), replay_pipe, {"adapter_validation": {"validated": True}}

    def generate(pipe, prompt, reference, seed):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second matched generation failed")
        return Image.new("RGB", (512, 512), (1, 2, 3))

    monkeypatch.setattr(reload_diagnostic, "_load_replay_model", load_replay_model)
    monkeypatch.setattr(remote, "_generate", generate)
    assets = {
        f"pair-{index}": {"prompt": f"prompt {index}", "source": object()} for index in range(3)
    }
    progress: dict[str, object] = {"saved_artifacts": []}
    with pytest.raises(RuntimeError, match="second matched generation"):
        remote._reload_and_generate_pairs(
            {},
            tmp_path / "adapter.safetensors",
            assets,
            7321,
            torch,
            work_root=tmp_path,
            progress=progress,
        )

    result = remote._failure_bundle(
        "matched-001",
        "adapter_reload",
        RuntimeError("failed"),
        progress,
        tmp_path,
        remote._MATCHED_PROFILE,
    )
    with zipfile.ZipFile(io.BytesIO(result)) as archive:
        names = set(archive.namelist())
    assert "images/pair-0/reloaded.png" in names
    assert names == {"receipt.json", "images/pair-0/reloaded.png"}


def test_controlled3d_profile_is_canonical_and_shares_the_144_update_contract() -> None:
    profile = getattr(remote, "_CONTROLLED3D_PROFILE", None)
    assert profile is not None, "Controlled3D study profile is missing"
    assert remote._require_training_profile(profile) is profile
    remote._validate_training_profile_contract(
        profile,
        train_pair_count=18,
        batch_size=1,
        dataset_repeat=1,
        num_epochs=8,
        gradient_accumulation_steps=1,
        process_count=1,
        save_steps=144,
    )
    remote._guard_profile_optimizer_update(profile, 143)
    with pytest.raises(RuntimeError, match="144"):
        remote._guard_profile_optimizer_update(profile, 144)


def test_controlled3d_source_and_target_foreground_changes_are_profile_scoped() -> None:
    from PIL import Image

    source = Image.new("RGB", (512, 512), (10, 20, 30))
    target = source.copy()
    mask = Image.new("L", (512, 512), 0)
    for x in range(100, 120):
        for y in range(100, 120):
            mask.putpixel((x, y), 255)
            target.putpixel((x, y), (200, 210, 220))

    validator = getattr(remote, "_validate_controlled3d_foreground_inputs", None)
    assert callable(validator), "Controlled3D needs its scoped foreground validator"
    validator(source, target, mask)
    with pytest.raises(ValueError, match="mask"):
        validator(source, target, Image.new("L", (512, 512), 128))


def test_controlled3d_metrics_name_source_and_target_references() -> None:
    from PIL import Image, ImageDraw

    source = Image.new("RGB", (512, 512), (0, 0, 0))
    target = source.copy()
    ImageDraw.Draw(target).rectangle((20, 20, 39, 39), fill=(200, 200, 200))
    mask = Image.new("L", (512, 512), 0)
    ImageDraw.Draw(mask).rectangle((20, 20, 39, 39), fill=255)
    assets = {"p-dev-cream": {"source": source, "target": target, "mask": mask}}
    generated = {mode: {"p-dev-cream": target.copy()} for mode in ("base", "trained", "reloaded")}

    metrics = remote._score_controlled3d_generations(assets, generated)

    assert metrics["p-dev-cream"]["trained"]["foreground_mean_absolute_error"] > 0
    assert metrics["p-dev-cream"]["trained"]["target_foreground_mean_absolute_error"] == 0
    assert metrics["p-dev-cream"]["base"]["source_to_target_foreground_mean_absolute_error"] > 0
    assert (
        remote.CONTROLLED3D_METRIC_REFERENCE["source_to_target_foreground_mean_absolute_error"]
        == "source_to_target_baseline"
    )
    assert remote.CONTROLLED3D_METRIC_REFERENCE["foreground_mean_absolute_error"] == "source"
    assert remote.CONTROLLED3D_METRIC_REFERENCE["target_foreground_mean_absolute_error"] == "target"
    assert "label" in remote.CONTROLLED3D_METRIC_REFERENCE["limitation"]


def test_controlled3d_worker_fails_closed_before_gpu_when_manifest_pin_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import io
    import json
    import zipfile

    from product_scene_editor import controlled3d_study

    monkeypatch.setattr(controlled3d_study, "APPROVED_CONTROLLED3D_MANIFEST_SHA256", None)
    result = remote.run_remote_controlled3d_study(b"invalid archive", "controlled3d-test-001")
    with zipfile.ZipFile(io.BytesIO(result)) as archive:
        receipt = json.loads(archive.read("receipt.json"))

    assert receipt["study_type"] == "controlled3d144"
    assert receipt["status"] == "failed"
    assert receipt["failure"]["stage"] == "input_validation"
    assert "manifest digest pin is unset" in receipt["failure"]["message"]
