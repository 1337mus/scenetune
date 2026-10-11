from __future__ import annotations

from product_scene_editor import real_photo_comparison as comparison
from product_scene_editor.real_photo_runner import CASE_IDS


def test_generation_schedule_covers_five_fixed_arms_and_replays() -> None:
    schedule = comparison.generation_schedule()

    assert len(schedule) == 15
    assert [(task["arm"], task["case_id"]) for task in schedule] == [
        (arm, case_id)
        for arm in ("base", "prior", "prior_reloaded", "new", "new_reloaded")
        for case_id in CASE_IDS
    ]
    assert {task["session"] for task in schedule} == {
        "base",
        "prior-strict-1",
        "prior-strict-2",
        "new-strict-1",
        "new-strict-2",
    }


def test_clearance_bindings_returns_only_fixed_science_fields() -> None:
    marker = {
        "profile": "preservation-pair",
        "run_id": "pair-r1",
        "case_ids": list(CASE_IDS),
        "cases": [{"case_id": CASE_IDS[0], "input_sha256": "a" * 64}],
        "legacy_input_sha256": "b" * 64,
        "legacy_run_id": "real-photo-off-r1-2026-10-10-r2",
        "preparation_sha256": "c" * 64,
        "label_confirmation_sha256": "d" * 64,
        "confirmation": {"path": "confirmation.json", "sha256": "e" * 64},
        "adapters": {"prior": {"run_id": "prior"}, "new": {"run_id": "new"}},
        "prompt": "fixed prompt",
        "generation_settings": {"seed": 7321},
        "schedule": [{"case_id": CASE_IDS[0], "arm": "base", "session": "base"}],
        "modal_sdk": "1.6.1",
        "diffsynth_commit": "f" * 40,
        "runtime_pins": {"torch": "pinned"},
        "model_components": {"transformer": "pinned"},
        "trainer": {"repo": "pinned", "commit": "f" * 40},
        "training_function_called": False,
        "optimizer_updates": 0,
        "payload_sha256": "ignored",
    }

    bindings = comparison.clearance_bindings(marker)

    assert bindings["profile"] == comparison.PROFILE
    assert bindings["adapters"] == marker["adapters"]
    assert bindings["confirmation"] == marker["confirmation"]
    assert bindings["schedule"] == marker["schedule"]
    assert "payload_sha256" not in bindings


def test_payload_rejects_empty_archive() -> None:
    try:
        comparison.validate_payload_bytes(b"", "pair-r1")
    except ValueError as error:
        assert "empty" in str(error) or "input limit" in str(error)
    else:
        raise AssertionError("empty pair payload was accepted")


def _canonical_sha(value: object) -> str:
    import hashlib
    import json

    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _png(mode: str, color: int | tuple[int, int, int]) -> bytes:
    from io import BytesIO

    from PIL import Image

    stream = BytesIO()
    Image.new(mode, (512, 512), color).save(stream, format="PNG")
    return stream.getvalue()


def _binary_mask_png() -> bytes:
    from io import BytesIO

    from PIL import Image

    image = Image.new("L", (512, 512), 0)
    image.paste(255, (0, 0, 256, 512))
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def _pair_payload(monkeypatch) -> tuple[bytes, dict[str, object], dict[str, bytes]]:
    import hashlib
    import json
    import zipfile
    from io import BytesIO

    from product_scene_editor import real_photo_runner as legacy
    from product_scene_editor import remote

    adapter_bytes = {"prior": b"prior adapter fixture", "new": b"new adapter fixture"}
    adapter_pins = {
        arm: {
            "run_id": f"{arm}-source-r1",
            "archive_path": f"artifacts/{arm}-result.zip",
            "archive_sha256": hashlib.sha256(f"{arm} archive".encode()).hexdigest(),
            "receipt_member": "receipt.json",
            "receipt_sha256": hashlib.sha256(f"{arm} receipt".encode()).hexdigest(),
            "study_type": "controlled3d144" if arm == "prior" else "preservation144",
            "member_name": "adapter/step-144.safetensors",
            "member_sha256": hashlib.sha256(adapter_bytes[arm]).hexdigest(),
            "member_bytes": len(adapter_bytes[arm]),
            "payload_path": f"adapters/{arm}/step-144.safetensors",
        }
        for arm in ("prior", "new")
    }
    monkeypatch.setattr(comparison, "ADAPTER_PINS", adapter_pins, raising=False)
    monkeypatch.setattr(comparison, "LEGACY_INPUT_SHA256", "a" * 64, raising=False)
    monkeypatch.setattr(comparison, "CONFIRMATION_SHA256", "b" * 64, raising=False)

    confirmed_cases = []
    case_rows = []
    files: dict[str, bytes] = {}
    names = {
        "OFF-R1-003": "Barilla",
        "OFF-R1-005": "kinder",
        "OFF-R1-006": "QUAKER",
    }
    for index, case_id in enumerate(CASE_IDS):
        source_sha = f"{index + 1:x}" * 64
        image = _png("RGB", (20 + index, 30, 40))
        mask = _binary_mask_png()
        input_path = f"photos/{case_id}/input.png"
        mask_path = f"photos/{case_id}/mask.png"
        input_sha = hashlib.sha256(image).hexdigest()
        mask_sha = hashlib.sha256(mask).hexdigest()
        label_boxes = [
            {
                "text": names[case_id],
                "original_box": [1, 2, 20, 30],
                "transformed_box": [2.0, 3.0, 40.0, 50.0],
            }
        ]
        boxes_sha = _canonical_sha(label_boxes)
        outline = "confirmed_by_Raj" if index < 2 else "not_confirmed_by_Raj"
        shape_score = index < 2
        confirmed_cases.append(
            {
                "case_id": case_id,
                "product_id": f"product-{index}",
                "family_id": f"family-{index}",
                "role": "development_only",
                "source": {"path": f"source/{case_id}.jpg", "sha256": source_sha},
                "prepared_image": {"path": f"prepared/{case_id}.png", "sha256": input_sha},
                "mask": {"path": f"masks/{case_id}.png", "sha256": mask_sha},
                "labels_and_boxes": label_boxes,
                "label_boxes_canonical_sha256": boxes_sha,
                "label_box_geometry": "confirmed_by_Raj",
                "blue_outline_review": outline,
                "whole_package_shape_scoring_in_scope": shape_score,
                "masked_product_pixel_scoring_in_scope": shape_score,
                "scoring_applicability": (
                    "Output position and size are unchanged; otherwise mark score inapplicable."
                ),
                "exclusion_reason": None
                if shape_score
                else "Cropped package front; whole-package shape is unsupported.",
            }
        )
        case_rows.append(
            {
                "case_id": case_id,
                "product_id": f"product-{index}",
                "family_id": f"family-{index}",
                "role": "development_only",
                "source_sha256": source_sha,
                "input_path": input_path,
                "input_sha256": input_sha,
                "mask_path": mask_path,
                "mask_sha256": mask_sha,
                "label_boxes": label_boxes,
                "label_boxes_sha256": boxes_sha,
                "label_box_geometry": "confirmed_by_Raj",
                "blue_outline_review": outline,
                "whole_package_shape_scoring_in_scope": shape_score,
                "masked_product_pixel_scoring_in_scope": shape_score,
                "scoring_applicability": (
                    "Output position and size are unchanged; otherwise mark score inapplicable."
                ),
                "exclusion_reason": None
                if shape_score
                else "Cropped package front; whole-package shape is unsupported.",
            }
        )
        files[input_path] = image
        files[mask_path] = mask

    confirmation = {
        "schema_version": 1,
        "created_at_utc": "2026-10-11T21:00:00+00:00",
        "verification_performed_at_utc": "2026-10-11T21:00:00+00:00",
        "status": "current_source_confirmation_provenance_verified",
        "reviewed_by": "Raj",
        "purpose": (
            "Derived provenance summary of prior human confirmations; "
            "this file is not a new user confirmation."
        ),
        "preparation": {
            "path": ".context/real-photo-preparation.json",
            "sha256": "c" * 64,
            "aggregate_sha256": "d" * 64,
        },
        "prior_human_statements": {
            "spellings": {
                "path": ".context/real-photo-label-confirmation.json",
                "sha256": "e" * 64,
                "statement_exact": "Yes, the words match",
                "recorded_date": "2026-10-10",
                "recorded_at_utc": None,
                "timestamp_note": "Date only.",
                "scope": "Listed visible spellings only.",
            },
            "pink_boxes": {
                "path": ".context/real-photo-human-label-box-feedback-20261010-r1.json",
                "sha256": "f" * 64,
                "statement_exact": "Yes, the pink boxes are correct on all three",
                "recorded_at_utc": "2026-10-11T00:01:29.805520+00:00",
                "scope": "All three pink boxes.",
            },
            "blue_outlines": {
                "path": ".context/real-photo-human-geometry-feedback-20261010-r1.json",
                "sha256": "1" * 64,
                "statement_exact": (
                    "The blue lines do follow the packet edges in the first two pictures "
                    "but I do not see it happening in the last picture."
                ),
                "recorded_at_utc": "2026-10-11T00:00:23.605367+00:00",
                "scope": "First two outline confirmations only.",
            },
        },
        "preserved_source_records": {
            "label_draft": {
                "path": ".context/real-photo-label-draft.json",
                "sha256": "2" * 64,
                "scope": "Seven listed words.",
            },
            "original_geometry_review": {
                "path": ".context/real-photo-source-geometry-review.json",
                "sha256": "3" * 64,
                "status": "awaiting_human_review",
                "note": "Preserved.",
            },
        },
        "cases": confirmed_cases,
        "scoring_revision": {
            "path": ".context/real-photo-scoring-revision-r3.json",
            "sha256": "4" * 64,
            "status": "frozen_source_only_scoring_revision",
            "scope": "Source only.",
        },
        "rights_and_use": {
            "rights_review": "pending",
            "authorized_scope": "private inference development only",
            "training_authorized": False,
            "public_redistribution_authorized": False,
            "label_scope": "Only the seven listed visible strings and their confirmed boxes.",
        },
        "new_user_confirmation": False,
        "annotations_changed": False,
        "new_compute": False,
    }
    confirmation_bytes = (json.dumps(confirmation, ensure_ascii=False, indent=2) + "\n").encode()
    confirmation_sha = hashlib.sha256(confirmation_bytes).hexdigest()
    monkeypatch.setattr(comparison, "CONFIRMATION_SHA256", confirmation_sha, raising=False)
    confirmation_member = "confirmation/real-photo-current-confirmation-20261011.json"
    files[confirmation_member] = confirmation_bytes
    for arm in ("prior", "new"):
        files[adapter_pins[arm]["payload_path"]] = adapter_bytes[arm]
    monkeypatch.setattr(
        comparison,
        "LEGACY_CASE_PINS",
        {
            case["case_id"]: {
                key: case[key] for key in ("source_sha256", "input_sha256", "mask_sha256")
            }
            for case in case_rows
        },
    )

    bindings = {
        "profile": comparison.PROFILE,
        "run_id": "pair-r1",
        "case_ids": list(CASE_IDS),
        "cases": case_rows,
        "legacy_input_sha256": "a" * 64,
        "legacy_run_id": "real-photo-off-r1-2026-10-10-r2",
        "preparation_sha256": "d" * 64,
        "geometry_review_sha256": "3" * 64,
        "geometry_review_status": "awaiting_human_review",
        "quality_review_status": "pending_human_geometry_review",
        "active_intake_sha256": "5" * 64,
        "label_confirmation_sha256": "e" * 64,
        "private_source_review_sha256": "6" * 64,
        "confirmation": {
            "path": ".context/real-photo-current-confirmation-20261011.json",
            "member": confirmation_member,
            "sha256": confirmation_sha,
        },
        "adapters": adapter_pins,
        "prompt": legacy.PROMPT,
        "generation_settings": dict(legacy.GENERATION_SETTINGS),
        "schedule": list(comparison.generation_schedule()),
        "modal_sdk": "1.6.1",
        "diffsynth_commit": remote.DIFFSYNTH_COMMIT,
        "runtime_pins": dict(remote.RUNTIME_PINS),
        "model_components": remote.MODEL_COMPONENTS,
        "trainer": {
            "repo": "https://github.com/modelscope/DiffSynth-Studio",
            "commit": remote.DIFFSYNTH_COMMIT,
        },
        "training_function_called": False,
        "optimizer_updates": 0,
    }
    manifest = {"schema_version": 1, **bindings}
    files["manifest.json"] = legacy.canonical_json_bytes(manifest)
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, content)
    return stream.getvalue(), manifest, files


def test_payload_accepts_exact_pinned_pair_members(monkeypatch) -> None:
    payload, expected_manifest, expected_files = _pair_payload(monkeypatch)

    manifest, files = comparison.validate_payload_bytes(payload, "pair-r1")

    assert manifest == expected_manifest
    assert files.keys() == expected_files.keys()
    assert "adapter/step-144.safetensors" not in files


def test_payload_rejects_unreferenced_members(monkeypatch) -> None:
    import pytest

    from product_scene_editor import real_photo_runner as legacy

    payload, _, expected_files = _pair_payload(monkeypatch)
    expected_files["unexpected.bin"] = b"extra"

    with pytest.raises(ValueError, match="unexpected or unreferenced"):
        comparison.validate_payload_bytes(legacy.canonical_archive(expected_files), "pair-r1")


def test_payload_rejects_swapped_adapter_members(monkeypatch) -> None:
    import pytest

    from product_scene_editor import real_photo_runner as legacy

    _, _, expected_files = _pair_payload(monkeypatch)
    prior_path = comparison.ADAPTER_PINS["prior"]["payload_path"]
    new_path = comparison.ADAPTER_PINS["new"]["payload_path"]
    expected_files[prior_path], expected_files[new_path] = (
        expected_files[new_path],
        expected_files[prior_path],
    )

    with pytest.raises(ValueError, match="adapter payload"):
        comparison.validate_payload_bytes(legacy.canonical_archive(expected_files), "pair-r1")


def test_create_and_load_plan_are_write_once_and_bound(monkeypatch, tmp_path) -> None:
    import hashlib
    import json
    import zipfile
    from io import BytesIO

    import pytest

    payload, manifest, files = _pair_payload(monkeypatch)
    legacy_files = dict(files)
    legacy_manifest = manifest
    confirmation_bytes = files[manifest["confirmation"]["member"]]
    legacy_bytes = b"synthetic validated legacy archive"
    monkeypatch.setattr(comparison, "LEGACY_INPUT_SHA256", hashlib.sha256(legacy_bytes).hexdigest())
    monkeypatch.setattr(
        comparison.legacy,
        "validate_payload_bytes",
        lambda data, run_id: (legacy_manifest, legacy_files),
    )
    monkeypatch.setattr(
        comparison, "CONFIRMATION_SHA256", hashlib.sha256(confirmation_bytes).hexdigest()
    )

    adapter_archives = {}
    adapter_pins = {}
    for arm in ("prior", "new"):
        adapter = f"{arm} adapter archive member".encode()
        receipt = json.dumps({"run_id": f"{arm}-source-r1", "status": "passed"}).encode()
        archive_stream = BytesIO()
        with zipfile.ZipFile(archive_stream, "w") as archive:
            archive.writestr("receipt.json", receipt)
            archive.writestr("adapter/step-144.safetensors", adapter)
        archive_bytes = archive_stream.getvalue()
        adapter_archives[arm] = archive_bytes
        adapter_pins[arm] = {
            "run_id": f"{arm}-source-r1",
            "archive_path": f"artifacts/{arm}-result.zip",
            "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
            "receipt_member": "receipt.json",
            "receipt_sha256": hashlib.sha256(receipt).hexdigest(),
            "study_type": "controlled3d144" if arm == "prior" else "preservation144",
            "member_name": "adapter/step-144.safetensors",
            "member_sha256": hashlib.sha256(adapter).hexdigest(),
            "member_bytes": len(adapter),
            "payload_path": f"adapters/{arm}/step-144.safetensors",
        }
    monkeypatch.setattr(comparison, "ADAPTER_PINS", adapter_pins)

    plan = comparison.create_comparison_plan(
        tmp_path,
        tmp_path / "pair-plan",
        run_id="pair-r1",
        legacy_input=legacy_bytes,
        prior_archive=adapter_archives["prior"],
        new_archive=adapter_archives["new"],
        confirmation=confirmation_bytes,
    )
    loaded = comparison.load_plan_record(tmp_path / "pair-plan")

    assert loaded["payload_sha256"] == plan["payload_sha256"]
    assert loaded["plan_sha256"] == plan["plan_sha256"]
    with pytest.raises(ValueError, match="existing comparison plan"):
        comparison.create_comparison_plan(
            tmp_path,
            tmp_path / "pair-plan",
            run_id="pair-r1",
            legacy_input=legacy_bytes,
            prior_archive=adapter_archives["prior"],
            new_archive=adapter_archives["new"],
            confirmation=confirmation_bytes,
        )

    marker_path = tmp_path / "pair-plan" / comparison.RUN_MARKER_NAME
    marker_bytes = marker_path.read_bytes()
    altered = json.loads(marker_bytes)
    altered["profile"] = "wrong-profile"
    marker_path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="profile"):
        comparison.load_plan_record(tmp_path / "pair-plan")
    marker_path.write_bytes(marker_bytes)
    payload_path = tmp_path / "pair-plan" / comparison.PAYLOAD_NAME
    payload_path.write_bytes(payload + b"tamper")
    with pytest.raises(ValueError, match="does not bind its payload"):
        comparison.load_plan_record(tmp_path / "pair-plan")


def test_result_validates_all_outputs_strict_loads_and_replays(monkeypatch) -> None:
    import hashlib
    import json
    import zipfile
    from io import BytesIO

    import pytest

    payload, manifest, _ = _pair_payload(monkeypatch)
    payload_sha = hashlib.sha256(payload).hexdigest()
    plan = {**manifest, "payload_sha256": payload_sha}
    image = _png("RGB", (20, 30, 40))
    outputs = {
        f"outputs/{task['case_id']}/{task['arm']}.png": image
        for task in comparison.generation_schedule()
    }
    observed = {"pipe_training": True, "lora_dropout_training": [True]}
    validation = {
        "observed": observed,
        "mode_changed": False,
        "adapter_validation": {
            "validated": True,
            "restored_tensor_count": 200,
            "module_count": 100,
        },
    }
    strict = [
        {"arm": arm, "session": session, "validation": validation}
        for arm, session in (
            ("prior", "prior-strict-1"),
            ("prior", "prior-strict-2"),
            ("new", "new-strict-1"),
            ("new", "new-strict-2"),
        )
    ]
    receipt = {
        **manifest,
        "schema_version": 1,
        "profile": comparison.PROFILE,
        "run_id": "pair-r1",
        "status": "passed",
        "payload_sha256": payload_sha,
        "generation_settings": comparison.legacy.GENERATION_SETTINGS,
        "schedule": list(comparison.generation_schedule()),
        "training_function_called": False,
        "optimizer_updates": 0,
        "quality_claim": False,
        "worker_failure_confirmed": False,
        "output_artifacts": [
            {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "width": 512, "height": 512}
            for name, data in outputs.items()
        ],
        "strict_load_validations": strict,
        "replay_pixel_equal": {case_id: {"prior": True, "new": True} for case_id in CASE_IDS},
        "elapsed_seconds": 1.0,
    }
    files = {"receipt.json": json.dumps(receipt).encode(), **outputs}
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)

    actual, actual_paths = comparison.validate_result_bundle(
        stream.getvalue(), "pair-r1", expected_plan=plan, expected_payload_sha256=payload_sha
    )

    assert actual["status"] == "passed"
    assert actual_paths == set(outputs)

    mismatch_receipt = json.loads(json.dumps(receipt))
    mismatch_receipt["strict_load_validations"][1]["validation"]["observed"]["pipe_training"] = (
        False
    )
    mismatch_stream = BytesIO()
    with zipfile.ZipFile(mismatch_stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("receipt.json", json.dumps(mismatch_receipt).encode())
        for name, content in outputs.items():
            archive.writestr(name, content)
    with pytest.raises(ValueError, match="observed modes differ"):
        comparison.validate_result_bundle(
            mismatch_stream.getvalue(),
            "pair-r1",
            expected_plan=plan,
            expected_payload_sha256=payload_sha,
        )

    partial = dict(receipt)
    partial.update(
        status="failed",
        worker_failure_confirmed=True,
        output_artifacts=[],
        strict_load_validations=[],
        replay_pixel_equal=None,
        failure={"stage": "load", "type": "RuntimeError", "message": "fixture failure"},
    )
    partial_stream = BytesIO()
    with zipfile.ZipFile(partial_stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("receipt.json", json.dumps(partial).encode())
    failed, failed_paths = comparison.validate_result_bundle(
        partial_stream.getvalue(),
        "pair-r1",
        expected_plan=plan,
        expected_payload_sha256=payload_sha,
    )
    assert failed["status"] == "failed"
    assert failed_paths == set()

    tampered = dict(outputs)
    tampered["outputs/OFF-R1-003/prior_reloaded.png"] = _png("RGB", (21, 30, 40))
    false_claim_receipt = dict(receipt)
    false_claim_receipt["output_artifacts"] = [
        {
            "path": name,
            "sha256": hashlib.sha256(content).hexdigest(),
            "width": 512,
            "height": 512,
        }
        for name, content in tampered.items()
    ]
    tampered_files = {"receipt.json": json.dumps(false_claim_receipt).encode(), **tampered}
    false_stream = BytesIO()
    with zipfile.ZipFile(false_stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in tampered_files.items():
            archive.writestr(name, content)
    with pytest.raises(ValueError, match="replay equality claim"):
        comparison.validate_result_bundle(
            false_stream.getvalue(),
            "pair-r1",
            expected_plan=plan,
            expected_payload_sha256=payload_sha,
        )


def test_worker_failure_before_first_load_returns_bound_partial_receipt(monkeypatch) -> None:
    import hashlib
    import sys
    from types import ModuleType, SimpleNamespace

    from product_scene_editor import real_photo_comparison_inference as inference
    from product_scene_editor import real_photo_inference as legacy_inference

    payload, manifest, _ = _pair_payload(monkeypatch)
    payload_sha = hashlib.sha256(payload).hexdigest()
    plan = {**manifest, "payload_sha256": payload_sha}
    fake_cuda = SimpleNamespace(max_memory_allocated=lambda: 0)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))
    hub = ModuleType("huggingface_hub")
    hub.snapshot_download = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    def fail_runtime_import() -> dict[str, str]:
        raise RuntimeError("synthetic failure before the first strict load")

    monkeypatch.setattr(legacy_inference, "_runtime_versions", fail_runtime_import)
    result = inference.run_remote_real_photo_comparison(payload, "pair-r1")

    receipt, paths = comparison.validate_result_bundle(
        result, "pair-r1", expected_plan=plan, expected_payload_sha256=payload_sha
    )
    assert receipt["status"] == "failed"
    assert receipt["strict_load_validations"] == []
    assert paths == set()
