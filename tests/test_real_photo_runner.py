from __future__ import annotations

from product_scene_editor import real_photo_runner as runner


def test_generation_schedule_is_three_bases_three_strict_reloads_and_one_fresh_repeat() -> None:
    schedule = runner.generation_schedule()

    assert len(schedule) == 7
    assert [(item["case_id"], item["mode"]) for item in schedule] == [
        ("OFF-R1-003", "base"),
        ("OFF-R1-005", "base"),
        ("OFF-R1-006", "base"),
        ("OFF-R1-003", "strict_adapter"),
        ("OFF-R1-005", "strict_adapter"),
        ("OFF-R1-006", "strict_adapter"),
        ("OFF-R1-003", "strict_repeat"),
    ]


def _valid_prep_and_review() -> tuple[dict[str, object], dict[str, object]]:
    from product_scene_editor.real_photo_runner import canonical_sha256

    cases = []
    reviewed_cases = []
    for case_id in runner.CASE_IDS:
        boxes = [
            {"text": "word", "original_box": [1, 2, 3, 4], "transformed_box": [2.0, 3.0, 4.0, 5.0]}
        ]
        box_hash = canonical_sha256(boxes)
        case = {
            "case_id": case_id,
            "split": "dev",
            "source_sha256": "a" * 64,
            "derived_sha256": "b" * 64,
            "mask_sha256": "c" * 64,
            "label_boxes": boxes,
            "label_boxes_sha256": box_hash,
        }
        cases.append(case)
        reviewed_cases.append(
            {
                "case_id": case_id,
                "prep_sha256": canonical_sha256(case),
                "source_sha256": case["source_sha256"],
                "derived_sha256": case["derived_sha256"],
                "mask_sha256": case["mask_sha256"],
                "label_boxes_sha256": box_hash,
            }
        )
    prep: dict[str, object] = {
        "schema_version": 1,
        "status": "prepared",
        "active_intake_sha256": "e" * 64,
        "cases": cases,
    }
    review: dict[str, object] = {
        "schema_version": 1,
        "status": "approved",
        "reviewed_by": "Raj",
        "confirmation_statement": (
            "I reviewed every source geometry, package mask, and confirmed label box."
        ),
        "reviewed_at": "2026-10-10T12:00:00-07:00",
        "case_ids": list(runner.CASE_IDS),
        "cases": reviewed_cases,
    }
    return prep, review


def test_canonical_sha256_uses_compact_sorted_unicode_json() -> None:
    from product_scene_editor.real_photo_runner import canonical_sha256

    assert canonical_sha256({"z": "é", "a": 1}) == canonical_sha256({"a": 1, "z": "é"})


def test_review_must_bind_every_case_preparation_and_annotation_hash() -> None:
    from product_scene_editor.real_photo_runner import validate_source_geometry_review

    prep, review = _valid_prep_and_review()
    validate_source_geometry_review(prep, review)
    changed = dict(review)
    changed["cases"] = list(review["cases"])
    changed_case = dict(changed["cases"][0])
    changed_case["mask_sha256"] = "f" * 64
    changed["cases"][0] = changed_case
    try:
        validate_source_geometry_review(prep, changed)
    except ValueError:
        pass
    else:
        raise AssertionError("review accepted a mask hash that differs from preparation")


def test_source_geometry_review_requires_human_reply_and_timezone() -> None:
    from product_scene_editor.real_photo_runner import validate_source_geometry_review

    prep, review = _valid_prep_and_review()
    review["confirmation_statement"] = ""
    try:
        validate_source_geometry_review(prep, review)
    except ValueError:
        pass
    else:
        raise AssertionError("review accepted an empty human confirmation")
    prep, review = _valid_prep_and_review()
    review["reviewed_at"] = "2026-10-10T12:00:00"
    try:
        validate_source_geometry_review(prep, review)
    except ValueError:
        pass
    else:
        raise AssertionError("review accepted a timezone-free timestamp")


def test_fixed_generation_contract_is_exact_and_inference_only() -> None:
    assert runner.PROMPT == (
        "Replace only the background with a light stone counter and a softly blurred kitchen. "
        "Keep the visible package, text, logos, shape, size, orientation and position unchanged. "
        "Do not put any object in front of the package."
    )
    assert runner.GENERATION_SETTINGS == {
        "seed": 7321,
        "steps": 16,
        "cfg_scale": 4,
        "width": 512,
        "height": 512,
    }
    assert (
        runner.generation_schedule()[-1]["model_session"]
        != runner.generation_schedule()[3]["model_session"]
    )


def test_adapter_bytes_are_checked_against_explicit_hash_and_size() -> None:
    import hashlib

    from product_scene_editor.real_photo_runner import validate_adapter_bytes

    adapter = b"small test adapter"
    digest = hashlib.sha256(adapter).hexdigest()
    assert (
        validate_adapter_bytes(adapter, expected_digest=digest, expected_bytes=len(adapter))
        == digest
    )
    try:
        validate_adapter_bytes(adapter, expected_digest="0" * 64, expected_bytes=len(adapter))
    except ValueError:
        pass
    else:
        raise AssertionError("adapter bytes with the wrong hash were accepted")


def test_preparation_must_match_active_dev_intake_and_aggregate_digest() -> None:
    from product_scene_editor.real_photo_runner import (
        canonical_sha256,
        validate_preparation_and_intake,
    )

    photos = []
    cases = []
    for case_id in runner.CASE_IDS:
        digest = case_id.encode("utf-8").hex()
        photo = {
            "case_id": case_id,
            "split": "dev",
            "sha256": digest,
            "source_path": f"dev/source/{case_id}.jpg",
            "product_id": case_id,
            "family_id": f"family-{case_id}",
            "rights_review": "pending",
        }
        photos.append(photo)
        cases.append(
            {
                "case_id": case_id,
                "split": "dev",
                "source": {"sha256": digest, "path": f"dev/source/{case_id}.jpg"},
                "product_id": case_id,
                "family_id": f"family-{case_id}",
                "role": "development_only",
                "label_boxes": [
                    {
                        "text": case_id,
                        "original_box": [0, 0, 2, 2],
                        "transformed_box": [1.0, 1.0, 2.0, 2.0],
                    }
                ],
                "label_boxes_review_status": "unreviewed",
            }
        )
    intake: dict[str, object] = {"schema_version": 1, "photos": photos}
    preparation: dict[str, object] = {
        "schema_version": 1,
        "input_intake": {"active_sha256": canonical_sha256(intake)},
        "cases": cases,
    }
    preparation["aggregate_sha256"] = canonical_sha256(preparation)
    validate_preparation_and_intake(preparation, intake)
    photos[0]["split"] = "train"
    with_error = dict(preparation)
    with_error["aggregate_sha256"] = canonical_sha256(
        {key: value for key, value in with_error.items() if key != "aggregate_sha256"}
    )
    try:
        validate_preparation_and_intake(with_error, intake)
    except ValueError:
        pass
    else:
        raise AssertionError("runner accepted a selected case still marked train")


def test_prepared_input_is_deterministic_rgb_with_fixed_white_padding() -> None:
    from io import BytesIO

    from PIL import Image

    source = BytesIO()
    Image.new("RGB", (4, 2), (20, 40, 60)).save(source, format="PNG")
    first, details = runner.prepare_rgb_input(source.getvalue())
    second, repeated_details = runner.prepare_rgb_input(source.getvalue())
    assert first == second
    assert details == repeated_details
    assert details == {
        "source_width": 4,
        "source_height": 2,
        "resized_width": 512,
        "resized_height": 256,
        "pad_left": 0,
        "pad_top": 128,
        "pad_right": 0,
        "pad_bottom": 128,
    }
    with Image.open(BytesIO(first)) as prepared:
        assert prepared.mode == "RGB"
        assert prepared.size == (512, 512)
        assert prepared.getpixel((0, 0)) == (255, 255, 255)
        assert prepared.getpixel((255, 128)) == (20, 40, 60)


def test_payload_archive_rejects_wrong_settings_hashes_and_extra_members() -> None:
    import hashlib
    from io import BytesIO

    from PIL import Image

    from product_scene_editor import remote
    from product_scene_editor.real_photo_runner import (
        canonical_archive,
        canonical_json_bytes,
        canonical_sha256,
        validate_payload_bytes,
    )

    image_stream = BytesIO()
    Image.new("RGB", (512, 512), (1, 2, 3)).save(
        image_stream, format="PNG", compress_level=9, optimize=False
    )
    image = image_stream.getvalue()
    adapter = b"fixture adapter bytes"
    cases = []
    files: dict[str, bytes] = {runner.ADAPTER_MEMBER_NAME: adapter}
    for case_id in runner.CASE_IDS:
        source_path = f"photos/{case_id}/source.bin"
        input_path = f"photos/{case_id}/input.png"
        mask_path = f"photos/{case_id}/mask.png"
        source = b"private source fixture " + case_id.encode()
        mask_stream = BytesIO()
        mask_image = Image.new("L", (512, 512), 0)
        mask_image.paste(255, (100, 100, 400, 400))
        mask_image.save(mask_stream, format="PNG")
        mask = mask_stream.getvalue()
        files.update({source_path: source, input_path: image, mask_path: mask})
        cases.append(
            {
                "case_id": case_id,
                "split": "dev",
                "source_path": source_path,
                "source_sha256": hashlib.sha256(source).hexdigest(),
                "input_path": input_path,
                "input_sha256": hashlib.sha256(image).hexdigest(),
                "mask_path": mask_path,
                "mask_sha256": hashlib.sha256(mask).hexdigest(),
                "label_boxes": [
                    {
                        "text": "x",
                        "original_box": [1, 2, 3, 4],
                        "transformed_box": [1.0, 2.0, 3.0, 4.0],
                    }
                ],
                "label_boxes_sha256": canonical_sha256(
                    [
                        {
                            "text": "x",
                            "original_box": [1, 2, 3, 4],
                            "transformed_box": [1.0, 2.0, 3.0, 4.0],
                        }
                    ]
                ),
            }
        )
    manifest = {
        "schema_version": 1,
        "run_id": "fixture",
        "case_ids": list(runner.CASE_IDS),
        "cases": cases,
        "prompt": runner.PROMPT,
        "generation_settings": runner.GENERATION_SETTINGS,
        "adapter_sha256": hashlib.sha256(adapter).hexdigest(),
        "training_function_called": False,
        "optimizer_updates": 0,
        "preparation_sha256": "a" * 64,
        "geometry_review_sha256": None,
        "geometry_review_status": "not_present",
        "active_intake_sha256": "b" * 64,
        "label_confirmation_sha256": "c" * 64,
        "private_source_review_sha256": "d" * 64,
        "quality_review_status": "pending_human_geometry_review",
        "adapter_archive_sha256": runner.ADAPTER_ARCHIVE_SHA256,
        "adapter_member_sha256": runner.ADAPTER_MEMBER_SHA256,
        "modal_sdk": "1.6.1",
        "diffsynth_commit": remote.DIFFSYNTH_COMMIT,
        "runtime_pins": remote.RUNTIME_PINS,
        "model_components": remote.MODEL_COMPONENTS,
    }
    files["manifest.json"] = canonical_json_bytes(manifest)
    valid = canonical_archive(files)
    parsed_manifest, _ = validate_payload_bytes(
        valid,
        "fixture",
        expected_adapter_digest=hashlib.sha256(adapter).hexdigest(),
        expected_adapter_bytes=len(adapter),
    )
    assert parsed_manifest["case_ids"] == list(runner.CASE_IDS)
    bad_manifest = dict(manifest)
    bad_manifest["generation_settings"] = dict(runner.GENERATION_SETTINGS, seed=5)
    bad_files = dict(files)
    bad_files["manifest.json"] = canonical_json_bytes(bad_manifest)
    try:
        validate_payload_bytes(
            canonical_archive(bad_files),
            "fixture",
            expected_adapter_digest=hashlib.sha256(adapter).hexdigest(),
            expected_adapter_bytes=len(adapter),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("payload with changed inference settings was accepted")
    bad_files = dict(files)
    bad_files["extra.txt"] = b"unexpected"
    try:
        validate_payload_bytes(
            canonical_archive(bad_files),
            "fixture",
            expected_adapter_digest=hashlib.sha256(adapter).hexdigest(),
            expected_adapter_bytes=len(adapter),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("payload with an unreferenced member was accepted")


def test_result_bundle_binds_seven_outputs_and_exact_replay_pixels() -> None:
    import hashlib
    from io import BytesIO

    from PIL import Image

    from product_scene_editor import remote
    from product_scene_editor.real_photo_runner import (
        canonical_archive,
        canonical_json_bytes,
        validate_result_bundle,
    )

    output_files: dict[str, bytes] = {}
    output_records = []
    schedule = runner.generation_schedule()
    for task in schedule:
        path = f"outputs/{task['case_id']}/{task['mode']}.png"
        stream = BytesIO()
        Image.new("RGB", (512, 512), (10, 20, 30)).save(
            stream, format="PNG", compress_level=9, optimize=False
        )
        output = stream.getvalue()
        output_files[path] = output
        output_records.append(
            {
                "path": path,
                "sha256": hashlib.sha256(output).hexdigest(),
                "width": 512,
                "height": 512,
            }
        )
    strict_evidence = {
        "observed": {"pipe_training": False, "lora_dropout_training": []},
        "mode_changed": False,
        "adapter_validation": {
            "validated": True,
            "restored_tensor_count": 200,
            "module_count": 100,
        },
    }
    receipt = {
        "schema_version": 1,
        "run_id": "fixture",
        "status": "passed",
        "generation_settings": runner.GENERATION_SETTINGS,
        "schedule": list(schedule),
        "training_function_called": False,
        "optimizer_updates": 0,
        "output_artifacts": output_records,
        "replay_pixel_equal": True,
        "quality_claim": False,
        "worker_failure_confirmed": False,
        "payload_sha256": "1" * 64,
        "preparation_sha256": "2" * 64,
        "geometry_review_sha256": None,
        "geometry_review_status": "not_present",
        "active_intake_sha256": "3" * 64,
        "label_confirmation_sha256": "4" * 64,
        "private_source_review_sha256": "5" * 64,
        "quality_review_status": "pending_human_geometry_review",
        "adapter_archive_sha256": runner.ADAPTER_ARCHIVE_SHA256,
        "adapter_member_sha256": runner.ADAPTER_MEMBER_SHA256,
        "modal_sdk": "1.6.1",
        "diffsynth_commit": remote.DIFFSYNTH_COMMIT,
        "observed_diffsynth_commit": remote.DIFFSYNTH_COMMIT,
        "runtime_pins": remote.RUNTIME_PINS,
        "runtime_versions": {"modal": "1.6.1", **remote.RUNTIME_PINS},
        "model_components": remote.MODEL_COMPONENTS,
        "strict_load_validations": [strict_evidence, strict_evidence],
        "elapsed_seconds": 1.25,
        "peak_allocated_bytes": 123,
        "failure": None,
    }
    archive = canonical_archive({**output_files, "receipt.json": canonical_json_bytes(receipt)})
    expected_plan = {
        key: receipt[key]
        for key in (
            "run_id",
            "payload_sha256",
            "preparation_sha256",
            "geometry_review_sha256",
            "geometry_review_status",
            "active_intake_sha256",
            "label_confirmation_sha256",
            "private_source_review_sha256",
            "quality_review_status",
            "adapter_archive_sha256",
            "adapter_member_sha256",
            "modal_sdk",
            "diffsynth_commit",
            "runtime_pins",
            "model_components",
        )
    }
    checked_receipt, checked_outputs = validate_result_bundle(
        archive,
        "fixture",
        expected_plan=expected_plan,
        expected_payload_sha256=receipt["payload_sha256"],
    )
    assert checked_receipt["status"] == "passed"
    assert len(checked_outputs) == 7
    receipt["output_artifacts"][0]["sha256"] = "0" * 64
    invalid = canonical_archive({**output_files, "receipt.json": canonical_json_bytes(receipt)})
    try:
        validate_result_bundle(
            invalid,
            "fixture",
            expected_plan=expected_plan,
            expected_payload_sha256=receipt["payload_sha256"],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("result with a mismatched output digest was accepted")

    for field, value in (
        ("elapsed_seconds", float("nan")),
        ("elapsed_seconds", True),
        ("peak_allocated_bytes", 0),
    ):
        invalid_receipt = dict(receipt)
        invalid_receipt[field] = value
        invalid = canonical_archive(
            {**output_files, "receipt.json": canonical_json_bytes(invalid_receipt)}
        )
        try:
            validate_result_bundle(
                invalid,
                "fixture",
                expected_plan=expected_plan,
                expected_payload_sha256=receipt["payload_sha256"],
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"result with invalid {field} evidence was accepted")

    invalid_receipt = dict(receipt)
    invalid_receipt["preparation_sha256"] = "9" * 64
    invalid = canonical_archive(
        {**output_files, "receipt.json": canonical_json_bytes(invalid_receipt)}
    )
    try:
        validate_result_bundle(
            invalid,
            "fixture",
            expected_plan=expected_plan,
            expected_payload_sha256=receipt["payload_sha256"],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("result with a mismatched immutable plan binding was accepted")


def _write_pending_plan_fixture(root, monkeypatch):
    import hashlib
    from io import BytesIO
    from pathlib import Path

    from PIL import Image

    from product_scene_editor import real_photo_runner as photo_runner

    root = Path(root)
    (root / ".context").mkdir(parents=True)
    (root / "artifacts/plans").mkdir(parents=True)
    words_by_case = {
        "OFF-R1-003": ["Barilla", "SPAGHETTI"],
        "OFF-R1-005": ["kinder", "bueno"],
        "OFF-R1-006": ["QUAKER", "Cruesli", "MÉLANGE DE NOIX"],
    }
    collection_digest = "e" * 64
    source_rows = []
    preparation_cases = []
    source_assets = {}
    for index, case_id in enumerate(photo_runner.CASE_IDS):
        source_path = f"dev/source/{case_id}.jpg"
        derived_path = f"dev/prepared/images/{case_id}.png"
        mask_path = f"dev/prepared/masks/{case_id}.png"
        source = f"source bytes {case_id}".encode()
        derived_stream = BytesIO()
        Image.new("RGB", (512, 512), (index, 2, 3)).save(derived_stream, format="PNG")
        mask_stream = BytesIO()
        mask = Image.new("L", (512, 512), 0)
        mask.paste(255, (20, 20, 400, 400))
        mask.save(mask_stream, format="PNG")
        derived = derived_stream.getvalue()
        mask_bytes = mask_stream.getvalue()
        source_assets.update({source_path: source, derived_path: derived, mask_path: mask_bytes})
        boxes = [
            {"text": word, "original_box": [1, 2, 3, 4], "transformed_box": [1, 2, 3, 4]}
            for word in words_by_case[case_id]
        ]
        source_digest = hashlib.sha256(source).hexdigest()
        derived_digest = hashlib.sha256(derived).hexdigest()
        mask_digest = hashlib.sha256(mask_bytes).hexdigest()
        photo = {
            "case_id": case_id,
            "split": "dev",
            "sha256": source_digest,
            "source_path": source_path,
            "product_id": f"product-{index}",
            "family_id": f"family-{index}",
            "rights_review": "pending",
        }
        source_rows.append(photo)
        preparation_cases.append(
            {
                "case_id": case_id,
                "split": "dev",
                "product_id": photo["product_id"],
                "family_id": photo["family_id"],
                "role": "development_only",
                "source": {"path": source_path, "sha256": source_digest},
                "derived": {"path": derived_path, "sha256": derived_digest},
                "mask": {
                    "path": mask_path,
                    "sha256": mask_digest,
                    "human_approved": False,
                },
                "label_boxes": boxes,
                "label_boxes_sha256": photo_runner.canonical_sha256(boxes),
                "label_boxes_review_status": "unreviewed",
            }
        )
    intake = {"schema_version": 1, "photos": source_rows}
    intake_bytes = photo_runner.canonical_json_bytes(intake) + b"\n"
    intake_path = root / "data/real-photos-off-r1/intake.json"
    intake_path.parent.mkdir(parents=True)
    intake_path.write_bytes(intake_bytes)
    label_confirmation = {
        "status": "confirmed_visible_word_spellings_only",
        "confirmed_by": "Raj",
        "confirmation_statement_exact": "Yes, the words match",
        "source_intake_sha256": collection_digest,
        "annotations": [
            {
                "case_id": case_id,
                "confirmed_words": words_by_case[case_id],
                "word_boxes_reviewed": False,
                "mask_created_or_reviewed": False,
            }
            for case_id in photo_runner.CASE_IDS
        ],
    }
    label_bytes = photo_runner.canonical_json_bytes(label_confirmation) + b"\n"
    (root / ".context/real-photo-label-confirmation.json").write_bytes(label_bytes)
    preparation = {
        "schema_version": 1,
        "input_intake": {
            "active_sha256": hashlib.sha256(intake_bytes).hexdigest(),
            "collection_original_sha256": collection_digest,
        },
        "label_confirmation": {"sha256": hashlib.sha256(label_bytes).hexdigest()},
        "cases": preparation_cases,
    }
    preparation["aggregate_sha256"] = photo_runner.canonical_sha256(preparation)
    prep_bytes = photo_runner.canonical_json_bytes(preparation) + b"\n"
    (root / ".context/real-photo-preparation.json").write_bytes(prep_bytes)
    geometry_review = {
        "schema_version": 1,
        "status": "awaiting_human_review",
        "preparation_manifest_aggregate_sha256": preparation["aggregate_sha256"],
        "cases": [
            {
                "case_id": case["case_id"],
                "source_sha256": case["source"]["sha256"],
                "derived_sha256": case["derived"]["sha256"],
                "mask_sha256": case["mask"]["sha256"],
                "label_boxes_sha256": case["label_boxes_sha256"],
                "mask_review": "pending",
                "label_box_review": "pending",
            }
            for case in preparation_cases
        ],
    }
    geometry_bytes = photo_runner.canonical_json_bytes(geometry_review) + b"\n"
    (root / ".context/real-photo-source-geometry-review.json").write_bytes(geometry_bytes)
    (root / ".context/real-photo-private-inference-source-review.json").write_bytes(
        photo_runner.canonical_json_bytes(
            {
                "approved_for_private_inference": True,
                "human_rights_review": False,
                "legal_clearance_claim": False,
                "public_redistribution_clearance": False,
                "training_clearance": False,
                "compute_clearance": False,
                "retain_attribution_and_keep_all_images_private": True,
                "case_ids": list(photo_runner.CASE_IDS),
            }
        )
    )
    (root / ".context/real-photo-development-exclusions.json").write_bytes(
        photo_runner.canonical_json_bytes(
            {
                "schema_version": 1,
                "status": "frozen_development_only",
                "source_intake_sha256": hashlib.sha256(intake_bytes).hexdigest(),
                "groups": [
                    {
                        "case_id": case["case_id"],
                        "development_role": "dev",
                        "excluded_from": ["future_train", "final_test"],
                        "product_id": case["product_id"],
                        "family_id": case["family_id"],
                    }
                    for case in source_rows
                ],
            }
        )
    )
    for relative, content in source_assets.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    adapter = b"fixture adapter"
    adapter_digest = hashlib.sha256(adapter).hexdigest()
    original_validator = photo_runner.validate_payload_bytes
    monkeypatch.setattr(photo_runner, "_archive_adapter", lambda _path: adapter)
    monkeypatch.setattr(
        photo_runner,
        "validate_payload_bytes",
        lambda data, run_id: original_validator(
            data,
            run_id,
            expected_adapter_digest=adapter_digest,
            expected_adapter_bytes=len(adapter),
        ),
    )
    return intake_bytes, geometry_bytes


def test_plan_keeps_pending_geometry_review_pending_and_bound(tmp_path, monkeypatch) -> None:
    import hashlib
    import json
    import zipfile
    from io import BytesIO

    from product_scene_editor import real_photo_runner as photo_runner

    intake_bytes, geometry_bytes = _write_pending_plan_fixture(tmp_path, monkeypatch)
    plan = photo_runner.create_real_photo_plan(
        tmp_path,
        tmp_path / "artifacts/plans/pending-plan",
        run_id="pending-plan",
        adapter_archive=tmp_path / "unused-adapter.zip",
    )
    payload_path = tmp_path / "artifacts/plans/pending-plan" / photo_runner.PAYLOAD_NAME
    with zipfile.ZipFile(BytesIO(payload_path.read_bytes())) as archive:
        manifest = json.loads(archive.read("manifest.json"))

    expected_review_digest = hashlib.sha256(geometry_bytes).hexdigest()
    assert hashlib.sha256(intake_bytes).hexdigest() == manifest["active_intake_sha256"]
    assert manifest["geometry_review_sha256"] == expected_review_digest
    assert manifest["quality_review_status"] == "pending_human_geometry_review"
    assert plan["geometry_review_sha256"] == expected_review_digest
    assert plan["quality_review_status"] == "pending_human_geometry_review"


def test_worker_receipt_binds_all_seven_generations_and_two_strict_loads(monkeypatch) -> None:
    import hashlib
    import sys
    from contextlib import nullcontext
    from io import BytesIO
    from pathlib import Path
    from types import SimpleNamespace

    from PIL import Image

    from product_scene_editor import real_photo_inference as inference
    from product_scene_editor import real_photo_runner as photo_runner
    from product_scene_editor import remote

    image_files = {}
    cases = []
    for index, case_id in enumerate(photo_runner.CASE_IDS):
        stream = BytesIO()
        Image.new("RGB", (512, 512), (index + 1, 2, 3)).save(stream, format="PNG")
        path = f"photos/{case_id}/input.png"
        image_files[path] = stream.getvalue()
        cases.append({"case_id": case_id, "input_path": path})
    image_files[photo_runner.ADAPTER_MEMBER_NAME] = b"fixture-adapter"
    manifest = {
        "run_id": "worker-fixture",
        "preparation_sha256": "a" * 64,
        "geometry_review_sha256": "b" * 64,
        "geometry_review_status": "awaiting_human_review",
        "active_intake_sha256": "c" * 64,
        "label_confirmation_sha256": "d" * 64,
        "private_source_review_sha256": "e" * 64,
        "quality_review_status": "pending_human_geometry_review",
        "adapter_archive_sha256": photo_runner.ADAPTER_ARCHIVE_SHA256,
        "adapter_member_sha256": photo_runner.ADAPTER_MEMBER_SHA256,
        "modal_sdk": "1.6.1",
        "diffsynth_commit": remote.DIFFSYNTH_COMMIT,
        "runtime_pins": remote.RUNTIME_PINS,
        "model_components": remote.MODEL_COMPONENTS,
        "cases": cases,
    }
    payload = b"immutable fixture payload"
    monkeypatch.setattr(
        photo_runner, "validate_payload_bytes", lambda *_args: (manifest, image_files)
    )
    observed = {"generation_count": 0, "peak_queries": 0, "loads": []}

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def manual_seed_all(_seed):
            return None

        @staticmethod
        def reset_peak_memory_stats():
            return None

        @staticmethod
        def empty_cache():
            return None

        @staticmethod
        def max_memory_allocated():
            observed["peak_queries"] += 1
            assert observed["generation_count"] == 7, "peak was read before the complete schedule"
            return 987654

    fake_torch = SimpleNamespace(
        cuda=FakeCuda(),
        manual_seed=lambda _seed: None,
        inference_mode=nullcontext,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=None))
    monkeypatch.setattr(
        inference, "_observed_diffsynth_commit", lambda: remote.DIFFSYNTH_COMMIT, raising=False
    )
    versions = {**remote.RUNTIME_PINS, "modal": "1.6.1"}
    monkeypatch.setattr(
        "importlib.metadata.version", lambda package: versions.get(package, "unknown")
    )
    monkeypatch.setattr(
        remote, "_download_model_snapshots", lambda _download: {"fixture": Path(".")}
    )
    monkeypatch.setattr(remote, "_load_pipe", lambda _snapshots, _device: object())

    def generate(_pipe, _prompt, image, *, seed):
        assert seed == 7321
        observed["generation_count"] += 1
        return image

    monkeypatch.setattr(remote, "_generate", generate)
    strict_evidence = {
        "observed": {"pipe_training": False, "lora_dropout_training": []},
        "mode_changed": False,
        "adapter_validation": {
            "validated": True,
            "restored_tensor_count": 200,
            "module_count": 100,
        },
    }

    def load_strict(_snapshots, _adapter, _torch):
        observed["loads"].append(strict_evidence)
        return object(), object(), strict_evidence

    monkeypatch.setattr("product_scene_editor.reload_diagnostic._load_replay_model", load_strict)
    monkeypatch.setattr(
        "product_scene_editor.reload_diagnostic.validate_adapter_header", lambda _adapter: None
    )
    result_bytes = inference.run_remote_real_photo_inference(payload, "worker-fixture")
    expected_plan = {
        "run_id": "worker-fixture",
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "preparation_sha256": manifest["preparation_sha256"],
        "geometry_review_sha256": manifest["geometry_review_sha256"],
        "geometry_review_status": manifest["geometry_review_status"],
        "active_intake_sha256": manifest["active_intake_sha256"],
        "label_confirmation_sha256": manifest["label_confirmation_sha256"],
        "private_source_review_sha256": manifest["private_source_review_sha256"],
        "quality_review_status": manifest["quality_review_status"],
        "adapter_archive_sha256": photo_runner.ADAPTER_ARCHIVE_SHA256,
        "adapter_member_sha256": photo_runner.ADAPTER_MEMBER_SHA256,
        "modal_sdk": "1.6.1",
        "diffsynth_commit": remote.DIFFSYNTH_COMMIT,
        "runtime_pins": remote.RUNTIME_PINS,
        "model_components": remote.MODEL_COMPONENTS,
    }
    receipt, output_paths = photo_runner.validate_result_bundle(
        result_bytes,
        "worker-fixture",
        expected_plan=expected_plan,
        expected_payload_sha256=hashlib.sha256(payload).hexdigest(),
    )

    assert receipt["status"] == "passed", receipt["failure"]
    assert len(output_paths) == 7
    assert observed["generation_count"] == 7
    assert len(observed["loads"]) == 2
    assert observed["peak_queries"] == 1
    assert receipt["peak_allocated_bytes"] == 987654
    assert receipt["payload_sha256"] == hashlib.sha256(payload).hexdigest()
    assert receipt["quality_review_status"] == manifest["quality_review_status"]
    assert receipt["diffsynth_commit"] == remote.DIFFSYNTH_COMMIT
    assert receipt["runtime_versions"] == versions
    assert receipt["strict_load_validations"] == [strict_evidence, strict_evidence]
