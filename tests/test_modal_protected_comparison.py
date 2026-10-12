from __future__ import annotations

import hashlib
import json
import subprocess
import time
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from io import BytesIO
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from PIL import Image

from experiments import modal_protected_comparison as controller
from experiments import protected_comparison_receipts as receipts
from product_scene_editor import protected_comparison as comparison
from product_scene_editor import protected_comparison_inference as worker
from product_scene_editor import remote
from tests.test_protected_comparison_inference import _execute, _fixture


def _bundle(receipt: dict[str, object]) -> bytes:
    stream = BytesIO()
    with ZipFile(stream, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("receipt.json", json.dumps(receipt, sort_keys=True))
    return stream.getvalue()


def _bundle_with_members(receipt: dict[str, object], members: dict[str, bytes]) -> bytes:
    stream = BytesIO()
    with ZipFile(stream, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("receipt.json", json.dumps(receipt, sort_keys=True))
        for name, content in members.items():
            archive.writestr(name, content)
    return stream.getvalue()


def _successful_result(*, mutate: str | None = None) -> tuple[dict, dict, bytes, str]:
    manifest, payload_files = _fixture()
    state, _pipes, _runtime = _execute()
    assert state["status"] == "passed"
    if mutate == "raw_replay":
        replay_path = "replays/case-00/empty_scene_composite.png"
        background_hash = hashlib.sha256(state["outputs"][replay_path]).hexdigest()
        with Image.open(BytesIO(state["outputs"][replay_path])) as opened:
            changed = opened.copy()
        changed.putpixel((80, 80), (255, 0, 0))
        stream = BytesIO()
        changed.save(stream, format="PNG", compress_level=6, optimize=False)
        state["outputs"][replay_path] = stream.getvalue()
        new_hash = hashlib.sha256(stream.getvalue()).hexdigest()
        for record in state["generation_records"]:
            if record["output_path"] == replay_path:
                record["output_sha256"] = new_hash
        for record in state["composed_integrity_records"]:
            if record["output_path"] == "replays/case-00/empty_scene_composite_final.png":
                assert record["background_sha256"] == background_hash
                record["background_sha256"] = new_hash
    receipt = worker._receipt(
        "passed",
        state,
        run_id="comparison-r1",
        started=time.monotonic(),
        stage="passed",
        payload_sha256="c" * 64,
        manifest=manifest,
        runtime_versions=remote.RUNTIME_PINS,
        observed_commit=remote.DIFFSYNTH_COMMIT,
        peak_memory_bytes=100,
        failure=None,
    )
    receipt["pipeline_loads"] = [
        {
            "stage": stage,
            "device": "cuda",
            "component_pins": manifest["model_components"],
            "duration_seconds": 0.1,
            "status": "succeeded",
            "error": None,
        }
        for stage in ("base_load", "fresh_replay_load")
    ]
    if mutate == "schedule":
        receipt["schedule"] = []
    if mutate == "replay":
        receipt["replay_checks"][0]["final_pixel_equal"] = False
    if mutate == "raw_replay":
        receipt["replay_checks"][1]["raw_background_pixel_equal"] = False
    result = worker._pack_result(
        {
            **state["outputs"],
            "receipt.json": json.dumps(receipt, sort_keys=True).encode(),
        }
    )
    return manifest, payload_files, result, "c" * 64


def _failed_receipt() -> tuple[dict, dict, dict[str, object]]:
    manifest, payload_files = _fixture()
    receipt: dict[str, object] = {
        "schema_version": 1,
        "run_id": "comparison-r1",
        "status": "failed",
        "source_sha": manifest["source_sha"],
        "context": manifest["context"],
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "generation_settings": manifest["generation"],
        "schedule": [dict(task) for task in comparison.generation_schedule(manifest)],
        "generation_call_count": 0,
        "generation_call_timings": [],
        "arm_generation_seconds": {},
        "training_function_called": False,
        "optimizer_updates": 0,
        "output_artifacts": [],
        "generation_records": [],
        "replay_checks": [],
        "composed_integrity_records": [],
        "pipeline_loads": [],
        "payload_sha256": "c" * 64,
        "modal_sdk": "1.6.1",
        "diffsynth_commit": remote.DIFFSYNTH_COMMIT,
        "observed_diffsynth_commit": None,
        "runtime_pins": manifest["runtime_pins"],
        "runtime_versions": {},
        "model_components": manifest["model_components"],
        "elapsed_seconds": 0.0,
        "peak_allocated_bytes": None,
        "failure": {
            "stage": "runtime_import",
            "type": "RuntimeError",
            "message": "worker stopped before runtime import",
        },
        "failure_stage": "runtime_import",
    }
    return manifest, payload_files, receipt


def test_build_image_installs_the_exact_pinned_diffsynth_commit(tmp_path) -> None:
    operations = []

    class FakeImage:
        @classmethod
        def from_registry(cls, image):
            operations.append(("from_registry", image))
            return cls()

        def env(self, values):
            operations.append(("env", values))
            return self

        def apt_install(self, *packages):
            operations.append(("apt_install", packages))
            return self

        def pip_install(self, *packages, **kwargs):
            operations.append(("pip_install", packages, kwargs))
            return self

        def run_commands(self, *commands):
            operations.append(("run_commands", commands))
            return self

        def add_local_python_source(self, *packages, **kwargs):
            operations.append(("add_local_python_source", packages, kwargs))
            return self

    worker_source = tmp_path / "src/product_scene_editor/protected_comparison_inference.py"
    worker_source.parent.mkdir(parents=True)
    worker_source.write_text("# test worker source\n")

    image = controller._build_image(SimpleNamespace(Image=FakeImage), tmp_path)

    assert isinstance(image, FakeImage)
    assert operations == [
        ("from_registry", controller.IMAGE),
        ("env", {"USE_HUB_KERNELS": "NO"}),
        ("apt_install", ("git", "build-essential")),
        ("run_commands", ("cc --version && c++ --version && make --version",)),
        (
            "pip_install",
            tuple(
                [
                    f"{name}=={version}"
                    for name, version in remote.RUNTIME_PINS.items()
                    if name != "imageio"
                ]
                + [f"imageio[ffmpeg]=={remote.RUNTIME_PINS['imageio']}"]
            ),
            {"extra_index_url": remote.TORCH_INDEX},
        ),
        (
            "run_commands",
            (
                "git clone --filter=blob:none --no-checkout "
                "https://github.com/modelscope/DiffSynth-Studio.git /opt/DiffSynth-Studio",
                f"git -C /opt/DiffSynth-Studio checkout --detach {remote.DIFFSYNTH_COMMIT} "
                f'&& test "$(git -C /opt/DiffSynth-Studio rev-parse HEAD)" = '
                f"{remote.DIFFSYNTH_COMMIT} "
                "&& pip install --no-deps /opt/DiffSynth-Studio",
            ),
        ),
        ("add_local_python_source", ("product_scene_editor",), {"copy": True}),
    ]


def _write_plan(root, *, gpu: str = "A100-80GB"):
    manifest, files = _fixture()
    payload = comparison.build_payload(manifest, files)
    payload_path = root / "comparison-input.zip"
    payload_path.write_bytes(payload)
    value = {
        "schema_version": 1,
        "source_commit": manifest["source_sha"],
        "run_id": "comparison-r1",
        "payload_path": "comparison-input.zip",
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "output_dir": "artifacts/protected-comparisons/comparison-r1",
        "context": {
            "profile": "reflex-personal",
            "workspace": "rajath-61258",
            "environment": "main",
        },
        "resources": {
            "gpu": gpu,
            "cpu_cores": 4,
            "memory_mib": 65536,
            "startup_timeout_seconds": 300,
            "execution_timeout_seconds": 1800,
            "retries": 0,
        },
        "image": controller.IMAGE,
    }
    plan_path = root / "comparison-plan.json"
    plan_path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return plan_path


def _write_clearance(plan, *, root, checked: datetime) -> tuple[object, dict[str, object]]:
    source_review = {
        "status": "approved",
        "source_commit": plan.value["source_commit"],
        "plan_sha256": plan.plan_sha256,
        "payload_sha256": plan.payload_sha256,
    }
    data_review = {
        "status": "approved",
        "source_commit": plan.value["source_commit"],
        "plan_sha256": plan.plan_sha256,
        "payload_sha256": plan.payload_sha256,
        "mask_count": 24,
        "all_masks_accepted": True,
        "pool_counts": {"development": 6, "reserved": 18},
        "brand_family_disjoint": True,
        "case_inventory": [case["case_id"] for case in plan.manifest["cases"]],
        "individual_review_references": [
            f"review-{case['case_id']}" for case in plan.manifest["cases"]
        ],
    }
    source_bytes = json.dumps(source_review, sort_keys=True).encode()
    data_bytes = json.dumps(data_review, sort_keys=True).encode()
    (root / "source-review.json").write_bytes(source_bytes)
    (root / "data-review.json").write_bytes(data_bytes)
    checked_text = checked.isoformat()
    rates = {
        "a10080_usd_per_second": 0.001,
        "cpu_core_usd_per_second": 0.0001,
        "memory_gib_usd_per_second": 0.00001,
    }
    estimate = Decimal(2100) * (
        Decimal(str(rates["a10080_usd_per_second"]))
        + Decimal(4) * Decimal(str(rates["cpu_core_usd_per_second"]))
        + Decimal(64) * Decimal(str(rates["memory_gib_usd_per_second"]))
    )
    evidence = {
        "checked_utc": checked_text,
        "context": plan.value["context"],
        "active_apps": [],
        "billing": {
            "scope": "workspace-delayed-not-project-invoice",
            "workspace_gross_usd": "0",
            "workspace_net_usd": "0",
        },
        "rates": rates,
        "price_observation": {
            "checked_utc": checked_text,
            "pricing_url": "https://modal.com/pricing",
            "pricing_sha256": "d" * 64,
            "status": "verified",
        },
        "source_pin_review_path": "source-review.json",
        "source_pin_review_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "data_pin_review_path": "data-review.json",
        "data_pin_review_sha256": hashlib.sha256(data_bytes).hexdigest(),
    }
    evidence_bytes = json.dumps(evidence, sort_keys=True).encode()
    (root / "usage-rates.json").write_bytes(evidence_bytes)
    clearance: dict[str, object] = {
        "schema_version": 1,
        "status": "approved",
        "plan_sha256": plan.plan_sha256,
        "source_commit": plan.value["source_commit"],
        "payload_sha256": plan.payload_sha256,
        "run_id": plan.value["run_id"],
        "context": plan.value["context"],
        "ci_status": "success",
        "ci_run_id": 123,
        "ci_source_commit": plan.value["source_commit"],
        "source_signature_verified": True,
        "source_signature_reason": "verified signed source commit",
        "checked_utc": checked_text,
        "usage_rates_evidence_path": "usage-rates.json",
        "usage_rates_evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
        "estimated_seconds": 2100,
        "estimated_compute_usd": str(estimate),
        "mask_step_reserve_usd": 1,
        "allowance_usd": 8,
        "cost_exclusions": ["image_build", "storage", "transfer", "shutdown_lag"],
    }
    path = root / "clearance.json"
    path.write_text(json.dumps(clearance, sort_keys=True), encoding="utf-8")
    return path, clearance


def test_launch_records_one_call_validates_result_and_proves_shutdown(
    tmp_path, monkeypatch
) -> None:
    plan = controller._load_plan(_write_plan(tmp_path), root=tmp_path)
    clearance_path, _clearance = _write_clearance(plan, root=tmp_path, checked=datetime.now(UTC))
    manifest, _payload_files, result_bytes, _placeholder_sha = _successful_result()
    receipt, members = receipts._read_result_bundle(result_bytes)
    receipt["payload_sha256"] = plan.payload_sha256
    result_bytes = worker._pack_result(
        {**members, "receipt.json": json.dumps(receipt, sort_keys=True).encode()}
    )
    assert manifest == plan.manifest

    operations = {"apps": [], "spawns": [], "cli": []}

    class FakeImage:
        @classmethod
        def from_registry(cls, image):
            operations["image"] = image
            return cls()

        def env(self, values):
            operations["image_env"] = values
            return self

        def apt_install(self, *packages):
            operations["apt_packages"] = packages
            return self

        def pip_install(self, *packages, **kwargs):
            operations["pip_packages"] = (packages, kwargs)
            return self

        def run_commands(self, *commands):
            operations["image_commands"] = commands
            return self

        def add_local_python_source(self, *packages, **kwargs):
            operations["worker_source"] = (packages, kwargs)
            return self

    worker_source = tmp_path / "src/product_scene_editor/protected_comparison_inference.py"
    worker_source.parent.mkdir(parents=True)
    worker_source.write_text("# test worker source\n")

    class FakeFunctionCall:
        object_id = "fc-positive"

        def get(self, *, timeout):
            assert (
                timeout == controller.STARTUP_TIMEOUT_SECONDS + controller.EXECUTION_TIMEOUT_SECONDS
            )
            return result_bytes

    class FakeRun:
        def spawn(self, payload, run_id):
            operations["spawns"].append((payload, run_id))
            return FakeFunctionCall()

    class FakeApp:
        app_id = "ap-positive"

        def __init__(self, name, *, image):
            operations["apps"].append((name, image))

        def function(self, **kwargs):
            operations["function_options"] = kwargs
            return lambda _function: FakeRun()

        def run(self, *, environment_name, detach):
            operations["app_run"] = (environment_name, detach)
            return nullcontext()

    fake_modal = SimpleNamespace(
        Image=FakeImage,
        App=FakeApp,
        enable_output=nullcontext,
    )
    monkeypatch.setattr(controller, "_verify_source", lambda _plan: None)
    monkeypatch.setattr(controller.remote, "validate_launch_environment", lambda _context: None)
    monkeypatch.setattr(controller.remote, "verify_modal_identity", lambda **_context: None)
    monkeypatch.setattr(controller.importlib, "import_module", lambda _name: fake_modal)
    monkeypatch.setattr(controller.importlib.metadata, "version", lambda _name: "1.6.1")

    def fake_modal_cli(args, **kwargs):
        operations["cli"].append(args)
        if "stop" in args:
            return subprocess.CompletedProcess(args, 0, "", "")
        assert "list" in args
        return subprocess.CompletedProcess(
            args,
            0,
            json.dumps([{"app_id": "ap-positive", "state": "stopped", "tasks": "0"}]),
            "",
        )

    monkeypatch.setattr(controller.subprocess, "run", fake_modal_cli)

    assert controller._launch(plan, clearance_path) == 0
    marker = controller._read_attempt(plan)
    assert marker["status"] == "stopped"
    assert marker["worker_status"] == "passed"
    assert marker["app_id"] == "ap-positive"
    assert marker["call_id"] == "fc-positive"
    assert marker["spawn_count"] == 1
    assert marker["private_result_sha256"] == hashlib.sha256(result_bytes).hexdigest()
    assert len(operations["spawns"]) == 1
    assert operations["spawns"][0] == (plan.payload, plan.value["run_id"])
    assert operations["apps"] == [(controller.APP_NAME, operations["apps"][0][1])]
    assert operations["app_run"] == (plan.value["context"]["environment"], True)
    assert operations["function_options"]["retries"] == 0
    assert operations["function_options"]["max_containers"] == 1
    assert ["stop" if "stop" in args else "list" for args in operations["cli"]] == [
        "stop",
        "list",
    ]
    assert (plan.output_dir / controller.RESULT_NAME).read_bytes() == result_bytes
    assert not (plan.output_dir / controller.UNVERIFIED_RESULT_NAME).exists()
    app_list_events = [
        json.loads(path.read_text())
        for path in plan.output_dir.glob("protected-comparison-app-list-*.json")
    ]
    assert len(app_list_events) == 1
    assert app_list_events[0]["run_id"] == plan.value["run_id"]
    assert app_list_events[0]["call_id"] == "fc-positive"
    assert app_list_events[0]["source_commit"] == plan.value["source_commit"]
    assert app_list_events[0]["payload_sha256"] == plan.payload_sha256
    assert app_list_events[0]["confirmed_stopped"] is True
    assert app_list_events[0]["matching_apps"] == [
        {"app_id": "ap-positive", "state": "stopped", "tasks": "0"}
    ]

    with pytest.raises(FileExistsError, match="already has attempt or result evidence"):
        controller._launch(plan, clearance_path)
    assert len(operations["spawns"]) == 1


def test_result_rejects_receipt_bound_to_another_payload() -> None:
    manifest, payload_files, receipt = _failed_receipt()
    receipt["payload_sha256"] = "b" * 64

    with pytest.raises(ValueError, match="payload"):
        receipts.validate_result_bundle(
            _bundle(receipt),
            run_id="comparison-r1",
            payload_sha256="c" * 64,
            source_commit=str(manifest["source_sha"]),
            manifest=manifest,
            payload_files=payload_files,
        )


def test_result_rejects_receipt_that_reports_training() -> None:
    manifest, payload_files, receipt = _failed_receipt()
    receipt["training_function_called"] = True

    with pytest.raises(ValueError, match="training"):
        receipts.validate_result_bundle(
            _bundle(receipt),
            run_id="comparison-r1",
            payload_sha256="c" * 64,
            source_commit=str(manifest["source_sha"]),
            manifest=manifest,
            payload_files=payload_files,
        )


def test_result_rejects_runtime_pins_that_differ_from_payload() -> None:
    manifest, payload_files, receipt = _failed_receipt()
    receipt["runtime_pins"] = {}

    with pytest.raises(ValueError, match="runtime pins"):
        receipts.validate_result_bundle(
            _bundle(receipt),
            run_id="comparison-r1",
            payload_sha256="c" * 64,
            source_commit=str(manifest["source_sha"]),
            manifest=manifest,
            payload_files=payload_files,
        )


def test_failed_result_still_rejects_partial_artifact_hash_mismatch() -> None:
    member = b"partial output"
    manifest, payload_files, receipt = _failed_receipt()
    receipt["output_artifacts"] = [
        {
            "path": "outputs/case-00/full_image_edit.png",
            "sha256": "0" * 64,
            "width": 512,
            "height": 512,
        }
    ]

    with pytest.raises(ValueError, match="artifact.*hash"):
        receipts.validate_result_bundle(
            _bundle_with_members(receipt, {"outputs/case-00/full_image_edit.png": member}),
            run_id="comparison-r1",
            payload_sha256="c" * 64,
            source_commit=str(manifest["source_sha"]),
            manifest=manifest,
            payload_files=payload_files,
        )


@pytest.mark.parametrize("mutation", ["artifact_hash", "duration"])
def test_generation_record_must_match_artifact_and_timing(mutation: str) -> None:
    manifest, payload_files, receipt = _failed_receipt()
    task = dict(comparison.generation_schedule(manifest)[0])
    output_path = f"outputs/{task['case_id']}/full_image_edit.png"
    image = Image.new("RGB", (512, 512), (20, 40, 60))
    stream = BytesIO()
    image.save(stream, format="PNG")
    artifact = stream.getvalue()
    digest = hashlib.sha256(artifact).hexdigest()
    receipt["generation_call_count"] = 1
    timing = {key: task[key] for key in ("index", "purpose", "case_id", "arm")}
    timing["generation_seconds"] = 0.25
    record = {
        **timing,
        "seed": task["seed"],
        "output_path": output_path,
        "output_sha256": digest,
    }
    receipt["generation_call_timings"] = [timing]
    receipt["generation_records"] = [record]
    receipt["arm_generation_seconds"] = {f"{task['purpose']}/{task['arm']}": 0.25}
    receipt["output_artifacts"] = [
        {"path": output_path, "sha256": digest, "width": 512, "height": 512}
    ]
    if mutation == "artifact_hash":
        record["output_sha256"] = "0" * 64
    else:
        record["generation_seconds"] = 0.125

    with pytest.raises(ValueError, match="generation record"):
        receipts.validate_result_bundle(
            _bundle_with_members(receipt, {output_path: artifact}),
            run_id="comparison-r1",
            payload_sha256="c" * 64,
            source_commit=str(manifest["source_sha"]),
            manifest=manifest,
            payload_files=payload_files,
        )


def test_receipt_only_result_packaging_failure_keeps_partial_generation_evidence() -> None:
    manifest, payload_files, receipt = _failed_receipt()
    task = dict(comparison.generation_schedule(manifest)[0])
    output_path = f"outputs/{task['case_id']}/full_image_edit.png"
    timing = {key: task[key] for key in ("index", "purpose", "case_id", "arm")}
    timing["generation_seconds"] = 0.25
    receipt["generation_call_count"] = 1
    receipt["generation_call_timings"] = [timing]
    receipt["generation_records"] = [
        {
            **timing,
            "seed": task["seed"],
            "output_path": output_path,
            "output_sha256": "a" * 64,
        }
    ]
    receipt["arm_generation_seconds"] = {f"{task['purpose']}/{task['arm']}": 0.25}
    receipt["failure_stage"] = "result_packaging"
    receipt["failure"] = {
        "stage": "result_packaging",
        "type": "OSError",
        "message": "fixture packaging failure",
    }

    validated = receipts.validate_result_bundle(
        _bundle(receipt),
        run_id="comparison-r1",
        payload_sha256="c" * 64,
        source_commit=str(manifest["source_sha"]),
        manifest=manifest,
        payload_files=payload_files,
    )
    assert validated["failure_stage"] == "result_packaging"


def test_successful_result_requires_frozen_schedule_and_78_artifacts() -> None:
    manifest, payload_files, result, payload_sha = _successful_result(mutate="schedule")

    with pytest.raises(ValueError, match="schedule"):
        receipts.validate_result_bundle(
            result,
            run_id="comparison-r1",
            payload_sha256=payload_sha,
            source_commit=str(manifest["source_sha"]),
            manifest=manifest,
            payload_files=payload_files,
        )


def test_passed_receipt_requires_raw_background_replay_equality() -> None:
    manifest, payload_files, result, payload_sha = _successful_result(mutate="raw_replay")

    with pytest.raises(ValueError, match="replay"):
        receipts.validate_result_bundle(
            result,
            run_id="comparison-r1",
            payload_sha256=payload_sha,
            source_commit=str(manifest["source_sha"]),
            manifest=manifest,
            payload_files=payload_files,
        )


def test_compositor_accepts_png_reencoding_but_rejects_changed_pixels() -> None:
    manifest, payload_files, result, payload_sha = _successful_result()
    with ZipFile(BytesIO(result)) as archive:
        receipt = json.loads(archive.read("receipt.json"))
        members = {
            name: archive.read(name) for name in archive.namelist() if name != "receipt.json"
        }
    composite_path = "outputs/case-00/empty_scene_composite.png"

    def repackage(content: bytes) -> bytes:
        members[composite_path] = content
        digest = hashlib.sha256(content).hexdigest()
        for artifact in receipt["output_artifacts"]:
            if artifact["path"] == composite_path:
                artifact["sha256"] = digest
        for integrity in receipt["composed_integrity_records"]:
            if integrity["output_path"] == composite_path:
                integrity["output_sha256"] = digest
        return _bundle_with_members(receipt, members)

    with Image.open(BytesIO(members[composite_path])) as opened:
        opened.load()
        reencoded = opened.copy()
    stream = BytesIO()
    reencoded.save(stream, format="PNG", compress_level=0)
    encoded_result = repackage(stream.getvalue())
    assert (
        receipts.validate_result_bundle(
            encoded_result,
            run_id="comparison-r1",
            payload_sha256=payload_sha,
            source_commit=str(manifest["source_sha"]),
            manifest=manifest,
            payload_files=payload_files,
        )["status"]
        == "passed"
    )

    with Image.open(BytesIO(members[composite_path])) as opened:
        opened.load()
        changed = opened.copy()
    changed.putpixel((0, 0), (1, 2, 3))
    stream = BytesIO()
    changed.save(stream, format="PNG", compress_level=0)
    changed_result = repackage(stream.getvalue())
    with pytest.raises(ValueError, match="composit|integrity"):
        receipts.validate_result_bundle(
            changed_result,
            run_id="comparison-r1",
            payload_sha256=payload_sha,
            source_commit=str(manifest["source_sha"]),
            manifest=manifest,
            payload_files=payload_files,
        )


def test_observer_lock_excludes_competing_recovery_and_releases(tmp_path) -> None:
    lock_path = tmp_path / "observer.json"

    with controller._observer(lock_path, run_id="comparison-r1", action="launch"):
        with pytest.raises(RuntimeError, match="owns this run"):
            with controller._observer(lock_path, run_id="comparison-r1", action="recover"):
                pytest.fail("the second observer acquired the run")

    with controller._observer(lock_path, run_id="comparison-r1", action="recover"):
        pass


def test_plan_rejects_resource_contract_drift_before_any_operation(tmp_path) -> None:
    plan_path = _write_plan(tmp_path, gpu="A10")

    with pytest.raises(ValueError, match="resource contract"):
        controller._load_plan(plan_path, root=tmp_path)


def test_clearance_rejects_stale_usage_and_price_evidence(tmp_path) -> None:
    plan_path = _write_plan(tmp_path)
    plan = controller._load_plan(plan_path, root=tmp_path)
    now = datetime.now(UTC)
    clearance_path, _clearance = _write_clearance(
        plan, root=tmp_path, checked=now - timedelta(minutes=31)
    )

    with pytest.raises(ValueError, match="stale"):
        controller._validate_clearance(plan, clearance_path, now=now)


def test_fresh_clearance_recomputes_cost_and_accepts_approved_reviews(tmp_path) -> None:
    plan = controller._load_plan(_write_plan(tmp_path), root=tmp_path)
    now = datetime.now(UTC)
    clearance_path, _clearance = _write_clearance(plan, root=tmp_path, checked=now)

    validated = controller._validate_clearance(plan, clearance_path, now=now)

    assert validated["estimated_compute_usd"] == "4.28400"


def test_clearance_rejects_cross_pool_brand_family_data_review(tmp_path) -> None:
    plan = controller._load_plan(_write_plan(tmp_path), root=tmp_path)
    now = datetime.now(UTC)
    clearance_path, clearance = _write_clearance(plan, root=tmp_path, checked=now)
    evidence_path = tmp_path / "usage-rates.json"
    evidence = json.loads(evidence_path.read_text())
    review_path = tmp_path / evidence["data_pin_review_path"]
    review = json.loads(review_path.read_text())
    review["brand_family_disjoint"] = False
    review_bytes = json.dumps(review, sort_keys=True).encode()
    review_path.write_bytes(review_bytes)
    evidence["data_pin_review_sha256"] = hashlib.sha256(review_bytes).hexdigest()
    evidence_bytes = json.dumps(evidence, sort_keys=True).encode()
    evidence_path.write_bytes(evidence_bytes)
    clearance["usage_rates_evidence_sha256"] = hashlib.sha256(evidence_bytes).hexdigest()
    clearance_path.write_text(json.dumps(clearance, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen data split"):
        controller._validate_clearance(plan, clearance_path, now=now)


def _seed_attempt(plan, *, call_id: str | None) -> None:
    path = controller._reserve_attempt(plan)
    controller._update_attempt(path, status="app_started", app_id="ap-test")
    controller._update_attempt(path, status="spawn_reserved", spawn_count=1)
    updates = {"status": "observing"}
    if call_id is not None:
        updates["call_id"] = call_id
    controller._update_attempt(path, **updates)


def _modal_cli_stopped(app_id: str):
    def run(args, **kwargs):
        if "stop" in args:
            return subprocess.CompletedProcess(args, 0, "stopped", "")
        if "list" in args:
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps([{"app_id": app_id, "state": "stopped", "tasks": "0"}]),
                "",
            )
        raise AssertionError(f"unexpected Modal command: {args}")

    return run


def test_recovery_stops_recorded_app_when_reserved_call_id_is_missing(
    tmp_path, monkeypatch
) -> None:
    plan = controller._load_plan(_write_plan(tmp_path), root=tmp_path)
    _seed_attempt(plan, call_id=None)
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return _modal_cli_stopped("ap-test")(args, **kwargs)

    monkeypatch.setattr(controller.subprocess, "run", run)

    assert controller._recover(plan) == 1
    marker = controller._read_attempt(plan)
    assert marker["status"] == "stopped"
    assert marker["call_id"] is None
    assert len(calls) == 2
    assert "stop" in calls[0] and "list" in calls[1]
    app_list_events = [
        json.loads(path.read_text())
        for path in plan.output_dir.glob("protected-comparison-app-list-*.json")
    ]
    assert len(app_list_events) == 1
    assert app_list_events[0]["returncode"] == 0
    assert app_list_events[0]["observed_at_utc"]
    assert app_list_events[0]["matching_apps"] == [
        {"app_id": "ap-test", "state": "stopped", "tasks": "0"}
    ]
    assert app_list_events[0]["call_id"] is None
    assert app_list_events[0]["source_commit"] == plan.value["source_commit"]
    assert app_list_events[0]["payload_sha256"] == plan.payload_sha256


def test_recovery_validates_private_saved_result_before_modal_lookup(tmp_path, monkeypatch) -> None:
    plan = controller._load_plan(_write_plan(tmp_path), root=tmp_path)
    _seed_attempt(plan, call_id=None)
    manifest, _files, receipt = _failed_receipt()
    receipt["payload_sha256"] = plan.payload_sha256
    private = plan.output_dir / controller.UNVERIFIED_RESULT_NAME
    private.write_bytes(_bundle(receipt))
    monkeypatch.setattr(controller.subprocess, "run", _modal_cli_stopped("ap-test"))
    monkeypatch.setattr(
        controller.importlib,
        "import_module",
        lambda _name: pytest.fail("saved result must be read before importing Modal"),
    )

    assert controller._recover(plan) == 1
    assert (plan.output_dir / controller.RESULT_NAME).is_file()
    assert not private.exists()
    marker = controller._read_attempt(plan)
    assert marker["status"] == "stopped"
    assert marker["worker_status"] == "failed"
    assert (
        marker["private_result_sha256"]
        == hashlib.sha256((plan.output_dir / controller.RESULT_NAME).read_bytes()).hexdigest()
    )


def test_recovery_preserves_stopped_marker_when_cached_result_is_already_final(
    tmp_path, monkeypatch
) -> None:
    plan = controller._load_plan(_write_plan(tmp_path), root=tmp_path)
    _seed_attempt(plan, call_id="fc-test")
    attempt_path = controller._attempt_path(plan)
    controller._update_attempt(
        attempt_path,
        status="stopped",
        worker_status="passed",
        stopped_at_utc="2026-10-10T00:00:00+00:00",
    )
    (plan.output_dir / controller.UNVERIFIED_RESULT_NAME).write_bytes(b"cached")
    monkeypatch.setattr(
        controller.receipts,
        "validate_result_bundle",
        lambda *_args, **_kwargs: {"status": "passed"},
    )
    monkeypatch.setattr(
        controller.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("already stopped")
    )

    assert controller._recover(plan) == 0
    marker = controller._read_attempt(plan)
    assert marker["status"] == "stopped"
    assert marker["worker_status"] == "passed"
    assert marker["stopped_at_utc"] == "2026-10-10T00:00:00+00:00"
    assert (plan.output_dir / controller.RESULT_NAME).read_bytes() == b"cached"
    assert marker["private_result_sha256"] == hashlib.sha256(b"cached").hexdigest()


def test_recovery_rejects_symlinked_private_result(tmp_path, monkeypatch) -> None:
    plan = controller._load_plan(_write_plan(tmp_path), root=tmp_path)
    _seed_attempt(plan, call_id="fc-test")
    controller._update_attempt(
        controller._attempt_path(plan), status="stopped", worker_status="failed"
    )
    target = tmp_path / "outside.zip"
    target.write_bytes(b"not trusted")
    (plan.output_dir / controller.UNVERIFIED_RESULT_NAME).symlink_to(target)
    validations = []
    monkeypatch.setattr(
        controller.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("already stopped")
    )
    monkeypatch.setattr(
        controller.receipts,
        "validate_result_bundle",
        lambda *_args, **_kwargs: validations.append(True) or {"status": "passed"},
    )

    assert controller._recover(plan) == 1
    assert not (plan.output_dir / controller.RESULT_NAME).exists()
    assert controller._read_attempt(plan)["status"] == "stopped"
    assert validations == []


def test_unknown_same_call_recovery_error_stays_pending_without_stop(tmp_path, monkeypatch) -> None:
    plan = controller._load_plan(_write_plan(tmp_path), root=tmp_path)
    _seed_attempt(plan, call_id="fc-test")

    class PendingCall:
        def get(self, *, timeout):
            assert timeout == 0
            raise RuntimeError("temporary observation API failure")

    fake_modal = SimpleNamespace(
        FunctionCall=SimpleNamespace(from_id=lambda _call_id: PendingCall()),
        exception=SimpleNamespace(),
    )
    monkeypatch.setattr(controller.remote, "verify_modal_identity", lambda **_context: {})
    monkeypatch.setattr(controller.importlib, "import_module", lambda _name: fake_modal)
    monkeypatch.setattr(controller.importlib.metadata, "version", lambda _name: "1.6.1")
    monkeypatch.setattr(
        controller.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("must not stop pending call"),
    )

    assert controller._recover(plan) == 1
    marker = controller._read_attempt(plan)
    assert marker["status"] == "observation_pending"
    assert marker["call_id"] == "fc-test"
    assert marker["spawn_count"] == 1


def test_terminal_same_call_recovery_error_stops_the_recorded_app(tmp_path, monkeypatch) -> None:
    plan = controller._load_plan(_write_plan(tmp_path), root=tmp_path)
    _seed_attempt(plan, call_id="fc-test")

    class FunctionTimeoutError(Exception):
        pass

    class TerminalCall:
        def get(self, *, timeout):
            assert timeout == 0
            raise FunctionTimeoutError("remote function timed out")

    looked_up = []
    fake_modal = SimpleNamespace(
        FunctionCall=SimpleNamespace(
            from_id=lambda call_id: looked_up.append(call_id) or TerminalCall()
        ),
        exception=SimpleNamespace(FunctionTimeoutError=FunctionTimeoutError),
    )
    monkeypatch.setattr(controller.remote, "verify_modal_identity", lambda **_context: {})
    monkeypatch.setattr(controller.importlib, "import_module", lambda _name: fake_modal)
    monkeypatch.setattr(controller.importlib.metadata, "version", lambda _name: "1.6.1")
    monkeypatch.setattr(controller.subprocess, "run", _modal_cli_stopped("ap-test"))

    assert controller._recover(plan) == 1
    assert looked_up == ["fc-test"]
    marker = controller._read_attempt(plan)
    assert marker["status"] == "stopped"
    assert marker["call_id"] == "fc-test"
    assert marker["spawn_count"] == 1


def test_shutdown_refuses_unconfirmed_running_app_after_four_observations(
    tmp_path, monkeypatch
) -> None:
    plan = controller._load_plan(_write_plan(tmp_path), root=tmp_path)
    _seed_attempt(plan, call_id="fc-test")
    observations = []

    def run(args, **kwargs):
        if "stop" in args:
            return subprocess.CompletedProcess(args, 0, "", "")
        observations.append(args)
        return subprocess.CompletedProcess(
            args, 0, json.dumps([{"app_id": "ap-test", "state": "running", "tasks": "1"}]), ""
        )

    monkeypatch.setattr(controller.subprocess, "run", run)
    monkeypatch.setattr(controller.time, "sleep", lambda _seconds: None)
    with pytest.raises(RuntimeError, match="did not confirm"):
        controller._stop_and_confirm(plan, "ap-test")
    assert len(observations) == controller.MAX_SHUTDOWN_OBSERVATIONS == 4
    assert controller._read_attempt(plan)["status"] == "shutdown_pending"
