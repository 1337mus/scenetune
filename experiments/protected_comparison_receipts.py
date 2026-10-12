"""CPU-only validation for protected-comparison worker result bundles."""

from __future__ import annotations

import hashlib
import json
import math
import re
import stat
import zipfile
from io import BytesIO
from typing import Any, cast

from PIL import Image

from product_scene_editor import protected_comparison as comparison
from product_scene_editor import protected_product, remote

MAX_RESULT_BYTES = 64 * 1024 * 1024
MAX_EXPANDED_BYTES = 128 * 1024 * 1024
MAX_RECEIPT_BYTES = 4 * 1024 * 1024
_RESULT_MEMBER = re.compile(
    r"^(?:outputs|replays)/[A-Za-z0-9][A-Za-z0-9._-]{0,63}/"
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\.png$"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_SHA = re.compile(r"^[0-9a-f]{40}$")
_PIPELINE_LOAD_FIELDS = {
    "stage",
    "device",
    "component_pins",
    "duration_seconds",
    "status",
    "error",
}


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate receipt JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite receipt JSON value: {value}")


def _read_result_bundle(data: bytes) -> tuple[dict[str, Any], dict[str, bytes]]:
    if not isinstance(data, bytes) or not data or len(data) > MAX_RESULT_BYTES:
        raise ValueError("protected comparison result is empty or exceeds its byte limit")
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if not infos or len(names) != len(set(names)) or archive.comment:
                raise ValueError("result ZIP has duplicate members or a comment")
            total = 0
            for info in infos:
                if info.filename != "receipt.json" and (
                    _RESULT_MEMBER.fullmatch(info.filename) is None
                    or ".." in info.filename.split("/")
                ):
                    raise ValueError("result ZIP contains an unsafe member path")
                mode = info.external_attr >> 16
                total += info.file_size
                if (
                    info.is_dir()
                    or stat.S_ISLNK(mode)
                    or info.flag_bits & 1
                    or info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                    or info.file_size
                    > (MAX_RECEIPT_BYTES if info.filename == "receipt.json" else 4 * 1024 * 1024)
                    or total > MAX_EXPANDED_BYTES
                ):
                    raise ValueError("result ZIP contains an unsafe or oversized member")
            if names.count("receipt.json") != 1:
                raise ValueError("result ZIP must contain one receipt.json")
            raw_receipt = archive.read("receipt.json")
            receipt = json.loads(
                raw_receipt.decode("utf-8"),
                object_pairs_hook=_unique_pairs,
                parse_constant=_reject_constant,
            )
            if not isinstance(receipt, dict):
                raise ValueError("result receipt must be a JSON object")
            members = {
                info.filename: archive.read(info)
                for info in infos
                if info.filename != "receipt.json"
            }
            return receipt, members
    except (
        OSError,
        RuntimeError,
        EOFError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ) as exc:
        raise ValueError("protected comparison result is not a safe receipt ZIP") from exc


def _finite_nonnegative(value: object, role: str) -> float:
    if type(value) is int:
        numeric = float(value)
    elif type(value) is float:
        numeric = value
    else:
        raise ValueError(f"receipt {role} must be a finite nonnegative number")
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"receipt {role} must be a finite nonnegative number")
    return numeric


def _manifest_digest(manifest: dict[str, object]) -> str:
    canonical = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _artifact_paths(manifest: dict[str, object]) -> set[str]:
    schedule = comparison.generation_schedule(manifest)
    expected: set[str] = set()
    for task in schedule:
        case_id = task["case_id"]
        if task["purpose"] == "primary":
            name = (
                "full_image_edit.png"
                if task["arm"] == "base_full_image_edit"
                else "empty_scene_raw.png"
            )
            expected.add(f"outputs/{case_id}/{name}")
        else:
            expected.add(f"replays/{case_id}/{task['arm']}.png")
        if task["arm"] == "empty_scene_composite":
            composite = (
                f"outputs/{case_id}/empty_scene_composite.png"
                if task["purpose"] == "primary"
                else f"replays/{case_id}/empty_scene_composite_final.png"
            )
            expected.add(composite)
    return expected


def _validate_artifacts(
    receipt: dict[str, Any], members: dict[str, bytes], manifest: dict[str, object]
) -> dict[str, bytes]:
    rows = receipt.get("output_artifacts")
    if not isinstance(rows, list):
        raise ValueError("receipt output_artifacts must be a list")
    artifacts: dict[str, bytes] = {}
    allowed_paths = _artifact_paths(manifest)
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("receipt contains an invalid artifact record")
        path, digest = row.get("path"), row.get("sha256")
        if (
            not isinstance(path, str)
            or _RESULT_MEMBER.fullmatch(path) is None
            or path not in allowed_paths
            or path in artifacts
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or type(row.get("width")) is not int
            or row["width"] != 512
            or type(row.get("height")) is not int
            or row["height"] != 512
            or path not in members
        ):
            raise ValueError("receipt artifact record is malformed or does not match the ZIP")
        content = members[path]
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("receipt artifact hash does not match the ZIP member")
        try:
            with Image.open(BytesIO(content)) as image:
                if image.format != "PNG" or image.mode != "RGB" or image.size != (512, 512):
                    raise ValueError("artifact is not a 512x512 RGB PNG")
                image.load()
        except (OSError, ValueError) as exc:
            raise ValueError("receipt artifact is not a readable 512x512 RGB PNG") from exc
        artifacts[path] = content
    if set(artifacts) != set(members):
        raise ValueError("result ZIP members differ from receipt artifact records")
    return artifacts


def _record_matches_task(row: object, task: dict[str, Any], *, timing: bool) -> bool:
    if not isinstance(row, dict):
        return False
    common = {key: task[key] for key in ("index", "purpose", "case_id", "arm")}
    if any(row.get(key) != value for key, value in common.items()):
        return False
    try:
        _finite_nonnegative(row.get("generation_seconds"), "generation timing")
    except ValueError:
        return False
    if timing:
        return set(row) == set(common) | {"generation_seconds"}
    path = (
        f"outputs/{task['case_id']}/full_image_edit.png"
        if task["purpose"] == "primary" and task["arm"] == "base_full_image_edit"
        else f"outputs/{task['case_id']}/empty_scene_raw.png"
        if task["purpose"] == "primary"
        else f"replays/{task['case_id']}/{task['arm']}.png"
    )
    return (
        set(row) == set(common) | {"seed", "generation_seconds", "output_path", "output_sha256"}
        and row.get("seed") == task["seed"]
        and row.get("output_path") == path
        and isinstance(row.get("output_sha256"), str)
        and _SHA256.fullmatch(row["output_sha256"]) is not None
    )


def _validate_generations(
    receipt: dict[str, Any], manifest: dict[str, object]
) -> list[dict[str, Any]]:
    schedule = [dict(task) for task in comparison.generation_schedule(manifest)]
    count = receipt.get("generation_call_count")
    timings = receipt.get("generation_call_timings")
    records = receipt.get("generation_records")
    if (
        type(count) is not int
        or count < 0
        or count > len(schedule)
        or not isinstance(timings, list)
        or len(timings) != count
        or not isinstance(records, list)
        or len(records) > count
        or len(records) > len(schedule)
    ):
        raise ValueError("receipt generation call counts are inconsistent")
    for index, row in enumerate(timings):
        if index >= len(schedule) or not _record_matches_task(row, schedule[index], timing=True):
            raise ValueError("receipt generation timing rows do not match the schedule")
    for index, row in enumerate(records):
        if index >= len(schedule) or not _record_matches_task(row, schedule[index], timing=False):
            raise ValueError("receipt generation records do not match the schedule")
    return schedule


def _validate_generation_outputs(receipt: dict[str, Any], artifacts: dict[str, bytes]) -> None:
    records = receipt["generation_records"]
    timings = receipt["generation_call_timings"]
    receipt_only_packaging_failure = (
        receipt.get("status") == "failed"
        and receipt.get("failure_stage") == "result_packaging"
        and receipt.get("output_artifacts") == []
        and not artifacts
    )
    for record in records:
        index = record["index"]
        timing = timings[index]
        if record["generation_seconds"] != timing["generation_seconds"]:
            raise ValueError("generation record duration differs from its timing row")
        content = artifacts.get(record["output_path"])
        if content is None:
            if receipt_only_packaging_failure:
                continue
            raise ValueError("generation record output is missing from receipt artifacts")
        if hashlib.sha256(content).hexdigest() != record["output_sha256"]:
            raise ValueError("generation record hash does not match its artifact bytes")


def _validate_pipeline_loads(receipt: dict[str, Any], manifest: dict[str, object]) -> None:
    loads = receipt.get("pipeline_loads")
    if not isinstance(loads, list) or len(loads) > 2:
        raise ValueError("receipt pipeline loads must contain at most two records")
    expected_stages = ("base_load", "fresh_replay_load")
    for index, row in enumerate(loads):
        if not isinstance(row, dict) or set(row) != _PIPELINE_LOAD_FIELDS:
            raise ValueError("receipt pipeline load record has an invalid shape")
        if (
            row.get("stage") != expected_stages[index]
            or row.get("device") != "cuda"
            or row.get("component_pins") != manifest.get("model_components")
            or row.get("status") not in {"succeeded", "failed"}
        ):
            raise ValueError("receipt pipeline load does not match the pinned stage and device")
        _finite_nonnegative(row.get("duration_seconds"), "pipeline load duration")
        error = row.get("error")
        if (row["status"] == "succeeded" and error is not None) or (
            row["status"] == "failed" and (not isinstance(error, str) or not error)
        ):
            raise ValueError("receipt pipeline load error does not match its status")
        if row["status"] == "failed" and index != len(loads) - 1:
            raise ValueError("a failed pipeline load must be the last recorded load")
    if receipt.get("status") == "passed" and (
        len(loads) != 2 or any(row["status"] != "succeeded" for row in loads)
    ):
        raise ValueError("passed receipt must prove both fresh pipeline loads")


def _validate_composites(
    receipt: dict[str, Any],
    artifacts: dict[str, bytes],
    manifest: dict[str, object],
    payload_files: dict[str, bytes],
) -> None:
    rows = receipt.get("composed_integrity_records")
    if not isinstance(rows, list):
        raise ValueError("receipt composed_integrity_records must be a list")
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("manifest cases must be a list")
    cases: dict[str, dict[str, Any]] = {}
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict) or not isinstance(raw_case.get("case_id"), str):
            raise ValueError("manifest contains an invalid case record")
        cases[raw_case["case_id"]] = cast(dict[str, Any], raw_case)
    tasks = [
        task
        for task in comparison.generation_schedule(manifest)
        if task["arm"] == "empty_scene_composite"
    ]
    if len(rows) > len(tasks):
        raise ValueError("receipt has more compositor records than scheduled composites")
    fields = {
        "source_sha256",
        "background_sha256",
        "alpha_sha256",
        "protected_sha256",
        "output_sha256",
        "protected_count",
        "changed_protected_pixels",
        "max_abs_protected_change",
        "fully_background_count",
        "blended_count",
    }
    for index, row in enumerate(rows):
        task = tasks[index]
        case = cases[task["case_id"]]
        output_path = (
            f"outputs/{task['case_id']}/empty_scene_composite.png"
            if task["purpose"] == "primary"
            else f"replays/{task['case_id']}/empty_scene_composite_final.png"
        )
        background_path = (
            f"outputs/{task['case_id']}/empty_scene_raw.png"
            if task["purpose"] == "primary"
            else f"replays/{task['case_id']}/empty_scene_composite.png"
        )
        if not isinstance(row, dict) or set(row) != fields | {
            "purpose",
            "index",
            "case_id",
            "output_path",
            "compositing_seconds",
        }:
            raise ValueError("receipt compositor record has an invalid shape")
        if (
            row.get("purpose") != task["purpose"]
            or row.get("index") != task["index"]
            or row.get("case_id") != task["case_id"]
            or row.get("output_path") != output_path
            or output_path not in artifacts
            or background_path not in artifacts
        ):
            raise ValueError("receipt compositor record does not match its scheduled artifact")
        _finite_nonnegative(row.get("compositing_seconds"), "compositing duration")
        try:
            expected_png, integrity = protected_product.compose_protected_product(
                payload_files[case["source_rgb"]["path"]],
                artifacts[background_path],
                payload_files[case["alpha_l"]["path"]],
                payload_files[case["protected_l"]["path"]],
            )
            actual_png = artifacts[output_path]
            integrity["output_sha256"] = hashlib.sha256(actual_png).hexdigest()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("receipt compositor inputs cannot be independently checked") from exc
        if (
            _pixels(expected_png) != _pixels(actual_png)
            or {key: row[key] for key in fields} != integrity
            or row["changed_protected_pixels"] != 0
            or row["max_abs_protected_change"] != 0
        ):
            raise ValueError("receipt compositing integrity does not match recomputed pixels")
    if receipt.get("status") == "passed" and len(rows) != 26:
        raise ValueError("passed receipt must contain all 26 compositor records")


def _pixels(content: bytes) -> bytes:
    with Image.open(BytesIO(content)) as image:
        image.load()
        return image.tobytes()


def _validate_replays(
    receipt: dict[str, Any], artifacts: dict[str, bytes], schedule: list[dict[str, Any]]
) -> None:
    checks = receipt.get("replay_checks")
    replay_tasks = [task for task in schedule if task["purpose"] == "fresh_load_replay"]
    if not isinstance(checks, list) or len(checks) > len(replay_tasks):
        raise ValueError("receipt replay_checks do not match the frozen replay schedule")
    fields = {
        "case_id",
        "arm",
        "primary_index",
        "replay_index",
        "final_pixel_equal",
        "raw_background_pixel_equal",
    }
    for index, row in enumerate(checks):
        task = replay_tasks[index]
        primary_path = (
            f"outputs/{task['case_id']}/full_image_edit.png"
            if task["arm"] == "base_full_image_edit"
            else f"outputs/{task['case_id']}/empty_scene_composite.png"
        )
        replay_path = (
            f"replays/{task['case_id']}/{task['arm']}.png"
            if task["arm"] == "base_full_image_edit"
            else f"replays/{task['case_id']}/empty_scene_composite_final.png"
        )
        primary_background_path = (
            primary_path
            if task["arm"] == "base_full_image_edit"
            else f"outputs/{task['case_id']}/empty_scene_raw.png"
        )
        replay_background_path = f"replays/{task['case_id']}/{task['arm']}.png"
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError("receipt replay check has an invalid shape")
        final_equal = (
            primary_path in artifacts
            and replay_path in artifacts
            and _pixels(artifacts[primary_path]) == _pixels(artifacts[replay_path])
        )
        background_equal = (
            primary_background_path in artifacts
            and replay_background_path in artifacts
            and _pixels(artifacts[primary_background_path])
            == _pixels(artifacts[replay_background_path])
        )
        if (
            row.get("case_id") != task["case_id"]
            or row.get("arm") != task["arm"]
            or row.get("primary_index") != task["replay_of"]
            or row.get("replay_index") != task["index"]
            or row.get("final_pixel_equal") is not final_equal
            or row.get("raw_background_pixel_equal") is not background_equal
        ):
            raise ValueError("receipt replay pixel evidence differs from decoded artifacts")
    if receipt.get("status") == "passed" and (
        len(checks) != 4
        or any(
            row["final_pixel_equal"] is not True or row["raw_background_pixel_equal"] is not True
            for row in checks
        )
    ):
        raise ValueError("passed receipt must prove all four exact replays")


def _validate_arm_timings(receipt: dict[str, Any]) -> None:
    timings = receipt["generation_call_timings"]
    totals: dict[str, float] = {}
    for row in timings:
        key = f"{row['purpose']}/{row['arm']}"
        totals[key] = totals.get(key, 0.0) + float(row["generation_seconds"])
    expected = {key: round(value, 6) for key, value in totals.items()}
    if receipt.get("arm_generation_seconds") != expected:
        raise ValueError("receipt arm generation timing totals do not match call timings")


def validate_result_bundle(
    data: bytes,
    *,
    run_id: str,
    payload_sha256: str,
    source_commit: str,
    manifest: dict[str, object],
    payload_files: dict[str, bytes],
) -> dict[str, Any]:
    """Validate identity, partial evidence, and the complete successful worker contract."""
    receipt, members = _read_result_bundle(data)
    status = receipt.get("status")
    source_sha = manifest.get("source_sha")
    if (
        type(receipt.get("schema_version")) is not int
        or receipt["schema_version"] != 1
        or status not in {"passed", "failed"}
        or receipt.get("run_id") != run_id
    ):
        raise ValueError("receipt schema or run ID does not match the cleared comparison")
    if receipt.get("payload_sha256") != payload_sha256:
        raise ValueError("receipt payload hash does not match the cleared payload")
    if (
        not isinstance(source_sha, str)
        or _SOURCE_SHA.fullmatch(source_sha) is None
        or source_sha != source_commit
        or receipt.get("source_sha") != source_commit
    ):
        raise ValueError("receipt source commit does not match the cleared payload")
    if receipt.get("runtime_pins") != remote.RUNTIME_PINS or receipt.get(
        "runtime_pins"
    ) != manifest.get("runtime_pins"):
        raise ValueError("receipt runtime pins do not match the payload")
    if receipt.get("model_components") != remote.MODEL_COMPONENTS or receipt.get(
        "model_components"
    ) != manifest.get("model_components"):
        raise ValueError("receipt model component pins do not match the payload")
    if receipt.get("training_function_called") is not False or (
        type(receipt.get("optimizer_updates")) is not int or receipt["optimizer_updates"] != 0
    ):
        raise ValueError("receipt violates the inference-only training contract")
    if (
        receipt.get("context") != manifest.get("context")
        or receipt.get("manifest_sha256") != _manifest_digest(manifest)
        or receipt.get("generation_settings") != manifest.get("generation")
        or receipt.get("modal_sdk") != "1.6.1"
        or receipt.get("diffsynth_commit") != remote.DIFFSYNTH_COMMIT
    ):
        raise ValueError("receipt identity or frozen inference contract does not match the plan")
    expected_schedule = [dict(task) for task in comparison.generation_schedule(manifest)]
    if receipt.get("schedule") != expected_schedule:
        raise ValueError("receipt schedule does not match the frozen comparison schedule")
    observed_commit = receipt.get("observed_diffsynth_commit")
    if observed_commit is not None and observed_commit != remote.DIFFSYNTH_COMMIT:
        raise ValueError("receipt observed DiffSynth commit does not match its pin")
    runtime_versions = receipt.get("runtime_versions")
    if not isinstance(runtime_versions, dict):
        raise ValueError("receipt runtime_versions must be an object")
    expected_runtime_versions = {"modal": "1.6.1", **remote.RUNTIME_PINS}
    if status == "passed" and (
        observed_commit != remote.DIFFSYNTH_COMMIT or runtime_versions != expected_runtime_versions
    ):
        raise ValueError("passed receipt does not prove the pinned runtime")
    elapsed = _finite_nonnegative(receipt.get("elapsed_seconds"), "elapsed time")
    if elapsed > 1800:
        raise ValueError("receipt execution duration exceeds the frozen timeout")
    memory = receipt.get("peak_allocated_bytes")
    if memory is not None and (type(memory) is not int or memory < 0):
        raise ValueError("receipt peak memory must be a nonnegative integer or null")
    if status == "passed":
        if receipt.get("failure") is not None or receipt.get("failure_stage") is not None:
            raise ValueError("passed receipt must not contain failure details")
    else:
        failure = receipt.get("failure")
        if (
            not isinstance(failure, dict)
            or not isinstance(failure.get("stage"), str)
            or not failure["stage"]
            or not isinstance(failure.get("type"), str)
            or not isinstance(failure.get("message"), str)
            or receipt.get("failure_stage") != failure["stage"]
        ):
            raise ValueError("failed receipt must preserve its worker failure details")
    schedule = _validate_generations(receipt, manifest)
    _validate_arm_timings(receipt)
    _validate_pipeline_loads(receipt, manifest)
    artifacts = _validate_artifacts(receipt, members, manifest)
    _validate_generation_outputs(receipt, artifacts)
    _validate_composites(receipt, artifacts, manifest, payload_files)
    _validate_replays(receipt, artifacts, schedule)
    if status == "passed":
        expected_paths = _artifact_paths(manifest)
        if (
            receipt["generation_call_count"] != 52
            or len(receipt["generation_call_timings"]) != 52
            or len(receipt["generation_records"]) != 52
            or set(artifacts) != expected_paths
            or len(artifacts) != 78
        ):
            raise ValueError(
                "passed receipt does not contain the exact 52-call, 78-artifact result"
            )
    return receipt
