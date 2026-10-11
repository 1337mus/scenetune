"""Dry-first, one-shot Modal controller for the protected mask preflight."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, cast

APP_NAME = "product-scene-editor-mask-preflight"
IMAGE = (
    "python:3.12.12-slim-bookworm@sha256:"
    "593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c"
)
TORCH_INDEX = "https://download.pytorch.org/whl/cu124"
ATTEMPT_NAME = "mask-preflight-attempt.json"
RESULT_NAME = "mask-preflight-result.zip"
FAILURE_NAME = "mask-preflight-failure.json"
OBSERVER_NAME = "mask-preflight-observer.json"
MAX_RESULT_BYTES = 16 * 1024 * 1024
MAX_RESULT_EXPANDED_BYTES = 16 * 1024 * 1024
MAX_PLAN_JSON_BYTES = 1024 * 1024
MAX_CLEARANCE_JSON_BYTES = 256 * 1024
MAX_EVIDENCE_JSON_BYTES = 64 * 1024
STARTUP_TIMEOUT_SECONDS = 180
EXECUTION_TIMEOUT_SECONDS = 600
MAX_CPU_CORES = 2
MAX_MEMORY_MIB = 8192
LAUNCH_OBSERVER_TIMEOUT_SECONDS = STARTUP_TIMEOUT_SECONDS + EXECUTION_TIMEOUT_SECONDS
MAX_SHUTDOWN_CHECKS = 4
SHUTDOWN_WAIT_SECONDS = 2
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_RESULT_PATH = re.compile(
    r"^(?:outputs|replays)/[A-Za-z0-9][A-Za-z0-9._-]{0,63}/(?:alpha|core)\.png$"
)
_PLAN_FIELDS = {
    "schema_version",
    "source_commit",
    "run_id",
    "payload_path",
    "payload_sha256",
    "output_dir",
    "context",
    "resources",
    "runtime_pins",
    "model",
    "model_files",
    "settings",
    "image",
}
_RESOURCE_CONTRACT = {
    "gpu": "A10",
    "cpu_cores": MAX_CPU_CORES,
    "memory_mib": MAX_MEMORY_MIB,
    "startup_timeout_seconds": STARTUP_TIMEOUT_SECONDS,
    "execution_timeout_seconds": EXECUTION_TIMEOUT_SECONDS,
    "retries": 0,
}
_CLEARANCE_FIELDS = {
    "schema_version",
    "status",
    "plan_sha256",
    "source_commit",
    "payload_sha256",
    "run_id",
    "context",
    "ci_status",
    "ci_run_id",
    "ci_source_commit",
    "checked_utc",
    "usage_rates_evidence_path",
    "usage_rates_evidence_sha256",
    "estimated_seconds",
    "estimated_compute_usd",
    "step_reserve_usd",
    "allowance_usd",
    "cost_exclusions",
}
_COST_EXCLUSIONS = ["image_build", "storage", "transfer", "shutdown_lag"]


@dataclass(frozen=True)
class Plan:
    path: Path
    root: Path
    payload_path: Path
    output_dir: Path
    value: dict[str, Any]
    manifest: dict[str, Any]
    payload: bytes
    payload_sha256: str
    plan_sha256: str


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--launch", action="store_true", help="launch one cleared A10 call")
    action.add_argument("--recover", action="store_true", help="observe only the recorded call")
    parser.add_argument(
        "--clearance", type=Path, default=Path(".context/mask-preflight-clearance.json")
    )
    return parser


def _contract() -> Any:
    """Load the CPU-only builder only after a controller operation is called."""
    return importlib.import_module("product_scene_editor.mask_preflight")


def _remote() -> Any:
    return importlib.import_module("product_scene_editor.remote")


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(
    path: Path, role: str, *, max_bytes: int = MAX_CLEARANCE_JSON_BYTES
) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{role} is missing or unsafe")
    if path.stat().st_size > max_bytes:
        raise ValueError(f"{role} exceeds its JSON size limit")
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        raise ValueError(f"{role} exceeds its JSON size limit")
    try:
        value = json.loads(raw, object_pairs_hook=_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{role} is not valid unique-key JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object")
    return value, raw


def _project_path(value: object, root: Path, role: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{role} must be a project-relative path")
    raw = Path(value)
    posix = PurePosixPath(value)
    if (
        raw.is_absolute()
        or posix.as_posix() != value
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise ValueError(f"{role} must stay inside the project")
    path = root / raw
    if path.is_symlink():
        raise ValueError(f"{role} must not be a symlink")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"{role} must stay inside the project")
    return path


def _validate_plan(path: Path, *, root: Path = PROJECT_ROOT) -> Plan:
    value, raw_plan = _read_json(path, "mask preflight plan", max_bytes=MAX_PLAN_JSON_BYTES)
    if (
        set(value) != _PLAN_FIELDS
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != 1
    ):
        raise ValueError("mask preflight plan fields or schema version are invalid")
    source_commit, run_id = value.get("source_commit"), value.get("run_id")
    payload_sha256 = value.get("payload_sha256")
    if not isinstance(source_commit, str) or _SHA1.fullmatch(source_commit) is None:
        raise ValueError("plan source_commit must be a full lowercase Git SHA")
    if not isinstance(run_id, str) or _SAFE_ID.fullmatch(run_id) is None:
        raise ValueError("plan run_id must be a safe identifier")
    if not isinstance(payload_sha256, str) or _SHA256.fullmatch(payload_sha256) is None:
        raise ValueError("plan payload_sha256 must be a lowercase SHA-256 digest")
    contract = _contract()
    resources = value.get("resources")
    if not isinstance(resources, dict) or any(
        type(resources.get(key)) is not int
        for key in (
            "cpu_cores",
            "memory_mib",
            "startup_timeout_seconds",
            "execution_timeout_seconds",
            "retries",
        )
    ):
        raise ValueError("resource limits must use exact integer values")
    if (
        value["context"] != contract.CONTEXT
        or value["resources"] != _RESOURCE_CONTRACT
        or value["runtime_pins"] != contract.RUNTIME_PINS
        or value["model"] != contract.MODEL
        or value["model_files"] != contract.MODEL_FILES
        or value["settings"] != contract.SETTINGS
        or value["image"] != IMAGE
    ):
        raise ValueError("mask preflight plan differs from its frozen runtime contract")
    payload_path = _project_path(value["payload_path"], root, "payload_path")
    output_dir = _project_path(value["output_dir"], root, "output_dir")
    if payload_path.is_symlink() or not payload_path.is_file():
        raise ValueError("mask preflight payload is missing or unsafe")
    payload = payload_path.read_bytes()
    actual_payload_sha = hashlib.sha256(payload).hexdigest()
    if actual_payload_sha != payload_sha256:
        raise ValueError("mask preflight payload hash does not match the plan")
    try:
        manifest, _files = contract.validate_payload_bytes(payload)
    except Exception as exc:
        raise ValueError("mask preflight payload failed builder validation") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("source_sha") != source_commit
        or manifest.get("context") != value["context"]
        or manifest.get("model") != value["model"]
        or manifest.get("model_files") != value["model_files"]
        or manifest.get("runtime_pins") != value["runtime_pins"]
        or manifest.get("settings") != value["settings"]
    ):
        raise ValueError("plan does not bind the validated payload manifest")
    return Plan(
        path=path,
        root=root,
        payload_path=payload_path,
        output_dir=output_dir,
        value=value,
        manifest=manifest,
        payload=payload,
        payload_sha256=actual_payload_sha,
        plan_sha256=hashlib.sha256(raw_plan).hexdigest(),
    )


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False, timeout=30
    )
    if result.returncode:
        raise ValueError("mask preflight could not verify local Git state")
    return result.stdout.strip()


def _verify_source(plan: Plan) -> None:
    if _git(plan.root, "rev-parse", "HEAD") != plan.value["source_commit"]:
        raise ValueError("HEAD does not match the frozen source commit")
    if _git(plan.root, "rev-parse", "origin/main") != plan.value["source_commit"]:
        raise ValueError("cached origin/main does not match the frozen source commit")
    dirty = _git(
        plan.root,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        "src/product_scene_editor",
        "experiments",
    )
    if dirty:
        raise ValueError("mask preflight source directories must be clean")


def _validate_clearance(plan: Plan, path: Path, *, now: datetime | None = None) -> dict[str, Any]:
    clearance, _raw = _read_json(
        path, "mask preflight launch clearance", max_bytes=MAX_CLEARANCE_JSON_BYTES
    )
    if set(clearance) != _CLEARANCE_FIELDS:
        raise ValueError("mask preflight clearance fields are missing or unknown")
    expected = {
        "schema_version": 1,
        "status": "approved",
        "plan_sha256": plan.plan_sha256,
        "source_commit": plan.value["source_commit"],
        "payload_sha256": plan.payload_sha256,
        "run_id": plan.value["run_id"],
        "context": plan.value["context"],
        "ci_status": "success",
        "ci_source_commit": plan.value["source_commit"],
        "estimated_seconds": 780,
        "estimated_compute_usd": 0.2729688,
        "step_reserve_usd": 1,
        "allowance_usd": 8,
        "cost_exclusions": _COST_EXCLUSIONS,
    }
    if any(clearance.get(key) != value for key, value in expected.items()):
        raise ValueError("clearance does not match the plan, successful CI, or cost boundary")
    ci_run = clearance.get("ci_run_id")
    if type(ci_run) is not int or ci_run <= 0:
        raise ValueError("clearance must identify one successful CI run")
    timestamp = clearance.get("checked_utc")
    if not isinstance(timestamp, str):
        raise ValueError("clearance checked_utc is missing")
    try:
        checked = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("clearance checked_utc is invalid") from exc
    current = now or datetime.now(UTC)
    if checked.tzinfo is None or current.tzinfo is None:
        raise ValueError("clearance timestamps must include a timezone")
    age = current.astimezone(UTC) - checked.astimezone(UTC)
    if age < timedelta(0) or age > timedelta(minutes=30):
        raise ValueError("mask preflight clearance is stale or from the future")
    evidence_path = _project_path(
        clearance.get("usage_rates_evidence_path"), plan.root, "usage/rates evidence"
    )
    evidence_sha = clearance.get("usage_rates_evidence_sha256")
    if (
        not isinstance(evidence_sha, str)
        or _SHA256.fullmatch(evidence_sha) is None
        or not evidence_path.is_file()
    ):
        raise ValueError("usage/rates evidence path or SHA-256 does not match clearance")
    evidence, raw_evidence = _read_json(
        evidence_path, "usage/rates evidence", max_bytes=MAX_EVIDENCE_JSON_BYTES
    )
    if hashlib.sha256(raw_evidence).hexdigest() != evidence_sha:
        raise ValueError("usage/rates evidence path or SHA-256 does not match clearance")
    if (
        evidence.get("checked_utc") != timestamp
        or evidence.get("context") != plan.value["context"]
        or evidence.get("active_apps") != []
    ):
        raise ValueError(
            "usage/rates evidence is stale, targets another account, or shows active apps"
        )
    rates = evidence.get("rates")
    rate_values = {
        "a10_usd_per_second": Decimal("0.000306"),
        "cpu_core_usd_per_second": Decimal("0.0000131"),
        "memory_gib_usd_per_second": Decimal("0.00000222"),
    }
    if not isinstance(rates, dict) or set(rates) != set(rate_values):
        raise ValueError("usage/rates evidence does not contain the verified A10 rates")
    parsed_rates: dict[str, Decimal] = {}
    try:
        for key, expected_rate in rate_values.items():
            value = rates[key]
            if type(value) not in {int, float}:
                raise ValueError("usage/rates values must be numeric")
            parsed = Decimal(str(value))
            if not parsed.is_finite() or parsed != expected_rate:
                raise ValueError("usage/rates evidence differs from verified rates")
            parsed_rates[key] = parsed
    except (ArithmeticError, ValueError) as exc:
        raise ValueError("usage/rates evidence differs from verified rates") from exc
    computed = Decimal(780) * (
        parsed_rates["a10_usd_per_second"]
        + Decimal(2) * parsed_rates["cpu_core_usd_per_second"]
        + Decimal(8) * parsed_rates["memory_gib_usd_per_second"]
    )
    billing = evidence.get("billing")
    if (
        not isinstance(billing, dict)
        or billing.get("scope") != "workspace-delayed-not-project-invoice"
    ):
        raise ValueError("usage/rates evidence billing scope is unsupported")
    try:
        gross = Decimal(billing["workspace_gross_usd"])
        net = Decimal(billing["workspace_net_usd"])
    except (KeyError, ArithmeticError, TypeError, ValueError) as exc:
        raise ValueError("workspace billing totals must be decimal strings") from exc
    if (
        not isinstance(billing.get("workspace_gross_usd"), str)
        or not isinstance(billing.get("workspace_net_usd"), str)
        or not gross.is_finite()
        or not net.is_finite()
        or gross < 0
        or net < 0
    ):
        raise ValueError("workspace billing totals must be nonnegative decimal strings")
    if computed != Decimal("0.2729688"):
        raise ValueError("usage/rates evidence billing estimate does not match the fixed estimate")
    return clearance


def _verify_identity(context: dict[str, str]) -> None:
    remote = _remote()
    remote.validate_launch_environment(context)
    remote.verify_modal_identity(**context)


def _write_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def _write_event(output_dir: Path, prefix: str, value: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.time_ns()
    for suffix in range(1000):
        ending = "" if suffix == 0 else f"-{suffix:03d}"
        path = output_dir / f"{prefix}-{stamp}{ending}.json"
        try:
            _write_once(path, value)
        except FileExistsError:
            continue
        return path
    raise FileExistsError("mask preflight event evidence limit reached")


@contextmanager
def _observer(plan: Plan, action: str) -> Iterator[None]:
    path = plan.output_dir / OBSERVER_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("mask preflight observer ownership file must not be a symlink")
    stream = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another mask preflight observer owns this run") from exc
        owner = {
            "action": action,
            "pid": os.getpid(),
            "started_utc": datetime.now(UTC).isoformat(),
            "run_id": plan.value["run_id"],
        }
        stream.seek(0)
        stream.truncate()
        json.dump(owner, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        yield
    finally:
        stream.close()


def _attempt_path(plan: Plan) -> Path:
    return plan.output_dir / ATTEMPT_NAME


def _reserve_attempt(plan: Plan) -> Path:
    path = _attempt_path(plan)
    _write_once(
        path,
        {
            "schema_version": 1,
            "run_id": plan.value["run_id"],
            "source_commit": plan.value["source_commit"],
            "payload_sha256": plan.payload_sha256,
            "plan_sha256": plan.plan_sha256,
            "context": plan.value["context"],
            "status": "reserved",
            "app_id": None,
            "call_id": None,
            "spawn_count": 0,
            "reserved_at_utc": datetime.now(UTC).isoformat(),
            "stop_requested_at_utc": None,
        },
    )
    return path


def _update_attempt(path: Path, **updates: Any) -> dict[str, Any]:
    marker, _raw = _read_json(path, "mask preflight attempt marker")
    for key in ("app_id", "call_id"):
        if key in updates and marker.get(key) is not None and updates[key] != marker[key]:
            raise ValueError(f"attempt marker {key} is immutable")
    if "spawn_count" in updates and updates["spawn_count"] not in {marker.get("spawn_count"), 1}:
        raise ValueError("mask preflight permits one reserved spawn")
    marker.update(updates)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8") as stream:
        json.dump(marker, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    return marker


def _read_attempt(plan: Plan) -> dict[str, Any]:
    marker, _raw = _read_json(_attempt_path(plan), "mask preflight attempt marker")
    if (
        marker.get("schema_version") != 1
        or marker.get("run_id") != plan.value["run_id"]
        or marker.get("source_commit") != plan.value["source_commit"]
        or marker.get("payload_sha256") != plan.payload_sha256
        or marker.get("plan_sha256") != plan.plan_sha256
        or marker.get("context") != plan.value["context"]
        or marker.get("spawn_count") not in {0, 1}
        or marker.get("status")
        not in {
            "reserved",
            "app_started",
            "spawn_reserved",
            "observing",
            "observation_pending",
            "worker_returned",
            "worker_failed",
            "shutdown_pending",
            "stopped",
        }
    ):
        raise ValueError("attempt marker does not bind this plan and source")
    for key in ("app_id", "call_id"):
        value = marker.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"attempt marker {key} is invalid")
    if marker.get("call_id") is not None and marker.get("spawn_count") != 1:
        raise ValueError("recorded call ID has no reserved spawn")
    return marker


def _build_image(modal: Any, root: Path, pins: dict[str, str]) -> Any:
    packages = [f"{name}=={version}" for name, version in pins.items()]
    image = modal.Image.from_registry(IMAGE).pip_install(*packages, extra_index_url=TORCH_INDEX)
    if not (root / "src/product_scene_editor/mask_preflight_inference.py").is_file():
        raise FileNotFoundError("mask preflight worker source is missing")
    return image.add_local_python_source("product_scene_editor", copy=True)


def _worker(payload: bytes, run_id: str) -> bytes:
    from product_scene_editor.mask_preflight_inference import (  # type: ignore[import-untyped]
        run_remote_mask_preflight,
    )

    return cast(bytes, run_remote_mask_preflight(payload, run_id))


def _regular_zip_member(info: zipfile.ZipInfo) -> bool:
    mode = info.external_attr >> 16
    return (
        not info.is_dir()
        and info.compress_type in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
        and not info.flag_bits & 1
        and (mode == 0 or (mode >> 12) == 0o10)
    )


def _validate_result(data: bytes, plan: Plan) -> dict[str, Any]:
    if not isinstance(data, bytes) or not data or len(data) > MAX_RESULT_BYTES:
        raise ValueError("mask preflight result is empty or exceeds 16 MiB")
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if not infos or len(names) != len(set(names)) or archive.comment:
                raise ValueError("result ZIP members must be unique and un-commented")
            if any(
                not _regular_zip_member(info)
                or (
                    info.filename != "receipt.json"
                    and _RESULT_PATH.fullmatch(info.filename) is None
                )
                for info in infos
            ):
                raise ValueError("result ZIP contains an unsafe member")
            if sum(info.file_size for info in infos) > MAX_RESULT_EXPANDED_BYTES:
                raise ValueError("result ZIP exceeds the expanded size limit")
            raw_receipt = archive.read("receipt.json")
            receipt = json.loads(raw_receipt, object_pairs_hook=_pairs)
            members = {
                info.filename: archive.read(info)
                for info in infos
                if info.filename != "receipt.json"
            }
    except (
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ) as exc:
        raise ValueError("mask preflight result is not a safe receipt ZIP") from exc
    contract = _contract()
    if not isinstance(receipt, dict):
        raise ValueError("mask preflight receipt must be a JSON object")
    status = receipt.get("status")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("run_id") != plan.value["run_id"]
        or receipt.get("payload_sha256") != plan.payload_sha256
        or status not in {"passed", "failed"}
        or receipt.get("source_sha") != plan.value["source_commit"]
        or receipt.get("context") != plan.value["context"]
        or receipt.get("model") != contract.MODEL
        or receipt.get("model_files") != contract.MODEL_FILES
        or receipt.get("runtime_pins") != contract.RUNTIME_PINS
        or receipt.get("settings") != contract.SETTINGS
        or receipt.get("mask_status") != "draft_unaccepted"
        or receipt.get("training_function_called") is not False
        or type(receipt.get("optimizer_updates")) is not int
        or receipt["optimizer_updates"] != 0
    ):
        raise ValueError("mask preflight receipt does not match the cleared draft-only run")
    artifacts = receipt.get("output_artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("receipt output_artifacts must be a list")
    records: dict[str, dict[str, Any]] = {}
    for record in artifacts:
        if not isinstance(record, dict):
            raise ValueError("receipt contains an invalid output artifact record")
        name, digest = record.get("path"), record.get("sha256")
        if (
            not isinstance(name, str)
            or _RESULT_PATH.fullmatch(name) is None
            or name in records
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or record.get("width") != 512
            or record.get("height") != 512
            or hashlib.sha256(members.get(name, b"")).hexdigest() != digest
        ):
            raise ValueError("receipt artifact path or hash does not match the ZIP")
        records[name] = record
    if set(members) != set(records):
        raise ValueError("result ZIP members differ from receipt output records")
    if status == "passed":
        case_ids = [case["case_id"] for case in plan.manifest["cases"]]
        expected = {
            f"{group}/{case_id}/{kind}.png"
            for group, ids in (("outputs", case_ids), ("replays", case_ids[:4]))
            for case_id in ids
            for kind in ("alpha", "core")
        }
        strict_loads, replays = receipt.get("strict_loads"), receipt.get("replay_checks")
        if (
            type(receipt.get("inference_call_count")) is not int
            or receipt["inference_call_count"] != 28
            or set(records) != expected
            or len(records) != 56
            or not isinstance(strict_loads, list)
            or len(strict_loads) != 2
            or not isinstance(replays, list)
            or len(replays) != 4
            or any(
                not isinstance(row, dict)
                or row.get("validated") is not True
                or row.get("strict") is not True
                or row.get("missing_keys") != []
                or row.get("unexpected_keys") != []
                or row.get("bb_pretrained") is not False
                for row in strict_loads
            )
            or any(
                not isinstance(row, dict)
                or row.get("alpha_pixel_equal") is not True
                or row.get("core_pixel_equal") is not True
                for row in replays
            )
        ):
            raise ValueError("successful mask preflight receipt violates its inference contract")
    return receipt


def _save_or_confirm_result(data: bytes, path: Path) -> None:
    if path.is_symlink():
        raise ValueError("mask preflight result path must not be a symlink")
    try:
        with path.open("xb") as stream:
            stream.write(data)
    except FileExistsError:
        if not path.is_file() or path.read_bytes() != data:
            raise ValueError("recovered result differs from the durable result") from None


def _app_list_confirms_stopped(output: str, app_id: str) -> bool:
    try:
        apps = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(apps, list):
        return False
    matches = [row for row in apps if isinstance(row, dict) and row.get("app_id") == app_id]
    return (
        len(matches) == 1
        and matches[0].get("state") == "stopped"
        and matches[0].get("tasks") == "0"
    )


def _record_shutdown_event(
    plan: Plan,
    action: str,
    app_id: str,
    *,
    completed: subprocess.CompletedProcess[str] | None = None,
    error: BaseException | None = None,
    attempt: int | None = None,
) -> None:
    event: dict[str, Any] = {
        "schema_version": 1,
        "action": action,
        "app_id": app_id,
        "observed_at_utc": datetime.now(UTC).isoformat(),
    }
    if attempt is not None:
        event["attempt"] = attempt
    if completed is not None:
        stdout = completed.stdout if isinstance(completed.stdout, str) else ""
        stderr_value = getattr(completed, "stderr", "")
        stderr = stderr_value if isinstance(stderr_value, str) else ""
        event["returncode"] = completed.returncode
        event["stdout_sha256"] = hashlib.sha256(stdout.encode()).hexdigest()
        event["stderr_sha256"] = hashlib.sha256(stderr.encode()).hexdigest()
        if action == "app_list":
            try:
                apps = json.loads(stdout)
            except (json.JSONDecodeError, TypeError):
                apps = None
            matches = (
                [row for row in apps if isinstance(row, dict) and row.get("app_id") == app_id]
                if isinstance(apps, list)
                else []
            )
            event["json_list"] = isinstance(apps, list)
            event["matching_apps"] = [
                {key: row.get(key) for key in ("app_id", "state", "tasks")} for row in matches
            ]
            event["confirmed_stopped"] = _app_list_confirms_stopped(stdout, app_id)
    if error is not None:
        event["error_type"] = type(error).__name__
        event["error"] = _remote().sanitize_failure(error)
    _write_event(plan.output_dir, f"mask-preflight-{action}", event)


def _stop_and_confirm(plan: Plan, app_id: str) -> None:
    marker_path = _attempt_path(plan)
    marker = _read_attempt(plan)
    if marker.get("app_id") != app_id:
        raise ValueError("refusing to stop an app other than the recorded app")
    if marker.get("status") == "stopped":
        return
    context = plan.value["context"]
    env = dict(os.environ)
    env["MODAL_PROFILE"] = context["profile"]
    stop_count = int(marker.get("stop_request_count", 0)) + 1
    _update_attempt(
        marker_path,
        status="shutdown_pending",
        stop_requested_at_utc=datetime.now(UTC).isoformat(),
        stop_request_count=stop_count,
    )
    _record_shutdown_event(plan, "stop_request", app_id, attempt=stop_count)
    try:
        stopped = subprocess.run(
            [
                sys.executable,
                "-m",
                "modal",
                "app",
                "stop",
                "--yes",
                "--env",
                context["environment"],
                app_id,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
            env=env,
        )
    except Exception as exc:
        _record_shutdown_event(plan, "stop_response", app_id, error=exc, attempt=stop_count)
    else:
        _record_shutdown_event(plan, "stop_response", app_id, completed=stopped, attempt=stop_count)
    for attempt in range(1, MAX_SHUTDOWN_CHECKS + 1):
        try:
            observed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "modal",
                    "app",
                    "list",
                    "--json",
                    "--env",
                    context["environment"],
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
                env=env,
            )
        except Exception as exc:
            _record_shutdown_event(plan, "app_list", app_id, error=exc, attempt=attempt)
            confirmed = False
        else:
            confirmed = observed.returncode == 0 and _app_list_confirms_stopped(
                observed.stdout, app_id
            )
            _record_shutdown_event(plan, "app_list", app_id, completed=observed, attempt=attempt)
        if confirmed:
            _update_attempt(
                marker_path,
                status="stopped",
                shutdown_checked_at_utc=datetime.now(UTC).isoformat(),
            )
            return
        if attempt < MAX_SHUTDOWN_CHECKS:
            time.sleep(SHUTDOWN_WAIT_SECONDS)
    raise RuntimeError("Modal did not confirm the exact app stopped with zero tasks")


def _failure(plan: Plan, stage: str, error: BaseException, *, pending: bool) -> None:
    remote = _remote()
    receipt = {
        "schema_version": 1,
        "run_id": plan.value["run_id"],
        "status": "observation_pending" if pending else "failed",
        "stage": stage,
        "error_type": type(error).__name__,
        "message": remote.sanitize_failure(error),
        "app_id": None,
        "call_id": None,
    }
    marker_path = _attempt_path(plan)
    if marker_path.is_file() and not marker_path.is_symlink():
        marker = _read_attempt(plan)
        receipt["app_id"], receipt["call_id"] = marker.get("app_id"), marker.get("call_id")
        _update_attempt(
            marker_path,
            status="observation_pending" if pending else "worker_failed",
            failure_stage=stage,
        )
    _write_event(plan.output_dir, "mask-preflight-failure", receipt)


def _check_output_conflicts(plan: Plan) -> None:
    for name in (ATTEMPT_NAME, RESULT_NAME, FAILURE_NAME):
        path = plan.output_dir / name
        if path.exists() or path.is_symlink():
            raise FileExistsError("mask preflight output already has a result or attempt marker")
    if plan.output_dir.exists() and any(plan.output_dir.glob("mask-preflight-failure-*.json")):
        raise FileExistsError("mask preflight output already has failure evidence")


def _launch_locked(plan: Plan, *, clearance_path: Path) -> int:
    _check_output_conflicts(plan)
    _verify_source(plan)
    _validate_clearance(plan, clearance_path)
    context = plan.value["context"]
    _verify_identity(context)
    attempt_path = _reserve_attempt(plan)
    app: Any | None = None
    app_id: str | None = None
    call_id: str | None = None
    phase = "modal_import"
    pending = False
    inner_error: BaseException | None = None
    result: object | None = None
    modal: Any | None = None
    try:
        os.environ["MODAL_PROFILE"] = context["profile"]
        modal = importlib.import_module("modal")
        if importlib.metadata.version("modal") != "1.6.1":
            raise RuntimeError("Modal SDK 1.6.1 is required")
        image = _build_image(modal, plan.root, plan.value["runtime_pins"])
        app = modal.App(APP_NAME, image=image)
        run_once = app.function(
            gpu=plan.value["resources"]["gpu"],
            cpu=(plan.value["resources"]["cpu_cores"], MAX_CPU_CORES),
            memory=(plan.value["resources"]["memory_mib"], MAX_MEMORY_MIB),
            min_containers=0,
            max_containers=1,
            buffer_containers=0,
            scaledown_window=2,
            retries=0,
            single_use_containers=True,
            serialized=True,
            include_source=False,
            startup_timeout=STARTUP_TIMEOUT_SECONDS,
            timeout=EXECUTION_TIMEOUT_SECONDS,
        )(_worker)
        phase = "modal_spawn"
        with modal.enable_output(), app.run(environment_name=context["environment"], detach=True):
            app_id_value = getattr(app, "app_id", None) or getattr(app, "object_id", None)
            if not isinstance(app_id_value, str) or not app_id_value:
                raise RuntimeError("Modal app ID was unavailable before spawn")
            app_id = app_id_value
            _update_attempt(attempt_path, status="app_started", app_id=app_id)
            _update_attempt(attempt_path, status="spawn_reserved", spawn_count=1)
            try:
                call = run_once.spawn(plan.payload, plan.value["run_id"])
                call_value = getattr(call, "object_id", None) or getattr(call, "call_id", None)
                if not isinstance(call_value, str) or not call_value:
                    raise RuntimeError("Modal call ID was unavailable after spawn")
                call_id = call_value
                _update_attempt(attempt_path, status="observing", call_id=call_id)
                phase = "modal_observation"
                result = call.get(timeout=LAUNCH_OBSERVER_TIMEOUT_SECONDS)
            except BaseException as exc:
                inner_error = exc
        if inner_error is not None:
            raise inner_error
    except BaseException as exc:
        marker = _read_attempt(plan)
        reserved_spawn = marker.get("spawn_count") == 1
        terminal_call_error = phase == "modal_observation" and _is_terminal_call_error(exc, modal)
        pending = reserved_spawn and not terminal_call_error
        _failure(plan, phase, exc, pending=pending)
        if pending:
            return 1
        app_id = marker.get("app_id")
        if app_id is not None:
            _stop_and_confirm(plan, app_id)
        return 1
    if not isinstance(result, bytes):
        error = ValueError("Modal call did not return result bytes")
        _failure(plan, "modal_observation", error, pending=True)
        return 1
    result_path = plan.output_dir / RESULT_NAME
    try:
        _save_or_confirm_result(result, result_path)
        receipt = _validate_result(result, plan)
    except Exception as exc:
        _failure(plan, "result_validation", exc, pending=False)
        _stop_and_confirm(plan, app_id or "")
        return 1
    status = "worker_returned" if receipt["status"] == "passed" else "worker_failed"
    _update_attempt(attempt_path, status=status, worker_status=receipt["status"])
    _stop_and_confirm(plan, app_id or "")
    return 0 if receipt["status"] == "passed" else 1


def _is_terminal_call_error(error: BaseException, modal: Any | None) -> bool:
    """Recognize definitive remote call failures, not client observer timeouts."""
    exceptions = getattr(modal, "exception", None)
    terminal_types = tuple(
        error_type
        for error_type in (
            getattr(exceptions, "FunctionTimeoutError", None),
            getattr(exceptions, "InternalFailure", None),
        )
        if isinstance(error_type, type)
    )
    return bool(terminal_types) and isinstance(error, terminal_types)


def _launch(plan: Plan, *, clearance_path: Path) -> int:
    with _observer(plan, "launch"):
        return _launch_locked(plan, clearance_path=clearance_path)


def _recover_locked(plan: Plan) -> int:
    marker = _read_attempt(plan)
    if marker.get("spawn_count") == 1 and marker.get("call_id") is None:
        print(json.dumps({"status": "manual_stop_required", "app_id": marker.get("app_id")}))
        return 1
    app_id, call_id = marker.get("app_id"), marker.get("call_id")
    if not isinstance(app_id, str) or not isinstance(call_id, str):
        if marker.get("status") == "stopped":
            return 0 if marker.get("worker_status") == "passed" else 1
        raise ValueError("recovery requires the recorded app and call IDs")

    result_path = plan.output_dir / RESULT_NAME
    if result_path.exists() or result_path.is_symlink():
        try:
            if result_path.is_symlink() or not result_path.is_file():
                raise ValueError("mask preflight result path must be a regular file")
            cached_result = result_path.read_bytes()
            receipt = _validate_result(cached_result, plan)
        except Exception as exc:
            _verify_identity(plan.value["context"])
            _failure(plan, "cached_result_validation", exc, pending=False)
            _stop_and_confirm(plan, app_id)
            return 1
        result_status = receipt["status"]
        terminal_status = (
            "stopped"
            if marker.get("status") == "stopped"
            else "worker_returned"
            if result_status == "passed"
            else "worker_failed"
        )
        _update_attempt(
            _attempt_path(plan),
            status=terminal_status,
            worker_status=result_status,
        )
        if marker.get("status") == "stopped":
            return 0 if result_status == "passed" else 1
        _verify_identity(plan.value["context"])
        _stop_and_confirm(plan, app_id)
        return 0 if result_status == "passed" else 1

    if marker.get("status") == "stopped":
        return 0 if marker.get("worker_status") == "passed" else 1
    _verify_identity(plan.value["context"])
    modal = importlib.import_module("modal")
    if importlib.metadata.version("modal") != "1.6.1":
        raise RuntimeError("Modal SDK 1.6.1 is required")
    try:
        result = modal.FunctionCall.from_id(call_id).get(timeout=0)
    except BaseException as exc:
        if _is_terminal_call_error(exc, modal):
            _failure(plan, "modal_recovery_observation", exc, pending=False)
            _stop_and_confirm(plan, app_id)
        else:
            _update_attempt(
                _attempt_path(plan),
                status="observation_pending",
                last_observation_error=_remote().sanitize_failure(exc),
            )
        return 1
    if not isinstance(result, bytes):
        _update_attempt(_attempt_path(plan), status="observation_pending")
        return 1
    try:
        _save_or_confirm_result(result, result_path)
        receipt = _validate_result(result, plan)
    except Exception as exc:
        _failure(plan, "result_validation", exc, pending=False)
        _stop_and_confirm(plan, app_id)
        return 1
    result_status = receipt["status"]
    _update_attempt(
        _attempt_path(plan),
        status="worker_returned" if result_status == "passed" else "worker_failed",
        worker_status=result_status,
    )
    _stop_and_confirm(plan, app_id)
    return 0 if result_status == "passed" else 1


def _recover(plan: Plan) -> int:
    with _observer(plan, "recover"):
        return _recover_locked(plan)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = _validate_plan(args.plan, root=PROJECT_ROOT)
    if args.launch:
        return _launch(plan, clearance_path=args.clearance)
    if args.recover:
        return _recover(plan)
    print(
        json.dumps(
            {
                "status": "validated",
                "run_id": plan.value["run_id"],
                "payload_sha256": plan.payload_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
