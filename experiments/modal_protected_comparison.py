"""Guarded one-shot Modal controller for the protected-product comparison."""

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
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any

from experiments import protected_comparison_receipts as receipts
from product_scene_editor import protected_comparison as comparison
from product_scene_editor import remote

APP_NAME = "product-scene-editor-protected-comparison"
IMAGE = (
    "python:3.12.12-slim-bookworm@sha256:"
    "593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c"
)
PLAN_NAME = "protected-comparison-plan.json"
ATTEMPT_NAME = "protected-comparison-attempt.json"
OBSERVER_NAME = "protected-comparison-observer.json"
RESULT_NAME = "protected-comparison-result.zip"
UNVERIFIED_RESULT_NAME = ".protected-comparison-result-pending.zip"
MAX_PLAN_BYTES = 1024 * 1024
MAX_CLEARANCE_BYTES = 256 * 1024
MAX_EVIDENCE_BYTES = 128 * 1024
MAX_REVIEW_BYTES = 256 * 1024
STARTUP_TIMEOUT_SECONDS = 300
EXECUTION_TIMEOUT_SECONDS = 1800
MAX_CPU_CORES = 4
MAX_MEMORY_MIB = 65536
MAX_SHUTDOWN_OBSERVATIONS = 4
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
CONTEXT = {
    "profile": remote.PROFILE,
    "workspace": remote.WORKSPACE,
    "environment": remote.ENVIRONMENT,
}
RESOURCE_CONTRACT = {
    "gpu": "A100-80GB",
    "cpu_cores": MAX_CPU_CORES,
    "memory_mib": MAX_MEMORY_MIB,
    "startup_timeout_seconds": STARTUP_TIMEOUT_SECONDS,
    "execution_timeout_seconds": EXECUTION_TIMEOUT_SECONDS,
    "retries": 0,
}
PLAN_FIELDS = {
    "schema_version",
    "source_commit",
    "run_id",
    "payload_path",
    "payload_sha256",
    "output_dir",
    "context",
    "resources",
    "image",
}
CLEARANCE_FIELDS = {
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
    "source_signature_verified",
    "source_signature_reason",
    "checked_utc",
    "usage_rates_evidence_path",
    "usage_rates_evidence_sha256",
    "estimated_seconds",
    "estimated_compute_usd",
    "mask_step_reserve_usd",
    "allowance_usd",
    "cost_exclusions",
}
COST_EXCLUSIONS = ["image_build", "storage", "transfer", "shutdown_lag"]


@dataclass(frozen=True, slots=True)
class Plan:
    root: Path
    path: Path
    output_dir: Path
    payload_path: Path
    value: dict[str, Any]
    manifest: dict[str, object]
    payload: bytes
    payload_files: dict[str, bytes]
    payload_sha256: str
    plan_sha256: str


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _read_json(
    path: Path, role: str, *, max_bytes: int = MAX_CLEARANCE_BYTES
) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError(f"{role} is missing, unsafe, or oversized")
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        raise ValueError(f"{role} is oversized")
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_pairs, parse_constant=_reject_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{role} must be unique-key finite JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object")
    return value, raw


def _project_path(value: object, root: Path, role: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{role} must be a project-relative path")
    posix = PurePosixPath(value)
    if (
        posix.is_absolute()
        or posix.as_posix() != value
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise ValueError(f"{role} must stay inside the project")
    path = root / Path(*posix.parts)
    for parent in (path, *path.parents):
        if parent == root.parent:
            break
        if parent.is_symlink():
            raise ValueError(f"{role} may not traverse a symlink")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"{role} must stay inside the project")
    return path


def _load_plan(path: Path, *, root: Path) -> Plan:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_PLAN_BYTES:
        raise ValueError("protected comparison plan is missing, unsafe, or oversized")
    path = path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("protected comparison plan must stay inside the project")
    value, raw_plan = _read_json(path, "protected comparison plan", max_bytes=MAX_PLAN_BYTES)
    if (
        set(value) != PLAN_FIELDS
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != 1
    ):
        raise ValueError("protected comparison plan fields or schema version are invalid")
    source_commit, run_id, payload_sha = (
        value.get("source_commit"),
        value.get("run_id"),
        value.get("payload_sha256"),
    )
    if not isinstance(source_commit, str) or _SHA1.fullmatch(source_commit) is None:
        raise ValueError("plan source_commit must be a full lowercase Git SHA")
    if not isinstance(run_id, str) or _SAFE_ID.fullmatch(run_id) is None:
        raise ValueError("plan run_id must be a safe identifier")
    if not isinstance(payload_sha, str) or _SHA256.fullmatch(payload_sha) is None:
        raise ValueError("plan payload_sha256 must be a lowercase SHA-256 digest")
    if value.get("context") != CONTEXT or value.get("resources") != RESOURCE_CONTRACT:
        raise ValueError("plan differs from the frozen resource contract or Modal context")
    resources = value["resources"]
    if any(
        type(resources.get(key)) is not int
        for key in (
            "cpu_cores",
            "memory_mib",
            "startup_timeout_seconds",
            "execution_timeout_seconds",
            "retries",
        )
    ):
        raise ValueError("resource contract limits must use exact integer values")
    if value.get("image") != IMAGE:
        raise ValueError("plan image differs from the pinned Python OCI image")
    payload_path = _project_path(value.get("payload_path"), root, "payload_path")
    output_dir = _project_path(value.get("output_dir"), root, "output_dir")
    if not payload_path.is_file() or payload_path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("protected comparison payload is missing or oversized")
    payload = payload_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != payload_sha:
        raise ValueError("plan payload hash does not match its file")
    try:
        manifest, payload_files = comparison.validate_payload_bytes(payload)
    except Exception as exc:
        raise ValueError("protected comparison payload failed its frozen contract") from exc
    if manifest.get("source_sha") != source_commit or manifest.get("context") != CONTEXT:
        raise ValueError("plan source or context does not bind the validated payload")
    return Plan(
        root=root.resolve(),
        path=path,
        output_dir=output_dir,
        payload_path=payload_path,
        value=value,
        manifest=manifest,
        payload=payload,
        payload_files=payload_files,
        payload_sha256=payload_sha,
        plan_sha256=hashlib.sha256(raw_plan).hexdigest(),
    )


def _validate_clearance(plan: Plan, path: Path, *, now: datetime | None = None) -> dict[str, Any]:
    """Validate fresh signed-source, CI, usage, pricing, and exact-cost clearance."""
    clearance, _raw = _read_json(
        path, "protected comparison launch clearance", max_bytes=MAX_CLEARANCE_BYTES
    )
    if set(clearance) != CLEARANCE_FIELDS:
        raise ValueError("clearance fields are missing or unknown")
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
        "estimated_seconds": 2100,
        "mask_step_reserve_usd": 1,
        "allowance_usd": 8,
        "cost_exclusions": COST_EXCLUSIONS,
    }
    if type(clearance.get("schema_version")) is not int:
        raise ValueError("clearance schema_version must be an integer")
    if any(clearance.get(key) != value for key, value in expected.items()):
        raise ValueError("clearance does not match the plan, source, CI, or fixed cost boundary")
    if clearance.get("source_signature_verified") is not True:
        raise ValueError("clearance does not attest a verified source signature")
    if (
        type(clearance.get("ci_run_id")) is not int
        or clearance["ci_run_id"] <= 0
        or type(clearance.get("estimated_seconds")) is not int
        or type(clearance.get("mask_step_reserve_usd")) is not int
        or type(clearance.get("allowance_usd")) is not int
        or not isinstance(clearance.get("source_signature_reason"), str)
        or not clearance["source_signature_reason"].strip()
    ):
        raise ValueError("clearance must identify successful CI and verified source signature")
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
        raise ValueError("clearance or usage/price evidence is stale or from the future")

    evidence_path = _project_path(
        clearance.get("usage_rates_evidence_path"), plan.root, "usage/rates evidence"
    )
    evidence_sha = clearance.get("usage_rates_evidence_sha256")
    if not isinstance(evidence_sha, str) or _SHA256.fullmatch(evidence_sha) is None:
        raise ValueError("usage/rates evidence SHA-256 is invalid")
    evidence, raw_evidence = _read_json(
        evidence_path, "usage/rates evidence", max_bytes=MAX_EVIDENCE_BYTES
    )
    if hashlib.sha256(raw_evidence).hexdigest() != evidence_sha:
        raise ValueError("usage/rates evidence hash does not match clearance")
    if (
        evidence.get("checked_utc") != timestamp
        or evidence.get("context") != plan.value["context"]
        or evidence.get("active_apps") != []
    ):
        raise ValueError(
            "usage/rates evidence is stale, targets another account, or shows active apps"
        )
    price = evidence.get("price_observation")
    if not isinstance(price, dict) or (
        price.get("checked_utc") != timestamp
        or price.get("status") != "verified"
        or price.get("pricing_url") != "https://modal.com/pricing"
        or not isinstance(price.get("pricing_sha256"), str)
        or _SHA256.fullmatch(price["pricing_sha256"]) is None
    ):
        raise ValueError("fresh verified Modal price evidence is required")

    # Evidence binds successful local source-review and dataset-review results.
    review_fields = (
        ("source_pin_review_path", "source_pin_review_sha256", "source pin review"),
        ("data_pin_review_path", "data_pin_review_sha256", "data pin review"),
    )
    for path_key, digest_key, role in review_fields:
        review_path = _project_path(evidence.get(path_key), plan.root, role)
        digest = evidence.get(digest_key)
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(f"{role} SHA-256 is invalid")
        review, raw_review = _read_json(review_path, role, max_bytes=MAX_REVIEW_BYTES)
        if hashlib.sha256(raw_review).hexdigest() != digest:
            raise ValueError(f"{role} hash does not match usage/rates evidence")
        review_expected = {
            "status": "approved",
            "source_commit": plan.value["source_commit"],
            "plan_sha256": plan.plan_sha256,
            "payload_sha256": plan.payload_sha256,
        }
        if any(review.get(key) != value for key, value in review_expected.items()):
            raise ValueError(f"{role} does not bind this source, plan, and payload")
        if role == "data pin review" and (
            type(review.get("mask_count")) is not int
            or review["mask_count"] != 24
            or review.get("all_masks_accepted") is not True
            or review.get("pool_counts") != {"development": 6, "reserved": 18}
            or review.get("brand_family_disjoint") is not True
            or not isinstance(review.get("case_inventory"), list)
            or len(review["case_inventory"]) != 24
            or not isinstance(review.get("individual_review_references"), list)
            or len(review["individual_review_references"]) != 24
            or any(not item for item in review["case_inventory"])
            or any(not item for item in review["individual_review_references"])
        ):
            raise ValueError("data pin review does not certify the approved frozen data split")

    rates = evidence.get("rates")
    rate_keys = {
        "a10080_usd_per_second",
        "cpu_core_usd_per_second",
        "memory_gib_usd_per_second",
    }
    if not isinstance(rates, dict) or set(rates) != rate_keys:
        raise ValueError("usage/rates evidence does not contain the required resource rates")
    try:
        parsed_rates: dict[str, Decimal] = {}
        for key in rate_keys:
            value = rates[key]
            if type(value) not in {int, float}:
                raise ValueError("resource rates must be numeric")
            parsed = Decimal(str(value))
            if not parsed.is_finite() or parsed < 0:
                raise ValueError("resource rates must be finite and nonnegative")
            parsed_rates[key] = parsed
        computed = Decimal(2100) * (
            parsed_rates["a10080_usd_per_second"]
            + Decimal(MAX_CPU_CORES) * parsed_rates["cpu_core_usd_per_second"]
            + Decimal(MAX_MEMORY_MIB // 1024) * parsed_rates["memory_gib_usd_per_second"]
        )
        estimate = Decimal(str(clearance.get("estimated_compute_usd")))
    except (ArithmeticError, TypeError, ValueError) as exc:
        raise ValueError("usage/rates evidence has invalid compute rates or estimate") from exc
    if not estimate.is_finite() or estimate != computed:
        raise ValueError("clearance compute estimate does not match rates and fixed resources")
    if computed + Decimal(clearance["mask_step_reserve_usd"]) > Decimal(clearance["allowance_usd"]):
        raise ValueError("compute estimate and mask-step reserve exceed the cleared allowance")
    billing = evidence.get("billing")
    if (
        not isinstance(billing, dict)
        or billing.get("scope") != "workspace-delayed-not-project-invoice"
    ):
        raise ValueError("usage/rates evidence billing scope is unsupported")
    for key in ("workspace_gross_usd", "workspace_net_usd"):
        amount = billing.get(key)
        if not isinstance(amount, str):
            raise ValueError("workspace billing totals must be decimal strings")
        try:
            parsed = Decimal(amount)
        except ArithmeticError as exc:
            raise ValueError("workspace billing totals must be decimal strings") from exc
        if not parsed.is_finite() or parsed < 0:
            raise ValueError("workspace billing totals must be nonnegative finite decimals")
    return clearance


@contextmanager
def _observer(path: Path, *, run_id: str, action: str) -> Iterator[None]:
    """Take exclusive nonblocking ownership of a run before inspecting its attempt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("observer ownership file must not be a symlink")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    stream = os.fdopen(descriptor, "r+", encoding="utf-8")
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another observer owns this run") from exc
        owner = {
            "action": action,
            "pid": os.getpid(),
            "run_id": run_id,
            "started_utc": datetime.now(UTC).isoformat(),
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


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False, timeout=30
    )
    if completed.returncode:
        raise ValueError("could not verify local source state")
    return completed.stdout.strip()


def _verify_source(plan: Plan) -> None:
    if _git(plan.root, "rev-parse", "HEAD") != plan.value["source_commit"]:
        raise ValueError("HEAD differs from the source commit frozen into the plan")
    if _git(plan.root, "rev-parse", "origin/main") != plan.value["source_commit"]:
        raise ValueError("cached origin/main differs from the source commit frozen into the plan")
    if _git(
        plan.root,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        "src/product_scene_editor",
        "experiments",
    ):
        raise ValueError("source and experiment directories must be clean before launch")


def _write_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _attempt_path(plan: Plan) -> Path:
    return plan.output_dir / ATTEMPT_NAME


def _reserve_attempt(plan: Plan) -> Path:
    _write_once(
        _attempt_path(plan),
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
        },
    )
    return _attempt_path(plan)


def _update_attempt(path: Path, **updates: Any) -> dict[str, Any]:
    marker, _raw = _read_json(path, "protected comparison attempt")
    for key in ("app_id", "call_id", "private_result_sha256"):
        if key in updates and marker.get(key) is not None and marker[key] != updates[key]:
            raise ValueError(f"attempt marker {key} is immutable")
    if "spawn_count" in updates and (
        type(updates["spawn_count"]) is not int
        or updates["spawn_count"] not in {marker.get("spawn_count"), 1}
    ):
        raise ValueError("protected comparison permits one reserved Modal call")
    marker.update(updates)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(marker, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return marker


def _read_attempt(plan: Plan) -> dict[str, Any]:
    marker, _raw = _read_json(_attempt_path(plan), "protected comparison attempt")
    if (
        any(
            marker.get(key) != expected
            for key, expected in {
                "schema_version": 1,
                "run_id": plan.value["run_id"],
                "source_commit": plan.value["source_commit"],
                "payload_sha256": plan.payload_sha256,
                "plan_sha256": plan.plan_sha256,
                "context": plan.value["context"],
            }.items()
        )
        or type(marker.get("spawn_count")) is not int
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
        raise ValueError("attempt marker does not bind this plan or valid one-shot state")
    for key in ("app_id", "call_id"):
        value = marker.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"attempt marker {key} is invalid")
    result_sha = marker.get("private_result_sha256")
    if result_sha is not None and (
        not isinstance(result_sha, str) or _SHA256.fullmatch(result_sha) is None
    ):
        raise ValueError("attempt marker private result hash is invalid")
    if marker.get("call_id") is not None and marker.get("spawn_count") != 1:
        raise ValueError("recorded Modal call has no reserved spawn")
    return marker


def _build_image(modal: Any, root: Path) -> Any:
    packages = [
        f"{name}=={version}" for name, version in remote.RUNTIME_PINS.items() if name != "imageio"
    ]
    packages.append(f"imageio[ffmpeg]=={remote.RUNTIME_PINS['imageio']}")
    image = (
        modal.Image.from_registry(IMAGE)
        .env({"USE_HUB_KERNELS": "NO"})
        .apt_install("git", "build-essential")
        .run_commands("cc --version && c++ --version && make --version")
        .pip_install(*packages, extra_index_url=remote.TORCH_INDEX)
        .run_commands(
            "git clone --filter=blob:none --no-checkout "
            "https://github.com/modelscope/DiffSynth-Studio.git /opt/DiffSynth-Studio",
            f"git -C /opt/DiffSynth-Studio checkout --detach {remote.DIFFSYNTH_COMMIT} "
            f'&& test "$(git -C /opt/DiffSynth-Studio rev-parse HEAD)" = '
            f"{remote.DIFFSYNTH_COMMIT} "
            "&& pip install --no-deps /opt/DiffSynth-Studio",
        )
        .add_local_python_source("product_scene_editor", copy=True)
    )
    if not (root / "src/product_scene_editor/protected_comparison_inference.py").is_file():
        raise FileNotFoundError("protected comparison worker source is missing")
    return image


def _worker(payload: bytes, run_id: str) -> bytes:
    from product_scene_editor.protected_comparison_inference import (
        run_remote_protected_comparison_inference,
    )

    return run_remote_protected_comparison_inference(payload, run_id)


def _save_private_result(data: bytes, path: Path) -> None:
    if path.is_symlink():
        raise ValueError("private result path must not be a symlink")
    try:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if not path.is_file() or path.read_bytes() != data:
            raise ValueError("recovered result differs from the private saved result") from None


def _matching_app_rows(output: str, app_id: str) -> list[dict[str, Any]]:
    try:
        apps = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(apps, list):
        return []
    found = [
        {key: row.get(key) for key in ("app_id", "state", "tasks")}
        for row in apps
        if isinstance(row, dict) and row.get("app_id") == app_id
    ]
    return found


def _app_list_confirms_stopped(output: str, app_id: str) -> bool:
    found = _matching_app_rows(output, app_id)
    return len(found) == 1 and found[0].get("state") == "stopped" and found[0].get("tasks") == "0"


def _event(plan: Plan, name: str, details: dict[str, Any]) -> None:
    stamp = time.time_ns()
    _write_once(
        plan.output_dir / f"protected-comparison-{name}-{stamp}.json",
        {"schema_version": 1, "run_id": plan.value["run_id"], **details},
    )


def _record_error(plan: Plan, stage: str, error: BaseException) -> None:
    _event(
        plan,
        "failure",
        {
            "stage": stage,
            "error_type": type(error).__name__,
            "message": remote.sanitize_failure(error),
        },
    )


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
    _update_attempt(
        marker_path, status="shutdown_pending", stop_requested_at_utc=datetime.now(UTC).isoformat()
    )
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
        _event(plan, "stop", {"app_id": app_id, "returncode": stopped.returncode})
    except Exception as exc:
        _event(plan, "stop", {"app_id": app_id, "error": remote.sanitize_failure(exc)})
    for index in range(1, MAX_SHUTDOWN_OBSERVATIONS + 1):
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
            confirmed = observed.returncode == 0 and _app_list_confirms_stopped(
                observed.stdout, app_id
            )
            _event(
                plan,
                "app-list",
                {
                    "app_id": app_id,
                    "call_id": _read_attempt(plan).get("call_id"),
                    "source_commit": plan.value["source_commit"],
                    "payload_sha256": plan.payload_sha256,
                    "observed_at_utc": datetime.now(UTC).isoformat(),
                    "returncode": observed.returncode,
                    "observation": index,
                    "matching_apps": _matching_app_rows(observed.stdout, app_id),
                    "stdout_sha256": hashlib.sha256(observed.stdout.encode()).hexdigest(),
                    "stderr_sha256": hashlib.sha256(observed.stderr.encode()).hexdigest(),
                    "confirmed_stopped": confirmed,
                },
            )
        except Exception as exc:
            confirmed = False
            _event(
                plan,
                "app-list",
                {
                    "app_id": app_id,
                    "call_id": _read_attempt(plan).get("call_id"),
                    "source_commit": plan.value["source_commit"],
                    "payload_sha256": plan.payload_sha256,
                    "observed_at_utc": datetime.now(UTC).isoformat(),
                    "observation": index,
                    "error": remote.sanitize_failure(exc),
                },
            )
        if confirmed:
            _update_attempt(
                marker_path, status="stopped", stopped_at_utc=datetime.now(UTC).isoformat()
            )
            return
        if index < MAX_SHUTDOWN_OBSERVATIONS:
            time.sleep(2)
    raise RuntimeError("Modal did not confirm the exact app stopped with zero tasks")


def _publish_stopped_result(plan: Plan) -> None:
    pending = plan.output_dir / UNVERIFIED_RESULT_NAME
    result = plan.output_dir / RESULT_NAME
    if pending.exists():
        if pending.is_symlink() or not pending.is_file():
            raise ValueError("private result path is unsafe")
        if result.exists() or result.is_symlink():
            raise FileExistsError("public result already exists before private result promotion")
        os.replace(pending, result)


def _finish_result(plan: Plan, app_id: str, result: bytes) -> int:
    private_path = plan.output_dir / UNVERIFIED_RESULT_NAME
    try:
        _save_private_result(result, private_path)
        _update_attempt(
            _attempt_path(plan), private_result_sha256=hashlib.sha256(result).hexdigest()
        )
        receipt = receipts.validate_result_bundle(
            result,
            run_id=plan.value["run_id"],
            payload_sha256=plan.payload_sha256,
            source_commit=plan.value["source_commit"],
            manifest=plan.manifest,
            payload_files=plan.payload_files,
        )
    except Exception as exc:
        _record_error(plan, "result_validation", exc)
        _update_attempt(_attempt_path(plan), status="worker_failed")
        _stop_and_confirm(plan, app_id)
        return 1
    status = "worker_returned" if receipt["status"] == "passed" else "worker_failed"
    _update_attempt(_attempt_path(plan), status=status, worker_status=receipt["status"])
    _stop_and_confirm(plan, app_id)
    _publish_stopped_result(plan)
    return 0 if receipt["status"] == "passed" else 1


def _is_terminal_call_error(error: BaseException, modal: Any | None) -> bool:
    exceptions = getattr(modal, "exception", None)
    types = tuple(
        value
        for value in (
            getattr(exceptions, "FunctionTimeoutError", None),
            getattr(exceptions, "InternalFailure", None),
        )
        if isinstance(value, type)
    )
    return bool(types) and isinstance(error, types)


def _launch_locked(plan: Plan, clearance_path: Path) -> int:
    for name in (ATTEMPT_NAME, RESULT_NAME, UNVERIFIED_RESULT_NAME):
        path = plan.output_dir / name
        if path.exists() or path.is_symlink():
            raise FileExistsError("protected comparison already has attempt or result evidence")
    _verify_source(plan)
    _validate_clearance(plan, clearance_path)
    remote.validate_launch_environment(plan.value["context"])
    remote.verify_modal_identity(**plan.value["context"])
    attempt_path = _reserve_attempt(plan)
    app_id: str | None = None
    phase = "modal_import"
    modal: Any | None = None
    try:
        os.environ["MODAL_PROFILE"] = plan.value["context"]["profile"]
        modal = importlib.import_module("modal")
        if importlib.metadata.version("modal") != "1.6.1":
            raise RuntimeError("Modal SDK 1.6.1 is required")
        app = modal.App(APP_NAME, image=_build_image(modal, plan.root))
        run = app.function(
            gpu="A100-80GB",
            cpu=(4, 4),
            memory=(65536, 65536),
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
        with (
            modal.enable_output(),
            app.run(environment_name=plan.value["context"]["environment"], detach=True),
        ):
            app_id = getattr(app, "app_id", None) or getattr(app, "object_id", None)
            if not isinstance(app_id, str) or not app_id:
                raise RuntimeError("Modal app ID was unavailable before the one-shot call")
            _update_attempt(attempt_path, status="app_started", app_id=app_id)
            _update_attempt(attempt_path, status="spawn_reserved", spawn_count=1)
            phase = "modal_spawn"
            call = run.spawn(plan.payload, plan.value["run_id"])
            call_id = getattr(call, "object_id", None) or getattr(call, "call_id", None)
            if not isinstance(call_id, str) or not call_id:
                raise RuntimeError(
                    "Modal call ID was unavailable after spawn; recovery cannot respawn"
                )
            _update_attempt(attempt_path, status="observing", call_id=call_id)
            phase = "modal_observation"
            result = call.get(timeout=STARTUP_TIMEOUT_SECONDS + EXECUTION_TIMEOUT_SECONDS)
    except BaseException as exc:
        marker = _read_attempt(plan)
        reserved = marker.get("spawn_count") == 1
        _record_error(plan, phase, exc)
        if reserved and marker.get("call_id") is None:
            _update_attempt(
                attempt_path, status="worker_failed", last_error=remote.sanitize_failure(exc)
            )
            if marker.get("app_id"):
                _stop_and_confirm(plan, marker["app_id"])
            return 1
        if reserved and (phase != "modal_observation" or not _is_terminal_call_error(exc, modal)):
            _update_attempt(
                attempt_path, status="observation_pending", last_error=remote.sanitize_failure(exc)
            )
            return 1
        _update_attempt(
            attempt_path, status="worker_failed", last_error=remote.sanitize_failure(exc)
        )
        if marker.get("app_id"):
            _stop_and_confirm(plan, marker["app_id"])
        return 1
    if not isinstance(result, bytes):
        error = ValueError("Modal call did not return result bytes")
        _record_error(plan, "modal_observation", error)
        _update_attempt(
            attempt_path, status="worker_failed", last_error=remote.sanitize_failure(error)
        )
        _stop_and_confirm(plan, app_id)
        return 1
    return _finish_result(plan, app_id, result)


def _launch(plan: Plan, clearance_path: Path) -> int:
    with _observer(plan.output_dir / OBSERVER_NAME, run_id=plan.value["run_id"], action="launch"):
        return _launch_locked(plan, clearance_path)


def _recover_locked(plan: Plan) -> int:
    marker = _read_attempt(plan)
    app_id, call_id = marker.get("app_id"), marker.get("call_id")
    private_path = plan.output_dir / UNVERIFIED_RESULT_NAME
    has_private_result = private_path.exists() or private_path.is_symlink()
    if marker.get("spawn_count") == 1 and call_id is None and not has_private_result:
        app_id = marker.get("app_id")
        error = RuntimeError("one-shot call was reserved but its call ID was not recorded")
        _record_error(plan, "missing_call_id", error)
        if isinstance(app_id, str):
            _stop_and_confirm(plan, app_id)
        else:
            print(json.dumps({"status": "manual_stop_required", "app_id": None}))
        return 1
    if private_path.exists() or private_path.is_symlink():
        try:
            if private_path.is_symlink() or not private_path.is_file():
                raise ValueError("private saved result must be a regular file")
            data = private_path.read_bytes()
            result_sha = hashlib.sha256(data).hexdigest()
            if marker.get("private_result_sha256") not in {None, result_sha}:
                raise ValueError("cached private result hash differs from the attempt marker")
            if marker.get("private_result_sha256") is None:
                _update_attempt(_attempt_path(plan), private_result_sha256=result_sha)
            receipt = receipts.validate_result_bundle(
                data,
                run_id=plan.value["run_id"],
                payload_sha256=plan.payload_sha256,
                source_commit=plan.value["source_commit"],
                manifest=plan.manifest,
                payload_files=plan.payload_files,
            )
        except Exception as exc:
            _record_error(plan, "cached_result_validation", exc)
            if marker.get("status") != "stopped":
                _update_attempt(_attempt_path(plan), status="worker_failed")
                if isinstance(app_id, str):
                    _stop_and_confirm(plan, app_id)
                else:
                    print(json.dumps({"status": "manual_stop_required", "app_id": None}))
            return 1
        if marker.get("status") == "stopped":
            if marker.get("worker_status") != receipt["status"]:
                raise ValueError("stopped attempt status differs from its validated saved result")
        else:
            _update_attempt(
                _attempt_path(plan),
                status="worker_returned" if receipt["status"] == "passed" else "worker_failed",
                worker_status=receipt["status"],
            )
            if not isinstance(app_id, str):
                print(json.dumps({"status": "manual_stop_required", "app_id": None}))
                return 1
            _stop_and_confirm(plan, app_id)
        _publish_stopped_result(plan)
        return 0 if receipt["status"] == "passed" else 1
    if not isinstance(app_id, str) or not isinstance(call_id, str):
        if marker.get("status") == "stopped":
            return 0 if marker.get("worker_status") == "passed" else 1
        raise ValueError("recovery requires the recorded app and call IDs")
    if marker.get("status") == "stopped":
        return 0 if marker.get("worker_status") == "passed" else 1
    remote.verify_modal_identity(**plan.value["context"])
    modal = importlib.import_module("modal")
    if importlib.metadata.version("modal") != "1.6.1":
        raise RuntimeError("Modal SDK 1.6.1 is required")
    try:
        result = modal.FunctionCall.from_id(call_id).get(timeout=0)
    except BaseException as exc:
        if _is_terminal_call_error(exc, modal):
            _record_error(plan, "modal_recovery_observation", exc)
            _update_attempt(
                _attempt_path(plan), status="worker_failed", last_error=remote.sanitize_failure(exc)
            )
            _stop_and_confirm(plan, app_id)
        else:
            _record_error(plan, "modal_recovery_observation", exc)
            _update_attempt(
                _attempt_path(plan),
                status="observation_pending",
                last_error=remote.sanitize_failure(exc),
            )
        return 1
    if not isinstance(result, bytes):
        _update_attempt(_attempt_path(plan), status="observation_pending")
        return 1
    return _finish_result(plan, app_id, result)


def _recover(plan: Plan) -> int:
    with _observer(plan.output_dir / OBSERVER_NAME, run_id=plan.value["run_id"], action="recover"):
        return _recover_locked(plan)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--launch", action="store_true")
    group.add_argument("--recover", action="store_true")
    parser.add_argument("--clearance", type=Path, required=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    plan = _load_plan(args.plan, root=root)
    if args.launch:
        clearance = args.clearance or root / ".context/protected-comparison-clearance.json"
        return _launch(plan, clearance)
    if args.recover:
        return _recover(plan)
    print(
        json.dumps(
            {
                "status": "validated",
                "run_id": plan.value["run_id"],
                "payload_sha256": plan.payload_sha256,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
