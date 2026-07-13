"""Fail-closed launch authorization for the Full PETAL route."""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
from pathlib import Path
import subprocess


LAUNCH_CONTRACT_SCHEMA = "full-petal-launch-contract-v1"
LAUNCH_TICKET_SCHEMA = "full-petal-launch-ticket-v1"
B0_SCHEMA = "full-petal-b0-v1"
B0_TEST_REPORT_SCHEMA = "full-petal-b0-test-report-v1"
B0_AUDIT_REPORT_SCHEMA = "full-petal-b0-audit-report-v1"
REVIEW_SCHEMA = "full-petal-independent-review-v1"
PROFILE_SCHEMA = "full-petal-fixed-step-profile-v1"
PROFILE_MODE = "profile"
FORMAL_MODE = "formal"

_SHA256_ALPHABET = frozenset("0123456789abcdef")
_SCIENTIFIC_DIGEST_EXCLUSIONS = {
    "formal_training_ready",
    "gpu_authorization",
    "launch_contract",
    "work_dir",
}
_RESOLVED_DIGEST_EXCLUSIONS = {"work_dir"}


class FullPetalLaunchError(RuntimeError):
    """Raised before CUDA/DDP initialization when authorization is incomplete."""


@dataclass(frozen=True)
class RepositoryState:
    commit_sha: str
    clean: bool
    status: str = ""


@dataclass(frozen=True)
class FullPetalLaunchAuthorization:
    mode: str
    commit_sha: str
    resolved_config_sha256: str
    scientific_config_sha256: str
    ticket_path: str
    ticket_sha256: str
    b0_artifact_sha256: str
    review_artifact_sha256: str
    profile_artifact_sha256: str | None
    warmup_optimizer_events: int
    measured_optimizer_events: int
    world_size: int
    slurm_job_id: str

    @property
    def total_optimizer_events(self):
        return self.warmup_optimizer_events + self.measured_optimizer_events


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise FullPetalLaunchError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value):
    raise FullPetalLaunchError(f"non-finite JSON number is forbidden: {value}")


def load_strict_json(path):
    path = Path(path)
    if not path.is_file():
        raise FullPetalLaunchError(f"evidence file does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite,
            )
    except FullPetalLaunchError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FullPetalLaunchError(f"failed to load evidence file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FullPetalLaunchError(f"evidence file must contain one JSON object: {path}")
    return payload


def sha256_file(path, chunk_size=1024 * 1024):
    path = Path(path)
    if not path.is_file():
        raise FullPetalLaunchError(f"file does not exist: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise FullPetalLaunchError(f"failed to hash file {path}: {exc}") from exc
    return digest.hexdigest()


def _json_value(value):
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise FullPetalLaunchError("config contains a non-finite number")
        return value
    raise FullPetalLaunchError(
        f"config contains a non-JSON value of type {type(value).__name__}"
    )


def canonical_json_sha256(value):
    try:
        encoded = json.dumps(
            _json_value(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FullPetalLaunchError(f"value is not canonical JSON: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def resolved_config_sha256(cfg, *, scientific=False):
    payload = _json_value(cfg)
    exclusions = (
        _SCIENTIFIC_DIGEST_EXCLUSIONS
        if scientific
        else _RESOLVED_DIGEST_EXCLUSIONS
    )
    for key in exclusions:
        payload.pop(key, None)
    return canonical_json_sha256(payload)


def _cfg_get(cfg, key, default=None):
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _nested_get(cfg, *keys, default=None):
    current = cfg
    for key in keys:
        current = _cfg_get(current, key, None)
        if current is None:
            return default
    return current


def is_full_petal_route(cfg):
    return (
        _cfg_get(cfg, "route_stage") == "q2_persistent_binding_one_factor"
        or _nested_get(cfg, "model", "type") == "PersistentTrajectoryOnlineDetector"
        or _nested_get(cfg, "launch_contract", "schema_version")
        == LAUNCH_CONTRACT_SCHEMA
    )


def _valid_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _SHA256_ALPHABET
    )


def _valid_git_sha(value):
    return (
        isinstance(value, str)
        and len(value) == 40
        and set(value) <= _SHA256_ALPHABET
    )


def _require_sha256(value, label):
    if not _valid_sha256(value):
        raise FullPetalLaunchError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_git_sha(value, label):
    if not _valid_git_sha(value):
        raise FullPetalLaunchError(f"{label} must be a 40-character lowercase git SHA")
    return value


def _require_nonnegative_int(value, label, *, positive=False):
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise FullPetalLaunchError(f"{label} must be a {qualifier} integer")
    return value


def _require_positive_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FullPetalLaunchError(f"{label} must be a positive finite number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise FullPetalLaunchError(f"{label} must be a positive finite number")
    return value


def _require_exact_fields(payload, expected, label):
    found = set(payload)
    expected = set(expected)
    if found != expected:
        raise FullPetalLaunchError(
            f"{label} fields differ; missing={sorted(expected - found)}, "
            f"extra={sorted(found - expected)}"
        )


def _resolve_path(value, base_dir, label):
    if not isinstance(value, str) or not value.strip():
        raise FullPetalLaunchError(f"{label} path must be non-empty text")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path.resolve()


def _load_reference(reference, base_dir, label):
    if not isinstance(reference, dict):
        raise FullPetalLaunchError(f"{label} must be a path/hash object")
    _require_exact_fields(reference, {"path", "sha256"}, label)
    path = _resolve_path(reference["path"], base_dir, label)
    expected = _require_sha256(reference["sha256"], f"{label}.sha256")
    actual = sha256_file(path)
    if not hmac.compare_digest(actual, expected):
        raise FullPetalLaunchError(
            f"{label} hash mismatch: expected {expected}, found {actual}"
        )
    return load_strict_json(path), path, actual


def _file_reference(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def inspect_repository(repository_root):
    repository_root = Path(repository_root).resolve()
    try:
        commit = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repository_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FullPetalLaunchError(
            f"failed to inspect git repository {repository_root}: {exc}"
        ) from exc
    _require_git_sha(commit, "repository commit")
    return RepositoryState(commit_sha=commit, clean=not bool(status), status=status)


def _validate_b0_artifact(reference, ticket_dir, commit_sha):
    artifact, path, digest = _load_reference(reference, ticket_dir, "B0 evidence")
    _require_exact_fields(
        artifact,
        {
            "schema_version",
            "status",
            "commit_sha",
            "test_count",
            "blocking_findings",
            "protocol_violations",
            "test_report_path",
            "test_report_sha256",
            "audit_report_path",
            "audit_report_sha256",
        },
        "B0 artifact",
    )
    if artifact["schema_version"] != B0_SCHEMA:
        raise FullPetalLaunchError("B0 artifact schema is unsupported")
    if artifact["commit_sha"] != commit_sha:
        raise FullPetalLaunchError("B0 artifact does not bind the launch commit")
    test_count = _require_nonnegative_int(
        artifact["test_count"], "B0.test_count", positive=True
    )
    blockers = _require_nonnegative_int(
        artifact["blocking_findings"], "B0.blocking_findings"
    )
    violations = _require_nonnegative_int(
        artifact["protocol_violations"], "B0.protocol_violations"
    )

    test_ref = {
        "path": artifact["test_report_path"],
        "sha256": artifact["test_report_sha256"],
    }
    audit_ref = {
        "path": artifact["audit_report_path"],
        "sha256": artifact["audit_report_sha256"],
    }
    test_report, _, _ = _load_reference(test_ref, path.parent, "B0 test report")
    audit_report, _, _ = _load_reference(audit_ref, path.parent, "B0 audit report")
    _require_exact_fields(
        test_report,
        {
            "schema_version",
            "status",
            "commit_sha",
            "collected",
            "passed",
            "failed",
            "errors",
            "skipped",
            "suites",
        },
        "B0 test report",
    )
    _require_exact_fields(
        audit_report,
        {
            "schema_version",
            "status",
            "commit_sha",
            "blocking_findings",
            "protocol_violations",
            "checks",
        },
        "B0 audit report",
    )
    suites = test_report["suites"]
    if not isinstance(suites, list) or not suites:
        raise FullPetalLaunchError("B0 test report requires at least one suite")
    suite_names = set()
    suite_totals = {
        "collected": 0,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
    }
    for index, suite in enumerate(suites):
        label = f"B0 test suite {index}"
        if not isinstance(suite, dict):
            raise FullPetalLaunchError(f"{label} must be an object")
        _require_exact_fields(
            suite,
            {
                "name",
                "status",
                "command",
                "python_executable",
                "collected",
                "passed",
                "failed",
                "errors",
                "skipped",
                "log_path",
                "log_sha256",
                "junit_path",
                "junit_sha256",
            },
            label,
        )
        name = suite["name"]
        if not isinstance(name, str) or not name.strip() or name in suite_names:
            raise FullPetalLaunchError("B0 test suite names must be non-empty and unique")
        suite_names.add(name)
        command = suite["command"]
        if not isinstance(command, list) or not command or not all(
            isinstance(item, str) and item for item in command
        ):
            raise FullPetalLaunchError(f"{label} command must be a non-empty argv list")
        if not isinstance(suite["python_executable"], str) or not suite[
            "python_executable"
        ].strip():
            raise FullPetalLaunchError(f"{label} requires a Python executable")
        counts = {
            field: _require_nonnegative_int(suite[field], f"{label}.{field}")
            for field in suite_totals
        }
        if counts["collected"] != sum(
            counts[field] for field in ("passed", "failed", "errors", "skipped")
        ):
            raise FullPetalLaunchError(f"{label} test counts are inconsistent")
        if (
            suite["status"] != "PASS"
            or counts["failed"]
            or counts["errors"]
            or counts["skipped"]
        ):
            raise FullPetalLaunchError(f"{label} has not reached PASS")
        for kind in ("log", "junit"):
            evidence_path = _resolve_path(
                suite[f"{kind}_path"], path.parent, f"{label} {kind}"
            )
            expected_hash = _require_sha256(
                suite[f"{kind}_sha256"], f"{label}.{kind}_sha256"
            )
            if not hmac.compare_digest(sha256_file(evidence_path), expected_hash):
                raise FullPetalLaunchError(f"{label} {kind} hash mismatch")
        for field, count in counts.items():
            suite_totals[field] += count

    report_totals = {
        field: _require_nonnegative_int(
            test_report[field], f"B0 test report.{field}"
        )
        for field in suite_totals
    }
    if report_totals != suite_totals:
        raise FullPetalLaunchError("B0 test report totals differ from suite totals")

    checks = audit_report["checks"]
    if not isinstance(checks, list) or not checks:
        raise FullPetalLaunchError("B0 audit report requires machine-verifiable checks")
    check_names = set()
    for index, check in enumerate(checks):
        label = f"B0 audit check {index}"
        if not isinstance(check, dict):
            raise FullPetalLaunchError(f"{label} must be an object")
        _require_exact_fields(
            check,
            {"name", "status", "command", "log_path", "log_sha256"},
            label,
        )
        name = check["name"]
        if not isinstance(name, str) or not name.strip() or name in check_names:
            raise FullPetalLaunchError("B0 audit check names must be non-empty and unique")
        check_names.add(name)
        if check["status"] != "PASS":
            raise FullPetalLaunchError(f"{label} has not reached PASS")
        if not isinstance(check["command"], list) or not check["command"] or not all(
            isinstance(item, str) and item for item in check["command"]
        ):
            raise FullPetalLaunchError(f"{label} command must be a non-empty argv list")
        check_log = _resolve_path(check["log_path"], path.parent, f"{label} log")
        expected_hash = _require_sha256(
            check["log_sha256"], f"{label}.log_sha256"
        )
        if not hmac.compare_digest(sha256_file(check_log), expected_hash):
            raise FullPetalLaunchError(f"{label} log hash mismatch")

    test_passed = (
        test_report["schema_version"] == B0_TEST_REPORT_SCHEMA
        and test_report["status"] == "PASS"
        and test_report["commit_sha"] == commit_sha
        and report_totals["collected"] == test_count
        and report_totals["passed"] == test_count
        and report_totals["failed"] == 0
        and report_totals["errors"] == 0
        and report_totals["skipped"] == 0
    )
    audit_passed = (
        audit_report["schema_version"] == B0_AUDIT_REPORT_SCHEMA
        and audit_report["status"] == "PASS"
        and audit_report["commit_sha"] == commit_sha
        and _require_nonnegative_int(
            audit_report["blocking_findings"], "B0 audit blockers"
        )
        == blockers
        == 0
        and _require_nonnegative_int(
            audit_report["protocol_violations"], "B0 audit violations"
        )
        == violations
        == 0
    )
    if artifact["status"] != "PASS" or not test_passed or not audit_passed:
        raise FullPetalLaunchError("B0 evidence chain has not reached PASS")
    return artifact, path, digest


def _validate_review_artifact(
    reference,
    ticket_dir,
    *,
    commit_sha,
    b0_sha256,
    reviewer_id,
    required_scope,
):
    review, path, digest = _load_reference(
        reference, ticket_dir, "independent review evidence"
    )
    _require_exact_fields(
        review,
        {
            "schema_version",
            "reviewer_id",
            "reviewed_commit",
            "b0_artifact_sha256",
            "scope",
            "verdict",
            "blocking_findings",
            "protocol_violations",
        },
        "independent review artifact",
    )
    if review["schema_version"] != REVIEW_SCHEMA:
        raise FullPetalLaunchError("independent review schema is unsupported")
    if review["reviewer_id"] != reviewer_id:
        raise FullPetalLaunchError("independent review was not issued by the locked reviewer")
    if review["reviewed_commit"] != commit_sha:
        raise FullPetalLaunchError("independent review does not bind the launch commit")
    if review["b0_artifact_sha256"] != b0_sha256:
        raise FullPetalLaunchError("independent review does not bind the verified B0 artifact")
    if review["scope"] != list(required_scope):
        raise FullPetalLaunchError("independent review scope is incomplete or reordered")
    if not isinstance(review["blocking_findings"], list) or review["blocking_findings"]:
        raise FullPetalLaunchError("independent review still contains blocking findings")
    if not isinstance(review["protocol_violations"], list) or review["protocol_violations"]:
        raise FullPetalLaunchError("independent review still contains protocol violations")
    if review["verdict"] != "PASS":
        raise FullPetalLaunchError("independent reviewer verdict is not PASS")
    return review, path, digest


def _profile_dimensions(cfg):
    return {
        "batch_size": _nested_get(cfg, "solver", "train", "batch_size"),
        "chunk_size": _cfg_get(cfg, "chunk_size"),
        "memory_size": _cfg_get(cfg, "memory_size"),
        "slot_count": _cfg_get(cfg, "num_slots"),
    }


def _validate_profile_artifact(
    reference,
    ticket_dir,
    *,
    scientific_config_sha256,
    warmup_events,
    measured_events,
    world_size,
    dimensions,
    reviewer_id,
    required_scope,
):
    profile, path, digest = _load_reference(
        reference, ticket_dir, "fixed-step profile evidence"
    )
    _require_exact_fields(
        profile,
        {
            "schema_version",
            "status",
            "commit_sha",
            "resolved_config_sha256",
            "scientific_config_sha256",
            "launch_ticket_path",
            "launch_ticket_sha256",
            "slurm_job_id",
            "world_size",
            "precision",
            "hardware",
            "dimensions",
            "measurements",
        },
        "fixed-step profile artifact",
    )
    if profile["schema_version"] != PROFILE_SCHEMA or profile["status"] != "PASS":
        raise FullPetalLaunchError("fixed-step profile artifact has not reached PASS")
    profile_commit = _require_git_sha(
        profile["commit_sha"], "fixed-step profile commit"
    )
    profile_resolved_digest = _require_sha256(
        profile["resolved_config_sha256"], "profile resolved config hash"
    )
    if profile["scientific_config_sha256"] != scientific_config_sha256:
        raise FullPetalLaunchError("fixed-step profile used a different scientific config")
    profile_ticket, profile_ticket_path, _ = _load_reference(
        {
            "path": profile["launch_ticket_path"],
            "sha256": profile["launch_ticket_sha256"],
        },
        path.parent,
        "profile launch ticket",
    )
    _require_exact_fields(profile_ticket, _ticket_fields(), "profile launch ticket")
    if profile_ticket["schema_version"] != LAUNCH_TICKET_SCHEMA:
        raise FullPetalLaunchError("profile launch ticket schema is unsupported")
    if profile_ticket["mode"] != PROFILE_MODE:
        raise FullPetalLaunchError("fixed-step profile did not use a profile-mode ticket")
    if profile_ticket["commit_sha"] != profile_commit:
        raise FullPetalLaunchError("profile launch ticket commit differs from the profile")
    _require_sha256(
        profile_ticket["config_file_sha256"],
        "profile launch ticket config file hash",
    )
    if profile_ticket["resolved_config_sha256"] != profile_resolved_digest:
        raise FullPetalLaunchError("profile launch ticket resolved config hash differs")
    if profile_ticket["scientific_config_sha256"] != scientific_config_sha256:
        raise FullPetalLaunchError("profile launch ticket scientific config hash differs")
    if profile_ticket["profile_evidence"] is not None:
        raise FullPetalLaunchError("profile launch ticket cannot contain profile evidence")
    _, _, profile_b0_digest = _validate_b0_artifact(
        profile_ticket["b0_evidence"], profile_ticket_path.parent, profile_commit
    )
    _validate_review_artifact(
        profile_ticket["review_evidence"],
        profile_ticket_path.parent,
        commit_sha=profile_commit,
        b0_sha256=profile_b0_digest,
        reviewer_id=reviewer_id,
        required_scope=required_scope,
    )
    if not isinstance(profile["slurm_job_id"], str) or not profile["slurm_job_id"].strip():
        raise FullPetalLaunchError("fixed-step profile lacks a Slurm job ID")
    if profile["world_size"] != world_size:
        raise FullPetalLaunchError("fixed-step profile world size differs from the contract")
    if profile["precision"] not in {"fp32", "fp16", "bf16"}:
        raise FullPetalLaunchError("fixed-step profile precision is unsupported")
    hardware = profile["hardware"]
    if not isinstance(hardware, dict):
        raise FullPetalLaunchError("fixed-step profile hardware must be an object")
    _require_exact_fields(
        hardware,
        {"gpu_name", "torch_version", "cuda_version"},
        "fixed-step profile hardware",
    )
    for field in ("gpu_name", "torch_version"):
        if not isinstance(hardware[field], str) or not hardware[field].strip():
            raise FullPetalLaunchError(f"fixed-step profile {field} must be non-empty")
    if hardware["cuda_version"] is not None and (
        not isinstance(hardware["cuda_version"], str)
        or not hardware["cuda_version"].strip()
    ):
        raise FullPetalLaunchError("fixed-step profile cuda_version is invalid")
    if profile["dimensions"] != dimensions:
        raise FullPetalLaunchError("fixed-step profile dimensions differ from the config")
    measurements = profile["measurements"]
    if not isinstance(measurements, dict):
        raise FullPetalLaunchError("fixed-step profile measurements must be an object")
    _require_exact_fields(
        measurements,
        {
            "warmup_optimizer_events",
            "measured_optimizer_events",
            "total_optimizer_events",
            "skipped_optimizer_events",
            "elapsed_seconds",
            "peak_memory_bytes",
            "throughput_optimizer_events_per_second",
        },
        "fixed-step profile measurements",
    )
    if measurements["warmup_optimizer_events"] != warmup_events:
        raise FullPetalLaunchError("fixed-step profile warmup length differs")
    if measurements["measured_optimizer_events"] != measured_events:
        raise FullPetalLaunchError("fixed-step profile measured length differs")
    if measurements["total_optimizer_events"] != warmup_events + measured_events:
        raise FullPetalLaunchError("fixed-step profile total event count differs")
    if _require_nonnegative_int(
        measurements["skipped_optimizer_events"], "profile skipped events"
    ) != 0:
        raise FullPetalLaunchError("fixed-step profile contains skipped optimizer events")
    _require_positive_number(measurements["elapsed_seconds"], "profile elapsed seconds")
    _require_nonnegative_int(measurements["peak_memory_bytes"], "profile peak memory")
    _require_positive_number(
        measurements["throughput_optimizer_events_per_second"],
        "profile throughput",
    )
    return profile, path, digest


def build_fixed_step_profile_artifact(
    authorization,
    cfg,
    measurements,
    *,
    precision,
    gpu_name,
    torch_version,
    cuda_version,
):
    """Build the only profile artifact accepted by the formal launch gate."""

    if not isinstance(authorization, FullPetalLaunchAuthorization):
        raise FullPetalLaunchError("profile artifact requires launch authorization")
    if authorization.mode != PROFILE_MODE:
        raise FullPetalLaunchError("profile artifact requires profile-mode authorization")
    if precision not in {"fp32", "fp16", "bf16"}:
        raise FullPetalLaunchError("profile precision must be fp32, fp16, or bf16")
    hardware = {
        "gpu_name": gpu_name,
        "torch_version": torch_version,
        "cuda_version": cuda_version,
    }
    for field in ("gpu_name", "torch_version"):
        if not isinstance(hardware[field], str) or not hardware[field].strip():
            raise FullPetalLaunchError(f"profile {field} must be non-empty")
    if cuda_version is not None and (
        not isinstance(cuda_version, str) or not cuda_version.strip()
    ):
        raise FullPetalLaunchError("profile cuda_version is invalid")
    expected_measurement_fields = {
        "warmup_optimizer_events",
        "measured_optimizer_events",
        "total_optimizer_events",
        "skipped_optimizer_events",
        "elapsed_seconds",
        "peak_memory_bytes",
        "throughput_optimizer_events_per_second",
    }
    if not isinstance(measurements, Mapping):
        raise FullPetalLaunchError("profile measurements must be an object")
    _require_exact_fields(
        measurements, expected_measurement_fields, "profile measurements"
    )
    if measurements["warmup_optimizer_events"] != authorization.warmup_optimizer_events:
        raise FullPetalLaunchError("profile warmup event count differs from authorization")
    if measurements["measured_optimizer_events"] != authorization.measured_optimizer_events:
        raise FullPetalLaunchError("profile measured event count differs from authorization")
    if measurements["total_optimizer_events"] != authorization.total_optimizer_events:
        raise FullPetalLaunchError("profile total event count differs from authorization")
    if measurements["skipped_optimizer_events"] != 0:
        raise FullPetalLaunchError("profile cannot contain skipped optimizer events")
    _require_positive_number(measurements["elapsed_seconds"], "profile elapsed seconds")
    _require_nonnegative_int(measurements["peak_memory_bytes"], "profile peak memory")
    _require_positive_number(
        measurements["throughput_optimizer_events_per_second"],
        "profile throughput",
    )
    return {
        "schema_version": PROFILE_SCHEMA,
        "status": "PASS",
        "commit_sha": authorization.commit_sha,
        "resolved_config_sha256": authorization.resolved_config_sha256,
        "scientific_config_sha256": authorization.scientific_config_sha256,
        "launch_ticket_path": authorization.ticket_path,
        "launch_ticket_sha256": authorization.ticket_sha256,
        "slurm_job_id": authorization.slurm_job_id,
        "world_size": authorization.world_size,
        "precision": precision,
        "hardware": hardware,
        "dimensions": _profile_dimensions(cfg),
        "measurements": dict(measurements),
    }


def _launch_contract(cfg):
    contract = _cfg_get(cfg, "launch_contract")
    if not isinstance(contract, Mapping):
        raise FullPetalLaunchError("Full PETAL config is missing launch_contract")
    required = {
        "schema_version",
        "required_reviewer_id",
        "required_review_scope",
        "allowed_cfg_overrides",
        "require_clean_checkout",
    }
    _require_exact_fields(contract, required, "launch_contract")
    if contract["schema_version"] != LAUNCH_CONTRACT_SCHEMA:
        raise FullPetalLaunchError("Full PETAL launch contract schema is unsupported")
    reviewer_id = contract["required_reviewer_id"]
    if not isinstance(reviewer_id, str) or not reviewer_id.strip():
        raise FullPetalLaunchError("launch_contract requires a reviewer ID")
    scope = contract["required_review_scope"]
    if not isinstance(scope, (list, tuple)) or not scope or not all(
        isinstance(item, str) and item for item in scope
    ):
        raise FullPetalLaunchError("launch_contract requires an ordered review scope")
    overrides = contract["allowed_cfg_overrides"]
    if not isinstance(overrides, (list, tuple)) or not all(
        isinstance(item, str) and item for item in overrides
    ):
        raise FullPetalLaunchError("launch_contract allowed overrides are invalid")
    if contract["require_clean_checkout"] is not True:
        raise FullPetalLaunchError("Full PETAL requires a clean checkout")
    return contract


def _profile_contract(cfg):
    contract = _cfg_get(cfg, "profile_contract")
    if not isinstance(contract, Mapping):
        raise FullPetalLaunchError("Full PETAL config is missing profile_contract")
    warmup = _require_nonnegative_int(
        contract.get("warmup_steps"), "profile warmup optimizer events"
    )
    measured = _require_nonnegative_int(
        contract.get("measured_steps"),
        "profile measured optimizer events",
        positive=True,
    )
    world_size = _require_nonnegative_int(
        contract.get("world_size"), "profile world size", positive=True
    )
    if contract.get("step_unit") != "optimizer_event":
        raise FullPetalLaunchError("profile step_unit must be optimizer_event")
    if contract.get("submit_via_slurm_only") is not True:
        raise FullPetalLaunchError("Full PETAL profile must be submitted through Slurm")
    return warmup, measured, world_size


def _validate_environment(environ, world_size):
    slurm_job_id = str(environ.get("SLURM_JOB_ID", "")).strip()
    if not slurm_job_id:
        raise FullPetalLaunchError("Full PETAL GPU launch requires SLURM_JOB_ID")
    try:
        actual_world_size = int(environ.get("WORLD_SIZE", ""))
    except (TypeError, ValueError) as exc:
        raise FullPetalLaunchError("WORLD_SIZE must be explicitly supplied") from exc
    if actual_world_size != world_size:
        raise FullPetalLaunchError(
            f"Full PETAL world size must be {world_size}, found {actual_world_size}"
        )
    return slurm_job_id


def _validate_authorization_state(cfg, mode):
    if mode == PROFILE_MODE:
        if _cfg_get(cfg, "formal_training_ready") is not False:
            raise FullPetalLaunchError(
                "profile requires formal_training_ready to remain false"
            )
        if _cfg_get(cfg, "gpu_authorization") != "BLOCKED_UNTIL_B0_AND_PROFILE":
            raise FullPetalLaunchError(
                "profile config has an unexpected GPU authorization state"
            )
        return
    if _cfg_get(cfg, "formal_training_ready") is not True:
        raise FullPetalLaunchError(
            "formal training remains blocked by formal_training_ready"
        )
    if _cfg_get(cfg, "gpu_authorization") != "FORMAL_TRAINING_APPROVED":
        raise FullPetalLaunchError("formal training lacks explicit GPU authorization")


def _ticket_fields():
    return {
        "schema_version",
        "mode",
        "commit_sha",
        "config_file_sha256",
        "resolved_config_sha256",
        "scientific_config_sha256",
        "b0_evidence",
        "review_evidence",
        "profile_evidence",
    }


def build_launch_ticket(
    cfg,
    config_path,
    *,
    mode,
    b0_path,
    review_path,
    profile_path=None,
    repository_root,
    repository_state=None,
):
    """Build a ticket only after the complete prerequisite chain validates."""

    if not is_full_petal_route(cfg):
        raise FullPetalLaunchError("launch tickets are only valid for Full PETAL configs")
    if mode not in {PROFILE_MODE, FORMAL_MODE}:
        raise FullPetalLaunchError("launch mode must be profile or formal")
    contract = _launch_contract(cfg)
    warmup, measured, world_size = _profile_contract(cfg)
    _validate_authorization_state(cfg, mode)
    state = repository_state or inspect_repository(repository_root)
    if not state.clean:
        raise FullPetalLaunchError("launch ticket requires a clean git checkout")
    commit_sha = _require_git_sha(state.commit_sha, "repository commit")
    config_path = Path(config_path).resolve()
    resolved_digest = resolved_config_sha256(cfg)
    scientific_digest = resolved_config_sha256(cfg, scientific=True)
    b0_reference = _file_reference(b0_path)
    review_reference = _file_reference(review_path)
    _, _, b0_digest = _validate_b0_artifact(
        b0_reference, config_path.parent, commit_sha
    )
    _validate_review_artifact(
        review_reference,
        config_path.parent,
        commit_sha=commit_sha,
        b0_sha256=b0_digest,
        reviewer_id=contract["required_reviewer_id"],
        required_scope=contract["required_review_scope"],
    )
    profile_reference = None
    if mode == FORMAL_MODE:
        if profile_path is None:
            raise FullPetalLaunchError("formal launch ticket requires profile evidence")
        profile_reference = _file_reference(profile_path)
        _validate_profile_artifact(
            profile_reference,
            config_path.parent,
            scientific_config_sha256=scientific_digest,
            warmup_events=warmup,
            measured_events=measured,
            world_size=world_size,
            dimensions=_profile_dimensions(cfg),
            reviewer_id=contract["required_reviewer_id"],
            required_scope=contract["required_review_scope"],
        )
    elif profile_path is not None:
        raise FullPetalLaunchError("profile launch ticket cannot contain profile evidence")
    return {
        "schema_version": LAUNCH_TICKET_SCHEMA,
        "mode": mode,
        "commit_sha": commit_sha,
        "config_file_sha256": sha256_file(config_path),
        "resolved_config_sha256": resolved_digest,
        "scientific_config_sha256": scientific_digest,
        "b0_evidence": b0_reference,
        "review_evidence": review_reference,
        "profile_evidence": profile_reference,
    }


def validate_full_petal_launch(
    cfg,
    config_path,
    *,
    mode=None,
    ticket_path=None,
    entrypoint="train",
    cfg_override_keys=(),
    environ=None,
    repository_root=None,
    repository_state=None,
):
    """Validate all evidence before CUDA device selection or DDP initialization."""

    if not is_full_petal_route(cfg):
        if mode is not None or ticket_path is not None:
            raise FullPetalLaunchError(
                "Full PETAL launch options cannot be used with a non-Full-PETAL config"
            )
        return None
    if mode not in {PROFILE_MODE, FORMAL_MODE}:
        raise FullPetalLaunchError(
            "Full PETAL cannot start without --launch-mode profile|formal"
        )
    if ticket_path is None:
        raise FullPetalLaunchError("Full PETAL cannot start without --launch-ticket")
    if entrypoint not in {"train", "test"}:
        raise FullPetalLaunchError(f"unsupported Full PETAL entrypoint {entrypoint!r}")
    if mode == PROFILE_MODE and entrypoint != "train":
        raise FullPetalLaunchError("fixed-step profile is only valid through tools/train.py")

    contract = _launch_contract(cfg)
    allowed_overrides = set(contract["allowed_cfg_overrides"])
    unexpected_overrides = sorted(set(cfg_override_keys) - allowed_overrides)
    if unexpected_overrides:
        raise FullPetalLaunchError(
            "Full PETAL forbids scientific --cfg-options overrides: "
            + ", ".join(unexpected_overrides)
        )
    warmup, measured, world_size = _profile_contract(cfg)
    _validate_authorization_state(cfg, mode)
    environ = {} if environ is None else environ
    slurm_job_id = _validate_environment(environ, world_size)
    repository_root = Path(repository_root or Path(config_path).resolve().parents[2])
    state = repository_state or inspect_repository(repository_root)
    if not state.clean:
        raise FullPetalLaunchError(
            "Full PETAL launch requires a clean git checkout: " + state.status
        )
    commit_sha = _require_git_sha(state.commit_sha, "repository commit")

    ticket_path = Path(ticket_path).resolve()
    ticket = load_strict_json(ticket_path)
    _require_exact_fields(ticket, _ticket_fields(), "launch ticket")
    if ticket["schema_version"] != LAUNCH_TICKET_SCHEMA:
        raise FullPetalLaunchError("launch ticket schema is unsupported")
    if ticket["mode"] != mode:
        raise FullPetalLaunchError("launch ticket mode differs from the requested mode")
    if ticket["commit_sha"] != commit_sha:
        raise FullPetalLaunchError("launch ticket does not bind the checked-out commit")
    config_path = Path(config_path).resolve()
    if ticket["config_file_sha256"] != sha256_file(config_path):
        raise FullPetalLaunchError("launch ticket config file hash mismatch")
    resolved_digest = resolved_config_sha256(cfg)
    scientific_digest = resolved_config_sha256(cfg, scientific=True)
    if ticket["resolved_config_sha256"] != resolved_digest:
        raise FullPetalLaunchError("launch ticket resolved config hash mismatch")
    if ticket["scientific_config_sha256"] != scientific_digest:
        raise FullPetalLaunchError("launch ticket scientific config hash mismatch")

    _, _, b0_digest = _validate_b0_artifact(
        ticket["b0_evidence"], ticket_path.parent, commit_sha
    )
    _, _, review_digest = _validate_review_artifact(
        ticket["review_evidence"],
        ticket_path.parent,
        commit_sha=commit_sha,
        b0_sha256=b0_digest,
        reviewer_id=contract["required_reviewer_id"],
        required_scope=contract["required_review_scope"],
    )

    profile_digest = None
    if mode == PROFILE_MODE:
        if ticket["profile_evidence"] is not None:
            raise FullPetalLaunchError("profile ticket must not contain profile evidence")
    else:
        if ticket["profile_evidence"] is None:
            raise FullPetalLaunchError("formal training requires fixed-step profile evidence")
        _, _, profile_digest = _validate_profile_artifact(
            ticket["profile_evidence"],
            ticket_path.parent,
            scientific_config_sha256=scientific_digest,
            warmup_events=warmup,
            measured_events=measured,
            world_size=world_size,
            dimensions=_profile_dimensions(cfg),
            reviewer_id=contract["required_reviewer_id"],
            required_scope=contract["required_review_scope"],
        )

    return FullPetalLaunchAuthorization(
        mode=mode,
        commit_sha=commit_sha,
        resolved_config_sha256=resolved_digest,
        scientific_config_sha256=scientific_digest,
        ticket_path=str(ticket_path),
        ticket_sha256=sha256_file(ticket_path),
        b0_artifact_sha256=b0_digest,
        review_artifact_sha256=review_digest,
        profile_artifact_sha256=profile_digest,
        warmup_optimizer_events=warmup,
        measured_optimizer_events=measured,
        world_size=world_size,
        slurm_job_id=slurm_job_id,
    )


__all__ = [
    "B0_AUDIT_REPORT_SCHEMA",
    "B0_SCHEMA",
    "B0_TEST_REPORT_SCHEMA",
    "FORMAL_MODE",
    "FullPetalLaunchAuthorization",
    "FullPetalLaunchError",
    "LAUNCH_CONTRACT_SCHEMA",
    "LAUNCH_TICKET_SCHEMA",
    "PROFILE_MODE",
    "PROFILE_SCHEMA",
    "REVIEW_SCHEMA",
    "RepositoryState",
    "build_fixed_step_profile_artifact",
    "build_launch_ticket",
    "canonical_json_sha256",
    "inspect_repository",
    "is_full_petal_route",
    "load_strict_json",
    "resolved_config_sha256",
    "sha256_file",
    "validate_full_petal_launch",
]
