"""Fail-closed launch authorization for the Full PETAL route."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import subprocess

from .full_petal_attestation import (
    ATTESTATION_FIELD,
    AttestationError,
    public_key_base64,
    public_key_sha256,
    sign_payload,
    verify_payload,
)
from .full_petal_b0 import (
    B0_AUDIT_REPORT_SCHEMA,
    B0_MANIFEST_SCHEMA,
    B0_SCHEMA,
    B0_TEST_REPORT_SCHEMA,
    B0EvidenceError,
    validate_b0_evidence,
)
from .full_petal_identity import (
    IdentityError,
    SlurmAllocation,
    authorization_only_diff,
    build_data_identity,
    build_runtime_identity,
    canonical_json_sha256 as identity_json_sha256,
    inspect_slurm_allocation,
    scientific_source_identity,
    validate_data_identity,
    validate_slurm_allocation,
)


LAUNCH_CONTRACT_SCHEMA = "full-petal-launch-contract-v2"
LAUNCH_TICKET_SCHEMA = "full-petal-launch-ticket-v2"
REVIEW_SCHEMA = "full-petal-independent-review-v2"
PROFILE_SCHEMA = "full-petal-fixed-step-profile-v2"
LAUNCH_RECEIPT_SCHEMA = "full-petal-launch-receipt-v1"
PROFILE_MODE = "profile"
FORMAL_MODE = "formal"
REVIEW_ATTESTATION_ROLE = "independent-reviewer"
PROFILE_ATTESTATION_ROLE = "fixed-step-profile"

_SHA256_ALPHABET = frozenset("0123456789abcdef")
_SCIENTIFIC_DIGEST_EXCLUSIONS = {
    "formal_training_ready",
    "gpu_authorization",
    "launch_contract",
    "work_dir",
}
_RESOLVED_DIGEST_EXCLUSIONS = {"work_dir"}
_TRUST_ROLES = {"b0", "review", "profile"}


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
    source_tree_sha256: str
    data_identity_sha256: str
    runtime_identity_sha256: str
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
    slurm_allocation: SlurmAllocation

    @property
    def total_optimizer_events(self):
        return self.warmup_optimizer_events + self.measured_optimizer_events


def build_launch_receipt(authorization):
    """Capture the authenticated runtime and active Slurm allocation."""

    if not isinstance(authorization, FullPetalLaunchAuthorization):
        raise FullPetalLaunchError("launch receipt requires validated authorization")
    return {
        "schema_version": LAUNCH_RECEIPT_SCHEMA,
        "mode": authorization.mode,
        "commit_sha": authorization.commit_sha,
        "source_tree_sha256": authorization.source_tree_sha256,
        "data_identity_sha256": authorization.data_identity_sha256,
        "runtime_identity_sha256": authorization.runtime_identity_sha256,
        "resolved_config_sha256": authorization.resolved_config_sha256,
        "scientific_config_sha256": authorization.scientific_config_sha256,
        "launch_ticket": {
            "path": authorization.ticket_path,
            "sha256": authorization.ticket_sha256,
        },
        "b0_artifact_sha256": authorization.b0_artifact_sha256,
        "review_artifact_sha256": authorization.review_artifact_sha256,
        "profile_artifact_sha256": authorization.profile_artifact_sha256,
        "world_size": authorization.world_size,
        "slurm_job_id": authorization.slurm_job_id,
        "slurm_allocation": asdict(authorization.slurm_allocation),
    }


def default_launch_receipt_path(authorization):
    if not isinstance(authorization, FullPetalLaunchAuthorization):
        raise FullPetalLaunchError("launch receipt path requires validated authorization")
    return Path(f"{authorization.ticket_path}.receipt.json").resolve()


def persist_launch_receipt(authorization, output_path=None):
    """Write one canonical, non-overwritable receipt before CUDA initialization."""

    payload = build_launch_receipt(authorization)
    output = (
        default_launch_receipt_path(authorization)
        if output_path is None
        else Path(output_path).expanduser().resolve()
    )
    if output.exists():
        raise FullPetalLaunchError(f"refusing to overwrite launch receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        output,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0),
        0o600,
    )
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError("partial launch receipt write")
        os.fsync(descriptor)
    except OSError as exc:
        raise FullPetalLaunchError(f"failed to persist launch receipt: {exc}") from exc
    finally:
        os.close(descriptor)
    return output


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
    exclusions = _SCIENTIFIC_DIGEST_EXCLUSIONS if scientific else _RESOLVED_DIGEST_EXCLUSIONS
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
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256_ALPHABET


def _valid_git_sha(value):
    return isinstance(value, str) and len(value) == 40 and set(value) <= _SHA256_ALPHABET


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
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise FullPetalLaunchError(f"{label} must be a positive finite number")
    return number


def _require_exact_fields(payload, expected, label):
    if not isinstance(payload, Mapping):
        raise FullPetalLaunchError(f"{label} must be an object")
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


def _trust_roots(contract):
    roots = contract["attestation_trust_roots"]
    _require_exact_fields(roots, _TRUST_ROLES, "attestation trust roots")
    for role, root in roots.items():
        _require_exact_fields(root, {"key_id", "public_key"}, f"{role} trust root")
        if not isinstance(root["key_id"], str) or not root["key_id"].strip():
            raise FullPetalLaunchError(f"{role} trust root key_id must be non-empty")
        try:
            public_key_sha256(root["public_key"])
        except AttestationError as exc:
            raise FullPetalLaunchError(f"{role} trust root is invalid: {exc}") from exc
    return roots


def _validate_b0_artifact(
    reference,
    ticket_dir,
    commit_sha,
    *,
    trust_root,
    repository_root=None,
):
    try:
        artifact = validate_b0_evidence(
            reference,
            base_dir=ticket_dir,
            expected_commit=commit_sha,
            trust_root=trust_root,
            repository_root=repository_root,
        )
    except B0EvidenceError as exc:
        raise FullPetalLaunchError(f"B0 evidence is invalid: {exc}") from exc
    return artifact, Path(artifact["artifact_path"]), artifact["artifact_sha256"]


def _validate_review_artifact(
    reference,
    ticket_dir,
    *,
    commit_sha,
    b0_sha256,
    reviewer_id,
    required_scope,
    trust_root,
):
    signed, path, digest = _load_reference(
        reference, ticket_dir, "independent review evidence"
    )
    try:
        review = verify_payload(
            signed,
            trust_root=trust_root,
            role=REVIEW_ATTESTATION_ROLE,
        )
    except AttestationError as exc:
        raise FullPetalLaunchError(f"independent review attestation is invalid: {exc}") from exc
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


def _profile_measurement_fields():
    return {
        "warmup_optimizer_events",
        "measured_optimizer_events",
        "total_optimizer_events",
        "skipped_optimizer_events",
        "elapsed_seconds",
        "peak_memory_bytes",
        "throughput_optimizer_events_per_second",
    }


def _validate_measurements(measurements, *, warmup, measured, label):
    _require_exact_fields(measurements, _profile_measurement_fields(), label)
    if measurements["warmup_optimizer_events"] != warmup:
        raise FullPetalLaunchError(f"{label} warmup length differs")
    if measurements["measured_optimizer_events"] != measured:
        raise FullPetalLaunchError(f"{label} measured length differs")
    if measurements["total_optimizer_events"] != warmup + measured:
        raise FullPetalLaunchError(f"{label} total event count differs")
    if _require_nonnegative_int(
        measurements["skipped_optimizer_events"], f"{label} skipped events"
    ) != 0:
        raise FullPetalLaunchError(f"{label} contains skipped optimizer events")
    _require_positive_number(measurements["elapsed_seconds"], f"{label} elapsed seconds")
    _require_nonnegative_int(measurements["peak_memory_bytes"], f"{label} peak memory")
    _require_positive_number(
        measurements["throughput_optimizer_events_per_second"], f"{label} throughput"
    )


def _ticket_fields():
    return {
        "schema_version",
        "mode",
        "commit_sha",
        "source_tree_sha256",
        "config_file_sha256",
        "resolved_config_sha256",
        "scientific_config_sha256",
        "data_identity",
        "runtime_identity",
        "b0_evidence",
        "review_evidence",
        "profile_evidence",
    }


def _validate_profile_artifact(
    reference,
    ticket_dir,
    *,
    current_commit,
    repository_root,
    source_tree_sha256,
    data_identity_sha256,
    scientific_config_sha256,
    warmup_events,
    measured_events,
    world_size,
    dimensions,
    reviewer_id,
    required_scope,
    trust_roots,
):
    signed, path, digest = _load_reference(
        reference, ticket_dir, "fixed-step profile evidence"
    )
    try:
        profile = verify_payload(
            signed,
            trust_root=trust_roots["profile"],
            role=PROFILE_ATTESTATION_ROLE,
        )
    except AttestationError as exc:
        raise FullPetalLaunchError(f"fixed-step profile attestation is invalid: {exc}") from exc
    _require_exact_fields(
        profile,
        {
            "schema_version",
            "status",
            "commit_sha",
            "source_tree_sha256",
            "data_identity_sha256",
            "runtime_identity_sha256",
            "resolved_config_sha256",
            "scientific_config_sha256",
            "launch_ticket_path",
            "launch_ticket_sha256",
            "slurm_job_id",
            "slurm_allocation",
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
    profile_commit = _require_git_sha(profile["commit_sha"], "fixed-step profile commit")
    if profile_commit == current_commit:
        raise FullPetalLaunchError(
            "formal authorization must be a later authorization-only commit"
        )
    try:
        authorization_only_diff(repository_root, profile_commit, current_commit)
    except IdentityError as exc:
        raise FullPetalLaunchError(str(exc)) from exc
    if profile["source_tree_sha256"] != source_tree_sha256:
        raise FullPetalLaunchError("fixed-step profile used a different scientific source tree")
    if profile["data_identity_sha256"] != data_identity_sha256:
        raise FullPetalLaunchError("fixed-step profile used a different data identity")
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
    if profile_ticket["mode"] != PROFILE_MODE or profile_ticket["commit_sha"] != profile_commit:
        raise FullPetalLaunchError("fixed-step profile ticket identity differs")
    if profile_ticket["source_tree_sha256"] != source_tree_sha256:
        raise FullPetalLaunchError("profile ticket source tree differs")
    if profile_ticket["scientific_config_sha256"] != scientific_config_sha256:
        raise FullPetalLaunchError("profile ticket scientific config differs")
    if profile_ticket["data_identity"].get("identity_sha256") != data_identity_sha256:
        raise FullPetalLaunchError("profile ticket data identity differs")
    if canonical_json_sha256(profile_ticket["runtime_identity"]) != profile["runtime_identity_sha256"]:
        raise FullPetalLaunchError("profile runtime identity differs from its launch ticket")
    if profile_ticket["profile_evidence"] is not None:
        raise FullPetalLaunchError("profile launch ticket cannot contain profile evidence")
    _, _, profile_b0_digest = _validate_b0_artifact(
        profile_ticket["b0_evidence"],
        profile_ticket_path.parent,
        profile_commit,
        trust_root=trust_roots["b0"],
    )
    _validate_review_artifact(
        profile_ticket["review_evidence"],
        profile_ticket_path.parent,
        commit_sha=profile_commit,
        b0_sha256=profile_b0_digest,
        reviewer_id=reviewer_id,
        required_scope=required_scope,
        trust_root=trust_roots["review"],
    )
    if not isinstance(profile["slurm_job_id"], str) or not profile["slurm_job_id"].strip():
        raise FullPetalLaunchError("fixed-step profile lacks a Slurm job ID")
    allocation = profile["slurm_allocation"]
    _require_exact_fields(allocation, set(SlurmAllocation.__dataclass_fields__), "profile Slurm allocation")
    if allocation["job_id"] != profile["slurm_job_id"] or allocation["state"] != "RUNNING":
        raise FullPetalLaunchError("fixed-step profile lacks an authenticated running allocation")
    if profile["world_size"] != world_size:
        raise FullPetalLaunchError("fixed-step profile world size differs from the contract")
    if profile["precision"] not in {"fp32", "fp16", "bf16"}:
        raise FullPetalLaunchError("fixed-step profile precision is unsupported")
    hardware = profile["hardware"]
    _require_exact_fields(
        hardware, {"gpu_name", "torch_version", "cuda_version"}, "fixed-step profile hardware"
    )
    for field in ("gpu_name", "torch_version"):
        if not isinstance(hardware[field], str) or not hardware[field].strip():
            raise FullPetalLaunchError(f"fixed-step profile {field} must be non-empty")
    if hardware["cuda_version"] is not None and (
        not isinstance(hardware["cuda_version"], str) or not hardware["cuda_version"].strip()
    ):
        raise FullPetalLaunchError("fixed-step profile cuda_version is invalid")
    if profile["dimensions"] != dimensions:
        raise FullPetalLaunchError("fixed-step profile dimensions differ from the config")
    _validate_measurements(
        profile["measurements"],
        warmup=warmup_events,
        measured=measured_events,
        label="fixed-step profile",
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
    private_key_path,
    key_id,
):
    """Build and sign the only profile artifact accepted by the formal gate."""

    if not isinstance(authorization, FullPetalLaunchAuthorization):
        raise FullPetalLaunchError("profile artifact requires launch authorization")
    if authorization.mode != PROFILE_MODE:
        raise FullPetalLaunchError("profile artifact requires profile-mode authorization")
    if precision not in {"fp32", "fp16", "bf16"}:
        raise FullPetalLaunchError("profile precision must be fp32, fp16, or bf16")
    _validate_measurements(
        measurements,
        warmup=authorization.warmup_optimizer_events,
        measured=authorization.measured_optimizer_events,
        label="profile measurements",
    )
    hardware = {
        "gpu_name": gpu_name,
        "torch_version": torch_version,
        "cuda_version": cuda_version,
    }
    for field in ("gpu_name", "torch_version"):
        if not isinstance(hardware[field], str) or not hardware[field].strip():
            raise FullPetalLaunchError(f"profile {field} must be non-empty")
    body = {
        "schema_version": PROFILE_SCHEMA,
        "status": "PASS",
        "commit_sha": authorization.commit_sha,
        "source_tree_sha256": authorization.source_tree_sha256,
        "data_identity_sha256": authorization.data_identity_sha256,
        "runtime_identity_sha256": authorization.runtime_identity_sha256,
        "resolved_config_sha256": authorization.resolved_config_sha256,
        "scientific_config_sha256": authorization.scientific_config_sha256,
        "launch_ticket_path": authorization.ticket_path,
        "launch_ticket_sha256": authorization.ticket_sha256,
        "slurm_job_id": authorization.slurm_job_id,
        "slurm_allocation": asdict(authorization.slurm_allocation),
        "world_size": authorization.world_size,
        "precision": precision,
        "hardware": hardware,
        "dimensions": _profile_dimensions(cfg),
        "measurements": dict(measurements),
    }
    try:
        return sign_payload(
            body,
            private_key_path=private_key_path,
            key_id=key_id,
            role=PROFILE_ATTESTATION_ROLE,
        )
    except AttestationError as exc:
        raise FullPetalLaunchError(f"failed to attest fixed-step profile: {exc}") from exc


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
        "attestation_trust_roots",
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
    _trust_roots(contract)
    return contract


def _profile_contract(cfg):
    contract = _cfg_get(cfg, "profile_contract")
    if not isinstance(contract, Mapping):
        raise FullPetalLaunchError("Full PETAL config is missing profile_contract")
    warmup = _require_nonnegative_int(
        contract.get("warmup_steps"), "profile warmup optimizer events"
    )
    measured = _require_nonnegative_int(
        contract.get("measured_steps"), "profile measured optimizer events", positive=True
    )
    world_size = _require_nonnegative_int(
        contract.get("world_size"), "profile world size", positive=True
    )
    if contract.get("step_unit") != "optimizer_event":
        raise FullPetalLaunchError("profile step_unit must be optimizer_event")
    if contract.get("submit_via_slurm_only") is not True:
        raise FullPetalLaunchError("Full PETAL profile must be submitted through Slurm")
    return warmup, measured, world_size


def _validate_environment(
    environ,
    world_size,
    *,
    slurm_allocation=None,
    expected_user=None,
):
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
    try:
        allocation = slurm_allocation or inspect_slurm_allocation(slurm_job_id)
        validate_slurm_allocation(
            allocation,
            job_id=slurm_job_id,
            world_size=world_size,
            expected_user=expected_user,
        )
    except IdentityError as exc:
        raise FullPetalLaunchError(f"Slurm allocation authentication failed: {exc}") from exc
    return slurm_job_id, allocation


def _validate_authorization_state(cfg, mode):
    if mode == PROFILE_MODE:
        if _cfg_get(cfg, "formal_training_ready") is not False:
            raise FullPetalLaunchError("profile requires formal_training_ready to remain false")
        if _cfg_get(cfg, "gpu_authorization") != "BLOCKED_UNTIL_B0_AND_PROFILE":
            raise FullPetalLaunchError("profile config has an unexpected GPU authorization state")
        return
    if _cfg_get(cfg, "formal_training_ready") is not True:
        raise FullPetalLaunchError("formal training remains blocked by formal_training_ready")
    if _cfg_get(cfg, "gpu_authorization") != "FORMAL_TRAINING_APPROVED":
        raise FullPetalLaunchError("formal training lacks explicit GPU authorization")


def _runtime_identity(
    *,
    entrypoint,
    seed,
    run_id,
    deterministic,
    not_eval,
    resume_path,
    cfg_overrides,
):
    try:
        return build_runtime_identity(
            entrypoint=entrypoint,
            seed=seed,
            run_id=run_id,
            deterministic=deterministic,
            not_eval=not_eval,
            resume_path=resume_path,
            cfg_overrides=cfg_overrides,
        )
    except IdentityError as exc:
        raise FullPetalLaunchError(str(exc)) from exc


def build_launch_ticket(
    cfg,
    config_path,
    *,
    mode,
    b0_path,
    review_path,
    profile_path=None,
    repository_root,
    entrypoint,
    seed,
    run_id,
    deterministic,
    not_eval,
    resume_path,
    cfg_overrides,
    repository_state=None,
):
    """Build a ticket only after the complete prerequisite chain validates."""

    if not is_full_petal_route(cfg):
        raise FullPetalLaunchError("launch tickets are only valid for Full PETAL configs")
    if mode not in {PROFILE_MODE, FORMAL_MODE}:
        raise FullPetalLaunchError("launch mode must be profile or formal")
    contract = _launch_contract(cfg)
    trust_roots = _trust_roots(contract)
    warmup, measured, world_size = _profile_contract(cfg)
    del warmup, measured
    _validate_authorization_state(cfg, mode)
    root = Path(repository_root).resolve()
    state = repository_state or inspect_repository(root)
    if not state.clean:
        raise FullPetalLaunchError("launch ticket requires a clean git checkout")
    commit_sha = _require_git_sha(state.commit_sha, "repository commit")
    config_path = Path(config_path).resolve()
    resolved_digest = resolved_config_sha256(cfg)
    scientific_digest = resolved_config_sha256(cfg, scientific=True)
    try:
        source_digest = scientific_source_identity(root, scientific_digest)
        data_identity = build_data_identity(cfg)
    except IdentityError as exc:
        raise FullPetalLaunchError(f"launch identity cannot be established: {exc}") from exc
    runtime_identity = _runtime_identity(
        entrypoint=entrypoint,
        seed=seed,
        run_id=run_id,
        deterministic=deterministic,
        not_eval=not_eval,
        resume_path=resume_path,
        cfg_overrides=cfg_overrides,
    )
    if mode == PROFILE_MODE and runtime_identity["resume_checkpoint"] is not None:
        raise FullPetalLaunchError("fixed-step profile cannot resume from a checkpoint")
    if (
        mode == FORMAL_MODE
        and entrypoint == "train"
        and runtime_identity["resume_checkpoint"] is not None
    ):
        raise FullPetalLaunchError(
            "formal training resume is blocked until optimizer-event trace continuation is implemented"
        )
    if entrypoint != "train" and mode == PROFILE_MODE:
        raise FullPetalLaunchError("fixed-step profile is only valid through tools/train.py")
    if set(cfg_overrides) - set(contract["allowed_cfg_overrides"]):
        raise FullPetalLaunchError("launch ticket contains forbidden config overrides")
    b0_reference = _file_reference(b0_path)
    review_reference = _file_reference(review_path)
    _, _, b0_digest = _validate_b0_artifact(
        b0_reference,
        config_path.parent,
        commit_sha,
        trust_root=trust_roots["b0"],
        repository_root=root,
    )
    _validate_review_artifact(
        review_reference,
        config_path.parent,
        commit_sha=commit_sha,
        b0_sha256=b0_digest,
        reviewer_id=contract["required_reviewer_id"],
        required_scope=contract["required_review_scope"],
        trust_root=trust_roots["review"],
    )
    profile_reference = None
    if mode == FORMAL_MODE:
        if profile_path is None:
            raise FullPetalLaunchError("formal launch ticket requires profile evidence")
        profile_reference = _file_reference(profile_path)
        _validate_profile_artifact(
            profile_reference,
            config_path.parent,
            current_commit=commit_sha,
            repository_root=root,
            source_tree_sha256=source_digest,
            data_identity_sha256=data_identity["identity_sha256"],
            scientific_config_sha256=scientific_digest,
            warmup_events=_profile_contract(cfg)[0],
            measured_events=_profile_contract(cfg)[1],
            world_size=world_size,
            dimensions=_profile_dimensions(cfg),
            reviewer_id=contract["required_reviewer_id"],
            required_scope=contract["required_review_scope"],
            trust_roots=trust_roots,
        )
    elif profile_path is not None:
        raise FullPetalLaunchError("profile launch ticket cannot contain profile evidence")
    return {
        "schema_version": LAUNCH_TICKET_SCHEMA,
        "mode": mode,
        "commit_sha": commit_sha,
        "source_tree_sha256": source_digest,
        "config_file_sha256": sha256_file(config_path),
        "resolved_config_sha256": resolved_digest,
        "scientific_config_sha256": scientific_digest,
        "data_identity": data_identity,
        "runtime_identity": runtime_identity,
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
    seed=42,
    run_id=0,
    deterministic=True,
    not_eval=False,
    resume_path=None,
    cfg_overrides=None,
    cfg_override_keys=(),
    environ=None,
    repository_root=None,
    repository_state=None,
    slurm_allocation=None,
    expected_slurm_user=None,
    profile_signing_key_path=None,
):
    """Validate all evidence and runtime identity before CUDA or DDP initialization."""

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
    trust_roots = _trust_roots(contract)
    if mode == PROFILE_MODE:
        if profile_signing_key_path is None:
            raise FullPetalLaunchError(
                "fixed-step profile requires an external profile signing key"
            )
        try:
            signer_public_key = public_key_base64(profile_signing_key_path)
        except AttestationError as exc:
            raise FullPetalLaunchError(f"profile signing key is invalid: {exc}") from exc
        if signer_public_key != trust_roots["profile"]["public_key"]:
            raise FullPetalLaunchError("profile signing key does not match the trusted public key")
    cfg_overrides = dict(cfg_overrides or {})
    if cfg_override_keys and set(cfg_override_keys) != set(cfg_overrides):
        raise FullPetalLaunchError("config override keys/values identity is inconsistent")
    unexpected_overrides = sorted(set(cfg_overrides) - set(contract["allowed_cfg_overrides"]))
    if unexpected_overrides:
        raise FullPetalLaunchError(
            "Full PETAL forbids scientific --cfg-options overrides: "
            + ", ".join(unexpected_overrides)
        )
    warmup, measured, world_size = _profile_contract(cfg)
    _validate_authorization_state(cfg, mode)
    runtime_identity = _runtime_identity(
        entrypoint=entrypoint,
        seed=seed,
        run_id=run_id,
        deterministic=deterministic,
        not_eval=not_eval,
        resume_path=resume_path,
        cfg_overrides=cfg_overrides,
    )
    if mode == PROFILE_MODE and runtime_identity["resume_checkpoint"] is not None:
        raise FullPetalLaunchError("fixed-step profile cannot resume from a checkpoint")
    if (
        mode == FORMAL_MODE
        and entrypoint == "train"
        and runtime_identity["resume_checkpoint"] is not None
    ):
        raise FullPetalLaunchError(
            "formal training resume is blocked until optimizer-event trace continuation is implemented"
        )
    environ = {} if environ is None else environ
    slurm_job_id, allocation = _validate_environment(
        environ,
        world_size,
        slurm_allocation=slurm_allocation,
        expected_user=expected_slurm_user,
    )
    root = Path(repository_root or Path(config_path).resolve().parents[2]).resolve()
    state = repository_state or inspect_repository(root)
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
    if ticket["mode"] != mode or ticket["commit_sha"] != commit_sha:
        raise FullPetalLaunchError("launch ticket mode/commit identity differs")
    config_path = Path(config_path).resolve()
    if ticket["config_file_sha256"] != sha256_file(config_path):
        raise FullPetalLaunchError("launch ticket config file hash mismatch")
    resolved_digest = resolved_config_sha256(cfg)
    scientific_digest = resolved_config_sha256(cfg, scientific=True)
    if ticket["resolved_config_sha256"] != resolved_digest:
        raise FullPetalLaunchError("launch ticket resolved config hash mismatch")
    if ticket["scientific_config_sha256"] != scientific_digest:
        raise FullPetalLaunchError("launch ticket scientific config hash mismatch")
    try:
        source_digest = scientific_source_identity(root, scientific_digest)
        validate_data_identity(ticket["data_identity"], cfg)
    except IdentityError as exc:
        raise FullPetalLaunchError(f"launch source/data identity differs: {exc}") from exc
    if ticket["source_tree_sha256"] != source_digest:
        raise FullPetalLaunchError("launch ticket source tree hash mismatch")
    if ticket["runtime_identity"] != runtime_identity:
        raise FullPetalLaunchError("launch ticket runtime argv/seed/checkpoint identity differs")

    _, _, b0_digest = _validate_b0_artifact(
        ticket["b0_evidence"],
        ticket_path.parent,
        commit_sha,
        trust_root=trust_roots["b0"],
        repository_root=root,
    )
    _, _, review_digest = _validate_review_artifact(
        ticket["review_evidence"],
        ticket_path.parent,
        commit_sha=commit_sha,
        b0_sha256=b0_digest,
        reviewer_id=contract["required_reviewer_id"],
        required_scope=contract["required_review_scope"],
        trust_root=trust_roots["review"],
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
            current_commit=commit_sha,
            repository_root=root,
            source_tree_sha256=source_digest,
            data_identity_sha256=ticket["data_identity"]["identity_sha256"],
            scientific_config_sha256=scientific_digest,
            warmup_events=warmup,
            measured_events=measured,
            world_size=world_size,
            dimensions=_profile_dimensions(cfg),
            reviewer_id=contract["required_reviewer_id"],
            required_scope=contract["required_review_scope"],
            trust_roots=trust_roots,
        )

    return FullPetalLaunchAuthorization(
        mode=mode,
        commit_sha=commit_sha,
        source_tree_sha256=source_digest,
        data_identity_sha256=ticket["data_identity"]["identity_sha256"],
        runtime_identity_sha256=canonical_json_sha256(runtime_identity),
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
        slurm_allocation=allocation,
    )


__all__ = [
    "ATTESTATION_FIELD",
    "B0_AUDIT_REPORT_SCHEMA",
    "B0_MANIFEST_SCHEMA",
    "B0_SCHEMA",
    "B0_TEST_REPORT_SCHEMA",
    "FORMAL_MODE",
    "FullPetalLaunchAuthorization",
    "FullPetalLaunchError",
    "LAUNCH_CONTRACT_SCHEMA",
    "LAUNCH_RECEIPT_SCHEMA",
    "LAUNCH_TICKET_SCHEMA",
    "PROFILE_MODE",
    "PROFILE_SCHEMA",
    "REVIEW_SCHEMA",
    "RepositoryState",
    "build_launch_receipt",
    "build_fixed_step_profile_artifact",
    "build_launch_ticket",
    "canonical_json_sha256",
    "inspect_repository",
    "is_full_petal_route",
    "load_strict_json",
    "persist_launch_receipt",
    "resolved_config_sha256",
    "sha256_file",
    "validate_full_petal_launch",
]
