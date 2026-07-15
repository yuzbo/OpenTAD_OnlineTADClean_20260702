"""Fail-closed launch authorization for the Full PETAL route."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import weakref

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .full_petal_attestation import (
    ATTESTATION_ALGORITHM,
    ATTESTATION_FIELD,
    ATTESTATION_SCHEMA,
    AttestationError,
    _message,
    public_key_base64,
    public_key_sha256,
    verify_payload,
)
from .evidence_bundle import (
    EvidenceBundleError,
    bundle_file_reference,
    publish_exclusive_file,
    read_stable_file_bytes,
    read_verified_bundle_bytes,
    read_verified_bundle_json,
    relative_bundle_path,
    strict_json_from_bytes,
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
from .full_petal_role_signing import (
    sign_fixed_step_profile,
)
from .crs_eps_gold_gate import (
    AUDIT_ATTESTATION_ROLE,
    AUDIT_SCHEMA_VERSION,
    CrsEpsGoldGateError,
    MARGIN_ATTESTATION_ROLE,
    SELECTION_ATTESTATION_ROLE,
    evaluate_gold_audit,
    validate_gold_margins,
    validate_gold_selection,
)
from .crs_eps_sampling import CrsEpsSamplingError, validate_epoch_manifest
from .full_petal_runtime_attestation import (
    RuntimeAttestationError,
    issue_runtime_session,
    runtime_profile_trust_root,
)


LAUNCH_CONTRACT_SCHEMA = "full-petal-launch-contract-v4"
EVIDENCE_TRUST_MODEL_SCHEMA = "full-petal-evidence-trust-model-v1"
LAUNCH_TICKET_SCHEMA = "full-petal-launch-ticket-v3"
REVIEW_SCHEMA = "full-petal-independent-review-v2"
PROFILE_SCHEMA = "full-petal-fixed-step-profile-v3"
LAUNCH_RECEIPT_SCHEMA = "full-petal-launch-receipt-v3"
PROFILE_MODE = "profile"
FORMAL_MODE = "formal"
REVIEW_ATTESTATION_ROLE = "independent-reviewer"
PROFILE_ATTESTATION_ROLE = "fixed-step-profile"
LAUNCH_RECEIPT_ATTESTATION_ROLE = "launch-receipt"

_SHA256_ALPHABET = frozenset("0123456789abcdef")
_SCIENTIFIC_DIGEST_EXCLUSIONS = {
    "formal_training_ready",
    "gpu_authorization",
    "launch_contract",
    "work_dir",
}
_RESOLVED_DIGEST_EXCLUSIONS = {"work_dir"}
_TRUST_ROLES = {"b0", "review", "g0", "profile", "formal"}
_EVIDENCE_TRUSTED_COMPUTING_BASE = (
    "launch_validator",
    "train_engine",
    "runtime_evidence_session",
    "in_process_attestation_key_material",
    "g0_preregistration_and_audit_runner",
)
_EVIDENCE_GUARANTEES = (
    "fail_closed_lifecycle_wiring",
    "provenance_binding",
    "post_publication_tamper_evidence",
)
_EVIDENCE_OUT_OF_SCOPE = (
    "arbitrary_code_execution_inside_tcb",
    "in_process_private_key_compromise",
)
_CRS_EPS_PROFILE_DENOMINATORS = (
    "temporal_forward_tokens",
    "temporal_backward_tokens",
    "replay_tokens",
    "supervised_exposures",
    "unique_supervised_bins",
    "effective_sample_size",
    "visual_forward_frames",
    "visual_backward_frames",
    "data_wait_seconds",
    "control_unroll_seconds",
    "wall_seconds",
    "peak_memory_bytes",
    "gpu_hours",
)


class FullPetalLaunchError(RuntimeError):
    """Raised before CUDA/DDP initialization when authorization is incomplete."""


@dataclass(frozen=True)
class RepositoryState:
    commit_sha: str
    clean: bool
    status: str = ""


@dataclass(frozen=True, init=False)
class FullPetalLaunchAuthorization:
    mode: str
    commit_sha: str
    source_tree_sha256: str
    data_identity: dict
    data_identity_sha256: str
    runtime_identity_sha256: str
    runtime_identity: dict
    resolved_config_sha256: str
    scientific_config_sha256: str
    ticket_path: str
    ticket_sha256: str
    b0_artifact_sha256: str
    review_artifact_sha256: str
    g0_artifact_sha256: str | None
    crs_eps_manifest: dict | None
    profile_artifact_sha256: str | None
    warmup_optimizer_events: int
    measured_optimizer_events: int
    world_size: int
    slurm_job_id: str
    slurm_allocation: SlurmAllocation
    execution_attestation_key_id: str
    execution_attestation_public_key: str
    execution_session: dict

    def __init__(self, *args, **kwargs):
        del args, kwargs
        raise FullPetalLaunchError(
            "launch authorizations can only be issued by validate_full_petal_launch"
        )

    @property
    def total_optimizer_events(self):
        return self.warmup_optimizer_events + self.measured_optimizer_events


@dataclass
class _AuthorizationRuntimeState:
    runtime_session: object
    authorization_snapshot: str
    repository_root: Path
    expected_slurm_user: str | None
    trusted_scontrol_path: str
    receipt_path: Path | None = None
    receipt_sha256: str | None = None


_AUTHORIZATION_STATES = {}


def _authorization_snapshot(authorization):
    return identity_json_sha256(asdict(authorization))


def _authorization_state(authorization):
    if not isinstance(authorization, FullPetalLaunchAuthorization):
        raise FullPetalLaunchError("operation requires validated launch authorization")
    registration = _AUTHORIZATION_STATES.get(id(authorization))
    if registration is None or registration[0]() is not authorization:
        raise FullPetalLaunchError(
            "launch authorization is not an active validator-issued capability"
        )
    state = registration[1]
    if state.authorization_snapshot != _authorization_snapshot(authorization):
        raise FullPetalLaunchError("launch authorization changed after validation")
    return state


def runtime_evidence_session(authorization):
    """Return the in-memory signer bound to a validated live launch."""

    return _authorization_state(authorization).runtime_session


def _launch_receipt_body(authorization):
    """Capture the authenticated runtime and active Slurm allocation."""

    _authorization_state(authorization)
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
            "path": Path(authorization.ticket_path).name,
            "sha256": authorization.ticket_sha256,
        },
        "b0_artifact_sha256": authorization.b0_artifact_sha256,
        "review_artifact_sha256": authorization.review_artifact_sha256,
        "g0_artifact_sha256": authorization.g0_artifact_sha256,
        "profile_artifact_sha256": authorization.profile_artifact_sha256,
        "world_size": authorization.world_size,
        "slurm_job_id": authorization.slurm_job_id,
        "slurm_allocation": asdict(authorization.slurm_allocation),
        "execution_session": dict(authorization.execution_session),
    }


def build_launch_receipt(authorization, *, private_key_path):
    """Revalidate live launch state, then sign a one-session receipt."""

    state = _authorization_state(authorization)
    if state.receipt_path is not None:
        raise FullPetalLaunchError("launch authorization already issued a receipt")
    try:
        _, ticket_bytes = read_stable_file_bytes(
            authorization.ticket_path, "validated launch ticket"
        )
    except EvidenceBundleError as exc:
        raise FullPetalLaunchError(str(exc)) from exc
    if hashlib.sha256(ticket_bytes).hexdigest() != authorization.ticket_sha256:
        raise FullPetalLaunchError("launch ticket changed after authorization")
    repository = inspect_repository(state.repository_root)
    if not repository.clean or repository.commit_sha != authorization.commit_sha:
        raise FullPetalLaunchError(
            "repository changed after launch authorization"
        )
    try:
        allocation = inspect_slurm_allocation(
            authorization.slurm_job_id,
            scontrol_path=state.trusted_scontrol_path,
        )
        validate_slurm_allocation(
            allocation,
            job_id=authorization.slurm_job_id,
            world_size=authorization.world_size,
            expected_user=state.expected_slurm_user,
        )
    except IdentityError as exc:
        raise FullPetalLaunchError(
            f"launch receipt Slurm revalidation failed: {exc}"
        ) from exc
    if allocation != authorization.slurm_allocation:
        raise FullPetalLaunchError(
            "Slurm allocation changed after launch authorization"
        )
    body = _launch_receipt_body(authorization)
    try:
        key = serialization.load_pem_private_key(
            Path(private_key_path).read_bytes(), password=None
        )
        if not isinstance(key, Ed25519PrivateKey):
            raise TypeError("execution signing key is not Ed25519")
        raw_public_key = key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        signer_public_key = base64.b64encode(raw_public_key).decode("ascii")
        if signer_public_key != authorization.execution_attestation_public_key:
            raise FullPetalLaunchError(
                "execution signing key does not match the launch authorization"
            )
        signed = dict(body)
        signed[ATTESTATION_FIELD] = {
            "schema_version": ATTESTATION_SCHEMA,
            "algorithm": ATTESTATION_ALGORITHM,
            "role": LAUNCH_RECEIPT_ATTESTATION_ROLE,
            "key_id": authorization.execution_attestation_key_id,
            "public_key_sha256": public_key_sha256(signer_public_key),
            "signature": base64.b64encode(
                key.sign(_message(body, LAUNCH_RECEIPT_ATTESTATION_ROLE))
            ).decode("ascii"),
        }
        return signed
    except FullPetalLaunchError:
        raise
    except (OSError, TypeError, ValueError, AttestationError) as exc:
        raise FullPetalLaunchError(f"failed to attest launch receipt: {exc}") from exc


def verify_launch_receipt(payload, *, trust_root):
    """Verify a launch receipt before any semantic binding is trusted."""

    try:
        return verify_payload(
            payload,
            trust_root=trust_root,
            role=LAUNCH_RECEIPT_ATTESTATION_ROLE,
        )
    except AttestationError as exc:
        raise FullPetalLaunchError(f"launch receipt attestation is invalid: {exc}") from exc


def default_launch_receipt_path(authorization):
    _authorization_state(authorization)
    return Path(f"{authorization.ticket_path}.receipt.json").resolve()


def persist_launch_receipt(authorization, *, private_key_path, output_path=None):
    """Write one canonical, non-overwritable receipt before CUDA initialization."""

    payload = build_launch_receipt(
        authorization,
        private_key_path=private_key_path,
    )
    output = (
        default_launch_receipt_path(authorization)
        if output_path is None
        else Path(output_path).expanduser().resolve()
    )
    if output.parent != Path(authorization.ticket_path).resolve().parent:
        raise FullPetalLaunchError(
            "launch receipt must remain beside its validated launch ticket"
        )
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
    try:
        publish_exclusive_file(output, encoded)
    except EvidenceBundleError as exc:
        raise FullPetalLaunchError(f"failed to persist launch receipt: {exc}") from exc
    state = _authorization_state(authorization)
    state.receipt_path = output
    state.receipt_sha256 = hashlib.sha256(encoded).hexdigest()
    return output


def load_strict_json(path):
    path = Path(path)
    try:
        _, payload = read_stable_file_bytes(path, f"evidence file {path}")
        return strict_json_from_bytes(
            payload,
            f"evidence file {path}",
            require_object=True,
        )
    except EvidenceBundleError as exc:
        raise FullPetalLaunchError(str(exc)) from exc


def sha256_file(path, chunk_size=1024 * 1024):
    try:
        _, payload = read_stable_file_bytes(path, f"file {path}")
    except EvidenceBundleError as exc:
        raise FullPetalLaunchError(str(exc)) from exc
    return hashlib.sha256(payload).hexdigest()


def _json_value(value):
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_value(value.to_dict())
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


def resolved_config_payload(cfg):
    """Return a standalone JSON representation of a fully merged config."""

    return _json_value(cfg)


def resolved_config_sha256(cfg, *, scientific=False):
    payload = resolved_config_payload(cfg)
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


def _load_reference(reference, base_dir, label):
    try:
        path, _, payload = read_verified_bundle_json(
            reference,
            base_dir,
            label,
            require_object=True,
        )
    except EvidenceBundleError as exc:
        raise FullPetalLaunchError(str(exc)) from exc
    return payload, path, reference["sha256"]


def _file_reference(path, bundle_root, label):
    try:
        return bundle_file_reference(path, bundle_root, label)
    except EvidenceBundleError as exc:
        raise FullPetalLaunchError(str(exc)) from exc


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


def _validate_g0_artifact(
    reference,
    ticket_dir,
    *,
    commit_sha,
    source_tree_sha256,
    config_file_sha256,
    resolved_config_sha256,
    scientific_config_sha256,
    data_identity_sha256,
    crs_eps_contract,
    runtime_seed,
    trust_root,
):
    signed, path, digest = _load_reference(
        reference, ticket_dir, "CRS-EPS G0 audit evidence"
    )
    try:
        audit = verify_payload(
            signed,
            trust_root=trust_root,
            role=AUDIT_ATTESTATION_ROLE,
        )
    except AttestationError as exc:
        raise FullPetalLaunchError(f"CRS-EPS G0 audit attestation is invalid: {exc}") from exc
    _require_exact_fields(
        audit,
        {
            "schema_version",
            "status",
            "commit_sha",
            "source_tree_sha256",
            "resolved_config_sha256",
            "scientific_config_sha256",
            "data_identity_sha256",
            "config",
            "checkpoint",
            "episode_manifest",
            "selection",
            "margins",
            "rows",
            "gate",
        },
        "CRS-EPS G0 audit artifact",
    )
    if audit["schema_version"] != AUDIT_SCHEMA_VERSION or audit["status"] != "PASS":
        raise FullPetalLaunchError("CRS-EPS G0 audit has not reached PASS")
    expected_bindings = {
        "commit_sha": commit_sha,
        "source_tree_sha256": source_tree_sha256,
        "resolved_config_sha256": resolved_config_sha256,
        "scientific_config_sha256": scientific_config_sha256,
        "data_identity_sha256": data_identity_sha256,
    }
    drifted = sorted(
        key for key, value in expected_bindings.items() if audit[key] != value
    )
    if drifted:
        raise FullPetalLaunchError(
            "CRS-EPS G0 audit launch bindings differ: " + ", ".join(drifted)
        )
    _require_exact_fields(audit["config"], {"path", "sha256"}, "G0 config reference")
    if audit["config"]["sha256"] != config_file_sha256:
        raise FullPetalLaunchError("CRS-EPS G0 used a different config file")
    _require_exact_fields(
        audit["checkpoint"], {"path", "sha256", "state_key"}, "G0 checkpoint reference"
    )
    _require_sha256(audit["checkpoint"]["sha256"], "G0 checkpoint SHA256")
    if audit["checkpoint"]["state_key"] not in {"state_dict", "state_dict_ema"}:
        raise FullPetalLaunchError("CRS-EPS G0 checkpoint state key is unsupported")
    _require_exact_fields(
        audit["episode_manifest"],
        {"path", "file_sha256", "manifest_sha256", "sampling_specs_sha256"},
        "G0 episode-manifest reference",
    )
    manifest_reference = {
        "path": audit["episode_manifest"]["path"],
        "sha256": audit["episode_manifest"]["file_sha256"],
    }
    manifest, _, _ = _load_reference(
        manifest_reference, path.parent, "G0 episode manifest"
    )
    try:
        validate_epoch_manifest(manifest)
    except CrsEpsSamplingError as exc:
        raise FullPetalLaunchError(f"CRS-EPS G0 manifest is invalid: {exc}") from exc
    if (
        manifest["manifest_sha256"] != audit["episode_manifest"]["manifest_sha256"]
        or manifest["sampling_specs_sha256"]
        != audit["episode_manifest"]["sampling_specs_sha256"]
    ):
        raise FullPetalLaunchError("CRS-EPS G0 manifest identity differs from its audit")
    expected_sampling_contract = {
        "seed": int(runtime_seed),
        "epoch": 0,
        "draws_per_video": crs_eps_contract["draws_per_video"],
        "suffix_bins": crs_eps_contract["suffix_bins"],
        "context_bins": crs_eps_contract["context_bins"],
        "detach_interval": crs_eps_contract["detach_interval"],
        "mixture": dict(crs_eps_contract["mixture"]),
    }
    actual_sampling_contract = {
        field: manifest[field] for field in expected_sampling_contract
    }
    if actual_sampling_contract != expected_sampling_contract:
        raise FullPetalLaunchError(
            "CRS-EPS G0 manifest seed/epoch/sampling contract differs from launch"
        )

    preregistrations = {}
    for field, role, validator in (
        ("selection", SELECTION_ATTESTATION_ROLE, validate_gold_selection),
        ("margins", MARGIN_ATTESTATION_ROLE, validate_gold_margins),
    ):
        _require_exact_fields(audit[field], {"path", "sha256"}, f"G0 {field} reference")
        signed_preregistration, _, artifact_sha256 = _load_reference(
            audit[field], path.parent, f"G0 signed {field}"
        )
        try:
            body = verify_payload(
                signed_preregistration,
                trust_root=trust_root,
                role=role,
            )
            preregistrations[field] = validator(body)
        except (AttestationError, CrsEpsGoldGateError) as exc:
            raise FullPetalLaunchError(
                f"CRS-EPS G0 {field} preregistration is invalid: {exc}"
            ) from exc
        if artifact_sha256 != audit[field]["sha256"]:
            raise FullPetalLaunchError(f"CRS-EPS G0 {field} digest differs")
    selection = preregistrations["selection"]
    margins = preregistrations["margins"]
    common_bindings = {
        "commit_sha": commit_sha,
        "resolved_config_sha256": resolved_config_sha256,
        "scientific_config_sha256": scientific_config_sha256,
        "data_identity_sha256": data_identity_sha256,
        "episode_manifest_sha256": manifest["manifest_sha256"],
        "sampling_specs_sha256": manifest["sampling_specs_sha256"],
    }
    for label, preregistration in preregistrations.items():
        if any(preregistration[key] != value for key, value in common_bindings.items()):
            raise FullPetalLaunchError(
                f"CRS-EPS G0 {label} bindings differ from the launch"
            )
    if margins["selection_artifact_sha256"] != audit["selection"]["sha256"]:
        raise FullPetalLaunchError("CRS-EPS G0 margins do not bind the selection")
    try:
        recomputed_gate = evaluate_gold_audit(audit["rows"], margins)
    except CrsEpsGoldGateError as exc:
        raise FullPetalLaunchError(f"CRS-EPS G0 rows are invalid: {exc}") from exc
    if recomputed_gate != audit["gate"] or recomputed_gate["status"] != "PASS":
        raise FullPetalLaunchError("CRS-EPS G0 gate does not reproduce as PASS")
    selected = {
        (sample["video_id"], sample["draw_index"])
        for sample in selection["samples"]
    }
    observed = {
        (row["video_id"], row["draw_index"])
        for row in audit["rows"]
    }
    if observed != selected:
        raise FullPetalLaunchError("CRS-EPS G0 rows differ from the signed selection")
    videos = {video["video_id"]: video for video in manifest["videos"]}
    if any(
        video_id not in videos
        or not 0 <= draw_index < videos[video_id]["draws_per_video"]
        for video_id, draw_index in selected
    ):
        raise FullPetalLaunchError("CRS-EPS G0 selection escapes the bound manifest")
    return audit, path, digest, manifest


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
        "measurement_start_after_event_id",
        "measurement_end_event_id",
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
    expected_start = "runtime-genesis" if warmup == 0 else None
    for field in (
        "measurement_start_after_event_id",
        "measurement_end_event_id",
    ):
        if not isinstance(measurements[field], str) or not measurements[field].strip():
            raise FullPetalLaunchError(f"{label} {field} is invalid")
    if expected_start is not None and measurements[
        "measurement_start_after_event_id"
    ] != expected_start:
        raise FullPetalLaunchError(f"{label} zero-warmup boundary differs")
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
        "g0_evidence",
        "profile_evidence",
    }


def _validate_profile_artifact(
    reference,
    ticket_dir,
    *,
    cfg,
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
    untrusted_receipt_reference = {
        "path": signed.get("launch_receipt_path"),
        "sha256": signed.get("launch_receipt_sha256"),
    }
    try:
        signed_receipt, receipt_path, _ = _load_reference(
            untrusted_receipt_reference,
            path.parent,
            "profile launch receipt",
        )
        receipt = verify_launch_receipt(
            signed_receipt,
            trust_root=trust_roots["profile"],
        )
        profile_trust_root = runtime_profile_trust_root(
            receipt["execution_session"]
        )
        profile = verify_payload(
            signed,
            trust_root=profile_trust_root,
            role=PROFILE_ATTESTATION_ROLE,
        )
    except (AttestationError, RuntimeAttestationError, FullPetalLaunchError) as exc:
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
            "launch_receipt_path",
            "launch_receipt_sha256",
            "slurm_job_id",
            "slurm_allocation",
            "world_size",
            "precision",
            "hardware",
            "dimensions",
            "measurements",
            "optimizer_event_trace",
            "optimizer_event_commitment",
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
    expected_receipt = {
        "schema_version": LAUNCH_RECEIPT_SCHEMA,
        "mode": PROFILE_MODE,
        "commit_sha": profile_commit,
        "source_tree_sha256": source_tree_sha256,
        "data_identity_sha256": data_identity_sha256,
        "runtime_identity_sha256": profile["runtime_identity_sha256"],
        "resolved_config_sha256": profile["resolved_config_sha256"],
        "scientific_config_sha256": scientific_config_sha256,
        "launch_ticket": {
            "path": profile_ticket_path.name,
            "sha256": profile["launch_ticket_sha256"],
        },
        "b0_artifact_sha256": profile_ticket["b0_evidence"]["sha256"],
        "review_artifact_sha256": profile_ticket["review_evidence"]["sha256"],
        "g0_artifact_sha256": (
            None
            if profile_ticket["g0_evidence"] is None
            else profile_ticket["g0_evidence"]["sha256"]
        ),
        "profile_artifact_sha256": None,
        "world_size": world_size,
        "slurm_job_id": profile["slurm_job_id"],
        "slurm_allocation": profile["slurm_allocation"],
        "execution_session": receipt["execution_session"],
    }
    if receipt != expected_receipt or receipt_path.name != Path(
        profile["launch_receipt_path"]
    ).name:
        raise FullPetalLaunchError(
            "fixed-step profile launch receipt binding differs"
        )
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
    profile_crs_eps_manifest = None
    if _cfg_get(cfg, "crs_eps_contract") is not None:
        if profile_ticket["g0_evidence"] is None:
            raise FullPetalLaunchError("CRS-EPS profile ticket lacks G0 PASS evidence")
        _, _, _, profile_crs_eps_manifest = _validate_g0_artifact(
            profile_ticket["g0_evidence"],
            profile_ticket_path.parent,
            commit_sha=profile_commit,
            source_tree_sha256=source_tree_sha256,
            config_file_sha256=profile_ticket["config_file_sha256"],
            resolved_config_sha256=profile_ticket["resolved_config_sha256"],
            scientific_config_sha256=scientific_config_sha256,
            data_identity_sha256=data_identity_sha256,
            crs_eps_contract=_cfg_get(cfg, "crs_eps_contract"),
            runtime_seed=profile_ticket["runtime_identity"]["seed"],
            trust_root=trust_roots["g0"],
        )
    elif profile_ticket["g0_evidence"] is not None:
        raise FullPetalLaunchError("non-CRS profile ticket cannot contain G0 evidence")
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
    try:
        trace_path, trace_bytes = read_verified_bundle_bytes(
            profile["optimizer_event_trace"],
            path.parent,
            "fixed-step optimizer-event trace",
        )
        commitment_path, commitment_bytes = read_verified_bundle_bytes(
            profile["optimizer_event_commitment"],
            path.parent,
            "fixed-step optimizer-event commitment",
        )
    except EvidenceBundleError as exc:
        raise FullPetalLaunchError(str(exc)) from exc
    try:
        from .full_petal_training_evidence import (
            derive_fixed_step_profile_measurements,
        )
        from .full_petal_identity import derive_training_trace_identity

        expected_trace_identity = derive_training_trace_identity(
            cfg,
            profile_ticket["data_identity"],
            seed=profile_ticket["runtime_identity"]["seed"],
            world_size=world_size,
            crs_eps_manifest=profile_crs_eps_manifest,
        )
        derived_measurements = derive_fixed_step_profile_measurements(
            trace_path,
            commitment_path,
            warmup_optimizer_events=warmup_events,
            measured_optimizer_events=measured_events,
            expected_identity=expected_trace_identity,
            profiler_measurements=profile["measurements"],
            trace_bytes=trace_bytes,
            commitment_bytes=commitment_bytes,
            runtime_binding=receipt["execution_session"],
        )
        _validate_crs_eps_profile_measurements(
            cfg,
            derived_measurements,
            measured_events=measured_events,
        )
    except (ImportError, ValueError) as exc:
        raise FullPetalLaunchError(
            f"fixed-step optimizer-event evidence is invalid: {exc}"
        ) from exc
    if profile["measurements"] != derived_measurements:
        raise FullPetalLaunchError(
            "fixed-step profile measurements differ from the committed event trace"
        )
    return profile, path, digest


def build_fixed_step_profile_artifact(
    authorization,
    cfg,
    *,
    optimizer_event_trace_path,
    optimizer_event_commitment_path,
    bundle_root,
    precision,
    gpu_name,
    torch_version,
    cuda_version,
    profiler_measurements,
):
    """Build and sign the only profile artifact accepted by the formal gate."""

    state = _authorization_state(authorization)
    if authorization.mode != PROFILE_MODE:
        raise FullPetalLaunchError("profile artifact requires profile-mode authorization")
    if state.receipt_path is None or state.receipt_sha256 is None:
        raise FullPetalLaunchError(
            "profile artifact requires a persisted launch receipt"
        )
    if precision not in {"fp32", "fp16", "bf16"}:
        raise FullPetalLaunchError("profile precision must be fp32, fp16, or bf16")
    try:
        from .full_petal_training_evidence import (
            derive_fixed_step_profile_measurements,
        )
        from .full_petal_identity import derive_training_trace_identity

        expected_trace_identity = derive_training_trace_identity(
            cfg,
            authorization.data_identity,
            seed=authorization.runtime_identity["seed"],
            world_size=authorization.world_size,
            crs_eps_manifest=authorization.crs_eps_manifest,
        )
        trace_path, trace_bytes = read_stable_file_bytes(
            optimizer_event_trace_path, "profile optimizer-event trace"
        )
        commitment_path, commitment_bytes = read_stable_file_bytes(
            optimizer_event_commitment_path,
            "profile optimizer-event commitment",
        )
        measurements = derive_fixed_step_profile_measurements(
            trace_path,
            commitment_path,
            warmup_optimizer_events=authorization.warmup_optimizer_events,
            measured_optimizer_events=authorization.measured_optimizer_events,
            expected_identity=expected_trace_identity,
            profiler_measurements=profiler_measurements,
            trace_bytes=trace_bytes,
            commitment_bytes=commitment_bytes,
            runtime_binding=authorization.execution_session,
        )
        _validate_crs_eps_profile_measurements(
            cfg,
            measurements,
            measured_events=authorization.measured_optimizer_events,
        )
    except (ImportError, ValueError, EvidenceBundleError) as exc:
        raise FullPetalLaunchError(
            f"cannot derive fixed-step profile measurements: {exc}"
        ) from exc
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
        "launch_ticket_path": relative_bundle_path(
            authorization.ticket_path,
            bundle_root,
            "profile launch ticket",
        ),
        "launch_ticket_sha256": authorization.ticket_sha256,
        "launch_receipt_path": relative_bundle_path(
            state.receipt_path,
            bundle_root,
            "profile launch receipt",
        ),
        "launch_receipt_sha256": state.receipt_sha256,
        "slurm_job_id": authorization.slurm_job_id,
        "slurm_allocation": asdict(authorization.slurm_allocation),
        "world_size": authorization.world_size,
        "precision": precision,
        "hardware": hardware,
        "dimensions": _profile_dimensions(cfg),
        "measurements": dict(measurements),
        "optimizer_event_trace": {
            "path": relative_bundle_path(
                trace_path, bundle_root, "profile optimizer-event trace"
            ),
            "sha256": hashlib.sha256(trace_bytes).hexdigest(),
        },
        "optimizer_event_commitment": {
            "path": relative_bundle_path(
                commitment_path,
                bundle_root,
                "profile optimizer-event commitment",
            ),
            "sha256": hashlib.sha256(commitment_bytes).hexdigest(),
        },
    }
    try:
        return sign_fixed_step_profile(
            body, runtime_session=state.runtime_session
        )
    except (AttestationError, RuntimeAttestationError) as exc:
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
        "trusted_scontrol_path",
        "attestation_trust_roots",
        "evidence_trust_model",
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
    if (
        not isinstance(contract["trusted_scontrol_path"], str)
        or not Path(contract["trusted_scontrol_path"]).is_absolute()
    ):
        raise FullPetalLaunchError(
            "launch_contract trusted_scontrol_path must be absolute"
        )
    _evidence_trust_model(contract)
    _trust_roots(contract)
    return contract


def _evidence_trust_model(contract):
    model = contract["evidence_trust_model"]
    if not isinstance(model, Mapping):
        raise FullPetalLaunchError(
            "launch_contract requires an evidence_trust_model mapping"
        )
    required = {
        "schema_version",
        "purpose",
        "trusted_computing_base",
        "guarantees",
        "out_of_scope",
        "key_compromise_action",
    }
    _require_exact_fields(model, required, "evidence_trust_model")
    expected = {
        "schema_version": EVIDENCE_TRUST_MODEL_SCHEMA,
        "purpose": "scientific_reproducibility",
        "trusted_computing_base": list(_EVIDENCE_TRUSTED_COMPUTING_BASE),
        "guarantees": list(_EVIDENCE_GUARANTEES),
        "out_of_scope": list(_EVIDENCE_OUT_OF_SCOPE),
        "key_compromise_action": "BLOCK_ROTATE_AND_RERUN",
    }
    if dict(model) != expected:
        raise FullPetalLaunchError(
            "evidence_trust_model must match the locked scientific-reproducibility "
            "trust boundary"
        )
    return model


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
    expected_unit = (
        "video_group_optimizer_event"
        if _cfg_get(cfg, "crs_eps_contract") is not None
        else "optimizer_event"
    )
    if contract.get("step_unit") != expected_unit:
        raise FullPetalLaunchError(f"profile step_unit must be {expected_unit}")
    if _cfg_get(cfg, "crs_eps_contract") is not None and tuple(
        contract.get("required_workload_denominators", ())
    ) != _CRS_EPS_PROFILE_DENOMINATORS:
        raise FullPetalLaunchError(
            "CRS-EPS profile workload denominators differ from the frozen contract"
        )
    if contract.get("submit_via_slurm_only") is not True:
        raise FullPetalLaunchError("Full PETAL profile must be submitted through Slurm")
    return warmup, measured, world_size


def _validate_crs_eps_profile_measurements(cfg, measurements, *, measured_events):
    contract = _cfg_get(cfg, "crs_eps_contract")
    if contract is None:
        return
    workload = measurements.get("workload") if isinstance(measurements, Mapping) else None
    if not isinstance(workload, Mapping) or not isinstance(workload.get("totals"), Mapping):
        raise FullPetalLaunchError(
            "CRS-EPS fixed-step profile requires multi-denominator workload evidence"
        )
    draws_per_video = _require_nonnegative_int(
        contract.get("draws_per_video"), "CRS-EPS draws per video", positive=True
    )
    totals = workload["totals"]
    if totals["optimizer_events"] != measured_events:
        raise FullPetalLaunchError("CRS-EPS workload optimizer-event count differs")
    if totals["episode_draws"] != measured_events * draws_per_video:
        raise FullPetalLaunchError("CRS-EPS workload draw count differs from M")
    if totals["replay_tokens"] != totals["temporal_forward_tokens"]:
        raise FullPetalLaunchError("cached CRS-EPS replay/forward token counts differ")
    if totals["visual_forward_frames"] or totals["visual_backward_frames"]:
        raise FullPetalLaunchError(
            "cached CRS-EPS profile must report zero visual frame workload"
        )


def _validate_environment(
    environ,
    world_size,
    *,
    slurm_allocation=None,
    expected_user=None,
    scontrol_path=None,
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
        allocation = slurm_allocation or inspect_slurm_allocation(
            slurm_job_id, scontrol_path=scontrol_path
        )
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
    bundle_root,
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
            bundle_root=bundle_root,
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
    g0_path=None,
    profile_path=None,
    repository_root,
    entrypoint,
    seed,
    run_id,
    deterministic,
    not_eval,
    resume_path,
    cfg_overrides,
    bundle_root,
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
        bundle_root=bundle_root,
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
    b0_reference = _file_reference(b0_path, bundle_root, "B0 evidence")
    review_reference = _file_reference(
        review_path, bundle_root, "independent review evidence"
    )
    _, _, b0_digest = _validate_b0_artifact(
        b0_reference,
        bundle_root,
        commit_sha,
        trust_root=trust_roots["b0"],
        repository_root=root,
    )
    _validate_review_artifact(
        review_reference,
        bundle_root,
        commit_sha=commit_sha,
        b0_sha256=b0_digest,
        reviewer_id=contract["required_reviewer_id"],
        required_scope=contract["required_review_scope"],
        trust_root=trust_roots["review"],
    )
    g0_reference = None
    if _cfg_get(cfg, "crs_eps_contract") is not None:
        if g0_path is None:
            raise FullPetalLaunchError(
                "CRS-EPS launch ticket requires signed G0 PASS evidence"
            )
        g0_reference = _file_reference(
            g0_path, bundle_root, "CRS-EPS G0 audit evidence"
        )
        _validate_g0_artifact(
            g0_reference,
            bundle_root,
            commit_sha=commit_sha,
            source_tree_sha256=source_digest,
            config_file_sha256=sha256_file(config_path),
            resolved_config_sha256=resolved_digest,
            scientific_config_sha256=scientific_digest,
            data_identity_sha256=data_identity["identity_sha256"],
            crs_eps_contract=_cfg_get(cfg, "crs_eps_contract"),
            runtime_seed=runtime_identity["seed"],
            trust_root=trust_roots["g0"],
        )
    elif g0_path is not None:
        raise FullPetalLaunchError("non-CRS launch ticket cannot contain G0 evidence")
    profile_reference = None
    if mode == FORMAL_MODE:
        if profile_path is None:
            raise FullPetalLaunchError("formal launch ticket requires profile evidence")
        profile_reference = _file_reference(
            profile_path, bundle_root, "fixed-step profile evidence"
        )
        _validate_profile_artifact(
            profile_reference,
            bundle_root,
            cfg=cfg,
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
        "g0_evidence": g0_reference,
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
    execution_signing_key_path=None,
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
    if execution_signing_key_path is None:
        raise FullPetalLaunchError(
            "Full PETAL launch requires an external execution signing key"
        )
    try:
        signer_public_key = public_key_base64(execution_signing_key_path)
    except AttestationError as exc:
        raise FullPetalLaunchError(f"execution signing key is invalid: {exc}") from exc
    if signer_public_key != trust_roots["profile"]["public_key"]:
        raise FullPetalLaunchError(
            "execution signing key does not match the trusted public key"
        )
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
    ticket_path = Path(ticket_path).expanduser().resolve()
    runtime_identity = _runtime_identity(
        entrypoint=entrypoint,
        seed=seed,
        run_id=run_id,
        deterministic=deterministic,
        not_eval=not_eval,
        resume_path=resume_path,
        cfg_overrides=cfg_overrides,
        bundle_root=ticket_path.parent,
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
        scontrol_path=contract["trusted_scontrol_path"],
    )
    root = Path(repository_root or Path(config_path).resolve().parents[2]).resolve()
    state = repository_state or inspect_repository(root)
    if not state.clean:
        raise FullPetalLaunchError(
            "Full PETAL launch requires a clean git checkout: " + state.status
        )
    commit_sha = _require_git_sha(state.commit_sha, "repository commit")

    try:
        _, ticket_bytes = read_stable_file_bytes(ticket_path, "launch ticket")
        ticket = strict_json_from_bytes(
            ticket_bytes, "launch ticket", require_object=True
        )
    except EvidenceBundleError as exc:
        raise FullPetalLaunchError(str(exc)) from exc
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

    g0_digest = None
    crs_eps_manifest = None
    if _cfg_get(cfg, "crs_eps_contract") is not None:
        if ticket["g0_evidence"] is None:
            raise FullPetalLaunchError("CRS-EPS launch ticket lacks G0 PASS evidence")
        _, _, g0_digest, crs_eps_manifest = _validate_g0_artifact(
            ticket["g0_evidence"],
            ticket_path.parent,
            commit_sha=commit_sha,
            source_tree_sha256=source_digest,
            config_file_sha256=ticket["config_file_sha256"],
            resolved_config_sha256=resolved_digest,
            scientific_config_sha256=scientific_digest,
            data_identity_sha256=ticket["data_identity"]["identity_sha256"],
            crs_eps_contract=_cfg_get(cfg, "crs_eps_contract"),
            runtime_seed=runtime_identity["seed"],
            trust_root=trust_roots["g0"],
        )
    elif ticket["g0_evidence"] is not None:
        raise FullPetalLaunchError("non-CRS launch ticket cannot contain G0 evidence")

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
            cfg=cfg,
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

    runtime_session = issue_runtime_session()
    values = {
        "mode": mode,
        "commit_sha": commit_sha,
        "source_tree_sha256": source_digest,
        "data_identity": dict(ticket["data_identity"]),
        "data_identity_sha256": ticket["data_identity"]["identity_sha256"],
        "runtime_identity_sha256": canonical_json_sha256(runtime_identity),
        "runtime_identity": dict(runtime_identity),
        "resolved_config_sha256": resolved_digest,
        "scientific_config_sha256": scientific_digest,
        "ticket_path": str(ticket_path),
        "ticket_sha256": hashlib.sha256(ticket_bytes).hexdigest(),
        "b0_artifact_sha256": b0_digest,
        "review_artifact_sha256": review_digest,
        "g0_artifact_sha256": g0_digest,
        "crs_eps_manifest": crs_eps_manifest,
        "profile_artifact_sha256": profile_digest,
        "warmup_optimizer_events": warmup,
        "measured_optimizer_events": measured,
        "world_size": world_size,
        "slurm_job_id": slurm_job_id,
        "slurm_allocation": allocation,
        "execution_attestation_key_id": trust_roots["profile"]["key_id"],
        "execution_attestation_public_key": trust_roots["profile"]["public_key"],
        "execution_session": runtime_session.binding,
    }
    authorization = object.__new__(FullPetalLaunchAuthorization)
    for field in FullPetalLaunchAuthorization.__dataclass_fields__:
        object.__setattr__(authorization, field, values.pop(field))
    if values:
        raise FullPetalLaunchError(
            f"internal launch authorization fields differ: {sorted(values)}"
        )
    state = _AuthorizationRuntimeState(
        runtime_session=runtime_session,
        authorization_snapshot=_authorization_snapshot(authorization),
        repository_root=root,
        expected_slurm_user=expected_slurm_user,
        trusted_scontrol_path=contract["trusted_scontrol_path"],
    )
    authorization_id = id(authorization)

    def retire(reference):
        current = _AUTHORIZATION_STATES.get(authorization_id)
        if current is not None and current[0] is reference:
            _AUTHORIZATION_STATES.pop(authorization_id, None)

    _AUTHORIZATION_STATES[authorization_id] = (weakref.ref(authorization, retire), state)
    return authorization


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
    "LAUNCH_RECEIPT_ATTESTATION_ROLE",
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
    "verify_launch_receipt",
    "resolved_config_payload",
    "resolved_config_sha256",
    "runtime_evidence_session",
    "sha256_file",
    "validate_full_petal_launch",
]
