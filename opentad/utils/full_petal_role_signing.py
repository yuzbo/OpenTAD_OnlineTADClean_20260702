"""Fixed-role, schema-validating signers for Full PETAL evidence producers.

This module intentionally has no signer that accepts a caller-selected role. Each
entry point fixes both the attestation domain and the payload schema before it
loads private key material.
"""

from __future__ import annotations

import base64
import copy
import hashlib
from collections.abc import Mapping
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .full_petal_attestation import (
    ATTESTATION_ALGORITHM,
    ATTESTATION_FIELD,
    ATTESTATION_SCHEMA,
    AttestationError,
    _message,
)
from .crs_eps_gold_gate import (
    AUDIT_ATTESTATION_ROLE,
    AUDIT_SCHEMA_VERSION,
    MARGIN_ATTESTATION_ROLE,
    MARGIN_SCHEMA_VERSION,
    SELECTION_ATTESTATION_ROLE,
    SELECTION_SCHEMA_VERSION,
    validate_gold_margins,
    validate_gold_selection,
)


def _validated_body(payload, *, fields, schema):
    if not isinstance(payload, Mapping):
        raise AttestationError("attested payload must be an object")
    if ATTESTATION_FIELD in payload:
        raise AttestationError("payload is already attested")
    if set(payload) != set(fields):
        raise AttestationError("fixed-role attestation payload fields differ")
    if payload.get("schema_version") != schema:
        raise AttestationError("fixed-role attestation payload schema differs")
    return copy.deepcopy(dict(payload))


def _validated_key_id(key_id):
    if not isinstance(key_id, str) or not key_id.strip():
        raise AttestationError("attestation key_id must be non-empty text")
    return key_id


def _load_ed25519_key(private_key_path):
    path = Path(private_key_path)
    try:
        key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    except (OSError, ValueError, TypeError) as exc:
        raise AttestationError(f"failed to load Ed25519 private key {path}: {exc}") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise AttestationError(f"private key is not Ed25519: {path}")
    return key


def _public_digest(key):
    raw = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()


def _sign_body(body, *, private_key_path, key_id, role):
    key_id = _validated_key_id(key_id)
    key = _load_ed25519_key(private_key_path)
    signed = copy.deepcopy(dict(body))
    signed[ATTESTATION_FIELD] = {
        "schema_version": ATTESTATION_SCHEMA,
        "algorithm": ATTESTATION_ALGORITHM,
        "role": role,
        "key_id": key_id,
        "public_key_sha256": _public_digest(key),
        "signature": base64.b64encode(key.sign(_message(body, role))).decode("ascii"),
    }
    return signed


def sign_crs_eps_g0_selection(payload, *, private_key_path, key_id):
    body = _validated_body(
        payload,
        fields={
            "schema_version",
            "status",
            "commit_sha",
            "resolved_config_sha256",
            "scientific_config_sha256",
            "data_identity_sha256",
            "episode_manifest_sha256",
            "sampling_specs_sha256",
            "samples",
        },
        schema=SELECTION_SCHEMA_VERSION,
    )
    validate_gold_selection(body)
    return _sign_body(
        body,
        private_key_path=private_key_path,
        key_id=key_id,
        role=SELECTION_ATTESTATION_ROLE,
    )


def sign_crs_eps_g0_margins(payload, *, private_key_path, key_id):
    body = _validated_body(
        payload,
        fields={
            "schema_version",
            "status",
            "commit_sha",
            "resolved_config_sha256",
            "scientific_config_sha256",
            "data_identity_sha256",
            "episode_manifest_sha256",
            "sampling_specs_sha256",
            "selection_artifact_sha256",
            "min_gradient_cosine",
            "min_gradient_sign_agreement",
            "min_runtime_continuous_cosine",
            "max_relative_loss_error",
            "require_runtime_discrete_equal",
            "max_mean_dynamic_replay_ratio",
            "max_video_start_fallback_fraction",
            "min_dynamic_minus_fixed_gradient_cosine",
            "min_dynamic_minus_reset_gradient_cosine",
        },
        schema=MARGIN_SCHEMA_VERSION,
    )
    validate_gold_margins(body)
    return _sign_body(
        body,
        private_key_path=private_key_path,
        key_id=key_id,
        role=MARGIN_ATTESTATION_ROLE,
    )


def sign_crs_eps_g0_audit(payload, *, private_key_path, key_id):
    body = _validated_body(
        payload,
        fields={
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
        schema=AUDIT_SCHEMA_VERSION,
    )
    if body["status"] not in {"PASS", "KILL"}:
        raise AttestationError("G0 audit signer requires a terminal PASS or KILL")
    if not isinstance(body["gate"], Mapping) or body["gate"].get("status") != body["status"]:
        raise AttestationError("G0 audit status differs from its gate")
    return _sign_body(
        body,
        private_key_path=private_key_path,
        key_id=key_id,
        role=AUDIT_ATTESTATION_ROLE,
    )


def sign_b0_evidence(payload, *, private_key_path, key_id):
    body = _validated_body(
        payload,
        fields={
            "schema_version",
            "status",
            "commit_sha",
            "test_count",
            "blocking_findings",
            "protocol_violations",
            "manifest_path",
            "manifest_sha256",
            "test_report_path",
            "test_report_sha256",
            "audit_report_path",
            "audit_report_sha256",
        },
        schema="full-petal-b0-v2",
    )
    if body["status"] != "PASS":
        raise AttestationError("B0 signer only attests a PASS root")
    key_id = _validated_key_id(key_id)
    key = _load_ed25519_key(private_key_path)
    role = "b0-runner"
    body[ATTESTATION_FIELD] = {
        "schema_version": ATTESTATION_SCHEMA,
        "algorithm": ATTESTATION_ALGORITHM,
        "role": role,
        "key_id": key_id,
        "public_key_sha256": _public_digest(key),
        "signature": base64.b64encode(key.sign(_message(payload, role))).decode("ascii"),
    }
    return body


def sign_independent_review(payload, *, private_key_path, key_id):
    body = _validated_body(
        payload,
        fields={
            "schema_version",
            "reviewer_id",
            "reviewed_commit",
            "b0_artifact_sha256",
            "scope",
            "verdict",
            "blocking_findings",
            "protocol_violations",
        },
        schema="full-petal-independent-review-v2",
    )
    if body["reviewer_id"] != key_id:
        raise AttestationError("review signer key_id must equal reviewer_id")
    key_id = _validated_key_id(key_id)
    key = _load_ed25519_key(private_key_path)
    role = "independent-reviewer"
    body[ATTESTATION_FIELD] = {
        "schema_version": ATTESTATION_SCHEMA,
        "algorithm": ATTESTATION_ALGORITHM,
        "role": role,
        "key_id": key_id,
        "public_key_sha256": _public_digest(key),
        "signature": base64.b64encode(key.sign(_message(payload, role))).decode("ascii"),
    }
    return body


def sign_fixed_step_profile(payload, *, runtime_session):
    body = _validated_body(
        payload,
        fields={
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
        schema="full-petal-fixed-step-profile-v3",
    )
    if body["status"] != "PASS":
        raise AttestationError("profile signer only attests a PASS profile")
    return runtime_session.sign_profile(body)


def sign_formal_run(payload, *, private_key_path, key_id):
    body = _validated_body(
        payload,
        fields={
            "schema_version",
            "claim",
            "variant",
            "seed",
            "commit_sha",
            "protocol_sha256",
            "artifacts",
        },
        schema="full-petal-formal-run-manifest-v2",
    )
    key_id = _validated_key_id(key_id)
    key = _load_ed25519_key(private_key_path)
    role = "formal-run"
    body[ATTESTATION_FIELD] = {
        "schema_version": ATTESTATION_SCHEMA,
        "algorithm": ATTESTATION_ALGORITHM,
        "role": role,
        "key_id": key_id,
        "public_key_sha256": _public_digest(key),
        "signature": base64.b64encode(key.sign(_message(payload, role))).decode("ascii"),
    }
    return body


def sign_fineaction_license_authorization(payload, *, private_key_path, key_id):
    body = _validated_body(
        payload,
        fields={
            "schema_version",
            "dataset",
            "license_id",
            "subject",
            "access_scope",
            "authorized",
            "issued_at",
            "terms",
        },
        schema="full-petal-fineaction-license-authorization-v1",
    )
    if body["dataset"] != "FineAction" or body["authorized"] is not True:
        raise AttestationError(
            "FineAction license signer requires an explicit authorized grant"
        )
    key_id = _validated_key_id(key_id)
    key = _load_ed25519_key(private_key_path)
    role = "fineaction-license-authorization"
    body[ATTESTATION_FIELD] = {
        "schema_version": ATTESTATION_SCHEMA,
        "algorithm": ATTESTATION_ALGORITHM,
        "role": role,
        "key_id": key_id,
        "public_key_sha256": _public_digest(key),
        "signature": base64.b64encode(key.sign(_message(payload, role))).decode("ascii"),
    }
    return body


__all__ = [
    "sign_crs_eps_g0_audit",
    "sign_crs_eps_g0_margins",
    "sign_crs_eps_g0_selection",
    "sign_b0_evidence",
    "sign_fixed_step_profile",
    "sign_fineaction_license_authorization",
    "sign_formal_run",
    "sign_independent_review",
]
