"""Test-only helper for constructing deliberately malformed signed fixtures."""

import base64
import copy
import hashlib
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from opentad.utils.full_petal_attestation import (
    ATTESTATION_ALGORITHM,
    ATTESTATION_FIELD,
    ATTESTATION_SCHEMA,
    _message,
)


def attest_fixture(payload, *, private_key_path, key_id, role):
    body = copy.deepcopy(dict(payload))
    key = serialization.load_pem_private_key(
        Path(private_key_path).read_bytes(), password=None
    )
    assert isinstance(key, Ed25519PrivateKey)
    public = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    body[ATTESTATION_FIELD] = {
        "schema_version": ATTESTATION_SCHEMA,
        "algorithm": ATTESTATION_ALGORITHM,
        "role": role,
        "key_id": key_id,
        "public_key_sha256": hashlib.sha256(public).hexdigest(),
        "signature": base64.b64encode(key.sign(_message(payload, role))).decode("ascii"),
    }
    return body


def committed_optimizer_envelope(runtime_session, payload):
    proof = runtime_session.begin_optimizer_boundary()
    runtime_session._confirm_optimizer_step_completed(proof)
    runtime_session._confirm_transaction_commit_completed(proof)
    return runtime_session.sign_committed_optimizer_event(payload, proof)
