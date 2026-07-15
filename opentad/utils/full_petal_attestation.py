"""Detached trust roots for Full PETAL launch evidence."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


ATTESTATION_SCHEMA = "full-petal-ed25519-attestation-v1"
ATTESTATION_ALGORITHM = "ed25519"
ATTESTATION_FIELD = "attestation"


class AttestationError(ValueError):
    """Raised when signed evidence is malformed or untrusted."""


def _canonical_bytes(value):
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AttestationError(f"attested payload is not canonical JSON: {exc}") from exc


def _message(payload, role):
    if not isinstance(role, str) or not role.strip():
        raise AttestationError("attestation role must be non-empty text")
    domain = f"{ATTESTATION_SCHEMA}:{role}\n".encode("ascii")
    return domain + _canonical_bytes(payload)


def _read_private_key(path):
    path = Path(path)
    try:
        key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    except (OSError, ValueError, TypeError) as exc:
        raise AttestationError(f"failed to load Ed25519 private key {path}: {exc}") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise AttestationError(f"private key is not Ed25519: {path}")
    return key


def _decode_public_key(value):
    if not isinstance(value, str) or not value.strip():
        raise AttestationError("trusted public key must be base64 text")
    try:
        raw = base64.b64decode(value, validate=True)
        return Ed25519PublicKey.from_public_bytes(raw), raw
    except (ValueError, TypeError) as exc:
        raise AttestationError("trusted public key is not a valid Ed25519 key") from exc


def public_key_base64(private_key_path):
    key = _read_private_key(private_key_path).public_key()
    raw = key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return base64.b64encode(raw).decode("ascii")


def public_key_sha256(public_key_base64_value):
    _, raw = _decode_public_key(public_key_base64_value)
    return hashlib.sha256(raw).hexdigest()


def generate_private_key(path):
    """Create a new PEM Ed25519 key without ever returning private material."""

    path = Path(path)
    if path.exists():
        raise AttestationError(f"refusing to overwrite private key: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    key = Ed25519PrivateKey.generate()
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    path.write_bytes(pem)
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return public_key_base64(path)


def verify_payload(payload, *, trust_root, role):
    """Verify an attested object against a configured key, returning its body."""

    if not isinstance(payload, Mapping):
        raise AttestationError("attested payload must be an object")
    if not isinstance(trust_root, Mapping):
        raise AttestationError(f"{role} trust root must be an object")
    if set(trust_root) != {"key_id", "public_key"}:
        raise AttestationError(
            f"{role} trust root requires exactly key_id and public_key"
        )
    attestation = payload.get(ATTESTATION_FIELD)
    if not isinstance(attestation, Mapping):
        raise AttestationError(f"{role} evidence lacks a signed attestation")
    expected_fields = {
        "schema_version",
        "algorithm",
        "role",
        "key_id",
        "public_key_sha256",
        "signature",
    }
    if set(attestation) != expected_fields:
        raise AttestationError(f"{role} attestation fields are not exact")
    if (
        attestation["schema_version"] != ATTESTATION_SCHEMA
        or attestation["algorithm"] != ATTESTATION_ALGORITHM
        or attestation["role"] != role
    ):
        raise AttestationError(f"{role} attestation domain is invalid")
    if attestation["key_id"] != trust_root["key_id"]:
        raise AttestationError(f"{role} attestation key_id is not trusted")
    public_key, raw_public = _decode_public_key(trust_root["public_key"])
    expected_public_digest = hashlib.sha256(raw_public).hexdigest()
    if attestation["public_key_sha256"] != expected_public_digest:
        raise AttestationError(f"{role} attestation public key digest differs")
    try:
        signature = base64.b64decode(attestation["signature"], validate=True)
        body = {key: copy.deepcopy(value) for key, value in payload.items() if key != ATTESTATION_FIELD}
        public_key.verify(signature, _message(body, role))
    except (ValueError, TypeError, InvalidSignature) as exc:
        raise AttestationError(f"{role} attestation signature is invalid") from exc
    return body


__all__ = [
    "ATTESTATION_FIELD",
    "ATTESTATION_SCHEMA",
    "AttestationError",
    "generate_private_key",
    "public_key_base64",
    "public_key_sha256",
    "verify_payload",
]
