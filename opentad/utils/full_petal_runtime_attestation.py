"""One-launch, in-memory attestation for optimizer and profile evidence."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import secrets
from collections.abc import Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .full_petal_attestation import (
    ATTESTATION_ALGORITHM,
    ATTESTATION_FIELD,
    ATTESTATION_SCHEMA,
    AttestationError,
    _message,
)


RUNTIME_SESSION_SCHEMA = "full-petal-runtime-session-v1"
RUNTIME_EVENT_SCHEMA = "full-petal-runtime-event-v1"
RUNTIME_PROFILE_ROLE = "fixed-step-profile"
RUNTIME_GENESIS_HASH = "0" * 64
RUNTIME_EVENT_FIELDS = {
    "runtime_session_id",
    "runtime_sequence",
    "runtime_previous_hash",
    "runtime_event_hash",
    "runtime_signature",
}


class RuntimeAttestationError(ValueError):
    pass


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
        raise RuntimeAttestationError(
            f"runtime evidence is not canonical JSON: {exc}"
        ) from exc


def _event_message(envelope):
    return (RUNTIME_EVENT_SCHEMA + "\n").encode("ascii") + _canonical_bytes(envelope)


def _public_key_record(key, session_id):
    raw = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return {
        "schema_version": RUNTIME_SESSION_SCHEMA,
        "session_id": session_id,
        "public_key": base64.b64encode(raw).decode("ascii"),
        "public_key_sha256": hashlib.sha256(raw).hexdigest(),
    }


class RuntimeEvidenceSession:
    """Ephemeral signer whose private key never leaves the live launch process."""

    def __init__(self, *, _key, _session_id):
        if not isinstance(_key, Ed25519PrivateKey):
            raise RuntimeAttestationError(
                "runtime sessions can only be issued by the launch validator"
            )
        self.__key = _key
        self.__session_id = _session_id
        self.__sequence = 0
        self.__head = RUNTIME_GENESIS_HASH
        self.__profile_issued = False

    @property
    def binding(self):
        return copy.deepcopy(_public_key_record(self.__key, self.__session_id))

    def state_dict(self):
        return {
            "session_id": self.__session_id,
            "sequence": self.__sequence,
            "head": self.__head,
            "profile_issued": self.__profile_issued,
        }

    def load_state_dict(self, state):
        if not isinstance(state, Mapping) or set(state) != {
            "session_id",
            "sequence",
            "head",
            "profile_issued",
        }:
            raise RuntimeAttestationError("runtime session state fields differ")
        if state["session_id"] != self.__session_id:
            raise RuntimeAttestationError("runtime session identity cannot change")
        sequence = state["sequence"]
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise RuntimeAttestationError("runtime session sequence is invalid")
        head = state["head"]
        if (
            not isinstance(head, str)
            or len(head) != 64
            or any(character not in "0123456789abcdef" for character in head)
        ):
            raise RuntimeAttestationError("runtime session head is invalid")
        if not isinstance(state["profile_issued"], bool):
            raise RuntimeAttestationError("runtime profile state is invalid")
        self.__sequence = sequence
        self.__head = head
        self.__profile_issued = state["profile_issued"]

    def sign_event(self, event_kind, payload):
        if event_kind not in {"optimizer-event", "visual-parameter-event"}:
            raise RuntimeAttestationError("runtime event kind is unsupported")
        if not isinstance(payload, Mapping) or set(payload) & RUNTIME_EVENT_FIELDS:
            raise RuntimeAttestationError("runtime event payload is invalid")
        unsigned = {
            "schema_version": RUNTIME_EVENT_SCHEMA,
            "session_id": self.__session_id,
            "sequence": self.__sequence,
            "previous_hash": self.__head,
            "event_kind": event_kind,
            "payload_sha256": hashlib.sha256(_canonical_bytes(payload)).hexdigest(),
        }
        event_hash = hashlib.sha256(_event_message(unsigned)).hexdigest()
        signature = self.__key.sign(_event_message({**unsigned, "event_hash": event_hash}))
        envelope = {
            "runtime_session_id": self.__session_id,
            "runtime_sequence": self.__sequence,
            "runtime_previous_hash": self.__head,
            "runtime_event_hash": event_hash,
            "runtime_signature": base64.b64encode(signature).decode("ascii"),
        }
        self.__sequence += 1
        self.__head = event_hash
        return envelope

    def sign_profile(self, payload):
        if self.__profile_issued:
            raise RuntimeAttestationError("runtime session already issued a profile")
        if not isinstance(payload, Mapping) or ATTESTATION_FIELD in payload:
            raise RuntimeAttestationError("runtime profile payload is invalid")
        body = copy.deepcopy(dict(payload))
        binding = self.binding
        signature = self.__key.sign(_message(body, RUNTIME_PROFILE_ROLE))
        body[ATTESTATION_FIELD] = {
            "schema_version": ATTESTATION_SCHEMA,
            "algorithm": ATTESTATION_ALGORITHM,
            "role": RUNTIME_PROFILE_ROLE,
            "key_id": binding["session_id"],
            "public_key_sha256": binding["public_key_sha256"],
            "signature": base64.b64encode(signature).decode("ascii"),
        }
        self.__profile_issued = True
        return body


def issue_runtime_session():
    """Issue one fresh ephemeral session before CUDA/DDP initialization."""

    return RuntimeEvidenceSession(
        _key=Ed25519PrivateKey.generate(),
        _session_id=secrets.token_hex(32),
    )


def _public_key(binding):
    if not isinstance(binding, Mapping) or set(binding) != {
        "schema_version",
        "session_id",
        "public_key",
        "public_key_sha256",
    }:
        raise RuntimeAttestationError("runtime session binding fields differ")
    if binding["schema_version"] != RUNTIME_SESSION_SCHEMA:
        raise RuntimeAttestationError("runtime session schema differs")
    try:
        raw = base64.b64decode(binding["public_key"], validate=True)
        key = Ed25519PublicKey.from_public_bytes(raw)
    except (ValueError, TypeError) as exc:
        raise RuntimeAttestationError("runtime session public key is invalid") from exc
    if hashlib.sha256(raw).hexdigest() != binding["public_key_sha256"]:
        raise RuntimeAttestationError("runtime session public key digest differs")
    session_id = binding["session_id"]
    if (
        not isinstance(session_id, str)
        or len(session_id) != 64
        or any(character not in "0123456789abcdef" for character in session_id)
    ):
        raise RuntimeAttestationError("runtime session ID is invalid")
    return key


def verify_runtime_event(payload, envelope, *, event_kind, binding):
    key = _public_key(binding)
    if not isinstance(payload, Mapping):
        raise RuntimeAttestationError("runtime event payload must be an object")
    if not isinstance(envelope, Mapping) or set(envelope) != RUNTIME_EVENT_FIELDS:
        raise RuntimeAttestationError("runtime event envelope fields differ")
    unsigned = {
        "schema_version": RUNTIME_EVENT_SCHEMA,
        "session_id": envelope["runtime_session_id"],
        "sequence": envelope["runtime_sequence"],
        "previous_hash": envelope["runtime_previous_hash"],
        "event_kind": event_kind,
        "payload_sha256": hashlib.sha256(_canonical_bytes(payload)).hexdigest(),
    }
    if unsigned["session_id"] != binding["session_id"]:
        raise RuntimeAttestationError("runtime event session differs")
    event_hash = hashlib.sha256(_event_message(unsigned)).hexdigest()
    if envelope["runtime_event_hash"] != event_hash:
        raise RuntimeAttestationError("runtime event hash differs")
    try:
        signature = base64.b64decode(envelope["runtime_signature"], validate=True)
        key.verify(signature, _event_message({**unsigned, "event_hash": event_hash}))
    except (ValueError, TypeError, InvalidSignature) as exc:
        raise RuntimeAttestationError("runtime event signature is invalid") from exc
    return event_hash


def runtime_profile_trust_root(binding):
    _public_key(binding)
    return {"key_id": binding["session_id"], "public_key": binding["public_key"]}


__all__ = [
    "RUNTIME_EVENT_FIELDS",
    "RUNTIME_EVENT_SCHEMA",
    "RUNTIME_GENESIS_HASH",
    "RUNTIME_PROFILE_ROLE",
    "RUNTIME_SESSION_SCHEMA",
    "RuntimeAttestationError",
    "RuntimeEvidenceSession",
    "issue_runtime_session",
    "runtime_profile_trust_root",
    "verify_runtime_event",
]
