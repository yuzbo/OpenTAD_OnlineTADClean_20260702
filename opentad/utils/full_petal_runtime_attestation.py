"""One-launch lifecycle evidence for scientific reproducibility.

The launch validator, training engine, runtime session, and in-process signing
key are the trusted computing base. The signatures bind provenance and expose
post-publication modification; they are not hostile-process remote attestation.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import secrets
from collections.abc import Mapping
from dataclasses import dataclass

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
import torch

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
RUNTIME_STATE_DOMAIN = b"full-petal-runtime-state-v1\n"
RUNTIME_STEP_RECEIPT_DOMAIN = b"full-petal-optimizer-step-receipt-v1\n"
RUNTIME_COMMIT_RECEIPT_DOMAIN = b"full-petal-transaction-commit-receipt-v1\n"
RUNTIME_STEP_RECEIPT_SCHEMA = "full-petal-optimizer-step-receipt-v1"
RUNTIME_COMMIT_RECEIPT_SCHEMA = "full-petal-transaction-commit-receipt-v1"
RUNTIME_RECEIPT_GENESIS_HASH = "0" * 64
_BOUNDARY_STATE_FIELDS = {
    "boundary_nonce",
    "commit_receipt",
    "optimizer",
    "proof",
    "runtime_head",
    "runtime_sequence",
    "step_receipt",
    "transaction",
}
RUNTIME_EVENT_FIELDS = {
    "runtime_session_id",
    "runtime_sequence",
    "runtime_previous_hash",
    "runtime_event_hash",
    "runtime_signature",
}


class RuntimeAttestationError(ValueError):
    pass


class _OptimizerBoundaryProof:
    __slots__ = ("session_id", "nonce")

    def __init__(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeAttestationError(
            "optimizer boundary proofs are issued by the runtime session"
        )


@dataclass(frozen=True, slots=True)
class _OptimizerStepReceipt:
    payload: bytes
    signature: bytes


@dataclass(frozen=True, slots=True)
class _TransactionCommitReceipt:
    payload: bytes
    signature: bytes


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


def _object_identity(value):
    value_type = type(value)
    return {
        "python_id": id(value),
        "python_type": f"{value_type.__module__}.{value_type.__qualname__}",
    }


def _verified_receipt(receipt, expected_type, domain, public_key, label):
    if not isinstance(receipt, expected_type):
        raise RuntimeAttestationError(f"{label} is missing or has the wrong type")
    try:
        public_key.verify(receipt.signature, domain + receipt.payload)
        body = json.loads(receipt.payload.decode("utf-8"))
    except (InvalidSignature, UnicodeError, ValueError, TypeError) as exc:
        raise RuntimeAttestationError(f"{label} signature is invalid") from exc
    if not isinstance(body, dict) or _canonical_bytes(body) != receipt.payload:
        raise RuntimeAttestationError(f"{label} payload is not canonical")
    return body


def _receipt_sha256(domain, receipt):
    return hashlib.sha256(domain + receipt.payload + receipt.signature).hexdigest()


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
    """TCB-resident ephemeral signer for one validated launch process."""

    __slots__ = (
        "__active_boundary",
        "__head",
        "__key",
        "__profile_issued",
        "__sequence",
        "__session_id",
        "__visual_optimizer_event_id",
    )

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
        self.__active_boundary = None
        self.__visual_optimizer_event_id = None

    @property
    def binding(self):
        return copy.deepcopy(_public_key_record(self.__key, self.__session_id))

    def state_dict(self):
        if self.__active_boundary is not None:
            raise RuntimeAttestationError(
                "cannot snapshot an active optimizer boundary"
            )
        body = {
            "session_id": self.__session_id,
            "sequence": self.__sequence,
            "head": self.__head,
            "profile_issued": self.__profile_issued,
            "visual_optimizer_event_id": self.__visual_optimizer_event_id,
        }
        return {
            **body,
            "state_signature": base64.b64encode(
                self.__key.sign(RUNTIME_STATE_DOMAIN + _canonical_bytes(body))
            ).decode("ascii"),
        }

    def load_state_dict(self, state):
        if not isinstance(state, Mapping) or set(state) != {
            "session_id",
            "sequence",
            "head",
            "profile_issued",
            "visual_optimizer_event_id",
            "state_signature",
        }:
            raise RuntimeAttestationError("runtime session state fields differ")
        body = dict(state)
        encoded_signature = body.pop("state_signature")
        try:
            signature = base64.b64decode(encoded_signature, validate=True)
            self.__key.public_key().verify(
                signature, RUNTIME_STATE_DOMAIN + _canonical_bytes(body)
            )
        except (ValueError, TypeError, InvalidSignature) as exc:
            raise RuntimeAttestationError(
                "runtime session state signature is invalid"
            ) from exc
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
        visual_optimizer_event_id = state["visual_optimizer_event_id"]
        if visual_optimizer_event_id is not None and (
            not isinstance(visual_optimizer_event_id, str)
            or not visual_optimizer_event_id.strip()
        ):
            raise RuntimeAttestationError(
                "runtime visual optimizer event identity is invalid"
            )
        self.__sequence = sequence
        self.__head = head
        self.__profile_issued = state["profile_issued"]
        self.__active_boundary = None
        self.__visual_optimizer_event_id = visual_optimizer_event_id

    def begin_optimizer_boundary(self, optimizer, transaction):
        if self.__active_boundary is not None:
            raise RuntimeAttestationError("an optimizer boundary is already active")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise RuntimeAttestationError(
                "authenticated optimizer evidence requires a PyTorch optimizer"
            )
        has_pending = getattr(transaction, "has_pending_online_update", None)
        commit = getattr(transaction, "commit_online_update", None)
        if not callable(has_pending) or not callable(commit) or not has_pending():
            raise RuntimeAttestationError(
                "authenticated optimizer evidence requires a pending online transaction"
            )
        proof = object.__new__(_OptimizerBoundaryProof)
        proof.session_id = self.__session_id
        proof.nonce = secrets.token_hex(32)
        self.__active_boundary = {
            "boundary_nonce": proof.nonce,
            "commit_receipt": None,
            "proof": proof,
            "optimizer": optimizer,
            "runtime_head": self.__head,
            "runtime_sequence": self.__sequence,
            "step_receipt": None,
            "transaction": transaction,
        }
        self.__visual_optimizer_event_id = None
        return proof

    def _boundary(self, proof):
        active = self.__active_boundary
        if (
            active is None
            or not isinstance(active, dict)
            or set(active) != _BOUNDARY_STATE_FIELDS
            or not isinstance(proof, _OptimizerBoundaryProof)
            or active["proof"] is not proof
            or proof.session_id != self.__session_id
            or proof.nonce != active["boundary_nonce"]
            or active["runtime_sequence"] != self.__sequence
            or active["runtime_head"] != self.__head
        ):
            raise RuntimeAttestationError(
                "optimizer boundary proof is not active for this session"
            )
        return active

    def _expected_step_receipt_body(self, active):
        return {
            "schema_version": RUNTIME_STEP_RECEIPT_SCHEMA,
            "session_id": self.__session_id,
            "boundary_nonce": active["boundary_nonce"],
            "runtime_sequence": active["runtime_sequence"],
            "runtime_head": active["runtime_head"],
            "receipt_order": 1,
            "optimizer_identity": _object_identity(active["optimizer"]),
            "transaction_identity": _object_identity(active["transaction"]),
            "previous_receipt_sha256": RUNTIME_RECEIPT_GENESIS_HASH,
        }

    def _validated_step_receipt(self, active):
        receipt = active["step_receipt"]
        if not isinstance(receipt, _OptimizerStepReceipt):
            raise RuntimeAttestationError(
                "optimizer event lacks a completed optimizer step receipt"
            )
        body = _verified_receipt(
            receipt,
            _OptimizerStepReceipt,
            RUNTIME_STEP_RECEIPT_DOMAIN,
            self.__key.public_key(),
            "optimizer step receipt",
        )
        if body != self._expected_step_receipt_body(active):
            raise RuntimeAttestationError(
                "optimizer step receipt does not match the active boundary"
            )
        return receipt

    def _expected_commit_receipt_body(self, active, step_receipt):
        return {
            "schema_version": RUNTIME_COMMIT_RECEIPT_SCHEMA,
            "session_id": self.__session_id,
            "boundary_nonce": active["boundary_nonce"],
            "runtime_sequence": active["runtime_sequence"],
            "runtime_head": active["runtime_head"],
            "receipt_order": 2,
            "optimizer_identity": _object_identity(active["optimizer"]),
            "transaction_identity": _object_identity(active["transaction"]),
            "previous_receipt_sha256": _receipt_sha256(
                RUNTIME_STEP_RECEIPT_DOMAIN, step_receipt
            ),
        }

    def _validated_commit_receipt(self, active):
        step_receipt = self._validated_step_receipt(active)
        receipt = active["commit_receipt"]
        if not isinstance(receipt, _TransactionCommitReceipt):
            raise RuntimeAttestationError(
                "optimizer event lacks a committed online transaction receipt"
            )
        body = _verified_receipt(
            receipt,
            _TransactionCommitReceipt,
            RUNTIME_COMMIT_RECEIPT_DOMAIN,
            self.__key.public_key(),
            "transaction commit receipt",
        )
        if body != self._expected_commit_receipt_body(active, step_receipt):
            raise RuntimeAttestationError(
                "transaction commit receipt does not match the optimizer step"
            )
        return receipt

    def execute_optimizer_step(self, proof, optimizer, *, scaler=None):
        active = self._boundary(proof)
        if active["step_receipt"] is not None:
            raise RuntimeAttestationError("optimizer step was already confirmed")
        if active["commit_receipt"] is not None:
            raise RuntimeAttestationError("transaction receipt precedes optimizer step")
        if active["optimizer"] is not optimizer:
            raise RuntimeAttestationError(
                "optimizer boundary proof is bound to a different optimizer"
            )
        register_hook = getattr(optimizer, "register_step_post_hook", None)
        if not callable(register_hook):
            raise RuntimeAttestationError(
                "authenticated optimizer evidence requires an optimizer step hook"
            )
        optimizer_step_completed = False

        def confirm_step(observed_optimizer, *args, **kwargs):
            del args, kwargs
            nonlocal optimizer_step_completed
            if observed_optimizer is not optimizer:
                raise RuntimeAttestationError(
                    "optimizer step hook observed a different optimizer"
                )
            optimizer_step_completed = True

        hook = register_hook(confirm_step)
        try:
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
        finally:
            hook.remove()
        if not optimizer_step_completed:
            raise RuntimeAttestationError(
                "optimizer.step was not executed at the authenticated boundary"
            )
        receipt_payload = _canonical_bytes(self._expected_step_receipt_body(active))
        active["step_receipt"] = _OptimizerStepReceipt(
            payload=receipt_payload,
            signature=self.__key.sign(
                RUNTIME_STEP_RECEIPT_DOMAIN + receipt_payload
            ),
        )

    def commit_online_transaction(self, proof, transaction):
        active = self._boundary(proof)
        step_receipt = self._validated_step_receipt(active)
        if active["commit_receipt"] is not None:
            raise RuntimeAttestationError("transaction commit was already confirmed")
        if active["transaction"] is not transaction:
            raise RuntimeAttestationError(
                "optimizer boundary proof is bound to a different transaction"
            )
        has_pending = getattr(transaction, "has_pending_online_update", None)
        commit = getattr(transaction, "commit_online_update", None)
        if not callable(has_pending) or not callable(commit):
            raise RuntimeAttestationError(
                "authenticated optimizer evidence requires an online transaction"
            )
        if not has_pending():
            raise RuntimeAttestationError(
                "authenticated optimizer boundary has no pending online transaction"
            )
        commit()
        if has_pending():
            raise RuntimeAttestationError(
                "online transaction remained pending after commit"
            )
        receipt_payload = _canonical_bytes(
            self._expected_commit_receipt_body(active, step_receipt)
        )
        active["commit_receipt"] = _TransactionCommitReceipt(
            payload=receipt_payload,
            signature=self.__key.sign(
                RUNTIME_COMMIT_RECEIPT_DOMAIN + receipt_payload
            ),
        )

    def abort_optimizer_boundary(self, proof):
        if self.__active_boundary is None:
            return
        self._boundary(proof)
        self.__active_boundary = None

    def _sign_runtime_event(self, event_kind, payload):
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

    def sign_committed_optimizer_event(self, payload, proof):
        active = self._boundary(proof)
        self._validated_step_receipt(active)
        self._validated_commit_receipt(active)
        if not isinstance(payload, Mapping):
            raise RuntimeAttestationError("optimizer event payload is invalid")
        event_id = payload.get("event_id")
        if not isinstance(event_id, str) or not event_id.strip():
            raise RuntimeAttestationError("optimizer event identity is invalid")
        envelope = self._sign_runtime_event("optimizer-event", payload)
        self.__active_boundary = None
        self.__visual_optimizer_event_id = event_id
        return envelope

    def sign_visual_parameter_event(self, payload):
        if self.__active_boundary is not None:
            raise RuntimeAttestationError(
                "visual evidence cannot be issued during an active optimizer boundary"
            )
        if not isinstance(payload, Mapping):
            raise RuntimeAttestationError("visual event payload is invalid")
        if payload.get("optimizer_event_id") != self.__visual_optimizer_event_id:
            raise RuntimeAttestationError(
                "visual event is not bound to the committed optimizer event"
            )
        return self._sign_runtime_event("visual-parameter-event", payload)

    def sign_profile(self, payload):
        if self.__profile_issued:
            raise RuntimeAttestationError("runtime session already issued a profile")
        if self.__active_boundary is not None:
            raise RuntimeAttestationError(
                "runtime profile cannot be issued during an optimizer boundary"
            )
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
