"""Artifact-backed model and optimizer evidence for Prefix Route V2."""

from __future__ import annotations

import hashlib
import io
import json
import re

from .evidence_bundle import (
    EvidenceBundleError,
    read_verified_bundle_bytes,
    read_verified_bundle_json,
)


TENSOR_STATE_SCHEMA = "prefix-route-tensor-state-v2"
OPTIMIZER_TRACE_SCHEMA = "prefix-route-optimizer-artifact-trace-v2"
IDENTITY_ARTIFACT_SCHEMAS = {
    "model_config": "prefix-route-model-config-v2",
    "resolved_command": "prefix-route-resolved-command-v2",
    "environment_lock": "prefix-route-environment-lock-v2",
}
_TENSOR_NAME = re.compile(r"^[A-Za-z0-9_.:/-]+$")
_GENESIS_EVENT_SHA256 = "0" * 64


class PrefixRouteArtifactError(ValueError):
    pass


def canonical_json_bytes(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_text(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PrefixRouteArtifactError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PrefixRouteArtifactError(f"{label} must be a positive integer")
    return value


def _read_bytes(reference, bundle_root, label):
    try:
        _, payload = read_verified_bundle_bytes(reference, bundle_root, label)
    except EvidenceBundleError as exc:
        raise PrefixRouteArtifactError(str(exc)) from exc
    if not payload:
        raise PrefixRouteArtifactError(f"{label} is empty")
    return payload


def tensor_state_digest_from_named_arrays(named_arrays):
    """Derive the canonical tensor-state digest from a name-to-array mapping."""

    try:
        import numpy as np
    except Exception as exc:
        raise PrefixRouteArtifactError(
            "NumPy is required for tensor-state evidence"
        ) from exc
    if not isinstance(named_arrays, dict) or not named_arrays:
        raise PrefixRouteArtifactError(
            "tensor-state mapping must contain at least one tensor"
        )
    names = sorted(named_arrays)
    if len(names) != len(set(names)):
        raise PrefixRouteArtifactError("tensor-state names are duplicated")
    digest = hashlib.sha256()
    tensor_manifest = []
    for name in names:
        if not isinstance(name, str) or not _TENSOR_NAME.fullmatch(name):
            raise PrefixRouteArtifactError("tensor-state name is not canonical")
        try:
            array = np.asarray(named_arrays[name])
        except Exception as exc:
            raise PrefixRouteArtifactError(
                f"tensor-state array cannot be read: {name}"
            ) from exc
        if array.dtype.hasobject or array.dtype.kind in {"O", "S", "U", "V"}:
            raise PrefixRouteArtifactError(
                f"tensor-state array has unsupported dtype: {name}"
            )
        contiguous = np.ascontiguousarray(array)
        metadata = {
            "name": name,
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "nbytes": int(contiguous.nbytes),
        }
        tensor_manifest.append(metadata)
        digest.update(b"TENSOR\0")
        digest.update(canonical_json_bytes(metadata))
        digest.update(contiguous.tobytes(order="C"))
    return {
        "schema_version": TENSOR_STATE_SCHEMA,
        "tensor_count": len(names),
        "tensor_manifest": tensor_manifest,
        "tensor_state_sha256": digest.hexdigest(),
    }


def tensor_state_digest_from_npz_bytes(payload, *, allow_empty=False):
    """Derive a semantic digest from numeric NPZ tensors without pickle."""

    if not isinstance(payload, bytes) or not payload:
        raise PrefixRouteArtifactError("tensor-state artifact must be non-empty bytes")
    try:
        import numpy as np
    except Exception as exc:
        raise PrefixRouteArtifactError("NumPy is required for tensor-state evidence") from exc
    try:
        archive = np.load(io.BytesIO(payload), allow_pickle=False)
    except Exception as exc:
        raise PrefixRouteArtifactError(
            "tensor-state artifact is not a safe NPZ archive"
        ) from exc
    try:
        names = sorted(archive.files)
        if len(names) != len(set(names)):
            raise PrefixRouteArtifactError("tensor-state names are duplicated")
        if not names and not allow_empty:
            raise PrefixRouteArtifactError("tensor-state artifact contains no tensors")
        named_arrays = {}
        for name in names:
            try:
                named_arrays[name] = archive[name]
            except Exception as exc:
                raise PrefixRouteArtifactError(
                    f"tensor-state array cannot be read: {name}"
                ) from exc
        if names:
            summary = tensor_state_digest_from_named_arrays(named_arrays)
        else:
            summary = {
                "schema_version": TENSOR_STATE_SCHEMA,
                "tensor_count": 0,
                "tensor_manifest": [],
                "tensor_state_sha256": hashlib.sha256(b"").hexdigest(),
            }
        return {
            **summary,
            "artifact_sha256": hashlib.sha256(payload).hexdigest(),
        }
    finally:
        archive.close()


def read_tensor_state_artifact(
    reference,
    *,
    bundle_root,
    label,
    allow_empty=False,
):
    payload = _read_bytes(reference, bundle_root, label)
    return tensor_state_digest_from_npz_bytes(payload, allow_empty=allow_empty)


def read_identity_artifact(reference, *, bundle_root, kind, arm, seed):
    if kind not in IDENTITY_ARTIFACT_SCHEMAS:
        raise PrefixRouteArtifactError("identity artifact kind is unregistered")
    try:
        _, payload, value = read_verified_bundle_json(
            reference,
            bundle_root,
            f"{arm}/{seed} {kind}",
            require_object=True,
        )
    except EvidenceBundleError as exc:
        raise PrefixRouteArtifactError(str(exc)) from exc
    if payload != canonical_json_bytes(value):
        raise PrefixRouteArtifactError(f"{kind} is not canonical JSON")
    required = {"schema_version", "arm", "seed", "payload"}
    if not isinstance(value, dict) or set(value) != required:
        raise PrefixRouteArtifactError(f"{kind} fields differ")
    if (
        value["schema_version"] != IDENTITY_ARTIFACT_SCHEMAS[kind]
        or value["arm"] != arm
        or value["seed"] != seed
        or not isinstance(value["payload"], dict)
        or not value["payload"]
    ):
        raise PrefixRouteArtifactError(f"{kind} identity differs")
    return {
        "artifact_sha256": hashlib.sha256(payload).hexdigest(),
        "value": value,
    }


def _event_sha256(event):
    unsigned = dict(event)
    supplied = unsigned.pop("event_sha256", None)
    _sha256_text(supplied, "optimizer event commitment")
    derived = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if supplied != derived:
        raise PrefixRouteArtifactError("optimizer event commitment differs")
    return derived


def _read_auxiliary_artifact(reference, bundle_root, label):
    payload = _read_bytes(reference, bundle_root, label)
    return hashlib.sha256(payload).hexdigest()


def verify_optimizer_artifact_trace(
    trace_reference,
    *,
    bundle_root,
    arm,
    seed,
    initial_model_reference,
    final_model_reference,
    initial_optimizer_reference,
    final_optimizer_reference,
    learning,
):
    """Verify optimizer events against the bytes of every referenced state."""

    try:
        _, trace_bytes, trace = read_verified_bundle_json(
            trace_reference,
            bundle_root,
            f"{arm}/{seed} optimizer artifact trace",
            require_object=True,
        )
    except EvidenceBundleError as exc:
        raise PrefixRouteArtifactError(str(exc)) from exc
    if trace_bytes != canonical_json_bytes(trace):
        raise PrefixRouteArtifactError("optimizer artifact trace is not canonical JSON")
    required = {
        "schema_version",
        "arm",
        "seed",
        "learning",
        "events",
        "final_event_sha256",
    }
    if not isinstance(trace, dict) or set(trace) != required:
        raise PrefixRouteArtifactError("optimizer artifact trace fields differ")
    if (
        trace["schema_version"] != OPTIMIZER_TRACE_SCHEMA
        or trace["arm"] != arm
        or trace["seed"] != seed
        or trace["learning"] is not learning
        or not isinstance(trace["events"], list)
    ):
        raise PrefixRouteArtifactError("optimizer artifact trace identity differs")

    initial_model = read_tensor_state_artifact(
        initial_model_reference,
        bundle_root=bundle_root,
        label=f"{arm}/{seed} initial model",
    )
    final_model = read_tensor_state_artifact(
        final_model_reference,
        bundle_root=bundle_root,
        label=f"{arm}/{seed} final model",
    )
    initial_optimizer = read_tensor_state_artifact(
        initial_optimizer_reference,
        bundle_root=bundle_root,
        label=f"{arm}/{seed} initial optimizer",
        allow_empty=True,
    )
    final_optimizer = read_tensor_state_artifact(
        final_optimizer_reference,
        bundle_root=bundle_root,
        label=f"{arm}/{seed} final optimizer",
        allow_empty=True,
    )

    if not learning:
        if trace["events"]:
            raise PrefixRouteArtifactError("no-learning control contains optimizer events")
        if trace["final_event_sha256"] != _GENESIS_EVENT_SHA256:
            raise PrefixRouteArtifactError("no-learning control event tail differs")
        if (
            initial_model["tensor_state_sha256"]
            != final_model["tensor_state_sha256"]
            or initial_optimizer["tensor_state_sha256"]
            != final_optimizer["tensor_state_sha256"]
        ):
            raise PrefixRouteArtifactError(
                "no-learning control changed model or optimizer state"
            )
        return {
            "trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
            "learning": False,
            "optimizer_event_count": 0,
            "effective_token_count": 0,
            "gradient_accumulation_steps": 0,
            "initial_model_state_sha256": initial_model["tensor_state_sha256"],
            "final_model_state_sha256": final_model["tensor_state_sha256"],
            "initial_model_artifact_sha256": initial_model["artifact_sha256"],
            "final_model_artifact_sha256": final_model["artifact_sha256"],
            "initial_optimizer_state_sha256": initial_optimizer[
                "tensor_state_sha256"
            ],
            "final_optimizer_state_sha256": final_optimizer[
                "tensor_state_sha256"
            ],
            "final_event_sha256": _GENESIS_EVENT_SHA256,
        }

    if not trace["events"]:
        raise PrefixRouteArtifactError("learning run contains no optimizer events")
    event_fields = {
        "optimizer_event_index",
        "previous_event_sha256",
        "gradient_accumulation_steps",
        "effective_token_count",
        "status",
        "input_batches",
        "model_state_before",
        "model_state_after",
        "optimizer_state_before",
        "optimizer_state_after",
        "rng_state_before",
        "rng_state_after",
        "amp_scaler_state_before",
        "amp_scaler_state_after",
        "event_sha256",
    }
    previous_event = _GENESIS_EVENT_SHA256
    previous_model = initial_model["tensor_state_sha256"]
    previous_optimizer = initial_optimizer["tensor_state_sha256"]
    effective_tokens = 0
    accumulation_steps = set()
    for index, event in enumerate(trace["events"]):
        if not isinstance(event, dict) or set(event) != event_fields:
            raise PrefixRouteArtifactError("optimizer artifact event fields differ")
        if (
            event["optimizer_event_index"] != index
            or event["previous_event_sha256"] != previous_event
            or event["status"] != "APPLIED_FINITE"
        ):
            raise PrefixRouteArtifactError("optimizer artifact event chain differs")
        before_model = read_tensor_state_artifact(
            event["model_state_before"],
            bundle_root=bundle_root,
            label=f"{arm}/{seed} event {index} model before",
        )
        after_model = read_tensor_state_artifact(
            event["model_state_after"],
            bundle_root=bundle_root,
            label=f"{arm}/{seed} event {index} model after",
        )
        before_optimizer = read_tensor_state_artifact(
            event["optimizer_state_before"],
            bundle_root=bundle_root,
            label=f"{arm}/{seed} event {index} optimizer before",
            allow_empty=True,
        )
        after_optimizer = read_tensor_state_artifact(
            event["optimizer_state_after"],
            bundle_root=bundle_root,
            label=f"{arm}/{seed} event {index} optimizer after",
            allow_empty=True,
        )
        if (
            before_model["tensor_state_sha256"] != previous_model
            or before_optimizer["tensor_state_sha256"] != previous_optimizer
        ):
            raise PrefixRouteArtifactError("optimizer artifact state chain differs")
        if (
            before_model["tensor_state_sha256"]
            == after_model["tensor_state_sha256"]
        ):
            raise PrefixRouteArtifactError("optimizer event did not change model tensors")
        for field in (
            "input_batches",
            "rng_state_before",
            "rng_state_after",
            "amp_scaler_state_before",
            "amp_scaler_state_after",
        ):
            _read_auxiliary_artifact(
                event[field],
                bundle_root,
                f"{arm}/{seed} event {index} {field}",
            )
        accumulation_steps.add(
            _positive_int(
                event["gradient_accumulation_steps"],
                "gradient_accumulation_steps",
            )
        )
        effective_tokens += _positive_int(
            event["effective_token_count"],
            "effective_token_count",
        )
        previous_event = _event_sha256(event)
        previous_model = after_model["tensor_state_sha256"]
        previous_optimizer = after_optimizer["tensor_state_sha256"]
    if len(accumulation_steps) != 1:
        raise PrefixRouteArtifactError(
            "gradient accumulation changes across optimizer events"
        )
    if (
        trace["final_event_sha256"] != previous_event
        or final_model["tensor_state_sha256"] != previous_model
        or final_optimizer["tensor_state_sha256"] != previous_optimizer
    ):
        raise PrefixRouteArtifactError("optimizer artifact trace tail differs")
    return {
        "trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
        "learning": True,
        "optimizer_event_count": len(trace["events"]),
        "effective_token_count": effective_tokens,
        "gradient_accumulation_steps": next(iter(accumulation_steps)),
        "initial_model_state_sha256": initial_model["tensor_state_sha256"],
        "final_model_state_sha256": final_model["tensor_state_sha256"],
        "initial_model_artifact_sha256": initial_model["artifact_sha256"],
        "final_model_artifact_sha256": final_model["artifact_sha256"],
        "initial_optimizer_state_sha256": initial_optimizer[
            "tensor_state_sha256"
        ],
        "final_optimizer_state_sha256": final_optimizer[
            "tensor_state_sha256"
        ],
        "final_event_sha256": previous_event,
    }


__all__ = [
    "IDENTITY_ARTIFACT_SCHEMAS",
    "OPTIMIZER_TRACE_SCHEMA",
    "TENSOR_STATE_SCHEMA",
    "PrefixRouteArtifactError",
    "canonical_json_bytes",
    "read_identity_artifact",
    "read_tensor_state_artifact",
    "tensor_state_digest_from_named_arrays",
    "tensor_state_digest_from_npz_bytes",
    "verify_optimizer_artifact_trace",
]
