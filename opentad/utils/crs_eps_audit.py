"""Matched-stochasticity audits for CRS-EPS replay and Q2 binding twins."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import torch

from .evidence_bundle import publish_exclusive_file


AUDIT_SCHEMA_VERSION = "full-petal-crs-eps-paired-audit-v1"


class CrsEpsAuditError(RuntimeError):
    pass


def _update_digest(digest, value):
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    elif isinstance(value, dict):
        for key in sorted(value):
            digest.update(str(key).encode("utf-8"))
            _update_digest(digest, value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            _update_digest(digest, item)
    else:
        digest.update(repr(value).encode("utf-8"))


def object_digest(value):
    digest = hashlib.sha256()
    _update_digest(digest, value)
    return digest.hexdigest()


def capture_rng_snapshot():
    return {
        "cpu": torch.random.get_rng_state().clone(),
        "cuda": (
            tuple(state.clone() for state in torch.cuda.get_rng_state_all())
            if torch.cuda.is_available() and torch.cuda.is_initialized()
            else ()
        ),
    }


def restore_rng_snapshot(snapshot):
    torch.random.set_rng_state(snapshot["cpu"])
    if snapshot["cuda"]:
        if not torch.cuda.is_available():
            raise CrsEpsAuditError("CUDA RNG snapshot cannot be restored without CUDA")
        torch.cuda.set_rng_state_all(list(snapshot["cuda"]))


def rng_snapshot_digest(snapshot):
    return object_digest(snapshot)


def _gradient_record(model):
    parts = []
    names = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        names.append(name)
        gradient = parameter.grad
        if gradient is None:
            gradient = torch.zeros_like(parameter)
        parts.append(gradient.detach().float().reshape(-1).cpu())
    vector = torch.cat(parts) if parts else torch.empty(0)
    return {
        "names": names,
        "vector": vector,
        "digest": object_digest(vector),
        "norm": float(torch.linalg.vector_norm(vector).item()),
    }


def _pair_vector_metrics(left, right):
    if left.numel() != right.numel():
        return {
            "comparable": False,
            "cosine": None,
            "norm_ratio": None,
            "sign_agreement": None,
        }
    left_norm = float(torch.linalg.vector_norm(left).item())
    right_norm = float(torch.linalg.vector_norm(right).item())
    if left_norm == 0.0 and right_norm == 0.0:
        cosine = 1.0
    elif left_norm == 0.0 or right_norm == 0.0:
        cosine = 0.0
    else:
        cosine = float(torch.dot(left, right).item() / (left_norm * right_norm))
    nonzero = (left != 0) | (right != 0)
    sign_agreement = (
        float((torch.sign(left[nonzero]) == torch.sign(right[nonzero])).float().mean().item())
        if nonzero.any()
        else 1.0
    )
    return {
        "comparable": True,
        "cosine": cosine,
        "norm_ratio": right_norm / left_norm if left_norm else (1.0 if right_norm == 0 else None),
        "sign_agreement": sign_agreement,
    }


def _runtime_record(runtime):
    continuous = torch.cat(
        [
            runtime.queries.detach().float().reshape(-1).cpu(),
            runtime.feature_memory.detach().float().reshape(-1).cpu(),
            runtime.start_state.detach().float().reshape(-1).cpu(),
            runtime.score_state.detach().float().reshape(-1).cpu(),
        ]
    )
    discrete = {
        "slot_status": runtime.slot_status.detach().cpu().tolist(),
        "refractory": runtime.refractory.detach().cpu().tolist(),
        "label_state": runtime.label_state.detach().cpu().tolist(),
        "source_frames": list(runtime.source_frames),
        "last_decision_frame": runtime.last_decision_frame,
    }
    return {
        "digest": object_digest(runtime.__dict__),
        "continuous": continuous,
        "continuous_digest": object_digest(continuous),
        "continuous_norm": float(torch.linalg.vector_norm(continuous).item()),
        "discrete": discrete,
        "discrete_digest": object_digest(discrete),
    }


def _logits_digest(logits):
    rows = []
    for output in logits:
        rows.append(
            {
                key: value.detach().cpu()
                for key, value in sorted(output.items())
                if torch.is_tensor(value)
            }
        )
    return object_digest(rows)


def _run_arm(model, episode_kwargs):
    model.zero_grad(set_to_none=True)
    output = model.train_crs_eps_episode(**episode_kwargs)
    output.losses["cost"].backward()
    gradient = _gradient_record(model)
    runtime = _runtime_record(output.runtime_state)
    record = {
        "trajectory_binding_mode": model.trajectory_binding_mode,
        "losses": {
            key: float(value.detach().item())
            for key, value in output.losses.items()
            if not key.startswith("_")
        },
        "audit": deepcopy(output.audit),
        "logits_digest": _logits_digest(output.logits),
        "runtime": {key: value for key, value in runtime.items() if key != "continuous"},
        "gradient": {
            key: value for key, value in gradient.items() if key != "vector"
        },
    }
    return record, gradient, runtime


def run_matched_crs_eps_pair(
    left_model,
    right_model,
    left_episode_kwargs,
    right_episode_kwargs=None,
    *,
    comparison_type,
):
    if comparison_type not in {"binding_twin", "replay_fidelity"}:
        raise CrsEpsAuditError("unsupported CRS-EPS paired comparison type")
    right_episode_kwargs = (
        left_episode_kwargs if right_episode_kwargs is None else right_episode_kwargs
    )
    left_state = left_model.state_dict()
    right_state = right_model.state_dict()
    if set(left_state) != set(right_state) or any(
        not torch.equal(left_state[name].detach().cpu(), right_state[name].detach().cpu())
        for name in left_state
    ):
        raise CrsEpsAuditError("paired models do not share an identical initial state")
    if left_model.training != right_model.training:
        raise CrsEpsAuditError("paired models must share train/eval mode")
    original_rng = capture_rng_snapshot()
    matched_rng = capture_rng_snapshot()
    initial_state_digest = object_digest(left_state)
    try:
        restore_rng_snapshot(matched_rng)
        left_record, left_gradient, left_runtime = _run_arm(
            left_model, left_episode_kwargs
        )
        restore_rng_snapshot(matched_rng)
        right_record, right_gradient, right_runtime = _run_arm(
            right_model, right_episode_kwargs
        )
    finally:
        restore_rng_snapshot(original_rng)
        left_model.zero_grad(set_to_none=True)
        right_model.zero_grad(set_to_none=True)
    trace = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "comparison_type": comparison_type,
        "model_initialization_sha256": initial_state_digest,
        "matched_rng_sha256": rng_snapshot_digest(matched_rng),
        "left": left_record,
        "right": right_record,
        "gradient_comparison": _pair_vector_metrics(
            left_gradient["vector"], right_gradient["vector"]
        ),
        "runtime_continuous_comparison": _pair_vector_metrics(
            left_runtime["continuous"], right_runtime["continuous"]
        ),
        "runtime_discrete_equal": (
            left_runtime["discrete"] == right_runtime["discrete"]
        ),
        "logits_equal": left_record["logits_digest"] == right_record["logits_digest"],
    }
    trace["trace_sha256"] = object_digest(trace)
    return trace


def publish_crs_eps_audit(path, trace):
    path = Path(path)
    encoded = (
        json.dumps(trace, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    publish_exclusive_file(path, encoded)
    return path


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "CrsEpsAuditError",
    "capture_rng_snapshot",
    "object_digest",
    "publish_crs_eps_audit",
    "restore_rng_snapshot",
    "rng_snapshot_digest",
    "run_matched_crs_eps_pair",
]
