from copy import deepcopy
from dataclasses import asdict

import torch

from .causal_audit import audit_recorded_trace_equivalence
from .device import get_model_device, move_data_to_device


def _clone_data(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_data(item) for item in value)
    return deepcopy(value)


def _perturb_future_inputs(inputs, perturbation_scale):
    if inputs.is_floating_point() and inputs.numel() > 0:
        finite = bool(torch.isfinite(inputs).all().item())
        if finite and inputs.min().item() >= 0.0 and inputs.max().item() <= 1.0:
            # Raw-frame processors expect normalized pixels. A half-range cyclic
            # shift changes every finite value while keeping the valid domain.
            return torch.remainder(inputs + 0.5, 1.0), "domain_preserving"
    if inputs.dtype == torch.bool:
        return torch.logical_not(inputs), "domain_preserving"
    if inputs.dtype == torch.uint8:
        return torch.iinfo(inputs.dtype).max - inputs, "domain_preserving"
    return inputs + float(perturbation_scale), "additive"


def _run_model_trace(model, batches, device, forward_kwargs):
    target_model = getattr(model, "module", model)
    if hasattr(target_model, "reset_online_states"):
        target_model.reset_online_states()
    model.eval()
    trace = []
    with torch.no_grad():
        for raw_batch in batches:
            batch = move_data_to_device(raw_batch, device)
            model(**batch, return_loss=False, **forward_kwargs)
            row = getattr(target_model, "last_step_audit", None)
            if row is None:
                raise RuntimeError("model did not expose last_step_audit during causal replay")
            trace.append(deepcopy(row))
    return trace


def audit_model_future_perturbation(
    model,
    batches,
    cut_packet_index,
    perturbation_scale=17.0,
    device=None,
    forward_kwargs=None,
):
    """Replay a model after changing only packet inputs beyond a prefix cut."""

    batches = [_clone_data(batch) for batch in batches]
    cut_packet_index = int(cut_packet_index)
    if cut_packet_index < 0 or cut_packet_index >= len(batches) - 1:
        raise ValueError("cut_packet_index must leave at least one future packet to perturb")
    device = get_model_device(model) if device is None else torch.device(device)
    forward_kwargs = dict(forward_kwargs or {})

    perturbed_batches = [_clone_data(batch) for batch in batches]
    perturbation_kinds = set()
    for index in range(cut_packet_index + 1, len(perturbed_batches)):
        inputs = perturbed_batches[index].get("inputs")
        if not torch.is_tensor(inputs):
            raise TypeError("causal replay requires tensor inputs in every future packet")
        perturbed, kind = _perturb_future_inputs(inputs, perturbation_scale)
        perturbed_batches[index]["inputs"] = perturbed
        perturbation_kinds.add(kind)

    reference_trace = _run_model_trace(model, batches, device, forward_kwargs)
    perturbed_trace = _run_model_trace(model, perturbed_batches, device, forward_kwargs)
    suffix_changed = any(
        reference_trace[index] != perturbed_trace[index]
        for index in range(cut_packet_index + 1, len(reference_trace))
    )
    if not suffix_changed:
        raise RuntimeError("future perturbation did not change any post-cut model trace")
    through_time = float(reference_trace[cut_packet_index]["time"])
    audit = audit_recorded_trace_equivalence(
        reference_trace,
        perturbed_trace,
        through_time=through_time,
        name="model_future_perturbation",
    )
    return {
        "passed": audit.passed,
        "name": audit.name,
        "comparisons": audit.comparisons,
        "details": asdict(audit)["details"],
        "cut_packet_index": cut_packet_index,
        "through_time": through_time,
        "perturbation_scale": float(perturbation_scale),
        "perturbation_kind": (
            next(iter(perturbation_kinds)) if len(perturbation_kinds) == 1 else "mixed"
        ),
        "suffix_changed": suffix_changed,
        "reference_trace": reference_trace,
        "perturbed_trace": perturbed_trace,
    }
