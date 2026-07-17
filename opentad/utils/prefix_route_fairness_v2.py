"""Runtime-derived capacity and resource audit for Prefix-Route Protocol V2."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import statistics
from typing import Any, Callable, Mapping

from .evidence_bundle import read_verified_bundle_json


FAIRNESS_SCHEMA = "prefix-route-arm-runtime-audit-v2"
BUDGET_EVIDENCE_SCHEMA = "prefix-route-fairness-budget-evidence-v2"
EXECUTION_TRACE_SCHEMA = "prefix-route-fairness-execution-trace-v2"
CALIBRATION_MANIFEST_SCHEMA = "prefix-route-fairness-calibration-manifest-v2"
TRIAL_MANIFEST_SCHEMA = "prefix-route-fairness-trial-manifest-v2"
SEED_MANIFEST_SCHEMA = "prefix-route-fairness-seed-manifest-v2"
CORE_ARMS = ("B0", "B1", "B2", "B3", "B4")
FROZEN_RUN_SEEDS = (705, 706, 707)
PARAMETER_TOLERANCE = 0.05
RESOURCE_TOLERANCE = 0.10
LATENCY_UPPER_RATIO = 1.10
WARMUP_DECISIONS = 20
TIMED_DECISIONS = 100
EXACT_BUDGET_FIELDS = (
    "optimizer_event_count",
    "effective_token_count",
    "gradient_accumulation_steps",
    "hyperparameter_trial_count",
    "calibration_video_count",
    "seed_count",
)


class PrefixRouteFairnessError(ValueError):
    pass


@dataclass(frozen=True)
class ArmRuntimeAdapter:
    """Live arm hooks; serialized scalar/boolean fairness evidence is forbidden."""

    arm: str
    model: Any
    optimizer: Any
    optimizer_event_forwards: Callable[[], Any]
    inference_forward: Callable[[], tuple[Any, Any]]
    reset_runtime_state: Callable[[], None]
    budget_evidence_root: Any
    execution_trace_reference: Mapping[str, str]
    calibration_manifest_reference: Mapping[str, str]
    trial_manifest_reference: Mapping[str, str]
    seed_manifest_reference: Mapping[str, str]


def _finite_nonnegative(value, label):
    if isinstance(value, bool):
        raise PrefixRouteFairnessError(f"{label} must be finite and non-negative")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise PrefixRouteFairnessError(
            f"{label} must be finite and non-negative"
        ) from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise PrefixRouteFairnessError(f"{label} must be finite and non-negative")
    return parsed


def _positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PrefixRouteFairnessError(f"{label} must be a positive integer")
    return value


def _relative_difference(value, anchor):
    if anchor == 0:
        return 0.0 if value == 0 else math.inf
    return abs(value - anchor) / anchor


def _canonical_json_bytes(value):
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


def _exact(value, fields, label):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise PrefixRouteFairnessError(f"{label} fields differ")
    return value


def _read_budget_record(reference, root, label):
    try:
        _, payload, value = read_verified_bundle_json(
            reference,
            root,
            label,
            require_object=True,
        )
    except Exception as exc:
        raise PrefixRouteFairnessError(str(exc)) from exc
    if payload != _canonical_json_bytes(value):
        raise PrefixRouteFairnessError(f"{label} is not canonical JSON")
    return value, hashlib.sha256(payload).hexdigest()


def _tensor_bytes(value, torch):
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item, torch) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item, torch) for item in value)
    if value is None:
        return 0
    raise PrefixRouteFairnessError(
        "causal state may contain only tensors, mappings, arrays, or null"
    )


def _profiler_macs(profiler):
    total = 0
    for event in profiler.key_averages():
        flops = getattr(event, "flops", 0) or 0
        total += int(flops)
    if total <= 0:
        raise PrefixRouteFairnessError(
            "Torch profiler reported no FLOPs for the measured path"
        )
    return float(total) / 2.0


def _runtime_budget(adapter):
    trace, trace_sha256 = _read_budget_record(
        adapter.execution_trace_reference,
        adapter.budget_evidence_root,
        f"{adapter.arm} execution trace",
    )
    _exact(trace, {"schema_version", "arm", "events"}, "execution trace")
    if (
        trace["schema_version"] != EXECUTION_TRACE_SCHEMA
        or trace["arm"] != adapter.arm
        or not isinstance(trace["events"], list)
        or not trace["events"]
    ):
        raise PrefixRouteFairnessError("execution trace identity differs")
    effective_tokens = 0
    accumulation_steps = set()
    for index, event in enumerate(trace["events"]):
        _exact(
            event,
            {
                "optimizer_event_index",
                "gradient_accumulation_steps",
                "effective_token_count",
                "status",
            },
            "optimizer event",
        )
        if event["optimizer_event_index"] != index:
            raise PrefixRouteFairnessError(
                "optimizer event indexes are not contiguous"
            )
        if event["status"] != "APPLIED_FINITE":
            raise PrefixRouteFairnessError(
                "execution trace contains a skipped or non-finite update"
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
    if len(accumulation_steps) != 1:
        raise PrefixRouteFairnessError(
            "gradient accumulation changes across optimizer events"
        )

    calibration, calibration_sha256 = _read_budget_record(
        adapter.calibration_manifest_reference,
        adapter.budget_evidence_root,
        f"{adapter.arm} calibration manifest",
    )
    _exact(
        calibration,
        {"schema_version", "video_ids"},
        "calibration manifest",
    )
    video_ids = calibration["video_ids"]
    if (
        calibration["schema_version"] != CALIBRATION_MANIFEST_SCHEMA
        or not isinstance(video_ids, list)
        or len(video_ids) != 40
        or video_ids != sorted(video_ids)
        or len(video_ids) != len(set(video_ids))
        or any(not isinstance(video_id, str) or not video_id for video_id in video_ids)
    ):
        raise PrefixRouteFairnessError("calibration manifest identity differs")

    trials, trials_sha256 = _read_budget_record(
        adapter.trial_manifest_reference,
        adapter.budget_evidence_root,
        f"{adapter.arm} trial manifest",
    )
    _exact(trials, {"schema_version", "trial_ids"}, "trial manifest")
    trial_ids = trials["trial_ids"]
    if (
        trials["schema_version"] != TRIAL_MANIFEST_SCHEMA
        or not isinstance(trial_ids, list)
        or not trial_ids
        or trial_ids != sorted(trial_ids)
        or len(trial_ids) != len(set(trial_ids))
        or any(not isinstance(trial_id, str) or not trial_id for trial_id in trial_ids)
    ):
        raise PrefixRouteFairnessError("trial manifest identity differs")

    seeds, seeds_sha256 = _read_budget_record(
        adapter.seed_manifest_reference,
        adapter.budget_evidence_root,
        f"{adapter.arm} seed manifest",
    )
    _exact(seeds, {"schema_version", "seeds"}, "seed manifest")
    seed_values = seeds["seeds"]
    if (
        seeds["schema_version"] != SEED_MANIFEST_SCHEMA
        or not isinstance(seed_values, list)
        or tuple(seed_values) != FROZEN_RUN_SEEDS
    ):
        raise PrefixRouteFairnessError("seed manifest identity differs")

    return (
        {
            "optimizer_event_count": len(trace["events"]),
            "effective_token_count": effective_tokens,
            "gradient_accumulation_steps": next(iter(accumulation_steps)),
            "hyperparameter_trial_count": len(trial_ids),
            "calibration_video_count": len(video_ids),
            "seed_count": len(seed_values),
        },
        {
            "execution_trace": trace_sha256,
            "calibration_manifest": calibration_sha256,
            "trial_manifest": trials_sha256,
            "seed_manifest": seeds_sha256,
        },
        dict(trace["events"][0]),
    )


def derive_runtime_budget_evidence(adapter):
    """Derive exact run budgets from canonical hash-verified records."""

    if type(adapter) is not ArmRuntimeAdapter:
        raise PrefixRouteFairnessError(
            "budget evidence requires one live ArmRuntimeAdapter"
        )
    budget, source_sha256, _ = _runtime_budget(adapter)
    return {
        "schema_version": BUDGET_EVIDENCE_SCHEMA,
        "arm": adapter.arm,
        **budget,
        "source_sha256": source_sha256,
    }


def _measure_runtime(adapter, torch):
    if type(adapter) is not ArmRuntimeAdapter:
        raise PrefixRouteFairnessError(
            "fairness requires live ArmRuntimeAdapter objects, not serialized rows"
        )
    if adapter.arm not in CORE_ARMS:
        raise PrefixRouteFairnessError("runtime arm is unregistered")
    if not torch.cuda.is_available():
        raise PrefixRouteFairnessError(
            "production fairness measurement requires target CUDA runtime"
        )
    model = adapter.model
    optimizer = adapter.optimizer
    if not hasattr(model, "named_parameters") or not hasattr(
        optimizer,
        "zero_grad",
    ):
        raise PrefixRouteFairnessError("runtime model or optimizer interface differs")
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise PrefixRouteFairnessError("arm has no trainable parameters")
    parameter_count = sum(parameter.numel() for _, parameter in trainable)
    optimizer_parameters = [
        parameter
        for group in getattr(optimizer, "param_groups", ())
        for parameter in group.get("params", ())
    ]
    optimizer_parameter_ids = [id(parameter) for parameter in optimizer_parameters]
    if (
        len(optimizer_parameter_ids) != len(set(optimizer_parameter_ids))
        or set(optimizer_parameter_ids)
        != {id(parameter) for _, parameter in trainable}
    ):
        raise PrefixRouteFairnessError(
            "optimizer parameters differ from the model trainable parameters"
        )
    budget, budget_source_sha256, profile_event = _runtime_budget(adapter)
    budget_evidence = {
        "schema_version": BUDGET_EVIDENCE_SCHEMA,
        "arm": adapter.arm,
        **budget,
        "source_sha256": budget_source_sha256,
    }
    activities = [torch.profiler.ProfilerActivity.CPU]
    activities.append(torch.profiler.ProfilerActivity.CUDA)

    adapter.reset_runtime_state()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    baseline_training_memory = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.profiler.profile(activities=activities, with_flops=True) as profile:
        forwards = adapter.optimizer_event_forwards()
        if not hasattr(forwards, "__iter__"):
            raise PrefixRouteFairnessError(
                "optimizer_event_forwards must return an iterable"
            )
        smoke_tokens = 0
        microbatch_count = 0
        for item in forwards:
            if not isinstance(item, tuple) or len(item) != 2:
                raise PrefixRouteFairnessError(
                    "each optimizer-event forward must return loss and token count"
                )
            loss, effective_tokens = item
            if not torch.is_tensor(loss) or loss.numel() != 1:
                raise PrefixRouteFairnessError(
                    "optimizer-event loss must be one scalar tensor"
                )
            if not torch.isfinite(loss).all():
                raise PrefixRouteFairnessError("training loss is non-finite")
            smoke_tokens += _positive_int(
                effective_tokens,
                "smoke effective tokens",
            )
            microbatch_count += 1
            loss.backward()
        if (
            microbatch_count
            != profile_event["gradient_accumulation_steps"]
            or smoke_tokens != profile_event["effective_token_count"]
        ):
            raise PrefixRouteFairnessError(
                "profiled optimizer event differs from its execution trace"
            )
    gradient_inventory = []
    for name, parameter in trainable:
        gradient = parameter.grad
        if gradient is None:
            raise PrefixRouteFairnessError(
                f"trainable parameter lacks a smoke gradient: {adapter.arm}.{name}"
            )
        if not torch.isfinite(gradient).all():
            raise PrefixRouteFairnessError(
                f"trainable parameter has non-finite gradient: {adapter.arm}.{name}"
            )
        gradient_l1 = float(gradient.detach().abs().sum().item())
        if gradient_l1 <= 0:
            raise PrefixRouteFairnessError(
                f"dummy or inactive trainable parameter: {adapter.arm}.{name}"
            )
        gradient_inventory.append(
            {
                "name": name,
                "numel": int(parameter.numel()),
                "gradient_l1": gradient_l1,
            }
        )
    optimizer.step()
    torch.cuda.synchronize()
    train_macs = _profiler_macs(profile)
    peak_training_memory = float(
        max(
            0,
            torch.cuda.max_memory_allocated() - baseline_training_memory,
        )
    )

    adapter.reset_runtime_state()
    torch.cuda.synchronize()
    baseline_inference_memory = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        with torch.profiler.profile(
            activities=activities,
            with_flops=True,
        ) as profile:
            _, causal_state = adapter.inference_forward()
    inference_macs = _profiler_macs(profile)
    torch.cuda.synchronize()
    peak_inference_memory = float(
        max(
            0,
            torch.cuda.max_memory_allocated() - baseline_inference_memory,
        )
    )
    live_state_bytes = float(_tensor_bytes(causal_state, torch))

    adapter.reset_runtime_state()
    with torch.no_grad():
        for _ in range(WARMUP_DECISIONS):
            adapter.inference_forward()
        torch.cuda.synchronize()
        latencies = []
        for _ in range(TIMED_DECISIONS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            adapter.inference_forward()
            end.record()
            end.synchronize()
            latencies.append(float(start.elapsed_time(end)))
    latencies.sort()
    median = statistics.median(latencies)
    p95_index = max(0, math.ceil(0.95 * len(latencies)) - 1)
    return {
        "arm": adapter.arm,
        "trainable_parameters": int(parameter_count),
        "train_macs_per_optimizer_event": train_macs,
        "inference_macs_per_decision": inference_macs,
        "live_causal_state_bytes": live_state_bytes,
        "peak_training_memory_bytes": peak_training_memory,
        "peak_inference_memory_bytes": peak_inference_memory,
        "latency_median_ms": median,
        "latency_p95_ms": latencies[p95_index],
        **budget,
        "budget_source_sha256": budget_evidence["source_sha256"],
        "parameter_gradient_inventory": gradient_inventory,
    }


def _derive_checks(by_arm):
    anchor = by_arm["B2"]
    relative_fields = {
        "trainable_parameters": PARAMETER_TOLERANCE,
        "train_macs_per_optimizer_event": RESOURCE_TOLERANCE,
        "inference_macs_per_decision": RESOURCE_TOLERANCE,
        "live_causal_state_bytes": RESOURCE_TOLERANCE,
        "peak_training_memory_bytes": RESOURCE_TOLERANCE,
        "peak_inference_memory_bytes": RESOURCE_TOLERANCE,
    }
    checks = {}
    for arm in CORE_ARMS:
        row = by_arm[arm]
        exact = {
            field: row[field] == anchor[field] for field in EXACT_BUDGET_FIELDS
        }
        relative = {}
        for field, tolerance in relative_fields.items():
            difference = _relative_difference(row[field], anchor[field])
            relative[field] = {
                "relative_difference": difference,
                "tolerance": tolerance,
                "pass": difference <= tolerance,
            }
        latency = {
            "median_ratio_to_B2": (
                row["latency_median_ms"] / anchor["latency_median_ms"]
                if anchor["latency_median_ms"] > 0
                else math.inf
            ),
            "p95_ratio_to_B2": (
                row["latency_p95_ms"] / anchor["latency_p95_ms"]
                if anchor["latency_p95_ms"] > 0
                else math.inf
            ),
        }
        latency["pass"] = (
            latency["median_ratio_to_B2"] <= LATENCY_UPPER_RATIO
            and latency["p95_ratio_to_B2"] <= LATENCY_UPPER_RATIO
        )
        checks[arm] = {
            "exact_budget_checks": exact,
            "relative_resource_checks": relative,
            "latency_check": latency,
            "pass": (
                all(exact.values())
                and all(item["pass"] for item in relative.values())
                and latency["pass"]
            ),
        }
    return checks


def derive_fairness_audit(runtimes):
    """Measure live arm executions and derive fairness without asserted rows."""

    if not isinstance(runtimes, (list, tuple)) or any(
        type(runtime) is not ArmRuntimeAdapter for runtime in runtimes
    ):
        raise PrefixRouteFairnessError(
            "fairness requires live ArmRuntimeAdapter objects, not serialized rows"
        )
    if [runtime.arm for runtime in runtimes] != list(CORE_ARMS):
        raise PrefixRouteFairnessError(
            "runtime adapters must be ordered exactly B0, B1, B2, B3, B4"
        )
    try:
        import torch
    except Exception as exc:
        raise PrefixRouteFairnessError(
            "target PyTorch runtime is unavailable"
        ) from exc
    rows = [_measure_runtime(runtime, torch) for runtime in runtimes]
    by_arm = {row["arm"]: row for row in rows}
    checks = _derive_checks(by_arm)
    return {
        "schema_version": FAIRNESS_SCHEMA,
        "evidence_source": (
            "LIVE_MODEL_GRADIENT_PROFILER_CUDA_AND_STATE_MEASUREMENT_PLUS_"
            "HASH_VERIFIED_BUDGET_RECORDS"
        ),
        "mac_definition": "torch_profiler_flops_divided_by_two",
        "serialized_scalar_or_boolean_input_allowed": False,
        "anchor_arm": "B2",
        "warmup_decisions": WARMUP_DECISIONS,
        "timed_decisions": TIMED_DECISIONS,
        "parameter_relative_tolerance": PARAMETER_TOLERANCE,
        "resource_relative_tolerance": RESOURCE_TOLERANCE,
        "latency_upper_ratio": LATENCY_UPPER_RATIO,
        "rows": by_arm,
        "checks": checks,
        "status": (
            "PASS_BOTH_CAPACITY_AND_RESOURCE_MATCHED"
            if all(result["pass"] for result in checks.values())
            else "FAIL_ARM_FAIRNESS"
        ),
    }


__all__ = [
    "ArmRuntimeAdapter",
    "BUDGET_EVIDENCE_SCHEMA",
    "CALIBRATION_MANIFEST_SCHEMA",
    "CORE_ARMS",
    "EXECUTION_TRACE_SCHEMA",
    "EXACT_BUDGET_FIELDS",
    "FAIRNESS_SCHEMA",
    "FROZEN_RUN_SEEDS",
    "LATENCY_UPPER_RATIO",
    "PARAMETER_TOLERANCE",
    "RESOURCE_TOLERANCE",
    "SEED_MANIFEST_SCHEMA",
    "TIMED_DECISIONS",
    "TRIAL_MANIFEST_SCHEMA",
    "WARMUP_DECISIONS",
    "PrefixRouteFairnessError",
    "derive_fairness_audit",
    "derive_runtime_budget_evidence",
]
