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


_FAIRNESS_CAPABILITY_TOKEN = object()


@dataclass(frozen=True)
class VerifiedFairnessAudit:
    """In-process capability minted only after live CUDA measurement."""

    record: Mapping[str, Any]
    record_sha256: str
    _token: object


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
    model_tensor_artifact_reference: Mapping[str, str] | None = None


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


def _sha256_text(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PrefixRouteFairnessError(f"{label} must be a lowercase SHA-256")
    return value


def _runtime_state_sha256(value, torch):
    digest = hashlib.sha256()

    def update(item):
        if torch.is_tensor(item):
            tensor = item.detach().contiguous().cpu()
            digest.update(b"T")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(_canonical_json_bytes(list(tensor.shape)))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"D")
            for key in sorted(item, key=lambda candidate: repr(candidate)):
                update(key)
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"L")
            for child in item:
                update(child)
        elif item is None or isinstance(item, (bool, int, float, str)):
            digest.update(b"J")
            digest.update(_canonical_json_bytes(item))
        else:
            raise PrefixRouteFairnessError(
                "runtime state contains an unsupported value"
            )

    update(value)
    return digest.hexdigest()


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


def _profile_macs_from_rows(rows):
    if not isinstance(rows, list) or not rows:
        raise PrefixRouteFairnessError("profiler evidence is empty")
    total = 0
    keys = []
    for row in rows:
        _exact(row, {"key", "count", "flops"}, "profiler event")
        if (
            not isinstance(row["key"], str)
            or not row["key"]
            or isinstance(row["count"], bool)
            or not isinstance(row["count"], int)
            or row["count"] <= 0
        ):
            raise PrefixRouteFairnessError("profiler event identity differs")
        flops = _finite_nonnegative(row["flops"], "profiler event FLOPs")
        if not float(flops).is_integer():
            raise PrefixRouteFairnessError("profiler event FLOPs must be integral")
        total += int(flops)
        keys.append(row["key"])
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise PrefixRouteFairnessError(
            "profiler events must have sorted unique keys"
        )
    if total <= 0:
        raise PrefixRouteFairnessError(
            "Torch profiler reported no FLOPs for the measured path"
        )
    return float(total) / 2.0


def _profiler_evidence(profiler):
    rows = sorted(
        (
            {
                "key": str(event.key),
                "count": int(event.count),
                "flops": int(getattr(event, "flops", 0) or 0),
            }
            for event in profiler.key_averages()
        ),
        key=lambda row: row["key"],
    )
    return _profile_macs_from_rows(rows), rows


def _model_tensor_summary(model, torch):
    from .prefix_route_artifacts_v2 import (
        PrefixRouteArtifactError,
        tensor_state_digest_from_named_arrays,
    )

    state = model.state_dict()
    if not isinstance(state, Mapping) or not state:
        raise PrefixRouteFairnessError("profiled model state is empty")
    arrays = {}
    for name, tensor in state.items():
        if not isinstance(name, str) or not torch.is_tensor(tensor):
            raise PrefixRouteFairnessError(
                "profiled model state must contain named tensors only"
            )
        try:
            arrays[name] = tensor.detach().contiguous().cpu().numpy()
        except Exception as exc:
            raise PrefixRouteFairnessError(
                f"profiled model tensor cannot be canonicalized: {name}"
            ) from exc
    try:
        return tensor_state_digest_from_named_arrays(arrays)
    except PrefixRouteArtifactError as exc:
        raise PrefixRouteFairnessError(str(exc)) from exc


def _verify_model_tensor_artifact(
    model,
    torch,
    *,
    reference,
    bundle_root,
    arm,
):
    from .prefix_route_artifacts_v2 import (
        PrefixRouteArtifactError,
        read_tensor_state_artifact,
    )

    model_tensor_summary = _model_tensor_summary(model, torch)
    if reference is None:
        raise PrefixRouteFairnessError(
            "live fairness requires the profiled model tensor artifact"
        )
    try:
        model_artifact = read_tensor_state_artifact(
            reference,
            bundle_root=bundle_root,
            label=f"{arm} profiled model tensor artifact",
        )
    except PrefixRouteArtifactError as exc:
        raise PrefixRouteFairnessError(str(exc)) from exc
    if (
        model_artifact["tensor_state_sha256"]
        != model_tensor_summary["tensor_state_sha256"]
    ):
        raise PrefixRouteFairnessError(
            "profiled model tensors differ from the bound checkpoint artifact"
        )
    return model_tensor_summary, model_artifact


def _cuda_identity(torch):
    index = int(torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(index)
    return {
        "device_index": index,
        "device_name": str(properties.name),
        "total_memory_bytes": int(properties.total_memory),
        "compute_capability": [
            int(properties.major),
            int(properties.minor),
        ],
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
    }


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
    previous_model_after = None
    for index, event in enumerate(trace["events"]):
        _exact(
            event,
            {
                "optimizer_event_index",
                "gradient_accumulation_steps",
                "effective_token_count",
                "status",
                "input_batches_sha256",
                "model_state_before_sha256",
                "model_state_after_sha256",
                "optimizer_state_after_sha256",
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
        for field in (
            "input_batches_sha256",
            "model_state_before_sha256",
            "model_state_after_sha256",
            "optimizer_state_after_sha256",
        ):
            _sha256_text(event[field], f"optimizer event {index} {field}")
        if (
            previous_model_after is not None
            and event["model_state_before_sha256"] != previous_model_after
        ):
            raise PrefixRouteFairnessError(
                "optimizer event model-state chain differs"
            )
        if event["model_state_before_sha256"] == event["model_state_after_sha256"]:
            raise PrefixRouteFairnessError(
                "optimizer event did not change model state"
            )
        previous_model_after = event["model_state_after_sha256"]
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
    """Reject standalone records that are not tied to a live runtime audit."""

    raise PrefixRouteFairnessError(
        "standalone budget evidence is forbidden; use the live fairness audit"
    )


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
    model_state_before = _runtime_state_sha256(model.state_dict(), torch)
    model_tensor_before, model_artifact = _verify_model_tensor_artifact(
        model,
        torch,
        reference=adapter.model_tensor_artifact_reference,
        bundle_root=adapter.budget_evidence_root,
        arm=adapter.arm,
    )
    if model_state_before != profile_event["model_state_before_sha256"]:
        raise PrefixRouteFairnessError(
            "profiled model state differs from execution trace"
        )
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
        input_batch_hashes = []
        for item in forwards:
            if not isinstance(item, tuple) or len(item) != 3:
                raise PrefixRouteFairnessError(
                    "each optimizer-event forward must return loss, token count, "
                    "and input commitment"
                )
            loss, effective_tokens, input_batch_sha256 = item
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
            input_batch_hashes.append(
                _sha256_text(
                    input_batch_sha256,
                    "smoke input batch commitment",
                )
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
        profiled_input_sha256 = hashlib.sha256(
            _canonical_json_bytes(input_batch_hashes)
        ).hexdigest()
        if profiled_input_sha256 != profile_event["input_batches_sha256"]:
            raise PrefixRouteFairnessError(
                "profiled input batches differ from execution trace"
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
    model_state_after = _runtime_state_sha256(model.state_dict(), torch)
    optimizer_state_after = _runtime_state_sha256(optimizer.state_dict(), torch)
    if (
        model_state_after != profile_event["model_state_after_sha256"]
        or optimizer_state_after
        != profile_event["optimizer_state_after_sha256"]
    ):
        raise PrefixRouteFairnessError(
            "profiled post-update state differs from execution trace"
        )
    torch.cuda.synchronize()
    train_macs, train_profiler_events = _profiler_evidence(profile)
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
    inference_macs, inference_profiler_events = _profiler_evidence(profile)
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
    cuda_identity = _cuda_identity(torch)
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
        "profiled_model_artifact_reference": dict(
            adapter.model_tensor_artifact_reference
        ),
        "profiled_model_artifact_sha256": model_artifact["artifact_sha256"],
        "profiled_model_tensor_state_sha256": model_tensor_before[
            "tensor_state_sha256"
        ],
        "profiled_model_tensor_count": model_tensor_before["tensor_count"],
        "train_profiler_events": train_profiler_events,
        "inference_profiler_events": inference_profiler_events,
        "latency_samples_ms": latencies,
        "cuda_identity": cuda_identity,
        **budget,
        "budget_source_sha256": budget_evidence["source_sha256"],
        "budget_source_references": {
            "execution_trace": dict(adapter.execution_trace_reference),
            "calibration_manifest": dict(
                adapter.calibration_manifest_reference
            ),
            "trial_manifest": dict(adapter.trial_manifest_reference),
            "seed_manifest": dict(adapter.seed_manifest_reference),
        },
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
            "LIVE_CLEAN_PROCESS_MODEL_OPTIMIZER_GRADIENT_CUDA_PROFILER_AND_"
            "ARTIFACT_DERIVED_TENSOR_STATE"
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


def derive_live_fairness_capability(runtimes):
    """Mint the capability consumed by formal R6 in the same clean process."""

    record = derive_fairness_audit(runtimes)
    return VerifiedFairnessAudit(
        record=record,
        record_sha256=hashlib.sha256(_canonical_json_bytes(record)).hexdigest(),
        _token=_FAIRNESS_CAPABILITY_TOKEN,
    )


def require_live_fairness_capability(capability):
    if (
        type(capability) is not VerifiedFairnessAudit
        or capability._token is not _FAIRNESS_CAPABILITY_TOKEN
    ):
        raise PrefixRouteFairnessError(
            "formal fairness requires a live clean-process capability"
        )
    record = capability.record
    if (
        not isinstance(record, dict)
        or record.get("status") != "PASS_BOTH_CAPACITY_AND_RESOURCE_MATCHED"
        or hashlib.sha256(_canonical_json_bytes(record)).hexdigest()
        != capability.record_sha256
    ):
        raise PrefixRouteFairnessError("live fairness capability is invalid")
    return record, capability.record_sha256


def validate_fairness_audit_record(record, *, bundle_root):
    """Validate archive shape only; serialized rows can never authorize R6."""

    required = {
        "schema_version",
        "evidence_source",
        "mac_definition",
        "serialized_scalar_or_boolean_input_allowed",
        "anchor_arm",
        "warmup_decisions",
        "timed_decisions",
        "parameter_relative_tolerance",
        "resource_relative_tolerance",
        "latency_upper_ratio",
        "rows",
        "checks",
        "status",
    }
    _exact(record, required, "fairness audit")
    if (
        record["schema_version"] != FAIRNESS_SCHEMA
        or record["evidence_source"]
        != (
            "LIVE_CLEAN_PROCESS_MODEL_OPTIMIZER_GRADIENT_CUDA_PROFILER_AND_"
            "ARTIFACT_DERIVED_TENSOR_STATE"
        )
        or record["mac_definition"] != "torch_profiler_flops_divided_by_two"
        or record["serialized_scalar_or_boolean_input_allowed"] is not False
        or record["anchor_arm"] != "B2"
        or record["warmup_decisions"] != WARMUP_DECISIONS
        or record["timed_decisions"] != TIMED_DECISIONS
        or record["parameter_relative_tolerance"] != PARAMETER_TOLERANCE
        or record["resource_relative_tolerance"] != RESOURCE_TOLERANCE
        or record["latency_upper_ratio"] != LATENCY_UPPER_RATIO
    ):
        raise PrefixRouteFairnessError("fairness audit identity differs")
    rows = record["rows"]
    if not isinstance(rows, dict) or set(rows) != set(CORE_ARMS):
        raise PrefixRouteFairnessError("fairness audit arm rows differ")
    row_fields = {
        "arm",
        "trainable_parameters",
        "train_macs_per_optimizer_event",
        "inference_macs_per_decision",
        "live_causal_state_bytes",
        "peak_training_memory_bytes",
        "peak_inference_memory_bytes",
        "latency_median_ms",
        "latency_p95_ms",
        "profiled_model_artifact_reference",
        "profiled_model_artifact_sha256",
        "profiled_model_tensor_state_sha256",
        "profiled_model_tensor_count",
        "train_profiler_events",
        "inference_profiler_events",
        "latency_samples_ms",
        "cuda_identity",
        *EXACT_BUDGET_FIELDS,
        "budget_source_sha256",
        "budget_source_references",
        "parameter_gradient_inventory",
    }
    positive_fields = {
        "trainable_parameters",
        "train_macs_per_optimizer_event",
        "inference_macs_per_decision",
        "peak_training_memory_bytes",
        "peak_inference_memory_bytes",
        "latency_median_ms",
        "latency_p95_ms",
        "profiled_model_tensor_count",
        *EXACT_BUDGET_FIELDS,
    }
    for arm in CORE_ARMS:
        row = rows[arm]
        _exact(row, row_fields, f"fairness row {arm}")
        if row["arm"] != arm:
            raise PrefixRouteFairnessError("fairness row arm identity differs")
        for field in positive_fields:
            if _finite_nonnegative(row[field], f"{arm}.{field}") <= 0:
                raise PrefixRouteFairnessError(
                    f"{arm}.{field} must be strictly positive"
                )
        _finite_nonnegative(
            row["live_causal_state_bytes"],
            f"{arm}.live_causal_state_bytes",
        )
        _sha256_text(
            row["profiled_model_artifact_sha256"],
            f"{arm}.profiled_model_artifact_sha256",
        )
        _sha256_text(
            row["profiled_model_tensor_state_sha256"],
            f"{arm}.profiled_model_tensor_state_sha256",
        )
        reference = row["profiled_model_artifact_reference"]
        _exact(
            reference,
            {"path", "sha256"},
            f"{arm}.profiled_model_artifact_reference",
        )
        if (
            not isinstance(reference["path"], str)
            or not reference["path"]
            or reference["sha256"]
            != row["profiled_model_artifact_sha256"]
        ):
            raise PrefixRouteFairnessError(
                f"profiled model artifact reference differs for {arm}"
            )
        if not math.isclose(
            _profile_macs_from_rows(row["train_profiler_events"]),
            float(row["train_macs_per_optimizer_event"]),
            rel_tol=0.0,
            abs_tol=0.0,
        ) or not math.isclose(
            _profile_macs_from_rows(row["inference_profiler_events"]),
            float(row["inference_macs_per_decision"]),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise PrefixRouteFairnessError(
                f"profiler event totals differ for {arm}"
            )
        latency_samples = row["latency_samples_ms"]
        if (
            not isinstance(latency_samples, list)
            or len(latency_samples) != TIMED_DECISIONS
            or latency_samples != sorted(latency_samples)
            or any(
                _finite_nonnegative(value, f"{arm}.latency_sample") <= 0
                for value in latency_samples
            )
            or not math.isclose(
                statistics.median(latency_samples),
                float(row["latency_median_ms"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                latency_samples[
                    max(0, math.ceil(0.95 * len(latency_samples)) - 1)
                ],
                float(row["latency_p95_ms"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise PrefixRouteFairnessError(
                f"latency samples differ for {arm}"
            )
        cuda_identity = row["cuda_identity"]
        _exact(
            cuda_identity,
            {
                "device_index",
                "device_name",
                "total_memory_bytes",
                "compute_capability",
                "torch_version",
                "cuda_version",
            },
            f"{arm}.cuda_identity",
        )
        if (
            isinstance(cuda_identity["device_index"], bool)
            or not isinstance(cuda_identity["device_index"], int)
            or cuda_identity["device_index"] < 0
            or not isinstance(cuda_identity["device_name"], str)
            or not cuda_identity["device_name"]
            or isinstance(cuda_identity["total_memory_bytes"], bool)
            or not isinstance(cuda_identity["total_memory_bytes"], int)
            or cuda_identity["total_memory_bytes"] <= 0
            or not isinstance(cuda_identity["compute_capability"], list)
            or len(cuda_identity["compute_capability"]) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in cuda_identity["compute_capability"]
            )
            or not isinstance(cuda_identity["torch_version"], str)
            or not cuda_identity["torch_version"]
            or not isinstance(cuda_identity["cuda_version"], str)
            or not cuda_identity["cuda_version"]
        ):
            raise PrefixRouteFairnessError(
                f"CUDA identity differs for {arm}"
            )
        sources = row["budget_source_sha256"]
        _exact(
            sources,
            {
                "execution_trace",
                "calibration_manifest",
                "trial_manifest",
                "seed_manifest",
            },
            f"fairness budget source {arm}",
        )
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in sources.values()
        ):
            raise PrefixRouteFairnessError(
                f"fairness budget source hash differs for {arm}"
            )
        references = row["budget_source_references"]
        _exact(
            references,
            {
                "execution_trace",
                "calibration_manifest",
                "trial_manifest",
                "seed_manifest",
            },
            f"fairness budget source references {arm}",
        )
        inventory = row["parameter_gradient_inventory"]
        if not isinstance(inventory, list) or not inventory:
            raise PrefixRouteFairnessError(
                f"fairness gradient inventory differs for {arm}"
            )
        names = []
        total_parameters = 0
        for item in inventory:
            _exact(
                item,
                {"name", "numel", "gradient_l1"},
                f"fairness gradient row {arm}",
            )
            if (
                not isinstance(item["name"], str)
                or not item["name"]
                or isinstance(item["numel"], bool)
                or not isinstance(item["numel"], int)
                or item["numel"] <= 0
                or _finite_nonnegative(
                    item["gradient_l1"],
                    f"{arm}.{item['name']}.gradient_l1",
                )
                <= 0
            ):
                raise PrefixRouteFairnessError(
                    f"fairness gradient row differs for {arm}"
                )
            names.append(item["name"])
            total_parameters += item["numel"]
        if len(names) != len(set(names)) or total_parameters != (
            row["trainable_parameters"]
        ):
            raise PrefixRouteFairnessError(
                f"fairness gradient inventory total differs for {arm}"
            )
    del bundle_root
    return {
        "verification_status": "UNVERIFIED_SERIALIZED_AUDIT",
        "claimed_status": record["status"],
        "record_sha256": hashlib.sha256(
            _canonical_json_bytes(record)
        ).hexdigest(),
        "formal_fairness_pass": False,
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
    "VerifiedFairnessAudit",
    "derive_fairness_audit",
    "derive_live_fairness_capability",
    "require_live_fairness_capability",
    "validate_fairness_audit_record",
]
