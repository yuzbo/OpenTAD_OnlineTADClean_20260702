"""Outcome-blind gate for the four-arm CRS-EPS G0 replay audit."""

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath


SELECTION_SCHEMA_VERSION = "full-petal-crs-eps-g0-selection-v3"
MARGIN_SCHEMA_VERSION = "full-petal-crs-eps-g0-margins-v3"
GATE_SCHEMA_VERSION = "full-petal-crs-eps-g0-gate-v1"
AUDIT_SCHEMA_VERSION = "full-petal-crs-eps-g0-audit-v3"
CHECKPOINT_GENERATION_SCHEMA_VERSION = (
    "full-petal-crs-eps-g0-checkpoint-generation-v1"
)
CHECKPOINT_GENERATION_METHOD = (
    "python_numpy_torch_seed_then_build_detector_state_dict_v1"
)
SELECTION_ATTESTATION_ROLE = "crs-eps-g0-selection"
MARGIN_ATTESTATION_ROLE = "crs-eps-g0-margins"
AUDIT_ATTESTATION_ROLE = "crs-eps-g0-audit"
REQUIRED_MODES = ("dynamic_birth", "fixed_192", "reset")
_BINDING_FIELDS = {
    "commit_sha",
    "resolved_config_sha256",
    "scientific_config_sha256",
    "data_identity_sha256",
    "episode_manifest_sha256",
    "sampling_specs_sha256",
}
_SELECTION_FIELDS = {
    "schema_version",
    "status",
    "samples",
    "checkpoint",
    *_BINDING_FIELDS,
}
_CHECKPOINT_FIELDS = {
    "path",
    "sha256",
    "byte_size",
    "state_key",
    "generation",
}
_CHECKPOINT_GENERATION_FIELDS = {
    "schema_version",
    "method",
    "seed",
    "parameter_count",
    "not_evidence_of_model_quality",
}
_MARGIN_FIELDS = {
    "schema_version",
    "status",
    *_BINDING_FIELDS,
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
}


class CrsEpsGoldGateError(ValueError):
    pass


def _canonical_sha256(value):
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite(value, label, *, lower=None, upper=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CrsEpsGoldGateError(f"{label} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise CrsEpsGoldGateError(f"{label} must be finite")
    if lower is not None and value < lower:
        raise CrsEpsGoldGateError(f"{label} is below its allowed range")
    if upper is not None and value > upper:
        raise CrsEpsGoldGateError(f"{label} exceeds its allowed range")
    return value


def _sha256_text(value, label, *, length=64):
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CrsEpsGoldGateError(f"{label} is malformed")
    return value


def validate_gold_checkpoint(checkpoint):
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != _CHECKPOINT_FIELDS:
        raise CrsEpsGoldGateError("G0 checkpoint fields differ from the frozen schema")
    normalized = dict(checkpoint)
    path_value = normalized["path"]
    if not isinstance(path_value, str) or not path_value or "\\" in path_value:
        raise CrsEpsGoldGateError("G0 checkpoint path is not canonical relative text")
    relative = PurePosixPath(path_value)
    if (
        relative.is_absolute()
        or relative.as_posix() != path_value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise CrsEpsGoldGateError("G0 checkpoint path is not canonical relative text")
    _sha256_text(normalized["sha256"], "G0 checkpoint SHA256")
    byte_size = normalized["byte_size"]
    if isinstance(byte_size, bool) or not isinstance(byte_size, int) or byte_size <= 0:
        raise CrsEpsGoldGateError("G0 checkpoint byte_size must be positive")
    if normalized["state_key"] not in {"state_dict", "state_dict_ema"}:
        raise CrsEpsGoldGateError("G0 checkpoint state key is unsupported")
    generation = normalized["generation"]
    if (
        not isinstance(generation, Mapping)
        or set(generation) != _CHECKPOINT_GENERATION_FIELDS
    ):
        raise CrsEpsGoldGateError("G0 checkpoint generation fields differ")
    generation = dict(generation)
    if generation["schema_version"] != CHECKPOINT_GENERATION_SCHEMA_VERSION:
        raise CrsEpsGoldGateError("G0 checkpoint generation schema is unsupported")
    if generation["method"] != CHECKPOINT_GENERATION_METHOD:
        raise CrsEpsGoldGateError("G0 checkpoint generation method is unsupported")
    for field in ("seed", "parameter_count"):
        value = generation[field]
        minimum = 0 if field == "seed" else 1
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum
            or (field == "seed" and value > 0xFFFFFFFF)
        ):
            raise CrsEpsGoldGateError(f"G0 checkpoint generation {field} is invalid")
    if generation["not_evidence_of_model_quality"] is not True:
        raise CrsEpsGoldGateError(
            "G0 checkpoint must disclaim evidence of model quality"
        )
    normalized["generation"] = generation
    return normalized


def validate_gold_selection(selection):
    if not isinstance(selection, Mapping) or set(selection) != _SELECTION_FIELDS:
        raise CrsEpsGoldGateError("G0 selection fields differ from the frozen schema")
    normalized = dict(selection)
    if normalized["schema_version"] != SELECTION_SCHEMA_VERSION:
        raise CrsEpsGoldGateError("G0 selection schema is unsupported")
    if normalized["status"] != "PREREGISTERED_BEFORE_G0_EXECUTION":
        raise CrsEpsGoldGateError("G0 selection is not preregistered before execution")
    _sha256_text(normalized["commit_sha"], "G0 selection commit", length=40)
    for field in _BINDING_FIELDS - {"commit_sha"}:
        _sha256_text(normalized[field], f"G0 selection {field}")
    normalized["checkpoint"] = validate_gold_checkpoint(normalized["checkpoint"])
    samples = normalized["samples"]
    if not isinstance(samples, list) or not samples:
        raise CrsEpsGoldGateError("G0 selection requires at least one sample")
    seen = set()
    for sample in samples:
        if not isinstance(sample, Mapping) or set(sample) != {"video_id", "draw_index"}:
            raise CrsEpsGoldGateError("G0 selected-sample fields differ")
        video_id = sample["video_id"]
        draw_index = sample["draw_index"]
        if not isinstance(video_id, str) or not video_id:
            raise CrsEpsGoldGateError("G0 selected video ID is invalid")
        if isinstance(draw_index, bool) or not isinstance(draw_index, int) or draw_index < 0:
            raise CrsEpsGoldGateError("G0 selected draw index is invalid")
        key = (video_id, draw_index)
        if key in seen:
            raise CrsEpsGoldGateError("G0 selection contains a duplicate sample")
        seen.add(key)
    return normalized


def validate_gold_margins(margins):
    if not isinstance(margins, Mapping) or set(margins) != _MARGIN_FIELDS:
        raise CrsEpsGoldGateError("G0 margin fields differ from the frozen schema")
    normalized = dict(margins)
    if normalized["schema_version"] != MARGIN_SCHEMA_VERSION:
        raise CrsEpsGoldGateError("G0 margin schema is unsupported")
    if normalized["status"] != "PREREGISTERED_BEFORE_G0_EXECUTION":
        raise CrsEpsGoldGateError("G0 margins are not marked outcome-blind")
    _sha256_text(normalized["commit_sha"], "G0 margin commit", length=40)
    for field in (_BINDING_FIELDS - {"commit_sha"}) | {"selection_artifact_sha256"}:
        _sha256_text(normalized[field], f"G0 margin {field}")
    for field in (
        "min_gradient_cosine",
        "min_gradient_sign_agreement",
        "min_runtime_continuous_cosine",
    ):
        normalized[field] = _finite(normalized[field], field, lower=-1.0, upper=1.0)
    for field in (
        "max_video_start_fallback_fraction",
        "max_mean_dynamic_replay_ratio",
    ):
        normalized[field] = _finite(normalized[field], field, lower=0.0, upper=1.0)
    normalized["max_relative_loss_error"] = _finite(
        normalized["max_relative_loss_error"],
        "max_relative_loss_error",
        lower=0.0,
    )
    for field in (
        "min_dynamic_minus_fixed_gradient_cosine",
        "min_dynamic_minus_reset_gradient_cosine",
    ):
        normalized[field] = _finite(normalized[field], field, lower=-2.0, upper=2.0)
    if not isinstance(normalized["require_runtime_discrete_equal"], bool):
        raise CrsEpsGoldGateError("require_runtime_discrete_equal must be boolean")
    return normalized


def _comparison_value(trace, section, field):
    value = trace.get(section, {}).get(field)
    return None if value is None else _finite(value, f"{section}.{field}")


def _relative_loss_error(trace):
    left = trace.get("left", {}).get("losses")
    right = trace.get("right", {}).get("losses")
    if not isinstance(left, Mapping) or not isinstance(right, Mapping) or set(left) != set(right):
        raise CrsEpsGoldGateError("G0 paired losses differ in structure")
    errors = []
    for name in sorted(left):
        gold = _finite(left[name], f"gold loss {name}")
        candidate = _finite(right[name], f"candidate loss {name}")
        denominator = max(abs(gold), 1e-12)
        errors.append(abs(candidate - gold) / denominator)
    return max(errors, default=0.0)


def _replay_ratio(trace):
    left = trace.get("left", {}).get("audit", {}).get("replay_range")
    right = trace.get("right", {}).get("audit", {}).get("replay_range")
    if (
        not isinstance(left, (list, tuple))
        or not isinstance(right, (list, tuple))
        or len(left) != 2
        or len(right) != 2
    ):
        raise CrsEpsGoldGateError("G0 replay ranges are malformed")
    gold_tokens = int(left[1]) - int(left[0])
    candidate_tokens = int(right[1]) - int(right[0])
    if gold_tokens <= 0 or candidate_tokens <= 0 or candidate_tokens > gold_tokens:
        raise CrsEpsGoldGateError("G0 replay token geometry is invalid")
    return candidate_tokens / gold_tokens, int(right[0]) == int(left[0])


def _minimum_or_none(rows, field):
    values = [row[field] for row in rows]
    return min(values) if all(value is not None for value in values) else None


def evaluate_gold_audit(rows: Sequence[Mapping], margins):
    margins = validate_gold_margins(margins)
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise CrsEpsGoldGateError("G0 audit requires a non-empty trace sequence")
    grouped = defaultdict(dict)
    normalized_rows = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "video_id",
            "draw_index",
            "mode",
            "trace",
        }:
            raise CrsEpsGoldGateError("G0 trace-row fields differ")
        video_id = row["video_id"]
        draw_index = row["draw_index"]
        mode = row["mode"]
        trace = row["trace"]
        if not isinstance(video_id, str) or not video_id:
            raise CrsEpsGoldGateError("G0 trace video ID is invalid")
        if isinstance(draw_index, bool) or not isinstance(draw_index, int) or draw_index < 0:
            raise CrsEpsGoldGateError("G0 trace draw index is invalid")
        if mode not in REQUIRED_MODES:
            raise CrsEpsGoldGateError("G0 trace mode is unsupported")
        if not isinstance(trace, Mapping) or trace.get("comparison_type") != "replay_fidelity":
            raise CrsEpsGoldGateError("G0 trace is not a replay-fidelity comparison")
        key = (video_id, draw_index)
        if mode in grouped[key]:
            raise CrsEpsGoldGateError("G0 trace contains a duplicate arm")
        grouped[key][mode] = trace
        normalized_rows.append(dict(row))
    for key, arms in grouped.items():
        if set(arms) != set(REQUIRED_MODES):
            raise CrsEpsGoldGateError(f"G0 sample {key} lacks an exact four-arm comparison")

    violations = []
    dynamic_replay_ratios = []
    dynamic_fallbacks = 0
    dynamic_metrics = []
    deltas_fixed = []
    deltas_reset = []
    for key, arms in sorted(grouped.items()):
        mode_metrics = {}
        for mode, trace in arms.items():
            gradient_cosine = _comparison_value(trace, "gradient_comparison", "cosine")
            gradient_sign = _comparison_value(
                trace, "gradient_comparison", "sign_agreement"
            )
            runtime_cosine = _comparison_value(
                trace, "runtime_continuous_comparison", "cosine"
            )
            replay_ratio, fallback = _replay_ratio(trace)
            mode_metrics[mode] = {
                "gradient_cosine": gradient_cosine,
                "gradient_sign_agreement": gradient_sign,
                "runtime_continuous_cosine": runtime_cosine,
                "relative_loss_error": _relative_loss_error(trace),
                "runtime_discrete_equal": trace.get("runtime_discrete_equal") is True,
                "replay_ratio": replay_ratio,
                "video_start_fallback": fallback,
            }
        dynamic = mode_metrics["dynamic_birth"]
        dynamic_metrics.append(dynamic)
        dynamic_replay_ratios.append(dynamic["replay_ratio"])
        dynamic_fallbacks += int(dynamic["video_start_fallback"])
        prefix = f"{key[0]}:draw={key[1]}"
        for metric, threshold in (
            ("gradient_cosine", margins["min_gradient_cosine"]),
            ("gradient_sign_agreement", margins["min_gradient_sign_agreement"]),
            ("runtime_continuous_cosine", margins["min_runtime_continuous_cosine"]),
        ):
            if dynamic[metric] is None or dynamic[metric] < threshold:
                violations.append(f"{prefix}:{metric}")
        if dynamic["relative_loss_error"] > margins["max_relative_loss_error"]:
            violations.append(f"{prefix}:relative_loss_error")
        if margins["require_runtime_discrete_equal"] and not dynamic[
            "runtime_discrete_equal"
        ]:
            violations.append(f"{prefix}:runtime_discrete")
        fixed_cosine = mode_metrics["fixed_192"]["gradient_cosine"]
        reset_cosine = mode_metrics["reset"]["gradient_cosine"]
        if dynamic["gradient_cosine"] is None or fixed_cosine is None or reset_cosine is None:
            raise CrsEpsGoldGateError("G0 gradient comparisons are not vector-compatible")
        deltas_fixed.append(dynamic["gradient_cosine"] - fixed_cosine)
        deltas_reset.append(dynamic["gradient_cosine"] - reset_cosine)

    sample_count = len(grouped)
    mean_replay_ratio = sum(dynamic_replay_ratios) / sample_count
    fallback_fraction = dynamic_fallbacks / sample_count
    mean_delta_fixed = sum(deltas_fixed) / sample_count
    mean_delta_reset = sum(deltas_reset) / sample_count
    if mean_replay_ratio > margins["max_mean_dynamic_replay_ratio"]:
        violations.append("aggregate:mean_dynamic_replay_ratio")
    if fallback_fraction > margins["max_video_start_fallback_fraction"]:
        violations.append("aggregate:video_start_fallback_fraction")
    if mean_delta_fixed < margins["min_dynamic_minus_fixed_gradient_cosine"]:
        violations.append("aggregate:dynamic_minus_fixed_gradient_cosine")
    if mean_delta_reset < margins["min_dynamic_minus_reset_gradient_cosine"]:
        violations.append("aggregate:dynamic_minus_reset_gradient_cosine")

    result = {
        "schema_version": GATE_SCHEMA_VERSION,
        "status": "PASS" if not violations else "KILL",
        "sample_count": sample_count,
        "trace_count": len(normalized_rows),
        "margins_sha256": _canonical_sha256(margins),
        "traces_sha256": _canonical_sha256(normalized_rows),
        "metrics": {
            "mean_dynamic_replay_ratio": mean_replay_ratio,
            "video_start_fallback_fraction": fallback_fraction,
            "mean_dynamic_minus_fixed_gradient_cosine": mean_delta_fixed,
            "mean_dynamic_minus_reset_gradient_cosine": mean_delta_reset,
            "max_dynamic_relative_loss_error": max(
                item["relative_loss_error"] for item in dynamic_metrics
            ),
            "min_dynamic_gradient_cosine": _minimum_or_none(
                dynamic_metrics, "gradient_cosine"
            ),
            "min_dynamic_gradient_sign_agreement": _minimum_or_none(
                dynamic_metrics, "gradient_sign_agreement"
            ),
            "min_dynamic_runtime_continuous_cosine": _minimum_or_none(
                dynamic_metrics, "runtime_continuous_cosine"
            ),
        },
        "violations": sorted(violations),
    }
    result["gate_sha256"] = _canonical_sha256(result)
    return result


__all__ = [
    "AUDIT_ATTESTATION_ROLE",
    "AUDIT_SCHEMA_VERSION",
    "CHECKPOINT_GENERATION_METHOD",
    "CHECKPOINT_GENERATION_SCHEMA_VERSION",
    "CrsEpsGoldGateError",
    "GATE_SCHEMA_VERSION",
    "MARGIN_ATTESTATION_ROLE",
    "MARGIN_SCHEMA_VERSION",
    "SELECTION_ATTESTATION_ROLE",
    "SELECTION_SCHEMA_VERSION",
    "evaluate_gold_audit",
    "validate_gold_checkpoint",
    "validate_gold_margins",
    "validate_gold_selection",
]
