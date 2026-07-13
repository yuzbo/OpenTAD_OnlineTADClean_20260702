"""Strict separation of causal model metadata from stream control state."""

from __future__ import annotations

import copy
import math
import numbers
import re
from collections.abc import Mapping


MODEL_META_ALLOWLIST = frozenset(
    {
        "current_frame",
        "encoded_source_frames",
        "encoder",
        "encoder_id",
        "encoder_policy",
        "feature_dim",
        "feature_stride",
        "fps",
        "frame_policy",
        "frame_stride",
        "image_size",
        "input_format",
        "input_provenance_digest",
        "input_policy",
        "max_cache_source_frame",
        "offset_frames",
        "packet_end_frame",
        "packet_end_token",
        "packet_start_frame",
        "packet_start_token",
        "packet_stride",
        "processor",
        "processor_id",
        "processor_policy",
        "sample_stride",
        "snippet_stride",
        "source_frame",
        "source_frames",
        "stream_id",
        "stride",
        "video_id",
        "video_name",
        "window_end_frame",
        "window_start_frame",
    }
)

STREAM_CONTROL_ALLOWLIST = frozenset(
    {
        "batch_index",
        "chunk_index",
        "duration",
        "is_video_end",
        "is_video_start",
        "lane_index",
        "num_frames",
        "packet_index",
        "reset_stream",
        "stream_id",
        "total_frames",
        "training_targets",
        "video_end_frame",
        "video_id",
        "video_name",
    }
)

TRAINING_TARGET_ALLOWLIST = frozenset(
    {
        "annotations",
        "censored_targets",
        "event_targets",
        "ground_truth",
        "gt_labels",
        "gt_segments",
        "instances",
        "prefix_schedule",
        "targets",
    }
)

MODEL_META_ALIASES = {
    "decision_frame": "current_frame",
    "frame_rate": "fps",
    "input_mode": "input_format",
    "now_frame": "current_frame",
    "observed_frame": "source_frame",
    "observed_frames": "source_frames",
    "sampling_stride": "sample_stride",
    "source_indices": "source_frames",
    "stream": "stream_id",
    "video": "video_id",
    "video_uid": "video_id",
}

STREAM_CONTROL_ALIASES = {
    "duration_sec": "duration",
    "end_of_stream": "is_video_end",
    "eos": "is_video_end",
    "final_frame": "video_end_frame",
    "first_packet": "is_video_start",
    "frame_count": "num_frames",
    "is_final": "is_video_end",
    "is_first": "is_video_start",
    "is_last": "is_video_end",
    "last_frame": "video_end_frame",
    "last_packet": "is_video_end",
    "reset_model_state": "reset_stream",
    "reset_online_state": "reset_stream",
    "reset_state": "reset_stream",
    "sos": "is_video_start",
    "start_of_stream": "is_video_start",
    "terminal": "is_video_end",
    "video_duration": "duration",
    "video_duration_sec": "duration",
    "video_frame_count": "num_frames",
    "video_length_frames": "total_frames",
}

TRAINING_TARGET_ALIASES = {
    "annotation": "annotations",
    "event_schedule": "prefix_schedule",
    "ground_truths": "ground_truth",
    "gt": "ground_truth",
    "gt_label": "gt_labels",
    "gt_segment": "gt_segments",
    "labels": "gt_labels",
    "schedule": "prefix_schedule",
    "segments": "gt_segments",
    "target": "targets",
}

_DIRECT_MODEL_TAINT = frozenset(
    {
        "all_frames",
        "batch_index",
        "chunk_index",
        "emit_frame",
        "end_frame",
        "event_id",
        "full_annotations",
        "full_sequence",
        "future",
        "future_annotations",
        "future_frames",
        "future_schedule",
        "immutable",
        "lane_index",
        "packet_index",
        "previous_hash",
        "row_hash",
        "training_targets",
    }
)

FORBIDDEN_MODEL_META_KEYS = frozenset(
    (
        (STREAM_CONTROL_ALLOWLIST - {"stream_id", "video_id", "video_name"})
        | TRAINING_TARGET_ALLOWLIST
        | set(STREAM_CONTROL_ALIASES)
        | set(STREAM_CONTROL_ALIASES.values())
        | set(TRAINING_TARGET_ALIASES)
        | set(TRAINING_TARGET_ALIASES.values())
        | _DIRECT_MODEL_TAINT
    )
)

_CAMEL_BOUNDARY_1 = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_BOUNDARY_2 = re.compile(r"([a-z0-9])([A-Z])")
_NON_KEY_CHARACTER = re.compile(r"[^A-Za-z0-9]+")
_NESTED_TAINT_TOKENS = frozenset(
    {
        "annotation",
        "endpoint",
        "future",
        "groundtruth",
        "gt",
        "label",
        "lookahead",
        "target",
        "terminal",
    }
)
_IDENTIFIER_FIELDS = frozenset({"stream_id", "video_id", "video_name"})
_HASH_FIELDS = frozenset({"input_provenance_digest"})
_MODEL_SCALAR_FRAMES = frozenset(
    {
        "current_frame",
        "max_cache_source_frame",
        "offset_frames",
        "packet_end_frame",
        "packet_start_frame",
        "source_frame",
        "window_end_frame",
        "window_start_frame",
    }
)
_MODEL_FRAME_SEQUENCES = frozenset({"encoded_source_frames", "source_frames"})
_POSITIVE_STRIDES = frozenset(
    {
        "feature_stride",
        "frame_stride",
        "packet_stride",
        "sample_stride",
        "snippet_stride",
        "stride",
    }
)
_CONTROL_BOOLEAN_FIELDS = frozenset(
    {"is_video_end", "is_video_start", "reset_stream"}
)
_CONTROL_INTEGER_FIELDS = frozenset(
    {
        "batch_index",
        "chunk_index",
        "lane_index",
        "num_frames",
        "packet_index",
        "total_frames",
        "video_end_frame",
    }
)


class StreamMetadataError(ValueError):
    """Raised when metadata crosses the model/control trust boundary."""


def _normalize_key(key):
    if not isinstance(key, str) or not key.strip():
        raise StreamMetadataError("metadata keys must be non-empty strings")
    value = _CAMEL_BOUNDARY_1.sub(r"\1_\2", key.strip())
    value = _CAMEL_BOUNDARY_2.sub(r"\1_\2", value)
    return _NON_KEY_CHARACTER.sub("_", value).strip("_").lower()


def _json_native(value, path="$"):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise StreamMetadataError(
                f"metadata at {path} must be JSON serializable with finite numbers"
            )
        return normalized
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise StreamMetadataError(
                    f"metadata at {path} must be JSON serializable with string object keys"
                )
            normalized[key] = _json_native(item, f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _json_native(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise StreamMetadataError(
        f"metadata at {path} must be JSON serializable; found {type(value).__name__}"
    )


def _control_copy(value, path):
    try:
        return copy.deepcopy(value)
    except Exception as exc:
        raise StreamMetadataError(
            f"control metadata at {path} could not be safely detached"
        ) from exc


def _canonical_model_key(raw_key):
    key = _normalize_key(raw_key)
    return MODEL_META_ALIASES.get(key, key)


def _canonical_control_key(raw_key):
    key = _normalize_key(raw_key)
    return STREAM_CONTROL_ALIASES.get(key, key)


def _canonical_target_key(raw_key):
    key = _normalize_key(raw_key)
    return TRAINING_TARGET_ALIASES.get(key, key)


def _merge_value(destination, key, value, source):
    if key in destination and destination[key] != value:
        raise StreamMetadataError(
            f"conflicting values for {key!r} while reading {source}"
        )
    destination[key] = value


def _find_model_taint(value, path):
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = _normalize_key(raw_key)
            control_key = STREAM_CONTROL_ALIASES.get(key, key)
            target_key = TRAINING_TARGET_ALIASES.get(key, key)
            compact_key = key.replace("_", "")
            if (
                key in FORBIDDEN_MODEL_META_KEYS
                or control_key in FORBIDDEN_MODEL_META_KEYS
                or target_key in FORBIDDEN_MODEL_META_KEYS
                or any(token in compact_key for token in _NESTED_TAINT_TOKENS)
            ):
                raise StreamMetadataError(
                    f"model metadata taint at {path}.{raw_key}: {key!r} is control/target-only"
                )
            _find_model_taint(item, f"{path}.{raw_key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _find_model_taint(item, f"{path}[{index}]")


def _required_identifier(value, field):
    if not isinstance(value, str) or not value.strip():
        raise StreamMetadataError(f"{field} must be a non-empty string")


def _required_sha256(value, field):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StreamMetadataError(f"{field} must be a lowercase SHA-256 digest")


def _integer(value, field, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise StreamMetadataError(f"{field} must be an integer")
    normalized = int(value)
    if nonnegative and normalized < 0:
        raise StreamMetadataError(f"{field} must be nonnegative")
    return normalized


def _positive_number(value, field):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise StreamMetadataError(f"{field} must be numeric")
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise StreamMetadataError(f"{field} must be positive and finite")


def validate_model_meta(model_meta):
    """Validate and return a detached JSON-native causal metadata mapping."""

    if not isinstance(model_meta, Mapping):
        raise StreamMetadataError("model metadata must be a mapping")

    canonical = {}
    for raw_key, raw_value in model_meta.items():
        normalized_key = _normalize_key(raw_key)
        key = MODEL_META_ALIASES.get(normalized_key, normalized_key)
        control_key = STREAM_CONTROL_ALIASES.get(normalized_key, normalized_key)
        target_key = TRAINING_TARGET_ALIASES.get(normalized_key, normalized_key)
        if key not in MODEL_META_ALLOWLIST:
            if (
                normalized_key in FORBIDDEN_MODEL_META_KEYS
                or control_key in FORBIDDEN_MODEL_META_KEYS
                or target_key in FORBIDDEN_MODEL_META_KEYS
            ):
                raise StreamMetadataError(
                    f"forbidden model metadata field {raw_key!r} is control/target-only"
                )
            raise StreamMetadataError(f"unknown model metadata field {raw_key!r}")
        _find_model_taint(raw_value, f"model_meta.{key}")
        value = _json_native(raw_value, f"model_meta.{key}")
        _merge_value(canonical, key, value, f"model metadata field {raw_key!r}")

    for field in _IDENTIFIER_FIELDS.intersection(canonical):
        _required_identifier(canonical[field], field)
    for field in _HASH_FIELDS.intersection(canonical):
        _required_sha256(canonical[field], field)
    for field in _MODEL_SCALAR_FRAMES.intersection(canonical):
        canonical[field] = _integer(canonical[field], field)
    for field in _MODEL_FRAME_SEQUENCES.intersection(canonical):
        values = canonical[field]
        if not isinstance(values, list):
            raise StreamMetadataError(f"{field} must be a sequence of frame indices")
        values = [_integer(value, field) for value in values]
        if any(right <= left for left, right in zip(values, values[1:])):
            raise StreamMetadataError(f"{field} must be strictly increasing")
        canonical[field] = values
    if "current_frame" in canonical:
        current = canonical["current_frame"]
        observed = []
        for field in ("source_frame", "max_cache_source_frame"):
            if field in canonical:
                observed.append(canonical[field])
        for field in _MODEL_FRAME_SEQUENCES:
            observed.extend(canonical.get(field, ()))
        if any(frame > current for frame in observed):
            raise StreamMetadataError(
                f"model metadata contains a future source frame beyond current_frame={current}"
            )
    if "fps" in canonical:
        _positive_number(canonical["fps"], "fps")
    for field in _POSITIVE_STRIDES.intersection(canonical):
        canonical[field] = _integer(canonical[field], field)
        if canonical[field] <= 0:
            raise StreamMetadataError(f"{field} must be positive")
    for start_field, end_field in (
        ("packet_start_frame", "packet_end_frame"),
        ("packet_start_token", "packet_end_token"),
        ("window_start_frame", "window_end_frame"),
    ):
        if start_field in canonical and end_field in canonical:
            start = _integer(canonical[start_field], start_field)
            end = _integer(canonical[end_field], end_field)
            if end < start:
                raise StreamMetadataError(f"{end_field} must not precede {start_field}")
            canonical[start_field] = start
            canonical[end_field] = end
    return canonical


def validate_training_targets(training_targets):
    if not isinstance(training_targets, Mapping):
        raise StreamMetadataError("training_targets must be a mapping")
    canonical = {}
    for raw_key, raw_value in training_targets.items():
        key = _canonical_target_key(raw_key)
        if key not in TRAINING_TARGET_ALLOWLIST:
            raise StreamMetadataError(f"unknown training target field {raw_key!r}")
        value = _control_copy(raw_value, f"training_targets.{key}")
        _merge_value(canonical, key, value, f"training target field {raw_key!r}")
    return canonical


def validate_stream_control(stream_control):
    """Validate and return detached lifecycle/control metadata."""

    if not isinstance(stream_control, Mapping):
        raise StreamMetadataError("stream control must be a mapping")
    canonical = {}
    for raw_key, raw_value in stream_control.items():
        normalized_key = _normalize_key(raw_key)
        target_key = TRAINING_TARGET_ALIASES.get(normalized_key, normalized_key)
        if target_key in TRAINING_TARGET_ALLOWLIST and normalized_key != "training_targets":
            raise StreamMetadataError(
                f"training target field {raw_key!r} must be nested under training_targets"
            )
        key = STREAM_CONTROL_ALIASES.get(normalized_key, normalized_key)
        if key not in STREAM_CONTROL_ALLOWLIST:
            raise StreamMetadataError(f"unknown stream control field {raw_key!r}")
        if key == "training_targets":
            value = validate_training_targets(raw_value)
        else:
            value = _json_native(raw_value, f"stream_control.{key}")
        _merge_value(canonical, key, value, f"stream control field {raw_key!r}")

    for field in _IDENTIFIER_FIELDS.intersection(canonical):
        _required_identifier(canonical[field], field)
    for field in _CONTROL_BOOLEAN_FIELDS.intersection(canonical):
        if not isinstance(canonical[field], bool):
            raise StreamMetadataError(f"{field} must be boolean")
    for field in _CONTROL_INTEGER_FIELDS.intersection(canonical):
        canonical[field] = _integer(canonical[field], field, nonnegative=True)
    if "duration" in canonical:
        _positive_number(canonical["duration"], "duration")
    if canonical.get("reset_stream") and canonical.get("is_video_start"):
        raise StreamMetadataError("stream start and reset_stream cannot share one packet")
    if canonical.get("reset_stream") and not canonical.get("is_video_end"):
        raise StreamMetadataError(
            "reset_stream=true is only valid at a declared stream boundary with is_video_end=true"
        )
    return canonical


def _classify_metadata(metadata):
    if not isinstance(metadata, Mapping):
        raise StreamMetadataError("metadata must be a mapping")
    model = {}
    control = {}
    targets = {}

    for raw_key, raw_value in metadata.items():
        normalized_key = _normalize_key(raw_key)
        model_key = MODEL_META_ALIASES.get(normalized_key, normalized_key)
        control_key = STREAM_CONTROL_ALIASES.get(normalized_key, normalized_key)
        target_key = TRAINING_TARGET_ALIASES.get(normalized_key, normalized_key)
        if model_key in MODEL_META_ALLOWLIST:
            value = _json_native(raw_value, f"metadata.{model_key}")
            _merge_value(model, model_key, value, f"metadata field {raw_key!r}")
        elif normalized_key == "training_targets":
            nested = validate_training_targets(raw_value)
            for key, value in nested.items():
                _merge_value(targets, key, value, "metadata.training_targets")
        elif control_key in STREAM_CONTROL_ALLOWLIST:
            value = _json_native(raw_value, f"metadata.{control_key}")
            _merge_value(control, control_key, value, f"metadata field {raw_key!r}")
        elif target_key in TRAINING_TARGET_ALLOWLIST:
            value = _control_copy(raw_value, f"metadata.training_targets.{target_key}")
            _merge_value(targets, target_key, value, f"metadata field {raw_key!r}")
        elif normalized_key in FORBIDDEN_MODEL_META_KEYS:
            raise StreamMetadataError(
                f"forbidden metadata field {raw_key!r} has no allowlisted control destination"
            )
        else:
            raise StreamMetadataError(f"unknown metadata field {raw_key!r}")
    return model, control, targets


def sanitize_stream_metadata(
    metadata,
    *,
    stream_control=None,
    training_targets=None,
):
    """Split mixed dataloader metadata into model and control-plane mappings."""

    model, control, targets = _classify_metadata(metadata)

    if stream_control is not None:
        explicit_control = validate_stream_control(stream_control)
        nested_targets = explicit_control.pop("training_targets", {})
        for key, value in explicit_control.items():
            _merge_value(control, key, value, "explicit stream_control")
        for key, value in nested_targets.items():
            _merge_value(targets, key, value, "explicit stream_control.training_targets")
    if training_targets is not None:
        for key, value in validate_training_targets(training_targets).items():
            _merge_value(targets, key, value, "explicit training_targets")

    model = validate_model_meta(model)
    if targets:
        control["training_targets"] = validate_training_targets(targets)
    control = validate_stream_control(control)
    for field in _IDENTIFIER_FIELDS.intersection(model).intersection(control):
        if model[field] != control[field]:
            raise StreamMetadataError(
                f"conflicting values for {field!r} across model and control planes"
            )
    return model, control


split_stream_metadata = sanitize_stream_metadata
MODEL_META_FORBIDDEN_KEYS = FORBIDDEN_MODEL_META_KEYS


__all__ = [
    "FORBIDDEN_MODEL_META_KEYS",
    "MODEL_META_ALIASES",
    "MODEL_META_ALLOWLIST",
    "MODEL_META_FORBIDDEN_KEYS",
    "STREAM_CONTROL_ALIASES",
    "STREAM_CONTROL_ALLOWLIST",
    "StreamMetadataError",
    "TRAINING_TARGET_ALIASES",
    "TRAINING_TARGET_ALLOWLIST",
    "sanitize_stream_metadata",
    "split_stream_metadata",
    "validate_model_meta",
    "validate_stream_control",
    "validate_training_targets",
]
