"""Derived R1 cache-causality audit for Prefix-Route Protocol V2."""

from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
import struct

from .evidence_bundle import (
    read_verified_bundle_bytes,
    read_verified_bundle_json,
)


R1_REQUEST_SCHEMA = "prefix-route-r1-request-v2"
R1_SUPPORT_SCHEMA = "prefix-route-r1-support-map-v2"
R1_DYNAMIC_SCHEMA = "prefix-route-r1-dynamic-audit-v2"
R1_DERIVED_SCHEMA = "prefix-route-r1-derived-certificate-v2"
R1_AUDIT_SEED = 2026071702
TOKENS_PER_CATEGORY = 32
MINIMUM_UNIQUE_VIDEOS = 16
CATEGORIES = (
    "first_token",
    "interior_token",
    "last_complete_bin",
    "last_partial_bin",
)


class PrefixRouteR1Error(ValueError):
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


def _exact(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        actual = set(value) if isinstance(value, dict) else set()
        raise PrefixRouteR1Error(
            f"{label} fields differ; missing={sorted(set(expected) - actual)}, "
            f"extra={sorted(actual - set(expected))}"
        )
    return value


def _sha(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PrefixRouteR1Error(f"{label} must be lowercase SHA-256")
    return value


def _commit(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PrefixRouteR1Error(f"{label} must be a 40-hex commit")
    return value


def _positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PrefixRouteR1Error(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PrefixRouteR1Error(f"{label} must be a non-negative integer")
    return value


def _decode_float32(value, label):
    if not isinstance(value, str) or not value:
        raise PrefixRouteR1Error(f"{label} must be non-empty base64")
    try:
        payload = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise PrefixRouteR1Error(f"{label} is not canonical base64") from exc
    if not payload or len(payload) % 4:
        raise PrefixRouteR1Error(f"{label} is not a float32 byte vector")
    values = struct.unpack("<" + "f" * (len(payload) // 4), payload)
    if any(not math.isfinite(item) for item in values):
        raise PrefixRouteR1Error(f"{label} contains non-finite float32")
    if base64.b64encode(payload).decode("ascii") != value:
        raise PrefixRouteR1Error(f"{label} base64 is not canonical")
    return payload, values


def _array_identity(array):
    import numpy as np

    array = np.ascontiguousarray(array)
    header = canonical_json_bytes(
        {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
    )
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def _token_base64(array, target_index):
    import numpy as np

    array = np.asarray(array)
    if array.ndim != 2 or not 0 <= target_index < array.shape[0]:
        raise PrefixRouteR1Error("encode_batch must return [batch, feature_dim]")
    token = np.ascontiguousarray(array[target_index], dtype="<f4")
    if not np.isfinite(token).all():
        raise PrefixRouteR1Error("encoded token contains non-finite values")
    return base64.b64encode(token.tobytes(order="C")).decode("ascii")


def run_dynamic_token_record(
    *,
    encode_batch,
    selected_rgb_frames,
    video_id,
    token_index,
    category,
):
    """Execute one equal-shape perturbation record against a batch encoder."""

    import numpy as np

    frames = np.asarray(selected_rgb_frames)
    if (
        frames.ndim != 4
        or frames.shape[-1] != 3
        or frames.dtype != np.uint8
        or frames.shape[0] < 2
    ):
        raise PrefixRouteR1Error(
            "selected RGB frames must be uint8 [tokens,H,W,3] with at least two tokens"
        )
    if not isinstance(token_index, int) or not 0 <= token_index < frames.shape[0]:
        raise PrefixRouteR1Error("dynamic target token index is invalid")
    if category not in CATEGORIES:
        raise PrefixRouteR1Error("dynamic target category is invalid")
    baseline_frames = np.ascontiguousarray(frames.copy())
    if token_index < frames.shape[0] - 1:
        future_mode = "native_future_suffix"
        future_frames = baseline_frames.copy()
        future_frames[token_index + 1 :] = 255 - future_frames[token_index + 1 :]
    else:
        future_mode = "paired_synthetic_continuation_after_eos"
        digest = hashlib.sha256(
            f"{video_id}\x1f{token_index}\x1f{R1_AUDIT_SEED}".encode("utf-8")
        ).digest()
        required = int(np.prod(frames.shape[1:]))
        repeated = (digest * ((required + len(digest) - 1) // len(digest)))[:required]
        continuation = np.frombuffer(repeated, dtype=np.uint8).reshape(
            (1, *frames.shape[1:])
        )
        baseline_frames = np.concatenate((baseline_frames, continuation), axis=0)
        future_frames = baseline_frames.copy()
        future_frames[-1] = 255 - future_frames[-1]

    other_index = 0 if token_index != 0 else 1
    other_frames = baseline_frames.copy()
    other_frames[other_index] = 255 - other_frames[other_index]
    support_frames = baseline_frames.copy()
    support_frames[token_index] = 255 - support_frames[token_index]
    order = list(reversed(range(baseline_frames.shape[0])))
    ordered_frames = np.ascontiguousarray(baseline_frames[order])
    ordered_target_index = order.index(token_index)

    baseline_tokens = np.asarray(encode_batch(baseline_frames))
    future_tokens = np.asarray(encode_batch(future_frames))
    other_tokens = np.asarray(encode_batch(other_frames))
    ordered_tokens = np.asarray(encode_batch(ordered_frames))
    support_tokens = np.asarray(encode_batch(support_frames))
    return {
        "category": category,
        "video_id": video_id,
        "token_index": token_index,
        "future_suffix_mode": future_mode,
        "baseline_input_sha256": _array_identity(baseline_frames),
        "future_mutated_input_sha256": _array_identity(future_frames),
        "other_batch_mutated_input_sha256": _array_identity(other_frames),
        "batch_order_permuted_input_sha256": _array_identity(ordered_frames),
        "support_mutated_input_sha256": _array_identity(support_frames),
        "baseline_token_f32le_base64": _token_base64(
            baseline_tokens,
            token_index,
        ),
        "future_mutated_token_f32le_base64": _token_base64(
            future_tokens,
            token_index,
        ),
        "other_batch_mutated_token_f32le_base64": _token_base64(
            other_tokens,
            token_index,
        ),
        "batch_order_permuted_token_f32le_base64": _token_base64(
            ordered_tokens,
            ordered_target_index,
        ),
        "support_mutated_token_f32le_base64": _token_base64(
            support_tokens,
            token_index,
        ),
    }


def _selection_rank(video_id, token_index, category):
    payload = (
        f"{video_id}\x1f{token_index}\x1f{R1_AUDIT_SEED}\x1f{category}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_reference(reference, bundle_root, label):
    path, payload = read_verified_bundle_bytes(reference, bundle_root, label)
    return path, payload, hashlib.sha256(payload).hexdigest()


def _read_json_reference(reference, bundle_root, label):
    path, payload, value = read_verified_bundle_json(
        reference,
        bundle_root,
        label,
        require_object=True,
    )
    return path, payload, value


def _parse_support_map(value, canonical_ids, feature_stride):
    _exact(value, {"schema_version", "feature_stride", "videos"}, "support map")
    if value["schema_version"] != R1_SUPPORT_SCHEMA:
        raise PrefixRouteR1Error("support map schema differs")
    if value["feature_stride"] != feature_stride:
        raise PrefixRouteR1Error("support map feature stride differs")
    videos = value["videos"]
    if not isinstance(videos, list):
        raise PrefixRouteR1Error("support map videos must be an array")
    by_id = {}
    all_rows = []
    for video in videos:
        _exact(video, {"video_id", "frame_count", "tokens"}, "support video")
        video_id = video["video_id"]
        if not isinstance(video_id, str) or not video_id or video_id in by_id:
            raise PrefixRouteR1Error("support video ID is invalid or duplicated")
        frame_count = _positive_int(video["frame_count"], "frame_count")
        tokens = video["tokens"]
        if not isinstance(tokens, list) or not tokens:
            raise PrefixRouteR1Error("support video must contain tokens")
        expected_source_frames = tuple(
            min(start + feature_stride, frame_count) - 1
            for start in range(0, frame_count, feature_stride)
        )
        parsed = []
        for token_index, row in enumerate(tokens):
            _exact(
                row,
                {
                    "token_index",
                    "source_frame",
                    "decision_frame",
                    "support_frames",
                },
                "support token",
            )
            if row["token_index"] != token_index:
                raise PrefixRouteR1Error("support token indexes are not contiguous")
            source = _nonnegative_int(row["source_frame"], "source_frame")
            decision = _nonnegative_int(row["decision_frame"], "decision_frame")
            support = row["support_frames"]
            if (
                not isinstance(support, list)
                or not support
                or any(
                    isinstance(item, bool) or not isinstance(item, int) or item < 0
                    for item in support
                )
            ):
                raise PrefixRouteR1Error("support_frames must be non-negative integers")
            if source != expected_source_frames[token_index]:
                raise PrefixRouteR1Error("source frame differs from packet-recent policy")
            if decision != source:
                raise PrefixRouteR1Error("decision frame differs from source frame")
            if support != [decision]:
                raise PrefixRouteR1Error(
                    "current single-frame path support must equal [decision_frame]"
                )
            parsed_row = {
                "video_id": video_id,
                "frame_count": frame_count,
                **row,
            }
            parsed.append(parsed_row)
            all_rows.append(parsed_row)
        by_id[video_id] = {
            "frame_count": frame_count,
            "tokens": parsed,
            "source_frames": list(expected_source_frames),
        }
    if sorted(by_id) != list(canonical_ids):
        raise PrefixRouteR1Error("support map does not cover canonical population")
    return by_id, all_rows


def _category_candidates(support_by_id):
    candidates = {category: [] for category in CATEGORIES}
    for video_id, video in sorted(support_by_id.items()):
        tokens = video["tokens"]
        last = len(tokens) - 1
        candidates["first_token"].append((video_id, 0))
        candidates["interior_token"].extend(
            (video_id, index) for index in range(1, last)
        )
        category = (
            "last_complete_bin"
            if video["frame_count"] % 8 == 0
            else "last_partial_bin"
        )
        candidates[category].append((video_id, last))
    selected = []
    shortfalls = {}
    for category in CATEGORIES:
        ordered = sorted(
            candidates[category],
            key=lambda key: (
                _selection_rank(key[0], key[1], category),
                key[0],
                key[1],
            ),
        )
        chosen = ordered[:TOKENS_PER_CATEGORY]
        shortfalls[category] = max(0, TOKENS_PER_CATEGORY - len(chosen))
        selected.extend(
            {
                "category": category,
                "video_id": video_id,
                "token_index": token_index,
            }
            for video_id, token_index in chosen
        )
    return selected, shortfalls


def _validate_dynamic(value, expected_selection, support_by_id):
    _exact(value, {"schema_version", "audit_seed", "records"}, "dynamic audit")
    if value["schema_version"] != R1_DYNAMIC_SCHEMA:
        raise PrefixRouteR1Error("dynamic audit schema differs")
    if value["audit_seed"] != R1_AUDIT_SEED:
        raise PrefixRouteR1Error("dynamic audit seed differs")
    records = value["records"]
    if not isinstance(records, list):
        raise PrefixRouteR1Error("dynamic records must be an array")
    expected_keys = [
        (row["category"], row["video_id"], row["token_index"])
        for row in expected_selection
    ]
    actual_keys = []
    support_dimensions = set()
    for record in records:
        _exact(
            record,
            {
                "category",
                "video_id",
                "token_index",
                "future_suffix_mode",
                "baseline_input_sha256",
                "future_mutated_input_sha256",
                "other_batch_mutated_input_sha256",
                "batch_order_permuted_input_sha256",
                "support_mutated_input_sha256",
                "baseline_token_f32le_base64",
                "future_mutated_token_f32le_base64",
                "other_batch_mutated_token_f32le_base64",
                "batch_order_permuted_token_f32le_base64",
                "support_mutated_token_f32le_base64",
            },
            "dynamic record",
        )
        key = (record["category"], record["video_id"], record["token_index"])
        actual_keys.append(key)
        is_last = (
            record["token_index"]
            == len(support_by_id[record["video_id"]]["tokens"]) - 1
        )
        expected_future_mode = (
            "paired_synthetic_continuation_after_eos"
            if is_last
            else "native_future_suffix"
        )
        if record["future_suffix_mode"] != expected_future_mode:
            raise PrefixRouteR1Error("future suffix audit mode differs")
        for field in (
            "baseline_input_sha256",
            "future_mutated_input_sha256",
            "other_batch_mutated_input_sha256",
            "batch_order_permuted_input_sha256",
            "support_mutated_input_sha256",
        ):
            _sha(record[field], field)
        baseline_bytes, baseline = _decode_float32(
            record["baseline_token_f32le_base64"],
            "baseline token",
        )
        future_bytes, _ = _decode_float32(
            record["future_mutated_token_f32le_base64"],
            "future-mutated token",
        )
        other_bytes, _ = _decode_float32(
            record["other_batch_mutated_token_f32le_base64"],
            "other-batch token",
        )
        order_bytes, _ = _decode_float32(
            record["batch_order_permuted_token_f32le_base64"],
            "batch-order token",
        )
        support_bytes, support = _decode_float32(
            record["support_mutated_token_f32le_base64"],
            "support-mutated token",
        )
        support_dimensions.add(len(baseline))
        if not (
            baseline_bytes == future_bytes == other_bytes == order_bytes
        ):
            raise PrefixRouteR1Error("dynamic invariance token bytes differ")
        if support_bytes == baseline_bytes:
            raise PrefixRouteR1Error("support mutation did not change token bytes")
        if len(support) != len(baseline):
            raise PrefixRouteR1Error("support mutation changed token dimension")
        linf = max(abs(left - right) for left, right in zip(baseline, support))
        if not linf > 0.0:
            raise PrefixRouteR1Error("support mutation has zero L-infinity effect")
        if record["future_mutated_input_sha256"] == record["baseline_input_sha256"]:
            raise PrefixRouteR1Error("future input mutation is a no-op")
        if (
            record["other_batch_mutated_input_sha256"]
            == record["baseline_input_sha256"]
        ):
            raise PrefixRouteR1Error("other-batch input mutation is a no-op")
        if (
            record["batch_order_permuted_input_sha256"]
            == record["baseline_input_sha256"]
        ):
            raise PrefixRouteR1Error("batch-order input mutation is a no-op")
        if record["support_mutated_input_sha256"] == record["baseline_input_sha256"]:
            raise PrefixRouteR1Error("support input mutation is a no-op")
    if actual_keys != expected_keys:
        raise PrefixRouteR1Error("dynamic audit selection differs from frozen ranking")
    if len(support_dimensions) != 1:
        raise PrefixRouteR1Error("dynamic token feature dimensions differ")
    return {
        "record_count": len(records),
        "feature_dim": next(iter(support_dimensions), None),
        "future_mutation_invariant": True,
        "other_batch_image_invariant": True,
        "batch_order_invariant": True,
        "support_frame_sensitive": True,
    }


def validate_r1_bundle(
    request,
    *,
    bundle_root,
    protocol_id,
    protocol_sha256,
    review_attestation_sha256,
    canonical_video_ids,
):
    """Re-read every R1 source and derive the existing-cache disposition."""

    required = {
        "schema_version",
        "protocol_id",
        "protocol_sha256",
        "review_attestation_sha256",
        "repository_commit",
        "extractor_source",
        "resolved_command",
        "environment_lock",
        "software_versions",
        "hf_snapshot_revision",
        "model_config",
        "processor_config",
        "weight_shards",
        "annotation",
        "cache_manifest",
        "support_map",
        "dynamic_audit",
        "raw_videos",
        "feature_arrays",
    }
    _exact(request, required, "R1 request")
    if request["schema_version"] != R1_REQUEST_SCHEMA:
        raise PrefixRouteR1Error("R1 request schema differs")
    if request["protocol_id"] != protocol_id:
        raise PrefixRouteR1Error("R1 protocol ID differs")
    if request["protocol_sha256"] != protocol_sha256:
        raise PrefixRouteR1Error("R1 protocol hash differs")
    if request["review_attestation_sha256"] != review_attestation_sha256:
        raise PrefixRouteR1Error("R1 review attestation hash differs")
    _commit(request["repository_commit"], "R1 repository commit")
    if (
        not isinstance(request["hf_snapshot_revision"], str)
        or not request["hf_snapshot_revision"]
    ):
        raise PrefixRouteR1Error("HF snapshot revision is missing")
    root = Path(bundle_root)
    source_digests = {}
    for field in (
        "extractor_source",
        "resolved_command",
        "environment_lock",
        "software_versions",
        "model_config",
        "processor_config",
        "annotation",
        "cache_manifest",
        "support_map",
        "dynamic_audit",
    ):
        _, _, digest = _read_reference(request[field], root, f"R1 {field}")
        source_digests[field] = digest
    weights = request["weight_shards"]
    if not isinstance(weights, list) or not weights:
        raise PrefixRouteR1Error("R1 weight shard references are missing")
    weight_digests = [
        _read_reference(reference, root, f"weight shard {index}")[2]
        for index, reference in enumerate(weights)
    ]

    canonical_ids = tuple(canonical_video_ids)
    if (
        list(canonical_ids) != sorted(canonical_ids)
        or len(canonical_ids) != len(set(canonical_ids))
    ):
        raise PrefixRouteR1Error("canonical video IDs must be sorted and unique")
    raw_refs = request["raw_videos"]
    feature_refs = request["feature_arrays"]
    if not isinstance(raw_refs, dict) or sorted(raw_refs) != list(canonical_ids):
        raise PrefixRouteR1Error("raw-video references do not cover canonical IDs")
    if not isinstance(feature_refs, dict) or sorted(feature_refs) != list(canonical_ids):
        raise PrefixRouteR1Error("feature references do not cover canonical IDs")

    _, annotation_bytes, _ = _read_reference(
        request["annotation"],
        root,
        "R1 annotation",
    )
    _, _, manifest = _read_json_reference(
        request["cache_manifest"],
        root,
        "R1 cache manifest",
    )
    _exact(
        manifest,
        {
            "schema",
            "created_at",
            "annotation_file",
            "annotation_sha256",
            "encoder_id",
            "feature_policy",
            "timestamp_convention",
            "feature_stride",
            "feature_dim",
            "dtype",
            "videos",
        },
        "cache manifest",
    )
    if manifest["schema"] != "ontad_feature_cache_v1":
        raise PrefixRouteR1Error("cache manifest schema differs")
    if manifest["annotation_sha256"] != hashlib.sha256(annotation_bytes).hexdigest():
        raise PrefixRouteR1Error("cache manifest annotation hash differs")
    if manifest["feature_policy"] != "packet_recent_frame":
        raise PrefixRouteR1Error("cache feature policy differs")
    if manifest["timestamp_convention"] != "zero_based_source_frame":
        raise PrefixRouteR1Error("cache timestamp convention differs")
    feature_stride = _positive_int(manifest["feature_stride"], "feature_stride")
    if feature_stride != 8:
        raise PrefixRouteR1Error("feature stride must equal eight")
    feature_dim = _positive_int(manifest["feature_dim"], "feature_dim")
    if sorted(manifest["videos"]) != list(canonical_ids):
        raise PrefixRouteR1Error("cache manifest does not cover canonical IDs")

    _, _, support_value = _read_json_reference(
        request["support_map"],
        root,
        "R1 support map",
    )
    support_by_id, _ = _parse_support_map(
        support_value,
        canonical_ids,
        feature_stride,
    )
    raw_digests = {}
    feature_digests = {}
    for video_id in canonical_ids:
        raw_path, _, raw_digest = _read_reference(
            raw_refs[video_id],
            root,
            f"raw video {video_id}",
        )
        if raw_path.stem != video_id:
            raise PrefixRouteR1Error("raw video basename differs from video ID")
        raw_digests[video_id] = raw_digest
        feature_path, _, feature_digest = _read_reference(
            feature_refs[video_id],
            root,
            f"feature array {video_id}",
        )
        metadata = manifest["videos"][video_id]
        _exact(
            metadata,
            {
                "file",
                "num_tokens",
                "feature_dim",
                "dtype",
                "source_frames",
                "sha256",
            },
            f"cache metadata {video_id}",
        )
        if feature_path.name != metadata["file"]:
            raise PrefixRouteR1Error("feature basename differs from cache manifest")
        if feature_digest != metadata["sha256"]:
            raise PrefixRouteR1Error("feature hash differs from cache manifest")
        if metadata["source_frames"] != support_by_id[video_id]["source_frames"]:
            raise PrefixRouteR1Error("feature source frames differ from support map")
        try:
            import numpy as np

            array = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        except Exception as exc:
            raise PrefixRouteR1Error("failed to load feature array") from exc
        if (
            array.ndim != 2
            or list(array.shape)
            != [metadata["num_tokens"], metadata["feature_dim"]]
            or int(array.shape[1]) != feature_dim
            or str(array.dtype) != metadata["dtype"]
        ):
            raise PrefixRouteR1Error("feature array geometry or dtype differs")
        feature_digests[video_id] = feature_digest

    expected_selection, shortfalls = _category_candidates(support_by_id)
    unique_videos = {
        row["video_id"] for row in expected_selection
    }
    if len(canonical_ids) >= MINIMUM_UNIQUE_VIDEOS and (
        len(unique_videos) < MINIMUM_UNIQUE_VIDEOS
    ):
        raise PrefixRouteR1Error("dynamic audit has too few unique videos")
    _, _, dynamic_value = _read_json_reference(
        request["dynamic_audit"],
        root,
        "R1 dynamic audit",
    )
    dynamic = _validate_dynamic(
        dynamic_value,
        expected_selection,
        support_by_id,
    )
    if dynamic["feature_dim"] != feature_dim:
        raise PrefixRouteR1Error("dynamic token dimension differs from cache")
    return {
        "schema_version": R1_DERIVED_SCHEMA,
        "protocol_id": protocol_id,
        "protocol_sha256": protocol_sha256,
        "review_attestation_sha256": review_attestation_sha256,
        "repository_commit": request["repository_commit"],
        "claim_scope": "decoded_rgb_frames_to_feature_token_only",
        "compressed_bitstream_scope": "EXCLUDED",
        "source_digests": source_digests,
        "weight_shard_sha256": weight_digests,
        "hf_snapshot_revision": request["hf_snapshot_revision"],
        "raw_video_sha256_by_video": raw_digests,
        "feature_array_sha256_by_video": feature_digests,
        "static_support_audit": {
            "video_count": len(support_by_id),
            "all_support_le_decision": True,
            "all_support_equals_decision": True,
            "source_frame_equals_decision_frame": True,
        },
        "dynamic_perturbation_audit": {
            **dynamic,
            "category_shortfalls": shortfalls,
            "unique_video_count": len(unique_videos),
        },
        "existing_cache_linkage": {
            "exact_weight_snapshot_bound": True,
            "raw_video_hashes_bound": True,
            "extraction_environment_bound": True,
            "feature_arrays_bound": True,
            "cache_key_sets_equal": (
                sorted(raw_digests)
                == sorted(feature_digests)
                == sorted(support_by_id)
            ),
        },
        "status": "PASS_R1_EXISTING_CACHE_CERTIFIED",
    }


def derive_r1_status(*args, **kwargs):
    """Return a recordable failure instead of allowing partial PASS evidence."""

    try:
        return validate_r1_bundle(*args, **kwargs)
    except Exception as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        return {
            "schema_version": R1_DERIVED_SCHEMA,
            "status": "FAIL_UNVERIFIABLE",
            "failure_type": type(exc).__name__,
            "failure_reasons": [str(exc)],
        }


__all__ = [
    "CATEGORIES",
    "MINIMUM_UNIQUE_VIDEOS",
    "R1_AUDIT_SEED",
    "R1_DERIVED_SCHEMA",
    "R1_DYNAMIC_SCHEMA",
    "R1_REQUEST_SCHEMA",
    "R1_SUPPORT_SCHEMA",
    "TOKENS_PER_CATEGORY",
    "PrefixRouteR1Error",
    "canonical_json_bytes",
    "derive_r1_status",
    "run_dynamic_token_record",
    "validate_r1_bundle",
]
