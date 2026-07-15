"""Deterministic manifests for Full PETAL data and protocol qualification."""

from collections import Counter, defaultdict
from datetime import datetime
from fractions import Fraction
import hashlib
import hmac
import json
import math
import os
from pathlib import Path

from .evidence_bundle import (
    EvidenceBundleError,
    read_stable_file_bytes,
    read_verified_bundle_bytes,
    resolve_bundle_path,
    strict_json_from_bytes,
)
from .full_petal_attestation import AttestationError, verify_payload
from .full_petal_b0 import B0EvidenceError, junit_cases


SCHEMA_VERSION = 1
THUMOS_TRAIN_COUNT = 160
THUMOS_VALIDATION_COUNT = 40
THUMOS_DEVELOPMENT_UNIVERSE_COUNT = 200
THUMOS_DEVELOPMENT_SPLIT_SEED = 20260713
THUMOS_DEVELOPMENT_SPLIT_SCHEMA_VERSION = "thumos-development-split-v1"
THUMOS_DEVELOPMENT_SPLIT_ALGORITHM_VERSION = "1.0.0"
HISTORICAL_REPORTING_COUNT = 211
OBSERVED_REPORTING_COUNT = 213
FINEACTION_MANDATORY_GATES = (
    "protocol",
    "completeness",
    "causal_readiness",
)
FINEACTION_REQUIRED_CHECKS = {
    "protocol": (
        "license",
        "official_split",
        "annotation_sha256",
        "instance_interval_ids",
    ),
    "completeness": (
        "raw_video_access",
        "same_class_overlap_pairs",
        "same_class_repeated_instances",
        "qualified_ground_truth",
        "qualified_videos",
        "estimated_decode_storage_cost",
    ),
    "causal_readiness": (
        "causal_preprocessing_contract",
        "minimal_dataset_loader_smoke",
    ),
}
FINEACTION_SOURCE_KEYS = (
    "annotation",
    "media_inventory",
    "license",
    "preprocessing",
    "loader_smoke",
)
FINEACTION_LICENSE_ROLE = "fineaction-license-authorization"
FINEACTION_PREPROCESSING_ROLE = "fineaction-causal-preprocessing-run"
FINEACTION_LOADER_ROLE = "fineaction-loader-smoke-run"
HARDWARE_REQUIRED_DIMENSIONS = (
    "batch_size",
    "chunk_size",
    "memory_size",
    "slot_count",
)


class ContractValidationError(ValueError):
    """Raised when an input cannot satisfy a locked manifest contract."""


def canonical_json_bytes(value):
    """Serialize JSON data with one stable, hashable representation."""

    try:
        serialized = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"value is not canonical JSON data: {exc}") from exc
    return serialized.encode("utf-8")


def canonical_json_sha256(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path, chunk_size=1024 * 1024):
    path = Path(path)
    if not path.is_file():
        raise ContractValidationError(f"file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value):
    raise ContractValidationError(f"non-finite JSON number is not allowed: {value}")


def load_json(path):
    path = Path(path)
    if not path.is_file():
        raise ContractValidationError(f"JSON file does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(
                handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite_json,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractValidationError(f"failed to load JSON file {path.name}: {exc}") from exc


def save_json(path, payload):
    """Atomically save deterministic, human-readable JSON with a trailing newline."""

    path = Path(path)
    try:
        serialized = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"payload is not valid JSON data: {exc}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_bytes((serialized + "\n").encode("utf-8"))
        os.replace(temporary, path)
    except OSError as exc:
        raise ContractValidationError(f"failed to save JSON file {path.name}: {exc}") from exc
    return path


def _json_clone(value, label):
    try:
        return json.loads(canonical_json_bytes(value).decode("utf-8"))
    except ContractValidationError as exc:
        raise ContractValidationError(f"{label} must contain JSON data: {exc}") from exc


def _require_mapping(value, label):
    if not isinstance(value, dict) or not value:
        raise ContractValidationError(f"{label} must be a non-empty JSON object")
    return _json_clone(value, label)


def _require_created_at(value):
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError("created_at must be an explicitly supplied timestamp")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ContractValidationError("created_at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ContractValidationError("created_at must include a timezone")
    return value


def _require_seed(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractValidationError("seed must be a non-negative integer")
    return value


def _require_nonempty_string(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{label} must be a non-empty string")
    return value


def _require_positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractValidationError(f"{label} must be a positive integer")
    return value


def _require_nonnegative_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractValidationError(f"{label} must be a non-negative integer")
    return value


def _require_positive_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{label} must be a positive finite number")
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ContractValidationError(f"{label} must be a positive finite number")
    return value


def _validated_ids(values, label):
    if isinstance(values, (str, bytes)):
        raise ContractValidationError(f"{label} IDs must be a sequence, not a string")
    try:
        ids = list(values)
    except TypeError as exc:
        raise ContractValidationError(f"{label} IDs must be iterable") from exc
    for video_id in ids:
        if not isinstance(video_id, str) or not video_id:
            raise ContractValidationError(f"{label} contains a missing or non-string ID")
        if video_id != video_id.strip():
            raise ContractValidationError(
                f"{label} ID has leading or trailing whitespace: {video_id!r}"
            )
    duplicates = sorted(
        video_id for video_id, count in Counter(ids).items() if count > 1
    )
    if duplicates:
        raise ContractValidationError(f"duplicate {label} IDs: {duplicates}")
    return ids


def _annotation_database(annotation):
    if not isinstance(annotation, dict) or not isinstance(annotation.get("database"), dict):
        raise ContractValidationError("annotation JSON must contain a database object")
    database = annotation["database"]
    _validated_ids(database.keys(), "annotation database")
    return database


def _file_record(path, content=None, *, content_bytes=None):
    path = Path(path)
    if content_bytes is not None and not isinstance(content_bytes, bytes):
        raise ContractValidationError("file record content_bytes must be bytes")
    record = {
        "name": path.name,
        "sha256": (
            hashlib.sha256(content_bytes).hexdigest()
            if content_bytes is not None
            else sha256_file(path)
        ),
        "size_bytes": (
            len(content_bytes) if content_bytes is not None else path.stat().st_size
        ),
    }
    if content is not None:
        record["content_sha256"] = canonical_json_sha256(content)
    return record


def _finalize_manifest(payload):
    finalized = _json_clone(payload, "manifest")
    finalized.pop("content_sha256", None)
    finalized["content_sha256"] = canonical_json_sha256(finalized)
    return finalized


def _finalize_manifest_sha256(payload):
    finalized = _json_clone(payload, "manifest")
    finalized.pop("manifest_sha256", None)
    finalized["manifest_sha256"] = canonical_json_sha256(finalized)
    return finalized


def verify_content_hash(manifest):
    if not isinstance(manifest, dict):
        return False
    hash_field = (
        "content_sha256"
        if "content_sha256" in manifest
        else "manifest_sha256"
    )
    expected = manifest.get(hash_field)
    if not isinstance(expected, str):
        return False
    payload = _json_clone(manifest, "manifest")
    payload.pop(hash_field, None)
    return hmac.compare_digest(expected, canonical_json_sha256(payload))


def load_id_file(path):
    """Load IDs from a newline file, JSON list, or ``{"ids": [...]}``."""

    path = Path(path)
    if not path.is_file():
        raise ContractValidationError(f"ID file does not exist: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ContractValidationError(f"failed to read ID file {path.name}: {exc}") from exc
    stripped = text.lstrip()
    if stripped.startswith(("[", "{")):
        try:
            payload = json.loads(
                text,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite_json,
            )
        except json.JSONDecodeError as exc:
            raise ContractValidationError(f"failed to parse ID file {path.name}: {exc}") from exc
        if isinstance(payload, dict):
            if set(payload) != {"ids"}:
                raise ContractValidationError(
                    f"JSON ID file {path.name} must be a list or contain only an ids field"
                )
            payload = payload["ids"]
        if not isinstance(payload, list):
            raise ContractValidationError(f"JSON ID file {path.name} must contain a list")
        return _validated_ids(payload, f"ID file {path.name}")
    ids = [line for line in text.splitlines() if line]
    return _validated_ids(ids, f"ID file {path.name}")


def save_id_file(path, ids):
    """Atomically save canonical, newline-delimited IDs."""

    path = Path(path)
    canonical_ids = sorted(_validated_ids(ids, f"ID output {path.name}"))
    serialized = "\n".join(canonical_ids) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_bytes(serialized.encode("utf-8"))
        os.replace(temporary, path)
    except OSError as exc:
        raise ContractValidationError(
            f"failed to save ID file {path.name}: {exc}"
        ) from exc
    return path


def _require_finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{label} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ContractValidationError(f"{label} must be a finite number")
    return value


def _validated_development_video(video_id, record):
    duration = _require_positive_number(
        record.get("duration"),
        f"video {video_id} duration",
    )
    duration = float(duration)
    raw_annotations = record.get("annotations")
    if not isinstance(raw_annotations, list):
        raise ContractValidationError(
            f"video {video_id} annotations must be a JSON list"
        )

    annotations = []
    for index, annotation in enumerate(raw_annotations):
        label_prefix = f"video {video_id} annotation {index}"
        if not isinstance(annotation, dict):
            raise ContractValidationError(f"{label_prefix} must be a JSON object")
        segment = annotation.get("segment")
        if not isinstance(segment, (list, tuple)) or len(segment) != 2:
            raise ContractValidationError(
                f"{label_prefix} segment must contain exactly [start, end]"
            )
        start = _require_finite_number(segment[0], f"{label_prefix} segment start")
        end = _require_finite_number(segment[1], f"{label_prefix} segment end")
        if start < 0 or end <= start or end > duration:
            raise ContractValidationError(
                f"{label_prefix} segment must satisfy 0 <= start < end <= duration"
            )
        label = _require_nonempty_string(
            annotation.get("label"),
            f"{label_prefix} label",
        )
        if label != label.strip():
            raise ContractValidationError(
                f"{label_prefix} label has leading or trailing whitespace"
            )
        annotations.append({"start": start, "end": end, "label": label})
    return {"duration": duration, "annotations": annotations}


def _development_training_videos(annotation, train_subset):
    database = _annotation_database(annotation)
    videos = {}
    for video_id, record in sorted(database.items()):
        if not isinstance(record, dict):
            raise ContractValidationError(
                f"annotation database record {video_id} must be a JSON object"
            )
        if record.get("subset") != train_subset:
            continue
        videos[video_id] = _validated_development_video(video_id, record)
    return videos


def _contains_temporal_overlap(annotations, *, same_class):
    groups = defaultdict(list)
    for annotation in annotations:
        key = annotation["label"] if same_class else None
        groups[key].append((annotation["start"], annotation["end"]))
    for intervals in groups.values():
        maximum_end = None
        for start, end in sorted(intervals):
            if maximum_end is not None and start < maximum_end:
                return True
            maximum_end = end if maximum_end is None else max(maximum_end, end)
    return False


def _crosses_chunk_boundary(start, end, chunk_duration_seconds):
    start = Fraction(str(start))
    end = Fraction(str(end))
    chunk_duration_seconds = Fraction(str(chunk_duration_seconds))
    boundary_index = start // chunk_duration_seconds + 1
    boundary = boundary_index * chunk_duration_seconds
    return start < boundary < end


def _rank_quantile_bins(videos, value_key, bin_count=4):
    ranked_ids = sorted(
        videos,
        key=lambda video_id: (videos[video_id][value_key], video_id),
    )
    universe_count = len(ranked_ids)
    return {
        video_id: f"q{min(bin_count, rank * bin_count // universe_count + 1)}"
        for rank, video_id in enumerate(ranked_ids)
    }


def _stratum_key(feature_name, value):
    encoded_value = canonical_json_bytes(value).decode("utf-8")
    return f"{feature_name}:{encoded_value}"


def _development_stratum_vectors(videos, chunk_duration, bounded_memory):
    features = {}
    for video_id, video in videos.items():
        annotations = video["annotations"]
        class_counts = Counter(row["label"] for row in annotations)
        features[video_id] = {
            "duration": video["duration"],
            "instance_count": len(annotations),
            "class_counts": class_counts,
            "any_temporal_overlap": _contains_temporal_overlap(
                annotations,
                same_class=False,
            ),
            "same_class_repetition": any(
                count > 1 for count in class_counts.values()
            ),
            "same_class_temporal_overlap": _contains_temporal_overlap(
                annotations,
                same_class=True,
            ),
            "crosses_chunk_boundary": any(
                _crosses_chunk_boundary(
                    row["start"],
                    row["end"],
                    chunk_duration,
                )
                for row in annotations
            ),
            "start_precedes_bounded_memory_at_endpoint": any(
                row["start"] < row["end"] - bounded_memory
                for row in annotations
            ),
        }

    duration_bins = _rank_quantile_bins(features, "duration")
    instance_bins = _rank_quantile_bins(features, "instance_count")
    boolean_features = (
        "any_temporal_overlap",
        "same_class_repetition",
        "same_class_temporal_overlap",
        "crosses_chunk_boundary",
        "start_precedes_bounded_memory_at_endpoint",
    )
    vectors = {}
    for video_id in sorted(features):
        feature = features[video_id]
        vector = Counter()
        for label, count in sorted(feature["class_counts"].items()):
            vector[_stratum_key("class_instance_count", label)] = count
        vector[
            _stratum_key("video_duration_quartile", duration_bins[video_id])
        ] = 1
        vector[
            _stratum_key("instances_per_video_bin", instance_bins[video_id])
        ] = 1
        for feature_name in boolean_features:
            vector[_stratum_key(feature_name, feature[feature_name])] = 1
        vectors[video_id] = dict(vector)
    return vectors


def _stratum_universe_totals(vectors):
    totals = Counter()
    for vector in vectors.values():
        totals.update(vector)
    return totals


def _seeded_video_rank(seed, video_id):
    material = f"{seed}\0{video_id}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _greedy_calibration_ids(vectors, calibration_count, seed):
    """Select whole videos by exact normalized squared-deficit minimization."""

    universe_count = len(vectors)
    universe_totals = _stratum_universe_totals(vectors)
    selected_totals = Counter()
    remaining = set(vectors)
    selected = []
    tie_ranks = {
        video_id: _seeded_video_rank(seed, video_id) for video_id in vectors
    }

    for selected_count in range(calibration_count):
        next_count = selected_count + 1
        best = None
        for video_id in sorted(remaining):
            delta = Fraction(0, 1)
            for stratum, weight in vectors[video_id].items():
                total = universe_totals[stratum]
                baseline = (
                    universe_count * selected_totals[stratum]
                    - next_count * total
                )
                with_candidate = baseline + universe_count * weight
                delta += Fraction(
                    with_candidate * with_candidate - baseline * baseline,
                    total * total,
                )
            candidate = (delta, tie_ranks[video_id], video_id)
            if best is None or candidate < best:
                best = candidate
        chosen_id = best[2]
        selected.append(chosen_id)
        selected_totals.update(vectors[chosen_id])
        remaining.remove(chosen_id)
    return sorted(selected)


def _rounded_ratio(numerator, denominator):
    return round(float(Fraction(numerator, denominator)), 12)


def _development_split_diagnostics(vectors, fit_ids, calibration_ids):
    universe_count = len(vectors)
    calibration_count = len(calibration_ids)
    universe_totals = _stratum_universe_totals(vectors)
    calibration_totals = Counter()
    for video_id in calibration_ids:
        calibration_totals.update(vectors[video_id])

    per_stratum = {}
    diagnostic_rows = []
    for stratum in sorted(universe_totals):
        universe_total = universe_totals[stratum]
        calibration_total = calibration_totals[stratum]
        fit_total = universe_total - calibration_total
        target_numerator = universe_total * calibration_count
        target = Fraction(target_numerator, universe_count)
        delta = Fraction(calibration_total, 1) - target
        absolute_delta = abs(delta)
        relative_delta = absolute_delta / target
        per_stratum[stratum] = {
            "universe": universe_total,
            "fit": fit_total,
            "calibration": calibration_total,
            "target_calibration": _rounded_ratio(
                target_numerator,
                universe_count,
            ),
            "calibration_delta": round(float(delta), 12),
            "absolute_calibration_delta": round(float(absolute_delta), 12),
        }
        diagnostic_rows.append((stratum, absolute_delta, relative_delta))

    worst = sorted(
        diagnostic_rows,
        key=lambda row: (-row[1], row[0]),
    )[:10]
    absolute_total = sum((row[1] for row in diagnostic_rows), Fraction(0, 1))
    normalized_squared_error = sum(
        (row[2] * row[2] for row in diagnostic_rows),
        Fraction(0, 1),
    )
    diagnostics = {
        "stratum_count": len(per_stratum),
        "target_calibration_fraction": _rounded_ratio(
            calibration_count,
            universe_count,
        ),
        "mean_absolute_calibration_delta": round(
            float(absolute_total / len(diagnostic_rows)),
            12,
        ),
        "max_absolute_calibration_delta": round(
            float(max(row[1] for row in diagnostic_rows)),
            12,
        ),
        "max_relative_calibration_delta": round(
            float(max(row[2] for row in diagnostic_rows)),
            12,
        ),
        "normalized_squared_error": round(float(normalized_squared_error), 12),
        "worst_strata": [
            {
                "name": stratum,
                "absolute_calibration_delta": round(float(absolute_delta), 12),
            }
            for stratum, absolute_delta, _ in worst
        ],
        "counts_are_exact": (
            len(fit_ids) == universe_count - calibration_count
            and len(calibration_ids) == calibration_count
        ),
    }
    return per_stratum, diagnostics


def build_thumos_development_split(
    annotation_path,
    *,
    train_subset,
    chunk_duration_seconds,
    bounded_memory_seconds,
    seed=THUMOS_DEVELOPMENT_SPLIT_SEED,
    strict=True,
    fit_count=THUMOS_TRAIN_COUNT,
    calibration_count=THUMOS_VALIDATION_COUNT,
):
    """Build a deterministic fit-core/development-calibration split manifest."""

    if not isinstance(strict, bool):
        raise ContractValidationError("strict must be a boolean")
    train_subset = _require_nonempty_string(train_subset, "train_subset")
    if train_subset != train_subset.strip():
        raise ContractValidationError(
            "train_subset cannot have leading or trailing whitespace"
        )
    seed = _require_seed(seed)
    fit_count = _require_positive_int(fit_count, "fit_count")
    calibration_count = _require_positive_int(
        calibration_count,
        "calibration_count",
    )
    chunk_duration = float(
        _require_positive_number(
            chunk_duration_seconds,
            "chunk_duration_seconds",
        )
    )
    bounded_memory = float(
        _require_positive_number(
            bounded_memory_seconds,
            "bounded_memory_seconds",
        )
    )

    annotation_path = Path(annotation_path)
    annotation = load_json(annotation_path)
    videos = _development_training_videos(annotation, train_subset)
    universe_count = len(videos)
    if strict and universe_count != THUMOS_DEVELOPMENT_UNIVERSE_COUNT:
        raise ContractValidationError(
            "strict THUMOS development universe requires "
            f"{THUMOS_DEVELOPMENT_UNIVERSE_COUNT} IDs, got {universe_count}"
        )
    if strict and (
        fit_count != THUMOS_TRAIN_COUNT
        or calibration_count != THUMOS_VALIDATION_COUNT
    ):
        raise ContractValidationError(
            "strict THUMOS development split requires exactly "
            f"{THUMOS_TRAIN_COUNT}/{THUMOS_VALIDATION_COUNT} fit/calibration IDs"
        )
    if fit_count + calibration_count != universe_count:
        raise ContractValidationError(
            "requested split counts must equal the selected annotation universe; "
            f"fit={fit_count}, calibration={calibration_count}, "
            f"universe={universe_count}"
        )

    vectors = _development_stratum_vectors(
        videos,
        chunk_duration,
        bounded_memory,
    )
    calibration_ids = _greedy_calibration_ids(
        vectors,
        calibration_count,
        seed,
    )
    calibration_set = set(calibration_ids)
    fit_ids = sorted(set(videos).difference(calibration_set))
    universe_ids = sorted(videos)
    per_stratum, diagnostics = _development_split_diagnostics(
        vectors,
        fit_ids,
        calibration_ids,
    )
    canonical_annotation_hash = canonical_json_sha256(annotation)
    payload = {
        "schema": "full_petal.thumos_development_split",
        "schema_version": THUMOS_DEVELOPMENT_SPLIT_SCHEMA_VERSION,
        "dataset": "THUMOS14",
        "seed": seed,
        "annotation": {
            "name": annotation_path.name,
            "sha256": canonical_annotation_hash,
            "canonical_sha256": canonical_annotation_hash,
        },
        "algorithm": {
            "name": "deterministic_greedy_group_stratification",
            "version": THUMOS_DEVELOPMENT_SPLIT_ALGORITHM_VERSION,
            "kind": "deterministic_greedy_approximation",
            "group_unit": "video_id",
            "objective": (
                "at step k choose video v minimizing sum_s "
                "((N*(C_s+w_v_s)-k*U_s)/U_s)^2 using exact rational "
                "arithmetic, where N is universe size, C_s is the selected "
                "total, w_v_s is the video weight, and U_s is universe total"
            ),
            "tie_breaking": (
                "ascending sha256(seed + NUL + video_id), then ascending video_id"
            ),
        },
        "parameters": {
            "train_subset": train_subset,
            "strict": strict,
            "fit_count": fit_count,
            "calibration_count": calibration_count,
            "chunk_duration_seconds": chunk_duration,
            "bounded_memory_seconds": bounded_memory,
            "quantile_bin_count": 4,
            "quantile_method": (
                "ascending empirical rank; ties by video_id; "
                "bin=floor(rank*4/universe)+1"
            ),
            "interval_semantics": (
                "half-open for overlap: touching endpoints do not overlap"
            ),
        },
        "universe": {
            "count": universe_count,
            "ids": universe_ids,
            "ids_sha256": canonical_json_sha256(universe_ids),
        },
        "splits": {
            "fit_core": {
                "count": len(fit_ids),
                "ids": fit_ids,
                "ids_sha256": canonical_json_sha256(fit_ids),
            },
            "calibration": {
                "count": len(calibration_ids),
                "ids": calibration_ids,
                "ids_sha256": canonical_json_sha256(calibration_ids),
            },
        },
        "stratification": {
            "definitions": {
                "class_instance_count": (
                    "integer annotation count for each label in a video; "
                    "counts are weighted stratum contributions"
                ),
                "video_duration_quartile": (
                    "four equal-frequency empirical rank bins over video duration"
                ),
                "instances_per_video_bin": (
                    "four equal-frequency empirical rank bins over annotation count"
                ),
                "any_temporal_overlap": (
                    "true iff two half-open annotation intervals overlap"
                ),
                "same_class_repetition": (
                    "true iff one label occurs at least twice in the video"
                ),
                "same_class_temporal_overlap": (
                    "true iff overlapping half-open intervals share a label"
                ),
                "crosses_chunk_boundary": (
                    "true iff start < k*chunk_duration_seconds < end for an "
                    "integer k"
                ),
                "start_precedes_bounded_memory_at_endpoint": (
                    "true iff start < end - bounded_memory_seconds for an action"
                ),
            },
            "class_labels": sorted(
                {
                    annotation["label"]
                    for video in videos.values()
                    for annotation in video["annotations"]
                }
            ),
            "per_stratum_totals": per_stratum,
        },
        "imbalance_diagnostics": diagnostics,
    }
    return _finalize_manifest_sha256(payload)


def _require_distinct_output_paths(annotation_path, output_paths):
    resolved_annotation = Path(annotation_path).resolve()
    resolved_outputs = [Path(path).resolve() for path in output_paths]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ContractValidationError("development split output paths must be distinct")
    if resolved_annotation in resolved_outputs:
        raise ContractValidationError(
            "development split output paths cannot overwrite the annotation file"
        )


def write_thumos_development_split(
    annotation_path,
    *,
    fit_ids_path,
    calibration_ids_path,
    manifest_path,
    train_subset,
    chunk_duration_seconds,
    bounded_memory_seconds,
    seed=THUMOS_DEVELOPMENT_SPLIT_SEED,
    strict=True,
    fit_count=THUMOS_TRAIN_COUNT,
    calibration_count=THUMOS_VALIDATION_COUNT,
):
    """Write canonical ID files and their path-free development manifest."""

    _require_distinct_output_paths(
        annotation_path,
        (fit_ids_path, calibration_ids_path, manifest_path),
    )
    manifest = build_thumos_development_split(
        annotation_path,
        train_subset=train_subset,
        chunk_duration_seconds=chunk_duration_seconds,
        bounded_memory_seconds=bounded_memory_seconds,
        seed=seed,
        strict=strict,
        fit_count=fit_count,
        calibration_count=calibration_count,
    )
    fit_ids = manifest["splits"]["fit_core"]["ids"]
    calibration_ids = manifest["splits"]["calibration"]["ids"]
    fit_ids_path = save_id_file(fit_ids_path, fit_ids)
    calibration_ids_path = save_id_file(calibration_ids_path, calibration_ids)

    payload = _json_clone(manifest, "development split manifest")
    payload.pop("manifest_sha256", None)
    payload["artifacts"] = {
        "fit_ids": _file_record(fit_ids_path, content=fit_ids),
        "calibration_ids": _file_record(
            calibration_ids_path,
            content=calibration_ids,
        ),
    }
    finalized = _finalize_manifest_sha256(payload)
    save_json(manifest_path, finalized)
    return finalized


def build_thumos_protocol_manifest(
    train_ids,
    validation_ids,
    *,
    annotation_path,
    split_provenance,
    feature_identity,
    extraction_identity,
    seed,
    created_at,
    strict=True,
):
    """Lock a THUMOS train/validation protocol without embedding data paths."""

    if not isinstance(strict, bool):
        raise ContractValidationError("strict must be a boolean")
    train_ids = _validated_ids(train_ids, "train")
    validation_ids = _validated_ids(validation_ids, "validation")
    overlap = sorted(set(train_ids).intersection(validation_ids))
    if overlap:
        raise ContractValidationError(f"train/validation overlap: {overlap}")
    if strict and len(train_ids) != THUMOS_TRAIN_COUNT:
        raise ContractValidationError(
            f"strict train split requires {THUMOS_TRAIN_COUNT} IDs, got {len(train_ids)}"
        )
    if strict and len(validation_ids) != THUMOS_VALIDATION_COUNT:
        raise ContractValidationError(
            "strict validation split requires "
            f"{THUMOS_VALIDATION_COUNT} IDs, got {len(validation_ids)}"
        )

    annotation_path = Path(annotation_path)
    annotation = load_json(annotation_path)
    database = _annotation_database(annotation)
    missing = sorted((set(train_ids) | set(validation_ids)).difference(database))
    if missing:
        raise ContractValidationError(f"split IDs missing from annotations: {missing}")

    canonical_train = sorted(train_ids)
    canonical_validation = sorted(validation_ids)
    payload = {
        "schema": "full_petal.thumos_protocol",
        "schema_version": SCHEMA_VERSION,
        "dataset": "THUMOS14",
        "created_at": _require_created_at(created_at),
        "seed": _require_seed(seed),
        "annotation": _file_record(annotation_path, content=annotation),
        "split_provenance": _require_mapping(split_provenance, "split_provenance"),
        "feature_identity": _require_mapping(feature_identity, "feature_identity"),
        "extraction_identity": _require_mapping(
            extraction_identity, "extraction_identity"
        ),
        "protocol": {
            "name": "canonical_160_train_40_validation",
            "strict": strict,
            "expected_counts": {
                "train": THUMOS_TRAIN_COUNT,
                "validation": THUMOS_VALIDATION_COUNT,
            },
            "train": {"count": len(canonical_train), "ids": canonical_train},
            "validation": {
                "count": len(canonical_validation),
                "ids": canonical_validation,
            },
            "universe_count": len(canonical_train) + len(canonical_validation),
            "ids_sha256": canonical_json_sha256(
                {"train": canonical_train, "validation": canonical_validation}
            ),
        },
    }
    return _finalize_manifest(payload)


def build_thumos_manifest_from_split_files(
    train_split_path,
    validation_split_path,
    *,
    annotation_path,
    feature_identity,
    extraction_identity,
    seed,
    created_at,
    strict=True,
):
    train_ids = load_id_file(train_split_path)
    validation_ids = load_id_file(validation_split_path)
    split_provenance = {
        "kind": "explicit_split_files",
        "train": _file_record(train_split_path, content=sorted(train_ids)),
        "validation": _file_record(
            validation_split_path, content=sorted(validation_ids)
        ),
    }
    return build_thumos_protocol_manifest(
        train_ids,
        validation_ids,
        annotation_path=annotation_path,
        split_provenance=split_provenance,
        feature_identity=feature_identity,
        extraction_identity=extraction_identity,
        seed=seed,
        created_at=created_at,
        strict=strict,
    )


def build_thumos_manifest_from_annotation_subsets(
    annotation_path,
    *,
    train_subset,
    validation_subset,
    feature_identity,
    extraction_identity,
    seed,
    created_at,
    strict=True,
):
    if not isinstance(train_subset, str) or not train_subset:
        raise ContractValidationError("train_subset must be a non-empty string")
    if not isinstance(validation_subset, str) or not validation_subset:
        raise ContractValidationError("validation_subset must be a non-empty string")
    annotation = load_json(annotation_path)
    database = _annotation_database(annotation)
    train_ids = [
        video_id
        for video_id, record in database.items()
        if isinstance(record, dict) and record.get("subset") == train_subset
    ]
    validation_ids = [
        video_id
        for video_id, record in database.items()
        if isinstance(record, dict) and record.get("subset") == validation_subset
    ]
    split_provenance = {
        "kind": "annotation_subsets",
        "annotation": _file_record(annotation_path, content=annotation),
        "subsets": {"train": train_subset, "validation": validation_subset},
    }
    return build_thumos_protocol_manifest(
        train_ids,
        validation_ids,
        annotation_path=annotation_path,
        split_provenance=split_provenance,
        feature_identity=feature_identity,
        extraction_identity=extraction_identity,
        seed=seed,
        created_at=created_at,
        strict=strict,
    )


def build_reporting_universe_manifest(
    locked_ids,
    *,
    provenance,
    seed,
    created_at,
    strict=True,
):
    """Create the immutable historical 211-ID reporting universe artifact."""

    if not isinstance(strict, bool):
        raise ContractValidationError("strict must be a boolean")
    locked_ids = _validated_ids(locked_ids, "historical reporting universe")
    if strict and len(locked_ids) != HISTORICAL_REPORTING_COUNT:
        raise ContractValidationError(
            "strict historical reporting universe requires "
            f"{HISTORICAL_REPORTING_COUNT} IDs, got {len(locked_ids)}"
        )
    canonical_ids = sorted(locked_ids)
    payload = {
        "schema": "full_petal.historical_reporting_universe",
        "schema_version": SCHEMA_VERSION,
        "dataset": "THUMOS14",
        "created_at": _require_created_at(created_at),
        "seed": _require_seed(seed),
        "provenance": _require_mapping(provenance, "provenance"),
        "universe": {
            "role": "historical_reporting",
            "strict": strict,
            "expected_count": HISTORICAL_REPORTING_COUNT,
            "count": len(canonical_ids),
            "ids": canonical_ids,
            "ids_sha256": canonical_json_sha256(canonical_ids),
        },
    }
    return _finalize_manifest(payload)


def _locked_reporting_ids(manifest):
    if not isinstance(manifest, dict):
        raise ContractValidationError(
            "locked reporting universe must be a historical manifest"
        )
    if manifest.get("schema") != "full_petal.historical_reporting_universe":
        raise ContractValidationError("locked reporting universe has the wrong schema")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ContractValidationError("locked reporting universe has the wrong schema version")
    if not verify_content_hash(manifest):
        raise ContractValidationError("locked reporting universe content hash is invalid")
    universe = manifest.get("universe")
    if not isinstance(universe, dict):
        raise ContractValidationError("locked reporting universe is missing universe data")
    ids = _validated_ids(universe.get("ids", ()), "historical reporting universe")
    if universe.get("count") != len(ids):
        raise ContractValidationError("locked reporting universe count does not match its IDs")
    if universe.get("ids_sha256") != canonical_json_sha256(sorted(ids)):
        raise ContractValidationError("locked reporting universe ID hash is invalid")
    return sorted(ids)


def compare_reporting_universe(
    locked_manifest,
    observed_ids,
    *,
    observed_provenance,
    difference_reasons=None,
    seed,
    created_at,
    strict=True,
):
    """Compare an observed universe without changing the historical lock."""

    if not isinstance(strict, bool):
        raise ContractValidationError("strict must be a boolean")
    locked_ids = _locked_reporting_ids(locked_manifest)
    if strict and len(locked_ids) != HISTORICAL_REPORTING_COUNT:
        raise ContractValidationError(
            "strict historical reporting universe requires "
            f"{HISTORICAL_REPORTING_COUNT} IDs, got {len(locked_ids)}"
        )
    observed_ids = _validated_ids(observed_ids, "observed reporting universe")
    if strict and len(observed_ids) != OBSERVED_REPORTING_COUNT:
        raise ContractValidationError(
            "strict observed reporting universe requires "
            f"{OBSERVED_REPORTING_COUNT} IDs, got {len(observed_ids)}"
        )

    canonical_observed = sorted(observed_ids)
    locked_set = set(locked_ids)
    observed_set = set(canonical_observed)
    missing_ids = sorted(locked_set.difference(observed_set))
    extra_ids = sorted(observed_set.difference(locked_set))
    matches = not missing_ids and not extra_ids
    difference_ids = sorted([*missing_ids, *extra_ids])
    if difference_reasons is None:
        difference_reasons = {}
    if not isinstance(difference_reasons, dict):
        raise ContractValidationError("difference_reasons must be a JSON object")
    unknown_reason_ids = sorted(set(difference_reasons).difference(difference_ids))
    if unknown_reason_ids:
        raise ContractValidationError(
            "difference reasons contain IDs outside the observed difference: "
            + ", ".join(unknown_reason_ids)
        )
    normalized_reasons = {}
    for video_id in difference_ids:
        reason = difference_reasons.get(video_id)
        if isinstance(reason, str) and reason.strip():
            normalized_reasons[video_id] = reason.strip()
    unexplained_ids = sorted(set(difference_ids).difference(normalized_reasons))
    if strict and unexplained_ids:
        raise ContractValidationError(
            "strict reporting comparison requires a difference reason for every ID; "
            "missing: " + ", ".join(unexplained_ids)
        )
    status = (
        "MATCH"
        if matches
        else "EXPLAINED_MISMATCH"
        if not unexplained_ids
        else "UNEXPLAINED_MISMATCH"
    )
    payload = {
        "schema": "full_petal.reporting_universe_comparison",
        "schema_version": SCHEMA_VERSION,
        "dataset": "THUMOS14",
        "created_at": _require_created_at(created_at),
        "seed": _require_seed(seed),
        "strict": strict,
        "historical_universe": {
            "manifest_content_sha256": locked_manifest["content_sha256"],
            "count": len(locked_ids),
            "ids_sha256": canonical_json_sha256(locked_ids),
        },
        "observed_provenance": _require_mapping(
            observed_provenance, "observed_provenance"
        ),
        "observed_universe": {
            "expected_count": OBSERVED_REPORTING_COUNT,
            "count": len(canonical_observed),
            "ids": canonical_observed,
            "ids_sha256": canonical_json_sha256(canonical_observed),
        },
        "comparison": {
            "matches": matches,
            "missing_ids": missing_ids,
            "extra_ids": extra_ids,
            "difference_reasons": normalized_reasons,
            "difference_reasons_sha256": canonical_json_sha256(
                normalized_reasons
            ),
            "unexplained_ids": unexplained_ids,
        },
        "status": status,
    }
    return _finalize_manifest(payload)


def build_id_file_provenance(path, ids):
    path = Path(path)
    return {
        "kind": "explicit_id_file",
        "source": {
            "name": path.name,
            "sha256": sha256_file(path),
            "content_sha256": canonical_json_sha256(sorted(ids)),
        },
    }


def validate_reporting_artifacts_from_sources(
    locked_manifest,
    comparison_manifest,
    *,
    locked_ids_path,
    observed_ids_path,
    difference_reasons_path,
):
    """Rebuild the 211/213 artifacts from their bound source files."""

    if not isinstance(locked_manifest, dict) or not verify_content_hash(
        locked_manifest
    ):
        raise ContractValidationError("historical reporting manifest is invalid")
    if not isinstance(comparison_manifest, dict) or not verify_content_hash(
        comparison_manifest
    ):
        raise ContractValidationError("reporting comparison manifest is invalid")
    locked_ids = load_id_file(locked_ids_path)
    observed_ids = load_id_file(observed_ids_path)
    reasons = load_json(difference_reasons_path)
    if not isinstance(reasons, dict):
        raise ContractValidationError("reporting difference reasons must be an object")
    expected_locked_provenance = build_id_file_provenance(
        locked_ids_path, locked_ids
    )
    expected_observed_provenance = build_id_file_provenance(
        observed_ids_path, observed_ids
    )
    if locked_manifest.get("provenance") != expected_locked_provenance:
        raise ContractValidationError(
            "historical reporting provenance differs from the locked ID source"
        )
    if comparison_manifest.get("observed_provenance") != expected_observed_provenance:
        raise ContractValidationError(
            "observed reporting provenance differs from the observed ID source"
        )
    rebuilt_locked = build_reporting_universe_manifest(
        locked_ids,
        provenance=expected_locked_provenance,
        seed=locked_manifest.get("seed"),
        created_at=locked_manifest.get("created_at"),
        strict=locked_manifest.get("universe", {}).get("strict"),
    )
    if rebuilt_locked != locked_manifest:
        raise ContractValidationError(
            "historical reporting manifest differs from the locked ID source"
        )
    rebuilt_comparison = compare_reporting_universe(
        rebuilt_locked,
        observed_ids,
        observed_provenance=expected_observed_provenance,
        difference_reasons=reasons,
        seed=comparison_manifest.get("seed"),
        created_at=comparison_manifest.get("created_at"),
        strict=comparison_manifest.get("strict"),
    )
    if rebuilt_comparison != comparison_manifest:
        raise ContractValidationError(
            "reporting comparison differs from source-derived evidence"
        )
    return True


def _validated_software_versions(value):
    versions = _require_mapping(value, "software_versions")
    for name, version in versions.items():
        _require_nonempty_string(name, "software version name")
        _require_nonempty_string(version, f"software version {name}")
    return versions


def _validated_dimensions(value):
    dimensions = _require_mapping(value, "dimensions")
    for name in HARDWARE_REQUIRED_DIMENSIONS:
        if name not in dimensions:
            raise ContractValidationError(f"dimensions is missing required field {name}")
    for name, dimension in dimensions.items():
        _require_nonempty_string(name, "dimension name")
        _require_positive_int(dimension, f"dimension {name}")
    return dimensions


def build_hardware_runtime_manifest(
    *,
    gpu_name,
    gpu_count,
    software_versions,
    precision,
    dimensions,
    peak_memory_bytes,
    elapsed_seconds,
    steps,
    throughput,
    throughput_unit,
    seed,
    created_at,
):
    """Build a validated fixed-step hardware/runtime profiling manifest."""

    payload = {
        "schema": "full_petal.hardware_runtime_profile",
        "schema_version": SCHEMA_VERSION,
        "profile_mode": "fixed_step",
        "created_at": _require_created_at(created_at),
        "seed": _require_seed(seed),
        "hardware": {
            "gpu_name": _require_nonempty_string(gpu_name, "gpu_name"),
            "gpu_count": _require_positive_int(gpu_count, "gpu_count"),
        },
        "software_versions": _validated_software_versions(software_versions),
        "runtime": {
            "precision": _require_nonempty_string(precision, "precision"),
            "dimensions": _validated_dimensions(dimensions),
        },
        "measurements": {
            "peak_memory_bytes": _require_nonnegative_int(
                peak_memory_bytes, "peak_memory_bytes"
            ),
            "elapsed_seconds": _require_positive_number(
                elapsed_seconds, "elapsed_seconds"
            ),
            "steps": _require_positive_int(steps, "steps"),
            "throughput": {
                "value": _require_positive_number(throughput, "throughput"),
                "unit": _require_nonempty_string(
                    throughput_unit, "throughput_unit"
                ),
            },
        },
    }
    manifest = _finalize_manifest(payload)
    validate_hardware_runtime_manifest(manifest)
    return manifest


def validate_hardware_runtime_manifest(manifest):
    if not isinstance(manifest, dict):
        raise ContractValidationError("hardware runtime manifest must be a JSON object")
    if manifest.get("schema") != "full_petal.hardware_runtime_profile":
        raise ContractValidationError("hardware runtime manifest has the wrong schema")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ContractValidationError("hardware runtime manifest has the wrong schema version")
    if manifest.get("profile_mode") != "fixed_step":
        raise ContractValidationError("profile_mode must be fixed_step")
    _require_created_at(manifest.get("created_at"))
    _require_seed(manifest.get("seed"))

    hardware = manifest.get("hardware")
    if not isinstance(hardware, dict):
        raise ContractValidationError("hardware must be a JSON object")
    _require_nonempty_string(hardware.get("gpu_name"), "gpu_name")
    _require_positive_int(hardware.get("gpu_count"), "gpu_count")
    _validated_software_versions(manifest.get("software_versions"))

    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict):
        raise ContractValidationError("runtime must be a JSON object")
    _require_nonempty_string(runtime.get("precision"), "precision")
    _validated_dimensions(runtime.get("dimensions"))

    measurements = manifest.get("measurements")
    if not isinstance(measurements, dict):
        raise ContractValidationError("measurements must be a JSON object")
    _require_nonnegative_int(
        measurements.get("peak_memory_bytes"), "peak_memory_bytes"
    )
    _require_positive_number(
        measurements.get("elapsed_seconds"), "elapsed_seconds"
    )
    _require_positive_int(measurements.get("steps"), "steps")
    throughput = measurements.get("throughput")
    if not isinstance(throughput, dict):
        raise ContractValidationError("throughput must be a JSON object")
    _require_positive_number(throughput.get("value"), "throughput")
    _require_nonempty_string(throughput.get("unit"), "throughput_unit")
    if not verify_content_hash(manifest):
        raise ContractValidationError("hardware runtime manifest content hash is invalid")
    return True


def _require_exact_json_fields(value, fields, label):
    if not isinstance(value, dict):
        raise ContractValidationError(f"{label} must be a JSON object")
    expected = set(fields)
    missing = sorted(expected.difference(value))
    extra = sorted(set(value).difference(expected))
    if missing or extra:
        raise ContractValidationError(
            f"{label} fields do not match schema; missing={missing}, extra={extra}"
        )
    return value


def _valid_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _fineaction_source_paths(sources):
    _require_exact_json_fields(sources, FINEACTION_SOURCE_KEYS, "FineAction sources")
    paths = {}
    for name in FINEACTION_SOURCE_KEYS:
        try:
            path = Path(sources[name]).expanduser().resolve(strict=True)
        except (OSError, TypeError, ValueError) as exc:
            raise ContractValidationError(
                f"FineAction {name} source is not a readable file"
            ) from exc
        if not path.is_file():
            raise ContractValidationError(f"FineAction {name} source is not a file")
        paths[name] = path
    return paths


def _fineaction_annotation_stats(annotation):
    database = _annotation_database(annotation)
    instances = []
    split_assignments = {}
    durations = {}
    for video_id in sorted(database):
        record = database[video_id]
        if not isinstance(record, dict):
            raise ContractValidationError(
                f"FineAction annotation record {video_id} must be an object"
            )
        subset = _require_nonempty_string(
            record.get("subset"), f"FineAction subset for {video_id}"
        )
        duration = _require_positive_number(
            record.get("duration"), f"FineAction duration for {video_id}"
        )
        frame_count = _require_positive_int(
            record.get("frame"), f"FineAction frame count for {video_id}"
        )
        annotations = record.get("annotations")
        if not isinstance(annotations, list):
            raise ContractValidationError(
                f"FineAction annotations for {video_id} must be a list"
            )
        split_assignments[video_id] = subset
        durations[video_id] = float(duration)
        for annotation_index, item in enumerate(annotations):
            if not isinstance(item, dict):
                raise ContractValidationError(
                    f"FineAction annotation {video_id}[{annotation_index}] must be an object"
                )
            label = _require_nonempty_string(
                item.get("label"),
                f"FineAction label for {video_id}[{annotation_index}]",
            )
            segment = item.get("segment")
            if not isinstance(segment, (list, tuple)) or len(segment) != 2:
                raise ContractValidationError(
                    f"FineAction segment for {video_id}[{annotation_index}] must have two values"
                )
            start, end = segment
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in (start, end)
            ):
                raise ContractValidationError(
                    f"FineAction segment for {video_id}[{annotation_index}] is non-finite"
                )
            start = float(start)
            end = float(end)
            if start < 0 or end <= start or end > float(duration) + 1e-6:
                raise ContractValidationError(
                    f"FineAction segment for {video_id}[{annotation_index}] is outside the video"
                )
            identity = canonical_json_sha256(
                {
                    "annotation_index": annotation_index,
                    "end": end,
                    "label": label,
                    "start": start,
                    "video_id": video_id,
                }
            )
            instances.append(
                {
                    "annotation_index": annotation_index,
                    "end": end,
                    "instance_id": identity,
                    "label": label,
                    "start": start,
                    "video_id": video_id,
                }
            )
    identities = [item["instance_id"] for item in instances]
    if len(set(identities)) != len(identities):
        raise ContractValidationError("FineAction derived instance IDs are not unique")
    return {
        "database_ids": sorted(database),
        "durations": durations,
        "instances": instances,
        "split_assignments": split_assignments,
        "split_manifest_sha256": canonical_json_sha256(split_assignments),
        "subsets": sorted(set(split_assignments.values())),
    }


def _fineaction_media_inventory(path, inventory):
    _require_exact_json_fields(
        inventory,
        {"schema", "schema_version", "dataset", "entries"},
        "FineAction media inventory",
    )
    if (
        inventory["schema"] != "full_petal.fineaction_media_inventory"
        or inventory["schema_version"] != SCHEMA_VERSION
        or inventory["dataset"] != "FineAction"
    ):
        raise ContractValidationError("FineAction media inventory identity is invalid")
    entries = inventory["entries"]
    if not isinstance(entries, list) or not entries:
        raise ContractValidationError("FineAction media inventory entries must be non-empty")
    verified = []
    seen_ids = set()
    for index, entry in enumerate(entries):
        _require_exact_json_fields(
            entry,
            {"video_id", "path", "sha256", "size_bytes"},
            f"FineAction media inventory entry {index}",
        )
        video_id = _require_nonempty_string(
            entry["video_id"], f"FineAction media video_id {index}"
        )
        if video_id != video_id.strip() or video_id in seen_ids:
            raise ContractValidationError(
                f"FineAction media inventory has an invalid or duplicate video ID: {video_id!r}"
            )
        if not _valid_sha256(entry["sha256"]):
            raise ContractValidationError(
                f"FineAction media inventory entry {video_id} has an invalid SHA-256"
            )
        _require_positive_int(
            entry["size_bytes"], f"FineAction media size for {video_id}"
        )
        try:
            media_path = resolve_bundle_path(
                entry["path"], Path(path).parent, f"FineAction media {video_id}"
            )
        except EvidenceBundleError as exc:
            raise ContractValidationError(str(exc)) from exc
        try:
            _, media_bytes = read_stable_file_bytes(
                media_path, f"FineAction media {video_id}"
            )
        except EvidenceBundleError as exc:
            raise ContractValidationError(str(exc)) from exc
        actual_size = len(media_bytes)
        if actual_size != entry["size_bytes"]:
            raise ContractValidationError(
                f"FineAction media size differs for {video_id}: "
                f"expected {entry['size_bytes']}, found {actual_size}"
            )
        actual_sha256 = hashlib.sha256(media_bytes).hexdigest()
        if actual_sha256 != entry["sha256"]:
            raise ContractValidationError(
                f"FineAction media hash differs for {video_id}"
            )
        seen_ids.add(video_id)
        verified.append(
            {
                "path": entry["path"],
                "sha256": actual_sha256,
                "size_bytes": actual_size,
                "video_id": video_id,
            }
        )
    verified.sort(key=lambda item: item["video_id"])
    return {
        "file_count": len(verified),
        "files_sha256": canonical_json_sha256(verified),
        "storage_bytes": sum(item["size_bytes"] for item in verified),
        "video_ids": [item["video_id"] for item in verified],
    }


def _fineaction_trust_roots(trust_roots):
    _require_exact_json_fields(
        trust_roots, {"license", "execution"}, "FineAction trust roots"
    )
    normalized = {}
    for role in ("license", "execution"):
        root = _require_exact_json_fields(
            trust_roots[role], {"key_id", "public_key"}, f"FineAction {role} trust root"
        )
        _require_nonempty_string(root["key_id"], f"FineAction {role} key_id")
        _require_nonempty_string(root["public_key"], f"FineAction {role} public_key")
        normalized[role] = dict(root)
    return normalized


def _fineaction_verified_leaf(reference, base_dir, label):
    try:
        return read_verified_bundle_bytes(reference, base_dir, label)
    except EvidenceBundleError as exc:
        raise ContractValidationError(str(exc)) from exc


def _fineaction_execution_leaves(source, path, label, pass_marker):
    command = source["command"]
    if not isinstance(command, list) or not command or not all(
        isinstance(item, str) and item for item in command
    ):
        raise ContractValidationError(f"{label} command is invalid")
    source_path, source_bytes = _fineaction_verified_leaf(
        source["source"], Path(path).parent, f"{label} source"
    )
    if source["source"]["path"] not in command and source_path.name not in command:
        raise ContractValidationError(f"{label} command does not execute its bound source")
    if source_path.suffix != ".py" or not source_bytes.strip():
        raise ContractValidationError(f"{label} source is not a non-empty Python file")
    try:
        compile(source_bytes, str(source_path), "exec")
    except (SyntaxError, ValueError) as exc:
        raise ContractValidationError(f"{label} source is not valid Python: {exc}") from exc
    _, junit_bytes = _fineaction_verified_leaf(
        source["junit"], Path(path).parent, f"{label} JUnit"
    )
    try:
        counts, cases = junit_cases(junit_bytes=junit_bytes)
    except B0EvidenceError as exc:
        raise ContractValidationError(f"{label} JUnit is invalid: {exc}") from exc
    if (
        counts["collected"] <= 0
        or counts["passed"] != counts["collected"]
        or counts["failed"]
        or counts["errors"]
        or counts["skipped"]
    ):
        raise ContractValidationError(f"{label} JUnit has not reached a clean PASS")
    _, log_bytes = _fineaction_verified_leaf(
        source["log"], Path(path).parent, f"{label} log"
    )
    try:
        log_text = log_bytes.decode("utf-8")
    except UnicodeError as exc:
        raise ContractValidationError(f"{label} log is not UTF-8") from exc
    if pass_marker not in log_text:
        raise ContractValidationError(f"{label} log lacks its execution PASS marker")
    return {
        "command_sha256": canonical_json_sha256(command),
        "source_sha256": source["source"]["sha256"],
        "junit_sha256": source["junit"]["sha256"],
        "log_sha256": source["log"]["sha256"],
        "tests_collected": counts["collected"],
        "tests_passed": counts["passed"],
        "testcases_sha256": canonical_json_sha256(cases),
    }


def _fineaction_license(path, trust_root, signed):
    try:
        license_source = verify_payload(
            signed,
            trust_root=trust_root,
            role=FINEACTION_LICENSE_ROLE,
        )
    except AttestationError as exc:
        raise ContractValidationError(
            f"FineAction license authorization is not trusted: {exc}"
        ) from exc
    _require_exact_json_fields(
        license_source,
        {
            "schema_version",
            "dataset",
            "license_id",
            "subject",
            "access_scope",
            "authorized",
            "issued_at",
            "terms",
        },
        "FineAction license source",
    )
    if (
        license_source["schema_version"]
        != "full-petal-fineaction-license-authorization-v1"
        or license_source["dataset"] != "FineAction"
    ):
        raise ContractValidationError("FineAction license source identity is invalid")
    _require_nonempty_string(license_source["license_id"], "FineAction license_id")
    _require_nonempty_string(license_source["subject"], "FineAction license subject")
    _require_nonempty_string(
        license_source["access_scope"], "FineAction license access scope"
    )
    _require_created_at(license_source["issued_at"])
    if license_source["authorized"] is not True:
        raise ContractValidationError("FineAction access is not authorized")
    terms = _require_exact_json_fields(
        license_source["terms"], {"path", "sha256"}, "FineAction license terms"
    )
    if not _valid_sha256(terms["sha256"]):
        raise ContractValidationError("FineAction license terms SHA-256 is invalid")
    try:
        _, terms_bytes = read_verified_bundle_bytes(
            terms, Path(path).parent, "FineAction license terms"
        )
    except EvidenceBundleError as exc:
        raise ContractValidationError(str(exc)) from exc
    return {
        "access_authorized": True,
        "access_scope": license_source["access_scope"],
        "license_id": license_source["license_id"],
        "subject": license_source["subject"],
        "terms_sha256": hashlib.sha256(terms_bytes).hexdigest(),
    }


def _fineaction_preprocessing(path, trust_root, signed):
    try:
        source = verify_payload(
            signed,
            trust_root=trust_root,
            role=FINEACTION_PREPROCESSING_ROLE,
        )
    except AttestationError as exc:
        raise ContractValidationError(
            f"FineAction preprocessing execution is not trusted: {exc}"
        ) from exc
    _require_exact_json_fields(
        source,
        {
            "schema_version",
            "dataset",
            "status",
            "command",
            "source",
            "junit",
            "log",
            "future_frames_allowed",
            "timestamp_convention",
            "frame_stride",
            "annotation_sha256",
            "media_inventory_sha256",
        },
        "FineAction preprocessing source",
    )
    if (
        source["schema_version"] != "full-petal-fineaction-preprocessing-run-v1"
        or source["dataset"] != "FineAction"
        or source["status"] != "PASS"
    ):
        raise ContractValidationError("FineAction preprocessing source identity is invalid")
    if source["future_frames_allowed"] is not False:
        raise ContractValidationError("FineAction preprocessing permits future frames")
    _require_nonempty_string(
        source["timestamp_convention"], "FineAction timestamp convention"
    )
    _require_positive_int(source["frame_stride"], "FineAction frame stride")
    for field in ("annotation_sha256", "media_inventory_sha256"):
        if not _valid_sha256(source[field]):
            raise ContractValidationError(f"FineAction {field} is invalid")
    execution = _fineaction_execution_leaves(
        source,
        path,
        "FineAction causal preprocessing",
        "FINEACTION_CAUSAL_PREPROCESS_PASS",
    )
    return {**source, "execution": execution}


def _fineaction_loader_smoke(path, trust_root, signed):
    try:
        source = verify_payload(
            signed,
            trust_root=trust_root,
            role=FINEACTION_LOADER_ROLE,
        )
    except AttestationError as exc:
        raise ContractValidationError(
            f"FineAction loader execution is not trusted: {exc}"
        ) from exc
    _require_exact_json_fields(
        source,
        {
            "schema_version",
            "dataset",
            "status",
            "command",
            "source",
            "junit",
            "log",
            "annotation_sha256",
            "media_inventory_sha256",
            "preprocessing_sha256",
        },
        "FineAction loader smoke source",
    )
    if (
        source["schema_version"] != "full-petal-fineaction-loader-run-v1"
        or source["dataset"] != "FineAction"
        or source["status"] != "PASS"
    ):
        raise ContractValidationError("FineAction loader smoke source identity is invalid")
    for field in (
        "annotation_sha256",
        "media_inventory_sha256",
        "preprocessing_sha256",
    ):
        if not _valid_sha256(source[field]):
            raise ContractValidationError(f"FineAction loader smoke {field} is invalid")
    execution = _fineaction_execution_leaves(
        source,
        path,
        "FineAction loader smoke",
        "FINEACTION_LOADER_SMOKE_PASS",
    )
    return {**source, "execution": execution}


def _fineaction_check(passed, evidence, failure):
    return {
        "mandatory": True,
        "passed": bool(passed),
        "has_evidence": True,
        "evidence_valid": True,
        "evidence_error": None,
        "failure": None if passed else failure,
        "evidence": _json_clone(evidence, "FineAction derived evidence"),
    }


def _fineaction_gate(gate_name, checks):
    required = FINEACTION_REQUIRED_CHECKS[gate_name]
    _require_exact_json_fields(checks, required, f"FineAction {gate_name} checks")
    failed = [name for name in required if checks[name]["passed"] is not True]
    return {
        "status": "PASS" if not failed else "FAIL",
        "mandatory_check_count": len(required),
        "checks": {name: checks[name] for name in required},
        "failed_mandatory_checks": failed,
        "missing_evidence": [],
        "invalid_evidence": {},
    }


def _fineaction_overlap_stats(instances):
    grouped = defaultdict(list)
    for item in instances:
        grouped[(item["video_id"], item["label"])].append(item)
    overlap_pairs = 0
    repeated_groups = 0
    repeated_instances = 0
    for group in grouped.values():
        if len(group) > 1:
            repeated_groups += 1
            repeated_instances += len(group)
        for left_index, left in enumerate(group):
            for right in group[left_index + 1 :]:
                if max(left["start"], right["start"]) < min(left["end"], right["end"]):
                    overlap_pairs += 1
    return overlap_pairs, repeated_groups, repeated_instances


def build_fineaction_qualification_report(
    sources, *, seed, created_at, trust_roots, source_bytes=None
):
    """Derive FineAction qualification by reopening and auditing source artifacts."""

    trust_roots = _fineaction_trust_roots(trust_roots)
    paths = _fineaction_source_paths(sources)
    supplied_source_bytes = source_bytes
    if supplied_source_bytes is not None:
        _require_exact_json_fields(
            supplied_source_bytes,
            FINEACTION_SOURCE_KEYS,
            "FineAction verified source bytes",
        )
        if not all(isinstance(value, bytes) for value in supplied_source_bytes.values()):
            raise ContractValidationError(
                "FineAction verified source bytes must contain exact byte strings"
            )
    source_bytes = {}
    source_payloads = {}
    for name, path in paths.items():
        try:
            if supplied_source_bytes is None:
                _, payload = read_stable_file_bytes(
                    path, f"FineAction {name} source"
                )
            else:
                payload = supplied_source_bytes[name]
            source_payloads[name] = strict_json_from_bytes(
                payload, f"FineAction {name} source", require_object=True
            )
        except EvidenceBundleError as exc:
            raise ContractValidationError(str(exc)) from exc
        source_bytes[name] = payload
    annotation = source_payloads["annotation"]
    annotation_stats = _fineaction_annotation_stats(annotation)
    media = _fineaction_media_inventory(
        paths["media_inventory"], source_payloads["media_inventory"]
    )
    license_source = _fineaction_license(
        paths["license"], trust_roots["license"], source_payloads["license"]
    )
    preprocessing = _fineaction_preprocessing(
        paths["preprocessing"],
        trust_roots["execution"],
        source_payloads["preprocessing"],
    )
    loader_smoke = _fineaction_loader_smoke(
        paths["loader_smoke"],
        trust_roots["execution"],
        source_payloads["loader_smoke"],
    )

    source_records = {
        name: _file_record(
            path,
            content=source_payloads[name],
            content_bytes=source_bytes[name],
        )
        for name, path in paths.items()
    }
    annotation_sha256 = source_records["annotation"]["sha256"]
    inventory_sha256 = source_records["media_inventory"]["sha256"]
    preprocessing_sha256 = source_records["preprocessing"]["sha256"]

    annotation_ids = set(annotation_stats["database_ids"])
    inventory_ids = set(media["video_ids"])
    unknown_media_ids = sorted(inventory_ids.difference(annotation_ids))
    qualified_video_ids = sorted(inventory_ids.intersection(annotation_ids))
    qualified_instances = [
        item
        for item in annotation_stats["instances"]
        if item["video_id"] in inventory_ids
    ]
    qualified_instance_ids = sorted(
        item["instance_id"] for item in qualified_instances
    )
    videos_with_instances = sorted(
        {item["video_id"] for item in qualified_instances}
    )
    overlap_pairs, repeated_groups, repeated_instances = _fineaction_overlap_stats(
        qualified_instances
    )
    source_video_seconds = sum(
        annotation_stats["durations"][video_id] for video_id in qualified_video_ids
    )

    preprocessing_bound = (
        preprocessing["status"] == "PASS"
        and preprocessing["future_frames_allowed"] is False
        and preprocessing["annotation_sha256"] == annotation_sha256
        and preprocessing["media_inventory_sha256"] == inventory_sha256
        and preprocessing["execution"]["tests_passed"] > 0
    )
    loader_smoke_bound = (
        loader_smoke["status"] == "PASS"
        and loader_smoke["execution"]["tests_passed"] > 0
        and loader_smoke["annotation_sha256"] == annotation_sha256
        and loader_smoke["media_inventory_sha256"] == inventory_sha256
        and loader_smoke["preprocessing_sha256"] == preprocessing_sha256
    )

    normalized_gates = {
        "protocol": _fineaction_gate(
            "protocol",
            {
                "license": _fineaction_check(
                    license_source["access_authorized"],
                    license_source,
                    "dataset access is not authorized by the bound license source",
                ),
                "official_split": _fineaction_check(
                    len(annotation_stats["subsets"]) >= 2,
                    {
                        "manifest_sha256": annotation_stats["split_manifest_sha256"],
                        "subsets": annotation_stats["subsets"],
                        "video_count": len(annotation_stats["database_ids"]),
                    },
                    "annotation does not expose at least two locked subsets",
                ),
                "annotation_sha256": _fineaction_check(
                    True,
                    {"sha256": annotation_sha256},
                    "annotation source is not bound",
                ),
                "instance_interval_ids": _fineaction_check(
                    bool(qualified_instance_ids)
                    and len(set(qualified_instance_ids)) == len(qualified_instance_ids),
                    {
                        "field": "sha256(video_id,annotation_index,label,start,end)",
                        "verified_count": len(qualified_instance_ids),
                        "ids_sha256": canonical_json_sha256(qualified_instance_ids),
                    },
                    "no unique source-derived interval identities are available",
                ),
            },
        ),
        "completeness": _fineaction_gate(
            "completeness",
            {
                "raw_video_access": _fineaction_check(
                    bool(inventory_ids) and not unknown_media_ids,
                    {
                        "inventory_sha256": inventory_sha256,
                        "file_count": media["file_count"],
                        "files_sha256": media["files_sha256"],
                        "unknown_video_ids": unknown_media_ids,
                    },
                    "media inventory is empty or contains IDs absent from annotation",
                ),
                "same_class_overlap_pairs": _fineaction_check(
                    overlap_pairs >= 20,
                    {"count": overlap_pairs},
                    "fewer than 20 same-class overlap pairs were derived",
                ),
                "same_class_repeated_instances": _fineaction_check(
                    repeated_instances > 0,
                    {"count": repeated_instances, "group_count": repeated_groups},
                    "no repeated same-class instances were derived",
                ),
                "qualified_ground_truth": _fineaction_check(
                    len(qualified_instances) >= 30,
                    {"count": len(qualified_instances)},
                    "fewer than 30 source-derived GT instances are available",
                ),
                "qualified_videos": _fineaction_check(
                    bool(videos_with_instances),
                    {
                        "count": len(videos_with_instances),
                        "ids_sha256": canonical_json_sha256(videos_with_instances),
                    },
                    "no inventoried video has a valid GT instance",
                ),
                "estimated_decode_storage_cost": _fineaction_check(
                    source_video_seconds > 0 and media["storage_bytes"] > 0,
                    {
                        "decode_gpu_hours": source_video_seconds / 3600.0,
                        "source_video_seconds": source_video_seconds,
                        "storage_bytes": media["storage_bytes"],
                        "derivation": "one_realtime_decode_pass",
                    },
                    "source duration or verified storage is zero",
                ),
            },
        ),
        "causal_readiness": _fineaction_gate(
            "causal_readiness",
            {
                "causal_preprocessing_contract": _fineaction_check(
                    preprocessing_bound,
                    {
                        "timestamp_convention": preprocessing["timestamp_convention"],
                        "future_frames_allowed": preprocessing["future_frames_allowed"],
                        "frame_stride": preprocessing["frame_stride"],
                        "manifest_sha256": preprocessing_sha256,
                    },
                    "causal preprocessing does not bind the exact annotation and media inventory",
                ),
                "minimal_dataset_loader_smoke": _fineaction_check(
                    loader_smoke_bound,
                    {
                        "status": loader_smoke["status"],
                        "command_sha256": canonical_json_sha256(loader_smoke["command"]),
                        "tests_passed": loader_smoke["execution"]["tests_passed"],
                        "junit_sha256": loader_smoke["execution"]["junit_sha256"],
                        "log_sha256": loader_smoke["execution"]["log_sha256"],
                        "source_sha256": loader_smoke["execution"]["source_sha256"],
                        "attested_run_sha256": source_records["loader_smoke"]["sha256"],
                    },
                    "loader smoke did not pass cleanly against the exact bound sources",
                ),
            },
        ),
    }
    failure_reasons = [
        f"{gate_name}: mandatory checks failed: "
        + ", ".join(gate["failed_mandatory_checks"])
        for gate_name, gate in normalized_gates.items()
        if gate["failed_mandatory_checks"]
    ]
    qualified = all(
        gate["status"] == "PASS" for gate in normalized_gates.values()
    )
    payload = {
        "schema": "full_petal.fineaction_qualification",
        "schema_version": SCHEMA_VERSION,
        "dataset": "FineAction",
        "created_at": _require_created_at(created_at),
        "seed": _require_seed(seed),
        "derivation": {
            "schema": "full_petal.fineaction_source_derivation",
            "schema_version": SCHEMA_VERSION,
            "sources": source_records,
            "qualified_video_ids_sha256": canonical_json_sha256(
                qualified_video_ids
            ),
            "qualified_instance_ids_sha256": canonical_json_sha256(
                qualified_instance_ids
            ),
        },
        "mandatory_gates": list(FINEACTION_MANDATORY_GATES),
        "gates": normalized_gates,
        "qualified": qualified,
        "status": "PASS" if qualified else "FAIL",
        "failure_reasons": failure_reasons,
    }
    return _finalize_manifest(payload)


def validate_fineaction_qualification_report(
    report, sources, *, trust_roots, source_bytes=None
):
    if not isinstance(report, dict):
        raise ContractValidationError("FineAction qualification report must be an object")
    if (
        report.get("schema") != "full_petal.fineaction_qualification"
        or report.get("schema_version") != SCHEMA_VERSION
        or not verify_content_hash(report)
    ):
        raise ContractValidationError("FineAction qualification report identity is invalid")
    rebuilt = build_fineaction_qualification_report(
        sources,
        seed=report.get("seed"),
        created_at=report.get("created_at"),
        trust_roots=trust_roots,
        source_bytes=source_bytes,
    )
    if rebuilt != report:
        raise ContractValidationError(
            "FineAction qualification report differs from source-derived evidence"
        )
    return True
