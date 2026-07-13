"""Deterministic manifests for Full PETAL data and protocol qualification."""

from collections import Counter
from datetime import datetime
import hashlib
import hmac
import json
import math
import os
from pathlib import Path


SCHEMA_VERSION = 1
THUMOS_TRAIN_COUNT = 160
THUMOS_VALIDATION_COUNT = 40
HISTORICAL_REPORTING_COUNT = 211
OBSERVED_REPORTING_COUNT = 213
FINEACTION_MANDATORY_GATES = (
    "protocol",
    "completeness",
    "causal_readiness",
)
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


def _file_record(path, content=None):
    path = Path(path)
    record = {
        "name": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if content is not None:
        record["content_sha256"] = canonical_json_sha256(content)
    return record


def _finalize_manifest(payload):
    finalized = _json_clone(payload, "manifest")
    finalized.pop("content_sha256", None)
    finalized["content_sha256"] = canonical_json_sha256(finalized)
    return finalized


def verify_content_hash(manifest):
    if not isinstance(manifest, dict):
        return False
    expected = manifest.get("content_sha256")
    if not isinstance(expected, str):
        return False
    payload = _json_clone(manifest, "manifest")
    payload.pop("content_sha256", None)
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
        },
        "status": "MATCH" if matches else "MISMATCH",
    }
    return _finalize_manifest(payload)


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


def _has_evidence(value):
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return bool(value) and any(_has_evidence(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return bool(value) and any(_has_evidence(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return False


def _build_qualification_gate(gate_name, raw_checks):
    if raw_checks is None:
        raw_checks = {}
    if not isinstance(raw_checks, dict):
        raise ContractValidationError(f"FineAction gate {gate_name} must be an object")

    checks = {}
    for check_name in sorted(raw_checks):
        _require_nonempty_string(check_name, f"{gate_name} check name")
        raw_check = raw_checks[check_name]
        if not isinstance(raw_check, dict):
            raise ContractValidationError(
                f"FineAction check {gate_name}.{check_name} must be an object"
            )
        mandatory = raw_check.get("mandatory", True)
        passed = raw_check.get("passed", False)
        if not isinstance(mandatory, bool):
            raise ContractValidationError(
                f"FineAction check {gate_name}.{check_name} mandatory must be boolean"
            )
        if not isinstance(passed, bool):
            raise ContractValidationError(
                f"FineAction check {gate_name}.{check_name} passed must be boolean"
            )
        evidence = _json_clone(
            raw_check.get("evidence"),
            f"FineAction evidence {gate_name}.{check_name}",
        )
        checks[check_name] = {
            "mandatory": mandatory,
            "passed": passed,
            "has_evidence": _has_evidence(evidence),
            "evidence": evidence,
        }

    mandatory_names = [
        name for name, check in checks.items() if check["mandatory"]
    ]
    failed = [name for name in mandatory_names if not checks[name]["passed"]]
    missing_evidence = [
        name for name in mandatory_names if not checks[name]["has_evidence"]
    ]
    passed = bool(mandatory_names) and not failed and not missing_evidence
    return {
        "status": "PASS" if passed else "FAIL",
        "mandatory_check_count": len(mandatory_names),
        "checks": checks,
        "failed_mandatory_checks": failed,
        "missing_evidence": missing_evidence,
    }


def build_fineaction_qualification_report(gates, *, seed, created_at):
    """Derive FineAction qualification solely from mandatory evidenced checks."""

    if not isinstance(gates, dict):
        raise ContractValidationError("FineAction gates must be a JSON object")
    unknown_gates = sorted(set(gates).difference(FINEACTION_MANDATORY_GATES))
    if unknown_gates:
        raise ContractValidationError(f"unknown FineAction gates: {unknown_gates}")

    normalized_gates = {
        gate_name: _build_qualification_gate(gate_name, gates.get(gate_name))
        for gate_name in FINEACTION_MANDATORY_GATES
    }
    failure_reasons = []
    for gate_name, gate in normalized_gates.items():
        if gate["mandatory_check_count"] == 0:
            failure_reasons.append(f"{gate_name}: no mandatory checks declared")
        if gate["failed_mandatory_checks"]:
            failure_reasons.append(
                f"{gate_name}: mandatory checks failed: "
                + ", ".join(gate["failed_mandatory_checks"])
            )
        if gate["missing_evidence"]:
            failure_reasons.append(
                f"{gate_name}: mandatory checks missing evidence: "
                + ", ".join(gate["missing_evidence"])
            )
    qualified = all(
        gate["status"] == "PASS" for gate in normalized_gates.values()
    )
    payload = {
        "schema": "full_petal.fineaction_qualification",
        "schema_version": SCHEMA_VERSION,
        "dataset": "FineAction",
        "created_at": _require_created_at(created_at),
        "seed": _require_seed(seed),
        "mandatory_gates": list(FINEACTION_MANDATORY_GATES),
        "gates": normalized_gates,
        "qualified": qualified,
        "status": "PASS" if qualified else "FAIL",
        "failure_reasons": failure_reasons,
    }
    return _finalize_manifest(payload)
