from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from opentad.utils.full_petal_data_contract import (
    ContractValidationError,
    build_fineaction_qualification_report,
    build_hardware_runtime_manifest,
    build_reporting_universe_manifest,
    build_thumos_manifest_from_annotation_subsets,
    build_thumos_manifest_from_split_files,
    build_thumos_protocol_manifest,
    canonical_json_sha256,
    compare_reporting_universe,
    load_id_file,
    load_json,
    save_json,
    validate_hardware_runtime_manifest,
    verify_content_hash,
)


CREATED_AT = "2026-07-13T08:00:00Z"
ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "tools" / "build_full_petal_manifests.py"
FEATURE_IDENTITY = {
    "encoder": "synthetic-encoder",
    "feature_dim": 8,
    "dtype": "float32",
}
EXTRACTION_IDENTITY = {
    "policy": "synthetic-causal-extraction",
    "stride": 4,
}


def _ids(prefix, count):
    return [f"{prefix}_{index:03d}" for index in range(count)]


def _write_annotation(path, train_ids, validation_ids, extra_ids=()):
    database = {
        video_id: {"subset": "train", "annotations": []}
        for video_id in train_ids
    }
    database.update(
        {
            video_id: {"subset": "validation", "annotations": []}
            for video_id in validation_ids
        }
    )
    database.update(
        {
            video_id: {"subset": "test", "annotations": []}
            for video_id in extra_ids
        }
    )
    path.write_text(json.dumps({"database": database}), encoding="utf-8")
    return path


def _metadata():
    return {
        "feature_identity": FEATURE_IDENTITY,
        "extraction_identity": EXTRACTION_IDENTITY,
        "seed": 17,
        "created_at": CREATED_AT,
    }


def _run_cli(*arguments):
    result = subprocess.run(
        [sys.executable, str(CLI), *map(str, arguments)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result


def test_strict_thumos_annotation_subsets_lock_canonical_160_40(tmp_path):
    train_ids = _ids("train", 160)
    validation_ids = _ids("validation", 40)
    annotation = _write_annotation(
        tmp_path / "annotations.json",
        reversed(train_ids),
        reversed(validation_ids),
    )

    manifest = build_thumos_manifest_from_annotation_subsets(
        annotation,
        train_subset="train",
        validation_subset="validation",
        strict=True,
        **_metadata(),
    )

    assert manifest["schema"] == "full_petal.thumos_protocol"
    assert manifest["schema_version"] == 1
    assert manifest["protocol"]["train"] == {
        "count": 160,
        "ids": sorted(train_ids),
    }
    assert manifest["protocol"]["validation"] == {
        "count": 40,
        "ids": sorted(validation_ids),
    }
    assert manifest["split_provenance"]["kind"] == "annotation_subsets"
    assert manifest["feature_identity"] == FEATURE_IDENTITY
    assert manifest["extraction_identity"] == EXTRACTION_IDENTITY
    assert manifest["seed"] == 17
    assert manifest["created_at"] == CREATED_AT
    assert len(manifest["annotation"]["sha256"]) == 64
    assert len(manifest["annotation"]["content_sha256"]) == 64
    assert verify_content_hash(manifest)


def test_strict_thumos_explicit_split_files_are_supported(tmp_path):
    train_ids = _ids("train", 160)
    validation_ids = _ids("validation", 40)
    annotation = _write_annotation(
        tmp_path / "annotations.json",
        train_ids,
        validation_ids,
        extra_ids=["test_only"],
    )
    train_split = tmp_path / "train.txt"
    validation_split = tmp_path / "validation.json"
    train_split.write_text("\n".join(reversed(train_ids)) + "\n", encoding="utf-8")
    validation_split.write_text(
        json.dumps({"ids": list(reversed(validation_ids))}),
        encoding="utf-8",
    )

    manifest = build_thumos_manifest_from_split_files(
        train_split,
        validation_split,
        annotation_path=annotation,
        strict=True,
        **_metadata(),
    )

    assert manifest["protocol"]["train"]["ids"] == sorted(train_ids)
    assert manifest["protocol"]["validation"]["ids"] == sorted(validation_ids)
    assert manifest["split_provenance"]["kind"] == "explicit_split_files"
    assert manifest["split_provenance"]["train"]["name"] == "train.txt"
    assert manifest["split_provenance"]["validation"]["name"] == "validation.json"


def test_thumos_protocol_rejects_overlap_even_without_strict_counts(tmp_path):
    train_ids = ["shared", "train_only"]
    validation_ids = ["shared", "validation_only"]
    annotation = _write_annotation(
        tmp_path / "annotations.json",
        train_ids,
        validation_ids,
    )

    with pytest.raises(ContractValidationError, match="overlap.*shared"):
        build_thumos_protocol_manifest(
            train_ids,
            validation_ids,
            annotation_path=annotation,
            split_provenance={"kind": "synthetic"},
            strict=False,
            **_metadata(),
        )


def test_thumos_protocol_rejects_duplicates_missing_ids_and_wrong_counts(tmp_path):
    train_ids = _ids("train", 160)
    validation_ids = _ids("validation", 40)
    annotation = _write_annotation(
        tmp_path / "annotations.json",
        train_ids,
        validation_ids,
    )

    with pytest.raises(ContractValidationError, match="duplicate.*train_000"):
        build_thumos_protocol_manifest(
            [*train_ids, "train_000"],
            validation_ids,
            annotation_path=annotation,
            split_provenance={"kind": "synthetic"},
            strict=False,
            **_metadata(),
        )

    with pytest.raises(ContractValidationError, match="missing from annotations.*ghost"):
        build_thumos_protocol_manifest(
            [*train_ids[:-1], "ghost"],
            validation_ids,
            annotation_path=annotation,
            split_provenance={"kind": "synthetic"},
            strict=True,
            **_metadata(),
        )

    with pytest.raises(ContractValidationError, match="train.*160.*159"):
        build_thumos_protocol_manifest(
            train_ids[:-1],
            validation_ids,
            annotation_path=annotation,
            split_provenance={"kind": "synthetic"},
            strict=True,
            **_metadata(),
        )


def test_id_file_loader_rejects_whitespace_and_duplicate_coercion(tmp_path):
    ids_path = tmp_path / "ids.txt"
    ids_path.write_text("video_a \nvideo_b\n", encoding="utf-8")
    with pytest.raises(ContractValidationError, match="whitespace.*video_a"):
        load_id_file(ids_path)

    ids_path.write_text("video_a\nvideo_a\n", encoding="utf-8")
    with pytest.raises(ContractValidationError, match="duplicate.*video_a"):
        load_id_file(ids_path)


def test_historical_211_manifest_reports_exact_diff_against_observed_213():
    locked_ids = _ids("historical", 211)
    observed_ids = [*locked_ids[1:], "extra_a", "extra_b", "extra_c"]

    locked = build_reporting_universe_manifest(
        locked_ids,
        provenance={"kind": "synthetic_historical_source"},
        seed=23,
        created_at=CREATED_AT,
        strict=True,
    )
    comparison = compare_reporting_universe(
        locked,
        observed_ids,
        observed_provenance={"kind": "synthetic_observation"},
        seed=23,
        created_at=CREATED_AT,
        strict=True,
    )

    assert locked["universe"]["count"] == 211
    assert locked["universe"]["ids"] == sorted(locked_ids)
    assert comparison["observed_universe"]["count"] == 213
    assert comparison["observed_universe"]["ids"] == sorted(observed_ids)
    assert comparison["comparison"] == {
        "matches": False,
        "missing_ids": ["historical_000"],
        "extra_ids": ["extra_a", "extra_b", "extra_c"],
    }
    assert comparison["status"] == "MISMATCH"
    assert verify_content_hash(locked)
    assert verify_content_hash(comparison)


def test_historical_reporting_counts_are_strict():
    with pytest.raises(ContractValidationError, match="211.*210"):
        build_reporting_universe_manifest(
            _ids("historical", 210),
            provenance={"kind": "synthetic_historical_source"},
            seed=23,
            created_at=CREATED_AT,
            strict=True,
        )


def test_hashes_and_saved_json_are_deterministic(tmp_path):
    left = {"z": [3, 2, 1], "a": {"right": 2, "left": 1}}
    right = {"a": {"left": 1, "right": 2}, "z": [3, 2, 1]}

    assert canonical_json_sha256(left) == canonical_json_sha256(right)

    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    save_json(first_path, left)
    save_json(second_path, right)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert b"\r\n" not in first_path.read_bytes()
    assert first_path.read_bytes().endswith(b"\n")
    assert load_json(first_path) == left


def test_hardware_runtime_manifest_validates_fixed_step_profile():
    manifest = build_hardware_runtime_manifest(
        gpu_name="Synthetic GPU",
        gpu_count=2,
        software_versions={
            "python": "3.11.7",
            "pytorch": "synthetic-version",
            "cuda": "synthetic-version",
        },
        precision="bf16",
        dimensions={
            "batch_size": 2,
            "chunk_size": 16,
            "memory_size": 64,
            "slot_count": 8,
            "memory_dim": 32,
            "slot_dim": 32,
        },
        peak_memory_bytes=123456,
        elapsed_seconds=8.0,
        steps=100,
        throughput=25.0,
        throughput_unit="samples_per_second",
        seed=17,
        created_at=CREATED_AT,
    )

    validate_hardware_runtime_manifest(manifest)
    assert manifest["schema"] == "full_petal.hardware_runtime_profile"
    assert manifest["profile_mode"] == "fixed_step"
    assert manifest["hardware"] == {"gpu_name": "Synthetic GPU", "gpu_count": 2}
    assert manifest["runtime"]["dimensions"]["slot_count"] == 8
    assert manifest["measurements"]["steps"] == 100
    assert manifest["measurements"]["throughput"] == {
        "value": 25.0,
        "unit": "samples_per_second",
    }
    assert verify_content_hash(manifest)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value["hardware"].update(gpu_count=0), "gpu_count"),
        (lambda value: value["runtime"].update(precision=""), "precision"),
        (
            lambda value: value["runtime"]["dimensions"].pop("slot_count"),
            "slot_count",
        ),
        (
            lambda value: value["measurements"].update(elapsed_seconds=float("inf")),
            "elapsed_seconds",
        ),
    ],
)
def test_hardware_runtime_validation_rejects_invalid_required_fields(mutate, message):
    manifest = build_hardware_runtime_manifest(
        gpu_name="Synthetic GPU",
        gpu_count=1,
        software_versions={"python": "3.11.7"},
        precision="fp32",
        dimensions={
            "batch_size": 1,
            "chunk_size": 8,
            "memory_size": 32,
            "slot_count": 4,
        },
        peak_memory_bytes=1024,
        elapsed_seconds=4.0,
        steps=20,
        throughput=5.0,
        throughput_unit="steps_per_second",
        seed=17,
        created_at=CREATED_AT,
    )
    invalid = deepcopy(manifest)
    mutate(invalid)

    with pytest.raises(ContractValidationError, match=message):
        validate_hardware_runtime_manifest(invalid)


def test_incomplete_fineaction_evidence_remains_unqualified():
    report = build_fineaction_qualification_report(
        {
            "protocol": {
                "split_lock": {
                    "mandatory": True,
                    "passed": True,
                    "evidence": {"manifest_sha256": "a" * 64},
                }
            },
            "completeness": {
                "media_inventory": {
                    "mandatory": True,
                    "passed": True,
                    "evidence": "",
                }
            },
        },
        seed=29,
        created_at=CREATED_AT,
    )

    assert report["schema"] == "full_petal.fineaction_qualification"
    assert report["status"] == "FAIL"
    assert report["qualified"] is False
    assert report["gates"]["protocol"]["status"] == "PASS"
    assert report["gates"]["completeness"]["status"] == "FAIL"
    assert report["gates"]["completeness"]["missing_evidence"] == [
        "media_inventory"
    ]
    assert report["gates"]["causal_readiness"]["status"] == "FAIL"
    assert "causal_readiness: no mandatory checks declared" in report["failure_reasons"]
    assert verify_content_hash(report)


def test_cli_builds_deterministic_thumos_json_from_explicit_splits(tmp_path):
    train_ids = _ids("train", 160)
    validation_ids = _ids("validation", 40)
    annotation = _write_annotation(
        tmp_path / "annotations.json",
        train_ids,
        validation_ids,
    )
    train_split = tmp_path / "train.txt"
    validation_split = tmp_path / "validation.txt"
    train_split.write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    validation_split.write_text("\n".join(validation_ids) + "\n", encoding="utf-8")
    feature_identity = tmp_path / "feature.json"
    extraction_identity = tmp_path / "extraction.json"
    feature_identity.write_text(json.dumps(FEATURE_IDENTITY), encoding="utf-8")
    extraction_identity.write_text(json.dumps(EXTRACTION_IDENTITY), encoding="utf-8")
    first_output = tmp_path / "thumos-first.json"
    second_output = tmp_path / "thumos-second.json"

    common = (
        "thumos",
        "--annotation",
        annotation,
        "--train-split",
        train_split,
        "--validation-split",
        validation_split,
        "--feature-identity",
        feature_identity,
        "--extraction-identity",
        extraction_identity,
        "--seed",
        "17",
        "--timestamp",
        CREATED_AT,
    )
    _run_cli(*common, "--output", first_output)
    _run_cli(*common, "--output", second_output)

    assert first_output.read_bytes() == second_output.read_bytes()
    assert load_json(first_output)["protocol"]["train"]["count"] == 160


def test_cli_loads_and_saves_reporting_hardware_and_fineaction_json(tmp_path):
    locked_ids = _ids("historical", 211)
    observed_ids = [*locked_ids, "extra_a", "extra_b"]
    locked_ids_path = tmp_path / "locked-ids.json"
    observed_ids_path = tmp_path / "observed-ids.json"
    locked_ids_path.write_text(json.dumps(locked_ids), encoding="utf-8")
    observed_ids_path.write_text(json.dumps(observed_ids), encoding="utf-8")
    locked_manifest = tmp_path / "locked-manifest.json"
    comparison_manifest = tmp_path / "comparison-manifest.json"

    _run_cli(
        "reporting-lock",
        "--ids",
        locked_ids_path,
        "--seed",
        "31",
        "--timestamp",
        CREATED_AT,
        "--output",
        locked_manifest,
    )
    _run_cli(
        "reporting-compare",
        "--locked-manifest",
        locked_manifest,
        "--observed-ids",
        observed_ids_path,
        "--seed",
        "31",
        "--timestamp",
        CREATED_AT,
        "--output",
        comparison_manifest,
    )
    assert load_json(comparison_manifest)["comparison"]["extra_ids"] == [
        "extra_a",
        "extra_b",
    ]

    hardware_spec = tmp_path / "hardware-spec.json"
    hardware_spec.write_text(
        json.dumps(
            {
                "gpu_name": "Synthetic GPU",
                "gpu_count": 1,
                "software_versions": {"python": "3.11.7"},
                "precision": "fp32",
                "dimensions": {
                    "batch_size": 1,
                    "chunk_size": 8,
                    "memory_size": 32,
                    "slot_count": 4,
                },
                "peak_memory_bytes": 2048,
                "elapsed_seconds": 4.0,
                "steps": 20,
                "throughput": 5.0,
                "throughput_unit": "steps_per_second",
            }
        ),
        encoding="utf-8",
    )
    hardware_manifest = tmp_path / "hardware-manifest.json"
    _run_cli(
        "hardware",
        "--input",
        hardware_spec,
        "--seed",
        "31",
        "--timestamp",
        CREATED_AT,
        "--output",
        hardware_manifest,
    )
    validate_hardware_runtime_manifest(load_json(hardware_manifest))

    fineaction_spec = tmp_path / "fineaction-spec.json"
    fineaction_spec.write_text(
        json.dumps(
            {
                "gates": {
                    "protocol": {
                        "split_lock": {
                            "passed": True,
                            "evidence": {"sha256": "a" * 64},
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    fineaction_report = tmp_path / "fineaction-report.json"
    _run_cli(
        "fineaction",
        "--input",
        fineaction_spec,
        "--seed",
        "31",
        "--timestamp",
        CREATED_AT,
        "--output",
        fineaction_report,
    )
    assert load_json(fineaction_report)["status"] == "FAIL"
