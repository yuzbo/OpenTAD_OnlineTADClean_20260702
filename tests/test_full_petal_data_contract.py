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
    build_thumos_development_split,
    build_thumos_manifest_from_annotation_subsets,
    build_thumos_manifest_from_split_files,
    build_thumos_protocol_manifest,
    canonical_json_sha256,
    compare_reporting_universe,
    load_id_file,
    load_json,
    save_json,
    sha256_file,
    validate_hardware_runtime_manifest,
    verify_content_hash,
    _crosses_chunk_boundary,
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
DEVELOPMENT_SPLIT_SEED = 20260713


def test_chunk_boundary_uses_exact_decimal_arithmetic():
    assert _crosses_chunk_boundary(0.3, 0.31, 0.1) is False
    assert _crosses_chunk_boundary(0.29, 0.31, 0.1) is True


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


def _development_annotation_payload(count=200, *, tie_rich=False, reverse=False):
    database = {}
    indices = range(count - 1, -1, -1) if reverse else range(count)
    for index in indices:
        video_id = f"training_{index:03d}"
        if tie_rich:
            duration = 64.0
            annotations = [{"segment": [4.0, 8.0], "label": "Action"}]
        else:
            duration = float(40 + (index % 20) * 3)
            label = f"Class_{index % 5}"
            annotations = [
                {
                    "segment": [float(2 + index % 5), float(8 + index % 5)],
                    "label": label,
                }
            ]
            if index % 3 == 0:
                annotations.append(
                    {"segment": [6.0, 12.0], "label": f"Class_{(index + 1) % 5}"}
                )
            if index % 5 == 0:
                annotations.append({"segment": [20.0, 24.0], "label": label})
            if index % 7 == 0:
                annotations.append({"segment": [4.0, 10.0], "label": label})
            if index % 11 == 0:
                annotations.append({"segment": [15.0, 17.0], "label": "Cross"})
            if index % 13 == 0:
                annotations.append({"segment": [1.0, 30.0], "label": "Long"})
        database[video_id] = {
            "subset": "training",
            "duration": duration,
            "annotations": annotations,
        }
    return {"database": database}


def _write_development_annotation(path, **kwargs):
    path.write_text(
        json.dumps(_development_annotation_payload(**kwargs)),
        encoding="utf-8",
    )
    return path


def _build_development_split(annotation, **overrides):
    arguments = {
        "train_subset": "training",
        "chunk_duration_seconds": 16.0,
        "bounded_memory_seconds": 24.0,
        "seed": DEVELOPMENT_SPLIT_SEED,
        "strict": True,
    }
    arguments.update(overrides)
    return build_thumos_development_split(annotation, **arguments)


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


def test_thumos_development_split_is_exact_stratified_and_hashed(tmp_path):
    annotation = _write_development_annotation(tmp_path / "annotations.json")

    manifest = _build_development_split(annotation)

    assert manifest["schema"] == "full_petal.thumos_development_split"
    assert manifest["schema_version"] == "thumos-development-split-v1"
    assert manifest["seed"] == DEVELOPMENT_SPLIT_SEED
    assert manifest["algorithm"]["name"] == "deterministic_greedy_group_stratification"
    assert manifest["algorithm"]["version"]
    assert manifest["parameters"]["train_subset"] == "training"
    assert manifest["parameters"]["chunk_duration_seconds"] == 16.0
    assert manifest["parameters"]["bounded_memory_seconds"] == 24.0
    assert manifest["universe"]["count"] == 200
    assert manifest["splits"]["fit_core"]["count"] == 160
    assert manifest["splits"]["calibration"]["count"] == 40

    fit_ids = manifest["splits"]["fit_core"]["ids"]
    calibration_ids = manifest["splits"]["calibration"]["ids"]
    assert fit_ids == sorted(fit_ids)
    assert calibration_ids == sorted(calibration_ids)
    assert not set(fit_ids).intersection(calibration_ids)
    assert sorted([*fit_ids, *calibration_ids]) == manifest["universe"]["ids"]

    required_strata = {
        "class_instance_count",
        "video_duration_quartile",
        "instances_per_video_bin",
        "any_temporal_overlap",
        "same_class_repetition",
        "same_class_temporal_overlap",
        "crosses_chunk_boundary",
        "start_precedes_bounded_memory_at_endpoint",
    }
    assert set(manifest["stratification"]["definitions"]) == required_strata
    totals = manifest["stratification"]["per_stratum_totals"]
    assert totals
    assert all(
        row["universe"] == row["fit"] + row["calibration"]
        for row in totals.values()
    )
    for stratum in required_strata:
        assert any(name.startswith(f"{stratum}:") for name in totals)
    assert manifest["imbalance_diagnostics"]["stratum_count"] == len(totals)
    assert len(manifest["annotation"]["canonical_sha256"]) == 64
    assert len(manifest["manifest_sha256"]) == 64
    assert verify_content_hash(manifest)


def test_thumos_development_split_ignores_annotation_dictionary_order(tmp_path):
    annotation = _write_development_annotation(tmp_path / "annotations.json")
    first = _build_development_split(annotation)

    _write_development_annotation(annotation, reverse=True)
    second = _build_development_split(annotation)

    assert first == second


def test_thumos_development_split_seed_is_reproducible_and_breaks_ties(tmp_path):
    annotation = _write_development_annotation(
        tmp_path / "tie-rich.json",
        tie_rich=True,
    )

    first = _build_development_split(annotation)
    repeated = _build_development_split(annotation)
    alternate = _build_development_split(annotation, seed=DEVELOPMENT_SPLIT_SEED + 1)

    assert first == repeated
    assert first["splits"]["calibration"]["ids"] != alternate["splits"][
        "calibration"
    ]["ids"]
    for manifest in (first, alternate):
        assert manifest["splits"]["fit_core"]["count"] == 160
        assert manifest["splits"]["calibration"]["count"] == 40
        assert verify_content_hash(manifest)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["database"]["training_000"].update(
                duration=0.0
            ),
            "duration",
        ),
        (
            lambda payload: payload["database"]["training_000"]["annotations"][
                0
            ].update(segment=[4.0, 4.0]),
            "segment",
        ),
        (
            lambda payload: payload["database"]["training_000"]["annotations"][
                0
            ].pop("label"),
            "label",
        ),
        (
            lambda payload: payload["database"]["training_000"].update(
                duration=float("inf")
            ),
            "non-finite",
        ),
    ],
)
def test_thumos_development_split_rejects_malformed_annotations(
    tmp_path,
    mutate,
    message,
):
    payload = _development_annotation_payload()
    mutate(payload)
    annotation = tmp_path / "malformed.json"
    annotation.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ContractValidationError, match=message):
        _build_development_split(annotation)


def test_thumos_development_split_rejects_non_200_duplicate_and_impossible_counts(
    tmp_path,
):
    short_annotation = _write_development_annotation(
        tmp_path / "short.json",
        count=199,
    )
    with pytest.raises(ContractValidationError, match="strict.*200.*199"):
        _build_development_split(short_annotation)

    duplicate_annotation = tmp_path / "duplicate.json"
    record = '{"subset":"training","duration":10.0,"annotations":[]}'
    duplicate_annotation.write_text(
        '{"database":{"duplicate":' + record + ',"duplicate":' + record + "}}",
        encoding="utf-8",
    )
    with pytest.raises(ContractValidationError, match="duplicate JSON key.*duplicate"):
        _build_development_split(duplicate_annotation)

    annotation = _write_development_annotation(tmp_path / "annotations.json")
    with pytest.raises(ContractValidationError, match="split counts.*universe"):
        _build_development_split(
            annotation,
            strict=False,
            fit_count=159,
            calibration_count=40,
        )


def test_cli_writes_thumos_development_ids_and_path_free_manifest(tmp_path):
    annotation = _write_development_annotation(tmp_path / "annotations.json")
    fit_output = tmp_path / "outputs" / "thumos_fit_core_160.txt"
    calibration_output = tmp_path / "outputs" / "thumos_calibration_40.txt"
    manifest_output = tmp_path / "outputs" / "thumos_development_split.json"

    _run_cli(
        "thumos-development-split",
        "--annotation",
        annotation,
        "--train-subset",
        "training",
        "--chunk-duration-seconds",
        "16",
        "--bounded-memory-seconds",
        "24",
        "--fit-output",
        fit_output,
        "--calibration-output",
        calibration_output,
        "--seed",
        str(DEVELOPMENT_SPLIT_SEED),
        "--output",
        manifest_output,
    )

    manifest = load_json(manifest_output)
    assert load_id_file(fit_output) == manifest["splits"]["fit_core"]["ids"]
    assert load_id_file(calibration_output) == manifest["splits"]["calibration"][
        "ids"
    ]
    assert manifest["artifacts"]["fit_ids"]["name"] == fit_output.name
    assert manifest["artifacts"]["calibration_ids"]["name"] == (
        calibration_output.name
    )
    assert str(tmp_path) not in manifest_output.read_text(encoding="utf-8")
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
        difference_reasons={
            "historical_000": "synthetic canonical removal",
            "extra_a": "synthetic historical exclusion",
            "extra_b": "synthetic historical exclusion",
            "extra_c": "synthetic historical exclusion",
        },
        seed=23,
        created_at=CREATED_AT,
        strict=True,
    )

    assert locked["universe"]["count"] == 211
    assert locked["universe"]["ids"] == sorted(locked_ids)
    assert comparison["observed_universe"]["count"] == 213
    assert comparison["observed_universe"]["ids"] == sorted(observed_ids)
    assert comparison["comparison"]["matches"] is False
    assert comparison["comparison"]["missing_ids"] == ["historical_000"]
    assert comparison["comparison"]["extra_ids"] == [
        "extra_a",
        "extra_b",
        "extra_c",
    ]
    assert comparison["status"] == "EXPLAINED_MISMATCH"
    assert comparison["comparison"]["difference_reasons"] == {
        "extra_a": "synthetic historical exclusion",
        "extra_b": "synthetic historical exclusion",
        "extra_c": "synthetic historical exclusion",
        "historical_000": "synthetic canonical removal",
    }
    assert len(comparison["comparison"]["difference_reasons_sha256"]) == 64
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


def test_strict_reporting_comparison_rejects_unexplained_211_213_difference():
    locked = build_reporting_universe_manifest(
        _ids("historical", 211),
        provenance={"kind": "synthetic_historical_source"},
        seed=23,
        created_at=CREATED_AT,
        strict=True,
    )

    with pytest.raises(ContractValidationError, match="difference reason"):
        compare_reporting_universe(
            locked,
            [*_ids("historical", 211), "extra_a", "extra_b"],
            observed_provenance={"kind": "synthetic_observation"},
            difference_reasons={},
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
    assert report["gates"]["protocol"]["status"] == "FAIL"
    assert "annotation_sha256" in report["gates"]["protocol"][
        "failed_mandatory_checks"
    ]
    assert report["gates"]["completeness"]["status"] == "FAIL"
    assert "media_inventory" in report["gates"]["completeness"][
        "missing_evidence"
    ]
    assert "same_class_overlap_pairs" in report["gates"]["completeness"][
        "missing_evidence"
    ]
    assert report["gates"]["causal_readiness"]["status"] == "FAIL"
    assert any(
        reason.startswith("causal_readiness: mandatory checks failed")
        for reason in report["failure_reasons"]
    )
    assert verify_content_hash(report)


def _fineaction_qualification_gates(tmp_path, overlap_pairs=20):
    digest = "a" * 64
    counter = 0
    tmp_path.mkdir(parents=True, exist_ok=True)

    def check(evidence):
        nonlocal counter
        path = tmp_path / f"fineaction-evidence-{counter}.json"
        counter += 1
        save_json(path, evidence)
        return {
            "mandatory": True,
            "passed": True,
            "evidence": {
                **evidence,
                "artifact_path": str(path),
                "artifact_sha256": sha256_file(path),
            },
        }

    return {
        "protocol": {
            "license": check({"license_id": "FineAction-research"}),
            "official_split": check({"manifest_sha256": digest}),
            "annotation_sha256": check({"sha256": digest}),
            "instance_interval_ids": check(
                {"field": "instance_id", "verified_count": 30}
            ),
        },
        "completeness": {
            "raw_video_access": check({"inventory_sha256": digest}),
            "same_class_overlap_pairs": check({"count": overlap_pairs}),
            "same_class_repeated_instances": check({"count": 30}),
            "qualified_ground_truth": check({"count": 30}),
            "qualified_videos": check({"count": 10}),
            "estimated_decode_storage_cost": check(
                {"decode_gpu_hours": 12.0, "storage_bytes": 1024}
            ),
        },
        "causal_readiness": {
            "causal_preprocessing_contract": check(
                {
                    "timestamp_convention": "zero_based_source_frame",
                    "future_frames_allowed": False,
                    "frame_stride": 2,
                    "manifest_sha256": digest,
                }
            ),
            "minimal_dataset_loader_smoke": check(
                {"status": "PASS", "test_report_sha256": digest}
            ),
        },
    }


def test_fineaction_qualification_enforces_identity_sample_thresholds(tmp_path):
    failed = build_fineaction_qualification_report(
        _fineaction_qualification_gates(tmp_path / "failed", overlap_pairs=19),
        seed=29,
        created_at=CREATED_AT,
    )
    passed = build_fineaction_qualification_report(
        _fineaction_qualification_gates(tmp_path / "passed", overlap_pairs=20),
        seed=29,
        created_at=CREATED_AT,
    )

    assert failed["status"] == "FAIL"
    assert failed["gates"]["completeness"]["checks"][
        "same_class_overlap_pairs"
    ]["evidence_valid"] is False
    assert passed["status"] == "PASS"
    assert passed["qualified"] is True


def test_fineaction_qualification_rejects_forged_or_tampered_evidence(tmp_path):
    gates = _fineaction_qualification_gates(tmp_path / "tampered")
    evidence = gates["protocol"]["license"]["evidence"]
    Path(evidence["artifact_path"]).write_text(
        json.dumps({"license_id": "different"}) + "\n",
        encoding="utf-8",
    )

    report = build_fineaction_qualification_report(
        gates,
        seed=29,
        created_at=CREATED_AT,
    )

    assert report["status"] == "FAIL"
    assert report["gates"]["protocol"]["checks"]["license"]["evidence_valid"] is False


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
    difference_reasons_path = tmp_path / "difference-reasons.json"
    difference_reasons_path.write_text(
        json.dumps(
            {
                "extra_a": "synthetic historical exclusion",
                "extra_b": "synthetic historical exclusion",
            }
        ),
        encoding="utf-8",
    )
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
        "--difference-reasons",
        difference_reasons_path,
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
