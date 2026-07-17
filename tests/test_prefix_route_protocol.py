from copy import deepcopy
import json
from pathlib import Path

import pytest

from opentad.utils.prefix_route_protocol import (
    POPULATION_SCHEMA_VERSION,
    PROTOCOL_PASS,
    PROTOCOL_REVISE,
    PrefixRouteProtocolError,
    collection_authorization,
    load_protocol,
    validate_population_certificate,
    validate_protocol,
    validate_r0_evidence,
    validate_r1_certificate,
    validate_review_certificate,
)
from tools import validate_prefix_route_protocol as protocol_cli


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "causaltad"
    / "protocols"
    / "prefix_route_identifiability_v1.json"
)
AUTHOR_ID = "prefix-route-author"
REVIEWER_ID = "independent-protocol-reviewer"


def _record():
    return load_protocol(PROTOCOL_PATH, repo_root=ROOT)


def _review_certificate(record, verdict=PROTOCOL_PASS):
    return {
        "schema_version": "prefix-route-protocol-review-certificate-v1",
        "protocol_id": record["protocol"]["protocol_id"],
        "protocol_sha256": record["sha256"],
        "protocol_commit": "a" * 40,
        "reviewer_id": REVIEWER_ID,
        "review_task_id": "review-task-001",
        "reviewed_at": "2026-07-17T15:00:00+08:00",
        "verdict": verdict,
        "review_artifact_sha256": "b" * 64,
    }


def _population_certificate(record, review_sha="c" * 64):
    return {
        "schema_version": POPULATION_SCHEMA_VERSION,
        "protocol_id": record["protocol"]["protocol_id"],
        "protocol_sha256": record["sha256"],
        "protocol_review_certificate_sha256": review_sha,
        "historical_ids_sha256": "d" * 64,
        "canonical_ids_sha256": "e" * 64,
        "historical_count": 211,
        "canonical_count": 213,
        "missing_from_canonical": ["historical_only"],
        "extra_in_canonical": [
            "canonical_extra_a",
            "canonical_extra_b",
            "canonical_extra_c",
        ],
        "difference_reasons": {
            "historical_only": "HISTORICAL_LOCAL_FILE_EXTRA",
            "canonical_extra_a": "OFFICIAL_CANONICAL_MEMBERSHIP_CORRECTION",
            "canonical_extra_b": "HISTORICAL_LOCAL_FILE_ABSENT",
            "canonical_extra_c": "DOCUMENTED_VIDEO_ID_ALIAS",
        },
        "unexplained_ids": [],
        "status": "EXPLAINED_MISMATCH",
        "source_artifacts": {
            "historical_ids": "f" * 64,
            "canonical_ids": "1" * 64,
            "difference_reasons": "2" * 64,
        },
    }


def _r0_evidence(record, review_sha="c" * 64, population_sha="d" * 64):
    return {
        "schema_version": "prefix-route-r0-evidence-v1",
        "protocol_id": record["protocol"]["protocol_id"],
        "protocol_sha256": record["sha256"],
        "protocol_review_certificate_sha256": review_sha,
        "population_certificate_sha256": population_sha,
        "source_hashes": {},
        "definitions": deepcopy(record["protocol"]["r0"]["definitions"]),
        "aggregates": {},
        "claim_eligibility": {},
        "author_disclosure": "aggregate_only",
        "reviewer_detail_commitment_sha256": "e" * 64,
        "execution_environment": {},
        "status": "PASS_R0_COMPLETE",
    }


def _r1_certificate(record, review_sha="c" * 64, *, passing=False):
    payload = {
        field: None
        for field in record["protocol"]["r1"]["certificate_schema"][
            "required_top_level_fields"
        ]
    }
    payload.update(
        {
            "schema_version": "prefix-route-r1-certificate-v1",
            "protocol_id": record["protocol"]["protocol_id"],
            "protocol_sha256": record["sha256"],
            "protocol_review_certificate_sha256": review_sha,
            "motion_branch_enabled": False,
            "weight_shard_sha256": [],
            "raw_video_sha256_by_video": {},
            "feature_array_sha256_by_video": {},
            "software_versions": {},
            "static_support_audit": {},
            "dynamic_perturbation_audit": {},
            "existing_cache_linkage": {
                "failure_reasons": ["exact weight snapshot is unavailable"]
            },
            "status": "FAIL_UNVERIFIABLE",
        }
    )
    if not passing:
        return payload
    videos = {f"video_{index:03d}": "a" * 64 for index in range(213)}
    payload.update(
        {
            "repository_commit": "b" * 40,
            "extractor_source_sha256": "c" * 64,
            "resolved_command_sha256": "d" * 64,
            "environment_lock_sha256": "e" * 64,
            "software_versions": {
                "python": "3.11",
                "torch": "2.6",
                "transformers": "4.44",
                "opencv": "4.10",
            },
            "hf_snapshot_revision": "hf-commit",
            "model_config_sha256": "f" * 64,
            "processor_config_sha256": "1" * 64,
            "weight_shard_sha256": ["2" * 64],
            "raw_video_sha256_by_video": videos,
            "annotation_raw_sha256": "3" * 64,
            "cache_manifest_sha256": "4" * 64,
            "feature_array_sha256_by_video": videos,
            "support_map_sha256": "5" * 64,
            "decision_time_convention": "d_j=min((j+1)*8,F_v)-1",
            "decoded_rgb_stream_scope": "decoded_rgb_frames_to_feature_token_only",
            "static_support_audit": {
                "all_support_le_decision": True,
                "all_support_equals_decision": True,
            },
            "dynamic_perturbation_audit": {
                "future_mutation_invariant": True,
                "other_batch_image_invariant": True,
                "batch_order_invariant": True,
                "support_frame_sensitive": True,
            },
            "existing_cache_linkage": {
                "exact_weight_snapshot_bound": True,
                "raw_video_hashes_bound": True,
                "extraction_environment_bound": True,
                "feature_arrays_bound": True,
                "failure_reasons": [],
            },
            "status": "PASS_R1_EXISTING_CACHE_CERTIFIED",
        }
    )
    return payload


def test_protocol_is_complete_and_binds_current_sources():
    record = _record()
    assert record["protocol"]["status"] == "PROTOCOL_REVIEW_PENDING"
    assert record["protocol"]["population"]["reporting"]["primary_role"] == (
        "canonical_reporting_213"
    )
    assert record["protocol"]["r0"]["definitions"]["sequential_gap"] == (
        "next_start-current_end"
    )
    assert set(
        record["protocol"]["r2"]["arms"]["B4"]["allowed_route_specific_deltas"]
    ) == {
        "D1_ATOMIC_RELEASE_BEFORE_SAME_BIN_RESEED",
        "D2_CONTINUOUS_MASS_EPHEMERAL_NONCANONICAL_UOT",
    }


def test_protocol_rejects_unknown_top_level_and_section_fields():
    protocol = deepcopy(_record()["protocol"])
    protocol["surprise"] = True
    with pytest.raises(PrefixRouteProtocolError, match="protocol fields differ"):
        validate_protocol(protocol)

    protocol = deepcopy(_record()["protocol"])
    protocol["r0"]["surprise"] = True
    with pytest.raises(PrefixRouteProtocolError, match="r0 fields differ"):
        validate_protocol(protocol)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda p: p["task"].__setitem__("future_frame_access", True),
            "future frame access",
        ),
        (
            lambda p: p["r0"]["definitions"].__setitem__(
                "sequential_gap", "current_end-next_start"
            ),
            "sequential_gap",
        ),
        (
            lambda p: p["population"]["reporting"].__setitem__(
                "historical_population_may_be_primary", True
            ),
            "historical 211",
        ),
        (
            lambda p: p["r2"]["arms"]["B4"].__setitem__(
                "allowed_route_specific_deltas",
                ["persistent_queries"],
            ),
            "B4 route-specific delta",
        ),
        (
            lambda p: p["r3"]["controls"]["semantic_derangement"].__setitem__(
                "global_consistent_class_rename_forbidden", False
            ),
            "global class renaming",
        ),
        (
            lambda p: p["r6"]["equivalence_margins"].__setitem__(
                "mOnlineAP_absolute", 0.02
            ),
            "mOnlineAP equivalence",
        ),
    ],
)
def test_protocol_rejects_scientific_contract_drift(mutation, message):
    protocol = deepcopy(_record()["protocol"])
    mutation(protocol)
    with pytest.raises(PrefixRouteProtocolError, match=message):
        validate_protocol(protocol)


def test_protocol_rejects_source_substitution(tmp_path):
    protocol = deepcopy(_record()["protocol"])
    protocol["source_bindings"]["code_sources"][
        "tools/analyze_ontad_instances.py"
    ] = "0" * 64
    with pytest.raises(PrefixRouteProtocolError, match="bound source hash differs"):
        validate_protocol(protocol, repo_root=ROOT)


def test_review_certificate_is_independent_and_byte_bound():
    record = _record()
    certificate = _review_certificate(record)
    assert validate_review_certificate(
        certificate,
        protocol=record["protocol"],
        protocol_sha256=record["sha256"],
        author_id=AUTHOR_ID,
        expected_commit="a" * 40,
    )["verdict"] == PROTOCOL_PASS

    forged = deepcopy(certificate)
    forged["protocol_sha256"] = "0" * 64
    with pytest.raises(PrefixRouteProtocolError, match="protocol hash"):
        validate_review_certificate(
            forged,
            protocol=record["protocol"],
            protocol_sha256=record["sha256"],
            author_id=AUTHOR_ID,
        )

    self_review = deepcopy(certificate)
    self_review["reviewer_id"] = AUTHOR_ID
    with pytest.raises(PrefixRouteProtocolError, match="self-certify"):
        validate_review_certificate(
            self_review,
            protocol=record["protocol"],
            protocol_sha256=record["sha256"],
            author_id=AUTHOR_ID,
        )


def test_collection_remains_blocked_without_pass():
    record = _record()
    assert collection_authorization(
        protocol_record=record,
    ) == {
        "authorized": False,
        "status": "BLOCKED_PENDING_INDEPENDENT_PROTOCOL_REVIEW",
        "allowed": [],
    }
    revised = {
        "certificate": _review_certificate(record, verdict=PROTOCOL_REVISE)
    }
    assert collection_authorization(
        protocol_record=record,
        review_record=revised,
    )["authorized"] is False
    passed = {"certificate": _review_certificate(record)}
    authorization = collection_authorization(
        protocol_record=record,
        review_record=passed,
    )
    assert authorization["authorized"] is True
    assert authorization["allowed"] == [
        "READ_ONLY_R0_ANNOTATION_CENSUS",
        "READ_ONLY_R1_CACHE_CAUSALITY_AUDIT",
    ]
    assert "B0_B4_MODEL_IMPLEMENTATION" in authorization["blocked"]


def test_population_certificate_reconciles_exact_211_to_213():
    record = _record()
    certificate = _population_certificate(record)
    assert validate_population_certificate(
        certificate,
        protocol=record["protocol"],
        protocol_sha256=record["sha256"],
        review_certificate_sha256="c" * 64,
    )["status"] == "EXPLAINED_MISMATCH"

    bad_count = deepcopy(certificate)
    bad_count["extra_in_canonical"].pop()
    bad_count["difference_reasons"].pop("canonical_extra_c")
    with pytest.raises(PrefixRouteProtocolError, match="does not reconcile"):
        validate_population_certificate(
            bad_count,
            protocol=record["protocol"],
            protocol_sha256=record["sha256"],
            review_certificate_sha256="c" * 64,
        )

    bad_reason = deepcopy(certificate)
    bad_reason["difference_reasons"]["canonical_extra_a"] = "AUTHOR_PREFERENCE"
    with pytest.raises(PrefixRouteProtocolError, match="reason is not allowed"):
        validate_population_certificate(
            bad_reason,
            protocol=record["protocol"],
            protocol_sha256=record["sha256"],
            review_certificate_sha256="c" * 64,
        )


def test_r0_evidence_is_definition_and_disclosure_bound():
    record = _record()
    evidence = _r0_evidence(record)
    assert validate_r0_evidence(
        evidence,
        protocol=record["protocol"],
        protocol_sha256=record["sha256"],
        review_certificate_sha256="c" * 64,
        population_certificate_sha256="d" * 64,
    )["status"] == "PASS_R0_COMPLETE"

    leaked = deepcopy(evidence)
    leaked["author_disclosure"] = "per_video_hard_cases"
    with pytest.raises(PrefixRouteProtocolError, match="author disclosure"):
        validate_r0_evidence(
            leaked,
            protocol=record["protocol"],
            protocol_sha256=record["sha256"],
            review_certificate_sha256="c" * 64,
            population_certificate_sha256="d" * 64,
        )


def test_r1_can_record_fail_unverifiable_without_fake_hashes():
    record = _record()
    certificate = _r1_certificate(record)
    assert validate_r1_certificate(
        certificate,
        protocol=record["protocol"],
        protocol_sha256=record["sha256"],
        review_certificate_sha256="c" * 64,
    )["status"] == "FAIL_UNVERIFIABLE"

    no_reason = deepcopy(certificate)
    no_reason["existing_cache_linkage"]["failure_reasons"] = []
    with pytest.raises(PrefixRouteProtocolError, match="failure reasons"):
        validate_r1_certificate(
            no_reason,
            protocol=record["protocol"],
            protocol_sha256=record["sha256"],
            review_certificate_sha256="c" * 64,
        )


def test_r1_pass_requires_all_213_artifacts_and_dynamic_checks():
    record = _record()
    certificate = _r1_certificate(record, passing=True)
    assert validate_r1_certificate(
        certificate,
        protocol=record["protocol"],
        protocol_sha256=record["sha256"],
        review_certificate_sha256="c" * 64,
    )["status"] == "PASS_R1_EXISTING_CACHE_CERTIFIED"

    missing_video = deepcopy(certificate)
    missing_video["raw_video_sha256_by_video"].pop("video_000")
    with pytest.raises(PrefixRouteProtocolError, match="canonical reporting"):
        validate_r1_certificate(
            missing_video,
            protocol=record["protocol"],
            protocol_sha256=record["sha256"],
            review_certificate_sha256="c" * 64,
        )

    future_sensitive = deepcopy(certificate)
    future_sensitive["dynamic_perturbation_audit"][
        "future_mutation_invariant"
    ] = False
    with pytest.raises(PrefixRouteProtocolError, match="dynamic audit"):
        validate_r1_certificate(
            future_sensitive,
            protocol=record["protocol"],
            protocol_sha256=record["sha256"],
            review_certificate_sha256="c" * 64,
        )


def test_cli_validates_protocol_but_blocks_collection_without_review(capsys):
    assert protocol_cli.main(
        [
            "validate-protocol",
            "--protocol",
            str(PROTOCOL_PATH),
            "--repo-root",
            str(ROOT),
        ]
    ) == 0
    valid = json.loads(capsys.readouterr().out)
    assert valid["status"] == "PROTOCOL_VALID_REVIEW_REQUIRED"
    assert valid["gpu_hours_authorized"] == 0

    assert protocol_cli.main(
        [
            "authorize-collection",
            "--protocol",
            str(PROTOCOL_PATH),
            "--repo-root",
            str(ROOT),
        ]
    ) == 2
    blocked = json.loads(capsys.readouterr().out)
    assert blocked["status"] == "BLOCKED_PENDING_INDEPENDENT_PROTOCOL_REVIEW"
