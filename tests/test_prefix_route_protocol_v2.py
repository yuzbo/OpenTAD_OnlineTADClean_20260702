from __future__ import annotations

import base64
import hashlib
import inspect
import json
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from opentad.evaluations.prefix_route_r6_v2 import (
    REQUIRED_ARMS,
    PrefixRouteR6Error,
    canonical_emission,
    derive_per_video_cell,
    paired_crossed_bootstrap,
    terminal_route_decision,
)
from opentad.utils.evidence_bundle import (
    EvidenceBundleError,
    bundle_file_reference,
)
from opentad.utils.prefix_route_controls_v2 import (
    PrefixRouteControlError,
    feature_time_shuffle_permutation,
    semantic_derangement,
)
from opentad.utils.prefix_route_fairness_v2 import (
    CORE_ARMS,
    PrefixRouteFairnessError,
    derive_fairness_audit,
)
from opentad.utils.prefix_route_ood_v2 import (
    PrefixRouteOODError,
    audit_sequence_sets,
    generate_sequence,
    iter_balanced_public_set,
)
import opentad.utils.prefix_route_protocol_v2 as protocol_module
from opentad.utils.prefix_route_protocol_v2 import (
    DIFFERENCE_REASON_SCHEMA,
    ID_LIST_SCHEMA,
    POPULATION_REQUEST_SCHEMA,
    PROTOCOL_PASS,
    REVIEW_SCHEMA,
    PrefixRouteProtocolV2Error,
    canonical_json_bytes,
    canonical_sha256,
    derive_r0_envelope,
    load_protocol,
    load_signed_review,
    validate_exposure_ledger_bytes,
    validate_population_bundle,
    validate_protocol,
    validate_r0_bundle,
)
from opentad.utils.prefix_route_r0_v2 import collect_r0_census
import opentad.utils.prefix_route_r1_v2 as r1_module
from opentad.utils.prefix_route_r1_v2 import (
    R1_DYNAMIC_SCHEMA,
    R1_REQUEST_SCHEMA,
    R1_SUPPORT_SCHEMA,
    PrefixRouteR1Error,
    derive_r1_status,
    run_dynamic_token_record,
    validate_r1_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "causaltad"
    / "protocols"
    / "prefix_route_identifiability_v2.json"
)
MANIFEST_PATH = (
    ROOT
    / "configs"
    / "causaltad"
    / "protocols"
    / "prefix_route_identifiability_v2_manifest.json"
)
POLICY_SECTIONS = (
    "task",
    "governance",
    "population",
    "r0",
    "r1",
    "r2",
    "r3",
    "r4",
    "r5",
    "r6",
    "evidence_package",
)


def _protocol_value():
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _relock(value):
    value["policy_lock_sha256"] = canonical_sha256(
        {section: value[section] for section in POLICY_SECTIONS}
    )
    return value


def _write_bytes(root, relative, payload):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return bundle_file_reference(path, root)


def _write_json(root, relative, value):
    return _write_bytes(root, relative, canonical_json_bytes(value))


def _review_stub():
    return {"attestation_sha256": "a" * 64}


def test_protocol_v2_is_canonical_and_semantically_valid():
    record = load_protocol(PROTOCOL_PATH)
    assert record["protocol"]["status"] == "PROTOCOL_REVIEW_PENDING"
    assert record["sha256"] == hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
    assert PROTOCOL_PATH.read_bytes() == canonical_json_bytes(record["protocol"])


def _mutate_future(value):
    value["task"]["future_frame_access"] = True


def _mutate_formal_training(value):
    value["governance"]["current_authorization"].append("FORMAL_TRAINING")


def _mutate_early_stopping(value):
    value["r2"]["fairness"]["early_stopping"] = "ALLOWED"


def _mutate_r0(value):
    value["r0"]["pair_rules"]["pair_counting"] = "ordered_pairs"


def _mutate_r5(value):
    value["r5"]["factor_levels"]["event_topology"]["shift"].pop()


def _mutate_survival(value):
    value["r6"]["b4_survival"]["mechanism_requirement"] = "OPTIONAL"


def _mutate_r1_noop(value):
    value["r1"]["dynamic_mutations"]["future_frames"] = "no_op"


def _mutate_semantic_identity(value):
    value["r3"]["controls"]["semantic_derangement"][
        "change_every_instance_label"
    ] = False


@pytest.mark.parametrize(
    "mutation",
    [
        _mutate_future,
        _mutate_formal_training,
        _mutate_early_stopping,
        _mutate_r0,
        _mutate_r5,
        _mutate_survival,
        _mutate_r1_noop,
        _mutate_semantic_identity,
    ],
)
def test_relocked_forbidden_protocol_mutations_are_rejected(mutation):
    value = _protocol_value()
    mutation(value)
    _relock(value)
    with pytest.raises(PrefixRouteProtocolV2Error):
        validate_protocol(value)


def _signed_review_fixture(tmp_path, monkeypatch, *, verdict=PROTOCOL_PASS):
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(Encoding.OpenSSH, PublicFormat.OpenSSH)
    public_text = public.decode("ascii") + " test-reviewer"
    fingerprint = protocol_module._public_key_fingerprint(public_text)
    monkeypatch.setattr(protocol_module, "REVIEWER_PUBLIC_KEY", public_text)
    monkeypatch.setattr(protocol_module, "REVIEWER_FINGERPRINT", fingerprint)
    protocol = _protocol_value()
    trust = protocol["governance"]["reviewer_trust"]
    trust["public_key_openssh"] = public_text
    trust["public_key_fingerprint"] = fingerprint
    _relock(protocol)
    protocol_bytes = canonical_json_bytes(protocol)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_bytes(protocol_bytes)
    protocol_record = {
        "path": protocol_path,
        "bytes": protocol_bytes,
        "sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "protocol": protocol,
    }
    manifest_value = {
        "schema_version": "prefix-route-source-manifest-v2",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_record["sha256"],
        "entries": {},
    }
    manifest_bytes = canonical_json_bytes(manifest_value)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    manifest_record = {
        "path": manifest_path,
        "bytes": manifest_bytes,
        "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest": manifest_value,
    }
    attestation = {
        "schema_version": REVIEW_SCHEMA,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_record["sha256"],
        "protocol_commit": "1" * 40,
        "protocol_tree_sha1": "2" * 40,
        "source_manifest_sha256": manifest_record["sha256"],
        "reviewer_id": protocol_module.REVIEWER_ID,
        "review_task_id": protocol_module.REVIEWER_ID,
        "reviewed_at": "2026-07-17T10:00:00Z",
        "verdict": verdict,
        "scope": protocol_module.REVIEW_SCOPE,
        "findings": [] if verdict == PROTOCOL_PASS else [{"id": "P0"}],
        "commands": ["pytest"],
        "test_summary": {"passed": 1},
        "no_prohibited_access": True,
    }
    attestation_bytes = canonical_json_bytes(attestation)
    attestation_path = tmp_path / "review.json"
    attestation_path.write_bytes(attestation_bytes)
    signature_path = tmp_path / "review.sig"
    signature_path.write_bytes(
        base64.b64encode(private.sign(attestation_bytes)) + b"\n"
    )
    monkeypatch.setattr(
        protocol_module,
        "verify_frozen_git_tree",
        lambda **kwargs: True,
    )
    return {
        "private": private,
        "protocol_record": protocol_record,
        "manifest_record": manifest_record,
        "attestation": attestation,
        "attestation_path": attestation_path,
        "signature_path": signature_path,
    }


def test_signed_review_binds_content_verdict_and_reviewer(tmp_path, monkeypatch):
    fixture = _signed_review_fixture(tmp_path, monkeypatch)
    loaded = load_signed_review(
        attestation_path=fixture["attestation_path"],
        signature_path=fixture["signature_path"],
        protocol_record=fixture["protocol_record"],
        manifest_record=fixture["manifest_record"],
        repo_root=tmp_path,
    )
    assert loaded["attestation"]["verdict"] == PROTOCOL_PASS

    forged = dict(fixture["attestation"])
    forged["protocol_commit"] = "3" * 40
    fixture["attestation_path"].write_bytes(canonical_json_bytes(forged))
    with pytest.raises(PrefixRouteProtocolV2Error, match="signature"):
        load_signed_review(
            attestation_path=fixture["attestation_path"],
            signature_path=fixture["signature_path"],
            protocol_record=fixture["protocol_record"],
            manifest_record=fixture["manifest_record"],
            repo_root=tmp_path,
        )


def test_random_or_author_generated_signature_cannot_forge_pass(tmp_path, monkeypatch):
    fixture = _signed_review_fixture(tmp_path, monkeypatch)
    author_key = Ed25519PrivateKey.generate()
    payload = fixture["attestation_path"].read_bytes()
    fixture["signature_path"].write_bytes(
        base64.b64encode(author_key.sign(payload)) + b"\n"
    )
    with pytest.raises(PrefixRouteProtocolV2Error, match="signature"):
        load_signed_review(
            attestation_path=fixture["attestation_path"],
            signature_path=fixture["signature_path"],
            protocol_record=fixture["protocol_record"],
            manifest_record=fixture["manifest_record"],
            repo_root=tmp_path,
        )


def test_authorization_api_has_no_unvalidated_dictionary_argument():
    parameters = inspect.signature(protocol_module.authorize_collection).parameters
    assert "review_record" not in parameters
    assert "certificate" not in parameters


def _population_fixture(tmp_path):
    historical = [f"video_{index:03d}" for index in range(211)]
    canonical = [f"video_{index:03d}" for index in range(2, 215)]
    historical_only = sorted(set(historical) - set(canonical))
    canonical_only = sorted(set(canonical) - set(historical))
    historical_ref = _write_json(
        tmp_path,
        "population/historical.json",
        {
            "schema_version": ID_LIST_SCHEMA,
            "role": "historical_reporting_211_audit_only",
            "ids": historical,
        },
    )
    canonical_ref = _write_json(
        tmp_path,
        "population/canonical.json",
        {
            "schema_version": ID_LIST_SCHEMA,
            "role": "canonical_reporting_213",
            "ids": canonical,
        },
    )
    source_ref = _write_bytes(
        tmp_path,
        "population/official_source.txt",
        b"official canonical membership source\n",
    )
    reasons = [
        {
            "video_id": video_id,
            "side": (
                "historical_only"
                if video_id in historical_only
                else "canonical_only"
            ),
            "reason_code": "OFFICIAL_CANONICAL_MEMBERSHIP_CORRECTION",
            "source_key": "official",
        }
        for video_id in sorted(historical_only + canonical_only)
    ]
    reasons_ref = _write_json(
        tmp_path,
        "population/reasons.json",
        {
            "schema_version": DIFFERENCE_REASON_SCHEMA,
            "reasons": reasons,
        },
    )
    protocol_record = load_protocol(PROTOCOL_PATH)
    request = {
        "schema_version": POPULATION_REQUEST_SCHEMA,
        "protocol_id": protocol_record["protocol"]["protocol_id"],
        "protocol_sha256": protocol_record["sha256"],
        "review_attestation_sha256": "a" * 64,
        "historical_ids": historical_ref,
        "canonical_ids": canonical_ref,
        "difference_reasons": reasons_ref,
        "reason_sources": {"official": source_ref},
    }
    return protocol_record, request, canonical


def test_population_status_is_derived_from_contained_sources(tmp_path):
    protocol_record, request, canonical = _population_fixture(tmp_path)
    result = validate_population_bundle(
        request,
        bundle_root=tmp_path,
        protocol_record=protocol_record,
        review_record=_review_stub(),
    )
    assert result["status"] == "EXPLAINED_MISMATCH"
    assert result["canonical_ids"] == canonical
    assert len(result["historical_only"]) == 2
    assert len(result["canonical_only"]) == 4


def test_population_source_substitution_and_fabricated_reason_fail(tmp_path):
    protocol_record, request, _ = _population_fixture(tmp_path)
    canonical_path = tmp_path / request["canonical_ids"]["path"]
    canonical_path.write_bytes(canonical_path.read_bytes() + b" ")
    with pytest.raises(EvidenceBundleError):
        validate_population_bundle(
            request,
            bundle_root=tmp_path,
            protocol_record=protocol_record,
            review_record=_review_stub(),
        )

    protocol_record, request, _ = _population_fixture(tmp_path / "second")
    reason_path = (
        tmp_path / "second" / request["difference_reasons"]["path"]
    )
    value = json.loads(reason_path.read_text(encoding="utf-8"))
    value["reasons"][0]["reason_code"] = "AUTHOR_GUESS"
    reason_path.write_bytes(canonical_json_bytes(value))
    request["difference_reasons"]["sha256"] = hashlib.sha256(
        reason_path.read_bytes()
    ).hexdigest()
    with pytest.raises(PrefixRouteProtocolV2Error, match="reason code"):
        validate_population_bundle(
            request,
            bundle_root=tmp_path / "second",
            protocol_record=protocol_record,
            review_record=_review_stub(),
        )


def _r0_annotation():
    return {
        "database": {
            "v0": {
                "subset": "validation",
                "duration": 4.0,
                "frame": 32,
                "annotations": [
                    {"segment": [0.0, 0.5], "label": "A"},
                    {"segment": [0.5, 1.0], "label": "A"},
                    {"segment": [1.0, 2.0], "label": "B"},
                    {"segment": [2.0, 3.0], "label": "Ambiguous"},
                    {"segment": [3.0, 3.5], "label": "C"},
                    {"segment": [3.0, 3.5], "label": "C"},
                ],
            },
            "v1": {
                "subset": "validation",
                "duration": 4.0,
                "frame": 32,
                "annotations": [
                    {"segment": [0.0, 2.0], "label": "A"},
                    {"segment": [1.0, 3.0], "label": "A"},
                    {"segment": [3.0, 4.0], "label": "C"},
                ],
            },
            "v2": {
                "subset": "validation",
                "duration": 4.0,
                "frame": 32,
                "annotations": [],
            },
        }
    }


def test_r0_collector_freezes_first_bin_terminal_duplicates_and_pair_types():
    report, detail = collect_r0_census(
        _r0_annotation(),
        ("A", "B", "C"),
        ["v0", "v1", "v2"],
        bootstrap_resamples=100,
        bootstrap_seed=2026071701,
    )
    aggregate = report["aggregate"]
    assert aggregate["video_count"] == 3
    assert aggregate["legal_instance_count"] == 6
    assert aggregate["zero_action_video_count"] == 1
    assert aggregate["ambiguous"]["reason_counts"] == {
        "AMBIGUOUS_LABEL": 1,
        "DUPLICATE_EXACT_ANNOTATION": 2,
    }
    assert aggregate["overlap_pairs"]["same_class"]["pair_count"] == 1
    assert (
        aggregate["same_bin_end_start_pairs"]["same_class"][
            "handoff_or_touching"
        ]["pair_count"]
        == 1
    )
    assert (
        aggregate["same_bin_end_start_pairs"]["same_class"][
            "overlap_transition"
        ]["pair_count"]
        == 1
    )
    terminal = detail["videos"][1]["legal_instances"][-1]
    assert terminal["end_observation_count"] == 32
    assert terminal["end_bin"] == 3
    assert detail["definitions"]["first_previous_observation_count"] == 0


def _ledger_bytes():
    return (
        ROOT
        / "research-wiki"
        / "evidence"
        / "prefix-route-prior-exposure-ledger.jsonl"
    ).read_bytes()


def test_exposure_ledger_is_canonical_hash_chained_and_append_only():
    payload = _ledger_bytes()
    result = validate_exposure_ledger_bytes(payload, frozen_prefix=payload)
    assert result["entry_count"] == 3
    tampered = payload.replace(b"aggregate_only", b"aggregate_FULL", 1)
    with pytest.raises(PrefixRouteProtocolV2Error):
        validate_exposure_ledger_bytes(tampered, frozen_prefix=payload)


def test_r0_author_report_is_recomputed_not_asserted(tmp_path):
    protocol_record = load_protocol(PROTOCOL_PATH)
    review = _review_stub()
    population = {
        "canonical_ids": ["v0", "v1", "v2"],
        "derived_sha256": "b" * 64,
    }
    annotation_bytes = canonical_json_bytes(_r0_annotation())
    class_map_bytes = b"A\nB\nC\n"
    ledger_bytes = _ledger_bytes()
    envelope = derive_r0_envelope(
        annotation_bytes=annotation_bytes,
        class_map_bytes=class_map_bytes,
        exposure_ledger_bytes=ledger_bytes,
        population_record=population,
        protocol_record=protocol_record,
        review_record=review,
        frozen_ledger_prefix=ledger_bytes,
    )
    refs = {
        "annotation": _write_bytes(
            tmp_path,
            "r0/annotation.json",
            annotation_bytes,
        ),
        "class_map": _write_bytes(tmp_path, "r0/classes.txt", class_map_bytes),
        "exposure_ledger": _write_bytes(
            tmp_path,
            "r0/exposure.jsonl",
            ledger_bytes,
        ),
        "author_report": _write_json(tmp_path, "r0/report.json", envelope),
    }
    request = {
        "schema_version": "prefix-route-r0-request-v2",
        "protocol_id": protocol_record["protocol"]["protocol_id"],
        "protocol_sha256": protocol_record["sha256"],
        "review_attestation_sha256": review["attestation_sha256"],
        "population_derived_sha256": population["derived_sha256"],
        **refs,
    }
    validated = validate_r0_bundle(
        request,
        bundle_root=tmp_path,
        protocol_record=protocol_record,
        review_record=review,
        population_record=population,
        frozen_ledger_prefix=ledger_bytes,
    )
    assert validated["status"] == "PASS_R0_COMPLETE"

    forged = dict(envelope)
    forged["report"] = dict(forged["report"])
    forged["report"]["aggregate"] = {}
    report_path = tmp_path / refs["author_report"]["path"]
    report_path.write_bytes(canonical_json_bytes(forged))
    request["author_report"]["sha256"] = hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    with pytest.raises(PrefixRouteProtocolV2Error, match="differs"):
        validate_r0_bundle(
            request,
            bundle_root=tmp_path,
            protocol_record=protocol_record,
            review_record=review,
            population_record=population,
            frozen_ledger_prefix=ledger_bytes,
        )


def test_negative_controls_are_constructive_and_nontrivial():
    permutation = feature_time_shuffle_permutation("video", 9)
    assert sorted(permutation) == list(range(9))
    assert all(index != source for index, source in enumerate(permutation))
    with pytest.raises(PrefixRouteControlError):
        feature_time_shuffle_permutation("video", 1)

    instances = [
        {
            "video_id": f"v{index // 4}",
            "instance_id": index,
            "start_frame": float(index * 2),
            "end_frame": float(index * 2 + 1),
            "label": ("A", "B", "C", "D")[index % 4],
        }
        for index in range(16)
    ]
    result = semantic_derangement(instances, split_id="reporting")
    assert result["global_label_multiset_preserved"] is True
    assert result["all_instance_labels_changed"] is True
    assert result["global_consistent_class_rename"] is False
    assert sorted(row["label"] for row in result["instances"]) == sorted(
        row["label"] for row in instances
    )


def _ood_spec(observation="gaussian_noise_sigma_0.05"):
    return {
        "event_topology": "partial_overlap_pair",
        "temporal_geometry": "duration_8_gap_8_delay_1",
        "semantic_mapping": "identity_prototype_to_class",
        "observation_distribution": observation,
    }


def test_r5_hash_excludes_set_metadata_and_audit_rejects_overlap():
    left = generate_sequence(
        _ood_spec(),
        seed=11,
        sequence_index=0,
        set_name="TRAIN",
    )
    right = generate_sequence(
        _ood_spec(),
        seed=11,
        sequence_index=0,
        set_name="IID_HOLDOUT",
    )
    assert left["sequence_sha256"] == right["sequence_sha256"]
    with pytest.raises(PrefixRouteOODError, match="overlap"):
        audit_sequence_sets({"TRAIN": [left], "IID": [right]})

    train = list(iter_balanced_public_set("TRAIN", 4, seed=101))
    iid = list(iter_balanced_public_set("IID_HOLDOUT", 4, seed=202))
    audit = audit_sequence_sets({"TRAIN": train, "IID": iid})
    assert audit["pairwise_disjoint"] is True


def _fairness_fixture():
    rows = []
    inventories = {}
    for arm in CORE_ARMS:
        rows.append(
            {
                "arm": arm,
                "trainable_parameters": 100,
                "train_macs_per_optimizer_event": 1000.0,
                "inference_macs_per_decision": 100.0,
                "live_causal_state_bytes": 400.0,
                "peak_training_memory_bytes": 10000.0,
                "peak_inference_memory_bytes": 2000.0,
                "latency_median_ms": 4.0,
                "latency_p95_ms": 6.0,
                "optimizer_event_count": 10,
                "effective_token_count": 1000,
                "gradient_accumulation_steps": 2,
                "hyperparameter_trial_count": 3,
                "calibration_video_count": 40,
                "seed_count": 3,
            }
        )
        inventories[arm] = [
            {
                "name": "head.weight",
                "numel": 100,
                "requires_grad": True,
                "used_by_forward": True,
                "gradient_observed_in_contract_smoke": True,
                "purpose": "prediction",
            }
        ]
    return rows, inventories


def test_fairness_is_derived_and_rejects_dummy_padding():
    rows, inventories = _fairness_fixture()
    result = derive_fairness_audit(rows, inventories)
    assert result["status"] == "PASS_BOTH_CAPACITY_AND_RESOURCE_MATCHED"
    inventories["B4"][0]["used_by_forward"] = False
    with pytest.raises(PrefixRouteFairnessError, match="dummy"):
        derive_fairness_audit(rows, inventories)


def _r6_fixture():
    eligible = ("same_class_repetition", "same_bin_end_start")
    cells = []
    dataset = []
    for arm in REQUIRED_ARMS:
        for run_seed in (705, 706):
            dataset.append(
                {
                    "arm": arm,
                    "seed": run_seed,
                    "mOnlineAP": 0.50 if arm in {"B4", "B2"} else 0.49,
                }
            )
            for video_id in ("v0", "v1"):
                matched = {
                    "B4": 9,
                    "B2": 9,
                    "B3": 8,
                    "A1_NO_UOT": 8,
                    "A2_BINARY_MASS": 8,
                    "A1_A2_JOINT": 5,
                    "A3_NO_CONSISTENCY": 8,
                    "A4_NO_SAME_BIN_REUSE": 5,
                    "A5_NO_NEURAL_LATCH": 8,
                }[arm]
                cells.append(
                    {
                        "arm": arm,
                        "seed": run_seed,
                        "video_id": video_id,
                        "ground_truth_count": 10,
                        "matched_ground_truth_count": matched,
                        "false_emission_count": 1,
                        "duplicate_emission_count": 0,
                        "fragmented_ground_truth_count": 0,
                        "endpoint_latency_bins_sum": float(matched),
                        "endpoint_latency_observed_count": matched,
                        "stress": {
                            "same_class_repetition": {
                                "ground_truth_count": 5,
                                "matched_ground_truth_count": (
                                    1 if arm == "A1_A2_JOINT" else 5 if arm == "B4" else 4
                                ),
                            },
                            "same_bin_end_start": {
                                "ground_truth_count": 5,
                                "matched_ground_truth_count": (
                                    1
                                    if arm == "A4_NO_SAME_BIN_REUSE"
                                    else 5
                                    if arm == "B4"
                                    else 4
                                ),
                            },
                        },
                    }
                )
    return cells, dataset, eligible


def test_r6_uses_complete_crossed_pairs_and_required_joint_deletions():
    cells, dataset, eligible = _r6_fixture()
    inference = paired_crossed_bootstrap(
        cells,
        dataset,
        eligible_stress_families=eligible,
        resamples=200,
        seed=17,
    )
    assert inference["simultaneous_comparison_count"] == 8 * 8
    decision = terminal_route_decision(inference)
    assert decision["status"] == "PASS_B4_ROUTE_SURVIVES"
    assert decision["d1_established"] is True
    assert decision["d2_established"] is True

    with pytest.raises(PrefixRouteR6Error, match="Cartesian"):
        paired_crossed_bootstrap(
            cells[:-1],
            dataset,
            eligible_stress_families=eligible,
            resamples=10,
        )


def test_formal_output_uses_evaluator_field_names_only():
    row = {
        "emission_id": "e0",
        "stream_key": "v0",
        "sequence_id": 0,
        "start": 0.0,
        "end": 8.0,
        "class": "A",
        "score": 0.9,
        "source_frame": 8,
        "emit_frame": 8,
    }
    assert canonical_emission(row)["emit_frame"] == 8.0
    forged = dict(row)
    forged["decision_frame"] = forged["source_frame"]
    with pytest.raises(PrefixRouteR6Error, match="fields"):
        canonical_emission(forged)


def test_r6_per_video_cells_are_derived_from_gt_and_immutable_emissions():
    ground_truth = [
        {
            "gt_id": "g0",
            "stream_key": "v0",
            "label": "A",
            "start_frame": 0,
            "end_frame": 8,
        }
    ]
    emissions = [
        {
            "emission_id": "e0",
            "stream_key": "v0",
            "sequence_id": 0,
            "start": 0,
            "end": 8,
            "class": "A",
            "score": 0.9,
            "source_frame": 8,
            "emit_frame": 8,
        }
    ]
    cell = derive_per_video_cell(
        arm="B4",
        seed=705,
        video_id="v0",
        ground_truth=ground_truth,
        emissions=emissions,
        stress_ground_truth_ids={"direct_complete": ["g0"]},
        fps=30.0,
    )
    assert cell["matched_ground_truth_count"] == 1
    assert cell["false_emission_count"] == 0
    assert cell["stress"]["direct_complete"]["matched_ground_truth_count"] == 1


def _float32_base64(values):
    payload = struct.pack("<" + "f" * len(values), *values)
    return base64.b64encode(payload).decode("ascii")


def _r1_fixture(tmp_path):
    canonical_ids = [f"video_{index:02d}" for index in range(16)]
    annotation_bytes = canonical_json_bytes({"database": {}})
    annotation_ref = _write_bytes(
        tmp_path,
        "r1/annotation.json",
        annotation_bytes,
    )
    refs = {
        "extractor_source": _write_bytes(
            tmp_path,
            "r1/extractor.py",
            b"def extract(frame): return frame\n",
        ),
        "resolved_command": _write_bytes(
            tmp_path,
            "r1/command.txt",
            b"python extractor.py --local-only\n",
        ),
        "environment_lock": _write_bytes(
            tmp_path,
            "r1/environment.lock",
            b"python==3.11\nnumpy==2\n",
        ),
        "software_versions": _write_json(
            tmp_path,
            "r1/software.json",
            {"python": "3.11", "numpy": np.__version__},
        ),
        "model_config": _write_json(
            tmp_path,
            "r1/model.json",
            {"model": "fixture"},
        ),
        "processor_config": _write_json(
            tmp_path,
            "r1/processor.json",
            {"processor": "fixture"},
        ),
        "annotation": annotation_ref,
    }
    weight_ref = _write_bytes(tmp_path, "r1/weights/model.bin", b"weights")
    support_videos = []
    manifest_videos = {}
    raw_refs = {}
    feature_refs = {}
    for index, video_id in enumerate(canonical_ids):
        frame_count = 24 if index < 8 else 17
        source_frames = [
            min(start + 8, frame_count) - 1
            for start in range(0, frame_count, 8)
        ]
        support_videos.append(
            {
                "video_id": video_id,
                "frame_count": frame_count,
                "tokens": [
                    {
                        "token_index": token_index,
                        "source_frame": source,
                        "decision_frame": source,
                        "support_frames": [source],
                    }
                    for token_index, source in enumerate(source_frames)
                ],
            }
        )
        raw_refs[video_id] = _write_bytes(
            tmp_path,
            f"r1/raw/{video_id}.mp4",
            f"raw:{video_id}".encode("ascii"),
        )
        feature_path = tmp_path / f"r1/features/{video_id}.npy"
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(
            feature_path,
            np.arange(len(source_frames) * 2, dtype=np.float32).reshape(
                len(source_frames),
                2,
            ),
        )
        feature_refs[video_id] = bundle_file_reference(feature_path, tmp_path)
        manifest_videos[video_id] = {
            "file": feature_path.name,
            "num_tokens": len(source_frames),
            "feature_dim": 2,
            "dtype": "float32",
            "source_frames": source_frames,
            "sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
        }
    support = {
        "schema_version": R1_SUPPORT_SCHEMA,
        "feature_stride": 8,
        "videos": support_videos,
    }
    support_ref = _write_json(tmp_path, "r1/support.json", support)
    support_by_id, _ = r1_module._parse_support_map(
        support,
        tuple(canonical_ids),
        8,
    )
    selected, _ = r1_module._category_candidates(support_by_id)
    baseline = _float32_base64([1.0, 2.0])
    support_mutated = _float32_base64([1.0, 3.0])
    records = [
        {
            "category": row["category"],
            "video_id": row["video_id"],
            "token_index": row["token_index"],
            "future_suffix_mode": (
                "paired_synthetic_continuation_after_eos"
                if row["token_index"]
                == len(support_by_id[row["video_id"]]["tokens"]) - 1
                else "native_future_suffix"
            ),
            "baseline_input_sha256": "0" * 64,
            "future_mutated_input_sha256": "1" * 64,
            "other_batch_mutated_input_sha256": "2" * 64,
            "batch_order_permuted_input_sha256": "3" * 64,
            "support_mutated_input_sha256": "4" * 64,
            "baseline_token_f32le_base64": baseline,
            "future_mutated_token_f32le_base64": baseline,
            "other_batch_mutated_token_f32le_base64": baseline,
            "batch_order_permuted_token_f32le_base64": baseline,
            "support_mutated_token_f32le_base64": support_mutated,
        }
        for row in selected
    ]
    dynamic_ref = _write_json(
        tmp_path,
        "r1/dynamic.json",
        {
            "schema_version": R1_DYNAMIC_SCHEMA,
            "audit_seed": 2026071702,
            "records": records,
        },
    )
    manifest = {
        "schema": "ontad_feature_cache_v1",
        "created_at": "2026-07-17T00:00:00Z",
        "annotation_file": "annotation.json",
        "annotation_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
        "encoder_id": "fixture@revision",
        "feature_policy": "packet_recent_frame",
        "timestamp_convention": "zero_based_source_frame",
        "feature_stride": 8,
        "feature_dim": 2,
        "dtype": "float32",
        "videos": manifest_videos,
    }
    manifest_ref = _write_json(tmp_path, "r1/manifest.json", manifest)
    protocol_record = load_protocol(PROTOCOL_PATH)
    request = {
        "schema_version": R1_REQUEST_SCHEMA,
        "protocol_id": protocol_record["protocol"]["protocol_id"],
        "protocol_sha256": protocol_record["sha256"],
        "review_attestation_sha256": "a" * 64,
        "repository_commit": "5" * 40,
        **refs,
        "hf_snapshot_revision": "fixture-revision",
        "weight_shards": [weight_ref],
        "cache_manifest": manifest_ref,
        "support_map": support_ref,
        "dynamic_audit": dynamic_ref,
        "raw_videos": raw_refs,
        "feature_arrays": feature_refs,
    }
    return request, canonical_ids


def test_r1_dynamic_runner_uses_native_and_paired_eos_future_modes():
    frames = np.arange(4 * 2 * 2 * 3, dtype=np.uint8).reshape(4, 2, 2, 3)

    def encode_batch(batch):
        return np.asarray(batch, dtype=np.float32).reshape(len(batch), -1)[:, :4]

    interior = run_dynamic_token_record(
        encode_batch=encode_batch,
        selected_rgb_frames=frames,
        video_id="video_interior",
        token_index=1,
        category="interior_token",
    )
    assert interior["future_suffix_mode"] == "native_future_suffix"
    assert (
        interior["baseline_token_f32le_base64"]
        == interior["future_mutated_token_f32le_base64"]
    )
    assert (
        interior["baseline_token_f32le_base64"]
        != interior["support_mutated_token_f32le_base64"]
    )

    terminal = run_dynamic_token_record(
        encode_batch=encode_batch,
        selected_rgb_frames=frames,
        video_id="video_terminal",
        token_index=3,
        category="last_partial_bin",
    )
    assert (
        terminal["future_suffix_mode"]
        == "paired_synthetic_continuation_after_eos"
    )
    assert (
        terminal["baseline_input_sha256"]
        != terminal["future_mutated_input_sha256"]
    )
    assert (
        terminal["baseline_token_f32le_base64"]
        == terminal["future_mutated_token_f32le_base64"]
    )
    assert (
        terminal["baseline_token_f32le_base64"]
        != terminal["support_mutated_token_f32le_base64"]
    )


def test_r1_reloads_sources_arrays_support_and_dynamic_tokens(tmp_path):
    request, canonical_ids = _r1_fixture(tmp_path)
    result = validate_r1_bundle(
        request,
        bundle_root=tmp_path,
        protocol_id=request["protocol_id"],
        protocol_sha256=request["protocol_sha256"],
        review_attestation_sha256="a" * 64,
        canonical_video_ids=canonical_ids,
    )
    assert result["status"] == "PASS_R1_EXISTING_CACHE_CERTIFIED"
    assert result["existing_cache_linkage"]["cache_key_sets_equal"] is True
    assert result["dynamic_perturbation_audit"]["record_count"] > 0


def test_r1_noop_or_key_substitution_is_recordable_failure(tmp_path):
    request, canonical_ids = _r1_fixture(tmp_path)
    dynamic_path = tmp_path / request["dynamic_audit"]["path"]
    dynamic = json.loads(dynamic_path.read_text(encoding="utf-8"))
    dynamic["records"][0]["future_mutated_input_sha256"] = "0" * 64
    dynamic_path.write_bytes(canonical_json_bytes(dynamic))
    request["dynamic_audit"]["sha256"] = hashlib.sha256(
        dynamic_path.read_bytes()
    ).hexdigest()
    result = derive_r1_status(
        request,
        bundle_root=tmp_path,
        protocol_id=request["protocol_id"],
        protocol_sha256=request["protocol_sha256"],
        review_attestation_sha256="a" * 64,
        canonical_video_ids=canonical_ids,
    )
    assert result["status"] == "FAIL_UNVERIFIABLE"
    assert "no-op" in result["failure_reasons"][0]

    request, canonical_ids = _r1_fixture(tmp_path / "keys")
    request["feature_arrays"].pop(canonical_ids[-1])
    request["feature_arrays"]["unrelated"] = next(
        iter(request["feature_arrays"].values())
    )
    with pytest.raises(PrefixRouteR1Error, match="cover"):
        validate_r1_bundle(
            request,
            bundle_root=tmp_path / "keys",
            protocol_id=request["protocol_id"],
            protocol_sha256=request["protocol_sha256"],
            review_attestation_sha256="a" * 64,
            canonical_video_ids=canonical_ids,
        )


def test_cli_blocks_collection_without_signed_review():
    if not MANIFEST_PATH.exists():
        pytest.skip("source manifest is generated after this test file is frozen")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/validate_prefix_route_protocol_v2.py"),
            "authorize-collection",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "BLOCKED_PENDING_SIGNED_INDEPENDENT_PROTOCOL_REVIEW" in completed.stdout
