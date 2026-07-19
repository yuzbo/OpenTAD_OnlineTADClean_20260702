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

import opentad.evaluations.prefix_route_r6_v2 as r6_module
from opentad.evaluations.prefix_route_r6_v2 import (
    REQUIRED_CONTRASTS,
    PrefixRouteR6Error,
    canonical_emission,
    derive_per_video_cell,
    evaluate_r6_raw_evidence,
)
from opentad.utils.evidence_bundle import (
    bundle_file_reference,
)
from opentad.utils.immutable_event_ledger import (
    ImmutableEventLedger,
    persist_verified_ledger,
)
from opentad.utils.prefix_route_controls_v2 import (
    PrefixRouteControlError,
    feature_time_shuffle_permutation,
    semantic_derangement,
)
import opentad.utils.prefix_route_fairness_v2 as fairness_module
from opentad.utils.prefix_route_fairness_v2 import (
    ArmRuntimeAdapter,
    CALIBRATION_MANIFEST_SCHEMA,
    CORE_ARMS,
    EXECUTION_TRACE_SCHEMA,
    PrefixRouteFairnessError,
    SEED_MANIFEST_SCHEMA,
    TRIAL_MANIFEST_SCHEMA,
    derive_fairness_audit,
    derive_runtime_budget_evidence,
)
from opentad.utils.prefix_route_b2_contract_v2 import (
    NEWBORN_QUERY_COUNT,
    TRACK_CAPACITY,
    PrefixRouteB2Error,
    build_temporal_tracklet_assignment,
    instance_aware_risk_set,
    temporal_motr_transition,
)
from opentad.utils.prefix_route_ood_v2 import (
    EXPECTED_SET_COUNTS,
    EXPECTED_SET_SEEDS,
    FACTOR_FAMILIES,
    PrefixRouteOODError,
    audit_sequence_sets,
    generate_sequence,
    iter_balanced_public_set,
)
import opentad.utils.prefix_route_protocol_v2 as protocol_module
from opentad.utils.prefix_route_protocol_v2 import (
    HISTORICAL_INVENTORY_SCHEMA,
    POPULATION_REQUEST_SCHEMA,
    PROTOCOL_PASS,
    REVIEW_SCHEMA,
    PrefixRouteProtocolV2Error,
    canonical_json_bytes,
    canonical_sha256,
    load_protocol,
    load_signed_review,
    validate_exposure_ledger_bytes,
    validate_population_bundle,
    validate_protocol,
    validate_r0_bundle,
)
from opentad.utils.prefix_route_r0_v2 import (
    PrefixRouteR0Error,
    collect_r0_census,
)
import opentad.utils.prefix_route_r1_v2 as r1_module
from opentad.utils.prefix_route_r1_v2 import (
    R1_COMMAND_SCHEMA,
    R1_DYNAMIC_SCHEMA,
    R1_REQUEST_SCHEMA,
    R1_SOFTWARE_SCHEMA,
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


def _synthetic_registered_protocol():
    value = _protocol_value()
    registration = value["population"]["source_registration"]
    registration.update(
        {
            "state": "REGISTERED_IN_FIXED_REVIEWED_PROTOCOL",
            "authoritative_annotation_sha256": "a" * 64,
            "historical_inventory_sha256": "b" * 64,
            "source_origin_attestation_sha256": "c" * 64,
            "official_video_manifest_sha256": "d" * 64,
            "registration_commit": "1" * 40,
        }
    )
    binding = value["r1"]["execution_binding"]
    binding.update(
        {
            "state": "REGISTERED_IN_FIXED_REVIEWED_PROTOCOL",
            "repository_commit": "2" * 40,
            "hf_snapshot_revision": "fixture-snapshot",
            "extractor_source_sha256": "3" * 64,
            "snapshot_manifest_sha256": "4" * 64,
            "resolved_command_sha256": "5" * 64,
            "environment_lock_sha256": "6" * 64,
            "software_versions_sha256": "7" * 64,
            "annotation_sha256": "a" * 64,
            "cache_manifest_sha256": "8" * 64,
            "support_map_sha256": "9" * 64,
            "raw_video_manifest_sha256": "0" * 64,
        }
    )
    value["governance"]["post_pass_scope"] = [
        "READ_ONLY_R0_ANNOTATION_CENSUS",
        "READ_ONLY_R1_CACHE_CAUSALITY_AUDIT",
    ]
    return value


def test_registered_protocol_rejects_placeholder_and_cross_source_annotation():
    placeholder = _relock(_synthetic_registered_protocol())
    with pytest.raises(PrefixRouteProtocolV2Error, match="placeholder"):
        validate_protocol(placeholder)

    cross_source = _synthetic_registered_protocol()
    cross_source["population"]["source_registration"]["release_revision"] = (
        "THUMOS14_TEMPORAL_ANNOTATIONS_RELEASE_2014"
    )
    cross_source["r1"]["execution_binding"]["annotation_sha256"] = "f" * 64
    _relock(cross_source)
    with pytest.raises(PrefixRouteProtocolV2Error, match="annotation differs"):
        validate_protocol(cross_source)


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


def _signed_review_fixture(
    tmp_path,
    monkeypatch,
    *,
    verdict=PROTOCOL_PASS,
    registered=False,
):
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(Encoding.OpenSSH, PublicFormat.OpenSSH)
    public_text = public.decode("ascii") + " test-reviewer"
    fingerprint = protocol_module._public_key_fingerprint(public_text)
    monkeypatch.setattr(protocol_module, "REVIEWER_PUBLIC_KEY", public_text)
    monkeypatch.setattr(protocol_module, "REVIEWER_FINGERPRINT", fingerprint)
    protocol = (
        _synthetic_registered_protocol()
        if registered
        else _protocol_value()
    )
    if registered:
        protocol["population"]["source_registration"]["release_revision"] = (
            "THUMOS14_TEMPORAL_ANNOTATIONS_RELEASE_2014"
        )
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


def test_signed_registered_source_commit_must_be_reviewed_commit_parent(
    tmp_path,
    monkeypatch,
):
    fixture = _signed_review_fixture(
        tmp_path,
        monkeypatch,
        registered=True,
    )
    monkeypatch.setattr(
        protocol_module,
        "_git",
        lambda *args: ("4" * 40 + "\n").encode("ascii"),
    )
    with pytest.raises(PrefixRouteProtocolV2Error, match="direct parent"):
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
    assert not hasattr(protocol_module, "derive_r0_envelope")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows 8.3 path regression")
def test_evidence_bundle_accepts_equivalent_windows_short_path(tmp_path):
    import ctypes

    root = tmp_path / "Long Evidence Bundle Directory"
    root.mkdir()
    evidence = root / "Long Evidence Filename.txt"
    evidence.write_bytes(b"bound evidence\n")
    get_short_path = ctypes.windll.kernel32.GetShortPathNameW
    required = get_short_path(str(root), None, 0)
    if not required:
        pytest.skip("8.3 aliases are disabled on this volume")
    buffer = ctypes.create_unicode_buffer(required)
    written = get_short_path(str(root), buffer, required)
    if not written or "~" not in buffer.value:
        pytest.skip("this volume did not create an 8.3 alias")
    reference = bundle_file_reference(evidence, Path(buffer.value))
    assert reference["path"] == evidence.name


def _population_fixture(tmp_path):
    canonical = [f"video_{index:03d}" for index in range(213)]
    annotation = {
        "database": {
            video_id: {
                "subset": "validation",
                "duration": 1.0,
                "frame": 8,
                "annotations": [],
            }
            for video_id in canonical
        }
    }
    annotation_ref = _write_json(
        tmp_path,
        "population/official_annotation.json",
        annotation,
    )
    artifacts = {}
    entries = []
    for index, canonical_id in enumerate(canonical[:211]):
        source_id = "legacy_000" if index == 0 else canonical_id
        reference = _write_bytes(
            tmp_path,
            f"population/videos/{source_id}.mp4",
            f"video-bytes:{source_id}\n".encode("ascii"),
        )
        artifacts[source_id] = reference
        entries.append(
            {
                "source_video_id": source_id,
                "canonical_video_id": canonical_id,
                "artifact_path": reference["path"],
                "artifact_sha256": reference["sha256"],
            }
        )
    entries.sort(key=lambda row: row["source_video_id"])
    inventory_ref = _write_json(
        tmp_path,
        "population/historical_inventory.json",
        {
            "schema_version": HISTORICAL_INVENTORY_SCHEMA,
            "release_id": "THUMOS14_TEMPORAL_ANNOTATIONS",
            "release_revision": "THUMOS14_TEMPORAL_ANNOTATIONS_RELEASE_2014",
            "reporting_subset": "validation",
            "entries": entries,
        },
    )
    base = load_protocol(PROTOCOL_PATH)
    protocol = json.loads(json.dumps(base["protocol"]))
    registration = protocol["population"]["source_registration"]
    registration.update(
        {
            "state": "REGISTERED_IN_FIXED_REVIEWED_PROTOCOL",
            "release_revision": "THUMOS14_TEMPORAL_ANNOTATIONS_RELEASE_2014",
            "authoritative_annotation_sha256": annotation_ref["sha256"],
            "historical_inventory_sha256": inventory_ref["sha256"],
            "source_origin_attestation_sha256": "b" * 64,
            "official_video_manifest_sha256": "c" * 64,
            "registration_commit": "1" * 40,
        }
    )
    protocol["r1"]["execution_binding"].update(
        {
            "state": "REGISTERED_IN_FIXED_REVIEWED_PROTOCOL",
            "repository_commit": "2" * 40,
            "hf_snapshot_revision": "fixture-snapshot",
            "extractor_source_sha256": "3" * 64,
            "snapshot_manifest_sha256": "4" * 64,
            "resolved_command_sha256": "5" * 64,
            "environment_lock_sha256": "6" * 64,
            "software_versions_sha256": "7" * 64,
            "annotation_sha256": annotation_ref["sha256"],
            "cache_manifest_sha256": "8" * 64,
            "support_map_sha256": "9" * 64,
            "raw_video_manifest_sha256": "a" * 64,
        }
    )
    protocol["governance"]["post_pass_scope"] = [
        "READ_ONLY_R0_ANNOTATION_CENSUS",
        "READ_ONLY_R1_CACHE_CAUSALITY_AUDIT",
    ]
    _relock(protocol)
    validate_protocol(protocol)
    protocol_bytes = canonical_json_bytes(protocol)
    protocol_path = tmp_path / "registered-protocol.json"
    protocol_path.write_bytes(protocol_bytes)
    protocol_record = {
        "protocol": protocol,
        "sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "bytes": protocol_bytes,
        "path": protocol_path,
    }
    request = {
        "schema_version": POPULATION_REQUEST_SCHEMA,
        "protocol_id": protocol_record["protocol"]["protocol_id"],
        "protocol_sha256": protocol_record["sha256"],
        "review_attestation_sha256": "a" * 64,
        "authoritative_annotation": annotation_ref,
        "historical_inventory": inventory_ref,
        "historical_artifacts": artifacts,
    }
    return protocol_record, request, canonical


def test_population_requires_cryptographically_verified_pass(tmp_path):
    protocol_record, request, _ = _population_fixture(tmp_path)
    with pytest.raises(PrefixRouteProtocolV2Error, match="formal.*context"):
        protocol_module._validate_population_bundle_objects(
            request,
            bundle_root=tmp_path,
            protocol_record=protocol_record,
            review_record=_review_stub(),
            formal_context=object(),
        )
    parameters = inspect.signature(validate_population_bundle).parameters
    assert "protocol_record" not in parameters
    assert "review_record" not in parameters


def test_unregistered_protocol_blocks_population_before_arbitrary_lists(tmp_path):
    protocol_record = load_protocol(PROTOCOL_PATH)
    request = {
        "schema_version": POPULATION_REQUEST_SCHEMA,
        "protocol_id": protocol_record["protocol"]["protocol_id"],
        "protocol_sha256": protocol_record["sha256"],
        "review_attestation_sha256": "a" * 64,
        "source_origin_attestation": {
            "path": "fake.json",
            "sha256": "0" * 64,
        },
        "official_video_manifest": {
            "path": "fake.json",
            "sha256": "0" * 64,
        },
        "authoritative_annotation": {"path": "fake.json", "sha256": "0" * 64},
        "historical_inventory": {"path": "fake.json", "sha256": "0" * 64},
        "historical_artifacts": {},
    }
    request_reference = _write_json(
        tmp_path,
        "population/request.json",
        request,
    )
    with pytest.raises(PrefixRouteProtocolV2Error, match="not frozen"):
        validate_population_bundle(
            request_reference,
            bundle_root=tmp_path,
            repo_root=ROOT,
            review_attestation_path=tmp_path / "missing-review.json",
            review_signature_path=tmp_path / "missing-review.sig",
        )


def test_population_source_substitution_and_asserted_membership_fail(tmp_path):
    protocol_record, request, _ = _population_fixture(tmp_path)
    annotation_path = tmp_path / request["authoritative_annotation"]["path"]
    annotation_path.write_bytes(annotation_path.read_bytes() + b" ")
    with pytest.raises(PrefixRouteProtocolV2Error, match="formal.*context"):
        protocol_module._validate_population_bundle_objects(
            request,
            bundle_root=tmp_path,
            protocol_record=protocol_record,
            review_record=_review_stub(),
            formal_context=object(),
        )

    request["asserted_canonical_ids"] = ["author-selected"]
    with pytest.raises(PrefixRouteProtocolV2Error, match="formal.*context"):
        protocol_module._validate_population_bundle_objects(
            request,
            bundle_root=tmp_path,
            protocol_record=protocol_record,
            review_record=_review_stub(),
            formal_context=object(),
        )


def test_population_rejects_nonvideo_mp4_placeholder(tmp_path):
    path = tmp_path / "placeholder.mp4"
    payload = b"video-bytes:not-an-mp4\n"
    path.write_bytes(payload)
    with pytest.raises(PrefixRouteProtocolV2Error, match="decodable"):
        protocol_module._validate_historical_video_artifact(
            path,
            payload,
            "historical artifact fixture",
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


def test_r0_rejects_wrong_subset_and_weaker_exposure_disclosure():
    annotation = _r0_annotation()
    annotation["database"]["v0"]["subset"] = "test"
    with pytest.raises(PrefixRouteR0Error, match="registered reporting subset"):
        collect_r0_census(
            annotation,
            ("A", "B", "C"),
            ["v0", "v1", "v2"],
            bootstrap_resamples=10,
        )
    with pytest.raises(PrefixRouteR0Error, match="exposure status"):
        collect_r0_census(
            _r0_annotation(),
            ("A", "B", "C"),
            ["v0", "v1", "v2"],
            bootstrap_resamples=10,
            annotation_exposure_status="DESIGN_EXPOSED_ROUTE_SELECTION_ONLY",
        )


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
    with pytest.raises(PrefixRouteProtocolV2Error, match="not frozen"):
        validate_r0_bundle(
            {"path": "r0.json", "sha256": "0" * 64},
            population_request_reference={
                "path": "population.json",
                "sha256": "0" * 64,
            },
            bundle_root=tmp_path,
            repo_root=ROOT,
            review_attestation_path=tmp_path / "missing-review.json",
            review_signature_path=tmp_path / "missing-review.sig",
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


def _full_r5_package():
    package = {
        "TRAIN": list(
            iter_balanced_public_set(
                "TRAIN",
                EXPECTED_SET_COUNTS["TRAIN"],
                seed=EXPECTED_SET_SEEDS["TRAIN"],
            )
        ),
        "IID_HOLDOUT": list(
            iter_balanced_public_set(
                "IID_HOLDOUT",
                EXPECTED_SET_COUNTS["IID_HOLDOUT"],
                seed=EXPECTED_SET_SEEDS["IID_HOLDOUT"],
            )
        ),
        "COMPOUND_OOD": list(
            iter_balanced_public_set(
                "COMPOUND_OOD",
                EXPECTED_SET_COUNTS["COMPOUND_OOD"],
                seed=EXPECTED_SET_SEEDS["COMPOUND_OOD"],
            )
        ),
    }
    for family in FACTOR_FAMILIES:
        name = f"SINGLE_SHIFT_OOD:{family}"
        package[name] = list(
            iter_balanced_public_set(
                "SINGLE_SHIFT_OOD",
                EXPECTED_SET_COUNTS[name],
                seed=EXPECTED_SET_SEEDS[name],
                shifted_family=family,
            )
        )
    return package


def test_r5_complete_frozen_package_is_executable_disjoint_and_balanced():
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
    assert left["row_commitment_sha256"] != right["row_commitment_sha256"]
    with pytest.raises(PrefixRouteOODError, match="names differ"):
        audit_sequence_sets({"TRAIN": [left], "IID_HOLDOUT": [right]})

    package = _full_r5_package()
    audit = audit_sequence_sets(package)
    assert audit["evidence_role"] == (
        "DIAGNOSTIC_ONLY_REQUIRES_FORMAL_ENVELOPE"
    )
    assert "status" not in audit
    assert "complete_frozen_set_contract" not in audit
    assert audit["total_sequence_count"] == 3800
    assert all(
        report["maximum_cell_count"] - report["minimum_cell_count"] <= 1
        for report in audit["sets"].values()
    )
    handoff = generate_sequence(
        {
            "event_topology": "same_bin_handoff",
            "temporal_geometry": "duration_2_gap_0_delay_0",
            "semantic_mapping": "identity_prototype_to_class",
            "observation_distribution": "identity",
        },
        seed=7,
        sequence_index=0,
        set_name="diagnostic",
    )
    assert handoff["events"][0]["end_bin"] == handoff["events"][1]["start_bin"]
    left = package["TRAIN"][0]
    right = package["TRAIN"][1]
    for field in (
        "events",
        "observations",
        "derived",
        "sequence_sha256",
        "row_commitment_sha256",
    ):
        left[field], right[field] = right[field], left[field]
    with pytest.raises(PrefixRouteOODError, match="frozen regeneration"):
        audit_sequence_sets(package)


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


def test_fairness_rejects_serialized_author_assertions_before_torch_import():
    rows, inventories = _fairness_fixture()
    with pytest.raises(PrefixRouteFairnessError, match="live"):
        derive_fairness_audit(rows)
    with pytest.raises(TypeError):
        derive_fairness_audit(rows, inventories)
    assert inspect.signature(derive_fairness_audit).parameters.keys() == {
        "runtimes"
    }


def test_fairness_budget_records_cannot_pass_without_live_runtime(tmp_path):
    execution = _write_json(
        tmp_path,
        "fairness/execution.json",
        {
            "schema_version": EXECUTION_TRACE_SCHEMA,
            "arm": "B2",
            "events": [
                {
                    "optimizer_event_index": index,
                    "gradient_accumulation_steps": 2,
                    "effective_token_count": 100,
                    "status": "APPLIED_FINITE",
                    "input_batches_sha256": "a" * 64,
                    "model_state_before_sha256": f"{index + 1:064x}",
                    "model_state_after_sha256": f"{index + 2:064x}",
                    "optimizer_state_after_sha256": "b" * 64,
                }
                for index in range(3)
            ],
        },
    )
    calibration = _write_json(
        tmp_path,
        "fairness/calibration.json",
        {
            "schema_version": CALIBRATION_MANIFEST_SCHEMA,
            "video_ids": [f"video_{index:03d}" for index in range(40)],
        },
    )
    trials = _write_json(
        tmp_path,
        "fairness/trials.json",
        {
            "schema_version": TRIAL_MANIFEST_SCHEMA,
            "trial_ids": ["trial_00", "trial_01"],
        },
    )
    seeds = _write_json(
        tmp_path,
        "fairness/seeds.json",
        {
            "schema_version": SEED_MANIFEST_SCHEMA,
            "seeds": [705, 706, 707],
        },
    )
    adapter = ArmRuntimeAdapter(
        arm="B2",
        model=None,
        optimizer=None,
        optimizer_event_forwards=lambda: None,
        inference_forward=lambda: None,
        reset_runtime_state=lambda: None,
        budget_evidence_root=tmp_path,
        execution_trace_reference=execution,
        calibration_manifest_reference=calibration,
        trial_manifest_reference=trials,
        seed_manifest_reference=seeds,
    )
    with pytest.raises(PrefixRouteFairnessError, match="standalone"):
        derive_runtime_budget_evidence(adapter)
    budget, _, _ = fairness_module._runtime_budget(adapter)
    assert budget["optimizer_event_count"] == 3
    assert budget["effective_token_count"] == 300

    broken = json.loads(
        (tmp_path / execution["path"]).read_text(encoding="utf-8")
    )
    broken["events"][1]["model_state_before_sha256"] = "f" * 64
    broken_reference = _write_json(
        tmp_path,
        "fairness/broken-execution.json",
        broken,
    )
    broken_adapter = ArmRuntimeAdapter(
        arm="B2",
        model=None,
        optimizer=None,
        optimizer_event_forwards=lambda: None,
        inference_forward=lambda: None,
        reset_runtime_state=lambda: None,
        budget_evidence_root=tmp_path,
        execution_trace_reference=broken_reference,
        calibration_manifest_reference=calibration,
        trial_manifest_reference=trials,
        seed_manifest_reference=seeds,
    )
    with pytest.raises(PrefixRouteFairnessError, match="model-state chain"):
        fairness_module._runtime_budget(broken_adapter)
    with pytest.raises(PrefixRouteFairnessError, match="standalone"):
        derive_runtime_budget_evidence(broken_adapter)


def _b2_decoder_rows(previous_tracks, *, complete_previous):
    propagated = [
        {
            "pool": "propagated",
            "slot": slot,
            "track_id": track["track_id"],
            "active_score": 0.9,
            "completion_score": 0.9 if complete_previous else 0.1,
            "class_score": 0.9,
            "start": 0,
            "end": 8,
            "label": "A",
            "state_ref": f"updated-{slot}",
        }
        for slot, track in enumerate(previous_tracks)
    ]
    newborn = [
        {
            "pool": "newborn",
            "slot": slot,
            "track_id": None,
            "active_score": 0.9,
            "completion_score": 0.1,
            "class_score": 0.9,
            "start": 8,
            "end": 16,
            "label": "B",
            "state_ref": f"newborn-{slot}",
        }
        for slot in range(NEWBORN_QUERY_COUNT)
    ]
    return propagated + newborn


def test_b2_risk_set_keeps_delayed_first_emission_target():
    rows = instance_aware_risk_set(
        [
            {
                "instance_id": "g0",
                "start_frame": 0,
                "end_observation_count": 8,
                "label": "A",
            }
        ],
        previous_observation_count=8,
        decision_observation_count=16,
    )
    assert rows[0]["delayed_unemitted_completion"] is True
    assert rows[0]["first_emission_target"] == 1
    assert (
        instance_aware_risk_set(
            [
                {
                    "instance_id": "g0",
                    "start_frame": 0,
                    "end_observation_count": 8,
                    "label": "A",
                }
            ],
            previous_observation_count=8,
            decision_observation_count=16,
            emitted_instance_ids=("g0",),
        )
        == []
    )


def test_b2_training_assignment_locks_tracks_then_matches_newborn_one_to_one():
    risk_set = instance_aware_risk_set(
        [
            {
                "instance_id": "g0",
                "start_frame": 0,
                "end_observation_count": 16,
                "label": "A",
            },
            {
                "instance_id": "g1",
                "start_frame": 8,
                "end_observation_count": 24,
                "label": "B",
            },
        ],
        previous_observation_count=8,
        decision_observation_count=16,
    )
    costs = {
        slot: {"g1": 1.0 if slot == 3 else 10.0}
        for slot in range(NEWBORN_QUERY_COUNT)
    }
    result = build_temporal_tracklet_assignment(
        risk_set=risk_set,
        previous_track_assignments={"track-0": "g0"},
        newborn_costs=costs,
    )
    assert result["propagated_assignment"] == {"track-0": "g0"}
    assert result["newborn_assignment"][3] == "g1"
    assert sum(
        instance_id != "DUSTBIN"
        for instance_id in result["newborn_assignment"].values()
    ) == 1


def test_b2_hungarian_contains_optional_dustbins_during_optimization():
    risk_set = [{"instance_id": "A"}, {"instance_id": "B"}]
    costs = {}
    for slot in range(NEWBORN_QUERY_COUNT):
        if slot == 0:
            costs[slot] = {"A": 0.0, "B": 10.0}
        elif slot == 1:
            costs[slot] = {"A": 1.0, "B": 100.0}
        else:
            costs[slot] = {"A": 100.0, "B": 100.0}
    assignment = build_temporal_tracklet_assignment(
        risk_set=risk_set,
        previous_track_assignments={},
        newborn_costs=costs,
    )
    assert assignment["newborn_assignment"][0] == "A"
    assert all(
        instance_id != "B"
        for instance_id in assignment["newborn_assignment"].values()
    )


def test_b2_executes_all_newborn_queries_but_cannot_reuse_released_slots():
    previous = sorted(
        [
            {
                "track_id": f"track-{index:02d}",
                "state_ref": f"state-{index}",
                "started_at_bin": 0,
                "last_decision_bin": 0,
            }
            for index in range(TRACK_CAPACITY)
        ],
        key=lambda row: row["track_id"],
    )
    result = temporal_motr_transition(
        stream_key="v0",
        decision_bin=1,
        decision_observation_count=16,
        calibrated_threshold=0.5,
        previous_tracks=previous,
        decoder_rows=_b2_decoder_rows(
            previous,
            complete_previous=True,
        ),
        next_track_serial=64,
        next_emission_sequence=0,
    )
    assert result["query_counts"]["newborn"] == NEWBORN_QUERY_COUNT
    assert result["query_counts"]["decoder_total"] == 128
    assert result["predecision_free_track_slots"] == 0
    assert result["released_slots_reusable_same_decision"] is False
    assert len(result["emissions"]) == TRACK_CAPACITY
    assert result["next_tracks"] == []

    next_result = temporal_motr_transition(
        stream_key="v0",
        decision_bin=2,
        decision_observation_count=24,
        calibrated_threshold=0.5,
        previous_tracks=[],
        decoder_rows=_b2_decoder_rows([], complete_previous=False),
        next_track_serial=result["next_track_serial"],
        next_emission_sequence=result["next_emission_sequence"],
    )
    assert len(next_result["next_tracks"]) == TRACK_CAPACITY


def test_b2_newborn_with_unconfirmed_class_remains_active():
    rows = _b2_decoder_rows([], complete_previous=False)
    rows[0]["completion_score"] = 0.9
    rows[0]["class_score"] = 0.4
    result = temporal_motr_transition(
        stream_key="v0",
        decision_bin=1,
        decision_observation_count=16,
        calibrated_threshold=0.5,
        previous_tracks=[],
        decoder_rows=rows,
        next_track_serial=0,
        next_emission_sequence=0,
    )
    assert len(result["next_tracks"]) == TRACK_CAPACITY
    assert any(
        track["state_ref"] == "newborn-0" for track in result["next_tracks"]
    )
    with pytest.raises(PrefixRouteB2Error, match="observation coordinate"):
        temporal_motr_transition(
            stream_key="v0",
            decision_bin=999,
            decision_observation_count=8,
            calibrated_threshold=0.5,
            previous_tracks=[],
            decoder_rows=_b2_decoder_rows([], complete_previous=False),
            next_track_serial=0,
            next_emission_sequence=0,
        )


def test_r6_public_entry_is_canonical_source_bound_and_currently_blocked(tmp_path):
    reference = {"path": "missing.json", "sha256": "0" * 64}
    with pytest.raises(PrefixRouteR6Error, match="source registration"):
        evaluate_r6_raw_evidence(
            reference,
            bundle_root=tmp_path,
            population_request_reference=reference,
            r0_request_reference=reference,
            r0_detail_reference=reference,
            fairness_audit_reference=reference,
            review_attestation_path=tmp_path / "missing-review.json",
            review_signature_path=tmp_path / "missing-review.sig",
        )
    parameters = inspect.signature(evaluate_r6_raw_evidence).parameters
    assert set(parameters) == {
        "raw_evidence_reference",
        "bundle_root",
        "population_request_reference",
        "r0_request_reference",
        "r0_detail_reference",
        "fairness_audit_reference",
        "review_attestation_path",
        "review_signature_path",
    }


def test_r6_control_construction_binds_source_population_and_emissions(
    tmp_path,
):
    protocol_record = {"sha256": "a" * 64}
    manifest_record = {"sha256": "b" * 64}
    raw_source = {
        "decision_bins": [0, 1, 2],
        "event_count_prior": 1,
        "class_prior": [{"label": "A", "probability": 1.0}],
    }
    raw_source_reference = _write_json(
        tmp_path,
        "r6/count-only-v0-source.json",
        raw_source,
    )
    constructed_reference = _write_json(
        tmp_path,
        "r6/count-only-v0-constructed.json",
        {
            "video_id": "v0",
            "scheduled_bins": [0],
            "labels": ["A"],
        },
    )
    source_reference = _write_json(
        tmp_path,
        "r6/count-only-source.json",
        {
            "schema_version": r6_module.R6_CONTROL_TRANSCRIPT_SCHEMA,
            "protocol_sha256": protocol_record["sha256"],
            "source_manifest_sha256": manifest_record["sha256"],
            "control": "COUNT_ONLY",
            "seed": 705,
            "reporting_video_ids": ["v0"],
            "records": [
                {
                    "video_id": "v0",
                    "source_artifact": raw_source_reference,
                    "constructed_artifact": constructed_reference,
                    "source_sha256": raw_source_reference["sha256"],
                    "constructed_sha256": constructed_reference["sha256"],
                }
            ],
        },
    )
    source_path = tmp_path / source_reference["path"]
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    emissions_sha256 = "e" * 64
    construction = {
        "schema_version": r6_module.R6_CONTROL_CONSTRUCTION_SCHEMA,
        "protocol_sha256": protocol_record["sha256"],
        "source_manifest_sha256": manifest_record["sha256"],
        "control": "COUNT_ONLY",
        "seed": 705,
        "algorithm": "fit_core_count_and_class_priors_plus_decision_bin",
        "parameters": {
            "feature_access": False,
            "sources": [
                "fit_core_event_count_prior",
                "fit_core_class_prior",
                "decision_bin_index",
            ],
        },
        "reporting_video_ids_sha256": hashlib.sha256(
            canonical_json_bytes(["v0"])
        ).hexdigest(),
        "source_artifact": source_reference,
        "source_artifact_sha256": source_sha256,
        "emissions_sha256": emissions_sha256,
    }
    construction_reference = _write_json(
        tmp_path,
        "r6/count-only-construction.json",
        construction,
    )
    expected = hashlib.sha256(
        (tmp_path / construction_reference["path"]).read_bytes()
    ).hexdigest()
    assert r6_module._validate_control_construction(
        construction_reference,
        bundle_root=tmp_path,
        arm="COUNT_ONLY",
        seed=705,
        video_ids=["v0"],
        protocol_record=protocol_record,
        manifest_record=manifest_record,
        emission_ledger_sha256=emissions_sha256,
    ) == expected

    construction["algorithm"] = "self_authored_control"
    broken_reference = _write_json(
        tmp_path,
        "r6/count-only-broken.json",
        construction,
    )
    with pytest.raises(PrefixRouteR6Error, match="identity differs"):
        r6_module._validate_control_construction(
            broken_reference,
            bundle_root=tmp_path,
            arm="COUNT_ONLY",
            seed=705,
            video_ids=["v0"],
            protocol_record=protocol_record,
            manifest_record=manifest_record,
            emission_ledger_sha256=emissions_sha256,
        )


def _r6_interval(lower, upper, margin=0.01):
    return {
        "orientation": "positive_means_left_arm_is_better",
        "estimate": (lower + upper) / 2.0,
        "lower": lower,
        "upper": upper,
        "practical_margin": margin,
    }


def _r6_unit_margin(metric):
    if metric == "mOnlineAP":
        return 0.005
    if metric == "class_mOnlineAP":
        return 0.05
    if metric == "endpoint_latency_bins":
        return 1.0
    if metric in r6_module.STRESS_METRICS:
        return 0.02
    return 0.01


def _r6_global_intervals(lower=0.1, upper=0.2):
    return {
        metric: _r6_interval(lower, upper, _r6_unit_margin(metric))
        for metric in r6_module.GLOBAL_METRICS
    }


def _complete_r6_inference():
    eligible = []
    metrics = r6_module.GLOBAL_METRICS + r6_module.CONTROL_METRICS
    margins = {metric: _r6_unit_margin(metric) for metric in metrics}
    comparison_count = len(REQUIRED_CONTRASTS) * len(metrics)
    alpha_each = 0.05 / comparison_count
    return {
        "schema_version": "prefix-route-r6-inference-v2",
        "method": "paired_crossed_video_and_global_seed_percentile_bootstrap",
        "fixed_population_estimand": True,
        "mOnlineAP_resampling_unit": "paired_global_training_seed",
        "additive_metric_resampling_units": [
            "video",
            "paired_global_training_seed",
        ],
        "resamples": 10000,
        "seed": 2026071707,
        "family_wise_alpha": 0.05,
        "contrast_count": len(REQUIRED_CONTRASTS),
        "metric_count": len(metrics),
        "simultaneous_comparison_count": comparison_count,
        "per_comparison_two_sided_alpha": alpha_each,
        "lower_probability": alpha_each / 2.0,
        "upper_probability": 1.0 - alpha_each / 2.0,
        "eligible_stress_families": eligible,
        "margins": margins,
        "arm_estimates": {
            arm: {metric: 0.0 for metric in metrics}
            for arm in r6_module.INFERENCE_ARMS
        },
        "intervals": {
            f"{left}_vs_{right}": {
                metric: _r6_interval(0.1, 0.2, margins[metric])
                for metric in metrics
            }
            for left, right in REQUIRED_CONTRASTS
        },
    }


def test_r6_terminal_rejects_arbitrary_intervals_and_mutable_resample_globals(
    monkeypatch,
):
    with pytest.raises(PrefixRouteR6Error, match="sealed raw cells"):
        r6_module._terminal_route_decision(
            {"intervals": {"B4_vs_B2": _r6_global_intervals()}}
        )
    monkeypatch.setattr(r6_module, "BOOTSTRAP_RESAMPLES", 1)
    valid = _complete_r6_inference()
    with pytest.raises(PrefixRouteR6Error, match="sealed raw cells"):
        r6_module._terminal_route_decision(valid)


def test_r6_negative_control_equivalence_kills_the_benchmark():
    inference = _complete_r6_inference()
    for arm in ("COUNT_ONLY", "TEMPLATE_TIMING", "LEDGER_ONLY"):
        inference["intervals"][f"B4_vs_{arm}"].update(
            {
                metric: _r6_interval(
                    -0.005,
                    0.005,
                    _r6_unit_margin(metric),
                )
                for metric in r6_module.GLOBAL_METRICS
            }
        )
    with pytest.raises(PrefixRouteR6Error, match="sealed raw cells"):
        r6_module._terminal_route_decision(inference)


def test_r6_simple_control_dominance_kills_the_benchmark():
    inference = _complete_r6_inference()
    inference["intervals"]["B4_vs_COUNT_ONLY"].update(
        _r6_global_intervals(-0.2, -0.1)
    )
    with pytest.raises(PrefixRouteR6Error, match="sealed raw cells"):
        r6_module._terminal_route_decision(inference)


def test_r6_any_clear_global_inferiority_to_b2_kills_the_route():
    inference = _complete_r6_inference()
    inference["intervals"]["B4_vs_B2"]["duplicate_per_gt"] = _r6_interval(
        -0.2,
        -0.1,
    )
    with pytest.raises(PrefixRouteR6Error, match="sealed raw cells"):
        r6_module._terminal_route_decision(inference)


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


def test_r6_per_video_cells_require_a_verified_immutable_ledger(tmp_path):
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
    with pytest.raises(PrefixRouteR6Error, match="verified immutable ledger"):
        derive_per_video_cell(
            arm="B4",
            seed=705,
            video_id="v0",
            ground_truth=ground_truth,
            emissions=emissions,
            stress_ground_truth_ids={"direct_complete": ["g0"]},
            fps=30.0,
        )

    run_binding = "f" * 64
    ledger = ImmutableEventLedger()
    ledger.append(
        {
            "event_id": "e0",
            "stream_id": "v0",
            "stream_key": "v0",
            "video_id": "v0",
            "immutable": True,
            "slot_id": 0,
            "label": "A",
            "score": 0.9,
            "start_frame": 0,
            "end_frame": 8,
            "emit_frame": 8,
            "source_frame": 8,
            "provenance_digest": run_binding,
            "segment": [0.0, 8.0 / 30.0],
            "fps": 30.0,
        }
    )
    ledger_path = tmp_path / "emissions.jsonl"
    commitment_path = tmp_path / "emissions.commitment.json"
    persist_verified_ledger(
        ledger_path,
        commitment_path,
        ledger.rows,
    )
    verified = r6_module._load_verified_run_emissions(
        ledger_reference=bundle_file_reference(ledger_path, tmp_path),
        commitment_reference=bundle_file_reference(commitment_path, tmp_path),
        bundle_root=tmp_path,
        expected_video_ids=["v0"],
        run_binding_sha256=run_binding,
    )
    batch = r6_module._canonical_video_emissions(
        verified["batches_by_video"]["v0"],
        "v0",
        16,
    )
    cell = derive_per_video_cell(
        arm="B4",
        seed=705,
        video_id="v0",
        ground_truth=ground_truth,
        emissions=batch,
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


def test_r1_execution_runtime_rejects_opaque_command_before_import():
    source_records = {
        "resolved_command": (
            Path("command.txt"),
            b"python extractor.py --local-only\n",
            "0" * 64,
        ),
        "software_versions": (
            Path("software.json"),
            canonical_json_bytes(
                {
                    "schema_version": R1_SOFTWARE_SCHEMA,
                    "python": "fixture",
                    "packages": {
                        "numpy": "fixture",
                        "opencv": "fixture",
                        "torch": "fixture",
                        "transformers": "fixture",
                    },
                }
            ),
            "1" * 64,
        ),
    }
    binding = {
        "device": "cpu",
        "image_size": 224,
        "batch_size": 64,
        "local_files_only": True,
    }
    with pytest.raises(PrefixRouteR1Error, match="valid UTF-8 JSON"):
        r1_module._validate_execution_runtime(
            source_records,
            binding,
            {"hf_snapshot_revision": "fixture"},
        )
    assert R1_COMMAND_SCHEMA == "prefix-route-r1-resolved-command-v2"


def test_r1_core_reexecutes_real_decode_cache_and_dynamic_transcript(tmp_path):
    cv2 = pytest.importorskip("cv2")
    video_path = tmp_path / "v0.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (8, 8),
    )
    assert writer.isOpened()
    for value in (20, 80, 140, 220):
        writer.write(np.full((8, 8, 3), value, dtype=np.uint8))
    writer.release()
    decoded = r1_module._decode_rgb_video(video_path, image_size=8)

    def encode_batch(batch):
        batch = np.asarray(batch, dtype=np.float32)
        return np.stack(
            (
                batch.mean(axis=(1, 2, 3)),
                batch[:, 0, 0, 0],
            ),
            axis=1,
        ).astype(np.float32)

    feature_path = tmp_path / "v0.npy"
    np.save(feature_path, encode_batch(decoded))
    tokens = [
        {
            "token_index": index,
            "source_frame": index,
            "decision_frame": index,
            "support_frames": [index],
        }
        for index in range(len(decoded))
    ]
    support = {
        "v0": {
            "video_id": "v0",
            "frame_count": len(decoded),
            "tokens": tokens,
            "source_frames": list(range(len(decoded))),
        }
    }
    selection = [
        {"category": "first_token", "video_id": "v0", "token_index": 0}
    ]
    supplied = {
        "schema_version": R1_DYNAMIC_SCHEMA,
        "audit_seed": 2026071702,
        "records": [
            run_dynamic_token_record(
                encode_batch=encode_batch,
                selected_rgb_frames=decoded,
                video_id="v0",
                token_index=0,
                category="first_token",
            )
        ],
    }
    result = r1_module._recompute_cache_and_dynamic(
        encode_batch=encode_batch,
        binding={"image_size": 8, "batch_size": 2},
        canonical_ids=("v0",),
        raw_paths={"v0": video_path},
        feature_paths={"v0": feature_path},
        support_by_id=support,
        expected_selection=selection,
        supplied_dynamic=supplied,
    )
    assert result["cache_rows_byte_identical_to_reexecution"] is True
    assert result["dynamic_transcript_byte_identical_to_reexecution"] is True

    tampered = np.load(feature_path)
    tampered[0, 0] += 1.0
    np.save(feature_path, tampered)
    with pytest.raises(PrefixRouteR1Error, match="cache rows"):
        r1_module._recompute_cache_and_dynamic(
            encode_batch=encode_batch,
            binding={"image_size": 8, "batch_size": 2},
            canonical_ids=("v0",),
            raw_paths={"v0": video_path},
            feature_paths={"v0": feature_path},
            support_by_id=support,
            expected_selection=selection,
            supplied_dynamic=supplied,
        )


def test_r1_unregistered_binding_blocks_fabricated_bytes_before_pass(tmp_path):
    request, canonical_ids = _r1_fixture(tmp_path)
    protocol_r1 = load_protocol(PROTOCOL_PATH)["protocol"]["r1"]
    with pytest.raises(PrefixRouteR1Error, match="not registered"):
        validate_r1_bundle(
            request,
            bundle_root=tmp_path,
            protocol_id=request["protocol_id"],
            protocol_sha256=request["protocol_sha256"],
            review_attestation_sha256="a" * 64,
            canonical_video_ids=canonical_ids,
            protocol_r1=protocol_r1,
        )


def test_r1_unregistered_binding_is_recordable_failure(tmp_path):
    request, canonical_ids = _r1_fixture(tmp_path)
    protocol_r1 = load_protocol(PROTOCOL_PATH)["protocol"]["r1"]
    result = derive_r1_status(
        request,
        bundle_root=tmp_path,
        protocol_id=request["protocol_id"],
        protocol_sha256=request["protocol_sha256"],
        review_attestation_sha256="a" * 64,
        canonical_video_ids=canonical_ids,
        protocol_r1=protocol_r1,
    )
    assert result["status"] == "FAIL_UNVERIFIABLE"
    assert "not registered" in result["failure_reasons"][0]


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
