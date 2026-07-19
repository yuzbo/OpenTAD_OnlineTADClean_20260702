from __future__ import annotations

import hashlib
import inspect
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import opentad.evaluations.prefix_route_r6_v2 as r6_module
from opentad.evaluations.prefix_route_r6_v2 import PrefixRouteR6Error
from opentad.utils.evidence_bundle import bundle_file_reference
from opentad.utils.immutable_event_ledger import (
    ImmutableEventLedger,
    canonical_json as ledger_canonical_json,
    persist_verified_ledger,
)
from opentad.utils.prefix_route_artifacts_v2 import (
    PrefixRouteArtifactError,
    canonical_json_bytes as artifact_json_bytes,
    tensor_state_digest_from_npz_bytes,
    verify_optimizer_artifact_trace,
)
from opentad.utils.prefix_route_b2_contract_v2 import (
    B2StreamMachine,
    NEWBORN_QUERY_COUNT,
    PrefixRouteB2Error,
    build_temporal_tracklet_assignment,
)
import opentad.utils.prefix_route_fairness_v2 as fairness_module
from opentad.utils.prefix_route_fairness_v2 import (
    CORE_ARMS,
    FAIRNESS_SCHEMA,
    PrefixRouteFairnessError,
    validate_fairness_audit_record,
)
from opentad.utils.prefix_route_formal_v2 import (
    PrefixRouteFormalError,
    required_parent_initializers,
    verify_runtime_import_closure,
)
from opentad.utils.prefix_route_protocol_v2 import (
    PrefixRouteProtocolV2Error,
    canonical_sha256,
    validate_protocol,
)
from opentad.utils.prefix_route_r0_v2 import collect_r0_census


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


def _write_bytes(root, relative, payload):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return bundle_file_reference(path, root)


def _write_json(root, relative, value):
    return _write_bytes(root, relative, artifact_json_bytes(value))


def _npz_bytes(**arrays):
    output = io.BytesIO()
    np.savez(output, **arrays)
    return output.getvalue()


def _relock(protocol):
    protocol["policy_lock_sha256"] = canonical_sha256(
        {section: protocol[section] for section in POLICY_SECTIONS}
    )
    return protocol


def _protocol_value():
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_low_level_r0_census_never_emits_pass_or_certificate():
    report, detail = collect_r0_census(
        {
            "database": {
                "v0": {
                    "subset": "validation",
                    "duration": 1.0,
                    "frame": 8,
                    "annotations": [],
                }
            }
        },
        ("A",),
        ("v0",),
        bootstrap_resamples=1,
    )
    serialized = json.dumps({"report": report, "detail": detail})
    assert report["evidence_role"] == (
        "DIAGNOSTIC_ONLY_REQUIRES_FORMAL_ENVELOPE"
    )
    assert "status" not in report
    assert "certificate" not in serialized.lower()
    assert "PASS_" not in serialized


def test_registered_source_requires_attested_origin_and_rejects_semantic_fixture():
    missing_origin = _protocol_value()
    registration = missing_origin["population"]["source_registration"]
    registration.update(
        {
            "state": "REGISTERED_IN_FIXED_REVIEWED_PROTOCOL",
            "release_revision": "THUMOS14_TEMPORAL_ANNOTATIONS_RELEASE_2014",
            "authoritative_annotation_sha256": "a" * 64,
            "historical_inventory_sha256": "b" * 64,
            "source_origin_attestation_sha256": "UNREGISTERED",
            "official_video_manifest_sha256": "d" * 64,
            "registration_commit": "1" * 40,
        }
    )
    missing_origin["governance"]["post_pass_scope"] = [
        "READ_ONLY_R0_ANNOTATION_CENSUS",
        "READ_ONLY_R1_CACHE_CAUSALITY_AUDIT",
    ]
    _relock(missing_origin)
    with pytest.raises(PrefixRouteProtocolV2Error, match="source-origin"):
        validate_protocol(missing_origin)

    provisional = _protocol_value()
    provisional_registration = provisional["population"]["source_registration"]
    provisional_registration.update(
        {
            "state": "REGISTERED_IN_FIXED_REVIEWED_PROTOCOL",
            "release_revision": "fixture-release-1",
            "authoritative_annotation_sha256": "a" * 64,
            "historical_inventory_sha256": "b" * 64,
            "source_origin_attestation_sha256": "c" * 64,
            "official_video_manifest_sha256": "d" * 64,
            "registration_commit": "1" * 40,
        }
    )
    provisional["governance"]["post_pass_scope"] = [
        "READ_ONLY_R0_ANNOTATION_CENSUS",
        "READ_ONLY_R1_CACHE_CAUSALITY_AUDIT",
    ]
    _relock(provisional)
    with pytest.raises(
        PrefixRouteProtocolV2Error,
        match="placeholder|provisional",
    ):
        validate_protocol(provisional)


def _forged_fairness_record():
    rows = {}
    for arm in CORE_ARMS:
        rows[arm] = {
            "arm": arm,
            "trainable_parameters": 100,
            "train_macs_per_optimizer_event": 1000.0,
            "inference_macs_per_decision": 100.0,
            "live_causal_state_bytes": 400.0,
            "peak_training_memory_bytes": 10000.0,
            "peak_inference_memory_bytes": 2000.0,
            "latency_median_ms": 4.0,
            "latency_p95_ms": 6.0,
            "profiled_model_artifact_reference": {
                "path": "forged-model.npz",
                "sha256": "e" * 64,
            },
            "profiled_model_artifact_sha256": "e" * 64,
            "profiled_model_tensor_state_sha256": "f" * 64,
            "profiled_model_tensor_count": 1,
            "train_profiler_events": [
                {"key": "train", "count": 1, "flops": 2000}
            ],
            "inference_profiler_events": [
                {"key": "inference", "count": 1, "flops": 200}
            ],
            "latency_samples_ms": [4.0] * 94 + [6.0] * 6,
            "cuda_identity": {
                "device_index": 0,
                "device_name": "forged-cuda",
                "total_memory_bytes": 1,
                "compute_capability": [9, 0],
                "torch_version": "forged",
                "cuda_version": "forged",
            },
            "optimizer_event_count": 10,
            "effective_token_count": 1000,
            "gradient_accumulation_steps": 2,
            "hyperparameter_trial_count": 3,
            "calibration_video_count": 40,
            "seed_count": 3,
            "budget_source_sha256": {
                "execution_trace": "a" * 64,
                "calibration_manifest": "b" * 64,
                "trial_manifest": "c" * 64,
                "seed_manifest": "d" * 64,
            },
            "budget_source_references": {
                "execution_trace": {"path": "forged", "sha256": "a" * 64},
                "calibration_manifest": {"path": "forged", "sha256": "b" * 64},
                "trial_manifest": {"path": "forged", "sha256": "c" * 64},
                "seed_manifest": {"path": "forged", "sha256": "d" * 64},
            },
            "parameter_gradient_inventory": [
                {
                    "name": "head.weight",
                    "numel": 100,
                    "gradient_l1": 1.0,
                }
            ],
        }
    return {
        "schema_version": FAIRNESS_SCHEMA,
        "evidence_source": (
            "LIVE_CLEAN_PROCESS_MODEL_OPTIMIZER_GRADIENT_CUDA_PROFILER_AND_"
            "ARTIFACT_DERIVED_TENSOR_STATE"
        ),
        "mac_definition": "torch_profiler_flops_divided_by_two",
        "serialized_scalar_or_boolean_input_allowed": False,
        "anchor_arm": "B2",
        "warmup_decisions": 20,
        "timed_decisions": 100,
        "parameter_relative_tolerance": 0.05,
        "resource_relative_tolerance": 0.10,
        "latency_upper_ratio": 1.10,
        "rows": rows,
        "checks": fairness_module._derive_checks(rows),
        "status": "PASS_BOTH_CAPACITY_AND_RESOURCE_MATCHED",
    }


def test_forged_serialized_fairness_pass_never_authorizes_r6(tmp_path):
    result = validate_fairness_audit_record(
        _forged_fairness_record(),
        bundle_root=tmp_path,
    )
    assert result["verification_status"] == "UNVERIFIED_SERIALIZED_AUDIT"
    assert result["claimed_status"] == (
        "PASS_BOTH_CAPACITY_AND_RESOURCE_MATCHED"
    )
    assert result["formal_fairness_pass"] is False
    with pytest.raises(PrefixRouteFairnessError, match="live clean-process"):
        fairness_module.require_live_fairness_capability(
            _forged_fairness_record()
        )


def test_arbitrary_checkpoint_bytes_are_not_tensor_state_evidence():
    with pytest.raises(PrefixRouteArtifactError, match="safe NPZ"):
        tensor_state_digest_from_npz_bytes(b"not-a-checkpoint")


def test_no_learning_control_cannot_hide_model_state_mutation(tmp_path):
    initial_model = _write_bytes(
        tmp_path,
        "artifacts/model-initial.npz",
        _npz_bytes(weight=np.asarray([1.0], dtype=np.float32)),
    )
    final_model = _write_bytes(
        tmp_path,
        "artifacts/model-final.npz",
        _npz_bytes(weight=np.asarray([2.0], dtype=np.float32)),
    )
    empty_optimizer = _npz_bytes()
    initial_optimizer = _write_bytes(
        tmp_path,
        "artifacts/optimizer-initial.npz",
        empty_optimizer,
    )
    final_optimizer = _write_bytes(
        tmp_path,
        "artifacts/optimizer-final.npz",
        empty_optimizer,
    )
    trace = _write_json(
        tmp_path,
        "artifacts/optimizer-trace.json",
        {
            "schema_version": "prefix-route-optimizer-artifact-trace-v2",
            "arm": "COUNT_ONLY",
            "seed": 705,
            "learning": False,
            "events": [],
            "final_event_sha256": "0" * 64,
        },
    )
    with pytest.raises(PrefixRouteArtifactError, match="changed model"):
        verify_optimizer_artifact_trace(
            trace,
            bundle_root=tmp_path,
            arm="COUNT_ONLY",
            seed=705,
            initial_model_reference=initial_model,
            final_model_reference=final_model,
            initial_optimizer_reference=initial_optimizer,
            final_optimizer_reference=final_optimizer,
            learning=False,
        )


def test_fairness_checkpoint_state_must_match_profiled_model(tmp_path):
    class FakeTensor:
        def __init__(self, value):
            self.value = np.asarray(value, dtype=np.float32)

        def detach(self):
            return self

        def contiguous(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class FakeModel:
        @staticmethod
        def state_dict():
            return {
                "weight": FakeTensor([[1.0, 2.0]]),
                "bias": FakeTensor([3.0]),
            }

    class FakeTorch:
        @staticmethod
        def is_tensor(value):
            return isinstance(value, FakeTensor)

    wrong_artifact = _write_bytes(
        tmp_path,
        "fairness/wrong-model.npz",
        _npz_bytes(
            weight=np.zeros((1, 2), dtype=np.float32),
            bias=np.zeros((1,), dtype=np.float32),
        ),
    )
    with pytest.raises(PrefixRouteFairnessError, match="profiled model tensors"):
        fairness_module._verify_model_tensor_artifact(
            FakeModel(),
            FakeTorch,
            reference=wrong_artifact,
            bundle_root=tmp_path,
            arm="B2",
        )


def _ledger_event(event_id, sequence_frame, run_binding):
    return {
        "event_id": event_id,
        "stream_id": "v0",
        "stream_key": "v0",
        "video_id": "v0",
        "immutable": True,
        "slot_id": sequence_frame,
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


def test_r6_rejects_unhashed_and_reordered_emission_rows(tmp_path):
    run_binding = "f" * 64
    plain_ledger = _write_json(
        tmp_path,
        "ledger/plain.json",
        [_ledger_event("plain", 0, run_binding)],
    )
    plain_commitment = _write_json(
        tmp_path,
        "ledger/plain.commitment.json",
        {"claimed": True},
    )
    with pytest.raises(PrefixRouteR6Error, match="ledger verification"):
        r6_module._load_verified_run_emissions(
            ledger_reference=plain_ledger,
            commitment_reference=plain_commitment,
            bundle_root=tmp_path,
            expected_video_ids=["v0"],
            run_binding_sha256=run_binding,
        )

    ledger = ImmutableEventLedger()
    ledger.append(_ledger_event("e0", 0, run_binding))
    ledger.append(_ledger_event("e1", 1, run_binding))
    ledger_path = tmp_path / "ledger" / "emissions.jsonl"
    commitment_path = tmp_path / "ledger" / "emissions.commitment.json"
    persist_verified_ledger(ledger_path, commitment_path, ledger.rows)
    commitment_reference = bundle_file_reference(commitment_path, tmp_path)
    reordered = "".join(
        ledger_canonical_json(row) + "\n"
        for row in reversed(ledger.rows)
    ).encode("utf-8")
    reordered_reference = _write_bytes(
        tmp_path,
        "ledger/reordered.jsonl",
        reordered,
    )
    with pytest.raises(PrefixRouteR6Error, match="ledger verification"):
        r6_module._load_verified_run_emissions(
            ledger_reference=reordered_reference,
            commitment_reference=commitment_reference,
            bundle_root=tmp_path,
            expected_video_ids=["v0"],
            run_binding_sha256=run_binding,
        )


def test_r6_does_not_upgrade_caller_immutable_flags_to_verified_rows():
    forged = [
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
            "immutable": True,
        }
    ]
    with pytest.raises(PrefixRouteR6Error, match="verified immutable ledger"):
        r6_module.derive_per_video_cell(
            arm="B4",
            seed=705,
            video_id="v0",
            ground_truth=[],
            emissions=forged,
            stress_ground_truth_ids={},
        )
    assert '"immutable": True' not in inspect.getsource(
        r6_module.derive_per_video_cell
    )


def test_r6_bootstrap_sealer_rejects_caller_cells():
    with pytest.raises(PrefixRouteR6Error, match="verified formal R6 evidence"):
        r6_module._seal_bootstrap_output(
            {
                "raw_evidence": {},
                "per_video_cells": [],
                "dataset_metrics": [],
            }
        )


def test_live_r6_is_private_and_requires_clean_process_capability(tmp_path):
    assert "evaluate_r6_live_evidence" not in r6_module.__all__
    with pytest.raises(PrefixRouteR6Error, match="clean-process capability"):
        r6_module._evaluate_r6_entry(
            None,
            bundle_root=tmp_path,
            population_request_reference=None,
            r0_request_reference=None,
            r0_detail_reference=None,
            fairness_capability=object(),
            formal_process_capability=None,
            review_attestation_path=tmp_path / "review.json",
            review_signature_path=tmp_path / "review.sig",
        )


def test_b2_tie_collision_has_one_frozen_lexicographic_solution():
    risk_set = [{"instance_id": value} for value in ("a", "b", "c")]
    costs = {
        slot: {instance["instance_id"]: 0.0 for instance in risk_set}
        for slot in range(NEWBORN_QUERY_COUNT)
    }
    result = build_temporal_tracklet_assignment(
        risk_set=risk_set,
        previous_track_assignments={},
        newborn_costs=costs,
    )
    assert [result["newborn_assignment"][slot] for slot in range(3)] == [
        "a",
        "b",
        "c",
    ]
    assert all(
        result["newborn_assignment"][slot] == "DUSTBIN"
        for slot in range(3, NEWBORN_QUERY_COUNT)
    )
    assert result["tie_break"] == (
        "exact_primary_integer_then_rowwise_lexicographic_column"
    )


def _decoder_rows(previous_tracks):
    propagated = [
        {
            "pool": "propagated",
            "slot": slot,
            "track_id": track["track_id"],
            "active_score": 0.9,
            "completion_score": 0.1,
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


def test_b2_clock_rejects_repeat_skip_reorder_reset_and_stride_drift():
    with pytest.raises(PrefixRouteB2Error, match="frozen B2 contract"):
        B2StreamMachine(
            stream_key="v0",
            frame_count=16,
            calibrated_threshold=0.5,
            feature_stride_frames=4,
            run_binding_sha256="a" * 64,
        )
    machine = B2StreamMachine(
        stream_key="v0",
        frame_count=16,
        calibrated_threshold=0.5,
        run_binding_sha256="a" * 64,
    )
    first = machine.advance(
        decision_observation_count=8,
        decoder_rows=_decoder_rows([]),
    )
    assert first["state_after"]["next_emission_sequence"] == 0
    for invalid_clock in (8, 24, 4):
        with pytest.raises(PrefixRouteB2Error, match="repeated, skipped"):
            machine.advance(
                decision_observation_count=invalid_clock,
                decoder_rows=_decoder_rows(
                    first["state_after"]["tracks"]
                ),
            )
    machine.advance(
        decision_observation_count=16,
        decoder_rows=_decoder_rows(first["state_after"]["tracks"]),
    )
    with pytest.raises(PrefixRouteB2Error, match="already terminal"):
        machine.advance(
            decision_observation_count=8,
            decoder_rows=_decoder_rows([]),
        )
    assert set(inspect.signature(machine.advance).parameters) == {
        "decision_observation_count",
        "decoder_rows",
    }


def test_import_closure_rejects_missing_parent_initializer(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    initializer = package / "__init__.py"
    leaf = package / "leaf.py"
    initializer.write_text("", encoding="ascii")
    leaf.write_text("VALUE = 1\n", encoding="ascii")
    manifest_record = {
        "manifest": {
            "entries": {
                "package/leaf.py": hashlib.sha256(
                    leaf.read_bytes()
                ).hexdigest(),
            }
        }
    }
    with pytest.raises(PrefixRouteFormalError, match="package initializers"):
        verify_runtime_import_closure(
            repo_root=tmp_path,
            manifest_record=manifest_record,
        )


def test_protocol_source_list_contains_every_existing_parent_initializer():
    protocol = _protocol_value()
    paths = set(protocol["source_bindings"]["required_paths"])
    assert required_parent_initializers(ROOT, paths).issubset(paths)
    assert "opentad/models/__init__.py" in paths
    assert "opentad/models/backbones/__init__.py" in paths


def test_parent_monkeypatch_is_not_inherited_by_formal_runner(monkeypatch, tmp_path):
    sentinel = "PARENT_MONKEYPATCH_EXECUTED"

    def forbidden_terminal(_):
        raise AssertionError(sentinel)

    monkeypatch.setattr(
        r6_module,
        "_terminal_route_decision",
        forbidden_terminal,
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "run_prefix_route_r6_v2.py"),
            "preflight",
            "--review-attestation",
            str(tmp_path / "missing-review.json"),
            "--review-signature",
            str(tmp_path / "missing-review.sig"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert sentinel not in completed.stdout
    assert sentinel not in completed.stderr
    assert "BLOCKED_FORMAL_R6" in completed.stdout
