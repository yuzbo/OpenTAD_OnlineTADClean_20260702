"""Pure-Python integrity contracts for the D1.5 diagnostic finalizer."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from scripts.finalize_eventmatr_d15_owner_counterfactual import (
    CHANNELS,
    PROTOCOL,
    ROUTES,
    _route_next_repair,
    _expect,
    _required_mapping,
    _validate_positive_control,
    _validate_route,
    _validate_source_provenance,
    _validate_trace,
)
from scripts.run_eventmatr_d15_owner_counterfactual import (
    EXPECTED_D14_COUNTS,
    OFFICIAL_TRAIN_ARTIFACTS,
    _route_summary,
    _validate_d14_gate,
    _validate_padding_runtime,
)


D14_COMMIT = "fa27b3657b72c5b713ee3d2a5c0e652e7ca14eb4"
D14_TREE = "7603fc6b8226fa8f9dcb3d5212136fb47631bc32"
TRAINING_COMMIT = "92cf34aa07bebee2a7a7e3661431d5055804b29b"
TRAINING_TREE = "aef4f64bc020df9d39ead9811fbc01407f1c754a"


def _routes() -> dict:
    return {
        channel: {
            route: {
                "counts": {
                    "decision_row_count": 1,
                    "end_argmax_count": 0,
                    "end_min_duration_suppression_count": 0,
                    "target_backed_end_transition_count": 0,
                    "target_backed_end_within_one_segment_count": 0,
                    "end_at_observed_target_end_count": 0,
                    "records_active_after_eos": 0,
                    "videos_with_active_records_after_eos": 0,
                },
                "decision_rows_by_source": {"predicted": 1},
                "first_owner_decision_state_counts": {"0": 1},
                "transition_counts": {
                    (
                        "cancel"
                        if route == "formal"
                        else "diagnostic_retain_cancel"
                    ): 1
                },
                "eos_active_record_counts_by_video": {},
                "eos_birth_counts_by_video": {},
                "linked_unique_target_count": 0,
                "raw_semantic_duplicate_count": 0,
            }
            for route in ROUTES
        }
        for channel in CHANNELS
    }


def _trace_row(channel: str, route: str) -> dict:
    return {
        "protocol": PROTOCOL,
        "channel": channel,
        "route": route,
        "video_name": "video_test_0000001",
        "runtime_event_id": 0,
        "diagnostic_target_event_id": None,
        "diagnostic_target_end_frame": None,
        "source": "predicted",
        "creation_frame": 0.0,
        "current_frame": 4.0,
        "is_eos": False,
        "first_owner_decision": True,
        "owner_query_before_refresh": 1,
        "owner_query_after_refresh": 1,
        "formal_owner_query_id_after_intervention": 1,
        "formal_owner_query_id_unchanged": True,
        "matched_current_query": None,
        "owner_to_birth_cosine": 1.0,
        "owner_to_oracle_query_cosine": None,
        "owner_attention_top_query": 1,
        "oracle_query_attention_rank": None,
        "owner_attention_entropy": 0.5,
        "cancel_logit": 1.0,
        "continue_logit": 0.0,
        "end_logit": -1.0,
        "cancel_margin": 1.0,
        "continue_margin": -1.0,
        "end_margin": -2.0,
        "state_argmax": 0,
        "end_suppressed_by_minimum_duration": False,
        "formal_transition": "cancel" if route == "formal" else None,
        "shadow_transition": "diagnostic_retain_cancel" if route == "shadow" else None,
        "target_end_observable_now": False,
        "frames_from_annotated_end": None,
        "formal_lifetime": 4.0 if route == "formal" else None,
        "shadow_lifetime": 4.0 if route == "shadow" else None,
    }


def _write_trace(path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _d14_gate(checkpoint, options) -> dict:
    checkpoint_sha256 = _sha256(checkpoint)
    options_sha256 = _sha256(options)
    lifecycle = {
        "runtime_birth_count": EXPECTED_D14_COUNTS["birth_count"],
        "runtime_cancel_count": EXPECTED_D14_COUNTS["cancellation_count"],
        "runtime_end_count": EXPECTED_D14_COUNTS["end_count"],
        "runtime_emit_count": EXPECTED_D14_COUNTS["emit_count"],
        "runtime_reacquisition_count": EXPECTED_D14_COUNTS["reacquisition_count"],
        "runtime_capacity_exhaustion_count": EXPECTED_D14_COUNTS[
            "capacity_exhaustion_count"
        ],
    }
    return {
        "protocol": "eventmatr_d14_cross_arm_structure_gate_v1",
        "status": "FAIL_STRUCTURE_GATE",
        "selected_variant": None,
        "test_access": False,
        "threshold_search": False,
        "threshold_lowering": False,
        "multi_seed": False,
        "raw_rgb_training": False,
        "strict_causal_paper_result_valid": False,
        "official_paper_performance_valid": False,
        "locked_test_release": False,
        "paper_claim_release": False,
        "source_identity": {
            "d14": {"commit": D14_COMMIT, "tree": D14_TREE},
        },
        "official_train_artifacts": OFFICIAL_TRAIN_ARTIFACTS,
        "arms": {
            "decision_aligned_bag": {
                "checkpoint_sha256": checkpoint_sha256,
                "terminal_lifecycle": lifecycle,
            },
        },
        "linked_training_artifacts": {
            "bag": {
                "checkpoint": {
                    "path": str(checkpoint.resolve()),
                    "bytes": checkpoint.stat().st_size,
                    "sha256": checkpoint_sha256,
                },
                "options": {
                    "path": str(options.resolve()),
                    "bytes": options.stat().st_size,
                    "sha256": options_sha256,
                },
            },
        },
    }


def _source_provenance_scan() -> dict:
    return {
        "training_source_identity": {
            "commit": TRAINING_COMMIT,
            "tree": TRAINING_TREE,
            "clean": True,
            "status": "PASS",
            "status_porcelain": "",
            "smoke": {"status": "PASS", "test_access": False},
        },
        "d14_structure_gate": {
            "binding": {
                "source_identity": {
                    "commit": D14_COMMIT,
                    "tree": D14_TREE,
                },
            },
        },
    }


def test_d15_d14_gate_accepts_distinct_training_and_gate_sources(tmp_path) -> None:
    checkpoint = tmp_path / "terminal_epoch1.pth"
    options = tmp_path / "opts.json"
    checkpoint.write_bytes(b"checkpoint")
    options.write_text("{}\n", encoding="utf-8")
    gate = _d14_gate(checkpoint, options)

    binding = _validate_d14_gate(
        gate,
        checkpoint=checkpoint.resolve(),
        options=options.resolve(),
        checkpoint_sha256=_sha256(checkpoint),
        options_sha256=_sha256(options),
        d14_source_commit=D14_COMMIT,
        d14_source_tree=D14_TREE,
    )

    assert binding["source_identity"] == {
        "commit": D14_COMMIT,
        "tree": D14_TREE,
    }
    with pytest.raises(RuntimeError, match="D1.4 gate source commit mismatch"):
        _validate_d14_gate(
            gate,
            checkpoint=checkpoint.resolve(),
            options=options.resolve(),
            checkpoint_sha256=_sha256(checkpoint),
            options_sha256=_sha256(options),
            d14_source_commit=TRAINING_COMMIT,
            d14_source_tree=TRAINING_TREE,
        )


def test_d15_finalizer_preserves_three_distinct_source_identities() -> None:
    provenance = _validate_source_provenance(
        _source_provenance_scan(),
        training_commit=TRAINING_COMMIT,
        training_tree=TRAINING_TREE,
        d14_source_commit=D14_COMMIT,
        d14_source_tree=D14_TREE,
    )

    assert provenance["training_source_identity"]["commit"] == TRAINING_COMMIT
    assert provenance["d14_source_identity"]["commit"] == D14_COMMIT
    assert provenance["training_source_identity"]["commit"] != (
        provenance["d14_source_identity"]["commit"]
    )


def test_d15_finalizer_rejects_non_mapping_receipt_sections_cleanly() -> None:
    with pytest.raises(ValueError, match="is not an object"):
        _expect(None, {"status": "PASS"}, "nested section")
    with pytest.raises(ValueError, match="is not an object"):
        _required_mapping([], "nested section")
    with pytest.raises(ValueError, match="is not an object"):
        _validate_route(
            None,
            channel="PF",
            route="formal",
            query_hash="0" * 64,
            consumption_hash="1" * 64,
        )


def test_d15_finalizer_fails_closed_on_source_identity_aliasing() -> None:
    scan = _source_provenance_scan()
    aliased = copy.deepcopy(scan)
    aliased["d14_structure_gate"]["binding"]["source_identity"]["commit"] = (
        TRAINING_COMMIT
    )
    with pytest.raises(ValueError, match="D1.5 D1.4 source identity commit mismatch"):
        _validate_source_provenance(
            aliased,
            training_commit=TRAINING_COMMIT,
            training_tree=TRAINING_TREE,
            d14_source_commit=D14_COMMIT,
            d14_source_tree=D14_TREE,
        )

    wrong_training = copy.deepcopy(scan)
    wrong_training["training_source_identity"]["commit"] = D14_COMMIT
    with pytest.raises(ValueError, match="training source identity commit mismatch"):
        _validate_source_provenance(
            wrong_training,
            training_commit=TRAINING_COMMIT,
            training_tree=TRAINING_TREE,
            d14_source_commit=D14_COMMIT,
            d14_source_tree=D14_TREE,
        )


def test_d15_padding_runtime_preserves_existing_active_records() -> None:
    runtime = {
        "padding_prefixes_ignored": torch.tensor([1]),
        "active_count": torch.tensor([3]),
        "birth_count": torch.tensor([0]),
        "end_count": torch.tensor([0]),
        "new_birth_mask": torch.tensor([[False]]),
    }

    _validate_padding_runtime(
        runtime,
        expected_active_count=3,
        label="OF/shadow",
    )

    wrong_active = dict(runtime)
    wrong_active["active_count"] = torch.tensor([0])
    with pytest.raises(RuntimeError, match="padding active_count drifted"):
        _validate_padding_runtime(
            wrong_active,
            expected_active_count=3,
            label="OF/shadow",
        )

    mutated = dict(runtime)
    mutated["birth_count"] = torch.tensor([1])
    with pytest.raises(RuntimeError, match="padding produced birth_count"):
        _validate_padding_runtime(
            mutated,
            expected_active_count=3,
            label="OF/shadow",
        )


def test_d15_trace_integrity_cross_checks_every_route(tmp_path) -> None:
    path = tmp_path / "trace.jsonl.gz"
    rows = [
        _trace_row(channel, route)
        for channel in CHANNELS
        for route in ROUTES
    ]
    _write_trace(path, rows)

    result = _validate_trace(path, _routes())

    assert result["line_count"] == len(CHANNELS) * len(ROUTES)
    assert result["chronological_per_route_video"] is True
    assert result["owner_decisions_unique_per_prefix"] is True
    assert result["transition_counts_cross_checked"] is True


def test_d15_trace_integrity_fails_closed_on_missing_route_row(tmp_path) -> None:
    path = tmp_path / "trace.jsonl.gz"
    rows = [
        _trace_row(channel, route)
        for channel in CHANNELS
        for route in ROUTES
    ][:-1]
    _write_trace(path, rows)

    with pytest.raises(ValueError, match="trace route counts differ"):
        _validate_trace(path, _routes())


def test_d15_trace_integrity_requires_every_registered_field(tmp_path) -> None:
    path = tmp_path / "trace.jsonl.gz"
    rows = [
        _trace_row(channel, route)
        for channel in CHANNELS
        for route in ROUTES
    ]
    del rows[0]["owner_to_birth_cosine"]
    _write_trace(path, rows)

    with pytest.raises(ValueError, match="omitted"):
        _validate_trace(path, _routes())


def test_d15_trace_integrity_recomputes_state_and_lifetime(tmp_path) -> None:
    path = tmp_path / "trace.jsonl.gz"
    rows = [
        _trace_row(channel, route)
        for channel in CHANNELS
        for route in ROUTES
    ]
    rows[0]["cancel_margin"] = float("nan")
    _write_trace(path, rows)

    with pytest.raises(ValueError, match="not finite"):
        _validate_trace(path, _routes())

    rows[0] = _trace_row(CHANNELS[0], ROUTES[0])
    rows[0]["formal_lifetime"] = 3.0
    _write_trace(path, rows)
    with pytest.raises(ValueError, match="route lifetime drifted"):
        _validate_trace(path, _routes())


def test_d15_trace_integrity_recomputes_target_end_distance(tmp_path) -> None:
    path = tmp_path / "trace.jsonl.gz"
    rows = [
        _trace_row(channel, route)
        for channel in CHANNELS
        for route in ROUTES
    ]
    rows[0]["diagnostic_target_event_id"] = 7
    rows[0]["diagnostic_target_end_frame"] = 3.0
    rows[0]["frames_from_annotated_end"] = 2.0
    _write_trace(path, rows)

    with pytest.raises(ValueError, match="target-end audit drifted"):
        _validate_trace(path, _routes())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("first_owner_decision", False), "first-decision audit drifted"),
        (("owner_query_before_refresh", 10), "exceeds query bandwidth"),
        (("owner_query_after_refresh", 10), "exceeds query bandwidth"),
        (
            ("formal_owner_query_id_after_intervention", 10),
            "exceeds query bandwidth",
        ),
        (("owner_attention_top_query", 10), "exceeds query bandwidth"),
        (("matched_current_query", 10), "exceeds query bandwidth"),
        (("oracle_query_attention_rank", 11), "exceeds query bandwidth"),
        (("formal_transition", "continue"), "cancel policy drifted"),
    ],
)
def test_d15_trace_integrity_rejects_decision_contract_drift(
    tmp_path, mutation, message
) -> None:
    path = tmp_path / "trace.jsonl.gz"
    rows = [
        _trace_row(channel, route)
        for channel in CHANNELS
        for route in ROUTES
    ]
    field, value = mutation
    rows[0][field] = value
    _write_trace(path, rows)

    with pytest.raises(ValueError, match=message):
        _validate_trace(path, _routes())


def test_d15_trace_integrity_cross_checks_source_counts(tmp_path) -> None:
    path = tmp_path / "trace.jsonl.gz"
    rows = [
        _trace_row(channel, route)
        for channel in CHANNELS
        for route in ROUTES
    ]
    routes = _routes()
    routes["PF"]["formal"]["decision_rows_by_source"] = {"other": 1}
    _write_trace(path, rows)

    with pytest.raises(ValueError, match="source rows differ"):
        _validate_trace(path, routes)


def test_d15_trace_integrity_rejects_unreported_target_sidecar(tmp_path) -> None:
    path = tmp_path / "trace.jsonl.gz"
    rows = [
        _trace_row(channel, route)
        for channel in CHANNELS
        for route in ROUTES
    ]
    rows[0]["diagnostic_target_event_id"] = 7
    rows[0]["diagnostic_target_end_frame"] = 3.0
    rows[0]["frames_from_annotated_end"] = 1.0
    _write_trace(path, rows)

    with pytest.raises(ValueError, match="sidecar accounting differs"):
        _validate_trace(path, _routes())


def test_d15_trace_integrity_rejects_overreported_pf_sidecar(tmp_path) -> None:
    path = tmp_path / "trace.jsonl.gz"
    rows = [
        _trace_row(channel, route)
        for channel in CHANNELS
        for route in ROUTES
    ]
    routes = _routes()
    routes["PF"]["formal"]["linked_unique_target_count"] = 1
    _write_trace(path, rows)

    with pytest.raises(ValueError, match="sidecar accounting differs"):
        _validate_trace(path, routes)


def _routing_routes() -> dict:
    return {
        channel: {
            route: {
                "counts": {
                    "end_count": 0,
                    "emit_count": 0,
                    "end_argmax_count": 0,
                    "end_min_duration_suppression_count": 0,
                    "target_backed_end_within_one_segment_count": 0,
                }
            }
            for route in ROUTES
        }
        for channel in CHANNELS
    }


def _route_receipt() -> dict:
    eos = {f"video_validation_{index:07d}": 0 for index in range(200)}
    return {
        "channel": "PF",
        "route": "formal",
        "admission": "predicted",
        "identity": "free",
        "query_stream_sha256": "0" * 64,
        "query_consumption_sha256": "1" * 64,
        "counts": {
            "real_prefix_count": 203363,
            "padding_prefix_count": 5917,
            "padding_noop_count": 5917,
            "observed_eos_count": 200,
            "birth_count": 1,
            "cancellation_count": 1,
            "end_count": 0,
            "emit_count": 0,
            "reacquisition_count": 0,
            "capacity_exhaustion_count": 0,
            "records_active_after_eos": 0,
            "records_active_at_scan_end": 0,
            "videos_with_active_records_after_eos": 0,
            "association_count": 0,
            "late_association_count": 0,
            "owner_refresh_count": 0,
            "record_target_exact_tie_count": 0,
            "query_prefix_consumption_count": 209280,
            "decision_row_count": 1,
            "end_argmax_count": 0,
            "end_min_duration_suppression_count": 0,
            "target_backed_end_transition_count": 0,
            "target_backed_end_within_one_segment_count": 0,
            "end_at_observed_target_end_count": 0,
            "end_without_emission_count": 0,
            "silent_record_loss_count": 0,
        },
        "unique_runtime_record_count": 1,
        "created_records_by_source": {"predicted": 1},
        "decision_rows_by_source": {"predicted": 1},
        "linked_unique_target_count": 0,
        "raw_semantic_duplicate_count": 0,
        "batch_local_owner_decision_fragment_count": 1,
        "first_owner_decision_state_counts": {"0": 1},
        "owner_state_winner_counts": {"0": 1},
        "transition_counts": {"birth": 1, "cancel": 1},
        "statistics": {},
        "eos_active_record_counts_by_video": eos,
        "eos_birth_counts_by_video": dict(eos),
        "lifecycle_integrity": {
            "immutable_ledger_verified": True,
            "positive_length_verified": True,
            "nonnegative_start_verified": True,
            "no_duplicate_event_verified": True,
            "contiguous_sequence_id_verified": True,
            "ledger_emit_count_closed": True,
            "birth_terminal_active_partition_closed": True,
            "ledger_row_count": 0,
            "video_ledger_count": 200,
            "ground_truth_stored_in_runtime_record": False,
        },
    }


def test_d15_route_summary_counts_decision_bearing_sidecars() -> None:
    first = ("video_validation_0000001", 1)
    second = ("video_validation_0000001", 2)
    state = SimpleNamespace(
        channel="PF",
        route="formal",
        memory=SimpleNamespace(_records={}),
        links={
            first: SimpleNamespace(target_event_id=7),
            second: SimpleNamespace(target_event_id=7),
            ("video_validation_0000001", 3): SimpleNamespace(target_event_id=8),
        },
        first_decisions={first, second},
        target_link_history=Counter(),
        counters=Counter(
            {
                "birth_count": 2,
                "cancellation_count": 2,
                "association_count": 3,
                "decision_row_count": 2,
            }
        ),
        created_records={first, second},
        created_by_source=Counter({"predicted": 2}),
        decision_rows_by_source=Counter({"predicted": 2}),
        eos_active_record_counts={},
        eos_birth_counts={},
        first_state_counts=Counter({"0": 2}),
        winner_counts=Counter({"0": 2}),
        transition_counts=Counter({"birth": 2, "cancel": 2}),
        ledger_snapshots={},
        query_consumption_digest=hashlib.sha256(),
        stats={},
    )

    summary = _route_summary(state, query_hash="0" * 64)

    assert summary["linked_unique_target_count"] == 1
    assert summary["raw_semantic_duplicate_count"] == 1


def test_d15_route_receipt_closes_all_new_accounting() -> None:
    parsed = _validate_route(
        _route_receipt(),
        channel="PF",
        route="formal",
        query_hash="0" * 64,
        consumption_hash="1" * 64,
    )

    assert parsed["created_records_by_source"] == {"predicted": 1}
    assert parsed["decision_rows_by_source"] == {"predicted": 1}
    assert len(parsed["eos_active_record_counts_by_video"]) == 200


def _right_censored_positive_control() -> tuple[dict, dict]:
    censored = [
        {
            "video_name": "video_validation_0000318",
            "event_id": 22,
            "class_id": 0,
            "class_label": "HammerThrow",
            "start_frame": 900.0,
            "end_frame": 932.2,
            "censor_frame": 932.0,
            "runtime_event_id": 10,
            "observation_status": "right_censored",
        },
        {
            "video_name": "video_validation_0000985",
            "event_id": 9,
            "class_id": 1,
            "class_label": "VolleyballSpiking",
            "start_frame": 900.0,
            "end_frame": 931.2,
            "censor_frame": 931.0,
            "runtime_event_id": 11,
            "observation_status": "right_censored",
        },
    ]
    census = {
        "events": [
            {
                "video_name": row["video_name"],
                "event_id": row["event_id"],
                "class_id": row["class_id"],
                "class_label": row["class_label"],
                "start_frame": row["start_frame"],
                "end_frame": row["end_frame"],
                "last_observed_frame": row["censor_frame"],
                "observation_status": "right_censored",
            }
            for row in censored
        ]
    }
    control = {
        "counts": {
            "real_prefix_count": 203363,
            "padding_prefix_count": 5917,
            "padding_noop_count": 5917,
            "observed_eos_count": 200,
            "birth_count": 3003,
            "cancellation_count": 0,
            "end_count": 3001,
            "emit_count": 3001,
            "reacquisition_count": 0,
            "capacity_exhaustion_count": 0,
            "records_active_after_eos": 2,
            "records_active_at_scan_end": 2,
            "videos_with_active_records_after_eos": 2,
            "right_censored_event_count": 2,
        },
        "admitted_unique_target_count": 3003,
        "sidecar_link_count": 3003,
        "right_censored_events": censored,
        "lifecycle_integrity": {
            "immutable_ledger_verified": True,
            "positive_length_verified": True,
            "nonnegative_start_verified": True,
            "no_duplicate_event_verified": True,
            "contiguous_sequence_id_verified": True,
            "ledger_emit_count_closed": True,
            "birth_terminal_active_partition_closed": True,
            "ledger_row_count": 3001,
            "video_ledger_count": 200,
            "ground_truth_stored_in_runtime_record": False,
        },
    }
    return control, census


def test_d15_positive_control_closes_right_censoring_without_forced_end() -> None:
    control, census = _right_censored_positive_control()

    parsed = _validate_positive_control(control, lifecycle_census=census)

    assert parsed["counts"]["birth_count"] == 3003
    assert parsed["counts"]["end_count"] == 3001
    assert parsed["counts"]["records_active_at_scan_end"] == 2
    assert parsed["counts"]["right_censored_event_count"] == 2
    forced_end = copy.deepcopy(control)
    forced_end["counts"]["end_count"] = 3003
    forced_end["counts"]["emit_count"] = 3003
    forced_end["lifecycle_integrity"]["ledger_row_count"] = 3003
    with pytest.raises(ValueError, match="did not close"):
        _validate_positive_control(forced_end, lifecycle_census=census)


def _effect_row(effect: float = 0.0, passed: bool = False) -> dict:
    return {
        "paired_effect": effect,
        "cluster_bootstrap_ci95": [max(0.0, effect - 0.01), effect + 0.01],
        "holm_adjusted_p": 0.001 if passed else 1.0,
        "scientific_gate_pass": passed,
    }


def _paired_analysis() -> dict:
    return {
        "route_summaries": {
            f"{channel}/{route}": {"primary_success_count": 0}
            for channel in CHANNELS
            for route in ROUTES
        },
        "formal_effect_family": {
            "comparisons": {
                "identity_refresh_under_predicted_admission": _effect_row(),
                "identity_refresh_under_oracle_visible_admission": _effect_row(),
                "oracle_visible_admission_under_free_identity": _effect_row(),
            }
        },
        "no_cancel_effect_family": {
            "comparisons": {
                f"no_cancel_shadow_under_{channel}": _effect_row()
                for channel in CHANNELS
            }
        },
    }


def test_d15_routing_refuses_a_drifted_pf_control() -> None:
    routes = _routing_routes()
    paired = _paired_analysis()
    paired["route_summaries"]["PF/formal"]["primary_success_count"] = 1

    decision = _route_next_repair(routes, paired)

    assert decision["diagnosis"] == "predicted_free_reference_control_drift"
    assert decision["model_training_authorized"] is False


def test_d15_routing_prioritizes_end_emission_mismatch() -> None:
    routes = _routing_routes()
    routes["OR"]["formal"]["counts"]["emit_count"] = 1
    routes["OR"]["shadow"]["counts"][
        "target_backed_end_within_one_segment_count"
    ] = 1

    decision = _route_next_repair(routes, _paired_analysis())

    assert decision["diagnosis"] == "end_to_emission_closure_fault"
    assert decision["model_training_authorized"] is False


def test_d15_routing_requires_replicated_material_identity_effect() -> None:
    routes = _routing_routes()
    paired = _paired_analysis()
    paired["formal_effect_family"]["comparisons"][
        "identity_refresh_under_predicted_admission"
    ] = _effect_row(0.08, True)
    paired["formal_effect_family"]["comparisons"][
        "identity_refresh_under_oracle_visible_admission"
    ] = _effect_row(0.07, True)

    decision = _route_next_repair(routes, paired)

    assert decision["diagnosis"] == "replicated_identity_transport_effect"
    assert decision["model_implementation_authorized"] is True
    assert decision["model_training_authorized"] is False


def test_d15_routing_does_not_use_one_or_a_few_events() -> None:
    routes = _routing_routes()
    paired = _paired_analysis()
    paired["formal_effect_family"]["comparisons"][
        "identity_refresh_under_predicted_admission"
    ] = _effect_row(1.0 / 3001.0, False)
    paired["route_summaries"]["PR/formal"]["primary_success_count"] = 1
    paired["route_summaries"]["OF/formal"]["primary_success_count"] = 1

    decision = _route_next_repair(routes, paired)

    assert (
        decision["diagnosis"]
        == "no_preregistered_material_structural_effect"
    )


def test_d15_routing_uses_only_a_corrected_shadow_effect() -> None:
    routes = _routing_routes()
    paired = _paired_analysis()
    paired["no_cancel_effect_family"]["comparisons"][
        "no_cancel_shadow_under_OR"
    ] = _effect_row(0.09, True)

    decision = _route_next_repair(routes, paired)

    assert (
        decision["diagnosis"]
        == "material_no_cancel_effect_without_formal_factor_effect"
    )
    assert decision["model_training_authorized"] is False
