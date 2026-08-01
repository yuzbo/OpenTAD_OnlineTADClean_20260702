"""Torch-free regression tests for the amended D1.5 finalizer."""

from __future__ import annotations

import copy

import pytest

from scripts.finalize_eventmatr_d15_owner_counterfactual import (
    CHANNELS,
    ROUTES,
    _build_event_outcome_rows,
    _route_next_repair,
    _validate_positive_control,
)


def _positive_control() -> tuple[dict, dict]:
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


def _routes() -> dict:
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


def _paired_analysis() -> dict:
    empty_effect = {
        "paired_effect": 0.0,
        "cluster_bootstrap_ci95": [0.0, 0.0],
        "holm_adjusted_p": 1.0,
        "scientific_gate_pass": False,
    }
    return {
        "route_summaries": {
            f"{channel}/{route}": {"primary_success_count": 0}
            for channel in CHANNELS
            for route in ROUTES
        },
        "formal_effect_family": {
            "comparisons": {
                "identity_refresh_under_predicted_admission": dict(empty_effect),
                "identity_refresh_under_oracle_visible_admission": dict(
                    empty_effect
                ),
                "oracle_visible_admission_under_free_identity": dict(empty_effect),
            }
        },
        "no_cancel_effect_family": {
            "comparisons": {
                f"no_cancel_shadow_under_{channel}": dict(empty_effect)
                for channel in CHANNELS
            }
        },
    }


def test_v2_positive_control_closes_two_right_censored_events() -> None:
    control, census = _positive_control()

    result = _validate_positive_control(control, lifecycle_census=census)

    assert result["counts"]["birth_count"] == 3003
    assert result["counts"]["end_count"] == 3001
    assert result["counts"]["records_active_at_scan_end"] == 2


def test_v2_positive_control_rejects_eos_scan_end_mismatch() -> None:
    control, census = _positive_control()
    drifted = copy.deepcopy(control)
    drifted["counts"]["records_active_at_scan_end"] = 1

    with pytest.raises(ValueError, match="did not close"):
        _validate_positive_control(drifted, lifecycle_census=census)


def test_v2_event_gate_never_authorizes_training() -> None:
    decision = _route_next_repair(_routes(), _paired_analysis())

    assert decision["diagnosis"] == (
        "frozen_owner_end_representation_or_risk_objective_insufficient"
    )
    assert decision["model_training_authorized"] is False


def _record_audit(runtime_event_id: int) -> dict:
    return {
        "channel": "PF",
        "route": "formal",
        "video_name": "video_validation_0000001",
        "runtime_event_id": runtime_event_id,
        "target_event_id": 0,
        "decision_row_count": 1,
        "first_owner_decision": None,
        "near_end_end": True,
        "immutable_emission": True,
        "near_end_immutable_emission": True,
        "primary_success": True,
        "premature_cancel": False,
        "cancel_winner_before_end": False,
        "target_end_observed_in_trace": True,
        "maximum_lifetime": 2.0,
        "owner_to_birth_cosine_total": 1.0,
        "owner_to_birth_cosine_count": 1,
        "owner_to_oracle_query_cosine_total": 1.0,
        "owner_to_oracle_query_cosine_count": 1,
        "oracle_query_attention_rank_total": 1.0,
        "oracle_query_attention_rank_count": 1,
        "target_backed_end_observations": [],
    }


def test_v2_ambiguous_target_fragments_cannot_create_primary_success() -> None:
    census = {
        "events": [
            {
                "video_name": "video_validation_0000001",
                "event_id": 0,
                "class_id": 0,
                "class_label": "A",
                "start_frame": 1.0,
                "end_frame": 3.0,
                "end_crossing_frame": 3.0,
                "birth_observed": True,
                "observation_status": "fully_observed",
            }
        ]
    }
    audits = {
        ("PF", "formal", "video_validation_0000001", event_id): _record_audit(
            event_id
        )
        for event_id in (1, 2)
    }

    rows, _ = _build_event_outcome_rows(census, audits)
    outcome = next(
        row
        for row in rows
        if row["channel"] == "PF" and row["route"] == "formal"
    )

    assert outcome["linked_runtime_record_count"] == 2
    assert outcome["ambiguous_identity"] is True
    assert outcome["primary_success"] == 0
