"""Pure-Python contracts for the preregistered D1.5.1 trace refinement."""

from __future__ import annotations

import pytest

from scripts import analyze_eventmatr_d151_endpoint_trace as analysis


def test_d151_temporal_bins_and_duration_strata_are_closed_at_boundaries() -> None:
    assert analysis.temporal_bin(-64.0001) == "pre_far"
    assert analysis.temporal_bin(-64.0) == "pre_window"
    assert analysis.temporal_bin(-0.0001) == "pre_window"
    assert analysis.temporal_bin(0.0) == "post_window"
    assert analysis.temporal_bin(64.0) == "post_window"
    assert analysis.temporal_bin(64.0001) == "post_far"
    assert [analysis.duration_stratum(value) for value in (8, 16, 32, 64, 65)] == [
        "le_8",
        "gt8_le16",
        "gt16_le32",
        "gt32_le64",
        "gt64",
    ]


def test_d151_nearest_rank_is_deterministic() -> None:
    assert analysis.nearest_rank([4, 1, 3, 2], 0.5) == 2
    assert analysis.nearest_rank(range(1, 101), 0.95) == 95


def _event(channel: str, route: str, video: str, event_id: int, margin: float):
    event = analysis.EventTrace(
        channel=channel,
        route=route,
        video_name=video,
        event_id=event_id,
        class_label="A",
        duration_frames=12.0,
        outcome={
            "unresolved_identity": 0,
            "ambiguous_identity": 0,
            "near_end_end": 0,
            "immutable_emission": 0,
            "primary_success": 0,
            "premature_cancel": 0,
            "target_end_observed_in_trace": 0,
        },
    )
    event.bin_maximum_end_margin["post_window"] = margin
    return event


def _small_event_map():
    result = {}
    for video_index in range(4):
        video = f"video_{video_index}"
        for channel in analysis.CHANNELS:
            for route in analysis.ROUTES:
                margin = 1.0 if route == "shadow" and video_index < 2 else -1.0
                event = _event(channel, route, video, video_index, margin)
                result[(channel, route, video, video_index)] = event
    return result


def test_d151_common_cluster_bootstrap_is_reproducible() -> None:
    events = _small_event_map()
    first = analysis.cluster_bootstrap_endpoint_rates(
        events,
        video_names=[f"video_{index}" for index in range(4)],
        replicates=200,
        seed=7,
    )
    second = analysis.cluster_bootstrap_endpoint_rates(
        events,
        video_names=[f"video_{index}" for index in range(4)],
        replicates=200,
        seed=7,
    )
    assert first == second
    assert first["rates"]["OF/shadow"]["point_rate"] == 0.5
    assert first["paired_oracle_identity_difference"]["point_difference"] == 0.0


def test_d151_cluster_bootstrap_rejects_incomplete_route_coverage() -> None:
    events = _small_event_map()
    del events[("OR", "shadow", "video_0", 0)]
    with pytest.raises(ValueError, match="route/event coverage drifted"):
        analysis.cluster_bootstrap_endpoint_rates(
            events,
            video_names=[f"video_{index}" for index in range(4)],
            replicates=10,
            seed=7,
        )


def _route_summaries(endpoint_rate: float = 0.0, p95: float = -1.0):
    count = round(
        endpoint_rate * analysis.EXPECTED_COUNTS["fully_observed_event_count"]
    )
    return {
        key: {
            "endpoint_eligible_end_without_emit_event_count": 0,
            "endpoint_maximum_end_margin": {
                "median_nearest_rank": -1.0,
                "p95_nearest_rank": p95,
            },
            "endpoint_positive_end_margin_event_count": count,
        }
        for key in analysis.ROUTE_KEYS
    }


def _uncertainty(
    rate: float = 0.0, interval=(0.0, 0.01), difference=0.0, diff_ci=(-0.01, 0.01)
):
    denominator = analysis.EXPECTED_COUNTS["fully_observed_event_count"]
    return {
        "rates": {
            key: {
                "positive_event_count": round(rate * denominator),
                "event_denominator": denominator,
                "point_rate": rate,
                "video_cluster_bootstrap_ci95": list(interval),
            }
            for key in analysis.ROUTE_KEYS
        },
        "paired_oracle_identity_difference": {
            "point_difference": difference,
            "video_cluster_bootstrap_ci95": list(diff_ci),
        },
    }


def test_d151_routing_prioritizes_material_state_machine_mismatch() -> None:
    summaries = _route_summaries()
    summaries["OF/shadow"]["endpoint_eligible_end_without_emit_event_count"] = 151
    decision = analysis.route_diagnosis(summaries, _uncertainty())
    assert decision["diagnosis"] == "material_endpoint_state_machine_mismatch"
    assert decision["model_training_authorized"] is False


def test_d151_routing_can_select_identity_cancel_or_risk_without_authorizing_training() -> (
    None
):
    summaries = _route_summaries()
    identity = analysis.route_diagnosis(
        summaries,
        _uncertainty(difference=0.06, diff_ci=(0.01, 0.10)),
    )
    assert identity["diagnosis"] == "material_oracle_identity_attribution_difference"

    cancel_uncertainty = _uncertainty(rate=0.06, interval=(0.01, 0.09))
    cancel_uncertainty["rates"]["OF/formal"]["positive_event_count"] = 0
    cancel = analysis.route_diagnosis(summaries, cancel_uncertainty)
    assert cancel["diagnosis"] == "material_cancel_transition_bottleneck"

    risk = analysis.route_diagnosis(summaries, _uncertainty())
    assert risk["diagnosis"].endswith("competing_risk_objective_insufficient")
    assert all(
        decision["model_implementation_authorized"] is False
        and decision["model_training_authorized"] is False
        for decision in (identity, cancel, risk)
    )


def test_d151_first_decision_and_nearest_endpoint_ties_are_deterministic() -> None:
    event = _event("OF", "shadow", "video", 0, -1.0)
    common = {
        "distance": -1.0,
        "cancel_margin": 0.5,
        "end_margin": -2.0,
        "winner": 0,
        "suppressed": False,
        "transition_parts": {"diagnostic_retain_cancel"},
        "target_endpoint_observed": False,
    }
    event.add_row(
        runtime_event_id=5, current_frame=9.0, first_owner_decision=True, **common
    )
    event.add_row(
        runtime_event_id=3,
        current_frame=9.0,
        first_owner_decision=True,
        **{**common, "cancel_margin": 0.7, "end_margin": -1.0},
    )
    assert event.first_decision_cancel_margin == 0.7
    assert event.last_pre_end_margin == -1.0


def _trace_row() -> dict:
    return {
        "protocol": analysis.D15_TRACE_PROTOCOL,
        "channel": "OF",
        "route": "shadow",
        "video_name": "video",
        "runtime_event_id": 0,
        "diagnostic_target_event_id": 0,
        "diagnostic_target_end_frame": 9.0,
        "source": "diagnostic_oracle_visible",
        "creation_frame": 1.0,
        "current_frame": 10.0,
        "is_eos": False,
        "first_owner_decision": True,
        "owner_query_before_refresh": 1,
        "owner_query_after_refresh": 1,
        "formal_owner_query_id_after_intervention": 1,
        "formal_owner_query_id_unchanged": True,
        "matched_current_query": None,
        "owner_to_birth_cosine": 0.9,
        "owner_to_oracle_query_cosine": 0.8,
        "owner_attention_top_query": 1,
        "oracle_query_attention_rank": 1,
        "owner_attention_entropy": 0.5,
        "cancel_logit": 2.0,
        "continue_logit": 1.0,
        "end_logit": 0.0,
        "cancel_margin": 1.0,
        "continue_margin": -1.0,
        "end_margin": -2.0,
        "state_argmax": 0,
        "end_suppressed_by_minimum_duration": False,
        "formal_transition": None,
        "shadow_transition": "diagnostic_retain_cancel",
        "target_end_observable_now": False,
        "frames_from_annotated_end": 1.0,
        "formal_lifetime": None,
        "shadow_lifetime": 9.0,
    }


def test_d151_trace_parser_rejects_lifetime_and_semantic_field_drift() -> None:
    analysis._parse_trace_row(_trace_row(), 1)
    bad_lifetime = _trace_row()
    bad_lifetime["shadow_lifetime"] = 8.0
    with pytest.raises(ValueError, match="route lifetime drifted"):
        analysis._parse_trace_row(bad_lifetime, 1)
    bad_source = _trace_row()
    bad_source["source"] = ""
    with pytest.raises(ValueError, match="source is invalid"):
        analysis._parse_trace_row(bad_source, 1)
