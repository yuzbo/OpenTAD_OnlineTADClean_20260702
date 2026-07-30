"""Pure-Python integrity contracts for the D1.5 diagnostic finalizer."""

from __future__ import annotations

import gzip
import json

import pytest

from scripts.finalize_eventmatr_d15_owner_counterfactual import (
    CHANNELS,
    PROTOCOL,
    ROUTES,
    _validate_trace,
)


def _routes() -> dict:
    return {
        channel: {
            route: {
                "counts": {
                    "decision_row_count": 1,
                    "end_argmax_count": 0,
                    "end_min_duration_suppression_count": 0,
                    "target_backed_end_transition_count": 0,
                }
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
        "source": "predicted",
        "creation_frame": 0.0,
        "current_frame": 4.0,
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

