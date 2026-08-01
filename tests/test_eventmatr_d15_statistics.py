"""Deterministic contracts for D1.5 v2 censoring and event-level effects."""

from __future__ import annotations

import copy

import pytest

from scripts import eventmatr_d15_statistics as statistics


def _annotation(label: str, start: float, end: float) -> dict:
    return {"label": label, "segment": [start, end]}


def test_d15_lifecycle_census_separates_observed_censored_and_unobservable() -> None:
    census = statistics.build_lifecycle_census(
        video_dict={
            "video_validation_0000001": {
                "duration": 10.0,
                "annotations": [
                    _annotation("A", 1.0, 3.0),
                    _annotation("A", 8.0, 10.0),
                    _annotation("A", 11.0, 12.0),
                ],
            }
        },
        video_len={"video_validation_0000001": 10},
        label_names=["A"],
        birth_mode="instant_transition",
        anti_len=16,
    )

    assert census["counts"] == {
        "alive_supervision_row_count": 2,
        "annotation_count": 3,
        "birth_supervision_row_count": 2,
        "completely_unobservable_event_count": 1,
        "end_supervision_row_count": 1,
        "fully_observed_event_count": 1,
        "left_truncated_event_count": 0,
        "observable_end_target_count": 1,
        "right_censored_event_count": 1,
        "video_count": 1,
        "visible_birth_target_count": 2,
    }
    assert [event["observation_status"] for event in census["events"]] == [
        "fully_observed",
        "right_censored",
        "completely_unobservable",
    ]
    statistics.validate_lifecycle_census(census, require_official_counts=False)


def test_d15_short_action_can_transition_directly_from_birth_to_end() -> None:
    census = statistics.build_lifecycle_census(
        video_dict={
            "video_validation_0000001": {
                "duration": 10.0,
                "annotations": [_annotation("A", 1.1, 1.2)],
            }
        },
        video_len={"video_validation_0000001": 10},
        label_names=["A"],
        birth_mode="instant_transition",
        anti_len=16,
    )

    event = census["events"][0]
    assert event["birth_supervision_rows"] == 1
    assert event["alive_supervision_rows"] == 0
    assert event["end_supervision_rows"] == 1
    assert census["supervision_distribution"][
        "fully_observed_direct_birth_to_end_count"
    ] == 1


def _paired_fixture(video_count: int = 12) -> tuple[dict, list[dict]]:
    video_dict = {}
    video_len = {}
    for index in range(video_count):
        video_name = f"video_validation_{index:07d}"
        video_dict[video_name] = {
            "duration": 10.0,
            "annotations": [_annotation("A", 1.0, 3.0)],
        }
        video_len[video_name] = 10
    census = statistics.build_lifecycle_census(
        video_dict=video_dict,
        video_len=video_len,
        label_names=["A"],
        birth_mode="instant_transition",
        anti_len=16,
    )
    rows = []
    for event in census["events"]:
        for channel in ("PF", "PR", "OF", "OR"):
            formal_success = int(channel in {"PR", "OR"})
            for route in ("formal", "shadow"):
                rows.append(
                    {
                        "channel": channel,
                        "route": route,
                        "video_name": event["video_name"],
                        "event_id": event["event_id"],
                        "primary_success": formal_success,
                        "near_end_end": formal_success,
                        "immutable_emission": formal_success,
                        "premature_cancel": 0,
                        "unresolved_identity": False,
                        "ambiguous_identity": False,
                    }
                )
    return census, rows


def test_d15_paired_gate_requires_effect_interval_and_holm(monkeypatch) -> None:
    monkeypatch.setattr(statistics, "BOOTSTRAP_REPLICATES", 2_000)
    monkeypatch.setattr(statistics, "PERMUTATION_REPLICATES", 2_000)
    census, rows = _paired_fixture()

    result = statistics.build_paired_event_analysis(
        census=census,
        outcome_rows=rows,
        channels=("PF", "PR", "OF", "OR"),
        routes=("formal", "shadow"),
        require_official_counts=False,
    )

    comparisons = result["formal_effect_family"]["comparisons"]
    assert comparisons[
        "identity_refresh_under_predicted_admission"
    ]["scientific_gate_pass"] is True
    assert comparisons[
        "identity_refresh_under_oracle_visible_admission"
    ]["scientific_gate_pass"] is True
    assert comparisons[
        "oracle_visible_admission_under_free_identity"
    ]["scientific_gate_pass"] is False
    assert all(
        not row["scientific_gate_pass"]
        for row in result["no_cancel_effect_family"]["comparisons"].values()
    )


def test_d15_paired_gate_rejects_subthreshold_net_effect(monkeypatch) -> None:
    monkeypatch.setattr(statistics, "BOOTSTRAP_REPLICATES", 100)
    monkeypatch.setattr(statistics, "PERMUTATION_REPLICATES", 100)
    census, rows = _paired_fixture(video_count=21)
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["channel"] == "PR" and row["video_name"].endswith("0000000"):
            row["primary_success"] = 1
            row["near_end_end"] = 1
            row["immutable_emission"] = 1
        elif row["channel"] == "PR":
            row["primary_success"] = 0
            row["near_end_end"] = 0
            row["immutable_emission"] = 0

    result = statistics.build_paired_event_analysis(
        census=census,
        outcome_rows=changed,
        channels=("PF", "PR", "OF", "OR"),
        routes=("formal", "shadow"),
        require_official_counts=False,
    )
    comparison = result["formal_effect_family"]["comparisons"][
        "identity_refresh_under_predicted_admission"
    ]
    assert comparison["paired_effect"] < 0.05
    assert comparison["effect_size_pass"] is False
    assert comparison["scientific_gate_pass"] is False


def test_d15_census_validator_recomputes_event_semantics() -> None:
    census, _ = _paired_fixture(video_count=1)
    drifted = copy.deepcopy(census)
    drifted["events"][0]["duration_frames"] += 1.0
    drifted["event_list_sha256"] = statistics.canonical_sha256(drifted["events"])
    drifted["supervision_distribution"] = statistics._distribution_summary(
        drifted["events"]
    )

    with pytest.raises(ValueError, match="duration drifted"):
        statistics.validate_lifecycle_census(
            drifted, require_official_counts=False
        )


def test_d15_paired_outcome_flags_are_binary_and_semantically_consistent(
    monkeypatch,
) -> None:
    monkeypatch.setattr(statistics, "BOOTSTRAP_REPLICATES", 10)
    monkeypatch.setattr(statistics, "PERMUTATION_REPLICATES", 10)
    census, rows = _paired_fixture(video_count=1)
    nonbinary = copy.deepcopy(rows)
    nonbinary[0]["near_end_end"] = 2
    with pytest.raises(ValueError, match="near_end_end is not binary"):
        statistics.build_paired_event_analysis(
            census=census,
            outcome_rows=nonbinary,
            channels=("PF", "PR", "OF", "OR"),
            routes=("formal", "shadow"),
            require_official_counts=False,
        )

    ambiguous_success = copy.deepcopy(rows)
    ambiguous_success[0]["primary_success"] = 1
    ambiguous_success[0]["near_end_end"] = 1
    ambiguous_success[0]["immutable_emission"] = 1
    ambiguous_success[0]["ambiguous_identity"] = True
    with pytest.raises(ValueError, match="primary-success semantics drifted"):
        statistics.build_paired_event_analysis(
            census=census,
            outcome_rows=ambiguous_success,
            channels=("PF", "PR", "OF", "OR"),
            routes=("formal", "shadow"),
            require_official_counts=False,
        )
