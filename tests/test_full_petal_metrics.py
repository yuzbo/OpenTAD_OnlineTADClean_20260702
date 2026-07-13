import pytest

from opentad.evaluations import compute_full_petal_metrics
from opentad.evaluations.full_petal_metrics import FullPetalProtocolError


def _emission(
    emission_id,
    segment,
    *,
    label="action",
    emit_frame=12,
    source_frame=None,
    sequence_id=0,
    stream_key="stream-1",
    immutable=True,
):
    return {
        "emission_id": emission_id,
        "stream_key": stream_key,
        "label": label,
        "segment": list(segment),
        "emit_frame": emit_frame,
        "source_frame": emit_frame if source_frame is None else source_frame,
        "sequence_id": sequence_id,
        "immutable": immutable,
    }


def test_metrics_keep_duplicate_fragmented_and_unmatched_emissions_auditable():
    ground_truth = [
        {
            "gt_id": "gt-action",
            "stream_key": "stream-1",
            "label": "action",
            "segment": [0, 10],
        },
        {
            "gt_id": "gt-other",
            "stream_key": "stream-1",
            "label": "other",
            "segment": [20, 30],
        },
    ]
    emissions = [
        _emission("emit-primary", [0, 10], emit_frame=12, sequence_id=0),
        _emission("emit-exact-duplicate", [0, 10], emit_frame=13, sequence_id=1),
        _emission("emit-fragment", [1, 9], emit_frame=14, sequence_id=2),
        _emission(
            "emit-wrong-class",
            [0, 10],
            label="other",
            emit_frame=15,
            sequence_id=3,
        ),
    ]

    metrics = compute_full_petal_metrics(ground_truth, emissions, tiou_threshold=0.5)

    assert metrics["counts"] == {
        "ground_truth": 2,
        "emissions": 4,
        "matched_ground_truth": 1,
        "unmatched_ground_truth": 1,
        "primary_matches": 1,
        "duplicate_emissions": 2,
        "unmatched_emissions": 1,
    }
    assert metrics["matching"]["policy"] == "chronological_greedy_class_aware_tiou"
    assert metrics["matching"]["pairs"][0]["ground_truth_id"] == "gt-action"
    assert metrics["matching"]["pairs"][0]["emission_id"] == "emit-primary"

    duplicates = metrics["duplicates"]
    assert duplicates["duplicate_emission_count"] == 2
    assert duplicates["rate"] == pytest.approx(0.5)
    assert duplicates["denominator"] == {"name": "all_emissions", "value": 4}
    assert duplicates["per_matched_ground_truth"] == [
        {
            "ground_truth_id": "gt-action",
            "primary_emission_id": "emit-primary",
            "duplicate_count": 2,
            "duplicate_emission_ids": ["emit-exact-duplicate", "emit-fragment"],
        }
    ]

    fragmentation = metrics["fragmentation"]
    assert fragmentation["fragmented_ground_truth_count"] == 1
    assert fragmentation["distinct_fragment_count"] == 2
    assert fragmentation["excess_fragment_count"] == 1
    assert fragmentation["rate"] == pytest.approx(1.0)
    assert fragmentation["denominator"] == {
        "name": "matched_ground_truth",
        "value": 1,
    }
    assert "disjoint or overlapping" in fragmentation["definition"]

    false_emissions = metrics["false_emissions"]
    assert false_emissions["unmatched_emission_count"] == 1
    assert false_emissions["rate"] == pytest.approx(0.25)
    assert false_emissions["emission_ids"] == ["emit-wrong-class"]
    assert metrics["duplicate_rate"] == pytest.approx(0.5)
    assert metrics["fragmentation_rate"] == pytest.approx(1.0)
    assert metrics["false_emission_rate"] == pytest.approx(0.25)

    latency = metrics["endpoint_detection_latency_frames"]
    assert latency == {"count": 1, "mean": 2.0, "p50": 2.0, "p90": 2.0}
    assert metrics["causal_validation"]["passed"] is True
    assert metrics["causal_validation"]["num_emissions"] == 4


def test_temporal_matching_has_deterministic_ground_truth_tie_breaking():
    ground_truth = [
        {"gt_id": "gt-first", "stream_key": "s", "label": "a", "segment": [0, 10]},
        {"gt_id": "gt-second", "stream_key": "s", "label": "a", "segment": [0, 10]},
    ]
    emissions = [
        _emission("e-first", [0, 10], label="a", stream_key="s", sequence_id=0),
        _emission(
            "e-second",
            [0, 10],
            label="a",
            stream_key="s",
            emit_frame=13,
            sequence_id=1,
        ),
    ]

    metrics = compute_full_petal_metrics(ground_truth, emissions, tiou_threshold=0.5)

    assert [pair["ground_truth_id"] for pair in metrics["matching"]["pairs"]] == [
        "gt-first",
        "gt-second",
    ]
    assert metrics["counts"]["duplicate_emissions"] == 0


def test_explicit_frame_bounds_take_precedence_over_display_segment_units():
    ground_truth = [
        {
            "gt_id": "gt-frame",
            "stream_id": "stream-a",
            "label": "action",
            "segment": [0.0, 1.0],
            "start_frame": 0,
            "end_frame": 10,
        }
    ]
    emissions = [
        {
            "event_id": "event-frame",
            "stream_id": "stream-a",
            "label": "action",
            "segment": [0.0, 1.0],
            "start_frame": 0,
            "end_frame": 10,
            "emit_frame": 12,
            "source_frame": 12,
            "sequence": 0,
            "immutable": True,
        }
    ]

    metrics = compute_full_petal_metrics(ground_truth, emissions)

    assert metrics["counts"]["primary_matches"] == 1
    assert metrics["endpoint_detection_latency_frames"]["mean"] == 2.0
    assert metrics["matching"]["coordinate_system"] == "frames"


@pytest.mark.parametrize(
    ("row", "violation"),
    [
        (
            _emission("future-end", [0, 13], emit_frame=12),
            "future_predicted_end",
        ),
        (
            _emission("future-source", [0, 10], emit_frame=12, source_frame=13),
            "future_source_frame",
        ),
        (
            _emission("mutable", [0, 10], immutable=False),
            "mutable_emission",
        ),
    ],
)
def test_causal_validation_rejects_future_tainted_or_mutable_rows(row, violation):
    with pytest.raises(FullPetalProtocolError, match=violation):
        compute_full_petal_metrics([], [row])


@pytest.mark.parametrize(
    ("emissions", "violation"),
    [
        (
            [
                _emission("e0", [0, 5], emit_frame=10, sequence_id=0),
                _emission("e1", [0, 5], emit_frame=9, sequence_id=1),
            ],
            "non_monotonic_emit_frame",
        ),
        (
            [
                _emission("e0", [0, 5], emit_frame=10, sequence_id=2),
                _emission("e1", [0, 5], emit_frame=11, sequence_id=1),
            ],
            "non_monotonic_sequence",
        ),
    ],
)
def test_causal_validation_rejects_nonmonotonic_stream_order(emissions, violation):
    with pytest.raises(FullPetalProtocolError, match=violation):
        compute_full_petal_metrics([], emissions)


def test_lifecycle_metrics_count_slot_survival_and_target_binding_shuffles():
    lifecycle_traces = [
        {
            "stream_key": "stream-1",
            "prefix_frame": 0,
            "slots": [
                {"slot_id": "slot-0", "target_id": "gt-0"},
                {"slot_id": "slot-1", "target_id": "gt-1"},
            ],
        },
        {
            "stream_key": "stream-1",
            "prefix_frame": 8,
            "slots": [
                {"slot_id": "slot-0", "target_id": "gt-1"},
                {"slot_id": "slot-1", "target_id": "gt-0"},
            ],
        },
        {
            "stream_key": "stream-1",
            "prefix_frame": 16,
            "slots": [
                {"slot_id": "slot-0", "target_id": "gt-1"},
                {"slot_id": "slot-1", "target_id": "gt-0"},
            ],
        },
    ]

    metrics = compute_full_petal_metrics([], [], lifecycle_traces=lifecycle_traces)

    lifecycle = metrics["lifecycle"]
    assert lifecycle["provided"] is True
    assert lifecycle["num_prefixes"] == 3
    assert lifecycle["slot_survival"] == {
        "count": 2,
        "rate": 0.5,
        "denominator": {
            "name": "targets_bound_in_consecutive_prefixes",
            "value": 4,
        },
    }
    assert lifecycle["binding_shuffle"] == {
        "count": 2,
        "rate": 0.5,
        "denominator": {
            "name": "targets_bound_in_consecutive_prefixes",
            "value": 4,
        },
    }
