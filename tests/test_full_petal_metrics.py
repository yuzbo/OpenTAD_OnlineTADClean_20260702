import pytest

from opentad.evaluations import compute_full_petal_metrics
from opentad.evaluations.full_petal_metrics import FullPetalProtocolError
from opentad.evaluations.full_petal_metrics import (
    IDENTITY_METRIC_CONTRACT_SHA256,
)


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
    score=0.75,
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
        "score": score,
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
        _emission(
            "emit-primary", [0, 10], emit_frame=12, sequence_id=0, score=0.9
        ),
        _emission(
            "emit-exact-duplicate",
            [0, 10],
            emit_frame=13,
            sequence_id=1,
            score=0.8,
        ),
        _emission(
            "emit-fragment", [1, 9], emit_frame=14, sequence_id=2, score=0.7
        ),
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
        "fragment_emissions": 0,
        "unmatched_emissions": 1,
    }
    assert metrics["matching"]["policy"] == "maximum_cardinality_then_total_tiou"
    assert metrics["matching"]["pairs"][0]["ground_truth_id"] == "gt-action"
    assert metrics["matching"]["pairs"][0]["emission_id"] == "emit-primary"

    duplicates = metrics["duplicates"]
    assert duplicates["duplicate_emission_count"] == 2
    assert duplicates["duplicate_per_gt"] == pytest.approx(1.0)
    assert duplicates["duplicate_fraction"] == pytest.approx(0.5)
    assert duplicates["denominators"] == {
        "duplicate_per_gt": {"name": "all_ground_truth", "value": 2},
        "duplicate_fraction": {"name": "all_emissions", "value": 4},
    }
    assert duplicates["per_matched_ground_truth"] == [
        {
            "ground_truth_id": "gt-action",
            "primary_emission_id": "emit-primary",
            "duplicate_count": 2,
            "duplicate_emission_ids": ["emit-exact-duplicate", "emit-fragment"],
        }
    ]

    fragmentation = metrics["fragmentation"]
    assert fragmentation["fragmented_ground_truth_count"] == 0
    assert fragmentation["fragment_count"] == 0
    assert fragmentation["rate"] == pytest.approx(0.0)
    assert fragmentation["denominator"] == {
        "name": "all_ground_truth",
        "value": 2,
    }
    assert "individually below" in fragmentation["definition"]

    false_emissions = metrics["false_emissions"]
    assert false_emissions["unmatched_emission_count"] == 1
    assert false_emissions["rate"] == pytest.approx(0.25)
    assert false_emissions["emission_ids"] == ["emit-wrong-class"]
    assert metrics["duplicate_per_gt"] == pytest.approx(1.0)
    assert metrics["duplicate_fraction"] == pytest.approx(0.5)
    assert metrics["fragmentation_rate"] == pytest.approx(0.0)
    assert metrics["false_emission_rate"] == pytest.approx(0.25)

    latency = metrics["endpoint_detection_latency_frames"]
    assert latency == {"count": 1, "mean": 2.0, "p50": 2.0, "p90": 2.0}
    assert metrics["causal_validation"]["passed"] is True
    assert metrics["causal_validation"]["num_emissions"] == 4


def test_identity_contract_freezes_same_class_subsets_and_censored_misses():
    ground_truth = [
        {"gt_id": "a0", "stream_key": "s", "label": "a", "segment": [0, 10]},
        {"gt_id": "a1", "stream_key": "s", "label": "a", "segment": [8, 18]},
        {"gt_id": "a2", "stream_key": "s", "label": "a", "segment": [30, 40]},
        {"gt_id": "b0", "stream_key": "s", "label": "b", "segment": [50, 60]},
    ]
    emissions = [
        _emission("a0-hit", [0, 10], label="a", emit_frame=10, sequence_id=0, stream_key="s"),
        _emission("a2-hit", [30, 40], label="a", emit_frame=40, sequence_id=1, stream_key="s"),
        _emission("b0-hit", [50, 60], label="b", emit_frame=60, sequence_id=2, stream_key="s"),
    ]

    metrics = compute_full_petal_metrics(
        ground_truth,
        emissions,
        tiou_threshold=0.5,
        latency_budget_sec=1.0,
    )

    assert metrics["schema_version"] == "full_petal_metrics.v3"
    assert metrics["identity_metric_contract_sha256"] == IDENTITY_METRIC_CONTRACT_SHA256
    subsets = metrics["identity_subsets"]
    assert subsets["all"]["ground_truth_count"] == 4
    assert subsets["all"]["miss_count"] == 1
    assert subsets["same_class_repeated"]["ground_truth_count"] == 3
    assert subsets["same_class_repeated"]["matched_count"] == 2
    assert subsets["same_class_concurrent"]["ground_truth_count"] == 2
    assert subsets["same_class_concurrent"]["miss_count"] == 1
    assert subsets["same_class_sequential"]["ground_truth_count"] == 1
    assert subsets["same_class_sequential"]["matched_count"] == 1
    assert (
        subsets["same_class_concurrent"]["endpoint_delay_frames"][
            "right_censored_miss_count"
        ]
        == 1
    )
    assert "never imputed" in metrics["identity_metric_contract"]["delay_policy"]


def test_fragmentation_requires_multiple_subthreshold_parts_with_joint_coverage():
    ground_truth = [
        {"gt_id": "gt", "stream_key": "s", "label": "a", "segment": [0, 10]}
    ]
    emissions = [
        _emission("left", [0, 4], label="a", stream_key="s", emit_frame=10),
        _emission(
            "right",
            [6, 10],
            label="a",
            stream_key="s",
            emit_frame=11,
            sequence_id=1,
        ),
    ]

    metrics = compute_full_petal_metrics(ground_truth, emissions, tiou_threshold=0.5)

    assert metrics["counts"]["matched_ground_truth"] == 0
    assert metrics["fragmentation"]["fragmented_ground_truth_count"] == 1
    assert metrics["fragmentation"]["fragment_count"] == 2
    assert metrics["fragmentation"]["rate"] == pytest.approx(1.0)
    assert metrics["fragmentation"]["per_ground_truth"][0]["union_coverage"] == pytest.approx(
        0.8
    )


def test_diagnostic_matching_maximizes_cardinality_before_pairwise_tiou():
    ground_truth = [
        {"gt_id": "g1", "stream_key": "s", "label": "a", "segment": [0, 10]},
        {"gt_id": "g2", "stream_key": "s", "label": "a", "segment": [5, 15]},
    ]
    emissions = [
        _emission(
            "flexible",
            [2, 13],
            label="a",
            stream_key="s",
            emit_frame=15,
            sequence_id=0,
            score=0.9,
        ),
        _emission(
            "g1-only",
            [0, 8],
            label="a",
            stream_key="s",
            emit_frame=16,
            sequence_id=1,
            score=0.8,
        ),
    ]

    metrics = compute_full_petal_metrics(ground_truth, emissions, tiou_threshold=0.5)

    assert metrics["counts"]["matched_ground_truth"] == 2
    assert {
        (pair["emission_id"], pair["ground_truth_id"])
        for pair in metrics["matching"]["pairs"]
    } == {("flexible", "g2"), ("g1-only", "g1")}


def test_diagnostic_matching_applies_endpoint_latency_budget():
    ground_truth = [
        {"gt_id": "gt", "stream_key": "s", "label": "a", "segment": [0, 10]}
    ]
    emissions = [
        _emission("late", [0, 10], label="a", stream_key="s", emit_frame=71)
    ]

    metrics = compute_full_petal_metrics(
        ground_truth,
        emissions,
        tiou_threshold=0.5,
        fps=30,
        latency_budget_sec=2.0,
    )

    assert metrics["counts"]["matched_ground_truth"] == 0
    assert metrics["counts"]["unmatched_emissions"] == 1


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
            "score": 0.75,
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
