import importlib.util
from pathlib import Path


TARGETS_PATH = (
    Path(__file__).resolve().parents[1]
    / "opentad"
    / "models"
    / "targets"
    / "prefix_event_targets.py"
)
SPEC = importlib.util.spec_from_file_location("prefix_event_targets_under_test", TARGETS_PATH)
TARGETS_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TARGETS_MODULE)
build_prefix_event_targets = TARGETS_MODULE.build_prefix_event_targets


def test_future_endpoint_does_not_change_prefix_targets():
    earlier_future = build_prefix_event_targets(
        [[2, 20]],
        [0],
        previous_frame=7,
        current_frame=8,
        num_classes=1,
        delay_budget_frames=4,
    )
    later_future = build_prefix_event_targets(
        [[2, 40]],
        [0],
        previous_frame=7,
        current_frame=8,
        num_classes=1,
        delay_budget_frames=4,
    )

    assert earlier_future.prefix_observable_dict() == later_future.prefix_observable_dict()


def test_no_end_regression_before_endpoint():
    targets = build_prefix_event_targets(
        [[2, 20]],
        [0],
        previous_frame=7,
        current_frame=8,
        num_classes=1,
        delay_budget_frames=4,
    )

    assert targets.end_event == (0.0,)
    assert targets.censor_mask == (1.0,)
    assert targets.ongoing_target == (1.0,)
    assert not hasattr(targets, "end_coordinate")


def test_emit_forbidden_before_endpoint():
    targets = build_prefix_event_targets(
        [[2, 20]],
        [0],
        previous_frame=7,
        current_frame=8,
        num_classes=1,
        delay_budget_frames=4,
    )

    assert targets.emit_forbidden == (1.0,)
    assert targets.emit_allowed == (0.0,)
    assert targets.emit_event == (0.0,)


def test_completion_is_monotonic_for_an_event():
    values = [
        build_prefix_event_targets(
            [[2, 10]],
            [0],
            previous_frame=previous,
            current_frame=current,
            num_classes=1,
            delay_budget_frames=4,
        ).completion_target[0]
        for previous, current in [(0, 4), (4, 10), (10, 14)]
    ]

    assert values == [0.0, 1.0, 1.0]
    assert values == sorted(values)


def test_endpoint_crossing_creates_distinct_end_and_first_emit_events():
    targets = build_prefix_event_targets(
        [[2, 10]],
        [0],
        previous_frame=8,
        current_frame=10,
        num_classes=1,
        delay_budget_frames=4,
    )

    assert targets.end_event == (1.0,)
    assert targets.emit_event == (1.0,)
    assert targets.emit_allowed == (1.0,)
    assert targets.emit_forbidden == (0.0,)
    assert targets.completion_target == (1.0,)


def test_late_prefix_remains_complete_but_is_outside_emit_budget():
    targets = build_prefix_event_targets(
        [[2, 10]],
        [0],
        previous_frame=14,
        current_frame=15,
        num_classes=1,
        delay_budget_frames=4,
    )

    assert targets.completion_target == (1.0,)
    assert targets.emit_allowed == (0.0,)
    assert targets.emit_forbidden == (0.0,)
    assert targets.emit_event == (0.0,)
    assert targets.late_target == (1.0,)


def test_targets_validate_segments_labels_and_class_range():
    try:
        build_prefix_event_targets(
            [[2, 10]],
            [],
            previous_frame=0,
            current_frame=1,
            num_classes=1,
            delay_budget_frames=4,
        )
    except ValueError as exc:
        assert "same number" in str(exc)
    else:
        raise AssertionError("expected mismatched labels to fail")

    try:
        build_prefix_event_targets(
            [[2, 10]],
            [2],
            previous_frame=0,
            current_frame=1,
            num_classes=1,
            delay_budget_frames=4,
        )
    except ValueError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError("expected out-of-range label to fail")
