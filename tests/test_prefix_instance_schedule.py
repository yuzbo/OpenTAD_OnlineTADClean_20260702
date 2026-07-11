import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _build(*args, **kwargs):
    path = ROOT / "opentad" / "models" / "targets" / "prefix_instance_schedule.py"
    name = "_prefix_instance_schedule_under_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module.build_prefix_instance_schedule(*args, **kwargs)


def test_schedule_reveals_endpoint_only_at_first_decision_after_crossing():
    schedule = _build(
        segments=[[2.0, 9.0]],
        labels=[3],
        decision_frames=[7, 15, 23],
        previous_frame=-1,
    )

    assert [step.current_frame for step in schedule] == [7, 15, 23]
    assert schedule[0].births[0].instance_id == 0
    assert schedule[0].births[0].end_frame is None
    assert schedule[0].active[0].end_frame is None
    assert schedule[0].ends == ()
    assert schedule[1].active == ()
    assert schedule[1].ends[0].end_frame == pytest.approx(9.0)
    assert schedule[2].ends == ()


def test_schedule_keeps_repeated_and_overlapping_same_class_instances_distinct():
    schedule = _build(
        segments=[[1.0, 10.0], [4.0, 12.0], [14.0, 18.0]],
        labels=[2, 2, 2],
        decision_frames=[3, 7, 11, 15, 19],
        previous_frame=-1,
    )

    assert [item.instance_id for item in schedule[0].births] == [0]
    assert [item.instance_id for item in schedule[1].births] == [1]
    assert [item.instance_id for item in schedule[1].active] == [0, 1]
    assert [item.instance_id for item in schedule[3].births] == [2]
    assert [item.instance_id for item in schedule[3].ends] == [1]
    assert all(item.label == 2 for step in schedule for item in step.births + step.active + step.ends)


def test_schedule_does_not_repeat_births_across_chunk_boundaries():
    first = _build(
        segments=[[2.0, 20.0]],
        labels=[1],
        decision_frames=[7, 15],
        previous_frame=-1,
    )
    second = _build(
        segments=[[2.0, 20.0]],
        labels=[1],
        decision_frames=[23, 31],
        previous_frame=15,
    )

    assert [len(step.births) for step in first] == [1, 0]
    assert [len(step.births) for step in second] == [0, 0]
    assert second[0].ends[0].instance_id == 0
    assert second[0].ends[0].end_frame == pytest.approx(20.0)


def test_schedule_handles_more_births_than_slots_without_merging_targets():
    schedule = _build(
        segments=[[1.0, 8.0], [2.0, 9.0], [3.0, 10.0]],
        labels=[0, 0, 0],
        decision_frames=[7],
        previous_frame=-1,
    )

    assert [item.instance_id for item in schedule[0].births] == [0, 1, 2]
    assert all(item.end_frame is None for item in schedule[0].births)


@pytest.mark.parametrize(
    "segments,labels,decision_frames,error",
    [
        ([[2.0, 1.0]], [0], [3], "endpoint"),
        ([[1.0, 2.0]], [0, 1], [3], "same number"),
        ([[1.0, 2.0]], [0], [3, 2], "strictly increasing"),
    ],
)
def test_schedule_rejects_invalid_inputs(segments, labels, decision_frames, error):
    with pytest.raises(ValueError, match=error):
        _build(
            segments=segments,
            labels=labels,
            decision_frames=decision_frames,
            previous_frame=-1,
        )
