from dataclasses import FrozenInstanceError, dataclass

import pytest

from opentad.utils.prefix_instance_schedule import build_prefix_instance_schedule
from opentad.utils.prefix_trajectory_supervision import (
    PrefixTrajectorySupervisionState,
    SupervisionMode,
)


@dataclass(frozen=True)
class _Target:
    instance_id: int
    label: int = 0
    start_frame: float = 0.0
    end_frame: float | None = None


@dataclass(frozen=True)
class _Step:
    current_frame: int
    births: tuple[_Target, ...] = ()
    active: tuple[_Target, ...] = ()
    ends: tuple[_Target, ...] = ()


def _bindings(rows):
    return {row.instance_id: row.slot_id for row in rows}


def test_one_visible_instance_has_identical_fixed_and_rematch_supervision():
    target = _Target(instance_id=7, label=2, start_frame=3.0)
    step = _Step(current_frame=7, births=(target,), active=(target,))
    cost_matrix = [[4.0, 0.0]]
    fixed = PrefixTrajectorySupervisionState(num_slots=2, mode="fixed")
    rematch = PrefixTrajectorySupervisionState(
        num_slots=2,
        mode=SupervisionMode.REMATCH,
    )

    fixed_result = fixed.transition(step, cost_matrix)
    rematch_result = rematch.transition(step, cost_matrix)

    assert fixed_result.birth_candidates == rematch_result.birth_candidates == (0, 1)
    assert fixed_result.birth_mask == rematch_result.birth_mask == (True, True)
    assert _bindings(fixed_result.birth_assignments) == {7: 1}
    assert fixed_result.birth_assignments == rematch_result.birth_assignments
    assert fixed_result.canonical_bindings == rematch_result.canonical_bindings
    assert fixed_result.loss_bindings == rematch_result.loss_bindings
    assert _bindings(fixed_result.loss_bindings) == {7: 1}
    assert fixed_result.at_risk_slots == rematch_result.at_risk_slots == (1,)
    assert fixed_result.at_risk_mask == rematch_result.at_risk_mask == (False, True)
    assert fixed_result.endpoint_slots == rematch_result.endpoint_slots == ()
    assert fixed_result.endpoint_mask == rematch_result.endpoint_mask == (False, False)
    assert fixed.instance_to_slot == rematch.instance_to_slot == {7: 1}
    assert fixed.slot_to_instance == rematch.slot_to_instance == {1: 7}


def test_birth_allocation_is_globally_optimal_and_shared_by_both_arms():
    first = _Target(instance_id=1, start_frame=1.0)
    second = _Target(instance_id=2, start_frame=2.0)
    step = _Step(
        current_frame=7,
        births=(first, second),
        active=(first, second),
    )
    costs = [[1.0, 2.0], [2.0, 100.0]]
    fixed = PrefixTrajectorySupervisionState(num_slots=2, mode="fixed")
    rematch = PrefixTrajectorySupervisionState(num_slots=2, mode="rematch")

    fixed_result = fixed.transition(step, costs)
    rematch_result = rematch.transition(step, costs)

    assert fixed_result.birth_assignments == rematch_result.birth_assignments
    assert _bindings(fixed_result.birth_assignments) == {1: 1, 2: 0}
    assert fixed.instance_to_slot == rematch.instance_to_slot == {1: 1, 2: 0}


def test_same_class_overlap_swaps_only_loss_bindings_in_rematch():
    first = _Target(instance_id=10, label=1, start_frame=1.0)
    second = _Target(instance_id=20, label=1, start_frame=2.0)
    birth_step = _Step(
        current_frame=7,
        births=(first, second),
        active=(first, second),
    )
    active_step = _Step(current_frame=15, active=(first, second))
    fixed = PrefixTrajectorySupervisionState(num_slots=2, mode="fixed")
    rematch = PrefixTrajectorySupervisionState(num_slots=2, mode="rematch")

    fixed_birth = fixed.transition(birth_step, [[0.0, 8.0], [8.0, 0.0]])
    rematch_birth = rematch.transition(birth_step, [[0.0, 8.0], [8.0, 0.0]])
    fixed_active = fixed.transition(active_step, [[9.0, 0.0], [0.0, 9.0]])
    rematch_active = rematch.transition(active_step, [[9.0, 0.0], [0.0, 9.0]])

    assert first.label == second.label
    assert fixed_birth.birth_assignments == rematch_birth.birth_assignments
    assert _bindings(fixed_birth.birth_assignments) == {10: 0, 20: 1}
    assert fixed_active.canonical_bindings == rematch_active.canonical_bindings
    assert _bindings(rematch_active.canonical_bindings) == {10: 0, 20: 1}
    assert _bindings(fixed_active.loss_bindings) == {10: 0, 20: 1}
    assert _bindings(rematch_active.loss_bindings) == {10: 1, 20: 0}
    assert fixed.instance_to_slot == rematch.instance_to_slot == {10: 0, 20: 1}
    assert fixed.slot_to_instance == rematch.slot_to_instance == {0: 10, 1: 20}
    assert rematch.rematch_swap_count == 2
    assert fixed_active.audit.rematch_swapped_instance_ids == ()
    assert rematch_active.audit.rematch_swapped_instance_ids == (10, 20)
    assert rematch_active.audit.rematch_swap_total == 2


def test_endpoint_uses_each_arms_loss_binding_once_before_canonical_retirement():
    ending = _Target(instance_id=10, label=1, start_frame=1.0)
    continuing = _Target(instance_id=20, label=1, start_frame=2.0)
    birth_step = _Step(
        current_frame=7,
        births=(ending, continuing),
        active=(ending, continuing),
    )
    end_step = _Step(
        current_frame=15,
        active=(continuing,),
        ends=(_Target(10, label=1, start_frame=1.0, end_frame=12.0),),
    )
    fixed = PrefixTrajectorySupervisionState(num_slots=2, mode="fixed")
    rematch = PrefixTrajectorySupervisionState(num_slots=2, mode="rematch")
    for state in (fixed, rematch):
        state.transition(birth_step, [[0.0, 8.0], [8.0, 0.0]])

    fixed_result = fixed.transition(end_step, [[9.0, 0.0], [0.0, 9.0]])
    rematch_result = rematch.transition(end_step, [[9.0, 0.0], [0.0, 9.0]])

    fixed_loss = _bindings(fixed_result.loss_bindings)
    rematch_loss = _bindings(rematch_result.loss_bindings)
    assert fixed_result.at_risk_slots == rematch_result.at_risk_slots == (0, 1)
    assert fixed_result.at_risk_mask == rematch_result.at_risk_mask == (True, True)
    assert fixed_result.at_risk_slots == tuple(sorted(fixed_loss.values()))
    assert rematch_result.at_risk_slots == tuple(sorted(rematch_loss.values()))
    assert fixed_result.endpoint_slots == (fixed_loss[10],) == (0,)
    assert fixed_result.endpoint_mask == (True, False)
    assert rematch_result.endpoint_slots == (rematch_loss[10],) == (1,)
    assert rematch_result.endpoint_mask == (False, True)
    assert 10 not in fixed.instance_to_slot
    assert 10 not in rematch.instance_to_slot
    assert fixed.instance_to_slot == rematch.instance_to_slot == {20: 1}
    assert fixed.retired_instance_ids == rematch.retired_instance_ids == {10}


@pytest.mark.parametrize("channel", ["births", "active", "ends"])
def test_retired_instance_cannot_resurrect_or_repeat_its_endpoint(channel):
    target = _Target(instance_id=3, start_frame=1.0)
    state = PrefixTrajectorySupervisionState(num_slots=1, mode="fixed")
    state.transition(
        _Step(current_frame=7, births=(target,), active=(target,)),
        [[0.0]],
    )
    state.transition(
        _Step(
            current_frame=15,
            ends=(_Target(3, start_frame=1.0, end_frame=11.0),),
        ),
        [[0.0]],
    )
    repeated = (
        _Target(3, start_frame=1.0, end_frame=11.0)
        if channel == "ends"
        else target
    )

    with pytest.raises(ValueError, match="retired"):
        state.transition(
            _Step(current_frame=23, **{channel: (repeated,)}),
            [[0.0]],
        )


def test_tied_slot_exhaustion_is_deterministic_shared_and_never_late_allocated():
    lower_id = _Target(instance_id=2, label=4, start_frame=1.0)
    higher_id = _Target(instance_id=5, label=4, start_frame=1.0)
    birth_step = _Step(
        current_frame=7,
        births=(higher_id, lower_id),
        active=(higher_id, lower_id),
    )
    fixed = PrefixTrajectorySupervisionState(num_slots=1, mode="fixed")
    rematch = PrefixTrajectorySupervisionState(num_slots=1, mode="rematch")

    fixed_birth = fixed.transition(birth_step, [[0.0], [0.0]])
    rematch_birth = rematch.transition(birth_step, [[0.0], [0.0]])

    assert fixed_birth.birth_assignments == rematch_birth.birth_assignments
    assert _bindings(fixed_birth.birth_assignments) == {2: 0}
    assert fixed_birth.exhausted_instance_ids == rematch_birth.exhausted_instance_ids == (5,)
    assert fixed_birth.exhaustion == rematch_birth.exhaustion == 1
    assert fixed.slot_exhaustion_count == rematch.slot_exhaustion_count == 1
    assert fixed.born_instance_ids == rematch.born_instance_ids == {2, 5}
    assert fixed_birth.audit.current_frame == 7
    assert fixed_birth.audit.visible_instance_ids == (2, 5)
    assert fixed_birth.audit.occupied_slots_before == ()
    assert fixed_birth.audit.occupied_slots_for_supervision == (0,)
    assert fixed_birth.audit.occupied_slots_after == (0,)
    assert fixed_birth.audit.exhausted_instance_ids == (5,)
    assert fixed_birth.audit.slot_exhaustion_total == 1
    assert not hasattr(fixed_birth.audit, "end_frame")

    fixed.transition(
        _Step(
            current_frame=15,
            active=(higher_id,),
            ends=(_Target(2, label=4, start_frame=1.0, end_frame=12.0),),
        ),
        [[0.0], [0.0]],
    )
    after_free = fixed.transition(
        _Step(current_frame=23, active=(higher_id,)),
        [[0.0]],
    )

    assert after_free.birth_assignments == ()
    assert after_free.canonical_bindings == ()
    assert fixed.instance_to_slot == {}


def test_clone_and_frozen_snapshot_do_not_share_mutable_lifecycle_storage():
    target = _Target(instance_id=7, start_frame=1.0)
    state = PrefixTrajectorySupervisionState(num_slots=2, mode="fixed")
    state.transition(
        _Step(current_frame=7, births=(target,), active=(target,)),
        [[3.0, 0.0]],
    )

    clone = state.clone()
    snapshot = state.snapshot()
    clone.instance_to_slot.clear()
    clone.slot_to_instance.clear()
    clone.born_instance_ids.add(99)

    assert state.instance_to_slot == {7: 1}
    assert state.slot_to_instance == {1: 7}
    assert state.born_instance_ids == {7}
    assert snapshot.instance_to_slot == ((7, 1),)
    assert snapshot.slot_to_instance == ((1, 7),)
    assert snapshot.born_instance_ids == frozenset({7})
    assert snapshot.retired_instance_ids == frozenset()
    assert snapshot.slot_exhaustion_count == 0
    assert snapshot.rematch_swap_count == 0
    with pytest.raises(FrozenInstanceError):
        snapshot.transition_count = 99


def test_cost_provider_is_phase_scoped_and_receives_ids_in_canonical_order():
    first = _Target(instance_id=8, start_frame=1.0)
    second = _Target(instance_id=3, start_frame=2.0)
    calls = []

    def provider(phase, instance_ids, slot_ids):
        calls.append((phase, instance_ids, slot_ids))
        assert all(type(value) is int for value in instance_ids + slot_ids)
        if phase == "birth":
            return [[0.0, 7.0], [7.0, 0.0]]
        assert phase == "rematch"
        return [[7.0, 0.0], [0.0, 7.0]]

    state = PrefixTrajectorySupervisionState(num_slots=2, mode="rematch")
    result = state.transition(
        _Step(
            current_frame=7,
            births=(first, second),
            active=(first, second),
        ),
        cost_provider=provider,
    )

    assert calls == [
        ("birth", (3, 8), (0, 1)),
        ("rematch", (3, 8), (0, 1)),
    ]
    assert _bindings(result.birth_assignments) == {3: 0, 8: 1}
    assert _bindings(result.canonical_bindings) == {3: 0, 8: 1}
    assert _bindings(result.loss_bindings) == {3: 1, 8: 0}


def test_state_rejects_non_bijections_and_slots_outside_capacity():
    with pytest.raises(ValueError, match="inverse"):
        PrefixTrajectorySupervisionState(
            num_slots=2,
            mode="fixed",
            instance_to_slot={4: 0},
            slot_to_instance={1: 4},
            born_instance_ids={4},
        )

    with pytest.raises(ValueError, match="outside"):
        PrefixTrajectorySupervisionState(
            num_slots=2,
            mode="fixed",
            instance_to_slot={4: 2},
            slot_to_instance={2: 4},
            born_instance_ids={4},
        )


def test_transition_rejects_unknown_active_ids_and_future_endpoint_leaks():
    state = PrefixTrajectorySupervisionState(num_slots=1, mode="fixed")
    pristine = state.snapshot()

    with pytest.raises(ValueError, match="unknown"):
        state.transition(
            _Step(current_frame=7, active=(_Target(4),)),
            [[0.0]],
        )
    assert state.snapshot() == pristine

    leaked = _Target(instance_id=4, start_frame=1.0, end_frame=99.0)
    with pytest.raises(ValueError, match="future endpoint"):
        state.transition(
            _Step(current_frame=7, births=(leaked,), active=(leaked,)),
            [[0.0]],
        )
    assert state.snapshot() == pristine


def test_failed_cost_provider_transition_is_atomic():
    target = _Target(instance_id=1, start_frame=1.0)
    state = PrefixTrajectorySupervisionState(num_slots=1, mode="rematch")
    pristine = state.snapshot()

    def provider(phase, instance_ids, slot_ids):
        if phase == "birth":
            return [[0.0]]
        raise RuntimeError("synthetic rematch failure")

    with pytest.raises(RuntimeError, match="synthetic rematch failure"):
        state.transition(
            _Step(current_frame=7, births=(target,), active=(target,)),
            provider,
        )

    assert state.snapshot() == pristine


def test_delayed_birth_endpoint_and_lifecycle_continue_across_chunk_boundary():
    segments = [[2.0, 20.0], [18.0, 22.0]]
    labels = [6, 6]
    first_chunk = build_prefix_instance_schedule(
        segments=segments,
        labels=labels,
        decision_frames=[7, 15],
        previous_frame=-1,
    )
    second_chunk = build_prefix_instance_schedule(
        segments=segments,
        labels=labels,
        decision_frames=[23],
        previous_frame=15,
    )
    state = PrefixTrajectorySupervisionState(num_slots=2, mode="rematch")

    state.transition(first_chunk[0], [[0.0, 8.0]])
    state.transition(first_chunk[1], [[0.0, 8.0]])

    assert state.instance_to_slot == {0: 0}
    assert state.last_current_frame == 15

    delayed = state.transition(
        second_chunk[0],
        [[9.0, 0.0], [0.0, 9.0]],
    )

    assert [item.instance_id for item in second_chunk[0].births] == [1]
    assert [item.instance_id for item in second_chunk[0].ends] == [0, 1]
    assert _bindings(delayed.birth_assignments) == {1: 1}
    assert _bindings(delayed.canonical_bindings) == {0: 0, 1: 1}
    assert _bindings(delayed.loss_bindings) == {0: 1, 1: 0}
    assert delayed.endpoint_slots == (0, 1)
    assert delayed.endpoint_mask == (True, True)
    assert delayed.audit.occupied_slots_before == (0,)
    assert delayed.audit.occupied_slots_for_supervision == (0, 1)
    assert delayed.audit.occupied_slots_after == ()
    assert state.instance_to_slot == {}
    assert state.born_instance_ids == {0, 1}
    assert state.retired_instance_ids == {0, 1}
    assert state.last_current_frame == 23


def test_repeated_birth_and_missing_occupied_lifecycle_are_rejected_atomically():
    target = _Target(instance_id=1, start_frame=1.0)
    state = PrefixTrajectorySupervisionState(num_slots=1, mode="fixed")
    state.transition(
        _Step(current_frame=7, births=(target,), active=(target,)),
        [[0.0]],
    )
    before_invalid = state.snapshot()

    with pytest.raises(ValueError, match="birth crossing"):
        state.transition(
            _Step(current_frame=15, births=(target,), active=(target,)),
            [[0.0]],
        )
    assert state.snapshot() == before_invalid

    with pytest.raises(ValueError, match="occupied.*visible"):
        state.transition(
            _Step(current_frame=15),
            lambda phase, instance_ids, slot_ids: [],
        )
    assert state.snapshot() == before_invalid


def test_non_finite_assignment_costs_are_rejected_without_state_changes():
    target = _Target(instance_id=1, start_frame=1.0)
    state = PrefixTrajectorySupervisionState(num_slots=1, mode="fixed")
    pristine = state.snapshot()

    with pytest.raises(ValueError, match="finite"):
        state.transition(
            _Step(current_frame=7, births=(target,), active=(target,)),
            [[float("nan")]],
        )

    assert state.snapshot() == pristine
