"""Prefix-observable lifecycle supervision for PERSIST-FIXED/REMATCH.

Both modes use one canonical first-crossing birth allocation.  REMATCH adds a
second loss-only assignment over exactly those canonical occupied slots; it
never rewrites lifecycle identity, capacity, or retirement.

``transition`` accepts either a dense matrix or a callable.  Dense rows are in
sorted order of the distinct instance ids present in ``births``, ``active``,
or ``ends`` for that step, and columns are slots ``0..num_slots-1``.  A callable
is invoked as ``provider(phase, instance_ids, slot_ids)`` for ``"birth"`` and,
when needed, ``"rematch"``.  Providers receive ids only, never target objects.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, Optional, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


class SupervisionMode(str, Enum):
    """Loss-binding mode; enum members are immutable validated strings."""

    FIXED = "fixed"
    REMATCH = "rematch"


class SupervisionInvariantError(ValueError):
    """Raised when canonical lifecycle or prefix-observability is violated."""

    pass


@dataclass(frozen=True, order=True)
class InstanceSlotBinding:
    instance_id: int
    slot_id: int


@dataclass(frozen=True)
class PrefixTrajectoryTransitionAudit:
    """Immutable, prefix-observable evidence for one successful transition."""

    transition_index: int
    current_frame: int
    mode: SupervisionMode
    visible_instance_ids: Tuple[int, ...]
    occupied_slots_before: Tuple[int, ...]
    occupied_slots_for_supervision: Tuple[int, ...]
    occupied_slots_after: Tuple[int, ...]
    retired_instance_ids: Tuple[int, ...]
    exhausted_instance_ids: Tuple[int, ...]
    rematch_swapped_instance_ids: Tuple[int, ...]
    slot_exhaustion_total: int
    rematch_swap_total: int


@dataclass(frozen=True)
class PrefixTrajectoryStateSnapshot:
    """Immutable deep snapshot suitable for chunk handoff/audit storage."""

    num_slots: int
    mode: SupervisionMode
    instance_to_slot: Tuple[Tuple[int, int], ...]
    slot_to_instance: Tuple[Tuple[int, int], ...]
    born_instance_ids: frozenset[int]
    retired_instance_ids: frozenset[int]
    slot_exhaustion_count: int
    rematch_swap_count: int
    transition_count: int
    last_current_frame: Optional[int]


@dataclass(frozen=True)
class PrefixTrajectoryTransitionResult:
    """Bindings and masks that must drive losses for the current endpoint step."""

    canonical_bindings: Tuple[InstanceSlotBinding, ...]
    loss_bindings: Tuple[InstanceSlotBinding, ...]
    birth_instance_ids: Tuple[int, ...]
    birth_candidates: Tuple[int, ...]
    birth_assignments: Tuple[InstanceSlotBinding, ...]
    at_risk_slots: Tuple[int, ...]
    endpoint_slots: Tuple[int, ...]
    birth_mask: Tuple[bool, ...]
    at_risk_mask: Tuple[bool, ...]
    endpoint_mask: Tuple[bool, ...]
    exhausted_instance_ids: Tuple[int, ...]
    exhaustion: int
    audit: PrefixTrajectoryTransitionAudit


def _mask(num_slots: int, slots: Tuple[int, ...]) -> Tuple[bool, ...]:
    selected = set(slots)
    return tuple(slot in selected for slot in range(num_slots))


def _validated_cost_matrix(values, expected_shape, phase):
    costs = np.asarray(values, dtype=np.float64)
    if costs.shape != expected_shape:
        raise ValueError(
            f"{phase} cost matrix must have shape {expected_shape}, got {costs.shape}"
        )
    if not np.isfinite(costs).all():
        raise ValueError(f"{phase} cost matrix must contain only finite values")
    return costs


def _assignment_cost(costs, rows, columns):
    return math.fsum(float(costs[row, column]) for row, column in zip(rows, columns))


def _solve_global_assignment(costs):
    """Return a primary-optimal assignment with explicit lexicographic ties.

    Rows and columns already follow canonical id/slot order.  Among assignments
    indistinguishable at float64 precision, earlier rows prefer lower columns;
    when rows outnumber columns, earlier rows also prefer being matched.
    """
    row_count, column_count = costs.shape
    if not row_count or not column_count:
        return ()

    base_rows, base_columns = linear_sum_assignment(costs)
    match_count = min(row_count, column_count)
    optimum = _assignment_cost(costs, base_rows, base_columns)
    scale = max(
        1.0,
        abs(optimum),
        float(np.abs(costs).max(initial=0.0)) * match_count,
    )
    tolerance = np.finfo(np.float64).eps * 64.0 * scale

    available_columns = list(range(column_count))
    fixed_costs = []
    assignments = []
    for row in range(row_count):
        candidates = list(available_columns)
        if row_count > column_count:
            candidates.append(None)

        found = False
        selected = None
        for candidate in candidates:
            remaining_columns = [
                column for column in available_columns if column != candidate
            ]
            remaining_rows = list(range(row + 1, row_count))
            matched_here = candidate is not None
            remaining_needed = match_count - len(assignments) - int(matched_here)
            if not 0 <= remaining_needed <= min(
                len(remaining_rows),
                len(remaining_columns),
            ):
                continue

            completion_cost = 0.0
            if remaining_needed:
                completion = costs[np.ix_(remaining_rows, remaining_columns)]
                completion_rows, completion_columns = linear_sum_assignment(completion)
                if len(completion_rows) != remaining_needed:
                    continue
                completion_cost = _assignment_cost(
                    completion,
                    completion_rows,
                    completion_columns,
                )
            candidate_costs = fixed_costs + (
                [float(costs[row, candidate])] if matched_here else []
            )
            total = math.fsum(candidate_costs) + completion_cost
            if math.isclose(total, optimum, rel_tol=0.0, abs_tol=tolerance):
                found = True
                selected = candidate
                break

        if not found:
            raise RuntimeError("could not refine an optimal assignment deterministically")
        if selected is not None:
            assignments.append((row, selected))
            fixed_costs.append(float(costs[row, selected]))
            available_columns.remove(selected)

    if len(assignments) != match_count:
        raise RuntimeError("deterministic assignment returned the wrong cardinality")
    return tuple(assignments)


@dataclass
class PrefixTrajectorySupervisionState:
    """Mutable canonical lifecycle state with atomic prefix transitions.

    ``slot_exhaustion_count`` counts unallocated birth instances.  The rematch
    counter counts displaced instance bindings (a two-way exchange adds two).
    """

    num_slots: int
    mode: SupervisionMode | str
    instance_to_slot: Dict[int, int] = field(default_factory=dict)
    slot_to_instance: Dict[int, int] = field(default_factory=dict)
    born_instance_ids: Set[int] = field(default_factory=set)
    retired_instance_ids: Set[int] = field(default_factory=set)
    slot_exhaustion_count: int = 0
    rematch_swap_count: int = 0
    transition_count: int = 0
    last_current_frame: Optional[int] = None

    def __post_init__(self):
        self.num_slots = int(self.num_slots)
        if self.num_slots <= 0:
            raise ValueError("num_slots must be positive")
        try:
            self.mode = SupervisionMode(self.mode)
        except ValueError as exc:
            raise ValueError("mode must be 'fixed' or 'rematch'") from exc
        self.validate()

    def validate(self):
        """Validate canonical bijections, lifecycle sets, slots, and counters."""
        if len(set(self.instance_to_slot.values())) != len(self.instance_to_slot):
            raise SupervisionInvariantError(
                "instance_to_slot must be a bijection without duplicate slots"
            )
        expected_reverse = {
            slot_id: instance_id
            for instance_id, slot_id in self.instance_to_slot.items()
        }
        if expected_reverse != self.slot_to_instance:
            raise SupervisionInvariantError(
                "instance_to_slot and slot_to_instance must be exact inverses"
            )
        for instance_id, slot_id in self.instance_to_slot.items():
            if type(instance_id) is not int or type(slot_id) is not int:
                raise SupervisionInvariantError("instance and slot ids must be integers")
            if not 0 <= slot_id < self.num_slots:
                raise SupervisionInvariantError(
                    f"slot {slot_id} is outside configured capacity {self.num_slots}"
                )
        if not set(self.instance_to_slot).issubset(self.born_instance_ids):
            raise SupervisionInvariantError("occupied instances must already be born")
        if not self.retired_instance_ids.issubset(self.born_instance_ids):
            raise SupervisionInvariantError("retired instances must already be born")
        if set(self.instance_to_slot).intersection(self.retired_instance_ids):
            raise SupervisionInvariantError("retired instances cannot occupy canonical slots")
        for name in (
            "slot_exhaustion_count",
            "rematch_swap_count",
            "transition_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise SupervisionInvariantError(f"{name} must be a non-negative integer")
        if self.last_current_frame is not None and type(self.last_current_frame) is not int:
            raise SupervisionInvariantError("last_current_frame must be an integer or None")

    def clone(self):
        """Return a mutable deep clone with independent maps and sets."""
        return type(self)(
            num_slots=self.num_slots,
            mode=self.mode,
            instance_to_slot=dict(self.instance_to_slot),
            slot_to_instance=dict(self.slot_to_instance),
            born_instance_ids=set(self.born_instance_ids),
            retired_instance_ids=set(self.retired_instance_ids),
            slot_exhaustion_count=self.slot_exhaustion_count,
            rematch_swap_count=self.rematch_swap_count,
            transition_count=self.transition_count,
            last_current_frame=self.last_current_frame,
        )

    def snapshot(self):
        """Return an immutable deep snapshot of canonical state."""
        return PrefixTrajectoryStateSnapshot(
            num_slots=self.num_slots,
            mode=self.mode,
            instance_to_slot=tuple(sorted(self.instance_to_slot.items())),
            slot_to_instance=tuple(sorted(self.slot_to_instance.items())),
            born_instance_ids=frozenset(self.born_instance_ids),
            retired_instance_ids=frozenset(self.retired_instance_ids),
            slot_exhaustion_count=self.slot_exhaustion_count,
            rematch_swap_count=self.rematch_swap_count,
            transition_count=self.transition_count,
            last_current_frame=self.last_current_frame,
        )

    def transition(
        self,
        schedule_step,
        cost_matrix=None,
        *,
        cost_provider=None,
        available_slots=None,
    ):
        """Apply one prefix step atomically and return pre-retirement bindings.

        Births are globally assigned to slots free at entry.  Endpoint instances
        remain occupied while bindings/masks are built, and retire only after the
        immutable result has been constructed.  Pass a callable either
        positionally or as ``cost_provider=``; dense matrices may be positional
        or use ``cost_matrix=``.
        """
        self.validate()
        if cost_provider is not None:
            if cost_matrix is not None:
                raise ValueError("pass exactly one of cost_matrix or cost_provider")
            cost_matrix = cost_provider
        if cost_matrix is None:
            raise ValueError("a cost_matrix or cost_provider is required")
        next_state = self.clone()
        result = next_state._transition_in_place(
            schedule_step,
            cost_matrix,
            available_slots=available_slots,
        )
        self.instance_to_slot = dict(next_state.instance_to_slot)
        self.slot_to_instance = dict(next_state.slot_to_instance)
        self.born_instance_ids = set(next_state.born_instance_ids)
        self.retired_instance_ids = set(next_state.retired_instance_ids)
        self.slot_exhaustion_count = next_state.slot_exhaustion_count
        self.rematch_swap_count = next_state.rematch_swap_count
        self.transition_count = next_state.transition_count
        self.last_current_frame = next_state.last_current_frame
        return result

    def _transition_in_place(self, schedule_step, cost_matrix, *, available_slots=None):
        self.validate()
        current_frame = int(schedule_step.current_frame)
        if self.last_current_frame is not None and current_frame <= self.last_current_frame:
            raise SupervisionInvariantError(
                "schedule current_frame must increase strictly across transitions"
            )
        occupied_slots_before = tuple(sorted(self.slot_to_instance))
        birth_items = tuple(schedule_step.births)
        active_items = tuple(schedule_step.active)
        end_items = tuple(schedule_step.ends)
        early_step_ids = {
            int(item.instance_id)
            for item in birth_items + active_items + end_items
        }
        resurrected = tuple(sorted(self.retired_instance_ids.intersection(early_step_ids)))
        if resurrected:
            raise SupervisionInvariantError(
                f"retired instance cannot reappear in a schedule: {resurrected}"
            )
        for item in birth_items + active_items + end_items:
            start_frame = float(item.start_frame)
            if not math.isfinite(start_frame) or start_frame > current_frame:
                raise SupervisionInvariantError(
                    f"instance {item.instance_id} contains a future start frame"
                )
        if self.last_current_frame is not None:
            delayed_births = tuple(
                sorted(
                    int(item.instance_id)
                    for item in birth_items
                    if float(item.start_frame) <= self.last_current_frame
                )
            )
            if delayed_births:
                raise SupervisionInvariantError(
                    f"birth crossings must occur after the previous decision: {delayed_births}"
                )
        leaked = tuple(
            sorted(
                int(item.instance_id)
                for item in birth_items + active_items
                if getattr(item, "end_frame", None) is not None
            )
        )
        if leaked:
            raise SupervisionInvariantError(
                f"birth/active schedule entries contain a future endpoint: {leaked}"
            )
        invalid_ends = []
        for item in end_items:
            end_frame = getattr(item, "end_frame", None)
            if end_frame is None:
                invalid_ends.append(int(item.instance_id))
                continue
            end_frame = float(end_frame)
            if (
                not math.isfinite(end_frame)
                or end_frame <= float(item.start_frame)
                or end_frame > current_frame
            ):
                invalid_ends.append(int(item.instance_id))
                continue
            if self.last_current_frame is not None and end_frame <= self.last_current_frame:
                invalid_ends.append(int(item.instance_id))
        if invalid_ends:
            raise SupervisionInvariantError(
                "end schedule entries contain a future endpoint or invalid endpoint crossing: "
                f"{tuple(sorted(invalid_ends))}"
            )
        births = tuple(sorted(int(item.instance_id) for item in birth_items))
        active = tuple(sorted(int(item.instance_id) for item in active_items))
        ends = tuple(sorted(int(item.instance_id) for item in end_items))
        for channel, instance_ids in (
            ("birth", births),
            ("active", active),
            ("end", ends),
        ):
            if len(instance_ids) != len(set(instance_ids)):
                raise SupervisionInvariantError(
                    f"{channel} schedule entries contain duplicate instance ids"
                )
        if set(active).intersection(ends):
            raise SupervisionInvariantError(
                "an instance cannot be active and end in the same schedule step"
            )
        known_after_birth = self.born_instance_ids.union(births)
        unknown = tuple(sorted(set(active + ends).difference(known_after_birth)))
        if unknown:
            raise SupervisionInvariantError(
                f"active/end schedule entries reference unknown instances: {unknown}"
            )
        step_ids = tuple(sorted(set(births + active + ends)))
        repeated_births = tuple(sorted(self.born_instance_ids.intersection(births)))
        if repeated_births:
            raise SupervisionInvariantError(
                f"each instance may have only one birth crossing: {repeated_births}"
            )
        visible_set = set(active + ends)
        missing_births = tuple(sorted(set(births).difference(visible_set)))
        if missing_births:
            raise SupervisionInvariantError(
                f"birth instances must be visible as active or ending: {missing_births}"
            )
        missing_occupied = tuple(
            sorted(set(self.instance_to_slot).difference(visible_set))
        )
        if missing_occupied:
            raise SupervisionInvariantError(
                "occupied canonical instances must remain visible until retirement: "
                f"{missing_occupied}"
            )
        row_by_instance = {instance_id: row for row, instance_id in enumerate(step_ids)}
        provider = cost_matrix if callable(cost_matrix) else None
        costs = None
        if provider is None:
            costs = _validated_cost_matrix(
                cost_matrix,
                (len(step_ids), self.num_slots),
                "dense",
            )

        def phase_costs(phase, instance_ids, slot_ids):
            if provider is not None:
                return _validated_cost_matrix(
                    provider(phase, instance_ids, slot_ids),
                    (len(instance_ids), len(slot_ids)),
                    phase,
                )
            return costs[
                np.ix_(
                    [row_by_instance[item] for item in instance_ids],
                    slot_ids,
                )
            ]

        if available_slots is None:
            runtime_available = set(range(self.num_slots))
        else:
            available_slots = tuple(available_slots)
            if (
                any(type(slot) is not int for slot in available_slots)
                or len(set(available_slots)) != len(available_slots)
                or any(not 0 <= slot < self.num_slots for slot in available_slots)
            ):
                raise SupervisionInvariantError(
                    "available_slots must contain unique in-capacity integer slots"
                )
            runtime_available = set(available_slots)
        birth_candidates = tuple(
            slot
            for slot in range(self.num_slots)
            if slot not in self.slot_to_instance and slot in runtime_available
        )
        birth_assignments = ()
        if births and birth_candidates:
            birth_costs = phase_costs("birth", births, birth_candidates)
            assignment = _solve_global_assignment(birth_costs)
            birth_assignments = tuple(
                sorted(
                    InstanceSlotBinding(births[row], birth_candidates[column])
                    for row, column in assignment
                )
            )
        for binding in birth_assignments:
            self.instance_to_slot[binding.instance_id] = binding.slot_id
            self.slot_to_instance[binding.slot_id] = binding.instance_id
        self.born_instance_ids.update(births)

        visible = tuple(sorted(set(active + ends)))
        canonical_bindings = tuple(
            InstanceSlotBinding(instance_id, self.instance_to_slot[instance_id])
            for instance_id in visible
            if instance_id in self.instance_to_slot
        )
        loss_bindings = canonical_bindings
        if self.mode is SupervisionMode.REMATCH and canonical_bindings:
            canonical_instance_ids = tuple(
                binding.instance_id for binding in canonical_bindings
            )
            occupied_slots = tuple(sorted(binding.slot_id for binding in canonical_bindings))
            rematch_costs = phase_costs(
                "rematch",
                canonical_instance_ids,
                occupied_slots,
            )
            assignment = _solve_global_assignment(rematch_costs)
            loss_bindings = tuple(
                sorted(
                    InstanceSlotBinding(
                        canonical_instance_ids[row],
                        occupied_slots[column],
                    )
                    for row, column in assignment
                )
            )

        loss_by_instance = {
            binding.instance_id: binding.slot_id for binding in loss_bindings
        }
        at_risk_slots = tuple(sorted(loss_by_instance.values()))
        endpoint_slots = tuple(
            sorted(loss_by_instance[item] for item in ends if item in loss_by_instance)
        )
        exhausted = tuple(
            instance_id
            for instance_id in births
            if instance_id not in self.instance_to_slot
        )
        self.slot_exhaustion_count += len(exhausted)
        rematch_swapped = tuple(
            binding.instance_id
            for binding in loss_bindings
            if binding.slot_id != self.instance_to_slot[binding.instance_id]
        )
        self.rematch_swap_count += len(rematch_swapped)
        occupied_for_supervision = tuple(sorted(self.slot_to_instance))
        ending_slots = {
            self.instance_to_slot[instance_id]
            for instance_id in ends
            if instance_id in self.instance_to_slot
        }
        occupied_after = tuple(
            slot for slot in occupied_for_supervision if slot not in ending_slots
        )
        audit = PrefixTrajectoryTransitionAudit(
            transition_index=self.transition_count + 1,
            current_frame=current_frame,
            mode=self.mode,
            visible_instance_ids=visible,
            occupied_slots_before=occupied_slots_before,
            occupied_slots_for_supervision=occupied_for_supervision,
            occupied_slots_after=occupied_after,
            retired_instance_ids=ends,
            exhausted_instance_ids=exhausted,
            rematch_swapped_instance_ids=rematch_swapped,
            slot_exhaustion_total=self.slot_exhaustion_count,
            rematch_swap_total=self.rematch_swap_count,
        )

        result = PrefixTrajectoryTransitionResult(
            canonical_bindings=canonical_bindings,
            loss_bindings=loss_bindings,
            birth_instance_ids=births,
            birth_candidates=birth_candidates,
            birth_assignments=birth_assignments,
            at_risk_slots=at_risk_slots,
            endpoint_slots=endpoint_slots,
            birth_mask=_mask(self.num_slots, birth_candidates),
            at_risk_mask=_mask(self.num_slots, at_risk_slots),
            endpoint_mask=_mask(self.num_slots, endpoint_slots),
            exhausted_instance_ids=exhausted,
            exhaustion=len(exhausted),
            audit=audit,
        )

        for instance_id in ends:
            slot = self.instance_to_slot.pop(instance_id, None)
            if slot is not None:
                self.slot_to_instance.pop(slot)
            self.retired_instance_ids.add(instance_id)
        self.transition_count += 1
        self.last_current_frame = current_frame
        self.validate()
        return result


__all__ = [
    "InstanceSlotBinding",
    "PrefixTrajectoryStateSnapshot",
    "PrefixTrajectorySupervisionState",
    "PrefixTrajectoryTransitionAudit",
    "PrefixTrajectoryTransitionResult",
    "SupervisionInvariantError",
    "SupervisionMode",
]
