"""Zero-GPU capacity and lifecycle diagnostics for Full PETAL Q2.

The helpers in this module deliberately separate numerical head recurrence
from the discrete runtime controller.  That separation is valid for the
current ``PersistentEventSetHead`` only: its logits do not read lifecycle
tensors.  Callers must re-audit that invariant after changing the head.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import math

import torch
import torch.nn.functional as F

AUDIT_SCHEMA_VERSION = "q2-capacity-lifecycle-audit-v1"
CHECKPOINT_SCHEMA_VERSION = "q2-capacity-init-checkpoints-v1"

# Frozen against persistent_event_set_head.py and asserted in unit tests.
SLOT_FREE = 0
SLOT_ACTIVE = 1
SLOT_REFRACTORY = 2

TRUE_CANONICAL_CAPACITY = "TRUE_CANONICAL_CAPACITY"
FALSE_ACTIVE_OCCUPANCY = "FALSE_ACTIVE_OCCUPANCY"
REFRACTORY_OCCUPANCY = "REFRACTORY_OCCUPANCY"
SAME_BIN_NON_REUSE = "SAME_BIN_NON_REUSE"
MIXED_CAUSE = "MIXED_CAUSE"
EXHAUSTION_CAUSES = (
    TRUE_CANONICAL_CAPACITY,
    FALSE_ACTIVE_OCCUPANCY,
    REFRACTORY_OCCUPANCY,
    SAME_BIN_NON_REUSE,
    MIXED_CAUSE,
)


class Q2CapacityAuditError(ValueError):
    """Raised when a capacity audit contract or trace is malformed."""


@dataclass(frozen=True)
class Q2LifecyclePolicy:
    name: str
    num_slots: int = 4
    birth_threshold: float = 0.5
    alive_threshold: float = 0.5
    end_threshold: float = 0.5
    refractory_steps: int = 2
    birth_logit_bias: float = 0.0
    release_before_birth: bool = False
    privileged_canonical_availability: bool = False
    eligible_shared_contract: bool = True

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise Q2CapacityAuditError("policy name must be non-empty")
        if int(self.num_slots) != self.num_slots or self.num_slots <= 0:
            raise Q2CapacityAuditError("policy num_slots must be positive")
        for field_name in (
            "birth_threshold",
            "alive_threshold",
            "end_threshold",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise Q2CapacityAuditError(f"policy {field_name} must lie in [0,1]")
        if self.refractory_steps < 0:
            raise Q2CapacityAuditError("refractory_steps must be non-negative")
        if not math.isfinite(float(self.birth_logit_bias)):
            raise Q2CapacityAuditError("birth_logit_bias must be finite")
        if self.privileged_canonical_availability and self.eligible_shared_contract:
            raise Q2CapacityAuditError(
                "privileged canonical availability can never be a shared trainer contract"
            )

    def as_record(self):
        return asdict(self)


@dataclass(frozen=True)
class Q2LifecycleState:
    slot_status: tuple[int, ...]
    refractory: tuple[int, ...]

    def __post_init__(self):
        if not self.slot_status or len(self.slot_status) != len(self.refractory):
            raise Q2CapacityAuditError(
                "lifecycle state tensors must be non-empty and aligned"
            )
        if any(
            int(value) not in {SLOT_FREE, SLOT_ACTIVE, SLOT_REFRACTORY}
            for value in self.slot_status
        ):
            raise Q2CapacityAuditError(
                "lifecycle state contains an unknown slot status"
            )
        if any(int(value) < 0 for value in self.refractory):
            raise Q2CapacityAuditError("refractory counters must be non-negative")

    @classmethod
    def initial(cls, num_slots):
        num_slots = int(num_slots)
        if num_slots <= 0:
            raise Q2CapacityAuditError("num_slots must be positive")
        return cls((SLOT_FREE,) * num_slots, (0,) * num_slots)


@dataclass(frozen=True)
class Q2LifecycleStep:
    available_slots: tuple[int, ...]
    state_after: Q2LifecycleState
    emitted_slots: tuple[int, ...]
    birth_probabilities: tuple[float, ...]
    alive_probabilities: tuple[float, ...]
    end_probabilities: tuple[float, ...]


@dataclass(frozen=True)
class ExhaustionAttribution:
    cause_by_instance: tuple[tuple[int, str], ...]
    blocker_slots: tuple[tuple[str, tuple[int, ...]], ...]
    attribution_closed: bool

    @property
    def counts(self):
        return dict(Counter(cause for _, cause in self.cause_by_instance))

    def as_record(self):
        return {
            "cause_by_instance": [
                {"instance_id": instance_id, "cause": cause}
                for instance_id, cause in self.cause_by_instance
            ],
            "blocker_slots": {
                cause: list(slots) for cause, slots in self.blocker_slots
            },
            "attribution_closed": self.attribution_closed,
        }


def frozen_policy_grid(num_slots=4):
    """Return the outcome-blind policy grid frozen before capacity outcomes."""

    num_slots = int(num_slots)
    policies = (
        Q2LifecyclePolicy("actual", num_slots=num_slots),
        Q2LifecyclePolicy("refractory_0", num_slots=num_slots, refractory_steps=0),
        Q2LifecyclePolicy(
            "birth_threshold_0p25", num_slots=num_slots, birth_threshold=0.25
        ),
        Q2LifecyclePolicy(
            "birth_threshold_0p75", num_slots=num_slots, birth_threshold=0.75
        ),
        Q2LifecyclePolicy(
            "alive_threshold_0p25", num_slots=num_slots, alive_threshold=0.25
        ),
        Q2LifecyclePolicy(
            "alive_threshold_0p75", num_slots=num_slots, alive_threshold=0.75
        ),
        Q2LifecyclePolicy(
            "end_threshold_0p25", num_slots=num_slots, end_threshold=0.25
        ),
        Q2LifecyclePolicy(
            "end_threshold_0p75", num_slots=num_slots, end_threshold=0.75
        ),
        Q2LifecyclePolicy(
            "birth_prior_bias_m1", num_slots=num_slots, birth_logit_bias=-1.0
        ),
        Q2LifecyclePolicy(
            "birth_prior_bias_m2", num_slots=num_slots, birth_logit_bias=-2.0
        ),
        Q2LifecyclePolicy(
            "release_before_birth",
            num_slots=num_slots,
            release_before_birth=True,
        ),
        Q2LifecyclePolicy(
            "release_before_birth_refractory_0",
            num_slots=num_slots,
            refractory_steps=0,
            release_before_birth=True,
        ),
        Q2LifecyclePolicy(
            "canonical_only_privileged",
            num_slots=num_slots,
            privileged_canonical_availability=True,
            eligible_shared_contract=False,
        ),
    )
    contractions = tuple(
        Q2LifecyclePolicy(
            f"capacity_k{capacity}",
            num_slots=capacity,
            eligible_shared_contract=False,
        )
        for capacity in range(2, num_slots)
    )
    result = policies + contractions
    names = [policy.name for policy in result]
    if len(names) != len(set(names)):
        raise Q2CapacityAuditError("frozen policy grid contains duplicate names")
    return result


def adjusted_outputs(outputs, policy):
    """Apply only the preregistered outcome-blind birth-logit prior shift."""

    adjusted = dict(outputs)
    if policy.birth_logit_bias:
        adjusted["birth_logits"] = outputs["birth_logits"] + float(
            policy.birth_logit_bias
        )
    return adjusted


def build_assignment_cost_provider(
    outputs,
    schedule_step,
    *,
    feature_stride,
    memory_size,
):
    """Build the exact fixed/rematch assignment cost used by Q2 training."""

    targets = {}
    for item in (
        tuple(schedule_step.births)
        + tuple(schedule_step.active)
        + tuple(schedule_step.ends)
    ):
        targets[int(item.instance_id)] = item
    class_log_probs = outputs["class_logits"][0].log_softmax(dim=-1)

    def provider(phase, instance_ids, slot_ids):
        rows = []
        for instance_id in instance_ids:
            target = targets[int(instance_id)]
            target_offset = (
                float(schedule_step.current_frame) - float(target.start_frame)
            ) / max(float(feature_stride), 1.0)
            target_offset = min(max(target_offset, 0.0), float(memory_size))
            row = []
            for slot in slot_ids:
                class_cost = -class_log_probs[int(slot), int(target.label)]
                start_cost = F.smooth_l1_loss(
                    outputs["start_offset"][0, int(slot)],
                    outputs["start_offset"].new_tensor(target_offset),
                    reduction="sum",
                )
                cost = class_cost + start_cost
                if phase == "birth":
                    cost = cost - F.logsigmoid(outputs["birth_logits"][0, int(slot)])
                elif phase != "rematch":
                    raise Q2CapacityAuditError(
                        f"unknown supervision assignment phase {phase!r}"
                    )
                row.append(cost.detach())
            rows.append(torch.stack(row))
        if not rows:
            return torch.empty((0, len(slot_ids))).numpy()
        return torch.stack(rows).float().cpu().numpy()

    return provider


def _probabilities(outputs, policy):
    num_slots = int(policy.num_slots)
    required = ("birth_logits", "alive_logits", "end_hazard_logits")
    for key in required:
        value = outputs.get(key)
        if not torch.is_tensor(value) or value.ndim != 2 or value.shape[0] != 1:
            raise Q2CapacityAuditError(f"{key} must have shape [1,K]")
        if value.shape[1] < num_slots:
            raise Q2CapacityAuditError(
                "same-logits replay cannot expand beyond checkpoint slot capacity"
            )
    adjusted = adjusted_outputs(outputs, policy)
    values = []
    for key in required:
        tensor = adjusted[key][0, :num_slots].detach().float().sigmoid().cpu()
        if not torch.isfinite(tensor).all():
            raise Q2CapacityAuditError("lifecycle probabilities must be finite")
        values.append(tuple(float(item) for item in tensor.tolist()))
    return tuple(values)


def _actual_decode(state, birth, alive, end, policy):
    status = list(state.slot_status)
    refractory = list(state.refractory)
    emitted = []
    for slot in range(policy.num_slots):
        just_born = False
        if status[slot] == SLOT_REFRACTORY:
            if refractory[slot] > 0:
                refractory[slot] -= 1
                continue
            if (
                alive[slot] < policy.alive_threshold
                and birth[slot] < policy.birth_threshold
            ):
                status[slot] = SLOT_FREE
            continue
        if status[slot] == SLOT_FREE:
            if birth[slot] < policy.birth_threshold:
                continue
            status[slot] = SLOT_ACTIVE
            just_born = True
        if (
            not just_born
            and alive[slot] < policy.alive_threshold
            and end[slot] < policy.end_threshold
        ):
            status[slot] = SLOT_FREE
            continue
        if end[slot] < policy.end_threshold:
            continue
        emitted.append(slot)
        status[slot] = SLOT_REFRACTORY
        refractory[slot] = int(policy.refractory_steps)
    return Q2LifecycleState(tuple(status), tuple(refractory)), tuple(emitted)


def _release_before_birth_decode(state, birth, alive, end, policy):
    """Run inference-visible release/emission before evaluating new births."""

    status = list(state.slot_status)
    refractory = list(state.refractory)
    emitted = []
    for slot in range(policy.num_slots):
        if status[slot] == SLOT_FREE:
            continue
        if status[slot] == SLOT_REFRACTORY:
            if refractory[slot] > 0:
                refractory[slot] -= 1
            elif (
                alive[slot] < policy.alive_threshold
                and birth[slot] < policy.birth_threshold
            ):
                status[slot] = SLOT_FREE
            continue
        if alive[slot] < policy.alive_threshold and end[slot] < policy.end_threshold:
            status[slot] = SLOT_FREE
            continue
        if end[slot] >= policy.end_threshold:
            emitted.append(slot)
            if policy.refractory_steps:
                status[slot] = SLOT_REFRACTORY
                refractory[slot] = int(policy.refractory_steps)
            else:
                status[slot] = SLOT_FREE
                refractory[slot] = 0

    available = tuple(slot for slot, value in enumerate(status) if value == SLOT_FREE)
    for slot in available:
        if birth[slot] < policy.birth_threshold:
            continue
        status[slot] = SLOT_ACTIVE
        if end[slot] >= policy.end_threshold:
            emitted.append(slot)
            if policy.refractory_steps:
                status[slot] = SLOT_REFRACTORY
                refractory[slot] = int(policy.refractory_steps)
            else:
                status[slot] = SLOT_FREE
                refractory[slot] = 0
    return (
        available,
        Q2LifecycleState(tuple(status), tuple(refractory)),
        tuple(sorted(set(emitted))),
    )


def replay_lifecycle_step(state, outputs, policy):
    """Replay one controller step and expose pre-assignment availability."""

    if len(state.slot_status) != policy.num_slots:
        raise Q2CapacityAuditError("lifecycle state capacity differs from policy")
    birth, alive, end = _probabilities(outputs, policy)
    if policy.release_before_birth:
        available, state_after, emitted = _release_before_birth_decode(
            state, birth, alive, end, policy
        )
    else:
        available = tuple(
            slot for slot, value in enumerate(state.slot_status) if value == SLOT_FREE
        )
        state_after, emitted = _actual_decode(state, birth, alive, end, policy)
    return Q2LifecycleStep(
        available_slots=available,
        state_after=state_after,
        emitted_slots=emitted,
        birth_probabilities=birth,
        alive_probabilities=alive,
        end_probabilities=end,
    )


def canonical_available_slots(supervision_state, num_slots):
    occupied = set(int(slot) for slot in supervision_state.slot_to_instance)
    return tuple(slot for slot in range(int(num_slots)) if slot not in occupied)


def attribute_exhaustions(
    exhausted_instance_ids,
    birth_instance_ids,
    ending_instance_ids,
    canonical_before,
    runtime_before,
    available_slots,
):
    """Assign every exhausted birth one preregistered root-cause label."""

    exhausted = tuple(sorted(int(value) for value in exhausted_instance_ids))
    births = tuple(sorted(int(value) for value in birth_instance_ids))
    endings = set(int(value) for value in ending_instance_ids)
    canonical = {
        int(instance): int(slot) for instance, slot in canonical_before.items()
    }
    available = set(int(slot) for slot in available_slots)
    if not exhausted:
        return ExhaustionAttribution((), (), True)
    if not set(exhausted).issubset(births):
        raise Q2CapacityAuditError("exhausted IDs must be current birth IDs")
    if len(runtime_before.slot_status) == 0:
        raise Q2CapacityAuditError("runtime capacity must be positive")

    capacity = len(runtime_before.slot_status)
    nonending = set(canonical).difference(endings)
    structural_count = min(
        len(exhausted),
        max(0, len(nonending) + len(births) - capacity),
    )
    assignments = [
        (instance_id, TRUE_CANONICAL_CAPACITY)
        for instance_id in exhausted[:structural_count]
    ]
    remaining = exhausted[structural_count:]

    canonical_slots = set(canonical.values())
    same_bin_slots = tuple(
        sorted(
            canonical[instance_id]
            for instance_id in endings.intersection(canonical)
            if canonical[instance_id] not in available
        )
    )
    false_active_slots = tuple(
        slot
        for slot, status in enumerate(runtime_before.slot_status)
        if slot not in canonical_slots
        and slot not in available
        and status == SLOT_ACTIVE
    )
    refractory_slots = tuple(
        slot
        for slot, status in enumerate(runtime_before.slot_status)
        if slot not in canonical_slots
        and slot not in available
        and status == SLOT_REFRACTORY
    )
    blocker_slots = (
        (SAME_BIN_NON_REUSE, same_bin_slots),
        (FALSE_ACTIVE_OCCUPANCY, false_active_slots),
        (REFRACTORY_OCCUPANCY, refractory_slots),
    )
    present = [cause for cause, slots in blocker_slots if slots]
    closed = True
    if remaining:
        if len(present) == 1:
            assignments.extend((instance_id, present[0]) for instance_id in remaining)
        elif present:
            assignments.extend((instance_id, MIXED_CAUSE) for instance_id in remaining)
        else:
            assignments.extend((instance_id, MIXED_CAUSE) for instance_id in remaining)
            closed = False
    if len(assignments) != len(exhausted):
        raise Q2CapacityAuditError("exhaustion attribution lost an instance")
    if any(cause not in EXHAUSTION_CAUSES for _, cause in assignments):
        raise Q2CapacityAuditError("exhaustion attribution emitted an unknown cause")
    return ExhaustionAttribution(tuple(assignments), blocker_slots, closed)


def audit_annotation_schedule(
    video_id,
    schedule,
    *,
    num_slots,
    refractory_steps,
):
    """Compute annotation-only per-bin demand without any runtime prediction."""

    occupied = set()
    labels = {}
    reserve = []
    rows = []
    maxima = Counter()
    for bin_index, step in enumerate(schedule):
        births = tuple(sorted(int(item.instance_id) for item in step.births))
        ends = tuple(sorted(int(item.instance_id) for item in step.ends))
        visible_items = tuple(step.active) + tuple(step.ends)
        for item in tuple(step.births) + visible_items:
            labels[int(item.instance_id)] = int(item.label)
        visible = tuple(sorted({int(item.instance_id) for item in visible_items}))
        ending = set(ends)
        nonending = occupied.difference(ending)
        oracle_free_demand = len(nonending) + len(births)
        current_order_demand = len(occupied) + len(births)
        refractory_reserve = len(reserve)
        label_counts = Counter(labels[item] for item in visible)
        same_class = max(label_counts.values(), default=0)
        distinct_classes = len(label_counts)
        row = {
            "video_id": str(video_id),
            "bin_index": int(bin_index),
            "current_frame": int(step.current_frame),
            "gt_active_count": len(visible),
            "gt_birth_count": len(births),
            "gt_end_count": len(ends),
            "same_bin_birth_end_count": min(len(births), len(ends)),
            "same_class_concurrency": same_class,
            "cross_class_concurrency": distinct_classes,
            "canonical_occupied_before": len(occupied),
            "canonical_occupied_after": len(
                (occupied | set(births)).difference(ending)
            ),
            "canonical_with_refractory_reserve": current_order_demand
            + refractory_reserve,
            "minimum_oracle_free_k": oracle_free_demand,
            "current_order_required_k": current_order_demand,
        }
        rows.append(row)
        for key in (
            "gt_active_count",
            "gt_birth_count",
            "gt_end_count",
            "same_bin_birth_end_count",
            "same_class_concurrency",
            "cross_class_concurrency",
            "canonical_with_refractory_reserve",
            "minimum_oracle_free_k",
            "current_order_required_k",
        ):
            maxima[key] = max(maxima[key], row[key])
        occupied = (occupied | set(births)).difference(ending)
        reserve = [remaining - 1 for remaining in reserve if remaining - 1 > 0]
        if refractory_steps > 0:
            reserve.extend([int(refractory_steps)] * len(ends))
    if occupied:
        raise Q2CapacityAuditError(
            f"annotation schedule for {video_id} ended with live instances: {sorted(occupied)}"
        )
    summary = {
        "video_id": str(video_id),
        "num_bins": len(rows),
        "configured_num_slots": int(num_slots),
        "maxima": dict(maxima),
        "oracle_k4_structurally_sufficient": maxima["minimum_oracle_free_k"]
        <= int(num_slots),
    }
    return summary, tuple(rows)


def policy_summary_template(policy):
    return {
        "policy": policy.as_record(),
        "bins": 0,
        "births": 0,
        "assignments": 0,
        "exhaustions": 0,
        "emissions": 0,
        "false_active_slot_bins": 0,
        "refractory_slot_bins": 0,
        "cause_counts": {cause: 0 for cause in EXHAUSTION_CAUSES},
        "attribution_closed": True,
        "fixed_rematch_canonical_equal": True,
    }


def state_dict_fingerprint(state_dict):
    records = []
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not torch.is_tensor(tensor):
            raise Q2CapacityAuditError(f"state value is not a tensor: {name}")
        contiguous = tensor.detach().cpu().contiguous()
        payload = contiguous.numpy().tobytes(order="C")
        records.append(
            {
                "name": name,
                "dtype": str(contiguous.dtype),
                "shape": list(contiguous.shape),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    encoded = json_canonical_bytes(records)
    return hashlib.sha256(encoded).hexdigest()


def json_canonical_bytes(value):
    import json

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Q2CapacityAuditError(f"value is not canonical JSON: {exc}") from exc


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "CHECKPOINT_SCHEMA_VERSION",
    "EXHAUSTION_CAUSES",
    "FALSE_ACTIVE_OCCUPANCY",
    "MIXED_CAUSE",
    "Q2CapacityAuditError",
    "Q2LifecyclePolicy",
    "Q2LifecycleState",
    "Q2LifecycleStep",
    "REFRACTORY_OCCUPANCY",
    "SAME_BIN_NON_REUSE",
    "TRUE_CANONICAL_CAPACITY",
    "adjusted_outputs",
    "attribute_exhaustions",
    "audit_annotation_schedule",
    "build_assignment_cost_provider",
    "canonical_available_slots",
    "frozen_policy_grid",
    "policy_summary_template",
    "replay_lifecycle_step",
    "state_dict_fingerprint",
]
