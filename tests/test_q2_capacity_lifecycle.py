import pytest
import torch

from opentad.cores.train_engine import train_one_epoch
from opentad.models.dense_heads.persistent_event_set_head import (
    SLOT_ACTIVE as HEAD_SLOT_ACTIVE,
    SLOT_FREE as HEAD_SLOT_FREE,
    SLOT_REFRACTORY as HEAD_SLOT_REFRACTORY,
    PersistentEventSetHead,
)
from opentad.utils.prefix_instance_schedule import build_prefix_instance_schedule
from opentad.utils.prefix_trajectory_supervision import (
    PrefixTrajectorySupervisionState,
    SupervisionMode,
)
from opentad.utils.q2_capacity_audit import (
    FALSE_ACTIVE_OCCUPANCY,
    MIXED_CAUSE,
    Q2LifecyclePolicy,
    Q2LifecycleState,
    REFRACTORY_OCCUPANCY,
    SAME_BIN_NON_REUSE,
    SLOT_ACTIVE,
    SLOT_FREE,
    SLOT_REFRACTORY,
    TRUE_CANONICAL_CAPACITY,
    attribute_exhaustions,
    audit_annotation_schedule,
    build_assignment_cost_provider,
    frozen_policy_grid,
    replay_lifecycle_step,
)


def _outputs(*, birth, alive, end, num_classes=3):
    num_slots = len(birth)
    return {
        "birth_logits": torch.tensor([birth], dtype=torch.float32),
        "alive_logits": torch.tensor([alive], dtype=torch.float32),
        "end_hazard_logits": torch.tensor([end], dtype=torch.float32),
        "class_logits": torch.zeros((1, num_slots, num_classes)),
        "start_offset": torch.zeros((1, num_slots)),
    }


def test_capacity_slot_constants_remain_frozen_to_production_head():
    assert (SLOT_FREE, SLOT_ACTIVE, SLOT_REFRACTORY) == (
        HEAD_SLOT_FREE,
        HEAD_SLOT_ACTIVE,
        HEAD_SLOT_REFRACTORY,
    )


def test_actual_replay_matches_q2_style_head_with_dropout_and_real_thresholds():
    torch.manual_seed(705)
    head = PersistentEventSetHead(
        in_channels=8,
        hidden_dim=16,
        num_classes=3,
        num_slots=4,
        memory_size=8,
        num_heads=4,
        dropout=0.1,
        query_mode="persistent",
        start_mode="scalar",
        endpoint_mode="binary",
        birth_threshold=0.5,
        alive_threshold=0.5,
        end_threshold=0.5,
        refractory_steps=2,
    ).train()
    head_state = head.initial_state(torch.device("cpu"), torch.float32, "parity")
    audit_state = Q2LifecycleState.initial(4)
    policy = Q2LifecyclePolicy("actual")

    for frame in range(12):
        feature = torch.randn((1, 8))
        outputs, stepped = head.step(feature, head_state, frame)
        emitted, decoded = head.decode_step(outputs, stepped, frame)
        replay = replay_lifecycle_step(audit_state, outputs, policy)
        assert replay.state_after.slot_status == tuple(decoded.slot_status.tolist())
        assert replay.state_after.refractory == tuple(decoded.refractory.tolist())
        assert replay.emitted_slots == tuple(row.slot_id for row in emitted)
        audit_state = replay.state_after
        head_state = decoded


@pytest.mark.parametrize(
    "exhausted,births,ends,canonical,status,candidates,expected",
    [
        (
            (3,),
            (3,),
            (),
            {1: 0, 2: 1},
            (SLOT_ACTIVE, SLOT_ACTIVE),
            (),
            TRUE_CANONICAL_CAPACITY,
        ),
        ((3,), (3,), (), {}, (SLOT_ACTIVE, SLOT_ACTIVE), (), FALSE_ACTIVE_OCCUPANCY),
        (
            (3,),
            (3,),
            (),
            {},
            (SLOT_REFRACTORY, SLOT_REFRACTORY),
            (),
            REFRACTORY_OCCUPANCY,
        ),
        (
            (3,),
            (3, 4),
            (1,),
            {1: 0},
            (SLOT_ACTIVE, SLOT_FREE),
            (1,),
            SAME_BIN_NON_REUSE,
        ),
        (
            (3, 4, 5),
            (3, 4, 5),
            (1,),
            {1: 0},
            (SLOT_ACTIVE, SLOT_ACTIVE, SLOT_REFRACTORY),
            (),
            MIXED_CAUSE,
        ),
    ],
)
def test_exhaustion_attribution_is_total_and_cause_specific(
    exhausted, births, ends, canonical, status, candidates, expected
):
    attribution = attribute_exhaustions(
        exhausted,
        births,
        ends,
        canonical,
        Q2LifecycleState(tuple(status), (0,) * len(status)),
        candidates,
    )
    assert attribution.attribution_closed
    assert len(attribution.cause_by_instance) == len(exhausted)
    assert set(cause for _, cause in attribution.cause_by_instance) == {expected}


def test_privileged_upper_bound_can_never_become_a_trainer_contract():
    policies = {policy.name: policy for policy in frozen_policy_grid(4)}
    privileged = policies["canonical_only_privileged"]
    assert privileged.privileged_canonical_availability
    assert privileged.eligible_shared_contract is False
    assert policies["actual"].eligible_shared_contract is True
    assert policies["capacity_k2"].eligible_shared_contract is False


def test_annotation_audit_distinguishes_same_bin_order_from_oracle_capacity():
    schedule = build_prefix_instance_schedule(
        ((1.0, 3.0), (3.0, 5.0)),
        (0, 1),
        decision_frames=(0, 2, 4, 6),
        previous_frame=-2,
    )
    summary, rows = audit_annotation_schedule(
        "same-bin", schedule, num_slots=1, refractory_steps=2
    )
    crossing = rows[2]
    assert crossing["same_bin_birth_end_count"] == 1
    assert crossing["minimum_oracle_free_k"] == 1
    assert crossing["current_order_required_k"] == 2
    assert summary["oracle_k4_structurally_sufficient"]


def test_fixed_and_rematch_share_canonical_capacity_lifecycle():
    schedule = build_prefix_instance_schedule(
        ((1.0, 5.0), (2.0, 6.0)),
        (0, 1),
        decision_frames=(0, 2, 4, 6, 8),
        previous_frame=-2,
    )
    fixed = PrefixTrajectorySupervisionState(4, SupervisionMode.FIXED)
    rematch = PrefixTrajectorySupervisionState(4, SupervisionMode.REMATCH)
    outputs = _outputs(
        birth=(1.0, 0.5, -0.5, -1.0),
        alive=(1.0, 1.0, 1.0, 1.0),
        end=(-1.0, -1.0, -1.0, -1.0),
    )
    for step in schedule:
        provider = build_assignment_cost_provider(
            outputs, step, feature_stride=2, memory_size=8
        )
        left = fixed.transition(step, provider, available_slots=(0, 1, 2, 3))
        right = rematch.transition(step, provider, available_slots=(0, 1, 2, 3))
        assert left.birth_assignments == right.birth_assignments
        assert left.canonical_bindings == right.canonical_bindings
        assert left.exhausted_instance_ids == right.exhausted_instance_ids
        assert fixed.instance_to_slot == rematch.instance_to_slot


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class _Scheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1

    def get_last_lr(self):
        return [0.1]


class _ChronologicalExhaustionToy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.pending = False
        self.rollbacks = 0
        self.commits = 0
        self.last_episode_audit = {}

    def reset_online_states(self):
        self.pending = False

    def has_pending_online_update(self):
        return self.pending

    def snapshot_online_update(self):
        return {"pending": self.pending}

    def restore_online_update(self, snapshot):
        self.pending = bool(snapshot["pending"])

    def commit_online_update(self):
        self.pending = False
        self.commits += 1

    def rollback_online_update(self):
        self.pending = False
        self.rollbacks += 1

    def forward(self, inputs, stream_control, return_loss=False):
        assert return_loss
        self.pending = True
        self.last_episode_audit = {"slot_exhaustion": 1}
        return {"cost": (self.weight - inputs).pow(2).mean()}


def test_chronological_engine_rolls_back_exhaustion_before_backward_or_step():
    model = _ChronologicalExhaustionToy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = _Scheduler()
    loader = [
        {
            "inputs": torch.tensor([1.0]),
            "stream_control": [
                {
                    "video_id": "video",
                    "is_video_start": True,
                    "is_video_end": True,
                }
            ],
        }
    ]

    with pytest.raises(
        RuntimeError, match="chronological scientific failure: slot exhaustion"
    ):
        train_one_epoch(
            loader,
            model,
            optimizer,
            scheduler,
            curr_epoch=0,
            logger=_Logger(),
            logging_interval=100,
        )

    assert model.rollbacks == 1
    assert model.commits == 0
    assert model.weight.item() == 0.0
    assert model.weight.grad is None
    assert scheduler.steps == 0
