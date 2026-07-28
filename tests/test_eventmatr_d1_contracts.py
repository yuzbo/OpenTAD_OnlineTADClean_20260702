"""Hard contracts and micro-experiments for EventMATR D1."""

from __future__ import annotations

import copy
import sys
import types
from types import SimpleNamespace

import pytest


try:
    import torch
except (ImportError, OSError) as exc:  # pragma: no cover - workstation DLL issue
    pytest.skip(f"PyTorch runtime unavailable: {exc}", allow_module_level=True)

try:  # pragma: no cover - lean CPU environments
    import typeguard  # noqa: F401
except ImportError:  # pragma: no cover
    module = types.ModuleType("typeguard")
    module.typechecked = lambda value: value
    sys.modules["typeguard"] = module

try:  # pragma: no cover - lean CPU environments
    from x_transformers.x_transformers import exists as _exists  # noqa: F401
except ImportError:  # pragma: no cover
    package = types.ModuleType("x_transformers")
    submodule = types.ModuleType("x_transformers.x_transformers")
    submodule.exists = lambda value: value is not None
    package.x_transformers = submodule
    sys.modules["x_transformers"] = package
    sys.modules["x_transformers.x_transformers"] = submodule

from criterion.criterion import CriterionMATR
from criterion.matcher import HungarianMatcher
from models.event_memory import DynamicEventMemory, temporal_viterbi_assignment
from models.models import MATR


NEG = -20.0
POS = 20.0


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        training=True,
        feat_dim=8,
        num_of_class=3,
        hidden_dim=8,
        ffn_dim=16,
        enc_layers=1,
        e_nheads=2,
        dec_layers=1,
        d_nheads=2,
        num_frame=4,
        num_queries=2,
        rgb=True,
        flow=False,
        dropout=0.0,
        drop_rate=0.0,
        pre_norm=False,
        activation="gelu",
        use_flag=True,
        flag_threshold=0.5,
        max_memory_len=2,
        memory_sampler="all",
        reduce=1,
        anti_len=2,
        use_empty_weight=False,
        eos_coef=1e-8,
        use_focal=True,
        model_variant="eventmatr",
        birth_mode="instant_transition",
        ownership_mode="sticky_owner",
        event_arm="b1o1",
        event_birth_logit_threshold=None,
        event_end_logit_threshold=None,
        event_resource_limit=0,
        event_lifecycle_version="d1_censored",
        event_teacher_forcing_ratio=1.0,
        event_identity_coef=1.0,
        event_d1_lane="th",
    )


def _criterion(args: SimpleNamespace | None = None) -> CriterionMATR:
    args = args or _args()
    return CriterionMATR(
        num_classes=args.num_of_class,
        matcher=HungarianMatcher(args),
        weight_dict={},
        losses=[],
        args=args,
    )


def _memory() -> DynamicEventMemory:
    return DynamicEventMemory(
        birth_mode="instant_transition",
        ownership_mode="sticky_owner",
        birth_logit_threshold=0.0,
        end_logit_threshold=0.0,
        min_duration_frames=1,
        emit_delay_frames=0,
        resource_limit=0,
        segment_size=1,
        enable_reacquisition=True,
        strict_causal_boundary=True,
    )


def _step(
    memory: DynamicEventMemory,
    frame: int,
    *,
    birth: float,
    end: float = NEG,
    start: float | None = None,
    is_eos: bool = False,
    true_duration: int | None = None,
    owner_runtime: dict | None = None,
):
    metadata = {}
    if true_duration is not None:
        metadata["true_durations"] = [true_duration]
    return memory.step(
        video_names=["video"],
        current_frames=torch.tensor([frame]),
        birth_logits=torch.tensor([[birth]]),
        alive_logits=torch.tensor([[POS]]),
        end_logits=torch.tensor([[end]]),
        end_offsets=torch.zeros((1, 1)),
        class_logits=torch.tensor([[[POS, NEG, NEG]]]),
        query_features=torch.tensor([[[1.0, 0.0]]]),
        candidate_start_frames=torch.tensor(
            [[frame if start is None else start]], dtype=torch.float32
        ),
        is_real_prefix=[True],
        is_eos=[is_eos],
        **metadata,
        **(owner_runtime or {}),
    )


def _empty_ragged(outputs: dict, hidden_dim: int = 4) -> None:
    device = outputs["event_state_logits"].device
    dtype = outputs["event_state_logits"].dtype
    outputs.update(
        {
            "event_ragged_state_logits": torch.zeros(
                (0, 4), device=device, dtype=dtype
            ),
            "event_ragged_state_targets": torch.zeros(
                (0,), device=device, dtype=torch.long
            ),
            "event_ragged_end_offsets": torch.zeros(
                (0,), device=device, dtype=dtype
            ),
            "event_ragged_end_targets": torch.zeros(
                (0,), device=device, dtype=dtype
            ),
            "event_ragged_class_logits": torch.zeros(
                (0, 3), device=device, dtype=dtype
            ),
            "event_ragged_class_targets": torch.zeros(
                (0,), device=device, dtype=torch.long
            ),
            "event_ragged_embeddings": torch.zeros(
                (0, hidden_dim), device=device, dtype=dtype
            ),
            "event_ragged_group_keys": [],
            "event_ragged_sources": [],
        }
    )


def test_temporal_viterbi_assignment_is_stable_across_query_ambiguity() -> None:
    state = torch.zeros((3, 2, 4))
    classes = torch.zeros((3, 2, 3))
    features = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.9, 0.1], [0.1, 0.9]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    state[:, 0, 1] = 3.0
    state[:, 1, 1] = 2.8
    state[1, 1, 1] = 3.1
    classes[:, :, 0] = 2.0

    assert temporal_viterbi_assignment(state, classes, features, 0) == [0, 0, 0]


@pytest.mark.parametrize(
    ("lane", "uses_identity", "uses_hazard"),
    [
        ("r", False, False),
        ("t", True, False),
        ("h", False, True),
        ("th", True, True),
    ],
)
def test_registered_d1_factor_switches_are_isolated(
    lane: str, uses_identity: bool, uses_hazard: bool
) -> None:
    args = _args()
    args.event_d1_lane = lane
    model = MATR(args)
    criterion = _criterion(args)
    assert model.event_d1_use_identity is uses_identity
    assert model.event_d1_use_hazard is uses_hazard
    assert model.event_memory.enable_reacquisition is uses_identity
    assert criterion.event_d1_use_identity is uses_identity
    assert criterion.event_d1_use_hazard is uses_hazard
    assert ("loss_event_identity" in criterion.weight_dict) is uses_identity


def test_d1_runtime_rejects_duration_but_accepts_observed_eos() -> None:
    memory = _memory()
    with pytest.raises(RuntimeError, match="forbids true_duration"):
        _step(memory, 3, birth=POS, start=1, true_duration=10)

    _step(memory, 3, birth=POS, start=1)
    eos = _step(memory, 4, birth=POS, is_eos=True)
    assert eos["eos_observed"].tolist() == [1]
    assert len(memory.records("video")) == 1
    assert memory.ledger("video") == ()


def test_negative_start_is_clamped_and_reacquisition_keeps_identity() -> None:
    memory = _memory()
    _step(memory, 3, birth=POS, start=-7)
    first = memory.records("video")[0]
    assert first.start_frame == 0.0

    owner_embeddings, owner_padding, owner_ids = memory.owner_batch(
        ["video"],
        device=torch.device("cpu"),
        dtype=torch.float32,
        embedding_dim=2,
    )
    background = {
        "owner_state_logits": torch.tensor([[[POS, NEG, NEG, NEG]]]),
        "owner_end_offsets": torch.zeros((1, 1)),
        "owner_updated_embeddings": owner_embeddings,
        "owner_valid_mask": ~owner_padding,
        "owner_record_ids": owner_ids,
    }
    _step(memory, 4, birth=NEG, owner_runtime=background)
    assert memory.records("video") == ()
    _step(memory, 5, birth=POS, start=5)
    reacquired = memory.records("video")[0]
    assert reacquired.event_id == first.event_id
    assert reacquired.reacquisition_count == 1


def test_fresh_rematch_refreshes_an_existing_owner_without_runtime_error() -> None:
    memory = DynamicEventMemory(
        birth_mode="instant_transition",
        ownership_mode="fresh_rematch",
        birth_logit_threshold=0.0,
        end_logit_threshold=0.0,
        min_duration_frames=1,
        emit_delay_frames=0,
        resource_limit=0,
        segment_size=1,
        enable_reacquisition=False,
        strict_causal_boundary=True,
    )
    _step(memory, 2, birth=POS, start=1)

    memory.rematch_active_owners(
        "video",
        query_features=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        class_logits=torch.tensor([[NEG, POS, NEG], [POS, NEG, NEG]]),
    )

    record = memory.records("video")[0]
    assert int(record.class_distribution.argmax().item()) == 0


def _birth_case(query_count: int):
    criterion = _criterion()
    state = torch.full((1, query_count, 4), -4.0)
    state[:, :, 0] = 4.0
    state[0, 0] = torch.tensor([-2.0, 2.0, -2.0, -2.0])
    birth = torch.full((1, query_count), -8.0, requires_grad=True)
    birth.data[0, 0] = 1.25
    classes = torch.full((1, query_count, 3), -4.0)
    classes[:, :, 2] = 4.0
    classes[0, 0] = torch.tensor([4.0, -4.0, -4.0])
    features = torch.eye(query_count).unsqueeze(0)
    outputs = {
        "event_state_logits": state,
        "event_birth_logits": birth,
        "event_candidate_start_frames": torch.zeros((1, query_count)),
        "event_query_features": features,
        "pred_cls": classes,
    }
    _empty_ragged(outputs, hidden_dim=query_count)
    event_targets = torch.zeros((1, query_count, 8))
    event_targets[0, 0] = torch.tensor([0, 0, 0, 5, 0, 1, 0, 0])
    losses = criterion.loss_event(
        outputs,
        {
            "event_targets": event_targets,
            "event_valid_mask": torch.tensor(
                [[True] + [False] * (query_count - 1)]
            ),
        },
        {
            "video_name": ["v"],
            "current_frame": torch.tensor([0]),
            "is_real_prefix": torch.tensor([True]),
        },
    )
    return losses, birth


def test_event_normalized_birth_hazard_is_not_diluted_by_background_queries() -> None:
    two, birth_two = _birth_case(2)
    twenty, birth_twenty = _birth_case(20)
    assert two["loss_event_birth"].item() == pytest.approx(
        twenty["loss_event_birth"].item()
    )

    two["loss_event_birth"].backward()
    twenty["loss_event_birth"].backward()
    assert birth_two.grad[0, 0] == pytest.approx(birth_twenty.grad[0, 0].item())
    assert birth_twenty.grad[0, 1:].abs().sum() == 0


def test_right_censored_end_hazard_has_survival_gradient() -> None:
    criterion = _criterion()
    dense_state = torch.zeros((1, 2, 4))
    outputs = {
        "event_state_logits": dense_state,
        "event_birth_logits": torch.zeros((1, 2)),
        "event_candidate_start_frames": torch.zeros((1, 2)),
        "event_query_features": torch.zeros((1, 2, 4)),
        "pred_cls": torch.zeros((1, 2, 3)),
    }
    ragged = torch.tensor(
        [[0.0, 0.0, 2.0, 0.5], [0.0, 0.0, 2.0, -0.5]],
        requires_grad=True,
    )
    outputs.update(
        {
            "event_ragged_state_logits": ragged,
            "event_ragged_state_targets": torch.tensor([2, 2]),
            "event_ragged_end_offsets": torch.zeros(2, requires_grad=True),
            "event_ragged_end_targets": torch.zeros(2),
            "event_ragged_class_logits": torch.zeros((2, 3), requires_grad=True),
            "event_ragged_class_targets": torch.tensor([0, 0]),
            "event_ragged_embeddings": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]],
                requires_grad=True,
            ),
            "event_ragged_group_keys": [
                ("v", 0, 7, 1.0),
                ("v", 0, 7, 2.0),
            ],
            "event_ragged_sources": ["oracle", "oracle"],
        }
    )
    targets = {
        "event_targets": torch.zeros((1, 2, 8)),
        "event_valid_mask": torch.zeros((1, 2), dtype=torch.bool),
    }
    losses = criterion.loss_event(
        outputs,
        targets,
        {
            "video_name": ["v"],
            "current_frame": torch.tensor([2]),
            "is_real_prefix": torch.tensor([True]),
        },
    )
    losses["loss_event_end"].backward()
    assert losses["loss_event_end"] > 0
    assert ragged.grad[:, 3].min() > 0


def test_same_class_overlap_contributes_identity_repulsion() -> None:
    criterion = _criterion()
    outputs = {
        "event_state_logits": torch.zeros((1, 2, 4)),
        "event_birth_logits": torch.zeros((1, 2)),
        "event_candidate_start_frames": torch.zeros((1, 2)),
        "event_query_features": torch.zeros((1, 2, 2)),
        "pred_cls": torch.zeros((1, 2, 3)),
        "event_ragged_state_logits": torch.zeros((2, 4), requires_grad=True),
        "event_ragged_state_targets": torch.tensor([2, 2]),
        "event_ragged_end_offsets": torch.zeros(2, requires_grad=True),
        "event_ragged_end_targets": torch.zeros(2),
        "event_ragged_class_logits": torch.zeros((2, 3), requires_grad=True),
        "event_ragged_class_targets": torch.tensor([0, 0]),
        "event_ragged_embeddings": torch.tensor(
            [[1.0, 0.0], [1.0, 0.0]], requires_grad=True
        ),
        "event_ragged_group_keys": [
            ("v", 0, 10, 5.0),
            ("v", 1, 11, 5.0),
        ],
        "event_ragged_sources": ["oracle", "oracle"],
    }
    losses = criterion.loss_event(
        outputs,
        {
            "event_targets": torch.zeros((1, 2, 8)),
            "event_valid_mask": torch.zeros((1, 2), dtype=torch.bool),
        },
        {
            "video_name": ["v"],
            "current_frame": torch.tensor([5]),
            "is_real_prefix": torch.tensor([True]),
        },
    )
    assert losses["loss_event_identity"] > 0


def test_full_d1_model_uses_one_differentiable_ragged_unroll() -> None:
    torch.manual_seed(11)
    args = _args()
    model = MATR(args).train()
    model.memory_queue = model.memory_queue_index = None
    event_targets = torch.zeros((3, 2, 8))
    event_targets[0, 0] = torch.tensor([0, 0, 2, 8, 2, 1, 0, 0])
    event_targets[1, 0] = torch.tensor([0, 0, 2, 8, 2, 0, 1, 0])
    event_targets[2, 0] = torch.tensor([0, 0, 2, 8, 2, 0, 1, 0])
    valid = torch.tensor([[True, False]] * 3)
    model_input = {
        "inputs": torch.randn((3, 4, 8)),
        "infos": {
            "st": torch.tensor([0, 1, 2]),
            "ed": torch.tensor([4, 5, 6]),
            "video_name": ["v", "v", "v"],
            "current_frame": torch.tensor([4, 5, 6]),
            "segment_flag": torch.zeros(3, dtype=torch.long),
            "is_real_prefix": torch.ones(3, dtype=torch.bool),
            "is_eos": torch.tensor([False, False, True]),
        },
        "event_targets": event_targets,
        "event_valid_mask": valid,
    }
    outputs = model(copy.deepcopy(model_input), torch.device("cpu"))
    assert outputs["event_ragged_state_logits"].size(0) >= 1
    criterion = _criterion(args)
    losses = criterion.loss_event(
        outputs,
        {"event_targets": event_targets, "event_valid_mask": valid},
        model_input["infos"],
    )
    total = sum(
        value
        for key, value in losses.items()
        if key.startswith("loss_event")
    )
    total.backward()
    assert model.event_owner_decoder.state.weight.grad is not None
    assert model.event_transition_head.state.weight.grad is not None


def test_d1_model_boundary_rejects_full_video_metadata() -> None:
    args = _args()
    model = MATR(args).train()
    payload = {
        "inputs": torch.randn((1, 4, 8)),
        "infos": {
            "st": torch.tensor([0]),
            "ed": torch.tensor([4]),
            "video_name": ["v"],
            "current_frame": torch.tensor([3]),
            "segment_flag": torch.tensor([0]),
            "is_real_prefix": torch.tensor([True]),
            "is_eos": torch.tensor([False]),
            "true_duration": torch.tensor([100]),
        },
        "event_targets": torch.zeros((1, 2, 8)),
        "event_valid_mask": torch.zeros((1, 2), dtype=torch.bool),
    }
    with pytest.raises(RuntimeError, match="future/full-video metadata"):
        model(payload, torch.device("cpu"))
