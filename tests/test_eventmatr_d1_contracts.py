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
from models.event_memory import (
    CausalTemporalHistory,
    DynamicEventMemory,
    causal_single_assignment,
    d1_owner_supervision,
    select_disjoint_teacher_query,
    temporal_viterbi_assignment,
)
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
        event_d13_variant="d12_control",
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


def _memory(enable_reacquisition: bool = True) -> DynamicEventMemory:
    return DynamicEventMemory(
        birth_mode="instant_transition",
        ownership_mode="sticky_owner",
        birth_logit_threshold=0.0,
        end_logit_threshold=0.0,
        min_duration_frames=1,
        emit_delay_frames=0,
        resource_limit=0,
        segment_size=1,
        enable_reacquisition=enable_reacquisition,
        strict_causal_boundary=True,
        owner_state_count=3,
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
    oracle_births: list[dict] | None = None,
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
        oracle_births=[oracle_births or []],
        **metadata,
        **(owner_runtime or {}),
    )


def _empty_ragged(outputs: dict, hidden_dim: int = 4) -> None:
    device = outputs["event_state_logits"].device
    dtype = outputs["event_state_logits"].dtype
    outputs.update(
        {
            "event_ragged_state_logits": torch.zeros(
                (0, 3), device=device, dtype=dtype
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
            "event_birth_risk_groups": [],
            "event_association_rows": [],
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


def test_d12_temporal_assignment_uses_independent_birth_under_bg_dominance() -> None:
    state = torch.full((3, 2, 4), -8.0)
    state[:, :, 0] = 8.0
    birth = torch.tensor([[3.0, -3.0], [3.0, 3.2], [3.0, -3.0]])
    classes = torch.zeros((3, 2, 3))
    classes[:, :, 0] = 2.0
    features = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.9, 0.1], [0.1, 0.9]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )

    assert bool((state.argmax(dim=-1) == 0).all().item())
    assert temporal_viterbi_assignment(
        state,
        classes,
        features,
        0,
        birth_logits=birth,
    ) == [0, 0, 0]


def test_d12_runtime_birth_is_independent_of_four_state_bg_winner() -> None:
    args = _args()
    model = MATR(args)
    head = model.event_transition_head
    assert head.independent_birth is True
    assert head.birth is not None
    with torch.no_grad():
        head.state.weight.zero_()
        head.state.bias.copy_(torch.tensor([8.0, -8.0, -8.0, -8.0]))
        head.birth.weight.zero_()
        head.birth.bias.fill_(2.0)
    state, birth, _, _, _ = head(
        torch.zeros((1, 2, args.hidden_dim)),
        torch.zeros((1, 2, args.hidden_dim)),
    )

    assert state.argmax(dim=-1).tolist() == [[0, 0]]
    assert bool((birth > 0.0).all().item())
    assert model.event_memory.preview_birth_queries(
        "v",
        candidate_state_logits=state[0],
        birth_logits=birth[0],
    ) == (0, 1)


def test_causal_single_assignment_ignores_teacher_query_identity() -> None:
    result = causal_single_assignment(
        predicted_query_indices=[1],
        candidate_start_frames=torch.tensor([10.0, 10.0]),
        class_logits=torch.tensor(
            [[NEG, POS, NEG], [POS, NEG, NEG]], dtype=torch.float32
        ),
        query_features=torch.tensor(
            [[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32
        ),
        target_specs=[
            {
                "target_event_id": 7,
                "class_id": 0,
                "start_frame": 10.0,
                # The causal teacher path may use another query index; only
                # its semantic anchor, not that integer index, is relevant.
                "anchor_feature": torch.tensor([1.0, 0.0]),
            }
        ],
        max_start_distance=4.0,
    )
    assert dict(result.assignments) == {1: 7}
    assert result.unmatched_queries == ()
    assert result.unmatched_targets == ()
    assert result.predicted_query_count == 1
    assert result.target_count == 1
    assert result.pair_count == 1
    assert result.class_mismatch_pair_count == 0
    assert result.start_distance_reject_pair_count == 0
    assert result.admissible_pair_count == 1


def test_causal_single_assignment_refuses_exactly_ambiguous_identity() -> None:
    result = causal_single_assignment(
        predicted_query_indices=[0, 1],
        candidate_start_frames=torch.tensor([10.0, 10.0]),
        class_logits=torch.tensor(
            [[POS, NEG, NEG], [POS, NEG, NEG]], dtype=torch.float32
        ),
        query_features=torch.tensor(
            [[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32
        ),
        target_specs=[
            {
                "target_event_id": 3,
                "class_id": 0,
                "start_frame": 10.0,
                "anchor_feature": torch.tensor([1.0, 0.0]),
            }
        ],
        max_start_distance=4.0,
    )
    assert dict(result.assignments) == {}
    assert result.ambiguous_queries == (0, 1)
    assert result.ambiguous_targets == (3,)
    assert result.unmatched_queries == (0, 1)
    assert result.unmatched_targets == (3,)
    assert result.pair_count == 2
    assert result.admissible_pair_count == 2


def test_causal_single_assignment_accounts_for_each_rejection_barrier() -> None:
    result = causal_single_assignment(
        predicted_query_indices=[0, 1],
        candidate_start_frames=torch.tensor([10.0, 30.0]),
        class_logits=torch.tensor(
            [[NEG, POS, NEG], [POS, NEG, NEG]], dtype=torch.float32
        ),
        query_features=torch.tensor(
            [[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32
        ),
        target_specs=[
            {
                "target_event_id": 7,
                "class_id": 0,
                "start_frame": 10.0,
                "anchor_feature": torch.tensor([1.0, 0.0]),
            }
        ],
        max_start_distance=4.0,
    )
    assert dict(result.assignments) == {}
    assert result.pair_count == 2
    assert result.class_mismatch_pair_count == 1
    assert result.start_distance_reject_pair_count == 1
    assert result.admissible_pair_count == 0
    assert result.class_argmax_reject_pair_count == 1
    assert result.admissible_class_argmax_mismatch_pair_count == 0


def test_d13_soft_assignment_uses_wrong_argmax_as_evidence_not_a_gate() -> None:
    result = causal_single_assignment(
        predicted_query_indices=[0],
        candidate_start_frames=torch.tensor([10.0]),
        class_logits=torch.tensor([[2.0, 3.0, -4.0]], dtype=torch.float32),
        query_features=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        target_specs=[
            {
                "target_event_id": 7,
                "class_id": 0,
                "start_frame": 10.0,
                "anchor_feature": torch.tensor([1.0, 0.0]),
            }
        ],
        max_start_distance=4.0,
        association_contract="soft_target_class_log_probability_v1",
    )
    assert dict(result.assignments) == {0: 7}
    assert result.class_mismatch_pair_count == 1
    assert result.class_argmax_reject_pair_count == 0
    assert result.start_distance_reject_pair_count == 0
    assert result.admissible_pair_count == 1
    assert result.admissible_class_argmax_mismatch_pair_count == 1


def test_causal_single_assignment_rejects_nonfinite_or_invalid_semantics() -> None:
    common = {
        "predicted_query_indices": [0],
        "candidate_start_frames": torch.tensor([10.0]),
        "class_logits": torch.tensor([[POS, NEG, NEG]], dtype=torch.float32),
        "query_features": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        "max_start_distance": 4.0,
    }
    target = {
        "target_event_id": 7,
        "class_id": 0,
        "start_frame": 10.0,
        "anchor_feature": torch.tensor([1.0, 0.0]),
    }
    with pytest.raises(ValueError, match="finite"):
        causal_single_assignment(
            **{**common, "candidate_start_frames": torch.tensor([float("nan")])},
            target_specs=[target],
        )
    with pytest.raises(ValueError, match="foreground range"):
        causal_single_assignment(
            **common,
            target_specs=[{**target, "class_id": 9}],
        )
    with pytest.raises(ValueError, match="foreground range"):
        causal_single_assignment(
            **{**common, "predicted_query_indices": []},
            target_specs=[{**target, "class_id": 9}],
        )
    with pytest.raises(ValueError, match="foreground and background"):
        causal_single_assignment(
            **{
                **common,
                "class_logits": torch.tensor([[POS]], dtype=torch.float32),
            },
            target_specs=[],
        )


def test_causal_temporal_history_is_independent_of_physical_batch_splits() -> None:
    whole = CausalTemporalHistory(window_size=4)
    split = CausalTemporalHistory(window_size=4)

    def append(history, frame):
        value = torch.tensor([[float(frame)]], requires_grad=True)
        history.append(
            video_name="v",
            frame=float(frame),
            state_logits=value.repeat(1, 4),
            birth_logits=value.reshape(1),
            class_logits=value.repeat(1, 3),
            query_features=value,
        )

    for frame in range(3):
        append(whole, frame)
    for frame in range(3, 6):
        append(whole, frame)

    for frame in range(2):
        append(split, frame)
    split.detach()
    for frame in range(2, 6):
        append(split, frame)

    assert whole.frames("v") == split.frames("v") == (2.0, 3.0, 4.0, 5.0)
    whole_window = whole.window("v", lower=2.0, upper=5.0)
    split_window = split.window("v", lower=2.0, upper=5.0)
    assert [entry["frame"] for entry in whole_window] == [
        entry["frame"] for entry in split_window
    ]
    assert torch.equal(
        torch.stack([entry["state_logits"] for entry in whole_window]),
        torch.stack([entry["state_logits"] for entry in split_window]),
    )

    frozen_frames = split.frames("v")
    split.append(
        video_name="v",
        frame=999.0,
        state_logits=torch.zeros((1, 4)),
        birth_logits=torch.zeros((1,)),
        class_logits=torch.zeros((1, 3)),
        query_features=torch.zeros((1, 1)),
        is_real_prefix=False,
    )
    assert split.frames("v") == frozen_frames


def _synthetic_d1_unroll(split_before_frame: int | None):
    history = CausalTemporalHistory(window_size=4)
    memory = DynamicEventMemory(
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
        owner_state_count=3,
    )
    assignments = []
    risk_members = []
    lifecycle_counts = []
    for frame in range(6):
        if split_before_frame == frame:
            history.detach()
            memory.detach_graph()
        state = torch.tensor(
            [
                [POS, NEG, NEG, NEG],
                [NEG, POS, NEG, NEG]
                if frame == 2
                else [POS, NEG, NEG, NEG],
            ],
            dtype=torch.float32,
        )
        birth = torch.tensor([NEG, POS if frame == 2 else NEG])
        classes = torch.tensor(
            [[NEG, POS, NEG], [POS, NEG, NEG]], dtype=torch.float32
        )
        features = torch.tensor(
            [[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32
        )
        history.append(
            video_name="v",
            frame=float(frame),
            state_logits=state,
            birth_logits=birth,
            class_logits=classes,
            query_features=features,
        )
        oracle_births = []
        if frame == 2:
            window = history.window("v", lower=0.0, upper=2.0)
            risk_members.append(tuple(entry["frame"] for entry in window))
            predicted = memory.preview_birth_queries(
                "v", candidate_state_logits=state, birth_logits=birth
            )
            association = causal_single_assignment(
                predicted_query_indices=predicted,
                candidate_start_frames=torch.tensor([2.0, 2.0]),
                class_logits=classes,
                query_features=features,
                target_specs=[
                    {
                        "target_event_id": 9,
                        "class_id": 0,
                        "start_frame": 2.0,
                        "anchor_feature": features[1],
                    }
                ],
                max_start_distance=4.0,
            )
            assignments.append(dict(association.assignments))
            oracle_births = [
                {
                    "query_index": query_index,
                    "target_event_id": target_event_id,
                    "start_frame": 2.0,
                    "source": "predicted_associated",
                    "association_status": "associated",
                    "force_create": False,
                    "merge_predicted": True,
                }
                for query_index, target_event_id in association.assignments.items()
            ]

        owner_runtime = {}
        owner_embeddings, owner_padding, owner_ids = memory.owner_batch(
            ["v"],
            device=torch.device("cpu"),
            dtype=torch.float32,
            embedding_dim=2,
        )
        if owner_embeddings.size(1):
            owner_state = (
                torch.tensor([[[NEG, NEG, POS]]])
                if frame == 4
                else torch.tensor([[[NEG, POS, NEG]]])
            )
            owner_runtime = {
                "owner_state_logits": owner_state,
                "owner_end_offsets": torch.zeros((1, 1)),
                "owner_updated_embeddings": owner_embeddings,
                "owner_valid_mask": ~owner_padding,
                "owner_record_ids": owner_ids,
            }
        result = memory.step(
            video_names=["v"],
            current_frames=torch.tensor([frame]),
            birth_logits=birth.unsqueeze(0),
            alive_logits=torch.zeros((1, 2)),
            end_logits=torch.zeros((1, 2)),
            end_offsets=torch.zeros((1, 2)),
            class_logits=classes.unsqueeze(0),
            query_features=features.unsqueeze(0),
            candidate_start_frames=torch.tensor([[frame, 2.0]]),
            candidate_state_logits=state.unsqueeze(0),
            is_real_prefix=[True],
            is_eos=[frame == 5],
            oracle_births=[oracle_births],
            **owner_runtime,
        )
        lifecycle_counts.append(
            (
                int(result["birth_count"].item()),
                int(result["cancellation_count"].item()),
                int(result["end_count"].item()),
                int(result["emit_count"].item()),
            )
        )
    return assignments, risk_members, lifecycle_counts, memory.ledger("v")


def test_d1_unroll_is_invariant_to_a_physical_batch_boundary() -> None:
    whole = _synthetic_d1_unroll(split_before_frame=None)
    split = _synthetic_d1_unroll(split_before_frame=2)
    assert split == whole
    assert whole[0] == [{1: 9}]
    assert whole[1] == [(0.0, 1.0, 2.0)]
    assert len(whole[3]) == 1


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
    assert model.event_memory.owner_state_count == 3
    assert model.event_owner_decoder.num_states == 3
    assert criterion.event_d1_use_identity is uses_identity
    assert criterion.event_d1_use_hazard is uses_hazard
    assert ("loss_event_identity" in criterion.weight_dict) is uses_identity


def test_d1_owner_supervision_has_explicit_ternary_semantics() -> None:
    assert d1_owner_supervision(
        None, current_frame=5, segment_size=4
    ) == (0, -100, 0.0)
    birth = torch.tensor([7, 1, 3, float("nan"), 3, 1, 0, 0])
    alive = torch.tensor([7, 1, 3, float("nan"), 3, 0, 1, 0])
    end = torch.tensor([7, 1, 3, 8, 3, 0, 0, 1])
    assert d1_owner_supervision(
        birth, current_frame=3, segment_size=4
    ) == (1, 1, 0.0)
    assert d1_owner_supervision(
        alive, current_frame=5, segment_size=4
    ) == (1, 1, 0.0)
    assert d1_owner_supervision(
        end, current_frame=9, segment_size=4
    ) == (2, 1, 0.25)


def test_teacher_query_selection_is_disjoint_and_deterministic() -> None:
    scores = torch.tensor([9.0, 4.0, 4.0])
    query, reassigned = select_disjoint_teacher_query(
        scores, preferred_query=0, occupied_queries=[0]
    )
    assert (query, reassigned) == (1, True)
    query, reassigned = select_disjoint_teacher_query(
        scores, preferred_query=0, occupied_queries=[0, 1, 2]
    )
    assert query is None
    assert reassigned is False


def test_d1_runtime_rejects_duration_but_accepts_observed_eos() -> None:
    memory = _memory()
    with pytest.raises(RuntimeError, match="forbids true_duration"):
        _step(memory, 3, birth=POS, start=1, true_duration=10)

    _step(memory, 3, birth=POS, start=1)
    eos = _step(memory, 4, birth=POS, is_eos=True)
    assert eos["eos_observed"].tolist() == [1]
    assert len(memory.records("video")) == 1
    assert memory.ledger("video") == ()


def test_d1_runtime_rejects_duration_without_identity_recovery() -> None:
    memory = _memory(enable_reacquisition=False)
    with pytest.raises(RuntimeError, match="forbids true_duration"):
        _step(memory, 3, birth=POS, start=1, true_duration=10)


def test_teacher_and_predicted_births_cannot_share_one_query() -> None:
    memory = _memory()
    teacher = {
        "query_index": 0,
        "target_event_id": 7,
        "start_frame": 1.0,
        "source": "teacher_birth",
        "association_status": "associated",
        "force_create": True,
        "merge_predicted": False,
    }
    with pytest.raises(RuntimeError, match="same query"):
        _step(memory, 3, birth=POS, start=1, oracle_births=[teacher])


def test_d1_runtime_continue_and_end_states_close_one_positive_interval() -> None:
    memory = _memory()
    _step(memory, 3, birth=POS, start=1)
    owner_embeddings, owner_padding, owner_ids = memory.owner_batch(
        ["video"],
        device=torch.device("cpu"),
        dtype=torch.float32,
        embedding_dim=2,
    )
    common = {
        "owner_end_offsets": torch.zeros((1, 1)),
        "owner_updated_embeddings": owner_embeddings,
        "owner_valid_mask": ~owner_padding,
        "owner_record_ids": owner_ids,
    }
    continued = _step(
        memory,
        4,
        birth=NEG,
        owner_runtime={
            **common,
            "owner_state_logits": torch.tensor([[[NEG, POS, NEG]]]),
        },
    )
    assert continued["end_count"].tolist() == [0]
    assert len(memory.records("video")) == 1

    ended = _step(
        memory,
        5,
        birth=NEG,
        owner_runtime={
            **common,
            "owner_state_logits": torch.tensor([[[NEG, NEG, POS]]]),
        },
    )
    assert ended["end_count"].tolist() == [1]
    ledger = memory.ledger("video")
    assert len(ledger) == 1
    assert ledger[0]["end_frame"] > ledger[0]["start_frame"]


def test_negative_start_is_clamped_and_targetless_birth_does_not_reuse_identity() -> None:
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
        "owner_state_logits": torch.tensor([[[POS, NEG, NEG]]]),
        "owner_end_offsets": torch.zeros((1, 1)),
        "owner_updated_embeddings": owner_embeddings,
        "owner_valid_mask": ~owner_padding,
        "owner_record_ids": owner_ids,
    }
    _step(memory, 4, birth=NEG, owner_runtime=background)
    assert memory.records("video") == ()
    _step(memory, 5, birth=POS, start=5)
    reborn = memory.records("video")[0]
    assert reborn.event_id == first.event_id + 1
    assert reborn.reacquisition_count == 0
    assert reborn.source == "predicted_unmatched"


def test_known_target_recovery_keeps_identity_and_is_labelled_as_error_recovery() -> None:
    memory = _memory()
    association = {
        "query_index": 0,
        "target_event_id": 7,
        "start_frame": 1.0,
        "source": "predicted_associated",
        "association_status": "associated",
        "force_create": False,
        "merge_predicted": True,
    }
    _step(memory, 3, birth=POS, start=1, oracle_births=[association])
    first = memory.records("video")[0]
    assert first.target_event_id == 7

    owner_embeddings, owner_padding, owner_ids = memory.owner_batch(
        ["video"],
        device=torch.device("cpu"),
        dtype=torch.float32,
        embedding_dim=2,
    )
    cancel = {
        "owner_state_logits": torch.tensor([[[POS, NEG, NEG]]]),
        "owner_end_offsets": torch.zeros((1, 1)),
        "owner_updated_embeddings": owner_embeddings,
        "owner_valid_mask": ~owner_padding,
        "owner_record_ids": owner_ids,
    }
    _step(memory, 4, birth=NEG, owner_runtime=cancel)
    _step(memory, 5, birth=POS, start=1, oracle_births=[association])
    recovered = memory.records("video")[0]
    assert recovered.event_id == first.event_id
    assert recovered.target_event_id == 7
    assert recovered.reacquisition_count == 1
    assert recovered.source == "associated_error_recovery"
    assert recovered.last_reacquisition_mode == "exact_target_id"


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


def _birth_case(
    query_count: int,
    *,
    variant: str = "d12_control",
    positive_count: int = 1,
):
    args = _args()
    args.event_d13_variant = variant
    criterion = _criterion(args)
    state = torch.full((1, query_count, 4), -4.0)
    state[:, :, 0] = 4.0
    birth = torch.full((1, query_count), -8.0, requires_grad=True)
    classes = torch.full((1, query_count, 3), -4.0)
    classes[:, :, 2] = 4.0
    for query_index in range(positive_count):
        state[0, query_index] = torch.tensor([-2.0, 2.0, -2.0, -2.0])
        birth.data[0, query_index] = 1.25
        classes[0, query_index] = torch.tensor([4.0, -4.0, -4.0])
    features = torch.eye(query_count).unsqueeze(0)
    outputs = {
        "event_state_logits": state,
        "event_birth_logits": birth,
        "event_candidate_start_frames": torch.zeros((1, query_count)),
        "event_query_features": features,
        "pred_cls": classes,
    }
    _empty_ragged(outputs, hidden_dim=query_count)
    outputs["event_birth_risk_groups"] = [
        {
            "batch_index": 0,
            "video_name": "v",
            "target_event_id": query_index,
            "class_id": 0,
            "start_frame": 0.0,
            "frames": torch.tensor([0.0]),
            "selected_logits": birth[0, query_index : query_index + 1],
            "terminal_query": query_index,
        }
        for query_index in range(positive_count)
    ]
    event_targets = torch.zeros((1, query_count, 8))
    for query_index in range(positive_count):
        event_targets[0, query_index] = torch.tensor(
            [query_index, 0, 0, 5, 0, 1, 0, 0]
        )
    losses = criterion.loss_event(
        outputs,
        {
            "event_targets": event_targets,
            "event_valid_mask": torch.tensor(
                [
                    [True] * positive_count
                    + [False] * (query_count - positive_count)
                ]
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
    assert birth_twenty.grad[0, 1:].abs().sum() > 0
    assert birth_two.grad[0, 1:].sum().item() == pytest.approx(
        birth_twenty.grad[0, 1:].sum().item()
    )


def test_d13_event_matched_birth_selects_exactly_one_negative_per_event() -> None:
    losses, birth = _birth_case(
        6,
        variant="event_matched_birth_only",
        positive_count=2,
    )
    assert losses["event_birth_positive_count"] == 2
    assert losses["event_birth_selected_negative_count"] == 2
    assert losses["event_birth_negative_candidate_count"] == 4
    assert losses["event_birth_positive_batch_count"] == 1
    assert losses["event_birth_zero_positive_batch_count"] == 0
    losses["loss_event_birth"].backward()
    assert int((birth.grad != 0).sum().item()) == 4


def test_d13_zero_birth_batch_has_no_birth_loss_or_birth_head_gradient() -> None:
    args = _args()
    args.event_d13_variant = "event_matched_birth_only"
    criterion = _criterion(args)
    state = torch.zeros((1, 4, 4), requires_grad=True)
    birth = torch.tensor(
        [[-1.0, 0.0, 1.0, 2.0]], requires_grad=True
    )
    outputs = {
        "event_state_logits": state,
        "event_birth_logits": birth,
        "event_candidate_start_frames": torch.zeros((1, 4)),
        "event_query_features": torch.zeros((1, 4, 4)),
        "pred_cls": torch.zeros((1, 4, 3)),
    }
    _empty_ragged(outputs, hidden_dim=4)
    losses = criterion.loss_event(
        outputs,
        {
            "event_targets": torch.zeros((1, 4, 8)),
            "event_valid_mask": torch.zeros((1, 4), dtype=torch.bool),
        },
        {
            "video_name": ["v"],
            "current_frame": torch.tensor([0]),
            "is_real_prefix": torch.tensor([True]),
        },
    )
    assert losses["loss_event_birth"].item() == 0.0
    assert losses["event_birth_selected_negative_count"] == 0
    assert losses["event_birth_negative_candidate_count"] == 4
    assert losses["event_birth_positive_batch_count"] == 0
    assert losses["event_birth_zero_positive_batch_count"] == 1
    losses["loss_event_birth"].backward()
    assert birth.grad is None


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
        [[0.0, 2.0, 0.5], [0.0, 2.0, -0.5]],
        requires_grad=True,
    )
    outputs.update(
        {
            "event_ragged_state_logits": ragged,
            "event_ragged_state_targets": torch.tensor([1, 1]),
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
            "event_ragged_sources": ["teacher_birth", "teacher_birth"],
            "event_birth_risk_groups": [],
            "event_association_rows": [{"source": "teacher_birth"}],
            "event_runtime_source_events": [
                {"source": "teacher_birth", "transition": "birth"},
                {"source": "teacher_birth", "transition": "end"},
            ],
            "event_birth_count": torch.tensor([2]),
            "event_end_count": torch.tensor([1]),
            "event_emit_count": torch.tensor([1]),
            "event_cancellation_count": torch.tensor([0]),
            "event_reacquisition_count": torch.tensor([0]),
            "event_runtime_capacity_exhaustions": torch.tensor([0]),
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
    assert ragged.grad[:, 2].min() > 0
    assert losses["event_source_teacher_birth_row_count"] == 2
    assert losses["event_source_teacher_birth_class_row_count"] == 2
    assert losses["event_source_teacher_birth_group_count"] == 1
    assert losses["event_source_teacher_birth_end_risk_group_count"] == 1
    assert losses["event_association_teacher_birth_count"] == 1
    assert losses["event_association_predicted_associated_count"] == 0
    assert losses["event_source_predicted_associated_row_count"] == 0
    assert losses["event_association_audit_prefix_count"] == 0
    assert losses["event_source_teacher_birth_birth_count"] == 1
    assert losses["event_source_teacher_birth_end_count"] == 1
    assert losses["event_runtime_birth_count"] == 2
    assert losses["event_runtime_end_count"] == 1
    assert losses["event_runtime_emit_count"] == 1
    assert losses["event_runtime_capacity_exhaustions"] == 0


def test_same_class_overlap_contributes_identity_repulsion() -> None:
    criterion = _criterion()
    outputs = {
        "event_state_logits": torch.zeros((1, 2, 4)),
        "event_birth_logits": torch.zeros((1, 2)),
        "event_candidate_start_frames": torch.zeros((1, 2)),
        "event_query_features": torch.zeros((1, 2, 2)),
        "pred_cls": torch.zeros((1, 2, 3)),
        "event_ragged_state_logits": torch.zeros((2, 3), requires_grad=True),
        "event_ragged_state_targets": torch.tensor([1, 1]),
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
        "event_ragged_sources": ["teacher_birth", "teacher_birth"],
        "event_birth_risk_groups": [],
        "event_association_rows": [],
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
    model_targets = event_targets.clone()
    model_targets[:, 0, 3] = float("nan")
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
        "event_targets": model_targets,
        "event_valid_mask": valid,
    }
    outputs = model(copy.deepcopy(model_input), torch.device("cpu"))
    assert outputs["event_ragged_state_logits"].size(0) >= 1
    assert len(outputs["event_association_audit_rows"]) == 3
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
    assert model.event_transition_head.birth.weight.grad is not None
    assert model.event_transition_head.birth.weight.grad.norm() > 0
    assert model.event_transition_head.fuse[0].weight.grad is not None


def test_d1_model_boundary_rejects_future_gt_endpoint() -> None:
    args = _args()
    model = MATR(args).train()
    targets = torch.zeros((1, 2, 8))
    targets[0, 0] = torch.tensor([0, 0, 2, 8, 2, 1, 0, 0])
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
        },
        "event_targets": targets,
        "event_valid_mask": torch.tensor([[True, False]]),
    }
    with pytest.raises(RuntimeError, match="future GT endpoint"):
        model(payload, torch.device("cpu"))


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


def test_d1_eval_boundary_rejects_ground_truth_segment_flag() -> None:
    args = _args()
    args.training = False
    model = MATR(args).eval()
    payload = {
        "inputs": torch.randn((1, 4, 8)),
        "infos": {
            "st": torch.tensor([0]),
            "ed": torch.tensor([4]),
            "video_name": ["v"],
            "current_frame": torch.tensor([3]),
            "segment_flag": torch.tensor([1]),
            "is_real_prefix": torch.tensor([True]),
            "is_eos": torch.tensor([False]),
        },
    }
    with pytest.raises(RuntimeError, match="segment_flag"):
        model(payload, torch.device("cpu"))
