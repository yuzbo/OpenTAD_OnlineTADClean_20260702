"""Deterministic scientific contracts for the MATR-derived event memory.

The tests drive the runtime with explicit logits and embeddings.  They do not
depend on a trained checkpoint and therefore separate lifecycle correctness
from localization quality.
"""

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

# The official repository imports two tiny helpers from optional packages.
# Unit tests only need their pure-Python semantics, so use local fakes when a
# clean CPU environment intentionally omits the full research dependency set.
try:  # pragma: no cover - exercised only in lean test environments
    import typeguard  # noqa: F401
except ImportError:  # pragma: no cover
    typeguard_module = types.ModuleType("typeguard")
    typeguard_module.typechecked = lambda value: value
    sys.modules["typeguard"] = typeguard_module

try:  # pragma: no cover - exercised only in lean test environments
    from x_transformers.x_transformers import exists as _exists  # noqa: F401
except ImportError:  # pragma: no cover
    package = types.ModuleType("x_transformers")
    submodule = types.ModuleType("x_transformers.x_transformers")
    submodule.exists = lambda value: value is not None
    package.x_transformers = submodule
    sys.modules["x_transformers"] = package
    sys.modules["x_transformers.x_transformers"] = submodule

from models.event_memory import DynamicEventMemory, OwnerEventDecoder
from models.models import MATR
from criterion.criterion import CriterionMATR
from criterion.matcher import HungarianMatcher


NEG = -20.0
POS = 20.0


def _memory(
    *,
    birth_mode: str = "instant_transition",
    ownership_mode: str = "sticky_owner",
    emit_delay_frames: int = 0,
    resource_limit: int = 0,
) -> DynamicEventMemory:
    return DynamicEventMemory(
        birth_mode=birth_mode,
        ownership_mode=ownership_mode,
        birth_logit_threshold=0.0,
        end_logit_threshold=0.0,
        min_duration_frames=1,
        emit_delay_frames=emit_delay_frames,
        resource_limit=resource_limit,
        segment_size=1,
    )


def _step(
    memory: DynamicEventMemory,
    frame: int,
    *,
    births: list[float],
    ends: list[float] | None = None,
    alive: list[float] | None = None,
    features: list[list[float]] | None = None,
    classes: list[list[float]] | None = None,
    starts: list[int] | None = None,
    video: str = "video_1",
    owner_runtime: dict[str, torch.Tensor] | None = None,
    candidate_states: list[list[float]] | None = None,
    is_real_prefix: bool | None = None,
    true_duration: int | None = None,
    is_eos: bool | None = None,
) -> dict[str, torch.Tensor]:
    queries = len(births)
    ends = ends if ends is not None else [NEG] * queries
    alive = alive if alive is not None else [POS] * queries
    features = features if features is not None else [
        [1.0 if i == j else 0.0 for j in range(queries)] for i in range(queries)
    ]
    classes = classes if classes is not None else [
        [POS, NEG, NEG] for _ in range(queries)
    ]
    starts = starts if starts is not None else [frame] * queries

    metadata = {}
    if candidate_states is not None:
        metadata["candidate_state_logits"] = torch.tensor(
            [candidate_states], dtype=torch.float32
        )
    if is_real_prefix is not None:
        metadata["is_real_prefix"] = [is_real_prefix]
    if true_duration is not None:
        metadata["true_durations"] = [true_duration]
    if is_eos is not None:
        metadata["is_eos"] = [is_eos]

    return memory.step(
        video_names=[video],
        current_frames=torch.tensor([frame], dtype=torch.long),
        birth_logits=torch.tensor([births], dtype=torch.float32),
        alive_logits=torch.tensor([alive], dtype=torch.float32),
        end_logits=torch.tensor([ends], dtype=torch.float32),
        end_offsets=torch.zeros((1, queries), dtype=torch.float32),
        class_logits=torch.tensor([classes], dtype=torch.float32),
        query_features=torch.tensor([features], dtype=torch.float32),
        candidate_start_frames=torch.tensor([starts], dtype=torch.long),
        **metadata,
        **(owner_runtime or {}),
    )


def _record_map(memory: DynamicEventMemory, video: str = "video_1") -> dict[int, object]:
    return {record.event_id: record for record in memory.records(video)}


def _native_args(*, explicit_variant: bool = True) -> SimpleNamespace:
    values = dict(
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
    )
    if explicit_variant:
        values.update(model_variant="native_matr", event_arm=None)
    return SimpleNamespace(**values)


def _official_input() -> dict:
    generator = torch.Generator().manual_seed(17)
    return {
        "inputs": torch.randn((1, 4, 8), generator=generator),
        "infos": {
            "st": torch.tensor([0]),
            "ed": torch.tensor([4]),
            "video_name": ["parity_video"],
            "current_frame": torch.tensor([4]),
            # Zero avoids exercising an unrelated native MATR axis bug in
            # memory admission; both compared paths still execute identically.
            "segment_flag": torch.tensor([0]),
        },
    }


def _event_args(*, arm: str = "b1o1", injected_thresholds: bool = False) -> SimpleNamespace:
    factors = {
        "b0o0": ("matr_delayed", "fresh_rematch"),
        "b1o0": ("instant_transition", "fresh_rematch"),
        "b0o1": ("matr_delayed", "sticky_owner"),
        "b1o1": ("instant_transition", "sticky_owner"),
    }
    birth_mode, ownership_mode = factors[arm]
    args = _native_args(explicit_variant=False)
    args.model_variant = "eventmatr"
    args.birth_mode = birth_mode
    args.ownership_mode = ownership_mode
    args.event_arm = arm
    args.event_birth_logit_threshold = 0.0 if injected_thresholds else None
    args.event_end_logit_threshold = 0.0 if injected_thresholds else None
    args.event_resource_limit = 0
    return args


def test_native_matr_lane_is_the_only_numerical_upstream_path() -> None:
    torch.manual_seed(3)
    native = MATR(_native_args(explicit_variant=False)).eval()
    explicit = MATR(_native_args(explicit_variant=True)).eval()
    explicit.load_state_dict(native.state_dict(), strict=True)

    assert tuple(native.state_dict()) == tuple(explicit.state_dict())
    native.memory_queue = native.memory_queue_index = None
    explicit.memory_queue = explicit.memory_queue_index = None

    model_input = _official_input()
    with torch.no_grad():
        native_output = native(copy.deepcopy(model_input), torch.device("cpu"))
        explicit_output = explicit(copy.deepcopy(model_input), torch.device("cpu"))

    official_keys = {"pred_cls", "pred_reg", "pred_stcls", "pred_flag", "att_1", "att_2"}
    assert set(native_output) == official_keys
    assert set(explicit_output) == official_keys
    for key in official_keys:
        assert torch.equal(native_output[key], explicit_output[key]), key


@pytest.mark.parametrize("arm", ["b0o0", "b1o0", "b0o1", "b1o1"])
def test_every_bxo_arm_is_eventized_including_b0o0(arm: str) -> None:
    native = MATR(_native_args())
    eventized = MATR(_event_args(arm=arm))

    assert native.event_enabled is False
    assert not hasattr(native, "event_transition_head")
    assert eventized.event_enabled is True
    assert eventized.model_variant == "eventmatr"
    assert eventized.event_arm == arm
    assert hasattr(eventized, "event_transition_head")
    assert hasattr(eventized, "event_owner_decoder")
    assert hasattr(eventized, "event_memory")
    assert set(eventized.state_dict()).difference(native.state_dict())


def test_bxo_cells_share_identical_trainable_capacity_and_loss_schema() -> None:
    models = {arm: MATR(_event_args(arm=arm)) for arm in (
        "b0o0", "b1o0", "b0o1", "b1o1"
    )}
    reference_keys = set(models["b0o0"].state_dict())
    reference_count = sum(parameter.numel() for parameter in models["b0o0"].parameters())
    for arm, model in models.items():
        assert set(model.state_dict()) == reference_keys, arm
        assert sum(parameter.numel() for parameter in model.parameters()) == reference_count

    schemas = {}
    for arm in models:
        args = _event_args(arm=arm)
        criterion = CriterionMATR(
            num_classes=args.num_of_class,
            matcher=HungarianMatcher(args),
            weight_dict={},
            losses=[],
            args=args,
        )
        schemas[arm] = set(criterion.weight_dict)
    assert all(schema == schemas["b0o0"] for schema in schemas.values())


def test_b1_birth_state_is_learned_and_not_the_native_fixed_flag_gate() -> None:
    args = _event_args()
    args.flag_threshold = 0.99
    model = MATR(args)

    assert model.event_enabled is True
    assert model.event_memory.birth_logit_threshold is None
    assert model.event_memory.end_logit_threshold is None
    assert not hasattr(model, "event_birth_probability_threshold")
    assert model.event_transition_head.state.weight.requires_grad
    assert model.event_transition_head.state.bias.requires_grad

    class_query = torch.randn((2, args.num_queries, args.hidden_dim), requires_grad=True)
    regression_query = torch.randn_like(class_query, requires_grad=True)
    state_logits, birth_logits, alive_logits, end_logits, _ = model.event_transition_head(
        class_query, regression_query
    )
    assert state_logits.shape == (2, args.num_queries, 4)
    expected_birth_margin = state_logits[..., 1] - torch.cat(
        (state_logits[..., :1], state_logits[..., 2:]), dim=-1
    ).max(dim=-1).values
    assert torch.allclose(birth_logits, expected_birth_margin)
    competitive_loss = (birth_logits - alive_logits + end_logits).mean()
    competitive_loss.backward()
    assert model.event_transition_head.state.weight.grad is not None
    assert model.event_transition_head.state.bias.grad is not None


def test_b1_event_losses_backpropagate_into_state_start_and_end_predictions() -> None:
    args = _event_args(arm="b1o0")
    matcher = HungarianMatcher(args)
    criterion = CriterionMATR(
        num_classes=args.num_of_class,
        matcher=matcher,
        weight_dict={},
        losses=[],
        args=args,
    )

    state_logits = torch.randn((1, 2, 4), requires_grad=True)
    candidate_starts = torch.tensor([[6.0, 3.0]], requires_grad=True)
    end_offsets = torch.tensor([[0.2, 0.3]], requires_grad=True)
    pred_cls = torch.randn((1, 2, 3), requires_grad=True)
    outputs = {
        "event_state_logits": state_logits,
        "event_birth_logits": criterion._state_margin(state_logits, 1),
        "event_alive_logits": criterion._state_margin(state_logits, 2),
        "event_end_logits": criterion._state_margin(state_logits, 3),
        "event_end_offsets": end_offsets,
        "event_candidate_start_frames": candidate_starts,
        "event_owner_state_logits": state_logits,
        "event_owner_end_offsets": end_offsets,
        "event_owner_class_logits": pred_cls,
        "pred_cls": pred_cls,
    }
    event_targets = torch.zeros((1, 2, 8), dtype=torch.float32)
    # One start crossing and one end crossing in the same prefix.
    event_targets[0, 0] = torch.tensor([0, 0, 5, 15, 10, 1, 0, 0])
    event_targets[0, 1] = torch.tensor([1, 1, 2, 10, 2, 0, 0, 1])
    targets = {
        "event_targets": event_targets,
        "event_valid_mask": torch.tensor([[True, True]]),
    }
    infos = {"video_name": ["loss_video"], "current_frame": torch.tensor([10])}

    losses = criterion.loss_event(outputs, targets, infos)
    expected_loss_keys = {
        "loss_event_birth",
        "loss_event_start_offset",
        "loss_event_alive",
        "loss_event_end",
        "loss_event_end_offset",
        "loss_event_owner_class",
        "loss_event_owner_state",
    }
    assert expected_loss_keys.issubset(losses)
    total = sum(losses[key] for key in expected_loss_keys)
    assert torch.isfinite(total)
    total.backward()

    assert state_logits.grad is not None and state_logits.grad.abs().sum() > 0
    assert candidate_starts.grad is not None and candidate_starts.grad.abs().sum() > 0
    assert end_offsets.grad is not None and end_offsets.grad.abs().sum() > 0


@pytest.mark.parametrize("arm", ["b0o0", "b1o0", "b0o1", "b1o1"])
def test_every_bxo_criterion_trains_event_lifecycle(arm: str) -> None:
    args = _event_args(arm=arm)
    criterion = CriterionMATR(
        num_classes=args.num_of_class,
        matcher=HungarianMatcher(args),
        weight_dict={},
        losses=[],
        args=args,
    )
    assert criterion.event_enabled is True
    expected = {
        "loss_event_birth",
        "loss_event_start_offset",
        "loss_event_alive",
        "loss_event_end",
        "loss_event_end_offset",
        "loss_event_owner_state",
    }
    assert expected.issubset(criterion.weight_dict)
    assert "loss_event_owner_class" in criterion.weight_dict


def test_native_matr_criterion_has_no_event_losses() -> None:
    args = _native_args()
    weight_dict = {"native_loss": 1.0}
    criterion = CriterionMATR(
        num_classes=args.num_of_class,
        matcher=HungarianMatcher(args),
        weight_dict=weight_dict,
        losses=[],
        args=args,
    )
    assert criterion.event_enabled is False
    assert criterion.weight_dict == {"native_loss": 1.0}


def test_future_perturbation_cannot_change_existing_lifecycle_or_ledger() -> None:
    left = _memory(emit_delay_frames=1)
    right = _memory(emit_delay_frames=1)

    prefix_left = [
        _step(left, 10, births=[POS], ends=[NEG], starts=[8]),
        _step(left, 12, births=[NEG], ends=[POS]),
        _step(left, 13, births=[NEG], ends=[NEG]),
    ]
    prefix_right = [
        _step(right, 10, births=[POS], ends=[NEG], starts=[8]),
        _step(right, 12, births=[NEG], ends=[POS]),
        _step(right, 13, births=[NEG], ends=[NEG]),
    ]
    frozen_ledger = copy.deepcopy(left.ledger("video_1"))

    for left_out, right_out in zip(prefix_left, prefix_right):
        for key in ("new_birth_mask", "ended_mask", "emitted_mask", "active_count"):
            assert torch.equal(left_out[key], right_out[key])

    _step(left, 20, births=[POS], ends=[NEG], features=[[1.0]], classes=[[POS, NEG, NEG]])
    _step(right, 20, births=[NEG], ends=[NEG], features=[[-1.0]], classes=[[NEG, POS, NEG]])

    assert left.ledger("video_1")[0] == frozen_ledger[0]
    assert right.ledger("video_1")[0] == frozen_ledger[0]


def test_b1o1_establishes_start_event_before_action_end() -> None:
    memory = _memory()
    output = _step(memory, 100, births=[POS], ends=[NEG], starts=[92])

    assert output["new_birth_mask"].tolist() == [[True]]
    records = memory.records("video_1")
    assert len(records) == 1
    assert records[0].start_frame == 92
    assert records[0].created_frame == 100
    assert records[0].end_frame is None
    assert records[0].status == "active"
    assert memory.ledger("video_1") == ()


def test_sticky_owner_survives_same_class_overlap_and_crossed_end_order() -> None:
    memory = _memory(ownership_mode="sticky_owner", emit_delay_frames=2)
    basis = [[1.0, 0.0], [0.0, 1.0]]
    same_class = [[POS, NEG, NEG], [POS, NEG, NEG]]

    _step(
        memory,
        10,
        births=[POS, NEG],
        features=basis,
        classes=same_class,
        starts=[8, 10],
    )
    first_id = memory.records("video_1")[0].event_id
    _step(
        memory,
        11,
        births=[NEG, POS],
        features=basis,
        classes=same_class,
        starts=[8, 9],
    )
    records = _record_map(memory)
    second_id = max(records)
    assert records[first_id].owner_query_id == 0
    assert records[second_id].owner_query_id == 1

    owner_embeddings, owner_padding, owner_ids = memory.owner_batch(
        ["video_1"], device=torch.device("cpu"), dtype=torch.float32, embedding_dim=2
    )
    owner_state_logits = torch.tensor(
        [[[NEG, NEG, POS, NEG], [NEG, NEG, NEG, POS]]], dtype=torch.float32
    )
    owner_runtime = {
        "owner_state_logits": owner_state_logits,
        "owner_end_offsets": torch.zeros((1, 2)),
        "owner_updated_embeddings": owner_embeddings,
        "owner_valid_mask": ~owner_padding,
        "owner_record_ids": owner_ids,
    }

    # Swap semantic evidence and deliberately make the query-level end signal
    # conflict with the owner-level signal.  The persistent owner decoder must
    # close event 1, not whichever event currently resembles query 0.
    output = _step(
        memory,
        20,
        births=[NEG, NEG],
        ends=[POS, NEG],
        features=[basis[1], basis[0]],
        classes=same_class,
        owner_runtime=owner_runtime,
    )
    assert output["ended_mask"].tolist() == [[False, True]]
    records = _record_map(memory)
    assert records[first_id].owner_query_id == 0
    assert records[first_id].status == "active"
    assert records[second_id].owner_query_id == 1
    assert records[second_id].end_frame is not None


def test_owner_decoder_supports_more_dynamic_records_than_matr_queries() -> None:
    decoder = OwnerEventDecoder(hidden_dim=8, num_classes=3, num_heads=2)
    owners = torch.randn((1, 12, 8))
    current_queries = torch.randn((1, 10, 8))
    padding = torch.zeros((1, 12), dtype=torch.bool)

    state, end_offset, classes, updated = decoder(owners, current_queries, padding)
    assert state.shape == (1, 12, 4)
    assert end_offset.shape == (1, 12)
    assert classes.shape == (1, 12, 3)
    assert updated.shape == (1, 12, 8)


def test_b1o0_fresh_rematch_is_explicit_and_feature_driven() -> None:
    memory = _memory(ownership_mode="fresh_rematch")
    basis = [[1.0, 0.0], [0.0, 1.0]]
    _step(memory, 10, births=[POS, POS], features=basis, starts=[8, 9])
    before = _record_map(memory)
    ids_by_owner = {record.owner_query_id: event_id for event_id, record in before.items()}

    _step(
        memory,
        11,
        births=[NEG, NEG],
        features=[basis[1], basis[0]],
        starts=[8, 9],
    )
    after = _record_map(memory)
    assert after[ids_by_owner[0]].owner_query_id == 1
    assert after[ids_by_owner[1]].owner_query_id == 0


def test_more_than_ten_concurrent_events_are_not_silently_truncated() -> None:
    memory = _memory(emit_delay_frames=0, resource_limit=0)
    query_count = 10
    basis = [
        [1.0 if row == column else 0.0 for column in range(query_count)]
        for row in range(query_count)
    ]
    classes = [[POS, NEG, NEG] for _ in range(query_count)]

    _step(
        memory,
        10,
        births=[POS] * query_count,
        ends=[NEG] * query_count,
        features=basis,
        classes=classes,
        starts=list(range(query_count)),
    )
    _step(
        memory,
        11,
        births=[NEG] * query_count,
        ends=[NEG] * query_count,
        features=basis,
        classes=classes,
        starts=list(range(query_count)),
    )
    _step(
        memory,
        12,
        births=[POS, POS] + [NEG] * (query_count - 2),
        ends=[NEG] * query_count,
        features=basis,
        classes=classes,
        starts=[10, 10] + list(range(2, query_count)),
    )
    records = memory.records("video_1")
    assert len(records) == 12
    assert sum(record.status == "active" for record in records) == 12
    assert len({record.event_id for record in records}) == 12
    assert memory.ledger("video_1") == ()


def test_false_birth_cancel_never_creates_a_final_interval() -> None:
    memory = _memory()
    _step(memory, 10, births=[POS], ends=[NEG], starts=[9])
    event_id = memory.records("video_1")[0].event_id

    assert memory.cancel("video_1", event_id, reason="false_birth") is True
    assert memory.ledger("video_1") == ()
    assert memory.last_audit["cancellations"][-1]["event_id"] == event_id

    _step(memory, 20, births=[NEG], ends=[POS])
    assert memory.ledger("video_1") == ()


@pytest.mark.parametrize("ownership_mode", ["fresh_rematch", "sticky_owner"])
def test_learned_owner_background_cancels_false_birth_and_allows_rebirth(
    ownership_mode: str,
) -> None:
    memory = _memory(ownership_mode=ownership_mode)
    _step(memory, 10, births=[POS], ends=[NEG], starts=[9])
    first_id = memory.records("video_1")[0].event_id

    owner_embeddings, owner_padding, owner_ids = memory.owner_batch(
        ["video_1"],
        device=torch.device("cpu"),
        dtype=torch.float32,
        embedding_dim=1,
    )
    owner_runtime = {
        "owner_state_logits": torch.tensor(
            [[[POS, NEG, NEG, NEG]]], dtype=torch.float32
        ),
        "owner_end_offsets": torch.zeros((1, 1)),
        "owner_updated_embeddings": owner_embeddings,
        "owner_valid_mask": ~owner_padding,
        "owner_record_ids": owner_ids,
    }
    cancelled = _step(
        memory,
        11,
        births=[NEG],
        ends=[NEG],
        owner_runtime=owner_runtime,
    )
    assert cancelled["cancelled_mask"].tolist() == [[True]]
    assert memory.records("video_1") == ()
    assert memory.ledger("video_1") == ()
    cancellation = memory.last_audit["cancellations"][-1]
    assert cancellation["video_name"] == "video_1"
    assert cancellation["event_id"] == first_id
    assert cancellation["reason"] == "learned_owner_background"
    assert cancellation["frame"] == 11.0
    assert cancellation["source"] == "predicted_unmatched"
    assert cancellation["target_event_id"] is None

    reborn = _step(memory, 12, births=[POS], ends=[NEG], starts=[12])
    assert reborn["new_birth_mask"].tolist() == [[True]]
    assert len(memory.records("video_1")) == 1
    assert memory.records("video_1")[0].event_id == first_id + 1


def test_same_frame_same_class_identical_embeddings_are_distinct_births() -> None:
    memory = _memory()
    identical_features = [[1.0, 0.0], [1.0, 0.0]]
    same_class = [[POS, NEG, NEG], [POS, NEG, NEG]]
    born = _step(
        memory,
        10,
        births=[POS, POS],
        ends=[NEG, NEG],
        starts=[8, 8],
        features=identical_features,
        classes=same_class,
    )

    assert born["new_birth_mask"].tolist() == [[True, True]]
    records = memory.records("video_1")
    assert len(records) == 2
    assert len({record.event_id for record in records}) == 2
    assert {record.start_frame for record in records} == {8.0}


def test_owner_four_state_ce_supervises_unmatched_queries_as_background() -> None:
    args = _event_args(arm="b1o1")
    criterion = CriterionMATR(
        num_classes=args.num_of_class,
        matcher=HungarianMatcher(args),
        weight_dict={},
        losses=[],
        args=args,
    )
    candidate_state = torch.randn((1, 2, 4), requires_grad=True)
    owner_state = torch.zeros((1, 2, 4), requires_grad=True)
    outputs = {
        "event_state_logits": candidate_state,
        "event_birth_logits": criterion._state_margin(candidate_state, 1),
        "event_alive_logits": criterion._state_margin(candidate_state, 2),
        "event_end_logits": criterion._state_margin(candidate_state, 3),
        "event_end_offsets": torch.zeros((1, 2), requires_grad=True),
        "event_candidate_start_frames": torch.zeros((1, 2), requires_grad=True),
        "event_owner_state_logits": owner_state,
        "event_owner_end_offsets": torch.zeros((1, 2), requires_grad=True),
        "event_owner_class_logits": torch.randn((1, 2, 3), requires_grad=True),
        "pred_cls": torch.randn((1, 2, 3), requires_grad=True),
    }
    event_targets = torch.zeros((1, 2, 8), dtype=torch.float32)
    event_targets[0, 0] = torch.tensor([0, 0, 5, 15, 5, 1, 0, 0])
    losses = criterion.loss_event(
        outputs,
        {
            "event_targets": event_targets,
            "event_valid_mask": torch.tensor([[True, False]]),
        },
        {"video_name": ["owner_ce"], "current_frame": torch.tensor([5])},
    )
    losses["loss_event_owner_state"].backward()

    assert owner_state.grad is not None
    # Assigned query learns START; unmatched query explicitly learns BG.
    assert owner_state.grad[0, 0, 1] < 0
    assert owner_state.grad[0, 1, 0] < 0


def test_end_emit_positive_length_and_exactly_once_contract() -> None:
    memory = _memory(emit_delay_frames=2)
    _step(memory, 10, births=[POS], ends=[NEG], starts=[8])
    event_id = memory.records("video_1")[0].event_id

    ended = _step(memory, 12, births=[NEG], ends=[POS])
    assert ended["ended_mask"].tolist() == [[True]]
    assert ended["emitted_mask"].tolist() == [[False]]
    assert memory.ledger("video_1") == ()

    emitted = _step(memory, 14, births=[NEG], ends=[NEG])
    assert emitted["emitted_mask"].tolist() == [[True]]
    ledger = memory.ledger("video_1")
    assert len(ledger) == 1
    row = ledger[0]
    assert row["event_id"] == event_id
    assert row["start_frame"] < row["end_frame"] <= row["emit_frame"]
    assert row["end_frame"] != row["emit_frame"]
    assert row["status"] == "emitted"

    _step(memory, 20, births=[NEG], ends=[POS])
    assert memory.ledger("video_1") == ledger


def test_positive_resource_limit_fails_closed_instead_of_dropping_event() -> None:
    memory = _memory(resource_limit=1)
    _step(memory, 10, births=[POS], ends=[NEG], starts=[8])
    _step(memory, 11, births=[NEG], ends=[NEG], starts=[8])

    with pytest.raises(RuntimeError, match="resource|capacity|limit"):
        _step(memory, 12, births=[POS], ends=[NEG], starts=[9])

    assert len(memory.records("video_1")) == 1


def test_padding_prefix_changes_no_lifecycle_or_previous_start_state() -> None:
    memory = _memory()
    _step(memory, 10, births=[POS], ends=[NEG], starts=[8], true_duration=20)
    before = memory.records("video_1")[0]
    frozen = (
        before.event_id,
        before.start_frame,
        before.status,
        before.owner_embedding.tolist(),
        before.class_distribution.tolist(),
    )

    padding = _step(
        memory,
        999,
        births=[NEG],
        ends=[POS],
        starts=[999],
        is_real_prefix=False,
        true_duration=20,
        is_eos=True,
    )
    assert padding["padding_prefixes_ignored"].tolist() == [1]
    assert padding["active_count"].tolist() == [1]
    assert padding["eos_observed"].tolist() == [0]
    assert memory.ledger("video_1") == ()

    # POS remains the same START state as frame 10.  If padding had modified
    # previous-state memory, this would create a duplicate event.
    _step(memory, 11, births=[POS], ends=[NEG], starts=[8], true_duration=20)
    after = memory.records("video_1")
    assert len(after) == 1
    assert (
        after[0].event_id,
        after[0].start_frame,
        after[0].status,
        after[0].owner_embedding.tolist(),
        after[0].class_distribution.tolist(),
    ) == frozen


def test_eos_is_observed_but_does_not_force_close_active_events() -> None:
    memory = _memory()
    _step(memory, 10, births=[POS], ends=[NEG], starts=[8], true_duration=12)
    eos = _step(
        memory,
        11,
        births=[POS],
        ends=[NEG],
        starts=[8],
        true_duration=12,
        is_eos=True,
    )
    assert eos["eos_observed"].tolist() == [1]
    assert len(memory.records("video_1")) == 1
    assert memory.records("video_1")[0].status == "active"
    assert memory.ledger("video_1") == ()


def test_formal_argmax_states_and_unified_score_contract() -> None:
    memory = DynamicEventMemory(
        birth_mode="instant_transition",
        ownership_mode="fresh_rematch",
        birth_logit_threshold=None,
        end_logit_threshold=None,
        min_duration_frames=1,
        emit_delay_frames=0,
        resource_limit=0,
        segment_size=1,
    )
    class_logits = [[2.0, 0.0, -1.0]]
    start_state = [[0.0, 3.0, 1.0, -2.0]]
    background_state = [[3.0, 0.0, 0.0, 0.0]]
    end_state = [[0.0, -1.0, 0.0, 3.0]]

    born = _step(
        memory,
        10,
        births=[NEG],
        ends=[NEG],
        starts=[8],
        classes=class_logits,
        candidate_states=start_state,
        true_duration=13,
    )
    assert born["new_birth_mask"].tolist() == [[True]]
    _step(
        memory,
        11,
        births=[POS],
        ends=[POS],
        starts=[8],
        classes=class_logits,
        candidate_states=background_state,
        true_duration=13,
    )
    closed = _step(
        memory,
        12,
        births=[POS],
        ends=[NEG],
        starts=[8],
        classes=class_logits,
        candidate_states=end_state,
        true_duration=13,
        is_eos=True,
    )
    assert closed["ended_mask"].tolist() == [[True]]
    assert closed["emitted_mask"].tolist() == [[True]]

    row = memory.ledger("video_1")[0]
    birth_confidence = torch.tensor(start_state[0]).softmax(dim=-1)[1].item()
    end_confidence = torch.tensor(end_state[0]).softmax(dim=-1)[3].item()
    class_confidence = torch.tensor(class_logits[0][:-1]).softmax(dim=-1).max().item()
    expected = (birth_confidence * class_confidence * end_confidence) ** (1.0 / 3.0)
    assert row["score"] == pytest.approx(expected)
    assert 0.0 <= row["score"] <= 1.0
    assert row["sequence_id"] == 0


def test_same_frame_emissions_have_strict_monotonic_sequence_ids() -> None:
    memory = _memory(emit_delay_frames=0)
    basis = [[1.0, 0.0], [0.0, 1.0]]
    _step(memory, 10, births=[POS, POS], ends=[NEG, NEG], features=basis, starts=[8, 9])
    _step(memory, 11, births=[NEG, NEG], ends=[POS, POS], features=basis, starts=[8, 9])

    ledger = memory.ledger("video_1")
    assert len(ledger) == 2
    assert {row["emit_frame"] for row in ledger} == {11.0}
    assert [row["sequence_id"] for row in ledger] == [0, 1]
    assert all(0.0 <= row["score"] <= 1.0 for row in ledger)


def test_scalar_stage_counts_do_not_collapse_records_that_share_one_query() -> None:
    memory = _memory(ownership_mode="sticky_owner", emit_delay_frames=0)
    _step(memory, 10, births=[POS], ends=[NEG], starts=[8])
    _step(memory, 11, births=[NEG], ends=[NEG], starts=[8])
    second = _step(memory, 12, births=[POS], ends=[NEG], starts=[9])
    assert second["birth_count"].tolist() == [1]
    assert len(memory.records("video_1")) == 2

    owner_embeddings, owner_padding, owner_ids = memory.owner_batch(
        ["video_1"],
        device=torch.device("cpu"),
        dtype=torch.float32,
        embedding_dim=1,
    )
    owner_runtime = {
        "owner_state_logits": torch.tensor(
            [[[NEG, NEG, NEG, POS], [NEG, NEG, NEG, POS]]],
            dtype=torch.float32,
        ),
        "owner_end_offsets": torch.zeros((1, 2)),
        "owner_updated_embeddings": owner_embeddings,
        "owner_valid_mask": ~owner_padding,
        "owner_record_ids": owner_ids,
    }
    closed = _step(
        memory,
        13,
        births=[NEG],
        ends=[NEG],
        owner_runtime=owner_runtime,
    )
    # Boolean masks are query-indexed and therefore contain one True.  Scalar
    # counters must preserve both dynamic records for scientific accounting.
    assert closed["ended_mask"].sum().item() == 1
    assert closed["emitted_mask"].sum().item() == 1
    assert closed["end_count"].tolist() == [2]
    assert closed["emit_count"].tolist() == [2]
    assert closed["cancellation_count"].tolist() == [0]
    assert len(memory.ledger("video_1")) == 2


def test_ragged_owner_decoder_backpropagates_across_two_prefixes() -> None:
    decoder = OwnerEventDecoder(hidden_dim=8, num_classes=3, num_heads=2)
    owner0 = torch.randn((1, 12, 8), requires_grad=True)
    queries1 = torch.randn((1, 10, 8), requires_grad=True)
    queries2 = torch.randn((1, 10, 8), requires_grad=True)
    padding = torch.zeros((1, 12), dtype=torch.bool)

    state1, end1, classes1, owners1 = decoder(owner0, queries1, padding)
    state2, end2, classes2, _ = decoder(owners1, queries2, padding)
    targets = torch.arange(12).reshape(1, 12) % 4
    class_targets = torch.arange(12).reshape(1, 12) % 3
    loss = (
        torch.nn.functional.cross_entropy(state1.reshape(-1, 4), targets.reshape(-1))
        + torch.nn.functional.cross_entropy(state2.reshape(-1, 4), targets.reshape(-1))
        + torch.nn.functional.cross_entropy(
            classes1.reshape(-1, 3), class_targets.reshape(-1)
        )
        + torch.nn.functional.cross_entropy(
            classes2.reshape(-1, 3), class_targets.reshape(-1)
        )
        + end1.square().mean()
        + end2.square().mean()
    )
    loss.backward()

    assert owner0.grad is not None and owner0.grad.abs().sum() > 0
    assert queries1.grad is not None and queries1.grad.abs().sum() > 0
    assert queries2.grad is not None and queries2.grad.abs().sum() > 0
    assert decoder.cross_attention.in_proj_weight.grad is not None
    assert decoder.state.weight.grad is not None
