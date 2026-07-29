"""Deterministic CPU pre-experiments for EventMATR D1.

These experiments establish problem reality and mechanism-level feasibility.
They are not dataset performance evidence and never access the locked test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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

from criterion.criterion import CriterionMATR  # noqa: E402
from criterion.matcher import HungarianMatcher  # noqa: E402
from models.event_memory import (  # noqa: E402
    DynamicEventMemory,
    temporal_viterbi_assignment,
)
from models.models import MATR  # noqa: E402


def _args(lane: str) -> SimpleNamespace:
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
        event_d1_lane=lane,
        event_teacher_forcing_ratio=0.5,
        event_identity_coef=1.0,
    )


def gradient_dilution_experiment() -> dict:
    result = {}
    for query_count in (2, 10, 100, 1000):
        dense_logits = torch.full((query_count,), -6.52, requires_grad=True)
        dense_target = torch.zeros(query_count)
        dense_target[0] = 1.0
        dense_loss = F.binary_cross_entropy_with_logits(
            dense_logits, dense_target
        )
        dense_loss.backward()

        event_logit = torch.tensor(-6.52, requires_grad=True)
        event_loss = F.softplus(-event_logit)
        event_loss.backward()
        result[str(query_count)] = {
            "dense_positive_gradient": abs(float(dense_logits.grad[0].item())),
            "event_normalized_positive_gradient": abs(
                float(event_logit.grad.item())
            ),
            "gradient_ratio": abs(
                float(event_logit.grad.item() / dense_logits.grad[0].item())
            ),
        }
    if result["1000"]["gradient_ratio"] < 900:
        raise RuntimeError("dense background dilution was not reproduced")
    return result


def censoring_experiment() -> dict:
    censored_logits = torch.tensor([-2.0, -1.0, 0.0], requires_grad=True)
    censored_loss = F.softplus(censored_logits).sum()
    censored_loss.backward()
    if not bool((censored_logits.grad > 0).all().item()):
        raise RuntimeError("right-censored survival gradient has wrong sign")

    observed_logits = torch.tensor([-2.0, -1.0, 0.0], requires_grad=True)
    observed_loss = (
        F.softplus(observed_logits[:-1]).sum()
        + F.softplus(-observed_logits[-1])
    )
    observed_loss.backward()
    if float(observed_logits.grad[-1].item()) >= 0:
        raise RuntimeError("observed end gradient does not increase end hazard")
    return {
        "right_censored_loss": float(censored_loss.item()),
        "right_censored_gradients": censored_logits.grad.tolist(),
        "observed_end_loss": float(observed_loss.item()),
        "observed_end_gradients": observed_logits.grad.tolist(),
        "observed_end_count": 1,
    }


def assignment_experiment() -> dict:
    state = torch.zeros((4, 2, 4))
    state[:, :, 0] = 8.0
    classes = torch.zeros((4, 2, 3))
    birth = torch.tensor(
        [[3.0, -3.0], [3.0, 3.2], [3.0, 3.1], [3.0, -3.0]]
    )
    classes[:, :, 0] = 2.0
    features = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.9, 0.1], [0.1, 0.9]],
            [[0.8, 0.2], [0.2, 0.8]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    path = temporal_viterbi_assignment(
        state,
        classes,
        features,
        0,
        birth_logits=birth,
    )
    if path != [0, 0, 0, 0]:
        raise RuntimeError("temporal assignment switched identity under ambiguity")
    if not bool((state.argmax(dim=-1) == 0).all().item()):
        raise RuntimeError("independent-birth assignment fixture lost BG dominance")
    return {
        "path": path,
        "switches": 0,
        "legacy_background_wins_all_queries": True,
    }


def _memory_step(
    memory: DynamicEventMemory,
    frame: int,
    births,
    starts,
    *,
    ends=None,
    oracle_births=None,
):
    query_count = len(births)
    ends = [-20.0] * query_count if ends is None else ends
    eye = torch.eye(query_count)
    return memory.step(
        video_names=["stress"],
        current_frames=torch.tensor([frame]),
        birth_logits=torch.tensor([births]),
        alive_logits=torch.full((1, query_count), 20.0),
        end_logits=torch.tensor([ends]),
        end_offsets=torch.zeros((1, query_count)),
        class_logits=torch.tensor(
            [[[20.0, -20.0, -20.0]] * query_count]
        ),
        query_features=eye.unsqueeze(0),
        candidate_start_frames=torch.tensor([starts], dtype=torch.float32),
        is_real_prefix=[True],
        oracle_births=[oracle_births or ()],
    )


def runtime_stress_experiment() -> dict:
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
    query_count = 10
    known_target = {
        "query_index": 0,
        "target_event_id": 77,
        "start_frame": 0.0,
        "source": "predicted_associated",
        "association_status": "associated",
        "force_create": False,
        "merge_predicted": True,
    }
    _memory_step(
        memory,
        10,
        [20.0] * query_count,
        [-9.0] + list(range(1, query_count)),
        oracle_births=[known_target],
    )
    _memory_step(
        memory,
        11,
        [-20.0] * query_count,
        list(range(query_count)),
    )
    _memory_step(
        memory,
        12,
        [20.0, 20.0] + [-20.0] * 8,
        [10.0, 11.0] + list(range(2, query_count)),
    )
    records = memory.records("stress")
    before_ids = {record.event_id for record in records}
    if len(records) != 12 or len(before_ids) != 12:
        raise RuntimeError("R>Q records were truncated or aliased")
    if min(record.start_frame for record in records) < 0:
        raise RuntimeError("negative start survived D1 clamp")

    targetless_record = next(
        record for record in records if record.target_event_id is None
    )
    targetless_cancelled_id = targetless_record.event_id
    memory.cancel(
        "stress", targetless_cancelled_id, reason="targetless_microexperiment"
    )
    _memory_step(
        memory,
        13,
        [-20.0] * query_count,
        list(range(query_count)),
    )
    _memory_step(
        memory,
        14,
        [-20.0, 20.0] + [-20.0] * 8,
        [14.0] * query_count,
    )
    after_targetless_birth = memory.records("stress")
    if any(
        record.event_id == targetless_cancelled_id
        for record in after_targetless_birth
    ):
        raise RuntimeError("targetless cancellation was incorrectly reacquired")
    targetless_rebirth = [
        record
        for record in after_targetless_birth
        if record.event_id not in before_ids and record.target_event_id is None
    ]
    if len(targetless_rebirth) != 1:
        raise RuntimeError("targetless rebirth did not allocate one new identity")

    known_record = next(
        record for record in after_targetless_birth if record.target_event_id == 77
    )
    known_cancelled_id = known_record.event_id
    memory.cancel("stress", known_cancelled_id, reason="known_target_microexperiment")
    recovery_result = _memory_step(
        memory,
        15,
        [20.0] + [-20.0] * 9,
        [15.0] * query_count,
        oracle_births=[known_target],
    )
    after = memory.records("stress")
    reacquired = [
        record for record in after if record.event_id == known_cancelled_id
    ]
    if (
        len(reacquired) != 1
        or reacquired[0].reacquisition_count != 1
        or reacquired[0].last_reacquisition_mode != "exact_target_id"
        or reacquired[0].source != "associated_error_recovery"
        or int(recovery_result["reacquisition_count"].item()) != 1
    ):
        raise RuntimeError("known target was not recovered by exact identity")
    return {
        "active_records_over_query_bandwidth": len(after),
        "unique_event_ids": len({record.event_id for record in after}),
        "minimum_start": min(record.start_frame for record in after),
        "targetless_cancelled_event_id": targetless_cancelled_id,
        "targetless_rebirth_event_id": targetless_rebirth[0].event_id,
        "known_target_recovered_event_id": known_cancelled_id,
        "reacquisition_count": reacquired[0].reacquisition_count,
        "reacquisition_mode": reacquired[0].last_reacquisition_mode,
        "capacity_exhaustions": memory.last_audit[
            "runtime_capacity_exhaustions"
        ],
    }


def integrated_lane_experiment(lane: str) -> dict:
    torch.manual_seed(29)
    args = _args(lane)
    model = MATR(args).train()
    model.memory_queue = model.memory_queue_index = None
    with torch.no_grad():
        model.event_transition_head.state.bias.copy_(
            torch.tensor([3.0, -3.0, -3.0, -3.0])
        )
        model.event_transition_head.birth.bias.fill_(3.0)
        model.event_owner_decoder.state.weight.zero_()
        model.event_owner_decoder.state.bias.copy_(
            torch.tensor([0.0, 3.0, 0.0])
        )

    event_targets = torch.zeros((3, 2, 8))
    event_targets[0, 0] = torch.tensor([0, 0, 2, float("nan"), 2, 1, 0, 0])
    event_targets[1, 0] = torch.tensor([0, 0, 2, float("nan"), 2, 0, 1, 0])
    event_targets[2, 0] = torch.tensor([0, 0, 2, 6, 2, 0, 0, 1])
    valid = torch.tensor([[True, False]] * 3)
    infos = {
        "st": torch.tensor([0, 1, 2]),
        "ed": torch.tensor([4, 5, 6]),
        "video_name": ["lane", "lane", "lane"],
        "current_frame": torch.tensor([4, 5, 6]),
        "segment_flag": torch.zeros(3, dtype=torch.long),
        "is_real_prefix": torch.ones(3, dtype=torch.bool),
        "is_eos": torch.tensor([False, False, True]),
    }
    outputs = model(
        {
            "inputs": torch.randn((3, 4, 8)),
            "infos": infos,
            "event_targets": event_targets,
            "event_valid_mask": valid,
        },
        torch.device("cpu"),
    )
    criterion = CriterionMATR(
        num_classes=args.num_of_class,
        matcher=HungarianMatcher(args),
        weight_dict={},
        losses=[],
        args=args,
    )
    losses = criterion.loss_event(
        outputs,
        {"event_targets": event_targets, "event_valid_mask": valid},
        infos,
    )
    total = sum(
        value
        for key, value in losses.items()
        if key.startswith("loss_event") and key in criterion.weight_dict
    )
    total.backward()
    transition_grad = float(model.event_transition_head.fuse[0].weight.grad.norm().item())
    birth_grad = float(model.event_transition_head.birth.weight.grad.norm().item())
    owner_grad = float(model.event_owner_decoder.state.weight.grad.norm().item())
    if transition_grad <= 0 or birth_grad <= 0 or owner_grad <= 0:
        raise RuntimeError("{} lane has a dead D1 gradient path".format(lane))
    return {
        "loss": float(total.item()),
        "ragged_rows": int(outputs["event_ragged_state_logits"].size(0)),
        "ragged_tracks": int(losses["event_ragged_track_count"].item()),
        "false_track_groups": int(
            losses["event_false_track_cancel_group_count"].item()
        ),
        "transition_gradient_norm": transition_grad,
        "birth_gradient_norm": birth_grad,
        "owner_gradient_norm": owner_grad,
        "uses_identity": bool(model.event_d1_use_identity),
        "uses_censored_hazard": bool(model.event_d1_use_hazard),
    }


def run() -> dict:
    receipt = {
        "protocol": "eventmatr_d12_local_microexperiments_v1",
        "scope": "synthetic problem-reality and differentiability evidence only",
        "strict_causal_paper_result_valid": False,
        "test_access": False,
        "checkpoint_updated": False,
        "gradient_dilution": gradient_dilution_experiment(),
        "censoring": censoring_experiment(),
        "temporal_assignment": assignment_experiment(),
        "runtime_stress": runtime_stress_experiment(),
        "lanes": {
            lane: integrated_lane_experiment(lane)
            for lane in ("r", "t", "h", "th")
        },
    }
    receipt["status"] = "PASS"
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = run()
    text = json.dumps(receipt, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
