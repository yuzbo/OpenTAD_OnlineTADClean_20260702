"""Official-train memory-closure gate for the staged D1.6 risk control.

The previous one-batch smoke could not expose owner-population growth later in
the chronological stream.  This gate therefore executes and discards the first
64 consecutive official physical training batches, crossing both observed OOM
locations (batches 33 and 43).  It performs the same causal forward, backward,
Adam update, data order, seed, batch width, and learning rate as the registered
mechanism run.  It never mounts locked-test features, writes a checkpoint, or
computes a detector-performance metric.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch

import run_eventmatr_real_smoke as common
from eventmatr_d15_contracts import validate_parallel_window_causality


STRESS_BATCHES = 64
MAX_RESERVED_FRACTION = 0.90


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-anno", default=os.environ.get("MATR_PROTOCOL_ANNO"))
    parser.add_argument(
        "--train-feature", default=os.environ.get("MATR_TRAIN_FEATURE")
    )
    parser.add_argument(
        "--video-len-pattern", default=os.environ.get("MATR_VIDEO_LEN_PATTERN")
    )
    parser.add_argument(
        "--label-pattern", default=os.environ.get("MATR_LABEL_PATTERN")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(os.environ["MATR_D16_SMOKE_RECEIPT"])
            if os.environ.get("MATR_D16_SMOKE_RECEIPT")
            else None
        ),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.video_anno = common._required_path(args.video_anno, "MATR_PROTOCOL_ANNO")
    args.train_feature = common._required_path(
        args.train_feature, "MATR_TRAIN_FEATURE"
    )
    args.video_len_pattern = common._required_path(
        args.video_len_pattern, "MATR_VIDEO_LEN_PATTERN", literal=False
    )
    args.label_pattern = common._required_path(
        args.label_pattern, "MATR_LABEL_PATTERN", literal=False
    )
    if args.output is None:
        raise SystemExit("--output or MATR_D16_SMOKE_RECEIPT is required")
    args.output = args.output.expanduser().resolve()
    if args.output.name != "eventmatr_d16_risk_smoke.json":
        raise SystemExit(
            "D1.6 smoke receipt filename must be eventmatr_d16_risk_smoke.json"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    return args


def _source_identity() -> dict:
    identity = {}
    for field, env_name in (
        ("commit", "MATR_SOURCE_COMMIT"),
        ("tree", "MATR_SOURCE_TREE"),
        ("manifest_sha256", "MATR_MANIFEST_SHA256"),
    ):
        value = os.environ.get(env_name)
        if not value:
            raise SystemExit(f"{env_name} is required")
        identity[field] = value
    return identity


def _d16_args(base_args):
    args = common._lane_args(base_args, "b1o1")
    args.event_lifecycle_version = "d1_censored"
    args.event_d1_lane = "th"
    args.event_d13_variant = "combined"
    args.event_d14_variant = "decision_aligned_bag"
    args.event_d16_variant = "policy_independent"
    args.event_teacher_forcing_ratio = 0.5
    args.event_identity_coef = 1.0
    args.d11_effective_dose = True
    args.study_protocol = "d16_mechanism"
    args.epochs = 1
    return args


def _required_gradient_parameters(module) -> dict[str, torch.nn.Parameter]:
    return {
        "owner_state": module.event_owner_decoder.state.weight,
        "owner_cross_attention": (
            module.event_owner_decoder.cross_attention.in_proj_weight
        ),
        "matr_segment_decoder": (
            module.segment_decoder.layers[0].multihead_attn.in_proj_weight
        ),
        "matr_memory_decoder": (
            module.memory_decoder.layers[0].multihead_attn.in_proj_weight
        ),
    }


def main() -> None:
    cli = _parse_args()
    identity = _source_identity()
    if cli.device != "cuda" or not torch.cuda.is_available():
        raise SystemExit("formal D1.6 official memory stress requires CUDA")
    if torch.cuda.device_count() != 1:
        raise SystemExit("D1.6 memory stress requires exactly one visible GPU")
    device = torch.device("cuda")
    random.seed(52)
    np.random.seed(52)
    torch.manual_seed(52)
    torch.cuda.manual_seed_all(52)

    common._load_project_modules()
    from util.utils import memory_initialize

    base_args = common._base_args(cli)
    args = _d16_args(base_args)

    # Match formal construction order: official train dataset first, then model.
    dataset = common.THUMOS14Dataset(args, subset="train")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    if len(loader) != 3270:
        raise RuntimeError(f"D1.6 official loader drifted: {len(loader)} != 3270")

    model = torch.nn.DataParallel(common.build_model(args)).to(device)
    criterion = common.build_criterion(args, device)
    training_lr = args.min_lr + (args.max_lr - args.min_lr) / args.lr_Tup
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    model.train()
    criterion.train()
    memory_initialize(model, args)
    args.training = True

    required_parameters = _required_gradient_parameters(model.module)
    required_gradient_max_norms = {name: 0.0 for name in required_parameters}
    loss_min = math.inf
    loss_max = -math.inf
    optimizer_steps = 0
    torch.cuda.reset_peak_memory_stats(device)

    for batch_index, (inputs, targets, infos) in enumerate(loader):
        if batch_index >= STRESS_BATCHES:
            break
        inputs, targets, infos = common.parrallel_collate_fn(
            inputs, targets, infos, args.p_videos
        )
        validate_parallel_window_causality(
            inputs,
            video_names=[str(value) for value in infos["video_name"]],
            current_frames=[
                float(value)
                for value in infos["current_frame"].reshape(-1).tolist()
            ],
            segment_size=int(args.num_frame),
        )
        inputs = inputs.to(device)
        targets = {name: value.to(device) for name, value in targets.items()}
        outputs = model(common._model_inputs(args, inputs, infos, targets), device)
        loss_dict = criterion(outputs, targets, infos, device)
        weighted = {
            name: value * criterion.weight_dict[name]
            for name, value in loss_dict.items()
            if name in criterion.weight_dict
        }
        if not weighted:
            raise RuntimeError(f"D1.6 memory stress batch {batch_index} has no loss")
        loss = sum(weighted.values())
        if not torch.isfinite(loss).all():
            raise RuntimeError(
                f"D1.6 memory stress batch {batch_index} has non-finite loss"
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise RuntimeError(
                    f"D1.6 memory stress has non-finite gradient in {name}"
                )
        for name, parameter in required_parameters.items():
            if parameter.grad is None:
                continue
            norm = float(parameter.grad.detach().float().norm().item())
            if math.isfinite(norm):
                required_gradient_max_norms[name] = max(
                    required_gradient_max_norms[name], norm
                )
        optimizer.step()
        optimizer_steps += 1
        loss_value = float(loss.detach().cpu().item())
        loss_min = min(loss_min, loss_value)
        loss_max = max(loss_max, loss_value)

    if optimizer_steps != STRESS_BATCHES:
        raise RuntimeError(
            f"D1.6 memory stress closed {optimizer_steps} batches, "
            f"expected {STRESS_BATCHES}"
        )
    missing_gradients = [
        name for name, value in required_gradient_max_norms.items() if value <= 0.0
    ]
    if missing_gradients:
        raise RuntimeError(
            "D1.6 memory stress lacks required gradients: "
            + ", ".join(missing_gradients)
        )
    expected_contract = "policy_independent_competing_risk_v1"
    if model.module.event_owner_risk_contract != expected_contract:
        raise RuntimeError("D1.6 memory stress resolved the wrong risk contract")

    torch.cuda.synchronize(device)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    peak_reserved_fraction = peak_reserved / total_memory
    if peak_reserved_fraction > MAX_RESERVED_FRACTION:
        raise RuntimeError(
            "D1.6 memory stress lacks the preregistered 10 percent reserve: "
            f"{peak_reserved_fraction:.6f} > {MAX_RESERVED_FRACTION:.2f}"
        )

    payload = {
        "status": "PASS",
        "protocol": "eventmatr_d16_policy_independent_risk_memory_stress_v1",
        "scientific_scope": (
            "first 64 consecutive official THUMOS14 train physical batches; "
            "causal integration, gradient, and memory-closure gate only"
        ),
        "strict_causal_paper_result_valid": False,
        "paper_performance_valid": False,
        "formal_training_started": False,
        "test_access": False,
        "threshold_search": False,
        "checkpoint_updated": False,
        "performance_metric_computed": False,
        "source_identity": identity,
        "seed": 52,
        "device": torch.cuda.get_device_name(0),
        "visible_gpu_count": torch.cuda.device_count(),
        "official_batch_size": int(args.batch),
        "official_loader_batches": int(len(loader)),
        "stress_batch_start": 0,
        "stress_batch_end_inclusive": STRESS_BATCHES - 1,
        "optimizer_steps_discarded": optimizer_steps,
        "crossed_prior_failure_batches": [33, 43],
        "parallel_window_causality": "PASS_EVERY_BATCH",
        "owner_risk_contract": model.module.event_owner_risk_contract,
        "training_learning_rate": training_lr,
        "required_gradient_max_norms": required_gradient_max_norms,
        "finite_loss_range": [loss_min, loss_max],
        "memory_gate": {
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "device_total_bytes": total_memory,
            "peak_reserved_fraction": peak_reserved_fraction,
            "maximum_reserved_fraction": MAX_RESERVED_FRACTION,
            "minimum_headroom_fraction": 1.0 - MAX_RESERVED_FRACTION,
            "status": "PASS",
        },
    }
    cli.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
