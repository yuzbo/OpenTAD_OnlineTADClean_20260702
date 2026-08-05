"""Exact official-train-batch gate for the staged D1.6 risk control.

This performs one forward/backward/Adam/reload cycle in a temporary directory.
It never mounts locked-test features and produces no detector performance result.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

import run_eventmatr_real_smoke as common
from eventmatr_d15_contracts import validate_parallel_window_causality


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


def main() -> None:
    cli = _parse_args()
    identity = _source_identity()
    if cli.device != "cuda" or not torch.cuda.is_available():
        raise SystemExit("formal D1.6 official-batch smoke requires CUDA")
    if torch.cuda.device_count() != 1:
        raise SystemExit("D1.6 smoke requires exactly one visible GPU")
    device = torch.device("cuda")
    random.seed(52)
    np.random.seed(52)
    torch.manual_seed(52)
    torch.cuda.manual_seed_all(52)

    common._load_project_modules()
    base_args = common._base_args(cli)
    dataset = common.THUMOS14Dataset(base_args, subset="train")
    source_index = common._find_supervised_index(dataset, "instant_transition")
    event_batch = common._load_real_batch(
        dataset, source_index, "instant_transition"
    )
    inputs, _, infos = event_batch
    validate_parallel_window_causality(
        inputs,
        video_names=[str(value) for value in infos["video_name"]],
        current_frames=[
            float(value)
            for value in infos["current_frame"].reshape(-1).tolist()
        ],
        segment_size=int(base_args.num_frame),
    )

    args = common._lane_args(base_args, "b1o1")
    args.event_lifecycle_version = "d1_censored"
    args.event_d1_lane = "th"
    args.event_d13_variant = "combined"
    args.event_d14_variant = "decision_aligned_bag"
    args.event_d16_variant = "policy_independent"
    args.event_teacher_forcing_ratio = 0.5
    args.event_identity_coef = 1.0
    args.study_protocol = "d16_mechanism"

    with tempfile.TemporaryDirectory(
        prefix="eventmatr-d16-risk-smoke-", dir=str(cli.output.parent)
    ) as directory:
        lane = common._run_lane(
            "d16_policy_independent",
            base_args,
            event_batch,
            device,
            Path(directory),
            lane_args=args,
        )

    expected_contract = "policy_independent_competing_risk_v1"
    if lane.get("owner_risk_contract") != expected_contract:
        raise RuntimeError("D1.6 smoke resolved the wrong owner-risk contract")
    required = lane.get("required_gradient_norms", {})
    for name in (
        "owner_state",
        "owner_cross_attention",
        "matr_segment_decoder",
        "matr_memory_decoder",
    ):
        if float(required.get(name, 0.0)) <= 0.0:
            raise RuntimeError(f"D1.6 smoke lacks required {name} gradient")

    payload = {
        "status": "PASS",
        "protocol": "eventmatr_d16_policy_independent_risk_official_batch_v1",
        "scientific_scope": (
            "one official THUMOS14 train batch; causal integration and gradient gate only"
        ),
        "strict_causal_paper_result_valid": False,
        "paper_performance_valid": False,
        "formal_training_started": False,
        "test_access": False,
        "threshold_search": False,
        "checkpoint_updated": False,
        "source_identity": identity,
        "seed": 52,
        "device": torch.cuda.get_device_name(0),
        "visible_gpu_count": torch.cuda.device_count(),
        "source_index": int(source_index),
        "official_batch_size": int(base_args.batch),
        "parallel_window_causality": "PASS",
        "lane": lane,
    }
    cli.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

