"""One official train-batch smoke for registered EventMATR D1 lanes.

The receipt is an execution/gradient gate only.  The locked test feature path
is an absent sentinel, and no metric or checkpoint from this script is a paper
result.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

import run_eventmatr_real_smoke as common


REGISTERED_LANES = ("N", "R", "T", "H", "TH")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video-anno", default=os.environ.get("MATR_PROTOCOL_ANNO")
    )
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
            Path(os.environ["MATR_D1_SMOKE_RECEIPT"])
            if os.environ.get("MATR_D1_SMOKE_RECEIPT")
            else None
        ),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.video_anno = common._required_path(
        args.video_anno, "MATR_PROTOCOL_ANNO"
    )
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
        raise SystemExit("--output or MATR_D1_SMOKE_RECEIPT is required")
    args.output = args.output.expanduser().resolve()
    if args.output.name != "eventmatr_d1_real_smoke.json":
        raise SystemExit(
            "D1 smoke receipt filename must be eventmatr_d1_real_smoke.json"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    return args


def _source_identity() -> dict:
    result = {}
    for field, env_name in (
        ("commit", "MATR_SOURCE_COMMIT"),
        ("tree", "MATR_SOURCE_TREE"),
        ("manifest_sha256", "MATR_MANIFEST_SHA256"),
    ):
        value = os.environ.get(env_name)
        if not value:
            raise SystemExit(f"{env_name} is required")
        result[field] = value
    return result


def _lane_args(base, lane: str):
    args = copy.deepcopy(base)
    if lane == "N":
        args.model_variant = "native_matr"
        args.event_arm = None
        args.event_lifecycle_version = "v1_dense"
        args.event_d1_lane = "th"
        args.birth_mode = "matr_delayed"
        args.ownership_mode = "fresh_rematch"
        return args

    d1_lane = lane.lower()
    uses_identity = lane in {"T", "TH"}
    args.model_variant = "eventmatr"
    args.event_lifecycle_version = "d1_censored"
    args.event_d1_lane = d1_lane
    args.birth_mode = "instant_transition"
    args.ownership_mode = "sticky_owner" if uses_identity else "fresh_rematch"
    args.event_arm = "b1o1" if uses_identity else "b1o0"
    args.event_teacher_forcing_ratio = 0.5
    args.event_identity_coef = 1.0
    return args


def main() -> None:
    cli = _parse_args()
    identity = _source_identity()
    if cli.device != "cuda" or not torch.cuda.is_available():
        raise SystemExit("formal D1 real-batch smoke requires CUDA")
    if torch.cuda.device_count() != 1:
        raise SystemExit("D1 smoke requires exactly one visible GPU")
    device = torch.device("cuda")
    random.seed(52)
    np.random.seed(52)
    torch.manual_seed(52)
    torch.cuda.manual_seed_all(52)

    common._load_project_modules()
    base_args = common._base_args(cli)
    base_args.event_lifecycle_version = "v1_dense"
    base_args.event_d1_lane = "th"
    base_args.event_teacher_forcing_ratio = 0.5
    base_args.event_identity_coef = 1.0
    dataset = common.THUMOS14Dataset(base_args, subset="train")
    index = common._find_supervised_index(dataset, "instant_transition")
    event_batch = common._load_real_batch(
        dataset, index, "instant_transition"
    )

    receipts = []
    with tempfile.TemporaryDirectory(
        prefix="eventmatr-d1-real-batch-", dir=str(cli.output.parent)
    ) as directory:
        checkpoint_dir = Path(directory)
        for lane in REGISTERED_LANES:
            args = _lane_args(base_args, lane)
            batch = (
                common._native_batch(event_batch)
                if lane == "N"
                else event_batch
            )
            internal_lane = "native_matr" if lane == "N" else lane
            receipt = common._run_lane(
                internal_lane,
                base_args,
                batch,
                device,
                checkpoint_dir,
                lane_args=args,
            )
            receipt["registered_lane"] = lane
            receipt["event_lifecycle_version"] = args.event_lifecycle_version
            receipt["event_d1_lane"] = (
                None if lane == "N" else args.event_d1_lane
            )
            receipts.append(receipt)

    payload = {
        "status": "PASS",
        "protocol": "eventmatr_d1_real_official_train_batch_v1",
        "scientific_scope": (
            "one official THUMOS14 train batch; execution and gradient evidence only"
        ),
        "strict_causal_paper_result_valid": False,
        "test_access": False,
        "checkpoint_updated": False,
        "source_identity": identity,
        "seed": 52,
        "device": torch.cuda.get_device_name(0),
        "visible_gpu_count": torch.cuda.device_count(),
        "source_index": index,
        "official_batch_size": int(base_args.batch),
        "lanes": receipts,
    }
    cli.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
