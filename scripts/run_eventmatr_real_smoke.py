"""One-real-batch scientific smoke for native MATR and all EventMATR cells.

The worker deliberately constructs ``THUMOS14Dataset(subset="train")`` and
uses the official feature/label path contract.  It never mounts the locked
test features.  Each lane performs forward, full criterion backward, one Adam
step, checkpoint serialization, and strict reload.  Event lanes additionally
prove gradients reach both the transition head and the ragged owner decoder.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ARM_MODES = {
    "b0o0": ("matr_delayed", "fresh_rematch"),
    "b1o0": ("instant_transition", "fresh_rematch"),
    "b0o1": ("matr_delayed", "sticky_owner"),
    "b1o1": ("instant_transition", "sticky_owner"),
}
LANES = ("native_matr", *ARM_MODES)


def _load_project_modules() -> None:
    # Delay Linux-only official dataset imports until after CLI parsing so
    # ``--help`` and static validation remain usable on workstations.
    global build_criterion, THUMOS14Dataset, build_model, parrallel_collate_fn
    from criterion import build_criterion
    from dataset import THUMOS14Dataset
    from models import build_model
    from util.utils import parrallel_collate_fn


def _required_path(
    value: Optional[str], name: str, *, literal: bool = True
) -> str:
    if not value:
        raise SystemExit(f"{name} is required")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise SystemExit(f"{name} must be absolute: {value}")
    if literal and not path.is_file():
        raise SystemExit(f"{name} does not exist: {path}")
    return str(path)


def _base_args(cli: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        video_anno=cli.video_anno,
        video_len_file=cli.video_len_pattern,
        num_of_class=21,
        num_frame=64,
        rgb=True,
        flow=True,
        video_feature_all_train=cli.train_feature,
        # A train-only dataset never opens this sentinel.
        video_feature_all_test=str(cli.output.parent / "LOCKED_TEST_NOT_MOUNTED.pickle"),
        ontal_label_file=cli.label_pattern,
        num_queries=10,
        p_videos=1,
        detect_len=16,
        anti_len=16,
        dropout=0.3,
        training=True,
        feat_dim=4096,
        hidden_dim=1024,
        ffn_dim=2048,
        e_nheads=8,
        enc_layers=3,
        d_nheads=4,
        dec_layers=5,
        pre_norm=False,
        activation="gelu",
        max_memory_len=7,
        memory_sampler="gap2",
        birth_mode="matr_delayed",
        ownership_mode="fresh_rematch",
        model_variant="eventmatr",
        event_arm="b0o0",
        event_birth_logit_threshold=None,
        event_end_logit_threshold=None,
        event_resource_limit=0,
        event_d13_variant="d12_control",
        event_d14_variant="none",
        event_runtime_during_training=False,
        study_protocol="matched_study",
        use_flag=True,
        flag_threshold=0.5,
        epochs=100,
        batch=64,
        min_lr=1e-8,
        max_lr=1e-5,
        weight_decay=1e-4,
        lr_gamma=0.9,
        lr_Tup=3,
        lr_Tcycle=10,
        drop_rate=0.3,
        use_empty_weight=False,
        eos_coef=1e-8,
        use_focal=True,
        reduce=1,
        cls_threshold=0.1,
        cls_coef=1,
        flag_coef=1,
        reg_l1_coef=1,
        reg_diou_coef=1,
        reg_stcls_coef=1,
        nms_threshold=0.3,
        random_seed=52,
    )


def _lane_args(base: SimpleNamespace, lane: str) -> SimpleNamespace:
    args = copy.deepcopy(base)
    if lane == "native_matr":
        args.model_variant = "native_matr"
        args.event_arm = None
        args.birth_mode = "matr_delayed"
        args.ownership_mode = "fresh_rematch"
    else:
        args.model_variant = "eventmatr"
        args.event_arm = lane
        args.birth_mode, args.ownership_mode = ARM_MODES[lane]
    return args


def _find_supervised_index(dataset: THUMOS14Dataset, birth_mode: str) -> int:
    dataset.event_birth_mode = birth_mode
    for index, (video_name, _, end_exclusive) in enumerate(dataset.inputs[0]):
        current_frame = int(end_exclusive) - 1
        if current_frame >= int(dataset.video_len[video_name]):
            continue
        _, valid = dataset._make_event_targets(video_name, current_frame)
        if bool(valid.any().item()):
            return index
    raise RuntimeError(f"no real {birth_mode} supervision prefix exists")


def _load_real_batch(
    dataset: THUMOS14Dataset, index: int, birth_mode: str
) -> Tuple[torch.Tensor, dict, dict]:
    dataset.event_enabled = True
    dataset.model_variant = "eventmatr"
    dataset.event_birth_mode = birth_mode
    # One smoke batch uses the official MATR batch size (64), not a special
    # singleton shape that would bypass or break native squeeze semantics.
    start = max(0, index - dataset.batch + 1)
    stop = min(len(dataset), start + dataset.batch)
    if stop - start < dataset.batch:
        start = max(0, stop - dataset.batch)
    indices = list(range(start, stop))
    if len(indices) != dataset.batch or index not in indices:
        raise RuntimeError("could not form one official-size real smoke batch")
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, indices),
        batch_size=dataset.batch,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    inputs, targets, infos = next(iter(loader))
    return parrallel_collate_fn(inputs, targets, infos, dataset.p_videos)


def _native_batch(event_batch: Tuple[torch.Tensor, dict, dict]):
    inputs, targets, infos = copy.deepcopy(event_batch)
    targets.pop("event_targets", None)
    targets.pop("event_valid_mask", None)
    return inputs, targets, infos


def _to_device(batch, device: torch.device):
    inputs, targets, infos = copy.deepcopy(batch)
    inputs = inputs.to(device)
    targets = {key: value.to(device) for key, value in targets.items()}
    return inputs, targets, infos


def _gradient_norm(parameter: torch.nn.Parameter, name: str) -> float:
    if parameter.grad is None:
        raise RuntimeError(f"required gradient is absent: {name}")
    if not torch.isfinite(parameter.grad).all():
        raise RuntimeError(f"required gradient is non-finite: {name}")
    norm = float(parameter.grad.detach().float().norm().item())
    if not math.isfinite(norm) or norm <= 0.0:
        raise RuntimeError(f"required gradient is zero/non-finite: {name}={norm}")
    return norm


def _model_inputs(args, inputs, infos, targets):
    if getattr(args, "event_lifecycle_version", "v1_dense") != "d1_censored":
        return {"inputs": inputs, "infos": infos}
    forbidden = {"duration", "true_duration", "video_time", "frame_to_time"}
    visible_infos = {
        key: value for key, value in infos.items() if key not in forbidden
    }
    event_targets = targets["event_targets"].clone()
    event_valid = targets["event_valid_mask"].to(event_targets.device).bool()
    observed_end = event_targets[..., 7] > 0.5
    event_targets[..., 3] = event_targets[..., 3].masked_fill(
        event_valid & ~observed_end, float("nan")
    )
    return {
        "inputs": inputs,
        "infos": visible_infos,
        "event_targets": event_targets,
        "event_valid_mask": targets["event_valid_mask"],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_lane(
    lane: str,
    base_args: SimpleNamespace,
    batch,
    device: torch.device,
    checkpoint_dir: Path,
    lane_args: Optional[SimpleNamespace] = None,
) -> dict:
    args = _lane_args(base_args, lane) if lane_args is None else lane_args
    inputs, targets, infos = _to_device(batch, device)
    model = torch.nn.DataParallel(build_model(args)).to(device)
    criterion = build_criterion(args, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.min_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    model.train()
    criterion.train()
    outputs = model(_model_inputs(args, inputs, infos, targets), device)
    loss_dict = criterion(outputs, targets, infos, device)
    weighted = {
        name: value * criterion.weight_dict[name]
        for name, value in loss_dict.items()
        if name in criterion.weight_dict
    }
    if not weighted:
        raise RuntimeError(f"{lane} produced no weighted loss")
    for name, value in weighted.items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"{lane} produced non-finite {name}")
    loss = sum(weighted.values())
    if not torch.isfinite(loss).all():
        raise RuntimeError(f"{lane} total loss is non-finite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    module = model.module
    gradient_norms = {}
    if lane != "native_matr":
        if args.event_lifecycle_version == "d1_censored":
            if module.event_transition_head.birth is None:
                raise RuntimeError(
                    "D1.2 real smoke requires an independent birth head"
                )
            gradient_norms = {
                "event_transition_shared": _gradient_norm(
                    module.event_transition_head.fuse[0].weight,
                    "event_transition_head.fuse.0.weight",
                ),
                "event_birth": _gradient_norm(
                    module.event_transition_head.birth.weight,
                    "event_transition_head.birth.weight",
                ),
            }
        else:
            gradient_norms = {
                "event_transition_state": _gradient_norm(
                    module.event_transition_head.state.weight,
                    "event_transition_head.state.weight",
                ),
            }
        gradient_norms.update(
            {
                "owner_state": _gradient_norm(
                    module.event_owner_decoder.state.weight,
                    "event_owner_decoder.state.weight",
                ),
                "owner_cross_attention": _gradient_norm(
                    module.event_owner_decoder.cross_attention.in_proj_weight,
                    "event_owner_decoder.cross_attention.in_proj_weight",
                ),
            }
        )
    finite_gradient_count = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"{lane} has non-finite gradient in {name}")
        finite_gradient_count += 1
    if finite_gradient_count == 0:
        raise RuntimeError(f"{lane} produced no model gradients")
    optimizer.step()

    valid_events = int(targets.get(
        "event_valid_mask", torch.zeros((), device=device)
    ).sum().item())
    required_event_gradients = {
        name: value
        for name, value in gradient_norms.items()
        if name.startswith("event_")
    }
    receipt = {
        "lane": lane,
        "model_variant": args.model_variant,
        "event_arm": args.event_arm,
        "birth_mode": args.birth_mode,
        "ownership_mode": args.ownership_mode,
        "event_d13_variant": getattr(args, "event_d13_variant", "d12_control"),
        "event_d14_variant": getattr(args, "event_d14_variant", "none"),
        "association_contract": getattr(
            module, "event_association_contract", None
        ),
        "birth_risk_contract": getattr(module, "event_birth_risk_contract", None),
        "birth_objective_contract": getattr(
            module, "event_birth_objective_contract", None
        ),
        "input_shape": list(inputs.shape),
        "video_names": [str(value) for value in infos["video_name"]],
        "current_frames": [
            int(value)
            for value in infos["current_frame"].detach().cpu().reshape(-1).tolist()
        ],
        "valid_event_rows": valid_events,
        "loss": float(loss.detach().cpu().item()),
        "weighted_losses": {
            loss_name: float(loss_value.detach().cpu().item())
            for loss_name, loss_value in sorted(weighted.items())
        },
        "finite_gradient_tensors": finite_gradient_count,
        "required_gradient_norms": gradient_norms,
        "event_gradient": lane == "native_matr"
        or (
            bool(required_event_gradients)
            and all(value > 0.0 for value in required_event_gradients.values())
        ),
        "birth_gradient": (
            gradient_norms["event_birth"] > 0.0
            if args.event_lifecycle_version == "d1_censored"
            else None
        ),
        "owner_gradient": lane == "native_matr" or gradient_norms[
            "owner_state"
        ] > 0.0,
    }

    checkpoint_path = checkpoint_dir / f"{lane}.one_step.pth"
    torch.save(
        {
            "lane": lane,
            "state_dict": model.state_dict(),
            "criterion_dict": criterion.state_dict(),
        },
        checkpoint_path,
    )
    checkpoint_hash = _sha256(checkpoint_path)
    checkpoint_bytes = checkpoint_path.stat().st_size
    receipt.update(
        {
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_bytes": checkpoint_bytes,
        }
    )

    # Avoid holding the trained model, Adam moments, reloaded model, and a
    # second GPU checkpoint copy simultaneously on the formal 32-GiB worker.
    del module, optimizer, criterion, model, outputs, loss_dict, weighted, loss
    del inputs, targets
    parameter = None
    value = None
    if device.type == "cuda":
        torch.cuda.empty_cache()
    reloaded_model = torch.nn.DataParallel(build_model(args)).to(device)
    reloaded_criterion = build_criterion(args, device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    reloaded_model.load_state_dict(checkpoint["state_dict"], strict=True)
    reloaded_criterion.load_state_dict(checkpoint["criterion_dict"], strict=True)
    if tuple(reloaded_model.state_dict()) != tuple(checkpoint["state_dict"]):
        raise RuntimeError(f"{lane} checkpoint key order changed on reload")
    for key, value in checkpoint["state_dict"].items():
        if not torch.equal(value, reloaded_model.state_dict()[key].detach().cpu()):
            raise RuntimeError(f"{lane} checkpoint tensor mismatch: {key}")

    receipt["strict_checkpoint_reload"] = True
    del checkpoint, reloaded_criterion, reloaded_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return receipt


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
            Path(os.environ["MATR_SMOKE_RECEIPT"])
            if os.environ.get("MATR_SMOKE_RECEIPT")
            else (
                Path(os.environ["MATR_OUTPUT_ROOT"]) / "eventmatr_real_smoke.json"
                if os.environ.get("MATR_OUTPUT_ROOT")
                else None
            )
        ),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.video_anno = _required_path(args.video_anno, "MATR_PROTOCOL_ANNO")
    args.train_feature = _required_path(args.train_feature, "MATR_TRAIN_FEATURE")
    args.video_len_pattern = _required_path(
        args.video_len_pattern, "MATR_VIDEO_LEN_PATTERN", literal=False
    )
    args.label_pattern = _required_path(
        args.label_pattern, "MATR_LABEL_PATTERN", literal=False
    )
    if args.output is None:
        raise SystemExit("--output or MATR_OUTPUT_ROOT is required")
    args.output = args.output.expanduser().resolve()
    if args.output.name != "eventmatr_real_smoke.json":
        raise SystemExit("smoke receipt filename must be eventmatr_real_smoke.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    return args


def main() -> None:
    cli = _parse_args()
    source_identity = {}
    for field, env_name in (
        ("commit", "MATR_SOURCE_COMMIT"),
        ("tree", "MATR_SOURCE_TREE"),
        ("manifest_sha256", "MATR_MANIFEST_SHA256"),
    ):
        value = os.environ.get(env_name)
        if not value:
            raise SystemExit(f"{env_name} is required")
        source_identity[field] = value
    if cli.device != "cuda":
        raise SystemExit("formal real-batch smoke requires --device cuda")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise SystemExit(
            "formal EventMATR smoke requires exactly one visible GPU per process"
        )
    device = torch.device("cuda")
    random.seed(52)
    np.random.seed(52)
    torch.manual_seed(52)
    torch.cuda.manual_seed_all(52)

    _load_project_modules()
    base_args = _base_args(cli)
    dataset = THUMOS14Dataset(base_args, subset="train")
    b0_index = _find_supervised_index(dataset, "matr_delayed")
    b1_index = _find_supervised_index(dataset, "instant_transition")
    b0_batch = _load_real_batch(dataset, b0_index, "matr_delayed")
    b1_batch = _load_real_batch(dataset, b1_index, "instant_transition")
    batches = {
        "native_matr": _native_batch(b0_batch),
        "b0o0": b0_batch,
        "b0o1": b0_batch,
        "b1o0": b1_batch,
        "b1o1": b1_batch,
    }

    receipts = []
    with tempfile.TemporaryDirectory(
        prefix="eventmatr-real-batch-", dir=str(cli.output.parent)
    ) as directory:
        checkpoint_dir = Path(directory)
        for lane in LANES:
            receipts.append(
                _run_lane(lane, base_args, batches[lane], device, checkpoint_dir)
            )

    payload = {
        "status": "PASS",
        "scientific_scope": "one real THUMOS14 train batch; no locked-test access",
        "test_access": False,
        "source_identity": source_identity,
        "seed": 52,
        "device": torch.cuda.get_device_name(0),
        "visible_gpu_count": torch.cuda.device_count(),
        "dataset_class": "THUMOS14Dataset",
        "source_indices": {
            "matr_delayed": b0_index,
            "instant_transition": b1_index,
        },
        "official_batch_size": int(base_args.batch),
        "lanes": receipts,
    }
    cli.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
