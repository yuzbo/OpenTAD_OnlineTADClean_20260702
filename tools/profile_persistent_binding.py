import argparse
import copy
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import statistics
import sys
import time


sys.dont_write_bytecode = True
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402
from mmengine.config import Config, DictAction  # noqa: E402

from opentad.cores import build_optimizer, build_scheduler  # noqa: E402
from opentad.datasets import build_dataloader, build_dataset  # noqa: E402
from opentad.models import build_detector  # noqa: E402
from opentad.utils import set_seed  # noqa: E402
from opentad.utils.device import move_data_to_device  # noqa: E402
from opentad.utils.online_protocol import (  # noqa: E402
    summarize_emission_ledger,
    validate_emission_ledger_summary,
)
from opentad.utils.training_audit import (  # noqa: E402
    audit_training_update,
    snapshot_trainable_parameters,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile persistent-binding training or causal inference"
    )
    parser.add_argument("config")
    parser.add_argument("--mode", choices=("train", "inference"), required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--measured-steps", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=705)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    return parser.parse_args()


def _percentile(values, percent):
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * float(percent) / 100.0
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _timing_summary(values):
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "min": min(values),
        "max": max(values),
    }


def _first_nonfinite_gradient(model):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            return name
    return None


def _canonical_sha256(value):
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main():
    args = parse_args()
    if args.warmup_steps < 0 or args.measured_steps <= 0:
        raise ValueError("warmup steps must be non-negative and measured steps positive")
    set_seed(args.seed)
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if cfg.inference.load_from_raw_predictions:
        raise RuntimeError("profiling cannot load raw predictions")
    if cfg.raw_video_finetuning:
        raise RuntimeError("persistent-binding profiling is feature-only")
    if cfg.solver.amp:
        raise RuntimeError("persistent-binding feature profiling requires FP32")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("PersistentBindingProfile")
    dataset = build_dataset(cfg.dataset[args.split], default_args=dict(logger=logger))
    loader_cfg = dict(cfg.solver[args.split])
    dataloader = build_dataloader(
        dataset,
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=False,
        **loader_cfg,
    )
    required_steps = args.warmup_steps + args.measured_steps
    if len(dataloader) < required_steps:
        raise RuntimeError(
            f"profile needs {required_steps} chunks but split has only {len(dataloader)}"
        )

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("persistent-binding profiling requires an allocated CUDA GPU")
    model = build_detector(cfg.model).to(device)
    model.reset_online_states()
    optimizer = None
    scheduler = None
    before = None
    if args.mode == "train":
        optimizer = build_optimizer(copy.deepcopy(cfg.optimizer), model, logger)
        scheduler, _ = build_scheduler(
            copy.deepcopy(cfg.scheduler),
            optimizer,
            len(dataloader),
        )
        before = snapshot_trainable_parameters(model)
        model.train()
    else:
        model.eval()

    iterator = iter(dataloader)
    measured_times = []
    loss_sums = {}
    dropped_gt_birth_targets = 0
    runtime_capacity_exhaustions = 0
    result_dict = {}
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(required_steps):
        if step == args.warmup_steps:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        raw_batch = next(iterator)
        batch = move_data_to_device(raw_batch, device)
        if args.mode == "train":
            optimizer.zero_grad(set_to_none=True)
            losses = model(**batch, return_loss=True)
            if not bool(torch.isfinite(losses["cost"]).all().item()):
                raise FloatingPointError(f"non-finite profile cost at step {step}")
            losses["cost"].backward()
            if cfg.solver.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.solver.clip_grad_norm,
                )
            bad_parameter = _first_nonfinite_gradient(model)
            if bad_parameter is not None:
                raise FloatingPointError(
                    f"non-finite profile gradient at step {step}: {bad_parameter}"
                )
            optimizer.step()
            scheduler.step()
            audit = model.last_episode_audit
            dropped_gt_birth_targets += int(audit["dropped_gt_birth_targets"])
            runtime_capacity_exhaustions += int(audit["runtime_capacity_exhaustions"])
            if step >= args.warmup_steps:
                for name, value in losses.items():
                    loss_sums[name] = loss_sums.get(name, 0.0) + float(
                        value.detach().float().item()
                    )
        else:
            with torch.no_grad():
                results = model(
                    **batch,
                    return_loss=False,
                    infer_cfg=cfg.inference,
                    post_cfg=cfg.post_processing,
                    ext_cls=dataset.class_map,
                )
            if step >= args.warmup_steps:
                for video_id, rows in results.items():
                    result_dict.setdefault(video_id, []).extend(
                        dict(row) for row in rows
                    )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if step >= args.warmup_steps:
            measured_times.append(elapsed)

    update_audit = None
    if args.mode == "train":
        update_audit = audit_training_update(
            model,
            before,
            required_module_prefixes=("head",),
        )
        update_audit.raise_for_errors()
        update_audit = {
            "passed": update_audit.passed,
            "module_summaries": update_audit.module_summaries,
            "missing_gradient_modules": list(update_audit.missing_gradient_modules),
            "missing_update_modules": list(update_audit.missing_update_modules),
        }
    emission_summary = None
    emission_ledger_sha256 = None
    if args.mode == "inference":
        emission_summary = summarize_emission_ledger(result_dict)
        validate_emission_ledger_summary(emission_summary)
        emission_ledger_sha256 = _canonical_sha256(result_dict)

    timings = _timing_summary(measured_times)
    estimated_full_pass_seconds = timings["mean"] * len(dataloader)
    report = {
        "passed": True,
        "mode": args.mode,
        "config": os.path.abspath(args.config),
        "binding_mode": str(cfg.model.trajectory_binding_mode),
        "split": args.split,
        "seed": args.seed,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measured_steps,
        "dataset_videos": len(dataset.packet_manifests),
        "dataset_chunks": len(dataloader),
        "dataset_test_mode": bool(dataset.test_mode),
        "strict_causal_control": bool(dataset.strict_causal_control),
        "dataset_video_ids_sha256": _canonical_sha256(
            sorted(dataset.packet_manifests)
        ),
        "timing_seconds": timings,
        "steps_per_second": 1.0 / timings["mean"],
        "estimated_full_pass_seconds": estimated_full_pass_seconds,
        "estimated_full_pass_gpu_hours": estimated_full_pass_seconds / 3600.0,
        "max_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / (1024.0**2),
        "dropped_gt_birth_targets": dropped_gt_birth_targets,
        "runtime_capacity_exhaustions": runtime_capacity_exhaustions,
        "mean_measured_losses": {
            name: value / args.measured_steps for name, value in loss_sums.items()
        },
        "update_audit": update_audit,
        "emission_summary": emission_summary,
        "emission_ledger_sha256": emission_ledger_sha256,
        "load_from_raw_predictions": bool(cfg.inference.load_from_raw_predictions),
        "save_raw_prediction": bool(cfg.inference.save_raw_prediction),
        "streaming_safe_emission": bool(
            cfg.post_processing.streaming_safe_emission
        ),
        "sliding_window": bool(cfg.post_processing.sliding_window),
        "raw_video_finetuning": bool(cfg.raw_video_finetuning),
        "amp": bool(cfg.solver.amp),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    logger.info("Persistent-binding profile written to %s", output)


if __name__ == "__main__":
    main()
