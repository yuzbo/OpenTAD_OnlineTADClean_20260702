import argparse
import copy
import json
import logging
import os
from pathlib import Path
import sys


sys.dont_write_bytecode = True
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402
from mmengine.config import Config, DictAction  # noqa: E402

from opentad.cores import build_optimizer  # noqa: E402
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
        description="Run bounded real-feature checks for persistent binding"
    )
    parser.add_argument("config")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--max-chunks", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=705)
    parser.add_argument("--train-step", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    return parser.parse_args()


def _jsonable(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _compact_update(report):
    payload = report.as_dict()
    return {
        "passed": payload["passed"],
        "module_summaries": payload["module_summaries"],
        "missing_gradient_modules": payload["missing_gradient_modules"],
        "missing_update_modules": payload["missing_update_modules"],
    }


def main():
    args = parse_args()
    if args.max_chunks <= 0:
        raise ValueError("max_chunks must be positive")
    set_seed(args.seed)
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if cfg.inference.load_from_raw_predictions:
        raise RuntimeError("load_from_raw_predictions is forbidden for binding smoke")
    if cfg.model.type != "PersistentTrajectoryOnlineDetector":
        raise RuntimeError("binding smoke requires PersistentTrajectoryOnlineDetector")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("PersistentBindingSmoke")
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
    device = torch.device(args.device)
    model = build_detector(cfg.model).to(device)
    model.reset_online_states()
    optimizer = None
    if args.train_step:
        optimizer = build_optimizer(copy.deepcopy(cfg.optimizer), model, logger)
        model.train()
    else:
        model.eval()

    required_module_prefixes = ("head",)
    steps = []
    result_dict = {}
    dropped_gt_birth_targets = 0
    runtime_capacity_exhaustions = 0
    update_audits = []
    for chunk_index, raw_batch in enumerate(dataloader):
        if chunk_index >= args.max_chunks:
            break
        batch = move_data_to_device(raw_batch, device)
        if args.train_step:
            before = snapshot_trainable_parameters(model)
            optimizer.zero_grad(set_to_none=True)
            losses = model(**batch, return_loss=True)
            cost = losses.get("cost")
            if cost is None or not bool(torch.isfinite(cost).all().item()):
                raise RuntimeError(f"non-finite binding smoke cost at chunk {chunk_index}")
            cost.backward()
            optimizer.step()
            update = audit_training_update(
                model,
                before,
                required_module_prefixes=required_module_prefixes,
            )
            update.raise_for_errors()
            audit = _jsonable(model.last_episode_audit)
            dropped = int(audit["dropped_gt_birth_targets"])
            if dropped:
                raise RuntimeError(
                    f"ground-truth birth supervision was dropped at chunk {chunk_index}: {dropped}"
                )
            dropped_gt_birth_targets += dropped
            runtime_capacity_exhaustions += int(audit["runtime_capacity_exhaustions"])
            compact_update = _compact_update(update)
            update_audits.append(compact_update)
            steps.append(
                {
                    "chunk_index": chunk_index,
                    "losses": {
                        key: float(value.detach().float().item())
                        for key, value in losses.items()
                    },
                    "episode_audit": audit,
                    "update_audit": compact_update,
                }
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
            for video_id, rows in results.items():
                result_dict.setdefault(video_id, []).extend(_jsonable(rows))
            steps.append(
                {
                    "chunk_index": chunk_index,
                    "emissions": sum(len(rows) for rows in results.values()),
                }
            )

    if not steps:
        raise RuntimeError("binding smoke processed no real feature chunks")
    emission_summary = None
    if not args.train_step:
        emission_summary = summarize_emission_ledger(result_dict)
        validate_emission_ledger_summary(emission_summary)
    report = {
        "passed": True,
        "mode": "train_step" if args.train_step else "inference",
        "config": os.path.abspath(args.config),
        "binding_mode": str(cfg.model.trajectory_binding_mode),
        "split": args.split,
        "seed": args.seed,
        "max_chunks": args.max_chunks,
        "chunks_processed": len(steps),
        "dataset_videos": len(dataset.packet_manifests),
        "dataset_chunks": len(dataset),
        "load_from_raw_predictions": bool(cfg.inference.load_from_raw_predictions),
        "dropped_gt_birth_targets": dropped_gt_birth_targets,
        "runtime_capacity_exhaustions": runtime_capacity_exhaustions,
        "all_update_audits_passed": all(item["passed"] for item in update_audits),
        "emission_summary": emission_summary,
        "steps": steps,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    logger.info("Persistent-binding smoke report written to %s", output)


if __name__ == "__main__":
    main()
