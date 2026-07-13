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
from opentad.utils.device import move_data_to_device  # noqa: E402
from opentad.utils.training_audit import (  # noqa: E402
    audit_training_update,
    snapshot_trainable_parameters,
)
from opentad.utils.full_petal_launch import is_full_petal_route  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run a bounded PersistentEventSet feature smoke")
    parser.add_argument("config")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--max-chunks", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
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


def main():
    args = parse_args()
    if args.max_chunks <= 0:
        raise ValueError("max_chunks must be positive")
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if is_full_petal_route(cfg):
        raise RuntimeError(
            "Full PETAL Q2 cannot use the legacy Stage-1 smoke launcher; "
            "complete B0 and independent review before the fixed-step profile"
        )
    if cfg.inference.load_from_raw_predictions:
        raise RuntimeError("load_from_raw_predictions is forbidden for Stage-1 smoke")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("PESStage1Smoke")
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
    optimizer = None
    if args.train_step:
        optimizer = build_optimizer(copy.deepcopy(cfg.optimizer), model, logger)
        model.train()
    else:
        model.eval()

    required_module_prefixes=("head",)
    steps = []
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
                raise RuntimeError(f"non-finite Stage-1 smoke cost at chunk {chunk_index}")
            cost.backward()
            optimizer.step()
            update = audit_training_update(
                model,
                before,
                required_module_prefixes=required_module_prefixes,
            )
            update.raise_for_errors()
            row = {
                "chunk_index": chunk_index,
                "losses": {key: float(value.detach().float().item()) for key, value in losses.items()},
                "update_audit": update.as_dict(),
            }
        else:
            with torch.no_grad():
                results = model(
                    **batch,
                    return_loss=False,
                    infer_cfg=cfg.inference,
                    post_cfg=cfg.post_processing,
                    ext_cls=dataset.class_map,
                )
            row = {"chunk_index": chunk_index, "results": _jsonable(results)}
        row["chunk_audit"] = _jsonable(model.last_chunk_audit)
        steps.append(row)

    if not steps:
        raise RuntimeError("Stage-1 smoke processed no feature chunks")
    report = {
        "passed": True,
        "mode": "train_step" if args.train_step else "inference",
        "config": os.path.abspath(args.config),
        "route_variant": str(cfg.get("route_variant", "base")),
        "max_chunks": args.max_chunks,
        "chunks_processed": len(steps),
        "load_from_raw_predictions": bool(cfg.inference.load_from_raw_predictions),
        "steps": steps,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Stage-1 smoke report written to %s", output)


if __name__ == "__main__":
    main()
