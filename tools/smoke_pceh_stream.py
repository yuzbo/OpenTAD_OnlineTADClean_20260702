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
from opentad.utils.stream_smoke import run_stream_smoke  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run a bounded PCEH stream smoke")
    parser.add_argument("config")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--max-packets", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-step", action="store_true")
    parser.add_argument("--required-module", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    logger = logging.getLogger("PCEHSmoke")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    split_cfg = cfg.dataset[args.split]
    dataset = build_dataset(split_cfg, default_args=dict(logger=logger))
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
    required_modules = tuple(args.required_module)
    if args.train_step and not required_modules:
        required_modules = ("backbone", "projection", "head")
    forward_kwargs = {}
    if not args.train_step:
        forward_kwargs = {
            "infer_cfg": cfg.inference,
            "post_cfg": cfg.post_processing,
            "ext_cls": dataset.class_map,
        }
    report = run_stream_smoke(
        model,
        dataloader,
        optimizer=optimizer,
        max_packets=args.max_packets,
        device=device,
        required_module_prefixes=required_modules,
        forward_kwargs=forward_kwargs,
    )
    report.update(
        {
            "config": os.path.abspath(args.config),
            "split": args.split,
            "formal_training_ready": bool(cfg.get("formal_training_ready", False)),
        }
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Smoke report written to %s", output)


if __name__ == "__main__":
    main()
