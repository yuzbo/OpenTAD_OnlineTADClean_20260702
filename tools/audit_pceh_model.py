import argparse
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

from opentad.datasets import build_dataloader, build_dataset  # noqa: E402
from opentad.models import build_detector  # noqa: E402
from opentad.utils.model_causal_replay import audit_model_future_perturbation  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Replay PCEH with a perturbed future suffix")
    parser.add_argument("config")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--max-packets", type=int, default=8)
    parser.add_argument("--cut-packet-index", type=int, default=3)
    parser.add_argument("--perturbation-scale", type=float, default=17.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    logger = logging.getLogger("PCEHCausalAudit")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

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
    batches = []
    for batch in dataloader:
        batches.append(batch)
        if len(batches) >= args.max_packets:
            break

    device = torch.device(args.device)
    model = build_detector(cfg.model).to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint.get("state_dict_ema", checkpoint.get("state_dict", checkpoint))
        model.load_state_dict(state_dict)
    report = audit_model_future_perturbation(
        model,
        batches,
        cut_packet_index=args.cut_packet_index,
        perturbation_scale=args.perturbation_scale,
        device=device,
        forward_kwargs={
            "infer_cfg": cfg.inference,
            "post_cfg": cfg.post_processing,
            "ext_cls": dataset.class_map,
        },
    )
    report.update(
        {
            "config": os.path.abspath(args.config),
            "checkpoint": None if args.checkpoint is None else os.path.abspath(args.checkpoint),
            "split": args.split,
        }
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Causal replay report written to %s", output)


if __name__ == "__main__":
    main()
