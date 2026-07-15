#!/usr/bin/env python3
"""Build a hash-bound Full PETAL profile or formal launch ticket."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmengine.config import Config, DictAction  # noqa: E402
from opentad.utils.full_petal_launch import (  # noqa: E402
    FullPetalLaunchError,
    build_launch_ticket,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--mode", choices=("profile", "formal"), required=True)
    parser.add_argument("--b0", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--entrypoint", choices=("train", "test"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--id", dest="run_id", type=int, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--not-eval", action="store_true")
    parser.add_argument("--disable-deterministic", action="store_true")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    return parser, parser.parse_args(argv)


def _external_output(path):
    output = Path(path).expanduser().resolve()
    try:
        output.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise FullPetalLaunchError(
            "launch ticket evidence must remain outside the source repository"
        )
    if output.exists():
        raise FullPetalLaunchError(f"refusing to overwrite launch ticket: {output}")
    return output


def main(argv=None):
    parser, args = parse_args(argv)
    try:
        output = _external_output(args.output)
        cfg = Config.fromfile(args.config)
        cfg_overrides = dict(args.cfg_options or {})
        if cfg_overrides:
            cfg.merge_from_dict(cfg_overrides)
        ticket = build_launch_ticket(
            cfg,
            args.config,
            mode=args.mode,
            b0_path=args.b0,
            review_path=args.review,
            profile_path=args.profile,
            repository_root=ROOT,
            entrypoint=args.entrypoint,
            seed=args.seed,
            run_id=args.run_id,
            deterministic=not args.disable_deterministic,
            not_eval=args.not_eval,
            resume_path=args.resume,
            cfg_overrides=cfg_overrides,
            bundle_root=output.parent,
        )
    except FullPetalLaunchError as exc:
        parser.error(str(exc))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(ticket, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"FULL_PETAL_LAUNCH_TICKET={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
