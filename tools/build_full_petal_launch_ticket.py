#!/usr/bin/env python3
"""Build a hash-bound Full PETAL profile or formal launch ticket."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmengine.config import Config  # noqa: E402
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
    return parser, parser.parse_args(argv)


def main(argv=None):
    parser, args = parse_args(argv)
    try:
        cfg = Config.fromfile(args.config)
        ticket = build_launch_ticket(
            cfg,
            args.config,
            mode=args.mode,
            b0_path=args.b0,
            review_path=args.review,
            profile_path=args.profile,
            repository_root=ROOT,
        )
    except FullPetalLaunchError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(ticket, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"FULL_PETAL_LAUNCH_TICKET={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
