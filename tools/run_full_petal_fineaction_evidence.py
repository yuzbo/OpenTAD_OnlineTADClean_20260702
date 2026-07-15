#!/usr/bin/env python3
"""Run one locked FineAction pytest and publish its signed evidence root."""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.utils.full_petal_fineaction_executor import (  # noqa: E402
    FineActionExecutionError,
    run_fineaction_loader_evidence,
    run_fineaction_preprocessing_evidence,
)


def _common(parser):
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--annotation-sha256", required=True)
    parser.add_argument("--media-inventory-sha256", required=True)
    parser.add_argument("--attestation-private-key", type=Path, required=True)
    parser.add_argument("--attestation-key-id", required=True)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--timeout-seconds", type=float, default=300.0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="kind", required=True)
    preprocessing = commands.add_parser("preprocessing")
    _common(preprocessing)
    preprocessing.add_argument("--timestamp-convention", required=True)
    preprocessing.add_argument("--frame-stride", type=int, required=True)
    loader = commands.add_parser("loader")
    _common(loader)
    loader.add_argument("--preprocessing-sha256", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    common = {
        "annotation_sha256": args.annotation_sha256,
        "media_inventory_sha256": args.media_inventory_sha256,
        "private_key_path": args.attestation_private_key,
        "key_id": args.attestation_key_id,
        "python_executable": args.python_executable,
        "timeout_seconds": args.timeout_seconds,
    }
    try:
        if args.kind == "preprocessing":
            path = run_fineaction_preprocessing_evidence(
                args.source,
                args.output_dir,
                timestamp_convention=args.timestamp_convention,
                frame_stride=args.frame_stride,
                **common,
            )
        else:
            path = run_fineaction_loader_evidence(
                args.source,
                args.output_dir,
                preprocessing_sha256=args.preprocessing_sha256,
                **common,
            )
    except FineActionExecutionError as exc:
        raise SystemExit(f"FineAction locked execution failed: {exc}") from exc
    print(f"FULL_PETAL_FINEACTION_EVIDENCE={path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
