import argparse
from dataclasses import asdict
import json
import os
import sys


sys.dont_write_bytecode = True
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from opentad.utils.causal_audit import (  # noqa: E402
    audit_emission_ledger,
    audit_packet_metadata,
    audit_recorded_trace_equivalence,
)
from opentad.utils.online_protocol import ProtocolViolation  # noqa: E402


def _load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _trace_rows(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and "trace" in value:
        return value["trace"]
    if isinstance(value, dict) and "results" in value:
        value = value["results"]
    if isinstance(value, dict):
        return [row for rows in value.values() for row in rows]
    raise ValueError("trace artifact must be a list, trace field, or results mapping")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Audit strict online TAD artifacts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    packets = subparsers.add_parser("packets", help="audit chronological packet metadata")
    packets.add_argument("artifact")

    ledger = subparsers.add_parser("ledger", help="audit immutable emission/read provenance")
    ledger.add_argument("artifact")

    future = subparsers.add_parser(
        "future-perturbation",
        help="compare reference and suffix-perturbed traces through a cut",
    )
    future.add_argument("reference")
    future.add_argument("perturbed")
    future.add_argument("--cut", type=float, required=True)

    chunks = subparsers.add_parser(
        "chunk-invariance",
        help="compare traces captured with two transport chunk sizes",
    )
    chunks.add_argument("reference")
    chunks.add_argument("candidate")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.command == "packets":
            data = _load_json(args.artifact)
            report = audit_packet_metadata(data.get("packets", data) if isinstance(data, dict) else data)
        elif args.command == "ledger":
            report = audit_emission_ledger(_load_json(args.artifact))
        elif args.command == "future-perturbation":
            report = audit_recorded_trace_equivalence(
                _trace_rows(_load_json(args.reference)),
                _trace_rows(_load_json(args.perturbed)),
                through_time=args.cut,
                name="future_perturbation_recorded",
            )
        else:
            report = audit_recorded_trace_equivalence(
                _trace_rows(_load_json(args.reference)),
                _trace_rows(_load_json(args.candidate)),
                name="chunk_invariance_recorded",
            )
    except (OSError, ValueError, KeyError, TypeError, ProtocolViolation, json.JSONDecodeError) as exc:
        print(f"causal compliance failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
