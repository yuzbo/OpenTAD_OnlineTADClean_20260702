#!/usr/bin/env python3
"""Validate the frozen prefix-route protocol and future evidence bindings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.utils.evidence_bundle import (  # noqa: E402
    read_stable_file_bytes,
    strict_json_from_bytes,
)
from opentad.utils.prefix_route_protocol import (  # noqa: E402
    PROTOCOL_PASS,
    PrefixRouteProtocolError,
    collection_authorization,
    load_protocol,
    load_review_certificate,
    validate_population_certificate,
    validate_r0_evidence,
    validate_r1_certificate,
    verify_protocol_git_binding,
)


DEFAULT_PROTOCOL = (
    ROOT
    / "configs"
    / "causaltad"
    / "protocols"
    / "prefix_route_identifiability_v1.json"
)


def _add_protocol(parser):
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--repo-root", type=Path, default=ROOT)


def _add_review(parser):
    parser.add_argument("--review-certificate", type=Path, required=True)
    parser.add_argument("--review-artifact", type=Path, required=True)
    parser.add_argument("--author-id", required=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser(
        "validate-protocol",
        help="Validate the protocol without authorizing evidence collection",
    )
    _add_protocol(validate)

    authorize = commands.add_parser(
        "authorize-collection",
        help="Evaluate whether an independent PASS authorizes read-only R0/R1",
    )
    _add_protocol(authorize)
    authorize.add_argument("--review-certificate", type=Path)
    authorize.add_argument("--review-artifact", type=Path)
    authorize.add_argument("--author-id")

    population = commands.add_parser(
        "validate-population",
        help="Validate the source-derived 211-versus-213 certificate",
    )
    _add_protocol(population)
    _add_review(population)
    population.add_argument("--population-certificate", type=Path, required=True)

    r0 = commands.add_parser(
        "validate-r0",
        help="Validate a future aggregate-only R0 evidence object",
    )
    _add_protocol(r0)
    _add_review(r0)
    r0.add_argument("--population-certificate", type=Path, required=True)
    r0.add_argument("--r0-evidence", type=Path, required=True)

    r1 = commands.add_parser(
        "validate-r1",
        help="Validate a future R1 cache-causality certificate",
    )
    _add_protocol(r1)
    _add_review(r1)
    r1.add_argument("--r1-certificate", type=Path, required=True)
    return parser.parse_args(argv)


def _read_json(path, label):
    resolved, payload = read_stable_file_bytes(path, label)
    value = strict_json_from_bytes(payload, label, require_object=True)
    return {
        "path": resolved,
        "bytes": payload,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "value": value,
    }


def _load_review(args, protocol_record):
    if args.review_certificate is None:
        return None
    if not args.author_id:
        raise PrefixRouteProtocolError(
            "--author-id is required with --review-certificate"
        )
    if args.review_artifact is None:
        raise PrefixRouteProtocolError(
            "--review-artifact is required with --review-certificate"
        )
    review = load_review_certificate(
        args.review_certificate,
        protocol=protocol_record["protocol"],
        protocol_sha256=protocol_record["sha256"],
        author_id=args.author_id,
        review_artifact_path=args.review_artifact,
    )
    certificate = review["certificate"]
    verify_protocol_git_binding(
        protocol_path=args.protocol,
        repo_root=args.repo_root,
        commit=certificate["protocol_commit"],
        expected_sha256=protocol_record["sha256"],
    )
    return review


def _require_pass(review):
    if review is None or review["certificate"]["verdict"] != PROTOCOL_PASS:
        raise PrefixRouteProtocolError(
            "evidence validation requires an independent protocol PASS"
        )


def _protocol_result(protocol_record):
    return {
        "authorized": False,
        "status": "PROTOCOL_VALID_REVIEW_REQUIRED",
        "protocol_id": protocol_record["protocol"]["protocol_id"],
        "protocol_sha256": protocol_record["sha256"],
        "gpu_hours_authorized": 0,
    }


def main(argv=None):
    args = parse_args(argv)
    try:
        protocol_record = load_protocol(
            args.protocol,
            repo_root=args.repo_root,
        )
        if args.command == "validate-protocol":
            result = _protocol_result(protocol_record)
        elif args.command == "authorize-collection":
            review = _load_review(args, protocol_record)
            result = collection_authorization(
                protocol_record=protocol_record,
                review_record=review,
            )
            result["protocol_sha256"] = protocol_record["sha256"]
        elif args.command == "validate-population":
            review = _load_review(args, protocol_record)
            _require_pass(review)
            population = _read_json(
                args.population_certificate,
                "reporting population certificate",
            )
            validate_population_certificate(
                population["value"],
                protocol=protocol_record["protocol"],
                protocol_sha256=protocol_record["sha256"],
                review_certificate_sha256=review["sha256"],
            )
            result = {
                "authorized": False,
                "status": "POPULATION_CERTIFICATE_VALID_R0_NOT_RUN",
                "population_certificate_sha256": population["sha256"],
            }
        elif args.command == "validate-r0":
            review = _load_review(args, protocol_record)
            _require_pass(review)
            population = _read_json(
                args.population_certificate,
                "reporting population certificate",
            )
            validate_population_certificate(
                population["value"],
                protocol=protocol_record["protocol"],
                protocol_sha256=protocol_record["sha256"],
                review_certificate_sha256=review["sha256"],
            )
            r0 = _read_json(args.r0_evidence, "R0 evidence")
            validate_r0_evidence(
                r0["value"],
                protocol=protocol_record["protocol"],
                protocol_sha256=protocol_record["sha256"],
                review_certificate_sha256=review["sha256"],
                population_certificate_sha256=population["sha256"],
            )
            result = {
                "authorized": False,
                "status": "R0_EVIDENCE_STRUCTURALLY_VALID_ROUTE_REVIEW_REQUIRED",
                "r0_evidence_sha256": r0["sha256"],
            }
        elif args.command == "validate-r1":
            review = _load_review(args, protocol_record)
            _require_pass(review)
            r1 = _read_json(args.r1_certificate, "R1 certificate")
            validate_r1_certificate(
                r1["value"],
                protocol=protocol_record["protocol"],
                protocol_sha256=protocol_record["sha256"],
                review_certificate_sha256=review["sha256"],
            )
            result = {
                "authorized": False,
                "status": "R1_CERTIFICATE_STRUCTURALLY_VALID_ROUTE_REVIEW_REQUIRED",
                "r1_certificate_sha256": r1["sha256"],
            }
        else:
            raise PrefixRouteProtocolError(f"unsupported command: {args.command}")
    except PrefixRouteProtocolError as exc:
        print(f"PREFIX_ROUTE_PROTOCOL_ERROR={exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            result,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0 if result.get("authorized") or args.command != "authorize-collection" else 2


if __name__ == "__main__":
    raise SystemExit(main())
