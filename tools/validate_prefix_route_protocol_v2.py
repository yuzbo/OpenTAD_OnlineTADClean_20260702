#!/usr/bin/env python3
"""Build and execute the fail-closed Prefix-Route Protocol V2 gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opentad.utils.evidence_bundle import (  # noqa: E402
    EvidenceBundleError,
    read_stable_file_bytes,
    strict_json_from_bytes,
)
from opentad.utils.prefix_route_protocol_v2 import (  # noqa: E402
    MANIFEST_SCHEMA,
    PROTOCOL_PASS,
    PrefixRouteProtocolV2Error,
    authorize_collection,
    canonical_json_bytes,
    load_protocol,
    load_signed_review,
    load_source_manifest,
    validate_population_bundle,
    validate_r0_bundle,
)
from opentad.utils.prefix_route_r1_v2 import derive_r1_status  # noqa: E402


DEFAULT_PROTOCOL = (
    REPO_ROOT
    / "configs"
    / "causaltad"
    / "protocols"
    / "prefix_route_identifiability_v2.json"
)
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "configs"
    / "causaltad"
    / "protocols"
    / "prefix_route_identifiability_v2_manifest.json"
)


def _common(parser):
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)


def _review(parser, *, required):
    parser.add_argument("--review-attestation", type=Path, required=required)
    parser.add_argument("--review-signature", type=Path, required=required)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser(
        "build-manifest",
        help="Hash the exact required source set into a canonical manifest",
    )
    _common(build)
    build.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)

    validate = commands.add_parser(
        "validate-protocol",
        help="Validate Protocol V2 and, when present, its source manifest",
    )
    _common(validate)

    authorize = commands.add_parser(
        "authorize-collection",
        help="Authorize only R0/R1 after a valid signed independent PASS",
    )
    _common(authorize)
    _review(authorize, required=False)

    population = commands.add_parser(
        "validate-population",
        help="Derive the 211/213 decision from contained source evidence",
    )
    _common(population)
    _review(population, required=True)
    population.add_argument("--bundle-root", type=Path, required=True)
    population.add_argument("--request", type=Path, required=True)

    r0 = commands.add_parser(
        "validate-r0",
        help="Recompute and verify a canonical aggregate-only R0 report",
    )
    _common(r0)
    _review(r0, required=True)
    r0.add_argument("--bundle-root", type=Path, required=True)
    r0.add_argument("--population-request", type=Path, required=True)
    r0.add_argument("--r0-request", type=Path, required=True)

    r1 = commands.add_parser(
        "validate-r1",
        help="Re-read sources and derive the R1 existing-cache status",
    )
    _common(r1)
    _review(r1, required=True)
    r1.add_argument("--bundle-root", type=Path, required=True)
    r1.add_argument("--population-request", type=Path, required=True)
    r1.add_argument("--r1-request", type=Path, required=True)
    return parser.parse_args()


def _read_json(path, label, *, canonical):
    _, payload = read_stable_file_bytes(path, label)
    value = strict_json_from_bytes(payload, label, require_object=True)
    if canonical and payload != canonical_json_bytes(value):
        raise PrefixRouteProtocolV2Error(f"{label} is not canonical JSON")
    return value


def _load_fixed(args):
    protocol = load_protocol(args.protocol)
    manifest = load_source_manifest(
        args.manifest,
        protocol_record=protocol,
        repo_root=args.repo_root,
        check_worktree=True,
    )
    return protocol, manifest


def _load_pass(args, protocol, manifest):
    review = load_signed_review(
        attestation_path=args.review_attestation,
        signature_path=args.review_signature,
        protocol_record=protocol,
        manifest_record=manifest,
        repo_root=args.repo_root,
        require_head=True,
    )
    if review["attestation"]["verdict"] != PROTOCOL_PASS:
        raise PrefixRouteProtocolV2Error(
            "signed review verdict does not authorize evidence collection"
        )
    return review


def _population(args, protocol, review):
    request = _read_json(
        args.population_request
        if hasattr(args, "population_request")
        else args.request,
        "population request",
        canonical=True,
    )
    return validate_population_bundle(
        request,
        bundle_root=args.bundle_root,
        protocol_record=protocol,
        review_record=review,
    )


def _build_manifest(args):
    protocol = load_protocol(args.protocol)
    entries = {}
    root = args.repo_root.resolve()
    for relative in protocol["protocol"]["source_bindings"]["required_paths"]:
        path = root.joinpath(*PurePosixPath(relative).parts)
        _, payload = read_stable_file_bytes(path, f"manifest source {relative}")
        entries[relative] = hashlib.sha256(payload).hexdigest()
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "protocol_id": protocol["protocol"]["protocol_id"],
        "protocol_sha256": protocol["sha256"],
        "entries": entries,
    }
    payload = canonical_json_bytes(manifest)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.staged")
    if temporary.exists():
        raise PrefixRouteProtocolV2Error(
            f"staged manifest already exists: {temporary}"
        )
    temporary.write_bytes(payload)
    os.replace(temporary, output)
    return {
        "status": "SOURCE_MANIFEST_BUILT",
        "path": str(output),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "entry_count": len(entries),
    }


def main():
    args = parse_args()
    try:
        if args.command == "build-manifest":
            result = _build_manifest(args)
        elif args.command == "validate-protocol":
            protocol = load_protocol(args.protocol)
            result = {
                "status": "PROTOCOL_V2_VALID_REVIEW_REQUIRED",
                "protocol_sha256": protocol["sha256"],
                "gpu_hours": 0,
            }
            if args.manifest.exists():
                manifest = load_source_manifest(
                    args.manifest,
                    protocol_record=protocol,
                    repo_root=args.repo_root,
                    check_worktree=True,
                )
                result["manifest_sha256"] = manifest["sha256"]
                result["manifest_entry_count"] = len(
                    manifest["manifest"]["entries"]
                )
        elif args.command == "authorize-collection":
            result = authorize_collection(
                protocol_path=args.protocol,
                manifest_path=args.manifest,
                repo_root=args.repo_root,
                attestation_path=args.review_attestation,
                signature_path=args.review_signature,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["authorized"] else 2
        else:
            protocol, manifest = _load_fixed(args)
            review = _load_pass(args, protocol, manifest)
            population = _population(args, protocol, review)
            if args.command == "validate-population":
                result = population
            elif args.command == "validate-r0":
                request = _read_json(
                    args.r0_request,
                    "R0 request",
                    canonical=True,
                )
                ledger_relative = protocol["protocol"]["population"][
                    "exposure_policy"
                ]["ledger_path"]
                frozen_ledger = subprocess_git_show(
                    args.repo_root,
                    review["attestation"]["protocol_commit"],
                    ledger_relative,
                )
                result = validate_r0_bundle(
                    request,
                    bundle_root=args.bundle_root,
                    protocol_record=protocol,
                    review_record=review,
                    population_record=population,
                    frozen_ledger_prefix=frozen_ledger,
                )
            elif args.command == "validate-r1":
                request = _read_json(
                    args.r1_request,
                    "R1 request",
                    canonical=True,
                )
                result = derive_r1_status(
                    request,
                    bundle_root=args.bundle_root,
                    protocol_id=protocol["protocol"]["protocol_id"],
                    protocol_sha256=protocol["sha256"],
                    review_attestation_sha256=review["attestation_sha256"],
                    canonical_video_ids=population["canonical_ids"],
                )
                if result["status"] != "PASS_R1_EXISTING_CACHE_CERTIFIED":
                    print(json.dumps(result, indent=2, sort_keys=True))
                    return 3
            else:
                raise PrefixRouteProtocolV2Error("unknown command")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        EvidenceBundleError,
        PrefixRouteProtocolV2Error,
        OSError,
        ValueError,
    ) as exc:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_PROTOCOL_V2",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2


def subprocess_git_show(repo_root, commit, relative_path):
    import subprocess

    try:
        return subprocess.run(
            ["git", "show", f"{commit}:{relative_path}"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PrefixRouteProtocolV2Error(
            "cannot read frozen exposure-ledger prefix"
        ) from exc


if __name__ == "__main__":
    raise SystemExit(main())
