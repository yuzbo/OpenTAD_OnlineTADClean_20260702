#!/usr/bin/env python3
"""Launch Prefix Route R6 only through a fresh repository-owned interpreter."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opentad.utils.prefix_route_formal_v2 import (  # noqa: E402
    PrefixRouteFormalError,
    _mint_formal_process_capability,
    build_formal_process_attestation,
)
from opentad.utils.evidence_bundle import EvidenceBundleError  # noqa: E402
from opentad.utils.prefix_route_protocol_v2 import (  # noqa: E402
    PrefixRouteProtocolV2Error,
    _load_formal_protocol_context,
)


_CHILD_ENV = "PREFIX_ROUTE_R6_ISOLATED_CHILD_NONCE"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("preflight", "evaluate"),
    )
    parser.add_argument("--review-attestation", type=Path, required=True)
    parser.add_argument("--review-signature", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--isolated-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--child-nonce", help=argparse.SUPPRESS)
    return parser.parse_args()


def _launch_child(args):
    nonce = secrets.token_hex(32)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        args.command,
        "--review-attestation",
        str(args.review_attestation),
        "--review-signature",
        str(args.review_signature),
        "--isolated-child",
        "--child-nonce",
        nonce,
    ]
    if args.bundle_root is not None:
        command.extend(["--bundle-root", str(args.bundle_root)])
    environment = dict(os.environ)
    environment[_CHILD_ENV] = nonce
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        check=False,
    )
    return completed.returncode


def _run_child(args):
    if (
        not args.isolated_child
        or not args.child_nonce
        or os.environ.pop(_CHILD_ENV, None) != args.child_nonce
    ):
        raise PrefixRouteFormalError(
            "formal R6 certificate requires the isolated launcher"
        )
    __import__("opentad.evaluations.prefix_route_r6_v2")
    __import__("opentad.utils.prefix_route_artifacts_v2")
    __import__("opentad.utils.prefix_route_fairness_v2")
    context = _load_formal_protocol_context(
        repo_root=REPO_ROOT,
        review_attestation_path=args.review_attestation,
        review_signature_path=args.review_signature,
        require_head=True,
    )
    process = build_formal_process_attestation(
        repo_root=REPO_ROOT,
        manifest_record=context.manifest_record,
        protocol_record=context.protocol_record,
        review_record=context.review_record,
        argv=sys.argv,
    )
    process_capability = _mint_formal_process_capability(process)
    if args.command == "preflight":
        print(
            json.dumps(
                {
                    "status": "PASS_FORMAL_R6_CLEAN_PROCESS_PREFLIGHT",
                    "process": process,
                    "process_capability_sha256": (
                        process_capability.attestation_sha256
                    ),
                    "gpu_hours": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    raise PrefixRouteFormalError(
        "MODEL_P0_RUNTIME_FACTORY_NOT_AUTHORIZED; "
        "Gate-A PASS does not authorize model construction or GPU execution"
    )


def clean_isolated_process(args):
    """The sole formal launcher; the parent can only spawn a fresh child."""

    if not args.isolated_child:
        return _launch_child(args)
    return _run_child(args)


def main():
    args = parse_args()
    try:
        return clean_isolated_process(args)
    except (
        EvidenceBundleError,
        PrefixRouteFormalError,
        PrefixRouteProtocolV2Error,
        OSError,
    ) as exc:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_FORMAL_R6",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "gpu_hours": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
