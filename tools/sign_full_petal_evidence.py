#!/usr/bin/env python3
"""Sign a complete Full PETAL evidence JSON object with a locked role."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.utils.full_petal_attestation import (  # noqa: E402
    AttestationError,
    sign_payload,
)


ROLES = (
    "b0-runner",
    "independent-reviewer",
    "fixed-step-profile",
    "formal-run",
)


def _load(path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AttestationError(f"failed to read evidence JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AttestationError("evidence JSON must contain one object")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--key-id", required=True)
    parser.add_argument("--role", choices=ROLES, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"refusing to overwrite signed evidence: {args.output}")
    try:
        signed = sign_payload(
            _load(args.input),
            private_key_path=args.private_key,
            key_id=args.key_id,
            role=args.role,
        )
    except AttestationError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(signed, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"FULL_PETAL_SIGNED_EVIDENCE={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
