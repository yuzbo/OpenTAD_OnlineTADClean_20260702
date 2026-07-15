#!/usr/bin/env python3
"""Generate one external Ed25519 trust root for Full PETAL evidence."""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.utils.full_petal_attestation import (  # noqa: E402
    AttestationError,
    generate_private_key,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        public_key = generate_private_key(args.output)
    except AttestationError as exc:
        parser.error(str(exc))
    print(f"FULL_PETAL_PRIVATE_KEY={args.output.resolve()}")
    print(f"FULL_PETAL_PUBLIC_KEY={public_key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
