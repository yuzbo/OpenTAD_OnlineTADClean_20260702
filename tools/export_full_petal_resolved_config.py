#!/usr/bin/env python3
"""Export a standalone, canonical Full PETAL resolved-config snapshot."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from mmengine import Config


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.utils.evidence_bundle import (  # noqa: E402
    EvidenceBundleError,
    publish_exclusive_file,
)
from opentad.utils.full_petal_launch import (  # noqa: E402
    resolved_config_payload,
    resolved_config_sha256,
)


def _outside_repository(path):
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError:
        return resolved
    raise EvidenceBundleError(
        "resolved-config evidence must remain outside the source repository"
    )


def export_resolved_config(config_path, output_path):
    source = Path(config_path).expanduser().resolve(strict=True)
    cfg = Config.fromfile(str(source))
    payload = resolved_config_payload(cfg)
    encoded = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    output = _outside_repository(output_path)
    publish_exclusive_file(output, encoded)
    return {
        "path": output,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "resolved_config_sha256": resolved_config_sha256(cfg),
        "scientific_config_sha256": resolved_config_sha256(cfg, scientific=True),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser, parser.parse_args(argv)


def main(argv=None):
    parser, args = parse_args(argv)
    try:
        result = export_resolved_config(args.config, args.output)
    except (EvidenceBundleError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"FULL_PETAL_RESOLVED_CONFIG={result['path']}")
    print(f"FULL_PETAL_RESOLVED_CONFIG_SHA256={result['sha256']}")
    print(f"FULL_PETAL_RESOLVED_IDENTITY={result['resolved_config_sha256']}")
    print(f"FULL_PETAL_SCIENTIFIC_IDENTITY={result['scientific_config_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
