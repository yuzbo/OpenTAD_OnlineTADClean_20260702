#!/usr/bin/env python3
"""Build the exhaustive, source-hash-locked Full PETAL B0 manifest."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.utils.full_petal_b0 import (  # noqa: E402
    B0_MANIFEST_SCHEMA,
    sha256_file,
    test_functions,
)


RUNNER_SOURCES = (
    "tools/run_full_petal_b0.py",
    "tools/testing/run_isolated_torch_pytest.py",
    "tools/build_full_petal_b0_manifest.py",
    "tools/build_full_petal_run_manifest.py",
    "opentad/utils/full_petal_b0.py",
    "opentad/utils/full_petal_attestation.py",
    "opentad/utils/full_petal_launch.py",
    "tools/check_full_petal_results.py",
)


def build_manifest(root=ROOT):
    root = Path(root).resolve()
    test_files = []
    for path in sorted((root / "tests").glob("test_*.py")):
        relative = path.relative_to(root).as_posix()
        functions = test_functions(path)
        if not functions:
            raise ValueError(f"test module has no declared test functions: {relative}")
        test_files.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "test_functions": functions,
            }
        )
    targets = [item["path"] for item in test_files]
    canonical_argv = [
        "$TORCH_PYTHON",
        "tools/testing/run_isolated_torch_pytest.py",
        "--repo-root",
        "$REPO_ROOT",
        "--",
        *targets,
        "-q",
        "-p",
        "no:cacheprovider",
        "--junitxml",
        "$JUNIT",
    ]
    return {
        "schema_version": B0_MANIFEST_SCHEMA,
        "suite_order": ["all_contracts"],
        "suites": [
            {
                "name": "all_contracts",
                "runner": "isolated_torch",
                "canonical_argv": canonical_argv,
                "test_files": test_files,
            }
        ],
        "runner_sources": [
            {"path": relative, "sha256": sha256_file(root / relative)}
            for relative in RUNNER_SOURCES
        ],
        "audit_checks": [
            "python_syntax",
            "git_diff_check",
            "repository_clean_after",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "tools" / "testing" / "full_petal_b0_manifest.json",
    )
    args = parser.parse_args(argv)
    payload = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"FULL_PETAL_B0_MANIFEST={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
