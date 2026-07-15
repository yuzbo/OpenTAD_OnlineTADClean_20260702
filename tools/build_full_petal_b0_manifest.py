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
    "FULL_PETAL_TRUST_MODEL.md",
    "FULL_PETAL_EXECUTION_GATES.md",
    "configs/causaltad/thumos_pes_q2_base.py",
    "configs/causaltad/thumos_pes_q2_crs_eps_base.py",
    "configs/causaltad/thumos_pes_q2_crs_eps_fixed.py",
    "configs/causaltad/thumos_pes_q2_crs_eps_rematch.py",
    "tools/run_full_petal_b0.py",
    "tools/testing/run_isolated_torch_pytest.py",
    "tools/build_full_petal_b0_manifest.py",
    "tools/build_full_petal_launch_ticket.py",
    "tools/read_full_petal_launch_ticket.py",
    "tools/build_crs_eps_episode_manifest.py",
    "tools/run_crs_eps_gold_audit.py",
    "tools/build_full_petal_run_manifest.py",
    "tools/export_full_petal_resolved_config.py",
    "tools/build_full_petal_manifests.py",
    "tools/run_full_petal_fineaction_evidence.py",
    "tools/train.py",
    "tools/test.py",
    "tools/remote/submit_full_petal_q2_n16r4.sh",
    "tests/full_petal_attestation_fixture.py",
    "opentad/utils/full_petal_b0.py",
    "opentad/utils/full_petal_attestation.py",
    "opentad/utils/full_petal_role_signing.py",
    "opentad/utils/full_petal_fineaction_executor.py",
    "opentad/utils/full_petal_runtime_attestation.py",
    "opentad/utils/evidence_bundle.py",
    "opentad/utils/full_petal_launch.py",
    "opentad/utils/full_petal_training_evidence.py",
    "opentad/utils/full_petal_data_contract.py",
    "opentad/utils/full_petal_identity.py",
    "opentad/utils/fixed_step_profile.py",
    "opentad/utils/crs_eps_sampling.py",
    "opentad/utils/crs_eps_audit.py",
    "opentad/utils/crs_eps_gold_gate.py",
    "opentad/utils/immutable_event_ledger.py",
    "opentad/utils/online_protocol.py",
    "opentad/utils/prefix_instance_schedule.py",
    "opentad/utils/prefix_trajectory_supervision.py",
    "opentad/utils/stream_packets.py",
    "opentad/datasets/__init__.py",
    "opentad/datasets/builder.py",
    "opentad/datasets/streaming_feature.py",
    "opentad/datasets/crs_eps_feature.py",
    "opentad/models/dense_heads/persistent_event_set_head.py",
    "opentad/models/detectors/persistent_trajectory_ontad.py",
    "opentad/evaluations/full_petal_metrics.py",
    "opentad/evaluations/online_budgeted_map.py",
    "opentad/cores/train_engine.py",
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
    args.output.write_bytes(
        (
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    )
    print(f"FULL_PETAL_B0_MANIFEST={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
