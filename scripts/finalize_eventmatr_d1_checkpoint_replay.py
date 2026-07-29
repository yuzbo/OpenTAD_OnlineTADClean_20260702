"""Validate and combine the four EventMATR D1 checkpoint replays."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


LANES = ("R", "T", "H", "TH")
REQUIRED_STAGE_KEYS = {
    "real_prefixes",
    "candidate_queries",
    "candidate_state_argmax",
    "candidate_start_prefixes",
    "start_rising_births",
    "gt_birth_prefixes",
    "gt_birth_prefixes_with_candidate_start",
    "gt_birth_prefixes_with_predicted_birth",
    "owner_rows",
    "owner_state_argmax",
    "gt_end_prefixes",
    "gt_end_prefixes_with_owner",
    "gt_end_prefixes_with_owner_end",
    "cancellations",
    "ends",
    "emissions",
    "reacquisitions",
    "runtime_capacity_exhaustions",
    "observed_eos",
    "active_at_observed_eos",
    "ledger_rows",
    "native_gate_predicted_on",
    "native_gate_gt_on",
    "native_gate_true_positive",
    "native_gate_false_positive",
    "native_gate_false_negative",
}


def _finite(value) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _load_identity(path: Path, expected: dict[str, str]) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "PASS" or value.get("clean") is not True:
        raise ValueError(f"source identity is not PASS/clean: {path}")
    for key, target in expected.items():
        if value.get(key) != target:
            raise ValueError(f"source identity {key} mismatch: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-root", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.replay_root.resolve()
    expected_source = {
        "commit": args.source_commit,
        "tree": args.source_tree,
        "manifest_sha256": args.manifest_sha256,
    }
    summaries = {}
    for lane in LANES:
        lane_root = root / lane
        _load_identity(lane_root / "source_identity_start.json", expected_source)
        _load_identity(lane_root / "source_identity_final.json", expected_source)
        path = lane_root / f"{lane}_replay_summary.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "status": "PASS",
            "protocol": "eventmatr_d1_strict_causal_checkpoint_replay_v1",
            "lane": lane,
            "complete_replay": True,
            "strict_causal_paper_result_valid": False,
            "test_access": False,
            "checkpoint_updated": False,
            "threshold_search": False,
            "ground_truth_visible_to_model": False,
        }
        for key, target in expected.items():
            if value.get(key) != target:
                raise ValueError(
                    f"{lane} summary {key} mismatch: {value.get(key)!r} != {target!r}"
                )
        if value.get("source_identity") != expected_source:
            raise ValueError(f"{lane} replay source identity mismatch")
        stage = value.get("stage_counts", {})
        missing = sorted(REQUIRED_STAGE_KEYS.difference(stage))
        if missing:
            raise ValueError(f"{lane} stage counts missing {missing}")
        if stage["runtime_capacity_exhaustions"] != 0:
            raise ValueError(f"{lane} exhausted runtime capacity")
        if stage["ledger_rows"] != stage["emissions"]:
            raise ValueError(f"{lane} ledger/emission count mismatch")
        if stage["observed_eos"] != 200:
            raise ValueError(f"{lane} did not replay all 200 observed EOS markers")
        if not _finite(value):
            raise ValueError(f"{lane} summary contains a non-finite value")
        summaries[lane] = value

    table = {
        lane: {
            "root_cause": summaries[lane]["root_cause"]["label"],
            "births": summaries[lane]["stage_counts"]["start_rising_births"],
            "owner_rows": summaries[lane]["stage_counts"]["owner_rows"],
            "owner_background": summaries[lane]["stage_counts"][
                "owner_state_argmax"
            ][0],
            "cancellations": summaries[lane]["stage_counts"]["cancellations"],
            "ends": summaries[lane]["stage_counts"]["ends"],
            "emissions": summaries[lane]["stage_counts"]["emissions"],
            "active_at_observed_eos": summaries[lane]["stage_counts"][
                "active_at_observed_eos"
            ],
            "proposal_count": summaries[lane]["scientific_metrics"][
                "proposal_count"
            ],
            "video_coverage": summaries[lane]["scientific_metrics"][
                "video_coverage"
            ],
        }
        for lane in LANES
    }
    payload = {
        "status": "PASS",
        "protocol": "eventmatr_d1_strict_causal_checkpoint_replay_v1",
        "test_access": False,
        "checkpoint_updated": False,
        "threshold_search": False,
        "strict_causal_paper_result_valid": False,
        "source_identity": expected_source,
        "lanes": table,
        "interpretation": (
            "This receipt localizes lifecycle liveness and calibration. "
            "It is not a paper result and does not release longer training."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
