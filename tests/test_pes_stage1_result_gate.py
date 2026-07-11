import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "check_pes_stage1_results.py"
SPEC = importlib.util.spec_from_file_location("check_pes_stage1_results", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(variant, seed, score, duplicate, fragmentation, gpu_hours=0.5):
    return {
        "variant": variant,
        "seed": seed,
        "average_mOnlineAP": score,
        "duplicate_rate": duplicate,
        "fragmentation_rate": fragmentation,
        "gpu_hours": gpu_hours,
        "protocol_violations": 0,
        "slot_exhaustion": 0,
    }


def _report(persistent_score=0.14, persistent_error=0.20):
    rows = []
    for seed in (705, 706, 707):
        rows.extend(
            [
                _row("fresh", seed, 0.10, 0.30, 0.30),
                _row("trackformer", seed, 0.11, 0.28, 0.28),
                _row(
                    "persistent",
                    seed,
                    persistent_score,
                    persistent_error,
                    persistent_error,
                ),
            ]
        )
    return {"hard_budget_gpu_hours": 10.0, "runs": rows}


def test_gate_passes_only_as_a_mechanism_pilot_not_a_final_claim():
    verdict = MODULE.evaluate_gate(_report())

    assert verdict["status"] == "PASS_MECHANISM_GATE"
    assert verdict["raw_video_training_allowed"] is False
    assert verdict["needs_five_seed_confirmation"] is True
    assert set(verdict["comparisons"]) == {"fresh", "trackformer"}
    assert set(verdict["comparisons"]["fresh"]["error_reduction_fraction"]) == {
        "duplicate_rate",
        "fragmentation_rate",
    }


def test_gate_kills_equivalence_to_temporal_trackformer():
    verdict = MODULE.evaluate_gate(_report(persistent_score=0.111, persistent_error=0.27))

    assert verdict["status"] == "KILL_OR_REVISE"
    assert verdict["comparisons"]["trackformer"]["passes"] is False


def test_gate_rejects_unmatched_seeds_and_budget_overrun():
    report = _report()
    report["runs"].pop()
    with pytest.raises(ValueError, match="matched seed sets"):
        MODULE.evaluate_gate(report)

    report = _report()
    for row in report["runs"]:
        row["gpu_hours"] = 2.0
    verdict = MODULE.evaluate_gate(report)
    assert verdict["status"] == "INVALID"
    assert "GPU-hour budget" in verdict["invalid_reasons"][0]
