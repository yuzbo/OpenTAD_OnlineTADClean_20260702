from copy import deepcopy

import pytest

from opentad.utils.crs_eps_gold_gate import (
    CrsEpsGoldGateError,
    MARGIN_SCHEMA_VERSION,
    evaluate_gold_audit,
)


def _margins():
    return {
        "schema_version": MARGIN_SCHEMA_VERSION,
        "status": "PREREGISTERED_BEFORE_Q2_EFFECTIVENESS",
        "commit_sha": "a" * 40,
        "episode_manifest_sha256": "b" * 64,
        "selection_sha256": "c" * 64,
        "min_gradient_cosine": 0.90,
        "min_gradient_sign_agreement": 0.90,
        "min_runtime_continuous_cosine": 0.90,
        "max_relative_loss_error": 0.10,
        "require_runtime_discrete_equal": True,
        "max_mean_dynamic_replay_ratio": 0.80,
        "max_video_start_fallback_fraction": 0.25,
        "min_dynamic_minus_fixed_gradient_cosine": 0.02,
        "min_dynamic_minus_reset_gradient_cosine": 0.05,
    }


def _trace(mode):
    values = {
        "dynamic_birth": (0.98, 0.98, 0.97, 0.6, False, 1.02),
        "fixed_192": (0.94, 0.95, 0.94, 0.5, False, 1.04),
        "reset": (0.88, 0.90, 0.85, 0.2, False, 1.10),
    }
    grad, sign, runtime, ratio, fallback, loss = values[mode]
    gold_tokens = 100
    candidate_tokens = int(gold_tokens * ratio)
    candidate_start = 0 if fallback else gold_tokens - candidate_tokens
    return {
        "comparison_type": "replay_fidelity",
        "left": {
            "losses": {"cost": 1.0},
            "audit": {"replay_range": [0, gold_tokens]},
        },
        "right": {
            "losses": {"cost": loss},
            "audit": {"replay_range": [candidate_start, gold_tokens]},
        },
        "gradient_comparison": {"cosine": grad, "sign_agreement": sign},
        "runtime_continuous_comparison": {"cosine": runtime},
        "runtime_discrete_equal": True,
    }


def _rows():
    return [
        {
            "video_id": "video",
            "draw_index": 0,
            "mode": mode,
            "trace": _trace(mode),
        }
        for mode in ("dynamic_birth", "fixed_192", "reset")
    ]


def test_gold_gate_passes_only_a_complete_outcome_blind_four_arm_audit():
    result = evaluate_gold_audit(_rows(), _margins())

    assert result["status"] == "PASS"
    assert result["sample_count"] == 1
    assert result["trace_count"] == 3
    assert result["metrics"]["mean_dynamic_replay_ratio"] == pytest.approx(0.6)
    assert result["violations"] == []
    assert len(result["gate_sha256"]) == 64


def test_gold_gate_kills_dynamic_state_or_gradient_fidelity_failure():
    rows = _rows()
    rows[0]["trace"]["gradient_comparison"]["cosine"] = 0.5
    rows[0]["trace"]["runtime_discrete_equal"] = False

    result = evaluate_gold_audit(rows, _margins())

    assert result["status"] == "KILL"
    assert "video:draw=0:gradient_cosine" in result["violations"]
    assert "video:draw=0:runtime_discrete" in result["violations"]


def test_gold_gate_rejects_missing_arm_and_posthoc_margin_status():
    with pytest.raises(CrsEpsGoldGateError, match="exact four-arm"):
        evaluate_gold_audit(_rows()[:-1], _margins())

    margins = deepcopy(_margins())
    margins["status"] = "CHOSEN_AFTER_RESULTS"
    with pytest.raises(CrsEpsGoldGateError, match="outcome-blind"):
        evaluate_gold_audit(_rows(), margins)
