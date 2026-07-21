import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from diagnose_persistent_binding_scores import (  # noqa: E402
    _normalized_state_dict,
    append_target_conditioned_scores,
    pairwise_auc,
    summarize_binary_discrimination,
    summarize_probabilities,
)


def test_probability_summary_exposes_threshold_margin_and_crossings():
    summary = summarize_probabilities([0.1, 0.4, 0.5, 0.9], threshold=0.5)

    assert summary["count"] == 4
    assert summary["mean"] == pytest.approx(0.475)
    assert summary["p50"] == pytest.approx(0.45)
    assert summary["max"] == pytest.approx(0.9)
    assert summary["threshold_crossings"] == 2
    assert summary["threshold_crossing_rate"] == pytest.approx(0.5)
    assert summary["max_minus_threshold"] == pytest.approx(0.4)


def test_probability_summary_rejects_empty_score_channels():
    with pytest.raises(ValueError, match="cannot be empty"):
        summarize_probabilities([], threshold=0.5)


def test_binary_discrimination_exposes_separation_without_selecting_threshold():
    summary = summarize_binary_discrimination(
        [0.7, 0.9],
        [0.1, 0.3],
        threshold=0.5,
    )

    assert summary["pairwise_auc"] == pytest.approx(1.0)
    assert summary["mean_score_gap"] == pytest.approx(0.6)
    assert summary["median_score_gap"] == pytest.approx(0.6)
    assert summary["threshold_true_positive_rate"] == pytest.approx(1.0)
    assert summary["threshold_false_positive_rate"] == pytest.approx(0.0)
    assert "selected_threshold" not in summary


def test_binary_discrimination_counts_ties_as_half_a_pairwise_win():
    summary = summarize_binary_discrimination(
        [0.5],
        [0.5],
        threshold=0.5,
    )

    assert summary["pairwise_auc"] == pytest.approx(0.5)


def test_pairwise_auc_is_invariant_to_positive_affine_calibration():
    positive = [-0.2, 0.4, 1.7]
    negative = [-2.0, -0.4, 0.1, 0.9]
    scale = 2.5
    bias = -0.7

    raw_auc = pairwise_auc(positive, negative)
    calibrated_auc = pairwise_auc(
        [scale * value + bias for value in positive],
        [scale * value + bias for value in negative],
    )

    assert calibrated_auc == raw_auc


def test_target_conditioning_follows_training_masks_and_bindings():
    store = {
        channel: {"positive": [], "negative": []}
        for channel in ("birth", "alive", "end")
    }
    transition = SimpleNamespace(
        birth_mask=(True, True, False, True),
        birth_assignments=(SimpleNamespace(slot_id=1),),
        at_risk_mask=(True, False, True, False),
        endpoint_slots=(2,),
        audit=SimpleNamespace(occupied_slots_for_supervision=(1, 2)),
    )

    append_target_conditioned_scores(
        store,
        {
            "birth": [0.1, 0.8, 0.9, 0.2],
            "alive": [0.1, 0.8, 0.7, 0.2],
            "end": [0.2, 0.9, 0.8, 0.1],
        },
        transition,
    )

    assert store["birth"] == {
        "positive": [0.8],
        "negative": [0.1, 0.2],
    }
    assert store["alive"] == {
        "positive": [0.8, 0.7],
        "negative": [0.1, 0.2],
    }
    assert store["end"] == {
        "positive": [0.8],
        "negative": [0.2],
    }


def test_checkpoint_state_normalization_removes_one_ddp_prefix():
    weight = torch.tensor([1.0])

    normalized = _normalized_state_dict(
        {"state_dict": {"module.head.birth_head.bias": weight}}
    )

    assert list(normalized) == ["head.birth_head.bias"]
    assert normalized["head.birth_head.bias"] is weight


def test_checkpoint_state_normalization_rejects_mixed_namespaces():
    with pytest.raises(ValueError, match="mixes wrapped and unwrapped"):
        _normalized_state_dict(
            {
                "state_dict": {
                    "module.head.birth_head.bias": torch.tensor([1.0]),
                    "head.alive_head.bias": torch.tensor([2.0]),
                }
            }
        )


def test_weighted_bce_stationary_logit_is_raw_logit_plus_log_weight():
    prior = 0.0055006867682647655
    positive_weight = 13.446021031128877
    raw_logit = math.log(prior / (1.0 - prior))
    stationary_logit = raw_logit + math.log(positive_weight)
    stationary_probability = 1.0 / (1.0 + math.exp(-stationary_logit))

    assert raw_logit == pytest.approx(-5.1973664563)
    assert stationary_logit == pytest.approx(-2.5986832282)
    assert stationary_probability == pytest.approx(0.0692232136)
