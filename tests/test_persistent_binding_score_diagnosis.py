import math
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from diagnose_persistent_binding_scores import (  # noqa: E402
    _normalized_state_dict,
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
