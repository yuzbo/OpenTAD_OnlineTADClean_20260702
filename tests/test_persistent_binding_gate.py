import pytest

from opentad.evaluations.persistent_binding_gate import (
    PersistentBindingGateInputError,
    evaluate_persistent_binding_gate,
)


def _rows(*, identity_shift=0.0, map_value=20.0, silent=False):
    rows = []
    for seed in (705, 706, 707):
        rows.append(
            {
                "seed": seed,
                "average_map": map_value,
                "duplicate_rate": 0.30 + identity_shift,
                "fragmentation_rate": 0.20 + identity_shift,
                "prediction_gt_ratio": 0.0 if silent else 1.0,
                "recall_tiou_0p3": 0.0 if silent else 0.60,
                "protocol_violations": 0,
                "dropped_gt_birth_targets": 0,
                "runtime_capacity_exhaustions": 0,
                "committed_predictions": 0 if silent else 1000,
            }
        )
    return rows


def test_gate_passes_only_when_technical_and_scientific_conditions_pass():
    fixed = _rows(identity_shift=-0.10, map_value=19.8)
    rematch = _rows(identity_shift=0.0, map_value=20.0)

    result = evaluate_persistent_binding_gate(fixed, rematch)

    assert result["technical_pass"] is True
    assert result["scientific_pass"] is True
    assert result["overall_pass"] is True
    assert result["identity_error_relative_reduction"] == pytest.approx(0.4)
    assert result["improved_seed_count"] == 3
    assert result["map_delta_points"] == pytest.approx(-0.2)


def test_gate_rejects_silent_output_even_if_identity_errors_look_better():
    fixed = _rows(identity_shift=-0.20, silent=True)
    rematch = _rows()

    result = evaluate_persistent_binding_gate(fixed, rematch)

    assert result["technical_pass"] is False
    assert result["overall_pass"] is False
    assert any("no committed predictions" in item for item in result["technical_failures"])
    assert any("prediction_gt_ratio" in item for item in result["technical_failures"])
    assert any("recall_tiou_0p3" in item for item in result["technical_failures"])


def test_gate_rejects_fixed_when_map_drop_exceeds_half_a_point():
    fixed = _rows(identity_shift=-0.10, map_value=19.4)
    rematch = _rows(identity_shift=0.0, map_value=20.0)

    result = evaluate_persistent_binding_gate(fixed, rematch)

    assert result["technical_pass"] is True
    assert result["scientific_pass"] is False
    assert result["overall_pass"] is False
    assert any("mAP delta" in item for item in result["scientific_failures"])


def test_gate_requires_exact_matched_seed_set():
    fixed = _rows()
    fixed.pop()

    with pytest.raises(PersistentBindingGateInputError, match="exactly seeds"):
        evaluate_persistent_binding_gate(fixed, _rows())
