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
                "metric_schema": "standard_ontad_map.v1",
                "instance_metric_schema": "online_instance_metrics.v2",
                "map_unit": "percentage_points",
                "tiou_thresholds": [0.3, 0.4, 0.5, 0.6, 0.7],
                "average_map_pct": map_value,
                "duplicate_rate": 0.30 + identity_shift,
                "fragmentation_rate": 0.20 + identity_shift,
                "prediction_gt_ratio": 0.0 if silent else 1.0,
                "recall_tiou_0p3": 0.0 if silent else 0.60,
                "protocol_violations": 0,
                "gt_supervision_exhaustions": 0,
                "gt_birth_runtime_entry_free_collisions": 0,
                "candidate_arbitration_suppressions": 0,
                "candidate_cancellations": 0,
                "active_abandonments": 0,
                "deferred_birth_due_to_release": 0,
                "committed_predictions": 0 if silent else 1000,
                "expected_updates": 100,
                "successful_updates": 100,
                "scheduler_steps": 100,
                "skipped_updates": 0,
                "provenance": {
                    "code_commit": "a" * 40,
                    "annotation_sha256": "b" * 64,
                    "calibration_protocol_sha256": "c" * 64,
                    "checkpoint_sha256": "d" * 64,
                    "census_sha256": "e" * 64,
                    "config_sha256": "f" * 64,
                    "emission_ledger_sha256": "1" * 64,
                    "reporting_manifest_sha256": "2" * 64,
                },
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


def test_gate_rejects_fraction_map_unit_and_schema_ambiguity():
    fixed = _rows()
    fixed[0]["map_unit"] = "fraction"
    fixed[0]["average_map_pct"] = 0.20

    with pytest.raises(PersistentBindingGateInputError, match="map_unit"):
        evaluate_persistent_binding_gate(fixed, _rows())


def test_gate_rejects_unmatched_common_provenance():
    fixed = _rows()
    rematch = _rows()
    rematch[2]["provenance"]["census_sha256"] = "9" * 64

    with pytest.raises(PersistentBindingGateInputError, match="census_sha256"):
        evaluate_persistent_binding_gate(fixed, rematch)


def test_gate_rejects_skipped_or_misaligned_updates():
    fixed = _rows(identity_shift=-0.10)
    fixed[1]["successful_updates"] = 99
    fixed[1]["skipped_updates"] = 1

    result = evaluate_persistent_binding_gate(fixed, _rows())

    assert result["technical_pass"] is False
    assert any("skipped_updates" in item for item in result["technical_failures"])
    assert any("updates expected" in item for item in result["technical_failures"])
