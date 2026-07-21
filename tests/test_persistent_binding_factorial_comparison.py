import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "tools" / "compare_persistent_binding_factorial.py"
    spec = importlib.util.spec_from_file_location("factorial_compare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _diagnosis(binding, calibration, boundary):
    target = {
        channel: {
            "pairwise_auc": 0.6,
            "threshold_true_positive_rate": 0.3,
            "threshold_false_positive_rate": 0.1,
        }
        for channel in ("birth", "alive", "end")
    }
    payload = {
        "binding_mode": binding,
        "reporting_accessed": False,
        "raw_rgb_authorized": False,
        "target_conditioned_score_distributions": target,
        "lifecycle_counts": {"committed_emissions": 4},
    }
    if calibration:
        payload["lifecycle_calibration_invariance"] = {
            "mode": "monotone_affine",
            "passed": True,
            "scale": [1.0, 1.1, 0.9],
        }
    if boundary:
        payload["boundary_factorization"] = {
            "end_transition_mode": "causal_delta_mlp",
            "endpoint_start_mode": "past_pointer",
            "runtime_state_contains_gt": False,
            "endpoint_pointer": {"decisions": 10, "past_only": True},
        }
    return payload


def _write_run(root, variant, value, eligible=True):
    root.mkdir()
    canonical_variant = {
        "calibration_batched": "calibration",
        "boundary_calibration_batched": "boundary_calibration",
    }.get(variant, variant)
    levels = {
        "reserve": (False, False),
        "calibration": (True, False),
        "boundary": (False, True),
        "boundary_calibration": (True, True),
    }[canonical_variant]
    hashes = {key: f"hash-{key}" for key in (
        "annotation_sha256",
        "calibration_manifest_sha256",
        "fit_manifest_sha256",
        "reporting_manifest_sha256",
        "census_sha256",
    )}
    arms = {}
    for arm, binding in (
        ("fixed", "fixed_birth_slot"),
        ("rematch", "prefix_rematch_active_pool"),
    ):
        arms[arm] = {
            "binding_mode": binding,
            "committed_predictions": value,
            "prediction_gt_ratio": value,
            "recall_tiou_0p3": value,
            "average_map_pct": value,
            "duplicate_rate": value,
            "fragmentation_rate": value,
            "provenance": hashes,
        }
        (root / f"{arm}_score_diagnosis.json").write_text(
            json.dumps(_diagnosis(binding, *levels)),
            encoding="utf-8",
        )
    screen = {
        "technical_pass": eligible,
        "technical_failures": [] if eligible else ["failed"],
        "actual_pair_gpu_hours": 1.0,
        "reporting_accessed": False,
        "raw_rgb_authorized": False,
        **arms,
    }
    (root / "screen_gate.json").write_text(json.dumps(screen), encoding="utf-8")
    (root / "pilot_contract.json").write_text(
        json.dumps(
            {
                "variant": variant,
                "code_commit": f"commit-{variant}",
                "reporting_accessed": False,
                "threshold_search": False,
                "raw_rgb_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    (root / "optimization_activation.json").write_text(
        json.dumps({"variant": variant, "passed": eligible}),
        encoding="utf-8",
    )


def test_factorial_comparison_computes_main_and_interaction_effects(tmp_path):
    module = _module()
    values = {
        "reserve": 1.0,
        "calibration": 3.0,
        "boundary": 4.0,
        "boundary_calibration": 10.0,
    }
    paths = {}
    for variant, value in values.items():
        path = tmp_path / variant
        _write_run(path, variant, value)
        paths[variant] = path

    result = module.compare_factorial(paths)
    effect = result["factorial_effects"]["fixed"]["recall_tiou_0p3"]

    assert effect["calibration_main_effect"] == pytest.approx(4.0)
    assert effect["boundary_main_effect"] == pytest.approx(5.0)
    assert effect["interaction"] == pytest.approx(4.0)
    assert result["selected_variant"] == "boundary_calibration"
    assert result["multi_seed_authorized_next"] is True
    assert result["raw_rgb_authorized_next"] is False


def test_factorial_comparison_rejects_interaction_when_mechanism_fails(tmp_path):
    module = _module()
    paths = {}
    for variant in module.VARIANTS:
        path = tmp_path / variant
        _write_run(path, variant, 1.0)
        paths[variant] = path
    diagnosis_path = paths["boundary_calibration"] / "fixed_score_diagnosis.json"
    diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    diagnosis["lifecycle_calibration_invariance"]["passed"] = False
    diagnosis_path.write_text(json.dumps(diagnosis), encoding="utf-8")

    result = module.compare_factorial(paths)

    assert result["selected_variant"] is None
    assert result["multi_seed_authorized_next"] is False
    assert result["raw_rgb_authorized_next"] is False
    assert result["variants"]["boundary_calibration"]["eligible"] is False


def test_factorial_comparison_accepts_matched_batched_v2_cells(tmp_path):
    module = _module()
    deployments = {
        "reserve": "reserve",
        "calibration": "calibration_batched",
        "boundary": "boundary",
        "boundary_calibration": "boundary_calibration_batched",
    }
    paths = {}
    for canonical, deployment in deployments.items():
        path = tmp_path / deployment
        _write_run(path, deployment, 1.0)
        paths[canonical] = path

    result = module.compare_factorial(paths)

    assert result["design"] == (
        "reserve6_calibration_batched_x_boundary_2x2_v2"
    )
    assert result["selected_variant"] == "boundary_calibration_batched"
    assert result["multi_seed_authorized_next"] is True


def test_factorial_comparison_rejects_mixed_calibration_aggregation(tmp_path):
    module = _module()
    deployments = {
        "reserve": "reserve",
        "calibration": "calibration_batched",
        "boundary": "boundary",
        "boundary_calibration": "boundary_calibration",
    }
    paths = {}
    for canonical, deployment in deployments.items():
        path = tmp_path / f"{canonical}-{deployment}"
        _write_run(path, deployment, 1.0)
        paths[canonical] = path

    with pytest.raises(ValueError, match="same aggregation"):
        module.compare_factorial(paths)
