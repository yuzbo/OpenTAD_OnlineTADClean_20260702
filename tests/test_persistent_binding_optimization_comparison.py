import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40


def _module():
    path = ROOT / "tools/compare_persistent_binding_optimization_pilots.py"
    spec = importlib.util.spec_from_file_location("optimization_comparison", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _diagnosis(binding_mode, auc, birth_tpr, birth_fpr, birth_gap):
    channels = {}
    for channel in ("birth", "alive", "end"):
        channels[channel] = {
            "pairwise_auc": auc,
            "mean_score_gap": birth_gap if channel == "birth" else 0.1,
            "threshold_true_positive_rate": (
                birth_tpr if channel == "birth" else 0.5
            ),
            "threshold_false_positive_rate": (
                birth_fpr if channel == "birth" else 0.1
            ),
        }
    return {
        "passed": True,
        "checkpoint_training_commit": COMMIT,
        "binding_mode": binding_mode,
        "reporting_accessed": False,
        "effectiveness_claim_authorized": False,
        "raw_rgb_authorized": False,
        "target_conditioned_score_distributions": channels,
    }


def _payload(
    variant,
    *,
    recall,
    auc=0.7,
    birth_tpr=0.6,
    birth_fpr=0.1,
    birth_gap=0.2,
    gpu_hours=1.0,
    eligible=True,
):
    rows = {}
    diagnoses = {}
    for arm, binding in (
        ("fixed", "fixed_birth_slot"),
        ("rematch", "prefix_rematch_active_pool"),
    ):
        rows[arm] = {
            "binding_mode": binding,
            "reporting_accessed": False,
            "effectiveness_claim_authorized": False,
            "raw_rgb_authorized": False,
            "committed_predictions": 100,
            "prediction_gt_ratio": 1.0,
            "recall_tiou_0p3": recall,
            "average_map_pct": 10.0,
            "identity_error": 0.1,
            "provenance": {"code_commit": COMMIT},
        }
        diagnoses[arm] = _diagnosis(
            binding,
            auc,
            birth_tpr,
            birth_fpr,
            birth_gap,
        )
    return {
        "contract": {
            "variant": variant,
            "code_commit": COMMIT,
            "seed": 705,
            "epochs": 1,
            "fit_only": True,
            "calibration_diagnosis_only": True,
            "reporting_accessed": False,
            "threshold_search": False,
            "raw_rgb_authorized": False,
        },
        "activation": {"variant": variant, "passed": eligible},
        "screen": {
            "screen_pass": eligible,
            "technical_pass": eligible,
            "budget_pass": True,
            **rows,
        },
        "resource": {
            "schema_version": "persistent_binding_resource.v1",
            "scope": "pair",
            "code_commit": COMMIT,
            "allocated_gpu_hours": gpu_hours,
        },
        "fixed_diagnosis": diagnoses["fixed"],
        "rematch_diagnosis": diagnoses["rematch"],
    }


def test_comparison_uses_frozen_worst_arm_recall_before_auc():
    evaluator = _module()
    result = evaluator.compare_pilots(
        {
            "sw": _payload("sw", recall=0.4, auc=0.99),
            "margin": _payload("margin", recall=0.6, auc=0.70),
            "transport": _payload("transport", recall=0.5, auc=0.80),
        },
        COMMIT,
    )

    assert result["selected_variant"] == "margin"
    assert result["eligible_variants_in_rank_order"] == [
        "margin",
        "transport",
        "sw",
    ]
    assert result["raw_rgb_authorized_next"] is False
    assert result["ranking_frozen_before_results"] is True


def test_comparison_excludes_gate_reject_even_with_better_metrics():
    evaluator = _module()
    result = evaluator.compare_pilots(
        {
            "sw": _payload("sw", recall=0.4),
            "margin": _payload("margin", recall=0.9, eligible=False),
            "transport": _payload("transport", recall=0.5),
        },
        COMMIT,
    )

    assert result["selected_variant"] == "transport"
    margin = next(
        row for row in result["variants"] if row["variant"] == "margin"
    )
    assert margin["eligible"] is False
    assert any("activation" in item for item in margin["eligibility_failures"])


def test_comparison_authorizes_no_next_stage_when_all_reject():
    evaluator = _module()
    result = evaluator.compare_pilots(
        {
            variant: _payload(variant, recall=0.9, eligible=False)
            for variant in ("sw", "margin", "transport")
        },
        COMMIT,
    )

    assert result["selected_variant"] is None
    assert result["next_stage_authorized"] is False
