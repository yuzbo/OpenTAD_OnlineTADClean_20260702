"""Compare the three frozen feature-level optimization pilots."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
VARIANT_ORDER = ("sw", "margin", "transport")
ARM_BINDINGS = {
    "fixed": "fixed_birth_slot",
    "rematch": "prefix_rematch_active_pool",
}
CHANNELS = ("birth", "alive", "end")
RANKING_RULE = (
    "maximize_worst_arm_recall_tiou_0p3",
    "maximize_worst_arm_and_channel_pairwise_auc",
    "maximize_worst_arm_birth_tpr_at_frozen_0p5",
    "maximize_worst_arm_birth_mean_score_gap",
    "minimize_worst_arm_birth_fpr_at_frozen_0p5",
    "minimize_actual_pair_gpu_hours",
    "stable_variant_order_sw_margin_transport",
)


def _load(path):
    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit():
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise ValueError("optimization comparison requires a clean checkout")
    return _validate_commit(commit, "analysis commit")


def _finite(value, label):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _validate_commit(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError(f"{label} must be a full lowercase Git commit")
    return value


def _extract_variant(variant, payloads, expected_commit):
    contract = payloads["contract"]
    activation = payloads["activation"]
    screen = payloads["screen"]
    resource = payloads["resource"]
    diagnoses = {
        "fixed": payloads["fixed_diagnosis"],
        "rematch": payloads["rematch_diagnosis"],
    }
    failures = []

    if contract.get("variant") != variant:
        failures.append(f"pilot contract variant={contract.get('variant')!r}")
    if contract.get("code_commit") != expected_commit:
        failures.append("pilot contract code commit mismatch")
    if contract.get("seed") != 705 or contract.get("epochs") != 1:
        failures.append("pilot contract changed seed or epoch count")
    for field in (
        "fit_only",
        "calibration_diagnosis_only",
    ):
        if contract.get(field) is not True:
            failures.append(f"pilot contract {field} is not true")
    for field in (
        "reporting_accessed",
        "threshold_search",
        "raw_rgb_authorized",
    ):
        if contract.get(field) is not False:
            failures.append(f"pilot contract {field} is not false")

    if activation.get("variant") != variant:
        failures.append("activation report variant mismatch")
    if activation.get("passed") is not True:
        failures.append("optimization loss activation gate rejected")

    for field in ("screen_pass", "technical_pass", "budget_pass"):
        if screen.get(field) is not True:
            failures.append(f"screen {field} is not true")
    if resource.get("schema_version") != "persistent_binding_resource.v1":
        failures.append("unexpected pair resource schema")
    if resource.get("scope") != "pair":
        failures.append("resource report is not pair scoped")
    if resource.get("code_commit") != expected_commit:
        failures.append("resource report code commit mismatch")
    pair_gpu_hours = _finite(
        resource.get("allocated_gpu_hours"),
        f"{variant} allocated GPU hours",
    )
    if pair_gpu_hours <= 0:
        failures.append("allocated pair GPU hours are not positive")

    arm_rows = {}
    lifecycle_aucs = []
    recalls = []
    birth_tprs = []
    birth_fprs = []
    birth_gaps = []
    for arm, binding_mode in ARM_BINDINGS.items():
        row = screen.get(arm)
        if not isinstance(row, dict):
            raise ValueError(f"{variant} screen lacks {arm} row")
        provenance = row.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError(f"{variant} {arm} lacks provenance")
        if provenance.get("code_commit") != expected_commit:
            failures.append(f"{arm} screen code commit mismatch")
        if row.get("binding_mode") != binding_mode:
            failures.append(f"{arm} screen binding mode mismatch")
        for field in (
            "reporting_accessed",
            "effectiveness_claim_authorized",
            "raw_rgb_authorized",
        ):
            if row.get(field) is not False:
                failures.append(f"{arm} screen changed {field}")

        diagnosis = diagnoses[arm]
        if diagnosis.get("passed") is not True:
            failures.append(f"{arm} score diagnosis did not pass")
        if diagnosis.get("checkpoint_training_commit") != expected_commit:
            failures.append(f"{arm} diagnosis training commit mismatch")
        if diagnosis.get("binding_mode") != binding_mode:
            failures.append(f"{arm} diagnosis binding mode mismatch")
        for field in (
            "reporting_accessed",
            "effectiveness_claim_authorized",
            "raw_rgb_authorized",
        ):
            if diagnosis.get(field) is not False:
                failures.append(f"{arm} diagnosis changed {field}")

        target_conditioned = diagnosis.get(
            "target_conditioned_score_distributions"
        )
        if not isinstance(target_conditioned, dict):
            raise ValueError(f"{variant} {arm} lacks target-conditioned scores")
        channel_rows = {}
        for channel in CHANNELS:
            channel_row = target_conditioned.get(channel)
            if not isinstance(channel_row, dict):
                raise ValueError(f"{variant} {arm} lacks {channel} diagnosis")
            auc = _finite(
                channel_row.get("pairwise_auc"),
                f"{variant} {arm} {channel} AUC",
            )
            if not 0 <= auc <= 1:
                raise ValueError(f"{variant} {arm} {channel} AUC is outside [0,1]")
            lifecycle_aucs.append(auc)
            channel_rows[channel] = {
                "pairwise_auc": auc,
                "mean_score_gap": _finite(
                    channel_row.get("mean_score_gap"),
                    f"{variant} {arm} {channel} mean gap",
                ),
                "threshold_true_positive_rate": _finite(
                    channel_row.get("threshold_true_positive_rate"),
                    f"{variant} {arm} {channel} TPR",
                ),
                "threshold_false_positive_rate": _finite(
                    channel_row.get("threshold_false_positive_rate"),
                    f"{variant} {arm} {channel} FPR",
                ),
            }

        recall = _finite(
            row.get("recall_tiou_0p3"),
            f"{variant} {arm} Recall@0.3",
        )
        recalls.append(recall)
        birth_tprs.append(
            channel_rows["birth"]["threshold_true_positive_rate"]
        )
        birth_fprs.append(
            channel_rows["birth"]["threshold_false_positive_rate"]
        )
        birth_gaps.append(channel_rows["birth"]["mean_score_gap"])
        arm_rows[arm] = {
            "binding_mode": binding_mode,
            "committed_predictions": int(row["committed_predictions"]),
            "prediction_gt_ratio": _finite(
                row.get("prediction_gt_ratio"),
                f"{variant} {arm} prediction/GT ratio",
            ),
            "recall_tiou_0p3": recall,
            "average_map_pct": _finite(
                row.get("average_map_pct"),
                f"{variant} {arm} average mAP",
            ),
            "identity_error": _finite(
                row.get("identity_error"),
                f"{variant} {arm} identity error",
            ),
            "lifecycle_discrimination": channel_rows,
        }

    metrics = {
        "worst_arm_recall_tiou_0p3": min(recalls),
        "worst_arm_and_channel_pairwise_auc": min(lifecycle_aucs),
        "worst_arm_birth_tpr_at_frozen_0p5": min(birth_tprs),
        "worst_arm_birth_mean_score_gap": min(birth_gaps),
        "worst_arm_birth_fpr_at_frozen_0p5": max(birth_fprs),
        "actual_pair_gpu_hours": pair_gpu_hours,
    }
    return {
        "variant": variant,
        "eligible": not failures,
        "eligibility_failures": failures,
        "metrics": metrics,
        "arms": arm_rows,
    }


def _ranking_key(row):
    metrics = row["metrics"]
    return (
        -metrics["worst_arm_recall_tiou_0p3"],
        -metrics["worst_arm_and_channel_pairwise_auc"],
        -metrics["worst_arm_birth_tpr_at_frozen_0p5"],
        -metrics["worst_arm_birth_mean_score_gap"],
        metrics["worst_arm_birth_fpr_at_frozen_0p5"],
        metrics["actual_pair_gpu_hours"],
        VARIANT_ORDER.index(row["variant"]),
    )


def compare_pilots(all_payloads, expected_commit):
    _validate_commit(expected_commit, "expected_commit")
    if set(all_payloads) != set(VARIANT_ORDER):
        raise ValueError("comparison requires exactly sw, margin, and transport")
    rows = [
        _extract_variant(
            variant,
            all_payloads[variant],
            expected_commit,
        )
        for variant in VARIANT_ORDER
    ]
    eligible = sorted(
        (row for row in rows if row["eligible"]),
        key=_ranking_key,
    )
    return {
        "schema_version": "persistent_binding_optimization_comparison.v1",
        "purpose": "feature_level_model_route_selection_only",
        "effectiveness_claim_authorized": False,
        "raw_rgb_authorized": False,
        "reporting_accessed": False,
        "seed": 705,
        "epochs": 1,
        "expected_training_commit": expected_commit,
        "ranking_frozen_before_results": True,
        "eligibility_rule": (
            "activation, technical screen, budget, provenance, and "
            "calibration-only diagnosis must all pass"
        ),
        "ranking_rule": list(RANKING_RULE),
        "variants": rows,
        "eligible_variants_in_rank_order": [
            row["variant"] for row in eligible
        ],
        "selected_variant": eligible[0]["variant"] if eligible else None,
        "next_stage_authorized": bool(eligible),
        "raw_rgb_authorized_next": False,
    }


def _run_payload(run_dir):
    root = Path(run_dir)
    return {
        "contract": _load(root / "pilot_contract.json"),
        "activation": _load(root / "optimization_activation.json"),
        "screen": _load(root / "screen_gate.json"),
        "resource": _load(root / "pair_resource_report.json"),
        "fixed_diagnosis": _load(root / "fixed_score_diagnosis.json"),
        "rematch_diagnosis": _load(root / "rematch_score_diagnosis.json"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sw-run", required=True)
    parser.add_argument("--margin-run", required=True)
    parser.add_argument("--transport-run", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = compare_pilots(
        {
            "sw": _run_payload(args.sw_run),
            "margin": _run_payload(args.margin_run),
            "transport": _run_payload(args.transport_run),
        },
        args.expected_commit,
    )
    payload["analysis_commit"] = _git_commit()
    payload["analysis_script_sha256"] = _sha256(__file__)
    payload["run_directories"] = {
        "sw": str(Path(args.sw_run).resolve()),
        "margin": str(Path(args.margin_run).resolve()),
        "transport": str(Path(args.transport_run).resolve()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
