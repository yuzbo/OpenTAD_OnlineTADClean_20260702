"""Bind the two D1.6 training arms before any endpoint-effect analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate_d16_pair(control: dict, risk: dict) -> dict:
    for arm, receipt in (("control", control), ("risk", risk)):
        if receipt.get("status") != "PASS_TRAIN_MECHANISM_ONLY":
            raise ValueError(f"D1.6 {arm} training receipt is not PASS")
        if receipt.get("arm") != arm:
            raise ValueError(f"D1.6 {arm} receipt identity drifted")
        if receipt.get("test_access") is not False:
            raise ValueError(f"D1.6 {arm} accessed locked test data")
        if receipt.get("official_paper_performance_valid") is not False:
            raise ValueError(f"D1.6 {arm} was incorrectly marked paper-valid")
    shared_fields = (
        "protocol",
        "lane",
        "epochs",
        "seed",
        "model_initialization_sha256",
        "model_state_structure_sha256",
        "source_identity",
    )
    for field in shared_fields:
        if control.get(field) != risk.get(field):
            raise ValueError(f"D1.6 paired field drifted: {field}")
    if control.get("event_d16_variant") != "none":
        raise ValueError("D1.6 control variant drifted")
    if risk.get("event_d16_variant") != "policy_independent":
        raise ValueError("D1.6 risk variant drifted")
    return {
        "status": "PASS_PAIRED_TRAINING_MECHANISM_ONLY",
        "protocol": control["protocol"],
        "arms": ["control", "risk"],
        "single_intended_factor": "owner_risk_contract",
        "model_initialization_sha256": control["model_initialization_sha256"],
        "model_state_structure_sha256": control["model_state_structure_sha256"],
        "source_identity": control["source_identity"],
        "test_access": False,
        "strict_causal_paper_result_valid": False,
        "official_paper_performance_valid": False,
        "endpoint_margin_gate_pending": True,
        "query_internalization_release": False,
        "official_comparison_release": False,
        "locked_test_release": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-root", required=True, type=Path)
    args = parser.parse_args()
    pair_root = args.pair_root.resolve()
    paths = {
        arm: pair_root / arm / "TH" / "mechanism_receipt.json"
        for arm in ("control", "risk")
    }
    receipts = {
        arm: json.loads(path.read_text(encoding="utf-8"))
        for arm, path in paths.items()
    }
    try:
        receipt = validate_d16_pair(receipts["control"], receipts["risk"])
    except ValueError as error:
        raise SystemExit(str(error)) from error
    receipt["arm_receipts"] = {
        arm: {
            "path": str(paths[arm]),
            "checkpoint": receipts[arm]["checkpoint"],
            "owner_risk_contract": receipts[arm]["owner_risk_contract"],
            "mechanism_validation": receipts[arm]["mechanism_validation"],
        }
        for arm in ("control", "risk")
    }
    output_path = pair_root / "paired_training_receipt.json"
    output_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output_path)


if __name__ == "__main__":
    main()
