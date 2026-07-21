"""Verify that each optimization pilot exercises only its intended loss."""

import argparse
import json
import math
from pathlib import Path


AUDIT_SCHEMA = "persistent_binding_training_audit.v1"
AUXILIARY_LOSSES = (
    "birth_margin_loss",
    "alive_margin_loss",
    "end_margin_loss",
    "causal_transport_loss",
    "birth_calibration_loss",
    "alive_calibration_loss",
    "end_calibration_loss",
)
ACTIVE_LOSSES = {
    "sw": (),
    "margin": ("birth_margin_loss",),
    "transport": ("causal_transport_loss",),
    "lifecycle": (
        "birth_margin_loss",
        "alive_margin_loss",
        "end_margin_loss",
    ),
    "reserve": (
        "birth_margin_loss",
        "alive_margin_loss",
        "end_margin_loss",
    ),
    "calibration": (
        "birth_margin_loss",
        "alive_margin_loss",
        "end_margin_loss",
        "birth_calibration_loss",
        "alive_calibration_loss",
        "end_calibration_loss",
    ),
}


def _load(path):
    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"training audit must be an object: {path}")
    return payload


def _normalize_arm(payload, arm):
    if payload.get("schema_version") != AUDIT_SCHEMA:
        raise ValueError(f"{arm} has an unexpected training-audit schema")
    epochs = payload.get("epochs")
    if not isinstance(epochs, list) or len(epochs) != 1:
        raise ValueError(f"{arm} activation pilot must contain exactly one epoch")
    epoch = epochs[0]
    means = epoch.get("mean_losses")
    counts = epoch.get("loss_nonzero_updates")
    if not isinstance(means, dict) or not isinstance(counts, dict):
        raise ValueError(f"{arm} audit lacks structured loss activation evidence")
    successful_updates = int(epoch.get("successful_updates", -1))
    if successful_updates <= 0:
        raise ValueError(f"{arm} has no successful optimizer updates")

    normalized = {}
    for key in AUXILIARY_LOSSES:
        if key not in means or key not in counts:
            if key.endswith("_calibration_loss"):
                mean = 0.0
                count = 0
            else:
                raise ValueError(f"{arm} audit lacks {key}")
        else:
            mean = float(means[key])
            count = int(counts[key])
        if not math.isfinite(mean) or mean < 0:
            raise ValueError(f"{arm} {key} mean must be finite and non-negative")
        if count < 0 or count > successful_updates:
            raise ValueError(f"{arm} {key} activation count is invalid")
        normalized[key] = {
            "mean": mean,
            "nonzero_updates": count,
        }
    return {
        "binding_mode": payload.get("binding_mode"),
        "successful_updates": successful_updates,
        "losses": normalized,
    }


def evaluate_activation(variant, fixed_payload, rematch_payload):
    if variant not in ACTIVE_LOSSES:
        raise ValueError(
            "variant must be sw, margin, transport, lifecycle, reserve, or calibration"
        )
    arms = {
        "fixed": _normalize_arm(fixed_payload, "fixed"),
        "rematch": _normalize_arm(rematch_payload, "rematch"),
    }
    expected_bindings = {
        "fixed": "fixed_birth_slot",
        "rematch": "prefix_rematch_active_pool",
    }
    failures = []
    active_losses = ACTIVE_LOSSES[variant]
    for arm, row in arms.items():
        if row["binding_mode"] != expected_bindings[arm]:
            failures.append(
                f"{arm}: binding_mode={row['binding_mode']!r}"
            )
        for key, evidence in row["losses"].items():
            should_be_active = key in active_losses
            observed_active = (
                evidence["mean"] > 0
                and evidence["nonzero_updates"] > 0
            )
            observed_dormant = (
                evidence["mean"] == 0
                and evidence["nonzero_updates"] == 0
            )
            if should_be_active and not observed_active:
                failures.append(
                    f"{arm}: intended {key} did not activate"
                )
            if not should_be_active and not observed_dormant:
                failures.append(
                    f"{arm}: unintended {key} activated"
                )
    return {
        "schema_version": "persistent_binding_optimization_activation.v1",
        "variant": variant,
        "active_loss": (
            active_losses[0] if len(active_losses) == 1 else None
        ),
        "active_losses": list(active_losses),
        "passed": not failures,
        "failures": failures,
        "arms": arms,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        required=True,
        choices=tuple(ACTIVE_LOSSES),
    )
    parser.add_argument("--fixed-audit", required=True)
    parser.add_argument("--rematch-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = evaluate_activation(
        args.variant,
        _load(args.fixed_audit),
        _load(args.rematch_audit),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    raise SystemExit(0 if payload["passed"] else 1)


if __name__ == "__main__":
    main()
