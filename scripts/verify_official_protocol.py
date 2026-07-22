"""Fail closed if native MATR or EventMATR BxO drifts from the matched study."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from util.config import make_parser  # noqa: E402


MANIFEST = ROOT / "experiment_configs" / "eventmatr_bxo_official_thumos14.json"
EXPECTED_COMMON = {
    "epochs": 100,
    "batch": 64,
    "random_seed": 52,
    "rgb": True,
    "flow": True,
    "feat_dim": 4096,
    "num_frame": 64,
    "num_queries": 10,
    "max_memory_len": 7,
    "memory_sampler": "gap2",
    "optimizer": "Adam",
    "min_lr": 1e-8,
    "max_lr": 1e-5,
    "weight_decay": 1e-4,
    "lr_Tup": 3,
    "lr_Tcycle": 10,
    "use_focal": True,
    "cls_threshold": 0.1,
    "nms_threshold": 0.3,
}
EXPECTED_LANES = {
    "native_matr": {
        "model_variant": "native_matr",
        "role": "exact official MATR architecture, head, labels, and losses under terminal-epoch training",
    },
    "b0o0": {
        "model_variant": "eventmatr",
        "event_arm": "b0o0",
        "birth_mode": "matr_delayed",
        "ownership_mode": "fresh_rematch",
    },
    "b1o0": {
        "model_variant": "eventmatr",
        "event_arm": "b1o0",
        "birth_mode": "instant_transition",
        "ownership_mode": "fresh_rematch",
    },
    "b0o1": {
        "model_variant": "eventmatr",
        "event_arm": "b0o1",
        "birth_mode": "matr_delayed",
        "ownership_mode": "sticky_owner",
    },
    "b1o1": {
        "model_variant": "eventmatr",
        "event_arm": "b1o1",
        "birth_mode": "instant_transition",
        "ownership_mode": "sticky_owner",
    },
}


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    common = manifest["common"]
    drift = {
        key: {"expected": expected, "actual": common.get(key)}
        for key, expected in EXPECTED_COMMON.items()
        if common.get(key) != expected
    }
    if manifest.get("lanes") != EXPECTED_LANES:
        drift["lanes"] = {"expected": EXPECTED_LANES, "actual": manifest.get("lanes")}
    expected_training = {
        "source": "official_complete_validation_train",
        "epochs": 100,
        "checkpoint_policy": "terminal_epoch",
        "checkpoint_epoch": 100,
        "checkpoint_filename": "terminal_epoch100.pth",
        "test_during_training": False,
        "test_loader_during_training": False,
        "test_driven_checkpoint_selection": False,
        "locked_test_once_per_lane": True,
    }
    training = manifest.get("training_protocol", {})
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            drift[f"training_protocol.{key}"] = {
                "expected": expected,
                "actual": training.get(key),
            }

    saved_argv = sys.argv
    try:
        sys.argv = ["verify_official_protocol.py"]
        defaults = make_parser()
    finally:
        sys.argv = saved_argv
    parser_expected = {
        key: expected
        for key, expected in EXPECTED_COMMON.items()
        if key not in {"rgb", "flow", "optimizer", "use_focal"}
    }
    parser_drift = {
        key: {"expected": expected, "actual": getattr(defaults, key)}
        for key, expected in parser_expected.items()
        if getattr(defaults, key) != expected
    }
    if defaults.birth_mode != "matr_delayed" or defaults.ownership_mode != "fresh_rematch":
        parser_drift["native_modes"] = {
            "expected": ("matr_delayed", "fresh_rematch"),
            "actual": (defaults.birth_mode, defaults.ownership_mode),
        }
    if defaults.model_variant != "native_matr":
        parser_drift["model_variant"] = {
            "expected": "native_matr",
            "actual": defaults.model_variant,
        }
    if defaults.event_birth_logit_threshold is not None:
        parser_drift["event_birth_logit_threshold"] = {
            "expected": None,
            "actual": defaults.event_birth_logit_threshold,
        }
    if defaults.event_end_logit_threshold is not None:
        parser_drift["event_end_logit_threshold"] = {
            "expected": None,
            "actual": defaults.event_end_logit_threshold,
        }
    if parser_drift:
        drift["parser_defaults"] = parser_drift
    if drift:
        raise SystemExit("official protocol drift:\n" + json.dumps(drift, indent=2, sort_keys=True))
    print("official MATR-parent EventMATR BxO protocol: PASS")


if __name__ == "__main__":
    main()
