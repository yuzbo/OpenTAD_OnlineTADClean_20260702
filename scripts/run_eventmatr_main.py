"""Run the native MATR entry point with an auditable EventMATR output tag."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main as matr_main  # noqa: E402
from util.config import make_parser  # noqa: E402


ARM_MODES = {
    "b0o0": ("matr_delayed", "fresh_rematch"),
    "b1o0": ("instant_transition", "fresh_rematch"),
    "b0o1": ("matr_delayed", "sticky_owner"),
    "b1o1": ("instant_transition", "sticky_owner"),
}

D1_LANES = {
    "R": ("r", "fresh_rematch", "b1o0"),
    "T": ("t", "sticky_owner", "b1o1"),
    "H": ("h", "fresh_rematch", "b1o0"),
    "TH": ("th", "sticky_owner", "b1o1"),
}

OFFICIAL_SETTING = {
    "feat_dim": 4096,
    "num_frame": 64,
    "num_queries": 10,
    "max_memory_len": 7,
    "memory_sampler": "gap2",
    "batch": 64,
    "epochs": 100,
    "min_lr": 1e-8,
    "max_lr": 1e-5,
    "weight_decay": 1e-4,
    "lr_Tup": 3,
    "lr_Tcycle": 10,
    "lr_gamma": 0.9,
    "test_freq": 1,
    "save_freq": 1,
    "random_seed": 52,
    "cls_threshold": 0.1,
    "nms_threshold": 0.3,
}


def _validate(args) -> str:
    lane = os.environ.get("MATR_LANE", "")
    d1_preexperiment = args.study_protocol == "d1_preexperiment"
    d11_mechanism = args.study_protocol == "d11_mechanism"
    d1_train_only = d1_preexperiment or d11_mechanism
    if d1_train_only:
        if args.mode != "train":
            raise RuntimeError(f"{args.study_protocol} is a train-only protocol")
        if d1_preexperiment and args.epochs not in {5, 10, 20}:
            raise RuntimeError("D1 pilot epochs must be one of 5, 10, or 20")
        if d11_mechanism and args.epochs != 1:
            raise RuntimeError("D1.1 mechanism protocol requires exactly one epoch")
        if args.train_eval_step != args.epochs:
            raise RuntimeError(
                "D1 train-only protocols evaluate the train prefix only at terminal epoch"
            )
        if d11_mechanism and lane != "TH":
            raise RuntimeError("D1.1 mechanism protocol is restricted to lane TH")
        if d11_mechanism and args.load_model:
            raise RuntimeError(
                "D1.1 mechanism protocol must start fresh and forbids checkpoint resume"
            )
        if lane == "N":
            if args.model_variant != "native_matr":
                raise RuntimeError("D1 lane N requires --model_variant native_matr")
            if args.event_arm is not None:
                raise RuntimeError("D1 lane N must not provide --event_arm")
            if args.event_lifecycle_version != "v1_dense":
                raise RuntimeError("D1 lane N must preserve the native lifecycle")
        elif lane in D1_LANES:
            d1_lane, ownership_mode, event_arm = D1_LANES[lane]
            expected = {
                "model_variant": "eventmatr",
                "event_lifecycle_version": "d1_censored",
                "event_d1_lane": d1_lane,
                "birth_mode": "instant_transition",
                "ownership_mode": ownership_mode,
                "event_arm": event_arm,
                "event_teacher_forcing_ratio": 0.5,
            }
            mismatched_lane = {
                name: {"expected": value, "actual": getattr(args, name)}
                for name, value in expected.items()
                if getattr(args, name) != value
            }
            if mismatched_lane:
                raise RuntimeError(
                    f"D1 lane {lane} drift is forbidden:\n"
                    + json.dumps(mismatched_lane, indent=2, sort_keys=True)
                )
            if args.event_birth_logit_threshold is not None:
                raise RuntimeError("D1 pilots forbid a fixed birth threshold")
            if args.event_end_logit_threshold is not None:
                raise RuntimeError("D1 pilots forbid a fixed end threshold")
        else:
            raise RuntimeError(
                f"D1 MATR_LANE must be one of {['N', *D1_LANES]}, got {lane!r}"
            )
    elif lane == "native_matr":
        if args.model_variant != "native_matr":
            raise RuntimeError("native_matr lane requires --model_variant native_matr")
        if args.event_arm is not None:
            raise RuntimeError("native_matr lane must not provide --event_arm")
    elif lane in ARM_MODES:
        if args.model_variant != "eventmatr":
            raise RuntimeError(f"{lane} is eventized and requires --model_variant eventmatr")
        expected_modes = ARM_MODES[lane]
        actual_modes = (args.birth_mode, args.ownership_mode)
        if actual_modes != expected_modes:
            raise RuntimeError(
                f"arm/mode mismatch for {lane}: expected {expected_modes}, got {actual_modes}"
            )
        if args.event_arm != lane:
            raise RuntimeError(f"{lane} requires explicit --event_arm {lane}")
        if args.event_birth_logit_threshold is not None:
            raise RuntimeError("formal EventMATR forbids a fixed birth threshold; use state argmax")
        if args.event_end_logit_threshold is not None:
            raise RuntimeError("formal EventMATR forbids a fixed end threshold; use state argmax")
    else:
        allowed = ["native_matr", *ARM_MODES]
        raise RuntimeError(f"MATR_LANE must be one of {allowed}, got {lane!r}")

    if args.study_protocol == "matched_study" and args.mode != "train":
        raise RuntimeError("matched_study is a train-only terminal-epoch protocol")
    if args.study_protocol == "locked_test" and args.mode != "eval":
        raise RuntimeError("locked_test is an eval-only protocol")

    official_setting = dict(OFFICIAL_SETTING)
    if d1_train_only:
        official_setting.pop("epochs")
    mismatches = {
        name: {"expected": expected, "actual": getattr(args, name)}
        for name, expected in official_setting.items()
        if getattr(args, name) != expected
    }
    if not args.rgb or not args.flow or not args.use_focal or not args.use_flag:
        mismatches["official_boolean_flags"] = {
            "expected": "rgb, flow, use_focal, use_flag all true",
            "actual": {
                "rgb": args.rgb,
                "flow": args.flow,
                "use_focal": args.use_focal,
                "use_flag": args.use_flag,
            },
        }
    if mismatches:
        raise RuntimeError(
            "official MATR setting drift is forbidden:\n"
            + json.dumps(mismatches, indent=2, sort_keys=True)
        )
    return lane


def main() -> None:
    args = make_parser()
    lane = _validate(args)

    run_tag = os.environ.get("MATR_RUN_TAG", "").strip()
    if not run_tag:
        raise RuntimeError("MATR_RUN_TAG is required")
    output_root_text = os.environ.get("MATR_OUTPUT_ROOT", "").strip()
    if not output_root_text:
        raise RuntimeError("MATR_OUTPUT_ROOT is required")
    output_root = Path(output_root_text).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    original_strftime = matr_main.time.strftime

    def tagged_strftime(fmt, time_tuple):
        stamp = original_strftime(fmt, time_tuple)
        return f"{run_tag}__{lane}__{stamp}"

    matr_main.time.strftime = tagged_strftime
    os.chdir(output_root)
    matr_main.main(args)


if __name__ == "__main__":
    main()
