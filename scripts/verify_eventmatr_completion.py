"""Verify five terminal-epoch lanes without constructing a THUMOS test loader."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


LANES = {
    "native_matr": {
        "model_variant": "native_matr",
        "event_arm": None,
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

SOURCE_FIELDS = ("commit", "tree", "manifest_sha256")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise SystemExit(f"{label} is not PASS: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--source-identity", type=Path)
    parser.add_argument("--smoke-receipt", type=Path)
    parser.add_argument("--lane-identity-dir", type=Path)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    source_identity_path = (
        args.source_identity.resolve()
        if args.source_identity is not None
        else output_root / "source_identity.json"
    )
    smoke_path = (
        args.smoke_receipt.resolve()
        if args.smoke_receipt is not None
        else output_root / "eventmatr_real_smoke.json"
    )
    lane_identity_dir = (
        args.lane_identity_dir.resolve()
        if args.lane_identity_dir is not None
        else output_root / "source_identity_lanes"
    )
    result_root = output_root / "result"
    locked_test_sentinel = (output_root / "LOCKED_TEST_NOT_MOUNTED.pickle").resolve()
    if locked_test_sentinel.exists():
        raise SystemExit("locked test sentinel unexpectedly exists during matched training")

    source_identity = load_json(source_identity_path, "source_identity.json")
    if source_identity.get("clean") is not True:
        raise SystemExit("source_identity.json did not certify a clean source")
    exact_identity = {}
    for field in SOURCE_FIELDS:
        value = source_identity.get(field)
        if not isinstance(value, str) or not value:
            raise SystemExit(f"source_identity.json lacks {field}")
        exact_identity[field] = value

    smoke = load_json(smoke_path, "eventmatr_real_smoke.json")
    if smoke.get("test_access") is not False:
        raise SystemExit("eventmatr_real_smoke.json accessed the locked test")
    smoke_identity = smoke.get("source_identity")
    if smoke_identity != exact_identity:
        raise SystemExit("eventmatr_real_smoke.json source identity mismatch")
    smoke_lanes = smoke.get("lanes")
    if not isinstance(smoke_lanes, list) or {
        row.get("lane") for row in smoke_lanes if isinstance(row, dict)
    } != set(LANES):
        raise SystemExit("eventmatr_real_smoke.json does not cover all five lanes")
    smoke_sha256 = sha256(smoke_path)

    lane_identities = {}
    for lane in LANES:
        lane_path = lane_identity_dir / f"{lane}.json"
        lane_identity = load_json(lane_path, f"source_identity_lanes/{lane}.json")
        if lane_identity.get("clean") is not True:
            raise SystemExit(f"source_identity_lanes/{lane}.json is not clean")
        for field, expected_value in exact_identity.items():
            if lane_identity.get(field) != expected_value:
                raise SystemExit(
                    f"source_identity_lanes/{lane}.json {field} mismatch"
                )
        smoke_binding = lane_identity.get("smoke", {})
        if (
            smoke_binding.get("status") != "PASS"
            or smoke_binding.get("test_access") is not False
            or smoke_binding.get("sha256") != smoke_sha256
        ):
            raise SystemExit(f"source_identity_lanes/{lane}.json smoke mismatch")
        lane_identities[lane] = {
            "receipt": str(lane_path.resolve()),
            "receipt_sha256": sha256(lane_path),
        }

    receipt = {
        "status": "PASS",
        "run_tag": args.run_tag,
        "training_source": "official_complete_validation_train",
        "checkpoint_policy": "terminal_epoch100",
        "test_access_during_training": False,
        "source_identity": exact_identity,
        "source_identity_receipt": {
            "path": str(source_identity_path),
            "sha256": sha256(source_identity_path),
        },
        "smoke": {
            "path": str(smoke_path),
            "sha256": smoke_sha256,
            "status": "PASS",
            "test_access": False,
        },
        "source_identity_lanes": lane_identities,
        "lanes": {},
    }
    for lane, lane_expected in LANES.items():
        matches = sorted(result_root.glob(f"*<{args.run_tag}__{lane}__*>"))
        if len(matches) != 1:
            raise SystemExit(f"expected one result directory for {lane}, found {len(matches)}")
        result_dir = matches[0]
        opts_path = result_dir / "opts.json"
        opts = json.loads(opts_path.read_text(encoding="utf-8"))
        expected = {
            "epochs": 100,
            "batch": 64,
            "random_seed": 52,
            "study_protocol": "matched_study",
            **lane_expected,
        }
        drift = {
            key: {"expected": value, "actual": opts.get(key)}
            for key, value in expected.items()
            if opts.get(key) != value
        }
        if opts.get("event_birth_logit_threshold") is not None:
            drift["event_birth_logit_threshold"] = {
                "expected": None,
                "actual": opts.get("event_birth_logit_threshold"),
            }
        if opts.get("event_end_logit_threshold") is not None:
            drift["event_end_logit_threshold"] = {
                "expected": None,
                "actual": opts.get("event_end_logit_threshold"),
            }
        if Path(opts["video_feature_all_test"]).resolve() != locked_test_sentinel:
            drift["video_feature_all_test"] = {
                "expected": str(locked_test_sentinel),
                "actual": opts.get("video_feature_all_test"),
            }
        if drift:
            raise SystemExit(f"{lane} setting drift: {json.dumps(drift, sort_keys=True)}")

        terminal = result_dir / "terminal_epoch100.pth"
        checkpoints = sorted(result_dir.glob("*.pth"))
        if checkpoints != [terminal]:
            raise SystemExit(
                f"{lane} must contain only terminal_epoch100.pth, found {[p.name for p in checkpoints]}"
            )
        receipt["lanes"][lane] = {
            "result_dir": str(result_dir.resolve()),
            "opts_sha256": sha256(opts_path),
            "checkpoint": str(terminal.resolve()),
            "checkpoint_sha256": sha256(terminal),
        }

    output = output_root / "eventmatr_bxo_completion.json"
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
