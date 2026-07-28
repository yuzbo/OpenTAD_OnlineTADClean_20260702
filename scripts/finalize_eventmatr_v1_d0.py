"""Fail-closed finalizer for the four-lane EventMATR-v1 D0 audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


LANES = ("b0o0", "b1o0", "b0o1", "b1o1")
PROTOCOL_ID = "eventmatr_v1_d0_checkpoint_replay_v1"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("missing D0 artifact: {}".format(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("D0 artifact is not a JSON object: {}".format(path))
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(
            "{} mismatch: {!r} != {!r}".format(label, actual, expected)
        )


def _require_finite_mapping(mapping: Any, label: str) -> None:
    if not isinstance(mapping, dict) or not mapping:
        raise RuntimeError("{} must be a non-empty object".format(label))
    for key, value in mapping.items():
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise RuntimeError(
                "{}.{} must be finite, got {!r}".format(label, key, value)
            )


def finalize(
    root: Path,
    manifest_path: Path,
    *,
    expected_source_run: Path,
    expected_diagnostic_commit: str,
    expected_diagnostic_tree: str,
    expected_manifest_sha256: str,
    expected_training_commit: str,
    expected_training_tree: str,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    expected_source_run = expected_source_run.expanduser().resolve()
    if (root / "d0_pair_completion.json").exists():
        raise RuntimeError("refusing to overwrite an existing D0 completion receipt")

    manifest = _read_json(manifest_path)
    _require_equal(manifest.get("protocol_id"), PROTOCOL_ID, "manifest protocol_id")
    _require_equal(
        _sha256(manifest_path),
        expected_manifest_sha256,
        "manifest sha256",
    )
    _require_equal(
        manifest.get("source_training_commit"),
        expected_training_commit,
        "manifest source training commit",
    )
    _require_equal(
        manifest.get("source_training_tree"),
        expected_training_tree,
        "manifest source training tree",
    )
    _require_equal(tuple(manifest.get("lanes", ())), LANES, "manifest lanes")

    launch_identity = _read_json(root / "source_identity.json")
    _require_equal(launch_identity.get("status"), "PASS", "launch identity status")
    _require_equal(launch_identity.get("clean"), True, "launch identity clean")
    _require_equal(
        launch_identity.get("commit"),
        expected_diagnostic_commit,
        "launch diagnostic commit",
    )
    _require_equal(
        launch_identity.get("tree"),
        expected_diagnostic_tree,
        "launch diagnostic tree",
    )
    _require_equal(
        launch_identity.get("manifest_sha256"),
        expected_manifest_sha256,
        "launch manifest sha256",
    )

    launch = _read_json(root / "d0_launch.json")
    _require_equal(launch.get("status"), "SUBMITTED", "launch status")
    _require_equal(launch.get("protocol_id"), PROTOCOL_ID, "launch protocol_id")
    _require_equal(launch.get("test_access"), False, "launch test_access")
    _require_equal(tuple(launch.get("array_lanes", ())), LANES, "launch lanes")
    _require_equal(
        Path(launch.get("output_root", "")).expanduser().resolve(),
        root,
        "launch output root",
    )
    _require_equal(
        launch.get("diagnostic_source"),
        {
            "commit": expected_diagnostic_commit,
            "tree": expected_diagnostic_tree,
            "manifest_sha256": expected_manifest_sha256,
        },
        "launch diagnostic source",
    )
    _require_equal(
        launch.get("source_training"),
        {
            "run": str(expected_source_run),
            "commit": expected_training_commit,
            "tree": expected_training_tree,
        },
        "launch source training",
    )

    receipt_paths = sorted(root.glob("*/d0_audit_receipt.json"))
    expected_receipt_paths = sorted(
        root / lane / "d0_audit_receipt.json" for lane in LANES
    )
    _require_equal(receipt_paths, expected_receipt_paths, "D0 receipt set")

    lanes: dict[str, Any] = {}
    shared_dataset_identity = None
    for lane in LANES:
        worker_identity = _read_json(root / "source_identity_lanes" / f"{lane}.json")
        for key, expected in (
            ("status", "PASS"),
            ("clean", True),
            ("commit", expected_diagnostic_commit),
            ("tree", expected_diagnostic_tree),
            ("manifest_sha256", expected_manifest_sha256),
        ):
            _require_equal(worker_identity.get(key), expected, f"{lane} identity {key}")

        path = root / lane / "d0_audit_receipt.json"
        payload = _read_json(path)
        for key, expected in (
            ("status", "PASS"),
            ("protocol_id", PROTOCOL_ID),
            ("test_access", False),
            ("checkpoint_updated", False),
            ("strict_causal_paper_result_valid", False),
        ):
            _require_equal(payload.get(key), expected, f"{lane} {key}")
        _require_equal(
            payload.get("diagnostic_source"),
            {
                "commit": expected_diagnostic_commit,
                "tree": expected_diagnostic_tree,
            },
            f"{lane} diagnostic source",
        )

        source = payload.get("source_training")
        if not isinstance(source, dict):
            raise RuntimeError("{} source_training is missing".format(lane))
        _require_equal(
            Path(source.get("run", "")).expanduser().resolve(),
            expected_source_run,
            f"{lane} source run",
        )
        _require_equal(source.get("lane"), lane, f"{lane} source lane")
        _require_equal(source.get("epoch"), 100, f"{lane} source epoch")
        identity = source.get("identity")
        if not isinstance(identity, dict):
            raise RuntimeError("{} source identity is missing".format(lane))
        for key, expected in (
            ("commit", expected_training_commit),
            ("tree", expected_training_tree),
            ("clean", True),
        ):
            _require_equal(identity.get(key), expected, f"{lane} training {key}")
        lane_dir = Path(source.get("lane_dir", "")).expanduser().resolve()
        if "__{}__".format(lane) not in lane_dir.name:
            raise RuntimeError("{} lane directory is not lane-bound".format(lane))
        _require_equal(
            Path(source.get("opts_json", "")).expanduser().resolve(),
            lane_dir / "opts.json",
            f"{lane} opts path",
        )
        _require_equal(
            Path(source.get("checkpoint", "")).expanduser().resolve(),
            lane_dir / "terminal_epoch100.pth",
            f"{lane} checkpoint path",
        )
        if not isinstance(source.get("checkpoint_sha256"), str) or len(
            source["checkpoint_sha256"]
        ) != 64:
            raise RuntimeError("{} checkpoint sha256 is malformed".format(lane))
        if not isinstance(source.get("opts_sha256"), str) or len(
            source["opts_sha256"]
        ) != 64:
            raise RuntimeError("{} opts sha256 is malformed".format(lane))

        dataset = payload.get("dataset")
        if not isinstance(dataset, dict):
            raise RuntimeError("{} dataset receipt is missing".format(lane))
        _require_equal(dataset.get("subset"), "train", f"{lane} dataset subset")
        if "LOCKED_TEST_NOT_MOUNTED" not in Path(
            dataset.get("locked_test_sentinel", "")
        ).name:
            raise RuntimeError("{} lacks locked-test absent sentinel".format(lane))
        chronology = dataset.get("chronology")
        if not isinstance(chronology, dict):
            raise RuntimeError("{} chronology receipt is missing".format(lane))
        _require_equal(
            chronology.get("video_transitions"),
            chronology.get("video_count"),
            f"{lane} video transitions",
        )
        _require_equal(chronology.get("cross_video_batches"), 0, f"{lane} cross batches")
        _require_equal(
            chronology.get("aligned_single_video_batches"),
            chronology.get("batch_boundary_count"),
            f"{lane} aligned batches",
        )
        dataset_identity = {
            "annotation": dataset.get("annotation"),
            "train_feature": dataset.get("train_feature"),
            "locked_test_sentinel": dataset.get("locked_test_sentinel"),
            "chronology": chronology,
        }
        if shared_dataset_identity is None:
            shared_dataset_identity = dataset_identity
        else:
            _require_equal(
                dataset_identity,
                shared_dataset_identity,
                f"{lane} shared dataset identity",
            )

        gradient = payload.get("gradient_audit")
        if not isinstance(gradient, dict):
            raise RuntimeError("{} gradient audit is missing".format(lane))
        if int(gradient.get("finite_gradient_tensors", 0)) <= 0:
            raise RuntimeError("{} recorded no finite gradient tensors".format(lane))
        gradient_norms = gradient.get("parameter_gradient_norms")
        if not isinstance(gradient_norms, dict) or not gradient_norms:
            raise RuntimeError("{} gradient norms are missing".format(lane))

        replay = payload.get("eval_full_prefix_replay")
        if not isinstance(replay, dict):
            raise RuntimeError("{} replay receipt is missing".format(lane))
        if replay.get("operational_verdict") not in {
            "learned_runtime_zero_births",
            "nonzero_births_but_zero_final_emissions",
            "runtime_emits_final_intervals",
        }:
            raise RuntimeError("{} operational verdict is invalid".format(lane))
        _require_finite_mapping(
            replay.get("train_replay_map_percent"), f"{lane} replay mAP"
        )

        lanes[lane] = {
            "checkpoint_sha256": source["checkpoint_sha256"],
            "operational_verdict": replay["operational_verdict"],
            "runtime_counts": replay["runtime_counts"],
            "ledger_audit": replay["ledger_audit"],
            "train_replay_map_percent": replay["train_replay_map_percent"],
            "gradient_norms": gradient_norms,
        }

    summary = {
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "scientific_scope": "diagnostic train replay only; not paper performance",
        "test_access": False,
        "checkpoint_updated": False,
        "diagnostic_source": {
            "commit": expected_diagnostic_commit,
            "tree": expected_diagnostic_tree,
            "manifest_sha256": expected_manifest_sha256,
        },
        "source_training": {
            "run": str(expected_source_run),
            "commit": expected_training_commit,
            "tree": expected_training_tree,
        },
        "dataset": shared_dataset_identity,
        "lanes": lanes,
    }
    output = root / "d0_pair_completion.json"
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-source-run", required=True, type=Path)
    parser.add_argument("--expected-diagnostic-commit", required=True)
    parser.add_argument("--expected-diagnostic-tree", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-training-commit", required=True)
    parser.add_argument("--expected-training-tree", required=True)
    args = parser.parse_args()
    try:
        summary = finalize(
            args.root,
            args.manifest,
            expected_source_run=args.expected_source_run,
            expected_diagnostic_commit=args.expected_diagnostic_commit,
            expected_diagnostic_tree=args.expected_diagnostic_tree,
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_training_commit=args.expected_training_commit,
            expected_training_tree=args.expected_training_tree,
        )
    except Exception as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
