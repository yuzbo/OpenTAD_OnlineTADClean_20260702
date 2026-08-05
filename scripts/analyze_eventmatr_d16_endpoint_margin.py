"""Compare frozen D1.6 arms on one common target-visible END risk set.

This is a train-split mechanism analysis, not deployable inference and not a
paper performance evaluation. Ground truth defines the common at-risk set only
after it becomes prefix-visible; both checkpoints receive the same censored
chronological replay and no locked-test artifact is constructed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import THUMOS14Dataset  # noqa: E402
from models import build_model  # noqa: E402
from on_tal_task import make_model_inputs  # noqa: E402
from scripts.eventmatr_d15_contracts import (  # noqa: E402
    validate_parallel_window_causality,
)
from util.utils import memory_initialize, parrallel_collate_fn  # noqa: E402


PROTOCOL = "eventmatr_d16_common_endpoint_margin_v1"
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 52016
EXPECTED_ENDPOINTS = 3001
EXPECTED_PHYSICAL_BATCHES = 3270


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_cluster_bootstrap_median(
    rows: list[dict],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    by_video: dict[str, np.ndarray] = {}
    for video_name in sorted({str(row["video_name"]) for row in rows}):
        by_video[video_name] = np.asarray(
            [
                float(row["margin_difference"])
                for row in rows
                if str(row["video_name"]) == video_name
            ],
            dtype=np.float64,
        )
    videos = sorted(by_video)
    if not videos or any(values.size == 0 for values in by_video.values()):
        raise ValueError("D1.6 endpoint bootstrap has an empty video cluster")
    generator = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected = generator.integers(0, len(videos), size=len(videos))
        values = np.concatenate([by_video[videos[item]] for item in selected])
        samples[index] = np.median(values)
    lower, upper = np.percentile(samples, [2.5, 97.5])
    return {
        "unit": "video",
        "cluster_count": len(videos),
        "replicates": int(replicates),
        "seed": int(seed),
        "statistic": "event_margin_difference_median",
        "lower_95": float(lower),
        "upper_95": float(upper),
    }


def evaluate_margin_gate(control_rows: list[dict], risk_rows: list[dict]) -> dict:
    control = {
        (str(row["video_name"]), int(row["target_event_id"])): row
        for row in control_rows
    }
    risk = {
        (str(row["video_name"]), int(row["target_event_id"])): row
        for row in risk_rows
    }
    if len(control) != EXPECTED_ENDPOINTS or len(risk) != EXPECTED_ENDPOINTS:
        raise ValueError("D1.6 common endpoint coverage did not close at 3001")
    if set(control) != set(risk):
        raise ValueError("D1.6 arms do not cover the same target endpoints")
    paired = []
    for key in sorted(control):
        control_margin = float(control[key]["end_margin"])
        risk_margin = float(risk[key]["end_margin"])
        paired.append(
            {
                "video_name": key[0],
                "target_event_id": key[1],
                "observed_end_frame": float(control[key]["observed_end_frame"]),
                "control_end_margin": control_margin,
                "risk_end_margin": risk_margin,
                "margin_difference": risk_margin - control_margin,
            }
        )
    differences = np.asarray(
        [row["margin_difference"] for row in paired], dtype=np.float64
    )
    median = float(np.median(differences))
    bootstrap = video_cluster_bootstrap_median(paired)
    passed = median > 0.0 and bootstrap["lower_95"] > 0.0
    return {
        "status": "PASS_ENDPOINT_MARGIN_GATE" if passed else "FAIL_ENDPOINT_MARGIN_GATE",
        "endpoint_count": len(paired),
        "event_margin_difference_median": median,
        "event_margin_difference_mean": float(differences.mean()),
        "net_positive_event_count": int((differences > 0.0).sum()),
        "net_zero_event_count": int((differences == 0.0).sum()),
        "net_negative_event_count": int((differences < 0.0).sum()),
        "control_positive_end_margin_count": int(
            sum(float(row["end_margin"]) > 0.0 for row in control_rows)
        ),
        "risk_positive_end_margin_count": int(
            sum(float(row["end_margin"]) > 0.0 for row in risk_rows)
        ),
        "video_cluster_bootstrap": bootstrap,
        "registered_gate": (
            "median risk-minus-control margin strictly positive and video-cluster "
            "bootstrap 95 percent lower bound strictly positive"
        ),
        "passed": passed,
        "paired_rows": paired,
    }


def _checkpoint_rows(
    arm: str,
    checkpoint_path: Path,
    expected_sha256: str,
    base_options: dict,
    dataset: THUMOS14Dataset,
    device: torch.device,
) -> tuple[list[dict], dict]:
    before = _sha256(checkpoint_path)
    if before != expected_sha256:
        raise RuntimeError(f"D1.6 {arm} checkpoint hash drifted before replay")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    options = dict(base_options)
    options.update(
        {
            "event_d16_variant": "policy_independent",
            "study_protocol": "d16_mechanism",
            "training": True,
            "load_model": False,
        }
    )
    args = SimpleNamespace(**options)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    if len(loader) != EXPECTED_PHYSICAL_BATCHES:
        raise RuntimeError(
            f"D1.6 diagnostic requires 3270 physical batches, got {len(loader)}"
        )
    model = torch.nn.DataParallel(build_model(args)).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    model.eval()
    model.module.set_event_risk_diagnostic_mode(True)
    memory_initialize(model, args)

    rows: dict[tuple[str, int], dict] = {}
    physical_batches = 0
    with torch.no_grad():
        for features, targets, infos in loader:
            features, targets, infos = parrallel_collate_fn(
                features, targets, infos, args.p_videos
            )
            validate_parallel_window_causality(
                features,
                video_names=[str(value) for value in infos["video_name"]],
                current_frames=[
                    float(value)
                    for value in infos["current_frame"].reshape(-1).tolist()
                ],
                segment_size=int(args.num_frame),
            )
            targets = {key: value.to(device) for key, value in targets.items()}
            payload = make_model_inputs(
                args, features.to(device, non_blocking=True), infos, targets
            )
            outputs = model(payload, device)
            logits = outputs["event_ragged_state_logits"].detach().cpu()
            state_targets = outputs["event_ragged_state_targets"].detach().cpu()
            sources = list(outputs["event_ragged_sources"])
            group_keys = list(outputs["event_ragged_group_keys"])
            for index in (state_targets == 2).nonzero(as_tuple=False).reshape(-1):
                item = int(index.item())
                if sources[item] != "target_visible":
                    raise RuntimeError("D1.6 observed END escaped target-visible risk")
                key = group_keys[item]
                target_event_id = key[2]
                if target_event_id is None:
                    raise RuntimeError("D1.6 observed END has no target event identity")
                event_key = (str(key[0]), int(target_event_id))
                if event_key in rows:
                    raise RuntimeError(f"D1.6 duplicated observed endpoint {event_key}")
                state_logits = logits[item]
                end_margin = state_logits[2] - state_logits[:2].max()
                rows[event_key] = {
                    "video_name": event_key[0],
                    "target_event_id": event_key[1],
                    "observed_end_frame": float(key[3]),
                    "end_margin": float(end_margin.item()),
                    "cancel_logit": float(state_logits[0].item()),
                    "continue_logit": float(state_logits[1].item()),
                    "end_logit": float(state_logits[2].item()),
                }
            physical_batches += 1
    if physical_batches != EXPECTED_PHYSICAL_BATCHES:
        raise RuntimeError("D1.6 diagnostic physical-batch loop did not close")
    if len(rows) != EXPECTED_ENDPOINTS:
        raise RuntimeError(
            f"D1.6 {arm} endpoint coverage is {len(rows)}, expected 3001"
        )
    after = _sha256(checkpoint_path)
    if after != before:
        raise RuntimeError(f"D1.6 {arm} checkpoint changed during replay")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return [rows[key] for key in sorted(rows)], {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": before,
        "physical_batches": physical_batches,
        "endpoint_count": len(rows),
        "common_policy_independent_risk_replay": True,
        "model_training_mode": False,
        "gradient_enabled": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-receipt", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--diagnostic-source-commit", required=True)
    parser.add_argument("--diagnostic-source-tree", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    cli = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("D1.6 endpoint analysis requires exactly one visible GPU")
    if cli.output_root.exists():
        raise SystemExit(f"D1.6 endpoint output already exists: {cli.output_root}")
    random.seed(52)
    np.random.seed(52)
    torch.manual_seed(52)
    torch.cuda.manual_seed_all(52)

    paired_path = cli.paired_receipt.resolve()
    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    if paired.get("status") != "PASS_PAIRED_TRAINING_MECHANISM_ONLY":
        raise SystemExit("D1.6 paired training receipt is not PASS")
    if paired.get("endpoint_margin_gate_pending") is not True:
        raise SystemExit("D1.6 endpoint analysis was not preregistered as pending")
    arm_receipts = paired["arm_receipts"]
    control_receipt = json.loads(
        Path(arm_receipts["control"]["path"]).read_text(encoding="utf-8")
    )
    base_options = json.loads(
        Path(control_receipt["options"]["path"]).read_text(encoding="utf-8")
    )
    if Path(base_options["video_feature_all_test"]).exists():
        raise SystemExit("locked-test sentinel unexpectedly exists")
    if base_options.get("event_birth_logit_threshold") is not None:
        raise SystemExit("D1.6 endpoint analysis forbids a birth threshold")
    if base_options.get("event_end_logit_threshold") is not None:
        raise SystemExit("D1.6 endpoint analysis forbids an END threshold")

    args = SimpleNamespace(**base_options)
    dataset = THUMOS14Dataset(args, subset="train")
    device = torch.device("cuda")
    all_rows = {}
    replay_receipts = {}
    for arm in ("control", "risk"):
        checkpoint = arm_receipts[arm]["checkpoint"]
        rows, replay = _checkpoint_rows(
            arm,
            Path(checkpoint["path"]).resolve(),
            str(checkpoint["sha256"]),
            base_options,
            dataset,
            device,
        )
        all_rows[arm] = rows
        replay_receipts[arm] = replay

    gate = evaluate_margin_gate(all_rows["control"], all_rows["risk"])
    paired_rows = gate.pop("paired_rows")
    cli.output_root.mkdir(parents=True, exist_ok=False)
    trace_path = cli.output_root / "endpoint_margin_pairs.jsonl"
    trace_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in paired_rows),
        encoding="utf-8",
    )
    receipt = {
        "status": gate["status"],
        "protocol": PROTOCOL,
        "scientific_scope": "official train-split common-risk mechanism analysis",
        "primary_outcome": "per-event owner END margin difference",
        "ground_truth_role": (
            "post-visible definition of one common censored at-risk set; not runtime input"
        ),
        "seed": 52,
        "test_access": False,
        "threshold_search": False,
        "checkpoint_updated": False,
        "optimizer_constructed": False,
        "strict_causal_paper_result_valid": False,
        "official_paper_performance_valid": False,
        "train_prefix_detection_metrics_used": False,
        "query_internalization_release": bool(gate["passed"]),
        "official_comparison_release": False,
        "locked_test_release": False,
        "paired_training_receipt": {
            "path": str(paired_path),
            "sha256": _sha256(paired_path),
        },
        "diagnostic_source_identity": {
            "commit": cli.diagnostic_source_commit,
            "tree": cli.diagnostic_source_tree,
            "manifest_sha256": cli.manifest_sha256,
        },
        "replays": replay_receipts,
        "gate": gate,
        "paired_trace": {
            "path": str(trace_path),
            "rows": len(paired_rows),
            "sha256": _sha256(trace_path),
        },
    }
    receipt_path = cli.output_root / "endpoint_margin_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
