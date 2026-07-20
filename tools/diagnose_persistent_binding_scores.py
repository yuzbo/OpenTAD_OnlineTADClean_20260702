"""Audit calibration-only lifecycle score distributions for a trained screen arm."""

import argparse
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from mmengine.config import Config  # noqa: E402

from opentad.datasets import build_dataloader, build_dataset  # noqa: E402
from opentad.models import build_detector  # noqa: E402
from opentad.utils import configure_strict_determinism, set_seed  # noqa: E402
from opentad.utils.device import move_data_to_device  # noqa: E402


CHANNELS = {
    "birth": ("birth_logits", "birth_threshold", "birth_prior_probability"),
    "alive": ("alive_logits", "alive_threshold", "alive_prior_probability"),
    "end": ("end_hazard_logits", "end_threshold", "end_prior_probability"),
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _ids(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return {line.strip() for line in file if line.strip()}


def _git_commit(repo):
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise ValueError("score diagnosis requires a clean checkout")
    return commit


def _percentile(values, percent):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * float(percent) / 100.0
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_probabilities(values, threshold):
    values = [float(value) for value in values]
    if not values:
        raise ValueError("a score channel cannot be empty")
    threshold = float(threshold)
    crossings = sum(value >= threshold for value in values)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "p01": _percentile(values, 1),
        "p10": _percentile(values, 10),
        "p25": _percentile(values, 25),
        "p50": _percentile(values, 50),
        "p75": _percentile(values, 75),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": max(values),
        "threshold": threshold,
        "threshold_crossings": crossings,
        "threshold_crossing_rate": crossings / len(values),
        "max_minus_threshold": max(values) - threshold,
    }


def _normalized_state_dict(checkpoint):
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("checkpoint has no non-empty state_dict")
    keys = tuple(state_dict)
    if all(key.startswith("module.") for key in keys):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    if any(key.startswith("module.") for key in keys):
        raise ValueError("checkpoint mixes wrapped and unwrapped parameter names")
    return state_dict


def _prior_audit(model, cfg):
    report = {}
    for channel, (_, _, prior_name) in CHANNELS.items():
        prior = float(getattr(cfg.model.head, prior_name))
        positive_weight = float(getattr(cfg.model, f"{channel}_positive_weight"))
        raw_logit = math.log(prior / (1.0 - prior))
        weighted_logit = raw_logit + math.log(positive_weight)
        weighted_probability = 1.0 / (1.0 + math.exp(-weighted_logit))
        layer = getattr(model.head, f"{channel}_head")
        trained_bias = float(layer.bias.detach().float().item())
        report[channel] = {
            "fit_positive_rate": prior,
            "positive_weight": positive_weight,
            "raw_prior_initial_logit": raw_logit,
            "weighted_bce_stationary_initial_logit": weighted_logit,
            "weighted_bce_stationary_probability": weighted_probability,
            "checkpoint_bias": trained_bias,
            "checkpoint_bias_minus_raw_prior": trained_bias - raw_logit,
            "checkpoint_bias_minus_weighted_stationary": (
                trained_bias - weighted_logit
            ),
        }
    return report


def _runtime_counts(runtime):
    return {
        "birth_proposals": int(runtime.birth_proposals),
        "birth_admissions": int(runtime.birth_admissions),
        "candidate_arbitration_suppressions": int(
            runtime.arbitration_suppressions
        ),
        "candidate_cancellations": int(runtime.candidate_cancellations),
        "active_abandonments": int(runtime.active_abandonments),
        "deferred_birth_due_to_release": int(
            runtime.deferred_birth_due_to_release
        ),
        "committed_emissions": len(runtime.committed),
    }


def _add_counts(total, row):
    for key, value in row.items():
        total[key] = total.get(key, 0) + int(value)


def diagnose(config, checkpoint_path, screen_result_path, device_name, seed):
    config = Path(config).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    screen_result_path = Path(screen_result_path).resolve()
    cfg = Config.fromfile(config)
    if not str(cfg.route_stage).startswith("persistent_binding"):
        raise ValueError("score diagnosis is restricted to persistent-binding routes")
    if bool(cfg.raw_video_finetuning):
        raise ValueError("score diagnosis is feature-only")
    if bool(cfg.inference.load_from_raw_predictions):
        raise ValueError("score diagnosis cannot load raw predictions")
    if bool(cfg.solver.amp):
        raise ValueError("score diagnosis requires the registered FP32 route")
    if bool(cfg.solver.ema):
        raise ValueError("score diagnosis does not permit an unregistered EMA path")

    screen_result = _load_json(screen_result_path)
    gate_row = screen_result.get("gate_row")
    if not isinstance(gate_row, dict):
        raise ValueError("source screen result has no gate_row")
    provenance = gate_row.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("source screen result has no provenance")
    expected_mode = str(cfg.model.trajectory_binding_mode)
    if screen_result.get("binding_mode") != expected_mode:
        raise ValueError("screen result and config binding modes differ")
    if provenance.get("checkpoint_sha256") != _sha256(checkpoint_path):
        raise ValueError("checkpoint hash differs from the source screen result")
    if provenance.get("config_sha256") != _sha256(config):
        raise ValueError("config hash differs from the source screen result")
    if gate_row.get("reporting_accessed") is not False:
        raise ValueError("source screen result is not calibration-only")

    calibration_ids = _ids(cfg.calibration_manifest)
    reporting_ids = _ids(cfg.reporting_manifest)
    if calibration_ids.intersection(reporting_ids):
        raise ValueError("calibration and reporting manifests overlap")
    dataset_cfg = dict(cfg.dataset.val)
    dataset_cfg["test_mode"] = True
    logger = logging.getLogger("PersistentBindingScoreDiagnosis")
    dataset = build_dataset(dataset_cfg, default_args=dict(logger=logger))
    dataset_ids = set(dataset.packet_manifests)
    if dataset_ids != calibration_ids:
        raise ValueError("diagnostic dataset is not exactly the calibration split")
    if dataset_ids.intersection(reporting_ids):
        raise ValueError("diagnostic dataset accessed a reporting video")
    dataloader = build_dataloader(
        dataset,
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=False,
        **dict(cfg.solver.val),
    )

    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("score diagnosis requires an allocated CUDA GPU")
    model = build_detector(cfg.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(_normalized_state_dict(checkpoint), strict=True)
    model.eval()
    model.reset_online_states()

    values = {channel: [] for channel in CHANNELS}
    per_slot = {
        channel: [[] for _ in range(model.head.num_slots)]
        for channel in CHANNELS
    }
    runtime_totals = {}
    runtime = None
    current_video = None
    tokens = 0
    chunks = 0
    direct_emissions = 0
    with torch.no_grad():
        for raw_batch in dataloader:
            batch = move_data_to_device(raw_batch, device)
            meta = batch["metas"][0]
            control = batch["stream_control"][0]
            video_id = str(meta.get("video_id", meta.get("video_name")))
            if bool(control.get("is_video_start", False)):
                if runtime is not None:
                    _add_counts(runtime_totals, _runtime_counts(runtime))
                runtime = None
                current_video = video_id
            elif runtime is None or video_id != current_video:
                raise ValueError("calibration chunks are not a chronological stream")
            output = model.infer_step(
                batch["inputs"],
                batch["masks"],
                meta,
                runtime_state=runtime,
                class_names=dataset.class_map,
            )
            runtime = output.runtime_state
            direct_emissions += len(output.emissions)
            chunks += 1
            tokens += len(output.logits)
            for token_output in output.logits:
                for channel, (logit_name, _, _) in CHANNELS.items():
                    probabilities = (
                        token_output[logit_name]
                        .detach()
                        .float()
                        .sigmoid()[0]
                        .cpu()
                        .tolist()
                    )
                    values[channel].extend(probabilities)
                    for slot, probability in enumerate(probabilities):
                        per_slot[channel][slot].append(probability)
    if runtime is not None:
        _add_counts(runtime_totals, _runtime_counts(runtime))
    if tokens <= 0 or chunks <= 0:
        raise ValueError("calibration score diagnosis produced no tokens")
    if direct_emissions != runtime_totals.get("committed_emissions", 0):
        raise ValueError("direct and runtime emission counts differ")

    channel_reports = {}
    for channel, (_, threshold_name, _) in CHANNELS.items():
        threshold = float(getattr(model.head, threshold_name))
        channel_reports[channel] = {
            "all_slots": summarize_probabilities(values[channel], threshold),
            "per_slot": [
                summarize_probabilities(slot_values, threshold)
                for slot_values in per_slot[channel]
            ],
        }
    return {
        "schema_version": "persistent_binding_score_diagnosis.v1",
        "passed": True,
        "purpose": "calibration_only_model_score_diagnosis",
        "effectiveness_claim_authorized": False,
        "raw_rgb_authorized": False,
        "reporting_accessed": False,
        "analysis_commit": _git_commit(ROOT),
        "checkpoint_training_commit": provenance["code_commit"],
        "config": str(config),
        "config_sha256": _sha256(config),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "source_screen_result": str(screen_result_path),
        "source_screen_result_sha256": _sha256(screen_result_path),
        "binding_mode": expected_mode,
        "prior_bias_mode": str(
            cfg.model.get("prior_bias_mode", "raw_probability")
        ),
        "seed": int(seed),
        "input": "fixed_cached_causal_features",
        "dataset_videos": len(dataset_ids),
        "dataset_chunks": chunks,
        "dataset_tokens": tokens,
        "dataset_video_ids_sha256": _canonical_sha256(sorted(dataset_ids)),
        "num_slots": int(model.head.num_slots),
        "channel_score_distributions": channel_reports,
        "lifecycle_counts": runtime_totals,
        "prior_and_checkpoint_bias_audit": _prior_audit(model, cfg),
        "strict_determinism": configure_strict_determinism(),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose calibration-only persistent-binding lifecycle scores"
    )
    parser.add_argument("config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--screen-result", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=705)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    set_seed(args.seed)
    configure_strict_determinism()
    payload = diagnose(
        args.config,
        args.checkpoint,
        args.screen_result,
        args.device,
        args.seed,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
