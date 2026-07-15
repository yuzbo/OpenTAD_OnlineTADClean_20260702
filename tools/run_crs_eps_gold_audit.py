#!/usr/bin/env python3
"""Run the CPU-only four-arm CRS-EPS G0 replay-fidelity audit."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
from mmengine import Config


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.datasets import build_dataset  # noqa: E402
from opentad.models import build_detector  # noqa: E402
from opentad.utils.crs_eps_audit import run_matched_crs_eps_pair  # noqa: E402
from opentad.utils.crs_eps_gold_gate import (  # noqa: E402
    CrsEpsGoldGateError,
    evaluate_gold_audit,
)
from opentad.utils.crs_eps_sampling import (  # noqa: E402
    CrsEpsSamplingError,
    validate_epoch_manifest,
)
from opentad.utils.evidence_bundle import (  # noqa: E402
    EvidenceBundleError,
    publish_exclusive_file,
    strict_json_from_bytes,
)


SELECTION_SCHEMA = "full-petal-crs-eps-g0-selection-v1"
AUDIT_SCHEMA = "full-petal-crs-eps-g0-audit-v1"
AUDIT_MODES = ("dynamic_birth", "fixed_192", "reset")


class GoldAuditRunnerError(RuntimeError):
    pass


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path, label):
    try:
        raw = Path(path).read_bytes()
        return dict(strict_json_from_bytes(raw, label, require_object=True))
    except (OSError, EvidenceBundleError) as exc:
        raise GoldAuditRunnerError(f"cannot load {label}: {exc}") from exc


def _clean_commit():
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise GoldAuditRunnerError("G0 audit requires a clean committed checkout")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise GoldAuditRunnerError("G0 audit commit identity is malformed")
    return commit


def _external_output(path):
    output = Path(path).expanduser().resolve()
    try:
        output.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise GoldAuditRunnerError("G0 audit evidence must remain outside the repository")
    if output.exists():
        raise GoldAuditRunnerError(f"refusing to overwrite G0 audit evidence: {output}")
    return output


def _selection(payload, *, commit_sha, manifest_sha256):
    if set(payload) != {
        "schema_version",
        "status",
        "commit_sha",
        "episode_manifest_sha256",
        "samples",
    }:
        raise GoldAuditRunnerError("G0 selection fields differ")
    if payload["schema_version"] != SELECTION_SCHEMA:
        raise GoldAuditRunnerError("G0 selection schema is unsupported")
    if payload["status"] != "PREREGISTERED_BEFORE_Q2_EFFECTIVENESS":
        raise GoldAuditRunnerError("G0 selection is not marked outcome-blind")
    if payload["commit_sha"] != commit_sha:
        raise GoldAuditRunnerError("G0 selection commit differs from the checkout")
    if payload["episode_manifest_sha256"] != manifest_sha256:
        raise GoldAuditRunnerError("G0 selection manifest hash differs")
    samples = payload["samples"]
    if not isinstance(samples, list) or not samples:
        raise GoldAuditRunnerError("G0 selection requires at least one sample")
    normalized = []
    seen = set()
    for sample in samples:
        if not isinstance(sample, dict) or set(sample) != {"video_id", "draw_index"}:
            raise GoldAuditRunnerError("G0 selected-sample fields differ")
        video_id = sample["video_id"]
        draw_index = sample["draw_index"]
        if not isinstance(video_id, str) or not video_id:
            raise GoldAuditRunnerError("G0 selected video ID is invalid")
        if isinstance(draw_index, bool) or not isinstance(draw_index, int) or draw_index < 0:
            raise GoldAuditRunnerError("G0 selected draw index is invalid")
        key = (video_id, draw_index)
        if key in seen:
            raise GoldAuditRunnerError("G0 selection contains a duplicate sample")
        seen.add(key)
        normalized.append(key)
    return normalized


def _load_checkpoint(model, checkpoint_path, checkpoint_key):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise GoldAuditRunnerError(f"cannot load frozen G0 checkpoint: {exc}") from exc
    if not isinstance(checkpoint, dict) or checkpoint_key not in checkpoint:
        raise GoldAuditRunnerError(f"checkpoint lacks the explicit {checkpoint_key} state")
    state = checkpoint[checkpoint_key]
    if not isinstance(state, dict):
        raise GoldAuditRunnerError("checkpoint model state is not a mapping")
    normalized = {
        (name[7:] if name.startswith("module.") else name): value
        for name, value in state.items()
    }
    try:
        model.load_state_dict(normalized, strict=True)
    except RuntimeError as exc:
        raise GoldAuditRunnerError(f"checkpoint does not strictly match the G0 model: {exc}") from exc


def _episode_kwargs(sample):
    return {
        "inputs": torch.from_numpy(sample["inputs"]).unsqueeze(0),
        "masks": torch.from_numpy(sample["masks"]).unsqueeze(0),
        "model_meta": sample["metas"],
        "supervision_schedule": sample["prefix_schedule"],
        "crs_eps_control": sample["crs_eps"],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-key",
        choices=("state_dict", "state_dict_ema"),
        default="state_dict",
    )
    parser.add_argument("--episode-manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--margins", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser, parser.parse_args(argv)


def main(argv=None):
    parser, args = parse_args(argv)
    try:
        output = _external_output(args.output)
        commit_sha = _clean_commit()
        config_path = args.config.resolve(strict=True)
        config_path.relative_to(ROOT.resolve())
        checkpoint_path = args.checkpoint.resolve(strict=True)
        manifest_path = args.episode_manifest.resolve(strict=True)
        selection_path = args.selection.resolve(strict=True)
        cfg = Config.fromfile(str(config_path))
        if cfg.get("route_stage") != "q2_crs_eps_hh_ipw_implementation_gate":
            raise GoldAuditRunnerError("G0 audit requires a CRS-EPS implementation config")
        manifest = _load_json(manifest_path, "CRS-EPS episode manifest")
        validate_epoch_manifest(manifest)
        manifest_sha256 = manifest["manifest_sha256"]
        selection_payload = _load_json(selection_path, "G0 selection")
        selected = _selection(
            selection_payload,
            commit_sha=commit_sha,
            manifest_sha256=manifest_sha256,
        )
        provenance_commit = manifest.get("provenance", {}).get("commit_sha")
        if provenance_commit != commit_sha:
            raise GoldAuditRunnerError("episode manifest commit differs from the checkout")
        dataset = build_dataset(dict(cfg.dataset.train))
        if getattr(dataset, "sampling_protocol", None) != "crs_eps":
            raise GoldAuditRunnerError("G0 audit dataset is not CRS-EPS")
        model = build_detector(dict(cfg.model)).cpu().train()
        _load_checkpoint(model, checkpoint_path, args.checkpoint_key)
        videos = {video["video_id"]: video for video in manifest["videos"]}
        rows = []
        for video_id, draw_index in selected:
            if video_id not in videos:
                raise GoldAuditRunnerError(f"selected video is absent from the manifest: {video_id}")
            video = videos[video_id]
            if not 0 <= draw_index < len(video["draws"]):
                raise GoldAuditRunnerError("selected draw index escapes the manifest")
            draw = video["draws"][draw_index]
            gold_sample = dataset.build_gold_audit_sample(
                video,
                draw,
                "video_start_full",
                manifest_sha256=manifest_sha256,
            )
            for mode in AUDIT_MODES:
                candidate_sample = dataset.build_gold_audit_sample(
                    video,
                    draw,
                    mode,
                    manifest_sha256=manifest_sha256,
                )
                left_model = deepcopy(model)
                right_model = deepcopy(model)
                trace = run_matched_crs_eps_pair(
                    left_model,
                    right_model,
                    _episode_kwargs(gold_sample),
                    _episode_kwargs(candidate_sample),
                    comparison_type="replay_fidelity",
                )
                rows.append(
                    {
                        "video_id": video_id,
                        "draw_index": draw_index,
                        "mode": mode,
                        "trace": trace,
                    }
                )
        margins_reference = None
        gate = None
        if args.margins is not None:
            margins_path = args.margins.resolve(strict=True)
            margins = _load_json(margins_path, "G0 margins")
            expected_margin_bindings = {
                "commit_sha": commit_sha,
                "episode_manifest_sha256": manifest_sha256,
                "selection_sha256": _sha256_file(selection_path),
            }
            if any(margins.get(key) != value for key, value in expected_margin_bindings.items()):
                raise GoldAuditRunnerError("G0 margins do not bind the exact audit inputs")
            gate = evaluate_gold_audit(rows, margins)
            margins_reference = {
                "path": str(margins_path),
                "sha256": _sha256_file(margins_path),
            }
        artifact = {
            "schema_version": AUDIT_SCHEMA,
            "status": gate["status"] if gate is not None else "OBSERVED_UNGATED",
            "commit_sha": commit_sha,
            "config": {"path": str(config_path), "sha256": _sha256_file(config_path)},
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": _sha256_file(checkpoint_path),
                "state_key": args.checkpoint_key,
            },
            "episode_manifest": {
                "path": str(manifest_path),
                "file_sha256": _sha256_file(manifest_path),
                "manifest_sha256": manifest_sha256,
            },
            "selection": {
                "path": str(selection_path),
                "sha256": _sha256_file(selection_path),
            },
            "margins": margins_reference,
            "rows": rows,
            "gate": gate,
        }
        encoded = (
            json.dumps(artifact, allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        publish_exclusive_file(output, encoded)
    except (
        CrsEpsGoldGateError,
        CrsEpsSamplingError,
        EvidenceBundleError,
        GoldAuditRunnerError,
        KeyError,
        OSError,
        subprocess.CalledProcessError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    print(f"CRS_EPS_G0_AUDIT={output}")
    print(f"CRS_EPS_G0_AUDIT_SHA256={_sha256_file(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
