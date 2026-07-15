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
    AUDIT_SCHEMA_VERSION,
    CrsEpsGoldGateError,
    MARGIN_ATTESTATION_ROLE,
    SELECTION_ATTESTATION_ROLE,
    evaluate_gold_audit,
    validate_gold_checkpoint,
    validate_gold_margins,
    validate_gold_selection,
)
from opentad.utils.crs_eps_gold_evidence import (  # noqa: E402
    CrsEpsGoldEvidenceError,
    bind_manifest_to_loaded_dataset,
    checkpoint_state_from_bytes,
)
from opentad.utils.crs_eps_sampling import (  # noqa: E402
    CrsEpsSamplingError,
    validate_epoch_manifest,
)
from opentad.utils.evidence_bundle import (  # noqa: E402
    EvidenceBundleError,
    publish_exclusive_file,
    read_verified_bundle_bytes,
    strict_json_from_bytes,
)
from opentad.utils.full_petal_attestation import (  # noqa: E402
    AttestationError,
    public_key_base64,
    verify_payload,
)
from opentad.utils.full_petal_launch import resolved_config_sha256  # noqa: E402
from opentad.utils.full_petal_role_signing import (  # noqa: E402
    sign_crs_eps_g0_audit,
)


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


def _verified_preregistration(payload, *, trust_root, role, bindings, label):
    try:
        body = verify_payload(payload, trust_root=trust_root, role=role)
        if role == SELECTION_ATTESTATION_ROLE:
            body = validate_gold_selection(body)
        else:
            body = validate_gold_margins(body)
    except (AttestationError, CrsEpsGoldGateError) as exc:
        raise GoldAuditRunnerError(f"{label} is not a valid signed preregistration: {exc}") from exc
    for key in (
        "commit_sha",
        "resolved_config_sha256",
        "scientific_config_sha256",
        "data_identity_sha256",
        "episode_manifest_sha256",
        "sampling_specs_sha256",
    ):
        if body.get(key) != bindings[key]:
            raise GoldAuditRunnerError(f"{label} does not bind the exact {key}")
    return body


def _read_bound_checkpoint(
    checkpoint_path,
    checkpoint_key,
    checkpoint_binding,
    *,
    bundle_root,
    expected_seed,
):
    try:
        checkpoint_binding = validate_gold_checkpoint(checkpoint_binding)
        if checkpoint_binding["state_key"] != checkpoint_key:
            raise GoldAuditRunnerError(
                "runtime checkpoint state key differs from signed preregistration"
            )
        if checkpoint_binding["generation"]["seed"] != expected_seed:
            raise GoldAuditRunnerError(
                "signed checkpoint seed differs from the immutable manifest"
            )
        bound_path, checkpoint_bytes = read_verified_bundle_bytes(
            {
                "path": checkpoint_binding["path"],
                "sha256": checkpoint_binding["sha256"],
            },
            bundle_root,
            "preregistered G0 checkpoint",
        )
        if checkpoint_path.resolve(strict=True) != bound_path:
            raise GoldAuditRunnerError(
                "runtime checkpoint path differs from signed preregistration"
            )
        if len(checkpoint_bytes) != checkpoint_binding["byte_size"]:
            raise GoldAuditRunnerError(
                "runtime checkpoint byte size differs from signed preregistration"
            )
        checkpoint_state_from_bytes(checkpoint_bytes, checkpoint_key)
    except CrsEpsGoldGateError as exc:
        raise GoldAuditRunnerError(
            f"signed checkpoint preregistration is invalid: {exc}"
        ) from exc
    except (CrsEpsGoldEvidenceError, EvidenceBundleError, OSError) as exc:
        raise GoldAuditRunnerError(
            f"cannot verify preregistered G0 checkpoint: {exc}"
        ) from exc
    return checkpoint_bytes


def _load_checkpoint(model, checkpoint_bytes, checkpoint_key):
    try:
        normalized = checkpoint_state_from_bytes(
            checkpoint_bytes, checkpoint_key
        )
        model.load_state_dict(normalized, strict=True)
    except (CrsEpsGoldEvidenceError, RuntimeError) as exc:
        raise GoldAuditRunnerError(f"checkpoint does not strictly match the G0 model: {exc}") from exc


def _build_bound_model(
    cfg,
    checkpoint_path,
    checkpoint_key,
    checkpoint_binding,
    *,
    bundle_root,
    expected_seed,
):
    checkpoint_bytes = _read_bound_checkpoint(
        checkpoint_path,
        checkpoint_key,
        checkpoint_binding,
        bundle_root=bundle_root,
        expected_seed=expected_seed,
    )
    model = build_detector(dict(cfg.model)).cpu().train()
    _load_checkpoint(model, checkpoint_bytes, checkpoint_key)
    return model


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
    parser.add_argument("--margins", type=Path, required=True)
    parser.add_argument("--signing-key", type=Path, required=True)
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
        margins_path = args.margins.resolve(strict=True)
        for path, label in (
            (manifest_path, "episode manifest"),
            (selection_path, "selection"),
            (margins_path, "margins"),
        ):
            if path.parent != output.parent:
                raise GoldAuditRunnerError(
                    f"G0 {label} must share the immutable output evidence bundle"
                )
        cfg = Config.fromfile(str(config_path))
        if cfg.get("route_stage") != "q2_crs_eps_hh_ipw_implementation_gate":
            raise GoldAuditRunnerError("G0 audit requires a CRS-EPS implementation config")
        manifest = _load_json(manifest_path, "CRS-EPS episode manifest")
        validate_epoch_manifest(manifest)
        manifest_sha256 = manifest["manifest_sha256"]
        dataset = build_dataset(dict(cfg.dataset.train))
        if getattr(dataset, "sampling_protocol", None) != "crs_eps":
            raise GoldAuditRunnerError("G0 audit dataset is not CRS-EPS")
        bindings = bind_manifest_to_loaded_dataset(
            manifest,
            dataset,
            cfg=cfg,
            config_path=config_path,
            repository_root=ROOT,
            commit_sha=commit_sha,
            resolved_config_sha256=resolved_config_sha256(cfg),
            scientific_config_sha256=resolved_config_sha256(cfg, scientific=True),
        )
        trust_root = dict(cfg.launch_contract.attestation_trust_roots.g0)
        if public_key_base64(args.signing_key) != trust_root["public_key"]:
            raise GoldAuditRunnerError("G0 audit signing key does not match the trust root")
        signed_selection = _load_json(selection_path, "G0 selection")
        selection = _verified_preregistration(
            signed_selection,
            trust_root=trust_root,
            role=SELECTION_ATTESTATION_ROLE,
            bindings=bindings,
            label="G0 selection",
        )
        selected = [
            (sample["video_id"], sample["draw_index"])
            for sample in selection["samples"]
        ]
        signed_margins = _load_json(margins_path, "G0 margins")
        margins = _verified_preregistration(
            signed_margins,
            trust_root=trust_root,
            role=MARGIN_ATTESTATION_ROLE,
            bindings=bindings,
            label="G0 margins",
        )
        if margins["selection_artifact_sha256"] != _sha256_file(selection_path):
            raise GoldAuditRunnerError("G0 margins do not bind the signed selection artifact")
        model = _build_bound_model(
            cfg,
            checkpoint_path,
            args.checkpoint_key,
            selection["checkpoint"],
            bundle_root=output.parent,
            expected_seed=manifest["seed"],
        )
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
        gate = evaluate_gold_audit(rows, margins)
        artifact_body = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "status": gate["status"],
            "commit_sha": commit_sha,
            "source_tree_sha256": bindings["source_tree_sha256"],
            "resolved_config_sha256": bindings["resolved_config_sha256"],
            "scientific_config_sha256": bindings["scientific_config_sha256"],
            "data_identity_sha256": bindings["data_identity_sha256"],
            "config": {"path": str(config_path), "sha256": _sha256_file(config_path)},
            "checkpoint": selection["checkpoint"],
            "episode_manifest": {
                "path": manifest_path.name,
                "file_sha256": _sha256_file(manifest_path),
                "manifest_sha256": manifest_sha256,
                "sampling_specs_sha256": bindings["sampling_specs_sha256"],
            },
            "selection": {
                "path": selection_path.name,
                "sha256": _sha256_file(selection_path),
            },
            "margins": {
                "path": margins_path.name,
                "sha256": _sha256_file(margins_path),
            },
            "rows": rows,
            "gate": gate,
        }
        artifact = sign_crs_eps_g0_audit(
            artifact_body,
            private_key_path=args.signing_key,
            key_id=trust_root["key_id"],
        )
        encoded = (
            json.dumps(artifact, allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        publish_exclusive_file(output, encoded)
    except (
        AttestationError,
        CrsEpsGoldEvidenceError,
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
