#!/usr/bin/env python3
"""Sign outcome-blind CRS-EPS G0 sample selection and decision margins."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmengine import Config  # noqa: E402
from opentad.datasets import build_dataset  # noqa: E402
from opentad.utils.crs_eps_gold_evidence import (  # noqa: E402
    CrsEpsGoldEvidenceError,
    bind_manifest_to_loaded_dataset,
    sha256_file,
)
from opentad.utils.crs_eps_gold_gate import (  # noqa: E402
    MARGIN_SCHEMA_VERSION,
    SELECTION_SCHEMA_VERSION,
)
from opentad.utils.evidence_bundle import (  # noqa: E402
    EvidenceBundleError,
    publish_exclusive_file,
    strict_json_from_bytes,
)
from opentad.utils.full_petal_attestation import (  # noqa: E402
    AttestationError,
    public_key_base64,
)
from opentad.utils.full_petal_launch import resolved_config_sha256  # noqa: E402
from opentad.utils.full_petal_role_signing import (  # noqa: E402
    sign_crs_eps_g0_margins,
    sign_crs_eps_g0_selection,
)


_THRESHOLD_FIELDS = {
    "min_gradient_cosine",
    "min_gradient_sign_agreement",
    "min_runtime_continuous_cosine",
    "max_relative_loss_error",
    "require_runtime_discrete_equal",
    "max_mean_dynamic_replay_ratio",
    "max_video_start_fallback_fraction",
    "min_dynamic_minus_fixed_gradient_cosine",
    "min_dynamic_minus_reset_gradient_cosine",
}


class GoldPreregistrationError(RuntimeError):
    pass


def _load_json(path, label):
    try:
        return dict(
            strict_json_from_bytes(
                Path(path).read_bytes(), label, require_object=True
            )
        )
    except (OSError, EvidenceBundleError) as exc:
        raise GoldPreregistrationError(f"cannot load {label}: {exc}") from exc


def _clean_commit():
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise GoldPreregistrationError("G0 preregistration requires a clean checkout")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _external_new_file(path):
    output = Path(path).expanduser().resolve()
    try:
        output.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise GoldPreregistrationError("G0 evidence must remain outside the repository")
    if output.exists():
        raise GoldPreregistrationError(f"refusing to overwrite G0 evidence: {output}")
    return output


def _publish(path, payload):
    encoded = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    publish_exclusive_file(path, encoded)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--episode-manifest", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--margin-thresholds", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path, required=True)
    parser.add_argument("--margins-output", type=Path, required=True)
    parser.add_argument("--signing-key", type=Path, required=True)
    return parser, parser.parse_args(argv)


def main(argv=None):
    parser, args = parse_args(argv)
    try:
        selection_output = _external_new_file(args.selection_output)
        margins_output = _external_new_file(args.margins_output)
        if selection_output.parent != margins_output.parent:
            raise GoldPreregistrationError(
                "G0 selection and margins must share one immutable evidence bundle"
            )
        commit_sha = _clean_commit()
        config_path = args.config.resolve(strict=True)
        config_path.relative_to(ROOT.resolve())
        cfg = Config.fromfile(str(config_path))
        manifest = _load_json(args.episode_manifest.resolve(strict=True), "G0 manifest")
        dataset = build_dataset(dict(cfg.dataset.train))
        resolved_sha256 = resolved_config_sha256(cfg)
        scientific_sha256 = resolved_config_sha256(cfg, scientific=True)
        bindings = bind_manifest_to_loaded_dataset(
            manifest,
            dataset,
            cfg=cfg,
            config_path=config_path,
            repository_root=ROOT,
            commit_sha=commit_sha,
            resolved_config_sha256=resolved_sha256,
            scientific_config_sha256=scientific_sha256,
        )
        trust_root = dict(cfg.launch_contract.attestation_trust_roots.g0)
        if public_key_base64(args.signing_key) != trust_root["public_key"]:
            raise GoldPreregistrationError("G0 signing key does not match the trust root")

        samples_payload = _load_json(args.samples.resolve(strict=True), "G0 samples")
        if set(samples_payload) != {"samples"}:
            raise GoldPreregistrationError("G0 sample input requires exactly samples")
        selection = sign_crs_eps_g0_selection(
            {
                "schema_version": SELECTION_SCHEMA_VERSION,
                "status": "PREREGISTERED_BEFORE_G0_EXECUTION",
                **{key: bindings[key] for key in (
                    "commit_sha",
                    "resolved_config_sha256",
                    "scientific_config_sha256",
                    "data_identity_sha256",
                    "episode_manifest_sha256",
                    "sampling_specs_sha256",
                )},
                "samples": samples_payload["samples"],
            },
            private_key_path=args.signing_key,
            key_id=trust_root["key_id"],
        )
        _publish(selection_output, selection)

        thresholds = _load_json(
            args.margin_thresholds.resolve(strict=True), "G0 margin thresholds"
        )
        if set(thresholds) != _THRESHOLD_FIELDS:
            raise GoldPreregistrationError("G0 margin threshold fields differ")
        margins = sign_crs_eps_g0_margins(
            {
                "schema_version": MARGIN_SCHEMA_VERSION,
                "status": "PREREGISTERED_BEFORE_G0_EXECUTION",
                **{key: bindings[key] for key in (
                    "commit_sha",
                    "resolved_config_sha256",
                    "scientific_config_sha256",
                    "data_identity_sha256",
                    "episode_manifest_sha256",
                    "sampling_specs_sha256",
                )},
                "selection_artifact_sha256": sha256_file(selection_output),
                **thresholds,
            },
            private_key_path=args.signing_key,
            key_id=trust_root["key_id"],
        )
        _publish(margins_output, margins)
    except (
        AttestationError,
        CrsEpsGoldEvidenceError,
        EvidenceBundleError,
        GoldPreregistrationError,
        KeyError,
        OSError,
        subprocess.CalledProcessError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    print(f"CRS_EPS_G0_SELECTION={selection_output}")
    print(f"CRS_EPS_G0_SELECTION_SHA256={sha256_file(selection_output)}")
    print(f"CRS_EPS_G0_MARGINS={margins_output}")
    print(f"CRS_EPS_G0_MARGINS_SHA256={sha256_file(margins_output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
