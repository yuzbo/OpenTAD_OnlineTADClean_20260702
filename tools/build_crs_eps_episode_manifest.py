#!/usr/bin/env python3
"""Build one immutable CPU-only CRS-EPS epoch sampling manifest."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmengine.config import Config  # noqa: E402
from opentad.datasets import build_dataset  # noqa: E402
from opentad.utils.crs_eps_sampling import (  # noqa: E402
    DEFAULT_CONTEXT_BINS,
    DEFAULT_DETACH_INTERVAL,
    DEFAULT_MIXTURE,
    DEFAULT_SUFFIX_BINS,
    CrsEpsSamplingError,
    build_epoch_manifest,
    sampling_specs_sha256,
    validate_epoch_manifest,
)
from opentad.utils.full_petal_identity import build_data_identity  # noqa: E402
from opentad.utils.full_petal_launch import resolved_config_sha256  # noqa: E402
from opentad.utils.evidence_bundle import (  # noqa: E402
    EvidenceBundleError,
    publish_exclusive_file,
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_commit():
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise CrsEpsSamplingError("CRS-EPS manifest generation requires a clean checkout")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _external_output(path):
    output = Path(path).expanduser().resolve()
    try:
        output.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise CrsEpsSamplingError("CRS-EPS evidence must remain outside the repository")
    if output.exists():
        raise CrsEpsSamplingError(f"refusing to overwrite CRS-EPS manifest: {output}")
    return output


def _validate_config_contract(cfg):
    contract = cfg.get("crs_eps_contract")
    if contract is None:
        raise CrsEpsSamplingError("config does not declare a crs_eps_contract")
    expected = {
        "schema_version": "full-petal-crs-eps-contract-v1",
        "suffix_bins": DEFAULT_SUFFIX_BINS,
        "context_bins": DEFAULT_CONTEXT_BINS,
        "detach_interval": DEFAULT_DETACH_INTERVAL,
        "mixture": DEFAULT_MIXTURE,
        "primary_estimator": "hansen_hurwitz_repeated_exposure_uncapped_ipw",
    }
    for key, value in expected.items():
        actual = dict(contract[key]) if key == "mixture" else contract[key]
        if actual != value:
            raise CrsEpsSamplingError(f"config CRS-EPS contract drifted at {key}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--draws-per-video", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser, parser.parse_args(argv)


def main(argv=None):
    parser, args = parse_args(argv)
    try:
        output = _external_output(args.output)
        commit_sha = _clean_commit()
        config_path = args.config.resolve(strict=True)
        cfg = Config.fromfile(config_path)
        _validate_config_contract(cfg)
        train_dataset = build_dataset(dict(cfg.dataset.train))
        specs = tuple(train_dataset.iter_crs_eps_sampling_specs())
        data_identity = build_data_identity(cfg)
        provenance = {
            "commit_sha": commit_sha,
            "config_path": str(config_path),
            "config_sha256": _sha256_file(config_path),
            "resolved_config_sha256": resolved_config_sha256(cfg),
            "scientific_config_sha256": resolved_config_sha256(
                cfg, scientific=True
            ),
            "data_identity_sha256": data_identity["identity_sha256"],
            "sampling_specs_sha256": sampling_specs_sha256(specs),
            "annotation_sha256": _sha256_file(train_dataset.ann_file),
            "feature_cache_manifest_sha256": train_dataset.cache_manifest_sha256,
            "split_manifest_sha256": train_dataset.split_manifest_sha256,
            "split_manifest_file_sha256": train_dataset.split_manifest_file_sha256,
            "split_role": train_dataset.split_role,
            "split_seed": train_dataset.split_seed,
        }
        manifest = build_epoch_manifest(
            specs,
            epoch=args.epoch,
            seed=args.seed,
            draws_per_video=args.draws_per_video,
            provenance=provenance,
        )
        validate_epoch_manifest(manifest)
        encoded = (
            json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        publish_exclusive_file(output, encoded)
    except (
        CrsEpsSamplingError,
        EvidenceBundleError,
        KeyError,
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        parser.error(str(exc))
    print(f"CRS_EPS_EPISODE_MANIFEST={output}")
    print(f"CRS_EPS_EPISODE_MANIFEST_SHA256={_sha256_file(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
