#!/usr/bin/env python3
"""Create immutable deterministic-init checkpoints for the Q2 CPU audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import numpy as np
import torch
from mmengine import Config


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.models import build_detector  # noqa: E402
from opentad.utils.full_petal_identity import canonical_json_sha256  # noqa: E402
from opentad.utils.full_petal_launch import resolved_config_sha256  # noqa: E402
from opentad.utils.misc import set_seed  # noqa: E402
from opentad.utils.q2_capacity_audit import (  # noqa: E402
    CHECKPOINT_SCHEMA_VERSION,
    Q2CapacityAuditError,
    state_dict_fingerprint,
)


FROZEN_SEEDS = (705, 706, 707)


class CheckpointBuildError(RuntimeError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_commit():
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise CheckpointBuildError("checkpoint generation requires a clean checkout")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40 or any(value not in "0123456789abcdef" for value in commit):
        raise CheckpointBuildError("git commit identity is malformed")
    return commit


def external_new_directory(path):
    output = Path(path).expanduser().resolve()
    try:
        output.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise CheckpointBuildError(
            "checkpoint evidence must stay outside the repository"
        )
    if output.exists():
        raise CheckpointBuildError(
            f"refusing to overwrite checkpoint evidence: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def validate_config(cfg):
    if cfg.get("route_stage") != "q2_persistent_binding_one_factor":
        raise CheckpointBuildError("capacity checkpoints require the frozen Q2 route")
    if tuple(int(seed) for seed in cfg.get("pilot_seeds", ())) != FROZEN_SEEDS:
        raise CheckpointBuildError("Q2 pilot seeds drifted from 705/706/707")
    head = cfg.model.head
    expected = {
        "num_slots": 4,
        "dropout": 0.1,
        "birth_threshold": 0.5,
        "alive_threshold": 0.5,
        "end_threshold": 0.5,
        "refractory_steps": 2,
    }
    observed = {key: head.get(key) for key in expected}
    if observed != expected:
        raise CheckpointBuildError(
            f"Q2 lifecycle config drifted: expected {expected}, observed {observed}"
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(FROZEN_SEEDS))
    return parser, parser.parse_args(argv)


def main(argv=None):
    parser, args = parse_args(argv)
    partial = None
    try:
        seeds = tuple(int(seed) for seed in args.seeds)
        if seeds != FROZEN_SEEDS:
            raise CheckpointBuildError(
                "capacity checkpoint seeds must be exactly 705 706 707"
            )
        output = external_new_directory(args.output)
        config_path = args.config.resolve(strict=True)
        config_path.relative_to(ROOT.resolve())
        if config_path.name != "thumos_pes_q2_persist_fixed.py":
            raise CheckpointBuildError(
                "capacity checkpoints require thumos_pes_q2_persist_fixed.py"
            )
        commit_sha = clean_commit()
        cfg = Config.fromfile(str(config_path))
        validate_config(cfg)
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        partial = output.parent / f".{output.name}.partial-{uuid.uuid4().hex}"
        partial.mkdir()
        checkpoint_records = []
        for seed in seeds:
            set_seed(seed)
            model = build_detector(dict(cfg.model)).cpu().train()
            state = model.state_dict()
            checkpoint_path = partial / f"seed_{seed}.pth"
            torch.save(
                {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "seed": seed,
                    "state_dict": state,
                },
                checkpoint_path,
            )
            checkpoint_records.append(
                {
                    "seed": seed,
                    "path": checkpoint_path.name,
                    "state_key": "state_dict",
                    "sha256": sha256_file(checkpoint_path),
                    "byte_size": checkpoint_path.stat().st_size,
                    "tensor_fingerprint_sha256": state_dict_fingerprint(state),
                }
            )
            del model

        manifest = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "commit_sha": commit_sha,
            "config": {
                "path": config_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(config_path),
                "resolved_config_sha256": resolved_config_sha256(cfg),
                "scientific_config_sha256": resolved_config_sha256(
                    cfg, scientific=True
                ),
            },
            "generation": {
                "seeds": list(seeds),
                "torch_version": str(torch.__version__),
                "numpy_version": str(np.__version__),
                "device": "cpu",
                "model_mode": "train",
                "deterministic_algorithms": True,
                "torch_threads": 1,
            },
            "checkpoints": checkpoint_records,
        }
        manifest["manifest_sha256"] = canonical_json_sha256(manifest)
        encoded = (
            json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        (partial / "manifest.json").write_bytes(encoded)
        os.replace(partial, output)
        partial = None
    except (
        CheckpointBuildError,
        KeyError,
        OSError,
        Q2CapacityAuditError,
        RuntimeError,
        subprocess.CalledProcessError,
        ValueError,
    ) as exc:
        if partial is not None and partial.exists():
            shutil.rmtree(partial)
        parser.error(str(exc))
    print(f"Q2_CAPACITY_CHECKPOINT_BUNDLE={output}")
    print(
        f"Q2_CAPACITY_CHECKPOINT_MANIFEST_SHA256={sha256_file(output / 'manifest.json')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
