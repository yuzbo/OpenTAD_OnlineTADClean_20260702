"""Write a small auditable GPU-allocation report for one screen scope."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys


SCHEMA_VERSION = "persistent_binding_resource.v1"


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
        raise ValueError("resource reporting requires a clean worktree")
    return commit


def build_resource_report(
    *,
    repo,
    scope,
    started_unix,
    ended_unix,
    gpu_count,
    gpu_name,
    slurm_job_id,
):
    started_unix = int(started_unix)
    ended_unix = int(ended_unix)
    gpu_count = int(gpu_count)
    if ended_unix < started_unix:
        raise ValueError("resource interval ends before it starts")
    if gpu_count <= 0:
        raise ValueError("gpu_count must be positive")
    elapsed_seconds = ended_unix - started_unix
    allocated_gpu_hours = elapsed_seconds * gpu_count / 3600.0
    if not math.isfinite(allocated_gpu_hours):
        raise ValueError("allocated GPU hours are non-finite")
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": str(scope),
        "started_unix": started_unix,
        "ended_unix": ended_unix,
        "elapsed_seconds": elapsed_seconds,
        "gpu_count": gpu_count,
        "gpu_name": str(gpu_name),
        "allocated_gpu_hours": allocated_gpu_hours,
        "slurm_job_id": str(slurm_job_id),
        "code_commit": _git_commit(repo),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--started-unix", required=True, type=int)
    parser.add_argument("--ended-unix", required=True, type=int)
    parser.add_argument("--gpu-count", default=1, type=int)
    parser.add_argument("--gpu-name", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = build_resource_report(
        repo=args.repo,
        scope=args.scope,
        started_unix=args.started_unix,
        ended_unix=args.ended_unix,
        gpu_count=args.gpu_count,
        gpu_name=args.gpu_name,
        slurm_job_id=args.slurm_job_id,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
