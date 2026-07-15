#!/usr/bin/env python3
"""Exclusively publish and submit the exact same Full PETAL Slurm bytes."""

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.utils.evidence_bundle import (  # noqa: E402
    EvidenceBundleError,
    publish_exclusive_file,
)


TRUSTED_SBATCH = Path("/usr/bin/sbatch")


class SlurmScriptSubmissionError(RuntimeError):
    pass


def _trusted_submitter(payload, submission_cwd):
    if not TRUSTED_SBATCH.is_file():
        raise SlurmScriptSubmissionError(
            f"trusted sbatch executable is unavailable: {TRUSTED_SBATCH}"
        )
    try:
        return subprocess.run(
            [str(TRUSTED_SBATCH)],
            input=payload,
            cwd=submission_cwd,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SlurmScriptSubmissionError(
            f"trusted sbatch submission failed: {exc}"
        ) from exc


def publish_and_submit(
    script_path,
    payload,
    *,
    submission_cwd,
    submitter=None,
):
    """Publish with exclusive creation, then submit those in-memory bytes."""

    script_path = Path(script_path)
    submission_cwd = Path(submission_cwd).resolve(strict=True)
    if not submission_cwd.is_dir():
        raise SlurmScriptSubmissionError("submission cwd must be a directory")
    if script_path.name != "job.sbatch" or not script_path.parent.is_dir():
        raise SlurmScriptSubmissionError(
            "Slurm script must be <existing ticket root>/job.sbatch"
        )
    if not isinstance(payload, bytes) or not payload.startswith(b"#!/usr/bin/env bash\n"):
        raise SlurmScriptSubmissionError("Slurm payload must be a complete Bash script")
    if b"\x00" in payload:
        raise SlurmScriptSubmissionError("Slurm payload cannot contain NUL bytes")

    publish_exclusive_file(script_path, payload)
    digest = hashlib.sha256(payload).hexdigest()
    runner = _trusted_submitter if submitter is None else submitter
    result = runner(payload, submission_cwd)
    return result, digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--submission-cwd", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        payload = sys.stdin.buffer.read()
        result, digest = publish_and_submit(
            args.output,
            payload,
            submission_cwd=args.submission_cwd,
        )
    except (EvidenceBundleError, OSError, SlurmScriptSubmissionError) as exc:
        parser.error(str(exc))
    if result.stdout:
        sys.stdout.buffer.write(result.stdout)
        if not result.stdout.endswith(b"\n"):
            sys.stdout.buffer.write(b"\n")
    if result.stderr:
        sys.stderr.buffer.write(result.stderr)
        if not result.stderr.endswith(b"\n"):
            sys.stderr.buffer.write(b"\n")
    print(f"FULL_PETAL_SLURM_SCRIPT_SHA256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SlurmScriptSubmissionError",
    "publish_and_submit",
]
