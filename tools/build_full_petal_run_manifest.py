#!/usr/bin/env python3
"""Build a signed Full PETAL run manifest from immutable run artifacts."""

import argparse
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.utils.full_petal_identity import canonical_json_sha256  # noqa: E402
from opentad.utils.full_petal_training_evidence import (  # noqa: E402
    TrainingEvidenceError,
    build_formal_run_manifest,
)


COMMON_ARTIFACT_ROLES = {
    "ledger",
    "commitment",
    "ground_truth",
    "allowed_videos",
    "evaluator_spec",
    "config",
    "checkpoint",
    "data_identity",
    "training_launch_ticket",
    "training_launch_receipt",
    "evaluation_launch_ticket",
    "evaluation_launch_receipt",
    "training_trace",
    "training_commitment",
}
_GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")


def _load_json(path, label):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingEvidenceError(f"failed to read {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TrainingEvidenceError(f"{label} must contain one JSON object")
    return payload


def _artifacts(values):
    result = {}
    for raw in values:
        role, separator, path = raw.partition("=")
        role = role.strip()
        if not separator or not role or not path.strip():
            raise TrainingEvidenceError("--artifact must use ROLE=PATH")
        if role in result:
            raise TrainingEvidenceError(f"duplicate formal run artifact role: {role}")
        result[role] = Path(path).expanduser().resolve()
    return result


def _outside_repository(path, label):
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError:
        return resolved
    raise TrainingEvidenceError(f"{label} must remain outside the source repository")


def build_from_inputs(
    *,
    claim,
    variant,
    seed,
    protocol_path,
    artifacts,
    private_key_path,
    key_id,
    bundle_root,
):
    expected = set(COMMON_ARTIFACT_ROLES)
    if claim == "C2":
        expected.update(
            {"visual_parameter_trace", "visual_parameter_commitment"}
        )
    elif claim != "C1":
        raise TrainingEvidenceError("formal run claim must be C1 or C2")
    if set(artifacts) != expected:
        raise TrainingEvidenceError(
            "formal run artifact roles differ; "
            f"missing={sorted(expected - set(artifacts))}, "
            f"extra={sorted(set(artifacts) - expected)}"
        )

    protocol = _load_json(protocol_path, "formal run protocol")
    tickets = {
        stage: _load_json(
            artifacts[f"{stage}_launch_ticket"],
            f"formal {stage} launch ticket",
        )
        for stage in ("training", "evaluation")
    }
    commit_shas = {ticket.get("commit_sha") for ticket in tickets.values()}
    if len(commit_shas) != 1:
        raise TrainingEvidenceError("training/evaluation launch ticket commits differ")
    commit_sha = commit_shas.pop()
    if not isinstance(commit_sha, str) or not _GIT_SHA.fullmatch(commit_sha):
        raise TrainingEvidenceError("formal launch ticket commit SHA is invalid")
    for stage, expected_entrypoint in (("training", "train"), ("evaluation", "test")):
        ticket = tickets[stage]
        runtime = ticket.get("runtime_identity")
        if ticket.get("mode") != "formal":
            raise TrainingEvidenceError(f"{stage} run manifest requires a formal ticket")
        if (
            not isinstance(runtime, dict)
            or runtime.get("seed") != seed
            or runtime.get("entrypoint") != expected_entrypoint
        ):
            raise TrainingEvidenceError(
                f"formal {stage} launch ticket runtime differs from the run"
            )

    return build_formal_run_manifest(
        claim=claim,
        variant=variant,
        seed=seed,
        commit_sha=commit_sha,
        protocol_sha256=canonical_json_sha256(protocol),
        artifacts=artifacts,
        bundle_root=bundle_root,
        private_key_path=_outside_repository(private_key_path, "formal signing key"),
        key_id=key_id,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim", choices=("C1", "C2"), required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--artifact", action="append", default=[], metavar="ROLE=PATH")
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--key-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser, parser.parse_args(argv)


def main(argv=None):
    parser, args = parse_args(argv)
    try:
        output = _outside_repository(args.output, "formal run manifest")
        if output.exists():
            raise TrainingEvidenceError(f"refusing to overwrite formal run manifest: {output}")
        payload = build_from_inputs(
            claim=args.claim,
            variant=args.variant,
            seed=args.seed,
            protocol_path=args.protocol,
            artifacts=_artifacts(args.artifact),
            private_key_path=args.private_key,
            key_id=args.key_id,
            bundle_root=output.parent,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
    except (OSError, TrainingEvidenceError) as exc:
        parser.error(str(exc))
    print(f"FULL_PETAL_RUN_MANIFEST={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
