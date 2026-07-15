import importlib.util
import json
from pathlib import Path

import pytest

from opentad.utils.full_petal_attestation import generate_private_key
from opentad.utils.full_petal_identity import canonical_json_sha256
from opentad.utils.full_petal_training_evidence import (
    TrainingEvidenceError,
    verify_formal_run_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_full_petal_run_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_full_petal_run_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _artifacts(root, *, seed=705):
    artifacts = {}
    for role in sorted(MODULE.COMMON_ARTIFACT_ROLES):
        path = root / f"{role}.json"
        if role in {"training_launch_ticket", "evaluation_launch_ticket"}:
            payload = {
                "mode": "formal",
                "commit_sha": "a" * 40,
                "runtime_identity": {
                    "seed": seed,
                    "entrypoint": (
                        "train" if role == "training_launch_ticket" else "test"
                    ),
                },
            }
        else:
            payload = {"role": role}
        artifacts[role] = _write_json(path, payload)
    return artifacts


def test_cli_builds_manifest_from_ticket_commit_and_canonical_protocol(tmp_path):
    private_key = tmp_path / "formal.pem"
    public_key = generate_private_key(private_key)
    protocol = {"decision_cadence": "packet_end", "immutable_emissions": True}
    protocol_path = _write_json(tmp_path / "protocol.json", protocol)
    artifacts = _artifacts(tmp_path)
    output = tmp_path / "signed-run.json"

    argv = [
        "--claim",
        "C1",
        "--variant",
        "fixed",
        "--seed",
        "705",
        "--protocol",
        str(protocol_path),
        "--private-key",
        str(private_key),
        "--key-id",
        "formal-test",
        "--output",
        str(output),
    ]
    for role, path in artifacts.items():
        argv.extend(("--artifact", f"{role}={path}"))

    assert MODULE.main(argv) == 0
    signed = json.loads(output.read_text(encoding="utf-8"))
    body = verify_formal_run_manifest(
        signed,
        trust_root={"key_id": "formal-test", "public_key": public_key},
    )

    assert body["commit_sha"] == "a" * 40
    assert body["protocol_sha256"] == canonical_json_sha256(protocol)
    assert set(body["artifacts"]) == MODULE.COMMON_ARTIFACT_ROLES
    with pytest.raises(SystemExit):
        MODULE.main(argv)


def test_manifest_builder_rejects_missing_role_and_ticket_seed_mismatch(tmp_path):
    private_key = tmp_path / "formal.pem"
    generate_private_key(private_key)
    protocol_path = _write_json(tmp_path / "protocol.json", {})
    artifacts = _artifacts(tmp_path, seed=706)

    with pytest.raises(TrainingEvidenceError, match="runtime differs"):
        MODULE.build_from_inputs(
            claim="C1",
            variant="fixed",
            seed=705,
            protocol_path=protocol_path,
            artifacts=artifacts,
            private_key_path=private_key,
            key_id="formal-test",
        )

    artifacts.pop("checkpoint")
    with pytest.raises(TrainingEvidenceError, match="missing=.*checkpoint"):
        MODULE.build_from_inputs(
            claim="C1",
            variant="fixed",
            seed=706,
            protocol_path=protocol_path,
            artifacts=artifacts,
            private_key_path=private_key,
            key_id="formal-test",
        )
