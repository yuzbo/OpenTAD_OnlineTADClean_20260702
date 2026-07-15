import hashlib
import json

import pytest

import opentad.utils.evidence_bundle as evidence_bundle_module
from opentad.utils.evidence_bundle import (
    EvidenceBundleError,
    bundle_file_reference,
    read_verified_bundle_bytes,
    resolve_bundle_path,
)
from opentad.utils.full_petal_attestation import _sign_payload, generate_private_key
from opentad.utils.full_petal_training_evidence import (
    OptimizerEventTraceRecorder,
    TrainingEvidenceError,
    derive_training_cost,
    load_training_trace,
    persist_training_trace,
    verify_formal_run_manifest,
)


def _event(index, *, elapsed=None):
    return {
        "event_id": f"event-{index}",
        "episode_id": f"video-{index}",
        "input_tokens": 64,
        "effective_batch_size": 1,
        "world_size": 2,
        "elapsed_seconds": float(elapsed if elapsed is not None else index + 1),
        "peak_memory_bytes": 1024 + index,
        "precision": "bf16",
        "optimizer_config_sha256": "1" * 64,
        "scheduler_config_sha256": "2" * 64,
        "data_order_sha256": "3" * 64,
        "loss_normalization_sha256": "4" * 64,
        "skipped": False,
    }


def test_training_cost_is_derived_from_committed_optimizer_events(tmp_path):
    trace = tmp_path / "training.jsonl"
    commitment = tmp_path / "training.commitment.json"
    persist_training_trace(trace, commitment, [_event(0), _event(1)])

    cost = derive_training_cost(trace, commitment)

    assert cost["optimizer_events"] == 2
    assert cost["successful_optimizer_events"] == 2
    assert cost["skipped_optimizer_events"] == 0
    assert cost["input_tokens"] == 128
    assert cost["wall_clock_sec"] == 2.0
    assert cost["gpu_hours"] == pytest.approx(4.0 / 3600.0)
    assert cost["scheduler_config_sha256"] == "2" * 64


def test_optimizer_event_recorder_commits_exact_runtime_events(tmp_path):
    timestamps = iter((10.0, 10.5, 12.0))
    peak_memory = iter((2048, 4096))
    recorder = OptimizerEventTraceRecorder(
        precision="bf16",
        effective_batch_size=1,
        world_size=1,
        optimizer_config_sha256="1" * 64,
        scheduler_config_sha256="2" * 64,
        data_order_sha256="3" * 64,
        loss_normalization_sha256="4" * 64,
        clock=lambda: next(timestamps),
        peak_memory_reader=lambda: next(peak_memory),
    )

    first = recorder.record(
        epoch=0,
        episode_id="video-a",
        input_tokens=128,
        skipped=False,
    )
    second = recorder.record(
        epoch=0,
        episode_id="video-b",
        input_tokens=32,
        skipped=True,
    )

    assert first["event_id"] == "optimizer-event-00000000"
    assert first["episode_id"] == "epoch=0|episode=video-a|sequence=0"
    assert first["elapsed_seconds"] == 0.5
    assert first["peak_memory_bytes"] == 2048
    assert second["event_id"] == "optimizer-event-00000001"
    assert second["elapsed_seconds"] == 2.0
    assert second["skipped"] is True

    trace = tmp_path / "formal_training_trace.jsonl"
    commitment = tmp_path / "formal_training_trace.commitment.json"
    recorder.persist(trace, commitment)
    rows = load_training_trace(trace, commitment)
    cost = derive_training_cost(trace, commitment)

    assert len(rows) == 2
    assert cost["optimizer_events"] == 2
    assert cost["successful_optimizer_events"] == 1
    assert cost["skipped_optimizer_events"] == 1
    assert cost["input_tokens"] == 160
    assert cost["wall_clock_sec"] == 2.0
    assert cost["peak_vram_gb"] == 4096 / float(1024**3)


def test_optimizer_event_recorder_refuses_empty_or_overwritten_evidence(tmp_path):
    timestamps = iter((3.0, 4.0))
    recorder = OptimizerEventTraceRecorder(
        precision="fp32",
        effective_batch_size=1,
        world_size=1,
        optimizer_config_sha256="1" * 64,
        scheduler_config_sha256="2" * 64,
        data_order_sha256="3" * 64,
        loss_normalization_sha256="4" * 64,
        clock=lambda: next(timestamps),
    )
    trace = tmp_path / "trace.jsonl"
    commitment = tmp_path / "trace.commitment.json"

    with pytest.raises(TrainingEvidenceError, match="empty"):
        recorder.persist(trace, commitment)

    recorder.record(
        epoch=0,
        episode_id="video-a",
        input_tokens=8,
        skipped=False,
    )
    recorder.persist(trace, commitment)
    with pytest.raises(TrainingEvidenceError, match="overwrite"):
        recorder.persist(trace, commitment)


def test_training_tail_commitment_rejects_valid_prefix_truncation(tmp_path):
    trace = tmp_path / "training.jsonl"
    commitment = tmp_path / "training.commitment.json"
    persist_training_trace(trace, commitment, [_event(0), _event(1)])
    first = trace.read_text(encoding="utf-8").splitlines()[0]
    trace.write_text(first + "\n", encoding="utf-8")

    with pytest.raises(TrainingEvidenceError, match="hash|tail|truncated"):
        load_training_trace(trace, commitment)


def test_split_evidence_publication_rolls_back_before_commitment(monkeypatch, tmp_path):
    trace = tmp_path / "training.jsonl"
    commitment = tmp_path / "training.commitment.json"
    real_link = evidence_bundle_module.os.link
    calls = 0

    def fail_commitment_link(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected commitment publication failure")
        return real_link(source, destination)

    monkeypatch.setattr(evidence_bundle_module.os, "link", fail_commitment_link)
    with pytest.raises(TrainingEvidenceError, match="publish evidence pair"):
        persist_training_trace(trace, commitment, [_event(0)])

    assert not trace.exists()
    assert not commitment.exists()
    assert not list(tmp_path.glob("*.staged"))


def test_formal_run_manifest_requires_trusted_signature_and_artifact_hashes(tmp_path):
    private_key = tmp_path / "formal.pem"
    public_key = generate_private_key(private_key)
    artifact = tmp_path / "artifact.json"
    artifact.write_text('{"locked":true}\n', encoding="utf-8")
    signed = _sign_payload(
        {
            "schema_version": "full-petal-formal-run-manifest-v1",
            "claim": "C1",
            "variant": "fixed",
            "seed": 705,
            "commit_sha": "a" * 40,
            "protocol_sha256": "b" * 64,
            "artifacts": {
                "checkpoint": {
                    "path": artifact.name,
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                }
            },
        },
        private_key_path=private_key,
        key_id="formal-test",
        role="formal-run",
    )
    trust_root = {"key_id": "formal-test", "public_key": public_key}

    assert verify_formal_run_manifest(
        signed, trust_root=trust_root, base_dir=tmp_path
    )["seed"] == 705

    forged = json.loads(json.dumps(signed))
    forged["seed"] = 706
    with pytest.raises(TrainingEvidenceError, match="signature"):
        verify_formal_run_manifest(forged, trust_root=trust_root, base_dir=tmp_path)

    artifact.write_text('{"locked":false}\n', encoding="utf-8")
    with pytest.raises(TrainingEvidenceError, match="hash mismatch"):
        verify_formal_run_manifest(signed, trust_root=trust_root, base_dir=tmp_path)


def test_evidence_bundle_rejects_absolute_parent_and_symlink_escape(tmp_path):
    inside = tmp_path / "inside.json"
    inside.write_text("{}\n", encoding="utf-8")
    outside = tmp_path.parent / f"{tmp_path.name}-outside.json"
    outside.write_text("{}\n", encoding="utf-8")

    with pytest.raises(EvidenceBundleError, match="relative|canonical POSIX"):
        resolve_bundle_path(str(inside.resolve()), tmp_path)
    with pytest.raises(EvidenceBundleError, match="relative"):
        resolve_bundle_path("../outside.json", tmp_path)

    link = tmp_path / "linked.json"
    try:
        link.symlink_to(outside)
    except OSError:
        return
    else:
        with pytest.raises(EvidenceBundleError, match="symbolic link|outside"):
            resolve_bundle_path(link.name, tmp_path)


def test_checkpoint_bytes_are_read_once_against_the_signed_reference(tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint-v1")
    reference = bundle_file_reference(checkpoint, tmp_path, "checkpoint")

    path, bound_bytes = read_verified_bundle_bytes(reference, tmp_path, "checkpoint")
    checkpoint.write_bytes(b"checkpoint-v2")

    assert path == checkpoint.resolve()
    assert bound_bytes == b"checkpoint-v1"
    with pytest.raises(EvidenceBundleError, match="bytes differ"):
        read_verified_bundle_bytes(reference, tmp_path, "checkpoint")
