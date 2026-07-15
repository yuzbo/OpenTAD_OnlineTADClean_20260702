import hashlib
import json

import pytest
import torch

import opentad.utils.evidence_bundle as evidence_bundle_module
from opentad.utils.evidence_bundle import (
    EvidenceBundleError,
    bundle_file_reference,
    read_verified_bundle_bytes,
    resolve_bundle_path,
)
from opentad.utils.full_petal_attestation import generate_private_key
from tests.full_petal_attestation_fixture import committed_optimizer_envelope
from opentad.utils.full_petal_role_signing import sign_formal_run
from opentad.utils.full_petal_runtime_attestation import (
    RuntimeAttestationError,
    issue_runtime_session,
)
from opentad.utils.full_petal_training_evidence import (
    OptimizerEventTraceRecorder,
    TrainingEvidenceError,
    derive_training_cost,
    load_training_trace,
    persist_training_trace,
    verify_formal_run_manifest,
)


def _event(index, runtime_session, *, elapsed=None):
    payload = {
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
    return {
        **payload,
        **committed_optimizer_envelope(runtime_session, payload),
    }


class _CommittedTransaction:
    def __init__(self):
        self.pending = True
        self.commits = 0

    def has_pending_online_update(self):
        return self.pending

    def commit_online_update(self):
        assert self.pending
        self.pending = False
        self.commits += 1


def _record_committed(recorder, **kwargs):
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    parameter.grad = torch.tensor(1.0)
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    transaction = _CommittedTransaction()
    proof = recorder.begin_optimizer_boundary(optimizer, transaction)
    recorder.execute_optimizer_step(proof, optimizer)
    recorder.commit_online_transaction(proof, transaction)
    return recorder.record(boundary_proof=proof, skipped=False, **kwargs)


def test_training_cost_is_derived_from_committed_optimizer_events(tmp_path):
    trace = tmp_path / "training.jsonl"
    commitment = tmp_path / "training.commitment.json"
    runtime_session = issue_runtime_session()
    persist_training_trace(
        trace,
        commitment,
        [_event(0, runtime_session), _event(1, runtime_session)],
    )

    cost = derive_training_cost(
        trace, commitment, runtime_binding=runtime_session.binding
    )

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
        runtime_session=issue_runtime_session(),
        clock=lambda: next(timestamps),
        peak_memory_reader=lambda: next(peak_memory),
    )

    first = _record_committed(
        recorder,
        epoch=0,
        episode_id="video-a",
        input_tokens=128,
    )
    second = _record_committed(
        recorder,
        epoch=0,
        episode_id="video-b",
        input_tokens=32,
    )

    assert first["event_id"] == "optimizer-event-00000000"
    assert first["episode_id"] == "epoch=0|episode=video-a|sequence=0"
    assert first["elapsed_seconds"] == 0.5
    assert first["peak_memory_bytes"] == 2048
    assert second["event_id"] == "optimizer-event-00000001"
    assert second["elapsed_seconds"] == 2.0
    assert second["skipped"] is False

    trace = tmp_path / "formal_training_trace.jsonl"
    commitment = tmp_path / "formal_training_trace.commitment.json"
    recorder.persist(trace, commitment)
    binding = recorder._runtime_session.binding
    rows = load_training_trace(
        trace, commitment, runtime_binding=binding, require_contiguous_runtime=True
    )
    cost = derive_training_cost(trace, commitment, runtime_binding=binding)

    assert len(rows) == 2
    assert cost["optimizer_events"] == 2
    assert cost["successful_optimizer_events"] == 2
    assert cost["skipped_optimizer_events"] == 0
    assert cost["input_tokens"] == 160
    assert cost["wall_clock_sec"] == 2.0
    assert cost["peak_vram_gb"] == 4096 / float(1024**3)


def test_optimizer_event_requires_step_and_transaction_commit():
    timestamps = iter(float(value) for value in range(10))
    runtime_session = issue_runtime_session()
    recorder = OptimizerEventTraceRecorder(
        precision="fp32",
        effective_batch_size=1,
        world_size=1,
        optimizer_config_sha256="1" * 64,
        scheduler_config_sha256="2" * 64,
        data_order_sha256="3" * 64,
        loss_normalization_sha256="4" * 64,
        runtime_session=runtime_session,
        clock=lambda: next(timestamps),
    )
    kwargs = {
        "epoch": 0,
        "episode_id": "video-a",
        "input_tokens": 8,
        "skipped": False,
    }

    with pytest.raises(TrainingEvidenceError, match="not active"):
        recorder.record(boundary_proof=object(), **kwargs)
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    parameter.grad = torch.tensor(1.0)
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    transaction = _CommittedTransaction()
    proof = recorder.begin_optimizer_boundary(optimizer, transaction)
    with pytest.raises(TrainingEvidenceError, match="completed optimizer step"):
        recorder.record(boundary_proof=proof, **kwargs)
    other_optimizer = torch.optim.SGD(
        [torch.nn.Parameter(torch.tensor(2.0))], lr=0.1
    )
    with pytest.raises(TrainingEvidenceError, match="different optimizer"):
        recorder.execute_optimizer_step(proof, other_optimizer)
    recorder.execute_optimizer_step(proof, optimizer)
    active = runtime_session._RuntimeEvidenceSession__active_boundary
    step_receipt = active["step_receipt"]
    active["step_receipt"] = step_receipt.__class__(
        payload=step_receipt.payload + b" ", signature=step_receipt.signature
    )
    with pytest.raises(TrainingEvidenceError, match="signature is invalid"):
        recorder.commit_online_transaction(proof, transaction)
    active["step_receipt"] = step_receipt
    with pytest.raises(TrainingEvidenceError, match="committed online transaction"):
        recorder.record(boundary_proof=proof, **kwargs)
    with pytest.raises(TrainingEvidenceError, match="different transaction"):
        recorder.commit_online_transaction(proof, _CommittedTransaction())
    recorder.commit_online_transaction(proof, transaction)
    commit_receipt = active["commit_receipt"]
    event = recorder.record(boundary_proof=proof, **kwargs)

    assert event["event_id"] == "optimizer-event-00000000"
    assert recorder.events == (event,)
    assert parameter.item() == pytest.approx(0.9)
    assert transaction.commits == 1

    replay_optimizer = torch.optim.SGD(
        [torch.nn.Parameter(torch.tensor(3.0))], lr=0.1
    )
    replay_transaction = _CommittedTransaction()
    replay_proof = recorder.begin_optimizer_boundary(
        replay_optimizer, replay_transaction
    )
    replay_state = runtime_session._RuntimeEvidenceSession__active_boundary
    replay_state["step_receipt"] = step_receipt
    replay_state["commit_receipt"] = commit_receipt
    with pytest.raises(TrainingEvidenceError, match="does not match"):
        recorder.record(boundary_proof=replay_proof, **kwargs)
    recorder.abort_optimizer_boundary(replay_proof)


def test_mutable_private_boundary_flags_cannot_mint_optimizer_evidence():
    session = issue_runtime_session()
    recorder = OptimizerEventTraceRecorder(
        precision="fp32",
        effective_batch_size=1,
        world_size=1,
        optimizer_config_sha256="1" * 64,
        scheduler_config_sha256="2" * 64,
        data_order_sha256="3" * 64,
        loss_normalization_sha256="4" * 64,
        runtime_session=session,
    )
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    transaction = _CommittedTransaction()
    proof = recorder.begin_optimizer_boundary(optimizer, transaction)
    state = session._RuntimeEvidenceSession__active_boundary
    state["optimizer_step_completed"] = True
    state["transaction_commit_completed"] = True

    with pytest.raises(TrainingEvidenceError, match="not active"):
        recorder.record(
            epoch=0,
            episode_id="bypass",
            input_tokens=1,
            skipped=False,
            boundary_proof=proof,
        )

    state.pop("optimizer_step_completed")
    state.pop("transaction_commit_completed")
    state["step_receipt"] = True
    state["commit_receipt"] = True
    with pytest.raises(TrainingEvidenceError, match="step receipt"):
        recorder.record(
            epoch=0,
            episode_id="forged-receipts",
            input_tokens=1,
            skipped=False,
            boundary_proof=proof,
        )
    recorder.abort_optimizer_boundary(proof)
    assert parameter.item() == pytest.approx(1.0)
    assert transaction.commits == 0
    assert recorder.events == ()


def test_optimizer_event_rejects_a_scaler_skipped_optimizer_step():
    class SkippingScaler:
        def step(self, optimizer):
            del optimizer

        def update(self):
            pass

    recorder = OptimizerEventTraceRecorder(
        precision="fp16",
        effective_batch_size=1,
        world_size=1,
        optimizer_config_sha256="1" * 64,
        scheduler_config_sha256="2" * 64,
        data_order_sha256="3" * 64,
        loss_normalization_sha256="4" * 64,
        runtime_session=issue_runtime_session(),
    )
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.tensor(1.0))], lr=0.1)
    transaction = _CommittedTransaction()
    proof = recorder.begin_optimizer_boundary(optimizer, transaction)

    with pytest.raises(TrainingEvidenceError, match="optimizer.step was not executed"):
        recorder.execute_optimizer_step(proof, optimizer, scaler=SkippingScaler())

    recorder.abort_optimizer_boundary(proof)
    assert recorder.events == ()


def test_runtime_session_state_rejects_unsigned_mutation():
    session = issue_runtime_session()
    state = session.state_dict()
    state["sequence"] = 1

    with pytest.raises(RuntimeAttestationError, match="state signature is invalid"):
        session.load_state_dict(state)


def test_runtime_session_has_no_direct_boundary_confirmation_shortcut():
    session = issue_runtime_session()

    assert not hasattr(session, "mark_optimizer_step_completed")
    assert not hasattr(session, "mark_transaction_commit_completed")
    assert not hasattr(session, "_confirm_optimizer_step_completed")
    assert not hasattr(session, "_confirm_transaction_commit_completed")


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
        runtime_session=issue_runtime_session(),
        clock=lambda: next(timestamps),
    )
    trace = tmp_path / "trace.jsonl"
    commitment = tmp_path / "trace.commitment.json"

    with pytest.raises(TrainingEvidenceError, match="empty"):
        recorder.persist(trace, commitment)

    _record_committed(
        recorder,
        epoch=0,
        episode_id="video-a",
        input_tokens=8,
    )
    recorder.persist(trace, commitment)
    with pytest.raises(TrainingEvidenceError, match="overwrite"):
        recorder.persist(trace, commitment)


def test_training_tail_commitment_rejects_valid_prefix_truncation(tmp_path):
    trace = tmp_path / "training.jsonl"
    commitment = tmp_path / "training.commitment.json"
    runtime_session = issue_runtime_session()
    persist_training_trace(
        trace,
        commitment,
        [_event(0, runtime_session), _event(1, runtime_session)],
    )
    first = trace.read_text(encoding="utf-8").splitlines()[0]
    trace.write_text(first + "\n", encoding="utf-8")

    with pytest.raises(TrainingEvidenceError, match="hash|tail|truncated"):
        load_training_trace(
            trace, commitment, runtime_binding=runtime_session.binding
        )


def test_self_consistent_trace_from_uncertified_runtime_session_is_rejected(tmp_path):
    certified_session = issue_runtime_session()
    synthetic_session = issue_runtime_session()
    trace = tmp_path / "synthetic.jsonl"
    commitment = tmp_path / "synthetic.commitment.json"
    persist_training_trace(trace, commitment, [_event(0, synthetic_session)])

    with pytest.raises(TrainingEvidenceError, match="session differs"):
        load_training_trace(
            trace,
            commitment,
            runtime_binding=certified_session.binding,
            require_contiguous_runtime=True,
        )


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
    runtime_session = issue_runtime_session()
    with pytest.raises(TrainingEvidenceError, match="publish evidence pair"):
        persist_training_trace(trace, commitment, [_event(0, runtime_session)])

    assert not trace.exists()
    assert not commitment.exists()
    assert not list(tmp_path.glob("*.staged"))


def test_formal_run_manifest_requires_trusted_signature_and_artifact_hashes(tmp_path):
    private_key = tmp_path / "formal.pem"
    public_key = generate_private_key(private_key)
    artifact = tmp_path / "artifact.json"
    artifact.write_text('{"locked":true}\n', encoding="utf-8")
    signed = sign_formal_run(
        {
            "schema_version": "full-petal-formal-run-manifest-v2",
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
    with pytest.raises(EvidenceBundleError, match="hash mismatch"):
        read_verified_bundle_bytes(reference, tmp_path, "checkpoint")
