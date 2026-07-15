import json
from pathlib import Path
import shutil

import pytest

import opentad.utils.full_petal_launch as launch_module
import opentad.utils.full_petal_identity as identity_module
import opentad.utils.full_petal_role_signing as role_signing_module
import opentad.utils.full_petal_b0 as b0_module
from tests.full_petal_attestation_fixture import committed_optimizer_envelope
from opentad.utils.full_petal_attestation import generate_private_key
from opentad.utils.full_petal_role_signing import (
    sign_b0_evidence,
    sign_independent_review,
    sign_posix_b0_leaf,
)
from opentad.utils.full_petal_b0 import (
    B0_AUDIT_REPORT_SCHEMA,
    B0_MANIFEST_SCHEMA,
    B0_POSIX_LEAF_SCHEMA,
    B0_SCHEMA,
    B0_TEST_REPORT_SCHEMA,
    B0EvidenceError,
    canonical_json_sha256,
    validate_b0_evidence,
)
from opentad.utils.full_petal_identity import (
    IdentityError,
    SlurmAllocation,
    build_runtime_identity,
)
from opentad.utils.full_petal_launch import (
    FullPetalLaunchError,
    LAUNCH_CONTRACT_SCHEMA,
    LAUNCH_TICKET_SCHEMA,
    REVIEW_SCHEMA,
    RepositoryState,
    FullPetalLaunchAuthorization,
    build_fixed_step_profile_artifact,
    build_launch_receipt,
    build_launch_ticket,
    persist_launch_receipt,
    runtime_evidence_session,
    resolved_config_sha256,
    sha256_file,
    validate_full_petal_launch,
    verify_launch_receipt,
)
from opentad.utils.full_petal_training_evidence import persist_training_trace


COMMIT = "a" * 40
PROFILE_COMMIT = "9" * 40
REVIEWER = "019f5abd-5104-79b3-882e-354ca796f2c1"
SCOPE = [
    "P0_evidence_chain",
    "training_lifecycle",
    "data_metrics",
    "optimizer_launch",
    "full_B0",
]
SOURCE_SHA = "5" * 64
DATA_IDENTITY = {"identity_sha256": "6" * 64}
TRACE_IDENTITY = {
    "precision": "bf16",
    "effective_batch_size": 1,
    "world_size": 1,
    "optimizer_config_sha256": "1" * 64,
    "scheduler_config_sha256": "2" * 64,
    "data_order_sha256": "3" * 64,
    "loss_normalization_sha256": "4" * 64,
}


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _reference(path, root):
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256_file(path),
    }


def _profile_trace(root, runtime_session):
    trace = root / "profile-optimizer-events.jsonl"
    commitment = root / "profile-optimizer-events.commitment.json"
    persist_training_trace(
        trace,
        commitment,
        [
            (lambda payload: {
                **payload,
                **committed_optimizer_envelope(runtime_session, payload),
            })(
                {
                "event_id": f"optimizer-event-{index:08d}",
                "episode_id": f"profile-episode-{index:08d}",
                "input_tokens": 64,
                "elapsed_seconds": float(index + 1),
                "peak_memory_bytes": 1024 + index,
                "skipped": False,
                **TRACE_IDENTITY,
                }
            )
            for index in range(250)
        ],
    )
    return trace, commitment


def _profile_measurements():
    return {
        "warmup_optimizer_events": 50,
        "measured_optimizer_events": 200,
        "total_optimizer_events": 250,
        "skipped_optimizer_events": 0,
        "measurement_start_after_event_id": "optimizer-event-00000049",
        "measurement_end_event_id": "optimizer-event-00000249",
        "elapsed_seconds": 5.0,
        "peak_memory_bytes": 123,
        "throughput_optimizer_events_per_second": 40.0,
    }


def _keys(root):
    b0_private = root / "keys" / "b0.pem"
    review_private = root / "keys" / "review.pem"
    g0_private = root / "keys" / "g0.pem"
    profile_private = root / "keys" / "profile.pem"
    formal_private = root / "keys" / "formal.pem"
    b0_public = generate_private_key(b0_private)
    review_public = generate_private_key(review_private)
    g0_public = generate_private_key(g0_private)
    profile_public = generate_private_key(profile_private)
    formal_public = generate_private_key(formal_private)
    roots = {
        "b0": {"key_id": "b0-test", "public_key": b0_public},
        "review": {"key_id": REVIEWER, "public_key": review_public},
        "g0": {"key_id": "g0-test", "public_key": g0_public},
        "profile": {"key_id": "profile-test", "public_key": profile_public},
        "formal": {"key_id": "formal-test", "public_key": formal_public},
    }
    return roots, b0_private, review_private, profile_private


def _config(root, roots, *, formal=False):
    root.mkdir(parents=True, exist_ok=True)
    trusted_scontrol = root / "trusted-scontrol"
    trusted_scontrol.write_text("fixture\n", encoding="utf-8")
    cfg = {
        "route_stage": "q2_persistent_binding_one_factor",
        "formal_training_ready": formal,
        "gpu_authorization": (
            "FORMAL_TRAINING_APPROVED" if formal else "BLOCKED_UNTIL_B0_AND_PROFILE"
        ),
        "chunk_size": 64,
        "memory_size": 192,
        "num_slots": 4,
        "model": {"type": "PersistentTrajectoryOnlineDetector", "hidden_dim": 256},
        "solver": {"train": {"batch_size": 1}},
        "profile_contract": {
            "warmup_steps": 50,
            "measured_steps": 200,
            "step_unit": "optimizer_event",
            "world_size": 1,
            "submit_via_slurm_only": True,
        },
        "launch_contract": {
            "schema_version": LAUNCH_CONTRACT_SCHEMA,
            "required_reviewer_id": REVIEWER,
            "required_review_scope": SCOPE,
            "allowed_cfg_overrides": ["work_dir"],
            "require_clean_checkout": True,
            "trusted_scontrol_path": str(trusted_scontrol.resolve()),
            "evidence_trust_model": {
                "schema_version": "full-petal-evidence-trust-model-v1",
                "purpose": "scientific_reproducibility",
                "trusted_computing_base": [
                    "launch_validator",
                    "train_engine",
                    "runtime_evidence_session",
                    "in_process_attestation_key_material",
                    "g0_preregistration_and_audit_runner",
                ],
                "guarantees": [
                    "fail_closed_lifecycle_wiring",
                    "provenance_binding",
                    "post_publication_tamper_evidence",
                ],
                "out_of_scope": [
                    "arbitrary_code_execution_inside_tcb",
                    "in_process_private_key_compromise",
                ],
                "key_compromise_action": "BLOCK_ROTATE_AND_RERUN",
            },
            "attestation_trust_roots": roots,
        },
        "work_dir": str(root / "work"),
    }
    config_path = root / "config.py"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("fixture = True\n", encoding="utf-8")
    return cfg, config_path


@pytest.mark.parametrize(
    "mutation",
    ("missing", "expanded_claim", "hidden_key_compromise"),
)
def test_launch_contract_rejects_changed_evidence_trust_boundary(tmp_path, mutation):
    roots, _, _, _ = _keys(tmp_path)
    cfg, _ = _config(tmp_path, roots)
    trust_model = cfg["launch_contract"]["evidence_trust_model"]
    if mutation == "missing":
        cfg["launch_contract"].pop("evidence_trust_model")
    elif mutation == "expanded_claim":
        trust_model["guarantees"].append("hostile_process_remote_attestation")
    else:
        trust_model["out_of_scope"].remove("in_process_private_key_compromise")

    with pytest.raises(FullPetalLaunchError, match="evidence_trust_model"):
        launch_module._launch_contract(cfg)


def _b0(root, private_key, *, commit):
    test_source = root / "tests" / "test_fixture.py"
    test_source.parent.mkdir(parents=True, exist_ok=True)
    test_source.write_text("def test_one():\n    assert True\n", encoding="utf-8")
    runner_source = root / "tools" / "runner.py"
    runner_source.parent.mkdir(parents=True, exist_ok=True)
    runner_source.write_text("# fixture runner\n", encoding="utf-8")
    evidence_dir = root / f"evidence-{commit[:4]}"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    log = evidence_dir / "suite.log"
    log.write_text("1 passed\n", encoding="utf-8")
    junit = evidence_dir / "suite.xml"
    junit.write_text(
        '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0">'
        '<testcase classname="tests.test_fixture" name="test_one" />'
        "</testsuite></testsuites>\n",
        encoding="utf-8",
    )
    cases = [{"classname": "tests.test_fixture", "name": "test_one"}]
    argv = ["python", "-m", "pytest", "tests/test_fixture.py"]
    manifest = {
        "schema_version": B0_MANIFEST_SCHEMA,
        "suite_order": ["all_contracts"],
        "suites": [
            {
                "name": "all_contracts",
                "runner": "isolated_torch",
                "canonical_argv": argv,
                "test_files": [
                    {
                        "path": "tests/test_fixture.py",
                        "sha256": sha256_file(test_source),
                        "test_functions": ["test_one"],
                    }
                ],
            }
        ],
        "runner_sources": [
            {"path": "tools/runner.py", "sha256": sha256_file(runner_source)}
        ],
        "audit_checks": ["python_syntax", "git_diff_check", "repository_clean_after"],
    }
    manifest_path = _write_json(evidence_dir / "manifest.json", manifest)
    report_path = _write_json(
        evidence_dir / "test-report.json",
        {
            "schema_version": B0_TEST_REPORT_SCHEMA,
            "status": "PASS",
            "commit_sha": commit,
            "manifest_sha256": sha256_file(manifest_path),
            "collected": 1,
            "passed": 1,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "suites": [
                {
                    "name": "all_contracts",
                    "status": "PASS",
                    "canonical_argv": argv,
                    "python_executable": "python",
                    "collected": 1,
                    "passed": 1,
                    "failed": 0,
                    "errors": 0,
                    "skipped": 0,
                    "log_path": log.name,
                    "log_sha256": sha256_file(log),
                    "junit_path": junit.name,
                    "junit_sha256": sha256_file(junit),
                    "testcase_manifest_sha256": canonical_json_sha256(cases),
                }
            ],
        },
    )
    checks = []
    for name in manifest["audit_checks"]:
        check_log = evidence_dir / f"{name}.log"
        check_log.write_text("PASS\n", encoding="utf-8")
        checks.append(
            {
                "name": name,
                "status": "PASS",
                "canonical_argv": [name],
                "log_path": check_log.name,
                "log_sha256": sha256_file(check_log),
            }
        )
    audit_path = _write_json(
        evidence_dir / "audit-report.json",
        {
            "schema_version": B0_AUDIT_REPORT_SCHEMA,
            "status": "PASS",
            "commit_sha": commit,
            "manifest_sha256": sha256_file(manifest_path),
            "blocking_findings": 0,
            "protocol_violations": 0,
            "checks": checks,
        },
    )
    posix_dir = evidence_dir / "posix"
    posix_dir.mkdir()
    posix_log = posix_dir / log.name
    posix_log.write_bytes(log.read_bytes())
    posix_junit = posix_dir / junit.name
    posix_junit.write_bytes(junit.read_bytes())
    posix_leaf = _write_json(
        posix_dir / "posix-b0.json",
        sign_posix_b0_leaf(
            {
                "schema_version": B0_POSIX_LEAF_SCHEMA,
                "status": "PASS",
                "commit_sha": commit,
                "manifest_sha256": sha256_file(manifest_path),
                "platform": {
                    "os_name": "posix",
                    "sys_platform": "linux",
                    "machine": "x86_64",
                    "python_version": "3.11.0",
                    "torch_version": "2.6.0",
                },
                "repository_clean_before": True,
                "repository_clean_after": True,
                "collected": 1,
                "passed": 1,
                "failed": 0,
                "errors": 0,
                "skipped": 0,
                "suites": [
                    {
                        "name": "all_contracts",
                        "status": "PASS",
                        "canonical_argv": argv,
                        "python_executable": "/usr/bin/python3",
                        "collected": 1,
                        "passed": 1,
                        "failed": 0,
                        "errors": 0,
                        "skipped": 0,
                        "log_path": posix_log.name,
                        "log_sha256": sha256_file(posix_log),
                        "junit_path": posix_junit.name,
                        "junit_sha256": sha256_file(posix_junit),
                        "testcase_manifest_sha256": canonical_json_sha256(cases),
                    }
                ],
            },
            private_key_path=private_key,
            key_id="b0-test",
        ),
    )
    signed = sign_b0_evidence(
        {
            "schema_version": B0_SCHEMA,
            "status": "PASS",
            "commit_sha": commit,
            "test_count": 1,
            "blocking_findings": 0,
            "protocol_violations": 0,
            "manifest_path": manifest_path.name,
            "manifest_sha256": sha256_file(manifest_path),
            "test_report_path": report_path.name,
            "test_report_sha256": sha256_file(report_path),
            "audit_report_path": audit_path.name,
            "audit_report_sha256": sha256_file(audit_path),
            "posix_leaf_path": posix_leaf.relative_to(evidence_dir).as_posix(),
            "posix_leaf_sha256": sha256_file(posix_leaf),
        },
        private_key_path=private_key,
        key_id="b0-test",
    )
    return _write_json(evidence_dir / "b0.json", signed), log


def _review(root, private_key, b0_path, *, commit, reviewer=REVIEWER, verdict="PASS"):
    signed = sign_independent_review(
        {
            "schema_version": REVIEW_SCHEMA,
            "reviewer_id": reviewer,
            "reviewed_commit": commit,
            "b0_artifact_sha256": sha256_file(b0_path),
            "scope": SCOPE,
            "verdict": verdict,
            "blocking_findings": [],
            "protocol_violations": [],
        },
        private_key_path=private_key,
        key_id=REVIEWER,
    )
    return _write_json(root / f"review-{commit[:4]}.json", signed)


def _runtime(
    seed=705, *, entrypoint="train", resume_path=None, bundle_root=None
):
    return build_runtime_identity(
        entrypoint=entrypoint,
        seed=seed,
        run_id=0,
        deterministic=True,
        not_eval=False,
        resume_path=resume_path,
        cfg_overrides={},
        bundle_root=bundle_root,
    )


def _ticket(
    root,
    cfg,
    config_path,
    b0_path,
    review_path,
    *,
    commit,
    mode="profile",
    profile=None,
    seed=705,
    entrypoint="train",
    resume_path=None,
):
    payload = {
        "schema_version": LAUNCH_TICKET_SCHEMA,
        "mode": mode,
        "commit_sha": commit,
        "source_tree_sha256": SOURCE_SHA,
        "config_file_sha256": sha256_file(config_path),
        "resolved_config_sha256": resolved_config_sha256(cfg),
        "scientific_config_sha256": resolved_config_sha256(cfg, scientific=True),
        "data_identity": DATA_IDENTITY,
        "runtime_identity": _runtime(
            seed,
            entrypoint=entrypoint,
            resume_path=resume_path,
            bundle_root=root,
        ),
        "b0_evidence": _reference(b0_path, root),
        "review_evidence": _reference(review_path, root),
        "g0_evidence": None,
        "profile_evidence": None if profile is None else _reference(profile, root),
    }
    return _write_json(
        root / f"{mode}-{entrypoint}-ticket-{commit[:4]}.json", payload
    )


def _allocation(root):
    return SlurmAllocation(
        job_id="12345",
        state="RUNNING",
        user="fixture-user",
        nodes=1,
        tasks=1,
        gpus=1,
        command="python tools/train.py",
        work_dir=str(root),
    )


def _patch_identities(
    monkeypatch, root, *, commit=COMMIT, authorization_error=None
):
    monkeypatch.setattr(
        launch_module,
        "scientific_source_identity",
        lambda repository_root, scientific_config_sha256: SOURCE_SHA,
    )
    monkeypatch.setattr(
        launch_module,
        "build_data_identity",
        lambda cfg: DATA_IDENTITY,
    )

    def validate(identity, cfg):
        if identity != DATA_IDENTITY:
            raise IdentityError("data mismatch")
        return DATA_IDENTITY

    monkeypatch.setattr(launch_module, "validate_data_identity", validate)
    monkeypatch.setattr(
        identity_module,
        "derive_training_trace_identity",
        lambda cfg, data_identity, seed, world_size, **kwargs: dict(TRACE_IDENTITY),
    )
    monkeypatch.setattr(
        launch_module,
        "inspect_slurm_allocation",
        lambda job_id, scontrol_path=None: _allocation(root),
    )
    monkeypatch.setattr(
        launch_module,
        "inspect_repository",
        lambda repository_root: RepositoryState(commit, True),
    )
    if authorization_error is None:
        monkeypatch.setattr(
            launch_module,
            "authorization_only_diff",
            lambda repository_root, profile_commit, formal_commit: "7" * 64,
        )
    else:
        def reject(*args, **kwargs):
            raise IdentityError(authorization_error)

        monkeypatch.setattr(launch_module, "authorization_only_diff", reject)


def _authorize(
    root,
    cfg,
    config_path,
    ticket_path,
    profile_private,
    *,
    commit=COMMIT,
    mode="profile",
    seed=705,
    entrypoint="train",
    resume_path=None,
    **kwargs,
):
    return validate_full_petal_launch(
        cfg,
        config_path,
        mode=mode,
        ticket_path=ticket_path,
        entrypoint=entrypoint,
        seed=seed,
        run_id=0,
        deterministic=True,
        not_eval=False,
        resume_path=resume_path,
        cfg_overrides={},
        environ=kwargs.pop("environ", {"SLURM_JOB_ID": "12345", "WORLD_SIZE": "1"}),
        repository_root=root,
        repository_state=kwargs.pop("repository_state", RepositoryState(commit, True)),
        slurm_allocation=kwargs.pop("slurm_allocation", _allocation(root)),
        expected_slurm_user="fixture-user",
        execution_signing_key_path=profile_private,
        **kwargs,
    )


def _profile_setup(root, monkeypatch, *, commit=COMMIT):
    roots, b0_private, review_private, profile_private = _keys(root)
    cfg, config_path = _config(root, roots)
    b0_path, suite_log = _b0(root, b0_private, commit=commit)
    review_path = _review(root, review_private, b0_path, commit=commit)
    ticket_path = _ticket(
        root, cfg, config_path, b0_path, review_path, commit=commit
    )
    _patch_identities(monkeypatch, root, commit=commit)
    return {
        "roots": roots,
        "b0_private": b0_private,
        "review_private": review_private,
        "profile_private": profile_private,
        "cfg": cfg,
        "config_path": config_path,
        "b0": b0_path,
        "review": review_path,
        "ticket": ticket_path,
        "suite_log": suite_log,
    }


def test_non_full_petal_config_does_not_require_a_ticket(tmp_path):
    assert validate_full_petal_launch(
        {"model": {"type": "OtherDetector"}}, tmp_path / "config.py"
    ) is None


def test_production_has_no_dictionary_to_launch_receipt_signer():
    assert not hasattr(role_signing_module, "sign_launch_receipt")
    assert not hasattr(launch_module, "_issue_authorization")


def test_manual_or_mutated_launch_authorization_is_rejected(tmp_path, monkeypatch):
    with pytest.raises(FullPetalLaunchError, match="only be issued"):
        FullPetalLaunchAuthorization()
    forged = object.__new__(FullPetalLaunchAuthorization)
    with pytest.raises(FullPetalLaunchError, match="active validator-issued"):
        runtime_evidence_session(forged)

    setup = _profile_setup(tmp_path, monkeypatch)
    authorization = _authorize(
        tmp_path,
        setup["cfg"],
        setup["config_path"],
        setup["ticket"],
        setup["profile_private"],
    )
    authorization.execution_session["session_id"] = "f" * 64
    with pytest.raises(FullPetalLaunchError, match="changed after validation"):
        build_launch_receipt(
            authorization, private_key_path=setup["profile_private"]
        )


def test_profile_requires_signed_b0_same_reviewer_and_bound_runtime(tmp_path, monkeypatch):
    setup = _profile_setup(tmp_path, monkeypatch)

    authorization = _authorize(
        tmp_path,
        setup["cfg"],
        setup["config_path"],
        setup["ticket"],
        setup["profile_private"],
    )

    assert authorization.mode == "profile"
    assert authorization.commit_sha == COMMIT
    assert authorization.source_tree_sha256 == SOURCE_SHA
    assert authorization.data_identity_sha256 == DATA_IDENTITY["identity_sha256"]
    assert authorization.total_optimizer_events == 250
    assert authorization.slurm_allocation.gpus == 1


def test_launch_receipt_binds_ticket_runtime_and_slurm_without_overwrite(tmp_path, monkeypatch):
    setup = _profile_setup(tmp_path, monkeypatch)
    authorization = _authorize(
        tmp_path,
        setup["cfg"],
        setup["config_path"],
        setup["ticket"],
        setup["profile_private"],
    )

    signed_receipt = build_launch_receipt(
        authorization,
        private_key_path=setup["profile_private"],
    )
    receipt = verify_launch_receipt(
        signed_receipt,
        trust_root=setup["roots"]["profile"],
    )
    assert receipt["launch_ticket"] == {
        "path": setup["ticket"].name,
        "sha256": sha256_file(setup["ticket"]),
    }
    assert receipt["runtime_identity_sha256"] == authorization.runtime_identity_sha256
    assert receipt["slurm_job_id"] == "12345"
    assert receipt["slurm_allocation"]["state"] == "RUNNING"

    output = tmp_path / "receipt.json"
    assert persist_launch_receipt(
        authorization,
        private_key_path=setup["profile_private"],
        output_path=output,
    ) == output.resolve()
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted == signed_receipt
    with pytest.raises(FullPetalLaunchError, match="already issued"):
        persist_launch_receipt(
            authorization,
            private_key_path=setup["profile_private"],
            output_path=output,
        )


def test_unsigned_self_consistent_review_is_rejected(tmp_path, monkeypatch):
    setup = _profile_setup(tmp_path, monkeypatch)
    payload = json.loads(setup["review"].read_text(encoding="utf-8"))
    payload.pop("attestation")
    _write_json(setup["review"], payload)
    ticket = json.loads(setup["ticket"].read_text(encoding="utf-8"))
    ticket["review_evidence"] = _reference(setup["review"], tmp_path)
    _write_json(setup["ticket"], ticket)

    with pytest.raises(FullPetalLaunchError, match="lacks a signed attestation"):
        _authorize(
            tmp_path,
            setup["cfg"],
            setup["config_path"],
            setup["ticket"],
            setup["profile_private"],
        )


def test_tampered_b0_leaf_is_rejected_even_when_root_is_unchanged(tmp_path, monkeypatch):
    setup = _profile_setup(tmp_path, monkeypatch)
    setup["suite_log"].write_text("forged PASS\n", encoding="utf-8")

    with pytest.raises(FullPetalLaunchError, match="hash mismatch"):
        _authorize(
            tmp_path,
            setup["cfg"],
            setup["config_path"],
            setup["ticket"],
            setup["profile_private"],
        )


def test_tampered_posix_b0_leaf_is_rejected_even_when_root_is_unchanged(tmp_path):
    roots, b0_private, _, _ = _keys(tmp_path)
    b0_path, _ = _b0(tmp_path, b0_private, commit=COMMIT)
    (b0_path.parent / "posix" / "suite.log").write_text(
        "forged Linux PASS\n", encoding="utf-8"
    )

    with pytest.raises(B0EvidenceError, match="hash mismatch"):
        validate_b0_evidence(
            _reference(b0_path, b0_path.parent),
            base_dir=b0_path.parent,
            expected_commit=COMMIT,
            trust_root=roots["b0"],
            repository_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("replacement", "message"),
    (
        ('"status": "FAIL", "status": "PASS",', "duplicate JSON key"),
        ('"test_count": NaN,', "non-finite JSON number"),
        ('"test_count": 1e309,', "non-finite JSON number"),
    ),
)
def test_b0_root_rejects_ambiguous_or_nonfinite_json(
    tmp_path, replacement, message
):
    roots, b0_private, _, _ = _keys(tmp_path)
    b0_path, _ = _b0(tmp_path, b0_private, commit=COMMIT)
    raw = b0_path.read_text(encoding="utf-8")
    if "status" in replacement:
        raw = raw.replace('"status": "PASS",', replacement, 1)
    else:
        raw = raw.replace('"test_count": 1,', replacement, 1)
    b0_path.write_text(raw, encoding="utf-8")

    with pytest.raises(B0EvidenceError, match=message):
        validate_b0_evidence(
            _reference(b0_path, b0_path.parent),
            base_dir=b0_path.parent,
            expected_commit=COMMIT,
            trust_root=roots["b0"],
            repository_root=tmp_path,
        )


def test_b0_bundle_remains_verifiable_after_directory_relocation(tmp_path):
    roots, b0_private, _, _ = _keys(tmp_path)
    b0_path, _ = _b0(tmp_path, b0_private, commit=COMMIT)
    moved = tmp_path / "relocated-b0"
    shutil.copytree(b0_path.parent, moved)
    moved_b0 = moved / b0_path.name

    artifact = validate_b0_evidence(
        _reference(moved_b0, moved),
        base_dir=moved,
        expected_commit=COMMIT,
        trust_root=roots["b0"],
        repository_root=tmp_path,
    )

    assert artifact["status"] == "PASS"
    assert Path(artifact["artifact_path"]) == moved_b0.resolve()


def test_b0_validation_consumes_the_same_bytes_after_verified_path_swap(
    tmp_path, monkeypatch
):
    roots, b0_private, _, _ = _keys(tmp_path)
    b0_path, _ = _b0(tmp_path, b0_private, commit=COMMIT)
    reference = _reference(b0_path, b0_path.parent)
    real_reader = b0_module.read_verified_path_bytes
    swapped = False

    def swap_after_verified_read(path, expected_sha256, label):
        nonlocal swapped
        resolved, payload = real_reader(path, expected_sha256, label)
        if resolved == b0_path.resolve() and not swapped:
            swapped = True
            b0_path.write_text('{"forged":true}\n', encoding="utf-8")
        return resolved, payload

    monkeypatch.setattr(
        b0_module, "read_verified_path_bytes", swap_after_verified_read
    )
    artifact = validate_b0_evidence(
        reference,
        base_dir=b0_path.parent,
        expected_commit=COMMIT,
        trust_root=roots["b0"],
        repository_root=tmp_path,
    )

    assert swapped is True
    assert artifact["status"] == "PASS"
    assert b0_path.read_text(encoding="utf-8") == '{"forged":true}\n'


def test_profile_rejects_runtime_slurm_dirty_and_signing_key_mismatch(tmp_path, monkeypatch):
    setup = _profile_setup(tmp_path, monkeypatch)
    wrong_private = tmp_path / "keys" / "wrong.pem"
    generate_private_key(wrong_private)
    with pytest.raises(FullPetalLaunchError, match="does not match the trusted public key"):
        _authorize(
            tmp_path,
            setup["cfg"],
            setup["config_path"],
            setup["ticket"],
            wrong_private,
        )
    with pytest.raises(FullPetalLaunchError, match="SLURM_JOB_ID"):
        _authorize(
            tmp_path,
            setup["cfg"],
            setup["config_path"],
            setup["ticket"],
            setup["profile_private"],
            environ={"WORLD_SIZE": "1"},
        )
    with pytest.raises(FullPetalLaunchError, match="clean git checkout"):
        _authorize(
            tmp_path,
            setup["cfg"],
            setup["config_path"],
            setup["ticket"],
            setup["profile_private"],
            repository_state=RepositoryState(COMMIT, False, "M config.py"),
        )
    with pytest.raises(FullPetalLaunchError, match="runtime argv/seed"):
        _authorize(
            tmp_path,
            setup["cfg"],
            setup["config_path"],
            setup["ticket"],
            setup["profile_private"],
            seed=706,
        )


def test_ticket_builder_uses_the_same_signed_prerequisites(tmp_path, monkeypatch):
    setup = _profile_setup(tmp_path, monkeypatch)

    ticket = build_launch_ticket(
        setup["cfg"],
        setup["config_path"],
        mode="profile",
        b0_path=setup["b0"],
        review_path=setup["review"],
        repository_root=tmp_path,
        entrypoint="train",
        seed=705,
        run_id=0,
        deterministic=True,
        not_eval=False,
        resume_path=None,
        cfg_overrides={},
        bundle_root=tmp_path,
        repository_state=RepositoryState(COMMIT, True),
    )

    assert ticket["source_tree_sha256"] == SOURCE_SHA
    assert ticket["data_identity"] == DATA_IDENTITY
    assert ticket["runtime_identity"] == _runtime()


def test_crs_profile_ticket_cannot_be_built_without_signed_g0_pass(tmp_path, monkeypatch):
    setup = _profile_setup(tmp_path, monkeypatch)
    setup["cfg"]["crs_eps_contract"] = {"draws_per_video": 4}
    setup["cfg"]["profile_contract"]["step_unit"] = "video_group_optimizer_event"
    setup["cfg"]["profile_contract"]["required_workload_denominators"] = list(
        launch_module._CRS_EPS_PROFILE_DENOMINATORS
    )

    with pytest.raises(FullPetalLaunchError, match="requires signed G0 PASS"):
        build_launch_ticket(
            setup["cfg"],
            setup["config_path"],
            mode="profile",
            b0_path=setup["b0"],
            review_path=setup["review"],
            repository_root=tmp_path,
            entrypoint="train",
            seed=705,
            run_id=0,
            deterministic=True,
            not_eval=False,
            resume_path=None,
            cfg_overrides={},
            bundle_root=tmp_path,
            repository_state=RepositoryState(COMMIT, True),
        )


def test_formal_ticket_rejects_resume_until_trace_continuation_exists(tmp_path, monkeypatch):
    setup = _profile_setup(tmp_path, monkeypatch)
    formal_cfg, formal_config = _config(tmp_path / "formal", setup["roots"], formal=True)
    checkpoint = tmp_path / "resume.pth"
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(FullPetalLaunchError, match="trace continuation"):
        build_launch_ticket(
            formal_cfg,
            formal_config,
            mode="formal",
            b0_path=setup["b0"],
            review_path=setup["review"],
            profile_path=None,
            repository_root=tmp_path,
            entrypoint="train",
            seed=705,
            run_id=0,
            deterministic=True,
            not_eval=False,
            resume_path=checkpoint,
            cfg_overrides={},
            bundle_root=tmp_path,
            repository_state=RepositoryState(COMMIT, True),
        )


def test_profile_artifact_is_signed_and_binds_exact_authorization(tmp_path, monkeypatch):
    setup = _profile_setup(tmp_path, monkeypatch)
    authorization = _authorize(
        tmp_path,
        setup["cfg"],
        setup["config_path"],
        setup["ticket"],
        setup["profile_private"],
    )
    persist_launch_receipt(
        authorization, private_key_path=setup["profile_private"]
    )
    trace, commitment = _profile_trace(
        tmp_path, runtime_evidence_session(authorization)
    )
    artifact = build_fixed_step_profile_artifact(
        authorization,
        setup["cfg"],
        optimizer_event_trace_path=trace,
        optimizer_event_commitment_path=commitment,
        bundle_root=tmp_path,
        precision="bf16",
        gpu_name="Fixture GPU",
        torch_version="2.6.0",
        cuda_version="12.4",
        profiler_measurements=_profile_measurements(),
    )

    assert artifact["attestation"]["role"] == "fixed-step-profile"
    assert artifact["source_tree_sha256"] == SOURCE_SHA
    assert artifact["runtime_identity_sha256"] == authorization.runtime_identity_sha256
    assert artifact["measurements"]["total_optimizer_events"] == 250
    assert artifact["measurements"]["elapsed_seconds"] == 5.0
    assert artifact["measurements"]["peak_memory_bytes"] == 123


def test_formal_launch_accepts_only_later_authorization_only_commit(tmp_path, monkeypatch):
    roots, b0_private, review_private, profile_private = _keys(tmp_path)
    profile_root = tmp_path / "profile-source"
    profile_cfg, profile_config = _config(profile_root, roots)
    profile_b0, _ = _b0(tmp_path, b0_private, commit=PROFILE_COMMIT)
    profile_review = _review(
        tmp_path, review_private, profile_b0, commit=PROFILE_COMMIT
    )
    profile_ticket = _ticket(
        tmp_path,
        profile_cfg,
        profile_config,
        profile_b0,
        profile_review,
        commit=PROFILE_COMMIT,
    )
    _patch_identities(monkeypatch, tmp_path, commit=PROFILE_COMMIT)
    profile_authorization = _authorize(
        tmp_path,
        profile_cfg,
        profile_config,
        profile_ticket,
        profile_private,
        commit=PROFILE_COMMIT,
    )
    persist_launch_receipt(
        profile_authorization, private_key_path=profile_private
    )
    trace, commitment = _profile_trace(
        tmp_path, runtime_evidence_session(profile_authorization)
    )
    profile_payload = build_fixed_step_profile_artifact(
        profile_authorization,
        profile_cfg,
        optimizer_event_trace_path=trace,
        optimizer_event_commitment_path=commitment,
        bundle_root=tmp_path,
        precision="bf16",
        gpu_name="Fixture GPU",
        torch_version="2.6.0",
        cuda_version="12.4",
        profiler_measurements=_profile_measurements(),
    )
    profile_path = _write_json(tmp_path / "profile.json", profile_payload)

    formal_root = tmp_path / "formal-source"
    formal_cfg, formal_config = _config(formal_root, roots, formal=True)
    formal_b0, _ = _b0(tmp_path, b0_private, commit=COMMIT)
    formal_review = _review(tmp_path, review_private, formal_b0, commit=COMMIT)
    formal_ticket = _ticket(
        tmp_path,
        formal_cfg,
        formal_config,
        formal_b0,
        formal_review,
        commit=COMMIT,
        mode="formal",
        profile=profile_path,
    )

    authorization = _authorize(
        tmp_path,
        formal_cfg,
        formal_config,
        formal_ticket,
        profile_private,
        mode="formal",
        commit=COMMIT,
    )

    assert authorization.mode == "formal"
    assert authorization.profile_artifact_sha256 == sha256_file(profile_path)

    checkpoint = tmp_path / "formal-eval-checkpoint.pth"
    checkpoint.write_bytes(b"formal checkpoint")
    evaluation_ticket = _ticket(
        tmp_path,
        formal_cfg,
        formal_config,
        formal_b0,
        formal_review,
        commit=COMMIT,
        mode="formal",
        profile=profile_path,
        entrypoint="test",
        resume_path=checkpoint,
    )
    evaluation_authorization = _authorize(
        tmp_path,
        formal_cfg,
        formal_config,
        evaluation_ticket,
        profile_private,
        mode="formal",
        commit=COMMIT,
        entrypoint="test",
        resume_path=checkpoint,
    )

    assert evaluation_authorization.mode == "formal"
    assert json.loads(evaluation_ticket.read_text(encoding="utf-8"))[
        "runtime_identity"
    ]["resume_checkpoint"]["sha256"] == sha256_file(checkpoint)


def test_formal_profile_reuse_rejects_non_authorization_diff(tmp_path, monkeypatch):
    _patch_identities(
        monkeypatch,
        tmp_path,
        authorization_error="profile reuse diff is scientific",
    )
    with pytest.raises(IdentityError, match="scientific"):
        launch_module.authorization_only_diff(tmp_path, PROFILE_COMMIT, COMMIT)
