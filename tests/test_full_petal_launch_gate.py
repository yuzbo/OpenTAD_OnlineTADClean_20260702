import hashlib
import json
from pathlib import Path

import pytest

from opentad.utils.full_petal_launch import (
    B0_AUDIT_REPORT_SCHEMA,
    B0_SCHEMA,
    B0_TEST_REPORT_SCHEMA,
    FullPetalLaunchError,
    FullPetalLaunchAuthorization,
    LAUNCH_CONTRACT_SCHEMA,
    LAUNCH_TICKET_SCHEMA,
    PROFILE_SCHEMA,
    REVIEW_SCHEMA,
    RepositoryState,
    build_fixed_step_profile_artifact,
    build_launch_ticket,
    resolved_config_sha256,
    validate_full_petal_launch,
)


COMMIT = "a" * 40
REVIEWER = "019f5abd-5104-79b3-882e-354ca796f2c1"
SCOPE = [
    "P0_evidence_chain",
    "training_lifecycle",
    "data_metrics",
    "optimizer_launch",
    "full_B0",
]


def _write(path, payload):
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ref(path):
    return {"path": str(path), "sha256": _sha(path)}


def _config(tmp_path, *, formal=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
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
        },
        "work_dir": str(tmp_path / "work"),
    }
    config_path = _write(tmp_path / "config.py", {"fixture": True})
    return cfg, config_path


def _evidence(
    tmp_path,
    cfg,
    *,
    review_verdict="PASS",
    reviewer=REVIEWER,
    commit=COMMIT,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    suite_log = tmp_path / "b0-suite.log"
    suite_log.write_text("256 passed\n", encoding="utf-8")
    suite_junit = tmp_path / "b0-suite.junit.xml"
    suite_junit.write_text(
        '<testsuites><testsuite tests="256" failures="0" errors="0" '
        'skipped="0"/></testsuites>\n',
        encoding="utf-8",
    )
    test_report = _write(
        tmp_path / "b0-tests.json",
        {
            "schema_version": B0_TEST_REPORT_SCHEMA,
            "status": "PASS",
            "commit_sha": commit,
            "collected": 256,
            "passed": 256,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "suites": [
                {
                    "name": "fixture-contracts",
                    "status": "PASS",
                    "command": ["python", "-m", "pytest"],
                    "python_executable": "python",
                    "collected": 256,
                    "passed": 256,
                    "failed": 0,
                    "errors": 0,
                    "skipped": 0,
                    "log_path": str(suite_log),
                    "log_sha256": _sha(suite_log),
                    "junit_path": str(suite_junit),
                    "junit_sha256": _sha(suite_junit),
                }
            ],
        },
    )
    audit_log = tmp_path / "b0-audit-check.log"
    audit_log.write_text("clean\n", encoding="utf-8")
    audit_report = _write(
        tmp_path / "b0-audit.json",
        {
            "schema_version": B0_AUDIT_REPORT_SCHEMA,
            "status": "PASS",
            "commit_sha": commit,
            "blocking_findings": 0,
            "protocol_violations": 0,
            "checks": [
                {
                    "name": "fixture-clean",
                    "status": "PASS",
                    "command": ["git", "status", "--porcelain"],
                    "log_path": str(audit_log),
                    "log_sha256": _sha(audit_log),
                }
            ],
        },
    )
    b0 = _write(
        tmp_path / "b0.json",
        {
            "schema_version": B0_SCHEMA,
            "status": "PASS",
            "commit_sha": commit,
            "test_count": 256,
            "blocking_findings": 0,
            "protocol_violations": 0,
            "test_report_path": str(test_report),
            "test_report_sha256": _sha(test_report),
            "audit_report_path": str(audit_report),
            "audit_report_sha256": _sha(audit_report),
        },
    )
    review = _write(
        tmp_path / "review.json",
        {
            "schema_version": REVIEW_SCHEMA,
            "reviewer_id": reviewer,
            "reviewed_commit": commit,
            "b0_artifact_sha256": _sha(b0),
            "scope": SCOPE,
            "verdict": review_verdict,
            "blocking_findings": [],
            "protocol_violations": [],
        },
    )
    profile_ticket = _write(
        tmp_path / "profile-source-ticket.json",
        {
            "schema_version": LAUNCH_TICKET_SCHEMA,
            "mode": "profile",
            "commit_sha": commit,
            "config_file_sha256": "d" * 64,
            "resolved_config_sha256": "b" * 64,
            "scientific_config_sha256": resolved_config_sha256(
                cfg, scientific=True
            ),
            "b0_evidence": _ref(b0),
            "review_evidence": _ref(review),
            "profile_evidence": None,
        },
    )
    profile = _write(
        tmp_path / "profile.json",
        {
            "schema_version": PROFILE_SCHEMA,
            "status": "PASS",
            "commit_sha": commit,
            "resolved_config_sha256": "b" * 64,
            "scientific_config_sha256": resolved_config_sha256(cfg, scientific=True),
            "launch_ticket_path": str(profile_ticket),
            "launch_ticket_sha256": _sha(profile_ticket),
            "slurm_job_id": "12345",
            "world_size": 1,
            "precision": "bf16",
            "hardware": {
                "gpu_name": "Test GPU",
                "torch_version": "2.6.0",
                "cuda_version": "12.4",
            },
            "dimensions": {
                "batch_size": 1,
                "chunk_size": 64,
                "memory_size": 192,
                "slot_count": 4,
            },
            "measurements": {
                "warmup_optimizer_events": 50,
                "measured_optimizer_events": 200,
                "total_optimizer_events": 250,
                "skipped_optimizer_events": 0,
                "elapsed_seconds": 10.0,
                "peak_memory_bytes": 1024,
                "throughput_optimizer_events_per_second": 20.0,
            },
        },
    )
    return b0, review, profile


def _ticket(tmp_path, cfg, config_path, b0, review, *, mode="profile", profile=None):
    ticket = {
        "schema_version": LAUNCH_TICKET_SCHEMA,
        "mode": mode,
        "commit_sha": COMMIT,
        "config_file_sha256": _sha(config_path),
        "resolved_config_sha256": resolved_config_sha256(cfg),
        "scientific_config_sha256": resolved_config_sha256(cfg, scientific=True),
        "b0_evidence": _ref(b0),
        "review_evidence": _ref(review),
        "profile_evidence": None if profile is None else _ref(profile),
    }
    return _write(tmp_path / f"{mode}-ticket.json", ticket)


def _authorize(cfg, config_path, ticket_path, *, mode="profile", **kwargs):
    return validate_full_petal_launch(
        cfg,
        config_path,
        mode=mode,
        ticket_path=ticket_path,
        entrypoint=kwargs.pop("entrypoint", "train"),
        cfg_override_keys=kwargs.pop("cfg_override_keys", ()),
        environ=kwargs.pop(
            "environ", {"SLURM_JOB_ID": "12345", "WORLD_SIZE": "1"}
        ),
        repository_root=config_path.parent,
        repository_state=kwargs.pop(
            "repository_state", RepositoryState(COMMIT, True)
        ),
        **kwargs,
    )


def test_non_full_petal_config_does_not_require_a_ticket(tmp_path):
    assert validate_full_petal_launch(
        {"model": {"type": "OtherDetector"}},
        tmp_path / "config.py",
    ) is None


def test_profile_requires_verified_b0_and_same_reviewer_pass(tmp_path):
    cfg, config_path = _config(tmp_path)
    b0, review, _ = _evidence(tmp_path, cfg)
    ticket = _ticket(tmp_path, cfg, config_path, b0, review)

    authorization = _authorize(cfg, config_path, ticket)

    assert authorization.mode == "profile"
    assert authorization.total_optimizer_events == 250
    assert authorization.b0_artifact_sha256 == _sha(b0)
    assert authorization.review_artifact_sha256 == _sha(review)


def test_ticket_builder_applies_the_same_authorization_state_gate(tmp_path):
    cfg, config_path = _config(tmp_path)
    b0, review, _ = _evidence(tmp_path, cfg)

    ticket = build_launch_ticket(
        cfg,
        config_path,
        mode="profile",
        b0_path=b0,
        review_path=review,
        repository_root=tmp_path,
        repository_state=RepositoryState(COMMIT, True),
    )
    assert ticket["mode"] == "profile"

    with pytest.raises(FullPetalLaunchError, match="formal_training_ready"):
        build_launch_ticket(
            cfg,
            config_path,
            mode="formal",
            b0_path=b0,
            review_path=review,
            profile_path=tmp_path / "unused-profile.json",
            repository_root=tmp_path,
            repository_state=RepositoryState(COMMIT, True),
        )


@pytest.mark.parametrize(
    ("filename", "message"),
    [
        ("b0-suite.log", "test suite 0 log hash mismatch"),
        ("b0-audit-check.log", "audit check 0 log hash mismatch"),
    ],
)
def test_profile_rejects_tampered_b0_leaf_evidence(tmp_path, filename, message):
    cfg, config_path = _config(tmp_path)
    b0, review, _ = _evidence(tmp_path, cfg)
    ticket = _ticket(tmp_path, cfg, config_path, b0, review)
    (tmp_path / filename).write_text("tampered\n", encoding="utf-8")

    with pytest.raises(FullPetalLaunchError, match=message):
        _authorize(cfg, config_path, ticket)


@pytest.mark.parametrize(
    ("reviewer", "verdict", "message"),
    [
        ("different-reviewer", "PASS", "locked reviewer"),
        (REVIEWER, "REVISE", "not PASS"),
    ],
)
def test_profile_rejects_wrong_reviewer_or_non_pass(
    tmp_path, reviewer, verdict, message
):
    cfg, config_path = _config(tmp_path)
    b0, review, _ = _evidence(
        tmp_path, cfg, review_verdict=verdict, reviewer=reviewer
    )
    ticket = _ticket(tmp_path, cfg, config_path, b0, review)

    with pytest.raises(FullPetalLaunchError, match=message):
        _authorize(cfg, config_path, ticket)


def test_launch_rejects_non_slurm_dirty_or_wrong_world_size(tmp_path):
    cfg, config_path = _config(tmp_path)
    b0, review, _ = _evidence(tmp_path, cfg)
    ticket = _ticket(tmp_path, cfg, config_path, b0, review)

    with pytest.raises(FullPetalLaunchError, match="SLURM_JOB_ID"):
        _authorize(cfg, config_path, ticket, environ={"WORLD_SIZE": "1"})
    with pytest.raises(FullPetalLaunchError, match="world size"):
        _authorize(
            cfg,
            config_path,
            ticket,
            environ={"SLURM_JOB_ID": "1", "WORLD_SIZE": "2"},
        )
    with pytest.raises(FullPetalLaunchError, match="clean git checkout"):
        _authorize(
            cfg,
            config_path,
            ticket,
            repository_state=RepositoryState(COMMIT, False, " M model.py"),
        )


def test_launch_rejects_scientific_config_overrides(tmp_path):
    cfg, config_path = _config(tmp_path)
    b0, review, _ = _evidence(tmp_path, cfg)
    ticket = _ticket(tmp_path, cfg, config_path, b0, review)

    _authorize(cfg, config_path, ticket, cfg_override_keys=("work_dir",))
    with pytest.raises(FullPetalLaunchError, match="scientific --cfg-options"):
        _authorize(cfg, config_path, ticket, cfg_override_keys=("model.hidden_dim",))


def test_formal_training_stays_blocked_before_explicit_config_authorization(tmp_path):
    cfg, config_path = _config(tmp_path)
    b0, review, profile = _evidence(tmp_path, cfg)
    ticket = _ticket(
        tmp_path,
        cfg,
        config_path,
        b0,
        review,
        mode="formal",
        profile=profile,
    )

    with pytest.raises(FullPetalLaunchError, match="formal_training_ready"):
        _authorize(cfg, config_path, ticket, mode="formal")


def test_formal_training_requires_passing_fixed_step_profile(tmp_path):
    cfg, config_path = _config(tmp_path, formal=True)
    b0, review, profile = _evidence(tmp_path, cfg)
    ticket = _ticket(
        tmp_path,
        cfg,
        config_path,
        b0,
        review,
        mode="formal",
        profile=profile,
    )

    authorization = _authorize(cfg, config_path, ticket, mode="formal")
    assert authorization.profile_artifact_sha256 == _sha(profile)

    payload = json.loads(profile.read_text(encoding="utf-8"))
    payload["measurements"]["skipped_optimizer_events"] = 1
    _write(profile, payload)
    ticket = _ticket(
        tmp_path,
        cfg,
        config_path,
        b0,
        review,
        mode="formal",
        profile=profile,
    )
    with pytest.raises(FullPetalLaunchError, match="skipped optimizer"):
        _authorize(cfg, config_path, ticket, mode="formal")


def test_formal_launch_rejects_profile_with_tampered_source_ticket(tmp_path):
    cfg, config_path = _config(tmp_path, formal=True)
    b0, review, profile = _evidence(tmp_path, cfg)
    profile_payload = json.loads(profile.read_text(encoding="utf-8"))
    profile_ticket = Path(profile_payload["launch_ticket_path"])
    ticket_payload = json.loads(profile_ticket.read_text(encoding="utf-8"))
    ticket_payload["scientific_config_sha256"] = "0" * 64
    _write(profile_ticket, ticket_payload)
    profile_payload["launch_ticket_sha256"] = _sha(profile_ticket)
    _write(profile, profile_payload)
    ticket = _ticket(
        tmp_path,
        cfg,
        config_path,
        b0,
        review,
        mode="formal",
        profile=profile,
    )

    with pytest.raises(FullPetalLaunchError, match="scientific config hash differs"):
        _authorize(cfg, config_path, ticket, mode="formal")


def test_formal_launch_accepts_profile_from_pre_authorization_commit(tmp_path):
    blocked_cfg, _ = _config(tmp_path / "blocked")
    _, _, profile = _evidence(
        tmp_path / "profile-evidence",
        blocked_cfg,
        commit="9" * 40,
    )
    formal_cfg, config_path = _config(tmp_path / "formal", formal=True)
    b0, review, _ = _evidence(tmp_path / "formal-evidence", formal_cfg)
    ticket = _ticket(
        tmp_path,
        formal_cfg,
        config_path,
        b0,
        review,
        mode="formal",
        profile=profile,
    )

    authorization = _authorize(formal_cfg, config_path, ticket, mode="formal")

    assert authorization.commit_sha == COMMIT
    assert authorization.profile_artifact_sha256 == _sha(profile)


def test_scientific_digest_ignores_only_authorization_state(tmp_path):
    blocked, _ = _config(tmp_path)
    approved, _ = _config(tmp_path, formal=True)

    assert resolved_config_sha256(blocked, scientific=True) == resolved_config_sha256(
        approved, scientific=True
    )
    assert resolved_config_sha256(blocked) != resolved_config_sha256(approved)


def test_profile_artifact_builder_binds_authorization_and_exact_measurements(tmp_path):
    cfg, _ = _config(tmp_path)
    authorization = FullPetalLaunchAuthorization(
        mode="profile",
        commit_sha=COMMIT,
        resolved_config_sha256="b" * 64,
        scientific_config_sha256="c" * 64,
        ticket_path=str(tmp_path / "ticket.json"),
        ticket_sha256="d" * 64,
        b0_artifact_sha256="e" * 64,
        review_artifact_sha256="f" * 64,
        profile_artifact_sha256=None,
        warmup_optimizer_events=50,
        measured_optimizer_events=200,
        world_size=1,
        slurm_job_id="12345",
    )
    measurements = {
        "warmup_optimizer_events": 50,
        "measured_optimizer_events": 200,
        "total_optimizer_events": 250,
        "skipped_optimizer_events": 0,
        "elapsed_seconds": 20.0,
        "peak_memory_bytes": 4096,
        "throughput_optimizer_events_per_second": 10.0,
    }

    artifact = build_fixed_step_profile_artifact(
        authorization,
        cfg,
        measurements,
        precision="bf16",
        gpu_name="Test GPU",
        torch_version="2.6.0",
        cuda_version="12.4",
    )

    assert artifact["schema_version"] == PROFILE_SCHEMA
    assert artifact["launch_ticket_path"] == str(tmp_path / "ticket.json")
    assert artifact["launch_ticket_sha256"] == "d" * 64
    assert artifact["measurements"] == measurements
