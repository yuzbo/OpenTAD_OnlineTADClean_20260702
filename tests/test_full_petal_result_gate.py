import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from opentad.utils.full_petal_attestation import generate_private_key, sign_payload
from opentad.utils.full_petal_b0 import (
    B0_AUDIT_REPORT_SCHEMA,
    B0_MANIFEST_SCHEMA,
    B0_SCHEMA,
    B0_TEST_REPORT_SCHEMA,
    canonical_json_sha256,
)
from opentad.utils.full_petal_data_contract import (
    build_fineaction_qualification_report,
    build_reporting_universe_manifest,
    compare_reporting_universe,
    save_json,
    sha256_file,
)
from opentad.utils.full_petal_identity import DATA_IDENTITY_SCHEMA
from opentad.utils.full_petal_training_evidence import (
    build_formal_run_manifest,
    persist_training_trace,
)
from opentad.utils.immutable_event_ledger import (
    ImmutableEventLedger,
    persist_verified_ledger,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "check_full_petal_results.py"
SPEC = importlib.util.spec_from_file_location("check_full_petal_results", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
COMMIT = "7" * 40
PROFILE_COMMIT = "6" * 40
REVIEW_SCOPE = [
    "P0_evidence_chain",
    "training_lifecycle",
    "data_metrics",
    "optimizer_launch",
    "full_B0",
]


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(path, payload)
    return path


def _reference(path):
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _protocol(claim):
    return {
        "input": "matched_features" if claim == "C1" else "raw_video",
        "decision_cadence": "packet_end",
        "nms": False,
        "offline_nms": False,
        "immutable_emissions": True,
        "load_from_raw_predictions": False,
        "runtime_identity": "persistent_slots",
        "lifecycle": "canonical_shared",
        "birth_rule": "first_threshold_crossing",
        "start_parameterization": "scalar",
        "endpoint_parameterization": "binary_first_crossing",
        "capacity": 4,
        "tracker": "fixed_across_comparison",
    }


def _keys(tmp_path, monkeypatch):
    profile_private = tmp_path / "profile.pem"
    formal_private = profile_private
    b0_private = tmp_path / "b0.pem"
    review_private = tmp_path / "review.pem"
    profile_public = generate_private_key(profile_private)
    b0_public = generate_private_key(b0_private)
    review_public = generate_private_key(review_private)
    roots = {
        "profile": {"key_id": "profile-test", "public_key": profile_public},
        "b0": {"key_id": "b0-test", "public_key": b0_public},
        "review": {"key_id": "review-test", "public_key": review_public},
    }
    monkeypatch.setattr(MODULE, "_attestation_trust_roots", lambda: roots)
    monkeypatch.setattr(
        MODULE,
        "_route_launch_requirements",
        lambda: {
            "reviewer_id": "review-test",
            "review_scope": REVIEW_SCOPE,
            "warmup_events": 50,
            "measured_events": 200,
            "world_size": 1,
            "dimensions": {
                "batch_size": 1,
                "chunk_size": 64,
                "memory_size": 192,
                "slot_count": 4,
            },
        },
    )
    monkeypatch.setattr(
        MODULE,
        "authorization_only_diff",
        lambda repository_root, profile_commit, formal_commit: "f" * 64,
    )
    return formal_private, b0_private, review_private, profile_private


def _b0_evidence(root, private_key):
    b0_dir = root / "b0"
    b0_dir.mkdir(parents=True)
    log = b0_dir / "tests.log"
    log.write_text("1 passed\n", encoding="utf-8")
    junit = b0_dir / "tests.xml"
    junit.write_text(
        '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0">'
        '<testcase classname="tests.test_fixture" name="test_one" />'
        "</testsuite></testsuites>\n",
        encoding="utf-8",
    )
    cases = [{"classname": "tests.test_fixture", "name": "test_one"}]
    manifest = {
        "schema_version": B0_MANIFEST_SCHEMA,
        "suite_order": ["all_contracts"],
        "suites": [
            {
                "name": "all_contracts",
                "runner": "isolated_torch",
                "canonical_argv": ["python", "-m", "pytest", "tests"],
                "test_files": [
                    {
                        "path": "tests/test_fixture.py",
                        "sha256": "1" * 64,
                        "test_functions": ["test_one"],
                    }
                ],
            }
        ],
        "runner_sources": [{"path": "tools/runner.py", "sha256": "2" * 64}],
        "audit_checks": ["python_syntax", "git_diff_check", "repository_clean_after"],
    }
    manifest_path = _write_json(b0_dir / "manifest.json", manifest)
    test_report = {
        "schema_version": B0_TEST_REPORT_SCHEMA,
        "status": "PASS",
        "commit_sha": COMMIT,
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
                "canonical_argv": ["python", "-m", "pytest", "tests"],
                "python_executable": sys.executable,
                "collected": 1,
                "passed": 1,
                "failed": 0,
                "errors": 0,
                "skipped": 0,
                "log_path": str(log),
                "log_sha256": sha256_file(log),
                "junit_path": str(junit),
                "junit_sha256": sha256_file(junit),
                "testcase_manifest_sha256": canonical_json_sha256(cases),
            }
        ],
    }
    test_report_path = _write_json(b0_dir / "test-report.json", test_report)
    checks = []
    for name in manifest["audit_checks"]:
        path = b0_dir / f"{name}.log"
        path.write_text("PASS\n", encoding="utf-8")
        checks.append(
            {
                "name": name,
                "status": "PASS",
                "canonical_argv": [name],
                "log_path": str(path),
                "log_sha256": sha256_file(path),
            }
        )
    audit_report_path = _write_json(
        b0_dir / "audit-report.json",
        {
            "schema_version": B0_AUDIT_REPORT_SCHEMA,
            "status": "PASS",
            "commit_sha": COMMIT,
            "manifest_sha256": sha256_file(manifest_path),
            "blocking_findings": 0,
            "protocol_violations": 0,
            "checks": checks,
        },
    )
    signed = sign_payload(
        {
            "schema_version": B0_SCHEMA,
            "status": "PASS",
            "commit_sha": COMMIT,
            "test_count": 1,
            "blocking_findings": 0,
            "protocol_violations": 0,
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "test_report_path": str(test_report_path),
            "test_report_sha256": sha256_file(test_report_path),
            "audit_report_path": str(audit_report_path),
            "audit_report_sha256": sha256_file(audit_report_path),
        },
        private_key_path=private_key,
        key_id="b0-test",
        role="b0-runner",
    )
    return _reference(_write_json(b0_dir / "b0.json", signed))


def _qualification_evidence(root, name, payload):
    path = _write_json(root / f"{name}.json", payload)
    return {**payload, "artifact_path": str(path), "artifact_sha256": sha256_file(path)}


def _reporting_evidence(root):
    reporting_dir = root / "reporting"
    reporting_dir.mkdir(parents=True)
    ids = [f"historical_{index:03d}" for index in range(211)]
    universe = build_reporting_universe_manifest(
        ids,
        provenance={"source": "locked-fixture"},
        seed=23,
        created_at="2026-07-13T08:00:00Z",
    )
    comparison = compare_reporting_universe(
        universe,
        [*ids, "extra_a", "extra_b"],
        observed_provenance={"source": "observed-fixture"},
        difference_reasons={
            "extra_a": "documented canonical addition",
            "extra_b": "documented canonical addition",
        },
        seed=23,
        created_at="2026-07-13T08:00:00Z",
    )
    counter = 0

    def checked(payload):
        nonlocal counter
        evidence = _qualification_evidence(reporting_dir, f"qualification-{counter}", payload)
        counter += 1
        return {"mandatory": True, "passed": True, "evidence": evidence}

    digest = "a" * 64
    qualification = build_fineaction_qualification_report(
        {
            "protocol": {
                "license": checked({"license_id": "FineAction-research"}),
                "official_split": checked({"manifest_sha256": digest}),
                "annotation_sha256": checked({"sha256": digest}),
                "instance_interval_ids": checked({"field": "instance_id", "verified_count": 30}),
            },
            "completeness": {
                "raw_video_access": checked({"inventory_sha256": digest}),
                "same_class_overlap_pairs": checked({"count": 20}),
                "same_class_repeated_instances": checked({"count": 30}),
                "qualified_ground_truth": checked({"count": 30}),
                "qualified_videos": checked({"count": 10}),
                "estimated_decode_storage_cost": checked(
                    {"decode_gpu_hours": 12.0, "storage_bytes": 1024}
                ),
            },
            "causal_readiness": {
                "causal_preprocessing_contract": checked(
                    {
                        "timestamp_convention": "zero_based_source_frame",
                        "future_frames_allowed": False,
                        "frame_stride": 2,
                        "manifest_sha256": digest,
                    }
                ),
                "minimal_dataset_loader_smoke": checked(
                    {"status": "PASS", "test_report_sha256": digest}
                ),
            },
        },
        seed=29,
        created_at="2026-07-13T08:00:00Z",
    )
    universe_path = _write_json(reporting_dir / "universe.json", universe)
    comparison_path = _write_json(reporting_dir / "comparison.json", comparison)
    qualification_path = _write_json(reporting_dir / "fineaction.json", qualification)
    return {
        "universe": _reference(universe_path),
        "comparison": _reference(comparison_path),
        "fineaction": _reference(qualification_path),
    }, qualification_path


def _emission(event_id, *, correct, score, slot_id):
    start, end = (30, 60) if correct else (90, 120)
    return {
        "event_id": event_id,
        "stream_id": "video_1",
        "stream_key": "video_1",
        "video_id": "video_1",
        "immutable": True,
        "slot_id": slot_id,
        "label": "Action",
        "score": score,
        "start_frame": start,
        "end_frame": end,
        "emit_frame": end,
        "source_frame": end,
        "segment": [start / 30.0, end / 30.0],
        "emit_time_sec": end / 30.0,
        "source_time_sec": end / 30.0,
        "fps": 30.0,
        "provenance_digest": "6" * 64,
    }


def _data_identity(path_records, fineaction_path):
    files = {}
    for name, path in path_records.items():
        files[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    optional_record = {
        "path": str(fineaction_path.resolve()),
        "sha256": sha256_file(fineaction_path),
        "size_bytes": fineaction_path.stat().st_size,
    }
    body = {
        "schema_version": DATA_IDENTITY_SCHEMA,
        "files": files,
        "feature_inventory_count": 1,
        "feature_inventory_sha256": "8" * 64,
        "optional_qualifications": {"fineaction_qualification": optional_record},
    }
    return {**body, "identity_sha256": MODULE._sha256_json(body, "data identity")}


def _training_event(seed):
    return {
        "event_id": f"optimizer-{seed}",
        "episode_id": f"episode-{seed}",
        "input_tokens": 64,
        "effective_batch_size": 1,
        "world_size": 1,
        "elapsed_seconds": 10.0,
        "peak_memory_bytes": 1024,
        "precision": "bf16",
        "optimizer_config_sha256": "1" * 64,
        "scheduler_config_sha256": "2" * 64,
        "data_order_sha256": "3" * 64,
        "loss_normalization_sha256": "4" * 64,
        "skipped": False,
    }


def _run(
    root,
    claim,
    variant,
    seed,
    *,
    formal_private,
    review_private,
    profile_private,
    b0,
    fineaction_path,
    pass_variant,
    evidence_mutator=None,
):
    run_dir = root / f"{claim}-{variant}-{seed}"
    run_dir.mkdir(parents=True)
    ground_truth = _write_json(
        run_dir / "ground-truth.json",
        {
            "database": {
                "video_1": {
                    "subset": "validation",
                    "frame": 300,
                    "duration": 10.0,
                    "annotations": [
                        {"instance_id": "gt-1", "segment": [1.0, 2.0], "label": "Action"}
                    ],
                }
            }
        },
    )
    allowed = run_dir / "allowed.txt"
    allowed.write_text("video_1\n", encoding="utf-8")
    config = run_dir / "config.py"
    config.write_text(f"claim = {claim!r}\nvariant = {variant!r}\n", encoding="utf-8")
    checkpoint = run_dir / "checkpoint.pth"
    checkpoint.write_bytes(f"{claim}:{variant}:{seed}".encode("ascii"))

    ledger = ImmutableEventLedger()
    if pass_variant:
        ledger.append(_emission(f"{claim}-{variant}-{seed}-correct", correct=True, score=0.9, slot_id=0))
    else:
        ledger.append(_emission(f"{claim}-{variant}-{seed}-correct", correct=True, score=0.5, slot_id=0))
        ledger.append(_emission(f"{claim}-{variant}-{seed}-wrong", correct=False, score=0.9, slot_id=1))
    ledger_path = run_dir / "emissions.jsonl"
    commitment_path = run_dir / "emissions.commitment.json"
    persist_verified_ledger(ledger_path, commitment_path, ledger.rows)

    evaluator_spec = _write_json(
        run_dir / "evaluator.json",
        {
            "schema_version": MODULE.EVALUATOR_SPEC_SCHEMA,
            "type": "OnlineAPBudgeted",
            "subset": "validation",
            "tiou_thresholds": [0.5, 0.7],
            "latency_budgets_sec": [0.5, 1.0],
            "fps": 30.0,
            "identity_tiou_threshold": 0.5,
            "identity_latency_budget_sec": 1.0,
        },
    )
    trace = run_dir / "training.jsonl"
    trace_commitment = run_dir / "training.commitment.json"
    persist_training_trace(trace, trace_commitment, [_training_event(seed)])
    data_identity = _data_identity(
        {"annotation": ground_truth, "allowed_videos": allowed}, fineaction_path
    )
    data_identity_path = _write_json(run_dir / "data-identity.json", data_identity)
    review = _write_json(
        run_dir / "review.json",
        sign_payload(
            {
                "schema_version": MODULE.REVIEW_SCHEMA,
                "reviewer_id": "review-test",
                "reviewed_commit": COMMIT,
                "b0_artifact_sha256": b0["sha256"],
                "scope": REVIEW_SCOPE,
                "verdict": "PASS",
                "blocking_findings": [],
                "protocol_violations": [],
            },
            private_key_path=review_private,
            key_id="review-test",
            role=MODULE.REVIEW_ATTESTATION_ROLE,
        ),
    )
    review_reference = _reference(review)
    profile_runtime = {
        "schema_version": "full-petal-runtime-identity-v1",
        "entrypoint": "train",
        "seed": seed,
        "run_id": 0,
        "deterministic": True,
        "not_eval": False,
        "resume_checkpoint": None,
        "cfg_overrides": {},
    }
    profile_ticket = _write_json(
        run_dir / "profile-launch-ticket.json",
        {
            "schema_version": MODULE.LAUNCH_TICKET_SCHEMA,
            "mode": "profile",
            "commit_sha": PROFILE_COMMIT,
            "source_tree_sha256": "9" * 64,
            "config_file_sha256": sha256_file(config),
            "resolved_config_sha256": "c" * 64,
            "scientific_config_sha256": "b" * 64,
            "data_identity": data_identity,
            "runtime_identity": profile_runtime,
            "b0_evidence": b0,
            "review_evidence": review_reference,
            "profile_evidence": None,
        },
    )
    profile_job = f"{seed}-profile"
    profile = _write_json(
        run_dir / "profile.json",
        sign_payload(
            {
                "schema_version": MODULE.PROFILE_SCHEMA,
                "status": "PASS",
                "commit_sha": PROFILE_COMMIT,
                "source_tree_sha256": "9" * 64,
                "data_identity_sha256": data_identity["identity_sha256"],
                "runtime_identity_sha256": MODULE._sha256_json(
                    profile_runtime, "profile runtime"
                ),
                "resolved_config_sha256": "c" * 64,
                "scientific_config_sha256": "b" * 64,
                "launch_ticket_path": str(profile_ticket.resolve()),
                "launch_ticket_sha256": sha256_file(profile_ticket),
                "slurm_job_id": profile_job,
                "slurm_allocation": {
                    "job_id": profile_job,
                    "state": "RUNNING",
                    "user": "fixture-user",
                    "nodes": 1,
                    "tasks": 1,
                    "gpus": 1,
                    "command": "python tools/train.py",
                    "work_dir": str(run_dir.resolve()),
                },
                "world_size": 1,
                "precision": "bf16",
                "hardware": {
                    "gpu_name": "Fixture GPU",
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
            private_key_path=profile_private,
            key_id="profile-test",
            role=MODULE.PROFILE_ATTESTATION_ROLE,
        ),
    )
    profile_reference = _reference(profile)

    def launch_evidence(stage, entrypoint, resume_checkpoint):
        runtime = {
            "schema_version": "full-petal-runtime-identity-v1",
            "entrypoint": entrypoint,
            "seed": seed,
            "run_id": 0,
            "deterministic": True,
            "not_eval": False,
            "resume_checkpoint": resume_checkpoint,
            "cfg_overrides": {},
        }
        ticket = {
            "schema_version": MODULE.LAUNCH_TICKET_SCHEMA,
            "mode": "formal",
            "commit_sha": COMMIT,
            "source_tree_sha256": "9" * 64,
            "config_file_sha256": sha256_file(config),
            "resolved_config_sha256": "a" * 64,
            "scientific_config_sha256": "b" * 64,
            "data_identity": data_identity,
            "runtime_identity": runtime,
            "b0_evidence": b0,
            "review_evidence": review_reference,
            "profile_evidence": profile_reference,
        }
        ticket_path = _write_json(run_dir / f"{stage}-launch-ticket.json", ticket)
        job_id = f"{seed}-{stage}"
        receipt = {
            "schema_version": MODULE.LAUNCH_RECEIPT_SCHEMA,
            "mode": "formal",
            "commit_sha": COMMIT,
            "source_tree_sha256": ticket["source_tree_sha256"],
            "data_identity_sha256": data_identity["identity_sha256"],
            "runtime_identity_sha256": MODULE._sha256_json(runtime, "runtime"),
            "resolved_config_sha256": ticket["resolved_config_sha256"],
            "scientific_config_sha256": ticket["scientific_config_sha256"],
            "launch_ticket": _reference(ticket_path),
            "b0_artifact_sha256": b0["sha256"],
            "review_artifact_sha256": review_reference["sha256"],
            "profile_artifact_sha256": profile_reference["sha256"],
            "world_size": 1,
            "slurm_job_id": job_id,
            "slurm_allocation": {
                "job_id": job_id,
                "state": "RUNNING",
                "user": "fixture-user",
                "nodes": 1,
                "tasks": 1,
                "gpus": 1,
                "command": f"python tools/{entrypoint}.py",
                "work_dir": str(run_dir.resolve()),
            },
        }
        receipt_path = _write_json(
            run_dir / f"{stage}-launch-receipt.json", receipt
        )
        return ticket_path, receipt_path

    training_ticket, training_receipt = launch_evidence("training", "train", None)
    evaluation_ticket, evaluation_receipt = launch_evidence(
        "evaluation",
        "test",
        {
            "path": str(checkpoint.resolve()),
            "sha256": sha256_file(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
        },
    )
    artifacts = {
        "ledger": ledger_path,
        "commitment": commitment_path,
        "ground_truth": ground_truth,
        "allowed_videos": allowed,
        "evaluator_spec": evaluator_spec,
        "config": config,
        "checkpoint": checkpoint,
        "data_identity": data_identity_path,
        "training_launch_ticket": training_ticket,
        "training_launch_receipt": training_receipt,
        "evaluation_launch_ticket": evaluation_ticket,
        "evaluation_launch_receipt": evaluation_receipt,
        "training_trace": trace,
        "training_commitment": trace_commitment,
    }
    if claim == "C2":
        adapted = variant == "adapted"
        artifacts["raw_visual_audit"] = _write_json(
            run_dir / "raw-visual-audit.json",
            {
                "schema_version": MODULE.RAW_VISUAL_AUDIT_SCHEMA,
                "claim": claim,
                "variant": variant,
                "seed": seed,
                "model_sha256": sha256_file(checkpoint),
                "dataset_manifest_sha256": sha256_file(data_identity_path),
                "registered_visual_params": 2 if adapted else 0,
                "nonzero_finite_grad_params": 2 if adapted else 0,
                "changed_visual_params": 1 if adapted else 0,
                "frozen_param_delta_max": 0.0,
                "status": "PASS",
            },
        )
    if evidence_mutator is not None:
        evidence_mutator(artifacts)
    protocol = _protocol(claim)
    signed = build_formal_run_manifest(
        claim=claim,
        variant=variant,
        seed=seed,
        commit_sha=COMMIT,
        protocol_sha256=MODULE._sha256_json(protocol, "protocol"),
        artifacts=artifacts,
        private_key_path=formal_private,
        key_id="profile-test",
    )
    manifest_path = _write_json(run_dir / "run-manifest.json", signed)
    return {
        "claim": claim,
        "variant": variant,
        "seed": seed,
        "protocol": protocol,
        "evidence": {
            "run_manifest_path": str(manifest_path),
            "run_manifest_sha256": sha256_file(manifest_path),
        },
    }


def _report(tmp_path, monkeypatch, *, c1_pass=True, c2_pass=True, reporting=True):
    root = tmp_path / "report"
    root.mkdir(parents=True)
    formal_private, b0_private, review_private, profile_private = _keys(
        root, monkeypatch
    )
    b0 = _b0_evidence(root, b0_private)
    reporting_evidence, fineaction_path = _reporting_evidence(root)
    rows = []
    for seed in (11, 12):
        rows.extend(
            [
                _run(
                    root,
                    "C1",
                    "rematch",
                    seed,
                    formal_private=formal_private,
                    review_private=review_private,
                    profile_private=profile_private,
                    b0=b0,
                    fineaction_path=fineaction_path,
                    pass_variant=not c1_pass,
                ),
                _run(
                    root,
                    "C1",
                    "fixed",
                    seed,
                    formal_private=formal_private,
                    review_private=review_private,
                    profile_private=profile_private,
                    b0=b0,
                    fineaction_path=fineaction_path,
                    pass_variant=True,
                ),
                _run(
                    root,
                    "C2",
                    "frozen",
                    seed,
                    formal_private=formal_private,
                    review_private=review_private,
                    profile_private=profile_private,
                    b0=b0,
                    fineaction_path=fineaction_path,
                    pass_variant=not c2_pass,
                ),
                _run(
                    root,
                    "C2",
                    "adapted",
                    seed,
                    formal_private=formal_private,
                    review_private=review_private,
                    profile_private=profile_private,
                    b0=b0,
                    fineaction_path=fineaction_path,
                    pass_variant=True,
                ),
            ]
        )
    payload = {"runs": rows, "b0_evidence": b0}
    if reporting:
        payload["reporting_evidence"] = reporting_evidence
    return payload


def test_signed_replay_can_pass_independent_claim_and_project_gates(tmp_path, monkeypatch):
    verdict = MODULE.evaluate_result_gates(_report(tmp_path, monkeypatch))

    assert verdict["decisions"]["C1"]["status"] == "PASS"
    assert verdict["decisions"]["C2"]["status"] == "PASS"
    assert verdict["decisions"]["project_full_system"]["status"] == "NARROW"
    assert verdict["decisions"]["project_full_system"]["requirements"] == {
        "C1": True,
        "C2": True,
        "protocol": True,
        "B0": True,
        "reporting_211_213": True,
        "FineAction": True,
    }


def test_c1_and_c2_remain_independent(tmp_path, monkeypatch):
    verdict = MODULE.evaluate_result_gates(
        _report(tmp_path, monkeypatch, c1_pass=False, c2_pass=True)
    )

    assert verdict["decisions"]["C1"]["status"] == "FAIL"
    assert verdict["decisions"]["C2"]["status"] == "PASS"
    assert verdict["decisions"]["project_full_system"]["status"] == "KILL"


def test_gate_rejects_handwritten_metrics_cost_and_pass_flags(tmp_path, monkeypatch):
    report = _report(tmp_path, monkeypatch)
    report["runs"][0]["metrics"] = {"average_mAP": 1.0}
    report["runs"][0]["cost"] = {"gpu_hours": 0.0}

    with pytest.raises(MODULE.ResultGateInputError, match="must be recomputed"):
        MODULE.evaluate_result_gates(report)


def test_gate_rejects_unsigned_or_tampered_run_evidence(tmp_path, monkeypatch):
    report = _report(tmp_path, monkeypatch)
    manifest_path = Path(report["runs"][0]["evidence"]["run_manifest_path"])
    signed = json.loads(manifest_path.read_text(encoding="utf-8"))
    signed.pop("attestation")
    _write_json(manifest_path, signed)
    report["runs"][0]["evidence"]["run_manifest_sha256"] = sha256_file(manifest_path)

    with pytest.raises(MODULE.ResultGateInputError, match="attestation"):
        MODULE.evaluate_result_gates(report)

    report = _report(tmp_path / "tamper", monkeypatch)
    manifest = json.loads(
        Path(report["runs"][0]["evidence"]["run_manifest_path"]).read_text(encoding="utf-8")
    )
    ledger = Path(manifest["artifacts"]["ledger"]["path"])
    ledger.write_text(ledger.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(MODULE.ResultGateInputError, match="hash differs"):
        MODULE.evaluate_result_gates(report)


def _semantic_run(tmp_path, monkeypatch, mutator):
    root = tmp_path / "semantic-run"
    root.mkdir(parents=True)
    formal_private, b0_private, review_private, profile_private = _keys(
        root, monkeypatch
    )
    b0 = _b0_evidence(root, b0_private)
    _, fineaction_path = _reporting_evidence(root)
    return _run(
        root,
        "C1",
        "fixed",
        11,
        formal_private=formal_private,
        review_private=review_private,
        profile_private=profile_private,
        b0=b0,
        fineaction_path=fineaction_path,
        pass_variant=True,
        evidence_mutator=mutator,
    )


def test_gate_rejects_self_consistent_evaluation_ticket_for_wrong_checkpoint(
    tmp_path, monkeypatch
):
    def wrong_checkpoint(artifacts):
        ticket_path = artifacts["evaluation_launch_ticket"]
        receipt_path = artifacts["evaluation_launch_receipt"]
        ticket = json.loads(ticket_path.read_text(encoding="utf-8"))
        ticket["runtime_identity"]["resume_checkpoint"]["sha256"] = "f" * 64
        _write_json(ticket_path, ticket)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["launch_ticket"] = _reference(ticket_path)
        receipt["runtime_identity_sha256"] = MODULE._sha256_json(
            ticket["runtime_identity"], "runtime"
        )
        _write_json(receipt_path, receipt)

    row = _semantic_run(tmp_path, monkeypatch, wrong_checkpoint)
    with pytest.raises(MODULE.ResultGateInputError, match="evaluated checkpoint"):
        MODULE._validate_run_evidence(row, "C1", "fixed", 11, "semantic")


def test_gate_rejects_signed_receipt_without_active_slurm_allocation(
    tmp_path, monkeypatch
):
    def completed_job(artifacts):
        receipt_path = artifacts["training_launch_receipt"]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["slurm_allocation"]["state"] = "COMPLETED"
        _write_json(receipt_path, receipt)

    row = _semantic_run(tmp_path, monkeypatch, completed_job)
    with pytest.raises(MODULE.ResultGateProtocolError, match="active Slurm"):
        MODULE._validate_run_evidence(row, "C1", "fixed", 11, "semantic")


@pytest.mark.parametrize(
    ("role", "receipt_field"),
    (("review", "review_artifact_sha256"), ("profile", "profile_artifact_sha256")),
)
def test_gate_rejects_unsigned_prerequisite_after_all_outer_hashes_are_rebuilt(
    tmp_path, monkeypatch, role, receipt_field
):
    def unsigned_prerequisite(artifacts):
        ticket_paths = [
            artifacts["training_launch_ticket"],
            artifacts["evaluation_launch_ticket"],
        ]
        first_ticket = json.loads(ticket_paths[0].read_text(encoding="utf-8"))
        prerequisite_path = Path(first_ticket[f"{role}_evidence"]["path"])
        prerequisite = json.loads(prerequisite_path.read_text(encoding="utf-8"))
        prerequisite.pop("attestation")
        _write_json(prerequisite_path, prerequisite)
        prerequisite_hash = sha256_file(prerequisite_path)

        for stage, ticket_path in zip(("training", "evaluation"), ticket_paths):
            ticket = json.loads(ticket_path.read_text(encoding="utf-8"))
            ticket[f"{role}_evidence"]["sha256"] = prerequisite_hash
            _write_json(ticket_path, ticket)
            receipt_path = artifacts[f"{stage}_launch_receipt"]
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["launch_ticket"] = _reference(ticket_path)
            receipt[receipt_field] = prerequisite_hash
            _write_json(receipt_path, receipt)

    row = _semantic_run(tmp_path, monkeypatch, unsigned_prerequisite)
    with pytest.raises(MODULE.ResultGateInputError, match="attestation"):
        MODULE._validate_run_evidence(row, "C1", "fixed", 11, "semantic")


def test_project_gate_requires_reporting_and_fineaction_evidence(tmp_path, monkeypatch):
    verdict = MODULE.evaluate_result_gates(
        _report(tmp_path, monkeypatch, reporting=False)
    )

    requirements = verdict["decisions"]["project_full_system"]["requirements"]
    assert requirements["reporting_211_213"] is False
    assert requirements["FineAction"] is False
    assert verdict["decisions"]["project_full_system"]["status"] == "KILL"


def test_c1_map_gain_cannot_hide_worse_duplicate_and_fragmentation():
    def metrics(online_map, duplicate, fragmentation):
        return {
            "average_mAP": online_map,
            "average_mOnlineAP": online_map,
            "identity_recall": 0.8,
            "duplicate_per_gt": duplicate,
            "fragmentation_rate": fragmentation,
            "false_emission_rate": 0.1,
            "endpoint_latency_frames_mean": 1.0,
        }

    grouped = {
        "C1": {
            "fixed": {1: {"metrics": metrics(0.9, 0.4, 0.4)}},
            "rematch": {1: {"metrics": metrics(0.1, 0.1, 0.1)}},
        }
    }
    result = MODULE._c1_scientific_metrics(
        grouped,
        [1],
        map_gain_points=2.0,
        error_reduction=0.2,
        map_parity_tolerance_points=0.5,
        recall_tolerance_points=1.0,
        false_emission_tolerance=0.01,
        latency_tolerance_frames=0.0,
    )

    assert result["effect_paths"]["map_gain"] is True
    assert result["safety_checks"]["duplicate_and_fragmentation_noninferior"] is False
    assert result["passed"] is False


def test_cli_reports_malformed_input_without_claiming_science(tmp_path):
    path = _write_json(tmp_path / "malformed.json", {"runs": [{"claim": "C1"}]})
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert json.loads(completed.stdout)["error"]["kind"] == "malformed_input"
