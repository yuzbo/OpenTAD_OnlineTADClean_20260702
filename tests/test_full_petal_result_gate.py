import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch
from mmengine import Config

from opentad.utils.full_petal_attestation import generate_private_key, public_key_base64
from tests.full_petal_attestation_fixture import (
    attest_fixture as _sign_payload,
    committed_optimizer_envelope,
)
from opentad.utils.full_petal_b0 import (
    B0_AUDIT_REPORT_SCHEMA,
    B0_MANIFEST_SCHEMA,
    B0_SCHEMA,
    B0_TEST_REPORT_SCHEMA,
    canonical_json_sha256,
)
from opentad.utils.full_petal_data_contract import (
    build_fineaction_qualification_report,
    build_id_file_provenance,
    build_reporting_universe_manifest,
    compare_reporting_universe,
    save_json,
    sha256_file,
)
from opentad.utils.full_petal_identity import (
    DATA_IDENTITY_SCHEMA,
    derive_training_trace_identity,
)
from opentad.utils.full_petal_training_evidence import (
    TrainingEvidenceError,
    build_formal_run_manifest,
    derive_fixed_step_profile_measurements,
    persist_training_trace,
    persist_visual_parameter_trace,
    tensor_sha256,
)
from opentad.utils.full_petal_launch import resolved_config_sha256
from opentad.utils.full_petal_runtime_attestation import issue_runtime_session
from opentad.utils.full_petal_fineaction_executor import (
    run_fineaction_loader_evidence,
    run_fineaction_preprocessing_evidence,
)
from opentad.utils.full_petal_role_signing import (
    sign_fineaction_license_authorization,
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


def _reference(path, root=None):
    reference_path = (
        str(path.resolve())
        if root is None
        else path.resolve().relative_to(root.resolve()).as_posix()
    )
    return {"path": reference_path, "sha256": sha256_file(path)}


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
    formal_private = tmp_path / "formal.pem"
    b0_private = tmp_path / "b0.pem"
    review_private = tmp_path / "review.pem"
    fineaction_license_private = tmp_path / "fineaction-license.pem"
    fineaction_execution_private = tmp_path / "fineaction-execution.pem"
    profile_public = generate_private_key(profile_private)
    formal_public = generate_private_key(formal_private)
    b0_public = generate_private_key(b0_private)
    review_public = generate_private_key(review_private)
    fineaction_license_public = generate_private_key(fineaction_license_private)
    fineaction_execution_public = generate_private_key(fineaction_execution_private)
    fineaction_roots = {
        "license": {
            "key_id": "fineaction-license-test",
            "public_key": fineaction_license_public,
        },
        "execution": {
            "key_id": "fineaction-execution-test",
            "public_key": fineaction_execution_public,
        },
    }
    roots = {
        "profile": {"key_id": "profile-test", "public_key": profile_public},
        "formal": {"key_id": "formal-test", "public_key": formal_public},
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
            "fineaction_trust_roots": fineaction_roots,
        },
    )
    monkeypatch.setattr(
        MODULE,
        "authorization_only_diff",
        lambda repository_root, profile_commit, formal_commit: "f" * 64,
    )
    return (
        formal_private,
        b0_private,
        review_private,
        profile_private,
        fineaction_license_private,
        fineaction_execution_private,
        fineaction_roots,
    )


def _b0_evidence(root, private_key, *, commit=COMMIT):
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
                "canonical_argv": ["python", "-m", "pytest", "tests"],
                "python_executable": sys.executable,
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
                "log_path": path.name,
                "log_sha256": sha256_file(path),
            }
        )
    audit_report_path = _write_json(
        b0_dir / "audit-report.json",
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
    signed = _sign_payload(
        {
            "schema_version": B0_SCHEMA,
            "status": "PASS",
            "commit_sha": commit,
            "test_count": 1,
            "blocking_findings": 0,
            "protocol_violations": 0,
            "manifest_path": manifest_path.name,
            "manifest_sha256": sha256_file(manifest_path),
            "test_report_path": test_report_path.name,
            "test_report_sha256": sha256_file(test_report_path),
            "audit_report_path": audit_report_path.name,
            "audit_report_sha256": sha256_file(audit_report_path),
        },
        private_key_path=private_key,
        key_id="b0-test",
        role="b0-runner",
    )
    return _reference(_write_json(b0_dir / "b0.json", signed))


def _fineaction_sources(
    root, *, license_private, execution_private
):
    media_dir = root / "media"
    media_dir.mkdir(parents=True)
    database = {}
    entries = []
    for index in range(10):
        video_id = f"fineaction_{index:03d}"
        media_path = media_dir / f"{video_id}.mp4"
        media_path.write_bytes(f"verified-media-{index}".encode("ascii"))
        entries.append(
            {
                "video_id": video_id,
                "path": media_path.relative_to(root).as_posix(),
                "sha256": sha256_file(media_path),
                "size_bytes": media_path.stat().st_size,
            }
        )
        database[video_id] = {
            "subset": "training" if index < 5 else "validation",
            "duration": 10.0,
            "frame": 300,
            "annotations": [
                {"segment": segment, "label": "same-class"}
                for segment in ([0.0, 4.0], [1.0, 5.0], [2.0, 6.0])
            ],
        }
    annotation = _write_json(root / "annotation.json", {"database": database})
    inventory = _write_json(
        root / "media-inventory.json",
        {
            "schema": "full_petal.fineaction_media_inventory",
            "schema_version": 1,
            "dataset": "FineAction",
            "entries": entries,
        },
    )
    terms = root / "license-terms.txt"
    terms.write_text("FineAction research terms\n", encoding="utf-8")
    license_path = _write_json(
        root / "license.json",
        sign_fineaction_license_authorization(
            {
                "schema_version": "full-petal-fineaction-license-authorization-v1",
                "dataset": "FineAction",
                "license_id": "FineAction-research",
                "subject": "full-petal-test",
                "access_scope": "research-evaluation",
                "authorized": True,
                "issued_at": "2026-07-13T08:00:00Z",
                "terms": _reference(terms, root),
            },
            private_key_path=license_private,
            key_id="fineaction-license-test",
        ),
    )
    preprocessing_source = root / "fineaction-preprocess-test.py"
    preprocessing_source.write_text(
        "def test_causal_preprocessing():\n    assert True\n",
        encoding="utf-8",
    )
    preprocessing = run_fineaction_preprocessing_evidence(
        preprocessing_source,
        root / "preprocessing-evidence",
        annotation_sha256=sha256_file(annotation),
        media_inventory_sha256=sha256_file(inventory),
        timestamp_convention="zero_based_source_frame",
        frame_stride=2,
        private_key_path=execution_private,
        key_id="fineaction-execution-test",
    )
    loader_source = root / "fineaction-loader-smoke-test.py"
    loader_source.write_text(
        "def test_minimal_loader_smoke():\n    assert True\n",
        encoding="utf-8",
    )
    smoke = run_fineaction_loader_evidence(
        loader_source,
        root / "loader-evidence",
        annotation_sha256=sha256_file(annotation),
        media_inventory_sha256=sha256_file(inventory),
        preprocessing_sha256=sha256_file(preprocessing),
        private_key_path=execution_private,
        key_id="fineaction-execution-test",
    )
    return {
        "annotation": annotation,
        "media_inventory": inventory,
        "license": license_path,
        "preprocessing": preprocessing,
        "loader_smoke": smoke,
    }


def _reporting_evidence(
    root, *, license_private, execution_private, trust_roots
):
    reporting_dir = root / "reporting"
    reporting_dir.mkdir(parents=True)
    ids = [f"historical_{index:03d}" for index in range(211)]
    observed_ids = [*ids, "extra_a", "extra_b"]
    reasons = {
        "extra_a": "documented canonical addition",
        "extra_b": "documented canonical addition",
    }
    locked_ids_path = _write_json(reporting_dir / "locked-ids.json", ids)
    observed_ids_path = _write_json(reporting_dir / "observed-ids.json", observed_ids)
    reasons_path = _write_json(reporting_dir / "difference-reasons.json", reasons)
    universe = build_reporting_universe_manifest(
        ids,
        provenance=build_id_file_provenance(locked_ids_path, ids),
        seed=23,
        created_at="2026-07-13T08:00:00Z",
    )
    comparison = compare_reporting_universe(
        universe,
        observed_ids,
        observed_provenance=build_id_file_provenance(
            observed_ids_path, observed_ids
        ),
        difference_reasons=reasons,
        seed=23,
        created_at="2026-07-13T08:00:00Z",
    )
    fineaction_sources = _fineaction_sources(
        reporting_dir / "fineaction-sources",
        license_private=license_private,
        execution_private=execution_private,
    )
    qualification = build_fineaction_qualification_report(
        fineaction_sources,
        seed=29,
        created_at="2026-07-13T08:00:00Z",
        trust_roots=trust_roots,
    )
    universe_path = _write_json(reporting_dir / "universe.json", universe)
    comparison_path = _write_json(reporting_dir / "comparison.json", comparison)
    qualification_path = _write_json(reporting_dir / "fineaction.json", qualification)
    return {
        "universe": _reference(universe_path),
        "comparison": _reference(comparison_path),
        "locked_ids": _reference(locked_ids_path),
        "observed_ids": _reference(observed_ids_path),
        "difference_reasons": _reference(reasons_path),
        "fineaction": _reference(qualification_path),
        "fineaction_sources": {
            name: _reference(path) for name, path in fineaction_sources.items()
        },
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


def _training_event(seed, training_identity=None):
    return {
        "event_id": f"optimizer-{seed}",
        "episode_id": f"episode-{seed}",
        "input_tokens": 64,
        "effective_batch_size": 1,
        "world_size": 1,
        "elapsed_seconds": 10.0,
        "peak_memory_bytes": 1024,
        **(
            training_identity
            or {
                "precision": "bf16",
                "optimizer_config_sha256": "1" * 64,
                "scheduler_config_sha256": "2" * 64,
                "data_order_sha256": "3" * 64,
                "loss_normalization_sha256": "4" * 64,
                "effective_batch_size": 1,
                "world_size": 1,
            }
        ),
        "skipped": False,
    }


def _signed_runtime_event(runtime_session, event_kind, payload):
    if event_kind == "optimizer-event":
        envelope = committed_optimizer_envelope(runtime_session, payload)
    elif event_kind == "visual-parameter-event":
        envelope = runtime_session.sign_visual_parameter_event(payload)
    else:
        raise AssertionError(f"unsupported runtime fixture event kind: {event_kind}")
    return {
        **payload,
        **envelope,
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
    profile_b0,
    fineaction_path,
    pass_variant,
    evidence_mutator=None,
):
    run_dir = root / f"{claim}-{variant}-{seed}"
    run_dir.mkdir(parents=True)
    formal_b0_dir = run_dir / "formal-b0"
    profile_b0_dir = run_dir / "profile-b0"
    shutil.copytree(Path(b0["path"]).parent, formal_b0_dir)
    shutil.copytree(Path(profile_b0["path"]).parent, profile_b0_dir)
    formal_b0_reference = _reference(
        formal_b0_dir / Path(b0["path"]).name, run_dir
    )
    profile_b0_reference = _reference(
        profile_b0_dir / Path(profile_b0["path"]).name, run_dir
    )
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
    fit_core = run_dir / "fit-core.txt"
    fit_core.write_text("video_1\n", encoding="utf-8")
    cache_manifest = _write_json(
        run_dir / "cache-manifest.json",
        {"videos": {"video_1": {"source_frames": list(range(64))}}},
    )
    (run_dir / "base.py").write_text(
        "base_marker = 'resolved-from-base'\n", encoding="utf-8"
    )
    config = run_dir / "config.py"
    config.write_text(
        "\n".join(
            (
                "_base_ = ['./base.py']",
                f"claim = {claim!r}",
                f"variant = {variant!r}",
                "optimizer = dict(type='AdamW', lr=0.0002)",
                "scheduler = dict(type='LinearWarmupCosineAnnealingLR', warmup_epoch=1, max_epoch=12)",
                "model = dict(type='PersistentTrajectoryOnlineDetector')",
                "solver = dict(train=dict(batch_size=1), amp=True, amp_dtype='bf16')",
                "launch_contract = dict(attestation_trust_roots=dict(",
                "    profile=dict(key_id='profile-test',",
                f"        public_key={public_key_base64(profile_private)!r})))",
                "visual_parameter_contract = dict(",
                "    parameter_prefixes=['visual'],",
                "    adapted_trainable_prefixes=['visual.encoder'])",
                "dataset = dict(train=dict(",
                f"    ann_file={str(ground_truth.resolve())!r},",
                f"    allow_list={str(fit_core.resolve())!r},",
                f"    cache_manifest={str(cache_manifest.resolve())!r},",
                "    chunk_size=64, subset_name='validation'))",
                "evaluation = dict(",
                "    type='OnlineAPBudgeted', subset='validation',",
                f"    allowed_videos={str(allowed.resolve())!r},",
                "    tiou_thresholds=[0.5, 0.7], latency_budgets_sec=[0.5, 1.0],",
                "    fps=30.0, require_ledger=True, require_no_future=True,",
                f"    ground_truth_filename={str(ground_truth.resolve())!r},",
                "    identity_tiou_threshold=0.5, identity_latency_budget_sec=1.0)",
                "",
            )
        ),
        encoding="utf-8",
    )
    checkpoint = run_dir / "checkpoint.pth"
    checkpoint_parameter = torch.full((1,), float(seed + 1), dtype=torch.float32)
    torch.save(
        {"state_dict": {"visual.encoder.weight": checkpoint_parameter}},
        checkpoint,
    )

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
    data_identity = _data_identity(
        {
            "annotation": ground_truth,
            "calibration": allowed,
            "fit_core": fit_core,
            "feature_cache_manifest": cache_manifest,
        },
        fineaction_path,
    )
    resolved_cfg = Config.fromfile(str(config))
    training_identity = derive_training_trace_identity(
        resolved_cfg,
        data_identity,
        seed=seed,
        world_size=1,
    )
    training_runtime_session = issue_runtime_session()
    trace = run_dir / "training.jsonl"
    trace_commitment = run_dir / "training.commitment.json"
    training_event = _training_event(seed, training_identity)
    persist_training_trace(
        trace,
        trace_commitment,
        [
            _signed_runtime_event(
                training_runtime_session, "optimizer-event", training_event
            )
        ],
    )
    data_identity_path = _write_json(run_dir / "data-identity.json", data_identity)
    resolved_config = _write_json(
        run_dir / "resolved-config.json", resolved_cfg.to_dict()
    )
    resolved_cfg_sha256 = resolved_config_sha256(resolved_cfg)
    scientific_cfg_sha256 = resolved_config_sha256(resolved_cfg, scientific=True)
    review = _write_json(
        run_dir / "review.json",
        _sign_payload(
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
    review_reference = _reference(review, run_dir)
    profile_review = _write_json(
        run_dir / "profile-review.json",
        _sign_payload(
            {
                "schema_version": MODULE.REVIEW_SCHEMA,
                "reviewer_id": "review-test",
                "reviewed_commit": PROFILE_COMMIT,
                "b0_artifact_sha256": profile_b0["sha256"],
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
    profile_review_reference = _reference(profile_review, run_dir)
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
            "resolved_config_sha256": resolved_cfg_sha256,
            "scientific_config_sha256": scientific_cfg_sha256,
            "data_identity": data_identity,
            "runtime_identity": profile_runtime,
            "b0_evidence": profile_b0_reference,
            "review_evidence": profile_review_reference,
            "profile_evidence": None,
        },
    )
    profile_runtime_session = issue_runtime_session()
    profile_job = f"{seed}-profile"
    profile_receipt = _write_json(
        run_dir / "profile-launch-receipt.json",
        _sign_payload(
            {
                "schema_version": MODULE.LAUNCH_RECEIPT_SCHEMA,
                "mode": "profile",
                "commit_sha": PROFILE_COMMIT,
                "source_tree_sha256": "9" * 64,
                "data_identity_sha256": data_identity["identity_sha256"],
                "runtime_identity_sha256": MODULE._sha256_json(
                    profile_runtime, "profile runtime"
                ),
                "resolved_config_sha256": resolved_cfg_sha256,
                "scientific_config_sha256": scientific_cfg_sha256,
                "launch_ticket": _reference(profile_ticket, run_dir),
                "b0_artifact_sha256": profile_b0["sha256"],
                "review_artifact_sha256": profile_review_reference["sha256"],
                "profile_artifact_sha256": None,
                "world_size": 1,
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
                "execution_session": profile_runtime_session.binding,
            },
            private_key_path=profile_private,
            key_id="profile-test",
            role=MODULE.LAUNCH_RECEIPT_ATTESTATION_ROLE,
        ),
    )
    profile_trace = run_dir / "profile-optimizer-events.jsonl"
    profile_commitment = run_dir / "profile-optimizer-events.commitment.json"
    profile_events = []
    for index in range(250):
        profile_event = {
            **_training_event(seed, training_identity),
            "event_id": f"profile-optimizer-{index:08d}",
            "episode_id": f"profile-episode-{index:08d}",
            "elapsed_seconds": float(index + 1),
            "peak_memory_bytes": 1024 + index,
        }
        profile_events.append(
            _signed_runtime_event(
                profile_runtime_session, "optimizer-event", profile_event
            )
        )
    persist_training_trace(
        profile_trace,
        profile_commitment,
        profile_events,
    )
    synchronized_profile_measurements = {
        "warmup_optimizer_events": 50,
        "measured_optimizer_events": 200,
        "total_optimizer_events": 250,
        "skipped_optimizer_events": 0,
        "measurement_start_after_event_id": "profile-optimizer-00000049",
        "measurement_end_event_id": "profile-optimizer-00000249",
        "elapsed_seconds": 5.0,
        "peak_memory_bytes": 4096,
        "throughput_optimizer_events_per_second": 40.0,
    }
    profile_measurements = derive_fixed_step_profile_measurements(
        profile_trace,
        profile_commitment,
        warmup_optimizer_events=50,
        measured_optimizer_events=200,
        expected_identity=training_identity,
        profiler_measurements=synchronized_profile_measurements,
        runtime_binding=profile_runtime_session.binding,
    )
    profile = _write_json(
        run_dir / "profile.json",
        profile_runtime_session.sign_profile(
            {
                "schema_version": MODULE.PROFILE_SCHEMA,
                "status": "PASS",
                "commit_sha": PROFILE_COMMIT,
                "source_tree_sha256": "9" * 64,
                "data_identity_sha256": data_identity["identity_sha256"],
                "runtime_identity_sha256": MODULE._sha256_json(
                    profile_runtime, "profile runtime"
                ),
                "resolved_config_sha256": resolved_cfg_sha256,
                "scientific_config_sha256": scientific_cfg_sha256,
                "launch_ticket_path": profile_ticket.name,
                "launch_ticket_sha256": sha256_file(profile_ticket),
                "launch_receipt_path": profile_receipt.name,
                "launch_receipt_sha256": sha256_file(profile_receipt),
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
                "measurements": profile_measurements,
                "optimizer_event_trace": _reference(profile_trace, run_dir),
                "optimizer_event_commitment": _reference(profile_commitment, run_dir),
            },
        ),
    )
    profile_reference = _reference(profile, run_dir)

    def launch_evidence(stage, entrypoint, resume_checkpoint, runtime_session):
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
            "resolved_config_sha256": resolved_cfg_sha256,
            "scientific_config_sha256": scientific_cfg_sha256,
            "data_identity": data_identity,
            "runtime_identity": runtime,
            "b0_evidence": formal_b0_reference,
            "review_evidence": review_reference,
            "profile_evidence": profile_reference,
        }
        ticket_path = _write_json(run_dir / f"{stage}-launch-ticket.json", ticket)
        job_id = f"{seed}-{stage}"
        receipt = _sign_payload(
            {
                "schema_version": MODULE.LAUNCH_RECEIPT_SCHEMA,
                "mode": "formal",
                "commit_sha": COMMIT,
                "source_tree_sha256": ticket["source_tree_sha256"],
                "data_identity_sha256": data_identity["identity_sha256"],
                "runtime_identity_sha256": MODULE._sha256_json(runtime, "runtime"),
                "resolved_config_sha256": ticket["resolved_config_sha256"],
                "scientific_config_sha256": ticket["scientific_config_sha256"],
                "launch_ticket": _reference(ticket_path, run_dir),
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
                "execution_session": runtime_session.binding,
            },
            private_key_path=profile_private,
            key_id="profile-test",
            role=MODULE.LAUNCH_RECEIPT_ATTESTATION_ROLE,
        )
        receipt_path = _write_json(
            run_dir / f"{stage}-launch-receipt.json", receipt
        )
        return ticket_path, receipt_path

    training_ticket, training_receipt = launch_evidence(
        "training", "train", None, training_runtime_session
    )
    evaluation_ticket, evaluation_receipt = launch_evidence(
        "evaluation",
        "test",
        {
            "path": checkpoint.name,
            "sha256": sha256_file(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
        },
        issue_runtime_session(),
    )
    artifacts = {
        "ledger": ledger_path,
        "commitment": commitment_path,
        "ground_truth": ground_truth,
        "allowed_videos": allowed,
        "fit_core": fit_core,
        "feature_cache_manifest": cache_manifest,
        "evaluator_spec": evaluator_spec,
        "config": config,
        "resolved_config": resolved_config,
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
        visual_trace = run_dir / "visual-parameters.jsonl"
        visual_commitment = run_dir / "visual-parameters.commitment.json"
        before = torch.zeros_like(checkpoint_parameter) if adapted else checkpoint_parameter
        gradient = torch.ones_like(checkpoint_parameter) if adapted else None
        visual_event = {
            "optimizer_event_id": f"optimizer-{seed}",
            "parameter_name": "visual.encoder.weight",
            "requires_grad": adapted,
            "optimizer_member": adapted,
            "numel": checkpoint_parameter.numel(),
            "dtype": str(checkpoint_parameter.dtype),
            "before_sha256": tensor_sha256(before),
            "after_sha256": tensor_sha256(checkpoint_parameter),
            "gradient_sha256": None if gradient is None else tensor_sha256(gradient),
            "gradient_finite": adapted,
            "gradient_norm": 1.0 if adapted else 0.0,
            "delta_norm": float(checkpoint_parameter.norm().item()) if adapted else 0.0,
        }
        persist_visual_parameter_trace(
            visual_trace,
            visual_commitment,
            [
                _signed_runtime_event(
                    training_runtime_session,
                    "visual-parameter-event",
                    visual_event,
                )
            ],
        )
        artifacts["visual_parameter_trace"] = visual_trace
        artifacts["visual_parameter_commitment"] = visual_commitment
    if evidence_mutator is not None:
        artifacts["_profile_private"] = profile_private
        try:
            evidence_mutator(artifacts)
        finally:
            artifacts.pop("_profile_private", None)
    protocol = _protocol(claim)
    signed = build_formal_run_manifest(
        claim=claim,
        variant=variant,
        seed=seed,
        commit_sha=COMMIT,
        protocol_sha256=MODULE._sha256_json(protocol, "protocol"),
        artifacts=artifacts,
        bundle_root=run_dir,
        private_key_path=formal_private,
        key_id="formal-test",
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
    (
        formal_private,
        b0_private,
        review_private,
        profile_private,
        fineaction_license_private,
        fineaction_execution_private,
        fineaction_roots,
    ) = _keys(root, monkeypatch)
    b0 = _b0_evidence(root, b0_private)
    profile_b0 = _b0_evidence(
        root / "profile-prerequisites",
        b0_private,
        commit=PROFILE_COMMIT,
    )
    reporting_evidence, fineaction_path = _reporting_evidence(
        root,
        license_private=fineaction_license_private,
        execution_private=fineaction_execution_private,
        trust_roots=fineaction_roots,
    )
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
                    profile_b0=profile_b0,
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
                    profile_b0=profile_b0,
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
                    profile_b0=profile_b0,
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
                    profile_b0=profile_b0,
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
    manifest_path = Path(report["runs"][0]["evidence"]["run_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ledger = manifest_path.parent / manifest["artifacts"]["ledger"]["path"]
    ledger.write_text(ledger.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(MODULE.ResultGateInputError, match="hash mismatch"):
        MODULE.evaluate_result_gates(report)


def test_gate_rejects_rehashed_reporting_and_fineaction_semantic_forgery(
    tmp_path, monkeypatch
):
    report = _report(tmp_path, monkeypatch)
    reporting = report["reporting_evidence"]

    observed_path = Path(reporting["observed_ids"]["path"])
    observed_original = observed_path.read_bytes()
    observed = json.loads(observed_original)
    observed[-1] = "forged_extra"
    _write_json(observed_path, observed)
    reporting["observed_ids"]["sha256"] = sha256_file(observed_path)
    with pytest.raises(MODULE.ResultGateInputError, match="source-derived.*provenance"):
        MODULE.evaluate_result_gates(report)

    observed_path.write_bytes(observed_original)
    reporting["observed_ids"]["sha256"] = sha256_file(observed_path)
    fineaction_path = Path(reporting["fineaction"]["path"])
    fineaction = json.loads(fineaction_path.read_text(encoding="utf-8"))
    fineaction["gates"]["completeness"]["checks"]["qualified_ground_truth"][
        "evidence"
    ]["count"] = 3000
    fineaction.pop("content_sha256")
    fineaction["content_sha256"] = MODULE._sha256_json(
        fineaction, "forged FineAction"
    )
    _write_json(fineaction_path, fineaction)
    reporting["fineaction"]["sha256"] = sha256_file(fineaction_path)
    with pytest.raises(
        MODULE.ResultGateInputError,
        match="evidence hash mismatch|not source-derived",
    ):
        MODULE.evaluate_result_gates(report)


def _semantic_run(tmp_path, monkeypatch, mutator):
    root = tmp_path / "semantic-run"
    root.mkdir(parents=True)
    (
        formal_private,
        b0_private,
        review_private,
        profile_private,
        fineaction_license_private,
        fineaction_execution_private,
        fineaction_roots,
    ) = _keys(root, monkeypatch)
    b0 = _b0_evidence(root, b0_private)
    profile_b0 = _b0_evidence(
        root / "profile-prerequisites",
        b0_private,
        commit=PROFILE_COMMIT,
    )
    _, fineaction_path = _reporting_evidence(
        root,
        license_private=fineaction_license_private,
        execution_private=fineaction_execution_private,
        trust_roots=fineaction_roots,
    )
    return _run(
        root,
        "C1",
        "fixed",
        11,
        formal_private=formal_private,
        review_private=review_private,
        profile_private=profile_private,
        b0=b0,
        profile_b0=profile_b0,
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
        receipt["launch_ticket"] = _reference(ticket_path, ticket_path.parent)
        receipt["runtime_identity_sha256"] = MODULE._sha256_json(
            ticket["runtime_identity"], "runtime"
        )
        _write_json(receipt_path, receipt)

    with pytest.raises(TrainingEvidenceError, match="checkpoint hash mismatch"):
        _semantic_run(tmp_path, monkeypatch, wrong_checkpoint)


def test_gate_rejects_signed_receipt_without_active_slurm_allocation(
    tmp_path, monkeypatch
):
    def completed_job(artifacts):
        receipt_path = artifacts["training_launch_receipt"]
        signed_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt = {key: value for key, value in signed_receipt.items() if key != "attestation"}
        receipt["slurm_allocation"]["state"] = "COMPLETED"
        _write_json(
            receipt_path,
            _sign_payload(
                receipt,
                private_key_path=artifacts["_profile_private"],
                key_id="profile-test",
                role=MODULE.LAUNCH_RECEIPT_ATTESTATION_ROLE,
            ),
        )

    with pytest.raises(TrainingEvidenceError, match="active one-GPU"):
        _semantic_run(tmp_path, monkeypatch, completed_job)


@pytest.mark.parametrize(
    ("role", "message"),
        (
            ("ground_truth", "ground-truth artifact path differs"),
            ("allowed_videos", "allowed-videos artifact path differs"),
        ),
)
def test_formal_signer_rejects_data_artifact_substitution(
    tmp_path, monkeypatch, role, message
):
    def substitute(artifacts):
        original = artifacts[role]
        replacement = original.with_name(f"substituted-{original.name}")
        replacement.write_bytes(original.read_bytes())
        artifacts[role] = replacement

    with pytest.raises(TrainingEvidenceError, match=message):
        _semantic_run(tmp_path, monkeypatch, substitute)


@pytest.mark.parametrize("role", ("fit_core", "feature_cache_manifest"))
def test_formal_signer_rejects_training_order_artifact_content_substitution(
    tmp_path, monkeypatch, role
):
    def substitute(artifacts):
        artifacts[role].write_bytes(artifacts[role].read_bytes() + b" ")

    with pytest.raises(TrainingEvidenceError, match=f"data identity {role} file hash differs"):
        _semantic_run(tmp_path, monkeypatch, substitute)


def test_formal_signer_rejects_resolved_config_not_bound_to_ticket(
    tmp_path, monkeypatch
):
    def substitute(artifacts):
        path = artifacts["resolved_config"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["model"]["type"] = "SubstitutedDetector"
        _write_json(path, payload)

    with pytest.raises(TrainingEvidenceError, match="resolved config digest differs"):
        _semantic_run(tmp_path, monkeypatch, substitute)


def test_formal_signer_rejects_evaluator_spec_substitution(tmp_path, monkeypatch):
    def substitute(artifacts):
        spec_path = artifacts["evaluator_spec"]
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        spec["latency_budgets_sec"] = [0.25, 0.5]
        _write_json(spec_path, spec)

    with pytest.raises(TrainingEvidenceError, match="evaluator specification differs"):
        _semantic_run(tmp_path, monkeypatch, substitute)


def test_formal_signer_rejects_self_reported_training_identity(tmp_path, monkeypatch):
    def substitute(artifacts):
        trace = artifacts["training_trace"]
        commitment = artifacts["training_commitment"]
        trace.unlink()
        commitment.unlink()
        runtime_session = issue_runtime_session()
        event = _training_event(11)
        event["optimizer_config_sha256"] = "f" * 64
        persist_training_trace(
            trace,
            commitment,
            [_signed_runtime_event(runtime_session, "optimizer-event", event)],
        )
        receipt_path = artifacts["training_launch_receipt"]
        signed_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt = {
            key: value
            for key, value in signed_receipt.items()
            if key != "attestation"
        }
        receipt["execution_session"] = runtime_session.binding
        _write_json(
            receipt_path,
            _sign_payload(
                receipt,
                private_key_path=artifacts["_profile_private"],
                key_id="profile-test",
                role=MODULE.LAUNCH_RECEIPT_ATTESTATION_ROLE,
            ),
        )

    with pytest.raises(TrainingEvidenceError, match="training trace identity differs"):
        _semantic_run(tmp_path, monkeypatch, substitute)


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
        prerequisite_path = (
            ticket_paths[0].parent / first_ticket[f"{role}_evidence"]["path"]
        )
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
            receipt["launch_ticket"] = _reference(ticket_path, ticket_path.parent)
            receipt[receipt_field] = prerequisite_hash
            _write_json(receipt_path, receipt)

    with pytest.raises(TrainingEvidenceError, match="receipt is not trusted"):
        _semantic_run(tmp_path, monkeypatch, unsigned_prerequisite)


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
