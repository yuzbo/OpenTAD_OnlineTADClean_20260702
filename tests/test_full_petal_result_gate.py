import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import uuid

import pytest

from opentad.utils.immutable_event_ledger import (
    ImmutableEventLedger,
    persist_verified_ledger,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "check_full_petal_results.py"
SPEC = importlib.util.spec_from_file_location("check_full_petal_results", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_json(path, payload):
    path.write_text(json.dumps(payload, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(payload):
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
        "capacity": 8,
        "tracker": "fixed_across_comparison",
    }


def _run_row(claim, variant, seed, score, *, error_rate=0.10, protocol=None):
    row = {
        "claim": claim,
        "variant": variant,
        "seed": seed,
        "protocol": _protocol(claim) if protocol is None else protocol,
        "metrics": {
            "average_mAP": score,
            "recall": 0.70,
            "duplicate_rate": error_rate,
            "fragmentation_rate": error_rate,
            "false_emission_rate": 0.08,
            "endpoint_latency_frames_mean": 2.0,
        },
        "protocol_violations": 0,
    }
    if claim == "C1":
        row["cost"] = {
            "optimizer_events": 10,
            "successful_optimizer_events": 10,
            "skipped_optimizer_events": 0,
            "input_tokens": 128,
            "episodes": 2,
            "effective_batch_size": 2,
            "gpu_hours": 1.0,
            "wall_clock_sec": 100.0,
            "peak_vram_gb": 8.0,
            "precision": "bf16",
            "optimizer_config_sha256": "1" * 64,
            "scheduler_config_sha256": "2" * 64,
            "data_order_sha256": "3" * 64,
            "loss_normalization_sha256": "4" * 64,
        }
    return row


def _materialize_run_evidence(root, row, *, include_raw_visual_audit=True):
    claim = row["claim"]
    variant = row["variant"]
    seed = row["seed"]
    run_dir = root / f"{claim}-{variant}-{seed}"
    run_dir.mkdir(parents=True)

    config_path = (run_dir / "config.py")
    config_path.write_text(f"claim={claim!r}\nvariant={variant!r}\n", encoding="utf-8")
    model_path = run_dir / "model.bin"
    model_path.write_bytes(f"model:{claim}:{variant}:{seed}".encode("ascii"))
    dataset_path = _write_json(
        run_dir / "dataset-manifest.json",
        {"dataset": "fixture", "seed": seed, "split_sha256": "5" * 64},
    )
    evaluator_path = run_dir / "evaluator.py"
    evaluator_path.write_text("EVALUATOR_VERSION = 1\n", encoding="utf-8")
    metrics_path = _write_json(run_dir / "metrics.json", row["metrics"])

    ledger_builder = ImmutableEventLedger()
    ledger_builder.append(
        {
            "event_id": f"{claim}-{variant}-{seed}-event",
            "stream_id": f"video-{seed}",
            "video_id": f"video-{seed}",
            "start_frame": 1,
            "end_frame": 2,
            "emit_frame": 2,
            "source_frame": 2,
            "slot_id": 0,
            "label": 1,
            "score": 0.75,
            "provenance_digest": "6" * 64,
            "immutable": True,
        }
    )
    ledger_path = run_dir / "emissions.jsonl"
    commitment_path = run_dir / "emissions.commitment.json"
    commitment = persist_verified_ledger(
        ledger_path,
        commitment_path,
        ledger_builder.rows,
    )

    hashes = {
        "ledger": _sha256_file(ledger_path),
        "commitment": _sha256_file(commitment_path),
        "config": _sha256_file(config_path),
        "model": _sha256_file(model_path),
        "dataset_manifest": _sha256_file(dataset_path),
        "evaluator": _sha256_file(evaluator_path),
        "metrics": _sha256_file(metrics_path),
    }
    paths = {
        "ledger": ledger_path,
        "commitment": commitment_path,
        "config": config_path,
        "model": model_path,
        "dataset_manifest": dataset_path,
        "evaluator": evaluator_path,
        "metrics": metrics_path,
    }

    cost_hash = None
    if claim == "C1":
        cost_path = _write_json(run_dir / "cost.json", row["cost"])
        paths["cost"] = cost_path
        hashes["cost"] = _sha256_file(cost_path)
        cost_hash = hashes["cost"]

    raw_visual_hash = None
    if claim == "C2" and include_raw_visual_audit:
        adapted = variant == "adapted"
        raw_visual_path = _write_json(
            run_dir / "raw-visual-audit.json",
            {
                "schema_version": MODULE.RAW_VISUAL_AUDIT_SCHEMA,
                "claim": claim,
                "variant": variant,
                "seed": seed,
                "model_sha256": hashes["model"],
                "dataset_manifest_sha256": hashes["dataset_manifest"],
                "registered_visual_params": 2 if adapted else 0,
                "nonzero_finite_grad_params": 2 if adapted else 0,
                "changed_visual_params": 1 if adapted else 0,
                "frozen_param_delta_max": 0.0,
                "status": "PASS",
            },
        )
        paths["raw_visual_audit"] = raw_visual_path
        hashes["raw_visual_audit"] = _sha256_file(raw_visual_path)
        raw_visual_hash = hashes["raw_visual_audit"]

    run_manifest_path = _write_json(
        run_dir / "run-manifest.json",
        {
            "schema_version": MODULE.RUN_MANIFEST_SCHEMA,
            "claim": claim,
            "variant": variant,
            "seed": seed,
            "ledger_sha256": hashes["ledger"],
            "commitment_sha256": hashes["commitment"],
            "config_sha256": hashes["config"],
            "model_sha256": hashes["model"],
            "dataset_manifest_sha256": hashes["dataset_manifest"],
            "evaluator_sha256": hashes["evaluator"],
            "metrics_sha256": hashes["metrics"],
            "cost_sha256": cost_hash,
            "protocol_sha256": _sha256_json(row["protocol"]),
            "raw_visual_audit_sha256": raw_visual_hash,
        },
    )
    paths["run_manifest"] = run_manifest_path
    hashes["run_manifest"] = _sha256_file(run_manifest_path)

    evidence = {
        "ledger_count": commitment["count"],
        "ledger_final_hashes": commitment["final_hashes"],
    }
    for prefix, path in paths.items():
        evidence[f"{prefix}_path"] = str(path)
        evidence[f"{prefix}_sha256"] = hashes[prefix]
    row["evidence"] = evidence


def _b0_evidence(root):
    commit_sha = "7" * 40
    test_report = _write_json(
        root / "b0-tests.json",
        {
            "schema_version": MODULE.B0_TEST_REPORT_SCHEMA,
            "status": "PASS",
            "commit_sha": commit_sha,
            "collected": 41,
            "passed": 41,
            "failed": 0,
        },
    )
    audit_report = _write_json(
        root / "b0-audit.json",
        {
            "schema_version": MODULE.B0_AUDIT_REPORT_SCHEMA,
            "status": "PASS",
            "commit_sha": commit_sha,
            "blocking_findings": 0,
            "protocol_violations": 0,
        },
    )
    b0_path = _write_json(
        root / "b0.json",
        {
            "schema_version": MODULE.B0_SCHEMA,
            "status": "PASS",
            "commit_sha": commit_sha,
            "test_count": 41,
            "blocking_findings": 0,
            "protocol_violations": 0,
            "test_report_path": str(test_report),
            "test_report_sha256": _sha256_file(test_report),
            "audit_report_path": str(audit_report),
            "audit_report_sha256": _sha256_file(audit_report),
        },
    )
    return {"path": str(b0_path), "sha256": _sha256_file(b0_path)}


def _report(tmp_path, *, c1_pass=True, c2_pass=True, b0_pass=True):
    root = tmp_path / f"report-{uuid.uuid4().hex}"
    root.mkdir()
    rows = []
    for seed in (11, 12):
        rows.extend(
            [
                _run_row("C1", "rematch", seed, 0.10, error_rate=0.20),
                _run_row(
                    "C1",
                    "fixed",
                    seed,
                    0.13 if c1_pass else 0.101,
                    error_rate=0.10 if c1_pass else 0.19,
                ),
                _run_row("C2", "frozen", seed, 0.10),
                _run_row("C2", "adapted", seed, 0.13 if c2_pass else 0.105),
            ]
        )
    for row in rows:
        _materialize_run_evidence(root, row)
    report = {"runs": rows}
    if b0_pass:
        report["b0_evidence"] = _b0_evidence(root)
    return report


def _refresh_run_manifest(row):
    path = Path(row["evidence"]["run_manifest_path"])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["protocol_sha256"] = _sha256_json(row["protocol"])
    _write_json(path, manifest)
    row["evidence"]["run_manifest_sha256"] = _sha256_file(path)


def _run_cli(report_path, *extra):
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(report_path), *extra],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_c1_pass_does_not_make_the_independent_c2_gate_pass(tmp_path):
    verdict = MODULE.evaluate_result_gates(_report(tmp_path, c1_pass=True, c2_pass=False))

    assert set(verdict) == {"schema_version", "decisions", "prerequisites"}
    assert verdict["decisions"]["C1"]["status"] == "PASS"
    assert verdict["decisions"]["C1"]["paired_seed_coverage"]["passed"] is True
    assert verdict["decisions"]["C1"]["protocol_equality"]["passed"] is True
    assert verdict["decisions"]["C1"]["scientific_metrics"]["passed"] is True
    assert verdict["decisions"]["C1"]["cost_parity"]["passed"] is True
    assert verdict["decisions"]["C2"]["status"] == "FAIL"
    assert verdict["decisions"]["project_full_system"]["status"] == "FAIL"
    assert "C1" not in verdict["decisions"]["C2"]["requirements"]


def test_c2_can_pass_when_c1_fails_but_the_project_gate_cannot(tmp_path):
    verdict = MODULE.evaluate_result_gates(_report(tmp_path, c1_pass=False, c2_pass=True))

    assert verdict["decisions"]["C1"]["status"] == "FAIL"
    assert verdict["decisions"]["C2"]["status"] == "PASS"
    assert verdict["decisions"]["C2"]["raw_visual_audit"]["passed"] is True
    assert verdict["decisions"]["project_full_system"]["status"] == "FAIL"
    assert verdict["decisions"]["project_full_system"]["requirements"] == {
        "C1": False,
        "C2": True,
        "protocol": True,
        "B0": True,
    }


def test_project_gate_requires_c1_c2_protocol_and_b0_separately(tmp_path):
    without_b0 = MODULE.evaluate_result_gates(_report(tmp_path, b0_pass=False))
    complete = MODULE.evaluate_result_gates(_report(tmp_path))

    assert without_b0["decisions"]["C1"]["status"] == "PASS"
    assert without_b0["decisions"]["C2"]["status"] == "PASS"
    assert without_b0["decisions"]["project_full_system"]["status"] == "FAIL"
    assert complete["decisions"]["project_full_system"]["status"] == "PASS"


def test_gate_rejects_unpaired_c1_seeds_and_protocol_mismatch(tmp_path):
    report = _report(tmp_path)
    report["runs"] = [
        row
        for row in report["runs"]
        if not (row["claim"] == "C1" and row["variant"] == "fixed" and row["seed"] == 12)
    ]
    with pytest.raises(MODULE.ResultGateInputError, match="paired seed coverage"):
        MODULE.evaluate_result_gates(report)

    report = _report(tmp_path)
    report["runs"][0]["protocol"] = {
        **report["runs"][0]["protocol"],
        "tracker": "different",
    }
    _refresh_run_manifest(report["runs"][0])
    with pytest.raises(MODULE.ResultGateProtocolError, match="protocol mismatch"):
        MODULE.evaluate_result_gates(report)


def test_cli_scientific_fail_is_valid_unless_require_pass_is_requested(tmp_path):
    report_path = tmp_path / "scientific-fail.json"
    report_path.write_text(
        json.dumps(_report(tmp_path, c1_pass=False, c2_pass=True)),
        encoding="utf-8",
    )

    valid_fail = _run_cli(report_path)
    required_pass = _run_cli(report_path, "--require-pass")

    assert valid_fail.returncode == 0, valid_fail.stderr
    assert json.loads(valid_fail.stdout)["decisions"]["C1"]["status"] == "FAIL"
    assert required_pass.returncode == 4
    assert json.loads(required_pass.stdout)["decisions"]["project_full_system"]["status"] == "FAIL"


def test_cli_returns_nonzero_json_errors_for_malformed_or_protocol_invalid_input(tmp_path):
    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text(json.dumps({"runs": [{"claim": "C1"}]}), encoding="utf-8")

    malformed = _run_cli(malformed_path)

    assert malformed.returncode == 2
    assert json.loads(malformed.stdout)["error"]["kind"] == "malformed_input"

    report = _report(tmp_path)
    report["runs"][0]["protocol_violations"] = 1
    invalid_path = tmp_path / "protocol-invalid.json"
    invalid_path.write_text(json.dumps(report), encoding="utf-8")

    invalid = _run_cli(invalid_path)

    assert invalid.returncode == 3
    assert json.loads(invalid.stdout)["error"]["kind"] == "protocol_violation"


def test_gate_rejects_self_attested_pass_without_verified_artifacts(tmp_path):
    report = _report(tmp_path)
    for row in report["runs"]:
        row.pop("evidence")
    report["prerequisites"] = {
        "protocol": {"passed": True, "violations": []},
        "B0": {"passed": True},
    }

    with pytest.raises(MODULE.ResultGateInputError, match="verified ledger evidence"):
        MODULE.evaluate_result_gates(report)


def test_gate_rejects_placeholder_cost_and_online_ap_alias(tmp_path):
    report = _report(tmp_path)
    for row in report["runs"]:
        if row["claim"] == "C1":
            row["cost"] = {"bogus": 1}
        row["metrics"]["average_mOnlineAP"] = row["metrics"].pop("average_mAP")

    with pytest.raises(MODULE.ResultGateInputError, match="evidence|cost|average_mAP|metrics"):
        MODULE.evaluate_result_gates(report)


def test_c2_cannot_pass_without_raw_visual_audit(tmp_path):
    report = _report(tmp_path)
    c2_row = next(row for row in report["runs"] if row["claim"] == "C2")
    c2_row["evidence"].pop("raw_visual_audit_path")
    c2_row["evidence"].pop("raw_visual_audit_sha256")

    with pytest.raises(MODULE.ResultGateInputError, match="raw_visual_audit"):
        MODULE.evaluate_result_gates(report)


def test_gate_rejects_tampered_ledger_and_self_attested_b0(tmp_path):
    report = _report(tmp_path)
    ledger_path = Path(report["runs"][0]["evidence"]["ledger_path"])
    ledger_path.write_text(ledger_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(MODULE.ResultGateInputError, match="hash mismatch"):
        MODULE.evaluate_result_gates(report)

    report = _report(tmp_path)
    report.pop("b0_evidence")
    report["b0_passed"] = True
    with pytest.raises(MODULE.ResultGateInputError, match="self-attested"):
        MODULE.evaluate_result_gates(report)
