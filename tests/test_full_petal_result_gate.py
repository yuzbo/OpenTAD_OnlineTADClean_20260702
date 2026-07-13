import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "check_full_petal_results.py"
SPEC = importlib.util.spec_from_file_location("check_full_petal_results", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _run_row(claim, variant, seed, score, *, error_rate=0.10, protocol=None):
    if protocol is None:
        protocol = {
            "input": "matched_features" if claim == "C1" else "raw_video",
            "decision_cadence": "packet_end",
            "nms": False,
            "tracker": "fixed_across_comparison",
        }
    row = {
        "claim": claim,
        "variant": variant,
        "seed": seed,
        "protocol": protocol,
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
            "gpu_hours": 1.0,
            "wall_clock_sec": 100.0,
            "peak_vram_gb": 8.0,
        }
    return row


def _report(*, c1_pass=True, c2_pass=True, protocol_pass=True, b0_pass=True):
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
    return {
        "prerequisites": {
            "protocol": {"passed": protocol_pass, "violations": []},
            "B0": {"passed": b0_pass},
        },
        "runs": rows,
    }


def _run_cli(report_path, *extra):
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(report_path), *extra],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_c1_pass_does_not_make_the_independent_c2_gate_pass():
    verdict = MODULE.evaluate_result_gates(_report(c1_pass=True, c2_pass=False))

    assert set(verdict) == {"schema_version", "decisions", "prerequisites"}
    assert verdict["decisions"]["C1"]["status"] == "PASS"
    assert verdict["decisions"]["C1"]["paired_seed_coverage"]["passed"] is True
    assert verdict["decisions"]["C1"]["protocol_equality"]["passed"] is True
    assert verdict["decisions"]["C1"]["scientific_metrics"]["passed"] is True
    assert verdict["decisions"]["C1"]["cost_parity"]["passed"] is True
    assert verdict["decisions"]["C2"]["status"] == "FAIL"
    assert verdict["decisions"]["project_full_system"]["status"] == "FAIL"
    assert "C1" not in verdict["decisions"]["C2"]["requirements"]


def test_c2_can_pass_when_c1_fails_but_the_project_gate_cannot():
    verdict = MODULE.evaluate_result_gates(_report(c1_pass=False, c2_pass=True))

    assert verdict["decisions"]["C1"]["status"] == "FAIL"
    assert verdict["decisions"]["C2"]["status"] == "PASS"
    assert verdict["decisions"]["project_full_system"]["status"] == "FAIL"
    assert verdict["decisions"]["project_full_system"]["requirements"] == {
        "C1": False,
        "C2": True,
        "protocol": True,
        "B0": True,
    }


def test_project_gate_requires_c1_c2_protocol_and_b0_separately():
    without_b0 = MODULE.evaluate_result_gates(_report(b0_pass=False))
    complete = MODULE.evaluate_result_gates(_report())

    assert without_b0["decisions"]["C1"]["status"] == "PASS"
    assert without_b0["decisions"]["C2"]["status"] == "PASS"
    assert without_b0["decisions"]["project_full_system"]["status"] == "FAIL"
    assert complete["decisions"]["project_full_system"]["status"] == "PASS"


def test_gate_rejects_unpaired_c1_seeds_and_protocol_mismatch():
    report = _report()
    report["runs"] = [
        row
        for row in report["runs"]
        if not (row["claim"] == "C1" and row["variant"] == "fixed" and row["seed"] == 12)
    ]
    with pytest.raises(MODULE.ResultGateInputError, match="paired seed coverage"):
        MODULE.evaluate_result_gates(report)

    report = _report()
    report["runs"][0]["protocol"] = {
        **report["runs"][0]["protocol"],
        "decision_cadence": "window_end",
    }
    with pytest.raises(MODULE.ResultGateProtocolError, match="protocol mismatch"):
        MODULE.evaluate_result_gates(report)


def test_cli_scientific_fail_is_valid_unless_require_pass_is_requested(tmp_path):
    report_path = tmp_path / "scientific-fail.json"
    report_path.write_text(
        json.dumps(_report(c1_pass=False, c2_pass=True)),
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

    report = _report()
    report["runs"][0]["protocol_violations"] = 1
    invalid_path = tmp_path / "protocol-invalid.json"
    invalid_path.write_text(json.dumps(report), encoding="utf-8")

    invalid = _run_cli(invalid_path)

    assert invalid.returncode == 3
    assert json.loads(invalid.stdout)["error"]["kind"] == "protocol_violation"
