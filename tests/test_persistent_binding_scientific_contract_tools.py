import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module(relative_path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def test_split_census_passes_a_complete_nonoverlapping_synthetic_route(tmp_path):
    census = _module(
        "tools/census_persistent_binding.py",
        "census_persistent_binding",
    )
    annotation_path = tmp_path / "annotation.json"
    cache_path = tmp_path / "cache.json"
    split_paths = {
        "fit_core": tmp_path / "fit.txt",
        "calibration": tmp_path / "calibration.txt",
        "reporting_locked": tmp_path / "reporting.txt",
    }
    videos = {
        "fit": {
            "subset": "training",
            "duration": 10.0,
            "frame": 80,
            "annotations": [{"label": "A", "segment": [1.0, 2.0]}],
        },
        "cal": {
            "subset": "training",
            "duration": 10.0,
            "frame": 80,
            "annotations": [{"label": "A", "segment": [1.0, 2.0]}],
        },
        "report": {
            "subset": "validation",
            "duration": 10.0,
            "frame": 80,
            "annotations": [{"label": "A", "segment": [1.0, 2.0]}],
        },
    }
    _write_json(annotation_path, {"database": videos})
    source_frames = list(range(7, 80, 8))
    _write_json(
        cache_path,
        {
            "annotation_sha256": _sha256(annotation_path),
            "encoder_id": "frozen-test-encoder",
            "feature_stride": 8,
            "videos": {
                video_id: {
                    "num_tokens": len(source_frames),
                    "source_frames": source_frames,
                }
                for video_id in videos
            },
        },
    )
    for path, video_id in zip(
        split_paths.values(),
        ("fit", "cal", "report"),
    ):
        path.write_text(video_id + "\n", encoding="utf-8")
    config_path = tmp_path / "config.py"
    config_path.write_text(
        "\n".join(
            [
                "route_stage = 'persistent_binding_test'",
                f"annotation_path = {str(annotation_path)!r}",
                f"feature_cache_manifest = {str(cache_path)!r}",
                f"fit_core_manifest = {str(split_paths['fit_core'])!r}",
                f"calibration_manifest = {str(split_paths['calibration'])!r}",
                f"reporting_manifest = {str(split_paths['reporting_locked'])!r}",
                "feature_stride = 8",
                "num_slots = 2",
                "memory_size = 4",
                "model = dict(head=dict(max_births_per_step=1))",
                "census_contract = dict(",
                "    expected_split_counts=dict(fit_core=1, calibration=1, reporting_locked=1),",
                "    max_gt_entry_free_deficits=0,",
                "    max_oracle_capacity_overflow_steps=0,",
                "    max_uncovered_birth_instances=0,",
                "    max_uncovered_endpoint_instances=0,",
                "    max_clipped_start_supervision_targets=0,",
                ")",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = census.build_census(config_path)

    assert payload["passed"] is True
    assert payload["totals"]["videos"] == 3
    assert payload["totals"]["instances"] == 3
    assert payload["totals"]["max_births_per_step"] == 1
    assert payload["totals"]["uncovered_endpoint_instances"] == 0


def test_calibration_selection_is_deterministic_and_one_shot(tmp_path):
    selector = _module(
        "tools/select_persistent_binding_checkpoint.py",
        "select_persistent_binding_checkpoint",
    )
    common = {
        "schema_version": "persistent_binding_calibration_candidate.v1",
        "arm": "fixed",
        "seed": 705,
        "code_commit": "a" * 40,
        "config_sha256": "b" * 64,
        "calibration_manifest_sha256": "c" * 64,
        "census_sha256": "d" * 64,
        "calibration_protocol_sha256": "e" * 64,
        "metric_name": "average_mOnlineAP",
        "metric_unit": "fraction",
        "reporting_accessed": False,
    }
    candidates = []
    for epoch, metric in ((1, 0.4), (0, 0.4)):
        checkpoint = tmp_path / f"epoch_{epoch}.pth"
        checkpoint.write_bytes(f"checkpoint-{epoch}".encode())
        candidate = tmp_path / f"candidate_{epoch}.json"
        _write_json(
            candidate,
            {
                **common,
                "epoch": epoch,
                "metric_value": metric,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
            },
        )
        candidates.append(candidate)
    output = tmp_path / "receipt.json"

    receipt = selector.select_checkpoint(candidates, output)

    assert receipt["selected_epoch"] == 0
    assert receipt["reporting_accessed"] is False
    with pytest.raises(FileExistsError):
        selector.select_checkpoint(candidates, output)


def test_locked_reporting_refuses_a_second_run(tmp_path):
    reporter = _module(
        "tools/run_locked_persistent_binding_report.py",
        "run_locked_persistent_binding_report",
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo,
        check=True,
    )
    tracked = repo / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    config = tmp_path / "config.py"
    checkpoint = tmp_path / "checkpoint.pth"
    manifest = tmp_path / "reporting.txt"
    ledger = tmp_path / "ledger.json"
    calibration = tmp_path / "calibration.json"
    lock = tmp_path / "report.lock"
    receipt = tmp_path / "report.receipt.json"
    config.write_text("x = 1\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    manifest.write_text("video\n", encoding="utf-8")
    _write_json(
        calibration,
        {
            "schema_version": "persistent_binding_calibration_receipt.v1",
            "selected_checkpoint_sha256": _sha256(checkpoint),
            "reporting_accessed": False,
        },
    )
    command = [
        sys.executable,
        "-c",
        (
            "import json,sys;"
            "json.dump({'results': {}, 'summary': {}},"
            "open(sys.argv[1], 'w'))"
        ),
        str(ledger),
        "--evaluation-role",
        "reporting",
        "--checkpoint",
        str(checkpoint),
    ]

    payload = reporter.run_locked_report(
        repo=repo,
        config=config,
        checkpoint=checkpoint,
        calibration_receipt=calibration,
        reporting_manifest=manifest,
        ledger=ledger,
        lock=lock,
        receipt=receipt,
        command=command,
    )

    assert payload["status"] == "completed"
    assert payload["emission_ledger_sha256"] == _sha256(ledger)
    with pytest.raises(ValueError, match="already locked"):
        reporter.run_locked_report(
            repo=repo,
            config=config,
            checkpoint=checkpoint,
            calibration_receipt=calibration,
            reporting_manifest=manifest,
            ledger=ledger,
            lock=lock,
            receipt=receipt,
            command=command,
        )


def test_entrypoints_encode_fit_calibration_reporting_isolation():
    train = (ROOT / "tools/train.py").read_text(encoding="utf-8")
    test = (ROOT / "tools/test.py").read_text(encoding="utf-8")

    assert "--allow-unready-smoke" in train
    assert "formal_training_ready is false" in train
    assert "fit-only training forbids reporting-set evaluation" in train
    assert "training_audit.json" in train
    assert "--evaluation-role" in test
    assert "require_explicit_checkpoint" in test
