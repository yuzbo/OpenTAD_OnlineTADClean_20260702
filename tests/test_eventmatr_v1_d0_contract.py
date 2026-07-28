from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_d0_replay_is_train_only_fresh_and_checkpoint_read_only():
    source = (ROOT / "scripts" / "run_eventmatr_v1_d0_audit.py").read_text(
        encoding="utf-8"
    )
    assert 'THUMOS14Dataset(args, subset="train")' in source
    assert 'subset="test"' not in source
    assert 'proposal_txt.write_text("", encoding="utf-8")' in source
    assert 'model.load_state_dict(checkpoint["state_dict"], strict=True)' in source
    assert '"checkpoint_updated": False' in source
    assert "optimizer.step(" not in source
    assert "model.eval()" in source
    assert '"cross_video_batches": cross_video_batches' in source
    assert "supervised D0 gradient batch crosses a video boundary" in source


def test_d0_slurm_releases_exact_four_event_lanes_without_test():
    worker = (ROOT / "scripts" / "slurm_eventmatr_v1_d0.sh").read_text(
        encoding="utf-8"
    )
    finalizer = (
        ROOT / "scripts" / "slurm_eventmatr_v1_d0_finalize.sh"
    ).read_text(encoding="utf-8")
    submit = (ROOT / "scripts" / "submit_eventmatr_v1_d0.sh").read_text(
        encoding="utf-8"
    )
    assert "LANES=(b0o0 b1o0 b0o1 b1o1)" in worker
    assert "--array=0-3%4" in submit
    assert "D0 audit forbids locked-test inputs" in worker
    assert "#SBATCH --mem=" not in worker
    assert "#SBATCH --mem=" not in finalizer
    assert "#SBATCH --gpus=1" in worker
    assert "#SBATCH --gres=" not in worker
    assert "D0 audit forbids locked-test inputs" in submit
    assert "afterok:${ARRAY_JOB}" in submit
    assert "D0 output root already exists" in submit


def test_d0_manifest_declares_diagnostic_not_paper_performance():
    manifest = (
        ROOT / "experiment_configs" / "eventmatr_v1_d0_audit.json"
    ).read_text(encoding="utf-8")
    assert '"locked_test_access": false' in manifest
    assert '"raw_rgb_claim": false' in manifest
    assert '"paper_performance_result": false' in manifest
    assert '"strict_causal_valid": false' in manifest


def test_d0_finalizer_binds_every_lane_to_both_source_identities():
    finalizer = (ROOT / "scripts" / "finalize_eventmatr_v1_d0.py").read_text(
        encoding="utf-8"
    )
    assert "expected_diagnostic_commit" in finalizer
    assert "expected_diagnostic_tree" in finalizer
    assert "expected_training_commit" in finalizer
    assert "expected_training_tree" in finalizer
    assert '"D0 receipt set"' in finalizer
    assert "shared dataset identity" in finalizer
    assert "refusing to overwrite an existing D0 completion receipt" in finalizer
