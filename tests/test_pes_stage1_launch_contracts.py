from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def test_stage1_smoke_tool_audits_real_parameter_updates_and_protocol_state():
    source = _read("tools/smoke_pes_stage1.py")

    assert "snapshot_trainable_parameters" in source
    assert "audit_training_update" in source
    assert "required_module_prefixes=(\"head\",)" in source
    assert "last_chunk_audit" in source
    assert "max_chunks" in source
    assert "load_from_raw_predictions" in source


def test_stage1_submitter_is_slurm_only_clean_checkout_and_budget_gated():
    source = _read("tools/remote/submit_pes_stage1_n16r4.sh")

    assert "MODE=${1:-smoke}" in source
    assert 'MODE" != "cache"' in source
    assert 'MODE" != "smoke"' in source
    assert 'MODE" != "pilot"' in source
    assert "status --porcelain" in source
    assert "rev-parse --is-inside-work-tree" in source
    assert "sbatch" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "HARD_BUDGET_GPU_HOURS=10" in source
    assert "PILOT_MAX_SECONDS=4000" in source
    assert "CACHE_TIME=${CACHE_TIME:-02:00:00}" in source
    assert "SMOKE_TIME=${SMOKE_TIME:-00:30:00}" in source
    assert "PILOT_TIME=${PILOT_TIME:-01:00:00}" in source
    assert "ALLOW_PILOT" in source
    assert "gate_summary.json" in source
    assert "705|706|707" in source
    assert ".pilot_registry" in source
    assert "mkdir \"$PILOT_MARKER\"" in source
    assert "cat >> \"$SCRIPT_PATH\" <<'SBATCH'" in source
    assert "thumos_pceh_ontad_finetune.py" in source
    assert "cache_ontad_features.py" in source
    assert "analyze_ontad_instances.py" in source
    assert "smoke_pes_stage1.py" in source
    assert "tests/test_pes_stage1_result_gate.py" in source
    assert "tests/test_pes_stage1_run_summary.py" in source
    assert "summarize_pes_stage1_run.py" in source
    assert "pes_stage1_emission_ledger.json" in source
    assert "audit_slot_exhaustion_total" not in source


def test_stage1_checker_reports_gate_and_slurm_accounting():
    source = _read("tools/remote/check_pes_stage1_n16r4.sh")

    assert "squeue" in source
    assert "sacct" in source
    assert "PYTHON_BIN=${PYTHON_BIN:-python3}" in source
    assert '"$PYTHON_BIN" - "$path"' in source
    assert "gate_summary.json" in source
    assert "train_smoke" in source
    assert "tail -n" in source
