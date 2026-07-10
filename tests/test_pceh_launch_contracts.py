from pathlib import Path

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "causaltad"


def _load(name):
    return Config.fromfile(CONFIG_DIR / name)


def test_pilot_configs_are_short_strict_and_comparable():
    main = _load("thumos_pceh_pilot.py")
    endpoint = _load("thumos_pceh_endpoint_only_pilot.py")

    assert main.route_stage == "pceh_short_pilot"
    assert endpoint.route_stage == "endpoint_only_short_pilot"
    assert main.workflow.end_epoch == 3
    assert endpoint.workflow.end_epoch == 3
    assert main.scheduler.max_epoch == 3
    assert endpoint.scheduler.max_epoch == 3
    assert main.scheduler.warmup_epoch == 1
    assert endpoint.scheduler.warmup_epoch == 1
    assert main.formal_training_ready is False
    assert endpoint.formal_training_ready is False
    assert main.inference.load_from_raw_predictions is False
    assert endpoint.inference.load_from_raw_predictions is False
    assert main.dataset == endpoint.dataset
    assert main.model.backbone == endpoint.model.backbone
    assert main.model.projection == endpoint.model.projection
    assert main.model.cache_size == endpoint.model.cache_size
    assert main.evaluation == endpoint.evaluation
    assert main.model.head.emission_policy == "pceh"
    assert endpoint.model.head.emission_policy == "endpoint_only"


def test_slurm_submitter_enforces_smoke_gate_before_pilot():
    source = (ROOT / "tools" / "remote" / "submit_pceh_n16r4.sh").read_text(encoding="utf-8")

    for token in (
        "sbatch",
        "smoke_pceh_stream.py",
        "audit_pceh_model.py",
        "SMOKE_RUN_DIR",
        "ALLOW_PILOT",
        "GPUS_PER_NODE",
        "GPUS_PER_NODE != 1",
        "packet_audit",
        "update_audit",
        "causal_replay",
        "torchrun",
        "PILOT_TIME",
    ):
        assert token in source
    assert "screen" not in source
    assert "load_from_raw_predictions=True" not in source


def test_slurm_submitter_keeps_generated_runs_outside_code_checkout():
    source = (ROOT / "tools" / "remote" / "submit_pceh_n16r4.sh").read_text(encoding="utf-8")

    assert "RUNS_ROOT" in source
    assert "/data/run01/sczc063/yuzibo/runs/pceh" in source
    assert 'RUN_DIR="$RUNS_ROOT/${JOB_NAME}_${STAMP}"' in source
    assert 'RUN_DIR="$BASE_DIR/slurm/' not in source
    assert '--cfg-options work_dir="${RUN_DIR}/work"' in source


def test_slurm_submitter_uses_n16r4_accepted_cpu_default():
    source = (ROOT / "tools" / "remote" / "submit_pceh_n16r4.sh").read_text(encoding="utf-8")

    assert "CPUS_PER_TASK=${CPUS_PER_TASK:-4}" in source
    assert "CPUS_PER_TASK=${CPUS_PER_TASK:-8}" not in source


def test_remote_check_helper_reads_slurm_and_gate_artifacts():
    source = (ROOT / "tools" / "remote" / "check_pceh_n16r4.sh").read_text(encoding="utf-8")

    assert "activate_n16r4_causaltad.sh" in source
    assert "squeue" in source
    assert "sacct" in source
    assert "train_step_report.json" in source
    assert "causal_replay_report.json" in source
    assert "isinstance(packet_audit, dict)" in source
    assert "passed" in source
