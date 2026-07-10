from pathlib import Path

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "causaltad"


def _load(name):
    return Config.fromfile(CONFIG_DIR / name)


def test_main_pceh_config_wires_strict_streaming_route():
    cfg = _load("thumos_pceh_ontad.py")

    assert cfg.route_stage == "pceh_ontad_candidate"
    assert cfg.formal_training_ready is False
    assert cfg.stream_protocol.decision_cadence_frames == 8
    assert cfg.stream_protocol.primary_latency_budget_sec == 1.0
    assert cfg.stream_protocol.latency_definition == "emit_time_minus_matched_gt_end"
    assert cfg.dataset.train.type == "StreamingRawFrameDataset"
    assert cfg.dataset.train.packet_size_frames == 8
    assert cfg.dataset.train.frame_policy == "packet_recent_frame"
    assert cfg.solver.train.streaming is True
    assert cfg.solver.train.stream_batch_size == 1
    assert cfg.solver.train.num_workers == 0
    assert cfg.model.type == "PCEHOnlineDetector"
    assert cfg.model.head.type == "PrefixEventEmissionHead"
    assert cfg.optimizer.audit.fail_on_frozen is True
    assert cfg.evaluation.type == "OnlineAPBudgeted"
    assert cfg.evaluation.latency_budgets_sec == [0.5, 1.0, 2.0, 4.0]
    assert cfg.inference.load_from_raw_predictions is False


def test_controlled_baselines_change_only_declared_stream_schedule():
    chunk = _load("thumos_pceh_chunk_end_baseline.py")
    rolling = _load("thumos_pceh_rolling_fixed_stride2.py")

    assert chunk.route_stage == "controlled_chunk_end_pceh"
    assert chunk.dataset.train.packet_size_frames == 1536
    assert chunk.dataset.train.frame_policy == "fixed_causal_stride8"
    assert chunk.stream_protocol.decision_cadence_frames == 1536
    assert rolling.route_stage == "controlled_rolling_stride2_pceh"
    assert rolling.dataset.train.packet_size_frames == 8
    assert rolling.dataset.train.frame_policy == "fixed_causal_stride2"
    assert rolling.stream_protocol.decision_cadence_frames == 8


def test_streaming_train_entry_disables_shuffle_and_handles_batch_sampler_epoch():
    source = (ROOT / "tools" / "train.py").read_text(encoding="utf-8")

    assert "_streaming_shuffle" in source
    assert "_set_dataloader_epoch" in source
    assert "reset_online_states" in (
        ROOT / "opentad" / "cores" / "train_engine.py"
    ).read_text(encoding="utf-8")
