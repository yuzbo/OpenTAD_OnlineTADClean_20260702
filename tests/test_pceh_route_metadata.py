from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "causaltad" / "thumos_siglip2_adaptive_matr_ontad_final.py"
ADAPTER = ROOT / "opentad" / "models" / "backbones" / "online_siglip_adapter.py"


def test_chunk_end_baseline_uses_truthful_fixed_stride_metadata():
    cfg = runpy.run_path(str(CONFIG))

    assert cfg["route_stage"] == "chunk_end_baseline"
    assert cfg["formal_training_ready"] is False
    assert cfg["fixed_raw_frame_protocol"] == dict(
        frame_policy="fixed_causal_stride2",
        method_stage="chunk_end_baseline",
        head_name="MATRHead_memory0_chunk_end_baseline",
        decision_cadence="window_end",
    )
    assert cfg["dataset"]["train"]["frame_policy"] == "fixed_causal_stride2"
    assert cfg["dataset"]["val"]["frame_policy"] == "fixed_causal_stride2"
    assert cfg["dataset"]["test"]["frame_policy"] == "fixed_causal_stride2"


def test_siglip_metadata_is_derived_from_runtime_selector_policy():
    source = ADAPTER.read_text(encoding="utf-8")

    assert 'meta["frame_policy"] = "adaptive_selected_frames"' not in source
    assert "def selector_policy_name(" in source
    assert 'meta["frame_policy"] = selector_policy_name(self.frame_selector)' in source


def test_legacy_named_config_does_not_claim_adaptive_or_low_latency():
    source = CONFIG.read_text(encoding="utf-8").lower()

    assert "budgeted adaptive online tad" not in source
    assert "low-latency" not in source
    assert "paper-ready" not in source
