from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def test_final_route_declares_adaptive_selected_only_irregular_online_metric():
    Config = __import__("mmengine.config", fromlist=["Config"]).Config

    cfg = Config.fromfile(str(ROOT / "configs/causaltad/thumos_siglip2_adaptive_matr_ontad_final.py"))

    assert cfg.route_stage == "P3P4-final-target"
    assert cfg.formal_training_ready is False
    assert cfg.model.backbone.backbone.frame_selector.type == "CausalFrameSelector"
    assert cfg.model.backbone.backbone.encode_policy == "selected_only"
    assert cfg.model.backbone.backbone.assert_selected_only is True
    assert cfg.model.backbone.backbone.return_token_times is True
    assert cfg.model.rpn_head.online_censored_training is True
    assert cfg.post_processing.streaming_safe_emission is True
    assert cfg.post_processing.online_map.enabled is True
    assert cfg.evaluation.online_map.enabled is True


def test_selector_source_is_selected_only_before_heavy_encoder():
    selector_source = read("opentad/models/selectors/causal_frame_selector.py")
    encoder_source = read("opentad/models/backbones/online_siglip_adapter.py")
    init_source = read("opentad/models/selectors/__init__.py")

    assert "class CausalFrameSelector" in selector_source
    assert "def select(" in selector_source
    assert "selected_frames" in selector_source
    assert "selected_positions" in selector_source
    assert "causal_stride" in selector_source

    assert "frame_selector" in encoder_source
    assert "_select_frames_before_vision" in encoder_source
    assert "selected.frames.reshape" in encoder_source
    assert "irregular_selected_positions" in encoder_source
    assert "token_times_sec" in encoder_source
    assert '"CausalFrameSelector"' in init_source


def test_irregular_decode_and_online_map_modules_exist():
    irregular_source = read("opentad/models/utils/irregular_time_decode.py")
    post_utils_source = read("opentad/models/utils/post_processing/utils.py")
    online_map_source = read("opentad/evaluations/online_map.py")
    eval_init_source = read("opentad/evaluations/__init__.py")

    assert "def decode_irregular_segments_to_seconds" in irregular_source
    assert "token_times_sec" in irregular_source
    assert "decode_irregular_segments_to_seconds" in post_utils_source
    assert "def compute_online_detection_metrics" in online_map_source
    assert "latency_sec" in online_map_source
    assert "future_end_violations" in online_map_source
    assert "OnlineMAP" in eval_init_source


def test_streaming_protocol_tracks_source_frame_separately_from_segment_end():
    protocol_source = read("opentad/utils/online_protocol.py")
    detector_source = read("opentad/models/detectors/single_stage.py")
    matr_source = read("opentad/models/dense_heads/matr_head.py")

    assert "source_frame" in protocol_source
    assert "if cand.source_frame is not None" in protocol_source
    assert "source_frame" in detector_source
    assert "reset_stream_state" in detector_source
    assert "rpn_train_kwargs[\"metas\"] = metas" in detector_source
    assert "_build_irregular_points" in matr_source
    assert "irregular_selected_positions" in matr_source
    assert "proposal_axis" in matr_source
