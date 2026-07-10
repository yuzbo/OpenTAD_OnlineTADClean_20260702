_base_ = "./thumos_siglip2_motion_matr_ontad_p2.py"

# Legacy-named fixed-stride chunk-end baseline. The selector encodes every
# second causal frame and the detector checks emission only at the end of each
# non-overlapping window. Keep this config as a controlled baseline for PCEH;
# it is not the final strict-online method.

route_stage = "chunk_end_baseline"
formal_training_ready = False
raw_frame_stream_id = "thumos_siglip2_adaptive_final"

fixed_raw_frame_protocol = dict(
    frame_policy="fixed_causal_stride2",
    method_stage="chunk_end_baseline",
    head_name="MATRHead_memory0_chunk_end_baseline",
    decision_cadence="window_end",
)

dataset = dict(
    train=dict(stream_id=raw_frame_stream_id, frame_policy="fixed_causal_stride2"),
    val=dict(stream_id=raw_frame_stream_id, frame_policy="fixed_causal_stride2"),
    test=dict(stream_id=raw_frame_stream_id, frame_policy="fixed_causal_stride2"),
)

model = dict(
    backbone=dict(
        backbone=dict(
            frame_selector=dict(
                type="CausalFrameSelector",
                policy="causal_stride",
                keep_ratio=0.5,
                stride=2,
                max_gap=2,
                always_first=True,
            ),
            encode_policy="selected_only",
            assert_selected_only=True,
            return_token_times=True,
        ),
    ),
)

post_processing = dict(
    streaming=True,
    sliding_window=False,
    streaming_safe_emission=True,
    max_latency=0.0,
    max_latency_frames=192 * 8,
    pre_nms_thresh=0.001,
    pre_nms_topk=300,
    streaming_nms_iou=0.5,
    save_emission_ledger=True,
    save_latency_summary=True,
    emission_ledger_filename="adaptive_final_emission_ledger.json",
    latency_summary_filename="adaptive_final_latency_summary.json",
    online_map=dict(enabled=True),
)

evaluation = dict(
    type="OnlineMAP",
    online_map=dict(enabled=True),
    require_ledger=True,
    require_no_future=True,
    thread=4,
)

work_dir = "exps/thumos/siglip2_adaptive_matr_ontad_final"
