_base_ = "./thumos_pceh_ontad.py"

# Controlled higher-compute rolling baseline. It preserves the 8-frame
# decision cadence but encodes every second frame in each packet.
route_stage = "controlled_rolling_stride2_pceh"
formal_training_ready = False

stream_protocol = dict(
    acquisition_policy="fixed_causal_stride2",
    packet_size_frames=8,
    decision_cadence_frames=8,
    decision_cadence_sec=8.0 / 30.0,
)

dataset = dict(
    train=dict(packet_size_frames=8, frame_policy="fixed_causal_stride2"),
    val=dict(packet_size_frames=8, frame_policy="fixed_causal_stride2"),
    test=dict(packet_size_frames=8, frame_policy="fixed_causal_stride2"),
)

work_dir = "exps/thumos/pceh_rolling_fixed_stride2"
