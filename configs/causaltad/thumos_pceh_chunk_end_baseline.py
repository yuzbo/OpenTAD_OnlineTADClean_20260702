_base_ = "./thumos_pceh_ontad.py"

# Controlled high-latency baseline: the same PCEH lifecycle learner receives
# one 51.2-second packet and decides only at its end. This config exists to
# isolate decision cadence; it must not be called low latency.
route_stage = "controlled_chunk_end_pceh"
formal_training_ready = False

packet_size_frames = 1536
stream_protocol = dict(
    acquisition_policy="fixed_causal_stride8",
    packet_size_frames=packet_size_frames,
    decision_cadence_frames=packet_size_frames,
    decision_cadence_sec=51.2,
)

dataset = dict(
    train=dict(packet_size_frames=packet_size_frames, frame_policy="fixed_causal_stride8"),
    val=dict(packet_size_frames=packet_size_frames, frame_policy="fixed_causal_stride8"),
    test=dict(packet_size_frames=packet_size_frames, frame_policy="fixed_causal_stride8"),
)

work_dir = "exps/thumos/pceh_chunk_end_baseline"
