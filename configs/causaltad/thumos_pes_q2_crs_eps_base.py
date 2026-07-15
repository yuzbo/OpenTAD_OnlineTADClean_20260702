_base_ = ["./thumos_pes_q2_base.py"]

# CRS-EPS is a cost surrogate under audit, not the Full PETAL headline claim.
route_stage = "q2_crs_eps_hh_ipw_implementation_gate"

crs_eps_contract = dict(
    schema_version="full-petal-crs-eps-contract-v1",
    target_risk="video_uniform_then_decision_bin_uniform",
    primary_estimator="hansen_hurwitz_repeated_exposure_uncapped_ipw",
    union_ht_role="deduplicated_diagnostic_only",
    snipw_role="diagnostic_only",
    suffix_bins=8,
    context_bins=192,
    detach_interval=64,
    mixture=dict(
        uniform=0.40,
        start=0.15,
        end=0.20,
        ongoing=0.10,
        hardbg=0.15,
    ),
    draws_per_video=4,
    draw_count_status="PROVISIONAL_IMPLEMENTATION_PROBE_UNTIL_G0",
    independent_episode_state=True,
    one_optimizer_event_per_video_group=True,
    final_evaluation="complete_chronological_one_token_streaming",
)

experiment_contract = dict(
    training_protocol="crs_eps_hh_ipw_v1",
    training_target_risk="video_uniform_then_decision_bin_uniform",
    dynamic_replay_is_state_surrogate=True,
    video_start_replay_is_gold=True,
    exact_ipw_does_not_correct_state_bias=True,
    repeated_exposures_not_deduplicated=True,
    sampled_episode_runtime_state_shared=False,
    sampled_episode_gradient_update_shared=True,
)

dataset = dict(
    train=dict(
        type="CrsEpsFeatureDataset",
        sampling_seed=None,
        draws_per_video=crs_eps_contract["draws_per_video"],
        suffix_bins=crs_eps_contract["suffix_bins"],
        context_bins=crs_eps_contract["context_bins"],
        detach_interval=crs_eps_contract["detach_interval"],
        proposal_mixture=crs_eps_contract["mixture"],
    ),
    # Final quality is measured at the serving cadence. Validation may retain
    # larger packets for loss-only diagnostics, but the reported test stream is
    # strictly one observed feature token per call.
    test=dict(chunk_size=1),
)

profile_contract = dict(
    warmup_steps=8,
    measured_steps=32,
    step_unit="video_group_optimizer_event",
    world_size=1,
    required_workload_denominators=[
        "temporal_forward_tokens",
        "temporal_backward_tokens",
        "replay_tokens",
        "supervised_exposures",
        "unique_supervised_bins",
        "effective_sample_size",
        "visual_forward_frames",
        "visual_backward_frames",
        "data_wait_seconds",
        "control_unroll_seconds",
        "wall_seconds",
        "peak_memory_bytes",
        "gpu_hours",
    ],
    b1_total_gpu_hour_cap=2,
    b2_total_gpu_hour_cap=10,
    submit_via_slurm_only=True,
)

work_dir = "exps/thumos/pes_q2_crs_eps_base"
