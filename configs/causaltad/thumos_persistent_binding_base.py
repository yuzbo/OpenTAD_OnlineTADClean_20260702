# Matched feature-level FIXED versus REMATCH falsification route.

route_stage = "persistent_binding_feature_falsification"
formal_training_ready = False
visual_training_allowed = False
raw_video_finetuning = False
pilot_seeds = [705, 706, 707]

annotation_path = "/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json"
class_map = "/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt"
manifest_root = "/data/run01/sczc063/yuzibo/thumos14/manifests/persistent_binding"
fit_core_manifest = manifest_root + "/thumos_fit_core_160.txt"
calibration_manifest = manifest_root + "/thumos_calibration_40.txt"
reporting_manifest = manifest_root + "/thumos_reporting_locked_211.txt"

feature_cache_path = "/data/run01/sczc063/yuzibo/thumos14/features/pes_siglip2_stride8"
feature_cache_manifest = feature_cache_path + "/manifest.json"

fps = 30.0
feature_stride = 8
feature_dim = 768
chunk_size = 64
num_slots = 4
hidden_dim = 256
memory_size = 192

experiment_contract = dict(
    task="standard_fully_supervised_completion_triggered_ontad",
    input="fixed_cached_causal_features",
    changed_axis="post_birth_target_to_slot_loss_binding",
    shared_first_crossing_birth=True,
    canonical_lifecycle_shared=True,
    candidate_negative_semantics_shared=True,
    runtime_state_contains_gt=False,
    rematch_is_supervision_control_only=True,
    final_emissions=True,
    future_endpoint_prediction=False,
    offline_nms=False,
    raw_video_joint_training=False,
)

profile_contract = dict(
    seed=705,
    warmup_steps=50,
    measured_steps=200,
    one_seed_gpu_hour_cap=2,
    three_seed_gpu_hour_cap=10,
    submit_via_slurm_only=True,
)

optimization_contract = dict(
    schedule_revision="short_warmup_v1",
    changed_axis="shared_linear_warmup_fraction_only",
    previous_warmup_epoch=1.0,
    warmup_epoch=0.1,
    max_epoch=12,
    peak_learning_rate=2e-4,
    update_count_unchanged=True,
    fixed_rematch_shared=True,
    motivation="one_epoch_screen_reached_peak_lr_only_at_final_update",
)

supervision_balance_contract = dict(
    source_split="fit_core",
    tokens=123940,
    birth_positive_targets=2523,
    birth_supervised_targets=458670,
    alive_positive_targets=39613,
    alive_supervised_targets=495760,
    end_positive_targets=2523,
    end_supervised_targets=39613,
    birth_positive_rate=0.0055006867682647655,
    alive_positive_rate=0.07990358237857027,
    end_positive_rate=0.06369121248075127,
    weighting_rule="sqrt_negative_to_positive_ratio",
    start_offset_unit="feature_tokens",
    max_start_offset_tokens=1.0,
)

census_contract = dict(
    expected_split_counts=dict(
        fit_core=160,
        calibration=40,
        reporting_locked=211,
    ),
    max_gt_entry_free_deficits=0,
    max_oracle_capacity_overflow_steps=0,
    max_uncovered_birth_instances=0,
    max_uncovered_endpoint_instances=0,
    max_clipped_start_supervision_targets=0,
)

_dataset_common = dict(
    type="StreamingFeatureDataset",
    ann_file=annotation_path,
    class_map=class_map,
    data_path=feature_cache_path,
    cache_manifest=feature_cache_manifest,
    chunk_size=chunk_size,
    feature_stride=feature_stride,
    stream_id="thumos-persistent-binding-siglip2-stride8-v1",
    strict_causal_control=True,
)

dataset = dict(
    train=dict(
        **_dataset_common,
        subset_name="training",
        allow_list=fit_core_manifest,
    ),
    val=dict(
        **_dataset_common,
        subset_name="training",
        allow_list=calibration_manifest,
    ),
    test=dict(
        **_dataset_common,
        subset_name="validation",
        allow_list=reporting_manifest,
        test_mode=True,
    ),
)

model = dict(
    type="PersistentTrajectoryOnlineDetector",
    trajectory_binding_mode="UNSET_BY_VARIANT",
    birth_assignment_mode="first_crossing_shared",
    canonical_supervision_lifecycle=True,
    detach_stream_state=True,
    birth_loss_weight=1.0,
    alive_loss_weight=0.5,
    class_loss_weight=1.0,
    start_loss_weight=1.0,
    end_loss_weight=1.0,
    birth_positive_weight=13.446021031128877,
    alive_positive_weight=3.393388193562092,
    end_positive_weight=3.8341561094639838,
    prior_bias_mode="weighted_bce_stationary",
    fail_on_supervision_exhaustion=True,
    head=dict(
        type="PersistentEventSetHead",
        in_channels=feature_dim,
        hidden_dim=hidden_dim,
        num_classes=20,
        num_slots=num_slots,
        memory_size=memory_size,
        num_heads=8,
        dropout=0.1,
        query_mode="persistent",
        start_mode="scalar",
        max_start_offset=1.0,
        endpoint_mode="binary",
        birth_prior_probability=0.0055006867682647655,
        alive_prior_probability=0.07990358237857027,
        end_prior_probability=0.06369121248075127,
        birth_threshold=0.5,
        alive_threshold=0.5,
        end_threshold=0.5,
        refractory_steps=0,
        lifecycle_mode="candidate_recycle",
        candidate_confirmation_steps=1,
        max_births_per_step=2,
    ),
)

optimizer = dict(type="AdamW", lr=2e-4, weight_decay=0.05)
scheduler = dict(
    type="LinearWarmupCosineAnnealingLR",
    warmup_epoch=0.1,
    max_epoch=12,
)

inference = dict(
    load_from_raw_predictions=False,
    save_raw_prediction=False,
    require_explicit_checkpoint=True,
)
post_processing = dict(
    streaming=True,
    streaming_safe_emission=True,
    sliding_window=False,
    save_emission_ledger=True,
    emission_ledger_filename="persistent_binding_emissions.json",
    save_latency_summary=True,
    latency_summary_filename="persistent_binding_latency.json",
    save_dict=False,
)

solver = dict(
    train=dict(batch_size=1, stream_batch_size=1, streaming=True, num_workers=0),
    val=dict(batch_size=1, stream_batch_size=1, streaming=True, num_workers=0),
    test=dict(batch_size=1, stream_batch_size=1, streaming=True, num_workers=0),
    static_graph=False,
    clip_grad_norm=1.0,
    ema=False,
    amp=False,
    strict_determinism=True,
)

calibration_evaluation = dict(
    type="OnlineAPBudgeted",
    subset="training",
    allowed_videos=calibration_manifest,
    tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
    latency_budgets_sec=[0.5, 1.0, 2.0, 4.0],
    fps=fps,
    require_ledger=True,
    require_no_future=True,
    ground_truth_filename=annotation_path,
)

reporting_evaluation = dict(
    type="OnlineAPBudgeted",
    subset="validation",
    allowed_videos=reporting_manifest,
    tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
    latency_budgets_sec=[0.5, 1.0, 2.0, 4.0],
    fps=fps,
    require_ledger=True,
    require_no_future=True,
    ground_truth_filename=annotation_path,
)

evaluation = reporting_evaluation

calibration_contract = dict(
    selection_metric="average_mOnlineAP",
    selection_direction="maximize",
    tie_breaking=("lower_epoch", "checkpoint_sha256"),
    uses_reporting_split=False,
)

reporting_contract = dict(
    one_shot_lock_required=True,
    explicit_checkpoint_required=True,
    calibration_receipt_required=True,
    reporting_split_access_during_fit=False,
)

workflow = dict(
    fit_only=True,
    logging_interval=50,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=0,
    end_epoch=12,
    fail_on_nonfinite=True,
)

work_dir = "exps/thumos/persistent_binding_base"
