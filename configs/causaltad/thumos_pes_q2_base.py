# One-factor C1 bridge: persistent fixed binding versus supervision rematching.

route_stage = "q2_persistent_binding_one_factor"
formal_training_ready = False
visual_training_allowed = False
raw_video_finetuning = False
gpu_authorization = "BLOCKED_UNTIL_B0_AND_PROFILE"
pilot_seeds = [705, 706, 707]

annotation_path = "/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json"
class_map = "/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt"
manifest_root = "/data/run01/sczc063/yuzibo/thumos14/manifests/full_petal_q2"
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
    immutable_emissions=True,
    future_endpoint_prediction=False,
    offline_nms=False,
    raw_video_joint_training=False,
)

profile_contract = dict(
    warmup_steps=50,
    measured_steps=200,
    b1_total_gpu_hour_cap=2,
    b2_total_gpu_hour_cap=10,
    submit_via_slurm_only=True,
)

_dataset_common = dict(
    type="StreamingFeatureDataset",
    ann_file=annotation_path,
    class_map=class_map,
    data_path=feature_cache_path,
    cache_manifest=feature_cache_manifest,
    chunk_size=chunk_size,
    feature_stride=feature_stride,
    stream_id="thumos-pes-q2-siglip2-stride8-v1",
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
        endpoint_mode="binary",
        birth_threshold=0.5,
        alive_threshold=0.5,
        end_threshold=0.5,
        refractory_steps=2,
        max_endpoint_offset=feature_stride,
    ),
)

optimizer = dict(type="AdamW", lr=2e-4, weight_decay=0.05)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=1, max_epoch=12)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    streaming=True,
    streaming_safe_emission=True,
    sliding_window=False,
    save_emission_ledger=True,
    emission_ledger_filename="q2_emission_ledger.jsonl",
    save_latency_summary=True,
    latency_summary_filename="q2_latency_summary.json",
    save_dict=False,
)

solver = dict(
    train=dict(batch_size=1, stream_batch_size=1, streaming=True, num_workers=0),
    val=dict(batch_size=1, stream_batch_size=1, streaming=True, num_workers=0),
    test=dict(batch_size=1, stream_batch_size=1, streaming=True, num_workers=0),
    static_graph=False,
    clip_grad_norm=1.0,
    ema=False,
    amp=True,
    amp_dtype="bf16",
)

evaluation = dict(
    type="OnlineAPBudgeted",
    subset="validation",
    tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
    latency_budgets_sec=[0.5, 1.0, 2.0, 4.0],
    fps=fps,
    require_ledger=True,
    require_no_future=True,
    ground_truth_filename=annotation_path,
)

workflow = dict(
    logging_interval=50,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=1,
    val_start_epoch=0,
    end_epoch=12,
)

work_dir = "exps/thumos/pes_q2_base"
