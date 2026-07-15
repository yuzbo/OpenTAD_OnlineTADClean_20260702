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
development_split_manifest = manifest_root + "/thumos_development_split.json"
fit_core_manifest = manifest_root + "/thumos_fit_core_160.txt"
calibration_manifest = manifest_root + "/thumos_calibration_40.txt"
reporting_manifest = manifest_root + "/thumos_reporting_locked_211.txt"
reporting_universe_manifest = manifest_root + "/thumos_reporting_universe_211.json"
reporting_comparison_manifest = manifest_root + "/thumos_reporting_211_vs_213.json"
fineaction_qualification_manifest = None
development_split_seed = 20260713

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
    step_unit="optimizer_event",
    world_size=1,
    b1_total_gpu_hour_cap=2,
    b2_total_gpu_hour_cap=10,
    submit_via_slurm_only=True,
)

launch_contract = dict(
    schema_version="full-petal-launch-contract-v3",
    required_reviewer_id="019f5abd-5104-79b3-882e-354ca796f2c1",
    required_review_scope=[
        "P0_evidence_chain",
        "training_lifecycle",
        "data_metrics",
        "optimizer_launch",
        "full_B0",
    ],
    allowed_cfg_overrides=["work_dir"],
    require_clean_checkout=True,
    trusted_scontrol_path="/usr/bin/scontrol",
    evidence_trust_model=dict(
        schema_version="full-petal-evidence-trust-model-v1",
        purpose="scientific_reproducibility",
        trusted_computing_base=[
            "launch_validator",
            "train_engine",
            "runtime_evidence_session",
            "in_process_attestation_key_material",
        ],
        guarantees=[
            "fail_closed_lifecycle_wiring",
            "provenance_binding",
            "post_publication_tamper_evidence",
        ],
        out_of_scope=[
            "arbitrary_code_execution_inside_tcb",
            "in_process_private_key_compromise",
        ],
        key_compromise_action="BLOCK_ROTATE_AND_RERUN",
    ),
    attestation_trust_roots=dict(
        b0=dict(
            key_id="full-petal-b0-20260713",
            public_key="rZMjZ/ST5X5cf+u7dSm1T8sStP/PkACL+cr8b/KLB3Y=",
        ),
        review=dict(
            key_id="019f5abd-5104-79b3-882e-354ca796f2c1",
            public_key="yjZXBpBY917paABqnDxL15y/eyKECmbhJDQuq4ce7Xk=",
        ),
        profile=dict(
            key_id="full-petal-profile-20260713",
            public_key="+3fu1yJFnAtE0U/wKdMw2X0sMKdfWcBqyUglXyAi6rE=",
        ),
        formal=dict(
            key_id="full-petal-formal-20260715",
            public_key="89P0h2h+msjeoXdrvWWLa89DB63KOf99COW31Q/397w=",
        ),
    ),
)

reporting_contract = dict(
    locked_population=reporting_manifest,
    expected_historical_count=211,
    canonical_expected_count=213,
    allow_during_training=False,
    disclosure="locked after prior project-level exposure; not untouched",
    fineaction_trust_roots=dict(
        license=dict(
            key_id="full-petal-fineaction-license-20260715",
            public_key="UNlvM0KrESOjzOCOC19bb14z1ZEL88+rULyyANnpyow=",
        ),
        execution=dict(
            key_id="full-petal-fineaction-execution-20260715",
            public_key="LAsJV8X230X8vSJd7nI23uaTuNVCfi0qwTb2/M5g0Ic=",
        ),
    ),
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
    split_manifest=development_split_manifest,
    split_seed=development_split_seed,
)

dataset = dict(
    train=dict(
        **_dataset_common,
        subset_name="training",
        allow_list=fit_core_manifest,
        split_role="fit_core",
    ),
    val=dict(
        **_dataset_common,
        subset_name="training",
        allow_list=calibration_manifest,
        split_role="calibration",
    ),
    test=dict(
        **_dataset_common,
        subset_name="training",
        allow_list=calibration_manifest,
        split_role="calibration",
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

optimizer = dict(
    type="AdamW",
    lr=2e-4,
    weight_decay=0.05,
    audit=dict(fail_on_frozen=True),
)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=1, max_epoch=12)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    streaming=True,
    streaming_safe_emission=True,
    sliding_window=False,
    save_emission_ledger=True,
    emission_ledger_filename="q2_emission_ledger.jsonl",
    emission_ledger_commitment_filename="q2_emission_ledger.commitment.json",
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
    subset="training",
    allowed_videos=calibration_manifest,
    tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
    latency_budgets_sec=[0.5, 1.0, 2.0, 4.0],
    fps=fps,
    identity_tiou_threshold=0.5,
    identity_latency_budget_sec=1.0,
    identity_metric_contract_sha256="48905225ab822605e925950348fe036bf8aab8750d60f55aa45abcb05912c069",
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
