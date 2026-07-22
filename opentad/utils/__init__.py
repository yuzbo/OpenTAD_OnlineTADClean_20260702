_EXPORTS = {
    "set_seed": ("misc", "set_seed"),
    "configure_strict_determinism": (
        "misc",
        "configure_strict_determinism",
    ),
    "update_workdir": ("misc", "update_workdir"),
    "create_folder": ("misc", "create_folder"),
    "save_config": ("misc", "save_config"),
    "AverageMeter": ("misc", "AverageMeter"),
    "setup_logger": ("logger", "setup_logger"),
    "ModelEma": ("ema", "ModelEma"),
    "save_checkpoint": ("checkpoint", "save_checkpoint"),
    "save_best_checkpoint": ("checkpoint", "save_best_checkpoint"),
    "AuditReport": ("causal_audit", "AuditReport"),
    "audit_future_perturbation": ("causal_audit", "audit_future_perturbation"),
    "audit_chunk_invariance": ("causal_audit", "audit_chunk_invariance"),
    "audit_packet_metadata": ("causal_audit", "audit_packet_metadata"),
    "audit_batch_isolation": ("causal_audit", "audit_batch_isolation"),
    "audit_recorded_trace_equivalence": ("causal_audit", "audit_recorded_trace_equivalence"),
    "audit_emission_ledger": ("causal_audit", "audit_emission_ledger"),
    "StreamPacketSpec": ("stream_packets", "StreamPacketSpec"),
    "build_packet_manifest": ("stream_packets", "build_packet_manifest"),
    "ChronologicalStreamBatchSampler": ("stream_packets", "ChronologicalStreamBatchSampler"),
    "OptimizerAuditError": ("optimizer_audit", "OptimizerAuditError"),
    "OptimizerAuditReport": ("optimizer_audit", "OptimizerAuditReport"),
    "audit_optimizer_coverage": ("optimizer_audit", "audit_optimizer_coverage"),
    "get_model_device": ("device", "get_model_device"),
    "move_data_to_device": ("device", "move_data_to_device"),
    "TrainingUpdateAuditError": ("training_audit", "TrainingUpdateAuditError"),
    "TrainingUpdateAuditReport": ("training_audit", "TrainingUpdateAuditReport"),
    "snapshot_trainable_parameters": ("training_audit", "snapshot_trainable_parameters"),
    "audit_training_update": ("training_audit", "audit_training_update"),
    "StreamSmokeError": ("stream_smoke", "StreamSmokeError"),
    "run_stream_smoke": ("stream_smoke", "run_stream_smoke"),
    "PrefixInstanceTarget": ("prefix_instance_schedule", "PrefixInstanceTarget"),
    "PrefixScheduleStep": ("prefix_instance_schedule", "PrefixScheduleStep"),
    "build_prefix_instance_schedule": (
        "prefix_instance_schedule",
        "build_prefix_instance_schedule",
    ),
    "audit_model_future_perturbation": (
        "model_causal_replay",
        "audit_model_future_perturbation",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = __import__(f"{__name__}.{module_name}", fromlist=[attr_name])
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
