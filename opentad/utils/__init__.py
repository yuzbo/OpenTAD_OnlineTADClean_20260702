_EXPORTS = {
    "set_seed": ("misc", "set_seed"),
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
