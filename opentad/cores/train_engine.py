import copy
from collections.abc import Mapping
import time

import torch
import tqdm
from opentad.utils.misc import AverageMeter, reduce_loss
from opentad.utils.device import get_model_device, move_data_to_device
from opentad.utils.crs_eps_sampling import (
    canonical_json_sha256,
    episode_payload_sha256,
    epoch_manifest_runtime_control,
    validate_epoch_manifest,
)


def resolve_amp_dtype(enabled, amp_dtype="fp16"):
    if not bool(enabled):
        return None
    normalized = str(amp_dtype).strip().lower()
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError("amp_dtype must be one of fp16, float16, bf16, or bfloat16")


def _unwrap_model(model):
    return getattr(model, "module", model)


def _find_first_nonfinite_grad(model):
    for name, param in model.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            return name
    return None


def _summarize_grad_health(model, topk=8):
    nonfinite = []
    large_finite = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        finite = torch.isfinite(grad)
        if not finite.all():
            nonfinite.append(
                dict(
                    name=name,
                    shape=tuple(grad.shape),
                    finite_count=int(finite.sum().item()),
                    nonfinite_count=int(grad.numel() - finite.sum().item()),
                )
            )
            continue
        absmax = float(grad.abs().max().item()) if grad.numel() > 0 else 0.0
        large_finite.append((absmax, name, tuple(grad.shape)))

    large_finite.sort(reverse=True, key=lambda item: item[0])
    return {
        "grad_nonfinite_param_count": len(nonfinite),
        "grad_nonfinite_params": nonfinite[:topk],
        "grad_top_absmax_params": [
            dict(name=name, absmax=absmax, shape=shape) for absmax, name, shape in large_finite[:topk]
        ],
    }


def _format_debug_report(report):
    ordered_items = []
    for key in sorted(report.keys()):
        ordered_items.append(f"{key}={report[key]}")
    return " | ".join(ordered_items)


def _collect_runtime_debug(model, data_dict, bad_param_name):
    target = _unwrap_model(model)
    if hasattr(target, "collect_runtime_debug"):
        try:
            return target.collect_runtime_debug(data_dict, bad_param_name)
        except Exception as exc:
            return {"debug_collection_error": repr(exc), "bad_param_name": bad_param_name}
    return None


def _grad_clip_parameters(model):
    target = _unwrap_model(model)
    if hasattr(target, "grad_clip_parameters"):
        return target.grad_clip_parameters()
    return model.parameters()


def _transaction_target(model):
    target = _unwrap_model(model)
    required = (
        "has_pending_online_update",
        "snapshot_online_update",
        "restore_online_update",
        "commit_online_update",
        "rollback_online_update",
    )
    return target if all(callable(getattr(target, name, None)) for name in required) else None


def _transaction_control(data_dict):
    controls = data_dict.get("stream_control")
    if not isinstance(controls, (list, tuple)) or len(controls) != 1:
        raise RuntimeError("transactional online training requires one stream_control lane")
    control = controls[0]
    if not isinstance(control, Mapping):
        raise RuntimeError("transactional stream_control must be a mapping")
    for field in ("is_video_start", "is_video_end"):
        if field not in control or not isinstance(control[field], bool):
            raise RuntimeError(f"transactional stream_control requires boolean {field}")
    return control


def _optimizer_weight(losses):
    value = losses.get("_optimizer_weight")
    if value is None:
        return 1.0
    if not torch.is_tensor(value) or value.numel() != 1:
        raise RuntimeError("_optimizer_weight must be a scalar tensor")
    weight = float(value.detach().item())
    if not torch.isfinite(value.detach()).item() or weight <= 0:
        raise RuntimeError("_optimizer_weight must be positive and finite")
    return weight


def _optimizer_denominator(losses, default):
    value = losses.get("_optimizer_denominator")
    if value is None:
        return float(default)
    if not torch.is_tensor(value) or value.numel() != 1:
        raise RuntimeError("_optimizer_denominator must be a scalar tensor")
    denominator = float(value.detach().item())
    if not torch.isfinite(value.detach()).item() or denominator <= 0:
        raise RuntimeError("_optimizer_denominator must be positive and finite")
    return denominator


def _crs_eps_control(data_dict):
    controls = data_dict.get("crs_eps")
    if controls is None:
        return None
    if not isinstance(controls, (list, tuple)) or len(controls) != 1:
        raise RuntimeError("CRS-EPS training requires one episode-control lane")
    control = controls[0]
    if not isinstance(control, Mapping):
        raise RuntimeError("CRS-EPS episode control must be a mapping")
    required = {
        "video_id",
        "video_group_index",
        "video_group_size",
        "draw_index",
        "is_video_group_start",
        "is_video_group_end",
        "episode_id",
        "episode_payload_sha256",
        "episode_manifest_sha256",
        "episode_sequence_sha256",
        "replay_range",
        "supervised_range",
        "gradient_ranges",
        "video_covered_unique_bins",
        "video_effective_sample_size",
        "video_ipw_weight_sum",
        "video_ipw_weight_squared_sum",
    }
    if not required.issubset(control):
        raise RuntimeError("CRS-EPS episode control lacks video-group fields")
    return control


def _buffer_snapshot(model):
    return {
        name: value.detach().clone()
        for name, value in model.named_buffers()
    }


def _assert_buffer_snapshot(model, snapshot):
    current = dict(model.named_buffers())
    if set(current) != set(snapshot):
        raise RuntimeError("CRS-EPS model buffer set changed within a video group")
    changed = [
        name
        for name, value in current.items()
        if value.shape != snapshot[name].shape
        or value.dtype != snapshot[name].dtype
        or not torch.equal(value.detach(), snapshot[name].to(value.device))
    ]
    if changed:
        raise RuntimeError(
            "CRS-EPS mutable model buffers leaked across independent draws: "
            + ", ".join(changed[:8])
        )


def _restore_buffer_snapshot(model, snapshot):
    current = dict(model.named_buffers())
    if set(current) != set(snapshot):
        raise RuntimeError("CRS-EPS model buffer set changed and cannot be restored")
    with torch.no_grad():
        for name, value in current.items():
            source = snapshot[name]
            if value.shape != source.shape or value.dtype != source.dtype:
                raise RuntimeError(
                    f"CRS-EPS model buffer {name} changed shape or dtype and cannot be restored"
                )
            value.copy_(source.to(device=value.device))


def _new_workload():
    return {
        "optimizer_events": 1,
        "episode_draws": 0,
        "temporal_forward_tokens": 0,
        "temporal_backward_tokens": 0,
        "replay_tokens": 0,
        "supervised_exposures": 0,
        "unique_supervised_bins": 0,
        "ipw_weight_sum": 0.0,
        "ipw_weight_squared_sum": 0.0,
        "visual_forward_frames": 0,
        "visual_backward_frames": 0,
        "data_wait_seconds": 0.0,
        "control_unroll_seconds": 0.0,
        "wall_seconds": 0.0,
    }


def _normalize_accumulated_gradients(model, denominator):
    if denominator <= 0:
        raise RuntimeError("episode gradient denominator must be positive")
    for param in model.parameters():
        if param.grad is not None:
            param.grad.div_(denominator)


def _rollback_online_transaction(transaction):
    if transaction is not None and transaction.has_pending_online_update():
        transaction.rollback_online_update()


def _rollback_crs_group(model, transaction, optimizer, buffer_snapshot):
    """Restore every state that a pre-boundary CRS-EPS draw may mutate."""

    restore_error = None
    try:
        if buffer_snapshot is not None:
            _restore_buffer_snapshot(model, buffer_snapshot)
    except Exception as exc:
        restore_error = exc
    try:
        _rollback_online_transaction(transaction)
    except Exception as exc:
        if restore_error is None:
            restore_error = exc
    finally:
        optimizer.zero_grad(set_to_none=True)
    if restore_error is not None:
        raise RuntimeError("strict CRS-EPS group rollback failed") from restore_error


def _snapshot_component(component, label):
    if component is None:
        return None
    state_dict = getattr(component, "state_dict", None)
    load_state_dict = getattr(component, "load_state_dict", None)
    if not callable(state_dict) or not callable(load_state_dict):
        raise RuntimeError(
            f"strict online transaction requires state_dict/load_state_dict for {label}"
        )
    try:
        return copy.deepcopy(state_dict())
    except Exception as exc:
        raise RuntimeError(f"failed to snapshot {label} state") from exc


def _capture_training_mutation_snapshot(
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    model_ema,
    optimizer_event_recorder,
    visual_parameter_event_recorder,
    transaction,
):
    """Capture every persistent state mutated at an optimizer boundary."""

    snapshot = {
        "model": _snapshot_component(model, "model"),
        "optimizer": _snapshot_component(optimizer, "optimizer"),
        "scheduler": _snapshot_component(scheduler, "scheduler"),
        "scaler": _snapshot_component(scaler, "scaler"),
        "model_ema": _snapshot_component(model_ema, "EMA"),
        "optimizer_event_recorder": _snapshot_component(
            optimizer_event_recorder, "optimizer event recorder"
        ),
        "visual_parameter_event_recorder": _snapshot_component(
            visual_parameter_event_recorder, "visual parameter event recorder"
        ),
    }
    snapshot["online_update"] = (
        transaction.snapshot_online_update() if transaction is not None else None
    )
    return snapshot


def _restore_training_mutation_snapshot(
    snapshot,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    model_ema,
    optimizer_event_recorder,
    visual_parameter_event_recorder,
    transaction,
):
    components = (
        (model, "model"),
        (optimizer, "optimizer"),
        (scheduler, "scheduler"),
        (scaler, "scaler"),
        (model_ema, "EMA"),
        (optimizer_event_recorder, "optimizer event recorder"),
        (visual_parameter_event_recorder, "visual parameter event recorder"),
    )
    state_keys = (
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "model_ema",
        "optimizer_event_recorder",
        "visual_parameter_event_recorder",
    )
    for state_key, (component, label) in zip(state_keys, components):
        state = snapshot[state_key]
        if component is None:
            if state is not None:
                raise RuntimeError(f"rollback snapshot unexpectedly contains {label} state")
            continue
        try:
            component.load_state_dict(copy.deepcopy(state))
        except Exception as exc:
            raise RuntimeError(f"failed to restore {label} state") from exc
    if transaction is not None:
        transaction.restore_online_update(snapshot["online_update"])


def _rollback_failed_training_mutation(
    snapshot,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    model_ema,
    optimizer_event_recorder,
    visual_parameter_event_recorder,
    transaction,
):
    restore_error = None
    try:
        _restore_training_mutation_snapshot(
            snapshot,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            model_ema=model_ema,
            optimizer_event_recorder=optimizer_event_recorder,
            visual_parameter_event_recorder=visual_parameter_event_recorder,
            transaction=transaction,
        )
    except Exception as exc:
        restore_error = exc
    finally:
        _rollback_online_transaction(transaction)
        optimizer.zero_grad(set_to_none=True)
    if restore_error is not None:
        raise RuntimeError("strict online training rollback failed") from restore_error


def _input_token_count(data_dict):
    masks = data_dict.get("masks")
    if torch.is_tensor(masks):
        count = int(masks.detach().to(dtype=torch.int64).sum().item())
    else:
        inputs = data_dict.get("inputs")
        if not torch.is_tensor(inputs) or inputs.ndim == 0:
            raise RuntimeError("formal optimizer evidence cannot determine input tokens")
        count = int(inputs.shape[0] * inputs.shape[-1])
    if count <= 0:
        raise RuntimeError("formal optimizer evidence requires positive input tokens")
    return count


def _episode_identity(control, curr_epoch, iter_idx, crs_eps=None):
    if isinstance(crs_eps, Mapping):
        video_id = crs_eps.get("video_id")
        group_index = crs_eps.get("video_group_index")
        manifest_hash = crs_eps.get("episode_manifest_sha256")
        group_size = crs_eps.get("video_group_size")
        sequence_hash = crs_eps.get("episode_sequence_sha256")
        if isinstance(video_id, str) and video_id.strip():
            return (
                f"epoch={curr_epoch}|video={video_id}|group={group_index}"
                f"|M={group_size}|sequence={sequence_hash}|manifest={manifest_hash}"
            )
    if isinstance(control, Mapping):
        video_id = control.get("video_id")
        if isinstance(video_id, str) and video_id.strip():
            return video_id
    return f"epoch-{curr_epoch}-iter-{iter_idx}"


def train_one_epoch(
    train_loader,
    model,
    optimizer,
    scheduler,
    curr_epoch,
    logger,
    model_ema=None,
    clip_grad_l2norm=-1,
    logging_interval=200,
    runtime_debug_interval=-1,
    scaler=None,
    amp_dtype=None,
    fixed_step_profiler=None,
    optimizer_event_recorder=None,
    visual_parameter_event_recorder=None,
    crs_eps_manifest=None,
):
    """Training the model for one epoch"""

    logger.info("[Train]: Epoch {:d} started".format(curr_epoch))
    losses_tracker = {}
    num_iters = len(train_loader)
    if amp_dtype is None and scaler is not None:
        amp_dtype = torch.float16
    use_amp = amp_dtype is not None

    target = _unwrap_model(model)
    if hasattr(target, "reset_online_states"):
        target.reset_online_states()
    if hasattr(target, "set_train_epoch"):
        target.set_train_epoch(curr_epoch)

    model.train()
    model_device = get_model_device(model)
    transaction = _transaction_target(model)
    episode_weight = 0.0
    episode_loss_records = []
    episode_input_tokens = 0
    skip_until_boundary = False
    optimizer_events = 0
    successful_optimizer_events = 0
    skipped_optimizer_events = 0
    optimizer.zero_grad(set_to_none=True)
    active_crs_group = None
    crs_buffer_snapshot = None
    crs_expected_draw_index = None
    crs_episode_payload_sha256s = None
    crs_expected_sequence_sha256 = None
    crs_expected_group_index = 0
    trusted_crs_manifest = None
    if crs_eps_manifest is not None:
        trusted_crs_manifest = copy.deepcopy(crs_eps_manifest)
        validate_epoch_manifest(trusted_crs_manifest)
        if int(trusted_crs_manifest["epoch"]) != int(curr_epoch):
            raise RuntimeError("CRS-EPS manifest epoch differs from the training epoch")
    group_workload = None
    group_started_at = None
    if fixed_step_profiler is not None:
        fixed_step_profiler.start()
    if visual_parameter_event_recorder is not None and optimizer_event_recorder is None:
        raise RuntimeError(
            "visual parameter evidence requires the authenticated optimizer event recorder"
        )
    if optimizer_event_recorder is not None and transaction is None:
        raise RuntimeError(
            "authenticated optimizer evidence requires transactional online state"
        )

    loader_iterator = iter(train_loader)
    for iter_idx in range(num_iters):
        try:
            data_wait_started_at = time.perf_counter()
            raw_data_dict = next(loader_iterator)
            data_wait_seconds = time.perf_counter() - data_wait_started_at
            control = (
                _transaction_control(raw_data_dict)
                if transaction is not None
                else None
            )
            crs_control = _crs_eps_control(raw_data_dict)
        except Exception:
            if active_crs_group is not None:
                _rollback_crs_group(
                    model, transaction, optimizer, crs_buffer_snapshot
                )
            raise
        if crs_control is not None:
            try:
                if trusted_crs_manifest is None:
                    raise RuntimeError(
                        "CRS-EPS batches require a trusted immutable epoch manifest"
                    )
                group_size = int(crs_control["video_group_size"])
                draw_index = int(crs_control["draw_index"])
                starts_group = bool(crs_control["is_video_group_start"])
                boundary = bool(crs_control["is_video_group_end"])
                episode_id = crs_control["episode_id"]
                payload_sha256 = crs_control["episode_payload_sha256"]
                manifest_sha256 = crs_control["episode_manifest_sha256"]
                sequence_sha256 = crs_control["episode_sequence_sha256"]
                if group_size <= 0 or not 0 <= draw_index < group_size:
                    raise RuntimeError("CRS-EPS video-group geometry is invalid")
                if starts_group != (draw_index == 0):
                    raise RuntimeError("CRS-EPS group-start marker is inconsistent")
                if boundary != (draw_index + 1 == group_size):
                    raise RuntimeError("CRS-EPS group-end marker is inconsistent")
                if not isinstance(episode_id, str) or not episode_id.strip():
                    raise RuntimeError("CRS-EPS episode ID is invalid")
                for value, label in (
                    (manifest_sha256, "manifest"),
                    (payload_sha256, "episode payload"),
                    (sequence_sha256, "episode sequence"),
                ):
                    if (
                        not isinstance(value, str)
                        or len(value) != 64
                        or any(character not in "0123456789abcdef" for character in value)
                    ):
                        raise RuntimeError(f"CRS-EPS {label} hash is malformed")
                group_identity = (
                    str(crs_control["video_id"]),
                    int(crs_control["video_group_index"]),
                    group_size,
                    manifest_sha256,
                    sequence_sha256,
                )
                if starts_group:
                    if active_crs_group is not None:
                        raise RuntimeError("CRS-EPS video groups overlap")
                    if int(crs_control["video_group_index"]) != crs_expected_group_index:
                        raise RuntimeError(
                            "CRS-EPS video-group order differs from immutable manifest"
                        )
                    active_crs_group = group_identity
                    crs_buffer_snapshot = _buffer_snapshot(model)
                    crs_expected_draw_index = 0
                    crs_episode_payload_sha256s = []
                    crs_expected_sequence_sha256 = sequence_sha256
                elif active_crs_group != group_identity:
                    raise RuntimeError("CRS-EPS draw escaped its active video group")
                if draw_index != crs_expected_draw_index:
                    raise RuntimeError(
                        "CRS-EPS draw membership/order differs from immutable manifest"
                    )
                expected_control = epoch_manifest_runtime_control(
                    trusted_crs_manifest,
                    crs_expected_group_index,
                    crs_expected_draw_index,
                    validate=False,
                )
                runtime_payload_sha256 = episode_payload_sha256(crs_control)
                if (
                    runtime_payload_sha256 != payload_sha256
                    or payload_sha256 != expected_control["episode_payload_sha256"]
                ):
                    raise RuntimeError(
                        "CRS-EPS draw payload differs from immutable manifest"
                    )
                runtime_binding = {
                    field: crs_control[field]
                    for field in (
                        "video_id",
                        "video_group_index",
                        "video_group_size",
                        "is_video_group_start",
                        "is_video_group_end",
                        "episode_manifest_sha256",
                        "episode_sequence_sha256",
                        "video_covered_unique_bins",
                        "video_effective_sample_size",
                        "video_ipw_weight_sum",
                        "video_ipw_weight_squared_sum",
                    )
                }
                expected_binding = {
                    field: expected_control[field] for field in runtime_binding
                }
                if canonical_json_sha256(runtime_binding) != canonical_json_sha256(
                    expected_binding
                ):
                    raise RuntimeError(
                        "CRS-EPS runtime binding differs from immutable manifest"
                    )
                crs_episode_payload_sha256s.append(payload_sha256)
                crs_expected_draw_index += 1
                if boundary:
                    if crs_expected_draw_index != group_size:
                        raise RuntimeError("CRS-EPS group ended before all M draws")
                    if (
                        canonical_json_sha256(crs_episode_payload_sha256s)
                        != crs_expected_sequence_sha256
                    ):
                        raise RuntimeError(
                            "CRS-EPS episode sequence differs from immutable manifest"
                        )
            except Exception:
                if active_crs_group is not None:
                    _rollback_crs_group(
                        model, transaction, optimizer, crs_buffer_snapshot
                    )
                raise
        else:
            if active_crs_group is not None:
                _rollback_crs_group(model, transaction, optimizer, crs_buffer_snapshot)
                raise RuntimeError("non-CRS batch interrupted a CRS-EPS video group")
            boundary = True if control is None else bool(
                control["is_video_end"] or control.get("reset_stream", False)
            )
            starts_group = control is None or bool(control["is_video_start"])
        if starts_group:
            if group_workload is not None:
                if crs_control is not None:
                    _rollback_crs_group(
                        model, transaction, optimizer, crs_buffer_snapshot
                    )
                raise RuntimeError("optimizer workload groups overlap")
            group_workload = _new_workload()
            group_started_at = data_wait_started_at
            if crs_control is not None:
                group_workload["unique_supervised_bins"] = int(
                    crs_control.get("video_covered_unique_bins", 0)
                )
                group_workload["ipw_weight_sum"] = float(
                    crs_control.get("video_ipw_weight_sum", 0.0)
                )
                group_workload["ipw_weight_squared_sum"] = float(
                    crs_control.get("video_ipw_weight_squared_sum", 0.0)
                )
        if group_workload is not None:
            group_workload["data_wait_seconds"] += data_wait_seconds
        if skip_until_boundary:
            if crs_control is None and control["is_video_start"]:
                raise RuntimeError("a new stream started before the failed episode boundary")
            if crs_control is not None and bool(crs_control["is_video_group_start"]):
                raise RuntimeError("a new CRS-EPS group started before the failed boundary")
            if boundary:
                skip_until_boundary = False
                if crs_control is not None:
                    crs_expected_group_index += 1
                active_crs_group = None
                crs_buffer_snapshot = None
                crs_expected_draw_index = None
                crs_episode_payload_sha256s = None
                crs_expected_sequence_sha256 = None
                group_workload = None
                group_started_at = None
            continue

        try:
            input_tokens = _input_token_count(raw_data_dict)
            if crs_control is not None:
                replay_start, replay_end = (
                    int(value) for value in crs_control["replay_range"]
                )
                supervised_start, supervised_end = (
                    int(value) for value in crs_control["supervised_range"]
                )
                backward_tokens = sum(
                    int(end) - int(start)
                    for start, end in crs_control["gradient_ranges"]
                )
                if replay_end - replay_start != input_tokens:
                    raise RuntimeError(
                        "CRS-EPS workload token count differs from replay range"
                    )
                group_workload["episode_draws"] += 1
                group_workload["temporal_forward_tokens"] += input_tokens
                group_workload["temporal_backward_tokens"] += backward_tokens
                group_workload["replay_tokens"] += input_tokens
                group_workload["supervised_exposures"] += (
                    supervised_end - supervised_start
                )
            else:
                group_workload["temporal_forward_tokens"] += input_tokens
                group_workload["temporal_backward_tokens"] += input_tokens
                group_workload["supervised_exposures"] += input_tokens
                group_workload["unique_supervised_bins"] += input_tokens
            data_dict = move_data_to_device(raw_data_dict, model_device)
        except Exception:
            if crs_control is not None:
                _rollback_crs_group(model, transaction, optimizer, crs_buffer_snapshot)
            raise
        if optimizer_event_recorder is not None:
            episode_input_tokens += input_tokens
        curr_backbone_lr = None
        if hasattr(target, "backbone") and not getattr(
            target.backbone, "freeze_backbone", True
        ):
            curr_backbone_lr = scheduler.get_last_lr()[0]
        curr_det_lr = scheduler.get_last_lr()[-1]

        try:
            with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
                losses = model(**data_dict, return_loss=True)
        except Exception:
            if crs_control is not None:
                _rollback_crs_group(model, transaction, optimizer, crs_buffer_snapshot)
            else:
                _rollback_online_transaction(transaction)
                optimizer.zero_grad(set_to_none=True)
            raise
        if crs_control is not None:
            try:
                _assert_buffer_snapshot(model, crs_buffer_snapshot)
            except Exception:
                _rollback_crs_group(model, transaction, optimizer, crs_buffer_snapshot)
                raise
            episode_audit = getattr(target, "last_episode_audit", {})
            if not isinstance(episode_audit, Mapping):
                _rollback_crs_group(model, transaction, optimizer, crs_buffer_snapshot)
                raise RuntimeError("CRS-EPS detector did not publish an episode audit")
            slot_exhaustion = int(episode_audit.get("slot_exhaustion", 0))
            if slot_exhaustion:
                _rollback_crs_group(model, transaction, optimizer, crs_buffer_snapshot)
                raise RuntimeError(
                    "CRS-EPS scientific failure: slot exhaustion invalidates the "
                    f"training trajectory (count={slot_exhaustion})"
                )
            group_workload["control_unroll_seconds"] += float(
                episode_audit.get("control_unroll_seconds", 0.0)
            )

        cost = losses.get("cost")
        if not torch.is_tensor(cost) or cost.numel() != 1:
            if crs_control is not None:
                _rollback_crs_group(model, transaction, optimizer, crs_buffer_snapshot)
            else:
                _rollback_online_transaction(transaction)
                optimizer.zero_grad(set_to_none=True)
            raise RuntimeError("model losses must contain one scalar cost tensor")
        if not torch.isfinite(cost.detach()).item():
            logger.error(
                "[Train]: non-finite cost at epoch=%d iter=%d; rollback episode",
                curr_epoch,
                iter_idx,
            )
            if crs_control is not None:
                _rollback_crs_group(model, transaction, optimizer, crs_buffer_snapshot)
            else:
                _rollback_online_transaction(transaction)
                optimizer.zero_grad(set_to_none=True)
            episode_weight = 0.0
            episode_loss_records.clear()
            optimizer_events += 1
            skipped_optimizer_events += 1
            if fixed_step_profiler is not None:
                fixed_step_profiler.record_skipped_optimizer_event()
            if optimizer_event_recorder is not None:
                raise RuntimeError(
                    "authenticated training cannot continue after a skipped "
                    "optimizer boundary"
                )
            episode_input_tokens = 0
            skip_until_boundary = not boundary
            if boundary:
                if crs_control is not None:
                    crs_expected_group_index += 1
                active_crs_group = None
                crs_buffer_snapshot = None
                crs_expected_draw_index = None
                crs_episode_payload_sha256s = None
                crs_expected_sequence_sha256 = None
                group_workload = None
                group_started_at = None
            continue

        try:
            weight = _optimizer_weight(losses)
            denominator = _optimizer_denominator(losses, weight)
            weighted_cost = cost * weight
            if scaler is not None:
                scaler.scale(weighted_cost).backward()
            else:
                weighted_cost.backward()
        except Exception:
            if crs_control is not None:
                _rollback_crs_group(model, transaction, optimizer, crs_buffer_snapshot)
            else:
                _rollback_online_transaction(transaction)
                optimizer.zero_grad(set_to_none=True)
            raise
        episode_weight += denominator
        episode_loss_records.append(
            {
                key: value.detach()
                for key, value in losses.items()
                if not key.startswith("_")
            }
        )
        if not boundary:
            continue

        optimizer_events += 1
        if transaction is not None and not transaction.has_pending_online_update():
            if crs_control is not None:
                _rollback_crs_group(model, transaction, optimizer, crs_buffer_snapshot)
            else:
                optimizer.zero_grad(set_to_none=True)
            raise RuntimeError("episode boundary has no staged online state")
        # A failed state commit must occur before optimizer/scaler mutation. Scaled
        # gradients are sufficient for the first non-finite guard.
        bad_param_name = _find_first_nonfinite_grad(model)
        if bad_param_name is not None:
            logger.error(
                "[Train]: non-finite episode gradients at epoch=%d iter=%d param=%s; rollback",
                curr_epoch,
                iter_idx,
                bad_param_name,
            )
            debug_report = _collect_runtime_debug(model, data_dict, bad_param_name)
            if debug_report is not None:
                debug_report.update(_summarize_grad_health(model))
                logger.error(
                    "[Train][Diag]: epoch=%d iter=%d %s",
                    curr_epoch,
                    iter_idx,
                    _format_debug_report(debug_report),
                )
            if crs_control is not None:
                _rollback_crs_group(model, transaction, optimizer, crs_buffer_snapshot)
            else:
                _rollback_online_transaction(transaction)
                optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.update()
            episode_weight = 0.0
            episode_loss_records.clear()
            skipped_optimizer_events += 1
            if fixed_step_profiler is not None:
                fixed_step_profiler.record_skipped_optimizer_event()
            if optimizer_event_recorder is not None:
                raise RuntimeError(
                    "authenticated training cannot continue after a skipped "
                    "optimizer boundary"
                )
            episode_input_tokens = 0
            active_crs_group = None
            crs_buffer_snapshot = None
            crs_expected_draw_index = None
            crs_episode_payload_sha256s = None
            crs_expected_sequence_sha256 = None
            group_workload = None
            group_started_at = None
            continue

        mutation_snapshot = None
        boundary_proof = None
        try:
            if transaction is not None:
                mutation_snapshot = _capture_training_mutation_snapshot(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    model_ema=model_ema,
                    optimizer_event_recorder=optimizer_event_recorder,
                    visual_parameter_event_recorder=visual_parameter_event_recorder,
                    transaction=transaction,
                )
            if optimizer_event_recorder is not None:
                boundary_proof = optimizer_event_recorder.begin_optimizer_boundary(
                    optimizer, transaction
                )
            if scaler is not None:
                scaler.unscale_(optimizer)
            _normalize_accumulated_gradients(model, episode_weight)
            if clip_grad_l2norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    _grad_clip_parameters(model), clip_grad_l2norm
                )
            bad_param_name = _find_first_nonfinite_grad(model)
            if bad_param_name is not None:
                raise RuntimeError(
                    "gradient state became non-finite before optimizer mutation at "
                    f"parameter {bad_param_name}"
                )
            if visual_parameter_event_recorder is not None:
                visual_parameter_event_recorder.capture_before(model, optimizer)
            if optimizer_event_recorder is not None:
                optimizer_event_recorder.execute_optimizer_step(
                    boundary_proof, optimizer, scaler=scaler
                )
            elif scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            if model_ema is not None:
                model_ema.update(model)
            if optimizer_event_recorder is not None:
                optimizer_event_recorder.commit_online_transaction(
                    boundary_proof, transaction
                )
            elif transaction is not None:
                transaction.commit_online_update()
            optimizer_event = None
            if optimizer_event_recorder is not None:
                optimizer_event = optimizer_event_recorder.record(
                    epoch=curr_epoch,
                    episode_id=_episode_identity(
                        control, curr_epoch, iter_idx, crs_eps=crs_control
                    ),
                    input_tokens=episode_input_tokens,
                    skipped=False,
                    boundary_proof=boundary_proof,
                )
            if visual_parameter_event_recorder is not None:
                if not isinstance(optimizer_event, Mapping):
                    raise RuntimeError(
                        "optimizer event recorder did not return an authenticated event"
                    )
                visual_parameter_event_recorder.record_after(
                    optimizer_event["event_id"], model
                )
        except Exception:
            boundary_abort_error = None
            if boundary_proof is not None and optimizer_event_recorder is not None:
                try:
                    optimizer_event_recorder.abort_optimizer_boundary(boundary_proof)
                except Exception as exc:
                    boundary_abort_error = exc
            if mutation_snapshot is not None:
                _rollback_failed_training_mutation(
                    mutation_snapshot,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    model_ema=model_ema,
                    optimizer_event_recorder=optimizer_event_recorder,
                    visual_parameter_event_recorder=visual_parameter_event_recorder,
                    transaction=transaction,
                )
            else:
                if crs_control is not None:
                    _rollback_crs_group(
                        model, transaction, optimizer, crs_buffer_snapshot
                    )
                else:
                    _rollback_online_transaction(transaction)
                    optimizer.zero_grad(set_to_none=True)
            if boundary_abort_error is not None:
                raise RuntimeError(
                    "failed to abort optimizer evidence boundary"
                ) from boundary_abort_error
            raise

        successful_optimizer_events += 1
        group_workload["wall_seconds"] = time.perf_counter() - group_started_at
        active_crs_group = None
        crs_buffer_snapshot = None
        crs_expected_draw_index = None
        crs_episode_payload_sha256s = None
        crs_expected_sequence_sha256 = None
        if crs_control is not None:
            crs_expected_group_index += 1
        episode_input_tokens = 0
        optimizer.zero_grad(set_to_none=True)
        if fixed_step_profiler is not None:
            if not isinstance(optimizer_event, Mapping):
                raise RuntimeError(
                    "fixed-step profiling requires an authenticated optimizer event"
                )
            if fixed_step_profiler.record_optimizer_event(
                optimizer_event["event_id"], workload=group_workload
            ):
                episode_weight = 0.0
                episode_loss_records.clear()
                group_workload = None
                group_started_at = None
                break
        group_workload = None
        group_started_at = None

        for loss_record in episode_loss_records:
            reduced = reduce_loss(loss_record)
            for key, value in reduced.items():
                if key not in losses_tracker:
                    losses_tracker[key] = AverageMeter()
                losses_tracker[key].update(value.item())
        episode_weight = 0.0
        episode_loss_records.clear()

        should_print = (
            ((iter_idx != 0) and (iter_idx % logging_interval) == 0)
            or ((iter_idx + 1) == num_iters)
        ) and "cost" in losses_tracker
        if should_print:
            block1 = "[Train]: [{:03d}][{:05d}/{:05d}]".format(
                curr_epoch, iter_idx, num_iters - 1
            )
            block2 = "Loss={:.4f}".format(losses_tracker["cost"].avg)
            block3 = [
                "{:s}={:.4f}".format(key, value.avg)
                for key, value in losses_tracker.items()
                if key != "cost"
            ]
            block4 = "lr_det={:.1e}".format(curr_det_lr)
            if curr_backbone_lr is not None:
                block4 = (
                    "lr_backbone={:.1e}".format(curr_backbone_lr)
                    + "  "
                    + block4
                )
            block5 = "mem={:.0f}MB".format(
                torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
            )
            logger.info("  ".join([block1, block2, "  ".join(block3), block4, block5]))

        should_log_runtime_debug = (
            runtime_debug_interval > 0
            and ((iter_idx + 1) == num_iters)
            and (((curr_epoch + 1) % runtime_debug_interval) == 0)
        )
        if should_log_runtime_debug:
            is_rank0 = True
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                is_rank0 = torch.distributed.get_rank() == 0
            if is_rank0:
                debug_report = _collect_runtime_debug(model, data_dict, "periodic")
                if debug_report is not None:
                    logger.info(
                        "[Train][RuntimeDebug]: epoch=%d iter=%d %s",
                        curr_epoch,
                        iter_idx,
                        _format_debug_report(debug_report),
                    )

    if skip_until_boundary:
        raise RuntimeError("failed online episode ended before its declared boundary")
    if active_crs_group is not None or crs_buffer_snapshot is not None:
        _rollback_crs_group(model, transaction, optimizer, crs_buffer_snapshot)
        raise RuntimeError("CRS-EPS epoch ended inside a video group")
    if (
        trusted_crs_manifest is not None
        and not (fixed_step_profiler is not None and fixed_step_profiler.complete)
        and crs_expected_group_index != len(trusted_crs_manifest["videos"])
    ):
        raise RuntimeError("CRS-EPS epoch did not consume the immutable manifest exactly")
    if group_workload is not None or group_started_at is not None:
        raise RuntimeError("training epoch ended inside an optimizer workload group")
    if episode_weight or (
        transaction is not None and transaction.has_pending_online_update()
    ):
        _rollback_online_transaction(transaction)
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError("online training epoch ended with an uncommitted episode")
    if episode_input_tokens:
        raise RuntimeError("online training epoch ended with uncommitted optimizer evidence")
    return {
        "optimizer_events": optimizer_events,
        "successful_optimizer_events": successful_optimizer_events,
        "skipped_optimizer_events": skipped_optimizer_events,
        "fixed_step_profile_complete": bool(
            fixed_step_profiler is not None and fixed_step_profiler.complete
        ),
    }


def val_one_epoch(
    val_loader,
    model,
    logger,
    rank,
    curr_epoch,
    model_ema=None,
    use_amp=False,
    amp_dtype=None,
):
    """Validating the model for one epoch: compute the loss"""

    if amp_dtype is None and use_amp:
        amp_dtype = torch.float16
    use_amp = amp_dtype is not None

    # load the ema dict for evaluation
    if model_ema != None:
        current_dict = copy.deepcopy(model.state_dict())
        model.load_state_dict(model_ema.module.state_dict())

    logger.info("[Val]: Epoch {:d} Loss".format(curr_epoch))
    losses_tracker = {}

    model.eval()
    target = _unwrap_model(model)
    if hasattr(target, "reset_online_states"):
        target.reset_online_states()
    transaction = _transaction_target(model)
    model_device = get_model_device(model)
    for raw_data_dict in tqdm.tqdm(val_loader, disable=(rank != 0)):
        control = _transaction_control(raw_data_dict) if transaction is not None else None
        boundary = True if control is None else bool(
            control["is_video_end"] or control.get("reset_stream", False)
        )
        data_dict = move_data_to_device(raw_data_dict, model_device)
        try:
            with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
                with torch.no_grad():
                    losses = model(**data_dict, return_loss=True)
        except Exception:
            _rollback_online_transaction(transaction)
            raise
        if not torch.isfinite(losses["cost"].detach()).item():
            _rollback_online_transaction(transaction)
            raise RuntimeError("validation produced a non-finite online episode loss")
        if transaction is not None and boundary:
            transaction.commit_online_update()

        # track all losses
        losses = reduce_loss(losses)  # only for log
        for key, value in losses.items():
            if key not in losses_tracker:
                losses_tracker[key] = AverageMeter()
            losses_tracker[key].update(value.item())

    if transaction is not None and transaction.has_pending_online_update():
        transaction.rollback_online_update()
        raise RuntimeError("validation ended with an uncommitted online episode")

    # print to terminal
    block1 = "[Val]: [{:03d}]".format(curr_epoch)
    block2 = "Loss={:.4f}".format(losses_tracker["cost"].avg)
    block3 = ["{:s}={:.4f}".format(key, value.avg) for key, value in losses_tracker.items() if key != "cost"]
    logger.info("  ".join([block1, block2, "  ".join(block3)]))

    # load back the normal model dict
    if model_ema != None:
        model.load_state_dict(current_dict)
    return losses_tracker["cost"].avg
