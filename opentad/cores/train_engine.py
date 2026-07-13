import copy
from collections.abc import Mapping

import torch
import tqdm
from opentad.utils.misc import AverageMeter, reduce_loss
from opentad.utils.device import get_model_device, move_data_to_device


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


def _normalize_accumulated_gradients(model, denominator):
    if denominator <= 0:
        raise RuntimeError("episode gradient denominator must be positive")
    for param in model.parameters():
        if param.grad is not None:
            param.grad.div_(denominator)


def _rollback_online_transaction(transaction):
    if transaction is not None and transaction.has_pending_online_update():
        transaction.rollback_online_update()


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
    skip_until_boundary = False
    optimizer_events = 0
    successful_optimizer_events = 0
    skipped_optimizer_events = 0
    optimizer.zero_grad(set_to_none=True)
    if fixed_step_profiler is not None:
        fixed_step_profiler.start()

    for iter_idx, raw_data_dict in enumerate(train_loader):
        control = _transaction_control(raw_data_dict) if transaction is not None else None
        boundary = True if control is None else bool(
            control["is_video_end"] or control.get("reset_stream", False)
        )
        if skip_until_boundary:
            if control["is_video_start"]:
                raise RuntimeError("a new stream started before the failed episode boundary")
            if boundary:
                skip_until_boundary = False
            continue

        data_dict = move_data_to_device(raw_data_dict, model_device)
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
            _rollback_online_transaction(transaction)
            optimizer.zero_grad(set_to_none=True)
            raise

        cost = losses.get("cost")
        if not torch.is_tensor(cost) or cost.numel() != 1:
            _rollback_online_transaction(transaction)
            optimizer.zero_grad(set_to_none=True)
            raise RuntimeError("model losses must contain one scalar cost tensor")
        if not torch.isfinite(cost.detach()).item():
            logger.error(
                "[Train]: non-finite cost at epoch=%d iter=%d; rollback episode",
                curr_epoch,
                iter_idx,
            )
            _rollback_online_transaction(transaction)
            optimizer.zero_grad(set_to_none=True)
            episode_weight = 0.0
            episode_loss_records.clear()
            optimizer_events += 1
            skipped_optimizer_events += 1
            if fixed_step_profiler is not None:
                fixed_step_profiler.record_skipped_optimizer_event()
            skip_until_boundary = not boundary
            continue

        weight = _optimizer_weight(losses)
        weighted_cost = cost * weight
        if scaler is not None:
            scaler.scale(weighted_cost).backward()
        else:
            weighted_cost.backward()
        episode_weight += weight
        episode_loss_records.append(
            {
                key: value.detach()
                for key, value in losses.items()
                if key != "_optimizer_weight"
            }
        )
        if not boundary:
            continue

        optimizer_events += 1
        if transaction is not None and not transaction.has_pending_online_update():
            optimizer.zero_grad(set_to_none=True)
            raise RuntimeError("episode boundary has no staged online state")
        if scaler is not None:
            scaler.unscale_(optimizer)
        _normalize_accumulated_gradients(model, episode_weight)
        bad_param_name = _find_first_nonfinite_grad(model)
        if bad_param_name is None and clip_grad_l2norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                _grad_clip_parameters(model), clip_grad_l2norm
            )
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
            _rollback_online_transaction(transaction)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.update()
            episode_weight = 0.0
            episode_loss_records.clear()
            skipped_optimizer_events += 1
            if fixed_step_profiler is not None:
                fixed_step_profiler.record_skipped_optimizer_event()
            continue

        try:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if transaction is not None:
                transaction.commit_online_update()
        except Exception:
            _rollback_online_transaction(transaction)
            optimizer.zero_grad(set_to_none=True)
            raise

        scheduler.step()
        successful_optimizer_events += 1
        if model_ema is not None:
            model_ema.update(model)
        optimizer.zero_grad(set_to_none=True)
        if fixed_step_profiler is not None and fixed_step_profiler.record_optimizer_event():
            episode_weight = 0.0
            episode_loss_records.clear()
            break

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
    if episode_weight or (
        transaction is not None and transaction.has_pending_online_update()
    ):
        _rollback_online_transaction(transaction)
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError("online training epoch ended with an uncommitted episode")
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
