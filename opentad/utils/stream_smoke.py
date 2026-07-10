from dataclasses import asdict, is_dataclass
import math

import torch

from .causal_audit import audit_packet_metadata
from .device import get_model_device, move_data_to_device
from .training_audit import audit_training_update, snapshot_trainable_parameters


class StreamSmokeError(RuntimeError):
    pass


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return value


def _packet_meta(batch):
    metas = batch.get("metas")
    if not isinstance(metas, (list, tuple)) or len(metas) != 1:
        raise StreamSmokeError("stream smoke requires exactly one metadata lane per packet")
    meta = dict(metas[0])
    meta.setdefault("video_id", meta.get("video_name"))
    return meta


def run_stream_smoke(
    model,
    dataloader,
    optimizer=None,
    max_packets=8,
    device=None,
    required_module_prefixes=(),
    forward_kwargs=None,
):
    """Run a bounded chronological smoke and return a JSON-safe audit report."""

    max_packets = int(max_packets)
    if max_packets <= 0:
        raise ValueError("max_packets must be positive")
    forward_kwargs = dict(forward_kwargs or {})
    device = get_model_device(model) if device is None else torch.device(device)
    target_model = getattr(model, "module", model)
    if hasattr(target_model, "reset_online_states"):
        target_model.reset_online_states()

    training = optimizer is not None
    model.train(training)
    steps = []
    packet_rows = []
    read_traces = []
    for packet_index, raw_batch in enumerate(dataloader):
        if packet_index >= max_packets:
            break
        batch = move_data_to_device(raw_batch, device)
        meta = _packet_meta(batch)
        packet_rows.append(
            {
                key: meta[key]
                for key in (
                    "video_id",
                    "packet_start_frame",
                    "packet_end_frame",
                    "is_video_start",
                    "is_video_end",
                )
            }
        )
        step = {"packet_index": packet_index, "meta": _jsonable(meta)}
        if training:
            before = snapshot_trainable_parameters(model)
            optimizer.zero_grad(set_to_none=True)
            losses = model(**batch, return_loss=True, **forward_kwargs)
            if not isinstance(losses, dict) or "cost" not in losses:
                raise StreamSmokeError("training smoke requires a loss dict containing cost")
            cost = losses["cost"]
            if not torch.is_tensor(cost) or not bool(torch.isfinite(cost).all().item()):
                raise StreamSmokeError(f"non-finite cost at packet {packet_index}")
            cost.backward()
            optimizer.step()
            update_report = audit_training_update(
                model,
                before,
                required_module_prefixes=required_module_prefixes,
            )
            update_report.raise_for_errors()
            step["losses"] = {
                key: float(value.detach().float().mean().item())
                for key, value in losses.items()
                if torch.is_tensor(value) and math.isfinite(float(value.detach().float().mean().item()))
            }
            step["update_audit"] = update_report.as_dict()
        else:
            with torch.no_grad():
                results = model(**batch, return_loss=False, **forward_kwargs)
            step["results"] = _jsonable(results)

        trace = getattr(target_model, "last_read_trace", None)
        if trace is None:
            raise StreamSmokeError(f"model did not expose last_read_trace at packet {packet_index}")
        trace = _jsonable(trace)
        read_traces.append(trace)
        step["read_trace"] = trace
        steps.append(step)

    if not steps:
        raise StreamSmokeError("stream smoke did not process any packets")
    packet_report = audit_packet_metadata(packet_rows)
    return {
        "mode": "train_step" if training else "inference",
        "device": str(device),
        "packets_processed": len(steps),
        "packet_audit": asdict(packet_report),
        "read_traces": read_traces,
        "steps": steps,
    }
