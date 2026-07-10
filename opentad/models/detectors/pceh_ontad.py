from dataclasses import asdict, dataclass
import inspect
from typing import Any

import torch
import torch.nn as nn

from opentad.utils.online_protocol import ProtocolViolation


@dataclass(frozen=True)
class ReadTrace:
    packet_start_frame: int
    packet_end_frame: int
    max_raw_frame_read: int
    max_cache_source_frame: int
    encoded_source_frames: tuple


@dataclass
class PCEHStreamState:
    stream_key: str
    last_packet_end_frame: int
    feature_cache: torch.Tensor
    cache_mask: torch.Tensor
    cache_source_frames: tuple
    head_state: Any


class PCEHOnlineDetector(nn.Module):
    """Incremental detector for prefix-censored event/emission hazards."""

    def __init__(
        self,
        backbone=None,
        projection=None,
        head=None,
        rpn_head=None,
        cache_size=192,
        detach_stream_state=True,
    ):
        super().__init__()
        self.backbone = self._build_component(backbone, "backbone")
        self.projection = self._build_component(projection, "projection")
        head = head if head is not None else rpn_head
        self.head = self._build_component(head, "head")
        if self.head is None:
            raise ValueError("PCEHOnlineDetector requires a prefix event emission head")
        self.cache_size = int(cache_size)
        self.detach_stream_state = bool(detach_stream_state)
        if self.cache_size <= 0:
            raise ValueError("cache_size must be positive")
        self._stream_states = {}
        self.last_read_trace = None
        self.last_step_audit = None

    @staticmethod
    def _build_component(component, kind):
        if component is None or isinstance(component, nn.Module):
            return component
        if not isinstance(component, dict):
            raise TypeError(f"{kind} must be an nn.Module, config dict, or None")
        from opentad.models.builder import build_backbone, build_head, build_projection

        builders = {
            "backbone": build_backbone,
            "projection": build_projection,
            "head": build_head,
        }
        return builders[kind](component)

    @staticmethod
    def _stream_key(packet_meta):
        video_id = packet_meta.get("video_id", packet_meta.get("video_name"))
        if video_id is None:
            raise ProtocolViolation("stream packet requires video_id or video_name")
        return f"video={video_id}|stream={packet_meta.get('stream_id', 'default')}"

    def reset_online_states(self):
        self._stream_states.clear()
        self.last_read_trace = None
        self.last_step_audit = None

    def _call_backbone(self, new_frames, masks, packet_meta):
        if self.backbone is None:
            return new_frames, masks
        try:
            params = inspect.signature(self.backbone.forward).parameters
        except (TypeError, ValueError):
            return self.backbone(new_frames), masks

        accepts_kwargs = any(param.kind == param.VAR_KEYWORD for param in params.values())
        if accepts_kwargs or ("masks" in params and "metas" in params):
            output = self.backbone(new_frames, masks=masks, metas=[packet_meta])
        elif "masks" in params:
            output = self.backbone(new_frames, masks=masks)
        else:
            output = self.backbone(new_frames)
        return self._unpack_features(output, masks)

    @staticmethod
    def _unpack_features(output, masks):
        if isinstance(output, dict):
            features = output.get("features", output.get("feats", output.get("last_hidden_state")))
            if features is None:
                raise ValueError("component output dict did not contain features")
            return features, output.get("masks", masks)
        if isinstance(output, tuple) and len(output) >= 2 and torch.is_tensor(output[1]):
            return output[0], output[1]
        return output, masks

    @staticmethod
    def _ensure_feature_sequence(features):
        if isinstance(features, (list, tuple)):
            if not features:
                raise ValueError("feature list must not be empty")
            features = features[0]
        if not torch.is_tensor(features) or features.ndim != 3:
            raise ValueError("incremental features must have shape [B,C,T]")
        if features.shape[0] != 1:
            raise ProtocolViolation("PCEH forward_step currently supports one stream lane per call")
        return features

    @staticmethod
    def _ensure_mask(masks, features):
        if isinstance(masks, (list, tuple)):
            masks = masks[0]
        if masks is None or not torch.is_tensor(masks) or masks.shape != (features.shape[0], features.shape[-1]):
            return torch.ones(
                (features.shape[0], features.shape[-1]),
                dtype=torch.bool,
                device=features.device,
            )
        return masks.to(device=features.device, dtype=torch.bool)

    @staticmethod
    def _resolve_encoded_source_frames(packet_meta, token_count):
        start = int(packet_meta["packet_start_frame"])
        end = int(packet_meta["packet_end_frame"])
        explicit = packet_meta.get("encoded_source_frames")
        if explicit is None and packet_meta.get("irregular_selected_positions") is not None:
            stride = int(packet_meta.get("snippet_stride", 1))
            offset = int(packet_meta.get("offset_frames", 0))
            explicit = [
                start + int(position) * stride + offset
                for position in packet_meta["irregular_selected_positions"]
            ]
        if explicit is not None:
            source_frames = tuple(int(value) for value in explicit)
            if len(source_frames) != token_count:
                raise ProtocolViolation(
                    f"encoded source count {len(source_frames)} does not match token count {token_count}"
                )
        elif token_count == end - start:
            source_frames = tuple(range(start, end))
        elif token_count == 1:
            source_frames = (end - 1,)
        else:
            source_frames = tuple(
                int(round(value))
                for value in torch.linspace(start, end - 1, token_count).tolist()
            )
        if tuple(sorted(source_frames)) != source_frames:
            raise ProtocolViolation("encoded source frames must be chronological")
        if any(value < start or value >= end for value in source_frames):
            raise ProtocolViolation(
                f"encoded source frames must stay inside packet [{start}, {end}): {source_frames}"
            )
        return source_frames

    def _project(self, features, masks):
        if self.projection is None:
            return features, masks
        output = self.projection(features, masks)
        if not isinstance(output, tuple) or len(output) < 2:
            raise ValueError("projection must return (features, masks)")
        projected, projected_masks = output[0], output[1]
        projected = self._ensure_feature_sequence(projected)
        projected_masks = self._ensure_mask(projected_masks, projected)
        return projected, projected_masks

    def forward_step(self, new_frames, state, packet_meta, masks=None, targets=None):
        required = (
            "packet_start_frame",
            "packet_end_frame",
            "is_video_start",
            "is_video_end",
        )
        missing = [key for key in required if key not in packet_meta]
        if missing:
            raise ProtocolViolation(f"stream packet metadata missing fields: {missing}")
        packet_start = int(packet_meta["packet_start_frame"])
        packet_end = int(packet_meta["packet_end_frame"])
        if packet_end <= packet_start:
            raise ProtocolViolation("packet_end_frame must exceed packet_start_frame")
        stream_key = self._stream_key(packet_meta)

        if state is None:
            if not bool(packet_meta["is_video_start"]):
                raise ProtocolViolation("first packet must set is_video_start=True")
        else:
            if state.stream_key != stream_key:
                raise ProtocolViolation(
                    f"stream key mismatch: state={state.stream_key} packet={stream_key}"
                )
            if bool(packet_meta["is_video_start"]):
                raise ProtocolViolation("existing stream state cannot receive is_video_start=True")
            if packet_start != state.last_packet_end_frame:
                raise ProtocolViolation(
                    f"packets must be chronological: expected_start={state.last_packet_end_frame} "
                    f"actual_start={packet_start}"
                )

        current_features, current_masks = self._call_backbone(new_frames, masks, packet_meta)
        current_features = self._ensure_feature_sequence(current_features)
        current_masks = self._ensure_mask(current_masks, current_features)
        encoded_sources = self._resolve_encoded_source_frames(packet_meta, current_features.shape[-1])

        if state is None:
            feature_cache = current_features
            cache_mask = current_masks
            cache_sources = encoded_sources
            head_state = self.head.initial_state(stream_key=stream_key)
        else:
            feature_cache = torch.cat(
                [state.feature_cache.to(current_features), current_features],
                dim=-1,
            )
            cache_mask = torch.cat(
                [state.cache_mask.to(current_masks), current_masks],
                dim=-1,
            )
            cache_sources = tuple(state.cache_source_frames) + encoded_sources
            head_state = state.head_state

        feature_cache = feature_cache[..., -self.cache_size :]
        cache_mask = cache_mask[..., -self.cache_size :]
        cache_sources = cache_sources[-self.cache_size :]
        projected, projected_masks = self._project(feature_cache, cache_mask)
        valid_indices = projected_masks[0].nonzero(as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            raise ProtocolViolation("stream packet produced no valid causal feature token")
        newest = projected[..., int(valid_indices[-1].item())]
        logits = self.head(newest)

        current_frame = packet_end - 1
        max_raw_frame_read = max(encoded_sources)
        max_cache_source_frame = max(cache_sources)
        decode_meta = {
            "current_frame": current_frame,
            "max_raw_frame_read": max_raw_frame_read,
            "max_cache_source_frame": max_cache_source_frame,
        }
        emissions, head_state = self.head.decode_step(logits, head_state, decode_meta)
        losses = self.head.losses(logits, targets) if targets is not None else None

        stored_features = feature_cache.detach() if self.detach_stream_state else feature_cache
        stored_masks = cache_mask.detach() if self.detach_stream_state else cache_mask
        new_state = PCEHStreamState(
            stream_key=stream_key,
            last_packet_end_frame=packet_end,
            feature_cache=stored_features,
            cache_mask=stored_masks,
            cache_source_frames=tuple(cache_sources),
            head_state=head_state,
        )
        trace = ReadTrace(
            packet_start_frame=packet_start,
            packet_end_frame=packet_end,
            max_raw_frame_read=max_raw_frame_read,
            max_cache_source_frame=max_cache_source_frame,
            encoded_source_frames=encoded_sources,
        )
        self.last_read_trace = trace
        active_tracks = getattr(head_state, "active_tracks", {})
        ledger = getattr(head_state, "ledger", ())
        self.last_step_audit = {
            "time": current_frame,
            "packet_start_frame": packet_start,
            "packet_end_frame": packet_end,
            "max_raw_frame_read": max_raw_frame_read,
            "max_cache_source_frame": max_cache_source_frame,
            "encoded_source_frames": list(encoded_sources),
            "logits": {
                key: value.detach().float().cpu().tolist()
                for key, value in sorted(logits.items())
                if torch.is_tensor(value)
            },
            "active_tracks": {
                str(key): asdict(value) if hasattr(value, "__dataclass_fields__") else repr(value)
                for key, value in sorted(active_tracks.items(), key=lambda item: str(item[0]))
            },
            "ledger_size": len(ledger),
            "emissions": [
                asdict(record) if hasattr(record, "__dataclass_fields__") else dict(vars(record))
                for record in emissions
            ],
        }
        return {"logits": logits, "losses": losses, "emissions": emissions}, new_state, trace

    @staticmethod
    def _targets_to_tensors(prefix_targets, reference):
        target_names = (
            "class_target",
            "start_target",
            "ongoing_target",
            "end_event",
            "censor_mask",
            "completion_target",
            "emit_allowed",
            "emit_forbidden",
            "emit_event",
            "late_target",
        )
        return {
            name: torch.as_tensor(
                [getattr(prefix_targets, name)],
                dtype=reference.dtype,
                device=reference.device,
            )
            for name in target_names
        }

    def forward(
        self,
        inputs,
        masks,
        metas,
        gt_segments=None,
        gt_labels=None,
        return_loss=True,
        infer_cfg=None,
        post_cfg=None,
        stream_gt_segments=None,
        stream_gt_labels=None,
        **kwargs,
    ):
        ext_cls = kwargs.pop("ext_cls", None)
        del gt_segments, gt_labels, infer_cfg, post_cfg, kwargs
        if inputs.shape[0] != 1 or len(metas) != 1:
            raise ProtocolViolation("PCEH standard forward currently requires batch_size=1")
        packet_meta = dict(metas[0])
        packet_meta.setdefault("video_id", packet_meta.get("video_name"))
        stream_key = self._stream_key(packet_meta)
        state = None if packet_meta.get("is_video_start") else self._stream_states.get(stream_key)

        targets = None
        if return_loss:
            if stream_gt_segments is None or stream_gt_labels is None:
                raise ProtocolViolation("PCEH training requires absolute stream GT targets")
            from opentad.models.targets.prefix_event_targets import build_prefix_event_targets

            previous_frame = (
                int(packet_meta["packet_start_frame"]) - 1
                if state is None
                else state.last_packet_end_frame - 1
            )
            prefix_targets = build_prefix_event_targets(
                stream_gt_segments[0],
                stream_gt_labels[0],
                previous_frame=previous_frame,
                current_frame=int(packet_meta["packet_end_frame"]) - 1,
                num_classes=self.head.num_classes,
                delay_budget_frames=float(packet_meta.get("delay_budget_frames", 0)),
            )
            reference = inputs.new_zeros((1, self.head.num_classes), dtype=torch.float32)
            targets = self._targets_to_tensors(prefix_targets, reference)

        output, state, trace = self.forward_step(
            new_frames=inputs,
            state=state,
            packet_meta=packet_meta,
            masks=masks,
            targets=targets,
        )
        self._stream_states[stream_key] = state
        if return_loss:
            return output["losses"]

        fps = float(packet_meta.get("fps", 1.0))
        video_id = str(packet_meta["video_id"])
        rows = []
        for record in output["emissions"]:
            label = record.label
            if isinstance(ext_cls, (list, tuple)) and 0 <= int(label) < len(ext_cls):
                label = ext_cls[int(label)]
            elif isinstance(ext_cls, dict):
                label = ext_cls.get(label, ext_cls.get(str(label), label))
            rows.append(
                {
                    "segment": [record.start_frame / fps, record.end_frame / fps],
                    "label": label,
                    "score": record.score,
                    "start_frame": record.start_frame,
                    "end_frame": record.end_frame,
                    "emit_frame": record.emit_frame,
                    "emit_time_sec": record.emit_frame / fps,
                    "source_frame": record.max_raw_frame_read,
                    "source_time_sec": record.max_raw_frame_read / fps,
                    "max_raw_frame_read": record.max_raw_frame_read,
                    "max_cache_source_frame": record.max_cache_source_frame,
                    "fps": fps,
                    "latency_sec": (record.emit_frame - record.end_frame) / fps,
                    "stream_key": stream_key,
                    "immutable": True,
                    "read_trace": trace.__dict__,
                }
            )
        return {video_id: rows}
