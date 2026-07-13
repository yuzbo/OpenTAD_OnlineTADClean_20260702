from dataclasses import dataclass, replace
import hashlib
import inspect
import json
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import DETECTORS, build_backbone, build_head, build_projection
from ..dense_heads.persistent_event_set_head import SLOT_FREE, PersistentEventSetState
from opentad.utils.immutable_event_ledger import ImmutableEventLedger
from opentad.utils.online_protocol import ProtocolViolation
from opentad.utils.prefix_trajectory_supervision import (
    PrefixTrajectorySupervisionState,
    SupervisionMode,
)
from opentad.utils.stream_control import (
    StreamMetadataError,
    validate_model_meta,
    validate_stream_control,
)


_BINDING_MODES = {
    "fixed": SupervisionMode.FIXED,
    "fixed_birth_slot": SupervisionMode.FIXED,
    "rematch": SupervisionMode.REMATCH,
    "prefix_rematch_active_pool": SupervisionMode.REMATCH,
}


def _formal_provenance_digest(input_digest, record):
    if (
        not isinstance(input_digest, str)
        or len(input_digest) != 64
        or any(character not in "0123456789abcdef" for character in input_digest)
    ):
        raise ProtocolViolation("input provenance must be a lowercase SHA-256 digest")
    material = {
        "input_provenance_digest": input_digest,
        "stream_key": record.stream_key,
        "slot_id": int(record.slot_id),
        "emit_frame": int(record.emit_frame),
        "max_source_frame": int(record.max_source_frame),
    }
    encoded = json.dumps(
        material,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _zero(reference):
    return reference.float().sum() * 0.0


def _masked_bce(logits, target, mask):
    mask = mask.to(device=logits.device, dtype=torch.bool)
    if not mask.any():
        return _zero(logits)
    return F.binary_cross_entropy_with_logits(logits[mask], target[mask])


def _stream_key(meta):
    video = meta.get("video_id", meta.get("video_name"))
    if video is None:
        raise ProtocolViolation("model metadata requires video_id or video_name")
    return (
        f"video={video}|stream={meta.get('stream_id', 'default')}|"
        f"input={meta.get('input_format', 'unknown')}|"
        f"stride={meta.get('feature_stride', meta.get('snippet_stride', 'unknown'))}"
    )


@dataclass
class PersistentTrajectoryRuntimeState:
    """Inference-visible state. Training identities are intentionally absent."""

    stream_key: str
    queries: torch.Tensor
    feature_memory: torch.Tensor
    source_frames: Tuple[int, ...]
    slot_status: torch.Tensor
    refractory: torch.Tensor
    start_state: torch.Tensor
    score_state: torch.Tensor
    label_state: torch.Tensor
    ledger_rows: Tuple[dict, ...]
    last_decision_frame: int


@dataclass(frozen=True)
class TrajectoryInferenceOutput:
    logits: Tuple[dict, ...]
    emissions: Tuple[dict, ...]
    provisional: Tuple[dict, ...]
    runtime_state: PersistentTrajectoryRuntimeState


@dataclass(frozen=True)
class TrajectoryEpisodeLossOutput:
    losses: Dict[str, torch.Tensor]
    runtime_state: PersistentTrajectoryRuntimeState
    supervision_state: PrefixTrajectorySupervisionState
    logits: Tuple[dict, ...]
    audit: dict


@DETECTORS.register_module()
class PersistentTrajectoryOnlineDetector(nn.Module):
    """Causal persistent-trajectory detector with isolated training supervision."""

    def __init__(
        self,
        head,
        backbone=None,
        projection=None,
        trajectory_binding_mode="fixed_birth_slot",
        birth_assignment_mode="first_crossing_shared",
        canonical_supervision_lifecycle=True,
        detach_stream_state=True,
        birth_loss_weight=1.0,
        alive_loss_weight=0.5,
        class_loss_weight=1.0,
        start_loss_weight=1.0,
        end_loss_weight=1.0,
    ):
        super().__init__()
        self.head = head if isinstance(head, nn.Module) else build_head(head)
        self.backbone = self._build_optional(backbone, build_backbone)
        self.projection = self._build_optional(projection, build_projection)
        try:
            self.supervision_mode = _BINDING_MODES[str(trajectory_binding_mode)]
        except KeyError as exc:
            raise ValueError(
                "trajectory_binding_mode must be fixed_birth_slot or "
                "prefix_rematch_active_pool"
            ) from exc
        self.trajectory_binding_mode = str(trajectory_binding_mode)
        if birth_assignment_mode != "first_crossing_shared":
            raise ValueError("only shared first-crossing birth assignment is supported")
        if canonical_supervision_lifecycle is not True:
            raise ValueError("canonical supervision lifecycle must remain shared")
        if self.head.query_mode != "persistent":
            raise ValueError("the Q2 detector requires persistent query recurrence")
        if self.head.start_mode != "scalar":
            raise ValueError("the Q2 mechanism gate requires scalar start supervision")
        if self.head.endpoint_mode != "binary":
            raise ValueError("the Q2 mechanism gate requires binary endpoint supervision")
        self.detach_stream_state = bool(detach_stream_state)
        self.loss_weights = {
            "birth_loss": float(birth_loss_weight),
            "alive_loss": float(alive_loss_weight),
            "class_loss": float(class_loss_weight),
            "start_loss": float(start_loss_weight),
            "end_loss": float(end_loss_weight),
        }
        self._runtime_states = {}
        self._supervision_states = {}
        self._staged_runtime_states = {}
        self._staged_supervision_states = {}
        self._staged_terminal_keys = set()
        self.last_episode_audit = {}

    @staticmethod
    def _build_optional(component, builder):
        if component is None or isinstance(component, nn.Module):
            return component
        if not isinstance(component, dict):
            raise TypeError("model components must be config dictionaries or nn.Module objects")
        return builder(component)

    def reset_online_states(self):
        self._runtime_states.clear()
        self._supervision_states.clear()
        self._staged_runtime_states.clear()
        self._staged_supervision_states.clear()
        self._staged_terminal_keys.clear()
        self.last_episode_audit = {}

    def has_pending_online_update(self):
        return bool(self._staged_runtime_states or self._staged_supervision_states)

    def commit_online_update(self):
        if not self.has_pending_online_update():
            raise ProtocolViolation("no staged online training state is available to commit")
        if set(self._staged_runtime_states) != set(self._staged_supervision_states):
            raise ProtocolViolation("staged runtime and supervision keys are inconsistent")
        if set(self._staged_runtime_states) != self._staged_terminal_keys:
            raise ProtocolViolation("only a complete terminal episode may be committed")
        for key, runtime in self._staged_runtime_states.items():
            if key in self._staged_terminal_keys:
                self._runtime_states.pop(key, None)
                self._supervision_states.pop(key, None)
            else:
                self._runtime_states[key] = runtime
                self._supervision_states[key] = self._staged_supervision_states[key]
        self._staged_runtime_states.clear()
        self._staged_supervision_states.clear()
        self._staged_terminal_keys.clear()

    def rollback_online_update(self):
        self._staged_runtime_states.clear()
        self._staged_supervision_states.clear()
        self._staged_terminal_keys.clear()

    def _validate_meta(self, model_meta):
        try:
            meta = validate_model_meta(model_meta)
        except StreamMetadataError as exc:
            raise ProtocolViolation(f"forbidden or tainted model metadata: {exc}") from exc
        if "current_frame" not in meta:
            raise ProtocolViolation("model metadata requires current_frame")
        return meta

    @staticmethod
    def _validate_control(stream_control):
        try:
            return validate_stream_control(stream_control)
        except StreamMetadataError as exc:
            raise ProtocolViolation(f"invalid stream control: {exc}") from exc

    def _initial_runtime_state(self, reference, stream_key):
        state = self.head.initial_state(
            device=reference.device,
            dtype=reference.dtype,
            stream_key=stream_key,
        )
        return PersistentTrajectoryRuntimeState(
            stream_key=stream_key,
            queries=state.queries,
            feature_memory=state.feature_memory,
            source_frames=(),
            slot_status=state.slot_status,
            refractory=state.refractory,
            start_state=state.start_frames,
            score_state=state.peak_class_scores,
            label_state=state.peak_class_labels,
            ledger_rows=(),
            last_decision_frame=-1,
        )

    @staticmethod
    def _to_head_state(state):
        return PersistentEventSetState(
            stream_key=state.stream_key,
            queries=state.queries,
            feature_memory=state.feature_memory,
            memory_frames=tuple(state.source_frames),
            slot_status=state.slot_status,
            refractory=state.refractory,
            start_frames=state.start_state,
            peak_class_scores=state.score_state,
            peak_class_labels=state.label_state,
            ledger=[],
            instance_to_slot={},
            slot_to_instance={},
        )

    def _from_head_state(self, state, ledger_rows, decision_frame):
        runtime = PersistentTrajectoryRuntimeState(
            stream_key=state.stream_key,
            queries=state.queries,
            feature_memory=state.feature_memory,
            source_frames=tuple(int(frame) for frame in state.memory_frames),
            slot_status=state.slot_status,
            refractory=state.refractory,
            start_state=state.start_frames,
            score_state=state.peak_class_scores,
            label_state=state.peak_class_labels,
            ledger_rows=tuple(dict(row) for row in ledger_rows),
            last_decision_frame=int(decision_frame),
        )
        return self._detach_runtime(runtime) if self.detach_stream_state else runtime

    @staticmethod
    def _detach_runtime(state):
        return replace(
            state,
            queries=state.queries.detach(),
            feature_memory=state.feature_memory.detach(),
            slot_status=state.slot_status.detach(),
            refractory=state.refractory.detach(),
            start_state=state.start_state.detach(),
            score_state=state.score_state.detach(),
            label_state=state.label_state.detach(),
            ledger_rows=tuple(dict(row) for row in state.ledger_rows),
        )

    @staticmethod
    def _unpack_features(output, masks):
        if isinstance(output, dict):
            features = output.get(
                "features",
                output.get("feats", output.get("last_hidden_state")),
            )
            if features is None:
                raise ValueError("component output did not contain a feature tensor")
            return features, output.get("masks", masks)
        if isinstance(output, (tuple, list)) and len(output) >= 2:
            if torch.is_tensor(output[1]) and output[1].ndim <= 2:
                return output[0], output[1]
        return output, masks

    def _call_backbone(self, inputs, masks, meta):
        if self.backbone is None:
            return inputs, masks
        try:
            params = inspect.signature(self.backbone.forward).parameters
        except (TypeError, ValueError):
            return self._unpack_features(self.backbone(inputs), masks)
        accepts_kwargs = any(param.kind == param.VAR_KEYWORD for param in params.values())
        if accepts_kwargs or ("masks" in params and "metas" in params):
            output = self.backbone(inputs, masks=masks, metas=[meta])
        elif "masks" in params:
            output = self.backbone(inputs, masks=masks)
        else:
            output = self.backbone(inputs)
        return self._unpack_features(output, masks)

    def _encode(self, inputs, masks, meta):
        features, masks = self._call_backbone(inputs, masks, meta)
        if isinstance(features, (tuple, list)):
            if len(features) != 1:
                raise ValueError("persistent trajectory detector requires one feature scale")
            features = features[0]
        if not torch.is_tensor(features) or features.ndim != 3 or features.shape[0] != 1:
            raise ProtocolViolation("encoded causal features must have shape [1,C,T]")
        if masks is None or not torch.is_tensor(masks) or masks.shape != (1, features.shape[-1]):
            masks = torch.ones(
                (1, features.shape[-1]),
                dtype=torch.bool,
                device=features.device,
            )
        else:
            masks = masks.to(device=features.device, dtype=torch.bool)
        if self.projection is not None:
            projected = self.projection(features, masks)
            if not isinstance(projected, tuple) or len(projected) < 2:
                raise ValueError("projection must return (features, masks)")
            features, masks = projected[0], projected[1]
            if isinstance(features, (tuple, list)):
                if len(features) != 1:
                    raise ValueError("projection must expose exactly one causal feature scale")
                features = features[0]
            if isinstance(masks, (tuple, list)):
                masks = masks[0]
        if features.shape[1] != self.head.in_channels:
            raise ValueError(
                f"feature channels {features.shape[1]} do not match head input {self.head.in_channels}"
            )
        source_frames = meta.get("encoded_source_frames", meta.get("source_frames"))
        if source_frames is None:
            raise ProtocolViolation("model metadata requires source_frames provenance")
        source_frames = tuple(int(frame) for frame in source_frames)
        if len(source_frames) != features.shape[-1]:
            raise ProtocolViolation("source frame provenance must align with encoded tokens")
        if any(right <= left for left, right in zip(source_frames, source_frames[1:])):
            raise ProtocolViolation("source frame provenance must be strictly increasing")
        if source_frames and source_frames[-1] > int(meta["current_frame"]):
            raise ProtocolViolation("encoded source frame exceeds the decision frame")
        return features, masks.to(dtype=torch.bool), source_frames

    @staticmethod
    def _targets_by_id(schedule_step):
        targets = {}
        for item in tuple(schedule_step.births) + tuple(schedule_step.active) + tuple(schedule_step.ends):
            targets[int(item.instance_id)] = item
        return targets

    def _validate_schedule_step(
        self,
        schedule_step,
        source_frame,
        previous_decision_frame,
        *,
        masked,
    ):
        if not all(hasattr(schedule_step, field) for field in ("current_frame", "births", "active", "ends")):
            raise ProtocolViolation("supervision schedule step has an invalid schema")
        current_frame = int(schedule_step.current_frame)
        if current_frame != int(source_frame):
            raise ProtocolViolation("schedule decision frame does not match source provenance")
        channels = {
            "birth": tuple(schedule_step.births),
            "active": tuple(schedule_step.active),
            "end": tuple(schedule_step.ends),
        }
        if masked:
            if any(channels.values()):
                raise ProtocolViolation("masked stream tokens cannot carry supervision")
            return

        for channel, items in channels.items():
            for item in items:
                instance_id = getattr(item, "instance_id", None)
                label = getattr(item, "label", None)
                if type(instance_id) is not int or instance_id < 0:
                    raise ProtocolViolation("schedule instance ids must be non-negative integers")
                if type(label) is not int or not 0 <= label < self.head.num_classes:
                    raise ProtocolViolation("schedule labels must lie in the configured class range")
                try:
                    start_frame = float(item.start_frame)
                except (TypeError, ValueError) as exc:
                    raise ProtocolViolation("schedule start frame must be finite") from exc
                if not math.isfinite(start_frame) or start_frame > current_frame:
                    raise ProtocolViolation("schedule contains a future start frame")
                endpoint = getattr(item, "end_frame", None)
                if channel in {"birth", "active"} and endpoint is not None:
                    raise ProtocolViolation("birth/active schedule contains a future endpoint")
                if channel == "birth" and start_frame <= previous_decision_frame:
                    raise ProtocolViolation("birth target is not a first-observable crossing")
                if channel == "end":
                    try:
                        endpoint = float(endpoint)
                    except (TypeError, ValueError) as exc:
                        raise ProtocolViolation("end target requires a finite endpoint") from exc
                    if (
                        not math.isfinite(endpoint)
                        or endpoint <= start_frame
                        or endpoint > current_frame
                    ):
                        raise ProtocolViolation("end target contains a future endpoint")
                    if endpoint <= previous_decision_frame:
                        raise ProtocolViolation("endpoint target is not a first-observable crossing")

    def _cost_provider(self, outputs, schedule_step, feature_stride):
        targets = self._targets_by_id(schedule_step)
        class_log_probs = outputs["class_logits"][0].log_softmax(dim=-1)

        def provider(phase, instance_ids, slot_ids):
            rows = []
            for instance_id in instance_ids:
                target = targets[int(instance_id)]
                target_offset = (
                    float(schedule_step.current_frame) - float(target.start_frame)
                ) / max(float(feature_stride), 1.0)
                target_offset = min(max(target_offset, 0.0), float(self.head.memory_size))
                row = []
                for slot in slot_ids:
                    class_cost = -class_log_probs[int(slot), int(target.label)]
                    start_cost = F.smooth_l1_loss(
                        outputs["start_offset"][0, int(slot)],
                        outputs["start_offset"].new_tensor(target_offset),
                        reduction="sum",
                    )
                    cost = class_cost + start_cost
                    if phase == "birth":
                        cost = cost - F.logsigmoid(outputs["birth_logits"][0, int(slot)])
                    elif phase != "rematch":
                        raise ValueError(f"unknown supervision assignment phase {phase!r}")
                    row.append(cost.detach())
                rows.append(torch.stack(row))
            if not rows:
                return torch.empty((0, len(slot_ids))).numpy()
            return torch.stack(rows).float().cpu().numpy()

        return provider

    def _step_losses(self, outputs, schedule_step, transition, feature_stride):
        birth_target = torch.zeros_like(outputs["birth_logits"])
        birth_mask = torch.as_tensor(
            transition.birth_mask,
            dtype=torch.bool,
            device=birth_target.device,
        ).unsqueeze(0)
        for binding in transition.birth_assignments:
            birth_target[:, int(binding.slot_id)] = 1.0

        alive_target = torch.zeros_like(outputs["alive_logits"])
        for slot in transition.audit.occupied_slots_for_supervision:
            alive_target[:, int(slot)] = 1.0
        alive_mask = torch.ones_like(alive_target, dtype=torch.bool)

        end_target = torch.zeros_like(outputs["end_hazard_logits"])
        end_mask = torch.as_tensor(
            transition.at_risk_mask,
            dtype=torch.bool,
            device=end_target.device,
        ).unsqueeze(0)
        for slot in transition.endpoint_slots:
            end_target[:, int(slot)] = 1.0

        targets = self._targets_by_id(schedule_step)
        class_loss = _zero(outputs["class_logits"])
        start_loss = _zero(outputs["start_offset"])
        if transition.loss_bindings:
            slots = torch.as_tensor(
                [int(binding.slot_id) for binding in transition.loss_bindings],
                dtype=torch.long,
                device=outputs["class_logits"].device,
            )
            labels = torch.as_tensor(
                [int(targets[binding.instance_id].label) for binding in transition.loss_bindings],
                dtype=torch.long,
                device=outputs["class_logits"].device,
            )
            class_loss = F.cross_entropy(outputs["class_logits"][0, slots], labels)
            start_targets = torch.as_tensor(
                [
                    min(
                        max(
                            (
                                float(schedule_step.current_frame)
                                - float(targets[binding.instance_id].start_frame)
                            )
                            / max(float(feature_stride), 1.0),
                            0.0,
                        ),
                        float(self.head.memory_size),
                    )
                    for binding in transition.loss_bindings
                ],
                dtype=outputs["start_offset"].dtype,
                device=outputs["start_offset"].device,
            )
            start_loss = F.smooth_l1_loss(outputs["start_offset"][0, slots], start_targets)

        return {
            "birth_loss": _masked_bce(outputs["birth_logits"], birth_target, birth_mask),
            "alive_loss": _masked_bce(outputs["alive_logits"], alive_target, alive_mask),
            "class_loss": class_loss,
            "start_loss": start_loss,
            "end_loss": _masked_bce(outputs["end_hazard_logits"], end_target, end_mask),
        }

    @staticmethod
    def _binding_rows(bindings):
        return tuple((int(row.instance_id), int(row.slot_id)) for row in bindings)

    def _append_emissions(
        self,
        existing_rows,
        records,
        fps,
        video_id,
        input_provenance_digest,
        class_names=None,
    ):
        ledger = ImmutableEventLedger.from_rows(existing_rows)
        new_rows = []
        fps = float(fps)
        if not math.isfinite(fps) or fps <= 0:
            raise ProtocolViolation("fps must be positive for formal emission timestamps")
        for record in records:
            label = int(record.label)
            if class_names is not None:
                if not isinstance(class_names, (list, tuple)):
                    raise ProtocolViolation("class_names must be a list or tuple")
                label = class_names[label]
            event_id = (
                f"{record.stream_key}:slot={record.slot_id}:emit={record.emit_frame}:"
                f"sequence={len(ledger)}"
            )
            row = ledger.append(
                {
                    "event_id": event_id,
                    "stream_id": record.stream_key,
                    "stream_key": record.stream_key,
                    "video_id": str(video_id),
                    "immutable": True,
                    "slot_id": int(record.slot_id),
                    "label": label,
                    "score": float(record.score),
                    "start_frame": int(record.start_frame),
                    "end_frame": int(record.end_frame),
                    "emit_frame": int(record.emit_frame),
                    "source_frame": int(record.max_source_frame),
                    "input_provenance_digest": input_provenance_digest,
                    "provenance_digest": _formal_provenance_digest(
                        input_provenance_digest, record
                    ),
                    "segment": [record.start_frame / fps, record.end_frame / fps],
                    "emit_time_sec": record.emit_frame / fps,
                    "source_time_sec": record.max_source_frame / fps,
                    "fps": fps,
                    "latency_frames": int(record.emit_frame - record.end_frame),
                    "latency_sec": (record.emit_frame - record.end_frame) / fps,
                    "predicted_end_latency_sec": (
                        record.emit_frame - record.end_frame
                    )
                    / fps,
                    "latency_definition": "emit_time_minus_predicted_end_time",
                }
            )
            new_rows.append(row)
        return ledger.rows, tuple(new_rows)

    def _scan_and_decode(
        self,
        features,
        masks,
        source_frames,
        runtime_state,
        feature_stride,
        fps,
        video_id,
        input_provenance_digest,
        class_names=None,
    ):
        state = self._to_head_state(runtime_state)
        logits = []
        emissions = []
        ledger_rows = runtime_state.ledger_rows
        provisional = []
        last_decision = runtime_state.last_decision_frame
        for index, source_frame in enumerate(source_frames):
            if not bool(masks[0, index].item()):
                continue
            outputs, state = self.head.step(features[:, :, index], state, source_frame)
            records, state = self.head.decode_step(
                outputs,
                state,
                current_frame=source_frame,
                feature_stride=feature_stride,
            )
            ledger_rows, rows = self._append_emissions(
                ledger_rows,
                records,
                fps=fps,
                video_id=video_id,
                input_provenance_digest=input_provenance_digest,
                class_names=class_names,
            )
            emissions.extend(rows)
            class_probabilities = outputs["class_logits"].detach().softmax(dim=-1)[0]
            provisional.append(
                {
                    "decision_frame": int(source_frame),
                    "max_source_frame": int(source_frame),
                    "slot_status": tuple(int(value) for value in state.slot_status.tolist()),
                    "labels": tuple(int(value) for value in class_probabilities.argmax(dim=-1).tolist()),
                    "scores": tuple(float(value) for value in class_probabilities.max(dim=-1).values.tolist()),
                }
            )
            logits.append(outputs)
            last_decision = int(source_frame)
        runtime = self._from_head_state(state, ledger_rows, last_decision)
        return tuple(logits), tuple(emissions), tuple(provisional), runtime

    def infer_step(
        self,
        inputs,
        masks,
        model_meta,
        runtime_state=None,
        class_names=None,
    ):
        meta = self._validate_meta(model_meta)
        features, masks, source_frames = self._encode(inputs, masks, meta)
        stream_key = _stream_key(meta)
        if runtime_state is None:
            runtime_state = self._initial_runtime_state(features, stream_key)
        elif runtime_state.stream_key != stream_key:
            raise ProtocolViolation("runtime state stream key does not match model metadata")
        valid_mask = tuple(bool(value) for value in masks[0].tolist())
        if any(valid_mask[index] and not valid_mask[index - 1] for index in range(1, len(valid_mask))):
            raise ProtocolViolation("stream masks must be right-padded without later valid tokens")
        valid_frames = tuple(
            frame for frame, is_valid in zip(source_frames, valid_mask) if is_valid
        )
        if not valid_frames:
            raise ProtocolViolation("inference packet contains no valid causal token")
        if valid_frames[0] <= runtime_state.last_decision_frame:
            raise ProtocolViolation("stream tokens must be chronological and non-overlapping")
        logits, emissions, provisional, runtime = self._scan_and_decode(
            features,
            masks,
            source_frames,
            runtime_state,
            feature_stride=meta.get("feature_stride", meta.get("snippet_stride", 1)),
            fps=meta.get("fps", 1.0),
            video_id=meta["video_id"],
            input_provenance_digest=meta["input_provenance_digest"],
            class_names=class_names,
        )
        return TrajectoryInferenceOutput(logits, emissions, provisional, runtime)

    def train_episode(
        self,
        inputs,
        masks,
        model_meta,
        supervision_schedule,
        initial_runtime_state=None,
        initial_supervision_state=None,
    ):
        meta = self._validate_meta(model_meta)
        features, masks, source_frames = self._encode(inputs, masks, meta)
        if len(supervision_schedule) != len(source_frames):
            raise ProtocolViolation("supervision schedule must align with encoded source frames")
        stream_key = _stream_key(meta)
        runtime = initial_runtime_state or self._initial_runtime_state(features, stream_key)
        if runtime.stream_key != stream_key:
            raise ProtocolViolation("runtime state stream key does not match the episode")
        supervision = (
            initial_supervision_state.clone()
            if initial_supervision_state is not None
            else PrefixTrajectorySupervisionState(
                num_slots=self.head.num_slots,
                mode=self.supervision_mode,
            )
        )
        if supervision.mode is not self.supervision_mode:
            raise ProtocolViolation("supervision state binding mode does not match the detector")
        valid_mask = tuple(bool(value) for value in masks[0].tolist())
        if any(valid_mask[index] and not valid_mask[index - 1] for index in range(1, len(valid_mask))):
            raise ProtocolViolation("stream masks must be right-padded without later valid tokens")
        valid_frames = tuple(
            frame for frame, is_valid in zip(source_frames, valid_mask) if is_valid
        )
        if valid_frames and valid_frames[0] <= runtime.last_decision_frame:
            raise ProtocolViolation("training episodes must be chronological and non-overlapping")

        previous_decision_frame = runtime.last_decision_frame
        for source_frame, schedule_step, is_valid in zip(
            source_frames,
            supervision_schedule,
            valid_mask,
        ):
            self._validate_schedule_step(
                schedule_step,
                source_frame,
                previous_decision_frame,
                masked=not is_valid,
            )
            if is_valid:
                previous_decision_frame = int(source_frame)

        state = self._to_head_state(runtime)
        sums = None
        logits = []
        birth_trace = []
        canonical_trace = []
        loss_binding_trace = []
        birth_mask_trace = []
        alive_mask_trace = []
        endpoint_trace = []
        exhaustion = 0
        rematch_swaps = 0
        valid_steps = 0
        feature_stride = meta.get("feature_stride", meta.get("snippet_stride", 1))
        for index, (source_frame, schedule_step) in enumerate(
            zip(source_frames, supervision_schedule)
        ):
            if not bool(masks[0, index].item()):
                continue
            available_slots = tuple(
                slot
                for slot, status in enumerate(state.slot_status.tolist())
                if int(status) == SLOT_FREE
            )
            outputs, state = self.head.step(features[:, :, index], state, source_frame)
            transition = supervision.transition(
                schedule_step,
                self._cost_provider(
                    outputs,
                    schedule_step,
                    feature_stride,
                ),
                available_slots=available_slots,
            )
            raw = self._step_losses(
                outputs,
                schedule_step,
                transition,
                feature_stride,
            )
            if sums is None:
                sums = {name: value for name, value in raw.items()}
            else:
                sums = {name: sums[name] + raw[name] for name in sums}
            _, state = self.head.decode_step(
                outputs,
                state,
                current_frame=source_frame,
                feature_stride=feature_stride,
            )
            logits.append(outputs)
            birth_trace.append(self._binding_rows(transition.birth_assignments))
            canonical_trace.append(self._binding_rows(transition.canonical_bindings))
            loss_binding_trace.append(self._binding_rows(transition.loss_bindings))
            birth_mask_trace.append(tuple(bool(value) for value in transition.birth_mask))
            alive_mask_trace.append(tuple(True for _ in range(self.head.num_slots)))
            endpoint_trace.append(tuple(int(value) for value in transition.endpoint_slots))
            exhaustion += int(transition.exhaustion)
            rematch_swaps = int(transition.audit.rematch_swap_total)
            valid_steps += 1
        if not valid_steps or sums is None:
            raise ProtocolViolation("training episode contains no valid supervised token")
        losses = {name: value / valid_steps for name, value in sums.items()}
        losses["cost"] = sum(
            losses[name] * self.loss_weights[name] for name in self.loss_weights
        )
        losses["_optimizer_weight"] = losses["cost"].new_tensor(float(valid_steps))
        runtime = self._from_head_state(
            state,
            runtime.ledger_rows,
            valid_frames[-1],
        )
        audit = {
            "binding_mode": self.trajectory_binding_mode,
            "birth_assignments": tuple(birth_trace),
            "canonical_lifecycle": tuple(canonical_trace),
            "loss_bindings": tuple(loss_binding_trace),
            "birth_mask_trace": tuple(birth_mask_trace),
            "alive_mask_trace": tuple(alive_mask_trace),
            "endpoint_slot_trace": tuple(endpoint_trace),
            "slot_exhaustion": exhaustion,
            "rematch_swap_count": rematch_swaps,
            "valid_supervised_steps": valid_steps,
            "max_source_frame": valid_frames[-1],
            "runtime_state_contains_gt": False,
        }
        self.last_episode_audit = audit
        return TrajectoryEpisodeLossOutput(
            losses=losses,
            runtime_state=runtime,
            supervision_state=supervision,
            logits=tuple(logits),
            audit=audit,
        )

    def _prepare_stream(self, inputs, masks, metas, stream_control, *, training):
        if inputs.shape[0] != 1 or len(metas) != 1 or len(stream_control) != 1:
            raise ProtocolViolation("persistent trajectory execution requires one stream lane")
        meta = self._validate_meta(metas[0])
        control = self._validate_control(stream_control[0])
        for field in ("video_id", "video_name", "stream_id"):
            if field in meta and field in control and meta[field] != control[field]:
                raise ProtocolViolation(
                    f"{field} conflicts across model and control planes"
                )
        key = _stream_key(meta)
        if training:
            if key in self._staged_terminal_keys:
                raise ProtocolViolation("terminal training state must be committed or rolled back")
            runtime = self._staged_runtime_states.get(key, self._runtime_states.get(key))
            supervision = self._staged_supervision_states.get(
                key, self._supervision_states.get(key)
            )
        else:
            if self.has_pending_online_update():
                raise ProtocolViolation("inference cannot run while training state is staged")
            runtime = self._runtime_states.get(key)
            supervision = self._supervision_states.get(key)
        if control.get("is_video_start", False):
            if runtime is not None or supervision is not None:
                raise ProtocolViolation("stream start received while state already exists")
        elif runtime is None:
            raise ProtocolViolation("non-start stream packet has no runtime state")
        return meta, control, key, runtime, supervision

    def forward(
        self,
        inputs,
        masks,
        metas,
        return_loss=True,
        prefix_schedule=None,
        stream_control=None,
        infer_cfg=None,
        post_cfg=None,
        **kwargs,
    ):
        del infer_cfg, post_cfg
        class_names = kwargs.pop("ext_cls", None)
        if kwargs:
            raise ProtocolViolation(
                "unexpected detector inputs may contain GT or terminal taint: "
                f"{sorted(kwargs)}"
            )
        if stream_control is None:
            raise ProtocolViolation("stream control is required outside the model plane")
        meta, control, key, runtime, supervision = self._prepare_stream(
            inputs,
            masks,
            metas,
            stream_control,
            training=return_loss,
        )
        if return_loss:
            if prefix_schedule is None or len(prefix_schedule) != 1:
                raise ProtocolViolation("training requires one separate prefix schedule")
            output = self.train_episode(
                inputs,
                masks,
                meta,
                prefix_schedule[0],
                initial_runtime_state=runtime,
                initial_supervision_state=supervision,
            )
            self._staged_runtime_states[key] = output.runtime_state
            self._staged_supervision_states[key] = output.supervision_state
            if control.get("is_video_end", False) or control.get("reset_stream", False):
                self._staged_terminal_keys.add(key)
            result = output.losses
        else:
            if prefix_schedule is not None:
                raise ProtocolViolation("inference rejects supervision schedules")
            if "training_targets" in control:
                raise ProtocolViolation("inference rejects training targets in the control plane")
            output = self.infer_step(
                inputs,
                masks,
                meta,
                runtime_state=runtime,
                class_names=class_names,
            )
            self._runtime_states[key] = output.runtime_state
            video_name = meta.get("video_name", meta.get("video_id", "unknown"))
            result = {video_name: [dict(row) for row in output.emissions]}
        if not return_loss and (
            control.get("is_video_end", False) or control.get("reset_stream", False)
        ):
            self._runtime_states.pop(key, None)
            self._supervision_states.pop(key, None)
        return result


__all__ = [
    "PersistentTrajectoryOnlineDetector",
    "PersistentTrajectoryRuntimeState",
    "TrajectoryEpisodeLossOutput",
    "TrajectoryInferenceOutput",
]
