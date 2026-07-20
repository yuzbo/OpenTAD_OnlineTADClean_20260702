from dataclasses import dataclass, replace
import inspect
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import DETECTORS, build_backbone, build_head, build_projection
from ..dense_heads.persistent_event_set_head import SLOT_FREE, PersistentEventSetState
from opentad.utils.online_protocol import ProtocolViolation
from opentad.utils.prefix_trajectory_supervision import (
    PrefixTrajectorySupervisionState,
    SupervisionMode,
)


_BINDING_MODES = {
    "fixed": SupervisionMode.FIXED,
    "fixed_birth_slot": SupervisionMode.FIXED,
    "rematch": SupervisionMode.REMATCH,
    "prefix_rematch_active_pool": SupervisionMode.REMATCH,
}

_FORBIDDEN_MODEL_META = {
    "annotation",
    "annotations",
    "duration",
    "end_frame",
    "eof",
    "future_frames",
    "ground_truth",
    "gt_segments",
    "is_video_end",
    "labels",
    "num_frames",
    "prefix_schedule",
    "segments",
    "total_frames",
    "video_end_frame",
}

_STREAM_CONTROL_FIELDS = {
    "is_video_start",
    "reset_stream",
    "stream_id",
    "video_id",
    "video_name",
}


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


def _forbidden_meta_paths(value, prefix=""):
    found = []
    if isinstance(value, dict):
        for key, nested in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if key_text.lower() in _FORBIDDEN_MODEL_META:
                found.append(path)
            found.extend(_forbidden_meta_paths(nested, path))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            found.extend(_forbidden_meta_paths(nested, f"{prefix}[{index}]"))
    return tuple(found)


@dataclass
class PersistentTrajectoryRuntimeState:
    """Inference-visible state. Training identities are intentionally absent."""

    stream_key: str
    queries: torch.Tensor
    feature_memory: torch.Tensor
    source_frames: Tuple[int, ...]
    slot_status: torch.Tensor
    candidate_age: torch.Tensor
    start_state: torch.Tensor
    score_state: torch.Tensor
    label_state: torch.Tensor
    committed: Tuple[dict, ...]
    birth_proposals: int
    birth_admissions: int
    arbitration_suppressions: int
    candidate_cancellations: int
    active_abandonments: int
    deferred_birth_due_to_release: int
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
        fail_on_supervision_exhaustion=False,
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
            raise ValueError("the matched binding route requires persistent query recurrence")
        if self.head.start_mode != "scalar":
            raise ValueError("the matched binding route requires scalar start supervision")
        if self.head.endpoint_mode != "binary":
            raise ValueError("the matched binding route requires binary endpoint supervision")
        if self.head.lifecycle_mode != "candidate_recycle":
            raise ValueError("persistent trajectory detection requires candidate_recycle")
        if self.head.refractory_steps != 0:
            raise ValueError("persistent trajectory detection forbids refractory occupancy")
        self.detach_stream_state = bool(detach_stream_state)
        self.fail_on_supervision_exhaustion = bool(
            fail_on_supervision_exhaustion
        )
        self.loss_weights = {
            "birth_loss": float(birth_loss_weight),
            "alive_loss": float(alive_loss_weight),
            "class_loss": float(class_loss_weight),
            "start_loss": float(start_loss_weight),
            "end_loss": float(end_loss_weight),
        }
        self._runtime_states = {}
        self._supervision_states = {}
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
        self.last_episode_audit = {}

    def _validate_meta(self, model_meta):
        if not isinstance(model_meta, dict):
            raise ProtocolViolation("model metadata must be a dictionary")
        meta = dict(model_meta)
        forbidden = sorted(_forbidden_meta_paths(meta))
        if forbidden:
            raise ProtocolViolation(
                f"model metadata contains forbidden terminal or target fields: {forbidden}"
            )
        if "current_frame" not in meta:
            raise ProtocolViolation("model metadata requires current_frame")
        try:
            meta["current_frame"] = int(meta["current_frame"])
        except (TypeError, ValueError) as exc:
            raise ProtocolViolation("current_frame must be an integer") from exc
        if meta["current_frame"] < 0:
            raise ProtocolViolation("current_frame must be non-negative")
        return meta

    @staticmethod
    def _validate_control(stream_control):
        if not isinstance(stream_control, dict):
            raise ProtocolViolation("stream control must be a dictionary")
        unknown = sorted(set(stream_control).difference(_STREAM_CONTROL_FIELDS))
        if unknown:
            raise ProtocolViolation(f"unknown stream control fields: {unknown}")
        control = dict(stream_control)
        for field in ("is_video_start", "reset_stream"):
            if field in control and not isinstance(control[field], bool):
                raise ProtocolViolation(f"{field} must be a boolean")
        return control

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
            candidate_age=state.candidate_age,
            start_state=state.start_frames,
            score_state=state.peak_class_scores,
            label_state=state.peak_class_labels,
            committed=(),
            birth_proposals=0,
            birth_admissions=0,
            arbitration_suppressions=0,
            candidate_cancellations=0,
            active_abandonments=0,
            deferred_birth_due_to_release=0,
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
            refractory=torch.zeros_like(state.slot_status),
            start_frames=state.start_state,
            peak_class_scores=state.score_state,
            ledger=[],
            instance_to_slot={},
            slot_to_instance={},
            candidate_age=state.candidate_age,
            peak_class_labels=state.label_state,
            birth_proposals=state.birth_proposals,
            birth_admissions=state.birth_admissions,
            arbitration_suppressions=state.arbitration_suppressions,
            candidate_cancellations=state.candidate_cancellations,
            active_abandonments=state.active_abandonments,
            deferred_birth_due_to_release=state.deferred_birth_due_to_release,
        )

    def _from_head_state(self, state, committed, decision_frame):
        runtime = PersistentTrajectoryRuntimeState(
            stream_key=state.stream_key,
            queries=state.queries,
            feature_memory=state.feature_memory,
            source_frames=tuple(int(frame) for frame in state.memory_frames),
            slot_status=state.slot_status,
            candidate_age=state.candidate_age,
            start_state=state.start_frames,
            score_state=state.peak_class_scores,
            label_state=state.peak_class_labels,
            committed=tuple(dict(row) for row in committed),
            birth_proposals=int(state.birth_proposals),
            birth_admissions=int(state.birth_admissions),
            arbitration_suppressions=int(state.arbitration_suppressions),
            candidate_cancellations=int(state.candidate_cancellations),
            active_abandonments=int(state.active_abandonments),
            deferred_birth_due_to_release=int(
                state.deferred_birth_due_to_release
            ),
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
            candidate_age=state.candidate_age.detach(),
            start_state=state.start_state.detach(),
            score_state=state.score_state.detach(),
            label_state=state.label_state.detach(),
            committed=tuple(dict(row) for row in state.committed),
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
            raise ProtocolViolation("model metadata requires source_frames")
        source_frames = tuple(int(frame) for frame in source_frames)
        if len(source_frames) != features.shape[-1]:
            raise ProtocolViolation("source_frames must align with encoded tokens")
        if any(right <= left for left, right in zip(source_frames, source_frames[1:])):
            raise ProtocolViolation("source_frames must be strictly increasing")
        if source_frames and source_frames[-1] > int(meta["current_frame"]):
            raise ProtocolViolation("encoded source frame exceeds the decision frame")
        return features, masks.to(dtype=torch.bool), source_frames

    @staticmethod
    def _targets_by_id(schedule_step):
        targets = {}
        for item in tuple(schedule_step.births) + tuple(schedule_step.active) + tuple(schedule_step.ends):
            targets[int(item.instance_id)] = item
        return targets

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
        video_id,
        class_names=None,
        fps=30.0,
    ):
        fps = float(fps)
        if not math.isfinite(fps) or fps <= 0:
            raise ProtocolViolation("fps must be a positive finite number")
        committed = [dict(row) for row in existing_rows]
        new_rows = []
        for record in records:
            label = int(record.label)
            if class_names is not None:
                if not isinstance(class_names, (list, tuple)):
                    raise ProtocolViolation("class_names must be a list or tuple")
                label = class_names[label]
            event_id = (
                f"{record.stream_key}:slot={record.slot_id}:emit={record.emit_frame}:"
                f"sequence={len(committed)}"
            )
            row = {
                "event_id": event_id,
                "stream_id": record.stream_key,
                "stream_key": record.stream_key,
                "runtime_stream_key": record.stream_key,
                "video_id": str(video_id),
                "sequence_id": len(committed),
                "final": True,
                "immutable": True,
                "slot_id": int(record.slot_id),
                "label": label,
                "score": float(record.score),
                "segment": [
                    float(record.start_frame) / fps,
                    float(record.end_frame) / fps,
                ],
                "fps": fps,
                "start_frame": int(record.start_frame),
                "end_frame": int(record.end_frame),
                "emit_frame": int(record.emit_frame),
                "source_frame": int(record.max_source_frame),
                "latency_frames": int(record.emit_frame - record.end_frame),
                "latency_sec": float(record.emit_frame - record.end_frame) / fps,
                "latency_definition": "emit_time_minus_predicted_end_time",
            }
            if not (
                row["start_frame"]
                <= row["end_frame"]
                <= row["source_frame"]
                <= row["emit_frame"]
            ):
                raise ProtocolViolation(
                    "final interval must be causal: start <= end <= source <= emit"
                )
            if any(existing["event_id"] == event_id for existing in committed):
                raise ProtocolViolation(f"duplicate final event id: {event_id}")
            committed.append(row)
            new_rows.append(row)
        return tuple(committed), tuple(new_rows)

    def _scan_and_decode(
        self,
        features,
        masks,
        source_frames,
        runtime_state,
        feature_stride,
        video_id,
        class_names=None,
        fps=30.0,
    ):
        state = self._to_head_state(runtime_state)
        logits = []
        emissions = []
        committed = runtime_state.committed
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
            committed, rows = self._append_emissions(
                committed,
                records,
                video_id=video_id,
                class_names=class_names,
                fps=fps,
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
        runtime = self._from_head_state(state, committed, last_decision)
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
        if source_frames and source_frames[0] <= runtime_state.last_decision_frame:
            raise ProtocolViolation("stream tokens must be chronological and non-overlapping")
        logits, emissions, provisional, runtime = self._scan_and_decode(
            features,
            masks,
            source_frames,
            runtime_state,
            feature_stride=meta.get("feature_stride", meta.get("snippet_stride", 1)),
            video_id=meta.get("video_id", meta.get("video_name")),
            class_names=class_names,
            fps=meta.get("fps", 30.0),
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
            initial_supervision_state
            or PrefixTrajectorySupervisionState(
                num_slots=self.head.num_slots,
                mode=self.supervision_mode,
            )
        ).clone()
        if supervision.mode is not self.supervision_mode:
            raise ProtocolViolation("supervision state binding mode does not match the detector")
        if source_frames and source_frames[0] <= runtime.last_decision_frame:
            raise ProtocolViolation("training episodes must be chronological and non-overlapping")

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
        runtime_entry_free_collisions = 0
        rematch_swaps = 0
        valid_steps = 0
        feature_stride = meta.get("feature_stride", meta.get("snippet_stride", 1))
        for index, (source_frame, schedule_step) in enumerate(
            zip(source_frames, supervision_schedule)
        ):
            if int(schedule_step.current_frame) != int(source_frame):
                raise ProtocolViolation("schedule decision frame does not match source frame")
            if not bool(masks[0, index].item()):
                continue
            outputs, state = self.head.step(features[:, :, index], state, source_frame)
            free_runtime_slots = int(state.slot_status.eq(SLOT_FREE).sum().item())
            runtime_entry_free_collisions += max(
                0,
                len(tuple(schedule_step.births)) - free_runtime_slots,
            )
            transition = supervision.transition(
                schedule_step,
                self._cost_provider(
                    outputs,
                    schedule_step,
                    feature_stride,
                ),
            )
            if self.fail_on_supervision_exhaustion and transition.exhaustion:
                raise ProtocolViolation(
                    "formal training forbids supervision exhaustion at "
                    f"frame {source_frame}: "
                    f"{transition.exhausted_instance_ids}"
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
        runtime = self._from_head_state(
            state,
            runtime.committed,
            source_frames[-1],
        )
        runtime_deltas = {
            "birth_proposals": runtime.birth_proposals - initial_runtime_state.birth_proposals
            if initial_runtime_state is not None
            else runtime.birth_proposals,
            "birth_admissions": runtime.birth_admissions - initial_runtime_state.birth_admissions
            if initial_runtime_state is not None
            else runtime.birth_admissions,
            "candidate_arbitration_suppressions": (
                runtime.arbitration_suppressions
                - initial_runtime_state.arbitration_suppressions
                if initial_runtime_state is not None
                else runtime.arbitration_suppressions
            ),
            "candidate_cancellations": (
                runtime.candidate_cancellations
                - initial_runtime_state.candidate_cancellations
                if initial_runtime_state is not None
                else runtime.candidate_cancellations
            ),
            "active_abandonments": (
                runtime.active_abandonments
                - initial_runtime_state.active_abandonments
                if initial_runtime_state is not None
                else runtime.active_abandonments
            ),
            "deferred_birth_due_to_release": (
                runtime.deferred_birth_due_to_release
                - initial_runtime_state.deferred_birth_due_to_release
                if initial_runtime_state is not None
                else runtime.deferred_birth_due_to_release
            ),
        }
        audit = {
            "binding_mode": self.trajectory_binding_mode,
            "birth_assignments": tuple(birth_trace),
            "canonical_lifecycle": tuple(canonical_trace),
            "loss_bindings": tuple(loss_binding_trace),
            "birth_mask_trace": tuple(birth_mask_trace),
            "alive_mask_trace": tuple(alive_mask_trace),
            "endpoint_slot_trace": tuple(endpoint_trace),
            "gt_supervision_exhaustions": exhaustion,
            "gt_birth_runtime_entry_free_collisions": runtime_entry_free_collisions,
            "candidate_arbitration_suppressions": runtime_deltas[
                "candidate_arbitration_suppressions"
            ],
            "active_abandonments": runtime_deltas["active_abandonments"],
            "deferred_birth_due_to_release": runtime_deltas[
                "deferred_birth_due_to_release"
            ],
            "slot_exhaustion": exhaustion,
            "dropped_gt_birth_targets": exhaustion,
            "runtime_capacity_exhaustions": runtime_entry_free_collisions,
            "rematch_swap_count": rematch_swaps,
            "valid_supervised_steps": valid_steps,
            "max_source_frame": max(source_frames),
            "runtime_state_contains_gt": False,
            "birth_proposals": runtime_deltas["birth_proposals"],
            "birth_admissions": runtime_deltas["birth_admissions"],
            "arbitration_suppressions": runtime_deltas[
                "candidate_arbitration_suppressions"
            ],
            "candidate_cancellations": runtime_deltas[
                "candidate_cancellations"
            ],
        }
        self.last_episode_audit = audit
        return TrajectoryEpisodeLossOutput(
            losses=losses,
            runtime_state=runtime,
            supervision_state=supervision,
            logits=tuple(logits),
            audit=audit,
        )

    def _prepare_stream(self, inputs, masks, metas, stream_control):
        if inputs.shape[0] != 1 or len(metas) != 1 or len(stream_control) != 1:
            raise ProtocolViolation("persistent trajectory execution requires one stream lane")
        meta = self._validate_meta(metas[0])
        control = self._validate_control(stream_control[0])
        for field in ("video_id", "video_name", "stream_id"):
            if field in meta and field in control and meta[field] != control[field]:
                raise ProtocolViolation(f"{field} conflicts across metadata and stream control")
        key = _stream_key(meta)
        runtime = self._runtime_states.get(key)
        supervision = self._supervision_states.get(key)
        if control.get("is_video_start", False):
            self._runtime_states.clear()
            self._supervision_states.clear()
            runtime = None
            supervision = None
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
            self._runtime_states[key] = output.runtime_state
            self._supervision_states[key] = output.supervision_state
            result = output.losses
        else:
            if prefix_schedule is not None:
                raise ProtocolViolation("inference rejects supervision schedules")
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
        if control.get("reset_stream", False):
            self._runtime_states.pop(key, None)
            self._supervision_states.pop(key, None)
        return result


__all__ = [
    "PersistentTrajectoryOnlineDetector",
    "PersistentTrajectoryRuntimeState",
    "TrajectoryEpisodeLossOutput",
    "TrajectoryInferenceOutput",
]
