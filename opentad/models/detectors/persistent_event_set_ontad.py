from dataclasses import replace
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ..builder import DETECTORS, build_head
from ..dense_heads.persistent_event_set_head import (
    SLOT_ACTIVE,
    SLOT_FREE,
    PersistentEventSetState,
)
from opentad.utils.online_protocol import ProtocolViolation


_FORBIDDEN_MODEL_META = {
    "duration",
    "total_frames",
    "num_frames",
    "is_video_end",
    "video_end_frame",
}


def _zero(reference):
    return reference.float().sum() * 0.0


def _masked_bce(logits, targets, mask):
    mask = mask.to(dtype=torch.bool)
    if not mask.any():
        return _zero(logits)
    return F.binary_cross_entropy_with_logits(logits[mask], targets[mask], reduction="mean")


def _make_stream_key(meta):
    return (
        f"video={meta.get('video_name', meta.get('video_id', 'unknown'))}|"
        f"stream={meta.get('stream_id', 'default')}|"
        f"stride={meta.get('feature_stride', 'unknown')}|"
        f"input_format={meta.get('input_format', 'cached_features')}"
    )


def _solve_linear_assignment(cost_matrix):
    """Return globally minimum row/column pairs for a rectangular cost matrix."""
    if cost_matrix.ndim != 2:
        raise ValueError("assignment cost matrix must be two-dimensional")
    if 0 in cost_matrix.shape:
        return ()
    row_indices, column_indices = linear_sum_assignment(
        cost_matrix.detach().float().cpu().numpy()
    )
    return tuple(
        (int(row), int(column))
        for row, column in sorted(zip(row_indices, column_indices))
    )


@DETECTORS.register_module()
class PersistentEventSetOnlineDetector(nn.Module):
    """Matched-feature detector for falsifying persistent On-TAD queries."""

    def __init__(
        self,
        head,
        assignment_mode="prefix",
        detach_stream_state=True,
        birth_loss_weight=1.0,
        alive_loss_weight=0.5,
        class_loss_weight=1.0,
        start_loss_weight=1.0,
        end_loss_weight=1.0,
        endpoint_offset_loss_weight=0.25,
    ):
        super().__init__()
        self.head = head if isinstance(head, nn.Module) else build_head(head)
        self.assignment_mode = str(assignment_mode)
        self.detach_stream_state = bool(detach_stream_state)
        if self.assignment_mode not in {"per_step", "prefix"}:
            raise ValueError("assignment_mode must be 'per_step' or 'prefix'")
        self.loss_weights = {
            "birth_loss": float(birth_loss_weight),
            "alive_loss": float(alive_loss_weight),
            "class_loss": float(class_loss_weight),
            "start_loss": float(start_loss_weight),
            "end_loss": float(end_loss_weight),
            "endpoint_offset_loss": float(endpoint_offset_loss_weight),
        }
        self.register_buffer(
            "audit_slot_exhaustion_total",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        self._stream_states = {}
        self.last_chunk_audit = {}

    def reset_online_states(self):
        self._stream_states.clear()
        self.last_chunk_audit = {}

    @staticmethod
    def _validate_meta(meta, inputs):
        if not isinstance(meta, dict):
            raise ProtocolViolation("event-set detector metadata must be a dictionary")
        leaked = sorted(_FORBIDDEN_MODEL_META.intersection(meta))
        if leaked:
            raise ProtocolViolation(f"terminal metadata leaked into model inputs: {leaked}")
        source_frames = tuple(int(frame) for frame in meta.get("source_frames", ()))
        if len(source_frames) != inputs.shape[-1]:
            raise ProtocolViolation("source_frames must align with the feature-token axis")
        if any(right <= left for left, right in zip(source_frames, source_frames[1:])):
            raise ProtocolViolation("source_frames must be strictly increasing")
        if source_frames and source_frames[-1] > int(meta.get("current_frame", source_frames[-1])):
            raise ProtocolViolation("feature source frame exceeds the current decision frame")
        return source_frames

    def _new_state(self, inputs, meta):
        return self.head.initial_state(
            device=inputs.device,
            dtype=inputs.dtype,
            stream_key=_make_stream_key(meta),
        )

    def scan_features(self, inputs, masks, meta, state=None):
        if inputs.ndim != 3 or inputs.shape[0] != 1:
            raise ProtocolViolation("event-set feature scan requires inputs [1,C,T]")
        if masks.shape != (1, inputs.shape[-1]):
            raise ProtocolViolation("event-set masks must align with [1,T]")
        source_frames = self._validate_meta(meta, inputs)
        state = self._new_state(inputs, meta) if state is None else state
        outputs = []
        for index, source_frame in enumerate(source_frames):
            if not bool(masks[0, index].item()):
                continue
            step_outputs, state = self.head.step(inputs[:, :, index], state, source_frame)
            outputs.append(step_outputs)
        return tuple(outputs), state

    @staticmethod
    def _visible_instances(schedule_step):
        visible = {item.instance_id: item for item in schedule_step.active}
        visible.update({item.instance_id: item for item in schedule_step.ends})
        return tuple(visible[key] for key in sorted(visible))

    def _start_cost(self, outputs, slot, target, current_frame, feature_stride):
        if self.head.start_mode == "pointer":
            pointer_target = self.head.pointer_target(outputs["memory_frames"], target.start_frame)
            log_probs = outputs["start_pointer_logits"][0, slot].log_softmax(dim=-1)
            return -log_probs[pointer_target]
        target_offset = (float(current_frame) - float(target.start_frame)) / max(float(feature_stride), 1.0)
        target_offset = min(max(target_offset, 0.0), float(self.head.memory_size))
        return (outputs["start_offset"][0, slot] - target_offset).abs()

    def _match(self, targets, candidate_slots, outputs, current_frame, feature_stride, use_birth):
        if not targets or not candidate_slots:
            return []
        class_log_probs = outputs["class_logits"][0].log_softmax(dim=-1)
        presence_logits = outputs["birth_logits" if use_birth else "alive_logits"][0]
        cost_rows = []
        for target in targets:
            row = []
            for slot in candidate_slots:
                cost = (
                    -F.logsigmoid(presence_logits[slot])
                    - class_log_probs[slot, int(target.label)]
                    + self._start_cost(outputs, slot, target, current_frame, feature_stride)
                )
                row.append(cost)
            cost_rows.append(torch.stack(row))
        cost_matrix = torch.stack(cost_rows)
        assignments = _solve_linear_assignment(cost_matrix)
        return [
            (targets[target_index], int(candidate_slots[slot_index]))
            for target_index, slot_index in assignments
        ]

    @staticmethod
    def _state_with_assignments(state, instance_to_slot, slot_to_instance):
        status = torch.full_like(state.slot_status, SLOT_FREE)
        for slot in slot_to_instance:
            status[int(slot)] = SLOT_ACTIVE
        return replace(
            state,
            slot_status=status,
            instance_to_slot=dict(instance_to_slot),
            slot_to_instance=dict(slot_to_instance),
        )

    def _assign(self, outputs, state, schedule_step, feature_stride):
        visible = self._visible_instances(schedule_step)
        if self.assignment_mode == "per_step":
            matches = self._match(
                visible,
                tuple(range(self.head.num_slots)),
                outputs,
                schedule_step.current_frame,
                feature_stride,
                use_birth=False,
            )
            instance_to_slot = {target.instance_id: slot for target, slot in matches}
            slot_to_instance = {slot: target.instance_id for target, slot in matches}
            return self._state_with_assignments(state, instance_to_slot, slot_to_instance), matches, len(visible) - len(matches)

        instance_to_slot = dict(state.instance_to_slot)
        slot_to_instance = dict(state.slot_to_instance)
        free_slots = tuple(slot for slot in range(self.head.num_slots) if slot not in slot_to_instance)
        births = tuple(item for item in schedule_step.births if item.instance_id not in instance_to_slot)
        birth_matches = self._match(
            births,
            free_slots,
            outputs,
            schedule_step.current_frame,
            feature_stride,
            use_birth=True,
        )
        for target, slot in birth_matches:
            instance_to_slot[target.instance_id] = slot
            slot_to_instance[slot] = target.instance_id
        state = self._state_with_assignments(state, instance_to_slot, slot_to_instance)
        visible_matches = [
            (target, instance_to_slot[target.instance_id])
            for target in visible
            if target.instance_id in instance_to_slot
        ]
        return state, visible_matches, len(births) - len(birth_matches)

    def _step_losses(self, outputs, state, schedule_step, feature_stride):
        state_before = state
        state, matches, exhaustion = self._assign(outputs, state, schedule_step, feature_stride)
        match_by_instance = {target.instance_id: (target, slot) for target, slot in matches}
        active_slots = tuple(sorted(slot for _, slot in matches))
        end_slots = tuple(
            match_by_instance[item.instance_id][1]
            for item in schedule_step.ends
            if item.instance_id in match_by_instance
        )

        birth_target = torch.zeros_like(outputs["birth_logits"])
        birth_mask = torch.zeros_like(birth_target, dtype=torch.bool)
        if self.assignment_mode == "per_step":
            birth_mask[:] = True
            for slot in active_slots:
                birth_target[:, slot] = 1.0
        else:
            previously_occupied = set(state_before.slot_to_instance)
            candidate_slots = tuple(slot for slot in range(self.head.num_slots) if slot not in previously_occupied)
            for slot in candidate_slots:
                birth_mask[:, slot] = True
            for item in schedule_step.births:
                slot = state.instance_to_slot.get(item.instance_id)
                if slot is not None and slot in candidate_slots:
                    birth_target[:, slot] = 1.0

        alive_target = torch.zeros_like(outputs["alive_logits"])
        for slot in active_slots:
            alive_target[:, slot] = 1.0
        alive_mask = torch.ones_like(alive_target, dtype=torch.bool)

        class_loss = _zero(outputs["class_logits"])
        start_loss = _zero(outputs["class_logits"])
        if matches:
            slots = torch.as_tensor([slot for _, slot in matches], device=outputs["class_logits"].device)
            labels = torch.as_tensor(
                [int(target.label) for target, _ in matches],
                dtype=torch.long,
                device=outputs["class_logits"].device,
            )
            class_loss = F.cross_entropy(outputs["class_logits"][0, slots], labels)
            if self.head.start_mode == "pointer":
                pointer_targets = torch.as_tensor(
                    [
                        self.head.pointer_target(outputs["memory_frames"], target.start_frame)
                        for target, _ in matches
                    ],
                    dtype=torch.long,
                    device=outputs["class_logits"].device,
                )
                start_loss = F.cross_entropy(outputs["start_pointer_logits"][0, slots], pointer_targets)
            else:
                target_offsets = torch.as_tensor(
                    [
                        min(
                            max(
                                (float(schedule_step.current_frame) - float(target.start_frame))
                                / max(float(feature_stride), 1.0),
                                0.0,
                            ),
                            float(self.head.memory_size),
                        )
                        for target, _ in matches
                    ],
                    dtype=outputs["start_offset"].dtype,
                    device=outputs["start_offset"].device,
                )
                start_loss = F.smooth_l1_loss(outputs["start_offset"][0, slots], target_offsets)

        if self.head.endpoint_mode == "hazard":
            end_target, end_mask = self.head.build_end_hazard_targets(
                outputs["end_hazard_logits"],
                at_risk_slots=active_slots,
                end_event_slots=end_slots,
            )
        else:
            end_target = torch.zeros_like(outputs["end_hazard_logits"])
            for slot in end_slots:
                end_target[:, slot] = 1.0
            end_mask = torch.ones_like(end_target, dtype=torch.bool)

        endpoint_offset_loss = _zero(outputs["endpoint_offset"])
        if end_slots:
            end_by_slot = {
                match_by_instance[item.instance_id][1]: item
                for item in schedule_step.ends
                if item.instance_id in match_by_instance
            }
            slots = torch.as_tensor(sorted(end_by_slot), device=outputs["endpoint_offset"].device)
            target_offsets = torch.as_tensor(
                [float(schedule_step.current_frame) - float(end_by_slot[int(slot)].end_frame) for slot in slots.tolist()],
                dtype=outputs["endpoint_offset"].dtype,
                device=outputs["endpoint_offset"].device,
            ).clamp(min=0.0, max=self.head.max_endpoint_offset)
            endpoint_offset_loss = F.smooth_l1_loss(outputs["endpoint_offset"][0, slots], target_offsets)

        raw = {
            "birth_loss": _masked_bce(outputs["birth_logits"], birth_target, birth_mask),
            "alive_loss": _masked_bce(outputs["alive_logits"], alive_target, alive_mask),
            "class_loss": class_loss,
            "start_loss": start_loss,
            "end_loss": _masked_bce(outputs["end_hazard_logits"], end_target, end_mask),
            "endpoint_offset_loss": endpoint_offset_loss,
        }

        if self.assignment_mode == "prefix" and schedule_step.ends:
            instance_to_slot = dict(state.instance_to_slot)
            slot_to_instance = dict(state.slot_to_instance)
            for item in schedule_step.ends:
                slot = instance_to_slot.pop(item.instance_id, None)
                if slot is not None:
                    slot_to_instance.pop(slot, None)
            state = self._state_with_assignments(state, instance_to_slot, slot_to_instance)
        return raw, state, exhaustion, len(end_slots), len(active_slots)

    @staticmethod
    def _detach_state(state):
        return replace(
            state,
            queries=state.queries.detach(),
            feature_memory=state.feature_memory.detach(),
            slot_status=state.slot_status.detach(),
            refractory=state.refractory.detach(),
            start_frames=state.start_frames.detach(),
            peak_class_scores=state.peak_class_scores.detach(),
            instance_to_slot=dict(state.instance_to_slot),
            slot_to_instance=dict(state.slot_to_instance),
            ledger=list(state.ledger),
        )

    def _prepare_forward(self, inputs, masks, metas, stream_control):
        if inputs.ndim != 3 or inputs.shape[0] != 1 or len(metas) != 1 or len(stream_control) != 1:
            raise ProtocolViolation("event-set detector currently requires one chronological stream lane")
        meta = dict(metas[0])
        control = dict(stream_control[0])
        source_frames = self._validate_meta(meta, inputs)
        stream_key = _make_stream_key(meta)
        state = self._stream_states.get(stream_key)
        if bool(control.get("is_video_start")):
            if state is not None:
                raise ProtocolViolation("video start received while stream state already exists")
            state = self._new_state(inputs, meta)
        elif state is None:
            raise ProtocolViolation("non-start feature chunk has no stream state")
        if state.memory_frames and source_frames and source_frames[0] <= state.memory_frames[-1]:
            raise ProtocolViolation("feature chunks must be chronological and non-overlapping")
        return meta, control, source_frames, stream_key, state

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
        ext_cls = kwargs.pop("ext_cls", None)
        if kwargs:
            raise ProtocolViolation(
                "unexpected detector inputs are forbidden because they may carry GT or terminal taint: "
                f"{sorted(kwargs)}"
            )
        if stream_control is None:
            raise ProtocolViolation("stream_control is required outside model metadata")
        meta, control, source_frames, stream_key, state = self._prepare_forward(
            inputs,
            masks,
            metas,
            stream_control,
        )
        valid_indices = [index for index in range(inputs.shape[-1]) if bool(masks[0, index].item())]
        if return_loss:
            if prefix_schedule is None or len(prefix_schedule) != 1:
                raise ProtocolViolation("training requires one prefix-observable schedule")
            schedule = tuple(prefix_schedule[0])
            if len(schedule) != len(valid_indices):
                raise ProtocolViolation("prefix schedule must align with valid feature tokens")
        elif prefix_schedule is not None:
            raise ProtocolViolation("inference must not receive prefix labels")

        loss_rows = []
        emitted = []
        endpoint_positive_count = 0
        end_event_frames = []
        slot_exhaustion = 0
        max_assigned_instances = 0
        same_class_concurrent_max = 0

        for schedule_index, input_index in enumerate(valid_indices):
            current_frame = source_frames[input_index]
            outputs, state = self.head.step(inputs[:, :, input_index], state, current_frame)
            if return_loss:
                schedule_step = schedule[schedule_index]
                if int(schedule_step.current_frame) != int(current_frame):
                    raise ProtocolViolation("prefix schedule timestamp does not match feature source frame")
                row, state, exhaustion, end_count, assigned_count = self._step_losses(
                    outputs,
                    state,
                    schedule_step,
                    feature_stride=int(meta.get("feature_stride", 1)),
                )
                loss_rows.append(row)
                slot_exhaustion += int(exhaustion)
                endpoint_positive_count += int(end_count)
                if end_count:
                    end_event_frames.append(int(current_frame))
                max_assigned_instances = max(max_assigned_instances, int(assigned_count))
                label_counts = {}
                for item in self._visible_instances(schedule_step):
                    label_counts[item.label] = label_counts.get(item.label, 0) + 1
                same_class_concurrent_max = max(
                    same_class_concurrent_max,
                    max(label_counts.values(), default=0),
                )
            else:
                step_emissions, state = self.head.decode_step(
                    outputs,
                    state,
                    current_frame,
                    feature_stride=int(meta.get("feature_stride", 1)),
                )
                emitted.extend(step_emissions)

        stored_state = self._detach_state(state) if self.detach_stream_state else state
        if bool(control.get("is_video_end")):
            self._stream_states.pop(stream_key, None)
        else:
            self._stream_states[stream_key] = stored_state
        self.last_chunk_audit = {
            "model_meta_keys": sorted(meta),
            "source_frames": list(source_frames),
            "endpoint_positive_count": endpoint_positive_count,
            "end_event_frames": end_event_frames,
            "slot_exhaustion": slot_exhaustion,
            "max_assigned_instances": max_assigned_instances,
            "same_class_concurrent_max": same_class_concurrent_max,
            "emission_count": len(emitted),
        }
        if return_loss and slot_exhaustion:
            self.audit_slot_exhaustion_total.add_(int(slot_exhaustion))

        if return_loss:
            if not loss_rows:
                raise ProtocolViolation("training chunk produced no valid prefix loss")
            losses = {
                key: torch.stack([row[key] for row in loss_rows]).mean() * self.loss_weights[key]
                for key in self.loss_weights
            }
            losses["cost"] = sum(losses.values())
            return losses

        fps = float(meta.get("fps", 1.0))
        video_id = str(meta.get("video_id", meta.get("video_name")))
        rows = []
        for record in emitted:
            label = record.label
            if isinstance(ext_cls, (list, tuple)) and 0 <= label < len(ext_cls):
                label = ext_cls[label]
            elif isinstance(ext_cls, dict):
                label = ext_cls.get(label, ext_cls.get(str(label), label))
            predicted_end_latency = (record.emit_frame - record.end_frame) / fps
            rows.append(
                {
                    "segment": [record.start_frame / fps, record.end_frame / fps],
                    "label": label,
                    "score": record.score,
                    "slot_id": record.slot_id,
                    "start_frame": record.start_frame,
                    "end_frame": record.end_frame,
                    "emit_frame": record.emit_frame,
                    "emit_time_sec": record.emit_frame / fps,
                    "source_frame": record.max_source_frame,
                    "source_time_sec": record.max_source_frame / fps,
                    "fps": fps,
                    "predicted_end_latency_sec": predicted_end_latency,
                    "latency_sec": predicted_end_latency,
                    "latency_definition": "emit_time_minus_predicted_end_time",
                    "stream_key": stream_key,
                    "immutable": True,
                }
            )
        return {video_id: rows}


__all__ = ["PersistentEventSetOnlineDetector"]
