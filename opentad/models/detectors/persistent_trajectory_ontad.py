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


def _masked_bce(logits, target, mask, positive_weight=1.0):
    mask = mask.to(device=logits.device, dtype=torch.bool)
    if not mask.any():
        return _zero(logits)
    pos_weight = logits.new_tensor(float(positive_weight))
    return F.binary_cross_entropy_with_logits(
        logits[mask],
        target[mask],
        pos_weight=pos_weight,
    )


def _balanced_binary_logit_margin(logits, target, mask, margin):
    """Balanced positive/negative margin on decisions that contain a positive."""

    mask = mask.to(device=logits.device, dtype=torch.bool)
    positive = mask & target.gt(0.5)
    if not positive.any():
        return _zero(logits)
    negative = mask & ~target.gt(0.5)
    margin = logits.new_tensor(float(margin))
    positive_loss = F.relu(margin - logits[positive]).square().mean()
    if not negative.any():
        return positive_loss
    negative_loss = F.relu(margin + logits[negative]).square().mean()
    return 0.5 * (positive_loss + negative_loss)


def _balanced_binary_calibration_bce(logits, target, mask):
    """Equal positive/negative BCE on current decisions that contain a positive."""

    mask = mask.to(device=logits.device, dtype=torch.bool)
    positive = mask & target.gt(0.5)
    if not positive.any():
        return _zero(logits)
    negative = mask & ~target.gt(0.5)
    positive_loss = F.binary_cross_entropy_with_logits(
        logits[positive],
        torch.ones_like(logits[positive]),
    )
    if not negative.any():
        return positive_loss
    negative_loss = F.binary_cross_entropy_with_logits(
        logits[negative],
        torch.zeros_like(logits[negative]),
    )
    return 0.5 * (positive_loss + negative_loss)


def _sinkhorn_plan(cost, source_mass, target_mass, temperature, iterations):
    """Return a balanced entropic transport plan in log space."""

    if cost.ndim not in (2, 3) or cost.shape[-2] != cost.shape[-1]:
        raise ValueError("causal transport cost must be square or batched square")
    expected_mass_shape = cost.shape[:-1]
    if source_mass.shape != expected_mass_shape:
        raise ValueError("source transport mass does not align with cost")
    if target_mass.shape != expected_mass_shape:
        raise ValueError("target transport mass does not align with cost")
    log_kernel = -cost / float(temperature)
    log_source = source_mass.log()
    log_target = target_mass.log()
    log_u = torch.zeros_like(log_source)
    log_v = torch.zeros_like(log_target)
    for _ in range(int(iterations)):
        log_u = log_source - torch.logsumexp(
            log_kernel + log_v.unsqueeze(-2),
            dim=-1,
        )
        log_v = log_target - torch.logsumexp(
            log_kernel + log_u.unsqueeze(-1),
            dim=-2,
        )
    return torch.exp(
        log_kernel + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
    )


def _causal_query_transport_loss(
    previous_queries,
    current_queries,
    previous_lifecycle_mass,
    current_lifecycle_mass,
    *,
    temperature,
    identity_cost,
    iterations,
    mass_floor,
):
    """Align current persistent queries to the immediately preceding queries."""

    if previous_queries.shape != current_queries.shape:
        raise ValueError("causal transport query shapes must match")
    if previous_queries.ndim != 3 or previous_queries.shape[0] < 1:
        raise ValueError("causal transport queries must have shape [N,S,D]")
    batch_size, num_slots, _ = previous_queries.shape
    expected_mass_shape = (batch_size, num_slots)
    if previous_lifecycle_mass.shape != expected_mass_shape:
        raise ValueError("previous lifecycle mass does not align with queries")
    if current_lifecycle_mass.shape != expected_mass_shape:
        raise ValueError("current lifecycle mass does not align with queries")

    # The past is a fixed source for this one-way causal alignment. Gradients
    # optimize the current representation only.
    previous = F.normalize(previous_queries.detach().float(), dim=-1)
    current = F.normalize(current_queries.float(), dim=-1)
    cosine_cost = (
        1.0 - torch.matmul(previous, current.transpose(-2, -1))
    ).clamp_min(0.0)
    with torch.no_grad():
        source_mass = previous_lifecycle_mass.float().clamp_min(
            float(mass_floor)
        )
        target_mass = current_lifecycle_mass.float().clamp_min(
            float(mass_floor)
        )
        source_mass = source_mass / source_mass.sum(dim=-1, keepdim=True)
        target_mass = target_mass / target_mass.sum(dim=-1, keepdim=True)
        off_diagonal = 1.0 - torch.eye(
            num_slots,
            device=cosine_cost.device,
            dtype=cosine_cost.dtype,
        )
        plan = _sinkhorn_plan(
            cosine_cost.detach() + float(identity_cost) * off_diagonal,
            source_mass,
            target_mass,
            temperature,
            iterations,
        ).detach()
    return (plan * cosine_cost).sum()


def _raw_lifecycle_logits(outputs, key):
    raw_key = {
        "birth_logits": "raw_birth_logits",
        "alive_logits": "raw_alive_logits",
        "end_hazard_logits": "raw_end_hazard_logits",
    }[key]
    return outputs.get(raw_key, outputs[key])


def _predicted_lifecycle_mass(outputs):
    """Prediction-only persistence mass; no supervision or future target enters."""

    return (
        _raw_lifecycle_logits(outputs, "alive_logits").sigmoid()
        * (1.0 - _raw_lifecycle_logits(outputs, "birth_logits").sigmoid())
        * (
            1.0
            - _raw_lifecycle_logits(outputs, "end_hazard_logits").sigmoid()
        )
    )


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
        birth_positive_weight=1.0,
        alive_positive_weight=1.0,
        end_positive_weight=1.0,
        prior_bias_mode="raw_probability",
        fail_on_supervision_exhaustion=False,
        birth_logit_margin_loss_weight=0.0,
        birth_logit_margin=0.25,
        alive_logit_margin_loss_weight=0.0,
        alive_logit_margin=0.25,
        end_logit_margin_loss_weight=0.0,
        end_logit_margin=0.25,
        causal_query_transport_loss_weight=0.0,
        causal_query_transport_temperature=0.25,
        causal_query_transport_identity_cost=0.25,
        causal_query_transport_iterations=56,
        causal_query_transport_mass_floor=0.05,
        birth_calibration_loss_weight=0.0,
        alive_calibration_loss_weight=0.0,
        end_calibration_loss_weight=0.0,
        lifecycle_calibration_aggregation="step_positive_mean",
        endpoint_start_pointer_loss_weight=0.0,
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
            "birth_margin_loss": float(birth_logit_margin_loss_weight),
            "alive_margin_loss": float(alive_logit_margin_loss_weight),
            "end_margin_loss": float(end_logit_margin_loss_weight),
            "causal_transport_loss": float(
                causal_query_transport_loss_weight
            ),
            "birth_calibration_loss": float(
                birth_calibration_loss_weight
            ),
            "alive_calibration_loss": float(
                alive_calibration_loss_weight
            ),
            "end_calibration_loss": float(end_calibration_loss_weight),
            "endpoint_start_pointer_loss": float(
                endpoint_start_pointer_loss_weight
            ),
        }
        if any(
            not math.isfinite(value) or value < 0
            for value in self.loss_weights.values()
        ):
            raise ValueError("loss weights must be non-negative and finite")
        calibration_weights = (
            self.loss_weights["birth_calibration_loss"],
            self.loss_weights["alive_calibration_loss"],
            self.loss_weights["end_calibration_loss"],
        )
        if (
            any(weight > 0 for weight in calibration_weights)
            and self.head.lifecycle_calibration_mode != "monotone_affine"
        ):
            raise ValueError(
                "calibration losses require monotone_affine lifecycle calibration"
            )
        self.lifecycle_calibration_aggregation = str(
            lifecycle_calibration_aggregation
        )
        if self.lifecycle_calibration_aggregation not in {
            "step_positive_mean",
            "episode_balanced",
        }:
            raise ValueError(
                "lifecycle_calibration_aggregation must be "
                "step_positive_mean or episode_balanced"
            )
        if (
            self.loss_weights["endpoint_start_pointer_loss"] > 0
            and self.head.endpoint_start_mode != "past_pointer"
        ):
            raise ValueError(
                "endpoint start pointer loss requires past_pointer mode"
            )
        self.logit_margins = {
            "birth": float(birth_logit_margin),
            "alive": float(alive_logit_margin),
            "end": float(end_logit_margin),
        }
        for channel, margin in self.logit_margins.items():
            if not math.isfinite(margin) or margin <= 0:
                raise ValueError(
                    f"{channel}_logit_margin must be positive and finite"
                )
        self.birth_logit_margin = self.logit_margins["birth"]
        self.alive_logit_margin = self.logit_margins["alive"]
        self.end_logit_margin = self.logit_margins["end"]
        self.causal_transport = {
            "temperature": float(causal_query_transport_temperature),
            "identity_cost": float(causal_query_transport_identity_cost),
            "iterations": int(causal_query_transport_iterations),
            "mass_floor": float(causal_query_transport_mass_floor),
        }
        if (
            not math.isfinite(self.causal_transport["temperature"])
            or self.causal_transport["temperature"] <= 0
        ):
            raise ValueError(
                "causal_query_transport_temperature must be positive and finite"
            )
        if (
            not math.isfinite(self.causal_transport["identity_cost"])
            or self.causal_transport["identity_cost"] < 0
        ):
            raise ValueError(
                "causal_query_transport_identity_cost must be non-negative and finite"
            )
        if self.causal_transport["iterations"] <= 0:
            raise ValueError(
                "causal_query_transport_iterations must be positive"
            )
        if (
            not math.isfinite(self.causal_transport["mass_floor"])
            or not 0 < self.causal_transport["mass_floor"] <= 1
        ):
            raise ValueError(
                "causal_query_transport_mass_floor must be in (0, 1]"
            )
        self.positive_weights = {
            "birth_loss": float(birth_positive_weight),
            "alive_loss": float(alive_positive_weight),
            "end_loss": float(end_positive_weight),
        }
        if any(
            not math.isfinite(value) or value <= 0
            for value in self.positive_weights.values()
        ):
            raise ValueError(
                "positive supervision weights must be positive and finite"
            )
        self.prior_bias_mode = str(prior_bias_mode)
        if self.prior_bias_mode not in {
            "raw_probability",
            "weighted_bce_stationary",
        }:
            raise ValueError(
                "prior_bias_mode must be raw_probability or "
                "weighted_bce_stationary"
            )
        if self.prior_bias_mode == "weighted_bce_stationary":
            self.head.initialize_prior_biases(
                birth_positive_weight=self.positive_weights["birth_loss"],
                alive_positive_weight=self.positive_weights["alive_loss"],
                end_positive_weight=self.positive_weights["end_loss"],
            )
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
        raw_birth_logits = _raw_lifecycle_logits(outputs, "birth_logits")
        raw_end_logits = _raw_lifecycle_logits(
            outputs,
            "end_hazard_logits",
        )
        ending_instance_ids = {
            int(item.instance_id) for item in schedule_step.ends
        }

        def provider(phase, instance_ids, slot_ids):
            rows = []
            for instance_id in instance_ids:
                target = targets[int(instance_id)]
                target_offset = (
                    float(schedule_step.current_frame) - float(target.start_frame)
                ) / max(float(feature_stride), 1.0)
                target_offset = min(
                    max(target_offset, 0.0),
                    float(self.head.max_start_offset),
                )
                row = []
                for slot in slot_ids:
                    class_cost = -class_log_probs[int(slot), int(target.label)]
                    endpoint_target = raw_end_logits.new_tensor(
                        float(int(instance_id) in ending_instance_ids)
                    )
                    endpoint_cost = F.binary_cross_entropy_with_logits(
                        raw_end_logits[0, int(slot)],
                        endpoint_target,
                        reduction="sum",
                    )
                    if phase == "birth":
                        start_cost = F.smooth_l1_loss(
                            outputs["start_offset"][0, int(slot)],
                            outputs["start_offset"].new_tensor(
                                target_offset
                            ),
                            reduction="sum",
                        )
                        cost = (
                            class_cost
                            + endpoint_cost
                            + start_cost
                            - F.logsigmoid(
                                raw_birth_logits[0, int(slot)]
                            )
                        )
                    elif phase == "rematch":
                        # Start is decoded and supervised only at birth. Using
                        # it again here would let an unsupervised nuisance
                        # prediction decide post-birth identity binding.
                        cost = class_cost + endpoint_cost
                    else:
                        raise ValueError(f"unknown supervision assignment phase {phase!r}")
                    row.append(cost.detach())
                rows.append(torch.stack(row))
            if not rows:
                return torch.empty((0, len(slot_ids))).numpy()
            return torch.stack(rows).float().cpu().numpy()

        return provider

    @staticmethod
    def _lifecycle_binary_targets(outputs, transition):
        raw_birth_logits = _raw_lifecycle_logits(outputs, "birth_logits")
        raw_alive_logits = _raw_lifecycle_logits(outputs, "alive_logits")
        raw_end_logits = _raw_lifecycle_logits(
            outputs,
            "end_hazard_logits",
        )
        birth_target = torch.zeros_like(raw_birth_logits)
        birth_mask = torch.as_tensor(
            transition.birth_mask,
            dtype=torch.bool,
            device=birth_target.device,
        ).unsqueeze(0)
        for binding in transition.birth_assignments:
            birth_target[:, int(binding.slot_id)] = 1.0

        alive_target = torch.zeros_like(raw_alive_logits)
        for slot in transition.audit.occupied_slots_for_supervision:
            alive_target[:, int(slot)] = 1.0
        alive_mask = torch.ones_like(alive_target, dtype=torch.bool)

        end_target = torch.zeros_like(raw_end_logits)
        end_mask = torch.as_tensor(
            transition.at_risk_mask,
            dtype=torch.bool,
            device=end_target.device,
        ).unsqueeze(0)
        for slot in transition.endpoint_slots:
            end_target[:, int(slot)] = 1.0

        return {
            "birth": (birth_target, birth_mask),
            "alive": (alive_target, alive_mask),
            "end": (end_target, end_mask),
        }

    def _step_losses(
        self,
        outputs,
        schedule_step,
        transition,
        feature_stride,
        lifecycle_targets=None,
    ):
        raw_birth_logits = _raw_lifecycle_logits(outputs, "birth_logits")
        raw_alive_logits = _raw_lifecycle_logits(outputs, "alive_logits")
        raw_end_logits = _raw_lifecycle_logits(
            outputs,
            "end_hazard_logits",
        )
        if lifecycle_targets is None:
            lifecycle_targets = self._lifecycle_binary_targets(
                outputs,
                transition,
            )
        birth_target, birth_mask = lifecycle_targets["birth"]
        alive_target, alive_mask = lifecycle_targets["alive"]
        end_target, end_mask = lifecycle_targets["end"]

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
        if transition.birth_assignments:
            birth_slots = torch.as_tensor(
                [
                    int(binding.slot_id)
                    for binding in transition.birth_assignments
                ],
                dtype=torch.long,
                device=outputs["start_offset"].device,
            )
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
                        float(self.head.max_start_offset),
                    )
                    for binding in transition.birth_assignments
                ],
                dtype=outputs["start_offset"].dtype,
                device=outputs["start_offset"].device,
            )
            start_loss = F.smooth_l1_loss(
                outputs["start_offset"][0, birth_slots],
                start_targets,
            )
        endpoint_start_pointer_loss = _zero(outputs["start_offset"])
        if (
            self.loss_weights["endpoint_start_pointer_loss"] > 0
            and transition.endpoint_slots
        ):
            instance_by_slot = {
                int(binding.slot_id): int(binding.instance_id)
                for binding in transition.loss_bindings
            }
            endpoint_slots = torch.as_tensor(
                [int(slot) for slot in transition.endpoint_slots],
                dtype=torch.long,
                device=outputs["endpoint_start_pointer_logits"].device,
            )
            pointer_targets = torch.as_tensor(
                [
                    self.head.pointer_target(
                        outputs["memory_frames"],
                        targets[instance_by_slot[int(slot)]].start_frame,
                    )
                    for slot in transition.endpoint_slots
                ],
                dtype=torch.long,
                device=outputs["endpoint_start_pointer_logits"].device,
            )
            endpoint_start_pointer_loss = F.cross_entropy(
                outputs["endpoint_start_pointer_logits"][0, endpoint_slots],
                pointer_targets,
            )

        return {
            "birth_loss": _masked_bce(
                raw_birth_logits,
                birth_target,
                birth_mask,
                self.positive_weights["birth_loss"],
            ),
            "alive_loss": _masked_bce(
                raw_alive_logits,
                alive_target,
                alive_mask,
                self.positive_weights["alive_loss"],
            ),
            "class_loss": class_loss,
            "start_loss": start_loss,
            "end_loss": _masked_bce(
                raw_end_logits,
                end_target,
                end_mask,
                self.positive_weights["end_loss"],
            ),
            "birth_margin_loss": (
                _balanced_binary_logit_margin(
                    raw_birth_logits,
                    birth_target,
                    birth_mask,
                    self.birth_logit_margin,
                )
                if self.loss_weights["birth_margin_loss"] > 0
                else _zero(raw_birth_logits)
            ),
            "alive_margin_loss": (
                _balanced_binary_logit_margin(
                    raw_alive_logits,
                    alive_target,
                    alive_mask,
                    self.alive_logit_margin,
                )
                if self.loss_weights["alive_margin_loss"] > 0
                else _zero(raw_alive_logits)
            ),
            "end_margin_loss": (
                _balanced_binary_logit_margin(
                    raw_end_logits,
                    end_target,
                    end_mask,
                    self.end_logit_margin,
                )
                if self.loss_weights["end_margin_loss"] > 0
                else _zero(raw_end_logits)
            ),
            "birth_calibration_loss": (
                _balanced_binary_calibration_bce(
                    outputs["birth_logits"],
                    birth_target,
                    birth_mask,
                )
                if (
                    self.loss_weights["birth_calibration_loss"] > 0
                    and self.lifecycle_calibration_aggregation
                    == "step_positive_mean"
                )
                else _zero(outputs["birth_logits"])
            ),
            "alive_calibration_loss": (
                _balanced_binary_calibration_bce(
                    outputs["alive_logits"],
                    alive_target,
                    alive_mask,
                )
                if (
                    self.loss_weights["alive_calibration_loss"] > 0
                    and self.lifecycle_calibration_aggregation
                    == "step_positive_mean"
                )
                else _zero(outputs["alive_logits"])
            ),
            "end_calibration_loss": (
                _balanced_binary_calibration_bce(
                    outputs["end_hazard_logits"],
                    end_target,
                    end_mask,
                )
                if (
                    self.loss_weights["end_calibration_loss"] > 0
                    and self.lifecycle_calibration_aggregation
                    == "step_positive_mean"
                )
                else _zero(outputs["end_hazard_logits"])
            ),
            "endpoint_start_pointer_loss": endpoint_start_pointer_loss,
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
        previous_queries = None
        previous_lifecycle_mass = None
        transport_previous_queries = []
        transport_current_queries = []
        transport_previous_lifecycle_mass = []
        transport_current_lifecycle_mass = []
        episode_calibration = None
        if self.lifecycle_calibration_aggregation == "episode_balanced":
            episode_calibration = {
                "birth_calibration_loss": {
                    "channel": "birth",
                    "output": "birth_logits",
                    "logits": [],
                    "targets": [],
                    "masks": [],
                },
                "alive_calibration_loss": {
                    "channel": "alive",
                    "output": "alive_logits",
                    "logits": [],
                    "targets": [],
                    "masks": [],
                },
                "end_calibration_loss": {
                    "channel": "end",
                    "output": "end_hazard_logits",
                    "logits": [],
                    "targets": [],
                    "masks": [],
                },
            }
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
            lifecycle_targets = self._lifecycle_binary_targets(
                outputs,
                transition,
            )
            raw = self._step_losses(
                outputs,
                schedule_step,
                transition,
                feature_stride,
                lifecycle_targets=lifecycle_targets,
            )
            if episode_calibration is not None:
                for loss_name, batch in episode_calibration.items():
                    if self.loss_weights[loss_name] <= 0:
                        continue
                    target, mask = lifecycle_targets[batch["channel"]]
                    batch["logits"].append(outputs[batch["output"]])
                    batch["targets"].append(target)
                    batch["masks"].append(mask)
            current_lifecycle_mass = None
            if self.loss_weights["causal_transport_loss"] > 0:
                with torch.no_grad():
                    current_lifecycle_mass = _predicted_lifecycle_mass(
                        outputs
                    )
                if (
                    previous_queries is not None
                    and previous_lifecycle_mass is not None
                ):
                    transport_previous_queries.append(previous_queries)
                    transport_current_queries.append(state.queries)
                    transport_previous_lifecycle_mass.append(
                        previous_lifecycle_mass
                    )
                    transport_current_lifecycle_mass.append(
                        current_lifecycle_mass
                    )
            raw["causal_transport_loss"] = _zero(state.queries)
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
            if current_lifecycle_mass is not None:
                previous_queries = state.queries.detach()
                previous_lifecycle_mass = current_lifecycle_mass
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
        if transport_previous_queries:
            sums["causal_transport_loss"] = _causal_query_transport_loss(
                torch.cat(transport_previous_queries, dim=0),
                torch.cat(transport_current_queries, dim=0),
                torch.cat(transport_previous_lifecycle_mass, dim=0),
                torch.cat(transport_current_lifecycle_mass, dim=0),
                temperature=self.causal_transport["temperature"],
                identity_cost=self.causal_transport["identity_cost"],
                iterations=self.causal_transport["iterations"],
                mass_floor=self.causal_transport["mass_floor"],
            )
        if episode_calibration is not None:
            for loss_name, batch in episode_calibration.items():
                if not batch["logits"]:
                    continue
                episode_loss = _balanced_binary_calibration_bce(
                    torch.cat(batch["logits"], dim=0),
                    torch.cat(batch["targets"], dim=0),
                    torch.cat(batch["masks"], dim=0),
                )
                # ``losses`` below preserves the established per-step mean.
                # Multiplying here makes the one chunk-level balanced loss the
                # final value instead of diluting it by the number of tokens.
                sums[loss_name] = episode_loss * valid_steps
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
            "prior_bias_mode": self.prior_bias_mode,
            "birth_logit_margin_loss_weight": self.loss_weights[
                "birth_margin_loss"
            ],
            "birth_logit_margin": self.birth_logit_margin,
            "alive_logit_margin_loss_weight": self.loss_weights[
                "alive_margin_loss"
            ],
            "alive_logit_margin": self.alive_logit_margin,
            "end_logit_margin_loss_weight": self.loss_weights[
                "end_margin_loss"
            ],
            "end_logit_margin": self.end_logit_margin,
            "causal_query_transport_loss_weight": self.loss_weights[
                "causal_transport_loss"
            ],
            "causal_query_transport": dict(self.causal_transport),
            "lifecycle_calibration_mode": (
                self.head.lifecycle_calibration_mode
            ),
            "lifecycle_calibration_aggregation": (
                self.lifecycle_calibration_aggregation
            ),
            "birth_calibration_loss_weight": self.loss_weights[
                "birth_calibration_loss"
            ],
            "alive_calibration_loss_weight": self.loss_weights[
                "alive_calibration_loss"
            ],
            "end_calibration_loss_weight": self.loss_weights[
                "end_calibration_loss"
            ],
            "lifecycle_calibration_scale": (
                self.head.lifecycle_calibration_log_scale.detach()
                .exp()
                .cpu()
                .tolist()
                if self.head.lifecycle_calibration_log_scale is not None
                else None
            ),
            "lifecycle_calibration_bias": (
                self.head.lifecycle_calibration_bias.detach().cpu().tolist()
                if self.head.lifecycle_calibration_bias is not None
                else None
            ),
            "end_transition_mode": self.head.end_transition_mode,
            "endpoint_start_mode": self.head.endpoint_start_mode,
            "endpoint_start_pointer_loss_weight": self.loss_weights[
                "endpoint_start_pointer_loss"
            ],
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
