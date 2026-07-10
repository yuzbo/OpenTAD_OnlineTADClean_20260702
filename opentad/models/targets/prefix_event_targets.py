from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class PrefixTargets:
    class_target: tuple
    start_target: tuple
    ongoing_target: tuple
    end_event: tuple
    censor_mask: tuple
    completion_target: tuple
    emit_allowed: tuple
    emit_forbidden: tuple
    emit_event: tuple
    late_target: tuple
    delay_budget: tuple

    def prefix_observable_dict(self):
        """Return only decisions that are legal at the current prefix."""
        data = asdict(self)
        data.pop("delay_budget", None)
        return data


def _zeros(num_classes):
    return [0.0 for _ in range(num_classes)]


def build_prefix_event_targets(
    segments: Sequence[Sequence[float]],
    labels: Sequence[int],
    previous_frame: float,
    current_frame: float,
    num_classes: int,
    delay_budget_frames: float,
):
    """Build prefix-only lifecycle targets without exposing future endpoints."""
    if len(segments) != len(labels):
        raise ValueError("segments and labels must contain the same number of events")
    num_classes = int(num_classes)
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    previous_frame = float(previous_frame)
    current_frame = float(current_frame)
    delay_budget_frames = float(delay_budget_frames)
    if previous_frame > current_frame:
        raise ValueError("previous_frame must not exceed current_frame")
    if delay_budget_frames < 0:
        raise ValueError("delay_budget_frames must be non-negative")

    class_target = _zeros(num_classes)
    start_target = _zeros(num_classes)
    ongoing_target = _zeros(num_classes)
    end_event = _zeros(num_classes)
    censor_mask = _zeros(num_classes)
    completion_target = _zeros(num_classes)
    emit_allowed = _zeros(num_classes)
    emit_event = _zeros(num_classes)
    late_target = _zeros(num_classes)

    any_started = [False for _ in range(num_classes)]
    any_completed = [False for _ in range(num_classes)]
    any_ongoing = [False for _ in range(num_classes)]

    for segment, raw_label in zip(segments, labels):
        if len(segment) != 2:
            raise ValueError(f"event segment must have two coordinates, got {segment!r}")
        start = float(segment[0])
        end = float(segment[1])
        if end <= start:
            raise ValueError(f"event endpoint must exceed start: start={start} end={end}")
        label = int(raw_label)
        if label < 0 or label >= num_classes:
            raise ValueError(f"event label {label} is outside [0, {num_classes})")

        started = start <= current_frame
        ongoing = start <= current_frame < end
        completed = end <= current_frame
        start_crossed = previous_frame < start <= current_frame
        end_crossed = previous_frame < end <= current_frame
        within_emit_budget = end <= current_frame <= end + delay_budget_frames
        late = current_frame > end + delay_budget_frames

        any_started[label] = any_started[label] or started
        any_ongoing[label] = any_ongoing[label] or ongoing
        any_completed[label] = any_completed[label] or completed

        if started and (ongoing or within_emit_budget):
            class_target[label] = 1.0
        if start_crossed:
            start_target[label] = 1.0
        if ongoing:
            ongoing_target[label] = 1.0
            censor_mask[label] = 1.0
        if end_crossed:
            end_event[label] = 1.0
            emit_event[label] = 1.0
        if completed:
            completion_target[label] = 1.0
        if within_emit_budget:
            emit_allowed[label] = 1.0
        if late:
            late_target[label] = 1.0

    emit_forbidden = []
    for label in range(num_classes):
        if any_ongoing[label]:
            emit_forbidden.append(1.0)
        elif any_completed[label]:
            emit_forbidden.append(0.0)
        elif any_started[label]:
            emit_forbidden.append(1.0)
        else:
            emit_forbidden.append(1.0)

    return PrefixTargets(
        class_target=tuple(class_target),
        start_target=tuple(start_target),
        ongoing_target=tuple(ongoing_target),
        end_event=tuple(end_event),
        censor_mask=tuple(censor_mask),
        completion_target=tuple(completion_target),
        emit_allowed=tuple(emit_allowed),
        emit_forbidden=tuple(emit_forbidden),
        emit_event=tuple(emit_event),
        late_target=tuple(late_target),
        delay_budget=tuple(delay_budget_frames for _ in range(num_classes)),
    )
