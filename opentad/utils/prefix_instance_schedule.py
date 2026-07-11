from dataclasses import asdict, dataclass
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class PrefixInstanceTarget:
    instance_id: int
    label: int
    start_frame: float
    end_frame: Optional[float] = None


@dataclass(frozen=True)
class PrefixScheduleStep:
    current_frame: int
    births: Tuple[PrefixInstanceTarget, ...]
    active: Tuple[PrefixInstanceTarget, ...]
    ends: Tuple[PrefixInstanceTarget, ...]

    def prefix_observable_dict(self):
        return asdict(self)


def _validate_inputs(segments, labels, decision_frames, previous_frame):
    if len(segments) != len(labels):
        raise ValueError("segments and labels must contain the same number of instances")

    parsed_segments = []
    for instance_id, (segment, raw_label) in enumerate(zip(segments, labels)):
        if len(segment) != 2:
            raise ValueError(f"instance segment must have two coordinates, got {segment!r}")
        start = float(segment[0])
        end = float(segment[1])
        if end <= start:
            raise ValueError(f"instance endpoint must exceed start: start={start} end={end}")
        parsed_segments.append((instance_id, int(raw_label), start, end))

    parsed_frames = [int(frame) for frame in decision_frames]
    prior = float(previous_frame)
    for frame in parsed_frames:
        if frame <= prior:
            raise ValueError("decision frames must be strictly increasing after previous_frame")
        prior = frame
    return tuple(parsed_segments), tuple(parsed_frames)


def build_prefix_instance_schedule(
    segments: Sequence[Sequence[float]],
    labels: Sequence[int],
    decision_frames: Sequence[int],
    previous_frame: float,
):
    """Build per-decision labels that never reveal an unobserved endpoint."""
    instances, frames = _validate_inputs(segments, labels, decision_frames, previous_frame)
    schedule = []
    prior = float(previous_frame)

    for current in frames:
        births = []
        active = []
        ends = []
        for instance_id, label, start, end in instances:
            if prior < start <= current:
                births.append(PrefixInstanceTarget(instance_id, label, start))
            if start <= current < end:
                active.append(PrefixInstanceTarget(instance_id, label, start))
            if prior < end <= current:
                ends.append(PrefixInstanceTarget(instance_id, label, start, end))

        schedule.append(
            PrefixScheduleStep(
                current_frame=current,
                births=tuple(births),
                active=tuple(active),
                ends=tuple(ends),
            )
        )
        prior = float(current)

    return tuple(schedule)
