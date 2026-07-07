import math
from dataclasses import dataclass, field
from typing import Iterable, List


class ProtocolViolation(ValueError):
    """Raised when an online protocol audit finds leaked or invalid evidence."""


@dataclass(frozen=True)
class PacketRead:
    video_id: str
    current_time: float
    buffer_start: float
    buffer_end: float
    packet_start: float
    packet_end: float
    reader: str
    source: str = "selected_policy"


@dataclass(frozen=True)
class PacketReadAuditSummary:
    num_reads: int
    total_packet_duration: float
    max_packet_delay: float


@dataclass(frozen=True)
class GridSpec:
    fps: float
    snippet_stride: int
    window_start_frame: int = 0
    offset_frames: int = 0

    def grid_to_frame(self, grid_idx: float) -> int:
        return int(round(float(grid_idx) * int(self.snippet_stride) + self.window_start_frame + self.offset_frames))


@dataclass(frozen=True)
class OnlineCandidate:
    source_grid: int
    label: object
    score: float
    start_grid: float
    end_grid: float
    source_frame: int = None


@dataclass(frozen=True)
class OnlineDetection:
    video_name: str
    label: object
    score: float
    start_frame: int
    end_frame: int
    emit_frame: int
    source_grid: int
    source_frame: int
    latency_sec: float


@dataclass
class OnlineState:
    video_name: str
    last_emit_frame: int = -1
    last_emitted_grid: int = -1
    emitted: List[OnlineDetection] = field(default_factory=list)


def _cfg_get(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def is_streaming_safe_emission(post_cfg) -> bool:
    return bool(_cfg_get(post_cfg, "streaming_safe_emission", False))


def validate_streaming_safe_world_size(post_cfg, world_size: int) -> None:
    if is_streaming_safe_emission(post_cfg) and int(world_size) > 1:
        raise ProtocolViolation(
            "streaming-safe online evaluation is single-rank for now; "
            "DDP splits the per-video OnlineState across ranks"
        )


def validate_streaming_safe_ext_cls(ext_cls, post_cfg) -> None:
    if is_streaming_safe_emission(post_cfg) and callable(ext_cls) and not isinstance(ext_cls, list):
        raise ProtocolViolation(
            "streaming-safe emission-ledger evaluation does not support callable external classifiers yet"
        )


def make_stream_key(meta, batch_index=0) -> str:
    if not isinstance(meta, dict):
        return f"batch:{batch_index}"
    video_name = meta.get("video_name", f"batch:{batch_index}")
    stream_id = meta.get("stream_id", "default")
    fps = meta.get("fps", "unknown")
    snippet_stride = meta.get("snippet_stride", meta.get("sample_stride", "unknown"))
    offset_frames = meta.get("offset_frames", "unknown")
    processor_id = meta.get("processor_id", meta.get("processor", "unknown"))
    encoder_id = meta.get("encoder_id", meta.get("encoder", "unknown"))
    image_size = meta.get("image_size", "unknown")
    frame_policy = meta.get("frame_policy", "unknown")
    input_format = meta.get("input_format", "unknown")
    return (
        f"video={video_name}|stream={stream_id}|fps={fps}|stride={snippet_stride}|"
        f"offset={offset_frames}|processor={processor_id}|encoder={encoder_id}|"
        f"image_size={image_size}|frame_policy={frame_policy}|input_format={input_format}"
    )


def resolve_sliding_window_for_post_processing(post_cfg, dataset_is_sliding_window: bool) -> bool:
    """Return whether post-processing may do video-level sliding-window merging."""
    if is_streaming_safe_emission(post_cfg):
        return False
    return bool(dataset_is_sliding_window)


def should_run_video_level_nms(post_cfg, dataset_is_sliding_window: bool) -> bool:
    return resolve_sliding_window_for_post_processing(post_cfg, dataset_is_sliding_window) and _cfg_get(
        post_cfg, "nms", None
    ) is not None


def candidate_source_grid(end_grid: float, grid_offset: int = 0, flattened_index: int = None) -> int:
    del flattened_index
    return int(grid_offset) + int(math.ceil(max(0.0, float(end_grid))))


def _result_sort_value(row, key, default):
    if not isinstance(row, dict):
        return default
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def sort_emission_ledger(result_dict):
    """Sort streaming-safe result rows into prefix emission order after DDP gather."""
    sorted_results = {}
    for video_name, rows in result_dict.items():
        sorted_results[video_name] = sorted(
            rows,
            key=lambda row: (
                _result_sort_value(row, "emit_frame", -1),
                _result_sort_value(row, "source_grid", -1),
                _result_sort_value(row, "start_frame", -1),
                _result_sort_value(row, "end_frame", -1),
                str(row.get("label", "")) if isinstance(row, dict) else "",
                -_result_sort_value(row, "score", 0),
            ),
        )
    return sorted_results


def _to_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _percentile(values, percent):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(float(ordered[0]), 6)
    rank = (len(ordered) - 1) * float(percent) / 100.0
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return round(float(ordered[lower]), 6)
    weight = rank - lower
    return round(float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight), 6)


def _summarize_values(values):
    if not values:
        return dict(count=0, min=None, mean=None, p50=None, p90=None, p95=None, max=None)
    return dict(
        count=len(values),
        min=round(float(min(values)), 6),
        mean=round(float(sum(values) / len(values)), 6),
        p50=_percentile(values, 50),
        p90=_percentile(values, 90),
        p95=_percentile(values, 95),
        max=round(float(max(values)), 6),
    )


def summarize_emission_ledger(result_dict):
    """Return streaming emission and latency statistics for auditable online eval."""
    per_video = {}
    per_stream = {}
    latencies = []
    emit_frames = []
    source_grids = []
    future_end_violations = 0
    future_source_violations = 0
    negative_latency_rows = 0
    non_monotonic_emit_rows = 0

    for video_name, rows in result_dict.items():
        per_video[video_name] = len(rows)
        last_emit_by_stream = {}
        for row in rows:
            if not isinstance(row, dict):
                continue

            stream_key = row.get("stream_key", f"video={video_name}")
            per_stream[stream_key] = per_stream.get(stream_key, 0) + 1

            latency = _to_float(row.get("latency_sec"))
            if latency is not None:
                latencies.append(latency)
                if latency < -1e-6:
                    negative_latency_rows += 1

            emit_frame = _to_float(row.get("emit_frame"))
            end_frame = _to_float(row.get("end_frame"))
            source_frame = _to_float(row.get("source_frame"))
            source_grid = _to_float(row.get("source_grid"))
            if emit_frame is not None:
                emit_frames.append(emit_frame)
                previous_emit = last_emit_by_stream.get(stream_key)
                if previous_emit is not None and emit_frame < previous_emit:
                    non_monotonic_emit_rows += 1
                last_emit_by_stream[stream_key] = emit_frame
            if source_grid is not None:
                source_grids.append(source_grid)
            if emit_frame is not None and end_frame is not None and end_frame > emit_frame:
                future_end_violations += 1
            if emit_frame is not None and source_frame is not None and source_frame > emit_frame:
                future_source_violations += 1

    num_emissions = sum(per_video.values())
    return dict(
        num_videos=len(result_dict),
        num_streams=len(per_stream),
        num_emissions=num_emissions,
        per_video_emissions=per_video,
        per_stream_emissions=per_stream,
        latency_sec=_summarize_values(latencies),
        emit_frame=_summarize_values(emit_frames),
        source_grid=_summarize_values(source_grids),
        no_future=dict(
            future_end_violations=future_end_violations,
            future_source_violations=future_source_violations,
            negative_latency_rows=negative_latency_rows,
            non_monotonic_emit_rows=non_monotonic_emit_rows,
        ),
    )


def validate_emission_ledger_summary(summary):
    no_future = summary.get("no_future", {})
    violations = {
        key: int(value)
        for key, value in no_future.items()
        if key.endswith("_violations") or key.endswith("_rows")
        if int(value) > 0
    }
    if violations:
        raise ProtocolViolation(f"streaming emission ledger failed no-future audit: {violations}")


def _segment_iou(a: OnlineDetection, b: OnlineDetection) -> float:
    inter = max(0, min(a.end_frame, b.end_frame) - max(a.start_frame, b.start_frame))
    union = max(a.end_frame, b.end_frame) - min(a.start_frame, b.start_frame)
    return inter / max(union, 1)


def prefix_nms(detections: Iterable[OnlineDetection], iou_threshold: float) -> List[OnlineDetection]:
    ordered = sorted(detections, key=lambda det: det.score, reverse=True)
    kept = []
    for det in ordered:
        if all(det.label != prev.label or _segment_iou(det, prev) <= iou_threshold for prev in kept):
            kept.append(det)
    return sorted(kept, key=lambda det: (det.emit_frame, det.source_grid, str(det.label)))


class OnlineEmitter:
    def __init__(self, grid_spec: GridSpec, score_threshold=0.0, nms_iou_threshold=1.0, latency_frames=0):
        self.grid_spec = grid_spec
        self.score_threshold = float(score_threshold)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.latency_frames = int(latency_frames)

    def step(self, video_name: str, now_frame: int, state: OnlineState, candidates: Iterable[OnlineCandidate]):
        if video_name != state.video_name:
            raise ProtocolViolation(f"state video mismatch: {state.video_name} vs {video_name}")
        if now_frame < state.last_emit_frame:
            raise ProtocolViolation(
                f"non-monotonic online stream for {video_name}: now_frame={now_frame}, "
                f"last_emit_frame={state.last_emit_frame}"
            )

        eligible_end_frame = int(now_frame) - self.latency_frames
        detections = []
        for cand in candidates:
            if int(cand.source_grid) <= state.last_emitted_grid:
                continue
            if float(cand.score) < self.score_threshold:
                continue
            if cand.source_frame is not None and int(cand.source_frame) > int(now_frame):
                continue
            start_frame = self.grid_spec.grid_to_frame(cand.start_grid)
            end_frame = self.grid_spec.grid_to_frame(cand.end_grid)
            if end_frame > eligible_end_frame:
                continue
            source_frame = (
                int(cand.source_frame)
                if cand.source_frame is not None
                else self.grid_spec.grid_to_frame(cand.source_grid)
            )

            detections.append(
                OnlineDetection(
                    video_name=video_name,
                    label=cand.label,
                    score=float(cand.score),
                    start_frame=start_frame,
                    end_frame=end_frame,
                    emit_frame=int(now_frame),
                    source_grid=int(cand.source_grid),
                    source_frame=source_frame,
                    latency_sec=max(0.0, (float(now_frame) - float(end_frame)) / float(self.grid_spec.fps)),
                )
            )

        emitted = []
        for det in prefix_nms(detections, self.nms_iou_threshold):
            if any(
                det.label == prev.label and _segment_iou(det, prev) > self.nms_iou_threshold
                for prev in state.emitted
            ):
                continue
            emitted.append(det)
        if emitted:
            state.last_emitted_grid = max(state.last_emitted_grid, max(det.source_grid for det in emitted))
            state.emitted.extend(emitted)
        state.last_emit_frame = int(now_frame)
        return emitted


def audit_packet_reads(reads: Iterable[PacketRead]) -> PacketReadAuditSummary:
    checked_reads = list(reads)
    total_duration = 0.0
    max_delay = 0.0

    for read in checked_reads:
        _validate_packet_read(read)
        total_duration += read.packet_end - read.packet_start
        max_delay = max(max_delay, read.current_time - read.packet_start)

    return PacketReadAuditSummary(
        num_reads=len(checked_reads),
        total_packet_duration=total_duration,
        max_packet_delay=max_delay,
    )


def _validate_packet_read(read: PacketRead) -> None:
    if read.source != "selected_policy":
        raise ProtocolViolation(f"forbidden packet source: {read.source}")
    if read.buffer_start > read.buffer_end:
        raise ProtocolViolation("buffer start must not exceed buffer end")
    if read.buffer_end > read.current_time:
        raise ProtocolViolation("buffer end is in the future")
    if read.packet_start < read.buffer_start or read.packet_end > read.buffer_end:
        raise ProtocolViolation("packet is outside bounded buffer")
    if read.packet_start >= read.packet_end:
        raise ProtocolViolation("packet start must be before packet end")
