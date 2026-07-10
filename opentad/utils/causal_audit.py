from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

from .online_protocol import ProtocolViolation


_DEFAULT_IGNORED_FIELDS = frozenset(
    {
        "batch_index",
        "chunk_size",
        "runtime_ms",
        "wall_time_sec",
    }
)
_TERMINAL_ONLY_FIELDS = frozenset(
    {
        "duration",
        "num_frames",
        "total_frames",
        "video_duration",
        "video_end_frame",
    }
)


@dataclass(frozen=True)
class AuditReport:
    name: str
    passed: bool
    comparisons: int
    details: dict


def _normalize_value(value: Any, ignored_fields: frozenset, float_digits: int):
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_value(item, ignored_fields, float_digits)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in ignored_fields
        }
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_value(item, ignored_fields, float_digits) for item in value)
    if isinstance(value, float):
        return round(value, float_digits)
    return value


def canonical_prefix_trace(
    rows: Iterable[Mapping[str, Any]],
    through_time: float = None,
    ignored_fields: Iterable[str] = _DEFAULT_IGNORED_FIELDS,
    float_digits: int = 8,
):
    """Normalize model outputs so logical prefix traces can be compared."""
    ignored_fields = frozenset(ignored_fields)
    normalized = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ProtocolViolation(f"audit trace row must be a mapping, got {type(row)!r}")
        logical_time = row.get("time", row.get("emit_frame"))
        if logical_time is None:
            raise ProtocolViolation("audit trace row must contain time or emit_frame")
        if through_time is not None and float(logical_time) > float(through_time):
            continue
        normalized.append(_normalize_value(row, ignored_fields, float_digits))

    return tuple(
        sorted(
            normalized,
            key=lambda row: (
                float(row.get("time", row.get("emit_frame", -1))),
                float(row.get("source_grid", -1)),
                str(row.get("label", "")),
                repr(row),
            ),
        )
    )


def _first_mapping_difference(expected: Mapping[str, Any], actual: Mapping[str, Any]):
    keys = sorted(set(expected) | set(actual))
    for key in keys:
        if key not in expected:
            return key, "<missing>", actual[key]
        if key not in actual:
            return key, expected[key], "<missing>"
        if expected[key] != actual[key]:
            return key, expected[key], actual[key]
    return None


def _assert_same_trace(name: str, expected, actual, context: str = "") -> None:
    common = min(len(expected), len(actual))
    for index in range(common):
        if expected[index] == actual[index]:
            continue
        difference = _first_mapping_difference(expected[index], actual[index])
        field, expected_value, actual_value = difference or ("row", expected[index], actual[index])
        logical_time = expected[index].get(
            "time",
            expected[index].get("emit_frame", actual[index].get("time", actual[index].get("emit_frame"))),
        )
        suffix = f" {context}" if context else ""
        raise ProtocolViolation(
            f"{name} failed{suffix} at time={logical_time} field={field}: "
            f"expected={expected_value!r} actual={actual_value!r}"
        )
    if len(expected) != len(actual):
        logical_time = None
        if common < len(expected):
            logical_time = expected[common].get("time", expected[common].get("emit_frame"))
        elif common < len(actual):
            logical_time = actual[common].get("time", actual[common].get("emit_frame"))
        suffix = f" {context}" if context else ""
        raise ProtocolViolation(
            f"{name} failed{suffix} at time={logical_time} field=row_count: "
            f"expected={len(expected)} actual={len(actual)}"
        )


def audit_future_perturbation(
    stream: Sequence[Any],
    cut_points: Iterable[int],
    runner: Callable[[Sequence[Any], int], Iterable[Mapping[str, Any]]],
    perturb_suffix: Callable[[Sequence[Any], int], Sequence[Any]],
    chunk_size: int = 1,
):
    """Require all outputs through each cut to ignore the unseen suffix."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    comparisons = 0
    checked_cuts = []
    for cut in cut_points:
        cut = int(cut)
        if cut < 0 or cut >= len(stream):
            raise ValueError(f"cut point {cut} is outside stream length {len(stream)}")
        reference_rows = runner(deepcopy(stream), chunk_size)
        perturbed_stream = perturb_suffix(deepcopy(stream), cut)
        if len(perturbed_stream) != len(stream):
            raise ValueError("perturb_suffix must preserve stream length")
        perturbed_rows = runner(perturbed_stream, chunk_size)
        expected = canonical_prefix_trace(reference_rows, through_time=cut)
        actual = canonical_prefix_trace(perturbed_rows, through_time=cut)
        _assert_same_trace("future perturbation", expected, actual, context=f"cut={cut}")
        comparisons += 1
        checked_cuts.append(cut)
    return AuditReport(
        name="future_perturbation",
        passed=True,
        comparisons=comparisons,
        details={"cut_points": tuple(checked_cuts), "chunk_size": int(chunk_size)},
    )


def audit_chunk_invariance(
    stream: Sequence[Any],
    runner: Callable[[Sequence[Any], int], Iterable[Mapping[str, Any]]],
    chunk_sizes: Iterable[int],
):
    """Require logically identical outputs for every transport chunk size."""
    chunk_sizes = tuple(int(size) for size in chunk_sizes)
    if len(chunk_sizes) < 2:
        raise ValueError("chunk invariance requires at least two chunk sizes")
    if any(size <= 0 for size in chunk_sizes):
        raise ValueError("chunk sizes must be positive")

    reference_size = chunk_sizes[0]
    reference = canonical_prefix_trace(runner(deepcopy(stream), reference_size))
    comparisons = 0
    for size in chunk_sizes[1:]:
        actual = canonical_prefix_trace(runner(deepcopy(stream), size))
        _assert_same_trace(
            "chunk invariance",
            reference,
            actual,
            context=f"reference_chunk={reference_size} actual_chunk={size}",
        )
        comparisons += 1
    return AuditReport(
        name="chunk_invariance",
        passed=True,
        comparisons=comparisons,
        details={"chunk_sizes": chunk_sizes},
    )


def audit_packet_metadata(packets: Iterable[Mapping[str, Any]]):
    """Validate new-frame packet order and hide terminal-only metadata."""
    previous_end = {}
    ended_streams = set()
    packet_count = 0

    for packet in packets:
        packet_count += 1
        if not isinstance(packet, Mapping):
            raise ProtocolViolation(f"packet metadata must be a mapping, got {type(packet)!r}")
        missing = [
            key
            for key in (
                "video_id",
                "packet_start_frame",
                "packet_end_frame",
                "is_video_start",
                "is_video_end",
            )
            if key not in packet
        ]
        if missing:
            raise ProtocolViolation(f"packet metadata missing required fields: {missing}")

        video_id = str(packet["video_id"])
        start = int(packet["packet_start_frame"])
        end = int(packet["packet_end_frame"])
        is_start = bool(packet["is_video_start"])
        is_end = bool(packet["is_video_end"])
        if start < 0 or end <= start:
            raise ProtocolViolation(
                f"packet range must satisfy 0 <= start < end for {video_id}: start={start} end={end}"
            )
        if video_id in ended_streams:
            raise ProtocolViolation(f"packet arrived after terminal packet for {video_id}")

        expected_start = previous_end.get(video_id)
        if expected_start is None:
            if not is_start:
                raise ProtocolViolation(f"first packet for {video_id} must set is_video_start=True")
        else:
            if is_start:
                raise ProtocolViolation(f"non-first packet for {video_id} cannot reset stream state")
            if start != expected_start:
                raise ProtocolViolation(
                    f"packet must contain only new frames for {video_id}: "
                    f"expected_start={expected_start} actual_start={start}"
                )

        if not is_end:
            leaked = sorted(
                key for key in _TERMINAL_ONLY_FIELDS if key in packet and packet[key] is not None
            )
            if leaked:
                raise ProtocolViolation(
                    f"non-terminal packet exposes terminal metadata for {video_id}: {', '.join(leaked)}"
                )

        previous_end[video_id] = end
        if is_end:
            ended_streams.add(video_id)

    return AuditReport(
        name="packet_metadata",
        passed=True,
        comparisons=packet_count,
        details={"streams": len(previous_end), "packets": packet_count},
    )


def audit_batch_isolation(
    stream_a: Sequence[Any],
    stream_b: Sequence[Any],
    runner: Callable[[Mapping[str, Sequence[Any]], int], Mapping[str, Iterable[Mapping[str, Any]]]],
    chunk_size: int = 1,
):
    """Require a stream trace to be unchanged by an unrelated batch neighbor."""
    single_a = runner({"a": deepcopy(stream_a)}, chunk_size)
    single_b = runner({"b": deepcopy(stream_b)}, chunk_size)
    batched = runner({"a": deepcopy(stream_a), "b": deepcopy(stream_b)}, chunk_size)
    for key, outputs in (("a", single_a), ("b", single_b), ("a", batched), ("b", batched)):
        if key not in outputs:
            raise ProtocolViolation(f"batch isolation runner omitted stream={key}")

    _assert_same_trace(
        "batch isolation",
        canonical_prefix_trace(single_a["a"]),
        canonical_prefix_trace(batched["a"]),
        context="stream=a",
    )
    _assert_same_trace(
        "batch isolation",
        canonical_prefix_trace(single_b["b"]),
        canonical_prefix_trace(batched["b"]),
        context="stream=b",
    )
    return AuditReport(
        name="batch_isolation",
        passed=True,
        comparisons=2,
        details={"streams": ("a", "b"), "chunk_size": int(chunk_size)},
    )


def audit_recorded_trace_equivalence(
    reference_rows,
    candidate_rows,
    through_time=None,
    name="recorded_trace_equivalence",
):
    """Compare traces captured by separate perturbation/chunk audit runs."""

    reference = canonical_prefix_trace(reference_rows, through_time=through_time)
    candidate = canonical_prefix_trace(candidate_rows, through_time=through_time)
    _assert_same_trace(str(name).replace("_", " "), reference, candidate)
    return AuditReport(
        name=str(name),
        passed=True,
        comparisons=1,
        details={"through_time": through_time, "rows": len(reference)},
    )


def audit_emission_ledger(result_dict):
    """Fail closed on mutable emissions or future read provenance."""

    if isinstance(result_dict, Mapping) and "results" in result_dict:
        result_dict = result_dict["results"]
    if not isinstance(result_dict, Mapping):
        raise ProtocolViolation("emission ledger must be a video-to-rows mapping")
    required = {
        "segment",
        "start_frame",
        "end_frame",
        "emit_frame",
        "max_raw_frame_read",
        "max_cache_source_frame",
        "label",
        "score",
        "stream_key",
        "immutable",
    }
    last_emit_by_stream = {}
    checked = 0
    streams = set()
    for video_id, rows in result_dict.items():
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise ProtocolViolation(f"ledger rows for {video_id} must be a sequence")
        for row in rows:
            checked += 1
            if not isinstance(row, Mapping):
                raise ProtocolViolation(f"ledger row for {video_id} must be a mapping")
            missing = sorted(required.difference(row))
            if missing:
                raise ProtocolViolation(f"ledger row for {video_id} missing fields: {missing}")
            if not bool(row["immutable"]):
                raise ProtocolViolation(f"mutable emission for {video_id}")
            stream_key = str(row["stream_key"])
            streams.add(stream_key)
            start = int(row["start_frame"])
            end = int(row["end_frame"])
            emit = int(row["emit_frame"])
            max_raw = int(row["max_raw_frame_read"])
            max_cache = int(row["max_cache_source_frame"])
            if start > end:
                raise ProtocolViolation(f"invalid emitted frame range for {video_id}: {start}>{end}")
            if end > emit:
                raise ProtocolViolation(f"future emitted endpoint for {video_id}: end={end} emit={emit}")
            if max_raw > emit:
                raise ProtocolViolation(
                    f"future raw source for {video_id}: source={max_raw} emit={emit}"
                )
            if max_cache > emit:
                raise ProtocolViolation(
                    f"future cache source for {video_id}: source={max_cache} emit={emit}"
                )
            previous_emit = last_emit_by_stream.get(stream_key)
            if previous_emit is not None and emit < previous_emit:
                raise ProtocolViolation(
                    f"non-monotonic emission for {stream_key}: previous={previous_emit} emit={emit}"
                )
            last_emit_by_stream[stream_key] = emit
            score = float(row["score"])
            if not math.isfinite(score):
                raise ProtocolViolation(f"non-finite emission score for {video_id}: {score}")
    return AuditReport(
        name="emission_ledger",
        passed=True,
        comparisons=checked,
        details={"videos": len(result_dict), "streams": len(streams), "rows": checked},
    )
