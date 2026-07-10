import pytest

from opentad.utils.causal_audit import (
    audit_batch_isolation,
    audit_chunk_invariance,
    audit_emission_ledger,
    audit_future_perturbation,
    audit_packet_metadata,
    audit_recorded_trace_equivalence,
)
from opentad.utils.online_protocol import ProtocolViolation


def _prefix_runner(stream, chunk_size):
    del chunk_size
    running = 0
    rows = []
    for logical_time, value in enumerate(stream):
        running += value
        rows.append(
            {
                "time": logical_time,
                "score": float(running),
                "status": "provisional",
                "runtime_ms": 0.1 + logical_time,
            }
        )
    return rows


def test_future_perturbation_accepts_prefix_only_runner():
    report = audit_future_perturbation(
        stream=[1, 2, 3, 4],
        cut_points=[1, 2],
        runner=_prefix_runner,
        perturb_suffix=lambda xs, cut: xs[: cut + 1] + [99] * (len(xs) - cut - 1),
    )

    assert report.passed is True
    assert report.name == "future_perturbation"
    assert report.comparisons == 2


def test_future_perturbation_reports_first_divergence():
    def leaky_runner(stream, chunk_size):
        del chunk_size
        total = float(sum(stream))
        return [{"time": logical_time, "score": total} for logical_time in range(len(stream))]

    with pytest.raises(ProtocolViolation, match=r"future perturbation.*time=0.*score"):
        audit_future_perturbation(
            stream=[1, 2, 3, 4],
            cut_points=[1],
            runner=leaky_runner,
            perturb_suffix=lambda xs, cut: xs[: cut + 1] + [99] * (len(xs) - cut - 1),
        )


def test_chunk_invariance_compares_logical_timestamps_not_transport_fields():
    report = audit_chunk_invariance(
        stream=[1, 2, 3, 4],
        runner=_prefix_runner,
        chunk_sizes=[1, 2, 4],
    )

    assert report.passed is True
    assert report.comparisons == 2


def test_chunk_invariance_rejects_chunk_dependent_predictions():
    def chunk_dependent_runner(stream, chunk_size):
        return [
            {"time": logical_time, "score": float(value + chunk_size)}
            for logical_time, value in enumerate(stream)
        ]

    with pytest.raises(ProtocolViolation, match=r"chunk invariance.*time=0.*score"):
        audit_chunk_invariance(
            stream=[1, 2, 3],
            runner=chunk_dependent_runner,
            chunk_sizes=[1, 3],
        )


def test_eos_audit_rejects_terminal_metadata_on_intermediate_packet():
    packets = [
        {
            "video_id": "v1",
            "packet_start_frame": 0,
            "packet_end_frame": 8,
            "is_video_start": True,
            "is_video_end": False,
            "duration": 10.0,
        }
    ]

    with pytest.raises(ProtocolViolation, match=r"terminal metadata.*duration"):
        audit_packet_metadata(packets)


def test_packet_audit_accepts_new_monotonic_packets_and_terminal_duration():
    packets = [
        {
            "video_id": "v1",
            "packet_start_frame": 0,
            "packet_end_frame": 8,
            "is_video_start": True,
            "is_video_end": False,
        },
        {
            "video_id": "v1",
            "packet_start_frame": 8,
            "packet_end_frame": 12,
            "is_video_start": False,
            "is_video_end": True,
            "duration": 0.4,
            "total_frames": 12,
        },
    ]

    report = audit_packet_metadata(packets)

    assert report.passed is True
    assert report.details["streams"] == 1
    assert report.details["packets"] == 2


def test_packet_audit_rejects_overlap_or_replayed_frames():
    packets = [
        {
            "video_id": "v1",
            "packet_start_frame": 0,
            "packet_end_frame": 8,
            "is_video_start": True,
            "is_video_end": False,
        },
        {
            "video_id": "v1",
            "packet_start_frame": 6,
            "packet_end_frame": 12,
            "is_video_start": False,
            "is_video_end": True,
        },
    ]

    with pytest.raises(ProtocolViolation, match=r"new frames.*expected_start=8.*actual_start=6"):
        audit_packet_metadata(packets)


def test_batch_isolation_accepts_stream_local_state():
    def isolated_runner(streams, chunk_size):
        return {key: _prefix_runner(values, chunk_size) for key, values in streams.items()}

    report = audit_batch_isolation(
        stream_a=[1, 2, 3],
        stream_b=[10, 20, 30],
        runner=isolated_runner,
    )

    assert report.passed is True
    assert report.comparisons == 2


def test_batch_isolation_detects_cross_stream_state():
    def leaky_runner(streams, chunk_size):
        del chunk_size
        batch_offset = float(sum(sum(values) for values in streams.values()))
        return {
            key: [
                {"time": logical_time, "score": float(value) + batch_offset}
                for logical_time, value in enumerate(values)
            ]
            for key, values in streams.items()
        }

    with pytest.raises(ProtocolViolation, match=r"batch isolation.*stream=a.*time=0.*score"):
        audit_batch_isolation(
            stream_a=[1, 2, 3],
            stream_b=[10, 20, 30],
            runner=leaky_runner,
        )


def test_recorded_trace_equivalence_supports_offline_audit_artifacts():
    reference = [{"time": 0, "score": 0.1}, {"time": 1, "score": 0.2}]
    candidate = [{"time": 0, "score": 0.1}, {"time": 1, "score": 99.0}]

    report = audit_recorded_trace_equivalence(
        reference,
        candidate,
        through_time=0,
        name="future_perturbation_recorded",
    )

    assert report.passed
    assert report.comparisons == 1


def test_emission_ledger_audits_immutable_read_provenance():
    rows = {
        "v1": [
            {
                "segment": [0.0, 0.2],
                "start_frame": 0,
                "end_frame": 6,
                "emit_frame": 7,
                "max_raw_frame_read": 7,
                "max_cache_source_frame": 6,
                "label": "Action",
                "score": 0.8,
                "stream_key": "video=v1|stream=default",
                "immutable": True,
            }
        ]
    }

    assert audit_emission_ledger(rows).passed
    rows["v1"][0]["max_cache_source_frame"] = 8
    with pytest.raises(ProtocolViolation, match="future cache source"):
        audit_emission_ledger(rows)
