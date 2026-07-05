import subprocess
import sys

import pytest

from opentad.utils.online_protocol import (
    PacketRead,
    ProtocolViolation,
    audit_packet_reads,
)


def test_online_protocol_import_does_not_load_torch():
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import opentad.utils.online_protocol; raise SystemExit(1 if 'torch' in sys.modules else 0)",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
    )

    assert probe.returncode == 0, probe.stderr


def test_audit_accepts_selected_packet_inside_bounded_buffer():
    reads = [
        PacketRead(
            video_id="v1",
            current_time=10.0,
            buffer_start=6.0,
            buffer_end=10.0,
            packet_start=7.0,
            packet_end=9.0,
            reader="videomae_v2",
            source="selected_policy",
        )
    ]

    summary = audit_packet_reads(reads)

    assert summary.num_reads == 1
    assert summary.total_packet_duration == pytest.approx(2.0)
    assert summary.max_packet_delay == pytest.approx(3.0)


def test_audit_rejects_future_or_out_of_buffer_packet():
    reads = [
        PacketRead(
            video_id="v1",
            current_time=10.0,
            buffer_start=6.0,
            buffer_end=10.0,
            packet_start=9.0,
            packet_end=10.5,
            reader="videomae_v2",
            source="selected_policy",
        )
    ]

    with pytest.raises(ProtocolViolation, match="outside bounded buffer"):
        audit_packet_reads(reads)


def test_audit_rejects_oracle_dense_cache_or_teacher_sources():
    forbidden_sources = ["gt", "oracle", "dense_cache", "teacher_cache"]

    for source in forbidden_sources:
        reads = [
            PacketRead(
                video_id="v1",
                current_time=10.0,
                buffer_start=6.0,
                buffer_end=10.0,
                packet_start=7.0,
                packet_end=9.0,
                reader="videomae_v2",
                source=source,
            )
        ]

        with pytest.raises(ProtocolViolation, match="forbidden packet source"):
            audit_packet_reads(reads)
