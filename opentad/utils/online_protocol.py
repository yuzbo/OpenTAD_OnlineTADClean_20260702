from dataclasses import dataclass
from typing import Iterable


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
