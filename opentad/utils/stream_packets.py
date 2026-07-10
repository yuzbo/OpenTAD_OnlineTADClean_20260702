from collections import deque
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence, Union

from .online_protocol import ProtocolViolation


@dataclass(frozen=True)
class StreamPacketSpec:
    video_id: str
    packet_start_frame: int
    packet_end_frame: int
    is_video_start: bool
    is_video_end: bool

    def model_meta(self):
        return asdict(self)


def build_packet_manifest(video_id: str, total_frames: int, packet_size_frames: int):
    total_frames = int(total_frames)
    packet_size_frames = int(packet_size_frames)
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if packet_size_frames <= 0:
        raise ValueError("packet_size_frames must be positive")

    packets = []
    for start in range(0, total_frames, packet_size_frames):
        end = min(start + packet_size_frames, total_frames)
        packets.append(
            StreamPacketSpec(
                video_id=str(video_id),
                packet_start_frame=start,
                packet_end_frame=end,
                is_video_start=start == 0,
                is_video_end=end == total_frames,
            )
        )
    return tuple(packets)


def select_packet_frame_indices(packet_start_frame, packet_end_frame, policy, stride=1):
    """Select only observable raw frames from one half-open stream packet."""

    start = int(packet_start_frame)
    end = int(packet_end_frame)
    stride = int(stride)
    if end <= start:
        raise ValueError("packet_end_frame must exceed packet_start_frame")
    if stride <= 0:
        raise ValueError("packet frame stride must be positive")
    policy = str(policy)
    if policy in {"packet_all_frames", "all"}:
        indices = tuple(range(start, end))
    elif policy in {"packet_recent_frame", "recent_frame_only", "recent"}:
        indices = (end - 1,)
    elif policy in {"causal_stride", "fixed_causal_stride"}:
        indices = tuple(range(start, end, stride))
    elif policy.startswith("fixed_causal_stride"):
        suffix = policy[len("fixed_causal_stride") :]
        parsed_stride = int(suffix) if suffix else stride
        if parsed_stride <= 0:
            raise ValueError("packet frame stride must be positive")
        indices = tuple(range(start, end, parsed_stride))
    else:
        raise ValueError(f"unsupported packet frame policy: {policy}")
    if not indices or any(index < start or index >= end for index in indices):
        raise ValueError(f"packet selection escaped [{start}, {end}): {indices}")
    return indices


class ChronologicalStreamBatchSampler:
    """Yield packet indices while preserving the order of every video lane."""

    def __init__(
        self,
        manifests: Union[Mapping[str, Sequence[int]], Iterable[Sequence[int]]],
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = False,
    ):
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.drop_last = bool(drop_last)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.world_size != 1 or self.rank != 0:
            raise ProtocolViolation(
                "chronological streaming batches are single-rank until stream-state sharding is implemented"
            )

        if isinstance(manifests, Mapping):
            ordered = manifests.values()
        else:
            ordered = manifests
        self.manifests = tuple(tuple(int(index) for index in manifest) for manifest in ordered if manifest)
        if not self.manifests:
            raise ValueError("at least one non-empty packet manifest is required")

    def _iter_batches(self):
        pending = deque(self.manifests)
        lanes = []
        for _ in range(min(self.batch_size, len(pending))):
            lanes.append([pending.popleft(), 0])

        while lanes:
            batch = []
            next_lanes = []
            for manifest, position in lanes:
                if position >= len(manifest):
                    if not pending:
                        continue
                    manifest = pending.popleft()
                    position = 0

                batch.append(manifest[position])
                position += 1
                next_lanes.append([manifest, position])

            lanes = next_lanes
            if batch and (not self.drop_last or len(batch) == self.batch_size):
                yield batch

    def __iter__(self):
        return self._iter_batches()

    def __len__(self):
        return sum(1 for _ in self._iter_batches())

    def set_epoch(self, epoch):
        # Chronological order is deliberately invariant across epochs.
        self.epoch = int(epoch)
