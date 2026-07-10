from pathlib import Path

import pytest

from opentad.utils.online_protocol import ProtocolViolation
from opentad.utils.stream_packets import (
    ChronologicalStreamBatchSampler,
    build_packet_manifest,
    select_packet_frame_indices,
)


ROOT = Path(__file__).resolve().parents[1]


def test_packet_manifest_contains_only_new_frames_and_hides_future_duration():
    packets = build_packet_manifest("v1", total_frames=20, packet_size_frames=8)

    assert [(p.packet_start_frame, p.packet_end_frame) for p in packets] == [
        (0, 8),
        (8, 16),
        (16, 20),
    ]
    assert packets[0].model_meta() == {
        "video_id": "v1",
        "packet_start_frame": 0,
        "packet_end_frame": 8,
        "is_video_start": True,
        "is_video_end": False,
    }
    assert packets[-1].model_meta() == {
        "video_id": "v1",
        "packet_start_frame": 16,
        "packet_end_frame": 20,
        "is_video_start": False,
        "is_video_end": True,
    }


def test_packet_manifest_rejects_invalid_lengths():
    with pytest.raises(ValueError, match="total_frames must be positive"):
        build_packet_manifest("v1", total_frames=0, packet_size_frames=8)
    with pytest.raises(ValueError, match="packet_size_frames must be positive"):
        build_packet_manifest("v1", total_frames=20, packet_size_frames=0)


def test_chronological_sampler_preserves_lane_order_and_visits_every_packet():
    manifests = {
        "v1": [0, 1, 2],
        "v2": [3, 4],
        "v3": [5, 6],
    }

    batches = list(
        ChronologicalStreamBatchSampler(
            manifests,
            batch_size=2,
            rank=0,
            world_size=1,
        )
    )
    flattened = [index for batch in batches for index in batch]

    assert batches == [[0, 3], [1, 4], [2, 5], [6]]
    assert sorted(flattened) == list(range(7))
    for indices in manifests.values():
        positions = [flattened.index(index) for index in indices]
        assert positions == sorted(positions)


def test_chronological_sampler_can_drop_partial_final_batch():
    sampler = ChronologicalStreamBatchSampler(
        {"v1": [0, 1, 2], "v2": [3, 4]},
        batch_size=2,
        rank=0,
        world_size=1,
        drop_last=True,
    )

    assert list(sampler) == [[0, 3], [1, 4]]
    assert len(sampler) == 2


def test_chronological_sampler_rejects_multi_rank_state_splitting():
    with pytest.raises(ProtocolViolation, match="single-rank"):
        ChronologicalStreamBatchSampler(
            {"v1": [0, 1]},
            batch_size=1,
            rank=0,
            world_size=2,
        )


@pytest.mark.parametrize(
    ("policy", "stride", "expected"),
    [
        ("packet_all_frames", 1, (8, 9, 10, 11, 12, 13, 14, 15)),
        ("packet_recent_frame", 1, (15,)),
        ("fixed_causal_stride2", 2, (8, 10, 12, 14)),
    ],
)
def test_packet_frame_selection_is_prefix_only(policy, stride, expected):
    assert select_packet_frame_indices(8, 16, policy=policy, stride=stride) == expected


def test_packet_frame_selection_rejects_unknown_or_invalid_policy():
    with pytest.raises(ValueError, match="unsupported packet frame policy"):
        select_packet_frame_indices(0, 8, policy="oracle_future", stride=1)
    with pytest.raises(ValueError, match="packet_end_frame"):
        select_packet_frame_indices(8, 8, policy="packet_recent_frame", stride=1)


def test_registered_dataset_sanitizes_non_terminal_metadata():
    source = (ROOT / "opentad" / "datasets" / "streaming_raw_frame.py").read_text(encoding="utf-8")
    init_source = (ROOT / "opentad" / "datasets" / "__init__.py").read_text(encoding="utf-8")
    builder_source = (ROOT / "opentad" / "datasets" / "builder.py").read_text(encoding="utf-8")

    assert "class StreamingRawFrameDataset" in source
    assert "sanitize_stream_packet_meta" in source
    assert "stream_gt_segments" in source
    assert "StreamingRawFrameDataset" in init_source
    assert "ChronologicalStreamBatchSampler" in builder_source
    assert "streaming=False" in builder_source
    transform_source = (
        ROOT / "opentad" / "datasets" / "transforms" / "streaming.py"
    ).read_text(encoding="utf-8")
    assert "class LoadStreamPacketFrames" in transform_source
    assert "encoded_source_frames" in transform_source
