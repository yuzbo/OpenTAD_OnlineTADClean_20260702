from copy import deepcopy

import numpy as np

from opentad.utils.stream_packets import build_packet_manifest

from .builder import DATASETS
from .raw_frame import FrameWindowDataset


_TERMINAL_ONLY_META = (
    "duration",
    "num_frames",
    "total_frames",
    "video_duration",
    "video_end_frame",
)


def sanitize_stream_packet_meta(meta, packet):
    meta = dict(meta or {})
    meta.update(packet.model_meta())
    meta["video_name"] = packet.video_id
    meta["window_start_frame"] = int(packet.packet_start_frame)
    meta["window_end_frame"] = int(packet.packet_end_frame)
    if not packet.is_video_end:
        for key in _TERMINAL_ONLY_META:
            meta.pop(key, None)
    return meta


@DATASETS.register_module()
class StreamingRawFrameDataset(FrameWindowDataset):
    """Chronological raw-frame packets for strict single-rank On-TAD."""

    def __init__(self, packet_size_frames=8, expose_terminal_duration=True, **kwargs):
        self.packet_size_frames = int(packet_size_frames)
        self.expose_terminal_duration = bool(expose_terminal_duration)
        if self.packet_size_frames <= 0:
            raise ValueError("packet_size_frames must be positive")
        super().__init__(**kwargs)

        packet_manifests = {}
        for index, item in enumerate(self.data_list):
            video_name = item[0]
            packet_manifests.setdefault(video_name, []).append(index)
        self.packet_manifests = packet_manifests

    def split_video_to_windows(self, video_name, video_info, video_anno):
        total_frames = self.get_num_frames(video_info)
        return [
            [video_name, video_info, deepcopy(video_anno), packet]
            for packet in build_packet_manifest(
                video_name,
                total_frames=total_frames,
                packet_size_frames=self.packet_size_frames,
            )
        ]

    def __getitem__(self, index):
        video_name, video_info, video_anno, packet = self.data_list[index]
        total_frames = self.get_num_frames(video_info)
        duration = float(video_info["duration"])
        avg_fps = total_frames / max(duration, 1e-6)

        snippet_centers = np.arange(
            packet.packet_start_frame,
            packet.packet_end_frame,
            max(self.snippet_stride, 1),
            dtype=np.int64,
        )
        if len(snippet_centers) == 0:
            snippet_centers = np.asarray([packet.packet_start_frame], dtype=np.int64)

        absolute_segments = None
        absolute_labels = None
        local_anno = {}
        if video_anno:
            absolute_segments = deepcopy(video_anno["gt_segments"])
            absolute_labels = deepcopy(video_anno["gt_labels"])
            local_anno = deepcopy(video_anno)
            local_anno["gt_segments"] = (
                local_anno["gt_segments"] - packet.packet_start_frame - self.offset_frames
            ) / max(self.snippet_stride, 1)

        pipeline_input = dict(
            video_name=video_name,
            data_path=self.data_path,
            input_format=self.input_format,
            stream_id=self.stream_id,
            processor_id=self.processor_id,
            encoder_id=self.encoder_id,
            image_size=self.image_size,
            frame_policy=self.frame_policy,
            total_frames=total_frames,
            avg_fps=avg_fps,
            window_size=len(snippet_centers),
            feature_start_idx=int(snippet_centers[0] / max(self.snippet_stride, 1)),
            feature_end_idx=int(snippet_centers[-1] / max(self.snippet_stride, 1)),
            sample_stride=self.sample_stride,
            fps=avg_fps,
            snippet_stride=self.snippet_stride,
            window_start_frame=int(packet.packet_start_frame),
            window_end_frame=int(packet.packet_end_frame),
            packet_start_frame=int(packet.packet_start_frame),
            packet_end_frame=int(packet.packet_end_frame),
            is_video_start=bool(packet.is_video_start),
            is_video_end=bool(packet.is_video_end),
            duration=duration,
            offset_frames=self.offset_frames,
            **local_anno,
        )
        results = self.pipeline(pipeline_input)
        results["metas"] = sanitize_stream_packet_meta(results.get("metas"), packet)
        if packet.is_video_end and not self.expose_terminal_duration:
            for key in _TERMINAL_ONLY_META:
                results["metas"].pop(key, None)
        if absolute_segments is not None:
            results["stream_gt_segments"] = absolute_segments
            results["stream_gt_labels"] = absolute_labels
        return results
