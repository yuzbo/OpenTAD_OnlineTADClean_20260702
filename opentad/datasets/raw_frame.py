import numpy as np
from copy import deepcopy

from .base import SlidingWindowDataset, filter_same_annotation
from .builder import DATASETS


@DATASETS.register_module()
class FrameWindowDataset(SlidingWindowDataset):
    """Sliding-window dataset for raw-frame online TAD experiments."""

    def __init__(
        self,
        input_format="raw_frames",
        online=True,
        stream_id="default",
        processor_id=None,
        encoder_id=None,
        image_size=None,
        frame_policy=None,
        **kwargs,
    ):
        if input_format != "raw_frames":
            raise ValueError("FrameWindowDataset only supports input_format='raw_frames'")
        self.input_format = input_format
        self.online = bool(online)
        self.stream_id = stream_id
        self.processor_id = processor_id
        self.encoder_id = encoder_id
        self.image_size = image_size
        self.frame_policy = frame_policy
        super().__init__(**kwargs)

    def get_gt(self, video_info, thresh=0.0):
        gt_segment = []
        gt_label = []
        if self.fps > 0:
            effective_frames = int(video_info["duration"] * self.fps)
        else:
            effective_frames = int(video_info["frame"])
        for anno in video_info.get("annotations", []):
            if anno.get("label") == "Ambiguous":
                continue
            gt_start = int(anno["segment"][0] / video_info["duration"] * effective_frames)
            gt_end = int(anno["segment"][1] / video_info["duration"] * effective_frames)

            if (not self.filter_gt) or (gt_end - gt_start > thresh):
                gt_segment.append([gt_start, gt_end])
                gt_label.append(0 if self.class_agnostic else self.class_map.index(anno["label"]))

        if len(gt_segment) == 0:
            return None
        annotation = dict(
            gt_segments=np.array(gt_segment, dtype=np.float32),
            gt_labels=np.array(gt_label, dtype=np.int32),
        )
        return filter_same_annotation(annotation)

    def __getitem__(self, index):
        video_name, video_info, video_anno, window_snippet_centers = self.data_list[index]

        if video_anno != {}:
            video_anno = deepcopy(video_anno)
            video_anno["gt_segments"] = video_anno["gt_segments"] - window_snippet_centers[0] - self.offset_frames
            video_anno["gt_segments"] = video_anno["gt_segments"] / self.snippet_stride

        total_frames = int(video_info["duration"] * self.fps) if self.fps > 0 else int(video_info["frame"])
        avg_fps = total_frames / max(float(video_info["duration"]), 1e-6)
        window_start_frame = int(window_snippet_centers[0])
        window_end_frame = int(window_start_frame + len(window_snippet_centers) * self.snippet_stride)

        results = self.pipeline(
            dict(
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
                window_size=self.window_size,
                feature_start_idx=int(window_snippet_centers[0] / self.snippet_stride),
                feature_end_idx=int(window_snippet_centers[-1] / self.snippet_stride),
                sample_stride=self.sample_stride,
                fps=avg_fps,
                snippet_stride=self.snippet_stride,
                window_start_frame=window_start_frame,
                window_end_frame=window_end_frame,
                duration=video_info["duration"],
                offset_frames=self.offset_frames,
                # LoadFrames creates frame_inds; LoadRawFrames consumes them.
                **video_anno,
            )
        )
        return results
