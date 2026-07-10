import numpy as np
import torch

from opentad.utils.stream_packets import select_packet_frame_indices

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadStreamPacketFrames:
    """Create raw-frame indices for exactly one observable stream packet."""

    def __init__(self, policy=None, stride=1):
        self.policy = policy
        self.stride = int(stride)

    def __call__(self, results):
        policy = self.policy or results.get("frame_policy")
        if policy is None:
            raise ValueError("LoadStreamPacketFrames requires an explicit causal frame policy")
        indices = select_packet_frame_indices(
            results["packet_start_frame"],
            results["packet_end_frame"],
            policy=policy,
            stride=self.stride,
        )
        results["frame_inds"] = np.asarray(indices, dtype=np.int64).reshape(1, -1)
        results["encoded_source_frames"] = list(indices)
        results["num_clips"] = 1
        results["clip_len"] = len(indices)
        results["masks"] = torch.ones(len(indices), dtype=torch.bool)
        return results
