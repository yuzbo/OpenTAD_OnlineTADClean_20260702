def resolve_raw_video_total_frames(video_info, fps):
    annotated_frames = int(video_info.get("frame", 0) or 0)
    if fps is not None and float(fps) > 0:
        protocol_frames = int(float(video_info["duration"]) * float(fps))
        if annotated_frames > 0:
            return max(1, min(protocol_frames, annotated_frames))
        return max(1, protocol_frames)
    return max(1, annotated_frames)
