import inspect

import torch
from ..builder import DETECTORS, build_backbone, build_projection, build_head, build_neck
from .base import BaseDetector
from ..utils.post_processing import batched_nms, convert_to_seconds
from opentad.utils.online_protocol import (
    GridSpec,
    OnlineCandidate,
    OnlineEmitter,
    OnlineState,
    candidate_source_grid,
    is_streaming_safe_emission,
    make_stream_key,
    validate_streaming_safe_ext_cls,
)


@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """
    Base class for single-stage detectors which should not have roi_extractors.
    """

    def __init__(self, backbone=None, projection=None, neck=None, rpn_head=None):
        super(SingleStageDetector, self).__init__()
        self._online_states = {}

        if backbone is not None:
            self.backbone = build_backbone(backbone)

        if projection is not None:
            self.projection = build_projection(projection)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            self.rpn_head = build_head(rpn_head)

    def reset_online_states(self):
        self._online_states.clear()
        if hasattr(self, "rpn_head") and hasattr(self.rpn_head, "reset_stream_state"):
            self.rpn_head.reset_stream_state()

    @property
    def with_backbone(self):
        """bool: whether the detector has backbone"""
        return hasattr(self, "backbone") and self.backbone is not None

    @property
    def with_projection(self):
        """bool: whether the detector has projection"""
        return hasattr(self, "projection") and self.projection is not None

    @property
    def with_neck(self):
        """bool: whether the detector has neck"""
        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_rpn_head(self):
        """bool: whether the detector has localization head"""
        return hasattr(self, "rpn_head") and self.rpn_head is not None

    @staticmethod
    def _unpack_backbone_output(output, masks):
        if isinstance(output, dict):
            x = output.get("features", output.get("feats", output.get("last_hidden_state")))
            if x is None:
                raise ValueError("Backbone output dict did not contain features")
            return x, output.get("masks", masks)
        if isinstance(output, (tuple, list)):
            if len(output) >= 2 and torch.is_tensor(output[1]) and output[1].dim() <= 2:
                return output[0], output[1]
            return output[-1], masks
        return output, masks

    def _forward_backbone(self, inputs, masks, metas=None):
        try:
            signature = inspect.signature(self.backbone.forward)
        except (TypeError, ValueError):
            output = self.backbone(inputs)
            return self._unpack_backbone_output(output, masks)

        params = signature.parameters
        accepts_kwargs = any(param.kind == param.VAR_KEYWORD for param in params.values())
        if accepts_kwargs or ("masks" in params and "metas" in params):
            output = self.backbone(inputs, masks=masks, metas=metas)
        elif "masks" in params:
            output = self.backbone(inputs, masks=masks)
        else:
            output = self.backbone(inputs)
        return self._unpack_backbone_output(output, masks)

    def forward_train(self, inputs, masks, metas, gt_segments, gt_labels, **kwargs):
        losses = dict()
        if self.with_backbone:
            x, masks = self._forward_backbone(inputs, masks, metas)
        else:
            x = inputs

        if self.with_projection:
            x, masks = self.projection(x, masks)

        if self.with_neck:
            x, masks = self.neck(x, masks)

        if self.with_rpn_head:
            rpn_train_kwargs = dict(gt_segments=gt_segments, gt_labels=gt_labels, **kwargs)
            try:
                signature = inspect.signature(self.rpn_head.forward_train)
                params = signature.parameters
                accepts_kwargs = any(param.kind == param.VAR_KEYWORD for param in params.values())
            except (TypeError, ValueError):
                accepts_kwargs = False
                params = {}
            if accepts_kwargs or "metas" in params:
                rpn_train_kwargs["metas"] = metas
            rpn_losses = self.rpn_head.forward_train(x, masks, **rpn_train_kwargs)
            losses.update(rpn_losses)

        # only key has loss will be record
        losses["cost"] = sum(_value for _key, _value in losses.items())
        return losses

    def forward_test(self, inputs, masks, metas=None, infer_cfg=None, **kwargs):
        if self.with_backbone:
            x, masks = self._forward_backbone(inputs, masks, metas)
        else:
            x = inputs

        if self.with_projection:
            x, masks = self.projection(x, masks)

        if self.with_neck:
            x, masks = self.neck(x, masks)

        if self.with_rpn_head:
            rpn_output = self.rpn_head.forward_test(x, masks, metas=metas)
            if isinstance(rpn_output, tuple) and len(rpn_output) == 3:
                rpn_proposals, rpn_scores, rpn_meta = rpn_output
            else:
                rpn_proposals, rpn_scores = rpn_output
                rpn_meta = None
        else:
            rpn_proposals = rpn_scores = None
            rpn_meta = None

        predictions = (rpn_proposals, rpn_scores, rpn_meta) if rpn_meta is not None else (rpn_proposals, rpn_scores)
        return predictions

    @torch.no_grad()
    def post_processing(self, predictions, metas, post_cfg, ext_cls, **kwargs):
        if isinstance(predictions, tuple) and len(predictions) == 3:
            rpn_proposals, rpn_scores, rpn_meta = predictions
        else:
            rpn_proposals, rpn_scores = predictions
            rpn_meta = None
        # rpn_proposals,  # [B,K,2]
        # rpn_scores,  # [B,K,num_classes] after sigmoid

        pre_nms_thresh = getattr(post_cfg, "pre_nms_thresh", 0.001)
        pre_nms_topk = getattr(post_cfg, "pre_nms_topk", 2000)
        streaming_safe = is_streaming_safe_emission(post_cfg)
        validate_streaming_safe_ext_cls(ext_cls, post_cfg)
        num_classes = rpn_scores[0].shape[-1]

        results = {}
        for i in range(len(metas)):  # processing each video
            segments = rpn_proposals[i].detach().cpu()  # [N,2]
            scores = rpn_scores[i].detach().cpu()  # [N,class]
            source_grids = segments[:, 1].detach().clone()
            source_frames = None
            has_explicit_source_grids = False
            if rpn_meta is not None:
                if isinstance(rpn_meta, dict):
                    if "source_grids" in rpn_meta:
                        source_grids = rpn_meta["source_grids"][i].detach().cpu()
                        has_explicit_source_grids = True
                    if "source_frames" in rpn_meta:
                        source_frames = rpn_meta["source_frames"][i].detach().cpu()

            if num_classes == 1:
                scores = scores.squeeze(-1)
                labels = torch.zeros(scores.shape[0]).contiguous()
            else:
                pred_prob = scores.flatten()  # [N*class]

                # Apply filtering to make NMS faster following detectron2
                # 1. Keep seg with confidence score > a threshold
                keep_idxs1 = pred_prob > pre_nms_thresh
                pred_prob = pred_prob[keep_idxs1]
                topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]

                # 2. Keep top k top scoring boxes only
                num_topk = min(pre_nms_topk, topk_idxs.size(0))
                pred_prob, idxs = pred_prob.sort(descending=True)
                pred_prob = pred_prob[:num_topk].clone()
                topk_idxs = topk_idxs[idxs[:num_topk]].clone()

                # 3. gather predicted proposals
                pt_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
                cls_idxs = torch.fmod(topk_idxs, num_classes)

                segments = segments[pt_idxs]
                scores = pred_prob
                labels = cls_idxs
                source_grids = segments[:, 1].detach().cpu()
                if has_explicit_source_grids:
                    source_grids = rpn_meta["source_grids"][i].detach().cpu()[pt_idxs]
                if source_frames is not None:
                    source_frames = source_frames[pt_idxs]

            video_id = metas[i]["video_name"]

            if streaming_safe:
                results_per_video = self._format_streaming_safe_results(
                    segments=segments,
                    scores=scores,
                    labels=labels,
                    source_grids=source_grids,
                    source_frames=source_frames,
                    meta=metas[i],
                    video_id=video_id,
                    post_cfg=post_cfg,
                    ext_cls=ext_cls,
                    score_threshold=pre_nms_thresh,
                    batch_index=i,
                )
                if video_id in results.keys():
                    results[video_id].extend(results_per_video)
                else:
                    results[video_id] = results_per_video
                continue

            # if not sliding window, do nms
            if post_cfg.sliding_window == False and post_cfg.nms is not None:
                segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)

            # convert segments to seconds
            seconds_meta = metas[i]
            if rpn_meta is not None and isinstance(rpn_meta, dict) and rpn_meta.get("proposal_axis") == "dense_grid":
                seconds_meta = dict(metas[i])
                seconds_meta["irregular_native_axis"] = True
            segments = convert_to_seconds(segments, seconds_meta)

            # merge with external classifier
            if isinstance(ext_cls, list):  # own classification results
                labels = [ext_cls[label.item()] for label in labels]
            else:
                segments, labels, scores = ext_cls(video_id, segments, scores)

            results_per_video = []
            for segment, label, score in zip(segments, labels, scores):
                # convert to python scalars
                results_per_video.append(
                    dict(
                        segment=[round(seg.item(), 2) for seg in segment],
                        label=label,
                        score=round(score.item(), 4),
                    )
                )

            if video_id in results.keys():
                results[video_id].extend(results_per_video)
            else:
                results[video_id] = results_per_video

        return results

    def _format_streaming_safe_results(
        self,
        segments,
        scores,
        labels,
        source_grids,
        source_frames,
        meta,
        video_id,
        post_cfg,
        ext_cls,
        score_threshold,
        batch_index=0,
    ):
        fps = float(meta["fps"])
        snippet_stride = int(meta.get("snippet_stride", 1))
        window_start_frame = int(meta.get("window_start_frame", 0))
        offset_frames = int(meta.get("offset_frames", 0))
        emit_frame = meta.get("window_end_frame")
        if emit_frame is None:
            valid_len = int(meta.get("window_size", segments.shape[0]))
            emit_frame = window_start_frame + valid_len * snippet_stride
        emit_frame = int(emit_frame)

        max_latency = float(getattr(post_cfg, "max_latency", 0.0))
        latency_frames = int(getattr(post_cfg, "max_latency_frames", round(max_latency * fps)))
        nms_iou = float(getattr(post_cfg, "streaming_nms_iou", 0.6))
        grid_spec = GridSpec(
            fps=fps,
            snippet_stride=snippet_stride,
            window_start_frame=window_start_frame,
            offset_frames=offset_frames,
        )
        emitter = OnlineEmitter(
            grid_spec=grid_spec,
            score_threshold=score_threshold,
            nms_iou_threshold=nms_iou,
            latency_frames=latency_frames,
        )
        stream_key = make_stream_key(meta, batch_index=batch_index)
        state = self._online_states.get(stream_key)
        if state is None:
            state = OnlineState(video_name=video_id)
            self._online_states[stream_key] = state

        grid_offset = int(round((window_start_frame + offset_frames) / max(snippet_stride, 1)))
        candidates = []
        if source_frames is None:
            source_frames = [None for _ in range(len(source_grids))]
        for segment, score, label, source_grid, source_frame in zip(segments, scores, labels, source_grids, source_frames):
            local_source_grid = float(source_grid.item() if torch.is_tensor(source_grid) else source_grid)
            absolute_source_grid = candidate_source_grid(local_source_grid, grid_offset=grid_offset)
            explicit_source_frame = None
            if source_frame is not None:
                explicit_source_frame = int(source_frame.item() if torch.is_tensor(source_frame) else source_frame)
            else:
                explicit_source_frame = grid_spec.grid_to_frame(local_source_grid)
            candidates.append(
                OnlineCandidate(
                    source_grid=absolute_source_grid,
                    label=int(label.item() if torch.is_tensor(label) else label),
                    score=float(score.item() if torch.is_tensor(score) else score),
                    start_grid=float(segment[0].item()),
                    end_grid=float(segment[1].item()),
                    source_frame=explicit_source_frame,
                )
            )
        emitted = emitter.step(video_name=video_id, now_frame=emit_frame, state=state, candidates=candidates)

        if len(emitted) == 0:
            return []

        def build_row(det, segment, label, score):
            if torch.is_tensor(segment):
                start_sec = float(segment[0].item())
                end_sec = float(segment[1].item())
            else:
                start_sec = float(segment[0])
                end_sec = float(segment[1])
            if torch.is_tensor(label):
                label = int(label.item())
            if torch.is_tensor(score):
                score = float(score.item())
            else:
                score = float(score)
            start_sec = max(0.0, start_sec)
            end_sec = min(duration, end_sec)
            return dict(
                segment=[round(start_sec, 2), round(end_sec, 2)],
                label=label,
                score=round(float(score), 4),
                emit_frame=int(det.emit_frame),
                source_grid=int(det.source_grid),
                source_frame=int(det.source_frame),
                source_grid_contract="absolute_source_grid",
                stream_key=stream_key,
                stream_id=meta.get("stream_id", "default"),
                input_format=meta.get("input_format", "unknown"),
                processor_id=meta.get("processor_id", meta.get("processor", "unknown")),
                encoder_id=meta.get("encoder_id", meta.get("encoder", "unknown")),
                image_size=meta.get("image_size", "unknown"),
                frame_policy=meta.get("frame_policy", "unknown"),
                window_start_frame=int(window_start_frame),
                window_end_frame=int(emit_frame),
                eval_rank=int(meta.get("eval_rank", 0)),
                batch_index=int(batch_index),
                start_frame=int(det.start_frame),
                end_frame=int(det.end_frame),
                latency_sec=round(float(det.latency_sec), 4),
            )

        duration = float(meta.get("duration", max(det.end_frame for det in emitted) / fps))
        emitted_segments = torch.tensor(
            [
                [
                    max(0.0, det.start_frame / fps),
                    min(duration, det.end_frame / fps),
                ]
                for det in emitted
            ],
            dtype=torch.float32,
        )
        emitted_scores = torch.tensor([float(det.score) for det in emitted], dtype=torch.float32)

        results_per_video = []
        if isinstance(ext_cls, list):
            for det, segment in zip(emitted, emitted_segments):
                results_per_video.append(build_row(det, segment, ext_cls[det.label], det.score))
        else:
            for det, segment in zip(emitted, emitted_segments):
                results_per_video.append(build_row(det, segment, det.label, det.score))
        return results_per_video
