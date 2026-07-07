import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..bricks import Scale
from ..builder import HEADS
from .anchor_free_head import AnchorFreeHead
from opentad.utils.online_protocol import make_stream_key


class CausalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, norm=True):
        super().__init__()
        self.left_pad = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=0)
        self.norm = nn.LayerNorm(out_channels, eps=1e-6) if norm else None
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, mask=None):
        x = F.pad(x, (self.left_pad, 0))
        x = self.conv(x)
        if mask is not None:
            x = x * mask.unsqueeze(1).to(dtype=x.dtype)
        if self.norm is not None:
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(x)
        if mask is not None:
            x = x * mask.unsqueeze(1).to(dtype=x.dtype)
        return x, mask


@HEADS.register_module()
class MATRHead(AnchorFreeHead):
    """MATR-style online TAL head with causal start/end/actionness branches.

    This head keeps the OpenTAD anchor-free regression target contract while
    adding online emission signals inspired by MATR's separate start/end
    reasoning. Full paper-level memory queue training can be layered on top of
    this interface without changing detector outputs.
    """

    online = True

    def __init__(
        self,
        num_classes,
        in_channels,
        feat_channels,
        num_convs=3,
        prior_generator=None,
        loss=None,
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        center_sample="radius",
        center_sample_radius=1.5,
        label_smoothing=0,
        cls_prior_prob=0.01,
        loss_weight=1.0,
        kernel_size=3,
        emit_threshold=0.0,
        memory_size=0,
        clamp_end_to_current=True,
        max_future_offset=0.0,
        online_censored_training=False,
        boundary_loss_weight=0.0,
        actionness_loss_weight=0.0,
        emit_loss_weight=0.0,
        boundary_target_radius=1.5,
        use_boundary_scores=False,
        use_actionness_scores=False,
        use_emit_scores=False,
        online=True,
        **kwargs,
    ):
        kwargs.pop("online", None)
        self.online = bool(online)
        self.kernel_size = int(kernel_size)
        self.emit_threshold = float(emit_threshold)
        self.memory_size = int(memory_size)
        self.clamp_end_to_current = bool(clamp_end_to_current)
        self.max_future_offset = float(max_future_offset)
        self.online_censored_training = bool(online_censored_training)
        self.boundary_loss_weight = float(boundary_loss_weight)
        self.actionness_loss_weight = float(actionness_loss_weight)
        self.emit_loss_weight = float(emit_loss_weight)
        self.boundary_target_radius = float(boundary_target_radius)
        self.use_boundary_scores = bool(use_boundary_scores)
        self.use_actionness_scores = bool(use_actionness_scores)
        self.use_emit_scores = bool(use_emit_scores)
        self.stream_memory = None
        self._stream_video_names = None
        self._stream_window_start_frames = None
        self._stream_window_end_frames = None
        self._stream_window_end_by_key = {}
        self._stream_current_keys = None
        self._stream_reuse_flags = None
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            feat_channels=feat_channels,
            num_convs=num_convs,
            cls_prior_prob=cls_prior_prob,
            prior_generator=prior_generator,
            loss=loss,
            loss_normalizer=loss_normalizer,
            loss_normalizer_momentum=loss_normalizer_momentum,
            loss_weight=loss_weight,
            label_smoothing=label_smoothing,
            center_sample=center_sample,
            center_sample_radius=center_sample_radius,
            online_censored_training=online_censored_training,
            online_censored_max_future_offset=max_future_offset,
            **kwargs,
        )

    def reset_stream_state(self):
        self.stream_memory = None
        self._stream_video_names = None
        self._stream_window_start_frames = None
        self._stream_window_end_frames = None
        self._stream_window_end_by_key = {}
        self._stream_current_keys = None
        self._stream_reuse_flags = None

    def _clear_stream_memory_for_key(self, stream_key):
        if self.stream_memory is not None:
            for level_memory in self.stream_memory.values():
                level_memory.pop(stream_key, None)
        self._stream_window_end_by_key.pop(stream_key, None)

    def _infer_window_end_frames(self, metas, mask_list):
        if metas is None:
            return None

        valid_lengths = None
        if mask_list is not None and len(mask_list) > 0:
            valid_lengths = mask_list[0].sum(dim=1).detach().cpu().tolist()

        window_ends = []
        for idx, meta in enumerate(metas):
            if not isinstance(meta, dict):
                window_ends.append(None)
                continue
            if meta.get("window_end_frame") is not None:
                window_ends.append(int(meta["window_end_frame"]))
                continue
            current_start = meta.get("window_start_frame")
            if current_start is None or valid_lengths is None:
                window_ends.append(None)
                continue
            snippet_stride = int(meta.get("snippet_stride", 1))
            window_ends.append(int(current_start) + int(valid_lengths[idx]) * snippet_stride)
        return tuple(window_ends)

    def _maybe_reset_stream_state(self, metas, mask_list=None):
        if self.training or metas is None or self.memory_size <= 0:
            self._stream_current_keys = None
            self._stream_reuse_flags = None
            return

        video_names = tuple(meta.get("video_name") if isinstance(meta, dict) else None for meta in metas)
        window_starts = tuple(meta.get("window_start_frame") if isinstance(meta, dict) else None for meta in metas)
        window_ends = self._infer_window_end_frames(metas, mask_list)
        if window_ends is None:
            self._stream_current_keys = None
            self._stream_reuse_flags = None
            return

        stream_keys = []
        reuse_flags = []
        for idx, (video_name, current_start, current_end) in enumerate(zip(video_names, window_starts, window_ends)):
            stream_key = make_stream_key(metas[idx], batch_index=idx)
            previous_end = self._stream_window_end_by_key.get(stream_key)
            can_reuse = previous_end is not None and current_start is not None and current_start == previous_end
            if (
                previous_end is not None
                and current_start is not None
                and current_start < previous_end
            ):
                self._clear_stream_memory_for_key(stream_key)
            elif previous_end is not None and not can_reuse:
                self._clear_stream_memory_for_key(stream_key)

            stream_keys.append(stream_key)
            reuse_flags.append(bool(can_reuse))
            if current_end is not None:
                self._stream_window_end_by_key[stream_key] = int(current_end)

        self._stream_video_names = video_names
        self._stream_window_start_frames = window_starts
        self._stream_window_end_frames = window_ends
        self._stream_current_keys = tuple(stream_keys)
        self._stream_reuse_flags = tuple(reuse_flags)

    def _append_stream_memory(self, level, feat, mask):
        current_len = feat.shape[-1]
        if self.memory_size <= 0 or self.training:
            return feat, mask, current_len

        if self.stream_memory is None:
            self.stream_memory = {}

        level_memory = self.stream_memory.setdefault(level, {})
        stream_keys = self._stream_current_keys or tuple(f"batch:{idx}" for idx in range(feat.shape[0]))
        reuse_flags = self._stream_reuse_flags or tuple(False for _ in range(feat.shape[0]))

        sample_feats = []
        sample_masks = []
        for idx, stream_key in enumerate(stream_keys):
            current_feat = feat[idx : idx + 1]
            current_mask = mask[idx : idx + 1]
            memory = level_memory.get(stream_key) if reuse_flags[idx] else None
            if memory is not None:
                memory_feat, memory_mask = memory
                memory_feat = memory_feat.to(device=feat.device, dtype=feat.dtype).unsqueeze(0)
                memory_mask = memory_mask.to(device=mask.device, dtype=mask.dtype).unsqueeze(0)
                sample_feats.append(torch.cat([memory_feat, current_feat], dim=-1))
                sample_masks.append(torch.cat([memory_mask, current_mask], dim=-1))
            else:
                sample_feats.append(current_feat)
                sample_masks.append(current_mask)

        context_len = max(sample_feat.shape[-1] for sample_feat in sample_feats)
        feat_context = feat.new_zeros((feat.shape[0], feat.shape[1], context_len))
        mask_context = mask.new_zeros((mask.shape[0], context_len))
        for idx, (sample_feat, sample_mask) in enumerate(zip(sample_feats, sample_masks)):
            feat_context[idx : idx + 1, :, -sample_feat.shape[-1] :] = sample_feat
            mask_context[idx : idx + 1, -sample_mask.shape[-1] :] = sample_mask
            level_memory[stream_keys[idx]] = (
                sample_feat.detach().squeeze(0)[..., -self.memory_size :],
                sample_mask.detach().squeeze(0)[..., -self.memory_size :],
            )
        return feat_context, mask_context, current_len

    def _init_cls_convs(self):
        self.cls_convs = nn.ModuleList()
        for idx in range(self.num_convs):
            self.cls_convs.append(
                CausalConvBlock(
                    self.in_channels if idx == 0 else self.feat_channels,
                    self.feat_channels,
                    kernel_size=self.kernel_size,
                )
            )

    def _init_reg_convs(self):
        self.reg_convs = nn.ModuleList()
        for idx in range(self.num_convs):
            self.reg_convs.append(
                CausalConvBlock(
                    self.in_channels if idx == 0 else self.feat_channels,
                    self.feat_channels,
                    kernel_size=self.kernel_size,
                )
            )

    def _init_heads(self):
        self.cls_head = nn.Conv1d(self.feat_channels, self.num_classes, kernel_size=1)
        self.reg_head = nn.Conv1d(self.feat_channels, 2, kernel_size=1)
        self.start_head = nn.Conv1d(self.feat_channels, self.num_classes, kernel_size=1)
        self.end_head = nn.Conv1d(self.feat_channels, self.num_classes, kernel_size=1)
        self.actionness_head = nn.Conv1d(self.feat_channels, 1, kernel_size=1)
        self.emit_head = nn.Conv1d(self.feat_channels, 1, kernel_size=1)
        self.scale = nn.ModuleList([Scale() for _ in range(len(self.prior_generator.strides))])

        if self.cls_prior_prob > 0:
            bias_value = -(math.log((1 - self.cls_prior_prob) / self.cls_prior_prob))
            nn.init.constant_(self.cls_head.bias, bias_value)
            nn.init.constant_(self.start_head.bias, bias_value)
            nn.init.constant_(self.end_head.bias, bias_value)

    def _predict_levels(self, feat_list, mask_list):
        cls_pred, reg_pred = [], []
        start_pred, end_pred = [], []
        actionness_pred, emit_pred = [], []

        for level, (feat, mask) in enumerate(zip(feat_list, mask_list)):
            feat, mask, current_len = self._append_stream_memory(level, feat, mask)
            cls_feat = feat
            reg_feat = feat
            for idx in range(self.num_convs):
                cls_feat, mask = self.cls_convs[idx](cls_feat, mask)
                reg_feat, mask = self.reg_convs[idx](reg_feat, mask)

            cls_pred.append(self.cls_head(cls_feat)[..., -current_len:])
            reg_pred.append(F.relu(self.scale[level](self.reg_head(reg_feat)))[..., -current_len:])
            start_pred.append(self.start_head(cls_feat)[..., -current_len:])
            end_pred.append(self.end_head(cls_feat)[..., -current_len:])
            actionness_pred.append(self.actionness_head(reg_feat)[..., -current_len:])
            emit_pred.append(self.emit_head(reg_feat)[..., -current_len:])
        return cls_pred, reg_pred, start_pred, end_pred, actionness_pred, emit_pred

    def _build_online_branch_targets(self, points, mask_list, gt_segments, gt_labels):
        concat_points = torch.cat(points, dim=0)
        centers = concat_points[:, 0]
        strides = concat_points[:, 3].clamp(min=1.0)
        valid_mask = torch.cat(mask_list, dim=1)
        batch_size, num_points = valid_mask.shape

        start_target = centers.new_zeros((batch_size, num_points, self.num_classes))
        end_target = centers.new_zeros((batch_size, num_points, self.num_classes))
        actionness_target = centers.new_zeros((batch_size, num_points, 1))
        emit_target = centers.new_zeros((batch_size, num_points, 1))

        for batch_idx, (segments, labels) in enumerate(zip(gt_segments, gt_labels)):
            if segments is None or len(segments) == 0:
                continue
            segments = segments.to(device=centers.device, dtype=centers.dtype)
            labels = labels.to(device=centers.device, dtype=torch.long)
            for segment, label in zip(segments, labels):
                label = label.clamp(min=0, max=self.num_classes - 1)
                start_mask = (centers - segment[0]).abs() <= self.boundary_target_radius * strides
                end_mask = (centers - segment[1]).abs() <= self.boundary_target_radius * strides
                inside_mask = torch.logical_and(centers >= segment[0], centers <= segment[1])
                if self.online_censored_training:
                    endpoint_observed = segment[1] <= centers + self.max_future_offset + 1e-6
                    end_mask = torch.logical_and(end_mask, endpoint_observed)

                start_target[batch_idx, start_mask, label] = 1.0
                end_target[batch_idx, end_mask, label] = 1.0
                actionness_target[batch_idx, inside_mask, 0] = 1.0
                emit_target[batch_idx, end_mask, 0] = 1.0

        return start_target, end_target, actionness_target, emit_target, valid_mask

    @staticmethod
    def _masked_bce_with_logits(pred, target, valid_mask):
        valid_mask = valid_mask.unsqueeze(-1).expand_as(target)
        if not valid_mask.any():
            return pred.float().sum() * 0
        return F.binary_cross_entropy_with_logits(pred[valid_mask], target[valid_mask], reduction="mean")

    def _online_branch_losses(
        self,
        points,
        mask_list,
        gt_segments,
        gt_labels,
        start_pred,
        end_pred,
        actionness_pred,
        emit_pred,
    ):
        start_target, end_target, actionness_target, emit_target, valid_mask = self._build_online_branch_targets(
            points, mask_list, gt_segments, gt_labels
        )
        start_logits = torch.cat(start_pred, dim=-1).permute(0, 2, 1)
        end_logits = torch.cat(end_pred, dim=-1).permute(0, 2, 1)
        actionness_logits = torch.cat(actionness_pred, dim=-1).permute(0, 2, 1)
        emit_logits = torch.cat(emit_pred, dim=-1).permute(0, 2, 1)

        losses = {}
        if self.boundary_loss_weight > 0:
            boundary_loss = 0.5 * (
                self._masked_bce_with_logits(start_logits, start_target, valid_mask)
                + self._masked_bce_with_logits(end_logits, end_target, valid_mask)
            )
            losses["boundary_loss"] = boundary_loss * self.boundary_loss_weight
        if self.actionness_loss_weight > 0:
            losses["actionness_loss"] = (
                self._masked_bce_with_logits(actionness_logits, actionness_target, valid_mask)
                * self.actionness_loss_weight
            )
        if self.emit_loss_weight > 0:
            losses["emit_loss"] = self._masked_bce_with_logits(emit_logits, emit_target, valid_mask) * self.emit_loss_weight
        return losses

    @staticmethod
    def _selected_axis_to_dense_axis(coords, positions, valid_len):
        xp = torch.arange(positions.numel(), dtype=coords.dtype, device=coords.device)
        xp = torch.cat([xp, xp.new_tensor([float(positions.numel())])], dim=0)
        fp = torch.cat([positions, positions.new_tensor([float(valid_len)])], dim=0)
        coord_shape = coords.shape
        coord_flat = coords.reshape(-1).clamp(min=0.0, max=float(positions.numel()))
        right_idx = torch.searchsorted(xp, coord_flat, right=True).clamp(min=1, max=xp.numel() - 1)
        left_idx = right_idx - 1
        x0 = xp[left_idx]
        x1 = xp[right_idx]
        y0 = fp[left_idx]
        y1 = fp[right_idx]
        weight = (coord_flat - x0) / (x1 - x0).clamp(min=1e-6)
        return (y0 + weight * (y1 - y0)).reshape(coord_shape)

    def _extract_shared_irregular_axis(self, metas):
        if metas is None or len(metas) == 0 or not isinstance(metas[0], dict):
            return None
        if metas[0].get("irregular_native_axis", False):
            return None

        positions = metas[0].get("irregular_selected_positions", None)
        valid_len = metas[0].get("irregular_selected_valid_len", None)
        if positions is None or valid_len is None:
            return None

        positions = [int(pos) for pos in positions]
        valid_len = int(valid_len)
        if len(metas) > 1:
            raise ValueError("adaptive irregular MATRHead currently requires batch_size=1")
        for meta in metas[1:]:
            if not isinstance(meta, dict) or meta.get("irregular_native_axis", False):
                raise ValueError("MATRHead irregular selected-axis metadata must be shared by every batch item")
            other_positions = meta.get("irregular_selected_positions", None)
            other_valid_len = meta.get("irregular_selected_valid_len", None)
            if other_positions is None or other_valid_len is None:
                raise ValueError("MATRHead got partial irregular selected-axis metadata in a batch")
            if [int(pos) for pos in other_positions] != positions or int(other_valid_len) != valid_len:
                raise ValueError(
                    "MATRHead irregular selected-axis currently requires identical selected positions per batch; "
                    "use batch_size=1 for adaptive online routes"
                )
        return positions, valid_len

    def _build_irregular_points(self, feat_list, metas):
        axis = self._extract_shared_irregular_axis(metas)
        if axis is None:
            return None

        positions, valid_len = axis
        if len(positions) == 0:
            return None

        pts_list = []
        for level, feat in enumerate(feat_list):
            length = feat.shape[-1]
            stride = float(self.prior_generator.strides[level])
            coords = torch.arange(length, dtype=torch.float32, device=feat.device) * stride
            if getattr(self.prior_generator, "use_offset", False):
                coords = coords + 0.5 * stride

            pos_tensor = torch.as_tensor(positions, dtype=torch.float32, device=feat.device)
            centers = self._selected_axis_to_dense_axis(coords, pos_tensor, valid_len)
            next_centers = self._selected_axis_to_dense_axis(coords + stride, pos_tensor, valid_len)
            prev_centers = self._selected_axis_to_dense_axis((coords - stride).clamp(min=0.0), pos_tensor, valid_len)
            right_scale = (next_centers - centers).clamp(min=1e-6)
            left_scale = torch.where(coords > 0, centers - prev_centers, right_scale).clamp(min=1e-6)
            point_scale = (0.5 * (left_scale + right_scale)).clamp(min=1e-6)

            reg_range = torch.as_tensor(
                self.prior_generator.regression_range[level],
                dtype=torch.float32,
                device=feat.device,
            )
            reg_range = reg_range[None].repeat(length, 1)
            pts_list.append(torch.cat((centers[:, None], reg_range, point_scale[:, None]), dim=1))
        return pts_list

    def _build_points(self, feat_list, metas=None):
        irregular_points = self._build_irregular_points(feat_list, metas)
        if irregular_points is not None:
            return irregular_points, "dense_grid"
        return self.prior_generator(feat_list), "selected_axis"

    def forward_train(self, feat_list, mask_list, gt_segments, gt_labels, metas=None, **kwargs):
        cls_pred, reg_pred, start_pred, end_pred, actionness_pred, emit_pred = self._predict_levels(feat_list, mask_list)
        points, _ = self._build_points(feat_list, metas=metas)
        losses = self.losses(cls_pred, reg_pred, mask_list, points, gt_segments, gt_labels)
        losses.update(
            self._online_branch_losses(
                points,
                mask_list,
                gt_segments,
                gt_labels,
                start_pred,
                end_pred,
                actionness_pred,
                emit_pred,
            )
        )
        return losses

    def forward_test(self, feat_list, mask_list, metas=None, **kwargs):
        self._maybe_reset_stream_state(metas, mask_list)
        cls_pred, reg_pred, start_pred, end_pred, actionness_pred, emit_pred = self._predict_levels(feat_list, mask_list)
        points, proposal_axis = self._build_points(feat_list, metas=metas)
        return self.get_valid_proposals_scores(
            points,
            reg_pred,
            cls_pred,
            mask_list,
            actionness_pred=actionness_pred,
            emit_pred=emit_pred,
            start_pred=start_pred,
            end_pred=end_pred,
            proposal_axis=proposal_axis,
        )

    def get_valid_proposals_scores(
        self,
        points,
        reg_pred,
        cls_pred,
        mask_list,
        actionness_pred=None,
        emit_pred=None,
        start_pred=None,
        end_pred=None,
        proposal_axis="selected_axis",
    ):
        proposals = self.get_refined_proposals(points, reg_pred)
        point_centers = torch.cat(points, dim=0)[:, 0][None].to(device=proposals.device, dtype=proposals.dtype)
        if self.clamp_end_to_current:
            max_end = point_centers + self.max_future_offset
            proposals = proposals.clone()
            proposals[..., 1] = torch.minimum(proposals[..., 1], max_end)
            proposals[..., 0] = torch.minimum(proposals[..., 0], proposals[..., 1])

        scores = torch.cat(cls_pred, dim=-1).permute(0, 2, 1).sigmoid()

        if self.use_boundary_scores and start_pred is not None:
            start_score = torch.cat(start_pred, dim=-1).permute(0, 2, 1).sigmoid()

        if self.use_boundary_scores and end_pred is not None:
            end_score = torch.cat(end_pred, dim=-1).permute(0, 2, 1).sigmoid()
            scores = scores * end_score
        elif self.use_boundary_scores and start_pred is not None:
            # Same-point start and end multiplication suppresses long actions.
            # Use start score only for start-only ablations; normal online
            # emission should be gated by the current/end side.
            scores = scores * start_score

        if self.use_actionness_scores and actionness_pred is not None:
            actionness = torch.cat(actionness_pred, dim=-1).permute(0, 2, 1).sigmoid()
            scores = scores * actionness

        if self.use_emit_scores and emit_pred is not None:
            emit = torch.cat(emit_pred, dim=-1).permute(0, 2, 1).sigmoid()
            if self.emit_threshold > 0:
                emit = emit * (emit >= self.emit_threshold).to(dtype=emit.dtype)
            scores = scores * emit

        masks = torch.cat(mask_list, dim=1)
        new_proposals, new_scores = [], []
        new_source_grids = []
        source_grids = point_centers.expand(proposals.shape[0], -1)
        for proposal, score, mask in zip(proposals, scores, masks):
            new_proposals.append(proposal[mask])
            new_scores.append(score[mask])
        for source_grid, mask in zip(source_grids, masks):
            new_source_grids.append(source_grid[mask])
        return new_proposals, new_scores, {"source_grids": new_source_grids, "proposal_axis": proposal_axis}
