import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment as linear_assignment
from .matcher import HungarianMatcher

class CrossEntropyLoss(nn.Module):
    def __init__(self, focal=False, weight=None, reduce=True):
        super(CrossEntropyLoss, self).__init__()
        self.focal = focal
        self.weight= weight
        self.reduce = reduce

    def forward(self, input, target, alpha=0.25, gamma=2):
        #IN: input: unregularized logits [B, C] target: multi-hot representaiton [B, C]
        logsoftmax = nn.LogSoftmax(dim=1).to(input.device)
        if not self.focal:
            if self.weight is None:
                output = torch.sum(-target * logsoftmax(input), 1)
            else:
                output = torch.sum(-target * logsoftmax(input) /self.weight, 1)
        else:
            softmax = nn.Softmax(dim=1).to(input.device)
            p = softmax(input)
            output = alpha*torch.sum(-target * (1 - p)**gamma * logsoftmax(input), 1)
            
        if self.reduce:
            return torch.mean(output)
        else:
            return output

def ctr_diou_loss_1d(
    input_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    reduction: str = 'none',
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Distance-IoU Loss (Zheng et. al)
    https://arxiv.org/abs/1911.08287

    This is an implementation that assumes a 1D event is represented using
    the same center point with different offsets, e.g.,
    (t1, t2) = (c - o_1, c + o_2) with o_i >= 0

    Reference code from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/giou_loss.py

    Args:
        input/target_offsets (Tensor): 1D offsets of size (N, 2)
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()

    sp, ep = -input_offsets[:, 0], input_offsets[:, 1]
    sg, eg = -target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    skis = torch.max(sp, sg)
    ekis = torch.min(ep, eg)
    
    # iou
    intsctk = ekis - skis
    unionk = (ep-sp) + (eg - sg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)
    
    # smallest enclosing box
    sc = torch.min(sp, sg)
    ec = torch.max(ep, eg)
    len_c = ec - sc
    
    # offset between centers
    rho = abs(0.5 * (ep+sp-eg-sg))
    
    # diou
    loss = 1.0 - iouk + torch.square(rho / len_c.clamp(min=eps))

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss
    
class CriterionMATR(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, losses, args=None):
        """ Create the criterion.
        Parameters:
        num_classes: number of object categories, omitting the special no-object category
        weight_dict: dict containing as key the names of the losses and as values their relative weight.
        losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.losses = losses
        self.reduce = args.reduce
        self.segment_size = args.num_frame
        self.reduction = 'mean' if self.reduce == 1 else 'none'
        self.matcher = matcher
        self.use_flag = args.use_flag
        self.num_queries = args.num_queries
        self.anti_len = args.anti_len
        self.max_memory_len = args.max_memory_len

        self.event_birth_mode = getattr(
            args, "birth_mode", getattr(args, "event_birth_mode", "matr_delayed")
        )
        self.event_ownership_mode = getattr(
            args,
            "ownership_mode",
            getattr(args, "event_owner_mode", "fresh_rematch"),
        )
        self.model_variant = getattr(args, "model_variant", None)
        if self.model_variant is None:
            self.model_variant = (
                "eventmatr" if getattr(args, "event_arm", None) else "native_matr"
            )
        if self.model_variant not in {"native_matr", "eventmatr"}:
            raise ValueError("invalid model_variant: {}".format(self.model_variant))
        if self.model_variant == "native_matr" and getattr(args, "event_arm", None):
            raise ValueError("native_matr must not carry an EventMATR event_arm")
        self.event_enabled = self.model_variant == "eventmatr"
        self.event_lifecycle_version = getattr(
            args, "event_lifecycle_version", "v1_dense"
        )
        if self.event_lifecycle_version not in {"v1_dense", "d1_censored"}:
            raise ValueError("invalid event_lifecycle_version")
        self.event_d1_enabled = (
            self.event_enabled and self.event_lifecycle_version == "d1_censored"
        )
        self.event_d1_lane = getattr(args, "event_d1_lane", "th")
        if self.event_d1_lane not in {"r", "t", "h", "th"}:
            raise ValueError("invalid event_d1_lane")
        self.event_d1_use_identity = self.event_d1_lane in {"t", "th"}
        self.event_d1_use_hazard = self.event_d1_lane in {"h", "th"}
        if self.event_enabled:
            self.weight_dict['loss_event_birth'] = float(
                getattr(args, "event_birth_coef", 1.0)
            )
            self.weight_dict['loss_event_start_offset'] = float(
                getattr(args, "event_start_offset_coef", 1.0)
            )
            self.weight_dict['loss_event_alive'] = float(
                getattr(args, "event_alive_coef", 1.0)
            )
            self.weight_dict['loss_event_end'] = float(
                getattr(args, "event_end_coef", 1.0)
            )
            self.weight_dict['loss_event_end_offset'] = float(
                getattr(args, "event_end_offset_coef", 1.0)
            )
            self.weight_dict['loss_event_owner_class'] = float(
                getattr(args, "event_owner_class_coef", 1.0)
            )
            self.weight_dict['loss_event_owner_state'] = float(
                getattr(args, "event_owner_state_coef", 1.0)
            )
            if self.event_d1_enabled and self.event_d1_use_identity:
                self.weight_dict['loss_event_identity'] = float(
                    getattr(args, "event_identity_coef", 1.0)
                )
            self._event_owner_by_video = {}
            self._event_last_frame_by_video = {}
        
        empty_weight = None
        if args.use_empty_weight:
            empty_weight = torch.ones(self.num_classes)
            empty_weight[-1] = args.eos_coef
                
        self.register_buffer('empty_weight', empty_weight)
        self.cls_loss_func = CrossEntropyLoss(focal=args.use_focal, weight=empty_weight, reduce=self.reduce)
    
    def sel_regoffset(self, fg_src_regs, fg_src_stcls):
        fg_src_stsel = torch.argmax(fg_src_stcls, dim=1)
        fg_src_stregs = fg_src_regs[:,:self.max_memory_len+2]
        fg_src_streg = torch.gather(fg_src_stregs, 1, fg_src_stsel.unsqueeze(-1)).squeeze()
        fg_src_edreg = fg_src_regs[:,-1].squeeze()
        fg_src_regs = torch.stack([fg_src_streg, fg_src_edreg], dim=-1)
        return fg_src_regs
    
    def edrstc2r2offset(self, regs, stcls):
        stsel = torch.argmax(stcls, dim=1)
        st_offset = (regs[:,0]+stsel)*self.segment_size
        ed_offset = regs[:,1]*self.segment_size
        return torch.stack([st_offset, -ed_offset], dim=-1)
    
    def loss_cls(self, outputs, targets, infos, indices, log=True):
        """ classification loss """
        assert 'pred_cls' in outputs
        src_logits = outputs['pred_cls'] # batch x num_queries x num_class
        bs, num_queries, _ = src_logits.shape
        targets = torch.cat([targets['cls_label'][b][indices[b][1]] for b in range(bs)]).reshape(bs, num_queries, -1)
        size = targets.size()
        src_logits = src_logits.reshape(-1, src_logits.size(-1))
        targets = targets.reshape(-1, targets.size(-1))
            
        loss = self.cls_loss_func(src_logits, targets)
        
        if not self.reduce:
            loss = loss.reshape(size[:-1])
            
        if (loss.isnan()):
            loss = torch.tensor([0.0], requires_grad=True).to(self.device)
        
        losses = {'loss_cls': loss}
        
        if self.use_flag:
            src_logits = outputs['pred_flag'].squeeze() # batch
            targets = infos['segment_flag'].float().to(self.device) # batch
            loss = F.binary_cross_entropy(src_logits, targets, reduction=self.reduction)
            losses['loss_flag'] = loss
        return losses

    def loss_reg(self, outputs, targets, infos, indices, log=True):
        """ regression loss """
        assert 'pred_reg' in outputs
        src_regs = outputs['pred_reg']
        bs, num_queries, _ = src_regs.shape
        src_stcls = outputs['pred_stcls']
        
        tgt_regs = torch.cat([targets['reg_label'][b][indices[b][1]] for b in range(bs)]).reshape(bs, num_queries, -1)
        tgt_stcls = torch.cat([targets['stcls_label'][b][indices[b][1]] for b in range(bs)]).reshape(bs, num_queries, -1)
        
        src_regs = src_regs.reshape(-1, src_regs.size(-1))
        src_stcls = src_stcls.reshape(-1, src_stcls.size(-1))
        tgt_regs = tgt_regs.reshape(-1, tgt_regs.size(-1))
        tgt_stcls = tgt_stcls.reshape(-1, tgt_stcls.size(-1))
        
        bgmask = tgt_regs[:,1] < -1e3
        fg_src_regs, bg_src_regs = src_regs[~bgmask], src_regs[bgmask]
        fg_src_stcls, bg_src_stcls = src_stcls[~bgmask], src_stcls[bgmask]
        fg_tgt_regs, bg_tgt_regs = tgt_regs[~bgmask], tgt_regs[bgmask]
        fg_tgt_stcls, bg_tgt_stcls = tgt_stcls[~bgmask], tgt_stcls[bgmask]
        
        loss_stcls = self.cls_loss_func(fg_src_stcls, fg_tgt_stcls)
        
        fg_src_regs = self.sel_regoffset(fg_src_regs, fg_src_stcls)
        fg_src_regs, fg_tgt_regs = fg_src_regs.reshape(-1,2), fg_tgt_regs.reshape(-1,2)
        loss_l1 = F.l1_loss(fg_src_regs, fg_tgt_regs)
        fg_src_offsets = self.edrstc2r2offset(fg_src_regs, fg_src_stcls)
        fg_tgt_offsets = self.edrstc2r2offset(fg_tgt_regs, fg_tgt_stcls)
        
        loss_diou = ctr_diou_loss_1d(fg_src_offsets, fg_tgt_offsets, reduction=self.reduction)            
                
        if(loss_l1.isnan()):
            loss_l1 = torch.tensor([0.0], requires_grad=True).to(self.device)

        if(loss_diou.isnan()):
            loss_diou = torch.tensor([0.0], requires_grad=True).to(self.device)
            
        losses = {'loss_reg_l1': loss_l1,
                  'loss_reg_diou': loss_diou}
        if(loss_stcls.isnan()):
            loss_stcls = torch.tensor([0.0], requires_grad=True).to(self.device)

        losses['loss_reg_stcls'] = loss_stcls
        return losses

    @staticmethod
    def _state_margin(state_logits, state_index):
        competing = torch.cat(
            (
                state_logits[..., :state_index],
                state_logits[..., state_index + 1 :],
            ),
            dim=-1,
        )
        return state_logits[..., state_index] - competing.max(dim=-1).values

    @staticmethod
    def _info_values(values, batch_size):
        if torch.is_tensor(values):
            values = values.detach().cpu().reshape(-1).tolist()
        elif not isinstance(values, (list, tuple)):
            values = [values]
        if len(values) != batch_size:
            raise ValueError("info field does not match EventMATR batch size")
        return list(values)

    def _reset_event_supervision_if_needed(self, video_name, current_frame):
        previous = self._event_last_frame_by_video.get(video_name)
        if previous is not None and current_frame <= previous:
            self._event_owner_by_video.pop(video_name, None)
        self._event_last_frame_by_video[video_name] = current_frame

    def loss_event(self, outputs, targets, infos):
        """Prefix-visible supervision shared by every EventMATR BxO cell."""
        if self.event_d1_enabled:
            return self._loss_event_d1(outputs, targets, infos)
        required = {
            'event_state_logits',
            'event_birth_logits',
            'event_alive_logits',
            'event_end_logits',
            'event_end_offsets',
            'event_candidate_start_frames',
            'event_owner_state_logits',
            'event_owner_end_offsets',
            'event_owner_class_logits',
        }
        missing = sorted(required.difference(outputs))
        if missing:
            raise KeyError("missing EventMATR outputs: {}".format(missing))
        if 'event_targets' not in targets or 'event_valid_mask' not in targets:
            raise KeyError("EventMATR arms require prefix-visible dataset targets")

        state_logits = outputs['event_state_logits']
        batch_size, num_queries, _ = state_logits.shape
        device = state_logits.device
        event_targets = targets['event_targets'].to(device)
        event_valid = targets['event_valid_mask'].to(device).bool()
        if event_targets.shape != (batch_size, num_queries, 8):
            raise ValueError(
                "event_targets must be [B,Q,8], got {}".format(
                    tuple(event_targets.shape)
                )
            )

        birth_target = torch.zeros(
            (batch_size, num_queries), device=device, dtype=state_logits.dtype
        )
        alive_target = torch.zeros_like(birth_target)
        end_target = torch.zeros_like(birth_target)
        start_target = torch.zeros_like(birth_target)
        end_offset_target = torch.zeros_like(birth_target)
        owner_class_target = torch.full(
            (batch_size, num_queries), -100, device=device, dtype=torch.long
        )
        owner_state_target = torch.zeros(
            (batch_size, num_queries), device=device, dtype=torch.long
        )

        video_names = [str(value) for value in self._info_values(
            infos['video_name'], batch_size
        )]
        current_frames = [float(value) for value in self._info_values(
            infos['current_frame'], batch_size
        )]
        real_prefixes = [bool(value) for value in self._info_values(
            infos.get('is_real_prefix', [True] * batch_size), batch_size
        )]

        owner_assignments = 0
        for batch_index, (video_name, current_frame) in enumerate(
            zip(video_names, current_frames)
        ):
            if not real_prefixes[batch_index]:
                continue
            self._reset_event_supervision_if_needed(video_name, current_frame)
            rows = event_targets[batch_index][event_valid[batch_index]]
            if rows.numel() == 0:
                continue

            if self.event_ownership_mode == "sticky_owner":
                owner_map = self._event_owner_by_video.setdefault(video_name, {})
                new_rows = [
                    row for row in rows if int(row[0].item()) not in owner_map
                ]
                if new_rows:
                    occupied = set(owner_map.values())
                    assigned = self.matcher.match_event_owners(
                        state_logits[batch_index],
                        outputs['pred_cls'][batch_index],
                        [int(row[1].item()) for row in new_rows],
                        state_index=1,
                        occupied_queries=occupied,
                    )
                    for row, query_index in zip(new_rows, assigned):
                        owner_map[int(row[0].item())] = int(query_index)
                        owner_assignments += 1
                assignments = [owner_map[int(row[0].item())] for row in rows]
            else:
                assignments = [-1] * len(rows)
                occupied = set()
                # Fresh O0 association is recomputed from the current prefix.
                for state_index, column in ((3, 7), (1, 5), (2, 6)):
                    selected = [
                        index
                        for index, row in enumerate(rows)
                        if bool(row[column].item())
                    ]
                    if not selected:
                        continue
                    matched = self.matcher.match_event_owners(
                        state_logits[batch_index],
                        outputs['pred_cls'][batch_index],
                        [int(rows[index, 1].item()) for index in selected],
                        state_index=state_index,
                        occupied_queries=occupied,
                    )
                    for row_index, query_index in zip(selected, matched):
                        assignments[row_index] = int(query_index)
                        occupied.add(int(query_index))
                        owner_assignments += 1
                if any(query_index < 0 for query_index in assignments):
                    raise RuntimeError("fresh EventMATR association was incomplete")

            for row, query_index in zip(rows, assignments):
                is_birth = bool(row[5].item())
                is_alive = bool(row[6].item())
                is_end = bool(row[7].item())
                if int(is_birth) + int(is_alive) + int(is_end) != 1:
                    raise RuntimeError(
                        "each EventMATR supervision row must have exactly one "
                        "START/ALIVE/END state"
                    )
                owner_class_target[batch_index, query_index] = int(row[1].item())
                if is_birth:
                    birth_target[batch_index, query_index] = 1.0
                    start_target[batch_index, query_index] = row[2]
                    owner_state_target[batch_index, query_index] = 1
                if is_alive:
                    alive_target[batch_index, query_index] = 1.0
                    owner_state_target[batch_index, query_index] = 2
                if is_end:
                    end_target[batch_index, query_index] = 1.0
                    owner_state_target[batch_index, query_index] = 3
                    end_offset_target[batch_index, query_index] = (
                        current_frame - float(row[3].item())
                    ) / self.segment_size

            if self.event_ownership_mode == "sticky_owner":
                owner_map = self._event_owner_by_video[video_name]
                for row in rows:
                    if bool(row[7].item()):
                        owner_map.pop(int(row[0].item()), None)

        owner_logits = outputs['event_owner_state_logits']
        owner_end_offsets = outputs['event_owner_end_offsets']
        owner_class_logits = outputs['event_owner_class_logits']

        real_query_mask = torch.tensor(
            real_prefixes, device=device, dtype=torch.bool
        ).unsqueeze(1).expand(-1, num_queries)

        def masked_bce(logits, labels):
            if real_query_mask.any():
                return F.binary_cross_entropy_with_logits(
                    logits[real_query_mask], labels[real_query_mask]
                )
            return logits.sum() * 0.0

        losses = {
            'loss_event_alive': masked_bce(
                outputs['event_alive_logits'], alive_target
            ),
            'loss_event_end': masked_bce(
                outputs['event_end_logits'], end_target
            ),
        }
        if real_query_mask.any():
            losses['loss_event_owner_state'] = F.cross_entropy(
                owner_logits[real_query_mask], owner_state_target[real_query_mask]
            )
        else:
            losses['loss_event_owner_state'] = owner_logits.sum() * 0.0
        class_positive = owner_class_target >= 0
        if class_positive.any():
            losses['loss_event_owner_class'] = F.cross_entropy(
                owner_class_logits[class_positive],
                owner_class_target[class_positive],
            )
        else:
            losses['loss_event_owner_class'] = owner_class_logits.sum() * 0.0
        end_positive = end_target.bool()
        if end_positive.any():
            losses['loss_event_end_offset'] = 0.5 * (
                F.smooth_l1_loss(
                    outputs['event_end_offsets'][end_positive],
                    end_offset_target[end_positive],
                )
                + F.smooth_l1_loss(
                    owner_end_offsets[end_positive],
                    end_offset_target[end_positive],
                )
            )
        else:
            losses['loss_event_end_offset'] = (
                outputs['event_end_offsets'].sum() + owner_end_offsets.sum()
            ) * 0.0

        losses['loss_event_birth'] = masked_bce(
            outputs['event_birth_logits'], birth_target
        )
        birth_positive = birth_target.bool()
        if birth_positive.any():
            losses['loss_event_start_offset'] = F.smooth_l1_loss(
                outputs['event_candidate_start_frames'][birth_positive]
                / self.segment_size,
                start_target[birth_positive] / self.segment_size,
            )
        else:
            losses['loss_event_start_offset'] = (
                outputs['event_candidate_start_frames'].sum() * 0.0
            )

        losses.update(
            {
                'event_birth_positive_count': birth_target.sum().detach(),
                'event_alive_positive_count': alive_target.sum().detach(),
                'event_end_positive_count': end_target.sum().detach(),
                'event_owner_assignment_count': torch.tensor(
                    float(owner_assignments), device=device
                ),
            }
        )
        return losses

    def _loss_event_d1(self, outputs, targets, infos):
        """Event-normalized censored hazards on chronological ragged tracks."""

        required = {
            "event_state_logits",
            "event_birth_logits",
            "event_candidate_start_frames",
            "event_query_features",
            "event_ragged_state_logits",
            "event_ragged_state_targets",
            "event_ragged_end_offsets",
            "event_ragged_end_targets",
            "event_ragged_class_logits",
            "event_ragged_class_targets",
            "event_ragged_embeddings",
            "event_ragged_group_keys",
            "event_ragged_sources",
            "event_birth_risk_groups",
            "event_association_rows",
        }
        missing = sorted(required.difference(outputs))
        if missing:
            raise KeyError("missing D1 EventMATR outputs: {}".format(missing))
        if "event_targets" not in targets or "event_valid_mask" not in targets:
            raise KeyError("D1 requires prefix-visible event targets")

        state_logits = outputs["event_state_logits"]
        birth_logits = outputs["event_birth_logits"]
        query_features = outputs["event_query_features"]
        class_logits = outputs["pred_cls"]
        device = state_logits.device
        dtype = state_logits.dtype
        batch_size = state_logits.size(0)
        event_targets = targets["event_targets"].to(device)
        event_valid = targets["event_valid_mask"].to(device).bool()
        real_prefixes = [
            bool(value)
            for value in self._info_values(
                infos.get("is_real_prefix", [True] * batch_size), batch_size
            )
        ]
        zero = state_logits.sum() * 0.0

        birth_event_losses = []
        balanced_background_losses = []
        start_losses = []
        assignment_count = 0
        positive_assignments = {}
        birth_risk_groups = list(outputs["event_birth_risk_groups"])
        for group in birth_risk_groups:
            birth_index = int(group["batch_index"])
            if birth_index < 0 or birth_index >= batch_size:
                raise RuntimeError("birth risk group batch index is invalid")
            if not real_prefixes[birth_index]:
                raise RuntimeError("padding prefix cannot own a birth risk group")
            selected = group["selected_logits"]
            selected_frames = group["frames"]
            if selected.ndim != 1 or selected_frames.shape != selected.shape:
                raise RuntimeError("birth risk logits/frames are inconsistent")
            if selected.numel() == 0:
                raise RuntimeError("birth risk group cannot be empty")
            start_frame = float(group["start_frame"])
            terminal_query = int(group["terminal_query"])
            assignment_count += 1
            positive_assignments.setdefault(birth_index, set()).add(
                terminal_query
            )
            if self.event_d1_use_hazard:
                interval_left = float(np.floor(start_frame))
                interval_right = float(np.ceil(start_frame))
                pre_birth = selected_frames < interval_left
                in_interval = (selected_frames >= interval_left) & (
                    selected_frames <= interval_right
                )
                if not bool(in_interval.any().item()):
                    # The retained causal window may begin at the observed
                    # crossing after an explicit truncated-BPTT boundary.
                    in_interval[-1] = True
                    pre_birth[-1] = False
                pre_survival_nll = F.softplus(selected[pre_birth]).sum()
                log_interval_survival = F.logsigmoid(
                    -selected[in_interval]
                ).sum()
                interval_event_probability = (
                    -torch.expm1(log_interval_survival)
                ).clamp_min(1e-8)
                birth_event_losses.append(
                    pre_survival_nll - interval_event_probability.log()
                )
            else:
                birth_event_losses.append(F.softplus(-selected[-1]))
            start_losses.append(
                F.smooth_l1_loss(
                    outputs["event_candidate_start_frames"][
                        birth_index, terminal_query
                    ]
                    / self.segment_size,
                    torch.as_tensor(
                        start_frame,
                        device=device,
                        dtype=outputs["event_candidate_start_frames"].dtype,
                    )
                    / self.segment_size,
                )
            )

        # One hardest currently-unassigned query per real prefix supplies
        # right-censored no-birth supervision.  Query count therefore cannot
        # dilute the event loss, but false births are not cost-free.
        for batch_index in range(batch_size):
            if not real_prefixes[batch_index]:
                continue
            available = torch.ones(
                birth_logits.size(1), device=device, dtype=torch.bool
            )
            for query_index in positive_assignments.get(batch_index, ()):
                available[int(query_index)] = False
            if available.any():
                balanced_background_losses.append(
                    F.softplus(birth_logits[batch_index][available].max())
                )

        ragged_logits = outputs["event_ragged_state_logits"]
        ragged_targets = outputs["event_ragged_state_targets"]
        ragged_end_offsets = outputs["event_ragged_end_offsets"]
        ragged_end_targets = outputs["event_ragged_end_targets"]
        ragged_class_logits = outputs["event_ragged_class_logits"]
        ragged_class_targets = outputs["event_ragged_class_targets"]
        ragged_embeddings = outputs["event_ragged_embeddings"]
        group_keys = list(outputs["event_ragged_group_keys"])
        if len(group_keys) != ragged_logits.size(0):
            raise RuntimeError("ragged tensor/metadata length mismatch")
        if ragged_logits.size(-1) != 3:
            raise RuntimeError(
                "D1 active owners must use CANCEL/CONTINUE/END logits"
            )
        ragged_sources = [str(value) for value in outputs["event_ragged_sources"]]
        if len(ragged_sources) != ragged_logits.size(0):
            raise RuntimeError("ragged source/tensor length mismatch")

        grouped = {}
        for index, key in enumerate(group_keys):
            group = (str(key[0]), int(key[1]))
            grouped.setdefault(group, []).append(index)

        owner_state_losses = []
        owner_class_losses = []
        end_hazard_losses = []
        end_offset_losses = []
        identity_losses = []
        false_track_groups = 0
        observed_end_groups = 0
        source_row_counts = {}
        source_class_row_counts = {}
        source_group_counts = {}
        source_end_risk_counts = {}
        for source in ragged_sources:
            source_row_counts[source] = source_row_counts.get(source, 0) + 1
        for source, target in zip(ragged_sources, ragged_class_targets):
            if int(target.item()) >= 0:
                source_class_row_counts[source] = (
                    source_class_row_counts.get(source, 0) + 1
                )
        for indices in grouped.values():
            index_tensor = torch.tensor(indices, device=device, dtype=torch.long)
            group_sources = {ragged_sources[index] for index in indices}
            for source in group_sources:
                source_group_counts[source] = source_group_counts.get(source, 0) + 1
            owner_state_losses.append(
                F.cross_entropy(
                    ragged_logits[index_tensor],
                    ragged_targets[index_tensor],
                )
            )
            class_mask = ragged_class_targets[index_tensor] >= 0
            if class_mask.any():
                owner_class_losses.append(
                    F.cross_entropy(
                        ragged_class_logits[index_tensor][class_mask],
                        ragged_class_targets[index_tensor][class_mask],
                    )
                )
            else:
                false_track_groups += 1

            target_event_id = group_keys[indices[0]][2]
            if target_event_id is None:
                continue
            for source in group_sources:
                source_end_risk_counts[source] = (
                    source_end_risk_counts.get(source, 0) + 1
                )
            group_targets = ragged_targets[index_tensor]
            end_margin = ragged_logits[index_tensor, 2] - ragged_logits[
                index_tensor, :2
            ].max(dim=-1).values
            observed = group_targets == 2
            if int(observed.sum().item()) > 1:
                raise RuntimeError("a D1 track has more than one observed end")
            at_risk = group_targets == 1
            hazard_nll = F.softplus(end_margin[at_risk]).sum()
            if observed.any():
                observed_end_groups += 1
                first_end = int(observed.nonzero(as_tuple=False)[0].item())
                hazard_nll = hazard_nll + F.softplus(-end_margin[first_end])
                end_offset_losses.append(
                    F.smooth_l1_loss(
                        ragged_end_offsets[index_tensor[first_end]],
                        ragged_end_targets[index_tensor[first_end]],
                    )
                )
            # If no observed end is present, this is a right-censored track and
            # only its survival terms contribute.
            if self.event_d1_use_hazard:
                end_hazard_losses.append(hazard_nll)

            if len(indices) > 1:
                ordered = sorted(
                    indices, key=lambda item: float(group_keys[item][3])
                )
                previous = ragged_embeddings[ordered[:-1]]
                current = ragged_embeddings[ordered[1:]]
                identity_losses.append(
                    (1.0 - F.cosine_similarity(previous, current, dim=-1)).mean()
                )

        # Same-class concurrent instances are explicit identity negatives.
        # Use one hardest negative per anchor so cost is linear in retained
        # anchors rather than an unbounded trajectory-pair queue.
        by_frame = {}
        for index, key in enumerate(group_keys):
            if key[2] is None or int(ragged_class_targets[index].item()) < 0:
                continue
            by_frame.setdefault((str(key[0]), float(key[3])), []).append(index)
        for indices in by_frame.values():
            for anchor in indices:
                negatives = [
                    other
                    for other in indices
                    if other != anchor
                    and group_keys[other][2] != group_keys[anchor][2]
                    and int(ragged_class_targets[other].item())
                    == int(ragged_class_targets[anchor].item())
                ]
                if not negatives:
                    continue
                similarities = F.cosine_similarity(
                    ragged_embeddings[anchor].unsqueeze(0),
                    ragged_embeddings[negatives],
                    dim=-1,
                )
                identity_losses.append(F.softplus(similarities.max()))

        def mean_or_zero(values):
            return torch.stack(values).mean() if values else zero

        birth_loss = mean_or_zero(birth_event_losses)
        if balanced_background_losses:
            background_loss = mean_or_zero(balanced_background_losses)
            birth_loss = (
                0.5 * (birth_loss + background_loss)
                if birth_event_losses
                else background_loss
            )
        losses = {
            "loss_event_birth": birth_loss,
            "loss_event_start_offset": mean_or_zero(start_losses),
            "loss_event_alive": zero,
            "loss_event_end": mean_or_zero(end_hazard_losses),
            "loss_event_end_offset": mean_or_zero(end_offset_losses),
            "loss_event_owner_class": mean_or_zero(owner_class_losses),
            "loss_event_owner_state": mean_or_zero(owner_state_losses),
            "loss_event_identity": (
                mean_or_zero(identity_losses)
                if self.event_d1_use_identity
                else zero
            ),
            "event_birth_positive_count": torch.tensor(
                float(len(birth_event_losses)), device=device
            ),
            "event_alive_positive_count": (ragged_targets == 1).sum().detach(),
            "event_end_positive_count": torch.tensor(
                float(observed_end_groups), device=device
            ),
            "event_owner_assignment_count": torch.tensor(
                float(assignment_count), device=device
            ),
            "event_ragged_track_count": torch.tensor(
                float(len(grouped)), device=device
            ),
            "event_false_track_cancel_group_count": torch.tensor(
                float(false_track_groups), device=device
            ),
        }
        association_counts = {}
        for row in outputs["event_association_rows"]:
            source = str(row["source"])
            association_counts[source] = association_counts.get(source, 0) + 1
        lifecycle_counts = {}
        for row in outputs.get("event_runtime_source_events", ()):
            key = (str(row["source"]), str(row["transition"]))
            lifecycle_counts[key] = lifecycle_counts.get(key, 0) + 1

        def metric_source_name(source):
            return "".join(
                character if character.isalnum() else "_"
                for character in str(source)
            )

        for source in sorted(
            set(source_row_counts)
            | set(source_class_row_counts)
            | set(source_group_counts)
            | set(source_end_risk_counts)
            | set(association_counts)
            | {source for source, _ in lifecycle_counts}
        ):
            suffix = metric_source_name(source)
            losses["event_source_{}_row_count".format(suffix)] = torch.tensor(
                float(source_row_counts.get(source, 0)), device=device
            )
            losses["event_source_{}_group_count".format(suffix)] = torch.tensor(
                float(source_group_counts.get(source, 0)), device=device
            )
            losses[
                "event_source_{}_class_row_count".format(suffix)
            ] = torch.tensor(
                float(source_class_row_counts.get(source, 0)), device=device
            )
            losses[
                "event_source_{}_end_risk_group_count".format(suffix)
            ] = torch.tensor(
                float(source_end_risk_counts.get(source, 0)), device=device
            )
            losses[
                "event_association_{}_count".format(suffix)
            ] = torch.tensor(
                float(association_counts.get(source, 0)), device=device
            )
            for transition in ("birth", "cancel", "end", "emit", "reacquire"):
                losses[
                    "event_source_{}_{}_count".format(suffix, transition)
                ] = torch.tensor(
                    float(lifecycle_counts.get((source, transition), 0)),
                    device=device,
                )
        return losses
    
    def get_loss(self, loss, outputs, targets, infos, indices, **kwargs):
        loss_map = {
            'cls_loss': self.loss_cls,
            'reg_loss': self.loss_reg
        }
        
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, infos, indices, **kwargs)
    
    def forward(self, outputs, targets, infos, device, log=True):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        self.device = device
        indices = self.matcher(outputs, targets, self.device)        
        
        # compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, infos, indices))

        if self.event_enabled:
            losses.update(self.loss_event(outputs, targets, infos))
            
        return losses
