# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
import copy

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, args, cost_class: float = 1, cost_iou: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_iou: This is the relative weight of the IoU between src and tgt in the matching cost
        """
        super().__init__()
        self.args = args
        self.segment_size = args.num_frame
        self.cost_class = cost_class
        self.cost_iou = cost_iou
        self.anti_len = args.anti_len
        self.max_memory_len = args.max_memory_len
        
        assert cost_class != 0 or cost_iou != 0, "all costs can't be 0"
    
    def _get_cost_iou(self, src_regs, tgt_regs, bs):
        sst = torch.min(src_regs[:,:,None,0], tgt_regs[:,None,:,0])
        led = torch.max(src_regs[:,:,None,1], tgt_regs[:,None,:,1])
        lst = torch.max(src_regs[:,:,None,0], tgt_regs[:,None,:,0])
        sed = torch.min(src_regs[:,:,None,1], tgt_regs[:,None,:,1])
        
        cost_iou = (sed-lst) / torch.maximum(led-sst, torch.ones_like(led-sst))
        return cost_iou.to(self.device)

    def _get_cost_class(self, src_probs, tgt_classes, bs):
        tgt_ids = torch.argmax(tgt_classes, dim=-1)
        cost_class = torch.cat([src_probs[b][:, tgt_ids[b]] for b in range(bs)]).reshape(bs, src_probs.size(1), tgt_classes.size(1))
        return cost_class

    def edrstc2r_pred(self, regs, stcls):
        stsel = torch.argmax(stcls, dim=2)
        st_regs = regs[:,:,:self.max_memory_len+2]
        
        streg = torch.gather(st_regs, 2, stsel.unsqueeze(-1)).squeeze()
        st_offset = (streg+stsel)*self.segment_size
        ed_offset = regs[:,:,-1]*self.segment_size
        sts = -st_offset
        ends = -ed_offset
        return torch.stack([sts, ends], dim=-1)
    
    def edrstc2r_gt(self, regs, stcls):
        stsel = torch.argmax(stcls, dim=2)
        st_offset = (regs[:,:,0] + stsel)*self.segment_size
        ed_offset = regs[:,:,1]*self.segment_size
        sts = -st_offset
        ends = -ed_offset
        return torch.stack([sts, ends], dim=-1)

    @torch.no_grad()
    def match_event_owners(
        self,
        state_logits,
        class_logits,
        class_ids,
        state_index,
        occupied_queries=None,
    ):
        """Assign prefix-visible GT events to MATR query identities.

        This auxiliary assignment is used only to construct training targets.
        Runtime ownership never receives GT.  ``occupied_queries`` makes O1
        identity persistent and fail-closed when the official prediction
        bandwidth is genuinely insufficient; no target is truncated.
        """
        if state_logits.ndim != 2 or state_logits.size(-1) != 4:
            raise ValueError("state_logits must be [Q,4]")
        class_ids = [int(class_id) for class_id in class_ids]
        if not class_ids:
            return []
        occupied = set(int(query) for query in (occupied_queries or ()))
        free_queries = [
            query for query in range(state_logits.size(0)) if query not in occupied
        ]
        if len(class_ids) > len(free_queries):
            raise RuntimeError(
                "{} EventMATR identities require assignment but only {} of {} "
                "official MATR queries are free; targets were not truncated".format(
                    len(class_ids), len(free_queries), state_logits.size(0)
                )
            )
        state_log_probs = state_logits.log_softmax(dim=-1)
        class_log_probs = class_logits.log_softmax(dim=-1)
        cost_rows = []
        for class_id in class_ids:
            class_id = min(max(0, class_id), class_logits.size(-1) - 2)
            score = (
                state_log_probs[free_queries, int(state_index)]
                + class_log_probs[free_queries, class_id]
            )
            cost_rows.append(-score)
        cost = torch.stack(cost_rows, dim=0).detach().cpu().numpy()
        target_indices, free_indices = linear_sum_assignment(cost)
        assignment = [-1] * len(class_ids)
        for target_index, free_index in zip(target_indices, free_indices):
            assignment[int(target_index)] = int(free_queries[int(free_index)])
        if any(query < 0 for query in assignment):
            raise RuntimeError("incomplete EventMATR owner assignment")
        return assignment
    
    @torch.no_grad()
    def forward(self, outputs, targets, device):
        self.device = device
        src_regs = outputs['pred_reg']
        tgt_regs = targets['reg_label']
        src_classes = outputs['pred_cls']
        tgt_classes = targets['cls_label']
        src_stcls = outputs['pred_stcls']
        tgt_stcls = targets['stcls_label']
        
        src_probs = src_classes.softmax(-1)
        
        bs, num_queries, _ = src_regs.shape
        _, num_tgt, _ = tgt_regs.shape
            
        src_steds = self.edrstc2r_pred(src_regs, src_stcls)
        tgt_steds = self.edrstc2r_gt(tgt_regs, tgt_stcls)
            
        cost_iou = -self._get_cost_iou(src_steds, tgt_steds, bs)
        cost_class = -self._get_cost_class(src_probs, tgt_classes, bs)
        # Final cost matrix            
        C = self.cost_class * cost_class + self.cost_iou * cost_iou
            
        C = C.view(bs, num_queries, -1).cpu()
        
        indices = [linear_sum_assignment(C[b]) for b in range(C.size(0))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(args, cost_class=1, cost_iou=1)
