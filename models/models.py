import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import math
import random
from .transformer import build_transformer
from .event_memory import (
    CausalTemporalHistory,
    DynamicEventMemory,
    EventTransitionHead,
    OwnerEventDecoder,
    PolicyIndependentRiskMemory,
    causal_single_assignment,
    d1_owner_supervision,
    resolve_d13_mechanism_contracts,
    resolve_d14_birth_objective,
    resolve_d16_risk_contract,
    resolve_event_modes,
    resolve_model_variant,
    select_disjoint_teacher_query,
    temporal_viterbi_assignment,
)

class MATR(nn.Module):
    def __init__(self, args):
        super(MATR, self).__init__()
        self.args = args
        self.training = args.training
        self.n_feature = args.feat_dim
        n_class = args.num_of_class
        n_embedding_dim = args.hidden_dim
        self.n_embedding_dim = n_embedding_dim
        n_enc_layer = args.enc_layers
        n_enc_head = args.e_nheads
        n_dec_layer = args.dec_layers
        n_dec_head = args.d_nheads
        n_seglen = args.num_frame
        self.n_seglen = n_seglen
        self.num_queries = args.num_queries
        self.rgb = args.rgb
        self.flow = args.flow
        self.dropout=args.dropout
        
        self.use_flag = args.use_flag
        self.flag_threshold = args.flag_threshold
        self.max_memory_len = args.max_memory_len
        
        # FC layers for multi-modals
        if self.rgb and self.flow:
            self.feature_reduction_rgb = nn.Linear(self.n_feature//2, n_embedding_dim//2)
            self.feature_reduction_flow = nn.Linear(self.n_feature//2, n_embedding_dim//2)
        else:
            self.feature_reduction_rgb = nn.Linear(self.n_feature, n_embedding_dim)
            self.feature_reduction_flow = nn.Linear(self.n_feature, n_embedding_dim)
        
        # separte attention
        self.segment_encoder, self.segment_decoder = build_transformer(args)
        self.memory_encoder, self.memory_decoder = build_transformer(args)

        self.classification_head = nn.Sequential(nn.Linear(n_embedding_dim*2,n_embedding_dim), nn.ReLU(), nn.Linear(n_embedding_dim,n_class)) 
        self.stcls_head = nn.Sequential(nn.Linear(n_embedding_dim,n_embedding_dim), nn.ReLU(), nn.Linear(n_embedding_dim,self.max_memory_len+2))
        self.streg_head = nn.Sequential(nn.Linear(n_embedding_dim,n_embedding_dim), nn.Tanh(), nn.Linear(n_embedding_dim,self.max_memory_len+2))
        self.edcls_head = nn.Sequential(nn.Linear(n_embedding_dim,n_embedding_dim), nn.ReLU(), nn.Linear(n_embedding_dim,1))
        self.edreg_head = nn.Sequential(nn.Linear(n_embedding_dim,n_embedding_dim), nn.Tanh(), nn.Linear(n_embedding_dim,1))
        
        if self.use_flag:
            self.flag_token = nn.Parameter(torch.randn(1, 1, n_embedding_dim))
            self.segment_flag_head = nn.Sequential(nn.Linear(n_embedding_dim,n_embedding_dim), nn.ReLU(), nn.Linear(n_embedding_dim, 1), nn.Sigmoid())     
        
        self.decoder_token = nn.Parameter(torch.randn(self.num_queries*2, 1, n_embedding_dim))
        
        # memory queue
        self.dropout = nn.Dropout(args.dropout)
        self.memory_sampler = args.memory_sampler
        # self.compress_layer = nn.Identity()
        # self.compressed_size = n_seglen
        # self.compressed_ratio = 1
        
        self.pos_token = nn.Parameter(torch.randn(self.num_queries, 1, n_embedding_dim))
        self.segment_pos_encoding = PositionalEncoding_segment(n_embedding_dim, args.dropout, maxlen=400)
        self.memory_pos_encoding = PositionalEncoding_memory_flag(n_embedding_dim, args.dropout, maxlen=400)       
            
        self.video_name = None
        self.memory_queue = None
        self.memory_queue_index = None

        # Exact upstream MATR is an independent model variant.  Every BxO cell,
        # including b0o0, is EventMATR and shares the event transition head.
        self.model_variant = resolve_model_variant(args)
        self.birth_mode, self.ownership_mode, self.event_arm = resolve_event_modes(args)
        self.event_enabled = self.model_variant == "eventmatr"
        self.event_lifecycle_version = getattr(
            args, "event_lifecycle_version", "v1_dense"
        )
        if self.event_lifecycle_version not in {"v1_dense", "d1_censored"}:
            raise ValueError(
                "event_lifecycle_version must be v1_dense or d1_censored"
            )
        if not self.event_enabled and self.event_lifecycle_version != "v1_dense":
            raise ValueError("native_matr cannot enable an EventMATR D1 lifecycle")
        self.event_d1_enabled = (
            self.event_enabled and self.event_lifecycle_version == "d1_censored"
        )
        self.event_d1_lane = getattr(args, "event_d1_lane", "th")
        if self.event_d1_lane not in {"r", "t", "h", "th"}:
            raise ValueError("event_d1_lane must be r, t, h, or th")
        self.event_d1_use_identity = self.event_d1_lane in {"t", "th"}
        self.event_d1_use_hazard = self.event_d1_lane in {"h", "th"}
        (
            self.event_d13_variant,
            self.event_association_contract,
            self.event_birth_risk_contract,
        ) = resolve_d13_mechanism_contracts(args)
        (
            self.event_d14_variant,
            self.event_birth_objective_contract,
        ) = resolve_d14_birth_objective(args)
        (
            self.event_d16_variant,
            self.event_owner_risk_contract,
        ) = resolve_d16_risk_contract(args)
        self.event_d16_enabled = self.event_d16_variant != "none"
        if (
            not self.event_d1_enabled
            and self.event_d13_variant != "d12_control"
        ):
            raise ValueError(
                "a D1.3 mechanism variant requires the d1_censored lifecycle"
            )
        if self.event_d14_variant != "none":
            if not self.event_d1_enabled or not self.event_d1_use_hazard:
                raise ValueError(
                    "a D1.4 birth objective requires a censored-hazard D1 lane"
                )
            if self.event_d13_variant != "combined":
                raise ValueError(
                    "D1.4 must layer on the frozen D1.3 combined contract"
                )
        if self.event_d16_enabled:
            if not self.event_d1_enabled or not self.event_d1_use_hazard:
                raise ValueError(
                    "D1.6 policy-independent risk requires a censored-hazard D1 lane"
                )
        if self.event_enabled:
            if torch.cuda.is_available() and torch.cuda.device_count() != 1:
                raise RuntimeError(
                    "EventMATR streaming state requires exactly one visible GPU "
                    "per process"
                )
            self.event_transition_head = EventTransitionHead(
                n_embedding_dim,
                independent_birth=self.event_d1_enabled,
            )
            # Keep decoder capacity identical across the 2x2 study.  O0/O1
            # changes only how a persistent record obtains its owner state:
            # O0 refreshes it from current queries, whereas O1 carries it
            # forward.  Both paths use this same lifecycle/class decoder.
            self.event_owner_decoder = OwnerEventDecoder(
                n_embedding_dim,
                n_class,
                num_heads=n_dec_head,
                num_states=3 if self.event_d1_enabled else 4,
            )
            self.event_memory = DynamicEventMemory(
                birth_mode=self.birth_mode,
                ownership_mode=self.ownership_mode,
                birth_logit_threshold=getattr(
                    args, "event_birth_logit_threshold", None
                ),
                end_logit_threshold=getattr(args, "event_end_logit_threshold", None),
                min_duration_frames=getattr(args, "event_min_duration_frames", 1),
                emit_delay_frames=getattr(args, "event_emit_delay_frames", 0),
                resource_limit=getattr(args, "event_resource_limit", 0),
                segment_size=n_seglen,
                enable_reacquisition=(
                    self.event_d1_enabled and self.event_d1_use_identity
                ),
                strict_causal_boundary=self.event_d1_enabled,
                owner_state_count=3 if self.event_d1_enabled else 4,
            )
            self.event_temporal_history = CausalTemporalHistory(n_seglen)
            self.event_risk_memory = (
                PolicyIndependentRiskMemory() if self.event_d16_enabled else None
            )
            # Read-only counterfactual scans may share one GT-free MATR query
            # stream across several isolated lifecycle memories.  The default
            # is production behavior; only an explicit eval-only setter can
            # disable the in-model runtime unroll.
            self._event_diagnostic_query_only = False
            self._event_risk_diagnostic_mode = False
    
    def forward(self, inputs, device):
        # inputs - batch x seq_len x featsize
        self.device = device
        infos = inputs['infos']
        event_targets = inputs.get("event_targets")
        event_valid_mask = inputs.get("event_valid_mask")
        event_supervision_mode = self.training or bool(
            getattr(self, "_event_risk_diagnostic_mode", False)
        )
        if (
            self.event_enabled
            and self._event_diagnostic_query_only
            and (
                self.training
                or torch.is_grad_enabled()
                or event_targets is not None
                or event_valid_mask is not None
            )
        ):
            raise RuntimeError(
                "EventMATR diagnostic query-only mode requires eval-mode no_grad "
                "and a GT-free model payload"
            )
        if self.event_d1_enabled:
            forbidden = {
                "duration",
                "true_duration",
                "video_time",
                "frame_to_time",
            }.intersection(infos)
            if not event_supervision_mode and "segment_flag" in infos:
                forbidden.add("segment_flag")
            if forbidden:
                raise RuntimeError(
                    "D1 model boundary received future/full-video metadata: {}".format(
                        sorted(forbidden)
                    )
                )
            if event_supervision_mode and event_targets is not None:
                if event_valid_mask is None:
                    raise RuntimeError(
                        "D1 training target visibility mask is missing"
                    )
                valid = event_valid_mask.to(event_targets.device).bool()
                observed_end = event_targets[..., 7] > 0.5
                exposed_future_end = (
                    valid
                    & ~observed_end
                    & torch.isfinite(event_targets[..., 3])
                )
                if bool(exposed_future_end.any().item()):
                    raise RuntimeError(
                        "D1 model boundary received a future GT endpoint"
                    )
                missing_observed_end = (
                    valid
                    & observed_end
                    & ~torch.isfinite(event_targets[..., 3])
                )
                if bool(missing_observed_end.any().item()):
                    raise RuntimeError(
                        "D1 observed END target is missing its current endpoint"
                    )
        inputs = inputs['inputs']
        st = infos['st']
        ed = infos['ed']
        bs = inputs.shape[0]
        
        base_x = self.input_projection(inputs) # seq_len x batch x featsize
        
        if self.use_flag:
            flag_token = self.flag_token.expand(-1, base_x.shape[1], -1)
            base_x = torch.cat((base_x, flag_token), dim=0)
        pos_x = self.segment_pos_encoding(base_x)
        
        encoded_x = self.segment_encoder(base_x, pos=pos_x)
        
        
        if self.use_flag:
            anc_flag = self.segment_flag_head(encoded_x[-1]) # batch x 1
        else:
            anc_flag = torch.ones((bs, 1)).to(device)
            
        base_x = base_x[:self.n_seglen]
        pos_x = pos_x[:self.n_seglen]
        encoded_x = encoded_x[:self.n_seglen]
            
        self.check_nan(encoded_x)

        memory_queue = self.memory_queue
        memory_queue_index = self.memory_queue_index
        if self.args.training or self.use_flag != True:
            memory_args = (bs, infos['video_name'], infos['current_frame'], infos['segment_flag'])
        else:
            memory_args = (bs, infos['video_name'], infos['current_frame'], (anc_flag > self.flag_threshold).int())
        
        memory_feature, memory_feature_index, memory_queue, memory_queue_index = self.memory_update(memory_args=memory_args,
                                                   memory_queue=memory_queue,
                                                   memory_queue_index=memory_queue_index, 
                                                   current_segment=base_x)

        decoder_token = self.decoder_token.expand(-1, encoded_x.shape[1], -1)
        pos_token = self.pos_token.repeat(2, decoder_token.shape[1], 1)

        decoded_x, att_seg = self.segment_decoder(decoder_token, encoded_x, query_pos=pos_token, pos=pos_x)
        
        end_cls_feature, anc_end_offset = self.end_offset_process(decoded_x)
        
        memory_feature_index = memory_feature_index.permute([1,0]) # batch x max len
        memory_feature_index = (infos['current_frame'].unsqueeze(dim=1) - memory_feature_index) / self.n_seglen
        memory_feature_index[memory_feature_index < 0] = -1
        mem_pos = self.memory_pos_encoding_process(memory_feature, memory_feature_index)

        decoded_x, att_mem = self.memory_decoder(decoded_x, memory_feature, query_pos=pos_token, pos=mem_pos)
        decoded_x = decoded_x.permute([1,0,2]) # batch x queires len x featsize
        
        self.check_nan(decoded_x)
        
        decoded_x_cls = decoded_x[:,:self.num_queries]
        decoded_x_reg = decoded_x[:,self.num_queries:]

        # Preserve the native query tensors for the optional EventMATR heads.
        event_class_query = decoded_x_cls
        event_regression_query = decoded_x_reg

        decoded_x_cls = torch.cat([end_cls_feature, decoded_x_cls], dim=2)
        anc_cls = self.classification_head(decoded_x_cls)

        anc_stcls = self.stcls_head(decoded_x_reg)
        anc_start_offset = self.streg_head(decoded_x_reg)        
        anc_reg = torch.cat([anc_start_offset, anc_end_offset], dim=2)
        
        if memory_queue != None:
            self.memory_queue = memory_queue.detach()
        if memory_queue_index != None:
            self.memory_queue_index = memory_queue_index
            
        self.video_name = infos['video_name']
        
        out = {
            "pred_cls": anc_cls,
            "pred_reg": anc_reg
        }
        out["pred_stcls"] = anc_stcls
        out["att_1"] = att_seg
        out["att_2"] = att_mem
            
        if self.use_flag:
            out['pred_flag'] = anc_flag

        if self.event_enabled:
            (
                event_state_logits,
                learned_birth_logits,
                event_alive_logits,
                event_end_logits,
                event_query_features,
            ) = self.event_transition_head(
                event_class_query, event_regression_query
            )

            # D1 uses an independently learned binary birth hazard.  Legacy
            # v1_dense keeps the original four-state START margin.  Neither path
            # introduces a tuned scalar threshold.
            event_birth_logits = learned_birth_logits

            start_bin = torch.argmax(anc_stcls, dim=-1)
            selected_start_residual = torch.gather(
                anc_start_offset,
                2,
                start_bin.unsqueeze(-1),
            ).squeeze(-1)
            current_frames = infos['current_frame']
            if not torch.is_tensor(current_frames):
                current_frames = torch.as_tensor(
                    current_frames, device=anc_cls.device, dtype=anc_cls.dtype
                )
            else:
                current_frames = current_frames.to(
                    device=anc_cls.device, dtype=anc_cls.dtype
                )
            candidate_start_frames = current_frames.reshape(-1, 1) - (
                start_bin.to(anc_cls.dtype) + selected_start_residual
            ) * self.n_seglen
            if self.event_d1_enabled:
                candidate_start_frames = candidate_start_frames.clamp_min(0.0)
            event_end_offsets = anc_end_offset.squeeze(-1)

            out.update(
                {
                    "event_birth_logits": event_birth_logits,
                    "event_alive_logits": event_alive_logits,
                    "event_end_logits": event_end_logits,
                    "event_state_logits": event_state_logits,
                    "event_end_offsets": event_end_offsets,
                    "event_query_features": event_query_features,
                    "event_candidate_start_frames": candidate_start_frames,
                }
            )

            ragged_state_logits = []
            ragged_state_targets = []
            ragged_end_offsets = []
            ragged_end_targets = []
            ragged_class_logits = []
            ragged_class_targets = []
            ragged_embeddings = []
            ragged_group_keys = []
            ragged_sources = []
            birth_risk_groups = []
            association_rows = []
            association_audit_rows = []
            runtime_source_events = []
            run_runtime = (
                self.event_d1_enabled
                or (not self.training)
                or bool(getattr(self.args, "event_runtime_during_training", False))
            ) and not self._event_diagnostic_query_only
            if run_runtime:
                # A MATR batch contains consecutive prefixes.  Lifecycle state
                # must therefore be unrolled in presentation order; decoding
                # all owners once before the batch would hide births from later
                # prefixes in the same batch.
                runtime_rows = []
                video_names = infos['video_name']
                if self.event_d1_enabled and event_supervision_mode:
                    if event_targets is None or event_valid_mask is None:
                        raise RuntimeError(
                            "D1 training requires prefix-visible event targets"
                        )
                    event_targets = event_targets.to(event_state_logits.device)
                    event_valid_mask = event_valid_mask.to(
                        event_state_logits.device
                    ).bool()
                for batch_index in range(event_birth_logits.size(0)):
                    owner_runtime = {}
                    real_prefix = bool(
                        self._slice_prefix_info(
                            infos, "is_real_prefix", batch_index, True
                        )[0]
                    )
                    video_name = str(video_names[batch_index])
                    frame_value = float(current_frames[batch_index].item())
                    target_by_id = {}
                    oracle_births = []
                    risk_seed_details = {}
                    if (
                        self.event_d1_enabled
                        and event_supervision_mode
                        and real_prefix
                    ):
                        rows = event_targets[batch_index][
                            event_valid_mask[batch_index]
                        ]
                        for row in rows:
                            target_by_id[int(row[0].item())] = row

                        self.event_temporal_history.append(
                            video_name=video_name,
                            frame=frame_value,
                            state_logits=event_state_logits[batch_index],
                            birth_logits=event_birth_logits[batch_index],
                            class_logits=anc_cls[batch_index],
                            query_features=event_query_features[batch_index],
                        )
                        active_target_ids = {
                            record.target_event_id
                            for record in self.event_memory.records(video_name)
                            if record.target_event_id is not None
                        }
                        risk_target_ids = (
                            set(
                                self.event_risk_memory.target_event_ids(video_name)
                            )
                            if self.event_d16_enabled
                            else set()
                        )
                        teacher_ratio = float(
                            getattr(
                                self.args,
                                "event_teacher_forcing_ratio",
                                0.5,
                            )
                        )
                        teacher_ratio = min(1.0, max(0.0, teacher_ratio))
                        visible_target_specs = []
                        visible_target_details = {}
                        for target_event_id, row in target_by_id.items():
                            is_birth = bool(row[5].item())
                            is_alive = bool(row[6].item())
                            is_end = bool(row[7].item())
                            missing_track = target_event_id not in active_target_ids
                            missing_risk = target_event_id not in risk_target_ids
                            if not (
                                is_birth
                                or ((missing_track or missing_risk) and is_alive)
                            ):
                                continue
                            start_frame = float(row[2].item())
                            lower = max(0.0, start_frame - self.n_seglen + 1.0)
                            window = self.event_temporal_history.window(
                                video_name, lower=lower, upper=frame_value
                            )
                            if not window:
                                raise RuntimeError(
                                    "causal history omitted the current real prefix"
                                )
                            window_state = torch.stack(
                                [item["state_logits"] for item in window],
                                dim=0,
                            )
                            window_class = torch.stack(
                                [item["class_logits"] for item in window],
                                dim=0,
                            )
                            window_birth = torch.stack(
                                [item["birth_logits"] for item in window],
                                dim=0,
                            )
                            if self.event_d1_lane == "r":
                                class_id = min(
                                    max(0, int(row[1].item())),
                                    window_class.size(-1) - 2,
                                )
                                score = (
                                    F.logsigmoid(window_birth[-1])
                                    + window_class[-1].log_softmax(dim=-1)[
                                        :, class_id
                                    ]
                                )
                                path = [int(score.detach().argmax().item())]
                            else:
                                path = temporal_viterbi_assignment(
                                    window_state,
                                    window_class,
                                    torch.stack(
                                        [
                                            item["query_features"]
                                            for item in window
                                        ],
                                        dim=0,
                                    ),
                                    int(row[1].item()),
                                    birth_logits=window_birth,
                                )
                            if not path:
                                continue
                            if is_birth:
                                risk_indices = list(range(len(window)))
                                if self.event_d1_lane == "r":
                                    risk_indices = [len(window) - 1]
                                birth_risk_groups.append(
                                    {
                                        "batch_index": int(batch_index),
                                        "video_name": video_name,
                                        "target_event_id": int(target_event_id),
                                        "class_id": int(row[1].item()),
                                        "start_frame": float(start_frame),
                                        "frames": torch.tensor(
                                            [
                                                float(window[index]["frame"])
                                                for index in risk_indices
                                            ],
                                            device=event_state_logits.device,
                                            dtype=event_state_logits.dtype,
                                        ),
                                        "selected_logits": torch.stack(
                                            [
                                                window[index]["birth_logits"][
                                                    int(path[path_index])
                                                ]
                                                for path_index, index in enumerate(
                                                    risk_indices
                                                )
                                            ]
                                        ),
                                        "terminal_query": int(path[-1]),
                                    }
                                )
                            detail = {
                                "row": row,
                                "path": path,
                                "start_frame": start_frame,
                                "is_birth": is_birth,
                                "anchor_feature": window[-1]["query_features"][
                                    int(path[-1])
                                ],
                            }
                            if self.event_d16_enabled and missing_risk and not is_end:
                                risk_seed_details[int(target_event_id)] = detail
                            if not missing_track or is_end:
                                continue
                            visible_target_details[int(target_event_id)] = detail
                            visible_target_specs.append(
                                {
                                    "target_event_id": int(target_event_id),
                                    "class_id": int(row[1].item()),
                                    "start_frame": start_frame,
                                    "anchor_feature": window[-1][
                                        "query_features"
                                    ][int(path[-1])],
                                }
                            )

                        predicted_birth_queries = (
                            self.event_memory.preview_birth_queries(
                                video_name,
                                candidate_state_logits=event_state_logits[
                                    batch_index
                                ],
                                birth_logits=event_birth_logits[batch_index],
                            )
                        )
                        association = causal_single_assignment(
                            predicted_query_indices=predicted_birth_queries,
                            candidate_start_frames=candidate_start_frames[
                                batch_index
                            ],
                            class_logits=anc_cls[batch_index],
                            query_features=event_query_features[batch_index],
                            target_specs=visible_target_specs,
                            max_start_distance=float(self.n_seglen),
                            association_contract=self.event_association_contract,
                        )
                        association_audit_rows.append(
                            {
                                "video_name": video_name,
                                "frame": frame_value,
                                "visible_target_count": int(
                                    association.target_count
                                ),
                                "predicted_birth_query_count": int(
                                    association.predicted_query_count
                                ),
                                "pair_count": int(association.pair_count),
                                "class_mismatch_pair_count": int(
                                    association.class_mismatch_pair_count
                                ),
                                "class_argmax_reject_pair_count": int(
                                    association.class_argmax_reject_pair_count
                                ),
                                "start_distance_reject_pair_count": int(
                                    association.start_distance_reject_pair_count
                                ),
                                "admissible_pair_count": int(
                                    association.admissible_pair_count
                                ),
                                "admissible_class_argmax_mismatch_pair_count": int(
                                    association.admissible_class_argmax_mismatch_pair_count
                                ),
                                "ambiguous_query_count": len(
                                    association.ambiguous_queries
                                ),
                                "ambiguous_target_count": len(
                                    association.ambiguous_targets
                                ),
                                "assignment_count": len(
                                    association.assignments
                                ),
                            }
                        )
                        for query_index, target_event_id in sorted(
                            association.assignments.items()
                        ):
                            detail = visible_target_details[int(target_event_id)]
                            oracle_births.append(
                                {
                                    "query_index": int(query_index),
                                    "target_event_id": int(target_event_id),
                                    "start_frame": max(
                                        0.0, float(detail["start_frame"])
                                    ),
                                    "source": "predicted_associated",
                                    "association_status": "associated",
                                    "force_create": False,
                                    "merge_predicted": True,
                                }
                            )
                            association_rows.append(
                                {
                                    "video_name": video_name,
                                    "frame": frame_value,
                                    "query_index": int(query_index),
                                    "target_event_id": int(target_event_id),
                                    "source": "predicted_associated",
                                }
                            )

                        for query_index in association.unmatched_queries:
                            association_rows.append(
                                {
                                    "video_name": video_name,
                                    "frame": frame_value,
                                    "query_index": int(query_index),
                                    "target_event_id": None,
                                    "source": (
                                        "predicted_ambiguous"
                                        if query_index
                                        in association.ambiguous_queries
                                        else "predicted_unmatched"
                                    ),
                                }
                            )

                        # Reproducible teacher/predicted mixture without a
                        # second RNG stream or future annotations.  Teacher
                        # rows are independent births and cannot silently
                        # relabel an inadmissible predicted query.
                        occupied_birth_queries = set(predicted_birth_queries)
                        for target_event_id in association.unmatched_targets:
                            detail = visible_target_details[int(target_event_id)]
                            draw = (
                                (
                                    target_event_id * 1103515245
                                    + int(frame_value) * 12345
                                )
                                & 0xFFFF
                            ) / 65536.0
                            force_create = draw < teacher_ratio
                            source = (
                                "teacher_birth"
                                if detail["is_birth"]
                                else "teacher_recovery"
                            )
                            teacher_query = int(detail["path"][-1])
                            teacher_query_reassigned = False
                            if force_create:
                                row = detail["row"]
                                class_id = min(
                                    max(0, int(row[1].item())),
                                    anc_cls.size(-1) - 2,
                                )
                                available = [
                                    query_index
                                    for query_index in range(
                                        event_state_logits.size(1)
                                    )
                                    if query_index not in occupied_birth_queries
                                ]
                                if available:
                                    with torch.no_grad():
                                        start_evidence = F.logsigmoid(
                                            event_birth_logits[batch_index]
                                        )
                                        class_evidence = anc_cls[
                                            batch_index
                                        ].log_softmax(dim=-1)[:, class_id]
                                        normalized_queries = F.normalize(
                                            event_query_features[batch_index],
                                            dim=-1,
                                        )
                                        normalized_anchor = F.normalize(
                                            detail["anchor_feature"].to(
                                                event_query_features
                                            ),
                                            dim=0,
                                        )
                                        scores = (
                                            start_evidence
                                            + class_evidence
                                            + normalized_queries
                                            @ normalized_anchor
                                        )
                                    (
                                        selected_teacher_query,
                                        teacher_query_reassigned,
                                    ) = select_disjoint_teacher_query(
                                        scores,
                                        preferred_query=teacher_query,
                                        occupied_queries=occupied_birth_queries,
                                    )
                                else:
                                    selected_teacher_query = None
                                if selected_teacher_query is None:
                                    force_create = False
                                    source = "teacher_query_conflict_skipped"
                                else:
                                    teacher_query = selected_teacher_query
                            if force_create:
                                occupied_birth_queries.add(teacher_query)
                            oracle_births.append(
                                {
                                    "query_index": int(teacher_query),
                                    "target_event_id": int(target_event_id),
                                    "start_frame": max(
                                        0.0, float(detail["start_frame"])
                                    ),
                                    "source": source,
                                    "association_status": "associated",
                                    "force_create": force_create,
                                    "merge_predicted": False,
                                }
                            )
                            association_rows.append(
                                {
                                    "video_name": video_name,
                                    "frame": frame_value,
                                    "query_index": int(teacher_query),
                                    "target_event_id": int(target_event_id),
                                    "source": source,
                                    "force_create": bool(force_create),
                                    "query_reassigned": bool(
                                        teacher_query_reassigned
                                    ),
                                }
                            )
                    if self.event_d16_enabled and event_supervision_mode and real_prefix:
                        risk_rows = self._decode_policy_independent_risks(
                            video_name=video_name,
                            current_frame=frame_value,
                            current_queries=event_query_features[
                                batch_index : batch_index + 1
                            ],
                            target_by_id=target_by_id,
                            risk_seed_details=risk_seed_details,
                            is_eos=bool(
                                self._slice_prefix_info(
                                    infos, "is_eos", batch_index, False
                                )[0]
                            ),
                        )
                        for risk_row in risk_rows:
                            ragged_state_logits.append(risk_row["state_logits"])
                            ragged_state_targets.append(risk_row["state_target"])
                            ragged_end_offsets.append(risk_row["end_offset"])
                            ragged_end_targets.append(risk_row["end_target"])
                            ragged_class_logits.append(risk_row["class_logits"])
                            ragged_class_targets.append(risk_row["class_target"])
                            ragged_embeddings.append(risk_row["embedding"])
                            ragged_group_keys.append(risk_row["group_key"])
                            ragged_sources.append(risk_row["source"])
                    if self.ownership_mode == "fresh_rematch" and real_prefix:
                        self.event_memory.rematch_active_owners(
                            video_names[batch_index],
                            event_query_features[batch_index],
                            anc_cls[batch_index],
                        )
                    (
                        owner_embeddings,
                        owner_padding_mask,
                        owner_record_ids,
                    ) = self.event_memory.owner_batch(
                        [video_names[batch_index]],
                        device=event_query_features.device,
                        dtype=event_query_features.dtype,
                        embedding_dim=self.n_embedding_dim,
                    )
                    if owner_embeddings.size(1) > 0:
                        (
                            owner_state_logits,
                            owner_end_offsets,
                            owner_class_logits,
                            owner_updated_embeddings,
                        ) = self.event_owner_decoder(
                            owner_embeddings,
                            event_query_features[
                                batch_index : batch_index + 1
                            ],
                            owner_padding_mask,
                        )
                        owner_runtime = {
                            "owner_state_logits": owner_state_logits,
                            "owner_end_offsets": owner_end_offsets,
                            "owner_class_logits": owner_class_logits,
                            "owner_updated_embeddings": owner_updated_embeddings,
                            "owner_valid_mask": ~owner_padding_mask,
                            "owner_record_ids": owner_record_ids,
                        }
                        if (
                            self.event_d1_enabled
                            and self.training
                            and not self.event_d16_enabled
                        ):
                            metadata = self.event_memory.record_metadata(
                                video_name, owner_record_ids
                            )
                            for owner_index, record_info in enumerate(metadata):
                                target_event_id = record_info["target_event_id"]
                                target_row = (
                                    None
                                    if target_event_id is None
                                    else target_by_id.get(int(target_event_id))
                                )
                                (
                                    target_state,
                                    target_class,
                                    target_end_offset,
                                ) = d1_owner_supervision(
                                    target_row,
                                    current_frame=frame_value,
                                    segment_size=self.n_seglen,
                                )
                                ragged_state_logits.append(
                                    owner_state_logits[0, owner_index]
                                )
                                ragged_state_targets.append(target_state)
                                ragged_end_offsets.append(
                                    owner_end_offsets[0, owner_index]
                                )
                                ragged_end_targets.append(target_end_offset)
                                ragged_class_logits.append(
                                    owner_class_logits[0, owner_index]
                                )
                                ragged_class_targets.append(target_class)
                                ragged_embeddings.append(
                                    owner_updated_embeddings[0, owner_index]
                                )
                                ragged_group_keys.append(
                                    (
                                        video_name,
                                        int(record_info["event_id"]),
                                        target_event_id,
                                        frame_value,
                                    )
                                )
                                ragged_sources.append(record_info["source"])
                    runtime_row = self.event_memory.step(
                            video_names=[video_names[batch_index]],
                            current_frames=current_frames[
                                batch_index : batch_index + 1
                            ],
                            birth_logits=event_birth_logits[
                                batch_index : batch_index + 1
                            ],
                            alive_logits=event_alive_logits[
                                batch_index : batch_index + 1
                            ],
                            end_logits=event_end_logits[
                                batch_index : batch_index + 1
                            ],
                            end_offsets=event_end_offsets[
                                batch_index : batch_index + 1
                            ],
                            class_logits=anc_cls[batch_index : batch_index + 1],
                            query_features=event_query_features[
                                batch_index : batch_index + 1
                            ],
                            candidate_start_frames=candidate_start_frames[
                                batch_index : batch_index + 1
                            ],
                            candidate_state_logits=event_state_logits[
                                batch_index : batch_index + 1
                            ],
                            is_real_prefix=self._slice_prefix_info(
                                infos, "is_real_prefix", batch_index, True
                            ),
                            is_eos=self._slice_prefix_info(
                                infos, "is_eos", batch_index, False
                            ),
                            preserve_graph=(
                                self.event_d1_enabled and self.training
                            ),
                            oracle_births=[oracle_births],
                            **owner_runtime,
                        )
                    runtime_rows.append(runtime_row)
                    runtime_source_events.extend(
                        dict(row)
                        for row in self.event_memory.last_audit.get(
                            "lifecycle_events", ()
                        )
                    )
                runtime = {
                    key: torch.cat([row[key] for row in runtime_rows], dim=0)
                    for key in runtime_rows[0]
                }
                if self.event_d1_enabled and self.training:
                    self.event_memory.detach_graph()
                    self.event_temporal_history.detach()
                    if self.event_d16_enabled:
                        self.event_risk_memory.detach_graph()
            else:
                shape = event_birth_logits.shape
                runtime = {
                    "new_birth_mask": torch.zeros(
                        shape, dtype=torch.bool, device=anc_cls.device
                    ),
                    "ended_mask": torch.zeros(
                        shape, dtype=torch.bool, device=anc_cls.device
                    ),
                    "emitted_mask": torch.zeros(
                        shape, dtype=torch.bool, device=anc_cls.device
                    ),
                    "cancelled_mask": torch.zeros(
                        shape, dtype=torch.bool, device=anc_cls.device
                    ),
                    "active_count": torch.zeros(
                        shape[0], dtype=torch.long, device=anc_cls.device
                    ),
                    "birth_count": torch.zeros(
                        shape[0], dtype=torch.long, device=anc_cls.device
                    ),
                    "end_count": torch.zeros(
                        shape[0], dtype=torch.long, device=anc_cls.device
                    ),
                    "emit_count": torch.zeros(
                        shape[0], dtype=torch.long, device=anc_cls.device
                    ),
                    "cancellation_count": torch.zeros(
                        shape[0], dtype=torch.long, device=anc_cls.device
                    ),
                    "reacquisition_count": torch.zeros(
                        shape[0], dtype=torch.long, device=anc_cls.device
                    ),
                    "runtime_capacity_exhaustions": torch.zeros(
                        shape[0], dtype=torch.long, device=anc_cls.device
                    ),
                    "padding_prefixes_ignored": torch.zeros(
                        shape[0], dtype=torch.long, device=anc_cls.device
                    ),
                    "eos_observed": torch.zeros(
                        shape[0], dtype=torch.long, device=anc_cls.device
                    ),
                }
            out.update(
                {
                    "event_new_birth_mask": runtime["new_birth_mask"],
                    "event_ended_mask": runtime["ended_mask"],
                    "event_emitted_mask": runtime["emitted_mask"],
                    "event_cancelled_mask": runtime["cancelled_mask"],
                    "event_active_count": runtime["active_count"],
                    "event_birth_count": runtime["birth_count"],
                    "event_end_count": runtime["end_count"],
                    "event_emit_count": runtime["emit_count"],
                    "event_cancellation_count": runtime[
                        "cancellation_count"
                    ],
                    "event_reacquisition_count": runtime[
                        "reacquisition_count"
                    ],
                    "event_runtime_capacity_exhaustions": runtime[
                        "runtime_capacity_exhaustions"
                    ],
                    "event_padding_prefixes_ignored": runtime[
                        "padding_prefixes_ignored"
                    ],
                    "event_eos_observed": runtime["eos_observed"],
                }
            )

            if self.event_d1_enabled:
                if ragged_state_logits:
                    out["event_ragged_state_logits"] = torch.stack(
                        ragged_state_logits
                    )
                    out["event_ragged_state_targets"] = torch.tensor(
                        ragged_state_targets,
                        device=event_state_logits.device,
                        dtype=torch.long,
                    )
                    out["event_ragged_end_offsets"] = torch.stack(
                        ragged_end_offsets
                    )
                    out["event_ragged_end_targets"] = torch.tensor(
                        ragged_end_targets,
                        device=event_state_logits.device,
                        dtype=event_state_logits.dtype,
                    )
                    out["event_ragged_class_logits"] = torch.stack(
                        ragged_class_logits
                    )
                    out["event_ragged_class_targets"] = torch.tensor(
                        ragged_class_targets,
                        device=event_state_logits.device,
                        dtype=torch.long,
                    )
                    out["event_ragged_embeddings"] = torch.stack(
                        ragged_embeddings
                    )
                else:
                    out["event_ragged_state_logits"] = event_state_logits.new_zeros(
                        (0, self.event_owner_decoder.num_states)
                    )
                    out["event_ragged_state_targets"] = torch.zeros(
                        (0,), device=event_state_logits.device, dtype=torch.long
                    )
                    out["event_ragged_end_offsets"] = event_state_logits.new_zeros(
                        (0,)
                    )
                    out["event_ragged_end_targets"] = event_state_logits.new_zeros(
                        (0,)
                    )
                    out["event_ragged_class_logits"] = anc_cls.new_zeros(
                        (0, anc_cls.size(-1))
                    )
                    out["event_ragged_class_targets"] = torch.zeros(
                        (0,), device=event_state_logits.device, dtype=torch.long
                    )
                    out["event_ragged_embeddings"] = event_query_features.new_zeros(
                        (0, event_query_features.size(-1))
                    )
                out["event_ragged_group_keys"] = ragged_group_keys
                out["event_ragged_sources"] = ragged_sources
                out["event_birth_risk_groups"] = birth_risk_groups
                out["event_association_rows"] = association_rows
                out["event_association_audit_rows"] = association_audit_rows
                out["event_runtime_source_events"] = runtime_source_events

            # Dense prototypes train the same decoder used for ragged runtime
            # records.  This path is deliberately present in every BxO cell,
            # so ownership persistence is the only O-axis intervention.
            (
                owner_prototype_state_logits,
                owner_prototype_end_offsets,
                owner_prototype_class_logits,
                _,
            ) = self.event_owner_decoder(
                event_query_features, event_query_features
            )
            out["event_owner_state_logits"] = owner_prototype_state_logits
            out["event_owner_end_offsets"] = owner_prototype_end_offsets
            out["event_owner_class_logits"] = owner_prototype_class_logits

        return out

    def reset_event_runtime(self, video_name=None):
        """Reset only EventMATR runtime records; native MATR is a no-op."""
        if self.event_enabled:
            self.event_memory.reset(video_name)
            self.event_temporal_history.reset(video_name)
            if self.event_d16_enabled:
                self.event_risk_memory.reset(video_name)

    def set_event_diagnostic_query_only(self, enabled: bool) -> None:
        """Enable the GT-free, eval-only query stream used by D1.5 diagnostics."""

        enabled = bool(enabled)
        if enabled and self.training:
            raise RuntimeError(
                "EventMATR diagnostic query-only mode cannot be enabled in training"
            )
        if not self.event_enabled and enabled:
            raise RuntimeError(
                "native MATR has no EventMATR diagnostic runtime to disable"
            )
        if self.event_enabled:
            self._event_diagnostic_query_only = enabled

    def set_event_risk_diagnostic_mode(self, enabled: bool) -> None:
        """Replay frozen weights on a common target-visible risk set in eval mode."""

        enabled = bool(enabled)
        if enabled and not self.event_d16_enabled:
            raise RuntimeError(
                "policy-independent risk diagnostics require the D1.6 risk contract"
            )
        if self.event_enabled:
            self._event_risk_diagnostic_mode = enabled

    @staticmethod
    def _slice_prefix_info(infos, key, index, default):
        if key not in infos:
            return [default]
        value = infos[key]
        if torch.is_tensor(value):
            return value[index : index + 1]
        if isinstance(value, (list, tuple)):
            return [value[index]]
        return [value]

    def _decode_policy_independent_risks(
        self,
        *,
        video_name,
        current_frame,
        current_queries,
        target_by_id,
        risk_seed_details,
        is_eos,
    ):
        """Decode D1.6 risks without allowing runtime policy to censor them."""

        if not self.event_d16_enabled or not (
            self.training or self._event_risk_diagnostic_mode
        ):
            return []
        self.event_risk_memory.sync_unmatched_runtime(
            video_name,
            self.event_memory.records(video_name),
            current_frame=current_frame,
        )
        for target_event_id, detail in sorted(risk_seed_details.items()):
            row = detail["row"]
            self.event_risk_memory.ensure_target(
                video_name,
                target_event_id=int(target_event_id),
                class_id=int(row[1].item()),
                owner_embedding=detail["anchor_feature"],
                current_frame=current_frame,
            )

        owner_embeddings, risk_ids = self.event_risk_memory.owner_batch(
            video_name,
            device=current_queries.device,
            dtype=current_queries.dtype,
            embedding_dim=self.n_embedding_dim,
        )
        if owner_embeddings.size(1) == 0:
            return []
        (
            state_logits,
            end_offsets,
            class_logits,
            updated_embeddings,
        ) = self.event_owner_decoder(owner_embeddings, current_queries)
        metadata = self.event_risk_memory.metadata(video_name, risk_ids)
        rows = []
        closing = []
        for owner_index, record_info in enumerate(metadata):
            target_event_id = record_info["target_event_id"]
            if target_event_id is None:
                # A model birth is not a false birth merely because it is not
                # matched yet.  Only an actually observed EOS closes the causal
                # compatibility window and supplies CANCEL supervision.
                target_state = 0 if is_eos else -100
                target_class = -100
                target_end_offset = 0.0
            else:
                target_row = target_by_id.get(int(target_event_id))
                if target_row is None:
                    raise RuntimeError(
                        "D1.6 target-backed risk disappeared before END/EOS"
                    )
                (
                    target_state,
                    target_class,
                    target_end_offset,
                ) = d1_owner_supervision(
                    target_row,
                    current_frame=current_frame,
                    segment_size=self.n_seglen,
                )
                if target_state == 0:
                    raise RuntimeError(
                        "D1.6 runtime policy cannot label a target risk CANCEL"
                    )
            rows.append(
                {
                    "state_logits": state_logits[0, owner_index],
                    "state_target": int(target_state),
                    "end_offset": end_offsets[0, owner_index],
                    "end_target": float(target_end_offset),
                    "class_logits": class_logits[0, owner_index],
                    "class_target": int(target_class),
                    "embedding": updated_embeddings[0, owner_index],
                    "group_key": (
                        str(video_name),
                        int(record_info["risk_id"]),
                        target_event_id,
                        float(current_frame),
                    ),
                    "source": str(record_info["source"]),
                }
            )
            if target_state == 2 or is_eos:
                closing.append(int(record_info["risk_id"]))

        self.event_risk_memory.update(
            video_name,
            risk_ids=risk_ids,
            owner_embeddings=updated_embeddings,
            current_frame=current_frame,
            preserve_graph=True,
        )
        if closing:
            self.event_risk_memory.close(video_name, closing)
        return rows
            
    def input_projection(self, inputs):
        if self.rgb and self.flow:
            base_x_rgb = self.feature_reduction_rgb(inputs[:,:,:self.n_feature//2])
            base_x_flow = self.feature_reduction_flow(inputs[:,:,self.n_feature//2:])
            base_x = torch.cat([base_x_rgb,base_x_flow],dim=-1)
        elif self.rgb:
            base_x_rgb = self.feature_reduction_rgb(inputs[:,:,:self.n_feature])
            base_x = base_x_rgb
        else:
            base_x_flow = self.feature_reduction_flow(inputs[:,:,:self.n_feature])
            base_x = base_x_flow
        return base_x.permute([1,0,2])
    
    def check_nan(self, values):
        if (values.isnan() == True).any():
            import pdb;pdb.set_trace()
    
    def memory_update(self, memory_args, memory_queue, memory_queue_index, current_segment):  
        bs, video_names, cur_frames, flags = memory_args      
        # memory initialization
        if memory_queue == None:
            memory_queue = torch.zeros((self.max_memory_len * self.n_seglen, bs, self.n_embedding_dim)).to(self.device) # max_len*seg len x batch x featsize
            memory_queue_index = 10000*torch.ones((self.max_memory_len, bs)).to(self.device) # max_memory_len x batch
        
        # reset memory when input segment belongs to another video.
        elif self.video_name != None:
            for b, video_name in enumerate(video_names):  
                if self.video_name[b] != video_name:
                    memory_queue[:, b:] = 0
                    memory_queue_index[:, b:] = 10000                    
        
        # memory features index        
        memory_feature_index = torch.cat((memory_queue_index,cur_frames.unsqueeze(dim=0)), dim=0) # max_len+1 x batch
        
        # memory queue + current segment
        current_segment = current_segment.detach()
        _memory_queue = torch.cat((memory_queue[:,:bs], current_segment), dim=0) # max_len+1*seg len x batch x featsize
        
        if self.use_flag:
            memory_queue = memory_queue.permute([1,0,2]) # batch x max_len*seg len x featsize
            memory_queue_index = memory_queue_index.permute([1,0]) # batch x max_len
            
            for i, b_memory_queue in enumerate(memory_queue[:bs]): # b_memory_queue: max_len*seg len x featsize
                if flags[i] == 1:
                    memory_queue[i] = torch.cat((b_memory_queue, current_segment[i]), dim=0)[self.n_seglen:]
                    memory_queue_index[i] = torch.cat((memory_queue_index[i], cur_frames[i].unsqueeze(dim=0)), dim=0)[1:]                 
            memory_queue = memory_queue.permute([1,0,2]).contiguous() # max_len*seg len x batch x featsize
            memory_queue_index = memory_queue_index.permute([1,0]).contiguous() # max_len x batch
        else:
            memory_queue = _memory_queue[self.n_seglen:]
            memory_queue_index = memory_feature_index[1:]
             
        # sampled memory
        memory_feature = self.sample_memory(_memory_queue)

        return memory_feature, memory_feature_index, memory_queue, memory_queue_index
    
    def end_offset_process(self, decoded_x):
        decoded_x_cls = decoded_x.permute([1,0,2])[:,:self.num_queries]
        decoded_x_end = decoded_x.permute([1,0,2])[:,self.num_queries:]

        endreg = self.edreg_head(decoded_x_end)

            
        return decoded_x_cls, endreg
    
    def memory_pos_encoding_process(self, memory_feature, memory_feature_index):         
        between_memory_index = memory_feature_index.repeat_interleave(self.n_seglen, dim=1)
        inside_memory_index = torch.arange(self.n_seglen-1,-1,-1).repeat(self.max_memory_len+1)
        inside_memory_index = inside_memory_index.unsqueeze(0).expand(between_memory_index.size(0),-1)
        mem_pos = self.memory_pos_encoding(between_memory_index, inside_memory_index, self.memory_sampler)

        return mem_pos
    
    def sample_memory(self, memory):
        if memory == None:
            return None
        if "gap" in self.memory_sampler:
            gap_size = int(self.memory_sampler[-1])
            return memory[0::gap_size]
        elif self.memory_sampler == 'all':
            return memory

class PositionalEncoding_segment(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float = 0.3,
                 maxlen: int = 750,
                 batch_first=False):
        super(PositionalEncoding_segment, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)
        self.register_buffer(
            'position_ids',
            torch.arange(maxlen).expand((1, -1))
            )
    def forward(self, token_embedding: torch.Tensor): 
        return self.pos_embedding[:token_embedding.size(0), :].flip(dims=(0,))
    
class PositionalEncoding_memory_flag(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float = 0.3,
                 maxlen: int = 750
                 ):
        super(PositionalEncoding_memory_flag, self).__init__()
        emb_size = int(emb_size/2)
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / (emb_size))
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)
        self.register_buffer(
            'position_ids',
            torch.arange(maxlen))
    def forward(self, between_memory_index, inside_memory_index, memory_sampler): 
        if memory_sampler == 'all':
            between_pos_embedding = self.pos_embedding[between_memory_index.long()].permute([1,0,2])
            inside_pos_embedding = self.pos_embedding[inside_memory_index.long()].permute([1,0,2])     
        elif 'gap' in str(memory_sampler):
            gap_size = int(memory_sampler[-1])
            between_pos_embedding = self.pos_embedding[between_memory_index[:,::gap_size].long()].permute([1,0,2])
            inside_pos_embedding = self.pos_embedding[inside_memory_index[:,::gap_size].long()].permute([1,0,2])
                
        return torch.cat((between_pos_embedding, inside_pos_embedding), dim=2)
