# Research Discussion Prompt: Online/Causal TAD Toward Paper-Ready Adaptive On-TAD

请基于公开 GitHub 仓库进行深入研究路线讨论，不要只做高层口号式总结。

公开仓库：

https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/tree/codex/online-tad-clean-20260702

当前分支：

`codex/online-tad-clean-20260702`

## 讨论目标

请围绕当前项目讨论以下问题：

1. 我们当前研究面向的任务领域是什么？
2. 这个方向到底要解决什么科学问题和工程问题？
3. 当前代码具体实现了什么方法？
4. 当前实现离论文级别的最终目标还有哪些差距？
5. 如果要把它推进到论文级实现结果，应该怎样设计实验计划？
6. 哪些关键代码需要实现或重构？

请给出严格、可执行、可审查的建议。不要默认当前实现已经 paper-ready。

## 当前研究定位

当前项目面向 **在线/因果 Temporal Action Detection (Online/Causal TAD, On-TAD)**。

任务设定：

- 输入是连续视频流或按时间到达的视频窗口。
- 模型在时刻 `t` 只能使用当前和过去的帧、token、memory 与预测状态。
- 模型需要在线输出动作类别、起止时间、置信度和 emission 时间。
- 评估不仅看 mAP，还必须看 no-future correctness、emission ledger、decision latency、memory/state 管理和计算预算。

核心区别：

- 这不是离线 TAD 改名叫 online。
- 不能用完整视频 feature cache、raw prediction cache、未来帧、未来标签 shortcut 或 video-level offline NMS 来支撑 online claim。
- 当前最安全的定位是：一个 windowed streaming-safe raw-frame TAD 原型，而不是最终论文级 adaptive On-TAD。

## 当前已实现方法

请重点检查并讨论这些当前实现：

- Raw-frame THUMOS route:
  `configs/causaltad/thumos_siglip2_matr_ontad_p0.py`
  `configs/causaltad/thumos_siglip2_matr_ontad_p1.py`
  `configs/causaltad/thumos_siglip2_matr_ontad_p1_fix.py`
  `configs/causaltad/thumos_siglip2_matr_ontad_p1_full_60.py`

- Frozen SigLIP/SigLIP2 recent-frame encoder:
  `opentad/models/backbones/online_siglip_adapter.py`

- Raw-frame adapter wrapper:
  `opentad/models/backbones/online_videomae_adapter.py`

- Causal temporal projection:
  `opentad/models/projections/causal_temporalmaxer_proj.py`
  `opentad/models/projections/causal_proj.py`

- MATR-style head and online branches:
  `opentad/models/dense_heads/matr_head.py`
  `opentad/models/dense_heads/anchor_free_head.py`

- Streaming-safe emission protocol:
  `opentad/utils/online_protocol.py`
  `opentad/cores/test_engine.py`

- Raw-frame dataset and temporal metadata:
  `opentad/datasets/raw_frame.py`
  `opentad/datasets/transforms/loading.py`
  `opentad/datasets/transforms/end_to_end.py`

- Training and optimizer path:
  `opentad/cores/train_engine.py`
  `opentad/cores/optimizer.py`
  `opentad/cores/layer_decay_optimizer.py`

Current implemented baseline:

- Frozen SigLIP/SigLIP2 visual encoder from raw frames.
- Causal temporal projection, avoiding future temporal context.
- MATR-style actionness/start/end/emission branches.
- `memory_size=0` for P0/P1 baseline to avoid overlapping-window state leakage.
- Raw prediction cache disabled.
- Streaming-safe emission ledger and latency summary.
- P1 full-60 config for full-dataset training/validation.

Current claim boundary:

- Safe claim: raw-frame, frozen-visual-encoder, windowed streaming-safe online TAD prototype.
- Unsafe claim until further evidence: real VideoMAE online TAD, fully end-to-end visual finetuning, true low-latency continuous streaming agent, adaptive frame selection, AdaTAD contribution, paper-ready full THUMOS mAP.

## Problems To Solve

请把问题拆成科学问题和工程问题。

Scientific problems:

- How to detect action boundaries online without future frames?
- How to train temporal localization when the true action end may not have been observed yet?
- How to emit predictions with bounded delay while preserving mAP?
- How to adaptively select frames/tokens under a computation budget without missing boundaries?
- How to prove causality with tests and protocols rather than only architectural claims?

Engineering problems:

- Raw frames to temporal grid to seconds must be exactly traceable.
- Per-video state must be keyed by `video_name`, `stream_id`, `window_start_frame`, and absolute time.
- DDP and single-process paths must both work.
- mAP evaluation must not silently use offline NMS or future-aware post-processing.
- Emission ledger must expose latency, source time, emission time, stream key, and no-future violations.
- Training configs must make online protocol fields explicit, not rely on fragile inherited defaults.

## Current Gaps

请严格评估这些现阶段差距：

1. P1 full-60 job has been configured, but paper claims require actual full validation mAP, emission ledger, latency summary, and stability evidence.
2. Current route is windowed streaming-safe. It is not yet per-frame/per-token continuous low-latency streaming.
3. Training still appears to use full GT action endpoints in anchor-free regression targets. This creates a likely future endpoint label shortcut for strict online training.
4. VideoMAE route remains contract-only stub unless a real causal/streaming VideoMAE backbone is wired with `use_stub_backbone=False`.
5. Adaptive frame selection and AdaTAD are not yet implemented.
6. Whole-model no-future perturb tests for the actual P1 chain are still needed.
7. Selector collapse prevention, irregular token decoding, and budget-aware evaluation are future work.

## Paper-Level Final Goal

请讨论一个论文级最终系统，目标可以是：

**Budgeted Adaptive Online TAD with Causal Emission**

核心贡献应该至少包含一个真正方法创新，而不是只把 frozen SigLIP 接到已有 head：

- Online-censored localization training:
  unobserved future endpoints are not directly regressed.
- Streaming-safe boundary/emission protocol:
  predictions are emitted when enough causal evidence exists, with explicit latency accounting.
- Adaptive frame/token selection:
  the model learns which frames/tokens to process under a budget, while preserving action boundaries.
- Irregular-time TAD head or AdaTAD bridge:
  selected sparse tokens are decoded using true token timestamps, not pretended to be a uniform dense grid.

最终产出目标：

- A clean open-source implementation.
- Full THUMOS14 experiments with mAP and latency.
- Ablations showing causality, budget, selector, boundary loss, and emission protocol effects.
- No-future perturb tests and protocol tests.
- A paper narrative that claims only what the evidence supports.

Expected empirical effect:

- Comparable or better mAP than fixed uniform online baseline under the same no-future protocol.
- Lower compute/frame budget than dense raw-frame processing.
- Lower or controlled emission latency.
- Better boundary recall under budget constraints.
- Robustness to long background videos and overlapping actions.

## Proposed Experiment Plan

Please refine or critique this plan.

### Stage 0: Current P1 Audit Baseline

Goal:

- Prove current frozen SigLIP/SigLIP2 raw-frame P1 route is valid as a windowed streaming-safe baseline.

Experiments:

- Full THUMOS train, 60 epochs.
- Validation every 10 epochs.
- Save mAP, emission ledger, latency summary, emission count, no-future violations.
- Compare P1-fix pilot vs P1-full.

Must-pass checks:

- No raw prediction cache.
- No feature cache.
- No offline video-level NMS in streaming-safe eval.
- `memory_size=0`.
- Resolved config uses raw frames and causal projection.
- Full validation evaluator uses all GT and predictions.

### Stage 1: Online-Censored Training

Goal:

- Remove future endpoint label shortcut.

Design:

- For a point/token at observed time `t`, do not regress action end `e` when `e > t + max_future_offset`.
- For censored positives, train actionness/inside-action and optional lower-bound duration.
- Enable end/boundary loss only when endpoint has been observed.

Experiments:

- Baseline full GT endpoint training.
- Censored endpoint training.
- Censored + actionness.
- Censored + boundary observed-only.

Metrics:

- mAP.
- online AP by latency bins.
- false early emission rate.
- boundary recall after endpoint observed.

### Stage 2: Causal Motion Or Boundary Branch

Goal:

- Add a real online signal beyond frozen image semantics.

Options:

- causal motion branch from frame differences,
- endpoint-aware emission branch,
- streaming-safe boundary confidence,
- uncertainty-based delayed emission.

Ablations:

- no motion,
- causal motion,
- non-causal motion forbidden control,
- boundary branch on/off,
- emit branch on/off.

### Stage 3: Adaptive Frame Selection

Goal:

- Reduce compute while maintaining online TAD accuracy and boundary quality.

Selector inputs:

- cheap causal frame features,
- frame difference/motion magnitude,
- past selected token features,
- past actionness/uncertainty,
- budget state,
- time since last selected frame.

Selector outputs:

- `keep_prob[t]`,
- hard keep/drop action,
- selected absolute frames,
- selected token timestamps,
- budget usage,
- max-gap state.

Training:

- warm-up on dense/fixed teacher actionness and boundary targets,
- train selected-token detector,
- joint fine-tune with straight-through or Gumbel-TopK selection,
- inference uses hard causal selection only.

Anti-collapse losses:

- budget loss,
- entropy target,
- diversity/repulsion,
- temporal coverage/max-gap,
- actionness distillation,
- boundary preservation,
- latency penalty,
- hard negative/background quota.

### Stage 4: Irregular-Time AdaTAD / MATR Head

Goal:

- Consume selected sparse tokens without pretending they are uniformly sampled.

Implementation route:

- Add `token_times_sec` and `cell_widths_sec`.
- Modify point generator to use true timestamps.
- Decode proposals directly in seconds.
- Clamp proposal end by current observed time.
- Use masks for missing/invalid selected tokens.

Ablations:

- uniform K-frame selection,
- random K-frame selection,
- motion-only selector,
- learned selector without boundary loss,
- learned selector with full losses,
- dense frozen teacher upper bound.

## Key Code Sketches

请基于当前仓库风格，把下面草案转化为可落地代码。

### Online-Censored Target Sketch

```python
def censor_segments_for_prefix(gt_segments, observed_time, max_future_offset=0.0):
    observed_end = observed_time + max_future_offset
    censored_segments = gt_segments.clone()
    endpoint_observed = gt_segments[:, 1] <= observed_end
    intersects_prefix = gt_segments[:, 0] <= observed_end

    censored_segments[:, 1] = torch.minimum(
        gt_segments[:, 1],
        gt_segments.new_full((gt_segments.shape[0],), observed_end),
    )
    valid_for_actionness = intersects_prefix
    valid_for_end_regression = endpoint_observed
    return censored_segments, valid_for_actionness, valid_for_end_regression
```

Expected integration:

- Add an `online_censored_training` mode to `MATRHead` or `AnchorFreeHead`.
- Pass point centers / observed times into target assignment.
- Mask regression loss for unobserved endpoints.
- Keep actionness/inside-action supervision for observed prefixes.

### Selector Packet Sketch

```python
@dataclass
class SelectionPacket:
    selected_abs_frames: torch.LongTensor
    selected_mask: torch.BoolTensor
    keep_logits: torch.Tensor
    keep_probs: torch.Tensor
    token_times_sec: torch.Tensor
    budget_loss: torch.Tensor
    coverage_loss: torch.Tensor
    entropy_loss: torch.Tensor
    state: dict
```

### Causal Selector Sketch

```python
class OnlineFrameSelector(nn.Module):
    def __init__(self, in_dim, hidden_dim, budget_k, max_gap_frames):
        super().__init__()
        self.rnn = nn.GRUCell(in_dim, hidden_dim)
        self.keep_head = nn.Linear(hidden_dim, 1)
        self.budget_k = budget_k
        self.max_gap_frames = max_gap_frames

    def forward_step(self, cheap_feat_t, frame_idx_t, state, train=True):
        h = self.rnn(cheap_feat_t, state["h"])
        logit = self.keep_head(h).squeeze(-1)
        prob = torch.sigmoid(logit)

        if train:
            hard = (prob > torch.rand_like(prob)).float()
            keep = hard.detach() - prob.detach() + prob
        else:
            keep = self._budgeted_causal_decision(prob, frame_idx_t, state)

        state["h"] = h
        state["budget_used"] = state["budget_used"] + keep.detach()
        state["last_selected_frame"] = torch.where(
            keep.bool(), frame_idx_t, state["last_selected_frame"]
        )
        return keep, logit, prob, state
```

### Irregular-Time Decode Sketch

```python
class IrregularOnlineTADHead(nn.Module):
    def forward(self, feats, token_times_sec, mask, current_time_sec):
        x = self.causal_temporal_block(feats, mask, token_times_sec)
        cls_logits = self.cls_head(x)
        left_sec = F.softplus(self.left_reg(x)).squeeze(-1)
        right_sec = F.softplus(self.right_reg(x)).squeeze(-1)

        start = token_times_sec - left_sec
        end = torch.minimum(token_times_sec + right_sec, current_time_sec[:, None])
        proposals = torch.stack([start, end], dim=-1)
        return cls_logits, proposals
```

## Required Discussion Output

Please produce:

1. Task/domain definition.
2. Problem formulation.
3. What current code actually implements.
4. What it does not yet implement.
5. Whether the current route is a publishable contribution or only an engineering baseline.
6. A paper-level method proposal.
7. A detailed experiment plan with configs, metrics, and ablations.
8. Key implementation changes by file.
9. Minimal code skeletons for censored training, selector, irregular token packing, and emission protocol.
10. Risk list: causality leakage, label leakage, latency inflation, selector collapse, mAP collapse, DDP/single-rank mismatch.
11. Final go/no-go milestones before writing the paper.

Be strict. If a claim is not supported by code or experiments, say it cannot be claimed yet.
