## 0. 总判决

**判决：WARN / HOLD。** 当前仓库可以支撑的安全表述是：**一个 windowed、single-rank、streaming-safe raw-frame online TAD prototype**，其中 SigLIP/SigLIP2 raw-frame 路线比 VideoMAE stub 路线更接近可训练基线。它还不能支撑“Budgeted Adaptive Online TAD with Causal Emission”的论文级主张，因为核心创新项里至少三项尚未真正落地：**online-censored localization training、adaptive selection、irregular-time AdaTAD/MATR decode**。

我下面的判断是基于你给出的公开 branch 静态审查；我没有看到仓库中附带的 full-60 真实训练结果、checkpoint、mAP/ledger/latency JSON 或多 seed 结果。README 本身也明确说 raw-frame SigLIP/SigLIP2 和 VideoMAE-adapter 路线仍是 validation candidates，并且不得 claim paper-ready Online TAD、DDP-auditable streaming eval 或 visual-tower finetuning，直到 emission-ledger evaluation、no-future tests、overfit checks 和真实训练结果附上。([GitHub][1])

---

## 1. 任务领域定义

当前研究面向的是 **Online / Causal Temporal Action Detection**，不是传统 offline TAD。形式上，视频流到达时间为 (t)，模型在任意输出时刻 (\tau) 只能访问 (x_{\le \tau})，并输出：

[
d_i = (c_i,\ \hat{s}_i,\ \hat{e}_i,\ p_i,\ \tau_i)
]

其中 (c_i) 是类别，(\hat{s}_i,\hat{e}_i) 是动作起止时间，(p_i) 是置信度，(\tau_i) 是 emission time。在线协议至少要满足：

[
\hat{e}*i \le \tau_i,\qquad d_i(\tau_i)\ \text{对}\ x*{>\tau_i}\ \text{扰动不变}
]

也就是说，**no-future correctness 不是一句“用 causal conv”就成立**，而是输入、模型、缓存、后处理、NMS、评估、日志全链条都必须可审计。仓库 README 的当前目标确实是建立在线/因果 TAD 起点，并为流式推理、低延迟检测、有限未来上下文、状态缓存和在线后处理实验提供干净基线；但它也说明默认输入仍有许多预提取特征路线，raw-frame On-TAD 只是实验入口。([GitHub][1])

当前更精确的任务定义应写成：

**Windowed Streaming-Safe Online TAD**：给定非重叠或受控窗口 ([a,b])，模型在窗口结束 (b) 时发射只依赖该窗口及历史状态的 detection ledger。它不同于真正 per-frame continuous streaming agent，因为 emission 目前主要绑定到 `window_end_frame`，不是每一帧都运行状态机并即时决定 whether to emit。

---

## 2. 问题形式化

### 2.1 Causal observation

输入流为：

[
\mathcal{X}_\tau = {x_t \mid t \le \tau}
]

模型内部状态为：

[
h_\tau = f_\theta(h_{\tau-1}, x_\tau, z_\tau)
]

其中 (z_\tau) 是是否选择该帧或 token 的预算决策。当前仓库的 P0/P1 还没有真正 adaptive (z_\tau)，实际更接近固定 stride raw-frame window observation。

### 2.2 Emission protocol

在线检测器输出：

[
\mathcal{D}*\tau = g*\theta(h_\tau)
]

每个 proposal 必须记录：

[
(\text{video},\ \text{stream_key},\ \text{segment},\ \text{score},\ \text{label},\ \text{emit_frame},\ \text{source_grid},\ \text{end_frame},\ \text{latency})
]

当前仓库已经实现了这类 ledger 字段和 no-future summary 检查。`OnlineEmitter` 会过滤掉 `end_frame > eligible_end_frame` 的候选，并记录 `emit_frame`、`end_frame` 和 `latency_sec`；测试也覆盖了 future row rejection、single-rank 限制和 ledger 字段存在性。([GitHub][2])

### 2.3 Budgeted adaptive acquisition

论文最终目标如果叫 **Budgeted Adaptive Online TAD with Causal Emission**，则需要增加：

[
z_t \in {0,1},\quad \sum_{t\le \tau} z_t \le B_\tau
]

并证明 heavy visual encoder / detector 只处理被选中的 (z_t=1) 帧或 token，而不是 dense raw frames 后再伪装成 sparse。这一点当前仓库还不能 claim。

---

## 3. 当前代码实际实现

### 3.1 Raw-frame window dataset

`FrameWindowDataset` 是当前 raw-frame online route 的核心数据入口。它继承 sliding-window dataset，只允许 `input_format="raw_frames"`，保存 `stream_id`、`processor_id`、`encoder_id`、`image_size`、`frame_policy` 等协议字段；`get_gt` 把 THUMOS annotation 秒级 segment 转成帧级 segment；`__getitem__` 再把完整 GT segment 按窗口起点平移到 snippet grid，并写入 `window_start_frame`、`window_end_frame`、`fps`、`snippet_stride` 等 metadata。([GitHub][3])

**严格判断：**这说明 raw-frame pipeline 已落地，但也说明当前训练 target 仍来自完整 annotation endpoint。它不是 online-censored supervision。

### 3.2 P0/P1 SigLIP2 route

P0 是 frozen SigLIP2 recent-frame baseline：配置注释明确说 raw frames 直接加载、无 feature cache / raw-prediction shortcut、一个 sampled frame 产生一个 TAD grid token、MATR memory disabled；模型由 `OnlineVideoMAEAdapter` wrapper 包住 `OnlineSigLIPFrameEncoder`，再接 `CausalTemporalMaxerProj`、`FPNIdentity` 和 `MATRHead`，其中 `memory_size=0`、`clamp_end_to_current=True`、`max_future_offset=0.0`、`inference.load_from_raw_predictions=False`、`save_raw_prediction=False`。([GitHub][4])

P1 在 P0 上把 trainable scope 明确为 adapter、projection、neck、rpn_head，并仍标为 validation candidate / `formal_training_ready=False`。P1-full-60 去掉 pilot allow-list，设定 60 epoch、full validation、streaming-safe emission、ledger 文件和 latency summary 文件，但同样 `formal_training_ready=False`。([GitHub][5])

**可 claim：**P0/P1 是 frozen visual tower + trainable causal temporal adapter/head 的 raw-frame windowed online TAD validation route。
**不能 claim：**full THUMOS paper result、E2E visual finetuning、true VideoMAE online TAD、adaptive frame selection。

### 3.3 Causal projection

`CausalTemporalMaxerProj` 是一个严格 no-future 的 temporal pyramid：它拒绝 nonzero attention stem，拒绝 `strict_causal=False`，用 left-only padding 的 causal conv 和 causal max-pooling。代码注释明确说每个输出 index (t) 不消费 (>t) 的输入 index。([GitHub][6])

这个模块是当前最干净的 causality 子模块之一。仓库也有 perturb-style test，验证未来输入变化不会影响 cutoff 前输出。([GitHub][7])

### 3.4 MATR-style head

`MATRHead` 由 causal conv block、cls/reg/start/end/actionness/emit branches 组成。其 `CausalConvBlock` 使用 left padding；head 文档称它保留 OpenTAD anchor-free regression target contract，同时增加 online emission signals。([GitHub][8])

推理时，如果 `clamp_end_to_current=True`，proposal end 会被 clamp 到 point center + `max_future_offset`，因此当前 P0/P1 的推理输出层面有 no-future end guard。([GitHub][8])

但训练时，`forward_train` 仍调用 base anchor-free `self.losses(... gt_segments ...)`，然后再加 online branch losses；`_build_online_branch_targets` 直接用 `segment[1]` 构造 end target 和 emit target。这是核心风险：**causal token 的 forward 不一定看未来帧，但 supervision 使用了完整未来 endpoint。**([GitHub][8])

### 3.5 Streaming-safe emission ledger

`SingleStageDetector.post_processing` 在 `streaming_safe_emission=True` 时走 `_format_streaming_safe_results`，用 `window_end_frame` 作为 `emit_frame`，通过 `OnlineEmitter` 过滤 future proposal，并写出 `emit_frame`、`source_grid`、`stream_key`、`window_start_frame`、`window_end_frame`、`latency_sec` 等字段。([GitHub][9])

`test_engine` 在 eval 时会重置 online states、禁用 video-level sliding-window merge、验证 single-rank、汇总 emission ledger、验证 no-future summary，并写 `emission_ledger.json` / latency summary。([GitHub][10])

**但注意：**标准 `mAP.py` 仍是传统 detection AP evaluator：它从 prediction 读取 `segment`、`label`、`score`，按 tIoU 计算 AP；emission/latency 是 side ledger，不是 AP 匹配逻辑的一部分。([GitHub][11])

---

## 4. 当前未实现内容

| 模块                             | 当前状态                                                                        | 结论                                             |
| ------------------------------ | --------------------------------------------------------------------------- | ---------------------------------------------- |
| Full-60 真实结果                   | 有 P1-full-60 config，无公开结果 artifact                                          | **不能 claim paper-ready THUMOS mAP**            |
| Per-frame continuous streaming | 现在是 windowed / sliding-window raw-frame route，emission 多绑定窗口结束              | **不能 claim true continuous streaming agent**   |
| Online-censored training       | 训练 target 使用完整 `gt_segments` 和 `segment[1]`                                 | **存在 future endpoint label shortcut**          |
| Real VideoMAE online route     | README/配置明确 VideoMAE adapter backbone 是 contract-only stub                  | **不能 claim real VideoMAE online TAD**          |
| Adaptive frame selection       | 有 packet audit 概念，但没有 selector→selected-only heavy encoder→detector 完整路径    | **不能 claim adaptive/budgeted selection**       |
| Irregular-time decode          | 当前 P0/P1 仍是 regular grid / point generator                                  | **不能 claim irregular-time MATR/AdaTAD head**   |
| AdaTAD contribution            | 当前核心是 VideoMambaSuite + MATR/ActionFormer-style route                       | **不能 claim AdaTAD contribution**               |
| Whole-model no-future perturb  | 有 projection 局部 perturb test；缺完整模型、dataset、postprocess、cache 级 perturb test | **不能 claim whole-model no-future correctness** |
| DDP streaming eval             | 测试和 README 都限制 streaming-safe eval 当前 single-rank                           | **不能 claim DDP-auditable streaming eval**      |

VideoMAE stub 的风险尤其明确：causaltad README 说 `thumos_videomae_adapter_matr_ontad.py` 是 raw-frame VideoMAE adapter + MATR-style head validation candidate，但 stub backbone 是 contract-only，正式训练前必须接入真实 causal/streaming VideoMAE 并关闭 stub；配置中也能看到 `use_stub_backbone=True` 和 `stub_backbone_contract_only=True`。([GitHub][12])

---

## 5. 是否已经可作为论文贡献

**不能作为完整论文主贡献。** 当前最强可发表形态不是“方法结果论文”，而是“协议与工程原型 + 初步 baseline”的 technical report / workshop artifact。CVPR/ICCV 主会论文级别至少需要一个真正新算法和完整实验闭环。

### 当前可以 claim

1. **Windowed streaming-safe raw-frame online TAD prototype**：raw-frame dataset、frozen SigLIP2 frame encoder route、causal projection、MATR-style head、single-rank emission ledger 已经有代码路径。
2. **Emission ledger protocol prototype**：包含 emit frame、end frame、latency、stream key、no-future summary 和 single-rank guard。
3. **P1-full-60 evaluation configuration exists**：full validation、ledger、latency summary、streaming-safe postprocess 都已配置。

### 当前不能 claim

1. **不能 claim paper-ready full THUMOS mAP**：没有公开 result artifact。
2. **不能 claim true low-latency continuous streaming**：当前仍是 windowed。
3. **不能 claim no-future training**：target 使用完整未来 endpoint。
4. **不能 claim real VideoMAE online TAD**：VideoMAE route 是 stub。
5. **不能 claim adaptive acquisition**：selector 未接入。
6. **不能 claim AdaTAD contribution**：AdaTAD path 未形成。
7. **不能 claim whole-model no-future correctness**：缺端到端 perturb audit。

仓库自己的 claim hygiene test 也要求 raw-frame routes 保持 `formal_training_ready=False`，并防止 P1/P2 文案过度宣称 paper-ready。([GitHub][13])

---

## 6. 论文级方法设计

论文级最终目标建议收敛为：

**Budgeted Adaptive Online Temporal Action Detection with Causal Emission**

核心不是“把 offline TAD head 改成 causal conv”，而是解决三个科学问题：

### 6.1 科学问题 A：未观察未来 endpoint 时如何训练 localization？

传统 TAD regression target 假设完整 segment ([s,e]) 已知且可监督。但在线时，在 (\tau < e) 时，模型不应被要求从当前 token 精确回归未来 (e)。因此需要 **online-censored localization training**：

[
e_\tau^{obs} = \min(e,\tau)
]

对未完成动作，右边界是 censored target，不是 final endpoint target。训练目标分三类：

* start：若 (s \le \tau)，可以监督；
* ongoing/actionness：若 (s \le t \le \min(e,\tau))，可以监督；
* end / emit：只有 (e \le \tau) 时才监督 final end 和 emission。

这会把“未来 endpoint label shortcut”改成统计上合理的 survival / hazard / censored regression 问题。

### 6.2 科学问题 B：何时发射 detection？

Emission 不应只是 score threshold，而应该学习一个 causal hazard：

[
p_{\text{emit}}(t)=P(\text{action ended and should emit}\mid x_{\le t})
]

最终 detection 只有在 `emit_prob`、end probability、classification score 和 NMS protocol 同时满足时输出。这样 mAP、latency、duplicate suppression 才能被统一建模。

### 6.3 科学问题 C：预算受限时选哪些帧？

Adaptive frame/token selection 要解决：

[
\max_{z_{1:\tau}} \text{Utility}*{TAD}(\mathcal{D}*\tau),\quad
\sum z_t \le B,\quad
\max_gap(z)\le G,\quad
z_t \text{ causal}
]

selector 应该选择对 boundary、state transition、uncertainty、emission 有帮助的帧，而不是 action coverage。论文要强调 **boundary/emission utility**，而不是“看更多动作主体”。

### 6.4 工程问题：irregular timestamp decode

一旦 adaptive selection 接入，token 时间不再等间隔。head 必须使用真实 timestamp：

[
\hat{s}_i = t_i - \Delta^-_i,\qquad
\hat{e}_i = \min(t_i + \Delta^+_i,\tau)
]

其中 (\Delta) 应基于真实 seconds / frame timestamp，而不是默认 stride grid。否则高 IoU mAP 会因时间轴错配崩掉。

---

## 7. 实验计划与 ablation

### Stage 0：P1 full baseline audit

目标：证明当前 windowed raw-frame baseline 真实能跑通，并且 ledger 不违规。

必须产出：

* THUMOS val mAP@0.3/0.4/0.5/0.6/0.7 + average mAP。
* emission ledger JSON。
* latency summary：pred-end latency、matched-GT latency、p50/p90/p95/max。
* no-future summary：future_end_violations、negative_latency_rows、non_monotonic_emit_rows。
* stability：至少 3 seeds 或 2 seeds + overfit sanity。
* compute：FPS、GPU memory、frames processed、state size。
* cache audit：确认 `load_from_raw_predictions=False`、无旧缓存、无 teacher/dense cache。

Go 条件：full val mAP 非零且稳定，ledger violation 为 0，训练无 NaN，overfit smoke 能显著下降 loss。
No-go 条件：full val mAP 接近随机、emission 数量爆炸、ledger 违规、loss 不收敛。

### Stage 1：online-censored training

对比：

1. 当前 full endpoint target；
2. censored endpoint target；
3. only-end-hazard，无右边界 regression；
4. censored regression + hazard emission。

核心指标：

* mAP；
* high tIoU mAP@0.6/0.7；
* false early close rate；
* matched-GT latency；
* no-future perturb pass rate；
* before-end proposal quality。

Stage 1 是论文能否站住的关键。若 censored training 不优于或至少不显著稳定于 full endpoint target，论文主张会很危险。

### Stage 2：causal motion / boundary / emission branch

Ablation：

* no motion branch；
* causal motion branch；
* start/end branch only；
* actionness only；
* emit branch only；
* end-score gating vs actionness gating vs emit gating；
* `clamp_end_to_current` on/off；
* memory_size 0 vs causal contiguous memory。

P2 配置当前只是 motion-branch candidate，仓库注释也明确它需要 ablation 和 streaming-safe ledger support 才能成为 claim。([GitHub][14])

### Stage 3：adaptive frame selection

对比：

* dense window；
* uniform sparse；
* random sparse；
* score top-k；
* boundary/uncertainty selector；
* learned causal selector；
* learned selector + max-gap guard；
* learned selector + anti-collapse loss。

预算：

* 100%、50%、25%、12.5%；
* fixed budget vs dynamic budget；
* max-gap (G\in{8,16,32,64})。

必须报告：

* selected frames per video；
* max gap / mean gap；
* boundary support；
* emission support；
* compute saving；
* mAP-latency-budget Pareto curve。

### Stage 4：irregular-time AdaTAD/MATR head

Ablation：

* selected-token treated as uniform grid；
* selected-token with true timestamp decode；
* true timestamp + local density/gap embedding；
* true timestamp + irregular NMS；
* MATR head vs AdaTAD-style head。

这是论文高 IoU 能不能成立的关键。没有真实 timestamp decode，adaptive selector 很容易提升低 IoU actionness，却损坏 mAP@0.7。

---

## 8. 关键文件修改计划

### 必须新增

1. `opentad/models/selectors/causal_frame_selector.py`
   实现 causal selector、budget loss、max-gap guard、anti-collapse。

2. `opentad/models/utils/online_censored_targets.py`
   实现 online-censored target builder、right-boundary censor mask、hazard labels。

3. `opentad/models/utils/selected_token_packet.py`
   统一表示 selected frames、timestamps、valid mask、budget metadata。

4. `opentad/models/utils/irregular_time_decode.py`
   用真实 timestamps decode proposal seconds。

5. `opentad/evaluations/online_map.py`
   ledger-aware mAP：AP 匹配时保留 emission time、latency、no-future validity。

6. `tests/test_whole_model_no_future.py`
   对 dataset → backbone → selector → head → postprocess 做 future perturb。

### 必须重构

1. `opentad/datasets/raw_frame.py`
   增加 `observed_until_frame`、`observed_until_grid`、prefix/censored GT 视图，避免训练时默认完整 endpoint target。

2. `opentad/models/dense_heads/matr_head.py`
   增加 `target_mode="offline_full" | "online_censored"`，并把 `_build_online_branch_targets` 改为 censor-aware。

3. `opentad/models/detectors/single_stage.py`
   接入 selector packet，并确保 selected-only path 的 metadata 写入 ledger。

4. `opentad/cores/test_engine.py`
   ledger-aware evaluator 应与 mAP evaluator 联动，而不只是 side JSON。

5. `configs/causaltad/*.py`
   分出清晰配置：`p1_full_baseline.py`、`p1_censored.py`、`p2_emit.py`、`p3_selector.py`、`p4_irregular.py`。

---

## 9. 关键代码 skeleton

下面是应新增或重构的代码草案，**不是当前仓库已经完整实现的 claim**。

### 9.1 Online-censored target / loss

```python
# opentad/models/utils/online_censored_targets.py
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F


@dataclass
class OnlineCensoredTargets:
    cls_target: torch.Tensor          # [B, P, C]
    actionness: torch.Tensor          # [B, P, 1]
    start_target: torch.Tensor        # [B, P, C]
    end_target: torch.Tensor          # [B, P, C], only final observed ends
    emit_target: torch.Tensor         # [B, P, 1], only when end observed
    left_reg: torch.Tensor            # [B, P]
    right_reg: torch.Tensor           # [B, P], target to min(gt_end, observed_until)
    left_valid: torch.Tensor          # [B, P]
    right_observed_valid: torch.Tensor # [B, P], final endpoint observed
    censored_valid: torch.Tensor      # [B, P], ongoing but true end unobserved
    valid_mask: torch.Tensor          # [B, P]


def build_online_censored_targets(
    points: torch.Tensor,             # [P, 4], point[:,0]=center, point[:,3]=stride
    valid_mask: torch.Tensor,         # [B, P]
    gt_segments: List[torch.Tensor],  # each [N, 2] in grid units
    gt_labels: List[torch.Tensor],    # each [N]
    observed_until: torch.Tensor,     # [B] in same grid units as points
    num_classes: int,
    boundary_radius: float = 1.5,
) -> OnlineCensoredTargets:
    device = points.device
    dtype = points.dtype
    centers = points[:, 0]
    strides = points[:, 3].clamp(min=1.0)
    B, P = valid_mask.shape

    cls_target = torch.zeros(B, P, num_classes, device=device, dtype=dtype)
    actionness = torch.zeros(B, P, 1, device=device, dtype=dtype)
    start_target = torch.zeros(B, P, num_classes, device=device, dtype=dtype)
    end_target = torch.zeros(B, P, num_classes, device=device, dtype=dtype)
    emit_target = torch.zeros(B, P, 1, device=device, dtype=dtype)

    left_reg = torch.zeros(B, P, device=device, dtype=dtype)
    right_reg = torch.zeros(B, P, device=device, dtype=dtype)
    left_valid = torch.zeros(B, P, device=device, dtype=torch.bool)
    right_observed_valid = torch.zeros(B, P, device=device, dtype=torch.bool)
    censored_valid = torch.zeros(B, P, device=device, dtype=torch.bool)

    for b, (segs, labels) in enumerate(zip(gt_segments, gt_labels)):
        if segs is None or len(segs) == 0:
            continue

        tau = observed_until[b].to(device=device, dtype=dtype)
        segs = segs.to(device=device, dtype=dtype)
        labels = labels.to(device=device, dtype=torch.long).clamp(0, num_classes - 1)

        for seg, lab in zip(segs, labels):
            s, e = seg[0], seg[1]
            if s > tau:
                continue

            visible_e = torch.minimum(e, tau)
            completed = bool(e <= tau)

            inside = (centers >= s) & (centers <= visible_e) & valid_mask[b]
            if not inside.any():
                continue

            cls_target[b, inside, lab] = 1.0
            actionness[b, inside, 0] = 1.0

            # Left/start is observable once the start has occurred.
            left_reg[b, inside] = (centers[inside] - s).clamp(min=0)
            left_valid[b, inside] = True

            start_mask = ((centers - s).abs() <= boundary_radius * strides) & valid_mask[b]
            start_target[b, start_mask, lab] = 1.0

            # Right side is censored unless true end has already occurred.
            right_reg[b, inside] = (visible_e - centers[inside]).clamp(min=0)
            if completed:
                right_observed_valid[b, inside] = True
                end_mask = ((centers - e).abs() <= boundary_radius * strides) & valid_mask[b]
                end_target[b, end_mask, lab] = 1.0
                emit_target[b, end_mask, 0] = 1.0
            else:
                censored_valid[b, inside] = True

    return OnlineCensoredTargets(
        cls_target=cls_target,
        actionness=actionness,
        start_target=start_target,
        end_target=end_target,
        emit_target=emit_target,
        left_reg=left_reg,
        right_reg=right_reg,
        left_valid=left_valid,
        right_observed_valid=right_observed_valid,
        censored_valid=censored_valid,
        valid_mask=valid_mask,
    )


def censored_regression_loss(
    pred_lr: torch.Tensor,        # [B, P, 2], nonnegative left/right distances
    targets: OnlineCensoredTargets,
    points: torch.Tensor,         # [P, 4]
    observed_until: torch.Tensor, # [B]
    lambda_bound: float = 0.1,
) -> torch.Tensor:
    centers = points[:, 0][None, :].to(pred_lr)
    tau = observed_until[:, None].to(pred_lr)

    left_loss = F.smooth_l1_loss(
        pred_lr[..., 0][targets.left_valid],
        targets.left_reg[targets.left_valid],
        reduction="mean",
    ) if targets.left_valid.any() else pred_lr.sum() * 0.0

    # Only completed actions supervise final right boundary.
    right_loss = F.smooth_l1_loss(
        pred_lr[..., 1][targets.right_observed_valid],
        targets.right_reg[targets.right_observed_valid],
        reduction="mean",
    ) if targets.right_observed_valid.any() else pred_lr.sum() * 0.0

    # For censored ongoing actions, forbid predicting beyond observed prefix.
    pred_end = centers + pred_lr[..., 1].clamp(min=0)
    no_future_penalty = F.relu(pred_end - tau)
    bound_loss = (no_future_penalty[targets.censored_valid] ** 2).mean() \
        if targets.censored_valid.any() else pred_lr.sum() * 0.0

    return left_loss + right_loss + lambda_bound * bound_loss
```

### 9.2 Selector packet + causal selector

```python
# opentad/models/selectors/causal_frame_selector.py
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FramePacket:
    frames: torch.Tensor          # [B, T, C, H, W] or already encoded [B, T, D]
    frame_times: torch.Tensor     # [B, T], seconds or frame index
    valid_mask: torch.Tensor      # [B, T]
    observed_until: torch.Tensor  # [B]
    meta: Dict


@dataclass
class SelectedTokenPacket:
    tokens: torch.Tensor          # [B, K, D]
    token_times: torch.Tensor     # [B, K]
    selected_indices: torch.Tensor # [B, K]
    selected_mask: torch.Tensor   # [B, K]
    budget_used: torch.Tensor     # [B]
    max_gap: torch.Tensor         # [B]
    ledger: Dict


class CausalFrameSelector(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        max_budget: int = 384,
        max_gap: int = 32,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.max_budget = max_budget
        self.max_gap = max_gap
        self.temperature = temperature

        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        frame_feats: torch.Tensor,       # [B, T, D], causal low-cost features
        frame_times: torch.Tensor,       # [B, T]
        valid_mask: torch.Tensor,        # [B, T]
        budget: Optional[int] = None,
        training_st: bool = True,
    ):
        B, T, _ = frame_feats.shape
        budget = int(budget or self.max_budget)
        budget = min(budget, T)

        x = self.in_proj(frame_feats)
        x = x.masked_fill(~valid_mask[..., None], 0.0)

        # GRU is causal in temporal order.
        h, _ = self.gru(x)
        logits = self.score(h).squeeze(-1).masked_fill(~valid_mask, -1e4)

        if self.training and training_st:
            prob = torch.sigmoid(logits / self.temperature)
            hard = torch.zeros_like(prob)
            topk = torch.topk(prob.masked_fill(~valid_mask, -1.0), k=budget, dim=1).indices
            hard.scatter_(1, topk, 1.0)
            select_mask = hard + prob - prob.detach()
        else:
            select_mask = torch.zeros_like(logits)
            topk = torch.topk(logits.masked_fill(~valid_mask, -1e4), k=budget, dim=1).indices
            select_mask.scatter_(1, topk, 1.0)

        select_mask = self._enforce_max_gap(select_mask, valid_mask)

        budget_loss = F.relu(select_mask.sum(dim=1) - budget).pow(2).mean()
        anti_collapse_loss = self._anti_collapse_loss(select_mask, valid_mask)

        aux = {
            "selector_budget_loss": budget_loss,
            "selector_anti_collapse_loss": anti_collapse_loss,
            "selector_logits": logits,
        }
        return select_mask, aux

    def _enforce_max_gap(self, select_mask: torch.Tensor, valid_mask: torch.Tensor):
        # Deterministic guard: add scaffold samples if a long valid interval has no selected point.
        guarded = select_mask.clone()
        B, T = guarded.shape
        for b in range(B):
            valid_idx = torch.where(valid_mask[b])[0]
            if valid_idx.numel() == 0:
                continue
            last = int(valid_idx[0])
            guarded[b, last] = 1.0
            for t in valid_idx.tolist():
                if t - last >= self.max_gap:
                    guarded[b, t] = 1.0
                    last = t
                elif guarded[b, t] > 0.5:
                    last = t
            guarded[b, int(valid_idx[-1])] = 1.0
        return guarded

    @staticmethod
    def _anti_collapse_loss(select_mask: torch.Tensor, valid_mask: torch.Tensor):
        # Penalize all selections collapsing into a tiny prefix/suffix.
        denom = valid_mask.sum(dim=1).clamp(min=1).float()
        positions = torch.arange(select_mask.shape[1], device=select_mask.device).float()[None, :]
        mean_pos = (positions * select_mask).sum(dim=1) / select_mask.sum(dim=1).clamp(min=1)
        center = denom / 2.0
        return ((mean_pos - center).abs() / denom).mean()
```

### 9.3 Selected-frame token packing

```python
# opentad/models/utils/selected_token_packet.py
import torch

from opentad.models.selectors.causal_frame_selector import SelectedTokenPacket


def pack_selected_tokens(
    frame_feats: torch.Tensor,      # [B, T, D]
    frame_times: torch.Tensor,      # [B, T]
    select_mask: torch.Tensor,      # [B, T], 0/1 or ST mask
    valid_mask: torch.Tensor,       # [B, T]
    meta: dict,
) -> SelectedTokenPacket:
    B, T, D = frame_feats.shape
    hard_mask = (select_mask > 0.5) & valid_mask
    K = int(hard_mask.sum(dim=1).max().item())
    K = max(K, 1)

    tokens = frame_feats.new_zeros(B, K, D)
    token_times = frame_times.new_zeros(B, K)
    selected_indices = torch.full((B, K), -1, device=frame_feats.device, dtype=torch.long)
    selected_valid = torch.zeros(B, K, device=frame_feats.device, dtype=torch.bool)

    max_gap = []
    budget_used = []

    for b in range(B):
        idx = torch.where(hard_mask[b])[0]
        if idx.numel() == 0:
            idx = torch.where(valid_mask[b])[0][-1:].clone()

        idx = idx.sort().values
        k = idx.numel()
        tokens[b, :k] = frame_feats[b, idx]
        token_times[b, :k] = frame_times[b, idx]
        selected_indices[b, :k] = idx
        selected_valid[b, :k] = True

        budget_used.append(k)
        if k > 1:
            max_gap.append((idx[1:] - idx[:-1]).max())
        else:
            max_gap.append(torch.tensor(0, device=frame_feats.device))

    return SelectedTokenPacket(
        tokens=tokens,
        token_times=token_times,
        selected_indices=selected_indices,
        selected_mask=selected_valid,
        budget_used=torch.as_tensor(budget_used, device=frame_feats.device),
        max_gap=torch.stack(max_gap),
        ledger={
            "selector": "causal_frame_selector",
            "selected_indices": selected_indices.detach().cpu().tolist(),
        },
    )
```

### 9.4 Irregular-time proposal decode

```python
# opentad/models/utils/irregular_time_decode.py
import torch


def local_time_scale(token_times: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """
    token_times: [B, K] seconds
    returns local dt scale [B, K]
    """
    B, K = token_times.shape
    dt = token_times.new_ones(B, K)

    if K == 1:
        return dt

    left = token_times[:, 1:] - token_times[:, :-1]
    left = left.clamp(min=1e-6)

    dt[:, 0] = left[:, 0]
    dt[:, -1] = left[:, -1]
    if K > 2:
        dt[:, 1:-1] = 0.5 * (left[:, :-1] + left[:, 1:])

    dt = dt.masked_fill(~valid_mask, 1.0)
    return dt


def decode_irregular_proposals(
    token_times: torch.Tensor,       # [B, K], seconds
    pred_lr: torch.Tensor,           # [B, K, 2], normalized distances
    valid_mask: torch.Tensor,        # [B, K]
    observed_until: torch.Tensor,    # [B], seconds
) -> torch.Tensor:
    scale = local_time_scale(token_times, valid_mask)
    left = pred_lr[..., 0].clamp(min=0) * scale
    right = pred_lr[..., 1].clamp(min=0) * scale

    start = token_times - left
    end = token_times + right

    # Online no-future output guard.
    end = torch.minimum(end, observed_until[:, None])
    start = torch.minimum(start, end)
    start = start.clamp(min=0.0)

    proposals = torch.stack([start, end], dim=-1)
    return proposals.masked_fill(~valid_mask[..., None], 0.0)
```

### 9.5 Emission ledger validation

```python
# opentad/utils/online_ledger_validation.py
from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class LedgerViolation:
    video: str
    stream_key: str
    kind: str
    row_index: int
    message: str


def validate_emission_rows(rows: Dict[str, List[dict]], strict: bool = True):
    violations: List[LedgerViolation] = []
    last_emit_by_stream = {}

    for video, video_rows in rows.items():
        for i, row in enumerate(video_rows):
            stream_key = row.get("stream_key", f"video={video}")
            emit = float(row.get("emit_frame", -1))
            end = float(row.get("end_frame", -1))
            start = float(row.get("start_frame", 0))
            used_frames = row.get("used_frame_indices", [])

            if end > emit:
                violations.append(LedgerViolation(video, stream_key, "future_end", i, f"end={end} > emit={emit}"))

            if start > end:
                violations.append(LedgerViolation(video, stream_key, "invalid_segment", i, f"start={start} > end={end}"))

            if used_frames and max(used_frames) > emit:
                violations.append(
                    LedgerViolation(video, stream_key, "future_packet", i, f"max_used={max(used_frames)} > emit={emit}")
                )

            prev = last_emit_by_stream.get(stream_key)
            if prev is not None and emit < prev:
                violations.append(
                    LedgerViolation(video, stream_key, "non_monotonic_emit", i, f"emit={emit} < prev={prev}")
                )
            last_emit_by_stream[stream_key] = emit

            if float(row.get("latency_sec", 0.0)) < -1e-6:
                violations.append(LedgerViolation(video, stream_key, "negative_latency", i, "latency_sec < 0"))

    if strict and violations:
        msg = "\n".join(f"{v.video} {v.stream_key} {v.kind}: {v.message}" for v in violations[:20])
        raise ValueError(f"Emission ledger validation failed with {len(violations)} violations:\n{msg}")

    return violations
```

---

## 10. 风险清单

### P0 风险：full endpoint supervision leakage

这是最大论文风险。当前 forward 可以 causal，但训练 target 不是 causal。`_build_online_branch_targets` 直接用 `segment[1]` 生成 end/emit target，`forward_train` 也继续调用 base anchor-free full segment loss。([GitHub][8])

**修复优先级：P0。**

### P0 风险：windowed streaming 被误写成 continuous streaming

当前实现可以叫 streaming-safe window evaluation，但不应写成每帧连续在线 agent。`FrameWindowDataset` 明确是 sliding-window raw-frame dataset，并将窗口边界写入 metadata。([GitHub][3])

**修复优先级：P0/P1。** 论文中必须明确 windowed setting，或真正实现 per-frame loop。

### P0 风险：standard mAP 与 latency 未统一

现有 mAP evaluator 仍按传统 segment AP 计算。ledger/latency 是额外 summary，不参与 AP 匹配。([GitHub][11])

**修复优先级：P1。** 需要 Online-mAP / Causal-AP。

### P1 风险：VideoMAE route 过早 claim

仓库明确说 VideoMAE adapter route 是 validation candidate，stub backbone 是 contract-only。([GitHub][12])

**修复优先级：P1。** 在 stub 关闭前，不能把 VideoMAE 写进主贡献。

### P1 风险：memory 重新引入 overlapping-window leak

当前 P0/P1 通过 `memory_size=0` 避免 memory leak。MATRHead 虽有 stream memory reuse/reset 逻辑，但只在 contiguous non-overlapping windows 下可复用；重开 memory 必须做强 audit。([GitHub][4])

### P1 风险：DDP eval state split

测试和 README 都说明 streaming-safe eval 目前 single-rank。([GitHub][15])

### P2 风险：adaptive selector 变成 action coverage

如果 selector 只优化 actionness / classification，很可能覆盖动作内部而不是边界与 emission 证据。最终要把 selector utility 定义为 boundary/emission/localization utility，而不是 action coverage。

### P2 风险：irregular timestamp decode 错误导致 high-IoU 崩溃

adaptive selection 后，token spacing 不均匀。若继续用 regular point generator stride decode，mAP@0.6/0.7 会成为最敏感的失败指标。

---

## 11. Go / No-Go milestones

### Milestone 0：P1-full-60 baseline audit

**Go：**

* full THUMOS val average mAP 有稳定非零结果；
* ledger violation = 0；
* latency summary 写出；
* no NaN；
* overfit smoke 通过；
* raw-prediction shortcut disabled；
* single-rank eval 清楚记录。

**No-Go：**

* mAP 接近 0 或 emission 数量异常爆炸；
* future_end_violations > 0；
* 训练无法稳定；
* 结果只来自 pilot allow-list 却被当 full split。

### Milestone 1：online-censored training

**Go：**

* censored target 相比 full endpoint target 在 latency/no-future/high-IoU 上至少一项显著改善，且 mAP 不明显崩；
* whole-model future perturb pass；
* false early close rate 下降。

**No-Go：**

* censored training 全面低于 full endpoint，且无 latency/no-future收益；
* model 只会输出极短 proposal；
* emit branch 学不起来。

### Milestone 2：emission branch

**Go：**

* emit gating 降低 duplicate / late false positives；
* matched-GT latency p90/p95 改善；
* mAP 不因过强 gating 明显下降。

**No-Go：**

* emit branch 只是 score threshold 的重复实现；
* gating 只减少 prediction 数量，AP/latency无真实收益。

### Milestone 3：adaptive selector

**Go：**

* 在 50% 或 25% budget 下超过 uniform sparse；
* max-gap 受控；
* boundary support 提升；
* compute saving 有真实 measured speed/FLOPs 支撑。

**No-Go：**

* learned selector 不如 uniform；
* selector collapse 到片头/片尾；
* heavy encoder 实际仍处理 dense frames；
* 只减少 token 后处理，不减少 visual compute。

### Milestone 4：irregular-time head

**Go：**

* true timestamp decode 明显优于 treating selected tokens as uniform grid；
* mAP@0.6/0.7 不崩；
* variable fps / jitter perturb 稳定。

**No-Go：**

* low IoU 上升但 high IoU 下降；
* NMS/seconds conversion 与 emission ledger 坐标不一致；
* selected timestamp 与 proposal decode 无法 round-trip。

---

**最终路线建议：**先不要急着写 “Budgeted Adaptive Online TAD” 主论文。当前最正确的推进顺序是：**P1-full-60 真实审计 → online-censored target → emission/hazard branch → adaptive selector → irregular timestamp head**。只要 Stage 1 不能证明 full endpoint label shortcut 被消除且性能不崩，后面的 adaptive selection 和 AdaTAD 接入都只是建立在不干净监督上的工程堆叠。

[1]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/tree/codex/online-tad-clean-20260702 "GitHub - yuzbo/OpenTAD_OnlineTADClean_20260702 · GitHub"
[2]: https://raw.githubusercontent.com/yuzbo/OpenTAD_OnlineTADClean_20260702/codex/online-tad-clean-20260702/opentad/utils/online_protocol.py "raw.githubusercontent.com"
[3]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/opentad/datasets/raw_frame.py "OpenTAD_OnlineTADClean_20260702/opentad/datasets/raw_frame.py at codex/online-tad-clean-20260702 · yuzbo/OpenTAD_OnlineTADClean_20260702 · GitHub"
[4]: https://raw.githubusercontent.com/yuzbo/OpenTAD_OnlineTADClean_20260702/codex/online-tad-clean-20260702/configs/causaltad/thumos_siglip2_matr_ontad_p0.py "raw.githubusercontent.com"
[5]: https://raw.githubusercontent.com/yuzbo/OpenTAD_OnlineTADClean_20260702/codex/online-tad-clean-20260702/configs/causaltad/thumos_siglip2_matr_ontad_p1.py "raw.githubusercontent.com"
[6]: https://raw.githubusercontent.com/yuzbo/OpenTAD_OnlineTADClean_20260702/codex/online-tad-clean-20260702/opentad/models/projections/causal_temporalmaxer_proj.py "raw.githubusercontent.com"
[7]: https://raw.githubusercontent.com/yuzbo/OpenTAD_OnlineTADClean_20260702/codex/online-tad-clean-20260702/tests/test_causal_temporalmaxer_proj.py "raw.githubusercontent.com"
[8]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/opentad/models/dense_heads/matr_head.py "OpenTAD_OnlineTADClean_20260702/opentad/models/dense_heads/matr_head.py at codex/online-tad-clean-20260702 · yuzbo/OpenTAD_OnlineTADClean_20260702 · GitHub"
[9]: https://raw.githubusercontent.com/yuzbo/OpenTAD_OnlineTADClean_20260702/codex/online-tad-clean-20260702/opentad/models/detectors/single_stage.py "raw.githubusercontent.com"
[10]: https://raw.githubusercontent.com/yuzbo/OpenTAD_OnlineTADClean_20260702/codex/online-tad-clean-20260702/opentad/cores/test_engine.py "raw.githubusercontent.com"
[11]: https://raw.githubusercontent.com/yuzbo/OpenTAD_OnlineTADClean_20260702/codex/online-tad-clean-20260702/opentad/evaluations/mAP.py "raw.githubusercontent.com"
[12]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/tree/codex/online-tad-clean-20260702/configs/causaltad "OpenTAD_OnlineTADClean_20260702/configs/causaltad at codex/online-tad-clean-20260702 · yuzbo/OpenTAD_OnlineTADClean_20260702 · GitHub"
[13]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/tests/test_claim_hygiene.py "OpenTAD_OnlineTADClean_20260702/tests/test_claim_hygiene.py at codex/online-tad-clean-20260702 · yuzbo/OpenTAD_OnlineTADClean_20260702 · GitHub"
[14]: https://raw.githubusercontent.com/yuzbo/OpenTAD_OnlineTADClean_20260702/codex/online-tad-clean-20260702/configs/causaltad/thumos_siglip2_motion_matr_ontad_p2.py "raw.githubusercontent.com"
[15]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/tests/test_online_emission_protocol.py "OpenTAD_OnlineTADClean_20260702/tests/test_online_emission_protocol.py at codex/online-tad-clean-20260702 · yuzbo/OpenTAD_OnlineTADClean_20260702 · GitHub"
