# Pro 深度讨论 Prompt：Full PETAL、Q2 与 CRS-EPS 高效训练协议

以下内容可直接完整提交给 GPT-5 Pro / GPT-5.5 Pro。建议启用联网检索、
Deep Research、GitHub 浏览和代码阅读能力。不要删减不可变仓库锚点、证据分级、
任务边界、两轮互动规则、成本分母审查、最强拒稿和最终机器可读裁决。

---

## BEGIN PROMPT

你是一名同时具备以下角色的最高强度研究与实现审查者：

- CVPR / ICCV / ECCV Senior PC；
- CCF-A 期刊高级编辑；
- fully supervised Online Temporal Action Detection / Localization 专家；
- causal streaming video、stateful sequence training 和 truncated BPTT 专家；
- DETR / Hungarian assignment / persistent query / TrackFormer / MOTR 专家；
- 高效视频训练、importance sampling、event-centric sampling 和统计估计专家；
- PyTorch、OpenTAD、AMP、训练事务、Slurm 和实验可复现性专家；
- 会主动寻找 future leakage、GT state initialization、sampling bias、错误成本分母、
  train/eval state mismatch 和多论文重构的严厉审稿人。

请使用中文回答，保留论文标题、模型名、公式、代码标识符和 URL 的原文。

你的任务不是替作者辩护，不是重新发散任务，也不是立即写一个过度自信的训练
方案。你必须基于公开 GitHub 的不可变提交，先核验当前代码究竟实现了什么，再
裁决以下核心冲突：

> 项目此前选择了 CRS-EPS，即 instance-aware causal event-centric prefix-episode
> training；但当前 Q2 实现使用完整视频的连续 frozen-feature chunks，并在完整视频
> 结束时才产生一个 optimizer event。当前路线是合理的 one-factor gold reference，
> 还是已经把高效训练主线意外降级成普通 cache-only 全序列训练？应该如何把
> CRS-EPS 接回 fixed-vs-rematch Q2，同时保持严格可识别性、在线因果性、状态连续性
> 和可审计的成本收益？

必须允许最终结论为：

```text
GO-HYBRID-PROTOCOL
REVISE-BEFORE-IMPLEMENTATION
KEEP-FULL-CHRONOLOGICAL-ONLY
KILL-CRS-EPS
KILL-FULL-PETAL-ROUTE
```

在没有正式实验结果时，任何 `GO` 只表示方案值得实现和做 falsification test，
不表示 Full PETAL 有效、创新成立或可开始 raw-video 正式训练。

# 0. 公开仓库与不可变审查锚点

## 0.1 Repository

首先打开并核验：

```text
Repository:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702

Branch:
codex/full-petal-implementation

Branch URL:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/tree/codex/full-petal-implementation

Immutable implementation/research anchor:
f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb

Anchor commit URL:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/commit/f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb

Pinned tree URL:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/tree/f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb
```

本 Prompt 的发布 commit 必然晚于上述 anchor，因为 Prompt 本身需要提交。科学代码、
配置、测试和当前研究状态的审查基线固定为
`f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb`。

如果 branch HEAD 更晚：

1. 报告实际 HEAD；
2. 检查 `f4ea53e..HEAD` 的 diff；
3. 如果只新增本 Prompt 或 provenance 文档，继续使用 `f4ea53e`；
4. 如果科学代码、配置、数据协议、测试或结果发生变化，明确列出变化；
5. 不得把未查看的 branch HEAD 当成已验证事实；
6. 不得从默认分支推断此分支的实现。

## 0.2 证据标签

每项重要判断必须标记以下标签之一：

```text
[CODE-VERIFIED]       由 pinned GitHub 代码直接支持
[DOC-VERIFIED]        由 pinned 项目文档直接支持，但不是运行结果
[PRIMARY-VERIFIED]    由 primary paper 或 official code 支持
[ARTIFACT-ASSERTED]   由作者记录的外部 B0/Slurm artifact 支持，未公开复现
[INFERENCE]           从多项证据推导
[UNKNOWN]             仓库和 primary source 都无法回答
[REQUIRES-EXPERIMENT] 必须由实验决定
```

约束：

- tests pass 不是科学 claim pass；
- cache 存在不是 raw-video end-to-end；
- smoke 可运行不是机制有效；
- 找不到直接工作不能写成“首次”；
- 作者文档中的建议不是代码事实；
- 任何数值阈值都要区分 resource gate、non-inferiority margin 和统计显著性。

# 1. 必读文件

请按顺序阅读，不得只读 README 或本 Prompt。

## 1.1 当前状态与历史决策

```text
RTK.md
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb/RTK.md

research-wiki/query_pack.md
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb/research-wiki/query_pack.md

research-wiki/decision_register.md
重点阅读 DR-006、DR-007、DR-008、DR-009、DR-010、DR-017、DR-025、
DR-026、DR-027、DR-028
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb/research-wiki/decision_register.md

research-wiki/discussion_timeline.md
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb/research-wiki/discussion_timeline.md

research-wiki/ideas/crs-eps-training.md
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb/research-wiki/ideas/crs-eps-training.md

research-wiki/ideas/state-transition-sampling.md
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb/research-wiki/ideas/state-transition-sampling.md

research-wiki/ideas/feature-cache-stage1.md
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb/research-wiki/ideas/feature-cache-stage1.md

research-wiki/ideas/lora-stage2.md
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb/research-wiki/ideas/lora-stage2.md

research-wiki/ideas/petal-ontad.md
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb/research-wiki/ideas/petal-ontad.md

research-wiki/experiments/formal-training-none.md
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb/research-wiki/experiments/formal-training-none.md
```

## 1.2 最新独立审查

```text
Full review record:
https://raw.githubusercontent.com/yuzbo/OpenTAD_OnlineTADClean_20260702/f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb/PRO_FULL_PETAL_CODE_SCIENCE_REVIEW_20260715.md

Independent absorption and disagreements:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb/PRO_FULL_PETAL_CODE_SCIENCE_REVIEW_ABSORPTION_20260715.md

Execution gates:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb/FULL_PETAL_EXECUTION_GATES.md

Trust model:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb/FULL_PETAL_TRUST_MODEL.md
```

## 1.3 当前 Q2 配置

```text
Base:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb/configs/causaltad/thumos_pes_q2_base.py

Fixed binding:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb/configs/causaltad/thumos_pes_q2_persist_fixed.py

Prefix rematching:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb/configs/causaltad/thumos_pes_q2_persist_rematch.py
```

## 1.4 数据、监督、模型与训练循环

```text
StreamingFeatureDataset:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb/opentad/datasets/streaming_feature.py

Streaming dataloader builder:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb/opentad/datasets/builder.py

Prefix trajectory supervision:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb/opentad/utils/prefix_trajectory_supervision.py

Persistent trajectory detector:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb/opentad/models/detectors/persistent_trajectory_ontad.py

Persistent event-set head:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb/opentad/models/dense_heads/persistent_event_set_head.py

Training engine:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb/opentad/cores/train_engine.py

Evaluation engine:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb/opentad/cores/test_engine.py

Online metrics:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb/opentad/evaluations/full_petal_metrics.py
```

## 1.5 Launch、结果门禁和关键测试

```text
Ticket builder:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb/tools/build_full_petal_launch_ticket.py

Launch validator:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb/opentad/utils/full_petal_launch.py

Slurm submitter:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb/tools/remote/submit_full_petal_q2_n16r4.sh

Result gate:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb/tools/check_full_petal_results.py

B0 runner:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb/tools/run_full_petal_b0.py

Mandatory tests:
tests/test_streaming_feature_dataset.py
tests/test_prefix_trajectory_supervision.py
tests/test_full_petal_detector_contracts.py
tests/test_full_petal_training_transaction.py
tests/test_full_petal_training_cost_controls.py
tests/test_full_petal_launch_gate.py
tests/test_full_petal_launch_contracts.py
tests/test_full_petal_result_gate.py
tests/test_full_petal_data_contract.py
```

# 2. 不可修改的任务边界

本项目必须留在标准 fully supervised completion-triggered On-TAD / On-TAL：

```text
Video stream: V = {x_t}_{t=1}^T
Filtration: F_t = sigma(x_1, ..., x_t, past model state, past immutable outputs)
GT instance: psi_i = (s_i, e_i, c_i)
Emission: d_j = (s_hat_j, e_hat_j, c_hat_j, score_j, tau_emit_j)
Constraints: e_hat_j <= tau_emit_j; d_j is immutable after emission
```

不可通过以下方式逃离问题：

- 改成 Online TAS、OAD frame classification、VideoQA 或 semantic memory；
- 改成 zero-shot / open-vocabulary 主任务；
- 新增传感器、新标签或新 benchmark 作为主要创新；
- 允许推理读取未来帧、EOF、完整视频长度或 GT；
- 使用 offline NMS 回写历史输出；
- 把训练 annotation-guided sampling 错写成“训练不使用未来信息”；
- 把内部可修正状态等同于用户可见输出无限回改。

允许完整训练 annotation 决定“抽取哪个 prefix episode”，但每个 episode 的模型输入、
初始状态重建和计算图必须只包含该 prefix 可获得的信息。最终评测必须是完整时间顺序
流式评测，不得使用抽样 episode 评测。

# 3. 术语和研究对象必须分开

## 3.1 Full PETAL 长期对象

本 Prompt 中 Full PETAL 仅表示待验证的长期工程/研究组合：

```text
causal raw-video encoder or causal visual adapter
+ persistent action-instance slots
+ birth / active / end lifecycle
+ identity-consistent trajectory supervision
+ low-cost causal prefix training
+ standard immutable On-TAD emission
```

当前不得假设它具有整体创新性。最新审查已把完整 package 暂定为
`RECONSTRUCTION` 风险。

## 3.2 Q2 当前科学对象

当前 Q2 只问：

> 在完全相同的 cached features、模型、birth rule、canonical lifecycle、容量、
> optimizer、evaluator 和 inference 下，`fixed_birth_slot` 是否比
> `prefix_rematch_active_pool` 更好，因为固定绑定保留实例身份，而不是因为 mask、
> normalization、negative semantics、sampling distribution 或成本预算不同？

Q2 不是 Full PETAL 整体，也不是 raw-video end-to-end claim。

## 3.3 CRS-EPS 训练协议

历史记录定义的 CRS-EPS 是：

```text
Instance-aware Causal Risk-Set Event-Centric Prefix-Episode Training

192 causal burn-in/context tokens
+ 4--8 consecutive supervised decision bins
+ start/end/ongoing/hard-background/uniform mixture
+ one optimizer event per episode
+ inclusion probability and bias audit
+ full chronological validation/evaluation
```

历史候选混合比例为：

| Episode type | Candidate ratio |
|---|---:|
| endpoint-centered | 20% |
| start-centered | 15% |
| ongoing middle | 10% |
| hard background near boundaries | 15% |
| uniform chronological time | 40% |

这些比例只是待审候选，不是既定真理。

## 3.4 Feature cache

Feature cache 只减少重复视觉前向成本。它不能替代 CRS-EPS，也不能支持
raw-video visual adaptation claim。请明确区分：

```text
representation cost control != temporal sampling protocol
cached feature training != end-to-end visual training
```

# 4. 当前状态：必须逐项核验，不能照抄

以下是项目当前陈述，必须从 pinned commit 核验：

```text
Task: standard fully supervised completion-triggered On-TAD
Input: frozen cached SigLIP2 features, dim 768, stride 8
Chunk size: 64 feature tokens
Memory: 192 tokens
Slots: 4
Cross-chunk numerical state: carried
Cross-chunk gradient: detached
World size: 1
Resume: prohibited for current formal route
Variants: fixed_birth_slot vs prefix_rematch_active_pool
Training order: complete chronological videos
Optimizer boundary: one complete video episode
Raw-video finetuning: false
Formal result: none
GPU profile: none
Formal training: none
```

先前 B0 在代码提交 `6d88610da34a695e07d29c5e08b50fb57d2aa5e9` 上记录
508 tests、0 blocking findings、0 protocol violations，但没有执行真实 submit-shell
组合。`f4ea53e` 只增加审查记录和 wiki 吸收，没有关闭下述 P0：

```text
P0-LAUNCH-WORKDIR:
ticket freezes exact cfg_overrides before submit script creates timestamped RUN_DIR;
submit script later injects --cfg-options work_dir=${RUN_DIR}/work;
pre-CUDA validator requires exact runtime identity equality.
```

因此当前状态必须保持：

```text
PROFILE=BLOCK
FORMAL_TRAINING=BLOCK
```

你应核验 P0，但不要浪费整个审查重新争论一个已经有明确代码证据的问题。

# 5. 第一项任务：重建当前真实训练数据流和成本流

不要先给建议。先从代码画出当前 Q2 的真实执行链：

```text
manifest
-> video/chunk enumeration
-> chronological sampler
-> model runtime state
-> per-token Python scan
-> per-chunk loss accumulation
-> video boundary
-> optimizer/scaler/scheduler step
-> state commit or rollback
```

必须回答：

1. `StreamingFeatureDataset` 是否生成完整视频连续 chunks？
2. `optimizer_events_per_epoch` 是否等于完整视频数，而非 chunk 数？
3. 一个 optimizer event 实际包含多少 temporal tokens、多少 supervised bins、多少 chunks？
4. loss 如何跨 chunks 归一化？长视频是否获得更大梯度权重？
5. 所有 tokens 是否都参与 Python sequential unroll？
6. 减少 optimizer steps 是否真正减少主要成本，还是只把全视频计算合并到一个 step？
7. cross-chunk state detach 对数值状态、梯度长度和长动作学习分别意味着什么？
8. fixed/rematch 是否真的只改变 post-birth loss binding？
9. 当前 route 是否包含 CRS-EPS 的 anchor sampling、burn-in、4--8 supervised bins、
   `pi_t`、IPW/SNIPW 或 ESS？
10. 如果不包含，应将当前 route 命名为什么？

输出一张 accounting table：

| Unit | Current full-chronological cached route | Proposed CRS-EPS | Comparable? | Required normalization |
|---|---:|---:|---|---|
| videos | | | | |
| chunks | | | | |
| temporal tokens | | | | |
| supervised bins | | | | |
| visual forward frames | | | | |
| visual backward frames | | | | |
| optimizer events | | | | |
| wall time | | | | |
| GPU-hours | | | | |
| effective sample size | | | | |

# 6. 第二项任务：审查 CRS-EPS 是否科学有效

## 6.1 形式化目标

请定义完整时间顺序经验风险和 sampled episode 估计量。至少明确：

```text
p(t | video): target chronological decision-time distribution
q(t, z | video): event-centric proposal over anchor t and episode type z
pi_t: inclusion probability
w_t: inverse-probability or self-normalized weight
L_full(theta)
L_episode(theta)
```

必须讨论：

- `pi_t` 能否准确计算；
- 同一决策位置可被多个 anchor window 覆盖时如何计算联合 inclusion probability；
- capped IPW 带来的偏差；
- self-normalization 带来的偏差；
- temporal correlation 和同一视频 cluster dependence；
- class/lifecycle imbalance；
- 每个视频、每个实例、每个 decision bin 应如何加权；
- loss denominator 应按 token、实例、风险集还是 episode 归一化；
- ESS 应全局、按类别、按 lifecycle state、按视频分别报告什么。

不要只写“使用 importance sampling”而不给可执行公式。

## 6.2 Episode 构造

对以下每项给出明确规则：

- anchor 是 start crossing、end crossing、ongoing、hard background 还是 uniform；
- burn-in 长度固定 192、可变，还是由动作跨度决定；
- supervised horizon 是 4--8 连续 bins 还是固定值；
- episode 在视频开头/结尾如何截断；
- 同一动作 start/end 距离很短时如何避免重复统计；
- same-class repeated actions 如何保持 instance identity；
- overlapping actions 如何保证所有相关 birth 都在状态历史中出现；
- hard background 如何定义而不把未来 endpoint 输入模型；
- uniform component 如何保证全支持，避免 `pi_t=0`；
- episode manifest 何时生成、如何冻结、如何跨 variants/seeds 共享。

## 6.3 Active-at-entry、长动作和左删失

这是 blocking scientific issue。请比较至少四种方案：

```text
A. replay from video start
B. bounded 192-token burn-in
C. replay from the earliest relevant observable birth
D. oracle-initialized active slots
```

方案 D 默认禁止，因为它可能把 GT instance state 注入模型。对 A/B/C 分别审查：

- 因果合法性；
- 计算成本；
- 是否遗漏长动作 birth；
- 是否改变 fixed/rematch 的 identity trajectory；
- 是否需要 left-censor mask；
- 如何处理动作 start 早于 burn-in 但 episode 内仍 active/end；
- 如何处理 pre-memory start；
- 如何与完整时间顺序状态做 gold audit。

必须给出一个主方案、一个 fallback 和明确 kill condition。

## 6.4 Burn-in 与梯度

审查以下候选：

```text
burn-in forward only -> detach -> supervised suffix backward
burn-in with temporal backward but frozen visual encoder
selective truncated BPTT through final K burn-in tokens
```

回答：

- 哪个方案适合 cached Q2；
- 哪个方案适合 raw-video LoRA Stage 2；
- detach 后 persistent state 学到什么、学不到什么；
- 如何避免状态值正确但状态构造机制从未获得梯度；
- 是否需要 controlled K ablation；
- 是否允许缓存 hidden state；若允许，如何绑定 model/optimizer/checkpoint revision 并避免 stale state；
- 为什么不能默认使用模型变化前生成的 hidden-state cache。

# 7. 第三项任务：保持 Q2 严格 one-factor

CRS-EPS 接入后，fixed 与 rematch 必须共享：

```text
same immutable episode manifest
same anchor frames and episode types
same inclusion probabilities and weights
same burn-in and supervised ranges
same source features
same initialization
same data order
same RNG draws and dropout masks where technically possible
same optimizer/scheduler/AMP
same birth assignment
same canonical lifecycle
same negative/ignore/censor semantics
same loss denominators
same inference path and thresholds
```

唯一允许差异：

```text
post-birth target-to-slot loss binding
```

请逐行审查当前代码是否已经满足这一点，并设计完整 paired trace。trace 至少包含：

```text
episode_id
video_id
source and supervised frame ranges
episode type
pi_t and final weight
RNG identity
birth assignments
canonical lifecycle bindings
loss bindings
birth/alive/end risk masks
negative/ignore/censor masks
per-loss numerator and denominator
reset/retire/refractory events
slot exhaustion and overflow
rematch swaps
loss values
gradient checksum or cosine
inference output checksum
```

必须回答：

1. annotation-guided episode selection是否会与 fixed/rematch 交互；
2. fixed binding 的历史依赖是否使随机 episode 无法重建；
3. rematch 的活跃池是否在 bounded burn-in 中与 full stream 不同；
4. 同一 episode list 是否足以保证 one-factor，还是还要 matched runtime state；
5. 哪些 full-stream gold traces 必须与 episode traces 对齐；
6. 如果无法做到严格 one-factor，Q2 应如何改名或降级。

# 8. 第四项任务：比较三种训练协议

必须比较，不得预设答案：

```text
Option A: full chronological cached training only
Option B: CRS-EPS only
Option C: CRS-EPS main training + full-stream gold audit and/or short chronological calibration
```

输出：

| Criterion | A | B | C |
|---|---|---|---|
| scientific fidelity | | | |
| state fidelity | | | |
| Q2 identifiability | | | |
| sampling bias | | | |
| cost | | | |
| implementation complexity | | | |
| raw-video extensibility | | | |
| reviewer defensibility | | | |

对于 Option C，必须具体裁决：

- full-stream 数据仅作无梯度 audit，还是参与训练；
- 是否需要 periodic chronological replay；
- 是否需要最后短程 chronological finetune；
- calibration 是否只调 threshold，还是更新模型；
- 如何避免根据结果临时选择混合比例；
- 如何把 full-stream 计算纳入总 GPU-hour 预算。

# 9. 第五项任务：重新设计成本 profile

当前配置使用：

```text
50 warmup optimizer events
200 measured optimizer events
```

但 full-video route 的一个 optimizer event 与 CRS-EPS 的一个 optimizer event 含义不同。
必须判断该 profile 是否错误或不足。

请设计一个公平 profile contract，至少同时报告：

```text
wall-clock seconds
GPU-hours
peak VRAM
data wait time
temporal forward tokens
temporal backward tokens
visual forward frames
visual backward frames
supervised bins
episodes
videos covered
optimizer events
examples or weighted ESS
throughput per token
throughput per supervised bin
time to a fixed effective amount of supervision
```

必须回答：

1. profile 应固定 optimizer events、tokens、supervised bins、videos、wall time，还是多轴报告？
2. 两种 route 的 warmup 如何匹配？
3. data loading/cache 热度如何控制？
4. Python sequential unroll 占比如何测量？
5. AMP BF16、gradient accumulation 和 scheduler 对比如何固定？
6. full-video route 太慢时，怎样用 bounded reference profile 而不选择性截断？
7. 什么证据才支持“CRS-EPS 降低训练成本”的 claim？
8. 速度提升但 sampled/full objective 偏差过大时是否必须 kill？

不要沿用任意 `20%` 或 `-1.0 pp` 阈值。先定义 primary cost endpoint、预期方差、
minimum meaningful effect 和不劣性边界。

# 10. 第六项任务：数据、cache 与真实在线时钟

必须审查：

- feature manifest 是否绑定 extractor repository commit、config、checkpoint SHA、
  processor、normalization、dtype、stride 和 raw-frame support interval；
- `packet_recent_frame` 是否有 raw-future invariance 证据；
- cache token 的 source frame 与真正 availability time 是否不同；
- chunk size 64、stride 8、30 FPS 时，一个 transport chunk 覆盖约 17.1 秒；
- 模型在 chunk 内逐 token scan 是否足以支持 low-latency claim；
- `emit_time_sec=emit_frame/fps` 是否只是 source clock，而非 availability/wall clock；
- training chunk size 与 serving/evaluation packet size 是否必须分离；
- 评测是否应 stepwise 或 micro-packet；
- source、availability、decision、completion wall-clock 应如何记录；
- cache 可否用于机制训练，但最终 raw-stream evaluation 重新运行视觉 encoder。

此外审查 211-versus-213 reporting population：

- 两个差异视频是谁；
- 为什么排除；
- 所有 baseline 是否使用同一 population；
- 211 结果应如何命名；
- 这是否阻塞 cost profile、机制训练或只阻塞正式 reporting。

# 11. 第七项任务：Stage 2 raw-video LoRA 是否可行

只有 cached Q2 机制和 CRS-EPS surrogate 通过后，才讨论 Stage 2。请设计但不得授权
当前直接训练。

候选 Stage 2：

```text
causal raw-frame episode
-> visual encoder with top-block LoRA/adapters
-> causal temporal persistent slots
-> burn-in forward
-> selective visual backward on a declared frame set
-> full chronological raw-stream evaluation
```

必须审查：

- context frames是否需要视觉梯度；
- 只对 supervised suffix 视觉反传是否会形成 representation mismatch；
- 是否需要 ETAD-style selective snippet gradients；
- LoRA target modules、rank 和学习率如何预注册；
- optimizer coverage、nonzero gradient 和 parameter delta 如何审计；
- base visual weights如何证明冻结；
- raw-video episode sampling是否复用 cached CRS-EPS manifest；
- cache 在 LoRA 参数改变后为什么不能继续作为 trainable visual representation；
- frozen-vs-LoRA 的 2x2 实验如何区分视觉适配与 persistent binding；
- 何种 cached result 才值得花费 Stage 2 GPU-hours；
- 什么结果必须永久阻止 full-tower finetuning。

# 12. 第八项任务：fresh literature review

必须联网检索到审查当日，仅使用 primary paper 和 official code 支撑技术判断。

至少覆盖：

## 12.1 Direct On-TAD / On-TAL

```text
OAT
SimOn
MATR
HAT
ActionSwitch
OnPoint
OZ-TAL
2025--2026 新出现的 Online TAL/TAD
```

## 12.2 Raw-video causal/online training

```text
E2E-LOAD
StreamFormer
OnlineTAS clip/state training
stateful streaming video models
truncated BPTT for long videos
```

## 12.3 Efficient TAL/video training

```text
ETAD
BasicTAD / E2E-TAD
Re2TAL
TALLFormer
LoSA
AdaTAD / TIA
selective snippet-gradient methods
importance or hard-example sampling in TAL
```

## 12.4 Persistent queries and identity

```text
TrackFormer
MOTR
MeMOTR
online VIS persistent query training
query birth/death/re-identification training
```

建议检索式：

```text
"online temporal action localization" efficient training
"online temporal action detection" prefix training
"temporal action localization" event-centric sampling
"temporal action detection" importance sampling training
"streaming video" truncated BPTT action detection
"stateful online video" episode training
"persistent query" action localization
"selective gradient" temporal action localization
"raw video" online temporal action localization end-to-end
```

输出 competition/ingredient matrix：

| Work | Task | Online inference | Training unit | State reconstruction | Sampling | Bias correction | Visual gradients | Persistent identity | Relevance/threat |
|---|---|---|---|---|---|---|---|---|---|

必须判断：

- CRS-EPS 是否只是已有高效采样技术的组合；
- 它应是 cost surrogate、engineering contribution，还是可能的算法贡献；
- 即使 CRS-EPS 新颖性很低，它是否仍是验证 Full PETAL 的必要训练协议；
- 哪个 prior work 是最强训练方法 baseline；
- 哪个 multi-paper combination 可以最容易重构整个 Full PETAL。

# 13. 第九项任务：Claim Map

建立以下 claim map：

```text
C0: Current Q2 implementation is a strict cached-feature causal On-TAD route.
C1: Current full-chronological cached training is a valid unbiased gold reference.
C2: CRS-EPS approximates full chronological training at materially lower cost.
C3: Fixed post-birth binding outperforms prefix rematching under a strict one-factor design.
C4: Any gain is identity-linked rather than sampling/mask/normalization linked.
C5: The mechanism reduces duplicate/fragmentation/same-class overlap errors without recall collapse.
C6: A positive cached Q2 result justifies raw-video LoRA Stage 2.
C7: Raw-video CRS-EPS permits end-to-end causal adaptation at acceptable cost.
C8: Full PETAL is more than TrackFormer + causal video features + temporal heads.
```

输出：

| Claim | Exact scope | Current evidence | Closest prior | Required baseline | Required metric | Falsifier | Minimum evidence | Paper status |
|---|---|---|---|---|---|---|---|---|

状态只能使用：

```text
code-supported
protocol-supported
unproven
partially identifiable
not identifiable
collapsed by prior art
requires experiment
```

特别禁止从 C3 自动推出 C6，从 C6 自动推出 C7，或从系统可运行自动推出 C8。

# 14. 第十项任务：最小实验闭环

先给最小闭环，不要立即扩张成完整论文矩阵。

必须至少讨论：

```text
G0: tiny full-chronological cached gold subset
G1: CRS-EPS fixed binding
G2: CRS-EPS rematch
G3: uniform-prefix episode control
G4: reset-state or insufficient-burn-in control
G5: simple recurrent/persistent control if Q2 survives
G6: FRESH and faithful Temporal TrackFormer only at the correct stage
```

将 baseline 分为：

```text
must-have before first profile
must-have before first scientific run
must-have only if positive
paper-stage only
```

要求：

- fixed/rematch 先做完全配对的一种子 mechanism kill；
- 三种子只能作为低成本 kill test；
- 保留路线至少五种子、paired uncertainty 和 dense/overlap dataset；
- full chronological validation/evaluation 始终保留；
- 不允许根据第一种子结果再改 episode mixture；
- 不允许用 custom OnlineAP 替代 standard tIoU mAP；
- 不允许只报告速度，不报告 sampled/full fidelity；
- 不允许只报告 mAP，不报告目标 identity-linked error。

## 14.1 指标

至少包括：

```text
standard temporal mAP at tIoU 0.3:0.7
completion delay relative to matched GT end
recall and unmatched FN
late FP
duplicate FP
fragmentation
same-class repeated/concurrent subset
overlap subset
long-action and active-at-entry subset
chunk-crossing subset
slot exhaustion and rematch swaps
calibration
sampled/full loss discrepancy
gradient cosine or direction agreement
state occupancy divergence
ESS by class and lifecycle state
wall time, VRAM, tokens, frames, optimizer events, GPU-hours
```

## 14.2 统计

设计：

- paired seeds；
- paired per-video bootstrap；
- confidence intervals；
- effect sizes；
- non-inferiority for sampled/full quality；
- minimum meaningful cost reduction；
- multiple-comparison control；
- failed run / OOM / timeout rules；
- predeclared early kill；
- resource gate 与科学 significance 的分离。

# 15. 第十一项任务：test-to-claim 与 adversarial cases

建立表格：

| Claim/invariant | Existing test | What it proves | What it does not prove | Required new test |
|---|---|---|---|---|

至少覆盖：

- complete-video optimizer boundary；
- current absence/presence of CRS-EPS；
- immutable episode manifest；
- inclusion probability correctness；
- duplicate anchor coverage；
- active-at-entry without oracle state；
- long action beyond burn-in；
- same-class overlap；
- fixed/rematch identical episode/RNG/weights；
- loss numerator/denominator equality；
- full-stream vs episode state equivalence；
- full-stream vs sampled gradient comparison；
- future suffix perturbation；
- chunk vs stepwise inference；
- availability clock；
- feature extractor provenance；
- 211/213 population；
- launch `work_dir` identity；
- profile denominator comparability；
- Stage 2 LoRA parameter updates。

提出至少 15 个可能让训练“看起来有效”但论文结论错误的 adversarial cases。

# 16. 第十二项任务：具体实现路线

如果结论不是 kill，请给出 file-level implementation plan。至少说明：

- 新 dataset/sampler 类的职责和文件位置；
- episode manifest schema；
- manifest build/audit CLI；
- 如何与 `ChronologicalStreamBatchSampler` 隔离；
- burn-in/supervised mask 如何进入 detector；
- runtime state 如何初始化、detach、commit、rollback；
- train engine optimizer boundary 如何定义；
- loss weighting/denominator API；
- paired trace schema；
- full-stream gold audit runner；
- revised profile artifact schema；
- config variants；
- launch ticket 新增哪些 identity fields；
- result gate 新增哪些 fail-closed checks；
- 应新增/修改的 tests；
- 哪些旧代码应保留为 gold reference，哪些不应重构。

每项标记：

```text
P0 before any profile
P1 before scientific run
P2 only if positive
defer
```

给出最小补丁范围，不要建议与当前目标无关的大规模重构。

# 17. 最强拒稿与路线淘汰

在提出推荐方案前，先写不少于 500 中文字：

```text
The Strongest Case for Killing CRS-EPS-Enabled Full PETAL
```

必须包含：

- Full PETAL 的多论文重构攻击；
- CRS-EPS 可能只是普通 importance/event sampling；
- sampled state 与真实 full-stream state 不一致；
- bounded burn-in 对长动作不合法或不足；
- feature cache 掩盖 raw-video 成本；
- optimizer-event 成本分母错误；
- fixed/rematch 的增益可能只是 supervision stabilization；
- THUMOS14 小规模和 211/213 protocol 风险；
- custom online metric 风险；
- 为什么即使 Q2 positive 也可能不足以发表。

然后给出明确 kill rules，例如但不限于：

- sampled/full loss 或 gradient 明显不一致；
- full-stream validation quality 超出预注册不劣性边界；
- fixed binding 不优于 rematch；
- 增益不出现在 identity-linked errors；
- cost reduction不足以抵消 protocol complexity；
- raw-video LoRA 没有额外收益；
- strongest prior/reconstruction 已覆盖 surviving delta。

# 18. 需要作者回答的问题

先从仓库寻找答案，只询问真正影响裁决的 unknown。按以下等级分组：

```text
blocking before protocol implementation
blocking before profile
blocking before scientific training
blocking before interpreting results
blocking before paper claim
non-blocking
```

每个问题给出：

```text
why it matters
conservative default
what changes under alternative answers
can work continue before answer
```

至少审查但不要机械重复以下问题：

- 作者要验证的主 claim 是固定 binding、persistent state，还是 raw-video adaptation？
- 是否接受 Q2 fail 后立即停止 Full PETAL？
- full chronological route 是否只作 gold audit，还是必须参与训练？
- 允许的总 GPU-hour 预算如何分配给 profile/gold/Q2/Stage 2？
- canonical primary quality metric 与 cost metric 是什么？
- 第二数据集是否可用？
- 211/213 差异视频和理由是什么？
- feature extractor exact provenance 是否可恢复？
- raw-video decoder/encoder cadence 和 serving packet cadence 是什么？
- profile 是否允许改为多分母而非 200 optimizer events？

不要一次抛出无优先级的几十个问题。最多保留 12 个真正 blocking questions。

# 19. 两轮互动规则

## Round 1：代码、定义和未知项

第一次回复必须完成：

1. repository、branch、anchor、HEAD 和 diff verification；
2. evidence table；
3. 当前真实训练数据流和成本流；
4. 当前 route 是否实现 CRS-EPS；
5. `P0-LAUNCH-WORKDIR` 复核；
6. task/scope/end-to-end boundary；
7. Q2 one-factor preliminary audit；
8. profile denominator preliminary audit；
9. blocking unknowns；
10. provisional verdict。

如果存在会改变协议的 blocking unknown，停下来等待作者回答。不得直接输出最终训练
配方。仓库已经能回答的问题不得反问作者。

## Round 2：作者回答后给最终裁决

第二轮必须完成：

1. fresh literature review；
2. three-option protocol comparison；
3. CRS-EPS mathematical protocol；
4. active-at-entry and burn-in resolution；
5. exact one-factor contract；
6. revised profile contract；
7. minimal experiment loop；
8. raw-video Stage 2 gate；
9. claim map；
10. strongest rejection；
11. file-level implementation plan；
12. final decision card。

若公开证据已足够判定致命问题，可在 Round 1 直接 kill，但仍需提供完整证据链。

# 20. 最终输出格式

最终回复必须严格包含：

```text
A. Executive Verdict
B. Repository and Commit Verification
C. Evidence Table
D. Unknown Register
E. Fixed Task and Scope Audit
F. Current Code Dataflow and Cost Accounting
G. Is CRS-EPS Actually Implemented?
H. P0-P3 Code and Protocol Findings
I. Q2 One-Factor Identifiability Audit
J. CRS-EPS Mathematical Protocol
K. Active-at-Entry, Burn-in, and State Reconstruction
L. Full Chronological vs CRS-EPS vs Hybrid Decision
M. Revised Compute/Profile Contract
N. Feature Provenance and Real Online Clock
O. Fresh Literature and Competition Matrix
P. Claim Map
Q. Minimal Baseline/Ablation Matrix
R. Metrics and Statistical Plan
S. Test-to-Claim and Adversarial Cases
T. Raw-Video LoRA Stage-2 Gate
U. The Strongest Case for Killing CRS-EPS-Enabled Full PETAL
V. File-Level Implementation Plan
W. Kill / Continue Rules
X. Questions for the Authors
Y. Final Decision Card
```

Final Decision Card 必须使用：

```text
Repository HEAD observed:
Immutable commit reviewed:
Research verdict:
Current training route:
CRS-EPS currently implemented: YES / PARTIAL / NO
Task protocol verdict:
Q2 identifiability verdict:
Feature provenance verdict:
Online clock verdict:
Recommended protocol: FULL-CHRONOLOGICAL / CRS-EPS / HYBRID / KILL
Profile contract verdict: KEEP / REVISE / REPLACE
Full PETAL novelty verdict: SURVIVES / MARGINAL / RECONSTRUCTION / COLLAPSED
Raw-video Stage-2 permission: BLOCKED / CONDITIONAL / ALLOWED
Profile permission: BLOCKED / ALLOWED
Formal training permission: BLOCKED / CONDITIONAL / ALLOWED
Open P0:
Open P1:
Minimum fixes before next GPU-hour:
Exact next implementation step:
Exact kill condition:
Confidence:
```

最后再输出机器可读块：

```text
RESEARCH_VERDICT=
CURRENT_ROUTE=
CRS_EPS_IMPLEMENTED=
RECOMMENDED_PROTOCOL=
Q2_IDENTIFIABILITY=
PROFILE_CONTRACT=
FULL_PETAL_NOVELTY=
RAW_VIDEO_STAGE2=
PROFILE=
FORMAL_TRAINING=
NEXT_STEP=
OPEN_P0=
OPEN_P1=
```

# 21. 禁止事项

禁止：

- 不打开 GitHub 就接受作者摘要；
- 把 feature cache 当成 CRS-EPS；
- 把一个完整视频 optimizer event 与一个短 episode optimizer event 直接比较；
- 把 optimizer step 数下降写成总计算下降；
- 用 GT active state 初始化随机 episode；
- 忽略长动作 start 早于 burn-in；
- 忽略同类重复和重叠动作；
- 把 annotation-guided sampling 写成训练完全 future-free；
- 把 tests pass 写成 sampled objective 无偏；
- 把 cache route 写成 raw-video end-to-end；
- 把 current source-time latency 写成 wall-clock latency；
- 把 CRS-EPS 包装成主创新而不查 ETAD、sequence sampling 和 truncated BPTT；
- 在 fixed/rematch 使用不同 episodes、weights、RNG 或 loss denominators；
- 用 custom metric 隐藏 standard mAP、recall、FN 或 late FP；
- 在看到结果后修改采样比例、subset 或阈值；
- 建议直接启动 raw-video multi-seed 正式训练；
- 因为代码已经很多而降低 kill 标准；
- 通过改变任务解决 Full PETAL 的新颖性问题；
- 在 blocking unknown 未解决时输出过度自信的最终方案。

你的首要职责是冻结一个可证伪、可归因、成本可比较的训练协议，并帮助作者尽早
停止无效路线，而不是继续生产昂贵但不可解释的实验。

## END PROMPT
