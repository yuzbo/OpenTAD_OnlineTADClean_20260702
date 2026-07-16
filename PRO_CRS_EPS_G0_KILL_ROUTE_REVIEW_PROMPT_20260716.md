# Pro 深度讨论 Prompt：CRS-EPS G0 KILL 后的机制诊断与训练路线裁决

以下内容可直接完整提交给 GPT-5 Pro / GPT-5.5 Pro。请启用联网检索、
GitHub 浏览、代码阅读和长上下文推理。不要删除不可变提交、证据标签、
任务边界、两轮讨论规则、禁止事项或输出合同。

---

## BEGIN PROMPT

你是一名同时具备以下角色的最高强度研究审查者：

- CVPR / ICCV / ECCV Senior PC；
- CCF-A 期刊高级编辑；
- fully supervised Online Temporal Action Detection / Localization 专家；
- causal streaming model、persistent state、truncated BPTT、recurrent credit
  assignment 与 stochastic objective estimation 专家；
- importance sampling、Horvitz-Thompson / Hansen-Hurwitz、IPW、clustered
  sampling 与统计可识别性专家；
- PyTorch、OpenTAD、stateful training、checkpointing、AMP、Slurm 和实验
  证据链专家；
- 会主动寻找 future leakage、GT taint、状态重建偏差、梯度截断偏差、
  denominator 错误、post-hoc threshold tuning 和论文 claim inflation 的严厉审稿人。

请使用中文回答，保留公式、代码标识符、论文标题和 URL 的原文。

你的任务不是替作者辩护，也不是看到一个负结果后立刻发散新任务。你必须先基于
公开 GitHub 的不可变代码提交，重建当前实现和 G0 的比较对象，判断这次 KILL
究竟否定了什么，再裁决下一条仍然严格属于 On-TAD 的低成本训练路线。

你必须允许最终结论为：

```text
STRUCTURAL-KILL-CRS-EPS
IMPLEMENTATION-FIX-CANDIDATE
INSUFFICIENT-EVIDENCE-NEW-CONFIRMATION-REQUIRED
KEEP-FULL-CHRONOLOGICAL-AS-PRIMARY
PIVOT-TO-STATE-FAITHFUL-COST-CONTROL
KILL-FULL-PETAL-RESEARCH-ROUTE
```

任何 `FIX` 或 `PIVOT` 只表示值得设计并进行新的可证伪验证，不表示方法有效、
创新成立或允许正式训练。

# 0. 公开仓库与不可变审查锚点

首先打开并核验：

```text
Repository:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702

Branch:
codex/full-petal-implementation

Branch URL:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/tree/codex/full-petal-implementation

Immutable scientific implementation and G0 anchor:
70df86ea3d38d70c658ae0ee9e04245d57b834d4

Commit URL:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/commit/70df86ea3d38d70c658ae0ee9e04245d57b834d4

Pinned tree URL:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/tree/70df86ea3d38d70c658ae0ee9e04245d57b834d4
```

本 Prompt 和 G0 failure dossier 会在上述代码提交之后以 documentation-only
commit 发布。因此 branch HEAD 可能晚于 `70df86e`。你必须：

1. 报告你实际看到的 branch HEAD；
2. 检查 `70df86e..HEAD` 的 diff；
3. 若差异仅为本 Prompt、wiki 和 failure dossier，科学代码仍固定在 `70df86e`；
4. 若模型、数据、配置、训练、审计、测试或结果代码发生变化，逐项报告，不能默认
   新 HEAD 继承旧 G0 证据；
5. 不得从默认分支推断此分支的实现；
6. 不得把未查看的文件写成已经核验。

## 0.1 证据标签

每一项重要判断必须使用以下标签之一：

```text
[CODE-VERIFIED]               由 pinned GitHub 代码直接支持
[DOC-VERIFIED]                由 pinned 项目文档直接支持，但不是运行结果
[SIGNED-ARTIFACT-REPORTED]    由外部签名 artifact 及其哈希报告支持，原始 bundle 不在 GitHub
[REVIEW-REPRODUCED]           由锁定独立审查员在精确提交上复现
[PRIMARY-VERIFIED]            由论文原文或官方代码直接支持
[INFERENCE]                   从多项证据推导，但没有直接观测
[UNKNOWN]                     当前材料无法回答
[REQUIRES-EXPERIMENT]         只能通过新的预注册实验回答
```

硬性约束：

- tests pass 不等于科学 claim pass；
- 签名有效不等于训练代理有效；
- cached-feature training 不等于 raw-video end-to-end training；
- G0 在一个随机初始化 checkpoint 上 KILL，不自动证明所有训练阶段 checkpoint
  都 KILL；但它确实终止了当前冻结协议；
- 找不到直接同名工作不能写成“首次”；
- 平均优于弱 baseline 不能覆盖绝对 fidelity margin 失败；
- 作者汇总的 artifact 数值不得标成 `[CODE-VERIFIED]`。

# 1. 不可修改的任务边界

本讨论必须留在标准 fully supervised completion-triggered On-TAD / On-TAL：

```text
Video stream: X = {x_t}_{t=1}^T
Filtration: F_t = sigma(x_1, ..., x_t, past model state, past visible outputs)
GT action instance: psi_i = (s_i, e_i, c_i)
Visible completion emission: d_j = (s_hat_j, e_hat_j, c_hat_j, score_j, tau_emit_j)
Causality: e_hat_j <= tau_emit_j and no future frame is available at tau_emit_j
Evaluation: full chronological stream, not sampled episodes
```

当前代码允许内部 persistent slots 随前缀更新，但用户可见的 completion emission
遵循冻结的在线协议。不得通过以下方式逃离问题：

- 改成 Online Action Detection frame classification、Online TAS、VideoQA、
  anomaly detection 或通用 semantic memory；
- 改成 zero-shot / open-vocabulary 主任务；
- 允许未来帧、EOF、完整视频长度、GT 初始化或 offline NMS 回写；
- 用新标签、新传感器或新 benchmark 代替方法创新；
- 把训练时 annotation-guided episode selection 描述成推理时因果性；
- 把 cached features 描述为端到端视觉编码器微调。

# 2. 当前研究对象必须分层

## 2.1 Full PETAL 长期工程研究对象

```text
causal visual stream or cached causal features
+ persistent action-instance slots
+ birth / active / end lifecycle
+ identity-consistent prefix supervision
+ low-cost causal training
+ immutable standard On-TAD completion emission
```

此前审查已经认为该 package 很容易被 causal backbone、direct On-TAD、
persistent query / TrackFormer 类方法重构。因此不能把“完整实现了 Full PETAL”本身
当作充分创新。

## 2.2 当前 Q2 科学对象

在相同 cached features、persistent detector、lifecycle、loss、optimizer、evaluator
和 streaming inference 下比较：

```text
fixed_birth_slot
vs.
prefix_rematch_active_pool
```

Q2 的 complete chronological cached route 是 gold/reference 训练路线，不是
CRS-EPS，也不是 raw-video end-to-end。

## 2.3 被 G0 检验的 CRS-EPS

CRS-EPS 表示：

```text
Instance-aware Causal Risk-Set Event-Centric Prefix-Episode Training

event/start/end/ongoing/hard-background/uniform episode proposal
+ causal replay context
+ short supervised suffix
+ HH/IPW weighting
+ isolated episode lifecycle
+ one optimizer event per exact M-draw group
+ full chronological validation and evaluation
```

当前 G0 的语义四臂是：

```text
video_start_full gold control
dynamic_birth candidate
fixed_192 candidate
reset candidate
```

存储中每个样本有三个 candidate trace row，每个 row 的 left side 是同一
`video_start_full` gold control，因此四个样本共有十二个 trace rows。

# 3. 必须逐行阅读的 GitHub 文件

所有路径都必须从 pinned tree `70df86e` 读取。不得只读 README 或本 Prompt。

## 3.1 任务、历史与冻结协议

```text
RTK.md
FULL_PETAL_EXECUTION_GATES.md
FULL_PETAL_TRUST_MODEL.md
PRO_FULL_PETAL_CRS_EPS_IMPLEMENTATION_REVIEW_20260716.md
research-wiki/query_pack.md
research-wiki/decision_register.md          # DR-029 through DR-038
research-wiki/discussion_timeline.md         # T22 through T31
research-wiki/ideas/crs-eps-training.md
research-wiki/ideas/petal-ontad.md
research-wiki/claims/c4-crs-eps-cost.md
research-wiki/experiments/formal-training-none.md
```

最新 documentation-only branch HEAD 还应读取：

```text
research-wiki/index.md
research-wiki/query_pack.md
research-wiki/experiments/crs-eps-g0-kill-20260716.md
research-wiki/experiments/formal-training-none.md
research-wiki/ideas/crs-eps-training.md
research-wiki/claims/c4-crs-eps-cost.md
research-wiki/decision_register.md           # DR-039
research-wiki/discussion_timeline.md          # T32
```

## 3.2 配置、数据与采样数学

```text
configs/causaltad/thumos_pes_q2_base.py
configs/causaltad/thumos_pes_q2_crs_eps_base.py
configs/causaltad/thumos_pes_q2_crs_eps_fixed.py
configs/causaltad/thumos_pes_q2_crs_eps_rematch.py
opentad/datasets/crs_eps_feature.py
opentad/datasets/streaming_feature.py
opentad/datasets/builder.py
opentad/utils/crs_eps_sampling.py
tools/build_crs_eps_episode_manifest.py
```

## 3.3 模型、监督、状态和 optimizer lifecycle

```text
opentad/models/detectors/persistent_trajectory_ontad.py
opentad/models/dense_heads/persistent_event_set_head.py
opentad/utils/prefix_trajectory_supervision.py
opentad/cores/train_engine.py
opentad/cores/optimizer.py
opentad/cores/scheduler.py
opentad/utils/stream_control.py
opentad/utils/stream_packets.py
```

## 3.4 G0 比较、签名和启动门禁

```text
opentad/utils/crs_eps_audit.py
opentad/utils/crs_eps_gold_gate.py
opentad/utils/crs_eps_gold_evidence.py
opentad/utils/full_petal_attestation.py
opentad/utils/full_petal_role_signing.py
opentad/utils/evidence_bundle.py
opentad/utils/full_petal_launch.py
tools/preregister_crs_eps_gold_audit.py
tools/run_crs_eps_gold_audit.py
tools/run_full_petal_b0.py
tools/run_full_petal_posix_b0.py
tools/check_full_petal_results.py
```

## 3.5 强制测试

```text
tests/test_crs_eps_sampling.py
tests/test_crs_eps_training_contracts.py
tests/test_crs_eps_paired_audit.py
tests/test_crs_eps_gold_gate.py
tests/test_crs_eps_g0_evidence.py
tests/test_full_petal_training_transaction.py
tests/test_full_petal_launch_gate.py
tests/test_full_petal_result_gate.py
tests/test_full_petal_identity.py
```

# 4. 当前可用证据，不得照抄结论

以下是作者提供的外部 evidence-chain 摘要。原始 checkpoint、features、manifest
和 `audit.json` 按仓库规则保存在 repo 外，因此必须标为
`[SIGNED-ARTIFACT-REPORTED]`，除非本次会话另行上传原始 bundle。

## 4.1 预结果链

```text
Implementation commit:
70df86ea3d38d70c658ae0ee9e04245d57b834d4

Local B0 and target-Linux B0:
588/588 tests passed

Same locked reviewer:
OVERALL_VERDICT=PASS
B0=PASS
SIX_PRIOR_BLOCKERS=CLOSED
SENTINEL_AWARE_METRIC=PASS
CHECKPOINT_PREREGISTRATION=PASS
PROFILE=ALLOW
FORMAL=BLOCK
NEXT_GATE=G0

Signed review artifact SHA-256:
be4b82aa9232a10d9da62239afaf9878cd8b4c53b8ad35784c75ac0a72b9a06b

Selection artifact SHA-256:
457838e8c3cb6aa05b5d80a9f9240d872a8a5e8d2b38058d4a37bf1b797fea4f

Margins artifact SHA-256:
5160adab25fb6538990e91c611f094eb59f3b01b1f268158e7e81fb913004329

Checkpoint SHA-256:
68169922bc34b77ee9ac210b4bc90137276972e9b492745b5927c69652ff8e5f

Checkpoint generation:
deterministic initialization, seed=705, parameter_count=1,126,424,
explicitly not evidence of model quality

Manifest identity:
96278dcef7835ceb31e96e499a20b0806f2d882167cd1fbb039534ee9a95b0e5
```

## 4.2 冻结 margins

```text
min_gradient_cosine = 0.90
min_gradient_sign_agreement = 0.90
max_relative_loss_error = 0.10
min_runtime_continuous_cosine = 0.90
require_runtime_discrete_equal = true
max_mean_dynamic_replay_ratio = 0.80
max_video_start_fallback_fraction = 0.25
min_dynamic_minus_fixed_gradient_cosine = 0.02
min_dynamic_minus_reset_gradient_cosine = 0.05
```

这些 margins、样本、checkpoint bytes 和 method 在 terminal outcome 前冻结。

## 4.3 Terminal G0

```text
Signed audit.json SHA-256:
a08183a8dba4ec5267b5a75eed3501882180a22f28d6b76ce2c7b4e65511009b

Attestation:
algorithm=ed25519
role=crs-eps-g0-audit
key_id=full-petal-crs-eps-g0-20260716

status=KILL
sample_count=4
candidate_trace_count=12
violation_count=14
failed_samples=3
```

Aggregate metrics：

| Metric | Observed | Frozen requirement | Status |
|---|---:|---:|---|
| max dynamic relative loss error | `0.8021221151` | `<=0.10` | FAIL |
| min dynamic gradient cosine | `0.2830554455` | `>=0.90` | FAIL |
| min dynamic gradient sign agreement | `0.5445116162` | `>=0.90` | FAIL |
| min dynamic runtime cosine | `0.1027812195` | `>=0.90` | FAIL |
| runtime discrete equality | failed in 3 cases | exact | FAIL |
| mean dynamic replay ratio | `0.7586404310` | `<=0.80` | PASS |
| video-start fallback fraction | `0.25` | `<=0.25` | PASS at boundary |
| mean dynamic minus fixed grad cosine | `0.1141427186` | `>=0.02` | PASS |
| mean dynamic minus reset grad cosine | `0.5005255434` | `>=0.05` | PASS |

Dynamic candidate 的逐样本摘要：

| Sample | grad cosine | sign agree | relative loss error | runtime cosine | discrete equal | replay ratio |
|---|---:|---:|---:|---:|---|---:|
| `video_validation_0000151:draw=0` | `1.0000` | `1.0000` | `0.0000` | `1.0000` | yes | `1.0000`, video-start fallback |
| `video_validation_0000946:draw=1` | `0.5130` | `0.6473` | `0.3885` | `0.1214` | no | `0.7704` |
| `video_validation_0000051:draw=1` | `0.2831` | `0.5445` | `0.8021` | `0.1028` | no | `0.7752` |
| `video_validation_0000163:draw=0` | `0.5041` | `0.6313` | `0.4197` | `0.9852` | no | `0.4890` |

额外观测：

- `0000051` 和 `0000163` 中，`dynamic_birth` 与 `fixed_192` 的上述 fidelity
  metrics 完全相同；
- `0000946` 中 dynamic gradient cosine `0.5130`，fixed 仅 `0.0564`，但 dynamic
  仍远低于绝对 margins；
- `reset` 在所有非 fallback 样本更差；
- terminal audit 已在目标 Linux 独立验签、重算 gate 并得到相同 `KILL`；
- launch validator 对此 artifact 返回
  `CRS-EPS G0 audit has not reached PASS`；
- 没有启动 fixed-step profile 或 formal training。

# 5. Round 1：先审计和诊断，不得直接给补丁

Round 1 不允许写实现代码，不允许提出“先跑大实验再看”，不允许改变任务。

## 5.1 仓库与执行链重建

从代码逐步重建：

```text
annotations/features
-> epoch manifest and sampling probabilities
-> exact M-draw group
-> singleton G0 controls
-> runtime-state initialization/replay
-> supervised suffix
-> loss numerator and denominator
-> backward graph
-> paired loss/gradient/state comparison
-> frozen gate
-> signed terminal artifact
-> profile launch rejection
```

逐项回答：

1. `dynamic_birth` 的 replay 起点如何确定？
2. replay prefix 哪些 token 参与 numerical state，哪些参与 autograd graph？
3. `fixed_192` 与 `dynamic_birth` 何时退化为相同 replay range？
4. `video_start_full` gold control 的 gradient path 到底覆盖哪些 token？
5. 三个 candidate 是否使用完全相同的 RNG、model bytes、supervised suffix、
   weights 和 denominator？
6. 比较的是相同标量 objective 的近似，还是不同 state-conditioned objective？
7. HH/IPW 修正发生在哪一层，不能修正哪些 state/gradient path？
8. discrete runtime mismatch 的具体字段是什么，是否由历史 lifecycle 决定？
9. slot exhaustion、birth assignment 和 endpoint ownership 是否影响这些样本？
10. gate 是否存在把实现差异误写成方法失败的明显代码错误？

必须给出 `file:line` 证据。

## 5.2 数学 estimand 与偏差分解

不要只说“truncated BPTT 有偏”。请明确写出：

```text
h_t(theta) = F_theta(h_{t-1}, x_t)
l_t(theta, h_t, y_t)
L_full(theta) = sum_t alpha_t l_t(theta, h_t(theta), y_t)
g_full(theta) = grad_theta L_full(theta)
```

再定义 CRS-EPS 对 anchor/episode `a` 的 proposal `q(a)`、inclusion probability
`pi_t`、weight `w_t`、replay state `h_hat_a`、suffix objective 和 sampled gradient。

至少分解以下误差：

```text
sampling-measure error
weight / denominator error
forward-state reconstruction error
omitted historical Jacobian error
lifecycle / assignment path divergence
stochasticity mismatch
finite-sample variance
```

核心问题：即使 HH/IPW 对 decision-bin sampling measure 无偏，当
`h_hat_a != h_a(theta)` 或历史 Jacobian 被 detach 时，sampled gradient 是否仍可能
对 `g_full(theta)` 有系统偏差？请给出条件，而不是口号。

## 5.3 根因假设矩阵

至少审查以下假设，但不能预设其中任何一个成立：

| ID | Hypothesis |
|---|---|
| H1 | 当前失败主要来自 omitted historical Jacobian，是结构性 truncated-gradient bias |
| H2 | forward numerical state 也没有被忠实重建 |
| H3 | HH/IPW target、union inclusion probability 或 denominator 不匹配 gold objective |
| H4 | lifecycle、slot birth、endpoint ownership 或 discrete state 在 replay 起点发生分叉 |
| H5 | `dynamic_birth` 实际经常退化为 `fixed_192`，没有实现预期动态上下文 |
| H6 | G0 paired-audit 或 metric 实现仍有局部错误 |
| H7 | deterministic random-init checkpoint 对训练中 fidelity 的代表性不足 |
| H8 | 四样本足以 kill 精确冻结协议，但不足以推广到整个方法族 |

每个假设必须输出：

```text
code evidence
observed trace signature
contradicting evidence
current identifiability
smallest zero-GPU diagnostic
whether a new confirmatory experiment is required
```

## 5.4 精确解释 KILL 的外延

必须分别裁决：

1. 当前 `70df86e + seed705 + frozen G0` 协议是否已被有效 KILL；
2. 当前 CRS-EPS `dynamic_birth` 是否还能作为 primary training surrogate；
3. event-centric prefix sampling 这个更大方法族是否被否定；
4. Q2 full chronological cached reference 是否仍有效；
5. Full PETAL 的 persistent lifecycle/model infrastructure 是否被否定；
6. raw-video end-to-end On-TAD 的长期目标是否被否定。

不得用一个总括性的“方法失败”替代这六项。

## 5.5 随机初始化 checkpoint 审查

这是必须单独回答的问题：

- outcome-blind deterministic initialization 解决了选择自由，但是否解决代表性？
- fidelity gate 应要求对随机初始化、早期训练、后期训练都成立吗？
- 如果需要 trajectory checkpoints，如何在没有先接受有偏训练的情况下获得它们？
- 能否使用 gold chronological route 产生独立 checkpoints，再测试 surrogate？
- 多 checkpoint 验证会不会把一个低成本 gate 变成不可接受的训练成本？
- 这只能影响未来新协议，还是足以推翻当前已冻结 G0 的 terminal decision？

## 5.6 最新竞争工作与方法查新

必须联网检索截至你执行审查当日的 primary sources 和 official repositories。
不要仅复述本仓库 wiki。至少覆盖：

1. direct Online TAL / On-TAD；
2. causal streaming video backbones；
3. persistent query / track-style temporal instance state；
4. efficient end-to-end TAL/TAD，例如 selective gradient、proposal sampling、
   sparse backward 或 feature reuse；
5. recurrent/state-space model 的 truncated-gradient bias、RTRL/UORO/ARTBP、
   activation checkpointing、state checkpointing 或其他 long-sequence credit
   assignment；
6. sampled objectives 中 state-dependent data selection 和 off-policy correction；
7. 与本项目最接近的 full-forward/sparse-backward 训练方法。

只使用论文原文、官方会议页面、作者代码仓库或官方文档。输出检索式、检索日期、
论文/代码 URL、与本项目的精确重叠和剩余 gap。不能因名称不同就判断不重叠。

## 5.7 最强拒稿论证

以最严厉 Senior PC 身份给出当前路线的 strongest rejection。至少考虑：

- persistent slots + causal model + lifecycle 是否只是已知组件组合；
- 训练成本代理已经被自己的 fidelity gate KILL；
- cached-feature Stage 1 无法支撑端到端视觉适配 claim；
- 当前没有 effectiveness、latency-quality frontier 或 formal benchmark 结果；
- 四个 G0 样本是否太少，或随机初始化是否过于局部；
- 即便修复 surrogate，论文 novelty 是否仍不足。

然后说明哪些证据能够真正回答该拒稿，而不是建议更强叙事。

# 6. Round 1 输出合同

严格按以下顺序输出：

## A. Executive Verdict

```text
CURRENT_FROZEN_G0 = VALID_KILL | INVALID_EVIDENCE | UNKNOWN
CURRENT_CRS_EPS_PRIMARY = STRUCTURAL_KILL | FIX_CANDIDATE | INSUFFICIENT
FULL_CHRONOLOGICAL_Q2 = KEEP_AS_GOLD | KEEP_AS_PRIMARY | REJECT
FULL_PETAL_INFRASTRUCTURE = KEEP | REVISE | KILL
PROFILE = BLOCK
FORMAL = BLOCK
NEW_GPU_HOURS_BEFORE_ROUND2 = 0
CONFIDENCE = [0,1]
```

## B. Repository and Anchor Verification

报告实际 HEAD、pinned commit、diff 范围、必读文件覆盖和无法访问项。

## C. Current Task and Executable Method

用一段话准确重述任务，再给真实执行链，不得使用论文式模糊命名。

## D. Evidence Integrity Table

逐项列 B0、review、selection、margins、checkpoint、manifest、audit、gate 和 launch
blocker 的证据标签、哈希、可复现范围和限制。

## E. G0 Metric Semantics Audit

说明每个 loss、gradient、continuous state、discrete state 和 replay ratio 比较的对象。

## F. Per-Sample Failure Analysis

四个样本逐一分析，特别解释两个 `dynamic_birth == fixed_192` case。

## G. Mathematical Bias Decomposition

给公式、成立条件、偏差项和哪些条件被代码满足。

## H. Root-Cause Matrix H1-H8

不得跳项。

## I. Scope of Falsification

分别回答 5.4 的六个层级。

## J. Random-Checkpoint Validity

给出对当前结论和未来协议的不同影响。

## K. Competition and Novelty Audit

包含 primary citations、overlap matrix 和 novelty 风险。

## L. Strongest Rejection

先拒稿，再列真正可消除该拒稿的证据。

## M. Unknown Register and Author Questions

只询问仓库、artifact 和 primary sources 无法回答且会改变路线裁决的问题。每个问题
说明：为什么重要、不同答案会怎样改变 decision。不要询问能通过读代码回答的问题。

Round 1 到此停止。不得在作者回复前输出逐文件 patch 或启动命令。

# 7. Round 2：作者回复后进行路线裁决

作者回答 Round 1 问题后，先逐项记录：

```text
ANSWERED
PARTIALLY_ANSWERED
UNANSWERED
CONTRADICTED_BY_CODE
```

然后比较至少以下路线。它们是候选，不是预设答案；你可以增加仍严格属于 On-TAD
的路线。

| Route | Definition | Principal risk |
|---|---|---|
| R0 | Kill CRS-EPS and keep full chronological cached Q2 | cost may remain high; novelty weak |
| R1 | Full causal forward state plus selective/sparse backward | forward state may be faithful while gradient remains approximate |
| R2 | State checkpoints plus bounded suffix replay | saved state may be parameter-stale or stop historical Jacobians |
| R3 | Exact chronological chunks with activation checkpoint/recompute | scientifically clean but potentially expensive |
| R4 | Gold-route teacher state/gradient distillation into episode training | teacher cost and objective leakage may dominate |
| R5 | Abandon Full PETAL route while retaining reusable infrastructure | loses current paper route but may avoid sunk-cost escalation |

## 7.1 路线评分

每条路线按 `1-5` 分并解释：

```text
task fidelity
causal correctness
gradient/state faithfulness
training-cost reduction
implementation risk
novelty after competition audit
ability to support raw-video adaptation later
falsifiability at <= one bounded profile
paper-level value
```

## 7.2 新方法的 claim map

任何推荐路线必须先形成 claim map：

| Claim ID | Exact claim | Baseline | Metric | Required evidence | Kill condition |
|---|---|---|---|---|---|

必须区分：

```text
engineering correctness
training-surrogate fidelity
resource efficiency
On-TAD effectiveness
latency-quality performance
raw-video end-to-end adaptation
novelty
```

## 7.3 开发集与确认集隔离

当前四个 G0 samples 已经暴露，今后只能作为 development diagnostics。新的
confirmatory G0 必须：

- 使用未被本轮分析查看 outcome 的 holdout samples；
- 在执行前冻结 selection rule、sample IDs、checkpoint policy、margins、config、
  code commit 和 data identity；
- 不把旧样本上的修复效果报告成 confirmatory evidence；
- 若多 checkpoints/seeds 是必要条件，预先定义聚合规则和 family-wise kill rule；
- 保留 `KILL` 作为合法终态；
- 新 route 从 B0 和独立 review 重新开始。

## 7.4 最小实验闭环

推荐路线必须给出：

1. 零 GPU synthetic invariants；
2. development diagnosis；
3. unseen confirmatory G0；
4. fixed-step resource profile；
5. 小规模 effectiveness pilot；
6. full chronological evaluation；
7. formal multi-seed experiment；
8. raw-video adaptation 进入条件。

每一步写明输入、输出、统计单位、预算、go/kill rule 和下一步授权。

## 7.5 成本分母

必须同时报告：

```text
videos
temporal tokens read
temporal tokens with gradient
visual frames forwarded
visual frames backpropagated
state-replay tokens
optimizer events
peak allocated/reserved memory
wall time
GPU-hours
effective sample size
full chronological evaluation cost
```

不得用 optimizer-step 数量单独代表训练成本。

# 8. Round 2 输出合同

## N. Author-Answer Reconciliation

逐项判定回答状态及其对路线的影响。

## O. Route Portfolio

给 R0-R5 和新增路线的评分表。

## P. Recommended Route and Why

只能推荐一条主路线，另给一个 fallback。说明为什么不是小修小补。

## Q. Exact Task and Method Design

包括状态、数据流、forward/backward 边界、loss、sampling、causality 和 inference。

## R. Claim Map

完整表格，不得把工程能力写成创新 claim。

## S. Code-Level Delta

按文件列：保留、删除、修改、新增。先写设计，不直接生成补丁。

## T. Baselines and Ablations

至少包含 full chronological gold、fixed context、reset、当前 killed CRS-EPS、
最接近 published method，以及隔离每个新机制的 ablation。

## U. Metrics and Statistics

包括 fidelity、effectiveness、latency、identity、cost、seed、confidence interval
和 multiple-comparison rule。

## V. Preregistered Experiment Ladder

从 zero-GPU 到 formal，每一级给预算、artifact、GO/KILL 条件。

## W. Reviewer-Risk Register

列出 novelty、task drift、end-to-end overclaim、biased training、weak baseline、
insufficient scale 和 cost-accounting 风险。

## X. Ordered Next Actions

给严格顺序的 P0/P1/P2，不允许把 profile 或 formal 放在新 G0 PASS 前。

## Y. Machine-Readable Final Verdict

最后必须原样输出并填值：

```text
CURRENT_FROZEN_G0=VALID_KILL|INVALID_EVIDENCE|UNKNOWN
CURRENT_CRS_EPS_PRIMARY=STRUCTURAL_KILL|FIX_CANDIDATE|INSUFFICIENT
RECOMMENDED_ROUTE=<one route id and name>
FALLBACK_ROUTE=<one route id and name>
CURRENT_EXPOSED_G0_SAMPLES=DEVELOPMENT_ONLY
NEW_HOLDOUT_PREREGISTRATION=REQUIRED
B0_RESTART=REQUIRED
INDEPENDENT_REVIEW_RESTART=REQUIRED
PROFILE=BLOCK
FORMAL=BLOCK
RAW_VIDEO_STAGE=BLOCK
NEXT_GATE=<exact next gate>
MAX_GPU_HOURS_BEFORE_NEXT_GATE=<number>
CONFIDENCE=<0..1>
```

# 9. 永久禁止事项

- 不得修改或弱化旧 G0 margins 来挽救当前路线；
- 不得换 checkpoint 后声称仍是同一 confirmatory test；
- 不得把已暴露四样本再次当作 unseen confirmation；
- 不得因 dynamic 平均优于 reset/fixed 就忽略绝对 fidelity 失败；
- 不得因随机初始化代表性有限就把签名 KILL 改写成 PASS；
- 不得在新 G0 PASS 前允许 profile；
- 不得在 profile 和 effectiveness gates 前允许 formal training；
- 不得把 cached-feature route 写成 raw-video end-to-end；
- 不得把 Full PETAL infrastructure completion 写成论文创新成立；
- 不得通过改变 On-TAD 任务定义寻找更容易的论文故事；
- 不得给没有 primary source 或代码证据的“首个”“无人区”结论。

现在开始 Round 1。先核验仓库和不可变提交，再进行代码、数学、证据和竞争工作审查。

## END PROMPT
