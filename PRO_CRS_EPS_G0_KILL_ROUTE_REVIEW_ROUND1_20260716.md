# A. Executive Verdict

```text
CURRENT_FROZEN_G0 = VALID_KILL
CURRENT_CRS_EPS_PRIMARY = STRUCTURAL_KILL
FULL_CHRONOLOGICAL_Q2 = KEEP_AS_PRIMARY
FULL_PETAL_INFRASTRUCTURE = KEEP
PROFILE = BLOCK
FORMAL = BLOCK
NEW_GPU_HOURS_BEFORE_ROUND2 = 0
CONFIDENCE = 0.92
```

**核心裁决：**

* `[SIGNED-ARTIFACT-REPORTED]` 当前 `70df86e + seed=705 + frozen samples + frozen margins + frozen checkpoint` 的 G0 已产生合法 terminal `KILL`。四个样本中三个违反绝对 fidelity margins，且失败同时覆盖 loss、完整参数梯度、连续 runtime state 和离散 lifecycle state，不能由“dynamic 平均优于 fixed/reset”抵消。
* `[CODE-VERIFIED][INFERENCE]` 当前 `dynamic_birth` 不是一个只差局部常数或 denominator 的近似。它从空 runtime state 开始，只重放固定上下文和 suffix-relevant GT birth history，无法一般性重建完整 chronological prefix 已形成的 persistent queries、slot lifecycle、refractory、peak label/score、start state 和 supervision ownership。因此当前 surrogate 的基本充分统计量不成立，裁决为 **`STRUCTURAL-KILL-CRS-EPS`**。
* `[CODE-VERIFIED]` `video_start_full` G0 control 并非全视频无截断 BPTT；它是从视频开头重建完整**数值状态**，但仍按全局 64-bin 边界 detach 的 chronological TBPTT control。候选与 gold 共用该截断规则，却仍发生大幅状态和梯度分叉，所以不能把此次失败主要归咎于两者共有的 TBPTT 截断。
* `[INFERENCE]` 下一条值得讨论的方向只能是 **`PIVOT-TO-STATE-FAITHFUL-COST-CONTROL`**，例如完整 causal forward state 加选择性 backward、精确 chronological recomputation 或受控 state checkpoints；这属于新方法、新提交和新 G0，不是对旧 CRS-EPS 的小修，也不代表该方向已经获准。
* `[CODE-VERIFIED][DOC-VERIFIED]` Q2 的完整 chronological cached-feature route 保持因果数值状态，仍是当前唯一干净的 primary/reference 训练路线。它只支持 cached-feature、chunk-detached Q2，不支持 raw-video end-to-end claim。
* `[CODE-VERIFIED]` persistent detector、lifecycle、prefix supervision、事务回滚、证据签名和 launch gate 基础设施没有被 G0 否定，应保留为工程底座，但不能作为论文创新成立的证据。

---

# B. Repository and Anchor Verification

## B.1 锚点

| 项目                       | 核验结果                                                       |
| ------------------------ | ---------------------------------------------------------- |
| Repository               | `https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702` |
| Branch                   | `codex/full-petal-implementation`                          |
| Pinned scientific commit | `70df86ea3d38d70c658ae0ee9e04245d57b834d4`                 |
| Actual branch HEAD       | `02f5f851defa1c7413d93f0f365895cd7a6509fd`                 |
| HEAD message             | `docs: record CRS-EPS G0 kill and Pro route review`        |
| Compare                  | `70df86e..02f5f85`，ahead 1，behind 0                        |
| Scientific-code anchor   | 仍为 `70df86e`                                               |

`02f5f851…` 的提交内容是本次 Pro prompt、G0 failure dossier 和 wiki 状态更新；GitHub compare 未显示模型、数据集、配置、训练引擎、审计器、测试或结果代码变化。

具体新增或修改的是：

```text
PRO_CRS_EPS_G0_KILL_ROUTE_REVIEW_PROMPT_20260716.md
research-wiki/experiments/crs-eps-g0-kill-20260716.md
research-wiki/index.md
research-wiki/query_pack.md
research-wiki/claims/c4-crs-eps-cost.md
research-wiki/decision_register.md
research-wiki/discussion_timeline.md
research-wiki/experiments/formal-training-none.md
research-wiki/ideas/crs-eps-training.md
research-wiki/graph/edges.jsonl
research-wiki/log.md
research-wiki/source_map.md
```

因此：

```text
SCIENTIFIC_CODE_SNAPSHOT = 70df86ea3d38d70c658ae0ee9e04245d57b834d4
FAILURE_DOSSIER_SNAPSHOT = 02f5f851defa1c7413d93f0f365895cd7a6509fd
```

## B.2 文件覆盖

我逐文件读取了 prompt 所列的：

* 49 个 pinned-tree 文档、配置、数据、模型、训练、审计、签名、门禁和测试文件；
* HEAD 上的 G0 failure dossier、DR-039、T32 和相关 status 文档；
* 当前分支 compare；
* primary papers、官方项目页和能够确认的作者代码仓库。

## B.3 当前不可访问项

以下原始对象没有进入公开 GitHub，也未在本会话上传：

```text
checkpoint bytes
feature arrays
persisted episode manifest bytes
selection.json
margins.json
signed audit.json
12 raw trace rows
target-Linux verification logs
signed B0 bundle
signed independent-review bundle
```

这与仓库的 RTK 规则一致：GitHub 只记录 hashes 和结论，不保存 checkpoint、features 或生成的 evidence archive。

因此：

* 数值结果不能标为 `[CODE-VERIFIED]`；
* 我可以审计代码是否会正确生成、签名、复算和阻断这些结果；
* 我不能在本会话重新验签 `a081…` 或逐字段重算真实十二条 trace；
* 原始 bundle 缺失降低的是本次审查的**独立复现强度**，不自动使已签 terminal outcome 无效。

---

# C. Current Task and Executable Method

## C.1 准确任务

当前任务是标准 fully supervised、completion-triggered On-TAD/On-TAL。模型在完整 chronological stream 上维护 persistent action-instance state，在动作结束后产生不可回写的 visible completion emission。Q2 只比较：

```text
fixed_birth_slot
vs.
prefix_rematch_active_pool
```

两臂共享 cached features、persistent detector、slot lifecycle、loss、optimizer、evaluator 和 causal inference。CRS-EPS 是一个候选低成本训练 surrogate，不是 Q2 本身，也不是 raw-video end-to-end 训练。基础配置明确规定 cached features、`chunk_size=64`、`memory_size=192`、4 slots、完整 chronological evaluation、无未来 endpoint、无 runtime GT 和无 raw-video joint training。

## C.2 真实执行链

```text
annotations + verified cached feature bytes
    ↓
VideoSamplingSpec / per-bin lifecycle diagnostics
    ↓
q(component), q(anchor|component), marginal q(anchor)
    ↓
rho_t = P(one draw supervises bin t)
raw HH/IPW weight = (1 / num_video_bins) / rho_t
    ↓
immutable epoch manifest with exact M draws per video
    ↓
for each draw:
    dynamic/fixed/reset replay range
    + short supervised suffix
    + global 64-bin gradient ranges
    ↓
independent empty runtime state and empty supervision state
    ↓
causal replay tokens update numerical recurrent/lifecycle state
    ↓
only gradient-range tokens enter autograd
    ↓
only suffix bins contribute weighted loss
    ↓
one backward contribution per draw
    ↓
exact M-draw group accumulation
    ↓
one optimizer event per video group
    ↓
full chronological validation/evaluation
```

采样代码对每个 bin 计算单 draw coverage `rho_t`，使用

[
w_t=\frac{1/N}{\rho_t},
]

而 `union_pi_by_bin` 只作为多 draw inclusion diagnostic，不用于当前 HH repeated-exposure estimator。`episode_geometry` 计算固定上下文、suffix-relevant births、left-censoring 和全局 64-bin gradient ranges。见：

```text
opentad/utils/crs_eps_sampling.py:300-318
opentad/utils/crs_eps_sampling.py:362-421
```

每个 episode 都切取自己的 replay feature slice，构造该 slice 上的 prefix schedule，并设置 `is_video_start=True`、`is_video_end=True`。也就是说候选 episode 在 runtime 语义上是一个从空状态启动的独立短视频，而不是从真实 chronological runtime state 续接。见：

```text
opentad/datasets/crs_eps_feature.py:176-263
opentad/datasets/crs_eps_feature.py:264-297
```

模型随后从 `_initial_runtime_state` 和新的 `PrefixTrajectorySupervisionState` 开始 replay。所有 replay token 都更新 numerical state 和 thresholded lifecycle；只有达到 `gradient_start` 后才开启 autograd，并在每个 global gradient boundary 再 detach。监督 loss 只在 `supervised_range` 内累积。见：

```text
opentad/models/detectors/persistent_trajectory_ontad.py:1016-1139
```

每个 draw 的 `_optimizer_weight` 和 `_optimizer_denominator` 都为 `1/M`；训练引擎对 M 次 backward 进行累积，group boundary 时 denominator 总和为 1，再执行一次 optimizer/scheduler/EMA/state commit。因此没有额外的第二次 `/M`。

## C.3 十项直接回答

### 1. `dynamic_birth` replay 起点如何确定？

设 suffix 为 `[s,e)`，固定上下文为 `H=192`：

[
b_{\mathrm{fixed}}=\max(0,s-H).
]

代码查找所有在 suffix 内 active 或在 suffix 内 end-crossing 的 relevant GT instances：

* 若 relevant instance 有 observable birth bin，则把 replay 起点向前扩展到其中最早 birth；
* 若任何 relevant instance 的 birth 在当前观测体系下不可见，则 fallback 到 video start；
* 否则：

[
b_{\mathrm{dynamic}}
====================

\min\left(
b_{\mathrm{fixed}},
\min_{i\in\mathrm{relevant}} b_i
\right).
]

`[CODE-VERIFIED]` 见 `opentad/utils/crs_eps_sampling.py:374-421`。

### 2. 哪些 replay token 参与 numerical state，哪些参与 autograd？

`[CODE-VERIFIED]`

* `[replay_start, supervised_end)` 的**全部 token**参与 forward numerical state、feature memory、persistent query 更新、supervision transition 和 runtime `decode_step`。

* `global_bin < gradient_start` 的 token 在 `torch.set_grad_enabled(False)` 下执行，只重建数值状态。

* `global_bin >= gradient_start` 的 token进入 autograd。

* 在每个 `gradient_ranges[*][0]` 再调用 `_detach_head_state`。

* loss 只在 `[supervised_start, supervised_end)` 累积。

### 3. `fixed_192` 与 `dynamic_birth` 何时退化为相同 replay range？

当：

1. 所有 suffix-relevant instance 的 observable birth 都不早于 `s-192`；且
2. 不存在需要 fallback-to-zero 的 unobservable relevant birth，

则：

[
b_{\mathrm{dynamic}}=b_{\mathrm{fixed}}.
]

`[CODE-VERIFIED]` 这是几何定义的直接结果。两个真实失败样本上完全相同的 fidelity metrics 与该退化高度一致，但没有 raw `replay_range` 时，不能仅由相同 metrics 反推起点必然相同。

### 4. `video_start_full` gold 的 gradient path 覆盖哪些 token？

`[CODE-VERIFIED]`

* numerical forward 从 bin 0 到 `supervised_end`；
* autograd 并不从 bin 0 连续保留；
* gradient 起点仍是：

[
\left\lfloor\frac{s}{64}\right\rfloor 64,
]

并在后续 64-bin boundaries detach；

* loss 仍只来自相同 suffix。

因此它是：

```text
full-prefix numerical-state control
+ globally aligned 64-bin TBPTT
+ sampled suffix objective
```

不是 full-video exact BPTT。

### 5. 三个 candidate 是否使用相同 RNG、model bytes、suffix、weights、denominator？

`[CODE-VERIFIED]` 是，除候选定义有意改变的 replay start、left-censored schedule 和 `reset` gradient range 外：

* paired models 在 forward 前要求 `state_dict` tensor 完全相等；
* left/right 恢复相同 RNG snapshot；
* 每次 pair 后恢复原始 RNG；
* G0 arms 从同一个原始 draw copy 出来；
* `supervised_range`、sampling weights 和 singleton denominator 相同；
* gold/candidate 都对 `cost` 做 backward。

### 6. 比较的是相同 scalar objective 的近似吗？

严格说，**不是同一个参数函数**。

表面公式、suffix、weights 和 loss names 相同，但：

[
L_a^{\mathrm{gold}}(\theta)
===========================

\sum_{t\in S(a)}w_t
\ell_t(\theta,h_t^{\mathrm{gold}}(\theta),y_t),
]

而 candidate 实际计算：

[
\widehat L_a(\theta)
====================

\sum_{t\in S(a)}w_t
\ell_t(\theta,\widehat h_t^{a}(\theta),y_t).
]

只要

[
\widehat h_t^{a}\neq h_t^{\mathrm{gold}},
]

二者就是不同的 state-conditioned objectives。G0 正是在检验“状态不同是否仍可忽略”，结果是否定的。

### 7. HH/IPW 修正发生在哪一层，不能修正什么？

`[CODE-VERIFIED]` 修正只发生在 suffix 中的逐 bin loss numerator：

```python
weighted = {name: value * weight for name, value in raw.items()}
```

它不能修正：

```text
replay_start 之前缺失的 runtime state
persistent query history
feature-memory history
slot_status / refractory / start state
peak score / peak label history
canonical instance-to-slot ownership
threshold crossing and emitted lifecycle
omitted historical Jacobians
```

配置本身也明确写明 exact IPW does not correct state bias。

### 8. discrete runtime mismatch 比较哪些字段？

`[CODE-VERIFIED]` 精确字段为：

```text
slot_status
refractory
label_state
start_state_nan_mask
source_frames
last_decision_frame
```

任一字段不同即 `runtime_discrete_equal=False`，没有 tolerance。连续向量则拼接：

```text
queries
feature_memory
nan_to_num(start_state)
score_state
```

真实三个失败样本究竟是上述哪些字段分叉，当前 `[UNKNOWN]`，因为 raw trace rows 不在 GitHub。

### 9. slot exhaustion、birth assignment 和 endpoint ownership 是否可能影响失败？

`[CODE-VERIFIED][INFERENCE]` 会，而且很可能是离散分叉的放大器。

* runtime decode 用 detached birth/alive/end/class score 和硬阈值更新 `FREE→ACTIVE→REFRACTORY/FREE`；

* supervision birth 只能使用 runtime-free 且 canonical-unoccupied 的 slots；

* canonical binding 在 birth 时建立；

* rematch 只改变 loss binding，不改变 canonical ownership；

* endpoint slot 从 canonical state 得出。

一个很小的 query/score 差异可能跨过 hard threshold，继而改变：

```text
slot availability
birth slot
canonical ownership
endpoint ownership
future refractory state
```

训练引擎会对非零 `slot_exhaustion` fail-closed，但 G0 direct paired runner 主要记录 trace，真实样本是否发生 exhaustion 当前未知。

### 10. gate 是否存在明显代码错误？

`[CODE-VERIFIED]` 未发现足以把 PASS 错写成 KILL 的明显错误。

已核验：

* exact four-arm completeness；

* outcome-blind margin status；

* matched model/RNG；

* finite and shape-compatible gradient/state vectors；

* all trainable-parameter gradient flattening；

* loss structure equality；

* candidate/gold replay token ratio；

* absolute dynamic margins；

* aggregate dynamic-minus-fixed/reset margins；

* zero violations 才 PASS；

* launch side重新计算 gate；

* KILL 无法授权 profile。

唯一值得单独检查的局部敏感性是：

[
\text{relative error}
=====================

\frac{|L_{\mathrm{cand}}-L_{\mathrm{gold}}|}
{\max(|L_{\mathrm{gold}}|,10^{-12})},
]

且 gate 取所有 loss component 中的最大值。若某个 gold component 极小，relative error 会放大。

但这不能解释：

* gradient cosine 低至 `0.2831`；
* sign agreement 低至 `0.5445`；
* continuous runtime cosine 低至 `0.1028`；
* 三个 exact discrete mismatch。

所以即使某一 loss-error violation 经 raw per-loss audit 被证明过敏，terminal `KILL` 仍有多条独立失败路径。

---

# D. Evidence Integrity Table

| Evidence                   | 标签                           | Hash / 状态                                         | 能支持什么                                         | 不能支持什么                        |
| -------------------------- | ---------------------------- | ------------------------------------------------- | --------------------------------------------- | ----------------------------- |
| Local B0                   | `[SIGNED-ARTIFACT-REPORTED]` | `588/588 PASS`；root hash 未提供                      | 作者报告本地测试通过                                    | 本次未执行；不证明 scientific fidelity |
| Target-Linux B0            | `[SIGNED-ARTIFACT-REPORTED]` | `588/588 PASS`；root hash 未提供                      | Linux 环境 contract tests 通过                    | 不证明 G0 或模型质量                  |
| Locked review              | `[SIGNED-ARTIFACT-REPORTED]` | `be4b82aa…a06b`，`PASS / NEXT_GATE=G0`             | prior blockers 被报告关闭                          | 原始签名 review 未上传；不能据此宣称方法有效    |
| Selection                  | `[SIGNED-ARTIFACT-REPORTED]` | `457838e8…ea4f`                                   | sample IDs/checkpoint policy outcome-blind 冻结 | 未看到选择规则全文和 strata             |
| Margins                    | `[SIGNED-ARTIFACT-REPORTED]` | `5160adab…4329`                                   | 阈值在 terminal outcome 前冻结                      | 不能证明阈值本身是领域最优                 |
| Checkpoint                 | `[SIGNED-ARTIFACT-REPORTED]` | `68169922…e5f`                                    | 精确 bytes、seed 705、参数数目被绑定                     | 明确不是模型质量或 trajectory 代表性证据    |
| Manifest                   | `[SIGNED-ARTIFACT-REPORTED]` | identity `96278dce…0e5`                           | sampling population 和 draws 被绑定               | 原始 manifest 未独立检查             |
| Terminal audit             | `[SIGNED-ARTIFACT-REPORTED]` | `a08183a8…109b`，`KILL`                            | 当前 frozen protocol 的终态                        | raw rows 不在 GitHub，无法本次逐字段复算  |
| Target-Linux recomputation | `[REVIEW-REPRODUCED]`        | 报告为相同 `KILL`                                      | 降低单机计算差异解释                                    | 原始复算日志未上传                     |
| Gate implementation        | `[CODE-VERIFIED]`            | deterministic `PASS iff violations=[]`            | 真实 rows 给定时可重算 terminal status                | 不证明输入 rows 本身真实，需签名链          |
| Launch blocker             | `[CODE-VERIFIED]`            | 非 PASS 抛出 `CRS-EPS G0 audit has not reached PASS` | KILL 后无法构造合法 CRS-EPS profile authorization    | 不说明替代路线应是什么                   |

HEAD dossier 明确把结果限制为当前 `dynamic_birth` surrogate，且明确不是 effectiveness、raw-video 或 infrastructure failure。

证据链中的 checkpoint、selection、margins、manifest 和 audit hashes 与 terminal metrics 被文档记录，但原始 bundle 位于 repo 外。

---

# E. G0 Metric Semantics Audit

| Metric                    | 代码实际比较对象                                                       | 解释                                    |        |   |             |                                    |
| ------------------------- | -------------------------------------------------------------- | ------------------------------------- | ------ | - | ----------- | ---------------------------------- |
| Per-loss value            | paired gold/candidate 的同名 weighted suffix losses，包括 `cost`     | 不是完整视频 epoch loss                     |        |   |             |                                    |
| Relative loss error       | 所有同名 loss 中最大 (                                                | c-g                                   | /\max( | g | ,10^{-12})) | 一个 component 即可触发 sample violation |
| Gradient cosine           | 所有 `requires_grad=True` 参数的 gradient flatten 后整体 cosine        | 缺失 gradient 按零 tensor 处理              |        |   |             |                                    |
| Gradient sign agreement   | 任一侧非零坐标上的 sign 相等比例                                            | 不受 gradient norm 绝对尺度直接决定             |        |   |             |                                    |
| Continuous runtime cosine | `queries + feature_memory + start_state(把NaN置0) + score_state` | 不含 label、status、refractory 等离散字段      |        |   |             |                                    |
| Discrete equality         | 六类字段的 Python structure 精确相等                                    | 任意 lifecycle 差异即失败                    |        |   |             |                                    |
| Logits equality           | 全 replay logits digest 是否相等                                    | 记录但不是 frozen gate 的单独 margin          |        |   |             |                                    |
| Replay ratio              | candidate replay token 数 / `video_start_full` replay token 数   | 不是 backward-token ratio，也不是 wall time |        |   |             |                                    |
| Fallback fraction         | candidate replay start 是否与 gold start 相同                       | `0.25` 表示 4 样本中 1 个从视频开头重放            |        |   |             |                                    |
| Dynamic-minus-fixed/reset | 每样本 gradient cosine 差后取平均                                      | 只评价相对优势，不能覆盖绝对 fidelity 失败            |        |   |             |                                    |

代码对 gradient 以 double precision 计算 cosine，两个零向量定义 cosine 1，一侧零则定义 0；sign agreement 只在至少一侧非零的位置计算。

`replay_ratio` 的分母就是同一 trace 左侧 `video_start_full` replay token 数，fallback 则判断 candidate 和 gold 的 replay start 是否相同。

绝对 dynamic violations 是逐样本触发；dynamic 相对 fixed/reset 的优势只在最后求平均。因此“平均 dynamic 更好”从设计上就不能挽救任何绝对 state/gradient failure。

---

# F. Per-Sample Failure Analysis

所有数字在本节均为 `[SIGNED-ARTIFACT-REPORTED]`。

## F.1 `video_validation_0000151:draw=0`

```text
grad cosine      = 1.0000
sign agreement   = 1.0000
relative error   = 0.0000
runtime cosine   = 1.0000
discrete equal   = yes
replay ratio     = 1.0000
fallback         = video start
```

裁决：

* 这是 audit 的正控。
* candidate 和 gold replay range 相同后，loss、gradient、continuous state 和 discrete state 全部精确闭合。
* 它显著反驳“paired RNG、model deepcopy、metric 或 gate 总是会制造假差异”的解释。
* 它不证明 dynamic replay 低成本，因为该样本没有节省任何 replay token。

## F.2 `video_validation_0000946:draw=1`

```text
dynamic gradient cosine = 0.5130
fixed gradient cosine   = 0.0564
relative loss error     = 0.3885
runtime cosine          = 0.1214
discrete equal          = no
replay ratio            = 0.7704
```

裁决：

* `[INFERENCE]` dynamic birth extension 在该样本确实比固定 192 context 携带了有用历史；否则很难解释 gradient cosine 从 `0.0564` 提升到 `0.5130`。
* 这反驳“H5：dynamic 总是退化为 fixed”的全称版本。
* 但 `0.5130` 仍远低于 `0.90`，runtime cosine `0.1214` 显示数值状态几乎不是 gold state，且离散 lifecycle 已分叉。
* 因而这是“动态上下文有相对作用，但机制仍不充分”，不是 fix success。

## F.3 `video_validation_0000051:draw=1`

```text
grad cosine      = 0.2831
sign agreement   = 0.5445
relative error   = 0.8021
runtime cosine   = 0.1028
discrete equal   = no
replay ratio     = 0.7752
dynamic metrics  = fixed_192 metrics
```

这是最强失败样本：

* gradient direction、coordinate signs、loss 和 runtime state 全部大幅偏离。
* `dynamic_birth == fixed_192` 的全部报告 metrics 与代码中的 replay-range 退化条件高度一致。
* 但 `[UNKNOWN]` 没有 raw ranges 时，仍不能排除两条不同 replay trajectory 最终偶然产生相同 summary metrics。
* 即使证明 dynamic range 比 fixed 更长，也只能说明“额外 suffix-relevant birth replay 没有改变终态”；这反而进一步削弱当前 dynamic criterion。

## F.4 `video_validation_0000163:draw=0`

```text
grad cosine      = 0.5041
sign agreement   = 0.6313
relative error   = 0.4197
runtime cosine   = 0.9852
discrete equal   = no
replay ratio     = 0.4890
dynamic metrics  = fixed_192 metrics
```

这是最有诊断价值的样本：

* continuous cosine `0.9852` 很高，但 exact discrete state 仍不相等。
* persistent head 的 birth/alive/end decision 是 detached probabilities 上的 hard threshold；很小的连续扰动可以改变 `slot_status`、start NaN mask、label 或 refractory，之后产生不同 canonical ownership 和 future trajectory。
* 因此 cosine 高不等于 state-equivalent。
* `[INFERENCE]` 该样本更符合“阈值附近 lifecycle bifurcation”而非大幅连续表示崩坏。
* exact diverging field 和 first-divergence bin 当前 `[UNKNOWN]`。

## F.5 两个 `dynamic_birth == fixed_192` case 的最终解释

代码上的充分条件是：

```text
dynamic_replay_start == max(0, supervised_start - 192)
```

真实 artifact summary 只说明两臂的报告 metrics 完全一致，不能独立证明 replay ranges 相同。要把 H5 从“高度一致”提升为“直接观测”，必须读取 raw rows 中：

```text
right.audit.replay_range
right.audit.dynamic_extension_instance_ids
right.audit.birth_assignments
right.audit.canonical_lifecycle
right.audit.endpoint_slot_trace
right.audit.slot_exhaustion
```

无论两臂是否真的同 range，都不改变 terminal KILL：

* 若同 range：dynamic criterion 在两样本没有提供额外状态；
* 若不同 range 但 metrics 同：额外 replay 对这些 state/gradient 没有可观测收益；
* 两种情况都不能满足绝对 fidelity。

---

# G. Mathematical Bias Decomposition

## G.1 完整 recurrent objective

设：

[
h_t(\theta)=F_\theta(h_{t-1}(\theta),x_t),
]

[
\ell_t(\theta,h_t,y_t)
]

为 bin (t) 的监督损失，目标为：

[
L_{\mathrm{full}}(\theta)
=========================

\sum_{t=1}^{T}\alpha_t
\ell_t(\theta,h_t(\theta),y_t).
]

定义：

[
J_t
===

\frac{\partial h_t}{\partial \theta}.
]

则：

[
J_t
===

\frac{\partial F_\theta}{\partial \theta}
+
\frac{\partial F_\theta}{\partial h_{t-1}}J_{t-1},
]

并且：

[
g_{\mathrm{full}}
=================

# \nabla_\theta L_{\mathrm{full}}

\sum_t \alpha_t
\left[
\partial_\theta \ell_t
+
\partial_h\ell_t,J_t
\right].
]

## G.2 实现中的 chronological TBPTT gold

当前真正实现的 complete chronological cached route 使用全视频因果数值状态，但在 64-bin chunk boundaries detach。记截断后的 Jacobian 为 (J_t^{(64)})：

[
J_b^{(64)}=0
\quad
\text{at each global boundary }b.
]

因此 implemented gold gradient 是：

[
g_{\mathrm{chrono\text{-}64}}
=============================

\sum_t\alpha_t
\left[
\partial_\theta \ell_t
+
\partial_h\ell_t J_t^{(64)}
\right],
]

而不是理论上的 (g_{\mathrm{full}})。

这意味着存在两个不同问题：

1. **chronological 64-bin TBPTT 相对 exact full BPTT 的历史 Jacobian bias；**
2. **CRS-EPS candidate 相对 chronological 64-bin gold 的 state/replay bias。**

当前 G0 直接检验的是第二项。

## G.3 CRS-EPS sampling estimator

令 anchor/episode (A\sim q(a))，suffix set 为 (S(a))。定义：

[
I_t(a)=\mathbf 1[t\in S(a)],
]

[
\rho_t
======

# \Pr_{A\sim q}[I_t(A)=1]

\sum_a q(a)I_t(a).
]

对于 per-video uniform decision-bin mean：

[
\alpha_t=\frac{1}{T},
]

HH weight 为：

[
w_t=\frac{\alpha_t}{\rho_t}
=\frac{1/T}{\rho_t}.
]

M 个独立 draws 的 estimator 为：

[
\widehat L_M(\theta)
====================

\frac1M
\sum_{m=1}^{M}
\sum_{t\in S(A_m)}
w_t,
\ell_t
\left(
\theta,
\widehat h_t^{A_m}(\theta),
y_t
\right).
]

其 sampled gradient 为：

[
\widehat g_M
============

\frac1M
\sum_{m=1}^{M}
\sum_t
I_t(A_m)w_t
\left[
\partial_\theta\ell_t
(\theta,\widehat h_t^{A_m})
+
\partial_h\ell_t
(\theta,\widehat h_t^{A_m})
\widehat J_t^{A_m}
\right].
]

## G.4 HH/IPW 无偏所需条件

即使 sampling measure 完全正确，要有：

[
\mathbb E[\widehat g_M]
=======================

g_{\mathrm{chrono\text{-}64}},
]

至少需要：

1. **Support**
   (\rho_t>0) 对所有 target bins 成立。

2. **Measure correctness**
   (q(a))、(\rho_t)、target (\alpha_t) 和 M-draw normalization 精确匹配。

3. **Forward-state equivalence**
   对所有被监督 bin：

   [
   \widehat h_t^a=h_t^{\mathrm{gold}}.
   ]

4. **Jacobian equivalence under the chosen estimand**

   [
   \widehat J_t^a=J_t^{(64)},
   ]

   或被省略项的加权期望严格为零。

5. **Discrete-path equivalence**
   相同 slot status、birth ownership、refractory、endpoint ownership 和 threshold branch。

6. **Stochasticity equivalence**
   RNG 或随机算子的条件分布相同。

当前代码较强地满足 1、2、6；G0 直接表明 3 和 5 在三样本上失败。条件 4 相对 exact full BPTT 不成立；相对 G0 的 64-bin gold，双方采用相同 boundary rule，但 candidate 的 boundary state 不同，因此其局部 Jacobian 仍不同。

## G.5 七类误差分解

可以写成：

[
\mathbb E[\widehat g_M]-g_{\mathrm{target}}
===========================================

B_{\mathrm{measure}}
+
B_{\mathrm{weight}}
+
B_{\mathrm{state}}
+
B_{\mathrm{Jacobian}}
+
B_{\mathrm{lifecycle}}
+
B_{\mathrm{stochastic}}
+
\varepsilon_M.
]

其中：

### 1. Sampling-measure error

[
B_{\mathrm{measure}}
]

来自 proposal support 或 sampling population 与 target bins 不匹配。

当前代码：

* uniform component 保证 support；
* manifest 固化 sampling population；
* synthetic enumeration tests 检验 `rho_t`。

所以它不是当前最强解释。

### 2. Weight / denominator error

[
B_{\mathrm{weight}}
]

来自错误 `rho_t`、错误 target denominator、错误 M normalization 或 double division。

当前实现：

```text
per-bin raw weight = (1/T)/rho_t
per-draw group scale = 1/M
sum of M denominators = 1
one optimizer event
```

没有发现 double division。

### 3. Forward-state reconstruction error

[
B_{\mathrm{state}}
]

来自：

[
\widehat h_{r(a)}
=================

h_{\mathrm{empty}}
\neq
h_{r(a)}^{\mathrm{chronological}}.
]

这是当前最直接的结构性误差。candidate 从空 state 开始，而 gold 在同一时间点已经积累了完整 prefix 的 recurrent query 和 lifecycle。

### 4. Omitted historical Jacobian error

[
B_{\mathrm{Jacobian}}
]

来自：

[
\widehat J_t
\neq J_t.
]

对 exact full BPTT，这是所有 64-bin TBPTT 路线共有的问题。对当前 paired G0，它不是主要差异源，因为 gold 和 dynamic 共享同一 global detach schedule；真正不同的是 detach boundary 上的数值 state。

### 5. Lifecycle / assignment path divergence

[
B_{\mathrm{lifecycle}}
]

来自不可微或 piecewise-constant branch：

```text
threshold crossing
FREE / ACTIVE / REFRACTORY
birth slot availability
canonical binding
endpoint slot ownership
left-censored supervision removal
```

一旦 branch 不同，后续 loss 不再是同一 smooth local objective。

### 6. Stochasticity mismatch

[
B_{\mathrm{stochastic}}
]

来自 dropout、random sampling 或 nondeterministic kernels未配对。

当前 paired audit 恢复同一 RNG，fallback sample 又精确闭合，因此没有证据把主要失败归因于此。

### 7. Finite-sample variance

[
\varepsilon_M
]

即便 estimator unbiased，M=4 draws 仍有 variance。但 G0 不是用四个 draws 估计一个总体均值后做宽松统计推断；它对四个预注册 stress samples 执行逐样本绝对 safety margins。三个直接失败已足以终止冻结 gate。有限样本限制的是方法族外推，而不是当前 terminal decision。

---

# H. Root-Cause Matrix H1–H8

| ID                                                      | Code evidence / observed signature                                                                                                                   | Contradicting evidence                                                                           | Current identifiability                                     | Smallest zero-GPU diagnostic                                                                                             | New confirmation?                                        |
| ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------- |
| **H1** omitted historical Jacobian 是主因                  | `[CODE-VERIFIED]` 所有臂按 global 64-bin boundaries detach；理论上相对 full BPTT 确有 bias。                                                                      | gold 与 dynamic 使用同一 `gradient_ranges`；fallback 同 range 时精确闭合。若 H1 是 paired 差异主因，双方共有的截断不应产生如此差异。 | **相对 exact full BPTT：成立。相对当前 G0 paired failure：不是主因。**      | 在线性/平滑 toy RNN 上比较 exact BPTT、chronological-64、从正确 boundary state 开始的 replay-64。                                         | 当前 KILL 不需要；任何声称 exact-gradient 的新路线需要。                  |
| **H2** forward numerical state 未忠实重建                    | candidate 从 empty runtime/supervision state 开始；三个非 fallback 样本 continuous/discrete state 失败。                                                         | fallback full replay 样本完全匹配。                                                                     | **高可识别：失败存在。** exact first divergence field/bin 未知。         | 对 exposed samples 逐 token 比较 gold/candidate `queries`、memory、score、status、refractory；二分 first-divergence bin。            | 不需要新 GPU；新 surrogate 必须 unseen G0。                       |
| **H3** HH target、union probability 或 denominator 错      | `rho_t`、raw HH weight、M group scale、one-event transaction 代码和 exhaustive tests 均闭合；union inclusion 只做 diagnostic。                                    | raw real manifest 未在本会话重算；per-loss relative error 可能对小 denominator 敏感。                           | **明显通用 denominator bug：低概率。** real-manifest 局部异常仍可检查。       | 从 signed manifest 枚举每 bin coverage，重算 expected HH mean、M denominator 和 raw per-loss values。                              | 只有发现真实 inconsistency 才需新 confirmation；否则不改变 KILL。        |
| **H4** lifecycle / birth / endpoint 路径在 replay start 分叉 | hard threshold decode；runtime-free birth mask；canonical slot ownership；三个 discrete failures；`0000163` continuous 0.9852 仍 discrete fail。             | 未知具体 discrete field，也未知 slot exhaustion。                                                         | **分叉存在，高可识别；具体机制部分可识别。**                                    | 输出每 bin六类 discrete fields、birth assignment、canonical binding、endpoint slot、exhaustion，并定位首个差异。                           | 当前 KILL 不需；新 route 的 holdout G0 必须 exact discrete equal。 |
| **H5** dynamic 经常退化为 fixed                              | 代码允许 `dynamic_start=base_start`；两个样本 summary 完全相同。                                                                                                   | `0000946` dynamic cosine `0.5130`、fixed `0.0564`，说明并非总是退化。                                       | **四样本上部分成立；总体频率未知。**                                        | 对完整 manifest 做 CPU-only geometry census：dynamic=fixed fraction、extension lengths、fallback、relevant-birth strata。         | 统计频率不挽救旧 KILL；会影响新方法设计。                                  |
| **H6** paired audit / metric 有局部错误                      | tests 覆盖 RNG、model equality、NaN sentinel、vector stability、four-arm completeness、post-hoc margins、KILL launch block；fallback 精确闭合；独立 Linux 报告相同 KILL。 | relative-loss max 对 near-zero gold component 敏感；真实 raw rows未复算。                                  | **足以逆转 KILL 的 audit bug：低概率。** 局部 metric sensitivity 未完全排除。 | 从 raw rows独立实现一次 loss、gradient、continuous/discrete 和 gate 重算；列每个 violation 的原始 numerator。                                | 仅当独立重算不一致才需要 replacement G0；不能事后改 margins。               |
| **H7** random-init checkpoint 不代表训练中 fidelity           | checkpoint 明确标为 “not model-quality evidence”；只有一个 parameter state。                                                                                   | 如果 formal surrogate 原本就从该 init 开始，则 first update failure 是直接相关，而非纯代表性问题。                         | **代表性不足成立；不推翻当前起点 KILL。**                                   | 用 chronological gold route 生成 outcome-independent early/mid/late checkpoints，在 exposed samples 上做 development diagnosis。 | 新 route 应在预注册 checkpoint strata 上做 holdout G0。           |
| **H8** 四样本只 kill 精确协议，不 kill 整个方法族                      | gate 是预注册 safety gate；3/4 失败足以 terminal KILL。                                                                                                        | 样本数太少，不能估计 population failure rate，也不能证明所有 checkpoint/所有 state-faithful event sampling 都失败。      | **成立。** 精确协议已死，方法族外推不足。                                     | 无需 GPU：冻结旧样本为 development，重新定义方法和 checkpoint policy，再选 unseen holdout。                                                   | 是；任何 replacement route 都需全新 confirmatory G0。             |

## H.1 根因优先级

```text
Primary:
H2 forward-state reconstruction failure
H4 lifecycle / assignment branch divergence

Secondary:
H5 dynamic context often provides no extra history

Common but not paired-G0 primary:
H1 global TBPTT historical-Jacobian bias

Low-probability reversal explanations:
H3 denominator bug
H6 audit bug

Scope limitations:
H7 one checkpoint
H8 four samples
```

---

# I. Scope of Falsification

| 层级                                                      | 裁决                    | 解释                                                                                                            |
| ------------------------------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------- |
| 1. `70df86e + seed705 + frozen G0`                      | **`VALID_KILL`**      | `[SIGNED-ARTIFACT-REPORTED][CODE-VERIFIED]` 冻结 gate 终态有效；不得改 margins、换 checkpoint 或重用样本挽救。                    |
| 2. 当前 CRS-EPS `dynamic_birth` primary surrogate         | **`STRUCTURAL_KILL`** | 它从空 state 开始，replay criterion不构成 chronological persistent state 的充分统计量；HH 不能修复。                               |
| 3. 更大的 event-centric prefix sampling 方法族                | **未被否定**              | state-faithful full-forward/sampled-loss、exact state checkpoints 或 unbiased recurrent-gradient methods 是不同方法。 |
| 4. Q2 full chronological cached reference               | **`KEEP_AS_PRIMARY`** | 仍是当前唯一保持完整 causal numerical state 的干净训练路线；其 estimand 是 cached、chunk-detached Q2。                              |
| 5. Full PETAL persistent lifecycle/model infrastructure | **`KEEP`**            | G0 反而依赖该基础设施和 audit 来发现失败；没有 evidence 表明 lifecycle implementation 本身无效。                                       |
| 6. raw-video end-to-end On-TAD 长期目标                     | **未被否定，但继续 BLOCK**    | 当前没有 raw frames forward/backward、visual parameter adaptation 或 formal effectiveness evidence。                 |

换言之：

```text
KILLED:
current isolated dynamic-birth replay surrogate
its current C4 cost claim
its current profile/formal authorization chain

NOT KILLED:
fully supervised completion-triggered On-TAD
Q2 fixed-vs-rematch question
persistent detector infrastructure
event-centric sampling as a broad research family
future raw-video adaptation
```

仓库的正式 failure dossier 也明确限制了 falsification scope，并要求 exposed samples 今后只能用于 development。

---

# J. Random-Checkpoint Validity

## J.1 Outcome-blind initialization解决了什么？

`[CODE-VERIFIED]` 它解决：

```text
checkpoint cherry-picking
看到 G0 结果后换 seed
事后选择最有利 parameter state
checkpoint bytes 替换
state_dict key 漂移
```

它不解决：

```text
训练轨迹代表性
早期/中期/后期 state sensitivity
不同 parameter basin
不同 lifecycle threshold calibration
模型质量
```

## J.2 是否应该在 init、early、late 都满足 fidelity？

对一个将替代 chronological training 的 primary surrogate，答案是 **原则上应该**。

更准确地说，需要覆盖 surrogate 实际会经过的 parameter-state support。至少应预注册：

```text
initial checkpoint
early gold-trained checkpoint
intermediate gold-trained checkpoint
late gold-trained checkpoint
```

如果 surrogate 从当前 random init 开始，则 init failure 尤其严重：它意味着第一批 optimizer events 已偏离 gold，不能用“也许训练后会变好”绕过，因为到达后期 checkpoint 的路径本身已经被有偏更新改变。

## J.3 如何在不接受有偏训练的前提下获得 trajectory checkpoints？

可以由 full chronological gold route 独立生成：

[
\theta_0,\theta_{\mathrm{early}},
\theta_{\mathrm{mid}},
\theta_{\mathrm{late}}.
]

然后冻结：

```text
checkpoint bytes
checkpoint selection epochs
sample IDs
margins
aggregation rule
family-wise kill rule
```

再评估 replacement surrogate。

这不依赖被审查 surrogate 的训练结果，因此不存在循环论证。

## J.4 多 checkpoint 会不会使低成本 gate 失去意义？

不必然。

* G0 本身是 CPU cached-feature fidelity audit；
* gold Q2 route 本来就需要作为 baseline/reference；
* checkpoint bank 可从一次 gold development trajectory 抽取；
* confirmatory 样本数和 checkpoint 数可以小，但必须 outcome-blind；
* family-wise rule 必须事先冻结，例如“任一 required checkpoint 出现绝对 state/gradient violation 即 KILL”。

真正昂贵的是 gold training；但若连 gold baseline 都不愿运行，Q2 effectiveness 本身也无法成立。

## J.5 这会推翻当前 terminal KILL 吗？

不会。

```text
CURRENT KILL:
same frozen checkpoint, samples, margins and code
→ immutable terminal decision

FUTURE MULTI-CHECKPOINT PROTOCOL:
new method, new code commit, new selection,
new signed artifacts and new terminal G0
```

随机初始化代表性不足只能约束未来协议，不能把当前 `KILL` 改写为 `PASS`、`UNKNOWN` 或 “暂不算”。

---

# K. Competition and Novelty Audit

## K.1 检索协议

```text
检索日期：2026-07-16
时区：America/New_York
来源限制：论文原文、arXiv primary page、官方项目页、作者代码仓库、PyTorch 官方文档
```

代表性检索式：

```text
"Online Temporal Action Localization" 2026
MATR HAT ActionSwitch SimOn official paper
causal streaming video backbone StreamFormer E2E-LOAD LSTR
persistent query TrackFormer MOTR temporal slot activation
ETAD full forward sparse backward
Stochastic Backpropagation video training
ARTBP UORO truncated BPTT unbiased recurrent gradient
activation checkpointing selective activation checkpoint PyTorch
state dependent sampling marginalized importance sampling
```

## K.2 Direct On-TAL / On-TAD

### SimOn

```text
Paper: https://arxiv.org/abs/2211.04905
Code:  https://github.com/TuanTNG/SimOn
```

SimOn 已明确使用只可访问过去帧、不可修改历史预测的 On-TAL 协议，并以当前 feature query 加过去视觉 context 直接预测 action instances。它削弱“严格在线 instance localization 本身”的新颖性。([arXiv][1])

### MATR

```text
Paper:   https://arxiv.org/abs/2408.02957
Project: https://cvlab.postech.ac.kr/research/MATR/
```

MATR 使用 long-term memory queue、instance queries、current segment end decoder 和 memory-based start decoder，并明确要求过去输出不可回写。其任务、long-term history、instance decoding 与 Full PETAL 高度接近；差异是 MATR 并未维护同一 action identity 的 persistent slot lifecycle。([arXiv][2])

### HAT

```text
Paper: https://arxiv.org/abs/2408.06437
Code:  https://github.com/sakibreza/ECCV24-HAT/
```

HAT 直接把 long/short historical context 引入 On-TAL。它削弱“利用完整历史改善 online localization”的一般性新颖性，但不提供 persistent identity slot 或当前训练代理。([arXiv][3])

### ActionSwitch

```text
Paper: https://arxiv.org/abs/2407.12987
Official code confirmed in this search: no
```

ActionSwitch 面向 completion-triggered simultaneous/overlapping actions，并引入 class-agnostic switch/conservativeness mechanism。它对 overlapping lifecycle 和 completion decision 构成直接竞争，但不等同于 persistent query identity。([arXiv][4])

### 2026 邻近新任务

```text
OnPoint:  https://arxiv.org/abs/2607.00289
OZ-TAL:  https://arxiv.org/abs/2605.09976
OpenHOUSE: https://arxiv.org/abs/2509.12145
```

OnPoint 改为 point-supervised POTAL，并使用 offline-teacher distillation；OZ-TAL 改为 zero-shot/training-free；OpenHOUSE 增加自由文本和 hierarchical streaming description。它们证明 On-TAL 仍活跃，但都改变监督或输出任务，不能作为当前 fully supervised Q2 的同任务替代 baseline。([arXiv][5])

## K.3 Causal streaming video backbones

### StreamFormer

```text
Paper: https://arxiv.org/abs/2504.20041
```

StreamFormer 在预训练 ViT 中加入 causal temporal attention，并面向 OAD、online VIS 和 streaming VideoQA 做多任务训练。它削弱“causal streaming visual backbone”作为 Full PETAL 独立贡献的空间，但没有 completion-triggered On-TAD lifecycle。([arXiv][6])

### E2E-LOAD

```text
Paper: https://arxiv.org/abs/2306.07703
```

E2E-LOAD 是 end-to-end long-form OAD，使用长序列 cache 和不对称 long/short temporal modeling。它是 raw-video causal efficiency 的强邻近工作，但任务仍是 frame-level OAD，不是 action-instance completion localization。([arXiv][7])

## K.4 Persistent query / track-style state

### TrackFormer

```text
Paper: https://arxiv.org/abs/2101.02702
Code:  https://github.com/timmeinhardt/trackformer
```

TrackFormer 已用 static object queries 建立 newborn tracks，并以 autoregressive identity-preserving track queries 跨帧维护实例。([arXiv][8])

### MOTR

```text
Paper: https://arxiv.org/abs/2105.03247
Code:  https://github.com/megvii-research/MOTR
```

MOTR 进一步使用 frame-by-frame updated track queries、newborn object queries 和 tracklet-aware label assignment。([arXiv][9])

### TSA，2026

```text
Paper: https://arxiv.org/abs/2606.13714
```

TSA 明确研究 persistent slots 的 activation lifecycle、inactive-state preservation 和 state drift。虽然它属于 unsupervised object-centric video representation，不是 On-TAD，但它进一步压缩了“persistent slots + lifecycle”作为通用机制的新颖性。([arXiv][10])

**结论：**

`[PRIMARY-VERIFIED][INFERENCE]` “persistent action slots + birth/active/end lifecycle”不是孤立的新机制。论文价值必须来自 On-TAD 特有的 endpoint/identity supervision、训练理论或可验证性能，而不是把 TrackFormer/MOTR 式状态搬到时间轴。

## K.5 Efficient end-to-end TAL/TAD 与 full-forward/sparse-backward

### ETAD

```text
Paper: https://arxiv.org/abs/2205.07134
Official public code confirmed in the primary page: no
```

ETAD 顺序 forward 全视频 snippets，但只让一部分 snippets 反向传播，并进一步采样 proposals。它是与未来 “full causal forward state + sparse backward” 最接近的 published TAL 工作。区别是 ETAD 面向 offline TAD encoder/proposals，并不维护 recurrent action lifecycle；因此它没有解决 state-faithful persistent On-TAD credit assignment。([arXiv][11])

### Stochastic Backpropagation

```text
Paper: https://arxiv.org/abs/2203.16755
```

SBP 保留全部 forward paths，逐层随机移除部分 backward paths，以降低 activation memory。它直接构成“完整 forward、稀疏 backward”先例，因此未来路线不能仅凭该口号声称新颖。([arXiv][12])

### TALLFormer、Re²TAL、LoSA

```text
TALLFormer: https://arxiv.org/abs/2204.01680
Code:       https://github.com/klauscc/TALLFormer

Re²TAL:    https://arxiv.org/abs/2211.14053
LoSA:      https://arxiv.org/abs/2404.01282
```

* TALLFormer 用 long-term memory/feature bank降低长视频端到端训练成本；
* Re²TAL 用 reversible backbone 精确重建中间 activations；
* LoSA 用 long/short adapters 做 memory- and parameter-efficient TAL adaptation。

它们分别覆盖 feature reuse、exact activation reconstruction 和 parameter-efficient adaptation，但均不是 persistent On-TAD state estimator。([arXiv][13])

### Activation checkpointing

```text
Official PyTorch:
https://pytorch.org/blog/activation-checkpointing-techniques/
https://docs.pytorch.org/docs/stable/checkpoint.html
```

PyTorch activation checkpointing、Selective Activation Checkpoint 和 memory-budget API 通过 backward 时重算 forward operations 来减少 activation memory；本质是 memory–compute trade-off，而不是缩短 causal state history或改变 target gradient。([PyTorch][14])

因此 exact chronological chunks + activation checkpointing 是科学上干净的成本路线，但未必节省 forward tokens 或 GPU-hours。

## K.6 Long-sequence recurrent credit assignment

### ARTBP

```text
Paper: https://arxiv.org/abs/1705.08209
```

ARTBP 明确指出 fixed truncated BPTT gradient 有偏，并通过随机 truncation length 加 compensation factors 构造无偏 estimator。([arXiv][15])

### UORO

```text
Paper: https://arxiv.org/abs/1702.05043
```

UORO 是 online RTRL approximation，避免回溯过去 activations，并提供无偏但高方差的 recurrent gradient estimate。([arXiv][16])

### Exact/rematerialized BPTT checkpointing

```text
Memory-Efficient BPTT:
https://arxiv.org/abs/1606.03401

Training Deep Nets with Sublinear Memory:
https://arxiv.org/abs/1604.06174

Optimal recurrent checkpointing:
https://arxiv.org/abs/2412.11810
```

这些方法通过保存部分 states 和 recomputation 降低内存，但保留或近似完整 credit path；它们与 CRS-EPS 的“直接删除 prefix state/Jacobian”有本质区别。([arXiv][17])

## K.7 State-dependent sampling 与 off-policy correction

```text
Not All Samples Are Created Equal:
https://arxiv.org/abs/1803.00942

Marginalized Importance Sampling:
https://arxiv.org/abs/1906.03393

Doubly Robust OPE:
https://arxiv.org/abs/1511.03722
```

普通 supervised importance sampling 可以在样本及其 gradient 定义不变时修正 sampling measure 或降低 variance。([arXiv][18])

在 sequential state-dependent 问题中，off-policy literature 必须处理 state marginal distribution，而不仅是当前 decision/action 的概率比；MIS 就显式估计 target/behavior state marginals。该文献不能直接当作 CRS-EPS 定理，但它支持一个重要反方结论：**只修正 decision-bin exposure probability，不能自动修正不同历史产生的 state distribution。** ([arXiv][19])

## K.8 Overlap Matrix

| 工作族                               | 与 Full PETAL / CRS-EPS 的精确重叠                                                  | 尚未覆盖的 gap                                    | 对 novelty 的影响                       |
| --------------------------------- | ----------------------------------------------------------------------------- | -------------------------------------------- | ----------------------------------- |
| SimOn / MATR / HAT / ActionSwitch | strict online action-instance localization、history、queries、completion/overlap | persistent identity lifecycle 和当前低成本训练问题     | 任务和历史建模不能单独作为创新                     |
| TrackFormer / MOTR / TSA          | persistent slots/queries、birth、identity、activation lifecycle                  | temporal action endpoint 与 On-TAD evaluation | persistent slot package 高度可重构       |
| StreamFormer / E2E-LOAD           | causal backbone、stream cache、long/short online modeling                       | completion-triggered instance localization   | causal backbone 不是独占贡献              |
| ETAD / SBP                        | full forward、sampled/sparse backward                                          | recurrent state/lifecycle fidelity           | “full-forward sparse-backward”已有强先例 |
| TALLFormer                        | long-memory feature reuse                                                     | cache staleness与 persistent state            | feature bank/reuse 本身不新             |
| Re²TAL / activation checkpointing | exact rematerialization、降低 activation memory                                  | 不减少必须 forward 的 causal tokens                | 工程路线强，但算法 novelty 有限                |
| ARTBP / UORO                      | truncated-gradient bias 与无偏 recurrent estimator                               | video/On-TAD-specific lifecycle              | 新方法必须正面处理 credit bias               |
| MIS / DR-OPE                      | sequential state-distribution correction                                      | 非监督 On-TAD直接解法                               | 反驳仅靠 bin-level IPW 的充分性             |

## K.9 Novelty verdict

```text
CURRENT FULL PETAL PACKAGE NOVELTY = WEAK / RECONSTRUCTIBLE
CURRENT CRS_EPS NOVELTY = SCIENTIFICALLY INACTIVE AFTER G0 KILL
EXACT "FIRST" CLAIM = NOT SUPPORTED
```

没有检索到与 `CRS-EPS` 完全同名同构的工作，但这不足以支持“首次”。当前 package 可以由以下邻域自然重构：

```text
direct On-TAL
+ history/memory
+ TrackFormer/MOTR-style persistent queries
+ lifecycle supervision
+ ETAD/SBP-style sparse backward
+ HH/IPW sampling
```

剩余可能达到论文级的 exact delta 只能是：

> 在不改变标准 On-TAD evaluation 和 causal inference 的情况下，提出一个能够保持完整 persistent numerical state、明确界定 recurrent gradient estimand、显著降低实际训练资源、并在 unseen multi-checkpoint fidelity gate 和 On-TAD effectiveness 上成立的方法。

当前代码和证据尚未实现这项 delta。

---

# L. Strongest Rejection

## L.1 Senior-PC rejection

> **Reject.** 当前 Full PETAL 尚不是一项被证据支持的方法贡献，而是一个完成度较高的 causal persistent detector 工程包。其主要组件——online action-instance localization、long-term memory、persistent queries、birth/identity lifecycle、sparse backward、importance sampling——均有直接或强邻近先例。当前唯一意图解决训练成本的 CRS-EPS surrogate 已被自身预注册 fidelity gate 终止：三项非 fallback 样本不能重建 chronological state，梯度 cosine 最低仅 `0.2831`，连续 state cosine 最低仅 `0.1028`，且离散 lifecycle 三次分叉。相对 fixed/reset 的平均优势不能覆盖绝对失败。
>
> 代码当前只训练 cached features，不能支持 end-to-end visual adaptation。仓库没有 fixed-step resource profile、没有 Q2 effectiveness result、没有 latency-quality frontier、没有 multi-seed formal experiment，也没有 raw-video evidence。四个 random-init G0 samples 足以终止冻结 surrogate，却不足以建立一个普遍的负理论；因此项目目前既没有正向方法结果，也没有足够一般化的负结果。
>
> 即使未来修复训练 surrogate，`persistent slots + causal model + lifecycle` 本身仍很容易被 MATR/HAT/TrackFormer/MOTR/TSA 和标准 recurrent training literature 重构。除非作者能给出一个 task-specific、state-faithful、成本可测、可证伪且在完整 chronological On-TAD 上有效的新技术核心，否则论文 novelty 仍不足。

## L.2 真正能够回答拒稿的证据

| Rejection point             | 必须提供的证据                                                                                                     |
| --------------------------- | ----------------------------------------------------------------------------------------------------------- |
| 当前 surrogate 不忠实            | 新方法在 unseen holdout、预注册 multi-checkpoint G0 上同时通过 loss、full-gradient、continuous 和 exact discrete margins    |
| Full PETAL 只是组件组合           | 对 MATR/HAT/ActionSwitch、TrackFormer/MOTR、ETAD/SBP、ARTBP/UORO 的 exact-delta 表和不可被替代的核心机制                     |
| Q2 没有 effectiveness         | full chronological cached route 上 fixed/rematch 的预注册 multi-seed effectiveness、identity 和 non-inferiority 结果 |
| 成本 claim 无证据                | 同硬件、同数据、同更新语义下的 tokens、backward tokens、state replay、memory、wall time、GPU-hours 和 ESS                        |
| cached route 过度主张           | 所有当前 claim 明确限定为 cached features；raw-video claim 必须有视觉参数梯度、更新和 raw-frame cost evidence                      |
| G0 scale 太局部                | gold-trained init/early/mid/late checkpoint strata，加 unseen samples 和 family-wise kill rule                 |
| 缺少 latency-quality frontier | 完整 chronological inference 下 latency、memory、emission delay、mAP/recall/identity 的 matched frontier           |
| 结果规模不足                      | 预注册 formal multi-seed、full reporting population、confidence intervals 和 multiple-comparison rule             |

更强叙事、更多模块或把 task 改成 OAD/VideoQA/zero-shot 都不能回答上述拒稿。

---

# M. Unknown Register and Author Questions

以下问题无法通过当前公开代码、HEAD dossier 和 primary sources回答，而且不同答案会改变 Round 2 路线选择。

| ID     | 问题                                                                                                           | 为什么重要                                                                                                                   | 不同答案如何改变 decision                                                                                                   |
| ------ | ------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **U1** | 能否提供原始 signed G0 bundle，至少包括 `audit.json`、12 raw rows、selection、margins 和 manifest？                          | 需要确定每个样本 exact replay range、驱动 `0.8021` 的 loss component、具体 discrete mismatch field，以及 first lifecycle divergence。      | 若独立重算不一致，进入 evidence-invalidity investigation；若一致，进一步巩固 structural diagnosis。当前 terminal KILL 在提供前仍保持有效。            |
| **U2** | 四个 G0 samples 的预注册 selection rule、strata 和 stress rationale 是什么？                                             | 当前只知道 IDs 和 outcome，不知道它们是否按 replay ratio、birth extension、overlap、duration 或 lifecycle complexity 分层。                   | 影响未来 holdout strata 和代表性判断；不改变当前 frozen gate。                                                                       |
| **U3** | 失败 arms 是否出现 `slot_exhaustion`，以及各样本的 `birth_assignments`、`canonical_lifecycle`、`endpoint_slot_trace` 分别是什么？ | 这决定 H4 是 threshold-only、canonical ownership divergence，还是 capacity violation。                                           | capacity failure 会要求新路线首先改变 state/capacity contract；纯 threshold bifurcation则优先要求 exact boundary-state preservation。 |
| **U4** | 当前正式 surrogate 原计划是否确实从 `681699…` 对应的 deterministic initialization 开始训练？                                     | 若是，G0 直接证明 first update 不忠实；若正式计划实际从其他 pretrained checkpoint 开始，则 preregistered checkpoint policy 与 intended route 不匹配。 | 前者强化 structural kill；后者不挽救旧 G0，但要求未来从真实起点重新建立全链。                                                                    |
| **U5** | full chronological cached Q2 是否已有可复用的 gold checkpoint trajectory，以及生成 init/early/mid/late checkpoints 的实际成本？ | 未来 multi-checkpoint fidelity gate 和 R0/R1/R3 成本比较需要该信息。                                                                 | 若 gold trajectory 已有，state-faithful confirmatory gate 成本较低；若没有，Round 2 应优先审计 R0/R3 成本，而不能假定 episode route必然更便宜。     |
| **U6** | 外部 signed B0、independent review 和 target-Linux recomputation bundles能否在本次审查环境中验签？                            | 当前对 chain 的判断依赖 reported hashes和代码对签名链的实现，而非本会话直接验签。                                                                    | 验签成功提高 evidence confidence；失败会把 `VALID_KILL` 降为 evidence-integrity dispute，但不能自动产生 PASS。                            |
| **U7** | complete chronological route 的第一份同硬件、同 workload-denominator profile 是否存在于 repo 外？                            | 当前只知道 formal/profile 未获授权，没有 R0 的实测成本。                                                                                  | 该值会直接影响 Round 2 中 R0、R1、R3 的排序，但不影响 G0 KILL。                                                                        |
| **U8** | raw feature cache 的 extractor checkpoint、causal preprocessing 和 frame-to-token provenance的原始 manifest是否可提供？  | 代码绑定其 hashes，但原始数据不在本次会话；这影响 cached Q2 的复现边界及未来 raw-video transition。                                                   | 仅影响 reproducibility 和 raw-video entry conditions，不改变当前 surrogate裁决。                                                 |

---

## Round 1 终止结论

```text
STRUCTURAL-KILL-CRS-EPS
KEEP-FULL-CHRONOLOGICAL-AS-PRIMARY
KEEP-FULL-PETAL-INFRASTRUCTURE-AS-ENGINEERING-SUBSTRATE
PIVOT-TO-STATE-FAITHFUL-COST-CONTROL = CANDIDATE_ONLY
PROFILE = BLOCK
FORMAL = BLOCK
NEW_GPU_HOURS = 0
```

当前最重要的路线纪律是：**不要把一次有效 KILL 降格为调参问题，也不要把整个 On-TAD/persistent-state 研究对象一并误杀。** 旧四样本只可用于零 GPU development diagnosis；任何 replacement method 都必须从新 commit、B0、独立 review、未暴露 holdout G0 和新的 terminal outcome 重新开始。

[1]: https://arxiv.org/abs/2211.04905 "https://arxiv.org/abs/2211.04905"
[2]: https://arxiv.org/abs/2408.02957 "https://arxiv.org/abs/2408.02957"
[3]: https://arxiv.org/abs/2408.06437 "https://arxiv.org/abs/2408.06437"
[4]: https://arxiv.org/abs/2407.12987 "https://arxiv.org/abs/2407.12987"
[5]: https://arxiv.org/abs/2607.00289 "https://arxiv.org/abs/2607.00289"
[6]: https://arxiv.org/abs/2504.20041 "https://arxiv.org/abs/2504.20041"
[7]: https://arxiv.org/abs/2306.07703 "https://arxiv.org/abs/2306.07703"
[8]: https://arxiv.org/abs/2101.02702 "https://arxiv.org/abs/2101.02702"
[9]: https://arxiv.org/abs/2105.03247 "https://arxiv.org/abs/2105.03247"
[10]: https://arxiv.org/abs/2606.13714 "https://arxiv.org/abs/2606.13714"
[11]: https://arxiv.org/abs/2205.07134 "https://arxiv.org/abs/2205.07134"
[12]: https://arxiv.org/abs/2203.16755 "https://arxiv.org/abs/2203.16755"
[13]: https://arxiv.org/abs/2204.01680 "https://arxiv.org/abs/2204.01680"
[14]: https://pytorch.org/blog/activation-checkpointing-techniques/ "https://pytorch.org/blog/activation-checkpointing-techniques/"
[15]: https://arxiv.org/abs/1705.08209 "https://arxiv.org/abs/1705.08209"
[16]: https://arxiv.org/abs/1702.05043 "https://arxiv.org/abs/1702.05043"
[17]: https://arxiv.org/abs/1604.06174 "https://arxiv.org/abs/1604.06174"
[18]: https://arxiv.org/abs/1803.00942 "https://arxiv.org/abs/1803.00942"
[19]: https://arxiv.org/abs/1511.03722 "https://arxiv.org/abs/1511.03722"
