本裁决严格沿用已冻结的两轮、fail-closed、零 GPU 前置门禁；旧四个 G0 样本只作开发诊断，不重新包装为新方法证据。

# N. Author-Answer Reconciliation

## N.1 Repository certificate

```text
Repository:
yuzbo/OpenTAD_OnlineTADClean_20260702

Branch:
codex/full-petal-implementation

Branch HEAD observed:
d5fc6a69466e6cb05429a3ebbc71832b755aa38c

Frozen scientific commit:
70df86ea3d38d70c658ae0ee9e04245d57b834d4
```

`codex/full-petal-implementation` 当前与 `d5fc6a6` 完全一致。`70df86e..d5fc6a6` ahead 2、behind 0；变化只包括 Round‑1 review、absorption、Prompt、wiki、decision register 和 `.gitattributes`。没有模型、dataset、sampling、detector、training engine、G0 gate、config 或 test 的科学实现变化。`d5fc6a6` 的提交信息也是 `docs: absorb CRS-EPS structural kill review`。

因此：

```text
SCIENTIFIC_IMPLEMENTATION = 70df86e
ROUND2_DOCUMENTATION_HEAD = d5fc6a6
HIDDEN_SCIENTIFIC_DIFF = NO
```

## N.2 U1–U8 disposition

| 项目                             | 状态           | 审查结论与影响                                                                                                                                                      |
| ------------------------------ | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **U1 signed G0 bundle**        | **RESOLVED** | 项目侧已经检查完整 bundle，terminal audit SHA‑256 为 `a081…109b`，12 rows 重现 KILL。它足以固定当前 route 的终局；原始 bundle 不公开仍限制第三方逐字节复现，但不再是路线裁决 blocker。                           |
| **U2 selection universe/rule** | **RESOLVED** | `160×4=640` candidates；一条/长度 quartile；最大化六类 stress union、proposal diversity、lexicographic tie；所有 stress flags 覆盖。它是强 falsification selection，不是总体失败率估计。      |
| **U3 exhaustion/divergence**   | **PARTIAL**  | 精确 counts 和 first-divergence bins 已知；但 exhaustion 的因果来源——K=4、false birth、refractory、阈值初始化或 lifecycle coupling——仍未分解。它单独阻塞 Q2 primary readiness。              |
| **U4 intended pilot start**    | **RESOLVED** | G0 checkpoint 是 seed‑705 deterministic initialization，正是正式 pilot 第一预定起点；因此不能用“只是任意 checkpoint”回避第一更新即不忠实的问题。                                                 |
| **U5 checkpoint trajectory**   | **RESOLVED** | 回答是“**不存在**”。不能把 init/early/mid/late trajectory 当成免费资产；新 G0 必须生成一个受限、CPU-only、结果不可见的 micro-trajectory。                                                       |
| **U6 B0/review/G0 validation** | **RESOLVED** | 项目本地及 target Linux 已独立重算；Round‑1 工具不可见是 reviewer environment boundary，不是证据缺失。                                                                                |
| **U7 matched cost profile**    | **PARTIAL**  | 已确认不存在同硬件、同 denominator profile；因此真实成本排序仍 OPEN。任何路线只能以 symbolic/token bound 比较，不能声称实际加速。                                                                     |
| **U8 data/cache identity**     | **PARTIAL**  | annotation、split、data identity 和 cache manifest 已 hash-bound，足以支持当前 G0；raw arrays 与 extractor artifacts 不在 bundle，继续阻塞严格第三方重建、raw-video causality 和 Stage 2。 |

这些状态与本地 absorption 记录一致：旧 route 的 structural kill 已关闭，而 Q2 capacity、cost 与 checkpoint trajectory 仍未关闭。

## N.3 Round‑2 starting facts

```text
CURRENT_G0 = valid and terminal
CURRENT_EMPTY_STATE_DYNAMIC_BIRTH = dead
ALL_SELECTIVE/EVENT-CENTRIC_TRAINING = not yet dead
CHRONOLOGICAL_Q2 = reference candidate, not primary
PROFILE = blocked
FORMAL = blocked
GPU_HOURS_BEFORE_NEW_G0_PASS = 0
```

旧 route 在四个样本中三个违反绝对 fidelity，最低 gradient cosine `0.2831`、最低 continuous-state cosine `0.1028`，而唯一完整回退到 video start 的 positive control 精确匹配。这一组合支持 omitted chronological state 是真实根因，不是 paired RNG 或 metric 普遍故障。

---

# O. Q2 Capacity/Lifecycle Verdict

## O.1 裁决

```text
Q2_CAPACITY_GATE = BLOCK
Q2_STATUS = GOLD_REFERENCE_CANDIDATE
```

当前证据**不能**判定“四槽在 THUMOS14 上结构性不足”，但已经足以判定：

> 当前 `K=4 + random-init logits + 0.5 birth/alive/end thresholds + refractory=2 + runtime-free-constrained GT birth assignment` 不具备 primary training readiness。

两个 `video_start_full` controls 分别发生 exhaustion 1 和 2。由于这两条 control 已从视频起点保持完整 numerical state，exhaustion 不是局部 replay 丢历史导致的；它是当前 Q2 detector/lifecycle/training-capacity 合同本身的独立失败。

## O.2 代码级因果链

当前配置：

```text
num_slots = 4
birth_threshold = 0.5
alive_threshold = 0.5
end_threshold = 0.5
refractory_steps = 2
```

并且 checkpoint 是随机初始化，而不是经过 lifecycle calibration 的模型。

每个时刻：

1. 训练器先从**模型当前 runtime state**选出 `SLOT_FREE` slots。
2. `head.step` 产生 birth/alive/end/class logits。
3. GT supervision 的 birth assignment 只能从：

   ```text
   runtime FREE
   ∩ canonical unoccupied
   ```

   中选择。
4. 没有得到 slot 的 GT birth 被计为 `exhausted`。
5. 该 instance 仍被加入 `born_instance_ids`，以后不会再次得到 birth allocation。
6. runtime `decode_step` 又可能根据随机 logits 创建 false active slot，提前结束、释放或进入 refractory。

推理 lifecycle 进一步规定：

* false birth 会把 free slot 变为 active；
* end crossing 会占用 refractory 两步；
* refractory 结束后，只有 alive 和 birth 都低于阈值才重新 free；
* predicted early free 和 false end 都可能改变下一次 GT birth 的候选集合。

所以真实 capacity demand 不是 annotation concurrency：

[
K_{\text{needed}}(t)
\neq
N_{\text{GT-active}}(t).
]

更准确地说：

[
K_{\text{needed}}(t)
====================

N_{\text{canonical occupied}}
+
N_{\text{false runtime active}}
+
N_{\text{refractory but canonically free}}
+
N_{\text{same-bin birth reserve}}.
]

`max GT concurrency = 2` 只约束第一项，不能证明四槽充分。

## O.3 当前最严重的实现不对称

CRS group 路径在 `slot_exhaustion != 0` 时会 rollback 并抛出 scientific failure。

但现有 full-chronological non-CRS 路径只让 detector 在 audit 中记录：

```text
slot_exhaustion
```

没有等价的 full-route fail-closed gate。

这意味着当前 Q2 “gold/reference”在工程上可能继续优化一个已经遗漏 GT birth 的 trajectory。它可以作为诊断 control，但还不能作为 accepted primary trainer。

## O.4 零 GPU capacity/lifecycle audit

首个新 commit 应只增加审计，不改变 slots、阈值或 binding。

### Audit A：annotation-only demand

对完整 fit-core 160 videos 逐 bin 计算：

```text
GT active count
GT births
GT ends
simultaneous birth/end
same-class concurrency
canonical occupancy
canonical occupancy + declared refractory reserve
minimum oracle-free K
```

这回答“即使 runtime controller 完美，K=4 是否结构性不足”。

### Audit B：真实 chronological controller replay

对 seed `705,706,707` 的 deterministic init checkpoints，在 CPU 上完整 replay 160 videos，记录每个 bin：

```text
birth/alive/end logits
runtime status before/after
canonical ownership
refractory
available slots
GT births
birth assignments
exhausted IDs
emissions
false-active count
```

每个 exhaustion 必须分解为：

```text
canonical occupancy
false ACTIVE occupancy
REFRACTORY occupancy
same-bin non-reusability
true K insufficiency
```

### Audit C：同 logits counterfactual controller replay

这是一个特别高杠杆的零 GPU诊断。当前 `head.step` 的 query/memory 更新不读取 `slot_status`、`refractory`、`start_frames`、`peak scores` 或 canonical maps；这些变量只进入后续 lifecycle decode 和 training assignment。

因此，在固定 checkpoint、features 和 RNG 下，可以只计算一次 logits，然后**精确**重放：

```text
actual: thresholds 0.5/0.5/0.5, refractory=2
refractory=0 diagnostic
birth/end prior-bias diagnostic
threshold sensitivity grid
canonical-only capacity upper-bound diagnostic
K = 2,3,4,5,... diagnostic
```

这不是合法训练方法，而是 root-cause decomposition；不得把 privileged canonical-only controller带进正式训练。

### Audit D：decision rule

| 发现                                                 | 允许的解释/行动                                                                               |
| -------------------------------------------------- | -------------------------------------------------------------------------------------- |
| canonical-only demand > 4                          | K=4 结构性不足；从完整 fit-core demand 推导最小 K，再在 unseen capacity holdout 验证。                    |
| canonical-only demand ≤4，但 actual K4 exhaustion >0 | 不是 annotation capacity；优先修 lifecycle initialization/gating，不得先扩大到16掩盖 false occupancy。 |
| removing refractory alone closes exhaustion        | refractory policy 是主因；任何变更必须共享于 fixed/rematch，并经过全新 G0。                                |
| prior-bias initialization closes exhaustion        | 可选择 outcome-blind lifecycle-prior initialization；不得用 Q2 mAP 选 bias。                    |
| 所有合理 shared policies仍 exhaustion                   | 当前 persistent lifecycle contract 无 primary readiness，停止 Q2。                            |

## O.5 Q2 晋级 primary 的必要证据

Q2 只能在全部满足后从 `GOLD_REFERENCE_CANDIDATE` 升为 `PRIMARY`：

1. 完整 fit-core capacity census 闭包。
2. 所选 K、threshold/bias 和 refractory policy 在任何 Q2效果可见前冻结。
3. fixed/rematch 使用完全相同的 capacity/lifecycle policy。
4. full-chronological engine 对任何 exhaustion fail closed。
5. 三个 init seeds 和新 G0 checkpoint strata均为零 exhaustion。
6. 不使用 GT active-slot、GT query 或 privileged runtime initialization。
7. 新 state-faithful route通过 unseen G0。
8. fixed-step profile证明资源可接受。
9. 一粒种子 mechanism kill 未出现 recall/delay collapse。

当前不满足第1、2、4、5、7、8、9项。

---

# P. Route Portfolio

评分采用 1–5，5为最好；“实现风险”一栏的5表示风险最低。

| Route                                       | State fidelity |                             Gradient validity | Cost potential | 实现风险 | Novelty | G0 testability | 裁决                             |
| ------------------------------------------- | -------------: | --------------------------------------------: | -------------: | ---: | ------: | -------------: | ------------------------------ |
| **R0 full chronological**                   |              5 |                 5，相对 chrono‑64 estimand exact |              1 |    4 |       1 |              5 | reference candidate；capacity未过 |
| **R1 full forward + selective backward**    |              5 | 4：HH-unbiased、高方差；masked implementation可exact |              3 |    3 |       2 |              5 | **唯一实施候选**                     |
| **R2 exact recomputation**                  |              5 |                                             5 |              2 |    3 |       1 |              5 | 被R1支配                          |
| **R3 runtime checkpoints**                  |              2 |                                             2 |          4（名义） |    1 |       1 |              3 | stale-state结构风险，NO-GO          |
| **R4 activation checkpoint/SBP/ARTBP/UORO** |              3 |                                             2 |              2 |    1 |       1 |              2 | 异质方法集合，非本轮单一路线                 |
| **R5 abandon surrogate/use chronological**  |              5 |                                             5 |              1 |    4 |       0 |              5 | honest fallback，不是高效路线         |

## P.1 Route-specific findings

### R0

R0是当前唯一完整 numerical-state route：

```text
all causal forward tokens
all token losses
global 64-bin TBPTT detach
one optimizer event per video
```

它对当前 locked estimand是 exact reference，但它不解决成本，也没有训练方法新颖性，而且当前 capacity gate未过。

### R1

R1保持每个 token 的完整 chronological forward、hard lifecycle、canonical ownership 和 RNG 消耗，只随机选择哪些 token loss进入 backward。它不尝试重建 state，所以旧 G0 的根因被结构性删除。

它可能减少：

```text
temporal backward FLOPs
activation retention
loss construction on unselected bins
```

它不减少：

```text
temporal forward tokens
Python sequential unroll
state/lifecycle update
feature loading
```

ETAD 已经采用“所有 snippets sequential forward，只对一小部分 encoder gradients backward”的总体效率思想，因此 R1最多是 On‑TAD persistent-state 下的必要 enabling protocol，不能先验作为算法创新。([arXiv][1])

### R2

R2先完整 no-grad forward，保存同参数版本的合法 block-start state，再对 selected blocks重算并反传。

若 selected positions 已由 immutable manifest 预先知道，则：

```text
R1 forward = T
R2 forward = T + selected_recompute_tokens
```

二者能得到同一 selected gradient，R2只有额外 forward，不具有优势。只有 selection 依赖第一次模型输出时R2才有必要，但那会引入 model-dependent proposal、概率修正和新的 one-factor问题。

### R3

checkpoint (H_c(\theta_k)) 在参数更新后用于 (\theta_{k+1})：

[
H_c(\theta_k)
\neq
H_c(\theta_{k+1}).
]

把 checkpoint 与参数版本 hash 绑定只能**检测 stale**，不能使 stale state 正确。若每次 optimizer update 都使所有 checkpoint失效并重新生成，就退化为R2或R0。

此外它仍丢失 checkpoint 之前的 Jacobian。故R3 NO-GO。

### R4

* Activation checkpointing 保存 exact gradient、降低 activation memory，但通过额外 forward recomputation换内存，不减少原始 forward或backward目标。([arXiv][2])
* SBP保留所有 forward paths，随机删除部分 layer backward paths；其收益主要是显存和有限速度，梯度不再是R0 exact estimator。([arXiv][3])
* ARTBP通过随机 truncation和补偿权重，对**full-BPTT**目标构造无偏估计；当前项目锁定的是global 64-bin TBPTT control，且含hard lifecycle branches，因此它改变了研究 estimand。([arXiv][4])
* UORO是RTRL的unbiased online approximation，但其rank-one估计存在显著variance问题；对attention、multi-slot和离散 lifecycle引入复杂度过高。([arXiv][5])

### R5

R5是治理上承认“只训练R0，不再声称高效 surrogate”。它比不忠实的 sampler诚实，但当前 capacity和cost仍未过，也无法提供训练方法贡献。

## The Strongest Case for Killing CRS-EPS-Enabled Full PETAL

最强拒稿意见是：Full PETAL当前仍像一组已知技术的工程拼装，而不是一个不可替代的科学机制。persistent query、birth/death、trajectory identity来自TrackFormer/MOTR类思路；causal video features和stateful streaming已经有直接先例；历史记忆、overlap和same-class动作又有MATR、HAT、ActionSwitch等直接On‑TAL竞争；高效视频反传则已被ETAD、SBP、activation checkpointing以及各类adapter路线覆盖。旧CRS‑EPS更加没有存活空间：它以annotation-guided event windows、HH/IPW和局部replay为核心，但真实persistent detector的hidden state、false births、refractory、peak scores、canonical ownership和历史Jacobian并不是sampling weight可以恢复的随机变量。terminal G0已经用三种相互独立的证据——loss、all-trainable-parameter gradient、continuous/discrete state——证明了这一点。继续微调192 context或增加少量birth history只会把已失败的充分统计量假设重新命名。

即使R1通过新G0，它也不自动成为论文贡献。它本质上是full chronological forward加稀疏loss/backward，和ETAD selective gradient、SBP incomplete backward、普通importance sampling具有明显重构关系。它能够成为一个诚实的cost-control substrate，但必须通过实际profile证明backward确实是瓶颈；如果主要成本来自完整forward、Python `head.step`、data wait或lifecycle bookkeeping，R1可能几乎不节省GPU-hours。当前Q2本身还存在更基础的问题：随机初始化下hard 0.5 thresholds和refractory会让模型预测状态决定GT birth是否有候选slot，两个完整gold controls已经出现slot exhaustion；而full chronological engine尚未等价fail closed。若通过增加slot数来隐藏false occupancy，研究对象将从fixed binding偷换为capacity engineering。

最后，即使fixed最终优于rematch，也可能只说明固定label assignment降低了训练噪声，而不说明模型学到了论文级persistent identity。duplicate减少可能来自更少birth或更少emission，并伴随recall、FN或completion delay恶化。THUMOS14规模小、locked‑211与canonical‑213仍有reporting边界，custom identity/online指标不能替代standard tIoU mAP。若FRESH、Temporal TrackFormer、简单GRU control或ActionSwitch-style lifecycle可以重构收益，那么positive Q2仍不足以支持Full PETAL；它最多是一条THUMOS-specific supervision-policy observation。故本轮选择R1不是对Full PETAL的认可，而是给唯一一个state-faithful、可被快速杀死的训练协议一次实现机会。

---

# Q. Recommended Route

## 唯一选择

```text
SELECTED_ROUTE = R1
ROUTE_NAME = Chronological State-Faithful Selective Backward
             (CSFSB)
DECISION = GO-FOR-IMPLEMENTATION
```

“GO”仅授权：

```text
CPU implementation
CPU tests
capacity audit
B0
independent review
unseen CPU G0
```

不授权 profile、effectiveness training、formal training或Stage 2。

## Q.1 为什么只选R1

R1同时满足：

1. **state faithful**：所有 token 都按真实 chronological order forward。
2. **lifecycle faithful**：所有 birth/alive/end、refractory、canonical assignment 都实际执行。
3. **无 stale state**：不使用跨参数版本 state cache。
4. **同一 chrono‑64 estimand**：selected loss 的图从原global 64-bin block start构造。
5. **可无偏**：HH/IPW只抽取loss contributions，不再尝试估计hidden state。
6. **一视频一optimizer event**：与R0 scheduler/optimizer semantics一致。
7. **可被严格G0验证**：同一 selected mask下，R1与all-grad masked gold应在state、logits、loss和gradient上闭合。
8. **潜在成本收益**：只减少backward和activation retention，收益范围明确、不会冒充forward savings。

## Q.2 为什么不是并行实施R2/R3/R4

* R2在selection预知时被R1严格支配。
* R3 parameter-stale，刷新后退化为R2。
* R4各子方法改变不同资源或gradient estimand，不是一个可归因主路线。
* Activation checkpointing可在R1因显存失败后作为独立辅助工程，但不进入首版科学协议。

## Q.3 前置 capacity condition

R1的选择不覆盖Q2 capacity失败。实际顺序必须是：

```text
capacity/lifecycle audit
→ freeze shared valid lifecycle contract
→ implement R1
→ new G0
```

若capacity audit不能找到一个无GT taint、零exhaustion、fixed/rematch共享的合同，则：

```text
R1_IMPLEMENTATION_STOPS
Q2_STATUS=INVALID
```

---

# R. Exact Method Design

## R.1 核心变化

旧 route：

```text
sample episode
→ reset to empty state
→ replay local prefix
→ approximate chronological state
```

R1：

```text
sample only loss locations
→ always scan the complete video chronologically
→ keep exact runtime/supervision state
→ build autograd only where selected losses need it
```

禁止继续使用：

```text
replay_range
dynamic_birth
fixed_192 as trainer
independent_episode_state
is_video_start=True per sampled draw
```

这些字段只保留在old-route negative artifacts中。

## R.2 Sampling manifest

保留当前 outcome-blind proposal：

```text
M = 4 draws/video/epoch
H = 8 supervised bins/window
mixture:
  uniform 0.40
  start   0.15
  end     0.20
  ongoing 0.10
  hardbg  0.15
```

但manifest不再生成episode replay。它只生成每个bin的exposure multiplicity：

[
m_{v,t}
=======

\sum_{r=1}^{M}
\mathbf 1[t\in S(A_{v,r})].
]

单draw coverage仍为：

[
\rho_{v,t}
==========

\Pr_{A\sim q_v}[t\in S(A)].
]

最终 per-bin coefficient：

[
a_{v,t}
=======

\frac{m_{v,t}}{M}
\frac{1/T_v}{\rho_{v,t}}.
]

重复windows不去重，因为 multiplicity 正是 Hansen–Hurwitz estimator的一部分。由于所有exposures现在共享同一个full chronological state，同一bin的重复loss可以合并为一个乘以 (m_{v,t}) 的loss计算。

Manifest必须包含：

```text
video_id
epoch
seed
M
mixture
draw components/anchors
rho_by_bin
multiplicity_by_bin
coefficient_by_bin
selected_unique_bins
selected_blocks
last_selected_bin_by_block
manifest SHA
data/config identity
```

不得包含 runtime state、GT slot、hidden checkpoint或future model output。

## R.3 State recurrence

连续state：

[
h_t=F_\theta(h_{t-1},x_t;\xi_t),
]

离散runtime lifecycle：

[
r_t=D(r_{t-1},z_t),
]

training-only canonical supervision：

[
c_t=C(c_{t-1},y_t,z_t).
]

其中：

* (x_t) 是 causal cached feature；
* (\xi_t) 是 dropout/RNG；
* (z_t) 是当前模型outputs；
* (y_t) 是prefix-observable training target；
* (r_t,c_t) 都在所有bins更新，不论该bin是否被采样监督。

## R.4 Detach contract

全局 block：

[
B_j=[64j,\min(64(j+1),T_v)).
]

在每个block入口：

[
h_{64j}\leftarrow\operatorname{stopgrad}(h_{64j}).
]

这与R0相同。

对block (B_j)，令最后一个有非零 (a_{v,t}) 的位置为 (u_j)。

* 若block没有selected bin：完整block `no_grad` forward。
* 若存在selected bin：

  * 从block start到 (u_j) 开图；
  * loss只在 (a_{v,t}>0) 的bin计算；
  * (u_j) 之后先detach state，再以 `no_grad` 运行block tail；
  * block末backward已累计到model gradients；
  * numerical state继续完整传入下一block。

这样：

* selected loss拥有与R0完全相同的64-bin Jacobian prefix；
* unselected tail没有无用graph；
* 下一block仍按R0 detach；
* numerical state不变。

## R.5 Loss

每bin仍使用现有：

```text
birth_loss
alive_loss
class_loss
start_loss
end_loss
```

不改变risk masks、class/start binding或component weights。

视频 sampled risk：

[
\widehat L_v
============

\sum_{t=1}^{T_v}
a_{v,t},
\ell_{v,t}.
]

不允许再除以：

```text
number of selected bins
sum of realized weights
number of selected blocks
```

因为这些操作分别会改变target或变成SNIPW。

## R.6 Optimizer boundary

```text
zero_grad at video start
full chronological selected-backward scan
accumulate gradients across all selected blocks
capacity and finite-gradient checks
one optimizer.step at video end
one scheduler.step at video end
transaction commit
```

因此optimizer/scheduler event仍是一视频一次，与R0完全匹配。

## R.7 RNG

同一 full video 必须消耗与R0相同的token顺序和dropout draws。

`torch.no_grad()`不能跳过forward，也不能跳过随机算子。G0必须证明：

```text
same checkpoint
same features
same RNG state
same full token sequence
→ same logits and final runtime state
```

## R.8 Rollback

任何以下事件导致整视频rollback：

```text
slot exhaustion
non-finite loss/gradient
manifest/hash mismatch
selected coefficient mismatch
RNG identity mismatch
mutable-buffer leakage
cross-video state leakage
optimizer/scaler/scheduler failure
transaction commit failure
```

---

# S. Mathematical Estimand

## S.1 R0 control

视频 (v) 有 (T_v) bins。R0目标：

[
L_v^{(0)}(\theta)
=================

\frac1{T_v}\sum_{t=1}^{T_v}
\ell_{v,t}
\bigl(\theta,h_t^{64}(\theta),r_t,c_t\bigr).
]

总体：

[
L^{(0)}(\theta)
===============

\frac1V\sum_v L_v^{(0)}(\theta).
]

其中 (h_t^{64}) 使用global 64-bin detach。它不是exact full-video BPTT。

对应gradient：

[
g_v^{(0)}
=========

\frac1{T_v}\sum_t g_{v,t}^{64}.
]

## S.2 R1 estimator

每视频采样 (M) 个windows：

[
A_{v,1},\ldots,A_{v,M}\sim q_v.
]

定义：

[
I_{v,t}^{(m)}
=============

\mathbf 1[t\in S(A_{v,m})],
\qquad
\rho_{v,t}
==========

\mathbb E[I_{v,t}^{(m)}].
]

系数：

[
a_{v,t}
=======

\frac1M
\sum_{m=1}^{M}
I_{v,t}^{(m)}
\frac{1/T_v}{\rho_{v,t}}.
]

估计量：

[
\widehat L_v^{(1)}
==================

\sum_t a_{v,t}
\ell_{v,t}
\bigl(\theta,h_t^{64}(\theta),r_t,c_t\bigr).
]

由于：

[
\mathbb E\left[\frac1M\sum_m I_{v,t}^{(m)}\right]
=================================================

\rho_{v,t},
]

有：

[
\mathbb E_A[a_{v,t}]
====================

\frac1{T_v},
]

从而：

[
\mathbb E_A[\widehat L_v^{(1)}]
===============================

L_v^{(0)}.
]

只要selected bin的graph从原64-bin block start开始：

[
\mathbb E_A[
\nabla_\theta\widehat L_v^{(1)}
]
=

g_v^{(0)}.
]

## S.3 分类

| 属性                  | R1 相对R0                                |
| ------------------- | -------------------------------------- |
| Numerical state     | **exact state-faithful**               |
| Discrete lifecycle  | **exact state-faithful**               |
| 每次 sampled gradient | 不等于full gradient                       |
| 期望gradient          | **unbiased relative to chrono‑64**     |
| Variance            | 高于R0                                   |
| Full-video BPTT     | 否                                      |
| Forward FLOPs       | 不降低                                    |
| Backward FLOPs      | 可能降低                                   |
| Sampling bias       | uncapped HH下无，前提是probability和state合同成立 |
| Capacity错误          | 完全不能修正                                 |

## S.4 Sampling correction能与不能做什么

可以校正：

```text
哪些 decision bins 的 loss 被抽中
event-heavy sampling 导致的 exposure imbalance
重复 window exposure
video-uniform / decision-bin-uniform target
```

不能校正：

```text
slot exhaustion
invalid runtime lifecycle
GT-tainted state
future-conditioned sampling
different RNG path
different 64-bin detach geometry
parameter-stale checkpoint
feature extractor future leakage
model-dependent selection without propensity
```

## S.5 Workload bounds

令：

[
J_v=\lceil T_v/64\rceil.
]

对第 (j) 个block，若无selected bin，(p_{v,j}=0)；否则 (p_{v,j}) 为从block start到最后selected bin的token数，(1\le p_{v,j}\le64)。

### R0

[
F_v^{R0}=T_v,
\qquad
B_v^{R0}=T_v.
]

### R1

[
F_v^{R1}=T_v,
\qquad
B_v^{R1}=\sum_{j=1}^{J_v}p_{v,j}
\le T_v.
]

### R2

[
F_v^{R2}=T_v+B_v^{R1},
\qquad
B_v^{R2}=B_v^{R1}.
]

所以在相同selection下：

[
F^{R1}<F^{R2}
]

除非 (B^{R1}=0)，而合法manifest不允许整个视频没有监督。

### Supervision

正常 (T_v\ge8)：

[
\text{repeated exposures}=M H=4\times8=32.
]

[
\text{unique selected bins}\le32.
]

### Optimizer

[
O_v^{R0}=O_v^{R1}=1.
]

### Activation memory

R0和R1的最大graph horizon均不超过64 bins：

[
M_{\text{activation}}^{R1}
\le
M_{\text{activation}}^{R0}
==========================

O(64\cdot \text{per-token graph state}).
]

实际peak可能相同；R1主要降低平均retained activation和backward workload，不能在profile前声称peak一定下降。

### Persistent state tensor payload

当前 (K=4,D=256,M_{\rm mem}=192)。浮点state至少包括：

[
KD+M_{\rm mem}D+2K
==================

50{,}184
]

个元素。

仅tensor lower bound约为：

```text
BF16: 100,464 bytes ≈ 98.1 KiB / stream
FP32: 200,832 bytes ≈ 196.1 KiB / stream
```

还不含：

```text
source_frames Python tuple
supervision maps/sets
audit rows
allocator overhead
ledger metadata
```

R1不新增persistent checkpoint storage。

R3若每64 bins存一份，则额外存储至少：

[
\lceil T_v/64\rceil
\times
S_{\text{state}},
]

并且仍有parameter-version staleness问题。

---

# T. Code-Level Delta

## T.1 P0 before any profile

| 文件                                            | 变更                                                                                          | Fail-closed invariant                      |
| --------------------------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------ |
| **新增** `opentad/utils/q2_capacity_audit.py`   | annotation demand、runtime occupancy、exhaustion attribution、counterfactual controller replay | 每个exhausted birth必须有唯一cause classification |
| **新增** `tools/audit_q2_capacity_lifecycle.py` | CPU-only全fit-core审计，输出hash-bound JSON                                                       | dirty repo、未知checkpoint、缺视频或重复视频立即失败       |
| **修改** `opentad/cores/train_engine.py`        | non-CRS full route也检查`last_episode_audit.slot_exhaustion`                                   | R0/R1/CRS任一exhaustion均在optimizer前rollback  |
| **新增** `tests/test_q2_capacity_lifecycle.py`  | false birth、refractory、simultaneous end/birth、K不足等                                          | privileged diagnostic不得进入trainer           |
| **修改** launch allowlist/gate                  | `70df86e` killed CRS config永久不能授权profile                                                    | old KILL artifact不能被新route名字绕过             |

首个commit不改：

```text
num_slots
thresholds
refractory
binding mode
loss
optimizer
```

它只生成因果证据。

## T.2 P1 selected-route implementation

| 文件                                                               | 变更                                                                                    |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **新增** `opentad/utils/state_faithful_sampling.py`                | 复用proposal概率数学，但只生成multiplicity/coefficient manifest；禁止replay geometry                |
| **新增** `opentad/datasets/state_faithful_selective_feature.py`    | 继承完整`StreamingFeatureDataset`，仍枚举全视频chunks，只给每chunk附coefficient mask                  |
| **修改** `opentad/models/detectors/persistent_trajectory_ontad.py` | 新增`train_state_faithful_selective_chunk`；所有bins更新state/lifecycle，selected bins才计算loss |
| **新增** `opentad/cores/state_faithful_train_engine.py`            | block-aware no-grad/grad scan，一视频一optimizer event，无realized-weight renormalization    |
| **新增** `opentad/utils/state_faithful_paired_audit.py`            | masked all-grad gold vs selective-backward twin trace                                 |
| **新增** `tools/build_state_faithful_manifest.py`                  | immutable outcome-blind manifest                                                      |
| **新增** `tools/audit_state_faithful_manifest.py`                  | support、rho、multiplicity、coefficient、ESS、selected-block census                        |
| **新增** `tools/preregister_state_faithful_g0.py`                  | samples、checkpoint strata、margins、config、commit、hashes                                |
| **新增** `tools/run_state_faithful_g0.py`                          | CPU-only terminal PASS/KILL                                                           |
| **新增** configs                                                   | `thumos_pes_q2_sfsb_base.py`, `_fixed.py`, `_rematch.py`                              |
| **修改** `opentad/utils/full_petal_launch.py`                      | 新route只接受新schema的terminal G0 PASS                                                     |
| **修改** `tools/check_full_petal_results.py`                       | capacity、G0、workload、standard metrics、population gate                                 |

## T.3 Detector pseudocode

```python
def train_state_faithful_selective_chunk(
    features,
    schedule,
    state,
    supervision,
    global_start,
    coefficients,
):
    # state already comes from the true preceding chronological chunk.
    state = detach_head_state(state)  # exact global 64-bin boundary

    selected = nonzero_indices(coefficients)
    last_selected = max(selected) if selected else None
    block_cost = None

    for local_idx in range(len(features)):
        global_bin = global_start + local_idx

        if last_selected is None:
            grad_enabled = False
        else:
            grad_enabled = local_idx <= last_selected

        with torch.set_grad_enabled(grad_enabled):
            outputs, state = head.step(features[local_idx], state, source_frame)
            transition = supervision.transition(
                schedule[local_idx],
                cost_provider(outputs),
                available_slots=runtime_free_slots(state_before_decode),
            )

            if coefficients[local_idx] > 0:
                raw = step_losses(outputs, transition)
                term = coefficients[local_idx] * weighted_component_sum(raw)
                block_cost = term if block_cost is None else block_cost + term

            _, state = head.decode_step(outputs, state, source_frame)

        if local_idx == last_selected:
            state = detach_head_state(state)

    return block_cost, state, supervision, audit
```

注意：真实实现必须保持当前“available slots在该step decode前确定”的语义，除非capacity audit另行冻结一个共享修订。不能在实现R1时顺便改变lifecycle timing。

## T.4 Engine pseudocode

```python
for video in chronological_video_order:
    zero_grad()
    snapshot_all_mutable_state()
    selected_block_count = manifest.selected_block_count(video)

    for chunk in full_video_chunks(video):
        cost, state, supervision, audit = selective_chunk(...)
        assert audit.slot_exhaustion == 0

        if cost is not None:
            backward(cost)  # coefficient already includes 1/M and 1/T/rho

    assert at_least_one_selected_bin
    assert gradients_finite
    optimizer.step()
    scheduler.step()
    commit_transaction()
```

禁止：

```python
gradient /= selected_block_count
gradient /= sum(realized_coefficients)
gradient /= unique_selected_bins
```

## T.5 Old code disposition

保留但锁死：

```text
CrsEpsFeatureDataset
dynamic_birth
fixed_192
reset
old G0 runner/gate
70df86e negative artifacts
```

用途：

```text
negative result
development regression
adversarial baseline
```

从任何新 launch allowlist移除；不得删除历史证据，也不得把它改名成R1。

## T.6 P2 only if positive

Raw-video LoRA、activation checkpointing、多rank、resume、full tower均 defer。R1 cached Q2通过三粒种子前不实现。

---

# U. Test Matrix

| 类别          | 必须测试的case                                     | 预期                                     |
| ----------- | --------------------------------------------- | -------------------------------------- |
| Capacity    | GT concurrency≤K、runtime false births占满K      | 精确归因false ACTIVE                       |
| Capacity    | refractory占满但canonical有空位                     | 精确归因REFRACTORY                         |
| Capacity    | actual canonical demand>K                     | 归因structural K insufficiency           |
| Capacity    | 同bin old end/new birth                        | 验证当前不可同bin复用语义                         |
| Capacity    | full R0 exhaustion                            | optimizer前rollback                     |
| Capacity    | fixed/rematch共享capacity policy                | config exact equality                  |
| State       | all-grad masked gold vs R1                    | logits、continuous/discrete state exact |
| Gradient    | selected bin位于block第1位                        | graph只需1 token                         |
| Gradient    | selected bin位于block第64位                       | graph覆盖完整block                         |
| Gradient    | block无selected bin                            | 全block无grad但state exact                |
| Gradient    | selected后有tail                                | tail no-grad，最终state与R0 exact          |
| Gradient    | selected bins跨两个blocks                        | 每个global boundary正确detach              |
| Gradient    | all bins selected、coeff=1/T                   | gradient与R0 exact                      |
| Sampling    | exhaustive toy enumeration                    | (E[a_t]=1/T)                           |
| Sampling    | overlapping windows                           | multiplicity而非union weighting          |
| Sampling    | component empty                               | fallback uniform且support>0             |
| Sampling    | edge-shifted H=8 window                       | anchor始终在window内                       |
| Sampling    | short video (T<8)                             | 合法系数与完整支持                              |
| Denominator | realized weight sum≠1                         | 不做SNIPW renormalization                |
| RNG         | dropout=0.1，grad/no-grad切换                    | RNG tail、logits、state一致                |
| RNG         | CPU/CUDA RNG digest mismatch                  | fail closed                            |
| Staleness   | 注入旧parameter-version state checkpoint         | 明确拒绝                                   |
| GT taint    | `active_slot_ids`, future endpoint, EOF进入meta | fail closed                            |
| Leakage     | video B state接到video A                        | fail closed                            |
| Leakage     | supervision state跨video                       | fail closed                            |
| Buffer      | BatchNorm/自定义buffer在video内泄漏                  | rollback                               |
| Transaction | selected block backward失败                     | 恢复state/buffers/optimizer              |
| Transaction | optimizer/scaler/scheduler/commit失败           | 全事务恢复                                  |
| Numerical   | non-finite selected loss                      | 整视频invalid                             |
| Numerical   | unselected bin non-finite logits              | 仍必须fail；不能因无loss忽略                     |
| Lifecycle   | same-class overlap                            | identity和canonical ownership稳定         |
| Lifecycle   | action >192 bins                              | R1不受context truncation                 |
| Lifecycle   | old exposed four samples                      | 仅development regression，不计G0           |
| Manifest    | draw reorder/substitution/re-hash             | fail closed                            |
| Manifest    | commit/config/data identity漂移                 | fail closed                            |
| Workload    | forward/backward counters                     | 与实际graph范围精确相等                         |
| DDP         | world_size>1                                  | 明确unsupported并fail closed              |
| Reporting   | locked‑211/canonical‑213 mismatch             | formal result blocked                  |

特别要求：

> `state exact` 测试必须使用当前真实 `dropout=0.1`、`threshold=0.5`、`refractory=2` 配置的缩小模型，而不能继续只用测试中的 `threshold=1.1` 避开lifecycle触发。

现有CRS测试中的主要toy detector确实使用 `birth/alive/end_threshold=1.1` 和 `dropout=0`，所以它不能证明真实Q2初始化capacity安全。

---

# V. Preregistered Experiment Ladder

## V.0 Development-only diagnosis

允许使用旧四个样本做：

```text
capacity cause trace
state first-divergence debugging
R1 code regression
```

禁止：

```text
选择新margin
选择新sample
选择M/H
宣称confirmatory PASS
```

## V.1 Zero-GPU capacity audit

1. 完整160 fit-core annotation census。
2. seed 705/706/707 init checkpoints全chronological CPU replay。
3. actual/counterfactual lifecycle attribution。
4. 冻结：

   ```text
   K
   thresholds or prior-bias initialization
   refractory policy
   same-bin reuse rule
   exhaustion semantics
   ```
5. 若无法得到无GT taint且零exhaustion的shared policy，立即：

```text
Q2_STATUS=INVALID
SELECTED_ROUTE=NONE
```

## V.2 Micro-trajectory checkpoint generation

因为不存在正式trajectory，新G0需自己生成一个结果不可见的CPU-only trajectory。

### Preferred policy

* 从fit-core选择16个`trajectory-core` videos。
* selection只使用metadata/stress flags。
* 与旧四个G0 videos及新holdout videos完全不重叠。
* 对seeds `705,706,707`分别按R0 chronological CPU训练一遍。
* 冻结checkpoint：

```text
init  = 0 optimizer events
early = 4
mid   = 8
late  = 16
```

这产生12个parameter states。

`late`只能称：

```text
micro-trajectory late
```

不能冒充正式12-epoch late checkpoint。

### Bounded fallback

若在任何fidelity output计算前，预注册CPU resource cap已被触发：

```text
seed 705: 0,4,8,16
seed 706: init only
seed 707: init only
```

fallback条件和CPU cap必须在selection artifact中预先签署；不能看结果后减少checkpoint strata。

## V.3 New holdout selection

从：

```text
remaining fit-core videos × 4 draws
```

中选择8个unseen samples：

* 每length quartile 2个；
* 排除旧四个samples及trajectory-core videos；
* 最大化以下flags的union：

  ```text
  detach crossing
  dynamic-extension geometry
  action >192
  overlap
  same-class repeat
  true-left-censor geometry
  capacity pressure
  refractory-adjacent birth
  ```
* 再最大化proposal-component diversity；
* 最后lexicographic tie-break。

即使R1不再使用replay，这些stress flags仍用于覆盖最难的state/lifecycle区域。

## V.4 G0 comparisons

### G0-A：implementation exactness

同一checkpoint、video、RNG、coefficients：

```text
Gold:
full chronological forward
all selected blocks fully graphed
loss only on selected bins

Candidate:
R1 no-grad on unselected blocks/tails
same selected losses
```

要求：

```text
all logits allclose within frozen FP32 tolerance
continuous runtime allclose
discrete runtime exact
canonical lifecycle exact
birth/endpoint ownership exact
selected losses relative error <= 1e-7
all-trainable gradient cosine >= 0.999999
relative gradient L2 error <= 1e-6
gradient sign agreement >= 0.999999
```

### G0-B：estimator adequacy

比较：

```text
R0 all-bin chrono-64 gradient
vs
R1 M=4 HH selected gradient
```

每个checkpoint state把8个holdout videos的gradient按video-uniform规则聚合后计算：

```text
gradient cosine >= 0.90
sign agreement >= 0.90
relative loss error <= 0.10
finite norm ratio in preregistered interval
```

这些是resource/fidelity safety margins，不是论文显著性标准。

### G0-C：capacity

任何seed、checkpoint、sample、gold或candidate出现：

```text
slot_exhaustion > 0
```

立即terminal KILL。即使两臂同时exhaust也不能相互抵消。

### Family-wise rule

```text
PASS iff:
  every hard invariant passes
  AND every checkpoint stratum passes aggregate estimator margins
  AND zero capacity violation
  AND no taint/RNG/hash/rollback violation
```

一项失败即terminal KILL。旧sample不参与判定。

## V.5 Evidence sequence

```text
new implementation commit
→ local B0
→ target-Linux B0
→ same independent reviewer PASS / NEXT_GATE=G0
→ signed selection/checkpoints/margins/config/hash
→ unseen CPU G0
```

只有terminal G0 PASS后：

```text
PROFILE may be requested
```

## V.6 Fixed-step profile

R0与R1使用：

```text
same 8 warmup videos
same 32 measured videos
same order
same features
same checkpoint
same BF16
same optimizer event count
```

这里optimizer event终于都是“完整视频一次”，所以可以作为一个共同轴，但仍必须报告全部physical denominators。

Profile只证明资源，不证明quality。

## V.7 One-seed mechanism kill

G0和profile均PASS后：

```text
seed 705
R1 fixed
vs
R1 rematch
```

完整chronological one-token evaluation，standard tIoU mAP、identity和safety metrics。

STOP if：

* fixed不优；
* gain不在identity errors；
* recall/FN/end delay恶化；
* exhaustion；
  -超预算。

## V.8 Novelty-killer controls

只有一粒种子fixed/rematch通过后，再运行：

```text
simple GRU recurrent control
FRESH
faithful Temporal TrackFormer
```

仍为一粒种子screen。

## V.9 Scientific stages

```text
3 paired seeds = route kill/low-cost confirmation
5 paired seeds + dense/overlap dataset = minimum retained paper claim
```

三粒种子前不得进入raw-video LoRA。

---

# W. Cost Table

## W.1 Per-video theoretical workload

| Route                    |       Forward tokens | Recomputed forward |      Backward graph tokens |    Supervised exposures | Optimizer events |   Peak graph horizon |
| ------------------------ | -------------------: | -----------------: | -------------------------: | ----------------------: | ---------------: | -------------------: |
| R0                       |                (T_v) |                  0 |                      (T_v) |                   (T_v) |                1 |                   64 |
| **R1**                   |                (T_v) |                  0 | (P_v=\sum_jp_{v,j}\le T_v) | 32 repeated, ≤32 unique |                1 |                  ≤64 |
| R2                       |                (T_v) |              (P_v) |                      (P_v) |                      32 |                1 |                  ≤64 |
| R3                       |                 名义可低 |              取决于刷新 |                  取决于suffix |                      32 |                1 | bounded，但state stale |
| Activation checkpoint R4 | (>T_v) due recompute |           positive |                      (T_v) |                   (T_v) |                1 |         lower memory |
| R5                       |                (T_v) |                  0 |                      (T_v) |                   (T_v) |                1 |                   64 |

这里：

```text
T_v = full video tokens
P_v = sum of selected-block prefixes
```

R1实际cost improvement只有在：

[
P_v/T_v
]

足够低且backward在wall time中占有足够比例时才成立。当前没有实测数据。

## W.2 Mandatory denominators

每个profile artifact必须同时记录：

```text
videos and exact video IDs
temporal forward tokens
temporal backward graph tokens
selected blocks
repeated supervised exposures
unique supervised bins
HH weight sum / squared sum / ESS
optimizer events
data wait seconds
controller/supervision CPU seconds
forward CUDA seconds
backward CUDA seconds
optimizer seconds
wall seconds
peak VRAM
GPU-hours
precision and hardware identity
```

禁止只报告：

```text
optimizer events/s
episodes/s
selected bins/s
```

## W.3 Resource caps

| Stage                             |          GPU-hour cap | 当前权限                         |
| --------------------------------- | --------------------: | ---------------------------- |
| Capacity/lifecycle audit          |             **0 GPU** | ALLOWED CPU                  |
| R1 implementation/tests           |             **0 GPU** | ALLOWED CPU                  |
| B0/review/new G0                  |             **0 GPU** | ALLOWED CPU                  |
| Fixed-step profile                |                    ≤2 | BLOCKED until G0 PASS        |
| One-seed fixed/rematch train+eval |                    ≤4 | BLOCKED until profile review |
| Remaining pre-Stage2 reserve      |                    ≤4 | not automatically authorized |
| Total pre-Stage2                  |                   ≤10 | hard project cap             |
| Raw-video Stage2                  | no current allocation | BLOCKED                      |

任何一项超限：

```text
STOP_AND_REVIEW
```

不得从另一项静默借预算。

## W.4 支持“降低成本”的最低条件

必须同时满足：

1. R1与R0同视频、同checkpoint、同硬件。
2. R1 forward tokens与R0相同且账目一致。
3. backward tokens和backward CUDA time显著下降。
4. total wall time和GPU-hours的paired CI均显示改善。
5. G0 estimator fidelity通过。
6. complete chronological quality不劣。
7. savings不是data-cache热度、减少optimizer events或少处理视频造成。
8. protocol complexity和G0 fixed overhead纳入总成本解释。

若只降低backward tokens但wall time/GPU-hours不降低，R1仍可作为memory tool，但“高效训练”claim死亡。

---

# X. Claim Map and Reviewer Risks

## X.1 Claim map

| Claim                      | 当前状态                               | 当前允许表述                                    | 所需证据                             | Falsifier                     |
| -------------------------- | ---------------------------------- | ----------------------------------------- | -------------------------------- | ----------------------------- |
| Frozen `70df86e` G0是有效KILL | protocol-supported                 | 当前empty-state dynamic_birth失败             | signed bundle + gate             | 独立重算发现证据/代码不一致                |
| Empty-state CRS-EPS结构性不忠实  | protocol-supported                 | omitted history造成state/lifecycle mismatch | 已有G0                             | replacement不应复活它              |
| 所有event-centric训练均失败       | **unproven**                       | 不允许该表述                                    | 新route证据                         | R1通过即反例                       |
| 当前Q2是primary trainer       | **unproven**                       | 仅gold/reference candidate                 | capacity+G0+profile              | 任一exhaustion                  |
| R1保持完整numerical state      | requires experiment                | 设计上state-faithful                         | exact G0-A                       | logits/state mismatch         |
| R1对chrono‑64 gradient无偏    | partially identifiable             | 数学上在合同条件下成立                               | probability tests + G0-B         | propensity/RNG/graph mismatch |
| R1实际降低GPU-hours            | requires experiment                | 不允许先声称                                    | matched profile                  | wall/GPU-hour无改善              |
| Fixed binding优于rematch     | requires experiment                | 未知                                        | one/three/five seed              | fixed不优                       |
| Gain是identity-linked       | not identifiable currently         | 未知                                        | FRESH/TTF/GRU + identity metrics | gain仅aggregate或少发射            |
| R1是论文算法贡献                  | collapsed by strong precedent risk | 暂定位enabling infrastructure                | 不能被ETAD/SBP/standard IS重构        | closest prior即可解释             |
| Positive cached Q2授权LoRA   | unproven                           | 不自动授权                                     | 3 seeds+novelty+cost             | 任一gate失败                      |
| Full PETAL超越多论文重构          | unproven/reconstruction risk       | 不允许headline                               | FRESH/TTF/raw 2×2/second data    | TrackFormer-style control匹配   |

## X.2 最强 reviewer risks

### 1. Capacity与监督目标相互缠绕

GT birth是否得到slot取决于随机初始化模型的hard runtime状态。这不是普通OOM，而是模型预测控制训练标签可见性。若不先解决，fixed/rematch比较可能建立在不同程度的GT实例丢失上。

### 2. 扩槽掩盖lifecycle错误

把4改16可能消除exhaustion，却不修false births、refractory miscalibration和train/test lifecycle不一致。K必须由完整capacity decomposition推导。

### 3. R1高度可重构

ETAD已建立full/sequential forward加selective gradient的高效TAD路线；SBP、activation checkpointing和importance sampling覆盖相邻空间。R1大概率只能作为基础设施。([arXiv][3])

### 4. Sampling gradient variance

即使无偏，M=4仍可能在关键checkpoint产生低gradient cosine。不得用“期望无偏”替代有限预算可训练性。

### 5. One-factor过度表述

相同checkpoint twin audit可证明直接干预只在binding；两臂训练后weights和runtime lifecycle分化属于treatment-mediated effect，不能再要求所有birth masks永远相同，也不能不记录这些中介路径。

### 6. THUMOS-specific过拟合

单一小数据集、locked‑211 reporting boundary和自定义identity指标不足以支持一般persistent On‑TAD claim。

### 7. Cost numerator欺骗

R1仍处理所有forward tokens。若Python sequential scan或attention forward占主成本，backward sparsification可能几乎没有wall-time价值。

### 8. Raw-video extrapolation

Cached R1通过不能证明raw-video LoRA可行。视觉context forward、selective visual gradient、extractor cadence和raw availability都仍是新问题。

## X.3 Immediate kill rules

无论已有代码量多少，发生任一项即停止：

```text
capacity audit无合法shared policy
任何新G0 exhaustion
R1 exact masked implementation不闭合
M=4 aggregate gradient fidelity失败
matched profile无GPU-hour改善
fixed不优于rematch
identity endpoint无改善
recall/FN/end delay退化
FRESH/TTF/GRU重构收益
second dataset不复现
raw LoRA无独立增益
```

---

# Y. Ordered Next Actions

## 1. Freeze this Round‑2 decision

新增一份纯文档decision：

```text
CURRENT_EMPTY_STATE_CRS_EPS = permanently killed
SELECTED_SUCCESSOR = R1 CSFSB
Q2 = reference candidate
capacity = blocking
GPU before G0 pass = 0
```

不得修改旧G0记录或margin。

## 2. First implementation commit：capacity audit only

首个代码commit只实现：

```text
opentad/utils/q2_capacity_audit.py
tools/audit_q2_capacity_lifecycle.py
tests/test_q2_capacity_lifecycle.py
full-route exhaustion fail-closed in train_engine.py
```

不修改K、threshold、refractory、binding、loss或sampling。

## 3. Run complete zero-GPU capacity census

执行：

```text
160 fit-core videos
× seed 705/706/707 init checkpoints
× actual lifecycle
× counterfactual controller diagnostics
```

输出source-bound、non-overwritable artifact。

## 4. Freeze one shared capacity/lifecycle contract

按O节decision rule选择：

```text
current K4 unchanged
or
derived K
or
outcome-blind prior-bias initialization
or
refractory policy correction
```

只允许一项最小修复；fixed/rematch必须共享。

若没有合格方案，终止Q2和R1。

## 5. Implement R1 in a new scientific commit

只新增full-forward selective-backward路径。保留R0 untouched，旧CRS read-only。

## 6. Complete CPU test matrix

重点验证：

```text
real threshold 0.5
dropout 0.1
refractory 2
same-class overlap
long action
block edge
RNG parity
state exactness
gradient exactness
rollback
```

## 7. Generate B0 evidence

对exact commit：

```text
local B0
target-Linux B0
clean checkout
external evidence output
```

## 8. Same independent review

Reviewer必须明确返回：

```text
PASS
NEXT_GATE=NEW_HOLDOUT_G0
PROFILE=BLOCK
FORMAL=BLOCK
```

## 9. Preregister micro-trajectory and unseen G0

冻结：

```text
trajectory-core IDs
8 unseen holdout samples
12 preferred checkpoint states
fallback rule
M=4/H=8/mixture
capacity policy
margins
family-wise rule
commit/config/data/checkpoint hashes
```

## 10. Run terminal CPU-only G0

```text
G0-A exact implementation
G0-B estimator fidelity
G0-C zero exhaustion
```

任何一项失败，R1终止。

## 11. Only after G0 PASS request profile authorization

Profile仍须新ticket、新review和≤2 GPU-hour cap。

## 12. Only after profile PASS request one-seed mechanism kill

先只运行：

```text
seed705 fixed vs rematch
```

不得直接恢复三种子矩阵。

## 13. Escalation order

```text
one-seed positive
→ FRESH/TTF/GRU novelty killers
→ three paired seeds
→ five seeds + dense/overlap dataset
→ renewed novelty review
→ only then discuss raw-video LoRA
```

---

```text
CURRENT_G0=VALID_KILL
CURRENT_EMPTY_STATE_CRS_EPS=STRUCTURAL_KILL
Q2_STATUS=GOLD_REFERENCE_CANDIDATE
Q2_CAPACITY_GATE=BLOCK
SELECTED_ROUTE=R1
IMPLEMENTATION=ALLOW
NEW_HOLDOUT_G0=REQUIRED
PROFILE=BLOCK
FORMAL=BLOCK
GPU_HOURS_BEFORE_NEW_G0_PASS=0
CONFIDENCE=0.95
```

[1]: https://arxiv.org/abs/2205.07134 "https://arxiv.org/abs/2205.07134"
[2]: https://arxiv.org/abs/1604.06174 "https://arxiv.org/abs/1604.06174"
[3]: https://arxiv.org/abs/2203.16755 "https://arxiv.org/abs/2203.16755"
[4]: https://arxiv.org/abs/1705.08209 "https://arxiv.org/abs/1705.08209"
[5]: https://arxiv.org/abs/1702.05043 "https://arxiv.org/abs/1702.05043"


# Round 2 最终裁决

```text
CURRENT_G0=VALID_KILL
CURRENT_EMPTY_STATE_CRS_EPS=STRUCTURAL_KILL
Q2_STATUS=GOLD_REFERENCE_CANDIDATE
Q2_CAPACITY_GATE=BLOCK
SELECTED_ROUTE=R1
IMPLEMENTATION=ALLOW
NEW_HOLDOUT_G0=REQUIRED
PROFILE=BLOCK
FORMAL=BLOCK
GPU_HOURS_BEFORE_NEW_G0_PASS=0
CONFIDENCE=0.95
```

其中：

> `IMPLEMENTATION=ALLOW` **只允许零 GPU 的 capacity/lifecycle 审计、R1 代码实现、CPU 测试、B0、独立复核与新 holdout G0**。它不允许 profile、effectiveness training、formal training 或 raw-video Stage 2。

唯一保留的候选训练路线是：

```text
R1 — Chronological State-Faithful Selective Backward
     完整时间顺序状态保持 + 选择性反向传播
```

它不再采样或重建 recurrent state，只采样 **loss/gradient contribution**。每个视频的所有 cached tokens 仍完整、因果、按顺序前向；所有 runtime lifecycle 和 canonical supervision state 都真实推进；只有包含已选监督位置的 64-bin block prefix 建立 autograd 图。该路线在满足严格实现合同时，对当前 chronological 64-bin TBPTT gradient estimand 是 **HH/IPW 无偏、高方差估计**。

旧 `dynamic_birth`、`fixed_192`、`reset` 和 `CrsEpsFeatureDataset` 不得复活为主训练路线。

---

# N. Author-Answer Reconciliation

## N.1 Repository / snapshot certificate

核验结果：

```text
Repository:
yuzbo/OpenTAD_OnlineTADClean_20260702

Branch:
codex/full-petal-implementation

Branch HEAD:
d5fc6a69466e6cb05429a3ebbc71832b755aa38c

Frozen scientific commit:
70df86ea3d38d70c658ae0ee9e04245d57b834d4
```

`codex/full-petal-implementation` 当前精确指向 `d5fc6a6`。`70df86e..d5fc6a6` 仅增加 Round‑1 review、absorption、Prompt、wiki、decision register 和 `.gitattributes`；没有修改 model、dataset、sampling、persistent detector、training engine、G0 gate、config 或 tests。`d5fc6a6` 本身的提交信息也是 `docs: absorb CRS-EPS structural kill review`。

因此：

```text
SCIENTIFIC_IMPLEMENTATION=70df86e
ROUND2_DOCUMENTATION_HEAD=d5fc6a6
HIDDEN_SCIENTIFIC_CODE_CHANGE=NO
```

Round‑1 对此前 HEAD 的文件级核验也明确把科学实现锚点固定在 `70df86e`，并将后续变化归类为 review/failure dossier/wiki 更新。

## N.2 U1–U8 disposition

| 项目                                         | 状态           | Round‑2 处理                                                                                                                                                                            |
| ------------------------------------------ | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **U1：完整 signed G0 bundle**                 | **RESOLVED** | 项目侧已检查 `audit.json`、12 rows、selection、margins、policy、manifest、preflight 和 checkpoint；terminal audit SHA 为 `a081…109b`。这足以固定当前 route 的终局。原 bundle 不公开仍限制第三方逐字节复算，但不再是本轮 route blocker。 |
| **U2：selection universe 与规则**              | **RESOLVED** | `160 videos × 4 draws = 640` candidates；每长度 quartile 选一条；先最大化六类 stress union，再 proposal diversity，再 lexicographic tie。它是强 falsification set，不是总体失败率估计。                                |
| **U3：slot exhaustion 与 first divergence**  | **PARTIAL**  | counts 和 first-divergence bins 已关闭，但 exhaustion 究竟来自 K=4、false birth、refractory、same-bin ordering 还是 lifecycle coupling 仍未分解。它独立阻塞 Q2 primary readiness。                              |
| **U4：正式 pilot 起点**                         | **RESOLVED** | G0 checkpoint 是 seed‑705 deterministic initialization，正是原计划第一训练 seed 的起点。不能用“任意 checkpoint”回避第一更新处的结构失败。                                                                              |
| **U5：chronological checkpoint trajectory** | **RESOLVED** | 答案是“不存在”。init/early/mid/late checkpoint 不是免费资产；新 G0 必须通过一个受限、outcome-blind 的 CPU micro-trajectory 生成。                                                                                 |
| **U6：B0 / same-review / G0 验证**            | **RESOLVED** | 项目本地和 target Linux 已独立重算。Round‑1 工具不可访问是 reviewer environment boundary，不是证据不存在。                                                                                                       |
| **U7：同硬件 matched profile**                 | **PARTIAL**  | 已确认不存在；因此 R0、R1、R2、R3 的真实 wall-time/GPU-hour 排序仍 OPEN。现阶段只能比较 token/FLOP 上界。                                                                                                          |
| **U8：数据与 feature provenance**              | **PARTIAL**  | annotation、split、data identity、cache manifest 已 hash-bound，足以支持当前 G0；raw arrays 与 extractor artifacts 不在 bundle，继续阻塞完整第三方重建、raw-video causality 和 Stage 2。                            |

这些处理与本地 absorption 一致：U1、U2、U4、U5、U6 已有明确回答；U3、U7、U8 仍分别留下 capacity、cost 和 raw provenance 边界。

## N.3 对当前 G0 的最终解释

当前 terminal G0 是有效、不可回改的 KILL：

* 4 个预注册样本中 3 个违反绝对 fidelity；
* 最低 gradient cosine 为 `0.2831`；
* 最低 gradient sign agreement 为 `0.5445`；
* 最低 continuous-state cosine 为 `0.1028`；
* 14 项 violations；
* 唯一完整回退至 video start 的 positive control 精确闭合。

原始 trace 进一步证明：

* 两个失败样本中 `dynamic_birth == fixed_192` 的 replay range；
* 唯一 dynamic extension 只比 fixed 多 8 bins，仍存在显著 loss/state 偏差；
* 最长样本在 bin 209 出现 canonical lifecycle/loss-binding 分叉，bin 216 出现 endpoint ownership 分叉，bin 352 出现 birth assignment 分叉。

因此正确标签是：

```text
STRUCTURAL-KILL-CURRENT-EMPTY-STATE-DYNAMIC-BIRTH-CRS-EPS
```

不是：

```text
KILL-ALL-SELECTIVE-OR-EVENT-CENTRIC-TRAINING
```

---

# O. Q2 Capacity/Lifecycle Verdict

## O.1 Verdict

```text
Q2_STATUS=GOLD_REFERENCE_CANDIDATE
Q2_CAPACITY_GATE=BLOCK
```

当前证据不足以断言“四槽在 THUMOS14 上结构性不足”，但足以判定：

> 当前 `K=4 + random-init logits + thresholds 0.5/0.5/0.5 + refractory=2 + runtime-FREE-constrained GT birth assignment` 尚不具备 primary training readiness。

两个 `video_start_full` controls 分别发生 1 和 2 次 slot exhaustion。它们从视频起点重建完整 chronological numerical state，因此 exhaustion 不能归因于局部 replay 丢失历史；它是 Q2 detector/lifecycle/capacity 合同自身的独立问题。

## O.2 为什么 annotation 最大并发 2 不能证明 K=4 足够

当前配置为：

```text
num_slots=4
birth_threshold=0.5
alive_threshold=0.5
end_threshold=0.5
refractory_steps=2
```

且 G0 checkpoint 是随机初始化，不是经过 lifecycle calibration 的 checkpoint。

真实所需容量不是：

[
K_{\text{needed}}(t)=N_{\text{GT active}}(t).
]

而至少是：

[
K_{\text{needed}}(t)
====================

N_{\text{canonical occupied}}(t)
+
N_{\text{false runtime ACTIVE}}(t)
+
N_{\text{runtime REFRACTORY but canonically free}}(t)
+
N_{\text{same-bin birth reserve}}(t).
]

其中只有第一项受 annotation concurrency 约束。

### 当前代码的关键耦合

每个时间步中：

1. 训练器先从**前一 runtime state**选择 `SLOT_FREE` slots；
2. `head.step` 产生当前 logits；
3. canonical GT birth assignment 只能使用：

   ```text
   runtime FREE ∩ canonical unoccupied
   ```
4. 未获分配的 GT birth 被计为 exhausted；
5. 它仍会进入 `born_instance_ids`，因此不会在之后重新 birth；
6. 随后的 `decode_step` 才根据 birth/alive/end logits 更新 runtime lifecycle。

这形成几个可能根因：

* 随机 birth logits 触发 false ACTIVE；
* 随机 end/alive logits 导致错误 free 或错误 refractory；
* refractory 占用本来 canonical 已空闲的 slot；
* 当前 step 中即将 end/free 的 slot，在 GT birth assignment 时仍不可复用；
* 真正 canonical demand 超过 K；
* runtime availability 与 supervision lifecycle 的耦合本身不合理。

`decode_step` 明确允许随机 birth 把 FREE 变 ACTIVE，允许 ACTIVE 因 alive/end 阈值释放或 emission 后进入 refractory。

## O.3 不是 audit-only 假象

`video_start_full` G0 control 走的是同一 detector、同一 `head.step`、同一 `supervision.transition` 和同一 runtime `decode_step`。它只是把 numerical replay 起点设为视频开始，并仍采用全局 64-bin TBPTT。故 exhaustion 不是另一个 audit controller 伪造出来的。

但是，这些结果只发生在：

```text
random-init checkpoint
selected G0 prefixes
```

它们尚不能区分：

* 暂态初始化问题；
* 四槽结构问题；
* refractory policy；
* threshold/prior bias；
* same-bin lifecycle ordering；
* 真正训练中会持续存在的问题。

因此状态是 `BLOCK`，不是直接 `INVALID`。

## O.4 当前 engine 的 fail-closed 不对称

CRS group 路径明确在 `slot_exhaustion != 0` 时 rollback 并抛出 scientific failure。

普通 chronological Q2 路径则只把 exhaustion 写入 detector audit，现有 engine 没有对 non-CRS route 做对称的 optimizer-before-commit 阻断。

因此当前 R0 可以作为诊断 reference，却还不能作为 accepted primary trainer：它可能继续更新一个已经永久漏掉 GT birth 的 trajectory。

## O.5 零 GPU capacity/lifecycle audit

### Audit A：annotation-only demand

对完整 fit-core 160 videos 逐 bin 计算：

```text
GT active count
GT births
GT ends
same-bin birth/end
same-class concurrency
cross-class concurrency
canonical occupied count
canonical occupancy + declared refractory reserve
minimum oracle-free K
```

目的：回答即使 runtime controller 完美，K=4 是否仍结构性不足。

### Audit B：actual chronological controller replay

对 deterministic init seeds `705,706,707`，CPU 完整 replay 160 videos，逐 bin 记录：

```text
birth/alive/end probabilities
runtime status before and after
canonical ownership
refractory counters
available slots
GT birth IDs
birth assignments
exhausted IDs
emissions
false-active count
same-bin ending/free candidates
```

每个 exhaustion 必须唯一归因为：

```text
TRUE_CANONICAL_CAPACITY
FALSE_ACTIVE_OCCUPANCY
REFRACTORY_OCCUPANCY
SAME_BIN_NON_REUSE
MIXED_CAUSE
```

不允许只输出总计数。

### Audit C：same-logits counterfactual lifecycle replay

这是最高杠杆的零 GPU 诊断。

当前 `head.step` 的 query/memory 更新不读取 `slot_status`、`refractory`、`start_frames`、peak score 或 canonical maps；这些状态主要进入 decode 与 supervision ownership。因此同一 checkpoint、features 和 RNG 下，可以只计算一次 logits，再精确重放多个 controller policy。

诊断网格：

```text
actual thresholds/refractory
refractory=0
birth threshold sensitivity
end/alive threshold sensitivity
outcome-blind birth-head prior bias
same-bin release-before-birth diagnostic
canonical-only privileged upper bound
K=2,3,4,5,... capacity census
```

其中 `canonical-only` 是 privileged diagnosis，绝对不能进入正式 trainer。

### Audit D：decision rule

| 结果                                            | 决策                                                                        |
| --------------------------------------------- | ------------------------------------------------------------------------- |
| canonical-only demand > 4                     | K=4 结构性不足；由全 fit-core demand 推导最小 K，再在未见 capacity holdout 验证。             |
| canonical demand ≤4，但 actual K4 exhaustion >0 | 优先修 runtime lifecycle initialization/policy；不得直接扩到 16 掩盖 false occupancy。 |
| refractory=0 单独消除 exhaustion                  | refractory 是主要根因；任何修订必须 fixed/rematch 共享并进入新 G0。                          |
| outcome-blind prior-bias 初始化消除 exhaustion     | 可考虑最小 prior 初始化，但不能用 Q2 mAP 选择 bias。                                      |
| same-bin ordering是主因                          | 必须明确“end 后释放是否能服务同 bin birth”的信息合同，并让训练/推理一致。                             |
| 所有合理 shared policy 仍 exhaustion               | Q2 与 R1 全部停止，`Q2_STATUS=INVALID`。                                         |

## O.6 晋级 PRIMARY 的必要条件

Q2 只有在全部满足后才能升级为 `PRIMARY`：

1. 完整 fit-core capacity census 完成；
2. K、threshold/prior initialization、refractory 和 same-bin reuse rule 在任何 effectiveness result 前冻结；
3. fixed/rematch 共享完全相同的 lifecycle policy；
4. R0、R1 和任何训练 route 均在 optimizer 前对 exhaustion fail closed；
5. seeds 705/706/707 及新 checkpoint strata 均零 exhaustion；
6. 不使用 GT active-slot、GT query 或 privileged state initialization；
7. R1 新 holdout G0 PASS；
8. matched profile 证明成本可接受；
9. 一粒种子 mechanism kill 没有 recall/delay collapse。

当前未满足第 1、2、4、5、7、8、9 项。

---

# P. Route Portfolio

评分范围 1–5；“实现风险”中 5 表示风险最低。

| Route                                         | State fidelity |    Gradient validity | Cost potential | 实现风险 | Novelty | G0 testability | Verdict                         |
| --------------------------------------------- | -------------: | -------------------: | -------------: | ---: | ------: | -------------: | ------------------------------- |
| **R0 full chronological**                     |              5 | 5，相对 chrono‑64 exact |              1 |    4 |       1 |              5 | reference candidate；capacity 未过 |
| **R1 full forward + selective backward**      |              5 |             4，无偏但高方差 |              3 |    3 |       2 |              5 | **唯一 GO-FOR-IMPLEMENTATION**    |
| **R2 exact chronological recomputation**      |              5 |                    5 |              2 |    3 |       1 |              5 | 在预知 selection 时被 R1 支配          |
| **R3 runtime-state checkpoints**              |              2 |                    2 |          4（名义） |    1 |       1 |              3 | parameter-stale，NO-GO           |
| **R4 activation checkpoint/SBP/ARTBP/UORO**   |            2–5 |           1–5，取决于子方法 |            1–2 |    1 |       1 |              2 | 不是一个统一可识别 route                 |
| **R5 chronological only / abandon surrogate** |              5 |                    5 |              1 |    4 |       0 |              5 | 最诚实 fallback，不是高效路线             |

## P.1 R0

R0：

```text
all causal forward tokens
all token losses
global 64-bin detach
one optimizer event per complete video
```

它对当前 locked chronological 64-bin TBPTT estimand 是 exact reference，但：

* 不降低 token forward/backward；
* 没有训练方法新颖性；
* 当前 slot-capacity gate 未通过；
* 没有同硬件 cost profile；
* 没有 effectiveness result。

因此不能直接提升为 primary。

## P.2 R1

R1 保持：

```text
所有 token 完整 chronological forward
所有 runtime lifecycle 真正推进
所有 canonical supervision transition 真正推进
与 R0 相同的 global 64-bin detach
一视频一次 optimizer/scheduler event
```

只选择哪些 bin 的 loss 进入 gradient estimator。

它可能减少：

```text
backward graph tokens
backward FLOPs
平均 activation retention
非监督 bin 的 loss construction
```

它不减少：

```text
forward tokens
feature loading
Python sequential head.step
attention/GRU forward
runtime decode
supervision lifecycle transition
```

因此它可能被 profile 直接杀死：若 forward/Python unroll 是主成本，R1 的 GPU-hour 收益可能接近零。

## P.3 R2

R2：

1. 完整 no-grad chronological scan；
2. 保存合法 block-start state；
3. 从同一参数版本重算 selected block prefixes；
4. backward selected losses。

若监督位置由 immutable manifest 预先知道，则：

[
F_{R1}=T,\qquad F_{R2}=T+P,
]

而二者 backward 都是 (P)。R2 只增加 recomputed forward，没有精度优势，因此被 R1 支配。

只有当 selection 依赖第一次模型输出时，R2 才可能必要；但那会引入 model-dependent propensity、新 sampling estimand 和新的 one-factor 风险，不属于本轮最小路线。

## P.4 R3

显式 checkpoint (H_c(\theta_k)) 在参数更新后用于 (\theta_{k+1}) 时：

[
H_c(\theta_k)\neq H_c(\theta_{k+1}).
]

将 checkpoint 与参数 SHA 绑定只能**检测 stale**，不能修复 stale state。

若每次 optimizer update 后重新生成所有 checkpoint，则 R3 退化为 R2/R0；若不重新生成，则优化 stale-state objective。它还丢失 checkpoint 之前的 historical Jacobian。

结论：

```text
R3=NO-GO
```

## P.5 R4

R4 实际包含多个不同科学对象：

* **Activation checkpointing**：保留 exact gradient、降低 activation memory，但用额外 forward recomputation 换显存；不减少目标 backward FLOPs。
* **Selective Backpropagation / SBP**：省部分 backward，但改变 layer/path gradient estimator，需要独立无偏性或偏差审计。
* **ARTBP**：对随机 truncation 做补偿，目标通常是 full-BPTT；当前项目锁定的是 deterministic chrono‑64 TBPTT，并包含 hard lifecycle branches。
* **UORO**：在线低秩 RTRL 近似，理论上可无偏，但对 multi-slot attention、hard threshold 与长视频状态的方差和实现风险都过高。

这些不能作为一个 route 并行实现。Activation checkpointing 以后可以作为 R1 的显存辅助，但不能混入首版方法和 G0。

## P.6 R5

R5 是治理上最诚实的 fallback：

```text
放弃高效 surrogate
只保留 R0 chronological Q2
```

若 R1 G0 或 profile 失败，应切换为 R5，而不是继续发明第三种 replay surrogate。

但 R5 本身仍需 capacity gate，也不能支持“高效训练方法”贡献。

---

# Q. Recommended Route

## Q.1 唯一选择

```text
SELECTED_ROUTE=R1
ROUTE_NAME=Chronological State-Faithful Selective Backward
SHORT_NAME=CSFSB
DECISION=GO-FOR-IMPLEMENTATION
```

该 GO 仅授权：

```text
zero-GPU capacity audit
CPU code implementation
CPU test matrix
B0
independent exact-commit review
new unseen CPU G0
```

不授权：

```text
GPU profile
one-seed training
formal training
raw-video LoRA
```

## Q.2 为什么 R1 是唯一保留路线

R1 是唯一同时满足下列条件的候选：

1. 完整 chronological numerical state；
2. 完整 discrete lifecycle；
3. 无 cross-parameter-version stale state；
4. 与 R0 相同的 global 64-bin TBPTT；
5. sampling 只作用于 loss contribution；
6. 可用 HH/IPW 对 chrono‑64 gradient 构造无偏估计；
7. 一视频一 optimizer event；
8. 可通过 exact implementation G0 与 estimator G0 分开证伪；
9. 明确承认只可能节省 backward，不冒充 forward savings。

## Q.3 Capacity 是前置而非并行问题

执行顺序必须是：

```text
capacity/lifecycle audit
→ freeze one shared valid lifecycle contract
→ implement R1
→ B0/review
→ new holdout G0
```

若 capacity audit 无法找到一个：

```text
无 GT taint
fixed/rematch shared
zero exhaustion
train/inference consistent
```

的合同，则立即：

```text
Q2_STATUS=INVALID
SELECTED_ROUTE=NONE
IMPLEMENTATION=BLOCK
```

---

# R. Exact Method Design

## R.1 与旧 CRS-EPS 的根本差异

旧 route：

```text
sample local episode
→ reset runtime/supervision to empty
→ replay bounded prefix
→ approximate chronological state
```

R1：

```text
sample only loss locations
→ scan complete video chronologically
→ preserve exact runtime/supervision state
→ selectively construct backward graph
```

以下旧字段不得进入新 trainer：

```text
dynamic_birth
fixed_192 replay
reset replay
independent_episode_state
replay_range as state surrogate
is_video_start=True per sampled draw
```

## R.2 Outcome-blind sampling manifest

保留原 proposal 设计：

```text
M=4 draws/video/epoch
H=8 bins/window

mixture:
  uniform  0.40
  start    0.15
  end      0.20
  ongoing  0.10
  hardbg   0.15
```

但 manifest 不再生成 episode replay，只生成每个 full video 的 loss coefficient mask。

对视频 (v) 和 bin (t)：

[
m_{v,t}
=======

\sum_{r=1}^{M}
\mathbf 1[t\in S(A_{v,r})].
]

单 draw coverage：

[
\rho_{v,t}
==========

\Pr_{A\sim q_v}{t\in S(A)}.
]

最终系数：

[
a_{v,t}
=======

\frac{m_{v,t}}{M}
\frac{1/T_v}{\rho_{v,t}}.
]

重复 window exposure 不去重。由于每次 exposure 现在都对应同一个 full chronological state，同一 bin 的重复 loss 可以合并为一次：

[
m_{v,t}\times \ell_{v,t}.
]

Manifest 至少包含：

```text
schema
commit/config/data identity
video_id
epoch
seed
M/H/mixture
draw components
draw anchors
rho_by_bin
multiplicity_by_bin
coefficient_by_bin
selected_unique_bins
selected_blocks
last_selected_bin_by_block
manifest SHA-256
```

不得包含：

```text
runtime state
hidden checkpoint
GT slot
model-dependent score
future model output
```

训练 annotation 可以决定 sampling proposal，但 sampling metadata不得进入模型 plane。

## R.3 State recurrence

连续 recurrent state：

[
h_t=F_\theta(h_{t-1},x_t;\xi_t).
]

Runtime lifecycle：

[
r_t=D(r_{t-1},z_t).
]

Training-only canonical supervision：

[
c_t=C(c_{t-1},y_t,z_t).
]

其中：

* (x_t)：causal cached feature；
* (\xi_t)：dropout/RNG；
* (z_t)：当前 model outputs；
* (y_t)：prefix-observable training target。

无论 (a_{v,t}) 是否为零，所有 bin 都必须执行：

```text
head.step
supervision.transition
runtime decode_step
birth/retirement/refractory update
canonical ownership update
```

只允许跳过 unselected bin 的 differentiable loss construction。

## R.4 Global 64-bin detach

令全局 block：

[
B_j=[64j,\min(64(j+1),T_v)).
]

在每个 block 入口：

[
h_{64j}\leftarrow \operatorname{stopgrad}(h_{64j}),
]

与 R0 完全一致。

对 block (B_j)，令该 block 最后一个非零 coefficient 的位置为 (u_j)。

### Block 无 selected bin

```text
完整 block 在 no_grad 下 forward
state/lifecycle 全部更新
不建立 autograd graph
```

### Block 有 selected bin

```text
block start → u_j:
    grad enabled
    所有 token 构建 recurrence graph
    仅 a_t > 0 的 bin 计算 loss

u_j 之后:
    detach numerical state
    block tail 在 no_grad 下 forward
    lifecycle 继续正常推进
```

这样：

* selected loss 拥有与 R0 相同的 64-bin Jacobian prefix；
* selected 之前的 unselected tokens 仍在图中；
* selected 之后的 tail 不保留无用图；
* 下一 block 原本就 detach，所以 tail historical Jacobian 不属于 R0 estimand；
* numerical state 始终完整。

## R.5 Loss

每个 selected bin 保持当前五项 loss：

```text
birth_loss
alive_loss
class_loss
start_loss
end_loss
```

不改变：

* risk masks；
* class/start loss binding；
* negative semantics；
* component loss weights；
* fixed/rematch intervention。

视频 sampled risk：

[
\widehat L_v
============

\sum_{t=1}^{T_v}
a_{v,t}\ell_{v,t}.
]

严禁再除以：

```text
selected bin count
selected block count
unique selected bin count
realized coefficient sum
```

最后一项会把 HH 变成 realized SNIPW，并改变 estimator。

## R.6 Optimizer event

每视频：

```text
zero_grad
snapshot model/optimizer/scheduler/runtime mutable state
complete chronological selective-backward scan
capacity check
finite loss/gradient check
clip
one optimizer.step
one scheduler.step
transaction commit
```

任何失败都 rollback 整个视频。

## R.7 RNG contract

R1 与 R0 必须处理完全相同的 token sequence，并消耗相同 dropout draws。

`no_grad()` 不得跳过 forward 或随机算子。

G0 必须证明：

```text
same checkpoint
same feature bytes
same complete token order
same initial RNG
same lifecycle contract
→ same logits and final numerical state
```

需要注意：grad-enabled 与 no-grad 在某些 kernel/backend 上可能选择不同实现。因此 exactness 是实验 gate，不是仅凭理论假设。

## R.8 Rollback contract

以下任一事件使完整视频失效并 rollback：

```text
slot exhaustion
non-finite logits/loss/gradient
manifest mismatch
coefficient mismatch
RNG mismatch
GT/terminal taint
cross-video state leakage
mutable-buffer leakage
optimizer/scaler/scheduler failure
transaction commit failure
workload counter mismatch
```

---

# S. Mathematical Estimand

## S.1 R0 control

视频 (v) 有 (T_v) bins。

当前 R0 的视频目标为：

[
L_v^{(0)}(\theta)
=================

\frac1{T_v}
\sum_{t=1}^{T_v}
\ell_{v,t}
\bigl(
\theta,h_t^{64}(\theta),r_t,c_t
\bigr).
]

总体：

[
L^{(0)}(\theta)
===============

\frac1V\sum_{v=1}^{V}L_v^{(0)}(\theta).
]

其中 (h_t^{64}) 使用 global 64-bin detach。它不是 full-video BPTT。

梯度：

[
g_v^{(0)}
=========

\frac1{T_v}\sum_{t=1}^{T_v}g_{v,t}^{64}.
]

## S.2 R1 estimator

每视频采样：

[
A_{v,1},\ldots,A_{v,M}\sim q_v.
]

定义：

[
I_{v,t}^{(m)}
=============

\mathbf1[t\in S(A_{v,m})],
]

[
\rho_{v,t}
==========

\mathbb E[I_{v,t}^{(m)}].
]

系数：

[
a_{v,t}
=======

\frac1M\sum_{m=1}^{M}
I_{v,t}^{(m)}
\frac{1/T_v}{\rho_{v,t}}.
]

R1 loss：

[
\widehat L_v^{(1)}
==================

\sum_t
a_{v,t}
\ell_{v,t}
\bigl(
\theta,h_t^{64}(\theta),r_t,c_t
\bigr).
]

因为：

[
\mathbb E\left[
\frac1M\sum_m I_{v,t}^{(m)}
\right]
=======

\rho_{v,t},
]

所以：

[
\mathbb E[a_{v,t}]
==================

\frac1{T_v},
]

从而：

[
\mathbb E_A[\widehat L_v^{(1)}]
===============================

L_v^{(0)}.
]

若 selected bin 的 graph 精确从其 global 64-bin block start 构造，则：

[
\mathbb E_A[
\nabla_\theta\widehat L_v^{(1)}
]
=

g_v^{(0)}.
]

## S.3 精确分类

| 属性                       | R1 相对 R0                                  |
| ------------------------ | ----------------------------------------- |
| Numerical state          | **Exact state-faithful**，待 G0-A 验证        |
| Discrete lifecycle       | **Exact state-faithful**，待 capacity/G0 验证 |
| 单次 sampled gradient      | 非 exact full gradient                     |
| 期望 gradient              | **Unbiased relative to chrono‑64**        |
| Gradient variance        | 高于 R0                                     |
| Full-video BPTT          | 否                                         |
| Forward FLOPs            | 不降低                                       |
| Backward FLOPs           | 可能降低                                      |
| Peak activation memory   | 不保证降低；最坏仍为 64-bin graph                   |
| Sampling correction      | 只校正 loss exposure                         |
| Capacity/lifecycle error | 完全不能校正                                    |

## S.4 Sampling probability 能校正什么

能校正：

```text
event-heavy bin exposure
window overlap multiplicity
video-uniform / decision-bin-uniform target
selected loss contribution
```

不能校正：

```text
slot exhaustion
invalid lifecycle
different runtime state
GT-tainted state
future-conditioned model-dependent selection
RNG path divergence
different detach schedule
parameter-stale checkpoint
feature extractor future leakage
```

这正是旧 CRS-EPS 失败的核心：HH/IPW 只作用于 suffix loss numerator，不能恢复 persistent query、feature memory、refractory、peak score/label、canonical ownership 或历史 Jacobian。

## S.5 有限采样与 optimizer trajectory

即使 gradient estimator 无偏，AdamW 更新是非线性的，有限 (M) 的高方差仍会导致不同 parameter trajectory。

因此：

```text
unbiased ≠ practically trainable
```

必须有 G0-B 的有限预算 gradient fidelity gate。

## S.6 Workload bounds

令：

[
J_v=\left\lceil\frac{T_v}{64}\right\rceil.
]

对 block (j)，若没有 selected bin，令 (p_{v,j}=0)；否则 (p_{v,j}) 为从 block start 到最后 selected bin 的 token 数。

### R0

[
F_v^{R0}=T_v,
\qquad
B_v^{R0}=T_v.
]

### R1

[
F_v^{R1}=T_v,
\qquad
B_v^{R1}
========

# P_v

\sum_{j=1}^{J_v}p_{v,j}
\le T_v.
]

### R2

[
F_v^{R2}=T_v+P_v,
\qquad
B_v^{R2}=P_v.
]

因此对 outcome-blind 已知 selection：

[
F^{R1}<F^{R2}.
]

### Supervision

正常 (T_v\ge8) 时：

[
MH=4\times8=32
]

个 repeated exposures，unique selected bins 不超过 32。

### Optimizer

[
O_v^{R0}=O_v^{R1}=1.
]

## S.7 Activation memory

R0 与 R1 的最坏 graph horizon 都是 64 bins：

[
M_{\text{activation}}^{R1}
\le
M_{\text{activation}}^{R0},
]

但若某 block 的最后 selected bin 位于 block 末尾，两者 peak activation 可能相同。

因此在 profile 前只允许说：

```text
R1 lowers expected backward/retained-graph workload
```

不能提前说：

```text
R1 definitely lowers peak VRAM
```

## S.8 Persistent-state storage lower bound

当前：

```text
K=4
D=256
memory_size=192
```

主要浮点 state：

[
KD + 192D + 2K
==============

50{,}184
]

个浮点元素。

加上 `slot_status/refractory/label_state` 三个 4-element int64 tensor，最低 tensor payload 约为：

```text
BF16 continuous:
100,464 bytes ≈ 98.1 KiB / stream

FP32 continuous:
200,832 bytes ≈ 196.1 KiB / stream
```

还不包括：

```text
source_frames
Python maps/sets
supervision state
ledger
audit rows
allocator overhead
```

R1 不新增 persistent checkpoint storage。

---

# T. Code-Level Delta

## T.1 First commit：capacity audit only

第一个代码提交**不得实现 R1**，也不得改变 K、threshold、refractory、binding 或 loss。

### 新增 `opentad/utils/q2_capacity_audit.py`

功能：

* annotation demand；
* chronological runtime occupancy；
* exhaustion cause attribution；
* fixed-logit counterfactual replay；
* K/threshold/refractory/same-bin sensitivity；
* canonical-only privileged diagnostic。

Fail-closed invariant：

```text
every exhausted birth has exactly one declared cause,
or the audit is invalid
```

### 新增 `tools/audit_q2_capacity_lifecycle.py`

要求：

* clean checkout；
* exact config；
* exact checkpoint bytes；
* exact fit-core population；
* non-overwritable external output；
* deterministic seed；
* hash-bound report。

### 新增 `tests/test_q2_capacity_lifecycle.py`

覆盖：

```text
true K insufficiency
false ACTIVE occupancy
REFRACTORY occupancy
same-bin end/birth non-reuse
mixed cause
invalid privileged policy leakage
```

### 修改 `opentad/cores/train_engine.py`

为 non-CRS chronological route 增加与 CRS 相同的：

```text
slot_exhaustion > 0
→ rollback before optimizer
→ scientific failure
```

### 不修改

```text
num_slots
thresholds
refractory
binding
sampling
optimizer
loss
```

## T.2 R1 implementation commit

只有 capacity contract 冻结后才允许。

### 新增 `opentad/utils/state_faithful_sampling.py`

可复用当前 proposal probability mathematics，但不得调用或生成：

```text
episode_geometry
dynamic replay range
left-censored replay
independent episode state
```

输出 full-video coefficient manifest。

### 新增 `opentad/datasets/state_faithful_selective_feature.py`

应继承或复用 `StreamingFeatureDataset`，而不是 `CrsEpsFeatureDataset`。

行为：

* 仍枚举完整视频全部 chronological chunks；
* 每个 chunk 附带 global bin coefficients；
* 不切 local episode；
* 不重置 stream；
* 不把 sampling annotation 字段送入 model metadata。

### 修改 `persistent_trajectory_ontad.py`

新增独立入口：

```python
train_state_faithful_selective_chunk(...)
```

必须保证：

* 所有 bins 执行 state/lifecycle；
* block prefix 根据 last selected bin 建图；
* selected tail 后 detach；
* loss 仅 selected bins；
* current available-slot timing 不被顺便改变。

不得复用旧 `train_crs_eps_episode` 的 empty-state 初始化。

### 新增 `opentad/cores/state_faithful_train_engine.py`

职责：

* full-video chronological transaction；
* block-wise backward；
* 一视频一 optimizer event；
* coefficient 已含 HH normalization；
* 不进行 realized-weight normalization；
* exhaustive workload counters；
* whole-video rollback。

### 新增 `opentad/utils/state_faithful_paired_audit.py`

比较：

```text
masked all-grad gold
vs
selective-backward candidate
```

记录：

* all logits；
* continuous/discrete runtime；
* canonical lifecycle；
* birth/endpoint ownership；
* selected losses；
* all-trainable gradients；
* RNG；
* workload。

### 新增工具

```text
tools/build_state_faithful_manifest.py
tools/audit_state_faithful_manifest.py
tools/preregister_state_faithful_g0.py
tools/run_state_faithful_g0.py
```

### 新增 configs

```text
configs/causaltad/thumos_pes_q2_sfsb_base.py
configs/causaltad/thumos_pes_q2_sfsb_fixed.py
configs/causaltad/thumos_pes_q2_sfsb_rematch.py
```

fixed/rematch 删除 binding mode 和 `work_dir` 后必须 exact equal。

### 修改 launch/result gates

`full_petal_launch.py`：

* old CRS KILL 永久不能授权新 route；
* R1 必须使用新 G0 schema；
* new G0 PASS 与 exact implementation commit 绑定；
* profile 继续 pre-CUDA fail closed。

`check_full_petal_results.py`：

* capacity artifact；
* R1 G0 artifact；
* full workload denominators；
* standard mAP；
* population identity；
* zero exhaustion。

## T.3 Core pseudocode

```python
def selective_block(
    features,
    schedule,
    state,
    supervision,
    coefficients,
    global_block_start,
):
    # Matches the chronological 64-bin boundary.
    state = detach_runtime_state(state)

    selected = [
        i for i, coeff in enumerate(coefficients)
        if coeff > 0.0
    ]
    last_selected = max(selected) if selected else None
    block_cost = None

    for local_idx, feature in enumerate(features):
        global_bin = global_block_start + local_idx
        grad_enabled = (
            last_selected is not None
            and local_idx <= last_selected
        )

        with torch.set_grad_enabled(grad_enabled):
            outputs, state = head.step(
                feature, state, source_frame[local_idx]
            )

            # Always advance supervision lifecycle.
            transition = supervision.transition(
                schedule[local_idx],
                cost_provider(outputs),
                available_slots=runtime_free_slots(state),
            )

            if coefficients[local_idx] > 0.0:
                raw_losses = step_losses(
                    outputs, schedule[local_idx], transition
                )
                term = (
                    coefficients[local_idx]
                    * weighted_component_sum(raw_losses)
                )
                block_cost = (
                    term if block_cost is None
                    else block_cost + term
                )

            # Always advance runtime lifecycle.
            _, state = head.decode_step(
                outputs,
                state,
                current_frame=source_frame[local_idx],
            )

        if local_idx == last_selected:
            state = detach_runtime_state(state)

    return block_cost, state, supervision, audit
```

实际实现必须保持当前“available slots 在本 step decode 前确定”的语义，除非 capacity audit 已独立冻结一个 shared lifecycle 修订。不能在实现 R1 时顺便改变它。

## T.4 Old route disposition

以下文件和逻辑保留为只读历史：

```text
CrsEpsFeatureDataset
dynamic_birth
fixed_192
reset
old G0 runner
old G0 gate
70df86e evidence
```

用途：

```text
negative result
development-only regression
adversarial control
```

从新 profile/formal allowlist 删除。不得删除、改写或把它们重命名成 R1。

---

# U. Test Matrix

| 类别          | 测试                                    | Expected                                             |
| ----------- | ------------------------------------- | ---------------------------------------------------- |
| Capacity    | GT concurrency ≤K，但 false births 占满 K | 精确归因 `FALSE_ACTIVE`                                  |
| Capacity    | refractory 占满而 canonical 有空位          | 精确归因 `REFRACTORY`                                    |
| Capacity    | canonical demand >K                   | 精确归因 `TRUE_CAPACITY`                                 |
| Capacity    | 同 bin old end / new birth             | 验证当前 same-bin ordering                               |
| Capacity    | full R0 exhaustion                    | optimizer 前 rollback                                 |
| Capacity    | fixed/rematch policy equality         | exact config equality                                |
| State       | all-grad masked gold vs R1            | logits/state/lifecycle exact within frozen tolerance |
| Gradient    | selected 在 block 第 1 bin              | graph 只覆盖 1 token                                    |
| Gradient    | selected 在 block 第 64 bin             | graph 覆盖完整 64 bins                                   |
| Gradient    | block 无 selected bin                  | 全 block no-grad，state 与 gold 一致                      |
| Gradient    | selected 后有 tail                      | tail no-grad，final state 一致                          |
| Gradient    | selected 跨两个 blocks                   | 两个 global boundaries 正确 detach                       |
| Gradient    | all bins selected、coeff=1/T           | gradient 与 R0 exact                                  |
| Sampling    | exhaustive toy enumeration            | (E[a_t]=1/T)                                         |
| Sampling    | overlapping windows                   | multiplicity weighting，不用 union weight               |
| Sampling    | empty component                       | fallback uniform，support 非零                          |
| Sampling    | edge H=8                              | anchor 始终在 window 内                                  |
| Sampling    | (T<8)                                 | 合法 full-support coefficients                         |
| Denominator | realized coefficient sum ≠1           | 不做 SNIPW normalization                               |
| RNG         | dropout=0.1，grad/no-grad切换            | RNG tail、logits、state一致                              |
| RNG         | CPU/CUDA RNG digest mismatch          | fail closed                                          |
| Staleness   | 注入旧 parameter-version state           | reject                                               |
| GT taint    | GT slot、future endpoint、EOF进 meta     | reject                                               |
| Leakage     | video A state进入 video B               | reject                                               |
| Leakage     | supervision maps跨视频                   | reject                                               |
| Buffer      | mutable buffer改变                      | rollback                                             |
| Transaction | block backward失败                      | whole-video rollback                                 |
| Transaction | optimizer/scheduler/commit失败          | all mutable state restored                           |
| Numerical   | selected loss non-finite              | invalid video                                        |
| Numerical   | unselected bin logits non-finite      | 仍然 invalid，不能因无 loss 忽略                              |
| Lifecycle   | same-class overlap                    | canonical ownership正确                                |
| Lifecycle   | action >192 bins                      | 与 context truncation无关                               |
| Lifecycle   | old four samples                      | development-only，不计新 G0                              |
| Manifest    | draw reorder/substitution             | reject                                               |
| Manifest    | commit/config/data drift              | reject                                               |
| Workload    | token/graph counters                  | 与实际执行精确一致                                            |
| DDP         | world_size>1                          | fail closed，当前 unsupported                           |
| Reporting   | locked‑211 / canonical‑213 mismatch   | formal blocked                                       |

特别要求：

> Exactness tests 必须使用真实 Q2 风格的 `dropout=0.1`、thresholds `0.5`、refractory `2`。不能继续只用 `dropout=0`、threshold `1.1` 的 toy head 避开真实 lifecycle crossing。

现有 CRS 主要 toy detector 确实使用 `dropout=0` 和 birth/alive/end threshold `1.1`；它不能证明真实随机初始化 Q2 的 capacity 安全。

---

# V. Preregistered Experiment Ladder

## V.0 Development-only diagnosis

旧四个 G0 samples 只允许用于：

```text
capacity root-cause debugging
first-divergence tracing
R1 regression development
```

禁止用于：

```text
选择 margin
选择 M/H
选择 checkpoint strata
选择新 holdout
confirmatory PASS
```

## V.1 Capacity gate

执行完整零 GPU capacity audit：

```text
160 fit-core videos
× seeds 705,706,707 init checkpoints
× actual lifecycle
× same-logits counterfactual policies
```

冻结一个且仅一个 shared capacity/lifecycle contract。

若找不到合格合同：

```text
Q2_STATUS=INVALID
R1=KILL
```

## V.2 Outcome-blind micro-trajectory

因为 accepted chronological trajectory 不存在，必须专门生成 checkpoint strata。

### Preferred policy

* 从 fit-core 选择 16 个 `trajectory-core` videos；
* selection 仅使用 metadata/stress flags；
* 与旧四个 G0 videos及新 holdout完全不重叠；
* 使用一个预冻结的 chronological carrier arm 生成 parameter states；
  -同一 checkpoint bytes 加载到 fixed/rematch 两个模式；
* checkpoint carrier 本身不作为质量证据。

对 seeds `705,706,707` 生成：

```text
init  = 0 optimizer events
early = 4
mid   = 8
late  = 16
```

共 12 个 parameter states。

`late` 必须称：

```text
micro-trajectory late
```

不得称正式 late checkpoint。

### Predeclared bounded fallback

若在任何 fidelity outcome 可见前触发预注册 CPU resource cap：

```text
seed 705: 0,4,8,16
seed 706: init only
seed 707: init only
```

fallback 条件必须预签，不能看结果后减少 strata。

## V.3 Unseen holdout selection

从剩余：

```text
fit-core videos × 4 draws
```

中选 8 个 unseen samples：

* 每 length quartile 2 个；
* 排除旧四个 G0 samples；
* 排除 trajectory-core videos；
* 最大化以下 flags union：

  ```text
  detach-boundary crossing
  action >192
  overlap
  same-class repetition
  true-left-censor geometry
  capacity pressure
  refractory-adjacent birth
  dense start/end crossing
  ```
* 再最大化 proposal diversity；
* 最后 lexicographic tie-break。

即使 R1 不再 replay，这些 flags 仍覆盖最难的 state/lifecycle/gradient 区域。

## V.4 G0-A：implementation exactness

对 fixed 和 rematch **分别**执行。

同 checkpoint、video、RNG、coefficients：

### Gold

```text
full chronological forward
all selected block prefixes fully graphed
loss only at selected bins
```

### Candidate

```text
R1 no-grad on unselected blocks and post-selected tails
same selected losses
```

Hard requirements，逐 sample、逐 checkpoint：

```text
all logits max error <= frozen 1e-6 tolerance
continuous runtime max error <= 1e-6
discrete runtime exact
canonical lifecycle exact
birth assignments exact
endpoint ownership exact
selected loss relative error <= 1e-6
all-trainable gradient cosine >= 0.999999
relative gradient L2 error <= 1e-6
gradient sign agreement >= 0.999999
RNG final digest exact
```

这些是实现闭包，不允许用 aggregate mean 抵消单例失败。

## V.5 G0-B：finite-M estimator adequacy

比较：

```text
R0 all-bin chrono-64 gradient
vs
R1 M=4 HH gradient
```

在每个 checkpoint stratum内，对 8 个 holdout videos按 video-uniform规则聚合。

fixed/rematch 均必须通过：

```text
gradient cosine >= 0.90
gradient sign agreement >= 0.90
relative aggregate loss error <= 0.10
finite gradient norm ratio within preregistered interval
```

这些是 resource/fidelity safety gates，不是论文 significance threshold。

## V.6 G0-C：capacity

任一：

```text
seed
checkpoint
sample
fixed/rematch
gold/candidate
```

出现：

```text
slot_exhaustion > 0
```

立即 terminal KILL。

即使 gold/candidate 同时 exhaustion，也不能相互抵消。

## V.7 Family-wise rule

```text
PASS iff:
  all G0-A hard invariants pass
  AND every checkpoint stratum passes G0-B
  AND zero slot exhaustion
  AND no taint/RNG/hash/rollback violation
```

一项失败即 terminal `KILL`。

## V.8 Evidence chain

```text
new implementation commit
→ local B0
→ target-Linux B0
→ same independent reviewer PASS / NEXT_GATE=G0
→ signed checkpoint/selection/margins/config/data hashes
→ unseen CPU-only G0
```

只有 terminal G0 `PASS` 后：

```text
PROFILE may be requested
```

## V.9 Fixed-step profile

R0 与 R1 必须共享：

```text
same 8 warmup videos
same 32 measured videos
same order
same checkpoint
same feature bytes
same BF16
same hardware
same one-video optimizer event
```

Profile 仅证明资源，不证明质量。

## V.10 One-seed mechanism kill

G0 与 profile 均 PASS 后，先只运行：

```text
seed 705
R1 fixed
vs
R1 rematch
```

完整 one-token chronological evaluation。

STOP if：

* fixed 不优；
* gain 不落在 identity errors；
* recall/FN/end delay恶化；
* exhaustion；
  -超预算；
* gain来自少发射。

## V.11 Novelty-killer controls

只有 one-seed fixed/rematch positive 后才运行：

```text
simple GRU recurrent control
FRESH
faithful Temporal TrackFormer
```

仍为 one-seed screen。

## V.12 Scientific escalation

```text
1 seed = mechanism/resource kill
3 paired seeds = low-cost route confirmation
5 paired seeds + dense/overlap dataset = minimum retained paper claim
```

三种子前不得实现 raw-video LoRA。

---

# W. Cost Table

## W.1 Per-video theoretical workload

| Route                 | Forward tokens | Recomputed forward | Backward graph tokens |    Supervised exposures | Optimizer events | Peak graph horizon |
| --------------------- | -------------: | -----------------: | --------------------: | ----------------------: | ---------------: | -----------------: |
| R0                    |          (T_v) |                  0 |                 (T_v) |                   (T_v) |                1 |                 64 |
| **R1**                |          (T_v) |                  0 |          (P_v\le T_v) | 32 repeated, ≤32 unique |                1 |                ≤64 |
| R2                    |          (T_v) |              (P_v) |                 (P_v) |                      32 |                1 |                ≤64 |
| R3                    |           名义可低 |              依刷新策略 |              依 suffix |                       1 |                1 |  bounded but stale |
| Activation checkpoint |         (>T_v) |           positive |                 (T_v) |                   (T_v) |                1 |       lower memory |
| R5                    |          (T_v) |                  0 |                 (T_v) |                   (T_v) |                1 |                 64 |

其中：

[
P_v=\sum_jp_{v,j}.
]

R1 是否有实际价值取决于：

[
P_v/T_v
]

以及 backward 在真实 wall time 中的占比。当前无 profile，不能声称具体加速比。

## W.2 Mandatory denominators

每个 profile artifact 必须记录：

```text
exact video IDs
temporal forward tokens
temporal backward graph tokens
selected blocks
repeated supervised exposures
unique supervised bins
HH weight sum
HH weight squared sum
ESS
optimizer events
data wait seconds
controller/supervision CPU seconds
forward CUDA seconds
backward CUDA seconds
optimizer seconds
wall seconds
peak VRAM
GPU-hours
precision
hardware identity
```

禁止只报告：

```text
optimizer events/sec
episodes/sec
selected bins/sec
```

## W.3 Resource caps

| Stage                             | GPU-hour cap | Permission                   |
| --------------------------------- | -----------: | ---------------------------- |
| Capacity/lifecycle audit          |            0 | CPU allowed                  |
| R1 implementation/tests           |            0 | CPU allowed                  |
| B0/review/new G0                  |            0 | CPU allowed                  |
| Fixed-step profile                |           ≤2 | BLOCKED until G0 PASS        |
| One-seed fixed/rematch train+eval |           ≤4 | BLOCKED until profile review |
| Remaining pre-Stage2 reserve      |           ≤4 | not automatically authorized |
| Total pre-Stage2                  |          ≤10 | hard cap                     |
| Raw-video Stage 2                 |         none | BLOCKED                      |

任一阶段超限：

```text
STOP_AND_REVIEW
```

不得从其他阶段静默借预算。

## W.4 支持“降低训练成本”的最低证据

必须同时满足：

1. R1/R0 同视频、同 checkpoint、同硬件；
2. R1 forward tokens 与 R0相同且账目闭合；
3. backward graph tokens和 backward CUDA time下降；
4. total wall time和 GPU-hours的 paired CI均显示改善；
5. G0 estimator fidelity PASS；
6. complete chronological quality不劣；
7. gain不是 cache heat、少处理视频或少 optimizer events造成；
8. protocol/G0 固定成本被披露。

若只减少 backward tokens，而 wall time/GPU-hours不下降：

```text
R1 may remain a memory tool
EFFICIENT-TRAINING CLAIM=KILL
```

---

# X. Claim Map and Reviewer Risks

## X.1 Claim map

| Claim                       | 当前状态                       | 当前允许表述                                     | 所需证据                          | Falsifier                     |
| --------------------------- | -------------------------- | ------------------------------------------ | ----------------------------- | ----------------------------- |
| `70df86e` G0 是有效 KILL       | protocol-supported         | 当前 empty-state dynamic_birth失败             | 已有 signed chain               | 独立复算发现证据错误                    |
| 当前 empty-state CRS-EPS结构不忠实 | supported                  | omitted history改变 state/lifecycle/gradient | 已有 G0                         | 不得复活该 route                   |
| 所有 event-centric训练均失败       | unproven                   | 不允许                                        | 新 state-faithful route        | R1 PASS 即反例                   |
| 当前 Q2 是 primary trainer     | unproven                   | 仅 reference candidate                      | capacity+G0+profile           | 任一 exhaustion                 |
| R1 numerical state faithful | requires experiment        | 设计目标                                       | G0-A                          | logits/state mismatch         |
| R1 对 chrono‑64 gradient无偏   | partially identifiable     | 数学上在合同条件下成立                                | probability tests + G0-B      | propensity/RNG/graph mismatch |
| R1 降低 GPU-hours             | requires experiment        | 不允许预先声称                                    | matched profile               | wall/GPU-hour无改善              |
| fixed优于 rematch             | requires experiment        | 未知                                         | one/three/five seed           | fixed不优                       |
| gain是 identity-linked       | not identifiable           | 未知                                         | identity metrics + controls   | gain仅aggregate/少发射            |
| R1 是论文创新                    | strong reconstruction risk | 暂定 enabling infrastructure                 | 不可被普通 selective backward重构    | prior family足以解释              |
| cached Q2 positive授权 LoRA   | false                      | 不自动授权                                      | 3 seeds+novelty+cost          | 任一 gate失败                     |
| Full PETAL 超越多论文重构          | unproven                   | 不允许 headline                               | FRESH/TTF/raw 2×2/second data | tracking-style control匹配      |

项目当前 closest-work map 已明确列出 MATR、HAT、ActionSwitch、E2E‑LOAD、StreamFormer、离线 E2E‑TAD/ETAD 和 TrackFormer/online VIS 等重构压力。

## X.2 最强拒稿理由

即使 R1 通过，Full PETAL 仍可能不够成为论文主方法：

1. **R1 可能只是已有 selective-backward 思路的 On‑TAD 工程化。**
2. **它不减少完整 forward。** Python `head.step`、attention、GRU、lifecycle bookkeeping 仍处理所有 tokens。
3. **Q2 capacity 目前不安全。** 模型预测的 hard lifecycle决定 GT birth 是否获得监督 slot。
4. **扩槽可能只是掩盖 false birth/refractory。**
5. **fixed 的收益可能只是 label-assignment 稳定化，不是 persistent identity。**
6. **duplicate下降可能由减少 birth/emission造成，并伴随 recall collapse。**
7. **THUMOS14 与 locked‑211 边界不足以支持一般性结论。**
8. **FRESH、Temporal TrackFormer 或简单 GRU 可能重构增益。**
9. **cached-feature positive 不能外推 raw-video LoRA。**
10. **训练成本控制本身不是自动成立的论文创新。**

因此本轮选择 R1 的含义是：

> 给唯一一个 state-faithful、数学可闭合、能够被廉价杀死的 surrogate 一次实现机会。

不是：

> Full PETAL 已获得方法级 GO。

## X.3 Immediate kill rules

任一发生即终止：

```text
capacity audit无合法shared policy
任何新G0 slot exhaustion
R1 exact implementation不闭合
M=4 aggregate gradient fidelity失败
matched profile无GPU-hour改善
fixed不优于rematch
identity endpoint无改善
recall/FN/end delay退化
FRESH/TTF/GRU重构收益
第二数据集不复现
raw LoRA无独立增益
```

---

# Y. Ordered Next Actions

## 1. 冻结 Round‑2 decision

新增纯文档 decision：

```text
CURRENT_EMPTY_STATE_CRS_EPS=permanently killed
SELECTED_SUCCESSOR=R1 CSFSB
Q2=gold/reference candidate
CAPACITY=blocking
GPU_BEFORE_NEW_G0_PASS=0
```

不得修改旧 G0 samples、margins 或 terminal result。

## 2. 首个 implementation commit：capacity audit only

精确范围：

```text
ADD opentad/utils/q2_capacity_audit.py
ADD tools/audit_q2_capacity_lifecycle.py
ADD tests/test_q2_capacity_lifecycle.py
MODIFY train_engine.py:
    full chronological exhaustion fail-closed
MODIFY launch/result route status:
    old killed CRS cannot profile
```

不得在该 commit 改：

```text
K
thresholds
refractory
binding
loss
optimizer
sampling
```

## 3. 运行完整零 GPU capacity census

```text
160 fit-core videos
× seeds 705/706/707 init checkpoints
× actual controller
× same-logits counterfactual policies
```

生成 source-bound、non-overwritable artifact。

## 4. 冻结一个 shared lifecycle contract

根据审计只选一项最小修订：

```text
K4 unchanged
or
data-derived minimal K
or
outcome-blind prior-bias initialization
or
refractory/same-bin ordering correction
```

fixed/rematch必须共享。

找不到合法方案则终止 Q2。

## 5. 实现 R1

在新的 scientific commit 中：

* R0 untouched；
* old CRS read-only；
* full-forward selective-backward；
* exact full-video state；
* block-prefix graph；
* one-video optimizer event；
* multi-denominator counters。

## 6. 完成 CPU test matrix

必须包含真实：

```text
dropout=0.1
thresholds=0.5
refractory=2
same-class overlap
long action
block edges
RNG parity
state exactness
gradient exactness
rollback
```

## 7. 生成 exact-commit B0

```text
local B0
target-Linux B0
clean checkout
external non-overwritable evidence
```

## 8. 同一独立 reviewer 复核

必须返回：

```text
PASS
NEXT_GATE=NEW_HOLDOUT_G0
PROFILE=BLOCK
FORMAL=BLOCK
```

## 9. Preregister micro-trajectory 与 unseen G0

冻结：

```text
trajectory-core IDs
8 unseen holdout samples
preferred 12 checkpoint states
predeclared fallback
M=4
H=8
mixture
capacity policy
margins
family-wise rule
commit/config/data/checkpoint hashes
```

## 10. 执行 terminal CPU G0

```text
G0-A exact implementation
G0-B finite-M estimator fidelity
G0-C zero exhaustion
```

任何一项失败：

```text
R1=KILL
PROFILE=BLOCK
```

## 11. G0 PASS 后才申请 profile

必须新 ticket、新 review、≤2 GPU-hours。

## 12. Profile PASS 后才申请 one-seed mechanism kill

只运行：

```text
seed 705
fixed vs rematch
```

不恢复三种子矩阵。

## 13. Escalation order

```text
one-seed positive
→ FRESH / TTF / GRU novelty killers
→ three paired seeds
→ five paired seeds + dense/overlap dataset
→ renewed novelty review
→ only then discuss raw-video LoRA
```

---

```text
CURRENT_G0=VALID_KILL
CURRENT_EMPTY_STATE_CRS_EPS=STRUCTURAL_KILL
Q2_STATUS=GOLD_REFERENCE_CANDIDATE
Q2_CAPACITY_GATE=BLOCK
SELECTED_ROUTE=R1
IMPLEMENTATION=ALLOW
NEW_HOLDOUT_G0=REQUIRED
PROFILE=BLOCK
FORMAL=BLOCK
GPU_HOURS_BEFORE_NEW_G0_PASS=0
CONFIDENCE=0.95
```
