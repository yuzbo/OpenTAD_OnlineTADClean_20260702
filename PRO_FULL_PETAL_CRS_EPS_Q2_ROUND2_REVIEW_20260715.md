以下按作者冻结的 Q1–Q12 决策和原 Prompt 的 Round‑2、A–Y 输出合同执行。任何 `GO` 都只表示“允许实现并做可证伪验证”，不构成 GPU、formal training 或 raw-video Stage‑2 授权。

# A. Executive Verdict

## 总裁决

```text
RESEARCH_VERDICT=GO-HYBRID-PROTOCOL
```

这是**有条件的协议级 GO**：

> 实现 CRS‑EPS 作为候选主训练协议；以预注册的小型 full-stream subset 做 state/loss/gradient gold audit；最终质量只在完整 chronological streaming evaluation 上判断。

它不表示：

* CRS‑EPS 已实现；
* fixed binding 有效；
* Full PETAL 具有论文级整体创新；
* 当前 cache 可支持 strict raw-video causality；
* 当前可以 profile；
* 当前可以训练；
* 当前可以进入 LoRA Stage 2。

## 对作者决策的接受与两项必要修正

作者的 Q1–Q12 决策总体合理，但必须增加两项限定。

### 修正 1：dynamic replay 不是天然的 full-stream state reconstruction

从 earliest relevant observable birth 向前 replay，能够重建与当前 supervised suffix 相关的 **GT canonical lifecycle**；它不能保证重建完整的**模型 runtime state**。早期背景、已结束动作、错误 birth、错误 active slot、refractory history 和长期 query/memory 都可能影响 episode entry state。

因此：

```text
dynamic replay = causal surrogate
video-start replay = gold reference
importance weighting ≠ state-bias correction
```

若 dynamic replay 的 state/loss/gradient 与 video-start replay 不一致，IPW 再精确也不能修复该偏差。

### 修正 2：长期训练后“实际 birth/mask 完全相同”不是合理要求

fixed 与 rematch 从相同权重开始时，局部直接干预只在 post-birth loss binding；但一次更新后两臂权重会分化，后续 logits、runtime slot status、available slots、birth assignments 和 masks 可能随之分化。这些差异可能是 treatment 的**下游中介效应**，不能简单称为外生 confound。

严格 Q2 应分别证明：

1. **局部直接效应**：同一权重、同一 state、同一 episode、同一 RNG 下，唯一直接变化是 `loss_bindings`。
2. **纵向政策效应**：独立训练后产生的 lifecycle/mask 差异必须完整记录，并证明最终收益确实落在 identity-linked failure，而非仅由更保守的 occupancy、birth 抑制或 recall 下降造成。

## 最终路线定位

| 对象                       | Round‑2 定位                                                                                    |
| ------------------------ | --------------------------------------------------------------------------------------------- |
| Full chronological route | 当前 cached-token、video-uniform、per-video-mean、64-token detach objective 的 exhaustive reference |
| CRS‑EPS                  | 必要但低新颖性的训练成本 surrogate                                                                        |
| Q2                       | fixed post-birth supervision policy vs prefix rematching policy                               |
| Full PETAL               | 仍属于 multi-paper reconstruction 风险                                                             |
| fixed binding            | 未验证的 marginal mechanism                                                                       |
| 推荐协议                     | HYBRID                                                                                        |
| 当前 profile               | BLOCKED                                                                                       |
| 当前 formal training       | BLOCKED                                                                                       |
| Raw-video Stage 2        | BLOCKED                                                                                       |

---

# B. Repository and Commit Verification

## B.1 当前分支状态

本轮再次解析 `codex/full-petal-implementation`：

```text
Observed branch HEAD:
3a297344df73d994c54cac082f34a66e597e9d44

HEAD message:
docs: absorb CRS-EPS Q2 Round 1 review

Immutable scientific anchor:
f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb
```

`f4ea53e..HEAD` 目前为两个文档提交：增加 Round‑1 review、absorption、author-response draft、Prompt 和 wiki 决策记录；没有修改模型、数据集、训练 engine、配置、测试协议或结果。故本轮科学代码事实仍固定为 `f4ea53e`。

## B.2 当前可接受的证据状态

| 项目                        | 状态                                          |
| ------------------------- | ------------------------------------------- |
| Anchor code 可访问           | `[CODE-VERIFIED]`                           |
| HEAD 仅为文档变化               | `[CODE-VERIFIED]`                           |
| CRS‑EPS 实现                | `[CODE-VERIFIED] NO`                        |
| 正式 GPU profile            | `[DOC-VERIFIED] NONE`                       |
| 正式 Q2 result              | `[DOC-VERIFIED] NONE`                       |
| Raw-video result          | `[DOC-VERIFIED] NONE`                       |
| 旧 B0 508 tests            | `[ARTIFACT-ASSERTED]`，未覆盖真实 submit-shell P0 |
| 当前 `PROFILE=ALLOW`        | 已撤销                                         |
| 当前 scientific anchor 能否启动 | 不能                                          |

项目 SSOT 明确记录：截至 2026‑07‑15，没有 Q2 fixed/rematch、CRS‑EPS、LoRA 或 full visual tower 的正式多种子结果，也没有被接受的 fixed-step profile。

---

# C. Evidence Table

| ID  | 事实                                                                          | 标签                   | Round‑2 含义                                      |
| --- | --------------------------------------------------------------------------- | -------------------- | ----------------------------------------------- |
| E1  | 当前 dataset 完整枚举每个选定视频的所有 cache tokens                                       | `[CODE-VERIFIED]`    | 不是 episode sampling                             |
| E2  | 当前每完整视频产生一个 optimizer event                                                 | `[CODE-VERIFIED]`    | 不是“低成本短 episode event”                          |
| E3  | 每个 token 进入 sequential `head.step` 和 loss path                              | `[CODE-VERIFIED]`    | 减少 optimizer steps 没有消除主要 temporal computation  |
| E4  | cross-chunk numerical state carried，但 tensor 被 detach                       | `[CODE-VERIFIED]`    | forward history 连续；跨 chunk credit assignment 截断 |
| E5  | 当前目标是 video-uniform per-video token mean                                    | `[CODE-VERIFIED]`    | 作者 Q1 选择正确地保持了 target risk                      |
| E6  | fixed/rematch 静态配置只改变 binding mode                                          | `[CODE-VERIFIED]`    | 局部干预较干净                                         |
| E7  | synthetic probe 证明 rematch 只交换 loss binding，并改变对应 gradient                  | `[CODE-VERIFIED]`    | 尚非长期训练 one-factor 证书                            |
| E8  | CRS‑EPS 文档有 burn-in、event mixture、IPW/ESS，但代码没有                             | `[DOC-VERIFIED]`     | 必须新实现                                           |
| E9  | ticket/runtime `work_dir` identity 仍不闭包                                     | `[CODE-VERIFIED]`    | P0 继续阻塞 profile                                 |
| E10 | extractor provenance 不含完整 commit/config/checkpoint/raw support              | `[CODE-VERIFIED]`    | strict raw-video causality 未证明                  |
| E11 | 当前 emission clock 是 source-frame clock                                      | `[CODE-VERIFIED]`    | 不能称 wall-clock latency                          |
| E12 | ETAD 已研究 selective snippet gradients 和 proposal sampling                    | `[PRIMARY-VERIFIED]` | CRS‑EPS 不能把“高效采样”本身作为主创新                        |
| E13 | TALLFormer、Re²TAL、TIA/AdaTAD、LoSA 已覆盖多种长视频高效适配方案                            | `[PRIMARY-VERIFIED]` | Raw-video 高效适配也不是空白                             |
| E14 | TrackFormer、MOTR、MeMOTR、在线 VIS 已建立 persistent query/identity propagation    | `[PRIMARY-VERIFIED]` | persistent query 本身不新                           |
| E15 | MATR、HAT、ActionSwitch 已覆盖历史 memory、anchor history、overlap/same-class On-TAL | `[PRIMARY-VERIFIED]` | Full PETAL 整体重构攻击很强                             |
| E16 | 2026 年 OnPoint 与 OZ‑TAL 分别改变监督形式或类别泛化任务                                     | `[PRIMARY-VERIFIED]` | 它们是邻近竞争者，不是当前 fully supervised Q2 的直接替代         |

当前 detector 在每个 token 上构造 loss，视频末再做一次 optimizer update；`_optimizer_weight` 是 valid-token 数，engine 最终按整个视频的 token 总数归一化累计梯度。

ETAD 明确采用 sequentialized gradient sampling，并仅对部分 snippets 做 encoder backward，同时采样部分 proposals；论文还报告 label-guided sampling并不自然优于简单随机/网格方案。([arXiv][1])

TALLFormer 通过 long-term feature memory 避免每次重算全部视频；Re²TAL 使用 reversible backbone；TIA/AdaTAD 与 LoSA 则通过轻量 adapter 扩展长视频端到端适配。([arXiv][2])

---

# D. Unknown Register

作者已关闭主要概念决策。剩余未知项不是继续发散问题，而是后续 gate 所需的证据。

| ID  | 未知项                                                                | 阻塞级别                   | Conservative default                             |
| --- | ------------------------------------------------------------------ | ---------------------- | ------------------------------------------------ |
| U1  | 旧 cache 的 extractor commit/config/checkpoint/processor/raw support | effectiveness run 前 P1 | 无法恢复则重建                                          |
| U2  | locked‑211 与 canonical‑213 的两个差异视频及理由                              | formal reporting 前 P1  | 只称 `locked-211 subset`                           |
| U3  | dynamic replay 与 full-state 的实际偏差                                  | scientific run 前 P1    | 先做 G0；不假设等价                                      |
| U4  | 最小可接受 quality non-inferiority margin                               | scientific run 前 P1    | 由作者在 gold variability 与 cost profile 后签署，不得从结果反推 |
| U5  | CRS 每视频 episode draws (M)                                          | scientific run 前 P1    | 由 outcome-blind G0 precision/power 选择            |
| U6  | Full vs CRS 实际 GPU-hour ratio                                      | profile 后才知道           | 不声称节省                                            |
| U7  | FineAction 许可与实际可访问状态                                              | paper claim 前          | 不假设可用                                            |
| U8  | MultiTHUMOS annotation 是否保留足够的 instance identity                   | paper claim 前          | 仅作为 dense multi-label screening                  |
| U9  | Q2 fixed 是否有任何有效性优势                                                | experiment             | UNKNOWN                                          |
| U10 | Full PETAL surviving delta 是否能躲过 multi-paper reconstruction        | paper claim 前          | 默认为 reconstruction                               |

以下不是当前 blocker：

* multi-rank DDP；
* formal resume；
* full-tower finetuning；
  -大规模第二数据集训练；
  -原始视频 Stage‑2 超参数搜索。

---

# E. Fixed Task and Scope Audit

## E.1 固定任务

任务继续固定为：

[
\mathcal F_t
=\sigma(x_{\le t},R_{<t},D_{<t})
]

模型在时刻 (t) 只能使用当前及过去可获得信息，输出：

[
d_j=(\hat s_j,\hat e_j,\hat c_j,\hat p_j,\tau_j),
\qquad
\hat e_j\le \tau_j
]

且 emission 一旦公开不可撤回或删除。

## E.2 训练 annotation 的允许边界

允许：

* 用完整训练 annotation 决定 episode anchor；
* 用 annotation 构造 training-only prefix schedule；
* 计算 inclusion probability；
* 标记 start/end/ongoing/hard-background proposal component；
* 构造 gold audit。

禁止：

* 把 GT instance ID 写入 runtime state；
* 用 GT 初始化 active slot、query 或 hidden state；
* 把未来 endpoint 放入模型 metadata；
* 推理时加载 episode manifest 或 prefix schedule；
* 把 annotation-guided sampling写成“训练完全不使用未来监督”。

## E.3 当前 end-to-end 范围

当前最准确描述是：

```text
frozen cached visual tokens
+ trainable causal temporal/persistent detector
+ 64-token chunk-internal BPTT
+ detached cross-chunk numerical state
```

`detach_stream_state=True` 将 queries、memory、slot status、start/class state全部 detach 后交给下一 chunk。

因此：

```text
CACHED-TOKEN CAUSALITY=SUPPORTED
RAW-VIDEO CAUSALITY=UNPROVEN
FULL-VIDEO BPTT=FALSE
RAW-VIDEO END-TO-END=FALSE
```

## E.4 Q2 的可主张对象

Q2 只能主张：

> 在共享数据、模型、birth algorithm、canonical lifecycle、训练预算和 evaluator 的条件下，固定 post-birth supervision policy 是否比 per-prefix rematching policy 更适合学习 identity-linked interval trajectories。

不能主张：

* persistent query 是新概念；
* raw-video On-TAD 已解决；
* Full PETAL 是统一新方法；
* CRS‑EPS 是论文主算法；
* positive Q2 自动授权 Stage 2。

---

# F. Current Code Dataflow and Cost Accounting

## F.1 当前执行链

```text
locked split + cached feature manifest
    ↓
完整视频 token enumeration
    ↓
64-token chronological chunks
    ↓
跨 chunk numerical state
    ↓
每 token Python head.step
    ↓
birth / alive / class / start / end losses
    ↓
每 chunk backward
    ↓
video end:
gradient normalization
optimizer.step
scheduler.step
state transaction commit
```

Dataset 将每个视频的所有 source frames 分成连续 chunks，并声明每个完整视频对应一个 optimizer event。

## F.2 当前 target risk 与 surrogate gradient

作者已正确冻结：

[
L_{\mathrm{target}}(\theta)
===========================

\frac1V\sum_{v=1}^{V}
\frac1{T_v}\sum_{t=1}^{T_v}
\ell_{v,t}(\theta).
]

但必须同时冻结当前 64-token detach 的梯度算子。定义：

[
g_{\mathrm{target}}(\theta)
===========================

\frac1V\sum_v\frac1{T_v}
\sum_t
\nabla_{\theta}^{(B=64,\operatorname{detach})}
\ell_{v,t}(\theta).
]

这意味着：

* CRS‑EPS 的 IPW 可以使**loss exposure**无偏；
* 只有在 episode state 与 full state 一致、且复现同一 global chunk detach 位置时，gradient 才逼近当前 full reference；
* 仅比较 sampled loss，不足以证明 sampled training 等价。

## F.3 成本分母

| Unit                     | Full chronological |                          CRS‑EPS v1 | 是否可直接比较          |
| ------------------------ | -----------------: | ----------------------------------: | ---------------- |
| video groups             |      1/video/epoch |                       1/video/epoch | 可以作为辅助           |
| episodes per video       |      1 full stream |                (M) sampled episodes | 不可               |
| temporal forward tokens  |              (T_v) |         所有实际 replay + suffix tokens | 必须报告             |
| temporal backward tokens |    所有 chunk tokens | global-chunk-aware gradient section | 必须报告             |
| supervised bins          |              (T_v) |                     (M\times H)，含重叠 | 必须加 weights/ESS  |
| visual forward frames    |   0 during Stage 1 |                    0 during Stage 1 | 可以               |
| visual backward frames   |                  0 |                                   0 | 可以               |
| optimizer events         |            1/video |                   建议仍为1/video group | 辅助，不是 primary    |
| GPU-hours                |                 未测 |                                  未测 | primary resource |
| wall time                |                 未测 |                                  未测 | co-primary       |
| ESS                      |                未报告 |                                必须报告 | CRS 专属           |
| state fidelity           |        exact route |                           surrogate | 必须由 G0 审计        |

---

# G. Is CRS-EPS Actually Implemented?

```text
CRS_EPS_IMPLEMENTED=NO
```

当前代码没有：

* mixture proposal；
* sampled anchor；
* immutable epoch-indexed episode manifest；
* fixed supervised horizon；
* dynamic replay；
* gold state audit；
* exact per-bin coverage probability；
* IPW/HH estimator；
* SNIPW diagnostic；
* ESS；
* full-vs-sampled gradient audit；
* video-group sampled optimizer boundary。

历史文档虽然定义了 192 context、4–8 supervised bins 和 20/15/10/15/40 mixture，但这是设计文档，不是实现。

Round‑2 冻结的 v1 目标是：

```text
H = 8 fixed consecutive supervised bins
alpha_uniform = 0.40
alpha_start = 0.15
alpha_end = 0.20
alpha_ongoing = 0.10
alpha_hard_background = 0.15

M = selected outcome-blind after G0,
then frozen before Q2 effectiveness training.
```

固定 (H=8) 而不是运行时从 4–8 随机变化，目的是减少 inclusion probability、loss denominator 和成本比较中的额外自由度。

---

# H. P0–P3 Code and Protocol Findings

## H.1 P0 — `P0-LAUNCH-WORKDIR`

### 状态

```text
OPEN
PROFILE-BLOCKING
FORMAL-BLOCKING
```

Ticket builder 先冻结 `cfg_overrides`；submit helper 随后生成 timestamped `RUN_DIR` 并添加 `work_dir=${RUN_DIR}/work`；validator 再要求 ticket runtime identity 与 live runtime identity 完全一致。

### 唯一可接受修复方向

```text
RUN_ID/RUN_DIR/WORK_DIR
must be frozen before ticket publication

Slurm argv
must consume ticket-bound overrides,
not independently rebuild them
```

必须新增真实 fake-`sbatch` shell integration test。

---

## H.2 P1 — CRS target-risk and probability contract

没有精确 (q,\rho,\pi,w) 就不得训练。

Fail-closed 条件：

* proposal distributions 不归一；
* 任一 target bin (\rho_{v,t}=0)；
* manifest 重算后的 probability/hash 不一致；
* 实际 suffix 与 manifest 不一致；
* dedup 后仍错误使用 exposure probability；
* 权重超过由 uniform floor 保证的 2.5 上界。

---

## H.3 P1 — State reconstruction consistency

Dynamic replay 只能是 surrogate。必须用 video-start gold audit验证：

* canonical supervision state；
* runtime discrete state；
* continuous query/memory；
* logits；
* losses；
* gradients；
* occupancy；
* replay cost。

若 state consistency 失败，IPW 不构成补救。

---

## H.4 P1 — Gradient-boundary consistency

CRS‑EPS 不能简单执行：

```text
192 forward-only
-> detach
-> 8-bin backward
```

因为它与当前 full route 的 64-token chunk-internal BPTT 不同。

必须复现原 global chunk boundaries：

[
g(t)=1+64\left\lfloor\frac{t-1}{64}\right\rfloor.
]

对 suffix 首 bin (u)：

* replay start 到 (g(u)-1)：forward-only；
* 从 (g(u)) 开始：gradient-enabled；
* 在每个原始 64-token boundary 重新 detach；
* loss 只作用于 supervised suffix。

---

## H.5 P1 — Longitudinal one-factor mediation

当前训练中 `available_slots` 来自模型预测的 runtime `slot_status`。两臂权重分化后，实际 birth masks 和 lifecycle 可能分化。

因此必须区分：

```text
direct local binding intervention
vs
downstream policy-mediated state divergence
```

若不做这一分解，C4“增益来自 identity binding”不能识别。

---

## H.6 P1 — Profile denominator

当前 profiler 主报告 `throughput_optimizer_events_per_second`，不适合 full video 与短 episode 跨协议比较。

需替换为多分母 contract。

---

## H.7 P1 — Feature provenance and real clocks

Cache loader 验证 array hash、source frames、stride、dtype 和 encoder ID，但没有绑定 immutable extractor implementation。

Ledger 中：

```text
emit_time_sec = emit_frame / fps
source_time_sec = source_frame / fps
```

是 source clock，不是 wall-clock publication time。

---

## H.8 P1 — Reporting population

真实 211/213 IDs 与原因仍 unknown。现有代码只证明系统能够验证 source-derived difference reasons，不证明真实差异已经关闭。

---

## H.9 P2 — Novelty

* persistent query：强 prior；
* causal streaming backbone：强 prior；
* long-memory On-TAL：强 prior；
* overlap handling：强 prior；
* efficient long-video training：强 prior；
* event sampling/IPW：低新颖性；
* fixed post-birth supervision binding：可能是剩余 marginal delta，但尚无结果。

---

## H.10 P3 — Python sequential unroll

当前 head 每 token 逐步执行 attention、GRU、decode 和 supervision transition。CRS‑EPS 可能通过减少 token 数降低成本，但不会自动解决 Python dispatch。

这应由 profiler 测量，不应提前重构为 prefix-parallel architecture；否则 Q2 会再次引入额外 treatment。

---

# I. Q2 One-Factor Identifiability Audit

## I.1 当前静态 intervention

固定与 rematch 配置删除 `trajectory_binding_mode` 和 `work_dir` 后完全一致。

Supervision state 中：

* shared birth assignment；
* shared canonical map；
* shared retirement；
* rematch 只重建 `loss_bindings`；
* canonical state 不被 rematch 改写。

Synthetic two-active test 证明：

* birth assignments 相同；
* canonical lifecycle 相同；
* masks 相同；
* rematch 只改变第二 prefix 的 loss binding；
  -对应 parameter gradient 发生变化。

## I.2 推荐的 Q2 识别层级

### Level 0：静态配置等价

必须 exact-equal：

* source config；
* model structure；
* data/cache；
* episode manifest；
* mixture；
* (q,\rho,w)；
* optimizer/scheduler；
* precision；
* seed；
* evaluator；
* thresholds。

### Level 1：局部 twin-forward direct effect

在同一 checkpoint、同一 runtime/supervision snapshot、同一 episode 和同一 token-wise RNG 下，同时计算 fixed/rematch：

允许的直接差异：

```text
loss_bindings
class/start/end target-to-slot association derived from loss_bindings
resulting losses and gradients
```

禁止的直接差异：

```text
input
source frames
birth candidates
birth assignments
canonical lifecycle
runtime decode
negative/ignore/censor semantics
loss denominator construction
```

### Level 2：纵向 paired policy effect

两臂独立更新后允许 weights、logits、runtime occupancy 和后续 lifecycle 分化，但必须记录：

* 首次 divergence 的 optimizer event；
* divergence 前直接 intervention；
* divergence 后 masks、birth、retire、refractory 的路径；
* 是否由 recall suppression 或低 birth rate造成“duplicate下降”。

### Level 3：identity-linked external effect

最终收益必须同时满足：

* standard mAP 不劣；
* recall/FN 不退化；
* completion delay 不恶化；
* duplicate/fragmentation 或 same-class/overlap 至少一个改善；
  -另一 identity endpoint 不恶化；
* improvement 不能只由发射更少预测获得。

## I.3 同一 episode list 是否足够？

**不够。**

还必须共享：

* replay policy；
* gradient boundaries；
* token-wise RNG；
* initial model/optimizer state；
* loss denominator；
* lifecycle/reset semantics；
* source arrays；
* video order；
* manifest multiplicities；
* profile workload。

## I.4 完整 paired trace schema

```text
schema_version
commit_sha
scientific_config_sha256
feature_cache_manifest_sha256
episode_manifest_sha256
model_initialization_sha256
optimizer_initialization_sha256

seed
epoch
video_group_index
video_id
episode_draw_index
episode_id
proposal_component
anchor_bin
supervised_range
replay_range
gradient_ranges
true_left_censored
dynamic_extension_reason
video_start_fallback

q_component
q_anchor_given_component
q_anchor_marginal
rho_by_supervised_bin
union_pi_by_supervised_bin
raw_weight_by_bin
final_weight_by_bin

rng_key
cpu_rng_digest_before_after
cuda_rng_digest_before_after
token_dropout_key_digest

runtime_state_digest_entry_exit
supervision_state_digest_entry_exit
query_norms_and_digest
memory_norms_and_digest
slot_status
refractory
start_state
score_state
label_state

birth_candidates
birth_assignments
canonical_bindings
loss_bindings
birth_mask
alive_mask
end_risk_mask
negative_mask
ignore_mask
censor_mask
endpoint_slots
retire_events
reset_events
slot_exhaustion
rematch_swaps

per_loss_numerator
per_loss_denominator
per_loss_value
weighted_total_loss
gradient_norm
gradient_digest
gradient_cosine_to_twin
optimizer_pre_post_digest
scheduler_pre_post_digest

inference_logits_digest
emission_digest
```

## I.5 RNG 合同

推荐采用 token-addressed RNG：

```text
rng_key = hash(
  global_seed,
  epoch,
  video_id,
  source_token_index,
  module_path,
  stochastic_op_index
)
```

这样 full replay 与 sampled replay 在相同 token 上获得相同 dropout mask，而不依赖此前执行了多少无关 token。

若不实现 token-addressed RNG，Q2 必须降级为：

```text
matched-distribution training-policy comparison
```

而不是严格 one-factor binding study。

## I.6 最终判定

```text
CURRENT_Q2_IDENTIFIABILITY=PARTIAL
POST-IMPLEMENTATION_POTENTIAL=CONDITIONALLY_IDENTIFIABLE
```

若 paired direct-effect trace无法闭包，建议名称改为：

> `Q2 Persistent Supervision Policy Comparison`

不得称为 pure fixed-binding ablation。

---

# J. CRS-EPS Mathematical Protocol

## J.1 Target risk

训练集视频为 (v=1,\ldots,V)，视频 (v) 有 (T_v) 个 decision bins。

[
p(v)=\frac1V,\qquad
p(t\mid v)=\frac1{T_v}.
]

目标：

[
L(\theta)
=========

\frac1V\sum_v
\frac1{T_v}\sum_{t=1}^{T_v}
\ell_{v,t}^{\mathrm{full}}(\theta).
]

其中 `full` 表示：

* video-start causal forward state；
  -当前 frozen cached tokens；
  -原 global 64-token detach boundaries；
  -现有 per-token loss definition。

## J.2 Proposal mixture

设：

[
z\in
{\mathrm{uniform,start,end,ongoing,hardbg}}
]

及：

[
\alpha=
(0.40,0.15,0.20,0.10,0.15).
]

每种 component 定义视频内 anchor distribution：

[
q_{v,z}(a),\qquad
\sum_{a=1}^{T_v}q_{v,z}(a)=1.
]

联合 proposal：

[
q_v(z,a)=\alpha_zq_{v,z}(a).
]

边际 anchor probability：

[
q_v(a)=\sum_z\alpha_zq_{v,z}(a).
]

若某视频不存在某 component 的候选 bin，则该 component 在该视频中回退为：

[
q_{v,z}(a)=1/T_v,
]

且 manifest 必须记录 `component_fallback_to_uniform=true`。

## J.3 Anchor definitions

令 (b_i) 为 instance (i) 的 first observable start-crossing bin，(d_i) 为 first observable end-crossing bin。

### Start component

候选集合：

[
\mathcal A_v^S
==============

\operatorname{unique}
{b_i+\delta:
\delta\in{-4,-2,-1,0,1,2,4}}
]

并截断到 ([1,T_v])。

### End component

[
\mathcal A_v^E
==============

\operatorname{unique}
{d_i+\delta:
\delta\in{-4,-2,-1,0,1,2,4}}.
]

### Ongoing component

候选 bin 满足：

* 至少一个 instance active；
* 不是 start/end crossing；
* 与任一 crossing 距离至少 2 bins。

### Hard-background component

候选 bin 满足：

* 当前没有 birth、active 或 end instance；
* 与任一 start/end crossing 距离不超过 4 bins。

这是 annotation-guided sampling，允许使用训练 annotation；该信息不得传入模型。

### Uniform component

[
q_{v,U}(a)=1/T_v.
]

## J.4 Supervised window

固定：

[
H=8.
]

对 anchor (a)，构造包含 anchor 的 8-bin window：

[
S_v(a)
======

[;a-3,;a+4;]
]

并整体平移到合法区间 ([1,T_v])。若 (T_v<8)，则使用整个视频并记录实际 (H_v=T_v)。

禁止：

* 随机改变 (H)；
* 在两臂使用不同 edge handling；
  -根据训练结果修改 (H)。

## J.5 One-draw inclusion probability

某 target bin (t) 被一条 sampled episode 覆盖的概率：

[
\rho_{v,t}
==========

\sum_z\sum_{a=1}^{T_v}
\alpha_zq_{v,z}(a)
\mathbf 1[t\in S_v(a)].
]

因为 uniform component 中 (a=t) 一定覆盖 (t)：

[
\rho_{v,t}\ge \frac{0.40}{T_v}.
]

因此 raw IPW：

[
w_{v,t}
=======

# \frac{p(t\mid v)}{\rho_{v,t}}

\frac{1/T_v}{\rho_{v,t}}
\le 2.5.
]

这使 v1 不需要真正的 clipping。

Fail-closed invariant：

```text
max(raw_weight) <= 2.5 + numeric_tolerance
```

若超过，说明 coverage probability 或 window construction 错误。

## J.6 Multiple episode draws and overlap

每视频每 epoch 独立、有放回采样 (M) 条 episodes：

[
E_{v,1},\ldots,E_{v,M}.
]

不对重叠 bins 去重。Hansen–Hurwitz 型 per-video estimator：

[
\widehat L_v^{HH}
=================

\frac1M
\sum_{m=1}^{M}
\sum_{t\in S(E_{v,m})}
w_{v,t}
\ell_{v,t}^{E_{v,m}}.
]

若满足 episode consistency：

[
\ell_{v,t}^{E}
==============

\ell_{v,t}^{\mathrm{full}}
\quad
\text{for every episode covering }t,
]

则：

[
\mathbb E[\widehat L_v^{HH}]
============================

\frac1{T_v}\sum_t\ell_{v,t}^{\mathrm{full}}.
]

总体 estimator：

[
\widehat L
==========

\frac1V\sum_v\widehat L_v^{HH}.
]

### Union inclusion probability

若只关心 (M) draws 后某 bin 是否至少出现一次：

[
\pi_{v,t}^{(M)}
===============

1-(1-\rho_{v,t})^M.
]

若实现选择对 unique bins 去重，才可使用：

[
\widehat L_v^{HT}
=================

\sum_{t:I_{v,t}=1}
\frac{1/T_v}{\pi_{v,t}^{(M)}}
\ell_{v,t}.
]

**v1 不推荐 dedup**，因为同一 bin 可能来自不同 replay windows，其 state 未必相同。错误地用 union (\pi) 给每次 exposure 加权会产生偏差。

## J.7 State inconsistency bias

若 episode state 不等于 full state，则：

[
\mathbb E[\widehat L_v]-L_v
===========================

\frac1{T_v}\sum_t
\mathbb E[
\ell_{v,t}^{E}
-\ell_{v,t}^{\mathrm{full}}
\mid t\in S(E)
].
]

这就是为什么：

```text
exact IPW cannot repair replay-state bias
```

## J.8 Clipped IPW

一般公式：

[
\widetilde w_{v,t}
==================

\min(w_{v,t},c).
]

任何 (c<2.5) 都会引入设计偏差，必须报告：

* clipped exposure count；
* clipped target mass；
* uncapped vs capped gold loss；
* uncapped vs capped gradient cosine。

v1 primary：

```text
weight_clip_cap=2.5
```

这是非绑定上限；任何实际 clipping 都使 run invalid。

## J.9 SNIPW / Hájek diagnostic

可计算：

[
\widehat L_v^{SN}
=================

\frac{
\sum_{m,t\in S_m}w_{v,t}\ell_{v,t}
}{
\sum_{m,t\in S_m}w_{v,t}
}.
]

它通常方差更低，但有限样本下有偏，并可能改变不同视频的相对贡献。因此：

```text
PRIMARY TRAINING ESTIMATOR = uncapped HH/IPW
SNIPW = diagnostic sensitivity only
```

除非 G0 明确证明 HH 方差不可接受，且作者在任何 Q2 quality result 可见前重新冻结 estimator。

## J.10 Loss denominator

Primary Q2 保持当前 token loss定义：

[
\ell_{v,t}
==========

\lambda_b\ell_b+
\lambda_a\ell_a+
\lambda_c\ell_c+
\lambda_s\ell_s+
\lambda_e\ell_e.
]

不得额外改成 instance-uniform 或 risk-set-uniform objective。

但 trace 必须记录每项：

```text
numerator
denominator
mask count
positive count
negative count
weighted value
```

例如 class/start loss 在同一 bin 有多个 active bindings 时，仍保持当前代码的 active-binding mean；IPW 作用于最终 token component，不重新给每个 instance 赋一套 primary target weight。

Instance-uniform 可作为 sensitivity analysis，但不是 Q2 primary。

## J.11 ESS

### Exposure-level ESS

[
ESS_{\mathrm{bin}}
==================

\frac{(\sum_jw_j)^2}{\sum_jw_j^2}.
]

### Episode-cluster ESS

令：

[
W_e=\sum_{t\in S_e}w_{e,t},
]

则：

[
ESS_{\mathrm{episode}}
======================

\frac{(\sum_eW_e)^2}{\sum_eW_e^2}.
]

### Video-cluster ESS

令：

[
W_v=\frac1M\sum_{e\in v}W_e,
]

则：

[
ESS_{\mathrm{video}}
====================

\frac{(\sum_vW_v)^2}{\sum_vW_v^2}.
]

还必须按以下 strata 报告：

* class；
* background；
* birth；
* ongoing；
* endpoint；
* post-end/refractory；
* same-class repeated；
* overlap；
* long-action；
* active-at-entry；
* chunk-crossing。

ESS 是 design diagnostic，不可把同一视频中的加权 bins 当作独立统计样本。最终 CI 必须以 video 为 cluster。

## J.12 Optimizer boundary

推荐每个视频形成一个 sampled video group：

```text
video v
  -> M frozen episode draws
  -> accumulate M HH episode gradients
  -> divide by M
  -> one optimizer.step
```

因此：

* 仍然一视频一 optimizer event；
* scheduler event semantics 与 full route一致；
  -视频权重保持均匀；
* full vs CRS 的主要差异是 temporal tokens 和 state surrogate，而不是 optimizer频率。

---

# K. Active-at-Entry, Burn-in, and State Reconstruction

## K.1 四方案比较

| 方案                                    | 因果合法 |            Runtime state fidelity |    成本 | 主要问题                                 |
| ------------------------------------- | ---- | --------------------------------: | ----: | ------------------------------------ |
| A. video-start replay                 | 是    |                              gold |     高 | 可能消除 CRS 成本优势                        |
| B. fixed 192 burn-in                  | 是    |                               未保证 |     低 | 漏掉长动作 birth 和更早 runtime history      |
| C. earliest relevant observable birth | 是    | canonical GT state 较好；runtime仍未保证 | 中等/可变 | 捕捉不了更早 false slots、memory、refractory |
| D. oracle active-slot initialization  | 否    |                              人工正确 |     低 | GT state leakage，禁止                  |

## K.2 主方案

设 supervised window 第一 bin 为 (u)。

基础 replay start：

[
r_0=\max(1,u-192).
]

令 (\mathcal R_e) 是在 supervised window 内 active 或 endpoint-crossing 的所有 GT instances，其 observable birth bins 为 (b_i)。

[
r_e
===

\min
\left(
r_0,;
\min_{i\in\mathcal R_e}b_i
\right).
]

如果 (r_e=1)，自然退化为 video-start replay。

这保证：

* 相关 GT instance 的 birth 不会被 artificial context truncation 丢失；
* same-class repeated instances 保持不同 instance IDs；
* overlap instances 的所有 relevant births 被 replay；
  -没有 oracle slot injection。

## K.3 关键限定

`earliest relevant birth` 不保证模型 runtime state 与 full stream 一致，因为此前不相关事件或错误预测也可能影响 query/memory/slot status。

因此主方案的合法状态是：

```text
CAUSAL=YES
SUPERVISION-LIFECYCLE-COMPLETE=YES
FULL-RUNTIME-EQUIVALENT=REQUIRES-G0
```

## K.4 Gradient replay

设 supervised first bin 的原 global chunk start：

[
g(u)=1+64\left\lfloor\frac{u-1}{64}\right\rfloor.
]

执行：

```text
[r_e, g(u)-1]:
    forward only, no supervised loss, no retained graph

[g(u), suffix end]:
    gradient enabled
    detach again at every original 64-token global boundary
    supervised loss only on the declared 8 bins
```

这比简单 `192 -> detach -> 8` 更接近当前 full reference 的 surrogate gradient。

## K.5 True left censor

仅当 annotation 表明 action 已在视频开始前发生时：

```text
true_left_censored = true
birth_loss_mask = 0 until first observable evidence
start_loss_mask = 0 or interval-censored target
alive/end risk remain explicitly defined
```

以下不是 true censor：

* 192 context 不够；
  -人为截断视频；
  -episode 从中间开始；
  -cache 缺失前缀；
  -为了节省成本删除 birth。

## K.6 Gold audit

G0 对同一 checkpoint、同一 token-wise RNG 比较：

```text
A: video-start full replay
B: fixed-192 replay
C: dynamic earliest-birth replay
D: reset-state control
```

比较：

* canonical supervision exact equality；
* slot status/refractory exact equality；
* query/memory relative L2 和 cosine；
* start/score/class state；
* logits；
* per-loss values；
* gradient cosine/norm ratio；
* state occupancy distribution；
* actual replay token cost。

通过规则不是“C 必须 bitwise 等于 A”，而是：

1. C 必须显著且稳定地比 B/D 更接近 A；
2. C–A discrepancy 必须落入预注册 fidelity margin；
3. discrete canonical state 必须 exact；
   4.任何 oracle state usage 直接 invalid；
   5.若 C 经常退化为 A，则成本 claim 失败。

## K.7 Hidden-state cache

训练中默认禁止 detector hidden-state cache。

原因：

* optimizer.step 后 detector parameters 已改变；
  -旧 state 是旧参数下的函数；
  -复用旧 state 会产生 stale-state objective；
  -仅绑定 checkpoint hash 不能解决同一 checkpoint 内 optimizer revision 变化。

允许的唯一例外：

* frozen checkpoint 下只读的 gold audit；
* artifact 明确绑定 model state SHA、optimizer revision、manifest、RNG；
* cache 不进入参数更新。

---

# L. Full Chronological vs CRS-EPS vs Hybrid Decision

| Criterion               | A. Full only                        | B. CRS only                  | C. Hybrid     |
| ----------------------- | ----------------------------------- | ---------------------------- | ------------- |
| Scientific fidelity     | 当前 objective 的 exhaustive reference | 依赖 replay consistency        | 有 gold anchor |
| State fidelity          | 最高                                  | 未知                           | 可测、可 kill     |
| Q2 identifiability      | 数据简单，但成本高                           | sampling/state confound 风险最大 | 最强            |
| Sampling bias           | 无 proposal bias                     | 有                            | 可由 G0 审计      |
| Cost                    | 最高                                  | 预期最低，未证明                     | 中等            |
| Complexity              | 低                                   | 高                            | 较高            |
| Raw-video extensibility | 差                                   | 潜在较好                         | 最合理           |
| Reviewer defensibility  | “贵但诚实”                              | 容易被质疑 surrogate              | 最强            |
| Failure diagnosability  | 一般                                  | 差                            | 最好            |

## L.1 裁决

```text
RECOMMENDED_PROTOCOL=HYBRID
```

具体为：

```text
Primary training:
    CRS-EPS HH/IPW
    one video-group optimizer event
    no periodic full-stream updates

Gold:
    preregistered tiny fit-core subset
    video-start state/loss/gradient audit

Validation/test:
    complete chronological one-token streaming

Calibration:
    threshold-only on locked calibration population
    no model parameter update
```

## L.2 不允许进入 primary protocol 的操作

* periodic full-stream optimizer updates；
  -训练末 chronological finetune；
  -看到第一种子结果后修改 mixture；
  -根据 fixed/rematch结果选择 (M)；
  -把 gold subset 作为额外训练数据；
  -根据 test set 调 threshold。

## L.3 Chronological finetune

仅在 Q2 positive 后，可作为独立、预注册 ablation：

```text
CRS-only model
vs
CRS + short chronological calibration update
```

它不得与 primary Q2 fixed/rematch混合，因为会改变 estimator、成本和 treatment。

---

# M. Revised Compute/Profile Contract

## M.1 Verdict

```text
PROFILE_CONTRACT=REPLACE
```

当前 50 warmup + 200 measured optimizer events只能保留为：

> 同一 full-chronological route 内 fixed/rematch 的 matched engineering diagnostic。

不能作为 CRS vs full 的 primary cost comparison。

## M.2 新 artifact schema

建议：

```text
schema_version: crs-eps-profile-v1

identity:
  commit_sha
  source_tree_sha256
  scientific_config_sha256
  feature_cache_manifest_sha256
  episode_manifest_sha256
  target_risk_sha256
  weighting_contract_sha256
  replay_contract_sha256
  hardware
  precision
  seed

workload:
  videos_total
  videos_unique
  episode_draws
  episode_keys_unique
  supervised_bin_exposures
  weighted_target_mass
  ess_bin
  ess_episode
  ess_video
  ess_by_class
  ess_by_lifecycle

compute:
  temporal_forward_tokens
  temporal_backward_tokens
  visual_forward_frames
  visual_backward_frames
  optimizer_events
  replay_tokens
  dynamic_extension_tokens
  video_start_fallback_tokens

time:
  wall_clock_seconds
  gpu_active_seconds
  host_data_wait_seconds
  h2d_seconds
  forward_seconds
  backward_seconds
  optimizer_seconds
  supervision_cpu_seconds
  python_unroll_cpu_seconds

memory:
  peak_vram_bytes
  peak_host_rss_bytes

derived:
  temporal_forward_tokens_per_second
  temporal_backward_tokens_per_second
  supervised_bins_per_second
  effective_bins_per_second
  gpu_hours_per_ess_episode
  gpu_hours_per_video_group
```

## M.3 Workload matching

必须多轴报告，不能寻找一个万能分母。

### Axis 1：相同视频群体

* 同一 deterministic video list；
* full route完整处理这些视频；
* CRS route处理这些视频的 frozen episode groups。

### Axis 2：相同 optimizer semantics

* 一视频一个 optimizer event；
  -同一 video order；
  -同一 scheduler event count；
  -同一 BF16；
  -同一 optimizer config。

### Axis 3：physical token accounting

报告实际：

* forward tokens；
* backward tokens；
* supervised bins；
* replay tokens。

### Axis 4：statistical workload

报告：

* raw weighted target mass；
* bin/episode/video ESS；
* class/lifecycle ESS；
* unique video coverage。

### Axis 5：最终 cost-to-quality

只有完成完整 chronological evaluation 后，才可报告：

```text
GPU-hours to reach preregistered quality
wall time to reach preregistered quality
```

Profile 本身不能证明 efficiency claim。

## M.4 Warmup

不用 optimizer-event 数跨协议匹配。

推荐：

1. 独立 warmup manifest；
   2.至少触发所有 kernels、replay paths、dynamic extension 和 optimizer path；
2. warmup 以累计 temporal forward/backward tokens和至少若干完整 video groups定义；
3. warmup 数据不进入 measured trace；
4. exact counts 由 frozen manifest 决定。

## M.5 Cache heat

分别报告：

* cold first pass；
* warm steady-state pass。

不允许 full route使用 cold cache、CRS 使用 warm cache。

若不能可靠清空 OS page cache，则：

* run order randomized；
  -做 AB/BA counterbalancing；
  -每种 route 至少三次短 profile replicate；
  -报告 data-wait 单独计时。

## M.6 Python unroll measurement

在以下位置加 NVTX/`torch.profiler` region：

```text
data_wait
feature_load
replay_no_grad
replay_grad
head_step
supervision_transition
loss_build
backward
optimizer
transaction
```

报告：

* CPU self time；
* CUDA kernel time；
  -GPU idle gaps；
  -每 token Python overhead。

## M.7 Bounded full reference

Full route太慢时，不得截断每个视频的尾部。

允许：

* deterministic whole-video subset；
  -按 length quartile、instance density、overlap、long-action、chunk-crossing stratify；
  -完整处理入选视频。

禁止：

-只选短视频；
-处理每视频前 N tokens；
-看到 profile 后换 subset。

## M.8 支持“CRS 降低训练成本”的最低证据

必须同时满足：

1. exact workload/accounting验证；
2. total GPU-hours 和 wall time均降低，CI 不跨无改善；
3. sampled/full fidelity通过；
4. complete chronological quality不劣；
5. recall/delay不退化；
6. savings 不是因为减少发射、减少 positives 或改变 target；
7. dynamic replay没有普遍退化为 video-start；
   8.实现/审计固定开销已计入总成本。

## M.9 预算

沿用作者冻结的不可借用预算：

| 子项                                | GPU-hour cap |
| --------------------------------- | -----------: |
| provenance-required cache rebuild |            1 |
| profiles                          |            2 |
| G0 gold audit                     |            1 |
| paired Q2 train+eval              |            6 |
| total pre-Stage2                  |           10 |

一项超限：

```text
STOP_AND_REVIEW
```

不得静默从另一项借预算。

---

# N. Feature Provenance and Real Online Clock

## N.1 新 provenance schema

```text
schema_version
extractor_repository_url
extractor_commit_sha
extractor_clean_checkout
extractor_source_tree_sha256
extractor_config_sha256
resolved_extractor_config_sha256
checkpoint_sha256
processor_name
processor_version
processor_config_sha256
normalization_mean_std
decode_library_and_version
dtype
feature_dim
feature_stride
frame_selection_policy
temporal_support_rule
padding_rule
causal_attention_rule
per_token_source_frame
per_token_raw_support_start
per_token_raw_support_end
per_token_nominal_availability_frame
cache_file_sha256
raw_future_invariance_report_sha256
```

## N.2 Raw-future invariance

构造两个视频：

```text
same raw prefix
different future suffix
```

重新执行 extractor，要求所有 prefix tokens：

* bitwise equal，或
  -在预注册数值容差内 equal。

只扰动未来 cached tokens 的现有 detector test不能证明 extractor causal。

## N.3 Training vs serving packet

```text
training chunk = 64 tokens
primary serving packet = 1 token
```

Training chunk是吞吐机制，不是 deployment cadence。

## N.4 五个时钟

每条 emission 必须记录：

```text
source_time
feature_availability_time
decision_start_time
decision_completion_time
emission_publication_time
```

以及：

```text
clock_mode = measured_raw_pipeline | simulated_cached_replay
```

不能把 cached replay 的 source time包装成真实 wall-clock latency。

## N.5 Primary formal evaluation

```text
one token per model call
stride = 8 frames
nominal source cadence ≈ 0.267 s at 30 FPS
no offline NMS
immutable emission
full chronological stream
```

Micro-packet 仅用于吞吐诊断。

## N.6 Cache 的合法角色

Cache 可以用于：

* cached Q2 mechanism training；
* CRS profile；
  -full chronological cached evaluation；
  -state/loss/gradient audit。

Cache 不能支持：

* raw-video end-to-end claim；
  -LoRA visual gradient；
  -真实 extractor latency；
  -严格 raw-stream wall-clock claim。

## N.7 Reporting population

关闭 211/213 前：

```text
dataset_name = THUMOS14 locked-211 reporting subset
```

所有 baseline必须在同一 211 population 重评。

Cost-only profile不被该问题阻塞；正式 effectiveness interpretation与论文 reporting被阻塞。

---

# O. Fresh Literature and Competition Matrix

## O.1 Direct On-TAL / On-TAL-adjacent work

| Work         | Task / training unit                            | State / sampling / visual gradients                                          | Threat                                                               |
| ------------ | ----------------------------------------------- | ---------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| SimOn        | On-TAL；逐时刻序列预测                                  | 当前 feature query + past visual/context embedding；论文使用 TSN feature extraction | 证明轻量 sequential state 和直接 action-instance生成早已有之。([arXiv][3])         |
| MATR         | On-TAL；current segment + memory queue           | selective past segment memory；当前 segment预测 end，memory估 start                 | 直接占据 long-history、memory-assisted interval localization。([arXiv][4]) |
| HAT          | On-TAL/PREGO                                    | history-augmented anchor features                                            | 占据 history + anchor refinement。([arXiv][5])                          |
| ActionSwitch | class-agnostic On-TAL                           | explicit action-state switch；面向 concurrent/same-class                        | 是 identity/overlap error最强直接 baseline之一。([arXiv][6])                 |
| OnPoint      | point-supervised online TAL                     | offline teacher→online student multi-level distillation                      | 2026 新工作，但改变监督合同，不是 fully supervised Q2直接 baseline。([arXiv][7])      |
| OZ-TAL       | online zero-shot TAL                            | training-free VLM framework                                                  | 改变类别泛化任务；阻止 zero-shot/VLM headline。([arXiv][8])                      |
| OpenHOUSE    | hierarchical/open-ended streaming understanding | On-TAL + free-form hierarchy                                                 | 任务越界，但阻止 broad streaming semantics claim。([arXiv][9])                |

### OAT 说明

项目 wiki 中的 `OAT / A Sliding Window Scheme for Online Temporal Action Localization` 没有记录可独立解析的 arXiv、DOI 或作者；本轮 fresh search 未找到足以进行 primary-source核验的 exact artifact。因此：

```text
OAT_PRIMARY_ARTIFACT=UNRESOLVED
```

本裁决不把 OAT 作为 load-bearing novelty证据。即使移除 OAT，SimOn、MATR、HAT 和 ActionSwitch 已足以击穿“history / early proposal / refinement / state switch 本身新颖”的主张。

## O.2 Raw-video causal/online training

| Work         | Task                                     | Training unit                                      | Relevance                                                                                |
| ------------ | ---------------------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| E2E-LOAD     | raw-video end-to-end OAD                 | long/short form online action detection            | 证明 raw-video online backbone可训练，但输出是 frame-level OAD而非 completed instances。([arXiv][10]) |
| StreamFormer | streaming representation；OAD/VIS/VideoQA | causal ViT + multitask alignment                   | 阻止“causal streaming backbone”本身作为主创新。([arXiv][11])                                       |
| TeSTra       | online video detection                   | constant-update temporal smoothing/cache           | 证明 streaming long-history compute reuse也已有强先例。([arXiv][12])                              |
| OnlineTAS    | online temporal action segmentation      | non-offline adaptive memory + feature augmentation | 与 clip/state training相关，但任务输出不是 instance On-TAL。([arXiv][13])                            |

## O.3 Efficient TAL / long-video training

| Work             | Mechanism                                                                    | Threat to CRS/Stage2                                                 |
| ---------------- | ---------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| ETAD             | sequentialized encoder forward；selective snippet gradients；proposal sampling | 最强 selective-gradient和sampling baseline；CRS不能声称“高效采样新颖”。([arXiv][1]) |
| TALLFormer       | long-term feature bank；只更新部分 video features                                  | 强 cache/memory/staleness baseline。([arXiv][2])                       |
| Re²TAL           | reversible backbone rewiring                                                 | 强 memory-efficient E2E TAL baseline。([arXiv][14])                    |
| BasicTAD/PlusTAD | 简化 raw-video E2E TAD pipeline                                                | 说明基础 raw-video E2E baseline必须很强。([arXiv][15])                        |
| TIA/AdaTAD       | lightweight temporal-informative adapters，大模型长序列适配                           | Stage2 LoRA/adapter claim受到直接威胁。([arXiv][16])                        |
| LoSA             | long/short-range adapters并行适配 foundation backbone                            | 阻止“为 TAL 增加 LoRA/adapter”本身成为贡献。([arXiv][17])                        |

## O.4 Persistent identity queries

| Work        | Identity mechanism                                         | Threat                                                     |
| ----------- | ---------------------------------------------------------- | ---------------------------------------------------------- |
| TrackFormer | static object queries birth + autoregressive track queries | identity-preserving persistent query的核心概念先例。([arXiv][18])  |
| MOTR        | frame-to-frame track query + tracklet-aware assignment     | birth、continuation和trajectory assignment强先例。([arXiv][19])  |
| MeMOTR      | memory-augmented stable track embeddings                   | long-term persistent identity memory先例。([arXiv][20])       |
| InsPro      | online VIS query/proposal propagation                      | 证明在线 query propagation可同时承担预测和隐式 association。([arXiv][21]) |
| ROVIS       | online VIS track queries                                   | TrackFormer式 query迁移到视频分割已有直接先例。([arXiv][22])              |

## O.5 Literature verdict

### CRS‑EPS 新颖性

```text
CRS_EPS_NOVELTY=LOW
```

它是以下思想的组合：

* annotation-guided event oversampling；
* uniform support；
* importance weighting；
* truncated/replayed causal state；
* ETAD-style selective computation；
* full-stream validation。

它应被定位为：

```text
audited cost surrogate / enabling protocol
```

不是 headline algorithm。

### 最强训练方法 baseline

* **Stage 1 sampling/cost**：uniform prefix sampling、full chronological reference。
* **Raw-video Stage 2**：ETAD-style selective visual gradients。
* **Long-video cache/staleness**：TALLFormer。
* **Adapter efficiency**：TIA/AdaTAD、LoSA。

### 最强整体重构

```text
StreamFormer or E2E-LOAD
+ MATR / ActionSwitch
+ TrackFormer / MOTR
+ ETAD / TALLFormer
+ ordinary interval heads
```

这一路径能够重构 Full PETAL 大多数系统组件。

### 仍可能存活的 delta

只有：

> 在严格 completion-triggered On-TAL 中，prefix-observable fixed post-birth supervision是否产生可测的 identity-linked interval benefit，并且该 benefit 能在 causal raw-video adaptation 后继续存在。

当前仍未证明。

---

# P. Claim Map

| Claim | Exact scope                                               | Current evidence              | Closest prior                           | Required baseline                     | Required metric                 | Falsifier                  | Paper status                              |
| ----- | --------------------------------------------------------- | ----------------------------- | --------------------------------------- | ------------------------------------- | ------------------------------- | -------------------------- | ----------------------------------------- |
| C0    | Cached tokens 上 causal sequential detector                | code-supported                | SimOn/MATR                              | future perturbation、stepwise replay   | protocol audit                  | future token影响前缀           | `code-supported`                          |
| C1    | Full route是当前 video-uniform、64-detach objective reference | code-supported with scope     | 无直接 prior需求                             | recomputed full route                 | exact loss/gradient trace       | normalization不符            | `protocol-supported`                      |
| C2    | CRS低成本逼近 full route                                       | 未实现                           | ETAD/TALLFormer                         | uniform-prefix、full gold              | fidelity + GPU-hours            | state/gradient/quality不劣失败 | `requires experiment`                     |
| C3    | fixed优于 rematch                                           | synthetic local evidence only | MOTR/TrackFormer assignment precedent   | matched rematch                       | standard mAP                    | fixed不优                    | `partially identifiable`                  |
| C4    | gain由identity而非mask/normalization导致                       | 无结果                           | ActionSwitch                            | direct twin trace、mediator trace      | duplicate/fragmentation + trace | gain只来自少发射/低recall         | `not identifiable`                        |
| C5    | 减少duplicate/fragmentation且无recall collapse                | 无结果                           | ActionSwitch/MATR                       | fixed/rematch + FRESH/TTF if positive | identity + safety               | recall/FN/delay恶化          | `requires experiment`                     |
| C6    | cached Q2 positive足以授权 Stage2                             | 协议上不成立                        | 无                                       | 三种子+FRESH/TTF+novelty review          | joint gate                      | aggregate-only gain        | `unproven`                                |
| C7    | Raw CRS+LoRA可接受成本适配                                       | 未实现                           | E2E-LOAD、ETAD、TIA、LoSA                  | frozen/LoRA 2×2                       | mAP、cost、grad audit             | LoRA无独立收益                  | `requires experiment`                     |
| C8    | Full PETAL超越多论文重构                                         | 当前不成立                         | TrackFormer/MOTR+MATR+StreamFormer+ETAD | strongest reconstruction              | direct matched comparison       | reconstruction匹配           | `collapsed by prior art` at package level |

C8 的“collapsed”指完整 package headline；不等于 fixed-binding marginal hypothesis已被直接发表。

---

# Q. Minimal Baseline/Ablation Matrix

## Q.1 Before first profile

不运行 effectiveness baseline。只完成：

| ID   | 内容                                         | 目的            |
| ---- | ------------------------------------------ | ------------- |
| P0-A | launch identity fake-sbatch integration    | 关闭 P0         |
| P0-B | episode manifest builder/auditor CPU tests | probability正确 |
| P0-C | full-vs-episode synthetic state tests      | 无oracle       |
| P0-D | revised profile schema tests               | 多分母闭包         |
| P0-E | fixed/rematch local twin trace             | 直接 one-factor |

## Q.2 Before first scientific run

### G0 — tiny full-stream gold subset

* 从 fit-core 选择 deterministic whole-video subset；
  -覆盖 length quartile、overlap、same-class repetition、active-at-entry、long action、chunk crossing；
  -禁止使用 final reporting population；
  -运行 A/B/C/D replay audit；
  -冻结 (M)、fidelity margins 和 exact episode manifest。

### G1 — CRS‑EPS fixed

```text
fixed_birth_slot
same manifest
same model init
same token-wise RNG
same cost
```

### G2 — CRS‑EPS rematch

与 G1 唯一直接代码干预为 post-birth loss binding。

### G3 — uniform-prefix control

在 G0 gold subset 上，以：

```text
alpha_uniform=1
```

比较 event mixture 是否改善 estimator variance/ESS，而不是改变 target。

首轮无需完整训练 G3。

### G4 — insufficient-state controls

仅 gold audit：

* reset state；
  -fixed 192；
  -dynamic earliest-birth；
  -video-start gold。

## Q.3 Only if G1/G2 positive

### G5 — simple recurrent/persistent control

一个无 Hungarian rebinding、无 sophisticated event set 的简单 GRU state control，判断收益是否只是普通 recurrence。

### G6 — FRESH and faithful Temporal TrackFormer

* same cache；
  -same CRS manifest；
  -same optimizer budget；
  -same output protocol；
  -no offline NMS。

这是 Full PETAL novelty的真正 kill stage。

### 三种子

只有在一 paired seed同时显示：

* primary metric方向正确；
  -identity endpoint方向正确；
  -safety不退化；
  -cost/fidelity通过；

才允许三种子低成本 kill confirmation。

## Q.4 Paper-stage only

* 至少五 paired seeds；
* hierarchical paired uncertainty；
  -FineAction 或其他合格 dense/overlap dataset；
* FRESH、Temporal TrackFormer、ActionSwitch/MATR-style controls；
  -raw-video 2×2；
  -standard external baselines；
  -full cost ledger。

---

# R. Metrics and Statistical Plan

## R.1 Primary quality

```text
standard temporal mAP averaged at tIoU 0.3,0.4,0.5,0.6,0.7
```

来源：

-完整 chronological immutable emissions；
-no offline NMS；
-固定 evaluator；
-固定 threshold/calibration。

## R.2 Mechanism endpoints

* duplicate FP / GT；
* duplicate fraction；
* fragmentation rate；
* same-class repeated subset；
* same-class concurrent subset；
* any-overlap subset；
* long-action subset；
* active-at-entry subset；
* chunk-crossing subset；
* rematch swaps；
* slot exhaustion。

## R.3 Safety endpoints

* recall；
* unmatched FN；
  -late FP；
  -completion delay relative to matched GT end；
  -calibration；
  -classwise collapse。

必须使用 GT-end latency，而非 `emit - predicted end` 作为 primary delay。项目 decision register 已冻结这一点。

## R.4 Fidelity metrics

* sampled/full per-loss discrepancy；
  -gradient cosine；
  -gradient norm ratio；
  -sign agreement；
  -entry-state query cosine/relative L2；
  -memory divergence；
  -discrete slot-state agreement；
  -state occupancy JS divergence；
  -CRS vs reset relative closeness；
  -ESS；
  -replay fallback rate；
  -replay token ratio。

## R.5 Cost metrics

* total GPU-hours；
  -wall time；
  -data wait；
  -forward/backward tokens；
  -visual frames；
  -optimizer events；
  -VRAM；
  -time per effective supervision；
  -time-to-quality after effectiveness run。

## R.6 Paired design

每个 seed 内：

* same initialization；
  -same episode manifest；
  -same video order；
  -same token RNG；
  -same evaluator；
  -fixed/rematch运行顺序 counterbalanced。

最终统计：

1. seed 为第一层 cluster；
2. video 为第二层 paired cluster；
   3.每个 bootstrap sample重算完整 mAP，而不是平均 per-video AP；
3. identity/safety endpoints做相同 paired resampling。

## R.7 One-seed vs science

```text
1 seed = resource kill test only
3 seeds = low-cost confirmation / route kill
5+ seeds = minimum retained claim
```

一 seed 的 CI 只能描述该 checkpoint/data order，不能支持跨 seed generalization。

## R.8 Non-inferiority margin

不应由 Pro 任意指定 `-1 pp` 或 `20%`。

冻结顺序：

1. G0 只使用 fit-core gold subset；
   2.估计 exact replay重复运行的 numerical noise；
   3.估计 sampled/full paired dispersion；
2. cost profile给出实际节省；
   5.作者签署最大可接受 quality loss (\delta_Q)，即为了该成本节省愿意接受的最大退化；
   6.用 development variance模拟 planned seed/video count对 (\delta_Q) 的 power；
   7.若预算内无法达到所需 power，kill，而不是放宽 margin。

Primary non-inferiority：

[
\operatorname{LCB}*{1-\alpha}
(Q*{\mathrm{CRS}}-Q_{\mathrm{full}})

>

-\delta_Q.
]

Safety metrics各自有独立、预签署 margin。

## R.9 Minimum meaningful cost reduction

同样不预设固定百分比。

定义 cost ratio：

[
R_C=
C_{\mathrm{CRS}}/C_{\mathrm{full}}.
]

最低要求：

* upper CI of (R_C<1)；
* projected paired Q2 完整训练+eval不超过6 GPU-hours；
* savings 超过 profile noise、额外 gold audit 和 manifest/replay fixed overhead；
* time-to-quality ratio最终也小于1。

作者若要使用“materially lower”，必须在 quality result可见前签署一个 utility threshold。

## R.10 Multiple comparisons

层级 testing：

1. primary mAP；
2. safety gate；
   3.预注册 identity endpoints；
   4.其他诊断。

只有前一层通过才解释后一层。Identity endpoints 内用 Holm adjustment；大量 subgroup只报告 effect/CI，不逐项宣称 significance。

## R.11 Failed-run rules

以下 run invalid，不作为零分或可删除 outlier：

* OOM；
  -timeout；
  -nonfinite；
  -slot exhaustion；
  -protocol taint；
  -manifest mismatch；
  -RNG identity mismatch；
  -population mismatch；
  -budget exceed；
  -skipped authenticated optimizer event。

同一 paired seed中一臂 invalid，则整对 invalid；是否允许重跑必须由预注册 retry rule决定，且两臂同时重跑。

---

# S. Test-to-Claim and Adversarial Cases

## S.1 Test-to-claim map

| Claim/invariant                   | Existing evidence           | Proves                        | Does not prove               | Required new test                |
| --------------------------------- | --------------------------- | ----------------------------- | ---------------------------- | -------------------------------- |
| complete-video optimizer boundary | train transaction tests     | 当前 full route一视频一step         | CRS group step               | sampled video-group test         |
| CRS absence                       | code inspection             | 当前未实现                         | 新实现正确                        | manifest route detector          |
| immutable manifest                | none                        | —                             | probability/replay identity  | canonical JSON/hash tests        |
| (q,\rho,\pi) correctness          | none                        | —                             | unbiased estimator           | exhaustive toy enumeration       |
| duplicate window overlap          | none                        | —                             | union/exposure handling      | multiplicity vs union tests      |
| uniform full support              | none                        | —                             | (\rho>0)                     | all-bin exhaustive test          |
| weight bound                      | none                        | —                             | (w\le2.5)                    | random/video-edge property tests |
| active-at-entry no oracle         | partial prefix tests        | GT not in runtime schema      | sampled replay fidelity      | adversarial long-action test     |
| long action >192                  | none                        | —                             | dynamic extension            | synthetic >192 birth test        |
| same-class overlap                | supervision unit tests      | canonical/rematch behavior    | sampled state reconstruction | episode-level integration        |
| identical manifest/RNG            | none                        | —                             | strict pairing               | stateless token RNG test         |
| loss denominator equality         | partial loss tests          | local component equality      | full trace equality          | numerator/denominator trace      |
| full vs episode state             | none                        | —                             | surrogate validity           | G0 runner                        |
| full vs sampled gradient          | none                        | —                             | training fidelity            | gradient audit                   |
| future suffix perturbation        | detector test               | cached-token prefix causality | raw extractor causality      | raw-video future test            |
| chunk vs stepwise inference       | existing detector test      | cached numerical equivalence  | availability clock           | one-token evaluator test         |
| availability clock                | none                        | —                             | wall latency                 | measured clock schema            |
| extractor provenance              | cache hash tests            | array integrity               | raw support causality        | provenance artifact test         |
| 211/213                           | generic data contract tests | validator works               | real IDs/reasons             | source-derived real diff         |
| launch workdir                    | missing shell integration   | —                             | launch identity closure      | deterministic fake-sbatch        |
| profile comparability             | current event trace         | event accounting              | cross-protocol fairness      | new profile schema               |
| LoRA parameter updates            | none in Q2                  | —                             | visual adaptation            | optimizer/grad/delta audit       |

## S.2 Adversarial cases

1. **未来帧藏在 cache token 内**：detector future-token test通过，但 extractor使用 centered clip。
2. **人工 context truncation伪装成 left censor**：birth loss被静默 mask，CRS看似稳定。
3. **动态 replay只重建 GT lifecycle，不重建错误 runtime slots**：sampled state与部署 state不同。
4. **旧 hidden-state cache**：模型更新后继续使用旧参数生成的 state。
5. **重叠 windows 使用错误 union probability**：每次 exposure都除以 (\pi)，产生系统偏差。
6. **event candidates为空时概率未重新归一**：某些视频 target mass丢失。
7. **uniform component有 anchor但 suffix不包含 anchor**：边缘 bins实际零支持。
8. **SNIPW 在视频间全局归一**：长视频/高事件视频重新获得更大权重，破坏 video-uniform target。
9. **class/start loss denominator改变**：fixed/rematch看似只换 binding，实际 active-instance weighting不同。
10. **两臂 dropout RNG不同**：局部 gradient差异被错误解释为 binding effect。
11. **两臂权重分化后 birth masks分化，却被报告为外生 one-factor failure或被完全忽略**。
12. **scheduler按短 episode event更新**：CRS比 full 获得更多 scheduler steps。
13. **full route cold cache、CRS warm cache**：虚假 wall-time gain。
14. **source time当 publication time**：64-token batch内部预测延迟被低估。
15. **locked‑211 排除了困难或长视频**：mAP与cost同时被美化。
16. **降低 birth/alive threshold输出更少 detection**：duplicate和delay变好，但 recall collapse。
17. **dynamic replay大量回退到 video start**：CRS名义 episode短，实际 forward token几乎等于 full。
18. **只报告 supervised 8 bins，不计 192+动态 replay tokens**：成本分母欺骗。
19. **只计 visual backward frames，不计所有 burn-in visual forward frames**：Stage2 savings夸大。
20. **fixed gain来自普通监督稳定化**，而非 persistent identity；简单 recurrent control同样获益。
21. **same-class subset在结果后定义**：选择最有利的 error slice。
22. **custom OnlineAP 上升但 standard mAP下降**。
23. **threshold在 locked reporting population上反复调优**：test contamination。
24. **slot exhaustion run被丢弃而不计失败率**。
25. **episode duplicates被静默去重**：实际 proposal与记录 probability不一致。
26. **每个视频 (M) 按结果难度动态变化**：video-uniform optimizer target被改变。
27. **full-reference gold subset只选短视频**：dynamic replay看似高 fidelity且低成本。
28. **cache rebuild与旧 baseline使用不同 encoder**：fixed/rematch或CRS/full不再 matched。
29. **LoRA后继续使用 frozen cache**：声称视觉参数更新，但输入 representation没有变化。
30. **positive Q2后直接跳过 FRESH/TTF**：把 tracking-query transplant包装成新方法。

---

# T. Raw-Video LoRA Stage-2 Gate

## T.1 当前权限

```text
RAW_VIDEO_STAGE2=BLOCKED
```

## T.2 进入 Stage 2 的必要前提

全部满足才可重新申请：

1. P0关闭；
2. feature provenance关闭；
3. CRS gold fidelity通过；
4. profile和10h Stage‑1预算通过；
5. fixed在至少三 paired seeds上优于 rematch；
   6.增益出现在identity-linked errors；
6. recall/FN/delay不退化；
7. FRESH与faithful Temporal TrackFormer未重构该收益；
8. renewed novelty review仍认为存在 surviving delta；
   10.第二阶段单独预算被批准。

## T.3 Stage‑2 2×2

| Visual adaptation | Binding |
| ----------------- | ------- |
| Frozen            | Rematch |
| Frozen            | Fixed   |
| LoRA              | Rematch |
| LoRA              | Fixed   |

所有四格共享：

* raw episode manifest；
  -replay ranges；
  -source frames；
  -token RNG；
  -head/temporal architecture；
  -evaluator；
  -training budget。

这才能分离：

* visual adaptation main effect；
  -fixed binding main effect；
  -adaptation × binding interaction。

## T.4 Conditional LoRA specification

可把历史配置作为**单一初始候选**，不是 hyperparameter sweep：

```text
top 4 visual transformer blocks
q_proj and v_proj
rank = 8
alpha = 16
dropout = 0.05
LoRA lr ≈ 2e-5
temporal lr ≈ 5e-5
head lr ≈ 1e-4
```

正式实现前必须由 runtime module discovery解析真实 parameter names；零命中 fail closed。项目文档中这些值目前只是候选设计。

## T.5 Context frames and gradients

所有 replay raw frames都必须以**当前 LoRA 参数**做 visual forward。

不能使用旧 frozen cache代替 trainable visual representation。

候选 gradient策略：

```text
all replay frames: visual forward
supervised suffix + declared final context K: visual backward
earlier burn-in: forward-only
```

这与 ETAD 的 selective snippet-gradient思想相近，因此不能作为独立新颖性。ETAD已明确研究只对部分 snippets反传 encoder gradient。([arXiv][1])

(K) 应由视觉 encoder causal receptive field和 gold gradient audit确定，不得在 test result后选择。

## T.6 Mandatory audits

* 每个 LoRA target parameter `requires_grad=True`；
  -全部被 optimizer覆盖；
  -base parameters frozen；
  -每个 target block有 finite nonzero gradient；
  -每个 optimizer event有 nonzero parameter delta；
  -无全局 `torch.no_grad()` 包围 LoRA；
  -prefix raw-future invariance通过；
  -visual forward/backward frame counts完整记录；
  -cache在 LoRA启用后不得进入视觉训练路径。

## T.7 Permanent full-tower block

发生任一情况，永久禁止 full-tower finetuning：

* LoRA无独立 mAP/identity收益；
  -LoRA增益只来自更大发射率或 recall/latency tradeoff；
  -LoRA × fixed interaction为零；
  -成本超预算；
  -prefix causality失败；
  -selective visual gradient与full visual gradient明显不一致；
  -第二数据集不复现；
  -strong reconstruction baseline匹配。

---

# U. The Strongest Case for Killing CRS-EPS-Enabled Full PETAL

最强拒稿观点如下：

Full PETAL 目前不像一个不可替代的新方法，更像一个由多个已知方向拼接出来的系统工程包。严格 causal video representation 已有 E2E‑LOAD、StreamFormer、TeSTra 等先例；On‑TAL 的历史记忆、anchor refinement 和 concurrent action handling 已由 MATR、HAT、ActionSwitch 等覆盖；persistent identity query、birth/death 与轨迹 assignment 是 TrackFormer、MOTR、MeMOTR 和在线 VIS 的成熟思想；长视频的 selective gradient、feature bank、reversible training 和 lightweight adapters 又分别由 ETAD、TALLFormer、Re²TAL、TIA/AdaTAD、LoSA 覆盖。把这些组件放到一维时间区间检测上，并不能自动形成论文级算法 delta。

CRS‑EPS 本身也很难承担创新。它的核心是 annotation-guided event oversampling、uniform support、importance weighting、短后缀监督和 truncated replay。这些分别属于普通 survey sampling、hard-example/event sampling 和 truncated BPTT 范畴。即便工程上必要，也更像一种 cost surrogate，而非主方法。更严重的是，IPW 只能修正 decision-time exposure distribution，不能修正 sampled episode 的 latent state 与真实 full-stream state不一致。固定 192-token burn-in可能漏掉长动作 birth、早期错误 slot、长期 query memory、refractory history 和之前动作形成的状态；dynamic earliest-birth replay虽然修复相关 GT lifecycle，却依然无法恢复模型自身的全部错误历史。若状态不一致，所谓“无偏”只对另一个 episode-induced objective成立。

当前 feature cache进一步掩盖了 Full PETAL 最昂贵、最难证明的部分：视觉 encoder是否严格 causal、raw support interval是什么、真实 feature availability和wall-clock latency是多少、LoRA后如何为全部 burn-in frames重新前向。Stage 1 的 GPU-hour savings不能外推到 raw-video Stage 2。当前 optimizer-event分母也具有误导性：一个完整视频 event与一个短 sampled episode event完全不是同一工作量；若只报告 events/s，任何加速结论都不可接受。

即使 fixed优于 rematch，结果也可能只是固定 label assignment降低了训练噪声，而非模型获得了可部署的identity tracking能力。它还可能通过更少 birth、更低 occupancy或更少 emission来降低duplicate，而同时损害 recall和completion delay。THUMOS14规模小，当前又有 locked‑211 vs canonical‑213问题；在这样的单一数据集上，数个 mAP点或某个自定义 identity metric改善不足以证明一般性方法。若不与 FRESH、faithful Temporal TrackFormer、ActionSwitch-style state control和 strongest reconstruction直接比较，positive Q2只能说明一种 supervision policy在一个配置上更稳定。

因此，即使 Q2 positive，论文仍可能不足：CRS‑EPS不新；persistent queries不新；causal backbone不新；adapter不新；identity gain可能由普通稳定化解释；raw-video系统尚未验证。只有当 fixed binding在严格配对下稳定改善标准 mAP和identity-linked errors、CRS在full-stream gold上保持state/gradient/quality fidelity、真实成本显著降低、FRESH/TTF不能重构、raw-video LoRA提供独立增益、第二数据集复现，并且最终 surviving delta不能被上述多论文组合解释时，Full PETAL才有继续作为主论文路线的资格。否则最科学的结果是终止 headline，保留基础设施和负结果。

---

# V. File-Level Implementation Plan

## V.1 P0 before any profile

### Modify

`tools/remote/submit_full_petal_q2_n16r4.sh`

* 在 ticket创建前冻结 `RUN_ID/RUN_DIR/WORK_DIR`；
  -运行 argv只能读取 ticket-bound overrides；
  -支持 deterministic dry-run/fake-sbatch；
  -禁止二次生成 `work_dir`。

`tools/build_full_petal_launch_ticket.py`

* 接受预冻结 artifact root；
  -把 work directory identity写入 ticket；
  -输出 canonical override order。

`opentad/utils/full_petal_launch.py`

* 验证 ticket artifact root、runtime override与live argv完全一致；
  -不得静默排除 `work_dir`。

### Add

`tests/test_full_petal_submit_shell_integration.py`

覆盖：

* frozen date/run ID；
  -fake `sbatch`；
  -ticket vs generated sbatch argv exact equality；
  -workdir single-character mutation；
  -pre-CUDA rejection；
  -no overwrite；
  -clean checkout。

---

## V.2 P1 before scientific run — Manifest and sampling

### Add

`opentad/utils/crs_eps_manifest.py`

职责：

* candidate sets；
  -mixture distributions；
  -window construction；
  -(\rho,\pi,w)；
  -video/epoch/draw manifest；
  -canonical hash；
  -probability validation。

`tools/build_crs_eps_manifest.py`

输入：

* locked split；
  -annotation；
  -feature source frames；
  -(H,M,\alpha)；
  -epochs/seeds。

输出外部、non-overwritable manifest。

`tools/audit_crs_eps_manifest.py`

验证：

* exact support；
  -probability sums；
  -weight bound；
  -duplicate multiplicity；
  -coverage；
  -ESS；
  -replay cost estimate；
  -strata coverage。

`tests/test_crs_eps_probability_contract.py`

使用 tiny exhaustive videos枚举所有 anchors和windows，验证 analytic (\rho,\pi)。

---

## V.3 P1 — Dataset isolation

### Add

`opentad/datasets/crs_eps_episode.py`

职责：

* 只读取 frozen episode manifest；
  -返回 replay tokens、gradient ranges、supervised mask；
  -不重新采样；
  -不把 GT sampler fields放入 model metadata；
  -每条 sample绑定 input provenance。

不要修改 `StreamingFeatureDataset` 的 full-reference semantics。

`opentad/utils/crs_eps_batch_sampler.py`

-按 video group组织 (M) episodes；
-固定 video order；
-同一 paired seed共享 manifest；
-不与 `ChronologicalStreamBatchSampler` 混用。

---

## V.4 P1 — Detector episode API

### Modify

`opentad/models/detectors/persistent_trajectory_ontad.py`

新增：

```python
train_prefix_episode(
    inputs,
    masks,
    model_meta,
    prefix_schedule,
    replay_control,
    supervised_mask,
    initial_state=None,
)
```

`replay_control` 只包含：

* causal source ranges；
  -gradient boundaries；
  -detach boundaries；
  -no-loss burn-in mask。

不得包含：

* future endpoint；
  -GT active slot；
  -video duration/EOF；
  -oracle query。

### Add

`opentad/utils/token_addressed_rng.py`

-由 video/token/module key生成 reproducible dropout RNG；
-记录 digest；
-full/sampled/paired arm共享。

---

## V.5 P1 — Weighting and denominators

### Add

`opentad/utils/crs_eps_weighting.py`

实现：

* HH/IPW；
  -union HT diagnostic；
  -SNIPW diagnostic；
  -ESS；
  -stratum ESS；
  -weight-bound fail closed。

### Modify

Detector loss返回：

```text
per_loss_numerator
per_loss_denominator
per_token_component_losses
weighted_cost
_optimizer_weight
```

禁止 engine通过标量平均猜测 denominator。

---

## V.6 P1 — Training engine

### Add

`opentad/cores/crs_eps_train_engine.py`

每个 video group：

1.加载 (M) episodes；
2.对每条 episode执行 replay；
3.累计 HH weighted gradient；
4.除以 (M)；
5.一次 optimizer/scheduler event；
6.任何 episode失败则整个 video group rollback；
7.记录完整 compute counters。

不要先把旧 `train_one_epoch` 大规模抽象重构；保留它作为 full reference。

---

## V.7 P1 — Paired trace

### Add

`opentad/utils/q2_paired_trace.py`

`tools/run_q2_paired_direct_effect_audit.py`

-同一模型 snapshot同时执行 fixed/rematch twin forward；
-记录 I 节全部字段；
-生成 hash-chained trace；
-验证唯一直接差异。

`tests/test_q2_paired_trace.py`

-人工注入 mask、RNG、denominator、birth差异；
-确保 verifier fail closed。

---

## V.8 P1 — Full-stream gold audit

### Add

`tools/run_crs_eps_gold_audit.py`

模式：

```text
video_start_full
fixed_192
dynamic_birth
reset
```

输出：

* state metrics；
  -loss metrics；
  -gradient metrics；
  -occupancy；
  -cost；
  -replay ratio；
  -control comparisons。

`opentad/utils/crs_eps_gold_gate.py`

根据预注册 margins判定 PASS/KILL。

---

## V.9 P1 — Revised profile

### Add

`opentad/utils/crs_eps_profile.py`

实现 M 节 schema。

### Modify

`tools/train.py`

-识别 full vs CRS route；
-安装 profiler regions；
-profile exit前不 checkpoint/eval；
-绑定 exact workload manifest。

`tools/check_full_petal_results.py`

增加：

* multi-denominator identity；
  -ESS；
  -token/frame counters；
  -gold fidelity artifact；
  -paired trace artifact；
  -no skipped group；
  -budget cap；
  -population identity。

---

## V.10 P1 — Configs

新增：

```text
configs/causaltad/thumos_q2_crs_eps_base.py
configs/causaltad/thumos_q2_crs_eps_fixed.py
configs/causaltad/thumos_q2_crs_eps_rematch.py
configs/causaltad/thumos_q2_crs_eps_uniform_gold.py
configs/causaltad/thumos_q2_crs_eps_reset_diagnostic.py
```

Static config差异测试必须证明 fixed/rematch只改变：

```text
trajectory_binding_mode
work_dir
```

manifest path、hash、(M,H,\alpha)、RNG、profile和evaluator完全相同。

---

## V.11 P1 — Feature provenance and clock

### Add

```text
opentad/utils/feature_provenance.py
tools/build_feature_provenance.py
tools/audit_raw_future_invariance.py
tests/test_feature_provenance_contract.py
```

### Modify

`streaming_feature.py`

-要求 new provenance artifact用于 effectiveness configs；
-旧 schema只允许 `compute_profile_only=true`。

`test_engine.py`

-支持 one-token primary packet；
-记录 availability/decision/publication clocks；
-cached replay明确标记 simulated。

---

## V.12 P2 only if positive

```text
opentad/models/adapters/causal_visual_lora.py
tools/audit_visual_parameter_updates.py
configs/causaltad/thumos_q2_raw_2x2_*.py
tests/test_raw_episode_prefix_invariance.py
tests/test_lora_optimizer_coverage.py
tests/test_lora_parameter_delta.py
```

---

## V.13 Defer

* multi-rank state sharding；
  -formal resume；
  -prefix-parallel architecture；
  -full tower；
  -new task/benchmark；
  -HSM/enclave attestation；
  -large-scale second dataset pipeline。

---

# W. Kill / Continue Rules

## W.1 Protocol implementation kill

立即 kill CRS‑EPS if：

* 任一 target bin零支持；
  -(w>2.5)；
  -probabilities无法由 manifest重算；
  -需要 GT slot/query initialization；
  -same-class/overlap birth无法重建；
  -episode manifest不能跨 variants exact share；
  -loss denominator不能闭包；
  -token RNG不能配对且 dropout差异足以污染 direct-effect audit。

## W.2 State-surrogate kill

完成 G0 后 kill if：

* dynamic replay没有显著优于 reset/fixed-192；
  -discrete canonical supervision state不一致；
  -sampled/full gradient方向越过预注册 fidelity margin；
  -runtime occupancy偏差无法接受；
  -大比例 episode退化到 video-start；
  -实际 replay cost使 CRS不再低于 full。

## W.3 Profile kill

Kill cost route if：

* projected pre-Stage2 total >10 GPU-hours；
  -Q2 train+eval >6 GPU-hours；
  -cost CI不显示降低；
  -data wait或cache heat解释全部加速；
  -savings只来自更少监督或不同 scheduler；
  -profile artifact不能闭包。

## W.4 One-seed Q2 kill

Kill Full PETAL headline if：

* fixed不优于 rematch；
  -primary mAP方向不正；
  -identity endpoints无改善；
  -recall/FN/delay恶化；
  -gain可由更少 emission解释；
  -direct-effect trace显示额外 mask/denominator/RNG差异；
  -slot exhaustion发生。

## W.5 Post-positive reconstruction kill

即使 fixed > rematch，仍 kill Full PETAL if：

-简单 recurrent control达到同样收益；
-FRESH/Temporal TrackFormer匹配；
-ActionSwitch-style control解释 overlap gain；
-增益只在 THUMOS、不在第二数据集；
-五种子 paired CI不稳定；
-surviving delta仍可写成 TrackFormer + temporal heads。

## W.6 Stage‑2 kill

Kill raw-video route if：

* frozen Q2证据不足；
  -LoRA没有独立收益；
  -LoRA只提高 aggregate mAP而不保留identity effect；
  -cost或causality失败；
  -需要 full tower 才工作；
  -继续使用 stale cached representation；
  -raw-stream wall-clock不可接受。

---

# X. Questions for the Authors

作者已经回答 Q1–Q12；本轮没有新的概念澄清问题。剩余的是必须在对应 gate 前提交的**签署项或证据项**：

1. 在 G0 variability 与 profile 完成后、任何 Q2 effectiveness result 可见前，签署：

   * (\delta_Q)；
   * recall margin；
   * delay margin；
   * minimum meaningful cost threshold。
2. 用 outcome-blind G0 precision/power 规则选择 (M)，然后冻结。
3. 恢复或重建 feature provenance。
4. 生成真实 211/213 source-derived diff。
5. 在 paper-stage 前确认 FineAction许可，或明确把最大 claim限制为 THUMOS-specific。

### 第二数据集裁决

**MultiTHUMOS 是更低成本 screening dataset**：它扩展 THUMOS，提供密集多标签标注，适合测试 overlap、occupancy 和 dense background。其原始定义强调 frame-level dense multi-label understanding，未自动保证对 same-class concurrent instances 的无歧义 identity。([arXiv][23])

**FineAction 是更科学充分的 confirmatory dataset**：它有 16,732 个视频、103,324 个 temporal instances、106 类、11.5% overlapping multi-label segments，但需要提交 academic license request，计算和数据成本显著更高。([arXiv][24])

因此：

```text
LOW-COST SCREENING = MultiTHUMOS, conditional on interval/identity audit
GENERAL IDENTITY CLAIM = FineAction preferred
```

若 FineAction不可用，MultiTHUMOS又无法提供可靠 instance identity，则最大结论必须限制为：

> THUMOS-specific fixed-supervision-policy observation。

---

# Y. Final Decision Card

```text
Repository HEAD observed:
3a297344df73d994c54cac082f34a66e597e9d44

Immutable commit reviewed:
f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb

Research verdict:
GO-HYBRID-PROTOCOL

Current training route:
FULL-CHRONOLOGICAL CACHED-FEATURE VIDEO-EPISODE
TRAINING WITH 64-TOKEN TRUNCATED BPTT AND ONE VIDEO-LEVEL
OPTIMIZER EVENT

CRS-EPS currently implemented:
NO

Task protocol verdict:
STANDARD FULLY SUPERVISED COMPLETION-TRIGGERED ON-TAD IS
PRESERVED; CACHED-TOKEN CAUSALITY IS SUPPORTED; RAW-VIDEO
CAUSALITY AND LOW WALL-CLOCK LATENCY REMAIN UNPROVEN

Q2 identifiability verdict:
CURRENTLY PARTIALLY IDENTIFIABLE;
CONDITIONALLY IDENTIFIABLE AFTER DIRECT-EFFECT TWIN TRACE,
TOKEN-ADDRESSED RNG, MATCHED REPLAY, DENOMINATOR TRACE AND
LONGITUDINAL MEDIATOR ACCOUNTING

Feature provenance verdict:
UNKNOWN FOR THE OLD CACHE; REBUILD REQUIRED BEFORE
EFFECTIVENESS TRAINING IF EXACT PROVENANCE CANNOT BE RECOVERED

Online clock verdict:
CURRENT LEDGER USES SOURCE CLOCK ONLY;
PRIMARY FORMAL EVALUATION MUST USE ONE-TOKEN PACKETS AND
SEPARATE SOURCE, AVAILABILITY, DECISION, COMPLETION AND
PUBLICATION CLOCKS

Recommended protocol:
HYBRID

Profile contract verdict:
REPLACE

Full PETAL novelty verdict:
RECONSTRUCTION AT PACKAGE LEVEL;
FIXED POST-BIRTH BINDING REMAINS A MARGINAL UNPROVEN DELTA

Raw-video Stage-2 permission:
BLOCKED

Profile permission:
BLOCKED

Formal training permission:
BLOCKED

Open P0:
P0-LAUNCH-WORKDIR

Open P1:
CRS-PROBABILITY-AND-WEIGHTING
STATE-RECONSTRUCTION-CONSISTENCY
GLOBAL-CHUNK-GRADIENT-CONSISTENCY
Q2-PAIRED-DIRECT-EFFECT-TRACE
LONGITUDINAL-MEDIATOR-TRACE
MULTI-DENOMINATOR-PROFILE
FEATURE-PROVENANCE
REAL-ONLINE-CLOCK
REPORTING-POPULATION-211
PRE-RESULT-NONINFERIORITY-MARGINS

Minimum fixes before next GPU-hour:
1. Fix launch work_dir identity from one immutable source.
2. Add deterministic fake-sbatch shell integration test.
3. Implement and CPU-audit immutable episode manifest, q/rho/pi/weights.
4. Implement token-addressed RNG and direct fixed/rematch twin trace.
5. Implement dynamic replay plus video-start gold audit runner.
6. Implement multi-denominator profile artifact and budget gate.
7. Freeze a new clean commit.
8. Regenerate complete B0.
9. Obtain a fresh independent PASS / PROFILE=ALLOW.

Exact next implementation step:
FIX P0-LAUNCH-WORKDIR AND ADD THE REAL DETERMINISTIC
FAKE-SBATCH IDENTITY-CLOSURE TEST; THEN IMPLEMENT ONLY THE
OUTCOME-BLIND CRS-EPS MANIFEST/PROBABILITY CPU PATH.

Exact kill condition:
KILL CRS-EPS AND THE FULL PETAL HEADLINE IF EXACT SAMPLING
SUPPORT/WEIGHTING CANNOT BE CLOSED, IF DYNAMIC REPLAY FAILS
THE FULL-STREAM STATE/LOSS/GRADIENT FIDELITY GATE, IF ITS
ACTUAL REPLAY COST REMOVES THE GPU-HOUR ADVANTAGE, OR IF
FIXED BINDING FAILS TO PRODUCE A PAIRED IDENTITY-LINKED GAIN
WITHOUT RECALL OR COMPLETION-DELAY REGRESSION.

Confidence:
HIGH ON CODE SCOPE, CURRENT CRS ABSENCE, P0, TARGET-RISK
FORMULATION, PROFILE REPLACEMENT AND MULTI-PAPER
RECONSTRUCTION RISK;
MEDIUM ON HYBRID PRACTICAL VALUE UNTIL G0/PROFILE;
UNKNOWN ON Q2 EFFECTIVENESS.
```

```text
RESEARCH_VERDICT=GO-HYBRID-PROTOCOL
CURRENT_ROUTE=FULL_CHRONOLOGICAL_CACHED_VIDEO_EPISODE_TRUNCATED_BPTT
CRS_EPS_IMPLEMENTED=NO
RECOMMENDED_PROTOCOL=HYBRID
Q2_IDENTIFIABILITY=PARTIAL_NOW_CONDITIONAL_AFTER_PAIRED_TRACE
PROFILE_CONTRACT=REPLACE
FULL_PETAL_NOVELTY=RECONSTRUCTION
RAW_VIDEO_STAGE2=BLOCKED
PROFILE=BLOCKED
FORMAL_TRAINING=BLOCKED
NEXT_STEP=FIX_P0_LAUNCH_WORKDIR_AND_ADD_FAKE_SBATCH_INTEGRATION_TEST
OPEN_P0=P0-LAUNCH-WORKDIR
OPEN_P1=CRS_PROBABILITY,STATE_FIDELITY,GRADIENT_BOUNDARIES,PAIRED_TRACE,PROFILE_DENOMINATORS,FEATURE_PROVENANCE,ONLINE_CLOCK,REPORTING_211,PRE_RESULT_MARGINS
```

[1]: https://arxiv.org/abs/2205.07134 "https://arxiv.org/abs/2205.07134"
[2]: https://arxiv.org/abs/2204.01680 "https://arxiv.org/abs/2204.01680"
[3]: https://arxiv.org/abs/2211.04905 "https://arxiv.org/abs/2211.04905"
[4]: https://arxiv.org/abs/2408.02957 "https://arxiv.org/abs/2408.02957"
[5]: https://arxiv.org/abs/2408.06437 "https://arxiv.org/abs/2408.06437"
[6]: https://arxiv.org/abs/2407.12987 "https://arxiv.org/abs/2407.12987"
[7]: https://arxiv.org/abs/2607.00289 "https://arxiv.org/abs/2607.00289"
[8]: https://arxiv.org/abs/2605.09976 "https://arxiv.org/abs/2605.09976"
[9]: https://arxiv.org/abs/2509.12145 "https://arxiv.org/abs/2509.12145"
[10]: https://arxiv.org/abs/2306.07703 "https://arxiv.org/abs/2306.07703"
[11]: https://arxiv.org/abs/2504.20041 "https://arxiv.org/abs/2504.20041"
[12]: https://arxiv.org/abs/2209.09236 "https://arxiv.org/abs/2209.09236"
[13]: https://arxiv.org/abs/2411.01122 "https://arxiv.org/abs/2411.01122"
[14]: https://arxiv.org/abs/2211.14053 "https://arxiv.org/abs/2211.14053"
[15]: https://arxiv.org/abs/2205.02717 "https://arxiv.org/abs/2205.02717"
[16]: https://arxiv.org/abs/2311.17241 "https://arxiv.org/abs/2311.17241"
[17]: https://arxiv.org/abs/2404.01282 "https://arxiv.org/abs/2404.01282"
[18]: https://arxiv.org/abs/2101.02702 "https://arxiv.org/abs/2101.02702"
[19]: https://arxiv.org/abs/2105.03247 "https://arxiv.org/abs/2105.03247"
[20]: https://arxiv.org/abs/2307.15700 "https://arxiv.org/abs/2307.15700"
[21]: https://arxiv.org/abs/2301.01882 "https://arxiv.org/abs/2301.01882"
[22]: https://arxiv.org/abs/2211.09108 "https://arxiv.org/abs/2211.09108"
[23]: https://arxiv.org/abs/1507.05738 "https://arxiv.org/abs/1507.05738"
[24]: https://arxiv.org/abs/2105.11107 "https://arxiv.org/abs/2105.11107"
