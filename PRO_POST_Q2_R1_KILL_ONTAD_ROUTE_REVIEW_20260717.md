## A. Repository/commit verification

**审查证书**

| 项目         | 核验结果                                                                                                                                                                  |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Repository | `[CODE-VERIFIED]` `yuzbo/OpenTAD_OnlineTADClean_20260702` 为公开仓库；GitHub 页面确认目标分支 `codex/full-petal-implementation` 存在。([GitHub][1])                                    |
| 固定提交       | `[CODE-VERIFIED]` `7a26e60fccec68cd2b547951320e669a81907461` 可由 GitHub connector 直接解析，提交信息为 `Record terminal Q2 R1 kill decision`。                                    |
| 移动分支       | `[CODE-VERIFIED]` 审查时该分支比固定提交前进 1 个提交、落后 0 个；新增内容仅为本轮 prompt 及 source/log 索引记录，没有把后续科学代码混入本审查。                                                                        |
| 实际审查对象     | `[CODE-VERIFIED]` 所有代码、配置、测试与研究文档读取均显式使用固定 ref `7a26e60f…`，而不是分支 HEAD 或默认分支。                                                                                          |
| 原始证据可见性    | `[UNKNOWN]` CRS-EPS 与 Q2 的原始 checkpoint、压缩 trace 和签名 evidence bundle 位于仓库外；仓库保存其哈希、审计结论和独立复核记录。本轮无法重新读取原始字节，因此不能声称由我再次重放了 524,258 行 trace。CRS 文档也明确说明原始 bundle 不进入仓库。 |
| 执行纪律       | `[CODE-VERIFIED]` 本轮未修改仓库、未运行训练、未运行 Gate、未创建 commit/PR、未使用 GPU。                                                                                                       |

**读取覆盖范围**

`RTK.md`、query/index、DR-041–044、T34–T37、CRS-EPS G0、Q2 capacity audit、CSFSB 文档、两份 Round-2 文档、独立 Q2 max review、persistent head/detector、prefix supervision、Q2 audit utilities/runner、fixed/rematch configs、train engine、optimizer audit、launch validator、dataset、evaluator和相关测试均已交叉核验。

---

## B. Executive verdict

**最终裁决：`GO_NEW_ROUTE_P0_ONLY`**

`[EVIDENCE-VERIFIED]` 当前 Q2 与 R1/CSFSB 的处置保持永久 `KILL`。不得采用 `-2` birth prior，不得更改 threshold、K、refractory 或 release order 来重新打开旧合同，不得运行新 Q2 确认审计。DR-044 已明确规定零 GPU、无 R1、无 formal training。

`[INFERENCE]` 仍存在一条只值得获得 **零 GPU、CPU-only P0 机制证伪机会** 的新路线：

> **R-A：Prefix-Shared Latent Event Filter with Immutable Ledger**
> 工作中文名：**前缀共享潜在事件过滤器与不可变账本**。

它不是 Q2 修补版。它保留固定数量的潜在事件载体，但废除：

* runtime `FREE/ACTIVE/REFRACTORY` 对监督资格的控制；
* GT canonical slot 作为训练状态；
* assignment 对下一时刻模型状态的修改；
* train-only hard lifecycle；
* 由静默策略换取“零容量冲突”的门禁。

其核心合同是：

1. train/inference 调用同一个 model-only `advance(state, causal_input)`；
2. GT assignment 只是一项 loss-side 前缀 OT 权重，永不写入 runtime；
3. completion、soft release、same-bin reseed 均由模型输出驱动；
4. hard ledger 只负责不可变提交，不参与 risk set 或 target availability；
5. no-birth、no-emission、always-background 必须在冻结 P0 中确定失败。

`[INFERENCE]` 这只构成 **P0 级条件创新通过**，不构成论文创新通过。其最大风险是被审稿人重建为“1D MOTR/TrackFormer + interval heads + online ledger”。因此：

* P0 code：允许；
* cached-feature G0：不允许，须待 P0 独立复核；
* GPU profile：不允许；
* formal/effectiveness training：不允许；
* raw-video joint training：不属于本次授权；
* 当前 GPU 授权：**0 小时**。

---

## C. Evidence ledger

| ID  | 核验事实                                                                                                                                           | 标签与处置                                                                                           |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| E1  | CRS-EPS G0 在预冻结门槛下返回 `KILL`；dynamic replay 的最差 loss error、gradient cosine、state cosine 均大幅越界。                                                  | `[EVIDENCE-VERIFIED]` 负证据成立，但不是 On-TAD 效果实验。                                                    |
| E2  | 后续原始 artifact 复核确认 dynamic replay 从空 runtime 开始，未携带完整 prefix 状态；HH/IPW 无法恢复缺失的状态分布和历史 Jacobian。                                                | `[EVIDENCE-VERIFIED]` 当前 empty-state CRS-EPS 为结构性 KILL，不否定所有 state-faithful selective backward。 |
| E3  | Q2 clean audit 覆盖 160 个 fit-core 视频、3 seeds；actual exhaustion=2,206，oracle minimum K=2，true canonical capacity=0。                              | `[EVIDENCE-VERIFIED]` 当前 Q2 不具备共享 lifecycle readiness。                                          |
| E4  | `-2` arm 只产生 10 次 emission，seeds 705/706 为零 emission；独立 reviewer 解析 524,258 行 trace 后裁决 `DEGENERATE_BIRTH_SUPPRESSION`。                        | `[EVIDENCE-VERIFIED]` 零 exhaustion 是静默逃逸，不是修复。                                                  |
| E5  | 训练循环在每 bin 先读取 runtime `FREE`，再执行 head，随后以这些槽构建 canonical transition 和 loss，最后才运行 inference `decode_step`。                                     | `[CODE-VERIFIED]` runtime lifecycle 与 supervision lifecycle 非同构。                                |
| E6  | canonical transition 把未获槽的 birth 也加入 `born_instance_ids`，随后记录为 exhausted；它之后不会作为新 birth 重试。                                                    | `[CODE-VERIFIED]` exhaustion 会造成永久监督所有权丢失。                                                      |
| E7  | runtime head 的 query/memory recurrence 不读取 slot status、refractory 或 canonical identity；hard lifecycle 只存在于 detached decoder。                   | `[CODE-VERIFIED]` Q2 的“事件身份”主要是 decoder bookkeeping，而不是条件化的神经状态。                                |
| E8  | hard decoder 以 birth/alive/end 阈值完成 `FREE→ACTIVE→REFRACTORY`，并提交不可变 event record。                                                              | `[CODE-VERIFIED]` inference 不含 GT，但与训练 canonical 风险集不同。                                         |
| E9  | fixed/rematch 两 arm 只改变 `trajectory_binding_mode`；canonical 生命周期仍共享，rematch 仅改变 loss binding。                                                  | `[CODE-VERIFIED]` Q2 是一个合法的一因素 binding 对照，但共同 lifecycle 本身失败。                                   |
| E10 | 精确提交中的 train engine 已在 backward 前检查 exhaustion，rollback 后抛出 scientific failure。                                                                | `[CODE-VERIFIED]` 现在不会继续优化坏轨迹；这只是 fail-close，不是机制修复。                                            |
| E11 | 当前 Q2 配置为固定 cached causal features、K=4、memory=192、`detach_stream_state=True`、frozen visual route；显式声明不是 raw-video joint training。              | `[CODE-VERIFIED]` 当前结果只能称 feature-level/head-level。                                             |
| E12 | evaluator 要求 immutable ledger、`predicted_end ≤ emit`、`source ≤ emit`，并报告 mOnlineAP、late FN/FP、duplicate、fragmentation 等。                       | `[CODE-VERIFIED]` 评测器具备关键在线完整性检查。                                                               |
| E13 | optimizer audit 可发现遗漏、重复、unknown 及 frozen-in-optimizer 参数。                                                                                     | `[CODE-VERIFIED]` 软件级 optimizer coverage 基础设施可复用。                                               |
| E14 | launch gate 绑定 commit、source tree、resolved/scientific config、data/runtime identity、B0/review/G0/profile evidence，并要求无 skipped optimizer event。 | `[CODE-VERIFIED]` provenance/launch 基础设施可复用，但旧 Q2 的 PASS ticket 不得移植到新路线。                       |

---

## D. Unknown Register

### D1. 已关闭项目

| ID  | 问题                                            | 关闭结论                                                                                                                                                 |
| --- | --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| U01 | 训练和推理状态转移是否逐 bin 同构                           | `[CODE-VERIFIED] NO`。训练先以 pre-decode runtime availability 分配 GT，再 decode；推理只有 decode。                                                                |
| U02 | GT assignment 能否进入 runtime 不可用槽               | `[CODE-VERIFIED]` 不能。birth candidates 是 canonical-free 与 runtime-free 的交集；结果是 exhaustion，而不是写入不可用槽。                                                  |
| U03 | exhausted birth 是否可稍后补分配                      | `[CODE-VERIFIED]` 不能。所有 birth 立即进入 `born_instance_ids`，重复 birth 被禁止。                                                                                 |
| U04 | birth/end 时刻与 first-event 语义                  | `[CODE-VERIFIED]` birth 和 end 都要求跨越上一 decision frame；birth/active 不含未来 endpoint，end 仅在 endpoint 首次可见时出现。                                             |
| U05 | loss risk set                                 | `[CODE-VERIFIED]` birth 只在 birth candidates；alive 在全部槽；end 只在 canonical/loss-bound at-risk 槽；class/start 只在 loss bindings。                           |
| U06 | same-bin end/birth reuse                      | `[CODE-VERIFIED]` canonical endpoint 直到本 bin supervision 完成后才退休，当前实现禁止 same-bin release/reuse。                                                       |
| U07 | detach/TBPTT 边界                               | `[CODE-VERIFIED]` runtime query、feature memory及 lifecycle tensors 在 packet 后被 `.detach()`；完整 chronological forward state 被保留，但跨 packet Jacobian 被切断。 |
| U08 | long action起点早于 memory                        | `[CODE-VERIFIED]` pointer sentinel 解码为最早 retained frame；当前 scalar start target 被截断到 `memory_size`。这不是精确的超窗起点恢复。                                      |
| U09 | evaluator 对 late、miss、duplicate、fragmentation | `[CODE-VERIFIED]` latency-budget 外预测为 FP，未匹配 GT 为 FN；一个 GT 只锁定一次，额外匹配构成 duplicate/false emission，并有 identity diagnostics。                            |
| U10 | 当前 route 是否 raw-video end-to-end              | `[CODE-VERIFIED] NO`。配置为 `fixed_cached_causal_features`，`raw_video_joint_training=False`。                                                            |

### D2. 仍开放项目

| ID  | 未知项                                                                                                   | 最小解法                                                                                         | 阻塞范围                                               |
| --- | ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| U11 | `[UNKNOWN]` `pes_siglip2_stride8` 外部 encoder 的内部时域 receptive field 是否严格只看 `≤source_frame`             | 发布 encoder 代码、权重 ID、clip/frame construction、mask、预处理和逐帧 perturbation certificate             | 阻塞 P1 数据级因果 claim、全部 P3 效果结论                       |
| U12 | `[UNKNOWN]` THUMOS fit/cal 中 same-class overlap、same-bin birth+end、同 bin 多 endpoint、超 memory 长动作的精确频率 | 运行 annotation-only、零模型、零 GPU census，并提交 manifest/hash                                        | 阻塞最终 K、direct-complete 分支和第二数据集选择；不阻塞 synthetic P0 |
| U13 | `[UNKNOWN]` 外部 CRS/Q2 原始 evidence bundle 是否仍可供新的独立 reviewer 逐字节重放                                     | 提供只读 bundle 和现有哈希路径                                                                          | 不阻塞接受冻结 KILL；阻塞“由本 reviewer 再次独立重放”主张              |
| U14 | `[UNKNOWN]` 2024 后是否存在未被当前检索命中的标准 fully supervised、raw-RGB joint On-TAL                               | 在投稿前做一次作者/引用网络和会议 proceedings 补搜                                                             | 不阻塞 P0；阻塞最终 novelty claim                          |
| U15 | `[UNKNOWN]` MATR、ActionSwitch、HAT、SimOn 在本仓库 evaluator 下的可复现相对强弱                                      | 固定公开 checkpoint/config 或做参数预算匹配重建                                                            | 阻塞 P3 论文级 baseline                                 |
| U16 | `[UNKNOWN]` raw-video joint training 的实际 VRAM、吞吐和训练成本                                                 | P0/P1 后，以确定 GPU 型号做固定步 profile                                                               | 阻塞任何 raw-video claim；当前不授权                         |
| U17 | `[UNKNOWN]` 最合适的第二数据集                                                                                 | 对 FineAction、EPIC-Kitchens-100、MUSES 做盲 annotation census：same-class overlap、并发度、长度、许可和特征可用性 | 阻塞双数据集论文结论                                         |
| U18 | `[UNKNOWN]` 全测试套件在固定提交上的实际运行结果                                                                        | 在新路线 commit 上执行本地与目标 Linux B0                                                                | 当前仅作静态代码核验；阻塞 P1                                   |
| U19 | `[UNKNOWN]` 新路线的阈值能否在 calibration 上稳定而非依靠 silent operating point                                      | P0 synthetic anti-silence + P1 unseen CPU G0                                                 | 阻塞 profile 和 formal training                       |

---

## E. Exact current task and implementation

### E1. 当前任务

`[CODE-VERIFIED]` 配置将任务声明为：

* standard fully supervised；
* completion-triggered On-TAD；
* 输入为固定 cached causal features；
* 不预测未来 endpoint；
* immutable emissions；
* 无 offline NMS；
* 无 raw-video joint training。

`[LITERATURE-VERIFIED]` 这与标准 On-TAL 定义一致：只能使用到当前为止的帧，先前提交的实例不可删除或回改。MATR 对这一合同有明确表述。([arXiv][2])

### E2. 当前实际优化对象

`[CODE-VERIFIED]`

[
L=
L_{\text{birth}}
+0.5L_{\text{alive}}
+L_{\text{class}}
+L_{\text{start}}
+L_{\text{end}}.
]

具体为：

* birth BCE：仅 runtime-free ∩ canonical-free 槽参与；
* alive BCE：所有槽参与，canonical occupied 为正；
* end BCE：仅 canonical/loss-bound at-risk 槽参与；
* class CE、start Smooth-L1：仅当前 loss bindings；
* 每个有效 bin 的 loss 先相加，再除以有效 bin 数。

`[INFERENCE]` 该目标优化的是“在 GT canonical 所有权和当前预测 runtime availability 共同允许的条件下，各 head 的局部正确性”，而不是一个对完整模型生成 lifecycle 的可微 likelihood。

### E3. Runtime lifecycle

`[CODE-VERIFIED]`

* 固定 K=4；
* 每个载体有 persistent query、feature memory；
* hard status 为 `FREE/ACTIVE/REFRACTORY`；
* FREE 中 birth 超阈值后 ACTIVE；
* ACTIVE 中 alive/end 决定静默释放或 emission；
* emission 后进入 refractory；
* record 立即写入 immutable ledger。

神经 query 每 bin 都更新，但不读取这些 hard status，因此 hard lifecycle 不重置或条件化 query recurrence。

### E4. Canonical supervision

`[CODE-VERIFIED]`

* GT instance 在 first start crossing 时分配 canonical slot；
* fixed arm 保持该 slot；
* rematch arm只对当前 loss binding 做重匹配，canonical ownership 不变；
* endpoint 本 bin仍占槽，loss 构建完成后退休；
* exhaustion birth 被标记为 born，因此永久失去后续 birth 监督。

### E5. Inference decode

`[CODE-VERIFIED]`

* 只使用模型 logits、thresholds和 runtime state；
* 不读取 GT；
* endpoint mode 在 Q2 中为 binary，因此常规 emission end 为当前 decision frame；
* start 使用 scalar offset；
* 输出包含 source/emit/start/end frame 并不可变提交。

### E6. 关键不一致

`[CODE-VERIFIED]`

1. 监督在 decoder 更新 lifecycle **之前**读取槽可用性；
2. GT canonical ownership 与预测 hard status 是两套状态；
3. assignment 决定哪些 logits 获得正监督，但 assignment 不可微；
4. inference 的 false ACTIVE/refractory 可阻断未来 GT birth；
5. exhausted birth 不会重试；
6. query recurrence 与 hard lifecycle 解耦；
7. packet detach 保留 forward state，但不保留完整历史 Jacobian。

这六项共同说明当前实现不是“同一 prefix-observable state machine 的 train/inference 两种模式”。

---

## F. Root-cause analysis of CRS-EPS and Q2 failures

### F1. CRS-EPS

`[EVIDENCE-VERIFIED]` empty-state dynamic replay 从空 persistent runtime 开始，缺少真实 chronological prefix 的：

* recurrent query；
* feature memory；
* hard lifecycle；
* canonical/loss ownership；
* 产生当前状态的历史 Jacobian。

三个非 fallback 样本的 absolute fidelity 均失败，部分 dynamic replay 与 fixed-192 甚至产生完全相同的坏指标。

`[INFERENCE]` HH/IPW 只能对“在正确状态下得到的随机梯度贡献”做抽样概率修正。它不能把在错误状态 (S'_t) 上计算的

[
\nabla_\theta L(\theta,S'_t)
]

变成真实 chronological 状态 (S_t) 上的

[
\nabla_\theta L(\theta,S_t),
]

更不能凭权重补造被切掉的

[
\frac{\partial S_t}{\partial\theta}
]

历史路径。因此这是 estimand 改变，不是 variance 过大。

### F2. Q2

`[EVIDENCE-VERIFIED]` Q2 的 2,206 exhaustion 不是 K 不足：annotation oracle 只需 K=2，且 `TRUE_CANONICAL_CAPACITY=0`。主要归因为 refractory 和 false-ACTIVE/mixed occupancy。release-before-birth 与 refractory=0 仍不能清零。

因果分类如下：

| 候选解释                                 | 裁决                                                                                                                                                                                               |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 单一实现 bug                             | `[INFERENCE]` **不是充分解释**。代码按既定合同执行，测试也刻意覆盖 exhaustion；问题是合同设计错误，而非单个 off-by-one。                                                                                                                 |
| 标准 On-TAD 任务本身矛盾                     | `[INFERENCE]` **否**。任务允许模型状态与不可变输出；矛盾来自训练监督与模型状态使用了不同 lifecycle。                                                                                                                                 |
| teacher-forcing exposure bias        | `[INFERENCE]` **部分成立**。GT 未直接写入 runtime tensor，但 GT canonical ownership 决定 loss identity/risk，推理只有预测 lifecycle。因此属于 supervision-conditioned state exposure mismatch，而不是传统意义的直接 GT runtime taint。 |
| 离散 assignment 不可微                    | `[INFERENCE]` **重要共因**。模型不能通过梯度学习“当前 hard lifecycle 如何改变未来 target availability”；但它不是 2,206 的唯一机械原因。                                                                                              |
| persistent-slot 路线整体失败               | `[INFERENCE]` **未被证明**。被杀死的是“hard runtime availability gating GT canonical supervision”的实现族。固定载体、但 assignment 纯 loss-side、状态完全 model-generated 的路线尚未被该证据否定。                                      |
| K、threshold、prior 或 refractory 调参可修复 | `[EVIDENCE-VERIFIED]` **否**。`-2` 通过几乎不发射来消除压力，是非识别性静默解。                                                                                                                                          |

### F3. 两次失败的共同教训

`[INFERENCE]`

* CRS-EPS 失败：**省略了生成状态的历史计算**；
* Q2 失败：**用不同状态机定义训练所有权和推理行为**；
* 两者共同要求：新路线必须保持完整 chronological forward state，并让 GT 仅评价模型状态，不能创造或修正模型状态。

---

## G. Current On-TAD gap audit

### G1. 已被现有工作部分解决的问题

`[LITERATURE-VERIFIED]`

* SimOn 已用当前 frame feature 查询过去视觉上下文，并明确遵守无未来、不可修改旧预测；但它所谓 “end-to-end” 是 temporal detector 层面，输入仍是 frame feature。([arXiv][3])
* OAT/MATR 将训练从 frame-wise grouping 推向 instance-level query/Hungarian；MATR用 memory queue改善长动作 start retrieval。([arXiv][2])
* HAT 明确针对长期 history，并在 THUMOS/MUSES及 egocentric 数据上验证。([arXiv][4])
* ActionSwitch 明确处理 class-agnostic simultaneous actions，包括 same-class overlap。([arXiv][5])

### G2. 为什么标准 On-TAD 仍难

`[INFERENCE]`

1. **完成时才可可靠提交。** 过早提交损害 endpoint/IoU，过晚提交损害 latency。
2. **start 与 end 的证据尺度不同。** end 在当前 prefix 附近，start 可能远在 memory 之外。
3. **不可变错误没有离线修复。** duplicate、fragmentation和错误 endpoint 一旦提交就永久存在。
4. **同类重复和重叠要求实例身份，而非只做 class state。**
5. **训练监督容易使用 GT identity/risk set，而推理只能使用预测状态。**
6. **冻结特征限制边界表征适配；raw video 又带来巨大训练成本。**
7. **threshold/decode 是任务机制的一部分，不能被当作无关后处理。**

MATR 本身也暴露了典型 shift：训练时 memory flag 使用 GT，推理时使用预测 flag；同时它采用在线 NMS 去除当前及过去 proposals 的重复。([arXiv][2])

### G3. 仍由公开证据支持的 gap

`[INFERENCE]`

> **标准 fully supervised On-TAD 尚缺少一个经过实证验证的合同：模型生成的 latent instance state 在 train/inference 中逐 prefix 完全共用，GT assignment 只参与 loss，不改变 runtime identity、availability、reset、risk set 或 emission。**

这是比“加 causal backbone”“加 memory”“加 persistent query”更窄、但更可识别的 gap。

### G4. “尚无真正端到端可训练 On-TAD”是否准确

**不准确，必须拆开说。**

| 含义                                                              | 审查结论                                                                                                      |
| --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| detector/head end-to-end                                        | `[LITERATURE-VERIFIED]` 已存在。SimOn、MATR都使用该措辞。([arXiv][3]) ([arXiv][2])                                    |
| cached/pre-extracted feature 上的 temporal model 联合训练             | `[LITERATURE-VERIFIED]` 已是 On-TAL 主流。                                                                     |
| frozen visual encoder + trainable temporal head                 | `[LITERATURE-VERIFIED]` 已存在；MATR等使用 pretrained backbone/features。([arXiv][2])                             |
| backbone fine-tuning                                            | `[UNKNOWN]` 在标准 fully supervised On-TAL 中，本次检索未确认一个被广泛接受的代表性实现。                                           |
| raw RGB → causal backbone → immutable interval 的 joint training | `[UNKNOWN]` 本次查新没有确认标准 fully supervised On-TAL 论文；不能据此宣称绝对无人做过。                                           |
| raw-video joint OAD                                             | `[LITERATURE-VERIFIED]` 已存在，E2E-LOAD明确是 raw/end-to-end OAD，但它输出当前帧动作，不是完成触发的 interval On-TAL。([arXiv][6]) |
| raw-video joint offline TAD                                     | `[LITERATURE-VERIFIED]` 已存在于 Re²TAL、AdaTAD等，但允许整段视频，任务不同。([arXiv][7]) ([arXiv][8])                        |

**可防御表述：**

> 截至 2026 年 7 月 16 日的本次检索，在核验到的标准 fully supervised On-TAL 工作中，尚未确认一个从 raw RGB 到不可变 action instance 的严格因果联合训练系统；但 detector-level “end-to-end On-TAL” 已存在，raw-video joint training 也已在 OAD 与 offline TAD 中存在。

---

## H. Literature search protocol and dated source table

### H1. 协议

* 检索日期：**2026-07-16，America/New_York**
* 时间覆盖：2020–2026；重点补搜 2025–2026。
* 来源优先级：原论文 arXiv/CVF/ECCV/正式 proceedings、官方项目页。
* 核心检索式：

  * `"online temporal action localization" 2025 2026 fully supervised`
  * `site:arxiv.org "online temporal action localization" 2025 2026`
  * `"end-to-end online action detection" raw frames`
  * `"StreamFormer" online action detection causal`
  * `"TrackFormer" OR "MOTR" track query assignment`
  * `"end-to-end temporal action detection" raw video`
  * `"Neural Hawkes Process"`, `"Transformer Hawkes Process"`
  * `"Sequence Transduction with Recurrent Neural Networks"`
* 排除规则：

  * OZ-TAL、POTAL、OpenHOUSE 等用于边界/竞争审查，但不得冒充标准 fully supervised On-TAD；
  * OAD 不冒充 interval On-TAD；
  * offline TAL 不冒充 online causal TAL；
  * 搜索未命中不等于绝对不存在。

### H2. 日期化来源表

| ID  | 来源与核验内容                                                                                                                                                                           |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| L1  | `[LITERATURE-VERIFIED]` **SimOn, 2022**：标准无未来、不可修改历史预测；输入是 current frame feature 与过去上下文。([arXiv][3])                                                                              |
| L2  | `[LITERATURE-VERIFIED]` **MATR, ECCV 2024**：memory queue、当前段检测 end、memory 回溯 start；训练 GT flag、推理 predicted flag；online NMS。([arXiv][2])                                           |
| L3  | `[LITERATURE-VERIFIED]` **ActionSwitch, ECCV 2024**：class-agnostic simultaneous/same-class overlap。([arXiv][5])                                                                   |
| L4  | `[LITERATURE-VERIFIED]` **HAT, ECCV 2024**：历史增强 anchor transformer，覆盖 THUMOS/MUSES和 procedural egocentric 数据。([arXiv][4])                                                         |
| L5  | `[LITERATURE-VERIFIED]` **OZ-TAL, 2026**：zero-shot、training-free 新任务，不能作为标准 fully supervised 路线占位。([arXiv][9])                                                                    |
| L6  | `[LITERATURE-VERIFIED]` **OnPoint, 2026**：point-supervised POTAL 与 offline teacher distillation，属于监督定义变化。([arXiv][10])                                                            |
| L7  | `[LITERATURE-VERIFIED]` **OpenHOUSE, 2025**：层次化开放描述 streaming task，属于任务扩展。([arXiv][11])                                                                                           |
| L8  | `[LITERATURE-VERIFIED]` **E2E-LOAD, 2023**：raw/end-to-end OAD、长序列 cache、当前帧分类；不是 interval On-TAL。([arXiv][6])                                                                     |
| L9  | `[LITERATURE-VERIFIED]` **StreamFormer, 2025**：因果 temporal attention streaming backbone，验证 OAD/VIS/VQA；提供 backbone competition，不直接解决 On-TAL lifecycle。([arXiv][12])               |
| L10 | `[LITERATURE-VERIFIED]` **TrackFormer / MOTR, 2021**：persistent track queries、newborn queries、frame-to-frame set prediction、tracklet-aware assignment。([arXiv][13]) ([arXiv][14]) |
| L11 | `[LITERATURE-VERIFIED]` **TadTR / TE-TAD**：offline query-set temporal detection，无 online immutable contract。([arXiv][15]) ([arXiv][16])                                           |
| L12 | `[LITERATURE-VERIFIED]` **Re²TAL / AdaTAD**：raw-video/offline TAD 的可逆或 adapter 高效联合训练。([arXiv][7]) ([arXiv][8])                                                                   |
| L13 | `[LITERATURE-VERIFIED]` **Neural Hawkes / Transformer Hawkes**：根据过去事件建模 event type/time intensity，但不直接预测 video interval identity。([arXiv][17]) ([arXiv][18])                      |
| L14 | `[LITERATURE-VERIFIED]` **RNN-T**：对无预对齐输入/输出序列边缘化 monotone alignment；可作为 causal event-token 路线的最近基础。([arXiv][19])                                                                 |

---

## I. Candidate route portfolio

### R-A. Prefix-Shared Latent Event Filter with Immutable Ledger

**状态：保留，排名第一。**

* `[INFERENCE]` **任务内 gap：** 现有 On-TAD 没有验证“模型状态逐 prefix train/inference 同构、GT 只评估而不驱动状态”的实例级合同。
* `[INFERENCE]` **输入：** 首阶段只用已审计的 causal cached features；后续 visual fine-tuning另立合同。
* `[INFERENCE]` **状态：** K 个连续潜在事件载体、连续 occupancy、start posterior、class evidence、causal memory及 model-only ledger latch。
* `[INFERENCE]` **转移：** 模型预测 completion mass，先 soft release，再允许 same-bin reseed；GT 不参与。
* `[INFERENCE]` **assignment：** prefix-visible unbalanced OT，仅形成 loss 权重；不写入 runtime，也不决定 availability。
* `[INFERENCE]` **risk set：** 所有载体均参与；end 使用 first-event survival risk；post-end 正样本不重复。
* `[INFERENCE]` **输出：** model-only threshold 产生 `{start,end,class,score}`，立即进入 immutable ledger。
* `[INFERENCE]` **避免 Q2：** 没有 runtime-free ∩ canonical-free 交集，也没有 exhaustion 作为监督所有权丢失。
* `[INFERENCE]` **反静默：** event recall、precision、count、same-class overlap、all-background FP 全部进入冻结 readiness gate。
* `[INFERENCE]` **最接近工作：** MOTR/TrackFormer + ActionSwitch/MATR。
* `[INFERENCE]` **最小实质差异：** tracking assignment不写模型状态；soft release/reseed与不可变 completion ledger共用同一 prefix state。
* `[INFERENCE]` **端到端含义：** P0/P1 为 head-only/cached-feature；不得称 raw-video end-to-end。
* `[INFERENCE]` **成本：** 约 850–1,200 行科学运行代码，另有 700–1,000 行测试/审计。
* `[INFERENCE]` **最强 kill：** 参数、features和 evaluator 匹配的 temporal-MOTR reconstruction 若得到相同 identity/latency结果，则论文 novelty 被杀死。
* `[INFERENCE]` **可证伪预测：** 相对 completion-only set baseline，在同类重复、重叠和长动作上 duplicate/fragmentation下降，而 recall 不下降。

### R-B. Endpoint-Set Completion Process

**状态：仅保留为最强简单 baseline，不作为主路线。**

* `[INFERENCE]` 每 bin 用 K 个局部 query 只预测本 bin 首次完成的事件集合；
* `[INFERENCE]` 只做当前-bin Hungarian/OT，不维持跨 bin 事件身份；
* `[INFERENCE]` 无容量 exhaustion、训练推理天然一致；
* `[INFERENCE]` first-event likelihood 可排除 repeated late positives；
* `[INFERENCE]` 但它与 MATR 的“current end + memory start”、OAT/TadTR query set，以及仓库已有 PCEH 高度重合。

仓库自身已记录 PCEH 的 endpoint/emission 混同、重复 late positives、class-level target 和 GT proximity 等问题，并要求 endpoint-only baseline。

**裁决：`occupied / reconstruction-risk`。**

### R-C. Causal Interval Transducer

**状态：保留为理论 fallback，不授权实现。**

* `[INFERENCE]` 把每个不可变 interval 表示为内部 token sequence：`BEGIN → class/start mark → COMMIT`，blank 表示无输出；
* `[INFERENCE]` 使用 RNN-T 类 monotone alignment marginalization，不需要预先固定每个输入 bin 对应哪个输出；
* `[INFERENCE]` 同 bin 多事件由 set-valued emission lattice 或确定性 canonical token order处理；
* `[INFERENCE]` runtime state 完全由历史输入与已生成 token构成；
* `[INFERENCE]` no-emission 会在正事件 lattice 上产生高 NLL，不能从训练目标上静默通过；
* `[INFERENCE]` 输出仍是标准 immutable interval，内部 token 不改变任务。

**主要风险：**

* 容易被审稿人视为 RNN-T/point-process 跨领域移植；
* same-bin unordered set 与 token order产生人为建模负担；
* 实现预计 1,400–2,200 行科学代码；
* P0 比 R-A 明显更大。

**裁决：`unknown / high reconstruction-risk`。**

### R-D. Teacher–Student Runtime State Alignment

**状态：拒绝。**

* `[INFERENCE]` GT-aware teacher 与 model-only student 对齐 hidden state；
* `[INFERENCE]` 它不能证明 teacher state 对推理可实现，也可能把 canonical mismatch隐式蒸馏进 student；
* `[LITERATURE-VERIFIED]` teacher/distillation 已被 OnPoint 等邻近路线占用，但监督任务又不同。([arXiv][10])
* `[INFERENCE]` 最多可作为 R-A 的 diagnostic upper bound，不是 headline method。

---

## J. Closest-work/reconstruction matrix

缩写：`F`=预提取/冻结特征，`RGB`=raw video，`I`=immutable，`M`=mutable internal hypothesis。

| 工作族                       | Task                   | Future access | Input                       | Trainable backbone | State identity                   | Assignment                | 输出            | Cleanup                     | Latency metric        | 训练成本 | 与候选重叠及裁决                                                                                                                             |
| ------------------------- | ---------------------- | ------------- | --------------------------- | ------------------ | -------------------------------- | ------------------------- | ------------- | --------------------------- | --------------------- | ---- | ------------------------------------------------------------------------------------------------------------------------------------ |
| SimOn / CAG-QIL           | On-TAL                 | 无             | F                           | 通常否                | frame/context，无强实例 carrier       | frame/grouping            | I             | online grouping             | endpoint/online AP    | 低–中  | `[LITERATURE-VERIFIED]` R-A 仅在 causal context 上重叠；shared latent identity 未占据。([arXiv][3])                                            |
| OAT / HAT                 | On-TAL                 | 无             | F                           | 通常否                | window anchors                   | local instance assignment | I             | online proposal suppression | online AP             | 中    | `[INFERENCE]` R-B 高重建风险；R-A 的跨 prefix model-only state仍不同。                                                                           |
| MATR                      | On-TAL                 | 无             | pretrained segment features | temporal model是    | memory flag/queue；非持续事件身份        | per-window Hungarian      | I             | online NMS                  | On-TAL AP             | 中    | `[LITERATURE-VERIFIED]` R-B 接近 occupied；R-A 与其 GT-train/pred-infer memory flag形成实质差异。([arXiv][2])                                    |
| ActionSwitch              | On-TAL                 | 无             | F                           | 否/有限               | class-agnostic switch states     | state supervision         | I             | state decode                | On-TAL AP             | 中    | `[LITERATURE-VERIFIED]` 已占 overlap claim；R-A 不能把“支持重叠”单独当创新。([arXiv][5])                                                             |
| E2E-LOAD                  | OAD                    | 无             | RGB                         | 是                  | stream cache，无 interval identity | frame labels              | 每帧分类          | 无 interval cleanup          | frame mAP/FPS         | 高    | `[LITERATURE-VERIFIED]` 只占 raw causal training/backbone，不占 immutable interval lifecycle。([arXiv][6])                                 |
| StreamFormer              | OAD/VIS/VQA backbone   | 无             | RGB                         | 是                  | generic streaming memory         | task-specific             | task-specific | task-specific               | efficiency            | 高    | `[LITERATURE-VERIFIED]` causal backbone 是组件竞争，不能成为 R-A headline。([arXiv][12])                                                        |
| TadTR / TE-TAD            | offline TAD            | 有完整视频         | F/RGB                       | 视实现                | full-video queries               | Hungarian                 | 最终可统一修改       | offline set decode          | offline mAP           | 中–高  | `[LITERATURE-VERIFIED]` R-B 的 set prediction 已被占据；缺少 online immutable state合同。([arXiv][15]) ([arXiv][16])                            |
| Re²TAL / AdaTAD           | offline TAD            | 有完整视频         | RGB                         | 是/adapter          | 无 online identity                | detector-specific         | offline       | offline                     | offline mAP           | 高    | `[LITERATURE-VERIFIED]` 只占高效 visual adaptation。([arXiv][7]) ([arXiv][8])                                                             |
| TrackFormer               | MOT/VIS                | 当前及过去         | RGB/features                | 是                  | persistent track queries         | detection + track queries | M trajectory  | online tracker              | HOTA/MOTA             | 高    | `[LITERATURE-VERIFIED]` R-A 最大 reconstruction risk：可被描述为 1D temporal TrackFormer。([arXiv][13])                                       |
| MOTR                      | MOT                    | 当前及过去         | RGB                         | 是                  | track query propagation          | tracklet-aware + newborn  | M trajectory  | online                      | HOTA/MOTA             | 高    | `[LITERATURE-VERIFIED]` 与 R-A 的 carriers/reseed高度重合；R-A 必须靠 loss-only assignment、first-event hazard和immutable ledger区分。([arXiv][14]) |
| Neural/Transformer Hawkes | event sequence         | 过去事件          | event marks                 | N/A                | hidden event history             | likelihood，无 Hungarian    | event stream  | 无                           | event-time likelihood | 中    | `[LITERATURE-VERIFIED]` R-C 的 event-time objective 高度重合，但没有视频 interval/start localization。([arXiv][17]) ([arXiv][18])                |
| RNN-T                     | streaming transduction | 过去输入/输出       | sequence                    | 是                  | decoder state                    | alignment marginalization | token stream  | beam/search                 | token latency         | 高    | `[LITERATURE-VERIFIED]` R-C 的核心数学已被占用；只有 interval/set adaptation可能是增量。([arXiv][19])                                                  |

**矩阵裁决**

* R-A：`novel-for-P0 / reconstruction-risk`
* R-B：`occupied`
* R-C：`unknown / reconstruction-risk`
* R-D：`reject`

---

## K. Claim map for each retained route

### K1. R-A

| Claim                                                             | 证据需求                                                | Falsifier                              |
| ----------------------------------------------------------------- | --------------------------------------------------- | -------------------------------------- |
| C-A1：train/inference 使用完全相同的 model-only state transition          | P0 state hash、step/batch equality、loss-off equality | 任一 GT-dependent runtime field或分叉逻辑     |
| C-A2：GT assignment 只影响 loss，不影响 state、availability、reset、emission | GT-taint、autograd/dataflow audit                    | 修改标签导致同 prefix runtime state变化         |
| C-A3：release/reseed 不产生 Q2 式 exhaustion                           | repeated/overlap/same-bin synthetic                 | 出现“因模型状态不可用而丢弃 GT target”              |
| C-A4：不能靠静默策略通过                                                    | frozen anti-silence gate                            | no-emission/always-bg达到 readiness PASS |
| C-A5：相对 temporal-MOTR reconstruction 改善 On-TAD identity error     | P3 matched-seed experiment                          | identity/latency差异落入预设等价区间             |
| C-A6：效率不是主 claim                                                  | matched profile                                     | 只有成本下降而无主指标/错误模式改善                     |

### K2. R-C

| Claim                                                          | 证据需求                        | Falsifier                      |
| -------------------------------------------------------------- | --------------------------- | ------------------------------ |
| C-C1：alignment marginalization 可覆盖 uncertain completion timing | synthetic exact lattice     | 需要 GT state或固定唯一对齐             |
| C-C2：same-bin多事件不依赖任意 token order                              | permutation-invariance test | 改变 canonical order改变概率/输出      |
| C-C3：相对 RNN-T/THP 有 On-TAD 特有实质机制                              | reconstruction baseline     | 标准 RNN-T + interval mark复现全部收益 |
| C-C4：成本可在单卡预算内                                                 | P2 profile                  | profile 超出既定 cap               |

### K3. R-B

`[INFERENCE]` 只作为 baseline，允许主张“简单、同构、无 persistent lifecycle”，不允许主张新的论文核心机制。

---

## L. Adversarial novelty review

### L1. 对 R-A 的最强拒稿论证

> 这是把 MOTR/TrackFormer 的 persistent queries、新生 query 和 assignment 搬到 1D 时间轴，再加一个 endpoint head与 ledger；soft occupancy和OT都是常见连续松弛，因此没有方法创新。

`[INFERENCE]` 这条拒稿论证是可信的。R-A 只有在以下三点同时产生可测收益时才可能存活：

1. assignment **严格 loss-only**，不参与任何 runtime transition；
2. completion/release/reseed 是同一个 prefix filter 的一部分，而不是外部 controller；
3. immutable ledger 下的 duplicate、fragmentation、same-class repetition 显著优于参数匹配的 temporal-MOTR reconstruction。

任一不成立，R-A应降为工程重构或 baseline。

### L2. 对 R-B 的最强拒稿论证

> MATR/OAT/PCEH 已经在做“当前判断 end、历史定位 start、集合输出”。

`[INFERENCE]` 成立。R-B 不具备 headline novelty。

### L3. 对 R-C 的最强拒稿论证

> 这是把 RNN-T 或 neural point process 的输出 token/mark 改成 interval。

`[INFERENCE]` 也成立。除非 same-bin unordered event lattice 和 interval start posterior形成不可由普通 transducer 重建的机制，否则 novelty 不足。

### L4. 条件 novelty verdict

`[INFERENCE]`

* **P0 级 novelty hypothesis：PASS**
* **论文级 novelty：NOT YET PASSED**
* **最主要重建风险：MOTR/TrackFormer**
* **禁止的 headline：causal backbone、memory、persistent query、LoRA、cache、soft OT、new loss中的任意单项**

---

## M. Scientific-scale and experiment-scale ranking

| 排名 | 路线                                    | 科学尺度                                                | 实验尺度                                   | 主要风险                                   | 裁决            |
| -: | ------------------------------------- | --------------------------------------------------- | -------------------------------------- | -------------------------------------- | ------------- |
|  1 | R-A Prefix-Shared Latent Event Filter | `[INFERENCE]` 中高：直接针对 On-TAD train/runtime contract | 中：P0 可纯 CPU，后续需 matched reconstruction | MOTR 重建风险                              | 只进 P0         |
|  2 | R-C Causal Interval Transducer        | `[INFERENCE]` 中高                                    | 高：lattice、same-bin set、decoder成本大      | RNN-T/TPP 重建风险                         | 暂不实现          |
|  3 | R-B Endpoint-Set Completion           | `[INFERENCE]` 低–中                                   | 低                                      | 已被 MATR/PCEH/OAT 占据                    | baseline-only |
|  4 | R-D Teacher–Student Alignment         | `[INFERENCE]` 低                                     | 中                                      | GT teacher taint、distillation occupied | reject        |

`[INFERENCE]` R-A 是唯一满足“问题尺度大于单一 loss、P0 又足够小”的候选。

---

## N. Selected route or project-stop decision

**选择：R-A，且只授权 P0。**

`[INFERENCE]` 不停止整个 On-TAD 项目，是因为现有负证据只杀死：

* empty-state selective replay；
* hard runtime/canonical coupled persistent slots；
* 通过抑制 birth 达到 readiness 的门禁。

它没有杀死“完整 chronological forward + model-only latent filter + loss-only assignment”。

`[INFERENCE]` 不授权 P1/P2/P3，是因为：

* 方法尚未实现；
* publication novelty仍有显著 MOTR reconstruction risk；
* cached visual causality未完全闭合；
* synthetic anti-silence尚未通过；
* 真实效果、成本和第二数据集均未知。

---

## O. Formal method definition for the selected route

### O1. 输入与输出

`[INFERENCE]`

在 decision bin (t)，模型只接收 causal token

[
x_t=E(v_{\leq r_t}),\qquad r_t\le t.
]

输出仍为标准不可变实例：

[
y_i=(s_i,e_i,c_i,p_i),\qquad e_i\le t.
]

内部 latent carrier 不作为 benchmark 输出，也不引入 event ID 任务。

### O2. Runtime state

[
S_t=(M_t,Z_t,o_t,A_t,C_t,\ell_t,\tau_t)
]

其中：

* (M_t)：有限 causal feature memory；
* (Z_t\in\mathbb{R}^{K\times d})：潜在事件载体；
* (o_t\in[0,1]^K)：连续 occupancy；
* (A_t)：每个载体的 start posterior，支持 retained frames 与 `before_memory` sentinel；
* (C_t)：累计 class evidence；
* (\ell_t)：model-only ledger latch，只控制首次提交；
* (\tau_t)：最后 source/decision frame。

**禁止字段：**

* GT instance ID；
* canonical slot ID；
* assigned target；
* future endpoint；
* teacher state；
* runtime-free target mask；
* calibration label。

### O3. 共用 transition

`[INFERENCE]`

首先执行 causal carrier update：

[
\widetilde Z_t=F_\theta(Z_{t-1},M_{t-1},x_t).
]

对上一时刻开放事件预测 completion hazard：

[
h_t=\sigma(g_h(\widetilde Z_t)),\qquad
\rho_t=o_{t-1}\odot h_t.
]

先 release：

[
o_t^{R}=o_{t-1}\odot(1-h_t),
]

[
Z_t^{R}=(1-\rho_t)\odot\widetilde Z_t+
\rho_t\odot z_{\emptyset}.
]

再从 released/free mass 中产生新 birth：

[
\beta_t=(1-o_t^{R})\odot
\sigma(g_b(Z_t^{R},x_t)),
]

[
o_t=o_t^{R}+\beta_t,
\qquad
Z_t=G_\theta(Z_t^{R},x_t,\beta_t).
]

此顺序允许“旧事件本 bin完成、同一 carrier 本 bin重新开始新事件”。

对于 start 与 end 都跨越同一 decision interval 的短事件，单独定义：

[
d_t=\sigma(g_{\mathrm{direct}}(Z_t^{R},x_t)),
]

仅输出 start/end 都位于当前 interval 的 direct-complete event。该分支必须由 annotation census 决定是否进入正式模型；P0 必须测试其合同。

### O4. Loss-side assignment

训练时构造截至 (t) 可观察的实例集合：

[
Y_t={y_j: s_j\le t,\ e_j\text{ 若未发生则 censored}}.
]

计算带 dustbin 的 unbalanced entropic OT：

[
P_t=
\operatorname{UOT}*{\epsilon}
\left(C*\theta(S_t,Y_t)\right).
]

`[INFERENCE]` (P_t) 只用于加权 loss；严禁：

* 写入 (S_t)；
* 选择 runtime availability；
* reset carrier；
* 决定 ledger emission；
* 在下一个 bin 作为模型输入。

### O5. Loss

[
L=
\lambda_bL_{\text{first-birth}}
+\lambda_oL_{\text{occupancy}}
+\lambda_eL_{\text{first-end-hazard}}
+\lambda_cL_{\text{class}}
+\lambda_sL_{\text{start}}
+\lambda_dL_{\text{direct}}
+\lambda_\tau L_{\text{transport-consistency}}.
]

* birth：仅 first-observable crossing 为正；
* occupancy：start 已观察且 endpoint 尚未观察；
* end：离散 survival/first-event NLL，endpoint 后完全 mask；
* class/start：由 OT positive mass 加权；
* consistency：只约束相邻 prefix 对同一可见 GT 的 loss assignment，不形成 runtime identity；
* dustbin mass 用于显式记录未解释 GT/载体，而不是静默丢弃。

### O6. Ledger

`[INFERENCE]`

* (\ell_t) 完全由模型输出和过去 ledger 决定；
* completion 超过冻结阈值时，使用 **release 前** 的 start/class state写入 event；
* event 一经写入不可回改；
* 同 bin reseed 使用 release 后的 state；
* latch 不参与 OT、loss masks或 neural carrier transition；
* direct-complete event 可在无 prior open latch 时提交，但必须满足专用阈值和当前-bin start/end约束。

---

## P. Training and inference contract

| 项目                | 冻结合同                                                                                                          |
| ----------------- | ------------------------------------------------------------------------------------------------------------- |
| 输入                | `[INFERENCE]` 每个 token必须有严格递增 source frame，且 source≤decision；P1 前须补 visual receptive-field certificate        |
| forward           | `[INFERENCE]` train与inference必须调用同一个 `advance_and_decode`；禁止 `_train_transition` / `_infer_transition` 两份科学逻辑 |
| GT使用              | `[INFERENCE]` 仅在 `build_prefix_targets` 和 `compute_loss_transport` 中出现                                        |
| state             | `[INFERENCE]` 只能包含输入、模型输出和过去 immutable ledger                                                                 |
| assignment        | `[INFERENCE]` loss-local；函数返回后不得持久化                                                                           |
| risk set          | `[INFERENCE]` 由 prefix-visible GT状态和全部 carriers定义，不依赖 predicted `FREE`                                        |
| output            | `[INFERENCE]` model-only；禁止 GT threshold、teacher cache和 offline NMS                                           |
| loss denominator  | `[INFERENCE]` 每类 loss 按实际 positive/risk mass归一；不以每 bin简单平均掩盖稀疏事件                                              |
| empty denominator | `[INFERENCE]` 记录 graph-connected zero 与 denominator=0；不得伪造一个样本                                                |
| optimizer event   | `[INFERENCE]` 每完整 chronological video一个事务性 optimizer event                                                    |
| TBPTT             | `[INFERENCE]` forward state完整保留；只允许在预登记 packet boundary detach；明确称 truncated objective，不得称完整视频 full-BPTT      |
| AMP               | `[INFERENCE]` P0 CPU FP32；P2 才允许 bf16/AMP profile                                                             |
| end-to-end称谓      | `[INFERENCE]` P0/P1/P3第一阶段均称 cached-feature/head-level；raw-video joint另立合同                                    |
| rollback          | `[INFERENCE]` 非有限 loss/gradient、状态不变式或 anti-taint失败时，必须在 optimizer mutation前 rollback                         |
| evidence          | `[INFERENCE]` commit、tree、config、data/cache、encoder、seed、runtime、ledger和测试字节全部绑定                              |

---

## Q. Anti-silence and anti-taint gates

### Q1. P0-A deterministic contract suite

`[INFERENCE]` 固定八类 synthetic sequence：

1. 单一事件；
2. delayed endpoint；
3. 同类重复、间隔小于旧 Q2 refractory；
4. 异类 overlap；
5. 同类 overlap；
6. 旧事件 end 与新事件 birth 同 bin；
7. start/end 同 decision interval；
8. 全背景。

使用 scripted/oracle logits时必须：

* event count 精确相等；
* recall=1；
* precision=1；
* duplicate=0；
* fragmentation=0；
* future access=0；
* all-background emission=0；
* immutable ledger byte sequence在stepwise/batched执行下相同。

任一失败：**P0 KILL**。

### Q2. Train/inference equality

`[INFERENCE]`

同一参数、同一输入、同一初始 state：

* training mode但关闭 loss；
* inference mode；
* stepwise；
* packet-batched；

四者每个 prefix 的 scientific state hash、logits和 ledger必须完全一致；允许的唯一差异是 dropout 被显式冻结后的非科学 bookkeeping。

### Q3. GT-taint

`[INFERENCE]`

对相同视频 prefix构造两个 annotation futures：

* 当前 prefix前完全相同；
* 未来 class/end/实例数不同。

截至当前 prefix：

* runtime state hash必须相同；
* emission bytes必须相同；
* source frame trace必须相同；
* 只有 loss-side target/transport可以不同。

### Q4. Future perturbation

`[INFERENCE]` 修改所有 (>t) 的视频帧、feature tokens和标签；截至 (t) 的 state、logits和 ledger不得变化。

### Q5. Learned anti-silence gate

`[INFERENCE]` P0 使用固定 seeds `9101/9102/9103`，CPU FP32，在程序生成的 512 条训练 synthetic 和 256 条 outcome-blind held-out synthetic上运行。三 seed **全部**满足：

* macro event recall ≥ 0.90；
* precision ≥ 0.90；
* same-class overlap recall ≥ 0.80；
* mean absolute event-count error ≤ 0.10/positive sequence；
* 64 条全背景序列合计 false emission=0；
* duplicate fraction=0；
* fragmentation rate=0；
* no-birth/no-emission/always-background controls 的 recall 必须为 0，且 readiness verdict 必须为 FAIL。

不得用平均 seed 掩盖单 seed 静默。

### Q6. Gradient/optimizer gate

`[INFERENCE]`

对 birth、occupancy、end、class、start、release/reseed、direct-complete各模块要求：

* finite gradient；
* 至少一个适用 synthetic family 中 gradient norm>0；
* positive/negative perturbation方向正确；
* 所有 trainable parameters恰好进入 optimizer 一次；
* frozen parameters不得进入 optimizer；
* endpoint/ledger record必须 detach，不得被后续 loss回改。

### Q7. 单一终止规则

任何 Q1–Q6 失败：

```text
P0_STATUS=KILL
P1_ALLOWED=false
GPU_PROFILE_ALLOWED=false
FORMAL_TRAINING_ALLOWED=false
GPU_HOURS_AUTHORIZED=0
```

不得在看到失败后修改阈值、synthetic distribution或 readiness margin。

---

## R. Baseline/ablation/metric plan

### R1. Reference、baseline与 upper bound

| 类别                     | 对象                                                                                             |
| ---------------------- | ---------------------------------------------------------------------------------------------- |
| Gold reference         | `[INFERENCE]` prefix schedule oracle；只验证合同，不算模型 baseline                                       |
| 主候选                    | R-A                                                                                            |
| 最强简单 baseline          | R-B endpoint-set completion，完全相同 encoder、hidden dim、query count、参数预算                           |
| 最近工作重建                 | `[INFERENCE]` temporal-MOTR：hard track/newborn query assignment + interval heads + same ledger |
| 公开 On-TAL baseline     | MATR、ActionSwitch、HAT、SimOn；只有在相同 causal features/evaluator下才进入主表                              |
| 负对照                    | 当前 Q2 只读 replay；禁止改阈值复活                                                                        |
| privileged upper bound | GT-visible carrier binding；只估计表示上界，不参与选择或推理                                                    |

### R2. 必要 ablation

1. loss-only OT → hard state-mutating assignment；
2. soft release/reseed → hard FREE/ACTIVE；
3. 去掉 transport consistency；
4. 去掉 continuous occupancy；
5. 去掉 start posterior，只用 scalar offset；
6. 去掉 direct-complete 分支；
7. readout latch参与/不参与 latent transition；
8. TBPTT horizon；
9. frozen features vs 后续允许时的 visual adaptation。

### R3. 指标

`[CODE-VERIFIED]` 可复用当前 evaluator 的：

* average mOnlineAP；
* 各 latency budget、tIoU；
* standard online mAP；
* TP/FP/FN；
* duplicate；
* fragmentation；
* false emission；
* endpoint latency；
* no-future ledger validation。

`[INFERENCE]` 另按预冻结子集报告：

* same-class repetition；
* same-class/cross-class overlap；
* same-bin events；
* long actions；
* start-before-memory；
* per-video throughput与peak memory。

### R4. P3 family-wise rule

`[INFERENCE]` 主候选相对最强非 privileged baseline，3 matched seeds、video-clustered bootstrap：

1. average mOnlineAP 差值的 95% lower bound ≥ −1.0 pp；
2. recall 差值的 95% lower bound ≥ −1.0 pp；
3. duplicate-per-GT 或 fragmentation 至少一项相对下降 ≥20%，95% CI 不跨 0；
4. 另一 identity 指标的相对恶化不得超过 5%；
5. same-class overlap recall 不下降；
6. temporal-MOTR reconstruction 若与候选在主指标 ±0.5 pp、identity error ±5% 内等价，则主方法 novelty KILL。

---

## S. Costed P0–P3 experiment ladder

当前仅 P0 获授权。后续成本只是上限设计，不是授权。

| 阶段 | 内容                                                                                                               |                                                      CPU/GPU上限 |      存储 | 停止条件                                                    |
| -- | ---------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------: | ------: | ------------------------------------------------------- |
| P0 | synthetic合同、anti-taint、anti-silence、gradient、optimizer、step/batch equality                                       |                        `[INFERENCE]` ≤2 CPU执行小时；0 GPU；工程约2–4人日 | ≤0.5 GB | 任一 Q gate失败即 KILL                                       |
| P1 | outcome-blind CPU/cached-feature mechanism G0；完整160视频×3 seed chronological forward/loss audit；不得访问 reporting set | `[INFERENCE]` hard cap 48 aggregate CPU-hours，≤8 threads；0 GPU |   ≤8 GB | state mismatch、silent pass、budget exceeded、证据不闭合任一 KILL |
| P2 | fixed-step单卡 profile；50 warmup + 200 measured optimizer events；记录tokens/s、video/s、backward比例、peak VRAM           |                                   `[INFERENCE]` 总计≤2 GPU-hours |  ≤15 GB | skipped event、OOM、非有限、超时或无法区分cached/raw成本即 KILL         |
| P3 | cached-feature最小效果实验；R-A、R-B、temporal-MOTR，matched seeds与相同 evaluator                                            |                `[INFERENCE]` 总 cap 12 GPU-hours，单卡；不含raw-video |  ≤60 GB | R4任一主规则失败或 reconstruction equivalence成立                 |
| P4 | raw-video/backbone fine-tuning                                                                                   |                                           `[INFERENCE]` 不在本轮合同 |     未定义 | 必须另行审查和授权                                               |

**最大失败成本**

* 已授权最大失败成本：**2 CPU-hours执行 + 0 GPU-hours**；
* 目前不得提交 P1/P2/P3；
* 不得以“已写完代码”为理由跳过 P0 reviewer gate。

---

## T. File-by-file implementation map

### T1. 新增科学代码

| 文件                                                              | 精确职责                                                                                                      | 映射 claim/gate   |        预计规模 |
| --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | --------------- | ----------: |
| `opentad/models/dense_heads/prefix_shared_event_filter_head.py` | `PrefixSharedEventState`、continuous occupancy、completion/release/reseed、start posterior、direct-complete输出 | C-A1/C-A3       | 280–380 LOC |
| `opentad/models/detectors/prefix_shared_event_filter_ontad.py`  | 唯一共用 `advance_and_decode`；chronological episode；immutable ledger                                          | C-A1/C-A2/Q2/Q3 | 320–450 LOC |
| `opentad/utils/prefix_loss_transport.py`                        | prefix target、dustbin UOT、risk mass、denominator和post-end mask                                             | C-A2            | 180–260 LOC |
| `opentad/utils/prefix_state_contract.py`                        | state schema、字段可观测性、GT-taint/future-taint审计                                                               | Q2–Q4           | 100–150 LOC |
| `opentad/utils/prefix_event_evidence.py`                        | state/ledger hash、atomic P0 evidence、identity binding                                                     | evidence gate   | 120–180 LOC |
| `tools/audit_prefix_shared_event_p0.py`                         | 全部 synthetic、learned anti-silence、machine-readable verdict                                                | Q1–Q7           | 300–450 LOC |

### T2. 配置

| 文件                                                                    | 内容                                                                     |
| --------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `configs/causaltad/thumos_prefix_shared_event_p0.py`                  | `[INFERENCE]` CPU-only；不得引用真实 feature cache；固定 synthetic seeds/margins |
| `configs/causaltad/thumos_prefix_shared_event_g0.py`                  | `[INFERENCE]` 默认 `enabled=False`；只有 P0 reviewer PASS commit 才能解锁       |
| `research-wiki/ideas/prefix-shared-latent-event-filter.md`            | task、exact delta、non-claims、closest work                               |
| `research-wiki/experiments/prefix-shared-event-p0-preregistration.md` | frozen cases、seeds、thresholds、hash schema、KILL rule                    |

### T3. 注册修改

* `opentad/models/dense_heads/__init__.py`
* `opentad/models/detectors/__init__.py`
* `opentad/utils/__init__.py`

`[INFERENCE]` 只增加 registry/export；不得修改当前 Q2 类来共享可变逻辑，避免旧/新证据对象混合。

### T4. 测试

* `tests/test_prefix_shared_event_state.py`
* `tests/test_prefix_loss_transport.py`
* `tests/test_prefix_shared_event_same_bin.py`
* `tests/test_prefix_shared_event_overlap_repeat.py`
* `tests/test_prefix_shared_event_future_taint.py`
* `tests/test_prefix_shared_event_train_infer_equality.py`
* `tests/test_prefix_shared_event_anti_silence.py`
* `tests/test_prefix_shared_event_optimizer.py`
* `tests/test_prefix_shared_event_atomic_evidence.py`

### T5. State dataclass可观测性

| 字段                             | 推理时可观测 |        GT允许 | 进入loss risk set |
| ------------------------------ | -----: | ----------: | --------------: |
| `queries/carriers`             |      是 |           否 |            作为预测 |
| `feature_memory/source_frames` |      是 |           否 |            作为预测 |
| `occupancy`                    | 是，模型生成 |           否 |            作为预测 |
| `start_posterior`              | 是，模型生成 |           否 |            作为预测 |
| `class_evidence`               | 是，模型生成 |           否 |            作为预测 |
| `ledger_latch`                 | 是，模型生成 |           否 |               否 |
| `ledger_rows`                  | 是，过去输出 |           否 |               否 |
| `loss_transport`               | 否，临时变量 | 可用prefix GT |         是；函数后销毁 |
| `canonical_instance_to_slot`   | **禁止** |           — |               — |

### T6. 旧代码处置

`[INFERENCE]`

**只读保留：**

* `persistent_event_set_head.py`
* `persistent_trajectory_ontad.py`
* Q2 fixed/rematch configs
* Q2 capacity audit与trace contract
* CRS-EPS及CSFSB文档
* terminal decision records

**明确废弃但不删除：**

* Q2/R1 scientific route；
* empty-state CRS-EPS；
* `-2` birth prior candidate。

不得重命名旧代码后冒充新路线，也不得把旧 Q2 evidence ticket复用到新 commit。

### T7. Atomic evidence

P0 artifact 至少绑定：

* exact commit和tree hash；
* source file hashes；
* resolved/scientific config；
* synthetic generator版本及全部 seeds；
* Python/PyTorch/CPU/runtime identity；
* test collection及完整 stdout/stderr；
* state/ledger trace bytes；
* optimizer coverage；
* verdict JSON；
  -临时目录写完、fsync、hash后 atomic rename；
* 已存在目录禁止覆盖。

**没有对应 claim 或 gate 的代码不得写。**

---

## U. Reviewer-risk register

| 风险                                  | 级别               | 处置                                                                     |
| ----------------------------------- | ---------------- | ---------------------------------------------------------------------- |
| 被重建为 temporal MOTR/TrackFormer      | `[INFERENCE]` 极高 | P3 前必须实现参数匹配 reconstruction；等价即 novelty KILL                           |
| soft occupancy长期处于模糊 0.5            | 高                | P0记录 occupancy entropy、release/birth mass；不能只看 gradient 非零             |
| readout latch重新形成隐藏 hard lifecycle  | 高                | latch不得进入 neural transition、assignment或risk；dataflow test              |
| OT跨 bin频繁 identity swap             | 高                | loss-only consistency diagnostic；若 identity gain不存在则停止                 |
| direct-complete 分支成为 post hoc patch | 中高               | annotation census前只做 synthetic合同；正式启用须预登记真实需求                          |
| no-emission仍可通过低阈值 calibration      | 高                | anti-silence使用 held-out event count和all-background FP，不以 exhaustion作主门 |
| TBPTT导致长动作 credit assignment不足      | 高                | 明确 truncated objective；比较 horizon，不再声称 full historical gradient        |
| cached feature含未来上下文                | 高                | P1前必须补 encoder receptive-field certificate                             |
| THUMOS14对重叠/identity证据不足            | 高                | 第二数据集按 annotation census盲选                                             |
| evaluator identity匹配规则偏向候选          | 中                | 固定 evaluator hash，主表另报标准 mOnlineAP与原始 counts                           |
| 路线收益只有效率                            | 中                | 效率不能替代 identity/latency主效应                                             |
| P0 tiny synthetic过拟合                | 中                | 独立 generator seeds与 unseen family combination；P1另设真实数据 G0              |
| raw-video成本失控                       | 高                | 本轮完全不授权 raw-video                                                      |
| 作者在失败后放宽margin                      | 极高               | prereg hash + terminal KILL；不得 outcome-dependent revision              |

---

## V. Exact author questions still blocking work

`[INFERENCE]` **没有问题阻塞本次 P0 实现**：任务、P0方法边界、anti-silence和终止规则已经冻结。

下列问题阻塞 P1 或更高阶段：

1. `[UNKNOWN]` 请提供 `pes_siglip2_stride8` feature cache 对应 encoder 的精确代码、权重 ID、clip/frame construction、temporal receptive field和预处理 manifest。它阻塞 cached-feature causal claim。
2. `[UNKNOWN]` 是否能向下一位独立 reviewer 提供 Q2/CRS 原始外部 evidence bundle 的只读路径？这不改变 KILL，但决定能否重新做字节级复核。
3. `[UNKNOWN]` FineAction、EPIC-Kitchens-100、MUSES 中哪些数据和许可当前可用？必须先做 outcome-blind annotation census，再冻结第二数据集。
4. `[UNKNOWN]` P2 目标单卡的精确 GPU 型号、VRAM、driver、CUDA和PyTorch版本是什么？缺失时不能冻结 profile cost。
5. `[UNKNOWN]` MATR、ActionSwitch、HAT、SimOn 是否有可合法使用的 checkpoint/feature protocol？缺失时只能做明确标注的 matched reconstruction，不能声称复现原论文。
6. `[UNKNOWN]` 下一轮是否能由新的 reviewer ID、而不是 Q2 reviewer，自主核验 P0？缺少独立 reviewer 时不得进入 P1。

---

## W. Final machine-readable decision JSON

```json
{
  "reviewed_commit": "7a26e60fccec68cd2b547951320e669a81907461",
  "current_q2_r1_disposition": "KILL",
  "selected_decision": "GO_NEW_ROUTE_P0_ONLY",
  "selected_route_id": "R-A_PREFIX_SHARED_LATENT_EVENT_FILTER",
  "task_definition_pass": true,
  "novelty_pass": true,
  "novelty_pass_scope": "P0_HYPOTHESIS_ONLY",
  "publication_novelty_pass": false,
  "runtime_supervision_contract_pass": true,
  "runtime_supervision_contract_status": "SPECIFIED_NOT_IMPLEMENTED",
  "anti_silence_gate_frozen": true,
  "p0_code_authorized": true,
  "p0_scope": [
    "CPU-only synthetic contract implementation",
    "train-inference state equality tests",
    "GT-taint and future-perturbation tests",
    "anti-silence learned synthetic gate",
    "optimizer and gradient coverage audit",
    "atomic P0 evidence publication"
  ],
  "p1_cached_feature_g0_allowed": false,
  "gpu_profile_allowed": false,
  "formal_training_allowed": false,
  "raw_video_joint_training_allowed": false,
  "gpu_hours_authorized": 0,
  "blocking_unknowns": [],
  "post_p0_blocking_unknowns": [
    "external cached-feature encoder receptive-field causality",
    "real-data concurrency and same-bin annotation census",
    "independent byte-level access to external Q2/CRS evidence bundles",
    "matched public On-TAL baseline reproducibility",
    "second-dataset availability and selection",
    "target GPU runtime identity and measured cost",
    "publication novelty versus temporal MOTR/TrackFormer reconstruction"
  ],
  "terminal_p0_rule": "ANY_FROZEN_P0_GATE_FAILURE_KILLS_ROUTE",
  "prohibited_actions": [
    "reopen Q2 or R1",
    "adopt birth_prior_bias_m2",
    "change Q2 threshold, K, refractory, release order, or nondegeneracy floor",
    "run GPU profiling",
    "run cached-feature effectiveness training",
    "run formal training",
    "run raw-video joint training",
    "reuse Q2 or CRS evidence tickets for the new route"
  ]
}
```

---

## X. Author-response draft

> We accept the terminal `KILL_Q2_R1` disposition. The `-2` birth prior is not a repaired lifecycle contract; it removes exhaustion by suppressing almost all runtime births. We will not change Q2 thresholds, K, refractory policy, release order, or post hoc nondegeneracy margins, and we will not implement CSFSB or spend GPU hours on the killed route.
>
> The only authorized successor is a new CPU-only P0 mechanism probe: a prefix-shared latent event filter whose model-generated state transition is identical in training and inference. GT assignment will be a loss-local prefix OT variable and will not determine runtime availability, state identity, reset, risk set, or emission. Hard ledger bookkeeping will be model-only and will not enter the neural transition or loss assignment.
>
> Before implementation, we will freeze the synthetic families, seeds, anti-silence margins, train/inference equality tests, GT/future-taint tests, gradient/optimizer tests, evidence schema, and terminal KILL rule. No-birth, no-emission, and always-background controls must fail the readiness gate.
>
> Passing P0 will not establish effectiveness or publication novelty. A subsequent independent review must decide whether a CPU/cached-feature G0 is authorized. GPU profile, formal training, backbone fine-tuning, and raw-video joint training remain prohibited with zero GPU hours authorized.

---

## Y. Next-round prompt

另一轮在 **P0 已实现并原子发布证据后** 才真正必要。下一轮不得重新讨论 Q2，也不得直接授权 GPU。

```text
# Prefix-Shared Latent Event Filter P0 Independent Review

你是一名独立的 On-TAD、streaming state machine、set prediction、PyTorch
autograd 和实验完整性 reviewer。只读审查新的 exact GitHub commit 和 P0
evidence bundle。

必须先核验：

1. exact repository/branch/commit/tree；
2. P0 preregistration 早于所有结果；
3. source/config/synthetic-generator/seed/runtime/evidence hashes；
4. atomic、non-overwritten evidence；
5. reviewer 与实现者身份独立。

只允许审查：

- shared train/inference `advance_and_decode`；
- runtime state dataclass 中是否含 GT、canonical assignment 或 future information；
- loss-side OT 是否在函数返回后销毁；
- release-before-reseed 与 direct-complete synthetic contract；
- immutable ledger；
- stepwise/batched、train/inference state equality；
- GT-taint、future perturbation；
- no-birth/no-emission/always-background anti-silence；
- finite/nonzero gradients；
- optimizer exact coverage；
- 全部冻结 synthetic seeds、families、margins和family-wise rule。

禁止：

- 修改代码或证据；
- 放宽 threshold/margin；
- 使用真实 reporting set；
- GPU profile；
- cached-feature effectiveness training；
- raw-video training；
- 复活 Q2/R1/CSFSB；
- 把 P0 PASS 解释为论文效果或 novelty PASS。

必须只选择：

1. PASS_AUTHORIZE_UNSEEN_CPU_G0_ONLY
2. REVISE_P0_WITHOUT_USING_EXPOSED_HOLDOUT
3. KILL_NEW_ROUTE

若选择 PASS，也只能授权新 outcome-blind CPU/cached-feature G0，GPU hours
仍为 0。输出必须包含 exact evidence ledger、state/dataflow proof、
anti-silence replay、Unknown Register、machine-readable JSON 和下一阶段
禁止事项。
```

[1]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/tree/codex/full-petal-implementation "https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/tree/codex/full-petal-implementation"
[2]: https://arxiv.org/abs/2408.02957 "https://arxiv.org/abs/2408.02957"
[3]: https://arxiv.org/abs/2211.04905 "https://arxiv.org/abs/2211.04905"
[4]: https://arxiv.org/abs/2408.06437 "https://arxiv.org/abs/2408.06437"
[5]: https://arxiv.org/abs/2407.12987 "https://arxiv.org/abs/2407.12987"
[6]: https://arxiv.org/abs/2306.07703 "https://arxiv.org/abs/2306.07703"
[7]: https://arxiv.org/abs/2211.14053 "https://arxiv.org/abs/2211.14053"
[8]: https://arxiv.org/abs/2311.17241 "https://arxiv.org/abs/2311.17241"
[9]: https://arxiv.org/abs/2605.09976 "https://arxiv.org/abs/2605.09976"
[10]: https://arxiv.org/abs/2607.00289 "https://arxiv.org/abs/2607.00289"
[11]: https://arxiv.org/abs/2509.12145 "https://arxiv.org/abs/2509.12145"
[12]: https://arxiv.org/abs/2504.20041 "https://arxiv.org/abs/2504.20041"
[13]: https://arxiv.org/abs/2101.02702 "https://arxiv.org/abs/2101.02702"
[14]: https://arxiv.org/abs/2105.03247 "https://arxiv.org/abs/2105.03247"
[15]: https://arxiv.org/abs/2106.10271 "https://arxiv.org/abs/2106.10271"
[16]: https://arxiv.org/abs/2404.02405 "https://arxiv.org/abs/2404.02405"
[17]: https://arxiv.org/abs/1612.09328 "https://arxiv.org/abs/1612.09328"
[18]: https://arxiv.org/abs/2002.09291 "https://arxiv.org/abs/2002.09291"
[19]: https://arxiv.org/abs/1211.3711 "https://arxiv.org/abs/1211.3711"
