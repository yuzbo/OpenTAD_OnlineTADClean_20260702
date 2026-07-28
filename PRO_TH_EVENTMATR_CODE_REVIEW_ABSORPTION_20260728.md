# TH-EventMATR Pro 代码审判：完整吸收与独立复核

## 来源与可见性

- 原始附件：
  `C:/Users/skywalker/.codex/attachments/5390141d-bf3e-4952-bfe3-aae4dd9997bc/pasted-text.txt`
- SHA-256：
  `1BC0FDDA22B694F1EEA4485DF9478D6E252FB22AE4E53156F1B2903814023513`
- 大小：`67,552` bytes
- 文本：`1,896` 个可读行（`wc` 为 1,895 个换行）
- 审查目标：
  `yuzbo/OpenTAD_OnlineTADClean_20260702@92cf34aa07bebee2a7a7e3661431d5055804b29b`
- 目标 tree：
  `aef4f64bc020df9d39ead9811fbc01407f1c754a`
- Pro 自报可见性：
  `COMMIT_VERIFIED / TREE_LINKAGE_UNATTESTED / CODE_REVIEW_COMPLETED`

本地 checkout 独立确认 HEAD/tree 与上述值一致。Pro 无法通过其 connector 独立证明
commit→tree OID，但固定 commit 和关键 blob 的可见性足以支撑代码阅读；这不是拒绝理由。

## 总裁决

**PARTIAL ACCEPT / REVISE。**

我们认可它对当前 v1 的主要代码诊断、保留 v1 合同并重建学习核心的路线，以及
`N/R/T/H/TH` 的归因思想；不完全认可它给出的数学细节、冻结超参数、数据划分和
“核心代码已经 9/9 通过”的声明。

正确吸收是：

> 冻结 EventMATR v1 为事实基线；先纠正因果/评测诊断，再以稳定的跨时间
> pre-birth assignment、事件归一化 hazard、可微 ragged unroll 和 birth 后身份锁
> 构造 EventMATR v2。分层视觉记忆继续后置。现有附件没有携带所声称的代码/patch，
> 因此不能把 Pro 的参考实现当成已验证实现直接合并。

## 逐项处置

| Pro 结论 | 处置 | 独立复核 |
|---|---|---|
| H1：稀疏 crossing + dense 背景平均诱导近背景常数解 | 接受 | `CriterionMATR.loss_event` 确实对全部真实 `B×Q` 做 mean BCE/CE；但真实正例率和梯度质量必须由官方 batch 实测 |
| H2：训练 dense prototype、推理 ragged record | 完全接受 | 正式训练默认不运行 runtime；`DynamicEventMemory.step` 是 `@torch.no_grad()`，runtime 输出不进入现有 event loss |
| H3：单帧 START crossing + rising edge 不可恢复 | 完全接受 | GT birth 只有一次；runtime 还要求 `current_start & ~previous_start` |
| H4：当前 owner 不是轨迹级学习 | 接受 | sticky 只是标签写入 query index；fresh 的训练 Hungarian 与运行时 greedy 目标不同；没有跨时刻身份损失 |
| H5：已有部分事件合同，但没有 survival/hierarchy | 接受 | 已有动态 EventRecord、四态、end/emit、ledger；没有 censored hazard、可微轨迹或 learned keep/merge |
| `true_duration` 进入 runtime | 完全接受，列为 P0 | `dataset.py:340-369` 暴露总长度；`models.py:336-340` 传给 runtime；`event_memory.py:604-620` 使用未来终点 |
| start 可能小于 0 | 接受 | birth 只做 `min(frame,candidate_start)`，没有 `max(0,·)` |
| sticky `owner_query_id % Q` 在 `R>Q` 时别名 | 接受 | 多个动态记录可映射到同一 query/mask 位置；“动态存储”不等于无限观测带宽 |
| `Path.touch()` 不清空旧预测 | 接受，列为 P0 | writer 继续 append，重用输出路径可能混入陈旧行 |
| 训练 mAP=0 证明网络全背景 | 拒绝该推断 | runtime-off 时 ledger 为空，writer 不读 dense logits；必须用 checkpoint 的独立 eval replay 判定 |
| 保留 v1 合同，重建学习核心 | 接受 | v1 不删除，也不再作为论文主模型继续堆 loss 权重 |
| TH-EventMATR 是唯一推荐主路线 | 条件接受 | 方向合理，但 birth assignment、cancel/rebirth、censor 编码和复杂度尚未闭合 |
| 层级视觉记忆后置 | 完全接受 | active-event memory 属核心；visual-history keep/merge 不能掩盖生命周期学习失败 |
| v1 不应再直接重跑 100 epoch | 接受为未来规则 | 已完成的 v1 训练不抹除；只是不再重复消耗算力或当作有效论文主结果 |
| “9/9 tests、1453 行核心、可用 patch” | 暂不采信 | 附件目录和两个工作区都没有四个实体文件，无法复算 SHA、编译、pytest 或 full-repo apply |

## 对我们此前判断的关键更正

### 1. 不再把 train mAP=0 直接写成“模型已经学成全背景”

代码链是：

```text
model.train()
→ event_runtime_during_training=False
→ 不创建 EventRecord
→ ledger 为空
→ writer 只读取 ledger
→ proposal 文件为空
→ train mAP=0
```

因此当前有两个独立问题：

1. **协议假零**：空 ledger 会无条件产生零 mAP；
2. **学习风险**：稀疏正例/背景平均仍高度可能产生近背景常数解。

必须使用 epoch-100 checkpoint：

```text
model.eval()
→ reset runtime
→ 从每个视频第一 prefix 完整 replay
→ 新建且显式清空的 proposal 路径
→ 固定决策 policy
→ ledger/writer 行数核对
```

在此之前，不能把零 train mAP 作为正式性能结论，也不能因为它直接抛弃 EventMATR。

### 2. 严格因果失败来自未来终点元数据，不是未来视觉特征

当前模型没有被证明读取未来视觉 feature，但它提前知道完整视频长度，并用其裁剪 end、
验证 EOS 和 emit。这违反本项目“只知道当前与过去；EOS 只能在真实到达时成为当前信号”
的合同。v2 forward/runtime 必须只接收：

```text
video_name
current_frame
st / ed
segment_flag
is_real_prefix
is_eos
```

`duration/true_duration/video_time/frame_to_time` 只能在模型外的评测/坐标转换层使用。

## 对 TH 数学路线的认可与修订

### 接受的核心

1. 零初始化 event adapter 保持初始前向 parity；
2. 每真实事件归一化 birth/end risk，而不是被 `B×Q` 背景线性稀释；
3. interval-censored first-birth；
4. right-censored end survival；
5. birth 前 permutation-aware、birth 后 identity lock；
6. 训练期 chronological ragged unroll；
7. locked calibration policy 与 immutable emit；
8. active-event memory / visual-history memory 分离。

### 必须先修正的数学问题

#### M1：per-prefix Hungarian 不能直接拼成事件 hazard

同一 GT 在窗口内若每个 prefix 匹配到不同 query，则
`h_{i,a},…,h_{i,u}` 并不来自同一候选轨迹，不能直接解释为一个 first-event hazard。

实现前必须在以下方案中冻结一个：

- temporal set assignment / Viterbi，训练期先得到稳定候选路径；
- 带连续性代价的窗口级 Hungarian；
- soft/Sinkhorn assignment 后对 query-set hazard 做可微聚合。

硬匹配可以作为 stop-gradient target assignment，但不能声称 matcher 本身可微。

#### M2：birth latency 期望缺少完整定义

应使用首次出生概率

`h_t ∏_{v<t}(1-h_v)`

作为权重，并除以窗口内至少出生一次的概率。不能只写一个未展开的条件期望。

#### M3：end observed 编码必须每事件最多一次

完结事件只能有一个 observed end；其前方是 survival risk，截断事件全部是 censor。
参考代码若允许多个 `event_observed=True`，会错误累计多个结束事件。

#### M4：teacher forcing 与 cancel 正例

纯 teacher-forced 轨迹不会产生足够 predicted false tracks，cancel head 可能无正例。
训练必须混合 oracle/predicted state，并显式报告：

- predicted-track coverage；
- false-track cancel positives；
- premature cancel；
- reacquisition/rebirth；
- teacher/predicted transition 比例。

#### M5：identity loss 需处理边界

- 无跨时刻正样本的 anchor 必须 mask；
- `(video_id,event_id)` 才是全局训练身份；
- 同类不同实例必须是负样本；
- `O(A²)` 复杂度需要 queue/subsample；
- identity lock 不能掩盖错误初生，必须有 confidence/reacquisition。

#### M6：adapter parity 只在初始化前向成立

`up=0` 时首步只有 up projection 获得梯度；之后 down 才开始学习。若 event loss 允许更新
共享 MATR，训练后的母体不再 parity。正式实验必须明确：

- event-gradient-protected/frozen-parent control；
- 允许更新最后 MATR 层的 candidate；
- optimizer parameter groups 和有效学习率；
- parameter- and compute-matched null adapter。

## 不接受直接冻结的数值

以下只能是 pilot 候选，不是当前科学定值：

- `G_late=8`
- adapter rank `32`
- hard negative top-k `3`
- `λ_bg=0.25`
- `λ_id=0.10`
- `λ_cancel=0.25`
- `λ_latency=0.01`
- 40 epoch teacher-forcing schedule
- epoch 5/10/20 的若干固定百分比门
- prediction/GT `[0.05,20]`
- 0.5/1.0 mAP 的论文门槛

这些值必须来自小规模 profile、数据统计、受控消融或预注册依据，不能因 Pro 写出就冻结。

## 对实验计划的修订

### 保留

- exact native MATR `N`；
- `R/T/H/TH` 因素分解；
- 官方输入和优化设置；
- v1 B×O 独立历史表；
- locked test 一次开启；
- 三 seed 与 paired per-video bootstrap；
- mAP、延时、identity、calibration 和资源共同报告。

### 修订

1. `80/10/10` 与“validation 上 5-fold OOF 校准”不能同时含糊使用。
   必须在实验前冻结单一数据角色；calibration 不得兼作 lane/epoch 选择。
2. 同为 100 epoch 不自动等于公平。还要统一 optimizer steps、batch sampler、参数组，
   并报告 GPU·h、吞吐、显存和 compute-matched control。
3. epoch 5/10/20 多数阈值只作诊断，不作论文优胜门。
4. 硬停止只用于未来泄漏、native parity、重复/可变 emit、静默驱逐、NaN 等正确性失败。
5. 软淘汰需要连续 checkpoint 证据，不能用一次早期波动杀死路线。
6. 单 seed 筛选必须记录所有失败，不补跑有利 seed。

### 更快且稳健的执行顺序

#### Stage D0：先诊断现有 v1，不训练

- checkpoint eval-mode full replay；
- fresh/truncated writer；
- ledger/writer 对齐；
- official batch 正例率与梯度质量；
- `true_duration` perturbation；
- start<0、R>Q、stale-file 检查。

#### Stage D1：TH 集成与硬合同

- full-repo apply/compile/tests；
- native parity；
- no-duration model boundary；
- stable temporal birth assignment；
- differentiable unroll；
- same-class overlap、EOS、Q+1 birth、R>Q tests。

#### Stage D2：五路并行 seed52、5 epochs

- `N/R/T/H/TH` 同 batch order、steps 和 official parent settings；
- 只判断数值、梯度、runtime、因果和 writer 健康。

#### Stage D3：10/20 epoch 机制筛选

- 独立 eval replay；
- frozen calibration；
- 连续两个 checkpoint 的机制与资源趋势；
- 不通过降低阈值或强制 proposal 修复。

#### Stage D4：正式确认

- 仅健康且预注册的 lane 进入三 seed、100 epoch；
- `N/R` 保持锚点；
- policy 冻结后 locked test 一次；
- 报告 paired CI 和完整 GPU 成本。

## 创新性处置

Pro 的竞品表大体准确，但还需补：

- OAT；
- CAG-QIL；
- SimOn；
- OadTR/TEO 等直接 On-TAL/OAD；
- temporal point process / survival 用于视频事件检测的工作；
- MeMOTR/MOTR 后续身份轨迹模型。

不能声称状态转换、persistent query、memory discrepancy、hierarchy 或 survival 本身首次。
当前可防守但仍待系统检索的主张应收窄为：

> 将稳定的 permutation-aware birth set assignment、interval-censored first-birth、
> right-censored instance end、birth 后身份锁和训练—推理一致 ragged causal unroll
> 联合用于标准 closed-set、全监督、严格因果 On-TAL。

ActionSwitch 允许 start 决策灵活延迟，并要求动作完成后输出；本项目若额外要求
“开始证据出现即建立内部事件”，必须写成内部状态/低延时研究目标，而不能悄悄改写
标准最终输出任务。

## 关于 Pro 声称的核心代码

Pro 文本引用：

- `th_eventmatr_core.py`
- `TH_EVENTMATR_UNIFIED_DIFF.patch`
- `test_th_eventmatr_core.py`
- `audit_eventmatr_batch.py`

但当前附件目录只有 `pasted-text.txt`，两个工作区也没有这些文件。因此：

- 文本中的 SHA-256 当前不可复算；
- `1453` 行与约 `2307` 行 diff 当前不可核验；
- `9 passed in 1.12s` 当前不可复跑；
- full-repo integration hunk 本来就未由 Pro 验证；
- 不能把该 patch 直接标记为“已实现”或“可合并”。

这些代码只能在用户重新提供实体附件后复核，或者由本项目根据已接受的数学设计重新实现。

## 当前最终态度

### 完全认可

- v1 不是删除对象，而是冻结归因基线；
- H1–H4 的代码根因；
- `true_duration`、writer 和空 ledger 的 P0 诊断；
- “保留系统合同、重建学习核心”；
- TH 的总体方向；
- `N/R/T/H/TH` 归因；
- 分层视觉记忆后置；
- feature→raw RGB 证据边界。

### 条件认可

- interval birth hazard；
- right-censored end；
- identity InfoNCE；
- teacher forcing；
- cancel/rebirth；
- calibration；
- 三阶段 raw-RGB。

### 不认可

- 把 train mAP=0 当作全背景学习的既成事实；
- 把缺失的 sandbox 文件当作已验证核心代码；
- 直接冻结任意超参数和效果门；
- 在稳定跨时间 assignment 尚未解决时直接开正式 100 epoch；
- 用 Pro 的 `IMPLEMENTATION_PERMISSION` 替代本项目自己的科学裁决。

## 下一步

1. 保全并冻结 v1 commit/checkpoint；
2. 先执行 v1 eval replay 与 batch audit，分离协议假零和模型真塌缩；
3. 在新分支实现 no-duration boundary、writer truncate 和基础不变量；
4. 冻结 temporal pre-birth assignment 设计，再实现 hazard/unroll/identity；
5. Stage D0/D1 通过后，才提交 `N/R/T/H/TH` 的 5-epoch 并行 pilot；
6. 不启动 learned hierarchy、raw-RGB 或新的正式 100-epoch，直到 TH 核心门完成。
