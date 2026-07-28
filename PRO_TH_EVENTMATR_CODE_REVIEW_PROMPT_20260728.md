# TH-EventMATR 下一代模型代码审判 Prompt

请你担任一名同时熟悉 Online Temporal Action Localization、DETR/MATR 类集合预测、
在线多实例关联、离散时间生存分析和严格因果视频模型的资深研究员与 PyTorch 代码审查者。
这不是泛泛的头脑风暴，也不是工程框架评审。你的任务是以精确 GitHub 代码为事实基础，
判断当前 EventMATR 为什么可能退化为全背景/零发射，并给出下一代模型的可实现算法、
核心代码和完整实验方案。

## 1. 强制代码版本与可见性

只审查下面这个公开 GitHub 分支和精确提交：

- 仓库：`https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702`
- 分支：`codex/matr-event-memory`
- 精确提交：`92cf34aa07bebee2a7a7e3661431d5055804b29b`
- 精确 tree：`aef4f64bc020df9d39ead9811fbc01407f1c754a`
- 分支链接：
  `https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/tree/codex/matr-event-memory`
- 提交链接：
  `https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/commit/92cf34aa07bebee2a7a7e3661431d5055804b29b`

开始审查前必须输出 `VISIBILITY_CERTIFICATE`，至少包含：

1. 实际读取的仓库、分支、HEAD SHA；
2. 是否与上述 SHA 完全一致；
3. 实际检查过的文件路径和核心符号；
4. 如果不能读取精确提交，立即停止，不得根据本 Prompt 猜测代码。

官方 MATR 只读母体：

- `https://github.com/skhcjh231/MATR_codebase`
- SHA：`ba05a98d451b3541c1a5377026f17dc1102fa217`

ActionSwitch 只作状态转移参考，不是母体：

- SHA：`838a6ccbd8f2cce414688ff2843380d712aa7b89`

HAT 与 Hierarchical Event Memory 只作后续历史记忆参考；HEM 属于 OnVTG，
不能被直接写成 closed-set On-TAD 基线。

## 2. 任务与不可改变的研究边界

目标是标准、closed-set、全监督、严格因果的 Online Temporal Action Localization：

- 时刻 `t` 只能使用来源时间不晚于 `t` 的视频和内部状态；
- 在动作开始证据出现后立即建立并维护动作实例；
- 动作结束后低延时一次性输出不可修改的
  `{start, end, class, score}`；
- 必须满足 `start < end <= source_time <= emit_time`；
- 推理不得读取 GT identity、未来帧、完整视频终点或事后修正输出；
- 当前是官方 MATR RGB+flow 特征级机制实验；
- 最终目标仍是从原始 RGB 输入端到端训练，而不是停留在特征输入；
- 官方 MATR native lane 必须保持原样，所有改进位于独立派生模型；
- 不允许用固定数量的长期语义槽位限制动作实例；
- 不允许把统一 sigmoid `0.5` 当作所有 birth/end 的科学规则；
- 不讨论网络安全、恶意宿主、TCB、复现篡改或通用工程框架；
- 第一优先级是模型能否有效学习、性能是否提升、创新是否成立。

## 3. 当前 EventMATR 不是被删除，而是作为 v1 与消融保留

当前代码已经实现：

- 官方 MATR 主干与 native lane；
- `BACKGROUND/START/ALIVE/END` 四态竞争；
- B0/B1：延迟 birth 与首次因果 start crossing 的即时 birth；
- O0/O1：fresh rematch 与 sticky owner；
- 数量可变的 `EventRecord`；
- owner-conditioned end；
- end 与 emit 分离；
- 严格正长度、一次关闭、输出不可修改等合同。

请判断下一代模型应当被视为：

1. EventMATR v1 的损失/训练修复；
2. EventMATR v2（暂称 Trajectory-Hazard EventMATR，TH-EventMATR）；
3. 或者需要完全不同的模型。

不得因为 v1 存在学习失败就把已经成立的动态事件、严格因果和不可变输出设计全部丢弃。
同时也不得因为已经投入实现成本而强行保留无效模块。

## 4. 必须逐行核验的代码问题

请至少检查：

- `dataset.py::_make_event_targets`
- `criterion/criterion.py::SetCriterion.loss_event`
- `models/models.py::MATR.forward`
- `models/event_memory.py::EventTransitionHead`
- `models/event_memory.py::OwnerEventDecoder`
- `models/event_memory.py::DynamicEventMemory.step`
- owner matching、event birth、cancel、end、emit 和 memory queue 相关实现

对下面每个假设给出 `CONFIRMED / PARTIAL / REFUTED`，附精确 `file:line`、
张量形状、梯度路径和原因。不能直接接受我们的判断。

### H1：稀疏正例被背景淹没

核验 birth/alive/end target 是否只在少数 prefix/query 上为正，而 BCE/CE 是否在所有
`B×Q` 背景位置平均。推导全背景预测为何可能得到很低 loss，并计算或给出可复现脚本统计：

- 每批 birth/alive/end 正例比例；
- owner state 的 BG/START/ALIVE/END 分布；
- 每种损失中正负样本的总梯度质量；
- 全背景常数解的理论 loss 与梯度。

### H2：训练与在线运行分布不一致

核验训练时 `event_runtime_during_training` 的默认值和实际 formal 配置；
判断训练是否真正逐 prefix 展开动态事件，还是只训练 dense per-query owner prototypes。
比较训练输入分布与推理时 ragged persistent owner records。

### H3：START 是脆弱的单帧 crossing

核验 GT birth 是否只在首次 crossing 的一个 prefix 出现，runtime 是否还要求
`current_start & ~previous_start`。说明漏掉该 prefix 后能否恢复，以及 late birth/rebirth
是否有真实可学习路径。

### H4：当前 owner 不是完整的实例轨迹学习

核验 fresh/sticky owner、greedy matching、owner embedding 更新和 end 决策。
判断它是否已经等价于 trajectory-set matching；尤其分析同类重叠、同时开始、交叉持续时间、
同时结束、短动作和超长动作。

### H5：当前是否已有 hazard/survival 与分层视觉记忆

明确区分：

- 四态分类或 logit margin；
- 离散时间 birth/end hazard 与 right-censored survival likelihood；
- active-event memory；
- visual-history memory；
- MATR 固定 memory queue；
- 样本自适应的 learned keep/merge/compress。

不得把名称相似当作算法已经实现。

## 5. 对“全零”的回答必须严格

请回答：

1. 什么修改只能增加非零发射，却不能保证正确检测？
2. 什么训练目标能使“所有 query 永远为背景”不再是低损失最优解？
3. 是否能数学保证非零提案？如果不能，明确能保证的是：
   - 全背景解受到正例归一化惩罚；
   - 正例获得稳定梯度；
   - 训练真实见到活动事件；
   - 零发射会在早期诊断门被发现，而不是再浪费 100 epochs。
4. 如何避免通过人为降低阈值或强制最少发射来制造假阳性？
5. 如何在 calibration 数据上选择决策代价，同时保持 locked test 只评一次？

## 6. 必须提出三种候选并推荐一种

至少比较：

### A. 最小学习修复

例如 class-balanced/focal state loss、正例归一化、start/end-centered prefix sampling、
训练时 runtime unroll。说明它能解决什么、不能形成什么创新。

### B. 推荐主模型：Trajectory-Hazard EventMATR

请审判并完善以下构想，而不是照抄：

1. 保留官方 MATR 主干和 native 输出；
2. 增加零初始化、低秩的 causal event adapter，避免一开始破坏 MATR 表示；
3. 用数量可变的 start-created event trajectories 代替长期固定槽位；
4. birth 前采用 permutation-aware set matching；
5. birth 后锁定事件身份，由自己的轨迹条件化 class/alive/end；
6. 使用离散时间 birth hazard、survival 和 end hazard；
7. 支持 right censoring、late birth、cancel/rebirth；
8. 训练时真实逐 prefix 展开 ragged event records；
9. 使用 teacher forcing → scheduled sampling → predicted-runtime curriculum；
10. 把 active-event memory 与 visual-history memory 分开。

### C. 更强但风险更大的替代模型

可考虑直接使用 event-centric transformer、point-process decoder、semi-Markov model、
neural survival process 或其他更好的严格因果实例模型。必须说明它是否比 B 更创新、
更容易训练，以及是否会破坏 MATR 公平母体。

最后只能推荐一个主路线，并明确哪些 v1 组件保留、替换、删除、后置。

## 7. 给出完整数学定义

至少定义：

- prefix 观测 query；
- 动态事件集合及每个事件的 identity/owner state；
- birth hazard、continue/survival、end hazard；
- right-censored 与完整事件的 likelihood；
- class 与 start/end boundary posterior；
- birth 前集合匹配 cost；
- birth 后身份一致性/对比损失；
- same-class overlap 的 permutation 处理；
- late-birth 与 cancel/rebirth 的合法状态机；
- 不可变 emit 条件；
- 总损失和每项的归一化方式。

解释为什么该目标不会再让大量背景数量线性淹没稀疏事件正例。

## 8. 必须提供可落地的核心 PyTorch 代码

不要只给伪代码。请按照当前仓库结构给出尽量可编译的核心代码或 unified diff，
至少覆盖：

1. `models/event_memory.py`
   - `EventHazardHead` 或你推荐的替代 head；
   - event trajectory state/update；
   - owner-conditioned end hazard；
   - ragged/padded mask 处理。
2. `models/models.py`
   - 从 MATR query 接入零初始化 event adapter；
   - 训练时按 prefix 顺序展开 runtime；
   - teacher-forcing/scheduled-sampling 接口；
   - 保证 native lane 完全不变。
3. `dataset.py`
   - 不再只有脆弱单帧正例的 prefix-visible target；
   - hazard risk set、right censoring、late-birth target；
   - 不读取未来信息的理由。
4. `criterion/criterion.py`
   - 正例归一化或 class-balanced hazard likelihood；
   - identity/set matching loss；
   - boundary/class/trajectory loss；
   - 数值稳定处理。
5. 必要的 matcher 与测试
   - 同类重叠；
   - 同时开始/同时结束；
   - 单帧动作；
   - 长动作和 EOS censoring；
   - missed crossing 后 late birth；
   - 零因果泄漏；
   - 输出一次且不可修改。

每段代码必须说明输入/输出 shape、与现有符号的连接点、哪些参数新增、哪些原参数复用。
若完整 patch 太长，核心类和所有关键调用点必须完整，不能只写函数名占位。

## 9. 分层/学习式记忆的定位

回答分层记忆是否应当：

1. 立即进入第一版 TH-EventMATR；
2. 与核心事件轨迹并行实现、但在核心通过后才正式启用；
3. 或完全删除。

我们的倾向是把记忆分成：

- 活动事件记忆：未结束事件，不能因视觉压缩随意删除；
- 视觉历史记忆：学习 keep/merge/compress，按事件不确定性和动作长度自适应。

不要把 `memory_size` 简化为一个全局可学习整数。请给出真正样本自适应、可微或可训练的
保留/合并机制、预算正则和物理安全上限，并判断它与 HAT/HEM 是否重叠。

## 10. 最新竞品与创新性审判

使用论文和官方代码等一手来源检查截至当前可见的相关方法，至少包括：

- MATR；
- ActionSwitch；
- HAT；
- Hierarchical Event Memory；
- Temporal TrackFormer/MOTR 类持久 query；
- 其他 2024–2026 Online TAD / streaming action localization 方法。

输出“组件—竞品—重叠—剩余非显然性”矩阵。不得声称以下单独构成创新：

- 状态转移；
- persistent query；
- 分层记忆；
- dynamic memory；
- fixed threshold calibration；
- 仅把多个现有模块相加。

重点审判下面的组合是否足够新：

> 在 closed-set、全监督、严格因果 On-TAD 中，开始证据出现时创建数量可变的动作实例轨迹，
> 使用 censoring-aware birth/end hazard 学习即时起止决策，birth 后保持同类实例身份，
> 并以训练—推理一致的动态事件展开实现低延时不可变区间输出。

如果创新仍不足，请提出更强、但仍服务于性能的核心学习机制。

## 11. 公平并行实验计划

保持官方 MATR 的数据、RGB+flow 4096 维特征、segment 64、queries 10、memory 7、
batch 64、100 epochs、seed 52、优化器、调度器和评测设置。建议审判以下并行归因：

| Lane | 模型 |
|---|---|
| N | exact native MATR |
| R | 平衡状态/风险损失 + 训练 runtime 展开 |
| T | R + 实例轨迹与身份匹配 |
| H | R + birth/end hazard |
| TH | T + H 完整模型 |

请给出：

- 一次真实 batch smoke；
- 共同的 5/10/20 epoch 诊断点；
- 哪些门只判断数值/学习健康，不能误当性能；
- 是否所有 lane 都应跑满 100 epochs；
- 单种子筛选与三种子确认；
- calibration/validation/locked-test 的合法划分；
- mAP@0.3:0.7、Recall、birth/end/emit latency；
- wrong-start、owner swap、ID switch、fragmentation、duplicate close；
- pred/GT、活动事件数、记忆长度、峰值显存、吞吐；
- future perturbation、same-class overlap 和 EOS tests；
- 明确的停止/转向条件。

不得用固定“至少发射 N 个”或降低阈值冒充学习成功。

## 12. Raw-RGB 最终阶段

特征级实验只用于隔离事件机制。请给出 TH-EventMATR 通过后接入严格因果 raw-RGB
骨干的路线：

1. frozen backbone；
2. adapter/LoRA；
3. joint training。

说明哪些事件机制必须原样继承，如何避免 raw-RGB 增益掩盖事件模型无效，以及最终论文
需要哪些特征级和 RGB 级证据。

## 13. 强制输出格式

请按以下顺序回答：

1. `VISIBILITY_CERTIFICATE`
2. 十行以内执行摘要
3. 当前代码事实表（每项带 `file:line`）
4. 对 H1–H5 的逐项裁决
5. “保留 EventMATR 还是重建”的明确结论
6. 三种候选路线及推荐
7. 推荐模型的结构图、数学定义和因果时间线
8. 可编译核心代码/unified diff
9. 分层记忆的保留方式
10. 最新竞品与创新性矩阵
11. 完整并行实验表
12. 风险、失败模式、停止/转向条件
13. 按文件排序的最小实现步骤
14. 最终总裁决：`PASS / REVISE / REJECT`

不要输出：

- 空泛“可以尝试”的建议；
- 与精确代码无关的通用教程；
- 工程平台重写；
- 未核验代码却声称已实现；
- 把训练 loss 下降等同于定位性能；
- 保证某个新模型一定获得非零 mAP；
- 任何网络安全、恶意宿主、TCB 或篡改可见性讨论。
