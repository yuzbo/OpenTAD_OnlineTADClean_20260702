---
type: experiment-design
updated: 2026-08-05
status: implementation-authorized / training-blocked
scope: EventMATR D1.6 policy-independent risk repair and MATR-internal persistent event-query redesign.
---

# EventMATR D1.6: MATR 内生持续查询与策略无关竞争风险

## 一句话裁决

D1.5.1 已排除“只禁止取消”“只刷新身份”和“只修账本”足以恢复结束的解释。用户于 2026-08-05 明确批准把 MATR 作为基线并打开其模型内部进行结构创新。因此 D1.6 获准按可归因的两阶段实现：先在现有表示上修复策略诱导的自删失，再把持续事件查询送入 MATR 的两级解码器。新训练仍须等待纯函数合同、严格因果集成测试和官方单批冒烟全部通过。

## 精确起点与证据边界

- 代码起点：`adcb0db3153e701644af5338335f266e80badeff`，tree `cd4da1f7a1f2866a39c399bccecc71262b4390db`。
- D1.5.1 回执：SHA-256 `ef5a437d613c92b08111ceb3764c10ed39872d532948b8892c2464461e22e4c5`。
- `3,001` 个可观察结束中，所有正式路径的正结束边际均为零；禁止取消只恢复 `7/6/3/3` 个事件，oracle 影子路径的上置信界低于 `0.6%`。
- 这些结果只定位机制，`strict_causal_paper_result_valid=false`、`paper_performance_valid=false`；不得作为论文性能。
- 原生 MATR 必须保持精确独立，不能被 D1.6 的状态、损失或加载器改动污染。

## 已定位的结构问题

1. 当前 EventMATR 的事件头位于 MATR 两级解码器之后；持久 owner 只能交叉注意已经生成的固定查询，不能反过来改变当前检测表示。
2. 运行时取消会把 owner 移出 active 集合；后续前缀不再产生该轨迹的结束风险，因此训练风险集受模型自身早期策略控制，形成自删失。
3. 三状态交叉熵与另一个二元结束 margin 同时优化，结束概率被两个不一致目标重复定义。
4. 官方物理批次是同一视频的连续前缀，并依赖 `batch=时间=64` 的 Toeplitz 对称滑窗。当前主解码器先并行计算整批，再顺序更新生命周期；直接把动态查询拼到整批 decoder 会使后一前缀读不到前一前缀的新状态。
5. 现有 temporal assignment、起点 bin 和运行时状态选择包含离散路径。它们可以用于训练匹配或不可微账本，但不能再被描述为端到端持续事件推理本身。

## 最多两个主要科学主张

### 主张一：策略无关风险闭合

对于已经出生的训练事件，`at-risk` 身份由当前时刻可见的出生/存活/结束状态决定，不由运行时是否取消决定。真实事件从首次出生到观察到结束持续贡献一个统一的 `取消/继续/结束` 离散竞争风险似然；视频在仍存活时结束只贡献继续项并右删失。未匹配预测轨迹在视频结束前保持未决并屏蔽状态标签，只在实际观察到视频结束且仍无匹配时才成为可监督的取消。

### 主张二：MATR 内生持续事件查询

事件查询不是输出后处理器。出生后，持续事件查询与原有发现查询一起进入 MATR 的当前片段解码器和历史记忆解码器；快速状态负责当前边界证据，慢速因果原型负责身份。运行时取消只令查询进入 dormant 状态，不删除身份；后续证据可以用同一身份重新获取。预测和训练调用同一逐时刻更新函数。

反主张：若增益可由相同参数量的加宽 MATR、单纯风险损失或仅后接 owner decoder 解释，则完整结构创新主张不成立。

## 可识别实施顺序

### 阶段 A：D1.6-R，现有表示上的风险修复

- 新建与运行时 `DynamicEventMemory` 分离的训练风险记忆。
- 真实轨迹在当前可见出生时由当前查询初始化；运行时取消不得删除该风险轨迹。
- 未匹配预测轨迹在首个后续前缀加入风险记忆；在 EOS 前标签为未决/屏蔽，EOS 时仍未匹配才监督为取消。
- 统一使用事件归一化的三状态负对数似然；删除该变体中重复的二元结束 margin 损失。
- 同一物理批次按前缀顺序更新，批次边界只 detach 计算图，不清除风险身份。
- 这是表示控制和问题真实性实验，不是最终论文模型。

### 阶段 B：D1.6-Q，MATR 两级解码器内部持续查询

- 保留原有 `Q` 个类别查询与 `Q` 个边界查询作为 discovery bank。
- 把 ragged 的 active/dormant event queries padding 后加入两个 decoder，并传递 padding mask；输出时明确拆分 discovery 与 event axes。
- 采用 fast state 与 slow prototype 的因果门控更新；slow prototype 只读当前与过去证据。
- 将一个物理批次改为共享函数的逐前缀 recurrent decode，使时刻 `t` 能读取 `t-1` 的事件状态；原生 MATR 路径保持原批处理实现。
- 取消进入 dormant；结束才关闭风险，提交后的 immutable ledger 只消费模型输出且不反写模型。

### 阶段 C：D1.6-F，完整组合

仅当 A 与 B 各自通过结构门后组合。不得以一次融合训练跳过归因。

## 实验矩阵

| 变体 | 风险闭合 | 内部持续查询 | 慢速身份原型 | 作用 |
|---|---:|---:|---:|---|
| N：精确原生 MATR | 否 | 否 | 否 | 官方基线；必须逐参数/配置保持原样 |
| R：风险控制 | 是 | 否 | 否 | 验证自删失和重复目标是否真实可修 |
| Q：查询控制 | 最小统一风险 | 是 | 否 | 验证模型内表示而非后处理 |
| F：完整模型 | 是 | 是 | 是 | 目标方法 |
| P：参数量匹配 MATR | 否 | 否 | 否 | 排除仅增加参数的解释 |

先做无性能声明的结构验证；只有结构门通过后，才允许固定种子 52 的训练集机制实验。论文可比结果仍要求 N/P/F 使用相同官方输入、优化器、100 轮预算、评测器、选择规则和终点，且只能在开发门冻结后访问 locked test。

## 实施前冻结的测试门

### 纯函数与生命周期合同

- 运行时取消后，真实风险轨迹仍在下一前缀产生继续或结束监督；
- 同一真实轨迹最多一个观察结束；无结束时只产生右删失继续项；
- 未决预测轨迹在 EOS 前不被错误标为取消；
- 三状态概率归一、事件等权归一、有限损失与非零梯度；
- 预测取消不会改变训练风险分母；批次边界 detach 不清空身份。

### 严格因果集成合同

- 拒绝 `true_duration`、完整视频时间和离线 EOS 进入 D1.6 模型边界；
- 当前输出对未来帧、未来元数据和未来标注不变；
- prefix `t` 对 `t-1` 事件状态敏感，但对 `t+1` 不敏感；
- 训练与推理调用同一时间更新函数；
- negative start 被截到零，ledger 正长度、不可变、无重复、无容量耗尽；
- 原生 MATR 输出和参数结构不因 D1.6 路径改变。

### 官方单批冒烟

- exact source/tree、官方 train batch、`test_access=false`；
- forward/backward/Adam/重载有限；
- 风险头、持续查询、片段解码器和记忆解码器均有非零梯度；
- 官方 Toeplitz 因果合同通过；工作树和 checkpoint 不被诊断修改。

## 训练授权门

当前：实现 `true`，本地合同 `PASS`，官方单批冒烟 `pending`，任何新训练 `false`。

只有上述门全部通过且形成 exact commit/tree 回执后，才可登记一个固定种子 52 的短机制实验；其目标是验证出生、取消、继续、结束、reacquisition 与风险覆盖，不是比较论文性能。五/十/二十轮矩阵、locked test、多种子、raw RGB、阈值搜索和降低阈值继续禁止。结构门成功后另立决策，才可申请匹配的 100 轮 N/P/F 官方比较。

## D1.6-R 实施回执

日期：2026-08-05。

- exact commit：`04a202a7c53bbfce62c573fad64d8742be57eb45`；
- exact tree：`a13cd191e3523f07f16cc94d02fc825ce0a0ac34`；
- protocol manifest SHA-256：`14b648cc3be8ddb85b0ad56b4cb146a26238a86b09160e26759075a86bf0b20c`；
- GitHub 分支：`codex/eventmatr-d1`，已推送；
- 本地完整套件：`191 passed`；Python compile、Slurm Bash syntax、JSON 协议和 `git diff --check` 均通过；
- 新路径只在显式 `event_d16_variant=policy_independent` 时启用，默认 `none` 保留冻结 D1.5；原生 MATR 不创建风险记忆；
- 新风险记忆与 runtime event memory 分离。真实轨迹从当前可见出生/恢复开始保留；未匹配预测在 EOS 前状态标签为 `-100`；实际 EOS 时仍未匹配才监督取消；
- D1.6 criterion 对真实风险只接受继续/结束，以事件片段等权的三状态交叉熵定义唯一竞争风险；旧的额外二元结束 margin 在该变体为零；
- 整模型回归证明 owner state loss 可反传到 owner decoder、MATR segment decoder 和 MATR memory decoder；
- 新 official-batch gate 为 `scripts/slurm_eventmatr_d16_risk_smoke.sh`，强制完整测试、Toeplitz 时间合同、单 GPU、官方训练批次、forward/backward/Adam/strict reload、四类关键梯度、`test_access=false`、`checkpoint_updated=false`。

当前门状态修订为：实现 `true`，本地合同 `PASS`，官方单批冒烟 `pending`，D1.6-Q `false`，任何模型训练 `false`。本地测试不是官方批次证据，也不改变论文边界。

### 官方单批提交

- Slurm job：`1222965`；
- checkout：`/data/run01/sczc063/yuzibo/EventMATR_D16_04a202a7`；
- output：`/data/run01/sczc063/yuzibo/runs/eventmatr_d16/risk_smoke_20260805_04a202a7`；
- 提交时状态：`PENDING`；
- 只提交 `slurm_eventmatr_d16_risk_smoke.sh`，没有提交训练、测试集评测、阈值搜索或自动监控任务。

## 创新与论文边界

持续查询、长短期记忆和生存/竞争风险各自都有先例，不能单独声称创新。潜在贡献是：在标准全监督严格因果 On-TAD 中，把 MATR 的发现查询改造成可出生、休眠、重新获取和结束的模型内持续事件查询，并以不受运行时决策控制的删失风险训练同一个时间展开。若 Q/F 未超过 R 与 P，或者严格因果官方协议未闭合，该贡献应降级或否决。
