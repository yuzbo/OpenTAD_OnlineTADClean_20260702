---
type: experiment-design
updated: 2026-08-06
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
- 最终状态：`COMPLETED 0:0`，用时 `00:01:11`，节点 `g0048`；
- receipt SHA-256：`63b855129d2f510a08e0a13414bf40fb3b379715aec4a09d09b2d01985a2085a`；
- 远端完整测试：`191 passed in 28.46s`；
- official batch：`64×64×4096`，`25` 条有效事件行，Toeplitz 时间合同 `PASS`；
- 关键梯度范数：birth `10.5803`、transition `62.1741`、owner state `26.4259`、owner cross-attention `31.2566`、MATR segment decoder `8.3973`、MATR memory decoder `1.3505`；
- strict checkpoint reload、起止 source identity 和 clean tree 均 `PASS`；
- `loss_event_end=0` 是 D1.6 唯一三状态似然的预期合同，不是结束未学习；三状态 owner loss 为 `1.2076` 且 owner/双 decoder 梯度均非零；
- 只提交 `slurm_eventmatr_d16_risk_smoke.sh`，没有提交训练、测试集评测、阈值搜索或自动监控任务。

### 下一项已注册机制实验

官方单批门通过后，只释放一个同源、同初始化、同训练暴露的成对实验：

| 臂 | D1.6 风险 | 其余模型/目标 |
|---|---|---|
| C：冻结控制 | `none` | D1.3 combined + D1.4 decision-aligned bag + TH |
| R：风险修复 | `policy_independent` | 与 C 完全相同 |

两臂均使用官方训练特征、1 轮/3270 物理批、seed 52、相同初始化哈希、相同 Adam/调度/teacher 比例，不访问 locked test。它不是论文性能比较，而是可以写入方法分析/消融的机制证据。

主结果不是训练 mAP，而是对两个冻结 checkpoint 在相同 `3,001` 个可观察结束事件上的逐事件 owner END margin。预注册判定：完整性与风险覆盖先通过；随后风险臂相对控制臂的逐事件 margin 差异必须具有正中位数，且按视频聚类的 95% bootstrap 区间下界高于零。零边界 END-positive fraction 作为共同推理规则下的第二结果，不搜索阈值。若失败，D1.6-R 不得解锁 D1.6-Q/F 或任何长训练。

## 创新与论文边界

持续查询、长短期记忆和生存/竞争风险各自都有先例，不能单独声称创新。潜在贡献是：在标准全监督严格因果 On-TAD 中，把 MATR 的发现查询改造成可出生、休眠、重新获取和结束的模型内持续事件查询，并以不受运行时决策控制的删失风险训练同一个时间展开。若 Q/F 未超过 R 与 P，或者严格因果官方协议未闭合，该贡献应降级或否决。

## 成对一轮机制实验的精确执行记录

- 训练代码：commit `2fad75ea9a2afb608a296d67ba0345f73375b0cf`，tree `e666fc57f7fdd61a4521fe4897c375c68b831b0c`，manifest SHA-256 `5bad72a6bd513e1276da7643e69a058381f6e641d783620fa6820327db334d78`；
- 本地完整合同：`193 passed`；新加入初始化模型全部张量逐字节 SHA-256 与参数名/类型/形状签名，成对收口器要求两臂完全相同；
- 同提交官方单批复核：job `1223295`，`COMPLETED 0:0`，`193 passed in 23.97s`，receipt SHA-256 `ba0148c21c81687aab8507c327c1c669533cb2f413dbb5831482dee46a646d53`，`test_access=false`、`formal_training_started=false`、`checkpoint_updated=false`；job `1223294` 仅因遗漏环境激活变量在两秒内技术失败，未进入测试或训练，不构成科学结果；
- 成对训练 array：job `1223320_[0-1]`；task 0 为冻结控制，task 1 为策略无关风险；依赖收口 job `1223332`；输出根 `/data/run01/sczc063/yuzibo/runs/eventmatr_d16/paired_mechanism_20260805_2fad75e`；
- 两臂唯一允许差异为 owner risk contract；共同固定 D1.3 combined、D1.4 decision-aligned bag、TH、官方训练数据、seed 52、1 轮、3270 物理批、3.34e-6 有效学习率、0.5 teacher 比例与无固定事件阈值；
- 当前状态：训练已提交、尚未形成终点结果；不得根据进度、训练集检测分数或中间日志提前判定。

## 冻结后的共同结束边际分析合同

诊断实现 commit `92e06fc8e78bd8a715000a9db82eaab929abf626`，tree `2e88ec5acd8d8dcd1a6df4378806cc6a89648ed2`。该提交不改变正在运行的训练源，只为两个冻结 checkpoint 提供共同分析器：

1. 两臂都在评估模式、关闭梯度与随机失活，并严格加载相同结构的全部权重；
2. 只在真实事件已因果可见后建立同一策略无关风险集，未观察终点仍保持右删失；这属于训练集机制诊断的共同 oracle risk set，不是部署推理输入或论文性能；
3. 必须各自覆盖同一 `3,001` 个可观察结束、无重复、完整遍历 `3,270` 个官方物理批，并证明 checkpoint 哈希前后不变；
4. 主统计量为逐事件“风险修复减控制”的结束边际中位数；按视频聚类 bootstrap 固定 `10,000` 次、随机种子 `52016`、95% 百分位区间；
5. 仅当中位数严格大于零且区间下界严格大于零时通过；零边界正结束数只作次要结果，不搜索或降低阈值；
6. 通过只解锁模型内部持续查询的设计与实现，不解锁论文性能、多种子、长训练或测试集；失败则否决当前风险修复作为后续结构起点。

## 首次成对执行失败与显存闭合修订

首次 array `1223320_[0-1]` 没有形成科学结果：风险臂在第 `33/3270` 物理批发生 CUDA 显存耗尽；控制臂当时仅运行至部分批次，随后主动取消；依赖收口 `1223332` 同步取消。没有任何终点 checkpoint、成对回执或效果判定被保留。

根因不是 target-backed 继续/结束轨迹，而是未解析预测风险在 EOS 前本来没有状态标签，却仍在每个前缀进入 owner decoder 并把 masked logit 的无损失计算图保留到物理批末。修订保持科学语义并关闭无效图：

- 未解析预测风险记录仍独立于运行时取消而持续存在；
- 运行时记录仍活跃时，用其当前 owner 的 detached snapshot 刷新风险记录；运行时记录消失后保留最后快照；
- 实际观察 EOS 前不解码为损失行；EOS 时全部未解析记录只解码一次并接受 CANCEL；
- target-backed 风险继续使用同一个按时间顺序、可微的 owner update/unroll 与继续/结束监督，不截断其有标签的计算图。

修复训练源：commit `ec1ac7e4b54922d17f0d919b1c58b5041aa482ee`，tree `b5f26659aece8b921cf966bd858f52ee9474c12e`，manifest SHA-256 `ec57542aec183b94e6de70f9e279e7e2c1844f01cd47b70e8d22d5df3a743190`。本地 `196 passed`；同提交官方单批 job `1223453` 为 `COMPLETED 0:0`，远端 `196 passed`，receipt SHA-256 `24433dcf09bd06346126849482e54243895dc9309bc63d18c1d9b8bd35057171`。全新成对 array `1223457` 与依赖收口 `1223458` 已从零提交，输出根 `/data/run01/sczc063/yuzibo/runs/eventmatr_d16/paired_mechanism_20260806_ec1ac7e`；旧执行不复用。

风险臂已安全越过首次失败点（新执行达到第 `37` 批时仍运行，旧执行在第 `33` 批已失败）。注册的共同结束边际分析 job `1223472` 依赖 `1223458` 成功后才会启动；它是实验 DAG 的结果分析节点，不是轮询或自动监控。任何上游失败都会阻止它运行。

## 第二次成对执行失败、双图根因与连续批压力门

上述“已越过第 33 批”只是运行进度，不是闭合证据，现由终态裁决取代。第二次 array `1223457_[0-1]` 仍未形成任何成对科学结果：

- 控制臂完成 `3270/3270` 个优化步骤并写出终点 checkpoint，但收据终结器失败；原因是正式训练前的冒烟原始收据含 `formal_training_started=false`，而 `verify_source_identity.py` 生成的压缩身份文件漏抄该字段，终结器又强制读取它。这是回执字段契约错误，不是训练效果失败；由于风险臂失败，控制臂产物不得单独复用为新成对实验。
- 风险臂在第 `43/3270` 物理批再次发生 CUDA 显存耗尽。首次修复已消除“未解析预测风险”的无损失图，但代码仍为运行时 `DynamicEventMemory` 保留另一条 owner 递归计算图。D1.6 的监督行已经完全来自独立 `PolicyIndependentRiskMemory`，因此这条运行时图不进入损失，只影响离散运行策略，却随活跃运行记录增长占用显存。
- 依赖收口 `1223458` 与共同端点分析 `1223472` 均因上游不可能成功而取消；没有删除训练产物，也没有运行结果分析。

最小修复保持路线的科学语义：D1.6 只将不受监督的运行时 owner 状态更新设为 detached，运行时出生、匹配、取消与审计仍按同一因果顺序执行；真正贡献继续/结束标签的策略无关风险记忆仍保留按时间顺序的可微更新。控制臂的原 D1.4 运行时监督图不变。压缩身份文件同时显式传递 `formal_training_started`、`paper_performance_valid`、`threshold_search` 与 `checkpoint_updated`。

修复源为 commit `3149049eece7ff92601ab822e7201a1438e53f78`，tree `2d33948b63367987a15d821b36d2d8453397d06d`，manifest SHA-256 `1f93bd7667d407c0f285375a89b9a881e0c4eba7877d45613ad357c61a31d71f`。本机静态编译、JSON、`git diff --check` 与独立身份回执回归测试通过；本机 PyTorch DLL 初始化失败，因此不得把本机神经网络测试标为通过，完整套件必须在同提交集群任务中复核。

原“一批冒烟”已被证实不足以覆盖活跃 owner 数随时间增长的显存风险。新的放行门固定执行正式顺序中的前 `64` 个官方物理训练批，跨过既有第 `33`、`43` 批失效点；使用相同 seed、批宽、因果前向、反向、Adam 更新和有效学习率，逐批要求有限损失/梯度，并要求 owner 状态、owner 交叉注意力及 MATR 两级解码器均出现正梯度。全部 `64` 次更新随后丢弃，不写 checkpoint、不计算检测性能、不访问 locked test。工程安全门预注册为峰值显存保留比例不高于 `0.90`，即至少留下 `10%` 设备余量；这不是模型效果阈值，也不会被用于搜索模型决策阈值。

连续批门的三次前置失败均发生在压力循环之前，且均保留为非科学结果：`1226957` 因 Slurm 未注入已验证的环境激活路径而在 1 秒内退出；`1226959` 的完整套件为 `195 passed, 1 failed`，唯一失败是协议版本仍写死 v12；`1226964` 同样为 `195 passed, 1 failed`，唯一失败是测试仍要求压力门通过前 `training_authorized=true`。两条期望随后改为 v13 且 fail-closed，未降低任何模型、显存或科学门槛。

修复提交 `89aff8d14b278262670e67ab2c20f8c601fb6812` / tree `a503209913fd6a8ff4c2046ea20c0a4e2e538a31` 的正式连续批门 `1232776` 在 `g0024` 完成 `0:0 / 00:15:13`：完整套件 `196 passed`，前 64 个官方物理批全部完成并丢弃 64 次 Adam 更新，逐批 Toeplitz 因果合同通过；owner state、owner cross-attention、MATR segment decoder、MATR memory decoder 的最大梯度范数分别为 `41.5337 / 34.2135 / 9.0277 / 2.8198`。RTX 4090 的峰值 allocated/reserved 为 `7,117,339,136 / 7,367,294,976` 字节，reserved 占总显存 `0.290207`，显著低于预注册上限 `0.90`。receipt SHA-256 为 `6efe2250a734ea6b843d2b1a160c403d214f68c2d62144be15ef76f89978cbe6`；`test_access=false`、`checkpoint_updated=false`、`performance_metric_computed=false`、`paper_performance_valid=false`，起止身份均 clean/PASS。

为避免“外部门已通过而源码清单仍写训练禁止”，该证据被写入 protocol v14，并形成最终授权源 commit `53744e6e565d72db36408574f7c8a6189e494ad1` / tree `5d684d673d1fa7e90c0723d7ea1291a6570b6caf` / manifest SHA-256 `0d1be21af0d19eb40f0fa15e4f9a8efcdcf93fc1f3a55c4790d8b268faaecdd7`。同提交 64 批复核 `1232786` 已完成 `0:0 / 00:13:23`，完整套件 `196 passed`，峰值 reserved 比例再次精确为 `0.2902069108`，receipt SHA-256=`7e62a54c08ad127d42caccd152268888b9defee6fbc31c6a6f336416cb58c284`，最终身份 clean/PASS。

该门通过后，`afterok` 实验 DAG 已启动成对训练 `1232787_[0-1]`：控制臂在 `g0024`、风险臂在 `g0041` 从零运行；收口 `1232788` 与共同 3,001 端点分析 `1232789` 仍按依赖等待，任何上游失败都会阻止下游。输出根 `/data/run01/sczc063/yuzibo/runs/eventmatr_d16/paired_mechanism_20260809_53744e6_r3`。这不是自动监控，不访问 locked test，也不释放论文性能。
