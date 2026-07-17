# 前置路线裁决

**路线前置裁决：`REVISE_ROUTE_BEFORE_IMPLEMENTATION`**

**最终裁决：`REVISE_ROUTE_AND_REVIEW_AGAIN`**

当前标准 On-TAL 任务本身有价值，但固定提交中的证据只证明了 **Q2 的局部训练—运行生命周期错误**，没有证明“前缀共享潜在事件过滤器”是解决该任务所必需的方法。R‑A 尚未排除无持续身份的完成集合预测、普通持续查询、干净的训练顺序修复，以及一维 TrackFormer/MOTR 重建。现有零阶段门禁也不能把 R‑A 特有机制与模板记忆、事件计数、阈值策略或普通集合预测区分开。

因此，本轮在强制前置阶段停止：

* **不冻结状态、状态转移、UOT、损失、提交阈值、生成器、训练参数或证据常量；**
* **不输出逐文件实现计划；**
* **不允许开始代码实现；**
* **不允许任何真实数据效果验证；**
* **不允许 GPU 测试；**
* **GPU 授权保持 0 小时。**

---

# A. 仓库和提交核验

| 项目       | 核验结果                                                                                                                          |
| -------- | ----------------------------------------------------------------------------------------------------------------------------- |
| 固定仓库     | `[代码核验]` 审查对象为 `yuzbo/OpenTAD_OnlineTADClean_20260702`。                                                                       |
| 固定提交     | `[代码核验]` `f9419612cc90860019cd9a124833568489c9d38c` 可由 GitHub 提交对象解析，提交信息为 `Record R-A CPU P0 route review`。                  |
| 快照纪律     | `[代码核验]` 本轮所有仓库文件读取均显式使用该 SHA，没有使用默认分支或移动分支。                                                                                  |
| R‑A 实现状态 | `[代码核验]` R‑A 文档明确标记为未实现、没有 P0 结果、没有效果证据、没有 GPU 权限。                                                                            |
| 外部证据边界   | `[未知]` 当前仓库不包含完整 THUMOS 标注字节、缓存特征、checkpoint 和原始 Q2/CRS trace bundle；因此本轮核验了代码和仓库内绑定记录，但没有重新逐字节重放外部实验包。仓库说明也明确区分代码、固定特征和外部数据。 |
| 执行纪律     | `[代码核验]` 本轮未修改仓库、未创建提交或 PR、未运行训练、未运行 P0、未使用 GPU。                                                                              |

至少核验了用户列出的任务文档、DR‑044/045、T37/T38、G16、R‑A 设计与 P0 清单、后 Q2 审查及吸收文档，并交叉读取了 Q2 head、detector、prefix supervision、数据集、不可变记录簿和评测器。

---

# B. 当前授权边界

1. `[实验核验]` Q2/R1 的终止结论保持不变：绑定记录显示实际运行产生 `5517` 次提交和 `2206` 次 exhaustion；`-2` 偏置仅剩 `10` 次提交且两个随机种子完全不提交，因此“零 exhaustion”来自静默抑制，而非生命周期修复。

2. `[代码核验]` DR‑045 的 `GO_NEW_ROUTE_P0_ONLY` 只是历史上的条件性设计授权；同一记录明确承认实现不存在、UOT、生成器、训练与数值常量未闭合，而且把 temporal MOTR/TrackFormer 视为严重重建威胁。

3. `[推断]` 本轮收到的新审查规则明确要求重新检查任务价值、简单替代方案和 P0 辨识性。因此，历史授权不能替代本轮裁决。

4. `[推断]` 当前唯一合法的项目动作是整理新的路线级证据：包括只读标注普查、正式定义简单基线和重新设计可辨识的 P0。它们不等于授权模型实现或训练。

5. `[代码核验]` 当前保持禁止：Q2/R1 修改、`-2` 路线、缓存特征效果训练、GPU profile、正式训练、视觉微调、原始视频训练；GPU 授权为零。

---

# C. 未闭合常量清单与强制前置路线审查

## C.1 当前真正未闭合的不是数值常量，而是科学前提

在下列问题闭合前，选择载体数、UOT 公式或网络维度没有科学意义：

| ID  | 未闭合前提                                               | 当前状态                                                                  |
| --- | --------------------------------------------------- | --------------------------------------------------------------------- |
| S1  | 精确 THUMOS 分割中，同类重复、同类重叠、异类重叠、同一 bin 结束/启动、最大并发数的发生率 | `[未知]` 固定提交引用外部标注与 split manifest，仓库中没有可供本轮独立普查的对应字节。                 |
| S2  | 当前缓存特征是否严格因果                                        | `[未知]` 数据集验证 source frame 单调且不晚于当前帧，但缓存编码器的 clip 构造与时域感受野证书不在固定提交中。   |
| S3  | 持续潜在身份是否是正式输出合同的必要变量                                | `[代码核验]` 当前评测器只消费不可变区间、类别、分数和提交时间；不要求模型公开或保持可观测的载体身份。                 |
| S4  | Q2 失败是否代表领域普遍缺口                                     | `[代码核验]` 目前只证明 Q2 在监督分配前读取硬 runtime availability 的具体错误。               |
| S5  | 普通集合预测、持续查询或干净顺序修复是否不足                              | `[未知]` 没有匹配的简单基线结果。                                                   |
| S6  | R‑A 与一维 TrackFormer/MOTR 的算法差异                      | `[推断]` 当前差异主要是时序区间头、连续占用、release/reseed 和输出记录簿；尚未证明这些构成不可重建的核心算法。     |
| S7  | 当前 P0 是否能识别 R‑A 特有机制                                | `[推断]` 不能；现有门禁主要证明任务完成和反静默，不识别模型为何完成。                                 |
| S8  | 记录簿/latch 是否参与科学状态                                  | `[代码核验]` 设计一方面把 model-only latch 列为状态字段，另一方面要求记录簿不参与状态转移；语义尚未统一。      |
| S9  | UOT 是否必要                                            | `[未知]` 没有与 Hungarian+dustbin、partial matching 或独立 focal/hazard 目标的比较。 |
| S10 | 固定 K 的真实容量上界                                        | `[未知]` 缺少精确标注并发普查；连续占用不会自动消除一载体一事件时的并发上限。                             |

---

## C.2 任务重新审查

### 1. 当前任务究竟是什么？

`[代码核验][文献核验]` 正式任务是标准、完全监督的 **Online Temporal Action Localization/Detection**：

* 在决策时刻 (t_j)，模型只能使用视频前缀 (X_{\le t_j})、此前模型状态和此前已经提交的不可变记录；
* 当模型判断一个动作已经结束时，提交 `{start, end, class, score}`；
* 不得读取未来帧，不得在后续用 NMS、合并或修订改变已经正式发布的结果。

这与 CAG‑QIL 和 SimOn 对 On‑TAL 的定义一致：从流式视频中在动作完成后输出动作实例，不使用未来帧，也不回头修改过去结果。([CVF Open Access][1]) 固定仓库的 query pack 也采用同一任务口径。

但当前可执行 Q2 路径实际使用的是 **固定缓存特征**，不是从 causal RGB 到区间的原始视频联合训练。数据集模型输入明确标记为 `cached_features`。

一个精确的协议表达是：

[
(S_j, P_j, E_j)
===============

F_\theta(S_{j-1}, X_{(t_{j-1},t_j]}, L_{j-1}),
\qquad
L_j=L_{j-1}\Vert E_j ,
]

其中：

* (S_j) 是可继续修正的内部模型状态；
* (P_j) 是未正式提交的内部假设；
* (E_j) 是本时刻新增的正式记录；
* (L_j) 是 append-only 记录簿；
* 每个正式记录必须满足
  [
  \hat s\leq \hat e\leq t_{\text{emit}},
  \qquad
  t_{\text{source}}\leq t_{\text{emit}}.
  ]

仓库评测器和 ledger 验证器明确检查预测终点和 source frame 不晚于提交时间。

### 2. 模型在每个时刻能看到什么、不能看到什么？

| 信息                                               |  推理可见性 | 结论                                     |
| ------------------------------------------------ | -----: | -------------------------------------- |
| 当前及过去的缓存特征 token                                 |      是 | `[代码核验]` source frame 必须严格递增并早于视频末尾。   |
| 当前 source frame、feature stride、fps、缓存 provenance |      是 | `[代码核验]` 属于因果元数据。                      |
| 过去模型状态                                           |      是 | `[代码核验]` 在线 recurrence 所必需。            |
| 过去已经提交的记录                                        |  协议上允许 | `[推断]` 它们是过去输出，不是未来信息；但是否进入神经状态必须显式声明。 |
| 当前视频总长度、`is_video_end`、未来终点                      |      否 | `[代码核验]` 数据集禁止将这些字段放入 model meta。      |
| 当前或未来 GT instance ID、区间、类别                       |   推理时否 | `[代码核验]` prefix schedule 只在训练样本中附加。    |
| 未来缓存 token                                       |      否 | `[代码核验]` packet 必须按 source frame 顺序处理。 |
| 缓存编码器是否自身用了未来上下文                                 | `[未知]` | 外部缓存生成器的准确时域感受野不在固定提交中。                |

### 3. 内部假设何时可修正，正式输出何时不可修改？

`[推断]` 在记录正式 append 之前，模型可以因新观测修正内部 start posterior、类别证据、占用概率乃至载体排列。标准任务并不要求这些内部假设不可变。

`[代码核验]` 只有进入 ledger 的记录是正式输出，之后不得删除、替换、retract 或修改。ledger 模块使用 hash chain 并显式拒绝更新、替换和撤回语义。

因此，“载体身份必须从动作开始一直不可交换”不是由正式输出协议直接推出的要求。真正必须保持的是：**不得重复正式提交同一个事件，也不得在提交后修改区间。**

### 4. 当前性能低的主要原因是什么？

必须分两层回答：

* `[实验核验]` 对 Q2 readiness 而言，已确定的主要失败是 **训练监督资格与 runtime decoder 生命周期顺序不一致**。
* `[未知]` 对真实 On‑TAL mAP、召回、重复、碎片和延迟而言，固定提交没有一个合法的 R‑A 结果，也没有可供本轮重放的 Q2 正式效果包，因此无法断言主要性能瓶颈是什么。

Q2 训练循环在每个 token：

1. 先从旧 runtime hard status 读取 `available_slots`；
2. 再执行 head；
3. 用旧 availability 构造 GT canonical transition 和损失；
4. 最后才运行模型的 decode/release。

这一顺序由代码直接证明。

同时，canonical supervision 把未分配的 birth 也立即加入 `born_instance_ids`，记录为 exhausted，并在后续不再作为 birth 重试。

### 5. 重复、碎片、同类重叠、起点遗忘和延时是否真是主要误差？

`[代码核验]` 评测器确实能测量 duplicate、fragmentation、false emission 和 endpoint latency，并且不允许 NMS、合并、过滤或重新排序掩盖错误。

`[文献核验]` 这些是 On‑TAL 的真实问题：OAT 专门提出在线重复 proposal 抑制；MATR 用长期记忆从当前片段估计终点、从历史估计起点；ActionSwitch 专门处理并发和同类重叠。([ECVA][2])

但：

* `[未知]` 没有当前模型的完整 error decomposition；
* `[未知]` 没有证据证明上述错误占当前损失的大多数；
* `[推断]` “领域中存在这种错误”不能推出“持续载体是唯一或最简单的解决方案”。

### 6. 这些问题来自标准 On‑TAL，还是旧代码错误？

| 问题                                               | 归属                                              |
| ------------------------------------------------ | ----------------------------------------------- |
| 不能读未来、正式输出不能修订、终点检测延迟                            | `[文献核验]` 标准 On‑TAL 固有约束。                        |
| 同一动作被多次正式提交、窗口碎片化                                | `[推断]` 是标准实例级在线检测可能出现的通用错误。                     |
| 同类并发和重叠                                          | `[文献核验]` 是真实领域问题，ActionSwitch 已明确处理。([ECVA][3]) |
| 固定槽位容量上限、查询交换                                    | `[推断]` 是固定槽位/持续查询家族的通用风险。                       |
| 在 decode 前读取 runtime availability，再用它筛掉 GT birth | `[代码核验]` Q2 特有实现错误。                             |
| 未获槽的 GT birth 被永久标记 born/exhausted               | `[代码核验]` 当前 canonical supervision 特有错误。         |
| `-2` 静默策略通过容量门禁                                  | `[实验核验]` 当前 Q2 gate 设计错误。                       |
| 训练和推理使用不同生命周期                                    | `[推断]` 是序列模型的通用风险，但目前证实的具体实例仅属于 Q2。             |

### 7. 当前数据集是否真能验证“持续动作身份”？

`[文献核验]` THUMOS 提供 untrimmed 视频、动作类别和时间区间，一个视频可以包含零个、一个或多个动作实例；正式 temporal detection 覆盖 20 个类别。([arXiv][4])

但它能验证的主要是：

* 最终区间是否正确；
* 是否重复提交；
* 是否把一个 GT 碎片化；
* 是否遗漏或延迟。

它不能直接验证：

* 某个内部载体在所有 prefix 上是否保持相同“真实身份”；
* 两个内部查询交换后、但最终输出不变时谁是“正确身份”。

`[代码核验]` 当前 identity diagnostics 是对最终 emissions 和 GT 区间做后验匹配，而不是观察内部 query ID。

因此裁决是：

> `[推断]` THUMOS 可以部分验证“输出层实例一致性”，但不能单独证明“持续潜在载体身份”是必要的科学变量。

### 8. 如果同类重叠和连续动作很少，路线是否解决人为问题？

`[未知]` 精确项目 split 中这些事件的发生率未核验。固定提交引用的标注、160/40 split 和 feature manifest 在仓库外。

`[推断]` 若 outcome-blind census 表明：

* 同类重叠样本极少；
* same-bin end/birth 极少；
* 最大并发远低于 K；
* duplicate/fragmentation 也不是主要错误；

那么以这些现象为主张中心的 R‑A 将缺乏可测量的真实数据效应，应终止或转向经盲普查证明更适合的数据集。不能先设计 direct-complete、同类重叠或 same-bin 专用机制，再从少数样本中寻找合理性。

### 9. “训练和推理状态不一致”是领域普遍缺口，还是 Q2 局部错误？

`[推断]` 它是所有 stateful sequence model 都必须避免的普遍工程原则，但当前项目发现的 failure mode 是 Q2 局部的：

* TrackFormer 将静态 object queries 用于新轨迹、持续 track queries 用于已有轨迹；
* MOTR 以逐帧传播的 track queries 和 tracklet-aware assignment 建模出生与存续；
* 它们说明“持续状态加训练分配”本身并不必然导致 Q2 式双状态机。([arXiv][5])

所以可防御的结论只能是：

> `[代码核验]` Q2 的具体 train/runtime lifecycle 不同构。
> `[未知]` 标准 On‑TAL 是否普遍缺少一个必须由 R‑A 才能补上的前缀状态合同。

### 10. 是否存在更简单的标准训练方法？

`[文献核验][推断]` 至少存在四类必须先排除的简单路线：

1. 每个 prefix 独立产生已完成动作集合，训练时做普通 set matching；
2. 持续查询加区间、类别和完成头；
3. 一维 TrackFormer/MOTR 重建；
4. 新建一个干净 baseline，仅修复 Q2 的执行顺序和 risk set，不修改旧 Q2 文件。

TadTR 已证明 action query 可以直接做区间集合预测；TrackFormer/MOTR 已提供持续查询和出生/存续分配范式；OAT、MATR、SimOn 已在 On‑TAL 中实现较简单的实例输出和历史建模。([arXiv][6])

当前没有证据证明这些方法不能满足同一正式输出合同。

---

## C.3 问题定义分层

| 层级           | 经核验的问题                                                                                          |  是否支持 R‑A 必要性 |
| ------------ | ----------------------------------------------------------------------------------------------- | ------------: |
| Q2 特有状态机错误   | `[代码核验]` runtime availability 在 decode 前进入 GT birth 资格；canonical ownership 与 runtime status 分离。 |   否，只支持终止 Q2。 |
| 所有固定槽位方法风险   | `[推断]` 固定 K 容量、查询交换、空槽/占用定义、释放顺序。                                                               |  只支持做容量和交换对照。 |
| 所有在线动作实例方法风险 | `[文献核验]` 未来不可见、终点延迟、重复、碎片、重叠、不可回改。                                                              | 支持研究任务，不指定方法。 |
| 当前训练代码产生的问题  | `[代码核验]` first-birth 永久丢失、硬 runtime 状态与 loss state 不同构、静默容量门禁。                                  | 强烈支持重新设计训练协议。 |
| 数据/评测造成的问题   | `[代码核验][未知]` 评测只看最终 intervals；内部身份不可观测；精确 overlap/repetition 发生率未知。                             |     削弱持续身份主张。 |

### 最强反方论证

> Q2 的失败只说明旧实现错误地让预测 hard availability 控制 GT 所有权。新建一个普通的 prefix set-prediction 模型，或使用持续 queries，但令匹配仅产生损失而不写 runtime state，就可以恢复 train/inference 一致性。不可变记录簿只是一项输出协议。无须连续占用、UOT、transport consistency 或新的潜在事件过滤器。

### 对该论证的裁决

`[代码核验]` 该论证解释了 Q2 已观察到的全部致命证据：pre-decode availability、birth 永久 exhaustion 和静默通过。

`[未知]` 没有任何 matched baseline 证据表明上述简单重建仍会在相同条件下失败。

`[推断]` 因而该反方论证目前成立到足以阻止 R‑A 合同冻结。R‑A 现在更像是在绕开 Q2 制造的错误，而不是已经建立的领域必要机制。

---

## C.4 对 R‑A 的二十项敌对审查

下面的“当前证据”引用五个锚点：

* **E1 `[代码核验]`**：正式输出只要求不可变区间及 no-future，内部载体身份不进入主 AP 匹配。
* **E2 `[代码核验][实验核验]`**：Q2 特有顺序错误和静默失败已确定。
* **E3 `[代码核验]`**：R‑A 目前只有设计草图，精确方程未冻结。
* **E4 `[文献核验]`**：OAT、MATR、ActionSwitch、TrackFormer、MOTR、TadTR 已占据实例输出、记忆、重叠处理、持续查询和集合预测等核心部件。
* **E5 `[未知]`**：精确数据普查和匹配基线结果缺失。

|  # | 支持路线的论证                                                                        | 反对路线的论证                                                                                                                                                   | 当前证据与未知                      | 可证伪实验                                                      | 裁决                  |
| -: | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- | ---------------------------------------------------------- | ------------------- |
|  1 | `[推断]` 长动作需要保存起点、类别和进行状态。                                                      | `[推断]` 保存历史不等于必须保存一一对应的持续载体；memory queue 或 prefix set 也可完成。                                                                                               | E1/E4；E5。                    | 无身份完成集合 vs persistent query vs R‑A，完全匹配输入、头和记录簿。           | **必要性未通过。**         |
|  2 | `[推断]` 逐时刻集合可能重复发现同一事件。                                                        | `[文献核验]` OAT 用在线重复抑制，MATR直接预测实例；简单方法尚未失败。                                                                                                                 | E4。                          | 冻结无持续身份基线，直接测 duplicate、fragmentation、recall。              | **普通集合预测不够尚未证明。**   |
|  3 | `[推断]` 连续占用可微，避免硬 FREE/ACTIVE 阶跃。                                              | `[推断]` 连续值可能长期停在模糊中间态，并把容量冲突隐藏为质量分摊。                                                                                                                      | E3；无 occupation trace。       | 与 hard sigmoid、hazard-only、普通 query validity 头比较；报告熵和校准。   | **未通过。**            |
|  4 | `[推断]` 连续占用允许渐进启动和释放。                                                          | `[推断]` 它可能只是把硬错误变成 0.3/0.7 的模糊错误。                                                                                                                         | E3；E5。                       | 对 occupancy 施加小扰动并检查事件数、交换率、释放稳定性。                         | **风险成立。**           |
|  5 | `[推断]` target 数和 carrier mass 不等，UOT 的 dustbin 可表达缺失质量。                        | `[文献核验][推断]` UOT 一般适用于不等质量，但这不证明本任务需要它；Hungarian+dustbin、partial matching 或独立背景损失更简单。UOT 的一般性质仅说明它能处理质量变化。([Proceedings of Machine Learning Research][7]) | E3；没有比较。                     | UOT、Hungarian+dustbin、partial OT、独立损失四臂比较。                 | **UOT 必要性不通过。**     |
|  6 | `[代码核验]` transport plan 不直接写 runtime tensor。                                   | `[推断]` plan 通过梯度改变参数，进而改变未来载体行为；只能声称“没有样本路径直接写入”，不能声称“不影响身份”。                                                                                             | E3。                          | detach plan 与可微 plan、随机置换 plan、loss-off 状态一致性。             | **原主张需收窄。**         |
|  7 | `[推断]` 相邻 prefix consistency 可减少 plan 抖动。                                      | `[推断]` 它会引入训练专用的跨 prefix 身份约束；如果依赖 GT instance ID，就是训练身份监督。                                                                                               | E3。                          | GT ID 随机重命名、prefix-wise permutation、删除 consistency 三组。     | **隐藏训练身份风险成立。**     |
|  8 | `[推断]` recurrence 可自行学会载体连续性。                                                  | `[推断]` 无显式身份约束时 carriers 具有置换对称性，频繁交换不影响某些集合损失。                                                                                                           | E1/E4。                       | 载体 permutation intervention、swap metric、输出不变但内部交换测试。       | **身份稳定性未证明。**       |
|  9 | `[推断]` immutable lock 只是输出协议，不是生命周期。                                           | `[推断]` 一旦 lock 阻止再次提交或影响 reseed，它就是硬状态机；区别仅在是否参与神经转移。                                                                                                     | E1/E3。                       | 强制翻转 latch，验证 latent state、提交和风险集的数据流。                     | **需重新定义。**          |
| 10 | `[推断]` latent state 可隐式记住已提交事件。                                                | `[推断]` 若记录/latch 完全不进入状态，模型无法被协议性保证知道已经提交；只能依赖未验证的隐式记忆。                                                                                                   | E3。                          | 提交后清除或保留 latent state，测试 duplicate；ledger inaccessible 对照。 | **当前合同不闭合。**        |
| 11 | `[推断]` 过去记录是因果信息，进入模型输入不构成未来泄漏。                                                | `[推断]` 但这会否定“记录不影响状态转移”的宽泛表述。允许的是 GT-free，不是 ledger-free。                                                                                                 | E1/E3。                       | 明确比较 ledger summary 输入/不输入，并审计无 GT 字段。                     | **必须修改 claim。**     |
| 12 | `[推断]` release-before-reseed 能支持同 bin 结束和开始。                                   | `[推断]` 若 commit snapshot、清空和 reseed 不是单一原子顺序，可能重复提交或污染新事件起点/类别。                                                                                           | E3；方程未冻结。                    | 同一载体同 bin 完成+新生、多载体冲突、packet 切分测试。                         | **未闭合。**            |
| 13 | `[推断]` 连续质量可以柔化 K 的使用。                                                         | `[推断]` 若一个 carrier 仍代表一个事件，最大并发仍受 K 限制；若一个 carrier 表示混合事件，则失去实例语义。                                                                                        | E5。                          | 盲普查最大并发；合成中令并发为 (K+1)。                                     | **容量问题仍存在。**        |
| 14 | `[推断]` 极短动作可能在一个决策区间内起止，需要 direct-complete。                                    | `[推断]` 在没有标注普查前专门增加分支属于事后补丁风险。                                                                                                                            | E5；吸收文档也把真实数据启用推迟到 census 后。 | 先盲普查；若发生率不足，默认禁用并终止相关主张。                                   | **当前不得启用。**         |
| 15 | `[推断]` 多随机种子和未见验证集可缓解过拟合。                                                      | `[推断]` 若训练/验证共享同一模板、仅改变 seed，模型仍可学习长度、计数或阈值规则。                                                                                                            | 当前 P0 只有 family/seed 分离。     | 结构 OOD：未见事件数、重叠图、长度、间隔、类别置换、噪声和组合。                         | **现有 P0 不足。**       |
| 16 | `[推断]` 0.90/0.80 是严格质量下限。                                                      | `[推断]` 它们没有由噪声上限、简单基线或统计功效推导；只能作为工程门槛，不能证明机制。                                                                                                             | 无理论或 pilot 依据。               | 先冻结基线分布和测量误差，再定义相对优势与 CI。                                  | **科学依据不足。**         |
| 17 | `[推断]` R‑A 面向一维动作区间而非二维物体轨迹。                                                   | `[文献核验][推断]` 持续 query、新生、存续、消亡、loss-side assignment 与 TrackFormer/MOTR 高度同构；区间头和 ledger 是任务适配。([arXiv][5])                                                | 无 matched reconstruction。    | 参数、状态维度、训练预算匹配的一维 MOTR。                                    | **当前实质差异不通过。**      |
| 18 | `[推断]` start posterior、class evidence、completion 和 release/reseed 的组合可能形成新过滤器。 | `[推断]` 普通持续 query + start/end/class/validity heads + append-only ledger 已足以重建正式合同。                                                                        | E1/E4。                       | 最小重建禁止 UOT、continuous occupancy 和 consistency，只保留标准 heads。 | **高度可重建。**          |
| 19 | `[推断]` 即使重建合同，R‑A 也可能在 identity errors 上更好。                                    | `[推断]` 若参数匹配重建在所有主指标和生命周期指标等价，则 R‑A 只剩命名和复杂度。                                                                                                             | 无结果。                         | 预登记等价区间；落入区间即路线 KILL。                                      | **创新取决于尚未存在的差异结果。** |
| 20 | `[推断]` P0 可证明机制至少能学会生命周期。                                                      | `[推断]` 合成 P0 最多证明某个程序能拟合冻结生成器，不能预测 THUMOS 上持续身份优势。                                                                                                        | R‑A 文档也明确 P0 不证明效果或论文创新。     | 先做可辨识 P0；通过后仍只能申请盲、CPU、真实标注 G0。                            | **不能从 P0 推出真实优势。**  |

### 二十项汇总

* 路线支持项：任务需要因果历史、不可变提交和对重复/碎片/重叠的显式评估。
* 路线失败项：持续载体、连续占用、UOT、transport consistency 和 latch 的必要性均未建立。
* 最强否定：当前方法在结构上可以被重建为 **一维 TrackFormer/MOTR + interval/completion heads + append-only ledger**。

---

## C.5 强制比较更简单的替代方案

### 科学合同与行为

| 方法                              |   严格因果 |   train/infer 一致 |               GT 写入运行状态 |   静默通过风险 |            同类重复/重叠 | 不可变提交 |
| ------------------------------- | -----: | ---------------: | ----------------------: | -------: | -----------------: | ----: |
| 1. 无持续身份的逐 prefix 完成集合预测        |    可做到 |              可做到 |                     不需要 | 高，须反静默门禁 |         取决于集合容量和去重 |  容易附加 |
| 2. 普通持续查询 + 区间头                 |    可做到 |              可做到 |          不需要；匹配仅用于 loss |        中 | 可建模，但可能 query swap |  容易附加 |
| 3. 一维 TrackFormer/MOTR 重建       |    可做到 | 其原理即共享持续 queries |         不需要真实 ID 写入推理状态 |        中 |          属于其核心能力范围 |  容易附加 |
| 4. 新建干净的顺序/risk-set 修复 baseline |    可做到 |              可做到 | 必须禁止 canonical state 写入 |        中 |                 未知 |  容易附加 |
| 5. R‑A                          | 设计上可做到 |     设计上声称可做到，未实现 |                   声称不写入 |   有反静默门禁 |              目标是处理 |  核心组成 |

### 复杂度、先例与可发表差异

| 方法                  | 实现复杂度 | 训练成本 | 最强先例                          | 当前可发表差异                 | 最小证伪成本 |
| ------------------- | ----: | ---: | ----------------------------- | ----------------------- | -----: |
| 无身份完成集合             |     低 |    低 | SimOn、OAT、MATR、TadTR 式集合/区间预测 | 低；主要是 baseline          |     最低 |
| 持续 query + 区间头      |     中 |    中 | TrackFormer、MOTR、TadTR        | 很低，除非出现 On‑TAL 特有机制效应   |      低 |
| 一维 TrackFormer/MOTR |    中高 |    中 | TrackFormer、MOTR              | 属于最强 obviousness attack |      中 |
| 干净顺序/risk 修复        |     低 |    低 | 标准 set-matching 训练原则          | 基本无论文新颖性，但科学上必须         |     最低 |
| R‑A                 |    最高 |   最高 | 上述全部部件的组合                     | **未建立**                 |     最高 |

`[文献核验]` On‑TAL 已有从简单 Transformer、在线 anchor、长期 memory 到同类重叠处理的多种路线；2026 年新工作还进一步研究 zero-shot 和 point-supervised On‑TAL，但这些任务扩展同样不构成 R‑A 持续身份必要性的证据。([arXiv][8])

### 比较裁决

`[推断]` 当前必须优先采用方案 1、2、3、4 作为证伪基线。R‑A 不能因为设计文档较完整而自动成为主方法。

---

## C.6 当前零阶段实验是否具有辨识能力？

| 需要区分的解释     | 当前 P0 能否排除 | 原因                                                           |
| ----------- | ---------: | ------------------------------------------------------------ |
| 真正学会动作生命周期  |          否 | 高召回和正确计数不等于内部载体稳定。                                           |
| 记住合成模板      |          否 | seed 不同但生成规则相同，缺少结构 OOD。                                     |
| 只学会事件数量     |          否 | 当前没有 count-only baseline 或 count-preserving feature shuffle。 |
| 通过阈值技巧减少误报  |         部分 | 全背景 FP=0 和 precision 有约束，但保守模板策略仍可能通过。                       |
| 普通集合预测      |          否 | 没有该基线。                                                       |
| 持续 query 模型 |          否 | 没有普通 persistent-query baseline。                              |
| R‑A 特有机制    |          否 | 没有 UOT、occupancy、consistency、release/reseed 的必要性删除实验。        |
| 真实数据优势      |          否 | P0 是合成 CPU 探针。                                               |

### 当前八个 scripted case 的真实意义

`[推断]` 八类确定性案例、stepwise/packet 一致性、no-future 和不可变记录检查适合证明：

* 接口和顺序无错误；
* ledger 不可修改；
* same-bin 边界条件被程序正确执行；
* 静默控制不会被误判 PASS。

它们不证明：

* 网络学到了有泛化能力的动作身份；
* UOT 优于普通 matching；
* 持续身份优于 per-prefix set；
* R‑A 优于 temporal MOTR；
* THUMOS 上这些事件足够常见。

### 必要的简单基线

下一版路线审查必须先冻结以下五臂，而不是只冻结 R‑A：

* **B0：无持续身份的 prefix completion set**
* **B1：普通 persistent query + start/end/class/validity heads**
* **B2：一维 TrackFormer/MOTR reconstruction**
* **B3：新建的 Q2-order/risk 修复 baseline**，不得修改或重开旧 Q2
* **B4：R‑A**

所有臂必须共享：

* 相同因果输入；
* 相同类别、区间和 completion 头；
* 相同 ledger；
* 相同参数和更新预算匹配规则；
* 相同生成器和评测器；
* 相同 anti-silence 门禁。

### 必要负对照

1. `[推断]` count-only predictor：只看序列级或前缀统计，不建模身份。
2. `[推断]` template-timing predictor：根据训练分布长度/间隔输出。
3. `[推断]` feature-time shuffle：保持事件数和类别边际，破坏生命周期信号。
4. `[推断]` label-feature permutation：检验模型是否真正利用动作特征。
5. `[推断]` ledger-only scripted deduplicator：检验收益是否只来自记录簿。
6. `[推断]` history-off baseline：检验持续历史是否必要。
7. `[推断]` conservative threshold sweep：排除只靠压低提交率获得精确率。

### 必要机制删除实验

* UOT → Hungarian+dustbin；
* UOT → partial matching；
* 删除 continuous occupancy；
* 删除 adjacent-prefix consistency；
* 删除 release-before-reseed；
* latch 不进入神经转移；
* latch 进入神经转移但只含模型输出；
* 固定 K 与 (K+1) 并发；
* start posterior → 普通 scalar start regression；
* 禁用 direct-complete。

### 防止合成过拟合

训练和验证不能只更换随机种子。必须预先构造未见的结构组合：

* 训练未见的事件数量；
* 训练未见的动作长度；
* 训练未见的重复间隔；
* 训练未见的 overlap graph；
* 未见 same-bin end/start 组合；
* 类别重新排列；
* 特征基底旋转或混合；
* 不同噪声族；
* 延迟分布迁移；
* 多个困难因素同时出现。

验证生成器代码、分布、隐藏 seed 和内容哈希必须在模型实现前由独立过程冻结。

### 什么结果才足以让 R‑A 进入真实数据验证？

至少需要同时满足：

1. `[推断]` B0–B3 均能通过相同接口和 anti-silence 正确性门禁；否则无法确认 R‑A 的增益不是基础设施差异。
2. `[推断]` R‑A 在所有三种 seed 和所有结构 OOD family 上，相对最强简单基线产生同方向、非偶然的 lifecycle 优势。
3. `[推断]` 该优势必须发生在预声明的 duplicate、fragmentation、同类重叠 recall 或 same-bin correctness 上，而不是只改善训练 loss。
4. `[推断]` 删除 R‑A 声称的核心机制后，该优势必须消失或显著减弱。
5. `[推断]` 不得以 precision、背景误报、召回或提交延迟恶化换取“身份”改善。
6. `[推断]` temporal-MOTR 重建若与 R‑A 等价，则 R‑A 主路线立即终止。
7. `[推断]` 即使全部通过，也只能申请一个新的、盲的、CPU 真实标注机制审查；不能自动授权效果训练或 GPU。

现有 P0 没有这些比较，因此不具备路线级辨识能力。

---

## C.7 前置路线裁决

### 为什么不是 `PASS_ROUTE_TO_P0_CONTRACT_FREEZE`

以下七项中只有任务价值本身通过：

| 条件                       |      裁决 |
| ------------------------ | ------: |
| 任务价值成立                   |  **通过** |
| 数据足以验证持续身份问题             | **未通过** |
| 问题属于领域缺口而非 Q2 局部错误       | **未通过** |
| 简单基线已被排除                 | **未通过** |
| R‑A 方法必要性                | **未通过** |
| 与 TrackFormer/MOTR 的实质差异 | **未通过** |
| P0 辨识能力                  | **未通过** |

### 为什么本轮选择修改而不是永久终止整个项目

`[推断]` 标准 On‑TAL 任务有效，duplicate、fragmentation、overlap、start-memory 和 delay 也是真实研究问题。由于精确数据普查和 matched baselines 尚不存在，本轮无法严谨证明“任何持续载体路线永远无价值”。

但当前 **R‑A 不能被保留为默认主方法**。它必须降级为基线竞赛中的一个候选臂。

### 最小修改要求

下一轮路线审查前必须完成以下内容，且不得提前写 R‑A 模型代码：

1. outcome-blind、只读的精确标注普查；
2. 缓存编码器严格因果感受野证书；
3. 把科学问题重写为：
   **“持续载体是否相对无身份集合和一维 MOTR 带来不可重建的在线实例一致性收益？”**
4. 冻结 B0–B4 的比较合同；
5. 冻结结构 OOD 和负对照；
6. 给出 R‑A 相对 TrackFormer/MOTR 的 exact-delta 表；
7. 预声明 temporal-MOTR 等价即 KILL；
8. 重新进行路线级审查。

---

# D. 完整状态定义

**未执行。**

`[推断]` 前置裁决不是 `PASS_ROUTE_TO_P0_CONTRACT_FREEZE`，因此不得选择载体数、隐藏维度、记忆长度、occupancy 类型、latch 字段或 direct-complete 分支。

任何此时给出的唯一状态定义都会把尚未证明必要的方法选择伪装成“合同闭合”。

---

# E. 完整状态转移公式

**未执行。**

`[推断]` release、commit、reset、reseed 的精确顺序只有在确定 R‑A 而非简单 baseline 是应实现路线之后才可冻结。

---

# F. 临时匹配定义

**未执行。**

`[推断]` UOT 尚未通过相对 Hungarian+dustbin 和 partial matching 的必要性审查。不得冻结代价、dustbin、熵正则、质量放松、迭代次数或梯度策略。

---

# G. 训练目标

**未执行。**

`[推断]` first-birth、occupancy、survival、consistency 和 short-action loss 目前属于候选机制，不是已通过的合同组成。

---

# H. 提交规则

**未执行。**

`[推断]` append-only 输出协议可以保留为通用基础设施，但 R‑A 特有的 latch、release snapshot、reseed 和阈值不得冻结。

---

# I. 合成生成器

**未执行。**

`[推断]` 现有 512/256/64 和 9101/9102/9103 可以作为历史提案保留，但在加入基线辨识、结构 OOD 和负对照前，不能构成可执行的最终生成器合同。

---

# J. 训练流程

**未执行。**

`[推断]` 不冻结初始化、优化器、学习率、更新数、TBPTT、线程数或时间上限。

---

# K. 数值比较规则

**未执行。**

`[推断]` 只有通用的 no-future、ledger append-only 和 provenance 验证仍可作为基础设施要求；R‑A 状态、logit 和 mechanism 指标的容差未获授权。

---

# L. 零阶段门禁

**未冻结。**

`[推断]` 历史八类场景和反静默要求可保留为候选 correctness tests，但当前 learned gate 缺少 matched baselines、机制删除和结构 OOD，不能成为最终路线门禁。

---

# M. 证据格式

**未冻结。**

`[推断]` 原子发布、不可覆盖、哈希绑定和环境记录是合理通用要求，但不得为一个尚未通过路线审查的方法创建正式 evidence schema。

---

# N. 逐文件实现计划

**未执行。**

`[推断]` 本轮明确禁止编写逐文件计划。旧审查中列出的 `prefix_shared_event_filter_head.py` 等文件只是历史提案，不构成本轮授权。旧 Q2、R1、CRS‑EPS 文件保持只读。

---

# O. 审稿风险

| 风险                         | 严重性 | 本轮结论                                    |
| -------------------------- | --: | --------------------------------------- |
| 把 Q2 局部 bug 包装成领域缺口        |  极高 | `[代码核验][推断]` 当前正存在该风险。                  |
| 一维 TrackFormer/MOTR 可完整重建  |  极高 | `[文献核验][推断]` 尚无实质差异证据。                  |
| THUMOS 缺少足够同类重叠/并发样本       |   高 | `[未知]` 精确 split census 缺失。              |
| “identity”只由后验区间匹配定义       |   高 | `[代码核验]` 评测器不观察内部载体身份。                  |
| continuous occupancy 模糊化错误 |   高 | `[推断]` 无 entropy、calibration 或 swap 证据。 |
| UOT 增加复杂度但没有因果贡献           |   高 | `[推断]` 没有简单 matching 对照。                |
| ledger/latch 重新引入硬状态机      |   高 | `[推断]` 当前语义矛盾。                          |
| synthetic P0 模板过拟合         |   高 | `[推断]` seed 分离不足以防结构过拟合。                |
| direct-complete 成为事后补丁     |  中高 | `[未知]` 尚无真实发生率。                         |
| 缓存特征含未来上下文                 |   高 | `[未知]` 编码器时域证书缺失。                       |
| 绝对 0.90/0.80 门槛被误解释为机制证明   |   高 | `[推断]` 它们只是质量下限。                        |
| 失败后修改生成器或 margin           |  极高 | `[推断]` 下一版必须以预登记和独立隐藏 OOD 防止。           |

---

# P. 最终裁决

## `REVISE_ROUTE_AND_REVIEW_AGAIN`

核心裁决如下：

1. **任务价值通过。** 标准 On‑TAL 的因果、完成触发、不可变输出问题成立。
2. **R‑A 的问题必要性未通过。** 当前证据只证明 Q2 的局部双生命周期错误。
3. **简单基线不足未证明。**
4. **与 TrackFormer/MOTR 的实质差异未证明。**
5. **当前 P0 不具有路线辨识能力。**
6. **R‑A 不得作为默认主方法进入合同冻结。**
7. **R‑A 只能降级为未来 B0–B4 比较中的一个候选臂。**
8. **本轮零阶段合同未冻结，不允许代码实现。**
9. **真实数据、GPU、正式训练和原始视频训练全部继续禁止。**
10. **GPU 授权保持 0 小时。**

---

# Q. 机器可读决定

```json
{
  "任务价值通过": true,
  "问题属于领域缺口": false,
  "问题不是旧代码局部错误": false,
  "简单基线不足": false,
  "路线必要性通过": false,
  "与跟踪方法存在实质差异": false,
  "零阶段具有辨识能力": false,
  "路线前置裁决": "修改",
  "零阶段合同已冻结": false,
  "允许开始代码实现": false,
  "允许真实数据验证": false,
  "允许图形处理器测试": false,
  "图形处理器授权小时": 0,
  "主要终止理由": [
    "当前证据只证明Q2中运行可用性先于监督分配的局部实现错误，没有证明持续潜在事件载体是领域必要机制。",
    "正式评测合同只要求因果、不可变的动作区间提交，不要求可观测的持续载体身份。",
    "精确THUMOS分割中的同类重复、同类重叠、同一决策区间结束与启动以及最大并发发生率尚未核验。",
    "无持续身份集合预测、普通持续查询、干净顺序修复和一维TrackFormer或MOTR重建尚未被排除。",
    "连续占用、非平衡最优传输、相邻前缀一致性和记录锁的必要性均未建立。",
    "当前合成零阶段不能排除模板记忆、事件计数、阈值策略、普通集合预测或一般持续查询等更简单解释。",
    "R-A与一维TrackFormer或MOTR加动作区间头和不可变记录簿之间尚无经验证的实质算法差异。"
  ],
  "仍未解决的问题": [
    "固定项目分割的只读标注普查及其内容哈希",
    "缓存特征编码器的准确代码、权重、clip构造和严格因果时域感受野证书",
    "无持续身份完成集合基线",
    "普通持续查询加区间头基线",
    "参数和预算匹配的一维TrackFormer或MOTR重建",
    "不修改旧Q2文件的干净训练顺序和风险集合基线",
    "记录簿、提交锁和神经状态之间唯一且无矛盾的数据流合同",
    "非平衡最优传输相对Hungarian加空项和partial matching的必要性",
    "载体交换、占用模糊度和提交后重复风险的可观测指标",
    "能够排除模板、计数和阈值策略的结构分布外合成验证",
    "R-A相对最强简单基线的预登记非等价标准",
    "与TrackFormer或MOTR的可发表exact-delta"
  ]
}
```

[1]: https://openaccess.thecvf.com/content/ICCV2021/html/Kang_CAG-QIL_Context-Aware_Actionness_Grouping_via_Q_Imitation_Learning_for_Online_ICCV_2021_paper.html "ICCV 2021 Open Access Repository"
[2]: https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2307_ECCV_2022_paper.php?utm_source=chatgpt.com "ECVA | European Computer Vision Association"
[3]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/1621_ECCV_2024_paper.php?utm_source=chatgpt.com "ECVA | European Computer Vision Association"
[4]: https://arxiv.org/abs/1604.06182 "The THUMOS Challenge on Action Recognition for Videos \"in the Wild\""
[5]: https://arxiv.org/abs/2101.02702?utm_source=chatgpt.com "TrackFormer: Multi-Object Tracking with Transformers"
[6]: https://arxiv.org/abs/2106.10271 "[2106.10271] End-to-end Temporal Action Detection with Transformer"
[7]: https://proceedings.mlr.press/v151/sejourne22a.html?utm_source=chatgpt.com "Faster Unbalanced Optimal Transport: Translation invariant Sinkhorn and 1-D Frank-Wolfe"
[8]: https://arxiv.org/abs/2211.04905 "SimOn: A Simple Framework for Online Temporal Action Localization"
