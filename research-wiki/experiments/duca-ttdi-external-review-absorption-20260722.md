---
type: external-review-archive
node_id: exp:duca-ttdi-external-review-absorption-20260722
title: "External DUCA-TTDI Attachment Archive: Verification and Scope Boundary"
status: archived-nonexecuting
outcome: partial-accept-for-reviewed-duca-only
updated: 2026-07-22
---

# DUCA-TTDI 外部审查：完整吸收、独立复核与路线边界

> **非当前实验。** DUCA 出现在这里，唯一原因是用户本次提供的附件本身审查
> `OpenTAD_C3_CoarseClean_20260702` 的 DUCA 离线取帧路线。本页只保存和复核该附件，
> 不把 DUCA 代码、模块、训练设置或实验作业带入当前 EventMATR 课题。

## 一句话裁决

**不完全认可。** 这份审查对 DUCA 离线全窗口稀疏采样器的代码重建和主要结构性风险判断总体可靠，尤其是“不规则真实时间被重解释为等间隔 selected rank”这一诊断值得优先证伪；但其 TTDI 主推荐、两阶段训练处方、数值门槛和“下一版唯一修改”仍是有根据的研究假设，不是已经成立的实验结论。它不能改变当前严格因果 EventMATR 的官方 MATR 母体、任务定义或 100-epoch 主协议。

## 来源与可见性

- 用户附件：`C:/Users/skywalker/.codex/attachments/e0863150-3e4c-4c29-907d-805f73b40eeb/pasted-text.txt`。
- 字节级 SHA-256：`36523B2F1A7456F8D4A4314EA445971F8066EEC59611F9632D7BC1D33E31A884`。
- 原件大小：`76,688` bytes，`1,705` logical lines；本页以逐节处置、哈希和代码复核保存其完整语义，不在当前 CausalTAD 库复制另一项目的 76 KB 原文。
- 附件审查对象：`yuzbo/OpenTAD_C3_CoarseClean_20260702`。
- 分支：`codex/duca-boundary-burst-20260722`。
- 精确提交：`a00498e15d69294f78d0abeadfb47bc456db0b0e`。
- 审查自报状态：分支与精确提交一致，ahead/behind=`0/0`。
- 本次独立复核使用 GitHub 原始代码读取固定提交，不修改被审查仓库。

附件的 `VISIBILITY_CERTIFICATE` 列出 selector、DUCA acquisition/structured-selection/transition/alignment、true-time geometry、ASFormer probe、VideoMAE adapter、ActionFormer/TemporalMaxer detector/head/projection、优化器、15 个配置、训练/评测/成本工具和 11 类 focused tests。它同时明确没有读取运行时外部 ASFormer 字节、THUMOS14 数据、预训练权重、生成配置、checkpoint、预测、Slurm 日志和 CUDA 运行环境。因此所有 terminal mAP、真实梯度范数、运行成本和硬件行为仍属于未知项，不能由静态审查代替。

## 审查覆盖的完整问题清单

本记录吸收原文全部十四个主节，而不是只摘取 TTDI 建议：

1. 可见性证书与未读依赖；
2. 代码正确性、模型机理、实验有效性、论文可发表性的 `HOLD_FIX_REQUIRED` 总裁决；
3. dense low-resolution evidence、ASFormer、transition/burst、mandatory union、exact-K/max-hole DP、RGB gather、VideoMAE、TAD head 和 TrueTimeMap 的实际前向图；
4. actionness、transition/burst、detector surrogate、hard DP 及各参数组的损失与梯度归属；
5. 一个 Critical、五个 High、三个 Medium 和一个 Low 发现；
6. 对原始设计意图、R2Q3/R4Q5、exact-K 可行域、max-hole 单位及三个反例的核对；
7. learned sampler 落后 uniform 的十二项根因排序和四分支根因树；
8. group-completable decoder、legal-swap ranking、TTDI、coverage-skeleton 四个模型选项；
9. `DUCA-TTDI` 推荐组合与模块冻结建议；
10. Stage A 2000 updates、Stage B 6000 updates、optimizer、curriculum、swap schedule 和停止条件；
11. cost parser、source equivalence、mandatory completion、TTDI、physical head、swap loss 和模型级测试 patch 计划；
12. 当前 60-cell、U/L/T 决定性试验、扩展门、P0/P1/P2 和 KILL 条件；
13. 当前可写/不可写主张、可防守 novelty、五类审稿攻击和三种结果场景；
14. 十项执行优先级与三个最终问题的终裁。

## 独立代码复核

### 已直接确认的代码事实

| 审查判断 | 独立复核 | 证据与解释 |
|---|---|---|
| 当前路线是离线 full-window selector，不是严格在线 On-TAD | **确认** | selector 接收完整 dense window；审查对象和当前 EventMATR 不是同一任务执行语义。历史类名带 `Online` 不能改变这一事实。 |
| detector 训练轴冻结为 `selected_axis_index` | **确认** | `duca_selected_axis_training.py` 要求 `detector_output_coordinate_space == "selected_axis_index"` 且要求 GT remap。 |
| 真实 selected positions 主要用于最终逆映射 | **确认** | `TrueTimeMap` 用分段线性函数在 `selected_axis_index` 与 `true_time_dense_index` 间映射；输出后映射不能逆转此前 backbone/projection/head 的表示运算。 |
| learned irregular frames 进入 rank-grid VideoMAE/projection | **确认** | selector 做 original-time hard gather；ActionFormer projection 使用长度索引的 sinusoidal positional embedding，没有读取 original timestamps。VideoMAE 路径按 packed temporal rank/tubelet 处理。 |
| 当前局部 detector bridge 是 RGB 左右斜率的 ST 梯度 | **确认** | selector 以 hard position 左右帧差、expected-position displacement 构造 backward path；这证明梯度存在，不证明其等于合法离散换帧的 detector utility。 |
| mandatory builder 最终返回 union mask | **确认** | `build_mandatory_bilateral_set` 构造并返回 `mandatory_mask`、retained center 和 group count；进入全局选择时 group identity 不再是显式约束对象。 |
| required mask 只在全局 DP 终点暴露不可补全 | **基本确认** | 当前接口会 fail closed，但缺少在逐组接纳前对 exact-K/max-hole completeability 的显式审计；低 K/紧 G 下可导致整次前向失败。 |
| profiler 只支持 K384/K256 | **确认** | `R5_MAX_UNSELECTED_HOLES={384:2,256:3}`，方法名解析仅接受 `384|256`；K320/K192/K128 的正式成本 cell 会被拒绝。 |
| aggregate 未验证模型语义等价 | **确认** | 聚合键只有 backend/arm/K/seed；虽然记录 `git_commit`，但没有 selector/backend/config semantic hash 或 source-equivalence receipt。 |
| physical-grid 基础设施已经存在但本轮 selected-axis 训练未启用 | **确认** | `AnchorFreeHead` 已有可选 `physical_grid_actionformer` 与 dense-axis GT 合同；当前 selected-axis 训练脚本仍强制 selected-axis GT remap。 |

### 代码支持但尚不能称为事实的推断

1. **selected-axis 时间扭曲很可能是主要瓶颈。** 这是强推断：learned policy 越形成微簇，rank gap 与真实 gap 的差异越大；uniform 则近似等间隔。但没有 U/L/T 控制实验前，不能断言它一定排第一。
2. **高 tIoU 会比低 tIoU 更受伤。** 时距错配影响 tubelet、注意力、卷积感受野和距离回归，方向合理；实际幅度仍须看 `mAP@0.6/0.7`。
3. **mandatory union 会伤害边界公平性。** group identity 的确消失，但在真实数据中造成多少配额偏差尚未测量。
4. **local RGB slope 可能与 legal swap 反向。** 机制上可能，必须用真实 hard-swap 的符号一致率和排序相关验证。
5. **TTDI 是最高信息增益的下一结构变量。** 我同意它是优先候选，不同意把它描述成已证明的唯一答案。

### 仍未被复核或需要运行产物的内容

- 当前 60-cell terminal epoch-59 EMA mAP、三种子方差和后端差异；
- actionness AP/AUC、transition center error、bilateral coverage 和 per-boundary quota；
- hard-swap utility 的符号一致率、Spearman 和置信区间；
- TTDI 的实际 mAP、高 tIoU、参数/FLOPs/显存/延时；
- 前端、DP、materialization、transfer 和 VideoMAE 的同会话完整成本；
- external ASFormer checkout、checkpoint 和运行环境与自报提交的逐字一致性。

## 对原审查建议的采纳矩阵

### 完全采纳：作为 DUCA 的事实修复或实验底线

1. **任务表述纠正。** 当前 DUCA 只能称为离线、全窗口、pre-backbone raw-RGB sampler；不得借类名宣称严格在线。
2. **selected-rank/true-time 风险必须显式暴露。** 非均匀采样不能在 detector 内被无说明地当作等物理时间。
3. **修复五预算 parser。** profiler 应从唯一预算表导入 K/G，五档均有参数化测试。
4. **加入 source-equivalence receipt。** 跨提交合并时必须核验 model/selector/backend/config 语义哈希，不能只记录 commit。
5. **mandatory completeability fail-closed。** 每次接纳 group 前检查是否仍存在 exact-K/max-hole completion，并记录拒绝原因。
6. **不要把非零 surrogate gradient 当作 utility 对齐。** 必须用合法 hard swap 直接测其方向。
7. **matched uniform、终点 checkpoint、多种子、高 tIoU、真实 full-stack 成本是论文门。** 不得用 R0、选帧图或中间最优 epoch 替代。
8. **谨慎写官方继承。** 应写“official-derived components + extended wrapper”，不能写整个 detector 源码逐字不变。
9. **现有运行只回答当前 selected-rank 模型。** 不应取消或改写；其结果既可支持也可否定当前模型，但不能自动验证 TTDI。

### 条件采纳：有价值，但必须由单变量实验决定

1. **TTDI。** 作为 U=uniform、L=current learned、T=learned+TTDI 的首个决定性结构试验；零初始化 identity 和 timestamp sensitivity 测试必须先过。它是研究假设，不是保证增益的修复。
2. **physical-coordinate head。** dense-time assignment/regression 比事后逆映射更合理，但必须同时支持 ActionFormer/TemporalMaxer，并与 uniform timestamp 路径做数值等价门。
3. **group-completable decoder。** correctness 修复应做；“每组全接纳或全拒绝”会减少预算弹性，需与 union baseline 做 allocator 消融。
4. **sparse legal-swap ranking。** 适合作为诊断或后续 loss；detector train loss 不等于 mAP，而且候选覆盖局部，不能直接升级为主机制。
5. **只改一个主要结构变量。** 对因果归因合理；但 P0 correctness 修复不应被错误算作一个模型变量。
6. **不先更换 coarse ASFormer。** 当前无证据指向 coarse backbone 是首因；如果 actionness/center 诊断失败，仍允许回到 evidence branch。

### 不原样采纳：缺少证据或超出当前路线

1. **不冻结精确 2000/6000 updates。** 这些数字、分段边界和 EMA terminal 是建议，不是由学习曲线或预算画像推导出的事实。
2. **不冻结给定 LR、WD、loss weights、温度和 mixture schedule。** 它们可以成为起始配置，但必须先与现有官方训练预算和曲线匹配。
3. **不冻结 `55%` sign agreement、Spearman `>0`、`0.2` mAP 容差、`<1%` cost 等阈值。** 这些数值没有置信度/功效分析，只能预注册为 pilot 候选。
4. **不接受“TTDI 完全解决时间扭曲”。** 它位于 VideoMAE 之后，不能恢复 tubelet 形成时已经发生的 rank-time 混合。失败后才考虑 patch-token timestamp injection，不能与 TTDI 首试捆绑。
5. **不把 DUCA-TTDI 直接移植进 EventMATR feature 阶段。** 当前 MATR 使用官方均匀 RGB+flow feature 时间轴，不存在 learned nonuniform selected-rank 问题。
6. **不把 DUCA 结果当严格因果 On-TAD 或 raw-RGB EventMATR 证据。** DUCA 使用完整窗口；EventMATR 在时刻 `t` 只能访问来源时间不晚于 `t` 的帧和状态。
7. **不因该审查更换 EventMATR 母体或训练协议。** MATR `ba05a98...`、官方 100 epochs、native parity 和 eventized `B×O` 因果试验保持不变。

## 对 DUCA 路线的最佳执行解释

若以后继续 DUCA，应采用最小、可证伪的顺序，而不是一次性实现整份处方：

1. 保全并终读现有 60-cell terminal 结果；不以中间 epoch 挑结果。
2. 立即修复 parser、source-equivalence 和 mandatory-completion 三个 correctness/validity 问题；它们不应改变已完成 mAP cell 的模型语义。
3. 先判断 hard selection quality 是否真的优于 uniform。如果选帧本身没有改善，TTDI 不是首要答案。
4. 仅在“selection 更好但 official mAP 不更好”时，运行 K384/ActionFormer/固定 seed 的 U/L/T 单变量 TTDI 试验。
5. 只有 T 在 Avg-mAP 和高 tIoU 上稳定优于 matched uniform，才扩到三种子、低 K、第二后端和 full-stack cost。
6. legal-swap 先做对齐测量；方向性不过门就关闭 bridge，不通过调低阈值掩盖。
7. 若 TTDI 失败，区分 VideoMAE 前 tubelet 时间扭曲、上下文被微簇挤占和 allocator/utility 问题，再决定 patch-token time、allocator 或 KILL。

这保留了原审查的高信息增益思想，但不预先承诺其任意训练数字或最终论文故事。

## 对当前 EventMATR 路线的影响与不影响

### 不改变

- 最终目标仍是原始 RGB 输入、标准 closed-set、全监督、严格因果 On-TAD；
- 唯一父模型仍是官方 MATR，不是 DUCA/OpenTAD；
- ActionSwitch 仍只提供即时 start-state transition；
- 当前 feature stage 仍只验证“即时 birth × sticky owner”机制；
- native MATR parity、官方 100-epoch 设置、locked test 和因果/不可变输出合同不变；
- DUCA 的 60-cell、K、R2Q3、VideoMAE 和 TTDI 不进入 EventMATR 第一阶段。

### 形成一个未来 raw-RGB 设计护栏

如果 EventMATR 的 raw-RGB 阶段以后引入**非均匀采样、token 删除、帧压缩或可学习时间步长**，则必须把原始 timestamp/physical time 一直传入 visual backbone 与 localization head；不能只在输出端把 rank 坐标映回真实时间。该原则来自本次 DUCA 复核，但它只是未来接口约束，不是当前 EventMATR 的新增模块或贡献主张。

## 结论

这份审查回答 DUCA 问题相当完整：代码图、损失所有权、失败根因、修补方案、实验门和论文边界均覆盖；它最大的不足不是遗漏主题，而是把若干合理提案写得过于接近冻结处方。最终吸收比例可概括为：

- **代码事实和实验完整性建议：高认可；**
- **selected-axis 为首要根因：强烈重视但待 U/L/T 证伪；**
- **TTDI 结构：条件认可；**
- **训练数字与数值门槛：不认可为已定规则；**
- **迁移到当前 EventMATR：仅吸收未来 true-time 护栏，当前路线不变。**
