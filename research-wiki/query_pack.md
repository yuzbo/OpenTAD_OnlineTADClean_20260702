---
type: query_pack
updated: 2026-07-22
status: active
scope: Current compressed context for the strictly causal On-TAD task.
---

# Query Pack: Strictly Causal On-TAD

## 最终任务

研究标准、closed-set、全监督、严格因果的 Online Temporal Action Localization。
最终模型直接读取原始 RGB；时刻 `t` 只能使用来源时间不晚于 `t` 的帧和内部状态。
模型在开始证据出现后创建并维护动作实例，在检测到结束后低延时一次性提交不可修改的
`{start,end,class,score}` 正长度区间。

`event_id`、provisional start/class、cancel、late birth 和 owner trajectory 是内部状态
及附加诊断，不是标准 On-TAD 基准字段。结束边界必须从已见证据独立解码，满足
`start < end <= source_time <= emit_time`，不能令 `end=emit_time`。OnVLLM 是独立后续
课题，不进入当前主论文。

feature-level 实验只用于快速机制归因，不能支持 raw-RGB claim。推理不得读取未来帧、
GT identity/annotation、terminal 信息、全视频回看、offline NMS 或事后修改输出。

## 已冻结历史证据 M53

旧 H2 FIXED/REMATCH 是 frozen SigLIP2 stride-8、768 维特征上的槽位负基线。源训练
FIXED `1179373`、REMATCH `1179374` 各完成 `12×2010=24,120` 更新；真实
4 占用+2 birth reserve、监督、因果、正长度、不可变和 recovery 门均通过。序列化
tie-break 修复 exact `0daf681b4e8e36a64066a2f6fa8e0458a98b429a`；checkpoint-only
重放 `1179456/1179457` 与终检 `1179458` 均 `COMPLETED 0:0`，8 份事件等价收据
证明只改同一 emit frame 内排序。累计双臂资源 `12.319444 GPU·h`。

预注册正式门使用 epoch 12、固定 `0.5`：

- FIXED：mAP=`1.614484`pp，Recall@0.3=`0.154167`，prediction/GT=`18.841667`；
- REMATCH：mAP=`1.178473`pp，Recall@0.3=`0.0875`，prediction/GT=`16.097917`。

技术链路 PASS，性能门因严重过量发射与低 Recall FAIL。FIXED 的单 seed 差值只作
方向信号。旧路线不再原样重跑；超过 1.614 mAP 也不足以证明新模型或创新。

## M54 创新边界

截至 2026-07-22，没有发现同时覆盖 original RGB、标准全监督 strict-causal On-TAD、
开始出生、持久实例身份、动态有效活动数、owner-conditioned end、同类重叠、学习式
历史压缩和不可变区间的单篇 exact match；但 ActionSwitch + MOTR/TrackFormer +
HEM/Backtrace Mamba + E2E-LOAD/StreamFormer 可以显然重构高层方案。

因此不能把以下内容单独列作贡献：即时开始、重叠动作、persistent query 概念、一般
长历史/层级记忆、active/visual memory 分离、strict causality、raw input、OpenTAD
接口、阈值校准或 Pareto 报告。最强待证问题是：在只有时间段监督、没有空间轨迹几何的
On-TAD 中，birth-time owner learning 是否能在同类重叠和交叉结束下减少 wrong-start、
owner swap、fragmentation 和 duplicate，而不增加提交延时与资源。

## M55 独立深度审判吸收

审判原文 SHA-256：
`ACA5BB6E9950993F170922250F6C915D41AF72BE027984406714315A92D96B8D`。
回答包含要求的 A--L 全部章节，任务/竞品/claim/接口/融合/状态机/raw/DAG/论文主题
覆盖近完整；但审判者没有读到目标分支及九项目标材料，所以结论是
`FORMALLY COMPLETE / EVIDENCE-INCOMPLETE / PARTIALLY ACCEPTED`，不能当当前代码审计。

完全采纳：标准输出不含 event ID；provisional 只作内部/诊断；OnVLLM 分离；donor
只读 native-first；ABCD 删除；`0.5` 只作控制；feature 不支持 raw claim；显式统计
late birth、orphan end、cancel、overflow、owner swap；active event 与 visual history
分账但总资源都计入。

不直接采纳：公开快照作为唯一最高科学优先级、四 donor parity 全部阻塞核心模型、
`end=t`、`K=32`、`B=128`、固定 loss 权重、30k updates、seed 3407、任意效应阈值、
单个未收敛 seed 直接杀死路线、前五天主要做接口工程、start-owned state 本身已经足够
新颖。E2E-LOAD/MViTv2-S 是 raw 候选，不是冻结唯一主干。

审判还留下两个算法缺口：其 late-birth 状态与“只匹配当前 start GT”的训练规则互相
冲突；同类、同前缀、无额外 identity 标注的交换等价实例在部分时刻本来不可辨识，需
permutation-aware matching/metric，而不能承诺绝对保证。

完整逐条处置：
`PRO_RAW_RGB_DYNAMIC_EVENT_MEMORY_REVIEW_ABSORPTION_20260722.md`。

## 当前执行路线 DR-030

审判前 A/B/C/D、AB/AD/BD/BC/ABD/ABCD 组合图保留为历史设计，不再授权执行。
当前使用同一 feature 代码路径做机制 `K×O`：

```text
K0 = fixed preallocated query/switch bank
K1 = birth-allocated packed event set，active cardinality 随样本变化
O0 = per-prefix rematching/reassignment
O1 = birth-time sticky owner until cancel/end

K0O0 = 固定 bank + rematch
K1O0 = birth-allocated + rematch
K0O1 = 固定 bank + sticky owner（Temporal TrackFormer 控制）
K1O1 = birth-allocated + sticky owner（候选模型）
```

四臂共享同一物理安全 guard、encoder、decoder depth、参数/数据暴露、成功更新数、
calibration 与 evaluator。`K1O1` 必须同时超过 `K0O1` 和 `K1O0`，并在 wrong-start、
owner-swap、同类重叠/交叉结束上形成对应增益；只提高普通 mAP 不够。

物理 guard 必须存在并 fail closed，但其值由并发分布和硬件压力测试决定，不是 semantic
slot。学习目标是有效活动事件数。late birth 需独立 prefix-visible training；cancel 后
是否 re-birth 必须冻结。所有数值训练预算、loss 权重、memory token 数和效应 gate 先
画像再注册。

## 下一步

1. 数小时内并行冻结最小本地/公开 snapshot 与直接 donor SHA，不做大接口工程；
2. 同时实现 `K×O`、Temporal TrackFormer 控制和 synthetic overlap/late-birth/
   cancel-rebirth/future-perturbation tests；
3. 同时允许轻量 raw prefix/gradient/cache/latency smoke，但不启动 formal raw training；
4. 画像父模型收敛、显存、吞吐和方差后，冻结更新数、guard 与 go/kill gate，四臂并行；
5. 核心通过后并行运行 fixed-budget `H×R` memory 和 raw frozen/adapter/joint；
6. 最后才做多种子、一个 annotation-audited 外部集和 report-once。

恢复入口：分支 `codex/ontad-rgb-event-memory`；工作区
`E:/DeskTop/TAD/OpenTAD_OnlineTADClean_20260702/_codex_worktrees/ontad-rgb-event-memory-clean`；
主决策 `research-wiki/decision_register.md` / DR-030；实验图
`research-wiki/experiments/ontad-rgb-dynamic-event-memory-design-20260722.md`。
