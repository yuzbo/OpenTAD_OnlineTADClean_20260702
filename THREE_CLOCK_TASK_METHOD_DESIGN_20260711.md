# PIVOT: 三时钟可观测性分解的完整任务与方法设计

日期：2026-07-11
状态：条件候选，尚未进入正式实现或大规模训练
仓库锚点：<https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/tree/codex/online-tad-clean-20260702>
公开 HEAD：`bfd0608b2996cba30d158d741ee193476a5078df`

## 1. 当前裁决

原始名称“Three-Clock Event Observability”过宽。最新查新后，能够保留的精确方向是：

> **PIVOT: Physically Anchored, Interval-Valued Visual Observability Timing for Streaming Event Verification.**

建议论文标题：

> **Not All Delay Is Model Delay: Separating Physical Change, Visual Evidence, and Decision Time in Streaming Video**

中文命题：

> 在线事件系统从物理变化到输出决定的总延迟中，只有视觉证据已经充分出现之后的部分才应归因于模型。我们用独立物理传感器锚定真实转移，以人群和视角条件的区间描述视觉可观测时刻，再评价模型的剩余决策延迟。

这不是“再做一个 Online TAD head”。核心是一项任务与评价协议；方法只负责证明该任务可以被学习。

## 2. Problem Anchor

### 2.1 底线问题

现有在线时序定位通常直接计算：

```text
reported latency = model commit time - annotated action endpoint
```

该计算隐含地把人工动作边界同时视为：

1. 物理状态真正发生变化的时刻；
2. RGB 视频中首次出现足够证据的时刻；
3. 模型理应能够作出可靠决定的时刻。

这三个时刻可能不同。遮挡、视角、传感器帧率、动作后果延迟显现和物体稳定过程都会产生无法由算法消除的视觉可观测延迟。将其全部归咎于模型，会错误评价算法速度，甚至改变方法排名。

### 2.2 必须解决的瓶颈

必须建立一个可重复测量的分解：

```text
physical transition -> visual verifiability -> model commit
```

并证明这种分解不是概念装饰，而会：

- 暴露传统 endpoint latency 看不到的失败；
- 改变至少部分模型的相对结论；
- 支持一个低成本、严格因果的学习方法。

### 2.3 非目标

- 不追求 THUMOS14 上再提高少量 mAP。
- 不把三时钟包装成通用 Online TAD 首创。
- 不把力/触觉蒸馏、区间删失损失或 survival model 单独当作创新。
- 不做 generic streaming VLM memory/readiness。
- 不在第一篇工作中同时解决开放词表、多类长流、在线学习和主动感知。
- 不宣称任意时刻的形式化误报保证；T01 风险控制路线与本任务分开。
- 不允许模型或标注者利用未来帧。

## 3. 任务归属

### 3.1 它不应被继续称为普通 On-TAL

PIVOT 的核心任务是：

> **Physically Anchored Streaming Event Verification**，即物理锚定的流式事件确认。

它比 Temporal Action Localization 更窄，也更物理可证伪：给定一个事件/状态转移定义，系统需要在线判断该转移是否已经真实发生，并在视觉证据充分后尽快确认。

动作类别和历史 span 可以作为上下文或次输出，但不应支配第一篇论文。若强行要求完整 `[start, end, class]` 为主输出，论文会重新滑回拥挤的 On-TAL。

### 3.2 第一篇论文的核心设置

- closed-set、已知事件查询；
- 物理可验证的瞬时或短时状态转移；
- 单次事件验证，重复实例作为压力测试；
- 训练/标注时可使用 force/contact/pressure/switch state；
- 测试时主设置仅使用单视角 RGB 前缀；
- 多视角仅用于验证可观测性依赖视角，不作为部署必需输入。

### 3.3 适合的事件类型

首轮应覆盖至少 3 类不同可观测结构：

1. **接触事件**：首次触碰、接触解除；
2. **离散机构事件**：按钮触发、开关闭合、卡扣锁定；
3. **结果状态事件**：物体稳定放置、容器真正闭合、抓取成功或滑脱。

仅做 contact onset 会被 TouchMoment 直接压缩为精确接触检测问题，因此必须包含“物理已发生但视觉结果延迟可见”的事件。

## 4. 三时钟的精确定义

对事件实例 `i`、事件问题 `q_i`、视角 `v`：

```text
x_i^v(<=t)  截止 t 的因果视频前缀
u_i(<=t)    同步的物理传感器流，仅用于标注/训练
```

### 4.1 物理时钟不是伪精确点

由于传感器采样、同步和事件规则也存在误差，将物理时刻表示为区间：

```text
C_phys,i = [l_phys,i, u_phys,i]
```

区间由预注册的事件规则得到，例如：

- 压力首次超过校准阈值并持续 `d` 毫秒；
- 触点电路首次闭合；
- 锁扣状态位翻转；
- 物体速度低于稳定阈值并持续 `d` 毫秒。

每类事件必须有自己的物理定义。不能用同一个 force threshold 代表所有语义结果。

### 4.2 视觉时钟是人群、视角、问题和风险阈值的函数

不存在脱离观察者和任务的唯一 `tau_vis`。定义人群级可验证概率：

```text
r_i^v(t; q, Pi_h) = P(
  observer answers event state/outcome correctly
  AND grounds the answer in visible evidence
  | x_i^v(<=t), q_i, observer population Pi_h
)
```

给定正确率阈值 `eta`、置信下界 `LCB` 和持续时间 `d_vis`：

```text
tau_vis,i^v(eta) = inf {
  t >= u_phys,i :
  LCB[r_i^v(t')] >= eta for all t' in [t, t + d_vis]
}
```

由于有限标注，最终保存的是 crossing-time posterior 或可信区间：

```text
C_vis,i^v = [l_vis,i^v, u_vis,i^v]
```

推荐第一版使用 `eta=0.8`，同时报告 `0.7/0.9` 敏感性。该阈值不是自然常数，必须预注册并做鲁棒性分析。

### 4.3 模型提交时钟

对模型 `m`，定义：

```text
tau_commit,i^(m,v)
```

为模型第一次写入不可回改、类别/状态正确的事件确认时刻。错误确认计为 false commit；未确认计为 censored miss，不能从 latency 样本中删除。

`tau_commit` 使用最后已读视频时间戳，不把 GPU wall-clock 偷换成视频时间。真实计算耗时另行报告：

```text
L_compute = wall-clock output time - wall-clock input availability time
```

### 4.4 严格区分预测与验证

人在物理转移之前可能根据动作趋势猜对结果。这是 anticipation，不是 verification。

PIVOT 主任务要求：

```text
tau_vis >= physical transition interval
```

任何 `tau_commit < l_phys` 都是预测性早报或误报，不能被称为更低验证延迟。若未来研究 anticipation，应另定义第四个变量 `tau_pred`，不混入本论文。

## 5. 评价指标

### 5.1 可观测性差距

```text
Delta_obs^v = E[tau_vis^v - tau_phys]
```

必须同时报告：

- 中位数、IQR、bootstrap CI；
- 按事件类型、视角、遮挡和对象分层；
- 同一物理实例跨视角的 paired difference。

### 5.2 算法决策延迟

对于区间 `C_vis=[l_vis,u_vis]`：

```text
Delta_alg_lower = max(0, tau_commit - u_vis)
Delta_alg_upper = max(0, tau_commit - l_vis)
```

主结果应对 crossing-time posterior 积分，报告期望算法延迟及可信区间，而不是只取区间中点。

### 5.3 早报与失败

- `physical false-early`: `tau_commit < l_phys`；
- `observability-unsupported commit`: `tau_commit < l_vis`；
- wrong-state/wrong-class commit；
- miss / no commit；
- duplicate commit；
- repeated-instance confusion。

### 5.4 传统指标仍保留为次指标

- 相对 `GT_end` 的 raw latency；
- event spotting AP；
- 若输出 span，则报告 tIoU/mAP；
- wall-clock throughput、显存与能耗。

### 5.5 最重要的审计指标

1. raw latency 与 decomposed algorithmic latency 的 Kendall/Spearman 排名相关；
2. 模型 pairwise rank reversal 数量；
3. 在 matched false-early rate 下的 algorithmic delay/recall；
4. 同一物理事件跨视角的 `Delta_obs` 差异。

如果没有模型排名或科学结论变化，这条路线失去主要价值。

## 6. 数据与标注设计

### 6.1 公开数据现实

| 数据源 | 可提供内容 | 当前限制 |
|---|---|---|
| FEEL 2026 | 约 300 万帧 force-synchronized egocentric video，接触分割 | 项目页的 Dataset/Code 链接当前仍指向占位 URL 并返回 404，不能假设已可下载 |
| EgoTouch / TouchAnything 2026 | 1,891 episodes、三视角 RGB、连续压力图 | 论文只承诺将公开，当前需单独确认下载可用性 |
| EgoTactile 2026 | RGB 与全手压力监督 | 适合视觉-物理歧义，但需检查时间同步和许可 |
| TouchMoment 2026 | 4,021 视频、8,456 接触时刻 | 物理时刻主要来自标注，不是独立物理传感器 |
| Ego4D PNR | PRE/CONTACT/PNR/POST、状态变化关键帧 | 公开且成熟，但 PNR 是视觉人工标注，不是独立 `tau_phys` |
| APT 2026 | 14 类原子物理转移、27,303 timed instances | human/simulator timing，不提供三时钟分解 |

### 6.2 推荐数据路线

优先级：

1. 先审计 FEEL/EgoTouch/EgoTactile 是否真实可下载且许可证允许派生标注；
2. 若不可用，做一个小型受控采集，而不是退回 THUMOS 拼接；
3. Ego4D PNR、APT、TouchMoment 只作为外部迁移/竞争基准，不能替代独立物理锚点。

### 6.3 最小受控 pilot

- 100-300 个事件实例；
- 3-5 类事件；
- 8-15 名参与者或至少 5 名执行者；
- 头戴视角 + 一个侧视角，理想情况下加腕部视角；
- force/contact/switch/pressure 之一作为事件特异物理信号；
- 硬件同步误差和视频帧时间量化单独记录；
- subject/object/environment disjoint split。

### 6.4 Prefix 可观测性标注协议

不能让同一个标注者按时间逐步看同一实例，否则其先前记忆会污染后续判断。

推荐协议：

1. 以 `C_phys` 为中心，在 `[-0.5s, +2.0s]` 内自适应采样 4-8 个 prefix endpoint；
2. 每个标注者对同一实例最多看一个 prefix，或采用严格 washout/counterbalancing；
3. 每个判断包含：
   - 事件是否已经发生；
   - 事件类型/结果；
   - confidence；
   - 指出支持判断的可见对象/区域/短时间段；
4. 插入无事件和提前截断 catch trials；
5. 对 prefix 顺序、正负样本和视角随机化；
6. 使用分层 item-response 或 mixed-effects 模型估计 `r_i^v(t)`，控制 rater ability、event difficulty 和 view effect；
7. 通过 rater holdout、split-half 和阈值敏感性检查可重复性。

只收集一个主观“最早看懂帧”是不合格的。

## 7. 最小方法：Ordered Multi-State Observer

### 7.1 方法定位

方法不是主贡献。它回答：在视觉前缀中，显式分离“物理已发生”和“视觉已可验证”是否比 endpoint-only 或 readiness-only 更合理。

### 7.2 状态空间

对每个已知事件查询维护三个有序状态：

```text
S0: physical transition has not occurred
S1: physical transition occurred, but visual evidence is not yet sufficient
S2: transition is visually verifiable from the observed prefix
```

允许 `S0 -> S2` 在一个采样 bin 内发生，但禁止 `S2 -> S0`。如果使用完整 prefix memory，视觉证据可用性应是累积的；模型忘记早期证据应被视为模型限制。

### 7.3 模型结构

```text
causal RGB packets
    -> frozen causal video encoder
    -> bounded temporal state/cache
    -> physical-transition hazard h_phys(t)
    -> observability-transition hazard h_vis(t | physical occurred)
    -> event class/outcome posterior
    -> calibrated deterministic commit rule
```

复杂度预算：

- 复用/冻结一个视觉编码器；
- 新增最多两个轻量 trainable heads；
- 不新增大型 VLM、生成模型、RL policy 或复杂 memory；
- 第一阶段不微调视觉塔。

### 7.4 区间删失似然

设离散时间 hazard 导出的事件时间分布为 `p_theta(T_phys=t)`：

```text
L_phys = -log sum_{t in C_phys} p_theta(T_phys=t | x_<=t)
```

视觉转移遵守顺序约束：

```text
p_theta(T_vis=t | T_vis >= T_phys, x_<=t)
L_vis = -log sum_{t in C_vis} p_theta(T_vis=t | T_vis >= T_phys)
```

总损失：

```text
L = L_phys
  + lambda_vis * L_vis
  + lambda_evt * L_event_or_outcome
  + lambda_cal * L_calibration
```

不建议第一版加入学习式 stopping loss。先在验证集上用同一规则校准所有方法，避免把结果归因于额外 policy 容量。

### 7.5 提交规则

在验证集预注册 `alpha_early`、confidence threshold 和 interval-width threshold：

```text
tau_commit = first t such that
  P(S2 and correct event | x_<=t) >= 1 - alpha_early
  and endpoint credible interval width <= w
```

这只是经验校准规则，不宣称无限时域保证。

### 7.6 训练阶段

**Stage 0: Measurement-only**

- 不训练视频模型；
- 构建 `C_phys`、人群 response curve 和 `C_vis`；
- 检查三时钟问题是否真实存在。

**Stage 1: Frozen-feature observer**

- 缓存因果视频特征；
- 训练 ordered multi-state heads；
- 预算 2-10 GPU-hours；
- 完整 chronological validation。

**Stage 2: Conditional LoRA**

- 仅在 Stage 1 证明任务有价值且 representation capacity 是瓶颈后启用；
- 不允许用 LoRA 提升本身作为创新点。

### 7.7 推理协议

- 只读取当前及过去 packet；
- 传感器不可进入主测试模型；
- 每次读取、状态更新和 commit 写入 ledger；
- commit 后不可删除；
- 多视角主实验分别单视角运行，以测量 view-conditioned observability；
- offline/bidirectional 模型只作上界。

## 8. Baseline 设计

所有方法使用相同因果输入、相同 packet cadence、相同 feature backbone 和相同验证数据校准。

### 8.1 简单强基线

- causal classifier + fixed confidence threshold；
- endpoint-only discrete hazard；
- readiness-only head，不使用 `C_phys`；
- independent two-head model，无 `T_vis >= T_phys` 顺序约束；
- current PCEH 的科学修正版。

### 8.2 任务近邻

- Ego4D PNR temporal localizer；
- TouchMoment/HiCE 风格 event spotting；
- StreamReady 风格 readiness mechanism；
- APT transition detector；
- Action Completion moment detector；
- OAT/ActionSwitch 作为 Online TAL 参照，而非完全同任务 SOTA。

### 8.3 Oracle 与上界

- sensor oracle；
- human observability oracle；
- multi-view oracle；
- offline bidirectional visual model；
- wall-clock zero-compute oracle，用于区分语义延迟与计算延迟。

### 8.4 必须防止的不公平比较

- 不允许 proposed model 看 force，而 RGB baseline 看不到；
- 不允许 proposed model 用完整 prefix replay，而 baseline 只保留短窗；
- 不允许给 proposed model 更密采样率；
- 不允许从 latency 统计中删除 miss；
- 不允许用 test set 调 commit threshold。

## 9. Claim Map

| ID | 论文 claim | 必须提供的证据 | 立即否定条件 |
|---|---|---|---|
| C1 | 物理变化与视觉可验证时间存在稳定、视角相关的差距 | 独立传感器、多人 prefix judgments、同实例跨视角 paired analysis、至少 3 类事件 | 差距小于一个采样单位，或人群区间不可重复 |
| C2 | 传统 endpoint latency 混淆不可约观测延迟与算法延迟 | 至少 3 个模型、2 个视角；raw/decomposed ranking 与 rank reversal | 没有排名变化，也没有新的失败解释 |
| C3 | ordered multi-state observer 在同等早报风险下更快或更准 | 相同 backbone/score、endpoint-only/readiness-only/independent-head baselines | 简单 threshold 或 readiness head 支配 |
| C4 | 多视角改变 `tau_vis` 而不改变 `tau_phys` | 同步 multi-view 同事件实例，within-instance analysis | 视角效应不显著或由同步误差解释 |

论文最多保留 C1+C2 为主 claim，C3 为支持。C4 可并入 C1，不应再扩展第五个贡献。

## 10. 三块核心实验

### Experiment A: Observability Existence Study

问题：三时钟是否真实、稳定且跨事件存在？

- 100-300 instances；
- 3-5 event types；
- 2-3 views；
- 5-10 judgments per sampled prefix；
- 报告 `Delta_obs`、区间宽度、rater reliability 和 view effect。

GO 条件：至少 3 类事件存在超过一个有效采样 bin 的非零 gap，并且区间对 rater holdout 稳定。

### Experiment B: Metric Audit and Rank Reversal

问题：传统 latency 是否给出误导结论？

- causal classifier、endpoint hazard、readiness head、一个强视频模型；
- 同时计算 raw latency、algorithmic latency、false early、miss；
- 报告模型排名相关与 reversal。

GO 条件：分解后至少暴露一个稳定、可解释且跨 seed 的排序/结论变化。

### Experiment C: Minimal Ordered Observer

问题：显式有序状态是否优于简单组合？

- frozen features；
- endpoint-only、readiness-only、independent two-head、ordered model；
- matched false-early rate；
- 3-5 seeds；
- algorithmic delay、recall、calibration、compute。

GO 条件：ordered model 在至少两个事件族上形成非平凡 Pareto 改善，而不是靠少报降低 delay。

## 11. 必要消融

1. point label vs interval-censored target；
2. 单人 timestamp vs population response curve；
3. 无 rater random effect；
4. 无独立物理传感器，只用视觉 boundary；
5. independent hazards vs ordered hazards；
6. 无 view conditioning；
7. 单一阈值 vs interval-width-aware commit；
8. frozen feature vs conditional LoRA，仅在 Stage 2。

## 12. 科学正确性审计

- 同步误差必须小于期望观测 gap，否则结论不可识别；
- 物理事件规则必须预注册且与语义结果一致；
- 同一个标注者不能通过逐步揭示看到同一实例的未来信息；
- `C_vis` 是给定人群、视角、问题和阈值的统计对象，不是客观自然常数；
- 预测性正确不能冒充事件已发生后的视觉验证；
- 多视角只是测量工具，不能在单视角主设置中泄漏；
- 所有 miss、early false commit 和 wrong commit 必须进入评价；
- 完整 chronological test，不使用 EOF retrospective NMS；
- 数据划分按人、对象和环境，避免同一动作模板泄漏。

## 13. Kill Criteria

出现任一项，停止把 PIVOT 作为主线：

1. `C_phys` 与 `C_vis` 的中位差小于一个可靠采样单位；
2. `C_vis` 在 rater holdout、阈值或措辞变化下完全不稳定；
3. 同一物理事件跨视角没有可重复差异；
4. raw latency 与 algorithmic latency 不改变任何方法排名或失败解释；
5. PaSBench、StreamReady、Ego4D PNR 或 APT 已能在相同定义下完成全部任务；
6. 只有 sensor-at-test 才能得到有效结果；
7. endpoint-only 或 readiness-only 强基线在同等早报风险下完全支配；
8. FEEL/EgoTouch 等数据不可获得，且无法进行最小受控采集；
9. 结论只在 contact onset 单一事件上成立；
10. 需要大型 VLM 或数百 GPU-hours 才出现差异。

## 14. 成本与阶段门

| 阶段 | 目标 | GPU | 人工/数据 | 输出 |
|---|---|---:|---:|---|
| P0-A | 数据可访问与同步审计 | 0 | 0.5-1 天 | 可用数据清单、同步误差 |
| P0-B | 30-50 instance 人群标注 micro-pilot | 0 | 10-20 小时 judgments | `C_vis` 可重复性 |
| P0-C | 100-300 instance observability pilot | 0-2h | 30-80 小时 judgments | C1/C2 GO/NO-GO |
| P1 | frozen-feature ordered observer | 2-10h | 无新增采集 | C3 |
| P2 | 扩大数据/多视角/外部集 | 20-80h | 视数据而定 | 完整论文证据 |
| P3 | LoRA 视觉适配 | 50-150h | 无 | 条件增强，不是必需 |

任何大规模训练都必须晚于 P0-C。

## 15. 与当前仓库的关系

当前 OpenTAD/CausalTAD 仓库可复用：

- causal packet reader；
- bounded state/cache；
- read/update/commit ledger；
- no-future replay audit；
- endpoint-only/PCEH score baseline；
- chronological evaluator 基础。

不能直接复用或尚需新增：

- multimodal timestamp alignment；
- event-specific physical interval labels；
- multi-view dataset adapter；
- population prefix annotation schema；
- ordered multi-state interval-censored target；
- decomposed-latency evaluator。

因此它是新任务分支，不是对现有 PCEH head 的小修补。正式实现前应先完成数据和标注 micro-pilot。

## 16. 当前最终建议

PIVOT 仍值得送 Pro 深审，但只能以以下形式送审：

> 一个物理锚定、视角条件、区间化的流式事件验证任务，核心证据是 observability gap 与 metric rank reversal；ordered multi-state observer 只是最小支持方法。

不能再使用以下宽泛故事：

> “我们首次区分物理时间、视觉时间和模型时间。”

APT、PaSBench、Ego4D PNR、StreamReady、TouchMoment 和实时感知评价已经占据了这句话的大部分组成区域。
