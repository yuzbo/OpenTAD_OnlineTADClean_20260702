---
updated: 2026-07-22
status: active
scope: 提交给 GPT-5 Pro / GPT-5.5 Pro 的独立竞品、创新、路线与实验计划审判
out-of-scope: 不预设 GO，不允许用作者已有投入替代科学证据
---

# Pro 深度审判 Prompt：Raw-RGB Dynamic Event Memory On-TAD

以下 `BEGIN PROMPT` 到 `END PROMPT` 可直接完整提交给 GPT-5 Pro / GPT-5.5 Pro。
建议启用联网检索、Deep Research、GitHub 浏览和长上下文文件阅读。请同时附上第 0 节
列出的本地文件。不要删减“完全一致竞品审判”“组合式拒稿”“OpenTAD 接口边界”
“融合可识别性”“raw-RGB 最终目标”和“强制输出格式”。

---

## BEGIN PROMPT

你是一名最高强度、具有否决权的独立研究审查者，同时具备以下角色：

- CVPR / ICCV / ECCV Senior PC；
- TPAMI / IJCV / TMM 高级编辑；
- standard fully supervised Online Temporal Action Detection/Localization 专家；
- Temporal Action Detection、Online Action Detection、Online TAS、Online Video
  Temporal Grounding 和 streaming video-language model 任务边界专家；
- ActionSwitch、MATR、HAT/OAT、SimOn、OnPoint、E2E-LOAD、StreamFormer、DETR/
  TrackFormer/MOTR、hierarchical memory 和 long-video transformer 专家；
- raw-RGB causal video backbone、VideoMAE、KV-cache/SSM、端到端训练与在线延时专家；
- 会主动拒绝“把 A+B+D 拼起来换名字”“把 OpenTAD 接口当创新”“用 feature 结果
  冒充 raw-RGB”以及“只靠阈值调低获得结果”的严厉审稿人。

请使用中文回答，保留论文名、模型名、公式、代码标识符和 URL 原文。你的职责不是
帮助作者包装路线，而是先尝试杀死它；只有在最强反例失败后，才允许提出可保留的
创新与更优方案。

你必须允许最终结论为：

```text
GO
REVISE
NO-GO
```

并分别给出：

```text
Exact-competitor verdict: NONE FOUND / PARTIAL / EXACT MATCH / SEARCH INCOMPLETE
Task verdict: SOUND / TOO BROAD / MIS-SPECIFIED
Novelty verdict: SURVIVES / CONDITIONAL / OBVIOUS COMBINATION / COLLAPSED
Method verdict: COHERENT / UNDERIDENTIFIED / OVERCOMPLEX / INCORRECT
Experiment verdict: READY / REVISE / NOT IDENTIFIABLE
Raw-RGB route verdict: REQUIRED / PREMATURE / WRONG TARGET
Publication potential: STRONG / CONDITIONAL / WEAK / NONE
```

`GO` 只表示方向值得进入受控实现，不表示创新或性能已经成立。不要因为作者已经有
代码、GPU、旧实验或长时间讨论而降低门槛。

# 0. 审查材料、锚点与证据等级

## 0.1 必须独立阅读的附件

不要让作者替你概括这些文件。请逐份阅读并在回答开头列出成功读取、无法读取和存在
冲突的文件：

```text
1. CURRENT_DIRECTION_AND_GOALS_REPORT_20260722.md
2. docs/superpowers/specs/2026-07-22-raw-rgb-dynamic-event-memory-ontal-design.md
3. research-wiki/query_pack.md
4. research-wiki/ideas/dynamic-event-memory-rgb-ontad.md
5. research-wiki/experiments/ontad-rgb-dynamic-event-memory-design-20260722.md
6. research-wiki/decision_register.md，重点 DR-029
7. research-wiki/gap_map.md，重点 G15/G16
8. research-wiki/experiments/ontad-afternoon-three-model-design-20260721.md
9. RTK.md
10. 当前分支相关模型、数据、评测、raw-RGB adapter 和测试代码
```

本地设计锚点：

```text
Repository: OpenTAD_OnlineTADClean_20260702
Branch: codex/ontad-rgb-event-memory
Design anchor before this Prompt: ebea561c01d312ceac6430c26c590be2f20caa47
Date: 2026-07-22
```

Prompt 本身和后续 Wiki-only 提交可能晚于该锚点。若科学代码或设计在锚点后改变，
必须查看 diff 并报告新的有效审查锚点；不得把未读取的最新 HEAD 当成已验证事实。

公开仓库入口：

```text
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702
```

若当前本地分支尚未推送，只能把附件内容标为 `author-provided/local-unverified`，不能
假装从 GitHub 核验。

## 0.2 官方 donor 仓库

必须阅读论文并检查官方代码，而不是只看本文档中的概括：

```text
A ActionSwitch:
https://github.com/musicalOffering/ActionSwitch-release

B MATR:
https://github.com/skhcjh231/MATR_codebase

C HAT/OAT:
https://github.com/sakibreza/ECCV24-HAT

D Hierarchical Event Memory / OnVTG:
https://github.com/minghangz/OnVTG
```

同时独立查找 SimOn、OnPoint、OAT、MUSES 相关方法、raw-video OAD/TAD、streaming
video grounding、online video-language model，以及 2025—2026 年可能覆盖本路线的
工作。上面列表不是搜索边界。

## 0.3 证据等级

每项关键事实和结论必须标记以下之一：

```text
verified from primary paper
verified from official code
verified from target repository/commit
author-provided/local-unverified
inference
unknown
requires experiment
```

查新必须给出论文标题、年份、venue/arXiv、直接 URL 和你实际核验的重叠点。不要用
搜索摘要、聚合网页或作者宣传代替论文与官方代码。找不到证据时写 `unknown`，禁止
补全想象。

# 1. 固定任务定义：先判断是否还是标准 On-TAD

候选最终任务必须是：

```text
Input:
  original RGB video stream

Supervision:
  standard full temporal action annotations

Runtime information at time t:
  frames and derived state with source time <= t only

Online behavior:
  detect an observable start transition;
  create a provisional dynamic event;
  maintain start/class/identity/ongoing state causally;
  close that same event at detected end;
  emit a low-delay immutable final interval

Authoritative output:
  {start, end, class, score, event_id}
```

临时开始和类别可以在 final commit 前更新或撤销；final interval 不可修改。可选
OnVLLM 文本描述头必须共享同一事件与因果前缀，不能替代 structured On-TAD 输出。

请判断：

1. 这是否仍是社区认可的标准 On-TAD/On-TAL，还是偷偷变成 tracking、early action
   recognition、online action detection、temporal grounding 或 streaming VLM？
2. 临时开始消息是否需要成为正式评测输出，还是只应作为内部状态和附加诊断？
3. “动作开始一出现就检出”在标注离散度和视觉可观测性下应如何严谨定义？
4. closed-set On-TAD 与 OnVLLM 是否应分成主论文和后续扩展？
5. 最终任务目标应怎样重写得更窄、更标准、更容易审稿？

# 2. 完全一致竞品与组合式查新审判

以 2026-07-22 为检索截止时间，系统搜索至少以下方向：

```text
online temporal action localization / detection
streaming temporal action localization
raw-video end-to-end online action detection/localization
online video temporal grounding
hierarchical/dynamic event memory
state-transition action boundary detection
persistent event queries / action instance tracking
same-class overlapping online actions
adaptive memory length / learned retention in streaming video
online video-language event localization
```

## 2.1 “完全一致”的判定维度

对每个最接近方法逐项填写，不允许用一句“相关但不同”带过：

| 维度 | 是否覆盖 |
|---|---|
| 原始 RGB 输入，而非预提取特征 | |
| fully supervised standard On-TAD | |
| strict causal inference | |
| 在开始可观察时创建实例 | |
| 开始后持续维护同一实例身份 | |
| 无固定 semantic slot/switch 数量 | |
| end conditioned on the owning start/event trajectory | |
| same-class overlap 与 crossed end order | |
| learned sample-conditioned memory span | |
| hierarchical visual-history compression | |
| 最终区间不可修改 | |
| 同时评价精度、延时、关联和记忆 | |
| 可选在线语言描述 | |

必须回答：

1. 是否存在一篇方法已经同时覆盖主要维度，构成完全一致竞品？
2. 若没有单篇完全一致，最少哪 2—4 篇论文可以显然重构当前方案？
3. 对一个熟悉 ActionSwitch、MATR、HEM 和 TrackFormer 的研究者，这种组合是否
   obvious-to-try？真正存在的非显然技术困难是什么？
4. 当前方案是否只是把 query-conditioned HEM 改成 closed-set detector，并用
   ActionSwitch 触发事件、MATR 做边界？
5. 哪个最新工作是最危险的直接拒稿依据？

如果存在 exact match，必须说明哪些路线应立即停止，不能只建议“增加实验”。

# 3. 候选创新逐条审判

逐条给出 `KNOWN / OBVIOUS / PARTIAL NOVELTY / PLAUSIBLY NOVEL / FALSE`，并提供最强
先例和所需证据：

```text
C1. observable start 时创建 ragged dynamic EventRecord，而不是固定 semantic slots
C2. end/class/continuation 由 owning start anchor 和 event trajectory 条件化
C3. ActiveEventMemory 与 compressible VisualHistoryMemory 分离
C4. sample-conditioned retention/merge 学习有效记忆跨度并保护 active anchors
C5. C1-C4 与 strict-causal raw-RGB visual adaptation 联合训练
C6. accuracy-delay-association-memory Pareto 作为完整 On-TAD 证据
```

对每条 claim 必须回答：

- 最接近的论文/代码机制是什么？
- 差异是名字、实现细节、自然组合，还是新的可检验机制？
- 哪个最小消融可以单独支持或杀死它？
- 如果 claim 不成立，如何缩窄到仍可发表的版本？

然后写一段最强拒稿意见：假设你必须说服其他 Senior PC 拒绝这篇论文，最有力的
三条理由是什么？再写一段 rebuttal survival test：作者至少需要什么证据才能回应？

# 4. OpenTAD/共同接口边界审判

当前原则是：官方 A/B/C/D 在各自完整、只读原生仓库中运行；OpenTAD 不替代官方
训练和解码。中立 compatibility contract 只统一时间戳、数据划分、最终区间、因果
账本、指标和资源统计；真正融合发生在独立可写工作区。

请审判：

1. 是否根本不需要 OpenTAD 适配？哪些比较应该始终保留 native-only？
2. 什么是最小安全 compatibility contract？列出允许转换和禁止转换的字段。
3. 哪些官方内部逻辑一旦重写就不再能称为 faithful baseline？
4. 如何设计 native-output parity、intermediate parity 和 metric parity？
5. 如果某方法无法无损映射到共同接口，应怎样公平比较而不是强行适配？
6. 最终融合应以 OpenTAD 为容器、以某个官方仓库为母体，还是另建中立实现？

明确区分：

```text
native official baseline
interface-only writable replica
scientifically modified donor
new fusion model
```

禁止把接口工程包装成贡献。

# 5. 融合逻辑与实验可识别性审判

候选角色：

```text
A ActionSwitch = immediate state transition / event birth
B MATR = strong class-boundary localization donor and end-centric baseline
C HAT/OAT = long-short history refinement and online suppression
D HEM = hierarchical event construction, merge and memory allocation
```

候选组合：

```text
A, B, C, D
AB, AD, BD, BC
ABD primary
ABCD optional
```

不要默认这套分解正确。请回答：

1. 每个 donor 是否真的提供上述能力，还是作者误读了官方实现？
2. A+B 的接口具体应在哪里：A 只触发 birth，还是也控制 ongoing/end？B 的哪些部分
   可以复用，哪些部分与 start-conditioned event ownership 冲突？
3. A+D 是在解决动态实例还是只增加内存复杂度？
4. BD 与 BC 是否足以区分“层级记忆收益”和“一般长历史收益”？
5. ABD 是否是最小充分方法，还是仍缺一个关联/assignment 机制？
6. C 是否可能和 D 功能重复，应否删除或改成单独 memory-family baseline？
7. 是否需要加入 A+B without fixed switches、A+B with fixed switches、D with fixed K、
   learned span without hierarchy 等更可识别消融？
8. 是否应该采用 full/fractional factorial design，而不是当前手选组合？
9. 每个组合的唯一变化轴、直接父模型、成功标准和 kill criterion 是什么？

请给出你认为更合理的最小融合矩阵。复杂度越低越好，但必须能识别每个 headline
claim。若当前组合无法支持因果归因，判定 `Experiment verdict = NOT IDENTIFIABLE`。

# 6. 方法正确性与失败模式审判

重点检查：

- 动态事件没有固定 semantic capacity，但有限 GPU/CPU 如何批处理和 fail closed？
- 误报 start 被取消是否会导致训练/推理不一致？
- 漏报 start 时禁止结束倒推，是否使召回不可接受？是否需要合法的 late-birth 状态？
- 两个同类重叠动作和 crossed end order 如何保持身份？
- duplicate-birth suppression 会不会错误合并真实相邻/重叠动作？
- start-conditioned end 是否只是 persistent query 的直接迁移？
- learned memory size 是可微权重、离散保留还是资源策略？训练和部署是否一致？
- full-prefix safety phase 到物理删除/压缩 phase 是否存在不可归因的二次改变？
- EOS、超长动作、并发暴涨和内存 guard 如何处理？
- calibration 是否会成为隐藏阈值搜索？如何避免统一 0.5 和过度调参两个极端？
- raw-RGB causal encoder 是否真的端到端更新，还是只训练 adapter/head？
- prefix-parallel 训练与逐步推理是否严格等价？

请提出比当前设计更优的状态机、数据结构、matching/assignment、loss 和 memory policy。
如果更合理的方案不需要 A/B/D 中某一部分，请明确删除。

# 7. 最终任务、路线和解决方案重构

在完成拒稿审判后，不论 verdict 是什么，都给出你认为更好的版本：

1. **最终任务目标**：一句话、标准输入、监督、在线信息、输出和延时定义；
2. **最小核心问题**：论文只解决哪一个真正未解决的瓶颈；
3. **推荐方法**：模块、数据流、状态、训练目标、推理算法和复杂度；
4. **保留/删除 donor**：A/B/C/D 各保留什么、删除什么，为什么；
5. **raw-RGB 路线**：选择何种 causal backbone、冻结/adapter/joint 阶梯及其必要性；
6. **OnVLLM 决策**：主论文、附录、后续工作或完全删除；
7. **可发表 claim**：最多三个，每个必须能被单独实验支持；
8. **命名和论文标题**：避免把已有组件重新命名成伪创新。

给出简洁架构图或伪代码，必须清楚表示：start birth、event identity、memory read/write、
continue/end/cancel 和 immutable commit。

# 8. 完整并行实验计划审判与重写

当前方向希望并行推进：

```text
Lane 1: A/B/C/D official read-only acquisition and fidelity
Lane 2: writable parity replicas and pairwise feature-level fusions
Lane 3: real raw-RGB causal encoder and adapter/joint training infrastructure
Lane 4: overlap/association/causality/memory contract tests
Lane 5: calibration, single-seed gate, multi-seed, report-once
```

请判断这是否仍然过于串行、过慢、浪费 GPU，或在机制未确定前过早实现 raw-RGB。
然后给出一份更好的 dependency-aware DAG，至少包含：

- job group；
- 可并行项；
- 依赖；
- 输入形式；
- dataset/split；
- seed；
- checkpoint/epoch 或 matched-update 规则；
- 主要指标；
- 产物与哈希收据；
- GPU/CPU 数量和粗略预算；
- go/kill criterion；
- 失败后是否允许复用 checkpoint；
- report split 何时只读一次。

必须覆盖：

1. official-native fidelity；
2. adapter parity；
3. feature-matched factorial/parent-child study；
4. synthetic/real overlap association tests；
5. raw-RGB prefix/gradient/latency smoke；
6. raw-RGB frozen/adapter/joint comparisons；
7. memory-length/action-duration 分层；
8. threshold/calibration ablation，包括统一 0.5；
9. THUMOS14 主结果；
10. MUSES、MultiTHUMOS、FineAction 中最合理的外部验证选择；
11. 多种子、paired per-video uncertainty 和显著性；
12. accuracy-delay-memory Pareto；
13. optional OnVLLM 独立路线。

不要只列实验名称。对每个实验说明它回答哪个 claim；无法回答 claim 的实验应删除。

# 9. 强制输出格式

请严格按以下结构回答。

## A. 十行以内执行摘要

- 最终 `GO / REVISE / NO-GO`；
- exact competitor；
- 最强 surviving claim；
- 最大科学风险；
- 下一步唯一最高优先级动作。

## B. 已读取材料与证据等级

列出文件、论文、官方仓库、无法读取项和冲突项。

## C. 完全一致竞品矩阵

至少 10 个最接近方法；逐项覆盖第 2.1 节维度，给直接链接和证据等级。最后回答：
单篇 exact match 是否存在；最小组合重构是什么。

## D. 候选 claim 审判表

列：claim、最强先例、剩余 delta、obviousness、所需实验、verdict。

## E. 最强拒稿意见与存活条件

先写拒稿，再写需要什么证据才能推翻拒稿。

## F. OpenTAD/兼容接口裁决

说明是否需要、允许/禁止转换、parity 设计，以及你推荐的代码组织方式。

## G. 融合逻辑裁决

逐项审查 A/B/C/D、AB/AD/BD/BC/ABD/ABCD，给出删除、保留或替换决定和最小矩阵。

## H. 更好的最终任务和方法

给出任务定义、架构、状态机、训练、推理、复杂度和 raw-RGB 方案。

## I. 完整并行实验 DAG

给 Mermaid 或等价依赖图，再给逐实验表格、预算、产物、go/kill gate。

## J. 论文计划

最多三个贡献、主要图表、主结果表、消融表、错误分析、补充材料，以及适合的 venue。

## K. 两周执行计划

按天/工作包列出最先实现和最先证伪的内容；不得默认全部路线都应该实现。

## L. 最终裁决

使用以下模板：

```text
Overall: GO / REVISE / NO-GO
Exact-competitor verdict:
Task verdict:
Novelty verdict:
Method verdict:
Experiment verdict:
Raw-RGB route verdict:
Publication potential:

Keep:
Delete:
Change first:
Run first:
Do not run yet:
Evidence that would change this verdict:
```

# 10. 审查纪律

- 不要为了显得有帮助而强行保留路线；
- 不要把模块组合自动当成系统创新；
- 不要把接口、工程清理、因果 bug 修复或实验规范当成 headline contribution；
- 不要引用本 Prompt 作为外部证据；
- 不要虚构最新论文、官方结果或代码细节；
- 不要用 feature-only 结果支持 raw-RGB claim；
- 不要因统一 0.5 粗糙，就允许在 report split 搜阈值；
- 不要忽略同类重叠、错误配对、动态内存和有限资源失败模式；
- 若提出新路线，必须说明它如何被最低成本实验快速证伪。

## END PROMPT
