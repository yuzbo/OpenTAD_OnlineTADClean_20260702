# PIVOT / 三时钟可观测性路线竞争工作查新

检索日期：2026-07-11
覆盖时间：重点 2024-2026，向前追溯关键任务定义
结论等级：`PROCEED WITH CAUTION`
建议查新复核时间：进入数据采集前再次检索

## 1. Executive Verdict

### 1.1 是否已有完全相同的工作

截至本次检索，**未发现完全同时满足以下四点的工作**：

1. 用独立 force/contact/pressure/switch 信号锚定真实物理转移；
2. 将视觉可验证时刻建模为人群、视角、问题和置信阈值条件下的区间；
3. 在严格因果流式协议下记录模型不可回改的 commit time；
4. 将总延迟分解为 observability gap 与 residual algorithmic delay，并审计模型排名变化。

但这不是“无人区”。四个组成部分周围已经非常拥挤，最近工作的组合几乎可以重构宽泛版三时钟故事。

### 1.2 最强拒稿句

> This is a benchmark-level union of Ego4D/TouchMoment/APT physical-transition timing, StreamReady/Thinking-QwenVL evidence readiness, PaSBench proactive warning timing, and FEEL/EgoTouch privileged sensors, followed by a standard interval-censored multi-state model.

如果论文不能用同一物理事件的多视角、传感器锚点和 population observability curve 证明稳定的不可约 gap 与 ranking reversal，这个拒稿理由成立。

### 1.3 新颖性评分

| 层面 | 评分 | 判断 |
|---|---:|---|
| 宽泛“三个时间点”概念 | 3/10 | 低；PaSBench 已接近 first visible risk -> accident -> warning，实时感知也已有 sensing/output latency 分解 |
| 物理锚定 + population visual interval + commit 分解任务 | 7/10 | 中高，但依赖精确定义和数据 |
| view-conditioned observability gap | 7.5/10 | 当前最有辨识度的部分 |
| ordered multi-state survival 方法 | 4/10 | 标准机制，只能作支持方法 |
| task/evaluation paper 潜力 | 7/10 | 必须证明 ranking reversal 和真实部署解释 |
| 纯模型论文潜力 | 4/10 | 不建议作为纯网络创新投稿 |

## 2. 检索范围与策略

检索主题簇：

- online/streaming temporal action localization；
- first-sufficient-evidence、readiness、trigger timing；
- object state change、physical transition、PNR、completion moment；
- force/tactile/contact synchronized egocentric video；
- early action recognition/anticipation；
- boundary uncertainty、annotator disagreement；
- streaming perception、latency-aware evaluation；
- privileged modality learning/distillation。

主要来源：arXiv、CVF Open Access、ECVA、OpenReview、Nature Communications、论文项目页。技术判断只使用论文或官方项目页作为主要依据。

检索局限：

- 未执行付费数据库的系统综述式穷尽检索；
- 2026 年新论文和数据发布状态变化快；
- 部分 OpenReview 稿件仍可能变更；
- “没有找到完全相同工作”不是首创证明。

## 3. 竞争版图

### Cluster A: 物理状态转移与关键时刻

| 工作 | 已覆盖内容 | 对 PIVOT 的威胁 | 剩余 delta |
|---|---|---|---|
| APT 2026 | 定义 14 类 Atomic Physical Transitions，27,303 个 timed instances，绑定视觉证据、物理机制和前后状态 | 极高；“物理转移的时间定位”已不能称新 | APT 不用独立传感器区分真实发生与视觉可见，也不评价 commit residual delay |
| Ego4D PNR 2022 | PRE/CONTACT/PNR/POST，定位状态变化开始或不可逆关键帧 | 极高；状态变化关键时刻和关键帧已有成熟 benchmark | PNR 是视觉人工定义，不是外部物理锚点；没有 population visual interval 与在线 commit 分解 |
| VidOSC / HowToChange 2024 | initial/transition/end 三阶段的开放世界 object state change localization | 高；三阶段状态建模和开放世界定位已存在 | 不是流式物理-视觉可观测性分解 |
| TouchMoment 2026 | 精确检测手与物体接触时刻，2-frame tolerance | 高；contact onset 的精确 event spotting 已被占据 | 主要是视觉接触检测，没有独立物理 clock 与 observer/commit clocks |
| Action Completion 2018 | 目标完成时刻和不完整动作 | 中高；完成时刻并非新对象 | 无独立物理传感、视觉可观测区间和流式残余延迟 |

主要来源：

- APT: <https://arxiv.org/abs/2606.18586>
- Ego4D: <https://openaccess.thecvf.com/content/CVPR2022/html/Grauman_Ego4D_Around_the_World_in_3000_Hours_of_Egocentric_Video_CVPR_2022_paper.html>
- Ego4D official Hands + Objects description: <https://ego4d-data.org/index.html>
- VidOSC: <https://openaccess.thecvf.com/content/CVPR2024/html/Xue_Learning_Object_State_Changes_in_Videos_An_Open-World_Perspective_CVPR_2024_paper.html>
- TouchMoment: <https://arxiv.org/abs/2604.12343>
- Action Completion: <https://arxiv.org/abs/1805.06749>

### Cluster B: 首次充分证据与响应时机

| 工作 | 已覆盖内容 | 对 PIVOT 的威胁 | 剩余 delta |
|---|---|---|---|
| StreamReady 2026 | evidence windows、Answer Readiness Score、早晚非对称惩罚、readiness head | 极高；“视觉证据何时足够”已经是正式任务 | 无独立 `C_phys`，不分 sensing/observability gap 与 algorithmic delay |
| Thinking-QwenVL 2026 | first-sufficient-evidence timestamp、progress/confidence、在线 response timing | 高；首次充分证据和何时回答不是新概念 | VideoQA，不是物理状态验证；无物理 anchor |
| Learning to Respond 2025/2026 | trigger-centric online video understanding，检测 sufficient evidence 后立即响应 | 高；trigger/readiness 任务已拥挤 | 无外部物理时钟和视角条件可观测性分解 |
| PaSBench-Video 2026 | first visible sign、risk onset、accident boundary、causal warning、safe-video FP | 极高；在安全领域已接近“三时间”结构 | 风险 onset/accident 仍是视觉人工标注；不是同一物理事件的跨视角/传感器测量 |

主要来源：

- StreamReady: <https://arxiv.org/abs/2603.08620>
- Thinking-QwenVL: <https://arxiv.org/abs/2604.18459>
- Learning to Respond: <https://openreview.net/forum?id=gmpnSSiJt7>
- PaSBench-Video: <https://arxiv.org/abs/2606.02443>

### Cluster C: 物理传感器与视觉-触觉学习

| 工作 | 已覆盖内容 | 对 PIVOT 的威胁 | 剩余 delta |
|---|---|---|---|
| FEEL 2026 | 约 300 万 force-synchronized egocentric frames，contact segmentation、force prediction pretraining | 极高的数据和表示近邻 | 不定义 population observability 或 online commit decomposition |
| TouchAnything / EgoTouch 2026 | 208 tasks、1,891 episodes、头部+双腕视角、连续压力图、vision-to-touch | 极高；提供 PIVOT 理想的数据结构 | 不研究 physical/visual/decision clocks；但公开状态尚未确认 |
| EgoTactile 2026 | RGB+全手压力、partial-observation ambiguity、视觉到压力估计 | 高；视觉-物理歧义已被显式研究 | 不定义事件时间分解 |
| Multimodal Distillation 2018/2023 | 训练时多模态、测试时 RGB，且改善 calibration | 高；sensor-at-train/RGB-at-test 不是创新 | PIVOT 必须把传感器当 measurement anchor，而非蒸馏 novelty |

主要来源：

- FEEL: <https://arxiv.org/abs/2603.15847>
- FEEL project: <https://www.cs.umd.edu/~edessale/feel>
- TouchAnything: <https://arxiv.org/abs/2605.13083>
- EgoTactile: <https://arxiv.org/abs/2606.09243>
- Multimodal Distillation for Egocentric Action Recognition: <https://openaccess.thecvf.com/content/ICCV2023/html/Radevski_Multimodal_Distillation_for_Egocentric_Action_Recognition_ICCV_2023_paper.html>
- Graph Distillation for Action Detection with Privileged Modalities: <https://openaccess.thecvf.com/content_ECCV_2018/html/Zelun_Luo_Graph_Distillation_for_ECCV_2018_paper.html>

### Cluster D: 边界不确定性与人类标注差异

| 工作 | 已覆盖内容 | 威胁 | 剩余 delta |
|---|---|---|---|
| Boundary Uncertainty 2020 | 用 Gaussian 表示 TAL boundary uncertainty | 中高；区间/分布边界不是新方法 | 预测不确定性不同于 population visual observability |
| Diagnosing Error in Temporal Action Detectors 2018 | 多人重标 ActivityNet 边界，报告边界共识问题 | 高；必须正视人类边界并不精确 | 没有物理传感锚点和 prefix response curve |
| Timestamp-supervised TAS 2024 | 显式承认边界不明确与 inter-rater disagreement | 中 | 任务不同，仍说明 point `tau_vis` 不可信 |

主要来源：

- Boundary Uncertainty: <https://arxiv.org/abs/2008.11170>
- Diagnosing Error: <https://openaccess.thecvf.com/content_ECCV_2018/html/Humam_Alwassel_Diagnosing_Error_in_ECCV_2018_paper.html>
- Random Walks for TAS: <https://openaccess.thecvf.com/content/WACV2024/html/Hirsch_Random_Walks_for_Temporal_Action_Segmentation_With_Timestamp_Supervision_WACV_2024_paper.html>

### Cluster E: 实时与流式延迟评价

| 工作 | 已覆盖内容 | 威胁 | 剩余 delta |
|---|---|---|---|
| Towards Streaming Perception 2020 | 将计算 latency 与实时 accuracy 联合评价，发现 accuracy/latency sweet spot | 高；“传统评价忽略 latency”已不是新论点 | 关注计算后输出过时，不关注证据尚未视觉出现 |
| STARE 2026 | continuous sampling、latency-aware evaluation、500 Hz GT，报告 ranking reversal | 极高；latency-aware metric 和 ranking reversal 都已有强先例 | event-camera tracking 的计算/采样延迟，不是 semantic observability gap |

主要来源：

- Towards Streaming Perception: <https://publications.ri.cmu.edu/towards-streaming-perception>
- STARE: <https://www.nature.com/articles/s41467-026-70240-6>

### Cluster F: Online TAL

OAT、CAG-QIL、MATR/HAT、ActionSwitch、OnPoint、OZ-TAL 已覆盖 early proposal、memory、state change、same-class repetition、distillation 与 zero-shot On-TAL。PIVOT 不能用以下内容作为主要 delta：

- causal prefix input；
- endpoint refinement；
- state-change head；
- immutable detection ledger；
- frozen VLM features；
- online latency metric 本身。

## 4. 最危险的五篇/五类工作

### 4.1 PaSBench-Video

它已经把 first visible danger、accident boundary 和 model warning 放在同一个因果评价框架中。宽泛“三时钟”会被直接攻击。

生存条件：PIVOT 必须依赖独立物理传感器、同事件跨视角测量和 population interval，而不是人工 risk onset/accident frame 的一般化。

### 4.2 APT

APT 将物理机制、前后状态、视觉证据和 timed transition 绑定，且覆盖 contact、gravity、friction、rotation/stability。任何“原子物理转移检测”表述都已被占据。

生存条件：贡献是 observability decomposition，不是 transition vocabulary 或 VLM fine-tuning。

### 4.3 StreamReady

它已有 evidence window、early/late penalty 和 readiness head。

生存条件：视觉 evidence window 必须由独立物理时间和人群统计定义，而不是任务作者主观给定；评价必须分离不可约观测 gap。

### 4.4 Ego4D PNR

PNR 是成熟的 state-change keyframe localization 基准，且有 PRE/CONTACT/PNR/POST。

生存条件：证明视觉 PNR 不是 physical clock，并用 sensor synchronization 测出二者系统性差异。

### 4.5 STARE

STARE 已经证明 latency-aware 评价可以造成模型 ranking reversal。

生存条件：PIVOT 不能把“rank reversal”本身当首创，而要证明 reversal 来自 semantic observability confounding，不是 compute throughput。

## 5. 数据可行性审计

### 5.1 FEEL 当前不能被视为已就绪数据

FEEL 官方项目页显示 Dataset 和 Code 链接，但实际链接分别指向 `DATASET_URL` 和 `CODE_URL` 占位地址并返回 404。论文内容真实存在，但当前公开可下载状态未通过审计。

### 5.2 EgoTouch 当前是承诺发布

TouchAnything 摘要使用 “We will publicly release” 表述。本次检索发现项目页，但没有确认可实际下载的数据入口。

### 5.3 结论

PIVOT 的最大瓶颈不是 GPU，而是物理同步数据。当前路线必须把以下任一项设为 P0：

1. 获得 FEEL/EgoTouch/EgoTactile 真实数据访问；
2. 与作者合作；
3. 自采 100-300 个实例的受控 pilot。

没有独立物理信号时，路线会退化成 Ego4D PNR + StreamReady，创新性明显不足。

## 6. 可防守的唯一核心 delta

建议只保留一句：

> We measure a view-conditioned observability gap between an independently sensed physical transition and the earliest population-level visual verification interval, then evaluate only the residual delay of causal model decisions.

这句话包含四个不可删除的限定词：

- independently sensed；
- view-conditioned；
- population-level interval；
- residual delay。

删除其中任意两个，路线很可能被现有工作覆盖。

## 7. 不可声称的内容

- “首次研究视频中的物理状态变化”；
- “首次研究 first sufficient evidence”；
- “首次研究何时在线回答”；
- “首次使用传感器辅助 RGB 视频理解”；
- “首次用区间/分布建模 action boundary”；
- “首次做 latency-aware streaming evaluation”；
- “首次证明 latency metric 会改变模型排名”；
- “首次做精确 contact moment detection”；
- “首次将状态转移分成 pre/transition/post”。

## 8. Senior-PC 风险评分

| 风险 | 严重度 | 缓解方式 |
|---|---:|---|
| 被认为是已有任务的组合 | 5/5 | 同实例传感器+多视角+人群区间，证明不可由单一已有任务表达 |
| `tau_vis` 主观不可识别 | 5/5 | population response curve、固定 query/threshold、rater random effects、区间而非点 |
| 物理传感器不等于语义事件 | 5/5 | event-specific operational definition 与同步误差区间 |
| 数据不可获得 | 5/5 | P0 数据访问审计或小型受控采集 |
| 方法过于标准 | 4/5 | 明确 task/evaluation 为主贡献，方法保持最小 |
| 只对 contact 有效 | 4/5 | 至少三类不同 observability pattern |
| 没有 ranking reversal | 5/5 | 作为 kill criterion，不靠改写 story 生存 |
| 偏离 Online TAD | 3/5 | 主动承认是 streaming event verification；On-TAL 仅 baseline |

## 9. Final Novelty Decision

### 当前建议

`PROCEED WITH CAUTION`，但只允许先做数据和测量 pilot。

### 进入实现的必要条件

1. 独立物理时钟数据可访问；
2. 至少 30-50 个实例上 population `C_vis` 可重复；
3. 至少两种视角产生可测 `Delta_obs` 差异；
4. PaSBench/APT/StreamReady/Ego4D PNR 无法直接覆盖该定义；
5. Pro 审查认为 task-level delta 足够支撑目标 venue。

### 当前不是的东西

它不是确定的新主线，也不是无人区。它是一个位于五个成熟方向交叉处的窄问题，只有非常严格的测量定义和实证反转才能使它成为强论文。
