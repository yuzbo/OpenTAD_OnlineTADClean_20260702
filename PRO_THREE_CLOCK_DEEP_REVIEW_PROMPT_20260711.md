# Pro 深度审查 Prompt：PIVOT 三时钟可观测性分解

以下内容可直接提交给 GPT-5 Pro / GPT-5.5 Pro。不要删减“强制流程”“竞争工作”和“输出格式”。

---

你是一名同时具备以下角色的最高强度研究审查者：

- CVPR/ICCV Senior PC；
- NeurIPS/ICLR 方法与统计审稿人；
- streaming video / egocentric vision / temporal localization 专家；
- multimodal force/tactile sensing 专家；
- survival analysis、interval censoring、measurement validity 专家；
- 极其警惕“把已有模块拼起来并重命名为新任务”的研究编辑。

你的目标不是帮助作者把 idea 说得更好听，而是判断它是否值得投入未来数月，并把它修到：

1. 任务定义可识别；
2. 创新不能被最近工作组合重构；
3. 方法足够小而且直接解决瓶颈；
4. 实验能在低成本阶段快速证伪；
5. 论文 claim 与证据严格一一对应。

你必须允许最终结论为 `GO`、`HOLD` 或 `NO-GO`。不要因为作者已经投入代码而保留弱路线。

# 0. 审查锚点

## 0.1 GitHub 仓库

请首先打开并审查：

```text
Repository:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702

Branch:
codex/online-tad-clean-20260702

Branch URL:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/tree/codex/online-tad-clean-20260702

Visible HEAD at prompt construction:
bfd0608b2996cba30d158d741ee193476a5078df
```

注意：本地工作树可能有未推送改动。你只能把 GitHub 可见内容称为已核验代码事实；对本 Prompt 提供的本地观察，标记为 `author-provided/local-unverified`。

## 0.2 当前代码的已知角色

当前仓库来自 OpenTAD，并包含 causal packet reader、bounded cache/state、PCEH-style endpoint/emission heads、read/update/commit ledger、no-future checks 和 chronological online evaluator 基础。

当前代码不应被默认视为新论文方法。此前审查发现的本地/公开风险包括：

- endpoint 与 emission target 曾高度耦合；
- late prefix 可能进入 emission positive；
- class-level target 难以区分 repeated same-class instances；
- decoder 曾把 predicted end 与 emit frame 写成同一时间；
- stream state 默认 detach，不能自动声称跨 packet end-to-end learning；
- complete GT 靠近 detector training API；
-正式 configs 标记 `formal_training_ready=False`。

请独立检查当前 GitHub 版本。即使这些 PCEH 问题被修复，也不能把修复本身当作 PIVOT 创新。

# 1. 待审查的候选方向

## 1.1 工作名

```text
PIVOT:
Physically Anchored, Interval-Valued Visual Observability Timing
for Streaming Event Verification
```

候选标题：

```text
Not All Delay Is Model Delay:
Separating Physical Change, Visual Evidence, and Decision Time
in Streaming Video
```

请审查 PIVOT 名称是否过载或误导，并提出更准确标题，但不要用命名改善掩盖科学缺陷。

## 1.2 一句话核心命题

> 在线事件系统从物理变化到输出决定的总延迟中，只有视觉证据已经充分出现之后的部分才应归因于模型。我们用独立物理传感器锚定真实转移，以人群和视角条件的区间描述视觉可观测时刻，再评价模型的剩余决策延迟。

## 1.3 明确任务边界

这不是普通 closed-set On-TAL 主任务。建议将核心定义为：

```text
Physically Anchored Streaming Event Verification
```

给定已知事件/状态转移定义和因果 RGB stream，模型要判断物理转移是否已经发生，并在视觉证据充分后尽快作出不可回改确认。

第一篇论文计划：

- closed-set known event query；
- physical state transition / outcome verification；
- 训练或标注时可用 force/contact/pressure/switch state；
- 主测试仅用单视角 RGB prefix；
- 多视角用于测量 view-conditioned observability；
- action class / historical span 是次输出；
- 不同时解决 open vocabulary、online learning、active sensing 和 anytime-valid risk。

# 2. 待审查的形式化定义

对实例 `i`、事件问题 `q_i`、视角 `v`：

```text
x_i^v(<=t): causal RGB prefix
u_i(<=t): synchronized privileged physical sensor stream
```

## 2.1 Physical clock

不是精确点，而是包含同步/采样误差的区间：

```text
C_phys,i = [l_phys,i, u_phys,i]
```

由事件特异 operational rule 得到，例如 contact switch、pressure onset、latch state、stable-placement criterion。

## 2.2 Visual observability clock

定义 population-level、view/query/threshold-conditioned response curve：

```text
r_i^v(t; q, Pi_h) = P(
  observer correctly verifies event state/outcome
  AND grounds it in visible evidence
  | x_i^v(<=t), q_i, observer population Pi_h
)
```

给定 threshold `eta`、置信下界和 persistence interval `d_vis`：

```text
tau_vis,i^v(eta) = inf {
  t >= u_phys,i:
  LCB[r_i^v(t')] >= eta
  for all t' in [t, t + d_vis]
}
```

最终保存 crossing-time posterior/interval：

```text
C_vis,i^v = [l_vis,i^v, u_vis,i^v]
```

必须审查：

- 这个量是否可识别；
- 是否应该用 correctness、confidence、evidence grounding 或 decision utility 定义；
- 是否能合理施加 `tau_vis >= C_phys`；
- 如何防止 anticipation 被误计为 verification；
- 人群、query wording、view 和 `eta` 改变时，该量是否仍有科学意义；
- 是否应改成 partial identification set，而不是单一 crossing time。

## 2.3 Model commit clock

```text
tau_commit,i^(m,v)
```

模型第一次写入不可回改且正确的 event verification 时刻。错误确认是 false commit；不确认是 censored miss，不能从 latency 样本删除。

视频时间与计算 wall-clock 必须分开：

```text
semantic decision delay != hardware inference latency
```

## 2.4 Proposed decomposition

```text
observability gap:
Delta_obs = tau_vis - tau_phys

residual algorithmic delay:
Delta_alg = tau_commit - tau_vis
```

区间情况下应输出 bounds 或对 crossing posterior 积分。

# 3. 候选最小方法

请把它当作待审对象，而不是正确答案。

## 3.1 Ordered multi-state observer

三个状态：

```text
S0: physical transition has not occurred
S1: transition occurred, but visual evidence is not yet sufficient
S2: transition is visually verifiable from the observed prefix
```

Pipeline：

```text
causal RGB packets
 -> frozen causal video encoder
 -> bounded temporal cache/state
 -> physical-transition hazard h_phys(t)
 -> observability hazard h_vis(t | physical occurred)
 -> class/outcome posterior
 -> calibrated deterministic commit rule
```

区间 likelihood：

```text
L_phys = -log sum_{t in C_phys} p_theta(T_phys=t)

L_vis = -log sum_{t in C_vis}
  p_theta(T_vis=t | T_vis >= T_phys)

L = L_phys + lambda_vis L_vis
  + lambda_evt L_event
  + lambda_cal L_calibration
```

提交规则在 validation set 预注册：

```text
first t such that
P(S2 and correct event | x_<=t) >= 1 - alpha_early
and credible interval width <= w
```

限制：

- 最多两个新 trainable heads；
- 第一阶段冻结 backbone；
- 不加大型 VLM、RL、生成模型或复杂 memory；
- 不宣称该 multistate survival mechanism 本身新；
- 不宣称经验 commit threshold 具有 anytime-valid guarantee。

请判断这个方法是否：

1. 足以验证任务；
2. 过于标准而无法支持论文；
3. 在统计上错误；
4. 将 `tau_vis` 与模型能力循环定义；
5. 应被更简单的 measurement-only study、isotonic model、change-point model 或 Bayesian latent-time model 替代。

# 4. 当前已发现的竞争工作

你必须打开原始论文/官方页面并继续检索 2024-2026 最新工作。不要只复述下面摘要。

## 4.1 最危险直接近邻

1. **PaSBench-Video 2026**：first visible sign、risk onset、accident boundary、causal warning、safe-scene false positives。
   <https://arxiv.org/abs/2606.02443>

2. **APT 2026**：Atomic Physical Transitions，14 类物理转移、27,303 timed instances，绑定视觉 cue、物理机制与 before/after states。
   <https://arxiv.org/abs/2606.18586>

3. **StreamReady 2026, CVPR**：annotated answer evidence windows、Answer Readiness Score、early/late penalty、readiness mechanism。
   <https://arxiv.org/abs/2603.08620>

4. **Thinking-QwenVL 2026, ICLR**：first-sufficient-evidence timestamp、progress/confidence、evidence-aligned response timing。
   <https://arxiv.org/abs/2604.18459>

5. **Ego4D PNR 2022**：PRE/CONTACT/PNR/POST 与 point-of-no-return temporal localization。
   <https://openaccess.thecvf.com/content/CVPR2022/html/Grauman_Ego4D_Around_the_World_in_3000_Hours_of_Egocentric_Video_CVPR_2022_paper.html>
   <https://ego4d-data.org/index.html>

6. **TouchMoment 2026, CVPR Findings**：精确 hand-object contact moment detection，2-frame tolerance。
   <https://arxiv.org/abs/2604.12343>

7. **VidOSC / HowToChange 2024, CVPR**：initial/transition/end 三阶段 open-world object state change localization。
   <https://openaccess.thecvf.com/content/CVPR2024/html/Xue_Learning_Object_State_Changes_in_Videos_An_Open-World_Perspective_CVPR_2024_paper.html>

8. **Action Completion 2018**：goal completion moment 与 incomplete action。
   <https://arxiv.org/abs/1805.06749>

## 4.2 物理/触觉数据与 privileged modalities

9. **FEEL 2026**：force-synchronized egocentric video，约 300 万帧。
   <https://arxiv.org/abs/2603.15847>
   <https://www.cs.umd.edu/~edessale/feel>

   重要事实：截至本 Prompt 构建时，官方项目页 Dataset/Code 按钮指向 `DATASET_URL`/`CODE_URL` 占位地址并返回 404。请重新核验，不要假设数据可用。

10. **TouchAnything / EgoTouch 2026**：208 manipulation tasks、1,891 episodes、head+dual-wrist RGB、continuous pressure maps。
    <https://arxiv.org/abs/2605.13083>

    论文写的是 “will publicly release”；请核验实际下载状态。

11. **EgoTactile 2026, ICML spotlight**：RGB + full-hand pressure supervision，显式讨论 partial-observation ambiguity。
    <https://arxiv.org/abs/2606.09243>

12. **Multimodal Distillation for Egocentric Action Recognition 2023**：训练多模态、测试 RGB，且研究 calibration。
    <https://openaccess.thecvf.com/content/ICCV2023/html/Radevski_Multimodal_Distillation_for_Egocentric_Action_Recognition_ICCV_2023_paper.html>

13. **Graph Distillation for Action Detection with Privileged Modalities 2018**。
    <https://openaccess.thecvf.com/content_ECCV_2018/html/Zelun_Luo_Graph_Distillation_for_ECCV_2018_paper.html>

## 4.3 不确定性与 latency-aware evaluation

14. **Boundary Uncertainty in TAL 2020**：Gaussian boundary distribution。
    <https://arxiv.org/abs/2008.11170>

15. **Diagnosing Error in Temporal Action Detectors 2018**：多人 temporal boundary 重标与 human agreement。
    <https://openaccess.thecvf.com/content_ECCV_2018/html/Humam_Alwassel_Diagnosing_Error_in_ECCV_2018_paper.html>

16. **Towards Streaming Perception 2020**：将 inference latency 纳入 streaming accuracy。
    <https://publications.ri.cmu.edu/towards-streaming-perception>

17. **STARE 2026**：continuous sampling、latency-aware evaluation、500 Hz annotations、model ranking reversals。
    <https://www.nature.com/articles/s41467-026-70240-6>

## 4.4 原始 Online TAL 竞争边界

请同时核验 CAG-QIL、OAT、MATR/HAT、ActionSwitch、OnPoint、OZ-TAL。不要把 causal prefix、state change、endpoint refinement、memory、immutable ledger、distillation、zero-shot 或 latency metric 当作 PIVOT 的新颖性。

# 5. 当前数据与资源约束

- 当前完整 PCEH full-packet training 曾估算为约 152,670 packets/epoch、约 7h/epoch、约 210 GPU-hours/model/seed；这些是 author-provided estimates，不是已复核公开日志。
- 新路线必须先做 measurement pilot；不得先训练大 backbone。
- 目标 P0：30-50 instances 的 `C_vis` 可重复性；再扩展至 100-300 instances。
- 第一模型阶段使用 frozen/cached features，预算 2-10 GPU-hours。
- 若公开 force/tactile data 不可下载，需要判断 100-300 个受控多视角实例的自采是否足以形成论文，而不是默认作者能采大数据。

# 6. 强制 Unknown Register 与问题澄清

在给出路线建议前，先输出一个 Unknown Register，至少覆盖以下问题：

1. FEEL/EgoTouch/EgoTactile 数据是否真实可下载、许可如何？
2. 作者是否能自采 force/contact/switch synchronized multi-view data？
3. 可用标注者数量、预算与周期？
4. 目标事件究竟是 contact、completion、outcome、state transition 还是 action endpoint？
5. 主任务是 verification 还是 anticipation？
6. `tau_vis` 的 observer population、query wording 和 threshold 如何固定？
7. 是否接受 task/evaluation paper，而不是纯模型论文？
8. 目标 venue 和 submission deadline？
9. 是否有同一物理实例的多视角数据？
10. 当前是否已有 packet-level causal logits/features 可直接复用？
11. 是否允许传感器只用于 label/measurement，不用于 test inference？
12. 实际部署错误代价：physical false early、visual-unsupported commit、miss、late commit 各是多少？

你必须：

- 将每项标为 `known`、`unknown`、`requires source verification`、`requires pilot` 或 `requires author decision`；
- 说明答案会如何改变结论；
- 提出最多 15 个高信息量问题；
- 即使用户暂未回答，也要基于显式假设继续给出条件化审查，不能停在提问列表。

# 7. 强制审查流程

严格按以下顺序，不得直接给一个自信的方法方案。

## Phase A: Repository Reality Check

- 核验 branch/HEAD；
- 列出当前代码实际能复用和不能复用的部分；
- 判断 PIVOT 应独立成新任务分支还是继续塞进 PCEH；
- 标记所有 author-provided、local-unverified 或 stale facts。

## Phase B: Task Ontology Audit

必须逐项比较：

- Online TAL/TAD；
- event spotting；
- object state change localization；
- PNR localization；
- action completion detection；
- early action recognition/anticipation；
- streaming readiness/triggered QA；
- proactive warning；
- streaming event verification。

输出：PIVOT 最准确的任务名称、输入、输出、监督、因果限制、错误类型和与上述任务的非语言学差异。

## Phase C: Identifiability and Measurement Audit

重点攻击：

- `C_phys` 是否真是语义物理事件，而不是 sensor threshold artifact；
- `C_vis` 是否循环依赖模型/标注者；
- visual evidence availability 是否应单调；
- population-level curve 是否可跨文化/经验/视角比较；
- 一个 prefix 是否应该允许 replay；
- 如何避免同一 rater 的 future/memory contamination；
- `tau_vis >= tau_phys` 是否对所有事件成立；
- anticipation 与 verification 如何分离；
- 是否应有第四个 computation/output clock；
- interval-censoring 是否足够，还是需要 partial identification。

必须给出修订后的数学定义或判定该任务不可识别。

## Phase D: Fresh Novelty Search

- 使用至少 12 组不同检索词；
- 覆盖 CVPR/ICCV/ECCV/NeurIPS/ICLR/ACL/CoRL/RA-L/IROS、arXiv 2024-2026；
- 对每个 core claim 搜方法而非只搜名称；
- 打开摘要和尽可能多的方法/实验部分；
- 给出 primary URL、年份、venue、精确 overlap、不可被简单重构的 delta；
- 明确检索盲区。

至少审查上面列出的 17 项工作。若发现更直接论文，必须提高其优先级。

## Phase E: Adversarial Kill Round

写一段至少 300 字的最强拒稿意见，标题：

```text
Why PIVOT Should Be Rejected
```

必须从以下角度攻击：

- task union；
- weak identifiability；
- subjective annotation；
- dataset scale/access；
- standard method；
- domain narrowness；
- evaluation engineering；
- lack of rank reversal；
- mismatch with target venue。

然后列出每个拒稿点需要什么证据才能被击败。不能用写作包装回答。

## Phase F: Compare Three Routes

必须比较，不得直接锁定原方案：

### Route A: PIVOT Task/Evaluation

独立 physical anchor + population visual interval + commit residual delay，方法最小。

### Route B: View-Conditioned Observability Science

把同一物理事件多视角下的 visual evidence arrival 作为主科学对象，模型只作测量工具。

### Route C: Abandon/Redirect

若 A/B 都是现有工作的组合，明确建议转向更强问题，或回到其他候选路线；不要勉强保留。

每条路线按以下 10 项打分，总分 100：

- importance 15；
- novelty 20；
- identifiability 15；
- data feasibility 10；
- falsifiability 10；
- method clarity 10；
- compute feasibility 5；
- benchmark value 5；
- top-venue narrative 5；
- reviewer defensibility 5。

出现以下情况必须扣至少 10 分：

- 贡献只是已有模块组合；
- 依赖未发布数据；
- 只有单类 contact；
- `tau_vis` 无可靠定义；
- 没有明确 rank-reversal experiment。

## Phase G: Minimal Method Review

若任务存活：

- 判断 ordered multi-state observer 是否足够；
- 给出不超过两个新 trainable components 的最小方法；
- 明确 frozen/reused/new；
- 给出公式、target construction、risk set、post-event masking、calibration 与 inference rule；
- 指出是否需要 rater model 与 video model 分离训练；
- 设计 delayed observability synthetic case：`T_phys < T_vis < T_commit`；
- 设计 direct-visibility case：`T_phys ≈ T_vis`；
- 设计 anticipation trap：视觉趋势可预测但物理事件尚未发生；
- 设计 occlusion/viewpoint case；
- 不允许用完整 GT、video duration 或 future frame 进入 model kwargs。

若方法只是标准 survival model，请直说，并说明论文必须靠什么 task evidence 生存。

## Phase H: Claim Map

最多允许两个主要 paper claims 和一个 supporting claim。每条包含：

- precise claim；
- closest prior；
- required evidence；
- baseline；
- metric；
- falsification threshold；
- unsupported wording to ban。

至少审查：

1. physical-to-visual observability gap 是否稳定存在；
2. traditional latency 是否造成错误归因或 rank reversal；
3. ordered observer 是否在 matched early-error 下形成 Pareto gain。

## Phase I: Data and Annotation Protocol

给出可执行的数据方案：

- public-data-first 与 self-collection fallback；
- event taxonomy；
- physical sensor operational rule；
- synchronization error budget；
- multi-view protocol；
- independent prefix annotation；
- rater count、prefix count、sample count；
- rater leakage prevention；
- mixed-effects/item-response model；
- subject/object/environment disjoint splits；
- ethics/privacy/licensing；
- estimated human hours and monetary cost。

必须说明 30-50、100-300 和 full-paper 三个规模分别能支持什么 claim，不能把 pilot 当最终 benchmark。

## Phase J: Experiment Loop

最多三块核心实验：

1. observability existence/reliability；
2. metric audit/rank reversal；
3. minimal ordered observer。

必须包含：

- simple baseline；
- recent direct baseline；
- oracle；
- 3-5 seeds；
- mean/std/CI；
- wall-clock and memory；
- miss/FP/early commit；
- matched-risk comparison；
- point-vs-interval、no-sensor、no-view、independent-vs-ordered ablations；
- negative result interpretation；
- exact kill criteria。

## Phase K: Training-Cost Plan

按以下阶段给预算：

```text
Stage 0 measurement only
Stage 1 frozen feature heads
Stage 2 conditional LoRA
Stage 3 full visual adaptation only if strictly justified
```

禁止默认 full-packet 训练。说明何时使用 event-centric episodes、何时必须 full chronological replay，以及如何审计 sampling bias。

## Phase L: Senior-PC Final Verdict

输出：

- `GO / HOLD / NO-GO`；
- confidence 0-1；
- task novelty /10；
- method novelty /10；
- expected CVPR/ICCV reviewer score distribution；
- 最危险 competitor；
- 一句话可防守 delta；
- 一句话最强拒稿理由；
- 是否值得放弃当前 On-TAL 主线；
- 下一步 72-hour falsification plan；
- 明确“现在绝对不要做什么”。

# 8. 强制 Baseline 与 Ablation 清单

至少审查：

```text
Simple:
- causal classifier + threshold
- endpoint-only hazard
- readiness-only head
- independent two-head model

Task-neighbor:
- Ego4D PNR localizer
- TouchMoment / event spotting
- StreamReady-style readiness
- APT transition detector
- Action Completion detector
- corrected PCEH

Oracle:
- physical sensor oracle
- human C_vis oracle
- multi-view oracle
- offline bidirectional upper bound

Evaluation/system:
- raw GT-end latency
- STARE-style compute-latency accounting
```

最低消融：

```text
- point vs interval target
- single timestamp vs population curve
- with/without rater effects
- with/without physical anchor
- independent vs ordered hazards
- with/without view conditioning
- confidence-only vs interval-aware commit
- frozen vs LoRA, only if Stage 2 is reached
```

# 9. 预注册 Kill Criteria

你可以修改阈值，但必须给出更合理理由。默认立即终止条件：

1. physical/visual gap 小于一个可靠采样 bin；
2. population `C_vis` 对 rater holdout/wording/eta 不稳定；
3. 同一 physical event 跨 view 无可重复差异；
4. raw/decomposed latency 不改变任何模型结论；
5. PaSBench/APT/StreamReady/Ego4D PNR 可直接表达全部任务；
6. sensor-at-test 才有效；
7. endpoint/readiness baseline 完全支配；
8. 公共数据不可得且最小自采不可行；
9. 仅 contact onset 有效；
10. 需要大型 VLM 或 >100 GPU-hours 才看见差异。

# 10. 禁止事项

你的回复不得：

- 未核验就声称 “first”；
- 把 paper title 相似度当查新；
- 把 sensor distillation 当新颖性；
- 把 interval-censored loss 当主创新；
- 把 `tau_vis` 当客观单点而不审计 observer dependence；
- 把 anticipation 正确预测当 verification；
- 混淆 video timestamp 和 wall-clock latency；
- 删除 miss 后只报告低 latency；
- 用未来帧、完整视频长度、GT end 或 EOF post-processing；
- 加入 generic VLM/memory/RL 让方案显得复杂；
- 在 task validity 未通过前建议昂贵训练；
- 因为当前仓库已有代码而降低创新门槛。

# 11. 强制输出格式

严格使用以下标题：

```text
A. Source and Repository Verification
B. Unknown Register and Clarification Questions
C. Task Ontology Verdict
D. Identifiability and Measurement Audit
E. Fresh Competition Map
F. Why PIVOT Should Be Rejected
G. What Evidence Could Defeat the Rejection
H. Route A/B/C Comparison
I. Revised Formal Task Definition
J. Minimal Method or Method NO-GO
K. Claim Map
L. Data and Annotation Protocol
M. Baselines and Ablations
N. Three Core Experiments
O. Training and Cost Plan
P. Reviewer-Risk Register
Q. 72-Hour Falsification Plan
R. Final GO / HOLD / NO-GO Verdict
S. Verified Primary Sources
```

每个事实结论必须区分：

```text
verified from primary source
verified from GitHub
author-provided
inference
unknown
requires experiment
```

最终不要写鼓励性总结。以最诚实的研究决策结束。

---
