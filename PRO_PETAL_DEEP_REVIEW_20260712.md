## A. Executive Verdict

```text
Provisional verdict: REVISE
Confidence: 0.86
Most likely fatal flaw:
PETAL 的大部分技术内容可被 StreamFormer/E2E-LOAD
+ MATR/ActionSwitch
+ TrackFormer/MOTR 直接重构；目前尚无不可约机制。

Most defensible surviving delta:
在完全 matched features、matched compute 和标准 completion-triggered
On-TAD 协议下，identity-bearing event-set decoder 能否持续绑定同一动作实例，
并在无需回溯式后处理时减少 fragmentation、duplicate 和 same-class overlap error。

Should any raw-video training start now: no
```

**核心裁决：问题值得研究，但 Full PETAL 当前不值得正式实现。**

标准实例级 On-TAD 确实是重要问题；我没有发现一个已经成熟公开、同时具备 raw RGB、严格因果视觉适配、实例级 fully supervised On-TAL、persistent identity 和 immutable completion emission 的单一方法。但这个“公开交集尚不成熟”的事实只是一个 literature gap，不等于方法新颖性。现有工作已经分别解决了该交集的每个主要构件：E2E-LOAD 与 StreamFormer 覆盖 raw-video causal representation；MATR、HAT、OAT 覆盖实例级 On-TAL；ActionSwitch 覆盖持续动作状态、并发和同类重叠；TrackFormer/MOTR 覆盖 persistent queries、birth/death、identity propagation 和 trajectory assignment；因果模型的 train-parallel/infer-incremental 等价、LoRA/adapter 与 cached inference 都是已知技术。([CVF Open Access][1])

因此：

* **现在没有任何 PETAL main claim 可以无条件存活。**
* 唯一应立即执行的是 **Route C：matched feature-level persistent mechanism pilot**。
* 只要 Temporal TrackFormer baseline 在 matched setting 下等价或更强，应立即转为 `NO-GO`。
* 在 novelty gate 和 feature-level mechanism gate 通过前，不得启动 raw-video、30-epoch、multi-seed 训练。

当前形态若直接投稿，我对 CVPR/ICCV 1–5 分审稿分布的估计为：

```text
1 Strong Reject: 0.28
2 Reject:        0.48
3 Borderline:    0.20
4 Weak Accept:   0.04
5 Strong Accept: 0.00
```

---

## B. Source and Repository Verification

### 检索时间边界

Prompt 写定的检索截止日是 `2026-07-12`，但本次实际审查日期是 **2026-07-11**。我完成了截至 2026-07-11 可公开访问内容的检索，不能把未来一天尚未发生的公开状态表述为已核验事实。本文按上传文档要求执行，但将 2026-07-12 视为作者目标截止日，而不是已经到达的日期。

### 仓库与提交状态

| 项目                    | 核验结果                                                        | 证据等级                               |
| --------------------- | ----------------------------------------------------------- | ---------------------------------- |
| Repository            | `yuzbo/OpenTAD_OnlineTADClean_20260702` 可访问                 | verified from GitHub target branch |
| Default/target branch | `codex/online-tad-clean-20260702`                           | verified from GitHub target branch |
| Prompt anchor         | `bfd0608b2996cba30d158d741ee193476a5078df` 可访问              | verified from GitHub target branch |
| 当前检索到的公开 HEAD         | `8165970c339fd10de9b1e6378c7261f754c3a2a0`，晚于 Prompt anchor | verified from GitHub target branch |
| 新 HEAD 的性质            | 主要新增研究史、wiki 和审查材料；没有公开 PETAL 模型实现                          | verified from GitHub target branch |
| PETAL 实验状态            | 未实现、未正式训练、未通过 novelty review                                | verified from GitHub target branch |
| 本地 finetune config    | `thumos_pceh_ontad_finetune.py` 在公开 HEAD 不存在                | author-provided / local-unverified |

最新公开研究史明确要求在本次 novelty review 与 matched feature-level pilot 通过前冻结 raw-video 正式训练。

### 当前真实计算图

公开 `thumos_pceh_ontad.py` 的真实口径是：

```text
raw RGB packet
→ frozen per-frame SigLIP2 encoder under no_grad
→ trainable causal temporal adapter
→ trainable causal projection
→ class-level PCEH heads
→ immutable emission ledger
```

它不是：

```text
raw-video fully trainable causal backbone
→ persistent event slots
→ trajectory-level instance assignment
→ prefix-parallel training
```

具体事实如下：

* `formal_training_ready=False`；
* packet size 为 8 帧，30 fps 下决策 cadence 约为 0.267 秒；
* `detach_stream_state=True`，因此没有跨 packet BPTT；
* SigLIP2 视觉塔被冻结并在 `no_grad` 下运行；
* batch size 与 stream batch size 均为 1，workers 为 0；
* 每个 packet 做一次 backward、optimizer step 和 scheduler step。

### 已核实的 PCEH correctness 缺陷

1. `active_tracks` 以类别为 key，因此每类最多表示一个 active instance；同类并发、同类紧邻或同类交叉实例无法被正确表示。

2. target builder 在同一个 `end_crossed` 时刻同时产生 endpoint 与 emission positive，随后整个 delay window 以及 late 区域仍给 emission 正向压力。endpoint 与 emission 并未形成可识别的独立学习问题。

3. decoder 发射时直接设置：

```text
end_frame = current_frame
emit_frame = current_frame
```

因此当前模型并没有独立预测动作结束位置；它预测的是“现在发射并把现在当作结束”。

4. 完整 stream GT 经 `stream_gt_segments` 和 `stream_gt_labels` 进入 detector forward，再在模型内部构造 prefix targets。训练时使用完整标注作为 label-side supervision 本身不必然违法，但这种接口使 GT taint 审计困难，PETAL 应把 target construction 移出模型可见 kwargs。

5. `is_video_end`、terminal metadata 与完整时长目前存在于数据/调度接口。非终端 packet 会清理 duration 等字段，且现有 head 看起来没有使用 EOF；但严格协议应让 model forward 完全不可见 `is_video_end` 和 terminal duration，由外部 orchestrator 负责流结束后的状态清理。

6. 通用 `OnlineEmitter` 会跳过超过 latency budget 的 current candidates，并对当前与历史发射做 suppression。这条路径若用于主评测，会使 late prediction 从错误账本消失。PCEH 专用路径似乎直接写 ledger 而未调用该 emitter，因此这是重构 baseline 的协议风险，不是已证明的 PCEH 结果污染。

### 可复用、必须重写、当前不存在

| 类别    | 内容                                                                                                                                                                                                  |
| ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 可复用   | chronological packet dataset；causal projection；bounded cache；read provenance；future perturbation audit；immutable ledger；optimizer coverage audit；THUMOS/OpenTAD scaffold                            |
| 必须重写  | PCEH class-keyed state；target builder；emission/end decoder；packet-wise optimizer loop；GT target construction boundary；late-output accounting；video-owned DDP sampler                                |
| 当前不存在 | persistent slot bank；birth/death/rearm；prefix-observable instance assignment；Temporal TrackFormer baseline；batched-vs-stepwise equivalence suite；chunk-parallel training path；matched feature pilot |

现有 tests 证明的是 chronology、cache bound、read-source provenance、ledger serialization 等 contract，不证明 mAP、identity persistence、no-NMS recall、训练效率或论文新颖性。

### 工作名冲突

`PETAL` 已被同一 TAL 领域中的 **Prior-enhanced Temporal Action Localization** 使用；它还被一个 vision-language PEFT 方法使用。这个 acronym 不应继续作为正式论文名。改名是必要的检索与识别修复，但不会提高科学新颖性。([arXiv][2])

---

## C. Unknown Register

| ID  | Unknown                                   | 状态                                   | 需要什么                                                     |
| --- | ----------------------------------------- | ------------------------------------ | -------------------------------------------------------- |
| U1  | canonical On-TAD protocol                 | requires primary-source verification | 明确 completion-triggered 与 early-forecast 两套协议            |
| U2  | exact legal emission timing               | requires author decision             | 主结果只能选一种，不得混报                                            |
| U3  | raw-video availability/licenses           | unknown                              | THUMOS/FineAction/MUSES 原视频、解码与再分发许可核查                   |
| U4  | overlap/same-class/concurrency statistics | requires code audit                  | 逐数据集实例统计                                                 |
| U5  | backbone initialization/pretraining data  | requires author decision             | 预训练数据、模型规模和许可证                                           |
| U6  | tubelet size/stride/cadence               | requires pilot                       | 决定理论最低延迟                                                 |
| U7  | maximum simultaneous instances `K`        | requires dataset audit               | train-only concurrency quantiles                         |
| U8  | chunk length/warm-up                      | requires pilot                       | 跨 chunk action 与显存折中                                     |
| U9  | trajectory assignment legality            | requires method definition           | prefix-observable matching 与 full-trajectory matching 分离 |
| U10 | DDP/video ownership                       | requires code audit                  | 当前仅单流、单 rank evaluator                                   |
| U11 | hardware/GPU budget                       | author-provided / local-unverified   | GPU 型号、VRAM、CPU、存储、并发额度                                  |
| U12 | baseline reproducibility                  | unknown                              | MATR/HAT/ActionSwitch/OAT 统一 evaluator 复现                |
| U13 | main metric/latency protocol              | requires author decision             | mAP 主指标，delay/AEDT 辅助                                    |
| U14 | whether online NMS is allowed             | partly known                         | 允许 current-only selection；禁止修改历史                         |
| U15 | remote/local implementation delta         | unknown                              | 公共 HEAD 与本地 diff、测试、配置清单                                 |

---

## D. Clarification Questions

以下是最高信息量的 12 个问题。即使暂未回答，后续分析按“默认假设”继续。

| 问题                                                | 答案会改变什么                  | 本审查默认假设                           |
| ------------------------------------------------- | ------------------------ | --------------------------------- |
| 1. 主协议是 completion-triggered 还是允许 early forecast？ | 决定合法 emission 与指标        | completion-triggered              |
| 2. 第二数据集究竟是 FineAction、MUSES 还是 MultiTHUMOS？      | 决定 overlap claim 的可检验性   | FineAction 优先                     |
| 3. 实际 GPU 型号、数量、VRAM？                             | 决定 Stage 2/3 上限          | 单卡 40–80 GB                       |
| 4. 使用何种 backbone 与预训练数据？                          | 决定公平性和 C2 claim          | 同一公开初始化，禁止额外私有预训练                 |
| 5. 原始视频及解码路径是否完整可用？                               | 决定 raw-video route 可行性   | 尚未核验                              |
| 6. tubelet 长度、stride、决策 timestamp 如何定义？           | 决定最低 acquisition delay   | 先用 8-frame cadence，另做 4/8/16      |
| 7. 每个数据集最大并发实例数是多少？                               | 决定 `K` 与 slot exhaustion | train 99.9% 分位数加 1                |
| 8. chunk length、burn-in 与 TBPTT 长度？               | 决定跨 chunk identity       | 先 128/256 tokens，burn-in 不计 loss  |
| 9. class target 从 start、midpoint 还是 end 开始监督？     | 决定 early-prefix shortcut | 标准 start 后监督，另做 end-only ablation |
| 10. DDP 是否按整视频 ownership？                         | 决定 chronology 合法性        | 每个视频只属于一个 rank                    |
| 11. 是否能统一复现 MATR/ActionSwitch/OAT features？       | 决定 matched pilot 可信度     | 必须做到同 features/evaluator          |
| 12. 本地尚未公开代码相对 HEAD 有哪些修改？                        | 决定 repo audit 是否过时       | 所有本地改动视为未核验                       |

---

## E. Canonical On-TAD Task Definition

### 任务区分

| 任务                                  | 典型输出                             | 与 PETAL 的关系         |
| ----------------------------------- | -------------------------------- | ------------------- |
| OAD                                 | 当前帧/片段动作类别                       | 不是实例级 TAD           |
| ODAS                                | 动作开始点                            | 不输出完整区间             |
| On-TAD / On-TAL                     | 在线输出 `{start,end,class,score}`   | 目标任务                |
| Offline TAL/TAD                     | 可看完整视频并全局后处理                     | 只能作上界               |
| Online TAS                          | 每帧标签序列                           | 不得转向                |
| Streaming spatio-temporal detection | 空间框/管道与类别                        | 可借机制，不是同任务          |
| Online VIS/MOT                      | 持续对象 identity 与 mask/box         | TrackFormer 类机制来源   |
| POTAL/OZ-TAL/OpenHOUSE              | point-supervised、zero-shot 或开放描述 | supervision/task 不同 |

CAG-QIL 首次形式化 On-TAL 时要求未来帧不可访问、过去 proposal 不可修改；OAT 同样明确历史 proposal 不可回改，但其 anchor model允许在动作真正结束前预测一个未来结束位置。由此可见，社区实际包含至少两种不完全一致的协议，而不是单一严格标准。([CVF Open Access][3])

### 修订后的形式化

设第 `t` 次决策时，最后一个**完全到达并可读取**的原始帧为 `a(t)`。对 tubelet 模型，通常 `a(t)≤t`，且 timestamp 必须约定为 tubelet 末端：

[
\mathcal F_t=
\sigma!\left(
x_{1:a(t)},,
S_{t-1},,
D_{<t}
\right),
]

其中：

* (S_{t-1}) 是有界、只由过去输入生成的 mutable latent state；
* (D_{<t}) 是此前已公开的 immutable detections；
* 模型不得访问总视频长度、未来 packet、EOF flag、test-time teacher、GT 或离线 prediction cache。

第 `t` 次决策可产生零个或多个新输出：

[
D_t^{new}
=========

\left{
(\hat s_j,\hat e_j,\hat c_j,p_j,\tau_j=t)
\right}.
]

任何 (d_j\in D_t^{new}) 一经公开，之后都不得更改其区间、类别、分数或删除。

### 两种合法但不可混合的协议

#### Protocol A：completion-triggered On-TAD

用于 PETAL 主结果：

[
\hat e_j \le \tau_j.
]

模型只有在认为动作已结束后才能发射。对匹配 GT，理想情况是：

[
\tau_j\ge e_i.
]

但后者是评测后的事实，不是模型运行时可强制知道的条件。若某个输出在 (\tau_j<e_i) 时被匹配到尚未完成的 GT，应被报告为 early/anticipatory output，而不能悄悄纳入 completion-only 主结果。

CAG-QIL、SimOn、ActionSwitch 与 OZ-TAL 的任务叙述基本采用“动作结束时或结束后产生实例”的语义。([CVF Open Access][4])

#### Protocol B：early-forecast On-TAL

OAT 明确允许：

[
\hat e_j>\tau_j,
]

即在动作结束前预测未来结束位置，并用 AEDT 描述提前量。它没有读取未来帧，但它做的是未来 boundary forecast。([ECVA][5])

**PETAL 不得同时使用 A 的“完成后可靠发射”叙事和 B 的负延迟收益。**

### 对六个不确定点的回答

1. **标准 On-TAD 是否严格要求 (\tau_{\text{emit}}\ge e_i)？**
   不是所有论文都严格要求。completion-triggered 系列要求这一语义；OAT 明确允许 early proposal。应把协议写进实验合同，而不是称为全社区唯一标准。

2. **允许 (\hat e>t) 吗？**
   在 OAT-style early-forecast 协议中允许，它是 anticipation/forecast，不是 future-frame leakage；在 PETAL 的 completion-triggered 主协议中应禁止。

3. **NMS、OSN、ONMS 允许什么？**

   * 禁止对历史已公开 detection 做全视频 NMS、Soft-NMS 或删除；
   * 允许在当前时刻生成的 proposals 内做 current-only NMS；
   * OSN 根据过去 proposal history 决定“当前 proposal 是否注册”，但不回改历史；
   * MATR 同样对当前 proposals 做 NMS，再与过去输出做因果 suppression；
   * “ONMS”不是统一标准名称，论文必须描述操作而不是只写缩写。([ECVA][5])

4. **不同 cadence 如何公平比较？**
   至少同时报告：

```text
raw acquisition FPS
tubelet length
tubelet stride
model decision cadence
timestamp convention
minimum acquisition delay
model wall-clock delay
```

8-frame tubelet、1-frame feature 与 64-frame segment 的“每步 delay”不能直接比较。

5. **主指标是什么？**
   主指标仍应是 dataset-standard classwise temporal mAP。辅助报告 recall/FN、duplicate、fragmentation、F1、endpoint-to-emission delay、AEDT 或 latency-budgeted AP。OAD frame-mAP 不得与 On-TAD instance-mAP 混合。CAG-QIL、OAT 与 MATR 的核心 On-TAL 结果均以 temporal mAP 为主；ActionSwitch 的 class-agnostic proposal实验额外强调同时衡量 precision 与 recall。([CVF Open Access][4])

6. **数据集协议可直接比较吗？**
   不可。THUMOS14、MUSES、FineAction、EPIC-Kitchens 与 MultiTHUMOS 的类别、标注密度、重叠结构、feature stride 和 evaluation conversion 均不同。MUSES 强调 multi-shot instances；FineAction 含密集细粒度动作；MultiTHUMOS 原始形式是 dense multi-label frame annotation，ActionSwitch 对其做了适应性评测。跨论文数字只能作背景，主结论必须来自统一代码、特征和 evaluator。([ECVA][6])

---

## F. Claim-by-Claim Audit

| Claim                            | Importance | Closest prior                          | Strongest reconstruction                        | Novelty | Current evidence             | Required evidence                                     | Verdict                             |
| -------------------------------- | ---------: | -------------------------------------- | ----------------------------------------------- | ------: | ---------------------------- | ----------------------------------------------------- | ----------------------------------- |
| C0 raw-video instance On-TAD gap |       8/10 | E2E-LOAD；StreamFormer；MATR             | StreamFormer + MATR                             |    4/10 | 狭义 gap 大体存在，但没有不可约方法         | 公平 raw/frozen distinction；完整 prior table；重构 baseline  | **REVISE**                          |
| C1 persistent-instance mechanism |       9/10 | ActionSwitch；TrackFormer               | Temporal TrackFormer + causal features          |    5/10 | 只有动机，无 matched experiment    | 同 features 下 ≥2 mAP 或 ≥20% error reduction，recall 不退化 | **REVISE；唯一主 claim 候选**             |
| C2 visual adaptation             |       7/10 | StreamFormer；E2E-LOAD；TIA；LoSA；BSP/TSP | causal StreamFormer/LoSA + ordinary On-TAL head |    2/10 | 当前视觉塔 frozen                 | tracker 固定，frozen/adapter/LoRA/top-block 对照           | **Supporting empirical claim only** |
| C3 prefix-equivalence/cost       |       6/10 | causal LM/Transformer cache；E2E-LOAD   | causal chunk forward + cache                    |    1/10 | 当前是逐 packet optimizer        | 数值等价、wall-clock、VRAM、GPU-hours                        | **Supporting property only**        |
| C4 post-processing-free          |       7/10 | DETR；TrackFormer；ActionSwitch          | set decoder + one-shot lifecycle                |    3/10 | 一个 slot 一次发射不等于全局无 duplicate | recall、cross-slot duplicate、rearm、slot exhaustion     | **Reject as main claim**            |

### 允许与禁止的 C0 表述

**允许：**

> In the audited public literature, we did not identify a mature system that jointly adapts raw visual representations under strict causal, fully supervised, instance-level On-TAL with immutable outputs.

**禁止：**

```text
the first end-to-end On-TAD
the first raw-video On-TAD
the first persistent-query On-TAD
```

SimOn 已经在 detector-internal 意义上使用 “end-to-end”；E2E-LOAD 在 raw-video OAD 意义上使用 “end-to-end”。只凭论文标题或自称无法证明 PETAL 的 first claim。([arXiv][7])

---

## G. Fresh Primary-Source Competition Map

### 检索策略

本次至少执行了以下 15 组不同检索表达，并继续追查 MATR、ActionSwitch、StreamFormer、E2E-LOAD、OnPoint 的参考文献和官方代码：

```text
C0:
1. "online temporal action localization" raw frames end-to-end
2. "online temporal action detection" trainable backbone
3. "causal temporal action detection" end-to-end

C1:
4. "streaming temporal action localization" persistent query
5. "online action instance" query tracking
6. "same-class overlap" online temporal action localization

C2:
7. "causal video backbone" online action localization
8. boundary-sensitive pretraining temporal localization
9. end-to-end TAL visual backbone adaptation

C3:
10. "prefix parallel" streaming video training
11. "train parallel infer recurrent" video detection
12. causal transformer cached inference equivalence video

C4:
13. "post-processing-free" online temporal action localization
14. online NMS OSN immutable On-TAL
15. persistent event query one-time emission
```

### Competition Map

| Paper                          | Year/Venue    | Task                                            | Input / backbone                                  | Instance persistence                                             | Post-processing                   | Exact overlap with PETAL                                         | Remaining delta                                                                          |
| ------------------------------ | ------------- | ----------------------------------------------- | ------------------------------------------------- | ---------------------------------------------------------------- | --------------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| CAG-QIL                        | ICCV 2021     | fully supervised On-TAL                         | TSN/OAD features                                  | decision/actionness queues，不是 slot identity                      | grouping；无回溯 NMS                  | 严格 online、immutable proposal、fragmentation diagnosis             | raw adaptation；set-valued persistent instances ([CVF Open Access][3])                    |
| OAT                            | ECCV 2022     | On-TAL，支持 early detection                       | pretrained feature encoder                        | 每时刻 fresh anchors                                                | current NMS + OSN                 | instance supervision、early boundary、online suppression           | persistent identity；raw joint adaptation ([ECVA][5])                                     |
| SimOn                          | 2022 preprint | On-TAL                                          | pretrained 3D-CNN features                        | recurrent context embedding；按类别 grouping                         | threshold/grouping                | history state、multi-class overlap、detector-internal E2E          | true instance identity；raw backbone ([arXiv][7])                                         |
| MATR                           | ECCV 2024     | instance-level On-TAL                           | frozen TSN/I3D                                    | memory queue，queries 每 segment 解码；无实例生命周期                        | current NMS + history suppression | set prediction、memory、boundary queries、Hungarian                 | identity-bearing persistent lifecycle ([ECVA][6])                                        |
| HAT                            | ECCV 2024     | On-TAL                                          | pretrained I3D/SlowFast features                  | long-history token，不是实例 track                                    | NMS + OSN                         | long history、future-supervised training、anchors                  | persistent instance state；raw adaptation ([ar5iv][8])                                    |
| ActionSwitch                   | ECCV 2024     | class-agnostic On-TAL                           | pretrained snippet features                       | **finite-state persistent switches**                             | state grouping                    | concurrent actions、same-class overlap、conservative state changes | class-conditioned instance set；raw adaptation；query geometry ([ar5iv][9])                |
| OnPoint                        | 2026 preprint | point-supervised online TAL                     | offline teacher → online student                  | 非 PETAL 式 identity                                               | anchor decoding                   | online TAL、offline-to-online distillation                        | supervision 不同，不能作直接覆盖 ([arXiv][10])                                                     |
| OZ-TAL                         | 2026 preprint | zero-shot/training-free On-TAL                  | off-the-shelf VLM                                 | 非 persistent supervised tracker                                  | task-specific online decoding     | strict streaming instance output                                 | 任务与 supervision 不同 ([arXiv][11])                                                         |
| E2E-LOAD                       | ICCV 2023     | **OAD**                                         | raw RGB；trainable backbone；stream buffer          | 长短期 feature state，无 action-instance identity                     | frame classification              | raw causal representation、cache、end-to-end efficiency            | instance boundaries、immutable event identity ([CVF Open Access][1])                      |
| StreamFormer                   | ICCV 2025     | streaming backbone；OAD/OVIS/VQA                 | raw frames；causal temporal attention；spatial LoRA | representation state；OVIS head另管 identity                        | task-dependent                    | causal video backbone、LoRA、multi-task temporal learning          | fully supervised instance On-TAL loss；persistent event lifecycle ([CVF Open Access][12]) |
| TrackFormer                    | CVPR 2022     | MOT                                             | frame features / trainable detector               | **autoregressive track queries、birth、death、identity assignment** | 仍可能用高阈值 NMS                       | persistent query、避免 re-detection、trajectory assignment           | 一维 event semantics；completion emission；strict On-TAL evaluator ([CVF Open Access][13])   |
| MOTR / MeMOTR / CTVIS          | 2021–2023     | MOT / online VIS                                | raw visual streams                                | query propagation、memory、track association                       | model-dependent                   | identity persistence 与在线生命周期                                     | temporal interval/event-specific target ([arXiv][14])                                    |
| BasicTAD / Re2TAL / TIA / LoSA | 2022–2025     | offline E2E TAD/TAL                             | raw video；adapter、reversible、large-scale tuning   | 通常 fresh proposals/queries                                       | offline NMS or set prediction     | raw visual adaptation、长视频训练、PEFT                                 | strict causality 与 online identity ([arXiv][15])                                         |
| OpenHOUSE                      | ICCV 2025     | hierarchical open-ended streaming understanding | VLM                                               | hierarchical event state                                         | 自定义 streaming module              | adjacent boundary detection                                      | 改变任务，不可用来支持标准 PETAL claim ([arXiv][16])                                                  |

### 检索盲区

* 2PESNet 的完整主文/官方实现未从开放 primary source 中可靠取得，不能对其全部机制作强断言。
* OnPoint 与 OZ-TAL 是很新的 2026 preprint，尚无独立复现证据。
* 部分论文没有公开完整训练代码或具有不同 annotation conversion。
* 没有检索到题名直接为 “Temporal TrackFormer for On-TAL” 的成熟公开方法；但“没有同名论文”不能消除显然组合问题。

---

## H. Strongest Multi-Paper Reconstruction

| Reconstruction                                          | 是否无需新原理即可实现 | 是否覆盖 PETAL 信号/输出                                                                  | PETAL 剩余 delta                                                 | 裁决             |
| ------------------------------------------------------- | ----------- | --------------------------------------------------------------------------------- | -------------------------------------------------------------- | -------------- |
| B1 StreamFormer + MATR                                  | 是           | causal raw features + instance queries + boundaries                               | 没有 persistent identity                                         | 必做             |
| B2 E2E-LOAD + ActionSwitch                              | 是           | raw trainable stream + persistent state + overlap                                 | 非 query-set；class/boundary耦合较弱                                 | 极危险            |
| B3 causalized LoSA/TIA + MATR                           | 基本是         | PEFT/raw adaptation + On-TAL set head                                             | 系统工程差异                                                         | 无主创新           |
| B4 Temporal TrackFormer + causal encoder                | 是           | persistent queries、birth/death、fixed identity、set competition                     | 仅剩 start-memory、end hazard、immutable completion specialization | **最危险**        |
| B5 StreamFormer + ActionSwitch + trajectory consistency | 是           | causal representation + persistent concurrent actions + trajectory regularization | query geometry 与精确边界                                           | 接近完整重构         |
| B6 current CausalTAD + persistent DETR queries          | 是           | 当前仓库基础设施 + ordinary query propagation                                             | 只有实验证据可能区分                                                     | 当前最便宜 baseline |

### Obvious-combination verdict

B4 与 B5 已经重构 Full PETAL 的大部分方法。TrackFormer 明确用 track queries 持续携带 identity、在新对象出现时 birth、在消失时移除，并通过 query self-attention 避免重检测；ActionSwitch 已在直接的 On-TAL 任务中建模 persistent switches、并发动作、同类重叠与保守状态转换。把二维 box 改成一维 temporal interval，本身不足以构成 CVPR full-paper 方法创新。([CVF Open Access][13])

PETAL 当前唯一可能的非显然 delta 是：

> **在 completion-triggered、immutable-output On-TAD 中，如何用 prefix-observable assignment 训练一个持续实例集合，使其既不依赖未来 GT 来固定早期 identity，又能在同类并发与跨 chunk 场景中保持 start evidence。**

这仍是待证明的问题，不是现有方法。

---

## I. Why PETAL-OnTAD Should Be Rejected

PETAL-OnTAD 应被拒稿，因为它没有提出新任务，方法核心又可以由几个已经非常成熟的部件直接拼出。任务仍是标准 On-TAD，所以 task novelty 为零。raw-video、strict causal visual encoding 和低成本 cache 已由 E2E-LOAD 与 StreamFormer 在 streaming/OAD 场景中充分建立；MATR、HAT 与 OAT 已经在实例级 On-TAL 中使用 memory、queries、anchors、Hungarian assignment、boundary regression 和 online suppression；ActionSwitch 已经在直接的 On-TAL 任务中针对 persistent action state、simultaneous actions、same-class overlap 与 fragmentation 给出显式解决方案。([CVF Open Access][1])

更严重的是，PETAL 所谓 persistent event query 的主体几乎就是 Temporal TrackFormer：static free queries 负责 birth，autoregressive track queries 负责 identity continuity，query interaction 减少 duplicate，track confidence 决定 death/rearm，trajectory-level identity 决定跨时间 assignment。TrackFormer 甚至明确指出 decoder self-attention 并不能完全消除 duplicate，实际仍可能需要高阈值 NMS。这直接削弱了 PETAL 的 post-processing-free claim：每个 slot 每个 lifecycle 只发射一次，并不阻止两个 slots 表示同一事件，也不阻止错误 rearm 后再次发射。([CVF Open Access][13])

prefix-parallel training 也不是方法贡献。只要模型被定义为 causal Transformer token sequence，训练时并行计算、推理时使用 KV cache 是标准 autoregressive execution property；若 slot update 真的是显式递归 (q_t=F(q_{t-1},h_t))，普通单次 Transformer forward 又并不能自动得到完全相同的 recurrent trajectory，作者必须改成 sequential scan、SSM parallel scan 或专门 block mask。无论哪种情况，它首先是正确性与系统实现问题，而不是 On-TAD 科学创新。PEFT、LoRA、adapter、activation checkpointing 和 gradient sampling 同样已有大量 offline E2E TAL 与 streaming backbone 先例。([CVF Open Access][12])

trajectory-level assignment 还有明显的 future-label shortcut 风险。若作者用完整 GT class、完整 end 和完整 trajectory 在 chunk 开头决定早期 slot identity，那么模型虽然没有在输入端读取未来帧，却从 label-side 获得了现实在线系统不可能获得的完美身份关联。该做法在宽泛监督学习意义上未必“违法”，但会使 persistent identity 的训练难度被大幅低估，并破坏“prefix-observable tracking”叙事。MATR 本身已经使用包含未来结束范围的 training target；PETAL 若再采用更强的 full-trajectory matching，只会让审稿人认为收益来自 privileged assignment，而不是新机制。([ECVA][6])

实验上，THUMOS14 只有有限训练视频，直接微调大型视觉塔非常容易把更强预训练、分辨率、stride 和参数规模混入收益。若 PETAL 同时更换 backbone、解冻视觉层、增加 persistent head、改变 cadence、改变 target 和删除 NMS，就无法判断提升来自哪里。visual adaptation claim 还会与 BSP、TSP、BasicTAD、Re2TAL、TIA 和 LoSA 直接碰撞。([arXiv][17])

最后，作者当前 raw-video route 的已提供成本约为 210 GPU-hours/model/seed，而其动机又包含低延迟和高效率。若 prefix batching 只是减少 optimizer step，却增加 activation memory、raw decode、backbone FLOPs 或总 GPU-hours，C3 就不成立。若 no-NMS 只是少发 proposal，从而降低 duplicate 但同时增加 FN，C4 也不成立。若 long actions、chunk boundaries、slot exhaustion 或 same-class overlap 使 identity 丢失，C1 不成立。若跨论文 protocol、features 和 cadence 不一致，则任何 SOTA 表格都不能支持结论。

因此，当前 PETAL 最可能被概括为：

> **“将 TrackFormer 的 persistent queries 放到 StreamFormer/E2E-LOAD 式因果视频 backbone 上，再接一个 MATR/ActionSwitch 式 On-TAL 输出头。”**

在没有 matched-feature causal diagnosis 和不可约机制之前，这是一项合理工程集成，不是一篇有充分方法新颖性的 CVPR 主会论文。

---

## J. Evidence Required to Defeat the Rejection

| 拒稿攻击                             | 必须提供的反证                                                                                                              |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Temporal TrackFormer 已等价         | 完整、强实现的 Temporal TrackFormer baseline；PETAL 在相同 features/params/schedule 下稳定显著胜出                                     |
| 收益来自更强 backbone                  | tracker 固定，frozen/adapter/LoRA/top-block paired comparison                                                           |
| persistence 没有因果作用               | frozen matched features 下 persistent 对 fresh queries 至少 +2.0 avg mAP，或 duplicate/fragmentation 相对下降 ≥20%，且 recall 不劣 |
| future-label assignment shortcut | prefix-observable matching 主方法；full-trajectory matching 仅作 ablation                                                  |
| no-NMS 只是少发检测                    | precision、recall、FN、proposal count、duplicate 同报；无外部 suppression 时仍不劣                                                 |
| long action 跨 chunk 失败           | duration-stratified 与 chunk-crossing subset；identity survival 与 start error                                          |
| same-class overlap 不成立           | 专门 subset、样本量、per-instance matching、slot identity audit                                                              |
| prefix equivalence 只是口号          | eval 下逐 prefix batched/stepwise outputs、state 与 ledger 数值一致                                                          |
| 成本没有降低                           | decode+preprocess+model wall-clock、VRAM、total GPU-hours、optimizer events 同报                                          |
| 单数据集偶然                           | THUMOS14 之外至少一个 dense/overlap benchmark 重复主要结论                                                                       |
| 协议不公平                            | 相同 feature stride、decision cadence、class universe、threshold tuning 和 evaluator                                       |
| 工程集成而非方法                         | 给出一个 Temporal TrackFormer/ActionSwitch 不能直接实现的、形式化且可消融的核心机制                                                          |

---

## K. Scientific Problem Diagnosis

待验证的因果链为：

[
\text{frozen generic features and/or window rediscovery}
\rightarrow
\text{fragmentation/duplicate/overlap/boundary error}
\rightarrow
\text{persistent raw-video joint training fixes them}.
]

这条链至少包含两个独立因素，必须做 (2\times2) 因子设计：

| Representation                  | Decoder            |
| ------------------------------- | ------------------ |
| frozen matched features         | fresh queries      |
| frozen matched features         | persistent queries |
| causally adapted representation | fresh queries      |
| causally adapted representation | persistent queries |

定义：

[
\Delta_P^{frozen}
=================

## M_{\text{persistent,frozen}}

M_{\text{fresh,frozen}},
]

[
\Delta_A^{fresh}
================

## M_{\text{fresh,adapted}}

M_{\text{fresh,frozen}},
]

[
\Delta_{\text{interaction}}
===========================

(\Delta_P^{adapted}-\Delta_P^{frozen}).
]

解释：

* `C1` 需要 (\Delta_P^{frozen}>0)；
* `C2` 需要 adaptation 在 tracker 固定时独立产生收益；
* 只有 interaction 显著，才可说 visual adaptation 特别帮助 persistent tracking；
* 只观察 Full PETAL 比 old baseline 好，无法支持任何因果 claim。

### 强制 diagnostics

1. oracle representation + fresh queries；
2. frozen representation + oracle identity；
3. persistent queries + randomly shuffled slot identity；
4. offline bidirectional backbone + causal head；
5. causal backbone + ordinary grouping；
6. persistent decoder + GT start；
7. persistent decoder + GT end；
8. persistent decoder + unlimited `K`；
9. error taxonomy：

   * class error；
   * start error；
   * end error；
   * duplicate；
   * fragmentation；
   * same-class merge；
   * overlap miss；
   * slot exhaustion；
10. duration、feature stride、boundary speed、overlap count 分层。

### 操作性错误定义

* **Duplicate**：某 GT 已被更高分 prediction 匹配后，另一个同类 prediction 与该 GT 的 tIoU 仍超过预注册阈值。
* **Fragmentation**：单个 GT 对应多个互不充分的 prediction，任何一个单独均未达到阈值，但其时间并集显著覆盖 GT。
* **Same-class adjacency**：同类相邻实例间 gap 小于预注册 cadence/receptive-field 阈值。
* **Overlap subset**：两个 GT 区间具有正交集；同类 overlap 单独统计。
* **Identity switch**：一个 GT 的训练/推理 slot 在 action lifecycle 内发生变化，或一个 slot 在未 emit/rearm 前绑定两个 GT。

若 persistent head 的平均 mAP 有提升，但 duplicate/fragmentation/overlap diagnosis 不改善，则不能保留“stop re-detecting”叙事，只能说是一般 decoder gain。

---

## L. Route A/B/C/D Comparison

评分权重按 Prompt 指定。表中是 mandatory deductions 后总分。

| Route                      | I15 | N20 | NO15 | C10 | F10 | Fa10 | Cost5 | Bench5 | Story5 | Def5 |                              扣分 |     总分 |
| -------------------------- | --: | --: | ---: | --: | --: | ---: | ----: | -----: | -----: | ---: | ------------------------------: | -----: |
| A Full PETAL               |  13 |  13 |    8 |   8 |   7 |    9 |     3 |      5 |      4 |    2 | -30：显然重构、backbone confound、成本未控 | **42** |
| B Minimal E2E On-TAD       |  13 |   8 |    7 |   9 |   9 |    9 |     4 |      5 |      3 |    4 |              -10：强预训练/视觉适配可解释收益 | **61** |
| C Persistent Feature-Level |  13 |  12 |    9 |  10 |  10 |   10 |     5 |      5 |      4 |    4 |   -10：Temporal TrackFormer 显然组合 | **72** |
| D Abandon Full PETAL       |  12 |  10 |    9 |   9 |  10 |    9 |     5 |      5 |      4 |    4 |                    -10：尚无替代方法核心 | **67** |

### 排序

```text
1. Route C — 72
2. Route D — 67
3. Route B — 61
4. Route A — 42
```

### Recommendation

**立即执行 Route C，但把它定义为 kill gate，不是论文主方法。**

* Route C 通过：再决定是否把视觉适配作为独立 supporting route。
* Route C 未通过：直接走 Route D，停止 Full PETAL。
* Route B 可以作为视觉适配对照，但单独并不足以构成新论文。
* Route A 现在不应实现。

---

## M. Technical Validity Audit

### 1. persistent query 的最小状态

[
z_t^k=
(q_t^k,,
a_t^k,,
r_t^k,,
p_t^k(c),,
P_t^k(s),,
\lambda_t^k)
]

其中：

* (q_t^k)：slot embedding；
* (a_t^k)：alive/active probability；
* (r_t^k)：emitted/refractory/rearm state；
* (p_t^k(c))：class posterior；
* (P_t^k(s))：start-time distribution；
* (\lambda_t^k)：end hazard。

不应再堆 completion head、emission head、delay head 和 calibration head。

### 2. birth、active、end、emit、rearm

```text
free
  -- birth probability + query competition -->
active
  -- first-event end hazard -->
emit exactly once
  -- fixed refractory + low foreground evidence -->
free
```

`ending` 不必作为独立长期状态；end hazard crossing 与 emit 应在同一 decision step 完成，除非实验明确证明延迟确认有独立价值。

### 3. 是否需要 explicit lifecycle classifier

需要最小的：

```text
free / active
end hazard
emitted-refractory
```

不需要一个未经证明的五状态分类器。

### 4. start 表示

首选 **distributional pointer**：

[
P_t^k(s)
\quad
s\in{t-M+1,\ldots,t,\texttt{before-memory}}.
]

它比单个 scalar 更适合：

* 长动作；
* multimodal start evidence；
* bounded memory；
* chunk 开始前已经发生的动作。

### 5. end 表示

使用离散时间 first-event hazard：

[
\lambda_t^k=P(e=t\mid e\ge t,\mathcal F_t,z_{t-1}^k).
]

发射时可预测小范围 retrospective end offset：

[
\hat e_t^k=t-\delta_t^k,\qquad \delta_t^k\ge0,
]

但不得预测 (\hat e>t) 进入 completion-triggered 主结果。

### 6. class 何时监督

标准 fully supervised 训练中，从 GT start 后使用 class label 不构成输入 future leakage。但为防止 early-prefix matching 依赖完整类别轨迹：

* birth matching cost 不使用未来 end；
* same-class identity 不能靠 class 区分；
* 提供 `class-from-start` 与 `class-at-end` ablation；
* 主文不得把早期 class certainty 解释为自然可观测事实。

### 7. trajectory matching 如何避免 future shortcut

主方法不采用一次性 full-chunk Hungarian。采用：

* 当前 step 新开始的 GT 与 free slots 做 birth matching；
* 已匹配 active slot 在后续 prefix 中固定 identity；
* GT end 只有在 end crossing 后进入 end/interval loss；
* chunk warm state 使用上一 chunk 的 identity 或 replay burn-in；
* full-trajectory assignment 只作 privileged upper bound。

### 8. overlapping/same-class assignment

训练 annotation 中每个 instance 有独立 index。新 birth 只匹配新开始的实例；active assignment 不因类别相同而合并。若同一时刻有 (m) 个新实例而 free slots 少于 (m)，溢出必须计入 slot-exhaustion FN。

### 9. long action 跨 chunk

两种合法方案：

1. **Replay burn-in**：重放前一段 features 建立 state，burn-in 不算 loss；
2. **Detached warm state**：传递 state 但截断梯度。

第二种只保证 inference state continuity，不允许声称 cross-chunk end-to-end credit assignment。

### 10. prefix-parallel token layout

若坚持全 Transformer 并行定义，可采用：

```text
[h1, q1^1,...,q1^K,
 h2, q2^1,...,q2^K,
 ...
 hL, qL^1,...,qL^K]
```

mask 允许：

* (q_t) 看 (h_{\le t})；
* (q_t) 看 (q_{<t})；
* 同时刻 slots 可通过固定顺序或单独 competition block 交互；
* 禁止任何 token 看未来时间。

但这定义的是 causal token model，不等同于任意显式 recurrence。

### 11. batched/stepwise 等价充分条件

* `eval()`；
* dropout/stochastic depth 关闭；
* 不使用跨时间 batch statistics；
* position encoding 完全一致；
* cache truncation 与 batched attention window 一致；
* deterministic routing；
* 同一 dtype 与 kernel tolerance；
* warm state 一致；
* current tubelet timestamp 一致。

建议预注册：

```text
FP32: max_abs <= 1e-5, max_rel <= 1e-4
BF16/FP16: 基于校准后阈值，初始可用 max_abs <= 2e-2
```

### 12. 参数 trainability

Stage 1：全部 frozen cached features。
Stage 2：只训练 temporal adapter 或 LoRA。
Stage 4 才允许 top-block finetuning。
不得一开始 full visual tower。

### 13. 是否真的无需 NMS

未知。只能在以下全部成立后宣称：

* 无 current NMS；
* 无 historical tIoU suppression；
* 无 classwise grouping；
* duplicate 不增加；
* recall/FN 不退化；
* two-slot duplicate 与 bad rearm 可控。

TrackFormer 本身仍需要高阈值 NMS 处理 decoder 未消除的 duplicate，这说明 query competition 不自动保证 post-processing-free。([CVF Open Access][13])

### 14. 复杂度

设视觉历史长度 (L)、slots 数 (K)、memory 长度 (M)、维度 (d)。

推荐 recurrent slot scan：

[
O(LKd^2+LK^2d)
]

模型状态：

[
O(Md+Kd).
]

若把所有 time-slot tokens 做全局 attention，最坏可接近：

[
O((LK)^2d),
]

很可能抵消 prefix batching 的成本收益。

### 15. 最多两个新组件

建议严格限制为：

1. **Persistent Event-Set Decoder**；
2. **Prefix-Observable Lifecycle Assignment**。

视觉 adapter、cache、causal backbone、ledger 都不算新方法组件。

---

## N. Revised Minimal Method or Method NO-GO

### 修订方法

暂用非论文名：

```text
Persistent Event-Set Decoder for On-TAD
```

### 模型

给定 causal features (h_t)：

[
Q_t=
D_\phi(Q_{t-1},h_t),
\qquad
Q_t={q_t^k}_{k=1}^{K}.
]

每个 slot 输出：

[
y_t^k=
\left(
p_{\text{birth}},
p_{\text{alive}},
p_{\text{class}},
P_{\text{start}},
\lambda_{\text{end}},
p_{\text{free}}
\right).
]

可见 GT 集：

[
G_t={i:s_i\le t}.
]

新生实例：

[
B_t={i:t-\Delta<s_i\le t}.
]

只对 (B_t) 与 free slots 做 birth matching：

[
\pi_t^\star
===========

\arg\min_\pi
\sum_{i\in B_t}
\left[
\alpha,C_{\text{birth}}
+
\beta,C_{\text{start}}
+
\gamma,C_{\text{competition}}
\right].
]

该 cost 不使用未来 endpoint。

一旦 (i\leftrightarrow k) 建立，直到 GT end crossing 或预测 emit 前保持固定。

### 损失

[
\mathcal L=
\lambda_b\mathcal L_{\text{birth}}
+
\lambda_a\mathcal L_{\text{alive}}
+
\lambda_s\mathcal L_{\text{start-pointer}}
+
\lambda_h\mathcal L_{\text{end-hazard}}
+
\mathbf 1[e_i\le t]
\left(
\lambda_c\mathcal L_{\text{class}}
+
\lambda_I\mathcal L_{\text{interval}}
\right)
+
\lambda_q\mathcal L_{\text{query-competition}}.
]

第一版**不加入单独 duplicate loss**。若 set competition 本身不能减少 duplicate，说明该机制尚未成立；立即加多个修补 loss 会掩盖核心失败。

### 训练伪代码

```python
# Cached-feature Stage 1
state = init_slots(K)

for chunk in chronological_video_chunks:
    features = load_cached_causal_features(chunk)

    # Burn-in rebuilds state but contributes no loss.
    for h_t in features.burn_in:
        state, _ = decoder.step(h_t, state)

    losses = []
    for h_t, prefix_labels_t in features.valid_steps:
        state, outputs_t = decoder.step(h_t, state)
        assignment_t = prefix_observable_match(
            outputs=outputs_t,
            previous_assignment=state.assignment,
            labels=prefix_labels_t,
        )
        losses.append(loss_fn(outputs_t, assignment_t, prefix_labels_t))

    optimizer.zero_grad(set_to_none=True)
    torch.stack(losses).mean().backward()
    optimizer.step()
```

### 推理伪代码

```python
state = init_slots(K)
ledger = []

for tubelet, timestamp in stream:
    h_t = causal_encoder.step(tubelet)
    state, outputs = decoder.step(h_t, state)

    for slot in ready_to_emit(outputs, state):
        detection = decode_completed_interval(slot, timestamp)
        assert detection.end <= timestamp
        ledger.append(freeze(detection))
        state.mark_emitted(slot)
```

### P0 训练执行方式

不要一开始强迫 slot head 全并行。推荐：

```text
one chunk-batched causal backbone forward
+ one lightweight sequential slot scan
+ one backward/optimizer step per chunk
```

这样已能消除 packet reopen、重复 backbone forward 和过细 optimizer events，又不会引入复杂 block-mask 错误。只有该版本机制通过后，才值得研究 SSM parallel scan 或 full autoregressive slot tokens。

### Method NO-GO 条件

以下任一成立即停止：

* Temporal TrackFormer 在 matched features 下不劣；
* persistence gain <2 mAP 且 error reduction <20%；
* recall 下降超过 1 个绝对点；
* 无 NMS 时 precision 明显崩溃；
* same-class overlap 仍需 classwise special case；
* chunk boundary identity 无法维持；
* 必须使用 full-future trajectory matching 才收敛。

---

## O. End-to-End Claim Boundary

| 实际计算图                                           | 合法名称                                             | 禁止名称                               |
| ----------------------------------------------- | ------------------------------------------------ | ---------------------------------- |
| raw input + frozen visual tower                 | raw-input detector with frozen visual encoder    | end-to-end representation learning |
| frozen visual tower + trainable temporal suffix | partial end-to-end temporal adaptation           | full visual finetuning             |
| visual adapters/LoRA updated by instance loss   | parameter-efficient end-to-end visual adaptation | full backbone finetuning           |
| top visual blocks updated                       | partial visual-tower finetuning                  | full tower adaptation              |
| all visual blocks updated                       | full visual-tower finetuning                     | —                                  |
| packet内有 gradient，跨 packet state detach         | local packet-level joint training                | long-stream end-to-end BPTT        |
| gradient 穿过 chunk boundary                      | cross-chunk BPTT                                 | —                                  |
| raw RGB merely appears in graph                 | raw-input execution                              | learned raw-video representation   |
| instance loss updates named visual params       | joint raw-video instance-loss adaptation         | 若视觉 params frozen，则不得使用            |

当前公开仓库的合法口径是：

> **raw-frame input with a frozen per-frame visual encoder and trainable causal temporal adapter/head, using detached cross-packet state.**

不是：

> full end-to-end raw-video On-TAD。

---

## P. Final Claim Map and Paper Thesis

### 当前无条件存活的 main claims

```text
none
```

### 条件性 Main Claim 1

**Exact wording**

> Under matched causal features and compute, persistent event-set decoding improves instance-level On-TAD by preserving one latent identity throughout each action lifecycle.

**Closest prior:** ActionSwitch、TrackFormer。
**Non-obvious delta:** prefix-observable one-dimensional event identity + immutable completion emission。
**Baseline:** fresh DETR/MATR queries、Temporal TrackFormer、ActionSwitch-style switches。
**Metric:** avg mAP、mAP@0.7、recall、duplicate、fragmentation、same-class overlap。
**Required result:** ≥2.0 avg mAP，或 ≥20% duplicate/fragmentation relative reduction，且 mAP/recall non-inferior。
**Falsification:** Temporal TrackFormer 置信区间覆盖或支配 PETAL。
**Negative interpretation:** persistent identity 不是 On-TAD 的主要瓶颈。
**Forbidden overclaim:** “persistent queries solve On-TAD” 或 “first persistent On-TAD”。

### 条件性 Main Claim 2

**Exact wording**

> On-TAD instance and boundary supervision provides an independent localization benefit when causally adapting a fixed pretrained visual backbone.

**Closest prior:** E2E-LOAD、StreamFormer、BSP、TSP、TIA、LoSA。
**Non-obvious delta:** On-TAD-specific causal adaptation，而不是一般视频预训练。
**Baseline:** 同 tracker 下 frozen、temporal adapter、LoRA、top-block FT。
**Metric:** avg mAP、high-tIoU、short actions、boundary error。
**Required result:** ≥2.0 avg mAP，或预注册 high-tIoU/short-action 显著收益且总体不退化。
**Falsification:** PEFT 不胜 frozen，或收益由不同预训练/stride 解释。
**Negative interpretation:** generic streaming features 已足够。
**Forbidden overclaim:** “raw input proves end-to-end learning”。

### Supporting Claim

> Chunk-batched causal training and cached incremental inference implement the same prefix function while reducing measured wall-clock and GPU-hours relative to packet-wise optimization.

必须同时通过数值等价与真实成本 gate。

### Defensible paper thesis，≤35 words

> **Under matched causal features, persistent event-set decoding preserves action identity across prefixes and emits each completed instance once, reducing fragmentation without retrospective post-processing.**

这是条件性 thesis；在 Stage 1 通过前不能作为摘要陈述。

---

## Q. Baseline and Ablation Matrix

### Direct 与 reconstructed baselines

| Baseline                                | 目的                                       |
| --------------------------------------- | ---------------------------------------- |
| CAG-QIL                                 | grouping/history baseline                |
| SimOn                                   | recurrent feature-context baseline       |
| OAT + OSN                               | fresh anchor + causal suppression        |
| MATR                                    | strongest instance-query On-TAL baseline |
| HAT                                     | long-history anchor baseline             |
| ActionSwitch                            | persistent finite-state/overlap baseline |
| OnPoint                                 | 仅在 supervision 可比子设置中参考                  |
| StreamFormer + MATR                     | C0 reconstruction                        |
| E2E-LOAD + ActionSwitch                 | raw+persistent reconstruction            |
| causal LoSA/TIA + fresh queries         | visual adaptation control                |
| Temporal TrackFormer                    | C1 novelty killer                        |
| current CausalTAD + fresh queries       | repository matched baseline              |
| current CausalTAD + persistent queries  | minimal delta                            |
| frame classifier + grouping             | simplest control                         |
| endpoint-only detector                  | lifecycle-head control                   |
| causal backbone + independent start/end | no identity control                      |

### Upper bounds

| Upper bound                    | 解释                     |
| ------------------------------ | ---------------------- |
| offline bidirectional TAD      | future-context ceiling |
| oracle instance identity       | persistence ceiling    |
| oracle action end              | emission ceiling       |
| oracle start                   | start-memory ceiling   |
| unlimited slots                | `K` ceiling            |
| full-video feature upper bound | representation ceiling |

### 必要 ablations

| Ablation                                      | 隔离的 claim                  |
| --------------------------------------------- | -------------------------- |
| persistent vs reset queries                   | C1                         |
| Temporal TrackFormer vs revised decoder       | non-obviousness            |
| trajectory vs per-step matching               | assignment contribution    |
| prefix-observable vs full-trajectory matching | future-label shortcut      |
| one slot vs multiple slots                    | overlap capacity           |
| with/without slot competition                 | duplicate control          |
| with/without explicit refractory state        | rearm error                |
| scalar vs start distribution                  | long/chunk-cross actions   |
| frozen vs adapter vs LoRA vs top-block        | C2                         |
| causal vs bidirectional                       | causality price            |
| chunk-batched vs packet-wise                  | C3                         |
| bounded vs unbounded cache                    | deployment                 |
| no suppression vs current NMS vs OSN          | C4                         |
| chunk/warm-up length                          | long action continuity     |
| tubelet stride 4/8/16                         | accuracy-latency trade-off |
| `K` sweep and `instances>K`                   | capacity failure           |

---

## R. Experiment Closure

### E0: Protocol and equivalence tests

必须全部在正式训练前通过：

1. future-frame perturbation；
2. prefix-cut invariance；
3. batched causal vs stepwise cached；
4. delayed endpoint synthetic case；
5. repeated same-class instances；
6. simultaneous same-class overlap；
7. action crossing chunk boundary；
8. action starting before memory window；
9. `instances > K`；
10. duplicate slots representing one event；
11. premature rearm；
12. no-EOF/no-duration model-input audit；
13. optimizer coverage；
14. gradient existence；
15. parameter delta；
16. immutable ledger replay。

### E1: Feature-level mechanism pilot

严格匹配：

```text
same cached features
same feature stride
same K or proposal capacity
same parameter budget
same schedule
same evaluator
same thresholds
same seeds
```

比较：

```text
Fresh MATR/DETR queries
Temporal TrackFormer
ActionSwitch-style finite-state head
Revised persistent event-set decoder
```

至少 3 seeds。Stage 1 不允许 visual finetuning。

### E2: Visual adaptation isolation

固定已经胜出的 tracker：

```text
Frozen visual representation
Temporal adapter
LoRA/PEFT
Optional top-block finetuning
```

不得同时改变：

```text
backbone family
pretraining data
resolution
frame stride
head
loss
decision cadence
```

### E3: Full benchmark

主 benchmark：THUMOS14。
第二 benchmark：优先 FineAction；若原视频和协议不可复现，则选择 MUSES。
MultiTHUMOS 只用于清晰定义的 class-agnostic/dense overlap 子实验，不能把 frame-level protocol 与 instance mAP 混为一谈。

### E4: Efficiency and deployment

必须报告：

```text
decode + preprocessing + model end-to-end FPS
tubelet acquisition delay
model compute delay
p50/p90/p95 wall-clock latency
peak VRAM
training wall-clock
total GPU-hours
optimizer events
features/tubelets per second
cache memory versus stream length
accuracy–latency–cost Pareto
```

---

## S. Statistics and Results-to-Claims Matrix

### 统计协议

* Stage 1 至少 3 seeds；关键最终结果最好 5 seeds。
* 报告 mean、std 与 95% CI：

[
\bar x
\pm
t_{0.975,n-1}\frac{s}{\sqrt n}.
]

* 对每视频结果做 paired bootstrap。
* threshold 只在 validation 上选择。
* overlap/same-class subset 必须报告样本量。
* mAP 提升必须同时报告 recall/FN。
* duplicate、fragmentation 定义在实验前固定。
* 不得只对 aggregate class mAP bootstrap，而忽略视频层相关性。

### Results-to-Claims Matrix

| Case                         | 允许 claim             | 禁止 claim                      | Decision                                 |
| ---------------------------- | -------------------- | ----------------------------- | ---------------------------------------- |
| 1. persistent win，PEFT win   | C1 + C2；C3 若成本通过     | first/full E2E                | 进入完整论文                                   |
| 2. persistent win，PEFT 不 win | 只保留 C1               | visual adaptation             | feature-level persistent paper；停止 raw FT |
| 3. persistent 不 win，PEFT win | 仅视觉适配经验结论            | persistent tracking           | PETAL NO-GO；另审 raw adaptation 是否值得       |
| 4. neither wins              | none                 | 所有 PETAL claims               | **NO-GO**                                |
| 5. accuracy win，cost lose    | 方法精度 claim           | efficiency/deployment         | 仅当精度效应大且重要才继续                            |
| 6. mAP win，但 diagnosis fail  | generic decoder gain | fragmentation/duplicate story | REVISE；通常不够 full paper                   |

### 实际显著性门槛

G1 不应只看 point estimate。建议要求：

```text
mean avg-mAP gain >= 2.0
and
paired 95% CI lower bound > 0
```

或者：

```text
duplicate/fragmentation relative reduction >= 20%
paired CI supports reduction
mAP loss <= 0.5
recall loss <= 1.0
```

---

## T. Training and GPU-Cost Plan

作者提供的当前成本：

```text
~152,670 packets / epoch
~7 hours / epoch
~6.06 packets / second
~210 GPU-hours / model / seed / 30 epochs
```

这些是 `author-provided / local-unverified`，但已足以否决默认 full-packet multi-seed 路线。

| Stage | 内容                                | GPU-hour hard gate |  预计 VRAM | 必须产出                                      |
| ----- | --------------------------------- | -----------------: | -------: | ----------------------------------------- |
| 0     | protocol/synthetic/equivalence    |                0–2 |  8–16 GB | tests + audit JSON                        |
| 1     | cached feature mechanism，3 seeds  |          ≤10 total | 12–24 GB | fresh/TrackFormer/persistent paired table |
| 2     | raw-video PEFT paired single seed |          ≤48 total | 24–48 GB | frozen vs PEFT；wall-clock                 |
| 3     | matched 3-seed benchmark          |    暂定 96–180 total | 24–48 GB | THUMOS + second dataset                   |
| 4     | top-block/full adaptation         |     额外 48–96，仅条件触发 | 40–80 GB | C2 necessity proof                        |

### 必须先修的成本面

1. 同一视频只打开一次 reader；
2. 禁止每 packet open/seek/close；
3. 移除 processor CPU roundtrip；
4. GPU-native resize/normalize；
5. video-contiguous chunk batching；
6. one backbone forward per chunk；
7. lightweight slot scan；
8. one backward/optimizer event per chunk；
9. BF16/AMP；
10. activation checkpointing；
11. DDP complete-video ownership；
12. validation 只在需要时做完整 chronological replay；
13. ETAD-style gradient/activation sampling只能在 matched accuracy 下采用。ETAD 已证明通过有选择的梯度与 proposal sampling 可显著降低 E2E TAD 训练成本，因此它是工程基线，不是 PETAL 新颖性。([arXiv][18])

### Highest acceptance lift per GPU-week

```text
Temporal TrackFormer
vs
fresh queries
vs
revised persistent event-set decoder
```

在 frozen matched features 上做 3–5 seeds，加完整 error taxonomy。它比任何 raw-video pilot 更可能决定论文生死。

---

## U. Reviewer-Risk Register

| Risk                                 | 概率 |     影响 | Mitigation / Kill                            |
| ------------------------------------ | -: | -----: | -------------------------------------------- |
| Temporal TrackFormer 等价              |  高 |     致命 | 强 baseline；若不胜立即 NO-GO                       |
| ActionSwitch 已覆盖 persistence/overlap |  高 |      高 | 证明 class-conditioned boundary identity 的额外价值 |
| raw gain 来自预训练                       |  高 |     致命 | 同初始化 paired adaptation                       |
| full-trajectory future shortcut      |  高 |      高 | prefix-observable matching                   |
| no-NMS 降低 recall                     |  高 |      高 | precision/recall/FN 同报                       |
| cross-slot duplicate                 | 中高 |      高 | competition audit；不得后加隐形 suppression         |
| long action 跨 chunk 丢 identity       | 中高 |      高 | burn-in、duration strata                      |
| `K` 不足                               |  中 |     中高 | concurrency audit、overflow metric            |
| THUMOS 过小                            |  高 |      高 | 第二 dense benchmark                           |
| protocol 不可比                         |  高 |      高 | 统一 evaluator/cadence/features                |
| raw-video 成本爆炸                       |  高 |      高 | Stage gates；≤48 GPUh paired pilot            |
| acronym collision                    | 确定 |      低 | 立即改名                                         |
| public/local drift                   |  中 |      中 | 固定 SHA、公开 diff 与 artifacts                   |
| DDP 破坏 chronology                    |  中 |      高 | full-video rank ownership                    |
| prefix equivalence 不成立               |  中 | 致命于 C3 | E0 fail-closed                               |

---

## V. 72-Hour Falsification Plan

### 0–8 小时：协议冻结

产出：

```text
PROTOCOL.md
strict evaluator
late-output retention test
no-EOF audit
duplicate/fragmentation definitions
synthetic sequence generator
```

必须修复/隔离当前通用 emitter 的 late-drop 路径。

### 8–20 小时：matched feature scaffold

* 选择一套固定公开 features；
* 实现 fresh-query baseline；
* 统一 evaluator；
* 建立 duration/overlap/same-class/concurrency statistics；
* 禁止视觉训练。

### 20–36 小时：Temporal TrackFormer

只实现必要机制：

```text
free queries
track queries
birth
fixed identity
end/death
one-time emission
```

不要加入 PETAL 专属 loss。

### 36–60 小时：3-seed pilot

比较：

```text
fresh queries
Temporal TrackFormer
revised minimal persistent decoder
```

总 GPU-hours ≤10。

### 60–72 小时：判伪报告

报告：

* avg mAP 与每 tIoU；
* recall/FN；
* duplicate/fragmentation；
* same-class overlap；
* chunk-cross；
* slot exhaustion；
* paired bootstrap；
* wall-clock；
* failure examples。

### 72-hour kill rule

满足任一即 `NO-GO`：

```text
Temporal TrackFormer >= revised decoder within uncertainty
persistent avg-mAP gain < 2.0 and error reduction < 20%
recall drops > 1.0 point
requires NMS/history suppression
same-class overlap does not improve
chunk-cross identity fails
```

**72 小时内不得启动 raw-video finetuning。**

---

## W. Conditional 30-Day Plan

### 若 72 小时 Stage 1 失败

```text
Day 4:
write NO-GO memo
archive PETAL branch
retain evaluator/tests
stop method development
```

不通过增加 duplicate loss、更多 lifecycle states、Mamba、LoRA 或更强 backbone 挽救。

### 若 Stage 1 通过

#### Days 4–8

* 5-seed feature-level confirmation；
* FineAction/MUSES 第二数据集；
* Temporal TrackFormer hyperparameter parity；
* complete error taxonomy。

#### Days 9–14

* prefix-observable matching；
* chunk-boundary burn-in；
* start-pointer ablation；
* `K` sweep；
* batched/stepwise equivalence。

#### Days 15–21

只做一个 paired raw-video gate：

```text
same tracker
frozen encoder
vs
one PEFT option
```

不同时测试 adapter、LoRA、top-block 三条路线。

#### Days 22–30

只有 G1、G2、G3、G4 全通过才：

* 扩展至 3 seeds；
* 做第二 benchmark；
* 写 claim map；
* 固定论文主表。

否则按结果矩阵缩减 claim 或停止。

---

## X. Final GO / REVISE / NO-GO Decision

```text
Final verdict: REVISE

Confidence: 0.86

Task importance /10: 7.5

Problem novelty /10: 5.0

Method novelty /10: 3.5

Non-obviousness /10: 3.0

Feasibility /10: 7.0

Compute feasibility /10: 5.5

Likely reviewer scores:
1: 28%
2: 48%
3: 20%
4: 4%
5: 0%

Most dangerous single paper:
TrackFormer: Multi-Object Tracking With Transformers

Strongest direct-task competitor:
ActionSwitch

Strongest multi-paper reconstruction:
StreamFormer
+ ActionSwitch or MATR
+ TrackFormer/MOTR-style persistent queries

Strongest rejection sentence:
PETAL is a Temporal TrackFormer placed on a causal streaming video backbone
with standard On-TAL losses, without evidence that persistence rather than
backbone strength, assignment privilege, or proposal suppression causes the gain.

Defensible delta sentence:
A prefix-observable event-set decoder may preserve one latent identity per action
until exactly one immutable completion emission under matched causal features.

Two main claims that survive:
none unconditionally

Conditional Claim 1:
matched-feature persistent event-set decoding improves mAP and specifically
reduces fragmentation/duplicate/same-class overlap versus both fresh queries
and Temporal TrackFormer.

Conditional Claim 2:
On-TAD-specific causal PEFT independently improves localization while holding
the persistent decoder, cadence, initialization and evaluator fixed.

Immediate next action:
Implement the 72-hour matched-feature fresh-query / Temporal-TrackFormer /
minimal-persistent comparison.

Do not do:
Do not implement Full PETAL.
Do not start raw-video 30-epoch training.
Do not run multi-seed PEFT.
Do not add extra lifecycle heads or duplicate losses.
Do not claim first/end-to-end/post-processing-free.
Do not use full-future trajectory assignment as the main method.
```

---

## Y. Verified Primary Sources

### Canonical On-TAL

* **CAG-QIL: Context-Aware Actionness Grouping via Q Imitation Learning for Online Temporal Action Localization**, ICCV 2021 — task definition、no-future、immutable past proposals。([CVF Open Access][3])
* **A Sliding Window Scheme for Online Temporal Action Localization / OAT**, ECCV 2022 — anchors、early prediction、OSN、AEDT。([ECVA][5])
* **SimOn: A Simple Framework for Online Temporal Action Localization**, 2022 preprint — feature-level “end-to-end” detector、past context、classwise instance grouping。([arXiv][7])
* **MATR: Online Temporal Action Localization with Memory-Augmented Transformer**, ECCV 2024 — frozen features、memory、set queries、Hungarian、online suppression。([ECVA][6])
* **HAT: History-Augmented Anchor Transformer**, ECCV 2024 — long history、pretrained features、OSN。([ar5iv][8])
* **ActionSwitch**, ECCV 2024 — persistent finite-state switches、same-class overlap、class-agnostic On-TAL。([ar5iv][9])

### 2025–2026 adjacent work

* **OnPoint**, 2026 preprint — point-supervised offline-to-online distillation；supervision 不同。([arXiv][10])
* **OZ-TAL**, 2026 preprint — zero-shot、training-free On-TAL；task 不同。([arXiv][11])
* **OpenHOUSE**, ICCV 2025 — hierarchical open-ended streaming task；不能作为标准 On-TAD 方法覆盖。([arXiv][16])

### Raw-video streaming representation

* **E2E-LOAD**, ICCV 2023 — raw RGB、trainable backbone、stream buffer、end-to-end OAD。([CVF Open Access][1])
* **StreamFormer**, ICCV 2025 — causal temporal attention、spatial LoRA、streaming representation；下游评测冻结 backbone。([CVF Open Access][12])

### Persistent tracking/query mechanisms

* **TrackFormer**, CVPR 2022 — persistent track queries、birth/death、identity assignment、duplicate control。([CVF Open Access][13])
* **MOTR**, 2021/2022 — end-to-end tracking with persistent transformer queries。([arXiv][14])
* **MeMOTR** 与 **CTVIS** — memory/query propagation 与 online video instance identity。([CVF Open Access][19])

### End-to-end TAL 与 visual adaptation

* **An Empirical Study of End-to-End Temporal Action Detection / BasicTAD**。([arXiv][15])
* **Re2TAL**, CVPR 2023。([CVF Open Access][20])
* **ETAD: Training Action Detection End to End on a Laptop**。([arXiv][18])
* **End-to-End TAD with 1B Parameters / TIA**, CVPR 2024。([CVF Open Access][21])
* **TE-TAD**, CVPR 2024。([CVF Open Access][22])
* **LoSA**, WACV 2025。([CVF Open Access][23])
* **TALLFormer**。([arXiv][24])
* **OpenTAD comprehensive study**, CVPRW 2025。([CVF Open Access][25])

### Boundary-sensitive representation learning

* **Boundary-Sensitive Pre-Training**, ICCV 2021。([arXiv][17])
* **TSP: Temporally-Sensitive Pretraining**, ICCV 2021。([arXiv][26])

**最终研究决策不变：`REVISE`。先用 ≤10 GPU-hours 的 matched feature-level Temporal TrackFormer kill test 证明 persistent mechanism 存活；否则停止 PETAL，不进入 raw-video 实现。**

[1]: https://openaccess.thecvf.com/content/ICCV2023/papers/Cao_E2E-LOAD_End-to-End_Long-form_Online_Action_Detection_ICCV_2023_paper.pdf "https://openaccess.thecvf.com/content/ICCV2023/papers/Cao_E2E-LOAD_End-to-End_Long-form_Online_Action_Detection_ICCV_2023_paper.pdf"
[2]: https://arxiv.org/abs/2211.05299 "https://arxiv.org/abs/2211.05299"
[3]: https://openaccess.thecvf.com/content/ICCV2021/html/Kang_CAG-QIL_Context-Aware_Actionness_Grouping_via_Q_Imitation_Learning_for_Online_ICCV_2021_paper.html "https://openaccess.thecvf.com/content/ICCV2021/html/Kang_CAG-QIL_Context-Aware_Actionness_Grouping_via_Q_Imitation_Learning_for_Online_ICCV_2021_paper.html"
[4]: https://openaccess.thecvf.com/content/ICCV2021/papers/Kang_CAG-QIL_Context-Aware_Actionness_Grouping_via_Q_Imitation_Learning_for_Online_ICCV_2021_paper.pdf "https://openaccess.thecvf.com/content/ICCV2021/papers/Kang_CAG-QIL_Context-Aware_Actionness_Grouping_via_Q_Imitation_Learning_for_Online_ICCV_2021_paper.pdf"
[5]: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136940640.pdf "https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136940640.pdf"
[6]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02834.pdf "https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02834.pdf"
[7]: https://arxiv.org/abs/2211.04905 "SimOn: A Simple Framework for Online Temporal Action Localization"
[8]: https://ar5iv.org/html/2408.06437v1 "https://ar5iv.org/html/2408.06437v1"
[9]: https://ar5iv.org/pdf/2407.12987 "https://ar5iv.org/pdf/2407.12987"
[10]: https://arxiv.org/abs/2607.00289?utm_source=chatgpt.com "OnPoint: Offline-to-Online Multi-Level Distillation for Point-Supervised Online Temporal Action Localization"
[11]: https://arxiv.org/abs/2605.09976?utm_source=chatgpt.com "OZ-TAL: Online Zero-Shot Temporal Action Localization"
[12]: https://openaccess.thecvf.com/content/ICCV2025/papers/Yan_Learning_Streaming_Video_Representation_via_Multitask_Training_ICCV_2025_paper.pdf "https://openaccess.thecvf.com/content/ICCV2025/papers/Yan_Learning_Streaming_Video_Representation_via_Multitask_Training_ICCV_2025_paper.pdf"
[13]: https://openaccess.thecvf.com/content/CVPR2022/papers/Meinhardt_TrackFormer_Multi-Object_Tracking_With_Transformers_CVPR_2022_paper.pdf "https://openaccess.thecvf.com/content/CVPR2022/papers/Meinhardt_TrackFormer_Multi-Object_Tracking_With_Transformers_CVPR_2022_paper.pdf"
[14]: https://arxiv.org/abs/2105.03247 "https://arxiv.org/abs/2105.03247"
[15]: https://arxiv.org/abs/2204.02932 "[2204.02932] An Empirical Study of End-to-End Temporal Action Detection"
[16]: https://arxiv.org/abs/2509.12145?utm_source=chatgpt.com "Open-ended Hierarchical Streaming Video Understanding with Vision Language Models"
[17]: https://arxiv.org/abs/2011.10830 "Boundary-sensitive Pre-training for Temporal Localization in Videos"
[18]: https://arxiv.org/abs/2205.07134 "ETAD: Training Action Detection End to End on a Laptop"
[19]: https://openaccess.thecvf.com/content/ICCV2023/html/Gao_MeMOTR_Long-Term_Memory-Augmented_Transformer_for_Multi-Object_Tracking_ICCV_2023_paper.html "https://openaccess.thecvf.com/content/ICCV2023/html/Gao_MeMOTR_Long-Term_Memory-Augmented_Transformer_for_Multi-Object_Tracking_ICCV_2023_paper.html"
[20]: https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_Re2TAL_Rewiring_Pretrained_Video_Backbones_for_Reversible_Temporal_Action_Localization_CVPR_2023_paper.html "CVPR 2023 Open Access Repository"
[21]: https://openaccess.thecvf.com/content/CVPR2024/html/Liu_End-to-End_Temporal_Action_Detection_with_1B_Parameters_Across_1000_Frames_CVPR_2024_paper.html "https://openaccess.thecvf.com/content/CVPR2024/html/Liu_End-to-End_Temporal_Action_Detection_with_1B_Parameters_Across_1000_Frames_CVPR_2024_paper.html"
[22]: https://openaccess.thecvf.com/content/CVPR2024/html/Kim_TE-TAD_Towards_Full_End-to-End_Temporal_Action_Detection_via_Time-Aligned_Coordinate_CVPR_2024_paper.html "CVPR 2024 Open Access Repository"
[23]: https://openaccess.thecvf.com/content/WACV2025/html/Gupta_LoSA_Long-Short-Range_Adapter_for_Scaling_End-to-End_Temporal_Action_Localization_WACV_2025_paper.html "https://openaccess.thecvf.com/content/WACV2025/html/Gupta_LoSA_Long-Short-Range_Adapter_for_Scaling_End-to-End_Temporal_Action_Localization_WACV_2025_paper.html"
[24]: https://arxiv.org/abs/2204.01680 "[2204.01680] TALLFormer: Temporal Action Localization with a Long-memory Transformer"
[25]: https://openaccess.thecvf.com/content/CVPR2025W/PVUW/html/Liu_OpenTAD_A_Unified_Framework_and_Comprehensive_Study_of_Temporal_Action_CVPRW_2025_paper.html "CVPR 2025 Open Access Repository"
[26]: https://arxiv.org/abs/2011.11479 "TSP: Temporally-Sensitive Pretraining of Video Encoders for Localization Tasks"
