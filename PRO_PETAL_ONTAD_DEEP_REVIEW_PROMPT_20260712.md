# Pro 深度审查 Prompt：PETAL-OnTAD 端到端持久事件跟踪

以下内容可直接完整提交给 GPT-5 Pro / GPT-5.5 Pro。建议启用联网检索或 Deep Research。不要删减仓库锚点、强制查新、组合式拒稿、实验门槛和输出格式。

---

## BEGIN PROMPT

你是一名同时具备以下角色的最高强度研究审查者：

- CVPR/ICCV/ECCV Senior PC；
- NeurIPS/ICLR 方法、优化与统计审稿人；
- Online Temporal Action Detection/Localization、Temporal Action Detection、Online Action Detection 专家；
- causal/streaming video backbone、state-space model、KV-cache 专家；
- DETR/set prediction、multi-object tracking、persistent query、online video instance segmentation 专家；
- 端到端长视频训练、PEFT、activation/gradient sampling 和系统效率专家；
- 极其警惕“把 E2E-LOAD、StreamFormer、MATR、ActionSwitch、TrackFormer 拼起来并改名”的研究编辑。

请使用中文回答，保留论文标题、技术术语、公式和 URL 的原文。

你的任务不是替作者辩护，也不是立即写一个自信的方法方案，而是回答：

> 在不修改标准 On-TAD 任务定义的前提下，PETAL-OnTAD 是否构成一个重要、可实现、不能被最近工作显然重构的研究贡献？它是否值得投入未来数月和正式 GPU 预算？

必须允许结论为：

```text
GO
REVISE
NO-GO
```

其中：

- `GO`：核心问题和至少一个主要创新 claim 存活，可以进入受控实现；
- `REVISE`：问题有价值，但当前方法或 claim 不成立，必须先重构；
- `NO-GO`：最强贡献可被已有工作直接覆盖或显然组合，不应继续投入。

不要因为作者已有代码、已有训练管线或已经讨论很久而降低门槛。

# 0. 审查锚点与证据等级

## 0.1 GitHub 仓库

请首先打开并检查：

```text
Repository:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702

Branch:
codex/online-tad-clean-20260702

Branch URL:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/tree/codex/online-tad-clean-20260702

Visible HEAD at prompt construction:
bfd0608b2996cba30d158d741ee193476a5078df

Commit URL:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/commit/bfd0608b2996cba30d158d741ee193476a5078df

Published research-history index:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/RESEARCH_HISTORY.md

Current research-wiki index:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/research-wiki/index.md

Chronological discussion timeline:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/research-wiki/discussion_timeline.md

Decision register:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/research-wiki/decision_register.md

Current PETAL idea node:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/research-wiki/ideas/petal-ontad.md

Source map:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/research-wiki/source_map.md
```

本 Prompt 构建日期：`2026-07-12`。

阅读优先级：

```text
1. RESEARCH_HISTORY.md
2. research-wiki/query_pack.md
3. research-wiki/discussion_timeline.md
4. research-wiki/decision_register.md
5. research-wiki/ideas/petal-ontad.md
6. research-wiki/source_map.md
7. target branch code
```

历史文档保留了曾经被选中后又被否决的路线。必须以 timeline 中较新的决策和 decision register 中较新的 resolution 为准，不能把旧文档里的“current best”误当成当前结论。

本地工作树包含尚未推送的研究文档、配置和代码修改。你只能把 GitHub 当前可见内容称为 `verified from GitHub`。本 Prompt 对本地状态、训练速度或新 idea 的描述必须标为：

```text
author-provided / local-unverified
```

论文、数据、代码和结果事实必须标记为以下之一：

```text
verified from primary paper
verified from official code
verified from GitHub target branch
author-provided / local-unverified
inference
unknown
requires experiment
```

## 0.2 优先审查的仓库文件

请在目标 branch 上寻找并审查至少以下区域；若文件不存在或版本不同，明确指出：

```text
AGENTS.md
RTK.md
configs/causaltad/README.md
configs/causaltad/thumos_pceh_ontad.py
configs/causaltad/thumos_pceh_endpoint_only.py
opentad/models/detectors/pceh_ontad.py
opentad/models/dense_heads/prefix_event_emission_head.py
opentad/models/backbones/online_videomae_adapter.py
opentad/models/backbones/online_siglip_adapter.py
opentad/models/projections/causal_proj.py
opentad/datasets/streaming_raw_frame.py
opentad/datasets/transforms/streaming.py
opentad/utils/online_protocol.py
opentad/utils/optimizer_audit.py
opentad/cores/train_engine.py
opentad/cores/optimizer.py
tools/smoke_pceh_stream.py
tests/test_pceh_incremental_detector.py
tests/test_online_protocol_audit.py
tests/test_streaming_raw_frame_dataset.py
```

以下本地候选配置在 Prompt 构建时存在，但可能不在可见 HEAD，必须标记为 `author-provided / local-unverified`：

```text
configs/causaltad/thumos_pceh_ontad_finetune.py
```

## 0.3 当前代码的已知角色

作者认为当前仓库可复用的只是基础设施：

- raw-frame packet dataset；
- causal projection / causal Mamba 支持；
- bounded stream state/cache；
- chronological streaming protocol；
- immutable emission ledger；
- future-perturbation/no-future audit；
- optimizer/gradient/parameter-delta audit；
- THUMOS14/OpenTAD 训练与评测骨架。

当前仓库不是 PETAL 实现，也没有 PETAL 正式实验。

此前审查指出的 PCEH 风险包括：

- endpoint 与 emission target 高度耦合；
- late prefixes 可能被重复标为 emission positive；
- class-level target 污染 repeated same-class instances；
- predicted end 曾被直接写成 emit frame；
- stream state 默认 detach，不能声称跨 packet BPTT；
- complete GT 接近 detector model kwargs；
- packet-wise 训练成本不可接受；
- formal training configs 仍标记 `formal_training_ready=False`。

请独立核验。即使这些问题已经修复，也不能把 P0 correctness repair 当作 PETAL 创新。

## 0.4 作者提供的训练成本约束

以下为 `author-provided / local-unverified`：

```text
约 152,670 packets / epoch
约 7 hours / epoch
约 6.06 packets / second
约 210 GPU-hours / model / seed for 30 epochs
```

初步归因：packet 级 video open/seek/close、CPU processor 往返、单帧小 kernel、每 packet backward/optimizer step、重复运行 192-token projection、batch size 1 和 single-stream execution。

作者不能再接受默认 full-packet、30 epochs、multi-seed 后才知道 idea 无效的路线。

# 1. 不可更改的任务边界

用户已经明确否决 Three-Clock/PIVOT，因为它把任务改成 physically anchored streaming event verification。

本研究必须留在标准 fully supervised On-TAD / On-TAL 内部。不得通过新增传感器、population observability 标注、开放式语义维护、VideoQA、online TAS 或新任务指标制造创新。

请先核验社区中最规范的 On-TAD/On-TAL 定义，并纠正以下形式化中不准确的部分。

给定未裁剪流式视频：

```text
V = {x_t}_{t=1}^T
```

在时刻 `t`，模型只能访问：

```text
F_t = sigma(x_1, ..., x_t, past model state, past immutable outputs)
```

GT action instance：

```text
psi_i = (s_i, e_i, c_i)
```

模型在检测到动作结束后产生：

```text
d_i = (s_hat_i, e_hat_i, c_hat_i, score_i, tau_emit_i)
```

硬约束：

- 不能读取未来帧；
- 不能在 inference 中读取 GT、完整视频长度或 EOF 提示；
- 已 emission 的 detection 不可修改或删除；
- late prediction 保留为 prediction，其未匹配 GT 不能从 FN 中消失；
- 训练可以使用完整 training annotation，但必须区分 causal model input 与 privileged label-side supervision；
- 测试阶段不得使用 offline teacher、raw prediction shortcut 或 video-level NMS 回改历史。

请重点核验以下不确定性：

1. 标准 On-TAD 是否严格要求 `tau_emit >= e_i`？
2. 允许预测 `e_hat > t` 吗？若允许，是否属于 anticipation 而非合法 detection？
3. online NMS、OSN、ONMS 在文献中分别允许什么操作？
4. 每帧、每 snippet、每 window 的 decision cadence 如何公平比较？
5. 标准主指标到底应是 mAP、Online mAP、F1、AEDT、delay-aware AP，还是组合报告？
6. THUMOS14、MUSES、FineAction、EPIC-Kitchens、MultiTHUMOS 的 On-TAD annotation/protocol 是否可直接比较？

不得在完成这个 Task Definition Audit 前讨论 PETAL 优点。

# 2. 待审查 Idea

## 2.1 工作名

```text
PETAL-OnTAD
Persistent Event Tracking for End-to-End
Online Temporal Action Detection
```

这是 working name。请检查 PETAL/相近 acronym 是否已被占用，但不要用改名掩盖科学问题。

候选论文标题：

```text
Stop Re-Detecting the Same Action:
End-to-End Persistent Event Tracking for Online Temporal Action Detection
```

## 2.2 一句话命题

> 现有实例级 On-TAD 大多在冻结的预提取特征上，让每个滑窗重新发现动作，再通过 grouping 或 online NMS 形成实例。PETAL 让一个 persistent event query 在严格因果 raw-video backbone 上持续表示同一动作实例，并用 prefix-parallel causal training 与 incremental cached inference 的等价执行，从原始视频到 immutable action instance 联合优化。

## 2.3 明确不声称什么

PETAL 不声称以下任何单项是新颖的：

- causal attention / causal Mamba；
- raw RGB input；
- LoRA / adapter tuning；
- memory / KV cache；
- DETR query；
- persistent tracking query；
- trajectory consistency；
- online action state；
- overlap handling；
- parallel causal training；
- end-to-end TAL；
- streaming OAD；
- prefix supervision。

候选新意只可能位于这些元素为标准实例级 On-TAD 服务时的不可约机制和证据。若最强 delta 只是“把 TrackFormer 用到一维时间轴”，必须 `NO-GO`。

# 3. 候选方法形式化

以下只是待审查设计，不是已实现事实。

## 3.1 Causal raw-video encoder

将流按 tubelet/snippet 到达：

```text
u_t = RawVideoEncoder_theta(x_{t-r+1:t})
```

严格因果 temporal encoder：

```text
h_{1:L} = CausalEncoder_theta(u_{1:L}; M_causal)
```

训练时使用 block-causal mask 在 chronological chunk 内一次计算所有位置；推理时仅编码新 tubelet，并通过 KV-cache、SSM state 或其他有界状态更新。

必须审查：

- local tubelet 是否内部偷看 decision timestamp 之后的帧；
- temporal stride 与最低可实现 latency；
- image encoder + temporal suffix 是否足以称 video backbone；
- PEFT 是否足以称 joint end-to-end adaptation；
- 是否需要至少解冻 temporal blocks 或 top spatial blocks；
- 是否可复用 StreamFormer/E2E-LOAD，而不产生不公平的大规模预训练优势。

## 3.2 Persistent event queries

维护 `K` 个 event slots：

```text
Q_t = {q_t^k}_{k=1}^K
```

每个 slot 的候选生命周期：

```text
free -> active -> ending -> emitted -> rearmed
```

更新：

```text
q_t^k = F_phi(
    q_{<t}^{1:K},
    h_{<=t},
    past immutable outputs;
    M_slot-causal
)
```

每个 active slot 输出：

```text
p_state_t^k
p_class_t^k
p_start_t^k or s_hat_t^k
p_end_t^k or e_hat_t^k
score_t^k
```

一个 slot 每个 lifecycle 最多 emission 一次。主方法目标是不依赖 classwise grouping 或 online NMS。

必须审查：

- slot 如何 birth/activate；
- slot 如何避免 identity switch；
- same-class adjacent actions 如何分开；
- overlapping actions 如何分配；
- `K` 不足时如何失败；
- emitted slot 何时 rearm；
- class posterior 在动作尚未结束时是否应监督；
- query collapse、duplicate tracks 和 dead slots；
- 动作在 chunk 开始前已开始时如何恢复状态；
- 动作跨越多个 chunk 时如何保持 identity；
- persistent slot 是否真的比 ActionSwitch finite states 或 Online VIS track queries 更强。

## 3.3 Trajectory-level assignment

候选做法是在一个 causal training chunk 内，将 GT instance `i` 固定分配给一个 slot `pi(i)`，而不是每个时间点重新 Hungarian matching：

```text
pi* = argmin_pi sum_i C_trajectory(
    predicted trajectory of slot pi(i),
    visible prefix targets of instance i
)
```

候选损失：

```text
L = lambda_state * L_state
  + lambda_cls   * L_class
  + lambda_start * L_start
  + lambda_end   * L_end
  + lambda_id    * L_trajectory_consistency
  + lambda_dup   * L_duplicate_suppression
```

请审查：

- 用完整 training GT trajectory 决定 earlier-prefix slot assignment 是否构成不当 future-label leakage；
- label-side privileged assignment 与 causal input 的合法边界；
- matching cost 是否会把未来 class/end 信息偷偷注入 earlier state；
- 是否需要 online/causal matching、teacher forcing、Viterbi、optimal transport 或 set-sequence loss；
- trajectory matching 是否只是训练工程，无法构成贡献；
- 是否有更简单且同等有效的 matching-free state supervision。

## 3.4 Prefix-parallel training and incremental inference

候选核心工程/方法性质：

```text
Train:
one chronological causal chunk
-> one batched raw-video forward
-> outputs/losses at all valid prefix endpoints
-> one backward and optimizer step

Inference:
new tubelet
-> update encoder cache and K event slots
-> optional immutable emission
```

要求在 `eval()`、相同数值精度下，对每个 prefix cut `t` 验证：

```text
f_batched(x_1:t)[t] ~= f_stepwise(x_t, state_{t-1})
```

请审查：

- 标准 causal Transformer 是否真的同时得到 persistent recurrent slot trajectory；
- token ordering / block mask 应如何设计；
- Mamba parallel scan 与 recurrent step 是否更合适；
- dropout、normalization、position encoding、cache truncation 是否破坏等价；
- truncated BPTT 如何影响 long action credit assignment；
- detached warm state 是否造成 train/inference mismatch；
- 这是否只是常规 causal-LM train-parallel/infer-recurrent 性质，不能作为创新；
- 即使 optimizer step 减少，raw-video activation memory 是否反而不可接受；
- 与 E2E-LOAD、ETAD、TallFormer、Re2TAL、TIA/AdaTAD、LoSA 的效率机制相比是否还有新意。

## 3.5 Standard On-TAD emission

Pre-end query 是模型内部 latent state，不是用户可见的 mutable detection。动作结束后，slot 输出一次：

```text
(s_hat, e_hat, c_hat, score, tau_emit)
```

输出随后 immutable。

这一区别是为了避免复活此前被降级的 CESR 路线：CESR 的 headline 是 mutable hypothesis、utility-based commit 和 trajectory metrics；PETAL 试图把 persistent trajectory 限制为标准 On-TAD 内部机制。

请判断这个区别是否实质成立，还是换名复活。

# 4. 候选 Claim Map，必须逐条否决或修订

以下不是允许直接写进论文的 claims，而是待审查对象。

## C0: Literature-gap claim

> 代表性实例级 On-TAD 方法仍依赖冻结或预提取视频特征；严格 raw-video、instance-level、causal、jointly trainable On-TAD 尚未形成成熟方法。

必须搜索反例。注意 SimOn、MATR 等论文会使用 “end-to-end” 一词，必须区分 detector-internal end-to-end 与 raw-video end-to-end。

## C1: Persistent-instance claim

> Window-by-window rediscovery 是 On-TAD fragmentation、duplicate emission、same-class merging 和 overlap error 的共同原因；persistent identity-bearing event queries 在 matched representation 下能改善实例级检测。

必须证明问题诊断成立，不能只展示模型更大。

## C2: Visual-adaptation claim

> On-TAD-specific instance/boundary loss 对 causal visual backbone 的联合适配，能提供冻结 classification features 缺失的边界和状态变化线索。

必须和 StreamFormer、BSP、TSP、offline E2E-TAD、LoSA/TIA 以及简单 top-block finetuning 比较。

## C3: Prefix-equivalence/cost claim

> Prefix-parallel causal chunk training 在保持 incremental streaming semantics 的同时，消除 overlapping packet/window 的重复计算和过细 optimizer events。

必须分别验证数学/数值等价、GPU memory、wall-clock、总 GPU-hours、throughput 和 accuracy，不得用 optimizer-step 数量代替真实成本。

## C4: Post-processing-free claim

> 每个 persistent slot 每个 lifecycle 只产生一次 detection，因此主方法无需在线 NMS 或 classwise grouping。

必须验证 precision/recall，而不是通过少发 detection 获得较少 duplicate。

最终论文最多保留：

```text
2 个 main claims
1 个 supporting system/property claim
```

请删除其余 claims，避免 kitchen-sink paper。

# 5. 已知竞争版图，必须重新联网核验

下面仅是检索起点，不是可信结论。必须搜索到 `2026-07-12`，优先 primary paper、official project、official code 和 venue proceedings。

## 5.1 直接 On-TAD / On-TAL

至少审查：

1. CAG-QIL, ICCV 2021
   https://openaccess.thecvf.com/content/ICCV2021/html/Kang_CAG-QIL_Context-Aware_Actionness_Grouping_via_Q_Imitation_Learning_for_Online_ICCV_2021_paper.html
2. 2PESNet, Pattern Recognition 2022
   https://doi.org/10.1016/j.patcog.2022.108871
3. A Sliding Window Scheme for Online Temporal Action Localization, ECCV 2022
   https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136940640.pdf
4. SimOn, 2022
   https://arxiv.org/abs/2211.04905
5. MATR, ECCV 2024
   https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02834.pdf
6. MATR official code
   https://github.com/skhcjh231/MATR_codebase
7. HAT, ECCV 2024
   https://arxiv.org/abs/2408.06437
8. ActionSwitch, ECCV 2024
   https://arxiv.org/abs/2407.12987
9. OnPoint, ECCV 2026
   https://arxiv.org/abs/2607.00289
10. OZ-TAL, 2026
    https://arxiv.org/abs/2605.09976
11. OpenHOUSE, ICCV 2025
    https://openaccess.thecvf.com/content/ICCV2025/html/Kang_Open-ended_Hierarchical_Streaming_Video_Understanding_with_Vision_Language_Models_ICCV_2025_paper.html

对每篇提取：

- exact task；
- raw frames or pre-extracted features；
- frozen/trainable backbone；
- window/frame/instance processing；
- query/anchor/state persistence；
- overlap/same-class handling；
- training loss and assignment；
- online post-processing；
- latency and compute protocol；
- public code evidence；
- 与 C0-C4 的精确 overlap。

## 5.2 Raw-video online action models

至少审查：

1. E2E-LOAD, ICCV 2023
   https://openaccess.thecvf.com/content/ICCV2023/html/Cao_E2E-LOAD_End-to-End_Long-form_Online_Action_Detection_ICCV_2023_paper.html
2. StreamFormer, ICCV 2025
   https://openaccess.thecvf.com/content/ICCV2025/html/Yan_Learning_Streaming_Video_Representation_via_Multitask_Training_ICCV_2025_paper.html
3. LSTR, GateHUB, TeSTra, MAT, Colar and recent OAD/OAA models；
4. E2E long-form OAD、stream buffer、token reuse、KV-cache、short-train/long-infer work；
5. 2025-2026 streaming video backbones and causal video representation learning。

必须回答：将 E2E-LOAD 或 StreamFormer 接到 MATR/ActionSwitch 后，PETAL 还剩什么不能被重构的机制？

## 5.3 Offline raw-video end-to-end TAL/TAD

至少审查：

1. An Empirical Study of End-to-End Temporal Action Detection / BasicTAD
   https://arxiv.org/abs/2204.02932
2. TallFormer；
3. Re2TAL, CVPR 2023
   https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_Re2TAL_Rewiring_Pretrained_Video_Backbones_for_Reversible_Temporal_Action_Localization_CVPR_2023_paper.html
4. ETAD: Training Action Detection End to End on a Laptop；
5. End-to-End TAD with 1B Parameters / TIA, CVPR 2024
   https://openaccess.thecvf.com/content/CVPR2024/html/Liu_End-to-End_Temporal_Action_Detection_with_1B_Parameters_Across_1000_Frames_CVPR_2024_paper.html
6. TE-TAD, CVPR 2024
   https://openaccess.thecvf.com/content/CVPR2024/html/Kim_TE-TAD_Towards_Full_End-to-End_Temporal_Action_Detection_via_Time-Aligned_Coordinate_CVPR_2024_paper.html
7. LoSA, WACV 2025
   https://openaccess.thecvf.com/content/WACV2025/html/Gupta_LoSA_Long-Short-Range_Adapter_for_Scaling_End-to-End_Temporal_Action_Localization_WACV_2025_paper.html
8. OpenTAD comprehensive study, CVPRW 2025
   https://openaccess.thecvf.com/content/CVPR2025W/PVUW/html/Liu_OpenTAD_A_Unified_Framework_and_Comprehensive_Study_of_Temporal_Action_CVPRW_2025_paper.html

必须回答：只把这些方法 causalize 或限制 future context，是否已经足以得到 PETAL 的全部贡献？

## 5.4 Persistent query、tracking 与 online VIS

至少审查：

- DETR / Deformable DETR set prediction；
- TrackFormer；
- MOTR / MeMOTR；
- InstanceFormer；
- CTVIS and online video instance segmentation；
- query propagation、track birth/death、trajectory matching、query interaction；
- streaming spatio-temporal action detection；
- temporal event tracking or action tube tracking。

必须构造一个最强 `Temporal TrackFormer` baseline：把空间 object boxes 替换为一维 temporal intervals，使用 persistent track queries 和 birth/death。若这个 baseline 与 PETAL 等价，结论必须是 `NO-GO` 或提出真正不可约的新机制。

## 5.5 Boundary-sensitive and causal pretraining

至少审查：

1. Boundary-Sensitive Pre-Training, ICCV 2021
   https://openaccess.thecvf.com/content/ICCV2021/html/Xu_Boundary-Sensitive_Pre-Training_for_Temporal_Localization_in_Videos_ICCV_2021_paper.html
2. Temporally-Sensitive Pretraining；
3. causal video prediction/pretraining；
4. StreamFormer multi-task pretraining；
5. online action progress/state supervision；
6. OnPoint offline-to-online distillation。

不得把“On-TAD-specific pretraining”当作默认新颖性。

# 6. 强制检索策略

对 C0-C4 每一条 claim 至少使用 3 种不同检索表达，总计至少 15 组 query。检索词不能只包含 PETAL 名称。

建议起点：

```text
"online temporal action localization" raw frames end-to-end
"online temporal action detection" trainable backbone
"streaming temporal action localization" persistent query
"online TAL" trajectory assignment
"online action instance" query tracking
"causal temporal action detection" end-to-end
"streaming DETR" temporal action localization
"persistent event query" video action
"track query" temporal interval detection
"prefix parallel" streaming video training
"train parallel infer recurrent" video detection
"causal video backbone" online action localization
"end-to-end online action detection" instance boundary
"same-class overlap" online temporal action localization
"post-processing-free" online temporal action localization
```

覆盖：

- CVPR/ICCV/ECCV 2021-2026；
- NeurIPS/ICLR/ICML 2021-2026；
- TPAMI/TIP/TCSVT/Pattern Recognition；
- arXiv 最近 12 个月；
- official GitHub repositories；
- references and citations of MATR, ActionSwitch, StreamFormer, E2E-LOAD and OnPoint。

每个重要结论给出 primary URL。明确搜索盲区、付费墙和未公开代码。

# 7. Unknown Register 与澄清问题

在提出改进方法前，建立 Unknown Register。至少覆盖：

```text
U1  canonical On-TAD protocol
U2  exact legal emission timing
U3  raw-video availability and licenses
U4  target datasets and overlap statistics
U5  backbone initialization and pretraining data
U6  tubelet size, stride and decision cadence
U7  maximum simultaneous instances K
U8  chunk length and state warm-up
U9  trajectory assignment legality
U10 DDP/video ownership and chronological batching
U11 exact hardware and GPU-hour budget
U12 baseline code reproducibility
U13 main metric and latency protocol
U14 whether online NMS is allowed
U15 current remote/local implementation delta
```

每项标记：

```text
known
unknown
requires primary-source verification
requires code audit
requires pilot
requires author decision
```

最多提出 12 个最高信息量澄清问题，并说明每个答案会如何改变结论。即使作者暂未回答，也必须基于显式假设继续给出条件化审查，不能停在问题列表。

# 8. 强制审查流程

必须严格按以下顺序执行。不得先输出完整方法设计，再回填审查。

## Phase A: Source and Repository Reality Check

- 核验 branch、HEAD、文件和当前实现边界；
- 区分 public verified 与 local unverified；
- 列出可以复用、必须重写、当前不存在的模块；
- 判断当前仓库是否真的支持 raw frame -> trainable visual parameters -> instance loss；
- 审计 optimizer coverage、`no_grad`、detach state、cache provenance、GT taint 和 DDP 限制；
- 明确 smoke/contract test 不能证明 scientific claim。

## Phase B: Task Definition Audit

对比并严格区分：

- Online Action Detection；
- Online Temporal Action Detection/Localization；
- Online Detection of Action Start；
- offline TAD/TAL；
- Online Temporal Action Segmentation；
- streaming spatio-temporal action detection；
- open-vocabulary/zero-shot On-TAL；
- tracking and Online VIS。

输出修订后的标准 On-TAD input、output、supervision、causal constraint、emission legality、error accounting 和 primary metrics。若 PETAL 暗中改变任务，立即指出。

## Phase C: Claim Decomposition

将 C0-C4 拆成最小可证伪命题。对每条输出：

- importance；
- novelty hypothesis；
- closest single work；
- strongest two/three-work reconstruction；
- required experiment；
- allowed wording；
- banned wording；
- current evidence level。

## Phase D: Fresh Novelty and Combination Audit

完成第 5-6 节要求的检索。除了单篇 overlap，还必须完成 `obvious combination test`。

强制构造并比较：

```text
B1 StreamFormer + MATR
B2 E2E-LOAD + ActionSwitch
B3 causalized LoSA/TIA + MATR
B4 Temporal TrackFormer + causal video encoder
B5 StreamFormer + ActionSwitch + trajectory consistency loss
B6 current CausalTAD + persistent DETR queries
```

对每个组合说明：

- 是否无需新原理即可实现；
- 是否产生与 PETAL 相同的训练信号和输出；
- PETAL 剩余 delta；
- delta 是否足以支撑 CCF-A full paper。

若没有非显然 delta，不得用“首次组合”挽救。

## Phase E: Adversarial Kill Round

先写至少 500 字的最强拒稿，标题必须是：

```text
Why PETAL-OnTAD Should Be Rejected
```

必须从以下角度攻击：

- task novelty is zero because task is unchanged；
- E2E-LOAD/StreamFormer already solve raw causal representation；
- MATR/HAT already solve instance On-TAD；
- ActionSwitch already solves persistent state and overlap；
- TrackFormer/MOTR already solve persistent queries and birth/death；
- prefix-parallel training is standard causal modeling；
- PEFT/LoRA is engineering；
- trajectory matching uses future GT；
- THUMOS14 is too small for visual finetuning；
- gains may come only from larger backbone/pretraining；
- no-NMS may trade recall for fewer duplicates；
- long actions/chunk boundaries break persistent identity；
- raw-video training cost contradicts low-latency motivation；
- evaluation protocols across papers are not comparable。

然后逐条列出什么证据才能击败该拒稿。不能用写作包装回答。

## Phase F: Scientific Problem Diagnosis

必须验证以下因果链，而不是直接接受：

```text
frozen generic features
and/or window rediscovery
        ->
fragmentation / duplicate / overlap / boundary error
        ->
persistent raw-video joint training fixes those errors
```

要求提出 diagnostics：

- oracle representation + fresh queries；
- frozen representation + oracle identity；
- persistent queries + shuffled identity；
- offline backbone + causal head；
- causal backbone + frame grouping；
- error taxonomy by boundary, class, duplicate, fragmentation, overlap, same-class adjacency；
- action-duration and temporal-resolution stratification。

若诊断不成立，建议更改方法或 `NO-GO`。

## Phase G: Route Comparison

不得默认 Full PETAL。比较至少四条路线：

### Route A: Full PETAL

Raw-video causal joint adaptation + persistent event queries + trajectory assignment + prefix-parallel training。

### Route B: Minimal End-to-End On-TAD

Raw-video causal backbone + matched MATR/OAT-style fresh queries，不引入 persistent tracking。

### Route C: Persistent Feature-Level On-TAD

Frozen/pre-extracted features + persistent event queries + trajectory assignment，先隔离 tracking mechanism。

### Route D: Abandon PETAL

若 PETAL 是显然组合，推荐一个仍严格位于 standard On-TAD 内部的更强问题，或者明确建议停止该方向。不得把任务转向 PIVOT、VideoQA、online TAS 或传感器任务。

每条路线按 100 分评分：

```text
importance                 15
method novelty             20
non-obviousness            15
scientific clarity         10
feasibility                10
falsifiability             10
compute efficiency          5
benchmark compatibility     5
top-venue narrative         5
reviewer defensibility      5
```

以下情况每项至少扣 10 分：

- 可由两篇现有工作直接组合；
- 主要收益来自更强预训练；
- 需要改变标准 On-TAD protocol；
- 无 matched feature/backbone baseline；
- 训练成本无法在 pilot 前控制；
- 必须依赖 online/offline NMS 才工作。

## Phase H: Method Validity Audit

若 A/C 任一路线存活，给出最小方法，而不是堆模块。

强制回答：

1. persistent queries 的最小状态是什么？
2. birth、active、end、emit、rearm 的 deterministic/probabilistic rule 是什么？
3. 是否需要 explicit lifecycle classifier？
4. start 应是 scalar、distribution、pointer 还是 memory retrieval？
5. end 是 boundary regression、hazard 还是 state transition？
6. class 何时监督，如何避免 early-prefix label leakage？
7. trajectory matching 如何避免 future-label shortcut？
8. overlapping/same-class instances 的 assignment 如何定义？
9. long action 跨 chunk 时 state 和 gradient 如何处理？
10. prefix-parallel block mask 的 token layout 是什么？
11. batched/stepwise equivalence 的充分条件是什么？
12. 哪些参数 frozen、LoRA、adapter、full trainable？
13. 主方法是否真的无需 NMS？
14. 训练和 inference 的 memory/FLOPs 随 stream length 如何增长？
15. 最多允许几个新 trainable components？建议不超过两个核心组件。

给出修订后的公式、target construction、伪代码和 complexity。若无法给出严谨定义，判 `REVISE` 或 `NO-GO`。

## Phase I: End-to-End Claim Audit

必须分别判断以下口径：

```text
raw input in the computation graph
trainable temporal suffix
trainable visual adapters / LoRA
trainable top visual blocks
full visual tower finetuning
cross-chunk BPTT
joint raw-video instance loss
```

输出一个合法命名表。禁止把：

- frozen image encoder + trainable temporal head 称为 full end-to-end；
- local packet gradient 称为 long-stream end-to-end；
- adapter-only update 称为 full visual tower finetuning；
- raw frame input 本身称为 representation learning。

## Phase J: Claim Map and Paper Story

最多保留两个 main claims 和一个 supporting claim。每条包含：

- exact wording；
- closest prior；
- non-obvious delta；
- baseline；
- metric；
- required result；
- falsification threshold；
- negative-result interpretation；
- forbidden overclaim。

给出一条不超过 35 个英文单词的 defensible paper thesis。若做不到，说明 story 尚未收敛。

## Phase K: Experimental Closure

设计从便宜到昂贵的闭环，不能直接 full training。

### E0: Protocol and equivalence tests

必须包括：

- future-frame perturbation；
- prefix cut invariance；
- batched versus stepwise equivalence；
- delayed endpoint synthetic case；
- repeated same-class synthetic case；
- overlapping instances synthetic case；
- action spanning chunk boundary；
- slot exhaustion `instances > K`；
- duplicate and rearm tests；
- optimizer/gradient/parameter-delta audit。

### E1: Feature-level mechanism pilot

同一 features、同一 parameter budget、同一 training schedule、同一 evaluator：

```text
fresh per-window queries
vs
persistent event queries
```

至少三 seeds。报告 mean/std/CI，并做 error taxonomy。

### E2: Visual adaptation isolation

保持 persistent tracker 不变：

```text
frozen backbone
vs temporal adapter
vs LoRA/PEFT
vs optional top-block finetuning
```

不能同时改变 backbone、head、resolution、stride 和 pretraining data。

### E3: Full benchmark comparison

主 benchmark：THUMOS14。

第二 benchmark 应优先验证 dense/overlap/same-class failure，例如 FineAction 或 MultiTHUMOS。MUSES 仅在数据可得且 protocol 可复现时使用。

与 OAT、MATR、HAT、ActionSwitch、SimOn、CAG-QIL、OnPoint 及 strongest reconstructed baselines 公平比较。

### E4: Efficiency and deployment

必须报告：

- video decode + preprocessing + model end-to-end FPS；
- decision cadence and acquisition delay；
- GPU memory；
- training wall-clock；
- total GPU-hours；
- optimizer steps；
- FLOPs/tubelet or measured latency；
- cache memory versus stream length；
- accuracy/latency/cost Pareto。

## Phase L: Baselines and Ablations

最低 baseline：

```text
Direct On-TAD:
- CAG-QIL
- SimOn
- OAT/OSN/ONMS variants
- MATR
- HAT
- ActionSwitch
- OnPoint where protocol is comparable

Reconstructed:
- StreamFormer + MATR-style head
- E2E-LOAD backbone + ActionSwitch-style state head
- causal LoSA/TIA + fresh query detector
- Temporal TrackFormer
- current CausalTAD + fresh queries
- current CausalTAD + persistent queries

Simple controls:
- frame classifier + grouping
- endpoint-only detector
- causal backbone + independent start/end heads
- fresh DETR queries with no persistence
- persistent queries with frozen random/standard features

Upper bounds:
- offline bidirectional TAD
- oracle identity assignment
- oracle action end
- full-video feature upper bound
```

最低 ablation：

```text
- persistent vs reset queries
- trajectory-level vs per-step matching
- one slot vs multiple slots
- with/without query competition
- with/without duplicate loss
- with/without lifecycle state
- scalar vs distributional start
- frozen vs PEFT vs optional top-block FT
- causal vs bidirectional upper bound
- prefix-parallel vs sequential training outputs
- bounded vs unbounded cache
- with/without online NMS/grouping
- chunk length and warm-up length
- tubelet stride / temporal resolution
```

每个 ablation 必须明确隔离哪个 claim。不要制作无法归因的大表。

## Phase M: Statistics and Result-to-Claim Matrix

要求：

- minimum 3 seeds，关键结果最好 5 seeds；
- mean ± std and 95% CI；
- per-video paired bootstrap where appropriate；
- matched threshold and class universe；
- recall/FN 与 latency 同报；
- duplicate/fragmentation rate 的 operational definition；
- overlap/same-class subset 的 sample size；
- no test-set threshold tuning；
- negative results must shrink claims。

给出 results-to-claims matrix，至少覆盖：

```text
Case 1 persistent head wins, PEFT wins
Case 2 persistent head wins, PEFT does not
Case 3 persistent head does not win, PEFT wins
Case 4 neither wins
Case 5 accuracy wins but cost loses
Case 6 aggregate mAP wins but overlap/duplicate diagnosis fails
```

对每种情况写：允许 claim、禁止 claim、下一步或 kill decision。

## Phase N: Training-Cost Plan

按以下阶段估算 wall-clock、GPU-hours、VRAM 和 artifact：

```text
Stage 0 protocol/equivalence only
Stage 1 cached feature mechanism pilot
Stage 2 raw-video PEFT single-seed gate
Stage 3 matched 3-seed benchmark
Stage 4 optional top-block/full adaptation only if justified
```

当前 provisional hard gates：

```text
Stage 1 <= 10 total GPU-hours
Stage 2 paired gate <= 48 total GPU-hours
no 30-epoch multi-seed raw-video run before Stage 1 and novelty pass
```

你可以修改阈值，但必须根据模型、GPU 和数据规模给出推导。必须提出 highest acceptance lift per GPU-week 的方案。

评估：

- causal chunk batching；
- video-contiguous batching；
- state warm-up；
- activation checkpointing；
- AMP/BF16；
- ETAD-style gradient sampling；
- PEFT；
- cached frozen Stage 1；
- video-reader pooling；
- GPU-native preprocessing；
- DDP stream ownership；
- full chronological validation only where required。

不得把 feature-cache result 当 raw-video end-to-end evidence。

## Phase O: Senior-PC Final Verdict

最终必须输出：

- `GO / REVISE / NO-GO`；
- confidence `0-1`；
- task importance `/10`；
- problem novelty `/10`；
- method novelty `/10`；
- non-obviousness `/10`；
- experimental feasibility `/10`；
- compute feasibility `/10`；
- expected CVPR/ICCV score distribution；
- most dangerous single paper；
- strongest multi-paper reconstruction；
- one-sentence defensible delta；
- one-sentence strongest rejection；
- two main claims that survive, or `none`；
- 72-hour falsification plan；
- next 30-day plan conditional on 72-hour result；
- what the authors must not implement or train now。

# 9. 预注册默认 Gate 与 Kill Criteria

你可以调整数值，但必须给出统计和工程理由。

## Gate G0: Novelty

至少一个 main claim 不能被任意两到三项直接 prior work 显然重构。

## Gate G1: Persistent mechanism

在完全 matched feature-level pilot 中，persistent queries 至少满足一个：

```text
average mAP improvement >= 2.0 points with uncertainty support
or
duplicate/fragmentation error relative reduction >= 20%
with non-inferior mAP and recall
```

## Gate G2: Visual adaptation

在相同 tracker 和 protocol 下，PEFT/raw-video adaptation 相对 frozen backbone 提供独立、可重复的 localization gain。默认目标：average mAP `>= 2.0` points，或在高 tIoU/short-action subset 上有预注册显著收益且总体不退化。

## Gate G3: Prefix equivalence

Eval mode 下 batched causal 与 stepwise cached 输出在声明精度容差内一致；任何 future perturbation 不改变 cut 前输出。

## Gate G4: Cost

相对当前 packet-wise route，必须同时减少 optimizer events 和实际 wall-clock/GPU-hours。仅 step 数下降而总成本上升不通过。

## Gate G5: Standard task

主结果必须使用标准 On-TAD output 和公平 protocol，不依赖新标注、future teacher at test、offline cleanup 或任务特制指标才能成立。

立即 `NO-GO` 或降级条件：

1. Temporal TrackFormer baseline 等价或支配 PETAL；
2. StreamFormer + MATR 已重构全部收益；
3. persistent queries 在 matched features 下无收益；
4. PEFT 收益完全来自更强预训练/分辨率；
5. prefix-parallel 与 incremental execution 不等价；
6. trajectory assignment 造成实质 future-label shortcut；
7. long/chunk-crossing actions 无法稳定跟踪；
8. 仍需 NMS 才能保持 precision；
9. 仅 THUMOS14 单数据集成立；
10. raw-video pilot 超预算且无明确 scaling path；
11. 主要贡献只能写成工程集成；
12. 需要改变 On-TAD task 才能讲出故事。

# 10. 禁止事项

你的回复不得：

- 未查新就使用 “first” 或 “first end-to-end On-TAD”；
- 把论文自称 end-to-end 当作 raw-video end-to-end 证据；
- 把 raw input、LoRA、cache、Mamba、persistent query 单独当创新；
- 用更强 backbone 的不公平结果证明 tracking mechanism；
- 用 feature cache 结果证明 visual adaptation；
- 混淆 OAD frame mAP 与 On-TAD instance mAP；
- 混淆 offline TAD 与 strict online TAD；
- 混淆 model video-time delay 与 wall-clock latency；
- 删除 late FP 或 missed GT 后只报告低 latency；
- 允许测试时 future frame、GT end、duration、EOF 或 teacher cache；
- 用 NMS 删除历史输出；
- 把 contract/smoke tests 当论文结果；
- 在 novelty 和 feature-level mechanism 未通过前建议正式 raw-video multi-seed training；
- 为保留作者投入而回避 `NO-GO`；
- 把方向转到 PIVOT、Online TAS、VideoQA、传感器或 broader streaming semantics。

# 11. 强制输出格式

严格使用以下标题和顺序：

```text
A. Executive Verdict
B. Source and Repository Verification
C. Unknown Register
D. Clarification Questions
E. Canonical On-TAD Task Definition
F. Claim-by-Claim Audit
G. Fresh Primary-Source Competition Map
H. Strongest Multi-Paper Reconstruction
I. Why PETAL-OnTAD Should Be Rejected
J. Evidence Required to Defeat the Rejection
K. Scientific Problem Diagnosis
L. Route A/B/C/D Comparison
M. Technical Validity Audit
N. Revised Minimal Method or Method NO-GO
O. End-to-End Claim Boundary
P. Final Claim Map and Paper Thesis
Q. Baseline and Ablation Matrix
R. Experiment Closure
S. Statistics and Results-to-Claims Matrix
T. Training and GPU-Cost Plan
U. Reviewer-Risk Register
V. 72-Hour Falsification Plan
W. Conditional 30-Day Plan
X. Final GO / REVISE / NO-GO Decision
Y. Verified Primary Sources
```

## A. Executive Verdict 必须先给

开头即给 provisional verdict，不要让 20 页分析掩盖结论：

```text
Provisional verdict:
Confidence:
Most likely fatal flaw:
Most defensible surviving delta:
Should any raw-video training start now: yes/no
```

## F. Claim-by-Claim Audit 使用表格

```text
Claim | Importance | Closest prior | Strongest reconstruction |
Novelty | Current evidence | Required evidence | Verdict
```

## G. Competition Map 使用表格

```text
Paper | Year/Venue | Task | Input | Backbone trainability |
Instance persistence | Post-processing | Exact overlap | Remaining delta | Source
```

## L. Route Comparison 使用评分表

必须给总分、排序和 recommendation，不允许四条路线都说有前景。

## P. Final Claim Map

最多两个 main claims 和一个 supporting claim。若没有，写 `none`。

## X. Final Decision

必须包含：

```text
Final verdict: GO / REVISE / NO-GO
Confidence:
Task importance /10:
Problem novelty /10:
Method novelty /10:
Non-obviousness /10:
Feasibility /10:
Compute feasibility /10:
Likely reviewer scores:
Strongest rejection sentence:
Defensible delta sentence:
Immediate next action:
Do not do:
```

不要以鼓励性总结结束。以最诚实、可执行的研究决策结束。

## END PROMPT

---
