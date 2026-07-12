# Pro 深度审核与讨论 Prompt：Persistent Event-Set Stage-1

以下内容可直接完整提交给 GPT-5 Pro / GPT-5.5 Pro。建议启用联网检索、Deep Research 和 GitHub 浏览能力。不要删减仓库锚点、证据分级、任务定义闸门、组合式拒稿、实验可识别性审查、成本闸门和互动规则。

---

## BEGIN PROMPT

你是一名同时具备以下角色的最高强度研究审查者：

- CVPR / ICCV / ECCV Senior PC；
- CCF-A 期刊高级编辑；
- fully supervised Online Temporal Action Detection / Localization 专家；
- Temporal Action Detection、Online Action Detection、Online TAS 任务边界专家；
- DETR / set prediction / Hungarian assignment 专家；
- TrackFormer / MOTR / online VIS / multi-object tracking 专家；
- causal streaming video、缓存推理和长视频训练专家；
- PyTorch、OpenTAD、训练协议、实验可复现性和统计检验专家；
- 会主动寻找“TrackFormer + MATR/ActionSwitch + causal feature cache”明显组合的严厉审稿人。

请使用中文回答，保留论文标题、模型名、公式、代码标识符和 URL 的原文。

你的任务不是替作者辩护，不是把已有代码包装成论文，也不是立即给出一个过度自信的新方法。你必须先判断：

> 当前公开仓库中的 Persistent Event-Set Stage-1 是否在代码、在线协议、对照公平性和实验可识别性上足以启动一个低成本机制实验；该实验是否有能力回答“persistent instance state 对标准 On-TAD 是否有独立价值”；如果不能，最小修订是什么；如果能，什么结果才值得继续，什么结果必须立即停止该路线？

必须允许最终结论为：

```text
RUN-STAGE1
REVISE-BEFORE-RUN
KILL-ROUTE
```

同时分别给出：

```text
Engineering verdict: PASS / FAIL / INCOMPLETE
Protocol verdict: PASS / FAIL / AMBIGUOUS
Scientific-identifiability verdict: PASS / FAIL / PARTIAL
Novelty verdict: SURVIVES / CONDITIONAL / COLLAPSED
Raw-video training permission: BLOCKED / CONDITIONAL / ALLOWED
```

在没有正式结果时，禁止把 `RUN-STAGE1` 写成方法有效或创新成立。`RUN-STAGE1` 只表示这个实验值得作为 falsification test 执行。

# 0. 公开仓库与不可变审查锚点

## 0.1 Repository

首先打开并核验：

```text
Repository:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702

Branch:
codex/online-tad-clean-20260702

Branch URL:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/tree/codex/online-tad-clean-20260702

Review anchor commit:
d6b0bca3f146046f9f6cb938f26561b32df105ed

Review anchor URL:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/commit/d6b0bca3f146046f9f6cb938f26561b32df105ed

Stage-1 implementation commit:
d06d0e992713ace3cf61f7e23c55bfc1a8afbb51

Implementation commit URL:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/commit/d06d0e992713ace3cf61f7e23c55bfc1a8afbb51
```

本 Prompt 的发布 commit 会晚于 review anchor，因为 Prompt 本身需要提交。科学代码、配置和部署记录的审查基线固定为 `d6b0bca3f146046f9f6cb938f26561b32df105ed`。如果 branch HEAD 更晚：

1. 先报告最新 HEAD；
2. 查看 `d6b0bca..HEAD` 的 diff；
3. 若只新增 Prompt/wiki provenance，可继续使用 review anchor；
4. 若科学代码、配置或结果发生变化，必须指出并改用最新相关 commit；
5. 不得把未查看的 branch HEAD 当成已验证事实。

## 0.2 必读研究记录

按以下顺序阅读：

```text
1. research-wiki/query_pack.md
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/research-wiki/query_pack.md

2. research-wiki/decision_register.md，重点 DR-024 / DR-025 / DR-026 / DR-027
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/research-wiki/decision_register.md

3. 完整上一轮 Pro 审查原文
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/d6b0bca3f146046f9f6cb938f26561b32df105ed/PRO_PETAL_DEEP_REVIEW_20260712.md

4. 对上一轮 Pro 审查的独立吸收与异议
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/d6b0bca3f146046f9f6cb938f26561b32df105ed/PRO_PETAL_DEEP_REVIEW_ABSORPTION_20260712.md

5. 当前实验状态
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/research-wiki/experiments/persistent-feature-kill-test-20260712.md

6. 当前实验设计
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/d6b0bca3f146046f9f6cb938f26561b32df105ed/research-wiki/experiments/persistent-feature-kill-test-design-20260712.md

7. 当前实施计划
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/research-wiki/experiments/persistent-feature-kill-test-plan-20260712.md

8. 已降级的原 PETAL idea 节点
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/d6b0bca3f146046f9f6cb938f26561b32df105ed/research-wiki/ideas/petal-ontad.md

9. TrackFormer novelty-threat 节点
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/d6b0bca3f146046f9f6cb938f26561b32df105ed/research-wiki/papers/trackformer2022-tracking-queries.md

10. discussion timeline 和 source map
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/research-wiki/discussion_timeline.md
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/research-wiki/source_map.md
```

第 1、2、5、7、10 项是随 Prompt 同步更新的 operational records，因此使用 branch URL；打开后必须报告当时 HEAD。其余历史材料和全部科学代码使用不可变 commit URL。若两者冲突，以 review anchor 的代码事实和最新 experiment record 的状态事实分别为准，不得混淆。

历史文档中可能仍出现“PETAL 是 lead candidate”或其他已过期表述。必须以 DR-026/DR-027、最新 experiment record 和 query pack 为准。当前事实是：

```text
Full PETAL: demoted
PETAL acronym: retired
Raw-video training: blocked
Only approved route: matched frozen-feature Stage-1 falsification test
```

## 0.3 证据等级

你的每项重要陈述必须标记以下等级之一：

```text
verified from GitHub target commit
verified from primary paper
verified from official code
author-provided remote evidence
inference
unknown
requires experiment
```

约束：

- GitHub 可见代码、配置、测试和文档可标记为 `verified from GitHub target commit`。
- Slurm 状态、私有数据、远端 cache 和未提交文件只能标记为 `author-provided remote evidence`，除非公开 artifact 已提交。
- 测试通过不等于科学 claim 通过。
- 论文摘要、二手博客、搜索摘要不能替代 primary paper。
- 找不到直接工作时，必须写明检索范围和未找到，不得把“未找到”写成“首次”。

# 1. 当前远端状态：只能作为作者提供证据

Prompt 构建时的远端状态如下：

```text
N16R4 clean worktree commit:
d06d0e992713ace3cf61f7e23c55bfc1a8afbb51

Local CPU-safe focused tests:
30 passed

Clean N16R4 combined tests:
43 passed

Synthetic integration:
one config-built train step passed
one chronological inference chunk passed
ledger-to-run-summary generation passed

Feature cache job:
Slurm job 1159510
state COMPLETED
elapsed 00:52:07
exit code 0:0

Feature cache contract audit:
411 videos
320,205 tokens total
feature dimension 768
dtype float16
per-video .npy SHA256 verified
per-video sidecar contract verified
annotation SHA256 verified

Training split:
200 videos
152,670 feature tokens
2,479 chronological chunks at chunk_size=64

Validation split:
211 videos
167,535 feature tokens
2,719 chronological chunks at chunk_size=64

GPU smoke job:
Slurm job 1159843
state COMPLETED
elapsed 00:03:46
exit code 0:0

GPU smoke gate audit:
gate_summary.json passed=true
FRESH train smoke passed, 4 chronological chunks
Temporal TrackFormer train smoke passed, 4 chronological chunks
PES train smoke passed, 4 chronological chunks
PES inference smoke passed, 4 chronological chunks
all three train routes had finite losses, non-missing gradients, changed head parameters, and slot_exhaustion=0
remote smoke test bundle: 43 passed in 36.73s

Deployment-checker incident:
the original post-run checker invoked bare `python`, which resolved to Python 2 on the login shell and failed while reading JSON
the Slurm smoke job and its reports were unaffected
review anchor d6b0bca changes the checker default to python3, permits explicit PYTHON_BIN override, and adds a launch-contract regression test
the fixed checker was re-run against job 1159843 and exited 0 while printing all five passed reports

Three-seed pilot:
not submitted; intentionally held for this Pro identifiability review

Formal scientific results:
none
```

你必须：

1. 不把以上远端状态称为 GitHub 可独立复现的结果；
2. 审查仓库是否提供足够脚本生成这些 artifact；
3. 指出哪些日志、manifest 摘要、环境 lockfile 或硬件信息必须公开；
4. 把 smoke 通过解释为“路径可执行且梯度/更新审计通过”，不得解释为 PES 有效；
5. 在没有 matched multi-seed scientific results 时，不评价 PES 的性能、统计优势或创新成立。

# 2. 不可修改的任务边界

本项目必须留在标准 fully supervised On-TAD / On-TAL 内部。不得通过以下方式制造新颖性：

- 改成 Three-Clock / PIVOT；
- 新增物理传感器或 population observability annotation；
- 改成 Online TAS、VideoQA、streaming semantic memory 或 open-ended VideoLLM；
- 改成 zero-shot / open-vocabulary 主任务；
- 允许用户可见输出无限回改；
- 用未来帧、EOF、完整视频长度或 offline NMS 回写历史结果。

候选任务形式化为：

```text
Video stream: V = {x_t}_{t=1}^T
Filtration: F_t = sigma(x_1, ..., x_t, past model state, past immutable outputs)
GT instance: psi_i = (s_i, e_i, c_i)
Emission: d_j = (s_hat_j, e_hat_j, c_hat_j, score_j, tau_emit_j)
```

当前主协议称为 `completion-triggered On-TAD`：

- 模型只能读 `F_t`；
- `e_hat_j <= tau_emit_j`；
- detection 一旦发射不可修改或删除；
- late detection 保留在 ranked prediction ledger 中；
- unmatched GT 保持 FN；
- model prediction computation 不得读取 EOF、terminal duration 或 GT；
- training 可以使用完整 annotation，但主 assignment 必须 prefix-observable；
- full-trajectory assignment 只能作为 privileged upper-bound ablation；
- 不允许 offline video-level NMS 清理历史结果。

## 2.1 Task Definition Audit

在讨论方法前，必须通过 primary papers 和 official code 回答：

1. `completion-triggered On-TAD` 是否是 MATR、ActionSwitch、CAG-QIL、OAT、SimOn 等工作的统一合法主协议？
2. 哪些工作允许 early proposal 或 future endpoint forecast？这些结果能否与当前协议直接比较？
3. `e_hat <= tau_emit` 是否必要，还是某些 On-TAL protocol 允许预测未来 endpoint？
4. online NMS、OSN、ONMS、history suppression 分别允许什么？是否会删除或修改历史输出？
5. standard mAP、OnlineAP@budget、F1、AEDT 和 latency-aware AP 中，哪些是主指标，哪些只能是补充？
6. 当前 `OnlineAPBudgeted` 是否是合理 evaluator，还是作者自定义指标可能改变排名？
7. 决策 cadence 为每 8 帧约 0.267 秒时，如何与其他 snippet/window cadence 公平比较？
8. THUMOS14 的训练/验证 split、class map 和 annotation protocol 是否与所引用 baseline 一致？

如果任务定义不能统一，必须先给出 protocol table，再讨论方法。

# 3. 当前科学问题，不是既定方法

当前唯一可辩护的问题是：

> 在完全相同的 frozen causal features、slot 数、hidden width、memory、optimizer、schedule、threshold protocol 和 evaluator 下，prefix-observable persistent instance state 是否比 fresh-query decoder 和 Temporal TrackFormer reconstruction 更好；若更好，增益是否来自 identity preservation，而不是 assignment privilege、pointer、hazard、额外容量、cache 或评测偏差？

当前没有以下 claim：

```text
No raw-video end-to-end claim
No SOTA claim
No first persistent-query claim
No first post-processing-free claim
No full PETAL claim
No raw-video PEFT permission
```

# 4. 必读代码和配置

请直接审查以下 pinned files，不要只读设计文档。

## 4.1 Configs

```text
Base:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/d6b0bca3f146046f9f6cb938f26561b32df105ed/configs/causaltad/thumos_pes_stage1_base.py

FRESH:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/d6b0bca3f146046f9f6cb938f26561b32df105ed/configs/causaltad/thumos_pes_stage1_fresh.py

Temporal TrackFormer:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/d6b0bca3f146046f9f6cb938f26561b32df105ed/configs/causaltad/thumos_pes_stage1_trackformer.py

Persistent Event-Set:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/d6b0bca3f146046f9f6cb938f26561b32df105ed/configs/causaltad/thumos_pes_stage1_persistent.py
```

## 4.2 Data and targets

```text
Streaming cached-feature dataset:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/d6b0bca3f146046f9f6cb938f26561b32df105ed/opentad/datasets/streaming_feature.py

Prefix-instance schedule:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/d6b0bca3f146046f9f6cb938f26561b32df105ed/opentad/utils/prefix_instance_schedule.py

Compatibility target export:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/d6b0bca3f146046f9f6cb938f26561b32df105ed/opentad/models/targets/prefix_instance_schedule.py
```

## 4.3 Model

```text
Persistent Event-Set head:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/d6b0bca3f146046f9f6cb938f26561b32df105ed/opentad/models/dense_heads/persistent_event_set_head.py

Persistent Event-Set detector:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/d6b0bca3f146046f9f6cb938f26561b32df105ed/opentad/models/detectors/persistent_event_set_ontad.py
```

## 4.4 Evaluation and tools

```text
Budget-aware evaluator:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/d6b0bca3f146046f9f6cb938f26561b32df105ed/opentad/evaluations/online_budgeted_map.py

Feature cache extractor:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/d6b0bca3f146046f9f6cb938f26561b32df105ed/tools/cache_ontad_features.py

Instance concurrency audit:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/d6b0bca3f146046f9f6cb938f26561b32df105ed/tools/analyze_ontad_instances.py

Bounded smoke runner:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/d6b0bca3f146046f9f6cb938f26561b32df105ed/tools/smoke_pes_stage1.py

Run summarizer:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/d6b0bca3f146046f9f6cb938f26561b32df105ed/tools/summarize_pes_stage1_run.py

Result gate:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/d6b0bca3f146046f9f6cb938f26561b32df105ed/tools/check_pes_stage1_results.py

Fail-closed Slurm submitter:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/d6b0bca3f146046f9f6cb938f26561b32df105ed/tools/remote/submit_pes_stage1_n16r4.sh

Post-run checker, including Python 3 fail-closed fix:
https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/d6b0bca3f146046f9f6cb938f26561b32df105ed/tools/remote/check_pes_stage1_n16r4.sh
```

## 4.5 Tests

至少检查：

```text
tests/test_prefix_instance_schedule.py
tests/test_streaming_feature_dataset.py
tests/test_persistent_event_set_head.py
tests/test_persistent_event_set_detector.py
tests/test_ontad_feature_cache.py
tests/test_analyze_ontad_instances.py
tests/test_pes_stage1_config_contracts.py
tests/test_pes_stage1_launch_contracts.py
tests/test_pes_stage1_result_gate.py
tests/test_pes_stage1_run_summary.py
```

不要把“测试存在”当成“测试覆盖了 claim”。必须建立 test-to-claim map。

# 5. 当前三个注册变体

共享设置：

```text
Input: frozen per-frame SigLIP2 cache
Feature policy: packet_recent_frame
Feature stride: 8 frames
Feature dim: 768
Chunk size: 64 tokens
Slots: 4
Hidden dim: 256
Memory: 192 tokens
Epochs: 12
Seeds: 705, 706, 707
Single rank
Cross-chunk state: detached
Offline NMS: forbidden
Raw-video gradients: forbidden
Duplicate repair loss: absent
```

变体：

| Variant | Assignment | Query state | Start representation | Endpoint objective |
|---|---|---|---|---|
| FRESH | per-step matching | fresh/reset | scalar offset | binary |
| TTF | prefix birth, fixed ID | persistent | scalar offset | binary |
| PES | prefix birth, fixed ID | persistent | memory pointer | first-event hazard |

## 5.1 必须严审的可识别性问题

当前三变体不是天然的一因素实验：

- FRESH → TTF 同时改变 query persistence 与 assignment persistence；
- TTF → PES 同时改变 start representation 与 endpoint objective；
- PES 的 loss mask、风险集和输出参数化也可能改变优化难度；
- FRESH 虽重置 query state，但仍共享 bounded feature memory；
- TTF 是否足够 faithful to TrackFormer，而不只是“persistent query 开关”？

必须回答：

1. 当前三变体能否识别 `persistent instance state` 的独立效应？
2. 如果不能，最小补充矩阵是什么？
3. 是否至少需要以下 controlled variants 中的一部分：

```text
fresh + prefix assignment + scalar + binary
persistent + per-step assignment + scalar + binary
persistent + prefix assignment + pointer + binary
persistent + prefix assignment + scalar + hazard
```

4. 在 10 GPU-hour硬预算内，哪两个补充变体信息增益最大？
5. 哪些变化是 baseline fidelity 所必需，哪些属于 proposed mechanism？
6. 如果完整 factorial 太贵，如何用 staged ablation 或 successive halving 保持可识别性？

不得接受“反正 PES 最终分数更高”作为机制证据。

# 6. 代码正确性与在线协议审计

请按 P0 / P1 / P2 / P3 输出 findings。每条必须包含：

```text
severity
file and line or function
observed behavior
why it matters scientifically
minimal reproducer or missing test
minimal correction
whether correction changes the registered experiment
```

## 6.1 Dataset and cache

重点检查：

- `packet_recent_frame` 是否严格因果；
- terminal short packet 的 source timestamp 是否正确；
- OpenCV sequential decode 是否可能丢帧或 frame count 不一致；
- annotation `frame` 与真实 decoded frame count 不一致时如何处理；
- cache manifest、annotation hash、dtype、feature dim、source frames 和 per-video sidecar 是否 fail closed；
- resume 是否可能混用不同 encoder revision、processor、weights 或 image preprocessing；
- cache 只记录 model path 而未记录 exact model file hash / transformers version，是否足够复现；
- 单帧 SigLIP2 缺乏 motion cue，是否使机制 pilot 失真；
- 使用弱特征是否可能让所有 query 机制一起失败，从而错误杀死 persistence；
- 是否需要第二套标准 TSN/I3D/MATR feature cache 作为低成本 robustness check；
- 训练与验证 cache 是否完整且无 test-label 驱动选择。

## 6.2 Prefix schedule and GT taint

重点检查：

- birth 只在 `previous_frame < start <= current_frame` 出现一次；
- active target 不暴露未来 endpoint；
- endpoint 只在 first crossing 出现一次；
- short action 在同一决策间隔内 start/end 时是否正确；
- same-class instance ID 是否独立；
- chunk boundary 是否丢 birth/end；
- dataset 使用完整 GT 生成稳定 instance ID 是否构成 privileged assignment；
- main method 的 prefix matching 是否真的不使用 future endpoint；
- training 使用 oracle GT start 决定 birth 时刻，而 inference 必须预测 birth，是否形成严重 train/test mismatch；
- class/start supervision 从何时开始最合理；
- full-trajectory assignment 应如何作为明确 upper bound，而不是偷偷进入主方法。

## 6.3 Head and state

重点检查：

- fresh query 是否真的忽略 carried query state；
- persistent query 与 feature memory 是否被公平控制；
- query、memory、slot state 在 chunk 内和 chunk 间如何传播；
- cross-chunk detach 后可以声称什么，不能声称什么；
- `before_memory` pointer bin 在 decode 时只能返回最早 memory frame，是否真正解决 pre-memory start；
- scalar start 的 feature-step/frame 单位是否一致；
- active false birth 的 release 规则是否过于脆弱；
- refractory rule 是否会制造重复、漏检或无法快速处理紧邻同类动作；
- alive/end/birth threshold 固定为 0.5 是否需要 calibration；
- class score、end score 和 final score 的组合是否有依据；
- slot attention 是否会导致 identity swapping；
- head 中所有 slots 每步更新、而 training slot status 仅用于 loss assignment，是否足以称为 identity-bearing state；
- inference lifecycle 与 training assignment 是否存在 exposure bias。

## 6.4 Hungarian assignment and losses

重点检查：

- assignment 是全局 Hungarian 而不是 greedy；
- cost 中 class、presence、start 三项无权重是否合理；
- cost detach 是否正确；
- birth assignment 使用 GT class/start 是否给 persistent variants 特权；
- TTF 和 PES 的 assignment 是否真正 matched；
- FRESH 的 per-step matching 是否过强或过弱；
- hazard risk set 是否 instance-aware；
- endpoint positive 是否 first-emission target；
- binary baseline 与 hazard variant 的负样本 mask 不同是否构成不公平；
- class/start loss 在每个 active prefix 重复计算是否使长动作权重过大；
- 无动作背景和 false birth 是否被充分监督；
- loss weight 是否需要预注册，能否在看到结果后调参。

## 6.5 Inference and EOF boundary

重点检查：

- `stream_control.is_video_end` 虽不在 model meta 中，但 detector `forward` 仍收到它；这是否满足严格 no-EOF prediction claim；
- prediction logits 是否在读取 `is_video_end` 前完成；
- EOF 是否只用于 return 后清理 state；
- source frame 不超过 decision frame；
- emitted endpoint 不超过 emit frame；
- ledger 是否真正 immutable；
- active slot 在视频结束但未 emit 时如何计为 FN；
- late rows 是否保留；
- 不运行 NMS 时，随机/重复 slot 是否会淹没 AP；
- threshold 是否固定于 validation 前，还是会隐性调参。

## 6.6 Evaluator and diagnostics

重点检查：

- `OnlineAPBudgeted` 的 TP/FP/FN 和 latency matching 是否正确；
- late prediction 是否同时保持 FP 和对应 GT 的 FN；
- duplicate prediction 是否计 FP；
- primary latency 必须使用 `emit_time - matched_GT_end`，不能用 `emit_time - predicted_end` 替代；
- `average_mOnlineAP` 对预算和 tIoU 简单平均是否有文献依据；
- duplicate rate 的分母是否合理；
- fragmentation 定义是否会把正常多个低 IoU 错误误称为 identity fragmentation；
- 20% duplicate/fragmentation reduction 是否可在低基线率下稳定解释；
- standard THUMOS mAP 是否仍需作为主结果；
- same-class overlap、chunk-crossing、long-action、short-action subset 如何定义且避免 post hoc；
- slot exhaustion counter 是否会被 train/val/test 混合累计；
- checkpoint 中 audit counter 是否可靠。

## 6.7 Training and deployment

重点检查：

- 训练集每 epoch 2,479 optimizer steps，12 epochs、9 runs 的真实成本；
- chunk size 64 与 memory 192 的交互；
- cross-chunk detach 是否让长动作只能通过数值 state、不能通过长程梯度学习；
- batch size 1 是否造成训练不稳定；
- single-rank sampler 是否完整遍历视频且每 epoch 顺序正确；
- validation/test 每 epoch 的额外成本是否计入预算；
- AMP、determinism、scheduler step 和 checkpoint 是否正确；
- 1-hour Slurm limit 是否可能中途 kill 而产生选择性结果；
- `10 GPU-hours` 是否只算训练，cache/smoke/evaluation 如何单列；
- unique variant/seed registry 是否真正阻止重复提交；
- failed run 的重试规则是否预注册；
- GitHub 是否缺少 environment lock、GPU 型号、CUDA、PyTorch、transformers 和 model revision。

# 7. 测试到 Claim 的映射

建立表格：

| Claim / invariant | Existing test | What it proves | What it does not prove | Missing adversarial test |
|---|---|---|---|---|

至少覆盖：

- future perturbation；
- chunk vs step equivalence；
- first endpoint crossing；
- same-class overlap；
- slot exhaustion；
- false-birth release；
- scalar start unit；
- GT kwarg rejection；
- EOF isolation；
- immutable one-time emission；
- resume cache contract；
- duplicate/fragmentation metrics；
- GPU-hour gate；
- variant config equality。

必须提出至少 10 个当前 tests 没覆盖、但可能导致错误论文结论的 adversarial cases。

# 8. 新颖性查新：必须联网并使用 primary sources

检索时间边界：执行审查当日，至少覆盖到 `2026-07-12`。

## 8.1 Direct On-TAD / On-TAL

必须核验：

- CAG-QIL；
- OAT / Sliding Window Scheme；
- SimOn；
- MATR；
- HAT；
- ActionSwitch；
- OnPoint；
- OZ-TAL；
- 2025--2026 新出现的 online temporal action localization 工作。

## 8.2 Persistent query / tracking

必须核验：

- TrackFormer；
- MOTR；
- MeMOTR；
- CTVIS；
- online video instance segmentation / streaming DETR 中的 query birth、death、rearm、identity propagation；
- 是否已有 temporal event tracking / action query tracking 的直接工作。

## 8.3 Causal raw-video and end-to-end TAL

必须核验：

- E2E-LOAD；
- StreamFormer；
- BasicTAD / E2E-TAD；
- Re2TAL；
- ETAD；
- TIA / AdaTAD；
- TE-TAD；
- LoSA；
- TALLFormer；
- boundary-sensitive / temporally-sensitive pretraining。

## 8.4 Search strings

至少使用：

```text
"online temporal action localization" persistent query
"online temporal action detection" event tracking
"streaming temporal action localization" transformer query
"causal temporal action detection" persistent instance
"online action localization" same-class overlap
"temporal TrackFormer" action localization
"event query tracking" untrimmed video
"end-to-end online temporal action localization" raw video
"completion-triggered" online action localization
"prefix assignment" temporal action localization
```

## 8.5 Competition matrix

输出表格：

| Work | Year/Venue | Exact task | Raw/video features | Trainable visual encoder | Persistent identity | Assignment | Start/end model | Emission mutability | NMS/suppression | Closest overlap | Surviving delta |
|---|---|---|---|---|---|---|---|---|---|---|---|

不允许仅比较标题和摘要。

# 9. Multi-paper reconstruction 和最强拒稿

必须构建至少 6 个重构 baseline：

```text
R1: MATR + fixed persistent query carry
R2: ActionSwitch + query slot representation
R3: TrackFormer projected onto one-dimensional temporal intervals
R4: StreamFormer/E2E-LOAD features + MATR decoder
R5: TrackFormer + scalar temporal start/end heads
R6: matched fresh DETR + bounded causal memory
```

对每个重构回答：

- 是否需要新机制，还是工程拼接即可；
- 是否已经覆盖 query persistence；
- 是否已经覆盖 same-class overlap；
- 是否已经覆盖 start retrieval / end detection；
- 是否已经覆盖 causal representation；
- 当前 PES 还剩什么不可约差异；
- 该差异是否足以成为 CCF-A 主贡献。

然后先写一段不少于 500 中文字的 strongest rejection，标题固定为：

```text
The Strongest Case for Killing Persistent Event-Set On-TAD
```

必须包含：

- obvious combination；
- baseline fidelity；
- assignment privilege；
- weak per-frame feature confound；
- custom metric risk；
- no result / no second dataset；
- train/inference lifecycle mismatch；
- why a positive THUMOS result may still be unpublishable。

完成最强拒稿前，不得提出美化后的论文故事。

# 10. Claim Map

为以下候选 claims 建立严格 claim map：

```text
C0: The implementation obeys a strict causal immutable On-TAD protocol.
C1: Persistent instance state improves matched-feature On-TAD over fresh queries.
C2: PES improves over a faithful Temporal TrackFormer baseline.
C3: The start-pointer / first-event-hazard design reduces identity-linked localization errors.
C4: The set lifecycle removes the need for online NMS without recall collapse.
C5: Chunked cached-feature training is a valid low-cost surrogate for testing the mechanism.
C6: A positive Stage-1 result would justify later raw-video causal PEFT.
```

输出：

| Claim | Exact scope | Current status | Closest prior | Required baseline | Required metric | Falsifier | Minimum evidence | Publishable if true? |
|---|---|---|---|---|---|---|---|---|

状态只能使用：

```text
code-supported
protocol-supported
unproven
partially testable
not identifiable
collapsed by prior art
```

特别审查：

- C1 与 C2 是否被当前三变体识别；
- C3 是否把 pointer 和 hazard 混成一个 claim；
- C4 是否仅由“代码没调用 NMS”支持，而没有性能证据；
- C5 的 cache surrogate 是否改变任务难度；
- C6 是否需要独立 2x2 frozen/trainable visual experiment，不能由 Stage-1 自动推出。

# 11. 实验设计审查

## 11.1 Baseline fairness

必须给出 `must-have before run`、`must-have if positive`、`nice-to-have` 三层 baseline。

至少讨论：

- FRESH；
- current TTF；
- corrected faithful Temporal TrackFormer；
- PES；
- MATR / ActionSwitch official or reproduced reference；
- simple recurrent/GRU state baseline；
- persistence state shuffle；
- instance-ID shuffle；
- memory-disabled control；
- pointer-only；
- hazard-only。

如果建议新增 baseline，必须说明它解决哪个 confound，不能无上限扩张实验。

## 11.2 Metrics and subsets

预注册建议必须包含：

- standard THUMOS mAP at tIoU 0.3:0.7；
- OnlineAP at fixed latency budgets；
- recall / FN；
- matched GT-end latency；
- duplicate FP；
- fragmentation；
- same-class concurrent subset；
- overlapping-instance subset；
- chunk-crossing subset；
- long-action / pre-memory subset；
- slot exhaustion；
- wall time、peak VRAM、optimizer steps、GPU-hours。

要求审查每个 subset 是否能在看结果前定义。

## 11.3 Statistics

当前三个 seeds 只用于 kill test。请设计：

- paired per-seed comparison；
- paired per-video bootstrap；
- confidence interval；
- effect size；
- non-inferiority margin for error-reduction claims；
- multiple comparison correction；
- seed failure / OOM / timeout 的预注册处理；
- 若 route survives，五种子确认规则。

不得用三个 seed 的均值差直接写“显著更好”。

## 11.4 Registered project gate

当前项目资源闸门是：PES 相对 FRESH 和 TTF 中的每一个，必须满足至少一项：

```text
average-mOnlineAP gain >= 2.0 points
or
duplicate or fragmentation error reduction >= 20%
while mOnlineAP is within 0.5 points
```

同时：

```text
protocol violations == 0
slot exhaustion == 0
matched seed sets
total Stage-1 training <= 10 GPU-hours
```

请判断：

- 该 gate 是否适合作为资源停止规则；
- 是否不应被称为统计显著性标准；
- duplicate 和 fragmentation 是否应分别判断；
- 若 baseline error 接近 0，20% 相对改善是否失真；
- 是否需要绝对改善下限；
- TTF 等价于 PES 时是否应直接 kill；
- aggregate mAP 提升但目标错误类型不改善时是否应 kill。

## 11.5 预算内修订

如果当前设计 `REVISE-BEFORE-RUN`，请给出不超过以下约束的最小修订：

```text
additional implementation <= 2 days
additional smoke <= 1 GPU-hour
Stage-1 total training <= 10 GPU-hours
no raw-video gradients
no full backbone training
no duplicate repair loss before mechanism survives
no new task or annotation
```

必须给出删减项，不能只增加实验。

# 12. 训练成本与下一阶段许可

当前 cache 是一次性 frozen-feature artifact，耗时约 0.87 GPU-hour。训练配置是 12 epochs、2,479 train chunks/epoch、三变体三 seeds。

请估算并审查：

1. 每 run 的 optimizer steps、wall time 和 GPU-hours；
2. 九个 runs 是否可能在 10 GPU-hours 内完成；
3. 是否应先用 seed 705 做三变体 screening，再决定 706/707；
4. 是否可用 early kill，而不引入选择性汇报；
5. validation cadence 是否需要降低；
6. 是否应减少 epoch、使用 successive halving 或共享 initialization；
7. cache cost、smoke cost、train cost、evaluation cost 应如何分别报告；
8. 何种结果才允许讨论 raw-video PEFT；
9. 即使 Stage-1 positive，为什么 raw-video 仍需要独立 novelty 和 2x2 adaptation gate。

Raw-video permission 默认为 `BLOCKED`。只有你明确列出所有前置证据并全部满足时，才可标为 `CONDITIONAL`。当前不得标为 `ALLOWED`。

# 13. 必须询问和讨论的不确定项

先从仓库寻找答案。只有仓库无法回答时才询问作者。问题分为：

```text
Resolved by cache/smoke evidence
Blocking before Stage-1 training
Blocking before interpreting results
Blocking before paper claim
Non-blocking but useful
```

至少审查以下 unknowns：

- canonical primary metric；
- exact TTF fidelity target；
- feature choice 是否足以测试机制；
- baseline threshold calibration；
- loss weight tuning protocol；
- epoch/early-stop selection；
- cache model exact revision/hash；
- hardware model and software environment；
- failed-run retry rule；
- second dataset access；
- whether official MATR/ActionSwitch features/results are comparable；
- public release plan for cache manifest and ledgers；
- author真正想保留的是 C1、C2 还是 C3；
- 如果 persistence 不赢，是否同意立即停止 Full PETAL。

每个问题必须提供：

```text
why it matters
default conservative assumption
what changes under alternative answers
whether work can continue before answer
```

不要一次抛出没有优先级的 50 个问题。

# 14. 互动规则

本次审查分两轮。

## Round 1：先审查和提问

第一次回复必须完成：

1. repository / commit verification；
2. evidence-level table；
3. task-definition audit；
4. P0/P1 code and protocol findings；
5. experimental-identifiability preliminary verdict；
6. blocking unknowns；
7. provisional `RUN-STAGE1 / REVISE-BEFORE-RUN / KILL-ROUTE`。

如果存在 blocking unknown，不得直接输出最终训练方案。必须停下来等待作者回答。

## Round 2：作者回答后

第二轮再完成：

1. fresh literature review；
2. competition matrix；
3. multi-paper reconstruction；
4. strongest rejection；
5. claim map；
6. corrected minimal experiment matrix；
7. statistics and budget；
8. final verdict and exact next actions。

如果 Round 1 已经可以凭公开证据判定致命问题，可直接 `KILL-ROUTE`，但仍须给出完整证据链。

# 15. 最终输出格式

最终回复必须严格包含以下章节：

```text
A. Executive Verdict
B. Repository and Evidence Verification
C. Unknown Register
D. Task Definition Audit
E. Code and Protocol Findings (P0-P3)
F. Test-to-Claim Map
G. Scientific Identifiability Audit
H. Fresh Novelty Search
I. Competition Matrix
J. Multi-Paper Reconstruction
K. The Strongest Case for Killing Persistent Event-Set On-TAD
L. Claim Map
M. Corrected Minimal Experiment Design
N. Metrics and Statistical Plan
O. Compute and Deployment Plan
P. Kill / Continue Rules
Q. Raw-Video Permission Gate
R. Questions for the Authors
S. Final Decision Card
```

Final Decision Card 必须使用：

```text
Repository HEAD reviewed:
Engineering verdict:
Protocol verdict:
Scientific-identifiability verdict:
Novelty verdict:
Stage-1 action: RUN-STAGE1 / REVISE-BEFORE-RUN / KILL-ROUTE
Raw-video training permission: BLOCKED / CONDITIONAL / ALLOWED
Most dangerous code flaw:
Most dangerous scientific confound:
Strongest direct-task competitor:
Strongest multi-paper reconstruction:
One claim that may survive:
Claims that do not survive:
Minimum fixes before the next GPU-hour:
Exact kill condition:
Confidence:
```

# 16. 禁止事项

禁止：

- 把 tests pass 写成方法正确；
- 把 cache completed 写成实验成功；
- 把“代码中没有 NMS”写成“post-processing-free 更好”；
- 把 persistent query 写成新颖性；
- 把 pointer 或 hazard 名称写成创新；
- 忽略 Temporal TrackFormer；
- 只比较 FRESH，不比较 TTF；
- 用 custom OnlineAP 隐藏 standard mAP；
- 在结果出来后再定义 overlap/fragmentation subset；
- 建议直接开始 raw-video 30-epoch multi-seed training；
- 通过新任务、新传感器、zero-shot 或 VideoLLM 逃离标准 On-TAD；
- 因为作者已经投入很多时间而降低审稿标准；
- 在 blocking unknown 尚未解决时输出过度自信的最终方法。

你的首要职责是帮助作者尽早停止无效路线，而不是让项目继续产生昂贵但不可解释的训练结果。

## END PROMPT
