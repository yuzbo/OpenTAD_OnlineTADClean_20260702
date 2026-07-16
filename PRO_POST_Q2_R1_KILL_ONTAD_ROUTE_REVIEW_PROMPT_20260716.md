# Pro Prompt: Post-Q2/R1-Kill On-TAD Route Review

请将下面整段内容交给 Pro 模型。不要删减仓库、提交、证据边界或输出合同。

---

你是一名对 Online Temporal Action Detection/Localization、流式视频模型、
时序检测、端到端训练和高效深度学习实验有直接发表与高级审稿经验的
CCF-A Senior PC / Area Chair。请执行一次 fail-closed、证据优先、允许否定
项目的研究路线审查。你的职责不是替作者包装现有想法，而是判断在当前
失败证据之后，On-TAD 内部是否仍有一个值得实现、可证伪、可发表的新路线。

## 1. 审查锚点

公开仓库：

`https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702`

分支：

`codex/full-petal-implementation`

固定审查提交：

`7a26e60fccec68cd2b547951320e669a81907461`

固定提交链接：

`https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/tree/7a26e60fccec68cd2b547951320e669a81907461`

请先核验仓库、分支和提交。不得用默认分支、旧 commit、作者摘要或过去聊天
代替代码与证据。无法访问的内容必须进入 Unknown Register，不能自行补全。

至少逐项阅读并交叉核验：

- `RTK.md`
- `research-wiki/query_pack.md`
- `research-wiki/index.md`
- `research-wiki/decision_register.md` 中 DR-041 至 DR-044
- `research-wiki/discussion_timeline.md` 中 T34 至 T37
- `research-wiki/experiments/crs-eps-g0-kill-20260716.md`
- `research-wiki/experiments/q2-capacity-lifecycle-audit-20260716.md`
- `research-wiki/ideas/chronological-state-faithful-selective-backward.md`
- `PRO_CRS_EPS_G0_KILL_ROUTE_REVIEW_ROUND2_20260716.md`
- `PRO_CRS_EPS_G0_KILL_ROUTE_REVIEW_ROUND2_ABSORPTION_20260716.md`
- `PRO_Q2_CAPACITY_INDEPENDENT_MAX_REVIEW_20260716.md`
- `opentad/models/dense_heads/persistent_event_set_head.py`
- `opentad/models/detectors/persistent_trajectory_ontad.py`
- `opentad/utils/prefix_trajectory_supervision.py`
- `opentad/utils/q2_capacity_audit.py`
- `tools/audit_q2_capacity_lifecycle.py`
- `configs/causaltad/thumos_pes_q2_base.py`
- fixed/rematch configs、训练 engine、optimizer audit、launch gates 和相关测试

## 2. 不可改写的任务边界

任务保持标准 fully supervised On-TAD：因果视频流输入；任何时刻不得访问未来
帧；输出动作实例 `{start, end, class, score}`；输出一旦提交不可离线回改；不得
依赖整段视频、offline NMS、未来端点、GT 驱动的推理状态或测试时标签。

不要把问题改写为 OAD、未来预测、在线语义记忆、zero-shot TAL、主动感知、
mutable alarm、三时钟验证或新的 benchmark/task。可以把这些工作作为竞争或
边界文献，但不能用任务变化制造创新。

输入形态必须明确区分：预提取因果特征、冻结视觉编码器、可微视觉编码器和
raw-video end-to-end。不得把 cached-feature 训练称为 raw-video end-to-end。

## 3. 已冻结的负证据

以下事实必须作为约束，而不是待包装的结果：

1. 当前 CRS-EPS empty-state dynamic replay 已在预注册 G0 中 terminal `KILL`；
   HH/IPW 无法恢复被省略的历史状态、生命周期和 Jacobian 路径。
2. clean Q2 容量审计在 160 个 fit-core 视频、seeds 705/706/707 上完成。
3. actual controller 有 2,206 次 GT-birth exhaustion；oracle minimum K=2；
   `TRUE_CANONICAL_CAPACITY=0`。
4. 唯一合法零耗尽 arm 是 birth-logit bias `-2`，但总共只发射 10 次，且两个
   seed 为零发射。
5. 唯一独立 max reviewer 核验 524,258 行 trace 后裁决
   `C) KILL_Q2_R1`，诊断为 `DEGENERATE_BIRTH_SUPPRESSION`。
6. Q2 的 summary status 永久保留 `REVISE_REQUIRED`；独立科学处置为 `KILL`。
7. R1/CSFSB 未实现且不得在本轮被默认复活；GPU profile/formal training 均
   BLOCK；当前 GPU 授权为 0 小时。

不得通过 `-2` prior、改 birth threshold、增 K、改 refractory、改 release order
或 post hoc nondegeneracy floor 重新包装 Q2。若认为某个组成值得复用，必须说明
为什么它属于一个新的、可识别的机制合同，而不是隐藏的多因素补丁。

## 4. 证据标签

每个实质判断必须标注以下之一：

- `[CODE-VERIFIED]`
- `[EVIDENCE-VERIFIED]`
- `[LITERATURE-VERIFIED]`
- `[INFERENCE]`
- `[UNKNOWN]`

涉及现有竞争工作、最新论文、年份、任务定义和 novelty 的判断必须在线查新，
优先原论文、官方项目页和正式会议记录。给出可点击来源、检索日期、检索式和
覆盖范围。不能因没搜到就声称“无人区”。

## 5. 必须先回答的核心问题

在提出方法前，先给出逐项、可否定的回答：

1. 当前代码实际优化的对象、runtime lifecycle、canonical supervision 和
   inference decode 各是什么？它们在哪里不一致？
2. 2,206 次 exhaustion 暴露的是当前实现 bug、任务建模矛盾、teacher-forcing
   exposure bias、离散 assignment 不可微，还是 persistent-slot 路线本身失败？
3. 为什么标准 fully supervised On-TAD 的性能仍低？哪些问题已有工作解决，
   哪些仍由公开证据支持为 gap？
4. “尚无真正端到端可训练 On-TAD”是否准确？必须分别核验 cached-feature、
   backbone fine-tuning、raw-video causal joint training，不允许混用术语。
5. 一个合法的新路线如何让 runtime state、训练 assignment、loss risk set 和
   inference emission 使用同一个 prefix-observable contract？
6. 如何设计一个不能被 no-birth、no-emission、always-background 或永不结束
   平凡通过的 readiness gate？
7. 如果离散 lifecycle 本身妨碍端到端训练，应该使用可微状态、latent set
   prediction、causal sequence transduction、survival/point-process objective、
   teacher-student state alignment，还是应放弃 persistent slots？不要预设答案。
8. 新路线是否真的改善 On-TAD 的核心指标和失败模式，而不只是降低训练成本？
9. 在单卡有限预算下，什么最小证据足以 kill，什么证据才允许扩展？

## 6. Unknown Register

先建立完整 Unknown Register，至少覆盖：

- 训练与推理状态转移是否逐 bin 完全同构；
- GT assignment 是否能进入 runtime 不可用槽；
- birth/end 的监督时刻、risk set、first-event target 和重复正样本语义；
- 同类重复、重叠、同 bin end/birth、长动作、动作起点早于 memory 的处理；
- detach/TBPTT 边界和历史 Jacobian 丢失；
- 视觉特征是否严格因果，是否含未来上下文或离线预处理污染；
- evaluator 对 late FP、miss、duplicate、fragmentation、delay 和 no-future 的处理；
- 当前最强 On-TAD、OAD、offline TAL、streaming transformer 和 tracking 竞争；
- raw-video joint training 的显存、吞吐和可复现实验成本；
- THUMOS14 是否足以支撑结论，第二数据集和重叠/密集子集如何选；
- 哪些现有测试只证明软件一致性，哪些构成科学证据；
- 任何阻止 claim、实现或实验裁决的未知项。

能从代码、证据和公开来源回答的未知项请直接关闭。只有确实无法核验的项目
才保留 `[UNKNOWN]`，并说明最小解法以及它阻塞哪个决策。

## 7. 不做任务发散的候选路线生成

只有完成任务审查、root-cause 和查新后，才生成 3 至 5 条真正不同的 On-TAD
方法路线。不要为了凑数保留弱路线，也不要默认 persistent query 必须保留。

每条路线必须包含：

- 一句话任务内 gap；
- 精确输入、在线状态、状态转移和输出；
- 训练目标、assignment/risk-set 定义和可微路径；
- 它如何避免 Q2 的 runtime/canonical mismatch；
- 它为何不能靠静默策略通过门禁；
- 相对最接近工作的最小差异和实质差异；
- 主要 claim、claim map 和可证伪预测；
- 端到端含义：head-only、feature-level、backbone fine-tuning 或 raw-video joint；
- 训练成本、内存、吞吐、TBPTT/缓存/混合精度策略；
- 最强 kill argument、可能的负结果和退出条件；
- 必须修改的仓库模块和预计实现规模。

禁止把 causal backbone、memory、query persistence、LoRA、feature cache、gradient
sampling、pretraining 或一个新 loss 单独当成 headline innovation。它们只能在
明确解决 On-TAD 特有机制问题时成为组件。

## 8. 创新性与竞争审查

对每条保留路线建立 closest-work matrix，至少比较：

- 标准 On-TAD/TAL 在线检测方法；
- window/anchor/query/state-switch 路线；
- raw-video end-to-end OAD 与 causal video backbone；
- offline end-to-end TAL、boundary pretraining、PEFT 和高效训练；
- TrackFormer、online VIS 和 set/trajectory prediction；
- causal survival、point process 或 sequence transduction 中真正相关的方法。

矩阵列必须包括 task、future access、input modality、trainable backbone、state
identity、assignment、mutable/immutable output、offline cleanup、latency metric、
training cost 和与候选路线的重叠。给出 `novel / reconstruction-risk / occupied /
unknown` 裁决，不接受仅凭模块组合数量判断创新。

## 9. 实验闭环与成本门禁

为排名第一的路线设计分阶段实验，但本轮不得执行训练：

### P0: 科学正确性

- synthetic delayed/repeated/overlap/same-bin cases；
- GT-taint、future perturbation、prefix cut、stepwise/batched equivalence；
- runtime/supervision state-transition equality；
- no-birth/no-emission/always-background anti-cheating gate；
- optimizer coverage、finite/nonzero gradients、endpoint freeze；
- 任一失败即 `KILL`，GPU 0 小时。

### P1: CPU/cached-feature mechanism kill

- 明确 gold/reference、候选和最强简单 baseline；
- full chronological execution，不允许只挑好看的 event；
- baseline、ablation、negative control 和 privileged upper bound 分离；
- 预声明 metric、seed、sample、阈值、family-wise decision rule；
- 先做 unseen CPU G0，再允许 profile。

### P2: fixed-step profile

- 仅在 P0/P1 PASS 后；
- 单卡不超过 2 GPU-hours；
- 报告 step unit、warmup、measured steps、峰值显存、tokens/s、video/s、
  backward 占比和总 GPU-hours；
- 不得把缓存训练成本冒充 raw-video joint-training 成本。

### P3: 最小效果实验

- matched seeds、相同 evaluator、相同 causal features/encoder、相同预算；
- mOnlineAP/online mAP、delay、recall/FN、duplicate、fragmentation、same-class
  repetition、overlap、长动作和吞吐；
- baseline 包含最接近公开方法和最强简单重建；
- 主效应、机制效应和效率效应必须分别有 ablation 支撑。

给出每阶段预计 CPU/GPU 时间、存储、最大失败成本和停止条件。禁止直接提出
全量训练。

## 10. 代码级实现审查

对排名第一的路线给出 file-by-file implementation map，但不要写伪代码掩盖
未知项。至少说明：

- 新增/修改类、函数、config 和数据合同；
- state dataclass 及每个字段的可观测性；
- train/inference 共用与禁止分叉的逻辑；
- loss mask、denominator、optimizer event 和 TBPTT 语义；
- atomic evidence、commit/config/data/checkpoint binding；
- 单元、集成、合成、B0、G0 和 reviewer gate；
- 哪些旧代码只读保留，哪些明确废弃但不删除历史证据。

每项实现必须映射到一个已批准 claim 或 gate。没有 claim/gate 的代码不应写。

## 11. 强制输出顺序

严格按以下顺序输出：

```text
A. Repository/commit verification
B. Executive verdict
C. Evidence ledger
D. Unknown Register
E. Exact current task and implementation
F. Root-cause analysis of CRS-EPS and Q2 failures
G. Current On-TAD gap audit
H. Literature search protocol and dated source table
I. Candidate route portfolio
J. Closest-work/reconstruction matrix
K. Claim map for each retained route
L. Adversarial novelty review
M. Scientific-scale and experiment-scale ranking
N. Selected route or project-stop decision
O. Formal method definition for the selected route
P. Training and inference contract
Q. Anti-silence and anti-taint gates
R. Baseline/ablation/metric plan
S. Costed P0-P3 experiment ladder
T. File-by-file implementation map
U. Reviewer-risk register
V. Exact author questions still blocking work
W. Final machine-readable decision JSON
X. Author-response draft
Y. Next-round prompt, only if another round is genuinely necessary
```

## 12. 最终裁决

必须且只能选择一个：

1. `GO_NEW_ROUTE_P0_ONLY`
2. `REVISE_BEFORE_IMPLEMENTATION`
3. `STOP_CURRENT_PROJECT`

`GO_NEW_ROUTE_P0_ONLY` 只允许实现零 GPU P0 和必要代码测试，不允许 profile、
效果训练或 formal training。只有当任务定义、novelty、runtime/supervision 合同、
anti-silence gate 和最小实现边界全部明确时才能选择 GO。

最终 JSON 至少包含：

```json
{
  "reviewed_commit": "7a26e60fccec68cd2b547951320e669a81907461",
  "current_q2_r1_disposition": "KILL",
  "selected_decision": "GO_NEW_ROUTE_P0_ONLY | REVISE_BEFORE_IMPLEMENTATION | STOP_CURRENT_PROJECT",
  "selected_route_id": null,
  "task_definition_pass": false,
  "novelty_pass": false,
  "runtime_supervision_contract_pass": false,
  "anti_silence_gate_frozen": false,
  "p0_code_authorized": false,
  "gpu_profile_allowed": false,
  "formal_training_allowed": false,
  "gpu_hours_authorized": 0,
  "blocking_unknowns": []
}
```

保持严厉。若没有足够强的新路线，明确选择 `STOP_CURRENT_PROJECT`，不要用复杂
模块堆叠制造继续工作的理由。

---
