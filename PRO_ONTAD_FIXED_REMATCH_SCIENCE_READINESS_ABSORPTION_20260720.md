# On-TAD FIXED/REMATCH 科学与实验就绪审查：独立复核与吸收

日期：2026-07-20
复核对象：`PRO_ONTAD_FIXED_REMATCH_SCIENCE_READINESS_REVIEW_20260720.md`
当前代码锚点：`95fa963e7e2f3a05779907999df7f9311be4f4fd`

## 结论

原审查的主裁决仍然成立：**必须先修复，当前不能进入 FIXED/REMATCH 单种子科学筛选。**

后续 smoke 和严格确定性画像证明了代码能执行、会更新参数、修复后的容量问题没有
立即复发，并且两臂在训练前具有相同的推理路径；它们没有修复审查指出的科学合同
问题，也没有提供方法效果证据。与此同时，画像实测推翻了原审查中“该实验低成本、
可直接进入单种子”的估计：注册的双臂 12 epoch 协议需要约 `12.572 GPU·小时`，
超过冻结的 `2 GPU·小时` 上限。

因此当前有两个彼此独立的阻断：

1. endpoint、出生步绑定、短动作、数据隔离、指标和结果证据链仍不闭合；
2. 原注册训练协议未通过预算门禁。

不提交 seed 705，不运行 seeds 706/707，不进入 raw-RGB。

## 来源完整性

| 字段 | 值 |
|---|---|
| 原始附件 | `pasted-text.txt` |
| 库内逐字节归档 | `PRO_ONTAD_FIXED_REMATCH_SCIENCE_READINESS_REVIEW_20260720.md` |
| 文件大小 | `61,985 bytes` |
| 逻辑行数 | `643` |
| UTF-8 BOM | 无 |
| SHA-256 | `D463A28BC64E8418AC697263EB7AB6866EBC719718A91735B7A53E52FCD0E88F` |
| 原审查代码锚点 | `27a59dec445f6b4ed9651abab1168c358a8db7e3` |
| 本次复核代码锚点 | `95fa963e7e2f3a05779907999df7f9311be4f4fd` |

原审查明确说明：它没有独立运行训练、Slurm 或聚焦测试。因此原文中的运行状态只能
视为对当时代码和仓库记录的静态审查。本次复核另外纳入了 `27a59de..95fa963`
之间的代码差异、N16R4 smoke 作业 `1176737`、严格画像作业 `1176983` 及其冻结
artifact。

## 复核方法

本次逐项检查了：

- `tools/train.py`、`tools/test.py` 和 `opentad/cores/train_engine.py`；
- `prefix_trajectory_supervision.py` 的 birth、canonical binding 和 REMATCH；
- `persistent_trajectory_ontad.py` 的损失、状态、审计与 emission；
- `persistent_event_set_head.py` 的 CANDIDATE/ACTIVE/COMMIT 生命周期；
- `streaming_feature.py` 与 `cache_ontad_features.py` 的缓存合同；
- `online_instance_metrics.py`、`online_budgeted_map.py` 和
  `persistent_binding_gate.py`；
- FIXED/REMATCH 基础配置、smoke/画像脚本、测试及当前实验记录；
- CAG-QIL、SimOn 和 MATR 的第一方论文页面。

裁决标签：

- **接受**：当前代码仍可直接支持该问题；
- **部分接受**：核心风险仍在，但后续已有局部修复或原措辞需收窄；
- **已解决**：当前代码或新实验证据已关闭原问题；
- **被新证据推翻**：原审查中的状态或估计已不再成立。

## P0 逐条裁决

| 编号 | 裁决 | 当前证据与吸收决定 |
|---|---|---|
| P0-1 未监督 endpoint offset | **接受** | `endpoint_offset_head` 仍参与 binary commit 的 `end_frame=current-offset`，但 detector 损失仍只有 birth/alive/class/start/end，没有 offset loss。主实验应让 binary endpoint 固定为当前 decision frame，并删除或忽略该支路；不得临时增加 sub-stride 回归。 |
| P0-2 REMATCH 在出生步换槽 | **接受** | 监督器先建立 canonical birth，随后把包括 newborn 在内的全部 visible binding 立即 rematch。birth BCE 与 class/start/end 可能落在不同槽。newborn 必须在出生步保持 canonical slot，REMATCH 最早从下一 decision 开始。 |
| P0-3 同步 birth+end 不提交 | **接受** | decoder 先处理进入本步时已有的 CANDIDATE/ACTIVE，再接纳 newborn；新 candidate 不再检查本步 end。最后 token 或同一 stride 内极短动作可能永不提交。 |
| P0-4 训练访问 locked reporting | **接受** | 当前 `tools/train.py` 仍无条件构建 `dataset.test`；基础配置的 test 是 211-video reporting manifest，`val_eval_interval=1`，因此每轮都会调用 reporting inference。必须实现 fit-train、calibration-freeze、report-once 专用流程。 |
| P0-5 结果证据链未闭合 | **接受** | smoke/画像新增了局部 artifact 和 hash，但常规正式训练仍不把 episode audit、checkpoint、配置、manifest、ledger、标准 mAP 和实例指标闭合成自动 gate row。正式字段仍不可手填。 |
| P0-6 实例指标 stream/坐标错配 | **接受** | GT 数据库默认使用视频名，emission 使用复合 runtime stream key；指标又优先读取预测的 frame bounds，而原始 GT segment 是秒。直接组合时会发生匹配键或单位错误。 |
| P0-7 fragmentation/matching 定义不符 | **接受** | 当前实现仍是按 emission 输入顺序的 greedy 匹配；不同边界即算 fragment，嵌套/重叠 duplicate 也会增加 fragmentation。必须冻结全局一对一主匹配和相交片段连通分量定义。 |
| P0-8 mAP 字段、schema 和单位未闭合 | **接受** | 主配置只运行 `OnlineAPBudgeted` 并产出 `average_mOnlineAP`（0–1）；gate 接受无 schema/range 的 `average_map`，同时把阈值写成 `-0.5 points`。必须明确 standard mAP、0–100 百分点和 tIoU 集合。 |
| P0-9 readiness/fail-closed/counter | **部分接受** | smoke/画像 launcher 已对代码 SHA、测试、更新和画像门禁做了额外检查；feature route 也显式 `fail_on_nonfinite=True`。但通用 formal runner 仍不强制 `formal_training_ready`，canonical exhaustion 仍不是原子失败，多个容量概念仍未完全分开。 |

## P1 逐条裁决

| 编号 | 裁决 | 当前证据与吸收决定 |
|---|---|---|
| P1-1 本步释放、下一步复用 | **接受并收窄** | `free_at_entry` 仍在释放之前冻结。这可以保留为明确的结构选择，但不能宣称无损；必须用 frozen census、`deferred_birth_due_to_release` 和相邻动作子集量化影响。若产生漏检，应修控制器，而不是事后加槽或降阈值。 |
| P1-2 birth/capacity census 未被 launcher 消费 | **接受** | 文档中的最大并发二不能自动证明每步 birth、same-bin end+birth 和 delayed-reuse headroom。需要哈希 census artifact 并由 launcher fail-closed 消费。 |
| P1-3 最后 token/缓存覆盖 | **接受** | 缓存生成器包含短终包 source frame，但当前没有正式 census 证明每个纳入评测的 GT end 都有合法的 `source_frame >= end`。 |
| P1-4 start 历史 clamp | **接受** | start target 仍被 clamp 到 `memory_size=192`，即 stride 8 下最多 1536 frames。必须先统计 clipped-start 数量，再统一决定扩大 memory 或报告能力边界。 |
| P1-5 reset fail-closed | **部分接受** | 官方 dataset 只在 token 0 产生 `is_video_start=True`，正常路径不会中途 reset；但 detector 自身只检查布尔类型，并会在任何 start 标记时清空全部状态。正式入口应验证 packet start 和视频转换。 |
| P1-6 缓存生产因果性 | **接受** | manifest 有 annotation hash、encoder ID、feature policy、source frame 和 per-video hash，但没有 producer commit/model hash，也没有 future-frame perturbation 证明。下游因果测试无法独立证明上游 encoder 感受野。 |
| P1-7 成功更新计数 | **部分解决** | feature 配置已用 FP32 和 `fail_on_nonfinite=True`；smoke/画像验证了真实参数更新。原先推荐 AMP 的做法已被实测的首步非有限梯度否定。但常规正式 runner 仍缺两臂 terminal successful-update/scheduler-step 对账。 |
| P1-8 checkpoint 规则 | **部分解决** | smoke 显式保存、加载并核对 checkpoint；通用 `tools/test.py` 仍在未指定时读取 `best.pth`，而本路线关闭 val-loss best。正式实验必须显式固定最后 epoch checkpoint 及 SHA。 |
| P1-9 ledger 扩展名与 sequence | **部分解决** | 文件名已经从 `.jsonl` 改为 `.json`。row 仍没有独立 `sequence_id`；sequence 只嵌在 `event_id` 中，且 `video_id` 与 `runtime_stream_key` 没有分离，正式 schema 仍需修复。 |
| P1-10 阈值算法未冻结 | **接受** | 当前只说“在 calibration 上选择共享阈值”，没有冻结对称目标、候选网格、tie-break 或一次性 reporting 规则。 |

## P2 逐条裁决

| 编号 | 裁决 | 当前证据与吸收决定 |
|---|---|---|
| P2-1 detached chunk state | **接受** | 这是两臂共享的 truncated-state 近似，不破坏当前配对，但通过主门后应做 chunk 32/64/128 单种子敏感性。 |
| P2-2 GT 去重 | **接受** | standard mAP 可按社区规则去重；实例指标必须保留可区分的实例 ID，并同时报告原始/去重 GT 数。 |
| P2-3 相对 E_id 不稳定 | **接受** | 冻结主门不变，同时报告绝对 `ΔE_id`、两个分量和 paired video bootstrap CI；不能事后换主指标。 |
| P2-4 缺 profiler | **已解决且改变决策** | 当前已有严格确定性 paired profiler。结果通过稳定性、更新、容量和推理等价检查，但得到 `12.572 GPU·小时`，使预算门禁失败。 |
| P2-5 sub-stride endpoint | **接受** | 继续禁止在当前主实验临时加入；只能在主门通过后作为独立消融。 |

## 原审查中已经过时或需要更正的结论

1. **“只能 crash-only 试跑、Slurm smoke 未完成”已过时。**
   job `1176737` 已完成真实 FP32 反向、checkpoint、reload、流式推理和 ledger
   一致性验证。它只提升工程就绪度，不消除 P0 科学问题。

2. **“FIXED/REMATCH 是低成本实验”被实测推翻。**
   job `1176983` 的严格画像给出安全系数后 `12.572 GPU·小时`，是注册单种子
   上限的约 `6.29×`。因此不应按原计划直接提交。

3. **原审查建议覆盖 AMP 的 smoke 不再适用。**
   实际 AMP 首步出现非有限梯度并跳过更新；当前 feature-only 路线使用 FP32 且
   fail-on-nonfinite。这是基于运行证据的正确收敛，不是配对变量。

4. **`.jsonl` 扩展名冲突已修为 `.json`。**
   但 ledger row 的显式 sequence、video/runtime key 分离和 provenance 仍未完成。

5. **feature-only optimizer 构建问题已修。**
   `backbone=None` 的优化器构建和真实更新已由 smoke 覆盖。

6. **P2-4 profiler 缺口已关闭。**
   画像同时证明两臂未训练推理各有 10,029 条 emission、规范 digest 完全相同，
   并且无 future-end、future-source、negative-latency 或非单调 emission。

7. **文献边界基本正确，但 MATR 引用需更正。**
   CAG-QIL 和 SimOn 的第一方页面都明确 On-TAL 不使用未来帧、不能修改过去输出；
   MATR 的“当前片段估计 end、历史 memory 估计 start”由
   `arXiv:2408.02957` 支持。原文将 MATR 句子挂到 CAG-QIL 的 CVF 脚注上，不能
   作为 MATR 的直接来源，应改用 MATR 原论文。

## 当前成立的因果与公平性结论

### 已有正证据

- feature cache 之后的 detector/head 只累积当前及过去 source frames；
- 模型元数据拒绝 annotation、duration、EOV、future endpoint 等字段；
- supervision state 与 runtime state 已分离，预测占用不能删除 canonical GT birth；
- FIXED/REMATCH 最终配置除 binding mode 和 work directory 外一致；
- REMATCH 只进入训练监督分配，不改变推理 decoder；
- 最终 emission 标记 immutable，提交后没有修改接口，也没有 offline NMS；
- 严格画像证明两臂训练前推理完全等价。

### 尚不能成立

- binary endpoint 具有可解释、受监督的终点坐标；
- REMATCH 只在 birth 之后改变 binding；
- newborn 同步 end、最后 token 和释放后相邻 birth 不会漏检；
- 上游 frozen cache 已用 producer provenance 和 future perturbation 证明严格因果；
- 训练、calibration、reporting 三者严格隔离；
- 实例指标、standard mAP、latency metric 与 gate 共享同一 schema 和单位；
- 一次 formal run 能自动生成不可手填、可哈希追溯的完整 result row；
- 原 12-epoch 协议满足已注册预算。

## 吸收后的唯一下一步

下一步不是再发起高成本讨论，也不是提交 seed 705，而是直接实现一个
`scientific-contract repair` 提交，最小范围如下：

1. binary endpoint 固定为当前 decision frame，移除无监督 offset 对主输出的影响；
2. newborn 在 birth step 固定 canonical slot，REMATCH 延迟到下一 decision；
3. 支持 newborn 的 same-step end，并为 last-token short action 加整链路测试；
4. 实现 fit-only train、calibration freeze、report-once 三阶段专用入口；
5. 修正 instance metric 的 video key、统一坐标、全局匹配和连通片段定义；
6. 冻结 standard mAP percentage-point gate schema，并生成带 provenance hash 的
   repository-owned result artifact；
7. 拆分 canonical exhaustion、runtime collision、arbitration suppression、
   cancellation/abandonment 计数，并让 formal launch fail-closed；
8. 生成并由 launcher 消费 birth/end/cache/start-history census。

修复后按相同顺序重新运行：聚焦反例测试 → Slurm smoke → strict paired profile。
由于计算量可能变化，必须重新画像；只有科学合同通过且新预算协议显式注册后，
才能运行 seed 705。seed 705 非退化后再冻结 calibration 阈值和执行三种子。

## 是否需要再次发起 Pro 讨论

**现在不需要。** 原审查已经给出足够具体、可直接编码和测试的反例；新增实验证据又
明确给出预算阻断。此时继续讨论不会替代修复。下一次外部审查只在以下任一条件满足
时有价值：

- 修复提交、整链路 smoke 和新画像全部通过，需要审查预算修订与单种子注册；
- 修复迫使 FIXED/REMATCH 改变自变量定义；
- seed 705 出现难以解释的指标矛盾，需要在看三种子结果前冻结处理规则。

## 冻结证据

- 原审查：`PRO_ONTAD_FIXED_REMATCH_SCIENCE_READINESS_REVIEW_20260720.md`
- 严格画像：`research-wiki/experiments/ontad-science-fixed-rematch-profile-20260720.md`
- smoke 记录：`research-wiki/experiments/ontad-science-fixed-rematch-smoke-20260720.md`
- 设计：`research-wiki/experiments/ontad-science-fixed-rematch-design-20260720.md`
- 问题地图：`research-wiki/experiments/ontad-science-fixed-rematch-problem-map-20260720.md`
- 实施计划：`research-wiki/experiments/ontad-science-fixed-rematch-plan-20260720.md`

## 本次落库验证

- 原文归档 SHA-256 与附件完全一致；
- 复核矩阵覆盖 `P0-1..P0-9`、`P1-1..P1-10`、`P2-1..P2-5`，共 24 项；
- 所有本次新增/修改 Markdown 的相对链接检查通过；
- `edges.jsonl` 64 条边均可解析，新 experiment node 与 `tested_by` 边存在；
- Wiki 共识节点无重复 ID，`query_pack.md` 为 7,807 字符，低于 8,000 字符上限；
- `git diff --check` 通过；
- 不依赖 Torch 的监督器、实例指标、结果门和流式特征数据测试：
  `35 passed`。

Windows 本地 Python 的 PyTorch 在导入 `c10.dll` 时发生 `WinError 1114`，所以本次
没有在 Windows 重跑 head/detector Torch 测试。该环境限制不被记作代码测试失败；
Torch 路径采用已经冻结的 N16R4 证据：当前提交相关的远端聚焦测试、job
`1176737` smoke 和 job `1176983` profile。此次提交只归档和更新研究记录，不修改
模型代码。
