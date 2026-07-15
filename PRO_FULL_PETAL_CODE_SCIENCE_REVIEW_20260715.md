# 1. Executive Verdict

**总裁决：`REVISE`，且当前禁止 profile 与 formal training。**

[CODE-VERIFIED] 固定提交、baseline commit 与 merge-base 均可解析；`b974f3d5f5d6ade15a383fa5c64138069977ba16` 是目标提交的祖先，三点 diff 与直接区间 diff 在本案等价。差异统计为 **81 个文件、28,646 行新增、454 行删除**，作者报告的数字准确。

[CODE-VERIFIED] 当前 Q2 不是 raw-video end-to-end 系统。它读取预先生成的缓存特征，在缓存特征之后训练 persistent temporal detector、slot state update 与 start/endpoint heads。`raw_video_finetuning=False`，因此任何“视觉 backbone 联合适配”“预训练视觉模型端到端适配”或“raw-video end-to-end On-TAD”主张均不成立。

[CODE-VERIFIED] `detach_stream_state=True` 不会阻止数值状态跨 chunk 持久化，但会阻止后续 chunk 的损失沿状态生成路径反传到前一 chunk。因此准确范围是：

> frozen cached features + chunk 内可训练 temporal/persistent computation + chunk 间 detached state persistence。

它不是仅有一个逐点分类头，故不应归类为 `HEAD_ONLY`；也不是全视频 temporal end-to-end。

[INFERENCE] Q2 的科学问题被代码压缩成了一个较小的干预：

> 动作实例出生后，固定 slot–instance 监督绑定，是否优于每个 prefix 重新匹配 active pool？

persistent slot、历史 memory、start/end 状态、completion emission、tracking-style identity 与因果流式推理都已有充分机制先例。完整 Full PETAL 目前属于 **RECONSTRUCTION**；fixed post-birth binding 本身至多是 **MARGINAL** 候选，且还没有训练结果支持。

[CODE-VERIFIED] `build_full_petal_launch_ticket.py`、`submit_full_petal_q2_n16r4.sh` 与 `validate_full_petal_launch` 之间存在一个 **profile-blocking P0**：ticket 先冻结 exact `cfg_overrides`，提交脚本随后生成带时间戳的 `RUN_DIR` 并加入运行时 `work_dir=${RUN_DIR}/work`，validator 又要求 ticket identity 与 runtime identity 严格相等。该动态值没有单一来源，当前 shell 顺序无法形成闭包。

[TEST-VERIFIED] 独立 contract reproducer 对当前跨脚本顺序失败；现有测试即便全部通过，也不能覆盖这个真实 shell 的动态 identity 组合。复现记录见：[P0 launch identity reproducer](sandbox:/mnt/data/ontad_p0_reproduction.md)。

[UNKNOWN] 缓存特征抽取器的不可变 commit、checkpoint、原始帧支持区间、centered/bidirectional clip 规则及整段归一化 provenance 没有形成可验证闭包。因而 detector 的 token-level 因果性不能升级为整个视觉系统的严格在线性。

**置信度：**

| 判断                           |                  置信度 |
| ---------------------------- | -------------------: |
| Git 对象、diff 与配置范围            |                    高 |
| launch-ticket P0             |                    高 |
| cached-feature end-to-end 范围 |                    高 |
| detector 内部因果执行顺序            |                   中高 |
| raw-frame 级 genuine online   |               低，证据缺失 |
| 方法有效性                        |             未知，无正式结果 |
| 完整方法创新性                      | 中高地倾向 RECONSTRUCTION |

完整逐文件报告及证据包：

* [完整审计报告（Markdown）](sandbox:/mnt/data/ontad_full_static_audit.md)
* [81-file changed-file 分类表](sandbox:/mnt/data/ontad_audit_evidence/07_changed_files_classified.csv)
* [全部 diff hunk 行范围](sandbox:/mnt/data/ontad_audit_evidence/09_diff_hunks.csv)
* [完整零遗漏 patch](sandbox:/mnt/data/ontad_audit_evidence/03_full_diff.patch)
* [全部 changed source 带行号版本](sandbox:/mnt/data/ontad_audit_evidence/18_all_changed_sources_numbered.txt.gz)
* [测试执行摘要](sandbox:/mnt/data/ontad_audit_evidence/23_test_summary.md)
* [完整证据包](sandbox:/mnt/data/ontad_audit_complete_bundle.tar.gz)

---

# 2. Repository / Commit Verification

| 检查项                        | 结果                                                         |
| -------------------------- | ---------------------------------------------------------- |
| Repository object 可访问      | [CODE-VERIFIED] 是                                          |
| Immutable target commit    | [CODE-VERIFIED] `6d88610da34a695e07d29c5e08b50fb57d2aa5e9` |
| Baseline object            | [CODE-VERIFIED] `b974f3d5f5d6ade15a383fa5c64138069977ba16` |
| Target object type         | [CODE-VERIFIED] commit                                     |
| Baseline object type       | [CODE-VERIFIED] commit                                     |
| Merge base                 | [CODE-VERIFIED] 等于指定 baseline                              |
| Baseline 是 target ancestor | [CODE-VERIFIED] 是                                          |
| Changed files              | [CODE-VERIFIED] 81                                         |
| Insertions                 | [CODE-VERIFIED] 28,646                                     |
| Deletions                  | [CODE-VERIFIED] 454                                        |
| Tree manifest              | [ARTIFACT-VERIFIED] 已固化                                    |
| Full binary-aware diff     | [ARTIFACT-VERIFIED] 已固化                                    |
| 作者外部 508/508 artifact      | [UNKNOWN] 未提供可独立绑定 commit、环境和命令的原始 artifact                |
| 作者外部 PASS / PROFILE=ALLOW  | [UNKNOWN] 仅为声明，不能覆盖本次发现的跨脚本 P0                             |

外部“508/508 passed”最多说明某组测试在某环境通过。它不证明：

1. shell 级动态启动顺序被执行；
2. ticket 和实际 Slurm argv 完全相同；
3. 缓存特征没有未来帧；
4. availability-clock latency 正确；
5. fixed binding 有效；
6. 该方法具有新颖性。

---

# 3. Unknown Register

以下项目不能从当前 immutable commit、可见数据 artifact 或 primary literature 完整回答：

| ID   | 未知项                                                  | 为什么重要                                  |
| ---- | ---------------------------------------------------- | -------------------------------------- |
| U-01 | 缓存特征 extractor 的 exact commit/config/checkpoint/hash | 决定 token 是否编码未来帧                       |
| U-02 | 每个 token 对应的 raw-frame support interval              | source timestamp 不能替代真实可获得时间           |
| U-03 | 是否使用 centered clip、双向视觉模型、整视频统计归一化                   | 任一项都可能破坏严格在线性                          |
| U-04 | 211 个 test cache 相对 canonical 213 缺失的两个视频 ID         | 决定 benchmark 总体是否被改变                   |
| U-05 | 两个缺失视频的缺失机制                                          | 若与长度、类别或异常视频相关，可能造成选择偏差                |
| U-06 | 外部 508/508 原始日志、环境锁和 commit attestation              | 无法将作者声明升级为 artifact evidence           |
| U-07 | 独立 review 的原文、检查范围和签名/hash                           | `PASS` 不能按名称接受                         |
| U-08 | 真实 GPU latency、显存、吞吐、kernel trace                    | 当前只能给复杂度公式                             |
| U-09 | 正式 checkpoint、baseline、seed 与 locked result          | 无有效性证据                                 |
| U-10 | FineAction 关闭决策的机器可验证状态                              | `None` 只能表示无对象，不能区分 disabled 与 missing |
| U-11 | 当前 Full PETAL/Q2 是否经过新的 versioned route decision 授权  | wiki 中存在 demoted/retired 冲突            |

---

# 4. 正式任务定义与 Genuine On-TAD 判定

## 4.1 应采用的正式信息合同

令缓存 token 为 (x_i)，其声明的 source time 为 (s_i)，真正可获得时间为 (a_i)。严格在线系统在决策时刻 (\tau) 只能使用：

[
\mathcal O_\tau={x_i\mid a_i\le \tau}
]

关键点是 (a_i) 不是简单的 (i\cdot\text{stride}/\text{fps})。它至少包括：

[
a_i=
\max \operatorname{RawSupport}(x_i)
+\text{feature extraction time}
+\text{batch/chunk wait}
+\text{queue time}
]

模型的 runtime state 必须满足：

[
R_t=F_\theta(R_{t-1},x_t)
]

而不能是：

[
R_t=F_\theta(R_{t-1},x_t,\text{GT schedule},\text{future length},\text{EOS oracle})
]

GT 可以进入训练损失构造，但不能进入：

* runtime slot state；
* inference birth decision；
* endpoint decision；
* emission ledger；
* NMS 前候选状态；
* evaluator之前的模型控制流。

## 4.2 当前代码能证明什么

[CODE-VERIFIED] 两个 Q2 配置走缓存特征输入，而非 raw frames。

[CODE-VERIFIED] detector 采用按 feature step 的 persistent state 更新；训练监督与 inference API 分离，assignment mode 是训练侧机制。

[CODE-VERIFIED] inference 侧没有把 GT instance ID 作为接口输入，fixed/rematch 不应改变推理函数。

[INFERENCE] 这足以支持“detector 在给定 token 序列上按顺序运行”，但不足以支持“视频系统严格在线”。

[UNKNOWN] 在缺少 raw-frame provenance 时，不能排除单个 (x_t) 本身已经由 (t) 之后的视频生成。

## 4.3 三种延迟不能混为一个数

配置给出：

* `feature_stride=8`
* `fps=30`
* `chunk_size=64`
* `memory_size=192`

因此：

[
\Delta_\text{feature}=8/30=0.2667\text{ s}
]

64 个 feature positions 的名义覆盖量：

[
64\cdot8/30=17.0667\text{ s}
]

首末 token timestamp 间隔：

[
63\cdot8/30=16.8\text{ s}
]

192-position history 的名义覆盖量：

[
192\cdot8/30=51.2\text{ s}
]

如果系统等完整 64-token chunk 到达后才调用 `_scan_and_decode`，则内部第一个 position 的额外等待最高约 **16.8 秒**，位置均匀时平均约 **8.4 秒**。这还没有包括 feature extractor 和 GPU compute。

必须分别报告：

| 延迟                        | 定义                                       |
| ------------------------- | ---------------------------------------- |
| Observation cadence       | 0.2667 秒/feature step                    |
| Temporal quantization     | endpoint 被离散到 feature grid 的误差           |
| Chunk batching delay      | 等待当前 chunk 其余 token                      |
| Model compute             | `_scan_and_decode`、head、ledger、metric 耗时 |
| Wall-clock emission delay | `emit_wall_time - GT_end_wall_time`      |
| Source-clock delay        | `declared_source_time - GT_end_time`     |

[INFERENCE] 如果 evaluator 使用内部 source index，而预测直到整 chunk 到达才实际产生，则 source-clock latency 会系统性低估部署延迟。

**判定：`GENUINE_ONTAD=UNPROVEN`。**

---

# 5. 方法、状态机、公式、张量、梯度与数据流重建

## 5.1 主要张量与状态

| 对象                                | 形状/规模                                  | 状态                                 |
| --------------------------------- | -------------------------------------- | ---------------------------------- |
| 缓存输入 chunk                        | `[B, L, 768]`，当前 `B=1`, `L≤64`         | frozen input                       |
| projected temporal representation | hidden dimension 256                   | trainable projection/temporal path |
| persistent slots                  | 4 slots，每 slot hidden 256              | runtime persistent                 |
| history memory                    | 最多 192 positions                       | runtime persistent                 |
| start state                       | 每 slot 一个 scalar start                 | slot lifecycle state               |
| endpoint output                   | 每 step、每 slot 一个 binary endpoint score | trainable                          |
| emitted/frozen flag               | 每 slot或ledger entry                    | immutable completion control       |
| train assignment                  | slot ↔ GT instance                     | train-only                         |
| emission ledger                   | 已提交 segment records                    | inference/runtime                  |

[CODE-VERIFIED] 四个 slot 的地址和 hidden state 可以跨 step/chunk 保存。

[INFERENCE] 这证明的是 **address persistence**，不是 learned instance identity。后者需要 slot–instance switch、same-class overlap 与 permutation intervention 证据。

## 5.2 GT schedule 与 endpoint survival contract

对 GT 实例 (j=(s_j,e_j,c_j))，合理的离散 first-crossing 定义是：

[
b_j=\min{t:\tau_t\ge s_j}
]

[
d_j=\min{t:\tau_t\ge e_j}
]

若 slot (k) 绑定实例 (j)，其 endpoint risk set 应是：

[
r_{k,t}=1[b_j\le t\le d_j,;k\leftrightarrow j,;\text{not emitted}]
]

first-event target：

[
y^{end}_{k,t}=1[t=d_j]
]

标准离散 hazard 损失相当于：

[
\mathcal L_{end,j}
==================

-\sum_{t=b_j}^{d_j-1}\log(1-h_{k,t})
-\log h_{k,d_j}
]

右删失情况下：

[
\mathcal L_{cens,j}
===================

-\sum_{t=b_j}^{c_j}\log(1-h_{k,t})
]

[CODE-VERIFIED] Q2 路径采用 instance-aware，而不是仅 class-aware 的 active-instance 表示；这使同类重叠在 slot 容量允许时可表达。

[CODE-VERIFIED] completion emission 后对应实例必须退出 endpoint risk set，并由 immutable ledger 防止重复提交。

[INFERENCE] “允许 delayed endpoint”应该体现在 latency metric 或明确的 event window，而不应通过对同一实例重复设置多个 endpoint positive 来实现，否则不再是 first-event likelihood。

[INFERENCE] 四 slot 是硬容量约束。任何时刻若存在超过四个并发、尚未完成的 GT 实例，必须明示 overflow policy：drop、censor、queue、replace 或 invalid episode。它不能静默变成负样本。

## 5.3 Fixed binding 与 prefix rematching

两个干预可形式化为：

### Fixed post-birth binding

[
\pi_t(k)=j,\qquad t\in[b_j,\text{completion/censor}]
]

出生时一旦 slot (k) 与实例 (j) 绑定，之后不重匹配。

### Prefix rematching

[
\pi_t
=====

\arg\min_{\pi\in\Pi_t}
\sum_{k,j}C_t(k,j)
]

每个 prefix 根据当前 active slots 和 active GT pool 重新计算 assignment。

[CODE-VERIFIED] Q2 inference 不需要 assignment mode，故 fixed/rematch 的推理行为可以共享。

[CODE-VERIFIED] 两份配置的核心显式变量是 assignment mode；birth、slot capacity、chunk、memory、head 与 inference 配置共享。

[INFERENCE] 这还不是完整 one-factor certificate。必须逐 step 比较：

* birth event；
* active-instance IDs；
* endpoint risk mask；
* negative-slot mask；
* normalization denominator；
* ignored/censored mask；
* slot reset；
* RNG consumption；
* inference output。

允许差异只能是 binding map，以及由 binding map直接推导的 target identity。若 negative pool 或 denominator 被另一条路径重新构建，Q2 就不是 one-factor。

## 5.4 梯度范围

`detach_stream_state=True` 时，若 chunk (q) 结束状态是：

[
R_q=F_\theta(X_q,R_{q-1})
]

下一 chunk 实际使用：

[
\widetilde R_q=\operatorname{stopgrad}(R_q)
]

因此：

[
\frac{\partial \mathcal L_{q+1}}{\partial R_q}=0
]

但 chunk 内的 step recurrence 仍可反传。

结论：

* [CODE-VERIFIED] 当前不是 full-video BPTT。
* [CODE-VERIFIED] 当前不是 raw-video end-to-end。
* [INFERENCE] 当前是 cached-feature temporal truncated-BPTT。
* [INFERENCE] “persistent”描述 forward state 合理；若暗示跨视频时间尺度的 temporal credit assignment，则不合理。

## 5.5 完整训练链

代码意图上的链条为：

```text
feature cache
  → chronological episode/chunk loader
  → stream-state snapshot
  → BF16 autocast
  → train_episode / _scan_and_decode
  → train-only GT schedule
  → birth / risk / binding targets
  → classification/start/endpoint losses
  → finite-loss check
  → backward
  → gradient-finite check / clip
  → optimizer update
  → scheduler successful-event update
  → state commit

failure:
  → zero gradients
  → skip optimizer/scheduler
  → restore runtime snapshot
  → record rejected optimizer event
```

[INFERENCE] BF16 通常不依靠 FP16 GradScaler 的 overflow detection。因此只检查 loss finite 不够；必须在 backward 后显式检查所有 gradients。

[UNKNOWN] 多 rank 下是否先对 `local_finite` 做 all-reduce AND，再让全部 rank 一致 commit/rollback，没有独立故障注入 artifact 支持。

## 5.6 推理与 evaluator 链

```text
causally available token
  → infer_step
  → slot hidden/start/endpoint update
  → threshold / endpoint decision
  → immutable ledger commit
  → optional slot reset/reuse
  → predictions with source_time + availability_time + emit_time
  → standard detection evaluator
  → supplementary budget/latency diagnostics
```

当前最大的协议风险是 evaluator 若只接收 source feature index，就无法知道预测实际在 chunk 完成后才产生。

---

# 6. 完整 Line-by-Line Coverage Ledger

全部 changed files、所有零上下文 diff hunks、changed Python symbols 与行范围已分别固化：

* [全部 changed files 分类](sandbox:/mnt/data/ontad_audit_evidence/07_changed_files_classified.csv)
* [全部 diff hunk 范围](sandbox:/mnt/data/ontad_audit_evidence/09_diff_hunks.csv)
* [changed Python symbol ledger](sandbox:/mnt/data/ontad_audit_evidence/10_python_symbol_ledger_seed.csv)
* [关键调用点与精确行号](sandbox:/mnt/data/ontad_audit_evidence/11_key_term_occurrences.csv)
* [全部 changed source 带行号](sandbox:/mnt/data/ontad_audit_evidence/18_all_changed_sources_numbered.txt.gz)

正文中的功能级 coverage ledger 如下；精确 `file:line-range` 在上述 ledger 中，避免把 81 个文件、全部 hunk 再复制成不可审阅的数百行正文。

| 类别            | 覆盖对象                                                            | 职责                             | 状态                       | Finding               |
| ------------- | --------------------------------------------------------------- | ------------------------------ | ------------------------ | --------------------- |
| model         | `PersistentTrajectoryOnlineDetector` 初始化、state、scan、train、infer | persistent slot detector       | OK / 部分 UNPROVEN         | P2-IDENTITY           |
| model         | `_scan_and_decode`                                              | 按 step 解码与状态更新                 | OK，顺序实现                  | P3-PYTHON-UNROLL      |
| model         | `PersistentEventSetHead` pointer/hazard 路径                      | 旧 set/head 机制                  | Q2 中 DEAD 或未调用           | P2-LEGACY-HEAD        |
| supervision   | first-crossing schedule                                         | GT 出生/结束离散化                    | 代码可执行；数学合同需锁定            | P1-ONE-FACTOR         |
| supervision   | fixed binding                                                   | 出生后固定 slot–instance            | ACTIVE                   | P1-ONE-FACTOR         |
| supervision   | prefix rematch                                                  | active pool 重匹配                | ACTIVE                   | P1-ONE-FACTOR         |
| supervision   | endpoint risk/first emission                                    | survival/event target          | ACTIVE                   | P1-OVERFLOW           |
| dataset       | cache reader                                                    | 读取 768-d frozen features       | ACTIVE                   | P1-FEATURE-PROVENANCE |
| dataset       | chunk stream                                                    | 64-step chronological chunks   | ACTIVE                   | P1-CHUNK-CLOCK        |
| dataset       | 160/40/211 manifests                                            | fit/cal/report split           | ACTIVE                   | P1-POPULATION-211     |
| metric        | `OnlineAPBudgeted`                                              | budget-aware诊断                 | ACTIVE，但非自动等价于标准 mAP     | P2-METRIC             |
| metric        | latency handling                                                | segment emission timing        | availability clock 未闭包   | P1-CHUNK-CLOCK        |
| training      | BF16/autocast                                                   | mixed precision                | ACTIVE                   | P1-TRANSACTION        |
| training      | snapshot/commit/rollback                                        | transactional stream state     | 单 rank 路径存在；故障闭包未证明      | P1-TRANSACTION        |
| training      | optimizer/scheduler                                             | 参数更新与事件计数                      | ACTIVE                   | P1-TRANSACTION        |
| launch        | ticket builder                                                  | exact identity                 | ACTIVE                   | P0-LAUNCH-WORKDIR     |
| launch        | runtime validator                                               | fail-closed identity compare   | ACTIVE                   | P0-LAUNCH-WORKDIR     |
| CLI/Slurm     | Q2 N16R4 submit script                                          | dynamic run directory与提交       | DEFECT                   | P0-LAUNCH-WORKDIR     |
| evidence      | B0/review/profile/formal tickets                                | gate chain                     | 结构存在；外部 artifact UNKNOWN | P0-LAUNCH-WORKDIR     |
| config        | fixed/rematch Q2 configs                                        | one-factor experiment identity | ACTIVE                   | P1-ONE-FACTOR         |
| config        | FineAction manifest                                             | qualification状态                | 语义含混                     | P2-FINEACTION         |
| test          | unit/contract tests                                             | local correctness              | P0跨脚本场景覆盖不足              | P0-LAUNCH-WORKDIR     |
| documentation | PETAL route wiki                                                | research SSOT                  | 与当前命名/路线漂移               | P2-DOC-DRIFT          |

---

# 7. P0–P3 Findings

## P0-LAUNCH-WORKDIR — P0

**位置：** `build_full_petal_launch_ticket.py` 的 canonical `cfg_overrides` 记录；`submit_full_petal_q2_n16r4.sh` 的 ticket 构造、动态 `RUN_DIR` 与 `work_dir` 追加；`validate_full_petal_launch` 的 strict identity comparator。精确行号见 [launch source ledger](sandbox:/mnt/data/ontad_audit_evidence/14_launch_ticket_sources.txt)。

**代码证据：[CODE-VERIFIED]**

执行顺序形成：

```text
O_ticket = canonical(ticket.cfg_overrides)

随后：
RUN_DIR=<timestamp-dependent path>
O_runtime = canonical(O_ticket ∪ {"work_dir": RUN_DIR + "/work"})

validator:
require O_runtime == O_ticket
```

因此：

[
O_\text{runtime}\setminus O_\text{ticket}
=========================================

{\texttt{work_dir=<dynamic path>}}
]

**违反不变量：**

> ticket attested identity 必须与实际运行 identity 完全一致，而且所有 identity-bearing values 必须在 ticket 创建前冻结。

**失败轨迹：**

1. ticket builder 写入 exact override identity；
2. shell 随后生成时间戳目录；
3. runtime command 新增或替换 `work_dir`；
4. pre-CUDA validator 比较完整 canonical identity；
5. mismatch；
6. profile job 在进入合法 CUDA 工作前失败，或只能通过绕开 validator 启动；
7. 任一种结果都不符合 attested profile。

**影响 claim：**

* transactional launch；
* exact attestation；
* independent replay；
* `PROFILE=ALLOW`；
* formal ticket 继承链。

**现有测试为何没有发现：**

builder 与 validator 的单元测试可以分别通过；使用固定 `work_dir` 的测试也可以通过。缺失的是：

> 执行真实 submit shell 的变量生成顺序，stub 掉 `sbatch`，捕获最终 runtime argv，再与 ticket 做 exact comparison。

**最小修复：**

```bash
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
WORK_DIR="${RUN_DIR}/work"

CFG_OVERRIDES=(
  ...
  "work_dir=${WORK_DIR}"
)

build_ticket \
  --cfg-overrides "${CFG_OVERRIDES[@]}" \
  --output "${TICKET}"

# 更优：运行参数从 ticket 反向读取，禁止第二次重建。
mapfile -t RUNTIME_OVERRIDES < <(
  read_ticket_cfg_overrides "${TICKET}"
)

validate_launch \
  --ticket "${TICKET}" \
  --runtime-overrides "${RUNTIME_OVERRIDES[@]}"

sbatch ... --cfg-overrides "${RUNTIME_OVERRIDES[@]}"
```

不建议简单从 identity 中删除 `work_dir`，因为它是输出与恢复位置的一部分。若确实要排除，应把它提升为单独 attested `artifact_root` 字段，而不是隐式忽略。

**新回归测试：**

* 冻结 `date`；
* 使用临时 `RUN_ROOT`；
* stub `sbatch` 只捕获 argv；
* 实际执行 submit script dry-run；
* 断言 ticket overrides、validator overrides 与 Slurm argv exact-equal；
* 修改一字符的 `work_dir`，断言 pre-CUDA fail-closed。

**阻断：** profile、formal training、evidence claim。

---

## P1-FEATURE-PROVENANCE — P1

**位置：** cache manifest、dataset/cache loader 与缺失的 extractor provenance chain。

**证据：[UNKNOWN]**

当前提交不足以证明每个 768-d token 只由其时间戳之前的 raw frames 生成。

**违反不变量：**

[
\max\operatorname{RawSupport}(x_t)\le a_t
]

**失败轨迹：**

centered clip或双向视觉 encoder读取未来帧
→ 生成被标记为时刻 (t) 的 token
→ detector 顺序处理 token
→ future-token perturbation test可能仍通过
→ 系统被错误标记为 causal。

**影响 claim：** genuine On-TAD、strict causal、deployable streaming。

**现有测试为何漏掉：** 测试通常扰动未来 token，而不是重新抽取“仅未来 raw frames 不同”的视频特征。

**最小修复：**

新增 versioned `feature_provenance.json`，至少记录：

```json
{
  "extractor_repo_commit": "...",
  "extractor_config_sha256": "...",
  "checkpoint_sha256": "...",
  "temporal_receptive_field": "...",
  "token_timestamp_rule": "...",
  "padding_rule": "...",
  "normalization_scope": "...",
  "causal": true
}
```

**新测试：** 构造两段视频，其前缀完全相同、未来帧不同；重新运行 extractor，要求前缀 token bitwise 或在预注册容差内相等。

**阻断：** genuine online、formal validity、投稿主张。

---

## P1-CHUNK-CLOCK — P1

**位置：** dataset chunk delivery、`_scan_and_decode`、emission ledger、evaluator timestamp。

**证据：[CODE-VERIFIED + INFERENCE]**

内部逐 step Python scan 不等于逐 step wall-clock availability。64-token chunk 对应约 17.07 秒视频。

**违反不变量：**

[
t_\text{emission}\ge t_\text{chunk-ready}
]

**失败轨迹：**

完整 chunk 到达
→ 函数内部回放 step 0…63
→ step 0 的预测被标为旧 source timestamp
→ evaluator认为早在约16.8秒前已可用
→ latency被显著低估。

**影响 claim：** online latency、low-delay completion、部署性。

**现有测试为何漏掉：** causal-mask测试只检查数值是否依赖未来 token，不检查函数何时被调用。

**最小修复：**

每条 prediction 同时保存：

```python
Prediction(
    source_time=...,
    feature_available_time=...,
    chunk_ready_time=...,
    model_done_time=...,
    emission_time=max(feature_available_time, chunk_ready_time) + compute_time,
)
```

或者直接用 `chunk_size=1` 的真正 step-wise serving path作为主在线评测，64-step batch仅作 throughput 模式。

**新测试：** 在第 (r) 个内部 position 的 ledger entry 上断言：

[
t_\text{emit}\ge t_\text{ready,last-token-of-chunk}
]

**阻断：**有效性、投稿。

---

## P1-ONE-FACTOR-TRACE — P1

**位置：** fixed/rematch assignment、active pool、risk mask、negative mask、loss normalization。

**证据：[CODE-VERIFIED]** 两配置显式核心差异是 assignment mode，inference path共享。

**仍未证明：[INFERENCE]** 所有派生监督量均完全相同。

**违反不变量：**

除 binding map 及其直接派生的 instance target 外：

```text
birth events
active lifecycle
candidate negatives
risk-set membership
normalization
RNG consumption
inference
```

必须一致。

**失败轨迹：**

rematch重建 active pool
→ mask或loss denominator改变
→ fixed/rematch差异不再只来自identity persistence
→ 结果无法归因。

**最小修复：** 生成 paired supervision trace：

```text
video_id, step, GT active IDs, slot active flags,
birth events, binding map, risk mask,
negative mask, ignored mask, denominator,
state reset, emitted IDs
```

运行同一权重、同一 seed、同一 episode，做白名单 diff。

**新测试：**

* same-class overlapping instances；
* GT列表排列置换；
* slot排列置换；
* chunk边界出生/结束；
  -四 slot满载及第五实例；
* rematch cost ties。

**阻断：** Q2 因果归因、formal experiment。

---

## P1-POPULATION-211 — P1

**位置：** THUMOS14 split/cache manifests、canonical expected count 与 locked reporting config。

**证据：[CODE-VERIFIED]**

```text
fit-core        160
calibration      40
validation total 200

locked report   211
canonical test  213
```

**违反不变量：**

benchmark总体必须由 annotation ID set决定，而不是由“成功生成的cache文件数”决定。

**失败轨迹：**

2个视频没有cache
→ count validator把211当预期
→ 所有结果只在211子集计算
→ 表格仍称THUMOS14 canonical test
→ 与213视频论文结果不可直接比较。

**影响 claim：** benchmark comparability、locked evaluation。

**最小修复：**

首选补齐213。不能补齐时：

* 列出缺失 video IDs；
* 解释缺失机制；
* 所有 baselines 在同一211子集重评；
* 表头写明 `THUMOS14-test-211-subset`；
* 不引用213总体上的公开数字作为直接对照。

**新测试：**

```python
assert annotation_test_ids == cache_test_ids
```

而不是：

```python
assert len(cache_test_ids) == 211
```

**阻断：** formal evaluation、投稿比较。

---

## P1-TRANSACTION-DDP — P1

**位置：** AMP、backward、gradient clipping、optimizer/scheduler、state snapshot/commit/rollback、resume。

**证据：[INFERENCE/UNKNOWN]**

单进程事务结构存在，但未获得以下故障闭包证据：

* loss finite、gradient nonfinite；
* BF16 无 scaler 的 overflow；
* 某一个 DDP rank 单独失败；
* optimizer skipped但scheduler推进；
* resume后successful optimizer-event计数错位；
* rollback只恢复runtime state而参数/optimizer部分更新。

**违反不变量：**

只有全部 rank 同意某次 update 成功时，才允许：

```text
parameters commit
optimizer state commit
scheduler step
optimizer_event += 1
runtime state commit
```

**正确顺序：**

```python
with autocast(dtype=torch.bfloat16):
    loss, proposed_state = ...

assert finite(loss)
backward()

local_ok = all_gradients_finite()
global_ok = distributed_all_reduce_and(local_ok)

if global_ok:
    clip_grad_norm_()
    optimizer.step()
    scheduler.step()
    commit(proposed_state)
    optimizer_event += 1
else:
    optimizer.zero_grad(set_to_none=True)
    rollback(runtime_snapshot)
```

**新测试：**

1. backward hook 注入 NaN grad；
2. FP16 scaler overflow；
3. 两 rank 中只让 rank 1 出错；
4. 断言全部参数、optimizer moment、scheduler、event counter、runtime state均未变化；
5. checkpoint后恢复并与不中断run逐 step相同。

**阻断：** formal training。

---

## P2-END-TO-END-SCOPE — P2

[CODE-VERIFIED] `raw_video_finetuning=False`，backbone不在训练图中。

[CODE-VERIFIED] `detach_stream_state=True` 截断chunk间反传。

**影响 claim：**

* “raw-video end-to-end”：错误；
* “full-stream end-to-end temporal learning”：错误；
* “cached-feature temporal end-to-end within truncated chunks”：可以接受；
* “persistent state”：可以接受，但只描述forward状态。

**最小修复：** 不改代码，先改 claim 和配置命名；将 scope 写进 ticket、checkpoint metadata 和结果表。

---

## P2-IDENTITY — P2

**位置：** persistent slot state与固定slot index。

[CODE-VERIFIED] slot index和hidden state被保存。

[INFERENCE] 这不证明模型学到了实例身份。它也可能只学到：

* slot 0偏向最早实例；
* slot地址作为固定容器；
* start scalar作为时钟；
* 同类重叠时任意交换。

**需要的 falsifier：**

| 实验                          | identity成立应出现的行为          |
| --------------------------- | ------------------------- |
| 中途交换两个slot hidden           | 对应实例预测应交换                 |
| 仅置换slot地址                   | 输出集合应等价                   |
| reset hidden但保留start        | identity错误显著增加            |
| same-class overlap          | switch率低于rematch/baseline |
| slot index embedding移除      | 性能不应完全崩溃到随机               |
| random post-birth rebinding | identity指标应显著恶化           |

没有这些证据，不得把“persistent trajectory identity”作为已验证机制。

---

## P2-LEGACY-HEAD — P2

**位置：** `PersistentEventSetHead` 的 pointer/hazard代码与Q2 scalar-start/binary-endpoint路径。

[CODE-VERIFIED] 两个Q2配置的主执行对象是 `PersistentTrajectoryOnlineDetector` 路径。

[INFERENCE] 对Q2而言，旧 pointer/hazard实现应视为 legacy/dead，除非 runtime hook证明它被调用并对loss或prediction有贡献。类存在、被import、单测通过或参数有梯度，都不能替代Q2调用证据。

**修复：**

* 给legacy class加明确 `deprecated_for_q2=True`；
* Q2 call-graph test记录forward hook计数；
* 方法图和论文不得把未调用的pointer/hazard算作Full PETAL组件。

---

## P2-METRIC — P2

**位置：** `OnlineAPBudgeted` 与标准 evaluator。

[INFERENCE] budgeted AP改变可参与排序或匹配的prediction集合，因此不能自动视为标准On-TAD主指标。

必须用手算golden cases验证：

* 同类重复预测；
* 两个同类重叠GT；
* 一条预测同时覆盖两个GT；
* score tie；
* timestamp tie；
* empty class；
* class有GT无prediction；
* class有prediction无GT；
* endpoint恰在feature边界；
* emission早于/晚于GT end时latency符号；
* slot重复emission；
  -跨chunk duplicate。

**裁决：**

* 标准 detection AP/mAP：主指标；
* source/availability/wall-clock latency：主在线指标；
* `OnlineAPBudgeted`：补充预算诊断，除非给出与标准定义的形式等价证明。

---

## P2-FINEACTION — P2

**位置：** `fineaction_qualification_manifest=None`。

[CODE-VERIFIED] 配置没有绑定FineAction qualification artifact。

[INFERENCE] `None` 不能同时表达：

1. FineAction明确不在本轮scope；
2. FineAction evidence尚未生成；
3. FineAction gate被关闭；
4. 配置漏填。

所以它本身不能证明“FineAction evidence 已关闭”。

**修复：**

```python
fineaction_evidence_mode = "disabled"  # 或 "required"
fineaction_disabled_reason = "Q2 is THUMOS14-only"
fineaction_qualification_manifest = None
```

`required + None` 必须 fail-closed；`disabled` 可以启动，但所有结果和claim中不得提FineAction支持。

---

## P2-DOC-DRIFT — P2

**位置：** `research-wiki/ideas/petal-ontad.md` 与当前 Full PETAL/Q2 configs、tickets、scripts。

[CODE-VERIFIED] wiki存在“full route demoted / PETAL名称应退役”的研究决策，而当前代码重新使用Full PETAL/Q2命名。

这并不使代码本身错误，但使“当前权威路线”成为 [UNKNOWN]。

**风险：**

* 旧路线以实现完成为理由自动复活；
* formal ticket引用过期claim；
* 实验代号被误写成论文方法；
* reviewer无法判断哪个文档是SSOT。

**修复：** 增加 versioned `route_status.yaml`：

```yaml
route_id: q2-fixed-vs-rematch
display_name: Q2 Binding Study
status: experimental
parent_route: petal-ontad
parent_status: demoted
paper_method_name_allowed: false
decision_commit: ...
```

---

## P3-PYTHON-UNROLL — P3

**位置：** `_scan_and_decode`。

[CODE-VERIFIED] 时间维采用逐 step Python unroll，而不是真正prefix-parallel训练。

复杂度至少为：

[
C_\text{chunk}
==============

\sum_{t=1}^{64}
\left(
C_\text{projection}
+C_\text{slot update}
+C_\text{memory read}
+C_\text{heads}
+C_\text{ledger/loss}
\right)
]

若memory attention使用 (S=4,M=192,H=256)，主要项可写成：

[
O(TSMH)+O(TSH^2)
]

但具体实现常数必须由profile确认。

训练activation近似随chunk length线性增长：

[
O(TSH)+O(T\cdot\text{auxiliary outputs})
]

跨chunk detach避免全视频activation保留，但不减少forward sequential dependence。

**结论：**

* 不得称prefix-parallel；
* Python/kernel launch overhead可能显著；
* 需要fixed-step profile；
* 当前不能给GPU-hour或吞吐数字。

---

# 8. Claim Map

| Claim                            | Mechanism                                         | Code evidence                | 必需实验                                                 | 当前状态                           | 直接 falsifier               |
| -------------------------------- | ------------------------------------------------- | ---------------------------- | ---------------------------------------------------- | ------------------------------ | -------------------------- |
| Full PETAL完整方法                   | slots、memory、start/end、immutable emission、binding | 组件存在；legacy边界需清理             | matched baselines、3 seeds、latency、identity ablations | [INFERENCE] RECONSTRUCTION     | 简单GRU/nonpersistent head等效 |
| cached-feature persistent On-TAD | frozen features + stateful detector               | detector顺序执行                 | raw-future perturbation、availability-clock评测         | [UNKNOWN] genuine online未闭包    | 改未来raw帧会改变过去token          |
| fixed post-birth binding         | 出生后不重匹配                                           | Q2 fixed路径存在                 | paired one-factor trace、3-seed paired统计              | [INFERENCE] MARGINAL candidate | 改善来自negative/denominator差异 |
| prefix rematch baseline          | 每prefix重匹配active pool                             | Q2 rematch路径存在               | 与fixed完全匹配的训练矩阵                                      | [CODE-VERIFIED] 可作为baseline    | inference或birth也发生变化       |
| persistent instance identity     | slot hidden/index跨步保存                             | address persistence存在        | switch、permutation、overlap干预                         | [INFERENCE] 未证明                | 交换slot state不改变实例行为        |
| strict online protocol           | causal token +真实availability clock                | token-level顺序存在              | extractor provenance、chunk-clock tests               | [UNKNOWN]                      | 未来raw支持或timestamp回填        |
| transactional launch             | ticket/runtime exact identity                     | builder/validator存在          | 真submit shell contract test                          | [CODE-VERIFIED] 被P0阻断          | 动态work_dir mismatch        |
| raw-video end-to-end             | backbone联合训练                                      | `raw_video_finetuning=False` | 不适用                                                  | [CODE-VERIFIED] FALSE          | backbone无梯度/不在图中           |
| formal THUMOS14 result           | canonical population                              | 211/213不一致                   | 补齐213或同子集重评                                          | [UNKNOWN]                      | 缺失视频改变结果                   |
| FineAction support/closure       | qualification manifest                            | manifest=None                | 显式disabled/required gate                             | [UNKNOWN]                      | None被多义解释                  |

---

# 9. End-to-End Truth Table

| 层级                    | Trainable | Frozen / detached          | Future-safe                    | 正确表述                        |
| --------------------- | --------- | -------------------------- | ------------------------------ | --------------------------- |
| Raw-video backbone    | 否         | 完全不在Q2训练图                  | [UNKNOWN] extractor provenance | 非raw-video E2E              |
| Cached features       | 否         | 固定                         | [UNKNOWN] token内部raw support   | frozen cached input         |
| Feature projection    | 是         | 否                          | 取决于输入token                     | cached-feature trainable    |
| Temporal scan         | 是         | chunk内可反传                  | detector调用顺序因果                 | truncated temporal training |
| Cross-chunk state     | 数值持久      | `detach_stream_state=True` | forward可因果                     | 无跨chunk credit              |
| Persistent slots      | 是         | 地址/hidden持久                | 无GT推理输入时可安全                    | identity尚未证明                |
| Start scalar head     | 是         | 否                          | 条件于causal token/state          | active                      |
| Binary endpoint head  | 是         | 否                          | 条件于causal token/state          | active                      |
| Pointer/hazard legacy | Q2中不应参与   | DEAD/legacy                | 不适用                            | 不得计入Q2贡献                    |
| Train assignment      | 无参数       | train-only                 | 必须与runtime隔离                   | supervised target builder   |
| Loss                  | 是         | GT可见                       | 仅训练合法                          | 不代表推理可见GT                   |
| Optimizer             | 是         | 失败时应rollback               | 不适用                            | DDP/BF16闭包未证明               |
| Streaming inference   | stateful  | 无梯度                        | cache/chunk clock未闭包           | genuine online UNPROVEN     |
| Standard evaluator    | 后处理       | 全GT可见合法                    | 不得影响模型决策                       | 主结果                         |
| OnlineAPBudgeted      | 后处理       | 全GT可见                      | metric定义待审                     | supplemental                |

**最终范围：`END_TO_END_SCOPE=CACHED_FEATURE_TEMPORAL`。**

---

# 10. Closest Work 与创新性 Subtraction Test

TrackFormer已经建立了跨帧持久 query、以query携带实例身份和生命周期的强机制先例；该工作属于目标跟踪而非On-TAD，但足以否定“persistent query/slot identity本身新颖”的主张。

OAT、MATR、HAT、StreamFormer等工作已经覆盖transformer式在线动作理解、历史/记忆建模和流式推理设计空间；它们与当前任务的输出合同可能不同，但构成直接的机制级近邻。

ActionSwitch构成动作切换、生命周期或时间转换建模的直接邻域，因而start/endpoint/switch状态不能按组件存在性主张新颖。

E2E-LOAD等端到端长时在线动作检测工作意味着作者绝不能使用“此前没有端到端在线动作检测”这一宽泛表述。需要逐项比较它们是frame-level current-action输出还是completed-segment emission，以及视觉backbone是否联合训练；任务合同差异可以保留窄增量，但不能抹去系统和机制先例。

## 贡献逐项裁决

| 候选贡献                                  | 裁决                             | 理由                                                            |
| ------------------------------------- | ------------------------------ | ------------------------------------------------------------- |
| A. Full PETAL完整方法                     | **RECONSTRUCTION**             | persistent query、memory、lifecycle、endpoint、tracking式绑定与缓存特征组合 |
| B. cached-feature persistent On-TAD   | **RECONSTRUCTION / 弱MARGINAL** | 可作为实验系统，但不是视觉端到端；online provenance未证明                         |
| C. fixed post-birth binding           | **MARGINAL candidate**         | 可能是特定任务下的监督策略增量，但需one-factor与显著结果                             |
| D. strict protocol/evidence framework | **工程价值较高，方法新颖性不足**             | ticket、attestation、fail-closed可提升复现性，但不是核心检测算法                |

## Subtraction test

1. **移除 evidence infrastructure：** 剩下普通persistent cached-feature detector；科学机制不变。
2. **移除 persistent slots：** 剩下recurrent start/endpoint classifier。
3. **移除 fixed binding：** 剩下prefix-rematched set prediction。
4. **移除 rematching：** 剩下teacher-forced slot identity。
5. **移除 cached pretrained features：** 当前没有可运行的raw-video视觉系统。
6. **移除 immutable ledger：** 剩下需要后处理去重的endpoint预测器。
7. **最终残余：** post-birth assignment policy 对completion-triggered detector训练稳定性和实例一致性的影响。

这个残余问题可以被严谨研究，但尺度不足以自动支撑“Full PETAL完整新方法”。

---

# 11. Baseline、Ablation、统计、数据划分与 Kill Criteria

## 11.1 必需 baseline

所有 baseline 必须共享相同 cache、split、hidden width、训练步数、optimizer events和 evaluator：

| Baseline                             | 作用                       |
| ------------------------------------ | ------------------------ |
| Prefix rematch active pool           | Q2直接对照                   |
| Fixed post-birth binding             | 候选机制                     |
| Random post-birth binding            | 检查固定地址本身                 |
| Permuted binding                     | 检查实例语义                   |
| Oracle stable binding                | 上界                       |
| Nonpersistent set head               | 检查persistent slots必要性    |
| GRU/LSTM completion head             | 简单temporal baseline      |
| Causal Transformer + start/end heads | 无slot强baseline           |
| Chunk-size 1 serving                 | 真正逐step latency baseline |
| Frozen slot state / reset each chunk | 检查跨chunk persistence     |
| `detach_stream_state=False` 小规模实验    | 检查credit truncation      |

## 11.2 必需 ablation

* slot数：2/4/8；
* memory：0/64/192；
* fixed/rematch；
* slot index embedding on/off；
* state reset；
* same-class overlap；
* active instances >4；
* endpoint freeze on/off；
* immutable ledger on/off；
* chunk size：1/8/64；
* source-clock vs availability-clock；
* causal extractor vs当前cache；
* detach on/off。

## 11.3 统计合同

[PROPOSAL]

* 至少3个独立训练seed；
* 固定与rematch使用paired seeds；
* per-video paired bootstrap；
* 主指标与停止条件在结果前冻结；
* calibration 40只允许阈值/温度/coverage校准，不得用于架构选择；
* locked reporting population在全部工程完成前不得用于调参。

建议主gate：

[
\Delta\text{Avg-mAP LCB}>-1.0\text{ pp}
]

并且至少一个预注册identity metric满足：

[
\text{relative ID error reduction LCB}>0
]

目标效应可设为平均至少20%，但不能只用point estimate。

## 11.4 Kill criteria

满足任一项即终止Full PETAL主方法路线：

1. fixed binding相对rematch的paired CI跨0，且identity metric没有稳定改善；
2. 简单GRU或causal Transformer baseline在同成本下相当或更好；
3. slot permutation/intervention表明slot没有实例身份语义；
4. same-class overlap下fixed不改善switch/duplicate；
   5.未来raw-frame perturbation改变过去token；
5. chunk=1真实在线评测后latency优势消失；
6. 补齐213后结论反转；
7. 改善来自不同negative pool或normalization，而不是binding；
8. 需要oracle GT birth/lifecycle才能保持结果；
9. 真实profile显示Python unroll成本不可接受且无性能收益。

---

# 12. Cost / Scale Audit

在profile实测前，不能给GPU-hour、峰值显存或吞吐数值。

允许的公式为：

令：

* (N)：episode总feature steps；
* (T=64)：chunk length；
* (S=4)：slots；
* (M=192)：memory；
* (H=256)：hidden dimension；
* (D=768)：input feature dimension。

总forward成本：

[
C_\text{episode}
\approx
N\left[
O(DH)
+
C_\text{slot}(S,H)
+
C_\text{memory}(S,M,H)
+
O(SH)
\right]
]

若memory交互为dense attention：

[
C_\text{memory}=O(SMH)
]

若slot内部有dense projection：

[
C_\text{slot}=O(SH^2)
]

训练activation在detach chunk边界时近似：

[
O(TSH)+O(T\cdot\text{loss auxiliaries})
]

持久runtime state近似：

[
O((M+S)H)
]

但实际吞吐还会被以下因素支配：

* 64次Python循环；
* 每step小kernel；
* ledger和assignment的CPU控制流；
* evaluator；
* cache I/O；
* BF16 kernel支持；
* synchronization；
* snapshot复制。

正确profile矩阵应固定总feature steps，而不是固定视频数：

| 变量           | 取值                                                                 |
| ------------ | ------------------------------------------------------------------ |
| chunk length | 16, 32, 64, 128                                                    |
| memory       | 0, 64, 192                                                         |
| slots        | 2, 4, 8                                                            |
| AMP          | FP32, BF16                                                         |
| mode         | train forward、backward、infer                                       |
| reporting    | p50/p95 step latency、peak allocated/reserved、tokens/s、kernel count |

当前P0未修复前不能启动该profile。

---

# 13. 文件、函数与测试级实现计划

## 13.1 P0修复文件

### `submit_full_petal_q2_n16r4.sh`

* 把 `RUN_ID/RUN_DIR/WORK_DIR` 创建提前到ticket之前；
* 只构造一次canonical override list；
* 不允许在builder之后追加identity-bearing override；
* Slurm argv从ticket读取。

### `build_full_petal_launch_ticket.py`

* 接收已经完全resolved的overrides；
* canonical sort/normalization；
* 保存原始列表与canonical map；
* 保存identity digest；
* 拒绝重复key和后覆盖。

### `validate_full_petal_launch`

* 比较canonical maps；
* 单独报告missing、extra、value mismatch；
* 验证runtime argv digest；
* 在任何CUDA import/device init之前执行。

### 新测试

```python
def test_submit_script_ticket_runtime_identity_exact(tmp_path, fake_date, fake_sbatch):
    result = run_real_submit_script_dry_run(...)
    ticket = load_ticket(result.ticket)
    runtime = parse_captured_sbatch_argv(result.argv)

    assert canonical(ticket["cfg_overrides"]) == canonical(runtime.cfg_overrides)


def test_dynamic_workdir_mutation_fails_closed(...):
    runtime.cfg_overrides["work_dir"] += "_mutated"
    assert validate(...) != 0
    assert cuda_was_never_initialized()
```

## 13.2 Provenance修复

新增：

```text
tools/build_feature_provenance.py
tools/validate_feature_provenance.py
tests/test_cached_feature_raw_future_invariance.py
```

dataset在加载cache前验证provenance hash与配置ticket一致。

## 13.3 三时钟 evaluator

扩展prediction schema：

```python
@dataclass(frozen=True)
class OnlineEmission:
    video_id: str
    class_id: int
    start_time: float
    end_time: float
    source_time: float
    available_time: float
    emitted_time: float
    score: float
    slot_id: int
    emission_id: str
```

主metric不得根据后验GT修改 `emitted_time`。

## 13.4 One-factor trace

新增无梯度trace API：

```python
trace = detector.trace_supervision_episode(
    features,
    gt_instances,
    assignment_mode=...,
)
```

比较时使用差异白名单，而非只比较总loss。

## 13.5 AMP事务

新增：

* `all_gradients_finite()`；
* distributed boolean consensus；
* successful optimizer-event counter；
* snapshot checksum；
* resume exact-continuation test。

## 13.6 文档与命名

在正式有效性证据前：

* 将“Full PETAL”改成中性实验代号，例如 `Q2-PersistentBindingStudy`；
* route status标为`experimental`；
* 禁止paper-method claim；
* external 508/PASS写成`DECLARED`，直到artifact被绑定。

---

# 14. 唯一下一步

**`FIX_BEFORE_PROFILE`**

理由不是“profile可能不好看”，而是当前 profile identity 不能形成有效的ticket—runtime闭包。继续提交GPU job会产生两种坏结果之一：

1. validator正确工作，job在pre-CUDA gate失败；
2. 为了运行而绕过validator，得到无法由ticket独立重放的profile。

在P0修复并通过真实submit-shell contract test之前，不能选择 `RUN_FIXED_STEP_PROFILE`，更不能进入 cached-feature kill test或formal training。

---

# 15. 仅列无法从代码、artifact或文献回答的作者问题

1. canonical 213中缺失的两个test视频ID分别是什么，为什么没有cache？
2. 缓存抽取器的exact repository commit、config、checkpoint hash与每token raw-frame support是什么？
3. 特征是否使用centered clip、双向时序attention或整视频级归一化？
4. 作者所称外部508/508、独立PASS和PROFILE=ALLOW的原始、不可变artifact在哪里？
5. 当前Full PETAL/Q2是单纯实验代号，还是正式推翻了wiki中demoted/retired决定？对应的versioned decision记录在哪里？

---

RESEARCH_VERDICT=REVISE
GENUINE_ONTAD=UNPROVEN
END_TO_END_SCOPE=CACHED_FEATURE_TEMPORAL
NOVELTY_VERDICT=RECONSTRUCTION
PROFILE=BLOCK
FORMAL_TRAINING=BLOCK
NEXT_STEP=FIX_BEFORE_PROFILE
OPEN_P0=P0-LAUNCH-WORKDIR
OPEN_P1=P1-FEATURE-PROVENANCE,P1-CHUNK-CLOCK,P1-ONE-FACTOR-TRACE,P1-POPULATION-211,P1-TRANSACTION-DDP
