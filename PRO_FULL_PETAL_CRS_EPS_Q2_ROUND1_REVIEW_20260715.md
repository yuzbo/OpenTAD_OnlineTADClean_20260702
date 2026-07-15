# Round 1：代码、定义与阻塞未知项

本轮严格按上传 Prompt 的 Round 1 边界执行：核验仓库锚点、重建现有 Q2 训练与成本链、判断 CRS‑EPS 是否存在、复核 `P0-LAUNCH-WORKDIR`、审计 Q2 one-factor 与 profile 分母，并冻结需要作者回答的协议问题。

## A. Executive Verdict

**本轮裁决：`REVISE-BEFORE-IMPLEMENTATION`。**

| 项目                          | Round-1 裁决                                                                                    |
| --------------------------- | --------------------------------------------------------------------------------------------- |
| 当前训练路线                      | `FULL-CHRONOLOGICAL CACHED-FEATURE VIDEO-EPISODE TRAINING`                                    |
| 梯度范围                        | chunk 内 BPTT；chunk 间 numerical state carried、gradient detached                                |
| CRS‑EPS 当前实现                | **NO**                                                                                        |
| Q2 one-factor               | **PARTIALLY IDENTIFIABLE**                                                                    |
| 当前 full-chronological route | 可作为当前“缓存 token、per-video mean、chunk-detached”目标的穷举 reference；不能泛称 full-sequence unbiased gold |
| 推荐方向                        | **HYBRID，暂定**：CRS‑EPS 主训练候选 + 小规模 full-stream gradient/state gold audit + 完整时间顺序验证/测试         |
| Profile contract            | **REPLACE**，而非沿用 optimizer-events/s 作为跨协议主分母                                                  |
| Full PETAL novelty          | **RECONSTRUCTION，暂定**；fixed binding 仍只是未验证的 marginal delta                                    |
| Raw-video Stage 2           | **BLOCKED**                                                                                   |
| Profile                     | **BLOCKED**                                                                                   |
| Formal training             | **BLOCKED**                                                                                   |
| 当前唯一 P0                     | `P0-LAUNCH-WORKDIR`                                                                           |

核心判断如下。

1. **[CODE-VERIFIED] 当前实现不是 CRS‑EPS。** 它把每个训练视频从 token 0 到结尾完整枚举为连续 chunks，维持跨 chunk 数值状态，并在完整视频结束时才执行一次 optimizer/scheduler event。CRS‑EPS 要求的 event-centric anchor、192-token burn-in、4–8 supervised bins、`pi_t`、IPW/SNIPW、ESS 和 episode manifest 均不存在。

2. **[CODE-VERIFIED] “optimizer event 少”不等于“主要训练计算少”。** 每个有效 token 仍逐一进入 Python `for` scan、`head.step`、监督状态转换、loss 和 decode；现有路线只是把整段视频的 token 梯度合并成一个 AdamW update。

3. **[CODE-VERIFIED] 当前 full route 的准确目标是 per-video mean token loss。** 长视频包含更多 forward/backward token，却不会因更长而获得更多 optimizer-event 权重。它不是 corpus-token-uniform objective，除非项目明确把目标风险定义为“先均匀选视频，再均匀选视频内决策点”。

4. **[CODE-VERIFIED] `P0-LAUNCH-WORKDIR` 仍然成立。** ticket 先冻结 exact `cfg_overrides`，Slurm helper 随后才产生时间戳 `RUN_DIR` 并追加新的 `work_dir`；validator 又要求完整 runtime identity 严格相等。正常启动路径无法形成单一、不可变的 identity 来源。

5. **[DOC-VERIFIED] 没有任何正式有效性结果。** 截至锚点文档，Q2、CRS‑EPS、LoRA 和 raw-video 路线均无被接受的 fixed-step profile 或 formal multi-seed result。

因此，目前既不应 `KEEP-FULL-CHRONOLOGICAL-ONLY`，也没有证据 `KILL-CRS-EPS`；正确状态是先修复 P0，并在任何 GPU profile 前重新冻结目标风险、episode 状态重建和多分母 profile 合同。

---

## B. Repository and Commit Verification

| 检查项                            | 结果                                               | 证据等级              |
| ------------------------------ | ------------------------------------------------ | ----------------- |
| Repository                     | `yuzbo/OpenTAD_OnlineTADClean_20260702` 可访问      | `[CODE-VERIFIED]` |
| Requested branch               | `codex/full-petal-implementation` 可解析            | `[CODE-VERIFIED]` |
| Branch HEAD observed           | `c56857b1e3bbbd445aa86e9e31a1e1ddd07ffe51`       | `[CODE-VERIFIED]` |
| HEAD message                   | `docs: add CRS-EPS Q2 Pro review prompt`         | `[CODE-VERIFIED]` |
| Immutable commit reviewed      | `f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb`       | `[CODE-VERIFIED]` |
| Anchor → HEAD                  | ahead 1 commit；只增加审查 Prompt 及 wiki provenance 索引 | `[CODE-VERIFIED]` |
| 科学代码/config/tests/results 是否变化 | **否**                                            | `[CODE-VERIFIED]` |
| 实际审查基线                         | 固定为 `f4ea53e...`                                 | `[CODE-VERIFIED]` |

HEAD 提交内容与 message 均表明它是文档提交，没有改变本轮需要审计的科学实现。

### 锚点证据状态

* **[ARTIFACT-ASSERTED]** 项目吸收文档记录了此前 B0 artifact SHA、508 tests、0 blocking findings、0 protocol violations；该外部 bundle 未在本轮会话中重新执行或独立展开，而且它没有覆盖真实 submit-shell 的动态 `work_dir` 组合。
* **[DOC-VERIFIED]** 当前 profile 与 formal training 均明确处于 blocked 状态。

---

## C. Evidence Table

| ID  | 关键事实                                                                  | 标签                | 当前能证明什么                                       | 不能证明什么                                                             |
| --- | --------------------------------------------------------------------- | ----------------- | --------------------------------------------- | ------------------------------------------------------------------ |
| E1  | `raw_video_finetuning=False`，输入为 768-d cached features                | `[CODE-VERIFIED]` | 当前是缓存特征后的 temporal detector 训练                | raw-video end-to-end、视觉 encoder 适配                                 |
| E2  | dataset 为每个视频枚举全部连续 chunks                                            | `[CODE-VERIFIED]` | 当前训练覆盖选定视频的全部 cache tokens                    | event-centric sampling                                             |
| E3  | `optimizer_events_per_epoch = number of videos`                       | `[CODE-VERIFIED]` | 一完整视频对应一个 optimizer event                     | 一个 event 具有固定计算量                                                   |
| E4  | 每个 token 进入 Python sequential scan                                    | `[CODE-VERIFIED]` | token-level 因果执行顺序                            | prefix-parallel 高效训练                                               |
| E5  | chunk loss 先按 token mean，再按 token 数加权，视频末统一除总 token 数                 | `[CODE-VERIFIED]` | 每个视频形成 mean-over-token gradient               | corpus-token-uniform 风险                                            |
| E6  | cross-chunk state 全部 `.detach()`                                      | `[CODE-VERIFIED]` | forward state 连续                              | 跨 chunk temporal credit assignment                                 |
| E7  | fixed/rematch 配置只显式改变 binding mode 和 `work_dir`                       | `[CODE-VERIFIED]` | 静态干预面较干净                                      | 正式 run 的完整 one-factor 证书                                           |
| E8  | 双活跃 probe 中 birth/lifecycle/masks 相同，loss binding 与对应 gradient 不同     | `[CODE-VERIFIED]` | binding intervention 的局部代码行为正确                | 全数据训练中的效果归因                                                        |
| E9  | CRS‑EPS 文档要求 burn-in、短 suffix、mixture、`pi_t`、IPW/ESS                  | `[DOC-VERIFIED]`  | 历史协议定义明确                                      | 当前代码已实现这些内容                                                        |
| E10 | ticket/runtime `work_dir` identity 不闭包                                | `[CODE-VERIFIED]` | 当前 profile 正常路径会被 validator 阻断或迫使绕过 validator | 合法、可审计的 GPU profile                                                |
| E11 | fixed-step profiler 的主吞吐为 optimizer-events/s                          | `[CODE-VERIFIED]` | 同一事件单位下的局部计时                                  | full video event 与 CRS episode 公平比较                                |
| E12 | cache manifest 只强制 encoder ID、stride、dtype、source frames、array hashes | `[CODE-VERIFIED]` | cache 文件和索引完整性                                | extractor commit/checkpoint、raw support interval、future invariance |
| E13 | `emit_time_sec=emit_frame/fps`                                        | `[CODE-VERIFIED]` | source-clock 时间                               | availability/wall-clock latency                                    |
| E14 | 无正式 Q2/CRS‑EPS/LoRA 结果                                                | `[DOC-VERIFIED]`  | 当前只能做协议和代码判断                                  | 方法有效性、SOTA、论文 claim                                                |

当前 Q2 配置明确冻结了 cached-feature、single-GPU、chunk 64、memory 192、state detach 和 50/200 optimizer-event profile。

---

## D. Fixed Task and Scope Audit

### D.1 固定任务

**[CODE-VERIFIED] PASS，限于任务定义。**

当前 task contract 仍是 fully supervised completion-triggered On-TAD：

```text
causal observation prefix
-> latent persistent slot state
-> predicted {start, end, class, score}
-> immutable emission after predicted completion
```

两个 Q2 arm 共用：

* first-crossing birth；
* canonical lifecycle；
* 4-slot capacity；
* persistent head；
* inference path；
* immutable ledger；
* no offline NMS；
* no raw prediction shortcut；
* no GT runtime state。

训练 schedule 可以由完整 annotation 构造，但 birth/active targets 不含未来 endpoint，endpoint 只在首次可观测 crossing 时出现。

### D.2 当前 end-to-end 边界

**[CODE-VERIFIED] 当前只能称：**

```text
frozen cached features
+ trainable temporal projection / persistent slots / output heads
+ chunk-internal BPTT
+ cross-chunk detached forward state
```

不能称：

```text
raw-video end-to-end
visual backbone adaptation
full-video temporal BPTT
CRS-EPS training
```

runtime state 在 chunk 末保留 queries、feature memory、slot status、start/class state，但所有 tensor 都被 detach。

### D.3 genuine online 与真实时钟

**[UNKNOWN] 整个视频系统的严格 causal 性未闭包。**

cache manifest 当前验证：

* `feature_policy="packet_recent_frame"`；
* timestamp convention；
* stride；
* `encoder_id`；
* dtype；
* feature hash；
* source-frame sequence。

但没有强制：

* extractor repository commit；
* extractor config hash；
* visual checkpoint SHA；
* processor/normalization；
* 每个 token 的 raw-frame support interval；
* centered/bidirectional clip policy；
* raw-prefix future invariance。

项目文档本身要求 checkpoint SHA、preprocessing hash、cache build revision 和 raw/cache equivalence，但这些尚未进入现有 cache schema。

### D.4 chunk 时间问题

当前：

```text
feature_stride = 8 frames
fps = 30
chunk_size = 64 tokens
```

因此一个 full chunk 名义覆盖：

```text
64 × 8 / 30 = 17.0667 seconds
```

模型在 chunk 内逐 token 计算，因此 cached-token 数值结果可与逐 token 调用一致；测试也验证了 episode scan 与 incremental scan 的 logits/state 等价。

但是 dataloader 一次交付完整 chunk，forward 返回后用户才能收到其中的 emission。当前 ledger 将：

```python
emit_time_sec = emit_frame / fps
source_time_sec = source_frame / fps
```

写成 source clock，而不是实际 packet availability 或 wall clock。

**Task protocol verdict：**

```text
CACHED-TOKEN CAUSAL EXECUTION = SUPPORTED
STRICT RAW-VIDEO CAUSALITY = UNPROVEN
LOW-WALL-CLOCK-LATENCY CLAIM = UNSUPPORTED
```

---

## E. Current Code Dataflow and Cost Accounting

### E.1 真实执行链

```text
external split manifest + feature cache manifest
    ↓
StreamingFeatureDataset
    ├─ verify selected video IDs
    ├─ verify cache array/hash/source frames
    └─ enumerate every chunk from token 0 to token T_v-1
    ↓
ChronologicalStreamBatchSampler
    ├─ no shuffle
    └─ preserve each video lane in chunk order
    ↓
train_engine: begin video transaction
    ↓
for each chunk:
    cached features [<=64 tokens]
    + training-only prefix schedule
    + previous detached runtime/supervision state
    ↓
for every valid token in Python:
    PersistentEventSetHead.step
    shared first-crossing birth assignment
    fixed or rematch loss binding
    birth/alive/class/start/end losses
    model-predicted runtime decode
    ↓
chunk mean loss
× valid token count
→ backward
    ↓
if not video end:
    keep staged numerical state
    no optimizer step
    ↓
at complete video end:
    divide accumulated gradients by all video tokens
    finite-gradient check
    optimizer.step
    scheduler.step
    commit online transaction
    or rollback all mutations on failure
```

Dataset 对所有 selected videos 做完整 chunk 枚举，并将 `optimizer_events_per_epoch` 定义为视频数。

训练 engine 在非视频边界只 backward，不 step；直到 `is_video_end` 才归一化累计梯度、执行 optimizer/scheduler 并提交事务。

### E.2 十个问题的直接答案

#### 1. `StreamingFeatureDataset` 是否生成完整视频连续 chunks？

**是。`[CODE-VERIFIED]`**

对视频 (v) 的 (T_v) 个 cached tokens，生成：

[
C_v=\left\lceil\frac{T_v}{64}\right\rceil
]

个连续 chunks，从 token 0 一直到 (T_v-1)，没有随机 anchor 或 episode 截取。

#### 2. `optimizer_events_per_epoch` 是否等于完整视频数？

**是。`[CODE-VERIFIED]`**

```python
optimizer_events_per_epoch = len(packet_manifests)
```

Q2 fit-core 配置意图为 160 videos；外部 manifest 必须在 runtime 精确匹配该集合。

#### 3. 一个 optimizer event 包含多少 tokens、bins 和 chunks？

对于视频 (v)：

```text
temporal tokens      = T_v
valid supervised bins = T_v
chunks               = ceil(T_v / 64)
optimizer events     = 1
```

这里“supervised bin”需要限定：每个有效 token 都产生完整 step loss 调用；birth/alive 覆盖背景与动作状态，class/start/end 根据 binding 和 risk mask 可能为零或被 mask。

实际 (T_v) 的 min/median/p95 不在仓库中，需由外部 fit-core cache manifest 统计。

#### 4. loss 如何跨 chunks 归一化？

每个 chunk 内：

[
\bar L_c=\frac{1}{n_c}\sum_{t\in c}L_t
]

detector 返回：

```python
_optimizer_weight = n_c
```

engine 对 (n_c\bar L_c) backward。视频结尾再把累计 gradient 除以：

[
T_v=\sum_c n_c
]

所以单次 optimizer update 对应：

[
L_v=\frac{1}{T_v}\sum_{t=1}^{T_v}L_{v,t}
]

代码证据位于 chunk loss 与 engine gradient normalization。

**长视频不会获得更大的 optimizer-event 权重。** 它们消耗更多计算，但仍只产生一次 AdamW update。当前隐含风险是：

[
L_{\text{current}}
=\frac{1}{V}\sum_v\frac{1}{T_v}\sum_tL_{v,t}
]

而不是：

[
\frac{1}{\sum_vT_v}\sum_v\sum_tL_{v,t}.
]

这一区别必须在 CRS‑EPS importance weighting 前先冻结。

#### 5. 所有 tokens 是否参与 Python sequential unroll？

**是。`[CODE-VERIFIED]`**

`_scan_and_decode` 和 `train_episode` 都使用显式 Python `for` 逐 token 调用 `head.step`。

#### 6. 减少 optimizer steps 是否真正减少主要成本？

**不能据此成立。`[INFERENCE]`**

当前改动减少的是：

* AdamW mutation 次数；
* scheduler calls；
* event attestation；
* transaction commit；
  -部分 logging/launch overhead。

没有减少：

* cached token loading；
* temporal projection；
* attention/GRU state update；
* per-token supervision transition；
* per-token loss；
* temporal backward。

因此“从 packet-wise 变为 video-wise optimizer event”可能降低 optimizer/dispatch overhead，但不是 CRS‑EPS 所主张的 temporal-token reduction。

#### 7. cross-chunk detach 意味着什么？

**数值状态连续，梯度历史中断。**

chunk (q) 的状态：

[
R_q=F_\theta(X_q,R_{q-1})
]

下一 chunk 实际使用：

[
\widetilde R_q=\operatorname{stopgrad}(R_q)
]

所以：

[
\frac{\partial L_{q+1}}{\partial R_q}=0.
]

这意味着：

* 后一 chunk 可以读取前面构造的 query/memory/slot 数值；
* endpoint loss 不能给更早 chunk 的 birth/state-construction 路径分配 credit；
* 长动作可以在 forward state 上跨 chunk 延续；
* 不能声称模型从长动作 endpoint 对早期 birth 学到全跨度 temporal credit；
* 当前可训练 BPTT 长度主要受 64-token chunk 限制，192-token memory 中来自前 chunk 的部分是 detached context。

#### 8. fixed/rematch 是否只改变 post-birth loss binding？

**代码干预面上是；正式实验闭包上尚未完全证明。**

配置测试先删除 `work_dir` 和 binding mode，剩余配置完全一致。

supervision 实现共享：

* birth candidates；
* first-crossing Hungarian birth assignment；
* canonical binding；
* retirement；
* capacity与exhaustion。

rematch 只重算当前 prefix 的 `loss_bindings`。同类重叠测试证明 canonical state 不变，而 rematch 交换 loss bindings。

双活跃 detector probe 进一步验证 birth/lifecycle/masks 相同，class/start binding 及其 gradient 不同。

但正式 run 仍缺完整 paired trace，因此结论只能是 `PARTIALLY IDENTIFIABLE`。

#### 9. 当前 route 是否包含 CRS‑EPS 元素？

**不包含。**

除“最终完整时间顺序验证/评测”这一共同原则外，其余 CRS‑EPS 核心均不存在。

#### 10. 当前 route 应叫什么？

建议准确命名为：

```text
Q2 Full-Chronological Cached-Feature
Video-Episode Truncated-BPTT Training
```

或中文：

```text
Q2 完整视频顺序缓存特征训练
（视频级 optimizer event，chunk 间梯度截断）
```

不要称为：

```text
CRS-EPS
prefix-episode training
event-centric sampling
prefix-parallel training
raw-video end-to-end
```

### E.3 Accounting Table

| Unit                           |                   Current full-chronological cached route |                                       Proposed CRS‑EPS | Comparable?          | Required normalization                                          |
| ------------------------------ | --------------------------------------------------------: | -----------------------------------------------------: | -------------------- | --------------------------------------------------------------- |
| videos                         | 1 complete video/event；intended 160 fit-core videos/epoch |                         episodes sampled across videos | 否，不能按 event 数        | 先定义 (p(v))：video-uniform 或 token-uniform                        |
| chunks                         |                                (\lceil T_v/64\rceil)，全部处理 | implementation 未存在；每 episode 可包含一个或多个 execution chunks | 否                    | 按实际 processed tokens 计                                          |
| temporal forward tokens        |                                               (T_v)/event |       historical candidate 192+4–8，即最多约196–200/episode | 否                    | 报总数和 tokens/s                                                   |
| temporal backward tokens       |              当前 chunk 内全部有效 tokens；跨 chunk state detached |            burn-in forward-only 或 selective-TBPTT 尚未冻结 | 否                    | 单独记录 forward/backward                                           |
| supervised bins                |                              (T_v)，但各 loss 有不同 risk masks |                                   4–8 consecutive bins | 否                    | 按 per-loss numerator/denominator 分开                             |
| visual forward frames          |                           0，cache extraction 不在训练 event 内 |                                             Stage 1 为0 | 表面可比，但遗漏 cache build | cache build 单独摊销或完整列出                                           |
| visual backward frames         |                                                         0 |                                             Stage 1 为0 | 是，仅限 cached Stage 1  | 固定为0并禁止 e2e claim                                               |
| optimizer events               |                                         (V)/epoch；1/video |                                  (N_{\text{episodes}}) | **绝对不可直接比较**         | 仅作为辅助计数                                                         |
| wall time                      |                                               未正式 profile |                                                    未实现 | 未知                   | 共同硬件、共同 workload contract                                       |
| GPU-hours                      |                                               未正式 profile |                                                    未实现 | 未知                   | primary resource endpoint 候选                                    |
| data wait time                 |                                                       未记录 |                                                    未实现 | 未知                   | profiler 分离 host wait / CUDA time                               |
| effective sample size          |                                 无 proposal sampling；当前不报告 |                        必须全局及按 class/lifecycle/video 报告 | 否                    | 根据最终 weights 计算                                                 |
| quality-equivalent supervision |                                                       未定义 |                                                    未定义 | 否                    | 固定 weighted supervised bins/ESS，并加 full-objective fidelity gate |

---

## F. Is CRS‑EPS Actually Implemented?

**结论：`NO`。**

| CRS‑EPS requirement                          | Current code   |
| -------------------------------------------- | -------------- |
| start-centered anchors                       | NO             |
| endpoint-centered anchors                    | NO             |
| ongoing anchors                              | NO             |
| hard-background anchors                      | NO             |
| uniform chronological component              | NO             |
| immutable episode manifest                   | NO             |
| 192-token burn-in/context                    | NO             |
| burn-in loss mask                            | NO             |
| 4–8 supervised suffix bins                   | NO             |
| active-at-entry reconstruction contract      | NO             |
| `pi_t` inclusion probability                 | NO             |
| duplicate-window union inclusion probability | NO             |
| IPW / capped IPW / SNIPW                     | NO             |
| ESS reporting                                | NO             |
| sampled/full loss comparison                 | NO             |
| sampled/full gradient comparison             | NO             |
| full chronological validation/evaluation     | YES，作为当前常规流式路径 |

历史 CRS‑EPS 文档明确规定 192 context + 4–8 supervised bins、五类 mixture、`pi_t`、IPW/SNIPW、ESS 和 full-stream gold comparison。

当前代码与这些要求之间不是“PARTIAL implementation”，而是**训练协议尚未接入**。

### 对“gold reference”的修正

当前 full route 可以称为：

> **当前 cached-token、per-video-normalized、chunk-detached objective 的 exhaustive chronological reference。**

不能不加限定地称为：

> unbiased full-sequence gold training。

原因是它同时固定了三个非中性的选择：

1. video-uniform 而不是 corpus-token-uniform；
2. chunk-boundary stop-gradient；
3. 每视频一次 adaptive optimizer update。

这三项都必须进入 sampled/full fidelity 的正式定义。

---

## G. `P0-LAUNCH-WORKDIR` Reverification

### G.1 确认的执行顺序

Ticket builder：

```python
cfg_overrides = dict(args.cfg_options or {})
ticket = build_launch_ticket(..., cfg_overrides=cfg_overrides)
```

即 ticket 在生成时冻结 exact override map。

之后 submit helper 才执行：

```bash
STAMP=$(date ...)
RUN_DIR=...
```

并在最终 `torchrun` argv 中追加：

```bash
--cfg-options work_dir="${RUN_DIR}/work"
```

Validator 则重新由 live argv 生成 runtime identity，并要求：

```python
ticket["runtime_identity"] == runtime_identity
```

否则 fail closed。

### G.2 现有测试为何没有关闭它

现有 launch contract test 只检查：

* builder 包含 `build_launch_ticket`；
* shell 包含 ticket 参数；
* shell 使用单进程；
* shell 包含 formal gate。

它没有实际运行 submit shell、冻结 `date`、stub `sbatch` 并比较最终 argv。

### G.3 最小正确修复

**P0 before any profile：**

1. 在 ticket 创建前生成唯一 `RUN_ID`、`RUN_DIR` 和 `WORK_DIR`。
2. 把 `work_dir` 写入 ticket 的 exact `cfg_overrides`。
3. Slurm argv 必须从 ticket 读取这组 overrides，不得独立重建第二份值。
4. 新增真实 shell integration test：

   * deterministic date/run ID；
   * temporary `RUNS_ROOT`；
   * fake `sbatch` 捕获生成脚本和 argv；
   * 断言 ticket、validator live identity、Slurm argv exact-equal；
   * 修改 `work_dir` 一个字符，确认 pre-CUDA rejection。
5. 冻结新 commit，重新生成 B0，重新独立审查。

**不建议**简单把 `work_dir` 从 runtime identity 删除。它决定输出、artifact 和可能的恢复位置；若要排除，应显式升级为独立 attested artifact-root 字段，而非静默忽略。

### G.4 Gate

```text
PROFILE=BLOCK
FORMAL_TRAINING=BLOCK
OPEN_P0=P0-LAUNCH-WORKDIR
```

---

## H. Q2 One-Factor Preliminary Audit

### H.1 已满足的部分

**[CODE-VERIFIED]**

| 项目                                  | 状态                                                    |
| ----------------------------------- | ----------------------------------------------------- |
| same source config                  | PASS                                                  |
| same cached features                | PASS by config                                        |
| same model/head/capacity            | PASS                                                  |
| same first-crossing birth           | PASS                                                  |
| same canonical lifecycle            | PASS                                                  |
| same runtime decode path            | PASS                                                  |
| same inference API                  | PASS                                                  |
| same negative alive semantics       | PASS at code level                                    |
| fixed/rematch difference            | post-birth target-to-slot loss binding                |
| single-instance null case           | losses identical                                      |
| overlapping same-class intervention | canonical state equal，loss binding differs            |
| gradient intervention               | synthetic probe confirms expected gradient difference |

### H.2 尚未闭包的部分

当前 episode audit 仅记录：

```text
binding mode
birth assignments
canonical lifecycle
loss bindings
birth mask
alive mask
endpoint slots
slot exhaustion
rematch swaps
valid supervised steps
max source frame
```

正式 paired trace 仍缺：

* immutable `episode_id` / `video_id` / source ranges；
* RNG seed、generator state、dropout-mask identity；
* input checksum；
* initial/final runtime query、memory、slot-state checksum；
* initial/final supervision-state checksum；
* negative/ignore/censor mask 的完整表示；
* endpoint risk-mask 全量 trace，而不只是 positive endpoint slots；
* 每项 loss numerator、denominator、weight；
* reset、retire、refractory 事件；
* overflow 的全部 instance IDs；
* per-parameter gradient checksum或cosine；
* optimizer/scheduler pre/post identity；
* inference output checksum。

配置中 dropout 为 0.1；即使两个 job 使用同一 seed，也没有正式 artifact 证明每个 paired event 消耗了相同 RNG stream。

### H.3 当前 full route 下的判定

当前从视频开头完整 replay，因此初始 state 是自然、非 oracle 的。对于当前 full route：

```text
Q2_IDENTIFIABILITY=PARTIALLY_IDENTIFIABLE
```

原因不是 treatment definition 含糊，而是：

* treatment 本身定义较干净；
* 正式 paired execution evidence 尚未闭包；
* 无 profile、无效果结果；
* evaluator population与cache provenance仍有外部缺口。

### H.4 CRS‑EPS 接入后的新风险

同一 episode list **不足以**保证 one-factor。

随机 episode 还必须控制 state reconstruction：

```text
same episode manifest
+ same replay rule
+ same prefix-observable initialization
+ same left-censor semantics
+ matched gold comparison
```

特别是：

* fixed binding 具有出生后的历史依赖；
* rematch active pool 依赖当前 active instances 和可用 runtime slots；
* 若动作 start 早于 192-token burn-in，但在 suffix 内仍 active/end，bounded replay 不知道它应该如何出生；
* 用 GT 直接初始化 active slot 会引入 oracle state；
* 从空 state 开始会系统性改变 fixed/rematch 的轨迹问题。

因此，CRS‑EPS 中 active-at-entry 的解决方式是**科学 blocker**，不是普通 dataloader 细节。

---

## I. Profile Denominator Preliminary Audit

### I.1 当前 profile 实际测什么

`FixedStepProfiler` 只把 successful optimizer event 当成计量单位，输出：

```text
warmup_optimizer_events
measured_optimizer_events
elapsed_seconds
peak_memory_bytes
throughput_optimizer_events_per_second
```

authenticated optimizer trace 额外记录：

```text
input_tokens
elapsed_seconds
peak_memory_bytes
precision
optimizer/scheduler/data/loss-normalization hashes
```

但没有 supervised bins、temporal backward tokens、data wait、visual frames、ESS 或 Python-unroll 时间。

### I.2 50 warmup + 200 measured 的额外问题

当前 fit-core 是 160 videos，而 profile 总计需要：

```text
50 + 200 = 250 successful video events
```

sampler 顺序固定，不随 epoch shuffle。因此 measured window 会跨 epoch：

```text
epoch 1: videos 51–160
epoch 2: videos 1–90
```

即 40 个视频会在 measured interval 内重复，而其他视频只出现一次。该 workload 可以是可复现的，但并非天然代表视频长度分布；若重复视频在长度或实例密度上有系统差异，events/s 会偏。

### I.3 哪些比较仍然勉强有效

在修复 P0 后，当前 50/200 profile 可以作为：

> fixed 与 rematch 在完全相同 full-video event order 下的 route-internal engineering comparison。

它不能作为：

> full chronological route 与 CRS‑EPS 的训练成本比较。

因为：

```text
1 full-video event = T_v tokens, all-video history
1 CRS-EPS event    = roughly 196–200 forward tokens, 4–8 supervised bins
```

两者 optimizer-event 的语义完全不同。

### I.4 Preliminary profile verdict

```text
PROFILE_CONTRACT=REPLACE
```

新的跨协议 profile 至少必须同时记录：

* wall-clock seconds；
* GPU-hours；
* peak VRAM；
* data wait；
* temporal forward tokens；
* temporal backward tokens；
* supervised bins，按 loss/risk set 分解；
* visual forward frames；
* visual backward frames；
* episodes；
* unique videos covered；
* optimizer events；
* weighted ESS；
* tokens/s；
* supervised bins/s；
* time to fixed effective supervision；
* Python sequential-unroll time fraction。

在作者冻结 primary cost endpoint 和目标风险前，不能预先选“固定 events”“固定 bins”或“固定 wall time”中的任一单分母作为唯一比较。

---

## J. Unknown Register

### J.1 已由仓库回答，不再询问

| 问题                            | 仓库答案                                          |
| ----------------------------- | --------------------------------------------- |
| 当前主科学问题是什么？                   | fixed post-birth binding vs prefix rematching |
| 当前是否 persistent-state 总体论文实验？ | 否，Q2 只是 binding study                         |
| 当前是否 raw-video adaptation？    | 否                                             |
| 当前是否 CRS‑EPS？                 | 否                                             |
| 任务是否仍为标准 On-TAD？              | 是                                             |
| world size 是否需要扩展？            | 当前锁定为1，不需要                                    |
| resume 是否需要实现？                | 当前 formal resume 明确禁止                         |
| 是否已有正式结果？                     | 没有                                            |
| Stage 2 是否已授权？                | 没有                                            |

Q2 的当前准确问题已在项目 SSOT 中明确写为 cached-feature binding study，而不是 Full PETAL 复活。

### J.2 真正需要作者回答的问题

#### Blocking before protocol implementation

| ID | 问题                                                                                                      | 为什么重要                                         | 保守默认                                                                       | 回答前可继续什么                       |
| -- | ------------------------------------------------------------------------------------------------------- | --------------------------------------------- | -------------------------------------------------------------------------- | ------------------------------ |
| Q1 | CRS‑EPS 要逼近的 primary risk 是 **video-uniform per-video mean**，还是 **corpus decision-time/token-uniform**？ | 决定 (p(v,t))、IPW、full gold normalization 和结论含义 | 保持当前 video-uniform target；不静默改变 full route                                 | 可修 P0；不可实现 weighting           |
| Q2 | full chronological route 在 hybrid 中是：无梯度 audit、tiny gradient gold subset，还是周期性参与训练？                     | 决定成本、目标漂移和 sampled/full fidelity              | tiny preregistered gradient gold subset + 完整无梯度 val/test；不做结果后决定的 finetune | 可修 P0；hybrid 实现需回答             |
| Q3 | 是否接受在 gold subset 上从视频开头 replay，以审计 active-at-entry；bounded 192 burn-in 只作为待验证 surrogate？               | 无此 gold replay 无法判断长动作状态是否被错误重建               | 接受 start-of-video replay；禁止 GT active-slot initialization                  | 可做 dataset schema；科学协议 blocked |
| Q4 | 若 start 早于 burn-in，主训练允许哪种处理：left-censor mask、earliest-observable-birth replay，还是排除该 episode？           | 直接改变长动作、fixed identity 和 end-risk supervision | 不全局排除；标记 left-censored，并要求 gold-state audit                                | CRS sampler blocked            |

#### Blocking before profile

| ID | 问题                                                                                                                      | 为什么重要                           | 保守默认                                                                  | 回答前可继续什么                        |
| -- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------- | --------------------------------------------------------------------- | ------------------------------- |
| Q5 | primary cost endpoint 是 GPU-hours、wall time、time-to-fixed-supervision，还是 time-to-fixed-quality？总 profile/gold/Q2 预算是多少？ | 不能从 200 optimizer events 推导训练成本 | GPU-hours + wall time 为资源主指标，fixed weighted supervision 为 workload 约束 | P0 可修；新 profile contract 不能最终冻结 |
| Q6 | 是否正式同意废除跨协议的“200 optimizer events 可比”假设，改为多分母报告？                                                                        | 决定 profile schema 与 result gate | 同意替换；events 仅作辅助字段                                                    | P0 可修                           |

#### Blocking before scientific training

| ID | 问题                                                                              | 为什么重要                          | 保守默认                                                                                                      | 回答前可继续什么   |
| -- | ------------------------------------------------------------------------------- | ------------------------------ | --------------------------------------------------------------------------------------------------------- | ---------- |
| Q7 | primary quality metric 与 sampled/full non-inferiority estimand 是什么？             | 无法冻结 margin、power 与 early kill | standard average temporal mAP@0.3:0.7 primary；duplicate/fragmentation 为 mechanism co-endpoints；delay为安全指标 | 只允许无效果工程测试 |
| Q8 | 是否接受：若 fixed 不优于 rematch，或增益不落在 identity-linked errors，立即停止 Full PETAL/Stage 2？ | 防止 aggregate noise 被包装成机制成功    | 接受 immediate kill                                                                                         | 可实现审计基础设施  |

#### Blocking before interpreting results

| ID  | 问题                                                                                          | 为什么重要                            | 保守默认                                                                  | 回答前可继续什么                                |
| --- | ------------------------------------------------------------------------------------------- | -------------------------------- | --------------------------------------------------------------------- | --------------------------------------- |
| Q9  | extractor exact commit/config/checkpoint/processor/normalization/raw support interval 能否恢复？ | 决定 genuine-video-level causality | 恢复不了则只称 cached-token study，不作 strict raw-video claim                  | compute-only profile可在P0后进行；效果解释受限      |
| Q10 | canonical 213 相对 locked 211 的两个差异 ID 和原因是什么？                                                | 可能产生 benchmark selection bias    | 所有结果命名为 locked-211 subset，所有 baseline 同人口重评                           | 不阻塞 compute profile；阻塞 formal reporting |
| Q11 | serving/evaluation 的真实 packet cadence、extractor cadence 和 availability clock 是什么？           | chunk 内 source time 不等于可交付时间     | training chunk 与 serving packet 分离；正式评测用 stepwise/micro-packet并记录三类时钟 | 机制训练可继续；低延迟 claim blocked               |

#### Blocking before paper claim

| ID  | 问题                                                                             | 为什么重要                                             | 保守默认                                        | 回答前可继续什么             |
| --- | ------------------------------------------------------------------------------ | ------------------------------------------------- | ------------------------------------------- | -------------------- |
| Q12 | 若 Q2 survives，是否有 FineAction/MultiTHUMOS 等 dense/overlap 第二数据集和统一 baseline 预算？ | THUMOS14 alone 很难支持 persistent identity 的广泛 claim | 首轮 kill test 不要求；任何 retained claim 必须有第二数据集 | 不阻塞首轮 mechanism kill |

reporting contract 的代码只证明系统能要求对 211/213 mismatch 给出原因；它没有提供真实两个视频及其理由。

---

## K. Provisional Decision Card

```text
Repository HEAD observed:
c56857b1e3bbbd445aa86e9e31a1e1ddd07ffe51

Immutable commit reviewed:
f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb

Research verdict:
REVISE-BEFORE-IMPLEMENTATION

Current training route:
FULL-CHRONOLOGICAL CACHED-FEATURE VIDEO-EPISODE TRUNCATED-BPTT

CRS-EPS currently implemented:
NO

Task protocol verdict:
STANDARD COMPLETION-TRIGGERED ON-TAD AT DETECTOR LEVEL;
GENUINE RAW-VIDEO CAUSALITY UNPROVEN

Q2 identifiability verdict:
PARTIALLY IDENTIFIABLE

Feature provenance verdict:
OPEN P1; CACHE INTEGRITY BOUND, EXTRACTOR CAUSALITY UNBOUND

Online clock verdict:
SOURCE CLOCK ONLY; AVAILABILITY/WALL CLOCK UNBOUND

Recommended protocol:
HYBRID — PROVISIONAL, NOT YET FROZEN

Profile contract verdict:
REPLACE

Full PETAL novelty verdict:
RECONSTRUCTION — PROVISIONAL;
FIXED BINDING REMAINS AN UNPROVEN MARGINAL DELTA

Raw-video Stage-2 permission:
BLOCKED

Profile permission:
BLOCKED

Formal training permission:
BLOCKED

Open P0:
P0-LAUNCH-WORKDIR

Open P1:
TARGET-RISK-NORMALIZATION
CRS-EPS-STATE-RECONSTRUCTION
PROFILE-DENOMINATORS
FEATURE-PROVENANCE
AVAILABILITY-CLOCK
ONE-FACTOR-PAIRED-TRACE
REPORTING-POPULATION-211

Minimum fixes before next GPU-hour:
1. Single-source immutable work_dir identity.
2. Deterministic fake-sbatch submit-shell integration test.
3. Replace event-only profile schema with multi-denominator accounting.
4. Freeze a new clean commit.
5. Regenerate B0 and obtain fresh independent PROFILE=ALLOW.

Exact next implementation step:
FIX P0-LAUNCH-WORKDIR AND ADD THE REAL SUBMIT-SHELL
FAKE-SBATCH IDENTITY-CLOSURE TEST.

Exact protocol kill condition:
KILL CRS-EPS FOR Q2 IF ACTIVE-AT-ENTRY/LONG-ACTION STATE
RECONSTRUCTION REQUIRES ORACLE GT SLOT INITIALIZATION, LEAVES
ANY TARGET DECISION BIN WITH ZERO SUPPORT, OR CANNOT MATCH
FULL-STREAM GOLD STATE/LOSS ON THE PREREGISTERED GOLD SUBSET.

Confidence:
HIGH on repository/code/dataflow/P0/CRS-EPS-absence;
MEDIUM on HYBRID recommendation pending Q1-Q6;
UNKNOWN on effectiveness.
```

```text
ROUND=1
RESEARCH_VERDICT=REVISE-BEFORE-IMPLEMENTATION
CURRENT_ROUTE=FULL_CHRONOLOGICAL_CACHED_VIDEO_EPISODE_TRUNCATED_BPTT
CRS_EPS_IMPLEMENTED=NO
RECOMMENDED_PROTOCOL=HYBRID_PROVISIONAL
Q2_IDENTIFIABILITY=PARTIALLY_IDENTIFIABLE
PROFILE_CONTRACT=REPLACE
FULL_PETAL_NOVELTY=RECONSTRUCTION_PROVISIONAL
RAW_VIDEO_STAGE2=BLOCKED
PROFILE=BLOCKED
FORMAL_TRAINING=BLOCKED
NEXT_STEP=FIX_P0_LAUNCH_WORKDIR_AND_ADD_FAKE_SBATCH_INTEGRATION_TEST
OPEN_P0=P0-LAUNCH-WORKDIR
OPEN_P1=TARGET_RISK,STATE_RECONSTRUCTION,PROFILE_DENOMINATORS,FEATURE_PROVENANCE,AVAILABILITY_CLOCK,PAIRED_TRACE,REPORTING_POPULATION
```

作者下一轮只需按 `Q1`–`Q12` 回答；其中最先决定协议的是 `Q1`–`Q6`。
