# A. Executive Verdict

## A.0 Unknown Register

审计时间：**2026-07-11 08:41 EDT**。代码锚点为公开分支 `codex/online-tad-clean-20260702`、提交 `bfd0608b2996cba30d158d741ee193476a5078df`。

| ID  | 会改变结论的未知项                                                | 状态                                | 对结论的影响                                      |
| --- | -------------------------------------------------------- | --------------------------------- | ------------------------------------------- |
| U1  | 指定 SHA 是否仍为公开分支最新提交                                      | `known from source`               | 已确认。公开提交序列中该 SHA 位于最前，且默认分支即目标分支。           |
| U2  | 是否已有未公开的正式 PCEH 多随机种子结果                                  | `unknown`                         | 若有显著且稳定结果，可影响代码资产价值，但不会自动修复任务新颖性。           |
| U3  | `152,670 packets/epoch`、`7 h/epoch`、`210 GPU-hours/seed` | `known from user-provided record` | 尚未由公开日志独立复核；用于可行性排序，不当作 GitHub 事实。          |
| U4  | 当前 PCEH endpoint-only 与 PCEH pilot 的实际曲线                 | `requires pilot experiment`       | 决定旧路线是否连“有效 baseline”都能成为，而不是决定是否为最佳新题。     |
| U5  | 实际部署中一次误报、漏报、晚报各自的代价                                     | `requires user/domain expert`     | 会改变风险控制、主动感知、遗漏监测三条路线的排序。                   |
| U6  | 是否能获得连续数小时乃至数天的真实负流，而非 THUMOS 剪接流                        | `requires user/domain expert`     | 是 anytime-valid alarm 最重要的外部有效性未知项。         |
| U7  | 是否能获得 force、IMU、接触、工具状态或多视角同步数据                          | `requires user/domain expert`     | 决定“物理事件—视觉证据—提交”三时钟路线是否可做成主线。               |
| U8  | “最早充分视觉证据”能否得到稳定的人类共识                                    | `requires pilot experiment`       | 若跨标注者不稳定，三时钟任务需改为部分识别区间，而不能用单一时间点。          |
| U9  | 部署期是否自然产生延迟反馈，例如用户纠正、质检结果或失败日志                           | `requires user/domain expert`     | 决定 continual/adaptive 路线是否是真实任务还是模拟包装。      |
| U10 | 是否愿意把贡献重心转为 task/evaluation/theory，而非新网络                 | `requires user/domain expert`     | 若必须做纯模型论文，推荐路线会明显变弱。                        |
| U11 | 能否对相机、编码、通信和 GPU 分别测量能耗与唤醒延迟                             | `requires user/domain expert`     | 决定风险约束主动感知能否从 replay 实验升级为系统论文。             |
| U12 | 视频分数在长负流中是否存在足够稳定的 null structure                        | `requires pilot experiment`       | 若背景分数极端非平稳，简单 conformal/e-process 不能获得有效保证。 |
| U13 | 可接受的统计假设强度：交换性、块交换性、条件 e-value 或仅经验风险                    | `requires user/domain expert`     | 决定推荐路线更适合 NeurIPS/ICLR 还是 CVPR。             |
| U14 | 目标投稿窗口和可投入标注周期                                           | `unknown`                         | 短周期更偏风险告警；长周期才可能完成物理传感数据路线。                 |

## A.1 总判定

### 1. 传统 closed-set On-TAL

**不建议再作为主研究问题投入。**

CAG-QIL 已经确立了无未来、不可回改的 On-TAL 任务；OAT 覆盖了滑窗 proposal、在线过滤和 early metric；MATR/HAT 覆盖长期记忆与起止分工；ActionSwitch 覆盖 class-agnostic、同类重叠和状态切换。2026 年 OZ-TAL 又进入在线零样本，OnPoint 进入点监督与离线教师蒸馏。继续在 THUMOS14 上做“causal encoder + memory + endpoint/emission head + latency loss”，边际科学价值很低。 ([CVF Open Access][1])

传统 On-TAL 仍可保留三种角色：

* **协议与工程测试床**：验证 no-future、no-revision、packet causality；
* **旧任务基线**：证明新任务不是因为模型能力不足才成立；
* **特征/分数供应器**：为新风险控制、遗漏监测或主动感知提供 causal evidence score。

但它不应继续支配选题。

### 2. online temporal localization 中仍有价值的部分

真正有价值的不是继续回归 `[start,end,class]`，而是以下尚未被旧 benchmark 正确衡量的问题：

* 无限运行时，误报风险如何随时间、类别和摄像头数量增长；
* 物理状态真正改变、视觉上首次可辨认、系统作出不可撤销决定，三者为何不同；
* 预期事件未发生、动作被中断、动作失败或恢复时，何时才能合法告警；
* 模型是否可以在看到像素之前决定何时唤醒传感器，而不只是看完再压缩；
* 部署期获得延迟纠正后，如何只改善未来预测而不篡改过去结果；
* 输出是否应为风险受控告警、状态改变、遗漏义务、证据集合或干预决定，而非完整 span。

### 3. generic streaming VideoLLM reasoning

**已经高度拥挤。**

2025–2026 年已有 StreamFormer、StreamForest、LiveStar、StreamReady、Thinking-QwenVL、SelectStream、StreamMem、Vista、ProtoKV，以及多条“边看边思考”路线。它们已覆盖 causal backbone、长期 memory、KV 压缩、query arrival、readiness、response/silence、first-sufficient-evidence、主动证据检索和多轮推理。再做“一个更好的 memory + readiness head + VLM”基本属于红海。 ([arXiv][2])

### 4. GO / REDEFINE / ABANDON

* **GO**：把连续视频重新定义为**无限次、不可回改的语义事件检验问题**，研究 anytime-valid false-alarm/FDR control 与 detection delay。
* **HIGH-RISK GO**：建立“物理事件时间—首次视觉可观测时间—系统提交时间”的三时钟任务。
* **REDEFINE**：将 On-TAL 代码降级为 causal score generator、协议审计器和 baseline。
* **ABANDON**：把当前 PCEH 六头结构直接修补后，继续在 THUMOS14 上以 latency-aware mAP 作为论文主线。
* **ABANDON**：generic streaming VLM memory/readiness 组合。

### 5. 置信度

* 总体结论置信度：**0.82**
* “传统 closed-set On-TAL 不值得作为主线”：**0.91**
* “anytime-valid semantic alarm 是当前最佳可证伪路线”：**0.82**
* “三时钟物理可观测性可成为强 CVPR 任务”：**0.73**
* 最大未知项：**长真实负流上的 null 非平稳性，以及用户是否拥有能代表物理状态的同步传感数据。**

---

# B. Repository Reality Check

## B.1 版本事实

公开仓库在审计时的目标分支 HEAD 为：

```text
branch: codex/online-tad-clean-20260702
commit: bfd0608b2996cba30d158d741ee193476a5078df
message: fix: make PCEH pilots finish within allocation
commit time: 2026-07-10T08:40:05Z
```

该提交没有引入新的科学机制，只将两个 3-epoch pilot 的验证从 epoch 0 推迟到 epoch 2，并把默认 pilot Slurm 上限由 24 小时调为 48 小时；测试仍明确断言 `formal_training_ready is False`。

## B.2 当前代码实际实现了什么

### 严格流式协议资产

1. **按视频顺序组织 packet**
   streaming dataloader 禁止 shuffle，并使用 chronological batch sampler。

2. **增量处理新 packet**
   `PCEHOnlineDetector` 只编码新输入 packet，并保留有限 feature cache；测试覆盖了 packet chronology、跨视频状态隔离、source-frame trace 和 bounded cache。

3. **no-future provenance ledger**
   输出记录 `max_raw_frame_read`、`max_cache_source_frame`、`emit_frame`、`end_frame`，并检查 future source、future end、负延迟和非单调 emission。

4. **不可回改的 prefix emission**
   streaming-safe 模式关闭视频级 sliding-window merge/NMS；历史 emission 不被后续结果修改。

5. **终止元数据的部分隔离**
   非最终 packet 会删除 `duration`、`num_frames`、`video_end_frame` 等 terminal-only metadata。

6. **可执行 smoke/contract 测试**
   smoke 默认只跑至多 8 个 packet，可选一次训练更新，并输出是否 formal-ready；这证明管线可执行，不构成模型有效性证据。

这些都是有价值的**科研基础设施资产**。

## B.3 当前 PCEH 实际没有实现什么

### 1. 没有 instance-aware risk set

训练 target 是 `num_classes` 维向量。相同类别的多个实例通过 `any_started/any_ongoing/any_completed` 聚合；解码状态也是每类最多一个 `ActiveActionTrack`。同类重复、相邻或重叠实例因此不是独立风险过程。

### 2. endpoint 与 emission target 没有被分离

对于每个 GT endpoint crossing：

```python
if end_crossed:
    end_event[label] = 1.0
    emit_event[label] = 1.0
```

同时，`emit_allowed` 在整个 endpoint 后 delay window 内持续为 1，`late_target` 在超过窗口后仍持续为 1。最终 emission BCE 的正目标是三者的并集。因此训练没有唯一的 first-commit target，也没有 post-commit masking。

### 3. 解码强制 `predicted end == commit time`

PCEH 只有在同一当前 packet 上同时满足：

```text
end hazard
AND completion
AND emission hazard
```

才 emit；写入 ledger 时：

```text
end_frame = current_frame
emit_frame = current_frame
```

所以即使概念叙事声称物理 endpoint 与决策 emission 不同，执行代码仍把二者设为同一时间。

这还造成更深的统计矛盾：若 `end_hazard` 真是离散 hazard，它理应在物理 endpoint crossing 时尖峰；但若系统需要等待后续证据才提交，稍后 packet 又必须重新把 endpoint hazard 拉高，才能满足联合 emit 条件。

### 4. 没有 first-event / first-commit supervision

当前 target 会把 endpoint 后的多个 prefix 标为 emission-positive。没有：

* instance ID；
* already-committed state；
* first positive only；
* commit 后 loss mask；
* 对“已经错过首次合法提交”的单独惩罚；
* delayed synthetic case 中 `pred_end < commit` 的契约测试。

### 5. 没有跨 packet 的可微分事件学习

配置设置 `detach_stream_state=True`。feature cache 可用于后续 forward，但梯度不会沿完整视频流穿越 packet；Python lifecycle state 也不是可微状态。因此它更接近**逐 packet 学习 + 推理时持久状态**，而不是完整 prefix-censored sequence training。

### 6. 没有论文级实证

公开 README 明确将这些配置描述为 validation candidates；三个 epoch 的 pilot 不是正式结果，所有 PCEH 配置仍为 `formal_training_ready=False`。

### 7. GT taint 尚未被充分证明为安全

`StreamingRawFrameDataset` 会在**每个训练 packet**中附带完整绝对 `stream_gt_segments` 和 `stream_gt_labels`，然后由 detector 在 forward 内构造 prefix target。当前所见代码只在 `return_loss=True` 时使用这些字段，因此我没有发现明确的 inference leakage；但完整未来 GT 已进入模型 API，仍应有自动 taint test，证明 backbone、projection、head、decode 和 metadata 路径无法读取。

## B.4 可复用资产

| 资产                                   | 可复用程度 | 推荐用途                                |
| ------------------------------------ | ----: | ----------------------------------- |
| chronological packet dataset/sampler |     高 | 所有严格流式新任务                           |
| no-future read trace                 |     高 | 风险告警、主动感知、系统压力测试                    |
| immutable emission ledger            |     高 | anytime alarm、延迟反馈学习                |
| single-rank state-safe evaluator     |    中高 | 小规模 pilot；后续需 video-contiguous DDP  |
| raw-frame packet loader              |     高 | replay 与真实设备流                       |
| budgeted latency evaluator           |     中 | 可保留为辅助指标，但不能做最终效用                   |
| frozen SigLIP causal score path      |     中 | 48 小时证伪中的 score generator           |
| PCEH six-head target                 |     低 | 仅作为失败 baseline，不建议沿用                |
| class-keyed active track             |     低 | 应替换为 instance/candidate-level state |
| 当前 THUMOS packet training plan       |     低 | 成本高、科学问题弱                           |

## B.5 最危险的锚定效应

1. 因为已有 emission ledger，就误以为论文必须研究 emission head。
2. 因为已有 THUMOS packet loader，就默认新任务必须继续输出 THUMOS span。
3. 因为 endpoint/emission 已命名为两个 head，就误认为二者在监督和解码中已被分离。
4. 因为 no-future 单元测试通过，就误认为模型已有科学正确性。
5. 因为训练成本很高，就产生 sunk-cost pressure，继续给旧路线增加 loss。
6. 因为 current code 能支持 latency metric，就忽略无限流误报、遗漏义务和传感成本等更真实目标。

---

# C. Field and Assumption Map

## C.1 任务谱系与 2026 年边际价值

| 任务                                        | 真实用户需求           | 现 benchmark 的主要缺口             | 2026 边际价值 | 判定                      |
| ----------------------------------------- | ---------------- | ----------------------------- | --------: | ----------------------- |
| closed-set On-TAL                         | 某些受控监控、体育、工业动作实例 | 多为离线数据重放；封闭类别；mAP；短视频；无真实误报成本 |         低 | **放弃主线**                |
| open-vocabulary / zero-shot On-TAL        | 未知动作、开放环境        | OZ-TAL 已占据直接入口；VLM 仍难细粒度时序判别  |        中低 | 只做强 baseline            |
| online temporal grounding                 | 用户给定查询后的流式定位     | 查询到达时间、历史/未来证据异步、无限 memory    |         中 | 保留，但 generic memory 已拥挤 |
| online TAS                                | AR/工业过程中的当前步骤和进度 | 多为标准流程；对遗漏、失败、恢复处理不足          |        中高 | 与义务/失败监测合并              |
| streaming VideoLLM reasoning              | 实时助手、多轮问答        | benchmark 泛化、真实交互成本、答案可靠性     |     中，但拥挤 | 不做 generic 模型           |
| continual/adaptive video learning         | 个体、环境和设备持续变化     | 大多仍是离线训练、在线推理；反馈协议缺失          |         高 | **值得重定义**               |
| active perception / sensing               | 电池、隐私、网络、算力有限    | 大量工作只在“已看见后”压缩，不控制未观察风险       |         高 | **值得做**                 |
| outcome/failure/interruption/intervention | 机器人、质检、辅助系统的真实决策 | 动作标签不等于成功；干预成本未进入目标           |   高，但迅速拥挤 | 做窄而可证伪的问题               |

On-TAL 的定义、在线 proposal、memory、class-agnostic overlap 已有明确覆盖；OnVTG、OpenHOUSE、ProTAS 和 streaming VLM 又扩展了 grounding、层次描述和进度。 ([CVF Open Access][1])

## C.2 已占据的创新区域

以下不能再作为单独主贡献：

* decision history、MDP、state switch；
* online boundary refinement、online suppression；
* generic long-term memory、hierarchical event memory；
* class-agnostic overlapping instance detection；
* zero-shot/unseen On-TAL；
* point supervision + offline-to-online distillation；
* progress head + task graph；
* query-evidence asynchrony；
* readiness / first-sufficient-evidence / when-to-answer；
* query-agnostic KV memory、scene compression、prototype memory；
* generic response/silence decoding；
* system-status-aware depth/resolution；
* generic proactive intervention/OOP recovery。 ([arXiv][3])

## C.3 Assumption Demolition

| Assumption            | Where it fails         | Real consequence          | Research opportunity                | Closest threat                 |
| --------------------- | ---------------------- | ------------------------- | ----------------------------------- | ------------------------------ |
| action 是单标签、单粒度、非重叠   | egocentric、多人物、同类同时发生  | class-keyed grouping 合并实例 | instance-free alarm 或组合事件过程         | ActionSwitch                   |
| action 有清晰 start/end  | 平滑接触、意图形成、工具释放         | 标注者和数据集边界不一致              | 多标注者集合、物理锚点                         | Trespassing the Boundaries     |
| endpoint = completion | 动作运动结束后结果尚未可见          | raw latency 混合感知与算法延迟     | 三时钟 observability                   | StreamReady/Thinking-QwenVL    |
| completion = success  | 错误组装、掉落、未锁紧            | “检测到动作”却错误判断任务完成          | outcome/state verification          | PREGO/HoloAssist               |
| query/taxonomy 预先已知   | 用户事后询问、开放事件            | memory 无法按查询定向保存          | delayed-query coverage              | AViLA/OZ-TAL                   |
| 视频连续且固定帧率             | 丢帧、网络抖动、热降频            | 以 frame index 表示的延迟失真     | 三时钟协议和 perturbation benchmark       | SAN                            |
| 所有帧同样值得计算             | 稀疏关键接触、长背景             | 无效能耗和隐私采集                 | risk-constrained sensing            | AdaFrame/AVP                   |
| 部署期间不学习               | 个体习惯、设备、光照和任务漂移        | 静态模型长期退化                  | delayed-feedback continual learning | on-device W-TAL/DAM            |
| 人工边界是真值               | annotator disagreement | mAP 奖励拟合单一主观标签            | calibrated boundary sets            | boundary-uncertainty TAL       |
| mAP 是最终效用             | 长负流、安全告警、低基率事件         | 少量 FP 可淹没系统               | anytime risk/FDR–delay              | video anomaly false-alarm work |
| 低延时是唯一成本              | 能耗、隐私、打断、通信            | 更快但更耗电或更扰民                | multi-cost utility                  | SAN/HoloAssist                 |
| 单视频预测相互独立             | 多摄像头、多类别、车队            | 总误报率随规模爆炸                 | fleet-level online FDR              | e-GAI/e-closure                |
| 用户只需要检测结果             | 医疗、工业、机器人              | 无法决定是否信任和干预               | evidence/risk certificate           | Thinking-QwenVL/AVP            |

边界不一致已在多套 egocentric 数据上得到直接观察；ActionSwitch 则明确说明同类和并行动作会破坏普通 per-class grouping。 ([arXiv][4])

## C.4 真正仍未解决的问题

1. 如何在**任意运行时长**上约束至少一次误报的概率。
2. 如何在类别、摄像头、用户和事件 proposal 持续增加时控制在线 FDR。
3. 如何在时序依赖和环境漂移下构造有效的 video e-values。
4. 如何区分物理状态变化、视觉可观测性和模型提交延迟。
5. 如何评价视觉上本来不可能立即观察到的 endpoint。
6. 如何检测“应发生但未发生”的 omission，而不是只检测已经发生的动作。
7. 如何在动作失败、被打断、恢复或替代执行时定义 event lifecycle。
8. 如何在看到像素前决定是否唤醒相机，并对漏检风险负责。
9. 如何把真实 wall-clock、sensor-clock、compute completion clock 和 prediction clock 分开。
10. 如何在反馈延迟到达后更新模型，同时保持过去输出不可修改。
11. 如何衡量流式模型在真实长期 negative prevalence 下的可靠性。
12. 如何建立多标注者、物理传感器或 outcome 状态所支持的非单点边界监督。
13. 如何把用户打断成本和错误干预成本纳入视频理解目标。
14. 如何在原始视频不能长期保存时保证未知未来查询的 coverage。
15. 如何证明所谓“及时回答”不是通过少报、拒答或牺牲 recall 获得。
16. 如何将 streaming benchmark 从离线视频重放升级为资源、传感、查询和反馈都异步的闭环。

---

# D. Raw Divergence Portfolio

下列 36 个卡片在本阶段**不排序、不合并**。其中 34 个不以传统 `[start,end,class]` 为必要输出，29 个可在不训练大型视频模型的条件下进行第一次证伪，27 个显式借鉴视频之外的统计检验、控制、runtime verification、传感器网络、隐私或人因领域。

## D.1 Task / Benchmark Lens

### T01 — Long-Horizon Semantic Alarm

**Problem failure:** clip-level mAP 不揭示误报随运行时长累积。
**Who cares and why:** 安全监控、工业告警、机器人，单次误报可能触发昂贵响应。
**New task or mechanism:** 给定语义事件，在无限 prefix 上输出不可撤销 alarm；主指标为 anytime false-alarm/FDR 与 detection delay。
**Why a larger model/memory is insufficient:** 这是重复检验与阈值有效性问题，不是表征容量问题。
**Closest known threat:** 带误报界的视频异常检测和 conformal martingale monitoring。 ([arXiv][5])
**Fastest falsification:** 把 background 区间拼成 5 分钟至 8 小时负流，检查固定阈值的 family-wise false alarm 是否随时长显著上升。

### T02 — Three-Clock Event Observability

**Problem failure:** GT endpoint、首次可见证据和系统 commit 被混成一个时间。
**Who cares and why:** 接触、装配、状态转换、医疗动作的视觉证据常晚于物理变化。
**New task or mechanism:** 标注 `τ_phys`、`τ_vis`、`τ_commit`，分别评价 sensing lag 和 algorithmic lag。
**Why larger model is insufficient:** 无模型能从尚未留下视觉证据的事件中恢复信息。
**Closest threat:** StreamReady/Thinking-QwenVL 研究 sufficient evidence，FEEL 提供同步 force，但尚未形成三时钟 endpoint 任务。 ([arXiv][6])
**Fastest falsification:** 对 100 个接触/释放片段做 force timestamp 与逐 prefix 人类判断。

### T03 — Deadline-Censored Omission Monitoring `[可能不应做检测]`

**Problem failure:** 最重要的失败可能是事件没有发生，例如未锁紧、未洗手、未服药。
**Who cares and why:** 工业、临床、程序辅助真正需要的是义务违例，而非动作 span。
**New task or mechanism:** 输入 precondition、required event、deadline、allowed recovery；输出 earliest legally valid violation alarm。
**Why larger model is insufficient:** “没有发生”只有结合期限和程序语义才可判定。
**Closest threat:** PREGO、ProTAS、EgoProactive 已研究错误、进度与恢复，但 omission 仍不是一等输出。 ([arXiv][7])
**Fastest falsification:** 从公开 procedural 视频构造自然/编辑 omission，测试普通 action detector 是否系统性漏掉。

### T04 — Adversarial Delayed-Query Memory Coverage

**Problem failure:** query 在关键证据消失很久后到达，现有 memory benchmark 多评价平均 QA。
**Who cares and why:** wearable assistant、审计和事故回溯。
**New task or mechanism:** 在固定字节预算下，对预先未见的 query family 测 worst-case recall/regret。
**Why larger memory is insufficient:** 固定容量下必须明确丢失了哪些可查询事实。
**Closest threat:** StreamMem、Vista、ProtoKV 已覆盖 query-agnostic/delayed query memory。 ([arXiv][8])
**Fastest falsification:** 对现有 memory 方法施加 adversarial late queries，观察平均分与 worst-case coverage 是否分离。

### T05 — Prefix-Valid Boundary Confidence Sets

**Problem failure:** 单一人工边界掩盖真实标注分歧。
**Who cares and why:** 高 tIoU 评价和边界监督可能奖励不可重复的标签习惯。
**New task or mechanism:** 输出 start/end 的 calibrated set 或 region，而非单点；评价 coverage 与 set width。
**Why larger model is insufficient:** 标签本身不是确定量。
**Closest threat:** Gaussian boundary uncertainty、elastic boundaries、boundary distributions 已部分覆盖。 ([arXiv][9])
**Fastest falsification:** 对 200 个片段重新多标注，测单点 GT 的 inter-annotator tIoU 上限。

### T06 — Real-Stream Perturbation Invariance Suite

**Problem failure:** 同一视频以不同 packet、帧率、丢帧和计算延迟播放时，模型结论可能改变。
**Who cares and why:** 真实网络和设备不会稳定按 benchmark FPS 运行。
**New task or mechanism:** 记录 sensor time、arrival time、compute-finish time、decision time，并测试语义不变性。
**Why larger model is insufficient:** 错误来自协议和时钟混淆。
**Closest threat:** SAN 处理系统负载，但没有完整 event-level invariance protocol。 ([CVF Open Access][10])
**Fastest falsification:** 对相同视频施加 jitter/drop/stall，检查模型排名是否反转。

## D.2 Learning Lens

### L01 — Delayed-Feedback Continual Event Learning

**Problem failure:** “online”通常只表示推理在线，模型从不利用部署反馈。
**Who cares and why:** 个体动作、环境和设备持续漂移。
**New task or mechanism:** label/correction 延迟到达；过去输出不可改；只评价未来 prequential utility 与 forgetting。
**Why larger model is insufficient:** 容量不能自动吸收新反馈或防止错误更新。
**Closest threat:** on-device W-TAL、continual VideoQA 和 delayed-feedback VLA。 ([arXiv][11])
**Fastest falsification:** 用时间排序的 domain shift 和延迟标签模拟，比较 static 与小 adapter update。

### L02 — Privileged Physical-Sensor Transition Learning

**Problem failure:** 视觉 boundary label 不能可靠代表接触、受力或机构锁止。
**Who cares and why:** 机器人和装配需要物理状态而非动作外观。
**New task or mechanism:** 训练时用 force/IMU/contact，部署时 video-only，学习 latent physical transition posterior。
**Why larger model is insufficient:** RGB 监督没有提供物理参照。
**Closest threat:** FEEL 已用 force 做 contact segmentation 和自监督预训练。 ([arXiv][12])
**Fastest falsification:** 冻结视觉特征，仅训练小 transition probe，比较人工边界与传感边界的可预测性。

### L03 — Self-Supervised Object-State Transition Discovery `[可能不应做动作检测]`

**Problem failure:** action taxonomy 不稳定，而 before/after object state 更可验证。
**Who cares and why:** 通用操作理解和机器人。
**New task or mechanism:** 从状态持久性、可逆性和 before/after consistency 自监督发现 change points。
**Why larger model is insufficient:** 需要状态守恒和持久性归纳偏置。
**Closest threat:** boundary-sensitive pretraining、force prediction 和一般视频自监督。 ([arXiv][13])
**Fastest falsification:** 在 Something-Something/装配片段上比较 discovered transition 与 object-state annotations。

### L04 — Null-First Rare Event Learning

**Problem failure:** 稀有危险事件正样本少，普通分类训练无法刻画长时间正常过程。
**Who cares and why:** 低基率安全事件。
**New task or mechanism:** 主要从正常流学习 conditional null，事件只用于选择 semantic direction。
**Why larger model is insufficient:** 关键是准确建模 null tail，而非平均分类精度。
**Closest threat:** one-class procedural mistake detection 和视频异常检测。 ([arXiv][7])
**Fastest falsification:** 只用正常训练，测跨视频 false alarms/hour 与事件召回。

### L05 — Correction-Value Active Annotation

**Problem failure:** 标注预算被均匀花在普通帧，而最有价值的是会改变未来决策的纠正。
**Who cares and why:** 部署中人工只能偶尔反馈。
**New task or mechanism:** query 人类“这个 alarm 是否错误/哪个义务未完成”，以 expected future risk reduction 选样。
**Why larger model is insufficient:** 这是反馈价值分配问题。
**Closest threat:** boundary-centric active learning 已按 boundary uncertainty 选标注。 ([arXiv][14])
**Fastest falsification:** 离线 replay active-feedback loop，与 entropy/random query 比较。

### L06 — Personalized Stream Adaptation with Safe Reset

**Problem failure:** 同一动作在不同人、工具和环境中表现不同。
**Who cares and why:** wearable 与居家助手。
**New task or mechanism:** 个体 adapter 持续更新；若风险恶化自动回滚。
**Why larger model is insufficient:** global model 不等于个体校准。
**Closest threat:** continual VideoQA 与长程 test-time adaptation。 ([arXiv][15])
**Fastest falsification:** leave-one-person-stream-out，比较 static、TTA、personal adapter。

## D.3 Sequential Decision / Theory Lens

### D01 — Fleet-Level Online FDR

**Problem failure:** 单摄像头误报可接受，不代表 100 个摄像头 × 100 类别可接受。
**Who cares and why:** 监控中心、机器人 fleet、医院。
**New task or mechanism:** 对持续产生的事件 hypotheses 分配 alpha wealth，控制跨流 FDR。
**Why larger model is insufficient:** 多重检验规模效应独立于分类精度。
**Closest threat:** SAFFRON、e-GAI 和 online e-closure。 ([arXiv][16])
**Fastest falsification:** 用多类、多视频 replay 测普通阈值的 fleet false discovery growth。

### D02 — Partial Identification of Latent Endpoint

**Problem failure:** 物理 endpoint 在 RGB 中可能只被限定在一个区间。
**Who cares and why:** 不应因不可观测性惩罚模型。
**New task or mechanism:** 给定传感延迟与观测模型，输出可识别区间而非虚假精确点。
**Why larger model is insufficient:** 这是信息论不可识别，而非参数不足。
**Closest threat:** boundary uncertainty 只建模预测分布，未区分不可识别区间。 ([arXiv][9])
**Fastest falsification:** 比较 RGB-only 与 privileged sensor 的 endpoint residual 下界。

### D03 — Cost-Asymmetric Optimal Commit

**Problem failure:** “越早越好”忽略早报和晚报代价极不对称。
**Who cares and why:** 医疗、机器人、安全告警。
**New task or mechanism:** 将 commit 建模为 optimal stopping，显式使用 false-early、late、miss、interruption cost。
**Why larger model is insufficient:** 同一 posterior 在不同效用下应产生不同决策。
**Closest threat:** StreamReady 使用早晚不对称 metric，但未形成事件风险约束。 ([arXiv][6])
**Fastest falsification:** 对固定 evidence score 改变 cost matrix，检查 mAP 最优策略是否显著非最优。

### D04 — Minimal Causal Witness Certificate

**Problem failure:** 模型告警时不知道哪段证据是必要的。
**Who cares and why:** 人工复核和安全审计。
**New task or mechanism:** 输出最小 prefix witness；移除该证据应使 alarm 不再成立。
**Why larger model is insufficient:** 解释文本不等于反事实必要性。
**Closest threat:** Thinking-QwenVL 的透明 progress/confidence 与 AVP 的 evidence seeking。 ([arXiv][17])
**Fastest falsification:** 删除模型声称的 witness，检查 alarm stability 和 sufficiency。

### D05 — Risk Control under Online Drift

**Problem failure:** 静态 calibration 在光照、人员、设备变化后失效。
**Who cares and why:** 长期部署。
**New task or mechanism:** 允许 predictable recalibration，同时保持指定风险或明确进入 abstain mode。
**Why larger model is insufficient:** calibration 可在准确率不变时失效。
**Closest threat:** WATCH 允许部分 online adaptation；AVRobustBench 显示多模态 TTA 不一定有效。 ([arXiv][5])
**Fastest falsification:** 逐域 replay，比较 static threshold 与 rolling/weighted calibration。

### D06 — Memory Regret against Future Query Adversary

**Problem failure:** 平均 QA 不刻画 memory 对未预见 query 的最坏损失。
**Who cares and why:** 审计型 wearable memory。
**New task or mechanism:** 定义 query family 和 oracle memory，给出 fixed-capacity memory regret。
**Why larger model is insufficient:** 在固定存储下不可避免地存在信息取舍。
**Closest threat:** SelectStream、ProtoKV 已优化 evidence allocation，但没有一般 regret 保证。 ([arXiv][18])
**Fastest falsification:** 构造 query-after-eviction 对抗集。

## D.4 Perception / System Lens

### P01 — Risk-Constrained Camera Duty Cycling

**Problem failure:** 大多数“高效视频模型”先读取像素再决定是否压缩，传感与编码成本已发生。
**Who cares and why:** 电池相机、wearable、边缘设备。
**New task or mechanism:** 在观测前选择 sleep/low-rate/high-rate，并约束 miss risk 和 delay。
**Why larger model is insufficient:** 看不看的决策发生在 heavy model 之前。
**Closest threat:** AdaFrame、SAN、AVP，但它们分别聚焦 clip selection、系统负载或已存视频证据搜索。 ([arXiv][19])
**Fastest falsification:** replay 不同采样策略，在相同 miss/delay 下比较 sensor-read 数。

### P02 — Asynchronous Multimodal Wake-Up

**Problem failure:** RGB 持续开启昂贵，而音频、IMU、事件流可能是廉价 sentinel。
**Who cares and why:** wearable、家庭和工业。
**New task or mechanism:** 低功耗模态触发高成本 RGB，并记录真正 sensor clock。
**Why larger RGB model is insufficient:** 它不能减少 RGB 采集本身。
**Closest threat:** FEEL、audio-visual robustness、event-camera HAR。 ([arXiv][12])
**Fastest falsification:** 在同步多模态数据上训练轻量 trigger，测 wake-up recall 和能耗 proxy。

### P03 — Edge–Cloud Staleness Certificate

**Problem failure:** 云端结果到达时，所依据证据可能已经过期。
**Who cares and why:** 网络不稳定的机器人和视频分析。
**New task or mechanism:** 每个输出携带 evidence age、network age 和 deadline validity。
**Why larger model is insufficient:** stale correctness 仍可能是错误决策。
**Closest threat:** E2E-LOAD 与 SAN 研究效率/负载，但不证明 decision freshness。 ([CVF Open Access][20])
**Fastest falsification:** 注入网络延迟，测 accuracy 相同但 utility 不同的案例。

### P04 — Privacy-Expiring Streaming Memory

**Problem failure:** 长期 memory 保存了未来可能被查询、但当前不应保留的私人信息。
**Who cares and why:** 家庭和 wearable。
**New task or mechanism:** memory item 带 expiry/privacy budget，评价未来 query utility 与隐私暴露。
**Why larger memory is insufficient:** 更大 memory 反而增加风险。
**Closest threat:** query-agnostic KV memory 只优化准确率与容量。 ([arXiv][8])
**Fastest falsification:** 在已有人/物 identity 标注的视频上测 privacy–utility frontier。

### P05 — Event-Camera Sentinel with RGB Escalation

**Problem failure:** RGB 在快速运动、低光和低功耗场景同时受限。
**Who cares and why:** 工业、安全、机器人。
**New task or mechanism:** event stream 持续运行，仅在风险上升时激活 RGB/VLM。
**Why larger RGB model is insufficient:** 传感器物理属性不同。
**Closest threat:** event-based HAR 已提供高分辨率动作数据。 ([arXiv][21])
**Fastest falsification:** event-only trigger + RGB classifier，与 continuous RGB 比较。

### P06 — Deadline-Aware Compute Scheduler

**Problem failure:** 模型报告 FPS，却忽略排队后已经错过决策期限的预测。
**Who cares and why:** 多任务机器人和边缘系统。
**New task or mechanism:** scheduler 可 drop、defer、approximate；迟到输出按无效处理。
**Why larger model is insufficient:** deadline miss 是调度问题。
**Closest threat:** SAN 已按系统状态选择深度/分辨率。 ([CVF Open Access][10])
**Fastest falsification:** 用真实 GPU contention trace 重放，比较 nominal FPS 与 deadline utility。

## D.5 Semantics / World Lens

### S01 — Attempt–Outcome–Abort–Recovery Lifecycle

**Problem failure:** 动作开始并不保证完成，完成也不保证成功。
**Who cares and why:** 程序辅助和质量控制。
**New task or mechanism:** 输出 lifecycle transition graph，而不是单一 span。
**Why larger model is insufficient:** 需要 outcome/state annotations。
**Closest threat:** PREGO、HoloAssist、EgoProactive 已覆盖 mistake、intervention、OOP recovery。 ([arXiv][7])
**Fastest falsification:** 在失败视频中测普通 TAD 对 success 的假阳性率。

### S02 — Object-State Transition Verification `[可能不应做动作检测]`

**Problem failure:** “拧螺丝”标签不能证明螺丝已锁紧。
**Who cares and why:** 工业和机器人关心 world state。
**New task or mechanism:** 输出 `(entity, state_before, transition, state_after, evidence)`。
**Why larger model is insufficient:** 动作外观和结果状态可分离。
**Closest threat:** STORM-PSR 识别正确完成步骤，FEEL 建模 contact。 ([arXiv][22])
**Fastest falsification:** 比较 action label accuracy 与 final-state verification accuracy。

### S03 — Prerequisite Violation Monitor

**Problem failure:** 单个动作都可识别，顺序仍可能非法。
**Who cares and why:** 安装、医疗流程、安全规程。
**New task or mechanism:** 监测 partially ordered obligations 和 irreversible constraints。
**Why larger model is insufficient:** 必须表示程序合法性。
**Closest threat:** ProTAS 的 task graph 和 PREGO 的 next-action mismatch。 ([CVF Open Access][23])
**Fastest falsification:** 在 step permutation/OOP 数据上比较 frame classifier 与 formal monitor。

### S04 — Multi-Agent Causal Responsibility

**Problem failure:** 检测到交互不等于知道谁导致状态变化。
**Who cares and why:** 体育、交通和协作机器人。
**New task or mechanism:** 输出 agent–action–effect causal attribution。
**Why larger model is insufficient:** 共现不等于责任。
**Closest threat:** StreamForest/ODV-Bench 包含多主体风险，但不专门标注责任。 ([NeurIPS Proceedings][24])
**Fastest falsification:** 小规模多视角人工标注，检查现有 VLM 的 counterfactual consistency。

### S05 — Streaming Temporal-Logic Monitor

**Problem failure:** 复杂事件由 before/until/overlap/count 等关系定义。
**Who cares and why:** 用户需求常不是单一动作类别。
**New task or mechanism:** 在线维护可执行 temporal logic formula 的满足/违例状态。
**Why larger model is insufficient:** 自由文本答案缺乏逻辑闭包。
**Closest threat:** TimeLogic 与 CoMET-Bench 已覆盖多事件时序推理/grounding。 ([arXiv][25])
**Fastest falsification:** 将公式 monitor 与 prompt-only VLM 在组合事件上比较。

### S06 — Mid-Stream Event Definition

**Problem failure:** 用户可能在流已开始后才定义新事件。
**Who cares and why:** 临时查询与事件回溯。
**New task or mechanism:** 新语义 spec 到达后，只能使用已保存的 bounded sufficient state。
**Why larger model is insufficient:** 未保存的过去信息无法恢复。
**Closest threat:** AViLA、OZ-TAL、OEM-VQL 已处理临时查询、开放类别或在线 episodic memory。 ([arXiv][26])
**Fastest falsification:** 在 query arrival 前后随机化关键证据时间。

## D.6 Application-First Lens

### A01 — Safe Robot Handover Readiness

**Problem failure:** 检测“伸手”不能决定是否安全交接。
**Who cares and why:** 协作机器人需要 readiness、abort 和 uncertainty。
**New task or mechanism:** 输出 handover-ready / wait / abort，代价由碰撞、掉落和等待定义。
**Why larger model is insufficient:** 目标是决策效用而非动作分类。
**Closest threat:** proactive assistance 和 early action anticipation。 ([arXiv][27])
**Fastest falsification:** 现有 handover clips 上离线估算 false-ready 风险。

### A02 — Physically Verified Industrial Quality Gate

**Problem failure:** 看到操作不代表产品通过质检。
**Who cares and why:** 装配线需要可验证状态。
**New task or mechanism:** video prediction 必须与 torque/contact/final-state sensor 一致。
**Why larger model is insufficient:** 视觉动作和产品状态并非同一变量。
**Closest threat:** FEEL、IndEgo、STORM-PSR。 ([arXiv][12])
**Fastest falsification:** 统计正确动作外观但错误最终状态的比例。

### A03 — Rare-Event Eldercare Escalation

**Problem failure:** 居家系统的主要问题通常是长期误报，而非单段 accuracy。
**Who cares and why:** 高频误报会导致用户关闭系统。
**New task or mechanism:** 控制每日/每周 escalation risk，同时检测跌倒、长时间静止等。
**Why larger model is insufficient:** 低基率下 calibration 主导实际效用。
**Closest threat:** egocentric fall detection，但其评价仍以 clip 为主。 ([arXiv][28])
**Fastest falsification:** 将短跌倒数据嵌入长日常负流，测 false alarms/day。

### A04 — Sports Evidence-Sufficient Review Trigger

**Problem failure:** 自动识别动作不等于值得中断比赛复核。
**Who cares and why:** 复核有时间和观众体验成本。
**New task or mechanism:** 输出 review/no-review 与最小证据，而非裁判最终结论。
**Why larger model is insufficient:** 需要 decision threshold 和 intervention cost。
**Closest threat:** generic evidence seeking 与 skill grading。 ([arXiv][29])
**Fastest falsification:** 用已裁决事件比较 detection confidence 与 review utility。

### A05 — AR Interruption Utility `[可能不应做检测]`

**Problem failure:** 助手即使识别正确，也可能在最糟糕的时刻打断用户。
**Who cares and why:** AR、培训和维修。
**New task or mechanism:** 输出 interrupt/wait/silent，评价认知中断成本和错误延迟。
**Why larger model is insufficient:** detection confidence 不等于 intervention utility。
**Closest threat:** HoloAssist 和 EgoProactive 已有人工干预与 when-to-interrupt。 ([arXiv][30])
**Fastest falsification:** 用人类 instructor intervention timing 训练轻量 utility model。

### A06 — Clinical Workflow Omission Escalation

**Problem failure:** 流程中漏做一步可能比错误识别一步更危险。
**Who cares and why:** 临床安全。
**New task or mechanism:** obligation/deadline monitor + calibrated escalation。
**Why larger model is insufficient:** 需要指南、期限和高成本风险控制。
**Closest threat:** procedural mistake/OOP work，但临床有效性与风险仍未解决。 ([arXiv][7])
**Fastest falsification:** 先用模拟流程；若无法获得真实专家标注则立即降级。

---

# E. Deduplication and Novelty Audit

## E.1 聚类图

```text
C1  长时风险有效性
    T01, D01, D03, D05, A03

C2  物理事件、可观测性与不确定边界
    T02, T05, L02, D02, S02, A02

C3  omission / outcome / failure / intervention
    T03, S01, S03, A01, A05, A06

C4  部署期学习与反馈
    L01, L04, L05, L06

C5  delayed query、memory 与 temporal logic
    T04, D06, P04, S05, S06

C6  主动传感与真实系统时钟
    T06, P01, P02, P03, P05, P06

C7  证据必要性与责任
    D04, S04, A04

C8  状态变化自监督
    L03
```

## E.2 分簇查新

### C1 — 长时风险有效性

**最近工作：**

* CAG-QIL/OAT/ActionSwitch：严格在线、即时 instance、状态与 proposal，但目标仍是 action localization/mAP。 ([CVF Open Access][1])
* WATCH：weighted conformal martingale，对在线 change monitoring 提供 false-alarm control。 ([arXiv][5])
* Online anomaly detection with false-alarm bounds：已把统计误报界引入 surveillance anomaly。 ([arXiv][31])
* SAFFRON/e-GAI/online e-closure：持续产生 hypotheses 时控制在线 FDR。 ([arXiv][16])

**A+B 风险：中。** 最强拒稿说法是“On-TAL score + conformal martingale”。

**剩余 delta：**

1. 已知语义事件而非无监督 distribution shift/anomaly；
2. repeated, overlapping, multi-class event instances；
3. event candidate 的 predictable creation/reset；
4. false-alarm/FDR 与 localization delay 的联合任务；
5. 长 negative streams 和 streaming video benchmark；
6. learned semantic evidence increment，而不是直接把现成 p-value 套公式。

**查新置信度：0.82。**
**盲区：** 信号处理、医学视频监测、工业 process control 中可能存在未被 CV/ML 索引的类似语义告警系统。

### C2 — 物理事件、可观测性与不确定边界

**最近工作：**

* StreamReady：证据窗和早晚惩罚；
* Thinking-QwenVL：first-sufficient-evidence timing；
* AViLA：query/evidence asynchrony；
* FEEL：force-synchronized egocentric video；
* Boundary Uncertainty / EMB：概率边界与 elastic boundaries。 ([arXiv][6])

**A+B 风险：低至中。**

“force data + readiness”不是完整答案，因为现有 readiness 工作把 evidence time 当回答目标，却没有一个独立的 latent physical time；FEEL 也没有评价物理变化、视觉可辨认和模型提交三种延迟。

**剩余 delta：**

* 证明 raw latency 混合了不可约 sensing lag 与 algorithmic lag；
* 提供多传感器物理锚点；
* 用人类 prefix reveal 或部分识别区间定义 visual observability；
* 产生模型排名反转或错误结论的实证。

**查新置信度：0.76。**
**盲区：** 手术 workflow、机器人 contact estimation、工业 quality control 文献。

### C3 — Omission / Outcome / Intervention

**最近工作：**

* PREGO：在线 open-set procedural mistake；
* ProTAS：progress 和 task graph；
* HoloAssist：mistake、intervention type、hand forecasting；
* EgoProactive/Pro²Bench：OOP、恢复、何时/如何打断；
* STORM-PSR：正确完成步骤和 completion delay。 ([arXiv][7])

**A+B 风险：高。** 泛化的“失败/恢复/干预”已不足以构成新题。

**仍可能成立的窄 delta：**

* 将**非事件**设为一等对象；
* 只有在 deadline 或无法恢复时才能宣布 omission；
* 给出“最早合法违例时间”的可识别性；
* 区分 wrong action、missing action、late action、substitution、recovery。

**查新置信度：0.70。**
**盲区：** runtime verification、business process monitoring、医疗流程合规研究。

### C4 — 部署期学习与反馈

**最近工作：**

* on-device streaming W-TAL；
* DAM continual VideoQA；
* long-term online test-time adaptation；
* delayed-feedback VLA adaptation。 ([arXiv][11])

**A+B 风险：中高。**

**剩余 delta：**

* event localization/alarm 而非 clip classification 或 QA；
* label 到达时刻显式建模；
* 过去 emission 永久不可修改；
* prequential risk、adaptation lag、future-only benefit；
* 错误反馈与安全 rollback。

**查新置信度：0.63。**
**盲区：** 在线学习、continual detection 和工业 human-in-the-loop 系统。

### C5 — delayed query / memory / temporal logic

**最近工作过密：** AViLA、StreamMem、Vista、ProtoKV、SelectStream、OEM-VQL、TimeLogic、CoMET-Bench 已分别覆盖异步 query、query-agnostic memory、scene memory、delayed query、latent evidence allocation、在线 episodic retrieval 和组合时序推理。 ([arXiv][26])

**A+B 风险：极高。**

只有严格的 worst-case coverage/regret 或 privacy contract 可能留下理论空间；作为 CV 模型主线不推荐。

**查新置信度：0.91。**

### C6 — 主动传感与真实系统

**最近工作：**

* AdaFrame：按 input utility 选择帧；
* SAN：按系统负载选择分辨率/深度；
* Active Video Perception：query-conditioned evidence seeking；
* SelectStream：固定 memory/compute 下分配历史 evidence；
* E2E-LOAD：端到端实时 OAD；
* event-camera HAR：不同物理传感器。 ([arXiv][19])

**A+B 风险：中。**

**剩余 delta：**

* 决策发生在像素采集之前；
* sensor energy、wake-up latency 和 blind intervals 被真实计账；
* 在 miss-risk/delay constraint 下优化，而非 accuracy–FLOPs 平均权衡；
* 实际硬件或至少 hardware-calibrated replay。

**查新置信度：0.72。**

### C7 — 证据必要性与责任

已有 work 会展示 evidence、progress 或 reasoning，但**反事实必要证据**与**因果责任**仍未充分解决。问题是标注和因果识别成本高，容易退化为 explanation benchmark。 ([arXiv][17])

**查新置信度：0.58。**

### C8 — 自监督状态变化

FEEL、boundary-sensitive pretraining 以及大量视频 SSL 都可解释为近邻。若没有新的可识别性原则或跨任务迁移结果，容易成为“另一种 pretext”。 ([arXiv][12])

**查新置信度：0.55。**

## E.3 检索盲区声明

本次检索覆盖了 CVF Open Access、ECVA、NeurIPS proceedings、ACL Anthology、OpenReview 可访问页面、arXiv 和公开 GitHub 项目。以下不在完整覆盖范围内：

* IEEE/ACM 付费全文中的工业监测和传感器论文；
* 医疗 workflow compliance；
* process mining/runtime verification；
* 专利与内部工业系统；
* 2026-07-11 之后的新预印本；
* 尚未公开的在审论文。

因此以下结论均是**在本次检索范围内的 novelty confidence**，不是绝对“首个”。

---

# F. Kill Matrix

| Idea | Strongest rejection                    | Closest prior-work combination      | What evidence could rescue it              | Verdict     |
| ---- | -------------------------------------- | ----------------------------------- | ------------------------------------------ | ----------- |
| T01  | 只是给 On-TAL 套 conformal wrapper         | On-TAL + WATCH + anomaly FAR        | 长时风险失控、有效保证、低 delay 三者同时成立                 | **SURVIVE** |
| T02  | `τ_vis` 主观且数据窄                         | FEEL + StreamReady                  | 稳定人类共识、显著 observability gap、模型排名反转         | **SURVIVE** |
| T03  | PREGO/EgoProactive 已做 mistake/recovery | task graph + mistake detection      | omission 的最早合法判定不能被现有方法表达                  | **SURVIVE** |
| T04  | memory 红海，仅换 worst-case metric         | StreamMem + ProtoKV                 | 给出非平凡 regret lower/upper bound             | KILL        |
| T05  | boundary uncertainty + conformal 的直接组合 | Gaussian boundary + EMB             | 多标注者 coverage 改变结论且在线 set 有必要              | HOLD        |
| T06  | 只有 benchmark novelty                   | SAN + stress testing                | 多模型排名在真实 perturbation 下稳定反转                | HOLD        |
| L01  | generic continual learning 应用到视频       | on-device W-TAL + DAM               | 真实延迟反馈、future-only gain 和 immutable audit  | **SURVIVE** |
| L02  | privileged distillation 不是创新           | FEEL + teacher/student              | 证明物理 latent state 而非普通 feature distill     | HOLD        |
| L03  | 又一个视频 SSL pretext                      | BSP + force prediction              | 跨 taxonomy 的 state-transition transfer 显著  | HOLD        |
| L04  | one-class anomaly detection 换名         | PREGO + anomaly detection           | semantic null 与 anomaly null 有决定性差异        | KILL        |
| L05  | active learning 已成熟                    | B-ACT + feedback selection          | risk reduction query 明显优于 uncertainty      | HOLD        |
| L06  | personalization 包装                     | TTA + continual adapters            | 真实用户长期数据和 safe rollback                    | KILL        |
| D01  | T01 的 fleet 扩展                         | T01 + online FDR                    | 多摄像头依赖下仍有效且更有 power                        | HOLD        |
| D02  | 只是区间回归理论                               | T02 + partial identification        | RGB 不可识别性的可证明/可测下界                         | HOLD        |
| D03  | generic optimal stopping               | readiness + cost-sensitive decision | 明确效用造成方法和排名改变                              | HOLD        |
| D04  | explanation benchmark                  | Thinking-QwenVL + AVP               | witness removal 有因果稳定性与用户价值                | HOLD        |
| D05  | 非平稳条件下难以保证                             | WATCH + TTA                         | 明确假设下风险有效且不 alpha-death                    | HOLD        |
| D06  | 理论脱离视频                                 | SelectStream + regret               | 有真实 adversarial query family 和 tight bound | KILL        |
| P01  | AdaFrame/SAN 的风险版本                     | AdaFrame + SAN + e-process          | 观测前调度、真实能耗、受控 miss                         | **SURVIVE** |
| P02  | 普通 multimodal trigger                  | audio/IMU/event fusion              | 真实 sensor power 和 wake-up trade-off        | HOLD        |
| P03  | edge-cloud engineering                 | E2E-LOAD + scheduler                | freshness certificate 改变安全结果               | KILL        |
| P04  | 应用型 privacy 包装                         | memory compression + privacy        | formal leakage/utility guarantee           | KILL        |
| P05  | event-camera 应用论文                      | event HAR + RGB fusion              | sentinel escalation 比 continuous RGB 实质更优  | HOLD        |
| P06  | SAN + deadline scheduler               | SAN + real-time systems             | dropped-decision accounting 导致新结论          | HOLD        |
| S01  | EgoProactive 已覆盖 lifecycle/recovery    | PREGO + HoloAssist + EgoProactive   | outcome taxonomy 有独立可证伪核心                  | HOLD        |
| S02  | state recognition 已有                   | STORM-PSR + FEEL                    | action accuracy 与 state verification 显著解耦  | HOLD        |
| S03  | ProTAS/PREGO 直接组合                      | task graph + mistake detection      | formal omission/irreversibility 能捕获独特错误    | HOLD        |
| S04  | 因果 responsibility 无可识别数据               | multi-agent VLM + causal QA         | 多视角干预或 counterfactual labels               | KILL        |
| S05  | temporal logic QA 已覆盖                  | TimeLogic + CoMET                   | 在线 monitor 有严格 correctness/latency         | KILL        |
| S06  | AViLA/OZ-TAL 已占据                       | async query + open vocab            | bounded-state sufficiency theorem          | KILL        |
| A01  | 机器人应用包装                                | anticipation + readiness            | false-ready 风险成为独立 benchmark               | HOLD        |
| A02  | 工业场景包装                                 | FEEL + STORM-PSR                    | 真实生产状态传感和失败实例                              | HOLD        |
| A03  | 数据和伦理门槛过高                              | fall detection + alarm calibration  | 长期真实负流与临床/护理效用                             | HOLD        |
| A04  | sports application only                | evidence seeking + review           | 真实裁判复核成本和受控决策                              | KILL        |
| A05  | EgoProactive/HoloAssist 已很近            | intervention timing + VLM           | 人因实验显示 detection metric 排名错误               | HOLD        |
| A06  | 临床数据不可得                                | omission + workflow rules           | 专家合作与真实错漏记录                                | HOLD        |

最终仅五项进入 `SURVIVE`：**T01、T02、T03、P01、L01**。

---

# G. Ranked Top 5

## 总评分

| Rank | Idea                                          | 重要性 /20 | 新颖性 /25 | 必要性 /15 | 清晰度 /10 | 可证伪 /10 | 算力可行 /10 | 数据 /5 | 叙事 /5 |               扣分 |     总分 |
| ---: | --------------------------------------------- | ------: | ------: | ------: | ------: | ------: | -------: | ----: | ----: | ---------------: | -----: |
|    1 | T01 Anytime-Valid Semantic Event Alarms       |      19 |      22 |      14 |       9 |      10 |        8 |     4 |     3 |                0 | **89** |
|    2 | T02 Three-Clock Event Observability           |      19 |      22 |      14 |       9 |       9 |        6 |     3 |     4 |                0 | **86** |
|    3 | T03 Deadline-Censored Omission Monitoring     |      19 |      23 |      14 |      10 |      10 |        9 |     4 |     5 |        −10 高重叠风险 | **84** |
|    4 | P01 Risk-Constrained Camera Duty Cycling      |      19 |      22 |      14 |       9 |      10 |        9 |     4 |     5 |         −10 近邻重叠 | **82** |
|    5 | L01 Delayed-Feedback Continual Event Learning |      18 |      22 |      14 |       9 |       9 |        9 |     4 |     5 | −10 continual 重叠 | **80** |

---

## Rank 1 — Anytime-Valid Semantic Event Alarms

### 1. 一句话论文命题

> 将流式视频事件理解从 clip-level localization 重定义为无限时域的 sequential hypothesis testing，并在给定语义事件上控制 anytime false-alarm 或 online FDR，同时最小化检测延迟。

### 2. 真实失败

一个 detector 在每个 30 秒 clip 上只有 1% 的误报概率，并不意味着可连续运行。若把近似独立的 30 秒机会重复 1,000 次，至少一次误报的概率会接近 1。当前 mAP、tIoU 和有限 latency budget 对这一故障基本失明。

这会直接导致：

* 安全告警系统被高频误报淹没；
* threshold 为了控制长期误报而被迫极端保守；
* 不同视频时长、类别数或摄像头数的结果不可比较；
* “更低延迟”可能只是更早触发更多 false alarms。

### 3. 精确任务定义

输入：

```text
causal video stream X1, X2, ...
semantic event specification q
optional class/query set Q
risk budget α
```

输出：

```text
immutable alarms A1, A2, ...
each alarm = {event/query, commit time, score/e-value, optional estimated past span}
```

约束：

* 只读当前和过去；
* alarm 后不可删除或修改；
* candidate creation、betting function 和 alpha allocation 必须 predictable；
* repeated instances 需有明确 reset/refractory/risk-set 规则。

主评价：

1. `P0(ever alarm) ≤ α` 或 empirical anytime-FWER；
2. 多事件、多流条件下 online FDR；
3. event recall；
4. detection delay；
5. risk–delay curve；
6. optional localization tIoU，降为次指标。

### 4. 相比现有任务改变了什么

* 从“每个视频结束后收集 proposals”改为“任意停止时刻仍须可靠”；
* 从 finite clip metric 改为 horizon-independent risk；
* 从每个 proposal 独立计分改为承认 repeated testing；
* span 可作为 alarm 的附属回溯，不再是唯一输出。

### 5. 单一主贡献与辅助贡献

* **主贡献**：semantic video event 的 anytime-valid alarm formulation 与风险控制机制。
* **唯一辅助贡献**：长时负流和 repeated-event evaluation protocol。

不应再叠加新 backbone、LLM 或复杂 memory。

### 6. 最小必要方法

令 causal model 在每个时间产生 event evidence score `s_t(q)`。在校准 null 上将其变为有效的条件 p-value/e-value `e_t`。构造非负 supermartingale：

[
E_0=1,\qquad E_t=E_{t-1},b_t(e_t),
]

其中 `b_t` 只能依赖过去。报警规则：

[
\tau=\inf{t:E_t\ge 1/\alpha}.
]

在 e-process 条件成立时，可用 Ville 型界控制 `P_0(τ<∞)`。重复实例可通过：

* predictable candidate spawning；
* event-specific e-process；
* post-alarm reset；
* e-LOND/e-SAFFRON 类 alpha allocation；

处理多类别、多流 discovery。

关键不是公式本身，而是如何从**时序依赖、非平稳、重复实例的视频证据**构造有效 increment。

### 7. 为什么不是模块拼接

若去掉风险有效性，论文主 claim 就消失；换任何 backbone 都不能替代该问题。方法的核心对象是“在任意停止时刻仍有效的告警过程”，不是 memory、head 或 loss 组合。

### 8. 最近竞争工作及逐项 delta

* CAG-QIL：定义 On-TAL 和不可回改，但不控制长期 repeated-testing risk。 ([CVF Open Access][1])
* OAT：提出 early detection metric，但仍是有限视频 mAP/proposal 评价。 ([ECVA][32])
* ActionSwitch：解决同时/同类实例与状态切换，但仍无 anytime error guarantee。 ([ECVA][33])
* WATCH：提供 change-monitoring false-alarm 理论，但目标是 distribution shift，不是多实例 semantic video event。 ([arXiv][5])
* Online video anomaly detection with FAR bound：证明视频领域存在风险界需求，但处理 anomaly，且主要是渐近 FAR，不是语义多类别、repeated instance 与 online FDR。 ([arXiv][31])

### 9. Claim map

| Claim                         | 必须提供的证据                                      |
| ----------------------------- | -------------------------------------------- |
| clip-level threshold 在长流中风险失控 | 不同时长负流上的 empirical ever-false-alarm 曲线       |
| 提议方法满足指定风险                    | 多 seed、block bootstrap、不同 horizon 的 coverage |
| 不是靠极端保守阈值                     | 在相同风险下的 recall/delay 优势                      |
| 支持 repeated instances         | 合成与真实重复事件流                                   |
| 支持多类/多流                       | online FDR 与 per-class power                 |
| 与 backbone 无关                 | 至少两种 causal score source                     |

### 10. 三块核心实验

1. **Long-null stress test**
   5 分钟、30 分钟、2 小时、8 小时负流；比较固定阈值、Bonferroni、CUSUM、conformal/e-process。

2. **Risk–delay event detection**
   在 THUMOS/MUSES/ActivityNet 或程序视频上比较相同风险下的 delay/recall。

3. **Multi-class/multi-stream FDR**
   将 20 类和多视频组成持续 hypothesis stream，测 FDR、power 和 alpha-death。

### 11. 最危险 baseline

* 直接把普通 threshold 调到与方法相同 empirical false-alarm rate；
* Bonferroni/alpha-spending；
* CUSUM/SPRT；
* WATCH-style conformal martingale；
* 视频 anomaly FAR 方法；
* endpoint-only PCEH score + conservative threshold。

必须在**完全相同 causal score、相同校准数据、相同 horizon**下比较。

### 12. Kill criteria

立即终止主线，若出现任一项：

1. block-preserving null streams 上 empirical risk 超过 `2α`；
2. 为满足风险，recall 相对 matched-risk fixed threshold 下降超过 20%；
3. median delay 增加超过一个完整动作时长或超过 100%；
4. 简单 Bonferroni/CUSUM 在所有设置中同样有效且延迟更低；
5. 只能在人工拼接 iid clips 上有效，真实连续负流失效；
6. 无法为 data-dependent reset 给出合法或至少可审计的处理。

### 13. 成本

* 48h pilot：`< 10 GPU-hours`，多数可 CPU 完成；
* 完整 frozen-feature 论文：约 `20–80 GPU-hours`；
* 若训练新 score model：约 `100–300 GPU-hours`，但非必要；
* 标注：第一次实验不需要新增人工标注。

### 14. Venue

* **NeurIPS / ICLR**：若有清楚的条件有效性、multiple testing 和理论；
* **CVPR / ICCV**：若核心是视频任务、长流 benchmark 和强实证。

### 15. 分数与风险

* **89/100**
* 置信度：**0.82**
* 最大查新风险：统计监测或工业过程领域可能已有“semantic event + sequential risk”但未以视频术语发表。
* 最大技术风险：真实视频 null 不满足简单交换性，导致 distribution-free claim 不能成立。

---

## Rank 2 — Three-Clock Event Observability

### 1. 一句话论文命题

> 流式视频事件不应只标一个 endpoint：必须分别测量物理状态变化时间、视觉首次可辨认时间和系统不可撤销提交时间，才能区分不可约感知延迟与算法延迟。

### 2. 真实失败

对“放下物体”“锁紧部件”“球越过线”“工具脱离接触”等事件：

* 物理变化可能发生在 `τ_phys`；
* 视觉结果直到遮挡解除或物体稳定后才可确认 `τ_vis`；
* 模型在 `τ_commit` 才作出不可回改结论。

当前 latency 往往直接计算 `τ_commit − GT_end`，把传感器无法观察的延迟也归咎于算法。

### 3. 精确任务定义

每个 event instance：

```text
τ_phys: privileged sensor or physically verifiable transition
C_vis: visual-observability interval or earliest stable human-observable prefix
τ_commit: model's immutable decision
```

输出：

* latent physical event posterior或置信区间；
* current observability state；
* commit/abstain。

评价：

[
\Delta_{\text{sense}}=\tau_{\text{vis}}-\tau_{\text{phys}},
\quad
\Delta_{\text{alg}}=\tau_{\text{commit}}-\tau_{\text{vis}}.
]

还应评价：

* physical endpoint error；
* false early commit；
* set coverage；
* algorithmic lag；
* raw latency 造成的模型排名变化。

### 4. 任务改变

从“预测人工时间边界”改为“估计一个受传感器观测过程约束的 latent transition”。

### 5. 主贡献

* **主贡献**：三时钟任务和 observability decomposition。
* **辅助贡献**：一个最小 latent-state estimator。

### 6. 最小方法

使用 state-space/event-time model：

* latent state `z_t∈{before, transitioned}`；
* privileged sensor 在训练中给 `τ_phys`；
* video observation likelihood 允许 transition 后延迟出现视觉 evidence；
* 输出 posterior `P(τ_phys | x_≤t)`；
* 当 posterior risk 达到 cost/risk criterion 时 commit。

若 `τ_vis` 无法稳定单点标注，应改为：

```text
τ_vis ∈ [lower, upper]
```

并使用 interval-censored likelihood，而不是制造虚假精确标签。

### 7. 非模块拼接原因

论文核心是**可观测性分解与新的时间变量**。即使只用线性 probe，只要证明现有 latency 评价混淆了 sensing 与 algorithm，贡献仍成立。

### 8. 最近工作与 delta

* StreamReady：给 evidence window 和早晚惩罚，但没有独立物理事件时间。 ([arXiv][6])
* Thinking-QwenVL：预测 first-sufficient-evidence，但 `t*` 仍是回答证据时刻，不是 latent physical transition。 ([arXiv][17])
* AViLA：处理 query/evidence asynchrony，不处理物理状态与视觉可见性的分离。 ([arXiv][26])
* FEEL：提供 force-synchronized video 和 contact tasks，但未建立 `phys–vis–commit` latency decomposition。 ([arXiv][12])
* Boundary uncertainty/EMB：建模标注或回归不确定性，但没有外部物理锚点。 ([arXiv][9])

### 9. Claim map

| Claim                | 证据                                                |
| -------------------- | ------------------------------------------------- |
| 三种时间确实不同             | force/contact + video + prefix-human timestamps   |
| sensing lag 不是模型可消除的 | 多模型共享相似 `τ_vis−τ_phys` 下界                         |
| 现有 latency 排名有偏      | raw latency 与 algorithmic latency 的 rank reversal |
| latent model 更合理     | physical endpoint error、coverage、false-early      |
| 不只适用于单类              | 至少 3 类不同 observability pattern                    |

### 10. 三块实验

1. **Observability study**：多标注者 prefix reveal + privileged sensor。
2. **Metric audit**：比较 raw latency、sensing lag、algorithmic lag。
3. **Minimal model**：固定特征 + latent event-time head，对比普通 endpoint regression/readiness。

### 11. 最危险 baseline

* human visual oracle；
* privileged sensor oracle；
* endpoint-only hazard；
* StreamReady-style readiness；
* Gaussian/distributional boundary regression；
* current PCEH corrected target。

### 12. Kill criteria

* `τ_phys` 与 `τ_vis` 的 median gap 小于一个采样单位；
* 人类对 `τ_vis` 的一致性很低且无法用区间稳定表示；
* raw latency 与 algorithmic latency 不改变任何模型结论；
* privileged sensor 事件本身不可靠；
* 只能在单一人为动作上成立。

### 13. 成本

* pilot：100–300 clips，约 10–30 人工小时，`<10 GPU-hours`；
* 完整数据论文：可能需要 50–200 小时同步采集；
* GPU：冻结特征时约 20–80 GPU-hours；
* 主要风险是数据，不是算力。

### 14. Venue

**CVPR / ICCV**；若可观测性理论充分，也可考虑 NeurIPS。

### 15. 分数

* **86/100**
* 置信度：**0.73**
* 剩余风险：first visually observable time 可能强依赖观察者、任务说明和容错范围。

---

## Rank 3 — Deadline-Censored Omission Monitoring

### 1. 一句话论文命题

> 与其检测“发生了什么”，流式系统应监测带期限的事件义务，并在预期事件未发生或已无法按时完成时，输出最早合法的 omission violation。

### 2. 真实失败

以下都不是普通 action detection：

* 手术前未消毒；
* 拧紧之后没有执行 torque confirmation；
* 到站后未服药；
* 装配步骤被跳过；
* 安全门打开后未在期限内关闭；
* 动作启动但在完成前中止。

没有 deadline 时，“未发生”在无限流中不可判定；有 deadline 后才成为可证伪事件。

### 3. 精确任务定义

输入：

```text
video stream
obligation contract:
  trigger/precondition P
  required event Q
  deadline d
  allowed substitutions R
  recovery rules G
```

典型逻辑：

[
P \rightarrow \mathbf{F}_{[0,d]} Q.
]

输出：

```text
satisfied / pending / violated / recovered
earliest legal violation time
evidence and confidence
```

不能在 deadline 之前仅因尚未观察到 Q 就宣告 violation，除非当前状态已使 Q 不可恢复地不可能完成。

### 4. 任务改变

从 positive event localization 转向 **absence、deadline 和 obligation state**。span 仅用于 predicate evidence。

### 5. 主贡献

* **主贡献**：deadline-censored omission task。
* **辅助贡献**：probabilistic timed monitor。

### 6. 最小方法

1. 用现有 causal video model 输出 predicate probabilities；
2. timed automaton 维护 obligation state；
3. survival model 估计剩余期限内完成概率；
4. 只有在：

   * deadline 到达，或
   * irreversible state 使完成不可能，

   才允许 violation；
5. recovery event 可撤销 pending violation，但不能回改已经合法提交的历史告警，除非任务明确允许 correction ledger。

### 7. 非模块拼接原因

核心是“非事件何时可被确认”的识别问题。普通 action detector、task graph 和更强 VLM 都不能绕过 deadline/censoring 逻辑。

### 8. 最近工作与 delta

* PREGO：把当前动作与预期下一步比较，未把缺失事件和期限定义为独立对象。 ([arXiv][7])
* ProTAS：progress/task graph 改善 segmentation，但主要输出当前步骤。 ([CVF Open Access][23])
* HoloAssist：有 mistake/intervention annotations，适合作为数据源，但任务不是 formal omission monitoring。 ([arXiv][30])
* EgoProactive：已覆盖 OOP、恢复和何时打断，是最危险竞争者；delta 必须是“期限约束下的非事件与最早合法违例”。 ([arXiv][27])
* STORM-PSR：识别正确完成步骤和 delay，但不是缺失义务。 ([arXiv][22])

### 9. Claim map

| Claim                            | 证据                                  |
| -------------------------------- | ----------------------------------- |
| action detection 无法表示 omission   | 自然 omission case 上 detector failure |
| deadline 是可识别性的必要条件              | 无 deadline 情况的理论/反例                 |
| timed monitor 减少 premature alarm | false-early violation               |
| 能更早发现真实 omission                 | violation delay                     |
| recovery-aware                   | 中断、替代、恢复子集                          |

### 10. 三块实验

1. **Natural omission benchmark**：HoloAssist/EgoOops/EgoProactive/IndEgo。
2. **Controlled omission construction**：删除、延迟、替代、打乱步骤，但必须过滤剪辑伪影。
3. **Monitor comparison**：frame classifier、sequence model、PREGO-like、task graph、timed monitor。

### 11. 最危险 baseline

EgoProactive-style planner/intervention model，以及 oracle procedure plan + 强 VLM。公平比较必须给所有方法相同 procedure contract 和相同 causal observations。

### 12. Kill criteria

* 自然数据中 omission 数量过少；
* synthetic deletion 可被剪辑 discontinuity 轻易识别；
* PREGO/EgoProactive baseline 已直接达到同样能力；
* deadline 只能人为随意设置；
* timed logic 在 predicate error 下没有任何鲁棒性优势。

### 13. 成本

* pilot：5–20 GPU-hours；
* 标注：约 20–50 小时即可做首轮；
* 完整论文：约 50–150 GPU-hours；
* 不需要训练大型 VideoLLM。

### 14. Venue

**CVPR / ICCV**；若 formal runtime-monitoring 部分强，也可考虑 NeurIPS。

### 15. 分数

* **84/100**
* 置信度：**0.68**
* 最大风险：EgoProactive 已非常接近，必须保持 omission/deadline 单核心，不能退化为 generic procedure assistant。

---

## Rank 4 — Risk-Constrained Camera Duty Cycling

### 1. 一句话论文命题

> 在流式视频中，模型应在读取像素之前决定何时以及用何种模态观察，并以明确的 missed-event risk 和 delay constraint 换取真实传感、编码和通信能耗下降。

### 2. 真实失败

多数 frame selection/token pruning/memory compression 在像素已经采集、解码甚至编码后才节省计算。对于电池设备：

```text
camera sensor + ISP + decode + transfer
```

往往已产生显著成本。只报告 backbone FLOPs 不是 active sensing。

### 3. 精确任务

动作：

```text
sleep
low-rate RGB
high-rate RGB
cheap sentinel only
heavy encoder
```

目标：

[
\min_\pi \mathbb{E}[\text{energy}]
]

subject to：

[
P_\pi(\text{miss event})\le\beta,\qquad
\mathbb{E}*\pi[\text{delay}]\le\delta,
\qquad
P*\pi(\text{false alarm})\le\alpha.
]

所有策略只能使用已观察数据、设备状态和低功耗 sentinel。

### 4. 改变

从“输入已存在后的动态计算”改为“输入是否被采集的 sequential decision”。

### 5. 主贡献

* **主贡献**：risk-constrained observation policy。
* **辅助贡献**：hardware-calibrated streaming evaluation。

### 6. 最小方法

* low-cost sentinel state；
* event risk upper confidence bound；
* safe action set：只有不会使 miss-risk 上界超过 β 的 sleep/skip 才可选；
* 在 safe set 内最小化能耗；
* uncertainty/risk 上升时升级采样率；
* 全部决策写入 read ledger。

### 7. 非拼接原因

若只替换成 RL selector 或 motion threshold 而不提供 risk constraint 和真实 pre-observation cost，论文即失败。核心不是 selector architecture。

### 8. 最近工作与 delta

* AdaFrame：自适应选择 clip frames，但重点是视频识别效率，不是无限流预观测风险。 ([arXiv][19])
* SAN：响应硬件负载选深度/分辨率，但依然处理当前输入，并以 accuracy/delay 为主。 ([CVF Open Access][10])
* Active Video Perception：在已有长视频中 query-conditioned 搜索证据，不是现实传感器唤醒。 ([arXiv][29])
* SelectStream：选择写入/保留/检索的 latent evidence，发生在观察之后。 ([arXiv][18])
* Event-camera HAR：说明低功耗异步传感可以成为独立输入。 ([arXiv][21])

### 9. Claim map

| Claim                      | 证据                        |
| -------------------------- | ------------------------- |
| post-capture FLOPs 不等于系统节能 | 传感/ISP/decode/encode 分项测量 |
| policy 满足 miss/delay risk  | 多 seed、事件稀有度和 shift       |
| 节能不是靠少报                    | matched recall/risk       |
| 不是数据集 replay 幻觉            | 至少一个 hardware-in-loop     |
| 可泛化                        | 两种事件分布或设备                 |

### 10. 三块实验

1. replay risk–energy frontier；
2. hardware-in-loop power/wake-up；
3. shift/stress：负载、低光、事件频率变化。

### 11. 最危险 baseline

* uniform low/high fps；
* motion threshold；
* periodic wake-up；
* AdaFrame；
* SAN；
* current DUCA/CADF 类 selector；
* oracle event-aware scheduler。

### 12. Kill criteria

* sensor/ISP 成本相对模型计算可忽略；
* replay 节能在真实设备上消失；
* risk guarantee 依赖 oracle motion；
* 相同 recall/delay 下不优于 periodic schedule；
* 只在长动作有效，短事件系统性漏掉。

### 13. 成本

* replay pilot：10–30 GPU-hours；
* full：50–150 GPU-hours；
* 需 1–2 台可测功耗设备；
* 人工标注通常可复用现有 event labels。

### 14. Venue

**CVPR / ICCV / NeurIPS Datasets & Benchmarks 或 systems-oriented track**。

### 15. 分数

* **82/100**
* 置信度：**0.66**
* 最大风险：真实硬件成本难以准确复现，且风险上界可能依赖强事件持续时间假设。

---

## Rank 5 — Delayed-Feedback Continual Event Learning

### 1. 一句话论文命题

> 定义真正的在线视频学习：模型在产生不可撤销事件输出后，接收延迟且不完整的纠正，只允许改善未来预测，并以 prequential risk、adaptation lag 和 forgetting 评价。

### 2. 真实失败

当前大量“online”工作只是：

```text
offline train → frozen model → online inference
```

真实系统会收到：

* 用户说“刚才不是跌倒”；
* 质检结果表明步骤失败；
* 机器人任务最终成功/失败；
* 人工复核确认 alarm；
* 某段时间后才得到 ground truth。

忽略反馈意味着模型不会适应个体和环境。

### 3. 精确任务

每个时间：

```text
observe packet
emit immutable prediction/alarm
possibly receive feedback about an earlier event
update bounded model state
predict future only
```

反馈字段：

```text
target event ID
feedback arrival time
correct/incorrect label
optional corrected class/time/outcome
confidence/noise
```

评价：

* prequential utility；
* adaptation lag；
* cumulative regret；
* false-alarm after update；
* forgetting；
* update cost；
* privacy/storage。

### 4. 任务改变

从 online inference 改为 **online learning with delayed supervision and immutable history**。

### 5. 主贡献

* **主贡献**：feedback-timed continual event protocol。
* **辅助贡献**：风险门控的小型 update/rollback 方法。

### 6. 最小方法

* frozen backbone；
* small event/query adapter；
* compressed sufficient-stat replay，而非 raw video；
* 更新前估计 expected future benefit；
* update 后在 recent calibration buffer 上做 safety test；
* 若风险恶化则 rollback；
* ledger 同时记录 prediction version 和 model version。

### 7. 非拼接原因

任务的核心变量是**反馈到达时间和 future-only effect**。普通 continual learning 即使使用相同 adapter，也没有该因果协议。

### 8. 最近工作与 delta

* on-device W-TAL：从长视频流学习，但以弱监督分段为主，没有 delayed correction 与 immutable prior emissions。 ([arXiv][11])
* DAM：continual VideoQA across datasets，不是持续事件输出。 ([arXiv][15])
* online TTA for pose：说明长期自适应会发生误差累积，但没有事件反馈协议。 ([arXiv][34])
* delayed-feedback VLA adaptation：有延迟结果反馈，但对象是 action policy，不是 semantic video event localization/alarm。 ([arXiv][35])

### 9. Claim map

| Claim                      | 证据                         |
| -------------------------- | -------------------------- |
| static model 随流漂移          | 时间排序 prequential curve     |
| delayed feedback 可改善未来     | update 后未来窗口，而非重算过去        |
| 不以 forgetting 换 adaptation | old/new domain 双指标         |
| rollback 有效                | noisy/adversarial feedback |
| raw replay 非必要             | memory/privacy ablation    |

### 10. 三块实验

1. domain/person/device drift；
2. feedback delay/noise sweep；
3. adaptation–forgetting–risk 三方权衡。

### 11. 最危险 baseline

* frozen model；
* periodic offline fine-tune；
* replay；
* EWC/LwF；
* test-time entropy minimization；
* DAM-style adapters；
* oracle immediate labels。

### 12. Kill criteria

* 没有可信的现实反馈来源；
* simulated feedback 的结论对 delay 分布极度敏感；
* 简单 periodic fine-tune 同样有效；
* adaptation 改善不足 2–3 个绝对点却显著增加风险；
* safe rollback 频繁拒绝更新，路线退化为 frozen model。

### 13. 成本

* pilot：10–30 GPU-hours；
* full：100–300 GPU-hours；
* 可使用冻结特征；
* 主要成本是构造可信的 feedback stream，而非训练。

### 14. Venue

**NeurIPS / ICLR** 偏 online learning；若以视频 benchmark 为主则 **CVPR/ICCV**。

### 15. 分数

* **80/100**
* 置信度：**0.58**
* 它刚好达到推荐阈值，是五项中最容易被查新或反馈真实性问题击穿的一项。

---

# H. One Recommended Route, One High-Risk Route, One No-Go

## 唯一推荐路线

### **Anytime-Valid Semantic Event Alarms**

原因不是它最“时髦”，而是它同时满足：

* 真实故障明确：误报随运行时间和系统规模累积；
* 不依赖训练大型模型；
* 可在 48 小时内直接证伪；
* 当前仓库的 causal score、read trace 和 immutable ledger 可复用；
* 与 generic memory/readiness 差异清楚；
* 可以产生理论、任务和系统三层贡献；
* 失败条件明确，不会靠改 story 生存。

建议工作名：

```text
AnytimeVideo:
Risk-Valid Semantic Event Alarms in Continuous Video Streams
```

或：

```text
AVERT:
Anytime-Valid Event Risk Testing for Streaming Video
```

## 高风险高回报路线

### **Three-Clock Event Observability**

若能获得 force/contact/IMU 或其他物理状态信号，这条路线可能比推荐路线更具 CVPR 影响力，因为它会直接指出：

> 现有 online localization latency 在科学上混淆了不可约 sensing delay 与 algorithmic delay。

但其数据和定义风险明显更高。

## 明确 NO-GO

### **继续把当前 PCEH 修成“endpoint + completion + emission + latency”的 THUMOS14 主方法**

它的问题不是只有几个 bug：

1. 当前 target/decoder 没有真正分离 endpoint 与 emission；
2. class-level risk 无法处理实例；
3. generic hazard/memory/head 已缺乏新颖性；
4. THUMOS replay 不代表无限流风险、物理 observability 或真实 intervention；
5. 每 seed 约数百 GPU-hour 的训练计划与科学信息增益不匹配；
6. 即使超过 endpoint-only baseline，也最多证明一个特定 head 在旧任务上有效。

正确定位是：**修到足够可靠，作为 baseline 和 causal score generator；不再作为论文主线。**

---

# I. 48-Hour Falsification Plan

目标仅针对唯一推荐路线：**Anytime-Valid Semantic Event Alarms**。

## I.1 需要的数据

### 必需

1. THUMOS14 validation 的 GT 和背景区间；
2. 至少一种 causal score：

   * 当前 PCEH/endpoint-only pilot logits，若已有；
   * 或现有 causal ActionFormer/MATR-like score；
   * 或冻结 SigLIP/VideoMAE feature 上的小型 logistic probe；
3. 按视频保持连续性的 timestamp 和 class labels。

### 优先补充

* ActivityNet 背景片段；
* MUSES 或程序视频；
* 一个持续 30 分钟以上的真实无事件视频。

不需要重新训练大型 backbone。

## I.2 数据构造

构造两类 stream：

### Null streams

* class-specific：目标类完全不存在；
* background-only：只保留 GT-free intervals；
* length：5、30、120、480 分钟；
* 使用 block concatenation，block 长度至少大于 score autocorrelation length；
* 100 个 bootstrap stream/length。

### Event streams

* 在长背景中插入真实 event instance；
* 单事件、重复事件、相邻事件；
* event prevalence 取多档；
* 保留原始局部时间结构。

## I.3 最小 prototype

### Step 1 — 基础 score audit

绘制：

* null score distribution；
* autocorrelation；
* drift by video/time/domain；
* fixed threshold 下 false alarms/hour；
* `P(any false alarm by horizon H)`。

如果没有明显随 horizon 增长的问题，主路线立即降级。

### Step 2 — 四类 baseline

1. fixed per-frame/per-packet threshold；
2. threshold calibrated per fixed clip；
3. Bonferroni/alpha-spending；
4. CUSUM/SPRT 或 WATCH-style conformal martingale。

### Step 3 — 最小 e-process

* 使用 calibration null blocks 得到 p-values；
* 先采用保守 predictable betting function；
* 每类只允许一个 active process；
* alarm 后固定 refractory period；
* 不在 48 小时内解决完整 overlapping instance FDR。

### Step 4 — 评价

主图：

```text
x-axis: median detection delay
y-axis: probability of any false alarm / FDR
curves: different methods
facets: stream length
```

附图：

* recall at α；
* false alarms/hour；
* risk calibration；
* score source sensitivity。

## I.4 成功阈值

初始 `α=0.05`。

必须同时满足：

1. 在所有 null horizon 上：

[
\widehat P(\text{ever false alarm}) \le 1.2\alpha
]

或置信区间覆盖 α；

2. 固定阈值的 risk 随 horizon 明显上升；
3. 在 matched empirical risk 下，方法的 event recall 不低于最佳 baseline 的 90%；
4. median delay 相对 matched-risk fixed threshold 增加不超过：

   * `0.5 s`，或
   * 15%，

   取较宽松者；
5. 至少两个 causal score source 上方向一致；
6. block bootstrap 与整视频 bootstrap 结论一致。

## I.5 立即终止条件

出现任一项就停止把它作为主线：

* fixed threshold 在真实 horizon 上并未产生风险增长；
* e-process 在 block-dependent null 上严重失校准；
* 为满足 α，recall 下降超过 20%；
* delay 增长超过 100%；
* Bonferroni/CUSUM 在所有设置中支配提议方法；
* 结果只在随机打乱时间后成立；
* 必须训练大型模型才能看出差异；
* 只能控制 false alarms/hour，无法给出任何 horizon-valid 或 FDR 解释。

## I.6 资源预算

| 资源                           |                               预算 |
| ---------------------------- | -------------------------------: |
| GPU feature/logit extraction |                    0–6 GPU-hours |
| CPU bootstrap / calibration  |                   4–12 CPU-hours |
| 新训练                          |            小型 probe，≤2 GPU-hours |
| 人工标注                         |                                0 |
| 存储                           |    现有 features/logits，通常 <100 GB |
| 代码改动                         | 独立 evaluator/wrapper，不动 backbone |

## I.7 48 小时结束时必须产出的材料

1. 一张 false-alarm-vs-horizon 图；
2. 一张 risk–delay frontier；
3. calibration table；
4. fixed threshold/CUSUM/e-process 三方结果；
5. null dependence audit；
6. 一页 GO/NO-GO 结论；
7. 若 GO，再决定是否修复 PCEH 作为 score baseline。

---

# J. Clarification Questions

以下问题按信息价值排序；答案会实际改变 Top-5 排名或 GO/NO-GO。

1. **真实目标场景中，一次误报、漏报和晚报分别造成什么代价？**
   若长期误报代价不高，T01 会降级；若漏过短事件代价极高，P01 会降级。

2. **你是否拥有连续数小时的真实负视频流，还是只有 THUMOS/ActivityNet 等离线数据？**
   没有真实负流不会阻止 48 小时 pilot，但会降低 T01 的最终论文上限。

3. **是否愿意让论文主贡献是 task/evaluation/theory，而不是新神经网络？**
   若答案是否定的，当前 Top-1 不适合作为主线。

4. **是否可获得 force、IMU、接触开关、工具状态、机器人日志或同步多视角中的任一种？**
   若可以，T02 可能超过 T01 成为主线。

5. **是否能组织 5–10 名标注者进行逐 prefix 的“首次可确认”标注？**
   若不能，T02 必须改为 sensor-only partial identification。

6. **部署系统是否会自然产生延迟反馈，例如用户纠错、质检结果、任务成败或人工复核？**
   若有，L01 排名会显著上升；若没有，应从 Top-5 删除。

7. **是否有可控制相机采样率并测量真实功耗/唤醒延迟的设备？**
   若有，P01 可上升到前三；若只有 feature replay，它更适合作为次线。

8. **你愿意接受哪类统计保证假设：块交换性、受限漂移、条件 e-value，还是只能做经验 calibration？**
   这决定 T01 是 NeurIPS/ICLR 级理论路线，还是 CVPR task/evaluation 路线。

9. **目标投稿是最近一届 CVPR，还是允许 6–12 个月的数据采集周期？**
   短周期推荐 T01；长周期且有传感数据时推荐 T02。

10. **当前是否已经存在可导出的 endpoint-only/PCEH packet-level logits，而不需要重跑完整 30 epoch？**
    若没有，48 小时 pilot 应直接使用冻结特征 probe，而不应先启动昂贵 PCEH 训练。

[1]: https://openaccess.thecvf.com/content/ICCV2021/html/Kang_CAG-QIL_Context-Aware_Actionness_Grouping_via_Q_Imitation_Learning_for_Online_ICCV_2021_paper.html "https://openaccess.thecvf.com/content/ICCV2021/html/Kang_CAG-QIL_Context-Aware_Actionness_Grouping_via_Q_Imitation_Learning_for_Online_ICCV_2021_paper.html"
[2]: https://arxiv.org/abs/2504.20041 "https://arxiv.org/abs/2504.20041"
[3]: https://arxiv.org/abs/2605.09976 "https://arxiv.org/abs/2605.09976"
[4]: https://arxiv.org/abs/1703.09026 "https://arxiv.org/abs/1703.09026"
[5]: https://arxiv.org/abs/2505.04608 "https://arxiv.org/abs/2505.04608"
[6]: https://arxiv.org/abs/2603.08620 "https://arxiv.org/abs/2603.08620"
[7]: https://arxiv.org/abs/2404.01933 "https://arxiv.org/abs/2404.01933"
[8]: https://arxiv.org/abs/2508.15717 "https://arxiv.org/abs/2508.15717"
[9]: https://arxiv.org/abs/2008.11170 "https://arxiv.org/abs/2008.11170"
[10]: https://openaccess.thecvf.com/content/CVPR2023/papers/Foo_System-Status-Aware_Adaptive_Network_for_Online_Streaming_Video_Understanding_CVPR_2023_paper.pdf "https://openaccess.thecvf.com/content/CVPR2023/papers/Foo_System-Status-Aware_Adaptive_Network_for_Online_Streaming_Video_Understanding_CVPR_2023_paper.pdf"
[11]: https://arxiv.org/abs/2208.12673 "https://arxiv.org/abs/2208.12673"
[12]: https://arxiv.org/abs/2603.15847 "https://arxiv.org/abs/2603.15847"
[13]: https://arxiv.org/abs/2011.10830 "https://arxiv.org/abs/2011.10830"
[14]: https://arxiv.org/abs/2604.15173 "https://arxiv.org/abs/2604.15173"
[15]: https://arxiv.org/abs/2403.08755 "https://arxiv.org/abs/2403.08755"
[16]: https://arxiv.org/abs/1802.09098 "https://arxiv.org/abs/1802.09098"
[17]: https://arxiv.org/abs/2604.18459 "https://arxiv.org/abs/2604.18459"
[18]: https://arxiv.org/abs/2606.16353 "https://arxiv.org/abs/2606.16353"
[19]: https://arxiv.org/abs/1811.12432 "https://arxiv.org/abs/1811.12432"
[20]: https://openaccess.thecvf.com/content/ICCV2023/html/Cao_E2E-LOAD_End-to-End_Long-form_Online_Action_Detection_ICCV_2023_paper.html "https://openaccess.thecvf.com/content/ICCV2023/html/Cao_E2E-LOAD_End-to-End_Long-form_Online_Action_Detection_ICCV_2023_paper.html"
[21]: https://arxiv.org/abs/2408.09764 "https://arxiv.org/abs/2408.09764"
[22]: https://arxiv.org/abs/2510.12385 "https://arxiv.org/abs/2510.12385"
[23]: https://openaccess.thecvf.com/content/CVPR2024/papers/Shen_Progress-Aware_Online_Action_Segmentation_for_Egocentric_Procedural_Task_Videos_CVPR_2024_paper.pdf "https://openaccess.thecvf.com/content/CVPR2024/papers/Shen_Progress-Aware_Online_Action_Segmentation_for_Egocentric_Procedural_Task_Videos_CVPR_2024_paper.pdf"
[24]: https://proceedings.neurips.cc/paper_files/paper/2025/file/6dd91fec726dbed8915a1fbadd91d1d2-Paper-Conference.pdf "https://proceedings.neurips.cc/paper_files/paper/2025/file/6dd91fec726dbed8915a1fbadd91d1d2-Paper-Conference.pdf"
[25]: https://arxiv.org/abs/2606.01631 "https://arxiv.org/abs/2606.01631"
[26]: https://arxiv.org/abs/2506.18472 "https://arxiv.org/abs/2506.18472"
[27]: https://arxiv.org/abs/2606.04970 "https://arxiv.org/abs/2606.04970"
[28]: https://arxiv.org/abs/2309.04579 "https://arxiv.org/abs/2309.04579"
[29]: https://arxiv.org/abs/2512.05774 "https://arxiv.org/abs/2512.05774"
[30]: https://arxiv.org/abs/2309.17024 "https://arxiv.org/abs/2309.17024"
[31]: https://arxiv.org/abs/2010.07110 "https://arxiv.org/abs/2010.07110"
[32]: https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2307_ECCV_2022_paper.php "https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2307_ECCV_2022_paper.php"
[33]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01621.pdf "https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01621.pdf"
[34]: https://arxiv.org/abs/2503.11194 "https://arxiv.org/abs/2503.11194"
[35]: https://arxiv.org/abs/2604.18107 "https://arxiv.org/abs/2604.18107"
