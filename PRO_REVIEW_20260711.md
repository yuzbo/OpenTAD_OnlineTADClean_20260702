审查锚点为目标分支当前可见最新提交 **`bfd0608b2996cba30d158d741ee193476a5078df`**。该提交仍把正式训练标记为未就绪，只是把 3-epoch pilot 的验证推迟到第 2 epoch，并把 Slurm pilot 配额从 24 小时延长到 48 小时；这说明当前变化解决的是“更可能跑完”，不是“方法和证据已经成立”。

---

# 一、调研结论表

## 1. 先区分四种经常被混淆的“在线”

| 层次       | 严格定义                                       | PCEH 当前状态                                                     |
| -------- | ------------------------------------------ | ------------------------------------------------------------- |
| 模型输入因果   | 在决策时刻 (t)，输入张量只包含来源时间 (\le t) 的帧或特征        | 基本满足                                                          |
| 模型状态因果   | cache、attention、卷积、Mamba state 不使用 (>t) 信息 | 基本满足，已有 replay smoke；但仍需覆盖正式 checkpoint                       |
| 标注/采样因果  | 训练时选择哪些 prefix、如何构造 target，也不使用未来 GT       | **不满足，也没有必要强求**；完整 GT 可合法用于训练采样，但必须披露为 privileged supervision |
| 推理输出不可修订 | 一旦 emit，不允许以后更新 segment、score 或删除记录        | 当前 ledger 骨架满足                                                |

因此，未来采用 event-centric prefix sampling 后，能成立的准确表述应是：

> 模型训练输入和计算图保持 prefix-causal；训练 sampler 使用完整标注选择监督时刻；测试始终在完整 chronological stream 上运行，推理不访问未来、不回改输出。

不能写成“training and inference never use future information”，因为 sampler 和 target 明确知道 GT endpoint。

## 2. 代表性在线视频方法

| 方法           |            年份 | 核心训练和推理方式                                                                | 训练期未来信息                             | 推理期未来帧         | Raw-video E2E           | 延时与计算判断                                                                             |
| ------------ | ------------: | ------------------------------------------------------------------------ | ----------------------------------- | -------------- | ----------------------- | ----------------------------------------------------------------------------------- |
| LSTR         |          2021 | 长期记忆加短期当前窗口，面向 Online Action Detection                                   | 通常按离线标注训练                           | 不需要            | 未证明视觉塔联合微调              | 依赖预提特征和长短记忆；是 OAD，不直接输出完整动作 span。([arXiv][1])                                       |
| OadTR        |          2021 | 历史编码器加未来动作表示 anticipation                                                | 可使用未来动作标签做 anticipation supervision | 不读取真实未来帧       | 未证明                     | “预测未来表示”不等于读取未来帧，但任务更接近帧级 OAD。([arXiv][2])                                          |
| TeSTra       |          2022 | 流式 Transformer，通过历史压缩避免每步重算整个窗口                                          | 离线训练序列                              | 不需要            | 通常为特征级                  | 相比长窗口滑动 Transformer 报告约 6 倍加速，核心贡献是 streaming update，而非 span emission。([arXiv][3])  |
| SimOn        |          2022 | 当前观测查询历史视觉上下文，输出不可回改的在线定位结果                                              | 完整标注监督                              | 不需要            | 检测器端到端；视觉塔是否联合更新不是其核心证据 | 与 PCEH 的 immutable online TAL 定义较接近。([arXiv][4])                                    |
| E2E-LOAD     |          2023 | 原始 RGB 端到端 OAD；短历史训练、长历史 cache 推理，历史分支 detach                            | 完整视频标签可用于监督                         | 不需要            | **是**                   | 明确针对 raw-video E2E 成本；论文报告约 17.3 FPS。其“短训练、长推理”思想对 PCEH 很有价值。([arXiv][5])           |
| MATR         |          2024 | 当前 segment 预测动作结束，历史 memory queue 检索开始位置                                 | 离线 GT 训练                            | 不需要            | 未证明 raw tower E2E       | segment/chunk 级更新，已覆盖 memory-based online span 建模，PCEH 必须证明不只是换了状态机。([arXiv][6])    |
| HAT          |          2024 | 历史增强 anchor，联合短期与长期历史                                                    | 离线 GT 训练                            | 不需要            | 未证明                     | 也是历史上下文增强的 online TAL；延时通常由 segment/anchor cadence 隐含决定。([arXiv][7])                |
| ActionSwitch |          2024 | 类无关 action-switch 状态，处理并发和同类动作，加入保守发射约束                                  | 离线 GT 训练                            | 不需要            | 未证明                     | 对当前“一类一条 active track”的设计构成直接攻击点。([arXiv][8])                                       |
| CMeRT        |          2025 | 用 recurrent/cross-memory 训练减轻短窗口训练与长流推理不一致                               | 离线序列监督                              | 不需要            | 未证明                     | 直接说明短 clip 训练如果没有 memory-consistency 设计，会出现 train–test history gap。([arXiv][9])     |
| Mamba-OTR    |          2025 | 短 clip 训练、recurrent Mamba streaming inference                            | 离线监督                                | 不需要            | 任务设置较专用                 | 证明 Mamba 可以作为流式状态更新器，但“使用 Mamba”本身不等于严格 On-TAL。([arXiv][10])                        |
| OnPoint      |          2026 | offline teacher 向 online student 蒸馏 pseudo-segment、CAS 和 anticipation 信息 | **offline teacher 可获得更完整上下文**       | student 不需要    | 未证明 raw tower E2E       | 说明未来信息可作为训练期 privileged teacher，但必须保证 teacher 在评测完全消失。论文已标注 ECCV 2026。([arXiv][11]) |
| OZ-TAL       | 2026 preprint | training-free VLM/ViCLIP，当前与历史窗口、在线 span state、不可回改                      | 不训练                                 | 不需要            | 否                       | 已把“开放词汇、零训练、严格在线 span”变成竞争面；重复 VLM 滑窗仍可能昂贵。([arXiv][12])                            |
| CausalTAD    |          2024 | offline TAD 中引入 causal attention/Mamba 路径                                | 离线完整视频训练                            | 某些设置可使用过去或未来方向 | 否，不能据名字判定               | 它不是天然 strict-online；保留未来方向或居中处理时仍是 offline TAD。([arXiv][13])                        |

### 关键调研判断

1. **Inference-time causal 与 training-time causal 必须分开报告。**
   E2E-LOAD、CMeRT、Mamba-OTR 都表明：可以用短片段训练、在长流上递归推理；但必须额外处理历史状态、detach 和训练—推理上下文不一致。([arXiv][5])

2. **尾随 sliding window 可以是 inference-causal，但不必然是真正的 online system。**
   只要窗口结束于当前时刻且无 retrospective NMS，它不读取未来；然而若每步重算大量重叠窗口、依赖全视频预提特征或在输出后回改，它仍不能宣称低成本、raw-video、immutable online。

3. **offline teacher 不会自动污染 online inference claim。**
   OnPoint 和早期 privileged-knowledge distillation 都允许 teacher 只在训练期工作；但论文必须写成“offline-to-online distillation”，不能声称训练过程没有未来知识。([arXiv][11])

## 3. 与训练降成本最相关的方法

| 路线          | 代表工作            | 可迁移结论                                                                                                   |
| ----------- | --------------- | ------------------------------------------------------------------------------------------------------- |
| 选择性视觉反向传播   | ETAD            | 可以对全部 snippet 做便宜前向，只让部分 snippet 保留视觉梯度，并对子提议训练；说明“不是每个时间点都做完整 visual backward”是合法的效率策略。([arXiv][14])   |
| 缓存长视频视觉特征   | TALLFormer      | 冻结或部分更新视觉表示，用缓存覆盖长上下文，仅更新一小部分帧；适合 PCEH 的 frozen Stage 1。([arXiv][15])                                   |
| 短历史训练、长历史推理 | E2E-LOAD        | 可将过去状态 detach，训练较短 history，但正式推理保持长 cache；必须做状态一致性实验。([arXiv][5])                                       |
| 视频 Adapter  | ST-Adapter、AIM  | 冻结图像 foundation model，仅训练时空 adapter；ST-Adapter 报告只更新约 8% 参数，AIM 也采用冻结图像模型加空间/时间 adapter。([arXiv][16])   |
| LoRA        | 官方 PEFT         | 可精确指定 attention projection 和 layer 范围；但必须 fail-closed 验证实际 target module，而不是假设模块命名。([Hugging Face][17]) |
| 离散时间生存损失    | Nnet-survival 等 | 合法 hazard 需要明确 risk set、唯一事件时刻或右删失；事件后时间点必须从 risk set 删除。([arXiv][18])                                  |

我没有在核验到的代表性 On-TAL/OAD 论文中找到“**endpoint hazard + first-emission hazard**”按正式离散生存风险集共同训练的直接对应方法。这不是不存在证明，但说明该方向可能有研究空间。问题是：**当前代码还没有实现这种统计模型。**

---

# 二、现有代码审查

## 1. 总体结论表

| 审查项                            | 当前判定                           | 代码事实                                                                       |
| ------------------------------ | ------------------------------ | -------------------------------------------------------------------------- |
| 推理输入 strict online             | **基本 PASS**                    | packet 只允许选择 `[start,end)` 内帧；当前配置每 8 帧只编码最近一帧。                            |
| cache 与 temporal projection 因果 | **PASS，需正式 checkpoint replay** | cache 只追加当前特征并截断；projection 使用 left-only 卷积。                               |
| GT 泄漏                          | **条件 PASS**                    | 完整 GT 被附加到每个训练 packet，但当前仅用于 target；接口隔离不够强。                               |
| provenance 审计                  | **smoke 级 PASS**               | 有 raw/cache 最大来源时间、suffix perturbation 和 prefix trace equivalence。         |
| endpoint 与 emission 解耦         | **FAIL**                       | 两者 event label 来自同一 endpoint crossing；late prefix 还被训练为 emission positive。 |
| immutable ledger               | **PASS 骨架**                    | record 冻结、只 append，检查 end/source 不超过 emit。                                 |
| OnlineAP late handling         | **主要逻辑 PASS**                  | late prediction 保留为 FP，对应 GT 仍为 FN；主延时使用 matched GT end。                   |
| full visual tower E2E          | **FAIL**                       | SigLIP2 被冻结，外层还用 `no_grad()`；不存在可核验的 finetune config。                      |
| 多 GPU streaming                | **FAIL/未实现**                   | sampler 明确禁止 `world_size != 1`。                                            |
| formal training readiness      | **FAIL**                       | config 明确 `formal_training_ready=False`。                                   |

## 2. 当前是否真的是 strict online？

### 成立的部分

* packet manifest 是连续、半开区间；
* `packet_recent_frame` 只选 `packet_end-1`；
* detector 要求下一 packet 的 start 严格等于上一 packet 的 end；
* cache source frame 和 raw source frame 都不得晚于当前 decision frame；
* `CausalTemporalMaxerProj` 使用 left padding，不使用 centered convolution 或右侧数据。

因此，**对当前 per-frame SigLIP2 + causal projection 路径，推理计算图是可信的 strict prefix computation**。

### 必须限制的表述

当前只证明了：

> 在已执行的 frozen/adaptor smoke 范围内，改变 cut 之后的 raw image tensors 不会改变 cut 及以前的 logits、track state 和 emission。

replay 确实在 image tensor 域扰动未来 packet，并要求未来 trace 本身发生变化，避免“扰动根本没生效”的假通过。

它尚未证明：

* 30-epoch checkpoint 的所有模块仍满足；
* LoRA/full-finetune 新路径满足；
* 多 worker、DDP、feature cache 路径满足；
* provenance 不是由错误 metadata 自报出来的。

## 3. 是否存在未来信息泄漏？

### 模型输入侧：未发现直接泄漏

非 terminal packet 会删除 duration、total frames 等 terminal-only metadata。

### 训练监督侧：存在完整未来标注，但属于监督，不应伪装成“prefix observable”

`StreamingRawFrameDataset` 在每个 packet 上附加整段视频的 `stream_gt_segments` 和 `stream_gt_labels`。

这不是 inference leakage，但有两个问题：

1. 模型 `forward()` 直接接收完整视频 GT，未来有人修改 detector 时很容易误用；
2. `ongoing`、`completion`、`late` 等 target 的判定都依赖完整 endpoint annotation。

建议改为：

```text
batch.inputs / batch.metas            → 只能给模型
batch.supervision.absolute_segments   → 只能给 target builder / loss wrapper
```

并增加 taint test：任何 `stream_gt_*`、duration、video_end_frame 都不能进入 backbone、projection、head 的 kwargs 或 metadata。

## 4. endpoint hazard 与 emission hazard 是否真的解耦？

**没有。当前只有参数解耦，没有统计目标解耦。**

### Critical issue 1：两个事件标签完全相同

当 `previous_frame < end <= current_frame` 时，代码同时设置：

```text
end_event = 1
emit_event = 1
```

因此，“first emission event”没有独立发生时间。

### Critical issue 2：late prefix 被持续标成 emission positive

代码构造：

```text
emit_target = emit_allowed + emit_event + late_target
```

只要某个事件已经超过延时预算，`late_target=1`，emission head 仍被要求输出正值；`delay_loss` 还会进一步推高该 logit。

这不符合 first-event survival：

* 真正 first emission 只能有一次；
* emission 后的时间点必须离开 risk set；
* “已经太晚”应是高代价状态，而不是新的正事件。

### Critical issue 3：target 按 class 聚合，而不是按 instance/track

只要同一视频某类动作已经完成过一次：

* `completion_target[class]` 从此可以一直为 1；
* 旧事件超过预算后，`late_target[class]` 可以在后续很长时间保持 1；
* 新的同类实例与旧实例状态发生混叠。

这会让 emission head 学到“该类历史上结束过，所以现在应该 emit”，而不是“当前 active instance 已完成”。

现有单元测试主要覆盖单实例，没有覆盖同视频多次同类动作。

### Critical issue 4：decoder 中 end 与 emit 仍是同一时刻

发射时：

```text
end_frame = current_frame
emit_frame = current_frame
```

于是当前模型生成的：

```text
emit_time - predicted_end_time = 0
```

`predicted_end_latency` 诊断几乎恒为 0，无法说明模型是在 endpoint 后等待，还是准确识别 endpoint。

### 裁决

当前方法最多可以称为：

> 独立参数的 endpoint、completion、emission gates，以及 absorbing ledger decoder。

不能称为：

> statistically decoupled endpoint hazard and first-emission hazard learning。

## 5. packet、cache、read provenance 是否可信？

可信度高于一般 online 代码，但仍是 **executable audit，不是形式证明**。

已有优点：

* frame index 不得逃出当前 packet；
* cache source frame 单调追加；
* raw/cache 最大来源时间都记录到 emission；
* future suffix perturbation 要求 prefix trace 完全一致；
* ledger 检查 `end≤emit`、`raw_source≤emit`、`cache_source≤emit`。

还需要增加：

* feature cache manifest 的模型 revision、processor hash、frame-policy hash；
* raw-frame 与 cached-feature 同帧等价测试；
* `encoded_source_frames` 与实际 decoder 请求位置的一致性日志；
* GT taint audit；
* LoRA 后 cache 是否由当前 checkpoint 生成的版本检查；
* emission 唯一 ID 和 append-only hash chain。

## 6. OnlineAPBudgeted 是否正确处理 late prediction？

主要逻辑是正确的。

它不是先删除 late prediction 再跑 offline mAP，而是：

* 所有 immutable emission 都进入 score 排序；
* 只有 class、tIoU 和 `0≤emit-GT_end≤budget` 同时满足才算 TP；
* late prediction 是 FP；
* 对应 GT 未匹配，仍是 FN。

主指标也明确写成：

```text
emit_time_sec - matched_ground_truth_end_sec
```

predicted-end latency 单列为 diagnostic。

### 需要修补的评测问题

1. **当前 predicted-end diagnostic 没有信息量**
   因为 decoder 设置 `predicted_end=emit_time`。

2. **必须同时报告 latency coverage**
   只对 TP 统计 p50/p90/p95，模型可以通过只检测少量容易事件获得很低 latency。必须同时报告 matched TP count、recall、FN。

3. **固定 class universe**
   evaluator 应显式接收 20-class map，而不是只从 GT/prediction union 推断。

4. **ledger 审计应进入 evaluator 前置条件**
   包括 `max_cache_source_frame`、唯一 emission ID、每 stream 单调 append。

5. **不同 budget 不应重新调阈值**
   主模型阈值应在 validation 一次锁定，0.5/1/2/4 秒只改变评测 eligibility。

## 7. 当前训练路径是否真的端到端？

需要分三个层次回答。

| Claim                                      | 当前是否成立           |
| ------------------------------------------ | ---------------- |
| 从 raw frame tensor 到检测 loss 位于同一个调用链       | 是                |
| temporal adapter、projection、PCEH head 联合更新 | smoke 证明至少部分参数更新 |
| SigLIP2 visual tower 参与梯度更新                | **否**            |
| 跨 packet BPTT                              | **否**            |
| full visual tower finetune                 | **否**            |

配置同时设置：

* 外层 `OnlineVideoMAEAdapter.freeze_backbone=True`；
* 内层 `OnlineSigLIPFrameEncoder.freeze_vision_encoder=True`；
* detector `detach_stream_state=True`。

外层 freeze 还把整个 frame backbone forward 包在 `torch.no_grad()` 中。

因此当前准确名称是：

> raw-input, frozen-SigLIP2, trainable temporal suffix route。

不是：

> end-to-end finetuned SigLIP2 On-TAL。

此外，当前提交中没有找到 `configs/causaltad/thumos_pceh_ontad_finetune.py`。

### 另一个隐藏问题：当前 temporal adapter 几乎没有跨 packet 时间上下文

执行顺序是：

```text
current packet raw frame
→ SigLIP2
→ CausalTemporalAdapter
→ append to detector cache
→ causal projection over cache
```

而主配置每个 packet 只编码一个 recent frame。于是 adapter 每次看到的 (T=1)，kernel-size 3 的左侧输入只是零 padding；它无法利用上一 packet 的视觉 token。真正的跨 packet temporal modeling 在 projection，而不是这个 adapter。

建议把 temporal adapter 移到 cache 之后，或让它维护显式 recurrent state。

## 8. frozen、adapter、LoRA、full-finetune 的 claim 边界

| 路线                                          | 合法 claim                                                                                                |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Frozen feature cache                        | “frozen SigLIP2 visual encoder + learned causal temporal detector”；不能说 raw-video E2E finetune           |
| Frozen raw-frame forward + temporal adapter | “raw-frame online inference with frozen visual foundation model and trainable temporal adaptation”      |
| SigLIP2 LoRA                                | 只有所有目标 attention block 的 LoRA 参数均有 nonzero gradient 和 delta 后，才能说 parameter-efficient visual adaptation |
| Full visual tower                           | 必须所有计划解冻层进入 optimizer、有梯度、有参数变化，并有完整训练结果，才能称 full finetune                                              |

当前 `training_audit` 只要求某个 module prefix 中“至少一个参数”有梯度和更新，不能证明每个 LoRA block 都被覆盖。

## 9. 当前证据等级

| 等级                | 当前状态                     | 能说明什么                                        |
| ----------------- | ------------------------ | -------------------------------------------- |
| 启动成功              | 是                        | config、路径、模型和训练入口可执行                         |
| Smoke 通过          | 是，仅 frozen/adaptor route | 8 packets 范围内 packet/update/causal replay 通过 |
| Short pilot       | **否**                    | 曾启动但取消；最新提交只是给 48 小时预算                       |
| Formal multi-seed | **否**                    | 没有完整单种子，更没有三种子                               |
| Paper-ready       | **否**                    | 方法目标、成本、基线和统计结果均未闭环                          |

远端脚本也明确只支持 single-GPU smoke/pilot，并要求 pilot 前必须已有 packet、update、causal replay 三项 smoke artifact。

---

# 三、训练成本诊断

## 1. 按现有观测估算

用户给出的约 152,670 packets/epoch、7 小时/epoch 意味着：

* 约 **6.06 packets/s**；
* 每个 packet/optimizer step 约 **165 ms**；
* 30 epochs 约 **210 GPU-hours/模型/种子**；
* PCEH 与 endpoint-only，3 seeds，共约
  [
  210\times2\times3=1260
  ]
  GPU-hours；
* 仅 3-epoch PCEH 与 endpoint pilot 配对也约 **42 GPU-hours**。

这还没有包含 stride-2、chunk-end、LoRA/full-finetune、消融和失败重跑。

## 2. 成本并不主要来自“视频很长”，而是执行粒度过细

### 瓶颈 1：152,670 次 Python/optimizer 事件

每个 packet 都执行：

```text
dataset __getitem__
→ video seek
→ processor
→ SigLIP2 forward
→ cache concat
→ projection over cache
→ head
→ backward
→ optimizer.step
```

即使每次只编码一帧，152,670 次小 kernel、Python dispatch、optimizer step 的固定开销也很大。

### 瓶颈 2：每个 packet 重新打开和关闭视频

`LoadRawFrames` 对每个样本：

* `cv2.VideoCapture(video_path)`；
* seek 到目标帧；
* 读取；
* `cap.release()`。

这意味着约 152,670 次 open/seek/close，完全没有利用 packet 在同一视频中连续的事实。

### 瓶颈 3：SigLIP processor 的 GPU→CPU→GPU 往返

transformers backend 把 pixels 显式 `.detach().cpu()` 交给 processor，再把结果搬回原设备。

对于每次一帧的调用，这一往返可能比部分 GPU 计算本身还低效。

### 瓶颈 4：`num_workers=0`

主配置 train/val/test 全是 `num_workers=0`。

视频解码、CPU processor 与 GPU 计算无法重叠。

### 瓶颈 5：每步重算完整 192-token projection

detector 每个 packet 都把 cache 截到 192，然后重新运行 projection；不是递归更新 projection state。

### 瓶颈 6：单 GPU 被代码强制锁死

`ChronologicalStreamBatchSampler` 明确拒绝 `world_size != 1`。

## 3. Feature cache 的实际规模并不大

当前每 packet 只产生一个 768-D frame feature。152,670 个 fp16 feature 大约：

[
152670\times768\times2 \approx 234.5\ \text{MB}
]

约 223.6 MiB，不含 metadata。即使加入 source frame、video index、mask、hash，也远小于反复做 30 次 SigLIP2 forward 的成本。

## 4. Event-sampled 的保守线性成本估计

| 每 epoch 监督决策点 | 相对 152,670 的压缩 | 按 7h 线性估算 |
| ------------: | -------------: | --------: |
|        24,000 |          6.36× |   约 1.10h |
|        12,000 |         12.72× |   约 0.55h |

这还没有计入：

* feature cache 消除视觉 forward；
* episode batch 化；
* 减少 optimizer step；
* 消除 VideoCapture per-packet open/close；
* GPU-native preprocessing。

因此 Stage 1 达到远低于 1 小时/epoch 是合理工程目标，但在 benchmark 前不能当作既成事实。

---

# 四、候选训练策略比较

| 方案                               | 科学有效性                               |            成本 | 主要风险                                      | 裁决                            |
| -------------------------------- | ----------------------------------- | ------------: | ----------------------------------------- | ----------------------------- |
| A. Full packet streaming         | 训练分布与完整流最一致                         |  当前约 7h/epoch | 152k 小 step、I/O、无法 DDP                    | **只保留为 gold subset protocol** |
| B. Event-Centric Prefix Sampling | 输入仍严格 prefix-causal，可显著减少监督时刻       |           低至中 | 采样偏差、状态上下文不足、使用未来 GT 选点                   | **主训练策略**                     |
| C. Two-Stage Training            | frozen→LoRA→可选 full FT，证据分层清晰       |            可控 | 阶段间 feature mismatch                      | **主框架**                       |
| D. Feature Cache Training        | frozen encoder 时可与 raw forward 精确等价 |            很低 | 不能宣称视觉塔 E2E；cache stale                   | **Stage 1 主实验，不只是 ablation**  |
| E. Offline Teacher Distillation  | 可改善低数据、prefix student               |             中 | 未来 teacher 让 claim 变成 privileged training | **后置 ablation**               |
| F. Video-Contiguous DDP          | 保留每个视频内部顺序，可降低 wall time            | 总 GPU-hours不降 | uneven streams、DDP hang、state ownership   | **第二优先级工程**                   |
| G. SigLIP2 LoRA/Adapter          | 比 full FT 更符合数据规模和成本约束              |             中 | 当前外层 `no_grad` 会直接阻断 LoRA                 | **Stage 2 首选**                |

## A. Full packet streaming

### 优点

* 不需要采样校正；
* state、ledger、rearm 和背景分布最真实；
* 可用于验证 sampled objective 是否偏离。

### 缺点

* 当前单个模型三 epoch 已约 21 小时；
* 30 epoch 单种子约 210 GPU-hours；
* 对 PCEH 与 endpoint-only 的正式三种子比较不可接受；
* full visual tower 反向传播只会进一步增加内存和时长。

### 推荐用途

只保留两种：

1. **5–10 个视频或不超过 10,000 packets 的 gold subset**；
2. 最佳 sampled checkpoint 上做一次短的 full-stream continuation/audit。

不能作为默认正式训练。

## B. Event-Centric Prefix Sampling

### 采样单位必须是“短连续 causal episode”，不是孤立单点

孤立 prefix 会破坏：

* start→ongoing→end 的状态迁移；
* absorbing first emission；
* rearm；
* cache age；
* hard background 后的误触发行为。

建议每个 episode 包含：

```text
192 个历史 decision tokens 作为 burn-in/context
+ 4–8 个连续监督 decision bins
```

Stage 1 从 feature cache 读取完整 192-token context；loss 只作用于采样出来的监督位置。

### 建议采样构成

令 (B=8) frames，使用与现有代码相同的 endpoint-crossing 量化规则：

```text
previous_frame < end <= current_frame
```

每个实例至少构造：

* endpoint：`e-4B, e-2B, e-B, e, e+B`；
* start：`s-B, s, s+B`；
* ongoing：动作内部 25%、50%、75% 或随机点；
* hard background：start 前、end 后 1–2B，但不与任何动作重叠；
* long background：距离任意 GT 至少 4B；
* 后续可加入上一 epoch 模型高 actionness 的 false-positive mining。

建议初始比例：

| 类型                     |  比例 |
| ---------------------- | --: |
| endpoint risk episodes | 35% |
| start episodes         | 15% |
| ongoing                | 20% |
| hard background        | 15% |
| long background        | 15% |

所有正 endpoint/start crossing 都应保证覆盖；主要对子采样背景和冗余 ongoing。

### 如何避免采样偏差

event-centric sampling 会大幅改变真实 base rate。必须记录每个 decision bin 的 inclusion probability (\pi_t)，用：

[
\widehat L=
\frac{1}{Z}
\sum_{t\in S}
\frac{1}{\pi_t}\ell_t
]

或按 target type 做 self-normalized importance weighting。建议：

* 对权重设置上限，避免极端方差；
* 报告 effective sample size；
* 在 gold subset 上比较 exhaustive loss 与 sampled loss；
* 比较两者 gradient cosine；
* 阈值必须在完整 chronological validation 上校准。

### 为什么不破坏 online claim

因为采样只决定“训练哪个 (t)”：

```text
GT endpoint → 选择训练 prefix t
模型输入   → 只包含 frame/source ≤ t
```

最终 test 仍从第一个 packet 跑到视频结束。应披露“annotation-guided prefix sampling”，但 inference-time online claim 保留。

## C. Two-Stage Training

### Stage 1：frozen SigLIP2

训练：

* cache 之后的 causal temporal adapter；
* projection；
* PCEH/endpoint head。

必须审计：

* visual encoder 全部 `requires_grad=False`；
* cache 与 raw feature 一致；
* adapter/projection/head 有非零梯度和 delta；
* episode forward 与逐 packet replay 等价。

### Stage 2：LoRA

训练：

* SigLIP2 顶部 4–6 个 visual blocks 的 attention `q_proj/v_proj`；
* 可选最后 LayerNorm；
* temporal adapter、projection、head。

必须审计：

* 每一个目标 block 都有 LoRA module；
* 每一个目标 block 都有非零梯度与参数更新；
* base weight 保持不变；
* outer wrapper 不得再用 `torch.no_grad()` 包住 LoRA；
* cache revision 与 LoRA checkpoint 一致。

### Stage 3：full visual tower

只在以下条件后做短 pilot：

* LoRA 有稳定正增益；
* raw-frame episodic path 已达到可接受吞吐；
* 显存和 wall time 已实测；
* full-tower update audit 可逐 block 通过。

它不是默认正式主路线。

## D. Feature Cache Training

冻结的 SigLIP2 当前是逐帧独立编码器；因此只要 preprocessing 和 checkpoint 完全相同，按 source frame 缓存 feature 与在线时现场编码该帧应当等价。模块本身也明确描述为 independently encoding each sampled frame。

### 可以作为主实验的 claim

> Frozen SigLIP2 features + learned causal online detector。

### 不能成立的 claim

> End-to-end visual representation learning from raw videos。

### 正式评测

主 paper result 应运行：

```text
raw chronological stream
→ SigLIP2
→ learned temporal detector
→ immutable ledger
```

cache 可以用于训练和完整 chronological validation 加速，但不能把保存的 raw predictions 当测试输入。

## E. Distillation From Offline Teacher

### 合法做法

* teacher 只在训练期；
* student 输入仍是 (V_{\le t})；
* test 不加载 teacher、teacher cache 或 teacher predictions；
* teacher signal 标注为 privileged supervision。

可蒸馏：

* class posterior；
* actionness/CAS；
* start confidence；
* endpoint posterior；
* proposal quality；
* hard-negative score。

### 更干净的优先级

1. 先尝试将 offline teacher 运行在截断视频 (V_{\le t}) 上；
2. 再考虑 full-video teacher；
3. full-video teacher 只能作为 ablation，不作为“future-free training”证据。

OnPoint 已占据 offline-teacher-to-online-student 的近邻位置，因此蒸馏不应成为 PCEH 的核心 novelty。([arXiv][11])

## F. Video-Contiguous DDP

当前 sampler 明确不支持。正确实现必须是：

```text
rank 0 owns complete videos A,C,...
rank 1 owns complete videos B,D,...
一个视频绝不能跨 rank
```

收益：

* 近似按 GPU 数降低 wall time；
* 每个 rank 内仍保留 chronological state。

风险：

* 视频长度不均导致 rank 步数不一致；
* DDP all-reduce 等待或 hang；
* state reset、ledger gather 容易错；
* 总 GPU-hours不会下降。

因此优先级低于 event sampler、cache 和批处理。Event-sampled episodes 本身可先用普通 DistributedSampler，更容易并行。

## G. LoRA / Adapter

LoRA 比 full FT 更合理，因为：

* THUMOS14 训练规模小；
* foundation model 很大；
* online TAD 需要的是动作边界和时序适配，不一定需要重写全部视觉语义；
* Adapter/LoRA 在视频迁移中已有较强参数效率先例。([arXiv][16])

建议初始设置：

```text
target blocks: top 4 visual transformer blocks
target modules: q_proj, v_proj
rank: 8
alpha: 16
dropout: 0.05
LoRA lr: 2e-5
temporal stack lr: 5e-5
head lr: 1e-4
```

但模块名必须从运行时 `named_modules()` 验证；零命中应直接报错，不能静默训练“LoRA config”。

---

# 五、推荐主路线

## 唯一推荐路线

> **Instance-aware Causal Risk-Set Event-Centric Prefix-Episode Training：Frozen SigLIP2 Feature Cache → Selective-Gradient LoRA，始终使用完整 chronological raw-stream evaluation。**

工作简称可用 **CRS-EPS**，但它首先是训练协议，不应在查新前包装为独立贡献。

## Stage 0：先修正方法，否则不要继续长训练

### 0.1 把 target 从 class-level 改为 instance/track-level

每个 GT instance (i) 单独维护：

* class；
* start decision bin；
* endpoint decision bin；
* end-risk set；
* completion state；
* emission-risk set；
* censor status。

然后再通过 assignment 聚合到有限 track slots，而不是先按 class 做 `any()`。

### 0.2 正确的 endpoint discrete hazard

令 (\tau_i) 为第一个跨过 GT endpoint 的 decision bin：

[
D^{end}_{i,t}=\mathbf 1[t=\tau_i]
]

[
R^{end}_{i,t}=
\mathbf 1[\text{instance }i\text{ 已开始且 endpoint 尚未发生}]
]

损失为：

[
L_{end}=
-\sum_{i,t}R^{end}*{i,t}
\left[
D^{end}*{i,t}\log h_{i,t}
+(1-D^{end}*{i,t})\log(1-h*{i,t})
\right]
]

事件发生后的 bins 必须 mask 掉。

### 0.3 endpoint first crossing 后冻结 predicted endpoint

状态机应改为：

```text
inactive
→ active
→ endpoint_detected_pending_emit
→ emitted_absorbing
```

在 endpoint hazard 第一次 crossing 时保存：

```text
predicted_end_frame
endpoint_score
endpoint_detection_frame
```

后续即使 emission 延迟，ledger 中：

```text
segment.end = frozen predicted_end_frame
emit_frame  = actual later decision frame
```

这样 predicted-end latency 才有意义。

### 0.4 对 emission 采取二选一的诚实定义

#### 推荐定义：latency-aware first-stop policy

利用训练 GT 和 causal candidate sequence 构造唯一 stop time (\sigma_i)：

* endpoint 前：禁止 emit；
* endpoint 后：评估当前冻结 segment 的 class/tIoU/置信度；
* 在延时预算内选择最早达到质量阈值或最大化
  [
  U(t)=Q(t)-\lambda(t-e_i)
  ]
  的时刻；
* 如果预算内无合格候选，则右删失；
* stop 之后全部 mask。

这属于 train-time oracle policy supervision，必须披露。

#### 不能继续的定义

不能再把所有：

```text
within_budget
late
```

都设为 emission positive。

如果不实现唯一 stop target，就应把模块改名为：

> completion-gated emission score

而不是 first-emission hazard。

### 0.5 支持多个同类 track

最低要求：

* 每类 (K) 个 active slots；
* instance-slot assignment；
* 同类 overlapping/back-to-back 单元测试；
* rearm 不依赖 start score必须先跌破阈值才能恢复。

否则 ActionSwitch 类工作会直接指出任务覆盖不完整。([arXiv][8])

## Stage 1：Frozen Cache Event-Sampled Training

### 数据和 batch

```text
decision cadence: 8 frames = 0.267s
context length: 192 decision tokens = 51.2s
supervised decision bins/epoch: 24,000
supervised bins/episode: 4–8
episodes/epoch: approximately 4,000–6,000
batch size: 32 episodes
AMP: on
grad clip: 1.0
```

### 模型结构

```text
cached frozen SigLIP2 per-frame feature
→ causal temporal adapter over 192-token cache
→ causal projection
→ instance-aware PCEH head
```

注意 adapter 必须移到 cache 之后，避免当前 (T=1) temporal adapter 退化。

### 优化器

```text
AdamW
adapter/projection lr: 1e-4
head lr: 2e-4
weight decay: 0.05
warmup: 5% optimizer steps
cosine decay
15 epochs
```

### 验证

* epoch 3 起每 2 epochs 做一次**完整 chronological validation**；
* 可使用 cache 加速，但不能 sampled validation；
* 最佳 checkpoint 再执行 raw-stream chronological validation；
* 所有 threshold 在 validation 锁定；
* test 不重新调 threshold。

## Stage 2：Selective-Gradient SigLIP2 LoRA

### 训练输入

每个 episode：

* 192-token 历史 context 从当前 LoRA checkpoint 的 feature cache 读取；
* 最后 2–4 个可见 raw frames 重新经 SigLIP2；
* 只有这些 raw frames 保留 visual gradient；
* projection/head 处理完整 causal context。

这与 ETAD 的 selective backprop 和 E2E-LOAD 的 detached history 思路一致，但最终任务与评测仍是 On-TAL。([arXiv][14])

### 配置

```text
supervised bins/epoch: 8,000–12,000
raw trainable frames/episode: 2–4
batch size: 8 episodes
gradient accumulation: 4
effective batch: 32 episodes
LoRA rank: 8
alpha: 16
dropout: 0.05
epochs: 3–5
```

### Cache refresh

LoRA 改变视觉表示后，旧 frozen cache 会 stale。推荐：

* 每个 LoRA epoch 后按视频批量刷新 cache；
* cache manifest 写入 checkpoint SHA；
* checkpoint SHA 不匹配直接拒绝训练；
* 刷新只做 batched forward，不做 backward。

## Stage 3：Full visual tower 只做条件性 pilot

满足以下全部条件后才允许：

* PCEH 相比 endpoint-only 有稳定 AP–latency Pareto 改善；
* LoRA 相比 frozen 有正增益；
* LoRA 更新 audit 全覆盖；
* raw episodic training 达到可接受 throughput；
* 1-epoch full-tower pilot 能在 allocation 内完成。

否则不做 full FT，也不影响 frozen/LoRA 论文路线。

## 为什么该路线保留科学 claim

1. 每个训练 episode 的模型输入严格来自 prefix；
2. temporal context 按 source time 排序；
3. feature cache 只存可见 frame 的独立冻结特征；
4. sampler 使用未来 GT 只决定监督点，不进入模型输入；
5. evaluation 始终是完整 chronological stream；
6. inference 不使用 cache predictions、teacher 或 GT；
7. ledger 不回改。

## 为什么显著降成本

* 152,670 个 packet losses 降为 12k–24k 个监督 bins；
* 多个 bins 合并到一个 episode forward；
* Stage 1 不重复 SigLIP2；
* 避免 152k 次 video open/seek；
* 减少 optimizer step 两到三个数量级；
* Stage 2 只对 2–4 个近期帧保留视觉梯度；
* DDP 可在 event episodes 上直接实施。

## 可成立与不可成立的 claim

### 完成 Stage 1 正式实验后可成立

* strict inference-time online；
* no-future raw reads；
* immutable emission ledger；
* annotation-guided causal-prefix training；
* frozen SigLIP2 online TAD；
* full chronological OnlineAPBudgeted evaluation。

### 完成 Stage 2 后可额外成立

* parameter-efficient visual adaptation；
* LoRA-updated SigLIP2 online TAD。

### 当前及短 pilot 后仍不能成立

* full visual tower end-to-end finetune；
* statistically independent first-emission hazard；
* 对同类重叠动作的完整支持；
* SOTA；
* paper-ready；
* training uses no future information。

---

# 六、实验矩阵

## 1. 必须完成的主矩阵

所有模型最后都运行完整 chronological validation/test。

| ID      | 训练协议          | Visual route                      | Head/decoder   |        决策 cadence | Seeds | 目的                         |
| ------- | ------------- | --------------------------------- | -------------- | ----------------: | ----: | -------------------------- |
| E-F     | event-sampled | frozen SigLIP2 + temporal adapter | endpoint-only  |          8 frames |     3 | 最公平主控制                     |
| P-F     | event-sampled | frozen SigLIP2 + temporal adapter | corrected PCEH |          8 frames |     3 | 验证 PCEH 核心贡献               |
| E-L     | event-sampled | SigLIP2 LoRA                      | endpoint-only  |          8 frames |     3 | 分离 visual adaptation 增益    |
| P-L     | event-sampled | SigLIP2 LoRA                      | corrected PCEH |          8 frames |     3 | 推荐最终模型                     |
| P-Chunk | event-sampled | frozen                            | corrected PCEH |       1536 frames |     3 | 控制 51.2s chunk-end latency |
| P-S2    | event-sampled | frozen                            | corrected PCEH | 8 frames，stride 2 |     3 | 高视觉计算 rolling baseline     |

当前 chunk-end baseline 只在 51.2 秒末决策，不能称低延时。
rolling stride-2 保留 8-frame cadence，但每 packet 编码 4 帧，是明确的高计算对照。

## 2. Full packet vs event-sampled 控制实验

在固定 5–10 个视频或不超过 10,000 packets 的 gold subset 上，做三种子配对：

| 设置                     | 模型初始化 | 训练步数/数据             | 比较内容         |
| ---------------------- | ----- | ------------------- | ------------ |
| Exhaustive full packet | 相同    | 全 packet            | 精确 objective |
| Uniform prefix sample  | 相同    | 与 event sample 相同数量 | 检查仅减少数据的影响   |
| Event-centric + IPW    | 相同    | 同数量                 | 检查采样效率与偏差    |

报告：

* exhaustive loss 与 sampled loss；
* gradient cosine；
* endpoint recall；
* calibration；
* 完整 subset stream OnlineAP；
* wall time；
* encoded frame 数。

这才足以支持“event sampling 没有改变任务，只改变了训练估计器”。

## 3. Frozen、Adapter、LoRA 消融

| 模型             | Trainable modules                                      |
| -------------- | ------------------------------------------------------ |
| Frozen-head    | projection + head                                      |
| Frozen-adapter | cache 后 temporal adapter + projection + head           |
| LoRA-adapter   | top visual LoRA + temporal adapter + projection + head |
| Full tower     | 只做条件性 pilot                                            |

当前 adapter 在 cache 前且每次 (T=1)，因此旧 adapter 结果不能代表真正 temporal adapter 消融。

## 4. PCEH 必须有的机制消融

* 当前 class-aggregated target vs instance-aware risk target；
* invalid late-positive emission loss vs corrected first-stop risk loss；
* emission gate on/off；
* endpoint freeze on/off；
* importance weighting on/off；
* 1 track/class vs multi-slot；
* completion score 是否进入最终 emission confidence；
* emission penalty (\lambda) sweep；
* budget-conditioned policy vs fixed 1s primary policy。

## 5. Latency budget sweep

固定同一 ledger，评估：

```text
OnlineAP@0.5s
OnlineAP@1s
OnlineAP@2s
OnlineAP@4s
```

每个 budget 至少报告：

* tIoU 0.5；
* tIoU 0.3:0.1:0.7 的均值；
* matched GT-end latency p50/p90/p95；
* matched TP count；
* FN count；
* late FP count。

## 6. 必须报告的效率指标

| 类别    | 指标                                                          |
| ----- | ----------------------------------------------------------- |
| 检测质量  | OnlineAP@0.5/1/2/4s；tIoU 0.3–0.7                            |
| 主延时   | `emit_time - matched_GT_end` p50/p90/p95                    |
| 诊断延时  | `emit_time - predicted_end`                                 |
| 覆盖    | matched TP、recall、FN、late FP                                |
| 输出质量  | duplicate emission rate、same-class overlap recall           |
| 因果    | future-read violations、ledger mutation violations           |
| 吞吐    | train packets/s、episodes/s、eval packets/s                   |
| 成本    | hours/epoch、total GPU-hours、peak GPU memory                 |
| 视觉计算  | encoded frames、visual forward frames、visual backward frames |
| 优化    | optimizer steps、trainable parameter count                   |
| Cache | build time、size、refresh time、cache hit、manifest SHA         |
| 数据    | video open 次数、seek 次数、data wait time                        |

## 7. 证据等级门控

| 等级          | 必须满足                                                    | 可对外表述       |
| ----------- | ------------------------------------------------------- | ----------- |
| 启动成功        | import、路径、model load、单步 forward                         | 仅“可运行”      |
| Smoke       | 32–128 episodes、8 raw packets、update/causal/cache audit | 仅“协议通过”     |
| Short pilot | 单种子、3–5 epochs、完整 val stream                            | 仅方向性观察      |
| Formal      | 完整训练、3 paired seeds、锁定协议                                | 可报告实验结论     |
| Paper-ready | 正式矩阵、置信区间、成本、第二数据集或充分泛化证据、代码 artifact                   | 才能写主要 claim |

---

# 七、实现任务清单

## P0：先修科学正确性

| 文件                                                           | 动作                                                   | 必须测试                                            |
| ------------------------------------------------------------ | ---------------------------------------------------- | ----------------------------------------------- |
| `opentad/models/targets/prefix_event_targets.py`             | 改为 instance-aware risk set；删除 late-positive emission | 单实例、多实例、同类重叠、右删失、事件后 mask                       |
| 新增 `opentad/models/targets/discrete_event_hazard_targets.py` | 独立构造 end hazard 与 unique first-stop target           | 一个 risk set 至多一个 event；post-event 全 mask        |
| `opentad/models/dense_heads/prefix_event_emission_head.py`   | 增加 endpoint-frozen、pending-emit、absorbing 状态；多 slot  | delayed emission 时 `pred_end < emit`；ledger 不回改 |
| `opentad/models/detectors/pceh_ontad.py`                     | 支持 `forward_prefix_episode`、burn-in、多个监督位置           | episode-vectorized 与逐 packet eval logits 等价     |
| `opentad/datasets/streaming_raw_frame.py`                    | 将完整 GT 移入独立 supervision namespace                    | backbone kwargs 中不得出现 GT/terminal metadata      |
| `opentad/evaluations/online_budgeted_map.py`                 | 固定 class universe、校验 cache provenance、emission ID    | late FP+FN、重复 FP、prediction-only class、预算不删行    |
| 新增 `tools/analyze_same_class_overlap.py`                     | 统计同视频同类重复与重叠事件                                       | 输出可复现 JSON report                               |

## P1：实现 event-sampled + cache 主路线

| 新增/修改文件                                               | 职责                                                | 单元测试                                     |
| ----------------------------------------------------- | ------------------------------------------------- | ---------------------------------------- |
| `opentad/datasets/streaming_prefix_episode.py`        | 返回 causal context、burn-in、监督 bins                 | 所有 source frame ≤ anchor time            |
| `opentad/datasets/samplers/risk_set_event_sampler.py` | start/end/ongoing/background 分层采样，记录 (\pi_t)      | 固定 seed 可复现；正事件完整覆盖                      |
| `opentad/datasets/cached_stream_features.py`          | 加载带 provenance 的 frame feature                    | 错误 checkpoint/processor hash fail closed |
| `opentad/utils/feature_cache_audit.py`                | 检查 source、排序、revision、raw/cache equivalence       | future source、乱序、hash mismatch 必须失败      |
| `tools/cache_stream_features.py`                      | 按视频批量解码与 SigLIP2 forward                          | cache 数量、frame index、dtype 正确            |
| `tools/audit_cache_equivalence.py`                    | 同一帧 raw 与 cache feature 对照                        | fp32/fp16 容差门                            |
| `tools/benchmark_pceh_training.py`                    | 分解 data、processor、vision、projection、backward 时间   | 生成稳定 JSON schema                         |
| `opentad/models/backbones/online_videomae_adapter.py` | 把 temporal adapter 移到 cache 后，或引入 recurrent state | packet replay 与整段 causal convolution 等价  |

### I/O 改造

新增 `opentad/datasets/video_reader_pool.py`：

* 每 worker 保持 LRU video handle；
* 连续 frame request 批量读取；
* cache build 按视频顺序读取，不对每帧随机 seek；
* worker 结束时显式关闭。

修改 `online_siglip_adapter.py`：

* 加入验证过的 GPU-native resize/normalize fast path；
* 不可盲目跳过 SigLIP2 processor；
* fast path 必须先通过 processor output equivalence test。

## P2：LoRA 与梯度覆盖

| 文件                                            | 职责                                       | 测试                                |
| --------------------------------------------- | ---------------------------------------- | --------------------------------- |
| 新增 `opentad/models/backbones/siglip2_lora.py` | 注入 top-block q/v LoRA，零 target 命中即报错     | target 数量、block index、base freeze |
| `online_videomae_adapter.py`                  | 不得用全局 `no_grad()` 包住可训练 LoRA             | LoRA grad 非零                      |
| 新增 `opentad/utils/gradient_coverage_audit.py` | 按每个 block/module 检查 grad 和 delta         | 缺任一目标 block 都 fail                |
| `optimizer_audit.py`                          | 输出 base/LoRA/adapter/head 参数分组           | 重复、遗漏、frozen-in-optimizer fail    |
| `training_audit.py`                           | 从 prefix-level 提升到 exact module coverage | 所有预期 module 更新                    |

## P3：DDP

新增 `opentad/utils/video_shard_sampler.py`：

* 完整视频归属于单 rank；
* rank 间视频集合 disjoint；
* union 覆盖全部视频；
* rank 内 packet 顺序严格；
* 支持 uneven input join 或按总 packet 数近似平衡。

测试：

```text
test_video_shards_are_disjoint
test_video_shards_cover_dataset
test_no_video_crosses_rank
test_rank_local_packet_order
test_ledger_gather_preserves_stream_identity
```

## P4：配置文件

新增：

```text
configs/causaltad/thumos_pceh_eventsample_frozen.py
configs/causaltad/thumos_endpoint_eventsample_frozen.py
configs/causaltad/thumos_pceh_eventsample_lora.py
configs/causaltad/thumos_endpoint_eventsample_lora.py
configs/causaltad/thumos_pceh_fullpacket_gold.py
configs/causaltad/thumos_pceh_eventsample_distill.py  # optional
```

不要直接新增名为 `full_finetune.py` 的正式配置。先新增：

```text
thumos_pceh_fulltower_one_epoch_diagnostic.py
```

并保持：

```text
formal_training_ready=False
```

直到 full-tower update audit 和 throughput gate 通过。

## Smoke gate

一次有效 smoke 必须同时生成：

```text
target_risk_audit.json
sampler_audit.json
cache_manifest.json
cache_equivalence.json
optimizer_coverage.json
training_update.json
causal_replay.json
ledger_audit.json
throughput.json
gate_summary.json
```

门槛：

* future-read violations = 0；
* target 每实例至多一个 end event、一个 emit event；
* post-event risk count = 0；
* cache hash 全匹配；
* raw/cache feature 在 dtype 容差内；
* adapter/projection/head 或所有 LoRA blocks 更新；
* future suffix 改变 post-cut trace，但不改变 prefix trace；
* no raw prediction shortcut；
* predicted endpoint 与 emit frame 能在 delayed synthetic case 中不同。

## Remote Slurm gate

建议模式：

```text
MODE=cache_smoke
MODE=sampled_smoke
MODE=sampled_throughput
MODE=sampled_pilot
MODE=lora_smoke
MODE=lora_pilot
MODE=fullstream_eval
MODE=fullpacket_gold
```

依赖关系：

```text
cache_smoke
→ sampled_smoke
→ sampled_throughput
→ sampled_pilot
→ fullstream_eval
→ formal seed launch
```

LoRA 必须额外经过：

```text
lora module coverage
→ lora update audit
→ cache refresh audit
→ lora pilot
```

## 失败诊断

| 症状                              | 首查位置                                                             |
| ------------------------------- | ---------------------------------------------------------------- |
| causality replay 失败             | `encoded_source_frames`、cache source、processor batch、state reset |
| cache 与 raw 不一致                 | processor config、resize、normalization、checkpoint SHA、pooling     |
| LoRA 没有更新                       | 外层 `no_grad()`、freeze flag、target module regex、optimizer group   |
| loss 很低但 OnlineAP≈0             | track assignment、start/end frame 单位、label map、endpoint freeze    |
| emission 全无                     | end risk mask、completion threshold、pending-emit state            |
| emission 爆炸                     | late-positive target 是否残留、旧实例 completion/late 污染、rearm           |
| predicted-end latency 恒 0       | decoder 是否仍把 end_frame 设为 emit_frame                             |
| event-sampled 与 full packet 差异大 | inclusion probability、IPW、burn-in、背景采样、gradient cosine           |
| GPU 利用率低                        | VideoCapture、processor CPU roundtrip、batch size、optimizer step 数 |
| DDP hang                        | rank 步数不一致、视频跨 rank、join context、ledger gather                   |
| p50 latency 很低但 AP 下降           | matched TP coverage、FN、只检测容易事件                                   |

---

# 八、审稿风险清单

| 风险                                     |    严重度 | 审稿人攻击方式                            | 必须证据                                                |
| -------------------------------------- | -----: | ---------------------------------- | --------------------------------------------------- |
| PCEH 不是合法 hazard                       | **致命** | end/emit 同标签、late 持续正、无 risk set   | instance-aware survival target 和 target audit       |
| 只是 endpoint head 加三个 gate              | **致命** | 参数拆分不等于方法贡献                        | endpoint-only 公平对照和 AP–latency Pareto               |
| 一类一 track                              |  **高** | 无法处理同类重复、并发动作                      | 多 slot 或清晰覆盖分析                                      |
| predicted end 等于 emit                  |  **高** | 没有真正边界—发射解耦                        | endpoint snapshot 与 delayed emission                |
| annotation future 被包装成 causal training |  **高** | event sampling 偷看 GT endpoint      | 明确“privileged sampler、causal inputs”                |
| Feature cache 被包装成 E2E                 |  **高** | 视觉塔未更新                             | frozen、LoRA、full FT claim 分表                        |
| LoRA 实际没更新                             |  **高** | outer `no_grad` 阻断                 | 每 block gradient/delta audit                        |
| 与 MATR/HAT/ActionSwitch 重合             |  **高** | memory、state、online span 已存在       | 风险集 emission、预算化评测、效率训练的独立证据                        |
| 与 OnPoint 重合                           |     中高 | offline-to-online distillation 已有  | teacher 只做 ablation，不作核心贡献                          |
| 与 OZ-TAL 重合                            |     中高 | VLM 和 strict online open-vocab 已出现 | 不把“用了 VLM”当 novelty；聚焦 closed-set hazard/latency    |
| OnlineAPBudgeted 非标准                   |     中高 | 自定义指标偏向本方法                         | 公开 evaluator、late FP 单测、同时报标准 tIoU 和完整 counts       |
| 只报 TP latency                          |      高 | 通过少发射作弊                            | latency + matched count + recall + FN               |
| sampled training 改变 base rate          |      高 | endpoint oversampling导致校准虚高        | (\pi_t)、IPW、fullpacket gold subset                  |
| 计算比较不公平                                |      高 | cache、不同帧数、不同 optimizer steps      | encoded frames、visual backward frames、GPU-hours 全账本 |
| THUMOS14 单数据集                          |     中高 | 过拟合小数据集                            | paper-ready 阶段增加第二数据集或强泛化证据                         |
| 单种子 pilot 被当结果                         | **致命** | 证据等级混淆                             | 三 paired seeds、均值/方差/置信区间                           |

## 反驳策略的核心不是文字，而是四份 artifact

1. **causal evidence package**：future perturbation、source provenance、ledger audit；
2. **objective evidence package**：risk-set statistics、采样概率、full-vs-sampled gradient；
3. **training evidence package**：每模块 update、视觉帧数、GPU-hours；
4. **evaluation evidence package**：完整 chronological ledger、GT-matched latency、late FP/FN。

---

# 九、最终裁决：HOLD

## 当前总体裁决

**HOLD：不允许直接启动 formal PCEH multi-seed training，也不允许形成 paper claim。**

原因不是在线协议骨架完全错误，而是：

1. 当前 endpoint 与 emission 没有真正的目标解耦；
2. class-level target 会让旧实例的 completion/late 状态污染后续同类实例；
3. current decoder 令 predicted end 等于 emit；
4. current adapter 不建模跨 packet temporal context；
5. 视觉塔冻结，full finetune config 与证据不存在；
6. 当前只有 frozen/adaptor smoke，没有完成 short pilot；
7. full packet 训练成本不适合作为默认路线。

## 立即允许的工作

**GO：只允许进入以下重构和 Gate 1–3。**

```text
instance-aware risk targets
→ endpoint-frozen state machine
→ event-centric causal episodes
→ frozen SigLIP2 feature cache
→ full chronological short pilot
→ LoRA update smoke
```

## 晋级 formal training 的最低条件

必须同时满足：

* 多实例/同类 target tests 全通过；
* late 不再作为重复 emission positive；
* delayed synthetic case 中 `predicted_end < emit`；
* sampled objective 在 gold subset 上接近 exhaustive；
* PCEH 单种子 short pilot 至少表现出相对 endpoint-only 的方向性 AP–latency优势；
* full chronological validation 完成；
* causality/ledger violations 为 0；
* frozen Stage 1 达到可接受 throughput；
* LoRA 每个目标 block 更新 audit 通过。

## 最终 NO-GO 条件

完成修正后，如出现以下任一情况，应停止 PCEH 主张，而不是继续包装：

* 三 paired seeds 下，PCEH 对 endpoint-only 没有稳定 AP–latency Pareto 改善；
* emission policy 只学习到 endpoint 的复制；
* event-sampled 与 full-packet gold objective 始终严重偏离；
* 同类多实例支持无法解决；
* LoRA 经确认真实更新后仍无增益，则停止 visual adaptation claim，退回 frozen 主模型；
* 低延时优势仅来自少发射、recall 显著下降。

所以最终不是 NO-GO：**严格在线 packet、provenance、immutable ledger 和 GT-end latency evaluator 是可信的研究基础。**
但当前 PCEH 方法本身仍处于 **“smoke-passed implementation candidate”**，距离 **short pilot、formal result 和 paper-ready claim** 还有清晰且不可跳过的门槛。

[1]: https://arxiv.org/abs/2107.03377 "https://arxiv.org/abs/2107.03377"
[2]: https://arxiv.org/abs/2106.11149 "https://arxiv.org/abs/2106.11149"
[3]: https://arxiv.org/abs/2209.09236 "https://arxiv.org/abs/2209.09236"
[4]: https://arxiv.org/abs/2211.04905 "https://arxiv.org/abs/2211.04905"
[5]: https://arxiv.org/abs/2306.07703 "https://arxiv.org/abs/2306.07703"
[6]: https://arxiv.org/abs/2408.02957 "https://arxiv.org/abs/2408.02957"
[7]: https://arxiv.org/abs/2408.06437 "https://arxiv.org/abs/2408.06437"
[8]: https://arxiv.org/abs/2407.12987 "https://arxiv.org/abs/2407.12987"
[9]: https://arxiv.org/abs/2503.18359 "https://arxiv.org/abs/2503.18359"
[10]: https://arxiv.org/abs/2507.16342 "https://arxiv.org/abs/2507.16342"
[11]: https://arxiv.org/abs/2607.00289 "https://arxiv.org/abs/2607.00289"
[12]: https://arxiv.org/abs/2605.09976 "https://arxiv.org/abs/2605.09976"
[13]: https://arxiv.org/abs/2407.17792 "https://arxiv.org/abs/2407.17792"
[14]: https://arxiv.org/abs/2205.07134 "https://arxiv.org/abs/2205.07134"
[15]: https://arxiv.org/abs/2204.01680 "https://arxiv.org/abs/2204.01680"
[16]: https://arxiv.org/abs/2206.13559 "https://arxiv.org/abs/2206.13559"
[17]: https://huggingface.co/docs/peft/en/package_reference/lora "https://huggingface.co/docs/peft/en/package_reference/lora"
[18]: https://arxiv.org/abs/1805.00917 "https://arxiv.org/abs/1805.00917"
