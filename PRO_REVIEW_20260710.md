# 审查基线与总裁决

本次审查固定在分支 `codex/online-tad-clean-20260702`、可见 HEAD `4e222ccf4de1e989c6faf83031772c552be7af59`。仓库没有附带可核验的正式 checkpoint、训练日志、完整预测文件或多 seed 结果，因此以下结论能够裁决**代码与协议是否支持 claim**，不能替作者补出尚不存在的性能证据。仓库 README 自己也明确将 raw-frame 路线定义为 validation candidate，并禁止声称 paper-ready、DDP 可审计或视觉塔微调完成。([GitHub][1])

**总裁决：**

* 对“当前仓库已经实现低延时、adaptive、end-to-end、MATR-style Online TAD”这一组 claim：**NO-GO**。
* 对“当前仓库是一个 raw-frame、selected-only heavy encoding、因果时序头和 emission ledger 的工程骨架”：**成立，但贡献级别很低**。
* 对“把研究问题重构为 prefix-censored event-time learning + bounded-delay emission hazard”：**HOLD，有条件推进**。
* 当前系统最准确的命名不是“low-latency Online TAD”，而是：

> **Non-overlapping chunk-end causal TAD with fixed-stride selected-only visual encoding**

即：**非重叠长窗口、窗末统一发射、固定因果步长选帧的 causalized TAD skeleton**。

---

# A. Task Definition Verdict

## A.1 当前究竟在做什么

| 候选任务                       | 是否符合 | 裁决                                                       |
| -------------------------- | ---: | -------------------------------------------------------- |
| OAD：逐帧/逐 snippet 动作类别判断    |    否 | 当前输出时间段而非单纯逐帧类别                                          |
| 严格 Online TAL / Online TAD |   部分 | 输出 `{start,end,class,score}` 并有不可未来结束约束，但发射粒度和全链路因果证明不合格 |
| Rolling-window On-TAL      |    否 | 当前不是每个新 snippet 推进一次并更新输出                                |
| Window-end prediction      |    是 | 每个 192-token 样本只在 `window_end_frame` 调用 emitter          |
| 离线 TAD 的滑窗近似               | 高度接近 | 虽然局部算子因果，但整体调度是长块输入、窗末检测                                 |

OAD 在文献中通常是流式视频上的逐帧分类；On-TAL 则要求输出动作实例并且已输出结果不可被未来修改。MATR 和 OAT 均采用后者定义。([欧洲计算机视觉协会][2])

## A.2 建议采用的严格 On-TAD 定义

设系统在时刻 (t_k) 获得新视频包 (x_k)，已观测前缀为 (V_{\le t_k})。严格 On-TAD 至少应满足：

1. **前缀因果性**

[
V_{\le t}^{(1)}=V_{\le t}^{(2)}
\Longrightarrow
S_{\le t}^{(1)}=S_{\le t}^{(2)},\quad
D_{\le t}^{(1)}=D_{\le t}^{(2)}
]

即两个视频在 (t) 之前完全相同，则系统在 (t) 之前的内部状态和输出必须完全相同。

2. **真实读取约束**

对第 (i) 个 detection：

[
r_i=\max{\text{所有参与该预测的原始帧索引}}\le a_i
]

其中 (a_i) 是 emit time，(r_i) 必须来自真实数据流 provenance，而不能由预测终点反推。

3. **确认式检测约束**

若论文讨论的是 completed action detection，而不是 anticipation：

[
\hat e_i\le a_i
]

其中 (\hat e_i) 是模型估计的 event end time。

4. **不可修订**

一旦 detection 在 (a_i) 发出，后续不能修改、删除或替换其边界、类别和分数。

5. **有界延时**

匹配到 GT 实例 (g) 后：

[
0\le a_i-e_g\le B
]

延时必须相对 **GT event end** 计算，而不能只相对模型自己的 (\hat e_i) 计算。

6. **全在线后处理**

排序、抑制、阈值决策只能使用当前及历史 proposal；不能在整段视频结束后统一执行 NMS、重打分或边界合并。

**窗口模型本身不违反 online。** 问题在于是否做到“长上下文、小步推进”。一个 51.2 秒 rolling cache 可以每 0.267 秒发射一次；当前代码却把 51.2 秒上下文长度同时变成了约 51.2 秒的决策周期。

## A.3 当前延时是否还能称为 low latency

配置为：

* `fps=30`
* `snippet_stride_frames=8`
* `window_size=192`
* `window_overlap_ratio=0.0`

因此：

[
192\times 8/30 = 51.2\text{ seconds}
]

数据集把 `window_end_frame` 设置为窗口起点加整个窗口长度；streaming-safe 分支在这个 `window_end_frame` 上只调用一次 emitter。([GitHub][3])

对于窗内动作：

* 最坏调度等待接近 **51.2 秒**；
* 若动作终点在窗口内近似均匀分布，纯调度等待均值约 **25.6 秒**；
* 同一假设下 p95 约 **48.64 秒**。

这不是测出来的模型延时，而是由调度方式直接推导的等待时间。它已经足以否定 **low-latency On-TAD** claim。

`max_latency_frames=192*8=1536` 也不是低延时保证。它只允许 emitter 接受预测终点位于当前窗口内的 proposal，即允许约 51.2 秒的预测内延时。([GitHub][4])

## A.4 测试时泄漏判断

### 目前可确认没有的 shortcut

* P0 配置显式设置 `load_from_raw_predictions=False`；
* raw-frame 数据管线直接加载视频帧；
* streaming-safe 分支在常规 `batched_nms` 之前 `continue`，实际走 `OnlineEmitter` 和 prefix suppression，而不是整视频离线 NMS。([GitHub][3])

### 仍然存在或尚未证明不存在的问题

1. **真实 source provenance 未建立。**
   缺少显式 source 信息时，代码使用 proposal/source grid 转换为 `source_frame`；而 fallback source grid 又可来自预测段终点。这只证明“声明的 source 不晚于 emit”，不能证明视觉塔、缓存、loader 实际没有读取未来。([GitHub][5])

2. **whole-video duration metadata 进入测试。**
   `FrameWindowDataset` 将总帧数和整视频 `duration` 写入 metadata；streaming 输出又用 `duration` 裁剪终点。对于 prerecorded-video protocol 这可能被允许，但对 live-stream claim 属于未来视频长度信息。([GitHub][6])

3. **不存在反事实 no-future 测试。**
   真正有说服力的测试是：改变 (t) 之后的帧，所有 (t) 之前输出和状态必须逐值不变。ledger 字段比较代替不了该测试。

4. **DDP 状态未闭环。**
   README 已明确在线状态会被 DDP 按 rank 拆分，当前只允许单卡审计。([GitHub][1])

## A.5 是否 raw-frame end-to-end trainable

准确说法是：

* **raw-frame input：是；**
* **selected-only heavy visual encoding：是；**
* **完整视觉预训练模型参与微调：否；**
* **adapter/head 是否真的更新：尚未由运行证据证明。**

P0/P1/P2 均保持 SigLIP2 vision tower frozen；编码函数在 `freeze_vision_encoder=True` 时进入 `torch.no_grad()`。P1 声称训练 causal adapter、projection、FPN 和 head，P2 增加 motion branch，但不是 full visual-tower fine-tuning。([GitHub][3])

更严重的是，`VideoMambaSuite.get_optim_groups()` 明确跳过所有名称以 `backbone` 开头的参数。如果实际 optimizer builder 使用该路径，P2 配置中宣称可训练的 `backbone.backbone.motion_branch`、`backbone.adapter` 等可能根本不会进入 optimizer。没有参数名清单、梯度范数和更新前后 checksum 前，**PEFT end-to-end claim 仍然 blocked**。([GitHub][7])

---

# B. Current Repo Critique

## B.1 文件级审查

| 位置                                            | 代码事实                                                                                   | 严厉结论                                                           |
| --------------------------------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `thumos_siglip2_matr_ontad_p0.py`             | 192 snippets、8-frame stride、零重叠、frozen SigLIP2、`memory_size=0`、普通 mAP evaluator        | 这是 51.2 秒 chunk baseline，不是 MATR，也不是低延时 rolling On-TAL         |
| `thumos_siglip2_adaptive_matr_ontad_final.py` | metadata 写 `adaptive_selected_frames`，实际 selector 是 `policy="causal_stride", stride=2` | “adaptive” 与实现直接矛盾                                             |
| `causal_frame_selector.py`                    | 仅支持固定 stride 和像素均值差 motion heuristic；无可学习参数；motion 路径使用 `.item()`                      | 不可称 learned selector、adaptive policy 或 end-to-end acquisition  |
| `online_siglip_adapter.py`                    | selector 在视觉塔前执行，视觉塔只编码 packed selected frames                                         | selected-only heavy encoding 成立                                |
| 同上                                            | 无论实际策略是什么，都把 `meta["frame_policy"]` 改为 `"adaptive_selected_frames"`                    | audit metadata 不可信，会掩盖固定 stride                                |
| 同上                                            | 视觉塔 frozen 时 `torch.no_grad()`                                                         | 不能称 pretrained visual encoder 被 On-TAD task end-to-end adapted |
| `matr_head.py`                                | 在线分支只在 endpoint 未观测时约束 `end_mask`；emit target 等于 end mask                              | 局部 endpoint masking，不是完整 prefix-censored event-time learning   |
| 同上                                            | actionness 仍由完整 `[start,end]` 区间构造；主 cls/reg 保留 anchor-free target contract            | pre-end target 并未系统性重定义为 prefix-observed target                |
| `raw_frame.py`                                | 一个样本含完整 192-token window，并写入 window end、duration、total frames                          | loader 层是 window sample，不是持续 step-by-step stream               |
| `single_stage.py`                             | `emit_frame=window_end_frame`，每个样本调用一次 emitter                                         | 发射时刻由数据窗口结束决定，不是 emission head 学出来的                            |
| `online_protocol.py`                          | `latency_sec=(emit_frame-predicted_end_frame)/fps`                                     | 这是“相对预测终点”的自引用延时，不是真实响应延时                                      |
| `online_map.py`                               | 继承普通 mAP；达到 `max_latency_sec` 的行在计算 mAP 前直接删除                                          | 晚预测可能被静默删除，而不是作为 late FP / missed GT 惩罚                        |
| `mamba.py`                                    | optimizer grouping 跳过 `backbone.*`                                                     | 训练范围可能与配置宣称不一致                                                 |

最终配置明确把 `frame_policy` 标为 adaptive，但 selector 实际是固定 `causal_stride`；selector 实现也只有固定模运算和不可微的像素差 heuristic。([GitHub][4])

`online_siglip_adapter.py` 的 selected-only 路径是真实存在的：只有打包后的 selected frames 被送入视觉编码器。但是原始 192 帧窗口已经由 loader 解码并进入 selector，因此目前最多证明**降低视觉编码 FLOPs**，不能证明降低视频解码、I/O 或端到端 latency。该文件还无条件将 metadata 改成 adaptive，构成明显的 provenance 污染。([GitHub][8])

`_build_online_branch_targets()` 中，所谓 censoring 仅把 end target 限制为 endpoint 已观测的位置；emit target 与 end target 完全相同。它没有 survival likelihood、没有“尚未结束”的截尾似然、没有首次发射时刻，也没有延时预算损失。([GitHub][9])

更严重的是，当前 `OnlineEmitter.step()` 计算：

[
\text{reported latency} =
\frac{\text{emit frame}-\text{predicted end frame}}{\text{fps}}
]

模型只要把预测终点向 emit 时刻移动，就能得到更小的“延时”，哪怕真实边界更差。因此这个字段只能称 **prediction staleness**，不能称 response latency。([GitHub][5])

`OnlineMAP` 本质上是普通 mAP 加 ledger summary；若设置 `max_latency_sec`，超时行会在 mAP 前被过滤掉。这样 late detection 不一定作为错误进入评估，可能产生“删除差预测反而提高 mAP”的指标漏洞。([GitHub][10])

## B.2 Safe claims

下列主张在限定措辞下可以保留：

1. **Raw-frame route exists.**
2. **Heavy visual encoding can be selected-only.**
3. **Final selector is a causal fixed-stride selector.**
4. **SigLIP2 vision tower is frozen.**
5. **Temporal projection配置为 causal。**
6. **Streaming-safe branch 不走常规离线 `batched_nms`。**
7. **Ledger 能检查声明的 `end_frame/source_frame <= emit_frame`。**
8. **配置禁用了 raw-prediction shortcut。**

## B.3 Blocked claims

以下主张当前不能写进论文：

1. **Learned/adaptive frame selection**
2. **Low-latency Online TAD**
3. **Snippet-level rolling inference**
4. **Complete prefix-observed censored training**
5. **Bounded-delay emission learning**
6. **Full MATR reproduction or MATR-style memory**
7. **Visual backbone fine-tuning**
8. **Full-chain no-future proof**
9. **DDP-auditable online evaluation**
10. **End-to-end compute reduction**
11. **Paper-ready/SOTA On-TAD**

## B.4 Overclaim risks

| 当前命名/叙述                    | 实际含义                                                             | 风险                       |
| -------------------------- | ---------------------------------------------------------------- | ------------------------ |
| `adaptive_selected_frames` | fixed stride-2                                                   | 高，属于实现与 metadata 不一致     |
| `MATRHead`                 | anchor-free head 加若干 start/end/actionness/emit branches，memory=0 | 高，容易被认为借用论文名称            |
| `online_censored_training` | 局部 endpoint target masking                                       | 高，不是 censored likelihood |
| `emit_head`                | endpoint-local BCE score branch                                  | 高，未决定何时 emit             |
| `OnlineMAP`                | 普通 mAP + ledger，且可删除 late rows                                   | 高，指标名夸大                  |
| raw-frame end-to-end       | raw frames 到 trainable head，但视觉塔 frozen                          | 中高                       |
| selected-only compute      | 仅 selected-only visual encoding，整窗仍被解码                           | 中高                       |
| no-future proof            | 检查声明字段，而非实际读取 lineage                                            | 高                        |
| low latency                | 最长允许约 51.2 秒                                                     | 致命                       |

---

# C. Unclear Questions List

以下问题无法仅凭当前仓库回答，必须通过运行证据解决：

1. 实际训练入口是否调用 `VideoMambaSuite.get_optim_groups()`？若是，哪些 adapter/motion 参数被漏掉？
2. 一个 optimizer step 后，所有宣称 trainable 的参数是否有非零 gradient 和非零 parameter delta？
3. val/test dataloader 是否严格按同一视频时间顺序产生窗口？多 worker 会不会乱序？
4. `stream_key` 在不同 worker、batch、rank 之间是否稳定且唯一？
5. 不同窗口间的 model cache 是否真的保留？当前 `memory_size=0` 时究竟保留什么长期状态？
6. 跨窗口动作的 GT 如何处理？是否被截断、复制、过滤或重新分配？
7. `source_frame` 是否存在来自 loader/encoder 的真实读帧 lineage，还是全部由 proposal grid 生成？
8. `ext_cls` 路径是否可能使用全视频外部分类结果？
9. tail window、短视频和不满 192-token 窗口的 emit time 如何定义？
10. 阈值、top-k、prefix NMS 参数是否在 test set 上调过？
11. 报告的延时到底是相对预测终点还是匹配 GT 终点？
12. 视频解码时间、图像预处理时间和 selected-only packing 开销是否进入 latency/FPS？
13. 是否存在未来帧扰动测试、cache contamination 测试和 target invariance 测试？
14. 是否存在真实 full-run checkpoint、每 epoch mAP、ledger 文件和多 seed variance？
15. `duration`、`total_frames` 是否会在 live-stream protocol 中被移除？

在这些问题关闭前，任何“已验证 end-to-end online”表述都应标为 **unproven**。

---

# D. Literature Positioning

## D.1 现有工作已经占据的空间

| 方法                               | 已经解决/占据的主张                                                                                      | 对本项目的约束                                              |
| -------------------------------- | ----------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| OAT                              | sliding-window On-TAL、anchor-based segment prediction、online suppression、early detection metric | “我们把离线 TAL 变成 sliding window online”不新               |
| MATR                             | memory queue；当前 segment 检测 end；从历史 memory 检索 start；输出不可修改                                       | “current end + past start + causal memory”不新         |
| PKD                              | 训练时 offline teacher 使用未来帧，在线 student 仅看前缀                                                       | “future privileged teacher”不新                        |
| OnPoint                          | pseudo-segment instance、CAS、anticipatory window 三层 offline-to-online distillation；POTAL         | “offline TAL teacher 多层蒸馏到 online TAL student”已被直接占据 |
| WACV 2026 streaming distillation | offline ViT teacher、causal cached student、logit/attention/RoI feature distillation              | “把 offline ViT 蒸馏成 cacheable streaming student”不新    |
| AdaTAD                           | raw-frame offline TAD；冻结大 backbone，训练 temporal-informative adapters 解决 task mismatch            | “raw frames + adapter tuning for TAD”不新              |

OAT 已在 2022 年明确使用 sliding window、在线抑制并讨论提前检测；因此 sliding-window On-TAL 本身不是贡献。([欧洲计算机视觉协会][11])

MATR 的核心是选择性 memory queue、当前 segment 预测动作终点、历史 memory 检索动作起点，并明确要求既往预测不可修改。当前仓库 `memory_size=0`，也没有其双 decoder 和 memory retrieval，不能以 MATR 作为当前方法身份。([欧洲计算机视觉协会][2])

PKD 已经把训练时未来帧定义为 privileged information，由 offline teacher 蒸馏给只能看过去和当前帧的 OAD student。([arXiv][12])

截至 2026 年 7 月，OnPoint 已进一步将 offline-to-online 多层蒸馏用于 point-supervised Online TAL，包括 pseudo-segment instance distillation、CAS distillation 和 anticipatory window-level distillation，并被标注为 ECCV 2026 accepted。([arXiv][13])

WACV 2026 的 streaming action detection 工作已经提出 causal cached student，并从 offline ViT teacher 蒸馏 logits、attention 和 RoI features。它研究的是 spatio-temporal action detection 而非 segment-level On-TAL，但“offline ViT → cached causal student”这个架构级叙述已经不空白。([CVF开放获取][14])

AdaTAD 则已经通过 temporal-informative adapter 在冻结大 backbone 的条件下进行 raw-frame end-to-end TAD adaptation。因此 PEFT 本身只能是系统手段，不能是 On-TAD 核心创新。([arXiv][15])

## D.2 为什么“Online TAD + causal + distillation”不新

因为这句话分别被已有工作拆解覆盖：

* Online segment localization：OAT、MATR；
* causal/current-past only：整个 OAD/On-TAL 文献；
* future privileged teacher：PKD；
* TAL-level offline-to-online distillation：OnPoint；
* cached causal visual student：WACV 2026；
* raw-frame adapter tuning：AdaTAD。

简单把这些部件拼在一起，只会得到一个工程组合。审稿人会问：

> 你的新问题、目标函数或统计假设是什么？为什么不是 OAT/MATR 加常规 KD 和 adapter？

目前最可能形成方法贡献的空隙不是 causal backbone，而是：

> **把物理事件时刻、前缀截尾状态和系统发射时刻作为三个不同变量进行统一建模。**

我没有在上述定向检索中发现与下文 PCEH 完全同构的 On-TAL 方法，但这只能说明它具有潜在新意，**不能在未做完整系统综述前声称“first”**。

---

# E. Claim Map

| Claim                           | Needed method component                           | Current repo support                           | Missing implementation                                            | Baseline needed                           | Ablation needed                           | Metric needed                              | Failure criterion                             |
| ------------------------------- | ------------------------------------------------- | ---------------------------------------------- | ----------------------------------------------------------------- | ----------------------------------------- | ----------------------------------------- | ------------------------------------------ | --------------------------------------------- |
| Strict no-future On-TAD         | 真实 read provenance、prefix equivalence、不可修订状态      | 只有 ledger 字段检查                                 | loader/backbone/cache 读取 lineage；future perturbation test         | OAT/MATR-style causal baseline            | 去掉 provenance audit                       | violation count；prefix output diff         | 任意未来扰动改变过去输出，或实际读取帧晚于 emit                    |
| Low-latency On-TAD              | 小步 rolling update、显式 (B)                          | 当前每 51.2 秒发射一次                                 | per-snippet `forward_step`、cache update                           | current chunk-end；rolling-window baseline | cadence 1/2/4/8 snippets                  | OnlineAP@0.5/1/2/4s；p95 latency            | p95 超预算，或 OnlineAP@1s 无法超过简单 rolling baseline |
| Prefix-censored localization    | right-censored likelihood；prefix-local assignment | 仅局部 end mask                                   | censor-aware target builder、survival loss、target invariance tests | ordinary regression masking               | remove censoring；endpoint BCE             | hazard NLL；boundary drift；missed endpoint  | 改变不可见未来终点会改变当前 target                         |
| Learned emission                | first-emission hazard、early/late cost             | emit target=end mask；发射仍由窗末驱动                  | emission state、first-event loss、吸收态                               | threshold/post-filter baseline            | remove hazard；replace with endpoint score | OnlineAP@B；false early；late miss           | hazard 不优于固定阈值/后处理                            |
| Long context without long delay | incremental cache；context与cadence解耦               | 192-token整窗重算                                  | per-step cacheable temporal encoder                               | 16/32/64/192 rolling contexts             | cache size与cadence交叉                      | mAP/OnlineAP、step latency、memory           | 长 context 只能通过更长发射周期获得                        |
| Raw-frame PEFT On-TAD           | 明确 trainable adapter、optimizer更新证据                | raw input；vision frozen；配置宣称 adapter trainable | optimizer修复、grad/parameter delta audit                            | frozen encoder；full/partial tune          | freeze vs PEFT                            | OnlineAP、mAP、VRAM、train time               | 宣称参数零梯度或零更新                                   |
| Adaptive evidence acquisition   | 可学习 content-dependent policy                      | 无；实际 fixed stride                              | learned policy、预算约束、因果输入                                          | uniform/random/motion same budget         | policy loss、utility、budget                | OnlineAP-vs-encoded-frames Pareto          | 不优于 uniform 同预算，或端到端 wall-clock不降             |
| Computational efficiency        | 避免重编码并包含解码/I/O                                    | 只减少视觉编码帧数                                      | cache、decode-inclusive profiler                                   | dense rolling、fixed stride                | selected-only vs full pipeline            | decoded/encoded frames、FPS、VRAM、wall-clock | 只降理论 FLOPs，不降实际时间/内存                          |
| Novelty beyond prior art        | 新统计问题与目标，而非部件拼装                                   | 当前没有                                           | event-time/emission-time统一建模                                      | MATR、OAT、PKD、OnPoint、vanilla KD           | 去掉核心目标的独立贡献                               | 多数据集结果和统计显著性                               | 核心增益可由普通 KD/causal head复现                     |

---

# F. Candidate Routes and Scores

## Route 1 — Minimal Route: Prefix-Observed Event-Time Learning

**One-sentence thesis**

> On-TAD 应被建模为“前缀条件下的右截尾动作终点事件 + 有界延时首次发射决策”，而不是离线边界回归加窗末过滤。

**System graph**

```text
new raw-frame packet x_t
→ cacheable visual PEFT adapter
→ current token z_t
→ causal temporal cache H_t
→ class / start-memory / completion / end-hazard heads
→ bounded-delay emission-hazard head
→ immutable online emitter
→ provenance ledger
```

**Trainable modules**

* visual PEFT adapter；基础视觉塔先冻结；
* causal temporal adapter/cache；
* class/start/completion/end-hazard heads；
* emission-hazard head。

**Losses**

* prefix-valid classification；
* start retrieval/localization；
* censored end-event survival loss；
* completion state loss；
* bounded-delay first-emission loss；
* calibration loss。

**Inference**

每 8 个新帧更新一次，只编码新帧，保留历史 cache；首次满足 emission policy 时发出并冻结结果。

**Novelty argument**

新意位于“右截尾 event-time + 独立 emission-time hazard”的任务建模，而不是 causal backbone、memory 或 KD。

**Feasibility**

最高。可以复用现有 raw-frame、causal projection、ledger 和大部分 detector plumbing，但必须重写数据调度、target builder、head 和 evaluator。

**Overclaim risk**

中等。必须证明它不是普通 endpoint BCE、阈值后处理或 survival-analysis 名词替换。

---

## Route 2 — OnlineTAD-PT: Task-Aligned Causal Video Pretraining

**One-sentence thesis**

> 在大规模动作段数据上，以随机前缀构造 start/ongoing/completion/end/emission 预训练任务，再对 On-TAD 数据进行 PEFT。

**System graph**

```text
large segment-labeled raw videos
→ random causal prefixes
→ cacheable video backbone
→ event-state pretraining objectives
→ THUMOS/MUSES PEFT
→ bounded-delay On-TAD inference
```

**Trainable modules**

* causal video backbone adapters；
* event-state heads；
* downstream detection heads。

**Losses**

Route 1 全部损失，加跨视频/跨数据集预训练目标、masked causal state prediction、boundary completion calibration。

**Inference**

与 Route 1 相同。

**Novelty argument**

预训练目标专门对齐 On-TAD 的 causal boundary、completion 和 emission，而不是分类或离线 TAD。

**Feasibility**

偏低。需要大量统一格式的 segment annotations、较大算力、跨数据集时间标度对齐，而且“人工定义 emission target”可能不具备自然预训练语义。

**Overclaim risk**

高。若主要收益来自更多数据或更大模型，核心 objective 很难被证明。

---

## Route 3 — Offline-Teacher Assisted Cached Student

**One-sentence thesis**

> 用完整视频 offline teacher 为 causal cached student 提供边界 posterior、状态轨迹和表示监督。

**System graph**

```text
offline raw-video teacher
        ↓ posterior/feature targets
causal cached raw-frame student
→ event heads
→ emission policy
```

**Trainable modules**

student adapters、cache、heads；teacher frozen。

**Losses**

监督检测损失 + logit/feature/CAS/boundary trajectory KD + Route 1 的 censor/emission losses。

**Inference**

teacher-free、per-snippet cached student。

**Novelty argument**

只能尝试声称“蒸馏的是 censored event posterior 或 emission utility”，不能声称 offline-to-online KD 本身新。

**Feasibility**

中高。

**Overclaim risk**

极高。PKD、OnPoint 和 WACV 2026 已覆盖相邻空间，审稿人极可能将其判为 incremental distillation。

---

## 严厉评分

分数越高越好；“风险”分数越高越危险。

| Route   | 问题定义价值 | 潜在新颖性 | 方法可证伪性 | 工程可行性 | 算力可控性 | Overclaim risk | 综合裁决             |
| ------- | -----: | ----: | -----: | ----: | ----: | -------------: | ---------------- |
| Route 1 |    9.0 |   7.5 |    9.0 |   8.5 |   8.0 |            4.0 | **唯一推荐主路线**      |
| Route 2 |    9.0 |   8.5 |    7.0 |   4.5 |   3.5 |            7.0 | 后续扩展，不应首篇主线      |
| Route 3 |    7.0 |   4.0 |    6.5 |   7.5 |   5.5 |            9.0 | 仅作 baseline/附加实验 |

---

# G. Recommended Method

## G.1 唯一推荐路线

推荐暂定名：

> **PCEH-OnTAD: Prefix-Censored Event–Emission Hazard Learning for Online Temporal Action Detection**

该名称只是方法设计标签，**不是已经获得文献首创证明的论文标题**。

核心贡献只能聚焦一件事：

> **把动作物理终点 (E) 和模型发射时刻 (A) 分开，并在只观察前缀的条件下同时学习右截尾终点分布和有界延时首次发射策略。**

不要把 selector、teacher、Mamba、causal attention、adapter 同时塞进贡献列表。它们是实现部件或 baseline。

## G.2 输入、输出和时间变量

### 输入

主方法输入为按时间到达的 raw-frame packets：

* 每个 packet 含新到达的 8 帧；
* 30 fps 下更新周期约 0.267 秒；
* 不使用预提特征作为主结果；
* 预提特征只用于快速方法消融和 MATR/OAT 对照。

### 输出

每个 confirmed detection 必须包含：

[
(\hat s,\hat e,\hat c,\hat p,a,r)
]

其中：

* (\hat s)：估计动作开始 event time；
* (\hat e)：估计动作结束 event time；
* (\hat c)：类别；
* (\hat p)：置信度；
* (a)：实际 emit time；
* (r)：该预测真实读取到的最大 source frame。

### 在线约束

[
r\le a,\qquad \hat e\le a
]

与 GT 匹配后：

[
0\le a-e_{\mathrm{gt}}\le B
]

主预算建议固定为 **1 秒**，并报告：

[
B\in{0.5,1,2,4}\text{ seconds}
]

不要选择 51.2 秒作为“延时预算”。51.2 秒可以是 cache horizon，而不能是 low-latency budget。

## G.3 长上下文与低延时解耦

保留当前 192-token 上下文容量：

[
H=192\times 8/30=51.2\text{ s}
]

但系统每个 token 更新一次：

[
\Delta=8/30\approx0.267\text{ s}
]

因此：

```text
context horizon H = 51.2 s
decision cadence Δ = 0.267 s
latency budget B = 1.0 s
```

三者必须是独立超参数。

每次只编码新 packet，历史视觉/时序状态进入 ring cache；不得重新编码完整 192-token 窗口。

## G.4 状态建模

建议显式建模以下状态：

```text
background
→ ongoing / right-censored
→ endpoint observed
→ completed but not emitted
→ emitted (absorbing)
```

`emitted` 必须是吸收态：进入后不得撤回或修改。

## G.5 Prefix-censored endpoint loss

设动作起点为 (s)，真实终点为 (e)，模型在时刻 (t) 输出终点 hazard：

[
h_t=P(E=t\mid E\ge t,x_{\le t})
]

对于完整观测到终点的样本：

[
\mathcal L_{\mathrm{end}}
=========================

-\log h_e
-\sum_{u=s}^{e-1}\log(1-h_u)
]

对于只观测到前缀 (c<e) 的 ongoing action：

[
\mathcal L_{\mathrm{cens}}
==========================

-\sum_{u=s}^{c}\log(1-h_u)
]

这与“把未来终点 regression loss 置零”不同：它明确监督模型在截止时刻之前**尚未结束**，而不是简单丢弃训练信号。

## G.6 Completion state

增加：

[
p_t^{\mathrm{comp}}=P(E\le t\mid x_{\le t})
]

并施加时间单调性或一致性约束：

[
p_{t+1}^{\mathrm{comp}}\ge p_t^{\mathrm{comp}}
]

该分支应表示动作是否已完成，而不是复制 actionness。

## G.7 Bounded-delay emission hazard

定义首次发射 hazard：

[
q_t=P(A=t\mid A\ge t,x_{\le t},\text{not emitted})
]

必须满足：

* 在动作未结束时发射受到强 early penalty；
* 在 (e+B) 前至少发射一次；
* 发射后进入 absorbing state；
* 延时越大有显式代价。

一个可实现的区间目标为：

[
P(A\le e+B)
===========

1-\prod_{u=e}^{e+B}(1-q_u)
]

[
\mathcal L_{\mathrm{emit}}
==========================

-\log P(A\le e+B)
+\lambda_{\mathrm{early}}\sum_{u<e}q_u
+\lambda_{\mathrm{delay}}\mathbb E[A-e]
]

还应对背景和 ongoing 状态施加 no-emission loss。

这样 emission head 才决定“何时发出”，而不是像当前实现一样只给 endpoint proposal 再乘一个分数。

## G.8 总损失

[
\mathcal L =
\mathcal L_{\mathrm{cls}}
+\lambda_s\mathcal L_{\mathrm{start}}
+\lambda_e(\mathcal L_{\mathrm{end}}+\mathcal L_{\mathrm{cens}})
+\lambda_c\mathcal L_{\mathrm{completion}}
+\lambda_a\mathcal L_{\mathrm{emit}}
+\lambda_{\mathrm{cal}}\mathcal L_{\mathrm{calibration}}
]

主论文第一版不加入 teacher loss，也不加入 learned selector loss。

## G.9 与已有工作的本质区别

* 与 MATR 的区别：MATR 重点是 memory architecture 和 current-end/past-start retrieval；PCEH 的贡献是**右截尾事件似然和独立发射 hazard**。
* 与 PKD/OnPoint 的区别：PCEH 不依赖 teacher，核心不是知识转移。
* 与 streaming distillation 的区别：cacheable student 是实现手段，不是核心 claim。
* 与 AdaTAD 的区别：PEFT 是训练策略；新问题在 causal event/emission supervision。
* 与当前仓库的区别：不再把窗末时刻当 emit，也不再把 end BCE 当 emission policy。

---

# H. Exact Training / Inference Protocol

## H.1 训练协议

### 1. 数据组织

禁止把互相独立的完整 192-token 窗口随机打乱后称为 streaming training。

每条 batch lane 必须保持：

```text
same video
→ chronological packets
→ persistent state
→ reset only at video boundary
```

不同视频可以并行组成 batch，但每个 lane 内时间必须单调。

### 2. 更新粒度

* packet size：8 raw frames；
* 更新周期：0.267 秒；
* cache horizon：192 packets；
* truncated BPTT：建议 32 或 64 packets；
* TBPTT 截断时 detach gradient，但不能清空流状态。

### 3. Prefix target construction

对于实例 ([s,e]) 和训练截止时刻 (c)：

| 条件           | 合法监督                                                              |
| ------------ | ----------------------------------------------------------------- |
| (c<s)        | background/no-emission                                            |
| (s\le c<e)   | class、start-past、ongoing、censored survival；禁止 endpoint regression |
| (c=e) 附近     | end-event hazard、completion transition                            |
| (e<c\le e+B) | completed-unemitted / on-time emission                            |
| (c>e+B)      | late-emission penalty或miss                                        |

GT endpoint 可以由 target builder 用来判定“该 prefix 是 censored 还是 observed”，但不得把 endpoint 数值、动作中心或完整 duration 注入 (c<e) 时的 feature、assignment 或 regression target。

### 4. 必须加入的 target 单元测试

构造两个训练样本：

```text
相同 prefix [0,c]
相同 start/class
不同未来 end e1、e2，且 c < min(e1,e2)
```

要求在时刻 (c)：

* cls target 相同；
* start target 相同；
* ongoing/censor target 相同；
* end regression target均不存在；
* emit target均为 no-emission；
* 所有 prefix-valid target逐值一致。

任何不一致都说明 endpoint-label leakage。

### 5. 视觉训练范围

第一篇主路线建议：

* 基础视觉塔 frozen；
* 训练 LoRA/TIA-style adapter、temporal adapter/cache 和 heads；
* 明确称 **PEFT raw-frame On-TAD**，不要称 full backbone fine-tuning。

每次运行保存：

* trainable parameter names；
* trainable/total parameter count；
* 每模块 grad norm；
* 首步和末步 parameter checksum；
* optimizer param-group dump。

### 6. 阈值校准

* emission threshold 只在 validation set 校准一次；
* test 固定；
* 每个 delay budget 不允许重新在 test 上调阈值；
* 可报告统一 threshold 和 per-budget validation threshold 两种设置，但必须明确区分。

## H.2 推理协议

每个 8-frame packet 执行：

```text
1. 记录本 packet 的真实 source-frame 范围
2. 只编码新帧
3. 更新 causal visual/temporal cache
4. 更新 ongoing hypotheses
5. 计算 end hazard、completion、emit hazard
6. 首次满足 emit policy 时发出 detection
7. 写入 immutable ledger
8. 对历史结果仅做 past-only duplicate suppression
```

### 禁止事项

* 禁止在 EOF 后重新跑整视频 NMS；
* 禁止修改已发出的边界；
* 禁止利用整个视频 duration 裁剪中间时刻输出；
* 禁止 test-time teacher、GT、feature cache shortcut；
* 禁止从预测终点伪造 source provenance；
* 禁止用未来帧刷新过去 cache；
* DDP 状态未修复前只报告 single-rank streaming。

## H.3 Ledger 字段

每条记录至少包含：

```text
video_id
stream_id
class
score
pred_start_event_frame
pred_end_event_frame
emit_frame
actual_max_raw_frame_read
actual_max_cache_source_frame
delay_to_pred_end
delay_to_matched_gt_end
cache_length
decoded_frames
encoded_frames
revision_count   # 必须为 0
```

## H.4 No-future 验证

必须做反事实测试：

1. 运行原视频到时刻 (t)；
2. 把 (t) 后所有帧替换为噪声、黑帧或另一视频；
3. 再运行；
4. 比较所有 (t) 前：

   * cache；
   * logits；
   * emitted detections；
   * ledger。

要求最大误差在数值容差内为零。只检查 `source_frame <= emit_frame` 不够。

## H.5 OnlineAP@B

不能在计算 AP 前删除 late prediction。

对 GT (g) 和 prediction (p)，TP 必须同时满足：

[
\text{class match},\quad
\operatorname{tIoU}(p,g)\ge\tau,\quad
e_g\le a_p\le e_g+B
]

若 prediction 空间上匹配但迟于 (e_g+B)：

* prediction 计为 late FP；
* GT 计为 missed-on-time instance。

这样指标才不会因删除迟到结果而虚高。

---

# I. Baseline / Ablation / Metric Matrix

## I.1 数据集

| Dataset       | 角色                            |       是否必需 | 裁决                                                                   |
| ------------- | ----------------------------- | ---------: | -------------------------------------------------------------------- |
| THUMOS14      | 快速开发、密集动作、与现有代码兼容             |          是 | 起步数据集，但单独不足以支撑 CCF-A 主张                                              |
| MUSES         | 长视频、多镜头、现有 On-TAL 主 benchmark |      **是** | MATR 已在 THUMOS14 和 MUSES 上验证；不做 MUSES 很难声称通用 On-TAL ([欧洲计算机视觉协会][2]) |
| ActivityNet   | 长时长、稀疏实例、规模更大                 |         建议 | 需重新定义流式 split 和负例协议，不能直接套 offline mAP                                |
| EPIC-Kitchens | egocentric、细粒度、连续动作           |         可选 | 适合 domain generalization，但任务标签与 THUMOS 差异较大                          |
| Ego4D         | 长视频、自然流、复杂事件                  | 可选/Route 2 | 更适合大规模预训练或外部泛化，不是最小论文闭环必需                                            |

**最低论文闭环：THUMOS14 + MUSES。**

## I.2 必需 baselines

| ID  | Baseline                                              | 控制目的                                |
| --- | ----------------------------------------------------- | ----------------------------------- |
| B0  | 当前 SHA + final config                                 | 证明重构相对现有骨架的收益                       |
| B1  | fixed causal stride-2 selected-only                   | 选帧和预算基本线                            |
| B2  | dense frozen visual encoder                           | 排除稀疏编码损失                            |
| B3  | dense PEFT encoder                                    | 分离 PEFT 与 censor/emission objective |
| B4  | same 192-token context、per-snippet rolling emission   | 证明收益不是仅来自更频繁运行                      |
| B5  | simple sliding-window anchor-free On-TAL              | 最直接工程 baseline                      |
| B6  | OAT-style sliding-window baseline                     | 对照既有 sliding-window On-TAL          |
| B7  | MATR-style feature baseline                           | 对照 memory/current-end/past-start    |
| B8  | offline TAD teacher upper bound                       | 精度上界，不参与在线主比较                       |
| B9  | vanilla logit KD / PKD-style baseline                 | 证明核心不是普通 privileged KD              |
| B10 | random/uniform/motion selector，同 encoded-frame budget | 仅在引入 learned acquisition 时使用        |
| B11 | offline AdaTAD PEFT upper bound                       | 衡量 causal/online 约束带来的损失            |

公平性必须控制：

* 同一 raw-frame resolution；
* 同一 visual backbone；
* 同一更新 cadence；
* 同一可见前缀；
* 同一 encoded-frame budget；
* 同一后处理候选数；
* 同一 validation calibration protocol。

## I.3 必需 ablations

| Ablation                                 | 要回答的问题                         | Kill condition                |
| ---------------------------------------- | ------------------------------ | ----------------------------- |
| 去掉 censored survival，恢复 endpoint masking | censor likelihood 是否真有贡献       | 与完整方法无显著差异                    |
| 去掉 emission hazard，改固定阈值/窗末 emit         | 学习发射是否必要                       | OnlineAP@B、p95 latency无改善     |
| emit head = end head                     | 独立 emission variable 是否必要      | 完整方法没有收益                      |
| 去掉 completion state                      | completion 是否提供额外信息            | 不改善 endpoint/emit calibration |
| freeze vs PEFT                           | raw-frame task adaptation 是否有效 | PEFT 无稳定增益或参数未更新              |
| cache 16/32/64/192                       | 长上下文是否有价值                      | 192 与短 cache无差异但成本更高          |
| cadence 1/2/4/8 snippets                 | 低延时与成本曲线                       | 只在极高计算下有效                     |
| budget 0.5/1/2/4s                        | 方法是否真正控制 delay                 | 只在 4s 或更大预算有效                 |
| no teacher vs vanilla KD                 | KD 是否只是附加增益                    | teacher 才是全部收益来源              |
| THUMOS → MUSES transfer                  | 是否过拟合短视频/单镜头                   | MUSES 崩溃                      |
| source provenance on/off                 | audit 是否暴露隐藏读取                 | 开启后出现 violations              |
| learned selector vs uniform              | selector 是否 detector-useful    | 不优于 uniform 同预算               |

## I.4 指标

### 检测质量

* Offline mAP@0.3:0.7：只作为兼容性指标；
* [OnlineAP@0.5s](mailto:OnlineAP@0.5s)、1s、2s、4s；
* 高 tIoU OnlineAP，特别是 0.6/0.7；
* start/end absolute error；
* boundary drift versus emit delay。

### 响应性

* latency mean、p50、p90、p95；
* false-early-emission rate；
* late-emission rate；
* missed-endpoint rate；
* no-emission-within-budget rate。

延时必须相对匹配 GT end 计算，并同时报告未匹配/迟到实例，避免 survivor bias。

### 协议正确性

* future raw-frame read violations；
* future cache-source violations；
* revision count；
* non-monotonic emit count；
* future perturbation max output difference；
* prefix target invariance failures。

### 计算

* decoded frames；
* encoded frames；
  -视觉 backbone FLOPs/MACs；
* 每 packet wall-clock；
* end-to-end FPS；
* real-time factor；
* peak GPU memory；
* cache GPU/CPU memory；
* training GPU-hours；
* inference energy或至少功耗采样；
* 数据解码、预处理、selector、encoder、head 分项时间。

### 统计

* 最低 3 seeds；
* 最好 5 seeds；
* mean ± std；
* paired per-video bootstrap confidence interval；
* 主指标差异做配对显著性检验。

---

# J. Reviewer Risk Register

| 风险                       | Reviewer 攻击方式                          | Mitigation                                                          | 失败/停线条件                         |
| ------------------------ | -------------------------------------- | ------------------------------------------------------------------- | ------------------------------- |
| Novelty weak             | “只是 causal head + survival loss”       | 明确 event time 与 emit time 的任务差异；与 endpoint BCE、post-filter、KD 做直接消融 | 核心目标移除后性能不变                     |
| Online protocol leakage  | “source frame 是自己写进 ledger 的”          | 全链路 provenance + future perturbation                                | 任意真实 future read 或前缀输出变化        |
| Latency too high         | “51.2s 不是 online responsiveness”       | per-snippet update；主预算 1s；报告 p95                                    | p95 仍大于 2s，或只能窗末发射              |
| Unfair baseline          | “你的 baseline cadence/backbone 更弱”      | 完全同 backbone、cadence、context、budget                                 | 无法复现公平 normalized baseline      |
| Endpoint label leakage   | “pre-end target 用了 GT duration/center” | target invariance 单元测试；删除 center-based future assignment            | 更改未来 endpoint 后 prefix target变化 |
| Distillation not novel   | “PKD/OnPoint 已做”                       | teacher 仅作 baseline，非主贡献                                            | 主要增益全部来自 teacher                |
| End-to-end unsupported   | “视觉塔 frozen/参数没进 optimizer”            | 精确称 PEFT；dump optimizer/grad/delta                                  | 宣称 trainable 参数未更新              |
| THUMOS-only              | “小数据集、短动作、过拟合”                         | MUSES 作为强制第二数据集                                                     | 不完成 MUSES 或跨数据集结论相反             |
| Adaptive claim false     | “配置写 adaptive，实际 stride-2”             | 立即重命名 metadata；在真正 learned 前不使用 adaptive                            | 继续使用错误标签                        |
| Metric gaming            | “late rows 被删除，延时相对预测 end”             | OnlineAP@B；GT-end latency；late作为错误                                  | 仍使用过滤后 mAP 作为主指标                |
| Efficiency unsupported   | “只是少编码，视频仍全解码”                         | decode-inclusive profiling、incremental cache                        | wall-clock/FPS 无收益              |
| Full MATR misnaming      | “memory=0 为什么叫 MATR？”                  | 重命名 head；MATR 仅作 baseline                                           | 继续以 MATR claim 发表               |
| DDP state corruption     | “不同 rank 拆散同一视频”                       | single-rank主结果或实现 state-aware video sharding                        | 多卡 ledger 与单卡不一致                |
| Dataset duration leakage | “模型知道未来视频长度”                           | 流中不提供 total duration；只在 EOF 得知结束                                    | 移除 duration 后结果显著下降             |
| Early anticipation混入     | “未结束就输出，却按 completed detection评估”      | 把 anticipation 与 confirmed detection 分成两个协议                         | 同一结果同时声称 early和confirmed        |

---

# K. GO / HOLD / NO-GO Verdict

## K.1 当前仓库

### **NO-GO**

不应基于当前实现启动“最终论文主方法”的长训练并对外声称：

* adaptive online selection；
* low-latency On-TAD；
* full MATR；
* prefix-censored learning；
* learned emission；
* visual backbone end-to-end adaptation；
* full-chain no-future。

继续训练当前 final config，最多会得到一个 **fixed-stride chunk-end baseline**。即使 mAP 不差，也无法修复任务定义和创新性问题。

## K.2 重构研究方向

### **HOLD**

PCEH Route 1 值得进入实现，但只有以下 gate 全部通过后才转 GO：

1. **Cadence gate**：每 8 帧最多一次更新和发射检查，不再整窗末发射。
2. **Causality gate**：future perturbation 零违规。
3. **Target gate**：prefix target invariance 全部通过。
4. **Optimizer gate**：所有宣称 PEFT 参数有非零更新。
5. **Metric gate**：实现不删除 late rows 的 OnlineAP@B。
6. **Latency gate**：主预算 1 秒，报告 p95，不以预测 end 代替 GT end。
7. **Baseline gate**：公平超过 rolling fixed-stride、OAT/MATR-style 或至少形成显著 Pareto improvement。
8. **Dataset gate**：THUMOS14 和 MUSES 均完成至少 3 seeds。
9. **Efficiency gate**：包含 decode/I/O 的端到端吞吐或延时有真实收益。
10. **Novelty gate**：censor + emission hazard 消融显示稳定独立贡献。

满足这些条件后，才可将项目状态改为 **GO**。

---

# L. Next Implementation Checklist

## L.1 立即停止的事项

* 停止把 `causal_stride` 写成 `adaptive_selected_frames`。
* 停止将 51.2 秒 `max_latency_frames` 描述成 low latency。
* 停止把 `MATRHead(memory_size=0)` 描述为 MATR implementation。
* 停止将 endpoint-local BCE 描述为 emission learning。
* 停止用预测终点计算的 latency 作为主延时指标。
* 停止在完整 streaming protocol 和 evaluator 修复前跑昂贵 final full training。
* 停止把 teacher、selector、pretraining 同时堆进第一版主方法。

## L.2 Gate 0：清理现有 claim 和 metadata

修改：

* `configs/causaltad/thumos_siglip2_adaptive_matr_ontad_final.py`
* `online_siglip_adapter.py`
* 实验命名和 README

要求：

```text
frame_policy = "fixed_causal_stride2"
method_stage = "chunk_end_baseline"
head_name = "CausalAnchorFreeEventHead"
```

`frame_policy` 必须由 selector 实际类型和参数自动生成，禁止 adapter 硬编码覆盖。

完成标准：ledger、config 和运行时 selector 三者一致。

## L.3 Gate 1：建立真实 streaming dataset

新增类似：

```text
opentad/datasets/streaming_raw_frame.py
StreamingRawFrameDataset
ChronologicalStreamBatchSampler
```

接口：

```python
packet = {
    "video_id": ...,
    "packet_start_frame": ...,
    "packet_end_frame": ...,
    "frames": ...,
    "is_video_start": ...,
    "is_video_end": ...,
}
```

要求：

* 每个样本只含新增 packet；
* dataset 不向中间 packet 暴露整视频 duration；
* batch sampler 保证 lane 内时间单调；
* state 只在 `is_video_start` 时清空。

## L.4 Gate 2：改成 incremental backbone/cache

在视觉与时序模块增加：

```python
features_t, state_t, read_trace = model.forward_step(
    new_frames=packet_frames,
    state=state_prev,
    packet_meta=meta,
)
```

要求：

* 只编码新帧；
* cache 最大长度显式；
* 每个 cache token记录来源帧；
* 输出真实 `max_raw_frame_read`；
* 不能由 prediction 反推 source。

## L.5 Gate 3：重写 prefix target builder

不要继续在现有 `_build_online_branch_targets()` 上零散加 mask。

新增独立模块：

```text
opentad/models/targets/prefix_event_targets.py
```

输出：

```python
PrefixTargets(
    class_target,
    start_target,
    ongoing_target,
    end_event,
    censor_mask,
    completion_target,
    emit_allowed,
    emit_forbidden,
    delay_budget,
)
```

附带单元测试：

```text
test_future_endpoint_does_not_change_prefix_targets
test_no_end_regression_before_endpoint
test_emit_forbidden_before_endpoint
test_completion_is_monotonic
```

## L.6 Gate 4：实现 event/emission hazard head

建议将当前 head 重构为：

```text
PrefixEventEmissionHead
├── class_head
├── start_memory_head
├── ongoing_head
├── end_hazard_head
├── completion_head
└── emission_hazard_head
```

删除：

* `emit_target = end_mask`
* 窗末固定 emit 作为主协议
* 将 emit score仅乘入 proposal score的做法

实现：

* censored survival likelihood；
* first-emission distribution；
* early/late delay cost；
* absorbing emission state。

## L.7 Gate 5：修复 optimizer scope

运行时强制输出：

```text
parameter_name
requires_grad
optimizer_group_id
learning_rate
grad_norm
parameter_delta_after_step
```

增加 fail-closed assertion：

```python
for name in claimed_trainable_names:
    assert name in optimizer_parameter_names
    assert grad_is_finite_and_nonzero(name)
```

若 `get_optim_groups()` 跳过 backbone adapters，必须修复命名匹配或不再使用该分组函数。

## L.8 Gate 6：重写 evaluator

新增：

```text
OnlineAPBudgeted
GTMatchedLatencyEvaluator
PrefixCausalityAudit
FuturePerturbationAudit
```

禁止：

```python
if latency > B:
    drop_prediction()
```

正确行为是：

```text
late prediction → late FP
matched GT → on-time miss
```

同时分别记录：

* `emit - predicted_end`
* `emit - matched_gt_end`

主论文只把后者称为 detection latency。

## L.9 Gate 7：实现公平 baseline configs

至少建立：

```text
rolling_dense_frozen.py
rolling_dense_peft.py
rolling_fixed_stride2.py
rolling_motion_same_budget.py
chunk_end_current_repo.py
matr_feature_baseline.py
oat_style_baseline.py
vanilla_kd_baseline.py
```

所有配置共享：

* backbone；
* image size；
* packet size；
* context horizon；
* post-processing quota；
* validation threshold protocol。

## L.10 Gate 8：最小实验放行标准

先做 THUMOS14：

* 3 seeds；
* current chunk-end；
* rolling fixed stride；
* rolling dense；
* PCEH；
* PCEH minus censor；
* PCEH minus emission hazard。

只有同时看到以下结果才扩展到 MUSES：

1. OnlineAP@1s 稳定提升；
2. false early emission 不恶化；
3. p95 latency 达标；
4. offline mAP 没有灾难性下降；
5. causal audit 零违规；
6. 参数更新审计通过；
7. 增益不是单个 seed；
8. rolling fixed-stride baseline 被公平超过。

随后在 MUSES 完成同样的 3–5 seed 闭环。若 PCEH 只在 THUMOS 有效，则不能写成通用 On-TAD 方法。

---

## 最终一句裁决

**当前代码验证了一个可运行的 raw-frame selected-only causal skeleton，但没有验证 adaptive selection、完整 prefix censoring、learned emission、低延时 streaming、MATR memory 或 full-chain end-to-end causality。当前 final config 不应进入论文主方法长训练。唯一值得推进的主线是先把任务重定义为“右截尾 event-time + 有界延时 emission-time”的 Route 1，并用 per-snippet cached inference、真实 read provenance 和 OnlineAP@B 重建整个训练—推理—评估闭环。**

[1]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/tree/codex/online-tad-clean-20260702 "https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/tree/codex/online-tad-clean-20260702"
[2]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02834.pdf "https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02834.pdf"
[3]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/configs/causaltad/thumos_siglip2_matr_ontad_p0.py "https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/configs/causaltad/thumos_siglip2_matr_ontad_p0.py"
[4]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/configs/causaltad/thumos_siglip2_adaptive_matr_ontad_final.py "https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/configs/causaltad/thumos_siglip2_adaptive_matr_ontad_final.py"
[5]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/opentad/utils/online_protocol.py "https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/opentad/utils/online_protocol.py"
[6]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/opentad/datasets/raw_frame.py "https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/opentad/datasets/raw_frame.py"
[7]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/opentad/models/detectors/mamba.py "https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/opentad/models/detectors/mamba.py"
[8]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/4e222ccf4de1e989c6faf83031772c552be7af59/opentad/models/backbones/online_siglip_adapter.py "https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/4e222ccf4de1e989c6faf83031772c552be7af59/opentad/models/backbones/online_siglip_adapter.py"
[9]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/opentad/models/dense_heads/matr_head.py "https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/opentad/models/dense_heads/matr_head.py"
[10]: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/opentad/evaluations/online_map.py "https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/online-tad-clean-20260702/opentad/evaluations/online_map.py"
[11]: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136940640.pdf "https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136940640.pdf"
[12]: https://arxiv.org/abs/2011.09158 "https://arxiv.org/abs/2011.09158"
[13]: https://arxiv.org/abs/2607.00289 "https://arxiv.org/abs/2607.00289"
[14]: https://openaccess.thecvf.com/content/WACV2026/papers/Patel_Distilling_Offline_Action_Detection_Models_into_Real-Time_Streaming_Models_WACV_2026_paper.pdf "https://openaccess.thecvf.com/content/WACV2026/papers/Patel_Distilling_Offline_Action_Detection_Models_into_Real-Time_Streaming_Models_WACV_2026_paper.pdf"
[15]: https://arxiv.org/abs/2311.17241 "https://arxiv.org/abs/2311.17241"
