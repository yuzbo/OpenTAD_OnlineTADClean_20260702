# 2026 Online Action Models Research Note

更新日期：2026-07-03

## 调研范围

本记录补充 `end-to-end-ontad-research.md`，关注截至 2026-07-03 最新的在线动作检测、在线动作分割和流式视频理解模型。

术语边界：

- `OAD`：online action detection，通常是帧级或 clip 级当前动作分类。
- `OAA`：online action anticipation，预测未来动作。
- `On-TAL` / `On-TAD`：online temporal action localization/detection，输出动作实例的起止边界和类别。
- `Online TAS`：online temporal action segmentation，流式逐帧分割动作阶段，常见于 egocentric/procedural task。
- `Streaming video understanding`：面向 VLM/MLLM 的连续视频问答、记忆、实时交互、主动响应等更宽泛任务。

## 总体判断

2026 年的最新趋势不是单一“端到端 On-TAD”路线，而是三股并行迁移：

1. **检测/定位**：从 Transformer memory 转向 Mamba、层次记忆压缩、状态图、offline-to-online distillation，以及开放词表/zero-shot。
2. **分割**：从 offline TAS 迁移到 causal/online TAS，并进一步关注 egocentric、procedural、能耗约束和多模态传感器选择。
3. **理解**：VLM/MLLM 把在线动作问题扩大成 streaming video understanding，核心指标从 mAP 扩展到 latency、TTFT、实时响应时机、长记忆和主动交互。

对当前 `CausalTAD` 最直接可借鉴的是：MOAD/Backtrace Mamba 的层次记忆压缩，OnPoint 的 offline-to-online distillation，OZ-TAL / OV-OAD 的开放词表迁移，以及 FluxMem / StreamingTOM 的 token/memory compression。

## 2026 候选模型表

| 模型/工作 | 年份/状态 | 子任务 | 核心思路 | 输入形态 | 对 CausalTAD 的价值 |
| --- | --- | --- | --- | --- | --- |
| MOAD / Backtrace Mamba | AAAI 2026 | OAD | Mamba + 层次 action/scene memory + quantization + temporal soft pruning | 预提取 RGB/flow 特征 | 高。与 causal Mamba 路线贴近，可借鉴记忆槽和软剪枝 |
| CAKE | arXiv 2026 | OAD | RGB-only motion distillation，Dynamic Motion Adapter，背景感知对比学习 | 推理用 RGB，教师含 optical flow | 中。适合降低 flow 依赖，但仍是帧级 OAD |
| State-Specific Model (SSM) | arXiv 2025, v2 2026 | OAD + OAA | critical-state memory compression + state-transition graph + cross-temporal interaction | RGB/flow/object 特征 | 中高。可借鉴“关键状态”而不是全历史缓存 |
| OZ-TAL | arXiv 2026 | zero-shot On-TAL | training-free VLM 框架，面向未见动作的在线 temporal localization | VLM 特征/表示 | 高。把 On-TAL 从闭集推向 open-vocabulary |
| OnPoint | ECCV 2026 | point-supervised On-TAL | offline TAL teacher 向严格 online student 做多级蒸馏 | 特征级在线学生 | 高。是弱标注 + online localization 的最新强信号 |
| COAD | ICPR 2026 listed; ICLR 2026 withdrawn submission | egocentric OAD | 连续 egocentric online action detection | 待进一步核验 | 观察项。可能贴近第一视角部署，但公开信息需复查 |
| DSTA | WACV 2026 | streaming spatio-temporal action detection | 将 offline action detector 蒸馏到 real-time streaming student | 视频帧/RoI 流 | 中。空间框检测任务，不是 temporal-only TAD |

## 在线动作检测 / 定位

**MOAD / Backtrace Mamba** 是 2026 最贴近本仓库路线的 OAD 工作。它把 OAD 定义为不访问未来帧的实时 ongoing action prediction，提出 Mamba-based framework，使用 action/scene 层次记忆、特征量化和 temporal soft pruning。论文实验仍使用预提取特征：TVSeries/THUMOS14 用 two-stream 网络特征，FineAction 用 I3D 特征。因此它不是 raw-frame 端到端 On-TAD，但它的 memory compression 和 Mamba block 很适合移植到 `CausalProj` 的在线缓存设计。

**CAKE** 关注低成本实时 OAD。它用 optical-flow teacher 训练 RGB student，通过 Dynamic Motion Adapter 从 RGB 变化中近似运动线索，并用 Floating Contrastive Learning 处理背景多样性。它报告单 CPU 超过 72 FPS，适合资源受限部署。局限是任务仍是帧级 OAD，不输出动作实例边界。

**SSM: Action-Dynamics Modeling and Cross-Temporal Interaction** 把 online action detection 和 anticipation 放进一个统一框架。它不保留完整历史，而是把 video features 压缩成 critical states，再构造 state-transition graph 生成 intention cues，并通过 cross-temporal interaction 同时服务当前检测和未来预测。这个思想对 CausalTAD 的启发是：历史记忆可以是“状态图/关键状态”，不必只是滑窗 token。

**OZ-TAL** 是 2026 在线定位方向的新点：提出 Online Zero-shot Temporal Action Localization，目标是在在线设置下定位未见动作。它明确把 On-TAL 定义为在动作完成时立即检测发生时间和类别，并指出近期 On-TAL 正从 OAD aggregation 走向 instance-level understanding。它的 training-free VLM 方案值得作为开放词表 On-TAD 的参考，但更偏表示/检索框架，不是 CausalTAD 这种 fully trained detector。

**OnPoint** 是 2026 On-TAL 的另一个重要信号。它提出 Point-Supervised Online TAL (POTAL)：训练时每个实例只给一个时间点，推理时仍要求严格 online。方法是 offline point-supervised TAL teacher 向 online student 蒸馏 pseudo segments、class-activation subsequences 和 anticipatory window-level cues。对当前项目来说，这是“offline 强教师 -> online 因果学生”的合理实验路线，但要避免在测试阶段引入 teacher cache 或 future context。

## 在线动作分割

在线 TAS 在 2024-2026 之间开始成形。**ProTAS / Progress-Aware Online Action Segmentation** 解决 egocentric procedural task 的 streaming segmentation：把 TCN/Transformer 改成 causal，预测动作 progress，并用 task graph 约束平滑、减少过分割。它是当前在线动作分割的强参考，尤其适合 AR/VR task assistant。

**OnlineTAS**（NeurIPS 2024）提出通用 Online TAS baseline：用 adaptive memory 适应动态上下文，用 feature augmentation 增强当前帧，并加入后处理缓解在线设置下严重过分割。虽然不是 2026 新作，但它是在线 TAS 的关键基线。

**Ego-METAS**（arXiv 2026）把在线 TAS 推向 embodied/egocentric 多模态与能耗约束。它提供超过 100 小时未裁剪第一视角视频，包含 RGB、audio、gaze、IMU、monochrome camera 五类模态，并要求模型在每个时间步动态选择传感器，同时遵守硬件能耗预算。它更像 benchmark/testbed，不是单个强模型，但说明 2026 年在线动作分割已经开始关心 always-on 设备部署。

## 流式视频理解

2026 的 streaming video understanding 主要由 VLM/MLLM 推动，和传统 TAD 的关系是“上层语义理解/交互”而不是直接替代 mAP 检测器。

**FluxMem**（CVPR 2026）是 training-free streaming video understanding 框架。它用 Temporal Adjacency Selection 删除相邻帧冗余 token，用 Spatial Domain Consolidation 合并空间重复区域，并自适应决定压缩率。报告在 StreamingBench / OVO-Bench 上达到强结果，同时显著降低 latency 和 GPU memory。它给 CausalTAD 的启发是：在线缓存可以做场景自适应 token retention，而不是固定长度 FIFO。

**StreamingTOM**（CVPR 2026）同样是 training-free token/memory compression。它用 causal temporal reduction 控制每帧视觉 token 预算，用 4-bit online quantized memory 保持 bounded KV-cache，强调可预测延迟和 bounded active memory。这和在线 TAD 的“长期历史但低延迟”目标非常一致。

**Think-as-You-See (TaYS)**（CVPR 2026）把 streaming video reasoning 做成并发式推理：视觉帧到达时模型同步生成/更新 reasoning，使用 streaming attention mask、position encoding 和 dual KV-cache。它不是动作检测器，但对“边看边推理”的在线解释和多步理解很有参考价值。

**Proact-VL**（ICML 2026）聚焦 proactive real-time VideoLLM：连续输入下低延迟推理、自动决定何时响应，并控制响应质量/数量。它引入 Live Gaming Benchmark，适合观察实时交互系统的评测指标。

**SimpleStream**（arXiv 2026）是一个很重要的“冷水提醒”：最近帧滑窗喂给 off-the-shelf VLM，已经能匹配或超过不少复杂 streaming 方法。它提示后续任何复杂 memory/cache 设计都应该和 recent-frame baseline 公平比较。

**StreamEQA / RIVER / Streaming Harness** 等 2026 benchmark 正在把 streaming understanding 从单轮 QA 推向 embodied perception、interaction、planning、long-horizon memory、live perception 和 proactive response。它们适合作为上层在线理解评测，不适合作为 TAD mAP 的直接替代。

## 对当前仓库的建议

1. 继续把 `CausalTAD` 主线定义为 **feature-level online/causal TAD**，不要因为 VLM streaming 工作兴起就混淆成同一任务。
2. 优先复用 MOAD 的思路做 feature memory ablation：action/scene memory bank、memory quantization、temporal soft pruning。
3. 增加一个“offline teacher -> online student”的实验草案，但必须在训练和测试协议里明确：teacher 只能训练期蒸馏，测试期不得读取 teacher cache、future context 或 raw prediction shortcut。
4. 增加 open-vocabulary / zero-shot 旁路线时，可以参考 OV-OAD 和 OZ-TAL；先做 frozen VLM/text embedding + CausalTAD head，不急于 full VLM 端到端。
5. 对在线后处理增加响应性指标：延迟、action completion 后 emission delay、proposal revision policy、FPS/throughput、memory footprint。
6. 若探索在线 TAS，可从 `FrameWindowDataset` / `FeatureWindowDataset` 加 causal segmentation head，而不是直接把 TAS 与 On-TAD 指标混在一起。
7. 对任何新配置继续写清数据形态：pre-extracted video features、raw frames、audio features、multimodal sensor features；不得把数据、特征或 checkpoint 放入仓库。

## 来源

- [Backtrace Mamba: Reviving Critical Temporal Contexts via Hierarchical Memory Compression for Online Action Detection, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/38139)
- [Backtrace Mamba PDF](https://ojs.aaai.org/index.php/AAAI/article/download/38139/42101)
- [CAKE: Real-time Action Detection via Motion Distillation and Background-aware Contrastive Learning](https://arxiv.org/html/2603.23988v1)
- [Action-Dynamics Modeling and Cross-Temporal Interaction for Online Action Understanding](https://arxiv.org/html/2510.10682v2)
- [OZ-TAL: Online Zero-Shot Temporal Action Localization](https://arxiv.org/abs/2605.09976)
- [OnPoint: Offline-to-Online Multi-Level Distillation for Point-Supervised Online Temporal Action Localization](https://arxiv.org/abs/2607.00289)
- [Distilling Offline Action Detection Models into Real-Time Streaming Models](https://www.nec-labs.com/blog/offline-to-online-streaming-distillation-of-action-detection-models/)
- [Progress-Aware Online Action Segmentation for Egocentric Procedural Task Videos, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Shen_Progress-Aware_Online_Action_Segmentation_for_Egocentric_Procedural_Task_Videos_CVPR_2024_paper.html)
- [OnlineTAS: An Online Baseline for Temporal Action Segmentation](https://arxiv.org/abs/2411.01122)
- [Ego-METAS: Egocentric online Multimodal Energy-efficient Temporal Action Segmentation benchmark](https://arxiv.org/abs/2606.02246)
- [FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding](https://arxiv.org/abs/2603.02096)
- [StreamingTOM: Streaming Token Compression for Efficient Video Understanding, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_StreamingTOM_Streaming_Token_Compression_for_Efficient_Video_Understanding_CVPR_2026_paper.html)
- [Think-as-You-See: Streaming Chain-of-Thought Reasoning for Large Vision-Language Models](https://arxiv.org/abs/2603.02872)
- [Proact-VL: A Proactive VideoLLM for Real-Time AI Companions](https://arxiv.org/abs/2603.03447)
- [A Simple Baseline for Streaming Video Understanding](https://arxiv.org/html/2604.02317v1)
- [StreamEQA: Towards Streaming Video Understanding for Embodied Scenarios](https://arxiv.org/abs/2512.04451)
- [RIVER: A Real-Time Interaction Benchmark for Video LLMs](https://github.com/OpenGVLab/RIVER)
- [Harnessing Streaming Video in the Wild](https://arxiv.org/abs/2606.08615)
