# End-to-End On-TAD Research Note

更新日期：2026-07-03

最新复核：2026-07-12

## 2026-07-12 任务内复核与新路线

用户明确否决 PIVOT/Three-Clock 路线，因为它把问题改成 physically anchored streaming event verification，超出了标准 On-TAD。当前研究问题重新固定为：

> 能否在不修改 On-TAD 输入、输出和评测定义的前提下，构建从 raw RGB 到动作实例输出联合训练、严格因果、低冗余计算的实例级在线检测器？

### 最新查新结论

截至本次复核，仍未找到同时满足以下条件的代表性公开方法：

1. raw RGB 输入而非预提取特征；
2. 可训练视觉骨干与实例检测头联合优化；
3. 严格无未来的 On-TAD 推理；
4. 输出动作实例 `{start, end, class, score}`，而不是逐帧 OAD 分类；
5. 训练与增量缓存推理具有可审计的 prefix equivalence；
6. 不依赖离线 NMS 或事后删除历史预测。

关键证据：

- MATR 论文在 THUMOS14 上冻结 two-stream TSN，在 MUSES 上使用 I3D；其官方代码输入是 `thumos_all_feature_*.pickle`。因此论文中的 “end-to-end architecture” 是检测器内部端到端，不是本文严格定义的 raw-video 端到端。
- HAT、ActionSwitch、SimOn、OAT 和 OnPoint 同样依赖预提取或冻结特征。ActionSwitch 解决了重叠和同类实例，但仍是 feature-level state machine 加独立分类器。
- E2E-LOAD 已证明 raw-video end-to-end online action detection 可行，但任务输出是 frame-level OAD。
- StreamFormer 已训练 raw-video causal streaming backbone，并支持 KV cache；但下游 OAD 冻结该 backbone，且仍是逐帧分类。
- TIA/AdaTAD、LoSA、Re2TAL、ETAD 等解决 raw-video/offline TAL 的联合训练或成本问题，但其检测任务可访问完整视频或未来上下文。

因此，“使用 causal backbone”“使用 LoRA”“使用 raw frames”任何一项单独都不新。仍值得验证的是它们与**持久动作实例状态、标准 On-TAD emission、prefix-equivalent training**的交集。

### 当前最佳候选：PETAL-OnTAD

PETAL 把窗口级重复检测改为内部动作实例跟踪：一个 persistent event query 从动作开始证据出现后持续表示同一实例，在动作结束时输出标准 On-TAD detection。训练时用 block-causal mask 在一个长 chunk 中计算所有前缀；推理时用同一模型和 KV/SSM cache 逐步更新。Raw RGB encoder、causal temporal layers、persistent queries 和检测头联合优化。

这个路线不引入新任务或新输出。Pre-end query 只是模型隐状态，最终 detection 仍然不可回改。核心风险是它可能被审稿人解释为 E2E-LOAD/StreamFormer、MATR 和 TrackFormer 的直接组合，因此 raw-video 实现前必须先完成：

1. matched feature-level persistent-query versus fresh-window-query pilot；
2. dedicated Pro novelty review；
3. batched causal versus incremental cached prefix-equivalence test。

完整节点见 [`research-wiki/ideas/petal-ontad.md`](research-wiki/ideas/petal-ontad.md)。

## 调研问题

是否已经存在真正端到端的 online temporal action detection/localization（On-TAD/On-TAL）路线：模型直接读取视频帧，而不是依赖离线预提取特征，并在在线/因果约束下输出动作实例的起止边界和类别。

这里采用较严格定义：

- `OAD`（online action detection）：流式视频中对当前帧/clip 做动作类别预测，通常是帧级概率。
- `On-TAL` / `On-TAD`：流式视频中输出动作实例，即开始时间、结束时间和类别；不能访问未来帧，也不能回改已经输出的 proposal。
- `raw-frame end-to-end`：训练/推理图里包含视频编码器，从 RGB 帧到检测结果联合优化；不以离线生成的 `.npy/.pkl` 特征作为模型输入。

## 结论

未找到一个已经成为主流基准、同时满足“raw-frame 端到端 + 严格在线 + 实例级 temporal localization”的 On-TAD/On-TAL 方法。

最接近的正例是 **E2E-LOAD**（ICCV 2023）：它明确提出首个端到端 Online Action Detection（OAD）模型，直接处理 raw RGB frames，并用 stream buffer / token reuse 做低成本在线推理。但它的任务是 OAD 帧级动作检测，不是完整的实例级 temporal action localization，因此不能直接等同于当前仓库关心的 CausalTAD/TAD 输出形态。

实例级 On-TAL 方向已经有 CAG-QIL、SimOn、OAT、MATR、HAT、OnPoint 等工作，但公开实验基本仍使用预训练/冻结特征或预提取特征。部分论文把 detector/head 侧称为 “end-to-end”，但按本记录的严格定义，它们不是从视频帧到动作实例的端到端训练。

离线 TAD 的 raw-frame 端到端训练已经存在，并且近年进展明显，例如 CVPR 2022 的 end-to-end TAD 实证研究，以及 CVPR 2024 的 AdaTAD / 1B 参数长视频端到端 TAD。但这些不是严格在线设置，不能直接证明 On-TAD 已经有成熟 raw-frame 端到端路线。

## 代表工作对照

| 工作 | 年份/类型 | 是否在线 | 输出粒度 | 输入形态 | 是否满足严格 raw-frame On-TAD |
| --- | --- | --- | --- | --- | --- |
| E2E-LOAD | ICCV 2023 OAD | 是 | 帧级动作概率 | raw RGB frames，端到端训练 | 否。raw-frame 端到端成立，但不是实例级起止边界 TAD |
| LSTR | NeurIPS 2021 OAD | 是 | 帧级动作概率 | 预训练特征抽取器输出特征 | 否 |
| OadTR | ICCV 2021 OAD | 是 | 帧级动作概率 | RGB/flow 预提取特征 | 否 |
| CAG-QIL | ICCV 2021 On-TAL | 是 | 动作实例 | OAD 输出/特征后处理 | 否 |
| SimOn | arXiv 2022 On-TAL | 是 | 动作实例 | 预训练 3D CNN 特征 | 否。论文称 On-TAL 模型端到端，但不是 raw-frame 端到端 |
| OAT / Sliding Window Scheme | ECCV 2022 On-TAL | 是 | 动作实例 | 预训练 feature encoder 输出特征 | 否 |
| MATR | ECCV 2024 On-TAL | 是 | 动作实例 | 冻结 TSN/I3D 特征 | 否。架构端到端，实验仍是冻结特征 |
| HAT | ECCV 2024 On-TAL | 是 | 动作实例 | I3D/SlowFast/TSN 等预训练特征 | 否 |
| OnPoint | arXiv 2026 point-supervised On-TAL | 是 | 动作实例 | 预提取特征 | 否 |
| AdaTAD / ETAD 类离线 TAD | 2022-2024 TAD | 否 | 动作实例 | raw-frame / 可端到端 | 否。端到端成立，但不是在线 |

## 对当前仓库的含义

当前 `CausalTAD` 路线仍应明确标注为 **特征级在线/因果 TAD**。仓库 README 和配置中“默认输入是预提取特征”的边界是准确的。

如果后续要探索 raw-frame 端到端 On-TAD，建议不要直接宣称已有成熟可复现路线，而是作为新实验方向：

1. 从 `feature-level CausalTAD` 保持主线，先做可靠在线协议、causal attention / causal Mamba、滑窗/有限未来上下文和在线后处理。
2. 新增 `frame-level` 数据形态时，配置必须显式写明输入是视频帧；不要把帧、特征缓存或 checkpoint 放入仓库。
3. 可借鉴 E2E-LOAD 的 stream buffer：把 raw-frame video backbone 产出的 token 缓存起来，再接 CausalProj / ActionFormer-style head。
4. 更稳的中间路线是“在线推理图包含 backbone，但 backbone 冻结或只训练 adapter”。这能减少对离线特征文件的依赖，但严格说仍不是完全端到端。
5. 真正 raw-frame 端到端会显著增加显存和训练成本，应只在远端 Slurm 环境做正式训练。
6. 评测必须继续遵守本仓库红线：不使用测试 GT、teacher cache、raw prediction shortcut 或隐藏缓存决策；同时补充在线 latency/FPS、proposal emission 时机和 AEDT 一类响应性指标。

## 可行研究方向

短期可做：

- 继续把 CausalTAD 做成强特征级在线基线，对外表述为 causal/online TAD over pre-extracted video features。
- 记录所有配置的数据形态：I3D、InternVideo2、SlowFast、AudioSlowFast 等都属于特征输入。
- 若要比较 E2E-LOAD，只把它作为 raw-frame online backbone/buffer 参考，不把它当成 On-TAD baseline。

中期可做：

- 增加 `FrameWindowDataset` 或类似数据入口，仅加载帧路径和 annotation，不缓存特征文件。
- 增加 `BackboneWrapper` 的在线缓存模式：每个 step 只编码新增帧/clip，并把历史 compact token 交给 CausalProj。
- 先冻结 backbone 验证在线端到端推理图，再逐步开放 adapter / later blocks 训练。

长期可做：

- 设计 raw-frame CausalTAD：video backbone + stream/token cache + causal temporal neck + instance-level boundary head。
- 对照离线 end-to-end TAD 的 memory-saving 技术，如 low-fidelity encoder、adapter、reversible/activation checkpointing。
- 用 On-TAL 任务指标和 TAD mAP 同时评价，避免只做 OAD 帧级分类。

## 主要来源

- [E2E-LOAD: End-to-End Long-form Online Action Detection, arXiv](https://arxiv.org/abs/2306.07703)
- [E2E-LOAD, ICCV 2023 CVF paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Cao_E2E-LOAD_End-to-End_Long-form_Online_Action_Detection_ICCV_2023_paper.pdf)
- [E2E-LOAD GitHub](https://github.com/sqiangcao99/e2e-load)
- [LSTR: Long Short-Term Transformer for Online Action Detection, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/08b255a5d42b89b0585260b6f2360bdd-Paper.pdf)
- [OadTR: Online Action Detection with Transformers, arXiv](https://arxiv.org/abs/2106.11149)
- [SimOn: A Simple Framework for Online Temporal Action Localization, arXiv PDF](https://arxiv.org/pdf/2211.04905)
- [A Sliding Window Scheme for Online Temporal Action Localization, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136940640.pdf)
- [MATR: Online Temporal Action Localization with Memory-Augmented Transformer](https://skhcjh231.github.io/MATR_project/)
- [MATR, arXiv HTML](https://arxiv.org/html/2408.02957v1)
- [HAT: History-Augmented Anchor Transformer for Online Temporal Action Localization, arXiv](https://arxiv.org/html/2408.06437v1)
- [OnPoint: Offline-to-Online Multi-Level Distillation for Point-Supervised Online Temporal Action Localization](https://arxiv.org/html/2607.00289v1)
- [An Empirical Study of End-to-End Temporal Action Detection, arXiv](https://arxiv.org/abs/2204.02932)
- [End-to-End Temporal Action Detection with 1B Parameters Across 1000 Frames](https://zhao-chen.com/publication/adatad/)
