# Pro Prompt: First-Principles Divergent Research-Idea Generation for Streaming Video

## 使用方式

将 `BEGIN PROMPT` 到 `END PROMPT` 之间的全部内容原样提交给具备联网能力、长上下文和高推理预算的 Pro 模型。不要在提交前补入某条偏好的技术路线。

---

## BEGIN PROMPT

你是一名对 CVPR/ICCV/ECCV/NeurIPS/ICLR/ACL 顶级论文负责的资深研究者、Area Chair 和极其苛刻的研究选题编辑。你的任务不是替现有方案辩护，也不是给当前代码增加模块，而是从第一性原理重新寻找：

> 在 Online Temporal Action Detection/Localization、online video temporal grounding、streaming video understanding/reasoning 及其合理邻域中，2026 年仍然真正重要、尚未被充分解决、具有不可替代技术核心，并且能被严谨实验验证的研究问题和方法。

本次工作是**完全重新选题**。已有项目、术语、模型和候选想法都只是可能带偏你的证据，不是约束条件。你必须允许以下结论：

- 传统 On-TAL 已不值得继续；
- 当前代码应全部舍弃；
- 最有价值的问题不属于 On-TAL；
- 最优贡献是新任务、数据、理论、评价或系统，而不是新网络；
- 经过查新后不存在足够强的可行 idea，应明确给出 `NO-GO`。

不要默认任何一项成立：

- 不默认必须检测 `[start, end, class]`；
- 不默认动作类别是封闭集合；
- 不默认人工动作边界是正确监督；
- 不默认每个动作都会成功完成；
- 不默认查询在视频开始前给定；
- 不默认模型每帧都要运行；
- 不默认模型只能在线推理而不能在线学习；
- 不默认视觉是唯一模态；
- 不默认 mAP/tIoU 是正确目标；
- 不默认深度模型、VLM、LLM、RL、memory、tracking 或 Transformer 是必要方案；
- 不默认当前仓库必须被复用；
- 不默认“实时”“在线”“低延时”“端到端”已有统一、正确的定义。

### 一、公开代码审查锚点

你必须先打开并审查以下公开仓库，而不是仅依赖本 Prompt 的总结：

- Repository: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702
- Target branch: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/tree/codex/online-tad-clean-20260702
- Visible audit commit: https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/commit/bfd0608b2996cba30d158d741ee193476a5078df
- Branch name: `codex/online-tad-clean-20260702`
- Visible HEAD: `bfd0608b2996cba30d158d741ee193476a5078df`

重点检查但不限于：

- `opentad/models/detectors/pceh_ontad.py`
- `opentad/utils/online_protocol.py`
- `opentad/cores/train_engine.py`
- `opentad/cores/test_engine.py`
- `configs/causaltad/`
- `tests/test_online_emission_protocol.py`
- `tests/test_pceh_incremental_detector.py`
- `tools/smoke_pceh_stream.py`

代码审查只用于理解现状和识别可复用资产。**第一轮 idea 生成时禁止以“最容易利用现有代码”为筛选标准。**代码复用只能在最终可行性排序阶段作为次要因素。

### 二、当前项目已知事实

下面是需要核验、可以推翻但不能无视的事实：

1. 当前仓库更接近 strict-prefix/causal streaming 骨架，不是已完成的论文级 On-TAL 模型。
2. 尚无正式多随机种子训练结果，不得把 smoke test 当作科学证据。
3. PCEH 路线曾试图区分 endpoint 与 emission/commit，但当前审查发现：
   - endpoint 与 emission target 由同一 crossing event 产生；
   - 晚到 prefix 被重复标成 emission positive；
   - target 按 class 聚合而非 instance-aware；
   - predicted end 曾与 emit time 退化为同一量；
   - GT/duration/video end 接近模型 kwargs，需要 taint audit。
4. 正式训练前的最低科学正确性修复包括：instance-aware risk set、first-event/first-commit target、endpoint freeze、post-event masking、GT taint audit，以及 `pred_end < commit` 的 delayed synthetic case。
5. 已观察/估计的完整 packet 训练成本约为：
   - `152,670` packets/epoch；
   - 约 `7 h/epoch`；
   - 约 `210 GPU-hours/model/seed`（30 epochs）；
   - 双模型、三随机种子可能约 `1,260 GPU-hours`。
6. 冻结特征缓存、事件片段训练、LoRA 等只能控制成本，不自动构成创新。
7. 先前的 identity-preserving belief trajectory、revision ledger、utility commit 路线已被内部降级为实现底座/保底对照，因为其组合创新性不足。
8. 最近提出的 “physical endpoint != first sufficient evidence time” 与 sequential risk control 也只是一个**低置信候选**。你必须像攻击其他想法一样攻击它，不得将其作为默认答案。

### 三、已完成的竞争调研

你必须重新核验这些论文，并继续检索截至 **2026-07-11** 的最新工作。以下列表不是完整 related work，而是已知的创新禁区和邻近威胁。

#### 3.1 直接 On-TAL / instance-level online localization

| 工作 | 已占据的核心 | 对新 idea 的约束 |
|---|---|---|
| [CAG-QIL, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Kang_CAG-QIL_Context-Aware_Actionness_Grouping_via_Q_Imitation_Learning_for_Online_ICCV_2021_paper.html) | MDP、decision history、actionness grouping、start/end 决策 | 不能把 sequential state 或 MDP 当创新 |
| [OAT, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2307_ECCV_2022_paper.php) | sliding window、early proposal、online boundary refinement、online suppression、early detection metric | 不能把早期 proposal/边界修正当创新 |
| [SimOn, 2022](https://arxiv.org/abs/2211.04905) | 轻量 sequential On-TAL、过去视觉/预测 context、start/end grouping | 不能把简单 recurrent/Transformer causal head 当创新 |
| [MATR, ECCV 2024](https://arxiv.org/abs/2408.02957) | 长期 memory、当前片段估计 end、历史 memory 估计 start | 不能把长期记忆和起止解耦当创新 |
| [HAT, ECCV 2024](https://eccv.ecva.net/virtual/2024/poster/2201) | history-augmented anchors | 不能把 history enhancement 当创新 |
| [ActionSwitch, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01621.pdf) | 显式 state switch、class-agnostic、同类与并行动作、conservativeness | 不能把状态变化、同类/并发处理当创新 |
| [OZ-TAL, 2026](https://arxiv.org/abs/2605.09976) | online zero-shot TAL、off-the-shelf VLM、unseen actions、bias mitigation | 不能把 VLM + zero-shot/unseen On-TAL 当创新 |
| [OnPoint, 2026](https://arxiv.org/abs/2607.00289) | point-supervised On-TAL、offline-to-online multi-level distillation | 不能把弱监督或离线教师蒸馏本身当创新 |

#### 3.2 在线分割、进度、层次事件与 grounding

| 工作 | 已占据的核心 |
|---|---|
| [ProTAS, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Shen_Progress-Aware_Online_Action_Segmentation_for_Egocentric_Procedural_Task_Videos_CVPR_2024_paper.pdf) | causal online action segmentation、ongoing progress、task graph、用进度修正预测 |
| [OnlineTAS, NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/hash/6c6c5fccf3c8661fcae219be7ca226f7-Abstract-Conference.html) | causal segmentation、adaptive memory、online correction/post-processing |
| [OpenHOUSE, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Kang_Open-ended_Hierarchical_Streaming_Video_Understanding_with_Vision_Language_Models_ICCV_2025_paper.html) | On-TAL + free-form hierarchical descriptions、progress-based class-agnostic boundaries、按需调用 VLM |
| [Hierarchical Event Memory / OnVTG, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_Hierarchical_Event_Memory_for_Accurate_and_Low-latency_Online_Video_Temporal_ICCV_2025_paper.html) | online temporal grounding、event proposals、层次长期记忆、低延时 start/end、future branch |
| [AViLA, 2025](https://tanveer81.github.io/publication/avila/) | query-evidence asynchrony、历史/当前/未来证据、time-aware trigger |

#### 3.3 流式视频基础模型、记忆与何时回答

| 工作 | 已占据的核心 |
|---|---|
| [StreamFormer, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Yan_Learning_Streaming_Video_Representation_via_Multitask_Training_ICCV_2025_paper.pdf) | causal streaming backbone、图像预训练模型因果化、global/temporal/spatial multitask learning、TAL/TVG/OAD/VQA |
| [LiveStar, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/2ce4f0b8e24c45318352068603153590-Paper-Conference.pdf) | streaming assistant、增量 video-language alignment、response-silence decoding、在线 temporal localization、KV cache |
| [Thinking-QwenVL, ICLR 2026](https://arxiv.org/abs/2604.18459) | compact causal state、progress/confidence、first-sufficient-evidence timing、transparent decisions |
| [StreamReady, 2026](https://openreview.net/forum?id=KXge8OA222) | readiness-aware streaming VideoQA、early/late penalty、何时证据充分 |
| [SelectStream, 2026](https://arxiv.org/abs/2606.16353) | budgeted online latent evidence allocation、selective write/consolidate/retrieve、fixed-capacity memory |
| [StreamForest, NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/file/6dd91fec726dbed8915a1fbadd91d1d2-Paper-Conference.pdf) | efficient online video understanding、stream-oriented data/model design |
| [TemporalVLM, ACL 2026](https://aclanthology.org/2026.findings-acl.70/) | 长视频 temporal reasoning、grounding、segmentation、工业流程数据 |

#### 3.4 端到端、蒸馏与计算效率

| 工作 | 已占据的核心 |
|---|---|
| [E2E-LOAD, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Cao_E2E-LOAD_End-to-End_Long-form_Online_Action_Detection_ICCV_2023_paper.html) | raw-video end-to-end OAD、长短期 cache、短历史训练/长历史推理、实时推理 |
| [ETAD, CVPRW 2023](https://openaccess.thecvf.com/content/CVPR2023W/ECV/html/Liu_ETAD_Training_Action_Detection_End_to_End_on_a_Laptop_CVPRW_2023_paper.html) | sequential backprop、selective snippet gradients、proposal sampling、低显存端到端 TAL |
| [SAN, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Foo_System-Status-Aware_Adaptive_Network_for_Online_Streaming_Video_Understanding_CVPR_2023_paper.pdf) | 随硬件负载动态选择分辨率/深度、在线低延时、设备适配 |
| [Offline-to-Online Streaming Distillation, WACV 2026](https://openaccess.thecvf.com/content/WACV2026/papers/Patel_Distilling_Offline_Action_Detection_Models_into_Real-Time_Streaming_Models_WACV_2026_paper.pdf) | 离线 ViT 教师到实时 streaming student、causal attention、uncertainty-guided distillation |

#### 3.5 已知但不能默认正确的空缺

以下只是需要审查的开放问题，不是答案：

- 真实动作的物理边界、语义完成、可观测证据时刻是否被错误地混成一个时间？
- 无限视频流中的 repeated testing 是否需要 anytime-valid false-alarm/risk control？
- “online” 是否长期被错误地用来表示“离线训练、在线推理”，而非持续学习？
- 固定封闭类别、成功完成动作、单一视觉模态、固定查询、固定帧率是否过度简化现实？
- 当前 benchmark 是否只是把离线 THUMOS/ActivityNet 重新播放，而没有真实 streaming 交互、异步事件、资源波动或反馈？
- 一个实时系统真正需要的是完整 span、状态变化、完成结果、异常、干预时机、因果证据，还是别的对象？

你必须对这些问题逐一质疑，而不是全部接受并堆成一个大系统。

### 四、严格禁止的低质量选题模式

出现以下任一种情况时，默认淘汰，除非你能证明存在不可替代的新问题和机制：

1. `现有 backbone + memory + boundary head + 新 loss`。
2. 把 OAT、MATR、ActionSwitch、ProTAS 的模块重新组合。
3. 在旧任务上加入 VLM/LLM/RL/SSM/Mamba/Diffusion 但没有解释其必要性。
4. 把 engineering polish、缓存、LoRA、mixed precision 或更快 DataLoader 当核心创新。
5. 仅换数据集、仅换监督强度、仅换 backbone、仅换指标名字。
6. 仅提出“更准确、更低延时、更鲁棒”，没有定义真实失败和机制。
7. 把 generic memory、generic readiness、generic open-vocabulary、generic distillation 当创新。
8. 依靠未来帧、离线 NMS、EOF 后处理或 oracle state，却声称严格在线。
9. 用更少输出或降低 recall 伪造低延时。
10. 只给 paper story，不给可证伪的核心 claim 和最小实验。
11. 把多个弱 idea 合并成一个看起来宏大的系统。
12. 声称“首个”或“无人区”但没有逐项检索和直接文献证据。

### 五、无假设工作协议

#### Phase 0：建立 Unknown Register

先列出所有会改变选题结论、但无法从代码和文献确定的未知项。每项必须标记：

- `known from source`；
- `inferred`；
- `unknown`；
- `requires user/domain expert`；
- `requires pilot experiment`。

禁止静默补全未知信息。不要因为未知就停止；为关键未知建立条件分支。

#### Phase 1：重新审查任务是否值得存在

分别审查：

1. closed-set On-TAL；
2. open-vocabulary/zero-shot On-TAL；
3. online temporal grounding；
4. online action segmentation；
5. streaming VideoLLM reasoning；
6. continual/adaptive video learning；
7. active perception / compute-aware sensing；
8. outcome-, failure-, interruption- or intervention-aware action understanding。

对每个任务回答：

- 谁在真实世界中需要它？
- 当前输出对象是否对应真实决策？
- 现有 benchmark 是否测到了真实困难？
- 2026 年继续优化该任务的边际价值是什么？
- 哪些失败不能靠扩大模型、增加 memory 或更多数据解决？
- 应该保留、重定义、合并，还是放弃该任务？

#### Phase 2：Assumption Demolition

至少拆除并重新评估以下假设：

- action 是单标签、单粒度、非重叠的；
- action 有清晰 start/end；
- endpoint 等于 completion；
- completion 等于 success；
- query 和 taxonomy 预先已知；
- 视频永远连续且帧率固定；
- 所有帧同等值得计算；
- 模型不会在部署期间学习；
- 人工边界是无噪声真值；
- mAP 是最终效用；
- 低延时是唯一代价；
- 单视频预测彼此独立；
- 用户只需要检测结果而不需要证据、置信度或干预。

输出一张表：`assumption -> where it fails -> real consequence -> research opportunity -> closest prior work`。

#### Phase 3：先完全发散，禁止提前收敛

在看代码复用成本之前，生成**至少 36 个彼此独立的原始 idea**。必须覆盖下列六个镜头，每个镜头至少 6 个：

1. **Task/benchmark lens**：改变任务输入、输出、监督、交互或评价。
2. **Learning lens**：新的预训练、后训练、continual/test-time/weak/self-supervised learning 问题。
3. **Sequential decision/theory lens**：optimal stopping、change detection、uncertainty、risk、calibration、information theory、causal inference。
4. **Perception/system lens**：active sensing、动态帧率、异步计算、硬件波动、边云协同、能源/隐私预算。
5. **Semantics/world lens**：组合动作、结果、失败、中断、恢复、层次、因果链、多人多事件、多模态。
6. **Application-first lens**：机器人协作、辅助医疗/养老、工业流程、安全监控、实时体育、AR 助手等真实决策闭环。

发散约束：

- 至少 40% 的 raw ideas 不得是传统 On-TAL 网络改造；
- 至少 25% 必须允许放弃固定 `[start,end,class]` 输出；
- 至少 20% 必须显式借鉴视频之外的领域，但要解释为什么可迁移；
- 至少 6 个 idea 必须是 task/data/evaluation/theory 主贡献，而非模型主贡献；
- 至少 6 个 idea 必须能在不训练大型视频模型的条件下完成第一次证伪；
- 至少 3 个 idea 必须以“可能不应该做检测”为出发点；
- 不得用同一个核心 idea 换名字凑数量；
- 不得在这一阶段排名或合并 idea。

每个 raw idea 只写一张简洁卡片：

```text
ID:
Problem failure:
Who cares and why:
New task or mechanism:
Why a larger model/memory is insufficient:
Closest known threat:
Fastest falsification:
```

#### Phase 4：扩展查新和去重

对 36 个 raw ideas 聚类去重，并针对每个独立核心执行新检索。检索必须覆盖：

- CVF Open Access；
- ECVA；
- NeurIPS proceedings；
- OpenReview；
- ACL Anthology；
- arXiv；
- 相关数据集和公开代码仓库。

技术事实优先引用论文、官方页面和官方代码。不得用搜索摘要、博客或二手综述替代关键证据。

每个 surviving idea 至少给出：

- 最接近的 3--5 篇工作；
- 它是否只是 `A + B`；
- 已有工作已经完成了哪一部分；
- 剩余 delta 是否足以支持一篇论文；
- 查新置信度；
- 仍未覆盖的检索盲区。

#### Phase 5：强制 Kill Round

你同时扮演最苛刻的 Senior PC。对每个 surviving idea 写出最强拒稿理由，并使用以下规则：

- 若核心贡献可以用两篇已有论文的直接组合解释，淘汰；
- 若只改变工程实现、不改变科学问题，淘汰；
- 若只能在不公平 backbone、更多数据或更多算力下赢，淘汰；
- 若核心 claim 不能被一个决定性实验隔离，淘汰；
- 若需要多个尚不存在的大型数据集和模型才能开始验证，降级；
- 若真实用户价值不清楚，淘汰；
- 若方法失败后仍能靠重写 story 生存，说明 claim 不可证伪，淘汰；
- 若最合理结论是已有任务不值得继续，明确保留 `NO-GO/ABANDON` 结论。

每个 idea 必须获得：`KILL / HOLD / SURVIVE`。

#### Phase 6：独立评分，而不是凭审美挑选

只对 `SURVIVE` idea 使用 100 分制：

| 维度 | 权重 |
|---|---:|
| 问题真实重要性与潜在影响 | 20 |
| 相对最新工作的实质新颖性 | 25 |
| 核心机制的必要性与不可替代性 | 15 |
| 任务定义和 claim 清晰度 | 10 |
| 最小实验的可证伪性 | 10 |
| 当前算力预算内的可行性 | 10 |
| 数据/标注可行性 | 5 |
| 顶会叙事完整性 | 5 |

额外扣分：

- 模块堆叠：`-10` 至 `-25`；
- 依赖未验证大规模训练：`-5` 至 `-20`；
- 与 2025--2026 工作高度重叠：`-10` 至 `-30`；
- 仅有 benchmark novelty、没有可推广科学问题：`-5` 至 `-15`；
- 仅有应用包装：`-10` 至 `-20`。

除非总分至少 `80/100`，且“新颖性”“重要性”“可证伪性”均不低于各自满分的 70%，否则不得推荐为主线。

### 六、最终输出要求

严格按下面结构输出，不得直接跳到一个过度自信的方案。

#### A. Executive Verdict

- 传统 On-TAL 是否还有足够研究价值；
- online temporal localization 哪些部分有价值；
- generic online video reasoning 是否已经过度拥挤；
- 是否建议继续、重定义任务、转向邻域，或直接停止；
- 结论置信度与最大未知项。

#### B. Repository Reality Check

- 当前代码实际实现了什么；
- 没实现什么；
- 哪些资产可复用；
- 哪些历史设计会锚定或限制新 idea；
- 不得把未推送的本地信息当成 GitHub 事实。

#### C. Field and Assumption Map

- 任务谱系；
- 已占据创新区域；
- assumption demolition 表；
- 真正未解决的问题，不少于 10 项。

#### D. Raw Divergence Portfolio

- 完整列出至少 36 个 raw idea cards；
- 保留被淘汰 idea，不能只展示成功者；
- 标记其所属 lens，但暂不排序。

#### E. Deduplication and Novelty Audit

- 聚类图；
- 每簇核心问题；
- 最近竞争工作及直接链接；
- `A+B` 风险；
- 查新盲区。

#### F. Kill Matrix

至少包含：

```text
Idea ID
Strongest rejection
Closest prior-work combination
What evidence could rescue it
KILL / HOLD / SURVIVE
```

#### G. Ranked Top 5

只从 `SURVIVE` 中选择。每个 Top-5 idea 必须完整说明：

1. 一句话论文命题；
2. 要解决的真实失败，而非抽象 gap；
3. 精确任务定义：输入、输出、时间/因果约束、监督、评价；
4. 与现有任务相比改变了什么；
5. 单一主贡献和至多一个辅助贡献；
6. 最小必要方法，具体到表示、状态、损失或决策规则；
7. 为什么不是模块拼接；
8. 3--5 个最接近竞争工作及逐项 delta；
9. claim map：每条 claim 对应什么证据；
10. 三块以内的核心实验；
11. 最危险 baseline 和公平对比方式；
12. kill criteria；
13. 训练/标注/GPU 成本级别；
14. 适合的 venue；
15. 分数、置信度和剩余查新风险。

#### H. One Recommended Route, One High-Risk Route, One No-Go

- 只推荐一条最值得投入的路线；
- 再给一条高风险高回报路线；
- 明确指出一条看似热门但不应做的路线；
- 若没有 idea 达到门槛，必须输出 `NO-GO`，不得为了完成任务强行推荐。

#### I. 48-Hour Falsification Plan

对唯一推荐路线给出一个无需大规模训练的 48 小时证伪计划：

- 需要读取/整理的数据；
- 最小 prototype 或统计分析；
- 需要比较的强 baseline；
- 成功阈值；
- 立即终止条件；
- 预计 GPU/CPU/人工标注成本。

#### J. Clarification Questions

最后只提出那些**答案会改变 Top-5 排名或 GO/NO-GO 结论**的问题。按信息价值排序，最多 12 个。不要询问可从代码、论文或公开数据中自行查明的问题。

### 七、写作与诚信要求

- 使用中文主报告，论文名、模型名和公式可以保留英文。
- 所有时效性事实必须联网核验。
- 关键 novelty 判断必须附原始论文或官方项目链接。
- 明确区分：文献事实、你的推断、未知项、待实验命题。
- 不得声称绝对新颖；只能报告检索范围内的 novelty confidence。
- 不得为了显得前沿而强行使用 VLM、LLM、RL、Diffusion、world model 或 agent。
- 不得把当前作者偏好、现有代码投入或此前讨论过的路线当作 sunk-cost constraint。
- 你的职责是找到值得做的问题，也包括阻止一个不值得做的项目继续消耗时间和算力。

## END PROMPT
