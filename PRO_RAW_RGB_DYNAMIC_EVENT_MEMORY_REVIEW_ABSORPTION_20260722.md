# Raw-RGB Dynamic Event Memory On-TAD 深度审判吸收记录

## 1. 来源与可见性

- 审判原文：`Raw-RGB Dynamic Event Memory On-TAD 独立深度审判`；
- 附件路径：
  `C:/Users/skywalker/.codex/attachments/479ced54-d744-478e-8fe7-6da0b4f258a8/pasted-text.txt`；
- 原文 SHA-256：
  `ACA5BB6E9950993F170922250F6C915D41AF72BE027984406714315A92D96B8D`；
- 原问题：`PRO_RAW_RGB_DYNAMIC_EVENT_MEMORY_REVIEW_PROMPT_20260722.md`；
- 本次吸收时的本地分支：`codex/ontad-rgb-event-memory`；
- 本次吸收前本地 HEAD：
  `3cd68c134ae06f29741e766478015d4ac58621ce`。

审判者没有读到目标分支和目标版本的九项关键材料，只读到了公开旧分支、论文和
官方 donor。因此它不是对当前本地实现的完整代码审计，而是一次有价值的任务、先例、
方法显然性和实验可识别性审判。

本次吸收重新核验了一手来源：

- [ActionSwitch](https://arxiv.org/abs/2407.12987) 明确以 On-TAL、同时动作和同类
  重叠为目标；
- [MATR](https://arxiv.org/abs/2408.02957) 明确使用当前片段预测 end、memory 估计
  start；
- [HEM](https://arxiv.org/abs/2508.04546) 是文本查询驱动的 OnVTG，而不是本项目的
  closed-set fully supervised On-TAD；
- [E2E-LOAD](https://arxiv.org/abs/2306.07703) 和
  [StreamFormer](https://arxiv.org/abs/2504.20041) 支持 raw/streaming backbone
  先例，但任务主要是 OAD/streaming representation；
- [OpenHOUSE](https://arxiv.org/abs/2509.12145) 已把 On-TAL 与自由文本层级描述结合；
- [Backtrace Mamba](https://ojs.aaai.org/index.php/AAAI/article/view/38139) 和
  [OASIS](https://arxiv.org/abs/2604.17052) 已覆盖宽泛的层级/自适应流视频记忆表述。

这些来源支持审判对“宽方案显然组合”的攻击，但不证明 On-TAD-specific owner
learning 已经无效；后者仍必须由直接控制实验裁决。

## 2. 总裁决

对审判回答本身的裁决是：

```text
格式覆盖：COMPLETE
主题覆盖：NEAR-COMPLETE
目标版本证据：INCOMPLETE
算法闭合度：INCOMPLETE
建议采纳：PARTIAL / REVISE
```

它按要求给出了 A--L 全部章节，也基本逐项回答了任务、竞品、C1--C6、OpenTAD
边界、A/B/C/D、状态机、raw-RGB、实验 DAG、论文和两周计划。它没有完整回答“当前
项目究竟已经实现了什么、现有设计与代码是否正确、现有正式负结果如何改变方案”，
因为这些正是它未能读取的材料。

所以不能把 `REVISE` 机械改写成路线已经被否定，也不能把它的新状态机和超参数直接
当成冻结设计。正确用法是：接受其最强拒稿攻击和实验分解，独立修复其中的矛盾，
再用最低成本模型实验裁决。

## 3. 对原问题的覆盖性复核

| 原问题块 | 覆盖情况 | 仍缺少或回答不充分之处 |
|---|---|---|
| 标准 On-TAD 边界 | **完整** | 正确区分标准最终区间、内部 birth 和 OnVLLM；但没有结合本地现有输出合同逐文件审计。 |
| 完全一致竞品 | **近完整** | 给出 10 个以上方法和组合式查新；但不少格子仍是推断，且没有读目标模型，不能判定当前实现是否已有额外 delta。 |
| C1--C6 claim | **完整但结论有张力** | 一方面把 start-owned ownership 判为 `OBVIOUS`，另一方面又直接把它列为主贡献；缺少真正非显然的学习机制。 |
| OpenTAD/接口边界 | **完整** | 原则正确；但把四个 donor 的完整 parity 都设为核心实验前置，会把接口工作变成模型瓶颈。 |
| A/B/C/D 融合 | **近完整** | 正确否定 ABCD soup，并提出 `K×O`；但 `K` 轴仍混合了数据结构、语义容量和物理上限，需要重新定义。 |
| 状态机和失败模式 | **部分完整** | 覆盖 cancel、late birth、orphan end、overflow；但 late-birth 训练、同类交换不可辨识性、取消后再出生没有闭合。 |
| 训练、推理和 loss | **部分完整** | 给出合理骨架；但固定损失权重没有依据，且伪代码错误地令 `end=t`。 |
| 动态记忆 | **部分完整** | 正确区分 active-event 与 visual history；但 `K=32`、`B=128`、`32+96` 是未画像的人工常数，不能冻结。 |
| raw-RGB 路线 | **近完整** | frozen/adapter/joint 阶梯合理；MViTv2-S 是可用候选，不是已经证明的唯一最优选择。 |
| 实验 DAG/预算 | **格式完整、科学上需修订** | 因子思想强；30k updates、单 seed 3407、效应阈值和 GPU 预算均未由当前方差与收敛曲线支持。 |
| 外部集和论文 | **部分完整** | FineAction 合理但仍需 annotation support audit；标题和 venue 在核心算法未成立前不应冻结。 |

## 4. 完全采纳的建议

1. 主任务保持标准、closed-set、全监督、严格因果 On-TAD。
2. 权威基准输出恢复为 `{start, end, class, score}`；`event_id` 是内部身份和审计元数据。
3. provisional start/class 保留为模型状态和附加 birth-delay 诊断，不冒充标准最终预测。
4. OnVLLM 从本论文主线移出，保留为独立后续课题。
5. 当前没有单篇 exact match，但 ActionSwitch、MOTR/TrackFormer、HEM/Backtrace
   Mamba、E2E-LOAD/StreamFormer 构成强组合式显然性攻击。
6. “dynamic event memory”“raw RGB”“接口”“因果审计”“Pareto 报告”都不能单独
   作为 headline innovation。
7. C3 只作架构分解，C5 只作最终证据层，C6 只作评测纪律。
8. 官方 donor 保持完整只读、native-first；共同接口只能做无损时间和结果映射。
9. 不做 ABCD 模块汤；核心模型必须用直接父模型和 matched-feature 控制来归因。
10. `0.5` 只保留为统一控制臂；主决策只允许在训练侧 calibration 上冻结。
11. feature-only 结果不能支持 raw-RGB claim；正式 raw joint training 必须由核心机制
    存活来授权。
12. cancel、late birth、orphan end、overflow、immutable commit 和错误关联必须成为
    显式指标，不能靠后处理隐藏。
13. active-event state 与可压缩 visual-history memory 必须分开计量；两者都要计入总
    资源，而不是把 active events 从成本中消失。

## 5. 带条件采纳的建议

### 5.1 `K×O` 因子实验

接受因子思想，但把轴重新定义为：

```text
K0 = 固定预分配 query/switch bank；活动数可变但语义容量预先存在
K1 = birth-allocated packed event set；有效活动数随样本变化

O0 = 每个 prefix 允许重新匹配/重分配身份
O1 = birth-time assignment 后 sticky owner 一直持有到 cancel/end
```

四臂使用相同的物理安全上限、encoder、decoder depth、参数预算、样本暴露和成功更新。
`K0O1` 必须等价于一个可信的 Temporal TrackFormer 控制；`K1O1` 才是候选方法。
如果做不到这一点，`K` 只是在比较 Python ragged list 和 padded tensor，不是科学问题。

### 5.2 有界资源

接受“部署一定需要硬上限和 fail-closed”，不接受未经画像的 `K_runtime=32`。候选模型
学习的是有效活动事件数和保留策略；物理 guard 由训练/验证并发分布、压力测试和硬件
预算确定，并报告触发率。不得把 guard 重新包装成固定 semantic slot。

### 5.3 late birth

接受合法 late-birth 状态，但必须增加独立训练通道：对已经开始、当前仍进行、尚未拥有
owner 的 GT 做 prefix-visible matching 或受控 missed-birth corruption。只允许用已见
历史估计 start，并同时记录真实 `birth_time`；不能把迟到出生伪装成更早时刻已发出。

### 5.4 raw backbone

E2E-LOAD 风格 MViTv2-S 是优先候选，不是冻结答案。先用一次小规模 prefix、gradient、
cache 和 latency smoke 比较可用性；StreamFormer 或其他严格因果视觉主干只在主候选
不满足精度/成本时进入对照。raw 接口准备可与 feature 核心并行，昂贵联合训练仍受门禁。

### 5.5 memory `H×R`

接受 ownership 通过后再测试 hierarchy 与 learned retention。`B=128`、recent 32、
archive 96 只可作 profile 起点；正式值必须来自固定总预算 sweep，并要求 train/deploy
使用同一硬选择。主动事件锚点不得静默压缩，但其资源必须单独计入总表。

### 5.6 数字 gate

`+0.7 mAP`、`+3` association points、`1.10×` latency、30k updates 和给定 loss
权重都只可作为候选 preregistration。正式 gate 必须先用父模型方差、学习曲线、单轮
吞吐和显存画像确定，不能把审稿人的示例数字当成定律。

## 6. 明确不采纳的建议或实现细节

1. **不接受把公开分支冻结设成唯一最高科学优先级。** 最小 immutable manifest 和
   commit 很重要，但应在数小时内与模型工作并行完成，不能阻塞核心假设实验。
2. **不接受四个 donor 全部 fidelity/parity 完成后才允许核心实验。** ActionSwitch、
   MATR 和 Temporal TrackFormer 是直接门；HAT/HEM 可在 memory claim 启动时再完整跑。
3. **不接受伪代码中的 `end=t`。** `t` 是决定/提交时刻；模型必须从已见证据独立解码
   `end_hat <= source_time <= emit_time`，否则 commit delay 会系统性拉长区间。
4. **不接受“start-owned persistent state”本身已经足够创新。** MOTR/TrackFormer 已
   形成强先例。论文需要新的 On-TAD-specific birth/ownership learning rule，并证明它
   解决无空间几何、同类重叠、迟到出生和区间监督下的关联问题。
5. **不接受只凭一个单种子固定步数失败就杀死全部路线。** 先满足收敛、实现、容量和
   父模型方差门；随后单种子可作资源 kill，保留 claim 仍需多种子 paired uncertainty。
6. **不接受把 provisional start 从模型中删除。** 只从标准 authoritative output 中
   删除；它仍是事件出生机制和 birth-delay 诊断的核心。
7. **不接受冻结 `lambda`、`K=32`、`B=128`、30k updates、seed 3407 或论文标题。**
   它们没有当前项目证据。
8. **不接受两周计划前五天主要做接口工程。** 最小复现和合同测试与模型核心并行，
   模型归因永远优先于完善框架。

## 7. 审判中未解决的两个关键算法问题

### 7.1 同类重叠的可辨识性

若两个同类动作在当前前缀具有完全相同的可见证据，标准 segment 标注没有人物/物体
track ID，则哪一个未来先结束在当下可能本来就不可辨识。模型不能“保证”解决信息论上
不可辨识的对应。正式方法必须：

- 对可辨识样本用 event trajectory 和 owner-conditioned hazard 学习对应；
- 对交换等价的 GT 使用 permutation-aware matching/metric，避免惩罚任意标签交换；
- 单独报告有可见区分证据与无可见区分证据的样本；
- 不把未来 end 顺序喂给运行时 assignment。

### 7.2 late birth 与 cancel 后再出生

审判的 newborn matching 只接收“start 落在当前单元”的 GT，却又允许 late birth，
两者不一致。还需要明确：一个已匹配事件错误 cancel 后，原 GT 是否允许重新出生；若
允许，如何避免 duplicate；若不允许，如何承担召回损失。该策略必须进入训练暴露和
错误表，不能只写状态名字。

## 8. 吸收后的冻结边界

冻结：

- 最终目标仍是 original-RGB、标准全监督、严格因果 On-TAD；
- feature 只作机制归因，不能代替 raw 结果；
- 最终标准输出 `{start,end,class,score}`，event identity 为内部审计；
- start-owned ownership 是待证核心问题，不是已成立创新；
- donor 只读，接口最小化，不做 ABCD soup，不访问 report split；
- 统一 `0.5` 是消融，不是主规则。

不冻结：

- 论文方法名和标题；
- raw backbone；
- 物理 active-event guard 与 visual token budget；
- loss 权重、训练步数、效应阈值和 seed 数；
- learned retention 是否成为贡献；
- FineAction/MUSES/MultiTHUMOS 的最终外部集选择。

## 9. 吸收后的下一步

1. 用同一路径实现并核验修订后的 `K×O` 四臂，显式包含 Temporal TrackFormer 控制；
2. 同时做最小 ActionSwitch/MATR native fidelity、synthetic overlap/late-birth/
   cancel/rebirth/future-perturbation 测试和轻量 raw causal backbone smoke；
3. 先画像再冻结训练步数、资源上限和效应 gate；四臂并行跑 feature-level single-seed
   screen，绝不以旧 FIXED `1.614484` mAP 的弱负基线作为唯一成功标准；
4. 只有 `K1O1` 同时超过 `K0O1` 和 `K1O0`，并减少 wrong-start/owner-swap，才启动
   `H×R` memory 因子和正式 raw frozen/adapter/joint；
5. 多种子、外部集和 report-once 只运行存活模型。

这一路线接受审判最有价值的“收窄和可识别性”建议，同时拒绝把工程完整性、任意常数
和自然组件迁移误当成最终算法。
