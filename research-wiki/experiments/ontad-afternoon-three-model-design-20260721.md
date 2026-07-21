# On-TAD 当日三版模型优化设计（2026-07-21）

status: approved-and-executing  
scope: feature-level, fully supervised, strict causal Online-TAD  
deadline: 2026-07-21 21:00 Asia/Shanghai

## 一句话目标

在完全相同的 SigLIP2 stride-8 因果 RGB 特征、fit/calibration 划分、seed
705、一轮训练和固定 0.5 运行阈值下，依次排除三种失败原因：容量不够、
生命周期分数刻度不对、结束边界缺少显式的时序变化建模。任何版本都只看
当前与过去，不读取 reporting split，不做阈值搜索，不做离线 NMS，也不启动
raw-RGB 联合训练。

## 冻结的公共实验条件

| 条件 | 固定值 |
|---|---|
| 输入 | 缓存的 causal SigLIP2 RGB 特征，768 维，stride 8 |
| 训练/诊断 | fit_core 160 / calibration 40 |
| reporting | 锁定，不访问 |
| 比较轴 | 每个版本均成对跑 FIXED 与 REMATCH |
| seed / epoch | 705 / 1 |
| 阈值 | birth、alive、end 全部固定 0.5 |
| 容量语义 | step-entry FREE 才能接 birth；本步释放的槽下一因果决策才能复用 |
| 输出语义 | 动作结束时一次性发射不可回改的最终区间 |
| 资源门 | 同提交 profile，双臂估算不超过 2 GPU·小时；正式 GPU 仅 Slurm |
| 晋级门 | 测试、画像、训练激活、因果、容量、区间发射和 calibration gate 全通过 |

## E：reserve6 容量对照

### 改了什么

在 D 的 birth/alive/end 三头 margin 基础上，只把共享 query 槽从 4 增到 6。
这 6 个槽来自完整冻结划分 census：最多 4 个同时可见实例，加最多 2 个同
一步新生实例。不会为了追结果继续扩槽，也不会把“本步先释放再出生”改成
同一 query 同时描述旧 end 和新 birth。

### 想证明什么

若唯一的两次 entry-free collision 消失且 0.5 下出现完整生命周期发射，说明
D 的主要阻断是过渡容量；若碰撞消失仍零发射，容量只是一项工程性前置条件，
主瓶颈仍在分数刻度或 end 建模。

### 已部署

- exact code：`380bc16947a75d9ca19cfb79e06fdfbab0ae43c0`
- Slurm：`1178040`
- run：`model_opt_reserve_seed705_20260721_093412`

## F：reserve6 + 单调生命周期校准器

### 核心结构

保留 raw birth/alive/end 三个 head，并为每个通道加一个严格单调仿射层：

`calibrated_logit = exp(log_scale) * stop_gradient(raw_logit) + bias`

`log_scale=0, bias=0` 初始化为恒等映射。固定阈值只读取 calibrated logit；
匹配代价、原 BCE、三项 margin 和 query transport 均只读取 raw logit。校准
损失采用“当前步正负类各占一半”的 BCE，只在该步存在正例时激活，三通道
各自独立。`stop_gradient` 保证这项损失只改变 6 个校准参数，不偷偷改 raw
排序。

### 为什么这是科学问题而不是调阈值

D 的 FIXED birth/alive AUC 已达 0.842/0.863，却在冻结 0.5 下完全没有 TPR。
这可能是排序能力存在、概率刻度错误。F 不搜索也不降低阈值，而是让模型用
fit supervision 学出一个保持排序的映射。因此 raw 与 calibrated 的 ROC-AUC
理论上必须相同；若固定 0.5 的越线率改善，说明问题是可学习的刻度失配。

### 证伪标准

- 校准层必须恒等初始化、scale 始终为正；
- calibration loss 对 raw head 无梯度，只对校准参数有梯度；
- FIXED/REMATCH 仍只差 binding；
- raw 与 calibrated 排序/AUC 不变，否则实现失败；
- 若仍无 end 越线或发射，则不再把主要问题归因于阈值刻度。

## G：reserve6 + 当前转移判终 / 历史起点检索

### 核心结构

G 保持 persistent query 和 reserve6，但把 end 从“只看当前 query 的线性头”
改成显式因果转移头：

`end_logit_t = MLP([q_(t-1), q_t, q_t-q_(t-1), x_t])`

其中 `x_t` 是当前已到达的投影特征。它只使用上一状态与当前输入，没有未来
帧。起点不再只依赖 birth 当步的 1-token scalar offset；在 endpoint slot
出现时，使用当前 query 对已经观察到的 feature memory 做 pointer，监督其
指向真实 start 的最近历史 token。若起点已早于有限 memory，则使用 sentinel，
运行时回退到该槽在 birth 时已经冻结保存的 start frame。

### 与前沿在线方法的关系

- MATR 的可取思想是“当前片段负责结束、过去记忆负责起点”；本版只吸收
  这种严格因果分工，不复制其完整网络。
- OpenHOUSE 的启发是相邻动作需要显式边界变化信号；本版不引入其 VLM、
  层级标签或开放词汇设定。
- StreamFormer 说明因果时序表征可由 streaming attention 学习；当前阶段仍
  固定特征，只在轻量 lifecycle head 内验证边界结构，避免与 raw-RGB 主干
  联合变化。
- 本项目先前所谓 ChronoTransport 只是 prediction-only past-to-current
  Sinkhorn 辅助项，已经被 C 实验证伪；G 不继续调其权重。

对应的一手资料：

- MATR: <https://arxiv.org/abs/2408.02957>
- OpenHOUSE: <https://openaccess.thecvf.com/content/ICCV2025/html/Kang_Open-ended_Hierarchical_Streaming_Video_Understanding_with_Vision_Language_Models_ICCV_2025_paper.html>
- StreamFormer: <https://openaccess.thecvf.com/content/ICCV2025/html/Yan_Learning_Streaming_Video_Representation_via_Multitask_Training_ICCV_2025_paper.html>

### 证伪标准

- 因果重放和 chunk invariance 必须通过；截断未来输入不得改变既有输出；
- end transition loss 和 endpoint-only start pointer loss 必须在训练审计中
  真实激活，其余非本版损失不得串扰；
- runtime state 不含 GT，pointer 只能访问 `memory_frames <= current_frame`；
- sentinel 回退必须使用运行时已保存 birth start，不得查询标注；
- 若 end AUC/0.5 TPR/最终发射无改善，则回到更长的因果历史编码或专门的
  boundary-progress 表征，而不是搜索阈值。

## 当日执行顺序

1. 监控并完成 E 的同提交 profile、双臂训练、诊断和 gate。
2. 实现 F 的 head、损失、审计、配置、提交路由与单元/因果测试；远端 exact
   clean 测试和 profile 通过后部署。
3. 实现 G 的 transition end、endpoint-only past-start pointer、运行时回退、
   审计、配置、提交路由与单元/因果测试；远端 exact clean 测试和 profile
   通过后部署。
4. 三版都以独立 run 目录、exact SHA 和 Slurm job 记录到 Wiki。21:00 报告
   代码完成度、部署状态、已完成结果、未完成作业的准确恢复点和下一门。

## 激活与晋级规则

E 预期只激活三项 lifecycle margin；F 预期再激活三项 calibration loss；G
预期激活三项 lifecycle margin、transition-end（由正式 end loss承载）和
endpoint start-pointer loss。技术 gate 与科学 gate 分开记录：程序完整跑完但
零发射属于科学拒绝，不写成系统崩溃。

只有某个特征级版本同时满足容量、因果、固定阈值生命周期发射和校准诊断，
才允许进入多 seed；多 seed 稳定后才讨论 raw-RGB 联合训练。

## 执行记录

### M15 F 实现、远端验证与部署

F 已在 `49b06aff30e8c540550d5d7bd0bd464532a9ed26` 完整接入：head 保留
raw birth/alive/end，并增加恒等初始化的正 scale/bias；匹配、原 BCE、margin
和 transport 全部显式读取 raw；固定阈值/runtime 读取 calibrated。训练审计新增
三项 calibration loss，诊断同时保存 raw/calibrated 分布，并在 logit 空间强制
三通道 pairwise AUC 不变。

- 本地无 Torch 的配置/审计/提交器套件：`19 passed`；Python compile 和
  Bash 语法通过。
- 本机 Torch 因 Windows `c10.dll` 初始化错误不可用；这被记录为本地环境
  限制，不冒充模型失败。
- N16R4 独立 checkout exact `49b06af`：扩展相关套件
  `133 passed in 90.20s`，结束 SHA 不变且 clean。
- Slurm：`1178214`。
- run：`/data/run01/sczc063/yuzibo/runs/persistent_binding/model_opt_calibration_seed705_20260721_095351`。

作业继续先跑 exact 测试、四项画像和双臂 2 GPU·小时门；通过才自动训练。

### M16 G 实现、远端验证与部署

G 已在 `8d8289391ccb74563a385d7fea33fad9c4d4d7de` 完整接入：end
特征为 `[q_(t-1), q_t, q_t-q_(t-1), x_t]` 经轻量 MLP 后进入共享 end
输出层；endpoint start query 只对当前已见 feature memory 做 pointer。训练仅在
endpoint slot 上计算 pointer CE；运行时 pointer 选择 sentinel 时回退到 birth
时已经保存在槽状态中的 start，不读取标注。

诊断新增 endpoint pointer 的预测数、准确率、sentinel 数和 selected source
frame 因果检查；boundary 作业强制两臂都有 endpoint decision、memory 全部不晚
于当前帧、runtime state 不含 GT。激活审计要求三项 lifecycle margin 与
`endpoint_start_pointer_loss` 激活，calibration/transport 保持休眠。

- 本地 CPU-safe 配置/审计/提交器套件：`20 passed`；Python compile 与
  Bash 语法通过。
- N16R4 独立 checkout exact `8d82893`：含因果重放/合规在内的扩展套件
  `154 passed in 93.86s`，结束 SHA 不变且 clean。
- Slurm：`1178279`。
- run：`/data/run01/sczc063/yuzibo/runs/persistent_binding/model_opt_boundary_seed705_20260721_100229`。

至此 E/F/G 三个当日版本均已完成实现并部署；结果仍按各自 profile、训练、
诊断和 scientific gate 到齐后判定。

### M17 注册 H：2×2 因子交互实验

重新读取 D 的完整 calibration 分布后，组合实验具有预先可解释的必要性：

- FIXED birth/alive AUC 为 `0.8422/0.8628`，但正例最大概率仅约
  `0.1984/0.3085`，因此 G 即使改善 end，仍会被 birth/alive 的固定 0.5 门
  截断；
- FIXED/REMATCH end AUC 仅 `0.4885/0.5574`，因此 F 的严格单调映射虽然能
  移动决策刻度，却不可能改变 end 排序。

由此把 reserve6 作为公共底座，注册如下 2×2：

| 版本 | 单调校准 | 因果 boundary | 作用 |
|---|---:|---:|---|
| E | 0 | 0 | 公共容量对照 |
| F | 1 | 0 | 校准主效应 |
| G | 0 | 1 | 边界结构主效应 |
| H | 1 | 1 | 两种机制的交互/完整生命周期 |

H 不增加新阈值、数据或未来信息：raw head 仍负责匹配与原损失，calibrator
负责固定 0.5 的可学习刻度，transition end/past-start 负责边界结构。H 同时
必须通过 F 的正 scale/logit AUC 不变门与 G 的 endpoint decisions/past-only/
runtime-no-GT 门。先实现并验证；部署仍受同提交 profile 和 2 GPU·小时门
约束。

### M18 H 实现、远端验证与部署

H 已在 `84da20f46f772a20bc90018117f9833e414ad97a` 以组合配置完成，
不增加第三种机制：相对 G 只打开 F 已验证的 monotone calibrator 与三项
calibration loss。激活预期为三项 lifecycle margin、三项 calibration 和一项
endpoint pointer；transport 必须为零。提交器对 H 同时执行 calibration
invariance gate 与 boundary past-only/no-GT gate。

- 本地 CPU-safe 配置/审计/提交器套件：`21 passed`；compile/Bash 通过。
- N16R4 独立 checkout exact `84da20f`：含因果套件
  `155 passed in 97.39s`，结束 SHA 不变且 clean。
- Slurm：`1178354`。
- run：`/data/run01/sczc063/yuzibo/runs/persistent_binding/model_opt_boundary_calibration_seed705_20260721_101605`。

H 仍先跑同提交 profile；预算或稳定性不通过时不会训练。E/F/G/H 至此构成
完整、同协议的 2×2 部署。

### M19 2×2 结果比较器

提交 `c0a4b01` 新增纯 JSON 比较器
`tools/compare_persistent_binding_factorial.py`。它固定读取 E/F/G/H 四个 run，
逐臂计算 calibration 主效应、boundary 主效应和交互项，覆盖 committed
predictions、prediction/GT 比、Recall@0.3、mAP、identity error 及三通道
AUC/固定阈值 TPR/FPR。它还统一检查 frozen data hashes、activation、F/G
机制门和 reporting/raw-RGB/threshold-search 状态。

只有 H 同时技术通过、激活通过、单调门通过、past-only/no-GT 门通过时，
比较器才设置 `multi_seed_authorized_next=true`；无论本轮结果如何，
`raw_rgb_authorized_next` 恒为 false。本地纯 JSON 回归 `2 passed`。

### M20 H v1 画像预算拒绝与修订方向

Slurm `1178354` 在 13:41 完成四项稳定画像后按预算门停止，未开始训练：

- fixed/rematch train mean：约 `0.9719/1.0505 s/step`；
- calibration inference 合计估算：`0.08264 GPU·h`；
- train + reporting inference raw total：`1.69089 GPU·h`；
- 乘冻结 `1.25` 安全系数后：`2.11361 GPU·h > 2.0`；
- capacity exhaustion 为零，binding inference equivalence 和 stability 均通过。

因此这是预算 gate reject，不是代码、数值或科学运行崩溃。对比 G 的
`1.70656 GPU·h` 安全估算，H v1 的主要新增开销来自每个 token 分别启动三次
calibration BCE。下一修订冻结为 **episode-batched balanced calibration**：
每个流式训练 chunk 先收集所有当前/过去可见的三通道监督，再各执行一次
正负平衡 BCE；raw detach、固定 0.5、匹配、margin 和 runtime 语义不变。

为保持 2×2 公平性，F 与 H 都建立 batched-v2 对应版本；先过 exact tests 和
自身 profile，不能用取消安全系数或放宽 2 小时上限解决。

### M21 E 完整结果与 batched-v2 exact 验证

E/reserve6 Slurm `1178040` 已封存完整双臂产物。两臂均完成 `2010/2010`
更新，监督耗尽、GT-birth/runtime 空槽碰撞和协议违规均为零，实际双臂
`1.13361 GPU·h`，证明 `4+2` 过渡容量修复有效且无需继续扩槽。但固定 0.5
下两臂仍为 0 committed interval、0 Recall@0.3、0 mAP，因此 E 科学门拒绝。
REMATCH 的 birth/alive/end AUC 为 `0.7604/0.7600/0.5637`，birth 正例最大值
仍仅 `0.4466`；这把剩余问题进一步锁定为 birth/alive 刻度与 end 排序，而非容量。

提交 `5a11881c00e2e20a9644b8592fc7b5b680d7799d` 实现 F2/H2 共用的
`episode_balanced` 校准聚合：每个严格按时间排列的训练 chunk 只对
birth/alive/end 各执行一次全段正负平衡 BCE，替代 v1 每 token 三次 BCE；
校准输入继续 `stop_gradient(raw_logit)`，固定 0.5、raw 匹配/原损失/margin、
runtime 和因果信息边界均不变。v1 继续作为可复核旧行为保留。

- 新增 F2/H2 成对配置、激活门、提交路由和 v1/v2 一致性因子比较门；
- 本地 CPU-safe/JSON/配置回归 `26 passed`，本机 Torch 仍被既知
  Windows `c10.dll` 环境故障阻断；
- N16R4 clean detached exact `5a11881` 的 Torch、因果、配置、诊断、比较与
  提交器扩展套件 `160 passed in 88.32s`，结束 SHA 不变且 worktree clean；
- 下一门为分别提交 F2 与 H2 自身 profile；两者仍受冻结 `1.25` 安全系数和
  双臂 `2 GPU·h` 上限，不以放宽预算换取通过。

### M22 F-v1 诊断、AUC 浮点门修复与 F2/H2 重新部署

F-v1 Slurm `1178214` 已完成双臂各 `2010/2010` 次更新。单调校准器 scale
全部为正，且 raw/calibrated AUC 的最大差值只有约 `1.27e-7`；旧诊断却使用
`1e-12` 作为相等容差，因 float32 舍入把机制门误判为失败。该问题只影响
“单调映射是否保持排序”的诊断判定，不改变模型权重、预测或固定 `0.5` 协议。

- FIXED birth/alive/end AUC 为 `0.7657/0.7894/0.5037`；birth 正例最大概率
  `0.2510`、alive 正例最大概率 `0.4175`，仍未越过固定阈值；
- REMATCH birth/alive/end AUC 为 `0.7379/0.7834/0.4402`；birth 正例最大概率
  `0.3977`，alive 已有 `0.6976` TPR，但 birth 与 end 仍阻断完整生命周期；
- 因而 v1 的科学结论是：一轮已学到部分排序，但逐 token 校准既慢又不足以
  把 birth 稳定校到可执行区间，不能仅靠事后降阈值宣称成功。

提交 `af585388dc84682434c6e8cf9266462a173b3963` 将 AUC 不变门固定为
`float32 atol=1e-6`，即最多容忍 `0.0001` 个 AUC 百分点的纯数值漂移；
`1e-5` 量级的实质变化仍会拒绝。N16R4 exact clean 扩展套件通过
`161 passed in 87.35s`，结束 SHA 与起始 SHA 一致。

此前基于旧诊断提交的 F2/H2 作业 `1178433/1178434` 在训练和 checkpoint
产生前被安全取消，原目录保留为审计证据且不复用。修复版已用新时间戳重新提交：

- F2：Slurm `1178448`，run
  `/data/run01/sczc063/yuzibo/runs/persistent_binding/model_opt_calibration_batched_seed705_20260721_112355`；
- H2：Slurm `1178449`，run
  `/data/run01/sczc063/yuzibo/runs/persistent_binding/model_opt_boundary_calibration_batched_seed705_20260721_112355`；
- 两者 exact code 均为 `af58538`，先独立通过画像预算门，之后才允许训练。

### M23 一轮筛查与十二轮正式训练的冻结关系

当前 E/F2/G/H2 每臂 `1 epoch = 2010` 次更新，只承担学习就绪筛查：确认监督
真实激活、梯度可达、容量不耗尽、数值/因果/预算门完整成立，并记录排序与固定
`0.5` 下的运行诊断。第 1 轮零发射只表示“当前 checkpoint 尚不可执行”，
不再作为永久淘汰模型结构的条件；否则会把尚未收敛误写成结构证伪。它可以证明
“是否已出现有效学习信号”，不能证明收敛，也不是论文主结果的最终训练轮次。
E 的 REMATCH birth AUC `0.7604` 但正例最大概率仅 `0.4466`，正是“一轮学到
排序、尚未学成可运行检测器”的直接例子。

只有学习就绪门通过后，候选版本才进入第一档完整训练：每臂 `12 epoch`，约
`24,120` 次更新，并只在 calibration split 记录第 `3/6/9/12` 轮 checkpoint
曲线；reporting split 继续锁定。固定 `0.5` 的完整生命周期发射与定位指标到
第 12 轮才作为正式科学门，前述 checkpoint 只用于判断收敛趋势，不搜索阈值。
若预先规定的第 `9→12` 轮仍持续改善且没有过拟合迹象，才允许同一协议续训到
`24 epoch`。不得只延长有利实验臂；FIXED/REMATCH 必须成对保持相同训练预算
与调度。

### M24 一轮门分层实现、G 完整诊断与 v2 画像通过

用户复核指出：在仅训练一轮时，用固定 `0.5` 的零发射直接永久否定模型，可能
把尚未收敛误写成结构失败。代码因此将同一个一轮产物拆成两层，不改阈值：

1. `learning_readiness_pass`：硬检查协议/因果违规、监督耗尽、容量碰撞、
   skipped/缺失更新、成对来源与一轮画像预算；
2. `operational_pass`：继续如实检查固定 `0.5` 下的非零发射、prediction/GT
   比和 Recall@0.3，但在 epoch 1 仅为诊断，不授权论文结论或多种子；
3. 因子比较器只有 `operational_pass` 才授权 multi-seed；若仅学习就绪，则最多
   授权 FIXED/REMATCH 成对的 12 轮特征训练。正式固定阈值门冻结在 epoch 12。

本地纯配置/JSON/协议回归 `22 passed`；其中新增回归明确证明“epoch 1 静默可进
多轮，但不可进 multi-seed”。本机只因 worktree 权限无法写 `__pycache__`，不
影响已通过的 pytest；静态 AST 检查另行执行。

G/boundary Slurm `1178279` 已完整结束。旧 v1 gate 仅因两臂 epoch-1 零发射
返回非零退出；两臂各完成全部更新，机制激活、预算和因果门通过，不能据此宣称
边界结构被证伪。诊断反而显示清晰学习信号：

- FIXED birth/alive/end AUC=`0.8835/0.9023/0.6575`；
- REMATCH birth/alive/end AUC=`0.6671/0.6592/0.6161`；
- endpoint pointer 两臂均有 `480` 次严格 past-only 决策、runtime 无 GT，
  但一轮准确率仅 `3/480` 与 `2/480`；这属于需要多轮验证的未收敛信号；
- end 在固定 `0.5` 下已出现少量 TPR（FIXED `0.0375`、REMATCH `0.0479`），
  birth 仍未越线，因此完整生命周期暂为零。

F2/H2 新作业的自身画像均已通过并开始训练：F2 `1178448` 安全估算
`1.50787 GPU·h`，H2 `1178449` 为 `1.72583 GPU·h`，均低于冻结的
`2 GPU·h` 上限。两条 exact code 均为 `af58538`，当前损失激活符合各自设计。

### M25 配额故障恢复与 exact-2a 重试

F2/H2 `1178448/1178449` 的画像门均通过，但在真实训练阶段同时遇到
`OSError: [Errno 122] Disk quota exceeded`，退出码为 `120`；这是共享存储
写入故障，不是模型、梯度或固定阈值的科学结果，因此两条不进入因子比较。
恢复过程只处理可由代码与 seed 重建的旧 checkpoint：先逐文件记录 SHA-256，
再删除 13 个已废弃 smoke/screen checkpoint（计划释放 `171,080,075` 字节），
并原位清空 12 个已完成旧实验 checkpoint（计划释放 `162,611,212` 字节）。
所有 JSON、Slurm 日志、诊断、当前 G/F2/H2、原始特征及其他项目均保留。

两份恢复清单位于：

- `/data/run01/sczc063/yuzibo/runs/persistent_binding/storage_cleanup_reproducible_checkpoints_20260721.txt`；
- `/data/run01/sczc063/yuzibo/runs/persistent_binding/storage_truncation_completed_checkpoints_20260721.txt`。

随后 4 KiB 强制写探针通过，配额查询恢复约 `159 GiB` 可用。分层一轮门提交
`2a5797c7a1df2b06c010d9acccd5060c49c6231c` 已在独立远端 checkout 通过
`162 passed in 75.62s`，共享 exact checkout 为
`/data/run01/sczc063/yuzibo/projects/OpenTAD_OnlineTAD_Epoch1Gate_2a5797c`。

第一次 exact-2a 重试 `1178500/1178501` 在 `g0063` 上以 `0 秒、0 GPU、0 CPU、
exit 0:53` 结束且没有作业输出，判为 Slurm/节点启动故障；未把它写成模型失败。
第二次重试使用新目录：

- F2 `1178504`：`model_opt_calibration_batched_seed705_20260721_125630`；
- H2 `1178505`：`model_opt_boundary_calibration_batched_seed705_20260721_125631`。

二者当前均稳定运行并再次通过自身画像，最新安全估算分别为
`1.51281/1.73497 GPU·h < 2`。F2 位于 `g0003`，H2 位于 `g0063`；H2 已越过
此前的零秒启动点，故不因节点名称主动取消，继续按真实日志和完整产物判定。

### M26 十二轮正式 H2 闭环实现

正式训练候选固定为 H2 的 `reserve6 + episode-balanced monotone calibration +
causal transition-end + past-only start pointer`，仍是缓存因果 RGB 特征实验，
不是 raw-RGB 联合训练。新增成对 FIXED/REMATCH 配置，从 seed 初始化重新训练
`12 epoch = 24,120` 次更新，不从一轮 pilot 续训；只在 calibration split 对
第 `3/6/9/12` 轮 checkpoint 评估，reporting 保持锁定，birth/alive/end 阈值
始终固定 `0.5`。

正式 Slurm 链路强制执行以下闭环：

1. 只有 exact-2a H2 的 `learning_readiness_pass`、单调校准门、pointer
   past-only 与 runtime-no-GT 门同时通过，才能提交；
2. FIXED/REMATCH 两臂分别用一张 GPU 并发；任一臂提交失败时自动撤销另一臂，
   避免不成对实验；
3. checkpoint 与四次校准推理先写节点本地盘，只回传第 12 轮恢复点、校准选中的
   checkpoint、四条 emission ledger、训练审计、资源报告和哈希清单；
4. 两臂成功后由依赖作业自动生成成对结果，并在第 12 轮执行固定 `0.5` 正式门：
   完整更新、零监督耗尽/容量碰撞/因果违规、非零最终区间、prediction/GT 比
   `[0.25,4.0]`、Recall@0.3 至少 `0.25`、配对训练不超过 `16 GPU·h`；
5. 正式门通过只授权 feature-level 多种子，仍不直接授权 reporting 或 raw-RGB。

当前本地正式配置、纯 JSON 门、提交器与既有协议的相关回归为 `31 passed`，
Python 编译、`git diff --check` 和 Bash 语法均通过。远端 exact-SHA 复核及正式
提交仍等待本提交固化，以及正在运行的 H2 一轮学习就绪门完成。

### M27 正式链路 exact-SHA 就绪

正式 12 轮闭环已固化并推送至
`f3958b45fba4d9d2cd83ccd0cb604e87e043c6f4`。独立共享 checkout：
`/data/run01/sczc063/yuzibo/projects/OpenTAD_OnlineTAD_Formal12_f3958b4`；
检出后强制 4 KiB 写探针通过。该精确 SHA 上的模型、因果重放、数据、配置、
诊断、科学门、正式 pair gate 与提交器扩展套件共 `167 passed in 79.84s`，
Python 编译和两条 Bash 提交器语法通过，结束 SHA 不变且 worktree clean。

该版本只增加正式配置、校准结果构造/门禁、Slurm 编排、测试和 Wiki，未修改
H2 pilot 所使用的 detector/head/runtime 模型代码。正式提交当前唯一前置条件
是正在运行的 H2 `1178505` 产出完整双臂一轮证据并通过分层学习就绪、激活、
单调 AUC 不变、endpoint past-only 与 runtime-no-GT 门；在此之前不抢跑 12 轮。

### M28 E/G 非破坏式分层门复核

为避免 2×2 比较混用旧 v1 与新 v2 语义，使用 exact-2a 的纯结果评估器从 E/G
既有 `screen_result.json` 与资源报告重新生成 `screen_gate_v2.json`；旧
`screen_gate.json` 保留不覆盖，并由 `screen_gate_v2.sha256.txt` 同时记录新旧
哈希。比较器现在优先读取 companion v2，不存在时才回退原文件。

- E：`learning_readiness_pass=true`，实际 `1.13361 GPU·h`，完整更新且零容量、
  监督、跳步、因果问题；`operational_pass=false` 只因一轮固定 0.5 静默；
- G：`learning_readiness_pass=true`，实际 `1.29361 GPU·h`，机制与完整性门通过；
  `operational_pass=false` 同样只是一轮固定 0.5 静默；
- 两次复核均未访问 reporting、未搜索阈值、未改变 checkpoint 或预测。

本地新增回归证明 companion v2 会被优先读取，同时仍保留旧门作审计证据。
因此 E/G 可公平参与“是否已具备 12 轮学习就绪”的 2×2 分析，但二者仍无
multi-seed 资格。

### M29 正式 Slurm 资源合同预检修复

在未提交真实作业前使用 `sbatch --test-only` 检查正式两类作业。集群提交策略
要求 GPU 分区的每个作业显式申请 `1–8` 张 GPU，并禁止覆盖其每卡默认内存；
初稿中的零 GPU finalizer 与显式 `--mem` 因此会被提交策略拒绝。现已在部署前
修复：训练臂各申请 1 GPU/8 CPU/8 小时，finalizer 申请 1 GPU/8 CPU/30 分钟，
两者均使用集群默认内存，并排除已出现零秒启动故障的 `g0063`。

finalizer 的最坏 30 分钟不漏算：正式门在两训练臂真实 GPU 时间之上固定加
`0.5 GPU·h` 保守预算，再与总上限 `16 GPU·h` 比较。新增预算边界回归证明：
两臂 `15.6 GPU·h` 时加 finalizer 后为 `16.1`，必须拒绝。修复后训练臂与
finalizer 两种 `sbatch --test-only` 均通过；该操作没有创建真实 Slurm 作业。

### M30 F2/H2 一轮收口、2×2 判定与正式十二轮启动

F2 `1178504` 与 H2 `1178505` 均完整结束：FIXED/REMATCH 每臂都是
`2010/2010` 次更新，零跳步、监督耗尽、容量碰撞和协议违规；实际双臂资源分别为
`1.17306/1.31972 GPU·h`。两者均通过 `learning_readiness_pass`、损失激活与
单调校准机制门，但固定 `0.5` 下的一轮运行门仍为 false，原因仅为零最终发射。
这继续按 M23/M24 解释为“已能学习但尚未收敛”，不作为永久结构淘汰。

一轮诊断中，F2 的 FIXED birth/alive/end AUC 为
`0.8650/0.8927/0.4557`，REMATCH 为 `0.5815/0.5478/0.5197`；H2 的 FIXED
为 `0.8457/0.8433/0.6497`，REMATCH 为 `0.3254/0.2719/0.5630`。H2 两臂各有
`480` 次 endpoint pointer 决策，均严格 past-only、runtime 无 GT。H2/REMATCH
一轮 birth/alive 低于随机是明确风险信号，但不是预注册的一轮完整性淘汰条件；
不得据此追加事后门或只对有利臂调参，必须由冻结的 12 轮收敛曲线和 epoch-12
固定阈值门裁决。

E/F2/G/H2 的 batched-v2 2×2 比较已生成：H2 同时通过学习就绪、激活、单调
AUC 不变和 boundary past-only/no-GT 机制门，故只授权下一门
`paired_12_epoch_feature_training`；`multi_seed=false`、`raw_rgb=false`。
比较产物位于
`model_opt_boundary_calibration_batched_seed705_20260721_125631/factorial_comparison_batched_v2_a6e16a8.json`，
SHA-256 为 `e6ed85d91dea7e972247b754eb8a9c2eb46cce5c725a8270d7556d2d0de00a0a`。

正式代码冻结并推送为 `a6e16a88bd5f95e284a76859ef2a4106f6b5fd27`；共享 clean
checkout `/data/run01/sczc063/yuzibo/projects/OpenTAD_OnlineTAD_Formal12_a6e16a8`
通过 `170 passed in 80.68s`、SHA 不变和写探针。首次启动包装器因登录节点默认
`python` 指向 Python 2 而在比较器解析阶段退出，尚未调用 `sbatch`、没有创建任何
作业；改为显式 `python3` 后成功提交：

- FIXED：`1178653`；
- REMATCH：`1178654`；
- 成对依赖终检：`1178655`；
- run：`/data/run01/sczc063/yuzibo/runs/persistent_binding/formal12_boundary_calibration_batched_seed705_20260721_141835`。

两条训练臂已在 `g0066` 运行，作业脚本均显式排除 `g0063`；终检处于正常
`Dependency` 等待。`formal12_launch.json` 固定 seed 705、12 轮、成对提交、
无阈值搜索、未访问 reporting、未授权 raw-RGB；训练从 seed 初始化而非 pilot
续训，只在 calibration 检查第 `3/6/9/12` 轮，并由 `1178655` 自动执行第 12 轮
固定 `0.5` 正式门。

### M31 正式训练第 3 轮节点本地 checkpoint

FIXED `1178653` 与 REMATCH `1178654` 分别在 15:44:08、15:45:31 完成第 3 轮，
连续三轮日志均到达 `[02009/02009]`，随后正常进入第 4 轮。每轮最后一个
minibatch 的总损失从第 1→2→3 轮分别为：FIXED `2.9177→2.2809→1.9829`，
REMATCH `2.9213→2.3640→2.0649`；数值有限，生命周期、校准与 pointer 损失
持续激活，transport 仍按设计为零。

通过各自 Slurm allocation 的只读 `srun --overlap` 核验，节点本地 checkpoint
真实存在：

- FIXED `/tmp/sczc063_ontad_formal12_1178653_fixed/.../checkpoint/epoch_2.pth`，
  `17,512,977` bytes；
- REMATCH `/tmp/sczc063_ontad_formal12_1178654_rematch/.../checkpoint/epoch_2.pth`，
  `17,512,977` bytes。

两臂仍在 `g0066`，无 fatal、OOM、磁盘或启动异常；依赖终检 `1178655` 正常等待。
按照冻结编排，第 3/6/9/12 轮 checkpoint 会先全部保留在节点本地，12 轮训练完成
后才依次在 calibration split 推理、选模并受控回传，因此当前不提前生成或查看
calibration 指标。逐轮容量、因果、跳步与完整更新计数也由训练结束时一次写出的
`training_audit.json` 和终检统一裁决，当前只记录可直接验证的训练与 checkpoint
事实，不把中途日志冒充最终完整性结论。

### M32 正式训练第 6 轮节点本地 checkpoint

FIXED 与 REMATCH 已分别于 17:08:44、17:11:39 完成第 6 轮并进入第 7 轮，
训练进度越过 50%。六轮最后一个 minibatch 的总损失严格依次为：

- FIXED：`2.9177→2.2809→1.9829→1.8628→1.7417→1.6392`；
- REMATCH：`2.9213→2.3640→2.0649→1.8779→1.7205→1.5142`。

此处只把最后 minibatch loss 作为数值稳定和优化方向检查，不把它当成
calibration 定位效果。学习率按共同 cosine schedule 从 `2.0e-4` 降至
`1.0e-4`；两臂均没有 NaN、fatal、OOM 或主作业退出。

通过各自 allocation 只读核验，节点本地 `epoch_5.pth` 均存在且为
`17,512,977` bytes。FIXED/REMATCH 的 checkpoint 间隔、大小和训练轮次保持
对称；依赖终检 `1178655` 继续正常等待。第 3/6 轮模型仍不提前回传或评估，
冻结脚本将在 12 轮训练完成后统一进行 calibration-only 的 3/6/9/12 比较；
下一关键持久化节点为第 9 轮 checkpoint。

### M33 正式训练第 9 轮节点本地 checkpoint

FIXED 与 REMATCH 已分别于 18:34:14、18:38:20 完成第 9 轮并进入第 10 轮，
整体训练超过 75%。节点本地 `epoch_8.pth` 均经 allocation 内只读核验存在，
大小同为 `17,512,977` bytes。两臂主作业仍无 fatal、OOM、磁盘或启动异常，
依赖终检继续等待。

第 7→8→9 轮最后一个 minibatch loss 为：FIXED
`1.4882→1.3629→1.3719`，REMATCH `1.4650→1.3392→1.2894`，共同学习率已降至
`3.0e-5`。FIXED 最后一个 minibatch 在 8→9 轮有 `+0.0090` 的小幅波动；
单个 minibatch 不能代表泛化或过拟合，也不能触发调参。预注册的 9→12 判断
仍只使用 12 轮训练结束后、固定阈值下统一产生的 calibration-only 指标曲线。

至此第 3/6/9 轮 checkpoint 的存在性与对称性均已确认，但均未提前回传、选模
或访问 reporting。下一关键节点是第 12 轮训练审计、四 checkpoint 校准重放、
受控产物提升以及成对固定 `0.5` 终检。

### M34 十二轮训练完成、校准区间合同故障与恢复修订

FIXED `1178653` 与 REMATCH `1178654` 都已完整完成 12 轮、每轮
`2010/2010` 次更新；最后一个 minibatch loss 分别为 `1.8342` 与 `2.0328`。
故障发生在训练完成后的 calibration-only 推理/评测，不是训练崩溃，也不是
固定 `0.5` 科学门的拒绝。两臂均先产生了真实最终发射，且未来端点、未来特征、
负延时和非单调发射违规都为零：FIXED 第 3 轮 checkpoint 共 `371` 条，REMATCH
第 6 轮 checkpoint 共 `7041` 条。随后预算 mAP 评测器分别拒绝以下零长度区间：

- FIXED：`video_validation_0000179`，`49.056397486535005 == 49.056397486535005`；
- REMATCH：`video_validation_0000055`，`0.5003204996928118 == 0.5003204996928118`。

根因是 same-step birth+end 已按设计提交一次，但 scalar/pointer start 与 binary end
都可能落在当前 source frame；模型层旧测试甚至明确接受 `start == end`，而正式
评测合同要求 `end > start`。修订不改训练图、损失、阈值、数据、seed 或匹配轴：
仅当解码得到同一点时，把短动作向左量化到一个已经观察到的 feature cell，裁到
零帧边界；最终写出层和因果审计层同时强制
`0 <= start < end <= source <= emit`。这不会读取未来，也没有搜索或降低 `0.5`。

原脚本还暴露出恢复时序缺陷：四个 checkpoint 与 `training_audit.json` 只保存在
作业私有 `/tmp`，校准失败后被节点回收，因此本次训练不能继续校准、必须按同一
冻结训练协议重跑。修订版在任何 calibration 开始前先验证 12 轮、`24,120` 次
成功更新、零跳步/监督耗尽/容量碰撞，再把第 3/6/9/12 轮 checkpoint、配置和
训练审计原子提升到共享 run 的 `arm/recovery/`，写 SHA-256 清单；后续校准直接
读取这些恢复副本。旧依赖终检 `1178655` 因 `DependencyNeverSatisfied` 已取消。

当前修复的 CPU-safe 合同测试为 `7 passed`，Python 编译与 `git diff --check`
通过；本机 Torch 回归仍由既知的 Windows `c10.dll` 环境故障阻断，不作代码失败
解读。下一步是在 N16R4 clean exact checkout 跑完整相关套件与 Bash/test-only，
通过后才提交同 seed、同双臂、同 12 轮重试。reporting、阈值搜索和 raw-RGB
继续锁定。

### M35 正长度/恢复修订 exact 验证与正式重试启动

修订提交已固化并推送为 `22c4028aa29db6cacdb4f057e830c43b68e673ab`。
独立共享 checkout
`/data/run01/sczc063/yuzibo/projects/OpenTAD_OnlineTAD_Formal12_22c4028`
通过 4 KiB 强制写探针、Python 编译、两条 Bash 语法和完整相关 Torch/因果/
配置/评测套件：`174 passed in 79.11s`。相对旧 exact 套件新增的 4 条回归覆盖
same-step 正长度量化、零长度写出拒绝、原子 recovery 成功路径和不完整更新拒绝。
精确 SHA 与 clean 状态在测试结束后保持不变。

冻结前置合同也在不创建作业的 `MAX_SUBMIT_JOBS=0` 路径通过，正确停在提交槽
上限检查；训练臂 `sbatch --test-only` 通过。旧 finalizer 已取消，直接复用其
`afterok` 依赖做 test-only 会得到预期 dependency error；删除该失效依赖的临时
只读资源副本后，finalizer 的 1 GPU/8 CPU/默认内存/30 分钟资源合同通过。

新正式重试已提交：

- run：`/data/run01/sczc063/yuzibo/runs/persistent_binding/formal12_boundary_calibration_batched_seed705_20260721_202804`；
- FIXED：`1178956`；REMATCH：`1178957`；依赖终检：`1178958`；
- exact commit：`22c4028aa29db6cacdb4f057e830c43b68e673ab`；
- protocol revision：`positive_duration_emission_and_precalibration_recovery.v1`；
- same-step 量化：`one_observed_feature_cell_left_clipped_at_zero`。

启动核验时 FIXED 已在 `g0003` 训练，第 1 轮到 `00050/02009`、loss=`2.6588`，
无 fatal；REMATCH 因 `Priority` 等待，终检正常处于 `Dependency`。两臂脚本均
显式包含精确提交、排除 `g0063`、校准前 recovery 提升以及从 recovery checkpoint
重放。阈值仍为 birth/alive/end=`0.5`，reporting=false、raw-RGB=false。

### M36 正式重试双臂均进入训练

REMATCH `1178957` 已从 `Priority` 等待转为 `RUNNING/g0045`，与运行在
`g0003` 的 FIXED `1178956` 形成独立节点并行训练；依赖终检 `1178958` 仍正常
等待。20:45–20:46 的直接日志状态为：

- FIXED：第 1 轮 `01100/02009`，loss=`3.1355`，作业已运行 16:45；
- REMATCH：第 1 轮 `00550/02009`，loss=`3.2979`，作业已运行 9:45。

两臂 stderr 目前均为 620 bytes，fatal 计数为零；尚未到第 3 轮 checkpoint，
所以 `arm/recovery/` 未出现是预期状态。recovery 只在完整 12 轮训练审计通过后、
任何 calibration 开始前原子落盘，不能把当前 pending 解读为恢复失败。下一硬
记录点仍是第 3 轮 checkpoint；期间只监控数值稳定、作业状态和资源，不读取
calibration 或 reporting。

### M37 北京时间 21:00 完整进度报告

正式重试处于“特征级、单种子、成对 12 轮收敛与可执行性门”，不是 raw-RGB，
也还不是论文 reporting 主表。它要回答：H2 的 reserve6 + 单调生命周期校准 +
因果 transition-end + past-only start pointer 在不改固定 `0.5` 的条件下，经过
完整训练后能否稳定输出合法区间；以及 FIXED/REMATCH 两种监督绑定是否都具备
进入 feature 多种子主实验的资格。

21:00 直接状态：

- FIXED `1178956`：`RUNNING/g0003`，1 GPU、8 CPU、8 小时时限；已进入第 2 轮，
  最新日志约 `2261/24120` 次更新（约 9.37%），第 1 轮末 loss=`2.9177`；
- REMATCH `1178957`：`RUNNING/g0045`，1 GPU、8 CPU、8 小时时限；第 1 轮约
  `1701/24120` 次更新（约 7.05%）；
- 终检 `1178958`：`PENDING/Dependency`，1 GPU、8 CPU、30 分钟上限，状态正常。

两臂所有已记录 loss 均有限：FIXED 范围 `2.3807–3.3941`，REMATCH 范围
`2.6588–3.3480`；fatal、NaN/Inf、OOM、CUDA、traceback 风险词均为零。stderr
只有 rendezvous 提示和 DDP `find_unused_parameters=True` 的性能警告，不影响
正确性；冻结正式运行期间不据此改训练配置。FIXED 第 1 轮末 loss 与修订前正式
运行的 `2.9177` 精确一致，支持“正长度修订只改变最终解码、不改变训练轨迹”的
预期；REMATCH 需等本轮结束后再做同样核对。

当前速度约为 FIXED `1.200`、REMATCH `1.188` updates/s，按已观测速度估算纯
训练分别在 7 月 22 日 `02:04`、`02:17` 左右结束；这只是调度估计，不是科学
结果。两臂距离各自 8 小时时限仍有约两小时以上的预计余量。MaxRSS 约为
`1.94/1.81 GiB`；共享盘仍有 `344 GiB` 可用，当前 run 仅 `440 KiB`，没有配额
或空间风险。

`recovery_manifest.json` 当前尚未出现，符合“12 轮完整审计通过后、校准开始前
才原子提升”的合同；因此目前没有 calibration 候选、曲线或最终区间指标，也没有
访问 reporting。下一步按冻结顺序执行：第 3/6/9/12 轮记录 checkpoint → 两臂
各验证 `24,120` 次完整更新和零容量/监督/因果异常 → recovery 四检查点及 SHA
清单落盘 → calibration-only 比较 3/6/9/12 → epoch-12 固定 `0.5` 成对终检。
只有终检通过，才进入 feature seeds 705/706/707；仍不直接进入 raw-RGB。

### M38 正式重试第 3 轮成对 checkpoint

FIXED 与 REMATCH 已分别在 21:53:34、22:02:49 完成第 3 轮 checkpoint，节点
本地只读核验结果均为 `17,512,977` bytes：

- FIXED：`/tmp/sczc063_ontad_formal12_1178956_fixed/.../checkpoint/epoch_2.pth`；
- REMATCH：`/tmp/sczc063_ontad_formal12_1178957_rematch/.../checkpoint/epoch_2.pth`。

两臂已经继续进入第 4 轮，当前分别约到 `01500/02009` 与 `00950/02009`，fatal
仍为零。前三轮末 minibatch loss 为：

- FIXED：`2.9177 → 2.2809 → 1.9829`；
- REMATCH：`2.9213 → 2.3640 → 2.0649`。

这六个数值与修订前 `1178653/1178654` 的对应轨迹逐项完全一致。因而在前三轮
可观测范围内，正长度区间修订和 calibration 前 recovery 提升没有改变训练图、
随机初始化或优化轨迹；它们只作用于训练完成后的解码/产物时序。当前 checkpoint
仍按合同保存在节点本地，正式共享 recovery 要等 12 轮完整审计通过后一次原子
提升，所以 `recovery=pending` 正常。未提前运行 calibration、未访问 reporting，
下一成对关键点为第 6 轮。

### M39 正式重试第 6 轮成对 checkpoint

FIXED 与 REMATCH 已分别在 23:18:06、23:27:49 完成第 6 轮 checkpoint，节点
本地 `epoch_5.pth` 均经 allocation 内只读核验存在且为 `17,512,977` bytes。
两臂随后进入第 7 轮，最新进度约为 FIXED `00950/02009`、REMATCH
`00350/02009`，主作业 fatal 仍为零，终检继续正常等待依赖。

前六轮末 minibatch loss 为：FIXED
`2.9177/2.2809/1.9829/1.8628/1.7417/1.6392`，REMATCH
`2.9213/2.3640/2.0649/1.8779/1.7205/1.5142`。十二个数值均与修订前
正式运行逐项一致；到 50% 训练剂量为止，没有发现正长度解码/recovery 修订引起
的训练轨迹漂移。共享 recovery 仍按设计等待 12 轮训练审计，当前不提前校准或
访问 reporting。下一成对持久节点为第 9 轮。

### M40 正式重试第 9 轮成对 checkpoint

FIXED 与 REMATCH 已分别在 00:42:53、00:53:48 完成第 9 轮 checkpoint，节点
本地 `epoch_8.pth` 均经各自 allocation 内只读核验存在且为
`17,512,977` bytes。两臂随后进入第 10 轮；01:01 的直接状态分别为 FIXED
`01300/02009`、REMATCH `00600/02009`，主作业 fatal、Traceback、OOM、NaN
和 non-finite loss 均为零，依赖终检 `1178958` 继续正常等待。

前九轮末 minibatch loss 为：

- FIXED：`2.9177/2.2809/1.9829/1.8628/1.7417/1.6392/1.4882/1.3629/1.3719`；
- REMATCH：`2.9213/2.3640/2.0649/1.8779/1.7205/1.5142/1.4650/1.3392/1.2894`。

十八个数值均与修订前正式运行逐项一致；到 75% 训练剂量，正长度区间解码和
calibration 前 recovery 修订仍未改变训练轨迹。当前 `recovery=pending` 符合
“12 轮完整训练审计通过后、校准开始前原子提升”的冻结合同，因此尚无
calibration-only 候选、曲线或正长度最终区间可裁决，也未访问 reporting、搜索
或降低 `0.5`、启动 raw-RGB。下一关键节点为第 12 轮完整更新审计、两份四
checkpoint recovery/SHA 清单落盘，以及随后的 3/6/9/12 calibration-only 重放。

### M41 十二轮容量科学门拒绝与真实 transition reserve 修订

FIXED `1178956` 与 REMATCH `1178957` 均完整完成 `12 × 2010 = 24,120`
次更新，最终一轮末 loss 分别为 `1.8342` 与 `2.0328`，与修订前正式运行
一致。两臂都不是训练崩溃、资源故障或零长度区间评测失败；它们在训练结束后的
pre-calibration recovery 完整性门被同一事实拒绝：
`gt_birth_runtime_entry_free_collisions > 0`。因此没有生成
`recovery_manifest.json`，没有运行 3/6/9/12 calibration-only 重放，也没有
产生可报告的定位指标。依赖终检 `1178958` 转为
`DependencyNeverSatisfied` 后已安全取消。

为防止再次丢失诊断证据，分别在原节点提交只读隔离保全作业 FIXED `1179359`
和 REMATCH `1179360`；两者均在 0 秒内失败，因为 Slurm 作业私有 `/tmp`
已随原 allocation 回收。它们没有创建 recovery、没有绕过容量门，也没有重训。
这暴露的是失败证据应在校验前先写入 quarantine 的可观测性缺口；不能把保全失败
改写成模型训练失败，也不能据此猜测碰撞次数。

冻结 census 重新核验为：全局/fit-core `max_visible_instances=4`、
`max_births_per_step=2`、`gt_entry_free_deficits=0`。代码复核表明当前
reserve6 只把物理槽数从 4 增到 6，却允许预测 candidate/active 状态占满全部
6 槽；训练审计在每个 GT birth 的决策入口、当前步 release 之前检查空槽，因此
“6 个物理槽”并不等于“始终保留 2 个过渡槽”。下一修订不继续盲目扩容，也不改
同一步 release/birth 次序，而是在 candidate admission 中冻结
`occupied <= 4`，使 6 槽始终至少保留 2 个 entry-free birth reserve。该上限
完全由 4+2 census 推导，保持数据、seed、FIXED/REMATCH、损失、匹配、严格因果
和 birth/alive/end=`0.5` 不变。另将失败审计与 checkpoint 先原子写入
evidence-only quarantine，再决定是否生成可用于校准的 recovery。

### M42 硬 transition reserve 已实现，等待 exact-SHA 远端门

容量修订已按冻结 census 直接落到运行时，而不是继续增加物理槽。模型仍有 6 个
query 槽，但 `candidate_recycle` 的每步 admission 现在先计算“决策入口的空槽数
减去 2 个硬 reserve”，因此最多只有 4 个槽可被 candidate/active 状态占用；
每次解码结束还会断言至少保留 2 个空槽。`transition_birth_reserve_slots=2`
必须覆盖冻结的 `max_births_per_step=2`，否则配置在构造时直接拒绝。同一步刚释放
的槽仍不参与当前出生排序，只能在下一个因果决策复用，所以没有改变原始
entry-free 时序合同，也没有读未来。

pre-calibration 保全路径同时改为 fail-closed：若 12 轮训练审计不合法，先把
`training_audit.json`、精确配置和第 3/6/9/12 轮 checkpoint 原子复制到共享
`arm/quarantine/`，逐项记录 SHA-256，并明确
`calibration_authorized=false`；只有审计全部通过才生成 recovery。这样下一次
即使科学门拒绝，也不会再因 Slurm 私有 `/tmp` 回收而丢失碰撞证据。

本地不依赖 Torch 的正式工具/配置回归为 `8 passed + 14 passed`；7 个改动 Python
文件通过 AST 解析，正式提交脚本通过 `bash -n`，`git diff --check` 通过。事件头
Torch 测试在收集阶段仍被已知 Windows `c10.dll` 初始化故障阻断，尚不能算模型
通过；下一门是在 N16R4 干净 exact-SHA checkout 跑完整相关套件。通过后先运行
同提交的一轮成对机制/容量验证，再决定是否提交新的 12 轮 FIXED/REMATCH。
旧 `1178956/57` 只有完整更新与 loss 轨迹证据，没有通过容量门后的 calibration、
mAP 或 Recall，故不能把 `1.8342/2.0328` 误写成定位性能。固定 `0.5`、数据、
seed、损失、匹配、reporting 隔离与 raw-RGB 禁止状态均保持不变。

### M43 exact-SHA 远端门通过并启动一轮成对验证

硬 reserve 实现已提交并推送为
`5edc46c34c0db56e79409fac69276450ad87e949`。独立 N16R4 checkout
`/data/run01/sczc063/yuzibo/projects/OpenTAD_OnlineTAD_HardReserve_5edc46c`
完成 4 KiB 写探针、Python 编译、正式/优化两条 Bash 语法和完整相关
Torch/因果/配置/调度/评测工具套件，结果为 `177 passed in 79.11s`；结束时
精确 SHA 不变且工作树干净。

在该 exact checkout 上已提交 H2 一轮成对验证：

- Slurm：`1179361`；
- run：`/data/run01/sczc063/yuzibo/runs/persistent_binding/model_opt_boundary_calibration_batched_seed705_20260722_024414`；
- 初始状态：`RUNNING/g0003`，无 fatal；
- 输入：冻结的 cached causal features，非 raw-RGB；
- 比较：同 seed-705、同模型与阈值的 FIXED/REMATCH；
- 硬门：same-commit profile/预算、各 `2010` 次完整更新、零 skip/监督耗尽/
  entry-free collision、机制 activation、单调校准、因果 transition-end、past-only
  start pointer 和分层 learning-readiness。

这一轮只验证新容量语义在完整 H2 运行中的激活和可执行性，不用一轮固定 `0.5`
静默作最终收敛判断。只有所有技术/机制门通过，才用本次 exact pilot 作为新 12 轮
前置证据；否则保全并分类失败，不提交长训练。自动监控已切换到
`C:\tmp\ontad_hard_reserve_pilot_progress_5edc46c.ps1`。

### M44 硬 reserve 一轮画像门通过并进入 FIXED 训练

Slurm `1179361` 在 `g0003` 完成 same-commit 测试与四段画像，画像门
`passed=true`、`budget_passed=true`、`stability_passed=true`、
`binding_inference_equivalence_passed=true`。原始测量为：

- FIXED/REMATCH 训练均值：`0.820642/0.834076 秒/step`；
- FIXED/REMATCH calibration 推理均值：`0.259574/0.257333 秒/step`；
- 两臂训练预算：`0.923884 GPU·h`；
- calibration 推理预算：`0.067342 GPU·h`；
- 仅用于资源上界计算的 reporting 推理预算：`0.390409 GPU·h`；
- 原始总计：`1.381635 GPU·h`，乘 `1.25` 安全系数后
  `1.727043 GPU·h < 2.0 GPU·h`；
- 两臂画像 `runtime_capacity_exhaustions=0`，最大模型 GPU 显存约
  `153.44 MiB`；作业当前 MaxRSS 约 `1.94 GB`。

画像通过后作业按冻结顺序进入 FIXED 一轮训练；17:54 的直接状态为
`00400/02009`、loss=`3.2569`，fatal=0。训练尚未结束，故 FIXED/REMATCH
`training_audit.json`、calibration-only 结果和成对 gate 仍为 pending，不能提前
宣称硬 reserve 的完整 H2 容量门已经通过，也没有可报告 mAP/Recall。资源预算中的
reporting 项只是按锁定 chunk 数推算成本，未访问 reporting 数据。阈值仍固定
birth/alive/end=`0.5`，当前仍为 cached-feature 实验。

### M45 FIXED 一轮硬容量门通过并切换 REMATCH

`1179361` 的 FIXED 已完成一轮，训练审计为 `successful_updates=2010`、
`scheduler_steps=2010`、`skipped_updates=0`、
`gt_supervision_exhaustions=0`、
`gt_birth_runtime_entry_free_collisions=0`。运行时另记录
`deferred_birth_due_to_release=132`、`active_abandonments=4`；前者证明原冻结的
“同一步释放槽只在下一决策复用”仍实际生效，而 2 个硬 reserve 使这些延迟没有
转化为 GT birth 入口碰撞。

机制损失的非零更新次数为：birth/alive/end margin
`1195/1321/1135`，birth/alive/end calibration
`1195/1349/1196`，endpoint start pointer `1196`；causal transport 为 `0`，
符合 H2 禁用 transport 的合同。epoch-0 checkpoint 为 `17,512,977` bytes，
SHA-256=`71c21778041f3e73e80864a8ae855a188122c925d35067c551116aa4d9196b95`。

FIXED calibration-only ledger 覆盖 40 个视频但一轮固定 `0.5` 下仍为 0 发射；
future-end、future-source、negative-latency 与 non-monotonic-emit 违规均为 0，
screen 明确 `effectiveness_claim_authorized=false`。这与冻结的分层门一致：零发射
只说明一轮尚未形成可执行定位，不否定容量/机制学习就绪，也不授权性能声明。
作业已顺序进入 REMATCH；47:55 的状态为 `00300/02009`、loss=`3.1978`、
fatal=0。成对 gate 仍等待 REMATCH 完成和后续双臂诊断。
