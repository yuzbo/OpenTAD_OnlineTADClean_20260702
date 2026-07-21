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
