---
type: experiment
updated: 2026-07-21
status: active
scope: Three strictly causal feature-level model-optimization pilots before any raw-RGB joint training.
---

# On-TAD 三版本模型优化实验

## 固定任务边界

- 标准、全监督、严格因果的 Online Temporal Action Detection。
- 决策时刻只读取当前与过去的冻结 SigLIP2 stride-8 特征。
- 模型在线维护实例出生、持续和结束，只在结束后写出不可修改的最终区间。
- FIXED 与 REMATCH 仍只允许在出生后的监督 target-to-slot loss binding
  上不同；数据、阈值、生命周期、推理和评测保持一致。
- 当前三个版本都是 seed-705、fit-only、一个 epoch 的技术优化 pilot，
  不访问 reporting split，不产生论文效果结论，也不授权 raw-RGB。

## 进入本节点前的证据

- 精确基础提交：`b1017d09723bdf99e77de99bdf855b87bb263aec`。
- 修复 weighted-BCE stationary prior 后，seed-705 双臂筛选
  `1177653` 完成 2010/2010 次稳定更新，但仍是零最终区间。
- 旧 repaired-checkpoint 的 calibration-only 分数诊断 `1177682` 已完成；
  三个生命周期通道均没有越过 0.5，继续证明旧模型整体静默。
- 旧的一轮筛选把全部 2,010 次更新都放在线性 warmup 内；已实现但尚未
  训练的 short-warmup 候选把 `warmup_epoch` 从 `1.0` 改为 `0.1`，
  峰值 LR、更新数、模型、阈值和数据不变。

## 三个冻结版本

### A — SW：short warmup 控制组

- 唯一新变化：`warmup_epoch=0.1`。
- 假设：旧 pilot 直到最后一次更新才到峰值 LR，平均有效 LR 暴露不足。
- 要证明：不改模型和阈值时，更多峰值附近更新能否产生非零 birth crossing。

### B — SW+BM：出生边界间隔

- 在 A 上只增加 birth-frame balanced logit-margin 辅助损失。
- 正出生槽被推到 `+margin` 以上，同一步受监督负槽被推到
  `-margin` 以下；没有正出生的时刻不计算该辅助损失。
- 该监督只使用当前 first-crossing 标签和当前输出，不读取未来终点。
- 假设：加权 BCE 可以排序但没有直接保证冻结的 logit-0 决策边界两侧
  留出间隔；显式间隔能提高 birth TPR，同时由同一步负槽约束 FPR。

### C — SW+CT：因果查询传输连续性

- 在 A 上只增加相邻有效 token 的 causal query transport 辅助损失。
- 源是 stop-gradient 的上一时刻 persistent queries，目标是当前 queries；
  传输边际由两时刻预测的 alive mass 构造并 stop-gradient，代价是 query
  cosine distance 加同槽位时间身份先验；Sinkhorn 计划也 stop-gradient，
  梯度只优化当前表示距离。
- 数值设置冻结为 soft temperature `0.25`、identity cost `0.25`、56 次
  log-space Sinkhorn；该设置由四槽不均衡质量的真实迭代收敛回归约束。
- 不使用 GT identity、未来帧、未来终点或 reporting 数据，因此不会修改
  FIXED/REMATCH 的监督比较轴。
- 假设：持续查询的跨时刻漂移会削弱生命周期头；轻量时序传输约束可稳定
  实例表示，并改善 alive/end 与最终提交链路。

这里的 “ChronoTransport” 只表示“按时间先后、从过去向当前做软传输”的
设计思想。当前检索没有核实到可唯一对应的同名 On-TAD 论文，因此不会把
本实现错误归因给一个未核实的方法。直接相关的公开依据是：

- [CausalTAD](https://arxiv.org/abs/2407.17792)：限制时间上下文方向，
  说明因果时序建模对 TAD 有效；
- [MATR](https://arxiv.org/abs/2408.02957)：使用过去记忆维护 On-TAL
  的长时上下文；
- [CAG-QIL](https://openaccess.thecvf.com/content/ICCV2021/html/Kang_CAG-QIL_Context-Aware_Actionness_Grouping_via_Q_Imitation_Learning_for_Online_ICCV_2021_paper.html)：
  明确要求未来不可见且历史提案不可回改；
- [Temporally Consistent Unbalanced OT](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_Temporally_Consistent_Unbalanced_Optimal_Transport_for_Unsupervised_Action_Segmentation_CVPR_2024_paper.html)：
  把时间一致性先验编码进
  optimal transport；本实验只借鉴该原则，不照搬其无监督分割设定。

补充边界：

- [ActionSwitch](https://arxiv.org/abs/2407.12987) 的 conservativeness loss
  说明抑制不必要的在线状态波动可减少 fragmentation；C 处理的是 persistent
  query 表示传输，不复刻其上一状态伪标签。
- [HAT](https://arxiv.org/abs/2408.06437) 与 MATR 都支持“历史信息有用”，
  但它们包含 anticipation/future-supervised 设计；本项目不采用这些部分，
  只保留推理时从过去到当前的严格因果信息流。
- [OpenHOUSE（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/html/Kang_Open-ended_Hierarchical_Streaming_Video_Understanding_with_Vision_Language_Models_ICCV_2025_paper.html)
  明确选择 OAD-based strict On-TAL：动作结束时立即产生并累积区间，不能
  回溯修改。它还用 actionness 检测开始、用 progress 突降检测结束，
  显著改善没有背景间隔的相邻动作边界。该论文与本项目的数据、开放词汇和
  层级任务不同，因此当前 A/B/C 不照搬其 VLM、层级标签或伪标签；只把
  progress-hazard 记录为“birth 已恢复但 end/相邻动作仍失败”时的有条件
  后续模型候选。
- [OZ-TAL（2026 预印本）](https://arxiv.org/abs/2605.09976) 同样把任务
  定义为动作完成时立即定位，并指出近期路线正从 OAD 帧聚合转向实例级
  理解；这支持 persistent instance 建模方向，但其 zero-shot、training-free
  VLM 设定不属于当前标准全监督实验。
- [OnPoint（2026 预印本）](https://arxiv.org/abs/2607.00289) 研究点监督
  Online TAL，并使用离线教师、伪区间和 anticipatory window distillation；
  这些都不是当前 full-supervision FIXED/REMATCH 问题，因此不吸收到
  A/B/C 代码或主张中。

截至 2026-07-21 的精确检索仍未找到可唯一对应的 “ChronoTransport”
On-TAD 论文；检索命中主要是无关词义。因此 C 的命名与引用边界保持不变：
它是本项目定义的 past-to-current causal transport，不冒认外部方法。

## 共同 pilot 门禁

每个版本均部署一条 Slurm 作业，作业内顺序运行 FIXED 与 REMATCH：

1. 精确 40 位提交号、干净 detached checkout 与配置配对检查；
2. 每个版本先在自己的精确提交和配置上完成双臂
   50-warmup/200-measured 画像；自身稳定性、未训练因果等价与一轮
   2 GPU·小时预算门通过后，才进入训练；
3. 真实冻结特征、seed 705、一个 epoch、每臂 2,010 次预期更新；
4. 零 non-finite、零 skipped update、零 GT supervision exhaustion、
   零 GT-birth/runtime collision、零未来信息违规；
5. 训练后只在 calibration split 做 target-conditioned 分数诊断；
6. 首要排序指标是 birth positive/negative gap、pairwise AUC、冻结 0.5
   下的 TPR/FPR 与 committed emissions；不搜索或下调阈值；
7. `training_audit.json` 必须结构化记录每项损失的 epoch mean 与非零
   update 数；A 的两个新增损失均应休眠，B 只能激活 birth margin，
   C 只能激活 causal transport；
8. reporting split、三种子主结果和 raw-RGB 均继续封锁。

三个版本是并行的机制 pilot，不以其中任一结果冒充 FIXED/REMATCH
论文主实验。若 B 或 C 单独通过技术门禁，再做一次受控组合实验；若均失败，
回到 head/lifecycle 表示而不是修改评测阈值。

## 结果前冻结的三版本选择规则

比较器在任何 A/B/C 结果可见前冻结于
`f742dce035d4cdc2bff3fd94bc934b2871ef87b0`，并由 12 项 CPU-safe
测试覆盖。一个版本只有同时满足机制激活、技术 screen、预算、精确
provenance 和 calibration-only 分数诊断，才有资格进入 P1。资格版本按
以下字典序比较；所有“最差”均取 FIXED/REMATCH 两臂中较差者，防止只优化
某一臂：

1. 最大化最差臂 Recall@tIoU 0.3；
2. 最大化两臂、birth/alive/end 三通道中的最低 pairwise AUC；
3. 最大化冻结 0.5 下最差臂 birth TPR；
4. 最大化最差臂 birth positive-negative mean gap；
5. 最小化冻结 0.5 下两臂较高的 birth FPR；
6. 最小化真实双臂 GPU·小时；
7. 仅在数值完全相同的最终平局下，使用固定 A→B→C 顺序。

比较器输出全部原始臂级数值、资格失败原因、训练提交、分析提交和脚本
SHA-256。若三个版本都被门禁拒绝，则 `selected_variant=null`，回到共享
head/lifecycle 设计；不得下调阈值。即使存在胜出版本，也只授权 P1
特征级多轮收敛，不授权论文效果结论或 raw-RGB。

结果解释也预先固定：若 birth TPR 已非零但 committed emission/Recall
仍失败，先查看 alive/end AUC、TPR/FPR 与 lifecycle counts；只有证据把
瓶颈定位到结束或相邻动作分割，才设计 OpenHOUSE 启发的 progress-hazard
版本。若 birth 本身仍无 crossing，则继续修 birth 表示/优化，不让 progress
分支掩盖问题；任何情形都不通过下调阈值“修复”。

## 从当前 pilot 到论文主实验

| 阶段 | 实验 | 主要证明什么 | 放行条件 |
| --- | --- | --- | --- |
| P0（当前） | A/B/C、seed 705、一轮、双臂 | 找到不静默且不爆量的共享模型/优化路线 | 自身画像过预算；两臂均有非零预测、合格 prediction/GT 与 Recall@0.3；零训练/容量/因果错误 |
| P1 | 胜出单版本，seed 705，预先冻结的多轮收敛 pilot | 改进是否能持续，而非一轮偶然 crossing | 在 calibration 上冻结 epoch/阈值；不访问 reporting；新画像仍过预算 |
| P2 | seeds 705/706/707 的 FIXED/REMATCH 特征主实验 | first-crossing 固定绑定是否稳定减少 duplicate/fragmentation | FIXED 在至少 2/3 seeds 改善，`E_id` 相对下降至少 20%，平均 mAP 下降不超过 0.5 个百分点 |
| P3 | 冻结 checkpoint 的一次性 reporting | 给出论文标准 mAP、online budgeted AP、延时和实例错误主表 | 一次性锁、完整 SHA provenance、零 future/monotonicity 违规 |
| P4 | 受控消融和失败子集 | 分清 warmup、margin、transport、FIXED binding 各自贡献 | 每项只改一个因素；报告重复同类、重叠、相邻动作、长短实例 |
| P5（条件） | raw-RGB frozen / PEFT / joint 三阶梯 | 特征级机制能否扩展到真正端到端视频训练 | 只有 P2/P3 通过才启动；严格因果视觉编码、相同 lifecycle 与评测合同 |

论文主结果不是当前三个一轮 pilot。当前 P0 只回答“模型能否学会可靠地开门
并完成生命周期”；P2/P3 才回答 FIXED/REMATCH 的论文主假设；P5 才回答
raw-RGB 端到端扩展。

## 关键节点

- [x] M0：恢复任务、代码、Wiki、Slurm 和预算上下文。
- [x] M1：冻结 A/B/C 三个版本与不越界合同。
- [x] M2：实现模型损失、配置、测试和实验清单。
- [x] M3：本地 CPU-safe 验证与远端 N16R4 轻量验证。
- [x] M4：用数值修订后的精确提交重新部署三条 Slurm 作业。
- [ ] M5：记录 job id、run dir、队列/完成状态和下一门禁。

## M2/M3 实现与验证记录

- 实现提交：
  `7ba049f530c5fac856de3c4d527aebfc8666fb04`，已推送到
  `origin/codex/ontad-science-fixed-rematch`。
- 新增三组各自成对的 FIXED/REMATCH 配置；配置测试证明每一组除
  `trajectory_binding_mode` 与 `work_dir` 外完全相同。
- 新增 `birth_margin_loss` 和 `causal_transport_loss`；A 中两者权重均为
  零，B 只启用前者 `0.5`，C 只启用后者 `0.05`。
- 新增按 `VARIANT=sw|margin|transport` 选择配置的 Slurm submitter，
  每个作业内跑完整一轮双臂训练、calibration-only 推理、target-conditioned
  分数诊断、资源审计和冻结技术门禁。
- 机制激活审计提交
  `3035f4e88033a68be651fc3b77292a34f3864b10` 已推送：
  `training_audit.json` 新增 `mean_losses` 和
  `loss_nonzero_updates`，并生成 `optimization_activation.json`。
  如果预期辅助损失从未产生非零信号，或另一个版本的损失意外激活，
  该 pilot 判为科学门禁失败；作业仍继续保留推理和分数诊断产物。
- Windows 本地完成 Python 编译、两个 Bash 语法检查、配置/工具等
  29 项 CPU-safe 测试；全部通过。本机 Torch 仍因既有 `c10.dll`
  初始化问题不可用，没有把该环境当作模型证据。
- N16R4 干净 detached worktree 在精确实现提交上完成 41 项
  PyTorch/配置/提交器测试，`41 passed in 47.37s`；测试后
  `git status --porcelain` 为零行。
- 首次远端 `git fetch` 遇到 GitHub TLS 中断，第三次有界重试成功；
  这是传输故障，发生在测试前，不是代码或实验失败。
- N16R4 候选 worktree 已前移到精确提交
  `3035f4e88033a68be651fc3b77292a34f3864b10`；包含模型、训练审计、
  三版本配置、提交器和新激活门禁的 116 项测试全部通过，
  `116 passed in 77.56s`，测试后仍为 clean detached。

部署前队列检查显示账户已有 16 个 Slurm 作业（9 RUNNING、7 PENDING），
其中既有诊断 `1177682` 仍为 `AssocGrpGRES`。当前没有空余 submit slot；
不取消或修改其他任务；每出现一个槽位就按 A→B→C 顺序提交一个尚缺
pilot，不等待三个槽同时出现，也不重复提交。

## M4 部署记录

2026-07-21 04:09 北京时间，账户活跃作业从 16 降为 12。再次确认三种
job name、`model_opt_*_seed705_*` 目录和 `pilot_contract.json` 均不存在
后，使用同一精确代码提交
`3035f4e88033a68be651fc3b77292a34f3864b10` 提交：

| 版本 | Slurm job | 运行目录 | 提交后状态 |
| --- | ---: | --- | --- |
| A / SW | `1177693` | `/data/run01/sczc063/yuzibo/runs/persistent_binding/model_opt_sw_seed705_20260721_041045` | CANCELLED / PENDING、零 GPU |
| B / SW+BM | `1177694` | `/data/run01/sczc063/yuzibo/runs/persistent_binding/model_opt_margin_seed705_20260721_041046` | CANCELLED / PENDING、零 GPU |
| C / SW+CT | `1177695` | `/data/run01/sczc063/yuzibo/runs/persistent_binding/model_opt_transport_seed705_20260721_041047` | CANCELLED / PENDING、零 GPU |

提交后三条均出现在 `squeue`，账户活跃作业为 15/16；没有取消、修改或
抢占任何无关作业。心跳已从“寻找提交槽”切换为“只读监控三条 pilot 与
旧诊断 `1177682`”，明确禁止重复提交。

### M4.1 Sinkhorn 数值修订与受控替换

在三条作业尚为 PENDING、运行时间为零时，用当前真实 C 配置复核 4×4
Sinkhorn。temperature `0.1`、20 次迭代的 1,000 个固定 seed 随机样本中：

- mean marginal error `0.00681308`；
- p95/p99 `0.0353077/0.0568880`；
- max `0.127763`；
- `484/1000` 样本误差大于 `1e-3`。

单纯增加到 200 次虽降低均值，但会把每 token 小算子循环放大十倍。把
soft temperature 改为 `0.5` 并保持 20 次后，同一 1,000 样本：

- mean error `1.48095e-7`；
- p99 `1.69873e-6`；
- max `1.76728e-5`；
- 零样本大于 `1e-3`。

但数值收敛不等于身份传输质量合适。模拟“当前 query = 上一 query +
0.2 噪声”的持续实例时，0.5/20 的平均同槽传输质量只有 `0.46641`、
熵为 `2.3036`；0.25/20 则为 `0.60779`、熵为 `2.0378`，更能保留
同一实例，同时其 mean/p99/max 边际误差仅
`8.68e-7/1.57e-5/4.50e-5`。因此进一步对 2,000 个固定 seed 四槽不均衡
压力样本扫描 0.25 的迭代边界：

- 50 次仍有 1 例超过 `1e-3`，max `0.00145200`；
- 55 次 max `0.00105056`，仍未通过；
- 56 次 max `0.000984639`，首次全部通过；
- 60 次 max `0.000759363`，但多出的 4 次不再是过门所需。

最终冻结 temperature `0.25`、56 iterations，并加入同一最坏样本
“55 失败、56 通过”的确定性回归。这样同时约束边际正确性、身份保留和
小算子成本，而不是只追求一个更软、更容易收敛的传输计划。

路径复核还确认，`head.step()` 先吸收当前 token 并更新 query，transport
随后比较“上一 token 解码后的 query”与“当前 token 更新后的 query”；
因此不是自对齐。原实现对一段内每个相邻 token 对各自启动 56 轮 4×4
Sinkhorn。提交 `5c02ccca6142a9e9c8919fe4b0830d1dad3480a3` 将这些
相邻对堆成 batch，一段只执行 56 轮向量化更新；上一 query、边际和计划
继续 stop-gradient，梯度仍只到当前 query。确定性测试证明 batched plan
等于逐对 plan、batched loss 等于逐对 loss 之和且反向梯度有限，因此
科学目标和损失标度均未改变，只消除小 kernel 启动放大。
N16R4 登录节点的单线程 CPU 代理计时（63 个相邻对）为
`4.507844 ms` batched 对 `194.348725 ms` individual，即 `43.11×`；
最大数值差为零。该计时只验证实现优化方向，不替代作业内 RTX 4090
strict profile 或 GPU-hour 门禁。

最后的控制路径复核发现 A/B 虽然 transport 权重为零，旧代码仍无用地
构造每 token 的 lifecycle mass。提交
`d390779443bccc4926bc8fb2bff1875834383c21` 将该计算严格放进
`causal_transport_loss_weight > 0` 分支，并在 no-grad 下构造 C 的预测
边际。新增回归直接把边际构造函数替换为报错函数，证明禁用 transport
的 A/B 训练不会触达它；因此 A/B 保持真正零 transport 计算，三版本
GPU profile 比较不会让控制组承担 C 的隐藏开销。

因此旧 `1177693/1177694/1177695` 在零 GPU 消耗时受控取消，旧目录保留
审计且均无 `pilot_contract.json`。没有取消或修改旧诊断及任何无关作业。
中间提交 `adbd0bbd6a3b3ffe3de36f3c0f8ef506eef7ed48` 与
`22aa93bd1b2b2441ba0810e1923a0fb35e0d5776` 分别完成初步数值修复和
比较器预检；最终部署代码提交为
`f6bd9e12b60749c1244815fca30f9fca7bbdf4bf`；批量版本为
`5c02ccca6142a9e9c8919fe4b0830d1dad3480a3`；最终部署代码为
`d390779443bccc4926bc8fb2bff1875834383c21`。本地 16 项 CPU-safe
配置/提交器/比较器测试通过；N16R4 依次完成 128、130 项，最终 exact
`d390779` 为 `131 passed in 81.02s`，候选始终 clean detached。

05:03 账户仍为 16/16；提交器在创建新目录前正确停止。现在只认 exact
`d390779` 的活跃 job 或 pilot contract，每出现一个 submit slot 就按
A→B→C 重新提交；旧取消目录不算重复。

### M4.2 最终精确提交重新部署

05:37 一个既有训练作业完成后，账户活跃数从 16 降至 15。先暂停自动
提交并重新核验：候选 HEAD 为 exact `d390779443bccc4926bc8fb2bff1875834383c21`、
worktree clean、无新版 `pb_opt_*`、无该 SHA 的 pilot contract。只使用
这一个空槽提交：

| 版本 | 新 Slurm job | 新运行目录 | 当前状态 |
| --- | ---: | --- | --- |
| A / SW | `1177706` | `/data/run01/sczc063/yuzibo/runs/persistent_binding/model_opt_sw_seed705_20260721_053819` | RUNNING / `g0003` |
| B / SW+BM | `1177707` | `/data/run01/sczc063/yuzibo/runs/persistent_binding/model_opt_margin_seed705_20260721_054136` | RUNNING / `g0003` |
| C / SW+CT | `1177708` | `/data/run01/sczc063/yuzibo/runs/persistent_binding/model_opt_transport_seed705_20260721_054210` | RUNNING / `g0003` |

三份作业脚本内 `COMMIT_SHA` 均已复核为 exact `d390779`。第二、第三个
槽位出现后依次提交 B/C；05:42 A 已在 `g0003` 开始，B/C 正常等待 GPU。
`pilot_contract.json` 要在各作业自身 exact 双臂 profile 完成后才生成，
当前未出现不算缺失或失败。没有取消、修改或抢占任何无关作业。至此
“三个版本完整实现并部署”的节点完成；下一门是各自画像、训练、激活
审计、v2 分数诊断和冻结 screen。

### M4.3 三版本自身画像门

05:59 三个 exact-commit profile 均完成并生成 pilot contract：

| 版本 | FIXED / REMATCH 训练均值（秒/step） | 安全系数后预计双臂 GPU·小时 | 门禁 |
| --- | --- | ---: | --- |
| A / SW | `0.67119 / 0.69310` | `1.396616` | PASS |
| B / SW+BM | `0.68916 / 0.69521` | `1.410778` | PASS |
| C / SW+CT | `0.68530 / 0.69477` | `1.407951` | PASS |

三条的稳定性、FIXED/REMATCH 未训练推理等价性和 2 GPU·小时预算全部通过，
且继续零未来信息违规。C 的画像与 A/B 基本等速，说明批量 Sinkhorn
消除了逐 token 小 kernel 放大；这是真实 GPU profile，不是 CPU 代理推断。
三条现均进入完整一轮训练。

旧诊断 `1177682` 同时以 `COMPLETED 0:0 / 00:04:07` 结束。旧 checkpoint
在 28,730 calibration token 上，FIXED birth/alive/end 最大分数为
`0.169824/0.320227/0.419161`，REMATCH 为
`0.326206/0.380038/0.388141`，三通道阈值 crossing 都是零。该 v1
诊断没有 target-conditioned gap/AUC；它只补强旧模型静默结论，不参与
A/B/C 冻结比较。

### M4.4 首臂训练完整性与启动端口隔离恢复

06:14 A/SW 的 FIXED 训练完成并写出结构化审计：`2010/2010` 次更新成功、
`scheduler_steps=2010`、`skipped_updates=0`，且
`gt_supervision_exhaustions=0`、`gt_birth_runtime_entry_free_collisions=0`。
birth/alive/class/start/end 损失按预期激活，`birth_margin_loss` 与
`causal_transport_loss` 的非零更新均为零，证明 A 是只修改预热的干净
控制组。06:18 B/SW+BM 的 FIXED 训练也完成同样的 `2010/2010`、零跳步、
零监督耗尽和零容量碰撞审计；其 birth margin 在 `1156/2010` 个更新中
非零，均值 `0.0826192`，transport 始终为零，证明 B 的新增机制真实激活
且没有串入 C。

随后发现原三条连续 Slurm job 的默认端口规则存在启动链路冲突。每个 job
以 `20000 + job_id % 40000` 为首端口，并在 FIXED 训练、FIXED 校准、
REMATCH 训练、REMATCH 校准之间逐次 `+1`；连续 job
`1177706/1177707/1177708` 因而让 A 的 FIXED 校准端口 `37707` 与 B 的
FIXED 训练端口重合，让 B 的下一端口与 C 的训练端口重合。A
`1177706` 因 `torchrun` rendezvous `Address already in use` 以
`FAILED 1:0 / 00:36:21` 退出；其完整 checkpoint/audit 已保留。这是
训练后的阶段启动冲突，不是模型、数值、数据、监督、容量或科学门禁失败。

不修改冻结模型、配置、数据、seed 或超参数，仍从 exact
`d390779443bccc4926bc8fb2bff1875834383c21` 干净 detached checkout
通过 Slurm 提交两条恢复作业，并显式划分不重叠端口块：

| 版本 | 恢复 job | 恢复运行目录 | 显式端口块 |
| --- | ---: | --- | --- |
| A / SW | `1177711` | `/data/run01/sczc063/yuzibo/runs/persistent_binding/model_opt_sw_seed705_20260721_062012` | `45100–45103` |
| B / SW+BM | `1177712` | `/data/run01/sczc063/yuzibo/runs/persistent_binding/model_opt_margin_seed705_20260721_062013` | `45200–45203` |

C `1177708` 保持原作业继续运行。旧失败目录只作为诊断证据，不进入最终
A/B/C 比较；最终比较只采用三条完整产物链。没有取消或修改任何无关作业，
也没有借该恢复改变模型版本。

06:22 C/SW+CT 的 FIXED 训练审计完成：`2010/2010` 更新成功、零跳步、
零监督耗尽，transport 在全部 `2010` 次更新中非零，均值
`0.0108388`，birth margin 始终为零；机制激活正确。但该臂出现
`5` 次 GT birth/runtime entry-free collision，且 calibration 在冻结
0.5 阈值下提交 `0` 个最终区间。C 因而已经出现两项科学拒绝信号：
传输可能让候选更易进入 active、占用出生容量，同时仍未打通 end/commit
链路。作业不是崩溃，已正常进入 REMATCH 训练；仍需用 REMATCH 和 v2
target-conditioned birth/alive/end 诊断区分“出生仍不足”与
“alive/end 生命周期失衡”。

06:48–06:51 C/REMATCH 也完成 `2010/2010` 更新，transport 在全部更新中
激活、均值 `0.0114319`，margin 为零；它仍出现 `2` 次容量碰撞。冻结
calibration 结果中，FIXED/REMATCH 均为 `0` 个 committed prediction、
`prediction/GT=0`、`Recall@0.3=0`、average mAP `0`。因此 C 在两个监督
绑定下都同时违反“零运行时容量碰撞”和“非零最终区间”门槛，确定不具备
P1 资格；其 Slurm 最终出现非零退出码将是预期科学 gate reject，而非
运行崩溃。仍保留随后生成的 birth/alive/end AUC、gap、TPR/FPR，用于
判断 future work 是否值得把 transport 严格限制到已确认 active 槽；若
这些判别指标也无改善，则停止该路线。

C 的完整产物随后生成并以预期科学拒绝结束：activation gate `PASS`，
真实双臂用量 `1.08 GPU·h`、budget `PASS`；technical gate 因上述容量
碰撞、零提交、零 prediction/GT 和零 Recall 而 `FAIL`。v2 分数诊断显示：

| C 通道 | FIXED AUC / 均值差 | REMATCH AUC / 均值差 | 两臂冻结 TPR |
| --- | --- | --- | ---: |
| birth | `0.31315 / -0.05976` | `0.55220 / +0.00115` | `0 / 0` |
| alive | `0.29555 / -0.15628` | `0.48246 / +0.00020` | `0 / 0` |
| end | `0.56641 / +0.01265` | `0.52970 / +0.00065` | `0 / 0` |

最终 checkpoint 的 FIXED/REMATCH birth 最大概率仅
`0.22687/0.14236`，calibration runtime 均为零 birth proposal。C 不仅
没有形成最终区间，FIXED 的 birth/alive 判别方向还明显反转；因此不做
transport 权重搜索，也不把“仅 confirmed-active 槽 transport”列为紧接
下一版。当前优先级回到 A/B 所检验的优化与生命周期决策边界。

### M5 下一轮 lifecycle 决策边界能力

三版 FIXED calibration 均已确认零最终区间后，提交
`6b0ffd63c07d6c9ea0db1ddcb8055ad90b7d115d` 实现默认关闭的下一轮核心
能力：在既有 birth balanced logit margin 旁，为 alive 和 end 分别增加
同构 margin。三者都只使用当前 prefix 的全监督 target/mask：

- birth 只在当前 first-crossing 事件步约束正 birth 与同一步负槽；
- alive 只在当前存在受监督 active 实例的步约束 occupied 槽与其他槽；
- end 只在当前 endpoint 事件步约束结束槽与同一步其他 at-risk 槽；
- 不读取未来帧、未来终点或 GT runtime identity，也不改变冻结 0.5
  推理阈值。

新参数在公共 pilot base 中全部默认为零；回归锁定原 A/B/C 的
alive/end margin 均为零，因此该提交不追认或改变正在运行的 exact
`d390779` 三版结果。activation gate 已扩展为能单独验证 lifecycle
候选必须同时激活 birth/alive/end 三项且 transport 休眠。本地 Python
编译与 11 项 CPU-safe 配置/activation 测试通过；Torch 梯度测试因已知
Windows `c10.dll` 初始化故障未在本机形成证据，等待 N16R4 干净环境。
具体 margin 数值和是否部署，继续由 A/B 两臂 v2 分数诊断决定。

### M6 三版本终局、冻结比较与第四版部署

07:24–07:25，恢复后的 A `1177711` 和 B `1177712` 完成全部双臂训练、
校准、activation、v2 分数诊断、资源审计与 screen gate。两条作业均以
预期的科学拒绝 `FAILED 1:0` 结束，不是程序崩溃：

| 版本 | 实际双臂 GPU·h | FIXED birth/alive/end AUC | REMATCH birth/alive/end AUC | 双臂最终区间 |
| --- | ---: | --- | --- | ---: |
| A / SW | `1.06583` | `0.451/0.370/0.529` | `0.778/0.823/0.519` | `0 / 0` |
| B / SW+BM | `1.07333` | `0.474/0.511/0.478` | `0.484/0.500/0.476` | `0 / 0` |
| C / SW+CT | `1.08000` | `0.313/0.296/0.566` | `0.552/0.482/0.530` | `0 / 0` |

A/B 两臂均为 `2010/2010` 更新、零 skip、零监督耗尽和零
GT-birth/runtime collision。A 的 REMATCH 已学到可用的相对排序：
birth positive 最大值 `0.495894`，仅略低于冻结 0.5；alive TPR 为
`0.8604`，但 FPR 仍为 `0.2028`，end 判别接近随机。B 的 0.5 birth
margin 虽把 FIXED birth/alive AUC 从 `0.451/0.370` 提至
`0.474/0.511`，却把 REMATCH 三通道压到约 `0.48–0.50`。这说明下一版
不能简单增大 birth margin，也不能继续调 transport。

预先冻结的比较器在分析提交
`68cc9d924d5deafee97c13f1147e340c03c5e53a` 上读取三条完整产物，
确认三版 activation、预算和 provenance 完整，但三版 technical screen
均因双臂零最终区间失败；正式输出为：

- `eligible_variants_in_rank_order=[]`；
- `selected_variant=null`；
- `next_stage_authorized=false`；
- `raw_rgb_authorized_next=false`。

因此 A/B/C 没有胜出者，不进入 P1 多轮、P2 多种子、reporting 或
raw-RGB。下一步仍是特征级 head/lifecycle 修复。

依据上述诊断，第四版冻结为保守的全生命周期 margin：

- birth/alive/end 的 margin 均保持 `0.25`；
- 三项辅助权重均为 `0.1`，总名义权重 `0.3`，低于 B 的单项 `0.5`；
- transport 权重为零，阈值仍为 0.5，不搜索阈值；
- 只使用当前 prefix 的 birth/occupied/endpoint target 与同一步负槽。

这样做的目的不是保证“抬分”，而是检验较小、分散的生命周期边界梯度
能否保留 A/REMATCH 的排序，同时让 FIXED 和 end 通道不再静默。部署提交
为 `d87a116d3a30fb5a83af7ee7c9c3c311bc8bc30c`。提交器同时把端口规则
修为“每个 Slurm job 预留连续四端口块”，避免连续 job 再次交叉占端口。

本地配置、activation、提交器共 `17 passed`；独立 N16R4 clean detached
检出
`/data/run01/sczc063/yuzibo/projects/OpenTAD_OnlineTAD_LifecycleMargin_6b0ffd6`
在 exact `d87a116` 上完成 Bash 语法、Torch 梯度、配置、评测和提交器
`126 passed in 74.03s`，测试后零改动。服务器访问 GitHub 超时后使用
增量 Git bundle 传递原始 Git 对象，提交 SHA 未重建或改写，也未触碰
旧 `d390779` 运行目录。

第四版已通过 Slurm 提交：

| 项目 | 值 |
| --- | --- |
| job | `1177720` |
| run | `/data/run01/sczc063/yuzibo/runs/persistent_binding/model_opt_lifecycle_seed705_20260721_073454` |
| exact commit | `d87a116d3a30fb5a83af7ee7c9c3c311bc8bc30c` |
| 配置 | `thumos_persistent_binding_opt_lifecycle_{fixed,rematch}.py` |
| 自动端口块 | `50880–50883` |
| 初始状态 | `RUNNING / g0003` |

该作业仍先跑自身 FIXED/REMATCH 画像和 2 GPU·小时门；通过后才运行一轮
双臂训练。无论结果通过或拒绝，都必须留下三项 margin 的非零更新计数、
三通道 AUC/gap/TPR/FPR、生命周期计数和完整因果/资源审计。

### M7 前沿方法给出的下一模型方向（条件项）

当前证据把后续问题拆成“出生校准”和“结束判别”，不再笼统归因于
persistent query：

- [MATR](https://arxiv.org/abs/2408.02957) 用当前段判断 end，再以 end
  query 从过去 memory 检索 start；其消融中把 start/end 双 decoder 合并
  为单 decoder 时 average mAP 从 `49.5` 降到 `42.7`。这支持“开始与结束
  不应只靠同一生命周期标量头”的方向。
- 但 MATR 还使用 sliding window、历史 proposal NMS，并在训练中包含
  当前时刻之后的 anticipation region；这些均不能直接移入本项目的
  immutable、无未来、无离线后处理协议。这里只吸收“当前 end + 过去
  start retrieval”的结构分解。
- [OpenHOUSE](https://openaccess.thecvf.com/content/ICCV2025/papers/Kang_Open-ended_Hierarchical_Streaming_Video_Understanding_with_Vision_Language_Models_ICCV_2025_paper.pdf)
  用 actionness 检测 start、用 progress 的突然下降检测相邻动作的 end。
  它说明相邻动作无背景时，单纯 actionness 转移会合并实例；但其 progress
  target 由完整区间计算，且任务含层级/VLM，不作为当前标准全监督主实验
  的直接替代。

因此，若 `1177720` 仍表现为 birth 接近过线而 end AUC/TPR 失败，下一项
预注册候选是**因果 transition-end / past-start factorization**：end 分支
只接收当前 query、上一时刻同槽 query 及其差分，start 只从已观察 memory
检索；不预测未来终点、不回改区间、不使用 NMS。若 `1177720` 的三头 margin
已经使双臂形成合格最终区间，则先做 P1 收敛，不再同时引入该结构。

截至本次精确检索，没有找到可核验的同名 “ChronoTransport” On-TAD
论文；该词继续只指本项目已经被 C 实验否证的 previous→current
prediction-only transport 思路，不作为外部论文或新颖性主张。

### M8 第四版自身画像放行

`1177720` 在 exact `d87a116` 上完成四条 profile，稳定性、未训练
FIXED/REMATCH 推理等价、严格因果和预算门全部通过：

- FIXED/REMATCH 训练均值 `0.716867/0.718284 秒/step`；
- FIXED/REMATCH calibration 推理均值 `0.205037/0.199186 秒/step`；
- 安全系数前双臂总量 `1.159255 GPU·h`，乘 `1.25` 后
  `1.449069 GPU·h < 2.0 GPU·h`；
- 两条训练画像均为零 runtime capacity exhaustion；
- 三项 margin 均产生非零画像损失，transport 为零；
- 未训练推理仍为零 emission，且 future-end/source、负延迟和非单调
  emission 违规均为零。

`pilot_contract.json` 已固定 commit、双臂配置、seed 705、一轮、
calibration-only、无 threshold search、无 reporting/raw-RGB，并记录
四端口块。画像通过后，作业已进入正式 FIXED 一轮训练。

### M9 第四版结果判读规则（训练结果出现前冻结）

第四版只有现有 `screen_gate.json` 同时通过 activation、technical、
budget、provenance 才能进入 P1；任何诊断改善都不能替代 technical pass。
若技术门失败，按以下互斥优先级决定下一模型，不进行本轮权重或阈值搜索：

1. 若两臂 birth TPR 仍为零，但 REMATCH birth/alive AUC 保持至少
   `0.70/0.70`，判为“排序存在、决策校准不足”；下一候选是独立的小型
   current-label calibration head，推理阈值仍固定 0.5。
2. 若 birth 已有 crossing，而任一臂 end TPR 为零或 end AUC 小于
   `0.55`，判为结束边界瓶颈；进入 M7 的 causal transition-end /
   past-start factorization。
3. 若 REMATCH birth 或 alive AUC 低于 `0.60`，判为三头 margin 破坏
   表示；直接拒绝本候选，不做 0.1 周围的调权 sweep。
4. 若出现任何 GT-birth/runtime collision、监督耗尽或 skipped update，
   先修 lifecycle/capacity；不得用 mAP 或 AUC 掩盖技术错误。

仅当双臂非零最终区间、prediction/GT、Recall 和所有完整性门均合格，
才讨论多轮收敛；本表不授权 reporting、多种子或 raw-RGB。

### M10 第四版 FIXED 结果

`1177720` 的 FIXED 一轮完成 `2010/2010` 更新、`scheduler_steps=2010`、
零 skip 和零监督耗尽。birth/alive/end margin 分别在
`1195/1284/1193` 次更新中非零，均值
`0.101839/0.593518/0.044138`，transport 始终为零；机制激活正确。

但训练路径出现 `1` 次 GT-birth/runtime entry-free collision，
active abandonment 为 `23`；冻结 calibration 仍为 `0` committed
prediction、`prediction/GT=0`、`Recall@0.3=0`、average mAP `0`。
因此第四版已确定不可能通过双臂 technical gate。作业继续 REMATCH 和
双臂 v2 诊断，目的只剩下判断三头 margin 是否保住 REMATCH 排序，以及
下一步应走 M9 的哪一条失败分流；不授权调权或后续结果阶段。
