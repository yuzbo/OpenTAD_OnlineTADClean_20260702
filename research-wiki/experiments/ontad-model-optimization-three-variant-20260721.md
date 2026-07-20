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
- calibration-only 分数诊断 `1177682` 仍因账户 16-GPU
  `AssocGrpGRES` 满载排队；不取消、不重复提交。
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
- [x] M4：提交精确代码提交并部署三条 Slurm 作业。
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
| A / SW | `1177693` | `/data/run01/sczc063/yuzibo/runs/persistent_binding/model_opt_sw_seed705_20260721_041045` | PENDING / Priority |
| B / SW+BM | `1177694` | `/data/run01/sczc063/yuzibo/runs/persistent_binding/model_opt_margin_seed705_20260721_041046` | PENDING / transitional None |
| C / SW+CT | `1177695` | `/data/run01/sczc063/yuzibo/runs/persistent_binding/model_opt_transport_seed705_20260721_041047` | PENDING / transitional None |

提交后三条均出现在 `squeue`，账户活跃作业为 15/16；没有取消、修改或
抢占任何无关作业。心跳已从“寻找提交槽”切换为“只读监控三条 pilot 与
旧诊断 `1177682`”，明确禁止重复提交。
