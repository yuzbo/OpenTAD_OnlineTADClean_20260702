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

- CausalTAD：限制时间上下文方向，说明因果时序建模对 TAD 有效；
- MATR：使用过去记忆维护 On-TAL 的长时上下文；
- CAG-QIL：明确要求未来不可见且历史提案不可回改；
- Temporally Consistent Unbalanced OT：把时间一致性先验编码进
  optimal transport；本实验只借鉴该原则，不照搬其无监督分割设定。

## 共同 pilot 门禁

每个版本均部署一条 Slurm 作业，作业内顺序运行 FIXED 与 REMATCH：

1. 精确 40 位提交号、干净 detached checkout 与配置配对检查；
2. 真实冻结特征、seed 705、一个 epoch、每臂 2,010 次预期更新；
3. 零 non-finite、零 skipped update、零 GT supervision exhaustion、
   零 GT-birth/runtime collision、零未来信息违规；
4. 训练后只在 calibration split 做 target-conditioned 分数诊断；
5. 首要排序指标是 birth positive/negative gap、pairwise AUC、冻结 0.5
   下的 TPR/FPR 与 committed emissions；不搜索或下调阈值；
6. reporting split、三种子主结果和 raw-RGB 均继续封锁。

三个版本是并行的机制 pilot，不以其中任一结果冒充 FIXED/REMATCH
论文主实验。若 B 或 C 单独通过技术门禁，再做一次受控组合实验；若均失败，
回到 head/lifecycle 表示而不是修改评测阈值。

## 关键节点

- [x] M0：恢复任务、代码、Wiki、Slurm 和预算上下文。
- [x] M1：冻结 A/B/C 三个版本与不越界合同。
- [x] M2：实现模型损失、配置、测试和实验清单。
- [x] M3：本地 CPU-safe 验证与远端 N16R4 轻量验证。
- [ ] M4：提交精确代码提交并部署三条 Slurm 作业。
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
- Windows 本地完成 Python 编译、两个 Bash 语法检查、配置/工具等
  29 项 CPU-safe 测试；全部通过。本机 Torch 仍因既有 `c10.dll`
  初始化问题不可用，没有把该环境当作模型证据。
- N16R4 干净 detached worktree 在精确实现提交上完成 41 项
  PyTorch/配置/提交器测试，`41 passed in 47.37s`；测试后
  `git status --porcelain` 为零行。
- 首次远端 `git fetch` 遇到 GitHub TLS 中断，第三次有界重试成功；
  这是传输故障，发生在测试前，不是代码或实验失败。

部署前队列检查显示账户已有 16 个 Slurm 作业（9 RUNNING、7 PENDING），
其中既有诊断 `1177682` 仍为 `AssocGrpGRES`。当前没有空余 submit slot；
不取消或修改其他任务，等待至少三个槽位后提交三条独立 pilot。
