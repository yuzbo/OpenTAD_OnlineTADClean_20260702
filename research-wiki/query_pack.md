---
type: query_pack
updated: 2026-07-21
status: active
scope: Compressed memory to prepend before any new ideation or implementation planning.
---

# Query Pack: Strictly Causal On-TAD

## Fixed Goal

研究对象是标准、全监督、严格因果的 Online Temporal Action
Detection/Localization。决策时刻 `t` 只能读取当前与过去视频证据；模型在线
维护实例出生、持续和结束，并在动作结束后低延时写出不可修改的
`{start, end, class, score}` 最终区间。

不要引入额外传感器、未来终点预测、offline NMS、全视频回改、开放词汇主线，
也不要转向网络安全、宿主或 TCB 叙事。

当前仍是**特征级**实验：输入为冻结的 causal SigLIP2 stride-8、768-d
缓存特征。只有完整特征级技术门与科学门通过后，才进入 raw-RGB
frozen/PEFT/joint 三阶梯。

## Frozen Scientific Comparison

主假设是 first-crossing **FIXED** supervision binding 与 per-prefix
**REMATCH** 的配对比较：

- 两臂共享 first-crossing birth、canonical supervision lifecycle、
  runtime lifecycle、槽位数、数据、优化器、阈值、推理与评测；
- 唯一主比较轴是出生后的 target-to-slot loss binding；
- runtime state 不含 GT identity；推理不接收 annotation、terminal 或
  future fields；
- 最终区间只提交一次，历史输出不能修改。

四槽容量与 candidate lifecycle 已修复。全 411 视频 census：
320,205 token、6,328 实例、最大同一步 birth 2、最大可见并发 4，
所有 birth/end 覆盖，零 oracle capacity deficit。

## Established Execution Evidence

- 科学合同修复 smoke `1177438`：76 tests、真实更新、checkpoint reload
  精确复现 84 条 causal emission。
- repaired profile `1177511`：稳定、严格确定性、未训练两臂推理完全相同；
  12-epoch pair 需约 `12.512897 GPU·h`，被 2-hour cap 拒绝。
- fit-only 正率：birth/alive/end =
  `0.00550069/0.0799036/0.0636912`；所有 birth start 在一个 token 内。
- 优化后 smoke/profile `1177580/1177582` 通过；首个 seed-705 screen
  `1177596` 用 `0.871944 GPU·h`，两臂均 2010/2010 更新但零 emission。
- diagnosis `1177634`：28,730 calibration token/arm，birth 最大值仅
  FIXED `0.433135`、REMATCH `0.346657`，冻结 0.5 crossing 均为零。
- shared `weighted_bce_stationary` prior 修复后 smoke/profile
  `1177637/1177639` 通过；screen `1177653` 用 `0.955278 GPU·h`，
  两臂再次 2010/2010 稳定更新但零 emission。
- 旧 repaired checkpoint 诊断 `1177682` 已完成：FIXED 的 birth/alive/end
  最大分数 `0.1698/0.3202/0.4192`，REMATCH 为
  `0.3262/0.3800/0.3881`，三通道均零次越过 0.5；v1 无 target-conditioned
  AUC，只作为旧模型静默证据。

这些证据只定位共同 birth/lifecycle 瓶颈，不证明 FIXED 的论文效果。

## Current Three Model-Optimization Pilots

实现提交：`7ba049f530c5fac856de3c4d527aebfc8666fb04`。
部署合同收紧提交：
`1b93a7d6421e38c89601580ca34d5003b1859cff`。

三个版本均为 seed 705、一个 epoch、fit-only、feature-only；每个版本内部
都有严格配对的 FIXED/REMATCH，B 与 C 不叠加：

1. **A / SW**：只用 `warmup_epoch=0.1`。检验旧一轮直到最后更新才达到
   peak LR 是否造成有效 LR 暴露不足。
2. **B / SW+BM**：A 加 birth-frame balanced logit margin（weight 0.5，
   margin 0.25）。只在当前 first-crossing 有正 birth 时启用，正槽推到
   logit `+0.25`，同一步负槽推到 `-0.25`；不读未来。
3. **C / SW+CT**：A 加 causal query transport（weight 0.05）。上一有效
   token queries stop-gradient 后向当前 queries 做 Sinkhorn 软传输；
   边际来自 prediction-only
   `alive*(1-birth)*(1-end)` mass，代价为 cosine distance 加同槽位时间
   identity prior；不使用 GT identity 或未来标签。数值设置为 temperature
   `0.25`、identity cost `0.25`、56 次 log-space Sinkhorn；一段内全部
   相邻 token 对批量计算，和逐对损失严格等价；A/B 权重为零时不构造
   transport 边际，C 的边际在 no-grad 下构造。

“ChronoTransport”仅作为时间有序的 past-to-current optimal-transport
思想。未核实到唯一对应的同名 On-TAD 论文，不得冒认。直接依据是
CausalTAD 的因果方向、MATR/HAT 的历史上下文、ActionSwitch 的在线状态
保守性，以及 temporally consistent OT 的时序传输原则；HAT/MATR 的
anticipation/future-supervised 部分不采用。

ICCV 2025 OpenHOUSE 进一步支持当前 strict OAD-based On-TAL 口径：
结束即输出、已输出区间不可回改。其 actionness-start + progress-drop-end
专门处理无背景间隔的相邻动作，但与本项目任务/数据不同；只在本轮显示
“birth 已恢复、end/相邻动作仍失败”时，才作为 progress-hazard 后续候选，
不采用 VLM、层级标签或伪标签。精确检索仍未找到可信的同名
“ChronoTransport” On-TAD 论文。

2026 前沿边界：OZ-TAL 的 zero-shot/training-free VLM 路线支持
“实例级 online localization”趋势，但不是本项目设定；OnPoint 是点监督、
离线教师与 anticipatory distillation，也不吸收到标准全监督主线。

验证：

- Windows Python 编译、Bash syntax 与 29 项 CPU-safe 测试通过；本机
  Torch `c10.dll` 故障不是模型门禁。
- N16R4 exact `7ba049f`：41 项 Torch/配置/提交器测试通过，47.37 秒；
- N16R4 exact `1b93a7d`：收紧提交器后 38 项测试通过，40.32 秒；
- N16R4 exact `3035f4e`：新增结构化损失激活审计后 116 项相关测试
  通过，77.56 秒；
- N16R4 exact `22aa93b`：加入 transport 收敛修订和比较器预检后
  120 项测试通过，76.64 秒；
- N16R4 exact `f6bd9e1`：最终 transport 温度/迭代边界与回归测试就绪，
  128 项相关测试通过；候选仍为 clean detached。
- N16R4 exact `5c02ccc`：把逐 token Sinkhorn 改为段内批量等价计算，
  新增批量/逐对计划、损失和梯度回归；130 项相关测试在 80.42 秒通过。
- N16R4 exact `d390779`：隔离 A/B 的零 transport 计算路径，新增禁用
  分支回归；131 项相关测试在 81.02 秒通过。
- 七次远端测试后 worktree 都是 clean。

每条 Slurm pilot 会先在自身 exact commit/config 上跑双臂
50-warmup/200-measured profile；只有自身稳定性、未训练因果等价与
one-epoch 2 GPU-hour gate 通过，才继续完整训练、calibration inference、
target-conditioned diagnosis 和 frozen technical gate。

## Deployment State

远端 clean detached worktree：
`/data/run01/sczc063/yuzibo/projects/OpenTAD_OnlineTAD_ShortWarmup_b61f56a`，
当前 exact `d390779443bccc4926bc8fb2bff1875834383c21`。

账户 association 固定 `GrpTRES=gres/gpu=16`、`MaxSubmitJobs=16`。
2026-07-21 04:09 北京时间作业数降到 12；完成 job-name、run-dir 和
pilot-contract 三重去重后，三条 exact-commit pilot 已提交：

- A/SW：job `1177693`，run `model_opt_sw_seed705_20260721_041045`；
- B/SW+BM：job `1177694`，run
  `model_opt_margin_seed705_20260721_041046`；
- C/SW+CT：job `1177695`，run
  `model_opt_transport_seed705_20260721_041047`。

完整 run root 均为
`/data/run01/sczc063/yuzibo/runs/persistent_binding/`。提交后账户为
15/16，三条均在 `squeue`。随后在它们仍为 PENDING、零 GPU 时发现 C 的
temperature 0.1/20-iteration Sinkhorn 边际未充分收敛；为保持同一提交，
旧三 job 受控取消并保留目录审计。temperature 0.5/20 虽收敛，但真实
query 模拟的同槽质量只有 `0.46641`；0.25/20 为 `0.60779`，身份保持
更强。2,000 个四槽压力样本中，0.25/55 的最坏边际误差
`1.05056e-3` 未过门，0.25/56 为 `9.84639e-4` 且零超标，故冻结
0.25/56。随后把一段内 `T-1` 个相邻传输从逐对各跑 56 轮改为批量共享
56 轮，损失与梯度语义不变，避免大量 4×4 小 kernel。05:03 账户仍为
16/16；随后移除 A/B 的无用边际计算。新三版本等待槽位按 A→B→C 重提；
只认 exact `d390779`，旧取消目录不算 duplicate。

05:37 后依次提交 A `1177706`、B `1177707`、C `1177708`；三条自身
profile 全过，A/B/C 安全系数后预计
`1.3966/1.4108/1.4080 GPU·h`，均低于 2。连续 job id 暴露出提交器的
端口块重叠：作业内每次 `torchrun` 把端口 `+1`，A 的 FIXED 校准因此与
B 的 FIXED 训练相撞，B 的下一阶段又与 C 相撞。A/B 均已先完成
`2010/2010` FIXED 更新、零跳步/监督耗尽/容量碰撞后，才在校准启动处
以 `Address already in use` 退出；这两条保留为启动诊断，不进入比较。

模型、配置、数据、seed 和 exact `d390779` 均未改变。使用显式不重叠
端口块重新通过 Slurm 提交最终 A/B：

- A/SW：job `1177711`，run
  `model_opt_sw_seed705_20260721_062012`，端口 `45100–45103`；
- B/SW+BM：job `1177712`，run
  `model_opt_margin_seed705_20260721_062013`，端口 `45200–45203`；
- C/SW+CT：继续 job `1177708`，run
  `model_opt_transport_seed705_20260721_054210`。

C/FIXED 已完成：`2010/2010` 稳定更新，transport 在全部更新中激活、
均值 `0.0108388`，margin 为零；但出现 5 次 GT-birth/runtime 容量碰撞，
calibration 在冻结 0.5 阈值下提交 0 个区间。C 正常进入 REMATCH，
不是程序崩溃；该结果已指向“transport 促使候选占用容量但未打通
end/commit”的风险。最终判断仍等待两臂三通道诊断。A/B 恢复作业已经
通过各自 exact profile 并进入确定性 FIXED 训练。

C/REMATCH 随后也完成 `2010/2010`，transport 均值 `0.0114319`，但有
2 次容量碰撞；FIXED/REMATCH 的冻结 calibration 均为 0 committed
prediction、0 Recall@0.3。C 已确定无 P1 资格，作业继续生成三通道诊断
和完整拒绝产物；这些分数只决定是否保留“仅 confirmed-active 槽
transport”的后续想法，不会挽回本版资格。

C 最终 `FAILED 1:0` 是预期 scientific gate reject：activation 和
`1.08 GPU·h` budget 均通过，technical gate 拒绝。FIXED 的
birth/alive/end AUC 为 `0.313/0.296/0.566`，birth/alive 均值差为负；
REMATCH 为 `0.552/0.482/0.530`，所有通道冻结 TPR 仍为零；两臂 birth
最大概率仅 `0.22687/0.14236`。因此停止 C 及其权重/作用域搜索，下一
模型选择只由 A/B 结果决定。

三版 FIXED 均零输出后，`6b0ffd63c07d6c9ea0db1ddcb8055ad90b7d115d`
已加入默认关闭的 alive/end balanced logit margin，与现有 birth margin
组成候选 lifecycle decision margins；三者都只用当前监督 mask，不读
未来。原 A/B/C 的新权重由测试锁为零，正在运行的实验仍是 exact
`d390779`。本地编译和 11 项 CPU-safe 配置/activation 测试通过；具体
启用权重等待 A/B v2 诊断后冻结，再做 N16R4 测试、画像与 Slurm pilot。

每臂训练审计必须写出 `mean_losses` 与 `loss_nonzero_updates`，随后生成
`optimization_activation.json`：SW 两项新增损失都必须休眠，margin
只激活 `birth_margin_loss`，transport 只激活
`causal_transport_loss`。未激活或串扰是科学 gate reject，不等同于
程序崩溃；推理和诊断产物仍须保留。

三版本选择规则已在结果出现前冻结于 `f742dce`：先要求 activation、
technical screen、budget、provenance、calibration diagnosis 全过；再按
“最差臂 Recall@0.3 → 两臂三通道最低 AUC → 最差 birth TPR → 最差 birth
gap → 较高 birth FPR → GPU·小时”字典序排名。若全拒绝则不选 winner，
回到 head/lifecycle；有 winner 也只进入 P1，不授权 raw-RGB。

## Pilot Gates

每个版本/每臂必须：

- nonzero committed predictions；
- prediction/GT in `[0.25, 4.0]`；
- Recall@tIoU 0.3 至少 `0.25`；
- zero causal violations、GT supervision exhaustion、
  GT-birth/runtime collision 和 skipped updates；
- successful updates = expected updates = scheduler steps；
- 报告 birth positive/negative gap、pairwise AUC、冻结 0.5 下 TPR/FPR；
- 不搜索/下调阈值，不访问 reporting。

P0 pilot 只回答“共享模型能否可靠开门并完成 lifecycle”。若 B 或 C 单独
通过，再做预注册的组合/多轮收敛实验；若都失败，修改 head/lifecycle
表示，不修改评测门槛。

## Path to Paper

1. P0：当前 A/B/C 技术优化 pilot。
2. P1：胜出版本的 seed-705 多轮收敛与 calibration freeze。
3. P2：seeds 705/706/707 的 feature-level FIXED/REMATCH 主实验。
4. P3：冻结 checkpoint 的一次性 reporting，给出 standard mAP、
   budgeted AP、延时和实例错误主表。
5. P4：warmup/margin/transport/binding 单因素消融及相邻、重复同类、
   重叠、长短实例失败分析。
6. P5：仅在 feature science gate 通过后，做 raw-RGB
   frozen encoder → PEFT → joint causal training。

三种子科学门：

- `E_id = 0.5 * (duplicate_rate + fragmentation_rate)`；
- FIXED 相对降低至少 20%，且至少 2/3 seeds 改善；
- standard average mAP delta 不低于 `-0.5` percentage points；
- 全部报告 paired video bootstrap uncertainty、资源和因果审计。

## Claim Limits

CAG-QIL/SimOn/OAT/MATR/HAT/ActionSwitch 是直接 On-TAL 邻近工作；
TrackFormer/MOTR 已占据 persistent query/identity assignment 先例；
CausalTAD、E2E-LOAD、StreamFormer 已占据因果 temporal/raw-video
表示先例。

不要把 persistence、memory、lifecycle、causal attention、end-to-end、
frozen features、LoRA 或 backbone 本身声称为新颖。仍存活的窄主张是：
在完全匹配的 strict On-TAD 协议下，first-crossing persistent FIXED
supervision binding 能否在不显著损害 standard mAP 的前提下，降低
identity-linked duplicate/fragmentation error。
