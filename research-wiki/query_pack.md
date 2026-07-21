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

## 2026-07-21 07:35 恢复点

- A `1177711`、B `1177712`、C `1177708` 均已完成完整 FIXED/REMATCH
  产物链；三版双臂最终区间均为零。
- 冻结比较器输出 `selected_variant=null`、`next_stage_authorized=false`、
  `raw_rgb_authorized_next=false`。旧三版不进多轮、多种子或 raw-RGB。
- A/REMATCH birth/alive AUC 为 `0.778/0.823`，说明已有排序信号，但
  birth positive 最大值 `0.495894` 尚未越过冻结 0.5；B 的 0.5 birth
  margin 把 REMATCH 三通道压到约随机，C transport 也被拒绝。
- 下一版不是放大 B，而是 exact `d87a116` 的 lifecycle 小权重版本：
  birth/alive/end margin 均 `0.25`、权重均 `0.1`、transport 为零。
- N16R4 exact `d87a116` 上 `126 passed in 74.03s`，clean detached；
  Slurm job `1177720` 已在 `g0003` 运行，run 为
  `/data/run01/sczc063/yuzibo/runs/persistent_binding/model_opt_lifecycle_seed705_20260721_073454`。
- 提交器现在按 job id 分配四端口块；`1177720` 使用
  `50880–50883`。下一恢复点先看自身 profile gate，再看三项 margin
  activation、双臂最终区间与三通道诊断。
- 若 lifecycle 仍是“birth 接近 0.5、end 失败”，下一模型候选只吸收
  MATR 的“当前 end、过去 memory 检索 start”分解和 OpenHOUSE 对相邻
  动作边界的启示，做 strictly causal transition-end 分支；不复制它们的
  anticipation、NMS、VLM 或层级任务。
- `1177720` 自身画像已 PASS：FIXED/REMATCH 训练
  `0.716867/0.718284 秒/step`，安全系数后双臂
  `1.449069 GPU·h < 2`，零容量/因果违规；现处于 FIXED 正式训练。
- 训练结果前已冻结失败分流：排序保留但 birth TPR=0 → 独立校准头；
  birth crossing 后 end 仍弱 → causal transition-end；REMATCH
  birth/alive AUC 任一 `<0.60` → 拒绝三头 margin、不做调权 sweep。
- lifecycle FIXED 已完成：三项 margin 正确激活、`2010/2010`、零
  skip/监督耗尽，但 `1` 次 runtime 空槽碰撞且 calibration 仍为零区间；
  technical gate 已无法通过，REMATCH 仅用于冻结失败分流。

## 2026-07-21 08:43 最终恢复点

- 第四版 `1177720` 已完成 exact `d87a116` 的双臂整链路；
  `FAILED 1:0 / 01:07:11` 是预设科学拒绝，不是程序崩溃。
- FIXED/REMATCH 都是 `2010/2010`、零 skip/监督耗尽，三项 margin
  activation 通过，实际双臂 `1.119167 GPU·h`。
- FIXED birth/alive/end AUC 为 `0.842/0.863/0.489`，REMATCH 为
  `0.733/0.758/0.557`；所有冻结 0.5 TPR 均为零。
- 双臂均 0 最终区间、ratio/Recall 为零，且各有 1 次 runtime
  entry-free collision；`technical_pass=false`，不放行 P1、reporting、
  多种子或 raw-RGB。
- 代码复核确认 released slot 的下一步复用是原设计的一部分，不能改成
  同步 release-before-birth。下一容量候选保留 entry-free birth pool，
  按 census 的最大可见 4 + 同一步最多 birth 2 注册 6 槽 transition
  reserve，并审计常驻/预留占用；6 槽仍碰撞就回到 lifecycle，不继续扩容。
- 容量门干净后做独立 current-label calibration head，阈值仍固定 0.5；
  只有 birth crossing 后 end 仍弱，才做 causal transition-end /
  past-start factorization。
- 09:00 汇总页：
  `research-wiki/experiments/ontad-model-optimization-0900-report-20260721.md`。
- reserve6 已实现成对配置、activation 和 Slurm 提交入口；本地
  `18 passed`、Python 编译/Bash 语法通过。尚未提交 GPU；下一恢复点先做
  N16R4 exact-SHA 全套测试与自身 profile，均通过后才跑一轮双臂。
- 远端 independent clean exact `369e263` 的目标套件也为 `18 passed`；
  扩大旧相关套件为 `101 passed, 1 failed`。唯一失败是 legacy 测试把
  `num_slots=1` 与默认 `max_births_per_step=2` 组合，和既有参数校验冲突；
  reserve6 未改该逻辑，但在测试合同修清前不提交 profile。
- 旧测试 helper 已在 `380bc16` 仅按测试槽数收紧 birth 上限；远端 clean
  exact 扩展套件最终 `102 passed in 48.75s`。reserve6 代码 preflight
  通过，下一恢复点为 Slurm 双臂自身 profile，profile 前不训练。
- reserve6 已部署为 Slurm `1178040`，run
  `/data/run01/sczc063/yuzibo/runs/persistent_binding/model_opt_reserve_seed705_20260721_093412`，
  initial `RUNNING/g0030`，exact `380bc16`。先看 profile gate；只有画像与
  2 GPU·小时预算通过，作业才会进入一轮 FIXED/REMATCH。
- 当日下午恢复点：按新设计页依次完成 E/reserve6、F/单调校准器、G/当前
  转移判终+历史起点检索。F 的校准损失只训练正 scale/bias，raw 排序、匹配
  和原损失不变；G 只看 `q_(t-1), q_t, q_t-q_(t-1), x_t` 与已见 memory。
  三版均保持 frozen 0.5、reporting 锁定、raw-RGB 禁止，关键 SHA/job/run/gate
  必须逐项回写 Wiki。
- F 已完成代码/远端验证并部署：exact `49b06af`，N16R4 clean 套件
  `133 passed in 90.20s`，Slurm `1178214`，run
  `model_opt_calibration_seed705_20260721_095351`。恢复时先查它的 profile gate；
  raw/calibrated logit AUC delta 必须三通道为零，scale 必须全为正。
- G 已完成代码/远端验证并部署：exact `8d82893`，含因果重放/合规的 clean
  套件 `154 passed in 93.86s`，Slurm `1178279`，run
  `model_opt_boundary_seed705_20260721_100229`。恢复时检查 profile gate、四项
  预期辅助损失激活、两臂 endpoint pointer decisions > 0、past-only=true、
  runtime_state_contains_gt=false，再看 end AUC/0.5 TPR/最终发射。
- D 的完整 score 分布给出下一条预注册路线：E/F/G/H 做 calibration×boundary
  2×2。H=F+G，因为 G 单独仍受 birth/alive 最大概率低于 0.5 阻断，F 单独
  又不能改变近随机 end 排序。H 必须同时过单调 AUC 不变门和 endpoint
  past-only/no-GT 门；禁止借组合实验搜索阈值。
- H 已实现并部署：exact `84da20f`，N16R4 clean 因果扩展套件
  `155 passed in 97.39s`，Slurm `1178354`，run
  `model_opt_boundary_calibration_seed705_20260721_101605`。恢复时先查 profile；
  activation 必须是 3 margin + 3 calibration + pointer，随后同时核验 F/G 两套
  诊断门，再与 E/F/G 做 2×2 主效应和交互解释。
- 四条 run 到齐后使用 `tools/compare_persistent_binding_factorial.py`，输入
  reserve/calibration/boundary/interaction 四个目录。比较器会自动输出两臂
  主效应/交互及 `multi_seed_authorized_next`；其
  `raw_rgb_authorized_next` 冻结为 false。
- H v1 `1178354` 已在 profile fail-closed：`2.113608 GPU·h > 2.0`，无训练；
  不是崩溃。G 同门仅 `1.706565 GPU·h`，因此修订 calibration loss 为
  episode-batched balanced BCE，消除每-token 三次 BCE 启动；必须同时重跑
  F-v2/H-v2 保持 2×2，安全系数和预算上限不变。
- E/reserve6 `1178040` 已完整结束：两臂 `2010/2010`、实际 `1.13361 GPU·h`、
  零容量冲突/监督耗尽/协议违规，但仍为零最终区间。REMATCH birth/alive/end
  AUC=`0.7604/0.7600/0.5637`，birth 正例最大值 `0.4466`，因此容量轴关闭，
  继续 F2/H2 的刻度×边界路线。
- batched-v2 已在 exact `5a11881` 实现：训练 chunk 内三通道各一次 balanced
  BCE，raw detach、固定 0.5、匹配与因果 runtime 不变；F2/H2 成对配置和
  因子比较 alias 已注册。N16R4 clean exact 扩展套件
  `160 passed in 88.32s`。恢复点是提交 F2/H2 自身 profile，并只接受冻结
  2 GPU·小时门内的运行。
- epoch-1 gate 已分层：固定 0.5 在一轮仍完整记录，但零发射只令
  `operational_pass=false`；只要协议/因果、监督、容量、更新和预算完整，
  `learning_readiness_pass` 可授权成对 12 轮，绝不直接授权 multi-seed。
- G `1178279` 已结束：FIXED 三通道 AUC=`0.8835/0.9023/0.6575`，REMATCH
  `0.6671/0.6592/0.6161`；pointer 严格 past-only/no-GT，但一轮仅
  `3/480`、`2/480`，应在多轮中判断收敛，不能用一轮零发射证伪结构。
- F2/H2 `1178448/1178449` 的画像均通过，安全估算分别
  `1.50787/1.72583 GPU·h`，exact `af58538`，均已进入一轮训练。恢复时先收口
  两臂完整更新、单调 AUC 门、past-only/no-GT 与 score 分布，再生成分层门。
- `1178448/1178449` 后续因共享盘 `Errno 122` 中断，不属于模型结果。可重建旧
  checkpoint 已按 SHA 清单释放空间，写探针与约 159 GiB 余量均确认；exact-2a
  checkout 已通过 `162 tests`。零秒启动失败 `1178500/01` 不复用，当前有效重试
  是 F2 `1178504` 与 H2 `1178505`，run 时间戳分别 `125630/125631`。
- H2 一轮若通过 `learning_readiness_pass`、单调门、past-only/no-GT 和激活门，
  立即使用正式成对 Slurm 链路从 seed 初始化训练 12 轮；只在 calibration 看
  3/6/9/12，epoch 12 才执行固定 0.5 正式门。该链路具备成对提交回滚、节点本地
  checkpoint、受控回传和依赖式 pair gate；通过后下一步仍是 feature 多种子，
  不是 reporting 或 raw-RGB。
- 正式链路代码 exact `f3958b45fba4d9d2cd83ccd0cb604e87e043c6f4` 已在共享
  clean checkout 通过 `167 tests in 79.84s` 与写探针；当前没有代码门阻塞，
  唯一启动条件是 H2 `1178505` 完成一轮并通过 learning-readiness、activation、
  monotone-AUC、past-only/no-GT。
- E/G 已从原始 screen result 非破坏式补出 `screen_gate_v2.json`，旧门与新旧
  SHA 都保留；两者均 learning-ready 但固定 0.5 一轮静默。最新因子比较器优先
  读取 v2 companion，待 F2/H2 完成后即可公平生成 batched 2×2 结果。
- 正式 Slurm 已做无提交 `sbatch --test-only`：集群要求 finalizer 也申请 1 GPU
  且禁止显式覆盖默认内存。脚本已修复并再次通过；finalizer 的最坏 0.5 GPU·h
  被保守计入 16 GPU·小时总门，不会漏算资源。
- F2/H2 `1178504/05` 已完整收口并均通过 learning-readiness、activation 与机制门；
  一轮固定 0.5 静默不作永久淘汰。H2/REMATCH birth/alive AUC=`0.3254/0.2719`
  是必须在多轮观察的风险信号，不允许据此事后改门或单臂调参。2×2 比较只授权
  paired 12-epoch feature training，仍未授权 multi-seed/reporting/raw-RGB。
- 当前正式恢复点：exact `a6e16a88bd5f95e284a76859ef2a4106f6b5fd27`，远端
  `170 passed in 80.68s`；FIXED `1178653`、REMATCH `1178654` 正在 `g0066`
  运行，依赖终检 `1178655` 等待，run 为
  `/data/run01/sczc063/yuzibo/runs/persistent_binding/formal12_boundary_calibration_batched_seed705_20260721_141835`。
  下一步只监控 calibration 的 3/6/9/12 曲线、完整性和资源；epoch 12 自动执行
  固定 0.5 正式门，通过后才进入 feature 多种子。
- 正式第 3 轮恢复点：FIXED `1178653` 与 REMATCH `1178654` 已完成三轮完整
  `[02009/02009]` 日志并进入第 4 轮；节点本地 `epoch_2.pth` 均已只读确认存在、
  大小 `17,512,977` bytes。三轮末 loss 均持续下降且无 fatal。冻结编排会在
  12 轮训练全部完成后再统一对 3/6/9/12 checkpoint 做 calibration-only 推理，
  所以当前没有提前选模或访问 reporting；下一关键记录为第 6 轮 checkpoint。
- 正式第 6 轮恢复点：双臂已经进入第 7 轮，六轮末 loss 持续降至 FIXED
  `1.6392`、REMATCH `1.5142`，学习率共同降至 `1.0e-4`；节点本地
  `epoch_5.pth` 均确认存在且为 `17,512,977` bytes。无主作业 fatal，整体已过
  50%；仍等待 12 轮结束后统一进行 calibration-only 评估，下一节点为第 9 轮。
- 正式第 9 轮恢复点：双臂已进入第 10 轮，节点本地 `epoch_8.pth` 均确认存在、
  大小 `17,512,977` bytes；第 9 轮末 loss 为 FIXED `1.3719`、REMATCH
  `1.2894`，共同 LR=`3.0e-5`，主作业无 fatal。FIXED 相对第 8 轮的单 minibatch
  `+0.0090` 波动不作过拟合或调参依据；下一节点是训练结束后的 3/6/9/12
  calibration-only 重放、完整训练审计和固定 0.5 成对终检。
- 最新恢复点 M34：`1178653/54` 已完整训练 12 轮，失败只在 calibration 输出
  零长度区间；FIXED epoch-3 发射 371 条、REMATCH epoch-6 发射 7041 条，均为
  零未来违规，因此不能写成模型静默或科学拒绝。same-step birth+end 现在向左量化
  为一个已观察 feature cell，并由写出与审计双层强制正长度。旧 `/tmp` checkpoint
  在失败后被回收，必须同协议重训；新脚本会在校准前验证 24,120 次完整更新并
  原子保存 3/6/9/12 checkpoint、audit、配置和 SHA recovery 清单，再从共享副本
  校准。旧终检 `1178655` 已取消。恢复顺序是：本地静态/CPU-safe 门 → N16R4
  clean exact Torch 全套与 Bash/test-only → 同 seed 双臂 12 轮重试。0.5、划分、
  reporting 锁和 raw-RGB 禁止状态都不变。
- 最新执行点 M35：修订 exact `22c4028aa29db6cacdb4f057e830c43b68e673ab`
  已在独立 N16R4 checkout 通过写探针、编译、Bash 和 `174 passed in 79.11s`；
  前置合同及训练/finalizer 资源 test-only 也通过。新 run 为
  `/data/run01/sczc063/yuzibo/runs/persistent_binding/formal12_boundary_calibration_batched_seed705_20260721_202804`，
  jobs 为 FIXED `1178956`、REMATCH `1178957`、终检 `1178958`。启动时 FIXED
  在 `g0003` 训练，REMATCH 等 Priority；自动监控脚本切换到
  `C:\tmp\ontad_formal12_progress_22c4028.ps1`。下一个关键点是双臂状态对齐和
  第 3 轮 checkpoint；第 12 轮后必须先看到两份四 checkpoint recovery 清单，
  才接受后续 calibration/终检。
- M36 状态变化：FIXED `1178956` 已在 `g0003` 跑到 epoch-1 `01100/02009`，
  REMATCH `1178957` 已从 Priority 转为 `RUNNING/g0045` 并到 `00550/02009`；
  两臂 fatal=0，终检 `1178958` 正常等待。recovery 当前 pending 是因为尚未完成
  12 轮，不是故障；下一次写 Wiki 的常规节点为双臂第 3 轮 checkpoint，若此前
  作业状态或 fatal 改变则立即提前记录。
- 21:00 恢复摘要：两臂均健康运行在 `g0003/g0045`，FIXED 约 9.37%、REMATCH
  约 7.05%，速度 `1.200/1.188 updates/s`，预计纯训练在 02:04/02:17 左右结束；
  所有已见 loss 有限，零 fatal/NaN/OOM/CUDA/traceback，内存和共享盘余量健康。
  FIXED epoch-1 末 loss=`2.9177` 与修订前一致，正长度解码修订未显示训练轨迹
  漂移。当前阶段仍是 feature-level seed-705 12轮可执行性门；没有 calibration
  结果、没有 reporting、没有 raw-RGB。下一硬节点依次是 epoch 3/6/9/12、两份
  recovery 清单、四 checkpoint calibration-only 曲线和固定 0.5 成对终检。
- 第 3 轮恢复点 M38：两份节点本地 `epoch_2.pth` 均为 `17,512,977` bytes，
  两臂已继续第 4 轮且 fatal=0。FIXED 前三轮末 loss=`2.9177/2.2809/1.9829`，
  REMATCH=`2.9213/2.3640/2.0649`，与修订前逐项完全一致；前三轮范围内训练图、
  初始化和优化轨迹没有漂移。当前不做 calibration，recovery 要等 12 轮完整审计
  后才共享落盘；下一成对节点为 epoch 6。
- 第 6 轮恢复点 M39：两份节点本地 `epoch_5.pth` 均为 `17,512,977` bytes，
  双臂已进入第 7 轮且 fatal=0。前六轮末 loss 与修订前逐项完全一致，50% 训练
  剂量内没有训练轨迹漂移。recovery 继续等待 12 轮完整审计，不提前 calibration
  或 reporting；下一成对节点为 epoch 9。
- 第 9 轮恢复点 M40：两份节点本地 `epoch_8.pth` 均为 `17,512,977` bytes，
  双臂已进入第 10 轮且无 fatal/Traceback/OOM/NaN/non-finite loss。前九轮末
  loss 与修订前十八个对应值逐项完全一致，到 75% 训练剂量未见训练轨迹漂移。
  recovery 继续等待第 12 轮完整审计；其后才运行 3/6/9/12 calibration-only
  重放和固定 0.5 成对终检，当前未访问 reporting 或启动 raw-RGB。
- M41 容量门结论：两臂均完成 24,120 更新后因
  `gt_birth_runtime_entry_free_collisions>0` 被 recovery 门拒绝；无
  calibration、无正式定位指标，终检已取消。这是 reserve6 的模型容量语义失败，
  不是环境或零长度区间故障。census 仍为 max-visible=4、max-birth/step=2、
  GT deficit=0；根因是 6 个物理槽可被预测状态占满。下一修订固定 4 个最大占用
  加 2 个硬 birth reserve，并在完整性裁决前写 evidence-only quarantine；
  不继续扩容，不改 0.5、数据、匹配、reporting 锁或 raw-RGB 状态。
- M42 实现恢复点：硬 reserve 已进入事件头，candidate/active 最多占 4/6 槽，
  每步必须留下 2 个 entry-free birth 槽；同一步 release 不提前复用。审计拒绝时
  会先原子保全 audit、配置、四个 checkpoint 和 SHA-256 到 quarantine，明确不
  授权 calibration。当前 CPU-safe `8+14 passed`，AST/Bash/diff 均通过；Windows
  Torch 仅因 `c10.dll` 无法收集。下一步是提交精确 SHA、N16R4 完整相关套件和
  同提交一轮成对机制/容量验证，通过后才重开 12 轮。旧 12 轮没有过容量门，
  所以没有合法 mAP/Recall；`1.8342/2.0328` 只是末轮训练 loss。
- M43 当前执行点：硬 reserve exact `5edc46c34c0db56e79409fac69276450ad87e949`
  已在 N16R4 独立 clean checkout 通过写探针、编译、Bash 和 `177 passed in
  79.11s`。一轮 H2 成对验证为 Slurm `1179361`，run
  `model_opt_boundary_calibration_batched_seed705_20260722_024414`，初始在
  `g0003` 运行。只检查 same-commit profile、各 2010 更新、零容量碰撞、机制与
  因果/learning-readiness；全门通过才提交新 12 轮。一轮固定 0.5 静默不作
  收敛淘汰，reporting 与 raw-RGB 仍锁定。
- M44 画像门：`1179361` 的 FIXED/REMATCH 训练均值为 `0.820642/0.834076`
  秒/step，安全系数后整对 `1.727043 GPU·h < 2.0`；稳定性、双臂等价和画像
  容量耗尽零均通过。当前 FIXED 到 `400/2009`、fatal=0；两臂训练审计和最终
  gate 尚未生成，故仍无定位性能结论。reporting 只做成本估算、未访问数据。
- M45 FIXED 完成：`2010/2010` 更新与调度，skip/监督耗尽/entry-free collision
  均为零；三项 margin、三项 calibration 与 pointer 均激活，transport 按合同为
  零。calibration 一轮仍 0 发射，但四类因果/时序违规为零，只记 operational
  未收敛。REMATCH 已到 `300/2009`、fatal=0；成对 gate 等待其完成。
