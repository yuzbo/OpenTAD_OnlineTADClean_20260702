# On-TAD 模型优化 09:00 进度报告（2026-07-21）

## 一句话结论

本轮已经完成并部署四个严格因果、全监督、特征级 FIXED/REMATCH 模型版本。
A/B/C/D 均已完成训练、校准、三通道诊断、资源与最终门禁；四版都被
科学门拒绝。当前没有任何版本获准进入多种子、reporting 或 raw-RGB。
最有价值的新结论是：D 保住了 birth/alive 相对排序，但没有把任何
lifecycle 分数校准到冻结 0.5，并且双臂各有 1 次运行时无空槽碰撞。

## 任务目的与输入

目标是标准 On-TAD：模型在时间 `t` 只能读取当前和过去的视频信息，在线
维护动作实例的开始、持续、结束和槽位释放；动作结束时低延时输出不可回改
的最终区间。

当前是**特征级实验**，输入为固定、审计过的 causal SigLIP2 stride-8、
768 维 RGB 视频特征，不是 raw-RGB 联合训练。这样先把实例监督、持久
query、生命周期头和评测协议验证清楚；只有特征级多种子主实验通过，才进入
raw-RGB frozen encoder → PEFT → joint causal training。

## 已实现和部署的模型版本

| 版本 | 模型改动 | exact 训练提交 | 最终作业 |
| --- | --- | --- | ---: |
| A / SW | 只把 warmup 从 1.0 缩到 0.1，峰值 LR 与更新数不变 | `d390779` | `1177711` |
| B / SW+BM | A + 当前 birth 正负平衡 logit margin，权重 0.5 | `d390779` | `1177712` |
| C / SW+CT | A + previous→current prediction-only causal query transport | `d390779` | `1177708` |
| D / lifecycle | A + birth/alive/end 三头当前步 margin，各 0.1×0.25，transport 零 | `d87a116` | `1177720` |

C 的 Sinkhorn 已完成温度/迭代压力测试和批量等价优化，但最终被效果与容量
证据共同拒绝。D 的三项 margin 只用当前 prefix 的监督 mask，不向 runtime
传入 GT identity 或未来信息。

## A/B/C 最终结果

三版均完整生成双臂训练、校准、activation、三通道 v2 诊断、资源和 gate：

| 版本 | 实际双臂 GPU·h | FIXED birth/alive/end AUC | REMATCH birth/alive/end AUC | 双臂最终区间 |
| --- | ---: | --- | --- | ---: |
| A | `1.06583` | `0.451/0.370/0.529` | `0.778/0.823/0.519` | `0/0` |
| B | `1.07333` | `0.474/0.511/0.478` | `0.484/0.500/0.476` | `0/0` |
| C | `1.08000` | `0.313/0.296/0.566` | `0.552/0.482/0.530` | `0/0` |

冻结比较器输出 `eligible=[]`、`selected_variant=null`、
`next_stage_authorized=false`、`raw_rgb_authorized_next=false`。

主要模型结论：

- A/REMATCH 已学到 birth/alive 排序，birth 正样本最大值 `0.495894`，
  但仍未越过冻结 0.5，end 也弱。
- B 略改善 FIXED，却把 REMATCH 三通道压到近随机；不能继续放大 birth
  margin。
- C 的 FIXED birth/alive 判别反转，并出现 FIXED/REMATCH `5/2` 次容量
  碰撞；停止 transport 调权。

## D/lifecycle 最终结果

- exact `d87a116` 的 N16R4 clean detached 套件：
  `126 passed in 74.03s`。
- 自身画像：FIXED/REMATCH 训练
  `0.716867/0.718284 秒/step`；安全系数后
  `1.449069 GPU·h < 2`，画像通过。
- FIXED：`2010/2010`、零 skip/监督耗尽；三项 margin 的非零更新数为
  `1195/1284/1193`，transport 零。
- REMATCH：`2010/2010`、零 skip/监督耗尽；三项 margin 的非零更新数为
  `1195/1336/1190`，transport 零。
- 实际双臂资源为 `1.119167 GPU·h`，低于 `2 GPU·h`；activation 与
  provenance 通过。
- FIXED birth/alive/end AUC 为 `0.842/0.863/0.489`，REMATCH 为
  `0.733/0.758/0.557`。两臂三通道的冻结 0.5 TPR 全为零。
- FIXED/REMATCH 均为 `0` 个最终区间、`prediction/GT=0`、
  `Recall@0.3=0`，且各出现 `1` 次 runtime 无空槽碰撞。
- 最终 `technical_pass=false`、`screen_pass=false`；
  Slurm `FAILED 1:0 / 01:07:11` 是预设的科学拒绝退出，不是程序崩溃。

## 当前路线是否仍与原始设计一致

一致。没有改变任务定义、数据划分、冻结 0.5 阈值、FIXED/REMATCH 比较轴、
最终区间评测或严格因果边界；没有访问 reporting，没有 raw prediction
捷径，没有 raw-RGB，也没有把失败包装成论文结果。

新增 margin、transport 和端口隔离都是受控变量或执行修复。连续 Slurm
job 的四阶段 `torchrun` 曾发生 rendezvous 端口重叠；A/B 用同提交、
同配置的独立端口块重跑，随后提交器改为每 job 自动预留四端口，未改变
科学变量。

## 当前困难

1. 模型能学习相对排序，但固定 0.5 下 birth/end 决策仍不稳定。
2. FIXED 的指定槽判别弱于 REMATCH，说明 slot identity 与内容表示仍有
   冲突。
3. end 是持续瓶颈；只加 birth margin 或 query transport 都不能形成完整
   “出生—持续—结束—提交”链。
4. D 说明更强 lifecycle 梯度会延长占槽；即使静态 census 的最大可见
   实例数不超过四槽，入口冻结和一拍复用延迟仍可能让相邻 birth 暂时没有
   合法入口槽。
5. 一轮 pilot 是非退化筛选，不是论文主结果；没有合格候选就不能越级。

## 从当前实验到论文主实验

| 阶段 | 内容 | 放行条件 |
| --- | --- | --- |
| P0（当前） | seed-705、一轮、双臂特征级模型筛选 | 双臂非零区间、ratio/Recall 合格、零完整性错误 |
| P1 | 胜出模型的单 seed 多轮收敛 | calibration 冻结 epoch，结果持续而非偶然 crossing |
| P2 | seeds 705/706/707 的 FIXED/REMATCH 特征主实验 | 至少 2/3 seeds 改善 identity error，mAP 不显著退化 |
| P3 | 冻结 checkpoint 的一次性 reporting | standard mAP、online AP、延迟、实例错误、完整 provenance |
| P4 | 单因素消融与相邻/重叠/重复同类失败分析 | 每次只改一个机制，报告不确定性 |
| P5（条件） | raw-RGB frozen → PEFT → joint | 仅 P2/P3 通过后启动 |

## 下一模型方向

按结果前冻结的优先级，下一版不是调 margin 权重，而是：

1. 先修 lifecycle/capacity：让同一步的旧实例结束/释放先于新实例出生
   的做法会让一个旧 query 同时描述另一个新实例，因此不采用。保留原设计
   的 entry-free birth pool 和一拍复用延迟，按冻结 census 的“最大可见
   4 + 同一步最多 birth 2”建立 6 槽 transition reserve 候选，并把常驻
   占用与过渡预留分开审计；目标是双臂 collision、监督耗尽和 skip 全为零。
2. 容量门干净后，保留 D 的小权重三头排序约束，新增独立
   current-label calibration head，把已经存在的 birth/alive 排序映射到
   冻结 0.5；不做阈值搜索。
3. 只有 birth 真正 crossing 后 end 仍弱，才实现 previous/current query
   transition-end + past-memory start retrieval。

D 的 REMATCH birth/alive AUC `0.733/0.758` 高于预注册的 `0.70/0.70`，
所以它属于“排序保留、决策校准不足”；但双臂碰撞触发更高优先级的容量
阻断，必须先完成第 1 步。FIXED end AUC `0.489` 也表明结束建模仍是后续
瓶颈，但当前不能越过容量问题直接做结构扩张。

这里的 6 槽不是随意扩容：现有控制器有意只允许“本步入口已经 FREE”的
query 接受 birth，刚释放的 query 下一步才能复用，以免一个 query 同时
解释旧实例 end 和新实例 birth；全量 census 又给出最大可见实例 4、同一步
最多 2 个 birth。因此 `4+2` 是可审计的单步过渡上界，先作为独立容量变量
验证，不改 FIXED/REMATCH 监督比较轴。

截至 08:52，这个 reserve6 候选已直接实现：新增成对 FIXED/REMATCH
配置、冻结容量合同、activation 映射和 Slurm 提交入口；本地相关测试
`18 passed`，Python 编译与 Bash 语法通过。它尚未提交 GPU，因为必须先在
N16R4 对新 exact SHA 完成全套 Torch 测试与自身 profile；这两门通过后
才允许一轮双臂筛选。08:56 的独立 clean detached checkout 已确认 exact
`369e263`，目标套件 `18 passed`；扩大到旧 persistent-event 测试后为
`101 passed, 1 failed`，唯一失败是 legacy 测试以 `num_slots=1` 配合默认
`max_births_per_step=2`，在既有参数合同处被拒绝。测试 helper 已在
`380bc16` 将 birth 上限与其测试槽数对齐，不改生产模型；09:00 远端 clean
exact 套件最终为 `102 passed in 48.75s`。因此代码级 preflight 已通过，
下一门是 Slurm 自身 FIXED/REMATCH profile，尚未直接训练。

前沿方法只作结构启发：MATR 支持把 current-end 与 past-start decoder
分开，但其 anticipation/NMS 不移入本严格协议；OpenHOUSE 说明相邻动作
需要独立 end 证据，但其 progress/VLM/层级任务不直接复制。没有找到可信
同名 “ChronoTransport” On-TAD 论文，该词只作为本项目 C 路线的内部简称。

## 仓库与证据位置

- GitHub：<https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702>
- Draft PR：<https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/pull/1>
- 分支：`codex/ontad-science-fixed-rematch`
- 详细实验 Wiki：`ontad-model-optimization-three-variant-20260721.md`
- 快速恢复入口：`research-wiki/query_pack.md`
- D run：
  `/data/run01/sczc063/yuzibo/runs/persistent_binding/model_opt_lifecycle_seed705_20260721_073454`

当前不需要再发起 Pro 讨论。先让模型形成一个通过特征级双臂技术门的候选；
到 P1/P2 通过、准备冻结论文主张和新颖性边界时，再做一次有明确问题清单的
Pro/审稿式讨论。
