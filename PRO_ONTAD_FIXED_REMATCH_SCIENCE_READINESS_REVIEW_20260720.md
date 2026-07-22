# On-TAD FIXED/REMATCH 科学与实验就绪审查

## 审计可见性

**PASS。** 本审查固定到代码提交 `27a59dec445f6b4ed9651abab1168c358a8db7e3`；当前分支上的实验计划仅用于核对“已完成/未完成”状态，不替代冻结代码。已读取设计、监督器、检测器、查询头、数据集、训练/测试入口、标准与实例评测、结果门及聚焦测试。

没有独立运行训练、Slurm 或 55 个聚焦测试；“55 tests passed”仅作为仓库计划文件记录，而不是本审查重新取得的运行证据。计划文件本身也承认端到端相邻动作测试、Slurm smoke、单种子、阈值冻结和三种子结果仍未完成。

## 1. 一句话结论

# **需先修复**

当前实现的**监督态/运行态隔离方向正确，且从缓存特征边界开始的严格因果主链基本成立**；但未监督的 endpoint offset、birth 当步 REMATCH、短动作无法立即提交、锁定 reporting split 被训练入口逐 epoch 访问，以及指标—结果门证据链未闭合，使当前 SHA 产生的单种子结果不可作科学解释。

当前代码最多可做非正式 crash-only 试跑，**不能进入单种子科学筛选**。

## 2. 总体裁决

| 范畴                    | 裁决                  | 核心理由                                                                                                                        |
| --------------------- | ------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| A. 严格因果性              | **基本成立，仅限缓存特征边界之后** | head 仅累积当前/过去特征；模型拒绝 annotation、duration、EOV、future 字段；聚焦测试验证改变后续 token 不影响先前输出。但缓存特征的生成过程在该提交中只有 manifest 声明，没有独立证明。       |
| B. 监督态/运行态隔离          | **核心成立**            | canonical GT birth 完全从 supervision pool 分配，代码不读取 runtime occupancy；任意预测占用不能删除真实 birth target。                               |
| C. birth/lifecycle 修复 | **不完整**             | 无 refractory 是正确修复，但同一 token 内 birth+end 的短动作不会立即提交；释放槽位被冻结到下一步，存在合法相邻动作漏检反例。                                               |
| D. FIXED/REMATCH 单变量  | **配置级成立，语义级未成立**    | resolved configs 的确只差 binding mode 和工作目录；但 REMATCH 会在 birth 当步立即交换 class/start/end target，造成 birth target 与其他 target 分属不同槽。 |
| E. 指标与门槛              | **不成立**             | 现有 runner 不产出 gate 所需完整字段；实例指标的 stream key、坐标系统、fragmentation 定义与冻结设计不一致；主配置计算的是 latency-budgeted AP，不是冻结主 mAP。             |
| F. 整链路 readiness      | **不成立**             | `formal_training_ready=False` 仅是配置字段，没有启动器强制；训练入口仍构建并逐 epoch 评测 locked reporting split。                                     |
| G. 论文实验路线             | **可在有限修复后证伪**       | 这不是结构性 KILL。修复 P0、完成 Slurm smoke 和单种子非退化筛选后，FIXED/REMATCH 仍是一个明确、低成本、可否定的机制实验。                                              |

---

# 3. P0 问题表

## P0：正式 smoke 或任何科学训练之前必须修复

| 编号       | 问题与代码证据                                                                                                                                                                                                                | 触发条件                                                                         | 影响                                                                                                                                       | 最小修复                                                                                                                                                                                                                                               | 必须新增的测试                                                                                                                               |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **P0-1** | **`endpoint_offset_head` 决定最终终点，却没有 endpoint-offset loss。** Head 输出该值，commit 用它计算 `end_frame=current-offset`；训练损失只有 birth/alive/class/start/end BCE，没有 offset 监督。                                                      | 任意动作提交                                                                       | 终点边界由没有直接目标的支路决定，污染 mAP、fragmentation、GT-end delay；FIXED/REMATCH 差异可能被不可识别的终点误差放大或掩盖。                                                    | 当前冻结的是 `endpoint_mode="binary"`，最小且最干净的修复是：**binary 模式下强制 `end_frame=current_frame`，移除或完全忽略 endpoint-offset head**。子步终点回归不得在当前主实验中临时加入。                                                                                                            | 1. 任意改变 `endpoint_offset` 不得改变 binary emission；2. 所有保留的 trainable head 参数均有有限梯度；3. end crossing 在当前 source frame 量化提交。                |
| **P0-2** | **REMATCH 在实例出生的同一前缀就重新绑定。** 监督器先做 canonical birth，再立即对全部 canonical visible instances rematch；已有测试明确期待 birth `{3→0,8→1}`，同一步 loss binding 却变成 `{3→1,8→0}`。                                                             | 两个以上可见实例，rematch cost 发生交叉                                                   | REMATCH 的 birth BCE 在 canonical 槽，而 class/start/end 在另一个槽；runtime birth 候选采用的 start/class 输出可能恰好没有获得该 newborn 的监督。比较不再只是“birth 后是否持续绑定”。 | **当前 step 的 newborn 必须固定在 canonical birth slot；REMATCH 从下一 causal decision 才允许。** 约束求解时保留 newborn 槽，旧实例只在剩余 canonical occupied pool 上 rematch。                                                                                                     | 同时 birth 的交叉代价测试：birth step 两臂全部损失及 target 槽相同；下一 prefix 才允许 REMATCH 发生交换。另加 newborn+旧 ACTIVE、newborn+same-step end 测试。               |
| **P0-3** | **冻结设计要求短动作 candidate 可 immediate-end commit，但代码没有实现。** Decoder 先处理旧 CANDIDATE/ACTIVE，随后才把 FREE 槽设成 CANDIDATE；新 candidate 的同一步 `end` 不再检查。                                                                             | 动作 start 和 end 在同一 stride-8 决策区间内；或 birth 出现在最后一个 token                      | 监督器给出同一步 birth+end target，runtime 却至少等下一 token；若下一步 end 下降或视频没有下一 token，合法动作永不提交。                                                        | 新 candidate admitted 后，若同一步 `end>=threshold`，立即提交并释放；其 start/class 使用当前 causal 输出，end 使用 binary current frame。                                                                                                                                     | 单次 `decode_step` 的 birth+end 测试；最后 token 短动作；同类短动作；短动作后相邻 birth；必须产生且仅产生一个 immutable emission。                                        |
| **P0-4** | **训练入口访问并逐 epoch 评测 locked reporting split。** Base config 的 `dataset.test` 是 211-video reporting manifest，`val_eval_interval=1`；`tools/train.py` 无条件构建 test dataset，并每个 epoch 调用 `eval_one_epoch(test_loader, ...)`。   | 当前配置直接训练                                                                     | sealed reporting split 被重复访问，任何 checkpoint、阈值或人工调试都可能被 test 表现污染。即使不手工选择 checkpoint，也不再是“一次性锁定报告”。                                       | 正式训练进程**不得构建 reporting dataset**。训练配置设 `val_eval_interval=-1`；单独新增 calibration-inference 配置；reporting 配置只能在阈值、checkpoint、代码 SHA 全部冻结后调用一次。正式 checkpoint 固定 epoch 12，不按 reporting 指标选择。                                                             | monkeypatch/open-audit：训练期间只要打开 reporting manifest 或其中 feature 文件即失败；训练日志不得出现 reporting video；reporting evaluator 第二次调用应 fail closed。 |
| **P0-5** | **结果证据链未闭合。** `train_episode` 生成 audit，但常规 `forward` 只返回 losses；训练引擎不汇总 dropped birth、runtime collision、admission 等计数。测试入口只运行配置中唯一 evaluator，metrics 不持久化也不返回给 gate。                                                   | 任意正式训练/评测                                                                    | 一次“成功训练”无法证明零 dropped target、零 causal violation、零 runtime failure；结果门只能由调用者手工填字典，不能追溯 checkpoint/ledger。                                 | 新增 repository-owned `PersistentBindingResultEvaluator`/CLI：汇总训练 audit、推理 runtime trace、emission ledger、standard mAP、instance metrics、latency、资源；绑定 code/config/checkpoint/manifest/ledger SHA；直接生成 gate row，禁止人工输入正式字段。                            | 真实 dataset→detector→checkpoint→reload→emission→evaluator→gate 的端到端测试；故意篡改任一 hash、计数或 ledger 后必须拒绝。                                    |
| **P0-6** | **实例指标输入合同当前会错配。** GT 数据库行的 stream key 是视频名，emission row 的 `stream_key` 是复合字符串；指标优先采用显式 frame bounds，而原始 GT `segment` 是秒，导致 frame 与 seconds 比较。                                                                        | 将原始 annotation JSON 与 emission ledger 直接传入 `compute_online_instance_metrics` | 同一视频的 GT/预测可能因 stream key 不同而全部 unmatched；即使修正 stream key，也会用帧数对秒数算 tIoU。                                                                | 评测适配器必须生成独立 `video_id`，保留 runtime stream key 仅供审计；GT 和预测统一使用同一坐标系。建议按 dataset 的 `num_frames/duration` 规则产生显式 GT frame bounds。                                                                                                                      | 30 fps 与非 30 fps 视频；同一 segment 的秒/帧两种表示必须得到相同 tIoU、duplicate、fragment、delay；复合 runtime key 不得影响 GT matching。                          |
| **P0-7** | **冻结的 duplicate/fragmentation 定义没有实现。** 设计要求“best matching prediction”及“temporally disjoint fragments”；代码按 emission chronological greedy 锁定 GT，并把任意不同 bounds——包括完全嵌套区间——算成 fragment。                                   | 同类重叠 GT、一个预测覆盖多个 GT、嵌套或相互重叠的重复预测                                             | `E_id` 会依赖输入顺序；嵌套 duplicate 被重复计成 fragmentation，直接改变 20% 主门槛。                                                                            | 主匹配改为确定性的全局一对一匹配：最大匹配数→最大总 tIoU→分数→emit/sequence/GT-ID tie-break。剩余 eligible predictions 才归为 duplicate。Fragmentation 按与 GT 相交部分的**连通分量数减一**计算；嵌套/相互重叠预测不增加 fragment。                                                                               | 两 GT/两预测的 greedy-vs-global 反例；嵌套 `[0,10]` 与 `[1,9]` 只计 duplicate、不计 fragment；真正 `[0,4]` 与 `[6,10]` 才计一个 excess fragment。              |
| **P0-8** | **主 mAP、门槛字段和单位未冻结成同一合同。** Base config 只运行 `OnlineAPBudgeted`，其 `average_mOnlineAP` 是跨 latency budget 与 tIoU 的 0–1 均值；gate 要求字段 `average_map`，并把阈值写成 `-0.5 points`，但未验证 0–1 还是 0–100。                                  | 将当前 evaluator 输出手工映射到 gate                                                   | 直接传 0–1 mAP 时，`-0.5` 等价于允许下降 50 个百分点，科学门几乎失效；budgeted AP 也不是冻结设计的标准 average mAP。                                                         | 主门使用 `average_map_pct = 100 × OnlineMAP.average_mAP`，明确单位为 percentage points；`OnlineAPBudgeted` 只作延时敏感次指标。Gate schema 必须验证字段名、范围、单位和 tIoU 集合。                                                                                                      | 0.60/0.595 应产生 `-0.5 pp`，不能产生 `-0.005 points`；错误单位、缺 metric schema、用 budgeted AP 冒充 standard mAP 均必须拒绝。                               |
| **P0-9** | **fail-closed readiness 只是文档状态。** `formal_training_ready=False` 没有被启动器检查；canonical exhaustion 只累加 audit 后继续训练；当前所谓 `runtime_capacity_exhaustions` 又是 `GT births - runtime free slots`，不是实际 proposal/admission failure。 | 直接运行 `tools/train.py`，或某 step canonical slots 不足                             | 未就绪代码可被当正式实验运行；真实 dropped target 不会即时停止；三个不同容量概念被混成一个字段。                                                                                 | Runner：`formal_training_ready=False` 时只允许显式 smoke mode。Formal 模式下 canonical exhaustion 立即抛错。分开记录：①GT supervision exhaustion；②GT-birth/runtime-entry-free collision；③candidate arbitration suppression；④active abandonment；⑤candidate cancellation。 | unready formal launch 拒绝；canonical exhaustion 原子失败；构造“无预测 proposal 但 runtime 全占用”时不得称 actual proposal exhaustion；各计数相互独立。             |

---

# 4. P1 问题表

## P1：P0 修复后、正式三种子前必须闭合

| 编号        | 问题与证据                                                                                                                                                    | 触发条件与影响                                                                                                                      | 最小修复与测试                                                                                                                                                |                                                                                      |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| **P1-1**  | **本步释放、下一步复用不是无损操作。** `free_at_entry` 在处理旧 ACTIVE/CANDIDATE 前冻结，释放槽不会进入当前 birth pool。                                                                    | 旧动作 end 与新动作 birth 同一步；其他槽被错误 ACTIVE/CANDIDATE 占用，或只有刚释放槽的 birth logit 高。合法新动作至少延迟一 token，短动作可能完全丢失。                         | 保留主设计可以，但必须新增预测级 `deferred_birth_due_to_release` 计数和相邻动作子集报告；在正式前做 exact-census 与对抗 synthetic。若计数非零且造成漏检，不能用加槽或降 threshold 事后救援。                       |                                                                                      |
| **P1-2**  | **`max_births_per_step=2` 和四槽安全性只被文档声明，未成为 launch-consumed census artifact。**                                                                            | validation/test 中若某 decision frame 有超过 2 个 first crossings，runtime 会抑制合法 proposals；连续并发≤2 并不能自动证明 delayed-reuse occupancy≤4。 | 生成并哈希 per-split census：`max_births_per_step`、canonical occupied before/after、same-bin end+birth、short actions、delayed-reuse headroom。超过预算直接阻断，不在结果后增槽。 |                                                                                      |
| **P1-3**  | **最后 token 与缓存覆盖合同未证明。** Source frame 只要求 `< num_frames`；schedule 只在现有 source frames 上构造。                                                                | 某 GT end 晚于最后一个 source frame，模型又不得接收 EOV，便永远没有合法 end crossing。                                                               | Census 必须证明所有纳入评测 GT 均有 `source_frame >= gt_end_frame`；否则修正缓存时间格或坐标约定，不得传 EOV。加入末尾动作测试。                                                                |                                                                                      |
| **P1-4**  | **可表示 start 历史被固定为 192×8=1536 frames。** Start target 被 clamp 到 `memory_size`。                                                                            | 动作持续时间超过约 `1536/fps` 秒时，真实 start 不可表示，长动作 mAP/fragmentation 可能系统性受损。                                                         | 预先统计每 split 的 clipped-start count。若非零，在任何结果前统一扩大两臂 memory，或明确将其列为模型能力上限并报告长动作分层。                                                                       |                                                                                      |
| **P1-5**  | **视频 reset 接口不够 fail-closed。** `reset_stream` 是允许字段；`is_video_start=True` 会直接清空全部状态，没有验证它确实是 token 0 或新视频。                                               | 错误 caller 在同一视频中间标记 start/reset，可制造 silent state loss、重复提交或虚假低容量计数。                                                          | Formal forward 禁止 mid-video reset；`is_video_start` 必须与 packet_start_token=0 和新 video key 一致。加入恶意 reset、重复 start、视频交错测试。                                |                                                                                      |
| **P1-6**  | **上游缓存因果性目前只是 manifest assertion。** Dataset 验证 `feature_policy="packet_recent_frame"`、时间戳、hash 和 encoder ID，但没有验证 encoder 实际是否使用 centered/future window。 | 特征生产脚本与 manifest 声明不一致时，下游严格因果测试仍会全部通过。                                                                                      | 注册 feature-producer commit、配置、模型 hash、每 token 原始 frame index；对样本视频做 future-frame perturbation，确认 prefix cached feature 不变。                             |                                                                                      |
| **P1-7**  | **配对优化执行没有正式成功-update 计数。** Train engine 遇到非有限 loss/gradient会跳过 optimizer step 后继续；没有终端汇总确保两臂 successful updates 相等。                                     | 某臂 AMP overflow/nonfinite 较多                                                                                                 | 两臂实际更新数、scheduler steps 不同，形成训练预算混杂。Formal 运行要求零 skipped update，或严格相同成功更新数；保存 terminal audit。                                                          |                                                                                      |
| **P1-8**  | **checkpoint 规则未闭合。** `val_loss_interval=-1` 不会生成 `best.pth`，而测试入口未显式指定 checkpoint 时默认读取 `best.pth`。                                                     | 按默认 test 命令运行                                                                                                                | smoke 直接找不到 checkpoint，或工程人员临时挑选 epoch。                                                                                                                | 正式规则固定 `epoch_11.pth`；所有 test 命令必须显式传 checkpoint 与 SHA。禁止 best-on-reporting。         |
| **P1-9**  | **ledger 扩展名与内容格式冲突，且缺显式 sequence。** 配置名为 `.jsonl`，测试入口实际用一次 `json.dump` 写完整 JSON；event ID 内嵌 sequence，但 row 没有 `sequence_id`。                           | 下游按 JSONL 逐行读取；同一 emit frame 有多个提交                                                                                           | 序列化解析错误；同刻事件在 sort 后的顺序可能改变实例匹配 tie-break。                                                                                                             | 改名 `.json` 或写真实 JSONL；每条 row 写 `video_id`、`runtime_stream_key`、`sequence_id`；序列严格递增。 |
| **P1-10** | **共享阈值选择算法尚未冻结。** 当前计划只写“freeze validation-selected thresholds”。                                                                                         | 查看 calibration 结果后人工选择                                                                                                       | 可直接针对 E_id 或某一臂调阈值，破坏单变量比较。                                                                                                                            | 采用预注册的对称算法，详见下文；阈值选择不得使用 reporting split，也不得直接优化 E_id。                               |

---

# 5. P2 问题表

| 编号       | 风险                                                                               | 裁决与处理                                                                                                      |
| -------- | -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **P2-1** | chunk 边界采用 detached state，且每个 chunk 后更新参数；下一 chunk 使用由旧参数产生的 hidden/query state。 | 这是共享的 truncated-BPTT 近似，不是当前单变量违规。主结果后做 `chunk_size=32/64/128` 单种子敏感性；若 FIXED 优势随 chunk size 反转，论文主张应降级。   |
| **P2-2** | standard mAP evaluator会按 ActionFormer 规则删除完全重复 GT annotation。                    | 标准 mAP可保持该规则以对齐社区；实例指标不得删除实例 ID。报告原始 GT 数与去重 GT 数。                                                         |
| **P2-3** | `E_id_REMATCH` 很小时，相对降低率不稳定。                                                     | 冻结主门不变，同时报告 `ΔE_id`、duplicate/fragmentation 分量和 paired bootstrap CI；不得用辅助指标替换主门。                           |
| **P2-4** | Feature route 的 profile contract 是配置声明，尚无正式 profiler artifact。                   | 在 smoke 中记录 wall time、successful updates/s、token/s、peak allocated/reserved memory、emission p50/p95；超过预算停止。 |
| **P2-5** | 未来可能希望做 sub-stride endpoint regression。                                          | 当前主实验禁止临时增加。先用可审计的 binary current-frame endpoint 完成证伪；sub-stride regression 只能作为通过主门后的独立消融。                |

---

# 6. 因果与状态不变量表

| 不变量                                                      | 裁决                   | 证据/限制                                                                                                   |
| -------------------------------------------------------- | -------------------- | ------------------------------------------------------------------------------------------------------- |
| 每个 token 的 logits 只依赖当前及过去缓存特征                           | **成立**               | Memory 只追加当前 feature；后续 feature perturbation 不改变先前 logits。                                              |
| 缓存特征本身严格因果                                               | **证据不足**             | Manifest 有声明与 hash，但本提交未证明 producer 实际感受野。                                                              |
| 推理模型接收 annotation/GT segment                             | **成立：不会**            | Forbidden metadata 递归检查；推理拒绝 prefix schedule。                                                           |
| 推理接收未来 endpoint、总时长、EOV                                  | **成立：不会**            | `duration/end_frame/is_video_end/total_frames/eof` 被拒绝；严格 dataset control 只给 `video_id/is_video_start`。 |
| 同一视频 source frames 严格单调、chunk 不重叠                        | **成立**               | Dataset 与 detector 均校验；非单调直接报错。                                                                         |
| 多 GPU/多 lane 下仍保持状态正确                                    | **不成立，但配置已规避**       | Streaming sampler明确只允许 single rank；正式运行必须固定一张可见 GPU。                                                    |
| 视频切换不会继承上一视频 state                                       | **成立于官方 dataset 路径** | 首 chunk 清空所有 runtime/supervision state；但外部错误 start/reset 尚未 fail-closed。                                |
| runtime state 不含 GT identity                             | **成立**               | Runtime schema只有 query/memory/status/start/score/label/committed counters。                              |
| 训练 chunk 间 state detached                                | **成立**               | 配置明确启用；当前设计是 truncated state training。                                                                  |
| 任意 runtime occupancy 都不能删除真实 first-crossing birth target | **成立**               | canonical birth candidates只看 supervision map；聚焦测试以全部 runtime slots 占用仍得到 GT birth assignment。           |
| canonical exhaustion 只源于真实监督并发/配置                        | **逻辑成立；执行不成立**       | 计数来源正确，但没有 fail immediately，且常规 runner 不持久化 audit。                                                      |
| REMATCH 不改变 runtime lifecycle                            | **成立**               | 同一 head/decode 路径；模式只进入 supervision binding。                                                            |
| REMATCH 只改变 birth 后 loss binding                         | **不成立**              | 当前 birth 当步已 rematch，且可使 birth target 与 start/class/end target 跨槽。                                      |
| CANDIDATE 一步确认与训练 target 完全对齐                            | **不成立**              | newborn 的同一步 end target 会被 runtime 忽略。                                                                  |
| 每个 runtime hypothesis 最多提交一次                             | **成立**               | commit 后 append record 并 reset slot；同一 event ID 不可重复。                                                   |
| 同一物理 GT 实例最多提交一次                                         | **不保证，且不应由控制器强行保证**  | commit 后模型可再次 birth；应由 duplicate rate 衡量，不得离线去重。                                                        |
| 已提交区间不可修改                                                | **成立**               | Row 标记 `final/immutable`，forward只返回新 emission；没有后续更新接口。                                                 |
| 不运行 offline NMS/full-video correction                    | **成立**               | Streaming-safe 模式关闭 video-level sliding-window NMS。                                                     |
| 四槽、每步最多两 birth 对所有 split 安全                              | **证据不足**             | 冻结文档给出声明，但缺启动器消费的精确 census artifact。                                                                    |
| 标准 mAP 坐标合同正确                                            | **基本成立**             | Emission 同时写 frame 与 seconds，标准 evaluator读取 seconds。                                                    |
| 实例指标坐标与 stream 合同正确                                      | **不成立**              | 当前原始 GT 与 emission 直接组合时发生 stream/单位错配。                                                                 |
| evaluator 会产出全部 gate 字段                                  | **不成立**              | 现行配置只产出 `OnlineAPBudgeted`。                                                                             |
| `formal_training_ready=True`                             | **不成立，且应继续为 False**  | 当前配置明确为 False，P0 尚未闭合。                                                                                  |

---

# 7. 最小反例

## 7.1 同一步 end + birth

设 decision frames 为 `7, 15, 23`。

* 旧实例 A 在 slot 0 为 ACTIVE，真实结束在 frame 12。
* 新实例 B 在 frame 12 开始。
* 在 decision 15，A 的 end 和 B 的 birth 同时成为可观察事实。
* slot 0 输出 `end=0.9, birth=0.9`。

当前 decoder 在 step 入口冻结 `free_at_entry[0]=False`，随后提交 A 并释放 slot 0，但 birth ranking 仍只看冻结的 entry-free pool，因此 B 不会在该槽出生。

**结论：**“本步释放、下一步复用”在一般意义上会漏合法动作。四槽只提供经验 headroom，不构成逻辑证明；若其他槽被错误 ACTIVE/CANDIDATE 占用，或 B 只在刚释放 query 上有高 birth logit，反例在四槽下仍成立。

## 7.2 极短动作

* `previous_frame=7`
* 动作 B 为 `[8,12]`
* 当前 decision 为 15

Schedule 合法地产生 `births=(B)`、`ends=(B)`，没有 active prefix。监督器同一步提供 birth/end/class/start target。

若一个 FREE runtime slot 在 frame 15 同时给出高 birth 与高 end，当前 decoder 只把它设为 CANDIDATE，不检查该 newborn 的 end。下一步若不存在，B 永不提交；若 frame 23 的 end/alive 都低，B 被取消。

这与冻结设计“candidate with an immediate end can commit as a short action”直接矛盾。

## 7.3 相邻同类动作

A=`[0,12]`，B=`[12,20]`，两者同类。

在 frame 15：

* A 需要 commit；
* B 需要 birth；
* 若 B 未出生，模型更可能把二者表现为一个长 ACTIVE，或者丢失 B；
* standard mAP、duplicate、fragmentation 和 repeated-same-class recall 会给出不同解释。

因此必须单独报告：

* 相邻同类 GT 数；
* 两个实例均被一对一匹配的 recall；
* merge-one-into-two-GT 错误数；
* end+birth 同步碰撞数。

## 7.4 错误 CANDIDATE 取消与真实 birth

slot 0 在 frame 7 因 false birth 进入 CANDIDATE。frame 15：

* candidate 的 alive/end 都低，因此被取消；
* 同时真实动作 B 在 frame 15 birth；
* slot 0 在 step entry 不是 FREE，所以 B 不能同一步重新使用它。

这造成一个额外 decision blackout。当前只计 candidate cancellation，不计“取消当步被延迟的 birth”。

## 7.5 同类重叠 GT 的 greedy matching 反例

* GT1=`[0,10]`
* GT2=`[5,15]`
* 先发 prediction P1=`[3,12]`，与两 GT 的 tIoU 均约 0.583
* 后发 P2=`[0,6]`，只与 GT1 达标

当前 chronological greedy 在 tie 时可先把 P1 锁给 GT1，随后 P2 成为 duplicate，GT2 未匹配。全局最优应为：

* P1→GT2
* P2→GT1

因此当前 duplicate、recall 和 overlap-subset recall 会依赖 emission 输入顺序，而不是预测集合本身。

## 7.6 嵌套预测不是 fragmentation

GT=`[0,10]`，预测：

* P1=`[0,10]`
* P2=`[1,9]`

P2 是 duplicate，但两个区间形成一个连续覆盖组件，不是两个 temporally disjoint fragments。当前代码因 bounds 不同而同时计一个 excess fragment，违反冻结定义。

## 7.7 最后一 token 的动作

若最后 source frame 为 15，动作 start/end 均在 `(7,15]`：

* 当前 step 能 birth，但不能立即 end；
* 严格协议又不提供 EOV；
* 下一视频的 `is_video_start` 清空 state。

所以该动作必定没有 final emission。P0-3 修复和 endpoint-coverage census 都必须覆盖此情况。

---

# 8. FIXED/REMATCH 单变量公平性结论

## 8.1 配置层面

**成立。** 已有测试展开两个配置并移除：

* `model.trajectory_binding_mode`
* `work_dir`

之后要求完整配置对象相等。

两臂共享：

* 数据与时间格；
* query head；
* runtime lifecycle；
* birth/alive/end thresholds；
* slot 数和每步 birth 上限；
* optimizer、scheduler、epoch、AMP；
* inference 与 evaluator。

## 8.2 算法语义层面

**当前不成立。**

主要原因不是 runtime controller 不同，而是 REMATCH 在 birth 当步已经让：

* birth BCE target 留在 canonical slot；
* class/start/end target 移到 rematched slot。

因此当前干预不纯粹是“持久绑定 vs 后续逐前缀 rematch”，还混入了“同一 birth step 的跨 head target 不一致”。

## 8.3 修复后的可支持论文命题

修复 P0-2 后，该对照可以支持一个**窄而有效**的因果命题：

> 在固定缓存因果特征、相同 first-crossing births、相同 runtime lifecycle、相同阈值与容量下，birth 后保持 target-to-slot 绑定，是否相对逐前缀 rematching 降低 duplicate 与 fragmentation。

它**不能单独支持**：

* persistent binding 普遍优于所有 On-TAD 方法；
* raw-RGB 端到端训练有效；
* 该方法达到新的 On-TAD SOTA；
* 优势与 candidate confirmation、slot budget、特征编码器或 truncated state 无关。

仍需控制或报告的因素是：

1. candidate one-step confirmation；
2. 四槽容量和 delayed reuse；
3. endpoint 量化；
4. chunk size/state detach；
5. shared threshold calibration；
6. 同类相邻/重叠分布；
7. 缓存特征生成方式；
8. metric matching policy。

---

# 9. 指标与结果门槛冻结建议

冻结主指标 `E_id` 不变，但在任何结果产生前，把如下定义写入 evaluator schema。

## 9.1 主标准指标

* `average_map_pct = 100 × mean(mAP@0.3,0.4,0.5,0.6,0.7)`
* 来源：`OnlineMAP`，不是 `OnlineAPBudgeted`
* 同时报告各 tIoU mAP
* `OnlineAPBudgeted` 作为延时敏感次指标单独报告，不送入 `-0.5 pp` 主门

## 9.2 completion-valid precision/recall/F1

在 tIoU 0.3 下：

* 预测只有在 `emit_time >= matched_GT_end` 时才有资格成为 TP；
* 提前 final emission 作为 unmatched/early false emission；
* 每个 GT 最多一个 primary prediction；
* 每个 prediction 最多匹配一个 GT。

冻结技术门中的 `Recall@0.3` 使用该 completion-valid recall，而不是 proposal top-k recall。

## 9.3 实例主匹配

在 **tIoU 0.5** 冻结 `E_id` 主匹配；同时报告 0.3 和 0.7 sensitivity，但不改变主门。

匹配顺序固定为：

1. 最大化匹配数；
2. 最大化总 tIoU；
3. 最大化总 score；
4. 更早 emit；
5. 更小 `sequence_id`；
6. 更小 GT stable ID。

该匹配仅用于离线测量，不修改任何 emission，因此不违反 online protocol。

## 9.4 Duplicate

对每个 GT：

* 一对一主匹配之外；
* 仍满足同类、同视频、tIoU≥0.5 的额外 predictions；
* 都算 duplicate。

分母保持全部 GT 数。

## 9.5 Fragmentation

对分配给同一 GT 的预测区间：

1. 裁剪到 GT 范围；
2. 合并相交或相接的半开区间；
3. `fragment_excess=max(0, connected_components-1)`。

这样嵌套和重叠 duplicates 不会重复计 fragmentation。

保留当前 `distinct_bound_excess` 作为辅助指标，不能代替冻结 fragmentation。

## 9.6 必须附加但不能替换 `E_id` 的指标

* `ΔE_id = E_id_FIXED - E_id_REMATCH`
* duplicate 与 fragmentation 分量
* false emission rate
* premature-final rate
* candidate cancellation
* active abandonment
* arbitration suppression
* release-deferred birth
* repeated-same-class recall
* overlapping-same-class recall
* adjacent-same-class pair recall
* unmatched GT/预测数
* GT-end commit delay：`emit_time - GT_end`
* predicted-end latency：`emit_time - predicted_end`
* per-video paired bootstrap CI

## 9.7 Gate schema

正式 gate row 至少必须含：

```text
schema_version
code_sha
config_sha256
checkpoint_sha256
annotation_sha256
class_map_sha256
feature_manifest_sha256
split_manifest_sha256
emission_ledger_sha256
runtime_trace_sha256
arm
seed
average_map_pct
duplicate_rate_tiou_0p5
fragmentation_rate_tiou_0p5
prediction_gt_ratio
completion_valid_recall_tiou_0p3
protocol_violations
dropped_gt_birth_targets
runtime_gt_birth_collision_targets
unexplained_runtime_capacity_failures
candidate_arbitration_suppressions
committed_predictions
successful_optimizer_updates
skipped_optimizer_updates
```

Gate 应拒绝：

* 人工构造、无 artifact provenance 的 row；
* mAP 单位为 0–1；
* 缺少 metric threshold/schema；
* reporting split 在阈值冻结前被访问；
* 任一 skipped update；
* 任一 canonical dropped target。

---

# 10. `formal_training_ready` 裁决

## 必须继续保持 `False`

变成 `True` 前，最少要同时满足：

1. P0-1 至 P0-9 全部修复；
2. newborn binding、short immediate-end、metric global matching、endpoint binary 的聚焦测试通过；
3. 生成并封存 exact split census；
4. feature producer provenance 与 prefix perturbation 通过；
5. 训练入口无法读取 reporting split；
6. repository-owned result evaluator 能从真实 checkpoint/ledger 直接生成 gate row；
7. Slurm 整链路 smoke 通过；
8. 两臂 smoke 的初始参数 hash 相同；
9. 零 nonfinite、零 skipped update；
10. checkpoint reload 后同一 causal stream 输出一致。

应在一个新的不可变修复提交中将该值改为 `True`。当前 SHA 不应被追认成 formal implementation commit。

---

# 11. Slurm 整链路 smoke 命令

以下是**P0 修复提交后**应冻结的命令形态。当前 SHA 缺少 `validate_persistent_binding_readiness.py`、完整 result evaluator 和 split-safe train 路径，因此不能原样满足最后验收。

`<ACCOUNT>`、`<GPU_PARTITION>` 和 `<N16R4_ENV>` 是仅有的站点字段。

```bash
sbatch <<'SBATCH'
#!/bin/bash
#SBATCH --job-name=ontad-pb-smoke
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/ontad-pb-smoke-%j.out
#SBATCH --error=logs/ontad-pb-smoke-%j.err

set -euo pipefail

BASE_SHA=27a59dec445f6b4ed9651abab1168c358a8db7e3
REPAIR_SHA=<IMMUTABLE_P0_REPAIR_SHA>
REPO=/path/to/OpenTAD_OnlineTADClean_20260702
ENV_ROOT=<N16R4_ENV>

# 只能取自 fit/calibration split；禁止使用 reporting_locked_211。
SMOKE_FIT=/data/run01/sczc063/yuzibo/thumos14/manifests/persistent_binding/smoke_fit_2.txt
SMOKE_CAL=/data/run01/sczc063/yuzibo/thumos14/manifests/persistent_binding/smoke_calibration_2.txt
OUT_ROOT="${REPO}/exps/thumos/persistent_binding_smoke/${SLURM_JOB_ID}"

source "${ENV_ROOT}/bin/activate"
cd "${REPO}"

test "$(git rev-parse HEAD)" = "${REPAIR_SHA}"
git merge-base --is-ancestor "${BASE_SHA}" "${REPAIR_SHA}"
test -z "$(git status --porcelain)"
test "$(python -c 'import torch; print(torch.cuda.device_count())')" = "1"

export PYTHONHASHSEED=704
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# 静态合同、数据 census、配置差异、reporting 禁访检查。
srun python tools/validate_persistent_binding_readiness.py \
  --base-sha "${BASE_SHA}" \
  --review-sha "${REPAIR_SHA}" \
  --fit-manifest "${SMOKE_FIT}" \
  --calibration-manifest "${SMOKE_CAL}" \
  --forbid-reporting-access \
  --mode smoke \
  --output "${OUT_ROOT}/readiness.json"

# 必须覆盖短动作、同一步 end+birth、相邻/同类重叠和序列化。
srun python -m pytest -q \
  tests/test_persistent_binding_e2e.py \
  tests/test_prefix_trajectory_supervision.py \
  tests/test_persistent_event_set_head.py \
  tests/test_online_instance_metrics.py

for ARM in fixed rematch; do
  TRAIN_CFG="configs/causaltad/thumos_persistent_binding_${ARM}_smoke.py"
  EVAL_CFG="configs/causaltad/thumos_persistent_binding_${ARM}_smoke_eval.py"
  ARM_OUT="${OUT_ROOT}/${ARM}"

  srun torchrun --standalone --nproc_per_node=1 tools/train.py \
    "${TRAIN_CFG}" \
    --seed 704 \
    --id 0 \
    --cfg-options \
      smoke_mode=True \
      formal_training_ready=False \
      dataset.train.allow_list="${SMOKE_FIT}" \
      dataset.val.allow_list="${SMOKE_CAL}" \
      workflow.end_epoch=1 \
      workflow.val_eval_interval=-1 \
      work_dir="${ARM_OUT}"

  CKPT="$(find "${ARM_OUT}" -type f -path '*/checkpoint/epoch_0.pth' \
          -print -quit)"
  test -n "${CKPT}"
  sha256sum "${CKPT}" > "${ARM_OUT}/checkpoint.sha256"

  # 必须显式给 checkpoint；评测只用 calibration smoke list。
  srun torchrun --standalone --nproc_per_node=1 tools/test.py \
    "${EVAL_CFG}" \
    --checkpoint "${CKPT}" \
    --seed 704 \
    --id 0 \
    --cfg-options \
      dataset.test.allow_list="${SMOKE_CAL}" \
      work_dir="${ARM_OUT}/eval"

  srun python tools/evaluate_persistent_binding_run.py \
    --arm "${ARM}" \
    --seed 704 \
    --config "${EVAL_CFG}" \
    --checkpoint "${CKPT}" \
    --split calibration_smoke \
    --emission-ledger "${ARM_OUT}/eval/persistent_binding_emissions.json" \
    --runtime-trace "${ARM_OUT}/eval/persistent_binding_runtime_trace.json" \
    --output "${ARM_OUT}/eval/result_contract.json"
done

srun python tools/compare_persistent_binding_runs.py \
  --fixed "${OUT_ROOT}/fixed/eval/result_contract.json" \
  --rematch "${OUT_ROOT}/rematch/eval/result_contract.json" \
  --require-same-init-hash \
  --require-config-single-axis \
  --smoke-only \
  --output "${OUT_ROOT}/smoke_terminal.json"

test "$(python - "${OUT_ROOT}/smoke_terminal.json" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    value = json.load(f)
print("PASS" if value["smoke_pass"] else "FAIL")
PY
)" = "PASS"
SBATCH
```

## Smoke 验收项

1. exact repair SHA、clean tree、单可见 GPU、single rank；
2. train 进程没有打开 reporting manifest；
3. FIXED/REMATCH resolved config 只差 mode/work_dir；
4. 两臂初始化参数 SHA 完全相同；
5. 至少两个连续真实 chunks，状态跨 chunk 延续；
6. AMP forward/backward、gradient clipping、optimizer、scheduler 均执行；
7. 所有预期 trainable 参数梯度有限；零 skipped update；
8. checkpoint 写出、显式 reload 后输出一致；
9. 新视频 start 清空旧 state，mid-video reset 被拒绝；
10. 短动作、birth+end、相邻动作产生正确且唯一 final emission；
11. ledger schema、sequence、frame/seconds/fps 一致；
12. 零 future/source/monotonic violation；
13. 零 dropped GT target；
14. result contract 直接包含标准 mAP、实例指标、延时和全部计数；
15. smoke 只验证调用链，不对 2-video 指标应用正式 20% 科学门。

---

# 12. 完整 feature 阶段论文实验计划

配置中的现有成本上限是单种子 `2 GPUh`、三种子 `10 GPUh`；以下把它们作为硬停止上限，而不是结果承诺。

| ID                      | 要证明的命题                                               | 自变量与控制量                                                              | Split / seeds / 输入                                 | 输出指标                                                                                                                    | 通过或停止条件                                                                                                             | 前置依赖                | 预估 GPU                   |
| ----------------------- | ---------------------------------------------------- | -------------------------------------------------------------------- | -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------- | ------------------------ |
| **F0 合同修复**             | 代码实现与冻结科学问题一致                                        | P0 修复；不训练                                                            | 全部源码/配置/测试；固定缓存 feature contract                   | readiness report、config diff、parameter-gradient coverage                                                                | P0 全过；任何一项失败都不得进入 Slurm                                                                                             | 无                   | 0                        |
| **F1 数据与时间格 census**    | 四槽、max birth=2、memory=192、无 EOV 在真实 split 安全         | 无模型自变量                                                               | fit160/cal40/report211；CPU；精确 manifest             | max concurrent、max births/step、same-bin end+birth、short count、unobserved endpoints、start-clamp count、same-class subsets | `max_births<=2`、canonical occupancy≤4、unobserved endpoint=0；否则先改设计，禁止训练                                             | F0                  | 0                        |
| **F2 Synthetic E2E**    | dataset→detector→serialization→evaluator 在最小反例上正确    | FIXED/REMATCH；相同人工 logits/features                                   | 短动作、相邻、重复/重叠同类、false candidate、wrong ACTIVE、视频切换   | 精确 emission、计数、duplicate/fragment、delay                                                                                 | 与手工 oracle 完全一致                                                                                                     | F0                  | ≤0.1 GPUh                |
| **F3 Slurm smoke**      | OpenTAD runner、AMP、state、checkpoint 和 evaluator 实际兼容 | 两臂仅 mode 不同                                                          | fit/cal 各 2 个真实视频；seed 704；缓存特征                    | 上节全部 smoke artifact                                                                                                     | 全部验收通过；任一 nonfinite/skipped/drop/report access 停止                                                                   | F0–F2               | ≤0.5 GPUh 总计             |
| **F4 单种子非退化筛选**         | 12 epoch 能收敛且 controller 非静默                         | FIXED/REMATCH；相同 seed704、初始化、顺序、epoch12、默认阈值0.5                      | fit160 训练；cal40 评测；缓存特征                            | loss/grad、提交数、ratio、Recall@0.3、mAP、E_id、runtime counters                                                                | 两臂均零技术错误；默认阈值下至少非零提交、ratio∈[0.1,10]、Recall@0.3≥0.10。失败即停止，不调槽、不降 birth 制造通过                                         | F3                  | **≤2 GPUh 总计**           |
| **F5 共享阈值冻结**           | 一组 threshold 能同时使两臂非退化，不针对 E_id 优化                   | 只扫描共享 `(birth,alive,end)`；其余完全固定                                     | seed704 checkpoints；cal40；候选值 `{0.30,0.35,…,0.70}` | 两臂 technical margins、standard mAP                                                                                       | 先要求两臂均满足正式 technical calibration；再最大化两臂 standard mAP 的算术平均；tie 取距 `(0.5,0.5,0.5)` 最近，再 lexicographic。无可行 triple 即停止 | F4                  | ≤1 GPUh                  |
| **F6 正式配对训练**           | 绑定策略在重复训练下产生稳定差异                                     | FIXED/REMATCH；共享冻结阈值、final epoch12、完全相同 run manifest                 | fit160；seeds 705/706/707；缓存特征                      | successful updates、train audit、checkpoint hash、cal metrics                                                              | 每臂每 seed 零 skipped、零 GT exhaustion、预算内完成；任何 seed 失败则整组 technical fail，不补种子                                          | F5                  | **≤10 GPUh 总计**          |
| **F7 Sealed reporting** | 在从未访问的 211-video split 上检验双门槛                        | 只改变 binding mode                                                     | report211；705/706/707；每 checkpoint 一次推理            | standard mAP、budgeted AP、precision/recall/F1、E_id、子集 recall、delay、资源                                                    | 每 arm/seed technical gate 全过；scientific gate严格按冻结值。失败后不得改阈值、槽数、epoch或 metric                                        | F6                  | ≤1 GPUh                  |
| **F8 统计与不确定性**          | 改善不是少量视频或单 seed 驱动                                   | 无额外训练                                                                | report211、三配对 seeds                                | 全 seed 明细；mean±SD；10k paired video bootstrap；`ΔE_id`、分量 CI                                                              | 不改变原 gate；若 CI 宽或优势集中于单类，论文主张降级                                                                                     | F7                  | 0                        |
| **F9 必要消融**             | 排除 candidate/容量预算对 binding 结论的替代解释                   | 预注册：confirmation 0 vs1；slots 2/4/6；主 binding 2 arms；所有变体共享其各自预注册校准规则 | 先 seed705；仅主交互明显时扩 706/707；缓存特征                    | mAP、E_id、capacity/cancel/abandon、interaction effect                                                                     | 不允许用 6 槽结果“救活”四槽主实验；四槽仍是唯一主结果                                                                                       | F7 technical pass 后 | 分阶段 ≤6–12 GPUh           |
| **F10 子集与失败分析**         | 直接验证目标机制作用场景                                         | 无训练改变                                                                | report211：短、相邻同类、重复同类、重叠同类、长动作、边界末端                | pair recall、merge、duplicate、fragment、delay、状态轨迹图                                                                        | 所有 subset 由 GT 预先定义；禁止看预测后划分                                                                                        | F7                  | 0                        |
| **F11 公平在线基线**          | 证明收益不只相对 REMATCH 成立                                  | 相同缓存特征、split、无未来、immutable emission、同 evaluator                      | 705/706/707 或先705筛选                                | standard/strict mAP、latency、resource、instance metrics                                                                   | 不能满足严格协议的公开结果只能放 contextual 表，不能参与直接排名                                                                              | F7                  | 每个重实现 baseline 约2–6 GPUh |
| **F12 论文冻结**            | 形成不可修改的 feature-level 结论                             | 无新调参                                                                 | 全部 sealed artifacts                                | registration、主表、子集表、失败图、resource table                                                                                  | 双门通过才进入 raw RGB；科学门失败则 STOP/DEMOTE                                                                                  | F7–F11              | 0                        |

## 统计单位

* 训练重复单位：seed；
* 机制误差单位：视频中的 GT instance；
* 置信区间：在每 seed 内按视频配对 bootstrap，再对 seed 平均；
* 三个 seed 的原始值必须全部列出，不能只给均值；
* 主 gate 不因 CI、p-value 或消融结果而事后修改。

---

# 13. 在线基线的公平边界

直接 On-TAL 文献基线至少应覆盖：

* **CAG-QIL**：明确以无未来帧、不能修改过去 proposal 的 On-TAL 为任务；
* **2PESNet hard-online 版本**：只能采用不访问未来 temporal tolerance 的 hard-online 条件；
* **SimOn**：直接输出 action instances，并明确禁止未来帧及修改过去预测；
* **MATR**：使用当前 segment 与过去 memory 预测 end/start，是近期直接 instance-level On-TAL 对照。([CVF Open Access][1])

SimOn 也明确区分：

* OAD：逐帧类别；
* ODAS：动作开始；
* On-TAL：完整 `{start,end,class}` 实例。

因此 OadTR、TRN、MAT、CMeRT 等纯 OAD 方法不能把其逐帧 mAP 与当前 interval mAP 直接放在同一主表；只有在加上**同一冻结、严格因果、不可修改的在线 grouping/emission layer**后，才能作为重实现基线。([arXiv][2])

公平比较要求：

1. 同一 160/40/211 split；
2. 同一缓存输入，或在单独表中明确 backbone 不同；
3. 同一无未来/EOV/NMS 协议；
4. 同一 standard mAP 与 completion-valid latency evaluator；
5. 相同 threshold calibration 数据边界；
6. published numbers只作 contextual reference，不能用于 FIXED 的直接显著性结论。

---

# 14. 条件 raw-RGB 阶段

Feature 双门未通过时，下面全部禁止执行。

| ID                           | 命题                                      | 输入与控制                                                       | Seeds / 输出                                                         | 停止条件                                                       | 初始规划成本          |
| ---------------------------- | --------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------- | --------------- |
| **R0 Causal encoder parity** | 在线 raw-frame producer 确实复现当前缓存语义且无未来感受野 | 冻结 encoder；逐 token causal extraction；detector/lifecycle完全不变 | 样本视频与 seed704；feature cosine/max-error、prefix perturbation、latency | 任一未来依赖；或缓存/在线 feature 不满足预注册容差                             | ≤2 GPUh         |
| **R1 Frozen encoder**        | raw input 工程链本身不破坏 feature-level结论      | frozen raw encoder + 相同 detector；FIXED/REMATCH              | 705/706/707；单独 raw-frozen 表                                        | technical gate失败或 FIXED作用反转且无法由数值误差解释，停止 PEFT              | 规划区间15–30 GPUh  |
| **R2 PEFT**                  | 有限视觉适配能提高 mAP且不破坏 identity effect       | 相同 causal encoder，只有预注册 adapter/LoRA 可训练；lifecycle不变        | 三种子；mAP、E_id、延时、显存                                                 | feature双门必须继续成立；成本或因果门失败即停                                 | 规划区间45–90 GPUh  |
| **R3 Joint training**        | 完整视觉联合训练是否值得                            | 仅在 R2 明确通过后；不改 binding/lifecycle/metric                     | 三种子；完整资源和稳定性报告                                                     | 100-step profile外推超过180 GPUh或出现 future/state violation即不启动 | 规划区间90–180 GPUh |

这些 raw-RGB 成本只是资源规划区间；正式 cap 必须由 100-step Slurm profile 的 `seconds/update × updates × seeds` 冻结。Feature 与 raw-RGB 结果必须分表、分标题、分结论，不能混合取平均。

---

# 15. 最终建议

## 唯一下一步任务

创建一个以 `27a59dec445f6b4ed9651abab1168c358a8db7e3` 为祖先的 **experiment-readiness repair commit**，且只完成以下闭环：

1. binary endpoint 改为 current-frame endpoint；
2. newborn 在 birth step 固定 canonical binding；
3. newly admitted candidate 支持 same-step immediate end；
4. 训练入口完全隔离 reporting split；
5. 加入有 provenance 的 composite evaluator/result gate；
6. 修正实例坐标、stream key、global matching 与 disjoint fragmentation；
7. formal readiness、canonical exhaustion 和所有 runtime 计数 fail closed；
8. 运行上述 Slurm smoke。

## 是否值得现在再发起高成本深入讨论

**现在不值得。**

当前阻断项是明确、局部、可编码和可测试的工程—科学合同错误，不需要继续发散模型路线。高成本讨论应推迟到：

* P0 repair commit 固定；
* Slurm smoke 通过；
* 单种子非退化筛选取得真实结果；

之后只需再进行一次正式 registration/结果解释审查，而不是重新设计方法。

[1]: https://openaccess.thecvf.com/content/ICCV2021/html/Kang_CAG-QIL_Context-Aware_Actionness_Grouping_via_Q_Imitation_Learning_for_Online_ICCV_2021_paper.html "https://openaccess.thecvf.com/content/ICCV2021/html/Kang_CAG-QIL_Context-Aware_Actionness_Grouping_via_Q_Imitation_Learning_for_Online_ICCV_2021_paper.html"
[2]: https://arxiv.org/abs/2211.04905 "https://arxiv.org/abs/2211.04905"
