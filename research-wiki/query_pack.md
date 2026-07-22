---
type: query_pack
updated: 2026-07-22
status: active
scope: Current compressed context for the strictly causal On-TAD task.
---

# Query Pack: Strictly Causal On-TAD

## 任务目标

研究对象是标准、全监督、严格因果的在线时序动作定位（On-TAD）。模型在时刻
`t` 只能读取当前及过去的视频证据，端到端维护动作实例的出生、持续和结束，并在
动作结束后以低延时一次性写出不可修改的 `{start, end, class, score}` 最终区间。

本库只讨论 On-TAD 方法、数据、训练和评测。不得引入未来帧、未来端点标签、全视频
回看、offline NMS 或事后改写历史区间。推理时不得读取 GT identity、annotation、
terminal 或其他未来字段。

当前是**特征级**实验：输入为冻结的 causal SigLIP2 stride-8、768 维缓存特征，
不是 raw-RGB 联合训练。只有特征级 FIXED/REMATCH 正式门、重复性和论文主结果规划
依次通过后，才讨论 raw-RGB 的 frozen/PEFT/joint 阶梯。

## 冻结科学问题

主假设比较 first-crossing **FIXED** 与 per-prefix **REMATCH** 两种监督绑定：

- 两臂使用同一 seed 705、fit/calibration 划分、输入特征、模型、损失、优化器、
  训练轮次、槽位、阈值、解码器、推理和指标；
- 唯一主比较轴是实例出生后，监督 target 是否固定绑定到原 slot；
- FIXED 检验持久身份是否能减少实例漂移和碎片化；REMATCH 是允许每个 prefix
  重新匹配的对照；
- runtime 状态只依赖预测和过去记忆，最终区间只提交一次。

正式 birth/alive/end 阈值固定为 `0.5`。校准集只用于预注册的第 3/6/9/12 轮
checkpoint 选择，不搜索或降低阈值；reporting split 在正式授权前保持未访问。

## 当前 H2 模型

当前模型使用 6 个物理 slot，但语义是“最多 4 个预测实例占用 + 2 个硬 birth
reserve”，不是允许 6 个实例长期占满。冻结的 411 视频 census 为：320,205 个
token、6,328 个动作实例、同一步最多 2 个 birth、最大可见并发 4，birth/end 全覆盖，
GT 容量缺口为零。因此每一步必须保留 2 个 entry-free 槽接收新 birth；同一步释放的
slot 到下一决策才可复用。

H2 还包括：

- episode-balanced monotone birth/alive/end calibration；
- causal-delta transition-end；
- 只回看已观察 token 的 past-only start pointer；
- 失败时先保全 audit、配置、checkpoint 与 SHA-256 到 quarantine，再 fail-closed；
- 全程审计监督耗尽、skip、容量碰撞、未来特征/端点、负延时、非正长度区间、
  immutable 和 sequence 顺序。

## 已完成的关键证据

1. 一轮硬 reserve 成对验证 `1179361` 在 exact
   `5edc46c34c0db56e79409fac69276450ad87e949` 上 `COMPLETED 0:0`。FIXED 与
   REMATCH 各完成 `2010/2010` 更新与 scheduler step，skip、监督耗尽、
   `gt_birth_runtime_entry_free_collisions` 均为零；七项 H2 机制、单调校准、
   causal-delta end、past-only pointer、runtime-no-GT、技术和 learning-readiness
   门全部通过。epoch 1 固定 0.5 零发射只表示尚未 operational，不是性能拒绝。

2. 正式十二轮源训练 FIXED `1179373` 与 REMATCH `1179374` 均
   `COMPLETED 0:0`，每臂完成 `12×2010=24,120` 更新和同数 scheduler step；
   skip、监督耗尽、GT-birth entry-free collision、fatal、OOM、NaN/Inf 均为零。
   第 3/6/9/12 轮 checkpoint、recovery manifest 和 SHA-256 完整，无 quarantine。
   这证明真实 4+2 reserve 首次通过两臂完整十二轮容量与训练完整性门。

3. 源 calibration-only 曲线（fraction）为：

   - FIXED e3/e6/e9/e12：`0.0000059524 / 0.0000802264 /
     0.0022547243 / 0.0072212087`，选择 e12，9,044 个最终发射；
   - REMATCH e3/e6/e9/e12：`0.0000087057 / 0.0010357575 /
     0.0095973653 / 0.0049836061`，选择 e9，12,434 个最终发射。

   这些值只用于 calibration checkpoint 选择，不是 reporting 主结果，不能据此宣称
   FIXED 优于 REMATCH。源两臂实际分配资源约为 `5.896111 + 5.908056 =
   11.804167 GPU·h`；后续重放资源必须累计报告，不能隐藏这一成本。

4. 旧终检 `1179375` 在产生 `formal12_gate.json` 之前因
   `non_monotonic_sequence` 退出，因此当时没有合法正式 mAP/Recall。对 8 份源
   calibration ledger 的独立只读审计证明：

   - FIXED e3/e6/e9/e12 的 sequence inversion 为 `19/1293/1582/433`，
     REMATCH 为 `0/45/608/740`；
   - 所有 inversion 都发生在相同 `emit_frame` 内，真正的 emit 时间倒退为零；
   - 每个 stream 的 sequence 唯一、连续覆盖 `0..N-1`，event id 与 sequence 一致；
   - 按原生 sequence 查看时 emit frame 不倒退；重复 event、非正长度、未来端点、
     未来特征、负延时和 immutable 违规均为零。

   根因是 DDP 汇总后的 ledger sort 在并列 emit frame 内忽略显式 sequence，按区间、
   类别和分数重新排序。这是序列化/评测合同缺口，不是模型 lifecycle、容量、因果或
   定位性能失败，也不需要重训。

## 当前执行点 M52

并列帧 tie-break、sequence fail-closed 摘要和逐事件 replay 等价验证已在 exact
`0daf681b4e8e36a64066a2f6fa8e0458a98b429a` 实现并推送。排序仍以
`emit_frame` 为第一关键字，同一帧内才按显式 sequence 保持提交顺序，所以跨帧错误
不会被静默修正。N16R4 exact clean 通过 `190 passed in 87.71s`、两条 Bash 语法
检查和 no-submit preflight，结束 SHA 与工作树干净。

当前仅从源 run
`/data/run01/sczc063/yuzibo/runs/persistent_binding/formal12_boundary_calibration_batched_seed705_20260722_040902`
保全的 checkpoint 重放 calibration/评测，不调用训练。新 run 为
`/data/run01/sczc063/yuzibo/runs/persistent_binding/formal12_calibration_replay_seed705_20260722_103003`：

- FIXED replay `1179456`；
- REMATCH replay `1179457`；
- 依赖正式终检 `1179458`。

首次核验时两条 replay 臂均在 `g0066` 运行、fatal=0，终检按依赖等待；尚未产生
epoch receipt 或最终 gate，属于正常早期状态。监控脚本为
`C:\tmp\ontad_calibration_replay_progress_0daf681.ps1`。

每臂第 3/6/9/12 轮 replay receipt 必须同时证明：

1. 源与重放包含相同视频、stream、event id 和逐事件 payload；
2. 唯一变化是同一 emit frame 内的序列化次序；
3. replay sequence、正长度、不可变与全部因果/时序违规为零；
4. calibration 指标、选择 epoch、checkpoint SHA 和选择收据与源完全一致；
5. 资源收据累计源训练和本次 replay。

两臂通过后，`1179458` 才能执行固定 0.5 正式门，并必须写出
`formal12_gate.json`、pair completion 和 artifact manifest。只有此时产生的指标才是
合法的 feature-level seed-705 正式结果。

## 下一步裁决

- 若 replay 事件或指标不等价：判为 replay/协议实现失败，保全证据并停止，不解释为
  模型性能。
- 若容量、因果、正长度、immutable 或 sequence 门失败：按对应科学合同拒绝，不绕门。
- 若 replay 与正式终检均通过：先完整记录 FIXED/REMATCH 单 seed 结果、延时、容量、
  因果和资源，再决定是否授权 feature-level multi-seed；不能直接跳到 raw-RGB。
- multi-seed 通过后才进入论文主实验：强在线基线、消融、不同动作长度/相邻动作/
  并发实例分层、延时—精度曲线、资源与确定性审计；最后才评估 raw-RGB 阶梯。

## M53 正式门结论与下一任务

重放 `1179456/1179457` 与终检 `1179458` 均已 `COMPLETED 0:0`，8 份事件等价/
因果收据和累计资源门全部通过。冻结协议明确：第 3/6/9/12 轮只组成 calibration
曲线并保留候选，固定 `0.5` 的正式科学门始终使用第 12 轮；REMATCH 曲线选择 e9
不替换正式 e12。此前把它解释为 checkpoint 绑定错误的同家族审计遗漏了预注册上下文，
已经纠正且没有据此提交代码或追加实验。

合法第 12 轮 calibration-only 结果：FIXED mAP=`1.614484`pp、Recall@0.3=
`0.154167`、prediction/GT=`18.841667`；REMATCH mAP=`1.178473`pp、Recall=`0.0875`、
ratio=`16.097917`。两臂完整 24,120 更新、容量、监督、因果、正长度、序列、固定阈值和
预算全部通过，所以技术链路成立；但 prediction/GT 都远高于上限 4，Recall 都低于
下限 0.25，运行性能门明确失败。FIXED 的 `+0.436011` mAP 点只作为单 seed 方向信号。

下一任务是 `model_optimization_on_fit_and_calibration_only`：先分解假阳性、类别、起止
边界和重复生命周期的来源，再成对改进损失/生命周期校准；不搜索或降低 0.5，不访问
reporting，不启动 multi-seed/raw-RGB，也不再原样重跑本次 12 轮。

## 结论边界

当前可以声称：H2 的训练、监督、严格因果和真实 4+2 容量机制已在一轮与十二轮成对
运行中通过；源账本失败被定位为同帧序列化合同问题，并已用 exact 回归修复。

当前不能声称：FIXED 优于 REMATCH、达到论文主结果、对外数据集泛化、raw-RGB
端到端有效，或已经获得合法 reporting mAP/Recall。正式终检完成前，最准确的状态是：
**十二轮模型已训练并保全，checkpoint-only 等价重放正在运行，最终性能裁决待定。**

## 恢复入口

- 分支：`codex/ontad-science-fixed-rematch`
- 代码库：`https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702`
- 完整实验记录：`research-wiki/experiments/ontad-afternoon-three-model-design-20260721.md`
- 时间日志：`research-wiki/log.md`
- 当前代码提交：`0daf681b4e8e36a64066a2f6fa8e0458a98b429a`
- 源训练提交：`5edc46c34c0db56e79409fac69276450ad87e949`
