# FIXED/REMATCH 严格确定性画像门禁记录

日期：2026-07-20
状态：技术与因果门禁通过，冻结预算门禁拒绝

## 本次要回答的问题

在提交完整 seed-705 配对训练之前，先用真实 THUMOS14 缓存特征回答四个问题：

1. FIXED 和 REMATCH 是否都能连续训练 250 个 chronological chunk；
2. 修复后的 birth/lifecycle 是否仍会丢失 GT birth 或耗尽运行时容量；
3. 两种配置在训练前是否具有完全相同的严格因果推理行为；
4. 原定双臂 12 epoch 实验是否落在冻结的 2 GPU·小时预算内。

这不是效果实验，不产生 FIXED 优于 REMATCH 的方法结论。

## 固定协议

- 代码提交：`95fa963e7e2f3a05779907999df7f9311be4f4fd`
- Slurm job：`1176983`
- 远端运行目录：
  `/data/run01/sczc063/yuzibo/runs/persistent_binding/profile_20260720_204036`
- GPU：单张 NVIDIA GeForce RTX 4090
- seed：`705`
- 输入：固定缓存因果特征，stride 8，维度 768
- 数值：FP32
- 计时：50 步预热，随后 200 步测量
- 训练：fit split，160 个视频，2010 个 chunk
- 推理：calibration split，40 个视频，469 个 chunk，`test_mode=True`
- reporting split：只读取冻结的 2719 个 chunk 数量用于成本估算，不读取效果
  标签或预测
- 确定性：严格 deterministic algorithms；关闭 flash/memory-efficient SDP，
  仅启用 math SDP
- 安全系数：`1.25`
- 双臂单种子预算上限：`2 GPU·小时`

## 实测结果

| 路径 | 均值秒/chunk | P95 秒/chunk | 峰值显存 MiB | GT birth 丢失 | 运行时容量耗尽 |
|---|---:|---:|---:|---:|---:|
| FIXED train | 0.712968 | 0.771883 | 140.534 | 0 | 0 |
| REMATCH train | 0.713178 | 0.782507 | 140.534 | 0 | 0 |
| FIXED calibration inference | 0.283217 | 0.414226 | 38.855 | 不适用 | 0 |
| REMATCH calibration inference | 0.284580 | 0.421410 | 38.855 | 不适用 | 0 |

两条训练路径都通过参数更新审计。两条未训练推理各产生 10,029 个不可修改的
最终发射，并满足：

- future-end violations：0
- future-source violations：0
- negative-latency rows：0
- non-monotonic emit rows：0

两条推理账本的规范摘要完全一致，SHA-256 均为：

`7b6c520a4ecf9d010531a66298c04e1d55cf8f11f4e4aae048274e2bd8dd9e47`

这证明 FIXED/REMATCH 的监督绑定开关在训练前没有改变推理路径。

## 冻结成本裁决

| 项目 | 预计 GPU·小时 |
|---|---:|
| 双臂 12 epoch 训练 | 9.555176 |
| 双臂 calibration 推理 | 0.073971 |
| 双臂 reporting 推理 | 0.428845 |
| 未加安全系数总计 | 10.057992 |
| 乘以 1.25 后 | 12.572490 |
| 冻结上限 | 2.000000 |

最终门禁：

- `stability_passed=true`
- `binding_inference_equivalence_passed=true`
- `budget_passed=false`
- `passed=false`

job `1176983` 的 Slurm 状态为 `FAILED`，退出码 `1:0`。这是画像门禁在预算超标后
主动以非零状态停止，不是训练崩溃、CUDA 错误或因果协议错误。

`profile_gate.json` 的 SHA-256 为：

`c824ca64934234277f889b30f683dde9a6899b1f38857bb721f3d8dad1462b98`

## 科学解释与停止决定

- 250 步真实数据画像支持“当前 birth/lifecycle 修复不再立即丢失监督或耗尽
  容量”，但不能替代完整训练和全数据技术门禁。
- 当前证据没有检验 FIXED 是否降低 duplicate/fragmentation，因此不能宣称方法
  有效或无效。
- 原定 12 epoch 双臂单种子计划超过冻结预算约 `6.29×`，不得提交 seed-705
  主训练。
- seeds 706/707、论文主结果和 raw-RGB 联合训练继续锁定。
- 下一步必须显式修订“训练时长、数据规模或资源上限”之一并重新注册；不能在
  已见画像结果后静默放宽预算。
