# On-TAD FIXED/REMATCH Scientific Review Prompt

Date: 2026-07-20
Status: review completed and absorbed; do not resend before repair

本 Prompt 已获得完整审查。审查原文固定到较早提交 `27a59de`，已逐字节归档为
`../../PRO_ONTAD_FIXED_REMATCH_SCIENCE_READINESS_REVIEW_20260720.md`；针对当前
`95fa963` 代码、Slurm smoke 和严格画像的独立复核已记录在
`../../PRO_ONTAD_FIXED_REMATCH_SCIENCE_READINESS_ABSORPTION_20260720.md`。

- 审查原文：
  https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/ontad-science-fixed-rematch/PRO_ONTAD_FIXED_REMATCH_SCIENCE_READINESS_REVIEW_20260720.md
- 独立复核与吸收：
  https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/ontad-science-fixed-rematch/PRO_ONTAD_FIXED_REMATCH_SCIENCE_READINESS_ABSORPTION_20260720.md

当前结论是“先修科学合同”，不是“继续讨论预算”。在 endpoint、出生步 REMATCH、
same-step short action、reporting 隔离、指标/gate/provenance 和 fail-closed census
修复并重新通过 smoke/profile 之前，不应再次发送本 Prompt，也不应提交 seed 705。

下面的中文 Prompt 可直接交给另一位研究审查者。代码链接固定到已通过
N16R4 严格确定性画像的提交
`95fa963e7e2f3a05779907999df7f9311be4f4fd`。

---

你是一位严格但务实的 Online Temporal Action Detection（On-TAD）研究审查者。
请只讨论视频时序动作定位的方法、训练、严格因果协议、评测和论文证据，不扩展到
本研究范围之外的话题。

## 研究目标

我们研究的是标准、全监督、严格因果的 On-TAD：模型在时刻 `t` 只能看到当前及
过去的视频前缀，端到端维护每个动作实例的开始、持续和结束，并在动作结束后以
低延时输出一次不可修改的最终区间 `{start, end, class, score}`。

当前阶段只做固定缓存因果视频特征上的 FIXED/REMATCH 证伪实验，不是 raw-RGB
实验。只有特征级技术门槛和科学门槛同时通过，才允许进入 raw-RGB 联合训练。

## 代码与设计入口

- 仓库：
  https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702
- 科学分支：
  https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/tree/codex/ontad-science-fixed-rematch
- 本次审查的固定代码提交：
  https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/tree/95fa963e7e2f3a05779907999df7f9311be4f4fd
- 冻结设计：
  https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/95fa963e7e2f3a05779907999df7f9311be4f4fd/research-wiki/experiments/ontad-science-fixed-rematch-design-20260720.md
- 问题地图：
  https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/95fa963e7e2f3a05779907999df7f9311be4f4fd/research-wiki/experiments/ontad-science-fixed-rematch-problem-map-20260720.md
- 当前实验计划：
  https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/ontad-science-fixed-rematch/research-wiki/experiments/ontad-science-fixed-rematch-plan-20260720.md
- 严格确定性画像与预算裁决：
  https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/ontad-science-fixed-rematch/research-wiki/experiments/ontad-science-fixed-rematch-profile-20260720.md

## 重点代码

- 持久实例监督器：
  https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/95fa963e7e2f3a05779907999df7f9311be4f4fd/opentad/utils/prefix_trajectory_supervision.py
- 持久查询头与生命周期：
  https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/95fa963e7e2f3a05779907999df7f9311be4f4fd/opentad/models/dense_heads/persistent_event_set_head.py
- 在线检测器：
  https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/95fa963e7e2f3a05779907999df7f9311be4f4fd/opentad/models/detectors/persistent_trajectory_ontad.py
- 严格因果特征数据：
  https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/95fa963e7e2f3a05779907999df7f9311be4f4fd/opentad/datasets/streaming_feature.py
- 实例错误指标：
  https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/95fa963e7e2f3a05779907999df7f9311be4f4fd/opentad/evaluations/online_instance_metrics.py
- 冻结结果门槛：
  https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/95fa963e7e2f3a05779907999df7f9311be4f4fd/opentad/evaluations/persistent_binding_gate.py
- 共享基础配置：
  https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/95fa963e7e2f3a05779907999df7f9311be4f4fd/configs/causaltad/thumos_persistent_binding_base.py
- FIXED 配置：
  https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/95fa963e7e2f3a05779907999df7f9311be4f4fd/configs/causaltad/thumos_persistent_binding_fixed.py
- REMATCH 配置：
  https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/95fa963e7e2f3a05779907999df7f9311be4f4fd/configs/causaltad/thumos_persistent_binding_rematch.py
- 聚焦测试：
  https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/tree/95fa963e7e2f3a05779907999df7f9311be4f4fd/tests
- 严格确定性画像器：
  https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/95fa963e7e2f3a05779907999df7f9311be4f4fd/tools/profile_persistent_binding.py
- 画像稳定性与预算门禁：
  https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/95fa963e7e2f3a05779907999df7f9311be4f4fd/tools/evaluate_persistent_binding_profile.py
- N16R4 Slurm 画像入口：
  https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/95fa963e7e2f3a05779907999df7f9311be4f4fd/tools/remote/submit_persistent_binding_profile_n16r4.sh

## 已知背景与当前状态

1. 旧控制器使用 4 个槽位，却出现 2,206 个 GT birth exhaustion；冻结标注普查显示
   同时可见实例只需要 2 个槽位。主要原因不是动作真实并发，而是预测错误占用和
   完成后的容量滞留。
2. 新实现把 `supervision_state` 与 `runtime_state` 分开。预测占用不得删除真实
   birth 监督；推理运行态不得包含 GT 实例身份。
3. 共享运行生命周期改为
   `FREE -> CANDIDATE -> ACTIVE -> COMMIT -> FREE`，无完成后滞留；
   每一步最多接纳 2 个候选 birth。
4. FIXED 与 REMATCH 应只改变“实例 birth 后，loss target 是否保持原槽位”：
   FIXED 保持绑定，REMATCH 在同一个规范占用池内逐前缀重匹配。运行生命周期、
   birth 机会、阈值、推理与评测必须完全相同。
5. 输入为 stride 8、768 维固定缓存特征；raw-RGB 路线仍被明确关闭。
6. 本地完成源码编译与无 Torch 测试；N16R4 的 PyTorch 2.0.1 环境已运行 64 个
   聚焦测试并通过。标注、类别表、特征 manifest 和冻结的 160/40/211 数据划分
   已核验。
7. Slurm 整链路 smoke 已在提交 `097bc72`、作业 `1176737` 上通过：两种绑定都
   完成真实反向传播且未丢 GT birth；标准 runner 的 checkpoint 相对同种子初始化
   有真实权重变化；重新加载后的流式账本逐项一致；最终区间非空且无未来违规。
   详细记录见：
   https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/blob/codex/ontad-science-fixed-rematch/research-wiki/experiments/ontad-science-fixed-rematch-smoke-20260720.md
8. 严格确定性画像已在提交 `95fa963`、作业 `1176983` 上完成。FIXED/REMATCH
   各自连续训练 250 个真实 chunk，均为零 dropped GT birth、零 runtime capacity
   exhaustion；两条未训练推理产生完全相同的 10,029 条最终发射账本，且无未来违规。
9. 冻结预算门禁拒绝了原定实验：双臂 12 epoch 预计为 12.572 GPU·小时，而上限为
   2 GPU·小时。因此 seed-705 未提交；这不是方法效果失败，而是当前训练协议不能
   在已注册预算内执行。
10. 尚未完成单种子配对筛选、三种子正式实验和论文主结果。因此不能说科学假设
    已经成立，也不能进入 raw-RGB。
11. 后续 readiness review 又确认了 9 个 P0 和 10 个 P1 问题。当前仍成立的核心
    P0 是：未监督 endpoint offset、出生步 REMATCH、same-step birth+end 漏提交、
    locked reporting 被逐 epoch 访问，以及 instance metric/standard mAP/result
    provenance 未闭合。完整逐条裁决见上面的 absorption 文件。

## 请逐项审查的全部问题

### A. 严格因果性

1. 沿着 dataset → detector → head → emission → evaluator 的真实调用链，检查任意
   时刻的预测是否只依赖 `<= t` 的特征与模型状态。
2. 检查推理接口是否可能接收 annotation、未来 endpoint、视频结束标志、总时长、
   全视频预测缓存或未来 chunk 信息。
3. 检查视频切换、chunk 边界、state detach、重置和输出时间戳是否可能造成跨视频
   状态污染、未来泄漏或非单调输出。
4. 检查最终区间是否只提交一次、提交后不可修改，且没有离线 NMS 或全视频修正。

### B. 监督与运行态隔离

1. 证明或否定：任意预测槽位占用模式都不能减少当前真实 first-crossing birth 的
   监督目标。
2. 检查 canonical slot capacity exhaustion 是否只代表真实监督并发/配置错误，而
   不是运行态副作用。
3. 检查训练前缀目标是否只使用当前已经可观察的 annotation 事实；尤其检查 start、
   alive、end 和短动作的定义。
4. 检查 REMATCH 是否只影响 loss binding，绝不能反向改变 canonical birth、
   lifecycle 或 runtime state。

### C. birth/lifecycle 容量修复

1. 对同一步发生旧实例结束与新实例 birth、相邻动作、极短动作、重复同类动作、
   同类重叠动作、错误 birth、错误 ACTIVE、候选取消等情况逐一构造反例。
2. 判断“本步释放、下一步可复用”是否会漏掉合法动作；若会，请给出最小反例和
   不破坏单变量比较的修复。
3. 检查 CANDIDATE 确认语义是否与训练 target 对齐，是否引入固定一拍延时、静默
   控制器或重复提交。
4. 检查每步最多接纳 2 个 birth 是否由冻结数据事实支持，是否对验证/测试分布仍然
   安全，以及超过时应记录还是直接失败。
5. 区分并核对三个计数：GT 监督槽位耗尽、运行态容量不足、候选抑制；禁止把它们
   混成一个“容量问题”。

### D. FIXED/REMATCH 单变量实验

1. 展开两个最终配置，机器级比较全部字段，确认除
   `model.trajectory_binding_mode` 和输出目录外没有差异。
2. 检查随机种子、数据顺序、初始化、优化器、学习率、epoch、阈值校准、checkpoint
   选择和评测均为配对设计。
3. 检查 REMATCH 的代价矩阵、tie-break 和 active pool 是否确定、因果且不会得到
   额外监督信息。
4. 判断这个对照能否单独支持论文主张“持久绑定降低实例重复与碎片化”，并列出所有
   仍可能混淆结论的因素。

### E. 指标与结果门槛

1. 审查 duplicate rate、fragmentation rate、重复同类/重叠同类 recall、commit
   delay 的匹配与分母定义，尤其处理 unmatched prediction、一个预测覆盖多个 GT、
   tie 和短区间。
2. 审查 `E_id = 0.5 * (duplicate_rate + fragmentation_rate)` 是否合理；如果建议
   额外报告别的指标，请保留冻结主指标，不允许看结果后替换。
3. 审查技术门槛：每个 arm/seed 都要零因果违规、零 dropped GT birth、零无法解释的
   runtime exhaustion、prediction/GT 比在 `[0.25, 4.0]`、Recall@0.3 至少 `0.25`
   且提交数非零。
4. 审查科学门槛：FIXED 相对降低 `E_id` 至少 20%，三种子中至少 2 个改善，平均
   mAP 相对 REMATCH 下降不超过 0.5 个百分点。
5. 检查当前 evaluator 和训练/测试入口是否真的会产出门槛函数要求的全部字段；
   不要只审查独立指标函数。

### F. 尚未完成的整链路风险

1. `formal_training_ready=False` 必须继续保持；列出预算协议修订并通过单种子
   配对筛选后，改成 `True` 之前仍缺少的最小检查。
2. 审计已经通过的 Slurm smoke 是否确实覆盖：真实 manifest、少量视频、一次前向、
   一次反向、一次 checkpoint、一次重载、一次流式推理、一次 emission 序列化和
   一次评测。
3. 检查 AMP、batch size 1、流状态、梯度截断、detached state 与 12 epoch 配置是否
   在 OpenTAD runner 中实际兼容。
4. 检查最终 emission 的帧坐标、秒坐标、fps、source/emit 时刻与 evaluator schema
   是否一致。
5. 给出单种子筛选失败时的停止条件；不得通过事后放宽阈值、压低 birth 或增加槽位
   来制造“零容量错误”。
6. 审查冻结的成本公式和画像结果。若保留 2 GPU·小时上限，给出不偷看 reporting
   结果、且不把欠训练当作方法失败的最小可证伪协议；若建议提高预算，必须同时说明
   与原三种子 10 GPU·小时总上限的冲突。
7. 当前通用 `tools/train.py` 会按照 `val_eval_interval=1` 在每个 epoch 调用
   `dataset.test`。正式路线必须改为 fit 训练、calibration 选阈值、冻结后仅一次
   reporting；请审查这一专用入口的最小实现要求。

### G. 从当前阶段到论文主实验

请给出完整、按依赖排序的实验表，至少包括：

1. 已完成 Slurm 整链路 smoke 的证据复核；
2. 已完成严格确定性画像及 2 GPU·小时预算拒绝的证据复核；
3. 明确注册预算兼容的训练协议，以及 fit/calibration/report-once 专用入口；
4. 单种子 FIXED/REMATCH 非退化筛选；
5. 仅用 calibration split 冻结共享阈值；
6. seeds `705/706/707` 的配对正式实验；
7. 标准 On-TAD 指标、实例指标、延时与资源报告；
8. 必要消融：绑定策略、候选确认、容量预算，以及重复/重叠同类子集；
9. 与合理在线基线的公平比较；
10. 失败分析和可视化；
11. 只有双门槛通过后才安排 raw-RGB frozen encoder 与 PEFT/joint training。

每项写清：要证明的命题、FIXED/REMATCH 自变量与控制量、数据 split、seeds、输入
类型、输出指标、通过/停止条件、前置依赖和预估 GPU 成本。不得把 raw-RGB 结果与
当前特征级结果混写。

## 期望输出格式

1. **一句话结论**：`可注册预算修订 / 必须先改训练协议 / 当前设计无法在预算内证伪`
   三选一。
2. **P0/P1/P2 问题表**：每项给出代码链接、触发条件、影响、最小修复和必须新增的
   测试；没有证据的问题不要臆测为已发生。
3. **因果与状态不变量表**：逐条写“成立/不成立/证据不足”。
4. **最小反例**：重点覆盖同一步 end+birth、相邻动作、短动作、重复同类与重叠同类。
5. **单变量公平性结论**：明确 FIXED/REMATCH 是否真的只差 loss binding。
6. **整链路 smoke 命令与验收项**：必须通过 Slurm，不要直接在登录节点正式训练。
7. **完整论文实验计划表**：区分 feature 阶段和条件 raw-RGB 阶段。
8. **最终建议**：明确下一步唯一任务，以及现在是否值得再发起一次高成本深入讨论。

不要因为聚焦测试通过就默认训练整链路可用，也不要在没有三种子结果时替方法宣称
有效。优先寻找能够推翻当前实现或实验设计的具体反例。
