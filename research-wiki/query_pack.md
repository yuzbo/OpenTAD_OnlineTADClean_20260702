---
type: query_pack
updated: 2026-07-23
status: active
scope: Current compressed context for the official-parent strict-causal On-TAD task.
---

# Query Pack: Official-MATR-Parent Strictly Causal On-TAD

## 最终任务

研究标准、closed-set、全监督、严格因果的 Online Temporal Action
Localization。最终模型直接读取原始 RGB；时刻 `t` 只能使用来源时间不晚于 `t`
的帧和内部状态。模型在开始证据出现后建立并维护动作实例，检测到结束后低延时一次性
提交不可修改的 `{start,end,class,score}` 正长度区间。

`event_id`、provisional start/class、cancel、late birth 和 owner trajectory 是内部状态
及诊断，不是标准输出。必须满足
`start < end <= source_time <= emit_time`。结束边界独立解码，不能令
`end=emit_time`。推理不得读取未来帧、GT identity/annotation、terminal 信息、全视频
回看或事后修改输出。OnVLLM 是独立后续课题。

## 已完成的旧路线证据

旧 FIXED/REMATCH 是 SigLIP2 特征上的槽位负基线。完整 12 轮技术链通过，但 epoch-12
合法结果很低：FIXED mAP `1.614484`pp、Recall@0.3 `0.154167`、prediction/GT
`18.841667`；REMATCH mAP `1.178473`pp、Recall `0.0875`、ratio `16.097917`。
这些结果只说明旧模型过量发射、召回很低。

用户在 M56 明确废弃旧路线对新模型的训练设置。新的实验不得继承其 12 epochs、
seed705、SigLIP2、batch1、AdamW `2e-4` 或六入口 guard。此前未提交、未部署的
OpenTAD `K×O` 实现与 launcher 同步废弃；历史结果不删除但不作为新母体。

## 官方母体与机制供体

唯一母体是 MATR ECCV 2024 官方实现：

- repo: `https://github.com/skhcjh231/MATR_codebase`;
- SHA: `ba05a98d451b3541c1a5377026f17dc1102fa217`;
- 作用：标准 On-TAD、当前片段 end decoder、过去 memory start decoder；
- 只读官方树不修改；派生修改位于独立库
  `E:/DeskTop/TAD/OpenTAD_OnlineTADClean_20260702/_codex_worktrees/matr-event-memory-official-parent`。

ActionSwitch 官方 SHA
`838a6ccbd8f2cce414688ff2843380d712aa7b89` 是 class-agnostic 即时状态转移供体和
独立参考，不是母体。HAT SHA `a38dad6...` 与 OnVTG/HEM SHA `4f629e8...`
只作后续记忆参考；HEM 的任务是文本查询 OnVTG，不能直接定义 On-TAD 设置。

选择 MATR 是因为它已经解决完整区间与长期记忆，距离我们的目标只差关键机制：
它通常在动作结束附近用 fresh queries 定位实例；我们要在 prefix-visible 开始证据处
创建事件、保持 owner、持续更新类/边界，并用 owner-conditioned end 关闭它。

## 官方实验设置

原生 parity 与新的四臂全部继承 MATR THUMOS14 默认设置：官方 RGB+flow 4096 维
feature pickles、segment 64、queries 10、memory 7、`gap2`、hidden 1024、FFN 2048、
3 encoder/5 decoder layers、batch 64、100 epochs、seed 52、Adam、LR
`1e-8→1e-5`、cosine warm-up/restarts `T_up=3/T_0=10/gamma=0.9`、wd `1e-4`、
focal loss、class threshold `0.1`、flag threshold `0.5`、online-order NMS `0.3`，
评测 tIoU `0.3:0.1:0.7`。

官方 native lane 保留原架构、标签和损失。严格因果 matched run 如需把批量 post-hoc online_nms 改成
逐 generation-time 执行，必须对所有臂对称应用并证明事件等价；这不是模型贡献。
训练只使用完整官方 validation/train 集，不划 calibration、不构造 test loader、不按 test
选择 checkpoint，只保留 epoch-100 终点。训练与合同通过后，每路冻结终点模型只允许一次
独立 locked test。

## 新模型 EventMATR（暂名）

保留官方 MATR encoder、memory queue、双 decoder 和 heads；exact native MATR 是独立
lane，四个事件化单元使用相同 transition head、owner decoder 和 loss，只研究两个因素：

```text
native_matr = exact official architecture/labels/losses，位于 B×O 外
B0O0 = delayed MATR-style event birth + fresh rematch
B1O0 = immediate transition event birth + fresh rematch
B0O1 = delayed MATR-style event birth + sticky owner
B1O1 = immediate transition event birth + sticky owner，候选 EventMATR
```

`B1O1` 在第一次因果 start crossing 建立 ragged event record，保存 owner query、start
分布、class/alive belief 和 causal memory view。active event 没有手工语义槽位数；物理
安全上限只能 fail closed，不得静默截断。结束由 owner query 条件化并与 emit 分离。
训练可由完整区间标注构造 prefix-visible birth/alive/first-crossing end target，但 runtime
state 禁止携带 GT。相同类重叠在 birth 前允许 permutation-aware matching，birth 后 owner
固定直到 cancel/end。

START/ALIVE/END/BACKGROUND 使用四态竞争 argmax；事件 birth/end 不使用统一 `0.5`。
active owner 预测 BACKGROUND 时执行可学习 CANCEL，不发射区间并允许以后 rebirth。官方
`flag_threshold=0.5` 只保留在 native MATR 的 memory-admission 语义中。手工 same-start
去重和 cosine `>0.95` 合并已删除，以免误并同帧同类的不同实例。十个 query 是每个 prefix
的预测带宽，不是十个持久语义槽位。

## 科学门

候选必须先证明官方 MATR parity，再同时超过 `B1O0` 和 `B0O1`。主要结果包括标准
mAP/Recall，同时报告 wrong-start、owner swap、fragmentation、duplicate close、birth/end/
emit latency、活动事件数、记忆和吞吐。因果、正长度、不可变、double-close 违规必须为零。
增益不能来自更多特征、未来信息、额外后处理或更晚提交。

特征实验只证明机制，不支持 raw-RGB claim。四臂通过后，保持成功的事件机制不变，
并行比较 raw-RGB causal backbone 的 frozen、adapter/LoRA 与 joint training；最终论文
证据必须来自 original RGB。

## 当前执行状态与下一步

1. 官方四库已只读克隆并记录精确 SHA；MATR 原始设置已逐文件核验。
2. 已创建独立 MATR 派生库和 `codex/matr-event-memory` 分支；官方只读树未修改。
   当前实现 exact=`64d7f78dd8ed1436bac08ebfb03b51c90142129b`，
   tree=`bc6f5057e954ed448dcb5173489eba7c4dcba27d`，manifest
   SHA-256=`B6D3B51A14253E66A9D5C110F5B08FDAF96C31EB48FB2DF9A3B57CD5E61FB1C9`。
3. native isolation、四个 B×O 模型、损失、ragged runtime、训练/终检/locked-test 脚本和
   精确源码身份收据均已实现。
4. 本地完整合同为 `62 passed`；official protocol、Python compile、Git-Bash syntax 与
   `git diff --check` 均通过。这证明实现链，不证明性能。
5. 下一门是单卡 Linux 上的真实官方训练批次 smoke：五路各 forward/backward/Adam step、
   finite loss/gradient、event/owner gradient、checkpoint strict reload，且 test access=false。
6. smoke PASS 后并行释放 native + 四个 EventMATR `100`-epoch lane，随后冻结终点各做一次
   locked test；不是先用 calibration 选 checkpoint。
7. 当前 N16R4 未找到精确官方 MATR feature/annotation，且远端 Google Drive 不可达；因此
   正式 DAG 尚未提交。不得用 OpenTAD/SigLIP2 特征替换后伪称官方 parity。
8. 结果写 Wiki并裁决机制后，才进入最终 raw-RGB 阶段。

主决策：`research-wiki/decision_register.md` / DR-031。
实验记录：
`research-wiki/experiments/ontad-matr-official-parent-event-model-20260722.md`。

## 外部 DUCA-TTDI 审查边界

2026-07-22 的 DUCA 审查固定到另一仓库
`yuzbo/OpenTAD_C3_CoarseClean_20260702@a00498e`，附件 SHA-256 为
`36523B2F1A7456F8D4A4314EA445971F8066EEC59611F9632D7BC1D33E31A884`。
它审查的是离线 full-window pre-backbone 不规则 RGB 采样，不是当前严格因果 EventMATR。
独立复核接受 selected-rank 时间扭曲、mandatory completeability、五预算 parser、
source-equivalence、matched uniform/high-tIoU/multi-seed/full-stack cost 等问题；TTDI 仅是
U/L/T 单变量待证伪假设，2000/6000 updates、LR/loss/schedule、55% sign 与 0.2 mAP
容差均未冻结。该审查不改变 MATR 母体、官方 100 epochs 或 EventMATR 第一阶段。
唯一迁移护栏是：未来 raw-RGB 若引入非均匀采样/压缩，原始 physical timestamps 必须
进入 backbone 和 localization head，不能只在输出端做坐标逆映射。完整记录见
`research-wiki/experiments/duca-ttdi-external-review-absorption-20260722.md` 与 DR-032。
