---
type: query_pack
updated: 2026-07-28
status: active
scope: Current compressed context for the official-parent strict-causal On-TAD task.
---
> N16 academic data-staging rule (2026-07-23): official MATR Google Drive

> Status update (2026-07-26): official data is now verified and hash-frozen;
> real five-lane smoke `1190483` is pending with N16 default per-GPU memory after
> its scheduler rejected a redundant explicit memory directive. No test access,
> performance result, or raw-RGB release exists yet.

> Correction: `1190483` failed before any model operation because a temporary-repo
> verifier test inherited launch identity. Exact fix `ba3f153` passed remote
> protocol plus 63 tests; replacement smoke `1190605` is pending. Formal lanes
> remain blocked.

> Latest: corrected-source smoke `1190635` is `PASS` (`COMPLETED 0:0`, g0013,
> 2:08): all five official train-batch lanes completed finite forward/backward/
> Adam, required event/owner gradients, strict reload, and `test_access=false`.
> This authorizes five matched 100-epoch training lanes only; it is not mAP,
> Recall, or model-ranking evidence, and locked test remains unmounted.
> Formal jobs `1190693`–`1190697` are the active five-lane release. Earlier
> `1190688`–`1190692` stopped before Python because the Slurm environment lacked
> `MATR_ENV_ACTIVATE`; they are environment-only failures, preserved for audit.
> Superseded: the four event-arm retries then exposed a non-executable common
> launcher and stopped before Python; native-only training was cancelled to keep
> the pair valid. Exact wrapper-only fix `92cf34a` passed 63 tests and has new
> same-commit official smoke `1190702`; do not release formal lanes until PASS.
> Current: smoke `1190702` passed and exact formal 100-epoch lanes are RUNNING:
> native `1190735`, B0O0 `1190736`, B1O0 `1190737`, B0O1 `1190738`, and B1O1
> `1190739`. They use the exact `92cf34a` source, official train features, and
> prewritten lane identity receipts; locked test remains unmounted.
> One-hour checkpoint: native is in epoch 4 and B0O0/B1O0/B0O1/B1O1 are in
> epoch 6, all RUNNING with finite losses and no fatal/OOM/non-finite-gradient
> marker. Do not mistake interim train-split zero-emission mAP/time diagnostics
> for a locked-test metric or a performance verdict.
> Two-hour checkpoint: native reached epoch 6; all four event arms reached epoch
> 12, all still RUNNING `0:0` with finite losses and no fatal/OOM/non-finite
> gradient marker. Only terminal epoch-100 checkpoints are valid artifacts.
> Three-hour checkpoint: native epoch 7; event arms epoch 18; all five still
> RUNNING `0:0` with finite losses. Treat this only as numerical-health evidence.
> Four-hour checkpoint: native passed epoch 9, event arms reached epoch 24; all
> remain RUNNING `0:0`, finite, and free of fatal/OOM/non-finite-gradient markers.
> Six-hour checkpoint: native passed epoch 12 (now epoch 13); event arms reached
> epochs 35–36. All remain RUNNING `0:0`, finite, and test remains unmounted.
> Partial terminal state: B0O0/B0O1/B1O1 completed `0:0` with one epoch-100
> checkpoint each; B1O0 is finalizing epoch 100 and native remains in epoch 32.
> Do not evaluate or interpret any completed arm until all five lanes pass the
> terminal artifact and completion-contract gate.
> Latest terminal state: B1O0 `1190737` also completed `0:0` with one
> 2,150,337,711-byte epoch-100 checkpoint. Native `1190735` alone remains
> running (epoch 42, batch 3204/3270); no inspected fatal/OOM/RuntimeError/
> non-finite-gradient marker. All performance interpretation and locked-test
> access remain blocked pending native completion, hashes, and the pair receipt.
> retrieval uses process-local `http_proxy` and `https_proxy` through the N16
> academic endpoint `10.244.6.36:3128`. Credentials are externally injected
> and must never be committed or written into this Wiki. The official Drive
> probe returned HTTP 200. The staged downloader stores its cache on the data
> disk, verifies/extracts the official `thumos_dataset.zip`, then validates and
> hashes the two feature pickles plus `thumos14_v2.json`; only the resulting
> `OFFICIAL_MATR_DATA_READY` sentinel may release the five-lane real-data
> smoke.

> Current staging status (third hourly monitor): the resumable official archive download is `1,041/11,622` 512KB ranges complete (about 533MB) and the N16 screen session is alive. This is a gain of 499 complete ranges since the preceding monitor. The proxy cannot sustain aggressive parallelism, so a single-range worker is intentionally used for integrity-preserving resume. No ready sentinel, feature/annotation file, manifest, smoke job, model result, or performance claim exists yet.

> Latest D0 status (2026-07-28): official data and five-lane real-batch smoke
> are complete. Four EventMATR lanes reached epoch 100; native MATR timed out
> without a terminal epoch-100 artifact. Empty event-arm training proposal files
> are protocol/runtime-off evidence because runtime is disabled in
> `model.train()`, not proof of learned all-background collapse. Frozen D0 exact
> `cc06a1e7d70fb8aadbd8282f1d928d34c71995ea` replays all official train
> prefixes in eval mode and audits real-batch gradients without updating a
> checkpoint. Slurm array `1199738` plus finalizer `1199739` are submitted and
> pending under run `eventmatr_v1_d0_seed52_20260728_155830_cc06a1e`.
> `test_access=false`; v1 true-duration/EOS use makes D0 diagnostic rather than
> strict-causal paper performance. D1/v2, locked test, multi-seed and raw-RGB
> remain gated on the D0 receipt.

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

1. 官方四库保持只读并记录精确 SHA；唯一可写母体仍是 MATR 派生库，官方参考树未修改。
2. 官方 THUMOS14 数据已完整下载、格式验证并冻结 SHA；五路真实批次 smoke 已 PASS，
   `test_access=false`。
3. exact `92cf34aa07bebee2a7a7e3661431d5055804b29b` 的四个 EventMATR 臂均完成
   100 epochs 并保存终点 checkpoint；native `1190735` 因 wall time 超时而没有终点，
   因此原五路 parity completion 仍未成立。
4. 训练期 proposal 全空首先是协议事实：默认训练不展开 event runtime，writer 又只读
   ledger；不能据此宣称网络已经学成全背景，也不能访问 locked test 猜性能。
5. 独立 D0 分支 `codex/eventmatr-v1-d0-replay` 的 exact 为
   `cc06a1e7d70fb8aadbd8282f1d928d34c71995ea`，tree
   `3755e090c49b4f132c80a6e81bb593f5d08fe79f`。它严格只读 checkpoint，
   在 official train 上做真实 batch 梯度/稀疏性审计与 eval-mode 全前缀 replay。
6. D0 run 为 `eventmatr_v1_d0_seed52_20260728_155830_cc06a1e`，Slurm 四路数组
   `1199738`，依赖终检 `1199739`；首检队列/依赖/commit/tree/manifest 全匹配。
7. D0 使用 v1 的 `true_duration`、offline EOS 和全视频 frame-to-time，所以结果只回答
   “训练协议零还是 learned-runtime 零”，不具严格因果论文性能效力。
8. D0 若显示非退化 logits/gradient 但 runtime 零，D1 优先修 runtime/decision 对齐；
   若显示 dense 背景主导，则优先 event-normalized censored hazard 和稳定 temporal birth
   assignment。D1 合同通过后才并行 N/R/T/H/TH v2 pilot。
9. locked test、multi-seed、层级视觉记忆贡献和 raw-RGB 仍全部锁定；最终目标没有变化。

主决策：`research-wiki/decision_register.md` / DR-031。
实验记录：
`research-wiki/experiments/ontad-matr-official-parent-event-model-20260722.md`。

## 2026-07-28：EventMATR v2 / Pro 代码审判交接

EventMATR 派生代码已经发布到公开 GitHub 分支
`codex/matr-event-memory`，远端精确 HEAD 为
`92cf34aa07bebee2a7a7e3661431d5055804b29b`，tree 为
`aef4f64bc020df9d39ead9811fbc01407f1c754a`。该提交仍是 EventMATR v1
事实基线，不因下一步模型修订而删除。

当前代码审计要重点复核四个学习风险：稀疏 START/ALIVE/END 被 dense 背景
BCE/CE 淹没；训练默认不逐 prefix 展开真实 event runtime；dense owner prototype
训练与 ragged persistent owner 推理分布不一致；单帧 START crossing 加 rising-edge
使漏过开始后缺乏恢复路径。下一步候选不是抛弃 EventMATR，而是在保留 MATR 母体、
动态 EventRecord、owner-conditioned end、end/emit 分离和严格因果合同的前提下，
审判是否升级为 trajectory + censoring-aware hazard 的 EventMATR v2。

分层记忆仍保留在最终路线中，但必须区分 active-event memory 与 visual-history
memory。核心实例轨迹学习通过前，层级视觉保留/合并不能掩盖全背景塌缩；可以并行实现，
正式归因应后置。完整外部审判 Prompt：
`PRO_TH_EVENTMATR_CODE_REVIEW_PROMPT_20260728.md`。

最新 Pro 回复已按附件 SHA-256
`1BC0FDDA22B694F1EEA4485DF9478D6E252FB22AE4E53156F1B2903814023513`
完整复核，项目裁决为 `PARTIAL ACCEPT / REVISE`。新增关键事实是：
`true_duration` 提前进入 runtime，训练 runtime 默认关闭，而 train mAP writer 只读
ledger，因此零 train mAP 首先可能是空 ledger 协议结果，不能直接证明网络已学成全背景。
下一步先用 v1 checkpoint 做 eval-mode full-prefix replay 和真实 batch 梯度审计，再实现
稳定跨时间 birth assignment、事件归一化 censored hazard、identity lock 和可微 ragged
unroll。Pro 声称的四个 sandbox 代码/patch 文件没有随附件提供，其 SHA 和 `9/9` 测试
不计入项目证据。完整处置见
`PRO_TH_EVENTMATR_CODE_REVIEW_ABSORPTION_20260728.md` 与 DR-035。

## 2026-07-28：D0 已完成，D1 当前交接

D0 recovery 数组 `1200180` 与终检 `1200181` 均 `COMPLETED 0:0`。run
`eventmatr_v1_d0_seed52_20260728_203713_ca914f3` 的四份
`d0_audit_receipt.json` 与 `d0_pair_completion.json` 全部 PASS；
`test_access=false`、`checkpoint_updated=false`，exact diagnostic/source
身份和四个 epoch-100 checkpoint SHA 已绑定。

| lane | birth | emit | EOS 后仍 active | 诊断 avg mAP (%) |
|---|---:|---:|---:|---:|
| B0O0 | 797 | 626 | 1 | 12.6698 |
| B1O0 | 2,116 | 1,716 | 4 | 39.1084 |
| B0O1 | 799 | 336 | 64 | 0.1326 |
| B1O1 | 1,854 | 6 | 1,407 | 0.1758 |

四臂都存在 finite、非零的 event transition、owner attention/state 与 birth/end
梯度，因此不是网络完全死亡。真正问题有两个：birth/end 正例仅约 `0.1477%`，
均值 logit 比常数最优值更负约 6–8 个单位；同时 sticky owner 大量出生却几乎不结束，
说明训练对象与 ragged runtime 生命周期错位。D1 必须同时完成：

1. no-duration/no-offline-EOS 模型边界；
2. event-normalized interval-censored birth 与 right-censored end hazard；
3. stable temporal birth assignment；
4. differentiable chronological ragged unroll；
5. birth 后 identity lock、错误初生 cancel 与 reacquisition；
6. negative-start=0 和既有 ledger 不变量回归。

B1O0 是 D1 的功能参考，不是论文结果。上述 mAP 全部来自 v1 official-train replay，
因 true-duration/EOS 明确 `strict_causal_paper_result_valid=false`。下一任务是在新 v2
分支完成 Stage D1 及测试，然后才提交 `N/R/T/H/TH` seed-52 五路 pilot；locked test、
multi-seed、视觉层级记忆贡献实验和 raw-RGB 继续锁定。

## 正确外部审判的最终吸收（2026-07-23 复核）

唯一有效附件是 `Raw-RGB Dynamic Event Memory On-TAD 独立深度审判`：新附件为
`70,393` bytes、`1,412` 行，SHA-256
`ACA5BB6E9950993F170922250F6C915D41AF72BE027984406714315A92D96B8D`，与 M55 已吸收的
正确附件逐字节一致。用户已确认此前介入的另一附件是复制错误；其内容、建议和所谓
可迁移护栏全部撤回，不属于当前课题知识。

最终态度不是“完全认可”，而是 `PARTIAL / REVISE`：接受标准 closed-set 全监督严格因果
On-TAD、标准最终区间输出、donor 只读、拒绝模块汤、start-owned identity/owner-conditioned
end 作为核心待证问题、关联/延时/资源指标和 feature→raw 证据边界；不接受固定槽位、
`K=32/B=128`、30k updates、seed 3407、固定 loss/effect gate/backbone、`end=t`、四 donor
parity 阻塞核心或单个未收敛种子自动杀死路线。

历史 `K×O` 只保留“因素必须可识别”的思想。当前正式设计以 DR-031 的 slot-free `B×O`
为准：比较延迟/即时 birth 与 fresh-rematch/sticky-owner，exact native MATR 为独立第五路；
capacity/learned retention 在 ownership 成立后另做固定总预算实验。EventMATR exact
`64d7f78dd8ed1436bac08ebfb03b51c90142129b` 已通过本地和 N16R4 合同测试，但官方真实
数据 smoke 与性能仍未完成，所以当前只能说“实现可复核”，不能说“模型有效”。完整处置见
`PRO_RAW_RGB_DYNAMIC_EVENT_MEMORY_REVIEW_ABSORPTION_20260722.md` 与 DR-034。
