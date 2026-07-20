---
id: exp:ontad-fixed-rematch-execution-20260720
type: experiment
status: active
updated: 2026-07-20
---

# FIXED/REMATCH Scientific-Contract Repair Execution

This page is the recovery checkpoint for the active implementation and
experiment task. Update it at every critical node before continuing.

## Frozen Objective

- Task: standard, fully supervised, strictly causal Online Temporal Action
  Detection.
- Input at this stage: cached causal video features, not raw RGB.
- Output contract: maintain instance birth, continuation, and end online, then
  emit one immutable final interval using only current and past evidence.
- Scientific comparison: FIXED versus REMATCH differs only in the post-birth
  target-to-slot loss binding.
- Raw RGB remains blocked until the feature-level technical and scientific
  gates pass.

## Recovery Checkpoint — 2026-07-20

- Branch: `codex/ontad-science-fixed-rematch`.
- Repository: `https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702`.
- Draft PR: `https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/pull/1`.
- Recovered HEAD: `9ce8aa0bfe4160350593c20349b2ff0ea8e39975`.
- Worktree was clean and matched
  `origin/codex/ontad-science-fixed-rematch`.
- Last engineering smoke: Slurm job `1176737`, passed at commit `097bc72`.
- Last strict paired profile: Slurm job `1176983`, technical/causal checks
  passed at commit `95fa963`; the registered 12-epoch pair was rejected at
  `12.572 GPU-hours` versus the frozen `2 GPU-hour` cap.
- No seed-705 training and no raw-RGB training are authorized by the current
  evidence.

## Active Repair Contract

1. Binary endpoint emission is the current causal decision frame; no
   unsupervised trainable endpoint offset.
2. A newborn remains on its canonical birth slot for the birth decision;
   REMATCH begins on the next causal decision.
3. Birth and end crossing in one step emits a short action exactly once.
4. Supervision exhaustion, runtime entry-free collision, arbitration,
   cancellation, abandonment, and released-slot deferral have separate
   counters; formal supervision exhaustion fails atomically.
5. Fit, calibration/checkpoint choice, and locked reporting are isolated.
6. Standard mAP uses a frozen percentage-point schema; instance matching uses
   video identity, one frame coordinate system, deterministic global
   one-to-one matching, and disjoint-component fragmentation.
7. A repository-owned artifact joins metrics, audits, resource use, code and
   input SHA-256 provenance.
8. A hashed split census is consumed before formal launch.

## Required Next Checkpoints

- [x] C1 — core lifecycle and supervision repairs implemented with focused
      regression tests.
- [x] C2 — metrics, gate, fit/report isolation, provenance, and census
      contracts implemented; all local CPU-safe checks pass.
- [x] C3 — repaired code committed and pushed; N16R4 Slurm end-to-end smoke
      passes from the exact commit.
- [x] C4 — strict paired profile rerun; budget-compatible protocol either
      passes the frozen cap or remains explicitly blocked.
- [ ] C5 — final decision and next paper-experiment route recorded; seed-705
      is submitted only if every preceding gate passes.

## Context-Safety Rule

Before moving past C1–C5, append the exact commit, commands/tests, Slurm job
IDs, artifact paths, hashes, pass/fail result, and scientific interpretation
to this page and `research-wiki/log.md`.

## C1/C2 Implementation Checkpoint — 2026-07-20

Core implementation is complete locally; C1 remains unchecked until the
PyTorch regressions run on N16R4:

- binary endpoint emission is exactly the decision frame and the binary route
  has no trainable endpoint-offset head;
- newborn REMATCH is prohibited on its birth decision;
- newborn birth+end commits once in the same step;
- active abandonment and released-slot birth deferral are separately counted;
- formal supervision exhaustion raises against a cloned supervision state.

C2 is complete locally:

- `online_instance_metrics.v2` separates `video_id` from
  `runtime_stream_key`, converts raw database seconds to frames, uses
  deterministic global maximum-cardinality/maximum-total-tIoU matching, and
  counts disjoint covered components for fragmentation;
- `persistent_binding_gate.v2` requires standard-mAP percentage points,
  frozen tIoUs, update/counter integrity, and code/input SHA-256 provenance;
- fit-only training cannot construct the reporting loader; unready execution
  is limited to an explicit smoke config; calibration and reporting roles
  require an explicit checkpoint;
- calibration candidate selection and final reporting both create one-shot
  receipts, with a retained lock on reporting failure or success;
- the repository-owned result builder joins standard mAP, budgeted OnlineAP,
  instance metrics, latency, training audit, resource use, census, receipts,
  and provenance;
- the launcher-consumed `persistent_binding_split_census.v1` checks the frozen
  160/40/211 split, birth/visibility capacity, endpoint coverage,
  old-end/new-birth bins, final-token short actions, delayed reuse, and clipped
  start supervision.

Local evidence:

- `python -m py_compile` passed for all modified Python entrypoints;
- 54 unique CPU-safe focused tests passed, including 16 supervision tests, 12
  instance-metric tests, seven result-gate tests, config/launcher tests, and
  four new isolation/census/one-shot tool tests;
- `git diff --check` passed;
- Windows PyTorch execution remains unavailable because of the previously
  recorded `c10.dll` initialization failure, so head/detector/gradient tests
  are queued inside the N16R4 Slurm smoke; four local Torch-dependent tests
  were skipped by their explicit environment probe.

## Census-Gated Smoke Attempt — Slurm 1177416

- Code commit: `71914470cf7cbabb429fb6360c552d98230665ac`.
- Run directory:
  `/data/run01/sczc063/yuzibo/runs/persistent_binding/smoke_20260720_221248`.
- Allocation: RTX 4090 on `g0003`; the in-allocation `nvidia-smi` preflight
  passed.
- Slurm result: `FAILED 2:0` after 14 seconds, before tests or training.
- Gate behavior: correct fail-closed census rejection; no training budget was
  consumed.

Frozen full-data census evidence:

- 411 videos, 320,205 causal feature tokens, and 6,328 action instances;
- all 6,328 births and endpoints are covered;
- maximum births per decision is 2 and maximum visible concurrency is 4,
  exactly within the registered budgets;
- zero GT entry-free deficits and zero oracle capacity overflow;
- 208 old-end/new-birth decisions (263 pairs), so the adjacent-action path is
  materially present in the real data;
- zero same-decision birth+end instances at stride 8;
- only failure: 820 clipped start targets versus a frozen limit of zero.

Diagnosis and correction:

- the detector decoded and stored start only at birth, but incorrectly applied
  start-offset regression again at every later active prefix;
- those 820 targets were therefore loss-only artifacts after the action had
  outlived the 192-token memory horizon, not runtime birth-start failures;
- start loss is now restricted to the shared canonical birth assignment, and
  the census audits only actual birth-start targets;
- this removes a nuisance post-birth loss from both arms and makes the
  FIXED/REMATCH comparison more tightly controlled.

The failed job is retained as negative evidence. A new commit and smoke job
must show zero clipped birth-start targets before C1 or C3 can pass.

## Repaired End-to-End Smoke — Slurm 1177438

C1 and C3 passed after the birth-only start-supervision correction.

- Exact code commit:
  `9036bd21838596543f87d705fe6daa06a49e0470`, pushed on
  `codex/ontad-science-fixed-rematch`.
- Run directory:
  `/data/run01/sczc063/yuzibo/runs/persistent_binding/smoke_20260720_221637`.
- Submission artifact: `job.sbatch` in that directory. It pins the commit,
  requires a clean checkout, runs the full census, the focused test bundle,
  direct real-feature FIXED and REMATCH train steps, one standard
  `tools/train.py --allow-unready-smoke` update, checkpoint reload inference,
  and `tools/verify_persistent_binding_smoke.py`.
- Allocation: one RTX 4090 on `g0003`; `nvidia-smi` passed inside the
  allocation.
- Slurm result: `COMPLETED 0:0` in `00:03:47`.
- Remote focused tests: `76 passed` in `100.40 s`.

Full frozen-data census:

- 411 videos, 320,205 causal feature tokens, and 6,328 instances;
- 6,328/6,328 births/endpoints covered;
- maximum two births per decision and maximum four visible instances;
- zero GT entry-free deficit and zero oracle capacity overflow;
- 208 old-end/new-birth decisions containing 263 pairs;
- zero same-decision birth+end instances at the registered stride;
- no census failures, including zero clipped *birth* start targets;
- artifact SHA-256:
  `10cd30f02a4f26ed1c8877e222c4437f9d91ce2d34d30be56c0c4a2a140b8184`.

Training and lifecycle evidence:

- both direct real-feature FIXED and REMATCH forward/backward/update steps
  passed with zero GT supervision exhaustion and zero runtime entry-free
  collision;
- the standard training path recorded exactly one expected update, one
  successful update, one scheduler step, and zero skipped updates;
- lifecycle totals were one arbitration suppression, five candidate
  cancellations, zero active abandonments, and 17 births deferred until a
  released slot became eligible;
- training-audit SHA-256:
  `7c2d4ac90a642479fef70d1decaa778348921557d54c6fcc8e5e8dfc92656c24`.

Checkpoint and strict-causal inference evidence:

- strict determinism was active with warning-only disabled, flash and
  memory-efficient SDP disabled, and math SDP enabled;
- 29 state tensors changed, 29 optimizer-state entries were created, and the
  total state delta L2 was `0.6226117403130047`;
- checkpoint SHA-256:
  `3a7ff1e952d1821c4a2bd2f9f03c41ab6893dc7f585e9b1ab1769b1c10609a6d`;
- train-time and reload inference ledgers were byte-identical, each with
  SHA-256
  `796a526159ab8f2ed94502b4b408f73a6c1042e161f4dced5950b9722b6ce2ce`;
- all 84 immutable final emissions carried separate video/runtime identities,
  ended and emitted on the current causal decision frame, and had zero future
  endpoint/source, negative-latency, or monotonicity violations;
- the joined smoke gate passed; its SHA-256 is
  `a27d91f046f89dfe9e183511dbfa05d968393a327f178b2680a38fb2928bdaa7`.

The zero AP values from one update on one held-out video are deliberately not
interpreted as effectiveness evidence. This smoke proves only that the
scientific contracts, gradients, checkpointing, strict-causal final emission,
reload determinism, and evaluator wiring work end to end. The next authorized
node is C4: rerun the paired resource profile at this exact repaired commit
before deciding whether any single-seed screening run is affordable.

## Repaired Strict Profile and Screen Registration — Slurm 1177511

C4 is complete. The repaired implementation remains technically stable, the
registered 12-epoch pair remains over budget, and a separate one-epoch
seed-705 *technical screen* fits the unchanged two-GPU-hour cap.

- Exact code commit:
  `caa42b682e53dbbc51b33c940e6c9a6222f91585`.
- Run directory:
  `/data/run01/sczc063/yuzibo/runs/persistent_binding/profile_20260720_222625`.
- Allocation: one RTX 4090 on `g0048`.
- Frozen measurement: seed 705; 50 warm-up and 200 measured chronological
  chunks per arm and mode; FP32; strict deterministic math SDP.
- Pre-profile tests: `19 passed`.
- Slurm state: `FAILED 1:0` in `00:10:12` only because the final frozen
  12-epoch budget evaluator deliberately exits nonzero when the cap fails.
  All four profile measurements completed and passed their stability,
  update, capacity, causality, and equivalence checks before that exit.

Measured training profile:

| Arm | Mean step (s) | Full fit pass (GPU h) | Peak MiB |
| --- | ---: | ---: | ---: |
| FIXED | 0.707804 | 0.395191 | 140.520 |
| REMATCH | 0.701788 | 0.391832 | 140.520 |

Both arms had zero GT supervision exhaustion and zero GT-birth/runtime
entry-free collision. Their measured lifecycle counts also agreed exactly.

Measured untrained calibration profile:

- FIXED and REMATCH had byte-identical canonical ledgers with SHA-256
  `dd75937a4096f03966bb49ab7e432cd4820f8b74d052906d9719d934bacf4e1a`;
- each measured prefix produced 17,014 emissions across 15 reached videos;
- all future-end, future-source, negative-latency, and non-monotonic-emission
  counts were zero;
- the large untrained emission count is a diagnostic baseline, not an
  effectiveness result.

Frozen 12-epoch budget result:

- training: `9.444269 GPU h`;
- calibration inference: `0.083274 GPU h`;
- reserved locked-report inference: `0.482775 GPU h`;
- raw total: `10.010318 GPU h`;
- total with the frozen 1.25 safety factor: `12.512897 GPU h`;
- gate: **FAIL**, versus the unchanged `2 GPU h` cap.

Registered seed-705 technical-screen budget:

- exactly one full epoch over all 160 fit-core videos for each arm;
- retain the exact first epoch of the 12-epoch optimizer/scheduler contract;
- evaluate only on the 40-video calibration split during this screen;
- do not access the 211-video locked reporting split;
- keep a conservative budget reserve for the eventual locked report anyway;
- estimated total with that reserve and the 1.25 factor:
  `1.691339 GPU h`;
- budget gate: **PASS**, versus `2 GPU h`.

The one-epoch run is authorized only as a convergence/non-degeneracy screen.
It may prove that both arms train, emit causally, avoid silent/explosive output
under the frozen technical thresholds, and produce complete calibration
artifacts. It cannot establish the 20% identity-error claim, three-seed
consistency, paper effectiveness, or permission for raw-RGB training.

Profile artifact SHA-256:

- FIXED train:
  `19fde8876177de4776ceb37e5c7feec49fe1aab3038a34aea76ae5b9a46132ef`;
- REMATCH train:
  `9cd2d3f22cfe58f35afe1c95583244279ea8fade1c32440ca59a1b47d0c70ed8`;
- FIXED calibration inference:
  `bcec50329929493ca75f5d6d0ea36e43d63ad07ea88315fc635490d553442812`;
- REMATCH calibration inference:
  `cbe71d216e43f05c0f8e1691155a3bbf11888e2c46712c5d7f7febeeb65b1278`;
- 12-epoch rejection gate:
  `680b3e8e347f7b65aac58e27ab306081adfcb7a182675b602cfea064d8cf303d`;
- one-epoch screen budget gate:
  `4f212c5ef823ea973283b962e7293c636093eaff9584373f96893c24dbfbc2f3`.

No new Pro discussion is needed before this screen: DR-028 already authorizes
registration of a budget-compatible one-seed screen after repaired smoke and
profiling. Before submission, the repository must add a screen-only config,
launcher, calibration-only result builder/gate, and the still-missing
adjacent-action end-to-end regression. C5 remains open until those checks and
the seed-705 screen finish.

## 模型优化与技术筛选实现节点 — commit 0258b85

本节点把工作重心从单纯“跑通流程”转到模型本身的可学习性。精确代码提交为
`0258b853aa284f9650d931e680116b63b289e192`。它仍然是固定特征级实验，
FIXED 与 REMATCH 仍只在出生后的监督绑定方式上不同；没有接入 raw RGB，
也没有读取 211 个 reporting 视频。

### 训练集事实

仅使用 160 个 fit-core 视频重新统计了监督分布：

- 123,940 个因果特征 token；
- birth：2,523 个正目标 / 458,670 个受监督目标，正率
  `0.0055006868`；
- alive：39,613 / 495,760，正率 `0.0799035824`；
- end：2,523 / 39,613，正率 `0.0636912125`；
- 2,523 个出生起点偏移全部位于一个特征 token 内，最大值约
  `0.998521`，中位数约 `0.507`。

这说明原始无先验普通 BCE 很容易先学成“全不出生”，而随机零偏置又会在
未训练推理时产生大量无意义区间；原先允许起点回归覆盖 192 个 token，也
远大于实际出生起点的可辨识范围。

### 共享模型改进

1. birth/alive/end 三个输出头用各自在 fit-core 上的正率初始化 bias；
2. 三个二分类损失使用温和的
   `sqrt(negative/positive)` 正样本权重，分别为
   `13.446021`、`3.393388`、`3.834156`，避免直接使用约 181 倍的
   极端 birth 权重；
3. scalar start 的合法范围从 192 token 收紧到 1 token，与全部真实出生
   目标一致；
4. REMATCH 的出生后匹配代价只使用有监督的 class 与 endpoint，不再让
   出生后未受监督的 start 输出左右身份匹配；
5. scalar start 路线不再创建或计算仅供 pointer start 使用的参数和
   相似度矩阵，减少无效参数与计算。

这些改动在两条实验臂完全共享，不改变严格因果推理、候选生命周期、最终
区间协议或 FIXED/REMATCH 的单变量比较。

### 新增验证与筛选闭环

- 新增相邻动作端到端回归：真实 `StreamingFeatureDataset` 输入经过
  detector、released-slot deferral、不可变 emission ledger、无未来审计、
  instance metrics 和 `OnlineAPBudgeted`；预期得到 `[0,2]` 与 `[2,4]`
  两个且仅两个区间；
- 新增一轮 seed-705、双臂各一 epoch 的 screen-only 配置和 Slurm
  启动/检查脚本；
- 筛选只评 40 个 calibration 视频，明确禁止 reporting 访问、论文效果
  结论和 raw-RGB 放行；
- 训练审计、校准收据、checkpoint、实际 emission ledger、census、
  smoke/profile gate 和资源报告必须用 SHA-256 串成同一证据链；
- 技术门禁继续拒绝静默输出、爆炸输出、漏更新、因果/容量错误和超过
  2 GPU-hour 的实际双臂资源消耗。

本地证据：相关 Python 文件编译通过，21 项 CPU-safe 配置/画像/census/
筛选测试通过，两个 Bash 启动脚本通过 `bash -n`，`git diff --check`
通过。Windows 本机仍因 PyTorch `c10.dll` 初始化故障无法执行 Torch
回归，因此上述相邻动作测试和全部 Torch 测试必须在 N16R4 Slurm 内完成。

旧 profile `1177511` 只能作为优化前预算上界，不能为新提交直接放行。
下一顺序固定为：同提交 smoke → 同提交 strict profile → 若一 epoch 仍在
2 GPU-hour 内，再提交 seed-705 FIXED/REMATCH 技术筛选。筛选通过也只说明
模型没有静默或爆炸且能开始收敛；C5、三种子论文主结果和 raw RGB 仍未完成。

## 模型优化后 Repaired Smoke — Slurm 1177580

模型优化提交的第一道远端门禁已通过。

- 实验提交：`534f85b38a61a5e7f02bef14d4968239059525b6`；
  其中模型实现提交为
  `0258b853aa284f9650d931e680116b63b289e192`；
- 运行目录：
  `/data/run01/sczc063/yuzibo/runs/persistent_binding/smoke_20260720_232522`；
- 单张 RTX 4090，节点 `g0013`；
- Slurm：`COMPLETED 0:0`，耗时 `00:06:04`；
- 完整远端 focused bundle：`82 passed in 75.46s`，新增的
  dataset→detector→ledger→instance metrics/OnlineAP 相邻动作回归通过。

完整 census 再次通过：411 个视频、320,205 个 token、6,328 个实例，
全部 birth/end 覆盖，最大同一步 birth 为 2、最大可见实例为 4，GT
entry-free deficit 为 0；全数据最大 birth start offset 为
`0.9999781689 token`，没有超过新冻结的 1-token 范围。

真实特征和标准训练路径均通过：

- FIXED 与 REMATCH 的直接前向/反向/更新各自通过；
- 标准 smoke 是 1 个 expected update、1 个 successful update、1 个
  scheduler step、0 skipped update；
- GT supervision exhaustion 与 GT-birth/runtime-entry collision 均为 0；
- 本次 lifecycle 记录 5 次 arbitration suppression、1 次 candidate
  cancellation、1 次 active abandonment、17 次 released-slot deferral；
- checkpoint/reload 各产生 67 个不可变最终区间，两个 ledger
  SHA-256 完全相同，且 future-end、future-source、negative-latency、
  non-monotonic 违规均为 0。

关键 SHA-256：

- census：
  `167833591c6d1c09a0e167cbfae4bb49987afd6e1dc41c2ee11769fe781bf770`；
- smoke gate：
  `e41757872aee4208181c8cb5ae67e4d4d376c89076fe280c623fdcfdcb0a3080`；
- training audit：
  `286212656045f7b517ac1339ddc80e6abeff3345d05da4a62597560bf0d8f52d`；
- checkpoint：
  `8e8cdc04b6e32181ca79615fdc845a7aad1b62592bb917845212d56179fdaefb`；
- train/reload ledger：
  `e21bc03a3a50cd4ae01767f36e00b3c9e7163309a27911000341f8d78c498359`。

该 smoke 证明优化后的模型、梯度、生命周期、相邻动作、序列化和严格因果
重载闭环有效，不证明方法效果。下一门禁是同提交 strict profile；profile
通过一轮 2 GPU-hour 上限后才运行 seed-705 技术筛选。

## 模型优化后 Strict Profile — Slurm 1177582

同提交第二道门禁完成；作业最终非零退出只代表冻结的 12-epoch 预算继续
超限，四个画像本身全部通过。

- 提交：`534f85b38a61a5e7f02bef14d4968239059525b6`；
- 目录：
  `/data/run01/sczc063/yuzibo/runs/persistent_binding/profile_20260720_233430`；
- RTX 4090 `g0013`，Slurm `FAILED 1:0`，`00:10:58`；
- 前置测试 `24 passed`；
- seed 705、FP32、严格确定性 math SDP、每臂/模式 50 warm-up +
  200 measured steps。

训练画像：

| Arm | Mean step | Full fit epoch | Peak MiB | 与优化前相比 |
| --- | ---: | ---: | ---: | ---: |
| FIXED | 0.674016 s | 0.376326 GPU h | 140.443 | -4.8% |
| REMATCH | 0.683778 s | 0.381776 GPU h | 140.443 | -2.6% |

两臂各有 29 个参数实际改变、28 个参数得到非零梯度，且 GT supervision
exhaustion、GT-birth/runtime collision、runtime capacity exhaustion 均为 0。

未训练 calibration 画像：

- FIXED `0.205306 s/step`、完整 split `0.026747 GPU h`；
- REMATCH `0.207096 s/step`，双臂 calibration 合计
  `0.053727 GPU h`；
- fit-only 低先验使两臂均产生 0 个无训练发射，取代优化前的 17,014 个
  测量前缀发射；
- 两臂 canonical ledger SHA-256 完全相同：
  `41539d028e27acc98117c09c4d78ae4c0c7f70fa2c2bcec9b65068ba06aa3309`；
- 所有 future-end/source、negative-latency、non-monotonic 计数为 0。

预算结论：

- 12 epoch：training `9.097223`、calibration `0.053727`、locked-report
  reserve `0.311478`，加 1.25 后 `11.828035 GPU h`，继续 **FAIL**；
- seed-705 一 epoch：training `0.758102`、calibration `0.053727`、
  reserve `0.311478`，加 1.25 后 `1.404134 GPU h`，在 2 小时内
  **PASS**；
- 新一轮估计比优化前注册值 `1.691339 GPU h` 下降约 17%。

Profile SHA-256：

- FIXED train：
  `562ebf1fe60b36e64ff1febce154b1ddcd80e157516ab9e2afbceca182b24664`；
- REMATCH train：
  `09630b8c4b97431014c4a2dcabaab3794640af56776feca83b907518c2591e83`；
- FIXED inference：
  `035f65b91614aa8e76f023c087643bbec919828b8ea83033095e9b014ae53066`；
- REMATCH inference：
  `8e6c31307d5445465f09a53c4f3eeb4b2b1fb467ab3cfcfbecd43b3f532ed57e`；
- 12-epoch rejection gate：
  `cb392fa41bb1a03bb80a156d094f89c243060bca6e4e74374d0c33d893079cdc`；
- one-epoch screen gate：
  `d6dbb8820f32809ced2d10901bb6f2cbb595caa2a181a4712b7b1028ac726f31`。

因此 seed-705 screen 的资源和技术前提均满足，可在不改代码、不改阈值、
不访问 reporting 的条件下提交。该放行仍不等于多 epoch 论文结果。

## 首轮 Seed-705 FIXED/REMATCH 技术筛选 — Slurm 1177596

首轮冻结筛选已完整执行，但技术门禁按预期拒绝了静默模型。这是有效负结果，
不是训练程序崩溃。

- 精确提交：`534f85b38a61a5e7f02bef14d4968239059525b6`；
- 运行目录：
  `/data/run01/sczc063/yuzibo/runs/persistent_binding/screen_seed705_20260720_234915`；
- RTX 4090 `g0024`，Slurm `FAILED 1:0`，耗时 `00:52:21`；
- 实际双臂资源为 `0.871944 GPU·hours`，低于冻结的
  `2 GPU·hours` 上限，故预算门禁通过；
- 前置测试、全量 census、FIXED 训练/校准、REMATCH 训练/校准和最终
  配对 gate 均执行到终点；非零退出来自最终技术 gate 的有意拒绝。

两臂训练数值稳定，且各自都有：

- `2010` expected updates；
- `2010` successful updates；
- `2010` scheduler steps；
- `0` skipped updates；
- `0` GT supervision exhaustion；
- `0` GT-birth/runtime-entry collision；
- `0` causal protocol violation。

但 calibration-only 最终结果完全静默：

| Arm | 最终区间 | prediction/GT | Recall@0.3 | average mAP |
| --- | ---: | ---: | ---: | ---: |
| FIXED | 0 | 0 | 0 | 0 |
| REMATCH | 0 | 0 | 0 | 0 |

因此两臂分别触发三个冻结失败项：无最终预测、prediction/GT 低于
`0.25`、Recall@0.3 低于 `0.25`。两臂显示的 duplicate rate 和
fragmentation rate 虽为 0，但分母对应的是零预测，不能解释为身份质量好。
reporting split 未访问，阈值未修改，论文效果结论和 raw RGB 均未放行。

预测驱动训练生命周期也支持“模型太静默”这一判断：FIXED 没有 birth
admission；REMATCH 虽出现 36 次 candidate cancellation 和 36 次
released-slot deferral，仍没有最终提交。对 checkpoint 的只读偏置检查为：

| Arm | birth bias / sigmoid | alive bias / sigmoid | end bias / sigmoid |
| --- | --- | --- | --- |
| FIXED | `-5.125290 / 0.005909` | `-2.412709 / 0.082209` | `-2.630291 / 0.067214` |
| REMATCH | `-5.124843 / 0.005912` | `-2.414131 / 0.082101` | `-2.636352 / 0.066835` |

fit-only 的原始 birth 先验 logit 为约 `-5.197366`。一轮 warm-up 后两臂
birth bias 只移动约 `+0.072`，远未接近 0.5 决策对应的 logit 0。
当前根因假设是：输出 bias 用未加权正率初始化，但 BCE 又使用大于 1 的
正样本权重，导致初始化没有位于实际加权目标的平衡点。这个假设必须先由
calibration-only 全分数分布诊断验证；不得通过降低冻结阈值来绕过。

关键 SHA-256：

- census：
  `a59dd5e4ce37f35be8062acdf614f4e45913354a0425c9ae70d15104a3fe3bea`；
- FIXED checkpoint：
  `da2a9c5621b80671377f07961840a91bff4fc8fb5b565f0f63118cb1cacae61a`；
- REMATCH checkpoint：
  `b9aaad071285c162978d557fa11717398d066f3a02ce91157f015f28017d5332`；
- 两臂空 emission ledger：
  `b6420469538bb0c48d595801634d5b1a2da569d5c7926e9ff57a4d62f2184c25`；
- FIXED screen result：
  `97f6bd39b4d694e4240b4dec84daf50f637a435643389a8da186c6e5a45a8fb1`；
- REMATCH screen result：
  `e03a8d57aab3c679dfefdeb30e66e43db70e944e6dd66ce884ace522ff93bc3e`；
- pair resource：
  `6bcbed5ac6cfd6ec5b8da1e9fe06dc210f5959157ef0aa0b125a936e8c353e96`；
- screen gate：
  `b96e13ef599f0584aa0c107e83464c3c785760e92f82d7e5a6bbfbc49c06c57b`。

下一节点固定为：对这两个 checkpoint 运行只读的 calibration-only
birth/alive/end 分数分布诊断；若证实初始化失配，则在两臂共享模型中实现
加权 BCE 一致的先验 bias 初始化，重新走 smoke → profile → seed-705
筛选。多种子主实验与 raw RGB 继续等待修复后的技术门禁。

## Calibration-only 分数诊断 — Slurm 1177634

只读分数诊断已在失败筛选的两个原始 checkpoint 上完成，并确认 birth
是共同的生命周期瓶颈。

- 分析提交：
  `5d3daecce03d6819638f0619b8fe51798f7c69de`；
- checkpoint 训练提交：
  `534f85b38a61a5e7f02bef14d4968239059525b6`；
- 运行目录：
  `/data/run01/sczc063/yuzibo/runs/persistent_binding/score_diagnosis_20260721_004912`；
- RTX 4090 `g0017`，Slurm `COMPLETED 0:0`，耗时 `00:05:50`；
- 22 项远端 Torch 前置测试全部通过；
- 两臂都只扫描同一批 40 个冻结 calibration 视频、28,730 个 token，
  reporting 未访问，阈值未修改，raw prediction/EMA/AMP 均未启用。

全 slot 分数结果：

| Arm / channel | Mean | P50 | P95 | P99 | Max | `>=0.5` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FIXED birth | 0.141209 | 0.201967 | 0.246720 | 0.281521 | 0.433135 | 0 |
| FIXED alive | 0.296946 | 0.411529 | 0.459273 | 0.478746 | 0.508811 | 23 |
| FIXED end | 0.300550 | 0.278174 | 0.411108 | 0.443589 | 0.661820 | 271 |
| REMATCH birth | 0.100255 | 0.123584 | 0.218989 | 0.276584 | 0.346657 | 0 |
| REMATCH alive | 0.232148 | 0.314106 | 0.387726 | 0.421811 | 0.567618 | 83 |
| REMATCH end | 0.175814 | 0.178640 | 0.237733 | 0.300635 | 0.414860 | 0 |

两臂 birth 均没有一次达到冻结的 0.5 门槛，因此没有 birth proposal、
admission 或最终 emission。FIXED 的 alive/end 已能过线，REMATCH 的 alive
也能过线；这排除了“所有头共同数值失效”，并把首要故障定位到 birth 门。

fit-only birth 原始先验 logit 为 `-5.197366`，但在
`13.446021×` positive weight 下，加权 BCE 对常数预测的平衡 logit 应为
`-2.598683`，对应概率 `0.069223`。一轮训练后：

- FIXED birth bias 为 `-5.125290`，距加权平衡点 `-2.526607`；
- REMATCH birth bias 为 `-5.124843`，距加权平衡点 `-2.526159`。

alive 和 end 也存在同类但较小的偏差。因而“损失按加权目标优化、输出头却按
未加权概率起步”的失配假设被两臂共同证实。

诊断 SHA-256：

- FIXED：
  `794145067f8a618342fc3c70cbfebebdb0bf6b93838549d4519ac2364b9b30db`；
- REMATCH：
  `ce65cfc810537f095b10f797c5d8b70b66e9fc8dadec0d6d7b2a607b34ca19f0`；
- hash 清单：
  `ffa8f5b2111fcf3f443146d77ff9fbd2beeb8a6069414a74451b9dbff3feebbb`。

## 共享模型单变量修复

根据上述证据，实现一个显式的 `prior_bias_mode`：

- 正式 FIXED/REMATCH 特征路线使用 `weighted_bce_stationary`；
- 三个二元头的初始 logit 统一为
  `logit(fit_positive_rate) + log(positive_weight)`；
- 经验先验、平方根正样本权重、0.5 阈值、优化器、scheduler、数据划分、
  lifecycle 和 FIXED/REMATCH 唯一比较轴全部不变；
- serialization smoke 显式保留 `raw_probability` 和 0.5 priors，避免把
  技术序列化用的高先验与正式模型初始化混为一谈；
- census、profile、training audit 和 screen gate 都记录或检查该模式。

该修改仍只是有根因证据支持的可学习性修复，不构成效果结论。下一步必须在
新精确提交上依次重跑 Slurm smoke、strict profile 和 seed-705 双臂筛选；
只有新筛选同时避免静默与爆炸，才可讨论多种子特征级主实验。

## Weighted-prior Repaired Smoke — Slurm 1177637

共享偏置修复后的同提交整链路 smoke 已通过。

- 精确提交：
  `8dff64c958ebbad2ce0dad285b74007536c7f5ab`；
- 运行目录：
  `/data/run01/sczc063/yuzibo/runs/persistent_binding/smoke_20260721_010459`；
- RTX 4090 `g0006`，Slurm `COMPLETED 0:0`，耗时 `00:06:11`；
- 87 项远端 Torch 测试通过；
- 全 411 视频、320,205 token、6,328 实例 census 再次通过。

Census 明确记录正式路线使用 `weighted_bce_stationary`，解析初始概率为
birth `0.069223`、alive `0.227615`、end `0.206861`。FIXED/REMATCH
真实特征直接更新均通过，且两臂都是 0 GT supervision exhaustion、
0 GT-birth/runtime collision 和 0 runtime capacity exhaustion。

标准 serialization smoke 仍显式使用 `raw_probability` 和 0.5 priors：
它完成 1/1 optimizer update、1 scheduler step、0 skip；checkpoint 与
严格因果重载各发射 67 个不可变最终区间，两个 ledger 逐字节同哈希，且
future-end/source、negative-latency、non-monotonic 违规全为 0。

关键 SHA-256：

- census：
  `103810a4df69a7fc876fbccd84b4ca28454e5dd978cdbcad20b32de1ed170d1b`；
- smoke gate：
  `b4849650fe4caa6001ec50fa3a0a7d65b335feb671cd55356fdfa6c619bb0efc`；
- training audit：
  `a665075ae4f786d45d4a17d6cc4bdec78bfb38b7caa9277e59f03e35cbbd7432`；
- checkpoint：
  `8e8cdc04b6e32181ca79615fdc845a7aad1b62592bb917845212d56179fdaefb`；
- train/reload ledger：
  `e21bc03a3a50cd4ae01767f36e00b3c9e7163309a27911000341f8d78c498359`。

该节点证明修复没有破坏模型构建、梯度、容量、序列化和严格因果协议；
它不证明一轮训练后能避免静默或爆炸。下一门禁是在同一提交上重新画像。

## Weighted-prior Strict Profile — Slurm 1177639

同提交画像四项全部完成；Slurm 最终 `FAILED 1:0` 只代表冻结的 12-epoch
方案继续超出 2 GPU-hour cap。

- 提交：`8dff64c958ebbad2ce0dad285b74007536c7f5ab`；
- 目录：
  `/data/run01/sczc063/yuzibo/runs/persistent_binding/profile_20260721_011216`；
- RTX 4090 `g0006`，耗时 `00:11:27`；
- 27 项前置测试通过；
- seed 705、FP32、严格确定性、每个画像 50 warm-up + 200 measured。

训练画像：

| Arm | Mean step | Full fit epoch | Peak MiB |
| --- | ---: | ---: | ---: |
| FIXED | 0.689554 s | 0.385001 GPU h | 140.443 |
| REMATCH | 0.692710 s | 0.386763 GPU h | 140.443 |

两臂均有 29 个参数改变、28 个参数获得非零梯度，且 GT supervision
exhaustion、GT-birth/runtime collision、runtime capacity exhaustion 均为 0。

未训练 calibration 画像仍没有发射，说明加权平衡初始化本身没有造成爆量：

- FIXED `0.204952 s/step`、完整 split `0.026701 GPU h`；
- REMATCH `0.205963 s/step`；
- 两臂 0 emission、同一 canonical ledger SHA-256
  `41539d028e27acc98117c09c4d78ae4c0c7f70fa2c2bcec9b65068ba06aa3309`；
- 所有未来信息和时序违规为 0。

预算结论：

- 12 epoch：training `9.261172`、calibration `0.053533`、
  locked-report reserve `0.310355`，加 1.25 后
  `12.031325 GPU h`，**FAIL**；
- seed-705 一 epoch：training `0.771764`、calibration `0.053533`、
  reserve `0.310355`，加 1.25 后 `1.419566 GPU h`，**PASS**。

Profile SHA-256：

- FIXED train：
  `f8593149b2eea5eb678a725b4cf6b0df91d78c4fca1afc3c5a8d2d5ec4096060`；
- REMATCH train：
  `8b4de6f89a05b27b7a21e238c0b2e0e29dcf2c9f73ef8f5ca296faeb9c9f18f7`；
- FIXED inference：
  `864e06421771ebffee5196091ee90dda2f4bde68de3fa487b266aa7eaa6b2aa9`；
- REMATCH inference：
  `930e65e6473271165214ca0aea4bc684a9706e48b645510b465684f3973e4ee8`；
- 12-epoch rejection gate：
  `0ac674c7e00f66c285d1fe75a6ca7cc16e8844102a12cc980066f9f8f36bab7a`；
- one-epoch screen gate：
  `375d907289b3dc3730ff53514dda77ade55fa491dd86f28720e99f9609f736df`。

因此同提交 seed-705 双臂筛选已获资源放行；该放行不改变技术门槛，也不
授权 reporting、效果结论、多种子或 raw RGB。

## Weighted-prior Seed-705 Screen — Slurm 1177653

加权 BCE 一致初始化后的双臂 seed-705 技术筛选已在同一精确提交上完整
结束。Slurm 最终 `FAILED 1:0` 是冻结技术 gate 的预期拒绝，不是训练、
checkpoint、推理或评测程序崩溃。

- 提交：`8dff64c958ebbad2ce0dad285b74007536c7f5ab`；
- 目录：
  `/data/run01/sczc063/yuzibo/runs/persistent_binding/screen_seed705_20260721_012521`；
- RTX 4090 `g0006`，Slurm `1177653`，耗时 `00:57:23`；
- 双臂实际资源 `0.955278 GPU·hours`，低于冻结的
  `2 GPU·hours` 上限；
- 前置测试、全量 census、FIXED 训练/校准、REMATCH 训练/校准及最终
  paired gate 全部执行到终点；
- reporting split 未访问，阈值、数据、损失权重、优化器、生命周期和
  FIXED/REMATCH 比较轴均未在看结果后修改。

两臂各自都完成：

- `2010/2010` successful/expected updates；
- `2010` scheduler steps、`0` skipped updates；
- `0` GT supervision exhaustion；
- `0` GT-birth/runtime-entry collision；
- `0` causal protocol violation。

但 calibration-only 最终结果仍共同静默：

| Arm | 最终区间 | prediction/GT | Recall@0.3 | average mAP |
| --- | ---: | ---: | ---: | ---: |
| FIXED | 0 | 0 | 0 | 0 |
| REMATCH | 0 | 0 | 0 | 0 |

因此冻结 gate 对每臂均触发“无最终预测、prediction/GT 低于 `0.25`、
Recall@0.3 低于 `0.25`”三项失败。该结果证明：

1. weighted-prior 修复没有导致预测爆量，也没有破坏训练或因果协议；
2. 单独修复初始化不足以在当前一轮全 warm-up 日程内解除 birth/lifecycle
   静默；
3. 两臂共同失败且训练轨迹接近，当前首要瓶颈位于共享模型/优化过程，
   而不是 FIXED/REMATCH 绑定差异；
4. 不能放行多种子、reporting、论文效果结论或 raw RGB。

关键 SHA-256：

- census：
  `d3da34ca23c35794b954db78ad401370812c8721c96eed5bbc95392b963553ef`；
- FIXED checkpoint：
  `5cd789c9035ac79a079b33670d46ab1ffafaaa1c6093e6a5b95f203f3d7ae254`；
- REMATCH checkpoint：
  `0f74595ebdb1bde686c66a4df5c567f8c2bbab7e37933fa87a7cb87318537832`；
- FIXED screen result：
  `087a24c2cf308d5083b00e6313005e0d1651b0f2319a0aab8ed270ba4e6277b1`；
- REMATCH screen result：
  `887ee6962591b04a59c49daecfc1082dfbc35fd6c1f57da204a573625a8f328a`；
- pair resource：
  `a09e2b9b5debfcf6e42b2e0baa64c4941af6819d31d4d76da9e9bb2b801eb2dd`；
- screen gate：
  `dc3fd87baf115722d8ddbf9214a9de09974b32363f908b883db8b68bbdbadb50`。

下一节点固定为只读 calibration score diagnosis
`1177682`（目录
`/data/run01/sczc063/yuzibo/runs/persistent_binding/score_diagnosis_20260721_022330`）。
诊断先量化新 checkpoint 的 birth/alive/end 分布、0.5 margin、过线次数及
bias 位移。只有诊断证据支持，才实施一个两臂共享的单变量优化；不得降低
阈值或访问 reporting。

## Shared Short-warmup Candidate — Local Only

GPU score diagnosis `1177682` 因账户 `GrpTRES=gres/gpu=16` 已被其他作业
占满而暂时排队。为不停止模型侧推进，先完成一个**未提交训练**的本地
共享优化候选；GPU 分布诊断仍是其 Slurm 放行门禁。

已有筛选产物给出两条独立证据：

1. 轻量只读 checkpoint 检查显示，一轮训练后 FIXED/REMATCH 的 birth
   bias 分别为 `-2.586621/-2.579974`，相对加权平衡初始值约
   `-2.598683` 仅移动 `+0.012/+0.019 logit`；alive/end bias 同样只作
   小幅移动；
2. 当前 `warmup_epoch=1.0` 使全部 2,010 次更新都处于线性 warm-up，
   平均 LR 为 `1.000995×10^-4`。若仅将 warm-up 压缩为 0.1 epoch，
   同样 2,010 次更新、同样峰值 `2×10^-4` 下，平均和累计 LR 都变为
   原来的 `1.890663×`。

候选修改严格限定为：

- 两臂共享 `warmup_epoch: 1.0 → 0.1`；
- 峰值 LR 仍为 `2×10^-4`，`max_epoch=12` 不变；
- 模型结构、先验、positive weights、loss weights、阈值、数据划分、
  update 数、lifecycle 和 FIXED/REMATCH 比较轴全部不变；
- config 中新增显式 `optimization_contract` 和 screen provenance 字段；
- 15 项 CPU-safe 配置、screen 与科学合约测试通过，Python 编译及
  `git diff --check` 通过。

该节点只代表候选实现就绪，不代表远端 Torch、稳定性、预算或技术门禁
通过。执行顺序仍为：GPU score diagnosis → 候选同提交 smoke →
strict profile → seed-705 screen。诊断若显示静默并非训练剂量问题，
则不得提交该候选训练。

### Checkpoint Parameter-delta Correction

随后在登录节点只重建 seed-705 初始参数并读取 checkpoint，不执行视频
前向或训练。相对同一确定性初始化：

- FIXED/REMATCH 的 birth-head **weight** 变化分别为初始 weight norm 的
  `19.17%/18.80%`，并非没有更新；
- alive-head weight 变化为 `20.51%/21.29%`，end-head weight 为
  `20.13%/16.14%`；
- 共享主干整体参数变化为其初始 norm 的 `12.61%/12.21%`，class head
  整体为 `47.50%/48.70%`；
- birth bias 仍仅移动 `0.012/0.019 logit`。

因此，bias 位移很小不能单独证明欠训练：该 bias 本来就位于加权 BCE
的近似平衡点。short-warmup 候选的准确动机应是：旧的一轮筛选直到最后
一次更新才到达峰值 LR，且最终模型仍无 birth crossing；它是否能改善
正负分离仍须由 `1177682` 的真实 calibration 分数决定。该纠偏不改变
候选代码轴，也不放行训练。

### Optimizer-update LR Semantics

训练循环的真实顺序是“读取当前 LR → optimizer update → scheduler
step”。据此按调度器源码重新计算 2,010 次实际 update 使用的 LR：

- 旧 `warmup_epoch=1.0`：累计 `0.201000`，平均 `1.0e-4`；
- 候选 `warmup_epoch=0.1`：累计 `0.380204214288`，平均
  `1.891563255×10^-4`；
- 新/旧累计曝光比为 `1.891563255×`，峰值仍为 `2e-4`，第 201 次
  update 首次使用峰值。

此前 `1.890663×` 的手工值使用了 scheduler step 之后的 LR，现已纠正。
新增远端 Torch 测试直接实例化正式 scheduler，逐次记录 optimizer
真正使用的 LR。登录节点上的 Torch 2.0.1 轻量单元检查（单个标量、
无模型/视频前向、无 GPU）得到 `warmup_steps=201.0`、第 201 次 update
LR `2e-4`、累计值 `0.3802042142882046`，与推导完全一致。该检查只验证
scheduler 语义；候选配置、真实前反向、checkpoint 和因果重载仍须由
下一次同提交 Slurm smoke 权威验证。
