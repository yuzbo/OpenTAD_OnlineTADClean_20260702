# Author Response Draft to CRS-EPS / Q2 Pro Review, Round 1

This is the recommended response to send to Pro. It is a protocol decision
draft, not an experimental result and not GPU authorization.

---

## BEGIN RESPONSE

感谢你严格按 Round 1 边界完成代码、定义和阻塞未知项审查。我们接受以下核心
裁决：

```text
RESEARCH_VERDICT=REVISE-BEFORE-IMPLEMENTATION
CURRENT_ROUTE=FULL_CHRONOLOGICAL_CACHED_VIDEO_EPISODE_TRUNCATED_BPTT
CRS_EPS_IMPLEMENTED=NO
Q2_IDENTIFIABILITY=PARTIALLY_IDENTIFIABLE
PROFILE_CONTRACT=REPLACE
FULL_PETAL_NOVELTY=RECONSTRUCTION_PROVISIONAL
RAW_VIDEO_STAGE2=BLOCKED
PROFILE=BLOCKED
FORMAL_TRAINING=BLOCKED
OPEN_P0=P0-LAUNCH-WORKDIR
```

我们也接受你的术语修正：当前 full-chronological route 只能作为当前
cached-token、video-uniform、per-video-mean、chunk-detached objective 的
exhaustive reference，不能无条件称为 unbiased full-sequence gold training。

以下逐项回答 Q1-Q12。请据此进入 Round 2；不要把任何回答解释为当前 GPU 许可。

## Q1. CRS-EPS 要逼近的 primary risk

我们冻结 primary target risk 为 **video-uniform per-video decision-time mean**：

```text
p(v) = 1 / V
p(t | v) = 1 / T_v

L_target(theta)
  = (1 / V) * sum_v [(1 / T_v) * sum_t L(v,t; theta)]
```

理由：这是当前 full-chronological cached reference 实际实现的目标，保持它可以避免
在引入 CRS-EPS 时同时改变 treatment 和目标风险。corpus-token-uniform objective
可以作为明确标记的 sensitivity analysis，但不是 Q2 primary objective。

请在 Round 2 给出在多 anchor 重叠下可计算的 inclusion probability、IPW/SNIPW、
weight clipping 和 ESS 公式，并确保该 proposal 对每个 target decision bin 都有非零
支持。

## Q2. full chronological route 在 hybrid 中的角色

我们选择：

```text
CRS-EPS = candidate main training protocol
tiny preregistered full-stream subset = gradient/state/loss gold audit
complete chronological validation/test = mandatory final evaluation
```

首个 Q2 primary protocol **不包含** periodic full-stream optimizer updates，也不包含
看结果后增加的 chronological finetuning。这样可以保持成本解释和 fixed/rematch
one-factor 清晰。

若 Round 2 认为短程 chronological calibration 必不可少，请把它放在“Q2 通过后才
允许的预注册独立 ablation”，不得混入 primary comparison。

## Q3. active-at-entry 的 gold reference

接受在 preregistered gold subset 上从视频开头完整 replay，以得到非 oracle 的
full-stream runtime/supervision state。192-token bounded burn-in 只是待验证 surrogate。

明确禁止：

```text
GT active-slot initialization
GT query initialization
GT hidden-state injection
future endpoint in model/runtime metadata
```

gold replay 必须比较 episode entry 的 query、memory、slot lifecycle、instance binding、
loss masks、loss values 和 gradient direction。

## Q4. start 早于 bounded burn-in 的处理

主规则不是全局排除，也不是用人工 left-censor 掩盖：

1. 默认 episode 使用不超过 192-token 的 burn-in。
2. 如果 supervised suffix 中有实例在 episode entry 已经 active，而其 prefix-observable
   birth 位于 192-token context 之前，则动态向前扩展 replay，至少覆盖所有相关实例
   中最早的 observable birth。
3. 必要时回退到 video-start replay。
4. 只有数据本身在视频边界发生真实左删失时，才使用显式 `true_left_censored`
   标记和对应 birth/start loss mask；人工 context 截断不得伪装成数据删失。
5. 不允许因长 context 直接让某些 target decision bins 变成零支持。

所有实际 replay tokens 必须进入成本统计。如果动态扩展普遍退化为完整视频 replay，
CRS-EPS 的成本价值应被判定失败，而不是隐藏该成本。

## Q5. primary cost endpoint 与预算

资源主指标冻结为：

```text
primary resource endpoint: total GPU-hours
co-primary engineering endpoint: wall-clock time
workload constraint: fixed weighted supervised bins / ESS
mandatory accounting: temporal forward/backward tokens, visual frames,
                      optimizer events, data wait, peak VRAM
```

`time-to-fixed-quality` 只有在有效性训练完成后才能作为科学效率结果，不能由无效果
profile 替代。

从现在到 raw-video Stage 2 之前的**新增总 GPU 上限为 10 GPU-hours**，包括：

```text
cache rebuild if provenance requires it: <= 1 GPU-hour
all fixed-step/multi-denominator profiles: <= 2 GPU-hours
full-stream gold gradient/state audit: <= 1 GPU-hour
Q2 fixed/rematch training + chronological evaluation: <= 6 GPU-hours
```

CPU B0 不计 GPU-hours；历史 cache/smoke 作为 sunk cost 单独披露。任何子项预计超限
都必须停止并重新审查，不得从其他子项静默借预算。Raw-video Stage 2 没有当前预算或
许可。

## Q6. optimizer-event 可比性

正式同意废除跨协议的“200 optimizer events 可比”假设。

`optimizer events` 只保留为辅助字段；50/200 profile 最多用于同一
full-chronological route 内 fixed/rematch 的 matched engineering comparison。

跨协议 profile 必须采用多分母报告，并比较达到固定 weighted supervision/ESS 所需
的 GPU-hours 和 wall time。请在 Round 2 给出替代 artifact schema 和 workload
matching 规则。

## Q7. primary quality metric 与 sampled/full estimand

我们冻结：

```text
primary external quality endpoint:
  standard temporal mAP averaged over tIoU 0.3:0.7,
  computed from complete chronological immutable emissions without offline NMS

Q2 mechanism endpoints:
  duplicate FP and fragmentation,
  including preregistered same-class repeated/concurrent and overlap subsets

safety endpoints:
  recall / unmatched FN,
  completion delay relative to matched GT end,
  late FP and calibration
```

CRS-EPS 对 full reference 的主要 quality estimand 是完整 chronological evaluation 上
的 paired difference in average temporal mAP，并要求 recall/delay 不发生实质退化。
在进入效果训练前，还必须通过 sampled/full loss、gradient direction、runtime-state
和 occupancy fidelity gate。

目前不冻结一个任意数值 non-inferiority margin。请在 Round 2 基于预结果的 gold
subset variability、paired design、minimum meaningful effect 和 power 给出方法；该
margin 必须在任何科学训练结果可见之前冻结。现有 `+2.0 points` / `20%` 只能作为
资源停止规则候选，不能冒充统计显著性标准。

## Q8. Full PETAL / Stage 2 kill commitment

接受 immediate kill：

- fixed binding 未在 matched design 中优于 rematch；或
- aggregate gain 没有落在 preregistered identity-linked errors；或
- recall、FN 或 completion delay 出现实质退化；或
- fixed/rematch difference 不能与 sampling/mask/normalization/RNG confound 分离。

发生任一情况，停止 Full PETAL 论文路线和 raw-video Stage 2。代码可保留为 baseline
或负结果基础设施，但不得通过增加模块继续挽救 headline claim。

## Q9. feature extractor provenance

当前不能确认旧 cache 的 exact extractor commit/config/checkpoint/processor/
normalization/raw support interval 可完整恢复，因此保持 `UNKNOWN`，不猜测。

处理规则：

1. P0 修复后，旧 cache 最多用于明确标记的 compute-only engineering profile。
2. 任何 effectiveness run 前必须恢复完整 provenance；若不能恢复，就重新构建 cache。
3. 新 schema 必须绑定 extractor repository commit、config hash、checkpoint SHA、
   processor/normalization、raw support interval、source frame 和 raw-future invariance。
4. provenance 未关闭时禁止 strict raw-video causality 或 low-latency claim。

## Q10. 211-versus-213 reporting population

当前两个差异视频及原因仍是 `UNKNOWN`。我们不会猜测或用历史口头解释替代证据。

在 formal reporting 前必须：

- 对 canonical 213 universe 与 locked 211 manifest 做 source-derived diff；
- 记录两个 video IDs、缺失/排除原因和可复现生成命令；
- 所有 baseline 在完全相同 population 上重新评测；
- 在关闭前只把结果命名为 `locked-211 subset`，不得称 canonical THUMOS14 validation。

该问题不阻塞 P0 修复或 compute-only profile，但阻塞正式结果解释与论文 reporting。

## Q11. serving/evaluation cadence 与 availability clock

训练 chunk size 与 serving/evaluation packet size 正式分离。

Primary formal evaluation 使用：

```text
one causal feature token per decision step
feature stride = 8 frames
nominal decision cadence = 8 / 30 s ~= 0.267 s
```

如果实现 micro-packet，仅可作为吞吐诊断；低延时主结果仍使用 one-token cadence。
每个 emission 必须记录：

```text
source_time
feature_availability_time
decision_start_time
decision_completion_time
emission_publication_time
```

raw extractor 的真实 cadence 和 token support interval 必须由 Q9 provenance 决定。
在端到端 raw-stream pipeline 实测前，不作 low-wall-clock-latency claim。

## Q12. 第二数据集

接受：首次 THUMOS Q2 只用于 mechanism kill，不要求第二数据集；任何保留的广泛
paper claim 必须增加一个 dense/overlap 第二数据集，并在相同 protocol 下重跑所有
关键 baselines。

FineAction 与 MultiTHUMOS 的访问、许可、annotation compatibility 和计算预算当前
尚未确认。请在 Round 2 比较两者并给出最低成本选择。若无法获得合格第二数据集，
最大允许结论只能是 THUMOS-specific mechanism observation，不能写成一般性的 Full
PETAL 或 persistent On-TAD 方法贡献。

## Round-2 请求

请基于以上决定继续 Prompt 的 Round 2，并完成：

1. fresh primary-source literature review；
2. full chronological / pure CRS-EPS / hybrid 三方案裁决；
3. 可执行的 target risk、union inclusion probability、weighting 和 ESS 公式；
4. active-at-entry、dynamic replay 和 true-left-censor 协议；
5. strict fixed/rematch paired trace；
6. multi-denominator profile contract；
7. gold-subset fidelity gates 与 power-aware non-inferiority 设计；
8. 最小 baseline/ablation 矩阵；
9. raw-video LoRA Stage-2 gate；
10. strongest rejection、claim map、file-level implementation plan 和最终裁决。

在 Round 2 结束、协议被作者确认、P0 修复、新 commit 的完整 B0 与独立
`PASS / PROFILE=ALLOW` 到位前，继续保持：

```text
PROFILE=BLOCKED
FORMAL_TRAINING=BLOCKED
RAW_VIDEO_STAGE2=BLOCKED
```

## AUTHOR DECISION BLOCK

```text
Q1_TARGET_RISK=VIDEO_UNIFORM_PER_VIDEO_DECISION_MEAN
Q2_HYBRID_ROLE=CRS_EPS_MAIN_PLUS_TINY_FULL_STREAM_GOLD_AUDIT_PLUS_FULL_CHRONOLOGICAL_EVAL
Q3_GOLD_STATE=REPLAY_FROM_VIDEO_START_NO_ORACLE_SLOT
Q4_ACTIVE_AT_ENTRY=DYNAMIC_REPLAY_TO_EARLIEST_OBSERVABLE_BIRTH_WITH_VIDEO_START_FALLBACK
Q5_PRIMARY_COST=GPU_HOURS_AND_WALL_TIME_AT_FIXED_WEIGHTED_SUPERVISION_ESS
Q5_TOTAL_FUTURE_GPU_CAP=10
Q6_CROSS_PROTOCOL_200_EVENTS_COMPARABLE=NO
Q7_PRIMARY_QUALITY=AVERAGE_STANDARD_TEMPORAL_MAP_TIOU_0.3_TO_0.7
Q7_MECHANISM_ENDPOINTS=DUPLICATE_FRAGMENTATION_SAME_CLASS_OVERLAP
Q8_KILL_ON_NO_FIXED_BINDING_IDENTITY_GAIN=YES
Q9_EXTRACTOR_PROVENANCE=UNKNOWN_REBUILD_IF_NOT_RECOVERABLE
Q10_REPORTING_211_VS_213=UNKNOWN_BLOCK_FORMAL_REPORTING
Q11_PRIMARY_SERVING_PACKET=ONE_TOKEN_WITH_SOURCE_AVAILABILITY_COMPLETION_CLOCKS
Q12_SECOND_DATASET=REQUIRED_IF_ROUTE_SURVIVES_FOR_GENERAL_CLAIM
PROFILE=BLOCKED
FORMAL_TRAINING=BLOCKED
RAW_VIDEO_STAGE2=BLOCKED
NEXT=PRO_ROUND2
```

## END RESPONSE
