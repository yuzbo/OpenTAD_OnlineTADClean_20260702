# Pro Review Absorption: 2026-07-10

Source record:

- Full external review: `PRO_REVIEW_20260710.md`
- Original source: private Codex attachment, not committed; integrity is tracked by the SHA256 below.
- Attachment SHA256: `C51D0FCCD9EE3F4F07AF1CA175384471548D9F1AFBF3FD13917CA245B3253FFB`
- Attachment line count: 910
- Reviewed public branch according to reviewer: `codex/online-tad-clean-20260702`
- Reviewed visible HEAD according to reviewer: `4e222ccf4de1e989c6faf83031772c552be7af59`
- Local absorption date: 2026-07-10
- Review verdict on current repo: `NO-GO`
- Review verdict on refactored research route: `HOLD`

## Absorbed Verdict

This review supersedes casual optimism about the current `final` route.

Current repo wording must be downgraded from any form of:

> low-latency, adaptive, end-to-end, MATR-style Online TAD.

to the reviewer's more precise label:

> Non-overlapping chunk-end causal TAD with fixed-stride selected-only visual encoding.

In Chinese working notes, this means:

> 非重叠长窗口、窗末统一发射、固定因果步长选帧的 causalized TAD skeleton.

Allowed current wording:

- raw-frame input route exists;
- selected-only heavy visual encoding exists;
- final selector is fixed causal stride, not adaptive;
- SigLIP2 vision tower is frozen;
- temporal projection is causalized;
- streaming-safe branch avoids the ordinary offline `batched_nms`;
- ledger can check declared `end_frame/source_frame <= emit_frame`;
- raw-prediction shortcut is disabled in the relevant config.

Blocked current wording:

- learned or adaptive frame selection;
- low-latency Online TAD;
- snippet-level rolling inference;
- complete prefix-observed censored training;
- bounded-delay emission learning;
- full MATR reproduction or MATR-style memory;
- visual backbone fine-tuning;
- full-chain no-future proof;
- DDP-auditable online evaluation;
- end-to-end compute reduction;
- paper-ready or SOTA On-TAD.

## New Hard Definition of Strict On-TAD

The review freezes a stricter On-TAD contract that should be treated as project law:

1. Prefix causality: identical prefixes must yield identical states and emitted detections before that time.
2. Real read provenance: every detection needs an actual maximum raw-frame source index `r_i`, and `r_i <= emit_time`.
3. Confirmed-detection constraint: if this is completed action detection, predicted end must not be later than emit time.
4. Immutable output: emitted detections cannot be modified, deleted, or replaced later.
5. Bounded latency: latency must be measured against matched GT event end, not predicted end.
6. Fully online post-processing: no EOF-time whole-video NMS, re-scoring, or boundary merging.

The important conceptual fix is:

> Window length, context horizon, decision cadence, and latency budget are different variables.

Current code ties context horizon and decision cadence together at roughly 51.2 seconds, which kills any low-latency claim.

## Current Latency Reclassification

The review derives current scheduling delay directly from config:

- `fps = 30`
- `snippet_stride_frames = 8`
- `window_size = 192`
- `window_overlap_ratio = 0.0`
- window span = `192 * 8 / 30 = 51.2s`

Absorbed conclusion:

- worst scheduling wait can approach `51.2s`;
- under uniform-in-window endpoints, mean scheduling wait is about `25.6s`;
- p95 scheduling wait is about `48.64s`;
- this is enough to reject `low-latency On-TAD`.

`max_latency_frames = 192 * 8` must not be described as low-latency. It only allows a proposal end to fall inside the current long window.

## Current Leakage and Audit Gaps

The review confirms some shortcuts are not present:

- `load_from_raw_predictions=False`;
- raw-frame loading path exists;
- streaming-safe branch does not use ordinary offline `batched_nms`.

But those are insufficient. Remaining gaps:

1. Declared ledger source is not true read lineage.
2. `duration` and `total_frames` may leak whole-video metadata in live-stream framing.
3. There is no counterfactual future-perturbation proof.
4. DDP online state is not closed and should remain single-rank by contract.
5. `source_frame <= emit_frame` is necessary but not sufficient to prove no-future computation.

## Training Scope Absorption

The review makes a sharper point than the previous absorption:

> Raw-frame input is not the same as raw-frame end-to-end trainable.

Current safe statement:

- raw frames enter the system;
- selected frames enter the heavy visual encoder;
- SigLIP2 vision tower remains frozen;
- adapter/head updates are not proven until optimizer groups, gradient norms, and parameter deltas are dumped.

Extra blocker:

`VideoMambaSuite.get_optim_groups()` may skip parameters whose names start with `backbone`, which can accidentally exclude claimed trainable adapters or motion branches. Therefore PEFT claims need a runtime optimizer audit.

## Prior Work Boundary

The review's literature positioning is absorbed as follows:

- OAT already covers sliding-window On-TAL and online suppression.
- MATR already covers current-end plus past-start retrieval with memory.
- PKD already covers future-privileged teacher to prefix-only student.
- OnPoint already covers TAL-level offline-to-online distillation.
- WACV 2026 streaming distillation already covers offline ViT teacher to causal cached student.
- AdaTAD already covers raw-frame adapter tuning for offline TAD.

Therefore:

> Online TAD + causal backbone + distillation + adapter is not a contribution.

The remaining defensible gap is:

> Model physical event time, prefix-censored state, and system emission time as distinct variables.

This fits the earlier 2026-07-09 absorption and further sharpens it.

## Recommended Main Route

Adopt the reviewer's single recommended route:

> PCEH-OnTAD: Prefix-Censored Event-Emission Hazard Learning for Online Temporal Action Detection.

Core contribution:

> Separate physical action endpoint `E` and model emission time `A`, then learn right-censored endpoint distribution and bounded-delay first-emission policy under prefix-only observation.

This must be the only first-paper contribution. Do not simultaneously claim teacher distillation, learned selector, pretraining, Mamba, causal attention, and adapter tuning as equal contributions.

Route decisions absorbed:

| Route | Verdict | Reason |
|---|---|---|
| Route 1: Prefix-Observed Event-Time Learning | Main route | Best balance of novelty, falsifiability, feasibility, and compute control |
| Route 2: OnlineTAD-PT pretraining | Future extension | Potentially stronger but too expensive and too broad for the first closed loop |
| Route 3: Offline-teacher cached student | Baseline/support only | Too close to PKD, OnPoint, and streaming distillation |

## Method Contract for PCEH

### Input

- raw-frame packets;
- suggested packet size: 8 frames;
- update cadence at 30 fps: about 0.267s;
- no pre-extracted features for the main result.

### Output

Each confirmed detection must include:

- predicted start event frame;
- predicted end event frame;
- class;
- score;
- emit frame;
- actual maximum raw frame read;
- actual maximum cache source frame.

### State Machine

Use explicit states:

1. background;
2. ongoing / right-censored;
3. endpoint observed;
4. completed but not emitted;
5. emitted absorbing state.

The emitted state must be absorbing.

### Losses

The absorbed first-version loss family:

- prefix-valid classification;
- start retrieval/localization;
- censored endpoint survival loss;
- completion state loss;
- bounded-delay first-emission loss;
- calibration loss.

Do not define `emit_target = end_mask`. Do not treat endpoint BCE as emission learning.

## Training Protocol To Implement Later

The future implementation plan must obey:

- batch lanes preserve same video chronological packet order;
- state persists within a lane and resets only at video boundary;
- truncated BPTT may detach gradients but must not clear stream state;
- prefix target builder must not inject future endpoint values into pre-end targets;
- optimizer dump must list trainable names, param groups, grad norm, and parameter deltas.

Mandatory target tests:

- `test_future_endpoint_does_not_change_prefix_targets`
- `test_no_end_regression_before_endpoint`
- `test_emit_forbidden_before_endpoint`
- `test_completion_is_monotonic`

## Inference Protocol To Implement Later

Each packet step must:

1. record true packet source-frame range;
2. encode only new frames;
3. update causal visual/temporal cache;
4. update ongoing hypotheses;
5. compute end hazard, completion, and emit hazard;
6. emit the first eligible immutable detection;
7. write ledger;
8. perform only past-only duplicate suppression.

Forbidden:

- EOF whole-video NMS;
- revision of emitted boundaries;
- whole-video duration for intermediate live outputs;
- test-time teacher, GT, feature-cache shortcut;
- source provenance derived from predicted endpoint;
- future frame refresh of past cache;
- DDP streaming claims before state sharding is solved.

## Evaluation Contract

The review rejects deleting late rows before AP.

Absorbed metric rule:

> A prediction is TP only if class matches, tIoU passes, and `gt_end <= emit_frame <= gt_end + B`.

If spatial/temporal overlap is good but emission is late:

- prediction is a late FP;
- matched GT is an on-time miss.

Latency in the paper must mean:

> `emit_frame - matched_gt_end`

not:

> `emit_frame - predicted_end`.

Keep both fields in the ledger but only the GT-matched one is detection latency.

## Required Baselines

First serious experimental loop must include at least:

- current chunk-end config;
- rolling fixed causal stride;
- rolling dense frozen encoder;
- rolling dense PEFT encoder;
- simple sliding-window anchor-free On-TAL;
- OAT-style baseline;
- MATR-style feature baseline;
- offline TAD teacher upper bound;
- vanilla KD / PKD-style baseline;
- random/uniform/motion same-budget selectors if any selector claim remains;
- offline AdaTAD PEFT upper bound if adapter adaptation is discussed.

Fairness controls:

- same raw-frame resolution;
- same visual backbone;
- same update cadence;
- same visible prefix;
- same encoded-frame budget where applicable;
- same candidate quota and validation-only calibration.

## Required Ablations

Minimum ablations before paper claims:

- remove censored survival and use endpoint masking;
- remove emission hazard and use fixed threshold or chunk-end emit;
- set emit head equal to end head;
- remove completion state;
- freeze versus PEFT;
- cache length sweep: 16/32/64/192;
- cadence sweep: 1/2/4/8 snippets;
- latency budget sweep: 0.5/1/2/4s;
- no-teacher versus vanilla KD if teacher is used;
- THUMOS to MUSES transfer;
- provenance audit on/off;
- learned selector versus uniform same budget if selector is used.

## GO/HOLD/NO-GO Gates

Current repo is `NO-GO` for final-method training or paper claims.

PCEH route is `HOLD` until all gates pass:

1. Cadence gate: update and emission check at most every 8 frames, not chunk-end.
2. Causality gate: future perturbation zero violation.
3. Target gate: prefix target invariance tests pass.
4. Optimizer gate: all claimed PEFT parameters update.
5. Metric gate: OnlineAP@B without deleting late rows.
6. Latency gate: primary 1s budget with p95 reported against GT end.
7. Baseline gate: fair improvement over rolling fixed-stride and OAT/MATR-style baselines or clear Pareto gain.
8. Dataset gate: THUMOS14 and MUSES with at least 3 seeds.
9. Efficiency gate: real end-to-end throughput/latency gains including decode/I/O.
10. Novelty gate: censor and emission hazard have stable independent ablation gains.

Only then may the project move from `HOLD` to `GO`.

## Immediate Working Rules

- Stop calling fixed `causal_stride` adaptive.
- Stop describing 51.2s `max_latency_frames` as low latency.
- Stop describing `MATRHead(memory_size=0)` as MATR implementation.
- Stop calling endpoint-local BCE emission learning.
- Stop using latency relative to predicted end as the primary latency metric.
- Stop expensive final full training until streaming protocol and evaluator are repaired.
- Stop mixing teacher, learned selector, and pretraining into the first main method.

## Implementation Checklist Absorbed

The next implementation sequence, if the user approves coding later:

1. Gate 0: clean claim metadata and rename policy to `fixed_causal_stride2`.
2. Gate 1: build `StreamingRawFrameDataset` and `ChronologicalStreamBatchSampler`.
3. Gate 2: add incremental `forward_step(..., state, packet_meta)` with true read trace.
4. Gate 3: create `opentad/models/targets/prefix_event_targets.py`.
5. Gate 4: implement `PrefixEventEmissionHead`.
6. Gate 5: repair and audit optimizer scope.
7. Gate 6: implement `OnlineAPBudgeted`, `GTMatchedLatencyEvaluator`, `PrefixCausalityAudit`, `FuturePerturbationAudit`.
8. Gate 7: build fair baseline configs.
9. Gate 8: run minimal THUMOS14 3-seed loop, then MUSES only if gates pass.

## Integration With 2026-07-09 Absorption

This review is stricter than `PRO_REVIEW_ABSORPTION_20260709.md` and should control current planning.

The 2026-07-09 absorption remains useful for:

- OZ-TAL threat boundary;
- previous source record;
- selected-only validation skeleton context.

This 2026-07-10 absorption adds:

- exact strict On-TAD definition;
- explicit 51.2s scheduling-latency derivation;
- optimizer-scope risk;
- stronger metric-gaming critique;
- PCEH route details;
- full training/inference/evaluation gates.

Combined rule:

> First repair the task/protocol and implement PCEH. Do not pursue zero-shot/open-vocabulary VLM claims without handling OZ-TAL, and do not pursue OnlineTAD-PT pretraining before PCEH has passed minimal gates.
