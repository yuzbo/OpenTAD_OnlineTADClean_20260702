# Pro Review Absorption: 2026-07-11

Source record:

- Full external review: `PRO_REVIEW_20260711.md`
- Original source: private Codex attachment, not committed; integrity is tracked by the SHA256 below.
- Attachment SHA256: `203AAF966F7AE1F3642439D94243C16EC7096A556A825DAFE4610F27C3B35FB6`
- Attachment line count by `ReadAllLines`: 1353
- Attachment character count by UTF-8 raw read: 49267
- Reviewed visible HEAD according to reviewer: `bfd0608b2996cba30d158d741ee193476a5078df`
- Local absorption date: 2026-07-11
- Overall verdict: `HOLD`
- Formal training verdict: blocked until P0 scientific-correctness gates pass

## Absorbed High-Level Verdict

This review tightens, rather than replaces, the 2026-07-10 verdict.

The current repository can be treated as a credible strict-prefix computation skeleton for the frozen/adaptor smoke path, but it is not yet a statistically valid PCEH method and must not be promoted to formal training or paper claims.

The most important change in understanding is:

> The current code has separate endpoint, completion, and emission parameters, but it does not yet have statistically decoupled endpoint hazard and first-emission hazard learning.

Current status:

- Inference input causality: mostly pass for the current per-frame SigLIP2 plus causal projection path.
- Model state causality: smoke-level pass, still needs formal checkpoint replay.
- Training sampler and target causality: future GT is used for supervision and future prefix selection; this is allowed only if disclosed as privileged or annotation-guided supervision.
- Immutable emission ledger: skeleton pass.
- Endpoint/emission statistical decoupling: fail.
- Full visual tower end-to-end finetuning: fail.
- DDP streaming: fail.
- Formal training readiness: fail.

## New Project Law

Use these terms precisely:

- Allowed: `strict inference-time online`, `prefix-causal model inputs`, `annotation-guided causal-prefix training`, `full chronological stream evaluation`, `immutable ledger`.
- Blocked: `training uses no future information`, `full visual tower end-to-end finetuning`, `statistically independent first-emission hazard`, `paper-ready`, `SOTA`, `multi-seed formal evidence`.

The correct sentence for event-centric training is:

> The sampler may use full annotations to choose supervised prefix times, while the model input and computation graph at each supervised time contain only sources no later than that prefix; final evaluation is a complete chronological stream with no future reads or output revision.

Do not write:

> Training and inference never use future information.

## Critical Scientific Failures to Fix First

### 1. Endpoint and emission labels are currently the same event

The current target logic marks endpoint and emission positive at the same crossing. That makes emission a copy of endpoint, not a first-stop policy.

Required fix:

- Build a separate endpoint risk set.
- Build a separate emission risk set.
- Ensure each instance has at most one endpoint event and one first-emission event.
- Mask post-event bins out of the risk set.

### 2. Late prefixes are wrongly treated as repeated emission positives

Current late-positive supervision pushes emission logits after the budget has already been missed. This violates first-event survival semantics.

Required fix:

- Remove late-positive repeated emission labels.
- Treat late as a penalty, invalid state, or missed opportunity, not as a new event.

### 3. Targets are class-aggregated instead of instance-aware

Class-level `any()` causes old same-class instances to pollute later instances. This is a direct reviewer attack vector, especially on repeated or overlapping same-class events.

Required fix:

- Represent every GT instance separately.
- Assign instance states into finite track slots only after instance-level risk construction.
- Add tests for repeated same-class and overlapping same-class videos.

### 4. Decoder currently sets predicted end equal to emit time

If `predicted_end_frame = emit_frame`, then `emit_time - predicted_end_time` is always near zero and cannot prove endpoint/emission separation.

Required fix:

- Freeze `predicted_end_frame` when endpoint is first detected.
- Allow emission later from a pending state.
- Add a delayed synthetic test where `predicted_end_frame < emit_frame`.

### 5. GT is still too close to model metadata

Full GT per packet is acceptable for target building, but risky if it enters model kwargs or metadata.

Required fix:

- Split model inputs from supervision namespace.
- Add GT taint tests that fail if `stream_gt_*`, duration, video end, or terminal metadata reach backbone/projection/head kwargs.

## Training Cost Absorption

The review treats current full-packet training cost as scientifically and practically unacceptable for formal evidence:

- Around 152,670 packets per epoch.
- Around 7 hours per epoch.
- Around 210 GPU-hours per model per seed for 30 epochs.
- PCEH vs endpoint-only with 3 seeds would be around 1260 GPU-hours before ablations and failures.

The cost source is not just long videos:

- too many tiny Python/optimizer events;
- per-packet video open/seek/close;
- SigLIP processor CPU round trips;
- `num_workers=0`;
- repeated 192-token projection recomputation;
- single-GPU chronological sampler restriction.

Feature cache is now elevated from optional speed trick to Stage 1 main training route for frozen SigLIP2, because the current selected-frame feature volume is small enough to audit and cache.

## Recommended Main Route to Absorb

The reviewer recommends one route:

> Instance-aware Causal Risk-Set Event-Centric Prefix-Episode Training: Frozen SigLIP2 Feature Cache -> Selective-Gradient LoRA, with complete chronological raw-stream evaluation.

Working shorthand: `CRS-EPS`.

This should be treated as the training protocol for PCEH, not yet as a named independent novelty claim.

### Stage 0: Scientific-Correctness Repair

Must precede long training:

- instance-aware risk targets;
- endpoint discrete hazard;
- endpoint first-crossing freeze;
- latency-aware first-stop emission policy;
- multi-slot or clearly bounded same-class instance handling;
- GT taint audit;
- delayed endpoint/emission synthetic case.

### Stage 1: Frozen Cache Event-Sampled Training

Train only:

- cache-side temporal adapter;
- projection;
- PCEH or endpoint head.

Use causal prefix episodes:

- 192 decision tokens as burn-in/context;
- 4-8 supervised decision bins;
- loss only on sampled supervised bins.

Sampling should cover:

- endpoint risk episodes;
- start episodes;
- ongoing episodes;
- hard background;
- long background;
- later false-positive mining.

Must record inclusion probability and compare sampled objective to a full-packet gold subset.

### Stage 2: Selective-Gradient SigLIP2 LoRA

Train:

- top visual blocks, initially q/v LoRA;
- temporal adapter;
- projection;
- head.

Hard requirements:

- runtime module target discovery;
- zero LoRA target hit must fail closed;
- each target block must show nonzero gradient and parameter delta;
- base weights remain frozen;
- outer wrapper cannot use global `no_grad()` around trainable LoRA;
- cache manifest must bind checkpoint SHA.

### Stage 3: Full Visual Tower Is Conditional Only

Do not make full-tower finetune a default route. It is allowed only after:

- PCEH beats endpoint-only on AP-latency Pareto;
- LoRA beats frozen;
- LoRA update audit passes;
- raw episodic training throughput is acceptable;
- a one-epoch full-tower diagnostic fits allocation.

## Evaluation and Evidence Requirements

Main evaluation must stay as full chronological validation or test stream.

Report at minimum:

- OnlineAP at 0.5/1/2/4 second budgets;
- tIoU 0.3 to 0.7;
- matched GT-end latency p50/p90/p95;
- matched TP count;
- recall;
- FN count;
- late FP count;
- duplicate emission rate;
- future-read and ledger mutation violations;
- encoded frames, visual forward frames, visual backward frames;
- optimizer steps, trainable parameter count, GPU-hours, peak memory;
- cache build time, size, refresh time, cache hit, manifest SHA.

Important rule:

- Late predictions remain FP, and their GT remains FN.
- Different latency budgets must evaluate the same locked ledger and threshold, not retune thresholds per budget.
- Latency must not be reported only over TP without coverage counts.

## Required Baselines and Ablations

Priority baselines:

- endpoint-only with same architecture and same training protocol;
- current full-packet route on a gold subset;
- event-sampled PCEH;
- frozen cache PCEH;
- LoRA PCEH after Stage 2;
- simple chunk/window baseline;
- relevant OAT/MATR/HAT/ActionSwitch-style comparisons if implementation scope allows.

Required mechanism ablations:

- class-aggregated targets vs instance-aware targets;
- invalid late-positive emission vs corrected first-stop risk loss;
- emission gate on/off;
- endpoint freeze on/off;
- importance weighting on/off;
- one track/class vs multi-slot;
- completion score in confidence on/off;
- emission penalty sweep;
- fixed-budget vs budget-conditioned policy.

## Implementation Queue

P0 must happen before any new long run:

1. Add `opentad/models/targets/prefix_event_targets.py` or equivalent instance-aware target builder.
2. Add discrete endpoint and first-emission risk-set target tests.
3. Change PCEH decoder/state machine so predicted endpoint can freeze before emission.
4. Move full GT into an explicit supervision namespace and add taint tests.
5. Fix `OnlineAPBudgeted` class universe, provenance preconditions, emission ID, and diagnostic latency usefulness.
6. Add same-class overlap analysis.

P1 then implements the cheaper main training path:

1. Prefix-episode dataset.
2. Risk-set event sampler.
3. Cached stream feature dataset.
4. Cache manifest and raw/cache equivalence audits.
5. Training benchmark JSON.
6. Video reader pool and batched decode path.

P2 is LoRA:

1. Runtime-verified SigLIP2 LoRA injection.
2. No global `no_grad()` blocking trainable visual modules.
3. Gradient coverage audit.
4. Optimizer coverage audit.

P3 DDP is second priority:

- shard complete videos by rank;
- never split one video across ranks;
- gather ledgers without stream identity loss.

## Review Risk Register

Fatal or high risks to keep visible:

- PCEH is not a legal hazard model until risk sets are fixed.
- Separate heads are not a method contribution without endpoint-only controls.
- One track per class cannot handle repeated or concurrent same-class actions.
- Predicted endpoint equal to emit destroys endpoint/emission claim.
- Annotation-guided sampling must not be described as future-free training.
- Feature cache must not be described as visual end-to-end learning.
- LoRA must be proven to update.
- Sampled training can distort base rates; include inclusion probabilities and gold-subset checks.
- Low latency can be cheated by emitting less; always report recall and FN.
- Single-seed pilot is not result evidence.

## Final Absorbed Decision

Do not launch formal PCEH multi-seed training from the current code.

Allowed now:

```text
instance-aware risk targets
-> endpoint-frozen state machine
-> event-centric causal episodes
-> frozen SigLIP2 feature cache
-> full chronological short pilot
-> LoRA update smoke
```

Formal training can start only after:

- multi-instance/same-class target tests pass;
- late is no longer repeated emission positive;
- delayed synthetic case shows `predicted_end < emit`;
- sampled objective matches full-packet gold subset closely enough;
- PCEH single-seed short pilot shows directional AP-latency advantage over endpoint-only;
- full chronological validation completes;
- causality and ledger violations are zero;
- frozen Stage 1 throughput is acceptable;
- LoRA target-block update audit passes.

If three paired seeds later show no stable AP-latency Pareto gain over endpoint-only, stop the PCEH main claim instead of packaging it.
