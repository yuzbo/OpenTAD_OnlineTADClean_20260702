---
type: query_pack
updated: 2026-07-16
status: active
scope: Compressed memory to prepend before any new ideation or implementation planning.
---

# Query Pack: Online/Causal TAD Project Memory

## Project Direction

The task is fixed to standard fully supervised Online Temporal Action Detection/Localization: causal RGB stream in, one immutable `{start, end, class, score}` instance emitted when an action end is detected. Do not introduce sensors, new observability labels, semantic-maintenance outputs, or a replacement task.

PIVOT is rejected as out of scope. Incremental PCEH/CESR remains demoted. Full PETAL is also demoted after a Pro `REVISE`: causal backbones, direct On-TAD state/query methods, and TrackFormer-style persistence can reconstruct most of the package. Read `PRO_PETAL_DEEP_REVIEW_ABSORPTION_20260712.md`, DR-026, and `experiments/persistent-feature-kill-test-20260712.md` before proposing or training anything.

The conditionally accepted CRS-EPS hybrid was fully implemented and falsified at commit `70df86e`. Its proposed successor R1/CSFSB was never implemented because the prerequisite Q2 capacity audit returned `REVISE_REQUIRED`: 2,206 GT births exhausted under the actual controller. The only legal zero-exhaustion arm, additive birth bias `-2`, emitted only ten events. Unique independent max review verified 524,258 trace rows, diagnosed degenerate birth suppression, and selected terminal `KILL_Q2_R1`. Q2/R1, GPU profile, effectiveness training, formal training, and raw-video Stage 2 are blocked. No active training route is currently authorized.

## Top Gaps

1. **Raw-video instance-level joint training is missing.** Current On-TAD leaders use frozen/pre-extracted TSN, I3D, SlowFast, or pickle features; raw-video online methods predominantly solve frame-level OAD.
2. **Window rediscovery is a structural failure.** Independent windows repeatedly rediscover one action, causing fragmentation, duplicates, same-class merging, and online-NMS dependence.
3. **Current PCEH is scientifically blocked.** End/emit coupling, repeated late positives, class-level targets, end=emit decode, detached state, and GT proximity remain core concerns.
4. **The complete-video reference is expensive and scientifically invalid as the R1 substrate.** It uses one optimizer event per video but runs every cached token sequentially, while its runtime/canonical availability contract exhausts 2,206 GT births.
5. **Generic streaming pretraining is occupied.** StreamFormer blocks the claim that a causal streaming backbone alone is new; BSP and offline E2E-TAD block generic boundary pretraining or PEFT claims.
6. **The required intersection remains open.** Strict causal raw-video adaptation, persistent action-instance identity, standard immutable On-TAD emission, and prefix-equivalent efficient training must all hold together.
7. **Same-class repetition and overlap remain direct risks.** ActionSwitch is the closest baseline and must be matched fairly.
8. **Evaluation must prevent future use and duplicate cleanup.** Full chronological evaluation, fixed thresholds, immutable outputs, recall/FN, delay, and no offline NMS remain mandatory.

## Candidate Portfolio

- **Q2 fixed/rematch:** terminal `KILL` disposition after a valid `REVISE_REQUIRED` capacity audit. It remains historical negative evidence, not a trainer or gold reference.
- **CRS-EPS dynamic replay:** negative for the current protocol. Signed G0 failed absolute state/loss/gradient fidelity. The exposed G0 cases are development-only; any replacement needs a new holdout protocol.
- **R1/CSFSB:** terminal `KILL` under the frozen protocol. It was never implemented because Q2 failed to provide a nondegenerate shared lifecycle contract.
- **Full raw-video PETAL:** demoted/blocked. A Stage-1 pass only retains it for five-seed confirmation and renewed novelty review.
- **OnlineTAD-specific pretraining:** supporting option only after PETAL's mechanism works; generic pretraining is not the headline.
- **CESR/PCEH:** causal infrastructure and negative baselines only, not paper framing.
- **PIVOT/T01/T03/P01/L01:** rejected as current main task or held outside the fixed On-TAD scope.
- **Cache/LoRA/ETAD-style gradient sampling:** cost-control tools, not standalone novelty.

## Current Q2 Exact Question

Historical, now terminated under the frozen protocol:

Defensible sentence:

> Under identical cached features, detector weights, lifecycle, birth rule, capacity, optimizer, and evaluator, does fixed post-birth slot-instance supervision outperform per-prefix active-pool rematching because it preserves identity-linked localization rather than changing masks, normalization, or negative semantics?

No raw-video, end-to-end, or novelty claim is currently active. Persistent tracks are internal state; standard immutable On-TAD outputs and metrics remain unchanged.

## Failed / Blocked Claims

- PIVOT/three-clock event verification: rejected because it changes the project task.
- Generic causal backbone, generic memory, generic pretraining, LoRA, raw-frame input, or online cache alone: occupied and insufficient.
- Early proposals or mutable user-visible revisions: outside the fixed standard On-TAD output protocol.
- Current THUMOS PCEH, full-packet training first, zero-shot wrapper, and offline distillation headline: rejected.
- Current CRS-EPS `dynamic_birth` as primary training: rejected by its preregistered G0; HH/IPW did not establish state/gradient fidelity.
- Q2/R1/CSFSB: rejected because the shared runtime/canonical capacity contract exhausts 2,206 GT births and its only zero arm passes by nearly silencing birth.
- PETAL is also rejected if it reduces to TrackFormer plus a one-dimensional interval head without an On-TAD-specific trajectory mechanism or measurable matched gain.

## Closest Prior Work

### PETAL

- **MATR/HAT/OAT:** direct window/query On-TAD references over frozen features; MATR uses online NMS.
- **ActionSwitch:** same-class/overlap state-switch baseline.
- **E2E-LOAD/StreamFormer:** raw-video causal OAD, but frame-level outputs.
- **E2E-TAD/TIA, LoSA, Re2TAL, ETAD:** offline TAL adaptation/efficiency precedents.
- **TrackFormer/online VIS:** strongest persistent-query obviousness attack.

## Persistent-State Mandatory Gates

1. PES must beat FRESH and Temporal TrackFormer on matched cached features, including duplicate/fragmentation or same-class/overlap errors.
2. Births must be prefix-observable; post-birth identity fixed; future assignment privileged only.
3. Future perturbation, prefix-cut, stepwise equivalence, immutable emission, no-cleanup, and miss/late-FP audits must pass.
4. Novelty must survive TrackFormer plus causal-backbone reconstruction; cost must improve matched wall time/GPU-hours.
5. Three seeds are kill-only; retained claims require five seeds, paired uncertainty, and a dense/overlap benchmark.

Protocol taint, slot exhaustion, unmatched seeds, or resource-cap breach invalidates Stage 1. Historical effect-size thresholds are project resource gates, not universal significance claims.

## Infrastructure Laws

- Keep PCEH repairs for baselines; do not inherit its hazard framing by default.
- Require causal reads, chronological validation, immutable outputs, miss/late-FP accounting, no-future audits, and stepwise/batched agreement.
- Identity must be trajectory-level. Training GT assignment is allowed; model input and inference state remain future-free.

## Current Final Goal

Do not start profile or training. Preserve both the `70df86e` CRS-EPS G0 `KILL` and the `1441219` Q2 capacity `REVISE_REQUIRED` plus independent `KILL_Q2_R1` disposition. Do not reopen Q2 with a birth prior, threshold, K, refractory, or release-order patch. The next research step, if continued, must define a new On-TAD mechanism whose runtime state and GT supervision share a non-silence-passable availability contract, then preregister a new zero-GPU falsification protocol before implementation. GPU hours: 0.
