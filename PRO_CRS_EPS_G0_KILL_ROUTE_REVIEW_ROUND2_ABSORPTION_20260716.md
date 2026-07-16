# Independent Absorption of the CRS-EPS G0 Kill Route Review, Round 2

Date: 2026-07-16

## 1. Source and Integrity

The complete private attachment was copied without normalization to
`PRO_CRS_EPS_G0_KILL_ROUTE_REVIEW_ROUND2_20260716.md`.

```text
attachment:
C:/Users/skywalker/.codex/attachments/82b1e480-cc2b-4c1e-bede-db57191e6f2a/pasted-text.txt

archive:
PRO_CRS_EPS_G0_KILL_ROUTE_REVIEW_ROUND2_20260716.md

bytes: 120508
lines: 3879
SHA-256: 2BE3C21E951F68F397CADA468B66820BA08B3FF2F0FFD0A4E59580D784C989EE
```

The attachment contains two complete N-Y answers. The first answer ends with
the same machine verdict later repeated by a second, more detailed answer
introduced by `# Round 2 final verdict`. Both are retained in the immutable
archive. When a detail differs in granularity, the second N-Y answer is the
normative version because it is later, more complete, and preserves the same
terminal route selection.

## 2. Independent Verdict

I accept the scientific core of the review, with four implementation and
measurement amendments recorded below.

```text
ROUND2_SOURCE=RECORDED_BYTE_EXACT
CURRENT_G0=VALID_KILL
CURRENT_EMPTY_STATE_CRS_EPS=STRUCTURAL_KILL
Q2_STATUS=GOLD_REFERENCE_CANDIDATE
Q2_CAPACITY_GATE=BLOCK
SELECTED_SUCCESSOR=R1_CSFSB
R1_STATUS=CONDITIONAL_GO
IMMEDIATE_IMPLEMENTATION=CAPACITY_AUDIT_ONLY
R1_IMPLEMENTATION=ALLOW_AFTER_SHARED_CAPACITY_CONTRACT
NEW_HOLDOUT_G0=REQUIRED
PROFILE=BLOCK
FORMAL=BLOCK
RAW_VIDEO_STAGE2=BLOCK
GPU_HOURS_BEFORE_NEW_G0_PASS=0
LOCAL_CONFIDENCE=0.94
```

`IMPLEMENTATION=ALLOW` is not a blanket implementation or training approval.
The immediately authorized scientific commit is the zero-GPU Q2
capacity/lifecycle audit. R1 may be implemented only after that audit freezes
one shared, non-tainted lifecycle/capacity contract. No effectiveness claim,
profile, formal training, or raw-video adaptation is authorized.

## 3. What Is Now Frozen

The following conclusions are accepted as project memory:

1. The signed `70df86e` G0 is a valid terminal negative result.
2. The current empty-state `dynamic_birth` CRS-EPS replay is not a faithful
   surrogate for chronological training.
3. This negative result does not kill every selective-backward or
   event-centric training method.
4. Current chronological Q2 is only a gold/reference candidate. Two complete
   `video_start_full` controls exhausted slots, so Q2 is not primary-ready.
5. Full PETAL has not received method-level approval. Its novelty and
   effectiveness remain unproven.
6. The old four G0 samples are development diagnostics only and cannot be
   reused as confirmatory evidence.
7. Zero GPU hours may be spent before a new, unseen, CPU-only G0 passes.

The correct scope of the negative result remains:

```text
STRUCTURAL-KILL-CURRENT-EMPTY-STATE-DYNAMIC-BIRTH-CRS-EPS
```

It is not:

```text
KILL-ALL-SELECTIVE-OR-EVENT-CENTRIC-TRAINING
```

## 4. Selected Successor: R1 / CSFSB

The only route retained for falsifiable implementation is:

```text
R1 = Chronological State-Faithful Selective Backward
     (CSFSB)
```

R1 preserves the complete causal forward trajectory over every cached token.
All numerical recurrent state, runtime lifecycle, canonical supervision,
birth/retirement/refractory transitions, and ownership decisions advance at
every bin. Sampling chooses only which per-bin loss and gradient contributions
are retained; it never chooses which history is replayed or which state is
constructed.

The inherited chronological reference objective is the video-uniform mean of
the per-bin losses under the existing global 64-bin TBPTT detach contract:

```text
L_v = (1 / T_v) * sum_t loss(v, t)
```

The retained outcome-blind proposal is:

```text
M=4 draws/video/epoch
H=8 bins/window

uniform=0.40
start=0.15
end=0.20
ongoing=0.10
hardbg=0.15
```

For bin `t`, with exposure multiplicity `m_vt` and one-draw inclusion
probability `rho_vt`, the frozen coefficient is:

```text
a_vt = (m_vt / M) * ((1 / T_v) / rho_vt)
```

Repeated exposures are not deduplicated. The sampled risk is:

```text
Lhat_v = sum_t a_vt * loss(v, t)
```

It must not be renormalized by selected-bin count, selected-block count,
unique-bin count, or realized weight sum.

Within each global 64-bin block:

- a block with no selected bin runs a complete `no_grad` forward;
- a block with selected bins builds the graph from block start through the
  last selected bin;
- losses are evaluated only where `a_vt > 0`;
- state is detached after the last selected bin and the remaining tail runs
  under `no_grad`;
- the numerical state still advances through every bin;
- the block entrance and inter-block detach points remain identical to R0.

Consequently, R1 keeps `forward_tokens=T`. It can reduce backward graph tokens
from `T` to `P <= T`, but it does not automatically reduce forward FLOPs,
wall time, or GPU-hours.

## 5. Local Code Verification

### 5.1 The R1 target objective matches the current Q2 normalization

`PersistentTrajectoryOnTAD.train_episode` sums each per-step loss component,
divides by `valid_steps`, and exports `_optimizer_weight=valid_steps` in
`opentad/models/detectors/persistent_trajectory_ontad.py`.
`opentad/cores/train_engine.py` multiplies chunk loss by that weight, accumulates
across the video, and normalizes accumulated gradients by the total episode
weight at the video boundary.

Therefore current R0 implements a bin-uniform average of the existing
per-step weighted loss, not an unnormalized event sum. The review's `1/T_v`
Hansen-Hurwitz coefficient is mathematically aligned with the actual Q2
objective, provided R1 preserves every per-step loss definition and adds no
realized-sample normalization.

### 5.2 The capacity failure can be replayed with fixed logits in this head

`PersistentEventSetHead.step` updates query and feature memory from the input
feature and prior numerical memory. It carries lifecycle fields through but
does not read `slot_status`, refractory counters, start frames, peak scores, or
canonical maps when producing the next outputs.
`decode_step` subsequently mutates lifecycle state and the emission ledger.

For the current implementation, the same checkpoint, features, and RNG may
therefore produce one chronological logits trace that is replayed through
multiple lifecycle policies for diagnostic attribution. This is valid only as
a zero-GPU diagnostic. If a future head makes numerical predictions depend on
lifecycle tensors, this invariant must be re-audited before reuse.

### 5.3 The present lifecycle ordering exposes a real training blocker

`PersistentTrajectoryOnTAD.train_episode` obtains available slots from the
previous runtime state, runs `head.step`, then applies the canonical
supervision transition and runtime decode. A GT birth that finds no eligible
slot can be consumed by the canonical transition without a later retry.

The true capacity demand is therefore not bounded by annotation concurrency
alone. It includes canonical occupancy, model-created ACTIVE slots,
canonically free but runtime-REFRACTORY slots, and same-bin birth reserve. The
observed exhaustion counts of one and two in complete chronological controls
are sufficient to block Q2 readiness, though they do not yet prove that
`K=4` is intrinsically insufficient.

## 6. Route Portfolio Absorbed

| Route | Decision | Reason |
|---|---|---|
| R0 full chronological Q2 | Reference only | Exact current estimand, but capacity-blocked and potentially expensive. |
| R1 CSFSB | Conditional GO | State-faithful, mathematically testable, and cheap to falsify before GPU use. |
| R2 exact recomputation | Dominated | With known selected positions it costs approximately `T+P` forward work for the same `P` backward graph work. |
| R3 stored runtime checkpoints | No-GO | Parameter-stale state loses historical Jacobians; refreshing collapses toward R2 or R0. |
| R4 heterogeneous checkpoint/SBP/ARTBP/UORO family | Not one frozen route | Activation checkpointing may later be a memory optimization, not the selected scientific surrogate. |
| R5 chronological-only fallback | Honest fallback | Retain if R1 dies, but it cannot support an efficient-training claim. |

The old `dynamic_birth`, `fixed_192`, `reset`, and
`CrsEpsFeatureDataset` routes must not be revived as primary training.

## 7. Mandatory Capacity Audit Before R1

The first scientific implementation commit must contain only the capacity and
lifecycle audit. It must not implement R1 and must not change `K`, thresholds,
refractory, binding, loss, optimizer, or sampling.

Required audit surface:

1. Annotation-only capacity demand over the full 160-video fit core.
2. Actual chronological controller replay for deterministic seeds 705, 706,
   and 707.
3. Same-logits counterfactual lifecycle replay covering at least:
   current policy, refractory zero, threshold/prior sensitivity, same-bin
   release-before-birth, `K` sensitivity, and a privileged canonical-only
   upper bound clearly marked diagnostic-only.
4. Cause attribution separating true canonical occupancy, false ACTIVE
   occupancy, refractory occupancy, and same-bin ordering.
5. Fail-closed symmetry: ordinary chronological training must reject slot
   exhaustion before any optimizer step, just as the rejected CRS route did.
6. Exactly one shared fixed/rematch lifecycle/capacity contract must be frozen
   before effectiveness is visible.

If no non-tainted, zero-exhaustion shared contract exists:

```text
Q2_STATUS=INVALID
SELECTED_ROUTE=NONE
R1=BLOCKED
```

## 8. New Unseen CPU G0 Contract

The preferred micro-trajectory uses 16 metadata-selected fit-core videos,
disjoint from both the old four G0 samples and the new holdout. Seeds
705/706/707 produce states after 0, 4, 8, and 16 optimizer events, for 12
preferred checkpoint states. A bounded fallback may retain all four seed-705
states and initialization only for seeds 706/707, but its CPU limit and trigger
must be signed before any fidelity output is observed.

The new holdout contains eight unseen samples, two per length quartile,
selected outcome-blind to maximize lifecycle and gradient stress coverage.

### G0-A: implementation exactness

Compare an all-graph chronological gold execution with the R1 execution using
the same selected-bin mask, checkpoint, video, RNG, coefficients, fixed/rematch
mode, and FP32 CPU platform. Required hard invariants include logits, numerical
state, discrete lifecycle, canonical state, assignments, final RNG digest,
selected loss, and all-trainable gradients. The review freezes gradient cosine
and sign agreement at `>=0.999999` and relative gradient L2 at `<=1e-6`.

### G0-B: finite-M estimator adequacy

For each checkpoint stratum, aggregate the eight holdout videos with a
video-uniform rule and compare R0 all-bin chrono-64 with R1 `M=4` HH. Both fixed
and rematch must satisfy:

```text
gradient cosine >= 0.90
gradient sign agreement >= 0.90
relative aggregate loss error <= 0.10
finite gradient norm ratio inside a preregistered interval
```

### G0-C: capacity

Any slot exhaustion in any seed, checkpoint, sample, mode, gold, or candidate
is a terminal KILL. The family-wise gate passes only if every G0-A invariant,
every G0-B stratum, zero-exhaustion condition, and provenance/RNG/rollback
check passes.

## 9. Resource and Evidence Gates

```text
capacity/lifecycle audit: 0 GPU-hours
R1 implementation/tests: 0 GPU-hours
B0/review/new G0: 0 GPU-hours
fixed-step profile: <=2 GPU-hours, only after G0 PASS
one-seed fixed/rematch: <=4 GPU-hours, only after profile review
remaining pre-Stage2 reserve: <=4 GPU-hours, not preauthorized
total pre-Stage2 hard cap: <=10 GPU-hours
raw-video Stage 2: blocked
```

The evidence chain is strictly ordered:

```text
capacity audit and shared contract
-> R1 implementation commit
-> local B0
-> target-Linux B0
-> same independent reviewer PASS / NEXT_GATE=NEW_HOLDOUT_G0
-> signed micro-trajectory, holdout, margins, config, data, and checkpoints
-> unseen CPU G0 PASS
-> new ticket and <=2 GPU-hour fixed-step profile
-> profile review
-> one-seed fixed-vs-rematch mechanism kill
-> novelty killers FRESH/TTF/GRU
-> three paired seeds
-> five paired seeds plus dense/overlap dataset
-> renewed novelty review
-> only then consider raw-video LoRA
```

## 10. Independent Amendments

### A1. Split implementation permission explicitly

The review's headline `IMPLEMENTATION=ALLOW` is easy to overread. Project
memory therefore splits it into `CAPACITY_AUDIT_ONLY` now and
`R1_IMPLEMENTATION=ALLOW_AFTER_SHARED_CAPACITY_CONTRACT` later.

### A2. Do not duplicate the transactional training engine by default

The review proposes a separate `state_faithful_train_engine.py`. That filename
is advisory, not a scientific requirement. The existing train engine already
owns video-boundary optimizer semantics, rollback, manifest/evidence checks,
and launch gates. A parallel engine risks behavioral drift. Prefer a narrow
R1 route/helper that reuses the existing transactional path unless code review
proves a separate engine is the cleaner single source of truth.

### A3. Freeze CPU platform and edge-case denominators

G0-A must preregister CPU architecture, PyTorch version, FP32 mode, thread
settings, deterministic settings, and tolerance semantics. G0-B's relative
loss calculation must preregister behavior for zero or near-zero reference
loss. No tolerance may be relaxed after outcomes are visible.

### A4. Add a hard CPU budget before micro-trajectory generation

The review specifies a bounded fallback but not a concrete CPU wall-time or
CPU-hour ceiling. That ceiling and deterministic fallback trigger must be
frozen before trajectory generation, so resource pressure cannot become an
outcome-dependent checkpoint-selection rule.

## 11. Claims That Remain Prohibited

The following statements are not supported:

- R1 is already an efficient training method;
- R1 is already a paper-level innovation;
- Q2 is already a valid primary trainer;
- fixed persistent binding is better than rematching;
- Full PETAL has surpassed TrackFormer-like or simple recurrent controls;
- cached-feature success would authorize raw-video LoRA;
- the current negative result kills all event-centric training.

R1 is presently an enabling, capacity-gated experimental substrate with strong
reconstruction risk from ETAD, selective-backward, stochastic-backpropagation,
and ordinary importance-sampling families. A matched profile must show lower
wall time and GPU-hours, not merely fewer backward tokens. The paper-level
story remains gated on identity-specific mechanism evidence and strong
reconstruction controls.

## 12. Immediate Next Action

Implement only the zero-GPU Q2 capacity/lifecycle audit, run it over the frozen
fit core and seeds 705/706/707, and freeze one shared lifecycle/capacity
contract or terminate Q2/R1. Do not launch profile or formal training.
