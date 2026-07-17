---
type: experiment
node_id: exp:prefix-shared-event-p0
title: "Prefix-Shared Event Filter CPU P0"
stage: withdrawn-before-preregistration
outcome: not-run
updated: 2026-07-17
superseded_by: exp:prefix-route-identifiability-gate
---

# Prefix-Shared Event Filter CPU P0

## Status

Historical design authorization: `GO_NEW_ROUTE_P0_ONLY`, superseded before
preregistration or implementation.

Current route verdict: `REVISE_PROTOCOL_BEFORE_COLLECTION`.

Execution status: withdrawn, not implemented, and not run.

GPU authorization: zero hours.

DR-046 revokes the model-P0 permission because the proposal cannot distinguish
R-A from simpler no-identity sets, persistent queries, temporal MOTR, or a
clean order/risk-set baseline. DR-047 also blocks new R0/R1 evidence
collection until an executable protocol receives an independent PASS. No
model, UOT, loss, generator, optimizer, runtime, threshold, comparison, or
evidence constant may now be frozen for this experiment.

All remaining sections preserve the prior proposal as decision history only.
They are not an executable contract. The current legal successor is the
[route-identifiability gate](prefix-route-identifiability-gate-20260717.md).

## Scientific Question

Can one shared model-only causal state transition support latent event
birth/occupancy/completion, same-bin release/reseed, and immutable interval
commit while:

- excluding GT from runtime state and emission;
- using prefix GT only in temporary loss transport;
- preventing silent/no-emission policies from passing readiness;
- preserving train/inference and stepwise/packet equality?

This P0 does not test THUMOS14 effectiveness, publication novelty, raw-video
training, or GPU efficiency.

## Historical Proposed Boundaries

- standard fully supervised On-TAD output;
- no future access, offline NMS, mutable committed output, teacher cache, or GT
  runtime state;
- new modules only; Q2/R1 code and evidence remain read-only;
- CPU FP32;
- seeds `9101/9102/9103`;
- 512 generated training sequences;
- 256 outcome-blind held-out sequences;
- 64 all-background sequences;
- zero GPU hours;
- at most the review's two-CPU-hour execution budget, after its exact
  aggregate/wall-clock interpretation is frozen;
- any frozen gate failure is terminal KILL.

## Historical Proposed Deterministic Families

1. single event;
2. delayed endpoint;
3. same-class repetition closer than old Q2 refractory;
4. cross-class overlap;
5. same-class overlap;
6. old-event end and new-event birth in the same bin;
7. start and end in one decision interval;
8. all background.

Under scripted/oracle logits:

- event count must match exactly;
- recall and precision must equal one;
- duplicate and fragmentation must equal zero;
- future access must equal zero;
- all-background emission must equal zero;
- stepwise and packet execution must produce byte-identical ledger rows.

## Shared-State Equality

For identical parameters, input, and initial state, compare:

- training mode with loss disabled;
- inference mode;
- stepwise execution;
- packet execution.

Every prefix must match under the preregistered exact/tolerance rule for
scientific state, logits, and ledger. Scientific train/inference branches are
forbidden.

## Anti-Taint

GT-future perturbation:

- keep all annotations through prefix t identical;
- change future class, endpoint, and instance count;
- runtime state, emissions, and source trace through t must remain unchanged;
- only ephemeral targets/transport may differ.

Video-future perturbation:

- change all video/features/labels after t;
- state, logits, and ledger through t must remain unchanged.

## Historical Proposed Anti-Silence Gate

Every seed must independently satisfy:

- macro event recall >= 0.90;
- precision >= 0.90;
- same-class overlap recall >= 0.80;
- mean absolute event-count error <= 0.10 per positive sequence;
- all 64 background sequences produce zero false emissions;
- duplicate fraction = 0;
- fragmentation rate = 0.

No-birth, no-emission, and always-background controls must have recall zero
and readiness `FAIL`. A mean across seeds cannot hide one silent seed.

## Gradient and Optimizer Gate

For birth, occupancy, end, class, start, release/reseed, and direct-complete
components where applicable:

- gradients are finite;
- at least one applicable family has nonzero gradient norm;
- positive and negative perturbations have the preregistered direction;
- every trainable parameter appears in the optimizer exactly once;
- frozen parameters do not appear;
- endpoint and ledger records are detached and cannot be revised by later
  loss.

## Historical Unclosed Constants

The following remain unfrozen:

- K, dimensions, memory, initial state, direct-complete default;
- exact transition tensors, clamping, detach, and tie order;
- UOT cost, dustbin, epsilon, mass relaxation, iterations, tolerance,
  stabilization, and autograd policy;
- loss weights, risk/positive-mass denominators, empty-loss zero, and
  consistency horizon;
- ledger thresholds, score, snapshot, tie-breaking, and latch rules;
- complete synthetic distribution and train/holdout generator;
- optimizer, LR/schedule, initialization, update count, batching/TBPTT,
  clipping, and stopping;
- exact/tolerance comparisons and metric denominators;
- CPU threads, deterministic runtime, timeout, resource interpretation, and
  atomic evidence schema.

The P0 runner remains forbidden. Closing these constants is no longer the next
step; the route-identifiability gate must pass first.

## Required Evidence

- exact commit and clean tree;
- source/config/generator hashes;
- all seeds and generated split identities;
- Python/PyTorch/CPU/thread/runtime identity;
- state/logit/ledger traces;
- test collection and complete stdout/stderr;
- optimizer coverage;
- per-family and per-seed metrics;
- negative-control verdicts;
- resource use;
- machine-readable verdict;
- partial-directory write, flush/hash, and atomic rename;
- refusal to overwrite an existing evidence root.

## Historical Proposed Decision Rule

```text
ANY Q1-Q6 FAILURE:
  P0_STATUS=KILL
  P1_ALLOWED=false
  GPU_PROFILE_ALLOWED=false
  FORMAL_TRAINING_ALLOWED=false
  GPU_HOURS_AUTHORIZED=0

ALL Q1-Q6 PASS:
  P0_STATUS=PASS_PENDING_INDEPENDENT_REVIEW
  P1_ALLOWED=false
  GPU_PROFILE_ALLOWED=false
  FORMAL_TRAINING_ALLOWED=false
  GPU_HOURS_AUTHORIZED=0
```

Only an independent read-only review may later choose
`PASS_AUTHORIZE_UNSEEN_CPU_G0_ONLY`.

This historical escalation rule is inactive under DR-046.

## Sources

- [`../../PRO_POST_Q2_R1_KILL_ONTAD_ROUTE_REVIEW_20260717.md`](../../PRO_POST_Q2_R1_KILL_ONTAD_ROUTE_REVIEW_20260717.md)
- [`../../PRO_POST_Q2_R1_KILL_ONTAD_ROUTE_REVIEW_ABSORPTION_20260717.md`](../../PRO_POST_Q2_R1_KILL_ONTAD_ROUTE_REVIEW_ABSORPTION_20260717.md)
- [`../../PRO_PREFIX_SHARED_ROUTE_PREIMPLEMENTATION_REVIEW_20260717.md`](../../PRO_PREFIX_SHARED_ROUTE_PREIMPLEMENTATION_REVIEW_20260717.md)
- [`../../PRO_PREFIX_SHARED_ROUTE_PREIMPLEMENTATION_REVIEW_ABSORPTION_20260717.md`](../../PRO_PREFIX_SHARED_ROUTE_PREIMPLEMENTATION_REVIEW_ABSORPTION_20260717.md)
- [idea:prefix-shared-latent-event-filter](../ideas/prefix-shared-latent-event-filter.md)
- [DR-046](../decision_register.md#dr-046-revoke-r-a-default-route-authorization-before-implementation)
- historical [DR-045](../decision_register.md#dr-045-authorize-r-a-preregistration-and-cpu-only-p0)
