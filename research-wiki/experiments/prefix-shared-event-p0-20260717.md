---
type: experiment
node_id: exp:prefix-shared-event-p0
title: "Prefix-Shared Event Filter CPU P0"
stage: preregistration-closure-required
outcome: not-run
updated: 2026-07-17
---

# Prefix-Shared Event Filter CPU P0

## Status

Design authorization: `GO_NEW_ROUTE_P0_ONLY`.

Execution status: not ready and not started.

GPU authorization: zero hours.

The review freezes the scientific purpose, test families, headline margins,
and terminal rule. An exact execution preregistration must still freeze all
model, UOT, loss, generator, optimizer, runtime, and evidence constants before
implementation or outcome observation.

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

## Inherited Immutable Boundaries

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

## P0-A Deterministic Families

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

## Learned Anti-Silence Gate

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

## Preimplementation Closure Checklist

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

The P0 runner is forbidden until every item is frozen and hashed.

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

## Decision Rule

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

## Sources

- [`../../PRO_POST_Q2_R1_KILL_ONTAD_ROUTE_REVIEW_20260717.md`](../../PRO_POST_Q2_R1_KILL_ONTAD_ROUTE_REVIEW_20260717.md)
- [`../../PRO_POST_Q2_R1_KILL_ONTAD_ROUTE_REVIEW_ABSORPTION_20260717.md`](../../PRO_POST_Q2_R1_KILL_ONTAD_ROUTE_REVIEW_ABSORPTION_20260717.md)
- [idea:prefix-shared-latent-event-filter](../ideas/prefix-shared-latent-event-filter.md)
- [DR-045](../decision_register.md#dr-045-authorize-r-a-preregistration-and-cpu-only-p0)
