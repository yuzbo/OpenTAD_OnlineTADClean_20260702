---
type: experiment
node_id: exp:prefix-route-identifiability-gate
title: "Persistent-Carrier On-TAL Route Identifiability Gate"
stage: protocol-design-required
outcome: not-run
updated: 2026-07-17
---

# Persistent-Carrier On-TAL Route Identifiability Gate

## Status

Route verdict: `REVISE_ROUTE_AND_REVIEW_AGAIN`.

This node records the evidence required before any R-A model P0 contract,
implementation, or training. It is not a model experiment, has not run, and
does not authorize B0-B4 implementation.

GPU authorization: zero hours.

## Scientific Question

> In standard, strictly causal On-TAL with immutable committed outputs, do
> persistent event carriers provide online instance-consistency gains that
> cannot be reconstructed by a no-identity prefix set or a matched
> one-dimensional TrackFormer/MOTR?

"Instance-consistency gains" must be observable in committed outputs:

- duplicate rate;
- fragmentation rate;
- same-class repetition and overlap recall;
- same-bin end/start correctness;
- event recall and false emission;
- endpoint/commit latency;
- mAP-latency or recall-latency Pareto.

Internal query identity and swap rate may be diagnostics, but cannot be the
sole outcome.

## Gate R0: Outcome-Blind Annotation Census

Before examining model outputs, bind:

- exact dataset and split manifest hashes;
- annotation bytes and class map hash;
- decision-bin convention and feature stride;
- per-video and per-class event counts;
- same-class repetition frequency and inter-event gaps;
- same-class and cross-class overlap frequency and duration;
- same-bin end/start frequency;
- actions starting and ending within one decision interval;
- action-duration distribution;
- maximum total and same-class concurrency;
- zero-action stream frequency.

The census must be read-only, deterministic, and outcome-blind. Its report and
source hashes must be immutable. It describes whether the proposed problem is
present; it does not measure model effectiveness.

Kill or revise conditions:

- the exact split cannot be bound;
- the intended hard cases are too rare for a powered main claim;
- K or direct-complete assumptions were chosen before the census;
- census definitions are altered after model results.

## Gate R1: Cached-Feature Strict-Causality Certificate

Bind:

- encoder repository, source commit, configuration, and weights;
- frame sampling and preprocessing;
- clip construction, padding, stride, center/end alignment, and batching;
- exact temporal receptive field for every emitted feature token;
- mapping from raw frames to token source and decision times;
- proof or executable audit that no token uses a frame after its decision time;
- feature manifest and content hashes.

Monotonic source-frame metadata is insufficient if the encoder clip itself
uses future context.

Kill or revise conditions:

- encoder provenance or weights are missing;
- any input frame exceeds token decision time;
- boundary padding or clip centering introduces future information;
- the certificate applies to a different feature cache than the intended
  comparison.

## Gate R2: B0-B4 Comparison Contract

Candidate arms:

| Arm | Route | Purpose |
|---|---|---|
| B0 | no-identity prefix completion set | tests whether persistent identity is needed |
| B1 | ordinary persistent query plus interval/class/completion heads | tests whether R-A-specific lifecycle machinery is needed |
| B2 | one-dimensional TrackFormer/MOTR reconstruction | strongest obviousness and reconstructibility attack |
| B3 | new clean order/risk-set baseline | tests whether fixing Q2's local contract is sufficient without reopening Q2 |
| B4 | R-A Prefix-Shared Latent Event Filter | candidate route, not default method |

Before implementation, freeze principles for:

- identical causal inputs and decision times;
- identical output heads where the model family permits;
- identical append-only ledger and evaluator;
- identical no-future and anti-silence requirements;
- parameter-count matching rule and tolerance;
- optimizer-update and token budget matching rule;
- identical train, validation, and hidden structural-OOD generators;
- identical reporting and terminal-failure policy.

The contract may later specify model details only after an independent route
review authorizes that step.

## Gate R3: Negative Controls

The comparison must include:

1. count-only predictor;
2. template-timing predictor;
3. feature-time shuffle preserving count and label marginals;
4. label-feature permutation;
5. ledger-only scripted deduplicator;
6. history-off model;
7. conservative threshold sweep with recall and latency accounting.

These controls must fail readiness or lose the preregistered structural-OOD
metrics. Otherwise the benchmark cannot identify lifecycle learning.

## Gate R4: Mechanism Deletions

For any future B4 contract, predeclare:

- UOT to Hungarian plus dustbin;
- UOT to partial matching;
- continuous occupancy removed;
- adjacent-prefix consistency removed;
- release-before-reseed removed;
- latch excluded from neural transition;
- model-output-only latch included in neural transition;
- K versus K+1 concurrency stress;
- start posterior to scalar start regression;
- direct-complete disabled.

No mechanism may be called necessary unless its deletion removes a
preregistered observable advantage without improving another primary metric by
an equivalent amount.

## Gate R5: Structural OOD

Random seeds under one generator are insufficient. Hidden validation must
include combinations not observed in training:

- event counts;
- action lengths;
- repetition intervals;
- overlap graphs;
- same-bin end/start patterns;
- class permutations;
- feature-basis rotations or mixtures;
- noise families;
- endpoint-delay distributions;
- multiple simultaneous shifts.

The validation generator, distributions, hidden seeds, and content hashes must
be frozen independently before model implementation.

## Gate R6: Temporal-MOTR Exact Delta

Before B4 implementation, provide a table mapping every proposed R-A element
to:

- the nearest TrackFormer/MOTR mechanism;
- the exact algorithmic difference;
- the On-TAL-specific failure it is intended to solve;
- the observable metric affected;
- the mechanism deletion that isolates it;
- the equivalence region.

Predeclared terminal rule:

> If matched temporal MOTR falls inside the frozen equivalence region on every
> primary lifecycle and standard detection metric, R-A is route `KILL`.

Renaming persistent queries as carriers, adding an interval head, or attaching
the shared ledger does not count as a delta.

## Gate R7: Independent Route Review

The reviewer receives paths to immutable evidence, not an author summary. The
only allowed terminal decisions are:

- `PASS_AUTHORIZE_MODEL_P0_CONTRACT_FREEZE`;
- `REVISE_ROUTE_AND_REVIEW_AGAIN`;
- `KILL_PERSISTENT_CARRIER_ROUTE`.

A PASS may authorize exact model-P0 contract design. It cannot directly
authorize implementation, real-data effectiveness, GPU profile, formal
training, visual fine-tuning, or raw-video training.

## Current Authorization Matrix

| Action | Status |
|---|---|
| Write and review R0-R6 evidence protocols | `ALLOW` |
| Read-only, outcome-blind annotation census after protocol freeze | `ALLOW_AFTER_PROTOCOL` |
| Read-only feature-provenance and causality audit after protocol freeze | `ALLOW_AFTER_PROTOCOL` |
| Freeze R-A model/UOT/loss/threshold constants | `BLOCKED` |
| Implement B0-B4 models | `BLOCKED` |
| Run the historical R-A P0 | `BLOCKED` |
| Real-data effectiveness | `BLOCKED` |
| GPU/profile/formal/raw-video work | `BLOCKED` |
| GPU hours | `0` |

## Sources

- [`../../PRO_PREFIX_SHARED_ROUTE_PREIMPLEMENTATION_REVIEW_20260717.md`](../../PRO_PREFIX_SHARED_ROUTE_PREIMPLEMENTATION_REVIEW_20260717.md)
- [`../../PRO_PREFIX_SHARED_ROUTE_PREIMPLEMENTATION_REVIEW_ABSORPTION_20260717.md`](../../PRO_PREFIX_SHARED_ROUTE_PREIMPLEMENTATION_REVIEW_ABSORPTION_20260717.md)
- [DR-046](../decision_register.md#dr-046-revoke-r-a-default-route-authorization-before-implementation)
- [T39](../discussion_timeline.md#t39-preimplementation-review-revokes-r-a-default-route-status)
- [idea:prefix-shared-latent-event-filter](../ideas/prefix-shared-latent-event-filter.md)

## Connections

[AUTO-GENERATED from graph/edges.jsonl; do not hand-edit]
