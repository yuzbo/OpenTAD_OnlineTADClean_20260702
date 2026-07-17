---
type: idea
node_id: idea:prefix-shared-latent-event-filter
title: "Prefix-Shared Latent Event Filter with Immutable Ledger"
stage: route-revision-required
outcome: untested
updated: 2026-07-17
target_gaps: ["G2", "G3", "G6", "G7", "G16"]
portfolio_role: candidate-arm-b4
---

# Prefix-Shared Latent Event Filter with Immutable Ledger

## One-Line Thesis

Use one model-generated latent event filter at every causal prefix in both
training and inference; keep GT assignment loss-local, and commit completed
intervals to an immutable model-only ledger.

## Status

`REVISE_PROTOCOL_BEFORE_COLLECTION`.

DR-045's design authorization was superseded by DR-046 before preregistration
or implementation. R-A is not the default route and has no model-P0
permission. It survives only as B4 in a future route-identifiability
comparison. It is not implemented, has no P0 result, has no effectiveness
evidence, and has no GPU permission.

DR-047 further blocks even new R0 annotation census and R1 cache audit
collection until an executable protocol-only commit receives an independent
protocol PASS. The current legal work is protocol revision, not evidence
collection.

Q2/R1 remains terminal `KILL`. This route may reuse generic evaluator,
optimizer-audit, evidence, and causal execution infrastructure, but it may not
modify Q2 or reuse its tickets.

## Problem

Q2 maintains two incompatible lifecycle notions:

- a predicted hard runtime `FREE/ACTIVE/REFRACTORY` decoder;
- a GT canonical ownership/risk state used for supervision.

Predicted occupancy can make a slot unavailable to a first-crossing GT birth,
causing permanent supervision loss. Suppressing runtime births can then pass a
capacity gate without performing On-TAD.

The new route removes runtime availability from supervision eligibility.

## State Contract

The scientific runtime state contains only prefix-observable inputs, model
outputs, and past immutable commits:

- causal feature memory;
- K latent event carriers;
- continuous model-generated occupancy;
- start posterior with a `before_memory` sentinel;
- accumulated class evidence;
- model-only commit latch;
- source/decision frame identity.

Forbidden state fields include GT instance IDs, canonical slot IDs, assigned
targets, future endpoints, teacher states, runtime target masks, and
calibration labels.

Training and inference must call the same `advance_and_decode`. GT may appear
only in ephemeral target and loss-transport functions.

## Method Sketch

At each prefix:

1. causally update all carriers;
2. predict first-completion mass for occupied carriers;
3. release completed mass before birth/reseed;
4. allow same-bin reseed from released/free mass;
5. update start and class evidence;
6. perform model-only immutable commits from release-before-reseed state;
7. during training only, construct prefix-visible unbalanced OT weights for
   loss, then destroy them without changing runtime state.

All carriers remain in the supervision risk set. Dustbin mass records
unexplained target/carrier mass instead of silently dropping a birth.

## Loss Family

- first-observable birth;
- continuous occupancy;
- first-end/survival hazard with post-end masking;
- OT-weighted class and start losses;
- optional direct-complete synthetic contract;
- adjacent-prefix loss-transport consistency that never becomes runtime
  identity.

Exact equations and constants remain to be frozen in the P0 preregistration.

## Claim Map

| Claim | P0/P3 evidence | Kill condition |
|---|---|---|
| C-A1: train and inference share one model-only transition | state/logit/ledger equality | any GT-dependent field or scientific branch |
| C-A2: GT affects loss only | taint and dataflow audit | annotations change runtime state or emission |
| C-A3: release/reseed avoids Q2-style target exhaustion | repeated/overlap/same-bin synthetic cases | a GT target is discarded because model state is unavailable |
| C-A4: silence cannot pass readiness | learned anti-silence gate | no-emission or always-background obtains PASS |
| C-A5: mechanism improves identity errors beyond temporal MOTR | future matched P3 | reconstruction falls in frozen equivalence region |
| C-A6: efficiency is not the main claim | future profile plus effects | only cost changes |

This table is a historical method hypothesis, not an authorized P0 contract.
Before any model probe, a route-level gate must determine whether C-A1 through
C-A4 are specific to R-A or equally satisfied by B0-B3.

## Novelty Boundary

Route hypothesis: conditionally retained as B4 only.

Publication novelty: not established.

The strongest attack is a one-dimensional MOTR/TrackFormer reconstruction with
interval heads and an online ledger. R-A must eventually beat that matched
reconstruction on duplicate, fragmentation, same-class repetition/overlap,
latency, and recall. Persistent queries, soft occupancy, OT, memory, and a new
loss are individually occupied ingredients.

Equivalence to the matched temporal-MOTR reconstruction is a predeclared route
KILL.

The protocol review narrows the only plausible B4-specific deltas to:

- **D1:** atomic release-before-same-bin-reseed within one transition;
- **D2:** continuous carrier mass plus ephemeral, loss-only, noncanonical
  unbalanced transport.

These are hypotheses, not accepted innovation claims. Persistent queries,
causal memory, interval heads, completion heads, a ledger, soft assignment,
and generic consistency losses are reconstructible ingredients. D1 and D2
must each survive an exact temporal-MOTR reconstruction, a mechanism deletion,
matched observable metrics, structural OOD, and a frozen equivalence test.
Reviewer-proposed numerical margins are not yet adopted.

## Mandatory Route Gates

- outcome-blind, read-only annotation census on the exact split;
- cached-feature encoder strict-causality certificate;
- B0 no-identity prefix set;
- B1 ordinary persistent query;
- B2 temporal TrackFormer/MOTR;
- B3 clean order/risk-set baseline;
- B4 R-A;
- shared inputs, heads, ledger, evaluator, anti-silence, parameter rule, and
  update budget;
- structural OOD, negative controls, and mechanism deletions;
- exact-delta table and frozen temporal-MOTR equivalence region;
- independent route re-review.

See the
[route-identifiability gate](../experiments/prefix-route-identifiability-gate-20260717.md).
The historical
[R-A P0 proposal](../experiments/prefix-shared-event-p0-20260717.md) was
withdrawn before preregistration.

## Non-Claims

- not raw-video end-to-end;
- not a Q2 or CSFSB revision;
- not a publication novelty pass;
- not an authorized P0 design;
- not a cached-feature effectiveness result;
- not a GPU-efficiency result;
- not a new task, event-ID benchmark, mutable final output, or offline cleanup
  method.

## Ordered Gate

```text
executable protocol-only revision
-> independent protocol PASS
-> model-outcome-blind annotation census and cache certificate
-> B0-B4/OOD/negative-control route contract
-> temporal-MOTR exact-delta and equivalence-KILL rule
-> independent route re-review
-> only then decide whether a model P0 contract may be frozen
```

## Sources

- [`../../PRO_PREFIX_ROUTE_PROTOCOL_FREEZE_REVIEW_20260717.md`](../../PRO_PREFIX_ROUTE_PROTOCOL_FREEZE_REVIEW_20260717.md)
- [`../../PRO_PREFIX_ROUTE_PROTOCOL_FREEZE_REVIEW_ABSORPTION_20260717.md`](../../PRO_PREFIX_ROUTE_PROTOCOL_FREEZE_REVIEW_ABSORPTION_20260717.md)
- [`../../PRO_POST_Q2_R1_KILL_ONTAD_ROUTE_REVIEW_20260717.md`](../../PRO_POST_Q2_R1_KILL_ONTAD_ROUTE_REVIEW_20260717.md)
- [`../../PRO_POST_Q2_R1_KILL_ONTAD_ROUTE_REVIEW_ABSORPTION_20260717.md`](../../PRO_POST_Q2_R1_KILL_ONTAD_ROUTE_REVIEW_ABSORPTION_20260717.md)
- [`../../PRO_PREFIX_SHARED_ROUTE_PREIMPLEMENTATION_REVIEW_20260717.md`](../../PRO_PREFIX_SHARED_ROUTE_PREIMPLEMENTATION_REVIEW_20260717.md)
- [`../../PRO_PREFIX_SHARED_ROUTE_PREIMPLEMENTATION_REVIEW_ABSORPTION_20260717.md`](../../PRO_PREFIX_SHARED_ROUTE_PREIMPLEMENTATION_REVIEW_ABSORPTION_20260717.md)
- [DR-047](../decision_register.md#dr-047-revise-and-independently-review-the-protocol-before-r0-or-r1)
- [T40](../discussion_timeline.md#t40-protocol-review-blocks-r0-and-r1-before-collection)
- [DR-046](../decision_register.md#dr-046-revoke-r-a-default-route-authorization-before-implementation)
- [T39](../discussion_timeline.md#t39-preimplementation-review-revokes-r-a-default-route-status)
- historical [DR-045](../decision_register.md#dr-045-authorize-r-a-preregistration-and-cpu-only-p0)
- [T38](../discussion_timeline.md#t38-post-kill-review-selects-a-new-model-only-prefix-filter-for-p0)
- [Q2 capacity audit](../experiments/q2-capacity-lifecycle-audit-20260716.md)

## Connections

[AUTO-GENERATED from graph/edges.jsonl; do not hand-edit]
