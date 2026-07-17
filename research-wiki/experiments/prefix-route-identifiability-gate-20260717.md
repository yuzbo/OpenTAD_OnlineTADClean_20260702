---
type: experiment
node_id: exp:prefix-route-identifiability-gate
title: "Persistent-Carrier On-TAL Route Identifiability Gate"
stage: protocol-v2-review-pending
outcome: not-run
updated: 2026-07-17
---

# Persistent-Carrier On-TAL Route Identifiability Gate

## Status

Protocol V1 verdict: `REVISE_PROTOCOL_BEFORE_COLLECTION`.

Protocol V2 status: `PROTOCOL_REVIEW_PENDING`.

This node records the evidence required before any R-A model P0 contract,
implementation, or training. It is not a model experiment, has not run, and
does not authorize B0-B4 implementation. V1 converted the checklist into code
but its independent review found self-forgeable authorization, asserted
evidence statuses, and incomplete R0-R6 execution. V2 implements the required
repairs but has not yet received a same-reviewer PASS. Therefore neither R0 nor
R1 evidence collection is currently authorized.

GPU authorization: zero hours.

## Gate P: Independent Protocol Review Before Collection

The next legal artifact is one protocol-only immutable V2 commit containing:

- `configs/causaltad/protocols/prefix_route_identifiability_v2.json`;
- `configs/causaltad/protocols/prefix_route_identifiability_v2_manifest.json`;
- `opentad/utils/prefix_route_protocol_v2.py`;
- bound R0, R1, controls, fairness, OOD, and R6 implementations;
- `tools/validate_prefix_route_protocol_v2.py`;
- `tests/test_prefix_route_protocol_v2.py`;
- `PREFIX_ROUTE_EVIDENCE_PROTOCOL_V2.md`;
- the archived V1 independent review.

The same sole reviewer must verify the exact commit/tree and sign canonical
attestation bytes using the pre-frozen Ed25519 identity. The reviewer may
return only:

- `PASS_PROTOCOL_TO_OUTCOME_BLIND_EVIDENCE_COLLECTION`;
- `REVISE_PROTOCOL_BEFORE_COLLECTION`.

A protocol PASS authorizes only the specified read-only R0/R1 collection. It
does not authorize inspecting model predictions or checkpoints, selecting
model constants, implementing B0-B4, running P0, or using a GPU.

Local validation is necessary but not sufficient. The V2 validator reports
`PROTOCOL_V2_VALID_REVIEW_REQUIRED`, and collection authorization returns
`BLOCKED_PENDING_SIGNED_INDEPENDENT_PROTOCOL_REVIEW` until the fixed commit,
manifest, review attestation, and detached signature all verify.

Review-confirmed blockers that the new protocol must close:

- the reporting split has 211 videos while the canonical expectation is 213;
- earlier annotation audits already exposed counts, overlap, and concurrency,
  so R0 can be model-outcome-blind but not annotation-unseen;
- cache-building code appears to select one frame per stride and encode frames
  independently, but the existing cache artifact lacks sufficient provenance
  to bind that code path, encoder revision, weights, environment, and raw
  videos;
- arm semantics, capacity/resource fairness, negative controls, structural
  OOD, and equivalence regions are not executable definitions;
- only the proposed atomic same-bin transition and noncanonical transport
  differ plausibly from a matched temporal MOTR, and neither delta is yet
  established.

V2 implements these choices at the protocol-definition level. It does not claim
that the canonical 213 population certificate exists, that the current cache
passes R1, or that D1/D2 work. Missing population or cache identity fails
closed after protocol PASS rather than being filled with an author guess.

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

## Gate R0: Model-Outcome-Blind Annotation Census

Before examining model outputs, bind:

- exact dataset and split manifest hashes;
- the complete 211-versus-213 reconciliation without silently selecting either
  count;
- annotation bytes and class map hash;
- decision-bin convention and feature stride;
- per-video and per-class event counts;
- same-class repetition frequency and inter-event gaps, with the canonical
  signed gap defined as `next_start - current_end` so positive means
  separation, zero means touching, and negative means overlap;
- same-class and cross-class overlap frequency and duration;
- same-bin end/start frequency;
- actions starting and ending within one decision interval;
- action-duration distribution;
- maximum total and same-class concurrency;
- zero-action stream frequency.

The census must be read-only, deterministic, and blind to model outputs,
predictions, checkpoints, and model-derived thresholds. Its report and source
hashes must be immutable. It describes whether the proposed problem is
present; it does not measure model effectiveness.

The prior-exposure ledger must state that the earlier THUMOS audit already
reported 200/3003 training videos/instances, 211/3325 validation
videos/instances, overlap in 23/32 videos, same-class overlap in 2/3 videos,
and maximum concurrency 2. The new census therefore cannot be described as
annotation-unseen. Any claim-eligibility rule must predeclare the statistical
unit, denominator, minimum support, multiplicity treatment, target effect,
power method, and terminal action. Values proposed by the reviewer are
candidate policies until independently justified and frozen.

Kill or revise conditions:

- the exact split cannot be bound;
- the intended hard cases fail the frozen support and power rule;
- K or direct-complete assumptions were chosen before the census;
- census definitions are altered after model results.

## Gate R1: Cached-Feature Strict-Causality Certificate

Bind:

- encoder repository, immutable revision, source commit, processor,
  configuration, weights, and environment;
- raw-video identities and content hashes;
- frame sampling and preprocessing;
- clip construction, padding, stride, center/end alignment, and batching;
- exact temporal receptive field for every emitted feature token;
- mapping from raw frames to token source and decision times;
- proof or executable audit that no token uses a frame after its decision time;
- extraction command, extraction-code commit, feature manifest, and content
  hashes;
- a machine-checkable per-token support map from source frames to decision
  time.

The existing cache is `UNVERIFIED_EXISTING_CACHE` until its artifact identity
is linked to this certificate. Static code inspection, same-environment
perturbation invariance, and historical-artifact linkage are separate tests.
A CPU re-extraction is not required to be byte-identical to a historical GPU
FP16 artifact.

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
| B0 | fresh prefix queries over the same causal visual memory, with no cross-prefix query identity | tests whether persistent identity is needed |
| B1 | always-persistent queries with matched heads but no birth pool, hard lifecycle, or R-A transport | tests whether persistence alone is sufficient |
| B2 | newborn plus propagated temporal track queries with a frozen TrackFormer/MOTR-style lifecycle | strongest obviousness and reconstructibility attack |
| B3 | clean hard lifecycle and risk-set baseline with no GT-written persistent identity | tests whether a simple correct lifecycle is sufficient without reopening Q2 |
| B4 | R-A Prefix-Shared Latent Event Filter | candidate route, not default method |

Before implementation, freeze principles for:

- identical causal inputs and decision times;
- identical output heads where the model family permits;
- identical append-only ledger and evaluator;
- identical no-future and anti-silence requirements;
- whether the study is capacity-matched, resource-matched, or reports both,
  with exact parameter, MAC, memory, update, and token tolerances;
- identical train, validation, and hidden structural-OOD generators;
- identical reporting and terminal-failure policy.

The contract may later specify model details only after an independent route
review authorizes that step.

## Gate R3: Negative Controls

The comparison must include:

1. count-only predictor;
2. template-timing predictor;
3. feature-time shuffle preserving count and label marginals;
4. a frozen semantic-destruction control that preserves the declared
   low-order marginals;
5. ledger-only scripted deduplicator;
6. history-off model;
7. conservative threshold sweep with recall and latency accounting.

These controls must fail readiness or lose the preregistered structural-OOD
metrics. Otherwise the benchmark cannot identify lifecycle learning.

A globally consistent class-label permutation only renames classes and is not
a valid semantic-destruction control. A threshold sweep is a sensitivity
analysis, not an independent learned-control arm.

## Gate R4: Mechanism Deletions

For any future B4 contract, predeclare:

- UOT to Hungarian plus dustbin;
- UOT to partial matching;
- continuous occupancy removed;
- adjacent-prefix consistency removed;
- atomic release-before-same-bin-reseed removed;
- latch excluded from neural transition;
- model-output-only latch included in neural transition;
- start posterior to scalar start regression;
- direct-complete disabled.

The final protocol must distinguish core mechanism deletions from conditional
tests: K/concurrency belongs in structural stress; start and direct-complete
tests are included only if those mechanisms remain in the frozen B4.

No mechanism may be called necessary unless its deletion removes a
preregistered observable advantage without improving another primary metric by
an equivalent amount.

## Gate R5: Structural OOD

Random seeds under one generator are insufficient. Hidden validation must
include combinations not observed in training:

- event counts;
- action lengths;
- repetition intervals using the R0 signed-gap convention;
- overlap graphs;
- same-bin end/start patterns;
- class permutations;
- feature-basis rotations or mixtures;
- noise families;
- endpoint-delay distributions;
- multiple simultaneous shifts.

The validation generator, distributions, hidden seeds, and content hashes must
be frozen independently before model implementation.

The protocol must separate four factor families: lifecycle geometry, semantic
mapping, representation/noise, and endpoint observation delay. A seed change
within one generator is not structural OOD.

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

The only currently plausible B4-specific deltas are:

- **D1:** one atomic transition that releases completion mass before allowing
  same-bin reseeding;
- **D2:** continuous carrier mass with ephemeral, loss-only, noncanonical
  unbalanced transport.

Persistent queries, causal memory, interval/class/completion heads, an
append-only ledger, generic soft assignment, and loss terms are not accepted
deltas by themselves. D1 and D2 remain hypotheses until the exact B2
reconstruction, deletion tests, and numerical equivalence region are frozen.
Reviewer-proposed margins are candidates, not adopted constants.

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
| Commit V2 and obtain same-reviewer signed protocol decision | `ALLOW` |
| New annotation census | `BLOCKED_PENDING_PROTOCOL_PASS` |
| New feature-provenance or causality audit | `BLOCKED_PENDING_PROTOCOL_PASS` |
| Inspect model outcomes, predictions, or checkpoints | `BLOCKED` |
| Freeze R-A model/UOT/loss/threshold constants | `BLOCKED` |
| Implement B0-B4 models | `BLOCKED` |
| Run the historical R-A P0 | `BLOCKED` |
| Real-data effectiveness | `BLOCKED` |
| GPU/profile/formal/raw-video work | `BLOCKED` |
| GPU hours | `0` |

## Sources

- [`../../PREFIX_ROUTE_EVIDENCE_PROTOCOL_V2.md`](../../PREFIX_ROUTE_EVIDENCE_PROTOCOL_V2.md)
- [`../../configs/causaltad/protocols/prefix_route_identifiability_v2.json`](../../configs/causaltad/protocols/prefix_route_identifiability_v2.json)
- [`../../PRO_PREFIX_ROUTE_PROTOCOL_V1_INDEPENDENT_REVIEW_20260717.md`](../../PRO_PREFIX_ROUTE_PROTOCOL_V1_INDEPENDENT_REVIEW_20260717.md)
- [`../../PREFIX_ROUTE_EVIDENCE_PROTOCOL_V1.md`](../../PREFIX_ROUTE_EVIDENCE_PROTOCOL_V1.md)
- [`../../configs/causaltad/protocols/prefix_route_identifiability_v1.json`](../../configs/causaltad/protocols/prefix_route_identifiability_v1.json)
- [`../../opentad/utils/prefix_route_protocol.py`](../../opentad/utils/prefix_route_protocol.py)
- [`../../tools/validate_prefix_route_protocol.py`](../../tools/validate_prefix_route_protocol.py)
- [`../../tests/test_prefix_route_protocol.py`](../../tests/test_prefix_route_protocol.py)
- [`../../PRO_PREFIX_ROUTE_PROTOCOL_FREEZE_REVIEW_20260717.md`](../../PRO_PREFIX_ROUTE_PROTOCOL_FREEZE_REVIEW_20260717.md)
- [`../../PRO_PREFIX_ROUTE_PROTOCOL_FREEZE_REVIEW_ABSORPTION_20260717.md`](../../PRO_PREFIX_ROUTE_PROTOCOL_FREEZE_REVIEW_ABSORPTION_20260717.md)
- [`../../PRO_PREFIX_SHARED_ROUTE_PREIMPLEMENTATION_REVIEW_20260717.md`](../../PRO_PREFIX_SHARED_ROUTE_PREIMPLEMENTATION_REVIEW_20260717.md)
- [`../../PRO_PREFIX_SHARED_ROUTE_PREIMPLEMENTATION_REVIEW_ABSORPTION_20260717.md`](../../PRO_PREFIX_SHARED_ROUTE_PREIMPLEMENTATION_REVIEW_ABSORPTION_20260717.md)
- [DR-047](../decision_register.md#dr-047-revise-and-independently-review-the-protocol-before-r0-or-r1)
- [T40](../discussion_timeline.md#t40-protocol-review-blocks-r0-and-r1-before-collection)
- [DR-049](../decision_register.md#dr-049-accept-the-independent-v1-revise-and-preserve-the-collection-block)
- [DR-050](../decision_register.md#dr-050-freeze-protocol-v2-for-same-reviewer-reassessment)
- [T43](../discussion_timeline.md#t43-protocol-v2-closes-v1-bypasses-and-awaits-the-same-reviewer)
- [DR-046](../decision_register.md#dr-046-revoke-r-a-default-route-authorization-before-implementation)
- [T39](../discussion_timeline.md#t39-preimplementation-review-revokes-r-a-default-route-status)
- [idea:prefix-shared-latent-event-filter](../ideas/prefix-shared-latent-event-filter.md)

## Connections

[AUTO-GENERATED from graph/edges.jsonl; do not hand-edit]
