# Prefix-Route Evidence Protocol V1

## Status

Protocol state: `PROTOCOL_REVIEW_PENDING`.

Current authorization:

- edit and validate protocol artifacts;
- request one independent protocol review;
- use zero GPU hours.

Current blocks:

- no new R0 annotation census;
- no new R1 cache audit;
- no model outcomes or checkpoints;
- no B0-B4 model contract or implementation;
- no P0, effectiveness, profile, or training.

The machine-readable source of truth is
[`configs/causaltad/protocols/prefix_route_identifiability_v1.json`](configs/causaltad/protocols/prefix_route_identifiability_v1.json).
This page is an operating summary and cannot override the JSON.

## Frozen Decisions

### Reporting Population

The only possible primary reporting population is `canonical_reporting_213`.
It becomes usable only when a source-derived certificate:

1. binds the historical 211-ID and canonical 213-ID files;
2. explains every differing ID with an allowed reason code;
3. has no unexplained ID;
4. reconciles the set cardinalities exactly;
5. binds the protocol and independent PASS certificate.

Until then the action is `BLOCK_R0_POPULATION_UNRESOLVED`. Historical 211 is
audit-only and cannot become an alternative primary population. The remote
manifest was not available while V1 was written, so no two-ID difference is
claimed as verified.

### R0 Annotation Census

R0 is model-outcome-blind, not annotation-unseen. The protocol freezes:

- frame conversion and stride-8 decision bins;
- half-open `[start,end)` intervals;
- touching as non-overlap;
- signed repeated-instance gap as `next_start-current_end`;
- same-bin end/start and direct-complete definitions;
- Ambiguous exclusion with separate disclosure;
- video-cluster statistics and 10,000-resample uncertainty;
- aggregate-only author output and reviewer-only hard-case detail;
- atomic, no-overwrite evidence publication.

A stress family is `PRIMARY_ELIGIBLE` only if it has at least 30 independent
videos, 100 GT instances, 5% of relevant positive videos, three classes, and
power at least 0.8 under the frozen conservative design-effect calculation.
The calculation uses a 10-point absolute recall effect, family-wise alpha
0.05 across four families, worst-variance baseline rate 0.5, and ICC 0.2.
These are protocol operating choices, not observed dataset facts.

### R1 Cache Certificate

R1 proves only decoded-RGB-frame to feature-token causality. It does not prove
compressed-bitstream or wall-clock decoding causality.

A PASS requires:

- exact Hugging Face snapshot, processor, config, and weight hashes;
- raw-video, annotation, cache-manifest, and feature-array hashes;
- extraction command, source commit, environment, and software versions;
- one support record per token;
- `max(support_frames) <= decision_frame`;
- `support_frames = [decision_frame]` for the current single-frame path;
- deterministic future-frame, batch-content, batch-order, and input-use
  perturbation checks;
- a separate cryptographic link to the existing cache.

Cross-environment CPU output is not required to byte-match historical GPU
FP16 output. Missing cache identity produces `FAIL_UNVERIFIABLE`; it does not
authorize creation of a replacement cache.

### B0-B4

The five arms now have non-overlapping roles:

| Arm | Frozen role |
|---|---|
| B0 | Fresh per-decision queries with the same causal visual memory and no query identity |
| B1 | Always-persistent queries without birth pools, hard lifecycle, UOT, continuous occupancy, or reseed |
| B2 | Temporal MOTR reconstruction with newborn and propagated track queries |
| B3 | New clean hard model-only lifecycle with Hungarian plus dustbin |
| B4 | Candidate route containing only the D1/D2-specific mechanisms |

Primary comparisons require both capacity and resource matching. B2 supplies
the trainable-parameter anchor; all arms must be within 5% trainable
parameters and 10% training MAC, with identical tokens, optimizer events,
trial count, calibration algorithm, ledger, evaluator, and three seeds.
Numeric model budgets remain blocked and must be bound in a later model-P0
contract before implementation.

### Identifiability

The protocol replaces global class renaming with a deterministic
instance-label derangement that preserves the global label multiset while
changing every instance label. Threshold sweeps are sensitivity analyses, not
negative controls.

Core B4 deletions are:

1. UOT to Hungarian plus dustbin;
2. continuous mass to binary validity;
3. adjacent-prefix consistency off;
4. atomic same-bin reuse off;
5. latch excluded from the neural transition.

Structural OOD is frozen into event topology, temporal geometry, semantic
mapping, and observation distribution, with IID, single-shift, and
reviewer-owned compound sets.

### Exact Novelty Boundary

Only two route-level deltas remain:

- **D1:** completion releases carrier mass and permits same-bin birth against
  that released capacity in one atomic transition;
- **D2:** runtime uses model-only continuous mass, while a dustbin-capable,
  noncanonical soft coupling exists only for the current loss and is destroyed
  immediately.

Persistent queries, memory, interval heads, completion heads, and the ledger
are explicitly `NO_DELTA`.

B4 is killed if it is equivalent or inferior to B2. Survival requires
noninferior mOnlineAP and event recall, no primary-metric degradation beyond
the frozen margin, and at least one D1/D2-linked lifecycle improvement whose
simultaneous confidence interval clears its margin.

## Commands

Validate the protocol:

```powershell
python tools/validate_prefix_route_protocol.py validate-protocol
```

Confirm that collection is currently blocked:

```powershell
python tools/validate_prefix_route_protocol.py authorize-collection
```

The second command must exit nonzero with
`BLOCKED_PENDING_INDEPENDENT_PROTOCOL_REVIEW` until a committed protocol is
bound to an independent
`PASS_PROTOCOL_TO_OUTCOME_BLIND_EVIDENCE_COLLECTION` certificate.

## Next Gate

The independent reviewer receives the committed protocol, validator, tests,
the raw protocol review, and its absorption. The reviewer returns exactly one
of:

- `PASS_PROTOCOL_TO_OUTCOME_BLIND_EVIDENCE_COLLECTION`;
- `REVISE_PROTOCOL_BEFORE_COLLECTION`.

A PASS authorizes only read-only R0/R1 collection. It does not authorize model
code, model P0, GPU work, or training.
