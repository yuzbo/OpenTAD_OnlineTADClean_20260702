# Prefix-Route Evidence Protocol V2

## Current Status

Protocol V1 received the independent verdict:

`REVISE_PROTOCOL_BEFORE_COLLECTION`

The full review is archived in
`PRO_PREFIX_ROUTE_PROTOCOL_V1_INDEPENDENT_REVIEW_20260717.md`.

Protocol V2 is therefore still:

`PROTOCOL_REVIEW_PENDING`

It does not authorize R0, R1, model implementation, profiling, or training.

## What V2 Changes

V2 closes the two P0 defects found in V1.

First, collection authorization cannot consume a caller-created dictionary.
The public authorization function must re-read:

1. canonical Protocol V2 bytes;
2. the canonical source manifest;
3. the current Git commit and tree;
4. the independent review attestation;
5. the detached Ed25519 signature.

The reviewer identity and public key were fixed before V2 review. The same sole
reviewer retains the private key outside the repository.

Second, population, R0, and R1 status values are computed from contained source
evidence. A submitted `PASS` string is never sufficient.

## V1 Review Response Map

| V1 finding | V2 disposition |
|---|---|
| P0-1, self-forgeable review | Freeze one reviewer ID and Ed25519 public key; sign canonical attestation bytes; bind protocol, manifest, commit, and tree; re-read files in every authorization path. |
| P0-2, asserted evidence | Re-read contained ID lists, reasons, annotations, class maps, raw videos, arrays, support rows, and perturbations; derive all statuses. |
| P1-1, shallow semantics | Validate exact nested policy values, lock all policy sections by hash, and reject each forbidden relaxation in adversarial tests. |
| P1-2, ambiguous R0 | Bind an executable collector with fixed interval, endpoint, duplicate, tie, pair, denominator, bootstrap, and power rules plus synthetic edge cases. |
| P1-3, annotation exposure | Mark THUMOS reporting annotations as design-exposed, maintain an append-only exposure ledger, and require a separately frozen annotation-unseen confirmatory dataset for such a claim. |
| P1-4, weak fairness/isolation | Make B2 a faithful newborn-plus-propagated-query attack; derive capacity/resource fairness; require A4 for D1 and joint A1+A2 for D2. |
| P1-5, nonconstructive controls/OOD | Bind executable shuffle, derangement, and R5 generators with seeds, valid-cross rules, content hashes, and failure actions. |
| P1-6, incomplete R6 | Bind evaluator fields and metric sources, use crossed paired resampling with full multiplicity, and define an indeterminate terminal result. |
| P2-1, incomplete source binding/tests | Bind all dependencies in an exact source manifest and frozen Git tree; use real temporary artifacts and hostile substitution tests. |

These are protocol implementations, not positive scientific evidence. R0, R1,
B0-B4, D1, D2, and route effectiveness remain unknown.

## Task Boundary

The task remains fully supervised, strict-causal online temporal action
localization. At an emission, the model provides:

```text
emission_id
stream_key
sequence_id
start
end
class
score
source_frame
emit_frame
```

`source_frame` is the internal decision frame. `emit_frame` is the immutable
ledger commit frame. Future frames, future ground truth, output revision, NMS,
merging, and offline cleanup are forbidden.

## Reporting-Set Exposure

THUMOS14 reporting annotations are explicitly designated:

`DESIGN_EXPOSED_ROUTE_SELECTION_AND_BENCHMARK`

The project may not describe them as annotation-unseen. A final confirmatory
claim requires a separately frozen annotation-unseen dataset selected before
model outcomes are inspected.

The append-only exposure ledger is:

`research-wiki/evidence/prefix-route-prior-exposure-ledger.jsonl`

## Population Gate

The historical 211-video set remains audit-only. The canonical 213-video set
can become the sole reporting population only after the validator re-reads:

- the sorted historical ID list;
- the sorted canonical ID list;
- one reason row for every differing ID;
- every cited reason-source artifact.

Counts, differences, reason coverage, and status are derived by the validator.

## R0 Gate

The R0 collector directly parses annotation and class-map bytes.

Its fixed boundary rules include:

- half-open intervals;
- no clipping;
- explicit exclusion of `Ambiguous`;
- source failure for unknown legal labels;
- exclusion of all members of an exact duplicate group;
- first previous observation count equal to zero;
- terminal endpoints represented by `frame_count`;
- end-before-start concurrency ties;
- each unordered pair counted once;
- separate same-bin handoff/touching and overlap-transition counts;
- consecutive start-sorted same-class gaps;
- video-cluster type-7 percentile bootstrap with 10,000 resamples.

The author report contains aggregates and no video IDs. Validation recomputes
the complete report and requires exact canonical-byte equality.

## R1 Gate

R1 is limited to decoded RGB frames through one feature token. It does not
claim compressed-stream or wall-clock decoding causality.

The validator re-reads:

- extractor source and resolved command;
- environment and software locks;
- model and processor configs;
- every weight shard;
- annotation and cache manifests;
- every raw video;
- every feature array;
- the complete support map;
- the dynamic perturbation record.

It loads each NumPy feature array and checks its geometry, dtype, source frames,
and hash. It derives the perturbation sample from the support map and compares
the actual float32 token bytes. Missing identity or a no-op mutation yields
`FAIL_UNVERIFIABLE`.

For every nonterminal token, the future intervention inverts the native decoded
RGB suffix after the decision frame. A terminal token has no native future
suffix, so it uses two equal-shape, deterministically paired batches that differ
only in a synthetic continuation appended after end of stream. The continuation
is an audit intervention, never a model input used for training or evaluation.
Both modes require exact target-token invariance; changing the target support
frame must still change that token.

## Arm Fairness

B2 is a faithful temporal tracking attack with newborn object queries and
propagated track queries present at each decision. It is not restricted to
capacity that happened to be free before the decision.

All B0-B4 arms must match B2 on:

- trainable parameters;
- training MACs;
- inference MACs;
- live causal-state bytes;
- peak training and inference memory;
- median and p95 latency;
- optimizer events;
- effective tokens;
- accumulation;
- trial count;
- calibration population;
- seed count.

Unused trainable padding is forbidden. Every trainable parameter must appear in
the forward path and receive a gradient in the contract smoke test.

## Controls and OOD

Feature-time shuffle and semantic derangement are executable functions.
Semantic derangement:

- operates within one frozen split;
- preserves the global label multiset;
- changes every instance label;
- forbids a global consistent class rename;
- fails if no legal construction exists.

The R5 generator fixes 64-bin sequences, 16-dimensional observations, four
classes, event topology, temporal geometry, semantic mapping, observation
distribution, valid crosses, balancing, seeds, and canonical scientific-content
hashes.

## R6 Decision

R6 uses exact evaluator field names and crossed paired resampling:

- one global seed resample is shared across every video and arm;
- one video resample is shared across every arm;
- mOnlineAP uses paired global training seeds;
- additive lifecycle metrics use paired video and seed units;
- multiplicity includes every registered contrast and eligible metric.

B4 survives only when:

1. it is noninferior to B2 on all global metrics; and
2. D1 is established by the A4 deletion, or D2 is established by the joint
   A1+A2 deletion; and
3. negative controls do not identify an uninformative benchmark.

An inconclusive interval gives `INDETERMINATE_NO_ROUTE_CLAIM`. It does not
automatically authorize additional seeds.

## Commands

Build the source manifest before committing:

```powershell
python tools/validate_prefix_route_protocol_v2.py build-manifest
```

Validate the unsigned protocol package:

```powershell
python tools/validate_prefix_route_protocol_v2.py validate-protocol
```

Without a signed independent PASS, authorization must exit with code 2:

```powershell
python tools/validate_prefix_route_protocol_v2.py authorize-collection
```

After the same reviewer signs a fixed commit, use:

```powershell
python tools/validate_prefix_route_protocol_v2.py authorize-collection `
  --review-attestation <review.json> `
  --review-signature <review.sig>
```

Even a valid PASS authorizes only outcome-blind R0 and R1.
