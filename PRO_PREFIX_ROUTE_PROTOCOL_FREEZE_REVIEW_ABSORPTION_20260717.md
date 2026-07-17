# Prefix-Route Protocol-Freeze Review: Independent Absorption

Date: 2026-07-17 Asia/Shanghai
Source file: `PRO_PREFIX_ROUTE_PROTOCOL_FREEZE_REVIEW_20260717.md`
Source attachment SHA-256:
`F9064827B05991FD63156770B2A1D6D6D40C3E53A73F4113B31F312B17F82BD0`
Source size: 47,878 bytes, 971 text lines
Reviewed repository anchor:
`b70baf3bb5d819d1df25ca47534c5d3bfbb6e695`

## Absorption Verdict

Accept the final decision `REVISE_PROTOCOL_BEFORE_COLLECTION`.

The previous route gate correctly identified the categories of evidence needed
before model work, but it did not freeze exact populations, definitions,
statistics, fairness algorithms, equivalence rules, disclosure, or terminal
actions. Therefore:

> No new R0 annotation census or R1 cache audit may begin yet. First produce a
> protocol-only immutable commit that closes the review's blocking definitions,
> then obtain a new independent read-only
> `PASS_PROTOCOL_TO_OUTCOME_BLIND_EVIDENCE_COLLECTION`.

This decision is narrower than a route kill:

- standard On-TAL retains conditional research value;
- the persistent-carrier field gap remains `NOT_ESTABLISHED`;
- R-A remains B4 only;
- Q2/R1 remains terminal `KILL`;
- the historical R-A P0 remains withdrawn;
- model contracts, model code, predictions, checkpoints, effectiveness
  experiments, profile, formal training, visual fine-tuning, and raw-video
  training remain blocked;
- GPU hours authorized remain zero.

## Independently Verified Repository Facts

### Reporting population conflict

`configs/causaltad/thumos_pes_q2_base.py` simultaneously binds:

- `reporting_locked_211.txt`;
- `expected_historical_count=211`;
- `canonical_expected_count=213`;
- a `reporting_211_vs_213.json` comparison artifact.

The conflict is real and blocks a unique R0 population. The review's
recommendation to prefer canonical 213 is plausible, but this absorption does
not mark 213 as selected until the external manifests and exact two-ID
difference are independently bound.

### Prior annotation exposure

`tools/analyze_ontad_instances.py` reports duration, per-video instance count,
maximum total/same-class concurrency, and videos with overlap. It does not
report pair-level overlap duration, repetition gaps, same-bin transitions,
short actions, class-map identity, split identity, or a blind disclosure
contract.

The existing wiki record already exposes:

- training: 200 videos, 3,003 instances, 23 overlap videos, 2 same-class
  overlap videos;
- historical validation: 211 videos, 3,325 instances, 32 overlap videos, 3
  same-class overlap videos;
- maximum total and same-class concurrency of two.

Consequently, a new R0 can be blind to B0-B4 outcomes, but cannot honestly be
called annotation-unseen. A prior-exposure ledger is mandatory.

### Cached-feature causality and identity

`tools/cache_ontad_features.py` selects exactly the most recent frame from each
stride-8 packet. It passes those selected frames as an image batch to
`OnlineSigLIPFrameEncoder`.

`OnlineSigLIPFrameEncoder` reshapes `[B,T,C,H,W]` to independent images before
the vision encoder. The cache builder leaves `use_motion_branch=False`, so the
static source path supports:

```text
support(z[v,j]) = {decision_frame[v,j]}
```

for the executed source implementation.

However, the builder passes no Hugging Face `revision`, and the cache manifest
records only an encoder name, annotation hash, stride, source frames,
feature-array hashes, dimensions, and dtype. It does not bind the exact model
snapshot, processor/config bytes, weight shards, extraction commit,
raw-video bytes, decoder/preprocessing environment, or support-map hash.

Thus:

- algorithmic single-frame causality is strongly supported;
- the existing cache artifact's exact identity and reproducibility remain
  unverified;
- source-frame monotonicity alone cannot yield R1 PASS.

## Accepted Protocol Corrections

### R0

- resolve 211 versus 213 before execution;
- freeze dataset-role and exact population hashes;
- use half-open intervals and explicit decision-bin formulas;
- report pair-, instance-, video-, class-, and duration-level overlap
  quantities without treating pairs as independent videos;
- define repetition, same-bin end/start, one-bin actions, Ambiguous exclusion,
  and zero-action videos before collection;
- add clustered uncertainty and predeclared claim-eligibility rules;
- rename the process `model-outcome-blind`;
- publish aggregate author-visible output and commit reviewer-only detail;
- forbid predictions, checkpoints, thresholds, and model work directories from
  the census environment.

### R1

- bind extraction code, command, environment, Hugging Face revision,
  config/processor, weight shards, raw-video bytes, annotation, cache, and
  per-token support map;
- prove `max(support(z[v,j])) <= decision_frame[v,j]`;
- keep compressed-bitstream latency outside the current decoded-RGB claim;
- fail closed if the current cache cannot be linked to exact source bytes and
  extraction provenance;
- do not automatically authorize creation of a replacement cache.

### R2-R6

- uniquely define B0 fresh queries, B1 always-persistent queries, B2 temporal
  MOTR, B3 clean order/risk lifecycle, and B4 R-A;
- give B0 the same causal visual history rather than crippling it;
- share the prediction interface, ledger, evaluator, split, token order, and
  calibration algorithm;
- freeze parameter/update/token/tuning fairness before model code;
- turn negative controls into exact input-permission contracts;
- reduce mechanism deletions to candidate exact deltas;
- reduce structural OOD to event-topology, temporal-geometry, semantic-map,
  and observation-distribution families;
- define a numerical equivalence region and simultaneous paired inference
  before model results;
- kill R-A if it is equivalent to temporal MOTR on all primary metrics.

## Current R-A Delta Boundary

The review correctly classifies the following as non-deltas:

- persistent query renamed as carrier;
- interval, class, or generic start head;
- immutable shared ledger;
- generic memory or matching.

Only two route-level candidates remain:

- D1: atomic release before same-bin reseed;
- D2: continuous carrier mass with ephemeral, noncanonical, loss-only UOT.

These are not accepted innovations. They are hypotheses that must be written as
an exact state-transition or estimand difference from temporal MOTR before
model implementation. If neither survives that exercise, the carrier route
should be killed before model P0.

## Qualifications: Recommendations Not Yet Frozen

The review contains useful proposed constants, but its own verdict says a
review response cannot replace a hashed repository protocol. The following are
therefore candidate amendments, not accepted frozen constants.

### Population selection

`canonical_213` should be preferred only after the canonical manifest, raw
annotation identity, missing/extra IDs, and eligibility rules are verified.
The historical 211 population should then remain a disclosed difference audit,
not an alternative main population.

### Repetition-gap sign

The review writes both `e_i - s_(i+1)` and the opposite convention. Freeze only
one:

```text
gap(i,i+1) = start(i+1) - end(i)
```

Positive means separation, zero means touching, and negative means overlap.

### Claim-eligibility thresholds

The proposed 30 videos, 100 instances, 5%, three classes, 10-point effect, and
80% power rule is a conservative candidate policy. The final protocol must
specify:

- the primary sampling unit;
- clustered data-generating assumptions;
- multiplicity family;
- effect-size justification;
- how a subset with many instances but few videos is classified.

The historically exposed 2/3 same-class-overlap videos are clearly
descriptive-only regardless of the final reasonable threshold.

### Equivalence margins

The proposed `0.005` mOnlineAP, `max(0.01, 2/N)` rate margin, stress margin,
and one-bin latency margin are not self-justifying. The protocol must define
what N counts, metric scale, simultaneous metric family, bootstrap algorithm,
bootstrap seed, seed aggregation, and whether margins represent measurement
resolution, practical equivalence, or power.

### Cache perturbation equality

Separate three tests:

1. source-code support proof;
2. same-environment future/batch perturbation invariance;
3. existing-cache byte linkage.

The existing cache was produced in a GPU extraction job and stored as
float16. A CPU re-extraction need not be byte-identical to GPU output. Exact
equality is valid only within a frozen deterministic environment that can
actually reproduce it; otherwise the protocol needs a predeclared numerical
tolerance and still must fail artifact linkage if the original cache cannot be
authenticated.

### Label-feature permutation

A single global class permutation applied consistently to both training and
evaluation merely renames classes and does not destroy semantic learnability.
The final control must specify a construction that breaks feature-label
dependence while preserving the intended marginals, without turning the test
into an arbitrary train/test label mismatch.

### Parameter and compute matching

The proposed parameter `+/-5%` and MAC `+/-10%` bands are candidate governance
rules. Dummy parameters are correctly forbidden, but architecture families
may not admit exact width matching without changing their mechanism. The
protocol should predeclare whether the primary comparison is
capacity-matched, resource-matched, or reported under both views.

## Current Authorization Matrix

| Action | Status |
|---|---|
| Archive and absorb this review | `ALLOW` |
| Modify and commit protocol documents only | `ALLOW` |
| Run a new R0 census | `BLOCKED_PENDING_PROTOCOL_REVIEW` |
| Run a new R1 cache audit | `BLOCKED_PENDING_PROTOCOL_REVIEW` |
| Read predictions/checkpoints/effect reports | `BLOCKED` |
| Freeze B0-B4 model contracts | `BLOCKED` |
| Implement B0-B4 or historical R-A P0 | `BLOCKED` |
| Real-data effectiveness or GPU work | `BLOCKED` |
| GPU hours | `0` |

## Next Legal Sequence

```text
archive and absorb review
-> write protocol-only revision and prior-exposure ledger
-> bind all protocol files in one immutable manifest
-> independent read-only protocol review
-> at most PASS_PROTOCOL_TO_OUTCOME_BLIND_EVIDENCE_COLLECTION
-> collect only R0/R1 evidence
-> independent route-evidence review
-> at most PASS_AUTHORIZE_MODEL_P0_CONTRACT_FREEZE
```

No step directly authorizes model code or GPU work.

## Machine-Readable Absorption

```json
{
  "source_sha256": "F9064827B05991FD63156770B2A1D6D6D40C3E53A73F4113B31F312B17F82BD0",
  "reviewed_commit": "b70baf3bb5d819d1df25ca47534c5d3bfbb6e695",
  "final_verdict_accepted": "REVISE_PROTOCOL_BEFORE_COLLECTION",
  "word_for_word_acceptance": false,
  "q2_r1_terminal_kill": true,
  "r_a_role": "B4_CANDIDATE_ONLY",
  "r0_collection_allowed": false,
  "r1_collection_allowed": false,
  "existing_cache_algorithmic_support": "LIKELY_SINGLE_FRAME_CAUSAL",
  "existing_cache_artifact_identity": "UNVERIFIED",
  "reporting_population": "UNRESOLVED_211_VS_213",
  "candidate_exact_deltas": [
    "ATOMIC_RELEASE_BEFORE_SAME_BIN_RESEED",
    "CONTINUOUS_MASS_WITH_EPHEMERAL_NONCANONICAL_UOT"
  ],
  "review_numeric_recommendations_frozen": false,
  "protocol_only_revision_allowed": true,
  "model_contract_allowed": false,
  "model_implementation_allowed": false,
  "gpu_hours_authorized": 0
}
```
