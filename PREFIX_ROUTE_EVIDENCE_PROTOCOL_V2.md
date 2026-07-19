# Prefix-Route Evidence Protocol V2

## Current Status

Protocol V1 received the independent verdict:

`REVISE_PROTOCOL_BEFORE_COLLECTION`

The full review is archived in
`PRO_PREFIX_ROUTE_PROTOCOL_V1_INDEPENDENT_REVIEW_20260717.md`.

The same reviewer then reassessed commit
`18cc27b8496f5effa7e4e93abca58da4e7ea3d0b` and again returned:

`REVISE_PROTOCOL_BEFORE_COLLECTION`

That review is archived byte-identically in
`PRO_PREFIX_ROUTE_PROTOCOL_V2_METHOD_REASSESSMENT_20260717.md`.

The same reviewer then reproduced six false-PASS paths at commit
`b0ac8b1c1d29fc252b472a6dc64c854524426593` and returned:

`REVISE_PROTOCOL_BEFORE_COLLECTION`

That review is archived in
`PRO_PREFIX_ROUTE_PROTOCOL_V2_ROUND3_INDEPENDENT_REVIEW_20260717.md`.

The current remediation candidate remains:

`PROTOCOL_REVIEW_PENDING`

An exact-commit Gate-A audit of commit
`48e78e43a77822ee0e51cace9ad758ca935c1450` subsequently returned:

`REVISE_PREFIX_PROTOCOL_BEFORE_ANY_CT_WORK`

That review and the repository-side absorption are archived as
`PRO_PREFIX_CT_CROSS_ROUTE_GATE_A_REVIEW_20260720.md` and
`PRO_PREFIX_CT_CROSS_ROUTE_GATE_A_REVIEW_ABSORPTION_20260720.md`.
The present changes are a new, unreviewed Gate-A closure candidate. Local tests
cannot replace a fixed-commit independent review.

For this closure review, the human-readable Gate-A headline
`PASS_PREFIX_PROTOCOL_FOR_ROUTE_EVIDENCE_DESIGN` maps exactly to the signed
V2 attestation field
`verdict=PASS_PROTOCOL_TO_OUTCOME_BLIND_EVIDENCE_COLLECTION`. A PASS response
must state both tokens. The latter is the only machine-consumable token; both
carry the same narrow scope described below and neither authorizes model work
or GPU use.

It does not authorize population evidence, R0, R1, model implementation,
profiling, or training. Even a future protocol PASS can initially authorize
only read-only source-identity registration because the authoritative THUMOS14
annotation, historical 211-file inventory, and R1 extractor snapshot have not
yet been registered in a reviewed commit.

## What The Remediation Changes

The current candidate retains V2's cryptographic review and Git/source binding,
then removes the scientific false-PASS paths found in the V2 reassessment.

First, collection authorization cannot consume a caller-created dictionary.
The public authorization function must re-read:

1. canonical Protocol V2 bytes;
2. the canonical source manifest;
3. the current Git commit and tree;
4. the independent review attestation;
5. the detached Ed25519 signature.

The reviewer identity and public key were fixed before V2 review. The same sole
reviewer retains the private key outside the repository.

The key changes are:

1. Population membership is derived only from registered authoritative
   annotation bytes, an exact reviewed source-origin attestation, an official
   per-video content manifest, a registered historical file inventory, and
   every actual historical artifact. Caller-supplied 211/213 lists and reason
   prose no longer exist.
2. R0 is fixed to the `validation` subset and the exact
   `DESIGN_EXPOSED_ROUTE_SELECTION_AND_BENCHMARK` disclosure. Its former
   heuristic power label is removed; R0 prevalence is descriptive only. The
   low-level collector emits diagnostics only. Only the canonical
   protocol/manifest/review-bound wrapper can issue an R0 status.
3. R1 must recreate the registered local-only SigLIP snapshot, decode every
   video, verify the canonical command and active software versions, rerun
   every cache row, and rerun the dynamic transcript. The current unregistered
   binding blocks R1 before reading author-crafted evidence.
4. B2 is an executable temporal-MOTR state machine, not a prose baseline. Its
   indivisible per-stream state owns the observation clock, track serial, and
   emission sequence, and its integer Hungarian objective has an exact
   row-wise lexicographic tie-break. Fairness is measured from live parameters,
   gradients, canonical Torch-profiler events, CUDA identity and memory,
   latency samples, and artifact-matched state tensors.
5. R5 is one public 3,800-sequence package with exact counts, seeds, compound
   table, valid cells, sequence-set commitments, and exact per-row regeneration
   from factor specification, seed, index, and set name. Its low-level auditor
   is diagnostic; only the formal wrapper can issue an R5 certificate.
6. R6 accepts only hash-verified bundle references under the canonical
   repository protocol. Every output must come from a verified append-only
   event ledger plus an external tail commitment. Model and optimizer state is
   derived from safe numeric NPZ tensors, and every optimizer event binds
   before/after state artifacts, inputs, RNG, AMP state, and a hash-chain tail.
7. A serialized fairness record is always an
   `UNVERIFIED_SERIALIZED_AUDIT`. Formal R6 requires a same-process live
   fairness capability and a fresh-process capability bound to executable,
   argv, repository root, exact commit/tree, environment, and loaded module
   origins. Bootstrap and terminal decisions additionally require internal
   capabilities derived from verified raw cells and immutable ledger rows.

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

The source identity is currently:

`UNREGISTERED_BLOCK_POPULATION_R0_R1`

A reviewed registration commit must bind:

- the authoritative THUMOS14 temporal-annotation bytes;
- an exact source-origin attestation with release, subset, source URI, and
  source-object commitments;
- an official 213-video manifest with per-video SHA-256 and byte count;
- release identity and revision;
- the `validation` reporting subset;
- a canonical 211-entry historical inventory;
- all 211 actual historical video artifacts and hashes;
- explicit source-video to canonical-video aliases.

A registered state rejects sentinel and semantic placeholder revisions. The R1
annotation hash must equal the reporting-population annotation hash. Public
population and R0 APIs accept only bundle references, then reload the canonical
protocol, manifest, review, HEAD, tree, annotation, origin, official manifest,
inventory, and artifacts. The registered source commit must equal the signed
review commit's direct parent. Every historical artifact must be an `.mp4`
that OpenCV can open and decode to a nonempty first frame; its bytes are re-read
after decoding to reject mutation during validation. Duplicate content is
forbidden unless the exact canonical-ID pair is frozen as a legitimate alias.

The canonical 213 set is derived from annotation rows whose subset equals
`validation`. The historical set is derived from the inventory and actual
artifacts. The two absent historical files and their
`HISTORICAL_LOCAL_FILE_ABSENT` reasons are computed as annotation-minus-
inventory. There is no API for author-supplied membership lists or explanations.

## R0 Gate

The R0 collector directly parses annotation and class-map bytes but returns
only `DIAGNOSTIC_ONLY_REQUIRES_FORMAL_ENVELOPE`. It cannot emit PASS or a
scientific certificate.

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

R0 family coverage is descriptive support only. The earlier heuristic
single-proportion power calculation was not aligned with the paired video/seed
estimand and has been removed.

## R1 Gate

R1 is limited to decoded RGB frames through one feature token. It does not
claim compressed-stream or wall-clock decoding causality.

The extractor execution identity is currently:

`UNREGISTERED_BLOCK_R1`

Before R1 can run, one reviewed protocol commit must bind the repository
commit, exact extractor source hash, Hugging Face snapshot revision, snapshot
manifest hash, configs, weight shard filenames, CPU device, image size 224,
batch size 64, local-files-only execution, the canonical resolved-command
schema, and the canonical active-software-version schema.

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

It loads each NumPy feature array, decodes every bound raw video, selects the
registered source frames, reconstructs the bound encoder from the snapshot,
and reruns every token. Cache rows must be byte-identical after conversion to
the cache dtype. The resolved command must describe the same direct execution
path, and Python, NumPy, OpenCV, Torch, and Transformers versions must equal
the registered runtime record. It also derives the perturbation sample from
the support map and reruns every transcript record. Missing identity, runtime
drift, cache drift, transcript drift, or a no-op mutation yields
`FAIL_UNVERIFIABLE`.

For every nonterminal token, the future intervention inverts the native decoded
RGB suffix after the decision frame. A terminal token has no native future
suffix, so it uses two equal-shape, deterministically paired batches that differ
only in a synthetic continuation appended after end of stream. The continuation
is an audit intervention, never a model input used for training or evaluation.
Both modes require exact target-token invariance; changing the target support
frame must still change that token.

## Arm Fairness

B2 is now executable in
`opentad/utils/prefix_route_b2_contract_v2.py`. It has 64 propagated-track
slots and executes all 64 newborn queries at every decision, for at most 128
decoder queries. Propagated identities are locked before quantized Hungarian
matching of newborn queries against targets plus 64 unique optional dustbin
columns. Dustbins participate in optimization rather than being applied after
a forced target assignment. Primary costs are quantized to integers; among all
primary-optimal assignments, one arbitrary-precision row-wise lexicographic
objective selects a unique result without floating perturbations. Completed
but un-emitted instances remain in the instance-aware risk set and keep a
first-emission target.

`B2StreamMachine` owns each stream's decision index, observation count, tracks,
track serial, emission sequence, state hash, and transcript. It accepts only
the next stride-8 observation count, including one final partial stride, and
rejects repeat, skip, reorder, reset, or caller-supplied serial/sequence state.

B2 can create persistent newborn tracks only from slots that were free before
the current decision. A slot released by completion or drop becomes reusable
at the next decision, not in the same decision. All newborn queries are still
executed, and a direct-complete newborn may emit immediately. This precise
restriction is the predeclared D1 difference: B4 must earn value from atomic
release and same-bin reseeding rather than from a vague "persistent query"
description.

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

Serialized rows that merely assert these values can only be archived as
`UNVERIFIED_SERIALIZED_AUDIT`; they can never authorize R6. The production
fairness function accepts live runtime adapters only. It counts actual
trainable parameters, runs backward, requires finite nonzero smoke gradients,
requires the optimizer parameter set to equal the model trainable parameter
set, replays every microbatch in one trace-bound optimizer event, measures
FLOPs from canonical profiler-event rows, reads CUDA identity and peak
allocation, retains all 100 post-warmup latency samples, and recursively counts
live causal-state tensor bytes. The live model `state_dict` is converted to the
same semantic tensor digest as its bound safe NPZ artifact; a byte-valid but
state-mismatched checkpoint is rejected.
Optimizer events, effective tokens, accumulation, calibration videos, trials,
and seeds are read from four canonical hash-verified budget-plan records;
these records cannot independently produce fairness evidence. Only the live
model/optimizer/CUDA audit may produce the fairness record. During R6, each
arm/seed execution ledger must bind the initial and final model, config,
command, environment, full optimizer-event trace, fairness audit, and
emissions. Every trace event carries input-batch, model-before,
model-after, and optimizer-after commitments; adjacent model states must form
one chain. The live audit reproduces the first event, while R6 checks the full
trace count, token sum, accumulation, first/last model states, and trace hash
from its verified source reference. Formal B2/B3/B4 run budgets and initial
model artifact/tensor digests must equal their corresponding live fairness
rows; ablation and diagnostic-control runs retain independent bound ledgers.
Production measurement remains blocked until a separate model-P0 contract is
authorized.

## Controls and OOD

Feature-time shuffle and semantic derangement are executable functions.
Semantic derangement:

- operates within one frozen split;
- preserves the global label multiset;
- changes every instance label;
- forbids a global consistent class rename;
- fails if no legal construction exists.

The R5 generator fixes 64 bins with eight ticks per bin, 16-dimensional
observations, four classes, event topology, temporal geometry, semantic
mapping, observation distribution, valid crosses, balancing, and seeds. The
full package contains:

- 1,000 training sequences;
- 400 IID holdout sequences;
- 400 sequences for each of four single-shift families;
- 800 public compound-OOD sequences over eight frozen combinations.

All 3,800 scientific-content hashes must be pairwise disjoint. Every shifted
factor must change scientific content relative to an IID counterfactual, every
cell count differs by at most one, and each complete set must match its frozen
sequence-set SHA-256. The low-level auditor regenerates every row from its exact
`factor_spec`, seed, sequence index, and set name and requires canonical-byte
equality. Each row also carries a separate canonical full-row SHA-256 that
commits the set metadata, factor specification, seed, index, scientific hash,
and generated payload. Payloads therefore cannot be exchanged while retaining
their factor labels, even though the scientific-content hash intentionally
excludes set metadata for cross-set disjointness.

The low-level output has no PASS field. `certify_r5_bundle` can issue the
formal status only after reloading the signed protocol, manifest, review,
HEAD, and tree.

Because the grammar, seeds, and compound table are public, R5 is explicitly a
design-exposed protocol stress test, not hidden confirmatory OOD evidence. A
final hidden-OOD claim requires a separately committed unseen grammar or seed
before any model artifact exists.

## R6 Decision

The historical public Python entry is:

`opentad.evaluations.prefix_route_r6_v2.evaluate_r6_raw_evidence`

It is retained only to reject serialized fairness evidence and cannot produce a
formal result. The sole certificate launcher is:

`tools/run_prefix_route_r6_v2.py`

The launcher starts a fresh child with a one-use nonce. The child reloads the
canonical protocol, manifest, signed review, and exact commit/tree, then
verifies every loaded repository module against the reviewed import closure,
including existing parent-package `__init__.py` files. Parent-process
monkeypatches are not inherited. Its private live-evaluation entry additionally
requires both a live fairness capability and the child process capability.

After a future model-P0 authorization, that path will consume verified
references inside one contained evidence bundle for all 15 model/control arms
and fixed seeds 705, 706, and 707. It rebuilds population and R0, validates the
complete 213-video detail, and checks each run's exact commit/tree, safe
initial/final model and optimizer tensor artifacts, typed config/command/
environment artifacts, optimizer trace, execution ledger, control
construction, live fairness binding, immutable emission ledger, and external
tail commitment.

Every control run additionally carries a canonical construction record. R6
rechecks its exact frozen algorithm and parameters, source-artifact bytes,
reporting-video population, a complete ordered per-video transcript of source
and constructed-input hashes, and the output-emissions commitment, then binds
that record's SHA-256 into the run ledger. Missing videos and no-op
transformations fail. A control name alone is never accepted.

Only a verified ledger can mint the internal emission-row capability consumed
by lifecycle metrics. Plain lists, rows with a caller-added `immutable=True`,
rewritten chains, reordered chains, and run/stream provenance substitutions are
rejected. `OnlineAPBudgeted` is always invoked with `require_ledger=True`.

Only after this source chain closes does R6 derive GT and stress-family IDs,
invoke `compute_full_petal_metrics` and `OnlineAPBudgeted`, and perform crossed
paired resampling:

- one global seed resample is shared across every video and arm;
- one video resample is shared across every arm;
- mOnlineAP uses paired global training seeds;
- additive lifecycle metrics use paired video and seed units;
- multiplicity includes every registered contrast and eligible metric.

The production bootstrap call uses literal `10000`, `2026071707`, and `0.05`
values rather than mutable module globals. Caller cells cannot invoke the
bootstrap sealer, and a hand-built complete inference dictionary cannot invoke
the terminal. The terminal accepts only an internal object binding raw
evidence, per-video cells, dataset metrics, and bootstrap-output commitments,
then validates the exact complete inference schema, all registered contrasts
and metrics, fixed parameters, margins, estimates, and interval fields.

B4 survives only when:

1. it is noninferior to B2 on all global metrics; and
2. D1 is established by the A4 deletion, or D2 is established by the joint
   A1+A2 deletion; and
3. count-only, template-timing, and ledger-only controls are not equivalent to
   B4; and
4. feature-time shuffle establishes temporal dependence and semantic
   derangement establishes at least a 0.05 `class_mOnlineAP` drop.

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

The formal R6 launcher has a zero-GPU preflight command, but it still requires
the fixed signed review and registered source chain:

```powershell
python tools/run_prefix_route_r6_v2.py preflight `
  --review-attestation <review.json> `
  --review-signature <review.sig>
```

Its `evaluate` command intentionally returns
`MODEL_P0_RUNTIME_FACTORY_NOT_AUTHORIZED` until a later fixed protocol review
explicitly authorizes model construction and production CUDA measurement.

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

With the current unregistered source state, even a valid PASS authorizes only
read-only source-identity registration. R0/R1 require a second fixed,
independently reviewed protocol commit containing the registered identities.
Model work, profiling, and training remain blocked.
