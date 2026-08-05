# EventMATR D1.5 frozen owner counterfactual protocol — 2026-07-30

Status: approved for implementation as a train-only, read-only structural
diagnostic. It is not a model-selection run and cannot produce paper
performance.

## Exact inputs

- frozen D1.4 code:
  `fa27b3657b72c5b713ee3d2a5c0e652e7ca14eb4`;
- frozen D1.4 tree:
  `7603fc6b8226fa8f9dcb3d5212136fb47631bc32`;
- frozen D1.4 manifest SHA-256:
  `fb1d4036f6954b48074038460cc3f148059805df0487ae21b8a4692fd9e112fa`;
- frozen D1.4 decision-aligned terminal checkpoint and options, whose hashes
  must be supplied by the existing D1.4 terminal-scan receipt and rechecked
  before and after D1.5;
- review attachment SHA-256:
  `582839bda328cd0f856dccb9083b6c10334613e6e9ce776a5f5a594099318001`;
- the review attachment was read in full before this protocol was frozen.

## Overall scientific verdict

The review's main verdict is accepted with corrections:

> D1.4 proves that the independent birth branch is not gradient-dead and that
> a decision-aligned event bag can cross the fixed zero-log-odds boundary. It
> does not identify one unique post-birth cause. The next admissible action is
> a frozen-checkpoint factorial diagnosis that separates admission from
> identity transport and separately tests model-induced cancellation
> truncation.

The project does not accept the stronger statement that end-risk learning is
already the single established root cause. Identity-supervision routing,
model-induced loss of future end exposure, owner-state competition and birth
sampling calibration remain competing explanations.

## Accepted, qualified and rejected interpretations

### Accepted

1. Aggregate interval probability can improve while every individual birth
   logit remains below zero. Such an aggregate objective is not sufficient for
   the unchanged rising-edge runtime decision.
2. The decision-aligned arm establishes functional birth activation only:
   `30,002` births followed by `29,974` cancellations and zero ends/emissions
   is not a closed lifecycle.
3. A predicted record without a training target ID is not necessarily a false
   event. Routing every such record directly to `CANCEL` creates a
   supervision/runtime semantic mismatch.
4. The current model can remove a record before its later observable end,
   making end exposure dependent on the model's own cancellation policy.
5. A post-forward oracle identity intervention is legitimate as a train-only
   falsification tool if it is physically isolated from model inputs and
   runtime identity state.
6. Score-dependent top-k birth negatives do not have a known fixed inclusion
   probability and cannot support a population-calibrated log-odds claim.

### Accepted only as hypotheses requiring D1.5 evidence

1. Excess unmatched-track supervision may dominate owner optimization.
2. Persistent owner embeddings may drift away from the current query carrying
   the same event.
3. Early cancellation may remove otherwise learnable end decisions.
4. Birth over-admission may be the main upstream cause of the failed owner
   population.
5. The inherited training-time `segment_flag` path may contribute a
   query-feature train/inference shift.

### Rejected as current evidence claims

1. `3,003` is not the number of successfully associated predicted tracks.
2. `305,064 / 3,003 = 101.6` is not an observed gradient ratio.
3. `3,573` post-forward admissible assignments are not runtime identity
   successes.
4. Zero reacquisition is not evidence that a GT-free learned reacquisition
   algorithm failed; that algorithm does not yet exist.
5. New records are not cancelled in their creation prefix. Their first owner
   decision occurs at the next real prefix.
6. The present end surrogate is not a calibrated three-state right-censored
   likelihood.

## D1.4 semantic correction

The historical raw values remain immutable, but their interpretation is
corrected prospectively:

- `event_owner_assignment_count` in the D1 criterion is incremented once per
  positive birth-risk group. It must be renamed in new receipts to
  `event_birth_risk_group_count`.
- `event_false_track_cancel_group_count` counts batch-local owner groups
  grouped by `(video_name, runtime_event_id)` whose class targets are all
  negative. It is neither a row count nor a unique cross-batch trajectory
  count.
- The two values have different statistical units. Their numerical quotient
  may be reported only as an obsolete descriptive count quotient, never as a
  loss weight, gradient dose or paper root-cause estimate.
- New accounting must distinguish unique runtime records, batch-local
  fragments, owner-state rows, target-backed groups, unresolved groups,
  causally adjudicated false groups, per-family loss sums and per-family
  owner-head gradient norms/cosines.

## Frozen scientific questions

D1.5 asks exactly four questions:

1. Can the unchanged predicted-admission path reproduce the D1.4 terminal
   lifecycle counts?
2. Holding those predicted births fixed, does oracle refresh of the current
   event-bearing query restore continue/end/emission?
3. With one clean oracle-visible admission per training event, can the frozen
   owner state close without later identity refresh?
4. If formal tracks cancel early, does a policy-independent no-cancel shadow
   later produce an end winner near the observed end?

It does not estimate detection accuracy, model superiority, generalization or
novelty.

## Frozen two-by-two diagnostic

Two factors are crossed:

- admission: unchanged predicted rising-edge admission (`P`) versus one
  oracle-visible admission (`O`);
- identity: free persistent owner state (`F`) versus post-forward oracle
  current-query refresh (`R`).

| Channel | Admission | Identity | Frozen meaning |
|---|---|---|---|
| `PF` | predicted | free | Exact D1.4 predicted-only control. |
| `PR` | predicted | refreshed | Preserve every predicted birth and false-birth burden; refresh only records causally matchable to a currently visible training event. |
| `OF` | oracle-visible | free | Create exactly one record at each first observable training birth, suppress predicted admissions, initialize from a frozen compatible model query, then run the frozen owner freely. |
| `OR` | oracle-visible | refreshed | Use the same clean admission as `OF` and refresh the owner input from the current compatible query at every later visible prefix; all owner states, endpoints and emissions remain frozen-model decisions. |

### Common controls

1. Every channel starts from the same exact checkpoint, options, official
   train-only artifacts and chronological prefix order.
2. Model evaluation is deterministic and uses `torch.no_grad()`.
3. No optimizer is constructed, no backward pass occurs and no checkpoint is
   written.
4. MATR memory, EventMATR memory, ledger, diagnostic sidecar and recurrent
   shadow are reset independently for each channel and video.
5. Current/past model queries are the only neural inputs. Ground truth may be
   consulted only after the current query/transition forward.
6. The query-output census and hash must match across all four channels.
7. The existing fixed zero-log-odds birth decision and three-state owner
   argmax remain unchanged. No threshold is introduced or searched.

### Predicted admission

`PF` and `PR` use the unmodified D1.4 decision-aligned rising edges, including
their original timing, count, candidate start and false-track burden.

`PF` must reproduce the frozen D1.4 counts:

- births: `30,002`;
- cancellations: `29,974`;
- ends: `0`;
- emissions: `0`;
- capacity exhaustion: `0`.

Any mismatch stops the study as a diagnostic implementation failure.

### Oracle-visible admission

`OF` and `OR` create one diagnostic record when a training event first becomes
visible at its causal birth crossing. They suppress all predicted admissions
in those channels. The record:

- uses the highest-scoring model-produced current query under one frozen
  post-forward compatibility rule, with one-to-one selection across
  simultaneously visible births;
- uses the model candidate start associated with that query, clipped only by
  the existing non-negative-start contract;
- does not use the annotated start as the emitted/model start;
- stores the training target ID only in a diagnostic sidecar;
- never writes that ID into formal `EventRecord.target_event_id`.

Every visible event receives one record. An exact compatibility tie is resolved
by the frozen deterministic query-index rule and counted explicitly; it is
never hidden as evidence of naturally identifiable ownership.

### Oracle identity refresh

`PR` and `OR` run normal current-prefix MATR query/transition computation first.
Immediately before owner decoding, a frozen one-to-one post-forward matcher
selects the current query corresponding to each sidecar target and supplies a
cloned owner input tensor for that diagnostic channel.

Only the owner-decoder input embedding is refreshed. The target ID must not:

- enter the query backbone, transition heads or MATR memory;
- alter formal runtime event IDs;
- activate exact-ID reacquisition;
- alter duplicate exclusion;
- create or suppress predicted births;
- route teacher recovery;
- be persisted in a production checkpoint or ledger.

Unmatched `PR` records remain byte-for-byte unchanged. A refreshed decoder
output may mutate only its own diagnostic channel state.

## No-cancel shadow

Every channel also owns a separate recurrent shadow cloned before its first
formal cancellation:

1. formal recurrence applies the frozen `CANCEL / CONTINUE / END` decision;
2. shadow recurrence records the same logits but treats a `CANCEL` winner as
   `CONTINUE` for state retention only;
3. the shadow carries the decoder's updated embedding forward and is decoded
   again at later prefixes;
4. it can record later end winners and margins but can never emit a paper
   prediction or mutate the formal ledger.

This is a policy-intervention diagnosis, not deployment behavior. Merely
re-reading the cancellation-prefix logits without recurrently carrying the
shadow is insufficient.

## Positive lifecycle control

A separate GT-forced lifecycle control creates one start/continue/end sequence
for every visible training event. It tests only:

- non-empty state-machine execution;
- one unique event ID per lifecycle;
- exact one-to-one end/emission closure;
- immutable ledger prefixes;
- non-negative starts and positive lengths;
- contiguous sequence IDs;
- zero duplicate IDs, silent drops and capacity exhaustion.

It cannot establish owner-model ability and is not part of the two-by-two
causal matrix.

## Required per-decision evidence

Each channel writes a chronological JSONL row with at least:

- channel, video, runtime event ID and diagnostic sidecar target ID;
- source, creation frame, current frame and first-owner-decision flag;
- owner query before/after refresh and matched current query;
- owner embedding cosine to birth and to the oracle current query;
- owner-attention top query, oracle-query rank and entropy when available;
- cancel, continue and end logits;
- all three one-versus-best-other margins and state argmax;
- end suppression by minimum duration;
- formal transition and shadow transition;
- whether the target end is observable now and frames from that observation;
- formal and shadow lifetime.

The aggregate receipt must also contain:

- unique records and batch-local fragments by source;
- actual and late diagnostic associations;
- first-decision cancel/continue/end counts;
- formal/shadow end winners and lifetimes;
- end argmax, minimum-duration suppression and end-without-emission counts;
- records active after observed EOS;
- raw semantic duplicate and fragmentation counts before NMS;
- checkpoint/options/data/source/tree/manifest hashes;
- the full contamination and no-update flags below.

## Mandatory receipt flags

Every output must state:

```text
train_only=true
counterfactual=true
paper_performance_valid=false
strict_causal_paper_result_valid=false
ground_truth_visible_to_query_backbone=false
ground_truth_visible_to_owner_intervention=true
ground_truth_stored_in_runtime_record=false
optimizer_constructed=false
optimizer_step_count=0
checkpoint_updated=false
threshold_search=false
test_access=false
```

Missing, false, contradictory or non-finite evidence fails closed.

## Frozen interpretation table

| Observation | Supported diagnosis | Only authorized next model intervention |
|---|---|---|
| `PF` does not reproduce D1.4 | Diagnostic implementation is not equivalent. | Fix the diagnostic only. |
| `PF` fails and `PR` restores end/emission | Identity transport is necessary under the same predicted-birth burden. | Training-only predicted-track supervision bridge. |
| `OF` fails and `OR` restores end/emission, while `PR` does not | Identity and admission interact; identity alone is insufficient under current predicted births. | At most one prospectively registered two-factor repair. |
| `OF` closes while `PR` does not | Clean admission/initial query is sufficient; continuous oracle refresh is not necessary. | Sampling-corrected birth admission/calibration. |
| `OF` and `OR` both fail, and their shadows never produce end | Frozen owner/end representation or objective remains insufficient. | Proper three-state competing-risk owner objective on a policy-independent cohort; first run the inherited-memory flag diagnostic. |
| Formal tracks fail but no-cancel shadows produce end near observed ends | Early cancellation/state competition truncates learnable end decisions. | Separate unresolved identity from false-track cancellation or add a justified hold state. |
| End margin is positive but runtime end is absent | Scores are sufficient; runtime gating/order is faulty. | Fix minimum-duration, masking, record identity or step ordering only. |
| End occurs but emission does not | Ledger/state-machine closure is faulty. | Fix end-to-emission closure only. |
| Only the GT lifecycle control fails | The diagnostic state machine is unsafe. | Stop all model experiments. |

These cells are structural routing rules, not statistical treatment-effect
estimates. If a later trained two-factor experiment is authorized, it must
report the two main contrasts and the interaction rather than using
"fewest factors that pass" as a causal estimator.

## Conditional model work after D1.5

Exactly one branch may be registered after the diagnostic:

1. **identity branch:** a training-only `supervision_target_id` sidecar attaches
   causally matched predicted records to continue/end/class/identity targets
   without entering runtime state;
2. **owner-risk branch:** a policy-independent shadow cohort trains a proper
   three-state competing-risk loss, with risk-time subsampling corrected by
   known inclusion probabilities if used;
3. **admission branch:** replace score-dependent top-k negatives with a
   pre-frozen, score-independent risk-set sampler and inclusion-probability
   correction;
4. **two-factor branch:** allowed only when the frozen table demonstrates an
   admission/identity or identity/owner-risk interaction that neither single
   factor can close.

No branch may combine all proposed repairs at once.

## Official-comparability debt

D1.5 itself is not blocked by the following output-layer debts because it emits
no official detection metric. They must be closed before a paper comparison:

1. EventMATR's writer still converts ledger frames with complete-video
   `frame_to_time`; the formal route must use arrival timestamps or a
   predeclared fixed feature stride/FPS.
2. The current full-file `online_nms` is not proven prefix-equivalent.
3. Native MATR ranks class probability, whereas EventMATR ranks a geometric
   mean of birth, class and end confidence.
4. Native and EventMATR common parameters require a common-initialization hash,
   not merely the same seed.
5. A future main comparison must keep the official split, extracted
   RGB-plus-flow features, 100 epochs, optimizer/scheduler, terminal checkpoint,
   evaluator, timestamp convention and prospectively frozen post-processing
   identical or explicitly controlled.

Until those items and a future structure gate pass, no locked test, multiple
seeds, raw-RGB training, threshold search or paper performance claim is
authorized.

## Minimal implementation boundary

The first implementation is restricted to:

- a diagnostic-only intervention point after current MATR query/transition
  computation and before owner decoding/EventMATR `step`;
- cloned diagnostic owner inputs and isolated per-channel/shadow memories;
- a new train-only scan script and fail-closed finalizer;
- protocol registration and deterministic tests;
- corrected metric names and new counters without changing the frozen D1.4
  checkpoint or production inference behavior.

Production `EventRecord` schema, official ledger schema, training objective and
runtime inference remain unchanged in this diagnostic revision.

## Pre-execution implementation registration — 2026-07-30

The diagnostic implementation is now frozen at:

- code commit:
  `ee59b096c6e47f4085e6236696dc7c6831d81dd5`;
- code tree:
  `af6de42f9e53f3142353f93c4809bf8a34b4d7be`;
- manifest SHA-256:
  `903a93a129c16b3bc4778d7ddf002e15eca1cbed1e3ba75e5261ff69bf2c8b22`;
- branch:
  `codex/eventmatr-d1`;
- remote:
  `https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702.git`.

The implementation uses one evaluation-only, no-gradient, GT-free MATR query
forward. `PF` formal keeps the original one-route batch shape so that its owner
decode remains the exact D1.4 reproduction control. The other seven isolated
routes are decoded together for throughput; whenever a formal route and its
shadow still have byte-identical pre-intervention owner inputs, the shadow
reuses the formal decode exactly. All eight memories are prepared before any
route mutates its own state.

Ground truth is parsed only after the shared query forward. Diagnostic target
IDs remain in Python sidecars; every active and archived runtime record is
checked for `target_event_id=None`. Oracle admission suppresses predicted
births but still updates the rising-edge history. The no-cancel shadow retains
the recurrently updated owner embedding and class distribution while changing
only the cancellation transition.

Evidence integrity is stricter than the minimum protocol text:

1. every route independently hashes the exact sequence of physical prefixes
   and shared batch-output tokens that it consumes;
2. all eight route-consumption hashes and counts must agree;
3. the finalizer decompresses the decision trace and validates mandatory
   fields, per-route row counts, per-video chronological order, one owner
   decision per runtime ID and prefix, and the reported end-winner,
   minimum-duration-suppression and target-backed-end aggregates;
4. output and receipt paths are append-only and must remain outside the source
   repository;
5. the positive lifecycle control checks both active and archived runtime
   records for GT contamination.

Local pre-execution checks pass Python compilation, static undefined-name
inspection, manifest parsing, shell syntax, Git whitespace validation, and two
pure-Python finalizer trace-integrity tests. The workstation PyTorch binary
cannot load its native DLL, so model-level tests are not claimed locally. A
clean exact-source Slurm CUDA test and official-data smoke remain mandatory
before the complete train scan.

Status at registration: **implementation complete, remote preflight and
complete diagnostic scan not yet run**. No scientific route, model repair,
performance pilot, official comparison or paper claim has been released.

## Exact-source and preflight amendment — 2026-07-30

The pre-execution registration above is historical and is superseded for
execution by the following exact source:

- code commit:
  `9b9189af46462e50dad90bf35248799300f974d8`;
- code tree:
  `45b1ab3b361681fd5a04f07500851c7bcacef093`;
- manifest protocol:
  `eventmatr_d1_preexperiments_v8`;
- manifest SHA-256:
  `8ed0d91cde2017f30c1815ad15fef9f590969cf7bd81110e8c9df53f4318cda3`;
- branch:
  `codex/eventmatr-d1`;
- remote:
  `https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702.git`.

The amendment preserves three independent source identities instead of
aliasing them:

1. exact source training:
   `92cf34aa07bebee2a7a7e3661431d5055804b29b`,
   tree `aef4f64bc020df9d39ead9811fbc01407f1c754a`;
2. D1.4 gate/checkpoint diagnostic source:
   `fa27b3657b72c5b713ee3d2a5c0e652e7ca14eb4`,
   tree `7603fc6b8226fa8f9dcb3d5212136fb47631bc32`;
3. current D1.5 diagnostic source:
   `9b9189af46462e50dad90bf35248799300f974d8`,
   tree `45b1ab3b361681fd5a04f07500851c7bcacef093`.

The runner now verifies those identities separately against the frozen
manifest, source-training receipt and D1.4 structure-gate receipt. The scan
embeds all three, and the finalizer rejects any missing, mismatched or aliased
identity. The D1.4 linked training-artifact key is bound to its actual frozen
receipt key, `bag`; no historical artifact was rewritten.

### Append-only preflight ledger

All failed attempts stopped before a complete D1.5 scan and remain preserved:

| Job | Exit | Furthest completed stage | Fail-closed reason |
|---|---:|---|---|
| `1207473` | `1:0` | source identity | Shared-filesystem temporary capture directory was not writable. No test or model execution. |
| `1207477` | `1:0` | tests | `135` tests passed and four tests exposed stale/incorrect test assumptions. No official-data scan. |
| `1207501` | `1:0` | tests | `139` tests passed; one stale D1.4 release-wording assertion remained. No official-data scan. |
| `1207506` | `1:0` | tests | `139` tests passed; one stale post-processing wording assertion remained. No official-data scan. |
| `1207516` | `1:0` | tests and synthetic mechanisms | `140` tests passed; official-train smoke refused to start without explicit source-identity environment variables. |
| `1207522` | `1:0` | official-train smoke | Tests, synthetic mechanisms and the official one-batch smoke passed; D1.5 then detected the training-source/D1.4-source identity alias and stopped before any diagnostic batch. |
| `1207543` | `0:0` | complete preflight | `144` tests, synthetic mechanisms, official one-batch smoke and a 12-batch real D1.5 partial scan all passed. |

The successful preflight root is
`/data/run01/sczc063/yuzibo/runs/eventmatr_d15/preflight_9b9189a_20260730_r7`.
Its immutable evidence includes:

- preflight script SHA-256:
  `10761a5583b29552f16e4cd095211090afe9e0c996a51c1518d549dcaf3fee40`;
- start identity SHA-256:
  `25d9ce5fd24f9e45be1e0fcf8188bcd46d14ab25b8d54c1528de3b558c9ec54c`;
- final identity SHA-256:
  `990704d43fdb935e12ef7fbf210b8b12f185e1a8c124f65990b632c780b4df12`;
- official-train smoke SHA-256:
  `5ce119d570bf3acdcbb21b4c3fa84323118419b1f06b105d8cb6d68320653d0f`;
- 12-batch partial scan SHA-256:
  `ffcdcab54088899864c4105eb045537f250e8d9e6bb4248b38e0d0de7b23cd20`;
- partial trace SHA-256:
  `af878d21e23df069d504adbd8e85dcc762a3ce434d12acfd1774d3495425018a`.

The partial scan consumed `12` verified causal physical batches and `768`
real prefixes with one common route-consumption hash. It observed three
visible births. The positive lifecycle control closed exactly three births,
three ends and three emissions with zero capacity exhaustion. The oracle
formal channels admitted and then cancelled those three records; their
recurrent shadows remained active and produced no end in this short prefix.
Predicted channels had no birth in these first batches. These values validate
execution only and are not a population diagnosis or paper result.

### Complete-scan launch

The complete official-train diagnostic was submitted as Slurm job `1207567`
under the append-only root
`/data/run01/sczc063/yuzibo/runs/eventmatr_d15/formal_9b9189a_20260730_r1`.
The preserved submission script SHA-256 is
`cd3360bf399b36976c31b475e82e847f5887fb040bf296c007685c588b7c19c9`.

At this amendment point the complete scan is running. No D1.5 structural
route, model-training authorization, short pilot, official comparison,
locked-test access or paper claim has been released.

## Evidence-gate amendment and exact rerun — 2026-07-31

The preceding launch statement is superseded without deleting its history.
Job `1207567` failed `1:0` after `2:51` because the first padding contract
incorrectly expected zero retained active records. This was an evidence
implementation failure, not a model result. Commit
`ebf53c57280cd9f6099f84b7b1ac02a0b53cd63d`, tree
`63aae98fcc8f0fef8f748017247f262a3780978b`, repaired that contract;
preflight `1207607` then completed `0:0`.

Two later complete scans were deliberately stopped before adjudication:

- `1207643` was clean after `12:20` but was cancelled only to replace the
  inherited 12-hour allocation with a 24-hour allocation;
- `1207757` was clean after `15:13` but was cancelled when an independent
  audit showed that the finalizer did not yet reconstruct enough evidence
  from the chronological trace. Its partial trace is preserved, but no
  lifecycle diagnosis is accepted from it.

The final evidence hardening is exact commit
`451825211171a9c95e423bab1c5889c43b217f7a`, tree
`83577119cbd5a71c581581b864997ba77298a145`, on
`codex/eventmatr-d1`. It does not alter the checkpoint, frozen decisions,
model outputs or training. It adds fail-closed checks for:

- every registered trace field, finite state logits/margins, recomputed
  winners, query bounds, route lifetime and target-end distance;
- decision-source, first-decision-state, transition and end/emission
  aggregate closure;
- per-video active records after observed EOS, reconstructed from surviving
  pre-existing records plus births at that same observed EOS;
- true active-record semantics rather than all retained-but-not-yet-cleaned
  records;
- malformed nested receipt objects as controlled validation failures.

Exact preflight `1207933` completed `0:0` in `3:04` with empty stderr and
`162 passed`, followed by the synthetic mechanisms, official one-batch
train smoke and 12-batch real D1.5 partial scan. The real partial trace has
`4,050` owner-decision rows; every new field and per-source aggregate was
independently cross-checked. Its three visible positive-control events again
closed as exactly three births, three ends and three emissions. The successful
root is
`/data/run01/sczc063/yuzibo/runs/eventmatr_d15/preflight_4518252_20260731_r11`.
Its preflight-script, start-identity, final-identity, official-smoke,
partial-scan and partial-trace SHA-256 values are respectively:

- `c62e9c9b51394b1a1c918121f11b904ccf29e7938f96edd7698c8dc0f6e2d463`;
- `e37f9efa75b938ad5b59dfd8be2b6e5a5d73bd1edb4b5a9be0a1112695fbeef4`;
- `0bccb46fbfc35650fa810b1b37d3bafa4b7c47c8430f588265570c0bb2dd13e0`;
- `1d0ee5b8eb993b11afbf83b75a279bff1bf0170030b4e0ef86f74419058ff091`;
- `b27d6e464e2943007fbf4c73e391d6425f6bcc8772b164a80194ba9aff580575`;
- `bfb318654e9a7fdf9a94bd09847f3e6ee5004042d1d4446f0bb0a04bc79bc06c`.

The source-exact complete scan is Slurm job `1207954` under
`/data/run01/sczc063/yuzibo/runs/eventmatr_d15/formal_4518252_20260731_r4`;
its submission-script SHA-256 is
`e397a170d8862cb6e68fe03d52a512fc773d174323358309a9c9bd1b553545a4`.
At this record point it is running with a 24-hour limit and empty stderr.
Only its fail-closed final receipt may choose the next structural repair.

This remains a GT-aided, post-forward, train-only causal diagnosis. It is not
an official comparison or paper performance result and cannot release the
locked test, multiple seeds, raw RGB, threshold search/lowering or a longer
pilot.

## Post-review v2 amendment: right censoring and event-level routing — 2026-08-02

This section is append-only and supersedes the v1 completion and routing
contracts without changing the historical record above. The external review
input has SHA-256
`a2f0511ac97225cad2f69c11ee124e8a5a3b57f6aa95c41bf22403c7feb1dc0b`.
The amended v2 implementation was first frozen and pushed on
`codex/eventmatr-d1` at commit
`ffff9e43fd2c5af7d6440bb4cf4fb34a37040260`, tree
`adb00a25ec4f4ab087adc72f0e6c891e6c6b1bee`. Remote preflight then exposed one
stale-test contract and one summary-only name error. The source-exact execution
identity after those two non-scientific repairs is commit
`dc530e2d99f928b1455a6e137f708306cc00d91c`, tree
`14e9aeb886e85df5dfb2fd3b321f39597eb040c9`. This identity authorizes only exact
remote tests, preflight and the complete diagnostic rerun described below; it
does not authorize model training or a performance claim.

### Invalidated v1 attempt

Slurm job `1207954` failed `1:0` after `13:54:08` because the positive
lifecycle control required all `3,003` visible births to produce an observed
END. The run correctly produced only `3,001` END transitions: two events were
still active when their streams ended. The v1 runner then raised
`positive control end_count did not close: 3001 != 3003` before writing the
scan object, final source identity or final receipt. Its partial trace contains
`4,298,731,909` bytes (approximately `4.00 GiB`) and is retained as forensic
execution evidence only. It is
not an admissible source for a structural route, a model decision or a paper
claim.

The status is therefore frozen as:

- `D1.5_PROTOCOL = INVALID_RIGHT_CENSOR_CLOSURE`;
- `COUNTERFACTUAL_ROUTE = UNRESOLVED`;
- `MODEL_TRAINING = NOT_AUTHORIZED`;
- `PAPER_PERFORMANCE = NOT_AVAILABLE`;
- `PROVISIONAL_ARCHITECTURE = DISCOVERY_PLUS_PERSISTENT_EVENT_QUERIES`;
- `NEXT_ACTION = PATCH_PROTOCOL_AND_FULL_RERUN`.

### Correct lifecycle census and positive control

The v2 runner must independently reconstruct a hash-bound lifecycle census
from the official annotation and feature-length artifacts before any query
forward. For every annotation it records its video, annotation index, class,
start, end, admission crossing, end crossing, observable-window status and
numbers of birth, alive and end supervision rows. The complete official-train
census must close as follows:

| Quantity | Frozen expectation |
|---|---:|
| annotations | 3,007 |
| visible birth crossings | 3,003 |
| observable end crossings | 3,001 |
| right-censored after a visible birth | 2 |
| left-truncated | 0 |
| completely outside the observed feature window | 4 |

The two right-censored events are frozen by semantic identity, not only by
count:

- `video_validation_0000318`, event `22`, `HammerThrow`;
- `video_validation_0000985`, event `9`, `VolleyballSpiking`.

The four completely unobservable annotations are also frozen:

- `video_validation_0000364`, events `4` and `5`, `HighJump`;
- `video_validation_0000856`, events `3` and `4`, `SoccerPenalty`.

The positive control must satisfy
`birth = 3003`, `end = emit = ledger = 3001`,
`right_censored = active_after_observed_eos = active_at_scan_end = 2`, and
`videos_with_active_records_after_observed_eos = 2`. Cancellation,
reacquisition, capacity exhaustion, duplicate semantic identity and silent
loss remain zero. Observed EOS may preserve or archive a censored active state;
it must never synthesize an END or a detection output.

### What the sparse-supervision statement does and does not mean

The previously reported approximately `30.7x` reduction compares only the
number of first-birth or first-end crossing rows with MATR's repeatedly
positive span rows. It is not a ratio of total lifecycle supervision,
gradients, effective sample size or learnability. EventMATR also provides
repeated alive, class, identity and survival supervision. The existing
official-data audit reports `56,551` alive rows; v2 must recompute their exact
distribution and bind it to the census receipt.

Short actions nevertheless create a real length-dependent weakness: they may
have zero or one alive update between birth and end. A correct lifecycle model
must therefore allow a direct `birth -> end` transition and must not manufacture
alive labels merely to equalize counts. The eventual learning remedy is
event-normalized interval-censored birth, right-censored trajectory risk and
length-stratified accounting, not duplicated endpoint labels.

### Event-level structural gate

Raw END or emission positivity is removed as a routing criterion. A single or
small number of transitions can never authorize a model branch. The primary
event outcome is one if a target-backed END and its immutable emission occur
within the already frozen symmetric one-segment (`64` feature steps) window
around the annotated end, and zero otherwise. Every one of the `3,001`
observable-end events remains in the denominator. An unmatched runtime record
is recorded as unresolved identity, not asserted to be a false action; an
unresolved event receives zero primary success and its unresolved rate is
reported separately.

The three prospectively frozen formal-route comparisons are:

1. `PR - PF`: identity-refresh effect under predicted admission;
2. `OR - OF`: identity-refresh effect under oracle-visible admission;
3. `OF - PF`: clean-admission effect under free identity.

For each comparison, the effect is the mean same-event paired success
difference. A comparison passes only if all of the following hold:

- effect is at least `0.05` absolute probability, which requires at least
  `151` net improved events out of `3,001`;
- a deterministic video-cluster bootstrap with `10,000` resamples and seed
  `52015` has a 95% percentile-interval lower bound strictly above zero;
- a two-sided video-cluster sign-flip test with `10,000` permutations and seed
  `52016` remains significant at familywise `0.05` after Holm correction over
  the three comparisons.

The five-percentage-point value is a preregistered scientific relevance floor,
not an estimated optimum and not a threshold tuned on the invalid v1 trace.
Effect size remains mandatory even if a p-value is small. Four
shadow-minus-formal no-cancel comparisons form a separate secondary family
with the same effect and uncertainty requirements and their own Holm
correction; they can route only a cancellation/unresolved-identity diagnostic,
not model training.

The receipt must contain one row per visible event and route, including target
identity, censor status, link/unresolved status, near-end END, emission,
premature cancellation, first owner decision, lifetime, owner cosine and
attention rank. It must also contain the event-list hash, video-list hash,
seeds, resample counts, raw and Holm-adjusted p-values, interval, effect,
denominator and pass/fail reason.

All four channels receive the same post-forward target-link audit. The link is
diagnostic sidecar metadata only: refreshed-identity channels may consume the
matched current query, whereas free-identity channels must keep their decoder
input unchanged. This separation is necessary for a fair paired event outcome;
otherwise the free-identity controls would be labelled unresolved by
construction rather than by observed matching failure. The sidecar remains
outside runtime memory, formal transition state, ledger and model output.

The finalizer reconstructs target linkage from the chronological trace and
requires exact equality with every route summary; oracle-visible routes also
require zero duplicate target assignment. The summary linkage denominator is
explicitly the set of runtime records that produced at least one owner decision,
so a birth at the final prefix cannot create an uncheckable scientific link.
If no runtime record links to an event, its identity is unresolved. If more than
one runtime record links to the same event, its identity is ambiguous. Both
states receive zero primary success and are reported separately; an arbitrary
successful fragment may not turn an identity-fragmented event into a primary
endpoint success. The lifecycle census validator also recomputes crossing
frames, visibility states and supervision-row counts from the stored endpoints
rather than trusting those fields as self-reported metadata.

### Conditional route after a valid full rerun

- both identity contrasts pass while clean admission does not: authorize only
  implementation of the minimal MATR-internal discovery-query plus persistent
  event-query mechanism and its new structure tests;
- only clean admission passes: authorize only a prospectively registered birth
  admission/risk repair;
- both families pass, only one identity context passes, or interactions are
  inconsistent: no unique model repair is selected;
- no formal comparison passes but a no-cancel comparison passes: separate
  unresolved identity from false-track cancellation or add a hold state before
  any new objective;
- oracle-visible formal and shadow routes still show no material endpoint
  recovery: diagnose owner representation/competing-risk learning before an
  architecture change.

Even a valid D1.5 route does not itself authorize a performance experiment. A
selected structure must first pass synthetic lifecycle, official one-batch,
12-batch chronological and gradient/causal-boundary gates. Complete-video time
conversion, full-file suppression, ranking parity and common-initialization
hashes remain separate official-comparability debts. Until those debts close,
no result is paper-comparable.

### Source-exact v2 preflight and complete rerun launch — 2026-08-02

The executable v2 source is commit
`dc530e2d99f928b1455a6e137f708306cc00d91c`, tree
`14e9aeb886e85df5dfb2fd3b321f39597eb040c9`, on
`codex/eventmatr-d1`. Its manifest SHA-256 is
`27d6e1143812c3c4033c5a0d39cc17d58667f6727cde3d7c1d4e61f84747f9af`.
The transfer bundle SHA-256 is
`66f0861163448137a8e9666bf7556e51ea5c777a4e55b17ba1050b1b5676ee13`, and
the exact remote checkout is
`/data/run01/sczc063/yuzibo/EventMATR_D15_dc530e2d`.

The append-only preflight ledger is:

1. job `1213432`, root
   `/data/run01/sczc063/yuzibo/runs/eventmatr_d15/preflight_ffff9e4_20260802_r1`,
   failed before any model/data scan because seven historical v1 routing tests
   still called the old interface and asserted the superseded route contract;
2. job `1213433`, root
   `/data/run01/sczc063/yuzibo/runs/eventmatr_d15/preflight_e957ce8_20260802_r2`,
   passed all `177` then-current tests, the synthetic mechanisms, official
   one-batch smoke and the 12-batch forward, then failed while constructing the
   summary because a local variable name was undefined. It wrote no admissible
   final diagnostic receipt;
3. job `1213434`, root
   `/data/run01/sczc063/yuzibo/runs/eventmatr_d15/preflight_dc530e2_20260802_r3`,
   completed `0:0` in `3:25` with empty stderr. All `178` tests passed, followed
   by the synthetic mechanisms, official one-batch smoke and 12-batch
   chronological diagnostic. The independent lifecycle census closed at
   `3,007` annotations, `3,003` visible births, `3,001` observable ends, two
   right-censored events, four unobservable events and `56,551` alive rows.

The successful preflight-script SHA-256 is
`7fe4104817eb1d52c0a3ecacf83dafc8f9c19fa8b8a392202bad5c17e4246d06`.
It retained `test_access=false`, did not update a checkpoint, and reported
`strict_causal_paper_result_valid=false`.

The complete source-exact diagnostic is Slurm job `1213435` under
`/data/run01/sczc063/yuzibo/runs/eventmatr_d15/formal_dc530e2_20260802_r1`.
Its submission-script SHA-256 is
`5be6587e7a09d298fdf21351214af178851c6427c2018a4cba3494e197ca5528`.
It started on node `g0006` at `2026-08-02T01:49:28` with a 24-hour limit. At
the final launch audit it was running, its trace was growing and stderr was
empty. No route effect, model branch, training authorization or paper result
exists until the fail-closed final receipt has been independently validated.

### Complete scan retained; finalizer invocation recovery — 2026-08-02

Job `1213435` finished the complete source-exact scan and wrote both source
identities, the scan object and the chronological trace, then exited `1:0` after
`13:55:16` before producing a receipt. The failure was the final wrapper's
direct-file invocation: `python3 scripts/finalize_eventmatr_d15_owner_counterfactual.py`
could not resolve the finalizer's `scripts.eventmatr_d15_statistics` import. It
was not a model-forward, scan, data, trace or statistical-gate failure.

The retained artifacts are:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `source_identity_start.json` | 384 | `87d1c195f061fbcf10c36409b745ce08fa6b7764bdbb12ec740136ffff25c00c` |
| `source_identity_final.json` | 384 | `87d1c195f061fbcf10c36409b745ce08fa6b7764bdbb12ec740136ffff25c00c` |
| `owner_counterfactual.json` | 1,979,888 | `92d816b1e6c8d4890757c0147f41d3b33ab3652223b3ab71e42340da1d3d90c1` |
| `owner_counterfactual_trace.jsonl.gz` | 4,359,073,296 | `4ed11c6ed002cccb05191f23eb5e9d50eb5d438c2b91bbde6526fc56fc05f528` |

The scan reported the preregistered complete-prefix census and positive-control
closure, including `3,003 = 3,001 + 2`. It is still not admissible for routing
without the finalizer's independent trace reconstruction and fail-closed
receipt.

Two independent read-only audits agreed that the scan need not be repeated.
The finalizer loads the completed scan and both identities, validates all linked
hashes, rereads the complete trace, reconstructs every route/event outcome and
only then writes the previously absent receipt. Recovery must run from the
unchanged clean `dc530e2d...` checkout and may add only that missing receipt to
the original append-only evidence root.

The permanent wrapper repair changes only the invocation to
`python3 -m scripts.finalize_eventmatr_d15_owner_counterfactual`; it is commit
`0419a0ff94ea26513a5282f2be90e184cdffd3c6`, tree
`94dcb480575231737cad922a23585bb72d5ad521`, with a source-contract regression
test. The first recovery submission root ended before scheduling because an
explicit memory request conflicted with the partition policy; it produced no
job and no receipt. Recovery job `1214235` uses the unchanged original scan
source under
`/data/run01/sczc063/yuzibo/runs/eventmatr_d15/finalizer_dc530e2_20260802_r2`;
its script SHA-256 is
`b93686e0edc1c6b5b1b7693fc56bddcaec6076aff0c47c1184bde5cbed182f6e`.
At this record point it is running after all four retained-artifact hashes pass.
No structural route, model implementation, training or paper claim is released.

### Finalizer recovery completed; D1.5 v2 routing closed — 2026-08-02

Finalizer-only recovery job `1214235` completed `0:0` in `00:47:00` on `g0014`
with empty stderr. It reread the original complete trace and added only the
previously missing receipt to the original evidence root. The recovery receipt
is `33,961,025` bytes with SHA-256
`f28d2ca243ed5bfc68f4cdf2d0dd20eb13899c6ccd3edc929f81d1208990a358`.
The retained scan remains
`92d816b1e6c8d4890757c0147f41d3b33ab3652223b3ab71e42340da1d3d90c1`,
and the start/final source identities remain byte-identical at
`87d1c195f061fbcf10c36409b745ce08fa6b7764bdbb12ec740136ffff25c00c`.

The receipt status is `PASS_DIAGNOSTIC`. Its semantics are explicitly
`integrity_and_routing_pass_not_model_performance_or_paper_pass`. It preserves
`test_access=false`, `checkpoint_updated=false`, `optimizer_step_count=0`,
`threshold_search=false`, `locked_test_release=false`,
`official_comparison_release=false`, `paper_claim_release=false`,
`paper_performance_valid=false` and
`strict_causal_paper_result_valid=false`. The registered feature filename
contains `val`, but it is the manifest-bound THUMOS training/validation split;
the locked benchmark test split was not accessed.

The complete event-level outcomes are:

| Channel/route | Observable ends | Primary success | Near-end END | Immutable emission | Premature cancel | Unresolved identity | Ambiguous identity |
|---|---:|---:|---:|---:|---:|---:|---:|
| `PF/formal` | 3,001 | 0 | 0 | 0 | 838 | 2,121 | 708 |
| `PR/formal` | 3,001 | 0 | 0 | 0 | 838 | 2,121 | 708 |
| `OF/formal` | 3,001 | 0 | 0 | 0 | 2,995 | 0 | 0 |
| `OR/formal` | 3,001 | 0 | 0 | 0 | 2,995 | 0 | 0 |
| `PF/shadow` | 3,001 | 3 | 3 | 9 | 0 | 911 | 0 |
| `PR/shadow` | 3,001 | 3 | 3 | 9 | 0 | 911 | 0 |
| `OF/shadow` | 3,001 | 7 | 7 | 21 | 0 | 0 | 0 |
| `OR/shadow` | 3,001 | 6 | 6 | 20 | 0 | 0 | 0 |

Every formal owner decision that exists starts in the cancel state. The oracle
admission routes create and link exactly `3,003` target records, then cancel all
`3,003` at lifetime one; neither formal oracle route produces an END or
emission. The no-cancel oracle shadows keep those same records alive and expose
only `21` and `20` END/emission transitions, of which only `7` and `6` lie in
the frozen symmetric 64-step endpoint window. Predicted-admission shadows
produce `333` runtime END/emissions, but only `9` are target-backed and only `3`
are near the paired target endpoint. There is no capacity exhaustion, silent
record loss or END-without-emission fault.

All three formal paired effects are exactly zero with confidence interval
`[0, 0]`, Holm-adjusted `p=1`, and zero net improved events. The four no-cancel
effects are `7/3001`, `6/3001`, `3/3001` and `3/3001`, respectively. Their
effects are approximately `0.233%`, `0.200%`, `0.100%` and `0.100%`; every
interval includes zero, every adjusted p-value is at least `0.9835`, and all
are far below the frozen five-percentage-point relevance floor (`151` net
events). The floor is not lowered after seeing these results.

An independent deterministic recomputation of the complete paired analysis is
byte-equivalent as a Python value. Both stored and recomputed canonical objects
have SHA-256
`65c9471037baa3880caff9c8600a1db47d701c0ecc59456b161ba606cc0250a2`.
The independent integrity audit is recorded in
[eventmatr-d15-integrity-audit-20260802.md](eventmatr-d15-integrity-audit-20260802.md).

The routing decision is therefore final:

```text
diagnosis = no_preregistered_material_structural_effect
authorized_next_intervention = no_model_repair_until_diagnostic_refinement
model_implementation_authorized = false
model_training_authorized = false
```

This result falsifies identity refresh and oracle-visible admission as a
material standalone repair under the frozen primary endpoint. No-cancel is a
necessary enabling intervention for any observed endpoint, but its recovery is
too small to make cancellation the sole or sufficient bottleneck. The evidence
does not yet distinguish an owner representation failure from a learned
competing-risk/logit-calibration failure; that attribution requires a frozen
read-only temporal-margin analysis. No model patch, short pilot, longer run,
multiple seed, threshold search, raw-RGB experiment, locked-test evaluation or
paper claim is authorized.

### Registered D1.5.1 endpoint-margin trace refinement — 2026-08-02

The only authorized next experiment is a read-only reaggregation of the frozen
D1.5 v2 trace. It performs no model loading, forward pass, backward pass,
optimizer step, checkpoint write, data resampling or threshold search. Its
inputs are fixed to the receipt, scan and trace hashes above, the `3,001`
fully-observed event denominator, and the exact D1.5 source identity. It writes
to a new append-only diagnostic root rather than modifying the D1.5 evidence.

The analysis unit is one annotated target event within one channel/route.
Right-censored events remain excluded from endpoint denominators. Ground-truth
end is used only as an evaluation sidecar and is never fed to model state. The
temporal bins are frozen before execution as `< -64`, `[-64, 0)`, `[0, 64]`
and `> 64` feature steps from the annotated end. Duration strata are frozen as
`<=8`, `(8,16]`, `(16,32]`, `(32,64]` and `>64` feature steps. No alternative
window or stratum may be substituted after seeing the output.

For every channel/route and temporal bin, the diagnostic must report:

1. target-linked event coverage and decision-row counts;
2. cancel/continue/END winner counts and per-event maximum END margin;
3. events with positive maximum END margin, target-backed END, immutable
   emission and primary endpoint closure;
4. first-decision cancel margin, first post-end END margin and the change from
   the last pre-end decision when both exist;
5. unresolved/ambiguous identity, pre-end cancellation and events surviving to
   an actually observed endpoint;
6. the same summaries by frozen duration stratum, with class labels descriptive
   only and never used for routing.

Event-level medians and 95th percentiles use the deterministic nearest-rank
definition on sorted finite per-event values. Endpoint-positive rates and the
paired `OR-OF` rate difference use `10,000` common video-cluster bootstrap
resamples with seed `52017` and percentile 95% intervals. These resamples and
the rules below are frozen before the trace is read.

If multiple runtime records link to one annotated event, all decision rows
contribute to temporal-bin counts and the per-event maximum END margin. The
event's first-decision cancel margin is taken from the earliest linked
first-decision row, breaking a same-frame tie by the lowest runtime event ID.
At the nearest pre- or post-end frame, the maximum END margin across linked
records is used. A state-machine mismatch means that at least `5%` of the fixed
event denominator has an unsuppressed positive-END winner in `[-64,64]` without
a same-row target-backed `end+emit`; isolated cases remain reported integrity
findings but do not select the route.

The preregistered interpretation is:

- if an oracle no-cancel shadow has a point rate of at least `5%`, its bootstrap
  interval excludes zero, and its formal partner has none, cancellation/
  transition policy is a material primary bottleneck and only a prospective
  hold/cancel diagnostic may be designed next;
- if both oracle-shadow point rates are below `5%`, both upper bootstrap bounds
  are below `5%`, and both the median and 95th percentile of per-event maximum
  END margin inside `[-64,64]` are non-positive, the frozen owner END
  representation/risk objective is insufficient even after perfect admission
  and identity; cancellation remains contributory but is not the selected
  repair;
- if positive END margins are common but target-backed emissions fail to match
  them, route/state-machine accounting must be repaired before any model work;
- if the paired oracle identity-route difference is at least five percentage
  points in absolute value and its bootstrap interval excludes zero, identity
  attribution remains unresolved and no architecture is selected;
- any uncovered, mixed or confidence-unstable case remains unresolved. The
  response is another preregistered diagnostic, never a lowered threshold.

This D1.5.1 result remains diagnostic-only. Even a decisive result can at most
select the mechanism to implement and test; it cannot authorize training or an
official paper comparison by itself.

### D1.5.1 execution launch — 2026-08-03

The frozen analyzer was implemented and pushed at commit
`adcb0db3153e701644af5338335f266e80badeff`, tree
`cd4da1f7a1f2866a39c399bccecc71262b4390db`. The analyzer and Slurm wrapper
SHA-256 values are respectively
`0c48440f6eca91c0f48eff2407686e9bfc36c7273fb8a010a69fccf9f55c0779`
and `b3d471997733618c5dd711a9da3d8f13eacd5e719122561d41f9b5b1c2c154f7`;
the complete source bundle SHA-256 is
`405eec917037da902f0a7c6cc08d18f1fd047b6623a90bd8099b07c8bf26df8e`.

Cluster preflight array element `1215353` completed `0:0` on `g0005` in eight
seconds. All `14` selected pure-Python contracts passed, stderr was empty, and
the frozen inputs reconstructed exactly `3,007` lifecycle annotations,
`24,024` visible-event route outcomes and `24,008` fully-observed event/route
analysis units. The full read-only trace job is `1215356`, with append-only root
`/data/run01/sczc063/yuzibo/runs/eventmatr_d151/endpoint_margin_adcb0db_20260803_r1`
and submission-script SHA-256
`d79aa09243b06787539fa651f74b3a0db8c08937f7e61434c4c13b6d20e65fb6`.
The cluster requires a nominal GPU allocation for every job, but this analyzer
does not import or load a model, execute a forward pass, construct an optimizer,
update a checkpoint or access the locked test split.

### D1.5.1 final result — 2026-08-05

Full trace job `1215356` completed `0:0` in `00:42:39` on `g0005`, with empty
stderr. The receipt status is `PASS_DIAGNOSTIC`; its SHA-256 is
`ef5a437d613c92b08111ceb3764c10ed39872d532948b8892c2464461e22e4c5`.
Start and final source identities are byte-identical, clean and bound to
`adcb0db3153e701644af5338335f266e80badeff` /
`cd4da1f7a1f2866a39c399bccecc71262b4390db`.

The result contains exactly `24,008` rows: `3,001` common fully observed events
in each of eight routes. Formal endpoint-positive counts are zero in all four
channels. No-cancel shadows yield only `7/3001`, `6/3001`, `3/3001` and
`3/3001`. The two oracle-shadow medians are `-1.602913/-1.673983` and their 95th
percentiles are `-0.574103/-0.586817`; upper video-bootstrap bounds are
`0.005997/0.004984`, both far below the frozen `0.05` floor. The paired oracle
identity-shadow difference is `-0.0003332223`, interval
`[-0.0011223345, 0]`. There are no material state-machine mismatches.

An independent implementation recomputed every route, temporal bin, duration
stratum and all `10,000` common video-cluster resamples with zero mismatches. Its
summary SHA-256 is
`0c790d69abd324bc3c4b6afab8a3c057cfcb8650e61d2400cffcde31100d05dc`.
The full integrity record is
[eventmatr-d151-integrity-audit-20260805.md](eventmatr-d151-integrity-audit-20260805.md).

The frozen routing decision is therefore
`frozen_owner_end_representation_or_competing_risk_objective_insufficient`.
Perfect admission, oracle identity refresh and cancellation suppression do not
make endpoint END evidence materially positive. The only released next action
is `preregister_policy_independent_three_state_competing_risk_model_repair`.
Model implementation, model training, official comparison, locked-test access
and every paper-performance claim remain blocked.
