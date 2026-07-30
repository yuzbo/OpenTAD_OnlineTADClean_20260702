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
