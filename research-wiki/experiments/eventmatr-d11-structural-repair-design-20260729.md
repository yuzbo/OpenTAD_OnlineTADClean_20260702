# EventMATR D1.1 structural repair design — 2026-07-29

Status: approved for implementation by the user's conditional instruction:
if the route is fully understood, execute directly.

## Exact inputs

- frozen D1 diagnostic code: `6a23ab3a3711bc1ecb5a2fe442e302964ee3afb8`;
- frozen replay/wiki evidence: `7983f1692d439fe40041436e4c354240ec6f4330`;
- Pro review attachment SHA-256:
  `c2cd3ce0cc52791aecc6cda9d85ef108ce4e18703c0debf51026196817231bd1`;
- the attachment has 577 lines and was read in full before this decision.

This document records the implementation decision. It does not promote the
five-epoch train-only replay to a paper result.

## Agreement with the Pro review

### Accepted

1. Hold the unchanged 10/20-epoch jobs. More epochs cannot change deterministic
   target construction, batch-local history, or category-only targetless
   reacquisition.
2. Treat owner cancellation plus owner-end starvation as the confirmed direct
   zero-output mechanism, while keeping its upstream causes counterfactual.
3. Repair predicted-track supervision before changing capacity, thresholds,
   backbone, data or training budget.
4. Associate predicted births to prefix-visible events causally and one-to-one;
   exact equality with the teacher path's query index cannot be the identity
   condition.
5. Give active records explicit `CANCEL / CONTINUE / END` semantics. Candidate
   birth competition and active-owner lifecycle must not share an ambiguous
   `BACKGROUND / START / ALIVE / END` target vocabulary.
6. Carry causal temporal history across physical batches and reset it by video,
   not by the data loader's batch boundary.
7. Stratify supervision and lifecycle accounting by track source.
8. Compare censored hazards against a fair event-normalized ordinary causal
   loss only after the common track repair is working.
9. Keep locked test, multiple seeds, raw RGB and threshold search blocked.

### Accepted with corrections

1. Category-only reacquisition is not universal. It occurs only for archived
   records and new births whose `target_event_id` is both unknown. A
   training-only record with a known target ID already uses exact target-ID
   recovery.
2. In the available annotation ontology there is no labelled
   suspended/unobserved state. Cancelling a correctly associated event before
   its labelled end is an error; a later same-event association is therefore
   error recovery, not a positive identity transition.
3. The candidate transition head may remain four-way in the first repair
   because only its learned START competition is used to create D1 births.
   The required semantic split is made at the active-owner head, which becomes
   ternary. Replacing the candidate head with a binary head is deferred until a
   deletion study shows it is necessary.
4. A single-seed train-only study can establish mechanism and video-sampling
   uncertainty, but not optimization stability or held-out generalization.

### Not accepted as hard scientific gates

The Pro review proposed absolute 3/5/2 percentage-point effects and a 20%
relative cycle reduction. These values have no supplied power analysis,
measurement-error study, baseline variance or external minimum-important-
difference justification. They may be retained as provisional planning
targets, but they will not be used as hard pass/fail science gates.

Before a new five-epoch result, the project will freeze:

- effect direction;
- paired comparison;
- video-level resampling unit;
- confidence interval construction;
- non-inferiority metrics;
- integrity and liveness pause conditions.

An absolute minimum effect will be frozen only after a train-only pilot variance
and sample-size audit that does not inspect locked test results.

### Accepted but not claimed complete in D1.1

The structural repair closes the directly observed cancellation/end-starvation
path. It is not a claim that every Pro recommendation is already implemented.
The following items remain mandatory before a five-epoch result can be treated
as a scientific mechanism comparison:

1. targetless category-only archive reuse is prohibited now, but a real
   inference-time reacquisition mechanism based on identity evidence
   (pre-cancel/current embeddings, class, time and one-to-one admissibility) has
   not been implemented; D1.1 therefore does not claim solved reacquisition;
2. ambiguous equal-cost predicted/visible-event associations must be labelled
   ambiguous and left unbound rather than resolved by an arbitrary index;
3. physical-batch invariance must cover loss and gradient values within a
   declared numerical tolerance, not only associations, risks, lifecycle
   counts and ledger rows;
4. source-stratified accounting must be extended from rows/groups/transitions
   to per-source state predictions, loss contributions, gradient norms and
   ledger outcomes;
5. the fixed teacher-gate versus learned-gate counterfactual remains required
   at the same official working point;
6. seconds, original frames, feature indices and prefix indices must have one
   explicit reversible coordinate contract;
7. event-conditional recall, closure timing, cancellation/recovery, identity
   consistency and calibration outputs must be implemented before performance
   conclusions;
8. the ordinary event-normalized causal-loss control and censored-risk variant
   must share association, unroll, state semantics, initialization, update
   count and fixed working point.

The Pro review's suggested liveness percentages may be used as engineering
pause diagnostics. They are not paper-effect gates without a prospective
variance, power or minimum-important-difference justification.

## Considered implementation routes

### Route A — query-index patch only

Only copy a target ID to a predicted query selected by another matcher.

Advantages: smallest code diff.

Rejected because it leaves active-owner state ambiguity, targetless
category-only reuse, batch-local birth risk and unstratified supervision.

### Route B — complete new lifecycle framework

Replace candidate, owner, archive, history, decoder and loss interfaces in one
rewrite.

Rejected because it confounds several causal interventions, delays
falsification and violates the repository's model-first/minimal-verification
rule.

### Route C — staged D1.1 repair

Recommended and authorized. Preserve the frozen D1 evidence and implement four
separable contracts in the current model:

1. causal predicted-birth association;
2. ternary active-owner state semantics;
3. video-keyed cross-batch causal history;
4. source-stratified accounting and targetless-reacquisition removal.

Each contract receives a deterministic test before real-data execution.

## D1.1 design

### 1. Causal predicted-birth association

Training may use only target facts visible at the current prefix. Ground-truth
end time must not be part of the association cost.

At each real training prefix:

1. preview learned START rising edges without mutating runtime state;
2. collect currently visible target rows in `birth` or `alive` state;
3. exclude a target already owned by an active record;
4. obtain each target's causal temporal path from current and retained past
   history;
5. form admissible predicted-target pairs only when:
   - the predicted foreground top class equals the visible target class; and
   - the predicted start is within one declared feature-prefix window of the
     visible target start;
6. score admissible pairs with start distance, class evidence and current
   feature continuity to the target's causal path;
7. refuse an ambiguous equal-cost tie instead of resolving it by query index;
8. solve a deterministic one-to-one assignment over the remaining pairs;
9. mark assigned predictions `predicted_associated`;
10. leave unmatched predictions `predicted_unmatched`;
11. allow the existing teacher ratio to inject only unmatched visible targets,
    recorded as `teacher_birth` or `teacher_recovery`.

The target ID is training supervision metadata, never an inference input.

### 2. Active-owner state semantics

For D1 training and runtime, owner logits have three states:

| Index | Meaning | Training target |
|---:|---|---|
| 0 | cancel | unmatched/false predicted record |
| 1 | continue | associated record at birth or alive prefix |
| 2 | end | associated record at first observable end |

The candidate transition head remains frozen at four competitive states in
this repair. Its START state creates candidate births; its other states are not
used as active-record semantics.

The censored end hazard compares owner `END` against `CANCEL/CONTINUE`, uses
`CONTINUE` as the at-risk survival state and uses the first `END` as the
observed event.

### 3. Reacquisition semantics

- Known target ID during training: exact-ID recovery is retained and counted as
  `associated_error_recovery`.
- Unknown target ID during inference: category-only archive reuse is removed.
  A new birth creates a new immutable event ID.
- No positive “correct suspend then recover” target is introduced, because the
  dataset contains no suspension annotation.
- Identity-embedding recovery remains a later hypothesis and requires a
  separately registered admissibility rule and counterfactual.

### 4. Cross-batch causal history

The model owns a cache keyed by video name. Each entry contains:

- feature-prefix index;
- candidate state logits;
- class logits;
- query features.

Rules:

1. append only real prefixes in monotonic order;
2. retain only the declared causal history window;
3. padding is a no-op;
4. clear on video reset;
5. detach carried tensors at the forward/optimizer boundary;
6. preserve the current forward's computation graph;
7. expose the selected birth-risk sequence to the criterion, avoiding a second
   inconsistent batch-local matcher.

Changing physical batch splits must not change target association, risk-set
membership, lifecycle transitions or ledger rows.

### 5. Source-stratified accounting

The following sources must be distinguishable:

- `predicted_associated`;
- `predicted_unmatched`;
- `teacher_birth`;
- `teacher_recovery`;
- `associated_error_recovery`.

For each source, report:

- number of dynamic records;
- owner-state rows;
- class-supervised rows;
- end-risk groups;
- cancellations;
- ends;
- emissions;
- recoveries.

Ground-truth matching used for scientific diagnostics remains post-forward and
must be labelled diagnostic-only.

## Deterministic implementation gates

The first code revision may proceed to real data only if all of these pass:

1. query-index permutation cannot prevent a uniquely admissible prediction
   from receiving the visible target identity;
2. one visible target cannot bind two active predicted records;
3. an associated birth/alive record receives `CONTINUE`, not `CANCEL`;
4. an unmatched predicted record receives `CANCEL`;
5. an associated first end receives `END`;
6. targetless category equality cannot reuse an archived event ID;
7. exact known-target recovery preserves the immutable event ID and is labelled
   error recovery;
8. whole-sequence and physically split processing have identical associations,
   risk members, lifecycle counts and ledger rows;
9. padding and observed stream end retain the frozen strict-causal contracts;
10. no future/full-video metadata reaches the model.

## Execution order

1. implement and test the pure association helper;
2. change D1 active-owner output/targets/runtime to ternary semantics;
3. remove targetless category-only archive reuse;
4. add the persistent causal history and birth-risk group output;
5. add source-stratified loss/accounting counters;
6. run local syntax and deterministic contracts;
7. run the remote controlled test suite;
8. only then run a real one-batch supervision audit;
9. if it passes, register one epoch;
10. a new five-epoch, seed-52, train-only matrix remains blocked until the
    one-epoch mechanism receipt passes.

## Scientific claim boundary

This revision tests one dominant claim:

> Closing predicted-track supervision and physical-batch invariance is
> necessary before censored birth/end risk and identity lifecycle can be
> evaluated fairly in strict-causal online temporal action localization.

It does not yet claim better detection, correct reacquisition, test
generalization or paper-level novelty.

## Implementation record

The repair was implemented on `codex/eventmatr-d1` as a sequence of exact,
reviewable revisions:

| Revision | Tree | Purpose |
|---|---|---|
| `3d26d0f3831461036a1c54e30f2d323bb504ee97` | `53013140...` | causal association, ternary owner, cross-batch history, censored risks, source accounting and strict contracts |
| `903541034ac2f9accb239478716c28366f0209fe` | `ba37e20b...` | preserve the frozen v1 candidate-birth decision while keeping D1.1 margin decisions |
| `d8b2bc80abe6afe2af02415fa68b3d76e7d0ac92` | `4100504d...` | align the runtime stress experiment with targetless-new-ID and known-target exact recovery semantics |
| `0419f93aaf131a3c02817771b1d4223b09f0083f` | `4944c666...` | censor synthetic non-END endpoints and expose an observed END only at its current visible prefix |
| `c6e53ad4081f6efa1ee6a4bd93622ea22472f3eb` | `6ebba374...` | register the one-epoch integrated mechanism gate, checkpoint schema and runtime/gradient observability |
| `de0837cf38e05d65a40f0744b863056edc2f433a` | `91d998e7...` | register the new mechanism protocol in the shared argument parser |
| `f37e9d191a06a8a703714a0cd857180c21fc069e` | `34b2a81b...` | record the failed one-epoch gate, emit explicit zero-valued source metrics, instrument every association barrier and register a read-only strict-causal checkpoint scan |
| `7687efe03981aeb1ee3cb62ae2fd94d6e6ca1dfa` | `a85303e0...` | preserve the frozen 64-row MATR physical batch during the scan and verify that padding is a lifecycle no-op only after an observed current-stream EOS |

The implemented contracts are:

1. a predicted START is associated one-to-one using only current/past class,
   start and feature-continuity evidence;
2. the active owner uses `CANCEL / CONTINUE / END`;
3. targetless rebirth receives a new event ID, whereas a known target ID may
   recover only its exact archived record and is labelled error recovery;
4. temporal state is video-keyed and invariant to physical batch splits;
5. birth and end learning use event-normalized censored risk groups;
6. source-specific rows, groups, risks and lifecycle transitions are logged;
7. runtime birth/end/cancel/emit/recovery/capacity counts and both key gradient
   norms are exposed to the training receipt;
8. a D1.1 checkpoint declares
   `eventmatr_d11_ternary_owner_v1`, lifecycle, lane and owner-state count.
   Missing or four-state metadata is rejected instead of silently mapped.

## Independent review findings and resolutions

Three independent read-only audits were performed before the one-epoch
protocol revision was committed.

- The protocol audit found no relaxation of the registered 5/10/20-epoch
  matrix. The new route is a singleton: lane `TH`, seed 52, one epoch, fresh
  start and train-only.
- The checkpoint audit confirmed that old four-state owner heads are
  semantically and dimensionally incompatible. It found that a D1.1
  checkpoint could still be offered to a v1/native loader; the loader was
  tightened to reject an explicitly mismatched lifecycle while retaining
  compatibility with old v1 checkpoints that have no new metadata.
- The finalizer audit found three unverified options. The final gate now also
  checks `reduce=1`, output generation enabled and unlimited dynamic
  capacity.

A second independent scientific audit was performed while the frozen
one-epoch job was running. It confirmed that the mechanism gate is reasonably
minimal: positive finite gradients and non-empty supervision prove only
liveness, while zero capacity exhaustion is an integrity condition. It also
confirmed the open items above and found that the existing 5/10/20-epoch
matrix is not, by itself, a clean two-factor experiment:

- `R -> H` and `T -> TH` can estimate conditional censored-risk effects;
- `R -> T` also changes identity and ownership behavior, so it is not a pure
  trajectory-only contrast;
- `N -> R` changes dense/native learning, event normalization and track repair
  together;
- the current pilot finalizer exposes train-prefix diagnostic values but does
  not yet operationalize all registered lifecycle, identity, calibration and
  systems metrics.

Consequently, a passing one-epoch receipt may release design work for the
five-epoch comparison, but cannot by itself release the old matrix unchanged
or support a novelty/performance claim.

The fixed upstream flag/class/non-maximum-suppression thresholds remain the
official MATR values. They are not EventMATR birth/end thresholds and are not
searched or lowered. Event birth/end thresholds remain absent.

## Remote evidence chronology

All jobs used the official training/validation annotation and feature file,
one visible accelerator, seed 52, and no locked test path.

| Job | Exact source | Outcome | Interpretation |
|---:|---|---|---|
| `1203957` | `3d26d0f...` | submission exited before work | malformed manifest export; not a scientific run |
| `1203958` | `3d26d0f...` | 95/97 tests | exposed one v1 compatibility mistake and one incorrect equal-max gradient expectation; both repaired |
| `1203969` | `9035410...` | 97/97 tests, then old stress assertion failed | the assertion still demanded forbidden category-only targetless recovery |
| `1203977` | `d8b2bc8...` | exited in one second | environment activation was not exported; not a scientific run |
| `1203985` | `d8b2bc8...` | 97/97 tests, then causal guard rejected the synthetic sample | synthetic birth/alive rows still exposed a future endpoint; guard behaved correctly |
| `1203995` | `0419f93...` | `COMPLETED 0:0` | full controlled test, mechanism and real-batch smoke passed |
| `1204044` | `c6e53ad...` | `COMPLETED 0:0`, 99/99 tests | exact-source smoke for the registered one-epoch gate passed |
| `1204046` | `c6e53ad...` | parser exit before dataset/model startup | the shared parser did not yet list the new protocol; no training or checkpoint update occurred |
| `1204052` | `de0837c...` | `COMPLETED 0:0`, 99/99 tests | corrected exact-source smoke passed |
| `1204061` | `de0837c...` | all 3,270 training batches and checkpoint completed; finalizer `FAILED 1:0` | the mechanism gate correctly rejected the run because no `predicted_associated` source occurred |
| `1204173` | `f37e9d1...` | exited in one second before Python/model startup | temporary smoke wrapper used `/bin/sh`, which rejected a Bash option; no checkpoint load or scientific result |
| `1204188` | `f37e9d1...` | exited in two seconds before Python/model startup | formal script correctly failed closed because the environment activation path was not exported; no checkpoint load or scientific result |
| `1204196` | `f37e9d1...` | failed after `00:01:53` at the first variable-width forward | filtering padding reduced one physical batch from 64 to 57 rows and violated frozen MATR memory-queue dimensions; no barrier result or scientific inference |
| `1204217` | `7687efe...` | submitted after `102/102` exact-source tests | corrected full scan preserves every 64-row physical batch and verifies post-EOS padding as a lifecycle no-op |

For job `1203995`:

- 97 tests passed;
- the microexperiment receipt is
  `eventmatr_d11_local_microexperiments_v2`, status `PASS`;
- dense positive-gradient dilution scaled exactly with query count: ratios
  2, 10, 100 and approximately 1000;
- all four eventized diagnostic lanes had non-zero transition and owner
  gradients;
- the runtime stress held 12 unique active records over a 10-query bandwidth,
  minimum start was zero and capacity exhaustion was zero;
- targetless cancelled event `1` rebirthed as new event `12`;
- known target event `0` recovered event `0` once by `exact_target_id`;
- the official real batch contained 25 valid event rows;
- every eventized registered lane had finite gradients and passed strict
  checkpoint reload;
- the integrated lane's required gradient norms were approximately 82.30
  (candidate transition), 262.38 (owner cross-attention) and 319.83
  (owner state);
- the source receipt was clean at commit
  `0419f93aaf131a3c02817771b1d4223b09f0083f`, tree
  `4944c666a2e87d69bec79f4a4ffe12649050c7b3`;
- `test_access=false`, `checkpoint_updated=false`, and
  `strict_causal_paper_result_valid=false`.

These are mechanism and execution facts, not a detection-performance claim.

## One-epoch mechanism verdict

Job `1204061` completed the complete one-epoch optimization trajectory before
the fail-closed finalizer ran:

- all `3,270/3,270` batches completed;
- the terminal D1.1 checkpoint exists and is `2,150,323,335` bytes;
- elapsed time was `00:23:45`, with no capacity exhaustion;
- transition and owner gradient norms were approximately `62.2584` and
  `51.5407`;
- birth, end, owner, ragged-track and false-track-cancel supervision were all
  non-empty;
- runtime birth, end/emission, cancellation and reacquisition paths were all
  active;
- predicted-unmatched association rows were non-empty;
- both predicted-associated metrics were absent, which under the then-current
  dynamic metric emitter means an exact observed count of zero, not a finalizer
  spelling error.

The finalizer therefore returned `FAILED 1:0`. This is a mechanism-gate
failure, not a training crash: the network and lifecycle had gradients, but the
learned predicted-birth path never acquired supervised event identity.
Teacher-created records kept the remaining lifecycle live and cannot substitute
for that missing learned path.

The raw epoch-average logger values used for this verdict are:

| Field | Value | Meaning |
|---|---:|---|
| total optimized loss | `13.5564959` | finite optimization trajectory |
| candidate-transition gradient norm | `62.2584` | candidate head is not gradient-dead |
| active-owner gradient norm | `51.5407` | owner head is not gradient-dead |
| birth-positive supervision | `0.91835` | non-empty logged batch average |
| end-positive supervision | `0.54832` | non-empty logged batch average |
| owner-assignment supervision | `0.91835` | non-empty logged batch average |
| ragged-track supervision | `7.5052` | chronological owner rows are present |
| false-track cancel groups | `6.1073` | explicit cancel learning is active |
| runtime births | `8.8523` | dynamic birth path executes |
| runtime ends/emissions | `2.39847` | close/write path executes |
| runtime cancellations | `6.41743` | learned cancel path executes |
| runtime reacquisitions | `2.17829` | known-target error recovery executes in training |
| runtime capacity exhaustion | `0` | no event was silently dropped |
| predicted-unmatched rows | `31.6158` | learned births exist but remain unbound |
| predicted-associated rows | `0` | required learned identity path never closes |
| train-prefix average detection score | `0.004903` | diagnostic only; invalid as paper performance |

These values are logger averages over the completed epoch, not independent
replicates or video-level effect estimates. The exact checkpoint is
`2,150,323,335` bytes with SHA-256
`a5e686f3e4800e654e9ed4366f13c298697ab6b076c9813f10e0f5241ba546b5`;
the exact options SHA-256 is
`726aebcbdf2275a5928a99792d18933824d0ff2878eb10eccd15a839eb58451f`.

The recorded training-prefix average detection score was approximately
`0.004903`. It uses the inherited train-prefix evaluator and is diagnostic only:
it is not a strict-causal paper result, does not use the locked test, and cannot
be compared as held-out performance.

### Gate audit

The one-epoch gate is retained unchanged. It asks only whether both learned
heads receive finite gradients, required supervision paths occur at least once,
capacity remains intact and the locked test is absent. It has no detection
effect-size threshold. Requiring at least one predicted-associated record is
reasonable because the D1.1 claim specifically requires the learned
train/inference track path to close; removing or lowering this condition would
let teacher forcing conceal the exact failure the repair was meant to solve.

The gate does not establish performance, statistical significance or novelty.
The proposed absolute percentage-point and relative-cycle thresholds remain
unjustified without a prospective variance and power analysis.

### Authorized root-cause scan

Commit `f37e9d191a06a8a703714a0cd857180c21fc069e`, tree
`34b2a81b2ee4a3df1d2523dcaeb0189200f68774`, adds the barrier
instrumentation. Commit `7687efe03981aeb1ee3cb62ae2fd94d6e6ca1dfa`,
tree `a85303e04f810fd582db848ecc6ce69fc7fbf196`, supplies the corrected
execution path for the only experiment authorized after this failure:

1. replay the saved epoch-one checkpoint in evaluation mode on all 200 official
   train/validation videos;
2. preserve the frozen 64-row MATR physical batch; expose the structural
   padding mask only so rows after an already observed current-stream EOS are
   guaranteed not to mutate lifecycle state;
3. reject padding before EOS, real prefixes after EOS, non-monotonic rows,
   non-64 physical batches, or any padding birth/end/cancel/emit/reacquisition;
4. pass no target, duration, complete-video timing or supervision flag to the
   model;
5. consult prefix-visible training targets only after forward;
6. count active START decisions, rising births, visible first-birth and
   alive-recovery opportunities, class mismatches, start-distance rejections,
   admissible pairs, ambiguities and assignments;
7. close every count globally and per video, require one observed EOS per
   complete video, and reject capacity exhaustion;
8. pin and recheck the unchanged checkpoint SHA-256
   `a5e686f3e4800e654e9ed4366f13c298697ab6b076c9813f10e0f5241ba546b5`
   and options SHA-256
   `726aebcbdf2275a5928a99792d18933824d0ff2878eb10eccd15a839eb58451f`;
9. keep the one-epoch mechanism verdict `FAIL_UNCHANGED` regardless of scan
   completion.

This is a necessary-condition opportunity scan. Because it deliberately keeps
ground truth outside the model, it does not retroactively reconstruct the
training run's teacher-owned target exclusions and cannot itself pass a
mechanism or performance gate. Remote exact-source tests at both `f37e9d1`
and the corrected `7687efe` report `102 passed`.

The first full-scan attempt, job `1204196`, exposed an implementation-only
conflict: deleting seven padding rows made the physical batch width 57 while
frozen MATR retained a 64-column memory queue. It failed before producing any
barrier counts. The correction does not modify the model or checkpoint; it
retains the inherited physical schedule and proves row by row that the
post-observed-EOS padding mask is a structural no-op. This mask is true on every
actual observed prefix and false only after EOS, so it supplies neither a future
duration nor an offline end signal for any real prediction.

The interpretation rule was frozen before reading the full scan result:

| Observed closure | Supported diagnosis | Authorized response |
|---|---|---|
| no learned rising births | START decision itself is inactive at the saved checkpoint | inspect learned START competition; do not alter association thresholds |
| learned births and visible targets, but no same-prefix overlap | one-frame rising-edge admission misses the supervision interval | replace ephemeral admission with a causal pending-candidate interval |
| overlap exists, but pairs are overwhelmingly class-mismatched | hard top-class equality is the blocking gate | test class evidence inside a registered one-to-one cost, not a searched cutoff |
| class-compatible pairs exist, but start distance rejects them | predicted start coordinates or the admission interval are inconsistent | repair the coordinate/interval contract; do not widen a threshold post hoc |
| admissible pairs exist, but ambiguity prevents assignment | identity evidence is insufficient for stable pre-birth binding | add predeclared temporal identity evidence and leave unresolved ties unbound |
| scan assignments exist while training assignments were zero | training ownership/exclusion or teacher handoff, not raw opportunity, blocks the learned path | repair oracle-to-predicted ownership handoff and rerun the unchanged one-epoch gate |

This table is diagnostic routing, not a performance gate. More than one barrier
may be present; raw counts and conditional rates must be reported before
selecting the smallest repair.

## Registered one-epoch mechanism gate

Revisions `c6e53ad4081f6efa1ee6a4bd93622ea22472f3eb` and
`de0837cf38e05d65a40f0744b863056edc2f433a` add an explicit gate before the
short training matrix:

- only the integrated trajectory-plus-censored-hazard lane;
- exactly one epoch and seed 52;
- fresh initialization; resume is forbidden;
- official train/validation data only;
- locked-test feature path is an absent sentinel;
- no EventMATR birth/end thresholds;
- a passing exact-source real-batch smoke is required;
- finite, positive transition and owner gradients;
- non-empty birth, end, owner, ragged, predicted-associated,
  predicted-unmatched and teacher-birth supervision;
- zero capacity exhaustion;
- a non-empty terminal checkpoint with the ternary-owner schema.

No minimum detection improvement is applied at this stage. Training-prefix
detection metrics remain diagnostic-only. The one-epoch receipt is currently
failed, so the five-epoch design is not released. Even a later passing
one-epoch training receipt would make only the scientific contract eligible for
prospective revision; a diagnostic scan completion can never substitute for
that receipt. The existing 5/10/20-epoch matrix also remains blocked until the
full scientific outputs and matched comparisons listed above are frozen. A
non-zero training-prefix detection score cannot release it.

## Five-epoch comparison that may follow a PASS

The next experiment remains train-only, strict causal, seed 52 and
predicted-track inference. Checkpoint age is repeated observation of one
optimization trajectory, not an independent replicate.

The five named lanes are interpreted as a parent anchor plus a bundled
two-factor comparison:

- `R` is the common repaired event-normalized/ragged base;
- the trajectory bundle is estimated by paired `T-R` and `TH-H`;
- the censored-risk bundle is estimated by paired `H-R` and `TH-T`;
- their interaction is `TH - T - H + R`;
- `N-R` is a parent-to-repaired-system comparison, not an isolated causal
  component effect.

Before submission, the implementation must emit per-video rows for detection,
birth/end/emit timing, closure, cancellation/recovery, duplicate/fragmentation,
identity consistency, calibration, throughput and memory. All metrics must be
computed from the same frozen checkpoints and strict-causal predicted-only
replay; ground truth may enter only the post-forward diagnostic matcher.

The statistical unit is a video. Report paired per-video differences, median
difference, win rate and a video-resampled 95% confidence interval, with
predeclared same-class-overlap and action-density strata. Epochs 5/10/20 are
checkpoints on one trajectory and must not be counted as three samples.

Hard stops remain source mismatch, future/ground-truth inference leakage,
non-finite values, non-positive intervals, duplicate or mutable ledger output,
silent capacity exhaustion, or missing metric rows. Zero terminal emission or
collapsed coverage is a functional-liveness pause that blocks longer training
and triggers diagnosis; it is not an arbitrary paper-performance rejection.
No non-inferiority margin or minimum effect is valid until the train-only
video-level variance audit has been completed and its rule frozen without
locked-test inspection.

## Post-failure diagnosis: optimization exposure is a controlled confound

The failed one-epoch receipt remains failed, but its original causal
interpretation must be narrowed. A scheduler audit established that all 3,270
updates in job `1204061` used the optimizer floor `1e-8`. The first registered
warmup rate, `3.34e-6`, was applied only by the epoch-end scheduler step after
the final update. Therefore the run proves path execution, supervision,
gradients, deterministic completion and checkpoint production. It does **not**
constitute a meaningful test that the repaired association/lifecycle can learn
under the registered schedule.

### Deterministic terminal checkpoint scan

Read-only job `1204338`, source
`d14ab88f90df1ab960dbc164700e884ff922f0c9`, tree
`633f94cef34ea0bb6bbec7fd84a49df8102c9377`, completed all 200 official
train/validation videos:

- 203,363 real prefixes, 5,917 verified post-observed-EOS padding no-ops and
  exactly 200 observed current-stream EOS rows;
- 2,033,630 query-prefix START margins, all below the argmax boundary zero;
  the maximum was `-0.246129`;
- 3,003 visible first-birth targets and 56,551 visible alive opportunities;
- zero predicted START, birth, assignment, runtime birth, cancel, end, emit,
  reacquisition and capacity exhaustion;
- checkpoint SHA-256 remained
  `a5e686f3e4800e654e9ed4366f13c298697ab6b076c9813f10e0f5241ba546b5`;
- scan JSON SHA-256 was
  `480e2e006240c39d419bdc99e432381655071780481aa42c970d52b3f9bb34aa`;
- source identities were clean and unchanged; `test_access=false`,
  `checkpoint_updated=false`, `threshold_search=false`.

The oracle-path diagnostics show a non-trivial **relative** query signal even
though the absolute START decision is suppressed:

- first-birth oracle query top-1/top-3 rates were approximately
  `45.89%/84.05%`;
- oracle-versus-other-query preference was approximately `87.82%`;
- top-class compatibility was `195/3003 = 6.49%`;
- registered start-distance compatibility was `192/3003 = 6.39%`;
- joint class-and-distance compatibility was `7/3003 = 0.233%`;
- every oracle-path START margin was still negative, with mean approximately
  `-3.2243` and maximum approximately `-0.9157`.

These observations establish absolute START suppression at the saved terminal
checkpoint. They do not identify whether the suppression is caused by
undertraining, the head structure or both.

### Same-trajectory training-mode barrier trace

Instrumented job `1204354` completed the same 3,270-batch trajectory under
source `d14ab88...`. Its terminal checkpoint was byte-identical to job
`1204061`, so the instrumentation did not change training. Across the actual
training trajectory:

- 23,039 prefix-visible target opportunities co-occurred with 18,021 predicted
  birth queries;
- those rows created 4,042 predicted-query/visible-target pairs;
- 3,900 pairs (`96.49%`) failed the frozen top-class equality condition;
- all remaining 142 pairs failed the frozen start-distance geometry;
- admissible pairs, ambiguities and assignments were all zero;
- predicted-unmatched rows totalled 76,840 and teacher-created rows totalled
  33,657;
- capacity exhaustion remained zero.

This trace proves that train-mode predicted births and visible targets do
co-occur. It also proves that, on this underexposed trajectory, class mismatch
is the dominant immediate barrier and geometry rejects every class-compatible
remainder. It does **not** prove either barrier is structurally incurable.

The training trace and deterministic terminal scan are not contradictory. The
former observes changing weights and train-mode stochasticity across the
optimization trajectory; the latter observes only the final checkpoint in
evaluation mode. Their difference is itself a reason to require a separate
deterministic terminal liveness gate.

### Exact parameter-delta and optimizer-step audit

Job `1204424`, audit source
`daea1149ecfc2cf7caedc627b4381832c5005b53`, tree
`394af68ba51256d085653a461e5e6508cd500be1`, reconstructed the exact
seed-52 initialization from training commit `de0837c...` and compared every
parameter with the terminal checkpoint without a forward or optimizer step.

Facts:

- all-parameter relative L2 delta was `0.000216732`;
- transition-head relative L2 delta was `0.000278531`;
- owner-decoder relative L2 delta was `0.000211624`;
- 170,126,395 of 196,459,591 parameter elements changed;
- the recorded average update learning rate was
  `1.0000000000000811e-8`;
- the optimizer contained 315 parameter tensors, 273 initialized Adam states
  and 42 states never initialized because those parameters never received a
  gradient;
- per-parameter Adam steps were
  `1258×2, 2016×2, 2022×4, 2345×14, 3270×251`.

The non-uniform Adam state steps are valid conditional-gradient behavior, not
evidence that global batches were skipped. The DataLoader and maximum
parameter step independently close all 3,270 batches. The audit therefore
supports “very small but non-zero parameter movement under a floor-rate
trajectory,” not “no optimization happened.”

## Revised mechanism gate and controlled effective-dose recheck

The previous verdict is preserved:

> Job `1204061` remains `FAIL_UNCHANGED` because its learned
> predicted-associated path was zero.

The stronger claim “the D1.1 mechanism cannot learn” is withdrawn. It was
confounded by the floor-rate trajectory. The only authorized intervention is a
new fresh seed-52 run with the same official data, order, model, losses,
teacher/predicted mixing and association decisions, but with one explicit
scheduler advance before training so all 3,270 updates use the first registered
warmup rate `3.34e-6`.

No decision threshold is changed:

- zero START margin is the equality boundary of the registered state argmax,
  not a searched scalar threshold;
- inherited flag `0.5`, class `0.1` and non-maximum-suppression `0.3` settings
  remain unchanged and are not EventMATR birth/end thresholds;
- EventMATR birth/end scalar thresholds remain absent;
- no detection score, percentage-point gain, relative improvement, coverage
  percentage or arbitrary cycle ratio is a gate.

Release now requires three independently closed artifacts from the same exact
checkpoint and training source:

1. **training receipt:** exactly 3,270 optimizer updates, a 3,270-row
   actual-learning-rate trace fixed at `3.34e-6`, finite positive transition
   and owner gradients, non-empty registered supervision sources, a non-empty
   predicted-associated source and zero capacity exhaustion;
2. **deterministic terminal scan:** evaluation mode, predicted-only model
   input, ground truth only after forward, positive START/birth/assignment and
   runtime birth/cancel/end paths, immutable emission count equal to END count,
   and zero capacity exhaustion;
3. **parameter audit:** exact-source reconstruction, closed global/conditional
   optimizer steps, complete all-parameter group accounting and finite non-zero
   transition/owner parameter deltas.

The combined receipt is a functional-liveness gate only. A pass may make the
five-epoch science contract eligible for prospective freezing. It is not
detection performance, held-out generalization, statistical significance,
correct reacquisition or novelty evidence. A fail identifies which functional
path remains unclosed and authorizes only the corresponding read-only
diagnosis.

Implementation source
`4116df154014915cc190eec0e108a94c8df5f762`, tree
`2114f097eefc24e3a776147fc201711878464ad2`, adds the fixed-dose trace,
terminal scan, exact parameter audit and three-artifact finalizer. Remote job
`1204465` passed all `106` protocol tests and the official real-batch smoke with
`test_access=false`. Job `1204464` exited before testing because its submission
omitted the environment activation path; it produced no scientific result.

Commit `8ceee52c102913ac08571c1bd876fb370f9b5797`, tree
`b3944ccef0ea752a4726bb75d35adcc33bed83e1`, additionally permits a failed
effective-dose run to undergo read-only diagnosis as
`FAIL_EFFECTIVE_DOSE`. That status can never satisfy the combined PASS gate.

The controlled one-epoch job `1204468` subsequently completed. Its failed
functional outcome and the authorized structural response are recorded below;
no longer pilot was released.

## Effective-dose outcome and D1.2 START repair decision

The controlled recheck is complete and did **not** pass the functional
mechanism gate.

- Slurm job `1204468` completed all 3,270 optimizer updates at the recorded
  learning rate `3.34e-6`. Its terminal checkpoint has SHA-256
  `74898f27607bfb724090844f6fd0e5534a313d2c46e03e684fc750e443bac91a`.
  The 3,270-row learning-rate trace has SHA-256
  `c29dff48f654ae476ede0020c595b36b1df35c07276b8b42e3f95d04d11c44b1`.
- The finalizer correctly rejected the run because both learned
  predicted-associated counts remained zero. During the chronological training
  trajectory there were only `13` predicted birth queries, `3` candidate
  query-target pairs, all `3` failed the class condition, and there were no
  admissible pairs or assignments.
- Read-only parameter audit job `1204510` completed successfully. Its JSON
  SHA-256 is
  `28568ebe0f107181f9608aaf29a58f41ade217f7f9069e47aaf3e65e2b09b2a9`.
  Relative parameter changes were `0.00979963` for the full model,
  `0.01019128` for the event transition head and `0.00897211` for the owner
  decoder. This is substantial movement relative to the floor-rate run and
  rules out “the only problem was the tiny learning rate” as a sufficient
  explanation.
- Predicted-only terminal scan job `1204508` completed successfully as a
  diagnostic. Its JSON SHA-256 is
  `aa33fde9122a259d3164243510567c194dc45109f369c7f6d1d28925f2b62c8a`.
  Across `203,363` real prefixes and `2,033,630` candidate queries, no START
  score was positive. The maximum START margin was `-2.319654`, the mean was
  `-4.158487`, and runtime birth/cancel/end/emit/reacquisition counts were all
  zero. Capacity exhaustion remained zero.
- The same terminal scan still found non-random conditional structure on the
  ground-truth-class oracle path: for `3,003` visible births, pairwise query
  preference was `0.767899`, top-class compatibility was `0.642691`, temporal
  start-distance compatibility was `1.0`, and their joint compatibility was
  `0.642691`. These are post-forward diagnosis values, not runtime assignments
  and not detection performance.

The evidence supports a narrow conclusion: optimization exposure and parameter
motion are real, but the absolute START decision is structurally suppressed.
It does not establish generalization, superiority to MATR, correct
reacquisition, novelty, or that EventMATR is unsolvable.

The next authorized repair is therefore D1.2:

1. only for `d1_censored`, add a learned binary birth-risk output independent
   of the four-state background/START/alive/end competition;
2. use that same scalar output for event-normalized interval-censored birth
   training, causal temporal assignment, rising-edge preview and runtime birth;
3. retain the current state output for legacy `v1_dense` compatibility and
   owner/end diagnostics, while the D1 ternary owner decoder remains the
   authority for cancel/continue/end after birth;
4. keep the formal decision at zero log-odds. No scalar threshold is searched
   or lowered;
5. write a new checkpoint schema and reject silent loading of pre-D1.2
   checkpoints;
6. add a deterministic contract proving that a positive binary birth risk
   produces a birth even when the legacy background state is the four-state
   winner.

Alternatives were rejected prospectively. Reweighting the old four-state START
class retains the background-dominated coupling, while threshold search changes
only the decision surface and would contaminate the mechanism diagnosis.

### Official-comparability boundary

All evidence in this section is train-only mechanism diagnosis with
`test_access=false` and `strict_causal_paper_result_valid=false`. None of its
training-prefix detection scores, oracle-path values, lifecycle counts or
parameter deltas may enter a paper performance table.

A paper-grade comparison is allowed only after the repaired route passes the
same functional mechanism gate and the protocol is prospectively frozen. The
native MATR anchor and EventMATR arm must then use the same official THUMOS14
train/test split, official pre-extracted RGB-plus-flow features, 100-epoch
training budget, optimizer and scheduler, seed policy, terminal-checkpoint
policy, post-processing and official evaluator. Inference must remain strictly
causal, with no true duration, offline end-of-stream signal, future feature,
future metadata or ground-truth model input. Locked test access remains
forbidden until that release.

## D1.2 prospective execution protocol

Implementation source `69039b990822d689592155d48f57b619ecd8e25e`, tree
`7fc5e014b919d033468c9b58afdedc33afc65646`, implements the narrow repair
above. The D1 path now has one independent binary birth-risk head; its scalar
feeds interval-censored birth learning, causal temporal assignment,
rising-edge preview and runtime birth. The legacy four-state START margin is
retained only for the frozen `v1_dense` route. A new
`eventmatr_d12_independent_birth_v1` checkpoint schema rejects older lifecycle
weights rather than silently mapping them.

The repair adds exactly `1,025` parameters to the previous
`196,459,591`-parameter model (`0.000521736%`). This must be disclosed in any
future matched comparison. It is too small to serve as a credible generic
capacity explanation, but a parameter-matched dummy-head control remains
available if reviewers challenge the attribution.

The prospective validation order is fail-closed:

1. deterministic contracts must show that positive binary birth risk remains
   live even when the four-state background class wins;
2. the complete remote suite and one official training-batch
   forward/backward/optimizer/checkpoint-reload smoke must pass on the exact
   source;
3. one fresh seed-52, train-only epoch must execute exactly 3,270 optimizer
   updates at `3.34e-6`;
4. the same terminal checkpoint must close the mechanism receipt, complete
   predicted-only terminal lifecycle scan and exact parameter-delta audit;
5. only a combined pass may release prospectively frozen development pilots.

The birth decision is fixed at strictly positive binary log-odds. This is the
natural untuned decision boundary, not a searched or lowered threshold. The
gate asks only whether every required learned lifecycle path is non-empty and
whether invariants close; it sets no detection-effect threshold.

Even a complete D1.2 mechanism pass is not an official performance result. The
5/10/20-epoch seed-52 matrix, if later released, remains development evidence.
Paper-comparable evidence requires a separately frozen, matched 100-epoch
native-MATR/EventMATR study using the identical official split, extracted
RGB-plus-flow features, optimizer/scheduler, post-processing, terminal
checkpoint policy and evaluator. Only then may the locked test be accessed.
Before submission, that contract must also hash the annotation/features,
source/tree/manifest and terminal checkpoints; name the exact evaluator commit
and temporal-overlap grid `0.3/0.4/0.5/0.6/0.7`; and freeze native MATR versus
TH as the main pair with R/T/H as component ablations. The retired four-cell v1
labels must not be silently mixed with the D1 factorization.

### Exact-source D1.2 integration validation

Remote job `1204791` completed `0:0` on exact source `69039b990822...`.
The complete suite passed `108/108`. The official THUMOS14 real-training-batch
smoke then ran native MATR plus R/T/H/TH; every lane completed forward,
backward, one optimizer step and strict checkpoint reload. All D1 lanes had
finite positive shared-transition, independent-birth, owner-state and
owner-cross-attention gradients. The smoke receipt SHA-256 is
`8b2598d10eaf466256a2cfd5e36860b555f299ed2d3b112e0abdc9cf2f5f5240`;
it records `test_access=false`, `checkpoint_updated=false` and
`strict_causal_paper_result_valid=false`.

Three earlier submissions are excluded from model evidence. Job `1204775`
stopped on an incorrect externally supplied receipt filename. Jobs `1204779`
and `1204786` showed that the old smoke contract still expected the now-unused
D1 four-state-head gradient and then its old receipt key. Those mismatches were
fixed and the full same-commit validation was rerun as `1204791`; none of the
excluded jobs trained a model or produced a paper metric.

## D1.3 factorial mechanism closure

Exact source `e2a65de9744ba01c2d32a1406af0cf1a076e697a`, tree
`0cee49dcb2dce2e04f3c436c95017771f773d64c`, ran the registered seed-52,
one-epoch factorial intervention over soft temporal assignment and
event-matched birth negatives. Every arm completed all `3,270` physical
batches. The common birth census was `3,003` positive events, `1,685`
positive-event batches, `1,585` zero-event batches, `180,182` pre-birth
exposures, `5,666` interval exposures and `4` deliberately ignored
post-interval exposures.

The interventions were operationally real. Event-matched arms selected exactly
`3,003` negatives rather than allowing the physical background count to
dominate. Soft-assignment arms produced five predicted associations, and the
audit observed fourteen admissible pairs for which class argmax would have
rejected the soft target. However, every one of the `2,033,630` terminal binary
birth logits remained non-positive. Predicted birth, cancel, end, immutable
emission, reacquisition and capacity exhaustion were all zero. D1.3 therefore
established that hard assignment and negative counting are intervenable
mechanisms, but neither was sufficient to restore terminal lifecycle liveness.
It released no longer pilot and no paper result.

## D1.4 decision-alignment structure gate

### Prospective contract and exact-source preflight

Exact source `fa27b3657b72c5b713ee3d2a5c0e652e7ca14eb4`, tree
`7603fc6b8226fa8f9dcb3d5212136fb47631bc32`, manifest SHA-256
`fb1d4036f6954b48074038460cc3f148059805df0487ae21b8a4692fd9e112fa`,
registered two and only two one-epoch arms:

1. `normalized_survival`: replace each event's pre-birth survival sum with its
   mean while retaining the interval-censored event likelihood;
2. `decision_aligned_bag`: train the unit-temperature interval
   log-mean-exp against an event-local/external hard-negative bag so the
   unchanged zero-log-odds runtime decision is directly represented.

The selection rule was frozen before training: one passing arm selects itself;
two passing arms select `normalized_survival`; zero passing arms select none.
No effect-size threshold, threshold search, threshold lowering, additional
seed, raw-video training or locked-test access was permitted.

Preflight job `1205227` completed `0:0`: `125/125` tests passed and the exact
official THUMOS14 training-batch forward/backward smoke passed with
`test_access=false`, `checkpoint_updated=false` and
`strict_causal_paper_result_valid=false`. Its real-batch receipt SHA-256 is
`85d016dd986a0c0e88c65b2b4a2d07de331bb4c70fdb489b78b0066a72c25cd5`.
Job `1205225` is excluded because an externally supplied output alias violated
the fixed receipt filename before any model training; the complete preflight
was rerun rather than partially reused.

Both mechanism jobs used seed 52, one epoch, exactly `3,270` updates at
`3.34e-6`, the same official annotation, pre-extracted RGB-plus-flow features,
proposal labels and video-length cache, and no test data. The official artifact
hashes closed identically across both arms and the frozen D1.3 control:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| training annotation | `1,575,585` | `8eb3e61cc758bcc08aea1d17cfbf1acb2fed8c2a51ed884116d766e9e4c04e66` |
| proposal labels | `277,325,536` | `ba7dfb10614cfacd1b26f3d50c2ffa41ae2c7fbce62e946777c217adaeebd22f` |
| training features | `3,331,932,341` | `d4660b31b8c6c00d48b590936b3574ab650423ede9b42016fa1d9dda5d45ac9b` |
| video lengths | `7,063` | `0fcc70d555af6198e7b22850aebe56998e9184c2be3f81a16e7a56667f7b9fd8` |

### Completed jobs and terminal evidence

Mechanism jobs `1205231` (`normalized_survival`) and `1205232`
(`decision_aligned_bag`) completed `0:0`. Parameter audits `1205272` and
`1205310` confirmed non-zero changes in the independent birth head, four-state
transition head, shared transition fusion and owner decoder without executing
a new forward or optimizer step. Complete predicted-only terminal scans
`1205271` and `1205309` covered all `203,363` real prefixes, `5,917` padding
no-ops, `3,003` visible births and `200` observed current-stream EOS events.
Both scans certified non-negative starts, positive emitted length, immutable
ledger prefix, no duplicate event, contiguous sequence ids, closed
emit-to-ledger counts and zero capacity exhaustion.

The two arms had sharply different but individually decisive failures:

| Terminal quantity | normalized survival | decision-aligned bag |
|---|---:|---:|
| positive individual birth logits / `2,033,630` | `0` | `923,712` |
| maximum individual birth logit | `-0.327579` | `0.526305` |
| positive prefix maxima / `203,363` | `0` | `112,398` |
| rising-edge runtime births | `0` | `30,002` |
| post-forward admissible assignments | `0` | `3,573` |
| learned cancellations | `0` | `29,974` |
| learned ends | `0` | `0` |
| immutable emissions | `0` | `0` |
| reacquisitions | `0` | `0` |
| capacity exhaustions | `0` | `0` |

For normalized survival, the mean oracle-path interval event probability was
`0.444412`, yet the largest individual logit and the largest interval
log-mean-exp remained negative. An interval likelihood can therefore improve by
combining several sub-boundary hazards without producing a single runtime
birth. This arm did not solve the measured decision mismatch.

The decision-aligned bag did cross the fixed runtime boundary: `45.42%` of all
query-prefix logits and `55.27%` of prefix maxima were positive; `58.91%` of
the `3,003` oracle interval bags were positive. This yielded about ten
rising-edge births per visible birth and cancelled `99.91%` of those runtime
tracks. It fixed absolute birth inactivity but overcorrected calibration and
did not close the post-birth lifecycle.

Training-source counts expose the next dominant mismatch. The decision-aligned
arm created approximately `332,525` runtime births and `332,217`
cancellations during its training epoch, but only four learned ends. Its ragged
owner loss contained `305,064` false-track cancel groups versus `3,003`
positive owner assignments, a ratio of about `101.6:1`; only `1,371` target
tracks had an observed end. The current owner-state cross entropy averages over
all ragged groups, while end-risk learning is restricted to tracks with a
known target identity. After birth activation, false predicted tracks therefore
dominate cancellation learning and the predicted-only runtime supplies no
identity bridge to a learned end. This is a measured distribution/semantic
mismatch, not proof that owner-conditioned end is impossible.

Formal cross-arm job `1205337` completed `0:0`. Gate receipt SHA-256
`c5cab5a5fcb92de6fb1483b5c57981bbb0a8c04af2e33a7e1b2ec11afd7ace0f`
records `FAIL_STRUCTURE_GATE`, `selected_variant=null`,
`development_pilot_contract_eligible_for_freeze=false`,
`official_comparison_release=false`, `locked_test_release=false` and
`paper_claim_release=false`. The normalized scan, decision-aligned scan,
normalized parameter-audit and decision-aligned parameter-audit SHA-256 values
are respectively
`6489c59084be4111d374a257ad966ffa99b6d6b7aa3359f02a2a7a34c620cdf2`,
`aa2b661d1389b8ffd8426de2e9d017a1dd47a490eb93a61284d1e9ebb3903f27`,
`bd8b6fa4650a2bf1931d781f757c5a4654ddd666fc283868ddea12f96d035a81`
and `4b140ae827e037aa9801b00470b61dcdbc751fa97413f8732c9b30a29e030ec3`.

### Scientific decision and next falsification

D1.4 is complete and failed. It is not an official performance experiment:
the training-prefix detection printouts use full-video timing/offline
termination and remain invalid by construction. No reported number from D1.3
or D1.4 may enter a paper performance table.

The next work must remain analysis-only until prospectively frozen:

1. run a read-only, train-only counterfactual owner unroll on the frozen
   decision-aligned checkpoint, separately forcing only observable birth
   identity and then the full oracle identity path after model forward;
2. record owner cancel/continue/end margins, track lifetime and source at the
   first owner decision, without exposing ground truth to the main predicted
   inference path;
3. if oracle-identity tracks still cannot end, repair the per-track
   right-censored end objective by exposure normalization and balance true-owner
   versus false-track families; if oracle identity restores ends, repair
   causal assignment/reacquisition instead;
4. derive any birth calibration correction from the registered risk-set
   sampling probability, not from a searched runtime threshold or the observed
   terminal count;
5. preregister a minimal next factorial only after this diagnostic separates
   end-hazard failure from identity-transport failure.

Only a future structure pass may release short development pilots. Paper
comparability still requires a separately frozen, matched 100-epoch
native-MATR/EventMATR comparison using the official split, the same extracted
features, optimization, post-processing, terminal checkpoint and evaluator,
followed by one locked-test evaluation.

## D1.4 semantic correction and D1.5 protocol freeze

A complete source-level review narrowed two D1.4 accounting statements without
changing any historical raw value or the failed gate:

1. the D1 `event_owner_assignment_count=3,003` is incremented once per positive
   birth-risk group. It is not a count of successfully associated predicted
   owner tracks;
2. `305,064` counts batch-local all-negative owner groups grouped by
   `(video_name, runtime_event_id)`;
3. because the two counters have different units, `101.6:1` is not a gradient
   ratio, loss weight or established root-cause dose;
4. the strong evidence remains that targetless predicted records are routed to
   cancellation while end-risk supervision is restricted to target-backed
   groups. Whether this mismatch, identity drift, early-cancel truncation or
   over-admission is dominant remains unresolved.

The next diagnostic is now prospectively frozen as a train-only, read-only
two-by-two crossing of predicted/oracle-visible admission and free/oracle-
refreshed identity, with a recurrent no-cancel shadow for every cell. The exact
protocol, contamination boundary, required fields, interpretation table and
official-comparability debt are recorded in
[eventmatr-d15-owner-counterfactual-protocol-20260730.md](eventmatr-d15-owner-counterfactual-protocol-20260730.md).

D1.5 cannot retroactively pass D1.4 and cannot produce paper performance. Its
only purpose is to select at most one next structural intervention.
