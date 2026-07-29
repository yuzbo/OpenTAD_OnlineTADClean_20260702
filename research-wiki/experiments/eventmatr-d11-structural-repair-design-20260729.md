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
7. solve a deterministic one-to-one assignment;
8. mark assigned predictions `predicted_associated`;
9. leave unmatched predictions `predicted_unmatched`;
10. allow the existing teacher ratio to inject only unmatched visible targets,
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
