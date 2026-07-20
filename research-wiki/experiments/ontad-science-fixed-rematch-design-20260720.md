# Clean Scientific FIXED/REMATCH Design

Date: 2026-07-20
Status: method boundary frozen; current implementation requires scientific-contract repair

## Objective

Build a standard, fully supervised, strictly causal Online Temporal Action
Detection route. At inference, the model sees only the current and past video
prefix, maintains action-instance start/ongoing/end state, and emits a final
`{start, end, class, score}` interval with low delay. A final interval is never
revised.

The immediate experiment uses fixed cached causal video features. Raw-RGB joint
training is blocked until the feature-level FIXED/REMATCH gate passes.

## Clean Source Boundary

The scientific branch starts from commit
`b974f3d5f5d6ade15a383fa5c64138069977ba16`.

Only these responsibilities may be ported from the historical implementation:

- prefix-observable persistent-instance supervision;
- the persistent trajectory detector and the minimum required query head;
- chronological cached-feature loading;
- standard On-TAD and instance-error metrics;
- matched FIXED/REMATCH configs and focused tests.

The historical branch is not merged as a unit. Unrelated infrastructure,
historical launch machinery, and non-method dependencies are excluded.

## Root Cause to Repair

The old controller allowed predicted runtime occupancy to restrict the slots
available to ground-truth birth supervision. False predicted ACTIVE states and
post-completion refractory occupancy could therefore remove supervision for a
real first-crossing birth.

The annotation census found a minimum oracle-free capacity of two, while the
model used four slots. The observed 2,206 exhausted births were therefore not
caused by true action concurrency. They were caused by the lifecycle controller.
Suppressing almost all births produced a degenerate zero-exhaustion result and
is not a valid repair.

## State Separation

Training keeps two independent states:

1. `supervision_state` is derived only from the current and past observable
   annotation prefix. It owns canonical instance-to-slot bindings and never
   reads predicted runtime occupancy.
2. `runtime_state` contains only model-visible query, memory, lifecycle, and
   emitted-interval state. It contains no ground-truth identity.

At inference only `runtime_state` exists.

Ground-truth births are assigned from the canonical supervision pool. If this
pool is exhausted with four slots, the run fails as a structural data/config
error. Predicted false occupancy cannot delete a target.

## Shared Runtime Lifecycle

FIXED and REMATCH share the same runtime controller:

```text
FREE -> CANDIDATE -> ACTIVE -> COMMIT -> FREE
                   \-------> FREE
```

- `FREE`: eligible for a new birth.
- `CANDIDATE`: a predicted birth waiting for one causal step of support.
- `ACTIVE`: a confirmed ongoing action instance.
- `COMMIT`: append one final interval and release the slot immediately.

At each decision step the controller performs:

1. freeze the slots that were free at step entry as the legal birth pool;
2. finish and release existing actions;
3. cancel unsupported one-step candidates;
4. rank new birth proposals over the entry-free pool;
5. admit at most two new candidates, matching the frozen annotation census;
6. update remaining active instances;
7. append final intervals.

A slot released by an end or candidate cancellation becomes birth-eligible at
the next decision step. This avoids asking one slot output to describe an old
instance end and a different new instance birth simultaneously. Four runtime
slots remain above the frozen two-slot annotation requirement, so this
one-decision reuse delay is not a true-capacity workaround.

A candidate becomes ACTIVE when the next causal step supports `alive`; a
candidate with an immediate `end` must commit as a short action. Otherwise it
is released. There is no capacity-holding refractory state. The readiness
review found that commit `95fa963` does not yet implement the immediate-end
part: newborns are admitted after the current step's end processing. This is a
blocking implementation defect, not a change to the intended lifecycle.

An ACTIVE slot is released without emission when both alive and end evidence
fall below their frozen thresholds. It commits exactly once when end evidence
crosses the threshold.

The initial shared thresholds remain `0.5` for birth, alive, and end. Thresholds
may be calibrated on the validation split before formal reporting, then must be
frozen for both arms.

## Single Changed Axis

Both arms use the same first-crossing births and canonical supervision
lifecycle:

- `FIXED`: the first-crossing birth slot remains the loss target through the
  observed end.
- `REMATCH`: active instances may be rematched over the same canonically
  occupied slot pool at each prefix.

REMATCH changes loss binding only. It cannot change birth opportunity, runtime
lifecycle, capacity, inference, output thresholds, or evaluation.

## Data and Causality Contract

- Feature experiment input: fixed cached causal features, stride 8, dimension
  768.
- The feature-level binding experiment trains in FP32. Actual N16R4 smoke
  execution found non-finite first-step gradients under the inherited FP16 AMP
  route; this small feature-only head does not need AMP to meet its budget.
- Training order: chronological within each video.
- Training annotations may construct prefix-observable targets.
- Validation and test inference receive no annotations, future endpoint,
  end-of-video signal, total duration, or cached predictions.
- `inference.load_from_raw_predictions` remains `False`.
- There is no offline NMS or full-video interval correction.

## Metrics

Primary metrics:

- average mAP over tIoU 0.3:0.7 and mAP at each threshold;
- precision, recall, and F1;
- GT-end and commit delay.

Instance metrics:

- duplicate rate: extra predictions assigned to a GT instance after its best
  matching prediction, divided by the GT count;
- fragmentation rate: extra temporally disjoint prediction fragments covering
  a GT instance, divided by the GT count;
- repeated same-class and overlapping same-class recall;
- committed prediction count and prediction/GT ratio;
- runtime capacity exhaustion and dropped-supervision counters.

The combined identity error is:

`E_id = 0.5 * (duplicate_rate + fragmentation_rate)`.

The relative FIXED improvement is:

`R_id = (E_id_REMATCH - E_id_FIXED) / E_id_REMATCH`.

If `E_id_REMATCH` is zero, the claimed 20% reduction is not established.

## Gates

### Technical gate

Every arm and every seed (`705`, `706`, `707`) must have:

- zero causal-protocol violations;
- zero dropped GT birth targets;
- zero unexplained runtime capacity exhaustions;
- prediction/GT count ratio in `[0.25, 4.0]`;
- recall at tIoU 0.3 of at least `0.25`;
- a nonzero committed-prediction count.

These conditions reject silent or nearly silent controllers.

### Scientific gate

Across the three paired seeds:

- `R_id >= 0.20`;
- FIXED improves `E_id` in at least two of three seeds;
- `mAP_FIXED - mAP_REMATCH >= -0.5` percentage points.

Passing the technical gate alone proves only that the lifecycle is executable.
Raw-RGB work is authorized only when both gates pass.

## Verification

Focused tests must cover:

- supervision/runtime state isolation;
- no target loss under arbitrary predicted occupancy;
- FIXED binding persistence and legal REMATCH;
- same-bin end/birth handling with bounded next-step slot reuse;
- bounded candidate lifetime and immediate slot reuse;
- short actions, adjacent actions, repeated same-class actions, and overlapping
  same-class actions;
- one final emission per instance hypothesis;
- no future or annotation fields in inference;
- matched configs differing only in binding mode;
- non-degenerate result-gate calculations.

The feature experiment proceeds through CPU tests, synthetic streams, a Slurm
end-to-end smoke, one-seed screening, then the matched three-seed run. The
end-to-end smoke passed on N16R4 for commit `097bc72` in job `1176737`; this
proves execution and protocol compatibility only. A failed scientific gate
stops this route. A pass unlocks a separate raw-RGB causal encoder and
PEFT/joint-training stage without changing the instance lifecycle.

## Pre-Training Cost Gate Outcome

The strict deterministic paired profiler ran on N16R4 in job `1176983` at
commit `95fa963`. Both 250-step training profiles had zero dropped GT births,
zero runtime capacity exhaustions, real head updates, and about 141 MiB peak
GPU memory. FIXED and REMATCH untrained inference produced the exact same
10,029-row causal ledger digest with zero no-future violations.

The registered 12-epoch pair nevertheless estimates to 12.572 GPU-hours after
the frozen 1.25 safety factor, compared with the 2 GPU-hour one-seed cap.
Therefore the budget gate is false and the seed-705 screen was not submitted.
This is a resource/protocol rejection, not an effectiveness result about the
FIXED hypothesis. The full record is
`ontad-science-fixed-rematch-profile-20260720.md`.

Any next run requires a separately registered budget revision. The project
must not silently reduce epochs, shrink the fit split, raise the cap, or inspect
the locked reporting results to justify that revision. Raw-RGB work remains
blocked.

## Scientific Readiness Review Outcome

The external readiness review pinned to `27a59de` was archived byte-identically
and independently rechecked against `95fa963`, smoke job `1176737`, and profile
job `1176983`.

The review's principal `REVISE BEFORE SCIENTIFIC RUN` verdict is accepted.
Smoke and profiling closed several engineering unknowns, but the following
scientific contracts remain open:

- binary endpoint currently uses an unsupervised offset;
- REMATCH can move a newborn away from its birth slot on the birth step;
- same-step birth+end and final-token short actions can be lost;
- generic training reads the locked reporting split every epoch;
- instance matching, fragmentation, coordinate, mAP-unit, and result-provenance
  contracts do not yet match this frozen design;
- formal readiness and capacity counters are not fully fail-closed.

The full source is
`../../PRO_ONTAD_FIXED_REMATCH_SCIENCE_READINESS_REVIEW_20260720.md`; the
accept/qualify/supersede matrix is
`../../PRO_ONTAD_FIXED_REMATCH_SCIENCE_READINESS_ABSORPTION_20260720.md`.
These repairs precede any budget revision or seed-705 submission.
