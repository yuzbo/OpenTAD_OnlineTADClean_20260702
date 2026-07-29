# EventMATR D1: problem-truth, solvability, and novelty pre-experiments

Date: 2026-07-28

Status: closed diagnostic record; superseded by the D1.1/D1.2 structural repair

The release ladder below is preserved as the prospective plan that governed
the original D1 run. Its 5/10/20-epoch actions are no longer active. Current
authority is
[eventmatr-d11-structural-repair-design-20260729.md](eventmatr-d11-structural-repair-design-20260729.md)
and DR-043.

Scope: standard closed-set, fully supervised, strict-causal Online TAD on
official MATR THUMOS14 RGB+flow features

## Scientific boundary

This study starts from the frozen D0 verdict in
[ontad-matr-official-parent-event-model-20260722.md](ontad-matr-official-parent-event-model-20260722.md):
the network has finite event/owner gradients, but birth/end positives are only
about `0.1477%`, and the unaligned sticky runtime does not close lifecycles.
D0 train-prefix mAP is not paper evidence because that system consumes
`true_duration`, complete-video timing, and offline EOS.

D1 removes those inputs from the model boundary and asks three separate
questions:

1. **Problem truth:** do sparse lifecycle risk, train/inference state mismatch,
   and unstable same-class ownership occur under audited strict causality?
2. **Solvability:** can event normalization, chronological ragged unrolling,
   censored hazards, and explicit trajectory state repair the corresponding
   failure rather than merely change a scalar loss?
3. **Innovation:** does the coupled event process contribute more than known
   survival, tracking-by-query, and scheduled-sampling primitives?

All current results remain train-only diagnostics with
`strict_causal_paper_result_valid=false`. Locked test, multiple seeds, raw RGB,
threshold search, and lowering thresholds to force output remain forbidden.

## Exact start and implementation identity

| Object | Exact identity |
|---|---|
| Canonical D0 evidence/Wiki | commit `efbe12ed133ca7f85fdcfefafe722b0895b14094`, tree `e11b65028369a506c417402e9bc11ed7e3dbe972` |
| D0 diagnostic code | commit `ca914f3ea337d1e8f0f005d394a05ec81edc0ee7`, tree `05e3cd8f11d30bdd65c5fba6c1b712e001c6a52d` |
| Source training | commit `92cf34aa07bebee2a7a7e3661431d5055804b29b`, tree `aef4f64bc020df9d39ead9811fbc01407f1c754a` |
| D1 core implementation | commit `dd394c6fd3509f707065297b18c8157e6c85a05f`, tree `51110f8746656b85cddda8f298701141a2d15325` |
| D1 registered pilot source | commit `1f4bb29ad58dddcc33f6ff2bdc57a5934ee5c53d`, tree `aad757531cfcbfb79b616756466251f946574035` |
| Pilot manifest | SHA-256 `1aa1524364a82853fe67bc04fded600274414d31eae139aec06f9d0261aa8582` |

The D1 implementation provides a duration-free/current-observation-only model
boundary, event-normalized interval-censored first birth, right-censored end
hazard, one chronological ragged train/inference event runtime, mixed
oracle/predicted tracks, temporal assignment, post-birth identity lock,
cancel/reacquisition, negative-start clamping, and audited ledger invariants.

## Completed pre-experiment evidence

### Local contract and synthetic mechanism tests

The registered pilot source passes `83` tests. The suite covers no future
metadata, observed-EOS-only behavior, negative-start clamping, positive-length
and immutable ledger output, no duplicates, no silent capacity eviction,
same-class overlap, `Q+1` births, `R>Q` active records, cross-prefix gradient
flow, and source/receipt fail-closed behavior.

The deterministic synthetic receipt establishes:

- dense-vs-event-normalized positive gradient ratio equals `Q` at
  `Q={2,10,100,1000}`; at `Q=1000`, dense positive gradient is `0.0009985`
  versus `0.998528` for event normalization;
- right-censored negative rows have positive logit gradients
  `[0.1192, 0.26894, 0.5]`, while an observed terminal event has gradient
  `-0.5`;
- temporal assignment returns the stable path `[0,0,0,0]` with zero switches;
- `12` concurrent active records with `Q=10` retain `12` unique IDs, minimum
  start `0`, one explicit reacquisition of ID `0`, and no capacity exhaustion;
- R/T/H/TH each produce nonzero transition and owner gradients through the
  ragged path.

These are mechanism proofs, not dataset-effect proofs.

### THUMOS14 identity-pressure truth audit

A read-only audit of the official train annotations (`200` videos, `3,007`
instances, `20` classes) separates two often conflated identity problems:

- `180/200` videos contain repeated instances of the same class;
- true same-class temporal overlap is rare: `5` overlapping pairs in `2`
  videos, involving `9/3,007` annotations (`0.299%`); the cases are
  `FrisbeeCatch` and `VolleyballSpiking`;
- near-neighbor repetition is common: `453` same-class pairs overlap or are
  separated by at most `16` frames, involving `677/3,007` annotations
  (`22.51%`) in `56` videos;
- at `64` frames, the corresponding count is `1,348` pairs and `1,509/3,007`
  participating annotations (`50.18%`) in `102` videos;
- any-class concurrency occurs in `23` videos and never exceeds two
  simultaneous annotated instances in this split.

Therefore THUMOS14 does support a real **nearby repeated-instance association
and reacquisition** problem, but it is a weak natural benchmark for a headline
**simultaneous same-class overlap** claim. The latter remains an adversarial
correctness stratum; T/TH must earn their value through repeated-instance owner
stability, fragmentation, duplicate rebirth, cancellation and reacquisition,
not through the existence of many same-class overlaps.

### Exact-source official-data smoke

Slurm job `1200932` completed `0:0` in `00:02:00`. Its receipt is `PASS` on one
official THUMOS14 train batch with source commit/tree/manifest exactly matching
the registered pilot source above. All N/R/T/H/TH lanes completed forward,
backward, one optimizer step, strict checkpoint reload, and finite event/owner
gradient checks. The receipt records `test_access=false`,
`checkpoint_updated=false`, and `strict_causal_paper_result_valid=false`.

The earlier core-source smoke `1200782` also completed `0:0`, but it is retained
only as implementation history; `1200932` is the operative pilot gate because
it matches the final registered pilot source exactly.

## Registered N/R/T/H/TH design

| Lane | Isolation | Problem tested | Evidence required to survive |
|---|---|---|---|
| N | official native MATR anchor | whether D1 gains exceed the parent under the same train-only budget | reproducible finite anchor; never treated as strict-causal paper evidence by itself |
| R | balanced state risk + shared chronological ragged train/inference runtime | whether lifecycle failure is an exposure/state-distribution problem | smaller train/inference lifecycle gap, real downstream false-track/cancel/rebirth supervision, stable gradient and closure |
| T | R + temporal assignment, identity lock, cancellation, reacquisition | whether sticky failure is a repeated-instance identity/lifecycle problem | fewer owner/ID switches, duplicates and fragments; better near-neighbor association and reacquisition; same-class overlap only as a small stress stratum |
| H | R + interval-censored first birth and right-censored end hazards | whether dense background averaging causes sparse under-learning | improved birth/end calibration, timing and closure versus R and ordinary duration-free BCE |
| TH | trajectory + censored-hazard composition | whether the repairs are complementary | distinct improvements from T and H and a closed strict-causal lifecycle, not merely lower training loss |

The fixed protocol is seed `52` at `5/10/20` epochs. Stage 1 array `1200955`
contains only the five 5-epoch lanes. The 10/20-epoch tasks remain unreleased
until every 5-epoch lane produces a finite, source-exact, train-only receipt
with the registered metrics and checkpoint payload.

## Complete candidate-idea review

| Candidate | Problem truth | Current solvability test | Innovation verdict | Decision |
|---|---|---|---|---|
| D1 chronological event repair (R) | D0 has gradients but v1 training does not expose the inference lifecycle to its own false births/cancels/rebirths | shared differentiable ragged runtime plus oracle/predicted track mixing; R vs N and detached/oracle-only controls later | strongest standalone D1 candidate only if errors change downstream state; otherwise scheduled-sampling/recurrent-unroll overlap dominates | active |
| D1 temporal identity lifecycle (T) | B1O1 closes `6/1854` records and leaves `1407` active; `90%` of train videos repeat a class and `50.18%` of instances have a same-class neighbor within 64 frames, but natural same-class overlap is only `0.299%` of instances | stable pre-birth assignment, post-birth lock, owner-conditioned end, cancel/reacquisition; T vs R on near-neighbor repeats, with overlap as a stress test | potentially task-specific state-machine contribution, but overlaps tracking-by-query and ActionSwitch; cannot use abundant same-class overlap as the paper premise | active, conditional |
| D1 censored lifecycle risk (H) | birth/end positives are about `0.1477%`; learned logits are substantially below the already rare constant prior | event-normalized interval-censored birth and right-censored end; H vs R and ordinary duration-free BCE later | survival/censoring is not new; contribution can only be strict-causal On-TAD coupling and demonstrated necessity | active component |
| D1 full composition (TH) | sparse risk and unaligned ownership are independent D0 failures | factorial N/R/T/H/TH, then component removals and stress strata | only currently plausible headline: one strict-causal lifecycle process coupling C1–C4 | conditionally promising |
| Ledger and causal audit | D0 exposed invalid duration/EOS use and negative starts; v1 nevertheless proved no-duplicate/capacity invariants useful | runtime assertions and adversarial tests | correctness/reproducibility protocol, not standalone novelty | mandatory substrate |
| CESR track-refine-commit | proposal lifecycle and identity are real needs | identity ledger and causal state survive as infrastructure | broad claim is covered by CAG-QIL, SimOn, OAT, MATR, ActionSwitch and related work | headline rejected; substrate only |
| PCEH | instance end uncertainty is real, but the old class-level/repeated-positive formulation was invalid | D1 H repairs instance risk/right censoring; separate commit/emission hazard is not yet implemented | hazard primitive not novel; old PCEH claim remains unproved | demoted/HOLD |
| CRS-EPS / state-transition sampling | uniform clips under-sample starts and transitions; full chronology may be expensive | only after D1 truth is established: event+uniform sampling with inclusion weights, ESS/calibration and gold full-packet comparison | efficiency/bias-control protocol, not headline | deferred enabling route |
| PETAL / persistent trajectory | D0 sticky failure and repeated-instance identity make persistent state relevant | feature-level T/TH is the narrow falsification; raw-video PETAL is not released | significant TrackFormer/MOT transplant risk | REVISE; feature-level only |
| Raw-RGB dynamic event memory | successor/extension of PETAL; appearance adaptation could matter only after the feature mechanism works | no current experiment by design | expensive substrate, not an independent candidate and not yet attributable | blocked by D1 gate |
| Causal On-TAD pretraining / feature adaptation | a causal representation might help only after the lifecycle mechanism is identifiable | current frozen RGB+flow feature protocol isolates the head; no pretraining experiment now | broad, expensive infrastructure rather than the first scientific question | long-term only |
| Evidence-aligned risk-controlled On-TAL | evidence-ready time is an interesting but different target | would require annotation-agreement and metric-value pilot | task-reformulation novelty is possible but outside the present standard benchmark | superseded for this route |
| Three-clock/PIVOT observability | sensor/action clocks may be physically distinct | requires synchronized physical measurements | leaves standard On-TAD | rejected for this route |
| Offline-teacher distillation | might stabilize a proven causal learner | no current experiment; would confound D1 attribution | support method only | forbidden before core D1 |
| Continual/open-world On-TAL | unknown-class drift is scientifically valuable | requires a different dataset/protocol | separate high-ambition problem | not a near-term closed-set route |

A fresh cross-file completeness audit found no omitted idea that should enter the
current standard-task mainline. One-shot emission remains an archived baseline;
zero-shot/open-vocabulary, anytime semantic alarms, active sensing/duty cycling,
full-packet/full-tower training and hardware-clock observability are intentionally
excluded because they are rejected baselines, cost/system choices, or different
tasks rather than missing D1 mechanism candidates.

## Novelty audit

The bounded search and independent zero-history review are stored in
`.aris/traces/novelty-check/2026-07-28_run01/`. The search found direct
component overlap with MATR and ActionSwitch in On-TAL, TrackFormer/MeMOTR and
autoregressive online tracking for identity/state propagation, CMeRT for
train/inference mismatch, and established discrete-time survival learning for
censoring.

Therefore no standalone claim is allowed for permutation matching, survival
hazards, right censoring, identity-preserving queries, autoregressive unrolling,
scheduled sampling, overlap handling, or the ledger. The defensible hypothesis
is narrower:

> In strict-causal closed-set On-TAL, stable pre-birth assignment, censored
> first-birth/end risks, explicit post-birth ownership with
> cancel/reacquisition, and event-level train/inference state alignment form one
> auditable lifecycle process.

This remains a hypothesis. It survives only if C1–C4 repair distinct measured
failures and TH improves both detection and lifecycle quality after matching
parameters and training budget.

## Kill criteria and release ladder

Immediate hard kill:

- any future frame/metadata, true duration, complete-video timing, offline EOS,
  or GT inference leakage;
- any non-finite loss/gradient, negative or non-positive interval, duplicate or
  mutable emission, silent capacity eviction, or source mismatch.

Scientific kill or demotion:

- R does not reduce error-state exposure/train-inference lifecycle gap;
- T does not improve near-neighbor repeated-instance owner swaps,
  fragmentation, duplicate rebirth, cancellation and reacquisition;
- H is matched by ordinary duration-free BCE on calibration, timing and closure;
- TH gains disappear after parameter/budget controls or do not exceed its
  strongest component;
- the only visible change is train-prefix mAP or scalar loss.

Release order:

1. finish and inspect all five 5-epoch receipts from array `1200955`;
2. release the registered 10/20-epoch seed-52 lanes only if stage 1 is finite
   and scientifically interpretable;
3. add component-removal, same-class overlap, near-boundary, truncation and
   oracle/predicted/detached-state stress tests;
4. only after the D1 mechanism gate may locked-test evaluation and multiple
   seeds be proposed; raw RGB, teacher distillation and threshold search remain
   separate later decisions.

## Five-epoch gate completed — 2026-07-29

All five tasks in array `1200955` completed `0:0` and wrote `PASS` pilot and
final source-identity receipts. Each run has exactly five finite metric rows, a
loadable terminal checkpoint, exact source
`1f4bb29ad58dddcc33f6ff2bdc57a5934ee5c53d` / tree
`aad757531cfcbfb79b616756466251f946574035`, `test_access=false`, and
`strict_causal_paper_result_valid=false`.

| Lane | Train-prefix average mAP (%) | mAP@0.3 | mAP@0.5 | mAP@0.7 | Terminal proposals | Videos with proposals |
|---|---:|---:|---:|---:|---:|---:|
| N | 7.0957 | 12.1895 | 7.3023 | 2.0648 | 21,256 | 176 / 200 |
| R | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 / 200 |
| T | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 / 200 |
| H | 4.5066 | 5.7738 | 4.4795 | 2.9937 | 333 | 47 / 200 |
| TH | 3.9445 | 4.4919 | 4.0751 | 3.1464 | 271 | 42 / 200 |

These values are diagnostic train-prefix results only. R and T produced some
text output in epochs 1–2, then zero-byte proposal files in epochs 3–5. Their
logged `nan` average-time values are consequences of an empty prediction set,
not non-finite model loss or gradient. H and TH retain only `1.57%` and `1.27%`
of N's terminal proposal count and cover only `23.5%` and `21.0%` of train
videos.

Optimization itself is not dead:

- N total loss falls `2.2854 -> 0.5914`;
- R birth risk falls `0.5155 -> 0.3889`, owner-class
  `1.9097 -> 0.1661`, and owner-state `1.1470 -> 0.4502`;
- T identity loss falls `0.1132 -> 0.0278`;
- H birth/end risk changes `4.5401 -> 1.7105` /
  `0.8788 -> 0.8482`;
- TH birth/end/identity changes `4.6506 -> 1.7776` /
  `2.1849 -> 1.1343` / `0.0524 -> 0.0197`.

The data therefore separates optimization success from detector success.
R/T learn their supervised terms but collapse terminal emission. H/TH avoid
complete collapse and improve the strictest `0.7` overlap diagnostic over N,
but average mAP and coverage remain substantially worse because emission is
overly sparse. Falling cancel-group and ragged-track counts cannot yet be called
better lifecycle behavior: they may simply reflect record suppression.

### Threshold and gate audit

The fixed `flag=0.5`, class-score `0.1`, and suppression-overlap `0.3` values are
the official native-MATR controls. The class-score threshold applies only to
the native writer. The flag threshold controls the native segment-memory flag
at inference, not EventMATR birth/end. EventMATR writes completed immutable
ledger rows without a class-score cutoff; the common online postprocessor still
uses suppression overlap `0.3`.

Formal D1 did **not** set numeric birth/end logit thresholds. Both options remain
`None`: birth is a rising transition into the argmax START state, owner
background cancels a record, and end is the argmax END state subject to one
positive frame of minimum duration. Therefore R/T's zero-byte terminal writer
output is upstream lifecycle starvation, not a consequence of lowering or
raising the native class-score threshold. No threshold was searched or changed.

The audit also corrects the earlier gate wording:

- the preregistered **execution/integrity gate** is precise and passes: causal
  contracts, exact source, finite metrics/gradients, complete epoch sequence,
  loadable checkpoint, absent locked test, and no ledger hard violation;
- the manifest lists detection, lifecycle, identity, calibration, and systems
  metrics, but the five-epoch finalizer enforces only core losses/counts and
  train-prefix mAP; it does not operationalize most listed scientific metrics;
- the preregistration contains qualitative phrases such as “smaller gap”,
  “fewer switches,” and “scientifically interpretable,” but no prospective
  effect-size, coverage, confidence, or sample-size cutoffs;
- zero R/T output and severe H/TH coverage loss are observations, not
  preregistered numeric failure thresholds.

Consequently it is too strong to say that a pre-existing quantitative
“scientific gate failed.” The accurate verdict is: the five-epoch execution
gate passes, while scientific sufficiency is **not yet adjudicable** because
required lifecycle evidence is absent and an unexpected emission pathology
appeared. Holding 10/20 epochs is a reversible diagnostic/resource decision,
not a retroactive claim that the method is disproved.

Before any replay, register a prospective no-threshold-search gate:

1. retain the exact argmax lifecycle decisions and fixed native controls;
2. require finite, zero-violation stage counts for candidate, birth,
   assignment, cancel, end, ledger and emission, with zero terminal emission
   treated only as a functional-liveness hold;
3. require every metric already named in the manifest to be produced, including
   recall/coverage, calibration, latency, closure, unresolved tracks,
   duplicates, reacquisition, owner switches, fragmentation, and nearby-repeat
   strata;
4. derive and freeze effect-size/sample-size tolerances before observing replay
   results; do not invent coverage cutoffs from the current `333/271/0` counts;
5. compare H to R at matched coverage for calibration/closure, T to R for
   identity behavior at a frozen recall tolerance, and TH to the stronger of T
   and H for both detection and lifecycle quality;
6. permit threshold-free score distributions and precision-recall curves for
   diagnosis, but do not choose a new operating threshold from them.

Do not release 10/20 epochs yet. First run a strict-causal train-only
chronological checkpoint replay and instrument the full path from candidate
logits through birth, assignment, cancellation, end, gating and final emission.
Threshold lowering/search remains forbidden; the purpose is to locate the
collapse before changing the model.

## Strict-causal checkpoint replay completed — 2026-07-29

The registered read-only replay source is commit
`6a23ab3a3711bc1ecb5a2fe442e302964ee3afb8`, tree
`8564dca45850675ce9ac366ee051aacbc4d97dc1`, manifest SHA-256
`b8980f21b34f3a862d8306c3a48249085351e4bd166f08d561016d29d3be58ed`.
Related Slurm tests `1203223` passed `54/54` in `37.75s`. Replay array
`1203224` completed all four tasks `0:0`; its combined finalizer receipt is
`PASS`. Every lane replayed `203,363` real chronological train prefixes and
observed exactly `200` current-time EOS markers. Source identity was unchanged
before/after, all checkpoint SHA-256 values were unchanged, locked-test access
was false, threshold search was false, runtime capacity exhaustion was zero,
and every emitted row had a unique immutable ID and positive length.

The replay ran in evaluation mode without targets at the model boundary.
`segment_flag`, which is derived from ground truth, was additionally removed
from the evaluation payload even though the inherited learned-flag branch did
not consume it. Ground truth was used only after each forward pass.

| Lane | Birth transitions | Cancels | Ends / emits | Reacquisitions | Active at observed EOS | Closure per birth | Videos with proposals |
|---|---:|---:|---:|---:|---:|---:|---:|
| R | 46,935 | 46,892 | 0 / 0 | 0 | 43 | 0.000% | 0 / 200 |
| T | 5,670 | 5,667 | 0 / 0 | 4,017 | 3 | 0.000% | 0 / 200 |
| H | 2,648 | 2,599 | 46 / 46 | 0 | 3 | 1.737% | 6 / 200 |
| TH | 5,621 | 5,072 | 519 / 519 | 3,611 | 30 | 9.233% | 26 / 200 |

The stage accounting is exact: each lane satisfies births = cancellations +
emissions + active records at observed EOS, with reacquisition counted as a
birth transition but not a new immutable record. R/T have many learned START
transitions, so candidate-birth starvation is not the explanation for zero
output. Instead:

- R owner argmax counts are `[46,892 BG, 0 START, 17,606 ALIVE, 0 END]`;
- T owner argmax counts are `[5,667 BG, 0 START, 2,378 ALIVE, 0 END]`;
- their maximum owner-END winning margins are still strictly negative
  (`-1.263/-1.486`), so no owner can enter END;
- `99.908%/99.947%` of birth transitions are cancelled;
- H/TH produce `46/519` END decisions and exactly `46/519` ledger emissions,
  proving that END-to-writer is live when END occurs;
- the inherited native memory gate retains similar ground-truth-positive recall
  in all lanes (about `86.6–88.6%`), so it is not the first-stage explanation
  for the factor-specific collapse.

The censored-hazard factor therefore changes the failed state transition, but
does not yet solve detection:

| Lane | Proposals | tIoU 0.3 matches | Precision | Recall | Birth match precision / recall within causal delay `[0,64]` |
|---|---:|---:|---:|---:|---:|
| R | 0 | 0 | n/a | 0.000% | 2.633% / 41.104% |
| T | 0 | 0 | n/a | 0.000% | 8.272% / 15.597% |
| H | 46 | 5 | 10.870% | 0.166% | 0.906% / 0.798% |
| TH | 519 | 30 | 5.780% | 0.998% | 1.583% / 2.960% |

Only `2/6` H/TH ground-truth end prefixes had any owner END argmax,
respectively; this is an upper bound because it does not establish that the
ending owner is the correct event. H/TH candidate Brier/ECE values look much
better than R/T, but their event recall collapses, showing that
background-dominated proper scores cannot substitute for conditional event
recall and timing. TH also has `198` excess overlapping predictions across
`23` fragmented ground-truth events. These values are diagnostic, not a paper
result or a pass threshold.

### Structural cause beyond the stage counts

The replay localizes the immediate runtime failure. Code audit identifies two
upstream mechanisms that make unchanged longer training scientifically
unjustified:

1. During training, a predicted START inherits a semantic `target_event_id`
   only when its query index exactly equals the oracle path's current query.
   Other predicted records retain `target_event_id=None`; their ragged owner
   target is BACKGROUND with no class target, and they are excluded from the
   true-event end hazard. At inference every record is predicted-only. Thus the
   intended oracle/predicted mixture does not causally associate all correct
   predicted tracks to prefix-visible events; it supplies a strong
   predicted-track/background bias. The epoch-five false-track/cancel group
   means (`10.712/1.627/0.889/0.242`) and replay cancellation rates are
   consistent with this mechanism.
2. `temporal_history` and the censored-birth risk indices are rebuilt inside
   each 64-prefix forward batch. Videos span many such batches, so the
   nominal 64-prefix pre-birth assignment/risk window is truncated at every
   batch boundary. This is especially relevant to the poor H/TH birth timing
   and must be repaired or isolated by a counterfactual before attributing the
   result to the hazard formulation itself.

An independent zero-history code/science reviewer agreed that cancellation and
END starvation, not candidate absence or the writer, are the immediate causes;
it independently identified the predicted/oracle target mismatch and
batch-local history. Its suggestion that replay proposals themselves were
oracle proposals was rejected: the replay passes no targets and all replay
births are predicted-only. The correct limitation is that the checkpoint was
trained under mixed oracle/predicted exposure.

### Scope and next decision

This is now sufficiently deep evidence for an external Pro discussion, but it
does not complete the research program. The problem is real and partly
intervenable: adding censored owner risk changes END from `0` to `46/519`, and
the full composition raises closure over H. It is not solved: cancellation
remains `90.2–98.1%`, detection recall remains at most `0.998%`, and
cancel/reacquisition is mostly churn rather than stable identity.

Do not release the existing 10/20-epoch jobs. First implement and test a minimal
D1.1 repair:

1. causally and one-to-one associate predicted training births to
   prefix-visible events rather than requiring exact oracle-query equality;
2. report oracle, predicted, merged, false and reacquired track supervision
   separately, and event-normalize owner/cancel/end risk so false groups cannot
   silently dominate;
3. give cancellation an explicit supervised meaning distinct from generic
   owner background, while retaining identity lock and explicit
   reacquisition;
4. carry a detached causal pre-birth history/risk window across batch
   boundaries;
5. isolate the inherited ground-truth-vs-learned native memory gate with a
   fixed counterfactual, without changing its registered `0.5` operating point;
6. define nearby-repeat strata in one declared coordinate system; the replay's
   `64` feature-prefix units must not be conflated with the earlier annotation
   audit's `64` source-frame units;
7. rerun synthetic contracts, one-batch and one-epoch mechanism checks before
   registering a new five-epoch seed-52 train-only pilot.

Locked test, multiple seeds, raw RGB, threshold lowering/search, and promotion
of train-only diagnostics remain blocked.
