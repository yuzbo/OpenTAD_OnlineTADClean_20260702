# EventMATR D1: problem-truth, solvability, and novelty pre-experiments

Date: 2026-07-28

Status: active, train-only mechanism study

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
| T | R + temporal assignment, identity lock, cancellation, reacquisition | whether sticky failure is an identity/lifecycle problem | fewer owner/ID switches, duplicates and fragments; better same-class overlap and reacquisition |
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
| D1 temporal identity lifecycle (T) | B1O1 closes `6/1854` records and leaves `1407` active; same-class ownership is a known hard case | stable pre-birth assignment, post-birth lock, owner-conditioned end, cancel/reacquisition; T vs R | potentially task-specific state-machine contribution, but overlaps tracking-by-query and ActionSwitch | active, conditional |
| D1 censored lifecycle risk (H) | birth/end positives are about `0.1477%`; learned logits are substantially below the already rare constant prior | event-normalized interval-censored birth and right-censored end; H vs R and ordinary duration-free BCE later | survival/censoring is not new; contribution can only be strict-causal On-TAD coupling and demonstrated necessity | active component |
| D1 full composition (TH) | sparse risk and unaligned ownership are independent D0 failures | factorial N/R/T/H/TH, then component removals and stress strata | only currently plausible headline: one strict-causal lifecycle process coupling C1–C4 | conditionally promising |
| Ledger and causal audit | D0 exposed invalid duration/EOS use and negative starts; v1 nevertheless proved no-duplicate/capacity invariants useful | runtime assertions and adversarial tests | correctness/reproducibility protocol, not standalone novelty | mandatory substrate |
| CESR track-refine-commit | proposal lifecycle and identity are real needs | identity ledger and causal state survive as infrastructure | broad claim is covered by CAG-QIL, SimOn, OAT, MATR, ActionSwitch and related work | headline rejected; substrate only |
| PCEH | instance end uncertainty is real, but the old class-level/repeated-positive formulation was invalid | D1 H repairs instance risk/right censoring; separate commit/emission hazard is not yet implemented | hazard primitive not novel; old PCEH claim remains unproved | demoted/HOLD |
| CRS-EPS / state-transition sampling | uniform clips under-sample starts and transitions; full chronology may be expensive | only after D1 truth is established: event+uniform sampling with inclusion weights, ESS/calibration and gold full-packet comparison | efficiency/bias-control protocol, not headline | deferred enabling route |
| PETAL / persistent trajectory | D0 sticky failure and same-class identity make persistent state relevant | feature-level T/TH is the narrow falsification; raw-video PETAL is not released | significant TrackFormer/MOT transplant risk | REVISE; feature-level only |
| Raw-RGB dynamic event memory | appearance adaptation could matter only after the feature mechanism works | no current experiment by design | expensive substrate, not yet attributable | blocked by D1 gate |
| Evidence-aligned risk-controlled On-TAL | evidence-ready time is an interesting but different target | would require annotation-agreement and metric-value pilot | task-reformulation novelty is possible but outside the present standard benchmark | superseded for this route |
| Three-clock/PIVOT observability | sensor/action clocks may be physically distinct | requires synchronized physical measurements | leaves standard On-TAD | rejected for this route |
| Offline-teacher distillation | might stabilize a proven causal learner | no current experiment; would confound D1 attribution | support method only | forbidden before core D1 |
| Continual/open-world On-TAL | unknown-class drift is scientifically valuable | requires a different dataset/protocol | separate high-ambition problem | not a near-term closed-set route |

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
- T does not improve same-class owner swaps, fragmentation, cancellation and
  reacquisition;
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
