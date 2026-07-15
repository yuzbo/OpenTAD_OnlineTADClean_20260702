# Full PETAL CRS-EPS/Q2 Round-2 Independent Absorption

Date: 2026-07-15

## Source Identity

| Field | Value |
|---|---|
| Original attachment | `d7a3bf31-eaee-4d75-a994-33a6be466079/pasted-text.txt` |
| Byte-identical archive | `PRO_FULL_PETAL_CRS_EPS_Q2_ROUND2_REVIEW_20260715.md` |
| Bytes | `98,003` |
| Lines | `2,947` |
| SHA-256 | `C7FBBC573D18DA7234CE787167AD1F07438101C3DC07D90899B09BE71CA7CEB3` |
| Review-observed HEAD | `3a297344df73d994c54cac082f34a66e597e9d44` |
| Immutable scientific anchor | `f4ea53e62b8bcf3e294d6ff047880a548bfb8dcb` |

The source is archived as evidence. This document is an independent technical
decision, not a summary that automatically adopts every recommendation.

## Independent Verdict

```text
ROUND2_DISPOSITION=ACCEPT_WITH_EXPLICIT_AMENDMENTS_AND_ONE_FACTUAL_CORRECTION
RESEARCH_ROUTE=GO_HYBRID_PROTOCOL_FOR_IMPLEMENTATION_AND_FALSIFICATION_ONLY
PROFILE=BLOCKED
FORMAL_TRAINING=BLOCKED
RAW_VIDEO_STAGE2=BLOCKED
FULL_PETAL_PACKAGE_NOVELTY=RECONSTRUCTION_RISK
FIXED_BINDING_DELTA=UNPROVEN
EXACT_NEXT_STEP=P0_LAUNCH_IDENTITY_THEN_CPU_ONLY_CRS_MANIFEST
```

I agree with the central verdict and most of the scientific controls. I do
**not** accept every implementation prescription or every kill rule literally.
The accepted object is a falsifiable protocol, not a positive method result.

## What Round 2 Correctly Freezes

1. The task remains standard fully supervised completion-triggered On-TAD.
   Internal hypotheses may evolve, but a public completed detection is
   immutable and may not use future observations.
2. The current route is frozen cached-token temporal training with complete
   chronological video enumeration, 64-token truncated BPTT, detached
   cross-chunk numerical state, and one optimizer event per video. It is not
   raw-video end-to-end training and it is not CRS-EPS.
3. The full chronological route is an exhaustive reference for its exact
   current objective. It is not automatically a practical primary trainer.
4. CRS-EPS is an enabling cost surrogate with low standalone novelty. It must
   not be the paper headline.
5. The recommended route is hybrid: CRS-EPS for candidate training, a small
   preregistered video-start full-stream subset for state/loss/gradient gold
   audits, and complete chronological one-token evaluation for final quality.
6. Q2 is only partially identifiable now. It needs a local direct-effect audit
   and a separate longitudinal policy-effect analysis.
7. Full PETAL remains vulnerable to a multi-paper reconstruction. The only
   potentially surviving method delta is fixed post-birth supervision, and it
   has no effectiveness evidence yet.
8. `P0-LAUNCH-WORKDIR` still blocks every GPU profile and formal run.

## Accepted Scientific Corrections

### Dynamic Replay Is a Surrogate

Extending replay to every supervised instance's earliest observable birth can
reconstruct the relevant canonical GT lifecycle, but it cannot recreate all
model-generated history. Earlier false births, occupied slots, completed
events, refractory state, queries, and memory can change the entry state.

```text
dynamic earliest-birth replay = causal state surrogate
video-start replay = gold reference
exact sampling weights != correction for latent-state bias
```

This distinction is essential and is accepted without reservation.

### Direct and Longitudinal Q2 Effects Must Be Separated

At a shared checkpoint and state, fixed/rematch can be a local binding-only
intervention. After either arm updates, weights and model-generated lifecycle
states may diverge. Those downstream differences are possible mediators of the
treatment, not automatically confounders. The study must therefore report:

- a local twin-forward/twin-gradient direct-effect trace;
- a longitudinal policy comparison with occupancy, birth, retirement,
  emission, recall, delay, duplicate, and fragmentation paths.

### Probability and Weighting Contract

For video-uniform, decision-time-uniform target risk,

```text
L = (1/V) sum_v (1/T_v) sum_t loss(v,t).
```

With one-draw coverage probability `rho(v,t)` and a 0.40 uniform anchor
component,

```text
w(v,t) = (1/T_v) / rho(v,t)
rho(v,t) >= 0.40/T_v
w(v,t) <= 2.5.
```

The `2.5` bound is not an arbitrary clipping threshold. It is a derived
implementation invariant. An observed raw weight above it indicates an error
in proposal normalization, edge handling, or coverage computation. The
primary estimator should preserve repeated exposures and use uncapped
Hansen-Hurwitz/IPW weighting; union inclusion probabilities apply only to an
explicit deduplicated Horvitz-Thompson variant. SNIPW is diagnostic unless a
new pre-result protocol version is signed.

### Gradient-Boundary Contract

The sampled route must reproduce original global 64-token detach boundaries.
A generic `192 no-grad -> detach -> 8 grad` route changes the gradient operator.
For supervised start bin `u`, replay to `g(u)-1` without a retained graph,
enable gradients from global chunk start `g(u)`, detach again at each original
boundary, and place loss only on the declared suffix.

This still does not make sampled training identical to full training. It makes
the gradient support comparable conditional on state fidelity.

### Cost and Clock Contracts

Optimizer events are not a valid cross-protocol work unit. The profile must
report actual temporal forward/backward tokens, replay tokens, supervised
exposures, ESS, visual frames, GPU-hours, wall time, data wait, CPU unroll, and
memory. Final efficiency requires cost-to-quality on complete chronological
evaluation.

Cached source time is not wall-clock latency. Formal evaluation needs separate
source, feature-availability, decision-start, decision-completion, and
publication clocks, with one token per primary serving call.

## Independent Amendments

### A1. Require Matched Stochasticity, Not One Mandatory RNG Architecture

The scientific requirement is that the local fixed/rematch comparison sees
identical stochastic realizations. `token-addressed RNG` is one strong design,
but making every stochastic module stateless can alter training semantics and
create a large invasive change before its necessity is demonstrated.

Accepted order:

1. close the local audit with full CPU/CUDA RNG snapshot and restoration, an
   audit-only deterministic path, or equivalent generator control;
2. verify whether skipped replay tokens make this insufficient for the G0
   full-versus-episode comparison;
3. implement token-addressed RNG only if the simpler audit cannot establish
   identical stochasticity without changing the scientific route.

Failure to match stochasticity still blocks a pure one-factor claim. The
implementation mechanism itself is not preordained.

### A2. Add an Exact Identity-Metric Contract to P1

The review lists duplicate and fragmentation endpoints but does not fully
freeze their evaluator definitions. Before effectiveness results, the project
must define and hash:

- prediction-to-GT matching order and tIoU policy;
- score/calibration policy and parity point;
- duplicate FP per GT and duplicate fraction;
- fragmentation attribution;
- same-class repeated and same-class concurrent subsets;
- how misses, unmatched emissions, and censored delay enter each statistic.

Without this contract, the proposed identity mechanism remains vulnerable to
post-result metric selection.

### A3. Slot Exhaustion Is Usually a Scientific Failure, Not a Rerunnable Invalid Run

Protocol corruption, manifest mismatch, non-finite arithmetic, and
infrastructure failure can invalidate a paired run under a preregistered retry
rule. Capacity exhaustion caused by the model or chosen slot budget is an
observed failure mode. It must trigger the route's kill/safety rule and remain
in the evidence ledger; it must not be erased by rerunning a new seed.

### A4. A Positive LoRA-by-Binding Interaction Is Not Required

The proposed Stage-2 2x2 design is correct, but `LoRA x fixed interaction = 0`
is not by itself a reason to permanently reject visual adaptation. Two stable
additive main effects can form a cleaner result:

- LoRA improves raw-video quality or efficiency;
- fixed binding retains an identity-linked benefit under LoRA;
- the interaction need not be positive.

Stage 2 should be killed if visual adaptation erases the fixed effect, has no
independent value, violates causality/cost, or needs stale caches. Zero
interaction alone is not a scientific falsifier.

### A5. One Seed Is a Resource Gate, Not a Scientific Null Result

One paired seed may stop spending under the registered resource policy. It
cannot establish that fixed binding has no effect in the population. The wiki
and paper must label such a stop as a resource kill, not hypothesis rejection.

### A6. Same Optimizer Event Semantics Do Not Imply the Same Training Trajectory

One sampled video group per optimizer step correctly preserves video weighting,
scheduler count, and update frequency. HH/IPW can make the sampled gradient
unbiased only under the stated state/gradient conditions. AdamW and the model
remain nonlinear, so increased gradient variance changes the optimization
trajectory. Quality non-inferiority and gradient-variance diagnostics are
therefore necessary; no optimizer-equivalence claim is allowed.

### A7. Report the Full THUMOS tIoU Vector as Well as Its Average

The average of mAP at tIoU 0.3, 0.4, 0.5, 0.6, and 0.7 may be the primary scalar,
but all five standard thresholds must be reported. A single average can hide a
boundary-quality regression.

### A8. FineAction Is Conditional Evidence for Identity

Primary sources verify FineAction's scale: 16,732 videos, 103,324 instances,
106 classes, and 11.5% overlapping multi-label segments. The paper emphasizes
co-occurring actions of different classes. These facts do not automatically
guarantee enough same-class concurrent identities for the proposed mechanism.
FineAction is a preferred dense confirmatory TAL dataset only after an exact
annotation and subset audit. MultiTHUMOS remains a low-cost dense multi-label
screen, not an automatic instance-identity benchmark.

### A9. Video-Group Episodes Share an Update, Not Runtime State

All `M` sampled episodes in a video group must evaluate the same parameter and
mutable-buffer snapshot. Each episode independently resets and causally replays
its declared range; runtime slots, supervision state, hidden state, and graph
objects may not leak from one draw to the next. Only gradients are accumulated.
Any mutable model buffer must be frozen, cloned/restored, or included in the
transaction and trace. A failed draw rolls back gradients and buffers for the
whole group. Otherwise episode order changes the estimator even before the
optimizer step.

### A10. Reject the Review's `OAT_PRIMARY_ARTIFACT=UNRESOLVED` Finding

Independent exact-title search resolves OAT to the ECCV 2022 paper **A Sliding
Window Scheme for Online Temporal Action Localization** by Young Hwi Kim,
Hyolim Kang, and Seon Joo Kim, DOI
`10.1007/978-3-031-19830-4_37`, with an official ECVA paper PDF. It proposes
Online Anchor Transformer (OAT), Online Suppression Network (OSN),
anchor-based pre-completion proposals, boundary offset/length refinement, and
Average Early Detected Time.

Therefore the source review's unresolved-artifact statement is factually
incorrect and is not adopted. The corrected evidence strengthens rather than
weakens the reconstruction attack: early proposals, sliding-window anchors,
boundary refinement, and online repetitive-proposal suppression are verified
prior art.

## Primary-Source Verification

The review's important competition claims are directionally supported:

- [OnPoint](https://arxiv.org/abs/2607.00289) introduces point-supervised
  online TAL using offline-to-online multi-level distillation; it changes the
  supervision contract and is adjacent rather than a direct Q2 substitute.
- [OZ-TAL](https://arxiv.org/abs/2605.09976) introduces online zero-shot TAL
  with a training-free VLM route; it blocks a zero-shot/VLM headline, not the
  fully supervised binding study.
- [OpenHOUSE](https://arxiv.org/abs/2509.12145) combines online TAL with
  hierarchical free-form streaming descriptions; it is task-adjacent.
- [ActionSwitch](https://arxiv.org/abs/2407.12987) directly addresses
  simultaneous and same-class actions in On-TAL and is a mandatory mechanism
  competitor.
- [MATR](https://arxiv.org/abs/2408.02957) directly occupies memory-assisted
  long-history On-TAL.
- [ETAD](https://arxiv.org/abs/2205.07134) already studies selective snippet
  gradients and proposal sampling for end-to-end TAD, so efficient sampling is
  not a headline novelty.
- [OAT](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136940640.pdf)
  is a verified ECCV 2022 On-TAL paper with anchor-based early proposals,
  boundary refinement, and online suppression; the source review was wrong to
  mark its primary artifact unresolved.
- [MultiTHUMOS](https://arxiv.org/abs/1507.05738) is a dense frame-level
  multi-label action dataset.
- [FineAction](https://arxiv.org/abs/2105.11107) is a dense instance-level TAL
  dataset with overlapping multi-label segments, subject to the identity audit
  above.

The literature supports the package-level reconstruction warning. It does not
prove the marginal fixed-binding hypothesis false; that remains experimental.

## Frozen Implementation Order

### Wave 0: Launch Identity Only

1. Freeze `RUN_ID`, `RUN_DIR`, `WORK_DIR`, and overrides before ticket
   publication.
2. Make Slurm argv consume only ticket-bound values.
3. Add a deterministic fake-`sbatch` shell integration test.
4. Keep profile and training blocked.

### Wave 1: CPU-Only Sampling Mathematics

1. Build immutable epoch/video/draw manifests.
2. Implement fixed `H=8` windows and the frozen mixture.
3. Compute and exhaustively test `q`, `rho`, union `pi`, raw weights,
   multiplicity, coverage, and ESS.
4. Fail closed on zero support, hash mismatch, or `w > 2.5 + tolerance`.

### Wave 2: Audit Contracts

1. Freeze the exact identity evaluator.
2. Implement the least invasive matched-stochasticity mechanism that closes
   the local direct-effect trace.
3. Add numerator/denominator and lifecycle traces.
4. Add episode-state and mutable-buffer isolation tests.
5. Add dynamic replay and video-start/reset/fixed-192 audit modes.

### Wave 3: Gold Fidelity and Profile

1. Run only the preregistered tiny G0 subset after a new clean commit, complete
   B0, and independent authorization.
2. Choose `M` and fidelity/non-inferiority margins without viewing Q2
   effectiveness results.
3. Replace the old optimizer-event profile with the multi-denominator schema.

### Wave 4: Conditional Q2

Only after all previous gates pass may one paired resource-kill seed be run.
Three seeds are a low-cost confirmation; five or more paired seeds plus a
qualified second dataset are required for a retained claim. Raw-video Stage 2
requires a new decision.

## Current Truth Card

```text
CURRENT_CODE=FULL_CHRONOLOGICAL_CACHED_FEATURE_Q2_REFERENCE
CRS_EPS_IMPLEMENTED=NO
ROUND2_PROTOCOL=ACCEPTED_WITH_AMENDMENTS
HYBRID_VALUE=UNKNOWN_UNTIL_G0_AND_PROFILE
Q2_EFFECTIVENESS=UNKNOWN
FULL_PETAL_HEADLINE=NOT_SUPPORTED
P0_LAUNCH_WORKDIR=OPEN
NEXT_ALLOWED_CODE=P0_FIX_AND_CPU_MANIFEST_PATH
NEXT_GPU_HOUR=NOT_AUTHORIZED
```

## Do Not Reinterpret This Record

- `GO-HYBRID-PROTOCOL` is not a positive result.
- Exact IPW is not state-bias correction.
- A direct twin trace is not the longitudinal treatment effect.
- One optimizer event per video is not equal compute.
- A one-seed resource kill is not a scientific null result.
- Cached-token causality is not raw-video causality or measured online latency.
- FineAction overlap is not automatically same-class identity evidence.
- A zero LoRA-by-binding interaction is not automatically a Stage-2 failure.
- Episodes in one video group share gradients only, never residual runtime state.
- OAT is verified ECCV 2022 prior art, not an unresolved literature lead.
