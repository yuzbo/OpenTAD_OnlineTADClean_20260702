# Post-Q2/R1-Kill On-TAD Route Review: Independent Absorption

Date: 2026-07-17 Asia/Shanghai
Source review date: 2026-07-16 America/New_York
Source file: `PRO_POST_Q2_R1_KILL_ONTAD_ROUTE_REVIEW_20260717.md`
Source attachment SHA-256:
`6E7935741C497D90000CED25AEB6D4A078919882065D8E082EDABBE7CB148C7F`
Source size: 80,830 bytes, 1,190 text lines
Reviewed code anchor: `7a26e60fccec68cd2b547951320e669a81907461`

## Absorption Verdict

Accept the review's final choice `GO_NEW_ROUTE_P0_ONLY`, with one execution
amendment:

> R-A, Prefix-Shared Latent Event Filter with Immutable Ledger, is authorized
> for preregistration closure and a CPU-only P0 mechanism probe. P0 code is not
> an effectiveness experiment and may not run until every execution-defining
> constant omitted by the review is frozen in a clean preregistration commit.

This does not reverse any prior terminal result:

- CRS-EPS empty-state dynamic replay remains `KILL`;
- Q2 remains valid `REVISE_REQUIRED` evidence with independent disposition
  `KILL`;
- R1/CSFSB remains terminal `KILL`;
- `birth_prior_bias_m2` and all Q2 threshold/K/refractory/release-order
  revisions remain forbidden;
- cached-feature G0, GPU profile, effectiveness/formal training, backbone
  fine-tuning, and raw-video training remain blocked;
- GPU hours authorized remain zero.

## What Is Accepted

### Task and gap

The task remains standard fully supervised On-TAD with causal input and
immutable `{start, end, class, score}` commits. The defensible gap is not
"end-to-end On-TAD does not exist." It is narrower:

> Standard On-TAD lacks a demonstrated prefix-level contract in which one
> model-generated latent instance state is shared by training and inference,
> while GT assignment evaluates that state only through loss and never changes
> runtime identity, availability, reset, risk set, or emission.

Detector-level and cached-feature end-to-end On-TAD already exist. Raw-RGB
causal joint interval training was not confirmed by the review, but absence
from one search is not an absolute novelty claim.

### Failure diagnosis

The review correctly separates the two prior failures:

- CRS-EPS changed the estimand by omitting the chronological state and its
  historical Jacobian;
- Q2 used different state machines for training ownership and runtime decode;
- both imply that a successor must preserve complete chronological forward
  state and keep GT outside model state transitions.

The Q2 evidence does not kill every persistent-carrier method. It kills the
family where predicted hard runtime availability gates GT canonical
supervision and can permanently discard first-crossing births.

### Selected route

R-A is retained as the sole P0 candidate:

- K continuous latent event carriers;
- model-generated continuous occupancy, start posterior, class evidence, and
  causal memory;
- one shared `advance_and_decode` transition for train and inference;
- completion-driven release before same-bin reseed;
- prefix-visible unbalanced OT used only as a temporary loss weighting;
- model-only immutable ledger latch;
- first-birth and first-end/survival targets;
- explicit anti-silence readiness, so no-birth/no-emission/always-background
  cannot pass.

R-A is a new route, not a Q2 patch. It must be implemented in new modules and
must not mutate or reuse Q2 evidence tickets.

### Portfolio disposition

- R-A: selected for P0 only.
- R-B Endpoint-Set Completion: baseline only; occupied by
  MATR/OAT/PCEH-style reconstruction.
- R-C Causal Interval Transducer: theoretical fallback; no implementation
  permission.
- R-D Teacher-Student Runtime State Alignment: rejected as potentially
  distilling the same GT-conditioned mismatch.

## Qualifications and Amendments

### Evidence scope

The Pro reviewer verified the repository at the fixed commit but did not have
the external CRS/Q2 checkpoint and trace bundle. Its statements about the
524,258-row replay are inherited from the repository's bound review record,
not a new byte-level replay. This does not reopen either KILL.

### Novelty scope

`novelty_pass=true` means a P0 hypothesis is worth falsifying. It is not a
publication novelty pass. The strongest reconstruction attack is temporal
MOTR/TrackFormer plus interval heads and an online ledger. R-A survives only
if a future matched reconstruction cannot reproduce its identity/latency
benefits. Persistent queries, soft occupancy, OT, causal memory, and a new loss
are not individually novel.

### Contract scope

The review's `runtime_supervision_contract_pass=true` means the written design
is coherent enough for P0. The implementation is explicitly
`SPECIFIED_NOT_IMPLEMENTED`; no code or evidence currently proves the
contract.

### Preimplementation preregistration gap

The review freezes the P0 families, seeds, headline anti-silence margins, and
terminal rule, but not all executable constants. Before scientific code or a
P0 runner is written, a clean preregistration must freeze:

1. carrier count K, hidden dimensions, memory length, initial state, and the
   exact direct-complete enable/disable rule;
2. every transition equation's tensor shape, clamping, detach policy, and
   release/reseed tie order;
3. UOT cost terms, dustbin semantics/capacity, entropic epsilon, mass
   relaxation, Sinkhorn iterations/tolerance, numerical stabilization, and
   gradient policy;
4. all loss weights, positive/risk-mass denominators, graph-connected zero
   behavior, and transport-consistency horizon;
5. ledger thresholds, score construction, class/start snapshot timing,
   collision/tie breaking, and latch-reset invariants;
6. synthetic generator distributions, sequence lengths, class counts,
   overlap/repetition/delay ranges, noise, train/holdout construction, and
   generator hash;
7. optimizer, learning rate, schedule, initialization, batch/episode order,
   number of updates, TBPTT boundary, clipping, stopping rule, and CPU
   determinism controls;
8. exact-versus-tolerance comparison rules for state/logit equality and all
   anti-silence metrics;
9. CPU thread count, interpretation of the two-CPU-hour cap, wall timeout,
   memory/storage cap, and atomic evidence schema;
10. a ban on changing any exposed holdout, threshold, margin, or distribution
    after outcome observation.

These are not reasons to reject R-A. They are required to turn a mathematical
route sketch into a reproducible P0 contract.

### Direct-complete branch

P0 may test the direct-complete contract on synthetic cases. Enabling that
branch in any real-data route remains blocked by an outcome-blind annotation
census. It cannot become a post hoc fix.

## Frozen P0 Outcome Contract

The following requirements are imported unchanged from the review:

- deterministic synthetic families: single event, delayed endpoint,
  same-class repetition, cross-class overlap, same-class overlap, same-bin
  end/birth, same-interval start/end, and all-background;
- scripted/oracle contract: exact event count, recall 1, precision 1,
  duplicate 0, fragmentation 0, future access 0, background emission 0, and
  identical stepwise/packet ledger bytes;
- train/inference/loss-off/stepwise/packet scientific state equality;
- GT-future and video-future perturbation invariance;
- learned CPU-FP32 seeds `9101/9102/9103`, 512 training sequences, 256
  outcome-blind holdout sequences, plus 64 all-background sequences;
- every seed separately: macro recall >= 0.90, precision >= 0.90,
  same-class-overlap recall >= 0.80, mean absolute count error <= 0.10 per
  positive sequence, background FP 0, duplicate fraction 0, fragmentation 0;
- no-birth/no-emission/always-background controls must have recall 0 and
  readiness `FAIL`;
- finite and nonzero applicable gradients, correct perturbation direction,
  exact optimizer coverage, detached immutable records;
- any failure is terminal `P0_STATUS=KILL`.

The learned gate is a falsification probe, not evidence of THUMOS14
effectiveness or publication novelty.

## Authorization Matrix

| Action | Status |
|---|---|
| Archive and absorb the review | `ALLOW` |
| Freeze an exact P0 preregistration | `ALLOW` |
| Implement new R-A P0 modules after preregistration closure | `ALLOW` |
| Run frozen synthetic CPU P0 | `ALLOW_AFTER_PREREG` |
| Modify or rerun Q2/R1 | `FORBIDDEN` |
| Access real reporting data for P0 | `FORBIDDEN` |
| Cached-feature real-data G0 | `BLOCKED_PENDING_P0_REVIEW` |
| GPU profile | `BLOCKED` |
| Effectiveness/formal training | `BLOCKED` |
| Raw-video/backbone training | `BLOCKED` |
| GPU hours | `0` |

## Next Legal Sequence

```text
archive and absorb review
-> freeze exact CPU-only P0 preregistration
-> implement new modules and tests without modifying Q2
-> local deterministic contract tests
-> atomically publish frozen P0 evidence
-> independent read-only P0 review
-> at most PASS_AUTHORIZE_UNSEEN_CPU_G0_ONLY
```

No result in this chain can directly authorize GPU work.

## Machine-Readable Absorption

```json
{
  "source_sha256": "6E7935741C497D90000CED25AEB6D4A078919882065D8E082EDABBE7CB148C7F",
  "reviewed_commit": "7a26e60fccec68cd2b547951320e669a81907461",
  "accepted_decision": "GO_NEW_ROUTE_P0_ONLY",
  "selected_route": "R-A_PREFIX_SHARED_LATENT_EVENT_FILTER",
  "q2_r1_disposition": "KILL_UNCHANGED",
  "publication_novelty_pass": false,
  "p0_design_authorized": true,
  "p0_preregistration_closed": false,
  "p0_code_present": false,
  "p0_run_started": false,
  "p1_allowed": false,
  "gpu_profile_allowed": false,
  "formal_training_allowed": false,
  "raw_video_training_allowed": false,
  "gpu_hours_authorized": 0
}
```
