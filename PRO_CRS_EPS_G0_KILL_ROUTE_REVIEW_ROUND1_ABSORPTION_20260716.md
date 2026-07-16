---
type: pro-review-absorption
date: 2026-07-16
source: PRO_CRS_EPS_G0_KILL_ROUTE_REVIEW_ROUND1_20260716.md
source_sha256: D0AB7B51961E61F9DBD45192743AC61B5522646F2FA1027B96489C8010ADC46F
source_lines: 1417
source_bytes: 67496
scientific_commit: 70df86ea3d38d70c658ae0ee9e04245d57b834d4
reviewed_head: 02f5f851defa1c7413d93f0f365895cd7a6509fd
verdict: partial_accept_with_material_amendments
---

# CRS-EPS G0 Kill Route Review: Independent Absorption

## Independent Verdict

I do **not** fully accept every statement in the Pro review. I accept its core
scientific conclusion with high confidence, but amend two route-level claims
and several pieces of terminology.

```text
FROZEN_G0_70df86e = VALID_KILL
CURRENT_EMPTY_STATE_DYNAMIC_BIRTH = STRUCTURAL_KILL
ALL_EVENT_CENTRIC_TRAINING = NOT_KILLED
CHRONOLOGICAL_Q2 = KEEP_AS_GOLD_REFERENCE_CANDIDATE
CHRONOLOGICAL_Q2_PRIMARY_STATUS = NOT_YET_EARNED
FULL_PETAL_INFRASTRUCTURE = KEEP_AS_ENGINEERING_SUBSTRATE
PROFILE = BLOCK
FORMAL = BLOCK
GPU_HOURS_BEFORE_ROUND2 = 0
LOCAL_CONFIDENCE = 0.96
```

The decisive point is narrow but strong: the current `dynamic_birth` arm
starts from an empty runtime state and replays only a selected suffix/context.
For a persistent-query detector, that history is not generally sufficient to
reconstruct the numerical state, slot lifecycle, supervision ownership, or
the gradient estimand of the chronological control. HH/IPW corrects which
decision bins are sampled; it cannot correct an unobserved recurrent state or
restore omitted historical Jacobian paths.

The result does **not** prove that every event-centric or selective-backward
training method is invalid. A replacement that preserves the exact
chronological forward state is a different method and requires a new
outcome-blind evidence chain.

## What I Accept

1. The exact `70df86e + seed 705 + frozen samples + frozen margins + frozen
   checkpoint` G0 is a valid terminal `KILL`.
2. Relative improvement over `fixed_192` or `reset` cannot override failed
   absolute fidelity margins.
3. The current empty-state `dynamic_birth` design has a structural state
   reconstruction problem, not a denominator-only or local-threshold bug.
4. `video_start_full` reconstructs the full chronological **numerical state**
   but shares the global 64-bin TBPTT detach schedule. It is not exact
   full-video BPTT.
5. The four exposed samples are development diagnostics only. Any replacement
   needs a new commit, clean B0, independent review, unseen holdout G0, and a
   new terminal result.
6. Profile and formal training remain blocked; no new GPU hour is justified
   before the Round-2 route decision.
7. Persistent detector, lifecycle, transactional rollback, provenance,
   signing, and launch gates remain useful engineering infrastructure. They
   are not evidence of method novelty or effectiveness.
8. Full causal forward plus selective backward, exact chronological
   recomputation, and state checkpoints are the correct replacement family to
   compare. None is yet selected or authorized.

## Material Amendments

### A1. Narrow the structural-kill label

The review headline `STRUCTURAL-KILL-CRS-EPS` is too broad. The defensible
label is:

> `STRUCTURAL-KILL-CURRENT-EMPTY-STATE-DYNAMIC-BIRTH-CRS-EPS`

This preserves the falsified result without implying that state-faithful
event selection, selective backward, checkpointed recomputation, or every
possible CRS-EPS successor has already been disproved.

### A2. Do not promote Q2 to primary yet

The review says `FULL_CHRONOLOGICAL_Q2 = KEEP_AS_PRIMARY`. The repository has
no accepted profile, effectiveness run, formal result, or gold checkpoint
trajectory for that route. More importantly, the raw G0 controls show
`slot_exhaustion` in two selected full-prefix controls. Project law treats
model-caused slot exhaustion as a scientific fail-closed condition.

Therefore the current status is:

> Keep chronological cached Q2 as the gold/reference **candidate** and
> diagnostic oracle for state fidelity. Do not call it the accepted primary
> training route until capacity/lifecycle readiness and cost are measured.

This amendment does not rescue the killed surrogate. It adds a separate
readiness blocker to the proposed fallback.

### A3. Name the gradient estimand precisely

Where the review says "full parameter gradient," the exact statement should
be:

> all trainable-parameter gradients under the shared global 64-bin TBPTT
> estimand.

Neither arm computes untruncated full-video BPTT. The paired G0 remains valid
because the truncation schedule is shared and the observed divergence is
therefore not explained by that common truncation alone.

### A4. Multi-checkpoint evidence is desirable, not free

Init/early/mid/late checkpoint strata are scientifically stronger than a
single deterministic initialization. However, no accepted chronological gold
trajectory exists, so generating those checkpoints may itself incur the cost
the route is meant to avoid. Round 2 must first define a bounded state-faithful
cost experiment and its minimum checkpoint strata; it must not assume the
gold trajectory already exists.

### A5. Correct the literature-to-reference map

The review's literature conclusions are broadly correct, but its footnote
mapping is incomplete:

- the marginal importance sampling claim should cite arXiv `1906.03393`, not
  only the doubly robust OPE paper `1511.03722`;
- the OnPoint/OZ-TAL/OpenHOUSE group is represented by an OnPoint-only
  footnote;
- the TALLFormer/Re2TAL/LoSA group is represented by a TALLFormer-only
  footnote;
- the checkpointing group is represented by only one of several works.

These are citation-audit defects, not contradictions of the core diagnosis.

## Local Raw-Artifact Verification

The Pro reviewer could not inspect the external signed bundle. This local
absorption did. The bundle remains outside the repository under the RTK data
boundary. Its terminal `audit.json` has SHA-256
`a08183a8dba4ec5267b5a75eed3501882180a22f28d6b76ce2c7b4e65511009b`.

### Exact selected cases

| Sample | Gold replay | Dynamic / fixed / reset replay | Gradient and supervised ranges | Largest loss mismatch | Gold / candidate exhaustion | Key final-state mismatch |
|---|---|---|---|---|---:|---|
| `video_validation_0000151:draw=0` | `[0,89]` | `[0,89] / [0,89] / [81,89]` | grad `[64,89]`; supervised `[81,89]` | none; positive fallback matches | `1 / 1` | none for dynamic |
| `video_validation_0000946:draw=1` | `[0,270]` | `[62,270] / [70,270] / [262,270]` | grad `[256,270]`; supervised `[262,270]` | birth `1.50543` vs `0.92055`, relative error `0.38851` | `0 / 0` | labels, slot status, and start-state NaN mask |
| `video_validation_0000051:draw=1` | `[0,258]` | `[58,258] / [58,258] / [250,258]` | grad `[192,256],[256,258]`; supervised `[250,258]` | alive `0.12856` vs `0.23168`, relative error `0.80212` | `0 / 0` | labels, slot status, and start-state NaN mask |
| `video_validation_0000163:draw=0` | `[0,409]` | `[209,409] / [209,409] / [401,409]` | grad `[384,409]`; supervised `[401,409]` | start `4.12198` vs `5.85178`, relative error `0.41965` | `2 / 0` | slot status |

Two failed cases therefore make `dynamic_birth` and `fixed_192` exactly equal
in replay range, not merely close in aggregate metrics. The only selected
dynamic-extension case extends fixed replay by eight bins and still has a
large state/loss mismatch. This directly supports the review's H2/H4/H5
diagnosis.

### First divergence in stored audit sequences

- `0000151`: no dynamic/gold divergence in birth assignment, canonical
  lifecycle, endpoint-slot trace, or loss bindings.
- `0000946`: those supervision audit sequences do not diverge over the shared
  range, but the learned runtime state still diverges. This is evidence that
  matching annotation-side bookkeeping is not sufficient state fidelity.
- `0000051`: the same pattern holds despite dynamic replay equaling fixed
  replay.
- `0000163`: canonical lifecycle and loss binding first diverge at global bin
  `209`; endpoint-slot trace first diverges at bin `216`; birth assignment
  first diverges at bin `352`.

The causal interpretation is stronger than the Pro review could establish
from summary metrics alone: omitted chronological history changes both
continuous recurrent state and, in a stress case, the discrete lifecycle and
ownership path.

### Selection representativeness

The signed metadata-only policy searched `160` videos x `4` draws = `640`
candidates, selected one draw from each video-length quartile, maximized union
coverage over six preregistered stress flags, then proposal-component
diversity, then a lexicographic tie-break. The four samples cover all six
flags: detach-boundary crossing, dynamic extension, action longer than 192
bins, overlap, same-class repetition, and true left censoring.

This is a strong mechanism-oriented falsification set, but four samples and
one random-initialization checkpoint do not justify a population-wide claim
about every model state or successor algorithm. They are sufficient to kill
the exact frozen gate because that gate was preregistered as conjunctive.

## Answers to the Review's U1-U8

### U1: Raw signed G0 bundle

Available locally and inspected. It contains `audit.json`, all 12 paired rows,
selection, margins, policy, manifest, deterministic seed-705 checkpoint, and
preflight records. The bundle is deliberately excluded from Git. Raw rows
reproduce the reported terminal `KILL` and add the case-level facts above.

### U2: Selection rule and strata

Resolved. Candidate universe `640`; one sample per exact length quartile;
selection objective is maximum stress-flag union, then proposal diversity,
then lexicographic tie-break. All six required flags and four proposal
components are covered. The policy explicitly discloses an earlier unsigned
diagnostic and freezes samples, margins, checkpoint, and method unchanged.

### U3: Slot exhaustion and lifecycle traces

Resolved materially. Gold/candidate dynamic exhaustion counts are `1/1`,
`0/0`, `0/0`, and `2/0` in sample order. The first three cases have no
annotation-side sequence divergence over their shared ranges; the fourth has
the exact lifecycle/endpoint/birth divergences listed above. Capacity is a
separate blocker for chronological Q2 readiness.

### U4: Intended training start

Resolved. The config's pilot seeds are `705,706,707`; normal training starts
fresh unless explicitly resumed; profile forbids resume. The G0 checkpoint is
the deterministic seed-705 initialization. The KILL is therefore directly
relevant to the intended first pilot seed and its first update estimand.

### U5: Gold checkpoint trajectory

Not available. There is no accepted chronological Q2 trajectory and no
init/early/mid/late checkpoint set. Profile and formal evidence directories
are absent. Round 2 must account for the cost of creating any such trajectory.

### U6: External evidence-chain validation

Available locally and previously revalidated on the target Linux chain. Known
roots include local B0
`33bbd46c69b75090ec30a7cd1893ea8c5a4596188d264fa4900c6089eba92371`,
POSIX B0
`4888446ec24c9b7d3a5ecbdbadd63199213f81dfb877f4de9106f99ca1e5b86d`,
same-review evidence
`be4b82aa9232a10d9da62239afaf9878cd8b4c53b8ad35784c75ac0a72b9a06b`,
and G0 audit
`a08183a8dba4ec5267b5a75eed3501882180a22f28d6b76ce2c7b4e65511009b`.
The Pro session's inability to access them was an environment limitation, not
an absent project artifact.

### U7: Same-hardware chronological profile

Not available. No profile artifact or accepted R0 cost measurement exists.
Consequently the relative cost ordering of chronological forward,
selective-backward, and state-checkpointed alternatives remains unknown.

### U8: Feature-cache provenance

Partially resolved. The evidence binds the annotation, data identity, feature
cache manifest, split manifest file, and split-manifest identity by hash. Raw
feature arrays and the source extractor artifacts are outside the G0 bundle,
so full third-party reconstruction remains an open reproducibility item. This
does not alter the G0 result.

## Updated Research Decision

The active research object remains standard fully supervised On-TAD with
persistent instance state and immutable online outputs. The current result
changes the training route, not the task:

1. Freeze the old CRS-EPS result as a negative scientific result.
2. Retain Full PETAL code only as an audited detector/lifecycle/evidence
   substrate.
3. Retain chronological cached Q2 as a gold/reference candidate, subject to a
   separate slot-capacity and cost gate.
4. Compare only state-faithful cost-control families in Round 2:
   chronological forward plus selective backward, exact chronological
   recomputation, and explicit state checkpointing.
5. Do not claim method novelty until a state-faithful route passes G0 and the
   persistent-binding mechanism beats matched FRESH and Temporal TrackFormer
   baselines under the immutable On-TAD evaluator.

## Exact Next Gate

The next task consumes **zero GPU hours**:

1. publish this raw-artifact postmortem and U1-U8 author response;
2. ask the same Pro reviewer for Round 2 to rank the state-faithful candidates
   under explicit estimands, memory/runtime bounds, capacity handling, and a
   minimum unseen multi-checkpoint G0;
3. select at most one replacement route and write a new preregistered
   scientific contract;
4. only then implement, run B0, obtain independent review, and execute a new
   holdout G0;
5. profile is allowed only after terminal G0 `PASS`; formal training remains a
   later gate.

No margin relaxation, sample recycling, route relabeling, profile launch, or
formal training is authorized by this absorption.
