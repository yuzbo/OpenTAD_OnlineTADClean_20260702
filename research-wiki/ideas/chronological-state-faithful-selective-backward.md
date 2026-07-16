---
type: idea
node_id: idea:csfsb-training
title: "Chronological State-Faithful Selective Backward (CSFSB)"
stage: killed-by-capacity-contract
outcome: terminal-kill
updated: 2026-07-16
target_gaps: ["G3", "G4", "G7"]
---

# Chronological State-Faithful Selective Backward

## Status

R1/CSFSB was the only successor conditionally retained by the 2026-07-16
Round-2 route adjudication. That conditional implementation permission is now
withdrawn because its prerequisite capacity contract failed.

The clean capacity audit returned terminal `REVISE_REQUIRED`: current Q2
exhausts 2,206 births, and the sole eligible zero-exhaustion counterfactual is
an additive `-2` birth-logit prior with only ten emissions. Unique independent
review diagnosed degenerate birth suppression and selected `KILL_Q2_R1`. No
shared contract is frozen; R1 implementation is terminated under this
protocol, with GPU profile and formal training blocked.

```text
CAPACITY_AUDIT=REVISE_REQUIRED
INDEPENDENT_DISPOSITION=KILL_Q2_R1
R1_IMPLEMENTATION=KILL
PROFILE=BLOCK
FORMAL=BLOCK
GPU_HOURS_AUTHORIZED=0
```

## Problem

Empty-state event-episode replay failed because HH/IPW can correct sampling
exposure but cannot reconstruct omitted persistent state, lifecycle ownership,
or historical Jacobian paths. Complete chronological Q2 preserves those paths
but may spend excessive backward compute on redundant bins.

## Historical Method Design

Run every cached token causally and chronologically. Advance persistent query
state, feature memory, runtime lifecycle, canonical supervision, births,
retirements, refractory counters, and ownership at every bin. Use the frozen
event proposal only to choose weighted loss and backward contributions.

```text
M=4
H=8
mixture=uniform .40 / start .15 / end .20 / ongoing .10 / hardbg .15
a_vt=(m_vt/M)*((1/T_v)/rho_vt)
```

Keep the existing global 64-bin TBPTT boundaries. A block without selected
bins runs entirely under `no_grad`. A selected block builds a graph from its
start through the last selected bin, computes losses only at selected bins,
then detaches and runs the tail under `no_grad`. State still advances through
all bins, and the optimizer still steps once per video.

## Scientific Estimand

R1 targets the current chronological 64-bin TBPTT objective, not full-video
BPTT. Under correct probabilities, identical selected-bin loss definitions,
and no realized-sample renormalization, its gradient is unbiased in expectation
relative to that objective. A single finite-M gradient is not exact and may be
too noisy for AdamW; G0-B must test practical adequacy.

## Capacity Gate Outcome

The required 160-video, seeds 705/706/707 same-logits audit completed. Actual
Q2 exhausted 2,206 GT births. No case was true canonical capacity, and the sole
legal zero arm achieved zero by suppressing emissions to ten. The independent
review therefore took the preregistered kill branch.

## Cancelled G0 Design

- G0-A: exact paired implementation closure between all-graph selected-bin
  gold and selective no-grad execution.
- G0-B: finite-M aggregate loss/gradient fidelity against all-bin chrono-64
  over eight unseen videos per checkpoint stratum.
- G0-C: zero slot exhaustion across every seed, checkpoint, sample, arm, and
  assignment mode.
- Family-wise PASS only; any failure is terminal KILL.

## Claim Boundary

CSFSB is initially an enabling training substrate. It does not reduce full
forward tokens and has strong reconstruction risk from ETAD, selective
backward, stochastic backpropagation, activation checkpointing, and standard
importance sampling. It becomes an efficiency result only if matched profiling
shows lower backward time, total wall time, and GPU-hours without quality loss.
It is not yet Full PETAL's paper-level contribution.

## Terminated Ordered Gate

```text
capacity audit: REVISE_REQUIRED
-> independent review: KILL_Q2_R1
-> STOP; no R1, B0, G0, profile, or training
```

## Sources

- [`../../PRO_CRS_EPS_G0_KILL_ROUTE_REVIEW_ROUND2_20260716.md`](../../PRO_CRS_EPS_G0_KILL_ROUTE_REVIEW_ROUND2_20260716.md)
- [`../../PRO_CRS_EPS_G0_KILL_ROUTE_REVIEW_ROUND2_ABSORPTION_20260716.md`](../../PRO_CRS_EPS_G0_KILL_ROUTE_REVIEW_ROUND2_ABSORPTION_20260716.md)
- [`../../PRO_Q2_CAPACITY_INDEPENDENT_MAX_REVIEW_20260716.md`](../../PRO_Q2_CAPACITY_INDEPENDENT_MAX_REVIEW_20260716.md)
- [DR-041](../decision_register.md#dr-041-select-csfsb-conditionally-behind-a-capacity-first-gate)
- [DR-044](../decision_register.md#dr-044-kill-q2-and-r1-after-independent-degeneracy-adjudication)
- [T34](../discussion_timeline.md#t34-round-2-selects-csfsb-but-authorizes-capacity-audit-first)
- [T37](../discussion_timeline.md#t37-independent-review-kills-q2-and-r1)
