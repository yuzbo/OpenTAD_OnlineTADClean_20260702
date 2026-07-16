---
type: idea
node_id: idea:csfsb-training
title: "Chronological State-Faithful Selective Backward (CSFSB)"
stage: capacity-gated-implementation-candidate
outcome: pending
updated: 2026-07-16
target_gaps: ["G3", "G4", "G7"]
---

# Chronological State-Faithful Selective Backward

## Status

R1/CSFSB is the only successor retained by the 2026-07-16 Round-2 route
adjudication. It has a conditional implementation GO, not method approval.
The immediate gate is the zero-GPU Q2 capacity/lifecycle audit. R1 code may be
implemented only after one non-tainted shared fixed/rematch contract is frozen.

```text
CAPACITY_AUDIT=ALLOW
R1_IMPLEMENTATION=CONDITIONAL
PROFILE=BLOCK
FORMAL=BLOCK
GPU_HOURS_BEFORE_NEW_G0_PASS=0
```

## Problem

Empty-state event-episode replay failed because HH/IPW can correct sampling
exposure but cannot reconstruct omitted persistent state, lifecycle ownership,
or historical Jacobian paths. Complete chronological Q2 preserves those paths
but may spend excessive backward compute on redundant bins.

## Method

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

## Capacity Gate

Before R1 implementation, audit all 160 fit-core videos with seeds 705/706/707
and same-logits lifecycle counterfactuals. Separate canonical occupancy, false
ACTIVE occupancy, refractory occupancy, and same-bin ordering. Make ordinary
chronological training fail before optimizer on any exhaustion. Freeze one
shared policy or kill Q2/R1.

## New G0

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

## Ordered Gate

```text
capacity audit
-> shared lifecycle contract
-> R1 implementation and CPU tests
-> local/target-Linux B0
-> same independent reviewer PASS
-> unseen CPU G0 PASS
-> <=2 GPU-hour profile
-> one-seed fixed/rematch mechanism kill
-> FRESH/TTF/GRU controls
-> multi-seed and second-dataset evidence
```

## Sources

- [`../../PRO_CRS_EPS_G0_KILL_ROUTE_REVIEW_ROUND2_20260716.md`](../../PRO_CRS_EPS_G0_KILL_ROUTE_REVIEW_ROUND2_20260716.md)
- [`../../PRO_CRS_EPS_G0_KILL_ROUTE_REVIEW_ROUND2_ABSORPTION_20260716.md`](../../PRO_CRS_EPS_G0_KILL_ROUTE_REVIEW_ROUND2_ABSORPTION_20260716.md)
- [DR-041](../decision_register.md#dr-041-select-csfsb-conditionally-behind-a-capacity-first-gate)
- [T34](../discussion_timeline.md#t34-round-2-selects-csfsb-but-authorizes-capacity-audit-first)
