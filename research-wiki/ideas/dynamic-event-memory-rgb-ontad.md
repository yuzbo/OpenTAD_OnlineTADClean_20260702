---
type: idea
node_id: idea:dynamic-event-memory-rgb-ontad
title: "Raw-RGB Dynamic Event Memory On-TAD"
stage: review-revised
outcome: core-factorial-pending
updated: 2026-07-22
target_gaps: ["G1", "G2", "G5", "G6", "G15", "G16"]
---

# Raw-RGB Dynamic Event Memory On-TAD

## Thesis

Build a standard, fully supervised, strictly causal On-TAD model that consumes
original RGB, detects a start transition as soon as it is observable, creates a
dynamic event record, maintains that same instance through its duration, and
emits one immutable positive-length interval when its end is detected.

Feature-level experiments are only the first reproduction and mechanism-
attribution layer. They are not the final input form or paper claim.

## Official Donors and Baselines

- A: ActionSwitch for immediate state transition, overlap protocol, and
  conservative lifecycle behavior.
- B: MATR for strong class/boundary localization and the end-centric memory
  baseline.
- C: HAT/OAT for short/long history fusion, anchors, and online suppression.
- D: 2025 Hierarchical Event Memory for multiscale event construction, merging,
  dynamic memory allocation, and latency accounting.

All four official repositories are preserved completely as detached, read-only
reference baselines. They are never patched. Only the direct parents needed by
the surviving hypothesis are reproduced before the core screen; HAT/OAT and HEM
do not block that screen and become full baselines only if the memory claim is
activated. Every port traces back to an official SHA and source path.

Official-native baselines do not require OpenTAD conversion. A neutral adapter is
allowed only after native fidelity and only to standardize timestamps, splits,
decoded messages, causal ledgers, metrics, and resource accounting. It must not
replace official target generation, loss, matching, memory, decoder, or post-
processing. OpenTAD is an experiment container for the new route, not an
innovation claim.

## Primary Model

- dense causal start evidence creates birth-allocated event records, not fixed
  semantic slots;
- each event owns its start anchor, identity, class trajectory, continuation and
  end state, so an end cannot freely match a different start;
- active-event state is retained until end/cancel with learned effective
  cardinality and an explicit, measured physical fail-closed guard;
- visual history is compressed through learned sample-conditioned retention and
  merge decisions, with active start anchors protected;
- start/end decision policies are calibrated on the training-side calibration
  split; a uniform 0.5 threshold is an ablation;
- the end boundary is decoded from observed evidence independently of the later
  commit time;
- a real causal streaming encoder receives raw RGB and participates in the
  registered training graph after the feature mechanism survives.

The authoritative standard output is `{start, end, class, score}`. Event IDs and
provisional starts remain internal audit/diagnostic state. OnVLLM is a separate
follow-up project, not a main-paper head.

## Feature-Level Novelty Audit — 2026-07-22

Verdict: **proceed with caution**. A fresh primary-source search through
2026-07-22 and an independent adversarial review found no single feature-input,
fully supervised, strictly causal On-TAD method that implements the whole
proposal. The feature route therefore retains a narrow publication opportunity,
but the broad C1--C5 package is not itself a defensible contribution.

The viable core is `variable-effective-cardinality identity-persistent On-TAD`:
an event is born with its own start anchor, and class/continue/end predictions
remain owned by that identity until closure. This directly targets MATR's stated
risk that multiple remembered instances can be paired with an incorrect start.
It also replaces ActionSwitch's manually selected switch count and combinatorial
state table with an event set whose effective cardinality follows observed
births. This core is still vulnerable to the objection that it is a temporal
adaptation of MOTR/TrackFormer-style persistent track queries, so the learning
rule and task-specific association evidence must be substantive.

The following are not standalone novelty claims:

- immediate start detection and overlapping same-class actions (ActionSwitch);
- long/history memory (MATR and HAT/OAT);
- hierarchical events, dynamic per-scale allocation, and adaptive merging (HEM);
- persistent identity queries in the abstract (MOTR/TrackFormer);
- strict causality, calibrated decisions, read-only donor fidelity, or an A+B+D
  module list.

`ActiveEventMemory` protection remains a necessary correctness invariant, not a
headline contribution. Learned physical compression remains a later claim and
cannot be made by the full-prefix safety variant. The feature model has not yet
been implemented or shown competitive: the completed fixed-slot result remains
a negative baseline, not evidence for the dynamic route.

The feature route survives only if matched-feature experiments beat its direct
parents and specifically reduce wrong-start association and identity switches
under same-class overlap, crossed end order, adjacency, long actions, and rising
concurrency without materially increasing end-commit delay. Required controls
include ActionSwitch at several switch counts, MATR, a MATR + MOTR-style temporal
track-query control, and the proposed event-owned trajectory. THUMOS14/MUSES
must be supplemented by a concurrency-sensitive set such as FineAction or
MultiTHUMOS.

## Active Core Factorial

The pre-review A/B/C/D fusion matrix is retired as the executable plan. It mixed
capacity, ownership, history, backbone and post-processing and could not identify
the source of a gain. The active feature-level screen uses two mechanism axes in
one code path:

```text
K0 = fixed preallocated query/switch bank
K1 = birth-allocated packed event set with sample-varying active cardinality

O0 = per-prefix rematching/reassignment
O1 = sticky birth-time owner until cancel/end
```

The four arms are `K0O0`, `K1O0`, `K0O1`, and `K1O1`. They share the same
physical safety guard, encoder, decoder depth, parameter budget, data exposure,
successful updates and calibration rule. `K0O1` must be a credible Temporal
TrackFormer control; `K1O1` is the proposed arm. Native ActionSwitch and MATR
remain direct task baselines. HAT/OAT and HEM are later memory-family baselines,
not mandatory parents of the primary model.

Only if ownership and birth allocation survive this screen does a fixed-budget
`hierarchy × learned retention` study begin. Raw-RGB frozen/adapter/joint training
then tests whether the same mechanism survives visual adaptation. No donor-letter
fusion or ABCD model is presumed to survive.

## Success Conditions

Zero future-information, sequence, positive-length, uniqueness, association, and
immutability violations; `K1O1` must beat both direct factorial parents and the
Temporal TrackFormer control while specifically reducing wrong-start and owner-
swap errors; no material commit-delay or resource regression; and, later, a raw-
RGB receipt proving no cached-feature fallback and real visual-parameter updates.

Exact update counts, physical guards, memory-token budgets, loss weights and
numeric effect gates remain unfrozen until parent variance and learning/resource
curves are measured.

## Connections

[AUTO-GENERATED from graph/edges.jsonl - do not edit manually]
