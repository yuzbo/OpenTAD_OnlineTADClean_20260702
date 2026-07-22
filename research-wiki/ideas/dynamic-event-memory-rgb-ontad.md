---
type: idea
node_id: idea:dynamic-event-memory-rgb-ontad
title: "Raw-RGB Dynamic Event Memory On-TAD"
stage: design-written
outcome: pending-implementation
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
reference baselines. They are never patched. Writable A/B/C/D compatibility
replicas and AB/AD/BD/BC/ABD/ABCD fusion workspaces are separate and trace every
ported or changed file back to an official SHA and source path.

Official-native baselines do not require OpenTAD conversion. A neutral adapter is
allowed only after native fidelity and only to standardize timestamps, splits,
decoded messages, causal ledgers, metrics, and resource accounting. It must not
replace official target generation, loss, matching, memory, decoder, or post-
processing. OpenTAD is an experiment container for the new route, not an
innovation claim.

## Primary Model

- dense causal start hazard creates ragged event records, not fixed slots;
- each event owns its start anchor, identity, class trajectory, continuation and
  end state, so an end cannot freely match a different start;
- active-event state is retained until end/cancel without a semantic capacity;
- visual history is compressed through learned sample-conditioned retention and
  merge decisions, with active start anchors protected;
- start/end decision policies are calibrated on the training-side calibration
  split; a uniform 0.5 threshold is an ablation;
- a real causal VideoMAE-style encoder receives raw RGB and participates in the
  registered training graph.

The optional OnVLLM adapter can describe the same active/finalized event IDs,
but the structured On-TAD interval remains authoritative.

## Fusion Matrix

Faithful A/B/C/D baselines precede AB, AD, BD, BC, ABD, and optional ABCD. The
paper main candidate is raw-RGB ABD. Each fusion is compared with both parents
under matched input, update budget, calibration, decoder contract, and metrics.

The matrix is hypothesis decomposition rather than module accumulation:

- AB asks whether immediate start birth and strong localization are compatible;
- AD asks whether a born event can be maintained by hierarchical history without
  fixed switches or fixed windows;
- BD asks whether memory structure alone improves MATR while its end-centric
  lifecycle remains unchanged;
- BC compares a different long/short-history donor against D;
- ABD survives only if its parent experiments expose complementary gains;
- ABCD is deleted unless C adds an isolated gain on top of ABD.

The faithful MATR arm retains its official current-end/past-start behavior. The
primary ABD arm does not: the event created at start owns all subsequent class,
continuation, memory and end decisions, while the MATR-derived localizer supplies
localization capacity under that event-conditioned contract.

## Success Conditions

Zero future-information, sequence, positive-length, uniqueness, association, and
immutability violations; credible official-baseline fidelity; improved accuracy-
delay-memory Pareto frontier; robust same-class/different-class overlap; and a
raw-RGB receipt proving no cached-feature fallback and real visual-parameter
updates.

## Connections

[AUTO-GENERATED from graph/edges.jsonl - do not edit manually]
