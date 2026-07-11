---
type: claim
node_id: claim:c2-mutable-hypothesis-commit
title: "Identity-linked belief trajectories plus utility-based immutable commit improve On-TAD"
status: unproven
updated: 2026-07-11
---

# Claim C2: Mutable Hypothesis + Immutable Commit

## Statement

Maintaining identity-linked action belief trajectories before commit, and learning a first-stop commit policy under localization, latency, and stability cost, improves OnlineAP-latency behavior over repeated anonymous proposals and fixed stopping rules.

## Why It Matters

This narrowed statement is now the central CESR claim. Mutable predictions, state transitions, or boundary refinement alone are not novel.

## Needed Evidence

- revision trajectories improve boundary accuracy over time;
- commit policy gives better AP-latency Pareto than one-shot endpoint emission;
- revision metrics are auditable and cannot be confused with illegal post-hoc modification;
- comparison with OAT/ActionSwitch-style baselines.
- fixed-threshold, wait-k, confidence-only, uncertainty-only, and oracle stopping controls;
- a test that identity-linked trajectories outperform post-hoc matching of repeated proposals.

## Failure Criterion

If identity tracking and utility stopping do not independently improve the latency-quality-stability frontier, demote CESR to a protocol/diagnostic rather than a main method.
