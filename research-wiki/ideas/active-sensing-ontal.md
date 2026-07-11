---
type: idea
node_id: idea:active-sensing-ontal
title: "Compute-Adaptive Active Sensing for Online Temporal Localization"
stage: proposed
outcome: pending
updated: 2026-07-11
target_gaps: ["G12"]
---

# Compute-Adaptive Active Sensing for Online Temporal Localization

## Thesis

A streaming localizer should actively choose temporal sampling rate, resolution, or backbone depth from boundary uncertainty so that it spends computation near informative transitions and monitors background cheaply.

## Value

This directly addresses deployment latency and energy instead of reporting model FLOPs after the fact. A useful system would run a cheap monitor, increase observation density around possible starts/ends, and return to low-cost monitoring after uncertainty resolves.

## Novelty Boundary

Adaptive frame selection, dynamic resolution/depth, system-aware scheduling, and dynamic event-boundary networks already exist. Therefore generic "adaptive computation" is not a defensible main claim.

The only plausible delta is a decision-theoretic policy tied specifically to instance localization risk: spend compute when the expected reduction in span/commit risk exceeds its latency or energy cost.

## Role

Secondary candidate or systems contribution. Do not make it the paper headline unless it beats SAN-style dynamic networks, AdaFrame-style sampling, and DyBDet-style dynamic boundary detection under equal hardware budgets.
