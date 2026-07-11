---
type: claim
node_id: claim:c4-crs-eps-cost
title: "CRS-EPS controls training cost without invalidating online evaluation"
status: unproven
updated: 2026-07-11
---

# Claim C4: CRS-EPS Cost Control

## Statement

Event-centric prefix-episode training plus frozen feature cache can reduce training wall time dramatically while preserving the online evaluation claim.

## Current Rationale

Full-packet training is too expensive and fragmented. Sampling start/end/ongoing/background episodes should concentrate learning on informative decision points.

## Needed Evidence

- throughput benchmark;
- packets/s or episodes/s;
- GPU-hours;
- full-packet gold subset comparison;
- sampled objective bias controls;
- full chronological evaluation after sampled training.

## Failure Criterion

If sampled objective diverges from full-packet gold subset or full chronological evaluation collapses, CRS-EPS is not valid as the main training protocol.
