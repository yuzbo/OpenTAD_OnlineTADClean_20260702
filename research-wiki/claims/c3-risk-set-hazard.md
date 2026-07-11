---
type: claim
node_id: claim:c3-risk-set-hazard
title: "Instance-aware endpoint and commit hazards"
status: refuted-current-code
updated: 2026-07-11
---

# Claim C3: Instance-Aware Endpoint and Commit Hazards

## Statement

The method learns valid discrete risk-set hazards for endpoint and first commit/emission events.

## Current Verdict

False in current code according to 2026-07-11 absorption.

## Reasons

- endpoint and emission labels are the same crossing event;
- late prefixes are repeated positives;
- class-level aggregation pollutes repeated same-class events;
- predicted endpoint equals emit time;
- post-event risk masking is not yet valid.

## Required Repair

- `prefix_event_targets` or equivalent;
- per-instance risk set;
- one endpoint event per instance;
- one first-commit/emission event per instance;
- post-event mask;
- delayed synthetic case.
