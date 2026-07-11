---
type: claim
node_id: claim:c6-budgeted-eval
title: "Budgeted online evaluation with GT-end latency prevents metric gaming"
status: partially-supported
updated: 2026-07-11
---

# Claim C6: Budgeted Online Evaluation

## Statement

OnlineAP should count a detection as TP only when class, tIoU, and latency budget relative to matched GT end all pass; late predictions remain FP and their GT remains FN.

## Current Support

Pro review accepted the main logic as mostly correct after local updates.

## Required Additions

- fixed class universe;
- emission IDs;
- ledger audit as evaluator precondition;
- latency coverage: TP count, recall, FN, late FP;
- revision and commit metrics for CESR.

## Failure Criterion

If the evaluator drops late predictions before AP, retunes thresholds per budget, or reports latency only over sparse TPs without coverage, the metric is invalid.
