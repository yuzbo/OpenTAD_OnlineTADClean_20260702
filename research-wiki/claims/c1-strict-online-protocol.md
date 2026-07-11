---
type: claim
node_id: claim:c1-strict-online-protocol
title: "Strict inference-time online protocol"
status: unproven
updated: 2026-07-11
---

# Claim C1: Strict Inference-Time Online Protocol

## Statement

The model can run with prefix-causal inputs and state, with no future raw-frame reads, no future cache reads, no EOF post-processing, and immutable committed detections.

## Current Support

Partial smoke-level support exists for the frozen/adaptor route:

- packet/update/causal replay smoke passed on one remote run;
- ledger skeleton records emission and source provenance;
- late predictions are intended to remain FP.

## Missing Evidence

- formal checkpoint replay;
- LoRA/full-tower replay;
- GT taint audit;
- cache manifest and raw/cache equivalence;
- append-only revision/commit ledger for CESR.

## Failure Criterion

Any future suffix perturbation that changes pre-cut state, hypothesis, or committed detection invalidates the claim.
