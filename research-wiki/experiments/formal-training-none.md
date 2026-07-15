---
type: experiment
node_id: exp:formal-training-none
title: "No formal multi-seed training evidence"
status: active
verdict: none
updated: 2026-07-15
---

# No Formal Multi-Seed Training Evidence

## Summary

As of 2026-07-15, there is no completed formal multi-seed training result for
PCEH, CESR, CRS-EPS, persistent event-set, Q2 fixed/rematch binding, LoRA, or
full visual tower routes. The current Q2 route has CPU/B0 engineering evidence
only; no fixed-step profile or effectiveness result is accepted.

## Current Pre-Training Block

- `P0-LAUNCH-WORKDIR` prevents an attested profile ticket from matching the
  dynamic Slurm `work_dir`.
- Cached-feature extractor provenance and real packet availability clocks are
  not closed for strict-online claims.
- The fixed/rematch one-factor trace and 211-versus-213 reporting population
  require external evidence completion.
- Multi-rank DDP and resume are not required for this single-GPU route and must
  remain disabled rather than silently supported.

## Consequence

Do not claim:

- SOTA;
- paper-ready;
- stable AP-latency improvement;
- full end-to-end online training;
- LoRA visual adaptation benefit.

## Required Before Paper Claim

- 3 paired seeds for main comparison;
- endpoint-only baseline;
- CESR/PCEH model;
- full chronological validation/test;
- confidence intervals or at least mean/std;
- cost and latency ledger.
