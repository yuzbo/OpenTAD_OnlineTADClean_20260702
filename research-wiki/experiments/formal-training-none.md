---
type: experiment
node_id: exp:formal-training-none
title: "No formal multi-seed training evidence"
status: active
verdict: none
updated: 2026-07-16
---

# No Formal Multi-Seed Training Evidence

## Summary

As of 2026-07-16, there is no completed formal multi-seed training result for
PCEH, CESR, CRS-EPS, persistent event-set, Q2 fixed/rematch binding, LoRA, or
full visual tower routes. The current implementation has CPU/B0 engineering
evidence and a signed negative G0 fidelity result; no fixed-step profile or
effectiveness result is accepted.

## Current Pre-Training Block

- Commit `70df86e` passed signed local and N16R4/Linux B0 at `588/588`, and the
  same locked reviewer returned `PASS / PROFILE=ALLOW / NEXT_GATE=G0`.
- The preregistered real-data G0 four-arm replay-fidelity audit then returned a
  signed terminal `KILL`: three of four selected samples violated absolute
  loss, gradient, and/or runtime-state fidelity. Current dynamic replay and
  provisional `M=4` are therefore not authorized as primary training.
- The four exposed G0 samples are development-only. Any replacement route
  requires a new holdout preregistration, clean commit, B0, and review.
- Cached-feature extractor provenance and real packet availability clocks are
  not closed for strict-online claims.
- The fixed/rematch one-factor trace and 211-versus-213 reporting population
  require external evidence completion.
- Multi-rank DDP and resume are not required for this single-GPU route and must
  remain disabled rather than silently supported.
- The launch validator rejects the signed G0 `KILL`; no fixed-step profile or
  formal launch ticket is authorized.

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
