---
type: experiment
node_id: exp:ontad-fixed-rematch-readiness-review-20260720
title: "FIXED/REMATCH Scientific Readiness Review and Independent Absorption"
date: 2026-07-20
status: revise-before-scientific-run
input: fixed_cached_causal_features
code_anchor_reviewed: 27a59dec445f6b4ed9651abab1168c358a8db7e3
code_anchor_rechecked: 95fa963e7e2f3a05779907999df7f9311be4f4fd
---

# FIXED/REMATCH Scientific Readiness Review

## Question

Can the current feature-level FIXED/REMATCH route move from engineering smoke
and profiling into a scientifically interpretable seed-705 screen?

## Verdict

No. The route remains **REVISE BEFORE SCIENTIFIC RUN**.

The later Slurm evidence improves engineering confidence but does not close the
scientific contract:

- job `1176737` proves FP32 update, checkpoint/reload, streaming inference, and
  ledger execution;
- job `1176983` proves 250-step stability, zero immediate capacity loss, and
  exact pre-training FIXED/REMATCH inference equivalence;
- neither job tests whether FIXED improves effectiveness;
- the registered 12-epoch pair fails the frozen budget gate at
  `12.572 GPU·hours`.

## Independent Disposition

| Priority | Accepted | Partly accepted/resolved | Fully resolved |
|---|---:|---:|---:|
| P0 | 8 | 1 | 0 |
| P1 | 6 | 4 | 0 |
| P2 | 4 | 0 | 1 |

The P2 profiler gap is the only review item fully closed. The ledger filename,
optimizer construction, smoke execution, FP32 numerical policy, and explicit
smoke checkpoint handling are partially or operationally resolved, but the
formal experiment contract remains open.

## Blocking Repairs

1. make binary endpoint equal the current decision frame;
2. keep newborns on canonical birth slots for their birth step;
3. support same-step birth+end and final-token short-action commit;
4. isolate fit, calibration, and one-shot locked reporting;
5. normalize video identity and coordinates for instance metrics;
6. freeze global matching, disjoint-fragmentation, standard-mAP units, and gate
   schema;
7. emit a repository-owned, provenance-linked result artifact;
8. make formal readiness, counters, and census fail-closed.

After repair, rerun focused counterexamples, Slurm smoke, and the strict paired
profile. Only then may a new budget protocol be registered.

## Sources

- [`../../PRO_ONTAD_FIXED_REMATCH_SCIENCE_READINESS_REVIEW_20260720.md`](../../PRO_ONTAD_FIXED_REMATCH_SCIENCE_READINESS_REVIEW_20260720.md)
- [`../../PRO_ONTAD_FIXED_REMATCH_SCIENCE_READINESS_ABSORPTION_20260720.md`](../../PRO_ONTAD_FIXED_REMATCH_SCIENCE_READINESS_ABSORPTION_20260720.md)
- [ontad-science-fixed-rematch-smoke-20260720.md](ontad-science-fixed-rematch-smoke-20260720.md)
- [ontad-science-fixed-rematch-profile-20260720.md](ontad-science-fixed-rematch-profile-20260720.md)
- [ontad-science-fixed-rematch-plan-20260720.md](ontad-science-fixed-rematch-plan-20260720.md)

## Next Action

Implement the scientific-contract repair. Do not request another high-cost
review, submit seed 705, or begin raw-RGB work first.
