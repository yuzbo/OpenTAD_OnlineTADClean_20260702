---
type: experiment
node_id: exp:persistent-feature-kill-test-20260712
title: "Persistent Event-Set Stage-1 Matched-Feature Kill Test"
status: cache-submitted
updated: 2026-07-12
---

# Persistent Event-Set Stage-1

## Scientific Question

Under identical frozen causal features, decoder capacity, optimizer, training schedule, thresholds, and evaluation, does persistent instance state improve standard completion-triggered On-TAD beyond both fresh queries and a Temporal TrackFormer reconstruction?

This is a falsification experiment, not a paper claim. Full raw-video PETAL is not approved.

## Registered Variants

| Variant | Assignment | Query state | Start | Endpoint |
|---|---|---|---|---|
| FRESH | per-prefix | reset every step | scalar offset | binary |
| TTF | prefix birth then fixed identity | persistent | scalar offset | binary |
| PES | prefix birth then fixed identity | persistent | memory pointer | first-event hazard |

All variants use the same SigLIP2 frame-feature cache, four event slots, hidden width 256, memory 192, chronological chunks of 64 tokens, optimizer, scheduler, seeds, and `OnlineAPBudgeted` evaluator. No variant uses raw-video gradients, offline NMS, future endpoint prediction, or a duplicate-repair loss.

## Implemented Protocol Controls

- training-only prefix-observable instance schedule;
- first endpoint crossing appears exactly once;
- independent IDs for concurrent or repeated same-class instances;
- Hungarian birth/visibility assignment;
- no EOF, total-duration, or future-frame metadata in prediction computation;
- unknown detector kwargs rejected as potential GT or terminal taint;
- immutable one-time completion ledger with `predicted_end <= emit_time`;
- chunk/step and future-perturbation equivalence tests;
- slot-exhaustion and same-class-concurrency audits;
- mmap feature dataset, atomic resumable cache files, manifest hashes, and exact source timestamps;
- per-video resume sidecars that bind encoder, stride, dtype, source frames, and feature hash;
- scalar-start train/inference unit consistency in feature-step coordinates;
- active-slot false-birth release and persisted slot-exhaustion audit counter;
- ledger-derived OnlineAP, duplicate, and fragmentation diagnostics plus a machine-readable three-variant result gate;
- unique variant/seed pilot registry and enforced nine-run share of the 10 GPU-hour budget.

## THUMOS14 Instance Audit

Audit run on N16R4 against `thumos_14_anno.json`:

| Split | Videos | Instances | Max concurrency | Max same-class concurrency | Videos with overlap | Same-class overlap videos |
|---|---:|---:|---:|---:|---:|---:|
| training | 200 | 3,003 | 2 | 2 | 23 | 2 |
| validation | 211 | 3,325 | 2 | 2 | 32 | 3 |

The previous `K=16` assumption had no support in these labels. Stage 1 uses four slots: observed maximum two plus two audit-margin slots. Slot exhaustion remains a hard invalidation condition.

Duration audit:

| Split | Mean | p50 | p90 | p95 | Max |
|---|---:|---:|---:|---:|---:|
| training | 4.04 s | 2.90 s | 8.00 s | 9.70 s | 118.10 s |
| validation | 4.45 s | 3.20 s | 9.10 s | 11.48 s | 86.60 s |

## Verification Ledger

Local CPU-safe tests:

```text
30 passed
```

Remote N16R4 combined tests after Hungarian and taint-audit repair:

```text
43 passed in 35.63s
```

Synthetic end-to-end integration on N16R4 CPU also passed for one real config-built training step, one chronological inference chunk, and ledger-to-run-summary generation. The training audit observed finite losses, nonzero gradients in 30 head parameters, 30 changed parameters, one first-crossing endpoint positive, and zero slot exhaustion.

The local Windows user-site Torch currently fails to load `c10.dll`; Torch tests are therefore executed in the existing N16R4 OpenTAD environment. This is an environment limitation, not counted as a model pass.

## Registered Gate

Three variants must use matched seeds `705/706/707`. The result is invalid if protocol violations, slot exhaustion, or total Stage-1 training above 10 GPU-hours occurs.

PES retains the mechanism only if it beats **each** of FRESH and TTF by either:

- at least 2.0 average-mOnlineAP points; or
- at least 20% mean duplicate/fragmentation error reduction while remaining within 0.5 mOnlineAP points.

A pass still blocks raw-video training until five-seed confirmation, paired uncertainty, and a renewed novelty review. These thresholds are project resource rules, not universal significance claims.

## Current State

- Implementation: complete for the registered Stage-1 scope.
- GitHub commit: `d06d0e992713ace3cf61f7e23c55bfc1a8afbb51` on `codex/online-tad-clean-20260702`.
- Clean N16R4 worktree: `/data/run01/sczc063/yuzibo/projects/OpenTAD_PES_Stage1_d06d0e9_20260712`.
- Cache extraction: Slurm job `1159510`, submitted 2026-07-12, currently `PENDING` at the first status check.
- Cache run directory: `/data/run01/sczc063/yuzibo/runs/pes_stage1/pes_cache_20260712_033905`.
- Cache target: `/data/run01/sczc063/yuzibo/thumos14/features/pes_siglip2_stride8`.
- GPU smoke: blocked on a complete cache manifest.
- Three-seed pilot: not submitted; smoke gate must pass first.
- Raw-video/PEFT route: blocked.

## Artifacts

- Design: [persistent-feature-kill-test-design-20260712.md](persistent-feature-kill-test-design-20260712.md)
- Plan: [persistent-feature-kill-test-plan-20260712.md](persistent-feature-kill-test-plan-20260712.md)
- Pro absorption: [`../../PRO_PETAL_DEEP_REVIEW_ABSORPTION_20260712.md`](../../PRO_PETAL_DEEP_REVIEW_ABSORPTION_20260712.md)
- Run summary: `tools/summarize_pes_stage1_run.py`
- Result gate: `tools/check_pes_stage1_results.py`
- Instance audit: `tools/analyze_ontad_instances.py`
