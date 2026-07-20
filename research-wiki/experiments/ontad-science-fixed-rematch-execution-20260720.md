---
id: exp:ontad-fixed-rematch-execution-20260720
type: experiment
status: active
updated: 2026-07-20
---

# FIXED/REMATCH Scientific-Contract Repair Execution

This page is the recovery checkpoint for the active implementation and
experiment task. Update it at every critical node before continuing.

## Frozen Objective

- Task: standard, fully supervised, strictly causal Online Temporal Action
  Detection.
- Input at this stage: cached causal video features, not raw RGB.
- Output contract: maintain instance birth, continuation, and end online, then
  emit one immutable final interval using only current and past evidence.
- Scientific comparison: FIXED versus REMATCH differs only in the post-birth
  target-to-slot loss binding.
- Raw RGB remains blocked until the feature-level technical and scientific
  gates pass.

## Recovery Checkpoint — 2026-07-20

- Branch: `codex/ontad-science-fixed-rematch`.
- Repository: `https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702`.
- Draft PR: `https://github.com/yuzbo/OpenTAD_OnlineTADClean_20260702/pull/1`.
- Recovered HEAD: `9ce8aa0bfe4160350593c20349b2ff0ea8e39975`.
- Worktree was clean and matched
  `origin/codex/ontad-science-fixed-rematch`.
- Last engineering smoke: Slurm job `1176737`, passed at commit `097bc72`.
- Last strict paired profile: Slurm job `1176983`, technical/causal checks
  passed at commit `95fa963`; the registered 12-epoch pair was rejected at
  `12.572 GPU-hours` versus the frozen `2 GPU-hour` cap.
- No seed-705 training and no raw-RGB training are authorized by the current
  evidence.

## Active Repair Contract

1. Binary endpoint emission is the current causal decision frame; no
   unsupervised trainable endpoint offset.
2. A newborn remains on its canonical birth slot for the birth decision;
   REMATCH begins on the next causal decision.
3. Birth and end crossing in one step emits a short action exactly once.
4. Supervision exhaustion, runtime entry-free collision, arbitration,
   cancellation, abandonment, and released-slot deferral have separate
   counters; formal supervision exhaustion fails atomically.
5. Fit, calibration/checkpoint choice, and locked reporting are isolated.
6. Standard mAP uses a frozen percentage-point schema; instance matching uses
   video identity, one frame coordinate system, deterministic global
   one-to-one matching, and disjoint-component fragmentation.
7. A repository-owned artifact joins metrics, audits, resource use, code and
   input SHA-256 provenance.
8. A hashed split census is consumed before formal launch.

## Required Next Checkpoints

- [ ] C1 — core lifecycle and supervision repairs implemented with focused
      regression tests.
- [x] C2 — metrics, gate, fit/report isolation, provenance, and census
      contracts implemented; all local CPU-safe checks pass.
- [ ] C3 — repaired code committed and pushed; N16R4 Slurm end-to-end smoke
      passes from the exact commit.
- [ ] C4 — strict paired profile rerun; budget-compatible protocol either
      passes the frozen cap or remains explicitly blocked.
- [ ] C5 — final decision and next paper-experiment route recorded; seed-705
      is submitted only if every preceding gate passes.

## Context-Safety Rule

Before moving past C1–C5, append the exact commit, commands/tests, Slurm job
IDs, artifact paths, hashes, pass/fail result, and scientific interpretation
to this page and `research-wiki/log.md`.

## C1/C2 Implementation Checkpoint — 2026-07-20

Core implementation is complete locally; C1 remains unchecked until the
PyTorch regressions run on N16R4:

- binary endpoint emission is exactly the decision frame and the binary route
  has no trainable endpoint-offset head;
- newborn REMATCH is prohibited on its birth decision;
- newborn birth+end commits once in the same step;
- active abandonment and released-slot birth deferral are separately counted;
- formal supervision exhaustion raises against a cloned supervision state.

C2 is complete locally:

- `online_instance_metrics.v2` separates `video_id` from
  `runtime_stream_key`, converts raw database seconds to frames, uses
  deterministic global maximum-cardinality/maximum-total-tIoU matching, and
  counts disjoint covered components for fragmentation;
- `persistent_binding_gate.v2` requires standard-mAP percentage points,
  frozen tIoUs, update/counter integrity, and code/input SHA-256 provenance;
- fit-only training cannot construct the reporting loader; unready execution
  is limited to an explicit smoke config; calibration and reporting roles
  require an explicit checkpoint;
- calibration candidate selection and final reporting both create one-shot
  receipts, with a retained lock on reporting failure or success;
- the repository-owned result builder joins standard mAP, budgeted OnlineAP,
  instance metrics, latency, training audit, resource use, census, receipts,
  and provenance;
- the launcher-consumed `persistent_binding_split_census.v1` checks the frozen
  160/40/211 split, birth/visibility capacity, endpoint coverage,
  old-end/new-birth bins, final-token short actions, delayed reuse, and clipped
  start supervision.

Local evidence:

- `python -m py_compile` passed for all modified Python entrypoints;
- 54 unique CPU-safe focused tests passed, including 16 supervision tests, 12
  instance-metric tests, seven result-gate tests, config/launcher tests, and
  four new isolation/census/one-shot tool tests;
- `git diff --check` passed;
- Windows PyTorch execution remains unavailable because of the previously
  recorded `c10.dll` initialization failure, so head/detector/gradient tests
  are queued inside the N16R4 Slurm smoke; four local Torch-dependent tests
  were skipped by their explicit environment probe.

## Census-Gated Smoke Attempt — Slurm 1177416

- Code commit: `71914470cf7cbabb429fb6360c552d98230665ac`.
- Run directory:
  `/data/run01/sczc063/yuzibo/runs/persistent_binding/smoke_20260720_221248`.
- Allocation: RTX 4090 on `g0003`; the in-allocation `nvidia-smi` preflight
  passed.
- Slurm result: `FAILED 2:0` after 14 seconds, before tests or training.
- Gate behavior: correct fail-closed census rejection; no training budget was
  consumed.

Frozen full-data census evidence:

- 411 videos, 320,205 causal feature tokens, and 6,328 action instances;
- all 6,328 births and endpoints are covered;
- maximum births per decision is 2 and maximum visible concurrency is 4,
  exactly within the registered budgets;
- zero GT entry-free deficits and zero oracle capacity overflow;
- 208 old-end/new-birth decisions (263 pairs), so the adjacent-action path is
  materially present in the real data;
- zero same-decision birth+end instances at stride 8;
- only failure: 820 clipped start targets versus a frozen limit of zero.

Diagnosis and correction:

- the detector decoded and stored start only at birth, but incorrectly applied
  start-offset regression again at every later active prefix;
- those 820 targets were therefore loss-only artifacts after the action had
  outlived the 192-token memory horizon, not runtime birth-start failures;
- start loss is now restricted to the shared canonical birth assignment, and
  the census audits only actual birth-start targets;
- this removes a nuisance post-birth loss from both arms and makes the
  FIXED/REMATCH comparison more tightly controlled.

The failed job is retained as negative evidence. A new commit and smoke job
must show zero clipped birth-start targets before C1 or C3 can pass.
