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

- [x] C1 — core lifecycle and supervision repairs implemented with focused
      regression tests.
- [x] C2 — metrics, gate, fit/report isolation, provenance, and census
      contracts implemented; all local CPU-safe checks pass.
- [x] C3 — repaired code committed and pushed; N16R4 Slurm end-to-end smoke
      passes from the exact commit.
- [x] C4 — strict paired profile rerun; budget-compatible protocol either
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

## Repaired End-to-End Smoke — Slurm 1177438

C1 and C3 passed after the birth-only start-supervision correction.

- Exact code commit:
  `9036bd21838596543f87d705fe6daa06a49e0470`, pushed on
  `codex/ontad-science-fixed-rematch`.
- Run directory:
  `/data/run01/sczc063/yuzibo/runs/persistent_binding/smoke_20260720_221637`.
- Submission artifact: `job.sbatch` in that directory. It pins the commit,
  requires a clean checkout, runs the full census, the focused test bundle,
  direct real-feature FIXED and REMATCH train steps, one standard
  `tools/train.py --allow-unready-smoke` update, checkpoint reload inference,
  and `tools/verify_persistent_binding_smoke.py`.
- Allocation: one RTX 4090 on `g0003`; `nvidia-smi` passed inside the
  allocation.
- Slurm result: `COMPLETED 0:0` in `00:03:47`.
- Remote focused tests: `76 passed` in `100.40 s`.

Full frozen-data census:

- 411 videos, 320,205 causal feature tokens, and 6,328 instances;
- 6,328/6,328 births/endpoints covered;
- maximum two births per decision and maximum four visible instances;
- zero GT entry-free deficit and zero oracle capacity overflow;
- 208 old-end/new-birth decisions containing 263 pairs;
- zero same-decision birth+end instances at the registered stride;
- no census failures, including zero clipped *birth* start targets;
- artifact SHA-256:
  `10cd30f02a4f26ed1c8877e222c4437f9d91ce2d34d30be56c0c4a2a140b8184`.

Training and lifecycle evidence:

- both direct real-feature FIXED and REMATCH forward/backward/update steps
  passed with zero GT supervision exhaustion and zero runtime entry-free
  collision;
- the standard training path recorded exactly one expected update, one
  successful update, one scheduler step, and zero skipped updates;
- lifecycle totals were one arbitration suppression, five candidate
  cancellations, zero active abandonments, and 17 births deferred until a
  released slot became eligible;
- training-audit SHA-256:
  `7c2d4ac90a642479fef70d1decaa778348921557d54c6fcc8e5e8dfc92656c24`.

Checkpoint and strict-causal inference evidence:

- strict determinism was active with warning-only disabled, flash and
  memory-efficient SDP disabled, and math SDP enabled;
- 29 state tensors changed, 29 optimizer-state entries were created, and the
  total state delta L2 was `0.6226117403130047`;
- checkpoint SHA-256:
  `3a7ff1e952d1821c4a2bd2f9f03c41ab6893dc7f585e9b1ab1769b1c10609a6d`;
- train-time and reload inference ledgers were byte-identical, each with
  SHA-256
  `796a526159ab8f2ed94502b4b408f73a6c1042e161f4dced5950b9722b6ce2ce`;
- all 84 immutable final emissions carried separate video/runtime identities,
  ended and emitted on the current causal decision frame, and had zero future
  endpoint/source, negative-latency, or monotonicity violations;
- the joined smoke gate passed; its SHA-256 is
  `a27d91f046f89dfe9e183511dbfa05d968393a327f178b2680a38fb2928bdaa7`.

The zero AP values from one update on one held-out video are deliberately not
interpreted as effectiveness evidence. This smoke proves only that the
scientific contracts, gradients, checkpointing, strict-causal final emission,
reload determinism, and evaluator wiring work end to end. The next authorized
node is C4: rerun the paired resource profile at this exact repaired commit
before deciding whether any single-seed screening run is affordable.

## Repaired Strict Profile and Screen Registration — Slurm 1177511

C4 is complete. The repaired implementation remains technically stable, the
registered 12-epoch pair remains over budget, and a separate one-epoch
seed-705 *technical screen* fits the unchanged two-GPU-hour cap.

- Exact code commit:
  `caa42b682e53dbbc51b33c940e6c9a6222f91585`.
- Run directory:
  `/data/run01/sczc063/yuzibo/runs/persistent_binding/profile_20260720_222625`.
- Allocation: one RTX 4090 on `g0048`.
- Frozen measurement: seed 705; 50 warm-up and 200 measured chronological
  chunks per arm and mode; FP32; strict deterministic math SDP.
- Pre-profile tests: `19 passed`.
- Slurm state: `FAILED 1:0` in `00:10:12` only because the final frozen
  12-epoch budget evaluator deliberately exits nonzero when the cap fails.
  All four profile measurements completed and passed their stability,
  update, capacity, causality, and equivalence checks before that exit.

Measured training profile:

| Arm | Mean step (s) | Full fit pass (GPU h) | Peak MiB |
| --- | ---: | ---: | ---: |
| FIXED | 0.707804 | 0.395191 | 140.520 |
| REMATCH | 0.701788 | 0.391832 | 140.520 |

Both arms had zero GT supervision exhaustion and zero GT-birth/runtime
entry-free collision. Their measured lifecycle counts also agreed exactly.

Measured untrained calibration profile:

- FIXED and REMATCH had byte-identical canonical ledgers with SHA-256
  `dd75937a4096f03966bb49ab7e432cd4820f8b74d052906d9719d934bacf4e1a`;
- each measured prefix produced 17,014 emissions across 15 reached videos;
- all future-end, future-source, negative-latency, and non-monotonic-emission
  counts were zero;
- the large untrained emission count is a diagnostic baseline, not an
  effectiveness result.

Frozen 12-epoch budget result:

- training: `9.444269 GPU h`;
- calibration inference: `0.083274 GPU h`;
- reserved locked-report inference: `0.482775 GPU h`;
- raw total: `10.010318 GPU h`;
- total with the frozen 1.25 safety factor: `12.512897 GPU h`;
- gate: **FAIL**, versus the unchanged `2 GPU h` cap.

Registered seed-705 technical-screen budget:

- exactly one full epoch over all 160 fit-core videos for each arm;
- retain the exact first epoch of the 12-epoch optimizer/scheduler contract;
- evaluate only on the 40-video calibration split during this screen;
- do not access the 211-video locked reporting split;
- keep a conservative budget reserve for the eventual locked report anyway;
- estimated total with that reserve and the 1.25 factor:
  `1.691339 GPU h`;
- budget gate: **PASS**, versus `2 GPU h`.

The one-epoch run is authorized only as a convergence/non-degeneracy screen.
It may prove that both arms train, emit causally, avoid silent/explosive output
under the frozen technical thresholds, and produce complete calibration
artifacts. It cannot establish the 20% identity-error claim, three-seed
consistency, paper effectiveness, or permission for raw-RGB training.

Profile artifact SHA-256:

- FIXED train:
  `19fde8876177de4776ceb37e5c7feec49fe1aab3038a34aea76ae5b9a46132ef`;
- REMATCH train:
  `9cd2d3f22cfe58f35afe1c95583244279ea8fade1c32440ca59a1b47d0c70ed8`;
- FIXED calibration inference:
  `bcec50329929493ca75f5d6d0ea36e43d63ad07ea88315fc635490d553442812`;
- REMATCH calibration inference:
  `cbe71d216e43f05c0f8e1691155a3bbf11888e2c46712c5d7f7febeeb65b1278`;
- 12-epoch rejection gate:
  `680b3e8e347f7b65aac58e27ab306081adfcb7a182675b602cfea064d8cf303d`;
- one-epoch screen budget gate:
  `4f212c5ef823ea973283b962e7293c636093eaff9584373f96893c24dbfbc2f3`.

No new Pro discussion is needed before this screen: DR-028 already authorizes
registration of a budget-compatible one-seed screen after repaired smoke and
profiling. Before submission, the repository must add a screen-only config,
launcher, calibration-only result builder/gate, and the still-missing
adjacent-action end-to-end regression. C5 remains open until those checks and
the seed-705 screen finish.
