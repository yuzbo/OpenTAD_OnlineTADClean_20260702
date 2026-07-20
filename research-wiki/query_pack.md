---
type: query_pack
updated: 2026-07-21
status: active
scope: Compressed memory to prepend before any new ideation or implementation planning.
---

# Query Pack: Online/Causal TAD Project Memory

## Project Direction

The task is fixed to standard, fully supervised, strictly causal Online
Temporal Action Detection/Localization. At decision time `t`, inference sees
only current and past video evidence, maintains action-instance
start/ongoing/end state, and emits one immutable final
`{start, end, class, score}` interval with low delay. Do not change the task,
add new labels/sensors, expose mutable outputs, or introduce offline cleanup.

The active experiment is feature-level, not raw RGB: compare first-crossing
**FIXED** supervision binding against per-prefix **REMATCH** while runtime
lifecycle, data, thresholds, optimizer, inference, and evaluation are matched.
Only the post-birth loss-binding rule may differ.

The optimized route passed same-commit Slurm smoke `1177580` and strict
profile `1177582` at `534f85b`. The first seed-705 paired technical screen
`1177596` then completed within budget but correctly failed: both FIXED and
REMATCH made zero calibration emissions after one stable epoch. No three-seed
or paper result exists.

Calibration-only diagnosis `1177634` then confirmed the common bottleneck:
across 28,730 tokens per arm, FIXED/REMATCH birth maxima were
`0.433135/0.346657` with zero 0.5 crossings. Both trained birth biases remained
about `2.526 logit` below the registered weighted-BCE stationary point. The
shared `weighted_bce_stationary` initialization repair is implemented; its
same-commit smoke `1177637` and one-epoch profile gate `1177639` passed.
Repaired screen `1177653` at `8dff64c` used `0.955278 GPU-hours`; both arms
again made zero emissions after 2010/2010 stable updates, so the frozen gate
failed. Calibration-only diagnosis `1177682` is now selecting the next single
shared change without threshold/reporting access. A short-warmup candidate is
locally implemented but remains untrained until that diagnosis.

The 2026-07-20 readiness review is absorbed as `REVISE BEFORE SCIENTIFIC RUN`.
Read:

1. `PRO_ONTAD_FIXED_REMATCH_SCIENCE_READINESS_ABSORPTION_20260720.md`;
2. `experiments/ontad-science-fixed-rematch-readiness-review-20260720.md`;
3. DR-028;
4. `experiments/ontad-science-fixed-rematch-plan-20260720.md`.

## Current Blocking Gaps

1. **Shared model/optimization bottleneck:** initialization repair did not
   remove silence; diagnose score margins before one shared change.
2. **Paper evidence:** one seed/one epoch is not the three-seed, multi-epoch
   main result and cannot prove the identity-error claim.
3. **Raw RGB:** joint visual training stays blocked until the complete
   feature-level scientific gate passes.

## Evidence Already Established

- full census: 411 videos, 320,205 tokens, 6,328 instances, every birth/end
  covered, max two births and four visible instances, zero capacity deficit;
- repaired smoke `1177438`: 76 tests, one real update, exact checkpoint reload,
  84 immutable emissions, zero future-information violations;
- profile `1177511`: both arms stable and exactly equivalent before training,
  but the 12-epoch pair costs `12.512897 GPU·hours` and is rejected;
- fit-only balance census: birth/alive/end positive rates are
  `0.00550069/0.0799036/0.0636912`; every birth start lies within one token;
- commit `0258b85` implements the shared model optimization, adjacent-action
  test, one-epoch screen, calibration-only result gate, and hash provenance;
- optimized smoke `1177580` passed 82 remote tests and exact causal reload;
- optimized one-epoch profile gate is `1.404134 GPU-hours`, below the cap;
- screen `1177596` used `0.871944 GPU-hours`; both arms completed 2010/2010
  updates with zero training/capacity/causal errors but emitted zero intervals;
- diagnosis `1177634` passed 22 tests and found zero birth-threshold crossings
  in either arm; trained birth biases were about `2.526 logit` below the
  weighted-BCE stationary initialization;
- the shared repair changes only binary-head initialization mode; empirical
  priors, positive weights, thresholds, data, lifecycle, and comparison axis
  remain frozen;
- repaired smoke `1177637` passed 87 tests and exact causal reload; profile
  `1177639` passed the one-epoch gate at `1.419566 GPU-hours`;
- repaired screen `1177653` used `0.955278 GPU-hours`; both arms had 2010/2010
  clean updates but zero intervals, so initialization alone was insufficient;
- none of this yet shows FIXED improves duplicate rate, fragmentation, mAP,
  latency, or any paper headline.

## Immediate Implementation Order

1. Complete score diagnosis `1177682` on both repaired checkpoints.
2. Confirm or reject the prepared shared short-warmup candidate.
3. Rerun same-commit smoke, strict profile, and seed-705 technical screen.
4. Only after a technical pass, design the affordable multi-epoch/three-seed
   protocol; raw RGB remains later and conditional.

## Feature-Level Gates

Every arm and seed must have:

- zero causal violations;
- zero dropped GT birth targets;
- zero unexplained runtime capacity failures;
- nonzero committed predictions;
- prediction/GT ratio in `[0.25, 4.0]`;
- Recall@tIoU 0.3 of at least `0.25`;
- explicit successful-update and scheduler-step parity.

Across paired seeds 705/706/707:

- `E_id = 0.5 × (duplicate_rate + fragmentation_rate)`;
- FIXED relative `E_id` reduction at least 20%;
- FIXED improves at least two of three seeds;
- standard average mAP decline no worse than `0.5` percentage points;
- report absolute delta, components, paired video bootstrap uncertainty,
  standard mAP, budgeted AP, GT-end latency, resource use, and failure subsets.

These are falsification/resource gates, not universal significance theorems.

## Closest Prior Work and Claim Limits

- **CAG-QIL/SimOn:** direct future-free On-TAL grouping or current-query/past-context instance prediction.
- **MATR:** current segment estimates end and memory estimates start.
- **ActionSwitch:** direct threat for overlap and repeated same-class actions.
- **TrackFormer/MOTR:** persistent query identity and assignment precedent;
  blocks generic “tracking queries over time” novelty.
- **E2E-LOAD/StreamFormer:** raw-video causal OAD/backbone precedents.
- **Offline E2E-TAD/PEFT:** later raw-RGB context, not current evidence.

Do not claim that persistence, memory, lifecycle state, causal attention,
end-to-end terminology, frozen features, LoRA, or a raw-video backbone alone is
novel. The surviving claim is narrow: under a matched strict On-TAD protocol,
does first-crossing persistent supervision binding reduce identity-linked
duplicate/fragmentation error without unacceptable standard-mAP loss?

## Failed / Blocked Routes

- PIVOT and other sensor/observability tasks: rejected because they leave
  standard On-TAD.
- PCEH/CESR: demoted to causal infrastructure/components.
- Full PETAL raw-video package first: demoted after reconstruction by prior work.
- FRESH/TTF/PES Stage 1 as-is: historical mechanism audit, superseded as the
  immediate executable route by the cleaner FIXED/REMATCH single-axis study.
- Full-packet, visual-tower-first, zero-shot, adaptive selection, and
  distillation headline: blocked or support-only.
- Silent or nearly silent birth control as a capacity fix: invalid.
- Changing epochs, split size, slots, thresholds, or budget after seeing
  reporting results: prohibited.

## Current Final Goal

Validate the evidence-backed birth/lifecycle repair now. No new Pro discussion
is required. Do not lower thresholds, access reporting, run three seeds, or
begin raw-RGB training first. Repeat smoke/profile/seed-705 at one exact
commit; only a technical pass can advance to the paired multi-seed feature
falsification. Raw-RGB remains conditional on the complete feature-level
technical and scientific gates.
