---
type: query_pack
updated: 2026-07-20
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

The repair branch passed an earlier real-data Slurm smoke and strict paired
profile. Shared learnability optimization is implemented at
`0258b853aa284f9650d931e680116b63b289e192`: fit-only event priors, tempered
positive weights, a one-token birth-start range, supervised REMATCH costs, and
removal of unused scalar-route pointer work. The old profile is now only a
pre-optimization bound; no seed-705 or three-seed result exists.

The 2026-07-20 readiness review is absorbed as `REVISE BEFORE SCIENTIFIC RUN`.
Read:

1. `PRO_ONTAD_FIXED_REMATCH_SCIENCE_READINESS_ABSORPTION_20260720.md`;
2. `experiments/ontad-science-fixed-rematch-readiness-review-20260720.md`;
3. DR-028;
4. `experiments/ontad-science-fixed-rematch-plan-20260720.md`.

## Current Blocking Gaps

1. **Same-commit Torch validation:** Windows cannot load PyTorch `c10.dll`;
   the new adjacent-action end-to-end regression and full focused bundle must
   pass inside N16R4 Slurm.
2. **Re-profile after model change:** old job `1177511` cannot authorize the
   optimized commit; rerun smoke and strict profile without changing code.
3. **One-epoch non-degeneracy:** seed-705 FIXED/REMATCH must both update,
   emit causally, and avoid silent/explosive outputs on calibration only.
4. **Paper evidence:** one seed/one epoch is not the three-seed, multi-epoch
   main result and cannot prove the identity-error claim.
5. **Raw RGB:** joint visual training stays blocked until the complete
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
- 21 current CPU-safe tests plus compilation and Bash syntax pass locally;
- none of this yet shows FIXED improves duplicate rate, fragmentation, mAP,
  latency, or any paper headline.

## Immediate Implementation Order

1. Commit this wiki checkpoint and run the optimized code through N16R4 smoke.
2. At the exact same commit, rerun the paired 50-warmup/200-measured profile.
3. If its one-epoch estimate remains within two GPU-hours, run seed 705 for
   FIXED and REMATCH sequentially on one RTX 4090.
4. Apply the frozen calibration-only non-degeneracy gate without threshold
   edits and write all job IDs, hashes, metrics, and failures back to the wiki.
5. Only after a pass, design the affordable multi-epoch/three-seed feature
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

Implement the scientific-contract repair now. Do not start a new Pro discussion,
submit seed 705, run three seeds, or begin raw-RGB training first.

After repair, smoke and re-profile. If the repaired protocol is scientifically
closed and a new budget is explicitly registered, run one matched feature seed
as a non-degeneracy screen, freeze shared calibration rules, and then run the
paired three-seed falsification. Raw-RGB frozen/PEFT/joint training is permitted
only after both feature-level technical and scientific gates pass.
