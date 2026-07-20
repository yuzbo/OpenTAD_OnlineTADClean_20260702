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

The clean branch has passed a real-data Slurm smoke and strict paired profile.
Those are engineering/protocol results, not method-effectiveness evidence. The
profile estimates the registered 12-epoch pair at `12.572 GPU·hours`, above the
frozen `2 GPU·hour` cap. No seed-705 or three-seed result exists.

The 2026-07-20 readiness review is absorbed as `REVISE BEFORE SCIENTIFIC RUN`.
Read:

1. `PRO_ONTAD_FIXED_REMATCH_SCIENCE_READINESS_ABSORPTION_20260720.md`;
2. `experiments/ontad-science-fixed-rematch-readiness-review-20260720.md`;
3. DR-028;
4. `experiments/ontad-science-fixed-rematch-plan-20260720.md`.

## Current Blocking Gaps

1. **Endpoint identifiability:** binary commit still uses an endpoint offset
   with no offset loss.
2. **Birth-step fairness:** REMATCH may move newborn class/start/end losses
   away from the canonical birth slot on the same decision.
3. **Short actions:** newborn same-step end and final-token short actions can be
   lost.
4. **Adjacent actions:** entry-free slot freezing delays reuse after a
   same-step old end; this is not yet proven harmless.
5. **Reporting isolation:** generic training constructs and evaluates the
   locked reporting split every epoch.
6. **Instance metrics:** video/runtime keys and frame/second coordinates can
   mismatch; greedy matching and distinct-bounds fragmentation do not implement
   the frozen definition.
7. **Gate closure:** standard mAP, budgeted AP, gate field names, ranges, tIoU
   sets, and percentage-point units are not one schema.
8. **Provenance:** no repository-owned result artifact joins checkpoint,
   config, manifest, ledger, audit, metrics, latency, resource use, and hashes.
9. **Fail-closed execution:** readiness, canonical exhaustion, runtime
   collision, arbitration suppression, cancellation, and abandonment are not
   fully separated and enforced.
10. **Census:** birth-per-step, end+birth, final-token coverage, delayed reuse,
    and clipped starts are not a hashed launcher-consumed artifact.
11. **Budget:** the registered protocol is too expensive; budget revision must
    wait for repair and re-profiling.

## Evidence Already Established

- supervision state is separate from runtime state;
- predicted runtime occupancy cannot remove canonical GT birth supervision;
- the capacity/refractory failure behind 2,206 exhausted births is repaired;
- four slots exceed observed concurrency two, but fuller census is required;
- FIXED/REMATCH resolved configs differ only in binding mode and work directory;
- the mode affects training supervision, not the inference decoder;
- job `1176737` passed FP32 update, checkpoint/reload, streaming inference, and ledger consistency;
- job `1176983` passed 250-step stability, zero immediate capacity loss, and exact pre-training ledger equivalence;
- both untrained arms emitted 10,029 causal rows with identical digest and zero protocol violations;
- none of this shows FIXED improves duplicate, fragmentation, mAP, or latency.

## Immediate Implementation Order

1. Make binary endpoint equal the current decision frame.
2. Keep newborns on canonical slots for the birth step; allow REMATCH next step.
3. Commit same-step birth+end exactly once, including final-token actions.
4. Add end-to-end adjacent/short/repeated/overlapping same-class counterexamples.
5. Implement fit-only train, symmetric calibration/checkpoint freeze, and
   report-once evaluation.
6. Normalize video identity and metric coordinates.
7. Freeze global one-to-one matching and disjoint-component fragmentation.
8. Freeze standard-mAP percentage-point gate schema.
9. Emit one hash-linked formal result artifact and split all capacity counters.
10. Generate and consume the exact frozen census.
11. Rerun focused tests, Slurm smoke, and strict paired profiling.
12. Only then register a budget-compatible seed-705 screen.

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
