# EventMATR D1 five-epoch experiment audit

Date: 2026-07-29

Overall verdict: **WARN**

This audit distinguishes result integrity from scientific sufficiency. A fresh
zero-history reviewer independently read the code and gate documents. Attempts
to obtain an additional Claude and Gemini cross-model review failed because the
Claude interface returned no JSON and the Gemini credential was invalid.
Therefore reviewer independence is useful but not fully cross-model.

## Integrity checks

### Ground-truth provenance: PASS

The pilot launcher points to the official annotation input
(`scripts/train_eventmatr_d1_pilot.sh:24-31`). The evaluation wrapper passes
`args.video_anno` to the official-style detector evaluator (`eval.py:24-31`),
whose ground truth is loaded from that JSON
(`Evaluation/eval_detection_gentime.py:53-70`). No model output is used as
ground truth.

### Score normalization: PASS

No self-normalization by a prediction maximum or mean was found. The reported
train-prefix values use fixed temporal-overlap levels `0.3:0.7`. They remain
diagnostic because they are computed on the training split, not because their
denominator is fraudulent.

### Result existence and identity: PASS

Slurm array `1200955` has five `COMPLETED 0:0` tasks. All five pilot receipts,
terminal checkpoints, five-row metric files, and final source-identity receipts
exist. The receipts match commit
`1f4bb29ad58dddcc33f6ff2bdc57a5934ee5c53d`, tree
`aad757531cfcbfb79b616756466251f946574035`, and manifest SHA-256
`1aa1524364a82853fe67bc04fded600274414d31eae139aec06f9d0261aa8582`.
The locked-test sentinel remained absent.

### Metric operationalization: WARN

The manifest registers detection, recall, lifecycle, identity, calibration and
systems metrics (`experiment_configs/eventmatr_d1_preexperiments.json:85-113`).
The finalizer requires only common losses/mAP plus six D1 loss/count fields
(`scripts/finalize_eventmatr_d1_pilot.py:14-36`). It does not require birth/end
calibration, latency, closure, unresolved tracks, duplicate close,
reacquisition, owner switches, fragmentation, throughput or memory. A `PASS`
pilot receipt is therefore an execution receipt, not a scientific-effect
receipt.

### Scope: WARN

The evidence is one fixed seed, five training epochs, and train-prefix
diagnostics. It supports optimization and failure-mode claims only. It cannot
support generalization, multi-seed stability, locked-test superiority or a
paper-level novelty claim.

### Evaluation type

`real_gt_train_only_diagnostic`: real official training annotations are used,
but the result is deliberately marked
`strict_causal_paper_result_valid=false`.

## Threshold audit

- Native flag threshold: `0.5`
  (`scripts/train_eventmatr_d1_pilot.sh:51`).
- Native class-score threshold: `0.1` (line 52). It applies to the native writer
  (`util/utils.py:90-114`) and is bypassed by the EventMATR ledger writer
  (`util/utils.py:125-181`).
- Postprocessing suppression overlap: `0.3` (launcher line 53;
  `util/utils.py:219-280`).
- D1 birth/end logit thresholds: both `None`
  (`util/config.py:71-80`). Birth and end use state argmax; a numeric logit
  threshold is test-only (`models/event_memory.py:790-830,871-937`).
- Minimum ended interval: one positive frame
  (`models/event_memory.py:831-897`).

No threshold search or threshold lowering occurred. R/T terminal zero output
arises before the EventMATR ledger writer and is not caused by the native
class-score cutoff.

## Scientific-gate audit: WARN

The execution gate is precise and passed. The scientific questions and kill
criteria are sensible, but no effect-size, coverage, confidence, sample-size or
paired-comparison thresholds were prospectively frozen. The observed zero
output and severe coverage loss justify a reversible hold on longer training,
but they do not constitute failure under a preregistered quantitative science
gate.

## Required actions

1. Correct the record from “scientific gate failed” to “scientific sufficiency
   unresolved; diagnostic hold.”
2. Before replay, prospectively register complete metric availability,
   functional-liveness handling, effect-size/sample-size tolerances and
   matched-coverage comparisons.
3. Run source-exact strict-causal chronological replay with stage counts and all
   manifest metrics.
4. Keep thresholds fixed; inspect score distributions and threshold-free curves
   only for diagnosis, not operating-point selection.
5. Do not release longer horizons, locked test, multiple seeds or raw RGB until
   this audit is closed.
