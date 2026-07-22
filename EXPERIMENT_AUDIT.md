# Experiment Audit Report

**Date**: 2026-07-22  
**Auditor**: same-family Codex `gpt-5.6-sol`, xhigh, read-only  
**Reconciliation**: primary protocol audit against frozen configs and Wiki  
**Project**: strictly causal feature-level On-TAD formal12 FIXED/REMATCH  
**Integrity verdict**: **PASS for the preregistered epoch-12 gate**  
**Operational verdict**: **FAIL; model optimization required**

The configured cross-family reviewer was unavailable, so this report is not a
cross-family acquittal. The initial same-family review found that REMATCH calibration
selected epoch 9 while the evaluator used epoch 12 and called this a binding failure.
That finding omitted the frozen protocol and is retracted below.

## Protocol reconciliation

The frozen configs, deployment contract, and milestones M23/M24/M30/M47 all define
two different roles:

- epochs 3/6/9/12 form a calibration-only convergence curve and retain a candidate;
- the formal fixed-threshold scientific gate is evaluated at epoch 12.

Therefore REMATCH selecting epoch 9 for the calibration candidate does not replace the
preregistered epoch-12 formal gate. The evaluator's epoch-12 call chain matches the
frozen protocol. No selected-checkpoint recovery or retrospective protocol change is
valid.

## Integrity checks

### Ground-truth provenance: PASS

Dataset annotations are filtered to the registered 40-video calibration manifest.
Standard OpenTAD mAP and instance metrics use real dataset GT. Reporting was not
accessed.

### Score normalization and thresholds: PASS

mAP is converted from fraction to percentage points only. Recall uses dataset-GT
counts. Birth/alive/end thresholds remain fixed at 0.5; no threshold search occurred.

### Artifact and replay chain: PASS

Both arms completed 24,120 updates. All eight replay receipts preserve event payloads
and calibration metrics and only repair same-frame serialization order. Capacity,
supervision, positive-length, immutable-emission, strict-causality, sequence, split,
checkpoint-hash, and cumulative-resource checks pass.

### Formal result call chain: PASS

The epoch-12 checkpoint and replay ledger are exactly the artifacts required by
`formal_fixed_threshold_gate_epoch=12`. Calibration receipt selection remains a
separate candidate-retention record.

### Scope: WARN

This is one seed, cached causal RGB features, and a 40-video calibration split. It does
not authorize reporting-set, multi-seed, raw-RGB, dataset-generalization, or paper-level
effectiveness claims.

## Valid formal result

- FIXED: average mAP 1.614484 percentage points, Recall@0.3 0.154167,
  prediction/GT ratio 18.841667.
- REMATCH: average mAP 1.178473 percentage points, Recall@0.3 0.0875,
  prediction/GT ratio 16.097917.
- Technical gate: PASS.
- Operational gate: FAIL because both prediction/GT ratios exceed 4.0 and both recalls
  are below 0.25.
- Budget: PASS at 12.319444 arm GPU-hours, or 12.819444 including the frozen finalizer
  reserve.

The FIXED-minus-REMATCH mAP difference of 0.436011 points is a single-seed
calibration-only directional diagnostic, not an effectiveness claim.

## Required next action

Proceed only with model optimization on fit and calibration data. Diagnose excessive
emissions and low recall, then make paired model/loss changes under the same fixed 0.5
threshold. Do not access reporting or start multi-seed/raw-RGB.
