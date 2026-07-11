---
type: idea
node_id: idea:feature-cache-stage1
title: "Frozen SigLIP2 Feature Cache as Stage 1"
stage: active
outcome: pending
updated: 2026-07-11
target_gaps: ["G4", "G5"]
---

# Frozen SigLIP2 Feature Cache as Stage 1

## One-Line Thesis

Use audited cached features for frozen SigLIP2 Stage 1 training to remove repeated visual forward cost while preserving causal source provenance.

## Why Selected

The review estimated the selected frame feature cache is modest in size compared with repeating SigLIP2 forward many times. Frozen per-frame encoder features can be cached if preprocessing, checkpoint, frame policy, and source frame provenance are identical.

## Safe Claim

> Frozen SigLIP2 features plus learned causal online detector.

## Unsafe Claim

> End-to-end visual representation learning from raw video.

## Required Cache Manifest

- visual checkpoint SHA;
- processor/preprocessing hash;
- frame policy hash;
- source frame index;
- dtype;
- feature shape;
- cache build code revision;
- raw/cache equivalence audit result.

## Evaluation Rule

Cache may accelerate training and validation, but final paper evaluation should still be expressible as:

```text
raw chronological stream
-> SigLIP2
-> learned temporal detector
-> revision/commit ledger
```

Do not load raw predictions as test input.
