---
type: idea
node_id: idea:rejected-one-shot-emission
title: "Rejected: One-Shot Post-End Immutable Emission as Main Story"
stage: archived
outcome: negative
updated: 2026-07-11
target_gaps: ["G1"]
---

# Rejected: One-Shot Post-End Emission

## Rejected Thesis

The model waits until an action ends, then emits `{start,end,class,score}` as quickly as possible, and the output is never corrected.

## Why Rejected

The user rejected this as not elegant and too narrow. It makes the model behave like an alarm instead of a streaming state estimator.

## What Was Salvaged

Immutable committed detections remain necessary for final evaluation. The correction is to add a mutable pre-commit hypothesis layer:

```text
mutable hypothesis -> revision history -> immutable commit
```

## Do Not Revive Unless

Only use one-shot immutable emission as a baseline or ablation, not as the main paper story.
