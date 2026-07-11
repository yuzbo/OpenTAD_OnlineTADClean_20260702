---
type: idea
node_id: idea:offline-teacher-distillation
title: "Offline Teacher to Online Student Distillation"
stage: support-only
outcome: mixed
updated: 2026-07-11
target_gaps: ["G5", "G8"]
---

# Offline Teacher to Online Student Distillation

## One-Line Thesis

Use an offline teacher only during training to provide privileged signals to a prefix-only online student.

## Why Useful

Could improve:

- class posterior;
- actionness/CAS;
- start confidence;
- endpoint posterior;
- proposal quality;
- hard-negative scores.

## Why Not Main Novelty

OnPoint and PKD-style methods already occupy this territory. It should be an ablation or enhancement, not the paper's core claim.

## Safe Wording

> Privileged teacher used for training only; no teacher, teacher cache, future context, or raw prediction shortcut at test time.

## Unsafe Wording

> Training uses no future information.
