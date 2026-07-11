---
type: idea
node_id: idea:continual-openworld-ontal
title: "Continual Open-World Online Temporal Action Localization"
stage: proposed
outcome: pending
updated: 2026-07-11
target_gaps: ["G13"]
---

# Continual Open-World Online Temporal Action Localization

## Thesis

Move from online inference with fixed weights to a stream that introduces domain shifts and new action classes, where the model must localize instances, abstain on unknowns, and adapt from delayed sparse feedback without forgetting prior classes.

## Why It Matters

Most On-TAL methods are trained offline and are only online at inference. OZ-TAL handles unseen actions without training, while class-incremental OAD and video test-time adaptation address adjacent pieces. Their intersection at instance-level temporal localization remains underdeveloped.

## Risks

- Combining zero-shot, continual learning, and On-TAL can look like a task union rather than one new mechanism.
- A credible benchmark needs chronological domain/class shifts and delayed-label protocol.
- Continual adaptation multiplies training and evaluation cost.

## Role

High-ambition alternative if the project is willing to build a benchmark. It is not the near-term recommendation under the current compute budget.
