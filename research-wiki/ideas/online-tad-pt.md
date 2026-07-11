---
type: idea
node_id: idea:online-tad-pt
title: "Online TAD-Specific Pretraining"
stage: future
outcome: pending
updated: 2026-07-11
target_gaps: ["G5"]
---

# Online TAD-Specific Pretraining

## One-Line Thesis

Develop pretraining or fine-tuning objectives that teach pretrained video models online event-state tracking, boundary refinement, and commit behavior.

## Why It Matters

The user repeatedly emphasized that existing pretrained models lack online TAD-specific adaptation. This is a real long-term gap.

## Why It Is Not First

It is too broad and expensive before the basic CESR/PCEH objective is validated. Without proof that event-state/commit objectives help, a pretraining story becomes diffuse and vulnerable.

## Future Direction

After Stage 1/2 evidence:

- pretrain on causal prefixes;
- predict start/ongoing/end/commit states;
- learn boundary refinement trajectories;
- add contrastive or masked-prefix objectives;
- evaluate transfer to On-TAD and maybe OnlineTAS.
