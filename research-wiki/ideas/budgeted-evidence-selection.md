---
type: idea
node_id: idea:budgeted-evidence-selection
title: "Budgeted Causal Evidence Selection and Irregular-Time Decoding"
stage: future
outcome: pending
updated: 2026-07-11
target_gaps: ["G4", "G8"]
---

# Budgeted Causal Evidence Selection

## One-Line Thesis

Learn or design a causal selector that chooses the most useful frames/tokens under a compute budget, then decode on irregular time evidence.

## Why Not Current Main Route

Pro review scored this route as potentially valuable but risky. Current code only has fixed causal stride, not learned/adaptive selection. Calling it adaptive is explicitly forbidden.

## Required Baselines

- uniform same-budget;
- random same-budget;
- recent-window;
- motion heuristic;
- dense frozen upper bound;
- fixed causal stride.

## Reviewer Risk

If uniform or random same-budget selection matches it, the selector is not a contribution. If irregular-time decode secretly uses whole-video duration or future positions, online claim fails.

## Status

Future stage only after CESR/PCEH target correctness and training cost gates pass.
