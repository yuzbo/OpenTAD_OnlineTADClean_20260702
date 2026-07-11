---
type: claim
node_id: claim:c5-visual-adaptation
title: "Parameter-efficient visual adaptation improves Online TAD"
status: unproven
updated: 2026-07-11
---

# Claim C5: Parameter-Efficient Visual Adaptation

## Statement

LoRA or adapter tuning of a pretrained visual backbone improves online event-state tracking beyond frozen features.

## Current Support

None formal. Current local code previously froze SigLIP2 and had wrapper/no-grad concerns.

## Required Evidence

- runtime-verified LoRA module injection;
- gradient and parameter-delta audit for each target block;
- base weights frozen;
- frozen vs LoRA paired comparison;
- full chronological evaluation;
- cache refresh/checkpoint binding.

## Failure Criterion

If audited LoRA updates do not improve over frozen Stage 1, stop visual adaptation claim.
