---
type: idea
node_id: idea:lora-stage2
title: "Selective-Gradient SigLIP2 LoRA Stage 2"
stage: conditional
outcome: pending
updated: 2026-07-11
target_gaps: ["G5"]
---

# Selective-Gradient SigLIP2 LoRA Stage 2

## One-Line Thesis

After frozen Stage 1 passes, adapt the top visual blocks with LoRA rather than full-tower finetuning.

## Initial Proposed Config

- target blocks: top 4 visual transformer blocks;
- target modules: `q_proj`, `v_proj`;
- rank: 8;
- alpha: 16;
- dropout: 0.05;
- LoRA learning rate: about `2e-5`;
- temporal stack learning rate: about `5e-5`;
- head learning rate: about `1e-4`.

## Why Selected Conditionally

- Lower cost than full visual tower.
- More plausible for THUMOS-scale data.
- Can address "pretrained model missing online TAD adaptation" without overclaiming full E2E.

## Hard Gates

- Runtime module discovery; zero target hits fail closed.
- Each target block has nonzero gradient and parameter delta.
- Base weights remain frozen.
- No global `torch.no_grad()` around trainable LoRA.
- Optimizer coverage audit lists every trainable target.
- Cache refresh or cache checkpoint binding after LoRA changes.

## Failure Criteria

- LoRA truly updates but gives no AP-latency or calibration gain over frozen.
- Cache mismatch invalidates training.
- Full visual tower is needed just to make it work, making the route too costly.
