---
type: idea
node_id: idea:rejected-full-visual-tower-first
title: "Rejected Default: Full Visual Tower Finetuning First"
stage: archived
outcome: negative
updated: 2026-07-11
target_gaps: ["G5"]
---

# Rejected Default: Full Visual Tower Finetuning First

## Rejected Thesis

Start by fully finetuning the visual tower end-to-end for Online TAD.

## Why Rejected

- Current code freezes SigLIP2 and may wrap visual path in `no_grad`.
- No formal update audit proves full-tower trainability.
- THUMOS-scale data may be too small.
- Cost is likely unacceptable before frozen/LoRA evidence.

## Allowed Later

Only after:

- PCEH/CESR beats endpoint-only;
- LoRA beats frozen;
- LoRA audit passes;
- raw episodic throughput is acceptable;
- one-epoch full-tower diagnostic fits allocation.
