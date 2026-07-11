---
type: idea
node_id: idea:frozen-siglip-baseline
title: "Frozen SigLIP/SigLIP2 Baseline"
stage: baseline
outcome: partial
updated: 2026-07-11
target_gaps: ["G4", "G5"]
---

# Frozen SigLIP/SigLIP2 Baseline

## One-Line Thesis

Frozen image/video foundation features can serve as the initial visual representation for Online TAD.

## Role

Baseline and Stage 1 infrastructure, not novelty.

## Safe Claims

- raw frame packets can feed a frozen visual encoder;
- selected-only encoding can be audited;
- feature cache can accelerate frozen training;
- causal detector/head learns on top of frozen features.

## Unsafe Claims

- pretrained model is fine-tuned end-to-end;
- frozen VLM makes the method novel;
- zero-shot/open-vocabulary novelty.
