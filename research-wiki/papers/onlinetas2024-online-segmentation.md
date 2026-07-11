---
type: paper
node_id: paper:onlinetas2024-online-segmentation
title: "OnlineTAS: An Online Baseline for Temporal Action Segmentation"
authors: []
year: 2024
venue: "NeurIPS / arXiv"
external_ids:
  arxiv: "2411.01122"
  doi: null
tags: ["OnlineTAS", "segmentation", "adaptive-memory", "prediction-correction"]
added: 2026-07-11
---

# OnlineTAS 2024

## One-line thesis

Online temporal action segmentation needs adaptive memory and correction mechanisms to handle partial observations and reduce erratic/over-segmented predictions.

## Relevance to This Project

Conceptually close to "state evolves and predictions can be corrected," but the task is dense segmentation, not instance-level On-TAD.

## Overlap

- online setting;
- partial observations;
- adaptive memory;
- prediction correction and boundary adjustment.

## Key Difference We Need

CESR should emphasize instance-level action hypotheses, revisable track state, commit ledger, and OnlineAP/latency metrics.
