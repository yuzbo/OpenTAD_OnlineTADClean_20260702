---
type: paper
node_id: paper:protas2024-progress-aware
title: "Progress-Aware Online Action Segmentation for Egocentric Procedural Task Videos"
authors: ["Yuhan Shen", "Ehsan Elhamifar"]
year: 2024
venue: "CVPR"
external_ids:
  arxiv: null
  doi: null
tags: ["online-TAS", "action-progress", "state-refinement", "task-graph", "causal-training"]
added: 2026-07-11
---

# ProTAS 2024

## One-line thesis

Online action segmentation improves when the model causally estimates ongoing action progress and uses it to refine framewise action predictions.

## Method

- Convert TCN/Transformer segmentation backbones to causal computation.
- Predict per-action progress with a GRU.
- Refine causal action probabilities using predicted progress.
- Use a learned task graph to enforce procedure-consistent transitions.

## Limitations / Failure Modes

This is dense framewise segmentation of procedural steps, not instance-level Online TAD with final detection commitments.

## Relevance to This Project

ProTAS is a direct adjacent threat to claims about dynamically maintaining ongoing action state or using progress to refine online predictions. Lifecycle/progress supervision alone cannot be the CESR novelty.

## Connections

[AUTO-GENERATED from graph/edges.jsonl - do not edit manually]
