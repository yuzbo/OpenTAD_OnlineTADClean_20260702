---
type: paper
node_id: paper:motr2021-track-queries
title: "MOTR: End-to-End Multiple-Object Tracking with Transformer"
authors: ["Fangao Zeng", "Bin Dong", "Yuang Zhang", "Tiancai Wang", "Xiangyu Zhang", "Yichen Wei"]
year: 2021
venue: "arXiv"
external_ids:
  arxiv: "2105.03247"
  doi: null
  s2: null
tags: ["tracking", "persistent-query", "set-prediction", "birth-survival", "closest-prior"]
added: 2026-07-17T12:00:00+08:00
---

# MOTR: End-to-End Multiple-Object Tracking with Transformer

## One-line thesis

MOTR propagates and updates track queries frame by frame, with
tracklet-aware assignment for existing and newborn objects.

## Problem / Gap

Tracking-by-detection separates detection and association, limiting
end-to-end temporal learning.

## Method

- DETR-style object and track queries;
- recurrent track-query propagation;
- tracklet-aware label assignment;
- end-to-end temporal aggregation and loss.

## Key Results

The paper reports strong association performance on MOT17 and DanceTrack and
positions MOTR as a baseline for Transformer-based tracking.

## Assumptions

The task observes spatial object detections and track identity, not temporal
action intervals under On-TAL output rules.

## Limitations / Failure Modes

It does not directly solve action start/end localization, immutable action
commit, or endpoint latency. A temporal reconstruction must define those
heads and outputs explicitly.

## Reusable Ingredients

Persistent query propagation, newborn-query assignment, identity-aware
training, and lifecycle handling.

## Open Questions

Can R-A produce an observable On-TAL advantage after these mechanisms are
mapped to one temporal dimension with matched interval and completion heads?

## Claims

No project claim is verified by this paper alone.

## Connections

[AUTO-GENERATED from graph/edges.jsonl; do not hand-edit]

## Relevance to This Project

MOTR is a mandatory obviousness and reconstructibility baseline. R-A is killed
as a distinct route if a matched temporal MOTR enters the preregistered
equivalence region.

## Source

https://arxiv.org/abs/2105.03247
