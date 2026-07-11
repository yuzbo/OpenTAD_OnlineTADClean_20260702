---
type: paper
node_id: paper:trackformer2022-tracking-queries
title: "TrackFormer: Multi-Object Tracking with Transformers"
authors: ["Tim Meinhardt", "Alexander Kirillov", "Laura Leal-Taixe", "Christoph Feichtenhofer"]
year: 2022
venue: "CVPR"
external_ids:
  arxiv: "2101.02702"
  doi: "10.1109/CVPR52688.2022.00861"
  s2: null
tags: ["tracking", "persistent-query", "Hungarian", "novelty-threat"]
added: 2026-07-12
---

# TrackFormer 2022

## One-line thesis

Represent object births with static queries and preserve existing identities autoregressively with track queries under set prediction.

## Problem / Gap

Tracking-by-detection pipelines split object detection, data association, and motion modeling into separate stages. TrackFormer asks whether one transformer can perform detection and identity association jointly.

## Method

Static object queries propose births. Track queries carry previously detected identities into the next frame. Transformer attention and bipartite set assignment jointly reason about births, continuations, and interactions.

## Key Results

The paper establishes persistent identity-bearing queries as a practical multi-object tracking mechanism. This wiki does not import its benchmark numbers as evidence for temporal action localization.

## Assumptions

- framewise visual observations arrive in order;
- spatial objects have identity trajectories;
- training annotations support set matching across frames.

## Limitations / Failure Modes

- It addresses spatial multi-object tracking, not action start/end localization.
- It does not establish that persistent queries improve On-TAD.
- A direct temporal transplant may be obvious and may add capacity without fixing On-TAD errors.

## Reusable Ingredients

- static birth queries;
- autoregressive identity queries;
- Hungarian assignment;
- explicit lifecycle reasoning without post-hoc identity association.

## Open Questions

- Does temporal action identity require a mechanism beyond scalar boundary regression?
- Can persistence reduce fragmentation without creating stale or duplicate event slots?

## Claims

No project claim is supported by this paper alone.

## Connections

[AUTO-GENERATED from `graph/edges.jsonl`; do not hand-edit.]

## Relevance to This Project

This is the strongest architectural obviousness attack against persistent On-TAD event queries. Query persistence, identity carry, birth/death handling, and Hungarian set assignment are not new merely because the axis is temporal rather than spatial.

## Required Baseline Role

Stage 1 includes a Temporal TrackFormer reconstruction with persistent queries, scalar start regression, binary endpoint classification, and the same frozen features/capacity as the proposed event-set variant. The route fails if the extra On-TAD-specific pointer/hazard design is equivalent or worse.

## Source

- https://arxiv.org/abs/2101.02702
- https://github.com/timmeinhardt/trackformer
