---
type: paper
node_id: paper:tadtr2021-action-queries
title: "End-to-end Temporal Action Detection with Transformer"
authors: ["Xiaolong Liu", "Qimeng Wang", "Yao Hu", "Xu Tang", "Shiwei Zhang", "Song Bai", "Xiang Bai"]
year: 2021
venue: "arXiv"
external_ids:
  arxiv: "2106.10271"
  doi: null
  s2: null
tags: ["temporal-action-detection", "action-query", "set-prediction", "interval-head", "closest-prior"]
added: 2026-07-17T12:00:00+08:00
---

# End-to-end Temporal Action Detection with Transformer

## One-line thesis

TadTR uses learnable action queries to directly predict action intervals and
classes with an end-to-end Transformer detector.

## Problem / Gap

Earlier TAD pipelines relied on anchors, NMS, and multiple separately trained
stages.

## Method

- learnable action queries;
- temporal deformable attention;
- direct interval and class prediction;
- segment refinement and actionness regression.

## Key Results

The paper reports competitive end-to-end TAD results with lower detector
complexity than multi-stage pipelines.

## Assumptions

The original formulation is offline and may use the full video context. It
does not establish strict causal On-TAL behavior.

## Limitations / Failure Modes

Action queries alone do not solve online persistence, immutable commit,
duplicate control, or endpoint latency.

## Reusable Ingredients

Direct action-query set prediction and matched temporal interval heads.

## Open Questions

Can a causal per-prefix version satisfy standard On-TAL without persistent
identity, and therefore serve as B0 or part of B1?

## Claims

No project claim is verified by this paper alone.

## Connections

[AUTO-GENERATED from graph/edges.jsonl; do not hand-edit]

## Relevance to This Project

TadTR blocks novelty claims based only on action queries and direct interval
prediction. It motivates matching B0-B4 output heads.

## Source

https://arxiv.org/abs/2106.10271
