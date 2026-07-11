---
type: paper
node_id: paper:selectstream2026-budgeted-memory
title: "What Should a Streaming Video Model Remember?"
authors: ["Haonan Ge", "Yiwei Wang", "Hang Wu", "Yujun Cai"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2606.16353"
  doi: null
  s2: null
tags: ["streaming-video", "budgeted-memory", "evidence-selection", "video-language-model"]
added: 2026-07-11T00:00:00Z
---

# What Should a Streaming Video Model Remember?

## One-line thesis

SelectStream treats streaming memory as budgeted online latent-evidence allocation and selectively writes, consolidates, and retrieves history for a frozen VLM.

## Problem / Gap

Injecting indiscriminate history can dilute current-scene perception under fixed memory and compute budgets.

## Method

Surprise-driven windows, priority-preserving consolidation, and query-conditioned retrieval over fixed-capacity latent memory.

## Key Results

Reports strong results on StreamingBench, OVO-Bench, and offline video benchmarks.

## Assumptions

Evaluation is query-conditioned streaming video understanding rather than instance On-TAL.

## Limitations / Failure Modes

Does not provide fine-grained temporal action span commitments or On-TAL risk guarantees.

## Reusable Ingredients

Fixed-capacity evidence allocation and strong recent-window baselines.

## Open Questions

Whether event-boundary evidence needs different write policies from general VideoQA evidence.

## Claims

None in this project.

## Connections

[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project

Shows that generic selective streaming memory is already a direct, current research direction and should not be revived as this paper's headline.
