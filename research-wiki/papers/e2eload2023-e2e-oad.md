---
type: paper
node_id: paper:e2eload2023-e2e-oad
title: "E2E-LOAD: End-to-End Long-form Online Action Detection"
authors: []
year: 2023
venue: "ICCV / arXiv"
external_ids:
  arxiv: "2306.07703"
  doi: null
tags: ["OAD", "raw-video", "end-to-end", "streaming-cache", "training-cost"]
added: 2026-07-11
---

# E2E-LOAD 2023

## One-line thesis

Raw-video end-to-end online action detection can train on short history and infer with longer cached history.

## Relevance to This Project

Useful for cost/control ideas, especially short training history and long streaming cache. It is OAD/frame-level, not instance-level On-TAD.

## Overlap

- raw-video online action model;
- stream/cache;
- short-train long-infer pattern;
- cost reduction concern.

## Key Difference We Need

CESR must produce action instances and revision/commit ledgers, not just frame-level online action detection.
