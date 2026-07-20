---
type: paper
node_id: paper:matr2024-memory
title: "Online Temporal Action Localization with Memory-Augmented Transformer"
authors: ["Youngkil Song", "Dongkeun Kim", "Minsu Cho", "Suha Kwak"]
year: 2024
venue: "ECCV"
external_ids:
  arxiv: "2408.02957"
  doi: null
tags: ["On-TAL", "memory", "start-retrieval", "baseline"]
added: 2026-07-11
---

# Online Temporal Action Localization with Memory-Augmented Transformer

## One-line thesis

Use current evidence to predict action end and retrieve action start from a historical memory queue.

## Relevance to This Project

MATR is a strong online span baseline. It shows that memory-based end-to-start localization is already covered.

Primary source: [arXiv:2408.02957](https://arxiv.org/abs/2408.02957). The
2026-07-20 FIXED/REMATCH readiness review correctly described MATR's role but
attached that sentence to the CAG-QIL CVF footnote; this page records the
correct direct source.

## Overlap

- online temporal action localization;
- historical memory;
- endpoint/current segment reasoning;
- start retrieval from past context.

## Key Difference We Need

CESR should not claim memory/history as novelty. The delta must be:

- revisable hypothesis state over time;
- explicit revision and commit ledger;
- start/ongoing/end lifecycle;
- boundary refinement trajectory before commit.

## Required Baseline Role

At least conceptual comparison. If implementable, use a MATR-style feature baseline.
