---
type: paper
node_id: paper:watch2025-conformal-monitoring
title: "WATCH: Adaptive Monitoring for AI Deployments via Weighted-Conformal Martingales"
authors: ["Drew Prinster", "Xing Han", "Anqi Liu", "Suchi Saria"]
year: 2025
venue: "ICML"
external_ids:
  arxiv: "2505.04608"
  doi: null
  s2: null
tags: ["change-detection", "conformal", "martingale", "false-alarm-control", "streaming"]
added: 2026-07-11T00:00:00Z
---

# WATCH: Adaptive Monitoring for AI Deployments via Weighted-Conformal Martingales

## One-line thesis

Develops weighted adaptive conformal testing and martingale machinery for online change monitoring with false-alarm control.

## Problem / Gap

Streaming change detection needs time-uniform error control while preserving detection power.

## Method

Uses weighted conformal evidence and adaptive betting within a sequential monitoring process.

## Key Results

Provides a strong generic baseline for risk-delay evaluation in streaming data.

## Assumptions

Validity relies on the paper's data and conformity-score assumptions; video dependence must be checked rather than presumed.

## Limitations / Failure Modes

It detects distributional changes, not semantic action instances with temporal spans.

## Reusable Ingredients

Conformal martingales, adaptive betting, and long-horizon false-alarm evaluation.

## Open Questions

Whether a semantic-event formulation needs anything beyond WATCH-style monitoring on a causal score.

## Claims

None in this project.

## Connections

[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project

Mandatory baseline and novelty threat for T01.
