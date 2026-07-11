---
type: paper
node_id: paper:pasbench2026-proactive-warning
title: "PaSBench-Video: A Streaming Video Benchmark for Proactive Safety Warning"
authors: ["Yusong Zhao", "Yuejin Xie", "Youliang Yuan", "Junjie Hu", "Jitian Guo", "Yujiu Yang", "Pinjia He"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2606.02443"
  doi: null
  s2: null
tags: ["streaming-video", "proactive-warning", "risk-onset", "accident-boundary", "response-timing"]
added: 2026-07-11T00:00:00Z
---

# PaSBench-Video: A Streaming Video Benchmark for Proactive Safety Warning

## One-line thesis

Evaluates whether streaming video models issue content-correct warnings between the first visible danger cue and the accident boundary while controlling false positives on safe scenes.

## Problem / Gap

Static safety benchmarks ignore warning timing and safe-scene false positives.

## Method

Introduces 740 videos with frame-level risk onset and accident boundaries across four domains and evaluates causal MLLM warnings.

## Key Results

Reports low strict-metric performance and a strong coupling between recall and false-positive rate across tested MLLMs.

## Assumptions

Risk onset and accident boundaries are visually annotated task labels rather than independently sensed physical transitions.

## Limitations / Failure Modes

The task is safety warning, not general physical event verification; it does not estimate population-level visual observability or residual algorithmic delay.

## Reusable Ingredients

First-visible-cue versus accident timing, causal warning protocol, safe-scene false-positive accounting.

## Open Questions

Whether PIVOT is more than a physically instrumented generalization of this timing structure.

## Claims

None in this project.

## Connections

[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project

Most dangerous conceptual threat to a broad “physical event, visible evidence, warning” three-clock claim.
