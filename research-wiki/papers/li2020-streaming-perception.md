---
type: paper
node_id: paper:li2020-streaming-perception
title: "Towards Streaming Perception"
authors: ["Mengtian Li", "Yu-Xiong Wang", "Deva Ramanan"]
year: 2020
venue: "ECCV"
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: ["streaming-perception", "latency", "real-time", "streaming-accuracy"]
added: 2026-07-11T00:00:00Z
---

# Towards Streaming Perception

## One-line thesis

Integrates algorithm latency and accuracy into streaming perception metrics because outputs computed from old frames are stale when they become available.

## Problem / Gap

Offline frame evaluation assumes instantaneous inference and misrepresents real-time video perception.

## Method

Introduces streaming accuracy and analyzes forecasting, asynchronous tracking, and dynamic scheduling.

## Key Results

Finds nontrivial latency-accuracy trade-offs and operational schedules for real-time detection and segmentation.

## Assumptions

The key delay is computation relative to a changing visual world.

## Limitations / Failure Modes

Does not model when a physical event becomes visually verifiable.

## Reusable Ingredients

Strict separation of input timestamp, output timestamp, and wall-clock latency.

## Open Questions

How semantic observability and computational staleness should compose.

## Claims

None in this project.

## Connections

[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project

Blocks claiming that real-time output latency accounting is new; PIVOT must add a distinct semantic evidence clock.
