---
type: paper
node_id: paper:stare2026-stream-latency
title: "Bridging the Latency Gap with a Continuous Stream Evaluation Framework in Event-Driven Perception"
authors: ["Jie Chu", "Runze Zhang", "Chu Yang", "Zongyou Yu", "Zongtao Bu", "Haotian Liu", "Florian Rohrbein", "Alois Knoll", "Guang Chen", "Changjun Jiang"]
year: 2026
venue: "Nature Communications"
external_ids:
  arxiv: null
  doi: "10.1038/s41467-026-70240-6"
  s2: null
tags: ["streaming-perception", "latency-aware-evaluation", "event-camera", "ranking-reversal", "real-time"]
added: 2026-07-11T00:00:00Z
---

# Bridging the Latency Gap with a Continuous Stream Evaluation Framework in Event-Driven Perception

## One-line thesis

Introduces STARE, continuous sampling and latency-aware evaluation for event-driven tracking, backed by 500 Hz annotations and robotic validation.

## Problem / Gap

Frame-based evaluation ignores real processing latency and temporal staleness in continuous event streams.

## Method

Aligns dense ground-truth queries with the latest available model output and measures accuracy degradation under real throughput.

## Key Results

Reports large latency-induced accuracy degradation and model ranking reversals under the stream-aware protocol.

## Assumptions

Dense object-state annotations approximate downstream continuous queries.

## Limitations / Failure Modes

The decomposed delay is computational/sampling latency in event-camera tracking, not the semantic delay between a physical transition and visually sufficient evidence.

## Reusable Ingredients

Timestamped output accounting, dense chronological evaluation, ranking-reversal analysis.

## Open Questions

How to combine semantic observability delay with wall-clock streaming latency without double counting.

## Claims

None in this project.

## Connections

[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project

Blocks novelty from latency-aware evaluation and ranking reversal alone; PIVOT must isolate semantic observability confounding.
