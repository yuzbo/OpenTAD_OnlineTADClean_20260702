---
type: paper
node_id: paper:vidosc2024-openworld-state-change
title: "Learning Object State Changes in Videos: An Open-World Perspective"
authors: ["Zihui Xue", "Kumar Ashutosh", "Kristen Grauman"]
year: 2024
venue: "CVPR"
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: ["object-state-change", "open-world", "temporal-localization", "egocentric-video"]
added: 2026-07-11T00:00:00Z
---

# Learning Object State Changes in Videos: An Open-World Perspective

## One-line thesis

Temporally localizes initial, transitioning, and end states for familiar and unseen object state changes.

## Problem / Gap

Prior object-state-change models were limited to closed vocabularies and narrower datasets.

## Method

Introduces VidOSC and HowToChange, using vision-language supervision and object-independent state representations.

## Key Results

Establishes open-world three-stage state-change localization as a benchmark.

## Assumptions

Visual state stages provide sufficient supervision for state-change timing.

## Limitations / Failure Modes

No independent physical sensor clock, population visual evidence time, or strict streaming commit protocol.

## Reusable Ingredients

Three-stage object-state formulation and external generalization benchmark.

## Open Questions

Whether state-stage labels systematically lag or precede physical transition signals.

## Claims

None in this project.

## Connections

[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project

Blocks novelty from pre/transition/post state modeling and makes PIVOT's measurement delta essential.
