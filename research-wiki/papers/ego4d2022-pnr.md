---
type: paper
node_id: paper:ego4d2022-pnr
title: "Ego4D: Around the World in 3,000 Hours of Egocentric Video"
authors: ["Kristen Grauman", "Andrew Westbury", "Eugene Byrne", "et al."]
year: 2022
venue: "CVPR"
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: ["egocentric-video", "PNR", "object-state-change", "keyframe-localization", "benchmark"]
added: 2026-07-11T00:00:00Z
---

# Ego4D: Around the World in 3,000 Hours of Egocentric Video

## One-line thesis

Introduces a large-scale egocentric benchmark suite whose Hands + Objects tasks include PRE, CONTACT, point-of-no-return, POST, state-change classification, and keyframe localization.

## Problem / Gap

Egocentric perception lacked large-scale benchmarks for hand-object interaction, state changes, memory, and anticipation.

## Method

Collects thousands of hours of egocentric video and provides task-specific annotations and baselines.

## Key Results

Established PNR temporal localization as a standard task for estimating the frame where a state change begins or becomes inevitable.

## Assumptions

PNR is an annotator-defined visually interpreted keyframe.

## Limitations / Failure Modes

PNR does not independently measure a physical transition or population evidence-arrival curve and is generally evaluated as clip-level localization rather than residual online decision delay.

## Reusable Ingredients

PRE/CONTACT/PNR/POST schema, public egocentric videos, external state-change baselines.

## Open Questions

How often visual PNR differs from a synchronized force/contact transition.

## Claims

None in this project.

## Connections

[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project

Blocks novelty from state-change keyframe localization itself and is a mandatory external baseline.
