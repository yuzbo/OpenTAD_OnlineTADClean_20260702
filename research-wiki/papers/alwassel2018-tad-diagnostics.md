---
type: paper
node_id: paper:alwassel2018-tad-diagnostics
title: "Diagnosing Error in Temporal Action Detectors"
authors: ["Humam Alwassel", "Fabian Caba Heilbron", "Victor Escorcia", "Bernard Ghanem"]
year: 2018
venue: "ECCV"
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: ["temporal-action-detection", "diagnostics", "human-annotation", "boundary-agreement"]
added: 2026-07-11T00:00:00Z
---

# Diagnosing Error in Temporal Action Detectors

## One-line thesis

Introduces a diagnostic tool for temporal action detection and studies error sources and human temporal-boundary agreement beyond scalar mAP.

## Problem / Gap

Aggregate localization metrics hide method-specific failure modes and the role of action characteristics and annotations.

## Method

Analyzes ActivityNet detectors and recollects multiple human boundary annotations.

## Key Results

Shows that temporal boundary agreement is imperfect while arguing it is not the sole blocker to detector progress.

## Assumptions

ActivityNet reannotations represent the relevant annotator population.

## Limitations / Failure Modes

It does not distinguish physical transition, visual evidence arrival, and model decision.

## Reusable Ingredients

Multi-rater boundary protocol and diagnostic evaluation philosophy.

## Open Questions

Whether population response curves around instrumented events are more stable than free boundary annotation.

## Claims

None in this project.

## Connections

[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project

Requires PIVOT to use population intervals and to prove that observability is not ordinary boundary disagreement.
