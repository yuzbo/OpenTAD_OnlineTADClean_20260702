---
type: paper
node_id: paper:heidarivincheh2018-action-completion
title: "Action Completion: A Temporal Model for Moment Detection"
authors: ["Farnoosh Heidarivincheh", "Majid Mirmehdi", "Dima Damen"]
year: 2018
venue: "arXiv"
external_ids:
  arxiv: "1805.06749"
  doi: null
  s2: null
tags: ["action-completion", "completion-moment", "incomplete-action", "temporal-evidence"]
added: 2026-07-11T00:00:00Z
---

# Action Completion: A Temporal Model for Moment Detection

## One-line thesis

Defines and detects the moment when an action's goal can be considered completed, including incomplete executions.

## Problem / Gap

Motion ending and goal completion are not always the same semantic event.

## Method

A recurrent classification/regression model votes for completion timing from frame-level evidence.

## Key Results

Evaluates completion-moment accuracy on 16 actions from three datasets.

## Assumptions

Actions have a semantically meaningful completion criterion.

## Limitations / Failure Modes

The task is narrower than multi-instance On-TAL and predates current streaming foundation models.

## Reusable Ingredients

Completion versus incompletion labels and goal-based event semantics.

## Open Questions

How completion evidence relates to annotated action endpoints and first valid online detection.

## Claims

None in this project.

## Connections

[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project

Prevents claiming action completion itself as new, while supporting the distinction between physical end and evidence-ready time.
