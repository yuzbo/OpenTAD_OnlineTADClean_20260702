---
type: paper
node_id: paper:munoz2026-tracking-eprocess
title: "Detecting Object Tracking Failure via Sequential Hypothesis Testing"
authors: ["Alejandro Monroy Muñoz", "Rajeev Verma", "Alexander Timans"]
year: 2026
venue: "WACV Workshops"
external_ids:
  arxiv: "2602.12983"
  doi: null
  s2: null
tags: ["e-process", "sequential-testing", "video", "tracking-failure", "false-alert-control"]
added: 2026-07-11T00:00:00Z
---

# Detecting Object Tracking Failure via Sequential Hypothesis Testing

## One-line thesis

Uses an e-process to detect real-time object-tracking failure with a provable false-alert guarantee, independently of the tracker and without retraining it.

## Problem / Gap

Tracking confidence scores are not reliable long-horizon failure alarms when repeatedly monitored.

## Method

Transforms tracker-derived evidence into a sequential hypothesis test and alarms when the e-process crosses a risk-controlled threshold.

## Key Results

The paper reports model-agnostic real-time failure detection and a formal false-alert guarantee.

## Assumptions

The guarantee depends on the validity of the sequential evidence construction under the declared null.

## Limitations / Failure Modes

The target is tracker failure, not repeated semantic action instances or temporal localization.

## Reusable Ingredients

Direct video precedent for e-process alarms, model-agnostic score wrapping, and false-alert control.

## Open Questions

Whether semantic event candidates, resets, repeated instances, and nonstationarity permit a comparably valid and useful process.

## Claims

None in this project.

## Connections

[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project

This is the strongest direct novelty threat to T01. It invalidates any broad claim that e-process-based false-alert control is new merely because the input is video.
