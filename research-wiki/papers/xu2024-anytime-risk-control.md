---
type: paper
node_id: paper:xu2024-anytime-risk-control
title: "Active, Anytime-Valid Risk Controlling Prediction Sets"
authors: ["Ziyu Xu", "Nikos Karampatziakis", "Paul Mineiro"]
year: 2024
venue: "NeurIPS"
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: ["risk-control", "anytime-valid", "e-process", "sequential-inference"]
added: 2026-07-11T00:00:00Z
---

# Active, Anytime-Valid Risk Controlling Prediction Sets

## One-line thesis

Extends risk-controlling prediction sets to sequential, adaptively collected settings with time-uniform guarantees.

## Problem / Gap

A fixed-time calibration guarantee can fail when predictions or stopping times are chosen adaptively.

## Method

Uses e-process and confidence-sequence machinery to maintain risk control simultaneously across time.

## Key Results

Provides finite-sample anytime-valid risk statements under its declared statistical assumptions.

## Assumptions

The formal data conditions must be checked carefully before applying the result to temporally dependent video frames.

## Limitations / Failure Modes

This is a general statistical method, not a temporal action localization model.

## Reusable Ingredients

Sequential calibration at adaptive stopping times.

## Open Questions

How to define exchangeable units and multiple-instance risk for video streams.

## Claims

None in this project.

## Connections

[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project

Provides a possible foundation for false-commit control, while also setting a high bar for honest assumptions.
