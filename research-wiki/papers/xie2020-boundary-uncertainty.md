---
type: paper
node_id: paper:xie2020-boundary-uncertainty
title: "Boundary Uncertainty in a Single-Stage Temporal Action Localization Network"
authors: ["Ting-Ting Xie", "Christos Tzelepis", "Ioannis Patras"]
year: 2020
venue: "arXiv"
external_ids:
  arxiv: "2008.11170"
  doi: null
  s2: null
tags: ["temporal-action-localization", "boundary-uncertainty", "Gaussian", "probabilistic-boundary"]
added: 2026-07-11T00:00:00Z
---

# Boundary Uncertainty in a Single-Stage Temporal Action Localization Network

## One-line thesis

Models predicted temporal action boundaries as univariate Gaussian distributions and trains uncertainty-aware boundary losses.

## Problem / Gap

Point boundary regression ignores predictive uncertainty.

## Method

Uses KL-based and expected L1 objectives over Gaussian boundary predictions.

## Key Results

Reports improved TAL localization performance with uncertainty-aware boundaries.

## Assumptions

Gaussian uncertainty is an adequate model for boundary predictions.

## Limitations / Failure Modes

Predictive uncertainty around one annotation is not the same object as population-level visual observability conditioned on a physical sensor time.

## Reusable Ingredients

Probabilistic boundary baseline.

## Open Questions

Whether simple Gaussian uncertainty can explain PIVOT observations without a separate visual clock.

## Claims

None in this project.

## Connections

[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project

Blocks interval/distributional boundaries as method novelty and provides a required baseline.
