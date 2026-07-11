---
type: paper
node_id: paper:radevski2023-multimodal-distillation
title: "Multimodal Distillation for Egocentric Action Recognition"
authors: ["Gorjan Radevski", "Dusan Grujicic", "Matthew Blaschko", "Marie-Francine Moens", "Tinne Tuytelaars"]
year: 2023
venue: "ICCV"
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: ["multimodal-distillation", "egocentric-video", "privileged-modality", "RGB-only-inference", "calibration"]
added: 2026-07-11T00:00:00Z
---

# Multimodal Distillation for Egocentric Action Recognition

## One-line thesis

Uses multimodal teachers during training to improve RGB-only egocentric action recognition and calibration at deployment.

## Problem / Gap

Complementary modalities improve egocentric recognition but are costly or unavailable at test time.

## Method

Distills modality-specific teachers into an RGB student and studies accuracy, calibration, and compute.

## Key Results

Shows that train-time multimodal information can improve an RGB-only student.

## Assumptions

Training modalities and RGB share task-relevant information.

## Limitations / Failure Modes

The task is action classification, not physical-event timing or observability measurement.

## Reusable Ingredients

RGB-only deployment protocol and fair privileged-modality baselines.

## Open Questions

Whether physical sensor labels help event timing beyond generic cross-modal distillation.

## Claims

None in this project.

## Connections

[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project

Prevents treating sensor-at-train/RGB-at-test as PIVOT's novelty.
