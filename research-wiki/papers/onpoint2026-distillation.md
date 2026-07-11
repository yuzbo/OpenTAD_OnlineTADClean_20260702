---
type: paper
node_id: paper:onpoint2026-distillation
title: "OnPoint: Offline-to-Online Multi-Level Distillation for Point-Supervised Online Temporal Action Localization"
authors: []
year: 2026
venue: "ECCV / arXiv"
external_ids:
  arxiv: "2607.00289"
  doi: null
tags: ["On-TAL", "distillation", "point-supervision", "competitor"]
added: 2026-07-11
---

# OnPoint 2026

## One-line thesis

An offline TAL teacher distills pseudo segments, class activation, and anticipatory cues into an online student.

## Relevance to This Project

This blocks offline-to-online distillation as the primary novelty. Distillation may still help as a training enhancement or ablation.

## Overlap

- online TAL;
- offline teacher to online student;
- privileged training signal;
- anticipation/prefix student.

## Key Difference We Need

If teacher distillation is used:

- teacher is training-only;
- no teacher cache or prediction at test time;
- CESR/PCEH objectives remain the main contribution;
- compare against non-teacher version.
