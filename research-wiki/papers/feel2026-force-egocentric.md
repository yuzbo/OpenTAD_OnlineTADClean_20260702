---
type: paper
node_id: paper:feel2026-force-egocentric
title: "FEEL (Force-Enhanced Egocentric Learning): A Dataset for Physical Action Understanding"
authors: ["Eadom Dessalene", "Botao He", "Michael Maynord", "Yonatan Tussa", "Pavan Mantripragada", "Yianni Karabati", "Nirupam Roy", "Yiannis Aloimonos"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2603.15847"
  doi: null
  s2: null
tags: ["egocentric-video", "force", "contact", "multimodal", "physical-state"]
added: 2026-07-11T00:00:00Z
---

# FEEL (Force-Enhanced Egocentric Learning): A Dataset for Physical Action Understanding

## One-line thesis

Provides force-synchronized egocentric video and force-aware learning tasks that connect visual observations to physical interaction signals.

## Problem / Gap

RGB video alone does not directly reveal contact force and some physical state transitions.

## Method

Collects synchronized egocentric imagery and force signals, with contact-oriented supervision and force-aware pretraining.

## Key Results

The dataset contains roughly three million frames and supports contact segmentation and force-related representation learning.

## Assumptions

Force/contact timestamps are relevant only to a subset of semantic events.

## Limitations / Failure Modes

It does not itself define physical-versus-visual-versus-commit timing or a general Online TAD benchmark.

## Reusable Ingredients

Privileged physical timestamps and synchronized video for a Three-Clock observability pilot.

## Open Questions

Whether its event taxonomy and release format permit stable `tau_phys` and prefix-level `tau_vis` annotation.

## Claims

None in this project.

## Connections

[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project

Makes T02 empirically plausible without immediately building a new sensor rig, subject to a dataset audit.
