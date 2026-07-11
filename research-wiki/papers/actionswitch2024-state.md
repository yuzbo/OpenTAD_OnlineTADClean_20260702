---
type: paper
node_id: paper:actionswitch2024-state
title: "ActionSwitch"
authors: []
year: 2024
venue: "ECCV"
external_ids:
  arxiv: "2407.12987"
  doi: null
tags: ["On-TAL", "state-change", "same-class", "concurrent-actions", "baseline"]
added: 2026-07-11
---

# ActionSwitch 2024

## One-line thesis

Model class-agnostic action-switch states to identify action instance boundaries and handle concurrent or same-class actions in online localization.

## Relevance to This Project

This is the most direct threat to "state changes define boundaries" as a novelty claim.

## Overlap

- state-change framing;
- boundary identification;
- overlapping/concurrent actions;
- same-class action issue;
- conservative state behavior.

## Key Difference We Need

CESR must go beyond state-change boundary:

- maintain revisable hypothesis tracks;
- revise start/end/class over time;
- commit only after sufficient evidence;
- evaluate revision trajectory and commit latency.

## Required Defensive Evidence

- same-class repeated action test;
- multi-slot or clearly bounded limitation;
- ablation against one-track/class state model;
- evidence that revision/commit improves over endpoint-only or state-change-only.
