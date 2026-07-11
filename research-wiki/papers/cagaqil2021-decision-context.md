---
type: paper
node_id: paper:cagaqil2021-decision-context
title: "CAG-QIL: Context-Aware Actionness Grouping via Q Imitation Learning for Online Temporal Action Localization"
authors: ["Hyolim Kang", "Kyungmin Kim", "Yumin Ko", "Seon Joo Kim"]
year: 2021
venue: "ICCV"
external_ids:
  arxiv: null
  doi: null
tags: ["On-TAL", "MDP", "imitation-learning", "decision-state", "start-end"]
added: 2026-07-11
---

# CAG-QIL 2021

## One-line thesis

Online TAL can be formulated as sequential actionness grouping whose current decision depends on past actionness and decision histories, and can be trained with Q imitation learning.

## Problem / Gap

Naive grouping of frame-level actionness causes short false action ticks and fragmentation because the model ignores its own decision history.

## Method

- Represent grouping as an MDP.
- Maintain actionness and decision queues.
- Use binary state changes to create action starts and ends.
- Train the grouping policy with a hard-Q variant of SQIL.

## Assumptions

The original grouping route assumes limited overlap and only emits a completed action instance when the active decision returns to background.

## Limitations / Failure Modes

- The active interval is primarily a binary grouping state, not a calibrated distribution over an identity-linked temporal span.
- It does not expose a revision trajectory for start, end, class, and uncertainty.
- Its final output remains a one-shot completed instance.

## Relevance to This Project

This is a direct threat to any claim that an MDP, decision history, lifecycle state, or start/end state transition is novel. CESR must go beyond CAG-QIL through identity-preserving belief trajectories, explicit revision records, and a commit risk objective.

## Connections

[AUTO-GENERATED from graph/edges.jsonl - do not edit manually]
