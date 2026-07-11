---
type: paper
node_id: paper:apt2026-atomic-physical-transitions
title: "APT: Atomic Physical Transitions for Causal Video-Language Understanding"
authors: ["Shang Wu", "Haoran Lu", "Songling Liu", "Chenwei Xu", "Lie Lu", "Pranav Maneriker", "Fan Du", "Manling Li", "Zhaoran Wang", "Han Liu"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2606.18586"
  doi: null
  s2: null
tags: ["physical-transition", "state-change", "temporal-localization", "causal-video", "VLM"]
added: 2026-07-11T00:00:00Z
---

# APT: Atomic Physical Transitions for Causal Video-Language Understanding

## One-line thesis

Represents physical events as ordered chains of temporally localized atomic state transitions tied to visible evidence, active physical mechanisms, and before/after regimes.

## Problem / Gap

Clip-level event names hide the physical transitions that make an event causally valid.

## Method

Builds human/simulator APT supervision and uses parameter-efficient VLM tuning to predict physical-transition chains without losing event-level capability.

## Key Results

Reports 14 transition types and 27,303 timed instances over 1,246 trials; zero-shot VLMs miss many transitions, while APT-Tune improves transition recall.

## Assumptions

Atomicity depends on the transition vocabulary and video timescale.

## Limitations / Failure Modes

The timing comes from human annotation or simulation rather than an independently measured physical sensor clock, and the paper does not decompose visual observability from model commit delay.

## Reusable Ingredients

Physical-transition taxonomy, timed transition instances, mechanism-conditioned state representation.

## Open Questions

Whether its timed labels align with sensor-defined physical events and whether the same transition has view-dependent evidence-arrival times.

## Claims

None in this project.

## Connections

[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project

Strongest threat to any claim that temporally localized physical state transitions are new. PIVOT must contribute independent physical anchoring and observability decomposition, not another transition vocabulary.
