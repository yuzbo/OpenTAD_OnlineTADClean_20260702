---
type: paper
node_id: paper:foo2023-system-adaptive
title: "System-Status-Aware Adaptive Network for Online Streaming Video Understanding"
authors: ["Lin Geng Foo", "Jia Gong", "Zhipeng Fan", "Jun Liu"]
year: 2023
venue: "CVPR"
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: ["adaptive-compute", "streaming-video", "latency", "dynamic-network"]
added: 2026-07-11T00:00:00Z
---

# System-Status-Aware Adaptive Network for Online Streaming Video Understanding

## One-line thesis

SAN dynamically selects resolution and network depth from video and device state to maintain online prediction quality under fluctuating compute budgets.

## Problem / Gap

Real device load changes over time, so fixed-cost online models can violate latency constraints.

## Method

A lightweight policy controls a dynamic main network, with self-supervised meta-adaptation to new hardware profiles.

## Key Results

Evaluated on online action recognition and online pose estimation with accuracy and delay metrics.

## Assumptions

Candidate resolutions/depths and system telemetry are available.

## Limitations / Failure Modes

It does not solve instance-level temporal action span localization.

## Reusable Ingredients

Hardware-aware policy and measured delay as part of the decision state.

## Open Questions

Whether compute can be allocated from localization risk rather than frame classification confidence.

## Claims

None in this project.

## Connections

[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project

Blocks generic adaptive-resolution/depth novelty and is a required baseline for active-sensing claims.
