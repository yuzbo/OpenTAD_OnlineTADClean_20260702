---
type: paper
node_id: paper:streamformer2025-streaming-representation
title: "Learning Streaming Video Representation via Multitask Training"
authors: ["Yibin Yan", "Jilan Xu", "Shangzhe Di", "Yikun Liu", "Yudi Shi", "Qirui Chen", "Zeqian Li", "Yifei Huang", "Weidi Xie"]
year: 2025
venue: "ICCV"
external_ids:
  arxiv: "2504.20041"
  doi: null
  s2: null
tags: ["streaming-video", "causal-backbone", "multitask-pretraining", "online-action-detection"]
added: 2026-07-11T00:00:00Z
---

# Learning Streaming Video Representation via Multitask Training

## One-line thesis

StreamFormer converts a pretrained image transformer into a causal streaming backbone and trains it with global, temporal, and spatial video-language objectives.

## Problem / Gap

Offline bidirectional video backbones do not naturally provide efficient frame-wise causal representations.

## Method

Causal temporal attention and spatial low-rank adaptation are combined with multitask supervision spanning recognition, retrieval, TAL, temporal grounding, and segmentation.

## Key Results

The frozen backbone is evaluated on online action detection, online video instance segmentation, and video question answering.

## Assumptions

The work uses multiple annotated datasets and substantial multitask pretraining.

## Limitations / Failure Modes

Its downstream action task is frame-level OAD rather than strict instance-level On-TAL with an immutable detection ledger.

## Reusable Ingredients

Causalizing a pretrained image backbone and alternating multi-granularity tasks.

## Open Questions

Whether its representations provide calibrated instance boundary evidence under strict On-TAL evaluation.

## Claims

None in this project.

## Connections

[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project

This paper blocks a generic claim that streaming video pretraining or causal foundation-backbone adaptation is an unoccupied direction.
