---
type: paper
node_id: paper:etad2023-efficient-e2e
title: "ETAD: Training Action Detection End to End on a Laptop"
authors: ["Shuming Liu", "Mengmeng Xu", "Chen Zhao", "Xu Zhao", "Bernard Ghanem"]
year: 2023
venue: "CVPR Workshop / arXiv"
external_ids:
  arxiv: "2205.07134"
  doi: null
tags: ["TAD", "end-to-end", "gradient-sampling", "proposal-sampling", "training-cost"]
added: 2026-07-11
---

# ETAD 2023

## One-line thesis

Raw-video TAD can be trained economically by sequentialized backpropagation, selective snippet gradients, and proposal sampling.

## Key Results

- About 30% snippet-gradient sampling retained nearly the full end-to-end result in the reported study.
- About 6% proposal sampling was sufficient in its detector.
- Random/grid sampling was more robust than label-guided or contiguous block sampling, which distorted proposal diversity.

## Relevance to This Project

ETAD is strong evidence that selective-gradient and proposal/event sampling can control visual training cost. It also warns that annotation-guided event sampling is not automatically unbiased. CRS-EPS must include a uniform chronological component, inclusion probabilities, calibration audits, and full-stream validation.

## Reusable Ingredients

- Selective visual backward while preserving broader forward context.
- Micro-batched sequential backpropagation.
- Combine temporal gradient sampling with mixed precision/checkpointing.

## Connections

[AUTO-GENERATED from graph/edges.jsonl - do not edit manually]
