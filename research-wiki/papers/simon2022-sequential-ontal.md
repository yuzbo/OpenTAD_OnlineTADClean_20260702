---
type: paper
node_id: paper:simon2022-sequential-ontal
title: "SimOn: A Simple Framework for Online Temporal Action Localization"
authors: ["Tuan N. Tang", "Jungin Park", "Kwonyoung Kim", "Kwanghoon Sohn"]
year: 2022
venue: "arXiv"
external_ids:
  arxiv: "2211.04905"
  doi: null
tags: ["On-TAL", "sequential-prediction", "end-to-end-head", "past-context", "efficient-training"]
added: 2026-07-11
---

# SimOn 2022

## One-line thesis

A lightweight Transformer can predict per-class online action states from the current feature and a short history of visual and prediction contexts, with state changes grouped into action instances.

## Method

- Query: current pre-extracted visual feature.
- Context: a short queue of past visual/prediction embeddings plus a learned recurrent context embedding.
- Training: focal loss at every time step, batch size 256 for 16 epochs.
- Decoding: thresholded class-state changes define action starts and ends.

## Key Results

The model reports 3.94M parameters and 5.2 ms inference on its setup, showing that lightweight feature-based online training can be inexpensive.

## Limitations / Failure Modes

- Uses pre-extracted TSN features rather than raw-video visual adaptation.
- Same-class adjacent actions can merge.
- Does not maintain an explicit identity-linked boundary distribution or a revision/commit ledger.

## Relevance to This Project

SimOn directly blocks claims that causal sequential prediction, recurrent semantic state, or efficient end-to-end On-TAL head training is new. A CESR model must demonstrate identity consistency and trajectory-level gains beyond a SimOn-style state sequence.

## Connections

[AUTO-GENERATED from graph/edges.jsonl - do not edit manually]
