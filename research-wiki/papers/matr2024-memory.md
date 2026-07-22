---
type: paper
node_id: paper:matr2024-memory
title: "Online Temporal Action Localization with Memory-Augmented Transformer"
authors: ["Youngkil Song", "Dongkeun Kim", "Minsu Cho", "Suha Kwak"]
year: 2024
venue: "ECCV"
external_ids:
  arxiv: "2408.02957"
  doi: null
tags: ["On-TAL", "memory", "start-retrieval", "baseline"]
added: 2026-07-11
---

# Online Temporal Action Localization with Memory-Augmented Transformer

## One-line thesis

Use current evidence to predict action end and retrieve action start from a historical memory queue.

## Relevance to This Project

MATR is a strong online span baseline. It shows that memory-based end-to-start localization is already covered.

Primary source: [arXiv:2408.02957](https://arxiv.org/abs/2408.02957). The
2026-07-20 FIXED/REMATCH readiness review correctly described MATR's role but
attached that sentence to the CAG-QIL CVF footnote; this page records the
correct direct source.

Official code: [skhcjh231/MATR_codebase](https://github.com/skhcjh231/MATR_codebase),
audited at `ba05a98d451b3541c1a5377026f17dc1102fa217` on 2026-07-22.

The released THUMOS14 training command uses the official RGB+flow feature
pickles and the defaults in `util/config.py`: 64-frame segments, 10 queries,
seven memory segments with `gap2`, batch 64, 100 epochs, seed 52, Adam from
`1e-8` to `1e-5` with cosine warm-up/restarts, focal classification, class
threshold 0.1 and NMS 0.3. These settings, rather than the former OpenTAD
12-epoch recipe, define the native parity and matched EventMATR experiments.

## Overlap

- online temporal action localization;
- historical memory;
- endpoint/current segment reasoning;
- start retrieval from past context.

## Key Difference We Need

CESR should not claim memory/history as novelty. The delta must be:

- revisable hypothesis state over time;
- explicit revision and commit ledger;
- start/ongoing/end lifecycle;
- boundary refinement trajectory before commit.

## Required Baseline Role

Use the exact native implementation as the primary parent and baseline. A
MATR-style rewrite is not sufficient evidence of fidelity.
