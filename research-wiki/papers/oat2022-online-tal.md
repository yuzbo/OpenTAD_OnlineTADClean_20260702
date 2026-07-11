---
type: paper
node_id: paper:oat2022-online-tal
title: "A Sliding Window Scheme for Online Temporal Action Localization"
authors: []
year: 2022
venue: "ECCV"
external_ids:
  arxiv: null
  doi: null
tags: ["On-TAL", "early-proposal", "boundary-refinement", "baseline"]
added: 2026-07-11
---

# OAT / Sliding Window Scheme for Online Temporal Action Localization

## One-line thesis

Online TAL can generate action proposals before full action completion and refine boundaries using online-visible windows.

## Relevance to This Project

This is one of the closest threats to naive "early hypothesis" and "boundary refinement" claims. If CESR only says "we update boundaries before final output", OAT may already cover enough of that idea.

## Overlap

- online TAL setting;
- early action proposals;
- boundary/offset refinement;
- online metric emphasis.

## Key Difference We Need

CESR must emphasize:

- mutable hypothesis lifecycle;
- timestamped revision history;
- explicit commit policy;
- state-transition objectives, not just proposal offsets;
- revision metrics in addition to final AP.

## Reviewer Warning

Never claim early proposal or online boundary refinement alone as novelty.
