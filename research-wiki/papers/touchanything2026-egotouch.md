---
type: paper
node_id: paper:touchanything2026-egotouch
title: "TouchAnything: A Dataset and Framework for Bimanual Tactile Estimation from Egocentric Video"
authors: ["Jianyi Zhou", "Ziteng Gao", "Feiyang Hong", "Zirui Liu", "Guannan Zhang", "Weisheng Dai", "Ruichen Zhen", "Chuqiao Lyu", "Haotian Wu", "Yinian Mao", "Xushi Wang", "Yuxiang Jiang", "Wenbo Ding", "Shuo Yang"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2605.13083"
  doi: null
  s2: null
tags: ["egocentric-video", "tactile", "pressure", "multi-view", "physical-supervision"]
added: 2026-07-11T00:00:00Z
---

# TouchAnything: A Dataset and Framework for Bimanual Tactile Estimation from Egocentric Video

## One-line thesis

Introduces EgoTouch, synchronized head/dual-wrist RGB, hand pose, and continuous bimanual pressure data, plus a vision-to-touch prediction baseline.

## Problem / Gap

Vision alone is ambiguous for contact, force, and pressure, while dense tactile hardware is expensive.

## Method

Collects 208 manipulation tasks over 1,891 episodes and predicts dense tactile signals from one or more egocentric views.

## Key Results

Reports improved contact and volumetric tactile IoU when wrist views supplement the head view.

## Assumptions

Pressure maps and synchronized views accurately reflect the physical interaction of interest.

## Limitations / Failure Modes

The paper does not define physical-to-visual observability intervals or online commit delay. The abstract promises future public release; current download availability requires verification.

## Reusable Ingredients

Ideal same-event multi-view plus physical-sensor structure for a PIVOT pilot.

## Open Questions

Whether the dataset is downloadable and whether its episodes include event labels suitable for physical transition timing.

## Claims

None in this project.

## Connections

[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project

Potentially the strongest PIVOT data source and also a threat to claiming multi-view physical grounding as method novelty.
