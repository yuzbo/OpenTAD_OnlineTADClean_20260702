---
type: paper
node_id: paper:touchmoment2026-contact
title: "Detecting Precise Hand Touch Moments in Egocentric Video"
authors: ["Huy Anh Nguyen", "Feras Dayoub", "Minh Hoai"]
year: 2026
venue: "CVPR Findings"
external_ids:
  arxiv: "2604.12343"
  doi: null
  s2: null
tags: ["egocentric-video", "contact", "event-spotting", "precise-timing", "hand-object"]
added: 2026-07-11T00:00:00Z
---

# Detecting Precise Hand Touch Moments in Egocentric Video

## One-line thesis

Detects exact hand-object contact moments in egocentric video using hand-informed context, grasp-aware training, and soft temporal labels.

## Problem / Gap

Contact timing is visually subtle because of occlusion, near-contact motion, and first-person camera dynamics.

## Method

Introduces HiCE and the TouchMoment dataset with 4,021 videos and 8,456 contact moments.

## Key Results

Reports strong gains under a strict two-frame tolerance event-spotting criterion.

## Assumptions

The annotated contact moment is a valid target for physical contact.

## Limitations / Failure Modes

No independent force/tactile clock, population visual-observability interval, or model-commit decomposition.

## Reusable Ingredients

Contact event taxonomy, strict timing tolerance, hand-centric baseline.

## Open Questions

Whether its ground-truth moment matches an instrumented physical contact time and how visibility changes by view.

## Claims

None in this project.

## Connections

[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project

Makes contact-only PIVOT scientifically insufficient; the task must include delayed-visibility event types and physical instrumentation.
