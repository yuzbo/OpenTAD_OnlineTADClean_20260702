---
type: idea
node_id: idea:risk-constrained-duty-cycling
title: "Risk-Constrained Camera Duty Cycling"
stage: proposed
outcome: hold
updated: 2026-07-11
target_gaps: ["G12"]
---

# Risk-Constrained Camera Duty Cycling

## One-Line Thesis

Choose whether and how to acquire pixels before capture, minimizing sensor-to-model energy subject to missed-event, false-alarm, and delay constraints.

## Distinction

Frame selection, token pruning, and dynamic backbones usually save compute after capture. This route is only distinct if it measures and controls sensor, ISP, encoding, transfer, and wake-up cost before observation.

## Required Evidence

- Hardware power decomposition.
- A low-power sentinel whose own information cost is counted.
- Matched recall and delay against periodic sampling, motion thresholds, AdaFrame-like selection, and SAN-like adaptation.
- Explicit minimum-duration or observability assumptions.

## Kill Criteria

- Capture cost is negligible relative to inference.
- Real-device gains disappear compared with replay.
- Short events are systematically missed.
- The guarantee depends on oracle motion or event timing.

## Status

Hold. Hardware and assumption gates precede any model work.
