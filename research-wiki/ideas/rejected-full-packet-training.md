---
type: idea
node_id: idea:rejected-full-packet-training
title: "Rejected Default: Full-Packet Streaming Training"
stage: archived
outcome: negative
updated: 2026-07-11
target_gaps: ["G4"]
---

# Rejected Default: Full-Packet Streaming Training

## Rejected Thesis

Train on every chronological packet as the default formal training protocol.

## Why Rejected

Cost is unacceptable:

- about 152,670 packets per epoch;
- about 7 hours per epoch;
- about 210 GPU-hours per model per seed for 30 epochs;
- multi-seed PCEH vs endpoint-only would be far too expensive before ablations.

It also wastes compute on many low-information background/ongoing packets.

## Allowed Uses

- 5-10 video gold subset;
- no more than about 10,000 packets for sampled-objective validation;
- best checkpoint short full-stream continuation/audit;
- final chronological evaluation.

## Replacement

CRS-EPS event-centric prefix episodes plus feature cache.
