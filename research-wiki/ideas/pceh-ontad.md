---
type: idea
node_id: idea:pceh-ontad
title: "PCEH-OnTAD: Prefix-Censored Event-Emission Hazard Learning"
stage: partial
outcome: mixed
updated: 2026-07-11
target_gaps: ["G3", "G7"]
---

# PCEH-OnTAD

## One-Line Thesis

Separate physical endpoint time from model emission time and learn endpoint hazard plus bounded-delay first-emission policy under prefix-causal observation.

## Original Appeal

PCEH was selected after Pro reviews because it identified a real gap:

> Online TAD needs to separate physical event time, prefix-censored state, and system emission time.

This was stronger than simply claiming a causal backbone, VLM, adapter, or distillation route.

## Why It Was Demoted

The user later identified a better story: online systems should revise hypotheses as state changes, not only emit after action end. This makes PCEH too narrow as the whole paper framing.

Also, Pro review 2026-07-11 found the current implementation is not a valid PCEH method yet:

- endpoint and emission labels are currently the same event;
- late prefixes are repeated emission positives;
- targets are class-level rather than instance-level;
- predicted end equals emit in the decoder;
- GT is too close to model metadata.

## Current Role

PCEH is retained as a component inside CESR:

- endpoint transition / endpoint hazard;
- completed-pending-commit state;
- commit or first-emission policy;
- latency-budget penalty.

It is no longer the full paper story.

## Do Not Claim

- statistically independent endpoint/emission hazard before risk-set repair;
- bounded-delay first-emission learning if `emit_target = end_event` or late positives remain;
- low-latency final method before full chronological validation.

## Required Repair

1. Instance-aware endpoint risk set.
2. Separate commit/emission risk set.
3. Post-event mask.
4. Remove repeated late-positive emission.
5. Freeze predicted endpoint before commit.
6. Add delayed synthetic `predicted_end < commit_time`.

## Key Baseline

Endpoint-only head with the same architecture and training protocol. If PCEH does not beat endpoint-only on AP-latency Pareto, stop claiming it.
