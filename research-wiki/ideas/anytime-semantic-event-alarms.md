---
type: idea
node_id: idea:anytime-semantic-event-alarms
title: "Anytime-Valid Semantic Event Alarms"
stage: proposed
outcome: conditional
updated: 2026-07-11
target_gaps: ["G11"]
---

# Anytime-Valid Semantic Event Alarms

## One-Line Thesis

Recast continuous semantic-video monitoring as sequential hypothesis testing that controls a declared long-horizon false-alarm risk while minimizing event-detection delay.

## What Is Actually New If It Survives

The project cannot claim novelty from merely attaching an e-process to a video score. A defensible contribution must solve a semantic-event-specific validity problem involving temporally dependent evidence, predictable candidate creation, repeated instances, reset/refractory logic, and possibly multiple classes or streams.

## Closest Threats

- WACV 2026 already uses an e-process with provable false-alert control for real-time object-tracking failure detection.
- WATCH provides weighted conformal martingales for online change monitoring.
- Online video anomaly detection already studies sequential false-alarm bounds.
- Operational video systems already use false alarms per camera hour.

## Required Task Decisions

- Define the null unit and filtration.
- Choose one primary risk target: ever-alarm probability, false alarms per hour, event-wise risk, online FDR, or renewal risk.
- Define candidate spawning and reset before using the data that trigger them.
- State dependence and drift assumptions explicitly.
- Use real long negatives for the final claim.

## Minimum Pilot

Use identical frozen causal scores for fixed threshold, matched empirical threshold, Bonferroni/alpha-spending, CUSUM/SPRT, WATCH-style martingale, and the proposed rule. Compare risk and delay across horizons with uncertainty intervals.

## Kill Criteria

- The route reduces to an existing e-process wrapper.
- Matched-risk simple baselines equal or beat delay and recall.
- Validity holds only on iid-like concatenated clips.
- The chosen infinite-horizon target causes practical alpha-death.
- Conventional and long-horizon evaluation do not change any scientific conclusion.

## Status

Conditional P0 candidate from the Pro divergent review. It is not the selected main route.
