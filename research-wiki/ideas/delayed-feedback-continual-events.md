---
type: idea
node_id: idea:delayed-feedback-continual-events
title: "Delayed-Feedback Continual Event Learning"
stage: proposed
outcome: hold
updated: 2026-07-11
target_gaps: ["G13"]
---

# Delayed-Feedback Continual Event Learning

## One-Line Thesis

Define true online video learning in which immutable event outputs receive delayed, incomplete corrections and model updates may improve future predictions only.

## Core Protocol

Each output records event ID, prediction time, model version, and immutable result. Later feedback points to an earlier event, but updates are evaluated only on subsequent stream segments using prequential utility, adaptation lag, forgetting, update cost, and risk.

## Novelty Boundary

Continual learning, test-time adaptation, replay, rollback, and adapters are established. The only defensible delta is the feedback-timed causal protocol with realistic event-level delayed corrections and immutable historical outputs.

## Kill Criteria

- No realistic source of delayed feedback exists.
- Results depend on an arbitrary simulated delay/noise distribution.
- Periodic finetuning is equally effective.
- Adaptation gains are small relative to forgetting or false-alarm growth.
- Safety checks reject nearly every update and reduce the method to a frozen model.

## Status

Hold. It requires a credible chronological feedback stream before method design.
