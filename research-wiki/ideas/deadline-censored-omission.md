---
type: idea
node_id: idea:deadline-censored-omission
title: "Deadline-Censored Omission Monitoring"
stage: proposed
outcome: hold
updated: 2026-07-11
target_gaps: ["G14"]
---

# Deadline-Censored Omission Monitoring

## One-Line Thesis

Monitor obligations of the form “after trigger P, event Q must occur by deadline d,” and emit the earliest legally defensible omission or irrecoverable-violation decision.

## Distinction

This is not ordinary positive action detection. Without a deadline or an irreversible impossibility state, the non-occurrence of an event in an infinite stream is not identifiable.

## Closest Threats

Procedure anticipation, PREGO, EgoProactive, mistake/recovery recognition, temporal-logic monitoring, and task-graph methods are close. The route survives only if earliest legal omission time is a genuinely missing target and natural omission data exist.

## Kill Criteria

- Existing proactive/procedure baselines already express the same decision.
- Deadlines are arbitrary rather than externally grounded.
- Natural omissions are too rare and synthetic deletion is detectable from editing artifacts.
- Predicate errors erase any benefit of the timed monitor.

## Status

Hold after the Pro Top-5 review; closest-task and data audits are required before implementation.
