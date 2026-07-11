---
type: idea
node_id: idea:evidence-risk-ontal
title: "Evidence-Aligned, Risk-Controlled Online Temporal Action Localization"
stage: superseded-umbrella
outcome: decomposed
updated: 2026-07-11
target_gaps: ["G10", "G11"]
---

# Evidence-Aligned, Risk-Controlled Online Temporal Action Localization

## One-Line Thesis

Online temporal localization should decide from the first causally sufficient visual evidence, not treat an annotator's physical action endpoint as the decision-ready timestamp, and should control false commits over an indefinitely monitored stream.

## Problem Anchor

Existing On-TAL evaluation usually assumes that the annotated end time is simultaneously:

1. the physical end of an action;
2. the first prefix at which the action identity and span are knowable;
3. the correct time to emit a detection.

These quantities need not coincide. Some actions are recognizable before motion cessation, while successful completion, failure, or interruption may only become visible later. A latency score measured only from the annotated endpoint therefore mixes visual observability, annotation convention, and model delay.

## Task Reformulation

For each action instance, retain the conventional physical interval `[s, e]` and additionally annotate an evidence-ready distribution or interval around `t*`, the earliest prefix at which independent observers can defend the class and completed span.

At every prefix, a model may maintain a mutable hypothesis. It may commit only when its sequential evidence passes a calibrated risk rule. The committed result remains immutable.

Primary outputs:

- action class or class set;
- temporal span or calibrated span set;
- evidence/uncertainty process;
- first valid commit time;
- abstention when evidence is insufficient.

## Dominant Contribution

The main contribution is the causal-observability task and evaluation protocol. The supporting method is an anytime-valid or sequentially calibrated commit rule attached to a causal temporal localizer.

Identity tracking, revision ledgers, and the current PCEH/CESR code are implementation substrate, not the headline claim.

## Minimal Method

1. A frozen or LoRA-adapted causal visual encoder produces prefix features.
2. A lightweight instance localizer maintains class/span beliefs.
3. A sequential evidence process accumulates support for a valid instance-level detection.
4. A calibration layer chooses commit thresholds under a declared false-commit or miss-risk target.

The first study should calibrate across independent videos. It must not claim distribution-free frame-level guarantees under arbitrary temporal dependence without a valid theorem.

## Required Evaluation

- standard On-TAL mAP/OnlineAP;
- delay from evidence-ready time `t*`, alongside delay from physical end `e`;
- false commits per video hour;
- risk-coverage and abstention curves;
- temporal interval coverage and width;
- incomplete/interrupted-action analysis;
- inter-annotator agreement and uncertainty for `t*`.

## Why This Is Not a Small Patch

- It challenges the target timestamp used by the task.
- It changes the output from an uncalibrated point decision to a risk-controlled sequential decision.
- It makes unsupported early outputs and avoidable late outputs separately measurable.
- It can reuse an inexpensive base detector; the scientific contribution does not require full-backbone training.

## Closest Threats

- Thinking-QwenVL and StreamReady model first-sufficient evidence or readiness for streaming VideoQA, but not instance-level action-span localization with formal false-commit control.
- Action Completion studies completion moments and incompletion, but not modern On-TAL instance detection and anytime risk.
- Boundary-uncertainty TAL models annotation ambiguity, but generally in offline proposal refinement.
- General conformal and anytime-valid risk-control methods provide statistical ingredients, not a video localization formulation.
- Munoz et al. WACV Workshops 2026 directly use an e-process with provable false-alert control for real-time video tracking-failure detection. This blocks novelty from "video score plus e-process" alone.
- Online video anomaly detection already studies sequential false-alarm bounds.

## Kill Criteria

- Human agreement on evidence-ready time is too low to define a stable target.
- A simple confidence threshold matches the calibrated method under the same risk constraint.
- The guarantee relies on assumptions violated by the benchmark and cannot be restated honestly.
- The new metric does not change model ranking or expose failures hidden by standard On-TAL metrics.

## Status

Superseded as a single umbrella after the Pro divergent review. Its two core questions now live separately in [anytime-semantic-event-alarms.md](anytime-semantic-event-alarms.md) and [three-clock-event-observability.md](three-clock-event-observability.md). Neither is yet selected.
