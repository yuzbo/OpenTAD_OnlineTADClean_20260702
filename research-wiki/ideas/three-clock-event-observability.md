---
type: idea
node_id: idea:three-clock-event-observability
title: "PIVOT: Physically Anchored, Interval-Valued Visual Observability Timing"
stage: rejected-out-of-scope
outcome: negative
updated: 2026-07-12
target_gaps: ["G10"]
---

# PIVOT: Physically Anchored, Interval-Valued Visual Observability Timing

## One-Line Thesis

Measure a view-conditioned observability gap between an independently sensed physical transition and the earliest population-level visual verification interval, then evaluate only the residual delay of causal model decisions.

## Correct Task Identity

The first paper should be framed as **Physically Anchored Streaming Event Verification**, not generic Online TAL. It verifies a known physical state transition or outcome from causal RGB prefixes. Action class and historical span are secondary outputs.

## Scientific Objects

```text
C_phys              sensor-anchored physical transition interval
C_vis(view,q,Pi,eta) population visual-verifiability interval
tau_commit(model)    first immutable correct causal decision
```

Primary decomposition:

```text
Delta_obs = tau_vis - tau_phys
Delta_alg = tau_commit - tau_vis
```

Video-time delay and wall-clock inference latency remain separate.

## Defensible Delta

All four qualifiers are necessary:

- independently sensed physical event;
- view-conditioned evidence arrival;
- population-level interval rather than a subjective point;
- residual model delay after evidence availability.

Without them, the route is largely reconstructable from PaSBench, APT, Ego4D PNR, StreamReady, TouchMoment, and privileged-sensor video learning.

## Minimal Method

Use a frozen causal encoder and an ordered multi-state observer:

```text
S0 pre-physical
S1 physical event occurred but not visually verifiable
S2 visually verifiable
```

Train physical and visual transition hazards with interval-censored likelihoods. Use a validation-calibrated deterministic commit rule. The method is supporting evidence, not the main novelty.

## Strongest Threats

- PaSBench already aligns first visible danger, accident boundaries, and causal warnings.
- APT already temporally localizes atomic physical transitions tied to visual evidence and mechanisms.
- Ego4D PNR already localizes state-change keyframes with PRE/CONTACT/PNR/POST.
- StreamReady and Thinking-QwenVL already model evidence windows and first-sufficient response time.
- TouchMoment already detects precise contact moments.
- FEEL, EgoTouch, and EgoTactile already provide or propose synchronized force/tactile video supervision.
- STARE and Towards Streaming Perception already make latency-aware evaluation and ranking reversal established ideas.

## Data Gate

FEEL's official project page currently exposes placeholder Dataset/Code links that return 404. TouchAnything states that EgoTouch will be released, but current downloadable access was not verified. The route therefore requires one of:

1. verified public FEEL/EgoTouch/EgoTactile access;
2. author collaboration;
3. a controlled 100-300-instance synchronized multi-view pilot.

Ego4D PNR and APT cannot replace the independent physical clock.

## Required Evidence

1. At least three event types have a reproducible gap larger than one reliable sampling bin.
2. Population `C_vis` survives rater holdout, wording, and threshold sensitivity.
3. The same physical event has view-dependent `C_vis` while `C_phys` remains fixed.
4. Raw endpoint latency and decomposed algorithmic latency change a model ranking or expose a stable new failure.
5. An ordered observer improves the matched-early-error frontier over endpoint-only and readiness-only baselines.

## Kill Criteria

- Physical/visual gap is negligible.
- `C_vis` cannot be identified as an interval.
- Cross-view observability differences are absent.
- No metric ranking or conclusion changes.
- The route reduces to PaSBench + StreamReady + sensor labels.
- Public data are unavailable and controlled collection is infeasible.
- Results hold only for contact onset.
- A simple readiness or endpoint threshold dominates.

## Status

Rejected by the user on 2026-07-12 because the task is physically anchored streaming event verification rather than standard Online Temporal Action Detection/Localization. Do not revive it as the main project route. Its observability analysis may remain background material only. Full design: [`../../THREE_CLOCK_TASK_METHOD_DESIGN_20260711.md`](../../THREE_CLOCK_TASK_METHOD_DESIGN_20260711.md). Competition audit: [`../../THREE_CLOCK_COMPETITION_REVIEW_20260711.md`](../../THREE_CLOCK_COMPETITION_REVIEW_20260711.md).

## Rejection Reason

The project must innovate inside the established On-TAD task: raw streaming video, no future observations, and instance-level `{start, end, class, score}` output when an action completes. PIVOT changes the scientific target, annotation object, data requirements, and evaluation protocol. Its novelty therefore does not answer the selected project question even if the measurement itself is valid.
