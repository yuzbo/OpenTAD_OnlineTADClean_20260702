---
type: idea
node_id: idea:cesr-ontad
title: "Identity-Preserving Causal Belief Refinement and Utility-Based Commit for Online Temporal Action Detection"
stage: hold
outcome: pending
updated: 2026-07-11
target_gaps: ["G1", "G2", "G3", "G7", "G8"]
---

# Identity-Preserving Causal Belief Refinement for Online TAD

## Strategic Status

Demoted on 2026-07-11 from prospective paper headline to implementation substrate and fallback baseline. The identity ledger and causal protocol remain useful, but the route is too close to the combined territory of CAG-QIL, OAT, MATR, ActionSwitch, ProTAS, and readiness-aware streaming video work to justify expensive training by itself.

## One-Line Thesis

Online TAD should maintain an identity-preserving belief trajectory for each action instance and learn when to irreversibly commit that trajectory under an explicit localization-quality versus latency utility.

## Problem

The previous framing, "emit `{start,end,class,score}` as soon as the action ends and never revise", is too rigid. It makes the system look like an alarm rather than a streaming video understanding model. The user explicitly requested a more elegant formulation: when observed state changes, such as action start or action end, the model should make judgments and should be able to optimize or correct earlier online outputs as more prefix evidence arrives.

## Core Design

Two output layers:

| Layer | Mutable? | Purpose |
|---|---|---|
| online hypothesis | yes | Identity-linked distribution over state, class, start, end, confidence, and uncertainty. |
| committed detection | no | Final ledger row after commit; used for formal OnlineAP and latency evaluation. |

State lifecycle:

```text
background
-> possible_start
-> active / ongoing
-> ending / endpoint_suspected
-> completed_pending_commit
-> committed_absorbing
```

The key protocol change is:

> Revisions before commit are allowed, but every revision must be timestamped and auditable. Committed detections remain immutable.

The identity requirement is essential: repeated proposals at different timestamps do not count as a belief trajectory unless the method explicitly associates them with the same action instance or slot.

## Why Selected

- More expressive than one-shot endpoint emission.
- Naturally covers start-stage sampling and rearm behavior.
- Gives a clearer paper story than PCEH alone.
- Creates a measurable object absent from ordinary repeated-proposal pipelines: an identity-linked belief trajectory.
- Lets the evaluation measure revision quality, revision latency, commit latency, and final AP.

## What Is Not Novel by Itself

- Boundary refinement: many offline and OAT-style methods do this.
- State transition: ActionSwitch directly uses state changes.
- Memory/history: MATR/HAT already cover history-enhanced On-TAL.
- Early proposal: OAT is close.
- Sequential decision history: CAG-QIL already formulates On-TAL as an MDP.
- Lightweight recurrent online semantic state: SimOn already predicts class states from past visual/prediction context.
- Ongoing action progress and prediction refinement: ProTAS already covers this in online segmentation.
- Generic hypothesis revision and response timing in streaming video: Thinking-QwenVL and StreamReady are direct adjacent threats.
- Hierarchical streaming event semantics: OpenHOUSE already combines On-TAL with open-ended event descriptions.

## Novelty Must Be

The broad slogan "track, refine, and commit" is not sufficient. The surviving claim package is:

1. Identity-preserving belief trajectories as a formal, observable On-TAL object.
2. A risk-set utility-based first-commit policy, not a fixed confidence threshold.
3. Revision-ledger metrics that expose trajectory quality hidden by final mAP.
4. Full chronological evaluation where final committed rows are immutable.

The lifecycle labels and CRS-EPS sampler are enabling machinery, not headline novelty.

## Utility-Based First Commit

For an instance belief at decision time t, define a commit cost:

    C_commit(t) = L_det(H_t, y) + lambda_delay * max(0, t - e_gt)
                  + lambda_instability * I[unstable trajectory]

The commit head defines a discrete first-stop distribution:

    P(T=t) = h_t * product_{u<t}(1-h_u)

and learns to minimize expected commit cost over the valid post-end risk set. At inference, the hazard uses only prefix evidence. Training may use full annotations to construct the oracle cost/risk set and must disclose that fact.

Required comparisons:

- fixed confidence threshold;
- fixed wait-k after endpoint;
- confidence-only stopping;
- uncertainty-only stopping;
- oracle stopping upper bound;
- the same commit policy attached to strong OAT/MATR/ActionSwitch-style representations.

## Required Loss Families

- start transition loss;
- action-state loss;
- endpoint transition loss;
- boundary refinement loss;
- revision stability or calibration loss;
- commit/confirmation loss;
- latency-budget loss.
- identity association and trajectory consistency loss.

## Evaluation Requirements

- final OnlineAP under latency budgets;
- commit latency against matched GT end;
- revision latency and number of revisions per instance;
- boundary error over revision time;
- false start / false active / false commit;
- TP count, recall, FN, late FP;
- immutable committed ledger audit.

## Risks

- If revisions are not logged, reviewers will call it illegal post-hoc modification.
- If committed detections can change, it violates online evaluation.
- If the model only copies endpoint into commit, it collapses back to PCEH/one-shot.
- If state-change improvement is not shown over ActionSwitch-like baselines, novelty is weak.
- If identity-linked tracking is not better than matching repeated OAT-style proposals after the fact, the identity claim fails.
- If utility stopping does not beat fixed threshold/wait-k on the same beliefs, the method collapses to bookkeeping.
- If the paper drifts into generic online video semantic maintenance, OpenHOUSE, Thinking-QwenVL, StreamReady, and related streaming VLM work make the framing non-novel.

## Next Evidence Gate

Before coding large training:

1. Write a protocol spec with identity, association, revision, and commit invariants.
2. Add delayed synthetic cases where start/end/class beliefs improve over time before commit.
3. Implement revision metrics and fixed-policy stopping baselines before a large model.
4. Test utility stopping on cached features against fixed threshold and wait-k.
5. Keep PCEH endpoint/commit hazard as the first-stop module, not the full story.
