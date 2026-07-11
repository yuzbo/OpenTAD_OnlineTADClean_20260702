---
type: idea
node_id: idea:state-transition-sampling
title: "Start/End State-Transition Sampling"
stage: active
outcome: pending
updated: 2026-07-11
target_gaps: ["G2", "G4"]
---

# State-Transition Sampling

## One-Line Thesis

Sampling must focus on both action start and action end, because online event-state tracking requires creating, updating, and committing hypotheses.

## Discussion Record

The user asked: if endpoint-near sampling is important, what about the start stage? The answer became a key correction:

> Start is not auxiliary. It is required for active-state creation, rearm, and avoiding false early triggers.

## Sampling Around Start

```text
s-4B, s-2B, s-B, s, s+B, s+2B, s+4B
```

Meanings:

- pre-start hard background: avoid premature hypothesis creation;
- start crossing: learn transition from background to active;
- early ongoing: keep active state stable and avoid early completion.

## Sampling Around End

```text
e-4B, e-2B, e-B, e, e+B, e+2B
```

Meanings:

- pre-end ongoing: avoid premature endpoint;
- endpoint crossing: learn completion;
- post-end: learn pending commit and late/missed states.

## Why Selected

This aligns the training distribution with the CESR story. It avoids the endpoint-only trap in which the model can close actions but cannot robustly open or maintain them.

## Required Tests

- Start target is stable under future suffix perturbation.
- Pre-start bins do not become active due to future GT.
- Same-class second instance can rearm after first commit.
- Ongoing bins are not mislabeled as completed.
