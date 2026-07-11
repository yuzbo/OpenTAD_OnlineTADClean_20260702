---
type: experiment_design
node_id: exp:pes-stage1-design-20260712
idea: idea:petal-ontad
status: implementing
verdict_owner: result-to-claim
updated: 2026-07-12
---

# Persistent Event-Set Stage-1 Kill-Test Design

## Goal

Determine whether persistent action-instance identity has measurable value in standard completion-triggered On-TAD after controlling features, compute, capacity, schedule, thresholds, and evaluator.

This experiment does not attempt to prove a raw-video or end-to-end representation-learning claim.

## Fixed Task Contract

At decision frame `t`, the detector receives only cached feature tokens whose source frames are `<= t`. It may maintain bounded latent state. It may emit zero or more completed instances:

```text
{start_frame, end_frame, class, score, emit_frame}
```

Required invariants:

- `end_frame <= emit_frame`;
- an emitted row is immutable;
- no model computation receives EOF, total duration, future feature tokens, GT, teacher predictions, or hidden suppression state;
- late predictions and duplicates remain visible to evaluation;
- training labels use a separate prefix-observable schedule and are never present during inference.

## Data Interface

Each feature cache has a manifest containing:

```text
encoder identifier and local checkpoint path/hash
source annotation hash
frame selection policy
feature stride and timestamp convention
dtype and channel dimension
per-video token count and source-frame list
```

The chronological feature dataset returns one contiguous chunk per item and a training-only schedule. A schedule row at time `t` may contain:

```text
births: instance id, class, observed start
active: instance id, class, observed start
ends: instance id, class, observed start, newly observed end
```

It contains no future endpoint for a currently active instance.

## Shared Model

All variants use the same input projection, bounded feature memory, query width, number of slots, self/cross-attention blocks, prediction heads, optimizer, and evaluator.

Outputs per slot:

```text
birth logit
alive logit
class logits
start estimate
first-event end hazard
retrospective endpoint offset
```

The first implementation uses one sequential slot scan per causal feature chunk. Cross-chunk query and feature state is detached. This supports inference continuity but does not claim cross-chunk BPTT.

## Controlled Variants

### Fresh Queries

Reset query embeddings at every decision step and rematch visible active instances. No identity is carried between prefixes. This is the rediscovery control.

### Temporal TrackFormer

Carry query embeddings across steps. Match new births to free queries using only current birth, class, and start evidence. Preserve the assignment until the observed endpoint. Use a scalar start-offset head and ordinary end/death supervision.

### Persistent Event-Set

Carry the same queries and assignment, but use a bounded distributional start pointer with a `before-memory` bin and an instance-aware first-event endpoint hazard. Emit exactly once per lifecycle and enter a refractory state before rearming.

This variant has no explicit duplicate loss and no NMS.

## Prefix-Observable Assignment

At time `t`, only newly visible births are matched to free slots. Matching cost may use:

- birth probability;
- class label after the annotated start becomes visible;
- observed start position;
- current query competition.

It may not use the future endpoint. Existing assignments remain fixed. If births exceed free slots, unmatched births increment slot-exhaustion FN.

Full-chunk trajectory matching is implemented only as a labeled privileged upper-bound ablation.

## Loss Contract

- birth BCE: free slots, with matched births positive;
- alive BCE: assigned active slots positive and free/refractory slots negative;
- class CE: assigned instances from observed start through endpoint;
- start loss: pointer CE or scalar offset regression according to variant;
- endpoint hazard BCE: only assigned instances currently in the risk set, with one positive at first endpoint crossing;
- endpoint offset regression: only at first endpoint crossing;
- no positive emission labels after the first endpoint event;
- no standalone duplicate or delay loss in Stage 1.

## Required Synthetic Cases

1. delayed endpoint between decision timestamps;
2. repeated same-class instances;
3. simultaneous same-class overlap;
4. action crossing a chunk boundary;
5. action starting before bounded memory;
6. more simultaneous instances than slots;
7. two slots attempting one event;
8. premature rearm;
9. future-feature perturbation;
10. prefix-cut invariance;
11. batched chunk scan versus stepwise scan;
12. no-EOF/no-duration inference-input audit;
13. immutable ledger replay.

## Experiment Matrix

All Stage-1 rows use the same cache and three seeds:

| Row | Query state | Start representation | End target | Assignment | Suppression |
|---|---|---|---|---|---|
| FRESH | reset each step | scalar | ordinary observed end | per-step visible | none |
| TTF | persistent | scalar | ordinary observed end | prefix birth + fixed identity | none |
| PES | persistent | pointer | first-event hazard | prefix birth + fixed identity | none |
| PES-IDSHUF | shuffled carried identity | pointer | first-event hazard | deliberately corrupted | none |
| PES-FULLMATCH | persistent | pointer | first-event hazard | privileged full trajectory | none |

## Metrics

- average mAP across tIoU 0.3:0.7 and mAP@0.7;
- recall and FN at preregistered score thresholds;
- duplicate and fragmentation rates;
- same-class overlap and adjacency subsets with sample counts;
- chunk-crossing identity survival and start error;
- slot-exhaustion count;
- endpoint-to-emission delay against matched GT;
- emitted proposals per video;
- p50/p90/p95 model latency, peak VRAM, wall time, optimizer events, and GPU-hours.

Thresholds are selected on validation only and then frozen. Results are paired by video and seed.

## Kill Gate

The persistence claim is stopped if any condition holds:

- TTF is equivalent or stronger than PES within paired uncertainty;
- PES mean average-mAP gain over both FRESH and TTF is below 2.0 and claimed error reduction is below 20%;
- recall drops by more than 1.0 absolute point;
- same-class overlap or chunk-crossing behavior does not improve;
- PES requires current/history NMS or hidden duplicate suppression;
- identity-shuffled queries do not degrade the claimed identity-sensitive metrics;
- the route only works with privileged full-trajectory matching;
- Stage-1 total training exceeds 10 GPU-hours without prior authorization.

The numeric effect thresholds are project resource gates, not universal statistical laws.

## Deployment Boundary

- Use a fresh remote checkout; never reset or overwrite the dirty N16R4 checkout.
- Store cache and runs only below `/data/run01/sczc063/yuzibo`.
- Submit GPU work through Slurm.
- Run cache extraction once, then freeze its manifest and hash for every row.
- Smoke on a small allow-list before the three-seed pilot.
- Do not launch raw-video visual adaptation from this design.

## Connections

Relationships are stored in `research-wiki/graph/edges.jsonl`.
