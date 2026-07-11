---
type: idea
node_id: idea:petal-ontad
title: "PETAL-OnTAD: Persistent Event Tracking for End-to-End Online Temporal Action Detection"
stage: lead-candidate
outcome: pending
updated: 2026-07-12
target_gaps: ["G4", "G5", "G6", "G8", "G15"]
---

# PETAL-OnTAD

## One-Line Thesis

Train a raw-video causal backbone and persistent event queries jointly so that one query tracks one action instance from start evidence to end evidence, while prefix-parallel training is exactly equivalent to incremental cached inference.

## Task Boundary

The task remains standard fully supervised On-TAD/On-TAL:

- input is an untrimmed RGB stream;
- the model may use only frames at or before the current time;
- output is an action instance `{start, end, class, score}` when its end is detected;
- emitted detections cannot be revised or deleted;
- no new sensor, annotation type, task output, or evaluation target is introduced.

Pre-end event queries are latent model state, not additional user-visible predictions and not a task redefinition.

## Core Diagnosis

Current On-TAD methods largely combine frozen/pre-extracted TSN, I3D, or SlowFast features with independent sliding-window or frame-grouping detectors. This creates three coupled bottlenecks:

1. the visual backbone is not adapted to causal instance boundaries;
2. the same action is rediscovered in each window instead of retaining an identity;
3. overlapping windows repeat visual computation and require grouping or online NMS.

MATR calls its detection architecture end-to-end but freezes two-stream TSN/I3D in its experiments and its public code consumes pickle features. StreamFormer and E2E-LOAD provide raw-video causal backbones for frame-level OAD, not instance-level On-TAD. Offline E2E-TAD, TIA/AdaTAD, LoSA, and Re2TAL adapt raw-video backbones but use future context.

## Method

### 1. Causal Raw-Video Encoder

Encode each newly arrived tubelet once. Temporal attention or state-space mixing is strictly causal. Training processes a chronological chunk with a causal mask; inference appends only new tubelets using KV/SSM state. The raw RGB encoder, causal temporal adapters, event tracker, and output heads share one optimization graph.

### 2. Persistent Event Queries

Maintain `K` event slots. A free slot can activate near an action start, then carries an identity-linked class posterior, start distribution, action state, and end probability until the instance ends. Multiple slots allow overlapping and repeated same-class instances.

Lifecycle:

```text
free -> active -> ending -> emitted -> rearmed
```

Each slot emits at most one immutable detection per lifecycle. No classwise grouping or online NMS is required for the main model.

### 3. Trajectory-Level Assignment

Match ground-truth instances to query trajectories once over a causal training chunk, rather than running independent Hungarian matching at every prefix. The same slot must explain the same instance across all visible prefixes. Losses cover lifecycle state, class, start distribution, observed endpoint, and duplicate suppression.

### 4. Prefix-Parallel Training, Incremental Inference

Arrange frame/tubelet tokens and event-query tokens under a block-causal mask. One batched forward computes supervision for all prefix endpoints in the chunk. At inference, the identical network executes one step at a time with cache state. A mandatory prefix-equivalence test compares batched and stepwise outputs at every cut.

This removes the current pattern of one video seek, one processor call, one backward pass, and one optimizer step for every 8-frame packet.

## Candidate Contributions

1. A strict raw-video, instance-level, jointly trainable On-TAD model rather than a feature-level detector or frame-level OAD model.
2. Persistent identity-bearing event queries with trajectory-level assignment, replacing window-by-window rediscovery and post-hoc grouping.
3. Prefix-parallel causal training with auditable equivalence to cached streaming inference, reducing redundant computation without changing the online task.

All first-work wording remains unproven until a dedicated Pro novelty review checks tracking-query, streaming-DETR, OAD, On-TAD, and end-to-end TAL literature.

## Closest Work and Required Delta

- **MATR/HAT/OAT:** instance-level On-TAD, but feature-based window/anchor processing; queries or anchors are not persistent instance tracks.
- **ActionSwitch:** handles overlap through anonymous finite-state switches, but uses pre-extracted features and a separate classifier; it does not jointly learn identity-linked interval trajectories.
- **E2E-LOAD:** raw-video end-to-end online action detection, but frame-level OAD rather than action-instance localization.
- **StreamFormer:** causal streaming raw-video representation, but downstream backbone is frozen and evaluation is frame-level OAD.
- **E2E-TAD/TIA/LoSA/Re2TAL:** raw-video backbone adaptation for offline TAL; future-aware processing violates On-TAD inference.
- **TrackFormer/online VIS:** persistent object queries are an architectural precedent; PETAL must prove that trajectory-level temporal interval supervision and prefix-equivalent training create more than a direct tracker transplant.

## Minimal Experiment Ladder

1. Feature-level pilot: persistent trajectories versus a matched fresh-query/window detector on identical features.
2. Raw-video PEFT: causal backbone adapters plus the same tracker, with optimizer/gradient/parameter-delta audits.
3. Optional top-block unfreezing only if PEFT shows a localization gain.
4. Full chronological single-rank evaluation with no future access and immutable emissions.

Primary datasets: THUMOS14 for direct comparison, then FineAction or MultiTHUMOS for dense/overlapping stress tests. MUSES is conditional on data access.

## Required Baselines and Ablations

- OAT, MATR, HAT, ActionSwitch, and the repository's strict endpoint control;
- frozen causal backbone plus fresh per-window queries;
- frozen causal backbone plus persistent queries;
- trainable causal backbone plus fresh queries;
- full PETAL;
- per-prefix matching versus trajectory-level matching;
- one slot versus multiple slots;
- with and without online NMS/grouping.

Report standard On-TAD mAP, F1/precision/recall, endpoint delay, duplicate rate, same-class overlap performance, throughput, memory, and training cost. Supplemental metrics do not redefine the task.

## Go / Kill Gates

Proceed only if persistent queries improve a matched feature-level detector before expensive raw-video training, and visual adaptation then adds an independent gain over the frozen tracker.

Kill or demote the route if:

- gains disappear against a matched StreamFormer/MATR or ActionSwitch baseline;
- persistent queries do not reduce fragmentation, duplicates, or overlap errors;
- improvement comes only from a larger backbone;
- batched causal and stepwise cached outputs are not prefix-equivalent;
- the method still requires offline NMS or future-aware training inputs;
- raw-video training remains computationally unacceptable after chunk batching and PEFT;
- a closest-prior audit reduces the contribution to applying TrackFormer to a one-dimensional interval task.

## Status

Lead candidate after the user rejected PIVOT as outside On-TAD. `GO` only for a feature-level mechanism pilot and a dedicated Pro novelty review. Raw-video formal training remains `HOLD` until those two gates pass.

Deep-review prompt: [`../../PRO_PETAL_ONTAD_DEEP_REVIEW_PROMPT_20260712.md`](../../PRO_PETAL_ONTAD_DEEP_REVIEW_PROMPT_20260712.md).
