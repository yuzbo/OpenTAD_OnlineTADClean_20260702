---
type: idea
node_id: idea:petal-ontad
title: "Persistent Event-Set On-TAD (former PETAL-OnTAD)"
stage: blocked-pre-profile-q2-binding
outcome: full-route-demoted
updated: 2026-07-15
target_gaps: ["G4", "G5", "G6", "G8", "G15"]
---

# Persistent Event-Set On-TAD

`PETAL` is retained only in this node ID for history. The acronym is retired because it is already used by Prior-enhanced Temporal Action Localization.

## 2026-07-15 Q2 Audit Revision

The current executable route is a narrower Q2 binding study on fixed cached
features: compare `fixed_birth_slot` with `prefix_rematch_active_pool` while
sharing first-crossing birth, lifecycle, slot capacity, head, inference, and
candidate semantics. This is not a revival of Full PETAL as a paper method.

The immutable-commit Pro audit returned `REVISE` and found a real
`P0-LAUNCH-WORKDIR`: the launch ticket freezes runtime overrides before the
Slurm helper creates and injects its dynamic `work_dir`. The previous profile
permission is revoked until that cross-script identity is fixed and audited.

Independent absorption accepts the central block and these claim boundaries:

- Q2 is `CACHED_FEATURE_TEMPORAL`, with cross-chunk detached state;
- raw-video causality is unproven without extractor provenance;
- a 64-token test packet cannot be reported as low wall-clock latency using
  source-frame time alone;
- fixed binding remains an unproven marginal mechanism hypothesis;
- the full package remains demoted and provisionally reconstructible.

DDP consensus and resume continuation are not current prerequisites: Q2 is
single-process/single-GPU and resume is forbidden. See
[`PRO_FULL_PETAL_CODE_SCIENCE_REVIEW_ABSORPTION_20260715.md`](../../PRO_FULL_PETAL_CODE_SCIENCE_REVIEW_ABSORPTION_20260715.md)
and [DR-028](../decision_register.md#DR-028-revoke-profile-permission-and-fix-the-q2-launch-identity-before-gpu-work).

## 2026-07-12 Revision

The Pro review returned `REVISE`, and independent verification accepts its central objection: the full package is largely reconstructible from causal video backbones, instance-level On-TAD, and TrackFormer-style persistent queries. Full raw-video implementation and training are therefore blocked.

The only active route is a matched cached-feature kill test:

> Compare fresh queries, a faithful Temporal TrackFormer reconstruction, and a minimal prefix-observable persistent event-set decoder. Test whether the On-TAD-specific mechanism improves identity-linked localization rather than merely adding generic tracking capacity.

See [the Stage-1 evidence record](../experiments/persistent-feature-kill-test-20260712.md) and [`PRO_PETAL_DEEP_REVIEW_ABSORPTION_20260712.md`](../../PRO_PETAL_DEEP_REVIEW_ABSORPTION_20260712.md).

## One-Line Thesis

Determine whether prefix-observable persistent action-instance state has measurable value over matched fresh and Temporal TrackFormer decoders before considering any raw-video integration.

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

## Demoted Full Method

The following design is preserved as rejected/conditional history. It is not approved for implementation or training unless the Stage-1 mechanism, five-seed confirmation, and renewed novelty review all pass.

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

## Former Candidate Contributions

1. A strict raw-video, instance-level, jointly trainable On-TAD model rather than a feature-level detector or frame-level OAD model.
2. Persistent identity-bearing event queries with trajectory-level assignment, replacing window-by-window rediscovery and post-hoc grouping.
3. Prefix-parallel causal training with auditable equivalence to cached streaming inference, reducing redundant computation without changing the online task.

These are hypotheses, not surviving claims. The first novelty review found a strong multi-paper reconstruction and forced the matched mechanism gate.

## Closest Work and Required Delta

- **MATR/HAT/OAT:** instance-level On-TAD, but feature-based window/anchor processing; queries or anchors are not persistent instance tracks.
- **ActionSwitch:** handles overlap through anonymous finite-state switches, but uses pre-extracted features and a separate classifier; it does not jointly learn identity-linked interval trajectories.
- **E2E-LOAD:** raw-video end-to-end online action detection, but frame-level OAD rather than action-instance localization.
- **StreamFormer:** causal streaming raw-video representation, but downstream backbone is frozen and evaluation is frame-level OAD.
- **E2E-TAD/TIA/LoSA/Re2TAL:** raw-video backbone adaptation for offline TAL; future-aware processing violates On-TAD inference.
- **TrackFormer/online VIS:** persistent object queries are an architectural precedent; PETAL must prove that trajectory-level temporal interval supervision and prefix-equivalent training create more than a direct tracker transplant.

## Revised Experiment Ladder

1. Frozen-feature pilot: FRESH versus Temporal TrackFormer versus Persistent Event-Set under identical inputs and capacity.
2. If killed, stop this route and preserve the result as negative evidence.
3. If retained, run five seeds, paired per-video uncertainty, error-mode analysis, and renewed closest-prior review.
4. Raw-video PEFT remains a separate later decision; a Stage-1 pass does not authorize it automatically.

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

`REVISE`. Full PETAL is demoted. Q2 is blocked before profile by the launch
identity P0. Raw-video and formal training remain `HOLD`; no effectiveness or
novelty claim is active.

Deep-review prompt: [`../../PRO_PETAL_ONTAD_DEEP_REVIEW_PROMPT_20260712.md`](../../PRO_PETAL_ONTAD_DEEP_REVIEW_PROMPT_20260712.md).
