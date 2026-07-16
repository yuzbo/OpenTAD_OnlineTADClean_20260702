---
type: idea
node_id: idea:crs-eps-training
title: "CRS-EPS: Instance-aware Causal Risk-Set Event-Centric Prefix-Episode Training"
stage: current-empty-state-route-structurally-killed
outcome: negative
updated: 2026-07-16
target_gaps: ["G3", "G4", "G7"]
---

# CRS-EPS Training

## 2026-07-16 Implementation Status

The Round-2 implementation exists on `codex/full-petal-implementation`.
Commit `70df86e` passed signed local/Linux B0 at `588/588` and the same locked
reviewer returned PASS. Its preregistered real-data G0 then returned a signed
terminal `KILL`. This closes the current `dynamic_birth` surrogate as an
acceptable primary training protocol under the frozen contract. No fixed-step
GPU profile or effectiveness result has been accepted.

Implemented and CPU-auditable:

- ticket-root-derived `work_dir`, ticket-bound seed/run ID, non-overwritable
  Slurm script, and fake-`sbatch`/shell contract coverage;
- deterministic per-epoch CRS-EPS manifests with the frozen five-component
  proposal, exact `q`, one-draw coverage `rho`, diagnostic union `pi`, uncapped
  HH/IPW weights, multiplicity, coverage, and ESS;
- independent per-draw replay, original global 64-token detach boundaries,
  `M`-draw gradient accumulation with one optimizer event per video, mutable
  buffer isolation, and fail-closed slot exhaustion;
- exact identity metric contract, hash binding, no NMS, and one-token final test
  stream;
- matched-RNG fixed/rematch and replay-fidelity traces;
- explicit G0 arms `video_start_full`, `fixed_192`, `dynamic_birth`, and `reset`,
  plus an outcome-blind margin gate bound to commit/manifest/selection hashes;
- a multi-denominator profile schema covering temporal tokens, replay,
  exposures, ESS, visual frames, data/control/wall time, memory, and GPU-hours.

Terminal evidence and remaining blocks:

- the exact target-Linux G0 bundle used four preregistered samples, frozen
  margins, a deterministic seed-705 checkpoint, and an immutable manifest;
- three samples produced fourteen absolute fidelity violations, including
  maximum relative loss error `0.8021`, minimum gradient cosine `0.2831`, and
  minimum continuous-state cosine `0.1028`;
- the launch validator rejects the signed terminal `KILL`, so GPU profile and
  formal training remain prohibited;
- those four samples are now development-only and cannot confirm a repair;
- the current empty-state `dynamic_birth` replay is structurally invalid as a
  chronological surrogate; this finding does not kill state-faithful
  selective-backward, exact-recomputation, or checkpointed successors;
- chronological Q2 remains a gold/reference candidate rather than an accepted
  primary route because no profile/effectiveness trajectory exists and two
  selected gold controls show slot exhaustion;
- Round 2 selected only the separate R1/CSFSB successor, conditionally behind
  a capacity-first gate; it did not revive this route;
- CRS-EPS was always a cost surrogate, not the Full PETAL headline contribution.

See [the terminal G0 record](../experiments/crs-eps-g0-kill-20260716.md),
[DR-039](../decision_register.md#dr-039-honor-the-signed-crs-eps-g0-kill-and-reopen-training-route-selection),
and [T32](../discussion_timeline.md#t32-signed-g0-kills-the-current-crs-eps-training-surrogate).

## 2026-07-15 Round-2 Protocol Decision

Round 2 returned `GO-HYBRID-PROTOCOL`, meaning permission to implement and
falsify the protocol, not permission to profile or train. The frozen route is:

```text
candidate primary training = CRS-EPS with uncapped HH/IPW
gold reference = preregistered tiny video-start full-stream subset
final quality = complete chronological one-token streaming evaluation
```

Dynamic earliest-birth replay is a causal state surrogate, not guaranteed
full runtime reconstruction. Exact sampling weights cannot repair a latent
state mismatch. Q2 must separate a local fixed/rematch direct-effect trace from
the longitudinal policy effect after model weights and lifecycle states diverge.

Independent amendments are binding project memory:

- require identical stochastic realizations, but do not mandate
  token-addressed RNG if a less invasive RNG snapshot/restore or audit-only
  deterministic route closes the trace;
- freeze exact duplicate/fragmentation/subset evaluator definitions before
  effectiveness results;
- treat model-caused slot exhaustion as a scientific kill, not a rerunnable
  invalid observation;
- do not require a positive LoRA-by-binding interaction; independent additive
  main effects are admissible;
- treat one seed as a resource gate only;
- report every tIoU 0.3:0.7 mAP value as well as the average;
- audit FineAction for same-class instance identity before using it for a
  general identity claim.

See the [Round-2 source](../../PRO_FULL_PETAL_CRS_EPS_Q2_ROUND2_REVIEW_20260715.md),
its [independent absorption](../../PRO_FULL_PETAL_CRS_EPS_Q2_ROUND2_ABSORPTION_20260715.md),
and [DR-030](../decision_register.md#DR-030-adopt-the-amended-round-2-hybrid-protocol).

## 2026-07-15 Implementation Audit

The Q2 implementation at `f4ea53e` does **not** implement CRS-EPS. It performs
complete-video chronological cached-feature training, carries detached
numerical state between 64-token chunks, and steps the optimizer once per
video. It has no event-anchor mixture, sampled-episode manifest, bounded
burn-in/supervised suffix, inclusion probabilities, weighting, ESS, or
sampled/full fidelity audit.

Round 1 returned `REVISE-BEFORE-IMPLEMENTATION` and provisionally recommends a
hybrid: CRS-EPS as candidate main training, a tiny preregistered full-stream
state/loss/gradient audit, and complete chronological validation/test. The
target risk, active-at-entry replay, profile denominators, and non-inferiority
contract must be frozen in Round 2 before implementation. See
[`../../PRO_FULL_PETAL_CRS_EPS_Q2_ROUND1_REVIEW_20260715.md`](../../PRO_FULL_PETAL_CRS_EPS_Q2_ROUND1_REVIEW_20260715.md)
and its [independent absorption](../../PRO_FULL_PETAL_CRS_EPS_Q2_ROUND1_ABSORPTION_20260715.md).

## One-Line Thesis

Train Online TAD on batched causal prefix episodes using a mixture of event-focused and uniformly sampled decision times, while keeping final evaluation as complete chronological streaming.

## Why Needed

Full packet training is too expensive and unfocused:

- about 152,670 packets per epoch;
- about 7 hours per epoch;
- about 210 GPU-hours per model per seed for 30 epochs;
- about 1260 GPU-hours for PCEH vs endpoint-only with 3 seeds.

Most packets are low-information background or redundant ongoing states. The training objective should concentrate on transitions and hard negatives.

## Episode Form

```text
base context = up to 192 decision tokens
dynamic extension = back to every relevant instance's earliest observable birth
supervised suffix H = exactly 8 consecutive decision bins
```

Loss applies only to the supervised bins. Replay remains prefix-causal. The
route must preserve the full reference's original global 64-token detach
boundaries rather than detaching generically at the supervised boundary.

## Sampling Types

The earlier 85% event-focused mixture is too likely to distort the true stream prior. Start with a conservative mixture and ablate it:

| Type | Ratio |
|---|---:|
| endpoint-centered | 20% |
| start-centered | 15% |
| ongoing middle | 10% |
| hard background near boundaries | 15% |
| uniform chronological time | 40% |

Start-centered bins:

```text
s-4B, s-2B, s-B, s, s+B, s+2B, s+4B
```

Endpoint-centered bins:

```text
e-4B, e-2B, e-B, e, e+B, e+2B
```

`B` is the decision stride, currently often 8 frames.

## Online Claim Boundary

Allowed:

> The sampler uses annotations to select supervised prefix times, but the model input and computation graph at each supervised time contain only sources no later than that prefix. Final evaluation is full chronological streaming.

Forbidden:

> Training never uses future information.

Annotation-guided sampling and oracle commit-cost construction use full training annotations. The valid claim is causal model input and causal inference, not future-free supervision.

## State Continuity

- Reconstruct canonical lifecycle and runtime state by causal replay.
- Extend replay to the earliest observable birth of every GT instance active or
  endpoint-crossing in the supervised suffix.
- Apply no supervised loss before the suffix.
- Preserve original global 64-token gradient/detach boundaries.
- Reset and replay every sampled episode independently from the same parameter
  and mutable-buffer snapshot; only gradients may accumulate across the `M`
  draws in one video group.
- Never initialize an episode with an oracle slot, query, hidden state, or
  future endpoint.
- Compare fixed-192 and dynamic replay against reset-state and video-start gold
  on discrete state, continuous state, logits, losses, gradients, occupancy,
  and actual replay cost.
- Kill the surrogate if dynamic replay fails the preregistered fidelity margin
  or frequently degenerates to video-start replay.
- Fail closed if runtime/supervision state or mutable model buffers leak across
  episode draws.

## Bias Controls

- Record the component proposal, marginal anchor probability, one-draw bin
  coverage `rho_t`, multi-draw union probability `pi_t`, and exposure
  multiplicity.
- Primary training uses repeated exposures with uncapped HH/IPW weight
  `(1/T_v)/rho_t`; union HT is only a deduplicated diagnostic.
- The 0.40 uniform component implies `raw_weight <= 2.5`; exceeding the bound
  is an implementation error, not a clipping opportunity.
- SNIPW is a sensitivity diagnostic unless a new pre-result protocol version
  is signed.
- Report effective sample size.
- Report effective sample size by class and lifecycle state, not only globally.
- Compare sampled loss and full-packet gold-subset loss.
- Compare sampled and exhaustive gradient cosine, norm ratio, and sign agreement.
- Compare state occupancy, calibration, duplicate rate, and late false positives against the gold subset.
- Calibrate thresholds on full chronological validation, not sampled episodes.

The current CRS-EPS is a falsified cost surrogate. It is not a headline
algorithmic contribution. Any successor must preserve exact chronological
forward state and restart the outcome-blind evidence chain.

## Stage Plan

1. Frozen SigLIP2 feature cache.
2. Cache-side temporal adapter + projection + CESR/PCEH heads.
3. Short full chronological validation.
4. LoRA only after Stage 1 evidence.
5. For LoRA, use short causal raw-frame episodes and selective visual backward, inspired by ETAD; do not pretend a frozen final-feature cache supports visual gradients.
6. Full-tower finetuning remains conditional.

## Cost-Control Gates

1. Benchmark a fixed number of steps before choosing epoch count.
2. Collapse multiple consecutive supervised bins into one optimizer step.
3. Screen mechanisms with one paired seed; run three seeds only for surviving configurations.
4. Record decode frames, visual forward frames, visual backward frames, temporal tokens, optimizer steps, peak memory, and GPU-hours.
5. Stop any stage that exceeds its predeclared GPU-hour budget without passing the previous evidence gate.

## Failure Criteria

- Event-sampled objective diverges from full-packet gold subset.
- Model learns low latency by suppressing recall.
- Sampled model cannot run full chronological evaluation without state mismatch.
- Same-class/rearm behavior fails.
- Episode state occupancy or calibration differs materially from full chronological training.
- Utility/commit confidence is miscalibrated because endpoints were oversampled.

## 2026-07-16 Round-2 Successor Decision

Round 2 does not revive CRS-EPS. It selects a distinct successor,
R1/CSFSB, that abandons episode replay and preserves the complete chronological
state trajectory. Event sampling survives only as a loss/backward coefficient
mask. The old `dynamic_birth`, `fixed_192`, reset, and empty-state dataset
routes remain negative evidence and development diagnostics.

R1 is documented separately in
[chronological-state-faithful-selective-backward.md](chronological-state-faithful-selective-backward.md).
Its first gate is a zero-GPU capacity/lifecycle audit, not implementation or
training. See [DR-041](../decision_register.md#dr-041-select-csfsb-conditionally-behind-a-capacity-first-gate)
and [T34](../discussion_timeline.md#t34-round-2-selects-csfsb-but-authorizes-capacity-audit-first).
