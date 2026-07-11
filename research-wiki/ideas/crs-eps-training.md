---
type: idea
node_id: idea:crs-eps-training
title: "CRS-EPS: Instance-aware Causal Risk-Set Event-Centric Prefix-Episode Training"
stage: active
outcome: pending
updated: 2026-07-11
target_gaps: ["G3", "G4", "G7"]
---

# CRS-EPS Training

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
192 decision tokens as burn-in/context
+ 4-8 consecutive supervised decision bins
```

Loss applies only to supervised bins. Context remains prefix-causal.

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

- Reconstruct tracker state by replaying the causal burn-in.
- Apply no supervised loss to burn-in tokens.
- Detach burn-in state at the supervised boundary unless a controlled truncated-BPTT experiment says otherwise.
- Never initialize an episode with an oracle action state.
- Compare replayed state against reset-state episodes and full chronological state on a gold subset.

## Bias Controls

- Record inclusion probability `pi_t`.
- Use inverse-probability or self-normalized weights.
- Cap weights to control variance.
- State clearly that clipping and self-normalization trade variance for bias.
- Report effective sample size.
- Report effective sample size by class and lifecycle state, not only globally.
- Compare sampled loss and full-packet gold-subset loss.
- Compare sampled and exhaustive gradient cosine if possible.
- Compare state occupancy, calibration, duplicate rate, and late false positives against the gold subset.
- Calibrate thresholds on full chronological validation, not sampled episodes.

CRS-EPS is a validated cost surrogate. It is not a headline algorithmic contribution.

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
