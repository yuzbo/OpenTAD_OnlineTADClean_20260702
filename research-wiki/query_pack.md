---
type: query_pack
updated: 2026-07-12
status: active
scope: Compressed memory to prepend before any new ideation or implementation planning.
---

# Query Pack: Online/Causal TAD Project Memory

## Project Direction

The task is fixed to standard fully supervised Online Temporal Action Detection/Localization: causal RGB stream in, one immutable `{start, end, class, score}` instance emitted when an action end is detected. Do not introduce sensors, new observability labels, semantic-maintenance outputs, or a replacement task.

PIVOT is rejected as out of scope. Incremental PCEH/CESR remains demoted. The new lead candidate is **PETAL-OnTAD**: jointly train a causal raw-video backbone and persistent event queries, use one identity-bearing query per action trajectory, and make batched causal training prefix-equivalent to incremental cached inference. Read `ideas/petal-ontad.md` and DR-024/DR-025 before new ideation.

## Top Gaps

1. **Raw-video instance-level joint training is missing.** Current On-TAD leaders use frozen/pre-extracted TSN, I3D, SlowFast, or pickle features; raw-video online methods predominantly solve frame-level OAD.
2. **Window rediscovery is a structural failure.** Independent windows repeatedly rediscover one action, causing fragmentation, duplicates, same-class merging, and online-NMS dependence.
3. **Current PCEH is scientifically blocked.** End/emit coupling, repeated late positives, class-level targets, end=emit decode, detached state, and GT proximity remain core concerns.
4. **Full-packet training is unaffordable.** About 152,670 packet-level optimizer events and 7h/epoch are dominated by fine-grained I/O, preprocessing, dispatch, and repeated cache projection.
5. **Generic streaming pretraining is occupied.** StreamFormer blocks the claim that a causal streaming backbone alone is new; BSP and offline E2E-TAD block generic boundary pretraining or PEFT claims.
6. **The required intersection remains open.** Strict causal raw-video adaptation, persistent action-instance identity, standard immutable On-TAD emission, and prefix-equivalent efficient training must all hold together.
7. **Same-class repetition and overlap remain direct risks.** ActionSwitch is the closest baseline and must be matched fairly.
8. **Evaluation must prevent future use and duplicate cleanup.** Full chronological evaluation, fixed thresholds, immutable outputs, recall/FN, delay, and no offline NMS remain mandatory.

## Candidate Portfolio

- **PETAL-OnTAD:** lead candidate. Persistent event trajectories plus raw-video causal joint training and prefix-parallel/incremental equivalence.
- **Feature-level persistent-query pilot:** immediate low-cost gate. It must beat a fresh-query/window detector before raw-video work.
- **OnlineTAD-specific pretraining:** supporting option only after PETAL's mechanism works; generic pretraining is not the headline.
- **CESR/PCEH:** causal infrastructure and negative baselines only, not paper framing.
- **PIVOT/T01/T03/P01/L01:** rejected as current main task or held outside the fixed On-TAD scope.
- **Cache/LoRA/ETAD-style gradient sampling:** cost-control tools, not standalone novelty.

## PETAL Exact Delta

Defensible sentence:

> Track each action instance with one persistent latent event query and jointly adapt a strictly causal raw-video backbone, while training all visible prefixes in batched causal chunks that reproduce incremental cached inference exactly.

All four qualifiers are necessary: action-instance identity, raw-video joint adaptation, strict causality, and train/inference prefix equivalence. Persistent tracks are internal state; standard On-TAD outputs and metrics remain unchanged.

## Failed / Blocked Claims

- PIVOT/three-clock event verification: rejected because it changes the project task.
- Generic causal backbone, generic memory, generic pretraining, LoRA, raw-frame input, or online cache alone: occupied and insufficient.
- Early proposals or mutable user-visible revisions: outside the fixed standard On-TAD output protocol.
- Current THUMOS PCEH, full-packet training first, zero-shot wrapper, and offline distillation headline: rejected.
- PETAL is also rejected if it reduces to TrackFormer plus a one-dimensional interval head without an On-TAD-specific trajectory mechanism or measurable matched gain.

## Closest Prior Work

### PETAL

- **MATR 2024:** strongest direct window/query On-TAD reference; freezes TSN/I3D and uses online NMS.
- **HAT/OAT:** historical or window-anchor On-TAD over pre-extracted features.
- **ActionSwitch 2024:** overlap and same-class state-switch baseline; no persistent semantic instance query or raw-video joint training.
- **E2E-LOAD 2023:** raw-video end-to-end OAD; frame-level output only.
- **StreamFormer 2025:** causal raw-video backbone; downstream frozen and frame-level OAD.
- **E2E-TAD/TIA, LoSA, Re2TAL, ETAD:** raw-video adaptation and efficient gradient methods for offline TAL.
- **TrackFormer/online VIS:** persistent query precedent and the strongest obviousness attack.

## PETAL Mandatory Gates

1. **Mechanism:** persistent queries beat a matched fresh-query/window detector on identical cached features.
2. **Failure mode:** gains include fewer duplicates/fragmentation or better same-class/overlap performance, not only aggregate mAP noise.
3. **Adaptation:** raw-video PEFT adds an independent gain over the frozen persistent tracker with gradient/update evidence.
4. **Causality:** future perturbation, prefix cut, and batched-versus-stepwise equivalence tests pass.
5. **Protocol:** standard immutable On-TAD output, no offline cleanup, fixed class map, late FP and missed GT retained.
6. **Novelty:** Pro review does not reduce the method to E2E-LOAD/StreamFormer + MATR or TrackFormer applied to time.
7. **Cost:** chunked training materially reduces optimizer events and wall time versus packet-wise training; profile before formal runs.
8. **Reproducibility:** THUMOS14 plus one dense/overlap benchmark, matched features/backbone, three seeds for headline comparisons.

PETAL kill conditions: no matched head gain, no independent visual-adaptation gain, no prefix equivalence, continued NMS dependence, cost remains unacceptable, or closest prior work collapses the claimed delta.

## Infrastructure Laws

- Keep PCEH correctness repairs only for baselines; do not transfer its endpoint/emission hazard framing into PETAL by default.
- Keep strict causal reads, full chronological validation, immutable committed outputs, miss/late-FP accounting, and no-future replay audits.
- Persistent slot identity must be explicit and trajectory-level; repeated proposals are not tracks.
- Batched causal training and incremental inference must agree numerically at every prefix cut.
- Training-time GT trajectory assignment is permitted; model inputs and inference state remain future-free.

## Current Final Goal

Do not start raw-video formal training. First submit `PRO_PETAL_ONTAD_DEEP_REVIEW_PROMPT_20260712.md` to Pro and archive/absorb the verdict, then implement the smallest feature-level persistent-query versus fresh-query comparison on identical data. Proceed to raw-video PEFT only if the tracking mechanism survives both gates.
