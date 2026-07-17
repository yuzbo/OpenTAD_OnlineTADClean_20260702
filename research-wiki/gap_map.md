---
type: gap_map
updated: 2026-07-17
status: active
scope: Stable gaps and anti-repetition constraints for Online/Causal TAD.
---

# Gap Map

## G1: Online TAD Needs a Better Protocol Than One-Shot Immutable Emission

Status: unresolved, active.

Traditional On-TAL often treats emitted detections as immutable. That is necessary for fair online evaluation, but too rigid for a model that should continuously understand events. The user corrected the direction: the model should react to state changes such as action start and action end, and should be able to refine earlier online hypotheses as more prefix evidence arrives.

Target framing:

> Mutable online hypotheses before commit, immutable detections after commit.

Do not regress to:

- "after the action ends, emit once and never revise";
- pure alarm-like endpoint detection;
- silent retrospective modification of final outputs.

## G2: Start, Ongoing, End, Commit Must Be Modeled as a Lifecycle

Status: unresolved, active.

Endpoint-only sampling and endpoint-only hazard are insufficient. Start-centered episodes are required so the model learns when to create/rearm an action hypothesis, not only when to close it.

Required lifecycle states:

- background / inactive;
- possible start;
- active / ongoing;
- ending / endpoint suspected;
- completed pending commit;
- committed absorbing detection.

## G3: Current PCEH Targets Are Not Statistically Correct

Status: unresolved, P0 blocker.

Pro review 2026-07-11 found:

- endpoint and emission labels currently come from the same crossing event;
- late prefixes are repeatedly labeled as emission positives;
- targets are class-aggregated rather than instance-aware;
- decoder sets predicted end equal to emit time;
- GT is too close to model metadata.

Required repair:

- instance-aware risk sets;
- first-event endpoint target;
- first-emission or commit target;
- post-event risk masking;
- delayed synthetic case where predicted end precedes commit/emit;
- GT taint audit.

## G4: Full-Packet Training Cost Is Not Acceptable

Status: unresolved, active.

Observed/provided estimate:

- about 152,670 packets per epoch;
- about 7 hours per epoch;
- about 210 GPU-hours per model per seed for 30 epochs;
- about 1260 GPU-hours for PCEH vs endpoint-only with 3 seeds.

Needed solution:

- event-centric causal prefix episodes;
- frozen SigLIP2 feature cache for Stage 1;
- episode batching;
- reduced optimizer steps;
- video-reader pooling and cached preprocessing;
- gold-subset full-packet audit only.

## G5: Pretrained Models Lack Online TAD-Specific Supervision

Status: unresolved, active but second-order.

The goal is not simply to attach a head to SigLIP/VideoMAE. The missing part is task-aligned online supervision:

- start transition;
- ongoing state;
- endpoint;
- boundary refinement;
- commit policy;
- latency budget.

Frozen backbone is allowed for Stage 1, LoRA for Stage 2, and full tower only as a conditional diagnostic.

## G6: Same-Class Repetition and Overlap Are Direct Reviewer Risks

Status: unresolved, P0 blocker.

Class-level targets cause old same-class events to pollute later instances. ActionSwitch is a direct competitor here because it explicitly handles concurrent and same-class actions through state change reasoning.

Needed:

- same-class overlap analysis;
- multi-slot or explicit limitation;
- repeated same-class tests.

## G7: Evaluation Must Prevent Latency Gaming

Status: partially addressed, still active.

Low latency can be cheated by emitting fewer detections. Therefore report:

- OnlineAP at fixed budgets;
- matched TP count;
- recall and FN;
- late FP;
- duplicate emission rate;
- revision latency and commit latency for the CESR direction.

Late predictions must remain FP and their GT must remain FN.

## G8: Competition Boundary Must Stay Sharp

Status: active.

Blocked novelty territories:

- VLM/open-vocabulary/zero-shot online TAL: OZ-TAL threat.
- Offline-to-online distillation: OnPoint and PKD-style prior.
- State-change boundary alone: ActionSwitch threat.
- Early proposal/refinement alone: OAT threat.
- Memory/history alone: MATR and HAT threat.
- Online prediction correction alone: OnlineTAS threat, albeit segmentation task.
- Online action progress/state refinement alone: ProTAS threat, albeit segmentation task.
- Hierarchical streaming action semantics: OpenHOUSE threat.
- Generic causal state maintenance, hypothesis revision, and evidence-aligned response timing: Thinking-QwenVL and StreamReady threat.
- Event-level online memory and low-latency start/end grounding: Hierarchical Event Memory / OnVTG threat.

Defensible delta:

> Maintain identity-preserving temporal action belief trajectories and learn a risk-set utility-based first commit, with an auditable revision ledger and immutable detection ledger under full chronological Online TAD evaluation.

## G9: Broad Online Video Semantic Maintenance Is Already Crowded

Status: active, framing constraint.

The phrase "online video understanding and semantic maintenance" is too broad to support novelty:

- OpenHOUSE combines On-TAL with hierarchical event descriptions.
- ProTAS tracks action progress to refine causal online predictions.
- Thinking-QwenVL explicitly updates a compact causal state, revises hypotheses, and times responses from evidence.
- StreamReady learns when sufficient streaming evidence has arrived.

Required boundary:

- task remains instance-level Online TAD/TAL;
- state is an identity-linked temporal span belief, not a generic semantic memory;
- commit is an irreversible detection decision evaluated by tIoU, GT-end latency, recall, FN, and late FP;
- generic VideoQA, captioning, proactive commentary, and open-ended semantic maintenance are outside the main claim.

## G10: Physical Transition, Visual Verifiability, and Model Decision Are Different Measurement Objects

Status: unresolved, P0 measurement gate.

Current On-TAL commonly evaluates latency from the annotated physical action end. This silently assumes that the action class, completion status, and temporal span become knowable at exactly that timestamp.

Refined distinction:

- `C_phys`: event-specific, independently sensed physical transition interval;
- `C_vis(view,q,population,eta)`: population-level visual verification interval;
- `tau_commit(model)`: first immutable correct causal model decision;
- wall-clock compute latency, reported separately.

The task is now physically anchored streaming event verification, not generic On-TAL. Anticipation before the physical event is a different task and cannot be counted as verification.

Required evidence before adopting the task:

- verified force/contact/pressure/switch timing with a synchronization error budget;
- independent prefix judgments and a population response curve, not one subjective timestamp;
- same-event multi-view evidence that `C_vis` changes while `C_phys` remains fixed;
- proof that decomposed algorithmic delay exposes failures or changes rankings beyond GT-end latency;
- explicit comparison with PaSBench, APT, Ego4D PNR, StreamReady, TouchMoment, and STARE.

Data blocker:

- FEEL's project Dataset/Code links currently resolve to placeholder 404 URLs;
- TouchAnything promises EgoTouch release, but downloadable access was not verified;
- without public access, the route needs a controlled 100-300-instance synchronized pilot.

## G11: Indefinite Streams Need Risk Control, Not Only AP and Tuned Thresholds

Status: unresolved, lead strategic gap.

Repeatedly testing a confidence threshold over a long stream can accumulate false alarms. Current OnlineAP and latency summaries do not state what false-commit risk is controlled at an adaptive stopping time.

Candidate target:

> Minimize detection delay subject to a declared false-commit, miss, or set-coverage risk over the streaming protocol.

Any formal guarantee must define its exchangeability/dependence assumptions honestly. Video frames cannot be treated as independent calibration samples by convenience.

## G12: Real Online Localization Should Allocate Observation Compute

Status: unresolved but crowded; supporting route only.

Processing every frame at the same rate, resolution, and depth is poorly matched to long background periods and short informative boundaries. Active sensing could allocate compute from expected localization-risk reduction.

However, adaptive frame selection, dynamic resolution/depth, hardware-aware online networks, and dynamic event-boundary models already exist. Generic adaptive compute is not a headline contribution.

## G13: Most "Online" Methods Do Not Learn Online

Status: unresolved but high-cost alternative.

Most On-TAL systems train offline and only infer causally. A genuinely continual task would introduce domain/class shifts, unknown actions, and delayed sparse feedback while measuring localization, adaptation speed, forgetting, and stream-time cost.

OZ-TAL, video test-time adaptation, and class-incremental OAD occupy adjacent pieces. A credible route must contribute more than their union and probably requires a new chronological benchmark.

## G14: Non-Events Are Identifiable Only Through Deadlines or Irreversibility

Status: unresolved, hold.

Ordinary action detection models positive events. A required action that has not yet occurred cannot be declared missing in an infinite stream unless:

- an externally grounded deadline expires; or
- the current world state makes later completion irreversibly impossible.

Candidate task:

> Given a causal stream and an obligation contract, predict satisfied, pending, violated, or recovered, and report the earliest legally defensible violation time.

Required novelty gate:

- distinguish from PREGO, EgoProactive, task-graph mistake detection, and temporal-logic monitoring;
- demonstrate natural omissions with grounded deadlines;
- prevent synthetic editing artifacts from defining the benchmark.

## G15: On-TAD Still Lacks Raw-Video Instance-Level Joint Training

Status: unresolved, lead gap.

The literature separates into two incomplete groups:

- instance-level On-TAD methods such as OAT, MATR, HAT, ActionSwitch, and OnPoint operate on frozen or pre-extracted features;
- raw-video causal methods such as E2E-LOAD and StreamFormer solve frame-level OAD or freeze the backbone downstream, while raw-video E2E-TAD methods are offline and future-aware.

The target is not merely to unfreeze a backbone. A defensible solution must jointly provide:

1. strict raw-video causal feature adaptation;
2. identity-preserving action-instance state across prefixes;
3. standard immutable On-TAD outputs at detected action ends;
4. prefix-equivalent batched training and incremental inference;
5. lower redundant training/inference cost than overlapping-window processing.

Candidate route: [ideas/petal-ontad.md](ideas/petal-ontad.md).

## G16: Training and Inference Need One Model-Only Prefix State

Status: candidate gap not established; Protocol V1 received independent
`REVISE_PROTOCOL_BEFORE_COLLECTION`. Hardened Protocol V2 awaits reassessment
by the same sole reviewer before new route-identifiability evidence may be
collected.

Q2 proved a repository-local contract defect: predicted runtime availability
gated GT canonical ownership, so false ACTIVE/refractory state permanently
discarded first-crossing targets and a silent birth policy could pass
capacity. It did not prove that standard On-TAL generally lacks a
model-only prefix state, or that persistent latent carriers are necessary.

Any successor still requires:

- one scientific state transition shared by train and inference;
- state contains only causal inputs, model outputs, and past immutable commits;
- GT assignment is ephemeral and loss-only;
- assignment cannot alter identity, availability, reset, risk set, or
  emission;
- no-birth, no-emission, and always-background policies must fail readiness;
- same-bin release/reseed, repeated instances, overlap, delayed endpoints, and
  immutable commits must close under synthetic tests.

Those are correctness requirements, not evidence that R-A is the best route.
Before a model P0, the project must compare:

- a no-identity prefix completion set;
- ordinary persistent queries;
- a one-dimensional TrackFormer/MOTR reconstruction;
- a clean order/risk-set baseline that does not modify old Q2;
- R-A as one candidate arm.

The exact split must also establish that repetition, overlap, same-bin
transitions, and concurrency are prevalent enough to support the intended
claim. Until then, field-gap status is `NOT_ESTABLISHED`.

Earlier repository analysis is prior annotation exposure, not a fresh R0:
training had 200 videos/3003 instances and validation had 211 videos/3325
instances; overlap appeared in 23/32 videos, same-class overlap in only 2/3
videos, and maximum concurrency was 2. The new census may be blind to model
outcomes, but cannot be called annotation-unseen. The unresolved reporting
count of 211 versus canonical expectation 213 must also be reconciled before
any prevalence claim.

Static code inspection suggests the cache builder samples the latest frame at
each stride and the default SigLIP path encodes frames independently. That is
not yet a certificate for the existing cache: its exact model revision,
processor, weights, raw-video hashes, extraction environment, and per-token
support map are not bound. Existing-cache strict causality therefore remains
`UNVERIFIED`.

DR-050 now supplies machine-readable V2, an exact source manifest,
cryptographic review binding, source-derived R0/R1 validators, executable
controls/OOD/R6, and adversarial tests. This closes known V1 protocol
bypasses, not scientific evidence: the exact 211/213 certificate and
current-cache R1 certificate still do not exist. R0, R1, model outcomes,
B0-B4 implementation, and GPU use remain blocked until V2 receives a signed
same-reviewer protocol PASS.

Candidate:
[ideas/prefix-shared-latent-event-filter.md](ideas/prefix-shared-latent-event-filter.md),
now B4 only.

Required evidence:
[experiments/prefix-route-identifiability-gate-20260717.md](experiments/prefix-route-identifiability-gate-20260717.md).
