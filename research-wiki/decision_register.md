---
type: decision_register
updated: 2026-07-29
status: active
scope: Major research decisions, reasons, counterarguments, sources, and reversibility.
---

# Decision Register

Each decision records what we chose, why, what was rejected, and when the decision should be revisited.

## DR-001: Do Not Treat the Current Code as Paper-Ready Online TAD

Decision:

> Current implementation is a streaming/causal validation skeleton, not a paper-ready Online TAD method.

Reason:

- Formal training remains incomplete.
- Early route was long-window/chunk-end rather than true low-latency continuous online inference.
- Current PCEH target is not statistically valid.

Counterargument:

- The code has smoke-tested pieces, ledger fields, causal projection, and some no-future audits.

Resolution:

- Keep it as implementation substrate and protocol skeleton.
- Do not write SOTA or paper-ready claims.

Sources:

- `PRO_REVIEW_ABSORPTION_20260709.md`
- `PRO_REVIEW_ABSORPTION_20260710.md`
- `PRO_REVIEW_ABSORPTION_20260711.md`
- `pceh-current-implementation-20260710.md`

Reversibility:

- Reversible only after formal multi-seed experiments and all P0/P1 gates pass.

Wiki nodes:

- [claims/c1-strict-online-protocol.md](claims/c1-strict-online-protocol.md)
- [experiments/formal-training-none.md](experiments/formal-training-none.md)

## DR-002: Reject One-Shot Post-End Emission as the Main Story

Decision:

> One-shot post-end immutable emission is rejected as the main paper framing.

Reason:

- User judged it inelegant and too alarm-like.
- Online video understanding should update hypotheses when state changes.
- It cannot naturally express progressive boundary/class refinement.

Counterargument:

- Immutable output is a standard On-TAL constraint and easier to evaluate.

Resolution:

- Use two layers: mutable online hypotheses before commit, immutable committed detections after commit.

Sources:

- user correction on 2026-07-11;
- CESR discussion;
- prior Pro review strict online constraints.

Reversibility:

- Not reversible as a main story unless CESR fails completely. One-shot emission remains a baseline/ablation.

Wiki nodes:

- [ideas/rejected-one-shot-emission.md](ideas/rejected-one-shot-emission.md)
- [ideas/cesr-ontad.md](ideas/cesr-ontad.md)

## DR-003: Select CESR-OnTAD as the Current Main Direction

Status:

- Superseded on 2026-07-11 by DR-015. Preserve this entry as decision history; do not use the broad "track, refine, commit" package as the current novelty claim.

Decision:

> The active direction is Causal Event-State Hypothesis Tracking / Refinement for Online Temporal Action Detection.

Reason:

- It captures user's preferred semantics: judge state changes and revise hypotheses.
- It gives a stronger story than endpoint/emission alone.
- It distinguishes mutable hypothesis from immutable commit.
- It creates measurable new outputs: revision latency, commit latency, boundary refinement trajectory.

Counterargument:

- OAT, ActionSwitch, OnlineTAS, and offline boundary refinement are close.

Resolution:

- Novelty cannot be "state transition" alone.
- Historical resolution: novelty was provisionally assigned to the whole track/refine/commit package.
- Current resolution (DR-015): retain that package only as system structure and narrow the claim to identity-preserving belief trajectories, utility-based first commit, and trajectory-level evaluation.

Sources:

- user correction on state-change-driven outputs;
- competition check around OAT, ActionSwitch, OnlineTAS;
- `online-action-models-2026-research.md`.

Reversibility:

- Reversible if CESR does not improve revision quality or AP-latency Pareto over one-shot/endpoint-only baselines.

Wiki nodes:

- [ideas/cesr-ontad.md](ideas/cesr-ontad.md)
- [claims/c2-mutable-hypothesis-commit.md](claims/c2-mutable-hypothesis-commit.md)

## DR-004: Demote PCEH from Full Story to Submodule

Decision:

> PCEH is retained as endpoint/commit hazard machinery inside CESR, not the full paper story.

Reason:

- PCEH's endpoint/emission framing is too narrow after CESR pivot.
- Current implementation fails valid risk-set hazard criteria.

Counterargument:

- PCEH identified a real gap: endpoint, completion, and emission should be distinct.

Resolution:

- Keep PCEH as a component for endpoint and commit modeling.
- Repair PCEH risk sets before using it in experiments.

Sources:

- `PRO_REVIEW_ABSORPTION_20260710.md`
- `PRO_REVIEW_ABSORPTION_20260711.md`
- user correction toward state refinement.

Reversibility:

- Could become a named method component if repaired and ablations show independent gains.

Wiki nodes:

- [ideas/pceh-ontad.md](ideas/pceh-ontad.md)
- [claims/c3-risk-set-hazard.md](claims/c3-risk-set-hazard.md)

## DR-005: P0 Scientific-Correctness Repair Must Precede Formal Training

Decision:

> Do not launch formal training before instance-aware risk targets, first-emission/commit targets, endpoint freeze, GT taint audit, and delayed synthetic tests.

Reason:

- Current labels conflate endpoint and emission.
- Late positives violate first-event semantics.
- Class aggregation breaks repeated same-class instances.
- Predicted end equals emit time, destroying the claimed separation.

Counterargument:

- Smoke already passed and training can technically start.

Resolution:

- Smoke is not scientific validity.
- Formal training from invalid targets wastes GPU and produces unclaimable results.

Sources:

- `PRO_REVIEW_ABSORPTION_20260711.md`

Reversibility:

- Not reversible. This is a hard gate.

Wiki nodes:

- [gap_map.md#G3-current-pceh-targets-are-not-statistically-correct](gap_map.md)
- [claims/c3-risk-set-hazard.md](claims/c3-risk-set-hazard.md)

## DR-006: Reject Full-Packet Training as Default

Decision:

> Full-packet chronological training is not the default formal training protocol.

Reason:

- About 152,670 packets per epoch.
- About 7h/epoch.
- About 210 GPU-hours/model/seed for 30 epochs.
- Too expensive for paired 3-seed baseline/PCEH/CESR comparisons.

Counterargument:

- Full-packet training is distribution-faithful and avoids sampling bias.

Resolution:

- Keep full-packet route as gold subset/audit and final chronological evaluation.
- Use CRS-EPS for training.

Sources:

- `PRO_REVIEW_ABSORPTION_20260711.md`
- remote cancelled pilots summary.

Reversibility:

- Reversible only if throughput improves by an order of magnitude and formal cost becomes acceptable.

Wiki nodes:

- [ideas/rejected-full-packet-training.md](ideas/rejected-full-packet-training.md)
- [ideas/crs-eps-training.md](ideas/crs-eps-training.md)

## DR-007: Select CRS-EPS as the Current Training Protocol

Decision:

> Use instance-aware causal risk-set event-centric prefix episodes, not blind full-packet training.

Reason:

- Focuses supervision on start/end/ongoing/hard-background transitions.
- Reduces optimizer and I/O overhead.
- Compatible with prefix-causal inputs if sampling is disclosed as annotation-guided.

Counterargument:

- Sampling can distort base rates and calibration.

Resolution:

- Record inclusion probability.
- Compare with full-packet gold subset.
- Calibrate on full chronological validation.

Sources:

- `PRO_REVIEW_ABSORPTION_20260711.md`
- user question about start-stage sampling.

Reversibility:

- Reversible if sampled objective diverges from full-packet gold subset or full chronological validation collapses.

Wiki nodes:

- [ideas/crs-eps-training.md](ideas/crs-eps-training.md)
- [claims/c4-crs-eps-cost.md](claims/c4-crs-eps-cost.md)

## DR-008: Start-Centered Sampling Is Required

Decision:

> Endpoint-centered sampling alone is insufficient; start-centered episodes are required.

Reason:

- Online hypotheses must be created and rearmed.
- The model needs pre-start hard negatives, start crossing, and early ongoing states.

Counterargument:

- Endpoint/commit is the most distinctive part of low-latency On-TAD.

Resolution:

- Keep endpoint as important, but train the full event-state lifecycle.

Sources:

- user question: "endpoint 附近重点采，那开始阶段怎么办"
- CESR shift.

Reversibility:

- Reversible only if ablations show start sampling adds no benefit and does not affect false starts/rearm.

Wiki nodes:

- [ideas/state-transition-sampling.md](ideas/state-transition-sampling.md)

## DR-009: Use Frozen Feature Cache for Stage 1

Decision:

> Stage 1 training should use audited frozen SigLIP2 feature cache.

Reason:

- Removes repeated visual forward cost.
- Cache size is manageable for selected-frame frozen features.
- Keeps training practical.

Counterargument:

- Feature cache weakens raw-video end-to-end claim.

Resolution:

- Make the claim explicit: frozen visual features plus learned causal online detector.
- Do not call Stage 1 visual end-to-end learning.

Sources:

- `PRO_REVIEW_ABSORPTION_20260711.md`
- `pceh-current-implementation-20260710.md`

Reversibility:

- Reversible after LoRA/full raw episodic path passes throughput and update audits.

Wiki nodes:

- [ideas/feature-cache-stage1.md](ideas/feature-cache-stage1.md)

## DR-010: LoRA Is Conditional Stage 2, Full Tower Is Later Diagnostic

Decision:

> Use LoRA after frozen Stage 1 evidence; do not start with full visual tower finetuning.

Reason:

- Full visual tower is too costly and not proven trainable in the current route.
- LoRA provides a controlled test of visual adaptation.

Counterargument:

- The user wants to solve missing pretrained-model adaptation for online TAD.

Resolution:

- Yes, but only with audited PEFT first.

Sources:

- `PRO_REVIEW_ABSORPTION_20260711.md`
- `end-to-end-ontad-research.md`

Reversibility:

- Full tower becomes allowed only after LoRA improves over frozen and one-epoch diagnostic fits allocation.

Wiki nodes:

- [ideas/lora-stage2.md](ideas/lora-stage2.md)
- [ideas/rejected-full-visual-tower-first.md](ideas/rejected-full-visual-tower-first.md)

## DR-011: Do Not Claim Zero-Shot/Open-Vocabulary Online TAL

Decision:

> Zero-shot/open-vocabulary/VLM online TAL is blocked as main claim.

Reason:

- OZ-TAL 2026 already covers Online Zero-Shot TAL with off-the-shelf VLMs.

Counterargument:

- VLMs are useful representations and may help closed-set Online TAD.

Resolution:

- Use VLM/foundation features as representation if needed, but do not claim zero-shot novelty without direct OZ-TAL comparison.

Sources:

- `PRO_REVIEW_ABSORPTION_20260709.md`
- `online-action-models-2026-research.md`

Reversibility:

- Reversible only for a separate OZ-TAL-facing study with primary baselines.

Wiki nodes:

- [ideas/open-vocabulary-zero-shot.md](ideas/open-vocabulary-zero-shot.md)
- [papers/oztal2026-zero-shot.md](papers/oztal2026-zero-shot.md)

## DR-012: Distillation Is Support/Ablation, Not Core Novelty

Decision:

> Offline teacher to online student distillation is not the main contribution.

Reason:

- OnPoint and PKD-style prior work are too close.

Counterargument:

- Teacher signals may improve training and hard negatives.

Resolution:

- Use only as ablation/enhancement, with teacher absent at test time.

Sources:

- `PRO_REVIEW_ABSORPTION_20260709.md`
- `PRO_REVIEW_ABSORPTION_20260710.md`
- `online-action-models-2026-research.md`

Reversibility:

- Reversible only if distillation is framed as auxiliary, not the central claim.

Wiki nodes:

- [ideas/offline-teacher-distillation.md](ideas/offline-teacher-distillation.md)
- [papers/onpoint2026-distillation.md](papers/onpoint2026-distillation.md)

## DR-013: Evaluation Must Use GT-End Latency and Coverage Counts

Decision:

> Primary latency is `commit/emit time - matched GT end`, not `emit - predicted end`; latency reports must include TP count, recall, FN, and late FP.

Reason:

- Predicted-end latency can be zero by construction if predicted end equals emit.
- Low latency can be faked by emitting few detections.

Counterargument:

- Predicted-end latency is useful for diagnostic endpoint/commit separation.

Resolution:

- Keep predicted-end latency as diagnostic only.

Sources:

- `PRO_REVIEW_ABSORPTION_20260710.md`
- `PRO_REVIEW_ABSORPTION_20260711.md`

Reversibility:

- Not reversible for paper metrics.

Wiki nodes:

- [claims/c6-budgeted-eval.md](claims/c6-budgeted-eval.md)

## DR-014: Same-Class / Repeated Instance Handling Is a Hard Reviewer Risk

Decision:

> Same-class repeated and overlapping actions must be explicitly tested or bounded.

Reason:

- Current class-level target aggregation can pollute later same-class instances.
- ActionSwitch directly targets state changes, concurrent, and same-class actions.

Counterargument:

- THUMOS may contain limited overlap and initial route could be one-track/class.

Resolution:

- Add analysis and tests; if not solved, state limitation clearly.

Sources:

- `PRO_REVIEW_ABSORPTION_20260711.md`
- ActionSwitch competition check.

Reversibility:

- Not reversible as a risk; implementation scope can choose multi-slot or explicit limitation.

Wiki nodes:

- [gap_map.md#G6-same-class-repetition-and-overlap-are-direct-reviewer-risks](gap_map.md)
- [papers/actionswitch2024-state.md](papers/actionswitch2024-state.md)

## DR-015: Reject Broad Track-Refine-Commit as the Headline Novelty

Decision:

> "Track, refine, and commit" remains an intuitive summary, but is no longer sufficient as the paper's technical novelty claim.

Reason:

- CAG-QIL already uses sequential decision state and an MDP.
- SimOn already performs lightweight recurrent online state prediction.
- OAT already generates early proposals and refines boundaries.
- ActionSwitch already uses explicit state changes.
- ProTAS already uses ongoing action progress to refine online predictions.

Resolution:

- Narrow the main claim to identity-preserving belief trajectories, utility-based first commit, and trajectory-level evaluation.

Sources:

- fresh literature audit on 2026-07-11;
- .aris/traces/novelty-check/2026-07-11_run01/.

Reversibility:

- Not reversible unless a future search or experiment establishes a stronger, independently novel mechanism.

Wiki nodes:

- [ideas/cesr-ontad.md](ideas/cesr-ontad.md)
- [papers/cagaqil2021-decision-context.md](papers/cagaqil2021-decision-context.md)
- [papers/simon2022-sequential-ontal.md](papers/simon2022-sequential-ontal.md)

## DR-016: Keep the Paper Inside Online TAD, Not Generic Streaming Semantics

Decision:

> Online semantic state maintenance is a mechanism; the task and paper claim remain identity-level Online Temporal Action Detection.

Reason:

- OpenHOUSE occupies hierarchical streaming action semantics.
- Thinking-QwenVL and StreamReady occupy causal state update and evidence-aligned response timing in streaming video understanding.
- A generic semantic-maintenance story would lose both task clarity and novelty.

Resolution:

- Evaluate temporal span beliefs, association, revision, and detection commit.
- Do not expand the main claim to VideoQA, captioning, commentary, or open-ended semantic memory.

Sources:

- [papers/openhouse2025-hierarchical-streaming.md](papers/openhouse2025-hierarchical-streaming.md)
- [papers/thinkingqwen2026-evidence-timing.md](papers/thinkingqwen2026-evidence-timing.md)
- [papers/streamready2026-readiness.md](papers/streamready2026-readiness.md)

Reversibility:

- A separate streaming-VLM paper could revisit this, but not the current Online TAD paper.

## DR-017: Treat CRS-EPS as an Audited Cost Surrogate, Not Main Novelty

Decision:

> CRS-EPS is the practical training protocol, but its scientific role is to approximate full chronological training under explicit bias audits.

Reason:

- ETAD already establishes selective snippet-gradient/proposal sampling for efficient TAD.
- OnlineTAS already trains with non-overlapping clips for efficiency and infers densely.
- Event-heavy sampling can distort background priors, state occupancy, calibration, duplicate behavior, and commit timing.

Resolution:

- Include at least 40% uniform chronological sampling initially.
- Report inclusion probabilities, clipped/self-normalized weighting, state/class ESS, calibration, and gold-subset comparisons.
- Use full chronological validation/test regardless of training sampler.

Sources:

- [ideas/crs-eps-training.md](ideas/crs-eps-training.md)
- [papers/etad2023-efficient-e2e.md](papers/etad2023-efficient-e2e.md)
- [papers/onlinetas2024-online-segmentation.md](papers/onlinetas2024-online-segmentation.md)

Reversibility:

- Sampling ratios are reversible from evidence; the audit requirement is not.

## DR-018: Reopen the Main Route and Demote CESR to a Substrate

Decision:

> Do not spend the formal training budget on identity-trajectory CESR as the paper headline. Retain its causal state, identity ledger, and commit protocol as reusable infrastructure and baselines.

Reason:

- Its components can be reconstructed from CAG-QIL, SimOn, OAT, MATR, ActionSwitch, ProTAS, and recent readiness-aware streaming work.
- A stronger paper should change the scientific question, not only the proposal lifecycle.
- The current route's expected novelty is insufficient to justify its training cost.

Resolution:

- Keep P0 correctness work because future routes still require causal targets and honest ledgers.
- Pause large CESR training until a task-level route is selected.

Reversibility:

- Reversible only if a direct experiment demonstrates a mechanism that cannot be reduced to proposal association plus thresholded stopping.

## DR-019: Make Evidence-Aligned Risk-Controlled On-TAL the Lead Candidate

Status: superseded as a sole lead by DR-021 after the divergent Pro review and independent closest-prior audit.

Decision:

> The leading candidate is to separate physical action boundaries from the first causally sufficient evidence time and to optimize online localization under explicit sequential risk control.

Reason:

- It attacks a hidden assumption in the task definition.
- No direct On-TAL work was found that combines evidence-ready timing with anytime false-commit control.
- It can be tested with a lightweight causal detector and calibration layer before expensive visual finetuning.

Counterargument:

- Thinking-QwenVL and StreamReady already use first-sufficient evidence/readiness in streaming VideoQA.
- Action Completion and offline boundary-uncertainty work cover adjacent concepts.
- Human evidence-time annotation may be unstable.

Resolution:

- Treat this as a lead candidate, not a selected final route.
- First run an annotation-agreement and metric-value pilot; kill the route if the target is not reproducible.

Reversibility:

- Fully reversible after the pilot.

## DR-020: Reject Generic Streaming Pretraining, Memory, or Adaptive Compute as Standalone Novelty

Decision:

> Do not pivot to generic causal backbone pretraining, streaming memory, asynchronous query handling, or dynamic computation without a sharper action-localization-specific problem.

Reason:

- StreamFormer already trains a causal streaming backbone with multi-granularity objectives.
- SelectStream and related work occupy budgeted streaming evidence memory.
- AViLA occupies query-evidence asynchrony.
- SAN and DyBDet occupy adaptive online computation and dynamic boundary processing.

Resolution:

- These mechanisms may support a task-level contribution but are not paper stories by themselves.

Reversibility:

- Reversible only with a clearly distinct mechanism and direct same-budget baselines.

## DR-021: Accept the Pro No-Go but Not Its Final Route Selection

Decision:

> Accept the recommendation to stop treating PCEH/CESR as the paper headline, but do not yet select Anytime-Valid Semantic Event Alarms as the new main route.

Reason:

- The Pro review correctly identifies current PCEH target, decoder, instance-risk, detach-state, and GT-taint blockers.
- Generic On-TAL memory, refinement, readiness, and hazard additions do not justify the current training cost.
- WACV Workshops 2026 already frames a real-time video problem as e-process sequential testing with provable false-alert control.
- Online video anomaly detection already contains long-horizon false-alarm-control precedent.
- T01 still lacks an exact null, filtration, reset rule, multiplicity model, risk unit, and real continuous null-stream dataset.
- T02 may offer a more CV-specific scientific object if synchronized physical signals are available.

Resolution:

- T01 Anytime-Valid Semantic Event Alarms becomes a conditional P0 candidate.
- T02 Three-Clock Event Observability becomes a conditional co-candidate.
- T03, P01, and L01 remain hold routes.
- Final route selection remains open.

Sources:

- `PRO_DIVERGENT_IDEA_REVIEW_20260711.md`;
- `PRO_DIVERGENT_IDEA_ABSORPTION_20260711.md`;
- [papers/munoz2026-tracking-eprocess.md](papers/munoz2026-tracking-eprocess.md);
- [papers/doshi2021-video-anomaly-far.md](papers/doshi2021-video-anomaly-far.md).

Reversibility:

- Candidate ordering is reversible after the defined gates; the PCEH headline no-go is reversible only with a genuinely new scientific object and direct evidence.

## DR-022: Novelty and Statistical Validity Precede the 48-Hour Empirical Pilot

Decision:

> Before training or celebrating a long-horizon alarm curve, define the statistical task and reproduce the closest generic and video sequential baselines on identical causal scores.

Required order:

1. choose one semantic event, null unit, reset rule, risk target, and application cost profile;
2. compare fixed/matched thresholds, Bonferroni or alpha-spending, CUSUM/SPRT, WATCH-style monitoring, and the WACV tracking e-process pattern;
3. distinguish real continuous negatives from synthetic block-concatenation diagnostics;
4. validate with uncertainty bounds and adequate replications;
5. train no backbone until the task changes a scientific decision or yields a nontrivial matched-risk advantage.

Reason:

- Repeated fixed-threshold risk growth is mathematically expected and is not by itself a paper contribution.
- One hundred null streams at `alpha=0.05` are too imprecise to establish near-alpha control.
- A positive pilot is meaningful only if simple matched-risk baselines fail and real-stream evidence supports the claim.

Sources:

- `PRO_DIVERGENT_IDEA_ABSORPTION_20260711.md`;
- [ideas/anytime-semantic-event-alarms.md](ideas/anytime-semantic-event-alarms.md).

Reversibility:

- The exact risk target and pilot design are reversible; the order “definition and closest baselines before expensive training” is not.

## DR-023: Refine Three-Clock Observability into PIVOT and Gate It on Measurement

Decision:

> Replace the broad “three timestamps” idea with PIVOT: independently sensed physical transition intervals, view-conditioned population visual-verification intervals, and residual causal model commit delay.

Reason:

- PaSBench already approximates visible-risk, accident, and warning timing.
- APT and Ego4D PNR already cover physical/state-transition temporal localization.
- StreamReady and Thinking-QwenVL already cover sufficient visual evidence and response timing.
- TouchMoment covers precise contact spotting.
- STARE and Towards Streaming Perception cover latency-aware evaluation and ranking reversals.
- The surviving delta is the independently anchored, view-conditioned, population-interval measurement and residual-delay attribution.

Resolution:

- Treat the task as streaming event verification, not generic Online TAL.
- Use intervals for both physical and visual clocks.
- Separate anticipation from verification and semantic video time from compute wall-clock.
- Make task/evaluation the dominant contribution and ordered multi-state survival a minimal supporting method.
- Permit only a 30-50-instance observability micro-pilot before Pro review and data gate completion.
- Keep model implementation and all large training on HOLD.

Data blocker:

- FEEL project download links currently resolve to placeholder 404 targets.
- EgoTouch public availability remains unverified.
- A controlled 100-300-instance collection is the fallback, not an assumed resource.

Sources:

- `THREE_CLOCK_TASK_METHOD_DESIGN_20260711.md`;
- `THREE_CLOCK_COMPETITION_REVIEW_20260711.md`;
- `PRO_THREE_CLOCK_DEEP_REVIEW_PROMPT_20260711.md`;
- [papers/pasbench2026-proactive-warning.md](papers/pasbench2026-proactive-warning.md);
- [papers/apt2026-atomic-physical-transitions.md](papers/apt2026-atomic-physical-transitions.md);
- [papers/stare2026-stream-latency.md](papers/stare2026-stream-latency.md).

Reversibility:

- Fully reversible after the measurement pilot or Pro review. The ban on broad “three clocks are new” claims is not reversible without new prior-art evidence.

## DR-024: Reject PIVOT Because It Leaves the On-TAD Task

Decision:

> Do not pursue PIVOT as this project's main task or method.

Reason:

- The user requires innovation inside established Online Temporal Action Detection/Localization.
- PIVOT changes the target to physically anchored streaming event verification.
- It requires new sensor anchors, population observability annotations, and a new evaluation decomposition.
- A valid new task does not solve the selected On-TAD performance and end-to-end-training problem.

Resolution:

- Preserve the prior-art audit as background material.
- Mark the PIVOT idea rejected-out-of-scope.
- Do not spend data-collection, annotation, implementation, or training budget on it.

Reversibility:

- Not reversible within the current project scope unless the user explicitly changes the task.

## DR-025: Select Persistent End-to-End Event Tracking as the New Lead Candidate

Decision:

> Keep the standard On-TAD task fixed and investigate PETAL-OnTAD: a raw-video causal backbone jointly trained with persistent event queries and trajectory-level prefix supervision.

Reason:

- Current instance-level On-TAD methods predominantly use frozen/pre-extracted visual features and independent windows or grouping.
- E2E-LOAD and StreamFormer establish raw-video causal OAD, but not action-instance start/end localization.
- Offline E2E-TAD establishes efficient backbone adaptation, but not strict streaming inference.
- Persistent instance state directly targets fragmentation, duplicate emission, same-class repetition, overlap, and the train/test mismatch of rediscovering an action at every window.
- Prefix-parallel causal training can reduce the current packet-wise Python, I/O, and optimizer overhead while retaining strict online inference.

Resolution:

- First run a matched feature-level persistent-query versus fresh-query pilot.
- Conduct a dedicated Pro novelty review before raw-video implementation claims.
- Permit raw-video PEFT only after the mechanism pilot passes.
- Preserve standard On-TAD outputs and immutable emission evaluation; latent pre-end tracks are not a new task output.

Sources:

- [ideas/petal-ontad.md](ideas/petal-ontad.md);
- [papers/matr2024-memory.md](papers/matr2024-memory.md);
- [papers/actionswitch2024-state.md](papers/actionswitch2024-state.md);
- [papers/e2eload2023-e2e-oad.md](papers/e2eload2023-e2e-oad.md);
- [papers/streamformer2025-streaming-representation.md](papers/streamformer2025-streaming-representation.md).

Reversibility:

- Fully reversible after the feature-level mechanism pilot or novelty review. The fixed On-TAD task boundary is not reversible without explicit user approval.

## DR-026: Demote Full PETAL and Run a Matched Persistent-State Kill Test

Status: active; supersedes DR-025 as the executable route.

Decision:

> Do not implement or train the full raw-video PETAL package. First compare FRESH, Temporal TrackFormer, and a minimal prefix-observable Persistent Event-Set decoder on one frozen causal feature cache under a hard 10 GPU-hour Stage-1 budget.

Accepted review findings:

- Full PETAL is substantially reconstructible from StreamFormer/E2E-LOAD, MATR/ActionSwitch, and TrackFormer/MOTR.
- Temporal TrackFormer is the novelty-killer baseline, not an optional ablation.
- Existing PCEH class-keyed state, end/emission coupling, end-equals-emit decode, and mixed GT/runtime interfaces are scientifically unsafe foundations for the new route.
- Raw-video adaptation would confound the persistence question and remains blocked.
- The first pilot must omit NMS and duplicate-repair losses so that persistence itself is tested.

Independent disagreements and qualifications:

- Complete trajectory annotations in offline supervised training are not automatically inference leakage; they are a privileged-assignment risk and may appear only as a labeled upper-bound ablation.
- Class supervision begins after the annotated start becomes prefix-observable; endpoint-only class supervision is an ablation, not the main rule.
- A start pointer is a candidate mechanism and must beat scalar regression.
- `+2.0` mOnlineAP points or `20%` error reduction is a project resource gate, not a universal significance theorem.
- Three seeds are enough only to kill the route; a retained claim needs at least five seeds and paired uncertainty.

Implementation resolution:

- use training-only prefix schedules, first-crossing endpoint targets, Hungarian assignment, fixed identity after birth, and one-time immutable emissions;
- reject EOF/terminal metadata and unknown potential-GT detector inputs;
- compare only registered mechanism deltas on identical cached features;
- use four slots because THUMOS14 training and validation each have observed maximum concurrency two;
- invalidate runs with protocol violations, slot exhaustion, unmatched seeds, or budget overrun;
- do not authorize raw-video training merely because Stage 1 passes.

Sources:

- `PRO_PETAL_DEEP_REVIEW_20260712.md`;
- `PRO_PETAL_DEEP_REVIEW_ABSORPTION_20260712.md`;
- [experiments/persistent-feature-kill-test-20260712.md](experiments/persistent-feature-kill-test-20260712.md);
- [papers/trackformer2022-tracking-queries.md](papers/trackformer2022-tracking-queries.md).

Reversibility:

- The Stage-1 architecture and thresholds are revisable before formal submission. The raw-video hold is reversible only after the registered mechanism, confirmation, novelty, and cost gates pass.

## DR-027: Hold the Scientific Pilot After Smoke Until Identifiability Review

Status: active procedural gate; refines DR-026 without changing the On-TAD task.

Decision:

> Treat the completed cache and GPU smoke as engineering readiness only. Do not submit the three-seed pilot until Pro audits whether FRESH, Temporal TrackFormer, and PES isolate persistent-state value or require a minimal controlled bridge set.

Reason:

- Cache job `1159510` and smoke job `1159843` passed their registered engineering contracts.
- Smoke cannot establish performance, novelty, fair attribution, or statistical support.
- FRESH to TTF changes assignment and persistence together.
- TTF to PES changes start representation and endpoint objective together.
- Running nine full pilot jobs before resolving this attribution problem could consume the budget yet leave the central claim uninterpretable.
- A small pre-registered bridge set may be cheaper than repeating an underidentified experiment after results arrive.

Resolution:

- publish the latest review anchor and complete Pro audit/discussion Prompt;
- require a task-definition verdict, code/protocol findings, test-to-claim map, strongest rejection, claim map, novelty search, and experiment-identifiability verdict;
- allow Pro to retain the three registered variants, replace them with a smaller controlled set, or kill the route;
- keep raw-video gradients and PEFT blocked in every outcome of this review;
- do not reinterpret the completed smoke artifacts as scientific results.

Sources:

- `PRO_PES_STAGE1_CODE_SCIENCE_DISCUSSION_PROMPT_20260712.md`;
- [experiments/persistent-feature-kill-test-20260712.md](experiments/persistent-feature-kill-test-20260712.md);
- [discussion_timeline.md#T20-cache-and-smoke-pass-pro-must-audit-identifiability-before-pilot](discussion_timeline.md).

Reversibility:

- Reversible after the Pro identifiability verdict and author discussion. Any revised pilot must remain within the 10 GPU-hour Stage-1 ceiling and preserve the fixed fully supervised On-TAD boundary.

## DR-028: Repair the FIXED/REMATCH Scientific Contract Before Any Seed Run

Status: active; supersedes DR-027 as the immediate executable decision while
preserving the standard On-TAD task.

Decision:

> Do not submit seed 705 or revise the budget first. Implement the verified
> scientific-contract repairs, rerun Slurm smoke and strict paired profiling,
> then decide whether a new one-seed protocol can be registered.

Accepted readiness findings:

- binary endpoint emission depends on an unsupervised offset;
- REMATCH can move newborn class/start/end losses away from the birth slot on
  the birth step;
- same-step birth+end and final-token short actions can be lost;
- generic training repeatedly reads the locked reporting split;
- instance stream/coordinate normalization, global matching, fragmentation,
  standard-mAP units, and gate schema are not closed;
- formal result provenance, readiness enforcement, counter separation, and
  launcher-consumed census are incomplete.

New evidence that qualifies the source review:

- smoke job `1176737` closes the old crash-only/Slurm/checkpoint uncertainty;
- strict profile job `1176983` closes the missing-profiler uncertainty and
  proves exact pre-training FIXED/REMATCH inference equivalence;
- the same profile rejects the registered pair at `12.572 GPU·hours`, so the
  source review's low-cost assumption is false;
- FP32 with fail-on-nonfinite is retained because the actual AMP smoke produced
  a non-finite first-step gradient;
- the ledger extension and feature-only optimizer construction are already
  repaired, while the formal ledger schema remains incomplete.

Resolution:

1. repair endpoint, newborn binding, and immediate short-action commit;
2. implement fit-only train, symmetric calibration/checkpoint freeze, and
   report-once evaluation;
3. freeze the instance matching/fragmentation and standard-mAP gate contracts;
4. emit one hash-linked result artifact and fail closed on readiness/census;
5. rerun counterexample tests, Slurm smoke, and strict paired profile;
6. only after those pass, register a new budget and one-seed screen.

No further high-cost discussion is needed before the repair. Raw-RGB remains
blocked until the feature-level technical and scientific gates pass.

Sources:

- `PRO_ONTAD_FIXED_REMATCH_SCIENCE_READINESS_REVIEW_20260720.md`;
- `PRO_ONTAD_FIXED_REMATCH_SCIENCE_READINESS_ABSORPTION_20260720.md`;
- [experiments/ontad-science-fixed-rematch-readiness-review-20260720.md](experiments/ontad-science-fixed-rematch-readiness-review-20260720.md);
- [experiments/ontad-science-fixed-rematch-profile-20260720.md](experiments/ontad-science-fixed-rematch-profile-20260720.md).

Reversibility:

- The exact implementation of the repairs and the later budget can be revised
  before seed 705. The prohibition on scientific runs with open P0 contracts,
  repeated locked-reporting access, or unregistered budget overrun is not
  reversible without new evidence.

## DR-029: Replace Fixed Slots with Raw-RGB Dynamic Event Memory and Preserve Official Baselines Read-Only

Status: active design decision; supersedes DR-028 as the current implementation
direction without rewriting the completed FIXED/REMATCH evidence.

Decision:

> Use complete official ActionSwitch, MATR, HAT/OAT, and 2025 HEM repositories
> as untouched read-only reference baselines. Implement compatibility changes
> and A+B family fusions only in separate writable workspaces. The final method
> must consume original RGB and maintain ragged start-conditioned event records
> with learned hierarchical memory rather than a manually fixed slot count.

Reason:

- the fixed-slot route passed its technical contracts but produced low recall
  and severe over-emission, and its capacity safety required dataset-specific
  census plus a hard birth reserve;
- the user's final target is raw-RGB On-TAD or an optional OnVLLM extension, not
  a cached-feature system;
- ActionSwitch, MATR, HAT/OAT, and HEM already contain mature solutions to
  different parts of the problem, so fusion must start from their full official
  designs rather than isolated paper descriptions or simplified rewrites;
- keeping official trees immutable makes baseline fidelity, debugging, and
  attribution possible even when an integration attempt fails.

Resolution:

1. maintain detached full official repositories under an external read-only
   root and assert SHA/tree cleanliness before and after use;
2. create independent writable A/B/C/D replicas and AB/AD/BD/BC/ABD/ABCD
   workspaces;
3. record upstream/source/diff/license manifests and parity tests for every port;
4. treat feature experiments as fast fidelity and mechanism gates only;
5. make raw-RGB ABD the primary paper candidate and keep structured On-TAD spans
   authoritative if an OnVLLM output adapter is added.
6. do not require official native baselines to use OpenTAD; permit only a minimal
   post-fidelity interchange contract, and keep a donor native-only if parity
   cannot be maintained;
7. treat AB/AD/BD/BC as parent-child hypothesis tests, not as a promise to keep
   every donor; ABD and ABCD are removed if their direct-parent evidence does not
   identify a complementary gain.

Sources:

- `docs/superpowers/specs/2026-07-22-raw-rgb-dynamic-event-memory-ontal-design.md`;
- [ideas/dynamic-event-memory-rgb-ontad.md](ideas/dynamic-event-memory-rgb-ontad.md);
- [experiments/ontad-rgb-dynamic-event-memory-design-20260722.md](experiments/ontad-rgb-dynamic-event-memory-design-20260722.md).
- `CURRENT_DIRECTION_AND_GOALS_REPORT_20260722.md`;
- `PRO_RAW_RGB_DYNAMIC_EVENT_MEMORY_REVIEW_PROMPT_20260722.md`.

Reversibility:

- Fusion membership and visual backbone may change after matched evidence. The
  read-only preservation of official baselines, strict causal contract, and
  requirement that final evidence include original-RGB input are not relaxed by
  feature-level results.

## DR-030: Narrow Dynamic Event Memory to an Identifiable Ownership Factorial

Status: superseded by DR-031 as an executable recipe; retained as the historical
disposition of the external review while preserving its scientific narrowing,
final raw-RGB target and read-only donor boundary.

Decision:

> Do not implement the A/B/C/D donor soup or launch full raw-RGB joint training.
> First test, in one matched feature-level code path, whether birth-allocated
> active events and sticky start ownership independently improve strict-causal
> On-TAD over fixed-bank, rematched and Temporal TrackFormer-style controls.

Accepted review findings:

- no single exact competitor was found, but ActionSwitch, MOTR/TrackFormer,
  HEM/Backtrace Mamba and E2E-LOAD/StreamFormer make the broad package an obvious
  combination;
- standard authoritative output is `{start,end,class,score}`; event ID and
  provisional birth remain internal diagnostics;
- generic dynamic memory, active/visual state separation, raw input, interface
  fidelity and Pareto reporting are not headline contributions;
- ABCD is scientifically underidentified; HAT/HEM memory must not precede the
  ownership screen;
- a uniform `0.5` remains a control only, and report-split search remains banned;
- raw-RGB evidence is mandatory eventually, but feature evidence must first
  isolate the event mechanism.

Independent qualifications and rejections:

- the Pro response could not read nine target-version materials and is not a
  complete code/design audit;
- `K×O` is accepted only after `K` is defined as fixed preallocation versus
  birth allocation under the same physical guard, not ragged versus padded code;
- `K0O1` must implement a credible Temporal TrackFormer control;
- `end=t` is rejected: decoded end time and later commit/emit time are distinct;
- late birth needs its own prefix-visible matching/training path; cancel/rebirth
  and exchange-equivalent same-class overlap remain open algorithm questions;
- `K=32`, `B=128`, loss weights, 30k updates, seed 3407 and numerical effect
  gates are unfrozen until direct-parent variance and learning/resource curves;
- a public immutable snapshot is required for reproducibility but runs in
  parallel and cannot displace the model-first priority;
- one unconverged single-seed failure does not kill the scientific question;
  after readiness and variance are closed it may kill further resource spend.

Executable matrix:

```text
K0O0 = fixed preallocated bank + per-prefix rematching
K1O0 = birth-allocated packed set + per-prefix rematching
K0O1 = fixed preallocated bank + sticky owner (Temporal TrackFormer control)
K1O1 = birth-allocated packed set + sticky owner (candidate)
```

All arms share the same encoder, decoder depth, physical guard, data exposure,
successful updates, calibration rule and evaluator. `K1O1` survives only if it
beats both direct parents, reduces wrong-start/owner-swap errors in the relevant
strata, and does not obtain the gain through excess delay, overflow or memory.
Exceeding the old FIXED result alone is insufficient.

Parallel work allowed before this gate:

1. minimal ActionSwitch/MATR native fidelity and immutable donor SHAs;
2. synthetic overlap, reversed/nested ends, late birth, cancel/rebirth,
   overflow and future-perturbation tests;
3. a light raw-backbone prefix/gradient/cache/latency smoke with no performance
   claim or formal joint-training budget.

After a pass, run the fixed-budget hierarchy × retention factorial and raw
frozen/adapter/joint comparisons in parallel, then multi-seed and one
annotation-audited external dataset. OnVLLM remains a separate future project.

Sources:

- `PRO_RAW_RGB_DYNAMIC_EVENT_MEMORY_REVIEW_PROMPT_20260722.md`;
- `PRO_RAW_RGB_DYNAMIC_EVENT_MEMORY_REVIEW_ABSORPTION_20260722.md`;
- [ideas/dynamic-event-memory-rgb-ontad.md](ideas/dynamic-event-memory-rgb-ontad.md);
- [experiments/ontad-rgb-dynamic-event-memory-design-20260722.md](experiments/ontad-rgb-dynamic-event-memory-design-20260722.md).

Reversibility:

- Numerical budgets, backbone and memory design are deliberately reversible
  after profiling. The standard task, strict causality, read-only donor policy,
  standard output, model-first priority and ban on report-split tuning are not
  relaxed without new scientific evidence.

## DR-031: Retire the OpenTAD K×O Recipe and Use Official MATR as the Sole Parent

Status: active; supersedes DR-030 as the executable route while retaining its
strict-causal task and raw-RGB final target.

Decision:

> Abandon the old FIXED/REMATCH-derived 12-epoch K×O experiment. Build the new
> event model as a controlled child of the exact official MATR implementation,
> preserve MATR's THUMOS14 training recipe, and use ActionSwitch only as the
> start-transition donor and native comparison.

Reason:

- the discarded K×O code inherited seed 705, 12 epochs, batch one, SigLIP2
  features, AdamW `2e-4` and a six-entry guard from a low-performing internal
  route rather than a front-line official method;
- those changes would confound the proposed ownership mechanism with a new
  feature representation, optimizer, schedule, batching regime and detector;
- MATR is the closest official parent because it already performs standard
  On-TAD with current-end and past-memory-start decoders on THUMOS14/MUSES;
- ActionSwitch directly supplies the complementary class-agnostic state-change
  idea, but its released objective and eight-epoch recipe are not a substitute
  for MATR's interval-localization setting.

Resolution:

1. freeze read-only MATR SHA
   `ba05a98d451b3541c1a5377026f17dc1102fa217` and ActionSwitch SHA
   `838a6ccbd8f2cce414688ff2843380d712aa7b89`;
2. require native MATR parity on the official supplied features and defaults;
3. keep exact `native_matr` outside the factorial, and use four eventized cells
   `B0O0/B1O0/B0O1/B1O1`, where `B` is immediate prefix-visible birth and `O`
   is sticky owner identity;
4. keep official MATR's 100 epochs, batch 64, seed 52, segment 64, ten queries,
   seven memory segments, optimizer/scheduler/loss/evaluation settings for all
   matched arms;
5. change only the registered event-birth and owner-conditioned lifecycle
   mechanisms; apply any stricter causal post-processing adaptation symmetrically
   and prove it event-equivalent to generation-order execution;
6. treat HAT and HEM as later memory references, not extra ingredients in the
   first candidate;
7. authorize raw-RGB work only after the candidate beats both direct parents
   and passes association, causality, latency and resource checks.

The uncommitted OpenTAD K×O implementation and launcher were never deployed and
are not migrated. Completed FIXED/REMATCH results remain historical negative
evidence only.

Sources:

- [experiments/ontad-matr-official-parent-event-model-20260722.md](experiments/ontad-matr-official-parent-event-model-20260722.md);
- [papers/matr2024-memory.md](papers/matr2024-memory.md);
- [papers/actionswitch2024-state.md](papers/actionswitch2024-state.md).

Reversibility:

- The exact birth/ownership implementation can be revised after direct-parent
  learning curves. The official-parent fidelity requirement, strict causality,
  standard output and raw-RGB final evidence cannot be removed without an
  explicit task change.

## DR-033: Release EventMATR Only Through a Real Official-Data Smoke

Status: active implementation gate.

Decision:

> Accept the current EventMATR implementation as locally contract-complete, but
> do not release any 100-epoch performance job until one real official MATR
> training batch passes native plus all four eventized lanes under an exact,
> clean source identity. Do not substitute another feature representation.

Reason:

- local unit/contract tests cannot demonstrate that the official THUMOS14
  pickles, annotation conventions, batch collation and GPU graph work together;
- the former smoke checked only imports and synthetic tests and could have
  released five expensive jobs without proving forward/backward, optimizer or
  checkpoint compatibility;
- exact native parity requires the official MATR RGB+flow 4096-dimensional
  feature and label package, not SigLIP2 or a convenient OpenTAD cache;
- every run must remain attributable to one clean commit, one Git tree and one
  frozen manifest rather than silently mixing worker sources;
- current N16R4 storage inventory did not find the exact official package, and
  its Google Drive endpoint timed out, so a formal job would fail before
  answering a model question.

Resolution:

1. the smoke uses only `THUMOS14Dataset(subset="train")` and never mounts test;
2. it runs `native_matr`, `B0O0`, `B1O0`, `B0O1` and `B1O1`, each through real
   forward, full loss backward, one Adam step, finite-gradient checks and strict
   checkpoint reload;
3. eventized lanes additionally require nonzero transition-head and owner-head
   gradients;
4. the launcher, smoke, five training workers and finalizer bind commit, tree,
   manifest SHA-256 and smoke receipt; dirty/untracked/mismatched source fails;
5. a PASS releases the native lane and four 100-epoch eventized lanes in
   parallel; every lane uses full official validation/train data and only the
   terminal epoch-100 checkpoint;
6. training performs no calibration or test-driven selection; one locked test
   per frozen lane is a later, separately submitted action;
7. until the exact official data are staged, record the blocker and do not
   launch substitute or partial-data experiments.

Local evidence on 2026-07-23:

- implementation commit `64d7f78dd8ed1436bac08ebfb03b51c90142129b`,
  tree `bc6f5057e954ed448dcb5173489eba7c4dcba27d`, manifest SHA-256
  `B6D3B51A14253E66A9D5C110F5B08FDAF96C31EB48FB2DF9A3B57CD5E61FB1C9`;
- complete suite `62 passed`;
- official-protocol validator PASS;
- Python compile, all Git-Bash entrypoint syntax and `git diff --check` PASS;
- no Linux real-data smoke, 100-epoch result or performance claim yet.

Source:

- [experiments/ontad-matr-official-parent-event-model-20260722.md](experiments/ontad-matr-official-parent-event-model-20260722.md).

Reversibility:

- Smoke implementation details may be optimized if they preserve the same
  scientific checks. Official-data parity, zero test access during training and
  exact-source fail-closed behavior are not relaxed for convenience.

## DR-034: Withdraw the Mis-Pasted Review and Revalidate the Correct On-TAD Review

Status: active provenance correction; does not alter the executable model gate.

Decision:

> Withdraw the previously supplied unrelated attachment and every conclusion
> derived from it. Treat only `Raw-RGB Dynamic Event Memory On-TAD 独立深度审判`,
> SHA-256 `ACA5BB6E9950993F170922250F6C915D41AF72BE027984406714315A92D96B8D`,
> as the valid external review. Keep its verdict as partial acceptance after
> rechecking it against the current EventMATR implementation.

Reason:

- the user explicitly confirmed that the intervening attachment was copied in
  error, so it cannot be retained as a scientific source, transferable design
  guardrail or archived experiment for this project;
- the newly supplied correct file is byte-identical to the review already
  absorbed at M55: `70,393` bytes, `1,412` lines and the same SHA-256;
- since M55, the target-version visibility gap has been materially reduced by
  exact EventMATR commit `64d7f78dd8ed1436bac08ebfb03b51c90142129b`, its
  clean tree/manifest and local plus N16R4 contract tests;
- performance evidence remains absent because the exact official MATR THUMOS14
  package has not passed the real-data smoke, so implementation verification is
  not a reason to upgrade the scientific verdict to PASS;
- the review's identifiability principle remains valid, but DR-031 deliberately
  replaces its historical fixed-slot `K×O` prescription with a slot-free `B×O`
  test of birth timing and sticky ownership under the exact official MATR parent.

Resolution:

1. delete the unrelated detailed review record and its current-Wiki source-map,
   index and query-pack entries;
2. retain one correction milestone so the withdrawn input cannot be mistaken for
   an accepted project source after context compaction;
3. use `PRO_RAW_RGB_DYNAMIC_EVENT_MEMORY_REVIEW_ABSORPTION_20260722.md` as the
   canonical claim-by-claim disposition, including the 2026-07-23 revalidation;
4. fully retain task narrowing, standard output, read-only donors, no module soup,
   owner-association diagnostics and the feature-to-raw evidence boundary;
5. do not freeze the review's arbitrary capacities, budgets, loss weights, seed,
   backbone or effect thresholds, and do not revive semantic slots as the main
   model factor;
6. keep DR-031 and DR-033 as the executable route: official-data five-lane smoke,
   then native MATR plus four official-setting `B×O` 100-epoch runs.

Source:

- `PRO_RAW_RGB_DYNAMIC_EVENT_MEMORY_REVIEW_ABSORPTION_20260722.md`;
- [experiments/ontad-matr-official-parent-event-model-20260722.md](experiments/ontad-matr-official-parent-event-model-20260722.md).

Reversibility:

- Future scientific evidence can change the model or experiment design. A source
  explicitly withdrawn by the user cannot be restored as evidence without a new,
  explicit request.

## DR-035: Freeze EventMATR v1, Separate Protocol-Zero from Learning-Zero, and Revise the Learning Core

Status: active scientific revision; supersedes DR-031's `B×O` candidate as the
paper mainline after preserving all completed v1 evidence.

Decision:

> Preserve `92cf34aa07bebee2a7a7e3661431d5055804b29b` as EventMATR v1 and do
> not erase its runs. Before calling v1 an all-background neural failure, replay
> its checkpoint in eval mode from the first prefix with a fresh writer, because
> the formal training path disabled runtime and therefore produced an empty
> ledger by construction. For the next model, retain the dynamic EventRecord,
> lifecycle, end/emit separation and immutable ledger, while replacing the
> single crossing, dense background loss and prototype-owner training with a
> temporally stable birth assignment, event-normalized censored hazards,
> identity-locked trajectories and training/inference-aligned ragged unroll.

Reason:

- the external Pro review, attachment SHA-256
  `1BC0FDDA22B694F1EEA4485DF9478D6E252FB22AE4E53156F1B2903814023513`,
  correctly confirmed the sparse-loss, runtime, crossing and owner mismatches;
- independent local inspection confirmed that `true_duration` enters runtime,
  `DynamicEventMemory.step` is non-differentiable, the training writer reads only
  the runtime ledger, `Path.touch()` does not truncate old predictions, starts
  may be negative and `owner_query_id % Q` aliases records when `R>Q`;
- zero train mAP is therefore not by itself proof of a learned all-background
  network, although the class-imbalance risk remains and must be measured on a
  real batch;
- the review's TH direction is scientifically promising but its per-prefix
  Hungarian does not yet define a stable event-level birth hazard, and its
  cancel/rebirth, censor encoding and data split require revision;
- the four claimed sandbox code artifacts were not supplied, so their hashes,
  9/9 tests and integration patch are not accepted as project evidence.

Resolution:

1. freeze v1 source/checkpoints and run an eval-only full-prefix replay plus
   positive/negative gradient audit before any new training;
2. treat `true_duration`, stale writer output, negative start and runtime/ledger
   mismatch as P0 scientific-contract defects for the v2 branch;
3. adopt `N/R/T/H/TH` as the intended factorization only after a stable temporal
   pre-birth assignment and differentiable ragged unroll pass integration tests;
4. do not freeze the review's rank, grace window, loss weights, teacher schedule,
   effect sizes or epoch gates without pilot evidence;
5. keep active-event memory in the core and defer learned visual hierarchy until
   the trajectory-hazard mechanism passes;
6. keep raw RGB as the final target, but do not use it to mask an unproven
   feature-level event mechanism;
7. distinguish hard correctness stops from soft mechanism diagnostics, then run
   seed-52 five-lane 5/10/20-epoch pilots before any three-seed 100-epoch study.

Source:

- `PRO_TH_EVENTMATR_CODE_REVIEW_ABSORPTION_20260728.md`;
- `PRO_TH_EVENTMATR_CODE_REVIEW_PROMPT_20260728.md`;
- EventMATR v1 commit
  `92cf34aa07bebee2a7a7e3661431d5055804b29b`.

Reversibility:

- The precise TH architecture remains reversible until Stage D1. The strict
  causal metadata boundary, fresh-writer evaluation, v1 provenance and ban on
  treating missing sandbox artifacts as verified code are not relaxed.

## DR-036: D0 Proves Dual Failure — Sparse Hazard Learning and Unaligned Sticky Runtime

Status: active D1 implementation decision; closes DR-035 Stage D0.

Decision:

> Close EventMATR-v1 D0 as a valid diagnostic PASS, not a paper-performance
> result. The frozen network has nonzero lifecycle/owner gradients and can emit
> intervals, so do not label it a uniform all-background collapse. Its sparse
> birth/end learning is still badly under-scaled, and sticky ownership creates
> records without a learned termination process. D1 must therefore combine
> event-normalized censored hazards with training/inference-aligned ragged
> trajectories; neither threshold adjustment nor runtime-only patching is
> sufficient.

Reason:

- D0 exact `ca914f3ea337d1e8f0f005d394a05ec81edc0ee7` completed four receipts and
  one pair receipt with `test_access=false` and unchanged checkpoints;
- every lane has finite nonzero event-transition, owner-attention/state and
  birth/end gradient evidence;
- birth/end positives are about `0.1477%`, while mean learned logits are much
  more negative than the corresponding constant-predictor optimum;
- B1O0 closes `1,716/2,116` born events with four active after EOS, whereas
  B1O1 closes only `6/1,854` and leaves `1,407` active after EOS;
- duplicate, non-positive, sequence, class, emission-order and capacity
  invariants pass, but ten negative-start rows and future-duration/EOS metadata
  remain D1 hard defects;
- all D0 mAP values are train-replay diagnostics with
  `strict_causal_paper_result_valid=false`.

Resolution:

1. use B1O0 as the functional v1 reference, not as the paper candidate;
2. remove `true_duration`, complete-video timing and offline EOS from runtime;
3. implement stable pre-birth assignment and chronological differentiable
   ragged unroll before claiming trajectory hazards;
4. implement event-normalized interval-censored first-birth and right-censored
   owner-conditioned end;
5. add identity lock with cancel/reacquisition rather than retaining an
   incorrectly born record indefinitely;
6. require negative-start, same-class overlap, EOS, Q+1 birth and R>Q tests;
7. after D1 contracts pass, release only the registered seed-52
   `N/R/T/H/TH` pilots; keep locked test, multi-seed, hierarchy and raw-RGB
   blocked.

Source:

- [experiments/ontad-matr-official-parent-event-model-20260722.md](experiments/ontad-matr-official-parent-event-model-20260722.md);
- D0 pair receipt from
  `eventmatr_v1_d0_seed52_20260728_203713_ca914f3`;
- `PRO_TH_EVENTMATR_CODE_REVIEW_ABSORPTION_20260728.md`.

Reversibility:

- The exact D1 implementation and pilot hyperparameters remain reversible.
  D0 provenance, the dual-failure diagnosis and the strict-causal metadata
  boundary may change only if a new audited receipt contradicts them.

## DR-037: Gate EventMATR D1 by Problem Truth, Repair Specificity, and Composite Novelty

Status: active pre-experiment decision.

Decision:

> Evaluate every surviving EventMATR idea against a named failure and a direct
> counterfactual. Use N/R/T/H/TH only as the first attribution layer: R tests
> chronological train/inference state alignment, T tests trajectory identity,
> H tests censored sparse lifecycle risk, and TH tests their complementarity.
> Do not claim novelty from any individual known primitive.

Reason:

- D0 separately identifies extreme birth/end sparsity and an unclosed sticky
  lifecycle, so loss repair and trajectory repair require separate tests;
- the official train annotations contain same-class repetition in `180/200`
  videos and place `50.18%` of instances within 64 frames of a same-class
  neighbor, but true same-class overlap involves only `9/3,007` instances in
  two videos; T is therefore a repeated-instance/reacquisition hypothesis, not
  an abundant-overlap premise;
- local D1 mechanism tests show the intended gradients, stable assignment,
  dynamic capacity, cancel/reacquisition and ledger invariants can execute, but
  synthetic solvability does not establish dataset effect;
- exact-source official-data smoke `1200932` proves the registered D1 source
  can execute all five lanes with finite gradients without locked-test access;
- the novelty audit finds strong primitive overlap with survival likelihood,
  tracking-by-query, same-class online TAL and autoregressive/scheduled-sampling
  training;
- the only conditionally defensible contribution is their task-specific
  composition as one audited strict-causal Online-TAD lifecycle process.

Resolution:

1. run fixed seed-52 5/10/20-epoch train-only pilots in that order, releasing a
   longer horizon only after complete finite receipts at the preceding gate;
2. interpret train-prefix mAP only as optimization diagnostics and keep
   `strict_causal_paper_result_valid=false`;
3. require later ordinary-duration-free-BCE, detached/oracle/predicted unroll,
   state-machine removal, near-neighbor repeat, same-class overlap and
   truncation counterfactuals;
4. demote R if it does not change downstream error-state recovery, T if it does
   not improve identity lifecycle metrics, H if ordinary causal BCE matches it,
   and TH if gains disappear under parameter/training-budget controls;
5. retain CESR, PCEH, CRS-EPS and PETAL only in the bounded substrate/component
   roles recorded in the D1 idea inventory;
6. keep locked test, multiple seeds, raw RGB, distillation and threshold search
   blocked until the D1 mechanism gate passes.

Sources:

- [experiments/eventmatr-d1-preexperiments-20260728.md](experiments/eventmatr-d1-preexperiments-20260728.md);
- `.aris/traces/novelty-check/2026-07-28_run01/`;
- EventMATR D1 source `1f4bb29ad58dddcc33f6ff2bdc57a5934ee5c53d`;
- exact-source smoke receipt from Slurm job `1200932`.

Reversibility:

- Candidate ranking and architecture details remain reversible after direct
  counterfactuals. The causal boundary, source/receipt audit, and prohibition
  against promoting train-prefix diagnostics to paper evidence are not relaxed.

## DR-038: Pass Execution, Mark Scientific Sufficiency Unresolved, and Hold Longer D1 Training

Status: active diagnostic hold.

Decision:

> Accept array `1200955` as an exact-source execution PASS, but do not release
> the registered 10/20-epoch jobs. R and T collapse terminal emission to zero;
> H and TH preserve only a very small proposal subset. Diagnose the causal
> lifecycle and emission path before spending a longer training budget. This is
> a reversible diagnostic hold, not a retroactive failure under a quantitative
> scientific threshold: no such effect-size/coverage threshold was registered.

Reason:

- all five jobs complete `0:0`, pass source/receipt/checkpoint validation, and
  retain `test_access=false`;
- every trained D1 objective has finite decreasing loss, so the result is not a
  dead gradient or numerical divergence;
- N emits `21,256` terminal proposals over `176/200` videos with diagnostic
  train-prefix average mAP `7.0957`;
- R and T emit no terminal proposal and score zero after their output files
  become empty from epoch 3 onward;
- H and TH emit only `333/271` proposals over `47/42` videos, with average mAP
  `4.5066/3.9445`;
- H and TH improve the strict `0.7` overlap diagnostic to `2.9937/3.1464`
  versus N's `2.0648`, which is a useful boundary-precision hint but cannot
  compensate for the severe coverage loss;
- fewer false-cancel groups or active ragged records are ambiguous while the
  method suppresses nearly all output.
- the manifest names recall, lifecycle, identity, calibration and systems
  metrics, but the pilot finalizer does not require most of them; the original
  science language is qualitatively sensible but not quantitatively
  operationalized.

Resolution:

1. preserve all five terminal checkpoints and receipts as immutable evidence;
2. run a strict-causal train-only chronological replay with stage-by-stage
   candidate, birth, assignment, cancel, end, gate and emission counts;
3. add score distributions, calibration, latency, closure, duplicate,
   reacquisition, owner-switch, fragmentation and nearby-repeat metrics;
4. distinguish loss-weight competition, gate calibration, state-machine
   starvation and writer/runtime mismatch without threshold search;
5. before replay, prospectively register complete metric availability,
   functional-liveness handling, matched-coverage comparisons and
   effect-size/sample-size tolerances; do not derive pass cutoffs from the
   observed proposal counts;
6. reconsider the 10-epoch release only after R/T's empty output is explained
   and H/TH lifecycle evidence is complete;
7. keep locked test, multiple seeds, raw RGB and threshold changes blocked.

Sources:

- [experiments/eventmatr-d1-preexperiments-20260728.md](experiments/eventmatr-d1-preexperiments-20260728.md);
- five pilot receipts under
  `pilots_seed52_20260728_1f4bb29/e5/{N,R,T,H,TH}`;
- Slurm array `1200955`.

Reversibility:

- The hold can be lifted by a source-exact diagnostic receipt that explains the
  emission collapse and satisfies a prospectively frozen lifecycle gate. The
  current five-epoch values remain diagnostic and cannot be promoted to paper
  evidence.

## DR-039: Accept the Replay Root Cause, Reject Unchanged Longer Training, and Open D1.1 Repair

Status: active redesign hold.

Decision:

> Accept strict-causal checkpoint replay array `1203224` as an integrity PASS
> and as sufficient evidence for root-cause discussion. R/T do not fail because
> START is absent or because the writer applies a score threshold; they fail
> because nearly every predicted record is classified as owner BACKGROUND and
> cancelled, while owner END never wins. H/TH show that censored owner risk can
> restore END-to-emission liveness, but their timing, coverage, recall and
> cancel/reacquisition churn are not a solved detector. Do not spend the
> registered 10/20-epoch budget on the unchanged implementation. Authorize only
> a minimal D1.1 repair and its train-only mechanism tests before a new pilot.

Reason:

- replay source `6a23ab3a3711bc1ecb5a2fe442e302964ee3afb8` passes
  `54/54` remote tests; all four array tasks complete `0:0`;
- every replay uses evaluation mode, predicted-only tracks, `200` observed EOS
  markers, no locked test, no threshold search and an unchanged checkpoint;
- R/T produce `46,935/5,670` births but cancel `46,892/5,667`, have zero owner
  END and therefore zero emissions;
- H/TH produce exactly `46/519` END and `46/519` emissions, excluding a writer
  failure and showing a real objective-dependent liveness intervention;
- H/TH still cover only `6/26` train videos and recall only
  `0.166%/0.998%` at temporal overlap `0.3`;
- T/TH record `4,017/3,611` reacquisitions but still cancel
  `99.947%/90.233%` of births, so identity handling is currently churn rather
  than a closed trajectory;
- training associates a predicted record to a ground-truth event only through
  exact oracle-query equality; unmatched predicted records are supervised as
  owner background and excluded from true-event end hazard;
- pre-birth temporal history and hazard risk windows reset at each batch
  boundary, truncating the nominal causal window;
- native memory-gate behavior is similar across lanes and is not the immediate
  factor-specific bottleneck, but its train/inference teacher mismatch still
  requires a fixed counterfactual.

Resolution:

1. preserve the five-epoch checkpoints, replay summaries and compressed traces
   as read-only evidence;
2. implement causal predicted-track/visible-event assignment, source-stratified
   supervision accounting, explicit cancellation semantics and cross-batch
   causal history;
3. retain event-normalized interval-censored birth and right-censored end as
   active components, but require conditional recall/timing evidence rather
   than background-dominated calibration alone;
4. test the repair with synthetic, batch-boundary, predicted-track and
   cancel/reacquisition counterfactuals, then one batch and one epoch of real
   train data;
5. prospectively register a new five-epoch seed-52 train-only pilot only after
   those contracts pass;
6. use the GitHub evidence address for one external Pro scientific discussion
   before freezing the D1.1 pilot matrix;
7. keep the locked test, multiple seeds, raw RGB, distillation and threshold
   changes blocked.

Sources:

- [experiments/eventmatr-d1-preexperiments-20260728.md](experiments/eventmatr-d1-preexperiments-20260728.md);
- replay code commit
  `6a23ab3a3711bc1ecb5a2fe442e302964ee3afb8`;
- Slurm tests `1203223` and replay array `1203224`;
- `.aris/traces/experiment-audit/2026-07-29_run02/`.

Reversibility:

- The redesign hold may be replaced only by source-exact counterfactual evidence
  that separates predicted-track association, cancellation supervision,
  cross-batch history and censored risk. It is not lifted by more epochs alone.

## DR-040: Keep the D1.1 Mechanism Gate Failed and Diagnose Predicted Association Before Any Pilot

Status: active diagnostic hold.

Decision:

> Treat job `1204061` as a completed one-epoch training trajectory whose
> mechanism receipt failed, not as a training crash. Gradients, supervised
> lifecycle risks, runtime birth/end/emission/cancellation/reacquisition and the
> terminal checkpoint are live, but no learned predicted birth was ever
> associated with a prefix-visible event. Keep the one-epoch gate unchanged and
> keep every five-epoch or longer pilot blocked. Authorize only a read-only,
> predicted-only, strict-causal association-barrier scan of the saved checkpoint.

Reason:

- all `3,270` training batches completed and wrote a
  `2,150,323,335`-byte checkpoint;
- transition/owner gradient norms were approximately `62.2584/51.5407`;
- capacity exhaustion was exactly zero and the locked test path was absent;
- predicted-unmatched association was non-empty, but
  `predicted_associated` never appeared;
- the old metric emitter created source keys only when a source occurred, so
  both missing predicted-associated keys mean an observed zero count;
- teacher births kept other lifecycle losses and transitions live, which shows
  why generic non-zero gradients cannot prove that the learned inference path
  is trained;
- the gate contains no detection effect-size threshold. Its non-zero
  predicted-associated condition is a direct liveness condition for the D1.1
  mechanism and should not be removed to force a pass.

Resolution:

1. freeze training source `de0837cf38e05d65a40f0744b863056edc2f433a`,
   tree `91d998e787c42895b08571ef0ed5ab5af5458a47`, and checkpoint
   SHA-256
   `a5e686f3e4800e654e9ed4366f13c298697ab6b076c9813f10e0f5241ba546b5`;
2. use instrumentation source
   `f37e9d191a06a8a703714a0cd857180c21fc069e`, tree
   `34b2a81b2ee4a3df1d2523dcaeb0189200f68774`, and corrected scan source
   `7687efe03981aeb1ee3cb62ae2fd94d6e6ca1dfa`, tree
   `a85303e04f810fd582db848ecc6ce69fc7fbf196`;
3. emit explicit zero metrics for all registered track sources so a future
   zero cannot be confused with a spelling or parser error;
4. decompose active START, rising birth, visible target, class mismatch,
   start-distance rejection, admissibility, ambiguity and assignment globally
   and per video;
5. keep ground truth strictly after forward and label the scan as a
   necessary-condition opportunity audit, not a reconstruction of teacher
   ownership;
6. preserve the frozen 64-row physical batch and allow its padding mask only as
   a verified lifecycle no-op after an already observed current-stream EOS;
   require unchanged checkpoint/options, existing dataset caches, zero capacity
   exhaustion and closed global/per-video counts;
7. record scan completion as `DIAGNOSTIC_COMPLETE` while preserving
   `one_epoch_mechanism_gate_status=FAIL_UNCHANGED`;
8. choose a minimal model repair only after the dominant barrier is measured;
   do not search/lower thresholds or change data, backbone, seeds or budget.

Sources:

- [experiments/eventmatr-d11-structural-repair-design-20260729.md](experiments/eventmatr-d11-structural-repair-design-20260729.md);
- Pro attachment SHA-256
  `c2cd3ce0cc52791aecc6cda9d85ef108ce4e18703c0debf51026196817231bd1`;
- Slurm job `1204061`;
- code commits `f37e9d191a06a8a703714a0cd857180c21fc069e` and
  `7687efe03981aeb1ee3cb62ae2fd94d6e6ca1dfa`.

Reversibility:

- The diagnostic hold may be lifted only by a new source-exact training run
  whose one-epoch mechanism receipt passes the unchanged learned-path liveness
  condition. A read-only scan, a teacher-forced path, a non-zero train-prefix
  score or extra epochs cannot retroactively pass job `1204061`.

## DR-041: Preserve the Old FAIL but Recheck It at a Valid Optimization Exposure

Status: completed; functional gate failed and superseded by DR-042.

Decision:

> Keep job `1204061` failed because predicted-associated supervision was zero,
> but withdraw the stronger interpretation that the repaired mechanism has
> already failed to learn. All 3,270 updates used `1e-8`, while the registered
> first warmup rate `3.34e-6` was reached only after training. Run exactly one
> fresh seed-52 epoch at `3.34e-6` with every other scientific decision
> unchanged, and require a training receipt, deterministic predicted-only
> terminal scan and exact parameter-delta audit before any five-epoch design may
> be frozen.

Reason:

- the old training path, gradients, supervision and checkpoint were live, so
  its finalizer failure is a real learned-path liveness failure rather than a
  crash;
- deterministic terminal scan `1204338` found all 2,033,630 START margins
  negative and no terminal lifecycle, but relative oracle-query ranking remained
  non-random;
- same-trajectory trace `1204354` found 18,021 train-mode predicted births,
  4,042 candidate pairs, 3,900 class mismatches, 142 start-distance rejections
  and zero assignments;
- exact audit `1204424` measured very small but non-zero parameter movement and
  confirmed the average update learning rate was approximately `1e-8`;
- terminal and train-mode observations differ in mode, checkpoint age and
  stochasticity, so neither alone identifies a unique structural cause;
- one controlled effective-dose run separates underexposure from an unchanged
  structural failure without changing a threshold, model, loss, data order,
  seed or budget.

Resolution:

1. use training source
   `4116df154014915cc190eec0e108a94c8df5f762`, tree
   `2114f097eefc24e3a776147fc201711878464ad2`;
2. require exactly 3,270 actual optimizer updates at `3.34e-6` and a
   hash-bound, sequential per-update trace;
3. retain the original positive learned-path liveness requirements and zero
   capacity exhaustion;
4. add a separate deterministic terminal requirement for positive
   START/birth/association, runtime birth/cancel/end, END-to-immutable-emission
   equality and zero capacity exhaustion;
5. reconstruct exact initialization and require closed optimizer accounting
   plus non-zero transition and owner parameter deltas;
6. leave Event birth/end scalar thresholds absent; retain inherited flag,
   class and non-maximum-suppression settings unchanged;
7. apply no detection-effect threshold and do not treat training-prefix
   detection values as paper results;
8. keep locked test, multiple seeds, raw RGB, threshold search and all longer
   pilots blocked until the three-artifact combined gate passes;
9. if the training gate fails, use diagnostic source
   `8ceee52c102913ac08571c1bd876fb370f9b5797` only to record the failed
   effective-dose terminal barrier; that diagnostic status cannot pass the
   combined gate.

Sources:

- [experiments/eventmatr-d11-structural-repair-design-20260729.md](experiments/eventmatr-d11-structural-repair-design-20260729.md);
- Slurm jobs `1204338`, `1204354`, `1204424`, `1204465` and controlled run
  `1204468`;
- code commits `d14ab88f90df1ab960dbc164700e884ff922f0c9`,
  `daea1149ecfc2cf7caedc627b4381832c5005b53`,
  `4116df154014915cc190eec0e108a94c8df5f762` and
  `8ceee52c102913ac08571c1bd876fb370f9b5797`.

Reversibility:

- Passing the combined three-artifact gate may replace this controlled hold
  with a prospectively frozen five-epoch mechanism plan. It still cannot
  establish performance, generalization or novelty.
- Failure preserves the hold and routes only to the measured terminal barrier.
  It does not prove that strict-causal EventMATR is impossible or authorize
  lowering a decision boundary.

Outcome:

- job `1204468` completed all 3,270 registered updates at `3.34e-6`;
- parameter audit `1204510` confirmed material model, transition-head and
  owner-head movement;
- terminal scan `1204508` found zero positive START decisions across
  2,033,630 candidates and therefore zero runtime lifecycle;
- conditional class/time ranking remained non-random, so invalid optimization
  exposure is no longer a sufficient explanation, while network death,
  impossibility and performance conclusions remain unsupported.

## DR-042: Separate D1 Birth Risk from Four-State Background Competition

Status: implemented; subsequent D1.3/D1.4 structure gate failed and is
superseded by DR-043.

Decision:

> For `d1_censored` only, replace the four-state START margin used as birth
> evidence with one independently trained binary birth-risk logit. Use the same
> scalar for event-normalized interval-censored birth learning, causal temporal
> assignment, rising-edge preview and runtime birth. Keep the post-birth
> ternary owner decoder as the authority for cancel/continue/end. Fix the birth
> decision at strictly positive log-odds, reject older checkpoints by schema,
> and rerun only the one-epoch mechanism gate before any longer development
> experiment.

Reason:

- valid effective-dose training moved parameters but did not make any absolute
  four-state START margin positive;
- the same checkpoint retained class-conditioned query and temporal structure,
  showing that association inputs are not uniformly random;
- reweighting the old competitive state leaves birth coupled to overwhelming
  background, while lowering/searching a threshold would not test whether the
  learning architecture is repaired;
- one independent scalar is the smallest intervention consistent with the
  already accepted censored-risk objective and creates one train/assignment/
  inference semantic instead of three mismatched birth rules.

Resolution:

1. exact code source is
   `69039b990822d689592155d48f57b619ecd8e25e`, tree
   `7fc5e014b919d033468c9b58afdedc33afc65646`;
2. checkpoint schema is `eventmatr_d12_independent_birth_v1`; older D1 weights
   must fail closed;
3. deterministic tests must prove independent birth under four-state
   background dominance and finite shared/birth/owner gradients;
4. remote validation must include the complete test suite and official
   training-batch forward/backward/optimizer/checkpoint reload;
5. the fresh seed-52 one-epoch run retains the official train features, data
   order, 3,270 updates and fixed `3.34e-6` exposure;
6. the training receipt, complete predicted-only terminal scan and exact
   parameter audit must all pass on one hash-bound checkpoint;
7. no threshold search, locked test, multiple seeds or raw RGB is authorized;
8. a mechanism pass may release only frozen development pilots. It cannot
   produce a paper performance claim.

Official-comparability boundary:

- one-epoch mechanisms, training-prefix scores and 5/10/20-epoch pilots are
  never official paper performance;
- the independent binary birth head is a necessary liveness repair, not a
  standalone novelty claim. Any eventual contribution must be the validated
  strict-causal composition of censored risk, causal assignment, explicit
  ownership lifecycle and train/inference state alignment;
- paper evidence requires a separately prospectively frozen, matched
  100-epoch native-MATR/EventMATR comparison with the same official split,
  extracted RGB-plus-flow features, optimization, post-processing, terminal
  checkpoint policy and evaluator under the strict causal boundary;
- locked test access is permitted only after a future structure gate passes and
  that full-budget comparison contract is frozen.

Reversibility:

- failure routes only to the newly measured barrier and cannot be repaired by
  lowering the zero-log-odds decision boundary;
- pass authorizes protocol freezing, not a superiority, generalization or
  novelty claim.

## DR-043: Fail D1.4, Preserve Decision-Aligned Birth as an Intervention, and Isolate Owner-End Failure

Status: completed structure gate; analysis-only hold.

Decision:

> Select neither D1.4 arm. Preserve the decision-aligned interval bag only as
> evidence that absolute birth inactivity is causally intervenable, not as a
> deployable model. Before any further training, use a frozen-checkpoint,
> train-only counterfactual owner unroll to separate failure of the learned
> right-censored end decision from failure to transport identity through
> predicted tracks. Do not lower the birth boundary, search thresholds, add
> seeds, run longer pilots, access locked test, or claim official performance.

Reason:

- normalized survival leaves all `2,033,630` individual birth logits
  non-positive even though the mean oracle interval event probability is
  `0.444412`; an aggregate interval likelihood is not sufficient to cross the
  runtime decision boundary;
- the decision-aligned bag makes `923,712` individual logits positive and
  produces `30,002` predicted-only births, proving that the birth barrier is
  intervenable;
- it also produces `29,974` cancellations and zero ends, emissions and
  reacquisitions, so birth liveness alone is not a closed lifecycle;
- during training, approximately `332,525` births and `332,217`
  cancellations coexist with only four learned ends;
- `305,064` false-track cancel groups outnumber the `3,003` positive owner
  assignments by about `101.6:1`; the current all-group owner-state average and
  target-identity-only end risk therefore face a measured source and
  train/inference mismatch;
- both arms changed the birth, transition, shared-fusion and owner modules, used
  identical official training artifacts, and exhausted no capacity, excluding
  a no-update, data-drift or capacity explanation;
- the post-forward assignment count is an opportunity diagnostic, not a
  reconstructed runtime identity path, and terminal reacquisition remains zero.

Resolution:

1. D1.4 exact source is
   `fa27b3657b72c5b713ee3d2a5c0e652e7ca14eb4`, tree
   `7603fc6b8226fa8f9dcb3d5212136fb47631bc32`;
2. exact preflight job `1205227` passed `125/125` tests and the official
   training-batch smoke with `test_access=false`;
3. mechanism jobs `1205231/1205232`, parameter audits `1205272/1205310`,
   terminal scans `1205271/1205309` and cross-arm gate `1205337` all completed
   `0:0`;
4. formal receipt status is `FAIL_STRUCTURE_GATE`, selected variant is null,
   and development-pilot, official-comparison, locked-test and paper-claim
   releases are all false;
5. the next diagnostic must keep the checkpoint frozen and compare predicted
   owner tracks with post-forward oracle-birth and oracle-identity
   counterfactual tracks while recording first owner decision, state margins,
   lifetime and end/emission closure;
6. if oracle identity still yields no end, prospectively test exposure-normalized
   per-track end survival and true-owner/false-track family balancing; if it
   restores end, repair causal assignment and reacquisition instead;
7. any birth calibration term must be derived from the registered risk-set
   sampling probability, never chosen to reproduce these observed counts.

Official-comparability boundary:

- every D1.3/D1.4 score is mechanism or diagnostic evidence only;
- training-prefix detection values remain invalid because their path uses
  full-video timing and offline termination;
- no EventMATR performance claim is valid until a future structure passes, its
  contract is frozen, and a matched 100-epoch native-MATR/EventMATR study uses
  the same official split, extracted features, optimizer/scheduler,
  post-processing, terminal checkpoint policy and evaluator before one locked
  test.

Sources:

- [experiments/eventmatr-d11-structural-repair-design-20260729.md](experiments/eventmatr-d11-structural-repair-design-20260729.md);
- exact jobs `1205227`, `1205231`, `1205232`, `1205271`, `1205272`,
  `1205309`, `1205310` and `1205337`;
- structure-gate SHA-256
  `c5cab5a5fcb92de6fb1483b5c57981bbb0a8c04af2e33a7e1b2ec11afd7ace0f`.

Reversibility:

- A source-exact counterfactual diagnostic may change which minimal repair is
  preregistered. It cannot retroactively pass D1.4.
- Only a fresh model that passes every unchanged structure-liveness and ledger
  check can lift the analysis hold; no threshold adjustment or longer training
  may substitute for that pass.
