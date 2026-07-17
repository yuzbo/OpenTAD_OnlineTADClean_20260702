---
type: decision_register
updated: 2026-07-17
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

## DR-028: Revoke Profile Permission and Fix the Q2 Launch Identity Before GPU Work

Status: active procedural gate; supersedes every earlier `PROFILE=ALLOW` for
commit `6d88610da34a695e07d29c5e08b50fb57d2aa5e9`.

Decision:

> Block fixed-step profile and formal training until the launch ticket and the
> real Slurm argv derive `work_dir` from one immutable source and a real
> submit-shell integration test proves exact identity closure.

Reason:

- the ticket builder records exact `cfg_overrides` before submission;
- the submit helper subsequently creates a timestamped `RUN_DIR` and injects a
  new `work_dir` override;
- the pre-CUDA launch validator requires exact equality between ticket and live
  runtime identity;
- unit-level builder/validator tests and the 508-test B0 did not execute this
  cross-script value-generation order;
- a profile that fails the validator is useless, while bypassing the validator
  destroys the evidence chain.

Scientific qualifications accepted with this decision:

- Q2 is cached-feature truncated temporal training, not raw-video end-to-end;
- strict video-level causality requires immutable extractor provenance and a
  raw-future prefix-invariance test;
- chunk/source time and actual availability time must be separated;
- fixed versus rematch requires a complete paired supervision trace;
- formal reporting must explain and bind the 211-versus-213 population.

Explicit non-requirements for the current gate:

- multi-rank DDP consensus is not required because the registered route is
  locked to one process and one GPU;
- resume continuation is not required because formal resume is fail-closed;
- the full paper-stage baseline and ablation inventory is not required before a
  minimal mechanism kill test.

Required order:

1. repair the launch-workdir identity and add a deterministic fake-`sbatch`
   integration test;
2. make route status, feature provenance, packet clocks, one-factor trace, and
   reporting population explicit;
3. keep multi-rank and resume unsupported;
4. freeze a new clean commit and regenerate complete B0 evidence;
5. obtain a new independent `PASS / PROFILE=ALLOW`;
6. only then run the bounded fixed-step profile;
7. keep formal training blocked behind profile and scientific gates.

Sources:

- `PRO_FULL_PETAL_CODE_SCIENCE_REVIEW_20260715.md`;
- `PRO_FULL_PETAL_CODE_SCIENCE_REVIEW_ABSORPTION_20260715.md`;
- `FULL_PETAL_EXECUTION_GATES.md`;
- `FULL_PETAL_TRUST_MODEL.md`.

Reversibility:

- The implementation details are reversible after a new audited commit. The
  ban on profiling a ticket/runtime identity mismatch is not reversible.

## DR-029: Answer Round 1 Before CRS-EPS Implementation

Status: proposed author-response contract; no GPU or implementation permission.

Decision:

> Accept the finding that current Q2 is not CRS-EPS. Answer Q1-Q12 and obtain a
> Round-2 protocol decision before implementing event-centric training. In
> parallel, only the already-proven launch P0 may be repaired without GPU work.

Proposed author choices:

- preserve video-uniform per-video decision-time mean as the primary target
  risk;
- use CRS-EPS as candidate main training, a tiny preregistered full-stream
  state/loss/gradient audit, and complete chronological validation/test;
- replay gold state from video start and dynamically extend sampled context to
  every active instance's earliest observable birth;
- never initialize slots or hidden state from GT;
- replace cross-protocol optimizer-event throughput with multi-denominator
  physical and ESS accounting;
- cap all new pre-Stage-2 GPU work at 10 GPU-hours;
- use standard average temporal mAP at tIoU 0.3:0.7 as primary external
  quality, identity-linked errors as mechanism endpoints, and recall/FN plus
  completion delay as safety endpoints;
- stop Full PETAL and raw-video Stage 2 if fixed binding lacks a meaningful,
  identity-linked advantage over rematch;
- keep unavailable extractor provenance, 211/213 IDs, and second-dataset
  access explicitly unknown and fail closed at their appropriate gates.

Reason:

- current code scans all selected tokens and only reduces optimizer mutations;
- implementing a sampler before freezing its target risk and state replay would
  make sampled/full comparison uninterpretable;
- active-at-entry is a scientific issue because oracle slot initialization
  would invalidate the intended online state trajectory;
- optimizer events have incompatible meanings across complete-video and short
  episode protocols;
- Round 2 is the last design discussion before implementation and should
  resolve these contracts without changing the task.

Sources:

- `PRO_FULL_PETAL_CRS_EPS_Q2_ROUND1_REVIEW_20260715.md`;
- `PRO_FULL_PETAL_CRS_EPS_Q2_ROUND1_ABSORPTION_20260715.md`;
- `PRO_FULL_PETAL_CRS_EPS_Q2_ROUND1_AUTHOR_RESPONSE_DRAFT_20260715.md`.

Reversibility:

- The proposed choices may be revised by the author after Round 2 and before
  sampler implementation. Once an experiment manifest is frozen, changes
  require a new protocol version, B0, review, and authorization.

## DR-030: Adopt the Amended Round-2 Hybrid Protocol

Status: active implementation contract; supersedes DR-029's pending Round-2
decision while preserving DR-028's GPU block.

Decision:

> Implement and falsify a hybrid CRS-EPS protocol, but first close the launch
> identity P0. The first post-P0 implementation is CPU-only immutable episode
> manifests and exhaustive probability tests. Profile, effectiveness training,
> raw-video Stage 2, and Full PETAL claims remain blocked.

Frozen core:

- primary target risk remains video-uniform per-video decision-time mean;
- candidate training uses fixed `H=8`, frozen event/uniform mixture,
  repeated-exposure HH/IPW, exact support, and ESS;
- dynamic earliest-birth replay is a causal surrogate; video-start replay is
  the gold reference;
- sampled gradients preserve original global 64-token detach boundaries;
- Q2 separates local direct binding effects from longitudinal policy effects;
- sampled episodes independently reset/replay from the same parameter and
  mutable-buffer snapshot; only gradients accumulate within a video group;
- validation/test is complete chronological one-token streaming with immutable
  emissions and no offline NMS;
- cost uses physical token/frame/time/memory denominators and statistical ESS;
- fixed binding, CRS efficiency, raw-video causality, and package novelty are
  all unproven.

Independent amendments:

1. require matched stochasticity but choose token-addressed RNG only if a less
   invasive audited mechanism is insufficient;
2. freeze exact identity metric and matching definitions before results;
3. record model-caused slot exhaustion as scientific failure and route kill;
4. do not require positive LoRA-by-binding interaction when independent main
   effects and fixed-effect persistence are demonstrated;
5. label one-seed stopping as a resource kill rather than a population-level
   scientific conclusion;
6. report the full tIoU 0.3:0.7 vector plus its average;
7. audit FineAction/MultiTHUMOS identity suitability before generalization;
8. fail closed on cross-episode runtime-state or mutable-buffer leakage;
9. treat OAT as verified ECCV 2022 prior art and reject the source review's
   unresolved-artifact finding.

Required order:

1. repair `P0-LAUNCH-WORKDIR` and add deterministic fake-`sbatch` shell
   integration coverage;
2. implement and CPU-audit manifest construction, `q`, `rho`, union `pi`,
   weights, multiplicities, coverage, and ESS;
3. freeze exact identity metrics and the least invasive matched-stochasticity
   audit;
4. implement replay, denominator/lifecycle traces, and G0 gold controls;
5. implement the replacement profile contract;
6. freeze a clean commit, regenerate B0, and obtain independent PASS;
7. only then request the first bounded G0/profile GPU-hour.

Sources:

- `PRO_FULL_PETAL_CRS_EPS_Q2_ROUND2_REVIEW_20260715.md`;
- `PRO_FULL_PETAL_CRS_EPS_Q2_ROUND2_ABSORPTION_20260715.md`;
- `PRO_FULL_PETAL_CRS_EPS_Q2_ROUND1_AUTHOR_RESPONSE_DRAFT_20260715.md`;
- [ideas/crs-eps-training.md](ideas/crs-eps-training.md).

Reversibility:

- The protocol may be killed by CPU probability closure, G0 state/gradient
  fidelity, cost, or Q2 identity evidence. Changes after manifest freeze require
  a new version and review. The prohibition on treating a protocol GO as an
  effectiveness or novelty result is not reversible.

## DR-031: Freeze the CRS-EPS Implementation Before Any Experiment

Status: implementation-complete; B0 and independent acceptance pending.

Decision:

> Treat the current CRS-EPS code as a candidate falsification instrument, not
> as experimental evidence. Freeze it in one clean commit, regenerate the
> exhaustive source-hash-locked B0, and obtain PASS from reviewer
> `019f5abd-5104-79b3-882e-354ca796f2c1` before running real-data G0 or a
> fixed-step profile.

Reasons:

- P0 launch identity now derives run/work/seed identifiers solely from the
  immutable ticket;
- the estimator, state lifecycle, optimizer event, metric, and cost contracts
  now have executable fail-closed tests;
- G0 has explicit video-start, fixed-192, dynamic-birth, and reset arms, but no
  real checkpoint/data evidence yet;
- `M=4` is deliberately provisional and cannot be promoted by implementation
  convenience;
- allowing an experiment before B0/review would break the user's frozen gate
  and contaminate outcome-blind protocol choices.

Rejected alternatives:

- launch a quick profile because the CPU tests pass;
- call cached-feature CRS-EPS end-to-end raw-video training;
- select G0 margins or `M` from Q2 effectiveness outcomes;
- treat code completeness as evidence for efficiency, fidelity, or novelty.

Sources:

- `PRO_FULL_PETAL_CRS_EPS_Q2_ROUND2_ABSORPTION_20260715.md`;
- `FULL_PETAL_EXECUTION_GATES.md`;
- [ideas/crs-eps-training.md](ideas/crs-eps-training.md);
- [discussion_timeline.md#T24-round-2-protocol-reaches-a-cpu-auditable-implementation](discussion_timeline.md).

Reversibility:

- Code may change after a failed B0 or review, but every change requires a new
  commit, regenerated B0, and another review. The no-experiment-before-PASS
  ordering is not reversible for this evidence chain.

## DR-032: Make G0 and Exact-M Lifecycle Launch-Enforced

Status: active correction; new B0 and same-reviewer acceptance pending.

Decision:

> Accept all four blocking findings on commit `0731f07`. CRS-EPS profile
> authorization now requires a signed, reproducible G0 PASS chain, and each
> optimizer event must prove exact manifest draw membership/order. Any failed
> draw restores the group-start buffer and staged online state.

Reasons:

- documentation-only G0 ordering can be bypassed by ticket construction;
- one update per video is not scientifically meaningful if the claimed `M`
  draws can contain duplicates or omissions;
- a raised exception is not fail-closed when model buffers retain mutations;
- self-consistent false provenance and post-result margins can manufacture an
  apparently valid G0 result unless source bytes and preregistration signatures
  are independently checked.

Rejected alternatives:

- rely on the operator to run G0 before profile;
- accept only draw count `M` without ordered manifest membership;
- clear gradients/state but leave buffers for the next group;
- treat a textual `PREREGISTERED` status as proof of outcome blindness;
- reuse the prior B0 after changing the correction code.

Source:

- `PRO_FULL_PETAL_CRS_EPS_IMPLEMENTATION_REVIEW_20260716.md`;
- [discussion_timeline.md#T25-first-exact-commit-review-blocks-four-enforceability-gaps](discussion_timeline.md).

Reversibility:

- Schema details may receive a new version, but no weaker chain may authorize a
  CRS-EPS profile. Every correction requires a new clean commit, B0, and review.

## DR-033: Bind Training and B0 to the Exact Manifest, Bytes, and Target OS

Status: correction implemented locally; clean-commit Linux leaf, B0 root, and
same-reviewer acceptance pending.

Decision:

> Accept all six blocking findings on commit `cd601ce`. A CRS optimizer event
> must consume the exact ordered payloads in a trusted epoch manifest; every
> group-interior loader/control failure must roll back; provenance must hash the
> same bytes consumed; Slurm must submit the same bytes published exclusively;
> and B0 PASS must include a signed target-Linux leaf for the same commit and
> manifest.

Reasons:

- internal hashes prove only self-consistency, not ownership by the published
  sampling population;
- rollback that starts after parsing is not a transaction around data intake;
- chronological packet identity is false for an M-draw CRS-EPS runtime;
- path re-opening permits a verified file to be replaced before consumption;
- a writable path passed to `sbatch` is not an immutable submitted script;
- an unsigned remote test claim cannot authorize a target-specific launch.

Rejected alternatives:

- trust recomputed episode hashes without comparing the manifest payload;
- catch only model-forward and optimizer failures;
- keep chronological packet identity as a convenient proxy;
- hash once at dataset construction and reopen `.npy` or `.pth` later;
- retain a caller-selected `SBATCH_BIN` or check-then-write shell script;
- cite an N16R4 terminal transcript outside the signed B0 chain.

Source:

- same-reviewer exact-commit audit of `cd601ce95fdd16ecfdd17f9a5d93b33578691133`;
- [discussion_timeline.md#T26-second-exact-commit-review-finds-six-remaining-evidence-gaps](discussion_timeline.md).

Reversibility:

- Implementation details can change only through a new commit and evidence
  cycle. Manifest ownership, same-byte consumption, exact submitted bytes, and
  a signed target-OS leaf are irreversible requirements for this route.

## DR-034: Require Persisted Manifest Reproduction Before G0

Status: correction implemented; replacement B0 and same-reviewer acceptance
pending.

Decision:

> Invalidate `ee439e1` as a launch-authorizing commit after its first real G0
> preregistration exposed a JSON round-trip mismatch. Emit JSON-native instance
> timelines, test serialize/read/validate reproduction explicitly, and restart
> the exact-commit B0 and review chain before any G0 outcome or GPU profile.

Reasons:

- an in-memory self-check does not prove that the evidence file consumed by a
  later process is reproducible;
- tuple-to-list conversion changed the loaded manifest structure even though
  the sampling semantics were unchanged;
- preregistration failed before model execution, so correcting the defect does
  not condition the sample set or margins on G0 outcomes;
- reusing the old PASS after a source change would break the signed
  commit-to-evidence chain.

Rejected alternatives:

- normalize the already-published manifest at preregistration time;
- weaken exact equality to ignore container types;
- edit the external G0 artifact in place;
- reuse the `ee439e1` B0 or reviewer verdict for the corrected commit.

Source:

- fail-closed real-data G0 preregistration on 2026-07-16;
- `opentad/utils/crs_eps_sampling.py`;
- `tests/test_crs_eps_sampling.py`;
- [discussion_timeline.md#T27-first-real-g0-attempt-exposes-a-json-round-trip-blocker](discussion_timeline.md).

Reversibility:

- The concrete list representation may be schema-versioned in the future, but
  persisted-byte reproduction and a fresh evidence chain after source changes
  are irreversible requirements.

## DR-035: Normalize G0 Singleton Index Before Payload Attestation

Status: correction implemented; replacement B0 and same-reviewer acceptance
pending.

Decision:

> A selected manifest draw may retain its original index in the outer G0
> selection record, but the model-facing singleton audit control must use
> runtime draw index zero and must compute its payload and sequence hashes only
> after that normalization. Any source correction restarts the evidence chain.

Reasons:

- `video_group_size=1` requires `draw_index=0` at model entry;
- hashing the original index and later overwriting it makes the signed runtime
  identity internally inconsistent;
- the outer audit row still records the selected manifest index, so selection
  provenance is not lost by normalizing the singleton control;
- failure occurred before a valid audit row or terminal result, preserving
  outcome-blind selection and margins.

Rejected alternatives:

- exempt G0 controls from payload-hash verification;
- allow singleton groups to carry an out-of-range original draw index;
- rewrite the already-signed selection or margins;
- replace the stress sample with a draw-zero case;
- reuse the `cde6196` B0/review after changing source code.

Source:

- fail-closed G0 execution on 2026-07-16;
- `opentad/datasets/crs_eps_feature.py`;
- `tests/test_crs_eps_training_contracts.py`;
- [discussion_timeline.md#T28-nonzero-g0-draw-exposes-singleton-runtime-index-drift](discussion_timeline.md).

Reversibility:

- A future schema may separate manifest and runtime draw indices explicitly,
  but the attested model-facing control must always hash the exact values it
  consumes.

## DR-036: Make G0 Similarity Sentinel-Aware Without Changing Its Margins

Status: correction implemented; replacement B0 and same-reviewer acceptance
pending.

Decision:

> Preserve the model's NaN sentinel semantics in a discrete start-state mask,
> compare a zero-canonicalized start-state vector continuously, reject
> non-finite values in all other runtime tensors, and compute vector metrics in
> float64. Do not change G0 samples, thresholds, checkpoint seed, or method
> after observing the failed diagnostic.

Reasons:

- free slots intentionally carry NaN start frames, so raw vector norms are not
  defined even when two runtime states are semantically identical;
- sentinel location is discrete state and must match exactly rather than being
  silently discarded;
- float32 norm/dot arithmetic produced a cosine above one for identical finite
  gradients;
- the gate must measure replay fidelity, not representation artifacts of an
  internal empty-slot sentinel.

Rejected alternatives:

- lower the preregistered runtime or gradient thresholds;
- drop the difficult preselected videos;
- replace the deterministic checkpoint after seeing diagnostics;
- convert every NaN/Inf in every runtime tensor to zero;
- cite unsigned diagnostic numbers as a G0 PASS or KILL;
- continue to profile while G0 lacks a signed terminal artifact.

Source:

- fail-closed G0 execution and read-only numeric diagnosis on 2026-07-16;
- `opentad/utils/crs_eps_audit.py`;
- `tests/test_crs_eps_paired_audit.py`;
- [discussion_timeline.md#T29-g0-reveals-sentinel-unsafe-runtime-similarity-measurement](discussion_timeline.md).

Reversibility:

- Alternative formally specified runtime metrics would require a new schema
  and preregistration. The current selection and margins remain frozen for
  this route's next terminal G0 run.

## DR-037: Prove Sentinel Lifecycle Legality Before Canonicalization

Status: correction implemented; replacement B0 and same-reviewer acceptance
pending.

Decision:

> Canonicalize a start-state NaN only after proving that its slot is not
> ACTIVE. Require the exact invariant
> `isnan(start_state) == (slot_status != ACTIVE)`, legal integer lifecycle
> values, and shape alignment. Any violation aborts the paired audit before a
> trace or G0 gate result can be produced.

Reasons:

- the previous discrete NaN mask preserved location but did not prove that the
  location was legal under the model lifecycle;
- two identically corrupt arms could otherwise receive cosine one and exact
  discrete equality;
- ACTIVE slots acquire finite starts at birth, while FREE and REFRACTORY slots
  intentionally carry NaN;
- G0 must fail closed on invalid runtime state rather than measure similarity
  between invalid states.

Rejected alternatives:

- accept any matching NaN masks as sufficient fidelity evidence;
- silently repair ACTIVE NaNs or inactive finite starts inside the audit;
- weaken G0 margins or remove stress samples;
- reuse the `5d27fca` B0/review chain after changing source code;
- proceed to G0 on the basis that all 580 prior tests passed.

Source:

- same-reviewer exact-commit audit of `5d27fcad36062a6496f8c330ed15fba623a9667f`;
- `opentad/utils/crs_eps_audit.py`;
- `tests/test_crs_eps_paired_audit.py`;
- [discussion_timeline.md#T30-same-reviewer-audit-rejects-unconstrained-sentinel-canonicalization](discussion_timeline.md).

Reversibility:

- Alternative lifecycle encodings require a schema change, but every accepted
  runtime metric must continue to prove that its sentinel representation is
  valid before canonicalization.

## DR-038: Preregister Exact G0 Checkpoint Bytes Before Any Replay Outcome

Status: correction implemented; replacement B0 and same-reviewer acceptance
pending.

Decision:

> G0 selection v3 must sign a contained checkpoint path, exact SHA-256, byte
> size, state key, and deterministic generation identity. Preregistration must
> reproduce the model state from the exact config and manifest seed. The audit
> runner must verify those signed bytes before model construction, and launch
> validation must recursively rehash them before accepting G0 evidence.

Reasons:

- the prior terminal audit accurately recorded the checkpoint it consumed but
  did not prove that the choice preceded all model diagnostics;
- signed samples and margins could be reused with another compatible
  checkpoint, leaving a post-diagnostic degree of freedom;
- deterministic state reproduction binds the checkpoint to the model/config
  rather than accepting an arbitrary loadable state;
- a bundle-relative raw-byte reference lets launch validation detect later
  replacement without trusting an absolute path.

Rejected alternatives:

- document the old checkpoint SHA only in an unsigned policy note;
- trust the checkpoint field written after traces complete;
- bind only a seed without reproducing state tensors;
- allow launch validation to trust a signed digest without reopening bytes;
- reuse any v2 selection, margins, audit, B0, or review artifact.

Source:

- same-reviewer exact-commit audit of `29bc0aee90faef61f16e32130b8c1368fc78e755`;
- `tools/preregister_crs_eps_gold_audit.py`;
- `tools/run_crs_eps_gold_audit.py`;
- `opentad/utils/crs_eps_gold_gate.py`;
- `opentad/utils/full_petal_launch.py`;
- [discussion_timeline.md#T31-same-reviewer-finds-post-diagnostic-checkpoint-choice-freedom](discussion_timeline.md).

Reversibility:

- Checkpoint packaging may change under a future schema, but outcome-blind
  byte identity, deterministic generation identity, and recursive launch
  verification are permanent requirements for this G0 route.

## DR-039: Honor the Signed CRS-EPS G0 KILL and Reopen Training-Route Selection

Status: active; profile and formal training blocked.

Decision:

> Accept the terminal `KILL` at implementation commit
> `70df86ea3d38d70c658ae0ee9e04245d57b834d4`. The current CRS-EPS
> `dynamic_birth` replay is not an acceptable primary training surrogate under
> its preregistered fidelity contract. Freeze the negative result, perform
> zero-GPU causal diagnosis, and obtain a Pro mechanism/route adjudication
> before implementing a replacement.

Reasons:

- local and target-Linux B0 passed `588/588`, and the same reviewer returned
  exact `PASS / PROFILE=ALLOW / NEXT_GATE=G0` before the outcome;
- the exact samples, margins, checkpoint bytes, code, config, and manifest were
  signed or hash-bound before terminal execution;
- three of four selected cases violated absolute gradient, loss, and/or
  runtime-state fidelity, producing fourteen violations;
- the launch validator independently rejects the signed `KILL` before
  profile authorization;
- changing margins, checkpoint, samples, or method after seeing the outcome
  would convert a confirmatory gate into post-hoc tuning.

Rejected alternatives:

- launch profile because dynamic replay beats reset or fixed replay on average;
- weaken the `0.90` fidelity margins or relax discrete-state equality;
- rerun the same exposed samples as fresh confirmatory evidence;
- call B0 test success scientific validation of CRS-EPS;
- abandon the On-TAD task or claim that the entire Full PETAL infrastructure
  was falsified by a training-surrogate gate.

Source:

- [experiments/crs-eps-g0-kill-20260716.md](experiments/crs-eps-g0-kill-20260716.md);
- signed external `audit.json`, SHA-256
  `a08183a8dba4ec5267b5a75eed3501882180a22f28d6b76ce2c7b4e65511009b`;
- `opentad/utils/crs_eps_gold_gate.py`;
- `opentad/utils/full_petal_launch.py`;
- [discussion_timeline.md#T32-signed-g0-kills-the-current-crs-eps-training-surrogate](discussion_timeline.md).

Reversibility:

- A genuinely new training surrogate may be evaluated, but it must use a new
  scientific design, clean commit, outcome-blind holdout G0, and full evidence
  chain. The historical `KILL` remains immutable.

## DR-040: Accept the Current Structural Kill Without Promoting Q2 Prematurely

Status: active historical constraint; route adjudicated by DR-041; zero new GPU hours.

Decision:

> Accept `STRUCTURAL-KILL-CURRENT-EMPTY-STATE-DYNAMIC-BIRTH-CRS-EPS`, not a
> blanket kill of all event-centric training. Keep chronological cached Q2 as
> a gold/reference candidate rather than an accepted primary route. Retain the
> Full PETAL lifecycle and evidence infrastructure as engineering substrate.
> Resolve state-faithful cost control and Q2 slot-capacity readiness before any
> new profile or training.

Reasons:

- local inspection of the signed raw bundle reproduces the terminal `KILL`
  and shows two failed dynamic arms exactly equal the fixed-192 replay range;
- the only selected dynamic-extension arm adds eight bins but remains far
  below the absolute fidelity margins;
- omitted chronological history changes learned runtime state even when
  annotation-side lifecycle traces agree, and changes discrete lifecycle,
  endpoint ownership, and birth assignment in the longest stress case;
- HH/IPW corrects sampling exposure, not hidden recurrent state or omitted
  historical Jacobian paths;
- no chronological Q2 profile, effectiveness result, or checkpoint trajectory
  exists;
- full-prefix gold controls record slot exhaustion in two selected cases,
  which creates a separate capacity/lifecycle blocker under the project's
  fail-closed training law.

Rejected alternatives:

- relabel the valid G0 `KILL` as a tunable margin failure;
- infer that every state-faithful event-centric or selective-backward method
  has already been disproved;
- call chronological Q2 the primary route before capacity, cost, and
  effectiveness gates pass;
- spend GPU hours before the same reviewer adjudicates the replacement
  estimand and minimum unseen holdout protocol;
- treat retained infrastructure as a paper contribution.

Source:

- `PRO_CRS_EPS_G0_KILL_ROUTE_REVIEW_ROUND1_20260716.md`, SHA-256
  `D0AB7B51961E61F9DBD45192743AC61B5522646F2FA1027B96489C8010ADC46F`;
- `PRO_CRS_EPS_G0_KILL_ROUTE_REVIEW_ROUND1_ABSORPTION_20260716.md`;
- signed external G0 bundle and terminal `audit.json`, SHA-256
  `a08183a8dba4ec5267b5a75eed3501882180a22f28d6b76ce2c7b4e65511009b`;
- [experiments/crs-eps-g0-kill-20260716.md](experiments/crs-eps-g0-kill-20260716.md);
- [discussion_timeline.md#t33-pro-round-1-confirms-the-current-structural-kill-but-overstates-q2-readiness](discussion_timeline.md).

Reversibility:

- Q2 may become primary only after an explicit capacity/lifecycle audit,
  matched cost profile, and scientific effectiveness gate. A successor
  state-faithful sampler may be evaluated only as a new method with a clean,
  outcome-blind evidence chain. The historical G0 result is irreversible.

## DR-041: Select CSFSB Conditionally Behind a Capacity-First Gate

Status: active; capacity audit authorized; R1 conditional; zero GPU hours.

Decision:

> Retain exactly one successor, R1 Chronological State-Faithful Selective
> Backward (CSFSB). Preserve complete chronological numerical state,
> lifecycle, supervision, and global 64-bin TBPTT boundaries; sample only
> HH-weighted loss and backward contributions. Before R1 implementation,
> perform a zero-GPU full-fit-core capacity/lifecycle audit and freeze exactly
> one shared fixed/rematch contract. Profile and formal training remain
> blocked until a new unseen CPU G0 passes.

Reasons:

- R1 removes the omitted-history defect that killed empty-state CRS-EPS;
- the current Q2 scalar objective is a bin-uniform mean, so the proposed
  `a_vt=(m_vt/M)*((1/T_v)/rho_vt)` estimator targets the actual chrono-64
  objective when implemented without realized-sample renormalization;
- R1 keeps every forward token and lifecycle transition, making state
  faithfulness directly falsifiable with an all-graph paired execution;
- exact recomputation adds forward work for the same backward graph extent,
  while stale runtime checkpoints lose historical Jacobian paths;
- two complete chronological controls exhausted slots, so capacity readiness
  must be resolved before training-route code can be trusted;
- all required pre-profile evidence can be generated with zero GPU hours.

Binding qualifications:

- `IMPLEMENTATION=ALLOW` means capacity audit now and R1 only after a shared
  capacity contract; it is not profile or training permission;
- the proposed separate R1 engine filename is non-binding; reuse the existing
  transactional engine unless a separate engine demonstrably avoids drift;
- same-logits lifecycle replay is valid only while the head's predictions do
  not depend on lifecycle tensors;
- freeze CPU platform, near-zero relative-error behavior, and a hard CPU
  micro-trajectory budget before outcomes are visible;
- R1 remains enabling infrastructure with strong prior-art reconstruction
  risk until matched wall/GPU-hour and mechanism evidence exists.

Rejected alternatives:

- revive `dynamic_birth`, `fixed_192`, reset, or empty-state episode replay;
- promote chronological Q2 before zero-exhaustion capacity and lifecycle
  readiness is demonstrated;
- implement R1 and silently change K, thresholds, refractory, binding, loss,
  optimizer, or sampling in the same scientific commit;
- use the old four G0 samples as confirmatory evidence;
- claim efficiency from fewer backward tokens without lower paired wall time
  and GPU-hours;
- spend GPU hours before the new family-wise G0 passes.

Source:

- `PRO_CRS_EPS_G0_KILL_ROUTE_REVIEW_ROUND2_20260716.md`, SHA-256
  `2BE3C21E951F68F397CADA468B66820BA08B3FF2F0FFD0A4E59580D784C989EE`;
- `PRO_CRS_EPS_G0_KILL_ROUTE_REVIEW_ROUND2_ABSORPTION_20260716.md`;
- [ideas/chronological-state-faithful-selective-backward.md](ideas/chronological-state-faithful-selective-backward.md);
- [discussion_timeline.md#t34-round-2-selects-csfsb-but-authorizes-capacity-audit-first](discussion_timeline.md).

Reversibility:

- R1 is killed if the capacity audit cannot freeze a legal shared contract,
  exact G0-A fails, finite-M G0-B fails, any new G0 case exhausts slots, or
  matched profile does not reduce wall time/GPU-hours. The historical CRS-EPS
  KILL is not reversible.

## DR-042: Invalidate the First Windows Capacity Run Before Outcome Publication

Status: active execution amendment; rerun required; zero GPU hours.

Decision:

> The `q2_capacity_3e20a01_20260716` partial directory is invalid evidence.
> Correct only the externally owned gzip raw-handle lifecycle, preserve cleanup
> diagnostics, add an atomic-publication regression, and rerun the complete
> frozen audit from a new clean commit and a new non-overwritten output root.

Reasons:

- all 480 replay units completed, but no terminal gate line or atomically
  published evidence directory exists;
- Windows correctly prevented directory publication while the raw trace file
  remained open;
- cleanup removed the summary before failing on the locked trace, so a valid
  outcome cannot be recovered from the partial directory;
- the failure occurs after computation but before outcome publication and does
  not justify changing any scientific policy or resource constant.

Rejected alternatives:

- infer the in-memory gate from partial files or console progress;
- reconstruct a new summary from the incomplete retained trace;
- reuse or overwrite the failed run root;
- change thresholds, K, lifecycle ordering, seeds, data, or the policy grid;
- proceed to R1 because all replay units happened to execute.

Source:

- [experiments/q2-capacity-lifecycle-audit-20260716.md](experiments/q2-capacity-lifecycle-audit-20260716.md);
- [discussion_timeline.md#t35-full-capacity-replay-exposes-a-pre-outcome-windows-handle-defect](discussion_timeline.md);
- `tools/audit_q2_capacity_lifecycle.py`;
- external invalid partial root
  `q2_capacity_3e20a01_20260716/.evidence.partial-*`.

Reversibility:

- the portability fix is reversible in code, but this failed run can never be
  promoted to evidence. Only a fresh, complete, atomically published rerun can
  change the capacity gate.

## DR-043: Accept REVISE_REQUIRED Without Promoting the -2 Birth Prior

Status: superseded by terminal DR-044; R1 killed.

Decision:

> Accept the clean `1441219` capacity audit as terminal `REVISE_REQUIRED` for
> the current Q2 contract. Do not automatically adopt the sole legal
> zero-exhaustion candidate, additive birth-logit bias `-2`, because its ten
> emissions may represent degenerate birth suppression. Require one independent
> max adjudication before any one-factor lifecycle revision. R1, GPU profile,
> and formal training remain blocked.

Reasons:

- summary, trace, commitment, content hashes, checkpoint/config/data identity,
  GT-taint audit, cause closure, and fixed/rematch equality all verify;
- current Q2 exhausts 2,206 births across all three seeds, so it is not a valid
  shared training contract;
- annotation-side minimum oracle K is two and true canonical capacity causes
  zero exhaustions, so increasing K would not address the measured mechanism;
- refractory-zero and release-order corrections remain nonzero, including 228
  exhaustions when combined;
- additive `-2` is the only eligible zero-exhaustion arm, but its emission
  count collapses from 5,517 to 10, which the capacity-only audit cannot
  distinguish from a scientifically useful low-prior initialization.

Rejected alternatives:

- call `REVISE_REQUIRED` a PASS because one legal counterfactual reaches zero;
- implement R1 before a shared contract is frozen;
- increase K despite zero true-capacity attribution;
- adopt the privileged canonical-only availability policy;
- use effectiveness data post hoc to choose a prior, threshold, or lifecycle;
- spend GPU hours to resolve a CPU-identifiable contract defect.

Source:

- external clean evidence root
  `q2_capacity_1441219_20260716_rerun/evidence`;
- summary SHA-256
  `dad8174f149240d8bba52eff9121b80588a90a4594061f7563fca476af4f30f9`;
- trace SHA-256
  `b5ca44d240d22a43365ecaa6e6fe07a5cd564d079b351b8807b970176a46a0ed`;
- commitment SHA-256
  `6cf2c0dc66ac6cc2beab9cb2e0eb64f8645914f28b218a47eb6717b93f83abb1`;
- [experiments/q2-capacity-lifecycle-audit-20260716.md](experiments/q2-capacity-lifecycle-audit-20260716.md);
- [discussion_timeline.md#t36-clean-capacity-audit-rejects-current-q2-and-exposes-a-degenerate-escape](discussion_timeline.md).

Reversibility:

- a new clean one-factor revision may be tested only after independent review
  freezes its exact nondegeneracy and rerun contract. The current Q2 failure is
  permanent evidence and cannot be relabeled.

## DR-044: Kill Q2 and R1 After Independent Degeneracy Adjudication

Status: terminal `KILL`; GPU profile and formal training blocked.

Decision:

> Accept the unique independent max review choice `C) KILL_Q2_R1`. The
> additive `-2` birth prior is a degenerate birth-suppression escape, not a
> repaired lifecycle contract. Make no model/config revision, run no further
> `-2` confirmation gate, do not implement R1/CSFSB, and authorize zero GPU
> hours. Preserve the clean capacity audit as terminal `REVISE_REQUIRED`
> evidence rather than relabelling it.

Reasons:

- the reviewer independently verified the complete evidence chain and exactly
  reproduced actual and sparse counterfactual exhaustion aggregates from
  524,258 trace rows;
- emissions and exhaustion move together under stronger birth suppression:
  actual is `5517/2206`, `-1` is `1722/80`, threshold `0.75` is `1270/33`,
  and `-2` is `10/0`;
- seeds 705 and 706 emit nothing under `-2`, while annotation oracle K is two
  and no exhaustion is true canonical capacity;
- the gate is therefore non-identifying: a controller that never starts an
  event can pass capacity while failing the task;
- training can raise birth logits out of the silent initialization and restore
  the already measured exhaustion, so `-2` is not a stable shared contract;
- adding a post-outcome emission floor or another lifecycle factor now would
  violate the frozen one-factor and capacity-only protocol.

Rejected alternatives:

- promote `birth_prior_bias_m2` because its positive BCE gradient is nonzero;
- add a post hoc nondegeneracy threshold after observing ten emissions;
- rerun the already identified suppression mechanism as confirmation;
- implement R1 behind an unresolved runtime/canonical availability mismatch;
- start a GPU profile or formal training despite the explicit zero-hour gate.

Source:

- [PRO_Q2_CAPACITY_INDEPENDENT_MAX_REVIEW_20260716.md](../PRO_Q2_CAPACITY_INDEPENDENT_MAX_REVIEW_20260716.md);
- reviewer `019f6b39-277a-7972-8ecb-beeb8a738605`;
- [experiments/q2-capacity-lifecycle-audit-20260716.md](experiments/q2-capacity-lifecycle-audit-20260716.md);
- [discussion_timeline.md#t37-independent-review-kills-q2-and-r1](discussion_timeline.md).

Reversibility:

- the Q2/R1 route is terminal under the frozen protocol. A future route would
  require a new task/mechanism contract and new preregistration; it may cite
  this evidence but cannot reopen Q2 by changing one threshold or prior.

## DR-045: Authorize R-A Preregistration and CPU-Only P0

Status: superseded on 2026-07-17 by DR-046 before preregistration or implementation; GPU authorization remained zero.

Decision:

> Accept `GO_NEW_ROUTE_P0_ONLY` for R-A, Prefix-Shared Latent Event Filter
> with Immutable Ledger. Preserve Q2/R1 as terminal KILL. Authorize a new,
> CPU-only synthetic P0 after a clean preregistration freezes every executable
> model, UOT, loss, generator, optimizer, runtime, comparison, and evidence
> constant. Do not authorize real-data G0, profile, effectiveness/formal
> training, visual fine-tuning, raw-video training, or any GPU hour.

Reasons:

- the new route directly targets the demonstrated Q2 contract defect:
  train/inference share one model-generated prefix state while GT assignment
  is ephemeral and loss-only;
- runtime availability no longer gates target eligibility, so a predicted
  false ACTIVE state cannot permanently erase first-birth supervision;
- release-before-reseed, first-event targets, immutable model-only ledger, and
  anti-silence controls form a task-specific P0 hypothesis rather than a Q2
  threshold/K/refractory patch;
- eight deterministic synthetic families, future/GT taint tests,
  train/inference equality, learned anti-silence, and optimizer/gradient gates
  can falsify the mechanism at zero GPU cost;
- the review retains temporal MOTR/TrackFormer as a severe reconstruction
  threat, so publication novelty and effectiveness remain unapproved.

Execution amendment:

- `runtime_supervision_contract_pass` is design-level only; implementation is
  `SPECIFIED_NOT_IMPLEMENTED`;
- the review does not freeze all UOT, ledger, generator, training, numerical,
  and resource constants, despite freezing families, seeds, margins, and the
  terminal rule;
- therefore implementation and P0 execution begin only after the
  [P0 closure checklist](experiments/prefix-shared-event-p0-20260717.md) is
  converted into an exact, hashed preregistration.

Rejected alternatives:

- reopen Q2/R1 or adopt `birth_prior_bias_m2`;
- treat a coherent method sketch as P0 PASS;
- start coding while preserving freedom over optimizer steps, synthetic
  distributions, UOT parameters, ledger thresholds, or equality tolerances;
- claim raw-video end-to-end novelty from a cached-feature/head-level P0;
- skip temporal MOTR reconstruction because persistent queries are internal;
- allow a P0 PASS to authorize GPU work automatically.

Source:

- private attachment
  `c9567761-7186-4f18-986d-f450881b8052/pasted-text.txt`, archived as
  `PRO_POST_Q2_R1_KILL_ONTAD_ROUTE_REVIEW_20260717.md`, SHA-256
  `6E7935741C497D90000CED25AEB6D4A078919882065D8E082EDABBE7CB148C7F`;
- [independent absorption](../PRO_POST_Q2_R1_KILL_ONTAD_ROUTE_REVIEW_ABSORPTION_20260717.md);
- [idea:prefix-shared-latent-event-filter](ideas/prefix-shared-latent-event-filter.md);
- [T38](discussion_timeline.md#t38-post-kill-review-selects-a-new-model-only-prefix-filter-for-p0).

Reversibility:

- R-A may be killed by any frozen P0 gate. A P0 PASS still requires independent
  read-only review and can authorize at most an unseen CPU G0. Q2/R1 remains
  irreversible under its prior contract.

## DR-046: Revoke R-A Default-Route Authorization Before Implementation

Status: active route constraint; evidence-collection execution refined by DR-047; model P0 and GPU authorization zero.

Decision:

> Accept `REVISE_ROUTE_AND_REVIEW_AGAIN`. DR-045 no longer authorizes an R-A
> preregistration, implementation, or synthetic model P0. Demote R-A to B4 in
> a route-level B0-B4 comparison. Before another route review, bind an
> outcome-blind annotation census, a cached-feature strict-causality
> certificate, matched simple baselines, structural OOD and negative controls,
> and an exact-delta/equivalence-KILL contract against temporal MOTR.

Reasons:

- Q2 evidence proves a local train/runtime lifecycle defect, but does not show
  that continuous carriers, UOT, adjacent-prefix transport consistency, or a
  model-only latch are necessary;
- the formal evaluator observes committed intervals, classes, scores, and
  times, not internal carrier identity, so any identity claim must produce
  observable duplicate, fragmentation, overlap-recall, latency, or recall
  gains;
- TrackFormer and MOTR already establish persistent track queries and
  birth/survival assignment, while TadTR establishes action-query interval
  prediction; a one-dimensional reconstruction is therefore a mandatory
  obviousness attack;
- the proposed P0 can test causal execution and anti-silence, but cannot
  distinguish R-A from template timing, event counting, threshold selection,
  ordinary prefix sets, persistent queries, or ledger-only deduplication;
- the exact project split has not yet established the prevalence of
  same-class repetition/overlap, same-bin transitions, short actions, or
  concurrency needed to support the proposed mechanism;
- source-frame ordering does not prove that cached encoder tokens exclude
  future frames.

Normalization amendment:

- the review's machine-readable `false` values for field-gap status,
  simple-baseline insufficiency, and non-locality are recorded as
  `NOT_ESTABLISHED`, not as proof that the corresponding propositions are
  permanently false;
- internal identity is not dismissed a priori, but it can only survive as a
  mechanism claim if it changes preregistered observable outputs;
- R-A remains a reversible B4 candidate, not a default route and not a
  permanently killed family.

Required route-level sequence:

1. freeze and run only an outcome-blind, read-only annotation census;
2. bind encoder code, weights, clip construction, temporal sampling, and exact
   receptive field in a strict-causality certificate;
3. rewrite the question as whether persistent carriers provide
   non-reconstructible online instance-consistency gains over a no-identity
   prefix set and temporal MOTR;
4. freeze B0-B4 shared inputs, heads, ledger, evaluator, anti-silence,
   parameter matching, and update-budget rules;
5. freeze structural OOD, negative controls, mechanism deletions, and the
   temporal-MOTR equivalence region;
6. conduct another independent route review before any model contract or code.

Rejected alternatives:

- continue DR-045 merely because it was previously authorized;
- interpret Q2's local failure as a field-wide proof;
- freeze R-A dimensions, UOT, losses, thresholds, generator, optimizer, or
  evidence schema before route identifiability is established;
- treat seed-held-out synthetic accuracy as structural generalization;
- implement only R-A and compare it after outcome observation;
- use internal carrier identity as a result without output-level consequences;
- perform cached-feature effectiveness, profile, formal, visual, or raw-video
  training.

Source:

- private attachment
  `0b362eae-64cd-4f53-830e-eb4f7524e31a/pasted-text.txt`, archived as
  `PRO_PREFIX_SHARED_ROUTE_PREIMPLEMENTATION_REVIEW_20260717.md`, SHA-256
  `B5BAC9E38C40883687193E37F5364A6A165C67EEDF0F496607A59B46FD88A3D4`;
- [independent absorption](../PRO_PREFIX_SHARED_ROUTE_PREIMPLEMENTATION_REVIEW_ABSORPTION_20260717.md);
- [route-identifiability gate](experiments/prefix-route-identifiability-gate-20260717.md);
- [T39](discussion_timeline.md#t39-preimplementation-review-revokes-r-a-default-route-status).

Reversibility:

- the route revision is reversible only after the complete route-evidence
  package receives an independent PASS. Such a PASS may authorize a model P0
  contract, but cannot directly authorize real-data effectiveness or GPU work.
  Q2/R1 remains terminal under its frozen protocol.

## DR-047: Revise and Independently Review the Protocol Before R0 or R1

Status: active; protocol-only revision allowed; evidence collection and model work blocked.

Decision:

> Accept `REVISE_PROTOCOL_BEFORE_COLLECTION`. The current route gate is a
> high-quality checklist, not an executable preregistration. Do not run a new
> annotation census or cache-causality audit until a protocol-only immutable
> commit closes every population, definition, statistic, disclosure,
> fairness, control, OOD, exact-delta, equivalence, and terminal-action choice,
> and a new independent reviewer returns
> `PASS_PROTOCOL_TO_OUTCOME_BLIND_EVIDENCE_COLLECTION`.

Reasons:

- the repository simultaneously records historical reporting count 211 and
  canonical expected count 213, so R0 has no unique reporting population;
- overlap, concurrency, and duration aggregates have already been exposed, so
  the census can be model-outcome-blind but not annotation-unseen;
- the existing R0 gate does not define interval/bin conventions, clustered
  uncertainty, claim eligibility, disclosure, or failure actions precisely
  enough to prevent post-observation route selection;
- the SigLIP cache source path is plausibly single-frame causal, but the
  manifest does not bind an exact Hugging Face revision, processor/config,
  weight shards, raw-video bytes, extraction commit/environment, or token
  support map;
- B0-B4 semantics, parameter/token/update/tuning fairness, negative-control
  input rights, structural OOD grammar, and temporal-MOTR equivalence are not
  yet executable;
- R-A currently has only two plausible, unproven route-level deltas: atomic
  release-before-same-bin-reseed and continuous carrier mass with ephemeral
  noncanonical UOT.

Required protocol-only revision:

1. resolve the 211/213 population using bound manifests and publish the exact
   difference audit;
2. freeze R0 formulas for time, bins, half-open intervals, overlap,
   repetition, same-bin transitions, one-bin actions, Ambiguous, and
   zero-action videos;
3. freeze clustered uncertainty, claim-eligibility, power, and disclosure;
4. add a prior-exposure ledger and define `model-outcome-blind`;
5. freeze R0 environment isolation, aggregate author disclosure,
   reviewer-only detailed commitments, atomic publication, and no-overwrite;
6. freeze R1 provenance schema, token support proof, perturbation audit, cache
   linkage, and fail-unverifiable action;
7. uniquely define B0-B4 and the common fairness/calibration/evaluation
   algorithms;
8. freeze exact negative controls, core mechanism deletions, compact OOD
   grammar, hidden-generation ownership, and identifiability KILL;
9. freeze an R-A versus temporal-MOTR exact-delta table and numerical/formula
   equivalence region.

Absorption amendments:

- do not automatically choose canonical 213 until external manifests and the
  exact two-ID difference are verified;
- define sequential gap only as `next_start - current_end`;
- treat the review's sample thresholds, parameter/MAC bands, equivalence
  margins, and bootstrap count as candidates requiring exact justification,
  not already frozen constants;
- separate static support proof, same-environment perturbation invariance, and
  existing-cache byte linkage; CPU output need not byte-match a historical GPU
  float16 cache;
- reject a globally consistent class-label permutation as a semantic control,
  because it merely renames classes;
- predeclare whether fairness is capacity-matched, resource-matched, or
  reported under both views.

Rejected alternatives:

- run R0 or R1 because they are read-only;
- use this Pro response itself as the hashed protocol;
- inspect model predictions, checkpoints, effects, or hard-case IDs while
  revising the protocol;
- revive historical P0 constants;
- freeze model dimensions, UOT solver, optimizer, ledger thresholds, or GPU
  budgets in a route-evidence protocol;
- interpret D1 or D2 as established novelty.

Source:

- private attachment
  `ae88cc1d-c95c-483c-9b04-5fdbbec2c1e5/pasted-text.txt`, archived as
  `PRO_PREFIX_ROUTE_PROTOCOL_FREEZE_REVIEW_20260717.md`, 47,878 bytes,
  971 lines, SHA-256
  `F9064827B05991FD63156770B2A1D6D6D40C3E53A73F4113B31F312B17F82BD0`;
- [independent absorption](../PRO_PREFIX_ROUTE_PROTOCOL_FREEZE_REVIEW_ABSORPTION_20260717.md),
  SHA-256
  `73F6C034C05F9FBAB37FD8650ECD0152B52FAFEC55487842AB93E4C4D532D54B`;
- [route-identifiability gate](experiments/prefix-route-identifiability-gate-20260717.md);
- [T40](discussion_timeline.md#t40-protocol-review-blocks-r0-and-r1-before-collection).

Reversibility:

- a new independent protocol PASS may authorize only R0/R1 evidence
  collection. It cannot authorize model-contract freeze, model code, model P0,
  real-data effectiveness, or GPU work.

## DR-048: Freeze Executable Protocol V1 Without Authorizing Collection

Status: active protocol candidate; independent protocol review pending; R0/R1 and model work blocked.

Decision:

> Freeze `prefix-route-identifiability-20260717-v1` as the only current
> protocol candidate. Its machine-readable JSON, validator, CLI, source
> bindings, and tests replace the earlier checklist. Local validity is not an
> independent PASS: collection remains blocked until one reviewer binds the
> committed protocol bytes and returns
> `PASS_PROTOCOL_TO_OUTCOME_BLIND_EVIDENCE_COLLECTION`.

Frozen choices:

- canonical 213 is the only possible primary reporting population, but only
  after a source-derived 211/213 certificate explains every differing ID;
  unavailable or unexplained evidence blocks R0, and historical 211 remains
  audit-only;
- R0 uses half-open intervals, stride-8 decision bins, signed gap
  `next_start-current_end`, video clusters, aggregate-only author disclosure,
  reviewer-only hard-case details, and no-overwrite publication;
- primary stress eligibility requires 30 videos, 100 GT, 5% positive-video
  prevalence, three classes, and conservative power at least 0.8 for a
  10-point recall effect under family-wise alpha 0.05 and ICC 0.2;
- R1 separates static support, same-environment perturbation, and historical
  cache identity; missing identity is a recordable `FAIL_UNVERIFIABLE`, not a
  reason to synthesize hashes or create a replacement cache;
- B0-B4 are unique; primary evidence requires both capacity and resource
  matching, with 5% trainable-parameter and 10% training-MAC tolerances plus
  equal tokens, updates, trials, calibration, evaluator, and seeds;
- global class renaming is forbidden as a semantic control; a deterministic
  instance-label derangement is used instead;
- R5 freezes four factor families and reviewer-owned compound OOD;
- D1 and D2 are the only provisional route deltas; all persistent-query,
  interval-head, memory, and ledger ingredients remain `NO_DELTA`;
- B4 equivalence or inferiority to temporal MOTR is terminal.

Local verification:

- protocol SHA-256:
  `E49E32E21E5DA10342694637FBBA7ACDD7FDB7DAC5CD6F1C248355A24FD018DC`;
- validator source SHA-256:
  `5B9D4D8E0DB8ACA25F870E7E65182456852D8895D94C02DAF1AD90B282B2DE3A`;
- CLI source SHA-256:
  `270D71F3CD6973E9CA8719A6E2C497D058652D959DC1CEC2D970C40F7B48AA27`;
- `tests/test_prefix_route_protocol.py`: 16 passed;
- protocol validation returns `PROTOCOL_VALID_REVIEW_REQUIRED`;
- collection authorization without an independent PASS returns
  `BLOCKED_PENDING_INDEPENDENT_PROTOCOL_REVIEW` with exit code 2.

Known unresolved evidence:

- the remote N16R4 host could not be resolved while V1 was written, so the
  exact 211/213 ID difference is not claimed as verified;
- the existing cache remains unverified and no R1 result has been collected;
- no annotation aggregate, hard-case ID, model output, checkpoint, or GPU
  result was opened or produced.

Rejected alternatives:

- call local tests an independent protocol PASS;
- guess the missing 211/213 IDs or select 213 without a source certificate;
- make failed R1 evidence structurally impossible to record;
- use a review certificate without binding the actual review artifact;
- freeze model dimensions, UOT constants, optimizer budgets, or results in
  this route-evidence protocol;
- run R0/R1 before the independent review.

Sources:

- [`../PREFIX_ROUTE_EVIDENCE_PROTOCOL_V1.md`](../PREFIX_ROUTE_EVIDENCE_PROTOCOL_V1.md);
- machine protocol
  [`../configs/causaltad/protocols/prefix_route_identifiability_v1.json`](../configs/causaltad/protocols/prefix_route_identifiability_v1.json);
- validator
  [`../opentad/utils/prefix_route_protocol.py`](../opentad/utils/prefix_route_protocol.py);
- CLI
  [`../tools/validate_prefix_route_protocol.py`](../tools/validate_prefix_route_protocol.py);
- [route gate](experiments/prefix-route-identifiability-gate-20260717.md);
- [T41](discussion_timeline.md#t41-executable-protocol-v1-is-ready-for-independent-review).

Reversibility:

- an independent REVISE creates a new version and does not overwrite V1. A
  PASS authorizes only the hash-bound read-only R0/R1 scope. Model contracts,
  code, P0, effectiveness, profile, training, and all GPU work remain blocked.

## DR-049: Accept the Independent V1 REVISE and Preserve the Collection Block

Status: terminal for Protocol V1; superseded as a candidate by DR-050; zero GPU hours.

Decision:

> Accept the sole independent review verdict
> `REVISE_PROTOCOL_BEFORE_COLLECTION` on commit
> `5e858a161254395488428609b441b64f3c08be76`. Protocol V1 is not an
> executable evidence authorization contract. Preserve it and the full review
> as immutable negative protocol evidence; do not collect R0 or R1.

Reasons:

- an author could forge V1's review certificate and pass an unvalidated
  in-memory authorization dictionary;
- population, R0, and R1 accepted asserted statuses and dummy identities
  without re-reading source evidence;
- eight hostile semantic weakenings passed validation;
- R0 boundary rules and collector, reporting-label exposure, B0-B4 fairness,
  D1/D2 isolation, controls, R5, R6 inference, source binding, and adversarial
  tests were incomplete;
- the review accessed no model output, checkpoint, R0/R1 result, or GPU.

Rejected alternatives:

- reinterpret the V1 local `60/60` tests as independent approval;
- patch the certificate text while leaving asserted evidence paths intact;
- proceed to population or cache collection because those stages are
  nominally read-only;
- hide annotation exposure or call THUMOS reporting annotation-unseen.

Sources:

- [`../PRO_PREFIX_ROUTE_PROTOCOL_V1_INDEPENDENT_REVIEW_20260717.md`](../PRO_PREFIX_ROUTE_PROTOCOL_V1_INDEPENDENT_REVIEW_20260717.md),
  SHA-256
  `1FDB36DB2D970D7B044EC26B3366208CD3D3A20BD6CB3544E62AACCBE57C7326`;
- V1 commit `5e858a161254395488428609b441b64f3c08be76`;
- [T42](discussion_timeline.md#t42-independent-review-rejects-protocol-v1-before-collection).

Reversibility:

- V1 remains terminal. Only a new immutable protocol version reviewed by the
  same sole reviewer may supersede it.

## DR-050: Freeze Protocol V2 for Same-Reviewer Reassessment

Status: terminal REVISE at commit `18cc27b`; remediation superseded by DR-051; all collection and model work blocked.

Decision:

> Replace V1 only as the active candidate with
> `prefix-route-identifiability-20260717-v2`. V2 must remain outcome-blind and
> zero GPU, be committed with an exact source manifest, and return to reviewer
> `019f6f63-496d-75a0-a77b-91425a8e7ea1`. Until that reviewer verifies the
> frozen commit and signs canonical attestation bytes with the pre-frozen
> Ed25519 identity, authorization remains closed.

V2 implementation choices:

- authorization accepts file paths only, re-reads canonical protocol,
  manifest, signed attestation, signature, commit, and tree, and has no
  caller-supplied review-dictionary bypass;
- population/R0/R1 statuses are derived from contained, hash-verified source
  artifacts;
- R0 binds one collector and exact interval, endpoint, duplicate, pair,
  denominator, bootstrap, power, and aggregate-only reporting rules;
- R1 binds every cache/source object, array geometry, token support, and
  dynamic perturbation; terminal tokens use a paired synthetic continuation
  because they have no native future suffix;
- THUMOS reporting is explicitly
  `DESIGN_EXPOSED_ROUTE_SELECTION_AND_BENCHMARK`; an annotation-unseen claim
  requires a separately frozen prospective dataset;
- B2 is the faithful temporal tracking attack; fairness covers parameter,
  MAC, state, memory, latency, optimizer, token, trial, calibration, and seed
  budgets with no dummy trainable capacity;
- controls, R5 structural OOD, R6 metric bindings, crossed paired inference,
  D1/A4, joint D2/A1+A2, and `INDETERMINATE_NO_ROUTE_CLAIM` are executable.

Current evidence:

- dedicated V2 suite: `27 passed`;
- focused cross-module zero-GPU suite: `133 passed, 2 skipped`; both skips are
  Torch subprocess imports failing on local `c10.dll`;
- unsigned validation returns `PROTOCOL_V2_VALID_REVIEW_REQUIRED`;
- authorization returns
  `BLOCKED_PENDING_SIGNED_INDEPENDENT_PROTOCOL_REVIEW`;
- no R0/R1 collection, model outcome, checkpoint, profile, training, or GPU
  access occurred.

Rejected alternatives:

- self-sign or generate a replacement reviewer key;
- use a different reviewer merely to obtain a PASS;
- treat V2 code coverage as scientific route evidence;
- start B0-B4, profile, or training before the signed protocol decision.

Sources:

- [`../PREFIX_ROUTE_EVIDENCE_PROTOCOL_V2.md`](../PREFIX_ROUTE_EVIDENCE_PROTOCOL_V2.md);
- machine protocol
  [`../configs/causaltad/protocols/prefix_route_identifiability_v2.json`](../configs/causaltad/protocols/prefix_route_identifiability_v2.json);
- [V1 independent review](../PRO_PREFIX_ROUTE_PROTOCOL_V1_INDEPENDENT_REVIEW_20260717.md);
- [T43](discussion_timeline.md#t43-protocol-v2-closes-v1-bypasses-and-awaits-the-same-reviewer).

Reversibility:

- an independent `REVISE` requires V3 or a corrected V2 commit and another
  same-reviewer pass. A signed `PASS` authorizes only R0/R1 collection, never
  model implementation or GPU work.

## DR-051: Reject V2 False-PASS Paths and Rebuild the Scientific Evidence Boundary

Status: active zero-GPU remediation; V2 collection authorization remains denied.

Decision:

> Accept the same reviewer's fixed-commit V2 verdict
> `REVISE_PROTOCOL_BEFORE_COLLECTION`. Preserve V2 commit `18cc27b` as a
> reproducible but scientifically insufficient protocol. Rebuild the
> population, R1, fairness, R5, and R6 paths so no caller-authored summary can
> create a scientific PASS.

Accepted findings:

- arbitrary 211/213 lists plus opaque nonempty provenance can still produce
  `EXPLAINED_MISMATCH`;
- one-byte pseudo-videos and caller-selected token bytes can still certify R1;
- R0 accepts the training subset and emits a weaker exposure label than the
  protocol;
- B2 is not uniquely instantiated and fairness is derived from asserted
  scalars/booleans;
- protocol-required R5 TRAIN fails its own audit, no-effect shifts can evade
  detection, same-bin handoff is geometrically wrong, and compound OOD is not
  executable;
- R6 parameters and intervals are caller-controlled, negative controls are
  disconnected, and arbitrary intervals can yield route PASS;
- Windows 8.3 aliases can make contained evidence unusable.

Preserved V2 closures:

- independent review identity and collection authorization;
- exact nested semantic lock;
- fixed commit/tree/20-blob manifest and LF reproduction;
- most R0 interval/pair/bootstrap mechanics;
- literal no-op terminal-future rejection;
- semantic derangement construction;
- crossed seed/video pairing and multiplicity structure.

Required remediation:

1. independently freeze authoritative source identities, then derive
   population membership and reasons from parsed annotation/inventory bytes;
2. decode videos and re-execute the bound extractor, recomputing perturbations
   and cache-row identity;
3. enforce reporting subset and exact exposure, and keep R0 eligibility
   descriptive unless aligned downstream power is defined;
4. bind exact arm contracts and derive fairness only through executable
   instrumentation;
5. make every R5 set/count/support/balance/compound rule self-consistent;
6. seal one raw-artifact R6 entry point with fixed inference parameters,
   source-derived controls, and complete terminal schema;
7. add every reproduced false accept as a regression and repair Windows path
   canonicalization.

Sources:

- [`../PRO_PREFIX_ROUTE_PROTOCOL_V2_METHOD_REASSESSMENT_20260717.md`](../PRO_PREFIX_ROUTE_PROTOCOL_V2_METHOD_REASSESSMENT_20260717.md),
  SHA-256
  `E411804F5BB744CBBE776A77B63701957EC607A3CA2293B4D523A23A656EC4CF`;
- fixed V2 commit `18cc27b8496f5effa7e4e93abca58da4e7ea3d0b`;
- [T44](discussion_timeline.md#t44-v2-reassessment-closes-reproducibility-but-rejects-scientific-evidence-derivation).

Reversibility:

- none of the false-PASS closures may be relaxed after source or model outcomes
  are observed. Only another same-reviewer fixed-commit PASS can advance the
  authorized scope.

## DR-052: Freeze the V2 Remediation Candidate Without Starting Collection

Status: zero-GPU implementation complete; same-reviewer fixed-commit
reassessment pending; all collection, model, profile, and training gates
remain blocked.

Decision:

> Treat the current V2 remediation as a review candidate only. Its local
> tests establish that the known V2 false-PASS paths are rejected; they do not
> establish source provenance, cache validity, model fairness, D1/D2 value, or
> route effectiveness. Commit and push the exact source-manifest-bound
> candidate, then return it to the same sole reviewer
> `019f6f63-496d-75a0-a77b-91425a8e7ea1`.

Implemented closures:

- population membership and 211/213 reasons are derived from registered
  annotation, inventory, alias, and artifact bytes; the current unregistered
  state blocks population, R0, and R1;
- R0 requires the `validation` subset, emits the exact design-exposure label,
  and is descriptive rather than a mismatched downstream power claim;
- R1 binds the exact extraction identity, canonical resolved command, and
  active software versions; it decodes every video, recreates the local-only
  encoder, reruns every cache row and perturbation, and compares transcript
  and cache bytes;
- B2 has one executable 64-track/64-newborn temporal-MOTR lifecycle, while
  fairness rejects serialized assertions and requires live model, gradient,
  profiler, CUDA, latency, and state measurements; exact budgets come from
  hash-verified records, the profiled event covers every accumulation
  microbatch, and the optimizer must contain exactly all trainable parameters;
- R5 generates and validates the exact public 3,800-sequence package,
  including no-effect counterfactuals, same-bin geometry, compound cells,
  balance, disjointness, and frozen set commitments;
- R6 accepts raw immutable emissions only, rereads the canonical protocol and
  registered-source R0 envelope plus committed 213-video detail, derives all
  metrics, fixes 10,000 resamples and alpha 0.05, and makes benchmark,
  temporal, semantic, any-global B2 inferiority, and D1/D2 terminal decisions
  reachable;
- Windows 8.3 and long-path containment are canonicalized consistently.

Local evidence:

- protocol SHA-256:
  `77DAC32A878A07316E3384B7BAE2D7768D57830479A3916EAF0273B788C8ECF8`;
- policy-lock SHA-256:
  `291FB2405D73B1AC2D7A64E773B583BFD9B558E4C851949AC9F87A249F4F5259`;
- source manifest: 22 entries, SHA-256
  `DDD2DBAFF706D12E8EBAED7E40AA53919B0836E623ACE9EFA5DFD80E1B1CDC27`;
- dedicated hostile suite: `40 passed`;
- focused zero-GPU cross-module suite: `135 passed, 6 skipped`; all six skips
  are local Torch import failures, not converted to PASS;
- unsigned protocol status:
  `PROTOCOL_V2_VALID_REVIEW_REQUIRED`;
- GPU hours and model outcomes inspected: zero.

Remaining unknowns:

- authoritative THUMOS14 annotation/inventory and R1 execution identities are
  intentionally unregistered;
- no real R0 or R1 evidence has been collected;
- no B0-B4 model contract or live fairness measurement is authorized;
- R5 is design-exposed stress, not hidden confirmatory OOD;
- D1, D2, B4 noninferiority, and the paper route remain untested.

Sources:

- [Protocol V2](../PREFIX_ROUTE_EVIDENCE_PROTOCOL_V2.md);
- [machine protocol](../configs/causaltad/protocols/prefix_route_identifiability_v2.json);
- [V2 REVISE review](../PRO_PREFIX_ROUTE_PROTOCOL_V2_METHOD_REASSESSMENT_20260717.md);
- [T45](discussion_timeline.md#t45-v2-remediation-closes-known-false-pass-paths-but-remains-review-only).

Reversibility:

- a same-reviewer `REVISE` returns to zero-GPU protocol repair;
- a valid signed PASS authorizes only the protocol's current
  `READ_ONLY_SOURCE_IDENTITY_REGISTRATION` scope;
- no local test result can authorize source collection, model code, profile,
  or training.

## DR-053: Accept Round-3 REVISE and Close the Six Reproduced Bypasses

Status: zero-GPU remediation complete locally; exact fixed-commit reassessment
by the same reviewer is pending.

Decision:

> Accept the sole reviewer's round-3 verdict
> `REVISE_PROTOCOL_BEFORE_COLLECTION` at commit `b0ac8b1`. Do not reinterpret
> the prior local test pass as protocol authorization. Repair all six hostile
> probes, freeze a new source-manifest-bound candidate, and return only that
> exact commit to reviewer `019f6f63-496d-75a0-a77b-91425a8e7ea1`.

Accepted findings:

- R6 accepted caller-selected protocol/R0/fairness/emissions and could produce
  a route PASS without a real run chain;
- the unregistered state still exposed a low-level R0 PASS path;
- B2 performed post-hoc dustbin deletion rather than augmented assignment and
  did not bind decision bins to observation counts;
- fairness records and later optimizer events were insufficiently tied to
  model states, inputs, and R6 runs;
- R5 scientific payloads could be exchanged while retaining factor metadata;
- mutable bootstrap globals and incomplete interval dictionaries could reach
  terminal PASS; import-time source files were omitted from the manifest.

Implemented closure:

- population and R0 PASS paths require registered source state, the canonical
  repository protocol, current manifest, a cryptographically verified
  same-reviewer PASS, and source-derived canonical-213 records;
- registered source revision, annotation identity, reviewed-parent commit, and
  decodable historical videos are mandatory;
- B2 uses one augmented Hungarian matrix with 64 unique dustbins,
  deterministic nonseparable target ties, and
  `decision_bin=(observation_count-1)//8`;
- standalone fairness evidence is forbidden; live audit and every formal run
  bind optimizer events, input batches, model/optimizer state chains,
  command, environment, model artifacts, fairness, and emissions;
- every R6 control additionally binds its exact algorithm and parameters, a
  complete ordered per-video source/construction transcript, and output
  emissions;
- R5 generator `20260717.4` regenerates each row from
  factor/seed/index/set, carries separate scientific-content and full-row
  commitments, and freezes seven newly recomputed set hashes;
- R6 loads only the canonical signed source chain, uses literal bootstrap
  constants, validates the complete inference schema, and binds each control
  construction; import-time package initializers are manifest-bound.

Local evidence:

- protocol SHA-256:
  `493A132BBC9998AD7DD1D4756EC882DEE858AC8C739DBF96C618B1459BE3AC65`;
- policy-lock SHA-256:
  `C9594FF7AB2E89549E512F5528BDDEFE127ECDA12D3C8B900F3E0E5D9BF598C0`;
- 32-entry source-manifest SHA-256:
  `05BF8A8925B465522BCB7470D1810D2421B99C3F7E15512AD3C87C6435AA8C25`;
- dedicated V2 suite: `46 passed`;
- broader 11-file zero-GPU suite: `148 passed, 1 skipped`; the skip is a
  fail-closed isolated Torch import failure;
- unsigned authorization remains
  `BLOCKED_PENDING_SIGNED_INDEPENDENT_PROTOCOL_REVIEW`, exit code 2;
- R0/R1 collection, model outcomes, checkpoints, profile, training, and GPU
  access remain zero.

Sources:

- [round-3 independent review](../PRO_PREFIX_ROUTE_PROTOCOL_V2_ROUND3_INDEPENDENT_REVIEW_20260717.md),
  SHA-256
  `039FCB7868F31DA9A7153DADF6CB60E29C14916DD6DBA81F926F7EE2B28A28A9`;
- [Protocol V2](../PREFIX_ROUTE_EVIDENCE_PROTOCOL_V2.md);
- [machine protocol](../configs/causaltad/protocols/prefix_route_identifiability_v2.json);
- [T46](discussion_timeline.md#t46-round-3-review-rejects-six-bypasses-and-a-new-candidate-closes-them-locally).

Reversibility:

- a same-reviewer `REVISE` requires another zero-GPU repair and fixed-commit
  reassessment;
- a valid signed PASS authorizes only read-only source-identity registration;
- no source collection, model implementation, profile, or training may start
  from these local results.
