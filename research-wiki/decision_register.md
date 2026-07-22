---
type: decision_register
updated: 2026-07-11
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

Sources:

- `docs/superpowers/specs/2026-07-22-raw-rgb-dynamic-event-memory-ontal-design.md`;
- [ideas/dynamic-event-memory-rgb-ontad.md](ideas/dynamic-event-memory-rgb-ontad.md);
- [experiments/ontad-rgb-dynamic-event-memory-design-20260722.md](experiments/ontad-rgb-dynamic-event-memory-design-20260722.md).

Reversibility:

- Fusion membership and visual backbone may change after matched evidence. The
  read-only preservation of official baselines, strict causal contract, and
  requirement that final evidence include original-RGB input are not relaxed by
  feature-level results.
