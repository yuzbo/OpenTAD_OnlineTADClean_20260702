---
type: timeline
updated: 2026-07-11
status: active
scope: Chronological record of major discussions, turns, reversals, and decisions for this Online/Causal TAD project.
---

# Discussion Timeline

This page records the project-level conversation history. It is not a verbatim transcript; it is the structured memory of turning points that should prevent repeated loops.

## How to Use This Page

Before reopening an old route, check:

- what question triggered it;
- what answer or Pro review changed the direction;
- why it was accepted, demoted, or rejected;
- which wiki node now owns the idea.

## Timeline

### T0: Project Entry and Initial Concern

User concern:

> The current online TAD implementation is poor and has no clear innovation.

Initial demand:

- build a complete prompt for a Pro model;
- force task-definition review, innovation review, clarification questions, claim map, candidate methods, experimental loop, baselines, ablations, metrics, and reviewer risks;
- use the GitHub branch/repo address as the concrete code anchor.

Resulting direction:

- stop writing overconfident method proposals;
- first audit whether the task is truly online and whether the innovation is meaningful.

Current wiki nodes:

- [gap_map.md](gap_map.md)
- [ideas/rejected-one-shot-emission.md](ideas/rejected-one-shot-emission.md)

### T1: What Is the Current Project Actually Doing?

User questions:

- What is the current project goal?
- What is the task?
- Is current online inference window-based or frame-by-frame?
- How large is the window?
- Is the delay too large?

Absorbed conclusion:

- Current early route was mostly a windowed or chunk-end causalized TAD skeleton.
- It was not a true low-latency continuous online agent.
- Window length, context horizon, decision cadence, and latency budget must be separated.

Key rejection:

- Do not call long-window chunk-end emission low-latency Online TAD.

Current wiki nodes:

- [gap_map.md#G1-online-tad-needs-a-better-protocol-than-one-shot-immutable-emission](gap_map.md)
- [ideas/rejected-one-shot-emission.md](ideas/rejected-one-shot-emission.md)

### T2: First Strong Method Candidate - Online-Censored Boundary / Emission Hazard

User pushed:

- current method lacks innovation;
- can we solve missing pretrained-model support for online TAD?
- can we build a truly end-to-end trainable, low-delay On-TAD model?
- how do existing methods convert offline to online?
- what matters most for end-to-end trainable On-TAD?

Pro reviews and discussion converged on:

> Prefix-observed online-censored localization with bounded-delay emission hazard learning.

This became:

> PCEH-OnTAD: Prefix-Censored Event-Emission Hazard Learning for Online Temporal Action Detection.

Original promise:

- separate physical endpoint `E`, prefix state/completion, and emission time `A`;
- learn endpoint hazard and bounded-delay first-emission policy;
- evaluate OnlineAP with GT-end latency.

Current status:

- demoted from full story to CESR submodule after later user correction.

Current wiki nodes:

- [ideas/pceh-ontad.md](ideas/pceh-ontad.md)
- [claims/c3-risk-set-hazard.md](claims/c3-risk-set-hazard.md)

### T3: OZ-TAL Discovery and Open-Vocabulary Boundary

User surfaced:

> Online / Streaming Zero-Shot TAL: OZ-TAL 2026 solves real-time unseen action detection using off-the-shelf VLMs.

Absorbed conclusion:

- OZ-TAL is a major competitive work.
- We must not claim first VLM/open-vocabulary/zero-shot online TAL.
- If using VLMs or unseen classes, OZ-TAL must be a primary competitor.

Safe zone after this:

> closed-set, trainable, task-aligned Online TAD objectives under prefix-observed event-state supervision.

Current wiki nodes:

- [ideas/open-vocabulary-zero-shot.md](ideas/open-vocabulary-zero-shot.md)
- [papers/oztal2026-zero-shot.md](papers/oztal2026-zero-shot.md)

### T4: TAD vs TAL Terminology Clarification

User asked:

> What is the difference between video temporal localization and temporal action localization?

Absorbed practical distinction:

- TAL / TAD in this project refers to action instance localization/detection: temporal start, temporal end, class, confidence.
- OAD is usually frame/clip-level current action classification.
- Online TAS is dense frame-level segmentation, conceptually relevant but not the same task.

Current wiki nodes:

- [papers/onlinetas2024-online-segmentation.md](papers/onlinetas2024-online-segmentation.md)
- [query_pack.md](query_pack.md)

### T5: Implementation Snapshot and Remote Training Reality

User asked to:

- start implementation;
- organize current project;
- check whether remote experiments were already running;
- ask whether current training would be slow;
- ask whether training event cost was unacceptable.

Observed/absorbed state:

- local implementation added PCEH skeleton and audits;
- local tests passed for the touched suite in one coding pass;
- remote PCEH smoke completed once;
- several pilots were cancelled/incomplete;
- full-packet training was too slow and not acceptable as default.

Cost estimate absorbed:

- about 152,670 packets per epoch;
- about 7 hours per epoch;
- about 210 GPU-hours per model per seed for 30 epochs;
- about 1260 GPU-hours for PCEH vs endpoint-only with 3 seeds.

Current wiki nodes:

- [experiments/pceh-smoke-20260710.md](experiments/pceh-smoke-20260710.md)
- [experiments/cancelled-pilots-20260710.md](experiments/cancelled-pilots-20260710.md)
- [ideas/rejected-full-packet-training.md](ideas/rejected-full-packet-training.md)
- [ideas/crs-eps-training.md](ideas/crs-eps-training.md)

### T6: Training-Cost Redesign - CRS-EPS

User requested:

> Form a complete prompt asking Pro to research current online video training methods and discuss how to train under our model design.

Pro review 2026-07-11 and subsequent discussion selected:

> Instance-aware Causal Risk-Set Event-Centric Prefix-Episode Training.

Working shorthand:

> CRS-EPS.

Core design:

- do not train every packet by default;
- sample start-centered, endpoint-centered, ongoing, hard background, and long background episodes;
- use 192-token context/burn-in plus 4-8 supervised bins;
- use frozen SigLIP2 feature cache for Stage 1;
- use LoRA only after Stage 1 evidence.

Critical user correction:

> Endpoint-near sampling is not enough; start stage must also be sampled.

Current wiki nodes:

- [ideas/crs-eps-training.md](ideas/crs-eps-training.md)
- [ideas/state-transition-sampling.md](ideas/state-transition-sampling.md)
- [ideas/feature-cache-stage1.md](ideas/feature-cache-stage1.md)
- [ideas/lora-stage2.md](ideas/lora-stage2.md)

### T7: Pro Review 2026-07-11 - PCEH Scientific Blockers

The 2026-07-11 review changed the state from "PCEH implementation candidate" to:

> HOLD; formal training blocked until P0 scientific correctness gates pass.

Critical blockers:

- endpoint and emission labels are the same event;
- late prefixes are wrongly treated as repeated emission positives;
- targets are class-aggregated, not instance-aware;
- predicted end equals emit time;
- GT is too close to model metadata.

Required P0 repairs:

- instance-aware risk sets;
- first-event endpoint and first-commit/emission targets;
- post-event masking;
- delayed synthetic case where predicted end precedes commit/emit;
- GT taint audit.

Current wiki nodes:

- [claims/c3-risk-set-hazard.md](claims/c3-risk-set-hazard.md)
- [gap_map.md#G3-current-pceh-targets-are-not-statistically-correct](gap_map.md)

### T8: User Reframes the Task - From One-Shot Emission to State Refinement

User objected:

> The one-shot formulation is not elegant. The model should judge after state changes, such as action start and action end, and should optimize/correct previous online outputs as observation changes.

This is the largest conceptual pivot after PCEH.

New direction:

> Causal Event-State Hypothesis Tracking / Refinement for Online Temporal Action Detection.

Core protocol:

```text
mutable online hypothesis
-> timestamped revision history
-> immutable committed detection
```

What changed:

- PCEH is no longer the full paper story.
- PCEH becomes one submodule for endpoint and commit hazards.
- The main story becomes track, refine, and commit.

Current wiki nodes:

- [ideas/cesr-ontad.md](ideas/cesr-ontad.md)
- [claims/c2-mutable-hypothesis-commit.md](claims/c2-mutable-hypothesis-commit.md)
- [ideas/rejected-one-shot-emission.md](ideas/rejected-one-shot-emission.md)

### T9: Competition Check for CESR

User asked:

> Check whether there is related competing work.

Initial competition conclusion:

- no exact equivalent found for mutable hypothesis plus immutable commit ledger as the central On-TAD protocol;
- but several near neighbors are strong.

Closest threats:

- OAT: early proposal and boundary refinement;
- MATR: memory-based online span localization;
- ActionSwitch: state-change boundary and same-class/concurrent actions;
- HAT: history-enhanced anchors;
- OnPoint: offline-to-online distillation;
- OZ-TAL: zero-shot/open-vocabulary online TAL;
- OnlineTAS: online prediction correction in segmentation.

Defensible delta:

> revision/commit lifecycle, auditable hypothesis updates, and instance-level Online TAD evaluation.

Current wiki nodes:

- [papers/oat2022-online-tal.md](papers/oat2022-online-tal.md)
- [papers/actionswitch2024-state.md](papers/actionswitch2024-state.md)
- [papers/matr2024-memory.md](papers/matr2024-memory.md)
- [papers/oztal2026-zero-shot.md](papers/oztal2026-zero-shot.md)

### T10: User Flags Memory Drift and Requests Research Wiki

User observed:

> You keep forgetting issues and spinning in place.

Action taken:

- removed repository rule that banned `research-wiki/`;
- unignored `research-wiki/` in `.gitignore`;
- created the first structured wiki with index, gap map, query pack, ideas, papers, claims, experiments, graph, and log.

Key design:

- `research-wiki/query_pack.md` is the first file to read before ideation;
- failed ideas are deliberately prominent;
- PCEH demotion and CESR selection are explicit.

Current wiki nodes:

- [index.md](index.md)
- [query_pack.md](query_pack.md)
- [gap_map.md](gap_map.md)

### T11: User Requests Timeline / Decision Register / Source Map

User requested:

- `research-wiki/discussion_timeline.md`;
- `research-wiki/decision_register.md`;
- `research-wiki/source_map.md`.

Action:

- this timeline page is the chronological memory layer;
- decision register records reversible/irreversible choices;
- source map links Pro reviews, attachments, online notes, and code snapshots to wiki nodes.

Current wiki nodes:

- [discussion_timeline.md](discussion_timeline.md)
- [decision_register.md](decision_register.md)
- [source_map.md](source_map.md)

### T12: Fresh Overlap Audit Narrows CESR

User asked:

> Investigate overlapping work and determine how online training can avoid unaffordable cost.

New literature findings:

- CAG-QIL and SimOn make sequential state/decision history old territory.
- OAT makes early proposal and boundary refinement old territory.
- ProTAS makes ongoing action-progress refinement old adjacent territory.
- OpenHOUSE blocks a broad hierarchical streaming semantics claim.
- Thinking-QwenVL and StreamReady block a generic hypothesis-revision / when-to-respond claim.
- ETAD provides direct precedent for selective-gradient and proposal sampling.

Independent novelty review:

- broad CESR / "track, refine, commit": about 5.5/10 novelty;
- recommendation: proceed with caution only after narrowing;
- strongest surviving route: identity-preserving belief trajectory + utility-based first commit + auditable trajectory metrics;
- CRS-EPS is a cost surrogate requiring full-stream bias audits.

Resulting pivot:

> The paper is no longer about generic event-state maintenance. It is about identity-linked temporal action belief trajectories and irreversible detection commit under explicit utility.

Current wiki nodes:

- [ideas/cesr-ontad.md](ideas/cesr-ontad.md)
- [ideas/crs-eps-training.md](ideas/crs-eps-training.md)
- [gap_map.md#G9-broad-online-video-semantic-maintenance-is-already-crowded](gap_map.md)

### T13: User Rejects Incremental On-TAL and Reopens the Scientific Question

User challenge:

> The route feels crowded and small modifications are not valuable. Are there still worthwhile problems in On-TAL, online temporal localization, or online reasoning?

Fresh strategic audit:

- Classical closed-set On-TAL remains small as a publication community, but its common method space is crowded: memory, state switches, early proposals, refinement, progress, and stopping all have close precedents.
- Generic streaming pretraining is directly threatened by StreamFormer.
- Generic selective memory is threatened by SelectStream and related streaming VLM work.
- Asynchronous queries are threatened by AViLA.
- Adaptive compute is threatened by SAN, AdaFrame, and DyBDet.
- Continual/open-world localization remains meaningful but combines several mature adjacent areas and requires a new benchmark.

Lead new gap:

> A physical action endpoint is not necessarily the first timestamp at which the action class and completed span are causally knowable, and current On-TAL does not control false commits over indefinite monitoring.

Resulting route portfolio:

1. Lead candidate: evidence-aligned, risk-controlled On-TAL.
2. High-ambition alternative: continual open-world On-TAL under chronological shifts and delayed feedback.
3. Supporting/systems route: compute-adaptive active sensing tied to localization risk.
4. CESR identity trajectory: retained as substrate/fallback, not headline.

Current wiki nodes:

- [ideas/evidence-risk-ontal.md](ideas/evidence-risk-ontal.md)
- [ideas/continual-openworld-ontal.md](ideas/continual-openworld-ontal.md)
- [ideas/active-sensing-ontal.md](ideas/active-sensing-ontal.md)

### T14: User Rejects Current Idea Taste and Requests Unanchored Pro Ideation

User assessment:

> Current topic taste is poor; produce a fully divergent, assumption-free prompt for a Pro model to generate genuinely innovative and valuable ideas.

Action:

- created `PRO_DIVERGENT_IDEA_PROMPT_20260711.md`;
- anchored public code review to branch `codex/online-tad-clean-20260702`, visible HEAD `bfd0608b2996cba30d158d741ee193476a5078df`;
- included current direct and adjacent literature threats through 2026-07-11;
- explicitly made CESR, evidence-time/risk control, On-TAL itself, and code reuse rejectable;
- required 36 raw ideas across six independent lenses before ranking;
- required fresh primary-source novelty checks, a Senior-PC kill round, scoring thresholds, Top 5, `NO-GO` permission, and a 48-hour falsification plan.

Artifact:

- [`../PRO_DIVERGENT_IDEA_PROMPT_20260711.md`](../PRO_DIVERGENT_IDEA_PROMPT_20260711.md)

### T15: Pro Divergent Review Received, Archived, and Partially Rejected

User requested:

> Completely record and absorb the Pro response, and judge whether its advice should be fully accepted.

Actions:

- archived the full 96,742-byte response byte-identically as `PRO_DIVERGENT_IDEA_REVIEW_20260711.md`;
- recorded all 36 raw ideas, the kill matrix, Top 5, one recommendation, one high-risk route, one no-go, and the 48-hour plan;
- reconciled the review with current local code;
- ran an independent closest-prior and statistical-validity audit;
- created `PRO_DIVERGENT_IDEA_ABSORPTION_20260711.md` as the decision layer.

Pro recommendation:

1. T01 Anytime-Valid Semantic Event Alarms;
2. T02 Three-Clock Event Observability as high-risk/high-reward;
3. no-go for continuing PCEH as the THUMOS headline.

Independent correction:

- accept the PCEH no-go and cheap-falsification discipline;
- do not accept T01 as the selected route;
- WACV 2026 already applies an e-process with provable false-alert control to a real-time video task;
- online video anomaly detection already studies false-alarm bounds;
- the proposed 100-null-stream pilot cannot substantiate near-`alpha` validity;
- block-concatenated THUMOS is diagnostic rather than final null-stream evidence;
- `P0(ever alarm) <= alpha` may be operationally too strict for indefinite deployment.

Current result:

> T01 and T02 are conditional P0 co-candidates. The final route remains open until novelty, statistical-validity, data, decision-value, and cost gates pass.

Current wiki nodes:

- [ideas/anytime-semantic-event-alarms.md](ideas/anytime-semantic-event-alarms.md)
- [ideas/three-clock-event-observability.md](ideas/three-clock-event-observability.md)
- [decision_register.md#DR-021-accept-the-pro-no-go-but-not-its-final-route-selection](decision_register.md)
- [`../PRO_DIVERGENT_IDEA_ABSORPTION_20260711.md`](../PRO_DIVERGENT_IDEA_ABSORPTION_20260711.md)

### T16: Three-Clock Route Refined into PIVOT After Direct Competition Audit

User requested:

> Complete task and method design, competition novelty review, and a Pro deep-review prompt for the Three-Clock direction.

Fresh findings:

- APT 2026 already defines and temporally localizes atomic physical transitions tied to visible evidence and mechanisms.
- PaSBench-Video already evaluates first visible danger, accident boundaries, and causal warning time.
- Ego4D PNR already provides PRE/CONTACT/PNR/POST and state-change keyframe localization.
- StreamReady and Thinking-QwenVL already formalize evidence windows or first-sufficient response timing.
- TouchMoment already detects precise hand-object contact moments.
- FEEL, EgoTouch, and EgoTactile already connect video with force/tactile supervision.
- STARE and Towards Streaming Perception already establish latency-aware evaluation and ranking reversal.

Resulting refinement:

> PIVOT measures a view-conditioned observability gap between an independently sensed physical transition and a population-level visual verification interval, then evaluates only the residual delay of a causal model decision.

Critical correction:

- the first paper is streaming event verification, not ordinary On-TAL;
- `C_phys` and `C_vis` are intervals, not fake-precise points;
- `C_vis` depends on view, query, observer population, and confidence threshold;
- anticipation before the physical event is not verification;
- ordered survival heads are a supporting method, not the novelty.

Data audit:

- FEEL's official Dataset/Code buttons currently point to placeholder URLs and return 404;
- TouchAnything states EgoTouch will be released, but access was not verified;
- model implementation is therefore blocked until a public-data or controlled-collection measurement pilot is feasible.

Artifacts:

- [`../THREE_CLOCK_TASK_METHOD_DESIGN_20260711.md`](../THREE_CLOCK_TASK_METHOD_DESIGN_20260711.md)
- [`../THREE_CLOCK_COMPETITION_REVIEW_20260711.md`](../THREE_CLOCK_COMPETITION_REVIEW_20260711.md)
- [`../PRO_THREE_CLOCK_DEEP_REVIEW_PROMPT_20260711.md`](../PRO_THREE_CLOCK_DEEP_REVIEW_PROMPT_20260711.md)

Current decision:

> PIVOT remains the strongest current scientific candidate, but only the low-cost measurement pilot is GO. Model implementation and training remain HOLD pending Pro review, data access, observability reliability, and rank-reversal evidence.

### T17: User Rejects PIVOT and Fixes the Task to Standard On-TAD

User correction:

> PIVOT has left the On-TAD task and is rejected. Do not innovate by changing the task. Investigate why On-TAD performance remains low and whether a genuinely end-to-end trainable On-TAD method is still missing.

Fresh audit:

- MATR reports 49.5 average mAP on THUMOS14 versus 66.8/69.3 for offline ActionFormer/TriDet in its comparison, and freezes TSN/I3D features.
- The public MATR repository consumes pre-extracted pickle features despite calling the detector architecture end-to-end.
- HAT, ActionSwitch, SimOn, OAT, and OnPoint also operate on pre-extracted or frozen features.
- E2E-LOAD trains raw-video online action detection end-to-end, but outputs frame-level OAD predictions rather than action instances.
- StreamFormer trains a causal raw-video backbone, but freezes it for downstream frame-level OAD.
- TIA/AdaTAD, LoSA, Re2TAL, and related methods adapt raw-video backbones for offline TAL, where future context remains available.

New lead candidate:

> PETAL-OnTAD keeps the standard On-TAD input/output protocol, tracks each latent action instance with one persistent event query, jointly adapts a causal raw-video backbone, and trains all prefixes in batched causal chunks that must match stepwise cached inference.

Scientific boundary:

- persistent pre-end tracks are internal model state, not a new task output;
- the model emits the ordinary `{start, end, class, score}` instance when its end is detected;
- no sensor, new annotation, task extension, revision metric, or broad streaming-video claim is required;
- the method is killed before raw-video training if persistent queries do not beat a matched fresh-window detector.

Current wiki nodes:

- [ideas/petal-ontad.md](ideas/petal-ontad.md)
- [gap_map.md#G15-On-TAD-Still-Lacks-Raw-Video-Instance-Level-Joint-Training](gap_map.md)
- [decision_register.md#DR-024-Reject-PIVOT-Because-It-Leaves-the-On-TAD-Task](decision_register.md)
- [decision_register.md#DR-025-Select-Persistent-End-to-End-Event-Tracking-as-the-New-Lead-Candidate](decision_register.md)

### T18: PETAL Pro Deep-Review Prompt Created

User requested:

> Provide a complete prompt for Pro to audit and discuss the current PETAL idea.

Action:

- created `PRO_PETAL_ONTAD_DEEP_REVIEW_PROMPT_20260712.md`;
- anchored public code review to branch `codex/online-tad-clean-20260702`, visible HEAD `bfd0608b2996cba30d158d741ee193476a5078df`;
- fixed the task to standard fully supervised On-TAD and prohibited PIVOT, Online TAS, VideoQA, sensors, and broader streaming semantics;
- decomposed PETAL into raw-video joint training, persistent instance queries, trajectory assignment, prefix-parallel/incremental equivalence, and no-NMS claims;
- required fresh primary-source search through 2026-07-12 and explicit comparisons with MATR, ActionSwitch, E2E-LOAD, StreamFormer, offline E2E-TAD, and tracking-query literature;
- required six strongest reconstructed baselines, including StreamFormer+MATR and Temporal TrackFormer;
- required a 500-word strongest rejection before method revision;
- required route comparison, exact end-to-end terminology, feature-level P0, raw-video adaptation isolation, cost gates, kill criteria, and `GO / REVISE / NO-GO`.

Artifact:

- [`../PRO_PETAL_ONTAD_DEEP_REVIEW_PROMPT_20260712.md`](../PRO_PETAL_ONTAD_DEEP_REVIEW_PROMPT_20260712.md)

Current decision:

> Send the prompt and absorb the Pro verdict before PETAL raw-video implementation or formal training.

### T19: Pro Returns REVISE; Full PETAL Is Demoted to a Stage-1 Kill Test

User requested:

> Archive and absorb the complete Pro response, state whether every point is accepted, verify it independently, and begin implementation plus experiment deployment.

Pro verdict:

- `REVISE`, not `GO`;
- Full PETAL can be reconstructed from causal streaming backbones, direct On-TAD methods, and tracking-query literature;
- Temporal TrackFormer must be the novelty-killer baseline;
- raw-video training must not begin before a cheap matched-feature pilot;
- current PCEH semantics and GT/runtime boundaries are unsafe for the new route.

Independent disposition:

- accepted the central demotion and most protocol findings;
- rejected the literal claim that full-trajectory training labels are automatically inference leakage;
- rejected endpoint-only class supervision as the main setting;
- treated the pointer and `2 mAP / 20%` rules as hypotheses/resource gates rather than established scientific truths;
- required five-seed confirmation if the three-seed kill test survives.

Implementation started and verified:

- archived the source byte-identically and recorded SHA-256 `74A5A88D0F0BFE389C974290FC6B118B834B13FE59E5D22231985F7018256D95`;
- implemented prefix-observable schedules, a chronological mmap cache dataset, shared persistent event-set head, online detector, feature extractor, matched configs, smoke launcher, instance audit, and result gate;
- replaced greedy assignment with Hungarian matching and rejected unexpected forward kwargs as potential GT taint;
- verified 30 local CPU-safe tests and 43 combined remote tests, plus config-built train/inference and ledger-summary integration;
- audited THUMOS14 maximum concurrency as two on both splits and reduced all variants from 16 to four slots;
- kept raw-video training blocked and prepared cache-first Slurm deployment.

Artifacts:

- [`../PRO_PETAL_DEEP_REVIEW_20260712.md`](../PRO_PETAL_DEEP_REVIEW_20260712.md)
- [`../PRO_PETAL_DEEP_REVIEW_ABSORPTION_20260712.md`](../PRO_PETAL_DEEP_REVIEW_ABSORPTION_20260712.md)
- [experiments/persistent-feature-kill-test-20260712.md](experiments/persistent-feature-kill-test-20260712.md)
- [decision_register.md#DR-026-Demote-Full-PETAL-and-Run-a-Matched-Persistent-State-Kill-Test](decision_register.md)

Current decision:

> The only approved experiment is the frozen-feature FRESH/TTF/PES falsification study. A positive result retains a mechanism for further review; it does not establish novelty and does not automatically authorize raw-video training.

### T20: Cache and Smoke Pass; Pro Must Audit Identifiability Before Pilot

User requested:

> Publish the latest checked code to GitHub and produce one complete repository-addressed Prompt for Pro code review, scientific audit, and discussion.

Completed evidence:

- feature-cache job `1159510` completed in `00:52:07` with exit `0:0`;
- all 411 feature arrays and sidecars passed shape, dtype, source, annotation, and SHA256 contract checks;
- the cache contains 320,205 768-dimensional `float16` tokens, with 2,479 training chunks and 2,719 validation chunks at chunk size 64;
- GPU smoke job `1159843` completed in `00:03:46` with exit `0:0`;
- FRESH, Temporal TrackFormer, and PES each passed four chronological train chunks with required gradients and parameter updates and zero slot exhaustion;
- PES passed four chronological inference chunks, and the smoke job test bundle reported `43 passed in 36.73s`.

Engineering correction:

- the original post-run checker used bare `python`, which resolved to Python 2 on the remote login shell;
- the generated smoke artifacts were valid, but the first inspection command could not parse them;
- commit `d6b0bca3f146046f9f6cb938f26561b32df105ed` defaults to `python3`, permits `PYTHON_BIN` override, adds a launch-contract regression test, and was revalidated against job `1159843`.

Scientific interpretation:

- smoke proves execution, gradient flow, parameter updates, causal path coverage, and basic ledger generation only;
- it does not show PES effectiveness, novelty, fair mechanism attribution, or statistical support;
- the current FRESH-to-TTF and TTF-to-PES changes may confound assignment, persistence, start representation, and endpoint objective;
- the three-seed pilot remains intentionally unsubmitted until Pro decides whether the registered comparison is identifiable or needs a smaller controlled bridge set.

Artifact:

- [`../PRO_PES_STAGE1_CODE_SCIENCE_DISCUSSION_PROMPT_20260712.md`](../PRO_PES_STAGE1_CODE_SCIENCE_DISCUSSION_PROMPT_20260712.md)

Current decision:

> Use the public review anchor plus the complete Prompt for a two-round Pro audit. Keep raw-video training blocked and do not treat any smoke evidence as a paper result.

### T21: Full Q2 Line Audit Finds a Real Launch P0 and Revokes Profile Permission

User provided a 1,379-line Pro review of immutable commit
`6d88610da34a695e07d29c5e08b50fb57d2aa5e9` and requested complete
archival, absorption, and an independent decision on whether every recommendation
should be accepted.

Pro verdict:

- `RESEARCH_VERDICT=REVISE`;
- `GENUINE_ONTAD=UNPROVEN`;
- `END_TO_END_SCOPE=CACHED_FEATURE_TEMPORAL`;
- `NOVELTY_VERDICT=RECONSTRUCTION`;
- `PROFILE=BLOCK`, `FORMAL_TRAINING=BLOCK`;
- next step `FIX_BEFORE_PROFILE`.

New blocking evidence:

- the launch ticket freezes exact runtime overrides before the submit helper
  creates its timestamped run directory;
- the helper later injects `work_dir=${RUN_DIR}/work`;
- the pre-CUDA validator requires exact ticket/runtime identity equality;
- the real submit-shell composition is not covered by the passing B0 matrix.

Independent disposition:

- accepted `P0-LAUNCH-WORKDIR` and revoked the previous profile permission;
- accepted raw-feature provenance, packet-clock, reporting-population, and
  complete one-factor trace as scientific/formal gates, with narrower scope;
- rejected DDP consensus and resume continuation as current P1 requirements
  because Q2 is locked to one process/one GPU and formal resume is prohibited;
- retained the package-level reconstruction concern, while treating fixed
  post-birth binding as an unproven marginal hypothesis rather than a settled
  negative result;
- rejected running the review's entire baseline/ablation inventory before a
  minimal matched kill test and rejected arbitrary effect thresholds without a
  power or minimum-meaningful-effect analysis.

Artifacts:

- [`../PRO_FULL_PETAL_CODE_SCIENCE_REVIEW_20260715.md`](../PRO_FULL_PETAL_CODE_SCIENCE_REVIEW_20260715.md), SHA-256
  `C256F68ADD9B3316D90D76557357550054C9B9B75D8BF85F91D11471E422CA49`;
- [`../PRO_FULL_PETAL_CODE_SCIENCE_REVIEW_ABSORPTION_20260715.md`](../PRO_FULL_PETAL_CODE_SCIENCE_REVIEW_ABSORPTION_20260715.md).

Current decision:

> Do not profile or train. First close the launch-workdir P0 with a real
> submit-shell integration test, then close the scientific evidence contracts,
> freeze a new commit, regenerate B0, and obtain a fresh independent PASS.

### T22: Round 1 Proves Current Q2 Is Not CRS-EPS

The focused Pro discussion reviewed public implementation anchor `f4ea53e` and
returned `REVISE-BEFORE-IMPLEMENTATION`.

Code-grounded correction:

- Q2 enumerates every cached token in every selected video;
- every token enters the sequential temporal/loss path;
- numerical state is carried but detached across 64-token chunks;
- gradients are accumulated and normalized to one per-video mean before one
  optimizer event at video end;
- event anchors, bounded sampled episodes, inclusion probabilities, weighting,
  ESS, and sampled/full audits are absent;
- therefore `CRS_EPS_IMPLEMENTED=NO`.

The full route remains useful only as an exhaustive reference for its exact
video-uniform, cached-token, chunk-detached objective. The provisional direction
is CRS-EPS main training plus a tiny full-stream state/loss/gradient gold audit
and complete chronological evaluation, but Round 2 must freeze the protocol.

The recommended author response selects video-uniform target risk, dynamic
replay to every active instance's earliest observable birth, a 10 GPU-hour cap
for all future pre-Stage-2 work, a multi-denominator profile, one-token formal
serving, and immediate route kill if fixed binding lacks identity-linked gain.
Extractor provenance, 211-versus-213 identities, and second-dataset access stay
explicitly unknown.

Artifacts:

- [`../PRO_FULL_PETAL_CRS_EPS_Q2_ROUND1_REVIEW_20260715.md`](../PRO_FULL_PETAL_CRS_EPS_Q2_ROUND1_REVIEW_20260715.md);
- [`../PRO_FULL_PETAL_CRS_EPS_Q2_ROUND1_ABSORPTION_20260715.md`](../PRO_FULL_PETAL_CRS_EPS_Q2_ROUND1_ABSORPTION_20260715.md);
- [`../PRO_FULL_PETAL_CRS_EPS_Q2_ROUND1_AUTHOR_RESPONSE_DRAFT_20260715.md`](../PRO_FULL_PETAL_CRS_EPS_Q2_ROUND1_AUTHOR_RESPONSE_DRAFT_20260715.md).

Current decision:

> Send Q1-Q12 for Round 2, fix the launch P0 without GPU work, and do not
> implement CRS-EPS until target risk, state replay, weighting, gold audit,
> profile, metrics, and kill contracts are frozen.

### T23: Round 2 Freezes an Amended Hybrid Protocol

The 2,947-line Round-2 response returned `GO-HYBRID-PROTOCOL`. This is a
protocol-level permission to implement and falsify CRS-EPS, not permission to
profile, train, enter raw-video Stage 2, or claim Full PETAL novelty.

Accepted core:

- current Q2 remains complete-video cached-feature truncated-BPTT and is not
  CRS-EPS;
- preserve video-uniform per-video decision-time target risk;
- use uncapped HH/IPW over repeated episode exposures with a 0.40 uniform floor
  and derived `w <= 2.5` invariant;
- treat dynamic earliest-birth replay as a causal surrogate and video-start
  replay as gold;
- preserve global 64-token gradient boundaries;
- separate local binding direct effects from longitudinal policy-mediated
  divergence;
- replace optimizer-event throughput with multi-denominator cost and ESS;
- require feature provenance, five clocks, one-token formal evaluation, and
  staged reconstruction baselines;
- keep every GPU stage blocked behind P0, B0, and fresh independent review.

Independent amendments:

- matched stochasticity is mandatory, but token-addressed RNG is not the only
  acceptable implementation;
- exact duplicate/fragmentation evaluator definitions become a new P1;
- model-caused slot exhaustion is a scientific kill rather than a rerunnable
  invalid run;
- zero LoRA-by-binding interaction is not itself a falsifier if both main
  effects survive;
- one seed is only a resource gate;
- FineAction requires a same-class identity audit;
- episodes within a video group share one gradient update but never residual
  runtime state or untracked mutable buffers;
- the review's `OAT_PRIMARY_ARTIFACT=UNRESOLVED` statement is rejected: OAT is
  a verified ECCV 2022 paper with an official ECVA PDF and DOI.

Artifacts:

- [`../PRO_FULL_PETAL_CRS_EPS_Q2_ROUND2_REVIEW_20260715.md`](../PRO_FULL_PETAL_CRS_EPS_Q2_ROUND2_REVIEW_20260715.md), SHA-256
  `C7FBBC573D18DA7234CE787167AD1F07438101C3DC07D90899B09BE71CA7CEB3`;
- [`../PRO_FULL_PETAL_CRS_EPS_Q2_ROUND2_ABSORPTION_20260715.md`](../PRO_FULL_PETAL_CRS_EPS_Q2_ROUND2_ABSORPTION_20260715.md).

Current decision:

> Fix `P0-LAUNCH-WORKDIR` and its deterministic fake-`sbatch` integration
> test, then implement only the CPU CRS-EPS manifest/probability path. Do not
> spend another GPU-hour before a clean commit, complete B0, and fresh PASS.
