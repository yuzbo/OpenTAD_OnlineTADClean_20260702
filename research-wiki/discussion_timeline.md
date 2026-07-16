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

### T24: Round-2 Protocol Reaches a CPU-Auditable Implementation

The implementation branch now closes the P0 launcher identity, deterministic
CRS-EPS sampling math, independent replay lifecycle, HH/IPW optimizer
normalization, exact identity metrics, multi-denominator profiling, and the G0
four-arm audit path. A real `Dataset -> DataLoader -> detector -> optimizer ->
transaction commit` CPU test demonstrates one update per video group without
double division. Slot exhaustion and mutable-buffer leakage fail closed.

This does not change the scientific verdict. The first ordinary full-suite run
found only environment/registry failures outside the B0 harness; the locked
isolated runner passed its pre-B0 run after removing the Windows shell-test
skip. The exact source-hash B0 still has to be regenerated on a clean commit,
then the same independent reviewer must return PASS before any profile.

Current decision:

> Freeze and commit the implementation, regenerate full B0 evidence, and send
> the exact commit plus B0 artifact to the locked reviewer. Do not run G0 on
> real data, profile, or formal training while that gate is pending.

### T25: First Exact-Commit Review Blocks Four Enforceability Gaps

Commit `0731f070e7d7f982a1df20cfaff801b21bed96fe` passed signed B0 with
`548/548` tests, but the same independent reviewer returned `REVISE`,
`PROFILE=BLOCK`, and `FORMAL_TRAINING=BLOCK`. The four accepted findings were:

- G0 was documented but not mandatory in profile launch tickets;
- exact `0..M-1` draw membership/order was not optimizer-bound;
- failed groups did not restore mutated model buffers;
- G0 provenance and outcome-blind margins were caller claims rather than a
  signed, byte-reconciled evidence chain.

The correction route upgrades the manifest with a sampling-population hash and
per-video episode-sequence hash, enforces exact draw order, restores buffers and
staged state on every pre-boundary failure, preregisters signed selection and
margins, signs the terminal G0 audit, and makes G0 PASS a profile-ticket field
that is re-evaluated before CUDA/DDP.

Current decision:

> Complete the four corrections and their adversarial tests, then freeze a new
> commit, regenerate full B0, and return to the same reviewer. The prior B0 PASS
> does not authorize profile because its associated review is REVISE.

Source:

- [`../PRO_FULL_PETAL_CRS_EPS_IMPLEMENTATION_REVIEW_20260716.md`](../PRO_FULL_PETAL_CRS_EPS_IMPLEMENTATION_REVIEW_20260716.md).

### T26: Second Exact-Commit Review Finds Six Remaining Evidence Gaps

The corrected commit `cd601ce95fdd16ecfdd17f9a5d93b33578691133` passed the
locked local and N16R4 matrix at `561/561`, but the same reviewer again returned
`REVISE / PROFILE=BLOCK / FORMAL=BLOCK`. The review found no protocol
violation, but identified six enforceability gaps:

- a self-consistent replacement M-draw group was not compared with the
  published epoch manifest;
- second-draw loader/control exceptions could bypass group rollback;
- training trace identity still described chronological packets rather than
  CRS-EPS draws;
- feature and G0 checkpoint hashes could be computed from bytes different from
  the bytes later consumed;
- Slurm resource fields and `job.sbatch` publication admitted injection or
  replacement races;
- the claimed Linux/N16R4 matrix was not a signed leaf of B0.

The accepted correction binds every draw to a deep-copied, reproduced epoch
manifest; wraps data loading and control parsing in the group transaction;
derives a stable CRS order identity from the manifest sampling contract;
verifies and consumes feature/checkpoint bytes through one stable read; submits
the exact exclusively published in-memory Slurm bytes through fixed
`/usr/bin/sbatch`; and makes a signed Linux leaf mandatory in B0 v3. The
pre-freeze isolated local matrix passes `578/578`. This remains engineering
evidence only: the target Linux leaf, signed root B0, and same-reviewer PASS do
not yet exist for the correction commit.

Current decision:

> Freeze the correction, generate the signed N16R4/Linux B0 leaf, build the
> local signed B0 root, and return both to the same reviewer. Do not issue G0,
> profile, or formal-training permission before exact `PASS / PROFILE=ALLOW`.

### T27: First Real G0 Attempt Exposes a JSON Round-Trip Blocker

Commit `ee439e1647567b4d1a0fec3f5e50498a34588bb8` passed the locked local and
N16R4/Linux B0 matrix at `578/578`; the same reviewer closed all six prior
findings and returned `PASS / PROFILE=ALLOW / FORMAL=BLOCK / NEXT_GATE=G0`.
Before any model replay outcome was executed, G0 sample selection covered four
video-length quartiles and all preregistered metadata stressors. The
preregistration then failed closed because the serialized epoch manifest could
not reproduce exactly.

The failure is a code defect: `InstanceTimeline.active_bins` remained a tuple
inside the in-memory manifest, while JSON persistence converted it to a list.
The builder's in-memory self-check therefore passed, but the first persisted
read failed exact equality. The correction emits JSON-native lists at manifest
construction and adds an explicit serialize/read/validate regression test. The
locked isolated runner now passes `579/579` before the replacement commit.

No replay loss, gradient, state, or model-quality output was observed before
the failure. The metadata-only sample policy and conservative margins remain
outcome-blind, but every artifact tied to `ee439e1` is superseded for launch
authorization because the source commit changed.

Current decision:

> Freeze the serialization correction, regenerate the target-Linux leaf and
> signed B0 root, and obtain a fresh PASS from the same reviewer. Only then
> rebuild and run G0; profile and formal training remain blocked.

### T28: Nonzero G0 Draw Exposes Singleton Runtime-Index Drift

The serialization correction commit
`cde619639a14a7cbb5f8cd4603ed8fcb66933d44` passed local and Linux B0 at
`579/579`; the same reviewer returned PASS with no P0/P1/P2 finding or protocol
violation. A fresh metadata-only G0 selection and margin preregistration then
succeeded. The first real audit execution stopped before producing any signed
row or `audit.json` because a selected manifest draw with index one failed its
episode-payload hash at model entry.

The audit helper correctly turns each selected draw into a singleton runtime
group, whose runtime draw index must be zero. It previously hashed the copied
payload while it still carried the original manifest index and only later
overrode the runtime index in `_build_sample`. Draw-zero fixtures therefore
passed while any selected nonzero draw failed closed. The correction sets the
singleton runtime index before computing the audit payload and sequence hashes;
a new regression uses a nonzero manifest draw and verifies both hashes.

No valid loss, gradient, runtime-state, gate, quality, profile, or GPU result
was emitted. The failed evidence bundle remains immutable and is superseded,
not edited.

Current decision:

> Freeze the singleton-index correction and restart B0 plus same-reviewer
> acceptance. Rebuild all G0 preregistration artifacts under the replacement
> commit before another audit attempt. Profile and formal training remain
> blocked.

### T29: G0 Reveals Sentinel-Unsafe Runtime Similarity Measurement

Commit `5639fa85f55ac03c12761977ef82b25ea4aabfa6` passed local/Linux B0 at
`579/579` and the same reviewer again returned PASS. The rebuilt G0 reused the
same sample and threshold input bytes as the superseded attempt. All 16 audit
controls passed payload and sequence preflight, but terminal signing stopped
because `runtime_continuous_comparison.cosine` was non-finite.

Read-only diagnosis showed finite losses and gradients. The non-finite runtime
norm came from the model's intentional `start_frames=NaN` sentinel for free
slots, which the audit had concatenated directly into its continuous vector.
The diagnostic also found a float32 identical-vector cosine of `1.000055`, so
the metric implementation lacked stable precision. The correction records the
start-state NaN mask as discrete state, maps only that legal sentinel to zero
for continuous comparison, rejects other non-finite runtime tensors, and uses
float64 norm/dot computation with cosine clamped to `[-1,1]`.

Unsigned diagnostics suggest that several preselected dynamic-replay cases may
violate the preregistered absolute gradient-fidelity margin. This is not a
terminal G0 result, but it must not be hidden: the next exact signed run may
legitimately return KILL. Samples, margins, checkpoint seed, and model logic are
frozen and will not be relaxed in response.

Current decision:

> Treat sentinel handling and numerical precision as audit-measurement fixes,
> not method tuning. Freeze them, restart B0/review, and rerun the unchanged G0
> contract to a signed PASS or KILL. A KILL blocks profile.

### T30: Same-Reviewer Audit Rejects Unconstrained Sentinel Canonicalization

Commit `5d27fcad36062a6496f8c330ed15fba623a9667f` passed the complete local and
N16R4/Linux B0 matrix at `580/580`, and both signed roots were independently
revalidated. The same locked reviewer nevertheless returned
`REVISE / SENTINEL_AWARE_METRIC=FAIL / PROFILE=BLOCK / NEXT_GATE=FIX`.

The reviewer reproduced a lifecycle-invalid runtime in which an ACTIVE slot
carried `start_state=NaN`. The audit recorded the NaN mask, canonicalized the
value to zero, and reported perfect runtime agreement because it never checked
whether sentinel placement was legal. The converse FREE or REFRACTORY slot
with a finite start was also accepted. This could let two equally corrupt arms
produce a false G0 fidelity PASS.

The correction now requires one-dimensional, shape-aligned lifecycle tensors,
an integer slot status drawn from FREE, ACTIVE, and REFRACTORY, and the exact
invariant `isnan(start_state) == (slot_status != ACTIVE)` before any
canonicalization. ACTIVE starts must therefore be finite; FREE and REFRACTORY
starts must be NaN. Focused tests cover all legal states, each mismatch, an
invalid status, shape drift, and rejection at the paired-audit entry point.

Current decision:

> Treat NaN canonicalization as conditional on a proven lifecycle invariant.
> Freeze the fail-closed correction, regenerate the exhaustive manifest, and
> restart Linux B0, signed root B0, and same-reviewer acceptance. G0 samples,
> margins, checkpoint, and model method remain unchanged; profile and formal
> training stay blocked.

### T31: Same Reviewer Finds Post-Diagnostic Checkpoint Choice Freedom

The lifecycle correction commit
`29bc0aee90faef61f16e32130b8c1368fc78e755` passed signed local and
N16R4/Linux B0 at `586/586`. The same reviewer closed the sentinel finding but
returned `REVISE / PROFILE=BLOCK / NEXT_GATE=FIX` because the G0 checkpoint was
not part of outcome-blind preregistration.

The v2 selection and margin signatures fixed the commit, data, manifest,
samples, and thresholds. The audit runner accepted `--checkpoint` only at
execution time and wrote its digest into the terminal audit. Consequently, one
already-signed selection/margin pair could be executed with two different
structure-compatible checkpoints; both terminal audits would truthfully name
the bytes they used, but neither could prove which checkpoint had been frozen
before the earlier unsigned diagnostic.

The correction replaces G0 selection, margins, and audit schemas with v3.
Selection now signs the checkpoint's bundle-relative path, SHA-256, byte size,
state key, and deterministic generation identity. Preregistration rebuilds the
model under the immutable manifest seed and exact config and requires every
state tensor to match. The runner verifies the signed bytes before model
construction, and launch validation reopens and rehashes the same contained
checkpoint before accepting a G0 PASS.

Current decision:

> A terminal audit that merely reports its checkpoint is insufficient after
> any prior diagnostic. Freeze checkpoint identity before execution, make
> margins bind that signed selection, and reject replacement bytes, state-key
> drift, seed drift, or deterministic-state drift before any trace. Restart the
> complete B0/review chain; G0, profile, and formal training remain blocked.

### T32: Signed G0 Kills the Current CRS-EPS Training Surrogate

Commit `70df86ea3d38d70c658ae0ee9e04245d57b834d4` completed the replacement
checkpoint-preregistration chain. Local and N16R4/Linux B0 passed `588/588`,
the signed B0 roots independently validated, and the same locked reviewer
returned exact `PASS / PROFILE=ALLOW / FORMAL=BLOCK / NEXT_GATE=G0`.

The real-data CPU-only G0 then executed against the preregistered four samples,
unchanged margins, exact deterministic checkpoint bytes, persisted manifest,
and fixed CRS-EPS config. The signed terminal audit returned `KILL`: three
samples produced fourteen absolute-fidelity violations. The worst dynamic
relative loss error was `0.8021221151`, minimum gradient cosine
`0.2830554455`, minimum gradient sign agreement `0.5445116162`, and minimum
continuous runtime-state cosine `0.1027812195`. Runtime discrete state also
diverged in all three failed dynamic cases.

The result still shows that dynamic replay beats reset and, on average, fixed
replay in gradient cosine. That relative ordering does not override the frozen
absolute fidelity requirements. Two failed samples made `dynamic_birth` and
`fixed_192` numerically identical on the reported fidelity metrics, while the
one case with a dynamic advantage remained far below the absolute margins.

The signed audit was independently verified on the target Linux platform and
its gate recomputed to the same `KILL`. Local signature and gate recomputation
also passed. The launch validator rejects this terminal artifact with
`CRS-EPS G0 audit has not reached PASS`, so no profile or formal job was
submitted.

Current decision:

> Freeze the result as a valid negative experiment. Do not tune the gate or
> reuse the exposed samples as confirmation. First determine whether HH/IPW
> event sampling is structurally unable to recover omitted state/gradient
> paths or whether a bounded implementation correction exists. Request a
> Pro mechanism and route adjudication before any new implementation.

### T33: Pro Round 1 Confirms the Current Structural Kill but Overstates Q2 Readiness

The 1,417-line Pro route review was archived byte-identically as
`PRO_CRS_EPS_G0_KILL_ROUTE_REVIEW_ROUND1_20260716.md`, SHA-256
`D0AB7B51961E61F9DBD45192743AC61B5522646F2FA1027B96489C8010ADC46F`.
It independently classified the frozen `70df86e` G0 as a valid terminal KILL,
the current empty-state `dynamic_birth` surrogate as structurally invalid,
profile/formal training as blocked, and state-faithful cost control as the
only replacement family worth discussing.

Local inspection of the raw signed bundle resolved the review's U1-U8 more
sharply than its repository-only evidence. Two failed samples make
`dynamic_birth` exactly equal to `fixed_192` in replay range. The single
selected dynamic-extension sample adds only eight bins and still diverges in
loss and final runtime state. In the longest sample, canonical lifecycle and
loss ownership diverge at global bin 209, endpoint-slot ownership at 216, and
birth assignment at 352. These facts strengthen the state/lifecycle diagnosis.

The local audit also found gold-control slot exhaustion counts of one and two
in two selected samples. This does not rescue CRS-EPS, but it prevents an
unqualified promotion of chronological Q2 to primary: Q2 remains the
gold/reference candidate until a separate capacity/lifecycle gate and matched
cost profile pass. The structural-kill label is correspondingly narrowed to
the current empty-state dynamic replay rather than all event-centric training.

Current decision:

> Spend zero GPU hours. Publish the postmortem and U1-U8 answers, then return
> to the same Pro reviewer for Round 2 over full causal forward plus selective
> backward, exact chronological recomputation, and explicit state checkpoints.
> Select at most one new route. A new profile remains forbidden until a clean
> commit, B0, independent review, unseen holdout G0, and terminal PASS.

### T34: Round 2 Selects CSFSB but Authorizes Capacity Audit First

The 3,879-line Round-2 attachment was archived byte-identically as
`PRO_CRS_EPS_G0_KILL_ROUTE_REVIEW_ROUND2_20260716.md`, SHA-256
`2BE3C21E951F68F397CADA468B66820BA08B3FF2F0FFD0A4E59580D784C989EE`.
The source contains two complete N-Y answers; both return the same machine
verdict, and the later, more detailed answer is treated as normative without
discarding the earlier text.

Round 2 retained only R1, Chronological State-Faithful Selective Backward
(CSFSB). Unlike the killed empty-state replay, R1 executes every cached token
in chronological order and advances all numerical, runtime, and canonical
state. The outcome-blind event proposal now selects only per-bin loss and
backward contributions using an HH coefficient. It preserves the existing
global 64-bin TBPTT estimand and one optimizer event per video. Exact
recomputation was dominated on workload, runtime checkpoints were rejected as
parameter-stale, and chronological-only remains an honest fallback.

The review did not clear Q2 capacity. Complete gold controls had exhaustion
counts one and two under K=4, random-init logits, 0.5 thresholds, refractory
two, and runtime-free-constrained birth assignment. Because annotation maximum
concurrency does not include false ACTIVE or REFRACTORY occupancy, the first
authorized commit is a zero-GPU capacity/lifecycle audit only. It must test the
full fit core, seeds 705/706/707, same-logits lifecycle counterfactuals, and
fail-closed chronological behavior, then freeze one shared fixed/rematch
contract or terminate Q2/R1.

Local code inspection confirms that current Q2 averages per-step loss over
valid bins and that the current head does not use lifecycle tensors to produce
query/memory predictions. Thus the proposed estimator targets the implemented
objective and same-logits controller replay is valid for this diagnostic. Both
facts are implementation-specific and must be revalidated after relevant code
changes.

Current decision:

> `R1=CONDITIONAL_GO`, not method approval. Capacity audit first, then R1 code,
> CPU tests, local/target-Linux B0, same-reviewer PASS, and a new unseen
> family-wise CPU G0. Profile, formal training, and raw-video Stage 2 remain
> blocked; GPU hours before new G0 PASS remain zero.
