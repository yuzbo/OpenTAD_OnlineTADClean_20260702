# Pro Review Absorption: 2026-07-09

Source record:

- Full external review: `PRO_REVIEW_20260709.md`
- Original source: private Codex attachment, not committed; integrity is tracked by the SHA256 below.
- Attachment SHA256: `BF6E24728577976F2161F335DF571D2A054F1915F8622E4470970F2B3FA097A8`
- Attachment line count: 757
- Reviewed public branch according to reviewer: `codex/online-tad-clean-20260702`
- Local commit when absorbed: `4e222ccf4de1e989c6faf83031772c552be7af59`
- Review verdict: `HOLD`
- Reviewer limitation: the reviewer could inspect the public GitHub branch, not this local working tree.

## Absorbed Verdict

The review should be treated as a method-level and claim-boundary review, not just a code review.

The current repository is not `GO` for a CCF-A full-paper claim. It is also not a total `NO-GO`: it has a serious raw-frame / selected-only / causal-stride / single-rank streaming-safe Online TAD validation skeleton with ledger-aware emitted-row evaluation.

Allowed current wording:

> A selected-only causal-stride raw-frame Online TAD validation skeleton with single-rank emission-ledger evaluation.

Disallowed current wording:

> A completed adaptive, learned-selector, paper-ready, DDP-safe, full online-censored Online TAD method.

The absorbed paper direction is:

> Prefix-observed online-censored localization with bounded-delay emission hazard learning.

The review explicitly rejects making `adaptive selected-frame Online TAD` the main claim until a genuine content-dependent or learned selector is implemented and validated.

## Critical Literature Update: OZ-TAL Threat

Source:

- OZ-TAL / VFEAL paper page: https://arxiv.org/html/2605.09976v1
- Search date: 2026-07-09
- User-supplied framing: `Online / Streaming Zero-Shot TAL` for unseen action detection in real-time video streams.

This is a major competitive threat if the project is framed as:

> VLM/pretrained-model-based online TAL/TAD for unseen or open-vocabulary actions.

OZ-TAL already defines Online Zero-Shot Temporal Action Localization with disjoint train/test action classes, no access to future frames during inference, and no post-hoc modification of emitted results. Its VFEAL route uses an off-the-shelf VLM, visual representation enhancement, memory-guided feature enhancement, background-aware classification, and an online action-span/state-machine style predictor. It also establishes a benchmark-style evaluation on THUMOS14 and ActivityNet-1.3.

Blocked or highly dangerous wording after this update:

- first VLM/pretrained-model route for online open-vocabulary or unseen-action TAL,
- using off-the-shelf VLMs to make online TAL zero-shot,
- memory-enhanced VLM online action span detection as the main novelty,
- background-aware bias mitigation for online zero-shot TAL as the main novelty,
- online unseen-action detection as the paper's core claim without a very clear distinction from OZ-TAL.

The opportunity left open by OZ-TAL is narrower but cleaner:

> OZ-TAL shows that off-the-shelf VLMs can support training-free online zero-shot TAL, but it does not solve task-aligned, trainable, low-latency, end-to-end Online TAD pretraining/fine-tuning under prefix-observed event-time supervision.

Defensible positioning should therefore shift away from `Online Zero-Shot TAL` and toward:

> Trainable low-latency Online TAD pretraining/fine-tuning with prefix-observed boundary, completion-state, and bounded-delay emission objectives.

Important distinction to preserve:

- OZ-TAL strong zone: training-free VLM, unseen/open-vocabulary classes, online span aggregation, memory feature enhancement, background bias mitigation.
- Our safer zone: trainable or parameter-efficient end-to-end Online TAD adaptation, causal/cached raw-frame inference, prefix-observed censored supervision, event-time/end hazard, bounded-delay emission hazard, and strict low-latency auditing.

Reviewer-facing implication:

If this project uses VLMs, unseen classes, or open-vocabulary language prompts, OZ-TAL must be treated as a core baseline or at least a primary related-work competitor. If the project does not target zero-shot/open-vocabulary TAL, the paper must say so explicitly and avoid drifting into OZ-TAL's claim territory.

## Local Verification Against Current Commit

Some P0 issues discussed in the review were already addressed in the local tree before this absorption:

1. `OnlineEmitter.step` now treats `latency_frames` as a maximum allowed delay and updates `state.last_emit_frame` after every step, including no-emission steps.
2. `CausalFrameSelector` fails closed on all-invalid masks.
3. `OnlineMAP._import_prediction` filters rows before computing both AP rows and online statistics.
4. `MATRHead` fail-closes irregular selected-axis metadata to `batch_size=1`.
5. `OnlineSigLIPFrameEncoder` has a selected-only runtime count invariant, covered by a monkeypatched tiny-backend test.
6. Streaming-safe evaluation rejects multi-rank state splitting.

Existing relevant coverage:

- `tests/test_p0_review_hardening.py`
- `tests/test_online_emission_protocol.py`
- `tests/test_final_online_tad_contracts.py`

Still worth adding as explicit behavior coverage:

- no-emission monotonic state update test,
- future-frame perturbation test for the full P1/P2 chain,
- window-bounded irregular-time decode test,
- dense-grid no-double-irregular-decode test,
- explicit proposal-axis contract test for future selected-axis paths.

## Claim Boundary

### Safe Claims

- The repo contains a selected-only causal-stride raw-frame Online TAD skeleton.
- The streaming-safe evaluator records emitted detections with `emit_frame`, `source_frame`, `start_frame`, `end_frame`, `latency_sec`, `stream_key`, and no-future summary fields.
- Current selected-only encoding can be runtime-tested with a tiny backend to prove that the heavy encoder sees selected frames only.
- Current streaming-safe evaluation is single-rank by contract.
- Current irregular selected-axis handling is a validation path with a `batch_size=1` fail-fast contract, not a full native sparse-time head.

### Blocked Claims

- paper-ready Online TAD,
- true adaptive selected-frame Online TAD,
- learned budgeted selector,
- full online-censored training,
- native irregular sparse TAD head,
- DDP-safe streaming evaluation,
- full end-to-end visual-tower training,
- SOTA or competitive full mAP without full remote experiments,
- compute reduction while preserving boundary quality before same-budget baselines pass.

## Core Review Absorption

The review reframes the project around a sharper problem:

> Offline TAD supervision silently teaches future endpoints, while strict online TAD must decide from prefix-observed evidence and emit predictions under bounded delay.

The current code partially approaches this through endpoint regression masking and emission-ledger filtering, but the review says this is not yet a complete training protocol. The missing mechanism is an explicit prefix-observed target builder that consistently handles classification, actionness, start/end boundary, regression, and emission targets under observed-prefix semantics.

## Question List To Resolve Before Method Lock

1. Is the primary objective OnlineAP/latency, offline mAP, compute budget, or strict online correctness?
2. Is THUMOS14 enough for the first paper loop, or is a second dataset required?
3. Is the accepted main input route raw frames with frozen SigLIP/SigLIP2, feature-level CausalTAD, or trainable visual adapter?
4. Is offline teacher distillation allowed during training only?
5. Is testing absolutely teacher/cache/future/offline-NMS free?
6. Is compute budget measured by encoded frames, FLOPs, FPS, memory, or wall-clock?
7. Is single-rank streaming audit acceptable for this paper, or must DDP streaming state be solved?
8. Is the user willing to make online-censored supervision the main contribution instead of adaptive selection?
9. Can fixed causal stride remain the validated skeleton while learned adaptive selection is future work?
10. If results are weak, should the work downgrade to a rigorous online-protocol baseline / negative-result report?

## Claim Map

| Claim | Current support | Required before paper claim | Failure criterion |
|---|---|---|---|
| Strict streaming-safe Online TAD protocol | Ledger rows, OnlineMAP, no-future row audit, single-rank contract | Whole-chain perturb tests and protocol smoke on actual route | Any future-frame perturb changes past emissions |
| Online-censored localization | Partial regression endpoint masking | Prefix-observed target builder for cls/actionness/reg/start/end/emit | No improvement or leakage remains in non-regression targets |
| Bounded-delay emission hazard | Emit branch and latency ledger exist | Hazard target, false-early/late penalties, latency/AP trade-off | Emit branch only changes scores without latency or AP benefit |
| Budgeted frame/token selection | selected-only causal stride exists | Same-budget dense/random/uniform/motion/learned comparisons | Uniform/random matches or beats claimed selector |
| Irregular-time decoding | Metadata path and dense-grid mapping exist | Window-bounded native sparse-time decode and tests | Decode depends on whole-video future duration |
| Causal motion branch | P2 motion branch candidate exists | Boundary-recall and latency ablations | Motion branch is trivial or hurts AP/latency |
| Offline-to-online distillation | Not a current safe claim | Train-only teacher, no inference cache, leakage audit | Any test-time teacher/cache dependency |

## Candidate Route Scores

| Route | Problem fidelity | Method specificity | Contribution quality | Feasibility | Validation clarity | CCF-A readiness | Overclaim risk |
|---|---:|---:|---:|---:|---:|---:|---:|
| Online-censored boundary / emission hazard | 9 | 8 | 8 | 6 | 8 | 7 | 4 |
| Budgeted causal evidence selection + irregular-time decode | 8 | 7 | 8 | 5 | 7 | 6 | 7 |
| Train-only offline-to-online distillation | 7 | 7 | 6 | 6 | 6 | 5 | 8 |
| Causal motion / boundary evidence branch | 6 | 6 | 4 | 8 | 7 | 4 | 5 |

Absorbed recommendation:

1. Make online-censored boundary / emission hazard learning the main route.
2. Treat selected-only causal stride as a validation skeleton and compute baseline.
3. Treat motion branch as supporting evidence only.
4. Treat learned adaptive selection and native irregular sparse-time head as future or second-stage work unless the first-stage evidence is strong.

## Experiment Gate

### P0 Protocol Gate

Must pass before long remote training:

- no raw prediction cache,
- no teacher/cache/future context at test time,
- streaming-safe emission only,
- single-rank streaming state or explicit fail-fast,
- `source_frame <= emit_frame`,
- `end_frame <= emit_frame`,
- latency budget means maximum delay,
- selected-only encoded-frame count test,
- no-emission monotonic state update test,
- future-frame perturbation test.

### P1 Baselines

- dense frozen encoder upper bound,
- fixed uniform causal stride,
- recent-window online baseline,
- random selected-frame baseline,
- motion heuristic selector,
- current selected-only causal-stride MATR skeleton.

### P2 Main Mechanism

Implement and test:

- prefix-observed target builder,
- online-censored classification/actionness/regression/boundary/emit targets,
- bounded-delay emission hazard loss,
- false-early and late/missed emission diagnostics.

### P3 Ablations

- full-GT endpoint training vs online-censored training,
- regression-only censoring vs full prefix-observed targets,
- with/without actionness censoring,
- with/without boundary branch,
- with/without emit branch,
- with/without hazard loss,
- with/without motion branch,
- with/without irregular-time decode if sparse selection is used.

### P4 Metrics

Report:

- standard mAP at IoU thresholds,
- OnlineAP / emitted-row AP,
- latency mean / p50 / p90 / p95,
- no-future violations,
- future source violations,
- false early emission rate,
- boundary recall after endpoint observed,
- encoded frames / FLOPs / FPS / GPU memory,
- selector collapse rate if any selector is trained,
- per-class AP,
- wall-clock training and inference throughput,
- rejected candidate counts by future-end, future-source, and latency timeout.

### P5 Statistical Discipline

- Minimum 3 seeds for main claims.
- Report mean and standard deviation.
- Keep all comparisons under the same budget, same post-processing, same visual encoder, and same split.

## Reviewer-Risk Register

1. Novelty weak: frozen visual encoder + causal projection + MATR-like head + ledger may read as module assembly.
2. Protocol leakage: ledger rows cannot prove the whole computation graph is future-free.
3. Label leakage: current online-censored training is not complete across all target types.
4. Baseline unfairness: same-budget random/uniform/recent-window/motion baselines are mandatory.
5. Ablation insufficiency: emit/actionness/boundary/motion/irregular decode must be isolated.
6. Metric mismatch: offline mAP alone does not support online claims.
7. Overclaiming: `adaptive`, `final`, and `paper-ready` names remain dangerous.
8. Engineering-only contribution: P0 hardening is necessary but not a CCF-A algorithmic contribution.
9. Dataset narrowness: THUMOS14-only evidence may be weak for a broad claim.
10. Reproducibility: remote full training, multi-seed evidence, and config manifests are still missing.
11. OZ-TAL competition: open-vocabulary / unseen-action Online TAL with off-the-shelf VLMs is no longer a clean novelty space; any VLM or zero-shot claim must distinguish against OZ-TAL.

## Updated Working Rules

- Do not rename fixed causal stride as adaptive.
- Do not call the method paper-ready before P0-P4 gates pass.
- Do not claim full online-censored training until all target branches use prefix-observed semantics.
- Do not claim learned selection until the selector is content-dependent and beats same-budget baselines.
- Do not claim DDP-safe streaming evaluation until video-contiguous or centralized online state is implemented.
- Do not claim `Online Zero-Shot TAL`, VLM-based unseen-action online localization, or open-vocabulary streaming TAL novelty without directly accounting for OZ-TAL.
- Treat OZ-TAL as a primary competitor whenever the method uses off-the-shelf VLMs, language prompts, unseen categories, or open-vocabulary evaluation.
- Treat every innovation as unproven until the claim map has supporting experiments.

## Immediate Next Step

Before code expansion, write a short design spec for:

> PrefixObservedTargetBuilder + bounded-delay emission hazard supervision.

The spec must freeze:

- observed-prefix semantics,
- target masks for classification/actionness/regression/start/end/emit,
- hazard label definition,
- false-early / late-emission diagnostics,
- exact baseline and ablation matrix,
- OZ-TAL-aware claim boundary if any VLM / unseen-class / open-vocabulary component remains in scope,
- tests required before remote training.
