# Pro Review Absorption: 2026-07-07

Source record:

- Full external review: `PRO_REVIEW_20260707.md`
- Reviewed public branch: `codex/online-tad-clean-20260702`
- Review verdict: `WARN / HOLD`
- Scope: research direction, task definition, current implementation, paper-level target, experiment plan, and key code sketches.

## Absorbed Verdict

The current project should be described as:

> A windowed, single-rank, streaming-safe raw-frame online TAD prototype with a stronger frozen SigLIP/SigLIP2 route than the current VideoMAE stub route.

It does not yet support the paper-level claim:

> Budgeted Adaptive Online TAD with Causal Emission.

The missing core method pieces remain:

- online-censored localization training,
- adaptive frame/token selection,
- irregular-time AdaTAD/MATR decode,
- full P1 mAP/ledger/latency artifacts,
- whole-pipeline no-future perturb audit.

## Accepted Claim Boundary

Allowed wording:

- windowed streaming-safe raw-frame On-TAD prototype,
- frozen visual tower + trainable causal temporal adapter/head baseline,
- single-rank emission ledger protocol prototype,
- P1-full-60 configuration exists and is ready to run, pending actual results.

Disallowed wording until verified:

- paper-ready THUMOS result,
- true continuous per-frame streaming agent,
- full visual end-to-end finetuning,
- real VideoMAE online TAD,
- adaptive/budgeted acquisition,
- AdaTAD contribution,
- whole-model no-future correctness.

## Key Technical Absorption

1. `no-future` must be audited end to end. Causal convolution alone is insufficient; dataset, cache, memory, post-processing, NMS, emission, and evaluation must all be checked.
2. Current P0/P1 is better formalized as **Windowed Streaming-Safe Online TAD**. Emission is tied mostly to `window_end_frame`, not continuous frame-by-frame decisions.
3. The current raw-frame pipeline is real, but it uses full annotation endpoints for training targets. This is not online-censored supervision.
4. The current mAP evaluator is traditional segment AP; ledger/latency are side outputs. A paper-level online metric should couple AP with emission time and validity.
5. The VideoMAE route is still contract-only stub until a real causal/streaming VideoMAE backbone is wired and tested.
6. Adaptive selection must prove that the heavy visual encoder/detector only processes selected frames/tokens. Dense processing followed by sparse masking is not enough.
7. Irregular token timestamps must be decoded in seconds. Pretending sparse selected tokens form a uniform dense grid will likely break high-tIoU mAP.

## Current-Commit Recheck: 2026-07-08

Confirmed code-level issue:

- P1 inference clamps predicted end times to the current point, but the training path still supervised full future endpoints through `AnchorFreeHead.losses` and MATR end/emit auxiliary targets.

Implemented guard:

- `AnchorFreeHead` now supports `online_censored_training` and `online_censored_max_future_offset`.
- When enabled, positive regression losses whose target end is later than `point_center + max_future_offset` are weighted to zero.
- `MATRHead` passes its `max_future_offset` into the base head and masks end/emit targets until the endpoint is observable.
- P1/P1-pilot/P1-fix/P1-full-60/P2 configs inherit `online_censored_training=True`.

Remaining limitation:

- This is the first online-censored training guard, not a full Stage-1 paper implementation. Center-sampling, positive assignment policy, action-prefix supervision, target audits, and ablation results still need formal validation.

## Paper-Level Route

Recommended final method name:

**Budgeted Adaptive Online Temporal Action Detection with Causal Emission**

Core method components:

1. Online-censored localization loss:
   do not regress unobserved future endpoints.
2. Causal emission/hazard branch:
   learn when to emit, not only what segment to score.
3. Adaptive frame/token selector:
   learn budgeted acquisition under max-gap and latency constraints.
4. Irregular-time detector head:
   use real token timestamps for proposal decoding and NMS.
5. Ledger-aware online evaluation:
   integrate mAP, validity, emission time, and latency.

## Stage Gates

### Stage 0: P1 Full Baseline Audit

Required artifacts:

- full THUMOS validation mAP at tIoU 0.3:0.7,
- emission ledger JSON,
- latency summary JSON,
- emission count statistics,
- no-future violation summary,
- training stability log,
- overfit/smoke evidence.

Go:

- non-random, stable mAP;
- zero ledger violations;
- no NaN/inf training collapse;
- latency summary present and interpretable.

No-go:

- near-zero mAP;
- emission count explosion;
- missing ledger/latency artifacts;
- future-end rows or negative latency rows.

### Stage 1: Online-Censored Training

Implement:

- `opentad/models/utils/online_censored_targets.py`,
- `online_censored_training` mode in the head/loss path,
- observed-end masks,
- censored right-boundary regression masks,
- actionness/inside-action prefix supervision.

Ablations:

- full endpoint target,
- censored endpoint target,
- censored + actionness,
- censored + boundary observed-only.

Go:

- mAP does not collapse;
- high-tIoU or latency improves;
- no-future target audit passes.

### Stage 2: Causal Boundary / Emission Branch

Implement:

- causal hazard/emission target,
- matched-GT latency reporting,
- duplicate suppression under streaming NMS,
- score-threshold baseline comparison.

Go:

- latency improves without large mAP loss;
- emission branch is not just a duplicate score threshold.

### Stage 3: Adaptive Selector

Implement:

- `opentad/models/selectors/causal_frame_selector.py`,
- selector state keyed by stream/video,
- selected-only heavy encoder path,
- budget, max-gap, entropy, diversity, coverage, boundary, actionness distillation, and latency losses.

Required baselines:

- uniform K,
- random K,
- motion-only K,
- learned selector without boundary loss,
- learned selector with full losses,
- dense teacher upper bound.

Go:

- selector beats uniform/random under the same budget;
- no-future selector perturb test passes;
- compute/frame budget reduction is real.

### Stage 4: Irregular-Time AdaTAD/MATR Head

Implement:

- selected token packet with `token_times_sec`,
- irregular proposal decode in seconds,
- timestamp-aware NMS,
- ledger coordinates consistent with decoded seconds.

Go:

- high-tIoU mAP does not collapse;
- sparse token timing is correct under roundtrip tests.

## Key Files To Create Or Refactor

Must add:

- `opentad/models/utils/online_censored_targets.py`
- `opentad/models/selectors/causal_frame_selector.py`
- `opentad/models/utils/selected_token_packet.py`
- `opentad/models/utils/irregular_time_decode.py`
- `opentad/evaluations/online_map.py`
- `opentad/utils/online_ledger_validation.py`

Must refactor:

- `opentad/models/dense_heads/matr_head.py`
- `opentad/models/dense_heads/anchor_free_head.py`
- `opentad/models/detectors/single_stage.py`
- `opentad/cores/test_engine.py`
- `configs/causaltad/` into clear P1/P2/P3/P4 stage configs.

## Risk Register

- P0: full endpoint supervision leakage.
- P0: windowed streaming mislabeled as continuous streaming.
- P0: standard mAP disconnected from latency.
- P1: VideoMAE route over-claimed before replacing stub.
- P1: memory reintroduces overlapping-window leakage.
- P1: DDP eval state split; streaming-safe eval remains single-rank unless redesigned.
- P2: selector optimizes action coverage but misses boundaries.
- P2: irregular timestamp decoding errors damage mAP@0.6/0.7.

## Working Rule

This review reinforces the current discipline:

- Claim only the weakest statement supported by code and artifacts.
- Treat P1-full-60 as a baseline audit, not a paper result, until mAP/ledger/latency are present.
- Implement Stage 1 before spending effort on adaptive selection.
- Do not make AdaTAD or VideoMAE the headline until the corresponding real path exists.
