# Pro Review Absorption: 2026-07-06

Source record:

- Full external review: `PRO_REVIEW_20260706.md`
- Public branch reviewed: `codex/online-tad-clean-20260702`
- Review stance: strict static code/research audit. It did not run the full 60 epoch job or validation mAP.

## Absorbed Verdict

The current branch must not be described as a paper-ready end-to-end raw-frame online TAD system. The safe wording is narrower:

> A windowed streaming-safe raw-frame TAD prototype with frozen SigLIP/SigLIP2 visual encoding, causal temporal projection, MATR-style head wiring, single-rank emission ledger, and no raw-prediction cache.

Do not claim any of the following until the listed blockers are fixed and verified:

- paper-ready full THUMOS mAP,
- true low-latency continuous streaming On-TAD,
- real VideoMAE online TAD,
- adaptive frame selection,
- AdaTAD contribution.

## Initial Local Verification

These checks were run after receiving the review:

```text
python -m pytest tests/test_siglip_ontad_contracts.py tests/test_videomae_matr_ontad_contracts.py tests/test_online_emission_protocol.py tests/test_map_evaluator_contracts.py -q -rs
45 passed, 1 skipped, 1 warning
```

The following review item appears stale or already addressed in the current branch:

- Review claim: `p1_pilot.py` has no `_base_`.
- Current code: `configs/causaltad/thumos_siglip2_matr_ontad_p1_pilot.py` has `_base_ = "./thumos_siglip2_matr_ontad_p1.py"`.
- Current chain: `p1_full_60.py -> p1_fix.py -> p1_pilot.py -> p1.py -> p0.py`.
- Action: still add stronger resolved-config tests, but do not treat the missing `_base_` claim as confirmed without rechecking the exact reviewed commit.

The following review items are valid or high-priority risks in the current branch:

- `formal_training_ready=False` remains a correct claim guard for P0/P1/full-60. It is not a runtime blocker by itself, but it blocks paper-style wording.
- `configs/causaltad/thumos_videomae_adapter_matr_ontad.py` still uses a contract-only stub path. It is not a real VideoMAE online TAD route.
- `opentad/cores/train_engine.py` still has direct `model.module` accesses in the backbone LR logging block. This should be made robust for non-DDP/single-process paths.
- Training supervision may still expose future action endpoints to prefix tokens. This needs a real online-censored target/loss audit and likely implementation.
- The current emission protocol is windowed streaming-safe emission, not per-frame or per-token continuous low-latency streaming. Latency claims must be ledger-based.

## Priority Backlog

### P0: Correctness And Claim Hygiene

1. Make `train_engine.py` robust to non-DDP models everywhere:
   `target_model = model.module if hasattr(model, "module") else model`.
2. Add resolved-config tests for P0, P1, P1-pilot, P1-fix, and P1-full-60:
   assert raw-frame dataset, SigLIP encoder, causal projection, streaming-safe emission, no raw prediction cache, and full-60 no allow-list.
3. Add a full-chain no-future perturb test for the actual P1 config path, not only isolated encoder/projection units.
4. Make the latency contract explicit:
   report and validate emission latency from ledger, and do not call the current route zero-latency.
5. Add synthetic ledger-to-mAP integration tests:
   verify no global video-level NMS, allowed-videos filtering, and no future rows.

### P1: Online Training Semantics

6. Audit raw-frame GT assignment and MATR losses for future endpoint leakage.
7. Implement online-censored target mode if the audit confirms prefix tokens regress unobserved action ends.
8. Add tests where an action ends outside the observed prefix:
   end/boundary loss must be masked or converted to an observed-prefix lower-bound/actionness objective.
9. Add optimizer group tests:
   frozen SigLIP params are absent or lr=0; adapter/projection/head params are trainable; motion branch grads exist when enabled.
10. Add frame-grid-seconds roundtrip tests for `window_start_frame`, `feature_stride`, `fps`, snippet centers, and output seconds.

### P2: Real Model Contribution

11. Treat the VideoMAE route as stub-only until a real causal/streaming VideoMAE backbone is wired with `use_stub_backbone=False`.
12. For any real VideoMAE claim, require:
    no-future perturb tests, gradient tests for trainable blocks, and strict state/memory tests.
13. Only after P1 is stable, implement adaptive frame selection as a separate contribution:
    selector budget, max-gap invariant, no-future selector state, selected-frame packing, irregular token times, and detector decode in seconds.
14. Prefer an irregular-token AdaTAD/MATR bridge over pretending sparse frames are a uniform dense grid.
15. Prevent selector collapse with budget, entropy, diversity, temporal coverage, actionness distillation, boundary preservation, and latency losses.

## Working Rule Going Forward

Use `PRO_REVIEW_20260706.md` as a review input, not as ground truth. Each item must be checked against the current branch before implementation. Confirmed bugs should be fixed in priority order with focused tests. Stale or context-mismatched findings should be documented and closed with evidence.
