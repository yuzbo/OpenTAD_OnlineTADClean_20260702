# Pro Review Absorption: 2026-07-08

Source record:

- Full external review: `PRO_REVIEW_20260708.md`
- Reviewed public branch: `codex/online-tad-clean-20260702`
- Reviewed commit: `7caf2b9 implement adaptive online tad target route`
- Full SHA: `7caf2b972958328753d36f0f0079fce2893468c8`
- Review verdict: `HOLD`
- Duplicate attachments received: yes; both pasted-text files had identical SHA256.

## Absorbed Verdict

The review does not mark the current implementation as `FAIL`: the code now has real skeleton pieces for selected-frame encoding, irregular proposal metadata, streaming ledger rows, and `OnlineMAP`.

However, the review explicitly blocks the following claims:

- no remote real training yet,
- no paper-level Online TAD claim,
- no true adaptive selected-frame Online TAD claim,
- no paper-level OnlineMAP / online AP claim.

Allowed wording for the current commit:

> A selected-only causal-stride raw-frame online TAD target skeleton / validation candidate with ledger-aware emitted-row mAP support.

Disallowed wording for the current commit:

> A completed end-to-end adaptive selected-frame Online TAD method.

## P0 Findings To Treat As Blocking

1. `OnlineEmitter.step` latency semantics are reversed.

   Current behavior treats `latency_frames` like a minimum waiting time:
   `end_frame <= now_frame - latency_frames`.

   Required behavior treats it like a maximum allowed delay:
   `end_frame <= now_frame` and `now_frame - end_frame <= latency_frames`.

   Also update `state.last_emit_frame` on every `step`, including no-emission steps.

2. `MATRHead` irregular axis is batch-shared.

   Current irregular points are built from `metas[0]` and require all batch samples to have identical selected positions. This conflicts with genuine adaptive selection.

   Short-term rule: final adaptive/irregular route must be explicit `batch_size=1` and fail fast for batch-size greater than one when irregular selected metadata is present.

   Long-term rule: implement per-sample irregular points and per-sample target/decode.

3. `OnlineMAP` statistics are computed before AP filtering.

   Current online side stats are computed from full `data["results"]`, while AP rows are later filtered by `allowed_videos`, `blocked_videos`, and `max_latency_sec`.

   Required fix: filter rows first, then compute online stats and AP dataframe from the same filtered result set.

4. `CausalFrameSelector` all-invalid mask fabricates frame 0.

   Current all-invalid mask returns index `0`, causing an invalid padded frame to enter the visual encoder.

   Required short-term fix: fail fast on all-invalid masks.

## P1 Findings To Absorb

1. The final route name and comments overstate adaptivity.

   `policy="causal_stride"` is fixed selection, not content-adaptive or learned adaptive selection. Until a content-dependent selector is used and tested, claims should say `selected-only causal stride`, not `adaptive`.

2. `assert_selected_only` is too weak.

   The current assertion only checks that selected frame count is not greater than dense frame count. It does not prove the heavy encoder saw only selected frames.

   Required runtime test: monkeypatch `_encode_pixels_chunked` and assert encoded frame count equals `selected.selected_masks.sum()`, not dense `B*T`.

3. NMS coordinate semantics are still fragile.

   The final dense-grid MATR path avoids double irregular conversion, but selected-axis fallback would still run NMS before seconds conversion.

   Required fix before using selected-axis proposals: run NMS in a single explicit coordinate system, preferably seconds.

4. `online_censored_training` is incomplete.

   Current auxiliary targets censor end/emit but still mark actionness inside the full future segment. This is not a full online-censored supervision protocol.

   Required research fix: observed/pending target protocol or dedicated censored GT builder.

5. `decode_irregular_segments_to_seconds` uses whole-video duration as the right sentinel.

   For sliding windows, this can stretch the last selected token interval to the full video end.

   Required fix: pass and validate a window right boundary such as `token_right_boundary_sec`.

6. Streaming path does not explicitly carry `proposal_axis`.

   It currently works only because the final MATR path returns dense-grid proposals. Future selected-axis/token-time-native proposals would be misinterpreted.

## Required Runtime Tests

Add behavior tests, not only source-string contract tests:

- `test_online_emitter_latency_budget_semantics`
- `test_selected_only_encoder_runtime_count`
- `test_matr_irregular_batch_size_fail_closed`
- `test_online_map_filtered_stats_match_ap_rows`
- `test_irregular_time_decode_window_boundary`
- `test_dense_grid_no_double_irregular_decode`

Minimum smoke after P0 fixes:

- tiny backend,
- `B=1`,
- `T=8` or `T=16`,
- `policy="causal_stride"`,
- `streaming_safe_emission=True`,
- finite loss/backward,
- encoded frames less than dense frames,
- `irregular_selected_positions` exists,
- `proposal_axis == "dense_grid"`,
- ledger has `emit_frame`, `source_frame`, `start_frame`, `end_frame`, `latency_sec`,
- zero no-future violations.

## Working Plan

1. Fix the four P0 blocking issues before any remote real training:
   latency semantics, irregular batch contract, OnlineMAP filtering consistency, all-invalid selector masks.

2. Add runtime tests for the fixes.

3. Run local static and runtime smoke tests. If local torch remains broken, run these tests on the remote environment before training.

4. Only after P0 tests pass, allow a short single-GPU `B=1` THUMOS smoke. Do not submit long full-data training before this gate.

5. After smoke passes, decide claim route:
   either rename final route to selected-only causal-stride skeleton, or implement a genuine content-dependent selector.

## Updated Claim Boundary

Allowed after this review:

- `selected-only causal-stride raw-frame online TAD skeleton`,
- `ledger-aware emitted-row mAP wrapper`,
- `irregular dense-grid metadata path under validation`,
- `single-rank streaming-safe emission ledger candidate`.

Not allowed:

- `paper-ready Online TAD`,
- `true adaptive selected-frame Online TAD`,
- `learned budgeted selector`,
- `paper-level OnlineMAP`,
- `remote training ready`,
- `full online-censored training`.

## Absorbed Priority

This review changes the immediate priority from P3/P4 expansion back to P0 hardening.

P0 implementation status after the follow-up hardening patch:

1. `opentad/utils/online_protocol.py::OnlineEmitter.step`
   now treats `latency_frames` as a maximum allowed delay and updates
   `last_emit_frame` on every step.
2. `opentad/models/selectors/causal_frame_selector.py::_valid_indices`
   now fails fast on all-invalid masks instead of fabricating frame 0.
3. `opentad/evaluations/online_map.py::OnlineMAP._import_prediction`
   now filters predictions before computing both online stats and AP rows.
4. `opentad/models/dense_heads/matr_head.py`
   now fail-closes irregular selected-axis metadata to `batch_size=1`.
5. `opentad/models/backbones/online_siglip_adapter.py`
   now enforces stricter selected-only runtime invariants.

Regression coverage:

- `tests/test_p0_review_hardening.py`
- `tests/test_online_emission_protocol.py`

Local verification:

- `python -m pytest tests -q -rs`
- `python -m py_compile` over changed implementation and test files

Remaining HOLD items are still the P1/P2 research items above: true adaptive
selection, online-censored supervision, explicit proposal-axis/coordinate-system
contracts for selected-axis decode, window-bounded irregular timing, and remote
real training/evaluation evidence.
