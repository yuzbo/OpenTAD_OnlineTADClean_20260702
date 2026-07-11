---
updated: 2026-07-10
status: active
scope: Current PCEH-OnTAD implementation snapshot after the local coding pass.
out-of-scope: Remote Slurm training results, dataset files, checkpoints, feature dumps, and paper-ready claims.
---

# PCEH-OnTAD Current Implementation

## Project Position

The project is now organized around a strict online temporal action detection
route named PCEH-OnTAD: Prefix-Censored Event-Emission Hazard Learning for
Online Temporal Action Detection.

The core target is not "another online TAD wrapper". The target is a trainable
streaming detector that separates physical action end time, prefix-censored
action lifecycle state, first immutable emission time, and the GT-matched
latency budget used by online evaluation.

The repository state is still a candidate implementation, not a finished
research result. Formal-result configs remain `formal_training_ready=False`
until remote smoke, causal replay, optimizer/update audit, and multi-seed
training evidence pass.

## Implemented Code Surfaces

### Streaming Data Route

- `opentad/datasets/streaming_raw_frame.py` builds chronological raw-frame
  packets and hides terminal-only video metadata from non-terminal packets.
- `opentad/datasets/transforms/streaming.py` selects observable frames inside
  one packet and records encoded source frames.
- `opentad/utils/stream_packets.py` provides packet manifests and a single-rank
  chronological sampler. Multi-rank streaming is intentionally rejected until
  stream-state sharding exists.

### Prefix Targets

- `opentad/models/targets/prefix_event_targets.py` builds prefix-only targets:
  `start_target`, `ongoing_target`, `end_event`, `censor_mask`,
  `completion_target`, `emit_allowed`, `emit_forbidden`, `emit_event`, and
  `late_target`.
- Tests verify that future endpoints do not affect pre-end prefix targets and
  that emission is forbidden before an endpoint is observed.

### Event/Emission Head

- `opentad/models/dense_heads/prefix_event_emission_head.py` implements
  independent class, start, ongoing, endpoint hazard, completion, and emission
  hazard heads.
- It supports `pceh` for the proposed route and `endpoint_only` for a fair
  controlled baseline.
- Decode state emits immutable records and keeps a per-stream ledger.

### Incremental Detector

- `opentad/models/detectors/pceh_ontad.py` implements `PCEHOnlineDetector`
  with bounded causal feature cache, source-frame provenance, per-packet
  `forward_step`, and OpenTAD-compatible `forward`.
- It records `max_raw_frame_read` and `max_cache_source_frame` from actual
  encoded/cache state rather than prediction coordinates.
- Terminal packets release online state for that stream.
- Emission rows include the backward-compatible `latency_sec` field plus
  explicit `predicted_end_latency_sec` and
  `latency_definition="emit_time_minus_predicted_end_time"`.

### Online Evaluation

- `opentad/evaluations/online_budgeted_map.py` implements `OnlineAPBudgeted`.
- The primary latency definition is
  `emit_time_sec - matched_ground_truth_end_sec`.
- Late predictions are retained as false positives and never removed before AP
  ranking.
- Predicted-end latency is reported only as a diagnostic field.

### Causal and Training Audits

- `opentad/utils/causal_audit.py` provides packet, future-perturbation,
  chunk-invariance, batch-isolation, and emission-ledger audits.
- `opentad/utils/optimizer_audit.py` checks trainable parameter coverage in
  optimizer groups.
- `opentad/utils/training_audit.py` checks whether required modules receive
  gradients and parameter updates.
- `tools/smoke_pceh_stream.py` now reads required update modules from
  `smoke_required_modules` or `trainable_scope`, so the fine-tuning route can
  require proof that the visual tower itself updates.

## Implemented Config Routes

### Main Candidate

- `configs/causaltad/thumos_pceh_ontad.py` is the strict packet-based
  frozen-visual-tower PCEH candidate.
- It uses 8-frame packets, `PrefixEventEmissionHead`, bounded cache, immutable
  ledgers, and `OnlineAPBudgeted`.
- It is useful for protocol validation and controlled pilot runs, but it does
  not by itself prove pretrained visual model fine-tuning.

### End-to-End Fine-Tuning Candidate

- `configs/causaltad/thumos_pceh_ontad_finetune.py` is the route intended to
  support the stronger claim that a pretrained visual tower can be adapted by
  the online TAD objective itself.
- It sets `freeze_backbone=False`, `freeze_vision_encoder=False`, low LR for
  the visual tower, and `smoke_required_modules` including
  `backbone.backbone.vision_encoder`, `backbone.adapter`, `projection`, and
  `head`.

### Controlled Baselines

- `configs/causaltad/thumos_pceh_endpoint_only.py` keeps the same stream,
  cache, visual compute, optimizer, and evaluator but removes
  completion/emission losses and emits from endpoint hazard only.
- `configs/causaltad/thumos_pceh_chunk_end_baseline.py` represents the
  high-latency chunk-end control.
- `configs/causaltad/thumos_pceh_rolling_fixed_stride2.py` is the rolling
  fixed-stride acquisition control.

## Local Verification

The latest local verification passed:

```text
python -m py_compile tools/train.py tools/test.py tools/smoke_pceh_stream.py ...
python -m pytest tests -q
125 passed, 21 skipped, 1 warning
```

The skipped tests are environment-dependent Torch/runtime checks, not formal
remote training evidence.

## Current Worktree Snapshot

Modified implementation files:

- `configs/causaltad/README.md`
- `opentad/cores/optimizer.py`
- `opentad/cores/test_engine.py`
- `opentad/cores/train_engine.py`
- `opentad/models/detectors/pceh_ontad.py`
- `opentad/utils/online_protocol.py`
- `tools/smoke_pceh_stream.py`

Modified tests:

- `tests/test_online_emission_protocol.py`
- `tests/test_pceh_config_contracts.py`
- `tests/test_pceh_incremental_detector.py`
- `tests/test_pceh_launch_contracts.py`
- `tests/test_training_update_audit.py`

New files:

- `configs/causaltad/thumos_pceh_ontad_finetune.py`
- `pceh-current-implementation-20260710.md`

Existing untracked review records remain separate and should not be mixed with
implementation commits unless explicitly requested:

- `PRO_REVIEW_20260709.md`
- `PRO_REVIEW_20260710.md`
- `PRO_REVIEW_ABSORPTION_20260709.md`
- `PRO_REVIEW_ABSORPTION_20260710.md`

## What Is Not Yet Proven

- No remote Slurm smoke report has been produced for the fine-tuning route.
- No optimizer/update audit artifact has yet shown that the SigLIP2 visual
  tower updates on real THUMOS14 packets.
- No multi-seed results exist for PCEH vs endpoint-only vs chunk-end vs rolling
  fixed-stride baselines.
- No MUSES or open-vocabulary/OZ-TAL-facing evaluation has been implemented.

Correct current claim:

> The repository now contains a locally tested strict-online PCEH-OnTAD
> implementation and an explicit end-to-end fine-tuning candidate route.

Incorrect current claim:

> PCEH-OnTAD is a validated SOTA online TAD method.

## Next Gate

Run the remote smoke gate for:

```text
configs/causaltad/thumos_pceh_ontad_finetune.py
```

The smoke should produce packet audit pass, causal replay pass, update audit
pass for the visual tower/adapter/projection/head, and immutable ledger rows
with no future-source violations.
