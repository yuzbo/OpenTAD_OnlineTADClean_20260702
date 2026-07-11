# Persistent Event-Set Stage-1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deploy a matched-feature falsification experiment for fresh, Temporal TrackFormer, and prefix-observable persistent event-set Online TAD.

**Architecture:** A chronological cached-feature dataset constructs a training-only prefix schedule. A shared detector scans each causal chunk sequentially and switches only the query persistence, start representation, endpoint objective, and assignment policy required by the registered variant. Inference writes an immutable ledger without EOF, future features, NMS, or GT.

**Tech Stack:** Python, PyTorch, MMEngine/OpenTAD registries, NumPy, pytest, Slurm, THUMOS14, frozen SigLIP2 feature cache.

---

### Task 1: Prefix-Observable Schedule

**Files:**
- Create: `opentad/models/targets/prefix_instance_schedule.py`
- Modify: `opentad/models/targets/__init__.py`
- Test: `tests/test_prefix_instance_schedule.py`

- [x] Write failing tests for delayed endpoints, same-class overlap, chunk cuts, and no future endpoint in active rows.
- [x] Run `python -m pytest tests/test_prefix_instance_schedule.py -q`; expect import failure.
- [x] Implement immutable `PrefixInstance`, `PrefixScheduleStep`, and `build_prefix_instance_schedule` values in absolute frame coordinates.
- [x] Re-run the focused test; expect all schedule tests to pass.

### Task 2: Chronological Cached-Feature Dataset

**Files:**
- Create: `opentad/datasets/streaming_feature.py`
- Modify: `opentad/datasets/__init__.py`
- Test: `tests/test_streaming_feature_dataset.py`

- [x] Write a failing temporary-cache test that checks contiguous chunks, source-frame timestamps, complete-video manifests, training-only schedules, and terminal metadata sanitization.
- [x] Run the test and verify the dataset type is missing.
- [x] Implement `StreamingFeatureDataset` with NumPy mmap reads, fixed feature stride, one-video chronological manifests, and no inference GT fields.
- [x] Re-run the focused test; expect pass.

### Task 3: Shared Event-Set Head

**Files:**
- Create: `opentad/models/dense_heads/persistent_event_set_head.py`
- Modify: `opentad/models/dense_heads/__init__.py`
- Test: `tests/test_persistent_event_set_head.py`

- [x] Write failing tests for fresh-query reset, persistent state carry, start-pointer `before-memory`, first-event risk masking, same-class independent slots, one-time emission, and refractory rearm.
- [x] Run the tests and verify import failure.
- [x] Implement the smallest shared query decoder and variant switches; do not add duplicate or delay repair losses.
- [x] Re-run focused tests; expect pass.

### Task 4: Online Detector and Ledger

**Files:**
- Create: `opentad/models/detectors/persistent_event_set_ontad.py`
- Modify: `opentad/models/detectors/__init__.py`
- Test: `tests/test_persistent_event_set_detector.py`

- [x] Write failing tests for chronological state, no-EOF prediction inputs, detached cross-chunk state, end `<=` emit, immutable rows, future perturbation, and chunk/step equivalence.
- [x] Run and verify the detector is missing.
- [x] Implement feature projection, sequential chunk scan, training assignment state, inference lifecycle state, and ledger output.
- [x] Re-run focused tests; expect pass.

### Task 5: Cache Extraction and Audit

**Files:**
- Create: `tools/cache_ontad_features.py`
- Test: `tests/test_ontad_feature_cache.py`

- [x] Write failing tests using a tiny fake encoder and short video/tensor fixture; require one manifest, deterministic source frames, atomic per-video files, and resume safety.
- [x] Run and verify missing tool module.
- [x] Implement sequential one-open-per-video extraction, selected-frame batching, SHA256 metadata, and fail-closed resume validation.
- [x] Re-run focused tests; expect pass.

### Task 6: Matched Configs and Result Contract

**Files:**
- Create: `configs/causaltad/thumos_pes_stage1_base.py`
- Create: `configs/causaltad/thumos_pes_stage1_fresh.py`
- Create: `configs/causaltad/thumos_pes_stage1_trackformer.py`
- Create: `configs/causaltad/thumos_pes_stage1_persistent.py`
- Test: `tests/test_pes_stage1_config_contracts.py`

- [x] Write failing tests asserting identical cache, features, dimensions, slots, optimizer, schedule, evaluator, thresholds, and seeds; only registered mechanism fields may differ.
- [x] Run and verify missing configs.
- [x] Add configs with `formal_training_ready=False`, fixed three-seed manifest, one-rank streaming, and raw-video finetuning disabled.
- [x] Re-run focused tests; expect pass.

### Task 7: Fail-Closed Slurm Deployment

**Files:**
- Create: `tools/remote/submit_pes_stage1_n16r4.sh`
- Create: `tools/remote/check_pes_stage1_n16r4.sh`
- Test: `tests/test_pes_stage1_launch_contracts.py`

- [x] Write failing source-contract tests for fresh checkout, cache gate, smoke gate, total GPU-hour budget, Slurm-only GPU execution, and prohibition of the raw finetune config.
- [x] Run and verify missing scripts.
- [x] Implement `cache`, `smoke`, and `pilot` modes with distinct run directories and explicit authorization gates.
- [x] Re-run focused tests; expect pass.

### Task 8: Verification and Deployment

**Files:**
- Update: `research-wiki/experiments/persistent-feature-kill-test-20260712.md`
- Update: `research-wiki/log.md`
- Update: `research-wiki/decision_register.md`
- Update: `research-wiki/source_map.md`

- [x] Run focused and full local/remote tests, compile checks, wiki JSONL parsing, private-path scan, and `git diff --check`.
- [x] Commit and push the isolated implementation without staging unrelated local PCEH edits.
- [x] Create a fresh remote checkout at the pushed SHA.
- [x] Submit cache extraction through Slurm and record job ID, run directory, SHA, config, budget, and current state.
- [ ] After the complete cache manifest passes validation, submit and inspect the GPU smoke.
- [x] Do not submit the three-seed pilot until cache and smoke artifacts pass.
