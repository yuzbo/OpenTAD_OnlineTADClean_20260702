# DF-RPA Gated Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build DF-RPA as a strict, auditable, bounded-latency On-TAL research route only after detector-feedback acquisition beats strong non-feedback controls.

**Architecture:** Keep the current CausalTAD/OpenTAD route clean and add small, testable protocol and diagnostic components before any raw heavy-reader model. The implementation proceeds through hard gates: baseline reproduction, feature-stream acquisition diagnostics, detector-in-loop replay, selected-only raw smoke, and only then the full online model.

**Tech Stack:** Python, pytest, MMEngine/OpenTAD configs, CausalTAD, future Slurm-controlled GPU runs, optional raw VideoMAE/VideoMAEv2 reader only after earlier gates pass.

---

## Claim Discipline

Safe claim:

> Under a strict bounded-latency On-TAL protocol, detector belief state drives selected-only raw-packet acquisition from a finite recent buffer to improve high-IoU localization and start-boundary recovery under matched compute and latency budgets.

Forbidden claims:

- first online temporal action localization model;
- first end-to-end online action model;
- first adaptive frame or clip selection method;
- first progress-aware online video model;
- streaming video reasoning as the main contribution;
- feature-stream AUPRC or oracle packet selection as final deployment evidence;
- dense VideoMAE cache as selected-only raw inference.

## Stop / Pivot Rule

Stop the DF-RPA mainline if detector-feedback acquisition does not improve detector-in-loop mAP@0.6/0.7 or start MAE over matched-budget uniform, boundary-only, progress-only, actionness-only, uncertainty-only, and no-feedback heavy-reader controls.

Allowed pivots:

- a MATR/HAT acquisition plugin;
- a missing-start diagnostic module;
- a feature-stream diagnostic paper with no strong raw-stream claim.

## File Structure

- `DF_RPA_GATED_IMPLEMENTATION_PLAN.md`: this gate plan and execution checklist. It is kept at the repository root because this clean repo intentionally ignores `docs/` as historical research material.
- `opentad/utils/online_protocol.py`: pure Python selected-only bounded-latency audit helpers. No torch, no training dependency, no raw data.
- `tests/test_online_protocol_audit.py`: contract tests for causality, selected-only reader provenance, and no forbidden cache/GT/teacher decisions.
- Future `tools/analysis/df_rpa_feature_gate.py`: feature-stream diagnostic runner after the audit helpers exist.
- Future `tools/analysis/df_rpa_replay.py`: detector-in-loop replay runner after feature gate passes.

## Task 1: Selected-Only Protocol Audit Foundation

**Files:**
- Create: `opentad/utils/online_protocol.py`
- Create: `tests/test_online_protocol_audit.py`
- Modify: `opentad/utils/__init__.py`

- [x] **Step 1: Write the failing tests**

```python
from opentad.utils.online_protocol import (
    PacketRead,
    ProtocolViolation,
    audit_packet_reads,
)


def test_audit_accepts_selected_packet_inside_bounded_buffer():
    reads = [
        PacketRead(
            video_id="v1",
            current_time=10.0,
            buffer_start=6.0,
            buffer_end=10.0,
            packet_start=7.0,
            packet_end=9.0,
            reader="videomae_v2",
            source="selected_policy",
        )
    ]

    summary = audit_packet_reads(reads)

    assert summary.num_reads == 1
    assert summary.total_packet_duration == 2.0
    assert summary.max_packet_delay == 3.0


def test_audit_rejects_future_or_out_of_buffer_packet():
    reads = [
        PacketRead(
            video_id="v1",
            current_time=10.0,
            buffer_start=6.0,
            buffer_end=10.0,
            packet_start=9.0,
            packet_end=10.5,
            reader="videomae_v2",
            source="selected_policy",
        )
    ]

    try:
        audit_packet_reads(reads)
    except ProtocolViolation as exc:
        assert "outside bounded buffer" in str(exc)
    else:
        raise AssertionError("expected ProtocolViolation")


def test_audit_rejects_oracle_dense_cache_or_teacher_sources():
    forbidden_sources = ["gt", "oracle", "dense_cache", "teacher_cache"]
    for source in forbidden_sources:
        reads = [
            PacketRead(
                video_id="v1",
                current_time=10.0,
                buffer_start=6.0,
                buffer_end=10.0,
                packet_start=7.0,
                packet_end=9.0,
                reader="videomae_v2",
                source=source,
            )
        ]

        try:
            audit_packet_reads(reads)
        except ProtocolViolation as exc:
            assert "forbidden packet source" in str(exc)
        else:
            raise AssertionError("expected ProtocolViolation")
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_online_protocol_audit.py -q`

Expected: FAIL because `opentad.utils.online_protocol` does not exist.

- [x] **Step 3: Write minimal implementation**

```python
from dataclasses import dataclass
from typing import Iterable


class ProtocolViolation(ValueError):
    pass


@dataclass(frozen=True)
class PacketRead:
    video_id: str
    current_time: float
    buffer_start: float
    buffer_end: float
    packet_start: float
    packet_end: float
    reader: str
    source: str = "selected_policy"


@dataclass(frozen=True)
class PacketReadAuditSummary:
    num_reads: int
    total_packet_duration: float
    max_packet_delay: float


def audit_packet_reads(reads: Iterable[PacketRead]) -> PacketReadAuditSummary:
    checked = list(reads)
    total_duration = 0.0
    max_delay = 0.0

    for read in checked:
        _validate_packet_read(read)
        total_duration += read.packet_end - read.packet_start
        max_delay = max(max_delay, read.current_time - read.packet_start)

    return PacketReadAuditSummary(
        num_reads=len(checked),
        total_packet_duration=total_duration,
        max_packet_delay=max_delay,
    )


def _validate_packet_read(read: PacketRead) -> None:
    if read.source != "selected_policy":
        raise ProtocolViolation(f"forbidden packet source: {read.source}")
    if read.buffer_end > read.current_time:
        raise ProtocolViolation("buffer end is in the future")
    if read.packet_start < read.buffer_start or read.packet_end > read.buffer_end:
        raise ProtocolViolation("packet is outside bounded buffer")
    if read.packet_start >= read.packet_end:
        raise ProtocolViolation("packet start must be before packet end")
```

- [x] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_online_protocol_audit.py -q`

Expected: PASS.

- [x] **Step 5: Run existing contract tests**

Run: `python -m pytest tests/test_causaltad_config_contracts.py tests/test_online_protocol_audit.py -q`

Expected: PASS.

## Task 2: Feature-Stream Acquisition Gate

**Files:**
- Create: `tools/analysis/df_rpa_feature_gate.py`
- Create: `tests/test_df_rpa_feature_gate.py`

- [x] **Step 1: Define a file-based diagnostic input contract**

The runner reads a JSONL file where each line contains:

```json
{"video_id":"v1","time":10.0,"candidate_start":7.0,"candidate_end":9.0,"score_feedback":0.9,"score_uniform":0.2,"score_boundary":0.4,"label_useful":1}
```

- [x] **Step 2: Add tests for matched candidate ranking**

The first test must assert that feedback precision@K is computed from the same candidates as boundary/progress/uniform controls.

- [x] **Step 3: Implement only deterministic metrics**

Implement precision@K and average score for positive candidate packets. Do not add model training, oracle deployment, or raw reading in this task.

## Task 3: Detector-In-Loop Replay Gate

**Files:**
- Create: `tools/analysis/df_rpa_replay.py`
- Create: `tests/test_df_rpa_replay_contract.py`

- [ ] **Step 1: Require downstream detector outputs**

The replay runner must require predicted segments from the same detector interface for feedback and all controls.

- [ ] **Step 2: Report localization metrics**

Report mAP proxy inputs, start MAE inputs, commit delay, and selected packet budget. This task should not claim final mAP until wired into official evaluator.

## Task 4: Selected-Only Raw Heavy-Reader Smoke

**Files:**
- Create: `tools/analysis/df_rpa_selected_reader_audit.py`
- Extend: `opentad/utils/online_protocol.py`

- [ ] **Step 1: Log every heavy-reader invocation**

Each log row must include `video_id`, `current_time`, `buffer_start`, `buffer_end`, `packet_start`, `packet_end`, `reader`, `source`, `num_frames`, and `wall_time_ms`.

- [ ] **Step 2: Reject dense cache provenance**

Validation/test logs with `source` equal to `dense_cache`, `teacher_cache`, `gt`, or `oracle` must fail before metrics are reported.

## Task 5: Full Online Model Only After Gates Pass

**Files:**
- Future configs and model files should be named with `df_rpa` only after Tasks 1-4 pass.

- [ ] **Step 1: Compare against strong baselines**

Run MATR/HAT/OAT/SimOn where possible, plus matched-budget uniform, random, boundary-only, progress-only, actionness-only, uncertainty-only, and no-feedback acquisition controls.

- [ ] **Step 2: Claim only supported results**

The final report must lead with mAP@0.6/0.7, start MAE, commit delay, FLOPs, packet count, and selected-only audit status.

## Self-Review

- Spec coverage: the plan encodes the accepted 7.0/10 posture, narrow claim, hard stop/pivot rule, matched compute controls, missing-start/high-IoU emphasis, and the requested implementation order.
- Placeholder scan: no `TBD` or unconstrained "implement later" steps remain; future tasks are explicitly gated and scoped.
- Type consistency: `PacketRead`, `ProtocolViolation`, `PacketReadAuditSummary`, and `audit_packet_reads` are introduced in Task 1 and reused by later audit tasks.
