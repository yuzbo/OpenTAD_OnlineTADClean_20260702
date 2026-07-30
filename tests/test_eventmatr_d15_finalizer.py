"""Pure-Python integrity contracts for the D1.5 diagnostic finalizer."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json

import pytest
import torch

from scripts.finalize_eventmatr_d15_owner_counterfactual import (
    CHANNELS,
    PROTOCOL,
    ROUTES,
    _validate_source_provenance,
    _validate_trace,
)
from scripts.run_eventmatr_d15_owner_counterfactual import (
    EXPECTED_D14_COUNTS,
    OFFICIAL_TRAIN_ARTIFACTS,
    _validate_d14_gate,
    _validate_padding_runtime,
)


D14_COMMIT = "fa27b3657b72c5b713ee3d2a5c0e652e7ca14eb4"
D14_TREE = "7603fc6b8226fa8f9dcb3d5212136fb47631bc32"
TRAINING_COMMIT = "92cf34aa07bebee2a7a7e3661431d5055804b29b"
TRAINING_TREE = "aef4f64bc020df9d39ead9811fbc01407f1c754a"


def _routes() -> dict:
    return {
        channel: {
            route: {
                "counts": {
                    "decision_row_count": 1,
                    "end_argmax_count": 0,
                    "end_min_duration_suppression_count": 0,
                    "target_backed_end_transition_count": 0,
                }
            }
            for route in ROUTES
        }
        for channel in CHANNELS
    }


def _trace_row(channel: str, route: str) -> dict:
    return {
        "protocol": PROTOCOL,
        "channel": channel,
        "route": route,
        "video_name": "video_test_0000001",
        "runtime_event_id": 0,
        "diagnostic_target_event_id": None,
        "source": "predicted",
        "creation_frame": 0.0,
        "current_frame": 4.0,
        "first_owner_decision": True,
        "owner_query_before_refresh": 1,
        "owner_query_after_refresh": 1,
        "formal_owner_query_id_after_intervention": 1,
        "formal_owner_query_id_unchanged": True,
        "matched_current_query": None,
        "owner_to_birth_cosine": 1.0,
        "owner_to_oracle_query_cosine": None,
        "owner_attention_top_query": 1,
        "oracle_query_attention_rank": None,
        "owner_attention_entropy": 0.5,
        "cancel_logit": 1.0,
        "continue_logit": 0.0,
        "end_logit": -1.0,
        "cancel_margin": 1.0,
        "continue_margin": -1.0,
        "end_margin": -2.0,
        "state_argmax": 0,
        "end_suppressed_by_minimum_duration": False,
        "formal_transition": "cancel" if route == "formal" else None,
        "shadow_transition": "diagnostic_retain_cancel" if route == "shadow" else None,
        "target_end_observable_now": False,
        "frames_from_annotated_end": None,
        "formal_lifetime": 4.0 if route == "formal" else None,
        "shadow_lifetime": 4.0 if route == "shadow" else None,
    }


def _write_trace(path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _d14_gate(checkpoint, options) -> dict:
    checkpoint_sha256 = _sha256(checkpoint)
    options_sha256 = _sha256(options)
    lifecycle = {
        "runtime_birth_count": EXPECTED_D14_COUNTS["birth_count"],
        "runtime_cancel_count": EXPECTED_D14_COUNTS["cancellation_count"],
        "runtime_end_count": EXPECTED_D14_COUNTS["end_count"],
        "runtime_emit_count": EXPECTED_D14_COUNTS["emit_count"],
        "runtime_reacquisition_count": EXPECTED_D14_COUNTS["reacquisition_count"],
        "runtime_capacity_exhaustion_count": EXPECTED_D14_COUNTS[
            "capacity_exhaustion_count"
        ],
    }
    return {
        "protocol": "eventmatr_d14_cross_arm_structure_gate_v1",
        "status": "FAIL_STRUCTURE_GATE",
        "selected_variant": None,
        "test_access": False,
        "threshold_search": False,
        "threshold_lowering": False,
        "multi_seed": False,
        "raw_rgb_training": False,
        "strict_causal_paper_result_valid": False,
        "official_paper_performance_valid": False,
        "locked_test_release": False,
        "paper_claim_release": False,
        "source_identity": {
            "d14": {"commit": D14_COMMIT, "tree": D14_TREE},
        },
        "official_train_artifacts": OFFICIAL_TRAIN_ARTIFACTS,
        "arms": {
            "decision_aligned_bag": {
                "checkpoint_sha256": checkpoint_sha256,
                "terminal_lifecycle": lifecycle,
            },
        },
        "linked_training_artifacts": {
            "bag": {
                "checkpoint": {
                    "path": str(checkpoint.resolve()),
                    "bytes": checkpoint.stat().st_size,
                    "sha256": checkpoint_sha256,
                },
                "options": {
                    "path": str(options.resolve()),
                    "bytes": options.stat().st_size,
                    "sha256": options_sha256,
                },
            },
        },
    }


def _source_provenance_scan() -> dict:
    return {
        "training_source_identity": {
            "commit": TRAINING_COMMIT,
            "tree": TRAINING_TREE,
            "clean": True,
            "status": "PASS",
            "status_porcelain": "",
            "smoke": {"status": "PASS", "test_access": False},
        },
        "d14_structure_gate": {
            "binding": {
                "source_identity": {
                    "commit": D14_COMMIT,
                    "tree": D14_TREE,
                },
            },
        },
    }


def test_d15_d14_gate_accepts_distinct_training_and_gate_sources(tmp_path) -> None:
    checkpoint = tmp_path / "terminal_epoch1.pth"
    options = tmp_path / "opts.json"
    checkpoint.write_bytes(b"checkpoint")
    options.write_text("{}\n", encoding="utf-8")
    gate = _d14_gate(checkpoint, options)

    binding = _validate_d14_gate(
        gate,
        checkpoint=checkpoint.resolve(),
        options=options.resolve(),
        checkpoint_sha256=_sha256(checkpoint),
        options_sha256=_sha256(options),
        d14_source_commit=D14_COMMIT,
        d14_source_tree=D14_TREE,
    )

    assert binding["source_identity"] == {
        "commit": D14_COMMIT,
        "tree": D14_TREE,
    }
    with pytest.raises(RuntimeError, match="D1.4 gate source commit mismatch"):
        _validate_d14_gate(
            gate,
            checkpoint=checkpoint.resolve(),
            options=options.resolve(),
            checkpoint_sha256=_sha256(checkpoint),
            options_sha256=_sha256(options),
            d14_source_commit=TRAINING_COMMIT,
            d14_source_tree=TRAINING_TREE,
        )


def test_d15_finalizer_preserves_three_distinct_source_identities() -> None:
    provenance = _validate_source_provenance(
        _source_provenance_scan(),
        training_commit=TRAINING_COMMIT,
        training_tree=TRAINING_TREE,
        d14_source_commit=D14_COMMIT,
        d14_source_tree=D14_TREE,
    )

    assert provenance["training_source_identity"]["commit"] == TRAINING_COMMIT
    assert provenance["d14_source_identity"]["commit"] == D14_COMMIT
    assert provenance["training_source_identity"]["commit"] != (
        provenance["d14_source_identity"]["commit"]
    )


def test_d15_finalizer_fails_closed_on_source_identity_aliasing() -> None:
    scan = _source_provenance_scan()
    aliased = copy.deepcopy(scan)
    aliased["d14_structure_gate"]["binding"]["source_identity"]["commit"] = (
        TRAINING_COMMIT
    )
    with pytest.raises(ValueError, match="D1.5 D1.4 source identity commit mismatch"):
        _validate_source_provenance(
            aliased,
            training_commit=TRAINING_COMMIT,
            training_tree=TRAINING_TREE,
            d14_source_commit=D14_COMMIT,
            d14_source_tree=D14_TREE,
        )

    wrong_training = copy.deepcopy(scan)
    wrong_training["training_source_identity"]["commit"] = D14_COMMIT
    with pytest.raises(ValueError, match="training source identity commit mismatch"):
        _validate_source_provenance(
            wrong_training,
            training_commit=TRAINING_COMMIT,
            training_tree=TRAINING_TREE,
            d14_source_commit=D14_COMMIT,
            d14_source_tree=D14_TREE,
        )


def test_d15_padding_runtime_preserves_existing_active_records() -> None:
    runtime = {
        "padding_prefixes_ignored": torch.tensor([1]),
        "active_count": torch.tensor([3]),
        "birth_count": torch.tensor([0]),
        "end_count": torch.tensor([0]),
        "new_birth_mask": torch.tensor([[False]]),
    }

    _validate_padding_runtime(
        runtime,
        expected_active_count=3,
        label="OF/shadow",
    )

    wrong_active = dict(runtime)
    wrong_active["active_count"] = torch.tensor([0])
    with pytest.raises(RuntimeError, match="padding active_count drifted"):
        _validate_padding_runtime(
            wrong_active,
            expected_active_count=3,
            label="OF/shadow",
        )

    mutated = dict(runtime)
    mutated["birth_count"] = torch.tensor([1])
    with pytest.raises(RuntimeError, match="padding produced birth_count"):
        _validate_padding_runtime(
            mutated,
            expected_active_count=3,
            label="OF/shadow",
        )


def test_d15_trace_integrity_cross_checks_every_route(tmp_path) -> None:
    path = tmp_path / "trace.jsonl.gz"
    rows = [
        _trace_row(channel, route)
        for channel in CHANNELS
        for route in ROUTES
    ]
    _write_trace(path, rows)

    result = _validate_trace(path, _routes())

    assert result["line_count"] == len(CHANNELS) * len(ROUTES)
    assert result["chronological_per_route_video"] is True
    assert result["owner_decisions_unique_per_prefix"] is True
    assert result["transition_counts_cross_checked"] is True


def test_d15_trace_integrity_fails_closed_on_missing_route_row(tmp_path) -> None:
    path = tmp_path / "trace.jsonl.gz"
    rows = [
        _trace_row(channel, route)
        for channel in CHANNELS
        for route in ROUTES
    ][:-1]
    _write_trace(path, rows)

    with pytest.raises(ValueError, match="trace route counts differ"):
        _validate_trace(path, _routes())
