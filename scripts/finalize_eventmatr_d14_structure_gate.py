"""Fail-closed D1.4 cross-arm structure gate.

This gate consumes train-only mechanism receipts, complete predicted-only
terminal scans, and read-only parameter-delta audits.  Artifact corruption or
protocol drift is an error.  A scientifically inactive but otherwise valid arm
is recorded as a failed arm rather than turned into a process failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Optional, Tuple


VARIANT_OBJECTIVES = {
    "normalized_survival": "event_normalized_censored_hazard_v1",
    "decision_aligned_bag": "decision_aligned_interval_bag_v1",
}
COMMON_CONTRACTS = {
    "association_contract": "soft_target_class_log_probability_v1",
    "birth_risk_contract": "event_matched_hard_negative_v1",
}
OFFICIAL_TRAIN_ARTIFACTS = {
    "annotation": {
        "bytes": 1575585,
        "sha256": "8eb3e61cc758bcc08aea1d17cfbf1acb2fed8c2a51ed884116d766e9e4c04e66",
    },
    "proposal_labels": {
        "bytes": 277325536,
        "sha256": "ba7dfb10614cfacd1b26f3d50c2ffa41ae2c7fbce62e946777c217adaeebd22f",
    },
    "train_features": {
        "bytes": 3331932341,
        "sha256": "d4660b31b8c6c00d48b590936b3574ab650423ede9b42016fa1d9dda5d45ac9b",
    },
    "video_len": {
        "bytes": 7063,
        "sha256": "0fcc70d555af6198e7b22850aebe56998e9184c2be3f81a16e7a56667f7b9fd8",
    },
}
EXACT_TERMINAL_COUNTS = {
    "real_prefix_count": 203363,
    "padding_prefix_count": 5917,
    "padding_noop_count": 5917,
    "visible_birth_target_count": 3003,
    "observed_eos_count": 200,
}
SHARED_CENSUS = {
    "d14_epoch_birth_positive_count_total": 3003,
    "d14_epoch_birth_selected_negative_count_total": 3003,
    "d14_epoch_birth_positive_batch_count_total": 1685,
    "d14_epoch_birth_zero_positive_batch_count_total": 1585,
    "d14_epoch_birth_interval_fallback_count_total": 0,
    "d14_epoch_birth_prebirth_exposure_count_total": 180182,
    "d14_epoch_birth_interval_exposure_count_total": 5666,
    "d14_epoch_birth_selected_risk_logit_count_total": 185852,
    "d14_epoch_birth_postinterval_ignored_exposure_count_total": 4,
}
LIFECYCLE_FIELDS = (
    "predicted_start_active_query_count",
    "predicted_birth_query_count",
    "assignment_count",
    "runtime_birth_count",
    "runtime_cancel_count",
    "runtime_end_count",
    "runtime_emit_count",
    "runtime_reacquisition_count",
    "runtime_capacity_exhaustion_count",
)
REQUIRED_LIVE_FIELDS = (
    "predicted_start_active_query_count",
    "predicted_birth_query_count",
    "assignment_count",
    "runtime_birth_count",
    "runtime_cancel_count",
    "runtime_end_count",
    "runtime_emit_count",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, label: str) -> dict:
    if not path.is_file():
        raise ValueError(f"{label} artifact is absent: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} artifact is not a JSON object: {path}")
    return payload


def _expect(payload: dict, expected: dict, label: str) -> None:
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                f"{label} {field} mismatch: {payload.get(field)!r} != {value!r}"
            )


def _require_identity(
    identity: dict,
    *,
    commit: str,
    tree: str,
    label: str,
    require_clean_receipt: bool = False,
) -> None:
    if not isinstance(identity, dict):
        raise ValueError(f"{label} identity is absent")
    if identity.get("commit") != commit or identity.get("tree") != tree:
        raise ValueError(f"{label} identity mismatch")
    if require_clean_receipt and (
        identity.get("status") != "PASS" or identity.get("clean") is not True
    ):
        raise ValueError(f"{label} identity is not PASS/clean")
    smoke = identity.get("smoke")
    if smoke is not None and (
        not isinstance(smoke, dict)
        or smoke.get("status") != "PASS"
        or smoke.get("test_access") is not False
    ):
        raise ValueError(f"{label} identity has no passing no-test smoke receipt")


def _nonnegative_integer(value, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} is not a count: {value!r}")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(numeric) or numeric < 0.0 or not numeric.is_integer():
        raise ValueError(f"{label} is not a finite non-negative integer: {value!r}")
    return int(numeric)


def _positive_delta(group: dict, label: str) -> dict:
    if not isinstance(group, dict):
        raise ValueError(f"{label} parameter group is absent")
    tensor_count = _nonnegative_integer(
        group.get("parameter_tensor_count"), f"{label}.parameter_tensor_count"
    )
    element_count = _nonnegative_integer(
        group.get("element_count"), f"{label}.element_count"
    )
    changed_count = _nonnegative_integer(
        group.get("changed_element_count"), f"{label}.changed_element_count"
    )
    delta_l2 = float(group.get("delta_l2", float("nan")))
    relative_l2 = float(group.get("relative_l2_delta", float("nan")))
    if (
        tensor_count <= 0
        or element_count <= 0
        or changed_count <= 0
        or changed_count > element_count
        or not math.isfinite(delta_l2)
        or delta_l2 <= 0.0
        or not math.isfinite(relative_l2)
        or relative_l2 <= 0.0
    ):
        raise ValueError(f"{label} parameter group did not update")
    return {
        "parameter_tensor_count": tensor_count,
        "element_count": element_count,
        "changed_element_count": changed_count,
        "delta_l2": delta_l2,
        "relative_l2_delta": relative_l2,
    }


def _validate_dataset_artifacts(delta: dict, label: str) -> dict:
    caches = delta.get("dataset_caches")
    if not isinstance(caches, dict):
        raise ValueError(f"{label} parameter audit has no dataset cache receipt")
    result = {}
    for name, expected in OFFICIAL_TRAIN_ARTIFACTS.items():
        row = caches.get(name)
        if not isinstance(row, dict):
            raise ValueError(f"{label} parameter audit omitted {name}")
        for field, value in expected.items():
            if row.get(field) != value:
                raise ValueError(
                    f"{label} is not on the frozen official train artifact: "
                    f"{name}.{field}={row.get(field)!r} != {value!r}"
                )
        result[name] = dict(expected)
    return result


def _validate_mechanism(
    mechanism: dict,
    *,
    variant: str,
    commit: str,
    tree: str,
) -> dict:
    objective = VARIANT_OBJECTIVES[variant]
    _expect(
        mechanism,
        {
            "status": "PASS_TRAIN_MECHANISM_ONLY",
            "protocol": "eventmatr_d14_seed52_decision_alignment_mechanism_v1",
            "variant": variant,
            "lane": "TH",
            "epochs": 1,
            "seed": 52,
            "fresh_start": True,
            "test_access": False,
            "checkpoint_updated": True,
            "strict_causal_paper_result_valid": False,
            "official_paper_performance_valid": False,
            "train_prefix_metrics_diagnostic_only": True,
            "threshold_search": False,
            "threshold_lowering": False,
            "multi_seed": False,
            "official_comparison_release": False,
            "structure_gate_release": False,
            "locked_test_release": False,
        },
        f"{variant} mechanism",
    )
    _expect(
        mechanism.get("contracts", {}),
        {**COMMON_CONTRACTS, "birth_objective_contract": objective},
        f"{variant} mechanism contracts",
    )
    _require_identity(
        mechanism.get("source_identity"),
        commit=commit,
        tree=tree,
        label=f"{variant} mechanism",
    )
    receipts = mechanism.get("source_identity_receipts")
    if not isinstance(receipts, dict):
        raise ValueError(f"{variant} mechanism lacks start/final identity receipts")
    for phase in ("start", "final"):
        _require_identity(
            receipts.get(phase),
            commit=commit,
            tree=tree,
            label=f"{variant} mechanism {phase}",
            require_clean_receipt=True,
        )

    checkpoint = mechanism.get("checkpoint", {})
    _expect(
        checkpoint,
        {
            "epoch": 1,
            "study_protocol": "d14_mechanism",
            "model_variant": "eventmatr",
            "checkpoint_schema": "eventmatr_d14_decision_alignment_mechanism_v1",
            "event_lifecycle_version": "d1_censored",
            "event_d1_lane": "th",
            "owner_state_count": 3,
            "birth_head": "independent_binary_hazard",
            "event_d13_variant": "combined",
            "event_d14_variant": variant,
            "birth_objective_contract": objective,
            **COMMON_CONTRACTS,
        },
        f"{variant} checkpoint",
    )
    for artifact_name in ("checkpoint", "options", "metrics_artifact"):
        row = mechanism.get(artifact_name, {})
        digest = row.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{variant} mechanism {artifact_name} hash is invalid")

    effective = mechanism.get("effective_dose", {})
    effective_expected = {
        "d11_effective_dose_enabled": 1.0,
        "d11_optimizer_step_count": 3270.0,
        "d11_initial_learning_rate": 1e-8,
        "d11_learning_rate_first": 3.34e-6,
        "d11_learning_rate_minimum": 3.34e-6,
        "d11_learning_rate_maximum": 3.34e-6,
        "d11_learning_rate_last": 3.34e-6,
        "d11_scheduler_last_epoch_at_train_start": 1.0,
        "d11_scheduler_t_cur_at_train_start": 1.0,
    }
    for field, expected in effective_expected.items():
        value = float(effective.get(field, float("nan")))
        if not math.isfinite(value) or not math.isclose(
            value, expected, rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError(f"{variant} effective dose drifted: {field}={value}")
    trace = mechanism.get("effective_dose_update_trace", {})
    _expect(
        trace,
        {
            "row_count": 3270,
            "first_optimizer_step": 1,
            "last_optimizer_step": 3270,
            "learning_rate": 3.34e-6,
        },
        f"{variant} update trace",
    )

    census = mechanism.get("mechanism_validation", {}).get("census")
    if not isinstance(census, dict):
        raise ValueError(f"{variant} mechanism census is absent")
    for field, expected in SHARED_CENSUS.items():
        if _nonnegative_integer(census.get(field), f"{variant}.{field}") != expected:
            raise ValueError(f"{variant} mechanism census drifted: {field}")
    prebirth_groups = _nonnegative_integer(
        census.get("d14_epoch_birth_prebirth_group_count_total"),
        f"{variant}.prebirth_groups",
    )
    zero_prebirth_groups = _nonnegative_integer(
        census.get("d14_epoch_birth_zero_prebirth_group_count_total"),
        f"{variant}.zero_prebirth_groups",
    )
    if prebirth_groups + zero_prebirth_groups != 3003:
        raise ValueError(f"{variant} prebirth event groups did not close")
    objective_counts = (
        _nonnegative_integer(
            census.get("d14_epoch_birth_normalized_survival_event_count_total"),
            f"{variant}.normalized_events",
        ),
        _nonnegative_integer(
            census.get("d14_epoch_birth_decision_aligned_positive_bag_count_total"),
            f"{variant}.positive_bags",
        ),
        _nonnegative_integer(
            census.get("d14_epoch_birth_decision_aligned_negative_bag_count_total"),
            f"{variant}.negative_bags",
        ),
    )
    expected_objective_counts = (
        (3003, 0, 0) if variant == "normalized_survival" else (0, 3003, 3003)
    )
    if objective_counts != expected_objective_counts:
        raise ValueError(f"{variant} objective event counts drifted")
    return {
        "checkpoint_sha256": checkpoint["sha256"],
        "options_sha256": mechanism["options"]["sha256"],
        "metrics_sha256": mechanism["metrics_artifact"]["sha256"],
        "census": {field: int(census[field]) for field in SHARED_CENSUS},
    }


def _validate_scan(
    scan: dict,
    *,
    variant: str,
    commit: str,
    tree: str,
    checkpoint_sha256: str,
    options_sha256: str,
) -> dict:
    objective = VARIANT_OBJECTIVES[variant]
    _expect(
        scan,
        {
            "status": "DIAGNOSTIC_COMPLETE",
            "execution_status": "PASS",
            "status_semantics": "scan_completed_not_mechanism_or_performance_pass",
            "protocol": "eventmatr_d14_terminal_association_scan_v1",
            "event_d13_variant": "combined",
            "event_d14_variant": variant,
            **COMMON_CONTRACTS,
            "birth_objective_contract": objective,
            "complete_scan": True,
            "seed": 52,
            "test_access": False,
            "checkpoint_updated": False,
            "strict_causal_paper_result_valid": False,
            "ground_truth_visible_to_model": False,
            "padding_contract_verified": True,
            "eos_semantics": "current_stream_termination_observation_only",
            "performance_gate_release": False,
            "one_epoch_mechanism_gate_status": "PASS_TRAIN_MECHANISM_ONLY",
            "threshold_search": False,
            "videos_scanned": 200,
        },
        f"{variant} terminal scan",
    )
    _require_identity(
        scan.get("source_identity"),
        commit=commit,
        tree=tree,
        label=f"{variant} scan execution",
    )
    _require_identity(
        scan.get("training_source_identity"),
        commit=commit,
        tree=tree,
        label=f"{variant} scan training",
        require_clean_receipt=True,
    )
    if scan.get("checkpoint", {}).get("sha256") != checkpoint_sha256:
        raise ValueError(f"{variant} scan checkpoint hash mismatch")
    if scan.get("options", {}).get("sha256") != options_sha256:
        raise ValueError(f"{variant} scan options hash mismatch")

    counts = scan.get("barrier_counts")
    if not isinstance(counts, dict):
        raise ValueError(f"{variant} scan has no barrier counts")
    for field, expected in EXACT_TERMINAL_COUNTS.items():
        if _nonnegative_integer(counts.get(field), f"{variant}.{field}") != expected:
            raise ValueError(f"{variant} official-train terminal census drifted: {field}")
    lifecycle = {
        field: _nonnegative_integer(counts.get(field), f"{variant}.{field}")
        for field in LIFECYCLE_FIELDS
    }
    if lifecycle["runtime_capacity_exhaustion_count"] != 0:
        raise ValueError(f"{variant} terminal runtime exhausted dynamic capacity")
    if lifecycle["predicted_birth_query_count"] != lifecycle["runtime_birth_count"]:
        raise ValueError(f"{variant} predicted/runtime birth counts did not close")
    if lifecycle["predicted_start_active_query_count"] < lifecycle[
        "predicted_birth_query_count"
    ]:
        raise ValueError(f"{variant} rising-edge births exceed positive birth logits")
    if lifecycle["runtime_emit_count"] != lifecycle["runtime_end_count"]:
        raise ValueError(f"{variant} terminal END and immutable emissions differ")

    integrity = scan.get("lifecycle_integrity")
    expected_integrity = {
        "immutable_ledger_verified": True,
        "positive_length_verified": True,
        "nonnegative_start_verified": True,
        "no_duplicate_event_verified": True,
        "contiguous_sequence_id_verified": True,
        "ledger_emit_count_closed": True,
        "video_ledger_count": 200,
    }
    if not isinstance(integrity, dict):
        raise ValueError(f"{variant} scan has no lifecycle-integrity receipt")
    _expect(integrity, expected_integrity, f"{variant} lifecycle integrity")
    if _nonnegative_integer(
        integrity.get("ledger_row_count"), f"{variant}.ledger_row_count"
    ) != lifecycle["runtime_emit_count"]:
        raise ValueError(f"{variant} ledger row count did not close")

    scores = scan.get("birth_score_diagnostics", {}).get("score_summaries", {})
    all_query = scores.get("all_query_margin")
    interval_bag = scores.get("birth_oracle_path_interval_logmeanexp")
    if not isinstance(all_query, dict) or not isinstance(interval_bag, dict):
        raise ValueError(f"{variant} terminal score diagnostics are incomplete")
    if _nonnegative_integer(
        all_query.get("count"), f"{variant}.all_query_margin.count"
    ) != 2033630:
        raise ValueError(f"{variant} all-query score census drifted")
    positive_logits = _nonnegative_integer(
        all_query.get("positive_count"), f"{variant}.all_query_margin.positive_count"
    )
    if positive_logits != lifecycle["predicted_start_active_query_count"]:
        raise ValueError(f"{variant} positive-logit score/runtime counts disagree")
    if _nonnegative_integer(
        interval_bag.get("count"), f"{variant}.interval_logmeanexp.count"
    ) != 3003:
        raise ValueError(f"{variant} interval-bag diagnostic census drifted")
    all_query_max = float(all_query.get("max", float("nan")))
    interval_bag_max = float(interval_bag.get("max", float("nan")))
    if not math.isfinite(all_query_max) or not math.isfinite(interval_bag_max):
        raise ValueError(f"{variant} terminal score maxima are non-finite")

    checks = {
        "individual_positive_birth_logit": positive_logits > 0,
        **{field: lifecycle[field] > 0 for field in REQUIRED_LIVE_FIELDS},
    }
    return {
        "structure_gate_pass": all(checks.values()),
        "required_liveness_checks": checks,
        "terminal_lifecycle": lifecycle,
        "positive_individual_birth_logit_count": positive_logits,
        "all_query_birth_logit_max": all_query_max,
        "interval_logmeanexp_max": interval_bag_max,
        "lifecycle_integrity": integrity,
    }


def _validate_delta(
    delta: dict,
    *,
    variant: str,
    commit: str,
    tree: str,
    checkpoint_sha256: str,
    options_sha256: str,
    metrics_sha256: str,
) -> dict:
    objective = VARIANT_OBJECTIVES[variant]
    _expect(
        delta,
        {
            "status": "PASS",
            "protocol": "eventmatr_d14_epoch1_parameter_delta_audit_v1",
            "event_d13_variant": "combined",
            "event_d14_variant": variant,
            **COMMON_CONTRACTS,
            "birth_objective_contract": objective,
            "test_access": False,
            "checkpoint_updated": False,
            "model_forward_executed": False,
            "optimizer_step_executed": False,
            "strict_causal_paper_result_valid": False,
            "initialization_reconstructed_from_exact_training_commit": True,
            "initialization_source_commit": commit,
            "seed": 52,
            "construction_order": "seed_then_train_dataset_then_model",
            "initialization_video_count": 200,
            "initialization_dataset_item_count": 209280,
            "initialization_loader_batch_size": 64,
            "initialization_loader_batch_count": 3270,
        },
        f"{variant} parameter audit",
    )
    _require_identity(
        delta.get("training_source_identity"),
        commit=commit,
        tree=tree,
        label=f"{variant} parameter training",
        require_clean_receipt=True,
    )
    _require_identity(
        delta.get("audit_source_identity"),
        commit=commit,
        tree=tree,
        label=f"{variant} parameter execution",
    )
    if delta.get("checkpoint", {}).get("sha256") != checkpoint_sha256:
        raise ValueError(f"{variant} parameter checkpoint hash mismatch")
    if delta.get("options", {}).get("sha256") != options_sha256:
        raise ValueError(f"{variant} parameter options hash mismatch")
    if delta.get("metrics", {}).get("sha256") != metrics_sha256:
        raise ValueError(f"{variant} parameter metrics hash mismatch")
    optimizer = delta.get("optimizer", {})
    if (
        optimizer.get("step_count_closed") is not True
        or _nonnegative_integer(
            optimizer.get("expected_step"), f"{variant}.optimizer.expected_step"
        )
        != 3270
        or _nonnegative_integer(
            optimizer.get("maximum_step"), f"{variant}.optimizer.maximum_step"
        )
        != 3270
    ):
        raise ValueError(f"{variant} optimizer steps did not close")
    groups = delta.get("parameter_delta_by_group")
    if not isinstance(groups, dict):
        raise ValueError(f"{variant} parameter groups are absent")
    selected_groups = {
        name: _positive_delta(groups.get(name), f"{variant}.{name}")
        for name in (
            "event_transition_birth_head",
            "event_transition_four_state_head",
            "event_transition_shared_fuse",
            "event_owner_decoder",
        )
    }
    dataset_artifacts = _validate_dataset_artifacts(delta, variant)
    return {
        "parameter_delta": selected_groups,
        "official_train_artifacts": dataset_artifacts,
    }


def _validate_control(
    mechanism: dict,
    scan: dict,
    delta: dict,
    *,
    commit: str,
    tree: str,
) -> dict:
    _expect(
        mechanism,
        {
            "status": "PASS_TRAIN_MECHANISM_ONLY",
            "protocol": "eventmatr_d13_seed52_factorial_mechanism_v1",
            "variant": "combined",
            "test_access": False,
            "checkpoint_updated": True,
            "strict_causal_paper_result_valid": False,
            "official_paper_performance_valid": False,
            "threshold_search": False,
            "official_comparison_release": False,
            "locked_test_release": False,
        },
        "D1.3 combined control mechanism",
    )
    _require_identity(
        mechanism.get("source_identity"),
        commit=commit,
        tree=tree,
        label="D1.3 control mechanism",
    )
    _expect(
        mechanism.get("contracts", {}),
        COMMON_CONTRACTS,
        "D1.3 control contracts",
    )
    checkpoint_sha256 = mechanism.get("checkpoint", {}).get("sha256")
    options_sha256 = mechanism.get("options", {}).get("sha256")
    metrics_sha256 = mechanism.get("metrics_artifact", {}).get("sha256")
    if not all(
        isinstance(value, str) and len(value) == 64
        for value in (checkpoint_sha256, options_sha256, metrics_sha256)
    ):
        raise ValueError("D1.3 control artifact hashes are invalid")

    _expect(
        scan,
        {
            "status": "DIAGNOSTIC_COMPLETE",
            "execution_status": "PASS",
            "protocol": "eventmatr_d13_terminal_association_scan_v1",
            "event_d13_variant": "combined",
            **COMMON_CONTRACTS,
            "complete_scan": True,
            "seed": 52,
            "test_access": False,
            "checkpoint_updated": False,
            "strict_causal_paper_result_valid": False,
            "ground_truth_visible_to_model": False,
            "performance_gate_release": False,
            "one_epoch_mechanism_gate_status": "PASS_TRAIN_MECHANISM_ONLY",
            "threshold_search": False,
            "videos_scanned": 200,
        },
        "D1.3 combined control scan",
    )
    _require_identity(
        scan.get("source_identity"),
        commit=commit,
        tree=tree,
        label="D1.3 control scan execution",
    )
    _require_identity(
        scan.get("training_source_identity"),
        commit=commit,
        tree=tree,
        label="D1.3 control scan training",
        require_clean_receipt=True,
    )
    if scan.get("checkpoint", {}).get("sha256") != checkpoint_sha256:
        raise ValueError("D1.3 control scan checkpoint hash mismatch")
    if scan.get("options", {}).get("sha256") != options_sha256:
        raise ValueError("D1.3 control scan options hash mismatch")
    counts = scan.get("barrier_counts", {})
    for field, expected in EXACT_TERMINAL_COUNTS.items():
        if _nonnegative_integer(counts.get(field), f"control.{field}") != expected:
            raise ValueError(f"D1.3 control terminal census drifted: {field}")
    zero_fields = (
        "predicted_start_active_query_count",
        "predicted_birth_query_count",
        "assignment_count",
        "runtime_birth_count",
        "runtime_cancel_count",
        "runtime_end_count",
        "runtime_emit_count",
        "runtime_reacquisition_count",
        "runtime_capacity_exhaustion_count",
    )
    for field in zero_fields:
        if _nonnegative_integer(counts.get(field), f"control.{field}") != 0:
            raise ValueError(f"D1.3 control is no longer the frozen zero-live control")
    control_scores = (
        scan.get("birth_score_diagnostics", {})
        .get("score_summaries", {})
        .get("all_query_margin", {})
    )
    if (
        _nonnegative_integer(control_scores.get("count"), "control.score_count")
        != 2033630
        or _nonnegative_integer(
            control_scores.get("positive_count"), "control.positive_logits"
        )
        != 0
    ):
        raise ValueError("D1.3 control terminal score census drifted")

    _expect(
        delta,
        {
            "status": "PASS",
            "protocol": "eventmatr_d13_epoch1_parameter_delta_audit_v1",
            "event_d13_variant": "combined",
            "test_access": False,
            "checkpoint_updated": False,
            "model_forward_executed": False,
            "optimizer_step_executed": False,
            "strict_causal_paper_result_valid": False,
            "initialization_reconstructed_from_exact_training_commit": True,
            "initialization_source_commit": commit,
            "seed": 52,
            "construction_order": "seed_then_train_dataset_then_model",
        },
        "D1.3 combined control parameter audit",
    )
    _require_identity(
        delta.get("training_source_identity"),
        commit=commit,
        tree=tree,
        label="D1.3 control parameter training",
        require_clean_receipt=True,
    )
    _require_identity(
        delta.get("audit_source_identity"),
        commit=commit,
        tree=tree,
        label="D1.3 control parameter execution",
    )
    if delta.get("checkpoint", {}).get("sha256") != checkpoint_sha256:
        raise ValueError("D1.3 control parameter checkpoint hash mismatch")
    if delta.get("options", {}).get("sha256") != options_sha256:
        raise ValueError("D1.3 control parameter options hash mismatch")
    if delta.get("metrics", {}).get("sha256") != metrics_sha256:
        raise ValueError("D1.3 control parameter metrics hash mismatch")
    dataset = _validate_dataset_artifacts(delta, "D1.3 control")
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "zero_terminal_lifecycle_verified": True,
        "zero_individual_positive_birth_logits_verified": True,
        "zero_emission_ledger_invariants_vacuously_satisfied": True,
        "official_train_artifacts": dataset,
    }


def select_d14_structure_variant(
    arm_passes: dict[str, bool],
) -> Tuple[Optional[str], str]:
    if set(arm_passes) != set(VARIANT_OBJECTIVES):
        raise ValueError("D1.4 selection requires exactly the two frozen arms")
    passed = [
        variant for variant in VARIANT_OBJECTIVES if arm_passes[variant] is True
    ]
    if len(passed) == 2:
        return (
            "normalized_survival",
            "both arms passed; pre-registered tie-break retains the formal "
            "interval-censored likelihood",
        )
    if len(passed) == 1:
        return (
            passed[0],
            "only one pre-registered arm passed every structure check",
        )
    return (
        None,
        "neither pre-registered arm passed every structure check",
    )


def validate_d14_structure_gate(
    arms: dict[str, dict],
    control: dict,
    *,
    d14_source_commit: str,
    d14_source_tree: str,
    control_source_commit: str,
    control_source_tree: str,
) -> dict:
    if set(arms) != set(VARIANT_OBJECTIVES):
        raise ValueError("D1.4 structure gate requires exactly the two frozen arms")
    arm_evidence = {}
    shared_census = None
    shared_dataset = None
    for variant in VARIANT_OBJECTIVES:
        bundle = arms[variant]
        mechanism = _validate_mechanism(
            bundle["mechanism"],
            variant=variant,
            commit=d14_source_commit,
            tree=d14_source_tree,
        )
        scan = _validate_scan(
            bundle["scan"],
            variant=variant,
            commit=d14_source_commit,
            tree=d14_source_tree,
            checkpoint_sha256=mechanism["checkpoint_sha256"],
            options_sha256=mechanism["options_sha256"],
        )
        delta = _validate_delta(
            bundle["delta"],
            variant=variant,
            commit=d14_source_commit,
            tree=d14_source_tree,
            checkpoint_sha256=mechanism["checkpoint_sha256"],
            options_sha256=mechanism["options_sha256"],
            metrics_sha256=mechanism["metrics_sha256"],
        )
        if shared_census is None:
            shared_census = mechanism["census"]
            shared_dataset = delta["official_train_artifacts"]
        elif mechanism["census"] != shared_census:
            raise ValueError("D1.4 cross-arm shared mechanism census differs")
        elif delta["official_train_artifacts"] != shared_dataset:
            raise ValueError("D1.4 cross-arm official train artifacts differ")
        arm_evidence[variant] = {
            **scan,
            "parameter_delta": delta["parameter_delta"],
            "checkpoint_sha256": mechanism["checkpoint_sha256"],
        }

    control_evidence = _validate_control(
        control["mechanism"],
        control["scan"],
        control["delta"],
        commit=control_source_commit,
        tree=control_source_tree,
    )
    if control_evidence["official_train_artifacts"] != shared_dataset:
        raise ValueError("D1.4 arms and D1.3 control used different train artifacts")

    selected, selection_reason = select_d14_structure_variant(
        {
            variant: arm_evidence[variant]["structure_gate_pass"]
            for variant in VARIANT_OBJECTIVES
        }
    )
    return {
        "status": "PASS_STRUCTURE_GATE" if selected is not None else "FAIL_STRUCTURE_GATE",
        "selected_variant": selected,
        "selection_reason": selection_reason,
        "effect_size_threshold": None,
        "threshold_search": False,
        "threshold_lowering": False,
        "test_access": False,
        "multi_seed": False,
        "raw_rgb_training": False,
        "strict_causal_paper_result_valid": False,
        "official_paper_performance_valid": False,
        "train_only_mechanism_evidence": True,
        "development_pilot_contract_eligible_for_freeze": selected is not None,
        "official_comparison_release": False,
        "locked_test_release": False,
        "paper_claim_release": False,
        "selection_rule": (
            "one passing arm selects itself; two passing arms select "
            "normalized_survival; zero passing arms select none"
        ),
        "arms": arm_evidence,
        "control": control_evidence,
        "official_train_artifacts": shared_dataset,
    }


def _validate_linked_artifact(row: dict, label: str) -> dict:
    if not isinstance(row, dict):
        raise ValueError(f"{label} artifact reference is absent")
    path_value = row.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{label} artifact path is absent")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{label} linked artifact is absent: {path}")
    size = _nonnegative_integer(row.get("bytes"), f"{label}.bytes")
    if path.stat().st_size != size:
        raise ValueError(f"{label} linked artifact size mismatch")
    digest = row.get("sha256")
    if _sha256(path) != digest:
        raise ValueError(f"{label} linked artifact hash mismatch")
    return {"path": str(path), "bytes": size, "sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser()
    for prefix in ("normalized", "bag", "control"):
        parser.add_argument(f"--{prefix}-mechanism", required=True, type=Path)
        parser.add_argument(f"--{prefix}-scan", required=True, type=Path)
        parser.add_argument(f"--{prefix}-delta", required=True, type=Path)
    parser.add_argument("--d14-source-commit", required=True)
    parser.add_argument("--d14-source-tree", required=True)
    parser.add_argument("--control-source-commit", required=True)
    parser.add_argument("--control-source-tree", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    input_paths = {}
    payloads = {}
    for prefix in ("normalized", "bag", "control"):
        payloads[prefix] = {}
        for artifact in ("mechanism", "scan", "delta"):
            path = getattr(args, f"{prefix}_{artifact}").expanduser().resolve()
            input_paths[f"{prefix}_{artifact}"] = path
            payloads[prefix][artifact] = _load(path, f"{prefix} {artifact}")

    arms = {
        "normalized_survival": payloads["normalized"],
        "decision_aligned_bag": payloads["bag"],
    }
    evidence = validate_d14_structure_gate(
        arms,
        payloads["control"],
        d14_source_commit=args.d14_source_commit,
        d14_source_tree=args.d14_source_tree,
        control_source_commit=args.control_source_commit,
        control_source_tree=args.control_source_tree,
    )
    linked_artifacts = {}
    for prefix in ("normalized", "bag", "control"):
        mechanism = payloads[prefix]["mechanism"]
        linked_artifacts[prefix] = {
            name: _validate_linked_artifact(
                mechanism[name],
                f"{prefix} {name}",
            )
            for name in (
                "checkpoint",
                "options",
                "metrics_artifact",
                "effective_dose_update_trace",
            )
        }

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "protocol": "eventmatr_d14_cross_arm_structure_gate_v1",
        **evidence,
        "source_identity": {
            "d14": {
                "commit": args.d14_source_commit,
                "tree": args.d14_source_tree,
            },
            "control": {
                "commit": args.control_source_commit,
                "tree": args.control_source_tree,
            },
        },
        "input_artifacts": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in input_paths.items()
        },
        "linked_training_artifacts": linked_artifacts,
    }
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "selected_variant": receipt["selected_variant"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
