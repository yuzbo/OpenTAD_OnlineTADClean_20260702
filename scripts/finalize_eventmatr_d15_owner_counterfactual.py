"""Fail-closed finalizer for the EventMATR D1.5 owner counterfactual.

The receipt validates execution integrity first, then applies the prospectively
frozen structural routing table.  ``PASS_DIAGNOSTIC`` means only that the
counterfactual is trustworthy enough to route the next model repair.  It is
never a model, performance, official-comparison, or paper gate.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


PROTOCOL = "eventmatr_d15_owner_counterfactual_v1"
FINALIZER_PROTOCOL = "eventmatr_d15_owner_counterfactual_finalizer_v1"
CHANNELS = ("PF", "PR", "OF", "OR")
ROUTES = ("formal", "shadow")
CHANNEL_FACTORS = {
    "PF": ("predicted", "free"),
    "PR": ("predicted", "refreshed"),
    "OF": ("oracle_visible", "free"),
    "OR": ("oracle_visible", "refreshed"),
}
QUERY_COMPATIBILITY_RULE = {
    "birth": (
        "target_class_log_probability_plus_birth_logsigmoid_minus_"
        "absolute_candidate_start_error_over_segment_size"
    ),
    "visible_refresh": (
        "target_class_log_probability_minus_absolute_candidate_start_"
        "error_over_segment_size"
    ),
    "record_target_link": (
        "owner_current_query_cosine_plus_record_target_class_log_"
        "probability_minus_start_error_over_segment_size"
    ),
    "assignment": "deterministic_greedy_one_to_one_no_effect_threshold",
    "target_query_tie_break": "target_event_id_then_query_index",
    "record_target_tie_break": "runtime_event_id_then_target_event_id",
}
EXPECTED_D14_COUNTS = {
    "birth_count": 30002,
    "cancellation_count": 29974,
    "end_count": 0,
    "emit_count": 0,
    "reacquisition_count": 0,
    "capacity_exhaustion_count": 0,
}
EXPECTED_CENSUS = {
    "real_prefix_count": 203363,
    "padding_prefix_count": 5917,
    "visible_birth_target_count": 3003,
    "observed_eos_count": 200,
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
MANDATORY_SCAN_FLAGS = {
    "status": "DIAGNOSTIC_COMPLETE",
    "execution_status": "PASS",
    "status_semantics": (
        "counterfactual_completed_not_model_or_performance_gate_pass"
    ),
    "protocol": PROTOCOL,
    "complete_scan": True,
    "seed": 52,
    "train_only": True,
    "counterfactual": True,
    "paper_performance_valid": False,
    "strict_causal_paper_result_valid": False,
    "ground_truth_visible_to_query_backbone": False,
    "ground_truth_visible_to_owner_intervention": True,
    "ground_truth_stored_in_runtime_record": False,
    "optimizer_constructed": False,
    "optimizer_step_count": 0,
    "checkpoint_updated": False,
    "threshold_search": False,
    "test_access": False,
    "locked_test_release": False,
    "official_comparison_release": False,
    "single_forward_query_stream_shared_across_channels": True,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, label: str) -> dict:
    if not path.is_file():
        raise ValueError(f"{label} is absent: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not a JSON object")
    return payload


def _expect(payload: dict, expected: dict, label: str) -> None:
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"{label} {key} mismatch: {payload.get(key)!r} != {value!r}"
            )


def _nonnegative_integer(value, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} is not a count")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric") from error
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise ValueError(f"{label} is not a finite non-negative integer")
    return int(numeric)


def _valid_sha256(value, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is not a lower-case SHA-256")
    return value


def _validate_identity(
    payload: dict,
    *,
    commit: str,
    tree: str,
    manifest_sha256: str,
    label: str,
) -> dict:
    _expect(
        payload,
        {
            "status": "PASS",
            "clean": True,
            "status_porcelain": "",
            "commit": commit,
            "tree": tree,
            "manifest_sha256": manifest_sha256,
        },
        label,
    )
    return {
        "commit": commit,
        "tree": tree,
        "manifest_sha256": manifest_sha256,
        "clean": True,
    }


def _validate_linked_file(row: dict, label: str) -> dict:
    if not isinstance(row, dict):
        raise ValueError(f"{label} reference is absent")
    path = Path(row.get("path", "")).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{label} linked file is absent: {path}")
    size = _nonnegative_integer(row.get("bytes"), f"{label}.bytes")
    digest = _valid_sha256(row.get("sha256"), f"{label}.sha256")
    if path.stat().st_size != size:
        raise ValueError(f"{label} linked size mismatch")
    if _sha256(path) != digest:
        raise ValueError(f"{label} linked hash mismatch")
    return {"path": str(path), "bytes": size, "sha256": digest}


def _validate_integrity(row: dict, label: str) -> dict:
    expected = {
        "immutable_ledger_verified": True,
        "positive_length_verified": True,
        "nonnegative_start_verified": True,
        "no_duplicate_event_verified": True,
        "contiguous_sequence_id_verified": True,
        "ledger_emit_count_closed": True,
        "ground_truth_stored_in_runtime_record": False,
    }
    _expect(row, expected, f"{label} lifecycle integrity")
    ledger_rows = _nonnegative_integer(
        row.get("ledger_row_count"), f"{label}.ledger_row_count"
    )
    video_ledgers = _nonnegative_integer(
        row.get("video_ledger_count"), f"{label}.video_ledger_count"
    )
    if video_ledgers != 200:
        raise ValueError(f"{label} did not audit all 200 video ledgers")
    return {
        **expected,
        "ledger_row_count": ledger_rows,
        "video_ledger_count": video_ledgers,
    }


def _validate_route(
    row: dict,
    *,
    channel: str,
    route: str,
    query_hash: str,
    consumption_hash: str,
) -> dict:
    _expect(
        row,
        {
            "channel": channel,
            "route": route,
            "admission": CHANNEL_FACTORS[channel][0],
            "identity": CHANNEL_FACTORS[channel][1],
            "query_stream_sha256": query_hash,
            "query_consumption_sha256": consumption_hash,
        },
        f"{channel}/{route}",
    )
    counts = row.get("counts")
    if not isinstance(counts, dict):
        raise ValueError(f"{channel}/{route} counts are absent")
    parsed = {
        name: _nonnegative_integer(
            counts.get(name), f"{channel}/{route}.{name}"
        )
        for name in (
            "real_prefix_count",
            "padding_prefix_count",
            "padding_noop_count",
            "observed_eos_count",
            "birth_count",
            "cancellation_count",
            "end_count",
            "emit_count",
            "reacquisition_count",
            "capacity_exhaustion_count",
            "records_active_after_eos",
            "videos_with_active_records_after_eos",
            "association_count",
            "late_association_count",
            "owner_refresh_count",
            "record_target_exact_tie_count",
            "query_prefix_consumption_count",
            "decision_row_count",
            "end_argmax_count",
            "end_min_duration_suppression_count",
            "target_backed_end_transition_count",
            "target_backed_end_within_one_segment_count",
            "end_at_observed_target_end_count",
            "end_without_emission_count",
        )
    }
    for name in ("real_prefix_count", "padding_prefix_count", "observed_eos_count"):
        if parsed[name] != EXPECTED_CENSUS[name]:
            raise ValueError(f"{channel}/{route} complete census drifted: {name}")
    if parsed["padding_noop_count"] != parsed["padding_prefix_count"]:
        raise ValueError(f"{channel}/{route} padding no-op accounting failed")
    if parsed["query_prefix_consumption_count"] != (
        parsed["real_prefix_count"] + parsed["padding_prefix_count"]
    ):
        raise ValueError(f"{channel}/{route} query-consumption accounting failed")
    if parsed["capacity_exhaustion_count"] != 0:
        raise ValueError(f"{channel}/{route} exhausted dynamic capacity")
    if parsed["reacquisition_count"] != 0:
        raise ValueError(
            f"{channel}/{route} unexpectedly used exact-ID reacquisition"
        )
    if parsed["end_without_emission_count"] != max(
        0, parsed["end_count"] - parsed["emit_count"]
    ):
        raise ValueError(f"{channel}/{route} end/emission accounting drifted")
    unique_records = _nonnegative_integer(
        row.get("unique_runtime_record_count"),
        f"{channel}/{route}.unique_runtime_record_count",
    )
    if unique_records != parsed["birth_count"]:
        raise ValueError(f"{channel}/{route} birth/unique-record count differs")
    linked_targets = _nonnegative_integer(
        row.get("linked_unique_target_count"),
        f"{channel}/{route}.linked_unique_target_count",
    )
    raw_duplicates = _nonnegative_integer(
        row.get("raw_semantic_duplicate_count"),
        f"{channel}/{route}.raw_semantic_duplicate_count",
    )
    fragments = _nonnegative_integer(
        row.get("batch_local_owner_decision_fragment_count"),
        f"{channel}/{route}.batch_local_owner_decision_fragment_count",
    )
    if fragments != parsed["decision_row_count"]:
        raise ValueError(f"{channel}/{route} decision fragment accounting differs")
    integrity = _validate_integrity(
        row.get("lifecycle_integrity", {}), f"{channel}/{route}"
    )
    if integrity["ledger_row_count"] != parsed["emit_count"]:
        raise ValueError(f"{channel}/{route} ledger/emission count differs")
    for stats_name, stats in row.get("statistics", {}).items():
        if not isinstance(stats, dict):
            raise ValueError(f"{channel}/{route} statistic {stats_name} is malformed")
        for key, value in stats.items():
            if key == "count":
                _nonnegative_integer(
                    value, f"{channel}/{route}.{stats_name}.count"
                )
            elif not math.isfinite(float(value)):
                raise ValueError(
                    f"{channel}/{route} statistic {stats_name}.{key} is non-finite"
                )
    return {
        "counts": parsed,
        "unique_runtime_record_count": unique_records,
        "linked_unique_target_count": linked_targets,
        "raw_semantic_duplicate_count": raw_duplicates,
        "lifecycle_integrity": integrity,
        "first_owner_decision_state_counts": row.get(
            "first_owner_decision_state_counts", {}
        ),
        "owner_state_winner_counts": row.get("owner_state_winner_counts", {}),
        "transition_counts": row.get("transition_counts", {}),
        "statistics": row.get("statistics", {}),
    }


def _validate_trace(path: Path, routes: dict[str, dict[str, dict]]) -> dict:
    required_fields = {
        "protocol",
        "channel",
        "route",
        "video_name",
        "runtime_event_id",
        "creation_frame",
        "current_frame",
        "first_owner_decision",
        "owner_query_before_refresh",
        "owner_query_after_refresh",
        "formal_owner_query_id_after_intervention",
        "formal_owner_query_id_unchanged",
        "matched_current_query",
        "owner_attention_top_query",
        "owner_attention_entropy",
        "cancel_logit",
        "continue_logit",
        "end_logit",
        "cancel_margin",
        "continue_margin",
        "end_margin",
        "state_argmax",
        "end_suppressed_by_minimum_duration",
        "formal_transition",
        "shadow_transition",
        "target_end_observable_now",
        "frames_from_annotated_end",
    }
    expected_counts = {
        (channel, route): routes[channel][route]["counts"]["decision_row_count"]
        for channel in CHANNELS
        for route in ROUTES
    }
    route_counts: Counter = Counter()
    end_argmax_counts: Counter = Counter()
    suppressed_end_counts: Counter = Counter()
    target_backed_end_counts: Counter = Counter()
    last_frame: dict[tuple[str, str, str], float] = {}
    event_ids_at_frame: dict[tuple[str, str, str], set[int]] = {}
    line_count = 0

    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"D1.5 trace line {line_number} is not valid JSON"
                    ) from error
                if not isinstance(row, dict):
                    raise ValueError(f"D1.5 trace line {line_number} is not an object")
                missing = required_fields.difference(row)
                if missing:
                    raise ValueError(
                        f"D1.5 trace line {line_number} omitted {sorted(missing)!r}"
                    )
                if row["protocol"] != PROTOCOL:
                    raise ValueError(f"D1.5 trace line {line_number} protocol drifted")
                channel = row["channel"]
                route = row["route"]
                if channel not in CHANNELS or route not in ROUTES:
                    raise ValueError(f"D1.5 trace line {line_number} route is invalid")
                if not isinstance(row["video_name"], str) or not row["video_name"]:
                    raise ValueError(
                        f"D1.5 trace line {line_number} video name is invalid"
                    )
                event_id = _nonnegative_integer(
                    row["runtime_event_id"],
                    f"trace[{line_number}].runtime_event_id",
                )
                frame = float(row["current_frame"])
                creation_frame = float(row["creation_frame"])
                if not math.isfinite(frame) or not math.isfinite(creation_frame):
                    raise ValueError(
                        f"D1.5 trace line {line_number} has a non-finite frame"
                    )
                if creation_frame < 0 or frame < creation_frame:
                    raise ValueError(
                        f"D1.5 trace line {line_number} has an invalid lifetime"
                    )
                key = (channel, route)
                stream_key = (channel, route, row["video_name"])
                previous = last_frame.get(stream_key)
                if previous is not None and frame < previous:
                    raise ValueError(
                        f"D1.5 trace line {line_number} is not chronological"
                    )
                if previous is None or frame > previous:
                    last_frame[stream_key] = frame
                    event_ids_at_frame[stream_key] = set()
                if event_id in event_ids_at_frame[stream_key]:
                    raise ValueError(
                        f"D1.5 trace line {line_number} duplicates one owner decision"
                    )
                event_ids_at_frame[stream_key].add(event_id)
                winner = _nonnegative_integer(
                    row["state_argmax"], f"trace[{line_number}].state_argmax"
                )
                if winner not in (0, 1, 2):
                    raise ValueError(
                        f"D1.5 trace line {line_number} state winner is invalid"
                    )
                if row["formal_owner_query_id_unchanged"] is not True:
                    raise ValueError(
                        f"D1.5 trace line {line_number} mutated formal owner identity"
                    )
                if int(row["formal_owner_query_id_after_intervention"]) != int(
                    row["owner_query_before_refresh"]
                ):
                    raise ValueError(
                        f"D1.5 trace line {line_number} owner identity audit drifted"
                    )
                transition_field = (
                    "formal_transition" if route == "formal" else "shadow_transition"
                )
                other_transition_field = (
                    "shadow_transition" if route == "formal" else "formal_transition"
                )
                transition = row[transition_field]
                if not isinstance(transition, str) or row[other_transition_field] is not None:
                    raise ValueError(
                        f"D1.5 trace line {line_number} transition fields drifted"
                    )
                transition_parts = set(transition.split("+"))
                route_counts[key] += 1
                end_argmax_counts[key] += int(winner == 2)
                suppressed_end_counts[key] += int(
                    row["end_suppressed_by_minimum_duration"] is True
                )
                target_backed_end_counts[key] += int(
                    "end" in transition_parts
                    and row.get("diagnostic_target_event_id") is not None
                )
                line_count += 1
    except (OSError, UnicodeError) as error:
        raise ValueError("D1.5 trace cannot be decoded as gzip JSONL") from error

    observed_counts = {
        key: route_counts[key] for key in expected_counts
    }
    if observed_counts != expected_counts:
        raise ValueError(
            f"D1.5 trace route counts differ: {observed_counts!r} "
            f"!= {expected_counts!r}"
        )
    for channel in CHANNELS:
        for route in ROUTES:
            key = (channel, route)
            counts = routes[channel][route]["counts"]
            if end_argmax_counts[key] != counts["end_argmax_count"]:
                raise ValueError(f"D1.5 trace {channel}/{route} end winners differ")
            if (
                suppressed_end_counts[key]
                != counts["end_min_duration_suppression_count"]
            ):
                raise ValueError(
                    f"D1.5 trace {channel}/{route} suppressed ends differ"
                )
            if (
                target_backed_end_counts[key]
                != counts["target_backed_end_transition_count"]
            ):
                raise ValueError(
                    f"D1.5 trace {channel}/{route} target-backed ends differ"
                )
    return {
        "line_count": line_count,
        "route_line_counts": {
            f"{channel}/{route}": route_counts[(channel, route)]
            for channel in CHANNELS
            for route in ROUTES
        },
        "chronological_per_route_video": True,
        "owner_decisions_unique_per_prefix": True,
        "transition_counts_cross_checked": True,
    }


def _validate_positive_control(row: dict) -> dict:
    if not isinstance(row, dict):
        raise ValueError("positive lifecycle control is absent")
    counts = row.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("positive lifecycle control counts are absent")
    parsed = {
        name: _nonnegative_integer(
            counts.get(name), f"positive_control.{name}"
        )
        for name in (
            "real_prefix_count",
            "padding_prefix_count",
            "padding_noop_count",
            "observed_eos_count",
            "birth_count",
            "cancellation_count",
            "end_count",
            "emit_count",
            "reacquisition_count",
            "capacity_exhaustion_count",
            "records_active_after_eos",
            "videos_with_active_records_after_eos",
        )
    }
    expected_events = EXPECTED_CENSUS["visible_birth_target_count"]
    expected = {
        "real_prefix_count": EXPECTED_CENSUS["real_prefix_count"],
        "padding_prefix_count": EXPECTED_CENSUS["padding_prefix_count"],
        "padding_noop_count": EXPECTED_CENSUS["padding_prefix_count"],
        "observed_eos_count": EXPECTED_CENSUS["observed_eos_count"],
        "birth_count": expected_events,
        "cancellation_count": 0,
        "end_count": expected_events,
        "emit_count": expected_events,
        "reacquisition_count": 0,
        "capacity_exhaustion_count": 0,
        "records_active_after_eos": 0,
        "videos_with_active_records_after_eos": 0,
    }
    if parsed != expected:
        raise ValueError(
            f"positive lifecycle control did not close: {parsed!r} != {expected!r}"
        )
    admitted = _nonnegative_integer(
        row.get("admitted_unique_target_count"),
        "positive_control.admitted_unique_target_count",
    )
    links = _nonnegative_integer(
        row.get("sidecar_link_count"), "positive_control.sidecar_link_count"
    )
    if admitted != expected_events or links != expected_events:
        raise ValueError("positive lifecycle control target coverage did not close")
    integrity = _validate_integrity(
        row.get("lifecycle_integrity", {}), "positive_control"
    )
    if integrity["ledger_row_count"] != expected_events:
        raise ValueError("positive lifecycle control ledger count differs")
    return {
        "counts": parsed,
        "admitted_unique_target_count": admitted,
        "sidecar_link_count": links,
        "lifecycle_integrity": integrity,
    }


def _route_next_repair(routes: dict[str, dict[str, dict]]) -> dict:
    formal = {
        channel: routes[channel]["formal"]["counts"] for channel in CHANNELS
    }
    shadow = {
        channel: routes[channel]["shadow"]["counts"] for channel in CHANNELS
    }

    # State-machine defects precede model-objective interpretation.
    for channel in CHANNELS:
        if formal[channel]["end_count"] > formal[channel]["emit_count"]:
            return {
                "diagnosis": "end_to_emission_closure_fault",
                "authorized_next_intervention": "ledger_state_machine_repair_only",
                "model_training_authorized": False,
                "reason": f"{channel} ended records without matching emissions",
            }
    for channel in CHANNELS:
        if (
            formal[channel]["end_argmax_count"] > 0
            and formal[channel]["end_count"] == 0
        ):
            return {
                "diagnosis": "runtime_end_gate_or_order_fault",
                "authorized_next_intervention": (
                    "minimum_duration_mask_identity_or_step_order_repair_only"
                ),
                "model_training_authorized": False,
                "reason": f"{channel} had end winners but no runtime end",
            }

    pr_closes = formal["PR"]["end_count"] > 0 and formal["PR"]["emit_count"] > 0
    of_closes = formal["OF"]["end_count"] > 0 and formal["OF"]["emit_count"] > 0
    or_closes = formal["OR"]["end_count"] > 0 and formal["OR"]["emit_count"] > 0

    if pr_closes and not of_closes:
        return {
            "diagnosis": "identity_transport_is_sufficient_under_predicted_burden",
            "authorized_next_intervention": (
                "training_only_predicted_track_supervision_bridge"
            ),
            "model_training_authorized": True,
            "reason": "PR restored frozen-owner end/emission while OF did not",
        }
    if of_closes and not pr_closes:
        return {
            "diagnosis": "clean_admission_is_sufficient_without_continuous_refresh",
            "authorized_next_intervention": (
                "score_independent_sampling_corrected_birth_admission"
            ),
            "model_training_authorized": True,
            "reason": "OF restored end/emission while PR did not",
        }
    if or_closes and not pr_closes and not of_closes:
        return {
            "diagnosis": "admission_identity_interaction",
            "authorized_next_intervention": (
                "one_prospectively_registered_two_factor_repair"
            ),
            "model_training_authorized": True,
            "reason": "only OR restored frozen-owner end/emission",
        }
    if pr_closes and of_closes:
        return {
            "diagnosis": "multiple_single_factor_sufficiency_ambiguous",
            "authorized_next_intervention": (
                "no_model_repair_until_predeclared_disambiguation"
            ),
            "model_training_authorized": False,
            "reason": (
                "identity refresh and clean admission each restored lifecycle; "
                "the frozen matrix does not identify one unique repair"
            ),
        }

    shadow_near_end = {
        channel: shadow[channel][
            "target_backed_end_within_one_segment_count"
        ]
        for channel in CHANNELS
    }
    if any(value > 0 for value in shadow_near_end.values()):
        return {
            "diagnosis": "early_cancellation_truncates_later_end_decisions",
            "authorized_next_intervention": (
                "separate_unresolved_identity_from_false_track_cancel_or_hold_state"
            ),
            "model_training_authorized": True,
            "reason": (
                "formal channels failed while recurrent no-cancel shadows produced "
                "target-backed end decisions within one segment of observed ends"
            ),
            "shadow_near_end_counts": shadow_near_end,
        }

    if all(
        formal[channel]["end_count"] == 0
        and shadow[channel]["end_count"] == 0
        for channel in ("OF", "OR")
    ):
        return {
            "diagnosis": "frozen_owner_end_representation_or_risk_objective_insufficient",
            "authorized_next_intervention": (
                "segment_flag_read_only_diagnostic_then_policy_independent_"
                "three_state_competing_risk_objective"
            ),
            "model_training_authorized": False,
            "reason": "OF/OR and their recurrent shadows never produced an end",
        }
    return {
        "diagnosis": "factorial_result_not_uniquely_routable",
        "authorized_next_intervention": "no_model_repair_until_diagnostic_refinement",
        "model_training_authorized": False,
        "reason": "observed cells do not match one prospectively frozen routing row",
    }


def validate_scan(
    scan: dict,
    *,
    diagnostic_commit: str,
    diagnostic_tree: str,
    manifest_sha256: str,
    checkpoint_sha256: str,
    options_sha256: str,
    d14_gate_sha256: str,
) -> dict:
    _expect(scan, MANDATORY_SCAN_FLAGS, "D1.5 scan")
    _expect(
        scan.get("source_identity", {}),
        {
            "commit": diagnostic_commit,
            "tree": diagnostic_tree,
            "clean_start_and_final": True,
        },
        "D1.5 scan source identity",
    )
    _expect(
        scan.get("manifest", {}),
        {"sha256": manifest_sha256},
        "D1.5 manifest",
    )
    if scan.get("checkpoint", {}).get("sha256") != checkpoint_sha256:
        raise ValueError("D1.5 checkpoint hash mismatch")
    if scan.get("options", {}).get("sha256") != options_sha256:
        raise ValueError("D1.5 options hash mismatch")
    if scan.get("d14_structure_gate", {}).get("sha256") != d14_gate_sha256:
        raise ValueError("D1.5 D1.4-gate hash mismatch")
    _expect(
        scan.get("query_compatibility_rule", {}),
        QUERY_COMPATIBILITY_RULE,
        "D1.5 query compatibility rule",
    )
    linked_manifest = _validate_linked_file(
        scan.get("manifest", {}), "D1.5 manifest"
    )
    linked_checkpoint = _validate_linked_file(
        scan.get("checkpoint", {}), "D1.5 checkpoint"
    )
    linked_options = _validate_linked_file(
        scan.get("options", {}), "D1.5 options"
    )
    linked_d14_gate = _validate_linked_file(
        scan.get("d14_structure_gate", {}), "D1.4 structure gate"
    )
    if linked_manifest["sha256"] != manifest_sha256:
        raise ValueError("linked D1.5 manifest hash mismatch")
    if linked_checkpoint["sha256"] != checkpoint_sha256:
        raise ValueError("linked D1.5 checkpoint hash mismatch")
    if linked_options["sha256"] != options_sha256:
        raise ValueError("linked D1.5 options hash mismatch")
    if linked_d14_gate["sha256"] != d14_gate_sha256:
        raise ValueError("linked D1.4 structure-gate hash mismatch")
    d14_binding = scan.get("d14_structure_gate", {}).get("binding", {})
    expected_d14_lifecycle = {
        "runtime_birth_count": EXPECTED_D14_COUNTS["birth_count"],
        "runtime_cancel_count": EXPECTED_D14_COUNTS["cancellation_count"],
        "runtime_end_count": EXPECTED_D14_COUNTS["end_count"],
        "runtime_emit_count": EXPECTED_D14_COUNTS["emit_count"],
        "runtime_reacquisition_count": EXPECTED_D14_COUNTS[
            "reacquisition_count"
        ],
        "runtime_capacity_exhaustion_count": EXPECTED_D14_COUNTS[
            "capacity_exhaustion_count"
        ],
    }
    if d14_binding.get("terminal_lifecycle") != expected_d14_lifecycle:
        raise ValueError("D1.5 D1.4 terminal lifecycle binding drifted")

    global_census = scan.get("global_census")
    if not isinstance(global_census, dict):
        raise ValueError("D1.5 global census is absent")
    for key, expected in EXPECTED_CENSUS.items():
        if _nonnegative_integer(
            global_census.get(key), f"global_census.{key}"
        ) != expected:
            raise ValueError(f"D1.5 global census drifted: {key}")

    query_hash = _valid_sha256(
        scan.get("query_stream_sha256"), "D1.5 query stream"
    )
    consumption_hash = _valid_sha256(
        scan.get("route_consumption_stream_sha256"),
        "D1.5 route-consumption stream",
    )
    channels = scan.get("channels")
    if not isinstance(channels, dict) or set(channels) != set(CHANNELS):
        raise ValueError("D1.5 requires exactly PF/PR/OF/OR channels")
    routes = {
        channel: {
            route: _validate_route(
                channels[channel].get(route, {}),
                channel=channel,
                route=route,
                query_hash=query_hash,
                consumption_hash=consumption_hash,
            )
            for route in ROUTES
        }
        for channel in CHANNELS
    }
    pf = routes["PF"]["formal"]["counts"]
    for key, expected in EXPECTED_D14_COUNTS.items():
        if pf[key] != expected:
            raise ValueError(f"PF did not reproduce D1.4 exactly: {key}")
    if routes["PF"]["shadow"]["counts"]["birth_count"] != EXPECTED_D14_COUNTS[
        "birth_count"
    ]:
        raise ValueError("PF shadow did not preserve the predicted admission stream")

    expected_targets = EXPECTED_CENSUS["visible_birth_target_count"]
    for channel in ("OF", "OR"):
        for route in ROUTES:
            evidence = routes[channel][route]
            if evidence["unique_runtime_record_count"] != expected_targets:
                raise ValueError(f"{channel}/{route} oracle admission coverage differs")
            if evidence["linked_unique_target_count"] != expected_targets:
                raise ValueError(f"{channel}/{route} target sidecar coverage differs")
            if evidence["raw_semantic_duplicate_count"] != 0:
                raise ValueError(f"{channel}/{route} duplicated an oracle target")

    positive = _validate_positive_control(
        scan.get("positive_lifecycle_control", {})
    )
    trace = _validate_linked_file(scan.get("trace_artifact", {}), "D1.5 trace")
    if scan.get("trace_artifact", {}).get("format") != "chronological_jsonl":
        raise ValueError("D1.5 trace format drifted")
    if scan.get("trace_artifact", {}).get("compression") != "gzip_level_1":
        raise ValueError("D1.5 trace compression drifted")
    trace_integrity = _validate_trace(Path(trace["path"]), routes)

    artifacts = scan.get("official_train_artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("D1.5 official train artifact receipt is absent")
    artifact_receipts = {}
    for name, expected in OFFICIAL_TRAIN_ARTIFACTS.items():
        row = artifacts.get(name)
        if not isinstance(row, dict):
            raise ValueError(f"D1.5 omitted official artifact {name}")
        _expect(row, expected, f"official artifact {name}")
        artifact_receipts[name] = _validate_linked_file(
            row, f"official artifact {name}"
        )

    decision = _route_next_repair(routes)
    return {
        "query_stream_sha256": query_hash,
        "route_consumption_stream_sha256": consumption_hash,
        "global_census": {
            key: int(global_census[key]) for key in EXPECTED_CENSUS
        },
        "routes": routes,
        "positive_lifecycle_control": positive,
        "trace_artifact": trace,
        "trace_integrity": trace_integrity,
        "linked_inputs": {
            "manifest": linked_manifest,
            "checkpoint": linked_checkpoint,
            "options": linked_options,
            "d14_structure_gate": linked_d14_gate,
        },
        "official_train_artifacts": artifact_receipts,
        "routing_decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", required=True, type=Path)
    parser.add_argument("--source-identity-start", required=True, type=Path)
    parser.add_argument("--source-identity-final", required=True, type=Path)
    parser.add_argument("--diagnostic-source-commit", required=True)
    parser.add_argument("--diagnostic-source-tree", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--options-sha256", required=True)
    parser.add_argument("--d14-structure-gate-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    paths = {
        "scan": args.scan.expanduser().resolve(),
        "source_identity_start": args.source_identity_start.expanduser().resolve(),
        "source_identity_final": args.source_identity_final.expanduser().resolve(),
    }
    scan = _load(paths["scan"], "D1.5 scan")
    start = _load(paths["source_identity_start"], "D1.5 start identity")
    final = _load(paths["source_identity_final"], "D1.5 final identity")
    identity_start = _validate_identity(
        start,
        commit=args.diagnostic_source_commit,
        tree=args.diagnostic_source_tree,
        manifest_sha256=args.manifest_sha256,
        label="D1.5 start identity",
    )
    identity_final = _validate_identity(
        final,
        commit=args.diagnostic_source_commit,
        tree=args.diagnostic_source_tree,
        manifest_sha256=args.manifest_sha256,
        label="D1.5 final identity",
    )
    evidence = validate_scan(
        scan,
        diagnostic_commit=args.diagnostic_source_commit,
        diagnostic_tree=args.diagnostic_source_tree,
        manifest_sha256=args.manifest_sha256,
        checkpoint_sha256=args.checkpoint_sha256,
        options_sha256=args.options_sha256,
        d14_gate_sha256=args.d14_structure_gate_sha256,
    )
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError("D1.5 receipt already exists; evidence is append-only")
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "status": "PASS_DIAGNOSTIC",
        "status_semantics": (
            "integrity_and_routing_pass_not_model_performance_or_paper_pass"
        ),
        "protocol": FINALIZER_PROTOCOL,
        "train_only": True,
        "counterfactual": True,
        "paper_performance_valid": False,
        "strict_causal_paper_result_valid": False,
        "test_access": False,
        "checkpoint_updated": False,
        "optimizer_step_count": 0,
        "threshold_search": False,
        "locked_test_release": False,
        "official_comparison_release": False,
        "paper_claim_release": False,
        "source_identity": {
            "start": identity_start,
            "final": identity_final,
        },
        "input_artifacts": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
        },
        **evidence,
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
                "routing_decision": receipt["routing_decision"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
