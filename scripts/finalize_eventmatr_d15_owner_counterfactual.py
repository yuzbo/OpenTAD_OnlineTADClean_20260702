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
SEGMENT_SIZE = 64.0
QUERY_COUNT = 10
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
    "parallel_window_causality_batch_count": 3270,
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
    "parallel_window_causality_verified": True,
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
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not an object")
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"{label} {key} mismatch: {payload.get(key)!r} != {value!r}"
            )


def _required_mapping(value, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object")
    return value


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


def _optional_nonnegative_integer(value, label: str):
    if value is None:
        return None
    return _nonnegative_integer(value, label)


def _finite_float(value, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} is not numeric")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{label} is not finite")
    return numeric


def _count_mapping(value, label: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is absent")
    parsed = {}
    for key, count in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{label} contains an invalid key")
        parsed[key] = _nonnegative_integer(count, f"{label}.{key}")
    return parsed


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
    if parsed["end_count"] != parsed["emit_count"]:
        raise ValueError(f"{channel}/{route} end/emission closure differs")
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
    created_by_source = _count_mapping(
        row.get("created_records_by_source"),
        f"{channel}/{route}.created_records_by_source",
    )
    if sum(created_by_source.values()) != unique_records:
        raise ValueError(f"{channel}/{route} source/unique-record accounting differs")
    decisions_by_source = _count_mapping(
        row.get("decision_rows_by_source"),
        f"{channel}/{route}.decision_rows_by_source",
    )
    if sum(decisions_by_source.values()) != fragments:
        raise ValueError(f"{channel}/{route} source/decision-row accounting differs")
    first_state_counts = _count_mapping(
        row.get("first_owner_decision_state_counts"),
        f"{channel}/{route}.first_owner_decision_state_counts",
    )
    winner_counts = _count_mapping(
        row.get("owner_state_winner_counts"),
        f"{channel}/{route}.owner_state_winner_counts",
    )
    for label, state_counts in (
        ("first-owner", first_state_counts),
        ("winner", winner_counts),
    ):
        if not set(state_counts).issubset({"0", "1", "2"}):
            raise ValueError(f"{channel}/{route} {label} states drifted")
    if sum(first_state_counts.values()) > unique_records:
        raise ValueError(f"{channel}/{route} first-owner decisions exceed records")
    if sum(winner_counts.values()) != fragments:
        raise ValueError(f"{channel}/{route} winner/decision-row accounting differs")
    transition_counts = _count_mapping(
        row.get("transition_counts"),
        f"{channel}/{route}.transition_counts",
    )
    allowed_transitions = {
        "birth",
        "cancel",
        "diagnostic_retain_cancel",
        "end",
        "emit",
        "reacquire",
    }
    if not set(transition_counts).issubset(allowed_transitions):
        raise ValueError(f"{channel}/{route} transition vocabulary drifted")
    expected_transition_counts = {
        "birth": parsed["birth_count"],
        "cancel": parsed["cancellation_count"],
        "end": parsed["end_count"],
        "emit": parsed["emit_count"],
        "reacquire": parsed["reacquisition_count"],
    }
    for name, expected in expected_transition_counts.items():
        if transition_counts.get(name, 0) != expected:
            raise ValueError(
                f"{channel}/{route} transition/count accounting differs: {name}"
            )
    if route == "formal" and transition_counts.get(
        "diagnostic_retain_cancel", 0
    ):
        raise ValueError(f"{channel}/{route} used the shadow cancel intervention")
    if route == "shadow" and transition_counts.get("cancel", 0):
        raise ValueError(f"{channel}/{route} applied a formal cancellation")
    eos_active_counts = _count_mapping(
        row.get("eos_active_record_counts_by_video"),
        f"{channel}/{route}.eos_active_record_counts_by_video",
    )
    eos_birth_counts = _count_mapping(
        row.get("eos_birth_counts_by_video"),
        f"{channel}/{route}.eos_birth_counts_by_video",
    )
    if (
        len(eos_active_counts) != parsed["observed_eos_count"]
        or set(eos_active_counts) != set(eos_birth_counts)
    ):
        raise ValueError(f"{channel}/{route} per-video EOS coverage differs")
    if sum(eos_active_counts.values()) != parsed["records_active_after_eos"]:
        raise ValueError(f"{channel}/{route} active-after-EOS accounting differs")
    if sum(value > 0 for value in eos_active_counts.values()) != parsed[
        "videos_with_active_records_after_eos"
    ]:
        raise ValueError(f"{channel}/{route} active-after-EOS videos differ")
    if sum(eos_birth_counts.values()) > parsed["birth_count"]:
        raise ValueError(f"{channel}/{route} EOS births exceed all births")
    integrity = _validate_integrity(
        row.get("lifecycle_integrity", {}), f"{channel}/{route}"
    )
    if integrity["ledger_row_count"] != parsed["emit_count"]:
        raise ValueError(f"{channel}/{route} ledger/emission count differs")
    statistics = row.get("statistics")
    if not isinstance(statistics, dict):
        raise ValueError(f"{channel}/{route} statistics are absent")
    for stats_name, stats in statistics.items():
        if not isinstance(stats, dict):
            raise ValueError(f"{channel}/{route} statistic {stats_name} is malformed")
        count = _nonnegative_integer(
            stats.get("count"), f"{channel}/{route}.{stats_name}.count"
        )
        expected_stat_fields = (
            {"count"}
            if count == 0
            else {"count", "mean", "std", "min", "max", "positive_count"}
        )
        if set(stats) != expected_stat_fields:
            raise ValueError(
                f"{channel}/{route} statistic {stats_name} schema drifted"
            )
        for key, value in stats.items():
            if key in {"count", "positive_count"}:
                _nonnegative_integer(
                    value, f"{channel}/{route}.{stats_name}.{key}"
                )
            else:
                _finite_float(
                    value, f"{channel}/{route}.{stats_name}.{key}"
                )
        if count:
            positive_count = _nonnegative_integer(
                stats["positive_count"],
                f"{channel}/{route}.{stats_name}.positive_count",
            )
            if positive_count > count:
                raise ValueError(
                    f"{channel}/{route} statistic {stats_name} positives exceed count"
                )
            if (
                float(stats["std"]) < 0
                or float(stats["min"]) > float(stats["mean"])
                or float(stats["mean"]) > float(stats["max"])
            ):
                raise ValueError(
                    f"{channel}/{route} statistic {stats_name} ordering drifted"
                )
    return {
        "counts": parsed,
        "unique_runtime_record_count": unique_records,
        "linked_unique_target_count": linked_targets,
        "raw_semantic_duplicate_count": raw_duplicates,
        "created_records_by_source": created_by_source,
        "decision_rows_by_source": decisions_by_source,
        "eos_active_record_counts_by_video": eos_active_counts,
        "eos_birth_counts_by_video": eos_birth_counts,
        "lifecycle_integrity": integrity,
        "first_owner_decision_state_counts": first_state_counts,
        "owner_state_winner_counts": winner_counts,
        "transition_counts": transition_counts,
        "statistics": statistics,
    }


def _validate_trace(path: Path, routes: dict[str, dict[str, dict]]) -> dict:
    required_fields = {
        "protocol",
        "channel",
        "route",
        "video_name",
        "runtime_event_id",
        "diagnostic_target_event_id",
        "diagnostic_target_end_frame",
        "source",
        "creation_frame",
        "current_frame",
        "is_eos",
        "first_owner_decision",
        "owner_query_before_refresh",
        "owner_query_after_refresh",
        "formal_owner_query_id_after_intervention",
        "formal_owner_query_id_unchanged",
        "matched_current_query",
        "owner_to_birth_cosine",
        "owner_to_oracle_query_cosine",
        "owner_attention_top_query",
        "oracle_query_attention_rank",
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
        "formal_lifetime",
        "shadow_lifetime",
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
    target_backed_near_end_counts: Counter = Counter()
    end_at_observed_target_end_counts: Counter = Counter()
    transition_counts: Counter = Counter()
    source_counts: Counter = Counter()
    first_state_counts: Counter = Counter()
    eos_survivor_counts: Counter = Counter()
    last_frame: dict[tuple[str, str, str], float] = {}
    event_ids_at_frame: dict[tuple[str, str, str], set[int]] = {}
    seen_event_ids: dict[tuple[str, str, str], set[int]] = {}
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
                if not isinstance(row["source"], str) or not row["source"]:
                    raise ValueError(
                        f"D1.5 trace line {line_number} source is invalid"
                    )
                event_id = _nonnegative_integer(
                    row["runtime_event_id"],
                    f"trace[{line_number}].runtime_event_id",
                )
                target_event_id = _optional_nonnegative_integer(
                    row["diagnostic_target_event_id"],
                    f"trace[{line_number}].diagnostic_target_event_id",
                )
                frame = _finite_float(
                    row["current_frame"], f"trace[{line_number}].current_frame"
                )
                creation_frame = _finite_float(
                    row["creation_frame"], f"trace[{line_number}].creation_frame"
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
                first_decision = row["first_owner_decision"]
                if not isinstance(first_decision, bool):
                    raise ValueError(
                        f"D1.5 trace line {line_number} first-decision flag is invalid"
                    )
                seen = seen_event_ids.setdefault(stream_key, set())
                if first_decision != (event_id not in seen):
                    raise ValueError(
                        f"D1.5 trace line {line_number} first-decision audit drifted"
                    )
                seen.add(event_id)
                if not isinstance(row["is_eos"], bool):
                    raise ValueError(
                        f"D1.5 trace line {line_number} EOS flag is invalid"
                    )

                owner_query_before = _nonnegative_integer(
                    row["owner_query_before_refresh"],
                    f"trace[{line_number}].owner_query_before_refresh",
                )
                owner_query_after = _nonnegative_integer(
                    row["owner_query_after_refresh"],
                    f"trace[{line_number}].owner_query_after_refresh",
                )
                formal_owner_query = _nonnegative_integer(
                    row["formal_owner_query_id_after_intervention"],
                    (
                        f"trace[{line_number}]."
                        "formal_owner_query_id_after_intervention"
                    ),
                )
                matched_query = _optional_nonnegative_integer(
                    row["matched_current_query"],
                    f"trace[{line_number}].matched_current_query",
                )
                attention_top_query = _nonnegative_integer(
                    row["owner_attention_top_query"],
                    f"trace[{line_number}].owner_attention_top_query",
                )
                oracle_rank = _optional_nonnegative_integer(
                    row["oracle_query_attention_rank"],
                    f"trace[{line_number}].oracle_query_attention_rank",
                )
                if oracle_rank is not None and oracle_rank < 1:
                    raise ValueError(
                        f"D1.5 trace line {line_number} oracle rank is invalid"
                    )
                for query_label, query_index in (
                    ("owner query before", owner_query_before),
                    ("owner query after", owner_query_after),
                    ("formal owner query", formal_owner_query),
                    ("attention top query", attention_top_query),
                ):
                    if query_index >= QUERY_COUNT:
                        raise ValueError(
                            f"D1.5 trace line {line_number} {query_label} "
                            "exceeds query bandwidth"
                        )
                if matched_query is not None and matched_query >= QUERY_COUNT:
                    raise ValueError(
                        f"D1.5 trace line {line_number} matched query "
                        "exceeds query bandwidth"
                    )
                if oracle_rank is not None and oracle_rank > QUERY_COUNT:
                    raise ValueError(
                        f"D1.5 trace line {line_number} oracle rank "
                        "exceeds query bandwidth"
                    )

                owner_to_birth = _finite_float(
                    row["owner_to_birth_cosine"],
                    f"trace[{line_number}].owner_to_birth_cosine",
                )
                if not -1.00001 <= owner_to_birth <= 1.00001:
                    raise ValueError(
                        f"D1.5 trace line {line_number} birth cosine is invalid"
                    )
                owner_to_oracle_raw = row["owner_to_oracle_query_cosine"]
                owner_to_oracle = (
                    None
                    if owner_to_oracle_raw is None
                    else _finite_float(
                        owner_to_oracle_raw,
                        f"trace[{line_number}].owner_to_oracle_query_cosine",
                    )
                )
                if owner_to_oracle is not None and not (
                    -1.00001 <= owner_to_oracle <= 1.00001
                ):
                    raise ValueError(
                        f"D1.5 trace line {line_number} oracle cosine is invalid"
                    )
                if (owner_to_oracle is None) != (oracle_rank is None):
                    raise ValueError(
                        f"D1.5 trace line {line_number} oracle-query audit drifted"
                    )
                entropy = _finite_float(
                    row["owner_attention_entropy"],
                    f"trace[{line_number}].owner_attention_entropy",
                )
                if entropy < 0:
                    raise ValueError(
                        f"D1.5 trace line {line_number} attention entropy is invalid"
                    )

                logits = [
                    _finite_float(
                        row[name], f"trace[{line_number}].{name}"
                    )
                    for name in ("cancel_logit", "continue_logit", "end_logit")
                ]
                margins = [
                    _finite_float(
                        row[name], f"trace[{line_number}].{name}"
                    )
                    for name in ("cancel_margin", "continue_margin", "end_margin")
                ]
                expected_margins = [
                    logits[index]
                    - max(logits[:index] + logits[index + 1 :])
                    for index in range(3)
                ]
                if any(
                    not math.isclose(
                        observed, expected, rel_tol=1e-6, abs_tol=1e-6
                    )
                    for observed, expected in zip(margins, expected_margins)
                ):
                    raise ValueError(
                        f"D1.5 trace line {line_number} state margins drifted"
                    )
                winner = _nonnegative_integer(
                    row["state_argmax"], f"trace[{line_number}].state_argmax"
                )
                if winner not in (0, 1, 2):
                    raise ValueError(
                        f"D1.5 trace line {line_number} state winner is invalid"
                    )
                expected_winner = max(range(3), key=lambda index: logits[index])
                if winner != expected_winner:
                    raise ValueError(
                        f"D1.5 trace line {line_number} state winner drifted"
                    )
                if row["formal_owner_query_id_unchanged"] is not True:
                    raise ValueError(
                        f"D1.5 trace line {line_number} mutated formal owner identity"
                    )
                if formal_owner_query != owner_query_before:
                    raise ValueError(
                        f"D1.5 trace line {line_number} owner identity audit drifted"
                    )
                if matched_query is None:
                    if owner_query_after != owner_query_before:
                        raise ValueError(
                            f"D1.5 trace line {line_number} unmatched refresh drifted"
                        )
                else:
                    if CHANNEL_FACTORS[channel][1] != "refreshed":
                        raise ValueError(
                            f"D1.5 trace line {line_number} refreshed a free route"
                        )
                    if owner_query_after != matched_query:
                        raise ValueError(
                            f"D1.5 trace line {line_number} refresh query drifted"
                        )
                if (
                    CHANNEL_FACTORS[channel][1] == "refreshed"
                    and oracle_rank is not None
                    and matched_query is None
                ):
                    raise ValueError(
                        f"D1.5 trace line {line_number} omitted a visible refresh"
                    )
                if (
                    CHANNEL_FACTORS[channel][1] == "free"
                    and matched_query is not None
                ):
                    raise ValueError(
                        f"D1.5 trace line {line_number} matched a free route"
                    )

                if not isinstance(row["target_end_observable_now"], bool):
                    raise ValueError(
                        f"D1.5 trace line {line_number} end-observable flag is invalid"
                    )
                if not isinstance(
                    row["end_suppressed_by_minimum_duration"], bool
                ):
                    raise ValueError(
                        f"D1.5 trace line {line_number} end-suppression flag is invalid"
                    )
                target_end_frame_raw = row["diagnostic_target_end_frame"]
                frame_from_end_raw = row["frames_from_annotated_end"]
                if target_event_id is None:
                    if (
                        target_end_frame_raw is not None
                        or frame_from_end_raw is not None
                        or row["target_end_observable_now"]
                    ):
                        raise ValueError(
                            f"D1.5 trace line {line_number} leaked an unlinked target"
                        )
                    target_end_frame = None
                    frame_from_end = None
                else:
                    target_end_frame = _finite_float(
                        target_end_frame_raw,
                        f"trace[{line_number}].diagnostic_target_end_frame",
                    )
                    frame_from_end = _finite_float(
                        frame_from_end_raw,
                        f"trace[{line_number}].frames_from_annotated_end",
                    )
                    if target_end_frame < 0 or not math.isclose(
                        frame - target_end_frame,
                        frame_from_end,
                        rel_tol=1e-6,
                        abs_tol=1e-6,
                    ):
                        raise ValueError(
                            f"D1.5 trace line {line_number} target-end audit drifted"
                        )

                formal_lifetime_raw = row["formal_lifetime"]
                shadow_lifetime_raw = row["shadow_lifetime"]
                expected_lifetime = frame - creation_frame
                if route == "formal":
                    if shadow_lifetime_raw is not None:
                        raise ValueError(
                            f"D1.5 trace line {line_number} mixed route lifetimes"
                        )
                    lifetime = _finite_float(
                        formal_lifetime_raw,
                        f"trace[{line_number}].formal_lifetime",
                    )
                else:
                    if formal_lifetime_raw is not None:
                        raise ValueError(
                            f"D1.5 trace line {line_number} mixed route lifetimes"
                        )
                    lifetime = _finite_float(
                        shadow_lifetime_raw,
                        f"trace[{line_number}].shadow_lifetime",
                    )
                if not math.isclose(
                    lifetime, expected_lifetime, rel_tol=1e-6, abs_tol=1e-6
                ):
                    raise ValueError(
                        f"D1.5 trace line {line_number} route lifetime drifted"
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
                transition_sequence = transition.split("+")
                transition_parts = set(transition_sequence)
                if (
                    not transition
                    or len(transition_sequence) != len(transition_parts)
                    or not transition_parts.issubset(
                        {
                            "cancel",
                            "diagnostic_retain_cancel",
                            "continue",
                            "end_suppressed_min_duration",
                            "end",
                            "emit",
                        }
                    )
                ):
                    raise ValueError(
                        f"D1.5 trace line {line_number} transition is invalid"
                    )
                suppressed = row["end_suppressed_by_minimum_duration"]
                if suppressed and winner != 2:
                    raise ValueError(
                        f"D1.5 trace line {line_number} suppressed a non-end winner"
                    )
                if winner == 0:
                    expected_transition = (
                        "cancel"
                        if route == "formal"
                        else "diagnostic_retain_cancel"
                    )
                    if transition != expected_transition:
                        raise ValueError(
                            f"D1.5 trace line {line_number} cancel policy drifted"
                        )
                elif winner == 1:
                    if transition != "continue":
                        raise ValueError(
                            f"D1.5 trace line {line_number} continue policy drifted"
                        )
                elif suppressed:
                    if transition != "end_suppressed_min_duration":
                        raise ValueError(
                            f"D1.5 trace line {line_number} suppression policy drifted"
                        )
                elif "end" not in transition_parts or not transition_parts.issubset(
                    {"end", "emit"}
                ):
                    raise ValueError(
                        f"D1.5 trace line {line_number} end policy drifted"
                    )
                route_counts[key] += 1
                source_counts[(channel, route, row["source"])] += 1
                if first_decision:
                    first_state_counts[(channel, route, str(winner))] += 1
                end_argmax_counts[key] += int(winner == 2)
                suppressed_end_counts[key] += int(suppressed)
                for part in transition_parts.intersection(
                    {"cancel", "diagnostic_retain_cancel", "end", "emit"}
                ):
                    transition_counts[(channel, route, part)] += 1
                target_backed_end = (
                    "end" in transition_parts and target_event_id is not None
                )
                target_backed_end_counts[key] += int(target_backed_end)
                target_backed_near_end_counts[key] += int(
                    target_backed_end
                    and abs(frame_from_end) <= SEGMENT_SIZE
                )
                end_at_observed_target_end_counts[key] += int(
                    target_backed_end and row["target_end_observable_now"]
                )
                if (
                    row["is_eos"]
                    and "cancel" not in transition_parts
                    and "end" not in transition_parts
                ):
                    eos_survivor_counts[
                        (channel, route, row["video_name"])
                    ] += 1
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
            if (
                target_backed_near_end_counts[key]
                != counts["target_backed_end_within_one_segment_count"]
            ):
                raise ValueError(
                    f"D1.5 trace {channel}/{route} near-target ends differ"
                )
            if (
                end_at_observed_target_end_counts[key]
                != counts["end_at_observed_target_end_count"]
            ):
                raise ValueError(
                    f"D1.5 trace {channel}/{route} observed-target ends differ"
                )
            observed_sources = {
                source: count
                for (seen_channel, seen_route, source), count in source_counts.items()
                if seen_channel == channel and seen_route == route
            }
            if observed_sources != routes[channel][route]["decision_rows_by_source"]:
                raise ValueError(
                    f"D1.5 trace {channel}/{route} source rows differ"
                )
            observed_first_states = {
                state: count
                for (
                    seen_channel,
                    seen_route,
                    state,
                ), count in first_state_counts.items()
                if seen_channel == channel and seen_route == route
            }
            if observed_first_states != routes[channel][route][
                "first_owner_decision_state_counts"
            ]:
                raise ValueError(
                    f"D1.5 trace {channel}/{route} first-owner states differ"
                )
            eos_active = routes[channel][route][
                "eos_active_record_counts_by_video"
            ]
            eos_births = routes[channel][route]["eos_birth_counts_by_video"]
            for video_name, active_count in eos_active.items():
                reconstructed = (
                    eos_survivor_counts[(channel, route, video_name)]
                    + eos_births[video_name]
                )
                if reconstructed != active_count:
                    raise ValueError(
                        f"D1.5 trace {channel}/{route} "
                        f"{video_name} active-after-EOS count differs"
                    )
            for transition_name in (
                "cancel",
                "diagnostic_retain_cancel",
                "end",
                "emit",
            ):
                if transition_counts[(channel, route, transition_name)] != (
                    routes[channel][route]["transition_counts"].get(
                        transition_name, 0
                    )
                ):
                    raise ValueError(
                        f"D1.5 trace {channel}/{route} "
                        f"{transition_name} transitions differ"
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
        "first_owner_decisions_cross_checked": True,
        "active_after_eos_cross_checked": True,
        "required_fields_and_finite_values_verified": True,
        "owner_state_logits_margins_and_winners_cross_checked": True,
        "route_lifetimes_cross_checked": True,
        "target_end_distances_cross_checked": True,
        "source_counts_cross_checked": True,
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
        if formal[channel]["end_count"] != formal[channel]["emit_count"]:
            return {
                "diagnosis": "end_to_emission_closure_fault",
                "authorized_next_intervention": "ledger_state_machine_repair_only",
                "model_training_authorized": False,
                "reason": f"{channel} end and emission counts did not match",
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

    closes = {
        channel: formal[channel]["end_count"] > 0
        and formal[channel]["emit_count"] > 0
        for channel in CHANNELS
    }
    if closes["PF"]:
        return {
            "diagnosis": "predicted_free_reference_control_drift",
            "authorized_next_intervention": "diagnostic_implementation_repair_only",
            "model_training_authorized": False,
            "reason": "PF unexpectedly closed despite the frozen D1.4 reference",
        }
    pr_closes = closes["PR"]
    of_closes = closes["OF"]
    or_closes = closes["OR"]
    pf_fails = not closes["PF"]

    if pf_fails and pr_closes and not of_closes:
        return {
            "diagnosis": "identity_transport_is_sufficient_under_predicted_burden",
            "authorized_next_intervention": (
                "training_only_predicted_track_supervision_bridge"
            ),
            "model_training_authorized": True,
            "reason": "PR restored frozen-owner end/emission while OF did not",
        }
    if pf_fails and of_closes and not pr_closes:
        return {
            "diagnosis": "clean_admission_is_sufficient_without_continuous_refresh",
            "authorized_next_intervention": (
                "score_independent_sampling_corrected_birth_admission"
            ),
            "model_training_authorized": True,
            "reason": "OF restored end/emission while PR did not",
        }
    if pf_fails and or_closes and not pr_closes and not of_closes:
        return {
            "diagnosis": "admission_identity_interaction",
            "authorized_next_intervention": (
                "one_prospectively_registered_two_factor_repair"
            ),
            "model_training_authorized": True,
            "reason": "only OR restored frozen-owner end/emission",
        }
    if pf_fails and pr_closes and of_closes:
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
    formal_model_cells_fail = not any(
        closes[channel] for channel in ("PR", "OF", "OR")
    )
    if formal_model_cells_fail and any(
        value > 0 for value in shadow_near_end.values()
    ):
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


def _validate_source_provenance(
    scan: dict,
    *,
    training_commit: str,
    training_tree: str,
    d14_source_commit: str,
    d14_source_tree: str,
) -> dict:
    training_identity = scan.get("training_source_identity")
    if not isinstance(training_identity, dict):
        raise ValueError("D1.5 training source identity is absent")
    _expect(
        training_identity,
        {
            "commit": training_commit,
            "tree": training_tree,
            "clean": True,
            "status": "PASS",
            "status_porcelain": "",
        },
        "D1.5 training source identity",
    )
    _expect(
        training_identity.get("smoke", {}),
        {"status": "PASS", "test_access": False},
        "D1.5 training source smoke",
    )
    d14_identity = (
        scan.get("d14_structure_gate", {})
        .get("binding", {})
        .get("source_identity")
    )
    if not isinstance(d14_identity, dict):
        raise ValueError("D1.5 D1.4 source identity is absent")
    _expect(
        d14_identity,
        {
            "commit": d14_source_commit,
            "tree": d14_source_tree,
        },
        "D1.5 D1.4 source identity",
    )
    return {
        "training_source_identity": training_identity,
        "d14_source_identity": d14_identity,
    }


def validate_scan(
    scan: dict,
    *,
    diagnostic_commit: str,
    diagnostic_tree: str,
    training_commit: str,
    training_tree: str,
    d14_source_commit: str,
    d14_source_tree: str,
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
    provenance = _validate_source_provenance(
        scan,
        training_commit=training_commit,
        training_tree=training_tree,
        d14_source_commit=d14_source_commit,
        d14_source_tree=d14_source_tree,
    )
    _expect(
        scan.get("manifest", {}),
        {"sha256": manifest_sha256},
        "D1.5 manifest",
    )
    manifest_ref = _required_mapping(scan.get("manifest"), "D1.5 manifest")
    checkpoint_ref = _required_mapping(scan.get("checkpoint"), "D1.5 checkpoint")
    options_ref = _required_mapping(scan.get("options"), "D1.5 options")
    d14_gate_ref = _required_mapping(
        scan.get("d14_structure_gate"), "D1.5 D1.4 structure gate"
    )
    if checkpoint_ref.get("sha256") != checkpoint_sha256:
        raise ValueError("D1.5 checkpoint hash mismatch")
    if options_ref.get("sha256") != options_sha256:
        raise ValueError("D1.5 options hash mismatch")
    if d14_gate_ref.get("sha256") != d14_gate_sha256:
        raise ValueError("D1.5 D1.4-gate hash mismatch")
    _expect(
        scan.get("query_compatibility_rule", {}),
        QUERY_COMPATIBILITY_RULE,
        "D1.5 query compatibility rule",
    )
    linked_manifest = _validate_linked_file(
        manifest_ref, "D1.5 manifest"
    )
    linked_checkpoint = _validate_linked_file(
        checkpoint_ref, "D1.5 checkpoint"
    )
    linked_options = _validate_linked_file(
        options_ref, "D1.5 options"
    )
    linked_d14_gate = _validate_linked_file(
        d14_gate_ref, "D1.4 structure gate"
    )
    if linked_manifest["sha256"] != manifest_sha256:
        raise ValueError("linked D1.5 manifest hash mismatch")
    if linked_checkpoint["sha256"] != checkpoint_sha256:
        raise ValueError("linked D1.5 checkpoint hash mismatch")
    if linked_options["sha256"] != options_sha256:
        raise ValueError("linked D1.5 options hash mismatch")
    if linked_d14_gate["sha256"] != d14_gate_sha256:
        raise ValueError("linked D1.4 structure-gate hash mismatch")
    d14_binding = _required_mapping(
        d14_gate_ref.get("binding"), "D1.5 D1.4 structure-gate binding"
    )
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
    channel_rows = {
        channel: _required_mapping(
            channels[channel], f"D1.5 channel {channel}"
        )
        for channel in CHANNELS
    }
    routes = {
        channel: {
            route: _validate_route(
                channel_rows[channel].get(route, {}),
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
        **provenance,
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
    parser.add_argument("--training-source-commit", required=True)
    parser.add_argument("--training-source-tree", required=True)
    parser.add_argument("--d14-source-commit", required=True)
    parser.add_argument("--d14-source-tree", required=True)
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
        training_commit=args.training_source_commit,
        training_tree=args.training_source_tree,
        d14_source_commit=args.d14_source_commit,
        d14_source_tree=args.d14_source_tree,
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
