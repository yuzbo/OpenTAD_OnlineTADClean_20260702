"""Read-only, preregistered endpoint-margin refinement for EventMATR D1.5.

This program never imports the model or training stack.  It binds the frozen
D1.5 scan/receipt/trace by SHA-256, streams the already validated trace, and
writes one new append-only diagnostic receipt.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping


PROTOCOL = "eventmatr_d151_endpoint_margin_trace_v1"
D15_RECEIPT_PROTOCOL = "eventmatr_d15_owner_counterfactual_finalizer_v2"
D15_TRACE_PROTOCOL = "eventmatr_d15_owner_counterfactual_v2"
CHANNELS = ("PF", "PR", "OF", "OR")
CHANNEL_FACTORS = {
    "PF": ("predicted", "free"),
    "PR": ("predicted", "refreshed"),
    "OF": ("oracle_visible", "free"),
    "OR": ("oracle_visible", "refreshed"),
}
ROUTES = ("formal", "shadow")
ROUTE_KEYS = tuple(f"{channel}/{route}" for channel in CHANNELS for route in ROUTES)
TEMPORAL_BINS = ("pre_far", "pre_window", "post_window", "post_far")
DURATION_STRATA = ("le_8", "gt8_le16", "gt16_le32", "gt32_le64", "gt64")
ENDPOINT_RADIUS = 64.0
MATERIAL_RATE = 0.05
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 52_017
QUERY_COUNT = 10
EXPECTED_COUNTS = {
    "video_count": 200,
    "annotation_count": 3007,
    "visible_birth_target_count": 3003,
    "fully_observed_event_count": 3001,
    "right_censored_event_count": 2,
    "left_truncated_event_count": 0,
    "completely_unobservable_event_count": 4,
}
PAPER_FLAGS = {
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
}
TRACE_FIELDS = {
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
TRANSITION_PARTS = {
    "cancel",
    "diagnostic_retain_cancel",
    "continue",
    "end_suppressed_min_duration",
    "end",
    "emit",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} is not a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} is not a SHA-256 digest") from error
    return value.lower()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_object(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} cannot be decoded as JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} is not numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite")
    return number


def _nonnegative_int(value: object, label: str) -> int:
    number = _finite(value, label)
    if number < 0 or not number.is_integer():
        raise ValueError(f"{label} is not a non-negative integer")
    return int(number)


def _optional_nonnegative_int(value: object, label: str) -> int | None:
    return None if value is None else _nonnegative_int(value, label)


def temporal_bin(distance: float) -> str:
    distance = _finite(distance, "endpoint distance")
    if distance < -ENDPOINT_RADIUS:
        return "pre_far"
    if distance < 0.0:
        return "pre_window"
    if distance <= ENDPOINT_RADIUS:
        return "post_window"
    return "post_far"


def duration_stratum(duration: float) -> str:
    duration = _finite(duration, "event duration")
    if duration <= 8.0:
        return "le_8"
    if duration <= 16.0:
        return "gt8_le16"
    if duration <= 32.0:
        return "gt16_le32"
    if duration <= 64.0:
        return "gt32_le64"
    return "gt64"


def nearest_rank(values: Iterable[float], probability: float) -> float:
    ordered = sorted(_finite(value, "nearest-rank value") for value in values)
    if not ordered:
        raise ValueError("nearest-rank statistic requires at least one value")
    if not 0.0 < probability <= 1.0:
        raise ValueError("nearest-rank probability must be in (0, 1]")
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def _value_summary(values: Iterable[float | None]) -> dict:
    finite = [_finite(value, "summary value") for value in values if value is not None]
    if not finite:
        return {"count": 0, "median_nearest_rank": None, "p95_nearest_rank": None}
    return {
        "count": len(finite),
        "mean": sum(finite) / len(finite),
        "minimum": min(finite),
        "maximum": max(finite),
        "median_nearest_rank": nearest_rank(finite, 0.5),
        "p95_nearest_rank": nearest_rank(finite, 0.95),
        "positive_count": sum(value > 0.0 for value in finite),
    }


def _git(project_dir: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=project_dir,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _source_identity(
    project_dir: Path, *, expected_commit: str, expected_tree: str
) -> dict:
    status = _git(project_dir, "status", "--porcelain=v1", "--untracked-files=all")
    commit = _git(project_dir, "rev-parse", "HEAD")
    tree = _git(project_dir, "rev-parse", "HEAD^{tree}")
    if status:
        raise ValueError(f"D1.5.1 analyzer source is dirty:\n{status}")
    if commit != expected_commit or tree != expected_tree:
        raise ValueError("D1.5.1 analyzer source identity mismatch")
    return {"commit": commit, "tree": tree, "clean": True}


@dataclass
class EventTrace:
    channel: str
    route: str
    video_name: str
    event_id: int
    class_label: str
    duration_frames: float
    outcome: dict
    row_count: int = 0
    runtime_record_ids: set[int] = field(default_factory=set)
    winner_counts: Counter = field(default_factory=Counter)
    bin_row_counts: Counter = field(default_factory=Counter)
    bin_winner_counts: dict[str, Counter] = field(
        default_factory=lambda: {name: Counter() for name in TEMPORAL_BINS}
    )
    maximum_end_margin: float | None = None
    bin_maximum_end_margin: dict[str, float | None] = field(
        default_factory=lambda: {name: None for name in TEMPORAL_BINS}
    )
    target_backed_end_bins: set[str] = field(default_factory=set)
    immutable_emission_bins: set[str] = field(default_factory=set)
    primary_closure_bins: set[str] = field(default_factory=set)
    target_backed_end_any: bool = False
    immutable_emission_any: bool = False
    primary_closure_any: bool = False
    target_endpoint_observed: bool = False
    pre_end_cancel: bool = False
    endpoint_eligible_end_without_emit: bool = False
    first_decision_choice: tuple[float, int, float] | None = None
    last_pre_frame: float | None = None
    last_pre_end_margin: float | None = None
    first_post_frame: float | None = None
    first_post_end_margin: float | None = None

    @property
    def route_key(self) -> str:
        return f"{self.channel}/{self.route}"

    @property
    def endpoint_maximum_end_margin(self) -> float | None:
        values = [
            self.bin_maximum_end_margin["pre_window"],
            self.bin_maximum_end_margin["post_window"],
        ]
        finite = [value for value in values if value is not None]
        return max(finite) if finite else None

    @property
    def first_decision_cancel_margin(self) -> float | None:
        return (
            None
            if self.first_decision_choice is None
            else self.first_decision_choice[2]
        )

    @property
    def post_minus_pre_end_margin(self) -> float | None:
        if self.last_pre_end_margin is None or self.first_post_end_margin is None:
            return None
        return self.first_post_end_margin - self.last_pre_end_margin

    def add_row(
        self,
        *,
        runtime_event_id: int,
        current_frame: float,
        distance: float,
        first_owner_decision: bool,
        cancel_margin: float,
        end_margin: float,
        winner: int,
        suppressed: bool,
        transition_parts: set[str],
        target_endpoint_observed: bool,
    ) -> None:
        bin_name = temporal_bin(distance)
        endpoint = -ENDPOINT_RADIUS <= distance <= ENDPOINT_RADIUS
        has_end = "end" in transition_parts
        has_emit = "emit" in transition_parts
        has_primary = has_end and has_emit
        self.row_count += 1
        self.runtime_record_ids.add(runtime_event_id)
        self.winner_counts[str(winner)] += 1
        self.bin_row_counts[bin_name] += 1
        self.bin_winner_counts[bin_name][str(winner)] += 1
        self.maximum_end_margin = (
            end_margin
            if self.maximum_end_margin is None
            else max(self.maximum_end_margin, end_margin)
        )
        previous = self.bin_maximum_end_margin[bin_name]
        self.bin_maximum_end_margin[bin_name] = (
            end_margin if previous is None else max(previous, end_margin)
        )
        if has_end:
            self.target_backed_end_any = True
            self.target_backed_end_bins.add(bin_name)
        if has_emit:
            self.immutable_emission_any = True
            self.immutable_emission_bins.add(bin_name)
        if has_primary:
            self.primary_closure_any = True
            self.primary_closure_bins.add(bin_name)
        self.target_endpoint_observed |= target_endpoint_observed
        self.pre_end_cancel |= (
            self.route == "formal" and "cancel" in transition_parts and distance < 0.0
        )
        self.endpoint_eligible_end_without_emit |= (
            endpoint
            and end_margin > 0.0
            and winner == 2
            and not suppressed
            and not has_primary
        )
        if first_owner_decision:
            choice = (current_frame, runtime_event_id, cancel_margin)
            if (
                self.first_decision_choice is None
                or choice[:2] < self.first_decision_choice[:2]
            ):
                self.first_decision_choice = choice
        if distance < 0.0:
            if self.last_pre_frame is None or current_frame > self.last_pre_frame:
                self.last_pre_frame = current_frame
                self.last_pre_end_margin = end_margin
            elif current_frame == self.last_pre_frame:
                self.last_pre_end_margin = max(self.last_pre_end_margin, end_margin)
        else:
            if self.first_post_frame is None or current_frame < self.first_post_frame:
                self.first_post_frame = current_frame
                self.first_post_end_margin = end_margin
            elif current_frame == self.first_post_frame:
                self.first_post_end_margin = max(self.first_post_end_margin, end_margin)

    def as_dict(self) -> dict:
        endpoint_margin = self.endpoint_maximum_end_margin
        endpoint_bins = {"pre_window", "post_window"}
        return {
            "channel": self.channel,
            "route": self.route,
            "video_name": self.video_name,
            "event_id": self.event_id,
            "class_label": self.class_label,
            "duration_frames": self.duration_frames,
            "duration_stratum": duration_stratum(self.duration_frames),
            "target_linked": self.row_count > 0,
            "target_linked_runtime_record_count": len(self.runtime_record_ids),
            "decision_row_count": self.row_count,
            "winner_row_counts": {
                state: self.winner_counts[state] for state in ("0", "1", "2")
            },
            "maximum_end_margin": self.maximum_end_margin,
            "endpoint_maximum_end_margin": endpoint_margin,
            "endpoint_positive_end_margin": bool(
                endpoint_margin is not None and endpoint_margin > 0.0
            ),
            "target_backed_end_any": self.target_backed_end_any,
            "immutable_emission_any": self.immutable_emission_any,
            "primary_closure_any": self.primary_closure_any,
            "endpoint_target_backed_end": bool(
                endpoint_bins.intersection(self.target_backed_end_bins)
            ),
            "endpoint_immutable_emission": bool(
                endpoint_bins.intersection(self.immutable_emission_bins)
            ),
            "endpoint_primary_closure": bool(
                endpoint_bins.intersection(self.primary_closure_bins)
            ),
            "first_decision_cancel_margin": self.first_decision_cancel_margin,
            "last_pre_end_margin": self.last_pre_end_margin,
            "first_post_end_margin": self.first_post_end_margin,
            "post_minus_pre_end_margin": self.post_minus_pre_end_margin,
            "target_endpoint_observed": self.target_endpoint_observed,
            "pre_end_cancel": self.pre_end_cancel,
            "endpoint_eligible_end_without_emit": self.endpoint_eligible_end_without_emit,
            "unresolved_identity": bool(self.outcome["unresolved_identity"]),
            "ambiguous_identity": bool(self.outcome["ambiguous_identity"]),
            "temporal_bins": {
                bin_name: {
                    "decision_row_count": self.bin_row_counts[bin_name],
                    "winner_row_counts": {
                        state: self.bin_winner_counts[bin_name][state]
                        for state in ("0", "1", "2")
                    },
                    "maximum_end_margin": self.bin_maximum_end_margin[bin_name],
                    "positive_maximum_end_margin": bool(
                        self.bin_maximum_end_margin[bin_name] is not None
                        and self.bin_maximum_end_margin[bin_name] > 0.0
                    ),
                    "target_backed_end": bin_name in self.target_backed_end_bins,
                    "immutable_emission": bin_name in self.immutable_emission_bins,
                    "primary_closure": bin_name in self.primary_closure_bins,
                }
                for bin_name in TEMPORAL_BINS
            },
        }


def _validate_receipt(
    receipt: dict,
    scan: dict,
    *,
    expected_scan_sha256: str,
    expected_trace_sha256: str,
    expected_d15_commit: str,
    expected_d15_tree: str,
) -> tuple[list[dict], dict[tuple[str, str, str, int], dict]]:
    if receipt.get("status") != "PASS_DIAGNOSTIC":
        raise ValueError("D1.5 receipt did not pass diagnostic integrity")
    if receipt.get("protocol") != D15_RECEIPT_PROTOCOL:
        raise ValueError("D1.5 receipt protocol drifted")
    for field, expected in PAPER_FLAGS.items():
        if receipt.get(field) != expected:
            raise ValueError(f"D1.5 receipt paper boundary drifted: {field}")
    decision = receipt.get("routing_decision", {})
    if decision.get("model_implementation_authorized") is not False:
        raise ValueError("D1.5 receipt unexpectedly authorized model implementation")
    if decision.get("model_training_authorized") is not False:
        raise ValueError("D1.5 receipt unexpectedly authorized model training")
    scan_ref = receipt.get("input_artifacts", {}).get("scan", {})
    trace_ref = receipt.get("trace_artifact", {})
    if scan_ref.get("sha256") != expected_scan_sha256:
        raise ValueError("D1.5 receipt scan binding drifted")
    if trace_ref.get("sha256") != expected_trace_sha256:
        raise ValueError("D1.5 receipt trace binding drifted")
    source = receipt.get("source_identity", {})
    expected_source = {
        "commit": expected_d15_commit,
        "tree": expected_d15_tree,
        "clean": True,
    }
    for phase in ("start", "final"):
        identity = source.get(phase)
        if not isinstance(identity, dict) or any(
            identity.get(name) != value for name, value in expected_source.items()
        ):
            raise ValueError("D1.5 receipt source identity drifted")
        _valid_sha256(identity.get("manifest_sha256"), f"D1.5 {phase} manifest digest")
    if source["start"] != source["final"]:
        raise ValueError("D1.5 receipt start/final source identities differ")
    if (
        scan.get("status") != "DIAGNOSTIC_COMPLETE"
        or scan.get("protocol") != D15_TRACE_PROTOCOL
    ):
        raise ValueError("D1.5 scan status or protocol drifted")
    for field, expected in PAPER_FLAGS.items():
        if field == "paper_claim_release":
            continue
        if field in scan and scan.get(field) != expected:
            raise ValueError(f"D1.5 scan paper boundary drifted: {field}")
    if scan.get("trace_artifact", {}).get("sha256") != expected_trace_sha256:
        raise ValueError("D1.5 scan trace binding drifted")
    if scan.get("source_identity") != {
        "commit": expected_d15_commit,
        "tree": expected_d15_tree,
        "clean_start_and_final": True,
    }:
        raise ValueError("D1.5 scan source identity drifted")
    census = receipt.get("lifecycle_census")
    if not isinstance(census, dict):
        raise ValueError("D1.5 lifecycle census is absent")
    counts = census.get("counts", {})
    for name, expected in EXPECTED_COUNTS.items():
        if counts.get(name) != expected:
            raise ValueError(f"D1.5 lifecycle count drifted: {name}")
    events = census.get("events")
    if (
        not isinstance(events, list)
        or len(events) != EXPECTED_COUNTS["annotation_count"]
    ):
        raise ValueError("D1.5 lifecycle event list drifted")
    if census.get("event_list_sha256") != _canonical_sha256(events):
        raise ValueError("D1.5 lifecycle event-list hash drifted")
    if scan.get("lifecycle_census", {}).get("event_list_sha256") != census.get(
        "event_list_sha256"
    ):
        raise ValueError("D1.5 scan and receipt event lists differ")
    event_keys = set()
    visible_keys = set()
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(f"D1.5 lifecycle event {index} is malformed")
        video_name = event.get("video_name")
        event_id = _nonnegative_int(event.get("event_id"), f"event[{index}].event_id")
        if not isinstance(video_name, str) or not video_name:
            raise ValueError(f"D1.5 lifecycle event {index} has no video")
        key = (video_name, event_id)
        if key in event_keys:
            raise ValueError("D1.5 lifecycle event identity is duplicated")
        event_keys.add(key)
        if bool(event.get("birth_observed")):
            visible_keys.add(key)
    outcome_rows = receipt.get("paired_event_analysis", {}).get("outcome_rows")
    if not isinstance(outcome_rows, list):
        raise ValueError("D1.5 event outcomes are absent")
    outcome_by_key = {}
    for index, row in enumerate(outcome_rows):
        if not isinstance(row, dict):
            raise ValueError(f"D1.5 event outcome {index} is malformed")
        key = (
            row.get("channel"),
            row.get("route"),
            row.get("video_name"),
            _nonnegative_int(row.get("event_id"), f"outcome[{index}].event_id"),
        )
        if key in outcome_by_key:
            raise ValueError("D1.5 event outcome identity is duplicated")
        for field in (
            "primary_success",
            "near_end_end",
            "immutable_emission",
            "premature_cancel",
            "unresolved_identity",
            "ambiguous_identity",
            "target_end_observed_in_trace",
        ):
            if row.get(field) not in (0, 1, False, True):
                raise ValueError(f"D1.5 event outcome {field} is not binary")
        outcome_by_key[key] = row
    expected_outcomes = {
        (channel, route, video_name, event_id)
        for channel in CHANNELS
        for route in ROUTES
        for video_name, event_id in visible_keys
    }
    if set(outcome_by_key) != expected_outcomes:
        raise ValueError("D1.5 event outcome coverage drifted")
    return events, outcome_by_key


def _build_event_traces(
    events: list[dict], outcome_by_key: Mapping[tuple[str, str, str, int], dict]
) -> tuple[dict[tuple[str, str, str, int], EventTrace], dict[tuple[str, int], dict]]:
    event_by_key = {
        (event["video_name"], int(event["event_id"])): event for event in events
    }
    traces = {}
    for event in events:
        if event.get("observation_status") != "fully_observed":
            continue
        video_name = event["video_name"]
        event_id = int(event["event_id"])
        for channel in CHANNELS:
            for route in ROUTES:
                key = (channel, route, video_name, event_id)
                traces[key] = EventTrace(
                    channel=channel,
                    route=route,
                    video_name=video_name,
                    event_id=event_id,
                    class_label=str(event["class_label"]),
                    duration_frames=_finite(event["duration_frames"], "event duration"),
                    outcome=outcome_by_key[key],
                )
    expected = EXPECTED_COUNTS["fully_observed_event_count"] * len(ROUTE_KEYS)
    if len(traces) != expected:
        raise ValueError("D1.5.1 fully observed route/event denominator drifted")
    return traces, event_by_key


def _parse_trace_row(row: dict, line_number: int) -> tuple:
    missing = TRACE_FIELDS.difference(row)
    if missing:
        raise ValueError(f"trace line {line_number} omitted {sorted(missing)!r}")
    if row["protocol"] != D15_TRACE_PROTOCOL:
        raise ValueError(f"trace line {line_number} protocol drifted")
    channel = row["channel"]
    route = row["route"]
    if channel not in CHANNELS or route not in ROUTES:
        raise ValueError(f"trace line {line_number} route drifted")
    video_name = row["video_name"]
    if not isinstance(video_name, str) or not video_name:
        raise ValueError(f"trace line {line_number} video is invalid")
    if not isinstance(row["source"], str) or not row["source"]:
        raise ValueError(f"trace line {line_number} source is invalid")
    runtime_event_id = _nonnegative_int(
        row["runtime_event_id"], f"trace[{line_number}].runtime_event_id"
    )
    creation_frame = _finite(
        row["creation_frame"], f"trace[{line_number}].creation_frame"
    )
    current_frame = _finite(row["current_frame"], f"trace[{line_number}].current_frame")
    if creation_frame < 0.0 or current_frame < creation_frame:
        raise ValueError(f"trace line {line_number} lifetime drifted")
    if not isinstance(row["is_eos"], bool):
        raise ValueError(f"trace line {line_number} EOS flag drifted")
    logits = [
        _finite(row[name], f"trace[{line_number}].{name}")
        for name in ("cancel_logit", "continue_logit", "end_logit")
    ]
    margins = [
        _finite(row[name], f"trace[{line_number}].{name}")
        for name in ("cancel_margin", "continue_margin", "end_margin")
    ]
    expected_margins = [
        logits[index] - max(logits[:index] + logits[index + 1 :]) for index in range(3)
    ]
    if any(
        not math.isclose(observed, expected, rel_tol=1e-6, abs_tol=1e-6)
        for observed, expected in zip(margins, expected_margins)
    ):
        raise ValueError(f"trace line {line_number} state margins drifted")
    winner = _nonnegative_int(row["state_argmax"], f"trace[{line_number}].winner")
    if winner != max(range(3), key=lambda index: logits[index]):
        raise ValueError(f"trace line {line_number} state winner drifted")
    if not isinstance(row["first_owner_decision"], bool):
        raise ValueError(f"trace line {line_number} first-decision flag drifted")
    if not isinstance(row["target_end_observable_now"], bool):
        raise ValueError(f"trace line {line_number} endpoint flag drifted")
    if not isinstance(row["end_suppressed_by_minimum_duration"], bool):
        raise ValueError(f"trace line {line_number} suppression flag drifted")
    owner_before = _nonnegative_int(
        row["owner_query_before_refresh"],
        f"trace[{line_number}].owner_query_before_refresh",
    )
    owner_after = _nonnegative_int(
        row["owner_query_after_refresh"],
        f"trace[{line_number}].owner_query_after_refresh",
    )
    formal_owner = _nonnegative_int(
        row["formal_owner_query_id_after_intervention"],
        f"trace[{line_number}].formal_owner_query_id_after_intervention",
    )
    matched_query = _optional_nonnegative_int(
        row["matched_current_query"], f"trace[{line_number}].matched_current_query"
    )
    attention_top = _nonnegative_int(
        row["owner_attention_top_query"],
        f"trace[{line_number}].owner_attention_top_query",
    )
    oracle_rank = _optional_nonnegative_int(
        row["oracle_query_attention_rank"],
        f"trace[{line_number}].oracle_query_attention_rank",
    )
    for query in (owner_before, owner_after, formal_owner, attention_top):
        if query >= QUERY_COUNT:
            raise ValueError(f"trace line {line_number} query exceeds bandwidth")
    if matched_query is not None and matched_query >= QUERY_COUNT:
        raise ValueError(f"trace line {line_number} matched query exceeds bandwidth")
    if oracle_rank is not None and not 1 <= oracle_rank <= QUERY_COUNT:
        raise ValueError(f"trace line {line_number} oracle rank drifted")
    if (
        row["formal_owner_query_id_unchanged"] is not True
        or formal_owner != owner_before
    ):
        raise ValueError(f"trace line {line_number} formal owner identity drifted")
    if matched_query is None and owner_after != owner_before:
        raise ValueError(f"trace line {line_number} unmatched refresh drifted")
    if matched_query is not None and (
        CHANNEL_FACTORS[channel][1] != "refreshed" or owner_after != matched_query
    ):
        raise ValueError(f"trace line {line_number} refreshed owner identity drifted")
    if CHANNEL_FACTORS[channel][1] == "free" and matched_query is not None:
        raise ValueError(f"trace line {line_number} free route consumed a refresh")
    owner_to_birth = _finite(
        row["owner_to_birth_cosine"], f"trace[{line_number}].owner_to_birth_cosine"
    )
    if not -1.00001 <= owner_to_birth <= 1.00001:
        raise ValueError(f"trace line {line_number} birth cosine drifted")
    owner_to_oracle = (
        None
        if row["owner_to_oracle_query_cosine"] is None
        else _finite(
            row["owner_to_oracle_query_cosine"],
            f"trace[{line_number}].owner_to_oracle_query_cosine",
        )
    )
    if owner_to_oracle is not None and not -1.00001 <= owner_to_oracle <= 1.00001:
        raise ValueError(f"trace line {line_number} oracle cosine drifted")
    if (owner_to_oracle is None) != (oracle_rank is None):
        raise ValueError(f"trace line {line_number} oracle query audit drifted")
    entropy = _finite(
        row["owner_attention_entropy"],
        f"trace[{line_number}].owner_attention_entropy",
    )
    if entropy < 0.0:
        raise ValueError(f"trace line {line_number} attention entropy drifted")
    lifetime_field = "formal_lifetime" if route == "formal" else "shadow_lifetime"
    other_lifetime_field = "shadow_lifetime" if route == "formal" else "formal_lifetime"
    if row[other_lifetime_field] is not None:
        raise ValueError(f"trace line {line_number} mixed route lifetimes")
    lifetime = _finite(row[lifetime_field], f"trace[{line_number}].route_lifetime")
    if not math.isclose(
        lifetime, current_frame - creation_frame, rel_tol=1e-6, abs_tol=1e-6
    ):
        raise ValueError(f"trace line {line_number} route lifetime drifted")
    transition_field = "formal_transition" if route == "formal" else "shadow_transition"
    other_field = "shadow_transition" if route == "formal" else "formal_transition"
    transition = row[transition_field]
    if not isinstance(transition, str) or row[other_field] is not None:
        raise ValueError(f"trace line {line_number} transition fields drifted")
    sequence = transition.split("+")
    parts = set(sequence)
    if (
        not transition
        or len(sequence) != len(parts)
        or not parts.issubset(TRANSITION_PARTS)
    ):
        raise ValueError(f"trace line {line_number} transition drifted")
    suppressed = row["end_suppressed_by_minimum_duration"]
    if suppressed and winner != 2:
        raise ValueError(f"trace line {line_number} suppressed a non-END winner")
    if winner == 0:
        expected = "cancel" if route == "formal" else "diagnostic_retain_cancel"
        if transition != expected:
            raise ValueError(f"trace line {line_number} cancel transition drifted")
    elif winner == 1 and transition != "continue":
        raise ValueError(f"trace line {line_number} continue transition drifted")
    elif winner == 2 and suppressed and transition != "end_suppressed_min_duration":
        raise ValueError(f"trace line {line_number} suppressed END drifted")
    elif (
        winner == 2
        and not suppressed
        and ("end" not in parts or not parts.issubset({"end", "emit"}))
    ):
        raise ValueError(f"trace line {line_number} END transition drifted")
    return (
        channel,
        route,
        video_name,
        runtime_event_id,
        current_frame,
        margins,
        winner,
        suppressed,
        parts,
    )


def analyze_trace(
    trace_path: Path,
    *,
    event_traces: dict[tuple[str, str, str, int], EventTrace],
    event_by_key: Mapping[tuple[str, int], dict],
    outcome_by_key: Mapping[tuple[str, str, str, int], dict],
    receipt: dict,
) -> dict:
    route_line_counts = Counter()
    route_winner_counts: dict[str, Counter] = defaultdict(Counter)
    route_suppressed_counts = Counter()
    route_transition_counts: dict[str, Counter] = defaultdict(Counter)
    last_frame: dict[tuple[str, str, str], float] = {}
    event_ids_at_frame: dict[tuple[str, str, str], set[int]] = {}
    seen_event_ids: dict[tuple[str, str, str], set[int]] = {}
    record_targets: dict[tuple[str, str, str, int], int | None] = {}
    excluded_target_status_row_counts = Counter()
    excluded_target_runtime_records: dict[
        tuple[str, str, str, int], set[int]
    ] = defaultdict(set)
    line_count = 0
    try:
        with gzip.open(trace_path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"trace line {line_number} is not valid JSON"
                    ) from error
                if not isinstance(row, dict):
                    raise ValueError(f"trace line {line_number} is not an object")
                (
                    channel,
                    route,
                    video_name,
                    runtime_event_id,
                    current_frame,
                    margins,
                    winner,
                    suppressed,
                    transition_parts,
                ) = _parse_trace_row(row, line_number)
                route_key = f"{channel}/{route}"
                stream_key = (channel, route, video_name)
                previous_frame = last_frame.get(stream_key)
                if previous_frame is not None and current_frame < previous_frame:
                    raise ValueError(f"trace line {line_number} is not chronological")
                if previous_frame is None or current_frame > previous_frame:
                    last_frame[stream_key] = current_frame
                    event_ids_at_frame[stream_key] = set()
                if runtime_event_id in event_ids_at_frame[stream_key]:
                    raise ValueError(
                        f"trace line {line_number} duplicates an owner decision"
                    )
                event_ids_at_frame[stream_key].add(runtime_event_id)
                seen = seen_event_ids.setdefault(stream_key, set())
                if row["first_owner_decision"] != (runtime_event_id not in seen):
                    raise ValueError(
                        f"trace line {line_number} first-decision audit drifted"
                    )
                seen.add(runtime_event_id)
                route_line_counts[route_key] += 1
                route_winner_counts[route_key][str(winner)] += 1
                route_suppressed_counts[route_key] += int(suppressed)
                for part in transition_parts:
                    route_transition_counts[route_key][part] += 1
                target_raw = row["diagnostic_target_event_id"]
                target_event_id_or_none = (
                    None
                    if target_raw is None
                    else _nonnegative_int(
                        target_raw, f"trace[{line_number}].target_event_id"
                    )
                )
                record_key = (channel, route, video_name, runtime_event_id)
                previous_target = record_targets.get(record_key)
                if (
                    previous_target is not None
                    and target_event_id_or_none != previous_target
                ):
                    raise ValueError(
                        f"trace line {line_number} dropped or changed target identity"
                    )
                if (
                    record_key not in record_targets
                    or target_event_id_or_none is not None
                ):
                    record_targets[record_key] = target_event_id_or_none
                if target_raw is not None:
                    target_event_id = target_event_id_or_none
                    target_key = (video_name, target_event_id)
                    event = event_by_key.get(target_key)
                    if event is None:
                        raise ValueError(
                            f"trace line {line_number} links an unknown event"
                        )
                    target_end = _finite(
                        row["diagnostic_target_end_frame"],
                        f"trace[{line_number}].target_end_frame",
                    )
                    distance = _finite(
                        row["frames_from_annotated_end"],
                        f"trace[{line_number}].frames_from_end",
                    )
                    if not math.isclose(
                        target_end,
                        _finite(event["end_frame"], "event end"),
                        rel_tol=1e-7,
                        abs_tol=1e-7,
                    ) or not math.isclose(
                        current_frame - target_end,
                        distance,
                        rel_tol=1e-6,
                        abs_tol=1e-6,
                    ):
                        raise ValueError(
                            f"trace line {line_number} target endpoint drifted"
                        )
                    accumulator = event_traces.get(
                        (channel, route, video_name, target_event_id)
                    )
                    if accumulator is not None:
                        accumulator.add_row(
                            runtime_event_id=runtime_event_id,
                            current_frame=current_frame,
                            distance=distance,
                            first_owner_decision=row["first_owner_decision"],
                            cancel_margin=margins[0],
                            end_margin=margins[2],
                            winner=winner,
                            suppressed=suppressed,
                            transition_parts=transition_parts,
                            target_endpoint_observed=row["target_end_observable_now"],
                        )
                    else:
                        status = event.get("observation_status")
                        if status != "right_censored":
                            raise ValueError(
                                f"trace line {line_number} linked a non-visible endpoint"
                            )
                        excluded_target_status_row_counts[status] += 1
                        excluded_target_runtime_records[
                            (channel, route, video_name, target_event_id)
                        ].add(runtime_event_id)
                elif (
                    any(
                        row[field] is not None
                        for field in (
                            "diagnostic_target_end_frame",
                            "frames_from_annotated_end",
                        )
                    )
                    or row["target_end_observable_now"]
                ):
                    raise ValueError(
                        f"trace line {line_number} leaked an unlinked endpoint"
                    )
                line_count += 1
    except (OSError, UnicodeError) as error:
        raise ValueError("D1.5 trace cannot be decoded as gzip JSONL") from error

    receipt_line_counts = receipt.get("trace_integrity", {}).get("route_line_counts")
    observed_line_counts = {key: route_line_counts[key] for key in ROUTE_KEYS}
    if observed_line_counts != receipt_line_counts:
        raise ValueError("D1.5.1 trace route line counts differ from D1.5 receipt")
    for channel in CHANNELS:
        for route in ROUTES:
            route_key = f"{channel}/{route}"
            route_receipt = receipt["routes"][channel][route]
            expected_winners = {
                state: int(route_receipt["owner_state_winner_counts"].get(state, 0))
                for state in ("0", "1", "2")
            }
            observed_winners = {
                state: route_winner_counts[route_key][state]
                for state in ("0", "1", "2")
            }
            if observed_winners != expected_winners:
                raise ValueError(f"D1.5.1 {route_key} winner counts drifted")
            expected_suppressed = route_receipt["counts"][
                "end_min_duration_suppression_count"
            ]
            if route_suppressed_counts[route_key] != expected_suppressed:
                raise ValueError(f"D1.5.1 {route_key} END suppression count drifted")
            for part in ("cancel", "diagnostic_retain_cancel", "end", "emit"):
                expected = int(route_receipt["transition_counts"].get(part, 0))
                if route_transition_counts[route_key][part] != expected:
                    raise ValueError(f"D1.5.1 {route_key} {part} count drifted")
    right_censored_events = {
        key
        for key, event in event_by_key.items()
        if event.get("observation_status") == "right_censored"
    }
    if len(right_censored_events) != EXPECTED_COUNTS["right_censored_event_count"]:
        raise ValueError("D1.5.1 right-censored identity count drifted")
    for video_name, event_id in right_censored_events:
        for channel in CHANNELS:
            for route in ROUTES:
                key = (channel, route, video_name, event_id)
                expected_records = _nonnegative_int(
                    outcome_by_key[key].get("linked_runtime_record_count"),
                    f"right-censored {channel}/{route} linked record count",
                )
                if (
                    len(excluded_target_runtime_records.get(key, set()))
                    != expected_records
                ):
                    raise ValueError(
                        f"D1.5.1 {channel}/{route} right-censored target coverage drifted"
                    )
    return {
        "line_count": line_count,
        "route_line_counts": observed_line_counts,
        "route_winner_counts_cross_checked": True,
        "route_transition_counts_cross_checked": True,
        "target_endpoints_cross_checked": True,
        "right_censored_events_excluded_from_endpoint_denominator": True,
        "excluded_target_status_row_counts": dict(
            sorted(excluded_target_status_row_counts.items())
        ),
        "right_censored_target_identities_cross_checked": True,
        "right_censored_runtime_record_coverage_cross_checked": True,
        "chronological_per_route_video": True,
        "owner_decisions_unique_per_prefix": True,
        "first_owner_decisions_cross_checked": True,
        "owner_identity_and_lifetime_cross_checked": True,
    }


def _cross_check_event_outcomes(event_traces: Iterable[EventTrace]) -> None:
    for event in event_traces:
        endpoint_end = bool(
            {"pre_window", "post_window"}.intersection(event.target_backed_end_bins)
        )
        endpoint_primary = bool(
            {"pre_window", "post_window"}.intersection(event.primary_closure_bins)
        )
        expected = event.outcome
        checks = {
            "near_end_end": endpoint_end,
            "immutable_emission": event.immutable_emission_any,
            "primary_success": endpoint_primary,
            "premature_cancel": event.pre_end_cancel,
            "target_end_observed_in_trace": event.target_endpoint_observed,
        }
        for field, observed in checks.items():
            if bool(expected[field]) != observed:
                raise ValueError(
                    f"D1.5.1 {event.route_key} {event.video_name}/{event.event_id} "
                    f"{field} differs from the frozen D1.5 outcome"
                )


def _temporal_group_summary(events: list[EventTrace], bin_name: str) -> dict:
    margins = [event.bin_maximum_end_margin[bin_name] for event in events]
    return {
        "event_denominator": len(events),
        "target_linked_event_count": sum(
            event.bin_row_counts[bin_name] > 0 for event in events
        ),
        "decision_row_count": sum(event.bin_row_counts[bin_name] for event in events),
        "winner_row_counts": {
            state: sum(event.bin_winner_counts[bin_name][state] for event in events)
            for state in ("0", "1", "2")
        },
        "per_event_maximum_end_margin": _value_summary(margins),
        "positive_maximum_end_margin_event_count": sum(
            margin is not None and margin > 0.0 for margin in margins
        ),
        "target_backed_end_event_count": sum(
            bin_name in event.target_backed_end_bins for event in events
        ),
        "immutable_emission_event_count": sum(
            bin_name in event.immutable_emission_bins for event in events
        ),
        "primary_closure_event_count": sum(
            bin_name in event.primary_closure_bins for event in events
        ),
    }


def summarize_events(events: Iterable[EventTrace]) -> dict:
    events = list(events)
    endpoint_margins = [event.endpoint_maximum_end_margin for event in events]
    endpoint_bins = {"pre_window", "post_window"}
    return {
        "event_denominator": len(events),
        "target_linked_event_count": sum(event.row_count > 0 for event in events),
        "target_linked_runtime_record_count": sum(
            len(event.runtime_record_ids) for event in events
        ),
        "decision_row_count": sum(event.row_count for event in events),
        "winner_row_counts": {
            state: sum(event.winner_counts[state] for event in events)
            for state in ("0", "1", "2")
        },
        "endpoint_maximum_end_margin": _value_summary(endpoint_margins),
        "endpoint_positive_end_margin_event_count": sum(
            margin is not None and margin > 0.0 for margin in endpoint_margins
        ),
        "endpoint_positive_end_margin_rate": (
            sum(margin is not None and margin > 0.0 for margin in endpoint_margins)
            / len(events)
            if events
            else None
        ),
        "target_backed_end_event_count": sum(
            event.target_backed_end_any for event in events
        ),
        "immutable_emission_event_count": sum(
            event.immutable_emission_any for event in events
        ),
        "primary_closure_event_count": sum(
            event.primary_closure_any for event in events
        ),
        "endpoint_target_backed_end_event_count": sum(
            bool(endpoint_bins.intersection(event.target_backed_end_bins))
            for event in events
        ),
        "endpoint_immutable_emission_event_count": sum(
            bool(endpoint_bins.intersection(event.immutable_emission_bins))
            for event in events
        ),
        "endpoint_primary_closure_event_count": sum(
            bool(endpoint_bins.intersection(event.primary_closure_bins))
            for event in events
        ),
        "first_decision_cancel_margin": _value_summary(
            event.first_decision_cancel_margin for event in events
        ),
        "last_pre_end_margin": _value_summary(
            event.last_pre_end_margin for event in events
        ),
        "first_post_end_margin": _value_summary(
            event.first_post_end_margin for event in events
        ),
        "post_minus_pre_end_margin": _value_summary(
            event.post_minus_pre_end_margin for event in events
        ),
        "unresolved_identity_event_count": sum(
            bool(event.outcome["unresolved_identity"]) for event in events
        ),
        "ambiguous_identity_event_count": sum(
            bool(event.outcome["ambiguous_identity"]) for event in events
        ),
        "pre_end_cancel_event_count": sum(event.pre_end_cancel for event in events),
        "target_endpoint_observed_event_count": sum(
            event.target_endpoint_observed for event in events
        ),
        "endpoint_eligible_end_without_emit_event_count": sum(
            event.endpoint_eligible_end_without_emit for event in events
        ),
        "temporal_bins": {
            bin_name: _temporal_group_summary(events, bin_name)
            for bin_name in TEMPORAL_BINS
        },
    }


def cluster_bootstrap_endpoint_rates(
    event_traces: Mapping[tuple[str, str, str, int], EventTrace],
    *,
    video_names: list[str],
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    if replicates <= 0 or not video_names:
        raise ValueError("cluster bootstrap requires positive replication and videos")
    event_keys_by_route: dict[str, set[tuple[str, int]]] = {
        key: set() for key in ROUTE_KEYS
    }
    for event in event_traces.values():
        event_keys_by_route[event.route_key].add((event.video_name, event.event_id))
    reference_event_keys = event_keys_by_route[ROUTE_KEYS[0]]
    if not reference_event_keys or any(
        keys != reference_event_keys for keys in event_keys_by_route.values()
    ):
        raise ValueError("cluster bootstrap route/event coverage drifted")
    if len(event_traces) != len(reference_event_keys) * len(ROUTE_KEYS):
        raise ValueError("cluster bootstrap duplicated route/event identities")
    cluster_counts = Counter(video_name for video_name, _ in reference_event_keys)
    cluster_positive: dict[str, Counter] = {key: Counter() for key in ROUTE_KEYS}
    for event in event_traces.values():
        if event.endpoint_maximum_end_margin is not None and (
            event.endpoint_maximum_end_margin > 0.0
        ):
            cluster_positive[event.route_key][event.video_name] += 1
    denominator = sum(cluster_counts.values())
    if denominator != len(reference_event_keys):
        raise ValueError("cluster bootstrap denominator drifted")
    if len(reference_event_keys) == EXPECTED_COUNTS["fully_observed_event_count"] and (
        set(video_names) != {video_name for video_name, _ in reference_event_keys}
    ):
        raise ValueError("cluster bootstrap official video clusters drifted")
    observed_rates = {
        route_key: sum(cluster_positive[route_key].values()) / denominator
        for route_key in ROUTE_KEYS
    }
    rate_samples = {route_key: [] for route_key in ROUTE_KEYS}
    difference_samples = []
    rng = random.Random(seed)
    for _ in range(replicates):
        sampled = [video_names[rng.randrange(len(video_names))] for _ in video_names]
        sampled_denominator = sum(cluster_counts[video] for video in sampled)
        if sampled_denominator <= 0:
            raise ValueError("cluster bootstrap sampled no fully observed event")
        sampled_rates = {}
        for route_key in ROUTE_KEYS:
            numerator = sum(cluster_positive[route_key][video] for video in sampled)
            sampled_rates[route_key] = numerator / sampled_denominator
            rate_samples[route_key].append(sampled_rates[route_key])
        difference_samples.append(
            sampled_rates["OR/shadow"] - sampled_rates["OF/shadow"]
        )
    rates = {
        route_key: {
            "positive_event_count": sum(cluster_positive[route_key].values()),
            "event_denominator": denominator,
            "point_rate": observed_rates[route_key],
            "video_cluster_bootstrap_ci95": [
                nearest_rank(rate_samples[route_key], 0.025),
                nearest_rank(rate_samples[route_key], 0.975),
            ],
        }
        for route_key in ROUTE_KEYS
    }
    observed_difference = observed_rates["OR/shadow"] - observed_rates["OF/shadow"]
    return {
        "unit": "video",
        "method": "percentile_nearest_rank",
        "replicates": replicates,
        "seed": seed,
        "common_resamples_across_routes": True,
        "rates": rates,
        "paired_oracle_identity_difference": {
            "definition": "OR_shadow_minus_OF_shadow_endpoint_positive_rate",
            "point_difference": observed_difference,
            "video_cluster_bootstrap_ci95": [
                nearest_rank(difference_samples, 0.025),
                nearest_rank(difference_samples, 0.975),
            ],
        },
    }


def route_diagnosis(route_summaries: Mapping[str, dict], uncertainty: dict) -> dict:
    rates = uncertainty["rates"]
    minimum_material_events = math.ceil(
        MATERIAL_RATE * EXPECTED_COUNTS["fully_observed_event_count"]
    )
    state_faults = [
        route_key
        for route_key in ROUTE_KEYS
        if route_summaries[route_key]["endpoint_eligible_end_without_emit_event_count"]
        >= minimum_material_events
    ]
    if state_faults:
        return {
            "diagnosis": "material_endpoint_state_machine_mismatch",
            "authorized_next_intervention": "state_machine_integrity_repair_design_only",
            "model_implementation_authorized": False,
            "model_training_authorized": False,
            "reason": "material unsuppressed END winners lacked same-row target-backed end and emission",
            "affected_routes": state_faults,
        }
    identity = uncertainty["paired_oracle_identity_difference"]
    identity_ci = identity["video_cluster_bootstrap_ci95"]
    identity_excludes_zero = identity_ci[0] > 0.0 or identity_ci[1] < 0.0
    if abs(identity["point_difference"]) >= MATERIAL_RATE and identity_excludes_zero:
        return {
            "diagnosis": "material_oracle_identity_attribution_difference",
            "authorized_next_intervention": "preregister_identity_attribution_diagnostic_only",
            "model_implementation_authorized": False,
            "model_training_authorized": False,
            "reason": "oracle refreshed and free identity routes differ materially with nonzero uncertainty interval",
        }
    material_cancel_routes = []
    for channel in ("OF", "OR"):
        shadow = rates[f"{channel}/shadow"]
        formal = rates[f"{channel}/formal"]
        if (
            shadow["point_rate"] >= MATERIAL_RATE
            and shadow["video_cluster_bootstrap_ci95"][0] > 0.0
            and formal["positive_event_count"] == 0
        ):
            material_cancel_routes.append(channel)
    if material_cancel_routes:
        return {
            "diagnosis": "material_cancel_transition_bottleneck",
            "authorized_next_intervention": "preregister_prospective_hold_cancel_diagnostic_only",
            "model_implementation_authorized": False,
            "model_training_authorized": False,
            "reason": "oracle no-cancel shadow crossed the frozen material endpoint-margin gate while formal did not",
            "affected_channels": material_cancel_routes,
        }
    oracle_shadow_keys = ("OF/shadow", "OR/shadow")
    insufficient = all(
        rates[key]["point_rate"] < MATERIAL_RATE
        and rates[key]["video_cluster_bootstrap_ci95"][1] < MATERIAL_RATE
        and route_summaries[key]["endpoint_maximum_end_margin"]["median_nearest_rank"]
        <= 0.0
        and route_summaries[key]["endpoint_maximum_end_margin"]["p95_nearest_rank"]
        <= 0.0
        for key in oracle_shadow_keys
    )
    if insufficient:
        return {
            "diagnosis": "frozen_owner_end_representation_or_competing_risk_objective_insufficient",
            "authorized_next_intervention": "preregister_policy_independent_three_state_competing_risk_model_repair",
            "model_implementation_authorized": False,
            "model_training_authorized": False,
            "reason": "perfect admission and no-cancel still left oracle endpoint END margins below the frozen material gate",
        }
    return {
        "diagnosis": "endpoint_margin_attribution_unresolved",
        "authorized_next_intervention": "no_model_repair_until_another_preregistered_diagnostic",
        "model_implementation_authorized": False,
        "model_training_authorized": False,
        "reason": "the frozen endpoint-margin routing rules did not select one mechanism",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--scan", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--expected-receipt-sha256", required=True)
    parser.add_argument("--expected-scan-sha256", required=True)
    parser.add_argument("--expected-trace-sha256", required=True)
    parser.add_argument("--expected-d15-source-commit", required=True)
    parser.add_argument("--expected-d15-source-tree", required=True)
    parser.add_argument("--expected-analyzer-source-commit", required=True)
    parser.add_argument("--expected-analyzer-source-tree", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    project_dir = args.project_dir.expanduser().resolve()
    paths = {
        "receipt": args.receipt.expanduser().resolve(),
        "scan": args.scan.expanduser().resolve(),
        "trace": args.trace.expanduser().resolve(),
    }
    output = args.output.expanduser().resolve()
    if not project_dir.is_dir():
        raise ValueError("D1.5.1 project directory is absent")
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"D1.5.1 {name} input is absent: {path}")
    try:
        output.relative_to(project_dir)
    except ValueError:
        pass
    else:
        raise ValueError("D1.5.1 output must be outside the source repository")
    if output.exists():
        raise ValueError("D1.5.1 output already exists; evidence is append-only")
    expected_hashes = {
        "receipt": _valid_sha256(args.expected_receipt_sha256, "receipt digest"),
        "scan": _valid_sha256(args.expected_scan_sha256, "scan digest"),
        "trace": _valid_sha256(args.expected_trace_sha256, "trace digest"),
    }
    analyzer_start = _source_identity(
        project_dir,
        expected_commit=args.expected_analyzer_source_commit,
        expected_tree=args.expected_analyzer_source_tree,
    )
    input_artifacts = {}
    for name, path in paths.items():
        digest = _sha256(path)
        if digest != expected_hashes[name]:
            raise ValueError(f"D1.5.1 {name} SHA-256 mismatch")
        input_artifacts[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
    receipt = _load_object(paths["receipt"], "D1.5 receipt")
    scan = _load_object(paths["scan"], "D1.5 scan")
    events, outcome_by_key = _validate_receipt(
        receipt,
        scan,
        expected_scan_sha256=expected_hashes["scan"],
        expected_trace_sha256=expected_hashes["trace"],
        expected_d15_commit=args.expected_d15_source_commit,
        expected_d15_tree=args.expected_d15_source_tree,
    )
    event_traces, event_by_key = _build_event_traces(events, outcome_by_key)
    trace_validation = analyze_trace(
        paths["trace"],
        event_traces=event_traces,
        event_by_key=event_by_key,
        outcome_by_key=outcome_by_key,
        receipt=receipt,
    )
    _cross_check_event_outcomes(event_traces.values())
    route_summaries = {
        route_key: summarize_events(
            event for event in event_traces.values() if event.route_key == route_key
        )
        for route_key in ROUTE_KEYS
    }
    duration_summaries = {
        route_key: {
            stratum: summarize_events(
                event
                for event in event_traces.values()
                if event.route_key == route_key
                and duration_stratum(event.duration_frames) == stratum
            )
            for stratum in DURATION_STRATA
        }
        for route_key in ROUTE_KEYS
    }
    videos = sorted({event["video_name"] for event in events})
    if len(videos) != EXPECTED_COUNTS["video_count"]:
        raise ValueError("D1.5.1 video cluster count drifted")
    uncertainty = cluster_bootstrap_endpoint_rates(
        event_traces,
        video_names=videos,
    )
    decision = route_diagnosis(route_summaries, uncertainty)
    analyzer_final = _source_identity(
        project_dir,
        expected_commit=args.expected_analyzer_source_commit,
        expected_tree=args.expected_analyzer_source_tree,
    )
    result = {
        "status": "PASS_DIAGNOSTIC",
        "status_semantics": "read_only_attribution_result_not_model_training_or_paper_performance",
        "protocol": PROTOCOL,
        **PAPER_FLAGS,
        "ground_truth_use": "post_forward_evaluation_sidecar_only",
        "model_loaded": False,
        "forward_pass_count": 0,
        "optimizer_constructed": False,
        "data_resampling_for_model": False,
        "input_artifacts": input_artifacts,
        "d15_source_identity": {
            "commit": args.expected_d15_source_commit,
            "tree": args.expected_d15_source_tree,
        },
        "analyzer_source_identity": {
            "start": analyzer_start,
            "final": analyzer_final,
        },
        "preregistered_contract": {
            "fully_observed_event_denominator": EXPECTED_COUNTS[
                "fully_observed_event_count"
            ],
            "temporal_bins": {
                "pre_far": "distance < -64",
                "pre_window": "-64 <= distance < 0",
                "post_window": "0 <= distance <= 64",
                "post_far": "distance > 64",
            },
            "duration_strata": list(DURATION_STRATA),
            "material_rate": MATERIAL_RATE,
            "minimum_material_event_count": math.ceil(
                MATERIAL_RATE * EXPECTED_COUNTS["fully_observed_event_count"]
            ),
            "nearest_rank_statistics": True,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "trace_validation": trace_validation,
        "route_summaries": route_summaries,
        "duration_stratum_summaries": duration_summaries,
        "endpoint_positive_uncertainty": uncertainty,
        "routing_decision": decision,
        "event_rows": [
            event.as_dict()
            for event in sorted(
                event_traces.values(),
                key=lambda value: (
                    value.video_name,
                    value.event_id,
                    value.channel,
                    value.route,
                ),
            )
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(output)
    print(
        json.dumps(
            {"status": result["status"], "routing_decision": decision}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
