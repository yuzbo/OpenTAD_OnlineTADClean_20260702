#!/usr/bin/env python3
"""Verify that a calibration replay changes only tied-frame ledger order."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

from opentad.evaluations.online_instance_metrics import audit_causal_emissions
from opentad.utils.online_protocol import (
    summarize_emission_ledger,
    validate_emission_ledger_summary,
)


SCHEMA_VERSION = "persistent_binding_calibration_replay_verification.v1"
_SEQUENCE_KEYS = ("sequence_id", "sequence", "emission_index")


def _load(path):
    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), dict):
        raise ValueError(f"ledger requires a results mapping: {path}")
    return payload


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(row, keys, label):
    for key in keys:
        if key not in row:
            continue
        value = float(row[key])
        if not math.isfinite(value):
            raise ValueError(f"{label}.{key} must be finite")
        return value
    raise ValueError(f"{label} is missing one of {keys}")


def _stream_key(row, video_id):
    for key in ("runtime_stream_key", "stream_key", "stream_id"):
        if key in row:
            return str(row[key])
    return str(video_id)


def _index_ledger(payload, label):
    events = {}
    streams = defaultdict(list)
    for video_id, rows in payload["results"].items():
        if not isinstance(rows, list):
            raise ValueError(f"{label}.{video_id} rows must be a list")
        for row_index, row in enumerate(rows):
            row_label = f"{label}.{video_id}[{row_index}]"
            if not isinstance(row, dict):
                raise ValueError(f"{row_label} must be an object")
            if str(row.get("video_id")) != str(video_id):
                raise ValueError(f"{row_label} video identity mismatch")
            event_id = row.get("event_id", row.get("emission_id"))
            if event_id is None:
                raise ValueError(f"{row_label} lacks an immutable event id")
            event_id = str(event_id)
            if event_id in events:
                raise ValueError(f"{label} repeats event id {event_id}")
            sequence = _finite(row, _SEQUENCE_KEYS, f"{row_label}.sequence")
            emit_frame = _finite(row, ("emit_frame",), f"{row_label}.emit")
            start_frame = _finite(
                row,
                ("start_frame", "predicted_start_frame"),
                f"{row_label}.start",
            )
            end_frame = _finite(
                row,
                ("end_frame", "predicted_end_frame"),
                f"{row_label}.end",
            )
            source_frame = _finite(
                row,
                ("source_frame",),
                f"{row_label}.source",
            )
            if not (0 <= start_frame < end_frame <= source_frame <= emit_frame):
                raise ValueError(f"{row_label} is not a positive causal interval")
            if row.get("immutable") is not True:
                raise ValueError(f"{row_label} is not immutable")
            events[event_id] = row
            streams[_stream_key(row, video_id)].append(
                {
                    "event_id": event_id,
                    "sequence": sequence,
                    "emit_frame": emit_frame,
                }
            )
    return events, streams


def _audit_source_order(streams):
    tied_sequence_violations = 0
    for stream_key, rows in streams.items():
        sequences = [row["sequence"] for row in rows]
        if len(sequences) != len(set(sequences)):
            raise ValueError(f"source stream repeats sequence: {stream_key}")
        if sorted(sequences) != [float(index) for index in range(len(rows))]:
            raise ValueError(f"source stream sequence is not contiguous: {stream_key}")
        for previous, current in zip(rows, rows[1:]):
            if current["emit_frame"] < previous["emit_frame"]:
                raise ValueError(f"source stream emit frame goes backward: {stream_key}")
            if current["sequence"] <= previous["sequence"]:
                if current["emit_frame"] != previous["emit_frame"]:
                    raise ValueError(
                        f"source sequence inversion crosses emit frames: {stream_key}"
                    )
                tied_sequence_violations += 1
        canonical = sorted(rows, key=lambda row: row["sequence"])
        for previous, current in zip(canonical, canonical[1:]):
            if current["emit_frame"] < previous["emit_frame"]:
                raise ValueError(
                    f"source native sequence has backward time: {stream_key}"
                )
    return tied_sequence_violations


def _audit_replay_order(streams):
    for stream_key, rows in streams.items():
        sequences = [row["sequence"] for row in rows]
        if sequences != [float(index) for index in range(len(rows))]:
            raise ValueError(f"replay stream sequence is not strictly ordered: {stream_key}")
        for previous, current in zip(rows, rows[1:]):
            if current["emit_frame"] < previous["emit_frame"]:
                raise ValueError(f"replay stream emit frame goes backward: {stream_key}")


def verify_replay(*, source_ledger, replay_ledger):
    source_ledger = Path(source_ledger).resolve()
    replay_ledger = Path(replay_ledger).resolve()
    source = _load(source_ledger)
    replay = _load(replay_ledger)
    if set(source["results"]) != set(replay["results"]):
        raise ValueError("calibration replay changed the video set")

    source_events, source_streams = _index_ledger(source, "source")
    replay_events, replay_streams = _index_ledger(replay, "replay")
    if source_events != replay_events:
        raise ValueError("calibration replay changed an emitted event payload")
    if set(source_streams) != set(replay_streams):
        raise ValueError("calibration replay changed the runtime stream set")

    source_tied_sequence_violations = _audit_source_order(source_streams)
    _audit_replay_order(replay_streams)
    replay_causal_audit = audit_causal_emissions(replay)
    if replay_causal_audit.get("passed") is not True:
        raise ValueError("calibration replay failed the full causal emission audit")
    replay_summary = summarize_emission_ledger(replay["results"])
    validate_emission_ledger_summary(replay_summary)

    return {
        "schema_version": SCHEMA_VERSION,
        "source_ledger_path": str(source_ledger),
        "source_ledger_sha256": _sha256(source_ledger),
        "replay_ledger_path": str(replay_ledger),
        "replay_ledger_sha256": _sha256(replay_ledger),
        "num_videos": len(replay["results"]),
        "num_streams": len(replay_streams),
        "num_emissions": len(replay_events),
        "source_tied_sequence_violations": source_tied_sequence_violations,
        "replay_violation_counts": replay_causal_audit["violation_counts"],
        "event_payloads_identical": True,
        "reorder_only": True,
        "positive_causal_intervals": True,
        "immutable_emissions": True,
        "reporting_accessed": False,
        "threshold_search": False,
        "raw_rgb_authorized": False,
        "passed": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ledger", required=True)
    parser.add_argument("--replay-ledger", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = verify_replay(
        source_ledger=args.source_ledger,
        replay_ledger=args.replay_ledger,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    json.dump(payload, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
