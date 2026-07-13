import hashlib
import json

import pytest

from opentad.utils.immutable_event_ledger import (
    GENESIS_HASH,
    AtomicLedgerWriter,
    ImmutableEventLedger,
    LedgerValidationError,
    LedgerVerificationError,
    canonical_json,
    compute_row_hash,
    load_verified_ledger,
    persist_verified_ledger,
    read_ledger,
    verify_ledger,
    verify_rows,
)


def _event(event_id, stream_id="stream-a", emit_frame=10, **overrides):
    event = {
        "event_id": event_id,
        "stream_id": stream_id,
        "video_id": "video-a",
        "emit_frame": emit_frame,
        "source_frame": emit_frame,
        "start_frame": max(0, emit_frame - 2),
        "end_frame": emit_frame,
        "immutable": True,
        "slot_id": 0,
        "label": "action",
        "score": 0.75,
        "provenance_digest": "a" * 64,
    }
    event.update(overrides)
    return event


def _write_rows(path, rows):
    payload = "".join(f"{canonical_json(row)}\n" for row in rows)
    path.write_text(payload, encoding="utf-8", newline="")


def test_canonical_json_and_row_hash_are_order_independent():
    left = {"z": 1, "nested": {"b": 2, "a": [3, 4]}}
    right = {"nested": {"a": [3, 4], "b": 2}, "z": 1}

    assert canonical_json(left) == '{"nested":{"a":[3,4],"b":2},"z":1}'
    assert canonical_json(left) == canonical_json(right)
    assert compute_row_hash(left) == compute_row_hash(right)
    assert compute_row_hash(left) == hashlib.sha256(canonical_json(left).encode("utf-8")).hexdigest()


def test_in_memory_ledger_builds_independent_per_stream_chains():
    ledger = ImmutableEventLedger()

    a0 = ledger.append(_event("a-0", emit_frame=4))
    b0 = ledger.append(_event("b-0", stream_id="stream-b", emit_frame=2))
    a1 = ledger.append(_event("a-1", emit_frame=7))

    assert [a0["sequence"], b0["sequence"], a1["sequence"]] == [0, 0, 1]
    assert a0["previous_hash"] == GENESIS_HASH
    assert b0["previous_hash"] == GENESIS_HASH
    assert a1["previous_hash"] == a0["row_hash"]
    assert all(row["schema"] == "opentad.immutable_event_ledger" for row in ledger.rows)
    assert all(row["version"] == 1 for row in ledger.rows)

    report = verify_rows(ledger.rows)
    assert report.count == 3
    assert report.stream_counts == {"stream-a": 2, "stream-b": 1}
    assert report.final_hashes["stream-a"] == a1["row_hash"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"immutable": False}, "immutable=true"),
        ({"immutable": None}, "immutable=true"),
        ({"source_frame": 11}, "source_frame"),
        ({"end_frame": 11}, "end_frame"),
        ({"operation": "delete"}, "append-only"),
        ({"op": "replacement"}, "append-only"),
        ({"type": "DELETE"}, "append-only"),
        ({"payload": {"action": "mutate"}}, "append-only"),
        ({"payload": {"replacesEventId": "old-event"}}, "append-only"),
        ({"score": float("nan")}, "finite"),
    ],
)
def test_writer_rejects_non_immutable_future_mutating_or_noncanonical_events(change, message):
    ledger = ImmutableEventLedger()

    with pytest.raises(LedgerValidationError, match=message):
        ledger.append(_event("bad", **change))


def test_writer_requires_immutable_marker_and_rejects_reserved_fields():
    ledger = ImmutableEventLedger()
    missing = _event("missing")
    del missing["immutable"]

    with pytest.raises(LedgerValidationError, match="immutable=true"):
        ledger.append(missing)
    with pytest.raises(LedgerValidationError, match="reserved"):
        ledger.append(_event("reserved", sequence=99))


@pytest.mark.parametrize(
    "field",
    ["video_id", "slot_id", "start_frame", "label", "score", "provenance_digest"],
)
def test_writer_requires_complete_formal_emission_provenance(field):
    event = _event("missing-formal-field")
    del event[field]

    with pytest.raises(LedgerValidationError, match=field):
        ImmutableEventLedger().append(event)


def test_writer_rejects_duplicate_ids_and_decreasing_emit_frames():
    ledger = ImmutableEventLedger()
    ledger.append(_event("first", emit_frame=10))

    with pytest.raises(LedgerValidationError, match="duplicate event_id"):
        ledger.append(_event("first", stream_id="stream-b", emit_frame=11))
    with pytest.raises(LedgerValidationError, match="decreased"):
        ledger.append(_event("second", emit_frame=9))


def test_builder_detaches_rows_from_caller_mutation():
    event = _event("detached", payload={"scores": [0.5]})
    ledger = ImmutableEventLedger()

    row = ledger.append(event)
    event["payload"]["scores"][0] = 99
    row["payload"]["scores"][0] = 88

    assert ledger.rows[0]["payload"]["scores"] == [0.5]
    verify_rows(ledger.rows)


def test_atomic_writer_only_appends_and_can_resume_existing_chain(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = AtomicLedgerWriter(path)

    first = writer.append(_event("first", emit_frame=3))
    original_prefix = path.read_bytes()
    second = writer.append(_event("second", emit_frame=5))

    appended_bytes = path.read_bytes()
    assert appended_bytes.startswith(original_prefix)
    assert appended_bytes != original_prefix
    assert first["sequence"] == 0
    assert second["sequence"] == 1

    writer.close()
    resumed = AtomicLedgerWriter(path)
    third = resumed.append(_event("third", emit_frame=8))
    assert third["sequence"] == 2
    assert third["previous_hash"] == second["row_hash"]

    rows = read_ledger(path)
    report = verify_ledger(
        path,
        expected_count=3,
        expected_final_hash=third["row_hash"],
    )
    assert len(rows) == report.count == 3
    resumed.close()


def test_atomic_writer_holds_one_exclusive_process_lease(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = AtomicLedgerWriter(path)

    with pytest.raises(LedgerValidationError, match="exclusive writer lease"):
        AtomicLedgerWriter(path)

    writer.close()
    resumed = AtomicLedgerWriter(path)
    resumed.close()


def test_persisted_ledger_requires_external_commitment_and_round_trips(tmp_path):
    source = ImmutableEventLedger()
    rows = (
        source.append(_event("first", emit_frame=3)),
        source.append(_event("second", emit_frame=5)),
    )
    ledger_path = tmp_path / "formal.jsonl"
    commitment_path = tmp_path / "formal.commitment.json"

    commitment = persist_verified_ledger(ledger_path, commitment_path, rows)
    report = load_verified_ledger(ledger_path, commitment_path)

    assert report.rows == rows
    assert commitment["count"] == 2
    assert commitment["final_hashes"] == report.final_hashes

    with pytest.raises(LedgerValidationError, match="already exists"):
        persist_verified_ledger(ledger_path, commitment_path, rows)

    commitment_path.unlink()
    with pytest.raises(LedgerVerificationError, match="commitment"):
        load_verified_ledger(ledger_path, commitment_path)


def test_atomic_writer_rejects_same_size_external_edits(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = AtomicLedgerWriter(path)
    writer.append(_event("first", emit_frame=3, label="action"))
    original = path.read_text(encoding="utf-8")
    edited = original.replace('"label":"action"', '"label":"edited"')
    assert len(edited.encode("utf-8")) == len(original.encode("utf-8"))
    path.write_text(edited, encoding="utf-8", newline="")

    with pytest.raises(LedgerVerificationError, match="changed outside"):
        writer.append(_event("second", emit_frame=4))


def test_verifier_detects_edited_and_reordered_rows(tmp_path):
    ledger = ImmutableEventLedger()
    rows = [ledger.append(_event("one", emit_frame=1)), ledger.append(_event("two", emit_frame=2))]
    path = tmp_path / "events.jsonl"

    edited = json.loads(canonical_json(rows[0]))
    edited["label"] = "edited"
    _write_rows(path, [edited, rows[1]])
    with pytest.raises(LedgerVerificationError, match="hash mismatch"):
        verify_ledger(path)

    _write_rows(path, [rows[1], rows[0]])
    with pytest.raises(LedgerVerificationError, match="reordered|sequence"):
        verify_ledger(path)


def test_verifier_detects_duplicate_event_ids_and_replayed_rows(tmp_path):
    ledger = ImmutableEventLedger()
    first = ledger.append(_event("one", emit_frame=1))
    second = dict(_event("one", emit_frame=2))
    second.update(
        schema=first["schema"],
        version=first["version"],
        sequence=1,
        previous_hash=first["row_hash"],
    )
    second["row_hash"] = compute_row_hash(second)
    path = tmp_path / "events.jsonl"

    _write_rows(path, [first, second])
    with pytest.raises(LedgerVerificationError, match="duplicate event_id"):
        verify_ledger(path)

    _write_rows(path, [first, first])
    with pytest.raises(LedgerVerificationError, match="replayed"):
        verify_ledger(path)


def test_verifier_detects_cross_stream_predecessor(tmp_path):
    ledger = ImmutableEventLedger()
    a0 = ledger.append(_event("a-0", stream_id="stream-a", emit_frame=1))
    b0 = ledger.append(_event("b-0", stream_id="stream-b", emit_frame=1))
    forged = dict(b0)
    forged["previous_hash"] = a0["row_hash"]
    forged["row_hash"] = compute_row_hash(forged)
    path = tmp_path / "events.jsonl"
    _write_rows(path, [a0, forged])

    with pytest.raises(LedgerVerificationError, match="cross-stream predecessor"):
        verify_ledger(path)


def test_expected_count_or_final_hash_detects_truncation(tmp_path):
    writer = AtomicLedgerWriter(tmp_path / "events.jsonl")
    writer.append(_event("one", emit_frame=1))
    final = writer.append(_event("two", emit_frame=2))
    complete = read_ledger(writer.path)
    _write_rows(writer.path, complete[:1])

    with pytest.raises(LedgerVerificationError, match="expected count"):
        verify_ledger(writer.path, expected_count=2)
    with pytest.raises(LedgerVerificationError, match="expected final hash"):
        verify_ledger(writer.path, expected_final_hash=final["row_hash"])
