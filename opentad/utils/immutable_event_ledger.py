"""Append-only, hash-chained JSONL records for online event emissions."""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import re
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path


LEDGER_SCHEMA = "opentad.immutable_event_ledger"
LEDGER_VERSION = 1
GENESIS_HASH = "0" * 64

_ENVELOPE_FIELDS = frozenset(
    {
        "schema",
        "version",
        "sequence",
        "previous_hash",
        "row_hash",
    }
)
_RESERVED_EVENT_FIELDS = _ENVELOPE_FIELDS | frozenset({"prev_hash"})
_MUTATION_OPERATION_KEYS = frozenset(
    {
        "action",
        "change",
        "command",
        "event_type",
        "kind",
        "mutation",
        "op",
        "operation",
        "type",
    }
)
_MUTATION_OPERATIONS = frozenset(
    {
        "delete",
        "deletion",
        "mutate",
        "mutation",
        "overwrite",
        "patch",
        "remove",
        "removed",
        "replace",
        "replacement",
        "retract",
        "retraction",
        "tombstone",
        "update",
        "upsert",
    }
)
_MUTATION_TARGET_KEYS = frozenset(
    {
        "deleted_event_id",
        "deletes_event_id",
        "mutation_of",
        "patch_of",
        "replaced_event_id",
        "replaces_event_id",
        "retracts_event_id",
        "supersedes",
        "supersedes_event_id",
        "tombstone",
        "updates_event_id",
    }
)
_HEX_DIGITS = frozenset("0123456789abcdef")
_CAMEL_BOUNDARY_1 = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_BOUNDARY_2 = re.compile(r"([a-z0-9])([A-Z])")
_NON_KEY_CHARACTER = re.compile(r"[^A-Za-z0-9]+")


class LedgerError(ValueError):
    """Base class for immutable-ledger contract failures."""


class LedgerValidationError(LedgerError):
    """Raised before a writer accepts an invalid event."""


class LedgerVerificationError(LedgerError):
    """Raised when persisted ledger provenance cannot be verified."""


def _normalize_json(value, path="$", error_type=LedgerValidationError):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise error_type(f"value at {path} is not canonical JSON: non-finite number")
        return normalized
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise error_type(f"value at {path} is not canonical JSON: object keys must be strings")
            normalized[key] = _normalize_json(item, f"{path}.{key}", error_type)
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _normalize_json(item, f"{path}[{index}]", error_type)
            for index, item in enumerate(value)
        ]
    raise error_type(
        f"value at {path} is not canonical JSON serializable: {type(value).__name__}"
    )


def canonical_json(value) -> str:
    """Return deterministic UTF-8 JSON text with strict JSON values."""

    normalized = _normalize_json(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_copy(value):
    return json.loads(canonical_json(value))


def compute_row_hash(row: Mapping) -> str:
    """Hash a row envelope and payload, excluding only its claimed row hash."""

    if not isinstance(row, Mapping):
        raise LedgerValidationError("ledger row must be a mapping")
    payload = dict(row)
    payload.pop("row_hash", None)
    encoded = canonical_json(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_key(key):
    value = _CAMEL_BOUNDARY_1.sub(r"\1_\2", str(key).strip())
    value = _CAMEL_BOUNDARY_2.sub(r"\1_\2", value)
    return _NON_KEY_CHARACTER.sub("_", value).strip("_").lower()


def _reject_mutation_semantics(value, path="event"):
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = _normalized_key(key)
            if normalized_key in _MUTATION_TARGET_KEYS:
                raise LedgerValidationError(
                    f"append-only ledger forbids mutation target {path}.{key}"
                )
            if normalized_key in _MUTATION_OPERATION_KEYS and isinstance(item, str):
                operation = _normalized_key(item)
                if operation in _MUTATION_OPERATIONS:
                    raise LedgerValidationError(
                        f"append-only ledger forbids {operation!r} semantics at {path}.{key}"
                    )
            _reject_mutation_semantics(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_mutation_semantics(item, f"{path}[{index}]")


def _required_text(value, field, error_type):
    if not isinstance(value, str) or not value.strip():
        raise error_type(f"ledger {field} must be a non-empty string")
    return value


def _required_frame(value, field, error_type):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise error_type(f"ledger {field} must be an integer frame index")
    return int(value)


def _validate_event_payload(event, *, error_type=LedgerValidationError, stream_id=None):
    if not isinstance(event, Mapping):
        raise error_type("ledger event must be a mapping")

    collisions = sorted(_RESERVED_EVENT_FIELDS.intersection(event))
    if collisions:
        raise error_type(f"ledger event contains reserved fields: {collisions}")

    payload = dict(event)
    embedded_stream_id = payload.get("stream_id")
    stream_key = payload.get("stream_key")
    if stream_id is None:
        stream_id = embedded_stream_id if embedded_stream_id is not None else stream_key
    if embedded_stream_id is not None and str(embedded_stream_id) != str(stream_id):
        raise error_type("ledger stream_id conflicts with the explicit stream id")
    if stream_key is not None and str(stream_key) != str(stream_id):
        raise error_type("ledger stream_key conflicts with stream_id")

    stream_id = _required_text(stream_id, "stream_id", error_type)
    event_id = _required_text(payload.get("event_id"), "event_id", error_type)
    if payload.get("immutable") is not True:
        raise error_type("ledger event must explicitly contain immutable=true")

    missing_frames = [
        field
        for field in ("emit_frame", "source_frame", "end_frame")
        if field not in payload
    ]
    if missing_frames:
        raise error_type(f"ledger event missing frame fields: {missing_frames}")
    emit_frame = _required_frame(payload["emit_frame"], "emit_frame", error_type)
    source_frame = _required_frame(payload["source_frame"], "source_frame", error_type)
    end_frame = _required_frame(payload["end_frame"], "end_frame", error_type)
    if source_frame > emit_frame:
        raise error_type(
            f"ledger source_frame={source_frame} exceeds emit_frame={emit_frame}"
        )
    if end_frame > emit_frame:
        raise error_type(f"ledger end_frame={end_frame} exceeds emit_frame={emit_frame}")

    _reject_mutation_semantics(payload)
    payload["stream_id"] = stream_id
    payload["event_id"] = event_id
    payload["emit_frame"] = emit_frame
    payload["source_frame"] = source_frame
    payload["end_frame"] = end_frame
    return _json_copy(payload)


def _payload_from_row(row):
    return {key: value for key, value in row.items() if key not in _ENVELOPE_FIELDS}


def _valid_hash(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_HEX_DIGITS)
    )


@dataclass(frozen=True)
class LedgerVerificationResult:
    rows: tuple
    count: int
    stream_counts: dict
    final_hashes: dict

    @property
    def final_hash(self):
        if not self.final_hashes:
            return GENESIS_HASH
        if len(self.final_hashes) != 1:
            raise LedgerVerificationError(
                "a single final_hash is ambiguous for a multi-stream ledger"
            )
        return next(iter(self.final_hashes.values()))


def _check_expected_count(actual_counts, total_count, expected_count):
    if expected_count is None:
        return
    if isinstance(expected_count, Mapping):
        normalized = {
            _required_text(key, "expected stream id", LedgerVerificationError): _required_frame(
                value,
                f"expected count for {key}",
                LedgerVerificationError,
            )
            for key, value in expected_count.items()
        }
        if actual_counts != normalized:
            raise LedgerVerificationError(
                f"ledger expected counts {normalized}, found {actual_counts}; ledger may be truncated"
            )
        return
    count = _required_frame(expected_count, "expected count", LedgerVerificationError)
    if total_count != count:
        raise LedgerVerificationError(
            f"ledger expected count {count}, found {total_count}; ledger may be truncated"
        )


def _check_expected_hash(final_hashes, expected_final_hash):
    if expected_final_hash is None:
        return
    if isinstance(expected_final_hash, Mapping):
        normalized = dict(expected_final_hash)
        if final_hashes != normalized:
            raise LedgerVerificationError(
                f"ledger expected final hashes {normalized}, found {final_hashes}; ledger may be truncated"
            )
        return
    if not _valid_hash(expected_final_hash):
        raise LedgerVerificationError("expected final hash must be a lowercase SHA-256 hex digest")
    if not final_hashes:
        actual = GENESIS_HASH
    elif len(final_hashes) == 1:
        actual = next(iter(final_hashes.values()))
    else:
        raise LedgerVerificationError(
            "expected_final_hash must be a stream-to-hash mapping for a multi-stream ledger"
        )
    if actual != expected_final_hash:
        raise LedgerVerificationError(
            f"ledger expected final hash {expected_final_hash}, found {actual}; ledger may be truncated"
        )


def verify_rows(
    rows: Iterable[Mapping],
    *,
    expected_count=None,
    expected_final_hash=None,
) -> LedgerVerificationResult:
    """Verify row hashes, stream chains, append order, and optional final state."""

    if isinstance(rows, (str, bytes, Mapping)) or not isinstance(rows, Iterable):
        raise LedgerVerificationError("ledger rows must be an iterable of mappings")

    verified = []
    counts = {}
    final_hashes = {}
    event_ids = set()
    row_hashes = set()
    hash_owners = {}

    for line_number, raw_row in enumerate(rows, start=1):
        if not isinstance(raw_row, Mapping):
            raise LedgerVerificationError(f"ledger row {line_number} must be a mapping")
        try:
            row = _normalize_json(
                raw_row,
                path=f"row[{line_number}]",
                error_type=LedgerVerificationError,
            )
        except LedgerVerificationError:
            raise

        missing = sorted(_ENVELOPE_FIELDS.difference(row))
        if missing:
            raise LedgerVerificationError(
                f"ledger row {line_number} missing envelope fields: {missing}"
            )
        if row["schema"] != LEDGER_SCHEMA or row["version"] != LEDGER_VERSION:
            raise LedgerVerificationError(
                f"ledger row {line_number} has unsupported schema/version"
            )

        row_hash = row["row_hash"]
        previous_hash = row["previous_hash"]
        if not _valid_hash(row_hash) or not _valid_hash(previous_hash):
            raise LedgerVerificationError(
                f"ledger row {line_number} contains an invalid SHA-256 hash"
            )
        if row_hash in row_hashes:
            raise LedgerVerificationError(
                f"ledger row {line_number} replayed an existing row hash"
            )

        payload = _payload_from_row(row)
        try:
            payload = _validate_event_payload(payload, error_type=LedgerVerificationError)
        except LedgerVerificationError as exc:
            raise LedgerVerificationError(f"ledger row {line_number}: {exc}") from exc
        stream_id = payload["stream_id"]
        event_id = payload["event_id"]
        if event_id in event_ids:
            raise LedgerVerificationError(
                f"ledger row {line_number} has duplicate event_id {event_id!r}"
            )

        sequence = row["sequence"]
        if isinstance(sequence, bool) or not isinstance(sequence, numbers.Integral):
            raise LedgerVerificationError(
                f"ledger row {line_number} sequence must be an integer"
            )
        sequence = int(sequence)
        expected_sequence = counts.get(stream_id, 0)
        if sequence < expected_sequence:
            raise LedgerVerificationError(
                f"ledger row {line_number} replayed sequence {sequence} for {stream_id!r}"
            )
        if sequence > expected_sequence:
            raise LedgerVerificationError(
                f"ledger row {line_number} is reordered or missing sequence {expected_sequence} "
                f"for {stream_id!r}"
            )

        expected_previous = final_hashes.get(stream_id, GENESIS_HASH)
        if previous_hash != expected_previous:
            owner = hash_owners.get(previous_hash)
            if owner is not None and owner != stream_id:
                raise LedgerVerificationError(
                    f"ledger row {line_number} has cross-stream predecessor from {owner!r}"
                )
            if owner == stream_id:
                raise LedgerVerificationError(
                    f"ledger row {line_number} has a reordered or replayed predecessor"
                )
            raise LedgerVerificationError(
                f"ledger row {line_number} previous hash mismatch; row was edited, reordered, or truncated"
            )

        calculated = compute_row_hash(row)
        if row_hash != calculated:
            raise LedgerVerificationError(
                f"ledger row {line_number} hash mismatch: expected {calculated}, found {row_hash}"
            )

        row["sequence"] = sequence
        verified.append(row)
        counts[stream_id] = expected_sequence + 1
        final_hashes[stream_id] = row_hash
        event_ids.add(event_id)
        row_hashes.add(row_hash)
        hash_owners[row_hash] = stream_id

    _check_expected_count(counts, len(verified), expected_count)
    _check_expected_hash(final_hashes, expected_final_hash)
    return LedgerVerificationResult(
        rows=tuple(_json_copy(row) for row in verified),
        count=len(verified),
        stream_counts=dict(counts),
        final_hashes=dict(final_hashes),
    )


def _reject_json_constant(value):
    raise ValueError(f"non-standard JSON constant {value!r}")


def _strict_json_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def read_ledger(path) -> tuple:
    """Read canonical JSONL rows without accepting alternate encodings."""

    ledger_path = Path(path)
    try:
        with ledger_path.open("r", encoding="utf-8", newline="") as handle:
            lines = handle.readlines()
    except (OSError, UnicodeError) as exc:
        raise LedgerVerificationError(f"unable to read ledger {ledger_path}: {exc}") from exc

    rows = []
    for line_number, raw_line in enumerate(lines, start=1):
        encoded = raw_line[:-1] if raw_line.endswith("\n") else raw_line
        if encoded.endswith("\r"):
            encoded = encoded[:-1]
        if not encoded:
            raise LedgerVerificationError(f"ledger row {line_number} is blank")
        try:
            row = json.loads(
                encoded,
                object_pairs_hook=_strict_json_object,
                parse_constant=_reject_json_constant,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise LedgerVerificationError(
                f"ledger row {line_number} is invalid JSON: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise LedgerVerificationError(f"ledger row {line_number} must be a JSON object")
        try:
            canonical = canonical_json(row)
        except LedgerValidationError as exc:
            raise LedgerVerificationError(f"ledger row {line_number}: {exc}") from exc
        if encoded != canonical:
            raise LedgerVerificationError(
                f"ledger row {line_number} does not use canonical JSON encoding"
            )
        rows.append(row)
    return tuple(rows)


def verify_ledger(
    path,
    *,
    expected_count=None,
    expected_final_hash=None,
) -> LedgerVerificationResult:
    return verify_rows(
        read_ledger(path),
        expected_count=expected_count,
        expected_final_hash=expected_final_hash,
    )


class ImmutableEventLedger:
    """In-memory builder that returns detached append-only event rows."""

    def __init__(self):
        self._rows = []
        self._stream_counts = {}
        self._final_hashes = {}
        self._last_emit_frames = {}
        self._event_ids = set()

    @classmethod
    def from_rows(cls, rows):
        report = verify_rows(rows)
        ledger = cls()
        for row in report.rows:
            ledger._accept_verified_row(row)
        return ledger

    def _prepare(self, event, *, stream_id=None):
        payload = _validate_event_payload(event, stream_id=stream_id)
        event_id = payload["event_id"]
        stream_id = payload["stream_id"]
        emit_frame = payload["emit_frame"]
        if event_id in self._event_ids:
            raise LedgerValidationError(f"duplicate event_id {event_id!r}")
        previous_emit = self._last_emit_frames.get(stream_id)
        if previous_emit is not None and emit_frame < previous_emit:
            raise LedgerValidationError(
                f"emit_frame decreased for {stream_id!r}: {emit_frame} < {previous_emit}"
            )

        row = dict(payload)
        row.update(
            schema=LEDGER_SCHEMA,
            version=LEDGER_VERSION,
            sequence=self._stream_counts.get(stream_id, 0),
            previous_hash=self._final_hashes.get(stream_id, GENESIS_HASH),
        )
        row["row_hash"] = compute_row_hash(row)
        return row

    def _accept_verified_row(self, row):
        stored = _json_copy(row)
        stream_id = stored["stream_id"]
        self._rows.append(stored)
        self._stream_counts[stream_id] = int(stored["sequence"]) + 1
        self._final_hashes[stream_id] = stored["row_hash"]
        self._last_emit_frames[stream_id] = int(stored["emit_frame"])
        self._event_ids.add(stored["event_id"])

    def append(self, event, *, stream_id=None):
        row = self._prepare(event, stream_id=stream_id)
        self._accept_verified_row(row)
        return _json_copy(row)

    append_event = append

    @property
    def rows(self):
        return tuple(_json_copy(row) for row in self._rows)

    @property
    def stream_counts(self):
        return dict(self._stream_counts)

    @property
    def final_hashes(self):
        return dict(self._final_hashes)

    def __len__(self):
        return len(self._rows)


class AtomicLedgerWriter:
    """Append canonical rows with one OS-level append write per event."""

    def __init__(self, path, *, fsync=True):
        self.path = Path(path)
        self.fsync = bool(fsync)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not self.path.is_file():
            raise LedgerValidationError(f"ledger path is not a file: {self.path}")
        if self.path.exists():
            existing = self.path.read_bytes()
            if existing and not existing.endswith(b"\n"):
                raise LedgerVerificationError(
                    "existing ledger does not end at an atomic JSONL record boundary"
                )
            rows = read_ledger(self.path)
            self._ledger = ImmutableEventLedger.from_rows(rows)
            self._size = len(existing)
        else:
            existing = b""
            self._ledger = ImmutableEventLedger()
            self._size = 0
        self._content_hasher = hashlib.sha256(existing)
        self._content_digest = self._content_hasher.digest()

    def _current_state(self):
        digest = hashlib.sha256()
        size = 0
        try:
            with self.path.open("rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    digest.update(chunk)
        except FileNotFoundError:
            pass
        return size, digest.digest()

    def append(self, event, *, stream_id=None):
        with self._lock:
            current_size, current_digest = self._current_state()
            if current_size != self._size or current_digest != self._content_digest:
                raise LedgerVerificationError(
                    "ledger changed outside this writer; reopen and verify before appending"
                )
            row = self._ledger._prepare(event, stream_id=stream_id)
            encoded = (canonical_json(row) + "\n").encode("utf-8")
            flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
            flags |= getattr(os, "O_BINARY", 0)
            descriptor = os.open(self.path, flags, 0o600)
            try:
                if os.fstat(descriptor).st_size != self._size:
                    raise LedgerVerificationError(
                        "ledger changed while preparing an append; reopen before retrying"
                    )
                written = os.write(descriptor, encoded)
                if written != len(encoded):
                    raise OSError(
                        f"partial ledger append wrote {written} of {len(encoded)} bytes"
                    )
                if self.fsync:
                    os.fsync(descriptor)
            finally:
                os.close(descriptor)

            self._ledger._accept_verified_row(row)
            self._size += len(encoded)
            self._content_hasher.update(encoded)
            self._content_digest = self._content_hasher.digest()
            return _json_copy(row)

    append_event = append

    @property
    def rows(self):
        return self._ledger.rows

    @property
    def final_hashes(self):
        return self._ledger.final_hashes

    def __len__(self):
        return len(self._ledger)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class ImmutableEventLedgerReader:
    def __init__(self, path):
        self.path = Path(path)

    def read(self):
        return read_ledger(self.path)

    def verify(self, *, expected_count=None, expected_final_hash=None):
        return verify_ledger(
            self.path,
            expected_count=expected_count,
            expected_final_hash=expected_final_hash,
        )


InMemoryEventLedger = ImmutableEventLedger
AtomicJSONLWriter = AtomicLedgerWriter
LedgerReader = ImmutableEventLedgerReader


__all__ = [
    "AtomicJSONLWriter",
    "AtomicLedgerWriter",
    "GENESIS_HASH",
    "ImmutableEventLedger",
    "ImmutableEventLedgerReader",
    "InMemoryEventLedger",
    "LEDGER_SCHEMA",
    "LEDGER_VERSION",
    "LedgerError",
    "LedgerReader",
    "LedgerValidationError",
    "LedgerVerificationError",
    "LedgerVerificationResult",
    "canonical_json",
    "compute_row_hash",
    "read_ledger",
    "verify_ledger",
    "verify_rows",
]
