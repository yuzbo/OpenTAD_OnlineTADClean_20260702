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
LEDGER_COMMITMENT_SCHEMA = "opentad.immutable_event_ledger.commitment"
LEDGER_COMMITMENT_VERSION = 1
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


def _required_nonnegative_integer(value, field, error_type):
    normalized = _required_frame(value, field, error_type)
    if normalized < 0:
        raise error_type(f"ledger {field} must be non-negative")
    return normalized


def _required_score(value, error_type):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise error_type("ledger score must be a finite real number")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise error_type("ledger score must be finite and lie in [0, 1]")
    return normalized


def _required_label(value, error_type):
    if isinstance(value, bool):
        raise error_type("ledger label must be a non-empty string or non-negative integer")
    if isinstance(value, numbers.Integral):
        normalized = int(value)
        if normalized < 0:
            raise error_type("ledger label integer must be non-negative")
        return normalized
    return _required_text(value, "label", error_type)


def _required_hash(value, field, error_type):
    if not _valid_hash(value):
        raise error_type(f"ledger {field} must be a lowercase SHA-256 hex digest")
    return value


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
    video_id = _required_text(payload.get("video_id"), "video_id", error_type)
    if payload.get("immutable") is not True:
        raise error_type("ledger event must explicitly contain immutable=true")

    missing_frames = [
        field
        for field in ("emit_frame", "source_frame", "end_frame")
        if field not in payload
    ]
    if missing_frames:
        raise error_type(f"ledger event missing frame fields: {missing_frames}")
    emit_frame = _required_nonnegative_integer(payload["emit_frame"], "emit_frame", error_type)
    source_frame = _required_nonnegative_integer(payload["source_frame"], "source_frame", error_type)
    end_frame = _required_nonnegative_integer(payload["end_frame"], "end_frame", error_type)
    start_frame = _required_nonnegative_integer(
        payload.get("start_frame"), "start_frame", error_type
    )
    slot_id = _required_nonnegative_integer(payload.get("slot_id"), "slot_id", error_type)
    label = _required_label(payload.get("label"), error_type)
    score = _required_score(payload.get("score"), error_type)
    provenance_digest = _required_hash(
        payload.get("provenance_digest"), "provenance_digest", error_type
    )
    if source_frame > emit_frame:
        raise error_type(
            f"ledger source_frame={source_frame} exceeds emit_frame={emit_frame}"
        )
    if end_frame > emit_frame:
        raise error_type(f"ledger end_frame={end_frame} exceeds emit_frame={emit_frame}")
    if start_frame > end_frame:
        raise error_type(
            f"ledger start_frame={start_frame} exceeds end_frame={end_frame}"
        )

    _reject_mutation_semantics(payload)
    payload["stream_id"] = stream_id
    payload["event_id"] = event_id
    payload["video_id"] = video_id
    payload["emit_frame"] = emit_frame
    payload["source_frame"] = source_frame
    payload["start_frame"] = start_frame
    payload["end_frame"] = end_frame
    payload["slot_id"] = slot_id
    payload["label"] = label
    payload["score"] = score
    payload["provenance_digest"] = provenance_digest
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


class _ExclusiveWriterLease:
    """Hold one OS-managed lease for a ledger path until explicitly released."""

    def __init__(self, ledger_path):
        self.path = Path(f"{ledger_path}.lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        self._closed = False
        try:
            if os.fstat(self._descriptor).st_size == 0:
                os.write(self._descriptor, b"0")
                os.fsync(self._descriptor)
            os.lseek(self._descriptor, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(self._descriptor)
            self._closed = True
            raise LedgerValidationError(
                f"unable to acquire exclusive writer lease for {ledger_path}"
            ) from exc

    def close(self):
        if self._closed:
            return
        try:
            os.lseek(self._descriptor, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self._descriptor)
            self._closed = True


class AtomicLedgerWriter:
    """Append canonical rows with one OS-level append write per event."""

    def __init__(self, path, *, fsync=True, create_new=False):
        self.path = Path(path)
        self.fsync = bool(fsync)
        self._lock = threading.Lock()
        self._closed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lease = _ExclusiveWriterLease(self.path)
        try:
            if create_new and self.path.exists():
                raise LedgerValidationError(f"ledger path already exists: {self.path}")
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
        except Exception:
            self._lease.close()
            self._closed = True
            raise

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
            if self._closed:
                raise LedgerValidationError("ledger writer is closed")
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
        self.close()
        return False

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._lease.close()
            self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive_canonical_json(path, payload):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    flags |= getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(output_path, flags, 0o600)
    except FileExistsError as exc:
        raise LedgerValidationError(f"commitment path already exists: {output_path}") from exc
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError(f"partial commitment write: {written} of {len(encoded)} bytes")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_canonical_json_file(path, description):
    input_path = Path(path)
    try:
        encoded = input_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise LedgerVerificationError(f"unable to read {description} {input_path}: {exc}") from exc
    if not encoded.endswith("\n") or encoded.count("\n") != 1:
        raise LedgerVerificationError(f"{description} must be one canonical JSON line")
    raw = encoded[:-1]
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise LedgerVerificationError(f"invalid {description}: {exc}") from exc
    if not isinstance(payload, dict) or canonical_json(payload) != raw:
        raise LedgerVerificationError(f"{description} must use canonical JSON encoding")
    return payload


def persist_verified_ledger(ledger_path, commitment_path, rows):
    """Publish verified formal rows and a separate immutable tail commitment."""

    ledger_path = Path(ledger_path)
    commitment_path = Path(commitment_path)
    if ledger_path.exists():
        raise LedgerValidationError(f"ledger path already exists: {ledger_path}")
    if commitment_path.exists():
        raise LedgerValidationError(f"commitment path already exists: {commitment_path}")

    source = verify_rows(rows)
    with AtomicLedgerWriter(ledger_path, create_new=True) as writer:
        for expected in source.rows:
            actual = writer.append(_payload_from_row(expected))
            if actual != expected:
                raise LedgerVerificationError("persisted ledger row differs from verified source row")

    persisted = verify_ledger(
        ledger_path,
        expected_count=source.count,
        expected_final_hash=source.final_hashes,
    )
    commitment = {
        "schema": LEDGER_COMMITMENT_SCHEMA,
        "version": LEDGER_COMMITMENT_VERSION,
        "ledger_filename": ledger_path.name,
        "ledger_sha256": _sha256_file(ledger_path),
        "count": persisted.count,
        "stream_counts": persisted.stream_counts,
        "final_hashes": persisted.final_hashes,
    }
    _write_exclusive_canonical_json(commitment_path, commitment)
    return _json_copy(commitment)


def load_verified_ledger(ledger_path, commitment_path):
    """Load a ledger only when its external immutable commitment verifies."""

    ledger_path = Path(ledger_path)
    commitment = _read_canonical_json_file(commitment_path, "ledger commitment")
    required = {
        "schema",
        "version",
        "ledger_filename",
        "ledger_sha256",
        "count",
        "stream_counts",
        "final_hashes",
    }
    if set(commitment) != required:
        raise LedgerVerificationError(
            f"ledger commitment fields differ: expected {sorted(required)}, found {sorted(commitment)}"
        )
    if (
        commitment["schema"] != LEDGER_COMMITMENT_SCHEMA
        or commitment["version"] != LEDGER_COMMITMENT_VERSION
    ):
        raise LedgerVerificationError("unsupported ledger commitment schema/version")
    if commitment["ledger_filename"] != ledger_path.name:
        raise LedgerVerificationError("ledger commitment filename does not match ledger path")
    expected_file_hash = _required_hash(
        commitment["ledger_sha256"], "commitment ledger_sha256", LedgerVerificationError
    )
    try:
        actual_file_hash = _sha256_file(ledger_path)
    except OSError as exc:
        raise LedgerVerificationError(f"unable to hash committed ledger {ledger_path}: {exc}") from exc
    if actual_file_hash != expected_file_hash:
        raise LedgerVerificationError("ledger file hash does not match commitment")
    return verify_ledger(
        ledger_path,
        expected_count=commitment["count"],
        expected_final_hash=commitment["final_hashes"],
    )


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
    "LEDGER_COMMITMENT_SCHEMA",
    "LEDGER_COMMITMENT_VERSION",
    "LEDGER_SCHEMA",
    "LEDGER_VERSION",
    "LedgerError",
    "LedgerReader",
    "LedgerValidationError",
    "LedgerVerificationError",
    "LedgerVerificationResult",
    "canonical_json",
    "compute_row_hash",
    "load_verified_ledger",
    "persist_verified_ledger",
    "read_ledger",
    "verify_ledger",
    "verify_rows",
]
