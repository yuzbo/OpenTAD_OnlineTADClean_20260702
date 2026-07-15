"""Canonical, contained references for portable evidence bundles."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import secrets


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class EvidenceBundleError(ValueError):
    pass


def _root(path):
    try:
        root = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise EvidenceBundleError(f"evidence bundle root is invalid: {path}") from exc
    if not root.is_dir():
        raise EvidenceBundleError(f"evidence bundle root is not a directory: {root}")
    return root


def _lexical_absolute(path):
    return Path(os.path.abspath(os.fspath(path)))


def _contained_regular_file(path, bundle_root, label):
    root = _root(bundle_root)
    lexical = _lexical_absolute(path)
    try:
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise EvidenceBundleError(f"{label} escapes the evidence bundle") from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise EvidenceBundleError(f"{label} traverses a symbolic link")
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise EvidenceBundleError(f"{label} does not resolve to a file") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise EvidenceBundleError(f"{label} resolves outside the evidence bundle") from exc
    if os.path.normcase(str(resolved)) != os.path.normcase(str(lexical)):
        raise EvidenceBundleError(f"{label} traverses a filesystem link or junction")
    if not resolved.is_file():
        raise EvidenceBundleError(f"{label} is not a regular file")
    return root, resolved


def contained_file(path, bundle_root, label="evidence file"):
    return _contained_regular_file(path, bundle_root, label)[1]


def relative_bundle_path(path, bundle_root, label="evidence file"):
    root, resolved = _contained_regular_file(path, bundle_root, label)
    return resolved.relative_to(root).as_posix()


def resolve_bundle_path(value, bundle_root, label="evidence file"):
    if not isinstance(value, str) or not value:
        raise EvidenceBundleError(f"{label} path must be non-empty text")
    if "\\" in value:
        raise EvidenceBundleError(f"{label} path must use canonical POSIX separators")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise EvidenceBundleError(f"{label} path must be a contained relative path")
    root = _root(bundle_root)
    return _contained_regular_file(root.joinpath(*relative.parts), root, label)[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_file_reference(path, bundle_root, label="evidence file"):
    resolved = contained_file(path, bundle_root, label)
    return {
        "path": relative_bundle_path(resolved, bundle_root, label),
        "sha256": sha256_file(resolved),
    }


def verify_bundle_reference(reference, bundle_root, label="evidence file"):
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise EvidenceBundleError(f"{label} reference fields differ")
    expected = reference["sha256"]
    if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
        raise EvidenceBundleError(f"{label} reference digest is invalid")
    path = resolve_bundle_path(reference["path"], bundle_root, label)
    actual = sha256_file(path)
    if actual != expected:
        raise EvidenceBundleError(
            f"{label} hash mismatch: expected {expected}, found {actual}"
        )
    return path


def read_verified_bundle_bytes(reference, bundle_root, label="evidence file"):
    """Open once, read once, and verify bytes against the signed reference."""

    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise EvidenceBundleError(f"{label} reference fields differ")
    path = resolve_bundle_path(reference["path"], bundle_root, label)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise EvidenceBundleError(f"{label} changed while it was being read")
    payload = b"".join(chunks)
    expected = reference.get("sha256")
    if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
        raise EvidenceBundleError(f"{label} reference digest is invalid")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected or len(payload) != before.st_size:
        raise EvidenceBundleError(f"{label} bytes differ from the signed reference")
    return path, payload


def _write_staged_file(final_path, payload):
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    staged = final_path.parent / (
        f".{final_path.name}.{secrets.token_hex(12)}.staged"
    )
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    descriptor = os.open(staged, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("staged evidence write made no progress")
            offset += written
        os.fsync(descriptor)
    except Exception:
        os.close(descriptor)
        staged.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    return staged


def _fsync_directory(path):
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(Path(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_exclusive_file(path, payload):
    """Publish one fully staged file without permitting overwrite."""

    path = Path(path)
    if path.exists():
        raise EvidenceBundleError(f"refusing to overwrite published evidence: {path}")
    if not isinstance(payload, bytes):
        raise EvidenceBundleError("published evidence payload must be bytes")
    staged = None
    published = False
    try:
        staged = _write_staged_file(path, payload)
        os.link(staged, path)
        published = True
        _fsync_directory(path.parent)
    except OSError as exc:
        if published and staged is not None:
            try:
                if path.exists() and os.path.samefile(path, staged):
                    path.unlink()
            except OSError:
                pass
        raise EvidenceBundleError(f"failed to publish evidence: {exc}") from exc
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)
    return path


def publish_exclusive_pair(first_path, first_payload, commitment_path, commitment_payload):
    """Publish two complete files without overwrite, with commitment visible last."""

    first_path = Path(first_path)
    commitment_path = Path(commitment_path)
    if first_path == commitment_path:
        raise EvidenceBundleError("evidence and commitment paths must differ")
    for path in (first_path, commitment_path):
        if path.exists():
            raise EvidenceBundleError(f"refusing to overwrite published evidence: {path}")
    if not isinstance(first_payload, bytes) or not isinstance(commitment_payload, bytes):
        raise EvidenceBundleError("published evidence payloads must be bytes")

    staged_first = None
    staged_commitment = None
    published_first = False
    published_commitment = False
    try:
        staged_first = _write_staged_file(first_path, first_payload)
        staged_commitment = _write_staged_file(
            commitment_path, commitment_payload
        )
        os.link(staged_first, first_path)
        published_first = True
        _fsync_directory(first_path.parent)
        os.link(staged_commitment, commitment_path)
        published_commitment = True
        _fsync_directory(commitment_path.parent)
    except (OSError, EvidenceBundleError) as exc:
        for final, staged, published in (
            (commitment_path, staged_commitment, published_commitment),
            (first_path, staged_first, published_first),
        ):
            if published and staged is not None:
                try:
                    if final.exists() and os.path.samefile(final, staged):
                        final.unlink()
                except OSError:
                    pass
        raise EvidenceBundleError(f"failed to publish evidence pair: {exc}") from exc
    finally:
        for staged in (staged_first, staged_commitment):
            if staged is not None:
                staged.unlink(missing_ok=True)
    return first_path, commitment_path


__all__ = [
    "EvidenceBundleError",
    "bundle_file_reference",
    "contained_file",
    "publish_exclusive_file",
    "publish_exclusive_pair",
    "read_verified_bundle_bytes",
    "relative_bundle_path",
    "resolve_bundle_path",
    "verify_bundle_reference",
]
