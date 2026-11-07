"""Local atomic publication and checked reads for checkpoint snapshots.

Only one writer may use a destination at a time. A fail-fast exclusive lock
guards cooperating local writers; it is not a distributed generation/fencing
protocol. After a crashed writer, remove its .mlzo-write.lock only after making
sure the writer has stopped. Incomplete snapshot directories are never listed.
The filesystem must support atomic same-directory replace and hard links.
"""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

from mlzo.errors import CheckpointError

from ._validation import MAX_JSON_BYTES

LOCK_NAME = ".mlzo-write.lock"
CHUNK_SIZE = 1024 * 1024


def present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def require_directory(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise CheckpointError(f"Expected a regular checkpoint directory at {path}; symlink directories are unsupported.")


def fsync_directory(path: Path) -> None:
    if os.name == "posix":
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


@contextmanager
def writer(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    require_directory(root)
    lock = root / LOCK_NAME
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise CheckpointError(
            f"Checkpoint destination is locked at {lock}. Use one writer per destination. "
            "For a stale lock, first ensure the previous writer has stopped, then remove that lock file."
        ) from None
    identity = os.fstat(descriptor)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write(f"MLZO single-writer checkpoint lock; pid={os.getpid()}\n")
            stream.flush()
            os.fsync(stream.fileno())
        fsync_directory(root)
        yield
    finally:
        try:
            current = lock.lstat()
        except FileNotFoundError:
            raise CheckpointError("Checkpoint writer lock disappeared. Do not remove an active writer's lock.") from None
        if (current.st_dev, current.st_ino) != (identity.st_dev, identity.st_ino):
            raise CheckpointError("Checkpoint writer lock was replaced. Refusing to remove another writer's lock.")
        lock.unlink()
        fsync_directory(root)


@contextmanager
def open_regular(path: Path) -> Iterator[BinaryIO]:
    if path.is_symlink():
        raise CheckpointError(f"Checkpoint files must not be symlinks: {path}.")
    # A corrupt entry must not block an inspection call by replacing a file with a FIFO.
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise CheckpointError(f"Expected a regular checkpoint file at {path}.")
        stream = os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise
    with stream:
        yield stream


def read_json_bytes(path: Path) -> bytes:
    with open_regular(path) as stream:
        if os.fstat(stream.fileno()).st_size > MAX_JSON_BYTES:
            raise CheckpointError(f"Checkpoint JSON exceeds the size limit at {path}.")
        data = stream.read(MAX_JSON_BYTES + 1)
    if len(data) > MAX_JSON_BYTES:
        raise CheckpointError(f"Checkpoint JSON exceeds the size limit at {path}.")
    return data


def size_of(path: Path) -> int:
    with open_regular(path) as stream:
        return os.fstat(stream.fileno()).st_size


def checksum(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with open_regular(path) as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def publish_json(path: Path, data: bytes, *, replace: bool = False) -> None:
    """Publish a fully fsynced file atomically; committed manifests never overwrite."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=".pending-", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            # Linking is atomic and fails if a committed manifest already exists.
            os.link(temporary, path)
        temporary.unlink(missing_ok=True)
        fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def verified_state(path: Path, *, expected_size: int, expected_sha256: str) -> Iterator[BinaryIO]:
    """Verify into a private seekable copy, then load those same verified bytes."""
    with tempfile.SpooledTemporaryFile(max_size=8 * CHUNK_SIZE, mode="w+b") as verified:
        digest = hashlib.sha256()
        size = 0
        with open_regular(path) as source:
            if os.fstat(source.fileno()).st_size != expected_size:
                raise CheckpointError(f"Checkpoint state size does not match the manifest at {path}.")
            while chunk := source.read(min(CHUNK_SIZE, expected_size - size + 1)):
                size += len(chunk)
                if size > expected_size:
                    raise CheckpointError(f"Checkpoint state grew while reading {path}.")
                digest.update(chunk)
                verified.write(chunk)
        if size != expected_size:
            raise CheckpointError(f"Checkpoint state is truncated at {path}.")
        if digest.hexdigest() != expected_sha256:
            raise CheckpointError(f"Checkpoint state SHA256 mismatch at {path}; refusing to load corrupted state.")
        verified.seek(0)
        yield verified
