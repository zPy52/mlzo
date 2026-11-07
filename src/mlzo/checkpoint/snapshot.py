"""Complete local snapshots, latest resolution, and manifest-only catalog access."""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mlzo._version import __version__
from mlzo.errors import CheckpointError, ConfigurationError

from . import storage
from ._validation import json_bytes, local_path, metadata_dict, parse_json, retention
from .manifest import (
    FORMAT,
    FORMAT_VERSION,
    LATEST_FORMAT,
    LATEST_NAME,
    MANIFEST_NAME,
    STATE_NAME,
    CheckpointRef,
    validate_latest,
    validate_manifest,
)
from .torch_adapter import prepare_state, read_state, write_state

Reference = CheckpointRef | str | os.PathLike[str] | Mapping[str, Any]


def _read_manifest(directory: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    storage.require_directory(directory)
    path = directory / MANIFEST_NAME
    raw = storage.read_json_bytes(path)
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise CheckpointError(f"Latest manifest checksum mismatch at {path}; refusing to choose another snapshot.")
    manifest = validate_manifest(parse_json(raw, path=str(path)), expected_id=directory.name)
    if storage.size_of(directory / STATE_NAME) != manifest["state"]["size_bytes"]:
        raise CheckpointError(f"Checkpoint state size does not match the committed manifest at {directory}.")
    return manifest


def _latest(root: Path) -> tuple[Path, dict[str, Any]]:
    storage.require_directory(root)
    path = root / LATEST_NAME
    if not storage.present(path):
        raise CheckpointError(
            f"No latest.json pointer exists at {root}. Supply an explicit committed snapshot directory; "
            "MLZO does not guess a latest snapshot."
        )
    pointer = validate_latest(parse_json(storage.read_json_bytes(path), path=str(path)))
    directory = root / pointer["snapshot_id"]
    return directory, _read_manifest(directory, expected_sha256=pointer["manifest_sha256"])


def _resolve(reference: Reference) -> tuple[Path, dict[str, Any]]:
    if isinstance(reference, Mapping):
        reference = CheckpointRef.from_dict(reference)
    if isinstance(reference, CheckpointRef):
        directory = local_path(reference.path)
        manifest = _read_manifest(directory)
        if manifest["snapshot_id"] != reference.snapshot_id:
            raise CheckpointError("Checkpoint reference snapshot_id does not match the committed snapshot.")
        if reference.run_id is not None and manifest["run_id"] != reference.run_id:
            raise CheckpointError("Checkpoint reference run_id does not match the committed snapshot.")
        return directory, manifest
    path = local_path(reference)
    if path.name == MANIFEST_NAME:
        return path.parent, _read_manifest(path.parent)
    if path.name == LATEST_NAME:
        return _latest(path.parent)
    storage.require_directory(path)
    if storage.present(path / MANIFEST_NAME):
        return path, _read_manifest(path)
    return _latest(path)


def _reference(directory: Path, manifest: dict[str, Any]) -> CheckpointRef:
    return CheckpointRef(
        path=str(directory),
        snapshot_id=manifest["snapshot_id"],
        run_id=manifest["run_id"],
        created_at=manifest["created_at"],
        metadata=manifest["metadata"],
    )


def _catalog(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    if not storage.present(root):
        return []
    storage.require_directory(root)
    if storage.present(root / MANIFEST_NAME):
        raise CheckpointError("list_snapshots expects a destination root, not an individual snapshot directory.")
    entries = []
    for directory in root.iterdir():
        if directory.name.startswith("."):
            continue
        if directory.is_symlink():
            raise CheckpointError(f"Checkpoint catalog entries must not be symlinks: {directory}.")
        if not directory.is_dir() or not storage.present(directory / MANIFEST_NAME):
            # An interrupted writer may have left an uncommitted directory.
            continue
        entries.append((directory, _read_manifest(directory)))
    entries.sort(key=lambda item: (item[1]["created_at"], item[1]["snapshot_id"]), reverse=True)
    return entries


def save_snapshot(
    state: dict[Any, Any],
    *,
    to: str | os.PathLike[str],
    run_id: str,
    metadata: dict[str, Any] | None = None,
    keep_last: int | None = None,
) -> CheckpointRef:
    """Save one complete full snapshot at a caller-selected safe boundary.

    State consists of tensors, None/bool/int/float/str, dict/OrderedDict, list,
    and tuple values. Integer dictionary keys and Python RNG tuples survive
    unchanged. Tensors are detached and copied to CPU; loading returns CPU
    state. OrderedDict's PyTorch ``_metadata`` is preserved. Custom objects,
    NumPy arrays, tensor attributes, and cyclic containers must be converted
    explicitly. Manifest metadata must be JSON data with string keys.

    This alpha supports local single-worker generation-zero snapshots only.
    Use one writer and a separate destination per run. Committed directories
    are never rewritten. Publication fsyncs state first, publishes the manifest
    atomically last, then atomically replaces latest.json. Retention runs only
    after successful latest publication and always keeps this new snapshot.
    A failed publication may leave an unadvertised committed directory, and a
    process crash may leave a lock requiring manual cleanup. There is no reader
    pinning against concurrent retention and no distributed writer fencing.
    """
    root = local_path(to)
    if type(run_id) is not str or not run_id:
        raise ConfigurationError("Checkpoint run_id must be a nonempty string.")
    retention(keep_last)
    if storage.present(root / MANIFEST_NAME):
        raise CheckpointError("A committed snapshot is immutable. Save to its destination root instead.")
    clean_metadata = metadata_dict(metadata)
    frozen, torch_version = prepare_state(state)
    directory: Path | None = None
    committed = False
    published = False
    try:
        with storage.writer(root):
            if storage.present(root / MANIFEST_NAME):
                raise CheckpointError("A committed snapshot is immutable. Save to its destination root instead.")
            if storage.present(root / LATEST_NAME):
                # Do not conceal a damaged pointer or advertise recovery by overwriting it.
                _latest(root)
            now = datetime.now(timezone.utc)
            identity = f"{now:%Y%m%dT%H%M%S%fZ}-{uuid.uuid4().hex}"
            directory = root / identity
            directory.mkdir(mode=0o700)
            descriptor = os.open(directory / STATE_NAME, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                write_state(frozen, stream)
                stream.flush()
                os.fsync(stream.fileno())
            digest, size = storage.checksum(directory / STATE_NAME)
            manifest = {
                "format": FORMAT,
                "format_version": FORMAT_VERSION,
                "snapshot_id": identity,
                "run_id": run_id,
                "created_at": now.isoformat(timespec="microseconds").replace("+00:00", "Z"),
                "mode": "full",
                "generation": 0,
                "world_size": 1,
                "versions": {"mlzo": __version__, "python": platform.python_version(), "torch": torch_version},
                "state": {"file": STATE_NAME, "format": "torch.weights_only", "sha256": digest, "size_bytes": size},
                "metadata": clean_metadata,
            }
            raw_manifest = json_bytes(manifest, path="manifest")
            storage.publish_json(directory / MANIFEST_NAME, raw_manifest)
            committed = True
            storage.fsync_directory(root)
            pointer = {
                "format": LATEST_FORMAT,
                "format_version": FORMAT_VERSION,
                "snapshot_id": identity,
                "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
            }
            storage.publish_json(root / LATEST_NAME, json_bytes(pointer, path="latest"), replace=True)
            published = True
            if keep_last is not None:
                # Explicitly protect the just-published ID even if the wall clock moved backward.
                previous = [item for item in _catalog(root) if item[1]["snapshot_id"] != identity]
                for obsolete, _ in previous[keep_last - 1:]:
                    shutil.rmtree(obsolete)
                storage.fsync_directory(root)
            return _reference(directory, manifest)
    except (CheckpointError, OSError) as exc:
        if directory is not None and not committed and not storage.present(directory / MANIFEST_NAME):
            try:
                shutil.rmtree(directory)
            except OSError:
                pass  # Uncommitted directories remain invisible to the catalog.
        if published:
            raise CheckpointError(
                f"Snapshot {directory} was committed and latest.json was published, but retention or lock cleanup failed. "
                "The new snapshot can be loaded explicitly; inspect the destination before saving again."
            ) from exc
        if committed or (directory is not None and storage.present(directory / MANIFEST_NAME)):
            raise CheckpointError(
                f"Snapshot {directory} was committed, but latest publication did not complete durably. "
                "Inspect latest.json or load that snapshot explicitly before retrying."
            ) from exc
        if isinstance(exc, CheckpointError):
            raise
        raise CheckpointError(f"Cannot save checkpoint at {root}. Check filesystem access, free space, and atomic-link support.") from exc


def load_snapshot(reference: Reference) -> dict[Any, Any]:
    """Verify format, exact size, and SHA256 before restricted CPU loading.

    Accept a CheckpointRef, its dictionary form, a committed snapshot directory,
    manifest.json, a destination root, or its latest.json pointer. A bad latest
    pointer or state file raises; loading never selects an older snapshot.
    """
    try:
        directory, manifest = _resolve(reference)
        state = manifest["state"]
        with storage.verified_state(directory / STATE_NAME, expected_size=state["size_bytes"], expected_sha256=state["sha256"]) as stream:
            return read_state(stream)
    except OSError as exc:
        raise CheckpointError("Cannot read checkpoint files. Supply an existing committed snapshot or valid destination root.") from exc


def resolve_latest(to: str | os.PathLike[str]) -> CheckpointRef:
    """Resolve latest.json strictly, without importing torch or guessing a snapshot."""
    root = local_path(to)
    if root.name == LATEST_NAME:
        root = root.parent
    try:
        directory, manifest = _latest(root)
        return _reference(directory, manifest)
    except OSError as exc:
        raise CheckpointError(f"Cannot resolve the latest checkpoint at {root}; no fallback was attempted.") from exc


def read_manifest(reference: Reference) -> dict[str, Any]:
    """Read verified primitive-only manifest data for catalog/cat operations.

    Validates the schema, latest manifest digest when used, and state file size.
    Full state SHA256 verification happens during load_snapshot, not listing.
    """
    try:
        _, manifest = _resolve(reference)
        return manifest
    except OSError as exc:
        raise CheckpointError("Cannot read the committed checkpoint manifest and its state file.") from exc


def list_snapshots(to: str | os.PathLike[str]) -> list[CheckpointRef]:
    """List committed snapshots newest first, without importing torch.

    Validates manifests and state sizes, skips incomplete directories, and
    raises on corrupt committed entries. A missing destination returns [].
    Listing does not use or repair latest.json and does not hash all state files.
    """
    root = local_path(to)
    try:
        return [_reference(directory, manifest) for directory, manifest in _catalog(root)]
    except OSError as exc:
        raise CheckpointError(f"Cannot list checkpoints at {root}.") from exc
