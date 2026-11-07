"""Version-one manifests and portable checkpoint references."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from mlzo.errors import CheckpointError, UnsupportedFeatureError

from ._validation import SHA256, destination, metadata_dict, snapshot_id

FORMAT = "mlzo.checkpoint"
FORMAT_VERSION = 1
LATEST_FORMAT = "mlzo.checkpoint.latest"
MANIFEST_NAME = "manifest.json"
STATE_NAME = "state.pt"
LATEST_NAME = "latest.json"


@dataclass(frozen=True, slots=True)
class CheckpointRef:
    """A committed snapshot directory and its primitive-only catalog data.

    ``path`` names a snapshot directory, not its destination root. Metadata in
    the reference is informational; loading reads and validates the manifest.
    """

    path: str | os.PathLike[str]
    snapshot_id: str
    run_id: str | None = None
    created_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", destination(self.path))
        snapshot_id(self.snapshot_id)
        if self.run_id is not None and (type(self.run_id) is not str or not self.run_id):
            raise CheckpointError("Checkpoint reference run_id must be a nonempty string or None.")
        if self.created_at is not None:
            _timestamp(self.created_at)
        object.__setattr__(self, "metadata", metadata_dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "snapshot_id": self.snapshot_id,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "metadata": metadata_dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CheckpointRef:
        fields = {"path", "snapshot_id", "run_id", "created_at", "metadata"}
        if not isinstance(value, Mapping) or not {"path", "snapshot_id"} <= set(value) or set(value) - fields:
            raise CheckpointError("A checkpoint reference needs path and snapshot_id fields.")
        return cls(**dict(value))


def _timestamp(value: Any) -> None:
    if type(value) is not str or not value.endswith("Z"):
        raise CheckpointError("manifest.created_at must be a UTC ISO-8601 timestamp ending in Z.")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise CheckpointError("manifest.created_at is not a valid timestamp.") from None
    if parsed.tzinfo != timezone.utc:
        raise CheckpointError("manifest.created_at must use UTC.")


def validate_manifest(value: dict[str, Any], *, expected_id: str) -> dict[str, Any]:
    """Validate the entire format before any state bytes reach torch.load."""
    if value.get("format") != FORMAT or type(value.get("format_version")) is not int:
        raise CheckpointError("Not a versioned MLZO checkpoint manifest.")
    if value["format_version"] != FORMAT_VERSION:
        raise UnsupportedFeatureError("Unsupported checkpoint format version. Use an MLZO version that supports this snapshot.")
    fields = {"format", "format_version", "snapshot_id", "run_id", "created_at", "mode", "generation", "world_size", "versions", "state", "metadata"}
    if set(value) != fields:
        raise CheckpointError("Checkpoint manifest has missing or unknown fields.")
    if snapshot_id(value["snapshot_id"]) != expected_id:
        raise CheckpointError("Checkpoint manifest snapshot_id does not match its directory or reference.")
    if type(value["run_id"]) is not str or not value["run_id"]:
        raise CheckpointError("manifest.run_id must be a nonempty string.")
    _timestamp(value["created_at"])
    if type(value["generation"]) is not int or type(value["world_size"]) is not int or type(value["mode"]) is not str:
        raise CheckpointError("Checkpoint mode, generation, or world_size has an invalid type.")
    if value["mode"] != "full" or value["generation"] != 0 or value["world_size"] != 1:
        raise UnsupportedFeatureError(
            "This alpha loads only full, single-worker snapshots with generation=0 and world_size=1. "
            "Per-block snapshots, distributed state, and generation recovery need a supported runtime."
        )
    versions = value["versions"]
    if type(versions) is not dict or set(versions) != {"mlzo", "python", "torch"} or any(type(item) is not str or not item for item in versions.values()):
        raise CheckpointError("manifest.versions must contain MLZO, Python, and PyTorch version strings.")
    state = value["state"]
    if type(state) is not dict or set(state) != {"file", "format", "sha256", "size_bytes"}:
        raise CheckpointError("manifest.state must describe exactly one complete state file.")
    if state["file"] != STATE_NAME or state["format"] != "torch.weights_only":
        raise CheckpointError("manifest.state must name state.pt in the restricted torch.weights_only format.")
    if type(state["sha256"]) is not str or SHA256.fullmatch(state["sha256"]) is None:
        raise CheckpointError("manifest.state.sha256 is invalid.")
    if type(state["size_bytes"]) is not int or state["size_bytes"] <= 0:
        raise CheckpointError("manifest.state.size_bytes must be a positive integer.")
    if type(value["metadata"]) is not dict:
        raise CheckpointError("manifest.metadata must be a JSON object.")
    value["metadata"] = metadata_dict(value["metadata"])
    return value


def validate_latest(value: dict[str, Any]) -> dict[str, Any]:
    if set(value) != {"format", "format_version", "snapshot_id", "manifest_sha256"} or value.get("format") != LATEST_FORMAT:
        raise CheckpointError("Invalid latest.json checkpoint pointer; refusing to guess another snapshot.")
    if type(value["format_version"]) is not int or value["format_version"] != FORMAT_VERSION:
        raise CheckpointError("Unsupported latest.json format version; refusing to guess another snapshot.")
    snapshot_id(value["snapshot_id"])
    if type(value["manifest_sha256"]) is not str or SHA256.fullmatch(value["manifest_sha256"]) is None:
        raise CheckpointError("Invalid latest.json manifest checksum; refusing to guess another snapshot.")
    return value
