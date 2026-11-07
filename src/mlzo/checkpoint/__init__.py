"""Local full-state checkpoints with restricted loading and atomic manifests.

Configuration and catalog APIs do not import torch or optional dependencies.
S3, per-block snapshots, distributed generations, and recovery fencing are
outside the 0.1.0a1 checkpoint implementation and fail explicitly.
"""

from .manifest import CheckpointRef
from .policy import CheckpointPolicy, checkpoint
from .snapshot import list_snapshots, load_snapshot, read_manifest, resolve_latest, save_snapshot

__all__ = [
    "CheckpointPolicy",
    "CheckpointRef",
    "checkpoint",
    "save_snapshot",
    "load_snapshot",
    "list_snapshots",
    "resolve_latest",
    "read_manifest",
]
