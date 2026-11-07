"""Checkpoint configuration independent of training and PyTorch imports."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mlzo.errors import ConfigurationError, UnsupportedFeatureError

from ._validation import destination, retention


@dataclass(frozen=True, slots=True)
class CheckpointPolicy:
    """Configure full, single-worker snapshots at an execution-engine interval.

    ``keep_last=None`` disables retention. Retention applies to the whole local
    destination, so use a separate destination for each run. The execution
    engine counts successful optimizer update events for ``every`` and decides
    which boundaries are safe for a complete snapshot; this storage module
    does not schedule training work.
    """

    to: str | os.PathLike[str]
    every: int = 1000
    keep_last: int | None = 3
    mode: str = "full"

    def __post_init__(self) -> None:
        object.__setattr__(self, "to", destination(self.to))
        if type(self.every) is not int or self.every <= 0:
            raise ConfigurationError("Checkpoint every must be a positive integer.")
        retention(self.keep_last)
        if type(self.mode) is not str:
            raise ConfigurationError("Checkpoint mode must be the string 'full' in MLZO 0.1.0a1.")
        if self.mode in ("per_block", "sharded", "distributed"):
            raise UnsupportedFeatureError(
                "Per-block and distributed checkpoints are not implemented in MLZO 0.1.0a1. "
                "Use mode='full' with a single execution worker."
            )
        if self.mode != "full":
            raise ConfigurationError("Checkpoint mode must be 'full' in MLZO 0.1.0a1.")

    def to_dict(self) -> dict[str, Any]:
        return {"to": self.to, "every": self.every, "keep_last": self.keep_last, "mode": self.mode}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CheckpointPolicy:
        if not isinstance(value, Mapping) or "to" not in value or set(value) - {"to", "every", "keep_last", "mode"}:
            raise ConfigurationError("A checkpoint policy needs to, with optional every, keep_last, and mode fields.")
        return cls(**dict(value))


def checkpoint(
    to: str | os.PathLike[str],
    every: int = 1000,
    keep_last: int | None = 3,
    mode: str = "full",
) -> CheckpointPolicy:
    """Create a validated checkpoint policy without allocating model state."""
    return CheckpointPolicy(to=to, every=every, keep_last=keep_last, mode=mode)
