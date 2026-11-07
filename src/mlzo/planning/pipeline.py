"""Validated pipeline requests for an explicitly unsupported alpha capability."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mlzo.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    stages: int
    split: str | None = None
    microbatches: int = 1
    schedule: str = "1f1b"
    stage_factory: Callable[..., Any] | None = None

    def __post_init__(self) -> None:
        for name in ("stages", "microbatches"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigurationError(f"Pipeline {name} must be a positive integer.")
        if self.split is not None and (not isinstance(self.split, str) or not self.split.strip()):
            raise ConfigurationError("Pipeline split must be a nonempty module path or None.")
        if not isinstance(self.schedule, str) or self.schedule not in {"1f1b", "gpipe"}:
            raise ConfigurationError("Pipeline schedule must be '1f1b' or 'gpipe'.")
        if self.stage_factory is not None and not callable(self.stage_factory):
            raise ConfigurationError("Pipeline stage_factory must be callable or None.")


def pipeline(
    *, stages: int, split: str | None = None, microbatches: int = 1,
    schedule: str = "1f1b", stage_factory: Callable[..., Any] | None = None,
) -> PipelineConfig:
    """Describe a pipeline. Planning rejects execution in MLZO 0.1.0a1."""
    return PipelineConfig(stages, split, microbatches, schedule, stage_factory)
