"""Algorithm contracts and bounded, detached run reporting."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Generic, TypeVar

import torch
from torch import nn

from mlzo.errors import ConfigurationError, TrainingError

BatchT = TypeVar("BatchT")


@dataclass
class Step:
    """A scalar mean and its optional normalization weight.

    A plain scalar loss has weight one. Use ``weight`` for means over varying
    numbers of examples or tokens. ``stop`` is a whole-run boolean.
    """

    loss: torch.Tensor | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    stop: bool | None = None
    weight: float | torch.Tensor | None = None


class Training(ABC, Generic[BatchT]):
    """A learning algorithm whose model is constructed before optimizer binding.

    Override ``iteration_source()`` to supply batches when train(data=None), or
    set ``step_driven = True`` to receive None on each collection iteration.
    Only ``state`` is checkpointed; keep it to tensors and primitive containers.
    """

    step_driven = False

    def __init__(self, model: nn.Module) -> None:
        if not isinstance(model, nn.Module):
            raise ConfigurationError("Training(model) requires a torch.nn.Module, including Lazy.")
        self.model = model
        self.state: Any = None
        self._engine: Any = None

    @abstractmethod
    def step(self, batch: BatchT) -> torch.Tensor | Step | None:
        """Run one algorithm iteration, optionally requesting an automatic update."""
        raise NotImplementedError

    def iteration_source(self) -> Any:
        """Return an iterable, or override with an epoch-aware data pipeline."""
        raise ConfigurationError("Supply data, override iteration_source(), or set step_driven=True.")

    def evaluate(self, batch: BatchT) -> torch.Tensor | Step | None:
        """Custom algorithms must implement a separate, non-mutating evaluator."""
        raise ConfigurationError("Custom training requires an evaluate() override or run.evaluate(..., step=...).")

    def update(self, name: str, loss: torch.Tensor, *, retain_graph: bool = False) -> None:
        """Complete a named optimizer update synchronously inside step()."""
        if self._engine is None:
            raise TrainingError("Training.update() is only available during an active training step.")
        self._engine.explicit_update(name, loss, retain_graph=retain_graph)


def detached_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(metrics, Mapping):
        raise ConfigurationError("Step.metrics must be a mapping of names to scalar values.")
    result: dict[str, float] = {}
    for name, value in metrics.items():
        if not isinstance(name, str):
            raise ConfigurationError("Metric names must be strings.")
        if isinstance(value, torch.Tensor):
            if value.ndim != 0 or value.is_complex():
                raise ConfigurationError(f"Metric {name!r} must be a real scalar tensor.")
            result[name] = float(value.detach().item())
        elif isinstance(value, Real):
            result[name] = float(value)
        else:
            raise ConfigurationError(f"Metric {name!r} must be a number or scalar tensor.")
    return result


@dataclass
class RunStatus:
    state: str
    id: str
    counters: dict[str, Any] = field(default_factory=dict)
    error: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {"state": self.state, "id": self.id, "counters": dict(self.counters), "error": self.error}


class History:
    """The most recent scalar records, bounded independently of run duration."""

    def __init__(self, run_id: str | None = None, *, max_records: int = 1000) -> None:
        if isinstance(max_records, bool) or not isinstance(max_records, int) or max_records < 1:
            raise ConfigurationError("max_records must be a positive integer.")
        self.run_id = run_id
        self.max_records = max_records
        self.records: deque[dict[str, Any]] = deque(maxlen=max_records)
        self.dropped = 0
        self.counters: dict[str, Any] = {}
        self.state = "created"
        self.notes: list[str] = []

    def append(self, metrics: Mapping[str, Any], *, counters: Mapping[str, Any] | None = None) -> None:
        record: dict[str, Any] = {"metrics": detached_metrics(metrics)}
        if counters is not None:
            record["iteration"] = counters.get("iteration", 0)
            record["epoch"] = counters.get("epoch", 0)
            record["update_events"] = counters.get("update_events", 0)
        if len(self.records) == self.max_records:
            self.dropped += 1
        self.records.append(record)

    @property
    def metrics(self) -> dict[str, list[float]]:
        values: dict[str, list[float]] = {}
        for record in self.records:
            for key, value in record["metrics"].items():
                values.setdefault(key, []).append(value)
        return values

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "state": self.state, "records": list(self.records),
                "counters": dict(self.counters), "dropped": self.dropped, "max_records": self.max_records,
                "notes": list(self.notes)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> History:
        history = cls(value.get("run_id"), max_records=value.get("max_records", 1000))
        for record in value.get("records", []):
            history.append(record.get("metrics", {}), counters=record)
        history.counters = dict(value.get("counters", {}))
        history.state = value.get("state", "completed")
        history.notes = list(value.get("notes", []))
        history.dropped += value.get("dropped", 0)
        return history

    def __len__(self) -> int:
        return len(self.records)

