"""Learning rates indexed by successful optimizer updates, starting at one."""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch

from mlzo.errors import ConfigurationError, UnsupportedFeatureError


@dataclass(frozen=True)
class ScheduleSpec:
    kind: str = "constant"
    warmup: int = 0
    total_steps: int | None = None
    end_factor: float = 0.0
    gamma: float = 0.99
    factor: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in {"constant", "linear", "cosine", "exponential"}:
            raise ConfigurationError(f"Unknown schedule {self.kind!r}.")
        if isinstance(self.warmup, bool) or not isinstance(self.warmup, int) or self.warmup < 0:
            raise ConfigurationError("Schedule warmup must be a nonnegative integer.")
        if self.total_steps is not None and (isinstance(self.total_steps, bool) or not isinstance(self.total_steps, int) or self.total_steps < 1):
            raise ConfigurationError("Schedule total_steps must be a positive integer.")
        for name in ("end_factor", "gamma", "factor"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ConfigurationError(f"Schedule {name} must be finite and nonnegative.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ScheduleSpec:
        return cls(**value)


def constant(factor: float = 1.0, *, warmup: int = 0) -> ScheduleSpec:
    return ScheduleSpec("constant", warmup=warmup, factor=factor)


def linear(*, warmup: int = 0, total_steps: int | None = None, end_factor: float = 0.0) -> ScheduleSpec:
    return ScheduleSpec("linear", warmup=warmup, total_steps=total_steps, end_factor=end_factor)


def cosine(*, warmup: int = 0, total_steps: int | None = None, min_factor: float = 0.0) -> ScheduleSpec:
    return ScheduleSpec("cosine", warmup=warmup, total_steps=total_steps, end_factor=min_factor)


def exponential(gamma: float = 0.99, *, warmup: int = 0) -> ScheduleSpec:
    return ScheduleSpec("exponential", warmup=warmup, gamma=gamma)


def _multiplier(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ConfigurationError("A schedule must return a finite nonnegative multiplier.")
    return float(value)


class Schedule:
    """Internal controller. Native scheduler factories receive the bound optimizer.

    A callable ``(upcoming_index, horizon) -> multiplier`` uses update indices.
    Native scheduler classes/factories must have a first argument named
    ``optimizer`` or ``opt``. Native epoch/metric policies are unsupported.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, source: Any, horizon: int | None) -> None:
        self.optimizer = optimizer
        self.source = ScheduleSpec(source) if isinstance(source, str) else source
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.successful = 0
        self.native: Any = None
        self.horizon = horizon
        self.inferred_horizon = False
        self.call_arity = 2
        if isinstance(self.source, ScheduleSpec):
            self._resolve_builtin_horizon(self.source)
        elif source is not None:
            self._resolve_custom_source(source)
        if self.native is not None:
            self._validate_native()
        else:
            self._apply_next()

    def _resolve_builtin_horizon(self, source: ScheduleSpec) -> None:
        if source.total_steps is not None:
            self.horizon = source.total_steps
        elif source.kind in {"linear", "cosine"}:
            self.inferred_horizon = True
            if self.horizon is None:
                raise ConfigurationError("Linear/cosine schedules require a known automatic-update horizon or explicit total_steps.")
        if (source.kind in {"linear", "cosine"}
                and self.horizon is not None and source.warmup > self.horizon):
            raise ConfigurationError("Decay schedule warmup cannot exceed total_steps.")

    def _resolve_custom_source(self, source: Any) -> None:
        schedulers = torch.optim.lr_scheduler
        native_type = getattr(schedulers, "LRScheduler", schedulers._LRScheduler)
        if isinstance(source, (native_type, schedulers._LRScheduler, schedulers.ReduceLROnPlateau)):
            self.native = source
            return
        if not callable(source):
            raise ConfigurationError("schedule requires a schedule spec, multiplier callable, or native scheduler factory.")
        try:
            signature = inspect.signature(source)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("Schedule callables need an inspectable signature.") from exc
        first_parameter = next(iter(signature.parameters), None)
        if first_parameter in {"optimizer", "opt"}:
            self.native = source(self.optimizer)
        else:
            self._configure_multiplier(signature)

    def _configure_multiplier(self, signature: inspect.Signature) -> None:
        self.inferred_horizon = True
        if self.horizon is None:
            raise ConfigurationError("Callable multipliers require a known automatic-update horizon.")
        try:
            signature.bind(1, self.horizon)
        except TypeError:
            try:
                signature.bind(1)
                self.call_arity = 1
            except TypeError as exc:
                raise ConfigurationError("Schedule must accept (upcoming_update, horizon), or be a factory(optimizer).") from exc

    def _validate_native(self) -> None:
        if isinstance(self.native, torch.optim.lr_scheduler.ReduceLROnPlateau):
            raise UnsupportedFeatureError("Metric-driven schedulers require an explicit policy and are unsupported in this alpha.")
        if getattr(self.native, "optimizer", None) is not self.optimizer:
            raise ConfigurationError("A native scheduler must reference its bound optimizer.")
        if (not callable(getattr(self.native, "step", None))
                or not callable(getattr(self.native, "state_dict", None))):
            raise ConfigurationError("A scheduler factory must return a PyTorch scheduler.")

    def validate_horizon(self, horizon: int | None) -> None:
        if self.inferred_horizon and horizon != self.horizon:
            raise ConfigurationError(
                f"Schedule horizon changed from {self.horizon} to {horizon}. "
                "Extending this run requires a new run/schedule with explicit total_steps; the alpha does not silently reconfigure decay."
            )

    def _factor(self, index: int) -> float:
        source = self.source
        if source is None:
            return 1.0
        if not isinstance(source, ScheduleSpec):
            return _multiplier(source(index, self.horizon) if self.call_arity == 2 else source(index))
        if source.warmup and index <= source.warmup:
            return source.factor * index / source.warmup
        if source.kind == "constant":
            return source.factor
        if source.kind == "exponential":
            return source.factor * source.gamma ** max(0, index - max(source.warmup, 1))
        assert self.horizon is not None
        start = max(1, source.warmup)
        fraction = min(1.0, max(0.0, (index - start) / max(1, self.horizon - start)))
        if source.kind == "cosine":
            fraction = (1.0 - math.cos(math.pi * fraction)) / 2.0
        return source.factor * (1.0 + (source.end_factor - 1.0) * fraction)

    def _apply_next(self) -> None:
        factor = self._factor(self.successful + 1)
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base * factor

    def advance(self) -> None:
        self.successful += 1
        if self.native is not None:
            self.native.step()
        else:
            self._apply_next()

    def state_dict(self) -> dict[str, Any]:
        return {
            "base_lrs": list(self.base_lrs),
            "successful": self.successful,
            "horizon": self.horizon,
            "inferred_horizon": self.inferred_horizon,
            "definition": self._definition(),
            "current_lrs": [float(group["lr"]) for group in self.optimizer.param_groups],
            "native": self.native.state_dict() if self.native is not None else None,
        }

    def _definition(self) -> Any:
        if isinstance(self.source, ScheduleSpec):
            return self.source.to_dict()
        if self.source is None:
            return None
        source = self.source
        return {
            "module": getattr(source, "__module__", type(source).__module__),
            "qualname": getattr(source, "__qualname__", type(source).__qualname__),
            "native": type(self.native).__qualname__ if self.native is not None else None,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        fixed = isinstance(self.source, ScheduleSpec) and self.source.total_steps is not None
        if state["definition"] != self._definition():
            raise ConfigurationError("Checkpoint schedule definition differs from this run.")
        if (bool(state["inferred_horizon"]) != self.inferred_horizon
                or ((self.inferred_horizon or fixed) and state["horizon"] != self.horizon)):
            raise ConfigurationError("Checkpoint schedule horizon differs from this run; preserve the original schedule and budget.")
        if len(state["base_lrs"]) != len(self.optimizer.param_groups):
            raise ConfigurationError("Checkpoint scheduler parameter groups differ from this run.")
        self.base_lrs = [float(value) for value in state["base_lrs"]]
        self.successful = int(state["successful"])
        if self.native is not None:
            if state["native"] is None:
                raise ConfigurationError("Checkpoint uses a different scheduler kind.")
            self.native.load_state_dict(state["native"])
            for group, rate in zip(self.optimizer.param_groups, state["current_lrs"]):
                group["lr"] = float(rate)
        else:
            if state["native"] is not None:
                raise ConfigurationError("Checkpoint requires its native scheduler factory.")
            self._apply_next()
