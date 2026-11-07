"""Deferred optimizer specifications with stable parameter ownership."""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from mlzo.errors import ConfigurationError, UnsupportedFeatureError

_BUILTINS = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam,
             "sgd": torch.optim.SGD, "rmsprop": torch.optim.RMSprop}


@dataclass(frozen=True)
class OptimizerSpec:
    name: Any = "adamw"
    params: Any = None
    kwargs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "params": self.params, "kwargs": dict(self.kwargs)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OptimizerSpec:
        return cls(value.get("name", "adamw"), value.get("params"), dict(value.get("kwargs", {})))


OptSpec = OptimizerSpec


def opt(name: Any = "adamw", *, params: Any = None, **kwargs: Any) -> OptimizerSpec:
    """Bind an optimizer later using a submodule path or parameter-name selector.

    ``params='encoder'`` selects that submodule, ``params=['head.weight', ...]``
    selects exact parameter names. A list of ``{'params': selector, ...}`` gives
    native parameter groups. Selectors remain valid after Lazy construction.
    """
    return OptimizerSpec(name, params, kwargs)


def _rate(value: Any, label: str = "lr") -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ConfigurationError(f"{label} must be a finite nonnegative number.")
    return float(value)


def _select(model: nn.Module, selector: Any) -> list[nn.Parameter]:
    named = dict(model.named_parameters())
    if selector is None or (isinstance(selector, str) and selector == "*"):
        return [p for p in named.values() if p.requires_grad]
    if isinstance(selector, str):
        if selector in named:
            return [named[selector]]
        try:
            return [p for p in model.get_submodule(selector).parameters() if p.requires_grad]
        except (AttributeError, KeyError) as exc:
            raise ConfigurationError(f"Unknown parameter or submodule selector {selector!r}.") from exc
    if isinstance(selector, (tuple, list)) and all(isinstance(item, str) for item in selector):
        parameters: list[nn.Parameter] = []
        for item in selector:
            parameters.extend(_select(model, item))
        return parameters
    if callable(selector):
        # A local stable selector sees names and parameters after materialization.
        return [p for name, p in named.items() if selector(name, p)]
    raise ConfigurationError("params must be a submodule path, parameter-name list, or (name, parameter) predicate.")


def _groups(model: nn.Module, selector: Any) -> Any:
    if (
        isinstance(selector, (tuple, list))
        and selector
        and all(isinstance(item, Mapping) for item in selector)
    ):
        groups = []
        for item in selector:
            if "params" not in item:
                raise ConfigurationError("Every optimizer parameter group requires a params selector.")
            groups.append({**item, "params": _select(model, item["params"])})
        return groups
    return _select(model, selector)


def bind_optimizers(model: nn.Module, specification: Any, lr: float | None) -> dict[str, torch.optim.Optimizer]:
    if lr is not None:
        lr = _rate(lr)
    specifications = dict(specification) if isinstance(specification, Mapping) else {"default": specification}
    if not specifications:
        raise ConfigurationError("At least one optimizer is required.")
    model_ids = {id(parameter) for parameter in model.parameters()}
    owners: dict[int, str] = {}
    result: dict[str, torch.optim.Optimizer] = {}
    for label, source in specifications.items():
        if not isinstance(label, str) or not label:
            raise ConfigurationError("Named optimizers need nonempty string names.")
        if isinstance(source, torch.optim.Optimizer):
            optimizer = source
            if lr is not None and any(_rate(group["lr"]) != lr for group in optimizer.param_groups):
                raise ConfigurationError(f"Explicit lr={lr} conflicts with optimizer {label!r}; omit lr to preserve its groups.")
        else:
            spec = source if isinstance(source, OptimizerSpec) else opt(source)
            kwargs = dict(spec.kwargs)
            if "params" in kwargs:
                raise ConfigurationError("Pass parameter selectors with opt(params=...).")
            if lr is not None:
                if "lr" in kwargs and _rate(kwargs["lr"]) != lr:
                    raise ConfigurationError(f"Explicit lr conflicts with opt() for {label!r}.")
                kwargs["lr"] = lr
            factory = spec.name
            if isinstance(factory, str):
                if factory.lower() not in _BUILTINS:
                    raise UnsupportedFeatureError(f"Unknown optimizer {factory!r}; supported names are {', '.join(_BUILTINS)}. LBFGS is unsupported.")
                factory = _BUILTINS[factory.lower()]
                kwargs.setdefault("lr", 3e-4)
                if factory is torch.optim.AdamW:
                    kwargs.setdefault("weight_decay", 0.01)
            if not callable(factory):
                raise ConfigurationError(f"Optimizer {label!r} requires a name, optimizer class/factory, instance, or opt().")
            if inspect.isclass(factory) and issubclass(factory, torch.optim.LBFGS):
                raise UnsupportedFeatureError("LBFGS closure-based updates are unsupported.")
            groups = _groups(model, spec.params)
            if not groups:
                raise ConfigurationError(f"Optimizer {label!r} selects no parameters.")
            uses_parameter_groups = isinstance(groups[0], Mapping)
            # Detect ownership before a constructor can hide duplicates.
            if uses_parameter_groups:
                selected_parameters = [p for group in groups for p in group["params"]]
            else:
                selected_parameters = groups
            if len({id(p) for p in selected_parameters}) != len(selected_parameters):
                raise ConfigurationError(f"Optimizer {label!r} has overlapping parameter groups/selectors.")
            if uses_parameter_groups and any(not group["params"] for group in groups):
                raise ConfigurationError(f"Optimizer {label!r} has an empty parameter group.")
            try:
                optimizer = factory(groups, **kwargs)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(f"Cannot construct optimizer {label!r}: {exc}") from exc
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise ConfigurationError(f"Optimizer factory {label!r} must return a torch.optim.Optimizer.")
        if isinstance(optimizer, torch.optim.LBFGS):
            raise UnsupportedFeatureError("LBFGS closure-based updates are unsupported.")
        if not optimizer.param_groups:
            raise ConfigurationError(f"Optimizer {label!r} has no parameter groups.")
        for group in optimizer.param_groups:
            if not group["params"]:
                raise ConfigurationError(f"Optimizer {label!r} has an empty parameter group.")
            _rate(group["lr"], f"{label}.lr")
            for parameter in group["params"]:
                if id(parameter) not in model_ids:
                    raise ConfigurationError(f"Optimizer {label!r} references a foreign parameter; bind it after model construction.")
                if id(parameter) in owners:
                    raise ConfigurationError(f"Optimizer {label!r} overlaps parameter ownership with {owners[id(parameter)]!r}.")
                owners[id(parameter)] = label
        result[label] = optimizer
    return result
