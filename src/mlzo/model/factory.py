"""Importable references for transporting factories without pickling code."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from mlzo.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class FactoryRef:
    module: str
    qualname: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.module, str)
            or not self.module
            or self.module in {"__main__", "__mp_main__"}
            or not isinstance(self.qualname, str)
            or not self.qualname
            or any(not part.isidentifier() for part in self.qualname.split("."))
        ):
            raise ConfigurationError(
                "Remote factories must be named callables in an importable module. "
                "Move lambdas, closures, and __main__ definitions to a project module."
            )
        self.resolve()

    @classmethod
    def from_callable(cls, value: Callable[..., Any]) -> FactoryRef:
        if not callable(value):
            raise ConfigurationError("A remote factory must be callable.")
        module = getattr(value, "__module__", None)
        qualname = getattr(value, "__qualname__", None)
        reference = cls(module, qualname)
        resolved = reference.resolve()
        same_method = (
            inspect.ismethod(value)
            and inspect.ismethod(resolved)
            and resolved.__func__ is value.__func__
            and resolved.__self__ is value.__self__
        )
        if resolved is not value and not same_method:
            raise ConfigurationError(
                f"Remote factory {module}.{qualname} does not resolve to this callable. "
                "Use an importable function or class without captured instance state."
            )
        return reference

    def resolve(self) -> Callable[..., Any]:
        try:
            value: Any = importlib.import_module(self.module)
            for part in self.qualname.split("."):
                value = getattr(value, part)
        except Exception as exc:
            raise ConfigurationError(
                f"Cannot import remote factory {self.module}.{self.qualname}. "
                "Include its project module in the submitted code."
            ) from exc
        if not callable(value):
            raise ConfigurationError(
                f"Remote factory {self.module}.{self.qualname} is not callable."
            )
        return value

    def to_dict(self) -> dict[str, str]:
        return {"module": self.module, "qualname": self.qualname}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FactoryRef:
        if not isinstance(value, Mapping) or set(value) != {"module", "qualname"}:
            raise ConfigurationError("A factory reference needs module and qualname fields.")
        return cls(module=value["module"], qualname=value["qualname"])
