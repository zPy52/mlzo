"""Restricted constructor recipes. Loading a recipe imports trusted user code."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from mlzo.errors import ConfigurationError
from mlzo.model import FactoryRef

MAX_DEPTH = 64


def _value(value: Any, *, encode: bool, depth: int = 0, active: set[int] | None = None) -> Any:
    if depth > MAX_DEPTH:
        raise ConfigurationError(f"Inference recipe nesting exceeds {MAX_DEPTH} levels.")
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if active is None:
        active = set()
    if id(value) in active:
        raise ConfigurationError("Cyclic inference recipe arguments are unsupported.")
    active.add(id(value))
    try:
        def visit(child: Any) -> Any:
            return _value(child, encode=encode, depth=depth + 1, active=active)

        if type(value) is list:
            return [visit(child) for child in value]
        if encode:
            if type(value) is tuple:
                return {"type": "tuple", "items": [visit(child) for child in value]}
            if type(value) is dict and all(type(key) is str for key in value):
                # Tag dictionaries too, so user keys cannot collide with tuple tags.
                return {"type": "dict", "items": [[key, visit(child)] for key, child in value.items()]}
        elif type(value) is dict and set(value) == {"type", "items"} and type(value["items"]) is list:
            if value["type"] == "tuple":
                return tuple(visit(child) for child in value["items"])
            if value["type"] == "dict":
                result: dict[str, Any] = {}
                for entry in value["items"]:
                    if type(entry) is not list or len(entry) != 2 or type(entry[0]) is not str or entry[0] in result:
                        raise ConfigurationError("Invalid dictionary entries in an inference recipe.")
                    result[entry[0]] = visit(entry[1])
                return result
    finally:
        active.remove(id(value))
    raise ConfigurationError(
        "Inference recipes support only None, bool, int, finite float, str, lists, "
        "tuples, and dictionaries with string keys. Convert devices, dtypes, tensors, "
        "nested callables, and custom objects to ordinary constructor arguments."
    )


def encode_recipe(recipe: Any) -> dict[str, Any]:
    if not isinstance(recipe, Mapping) or set(recipe) != {"factory", "args", "kwargs"}:
        raise ConfigurationError("An inference recipe must contain exactly factory, args, and kwargs.")
    if type(recipe["args"]) not in (list, tuple) or type(recipe["kwargs"]) is not dict:
        raise ConfigurationError("Recipe args must be a list/tuple and kwargs must be a dictionary.")
    try:
        factory = recipe["factory"]
        reference = factory if isinstance(factory, FactoryRef) else FactoryRef.from_callable(factory)
        reference.resolve()
    except ConfigurationError as exc:
        raise ConfigurationError(
            "Cannot export a non-importable inference recipe. Use a named factory in an "
            "importable module. Recipes are never silently discarded or pickled."
        ) from exc
    return {
        "factory": reference.to_dict(),
        "args": _value(tuple(recipe["args"]), encode=True),
        "kwargs": _value(recipe["kwargs"], encode=True),
    }


def decode_arguments(recipe: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Validate a serialized recipe without importing or invoking its factory."""
    if type(recipe) is not dict or set(recipe) != {"factory", "args", "kwargs"}:
        raise ConfigurationError("Invalid inference recipe fields in the manifest.")
    reference = recipe["factory"]
    if (
        type(reference) is not dict
        or set(reference) != {"module", "qualname"}
        or any(type(reference[key]) is not str or not reference[key] for key in reference)
        or reference["module"] in {"__main__", "__mp_main__"}
        or any(not part.isidentifier() for part in reference["module"].split("."))
        or any(not part.isidentifier() for part in reference["qualname"].split("."))
    ):
        raise ConfigurationError("Invalid importable factory reference in the inference manifest.")
    args = _value(recipe["args"], encode=False)
    kwargs = _value(recipe["kwargs"], encode=False)
    if type(args) is not tuple or type(kwargs) is not dict:
        raise ConfigurationError("An inference recipe must decode to tuple args and dictionary kwargs.")
    return args, kwargs
