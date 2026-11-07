"""Copy only restricted state_dict values, never Python model objects."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch

from mlzo.errors import ConfigurationError


def state_dict_copy(state: Any, *, freeze: bool) -> dict[str, Any]:
    if type(state) not in (dict, OrderedDict) or any(type(key) is not str for key in state):
        raise ConfigurationError("Inference weights must be a state_dict with string keys.")
    active: set[int] = set()
    copied: dict[int, Any] = {}

    def copy_value(value: Any, depth: int = 0) -> Any:
        if depth > 64:
            raise ConfigurationError("Inference state_dict nesting exceeds 64 levels.")
        if value is None or type(value) in (bool, int, float, str):
            return value
        object_id = id(value)
        if object_id in active:
            raise ConfigurationError("Cyclic state_dict values are unsupported for inference export.")
        if object_id in copied:
            return copied[object_id]
        if torch.nn.parameter.is_lazy(value):
            raise ConfigurationError("Materialize uninitialized PyTorch parameters/buffers before exporting inference weights.")
        if type(value) in (torch.Tensor, torch.nn.Parameter):
            if value.is_meta:
                raise ConfigurationError("Meta tensors have no weights. Materialize them before inference export.")
            if value.__dict__:
                raise ConfigurationError("Tensor attributes are unsupported custom inference state.")
            try:
                result = value.detach().cpu().clone() if freeze else value.detach()
            except Exception as exc:
                raise ConfigurationError("Cannot copy inference weights to portable CPU tensors.") from exc
            copied[object_id] = result
            return result
        if type(value) not in (dict, OrderedDict, list, tuple):
            raise ConfigurationError(
                "Inference state_dict values must be tensors or restricted primitive/container values; "
                "custom objects and tensor subclasses cannot be pickled into inference artifacts."
            )
        active.add(object_id)
        try:
            if type(value) in (dict, OrderedDict):
                if any(type(key) is not str for key in value):
                    raise ConfigurationError("Inference state dictionaries require string keys.")
                result = OrderedDict() if type(value) is OrderedDict else {}
                for key, child in value.items():
                    result[key] = copy_value(child, depth + 1)
                if type(value) is OrderedDict:
                    if set(value.__dict__) - {"_metadata"}:
                        raise ConfigurationError("Unsupported custom attributes on an inference state_dict.")
                    if hasattr(value, "_metadata"):
                        result._metadata = copy_value(value._metadata, depth + 1)
            elif type(value) is tuple:
                result = tuple(copy_value(child, depth + 1) for child in value)
            else:
                result = [copy_value(child, depth + 1) for child in value]
            copied[object_id] = result
            return result
        finally:
            active.remove(object_id)

    return copy_value(state)
