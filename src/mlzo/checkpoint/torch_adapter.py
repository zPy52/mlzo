"""Restricted tensor state serialization, with PyTorch imported only on use."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, BinaryIO

from mlzo.errors import CheckpointError

from ._validation import MAX_DEPTH, key_path


def _torch() -> Any:
    try:
        import torch
    except ImportError:
        raise CheckpointError("Saving or loading checkpoint state requires the MLZO PyTorch dependency.") from None
    return torch


def _state_tree(state: Any, *, torch: Any, freeze: bool) -> dict[Any, Any]:
    if type(state) not in (dict, OrderedDict):
        raise CheckpointError("state: a complete checkpoint must be a dictionary of tensor/primitive state.")
    active: set[int] = set()
    memo: dict[int, Any] = {}

    def copy_key(key: Any, path: str, depth: int) -> Any:
        if depth > MAX_DEPTH:
            raise CheckpointError(f"{path}: checkpoint key nesting exceeds {MAX_DEPTH} levels.")
        if key is None or type(key) in (str, int, bool, float):
            return key
        if type(key) is tuple:
            return tuple(copy_key(child, f"{path}[{index}]", depth + 1) for index, child in enumerate(key))
        raise CheckpointError(f"{path}: state dictionary keys must be scalar primitives or tuples of them.")

    def visit(item: Any, path: str, depth: int) -> Any:
        if depth > MAX_DEPTH:
            raise CheckpointError(f"{path}: checkpoint nesting exceeds {MAX_DEPTH} levels.")
        if item is None or type(item) in (str, int, bool, float):
            return item
        identity = id(item)
        if identity in active:
            raise CheckpointError(f"{path}: cyclic checkpoint state is unsupported.")
        if identity in memo:
            return memo[identity]
        if type(item) in (torch.Tensor, torch.nn.Parameter):
            if item.__dict__:
                raise CheckpointError(
                    f"{path}: tensor attributes are unsupported custom state. "
                    "Store their primitive/tensor values in a separate dictionary."
                )
            if item.is_meta:
                raise CheckpointError(f"{path}: meta tensors have no data. Materialize the model before saving.")
            try:
                result = item.detach().cpu().clone() if freeze else item.detach()
            except Exception:
                raise CheckpointError(f"{path}: cannot copy this tensor into portable CPU checkpoint state.") from None
            memo[identity] = result
            return result
        if type(item) not in (dict, OrderedDict, list, tuple):
            raise CheckpointError(
                f"{path}: unsupported custom checkpoint state. Use None, bool, int, float, str, "
                "tensors, dictionaries, lists, or tuples; convert custom objects explicitly."
            )
        active.add(identity)
        try:
            if type(item) in (dict, OrderedDict):
                result = OrderedDict() if type(item) is OrderedDict else {}
                for index, (key, child) in enumerate(item.items()):
                    child_path = key_path(path, key, index)
                    result[copy_key(key, child_path, depth + 1)] = visit(child, child_path, depth + 1)
                if type(item) is OrderedDict:
                    for attribute, child in item.__dict__.items():
                        if attribute != "_metadata":
                            raise CheckpointError(f"{path}: OrderedDict custom attributes other than _metadata are unsupported.")
                        result._metadata = visit(child, f"{path}._metadata", depth + 1)
            elif type(item) is tuple:
                result = tuple(visit(child, f"{path}[{index}]", depth + 1) for index, child in enumerate(item))
            else:
                result = [visit(child, f"{path}[{index}]", depth + 1) for index, child in enumerate(item)]
            memo[identity] = result
            return result
        finally:
            active.remove(identity)

    return visit(state, "state", 0)


def prepare_state(state: Any) -> tuple[dict[Any, Any], str]:
    """Validate and freeze state, preserving integer keys, tuples, and state_dict metadata."""
    torch = _torch()
    return _state_tree(state, torch=torch, freeze=True), str(torch.__version__)


def write_state(state: dict[Any, Any], stream: BinaryIO) -> None:
    try:
        _torch().save(state, stream)
    except CheckpointError:
        raise
    except Exception:
        raise CheckpointError("Cannot serialize the validated tensor/primitive state with PyTorch.") from None


def read_state(stream: BinaryIO) -> dict[Any, Any]:
    """Load verified bytes using the restricted loader, without a pickle fallback."""
    torch = _torch()
    try:
        state = torch.load(stream, map_location="cpu", weights_only=True)
    except Exception:
        raise CheckpointError(
            "Restricted checkpoint loading failed. The state must use the supported tensor/primitive "
            "format and a compatible PyTorch version; arbitrary pickle loading is never attempted."
        ) from None
    return _state_tree(state, torch=torch, freeze=False)
