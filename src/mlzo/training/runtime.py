"""Single-process numerical settings, nested device moves, and restricted state."""

from __future__ import annotations

import math
import random
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import fields, is_dataclass, replace
from threading import Lock
from typing import Any

import torch

from mlzo.errors import ConfigurationError, TrainingError, UnsupportedFeatureError

_EXECUTION_LOCK = Lock()


@contextmanager
def _evaluating(model: torch.nn.Module) -> Iterator[None]:
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        # Recursive train(mode) would overwrite intentionally different child modes.
        for module, mode in modes:
            module.training = mode


@contextmanager
def execution_settings(matmul: str):
    if not _EXECUTION_LOCK.acquire(blocking=False):
        raise TrainingError("Concurrent MLZO execution in one process is unsupported because numerical settings are process-wide.")
    old_precision = torch.get_float32_matmul_precision()
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    try:
        fast = matmul == "fast"
        torch.set_float32_matmul_precision("high" if fast else "highest")
        torch.backends.cuda.matmul.allow_tf32 = fast
        torch.backends.cudnn.allow_tf32 = fast
        yield
    finally:
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn
        _EXECUTION_LOCK.release()


def move_batch(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        moved = {key: move_batch(item, device) for key, item in value.items()}
        try:
            return type(value)(moved)
        except TypeError:
            return moved
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(move_batch(item, device) for item in value))
    if isinstance(value, tuple):
        return tuple(move_batch(item, device) for item in value)
    if isinstance(value, list):
        return [move_batch(item, device) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{f.name: move_batch(getattr(value, f.name), device) for f in fields(value) if f.init})
    return value


def normalization_weight(value: Any) -> float:
    if value is None:
        return 1.0
    if isinstance(value, torch.Tensor):
        if value.ndim != 0 or value.is_complex() or value.requires_grad or value.grad_fn is not None:
            raise ConfigurationError("Step.weight must be a detached real scalar with no gradient history.")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ConfigurationError("Step.weight must be finite and nonnegative.")
    return float(value)


def restricted_copy(value: Any, path: str = "state", *, cpu: bool = True, _active: set[int] | None = None) -> Any:
    """Snapshot without retaining autograd, device tensors, or arbitrary objects."""
    if isinstance(value, torch.Tensor):
        detached = value.detach()
        return detached.cpu().clone() if cpu else detached.clone()
    if value is None or type(value) in (str, bool, int, float):
        return value
    active = set() if _active is None else _active
    if id(value) in active:
        raise ConfigurationError(f"Cyclic checkpoint state at {path}.")
    active.add(id(value))
    try:
        if isinstance(value, Mapping):
            if any(not isinstance(key, (str, int)) or isinstance(key, bool) for key in value):
                raise ConfigurationError(f"Checkpoint mapping keys must be strings or integers at {path}.")
            copied = {key: restricted_copy(item, f"{path}.{key}", cpu=cpu, _active=active) for key, item in value.items()}
            if isinstance(value, OrderedDict):
                result = OrderedDict(copied)
                if hasattr(value, "_metadata"):
                    result._metadata = restricted_copy(value._metadata, f"{path}._metadata", cpu=cpu, _active=active)
                return result
            return copied
        if isinstance(value, (tuple, list)):
            items = [restricted_copy(item, f"{path}[{index}]", cpu=cpu, _active=active) for index, item in enumerate(value)]
            return tuple(items) if isinstance(value, tuple) else items
    finally:
        active.remove(id(value))
    raise ConfigurationError(f"Unsupported checkpoint object {type(value).__qualname__} at {path}; use tensors and primitive containers.")


def seed_process(seed: int | None, device: torch.device) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    elif device.type == "mps" and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)
    try:
        import numpy as np
    except ImportError:
        return
    np.random.seed(seed % (2**32))


def rng_state(device: torch.device) -> dict[str, Any]:
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    if device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
        state["mps"] = torch.mps.get_rng_state()
    try:
        import numpy as np
    except ImportError:
        return state
    name, keys, position, has_gauss, cached = np.random.get_state()
    state["numpy"] = (name, keys.tolist(), position, has_gauss, cached)
    return state


def restore_rng(state: Mapping[str, Any], device: torch.device) -> None:
    def tuples(value: Any) -> Any:
        return tuple(tuples(item) for item in value) if isinstance(value, (tuple, list)) else value
    random.setstate(tuples(state["python"]))
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state:
        if device.type != "cuda":
            raise UnsupportedFeatureError("Exact CUDA RNG restoration requires a CUDA execution device.")
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)
    if "mps" in state:
        if device.type != "mps" or not hasattr(torch.mps, "set_rng_state"):
            raise UnsupportedFeatureError("Exact MPS RNG restoration requires a compatible MPS execution device.")
        torch.mps.set_rng_state(state["mps"].cpu())
    if "numpy" in state:
        try:
            import numpy as np
        except ImportError as exc:
            raise ConfigurationError("This checkpoint requires NumPy RNG restoration.") from exc
        name, keys, position, has_gauss, cached = state["numpy"]
        np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached))
