"""Deferred PyTorch modules and canonical, identity-preserving materialization."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from threading import RLock
from typing import Any, Literal

import torch
from torch import nn

from mlzo.errors import ConfigurationError, TrainingError

from .placement import _explicit_gpu, place, placement_map

LazyState = Literal["unmaterialized", "materializing", "ready", "failed"]


class Lazy(nn.Module):
    """Store a local callable recipe without constructing its module.

    A ready wrapper shares the concrete module's registries. This keeps parameter
    and checkpoint names canonical even when a caller retains the wrapper.
    """

    def __init__(self, factory: Callable[..., nn.Module], *args: Any, gpu: int | None = None, **kwargs: Any) -> None:
        super().__init__()
        if not callable(factory):
            raise ConfigurationError("Lazy requires a callable module factory.")
        object.__setattr__(self, "factory", factory)
        self.args = args
        self.kwargs = dict(kwargs)
        self._state: LazyState = "unmaterialized"
        self._error: BaseException | None = None
        self._construction_lock = RLock()
        object.__setattr__(self, "_mlzo_target", None)
        if gpu is not None:
            place(self, gpu=gpu)

    @property
    def state(self) -> LazyState:
        return self._state

    @property
    def error(self) -> BaseException | None:
        return self._error

    def _share_registries(self, target: nn.Module) -> None:
        for name in ("_parameters", "_buffers", "_modules", "_non_persistent_buffers_set"):
            object.__setattr__(self, name, getattr(target, name))

    def _construct(self, device: torch.device) -> nn.Module:
        with self._construction_lock:
            if self._state == "ready":
                return self._mlzo_target
            if self._state == "failed":
                assert self._error is not None
                raise self._error
            if self._state == "materializing":
                raise TrainingError("Recursive materialization of the same Lazy module.")
            self._state = "materializing"
            try:
                context = device if hasattr(device, "__enter__") else nullcontext()
                with context:
                    target = self.factory(*self.args, **self.kwargs)
                if not isinstance(target, nn.Module) or target is self:
                    raise ConfigurationError("A Lazy factory must return a different torch.nn.Module.")
                # Respect an explicit eval() on the wrapper without resetting a
                # factory's custom train/eval choices during ordinary construction.
                if not self.training:
                    target.train(False)
                object.__setattr__(self, "_mlzo_target", target)
                self._share_registries(target)
                self.training = target.training
                self._state = "ready"
                return target
            except BaseException as exc:
                self._error = exc
                self._state = "failed"
                raise

    def materialize(self, device: str | torch.device = "cpu") -> nn.Module:
        return materialize(self, device)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self._state != "ready":
            raise TrainingError(
                f"Lazy is {self._state}; forward() never constructs weights. "
                "Call mlzo.materialize(model, device) or prepare through run.fit() first."
            )
        return self._mlzo_target(*args, **kwargs)

    def train(self, mode: bool = True) -> Lazy:
        if self._state == "ready":
            if not isinstance(mode, bool):
                raise ValueError("training mode must be a boolean")
            self._mlzo_target.train(mode)
            self.training = self._mlzo_target.training
        else:
            super().train(mode)
        return self

    def _apply(self, fn: Callable[..., Any], recurse: bool = True) -> Lazy:
        if self._state == "ready":
            self._mlzo_target._apply(fn, recurse=recurse)
            self._share_registries(self._mlzo_target)
        else:
            super()._apply(fn, recurse=recurse)
        return self

    def state_dict(self, *args: Any, **kwargs: Any) -> Any:
        if self._state == "ready":
            return self._mlzo_target.state_dict(*args, **kwargs)
        return super().state_dict(*args, **kwargs)

    def _load_from_state_dict(self, *args: Any, **kwargs: Any) -> None:
        if self._state == "ready":
            self._mlzo_target._load_from_state_dict(*args, **kwargs)
            self._share_registries(self._mlzo_target)
        else:
            super()._load_from_state_dict(*args, **kwargs)

    def extra_repr(self) -> str:
        name = getattr(self.factory, "__qualname__", type(self.factory).__qualname__)
        gpu = _explicit_gpu(self)
        return f"factory={name}, args={self.args!r}, kwargs={self.kwargs!r}, state={self._state}, gpu={gpu!r}"


@contextmanager
def _seeded(seed: int | None, device: torch.device) -> Iterator[None]:
    if seed is None:
        yield
        return
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigurationError("seed must be an integer or None.")
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    python_state = random.getstate()
    mps_state = None
    if device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
        mps_state = torch.mps.get_rng_state()
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.random.default_generator.manual_seed(seed)
            random.seed(seed)
            if cuda_devices:
                with torch.cuda.device(cuda_devices[0]):
                    torch.cuda.manual_seed(seed)
            if mps_state is not None:
                torch.mps.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


def get_recipe(module: nn.Module) -> dict[str, Any] | None:
    """Return the original root recipe with its ordinary local callable."""
    if isinstance(module, Lazy):
        return {"factory": module.factory, "args": module.args, "kwargs": dict(module.kwargs)}
    recipe = module.__dict__.get("_mlzo_recipe")
    return None if recipe is None else {**recipe, "kwargs": dict(recipe["kwargs"])}


def materialize(module: nn.Module, device: str | torch.device, seed: int | None = None) -> nn.Module:
    """Resolve lazy roots/children once, retain aliases, and move to one device.

    Factories run in the destination-device context where PyTorch supports it.
    Explicit CPU allocations in a factory are moved afterwards; initialization
    is never replaced with reset_parameters() or a second factory invocation.
    """
    if not isinstance(module, nn.Module):
        raise ConfigurationError("materialize() requires a torch.nn.Module.")
    try:
        destination = torch.device(device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ConfigurationError(f"Invalid materialization device {device!r}.") from exc
    if destination.type not in {"cpu", "cuda", "mps"}:
        raise ConfigurationError("Materialization supports cpu, cuda, or mps devices.")
    placement_map(module)
    root_recipe = get_recipe(module)
    resolved: dict[int, nn.Module] = {}
    active: set[int] = set()
    annotations: dict[int, int] = {}

    def visit(node: nn.Module) -> nn.Module:
        identity = id(node)
        if identity in active:
            raise ConfigurationError("Cyclic module or Lazy factory reference during materialization.")
        if identity in resolved:
            return resolved[identity]
        active.add(identity)
        try:
            if isinstance(node, Lazy):
                explicit = _explicit_gpu(node)
                target = visit(node._construct(destination))
                if explicit is not None:
                    earlier = annotations.setdefault(id(target), explicit)
                    if earlier != explicit:
                        raise ConfigurationError("Lazy factories returned the same module with conflicting GPU annotations.")
                    target._mlzo_gpu = explicit
                if "_mlzo_recipe" not in target.__dict__:
                    target._mlzo_recipe = get_recipe(node)
                object.__setattr__(node, "_mlzo_target", target)
                node._share_registries(target)
                node.training = target.training
            else:
                target = node
                for name, child in list(target._modules.items()):
                    if child is not None:
                        target._modules[name] = visit(child)
            resolved[identity] = target
            return target
        finally:
            active.remove(identity)

    with _seeded(seed, destination):
        result = visit(module)
        if root_recipe is not None:
            result._mlzo_recipe = root_recipe
        placement_map(result)
        result.to(destination)
    return result
