"""Synchronous local PyTorch inference without a network server."""

from __future__ import annotations

from collections import OrderedDict
from threading import RLock
from typing import Any

import torch

from mlzo.errors import ConfigurationError, MLZOError, UnsupportedFeatureError
from mlzo.transforms.tensor import Tensor

from .artifact import _load_artifact


def _move_inputs(value: Any, device: torch.device, *, active: set[int] | None = None, depth: int = 0) -> Any:
    if depth > 64:
        raise ConfigurationError("Prediction input nesting exceeds 64 levels.")
    if isinstance(value, Tensor):
        value = value.raw
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if not isinstance(value, (dict, list, tuple)):
        return value
    if active is None:
        active = set()
    if id(value) in active:
        raise ConfigurationError("Cyclic prediction input containers are unsupported.")
    active.add(id(value))
    try:
        def visit(child: Any) -> Any:
            return _move_inputs(child, device, active=active, depth=depth + 1)

        if isinstance(value, dict):
            pairs = ((key, visit(child)) for key, child in value.items())
            return OrderedDict(pairs) if isinstance(value, OrderedDict) else dict(pairs)
        if isinstance(value, list):
            return [visit(child) for child in value]
        if isinstance(getattr(type(value), "_fields", None), tuple):
            return type(value)(*(visit(child) for child in value))
        return tuple(visit(child) for child in value)
    finally:
        active.remove(id(value))


class Deployment:
    """A loaded local artifact with synchronous, serialized predict calls.

    Tensors in ordinary dictionaries, lists, tuples, and namedtuples move to the
    configured device. MLZO tensor leaves unwrap without reshaping. Other
    values pass through for the model to validate.
    Outputs retain the model's ordinary types and device. Shutdown releases this
    deployment's model reference, and never changes process-wide device caches.
    """

    def __init__(self, artifact: Any, *, device: str | torch.device = "cpu", factory: Any = None) -> None:
        try:
            self._device = torch.device(device)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ConfigurationError(f"Invalid inference device {device!r}.") from exc
        self._model, self.artifact = _load_artifact(artifact, device=self._device, factory=factory)
        self._lock = RLock()
        self._closed = False

    @property
    def device(self) -> str:
        return str(self._device)

    def _require_open(self) -> None:
        if self._closed:
            raise MLZOError("This local deployment has been shut down.")

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            self._require_open()
            moved_args, moved_kwargs = _move_inputs((args, kwargs), self._device)
            self._model.eval()
            with torch.inference_mode():
                return self._model(*moved_args, **moved_kwargs)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": "shutdown" if self._closed else "ready",
                "ready": not self._closed,
                "backend": "local",
                "device": self.device,
                "artifact_id": self.artifact.artifact_id,
                "artifact": self.artifact.to_dict(),
            }

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
            self._model = None

    def __enter__(self) -> Deployment:
        with self._lock:
            self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.shutdown()


def deploy(artifact: Any, *, on: Any = None, device: str | torch.device | None = None, factory: Any = None) -> Deployment:
    """Load a local deployment, using CPU unless device is explicitly supplied.

    ``on`` is reserved for remote placement. Any non-None target, including a
    cluster handle, raises UnsupportedFeatureError in this alpha.
    """
    if on is not None:
        raise UnsupportedFeatureError("Remote inference deployment is unsupported in MLZO 0.1.0a1. Use deploy(artifact, device=...) locally.")
    return Deployment(artifact, device="cpu" if device is None else device, factory=factory)
