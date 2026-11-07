"""A small named-axis wrapper around ordinary autograd-aware tensor operations."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Number
from typing import Any, Callable

import torch

from mlzo.errors import ConfigurationError


def _names(axes: Any) -> tuple[str, ...]:
    if isinstance(axes, str) or not isinstance(axes, Sequence):
        raise ConfigurationError("Axes must be a sequence of unique nonempty names.")
    result = tuple(axes)
    if any(not isinstance(axis, str) or not axis for axis in result):
        raise ConfigurationError("Each axis must have a nonempty string name.")
    if len(set(result)) != len(result):
        raise ConfigurationError("Axis names must be unique.")
    return result


def _arguments(axes: tuple[Any, ...]) -> tuple[str, ...]:
    if len(axes) == 1 and not isinstance(axes[0], str):
        return _names(axes[0])
    return _names(axes)


@dataclass(frozen=True, eq=False)
class NamedTensor:
    """A tensor and one unique name per dimension, with an explicit .raw accessor.

    Supported arithmetic aligns common names and appends new right-hand names
    to the left-hand order. Missing named dimensions broadcast as singleton
    dimensions. Unnamed tensors must be scalars; name other tensors explicitly.
    This wrapper does not implement transparent torch operator dispatch.
    """

    raw: torch.Tensor
    axes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.raw, torch.Tensor):
            raise ConfigurationError("NamedTensor requires a torch.Tensor; use tensor(values, axes=...).")
        axes = _names(self.axes)
        if len(axes) != self.raw.ndim:
            raise ConfigurationError(
                f"Received {len(axes)} axis names for a tensor with {self.raw.ndim} dimensions."
            )
        object.__setattr__(self, "axes", axes)

    @property
    def shape(self) -> torch.Size:
        return self.raw.shape

    @property
    def ndim(self) -> int:
        return self.raw.ndim

    @property
    def dtype(self) -> torch.dtype:
        return self.raw.dtype

    @property
    def device(self) -> torch.device:
        return self.raw.device

    @property
    def requires_grad(self) -> bool:
        return self.raw.requires_grad

    def align(self, *axes: str | Sequence[str]) -> NamedTensor:
        """Return exactly the requested permutation, with no added or omitted axes."""
        names = _arguments(axes)
        if len(names) != len(self.axes) or set(names) != set(self.axes):
            raise ConfigurationError("align() requires an exact permutation of the existing axes.")
        dimensions = tuple(self.axes.index(axis) for axis in names)
        return NamedTensor(self.raw.permute(dimensions), names)

    def flatten(self, axes: Sequence[str], *, into: str) -> NamedTensor:
        """Arrange selected axes in the supplied order, then merge them.

        The merged axis occupies the earliest selected position in the current
        order. Unselected axes retain their relative order.
        """
        names = _names(axes)
        if not names or not set(names).issubset(self.axes):
            raise ConfigurationError("flatten() requires a nonempty selection of existing axes.")
        if not isinstance(into, str) or not into:
            raise ConfigurationError("The flattened axis needs a nonempty string name.")
        remaining = tuple(axis for axis in self.axes if axis not in names)
        if into in remaining:
            raise ConfigurationError("The flattened axis name conflicts with an unselected axis.")
        start = min(self.axes.index(axis) for axis in names)
        before = tuple(axis for axis in self.axes[:start] if axis not in names)
        after = tuple(axis for axis in self.axes[start:] if axis not in names)
        order = (*before, *names, *after)
        arranged = self.align(*order)
        result = arranged.raw.flatten(start_dim=len(before), end_dim=len(before) + len(names) - 1)
        return NamedTensor(result, (*before, into, *after))

    def mean(
        self,
        *axes: str | Sequence[str],
        keepdim: bool = False,
        dtype: torch.dtype | None = None,
    ) -> NamedTensor:
        """Reduce named axes, or all axes if none are supplied."""
        names = _arguments(axes) if axes else self.axes
        if axes and not names:
            raise ConfigurationError("mean() needs at least one axis, or no arguments for all axes.")
        if not set(names).issubset(self.axes):
            raise ConfigurationError("mean() can reduce only existing axes.")
        if not isinstance(keepdim, bool):
            raise ConfigurationError("keepdim must be a boolean.")
        dimensions = tuple(self.axes.index(axis) for axis in names)
        result = self.raw.mean(dim=dimensions, keepdim=keepdim, dtype=dtype)
        remaining = self.axes if keepdim else tuple(axis for axis in self.axes if axis not in names)
        return NamedTensor(result, remaining)

    def to(self, *args: Any, **kwargs: Any) -> NamedTensor:
        return NamedTensor(self.raw.to(*args, **kwargs), self.axes)

    def clone(self, **kwargs: Any) -> NamedTensor:
        return NamedTensor(self.raw.clone(**kwargs), self.axes)

    def detach(self) -> NamedTensor:
        """Explicitly detach the underlying tensor, retaining its axis names."""
        return NamedTensor(self.raw.detach(), self.axes)

    def _broadcast(self, axes: tuple[str, ...]) -> torch.Tensor:
        order = tuple(axis for axis in axes if axis in self.axes)
        arranged = self.align(*order).raw
        sizes = {axis: self.raw.shape[index] for index, axis in enumerate(self.axes)}
        return arranged.reshape(tuple(sizes.get(axis, 1) for axis in axes))

    def _binary(self, other: Any, function: Callable, *, reverse: bool = False) -> NamedTensor:
        if isinstance(other, NamedTensor):
            axes = self.axes + tuple(axis for axis in other.axes if axis not in self.axes)
            for axis in set(self.axes).intersection(other.axes):
                left = self.shape[self.axes.index(axis)]
                right = other.shape[other.axes.index(axis)]
                if left != right and left != 1 and right != 1:
                    raise ConfigurationError(
                        f"Axis {axis!r} cannot broadcast between sizes {left} and {right}."
                    )
            left, right = self._broadcast(axes), other._broadcast(axes)
        else:
            if isinstance(other, torch.Tensor):
                if other.ndim != 0:
                    raise ConfigurationError(
                        "An unnamed nonscalar tensor cannot broadcast with NamedTensor. "
                        "Wrap it with explicit axes first."
                    )
            elif not isinstance(other, Number):
                raise TypeError("Named arithmetic supports named tensors, scalar tensors, and numbers.")
            axes = self.axes
            left, right = self.raw, other
        result = function(right, left) if reverse else function(left, right)
        return NamedTensor(result, axes)

    def __add__(self, other: Any) -> NamedTensor:
        return self._binary(other, operator.add)

    def __radd__(self, other: Any) -> NamedTensor:
        return self._binary(other, operator.add, reverse=True)

    def __sub__(self, other: Any) -> NamedTensor:
        return self._binary(other, operator.sub)

    def __rsub__(self, other: Any) -> NamedTensor:
        return self._binary(other, operator.sub, reverse=True)

    def __mul__(self, other: Any) -> NamedTensor:
        return self._binary(other, operator.mul)

    def __rmul__(self, other: Any) -> NamedTensor:
        return self._binary(other, operator.mul, reverse=True)

    def __truediv__(self, other: Any) -> NamedTensor:
        return self._binary(other, operator.truediv)

    def __rtruediv__(self, other: Any) -> NamedTensor:
        return self._binary(other, operator.truediv, reverse=True)

    def __floordiv__(self, other: Any) -> NamedTensor:
        return self._binary(other, operator.floordiv)

    def __rfloordiv__(self, other: Any) -> NamedTensor:
        return self._binary(other, operator.floordiv, reverse=True)

    def __pow__(self, other: Any) -> NamedTensor:
        return self._binary(other, operator.pow)

    def __rpow__(self, other: Any) -> NamedTensor:
        return self._binary(other, operator.pow, reverse=True)

    def __neg__(self) -> NamedTensor:
        return NamedTensor(-self.raw, self.axes)


def tensor(
    values: Any,
    axes: Sequence[str] | None = None,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    requires_grad: bool | None = None,
) -> NamedTensor | torch.Tensor:
    """Use torch.as_tensor and optionally attach names, preserving Tensor history.

    With no axes this returns a torch.Tensor. Existing Tensor inputs are retained
    when dtype/device agree. Disabling an existing gradient graph requires the
    caller to detach explicitly; this helper never silently detaches it.
    """
    if requires_grad is not None and not isinstance(requires_grad, bool):
        raise ConfigurationError("requires_grad must be a boolean or None.")
    raw = torch.as_tensor(values, dtype=dtype, device=device)
    if requires_grad is False and raw.requires_grad:
        raise ConfigurationError("Use an explicit .detach() to disable an existing tensor's autograd history.")
    if requires_grad is True and not raw.requires_grad:
        raw.requires_grad_(True)
    return raw if axes is None else NamedTensor(raw, _names(axes))
