"""A small named-axis wrapper around ordinary autograd-aware tensor operations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Number
from typing import Any, Callable

import torch

from mlzo.errors import ConfigurationError

from .tensor import Tensor, _basic_index, _dtype


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
class NamedTensor(Tensor):
    """A tensor and one unique name per dimension, with an explicit .raw accessor.

    Supported arithmetic aligns common names and appends new right-hand names
    to the left-hand order. Missing named dimensions broadcast as singleton
    dimensions. Unnamed tensors must be scalars; name other tensors explicitly.
    This wrapper does not implement transparent torch operator dispatch.
    Transpose, reorder, singleton removal and basic indexing retain names.
    Adding an axis, reshape, advanced indexing and positional matmul return an
    unnamed Tensor rather than guessing axis names. Reattach names explicitly.
    flatten(axes, into=...) retains its named rearrangement contract.
    """

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

    def _with_raw(self, raw: torch.Tensor) -> NamedTensor:
        return NamedTensor(raw, self.axes)

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

    def _dimensions(self, selection: Any) -> Any:
        if selection is None:
            return None
        values = (selection,) if isinstance(selection, (str, int)) else tuple(selection)
        dimensions = []
        for axis in values:
            if isinstance(axis, str):
                if axis not in self.axes:
                    raise ConfigurationError(f"Unknown reduction axis {axis!r}.")
                axis = self.axes.index(axis)
            dimensions.append(axis)
        return tuple(dimensions)

    def _reduced(self, raw: torch.Tensor, dimensions: Any, keep: bool) -> NamedTensor:
        if keep:
            return NamedTensor(raw, self.axes)
        removed = (
            set(range(self.ndim)) if not dimensions
            else {axis % max(self.ndim, 1) for axis in dimensions}
        )
        remaining = tuple(axis for index, axis in enumerate(self.axes) if index not in removed)
        return NamedTensor(raw, remaining)

    def transpose(self, axis1: int = -2, axis2: int = -1) -> NamedTensor:
        result = self.raw.transpose(axis1, axis2)
        if not self.axes:
            return NamedTensor(result, ())
        axes = list(self.axes)
        axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
        return NamedTensor(result, tuple(axes))

    def reorder(self, *axes: int) -> NamedTensor:
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        result = self.raw.permute(axes)
        return NamedTensor(result, tuple(self.axes[axis] for axis in axes))

    def remove_dim(self, axis: int) -> NamedTensor:
        result = super().remove_dim(axis)
        position = axis % max(self.ndim, 1)
        return NamedTensor(result.raw, tuple(name for index, name in enumerate(self.axes) if index != position))

    def __getitem__(self, index: Any) -> Tensor:
        result = super().__getitem__(index)
        items = index if isinstance(index, tuple) else (index,)
        if _basic_index(index) and all(item is not None for item in items):
            return NamedTensor(result.raw, self.axes)
        return result

    def _broadcast(self, axes: tuple[str, ...]) -> torch.Tensor:
        order = tuple(axis for axis in axes if axis in self.axes)
        arranged = self.align(*order).raw
        sizes = {axis: self.raw.shape[index] for index, axis in enumerate(self.axes)}
        return arranged.reshape(tuple(sizes.get(axis, 1) for axis in axes))

    def _operands(self, other: Any):
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
            if isinstance(other, Tensor):
                other = other.raw
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
        return axes, left, right

    def _binary(self, other: Any, function: Callable, *, reverse: bool = False) -> NamedTensor:
        axes, left, right = self._operands(other)
        return NamedTensor(function(right, left) if reverse else function(left, right), axes)

    def _inplace(self, other: Any, function: Callable) -> NamedTensor:
        axes, _, right = self._operands(other)
        if axes != self.axes:
            raise ConfigurationError("In-place named arithmetic cannot add axes; use ordinary arithmetic.")
        function(self.raw, right)
        return self


def tensor(
    values: Any,
    axes: Sequence[str] | None = None,
    *,
    dtype: str | torch.dtype | None = None,
    device: torch.device | str | None = None,
    requires_grad: bool | None = None,
    normalize: bool = True,
) -> Tensor:
    """Convert with torch.as_tensor, preserving aliases/history, then wrap.

    By default scalars become (1, 1) and vectors become (N, 1). normalize=False
    preserves shape. With axes, supply one name per *resulting* dimension; names
    are never invented for normalization. Existing NamedTensor names survive
    unless explicitly replaced. Use normalize=False for named rank 0/1 inputs.
    To disable an existing gradient graph, detach explicitly first.
    """
    if requires_grad is not None and not isinstance(requires_grad, bool):
        raise ConfigurationError("requires_grad must be a boolean or None.")
    if not isinstance(normalize, bool):
        raise ConfigurationError("normalize must be a boolean.")
    if isinstance(values, NamedTensor) and axes is None:
        axes = values.axes
    source = values.raw if isinstance(values, Tensor) else values
    if requires_grad is False and isinstance(source, torch.Tensor) and source.requires_grad:
        raise ConfigurationError("Use an explicit .detach() to disable an existing tensor's autograd history.")
    raw = torch.as_tensor(source, dtype=_dtype(dtype), device=device)
    if normalize and raw.ndim < 2:
        raw = raw.reshape(1, 1) if raw.ndim == 0 else raw.unsqueeze(-1)
        if axes is not None and len(_names(axes)) != 2:
            raise ConfigurationError(
                "Normalization requires names for both resulting dimensions. "
                "Provide both names or use normalize=False to preserve the original shape."
            )
    result = Tensor(raw) if axes is None else NamedTensor(raw, _names(axes))
    if requires_grad is True and not raw.requires_grad:
        raw.requires_grad_(True)
    return result
