"""Explicit tensor wrappers; all storage, kernels and differentiation belong to PyTorch."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Callable

import torch

from mlzo.errors import ConfigurationError


_UNSET = object()


def _dtype(value: str | torch.dtype | None) -> torch.dtype | None:
    if isinstance(value, str):
        value = getattr(torch, value, None)
        if not isinstance(value, torch.dtype):
            raise ConfigurationError("Unknown dtype. Use a torch dtype name such as 'float32'.")
    if value is not None and not isinstance(value, torch.dtype):
        raise ConfigurationError("dtype must be a torch.dtype or a string such as 'float32'.")
    return value


def _raw(value: Any) -> Any:
    return value.raw if isinstance(value, Tensor) else value


def _basic_index(index: Any) -> bool:
    items = index if isinstance(index, tuple) else (index,)
    return all(
        item is None or item is Ellipsis or isinstance(item, slice)
        or (isinstance(item, Integral) and not isinstance(item, bool))
        for item in items
    )


def _index(raw: torch.Tensor, index: Any) -> Any:
    """Preserve dimensions only when the entire index is basic, never for masks/arrays."""
    if not _basic_index(index):
        return index
    items = index if isinstance(index, tuple) else (index,)
    consumed = sum(item is not None and item is not Ellipsis for item in items)
    if consumed > raw.ndim or sum(item is Ellipsis for item in items) > 1:
        # Let PyTorch handle repeated ellipses and report too many dimensions.
        return index
    result = []
    dimension = 0
    for item in items:
        if item is None:
            result.append(None)
            continue
        if item is Ellipsis:
            missing = raw.ndim - consumed
            result.extend([slice(None)] * missing)
            dimension += missing
            continue
        if isinstance(item, Integral):
            size = raw.shape[dimension]
            position = int(item)
            if position < -size or position >= size:
                raise IndexError(
                    f"Index {position} is out of bounds for dimension {dimension} with size {size}."
                )
            if position < 0:
                position += size
            item = slice(position, position + 1)
        result.append(item)
        dimension += 1
    return tuple(result)


def _reduction_arguments(axes, axis, dim, keepdims, keepdim):
    if (axes and (axis is not _UNSET or dim is not _UNSET)) or (axis is not _UNSET and dim is not _UNSET):
        raise ConfigurationError("Specify reduction axes once, using positional axes, axis, or dim.")
    if len(axes) == 2 and isinstance(axes[1], bool):
        if keepdims is not None or keepdim is not None:
            raise ConfigurationError("Specify keepdims/keepdim only once.")
        axes, keepdim = axes[:1], axes[1]
    if axes:
        selection = axes[0] if len(axes) == 1 else axes
    else:
        selection = axis if axis is not _UNSET else dim if dim is not _UNSET else None
    for value in (keepdims, keepdim):
        if value is not None and not isinstance(value, bool):
            raise ConfigurationError("keepdims and keepdim must be booleans.")
    if keepdims is not None and keepdim is not None and keepdims != keepdim:
        raise ConfigurationError("keepdims and keepdim disagree.")
    keep = keepdims if keepdims is not None else keepdim if keepdim is not None else True
    return selection, keep


@dataclass(frozen=True, eq=False)
class Tensor:
    """A lightweight wrapper with explicit, nontransparent PyTorch interoperability.

    ``raw`` is the ordinary tensor, retaining its storage and autograd graph.
    Only the methods declared here are exposed. Use ``.raw`` for other native
    methods, attributes, torch functions and in-place methods such as ``add_``.
    Indexed assignment and augmented arithmetic mutate raw storage and retain
    PyTorch's broadcasting, casting and autograd restrictions.

    Basic indexing replaces each integer with a checked singleton
    slice. If any index is advanced or boolean, the whole index uses native
    semantics. Explicit shape methods never normalize their results.
    """

    raw: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.raw, torch.Tensor):
            raise ConfigurationError("Tensor requires a torch.Tensor; use tensor(values) to convert data.")

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

    def _with_raw(self, raw: torch.Tensor) -> Tensor:
        return Tensor(raw)

    def to(self, *args: Any, **kwargs: Any) -> Tensor:
        return self._with_raw(self.raw.to(*args, **kwargs))

    def astype(self, dtype: str | torch.dtype) -> Tensor:
        return self.to(dtype=_dtype(dtype))

    def on(self, device: str | torch.device, **kwargs: Any) -> Tensor:
        return self.to(device=device, **kwargs)

    def add_dim(self, axis: int) -> Tensor:
        """Insert a singleton axis. Named inputs return an unnamed Tensor."""
        return Tensor(self.raw.unsqueeze(axis))

    def remove_dim(self, axis: int) -> Tensor:
        if self.raw.size(axis) != 1:
            raise ConfigurationError("remove_dim() can remove only a dimension of size 1.")
        return Tensor(self.raw.squeeze(axis))

    def transpose(self, axis1: int = -2, axis2: int = -1) -> Tensor:
        return Tensor(self.raw.transpose(axis1, axis2))

    def reorder(self, *axes: int) -> Tensor:
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        return Tensor(self.raw.permute(axes))

    def reshape(self, *shape: int) -> Tensor:
        """Use the requested shape exactly; named inputs return an unnamed Tensor."""
        return Tensor(self.raw.reshape(*shape))

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> Tensor:
        return Tensor(self.raw.flatten(start_dim, end_dim))

    def _dimensions(self, selection: Any) -> Any:
        return selection

    def _reduced(self, raw: torch.Tensor, dimensions: Any, keep: bool) -> Tensor:
        return Tensor(raw)

    def _reduce(self, operation, axes, axis, dim, keepdims, keepdim, dtype):
        selection, keep = _reduction_arguments(axes, axis, dim, keepdims, keepdim)
        dimensions = self._dimensions(selection)
        result = getattr(self.raw, operation)(dim=dimensions, keepdim=keep, dtype=_dtype(dtype))
        return self._reduced(result, dimensions, keep)

    def mean(
        self, *axes, axis=_UNSET, dim=_UNSET,
        keepdims: bool | None = None, keepdim: bool | None = None,
        dtype: str | torch.dtype | None = None,
    ) -> Tensor:
        """Mean over integer or named axes; keep all reduced dimensions by default.

        ``dim`` and ``keepdim`` retain their PyTorch spellings. Omit axes or use
        None for a global reduction. A positional keepdim boolean is accepted.
        """
        return self._reduce("mean", axes, axis, dim, keepdims, keepdim, dtype)

    def sum(
        self, *axes, axis=_UNSET, dim=_UNSET,
        keepdims: bool | None = None, keepdim: bool | None = None,
        dtype: str | torch.dtype | None = None,
    ) -> Tensor:
        """Sum with the same axis and keepdims conventions as mean()."""
        return self._reduce("sum", axes, axis, dim, keepdims, keepdim, dtype)

    def detach(self) -> Tensor:
        return self._with_raw(self.raw.detach())

    def clone(self, **kwargs: Any) -> Tensor:
        return self._with_raw(self.raw.clone(**kwargs))

    def backward(self, gradient=None, *args: Any, **kwargs: Any) -> None:
        self.raw.backward(_raw(gradient), *args, **kwargs)

    def __getitem__(self, index: Any) -> Tensor:
        return Tensor(self.raw[_index(self.raw, index)])

    def __setitem__(self, index: Any, value: Any) -> None:
        self.raw[_index(self.raw, index)] = _raw(value)

    def _binary(self, other: Any, function: Callable, *, reverse: bool = False) -> Tensor:
        # NamedTensor owns alignment when either operand has axis names.
        if isinstance(other, Tensor) and hasattr(other, "axes"):
            return NotImplemented
        left, right = self.raw, _raw(other)
        return Tensor(function(right, left) if reverse else function(left, right))

    def __add__(self, other):
        return self._binary(other, operator.add)

    def __radd__(self, other):
        return self._binary(other, operator.add, reverse=True)

    def __sub__(self, other):
        return self._binary(other, operator.sub)

    def __rsub__(self, other):
        return self._binary(other, operator.sub, reverse=True)

    def __mul__(self, other):
        return self._binary(other, operator.mul)

    def __rmul__(self, other):
        return self._binary(other, operator.mul, reverse=True)

    def __truediv__(self, other):
        return self._binary(other, operator.truediv)

    def __rtruediv__(self, other):
        return self._binary(other, operator.truediv, reverse=True)

    def __floordiv__(self, other):
        return self._binary(other, operator.floordiv)

    def __rfloordiv__(self, other):
        return self._binary(other, operator.floordiv, reverse=True)

    def __pow__(self, other):
        return self._binary(other, operator.pow)

    def __rpow__(self, other):
        return self._binary(other, operator.pow, reverse=True)

    def matmul(self, other: Any) -> Tensor:
        """Positional PyTorch matmul; the result has no axis names."""
        return Tensor(self.raw.matmul(_raw(other)))

    def __matmul__(self, other):
        return self.matmul(other)

    def __rmatmul__(self, other):
        return Tensor(_raw(other) @ self.raw)

    def __neg__(self):
        return self._with_raw(-self.raw)

    def _inplace(self, other: Any, function: Callable) -> Tensor:
        if isinstance(other, Tensor) and hasattr(other, "axes"):
            raise ConfigurationError(
                "In-place arithmetic needs named axes on both tensors; use .raw for positional arithmetic."
            )
        function(self.raw, _raw(other))
        return self

    def __iadd__(self, other):
        return self._inplace(other, operator.iadd)

    def __isub__(self, other):
        return self._inplace(other, operator.isub)

    def __imul__(self, other):
        return self._inplace(other, operator.imul)

    def __itruediv__(self, other):
        return self._inplace(other, operator.itruediv)

    def __ifloordiv__(self, other):
        return self._inplace(other, operator.ifloordiv)

    def __ipow__(self, other):
        return self._inplace(other, operator.ipow)


def wrap(value: torch.Tensor | Tensor) -> Tensor:
    """Wrap a native tensor without normalization, conversion, copying or detaching."""
    return value if isinstance(value, Tensor) else Tensor(value)
