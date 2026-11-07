---
title: Tensors and transforms
description: Public MLZO 0.1.0a1 API signatures and behavior for tensors and transforms.
---

Read the [tensors and transforms guide](../../../guides/transforms/) for examples and usage constraints. Signatures and declared fields below come from MLZO `0.1.0a1` source. Type names follow their defining Python modules.

## Tensor

Declared fields:

```python
raw: torch.Tensor
```

An explicit lightweight wrapper over native tensor storage and autograd. Only declared methods are supported; use `.raw` for other native operations, modules, optimizers, function transforms, and compiled regions. Basic integer indexing keeps singleton axes; any advanced/boolean component makes the whole index native. Explicit shape operations keep their requested rank.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/transforms/tensor.py#L97).

### Tensor.shape

```python
Tensor.shape: torch.Size  # read-only property
```

Returns the underlying tensor's exact shape.

### Tensor.ndim

```python
Tensor.ndim: int  # read-only property
```

Returns the underlying tensor's rank.

### Tensor.dtype

```python
Tensor.dtype: torch.dtype  # read-only property
```

Returns the underlying tensor's PyTorch dtype.

### Tensor.device

```python
Tensor.device: torch.device  # read-only property
```

Returns the underlying tensor's device.

### Tensor.requires_grad

```python
Tensor.requires_grad: bool  # read-only property
```

Reports whether autograd tracks the underlying tensor.

### Tensor.to

```python
Tensor.to(*args: Any, **kwargs: Any) -> Tensor
```

Uses native PyTorch conversion semantics and wraps the result. Named inputs retain names.

### Tensor.astype

```python
Tensor.astype(dtype: str | torch.dtype) -> Tensor
```

Converts dtype using a torch dtype or its string name, such as `float32`. Named inputs retain names.

### Tensor.on

```python
Tensor.on(device: str | torch.device, **kwargs: Any) -> Tensor
```

Transfers to a device using native PyTorch semantics. Named inputs retain names.

### Tensor.add_dim

```python
Tensor.add_dim(axis: int) -> Tensor
```

Insert a singleton axis. Named inputs return an unnamed Tensor.

### Tensor.remove_dim

```python
Tensor.remove_dim(axis: int) -> Tensor
```

Removes only the specified singleton dimension; raises if its size is not one. The result is not normalized.

### Tensor.transpose

```python
Tensor.transpose(axis1: int=-2, axis2: int=-1) -> Tensor
```

Swaps two dimensions, defaulting to the last two. Explicit shape results are not normalized.

### Tensor.reorder

```python
Tensor.reorder(*axes: int) -> Tensor
```

Permutes dimensions in the supplied integer order.

### Tensor.reshape

```python
Tensor.reshape(*shape: int) -> Tensor
```

Use the requested shape exactly; named inputs return an unnamed Tensor.

### Tensor.flatten

```python
Tensor.flatten(start_dim: int=0, end_dim: int=-1) -> Tensor
```

Flattens the selected positional dimension range using native PyTorch semantics, without result normalization.

### Tensor.mean

```python
Tensor.mean(
    *axes,
    axis = _UNSET,
    dim = _UNSET,
    keepdims: bool | None = None,
    keepdim: bool | None = None,
    dtype: str | torch.dtype | None = None,
) -> Tensor
```

Mean over integer or named axes; keep all reduced dimensions by default.

`dim` and `keepdim` retain their PyTorch spellings. Omit axes or use
None for a global reduction. A positional keepdim boolean is accepted.
Integer/bool inputs with dtype omitted or None use
torch.get_default_dtype(). Other inputs and explicit dtypes follow
PyTorch. The input dtype/storage and native .raw.mean() are unchanged.

### Tensor.sum

```python
Tensor.sum(
    *axes,
    axis = _UNSET,
    dim = _UNSET,
    keepdims: bool | None = None,
    keepdim: bool | None = None,
    dtype: str | torch.dtype | None = None,
) -> Tensor
```

Sum with the same axis and keepdims conventions as mean().

### Tensor.detach

```python
Tensor.detach() -> Tensor
```

Detaches from autograd and wraps the result. Named inputs retain names.

### Tensor.clone

```python
Tensor.clone(**kwargs: Any) -> Tensor
```

Clones native storage with the supplied PyTorch options. Named inputs retain names.

### Tensor.backward

```python
Tensor.backward(gradient=None, *args: Any, **kwargs: Any) -> None
```

Delegates differentiation to the native tensor, unwrapping an optional MLZO gradient tensor.

### Tensor.matmul

```python
Tensor.matmul(other: Any) -> Tensor
```

Positional PyTorch matmul; the result has no axis names.

## wrap

```python
mlzo.wrap(value: torch.Tensor | Tensor) -> Tensor
```

Wraps a native tensor without conversion, normalization, copying, or detaching: `wrap(native).raw is native`. An existing MLZO wrapper is returned unchanged.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/transforms/tensor.py#L310).

## tensor

```python
mlzo.tensor(
    values: Any,
    axes: Sequence[str] | None = None,
    *,
    dtype: str | torch.dtype | None = None,
    device: torch.device | str | None = None,
    requires_grad: bool | None = None,
    normalize: bool = True,
) -> Tensor
```

Converts with `torch.as_tensor` and returns `Tensor`, or `NamedTensor` with axes. Default normalization turns scalars into `(1, 1)` and vectors into `(N, 1)`; `normalize=False` preserves input shape. Names must match the final shape, so named vectors need `normalize=False` or two names. Existing storage/history survive when dtype and device permit; detach explicitly to disable an existing graph.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/transforms/named.py#L186).

## NamedTensor

Declared fields:

```python
axes: tuple[str, ...]
```

A Tensor subclass with one unique nonempty name per dimension. Inherits Tensor conversion and reduction methods. Named arithmetic aligns and broadcasts shared axes, then appends new right-hand names. Unnamed nonscalar operands are rejected. See the guide for flatten placement and operations that drop names.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/transforms/named.py#L35).

### NamedTensor.align

```python
NamedTensor.align(*axes: str | Sequence[str]) -> NamedTensor
```

Return exactly the requested permutation, with no added or omitted axes.

### NamedTensor.flatten

```python
NamedTensor.flatten(axes: Sequence[str], *, into: str) -> NamedTensor
```

Arrange selected axes in the supplied order, then merge them.

The merged axis occupies the earliest selected position in the current
order. Unselected axes retain their relative order.

### NamedTensor.transpose

```python
NamedTensor.transpose(axis1: int=-2, axis2: int=-1) -> NamedTensor
```

Swaps dimensions and their axis names, defaulting to the last two dimensions.

### NamedTensor.reorder

```python
NamedTensor.reorder(*axes: int) -> NamedTensor
```

Permutes dimensions and their names in the supplied integer order.

### NamedTensor.remove_dim

```python
NamedTensor.remove_dim(axis: int) -> NamedTensor
```

Removes a singleton dimension and its name without normalizing the result.

## PyTorch aliases

`mlzo.transforms` contains the same aliases and tensor helpers exported at the top level. Use native tensor inputs and `.raw` inside transformed functions or compiled regions; wrappers have no transparent torch dispatch.

| MLZO | PyTorch reference |
| --- | --- |
| `mlzo.grad` | [`torch.func.grad`](https://docs.pytorch.org/docs/stable/generated/torch.func.grad.html) |
| `mlzo.vmap` | [`torch.func.vmap`](https://docs.pytorch.org/docs/stable/generated/torch.func.vmap.html) |
| `mlzo.jacrev` | [`torch.func.jacrev`](https://docs.pytorch.org/docs/stable/generated/torch.func.jacrev.html) |
| `mlzo.jacfwd` | [`torch.func.jacfwd`](https://docs.pytorch.org/docs/stable/generated/torch.func.jacfwd.html) |
| `mlzo.hessian` | [`torch.func.hessian`](https://docs.pytorch.org/docs/stable/generated/torch.func.hessian.html) |
| `mlzo.jvp` | [`torch.func.jvp`](https://docs.pytorch.org/docs/stable/generated/torch.func.jvp.html) |
| `mlzo.vjp` | [`torch.func.vjp`](https://docs.pytorch.org/docs/stable/generated/torch.func.vjp.html) |
| `mlzo.functional_call` | [`torch.func.functional_call`](https://docs.pytorch.org/docs/stable/generated/torch.func.functional_call.html) |
| `mlzo.jit` | [`torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html) |
