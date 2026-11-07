---
title: Transforms and named tensors
description: Public MLZO 0.1.0a1 API signatures and behavior for transforms and named tensors.
---

Read the [transforms and named tensors guide](../../../guides/transforms/) for examples and usage constraints. Signatures and declared fields below come from MLZO `0.1.0a1` source. Type names follow their defining Python modules.

## tensor

```python
mlzo.tensor(
    values: Any,
    axes: Sequence[str] | None = None,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    requires_grad: bool | None = None,
) -> NamedTensor | torch.Tensor
```

Use torch.as_tensor and optionally attach names, preserving Tensor history.

With no axes this returns a torch.Tensor. Existing Tensor inputs are retained
when dtype/device agree. Disabling an existing gradient graph requires the
caller to detach explicitly; this helper never silently detaches it.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/transforms/named.py#L206).

## NamedTensor

Declared fields:

```python
raw: torch.Tensor
axes: tuple[str, ...]
```

An explicit axis-name wrapper over an ordinary tensor. See the guide for broadcasting, flatten placement, and unsupported operations.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/transforms/named.py#L34).

### NamedTensor.shape

```python
NamedTensor.shape: torch.Size  # read-only property
```

Returns the underlying tensor's shape in axis order.

### NamedTensor.ndim

```python
NamedTensor.ndim: int  # read-only property
```

Returns the number of named dimensions.

### NamedTensor.dtype

```python
NamedTensor.dtype: torch.dtype  # read-only property
```

Returns the underlying tensor's PyTorch dtype.

### NamedTensor.device

```python
NamedTensor.device: torch.device  # read-only property
```

Returns the underlying tensor's execution device.

### NamedTensor.requires_grad

```python
NamedTensor.requires_grad: bool  # read-only property
```

Reports whether autograd tracks the underlying tensor.

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

### NamedTensor.mean

```python
NamedTensor.mean(
    *axes: str | Sequence[str],
    keepdim: bool = False,
    dtype: torch.dtype | None = None,
) -> NamedTensor
```

Reduce named axes, or all axes if none are supplied.

### NamedTensor.to

```python
NamedTensor.to(*args: Any, **kwargs: Any) -> NamedTensor
```

Converts the underlying tensor using PyTorch semantics while retaining names.

### NamedTensor.clone

```python
NamedTensor.clone(**kwargs: Any) -> NamedTensor
```

Clones the underlying tensor and retains axis names.

### NamedTensor.detach

```python
NamedTensor.detach() -> NamedTensor
```

Detaches the underlying tensor from autograd and retains axis names.

## PyTorch aliases

`mlzo.transforms` contains the same aliases and named-tensor helpers exported at the top level.

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
