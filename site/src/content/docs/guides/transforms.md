---
title: Transforms and named tensors
description: Use PyTorch function transforms and explicit axis-aware tensor operations.
---

## Use PyTorch transforms

MLZO directly re-exports `torch.func.grad`, `vmap`, `jacrev`, `jacfwd`, `hessian`, `jvp`, `vjp`, and `functional_call`. `mlzo.jit` is `torch.compile`. Their arguments, errors, and return values keep their PyTorch semantics.

```python
import mlzo
import torch

gradient = mlzo.grad(lambda x: x.square().sum())
batch_gradient = mlzo.vmap(gradient, randomness="error")
values = torch.arange(12, dtype=torch.float32).reshape(3, 4)
torch.testing.assert_close(batch_gradient(values), 2 * values)
```

MLZO provides no separate differentiation or compilation engine. Consult the [PyTorch torch.func reference](https://docs.pytorch.org/docs/stable/func.html) for transform semantics and restrictions.

## Attach axis names

```python
x = mlzo.tensor(torch.randn(2, 3, 5, 4), axes=("batch", "heads", "time", "dim"))
y = x.align("batch", "time", "heads", "dim")
y = y.flatten(("heads", "dim"), into="features").mean("time")
assert y.axes == ("batch", "features")
assert y.raw.shape == (2, 12)
```

Without `axes`, `tensor(values)` returns a plain tensor using `torch.as_tensor`. With axes it returns a `NamedTensor` wrapper with one unique nonempty name per dimension. `.raw` exposes the ordinary tensor. Names are immutable; tensor data remains mutable.

`align(*axes)` requires an exact permutation of existing names. `flatten(axes, into=...)` orders the selected dimensions as requested and merges them at their earliest selected position. Other axes retain their relative order. `mean(*axes, keepdim=False, dtype=None)` reduces named dimensions; no axes means all dimensions. `keepdim=True` keeps reduced names on singleton dimensions.

## Arithmetic and autograd

Addition, subtraction, multiplication, division, floor division, power, and negation use ordinary tensor operations. Between named tensors, shared names align and broadcast if their sizes match or one is one. The result lists the left axes followed by right axes absent on the left. Disjoint names produce outer broadcasting.

Numbers and scalar raw tensors broadcast. Nonscalar raw tensors are rejected because their names are unknown. Use `.raw` for indexing, matrix multiplication, comparisons, and unsupported operations, then reattach checked names if useful.

These operations preserve autograd. `requires_grad=False` does not detach an existing graph; call `.detach()` explicitly. The wrapper is not registered as a function-transform pytree, so transformed functions should operate on raw tensors. `.to()`, `.clone()`, and `.detach()` preserve checked axis metadata. See the [transform reference](../../reference/api/transforms/).
