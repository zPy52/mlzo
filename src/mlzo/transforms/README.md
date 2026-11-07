# Tensors and function transforms

`ml.tensor(values)` returns `ml.Tensor`, a lightweight wrapper around a native
PyTorch tensor. With `axes=...` it returns `ml.NamedTensor`, a `Tensor` subclass.
Conversion uses `torch.as_tensor`, so existing storage and autograd history are
preserved when dtype and device permit. Dtype/device conversions follow PyTorch
copy rules. `requires_grad=False` rejects an existing gradient graph; detach it
explicitly first.

## Create tensors and control shapes

```python
import mlzo as ml
import torch
from torch import nn

assert ml.tensor(5).shape == (1, 1)
x = ml.tensor([1., 2., 3.])
assert x.shape == (3, 1)
assert ml.tensor([1., 2., 3.], normalize=False).shape == (3,)

native = torch.randn(2, 3, requires_grad=True)
wrapped = ml.wrap(native)
assert wrapped.raw is native
assert ml.wrap(wrapped) is wrapped
(wrapped * 2).sum().backward()
torch.testing.assert_close(native.grad, torch.full_like(native, 2))
```

Default construction turns rank-zero inputs into `(1, 1)` and rank-one inputs
into `(N, 1)`. Higher ranks keep their shape. `normalize=False` disables this
construction rule. `ml.wrap(native)` always preserves native shape and identity,
without conversion, copying, or detachment. `.raw` exposes the native tensor.

| Operation | MLZO spelling and behavior |
| --- | --- |
| Dtype conversion | `x.astype("float32")` or `x.astype(torch.float32)` |
| Device transfer | `x.on("cpu")`, or `x.on("cuda")` on a supported installation |
| Insert singleton dimension | `x.add_dim(axis)` |
| Remove singleton dimension | `x.remove_dim(axis)`, error unless its size is one |
| Swap dimensions | `x.transpose(a, b)`, defaults to the last two |
| Permute dimensions | `x.reorder(*axes)` |
| Reshape or flatten | `x.reshape(*shape)`, `x.flatten(start_dim=0, end_dim=-1)` |
| Reduce | `x.mean(axis=...)`, `x.sum(axis=...)`, both default to `keepdims=True` |

`to`, `clone`, `detach`, `backward`, arithmetic, and positional `matmul` are also
available. The wrapper deliberately exposes a limited set of native methods.
Use `.raw` for other methods, comparisons, arbitrary `torch.*` functions, and
in-place methods such as `add_`. Indexed assignment and augmented arithmetic
mutate native storage and obey PyTorch's casting and autograd restrictions.

## Reduce and index

```python
matrix = ml.tensor([[1., 2.], [3., 4.]])
assert matrix.mean(axis=1).shape == (2, 1)
assert matrix.sum(axis=0).shape == (1, 2)
assert matrix.mean().shape == (1, 1)
assert matrix.mean(axis=1, keepdims=False).shape == (2,)
assert matrix[:, 0].shape == (2, 1)
assert matrix[0].shape == (1, 2)
assert matrix[0, 0].shape == (1, 1)
assert matrix.reshape(4).shape == (4,)
assert matrix.flatten().shape == (4,)
assert x.remove_dim(1).shape == (3,)
```

Reductions accept integer axes, negative axes, and tuples, with `dim` and
`keepdim` as native spelling aliases. Omitting axes reduces all dimensions.
Explicit `reshape`, `flatten`, `remove_dim`, and `keepdims=False` results stay
at the requested rank; MLZO does not normalize every operation's output.

For a wholly basic index, integers become bounds-checked singleton slices.
Slices, ellipses, and `None` follow native rules. If any part of an index is
advanced or boolean, the **whole index** uses native semantics, including its
integer components. These results keep their native shape without normalization.
Use `matrix.raw[0]` for native dimension-dropping integer indexing.

```python
assert matrix[[0, 1], 0].shape == (2,)
mask = torch.tensor([[True, False], [False, True]])
assert matrix[mask].shape == (2,)
```

## Attach and preserve axis names

```python
named = ml.tensor(torch.randn(2, 3, 5, 4),
                  axes=("batch", "heads", "time", "dim"))
arranged = named.align("batch", "time", "heads", "dim")
summary = arranged.flatten(("heads", "dim"), into="features").mean("time")
assert summary.axes == ("batch", "time", "features")
assert summary.shape == (2, 1, 12)
assert summary.remove_dim(1).axes == ("batch", "features")

# Names describe the final shape, after any construction normalization.
vector = ml.tensor([1., 2., 3.], axes=("features",), normalize=False)
column = ml.tensor([1., 2., 3.], axes=("sample", "value"))
assert vector.shape == (3,)
assert column.shape == (3, 1)
```

Supply one unique nonempty name per resulting dimension. A named rank-one
input therefore needs `normalize=False`, or two names for the normalized
column. `align(*axes)` accepts exactly a permutation of current names.
`flatten(axes, into=...)` orders the selected dimensions as requested and merges
them at their earliest selected position. Other dimensions keep their relative
order. Align first to choose another placement.

Named arithmetic aligns shared names and broadcasts sizes that match or are one.
The result lists left names followed by new right names. Disjoint names produce
outer broadcasting. Numbers and unnamed rank-zero tensors broadcast; unnamed
nonscalar tensors are rejected because their axis meanings are unknown.

`to`, `astype`, `on`, `clone`, `detach`, negation, `transpose`, `reorder`, and
singleton removal retain the corresponding names. Reductions keep reduced names
by default; `keepdims=False` removes them. Basic indexing retains names unless
it inserts a new axis. `add_dim`, `reshape`, advanced/boolean indexing, and
positional matrix multiplication return an unnamed `Tensor`. Reattach names
explicitly after these operations.

## Use native PyTorch boundaries

Native `nn.Module`, `torch.optim`, `torch.func`, and compiled regions use native
tensors. Pass `.raw` explicitly when using them directly. These wrappers do not
implement transparent torch dispatch and are not registered as transform pytrees.
MLZO training and deployment unwrap tensor leaves at their own input boundaries
without reshaping them. Training also transfers batches for `fit(on=...)`.

```python
linear = nn.Linear(2, 1)
prediction = linear(matrix.raw)
assert prediction.shape == (2, 1)

gradient = ml.grad(lambda value: value.square().sum())
batch_gradient = ml.vmap(gradient, randomness="error")
values = torch.arange(12, dtype=torch.float32).reshape(3, 4)
torch.testing.assert_close(batch_gradient(ml.wrap(values).raw), 2 * values)
```

`grad`, `vmap`, `jacrev`, `jacfwd`, `hessian`, `jvp`, `vjp`, and `functional_call`
are direct `torch.func` exports. `ml.jit` is `torch.compile`; arguments, return
values, errors, and backend restrictions retain their native meaning.

## Migrate from the previous native default

Code that previously received a native tensor without `axes` now receives
`Tensor`. Use `normalize=False` to keep an old vector shape and `.raw` at a
native API call. Named reductions now preserve dimensions by default.

```python
# Previously: native_vector = ml.tensor([1., 2., 3.])
native_vector = ml.tensor([1., 2., 3.], normalize=False).raw
assert native_vector.shape == (3,)

# Previously: named.mean("time") dropped the time axis.
old_shape = named.mean("time", keepdims=False)
assert old_shape.axes == ("batch", "heads", "dim")
```
