# Function transforms and named tensors

`grad`, `vmap`, `jacrev`, `jacfwd`, `hessian`, `jvp`, `vjp`, and `functional_call`
are direct exports from `torch.func`. `jit` is `torch.compile`, including its
decorator form. All backend arguments, keyword options, return values, and
errors retain their PyTorch meaning. No custom differentiation or compilation
engine is provided.

```python
from mlzo.transforms import grad, vmap, tensor

gradient = grad(lambda x: x.square().sum())
batch_gradient = vmap(gradient, randomness="error")
x = tensor(values, axes=("batch", "heads", "time", "dim"))
y = x.align("batch", "time", "heads", "dim")
y = y.flatten(("heads", "dim"), into="features").mean("time")
```

`tensor(values)` returns a plain PyTorch tensor using `torch.as_tensor`.
Specifying `axes` wraps it in `NamedTensor`, with one unique nonempty name per
dimension. An existing tensor retains its autograd graph. Dtype/device changes
use ordinary PyTorch conversion semantics. To disable an existing graph,
detach explicitly; `requires_grad=False` will not silently detach it.

`NamedTensor.raw` exposes the underlying tensor. Supported methods are `align`,
`flatten`, `mean`, `to`, `clone`, and `detach`. Axis metadata is immutable, but
the tensor remains an ordinary mutable PyTorch tensor. This is an explicit
wrapper, without automatic dispatch for arbitrary `torch.*` operations.

`align(*axes)` accepts exactly a permutation of all current names. A single
tuple/list of names is also accepted. `flatten(axes, into=...)` first arranges
selected dimensions in the supplied order, then merges them at the earliest
selected position. Other dimensions keep their relative order. For example,
flattening `("heads", "dim")` in `("batch", "heads", "time", "dim")` produces
`("batch", "features", "time")`. Align first if a different placement is wanted.

`mean(*axes, keepdim=False, dtype=None)` reduces named dimensions. No axis
arguments means all dimensions. `keepdim=True` retains the reduced names on
singleton dimensions.

Arithmetic supports addition, subtraction, multiplication, true/floor division,
power, and unary negation. Between named tensors the result order is all left
axis names followed by right names absent on the left. Shared names align even
if their original positions differ. Their sizes must agree or one must be 1;
missing names insert singleton dimensions. Disjoint names therefore produce
an outer broadcast. Numbers and scalar raw tensors broadcast normally. A raw
nonscalar tensor raises an error because its axis names are unknown.

Use `.raw` for indexing, reshaping, comparisons, matrix multiplication, and
other unsupported operations, then reattach checked names explicitly if needed.
Named arithmetic and rearrangement use ordinary tensor operations and preserve
autograd. Function transforms operate on functions of raw tensors; this wrapper
is not registered as a transform pytree.
