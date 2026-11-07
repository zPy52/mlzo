"""PyTorch function transforms and a small optional named-axis tensor wrapper.

Transforms are direct PyTorch exports, so positional arguments, keyword
arguments, return values, and backend errors retain their PyTorch semantics.
In particular, jit is torch.compile, including its decorator form and options.
"""

from torch import compile as jit
from torch.func import functional_call, grad, hessian, jacfwd, jacrev, jvp, vjp, vmap

from .named import NamedTensor, tensor

__all__ = [
    "grad",
    "vmap",
    "jacrev",
    "jacfwd",
    "hessian",
    "jvp",
    "vjp",
    "functional_call",
    "jit",
    "tensor",
    "NamedTensor",
]
