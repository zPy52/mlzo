---
title: Optimizers and schedules
description: Bind optimizers after model construction and schedule successful updates.
---

## Defer optimizer binding

`mlzo.opt()` records an optimizer specification. MLZO binds it after the model has its final execution parameters.

```python
optimizer = mlzo.opt("adamw", lr=1e-3, weight_decay=0.01)
```

Built-in names are `adamw`, `adam`, `sgd`, and `rmsprop`. Their default learning rate is `3e-4`; AdamW defaults to weight decay `0.01`. Other defaults come from PyTorch. You can supply an optimizer class/factory or an existing optimizer. An existing optimizer must own the final model parameters and agree with any explicit `lr=`. Closure-based LBFGS updates are unsupported.

## Select parameters

| Selector | Meaning |
| --- | --- |
| `None` or `"*"` | All trainable model parameters |
| `"encoder"` | Trainable parameters in that submodule |
| `"head.weight"` | A parameter by its exact name |
| List of names | The union of those selections |
| Predicate `(name, parameter)` | Local parameter selection after construction |

```python
optimizers = {
    "encoder": mlzo.opt("adamw", params="encoder", lr=1e-4),
    "head": mlzo.opt("sgd", params="head", lr=1e-2),
}
```

This example assumes a model with `encoder` and `head` submodules and an algorithm that issues those named updates. Owners must be disjoint. Empty groups, duplicate selections, foreign parameters, and conflicting learning rates raise configuration errors.

Native parameter groups use a list of dictionaries with `params` selectors and ordinary optimizer options:

```python
optimizer = mlzo.opt("adamw", params=[
    {"params": "encoder", "lr": 1e-4},
    {"params": "head", "lr": 1e-3},
])
```

## Schedule successful updates

```python
schedule = mlzo.schedules.cosine(warmup=5, total_steps=100, min_factor=0.1)
```

Schedules multiply each group's base learning rate. Their index is the upcoming successful optimizer update, starting at one. Skipped updates do not advance it. Warmup uses the update index divided by `warmup`.

| Constructor | Behavior after warmup |
| --- | --- |
| `constant(factor=1.0, warmup=0)` | Fixed multiplier |
| `linear(warmup=0, total_steps=None, end_factor=0.0)` | Linear decay toward the end factor |
| `cosine(warmup=0, total_steps=None, min_factor=0.0)` | Cosine decay toward the minimum factor |
| `exponential(gamma=0.99, warmup=0)` | Geometric decay per successful update |

For linear and cosine schedules, supply `total_steps` explicitly when the automatic-update horizon is unknown. A known supervised horizon can be inferred from the fit budget and accumulation. For unknown-length data with accumulation greater than one, a step limit cannot determine the number of partial accumulations flushed at exhausted passes; supply `total_steps` explicitly. Extending a run can change that inferred horizon and is rejected rather than silently reconfiguring decay. Compatible resume preserves the original schedule definition and horizon.

Custom local multipliers accept an upcoming update index, optionally with the horizon. A native scheduler factory must take its bound optimizer as the first argument named `optimizer` or `opt`. Metric-driven `ReduceLROnPlateau` is unsupported. Multiple optimizer schedules can be configured by name. See [configuration](../../reference/configuration/) and the [schedule API](../../reference/api/optimization/).
