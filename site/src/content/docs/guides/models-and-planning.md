---
title: Models and planning
description: Defer model construction, select a device, and read plans without executing training.
---

## Record a model factory

```python
import mlzo
from torch import nn

model = mlzo.Lazy(nn.Linear, 4, 2)
assert model.state == "unmaterialized"
```

Use an importable module-level function for a larger recipe:

```python title="my_project/models.py"
from torch import nn

def classifier():
    return nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 2))
```

```python
from my_project.models import classifier
model = mlzo.Lazy(classifier)
```

MLZO resolves nested lazy children and preserves aliases and parameter names. A failed lazy construction retains its failure; create a fresh recipe to retry. Eager `nn.Module` instances also work for local training, but their weights already exist and the instance cannot be submitted as a remote recipe.

## Materialize explicitly

```python
model = mlzo.Lazy(nn.Linear, 4, 2)
concrete = mlzo.materialize(model, "cpu", seed=7)
assert model.state == "ready"
```

Materialization calls each lazy factory once and moves the final module to the chosen device. Retaining the wrapper preserves access to the materialized parameters. Initialization is not repeated with `reset_parameters()`.

## Inspect a plan

```python
# `run` is a training program from the quickstart.
plan = run.plan(on="cpu", parallel="single")
print(plan.explain())
print(plan.to_dict())
```

Plans report the device, precision, single-device strategy, placements, known model metadata, data metadata, execution settings, decisions, limitations, and runtime checks. `validation_state` describes how much metadata could be checked. It is not a successful training result.

Planning never invokes model, training, or data factories, reads batches, runs a forward pass, or probes memory. It can inspect local device availability and contact a cluster to query worker inventory. `example=` records that an example was supplied but does not execute it. Unknown memory estimates remain unknown.

## Select devices and precision

Use `on="cpu"`, `on="cuda:0"`, or `on="mps"` at `fit()` time. Automatic selection prefers available CUDA devices, then MPS, then CPU. `train(device=...)` can also constrain the device; conflicting explicit choices raise `ConfigurationError`. A prepared run cannot change devices. Create a compatible new run and resume instead.

`precision="auto"` chooses BF16 only on CUDA that advertises support, and FP32 otherwise. CPU FP16 and MPS mixed precision are rejected. GPU and mixed-precision paths are implemented but unvalidated for this alpha. Use FP32 CPU examples for the verified path.

## Placement annotations

`mlzo.place(module, gpu=0)` records a logical GPU assignment without moving tensors. `gpus=[...]` annotates ordered direct children. Shared modules and tied tensors must have consistent placement. Multiple GPU assignments fail at planning or execution because this alpha runs on one device.

Logical GPU IDs belong to the execution target. Annotating a module does not enable pipeline parallelism. See the [configuration reference](../../reference/configuration/) for execution options.
