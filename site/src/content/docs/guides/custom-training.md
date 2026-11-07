---
title: Custom algorithms
description: Implement Training, explicit optimizer updates, and restricted algorithm state.
---

Subclass `mlzo.Training` when the algorithm needs its own state, iteration source, or optimizer update sequence. Pass the model to `super().__init__()` and implement `step(batch)`.

## An explicit update

```python
import mlzo
import torch
from torch import nn
from torch.nn import functional as F

class ClassifierTraining(mlzo.Training):
    def __init__(self):
        super().__init__(mlzo.Lazy(nn.Linear, 4, 2))
        self.state = {"examples_seen": 0}

    def step(self, batch):
        inputs, targets = batch
        # Native cross entropy requires class indices shaped (B,).
        targets = targets.squeeze(1)
        loss = F.cross_entropy(self.model(inputs), targets)
        self.update("classifier", loss)
        self.state["examples_seen"] += len(inputs)
        return mlzo.Step(metrics={"loss": loss.detach()})

    def evaluate(self, batch):
        inputs, targets = batch
        targets = targets.squeeze(1)
        return mlzo.Step(loss=F.cross_entropy(self.model(inputs), targets),
                         weight=len(targets))

generator = torch.Generator().manual_seed(7)
inputs = torch.randn(32, 4, generator=generator)
targets = (inputs.sum(dim=1) > 0).long()
batches = mlzo.data(inputs, targets).batch(8)
algorithm = ClassifierTraining()
run = mlzo.train(algorithm, batches,
                 optimizer={"classifier": mlzo.opt("sgd", lr=0.1)}, seed=7)
run.fit(steps=8, on="cpu")
assert algorithm.state["examples_seen"] == 64
print(run.evaluate(batches))
```

`self.update(name, loss)` performs that named optimizer update immediately during the active step. Return metrics without a loss afterward. Returning a second optimizable loss would request an invalid automatic update after an explicit one. Explicit updates require `accumulate=1`.

## Automatic updates

For an automatic algorithm, return a scalar loss or `Step(loss=..., metrics=..., weight=...)` and let the engine handle backward and optimization. Return `None` or a metrics-only `Step` for collection iterations with no automatic update. `stop=True` stops the whole run.

Use disjoint parameter selectors for multiple optimizers. MLZO rejects overlapping ownership. The [optimizer guide](../optimizers/) describes naming and parameter groups.

## Own the iteration source

Supply `data` to `train()` for ordinary batch-based algorithms. Alternatively, override `iteration_source()` and return an iterable. Set `step_driven = True` to receive `None` for each collection iteration without a data source, then execute with a finite total `steps=` target.

The built-in [DQN recipe](../dqn/) uses step-driven execution. MLZO does not automatically close algorithm-specific resources; close environments or files in your own `finally` block.

## Store resumable state

Only `self.state` is persisted as algorithm state. Keep it to tensors, `None`, booleans, numbers, strings, lists, tuples, and dictionaries with supported keys. Do not store environments, closures, open files, or arbitrary objects there. Keep runtime objects in other attributes and reconstruct or reset them after restore as the algorithm requires.

## Submit a custom algorithm remotely

Define an importable zero-argument factory in a project module that returns the algorithm. Pass that factory to `mlzo.train()`. Passing an existing `Training` instance remotely is unsupported. Put factory code and dependencies in the project submitted with `code=`. See [submission](../../remote/submission/).
