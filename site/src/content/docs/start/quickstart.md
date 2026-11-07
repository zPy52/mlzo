---
title: Your first training run
description: Train, evaluate, checkpoint, resume, and export a synthetic CPU classifier.
---

This example trains a two-class linear model on synthetic data. Run the snippets in order in one Python process after [installing MLZO](../installation/).

## Prepare the data

```python
import tempfile
from pathlib import Path

import mlzo
import torch
from torch import nn

generator = torch.Generator().manual_seed(7)
inputs = torch.randn(64, 4, generator=generator)
targets = (inputs.sum(dim=1) > 0).long()
batches = mlzo.data(inputs, targets).shuffle(seed=7).batch(16)
output = Path(tempfile.mkdtemp(prefix="mlzo-quickstart-"))
```

Each batch contains inputs of shape `[16, 4]` and integer targets of shape `[16, 1]`.
The supervised cross-entropy adapter handles those column labels. A fresh output
directory keeps these artifacts separate from earlier runs.

## Describe and inspect

```python
def make_run():
    return mlzo.train(
        mlzo.Lazy(nn.Linear, 4, 2), batches,
        loss="cross_entropy",
        optimizer=mlzo.opt("adamw", lr=0.03),
        seed=7,
    )

run = make_run()
plan = run.plan(on="cpu")
print(plan.explain())
assert run.status().state == "created"
```

The plan selects CPU, FP32, and single-device execution. Parameter and memory information that requires model construction remains unknown. Neither `train()` nor `plan()` invokes the `nn.Linear` factory.

## Fit and evaluate

```python
history = run.fit(epochs=2, on="cpu")
print(history.counters)
print(history.metrics["loss"][-1])
print(run.evaluate(batches))
assert run.status().state == "completed"
```

The run completes eight algorithm iterations across two epochs. With the default `accumulate=1`, each iteration requests one optimizer update. The history retains scalar records; evaluation returns averaged scalar metrics under evaluation mode without updating weights.

## Save and resume

```python
snapshot = run.checkpoint(to=output / "checkpoints")
resumed = make_run()
resumed.fit(epochs=3, on="cpu", resume=snapshot)
assert resumed.status().counters["epoch"] == 3
```

`epochs=3` is the total target, so this resumes for one more epoch. Checkpoints restore model, optimizer, scheduler, RNG, algorithm state, and counters. This example saves at an epoch boundary. A mid-epoch checkpoint restarts that data epoch rather than replaying an exact sampler position.

## Export and predict

```python
artifact = resumed.export(to=output / "classifier")
with mlzo.deploy(artifact, device="cpu") as predictor:
    logits = predictor.predict(inputs[:2])
    print(logits.argmax(dim=1))
    assert logits.shape == (2, 2)
```

The artifact stores inference weights and an importable `nn.Linear` construction recipe. It is separate from the resumable checkpoint. `deploy()` loads a synchronous predictor in this process; it does not start a server. See [checkpoints](../../guides/checkpoints/) and [inference artifacts](../../guides/inference/) for the storage contracts.
