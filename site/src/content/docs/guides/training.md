---
title: Supervised training
description: Choose losses, control accumulation, read metrics, and evaluate a local run.
---

## Use a built-in loss

Pass an `nn.Module` or `Lazy` model, a batch source, and one loss or step function:

```python
run = mlzo.train(model, batches, loss="cross_entropy",
                 optimizer=mlzo.opt("adamw", lr=1e-3),
                 accumulate=4, precision="fp32", seed=7)
history = run.fit(epochs=3, on="cpu")
```

This snippet uses `model` and `batches` prepared as in the [quickstart](../../start/quickstart/). The built-in adapter calls `model(inputs)` for a two-element tuple or list `(inputs, targets)`.

| Name | PyTorch loss |
| --- | --- |
| `cross_entropy` | Cross entropy for classification |
| `mse` | Mean squared error |
| `l1` | Mean absolute error |
| `bce` | Binary cross entropy for probabilities |
| `bce_with_logits`, `bce_logits` | Binary cross entropy with logits |

You can pass a callable `loss(prediction, target)`. Supported native PyTorch loss instances preserve their mean or sum reduction. A loss with `reduction="none"` needs a custom step that reduces it to a scalar.

## Write a function step

Use `step=` for another batch structure or model invocation. Do not pass `loss=` at the same time.

```python
import torch.nn.functional as F

def step(model, batch):
    prediction = model(batch["x"])
    loss = F.cross_entropy(prediction, batch["y"])
    return mlzo.Step(loss=loss, weight=len(batch["y"]),
                     metrics={"accuracy": (prediction.argmax(-1) == batch["y"]).float().mean()})

run = mlzo.train(model, record_batches, step=step, accumulate=4)
```

Return a differentiable scalar tensor or `Step(loss=...)` for automatic optimization. Metrics must be numbers or real scalar tensors. MLZO detaches reported metrics.

## Accumulate with the right denominator

A plain scalar loss contributes weight one. For means over different sample or token counts, return `Step(loss=mean_loss, weight=count)`. The engine backpropagates each weighted numerator and normalizes the accumulation window by the total weight. This avoids giving a small final batch the same influence as a full batch.

Built-in mean losses compute their actual PyTorch denominator, including ignored labels and supported class weights. A zero-weight contribution does not update the optimizer. Nonfinite losses or gradients skip updates; skipped updates do not advance the learning-rate schedule. A partial accumulation window flushes at a completed fit or epoch boundary.

`weight` must be finite, nonnegative, and detached when it is a tensor. Explicit named updates require `accumulate=1`.

## Read status and history

```python
print(run.status().to_dict())
print(history.counters)
print(history.metrics)
```

Status includes `state`, `id`, `counters`, and `error`. Counters distinguish `iteration`, `epoch`, `batch_in_epoch`, per-optimizer `optimizer_steps`, `skipped_updates`, and `update_events`.

History retains at most 1,000 recent scalar records by default, independent of run duration. `dropped` counts discarded older records. `metrics` groups retained values by name, and `to_dict()` serializes the bounded report. History is not a model checkpoint or an unlimited metrics log.

## Evaluate

```python
metrics = run.evaluate(validation_batches)
```

Local evaluation prepares the model if necessary, moves batches to the execution device, disables gradients, and restores the original per-module train/eval modes afterward. Loss aggregation uses its normalization weight; other scalar metrics average across reported batches. Use `run.evaluate(data, step=my_evaluator)` for custom evaluation. Custom `Training` algorithms need an `evaluate()` override or this explicit evaluator.

## Failures and cancellation

A failed run cannot continue with an ordinary `fit()` call. Create a fresh run or resume a committed checkpoint. Already completed optimizer updates cannot be rolled back. MLZO clears pending gradients after a failed step, but that does not restore earlier parameter values.

Local execution is synchronous and does not support `wait=False` or concurrent MLZO execution in one process. `Step(stop=True)` requests a whole-run stop. Remote runs support cancellation and durable reconnection; see [remote operations](../../remote/operations/).
