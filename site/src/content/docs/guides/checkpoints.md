---
title: Checkpoints and resume
description: Save verified full-state snapshots and understand epoch-restart recovery.
---

## Configure periodic saves

```python
policy = mlzo.checkpoint("outputs/run-01/checkpoints", every=100, keep_last=3)
run = mlzo.train(model, batches, loss="cross_entropy", checkpoint=policy)
run.fit(epochs=3, on="cpu")
```

This uses a model and batches prepared as in the quickstart. `every` counts successful optimizer update events. The engine saves at safe boundaries with no pending accumulated gradients. Intervals are not wall-clock times or algorithm-iteration counts.

Retention applies to the entire root. Use a separate root per run. `keep_last=None` disables retention. Only `mode="full"` is supported; S3, per-block, sharded, and distributed checkpoints raise explicit errors.

## Save manually

```python
snapshot = run.checkpoint(to="outputs/run-01/checkpoints")
print(snapshot.to_dict())
snapshots = mlzo.list_snapshots("outputs/run-01/checkpoints")
```

`run.save(path)` is an alias for `run.checkpoint(to=path)`. Prepare the run with `fit()` or `evaluate()` first. Concurrent local manual saves and snapshots of failed iterations are rejected. Use a configured policy to save during training.

## What a checkpoint contains

Full snapshots contain model weights, optimizer and scheduler state, gradient scaler state, algorithm `state`, Python/PyTorch and available NumPy/device RNG state, counters, parameter ownership metadata, precision, and data resume metadata. Saved state uses restricted tensors and primitive containers, without an executable Python model object.

Storage writes checksummed state and publishes a committed manifest atomically. Loading verifies the expected files and integrity before restricted deserialization. Checkpoint roots need filesystem support for the storage's locking and atomic publication operations.

## Resume a compatible run

```python
new_run = mlzo.train(mlzo.Lazy(nn.Linear, 4, 2), batches,
                     loss="cross_entropy", checkpoint=policy)
new_run.fit(epochs=4, on="cpu", resume=snapshot)
```

Keep the model structure, algorithm, optimizer types and parameter ownership, precision, and schedule definition compatible. The example assumes the original default optimizer and precision. `resume=` also accepts a checkpoint directory. `resume="latest"` requires a configured policy root.

The budget is a total target. A checkpoint at epoch three followed by `epochs=4` trains toward epoch four. This is an explicit restore, not automatic crash recovery.

## Data resume guarantee

Reiterable sources restart a partially completed epoch. The checkpoint restores algorithm and optimizer state but does not skip back to the exact sample or prefetch cursor. Samples already processed in the partial epoch can be seen again.

One-shot iterators cannot resume. Exact native sampler/worker replay is unavailable. DQN has an additional partial environment reset contract, documented in the [DQN guide](../dqn/).

## Inspect snapshots outside training

`save_snapshot()`, `load_snapshot()`, and `list_snapshots()` expose the low-level local storage API. `load_snapshot()` returns stored state; it does not reconstruct a training program. See the [checkpoint reference](../../reference/api/checkpoints/).

Remote manual checkpoint commands require a completed run and return worker-local references. They do not download files. Remote resume is unsupported. For predictions, use a separate [inference export](../inference/).
