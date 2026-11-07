---
title: Core concepts
description: Understand recipes, plans, algorithm steps, update events, and persisted artifacts.
---

MLZO coordinates ordinary PyTorch computation. You describe the program first and execute it through a `Run`.

## Recipes and materialization

`Lazy(factory, *args, **kwargs)` is an `nn.Module` wrapper that remembers how to construct a module. It allocates the module when MLZO materializes it for execution. Calling an unresolved wrapper's `forward()` raises an error.

Only construction inside the factory is deferred. `Lazy(MyModel, already_created_encoder)` cannot undo allocations made to build the encoder. Put the whole model in a factory when you need to defer it all.

Local factories may use ordinary Python callables. Remote code and stored inference recipes need named, importable factories, so workers and artifact loaders can reconstruct the module without pickling Python objects.

## The run lifecycle

| Operation | Effect |
| --- | --- |
| `train(...)` | Records the algorithm, data, and configuration |
| `run.plan(...)` | Inspects metadata and target inventory without executing the model or data |
| `run.fit(...)` | Prepares and executes locally, or submits a remote job |
| `run.status()` | Reports state, counters, and an error if present |
| `run.evaluate(...)` | Evaluates locally with gradients disabled |
| `run.checkpoint(...)` | Saves resumable training state |
| `run.export(...)` | Saves an inference artifact |

Local `fit()` blocks until completion. Remote `fit(wait=False)` returns the run handle after durable submission, and `wait()` waits for the terminal result. Reconnect to a remote run with `mlzo.run(id, on=cluster)`.

## Iterations and optimizer updates

An iteration is one call to the algorithm's `step()`. An optimizer update changes one optimizer's parameters. They are separate counters.

For ordinary supervised training with `accumulate=4`, four algorithm iterations usually produce one optimizer update. An explicit algorithm can issue multiple named updates in one iteration or collect data without updating anything. Schedules advance on successful optimizer updates. Checkpoint intervals count update events and save at a safe boundary.

`fit(steps=100)` and `fit(epochs=3)` specify total targets. Choose one. A second local call with a larger compatible target continues the run rather than adding that many new steps. Schedules with an inferred horizon can reject a changed target; specify an explicit schedule horizon when needed.

## Checkpoints and inference artifacts

A checkpoint contains the state required to resume training, including optimizer and RNG state. An inference artifact contains model weights and, where possible, a construction recipe. You cannot substitute one for the other.

Both formats restrict saved data to tensors and supported primitive containers. An importable factory is executable Python even when the weights file is restricted; load artifacts only with code you trust.

## Remote means one assigned worker

A coordinator accepts submissions and tracks workers. One worker executes each run on one device. Multiple enrolled workers do not make one run distributed. The alpha has no process group, distributed gradient exchange, automatic partitioning, or managed cloud provisioning.

The [support matrix](../../help/support/) distinguishes implemented behavior from the future design in the repository's `plan.md`.
