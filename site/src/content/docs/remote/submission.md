---
title: Submit and reconnect
description: Package importable code for one worker, submit asynchronously, and reconnect by run ID.
---

## Run the included project

After [setting up a coordinator and worker](../setup/), run from the MLZO repository root:

```sh
python examples/remote_project/submit.py --cluster demo --wait
```

The project contains an importable model constructor, a zero-argument synthetic data factory, and a guarded submission script. It declares no extra project dependencies and downloads no dataset. Inspect its [source](https://github.com/zPy52/mlzo/tree/main/examples/remote_project) for a complete project layout.

## Submit your own run

```python
import mlzo
from my_project.models import classifier
from my_project.data import training_data

cluster = mlzo.cluster("demo")
run = mlzo.train(mlzo.Lazy(classifier), mlzo.data(training_data).batch(16),
                 loss="cross_entropy", seed=7)
run.fit(epochs=2, on=cluster, code="path/to/project", wait=False)
print(run.id)
```

Here `my_project` must be importable locally and inside the submitted project directory. `code=` names a directory. MLZO snapshots source and transportable configuration; it does not pickle a live model or interpreter. Save the run ID to reconnect later.

The default `wait=True` waits for completion and returns `History`. Remote `wait=False` returns the submitted `Run`. Local `wait=False` is unsupported.

## Reconnect from another process

```python
import mlzo

attached = mlzo.run(saved_run_id, on=mlzo.cluster("demo"))
print(attached.status().to_dict())
history = attached.wait()
```

`saved_run_id` is the identifier printed after submission. Reconnecting uses saved client configuration and the coordinator's durable record. It does not resubmit the program. `wait()` returns history on success and raises `TrainingError` for failed or cancelled terminal jobs.

For a bounded control-client wait, use `cluster.wait(run_id, timeout=..., poll_interval=...)`. `cluster.events(run_id, after=cursor)` reads scalar events after a sequence cursor.

## What can travel

| Value | Remote requirement |
| --- | --- |
| Model | `Lazy` with an importable factory and transportable arguments |
| Custom algorithm | Importable zero-argument training factory |
| Data | Importable factory or supported pipeline/file recipe |
| Functions | Named importable definitions, not closures or lambdas |
| Dependencies | Available in the worker job environment |

An existing eager model, custom `Training` instance, arbitrary native iterator, or arbitrary pickle object cannot be submitted. Importable factories cannot live in `__main__`, a closure, or a notebook cell without a project module.

## Source and data staging

Source archives exclude common credentials, virtual environments, caches, and output paths, and have a 100 MiB bound. Keep the project directory focused. Explicit local pipeline file sources can be staged and rewritten. Files referenced only inside a factory are invisible to that inspection; your factory must use worker-accessible paths.

The worker prepares a virtual environment and uses its installed MLZO/PyTorch. Project dependencies can need installation and network access. The mechanism is not a hostile-code sandbox.

## Completion and artifacts

Completed remote runs retain worker-local checkpoint information. Checkpoint and export commands do not download artifacts. Keep the assigned worker's service and persistent job directory available for later export commands. Remote evaluation, resume, and extending an existing remote submission are unsupported. See [operations](../operations/).
