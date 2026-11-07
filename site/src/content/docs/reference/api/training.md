---
title: Training and runs
description: Public MLZO 0.1.0a1 API signatures and behavior for training and runs.
---

Read the [training and runs guide](../../../guides/training/) for examples and usage constraints. Signatures and declared fields below come from MLZO `0.1.0a1` source. Type names follow their defining Python modules.

## train

```python
mlzo.train(
    model_or_training: nn.Module | Training[Any] | Callable[[], Training[Any]],
    data: Any = None,
    *,
    loss: str | Callable[..., Any] | None = None,
    step: Callable[..., Any] | None = None,
    optimizer: Any = 'adamw',
    lr: float | None = None,
    schedule: Any = None,
    precision: Literal['auto', 'fp32', 'bf16', 'fp16'] = 'auto',
    matmul: Literal['auto', 'strict', 'fast'] = 'auto',
    grad_clip: float | None = None,
    accumulate: int = 1,
    compile: bool | Literal['auto'] = False,
    memory: Literal['auto'] | None = None,
    checkpoint: Any = None,
    device: str | None = None,
    seed: int | None = None,
) -> Run
```

Returns a deferred `Run`. Model inputs require data and exactly one of `loss` or `step`. Training instances and importable zero-argument training factories supply their own steps. See the [configuration reference](../../configuration/) for every option, default, and validation rule.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/training/run.py#L237).

## Run

```python
mlzo.Run(algorithm_input: Any, data: Any=None, config: Mapping[str, Any] | None=None)
```

Use `mlzo.train()` to create a configured run. `fit()` returns `History` locally or after a waited remote run; asynchronous remote submission returns the handle. Budgets are total targets. Remote resume and evaluation are unsupported.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/training/run.py#L63).

### Run.plan

```python
Run.plan(*, on: Any=None, parallel: Any='auto', example: Any=None) -> Any
```

Returns a metadata plan with `explain()` and `to_dict()`. It does not construct lazy models or consume data; remote inventory queries may contact the coordinator.

### Run.explain

```python
Run.explain() -> str
```

Returns the plan explanation with any execution notes available on a prepared run.

### Run.fit

```python
Run.fit(
    *,
    epochs: int | None = None,
    steps: int | None = None,
    on: Any = None,
    parallel: Any = 'auto',
    code: str | None = None,
    resume: Any = None,
    wait: bool = True,
) -> History | Run
```

Execute toward a total iteration/epoch target, or submit to a cluster.

Local execution is synchronous. `wait=False` is accepted only for a
cluster's durable submission. Steps count algorithm calls, not updates.

### Run.wait

```python
Run.wait() -> History
```

Returns successful terminal history. Raises `TrainingError` if no suitable run is attached or a remote run fails or is cancelled.

### Run.status

```python
Run.status() -> RunStatus
```

Returns `RunStatus` with state, ID, counters, and optional error information.

### Run.evaluate

```python
Run.evaluate(data: Any, *, step: Callable[..., Any] | None=None) -> dict[str, float]
```

Returns averaged scalar metrics from local evaluation. Requires an evaluator for custom algorithms and rejects remote evaluation.

### Run.checkpoint

```python
Run.checkpoint(*, to: Any=None) -> Any
```

Returns a full checkpoint reference at a safe local boundary or delegates to the completed remote run. A local destination or configured policy is required.

### Run.save

```python
Run.save(path: Any) -> Any
```

Alias for `checkpoint(to=path)`.

### Run.export

```python
Run.export(*, to: Any, block: str | None=None) -> Any
```

Exports immutable inference weights to a fresh local or assigned-worker directory, optionally selecting a registered block.

### Run.cancel

```python
Run.cancel() -> None
```

Requests cancellation of the attached remote job or an active local engine.

## Training

```python
mlzo.Training(model: nn.Module)
```

Subclass and implement `step`. Only `state` is checkpointed, so use tensors and supported primitive containers. Set `step_driven=True` for collection iterations receiving `None`.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/training/types.py#L73).

### Training.step

```python
Training.step(batch: BatchT) -> torch.Tensor | Step | None
```

Run one algorithm iteration, optionally requesting an automatic update.

### Training.iteration_source

```python
Training.iteration_source() -> Any
```

Return an iterable, or override with an epoch-aware data pipeline.

### Training.evaluate

```python
Training.evaluate(batch: BatchT) -> torch.Tensor | Step | None
```

Custom algorithms must implement a separate, non-mutating evaluator.

### Training.update

```python
Training.update(name: str, loss: torch.Tensor, *, retain_graph: bool=False) -> None
```

Complete a named optimizer update synchronously inside step().

## Step

Declared fields:

```python
loss: torch.Tensor | None = None
metrics: Mapping[str, Any] = field(default_factory=dict)
stop: bool | None = None
weight: float | torch.Tensor | None = None
```

`loss` is a scalar differentiable mean for automatic optimization. `metrics` maps names to scalar values. `stop=True` requests a whole-run stop. `weight` is the finite nonnegative normalization denominator; omitted weight means one.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/training/types.py#L60).

## History

```python
mlzo.History(run_id: str | None=None, *, max_records: int=1000)
```

Retains at most `max_records` recent detached scalar records. `records`, `dropped`, `counters`, `state`, and `notes` describe that bounded report.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/training/types.py#L139).

### History.append

```python
History.append(metrics: Mapping[str, Any], *, counters: Mapping[str, Any] | None=None) -> None
```

Appends one detached scalar metric record and optional iteration, epoch, and update-event counters. Old records are evicted at the configured bound.

### History.metrics

```python
History.metrics: dict[str, list[float]]  # read-only property
```

Returns retained metric values grouped by metric name.

### History.to_dict

```python
History.to_dict() -> dict[str, Any]
```

Returns a dictionary representation. Public invitation and cluster representations omit credentials.

### History.from_dict

```python
History.from_dict(value: Mapping[str, Any]) -> History
```

Reconstructs this value from its dictionary representation, applying its validation rules.

## RunStatus

Declared fields:

```python
state: str
id: str
counters: dict[str, Any] = field(default_factory=dict)
error: Any = None
```

A report containing `state`, `id`, `counters`, and `error`. Prefer `run.status()` over constructing reports manually.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/training/types.py#L129).

### RunStatus.to_dict

```python
RunStatus.to_dict() -> dict[str, Any]
```

Returns a dictionary representation. Public invitation and cluster representations omit credentials.
