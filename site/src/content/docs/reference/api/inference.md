---
title: Inference artifacts
description: Public MLZO 0.1.0a1 API signatures and behavior for inference artifacts.
---

Read the [inference artifacts guide](../../../guides/inference/) for examples and usage constraints. Signatures and declared fields below come from MLZO `0.1.0a1` source. Type names follow their defining Python modules.

## ArtifactRef

Declared fields:

```python
path: str | os.PathLike[str]
metadata: dict[str, Any] = field(default_factory=dict)
```

A local immutable inference artifact reference. It is distinct from a resumable training checkpoint.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/inference/artifact.py#L67).

### ArtifactRef.artifact_id

```python
ArtifactRef.artifact_id: str | None  # read-only property
```

Returns the identity recorded in the artifact manifest.

### ArtifactRef.to_dict

```python
ArtifactRef.to_dict() -> dict[str, Any]
```

Returns a dictionary representation. Public invitation and cluster representations omit credentials.

### ArtifactRef.from_dict

```python
ArtifactRef.from_dict(value: Mapping[str, Any]) -> ArtifactRef
```

Reconstructs this value from its dictionary representation, applying its validation rules.

## export_model

```python
mlzo.export_model(
    model: nn.Module,
    *,
    to: Any,
    recipe: Any = None,
    block: str | None = None,
) -> ArtifactRef
```

Export trained state to an immutable local inference directory.

The destination names the artifact itself. An identical existing artifact
can be reused; a different or invalid destination is never overwritten.
Block selection uses only that block's recipe, ignoring the root recipe.
A known recipe that is non-importable or has unsupported arguments fails
explicitly. An eager model without a recipe is labelled weights_only and
needs a zero-argument factory at load time. Export never constructs Lazy
modules, changes training mode, or serializes optimizer state.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/inference/artifact.py#L281).

## load_artifact

```python
mlzo.load_artifact(
    reference: Any,
    *,
    device: str | torch.device = 'cpu',
    factory: Any = None,
) -> nn.Module
```

Verify an inference artifact and return its materialized model in eval mode.

`factory` overrides a stored recipe and is called with no arguments. It
can return a Lazy model; materialize resolves it on the requested device.
Stored recipes import trusted user code, but weights always use the
restricted `torch.load(weights_only=True)` loader with strict model keys.
Training checkpoints and incomplete export directories are rejected.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/inference/artifact.py#L404).

## Deployment

```python
mlzo.Deployment(artifact: Any, *, device: str | torch.device='cpu', factory: Any=None)
```

A synchronous predictor in the current process. Use `deploy()` or its context manager to load and release the model.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/inference/deployment.py#L47).

### Deployment.device

```python
Deployment.device: str  # read-only property
```

Returns the device used by this local predictor.

### Deployment.predict

```python
Deployment.predict(*args: Any, **kwargs: Any) -> Any
```

Unwraps nested MLZO tensor leaves without reshaping, moves native tensors to the configured device, and predicts in eval mode under `torch.inference_mode()`. Raises after shutdown.

### Deployment.status

```python
Deployment.status() -> dict[str, Any]
```

Reports local readiness and artifact identity.

### Deployment.shutdown

```python
Deployment.shutdown() -> None
```

Releases the model. Repeated shutdown is safe.

## deploy

```python
mlzo.deploy(
    artifact: Any,
    *,
    on: Any = None,
    device: str | torch.device | None = None,
    factory: Any = None,
) -> Deployment
```

Load a local deployment, using CPU unless device is explicitly supplied.

`on` is reserved for remote placement. Any non-None target, including a
cluster handle, raises UnsupportedFeatureError in this alpha.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/inference/deployment.py#L107).
