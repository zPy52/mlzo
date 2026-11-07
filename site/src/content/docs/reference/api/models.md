---
title: Models and factories
description: Public MLZO 0.1.0a1 API signatures and behavior for models and factories.
---

Read the [models and factories guide](../../../guides/models-and-planning/) for examples and usage constraints. Signatures and declared fields below come from MLZO `0.1.0a1` source. Type names follow their defining Python modules.

## Lazy

```python
mlzo.Lazy(factory: Callable[..., nn.Module], *args: Any, gpu: int | None=None, **kwargs: Any)
```

An `nn.Module` wrapper that records a factory and its arguments. `state` is unmaterialized, materializing, ready, or failed. Forward never triggers construction. A recorded `gpu` value is an annotation, not a tensor move.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/model/lazy.py#L21).

### Lazy.state

```python
Lazy.state: LazyState  # read-only property
```

Returns the construction lifecycle state.

### Lazy.error

```python
Lazy.error: BaseException | None  # read-only property
```

Returns the retained construction exception, or `None`.

### Lazy.materialize

```python
Lazy.materialize(device: str | torch.device='cpu') -> nn.Module
```

Materializes the recipe once on the requested device and returns the concrete module.

### Lazy.forward

```python
Lazy.forward(*args: Any, **kwargs: Any) -> Any
```

Delegates to the concrete module after explicit materialization; raises `TrainingError` while unresolved.

### Lazy.train

```python
Lazy.train(mode: bool=True) -> Lazy
```

Sets the train/eval mode using ordinary PyTorch module semantics and returns this wrapper.

### Lazy.state_dict

```python
Lazy.state_dict(*args: Any, **kwargs: Any) -> Any
```

Returns the materialized module's canonical state dictionary when ready, preserving parameter names.

### Lazy.extra_repr

```python
Lazy.extra_repr() -> str
```

Returns PyTorch's extra representation of the stored factory, arguments, state, and GPU annotation.

## materialize

```python
mlzo.materialize(module: nn.Module, device: str | torch.device, seed: int | None=None) -> nn.Module
```

Resolve lazy roots/children once, retain aliases, and move to one device.

Factories run in the destination-device context where PyTorch supports it.
Explicit CPU allocations in a factory are moved afterwards; initialization
is never replaced with reset_parameters() or a second factory invocation.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/model/lazy.py#L167).

## place

```python
mlzo.place(module: nn.Module, *, gpu: int | None=None, gpus: Iterable[int] | None=None) -> nn.Module
```

Annotate one module or its ordered direct children with logical GPU IDs.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/model/placement.py#L32).

## FactoryRef

Declared fields:

```python
module: str
qualname: str
```

Names a trusted Python callable by importable module and qualified name. Rejects `__main__`, closures, and references that do not resolve consistently.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/model/factory.py#L15).

### FactoryRef.from_callable

```python
FactoryRef.from_callable(value: Callable[..., Any]) -> FactoryRef
```

Validates that a named callable can be reconstructed by its importable reference.

### FactoryRef.resolve

```python
FactoryRef.resolve() -> Callable[..., Any]
```

Imports the module and returns the named callable. This executes trusted Python module code.

### FactoryRef.to_dict

```python
FactoryRef.to_dict() -> dict[str, str]
```

Returns a dictionary representation. Public invitation and cluster representations omit credentials.

### FactoryRef.from_dict

```python
FactoryRef.from_dict(value: Mapping[str, Any]) -> FactoryRef
```

Reconstructs this value from its dictionary representation, applying its validation rules.
