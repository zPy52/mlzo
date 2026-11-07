---
title: Optimizers and schedules
description: Public MLZO 0.1.0a1 API signatures and behavior for optimizers and schedules.
---

Read the [optimizers and schedules guide](../../../guides/optimizers/) for examples and usage constraints. Signatures and declared fields below come from MLZO `0.1.0a1` source. Type names follow their defining Python modules.

## opt

```python
mlzo.opt(name: Any='adamw', *, params: Any=None, **kwargs: Any) -> OptimizerSpec
```

Bind an optimizer later using a submodule path or parameter-name selector.

`params='encoder'` selects that submodule, `params=['head.weight', ...]`
selects exact parameter names. A list of `{'params': selector, ...}` gives
native parameter groups. Selectors remain valid after Lazy construction.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/training/optim.py#L37).

## OptimizerSpec

Declared fields:

```python
name: Any = 'adamw'
params: Any = None
kwargs: dict[str, Any] = field(default_factory=dict)
```

Deferred optimizer name or factory, parameter selectors, and keyword options. Construct conveniently with `opt()`.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/training/optim.py#L21).

### OptimizerSpec.to_dict

```python
OptimizerSpec.to_dict() -> dict[str, Any]
```

Returns a dictionary representation. Public invitation and cluster representations omit credentials.

### OptimizerSpec.from_dict

```python
OptimizerSpec.from_dict(value: Mapping[str, Any]) -> OptimizerSpec
```

Reconstructs this value from its dictionary representation, applying its validation rules.

## ScheduleSpec

Declared fields:

```python
kind: str = 'constant'
warmup: int = 0
total_steps: int | None = None
end_factor: float = 0.0
gamma: float = 0.99
factor: float = 1.0
```

Validated built-in schedule configuration. Rates multiply base learning rates and advance on successful updates.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/training/schedules.py#L17).

### ScheduleSpec.to_dict

```python
ScheduleSpec.to_dict() -> dict[str, Any]
```

Returns a dictionary representation. Public invitation and cluster representations omit credentials.

### ScheduleSpec.from_dict

```python
ScheduleSpec.from_dict(value: Mapping[str, Any]) -> ScheduleSpec
```

Reconstructs this value from its dictionary representation, applying its validation rules.

## constant

```python
mlzo.constant(factor: float=1.0, *, warmup: int=0) -> ScheduleSpec
```

Returns a constant multiplier specification with optional update-indexed warmup.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/training/schedules.py#L45).

## linear

```python
mlzo.linear(*, warmup: int=0, total_steps: int | None=None, end_factor: float=0.0) -> ScheduleSpec
```

Returns a linear decay specification. An omitted horizon must be inferable for automatic training.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/training/schedules.py#L49).

## cosine

```python
mlzo.cosine(*, warmup: int=0, total_steps: int | None=None, min_factor: float=0.0) -> ScheduleSpec
```

Returns a cosine decay specification with `min_factor` as the final base-rate multiplier.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/training/schedules.py#L53).

## exponential

```python
mlzo.exponential(gamma: float=0.99, *, warmup: int=0) -> ScheduleSpec
```

Returns geometric decay per successful optimizer update after optional warmup.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/training/schedules.py#L57).

## schedules

`mlzo.schedules` is the schedule module. Its `constant`, `linear`, `cosine`, and `exponential` constructors are also exported directly from `mlzo`.
