---
title: DQN recipe
description: Public MLZO 0.1.0a1 API signatures and behavior for dqn recipe.
---

Read the [dqn recipe guide](../../../guides/dqn/) for examples and usage constraints. Signatures and declared fields below come from MLZO `0.1.0a1` source. Type names follow their defining Python modules.

## dqn

```python
mlzo.dqn(
    q_network: nn.Module,
    *,
    env: Any,
    replay: int = 50000,
    gamma: float = 0.99,
    exploration: str = 'epsilon_greedy',
    target_every: int = 1000,
    batch: int = 64,
    epsilon: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: int = 10000,
    learning_starts: int | None = None,
    seed: int | None = None,
) -> DQNTraining
```

Describe deferred DQN collection and learning. See :class:`DQNTraining`.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/recipes/dqn.py#L408).

## DQNTraining

```python
mlzo.DQNTraining(
    q_network: nn.Module,
    *,
    env: Any,
    replay: int = 50000,
    gamma: float = 0.99,
    exploration: str = 'epsilon_greedy',
    target_every: int = 1000,
    batch: int = 64,
    epsilon: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: int = 10000,
    learning_starts: int | None = None,
    seed: int | None = None,
)
```

A step-driven discrete-action algorithm. Construct with `dqn()`. Replay and target state resume, but the environment resets.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/recipes/dqn.py#L78).

### DQNTraining.step

```python
DQNTraining.step(batch: None) -> Step
```

Collects one environment transition and, after replay warmup, attempts a default optimizer update. Returns detached scalar metrics.

### DQNTraining.close

```python
DQNTraining.close() -> None
```

Closes environment resources and prevents further collection. Repeated calls are safe; the engine does not close it automatically.
