---
title: The DQN recipe
description: Train a discrete-action DQN and understand replay, target updates, and environment resume.
---

`mlzo.dqn()` returns a step-driven `DQNTraining`. Model and environment factories remain deferred until execution. This is a small discrete-action recipe, not a general reinforcement-learning framework.

## Collect and learn

Install `.[rl]` to use a Gymnasium environment ID:

```python
import mlzo
from torch import nn

def q_network():
    return nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 2))

algorithm = mlzo.dqn(mlzo.Lazy(q_network), env="CartPole-v1", seed=7)
try:
    run = mlzo.train(algorithm, optimizer="adam", lr=1e-3, seed=7)
    history = run.fit(steps=100, on="cpu")
finally:
    algorithm.close()
```

This factory is suitable for local training. Move it into an importable project module for transport or stored inference reconstruction. You can also pass a zero-argument environment builder or live local environment. Duck-typed toy environments need no Gymnasium installation.

Each algorithm iteration performs one `env.step()`. Use a finite total `steps=` target. One optimizer named `default` and `accumulate=1` are required.

## Defaults and update timing

| Option | Default | Meaning |
| --- | --- | --- |
| `replay` | `50_000` | Maximum detached CPU replay transitions |
| `gamma` | `0.99` | TD discount |
| `exploration` | `"epsilon_greedy"` | Supported action selection |
| `target_every` | `1000` | Successful optimizer updates between target copies |
| `batch` | `64` | Replay minibatch size |
| `epsilon`, `epsilon_end` | `1.0`, `0.05` | Initial and final exploration probabilities |
| `epsilon_decay` | `10_000` | Collection iterations over which epsilon changes linearly |
| `learning_starts` | `None` | Defaults to `batch` collected transitions |
| `seed` | `None` | Exploration, replay sampling, and first environment reset seed |

Warmup counts toward epsilon decay. Once replay has a full batch and the collection threshold is met, each iteration attempts one mean SmoothL1 TD update. Skipped updates do not advance target synchronization. `train(seed=...)` separately controls engine/model seeding.

## Environment contract

Observations must be fixed-length real vectors. Actions must use scalar Discrete spaces, including a nonzero start. The Q network returns floating `[batch, actions]` tensors, supports `copy.deepcopy()` after construction, and has tensor-only state dictionary values.

`step()` returns `observation, reward, terminated, truncated, info`. `reset()` can return an observation or `(observation, info)`. Seeded environments must accept `reset(seed=...)`.

Termination and truncation both reset the episode, but only termination disables TD bootstrapping. Replay stores the actual final observation. Same-step auto-reset wrappers must expose `final_observation`, `terminal_observation`, or `final_obs` in `info`. Opaque wrappers that hide their reset behavior are unsupported.

## Checkpoint and close

Checkpoints retain replay, target weights, RNG, and counters in algorithm state. Resume resets the environment and discards unfinished episode statistics. The state records `environment_resume="partial_reset"`; resumed metrics report `partial_environment_resume=1` and `environment_resumes`. Preserve the recipe configuration and environment spaces.

`close()` is idempotent and prevents further collection. The generic engine does not call it automatically. Remote use needs an importable zero-argument training factory returning DQN with an environment ID or worker-local builder. Continuous or multidiscrete actions, image/composite observations, vector environments, PPO, actor-critic, and distributed rollouts are unsupported.
