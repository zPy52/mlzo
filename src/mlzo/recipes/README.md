# DQN alpha recipe

`dqn(q_network, *, env, replay=50_000, gamma=0.99,
exploration="epsilon_greedy", target_every=1000, batch=64, epsilon=1.0,
epsilon_end=0.05, epsilon_decay=10_000, learning_starts=None, seed=None)`
returns a step-driven `DQNTraining`. The package entrypoint can re-export it for
`mlzo.train(mlzo.dqn(...), optimizer="adam", lr=...)`.

```python
from torch import nn
from mlzo.model import Lazy
from mlzo.recipes import dqn
from mlzo.training import train

def q_network():
    return nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 2))

algorithm = dqn(Lazy(q_network), env="CartPole-v1")
try:
    run = train(algorithm, optimizer="adam", lr=1e-3)
    history = run.fit(steps=100, on="cpu")
finally:
    algorithm.close()
```

Neither `dqn()` nor `train()` calls a model or environment factory. `env` accepts
a Gymnasium ID, zero-argument builder, or live local environment. IDs import
optional Gymnasium on the first collection iteration. Duck-typed toy
environments do not require Gymnasium.

Each iteration consumes exactly one `env.step()`. Use `fit(steps=...)` with a
total collection-iteration target. Epsilon for an action is
`epsilon + min(collected_steps / epsilon_decay, 1) * (epsilon_end - epsilon)`.
Warmup counts toward decay. Learning starts once `learning_starts` transitions
have been collected and replay holds `batch` entries. `learning_starts=None`
means `batch`. Each eligible iteration attempts one mean SmoothL1 TD update
through `self.update("default", loss)`, then returns metrics without an automatic
loss. Use one optimizer named `default` and `accumulate=1`.

The target is an independent, gradient-free copy of the materialized network
in evaluation mode. It copies online weights after successful optimizer updates
`target_every`, `2 * target_every`, and so on. Skips do not advance that schedule.
Action selection uses `no_grad` and temporarily evaluates the online network,
then restores all module modes. Networks and minibatches use the worker device;
bounded replay holds detached CPU float32 tensors.

Supported inputs are fixed-length real observation vectors and scalar Discrete
actions, including nonzero `start`. The Q network must return floating
`[batch, actions]` tensors, support `copy.deepcopy()` after materialization, and
have tensor-only `state_dict()` values. Network inputs use its floating parameter
dtype. `step()` must return `observation, reward, terminated, truncated, info`.
`reset()` can return an observation or `observation, info`. A recipe seed controls
exploration, replay sampling and the first environment reset; seeded toy envs
must accept `reset(seed=...)`. Use `train(seed=...)` for model/engine seeding too.

Both termination and truncation reset the episode; only `terminated` masks TD
bootstrapping. Replay keeps the actual final observation before reset.
Same-step auto-reset wrappers must expose `final_observation`,
`terminal_observation`, or `final_obs` in `info`. Advertised auto-reset behavior
without a resolvable final observation fails clearly. Declared next-step reset
modes and Gymnasium's next-step `Autoreset` work through an explicit reset at
episode boundaries. Opaque wrappers that hide their reset behavior are
unsupported. See the [Gymnasium environment contract](https://gymnasium.farama.org/api/env/)
and [auto-reset documentation](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.Autoreset).

Generic checkpoints capture replay, target weights, RNG and counters in `state`.
Restore discards cached runtime objects, preserves replay/target weights and
resets the environment. Factory/ID environments reopen; live objects reset in
place. Unfinished episode statistics are discarded. State records
`environment_resume="partial_reset"`; resumed metrics include
`partial_environment_resume=1` and `environment_resumes`. Exact environment
continuation is unavailable. Resume with the same recipe configuration and
spaces. Metrics also report reward, epsilon, replay size, environment steps,
episodes, successful/skipped updates, and target synchronization counters.
`History` separately records algorithm iterations. `close()` is idempotent,
releases environment resources and prevents further collection. The generic
engine does not call it automatically.

For remote use, submit an importable zero-argument training factory that returns
`dqn(Lazy(...), env=...)` with an ID or worker-local builder. Live environments
and Training instances are not automatically serialized. Install environment
dependencies on the worker. Continuous/multidiscrete actions, image/composite
observations, vector environments, PPO, actor-critic and distributed rollouts
are deferred.
