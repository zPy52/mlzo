"""Single-environment DQN with bounded replay and partial environment resume."""

from __future__ import annotations

import copy
import inspect
import math
import random
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mlzo.errors import ConfigurationError, TrainingError, UnsupportedFeatureError
from mlzo.training import Step, Training
from mlzo.training.runtime import _evaluating

__all__ = ["DQNTraining", "dqn"]


def _integer(name: str, value: Any, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ConfigurationError(f"DQN {name} must be an integer >= {minimum}.")
    return int(value)


def _probability(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ConfigurationError(f"DQN {name} must be finite and between zero and one.")
    return float(value)


def _flag(value: Any, name: str) -> bool:
    try:
        tensor = torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ConfigurationError(f"DQN env.step() {name} must be a scalar boolean.") from exc
    if tensor.ndim != 0 or tensor.dtype != torch.bool:
        raise ConfigurationError(f"DQN env.step() {name} must be a scalar boolean; vector environments are unsupported.")
    return bool(tensor.item())


def _autoreset_requires_final(env: Any) -> bool:
    """Recognize advertised same-step/unknown auto-reset wrappers without Gym."""
    seen: set[int] = set()
    requires_final = False
    while env is not None and id(env) not in seen:
        seen.add(id(env))
        metadata = getattr(env, "metadata", {})
        # Static lookup avoids deprecated wrapper __getattr__ forwarding for
        # absent attributes. Traversing .env inspects the underlying wrappers.
        mode = (getattr(env, "autoreset_mode")
                if inspect.getattr_static(env, "autoreset_mode", None) is not None else None)
        if mode is None and isinstance(metadata, Mapping):
            mode = metadata.get("autoreset_mode")
        if mode is not None:
            label = str(getattr(mode, "value", mode)).lower().replace("_", "").replace(" ", "")
            if label not in {"disabled", "nextstep"}:
                requires_final = True
        name = type(env).__name__.lower()
        advertised = any(inspect.getattr_static(env, key, False) is True for key in ("autoreset", "auto_reset"))
        if isinstance(metadata, Mapping):
            advertised = advertised or metadata.get("autoreset") is True or metadata.get("auto_reset") is True
        if "autoreset" in name or "auto_reset" in name or advertised:
            # Gymnasium >=1.0 Autoreset returns the actual final observation on
            # the terminal step. We explicitly reset before its next-step reset.
            known_next_step = (type(env).__name__ == "Autoreset"
                               and type(env).__module__.startswith("gymnasium.wrappers."))
            if mode is None and not known_next_step:
                requires_final = True
        env = getattr(env, "env", None)
    return requires_final


class DQNTraining(Training[None]):
    """DQN for one environment, real observation vectors, and Discrete actions.

    Each iteration collects exactly one env.step(). After warmup, it attempts
    one scalar SmoothL1 update through the optimizer named ``default``. Epsilon
    decays linearly from 1.0 to 0.05 over the first 10,000 environment steps by
    default. ``learning_starts=None`` means ``batch`` collected transitions.
    ``target_every`` counts successful optimizer updates, excluding skips.

    Construction never calls an environment or model factory. Replay, target
    weights, counters, and RNG state are checkpointed in ``state``. Environment
    state is not saved: restore keeps replay but resets the environment and
    discards an unfinished episode's statistics. Call ``close()`` when done.
    """

    step_driven = True

    def __init__(self, q_network: nn.Module, *, env: Any, replay: int = 50_000,
                 gamma: float = 0.99, exploration: str = "epsilon_greedy",
                 target_every: int = 1000, batch: int = 64,
                 epsilon: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay: int = 10_000, learning_starts: int | None = None,
                 seed: int | None = None) -> None:
        super().__init__(q_network)
        if isinstance(env, str):
            if not env.strip():
                raise ConfigurationError("DQN env ID must be nonempty.")
        elif not callable(env) and not (callable(getattr(env, "reset", None)) and callable(getattr(env, "step", None))):
            raise ConfigurationError("DQN env must be a Gymnasium ID, zero-argument environment builder, or environment object.")
        if exploration != "epsilon_greedy":
            raise UnsupportedFeatureError("DQN supports only exploration='epsilon_greedy'.")
        batch = _integer("batch", batch)
        replay = _integer("replay", replay)
        if replay < batch:
            raise ConfigurationError("DQN replay capacity must be at least batch.")
        epsilon = _probability("epsilon", epsilon)
        epsilon_end = _probability("epsilon_end", epsilon_end)
        if seed is not None:
            seed = _integer("seed", seed, minimum=0)
        self._config = {
            "replay": replay, "gamma": _probability("gamma", gamma), "batch": batch,
            "target_every": _integer("target_every", target_every),
            "epsilon": epsilon, "epsilon_end": epsilon_end,
            "epsilon_decay": _integer("epsilon_decay", epsilon_decay),
            "learning_starts": batch if learning_starts is None else _integer("learning_starts", learning_starts, minimum=0),
            "seed": seed,
        }
        self.state = {
            "schema_version": 1, "config": dict(self._config),
            "replay": [], "replay_cursor": 0, "target": None, "rng": None,
            "env_steps": 0, "optimizer_updates": 0, "skipped_updates": 0,
            "target_syncs": 0, "target_update": 0, "episodes": 0,
            "episode_return": 0.0, "episode_steps": 0,
            "observation_shape": None, "actions": None, "action_start": None,
            "environment_resume": "partial_reset", "environment_resumes": 0,
            "partial_environment_resume": False,
        }
        self._env_spec = env
        self._live_env = not isinstance(env, (str, type)) and callable(getattr(env, "step", None)) and callable(getattr(env, "reset", None))
        self._env: Any = None
        self._target: nn.Module | None = None
        self._rng: random.Random | None = None
        self._observation: torch.Tensor | None = None
        self._active_state: Any = None
        self._requires_final = False
        self._closed = False

    def _observation_tensor(self, value: Any) -> torch.Tensor:
        try:
            tensor = torch.as_tensor(value)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise UnsupportedFeatureError("DQN observations must be real numeric vectors convertible to tensors.") from exc
        if tensor.layout != torch.strided or tensor.ndim != 1 or tensor.numel() == 0 or tensor.is_complex():
            raise UnsupportedFeatureError("DQN supports nonempty real observation vectors only; scalar, image, and composite observations are unsupported.")
        tensor = tensor.detach().to(device="cpu", dtype=torch.float32).clone()
        if not bool(torch.isfinite(tensor).all().item()):
            raise ConfigurationError("DQN observation values must be finite.")
        shape = tuple(tensor.shape)
        if self.state["observation_shape"] is None:
            self.state["observation_shape"] = shape
        elif tuple(self.state["observation_shape"]) != shape:
            raise ConfigurationError("DQN observation shape differs from the environment or restored replay.")
        return tensor

    def _reset(self, *, seed: int | None = None) -> torch.Tensor:
        # Do not catch TypeError and retry: reset may already have side effects.
        result = self._env.reset() if seed is None else self._env.reset(seed=seed)
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], Mapping):
            result = result[0]
        return self._observation_tensor(result)

    def _open_env(self) -> None:
        if isinstance(self._env_spec, str):
            try:
                import gymnasium
            except ImportError as exc:
                raise UnsupportedFeatureError("DQN environment IDs require optional Gymnasium. Install mlzo[rl] or supply a duck-typed environment builder/object.") from exc
            self._env = gymnasium.make(self._env_spec)
        else:
            self._env = self._env_spec if self._live_env else self._env_spec()
        env = self._env
        if not callable(getattr(env, "reset", None)) or not callable(getattr(env, "step", None)):
            raise ConfigurationError("DQN environment builder must return an object with reset() and step().")
        if hasattr(env, "num_envs"):
            raise UnsupportedFeatureError("DQN supports a single environment; vector environments are unsupported.")
        space = getattr(env, "action_space", None)
        action_count = getattr(space, "n", None)
        if (isinstance(action_count, bool) or not isinstance(action_count, Integral)
                or action_count < 1 or getattr(space, "shape", ()) != ()):
            raise UnsupportedFeatureError("DQN requires a scalar Discrete action space with a positive integer .n.")
        action_start = getattr(space, "start", 0)
        if isinstance(action_start, bool) or not isinstance(action_start, Integral):
            raise UnsupportedFeatureError("DQN Discrete action_space.start must be an integer.")
        if (self.state["actions"] is not None
                and (self.state["actions"], self.state["action_start"]) != (int(action_count), int(action_start))):
            raise ConfigurationError("DQN action space differs from the restored state.")
        self.state["actions"], self.state["action_start"] = int(action_count), int(action_start)
        observation_space = getattr(env, "observation_space", None)
        if observation_space is not None:
            shape = getattr(observation_space, "shape", None)
            if shape is None or len(shape) != 1 or shape[0] < 1:
                raise UnsupportedFeatureError("DQN observation_space must describe a nonempty vector; image, scalar, and composite spaces are unsupported.")
            if self.state["observation_shape"] is not None and tuple(shape) != tuple(self.state["observation_shape"]):
                raise ConfigurationError("DQN observation space differs from the restored state.")
            self.state["observation_shape"] = tuple(int(size) for size in shape)
        self._requires_final = _autoreset_requires_final(env)
        self._observation = self._reset(seed=self._config["seed"])

    @staticmethod
    def _cpu_weights(model: nn.Module) -> dict[str, torch.Tensor]:
        weights = model.state_dict()
        if not all(isinstance(value, torch.Tensor) for value in weights.values()):
            raise UnsupportedFeatureError("DQN Q-network state_dict() must contain tensors only; custom extra state is unsupported.")
        return {name: value.detach().cpu().clone() for name, value in weights.items()}

    def _prepare_runtime(self) -> None:
        if self._closed:
            raise TrainingError("This DQNTraining is closed; create a fresh recipe to continue.")
        if self._engine is None or not self._engine._in_step:
            raise TrainingError("DQN step() must execute through mlzo.train(...).fit(steps=...).")
        if self._active_state is self.state:
            return
        if set(self._engine.optimizer_steps) != {"default"} or self._engine.config["accumulate"] != 1:
            raise ConfigurationError("DQN requires one optimizer named 'default' and accumulate=1.")
        if not isinstance(self.state, dict) or self.state.get("schema_version") != 1 or self.state.get("config") != self._config:
            raise ConfigurationError("DQN checkpoint configuration differs from this recipe or has an unsupported schema.")
        if (self.state["optimizer_updates"] != self._engine.optimizer_steps["default"]
                or self.state["skipped_updates"] != self._engine.skipped_updates["default"]):
            raise ConfigurationError("DQN checkpoint counters differ from the engine's default optimizer counters.")
        # Engine.load_state_dict replaces algorithm.state. Discard all derived
        # runtime objects rather than carrying a stale target or episode forward.
        previous_env = self._env
        self._env = None
        self._target = self._rng = self._observation = None
        self._active_state = None
        if previous_env is not None and not self._live_env:
            close = getattr(previous_env, "close", None)
            if callable(close):
                close()
        if self.state["env_steps"]:
            self.state["partial_environment_resume"] = True
            self.state["environment_resumes"] += 1
            self.state["episode_return"] = 0.0
            self.state["episode_steps"] = 0
        # Generic restore moves state tensors to the worker. Replay belongs on
        # CPU and only sampled minibatches need to visit the execution device.
        self.state["replay"] = [(obs.detach().cpu(), action, reward, nxt.detach().cpu(), terminated)
                                for obs, action, reward, nxt, terminated in self.state["replay"]]
        if len(self.state["replay"]) > self._config["replay"]:
            raise ConfigurationError("DQN checkpoint replay exceeds its configured capacity.")
        seed = self._config["seed"]
        if seed is None:
            seed = random.getrandbits(64)
        self._rng = random.Random(seed)
        if self.state["rng"] is not None:
            self._rng.setstate(self.state["rng"])
        self._open_env()
        try:
            self._target = copy.deepcopy(self._engine.model)
        except Exception as exc:
            raise UnsupportedFeatureError("DQN target creation requires a Q network that supports copy.deepcopy() after materialization.") from exc
        online_tensors = [*self._engine.model.parameters(), *self._engine.model.buffers()]
        target_tensors = [*self._target.parameters(), *self._target.buffers()]
        online_ids = {id(tensor) for tensor in online_tensors}
        online_storage = {(tensor.device, tensor.untyped_storage().data_ptr()) for tensor in online_tensors
                          if tensor.layout == torch.strided and tensor.numel()}
        if (self._target is self._engine.model or any(id(tensor) in online_ids for tensor in target_tensors)
                or any((tensor.device, tensor.untyped_storage().data_ptr()) in online_storage for tensor in target_tensors
                       if tensor.layout == torch.strided and tensor.numel())):
            raise UnsupportedFeatureError("DQN target copy must have independent parameters and buffers.")
        self._target.requires_grad_(False)
        self._target.zero_grad(set_to_none=True)
        self._target.eval()
        if self.state["target"] is None:
            self.state["target"] = self._cpu_weights(self._target)
        else:
            self.state["target"] = {name: value.detach().cpu() for name, value in self.state["target"].items()}
            self._target.load_state_dict(self.state["target"])
        self._active_state = self.state

    def _q_values(self, model: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        values = model(observations)
        expected = (observations.shape[0], self.state["actions"])
        if not isinstance(values, torch.Tensor) or tuple(values.shape) != expected or not values.is_floating_point():
            raise ConfigurationError(f"DQN Q network must return a floating tensor [batch, actions] of shape {expected}.")
        if values.device != observations.device:
            raise ConfigurationError("DQN Q-network output must remain on the worker execution device.")
        return values

    def _inputs(self, observations: torch.Tensor) -> torch.Tensor:
        dtype = next((p.dtype for p in self._engine.model.parameters() if p.is_floating_point()), torch.float32)
        return observations.to(device=self._engine.device, dtype=dtype)

    def _learn(self) -> tuple[float, bool]:
        transitions = self._rng.sample(self.state["replay"], self._config["batch"])
        observations, actions, rewards, next_observations, terminated = zip(*transitions)
        device = self._engine.device
        observations = self._inputs(torch.stack(observations))
        next_observations = self._inputs(torch.stack(next_observations))
        actions = torch.tensor(actions, device=device, dtype=torch.long)
        rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
        nonterminal = ~torch.tensor(terminated, device=device, dtype=torch.bool)
        with torch.no_grad():
            next_values = torch.zeros(len(transitions), device=device, dtype=torch.float32)
            if bool(nonterminal.any().item()):
                next_values[nonterminal] = self._q_values(self._target, next_observations[nonterminal]).float().max(dim=1).values
            targets = rewards + self._config["gamma"] * next_values
        predictions = self._q_values(self.model, observations).gather(1, actions[:, None]).squeeze(1).float()
        loss = F.smooth_l1_loss(predictions, targets, reduction="mean")
        before = self._engine.optimizer_steps["default"]
        self.update("default", loss)
        after = self._engine.optimizer_steps["default"]
        updated = after > before
        self.state["optimizer_updates"] = after
        self.state["skipped_updates"] = self._engine.skipped_updates["default"]
        if updated and after % self._config["target_every"] == 0:
            self._target.load_state_dict(self._engine.model.state_dict())
            self.state["target"] = self._cpu_weights(self._target)
            self.state["target_update"] = after
            self.state["target_syncs"] += 1
        return float(loss.detach().item()), updated

    def step(self, batch: None) -> Step:
        self._prepare_runtime()
        state, config = self.state, self._config
        fraction = min(state["env_steps"] / config["epsilon_decay"], 1.0)
        epsilon = config["epsilon"] + fraction * (config["epsilon_end"] - config["epsilon"])
        with torch.no_grad(), _evaluating(self.model):
            # Validate even during random exploration, before consuming a step.
            values = self._q_values(self.model, self._inputs(self._observation.unsqueeze(0)))
            action = (self._rng.randrange(state["actions"]) if self._rng.random() < epsilon
                      else int(values.argmax(dim=1).item()))
        result = self._env.step(action + state["action_start"])
        if not isinstance(result, (tuple, list)) or len(result) != 5:
            raise ConfigurationError("DQN env.step() must return (observation, reward, terminated, truncated, info); legacy done and vector APIs are unsupported.")
        observation, reward, terminated, truncated, info = result
        terminated, truncated = _flag(terminated, "terminated"), _flag(truncated, "truncated")
        if isinstance(reward, torch.Tensor) and reward.ndim == 0:
            reward = reward.detach().item()
        if isinstance(reward, bool) or not isinstance(reward, Real) or not math.isfinite(reward):
            raise ConfigurationError("DQN env.step() reward must be a finite real scalar.")
        reward = float(reward)
        if not isinstance(info, Mapping):
            raise ConfigurationError("DQN env.step() info must be a mapping.")
        done = terminated or truncated
        final = observation
        if done:
            for key in ("final_observation", "terminal_observation", "final_obs"):
                if key in info:
                    final = info[key]
                    if final is None:
                        raise UnsupportedFeatureError("DQN auto-reset transition has no usable final observation.")
                    break
            else:
                if self._requires_final or "final_info" in info or info.get("autoreset") is True or info.get("auto_reset") is True:
                    raise UnsupportedFeatureError("DQN cannot resolve an auto-reset wrapper's final observation; expose info['final_observation'] or use a non-auto-reset environment.")
        next_observation = self._observation_tensor(final)
        transition = (self._observation, action, reward, next_observation, terminated)
        if len(state["replay"]) < config["replay"]:
            state["replay"].append(transition)
        else:
            state["replay"][state["replay_cursor"]] = transition
        state["replay_cursor"] = (state["replay_cursor"] + 1) % config["replay"]
        state["env_steps"] += 1
        state["episode_return"] += reward
        state["episode_steps"] += 1
        metrics = {"reward": reward, "epsilon": epsilon, "terminated": int(terminated), "truncated": int(truncated)}
        if done:
            state["episodes"] += 1
            metrics.update(episode_return=state["episode_return"], episode_length=state["episode_steps"])
            # Also suppress a next-step autoreset's reset-only pseudo-transition.
            self._observation = self._reset()
            state["episode_return"], state["episode_steps"] = 0.0, 0
        else:
            self._observation = next_observation
        updated = False
        if state["env_steps"] >= config["learning_starts"] and len(state["replay"]) >= config["batch"]:
            metrics["loss"], updated = self._learn()
        state["rng"] = self._rng.getstate()
        metrics.update(env_steps=state["env_steps"], replay_size=len(state["replay"]),
                       optimizer_updates=state["optimizer_updates"], skipped_updates=state["skipped_updates"],
                       update_performed=int(updated), target_syncs=state["target_syncs"],
                       target_update=state["target_update"], episodes=state["episodes"],
                       partial_environment_resume=int(state["partial_environment_resume"]),
                       environment_resumes=state["environment_resumes"])
        # update() has already completed. Returning an automatic loss would
        # request a second update from the generic engine.
        return Step(metrics=metrics)

    def close(self) -> None:
        """Release environment resources once, including a supplied live env.

        This does not construct a deferred environment. A closed recipe cannot
        collect again; resume a checkpoint with a fresh recipe instead.
        """
        if self._closed:
            return
        self._closed = True
        if self._env is not None:
            env = self._env
        elif self._live_env:
            env = self._env_spec
        else:
            env = None
        self._env = self._target = self._rng = self._observation = self._active_state = None
        close = getattr(env, "close", None)
        if callable(close):
            close()


def dqn(q_network: nn.Module, *, env: Any, replay: int = 50_000, gamma: float = 0.99,
        exploration: str = "epsilon_greedy", target_every: int = 1000, batch: int = 64,
        epsilon: float = 1.0, epsilon_end: float = 0.05, epsilon_decay: int = 10_000,
        learning_starts: int | None = None, seed: int | None = None) -> DQNTraining:
    """Describe deferred DQN collection and learning. See :class:`DQNTraining`."""
    return DQNTraining(q_network, env=env, replay=replay, gamma=gamma, exploration=exploration,
                       target_every=target_every, batch=batch, epsilon=epsilon,
                       epsilon_end=epsilon_end, epsilon_decay=epsilon_decay,
                       learning_starts=learning_starts, seed=seed)
