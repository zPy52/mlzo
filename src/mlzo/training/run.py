"""Deferred run configuration and local/cluster lifecycle delegation."""

from __future__ import annotations

import inspect
import math
import uuid
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Literal

import torch
from torch import nn

from mlzo.errors import ConfigurationError, TrainingError, UnsupportedFeatureError
from mlzo.model import FactoryRef

from .runtime import execution_settings
from .types import History, RunStatus, Training


def _budget(epochs: Any, steps: Any) -> None:
    if epochs is not None and steps is not None:
        raise ConfigurationError("fit() accepts total epochs OR total iteration steps, not both.")
    for name, value in (("epochs", epochs), ("steps", steps)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ConfigurationError(f"{name} must be a positive integer total target.")


def _config(config: Mapping[str, Any]) -> None:
    if not isinstance(config["precision"], str) or config["precision"] not in {"auto", "fp32", "bf16", "fp16"}:
        raise ConfigurationError("precision must be 'auto', 'fp32', 'bf16', or 'fp16'.")
    if not isinstance(config["matmul"], str) or config["matmul"] not in {"auto", "strict", "fast"}:
        raise ConfigurationError("matmul must be 'auto', 'strict', or 'fast'.")
    accumulation = config["accumulate"]
    if isinstance(accumulation, bool) or not isinstance(accumulation, int) or accumulation < 1:
        raise ConfigurationError("accumulate must be a positive integer.")
    compilation = config["compile"]
    if type(compilation) is not bool and compilation != "auto":
        raise ConfigurationError("compile must be False, True, or 'auto'.")
    if config["memory"] is not None and config["memory"] != "auto":
        raise UnsupportedFeatureError("Only memory=None or advisory memory='auto' is supported.")
    for name in ("lr", "grad_clip"):
        value = config[name]
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
            raise ConfigurationError(f"{name} must be finite and nonnegative, or None.")
    seed = config["seed"]
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ConfigurationError("seed must be an integer or None.")
    policy = config["checkpoint"]
    if policy is not None:
        if not all(hasattr(policy, field) for field in ("to", "every", "keep_last", "mode")):
            raise ConfigurationError("checkpoint must be a CheckpointPolicy created with mlzo.checkpoint(...).")
        if isinstance(policy.every, bool) or not isinstance(policy.every, int) or policy.every < 1:
            raise ConfigurationError("Checkpoint every must be a positive update-event interval.")
        if policy.mode != "full":
            raise UnsupportedFeatureError("Resumable training supports checkpoint mode='full'; use export() for inference weights.")


class Run:
    """A deferred learning program. Construction never calls a model/data factory."""

    def __init__(self, algorithm_input: Any, data: Any = None, config: Mapping[str, Any] | None = None) -> None:
        self.algorithm_input = algorithm_input
        self.data = data
        self.config = dict(config or {})
        self.id = uuid.uuid4().hex
        self._engine: Any = None
        self._remote_cluster: Any = None
        self._event_callback: Callable[[dict[str, Any]], None] | None = None
        self._state = "created"
        self._error: Any = None
        self._history: History | None = None
        self._local_on: Any = None

    def plan(self, *, on: Any = None, parallel: Any = "auto", example: Any = None) -> Any:
        from mlzo.planning import plan_run
        return plan_run(self, on=on if on is not None else self._local_on, parallel=parallel, example=example)

    def explain(self) -> str:
        description = self.plan().explain()
        if self._engine is not None:
            description += "".join(f"\nExecution: {note}" for note in self._engine.notes)
        return description

    def _prepare(self, *, on: Any = None, parallel: Any = "auto") -> Any:
        if self._engine is None:
            from .engine import Engine
            self._engine = Engine(self, on=on, parallel=parallel)
            self._local_on = on
        else:
            # Validate strategies/device requests even for an already prepared run.
            plan = self.plan(on=on if on is not None else self._local_on, parallel=parallel)
            if plan.device != str(self._engine.device):
                raise ConfigurationError("Changing the execution device of a prepared run is unsupported; resume in a new run.")
        return self._engine

    def fit(self, *, epochs: int | None = None, steps: int | None = None,
            on: Any = None, parallel: Any = "auto", code: str | None = None,
            resume: Any = None, wait: bool = True) -> History | Run:
        """Execute toward a total iteration/epoch target, or submit to a cluster.

        Local execution is synchronous. ``wait=False`` is accepted only for a
        cluster's durable submission. Steps count algorithm calls, not updates.
        """
        _budget(epochs, steps)
        if type(wait) is not bool:
            raise ConfigurationError("wait must be a boolean.")
        target = on if on is not None else self._remote_cluster
        if target is not None and not isinstance(target, (str, torch.device)):
            if not callable(getattr(target, "submit", None)):
                raise ConfigurationError("on must be a Cluster handle or a local torch device.")
            if self._engine is not None:
                raise ConfigurationError("A prepared local run cannot be submitted remotely. Submit a fresh Lazy/training factory recipe.")
            if self._remote_cluster is not None:
                raise UnsupportedFeatureError("Re-submitting/extending a remote run is unsupported; reconnect and wait, or submit a new run.")
            identifier = target.submit(self, epochs=epochs, steps=steps, parallel=parallel, code=code, resume=resume)
            if not isinstance(identifier, str) or not identifier:
                raise TrainingError("Cluster.submit() did not return a durable run ID.")
            self.id = identifier
            self._remote_cluster = target
            self._state = "submitted"
            return self.wait() if wait else self
        if not wait:
            raise UnsupportedFeatureError("Local wait=False is unsupported. Use synchronous fit(), or submit to a cluster for background execution.")
        if code is not None:
            raise ConfigurationError("code= is a remote submission option and has no local execution meaning.")
        if self._state in {"running", "preparing"}:
            raise TrainingError("This run is already executing.")
        if self._state == "failed" and resume is None:
            raise TrainingError("A failed run must resume a committed checkpoint or use a fresh run; completed updates cannot be rolled back.")
        self._error = None
        try:
            with execution_settings(self.config["matmul"]):
                self._state = "preparing"
                engine = self._prepare(on=on, parallel=parallel)
                self._state = "running"
                self._history = engine.fit(epochs=epochs, steps=steps, resume=resume)
                self._state = self._history.state
                return self._history
        except BaseException as exc:
            self._state = "failed"
            self._error = {"type": type(exc).__name__, "message": str(exc)}
            if isinstance(exc, (ConfigurationError, TrainingError, KeyboardInterrupt, SystemExit)):
                raise
            raise TrainingError(f"Run {self.id} failed: {exc}") from exc

    def wait(self) -> History:
        if self._remote_cluster is None:
            if self._history is not None and self._state in {"completed", "cancelled"}:
                return self._history
            raise TrainingError("No submitted remote job or completed local history is attached to this run.")
        result = self._remote_cluster.wait(self.id)
        if is_dataclass(result):
            result = asdict(result)
        if not isinstance(result, Mapping):
            raise TrainingError("Cluster.wait() must return the terminal run record.")
        state = result.get("state", result.get("status"))
        if state not in {"completed", "succeeded", "success", "failed", "cancelled", "canceled"}:
            raise TrainingError(f"Cluster.wait() returned a nonterminal or unknown state {state!r}.")
        self._state = "completed" if state in {"completed", "succeeded", "success"} else state
        self._error = result.get("error")
        if self._state != "completed":
            raise TrainingError(f"Remote run {self.id} {self._state}: {self._error or 'see cluster run record'}")
        value = result.get("history")
        if value is None and isinstance(result.get("result"), Mapping):
            value = result["result"].get("history")
        self._history = History.from_dict(value) if isinstance(value, Mapping) else History(self.id)
        self._history.run_id = self.id
        self._history.state = "completed"
        if isinstance(result.get("counters"), Mapping):
            self._history.counters = dict(result["counters"])
        return self._history

    def status(self) -> RunStatus:
        if self._remote_cluster is not None:
            value = self._remote_cluster.get_run(self.id)
            if is_dataclass(value):
                value = asdict(value)
            return RunStatus(value.get("state", value.get("status", "unknown")), self.id,
                             dict(value.get("counters", {})), value.get("error"))
        return RunStatus(self._state, self.id, self._engine.counters() if self._engine is not None else {}, self._error)

    def evaluate(self, data: Any, *, step: Callable[..., Any] | None = None) -> dict[str, float]:
        if self._remote_cluster is not None:
            raise UnsupportedFeatureError("Remote evaluation is unsupported in MLZO 0.1.0a1.")
        if step is not None and not callable(step):
            raise ConfigurationError("Evaluation step must be callable(model, batch).")
        with execution_settings(self.config["matmul"]):
            engine = self._prepare(on=self._local_on)
            return engine.evaluate(data, step=step)

    def checkpoint(self, *, to: Any = None) -> Any:
        if self._remote_cluster is not None:
            return self._remote_cluster.checkpoint_run(self.id, to=to)
        if self._engine is None:
            raise TrainingError("Prepare the model with fit() or evaluate() before saving a training checkpoint.")
        if self._state in {"running", "preparing"}:
            raise UnsupportedFeatureError("Concurrent manual checkpoint requests are unsupported locally; use a checkpoint policy or save after fit().")
        return self._engine.checkpoint(to=to)

    def save(self, path: Any) -> Any:
        return self.checkpoint(to=path)

    def export(self, *, to: Any, block: str | None = None) -> Any:
        if self._remote_cluster is not None:
            return self._remote_cluster.export_run(self.id, to=to, block=block)
        if self._engine is None:
            raise TrainingError("Prepare the model with fit() or evaluate() before export().")
        if self._state in {"running", "preparing"}:
            raise TrainingError("Export requires a completed training boundary.")
        self._engine._safe()
        from mlzo.inference import export_model
        return export_model(self._engine.model, to=to, recipe=self._engine.recipe, block=block)

    def cancel(self) -> None:
        if self._remote_cluster is not None:
            self._remote_cluster.cancel(self.id)
        elif self._engine is not None and self._state == "running":
            self._engine._cancel_requested = True
        else:
            raise TrainingError("No running job is available to cancel.")


def train(model_or_training: nn.Module | Training[Any] | Callable[[], Training[Any]], data: Any = None, *,
          loss: str | Callable[..., Any] | None = None, step: Callable[..., Any] | None = None,
          optimizer: Any = "adamw", lr: float | None = None, schedule: Any = None,
          precision: Literal["auto", "fp32", "bf16", "fp16"] = "auto",
          matmul: Literal["auto", "strict", "fast"] = "auto", grad_clip: float | None = None,
          accumulate: int = 1, compile: bool | Literal["auto"] = False,
          memory: Literal["auto"] | None = None, checkpoint: Any = None,
          device: str | None = None, seed: int | None = None) -> Run:
    """Describe a learning program without materializing weights or iterating data."""
    if loss is not None and step is not None:
        raise ConfigurationError("loss= and step= are mutually exclusive.")
    if isinstance(model_or_training, nn.Module):
        if loss is None and step is None:
            raise ConfigurationError("A model requires loss= or step=; class algorithms use Training(model).")
        if data is None:
            raise ConfigurationError("Model-based training requires data; use a step-driven Training for collection algorithms.")
    else:
        if loss is not None or step is not None:
            raise ConfigurationError("Training instances/factories cannot receive separate loss= or step=.")
        if isinstance(model_or_training, Training):
            if data is None and type(model_or_training).iteration_source is Training.iteration_source and model_or_training.step_driven is not True:
                raise ConfigurationError("data=None requires iteration_source() or Training.step_driven=True.")
        elif callable(model_or_training):
            try:
                inspect.signature(model_or_training).bind()
            except (TypeError, ValueError) as exc:
                raise ConfigurationError("Training factories must be callable with zero arguments.") from exc
            FactoryRef.from_callable(model_or_training)
        else:
            raise ConfigurationError("train() requires a torch model, Training instance, or importable zero-argument training factory.")
    if loss is not None and not isinstance(loss, str) and not callable(loss):
        raise ConfigurationError("loss must be a built-in name or callable(prediction, target).")
    if isinstance(loss, str) and loss not in {"cross_entropy", "mse", "l1", "bce", "bce_with_logits", "bce_logits"}:
        raise ConfigurationError(f"Unknown loss {loss!r}.")
    if step is not None and not callable(step):
        raise ConfigurationError("step must be callable(model, batch).")
    config = {"loss": loss, "step": step, "optimizer": optimizer, "lr": lr, "schedule": schedule,
              "precision": precision, "matmul": matmul, "grad_clip": grad_clip, "accumulate": accumulate,
              "compile": compile, "memory": memory, "checkpoint": checkpoint, "device": device, "seed": seed}
    _config(config)
    return Run(model_or_training, data, config)
