"""The common local/worker single-device engine. No networking belongs here."""

from __future__ import annotations

import os

import math
from collections.abc import Iterator, Mapping
from contextlib import nullcontext
from itertools import repeat
from threading import get_ident
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from mlzo.errors import ConfigurationError, TrainingError, UnsupportedFeatureError
from mlzo.model import get_recipe, materialize
from mlzo.planning import plan_run

from .adapters import DeclarativeTraining, FunctionalTraining
from .optim import bind_optimizers
from .runtime import move_batch, normalization_weight, restricted_copy, restore_rng, rng_state, seed_process
from .schedules import Schedule
from .types import History, Step, Training, detached_metrics


def _length(value: Any) -> int | None:
    try:
        return len(value)
    except (TypeError, NotImplementedError):
        return None


class Engine:
    """Mutable execution state, constructed only when a Run starts execution.

    ``model`` always refers to the original concrete model, even when the
    algorithm uses a compiled forward wrapper. Snapshots are CPU copies.
    ``_event_callback(dict)`` is an optional worker-owned reporting hook.
    """

    def __init__(self, run: Any, *, on: Any = None, parallel: Any = "auto") -> None:
        self.run = run
        self.config = run.config
        self._event_callback = getattr(run, "_event_callback", None)
        self.notes: list[str] = []
        self.iteration = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.update_events = 0
        self._pending_count = 0
        self._pending_weight = 0.0
        self._pending_finite = True
        self._in_step = False
        self._in_fit = False
        self._fit_thread: int | None = None
        self._failed_boundary = False
        self._explicit_calls = 0
        self._iterator: Any = None
        self._epoch_length: int | None = None
        self._schedule_epoch_length: int | None = None
        self._source: Any = None
        self._source_resolved = False
        self._data_restartable = True
        self._cancel_requested = False
        self._last_checkpoint_event = 0
        self._pending_schedulers: Any = None
        self.schedulers: dict[str, Schedule] = {}
        self.history = History(run.id)
        initial_plan = run.plan(on=on, parallel=parallel)
        self.device = torch.device(initial_plan.device)
        seed_process(self.config.get("seed"), self.device)
        value = run.algorithm_input
        if isinstance(value, Training):
            algorithm = value
        elif isinstance(value, nn.Module):
            algorithm = (FunctionalTraining(value, self.config["step"]) if self.config.get("step") is not None
                         else DeclarativeTraining(value, self.config["loss"]))
        else:
            algorithm = value()
            if not isinstance(algorithm, Training):
                raise ConfigurationError("A zero-argument training factory must return Training.")
        if algorithm._engine is not None:
            raise ConfigurationError("This Training instance is already bound to a run; construct a fresh algorithm.")
        # Factories can introduce placement metadata that was unknown to planning.
        final_plan = plan_run(SimpleNamespace(algorithm_input=algorithm, data=run.data, config=self.config),
                              on=on, parallel=parallel)
        self.device = torch.device(final_plan.device)
        self.precision = final_plan.precision
        self.algorithm = algorithm
        self.model = materialize(algorithm.model, self.device, seed=self.config.get("seed"))
        self.recipe = get_recipe(self.model)
        algorithm.model = self.model
        if isinstance(algorithm, DeclarativeTraining) and isinstance(algorithm.loss, nn.Module):
            algorithm.loss.to(self.device)
        self.optimizers = bind_optimizers(self.model, self.config["optimizer"], self.config.get("lr"))
        self.optimizer_steps = {name: 0 for name in self.optimizers}
        self.skipped_updates = {name: 0 for name in self.optimizers}
        self.scalers = {name: self._make_scaler() for name in self.optimizers}
        compilation = self.config.get("compile", False)
        if compilation in {True, "auto"}:
            try:
                if not callable(getattr(torch, "compile", None)):
                    raise UnsupportedFeatureError("torch.compile is unavailable in this PyTorch runtime.")
                algorithm.model = torch.compile(self.model)
                self.notes.append("Only the model forward is compiled. A failed training call is never retried.")
            except Exception as exc:
                if compilation is True:
                    raise TrainingError(f"Model compilation setup failed: {exc}") from exc
                self.notes.append(f"compile='auto' kept eager execution after setup failed: {exc}")
        if self.config.get("memory") == "auto":
            self.notes.append("memory='auto' is advisory: memory use is unknown and no model transformation was applied.")
        self.model.zero_grad(set_to_none=True)
        algorithm._engine = self
        self._emit("prepared", device=str(self.device), precision=self.precision, notes=list(self.notes))

    def _emit(self, kind: str, **payload: Any) -> None:
        callback = self._event_callback or getattr(self.run, "_event_callback", None)
        if callback is not None:
            callback({"type": kind, "run_id": self.run.id, "counters": self.counters(), **payload})

    def counters(self) -> dict[str, Any]:
        return {"iteration": self.iteration, "epoch": self.epoch, "batch_in_epoch": self.batch_in_epoch,
                "optimizer_steps": dict(self.optimizer_steps), "skipped_updates": dict(self.skipped_updates),
                "update_events": self.update_events}

    def _make_scaler(self) -> Any:
        if self.precision != "fp16":
            return None
        if self.device.type != "cuda":
            raise UnsupportedFeatureError("FP16 gradient scaling is supported only on CUDA.")
        modern = getattr(getattr(torch, "amp", None), "GradScaler", None)
        return modern("cuda") if modern is not None else torch.cuda.amp.GradScaler()

    def _autocast(self) -> Any:
        if self.precision == "fp32":
            return nullcontext()
        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    def _resolve_source(self) -> Any:
        if not self._source_resolved:
            if self.run.data is not None:
                self._source = self.run.data
            elif type(self.algorithm).iteration_source is not Training.iteration_source:
                self._source = self.algorithm.iteration_source()
                if self._source is None:
                    raise ConfigurationError("iteration_source() must return an iterable.")
            elif self.algorithm.step_driven is True:
                self._source = None
            else:
                raise ConfigurationError("data=None requires iteration_source() or Training.step_driven=True.")
            if isinstance(self._source, (str, os.PathLike)):
                from mlzo.data import data
                self._source = data(self._source)
            self._source_resolved = True
        return self._source

    def _open_epoch(self) -> None:
        source = self._resolve_source()
        if source is None:
            batches = repeat(None)
        elif callable(getattr(source, "build", None)):
            batches = source.build(epoch=self.epoch)
        elif callable(source) and not hasattr(source, "__iter__"):
            batches = source()
        else:
            batches = source
        if isinstance(batches, torch.utils.data.Dataset) and not isinstance(batches, torch.utils.data.IterableDataset):
            from mlzo.data.sources import DatasetIterable
            batches = DatasetIterable(batches)
        underlying = getattr(source, "source", None)
        raw_source = getattr(underlying, "value", None)
        if isinstance(source, Iterator) or (getattr(underlying, "kind", None) in {"native", "memory"} and isinstance(raw_source, Iterator)):
            self._data_restartable = False
        self._epoch_length = _length(batches)
        if any(schedule.inferred_horizon for schedule in self.schedulers.values()) and self._epoch_length != self._schedule_epoch_length:
            raise ConfigurationError("Data epoch length changed after inferring the schedule horizon; use an explicit total_steps schedule.")
        self._iterator = iter(batches)

    def _horizon(self, *, epochs: int | None, steps: int | None) -> int | None:
        # Class algorithms may collect experience or perform many updates. Their
        # update horizon must be explicit in decay specs rather than inferred.
        if not getattr(self.algorithm, "automatic", False) or len(self.optimizers) != 1:
            return None
        accumulation = self.config["accumulate"]
        if epochs is not None and self._epoch_length is not None:
            return epochs * math.ceil(self._epoch_length / accumulation)
        if steps is not None:
            length = self._epoch_length
            if length:
                full, short = divmod(steps, length)
                return full * math.ceil(length / accumulation) + math.ceil(short / accumulation)
            return math.ceil(steps / accumulation)
        return None

    def _configure_schedules(self, horizon: int | None) -> None:
        source = self.config.get("schedule")
        if not self.schedulers:
            self._schedule_epoch_length = self._epoch_length
        if isinstance(source, Mapping) and set(source) != set(self.optimizers):
            raise ConfigurationError("Named schedules must have exactly the named optimizer keys.")
        for name, optimizer in self.optimizers.items():
            if name in self.schedulers:
                self.schedulers[name].validate_horizon(horizon)
            else:
                self.schedulers[name] = Schedule(optimizer, source.get(name) if isinstance(source, Mapping) else source, horizon)
        if self._pending_schedulers is not None:
            self._load_schedulers(self._pending_schedulers)
            self._pending_schedulers = None

    def _result(self, value: Any, *, training: bool) -> tuple[Step, float, dict[str, float]]:
        if value is None:
            value = Step()
        elif isinstance(value, torch.Tensor):
            value = Step(loss=value)
        if not isinstance(value, Step):
            raise ConfigurationError("A learning step must return a scalar tensor, Step, or None.")
        if value.stop is not None and type(value.stop) is not bool:
            raise ConfigurationError("Step.stop must be bool or None; tensor stop masks are unsupported.")
        weight = normalization_weight(value.weight)
        loss = value.loss
        if loss is not None:
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or loss.is_complex():
                raise ConfigurationError("Step.loss must be a real scalar tensor.")
            if training and weight > 0 and not loss.requires_grad:
                raise ConfigurationError("A nonzero-weight training loss must require gradients.")
        metrics = detached_metrics(value.metrics)
        if loss is not None:
            metrics.setdefault("loss", float(loss.detach().item()))
        return value, weight, metrics

    def _parameters(self, name: str) -> list[nn.Parameter]:
        return [p for group in self.optimizers[name].param_groups for p in group["params"]]

    @staticmethod
    def _finite_gradients(parameters: list[nn.Parameter]) -> bool:
        return all(bool(torch.isfinite(p.grad._values() if p.grad.is_sparse else p.grad).all().item())
                   for p in parameters if p.grad is not None)

    def _finish_update(self, name: str, weight: float, *, finite_loss: bool = True) -> bool:
        optimizer = self.optimizers[name]
        parameters = [p for p in self._parameters(name) if p.grad is not None]
        scaler = self.scalers[name]
        if weight == 0 or not parameters:
            self.skipped_updates[name] += 1
            self.model.zero_grad(set_to_none=True)
            return False
        if scaler is not None:
            scaler.unscale_(optimizer)
        for parameter in parameters:
            parameter.grad.div_(weight)
        finite = finite_loss and self._finite_gradients(parameters)
        clip = self.config.get("grad_clip")
        if finite and clip is not None:
            if any(p.grad.is_sparse for p in parameters):
                raise UnsupportedFeatureError("Gradient clipping of sparse gradients is unsupported in this alpha.")
            norm = torch.nn.utils.clip_grad_norm_(parameters, clip)
            finite = bool(torch.isfinite(norm).item()) and self._finite_gradients(parameters)
        if finite:
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
            self.optimizer_steps[name] += 1
            self.update_events += 1
            self.schedulers[name].advance()
        else:
            self.skipped_updates[name] += 1
            if scaler is not None:
                # An overflow introduced by normalization/clipping must also
                # skip, even if unscale_ itself observed finite values.
                scaler.update(new_scale=scaler.get_scale() * scaler.get_backoff_factor())
        self.model.zero_grad(set_to_none=True)
        return finite

    def explicit_update(self, name: str, loss: torch.Tensor, *, retain_graph: bool = False) -> None:
        if not self._in_step:
            raise TrainingError("Training.update() is only legal during step(), never during evaluation or outside fit().")
        if self.config["accumulate"] != 1:
            raise ConfigurationError("Explicit Training.update() with accumulate>1 is unsupported; use accumulate=1.")
        if name not in self.optimizers:
            raise ConfigurationError(f"Unknown optimizer {name!r}; available names: {', '.join(self.optimizers)}.")
        if type(retain_graph) is not bool:
            raise ConfigurationError("retain_graph must be a boolean.")
        self._result(Step(loss=loss), training=True)
        self._explicit_calls += 1
        self.model.zero_grad(set_to_none=True)
        scaler = self.scalers[name]
        with torch.autocast(device_type=self.device.type, enabled=False):
            (scaler.scale(loss) if scaler is not None else loss).backward(retain_graph=retain_graph)
            self._finish_update(name, 1.0, finite_loss=bool(torch.isfinite(loss.detach()).item()))

    def _contribute(self, loss: torch.Tensor, weight: float) -> None:
        if len(self.optimizers) != 1:
            raise ConfigurationError("Automatic loss optimization requires one optimizer; use Training.update(name, loss) for named optimizers.")
        if self._pending_count == 0:
            self.model.zero_grad(set_to_none=True)
        name = next(iter(self.optimizers))
        self._pending_count += 1
        self._pending_weight += weight
        if not math.isfinite(self._pending_weight):
            raise ConfigurationError("Accumulated loss weight overflowed; use smaller finite normalization weights.")
        if weight > 0:
            self._pending_finite = self._pending_finite and bool(torch.isfinite(loss.detach()).item())
            numerator = loss * weight
            scaler = self.scalers[name]
            (scaler.scale(numerator) if scaler is not None else numerator).backward()
        if self._pending_count >= self.config["accumulate"]:
            self._flush()

    def _flush(self) -> None:
        if self._pending_count:
            name = next(iter(self.optimizers))
            self._finish_update(name, self._pending_weight, finite_loss=self._pending_finite)
            self._pending_count = 0
            self._pending_weight = 0.0
            self._pending_finite = True

    def _end_epoch(self) -> None:
        self._flush()
        self.epoch += 1
        self.batch_in_epoch = 0
        self._iterator = None

    def fit(self, *, epochs: int | None, steps: int | None, resume: Any = None) -> History:
        if self._in_fit:
            raise TrainingError("This engine is already running.")
        self._in_fit = True
        self._fit_thread = get_ident()
        self._cancel_requested = False
        try:
            if self._iterator is None:
                self._open_epoch()
            if steps is None:
                if self._epoch_length is None:
                    raise ConfigurationError("Infinite or no-length data requires fit(steps=...), including step-driven algorithms.")
                if epochs is None:
                    epochs = 1
            horizon = self._horizon(epochs=epochs, steps=steps)
            self._configure_schedules(horizon)
            if resume is not None:
                from mlzo.checkpoint import load_snapshot
                policy = self.config.get("checkpoint")
                if resume == "latest":
                    if policy is None:
                        raise ConfigurationError("resume='latest' requires a configured checkpoint policy root.")
                    resume = policy.to
                self.load_state_dict(load_snapshot(resume))
            if self._iterator is None:
                self._open_epoch()
            self.algorithm.model.train()
            self.history.state = "running"
            self._emit("running")
            stop = False
            while not stop and not self._cancel_requested:
                if (steps is not None and self.iteration >= steps) or (epochs is not None and self.epoch >= epochs):
                    break
                if self._iterator is None:
                    self._open_epoch()
                try:
                    batch = next(self._iterator)
                except StopIteration:
                    if self.batch_in_epoch == 0:
                        if steps is not None:
                            raise TrainingError("Data source is empty or exhausted before the total steps target; use a reusable source or factory.")
                        self._end_epoch()
                    else:
                        self._end_epoch()
                    self._maybe_checkpoint()
                    continue
                batch = move_batch(batch, self.device)
                self._explicit_calls = 0
                self._in_step = True
                try:
                    with self._autocast():
                        value = self.algorithm.step(batch)
                finally:
                    self._in_step = False
                if self._explicit_calls and (isinstance(value, torch.Tensor) or (isinstance(value, Step) and value.loss is not None)):
                    raise TrainingError(
                        "step() called Training.update() and also returned an automatic loss. "
                        "Already-completed explicit updates cannot be rolled back; recover from a committed checkpoint."
                    )
                result, weight, metrics = self._result(value, training=True)
                if result.loss is not None:
                    self._contribute(result.loss, weight)
                self.iteration += 1
                self.batch_in_epoch += 1
                stop = result.stop is True
                if self._epoch_length is not None and self.batch_in_epoch >= self._epoch_length:
                    self._end_epoch()
                if stop:
                    self._flush()
                self.history.append(metrics, counters=self.counters())
                self._emit("iteration", metrics=metrics)
                self._maybe_checkpoint()
            self._flush()
            self._maybe_checkpoint()
            self.history.state = "cancelled" if self._cancel_requested else "completed"
            self.history.counters = self.counters()
            self._emit(self.history.state)
            return self.history
        except BaseException:
            # In-flight gradients cannot be resumed from a safe snapshot. Keep
            # completed updates/counters visible but require checkpoint recovery.
            self.model.zero_grad(set_to_none=True)
            self._pending_count = 0
            self._pending_weight = 0.0
            self._pending_finite = True
            self._failed_boundary = True
            self.history.state = "failed"
            self.history.counters = self.counters()
            raise
        finally:
            self._in_step = False
            self._in_fit = False
            self._fit_thread = None
            self.history.notes = list(self.notes)

    def evaluate(self, data: Any, *, step: Any = None) -> dict[str, float]:
        if self._in_fit:
            raise TrainingError("Evaluation during fit() is unsupported.")
        if step is None and type(self.algorithm).evaluate is Training.evaluate:
            raise ConfigurationError("Custom training requires run.evaluate(data, step=evaluator) or Training.evaluate().")
        if isinstance(data, (str, os.PathLike)):
            from mlzo.data import data as describe_data
            data = describe_data(data)
        if callable(getattr(data, "build", None)):
            data = data.build(epoch=self.epoch)
        elif callable(data) and not hasattr(data, "__iter__"):
            data = data()
        if isinstance(data, torch.utils.data.Dataset) and not isinstance(data, torch.utils.data.IterableDataset):
            from mlzo.data.sources import DatasetIterable
            data = DatasetIterable(data)
        if _length(data) is None:
            raise ConfigurationError("Evaluation requires finite data with a length in this alpha.")
        modes = [(module, module.training) for module in self.algorithm.model.modules()]
        totals: dict[str, float] = {}
        weights: dict[str, float] = {}
        try:
            self.algorithm.model.eval()
            with torch.no_grad():
                for batch in data:
                    with self._autocast():
                        value = (step(self.algorithm.model, move_batch(batch, self.device)) if step is not None
                                 else self.algorithm.evaluate(move_batch(batch, self.device)))
                    result, weight, metrics = self._result(value, training=False)
                    for name, value in metrics.items():
                        contribution = weight if name == "loss" and result.loss is not None else 1.0
                        if contribution:
                            totals[name] = totals.get(name, 0.0) + value * contribution
                            weights[name] = weights.get(name, 0.0) + contribution
                    if result.stop is True:
                        break
        finally:
            # Assign each flag directly; recursive train(mode) would overwrite
            # intentionally frozen children when their parent is restored.
            for module, mode in modes:
                module.training = mode
        return {name: total / weights[name] for name, total in totals.items()}

    def _safe(self, *, restoring: bool = False) -> None:
        if self._in_fit and get_ident() != self._fit_thread:
            raise UnsupportedFeatureError("Concurrent engine snapshots are unsupported; request a policy checkpoint at a safe boundary.")
        if self._in_step or self._pending_count:
            raise TrainingError("A checkpoint requires an iteration boundary with no pending accumulated gradients.")
        if self._failed_boundary and not restoring:
            raise TrainingError("A failed iteration cannot be checkpointed; resume the last committed checkpoint or create a new run.")

    def state_dict(self) -> dict[str, Any]:
        self._safe()
        if not self.schedulers:
            self._configure_schedules(None)
        return restricted_copy({
            "schema_version": 1,
            "model": self.model.state_dict(),
            "optimizers": {name: value.state_dict() for name, value in self.optimizers.items()},
            "schedulers": {name: value.state_dict() for name, value in self.schedulers.items()},
            "scalers": {name: value.state_dict() if value is not None else None for name, value in self.scalers.items()},
            "counters": self.counters(), "rng": rng_state(self.device), "algorithm_state": self.algorithm.state,
            "optimizer_parameters": {name: self._parameter_names(name) for name in self.optimizers},
            "optimizer_types": {name: f"{type(value).__module__}.{type(value).__qualname__}" for name, value in self.optimizers.items()},
            "precision": self.precision,
            "data_resume": {"guarantee": "epoch_restart" if self._data_restartable else "unavailable_one_shot", "epoch": self.epoch, "batch_in_epoch": self.batch_in_epoch,
                            "description": "Model/update state is restored. A partial data epoch restarts; native sampler/worker state is not serialized."},
            "last_checkpoint_event": self._last_checkpoint_event,
        })

    def _parameter_names(self, name: str) -> list[list[str]]:
        names = {id(parameter): path for path, parameter in self.model.named_parameters()}
        return [[names[id(p)] for p in group["params"]] for group in self.optimizers[name].param_groups]

    def _load_schedulers(self, state: Mapping[str, Any]) -> None:
        if set(state) != set(self.schedulers):
            raise ConfigurationError("Checkpoint scheduler names differ from this run.")
        for name, value in state.items():
            self.schedulers[name].load_state_dict(value)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._safe(restoring=True)
        if state.get("schema_version") != 1:
            raise ConfigurationError("Unsupported training snapshot schema.")
        if set(state["optimizers"]) != set(self.optimizers):
            raise ConfigurationError("Checkpoint optimizer names differ from this run.")
        if state["data_resume"]["guarantee"] == "unavailable_one_shot" and state["counters"]["iteration"]:
            raise UnsupportedFeatureError("One-shot data iterators cannot be resumed. Use a rebuildable data factory or native loader.")
        for name in self.optimizers:
            if state["optimizer_types"][name] != f"{type(self.optimizers[name]).__module__}.{type(self.optimizers[name]).__qualname__}":
                raise ConfigurationError(f"Checkpoint optimizer type differs for {name!r}.")
            if state["optimizer_parameters"][name] != self._parameter_names(name):
                raise ConfigurationError(f"Checkpoint parameter ownership differs for optimizer {name!r}.")
        if state["precision"] != self.precision:
            raise ConfigurationError("Checkpoint precision differs; scaler/precision migration is unsupported.")
        self.model.load_state_dict(state["model"])
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state["optimizers"][name])
            scaler_state = state["scalers"][name]
            if self.scalers[name] is not None:
                if scaler_state is None:
                    raise ConfigurationError("Checkpoint is missing FP16 scaler state.")
                self.scalers[name].load_state_dict(scaler_state)
        if self.schedulers:
            self._load_schedulers(state["schedulers"])
        else:
            self._pending_schedulers = state["schedulers"]
        counters = state["counters"]
        for name in ("iteration", "epoch", "update_events"):
            setattr(self, name, int(counters[name]))
        self.optimizer_steps = dict(counters["optimizer_steps"])
        self.skipped_updates = dict(counters["skipped_updates"])
        self.history = History(self.run.id, max_records=self.history.max_records)
        self.algorithm.state = move_batch(restricted_copy(state["algorithm_state"]), self.device)
        self._last_checkpoint_event = int(state.get("last_checkpoint_event", 0))
        self.batch_in_epoch = 0
        self._iterator = None
        self.history.counters = self.counters()
        if counters.get("batch_in_epoch", 0):
            self.notes.append("Resume restarted a partial data epoch; exact sampler/worker replay is not provided.")
        restore_rng(state["rng"], self.device)
        self.model.zero_grad(set_to_none=True)
        self._failed_boundary = False

    def checkpoint(self, *, to: Any = None) -> Any:
        self._safe()
        from mlzo.checkpoint import save_snapshot
        policy = self.config.get("checkpoint")
        destination = to if to is not None else getattr(policy, "to", None)
        if destination is None:
            raise ConfigurationError("checkpoint() requires to= or a configured checkpoint policy.")
        if policy is not None and getattr(policy, "mode", "full") != "full":
            raise UnsupportedFeatureError("Training checkpoints use mode='full'; use run.export() for weights-only inference artifacts.")
        reference = save_snapshot(self.state_dict(), to=destination, run_id=self.run.id,
                                  metadata={"counters": self.counters(), "resume_guarantee": "epoch_restart" if self._data_restartable else "unavailable_one_shot"},
                                  keep_last=getattr(policy, "keep_last", None))
        self._emit("checkpoint", reference=reference.to_dict() if hasattr(reference, "to_dict") else str(reference))
        return reference

    def _maybe_checkpoint(self) -> None:
        policy = self.config.get("checkpoint")
        if policy is None or self._pending_count or self._in_step:
            return
        every = policy.every
        if self.update_events // every > self._last_checkpoint_event // every:
            previous = self._last_checkpoint_event
            self._last_checkpoint_event = self.update_events
            try:
                self.checkpoint()
            except BaseException:
                self._last_checkpoint_event = previous
                raise
