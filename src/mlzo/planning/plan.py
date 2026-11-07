"""Side-effect-free model inspection and conservative single-device planning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import torch
from torch import nn

from mlzo.errors import ConfigurationError, UnsupportedFeatureError
from mlzo.model import Lazy, module_paths, placement_map

from .devices import device_name, local_inventory, remote_inventory
from .pipeline import PipelineConfig


def _metadata(value: Any, depth: int = 0) -> Any:
    """Describe configuration without invoking factories, reprs, or iterators."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if depth > 6:
        return {"type": type(value).__qualname__}
    if isinstance(value, torch.Tensor):
        return {
            "type": type(value).__qualname__,
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, Mapping):
        return {str(key): _metadata(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata(item, depth + 1) for item in value]
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.optim.Optimizer):
        return {
            "type": type(value).__qualname__,
            "kind": "instance",
            "groups": [
                {
                    str(key): _metadata(item, depth + 1)
                    for key, item in group.items() if key != "params"
                }
                for group in value.param_groups
            ],
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": type(value).__qualname__,
            **{
                field.name: _metadata(getattr(value, field.name), depth + 1)
                for field in fields(value)
            },
        }
    if callable(value):
        return {
            "kind": "callable",
            "module": getattr(value, "__module__", None),
            "qualname": getattr(value, "__qualname__", type(value).__qualname__),
        }
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


@dataclass(frozen=True, slots=True)
class Plan:
    device: str
    precision: str
    target: str
    strategy: str
    validation_state: str
    inventory: tuple[dict[str, Any], ...]
    placements: dict[str, int]
    aliases: dict[str, str]
    model: dict[str, Any]
    data: dict[str, Any]
    optimizer: Any
    schedule: Any
    execution: dict[str, Any]
    decisions: tuple[str, ...]
    limitations: tuple[str, ...]
    runtime_checks: tuple[str, ...]
    schema_version: int = 1

    @property
    def validation(self) -> str:
        return self.validation_state

    @property
    def validated(self) -> bool:
        return self.validation_state == "validated"

    @property
    def memory(self) -> dict[str, Any]:
        return {"status": "unknown", "parameter_bytes": None, "activation_bytes": None,
                "optimizer_bytes": None, "total_bytes": None}

    def to_dict(self) -> dict[str, Any]:
        result = {field.name: _metadata(getattr(self, field.name)) for field in fields(self)}
        result["memory"] = self.memory
        return result

    def explain(self) -> str:
        known = self.model["known_parameter_count"]
        total = self.model["parameter_count"]
        parameter_text = "unknown" if total is None else f"{total:,}"
        lines = [
            f"MLZO single-device plan for {self.target}",
            f"Device: {self.device}; precision: {self.precision}; validation: {self.validation_state}",
            f"Parameters: {parameter_text} total, {known:,} already known; memory: unknown.",
            *self.decisions,
        ]
        if self.placements:
            lines.append("Placement: " + ", ".join(f"{path or '<root>'}=gpu:{gpu}" for path, gpu in self.placements.items()))
        lines.extend(f"Runtime check: {check}" for check in self.runtime_checks)
        lines.extend(f"Limitation: {limit}" for limit in self.limitations)
        return "\n".join(lines)


def _model_info(algorithm: Any) -> tuple[dict[str, Any], dict[str, int], dict[str, str]]:
    model = algorithm if isinstance(algorithm, nn.Module) else None
    if model is None and not callable(algorithm):
        model = getattr(algorithm, "model", None)
    if not isinstance(model, nn.Module):
        if not callable(algorithm):
            raise ConfigurationError("Run.algorithm_input must be a model, Training instance, or training factory.")
        return (
            {
                "kind": "training_factory",
                "factory": _metadata(algorithm),
                "parameter_count": None,
                "known_parameter_count": 0,
                "trainable_parameter_count": None,
                "known_module_count": 0,
                "lazy_modules": {},
                "requires_construction": True,
            },
            {},
            {},
        )
    canonical: dict[int, str] = {}
    aliases: dict[str, str] = {}
    lazy: dict[str, str] = {}
    parameters: set[int] = set()
    known_count = trainable_count = 0
    unknown = False
    for path, node in module_paths(model):
        if id(node) in canonical:
            aliases[path] = canonical[id(node)]
        else:
            canonical[id(node)] = path
        if isinstance(node, Lazy) and node.state != "ready":
            lazy[path] = node.state
            unknown = True
            if node.state == "failed":
                raise ConfigurationError(
                    f"Lazy module {path or '<root>'!r} previously failed; create a new recipe to retry."
                ) from node.error
        for parameter in node._parameters.values():
            if parameter is None or id(parameter) in parameters:
                continue
            parameters.add(id(parameter))
            try:
                count = parameter.numel()
            except (ValueError, RuntimeError):
                unknown = True
                continue
            known_count += count
            if parameter.requires_grad:
                trainable_count += count
    return (
        {
            "kind": "lazy" if isinstance(model, Lazy) else "module",
            "type": f"{type(model).__module__}.{type(model).__qualname__}",
            "parameter_count": None if unknown else known_count,
            "known_parameter_count": known_count,
            "trainable_parameter_count": None if unknown else trainable_count,
            "known_module_count": len(canonical),
            "lazy_modules": lazy,
            "requires_construction": unknown,
        },
        placement_map(model),
        aliases,
    )


def _data_info(data: Any) -> dict[str, Any]:
    info: dict[str, Any] = {"type": type(data).__qualname__ if data is not None else None, "length": None}
    # Never call len() on user datasets/pipelines. Their implementation may
    # construct a source or advance an iterator.
    if type(data) in {list, tuple, dict, range}:
        info["length"] = len(data)
    elif isinstance(data, torch.Tensor) and data.ndim:
        info["length"] = data.shape[0]
    elif isinstance(data, torch.utils.data.DataLoader):
        info["batch_size"] = data.batch_size
        info["num_workers"] = data.num_workers
    if callable(data):
        info["source"] = _metadata(data)
    return info


def _strategy(parallel: Any) -> None:
    if isinstance(parallel, PipelineConfig):
        raise UnsupportedFeatureError(
            f"Pipeline execution ({parallel.stages} stages, {parallel.microbatches} microbatches) "
            "is unavailable in MLZO 0.1.0a1. Use parallel='auto' or 'single' on one device."
        )
    if parallel is None or (isinstance(parallel, str) and parallel in {"auto", "single", "none"}):
        return
    description = repr(parallel) if isinstance(parallel, str) else type(parallel).__qualname__
    raise UnsupportedFeatureError(
        f"Parallel strategy {description} is unavailable in MLZO 0.1.0a1. "
        "Data, pipeline, sharded, and tensor parallel execution are deferred; use 'auto' or 'single'."
    )


def _select_device(
    requested: str | None, gpu: int | None, inventory: list[dict[str, Any]],
    remote: bool, checks: list[str], decisions: list[str],
) -> tuple[str, dict[str, Any] | None]:
    if requested is not None:
        if requested == "cuda":
            requested = f"cuda:{gpu if gpu is not None else 0}"
        if gpu is not None and requested != f"cuda:{gpu}":
            raise ConfigurationError(
                f"Explicit device={requested!r} conflicts with placement gpu={gpu}. "
                "Choose the same CUDA device or remove the GPU annotation."
            )
    elif gpu is not None:
        requested = f"cuda:{gpu}"
    else:
        preferred = next((item for kind in ("cuda", "mps", "cpu") for item in inventory if item["type"] == kind), None)
        requested = preferred["device"] if preferred is not None else "auto"
        decisions.append(f"Automatic device selection chose {requested}.")
    selected = next((item for item in inventory if item["device"] == requested), None)
    if selected is None:
        if remote:
            checks.append(f"Validate device {requested!r} and its logical GPU mapping on the assigned worker; remote inventory is advisory.")
        else:
            available = ", ".join(item["device"] for item in inventory)
            raise ConfigurationError(
                f"Requested device {requested!r} is unavailable locally. Available devices: {available}. "
                "GPU IDs are logical IDs of the selected target; pass on=cluster for remote GPUs."
            )
    return requested, selected


def _precision(requested: Any, device: str, selected: dict[str, Any] | None, checks: list[str]) -> str:
    if not isinstance(requested, str) or requested not in {"auto", "fp32", "bf16", "fp16"}:
        raise ConfigurationError("precision must be 'auto', 'fp32', 'bf16', or 'fp16'.")
    kind = device.split(":")[0]
    capability = selected.get("bf16") if selected is not None else None
    if requested == "auto":
        return "bf16" if kind == "cuda" and capability is True else "fp32"
    if requested == "fp16" and kind == "cpu":
        raise ConfigurationError("FP16 training on CPU is unsupported; use precision='fp32' or 'bf16'.")
    if requested in {"bf16", "fp16"} and kind == "mps":
        raise UnsupportedFeatureError("MPS mixed precision is unavailable in this alpha; use precision='fp32'.")
    if requested == "bf16" and kind == "cuda" and capability is False:
        raise ConfigurationError(f"Device {device} does not advertise BF16 support; use precision='fp32' or 'fp16'.")
    if requested == "bf16" and (kind == "cpu" or capability is None):
        checks.append("Validate BF16 autocast and operator support on the execution worker.")
    if requested == "fp16" and selected is None:
        checks.append("Validate FP16 autocast and gradient scaling on the execution worker.")
    return requested


def plan_run(run: Any, *, on: Any = None, parallel: Any = "auto", example: Any = None) -> Plan:
    """Inspect recipes and metadata without creating weights or touching batches."""
    _strategy(parallel)
    config = getattr(run, "config", None)
    if not isinstance(config, Mapping):
        raise ConfigurationError("Run.config must be a mapping.")
    if config.get("parallel") is not None:
        _strategy(config["parallel"])
    if config.get("strategy") is not None:
        _strategy(config["strategy"])
    accumulation = config.get("accumulate", 1)
    if isinstance(accumulation, bool) or not isinstance(accumulation, int) or accumulation < 1:
        raise ConfigurationError("accumulate must be a positive integer.")
    compilation = config.get("compile", False)
    if (
        compilation is not None
        and not isinstance(compilation, bool)
        and not (isinstance(compilation, str) and compilation == "auto")
    ):
        raise ConfigurationError("compile must be True, False, 'auto', or None.")
    model, placements, aliases = _model_info(getattr(run, "algorithm_input", None))
    gpu_ids = set(placements.values())
    if len(gpu_ids) > 1:
        raise UnsupportedFeatureError(
            f"Placements require GPUs {sorted(gpu_ids)}, but MLZO 0.1.0a1 supports one device. "
            "Use one GPU assignment; placement does not enable pipeline execution."
        )
    requested = device_name(config["device"]) if config.get("device") is not None else None
    target = on if on is not None else getattr(run, "_remote_cluster", None)
    if isinstance(target, (str, torch.device)):
        on_device = device_name(target)
        if requested is not None and requested != on_device:
            raise ConfigurationError(f"on={on_device!r} conflicts with device={requested!r}.")
        requested = on_device
        target = None
    remote = target is not None
    if remote and not any(callable(getattr(target, method, None)) for method in ("workers", "status")):
        raise ConfigurationError("on= must be a Cluster handle with workers()/status(), or a local device.")
    inventory, inventory_notes = remote_inventory(target) if remote else local_inventory()
    decisions = ["Use single-device execution; no process group, model partitioning, or data sharding."]
    checks: list[str] = []
    limitations = [
        "Model, activation, and optimizer memory are unknown; planning does not execute forward or probe memory.",
        "Distributed strategies, automatic recovery, and provisioning are unavailable in this alpha.",
        *inventory_notes,
    ]
    if model["requires_construction"]:
        checks.append("Construct deferred model/training factories on the execution worker and validate final parameters before optimizer binding.")
    if remote:
        checks.append("Revalidate worker availability, device assignment, precision, code, and data access at execution; coordinator hardware is not used.")
    if example is not None:
        decisions.append("The supplied example is recorded as present and is not executed or consumed.")
    gpu = next(iter(gpu_ids)) if gpu_ids else None
    device, selected = _select_device(requested, gpu, inventory, remote, checks, decisions)
    precision = _precision(config.get("precision", "auto"), device, selected, checks)
    decisions.append(f"Precision {config.get('precision', 'auto')!r} resolves to {precision}; auto selects BF16 only on capable CUDA, FP32 otherwise.")
    optimizer = config.get("optimizer", "adamw")
    if isinstance(optimizer, torch.optim.Optimizer):
        checks.append("Verify that the existing optimizer references the final execution parameters.")
    execution = {key: _metadata(config.get(key, default)) for key, default in {
        "accumulate": 1, "compile": False, "matmul": "auto", "memory": None,
        "grad_clip": None, "lr": None, "seed": None, "checkpoint": None,
    }.items()}
    execution["example_provided"] = example is not None
    if execution["compile"] not in {False, None}:
        checks.append("Validate the requested compilation mode in the execution engine; planning does not compile code.")
    if remote and (not inventory or selected is None):
        validation = "awaiting_runtime_validation"
    else:
        validation = "partially_validated" if checks or inventory_notes else "validated"
    return Plan(
        device=device,
        precision=precision,
        target="remote" if remote else "local",
        strategy="single",
        validation_state=validation,
        inventory=tuple(inventory),
        placements=placements,
        aliases=aliases,
        model=model,
        data=_data_info(getattr(run, "data", None)),
        optimizer=_metadata(optimizer),
        schedule=_metadata(config.get("schedule")),
        execution=execution,
        decisions=tuple(decisions),
        limitations=tuple(limitations),
        runtime_checks=tuple(checks),
    )
