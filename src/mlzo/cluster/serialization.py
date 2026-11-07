"""Versioned JSON recipes for the remote execution worker.

``validate_run_spec`` only inspects JSON structure. It never imports application
modules. ``deserialize_run`` belongs in the execution subprocess: resolving an
importable reference imports trusted application code, but never calls a model,
training, or data factory here. There is no pickle or class-name dispatch.
"""

from __future__ import annotations

import inspect
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path, PurePath, PurePosixPath
from typing import Any

import torch
from torch import nn

from mlzo.checkpoint.policy import CheckpointPolicy
from mlzo.data import DataPipeline
from mlzo.errors import ConfigurationError
from mlzo.model import FactoryRef, Lazy
from mlzo.training import OptimizerSpec, Run, ScheduleSpec, Training, train

__all__ = ["serialize_run", "deserialize_run", "validate_run_spec", "collect_data_paths"]

RUN_FORMAT = "mlzo.run"
RUN_VERSION = 1
_TAG = "__mlzo__"
_MAX_DEPTH = 64
_MAX_NODES = 100_000
_MAX_JSON_BYTES = 8 * 1024 * 1024
_CONFIG_KEYS = frozenset(inspect.signature(train).parameters) - {"model_or_training", "data"}
_DTYPES = {
    name: getattr(torch, name)
    for name in (
        "bool", "uint8", "int8", "int16", "int32", "int64", "float16", "float32",
        "float64", "bfloat16", "complex32", "complex64", "complex128",
        "float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz",
    )
    if isinstance(getattr(torch, name, None), torch.dtype)
}
_GUIDANCE = "Use an unmaterialized root Lazy or an importable zero-argument training factory."


def _error(path: str, message: str) -> ConfigurationError:
    return ConfigurationError(f"{path}: {message}")


def _relative_path(value: Any, path: str) -> str:
    if (
        type(value) is not str or not value or "\\" in value or ":" in value
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or PurePosixPath(value).is_absolute()
    ):
        raise _error(path, "Data archive paths must be safe, nonempty relative POSIX paths.")
    return value


def _absolute(value: str | PurePath) -> Path:
    # Do not inspect files or follow symlinks during recipe transport.
    return Path(value).expanduser().absolute()


def _data_mapping(value: Mapping | None) -> dict[Path, str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise _error("data_paths", "Pass ProjectArtifact.data_mapping, a mapping of local paths to archive paths.")
    result: dict[Path, str] = {}
    for original, relative in value.items():
        if not isinstance(original, (str, PurePath)):
            raise _error("data_paths", "Original data paths must be strings or pathlib paths.")
        local = _absolute(original)
        if local in result:
            raise _error("data_paths", f"Duplicate normalized data path {str(local)!r}.")
        result[local] = _relative_path(relative, f"data_paths[{str(original)!r}]")
    return result


def _remap(value: str | PurePath, mapping: dict[Path, str] | None, path: str) -> str:
    if mapping is None:
        return str(value)
    local = _absolute(value)
    for original in sorted(mapping, key=lambda item: len(item.parts), reverse=True):
        if local == original:
            return mapping[original]
        if local.is_relative_to(original):
            return str(PurePosixPath(mapping[original]) / local.relative_to(original).as_posix())
    raise _error(path, f"Local data path {str(local)!r} was not declared in the project artifact's data_mapping.")


def collect_data_paths(run: Run) -> tuple[Path, ...]:
    """Discover explicit local data descriptors without reading or building data.

    Factory closures, factory arguments, and strings inside in-memory records
    are intentionally not guessed to be paths. Declare such files separately.
    """
    if not isinstance(run, Run):
        raise _error("run", "Expected a training Run.")
    found: set[Path] = set()
    active: set[int] = set()

    def visit(value: Any, depth: int = 0) -> None:
        if depth > _MAX_DEPTH:
            raise _error("run.data", "Data descriptors exceed the transport nesting limit.")
        if type(value) is DataPipeline:
            found.update(_absolute(path) for path in value.local_paths())
        elif isinstance(value, (Mapping, tuple, list)):
            object_id = id(value)
            if object_id in active:
                raise _error("run.data", "Cyclic data descriptors cannot be transported.")
            active.add(object_id)
            try:
                for item in value.values() if isinstance(value, Mapping) else value:
                    visit(item, depth + 1)
            finally:
                active.remove(object_id)

    if isinstance(run.data, (str, PurePath)):
        if not str(run.data).startswith("s3://"):
            found.add(_absolute(run.data))
    else:
        visit(run.data)
    return tuple(sorted(found, key=str))


class _Encoder:
    def __init__(self, mapping: dict[Path, str] | None) -> None:
        self.mapping = mapping
        self.active: set[int] = set()
        self.lazy_ids: dict[int, str] = {}
        self.nodes = 0

    def encode_value(self, value: Any, path: str, depth: int = 0) -> Any:
        self.nodes += 1
        if depth > _MAX_DEPTH or self.nodes > _MAX_NODES:
            raise _error(path, "Recipe exceeds the transport nesting or item limit.")
        if value is None or type(value) in {bool, int, str}:
            return value
        if type(value) is float:
            if not math.isfinite(value):
                raise _error(path, "Only finite floats are JSON transportable.")
            return value
        object_id = id(value)
        # An active Lazy is a cycle even though it already has a recipe ID.
        if object_id in self.active:
            raise _error(path, "Cyclic recipes cannot be transported; put construction in an importable factory.")
        if type(value) is Lazy and object_id in self.lazy_ids:
            return {_TAG: "lazy_ref", "id": self.lazy_ids[object_id]}
        self.active.add(object_id)
        try:
            return self._encode_recipe_value(value, path, depth)
        except ConfigurationError as exc:
            if (
                str(exc).startswith(f"{path}:")
                or str(exc).startswith(f"{path}[")
                or str(exc).startswith(f"{path}.")
            ):
                raise
            raise _error(path, str(exc)) from exc
        finally:
            self.active.remove(object_id)

    def _encode_recipe_value(self, value: Any, path: str, depth: int) -> Any:
        def encode_child(item: Any, suffix: str) -> Any:
            return self.encode_value(item, path + suffix, depth + 1)

        if type(value) is Lazy:
            if value.state != "unmaterialized":
                raise _error(path, f"Lazy is {value.state!r}; remote submission requires a fresh unmaterialized recipe. {_GUIDANCE}")
            if value._modules or value._parameters or value._buffers:
                raise _error(path, f"Lazy has registered model state outside its constructor recipe. {_GUIDANCE}")
            recipe_id = f"l{len(self.lazy_ids)}"
            self.lazy_ids[id(value)] = recipe_id
            return {
                _TAG: "lazy",
                "id": recipe_id,
                "factory": encode_child(value.factory, ".factory"),
                "args": encode_child(value.args, ".args"),
                "kwargs": encode_child(value.kwargs, ".kwargs"),
                "gpu": value.__dict__.get("_mlzo_gpu"),
                "training": value.training,
            }
        if isinstance(value, nn.Module):
            raise _error(path, f"Eager model instances cannot be serialized. {_GUIDANCE}")
        if isinstance(value, torch.optim.Optimizer):
            raise _error(path, "Eager optimizer instances cannot be serialized. Use opt(name_or_importable_factory, ...) with Lazy or an importable training factory.")
        if isinstance(value, Training):
            raise _error(path, f"Custom Training instances cannot be serialized. {_GUIDANCE}")
        if type(value) is FactoryRef:
            return {_TAG: "factory_ref", **value.to_dict()}
        if type(value) is DataPipeline:
            if self.mapping is not None:
                value = value.rewrite_local_paths(lambda local: _remap(local, self.mapping, path + ".source.value"))
            # The identity callback keeps exact option paths in encoder errors.
            # to_spec only describes configuration; it never builds the source.
            return {_TAG: "data_pipeline", "spec": encode_child(value.to_spec(lambda item: item), ".spec")}
        if type(value) is OptimizerSpec:
            return {
                _TAG: "optimizer",
                "name": encode_child(value.name, ".name"),
                "params": encode_child(value.params, ".params"),
                "kwargs": encode_child(value.kwargs, ".kwargs"),
            }
        if type(value) is ScheduleSpec:
            return {_TAG: "schedule", "spec": encode_child(value.to_dict(), ".spec")}
        if type(value) is CheckpointPolicy:
            return {_TAG: "checkpoint", "spec": encode_child(value.to_dict(), ".spec")}
        if isinstance(value, PurePath):
            return {_TAG: "path", "value": str(value)}
        if isinstance(value, torch.dtype):
            name = next((name for name, dtype in _DTYPES.items() if value is dtype), None)
            if name is None:
                raise _error(path, f"Unsupported torch dtype {value}.")
            return {_TAG: "dtype", "name": name}
        if isinstance(value, torch.device):
            return {_TAG: "device", "value": str(value)}
        if isinstance(value, Mapping):
            items = []
            for key, item in value.items():
                if type(key) is not str:
                    raise _error(path, f"Mappings require string keys; got key {key!r}.")
                items.append([key, encode_child(item, f"[{key!r}]")])
            # Every user mapping is escaped, including mappings that look tagged.
            return {_TAG: "mapping", "items": items}
        if type(value) in {tuple, list}:
            items = [encode_child(item, f"[{index}]") for index, item in enumerate(value)]
            return {_TAG: "tuple", "items": items} if type(value) is tuple else items
        if callable(value):
            try:
                reference = FactoryRef.from_callable(value)
            except ConfigurationError as exc:
                raise _error(path, f"{exc} Use a named importable factory or callable; local closures and __main__ are unsupported.") from exc
            return {_TAG: "callable", **reference.to_dict()}
        raise _error(path, f"Unsupported {type(value).__module__}.{type(value).__qualname__}. Use JSON values, specification objects, Lazy, or an importable factory; use a declared data path for tensors/datasets.")


def _fields(value: Any, path: str, expected: set[str]) -> None:
    if type(value) is not dict or set(value) != expected:
        raise _error(path, f"Expected fields {', '.join(sorted(expected))}.")


def _mapping_fields(value: Any, path: str) -> dict[str, Any]:
    _fields(value, path, {_TAG, "items"})
    if value[_TAG] != "mapping" or type(value["items"]) is not list:
        raise _error(path, "Expected a tagged mapping.")
    result: dict[str, Any] = {}
    for entry in value["items"]:
        if type(entry) is not list or len(entry) != 2 or type(entry[0]) is not str:
            raise _error(path, "Mapping items must be [string, value] pairs.")
        if entry[0] in result:
            raise _error(path, f"Duplicate mapping key {entry[0]!r}.")
        result[entry[0]] = entry[1]
    return result


def _factory_syntax(value: dict, path: str) -> None:
    for key in ("module", "qualname"):
        name = value[key]
        if type(name) is not str or not name or any(not part.isidentifier() for part in name.split(".")):
            raise _error(path + "." + key, "Expected a dotted importable identifier.")
    if value["module"] in {"__main__", "__mp_main__"}:
        raise _error(path, "__main__ factories cannot be transported; use an importable project module.")


class _Validator:
    def __init__(self) -> None:
        self.nodes = 0
        self.lazy_ids: set[str] = set()
        self.active: set[int] = set()

    def validate_value(self, value: Any, path: str, depth: int = 0) -> None:
        self.nodes += 1
        if depth > _MAX_DEPTH or self.nodes > _MAX_NODES:
            raise _error(path, "Recipe exceeds the transport nesting or item limit.")
        if value is None or type(value) in {bool, int, str}:
            return
        if type(value) is float:
            if not math.isfinite(value):
                raise _error(path, "Only finite floats are allowed.")
            return
        object_id = id(value)
        if object_id in self.active:
            raise _error(path, "Cyclic transport values are invalid.")
        self.active.add(object_id)
        try:
            self._validate_recipe_value(value, path, depth)
        finally:
            self.active.remove(object_id)

    def _validate_recipe_value(self, value: Any, path: str, depth: int) -> None:
        def validate_child(item: Any, suffix: str) -> None:
            self.validate_value(item, path + suffix, depth + 1)

        if type(value) is list:
            for index, item in enumerate(value):
                validate_child(item, f"[{index}]")
            return
        if type(value) is not dict or type(value.get(_TAG)) is not str:
            raise _error(path, "Expected a JSON value or a recognized tagged recipe.")
        tag = value[_TAG]
        if tag == "mapping":
            for key, item in _mapping_fields(value, path).items():
                validate_child(item, f"[{key!r}]")
        elif tag == "tuple":
            _fields(value, path, {_TAG, "items"})
            if type(value["items"]) is not list:
                raise _error(path, "Tuple items must be a list.")
            for index, item in enumerate(value["items"]):
                validate_child(item, f"[{index}]")
        elif tag in {"factory_ref", "callable"}:
            _fields(value, path, {_TAG, "module", "qualname"})
            _factory_syntax(value, path)
        elif tag in {"lazy", "lazy_ref"}:
            expected = (
                {_TAG, "id"} if tag == "lazy_ref"
                else {_TAG, "id", "factory", "args", "kwargs", "gpu", "training"}
            )
            _fields(value, path, expected)
            recipe_id = value["id"]
            if type(recipe_id) is not str or not re.fullmatch(r"l[0-9]+", recipe_id):
                raise _error(path + ".id", "Invalid Lazy recipe identity.")
            if tag == "lazy_ref":
                if recipe_id not in self.lazy_ids:
                    raise _error(path, "Lazy references must refer to an earlier complete recipe.")
                return
            if recipe_id in self.lazy_ids:
                raise _error(path, "Duplicate Lazy recipe identity.")
            if type(value["factory"]) is not dict or value["factory"].get(_TAG) != "callable":
                raise _error(path + ".factory", "Lazy requires an importable callable reference.")
            if type(value["args"]) is not dict or value["args"].get(_TAG) != "tuple":
                raise _error(path + ".args", "Lazy args must be a tagged tuple.")
            _mapping_fields(value["kwargs"], path + ".kwargs")
            if value["gpu"] is not None and (type(value["gpu"]) is not int or value["gpu"] < 0):
                raise _error(path + ".gpu", "GPU placement must be a nonnegative integer or None.")
            if type(value["training"]) is not bool:
                raise _error(path + ".training", "Lazy training mode must be a boolean.")
            for key in ("factory", "args", "kwargs"):
                validate_child(value[key], "." + key)
            if recipe_id in self.lazy_ids:
                raise _error(path, "Duplicate nested Lazy recipe identity.")
            # Aliases may reference this recipe only after its children validate.
            self.lazy_ids.add(recipe_id)
        elif tag == "optimizer":
            _fields(value, path, {_TAG, "name", "params", "kwargs"})
            _mapping_fields(value["kwargs"], path + ".kwargs")
            for key in ("name", "params", "kwargs"):
                validate_child(value[key], "." + key)
        elif tag in {"data_pipeline", "schedule", "checkpoint"}:
            _fields(value, path, {_TAG, "spec"})
            members = _mapping_fields(value["spec"], path + ".spec")
            required = {
                "data_pipeline": {"version", "source", "operations", "loader"},
                "schedule": {"kind", "warmup", "total_steps", "end_factor", "gamma", "factor"},
                "checkpoint": {"to", "every", "keep_last", "mode"},
            }[tag]
            if set(members) != required:
                raise _error(path + ".spec", f"Unexpected {tag} specification fields.")
            validate_child(value["spec"], ".spec")
        elif tag in {"path", "device"}:
            _fields(value, path, {_TAG, "value"})
            if type(value["value"]) is not str or not value["value"] or "\x00" in value["value"]:
                raise _error(path, f"Invalid {tag} string.")
        elif tag == "dtype":
            _fields(value, path, {_TAG, "name"})
            if type(value["name"]) is not str or value["name"] not in _DTYPES:
                raise _error(path, "Unsupported torch dtype name.")
        else:
            raise _error(path, f"Unknown recipe tag {tag!r}; arbitrary object deserialization is unsupported.")


def validate_run_spec(spec: Any) -> None:
    """Check the wire format without resolving references or constructing objects.

    Coordinators may call this. Application-specific signature and pipeline
    checks occur when the execution subprocess reconstructs the common Run.
    """
    _fields(spec, "run", {"format", "version", "id", "algorithm_input", "data", "config"})
    if spec["format"] != RUN_FORMAT or type(spec["version"]) is not int or spec["version"] != RUN_VERSION:
        raise _error("run", "Unsupported run transport format or version.")
    if type(spec["id"]) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", spec["id"]):
        raise _error("run.id", "Run IDs must contain 1-128 portable letters, digits, dots, underscores, or hyphens.")
    algorithm = spec["algorithm_input"]
    if (
        type(algorithm) is not dict or type(algorithm.get(_TAG)) is not str
        or algorithm[_TAG] not in {"lazy", "callable", "factory_ref"}
    ):
        raise _error("run.algorithm_input", _GUIDANCE)
    config = _mapping_fields(spec["config"], "run.config")
    if set(config) - _CONFIG_KEYS:
        raise _error("run.config", f"Unknown training options: {sorted(set(config) - _CONFIG_KEYS)!r}.")
    validator = _Validator()
    for key in ("algorithm_input", "data", "config"):
        validator.validate_value(spec[key], "run." + key)
    try:
        size = len(json.dumps(spec, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("ascii"))
    except (ValueError, TypeError, RecursionError) as exc:
        raise _error("run", "Recipe must be bounded, JSON-safe data.") from exc
    if size > _MAX_JSON_BYTES:
        raise _error("run", f"Encoded recipe exceeds {_MAX_JSON_BYTES} bytes; submit data as declared files.")


def serialize_run(run: Run, *, data_paths: Mapping | None = None) -> dict[str, Any]:
    """Encode a fresh Run without calling model, data, or training factories.

    ``data_paths`` is ``ProjectArtifact.data_mapping``. When supplied, every
    explicit local data descriptor must be covered by that mapping. Paths in
    application factory arguments are not guessed or rewritten.
    """
    if not isinstance(run, Run):
        raise _error("run", "Expected a training Run.")
    if run._engine is not None:
        raise _error("run._engine", f"A prepared run cannot be transported. {_GUIDANCE}")
    algorithm = run.algorithm_input
    if not isinstance(algorithm, (Lazy, FactoryRef)):
        if isinstance(algorithm, (nn.Module, Training)) or not callable(algorithm):
            raise _error("run.algorithm_input", f"Eager models and custom Training instances cannot be transported. {_GUIDANCE}")
        try:
            inspect.signature(algorithm).bind()
        except (TypeError, ValueError) as exc:
            raise _error("run.algorithm_input", "Training factories must accept zero arguments; use an importable zero-argument factory.") from exc
    mapping = _data_mapping(data_paths)
    encoder = _Encoder(mapping)
    data = run.data
    if isinstance(data, (str, PurePath)) and not str(data).startswith("s3://") and mapping is not None:
        remapped = _remap(data, mapping, "run.data")
        data = Path(remapped) if isinstance(data, PurePath) else remapped
    spec = {
        "format": RUN_FORMAT,
        "version": RUN_VERSION,
        "id": run.id,
        "algorithm_input": encoder.encode_value(algorithm, "run.algorithm_input"),
        "data": encoder.encode_value(data, "run.data"),
        "config": encoder.encode_value(run.config, "run.config"),
    }
    validate_run_spec(spec)
    return spec


class _Decoder:
    def __init__(self) -> None:
        self.lazy: dict[str, Lazy] = {}

    def decode(self, value: Any, path: str) -> Any:
        if value is None or type(value) in {bool, int, float, str}:
            return value
        if type(value) is list:
            return [self.decode(item, f"{path}[{index}]") for index, item in enumerate(value)]
        tag = value[_TAG]
        try:
            if tag == "mapping":
                return {key: self.decode(item, f"{path}[{key!r}]") for key, item in value["items"]}
            if tag == "tuple":
                return tuple(self.decode(item, f"{path}[{index}]") for index, item in enumerate(value["items"]))
            if tag in {"factory_ref", "callable"}:
                reference = FactoryRef(value["module"], value["qualname"])
                return reference if tag == "factory_ref" else reference.resolve()
            if tag == "lazy_ref":
                return self.lazy[value["id"]]
            if tag == "lazy":
                factory = self.decode(value["factory"], path + ".factory")
                args = self.decode(value["args"], path + ".args")
                kwargs = self.decode(value["kwargs"], path + ".kwargs")
                result = Lazy(factory, *args, gpu=value["gpu"], **kwargs)
                result.train(value["training"])
                self.lazy[value["id"]] = result
                return result
            if tag == "optimizer":
                return OptimizerSpec(**{key: self.decode(value[key], path + "." + key) for key in ("name", "params", "kwargs")})
            if tag in {"data_pipeline", "schedule", "checkpoint"}:
                spec = self.decode(value["spec"], path + ".spec")
                if tag == "data_pipeline":
                    pipeline = DataPipeline.from_spec(spec, lambda item: item)
                    return pipeline.rewrite_local_paths(lambda local: local)
                if tag == "schedule":
                    return ScheduleSpec.from_dict(spec)
                return CheckpointPolicy.from_dict(spec)
            if tag == "path":
                return Path(value["value"])
            if tag == "device":
                return torch.device(value["value"])
            if tag == "dtype":
                return _DTYPES[value["name"]]
        except (ConfigurationError, TypeError, ValueError, RuntimeError) as exc:
            raise _error(path, str(exc)) from exc
        raise _error(path, "Unsupported recipe tag.")


def deserialize_run(spec: Mapping[str, Any]) -> Run:
    """Reconstruct the common Run in a worker, without preparing or starting it.

    Import resolution executes ordinary Python module imports. Use only with
    trusted submitted code, after entering the extracted project directory.
    """
    validate_run_spec(spec)
    decoder = _Decoder()
    algorithm = decoder.decode(spec["algorithm_input"], "run.algorithm_input")
    if type(algorithm) is FactoryRef:
        algorithm = algorithm.resolve()
    data = decoder.decode(spec["data"], "run.data")
    if isinstance(data, (str, PurePath)) and not str(data).startswith("s3://"):
        resolved = _absolute(data)
        data = resolved if isinstance(data, PurePath) else str(resolved)
    config = decoder.decode(spec["config"], "run.config")
    try:
        result = train(algorithm, data, **config)
    except (ConfigurationError, TypeError, ValueError) as exc:
        raise _error("run", f"Cannot reconstruct training configuration: {exc}") from exc
    result.id = spec["id"]
    return result
