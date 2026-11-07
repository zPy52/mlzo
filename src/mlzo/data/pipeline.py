"""Immutable pipeline configuration and ordered, lazy iteration."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from numbers import Real
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader, Dataset

from mlzo.errors import ConfigurationError, UnsupportedFeatureError

from .loading import (
    FilteredIterable,
    IterableView,
    OrderedDataset,
    ShuffledIterable,
    collate_default,
    collate_normalized,
    finite,
    length as _length,
    map_style,
    mapped,
)
from .sources import (
    DatasetIterable,
    Source,
    StreamDataset,
    describe_source,
    indexed_source,
    is_s3,
    materialize_source,
    paired_source,
    path_string,
    stream_source,
)


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"{name} must be an integer >= {minimum}.")
    return value


def _identity(value: Any) -> Any:
    return value


@dataclass(frozen=True)
class _Operation:
    name: str
    options: tuple[tuple[str, Any], ...]


def _statistics(value: Any, name: str) -> tuple[float, ...]:
    if isinstance(value, Real) and not isinstance(value, bool):
        result = (float(value),)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if any(not isinstance(item, Real) or isinstance(item, bool) for item in value):
            raise ConfigurationError(f"Normalization {name} must contain real numbers.")
        result = tuple(float(item) for item in value)
    else:
        raise ConfigurationError(f"Normalization {name} must be a number or a sequence of numbers.")
    if not result or any(not math.isfinite(item) for item in result):
        raise ConfigurationError(f"Normalization {name} must contain finite numbers.")
    return result


def _field(sample: Any, key: str, function: Callable) -> dict:
    if not isinstance(sample, Mapping) or key not in sample:
        raise ConfigurationError(f"The data operation requires a record with field {key!r}.")
    result = dict(sample)
    result[key] = function(sample[key])
    return result


@dataclass(frozen=True)
class _Normalize:
    mean: tuple[float, ...]
    std: tuple[float, ...]
    dim: int
    key: str | None

    def _tensor(self, value: Any) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise ConfigurationError("Normalization requires a floating point tensor.")
        if value.ndim == 0 or not -value.ndim <= self.dim < value.ndim:
            raise ConfigurationError(
                f"Normalization dim={self.dim} is invalid for a tensor of rank {value.ndim}."
            )
        dim = self.dim % value.ndim
        if len(self.mean) not in {1, value.shape[dim]}:
            raise ConfigurationError("Normalization statistics must match the selected dimension.")
        shape = [1] * value.ndim
        shape[dim] = len(self.mean)
        mean = value.new_tensor(self.mean).reshape(shape)
        std = value.new_tensor(self.std).reshape(shape)
        return (value - mean) / std

    def __call__(self, sample: Any) -> Any:
        if self.key is not None:
            return _field(sample, self.key, self._tensor)
        if isinstance(sample, torch.Tensor):
            return self._tensor(sample)
        if isinstance(sample, (tuple, list)) and len(sample) == 2 and isinstance(sample[0], torch.Tensor):
            values = [self._tensor(sample[0]), sample[1]]
            return tuple(values) if isinstance(sample, tuple) else values
        raise ConfigurationError(
            "Normalize a tensor or (tensor, target) pair, or specify key= for record data."
        )


@dataclass(frozen=True)
class _Tokenize:
    tokenizer: Callable
    key: str | None

    def __call__(self, sample: Any) -> Any:
        if self.key is not None:
            return _field(sample, self.key, self.tokenizer)
        return self.tokenizer(sample)


@dataclass(frozen=True)
class DataPipeline:
    """A source description plus immutable operations, applied in call order.

    Source objects remain caller-owned. Immutability concerns configuration, not
    the contents of a supplied list, tensor, dataset, or arbitrary iterator.
    Factories and file reads run on build, never on serialization. JSONL and
    iterable sources stream. Full indexable shuffles only permute indices;
    full iterable shuffles buffer finite input records during iteration.
    """

    source: Any
    _operations: tuple[_Operation, ...] = field(default=(), repr=False)
    _num_workers: int | None = field(default=None, repr=False)
    _prefetch_factor: int | None = field(default=None, repr=False)
    _loader_configured: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", describe_source(self.source))

    def _append(self, name: str, **options: Any) -> DataPipeline:
        operation = _Operation(name, tuple(options.items()))
        return replace(self, _operations=(*self._operations, operation))

    def map(self, function: Callable[[Any], Any]) -> DataPipeline:
        """Apply a local callable to each current item, including batches."""
        if not callable(function):
            raise ConfigurationError("map() requires a callable.")
        return self._append("map", function=function)

    def _samples_only(self, name: str) -> None:
        if any(op.name == "batch" for op in self._operations) or (
            self.source.kind == "native" and isinstance(self.source.value, DataLoader)
        ):
            raise ConfigurationError(f"{name}() acts on samples; use map() for existing batches.")

    def transform(self, function: Callable[[Any], Any]) -> DataPipeline:
        """Transform individual samples. After batch(), use map() instead."""
        self._samples_only("transform")
        if not callable(function):
            raise ConfigurationError("transform() requires a callable.")
        return self._append("transform", function=function)

    def filter(self, function: Callable[[Any], bool]) -> DataPipeline:
        """Lazily retain samples with a scalar boolean predicate; length is unknown."""
        self._samples_only("filter")
        if not callable(function):
            raise ConfigurationError("filter() requires a callable.")
        return self._append("filter", function=function)

    def shuffle(
        self, seed: int = 42, *, buffer: int | None = None, buffer_size: int | None = None
    ) -> DataPipeline:
        """Shuffle with seed + epoch. Streams need a bounded buffer.

        Full indexable shuffles permute indices without reading samples. Finite
        iterable shuffles buffer samples during iteration. Buffered shuffle is
        bounded and is not a uniform full permutation. buffer_size is legacy.
        """
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ConfigurationError("Shuffle seed must be an integer.")
        if buffer is not None and buffer_size is not None:
            raise ConfigurationError("Use only one of buffer and buffer_size.")
        size = buffer if buffer is not None else buffer_size
        if size is not None:
            _integer(size, "shuffle buffer", minimum=1)
        return self._append("shuffle", seed=seed, buffer_size=size)

    def batch(
        self, size: int, drop_last: bool = False, collate_fn: Callable | None = None,
        *, shuffle: bool = False, workers: int | None = None,
        prefetch: int | None = None, combine: Callable | None = None,
        sampler: Any = None, pin_memory: bool = False, normalize: bool = True,
    ) -> DataPipeline:
        """Build batches with native DataLoader workers and collation.

        Default collation unwraps MLZO tensor leaves at their existing shape,
        then shapes scalar/1D batch leaves to at least two axes after stacking.
        normalize=False preserves native collated shapes. Custom combine or
        collate_fn receives original samples and preserves its exact output.
        Worker prefetch counts batches per worker. Construct a native DataLoader
        for advanced options such as batch_sampler, then pass it to data() unchanged.
        """
        _integer(size, "batch size", minimum=1)
        for name, value in (("drop_last", drop_last), ("shuffle", shuffle),
                            ("pin_memory", pin_memory), ("normalize", normalize)):
            if not isinstance(value, bool):
                raise ConfigurationError(f"{name} must be a boolean.")
        if workers is not None:
            _integer(workers, "worker count")
        if prefetch is not None:
            _integer(prefetch, "prefetch factor", minimum=1)
        if combine is not None and collate_fn is not None:
            raise ConfigurationError("Use only one of combine and collate_fn.")
        collate = combine if combine is not None else collate_fn
        if collate is not None and not callable(collate):
            raise ConfigurationError("combine/collate_fn must be callable or None.")
        if sampler is not None and (shuffle or any(op.name == "shuffle" for op in self._operations)):
            raise ConfigurationError("sampler and shuffle cannot be combined.")
        if sampler is not None and not hasattr(sampler, "__iter__"):
            raise ConfigurationError("sampler must be an iterable of indices.")
        if any(op.name == "batch" for op in self._operations):
            raise UnsupportedFeatureError("This pipeline is already batched. Use a native DataLoader for re-batching.")
        return self._append("batch", size=size, drop_last=drop_last, collate_fn=collate,
                            shuffle=shuffle, workers=workers, prefetch=prefetch,
                            sampler=sampler, pin_memory=pin_memory, normalize=normalize)

    def workers(self, count: int, *, prefetch_factor: int | None = None) -> DataPipeline:
        """Set native loading workers. Operations before batching run there."""
        _integer(count, "worker count")
        if self._num_workers is not None and count != self._num_workers:
            raise ConfigurationError("Conflicting worker counts.")
        result = replace(self, _num_workers=count, _loader_configured=True)
        return result.prefetch(prefetch_factor) if prefetch_factor is not None else result

    def prefetch(self, factor: int) -> DataPipeline:
        """Set native prefetch_factor, counting batches per worker."""
        _integer(factor, "prefetch factor", minimum=1)
        if self._prefetch_factor is not None and factor != self._prefetch_factor:
            raise ConfigurationError("Conflicting prefetch factors.")
        return replace(self, _prefetch_factor=factor, _loader_configured=True)

    def _loading_options(self) -> tuple[int, int | None]:
        batch = next((dict(op.options) for op in self._operations if op.name == "batch"), {})
        workers, prefetch = batch.get("workers"), batch.get("prefetch")
        if workers is not None and self._num_workers is not None and workers != self._num_workers:
            raise ConfigurationError("Conflicting batch workers and workers() counts.")
        if prefetch is not None and self._prefetch_factor is not None and prefetch != self._prefetch_factor:
            raise ConfigurationError("Conflicting batch prefetch and prefetch() factors.")
        workers = self._num_workers if workers is None else workers
        prefetch = self._prefetch_factor if prefetch is None else prefetch
        return workers or 0, prefetch

    def images(
        self,
        size: Sequence[int] | None = None,
        *,
        labels: str,
        mode: str = "RGB",
    ) -> DataPipeline:
        """Decode a local directory to float32 CHW tensors in [0, 1].

        labels must be 'unlabeled' or 'class_subdirectories'. The latter returns
        (image, integer_class) using sorted immediate directory names. size is
        (height, width); mode is RGB or L. Pillow is imported only when decoding.
        """
        if self.source.kind != "path" or self._operations:
            raise UnsupportedFeatureError("images() must configure a directory before pipeline operations.")
        if is_s3(self.source.value):
            raise UnsupportedFeatureError("Image directories must be local paths in this alpha.")
        if not isinstance(labels, str) or labels not in {"unlabeled", "class_subdirectories"}:
            raise ConfigurationError("Image labels must be 'unlabeled' or 'class_subdirectories'.")
        if not isinstance(mode, str) or mode not in {"RGB", "L"}:
            raise UnsupportedFeatureError("Image decoding supports only RGB and L modes.")
        if size is not None:
            if not isinstance(size, Sequence) or isinstance(size, (str, bytes)) or len(size) != 2:
                raise ConfigurationError("Image size must be (height, width).")
            size = tuple(_integer(item, "image dimension", minimum=1) for item in size)
        options = (("labels", labels), ("size", size), ("mode", mode))
        return replace(self, source=Source("images", self.source.value, options))

    def normalize(
        self,
        mean: Real | Sequence[Real] | str,
        std: Real | Sequence[Real] | None = None,
        *,
        dim: int = -3,
        key: str | None = None,
    ) -> DataPipeline:
        """Normalize a floating tensor, a pair's input, or an explicit record field.

        dim defaults to the image channel dimension. Use dim=-1 for features.
        normalize('imagenet') selects RGB ImageNet mean and standard deviation.
        """
        if isinstance(mean, str):
            if mean != "imagenet" or std is not None:
                raise ConfigurationError("The only named normalization is normalize('imagenet').")
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        means = _statistics(mean, "mean")
        deviations = _statistics(std, "std")
        if len(means) != len(deviations) or any(value <= 0 for value in deviations):
            raise ConfigurationError("Normalization needs matching mean/std lengths and strictly positive std.")
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise ConfigurationError("Normalization dim must be an integer.")
        if key is not None and not isinstance(key, str):
            raise ConfigurationError("Normalization key must be a string or None.")
        return self._append("normalize", mean=means, std=deviations, dim=dim, key=key)

    def tokenize(self, tokenizer: Callable, *, key: str | None = None) -> DataPipeline:
        """Apply the caller's tokenizer to each item or one explicit record field.

        No vocabulary, padding, truncation, or attention mask is inferred. Use a
        custom batch collate_fn for padding.
        """
        if not callable(tokenizer) or (key is not None and not isinstance(key, str)):
            raise ConfigurationError("tokenize() requires a callable and an optional string key.")
        return self._append("tokenize", tokenizer=tokenizer, key=key)

    def build(self, epoch: int = 0) -> Iterable:
        """Construct native loading lazily; shuffle uses seed + epoch.

        Iterable workers batch their own shards, following native PyTorch
        semantics, including drop_last per worker. Partial epochs have no exact
        cursor replay guarantee. Generic native IterableDatasets own sharding.
        """
        _integer(epoch, "epoch")
        workers, prefetch = self._loading_options()
        values = materialize_source(self.source)
        native_loader = isinstance(values, DataLoader)
        if native_loader and (self._loader_configured or any(op.name == "batch" for op in self._operations)):
            raise UnsupportedFeatureError(
                "A native DataLoader owns batching, workers and prefetch. Configure it directly or provide its Dataset."
            )
        if prefetch is not None and not workers:
            raise ConfigurationError("prefetch requires at least one CPU worker.")
        sampler = next(
            (dict(op.options)["sampler"] for op in self._operations if op.name == "batch"), None
        )
        if map_style(values) and _length(values) is None and sampler is None:
            raise UnsupportedFeatureError(
                "A map-style Dataset must provide a finite length (__len__) or an explicit sampler."
            )
        if workers and not isinstance(values, Dataset):
            raise UnsupportedFeatureError("Workers require a Dataset; arbitrary iterables cannot be sharded safely.")
        if workers > 1 and isinstance(values, StreamDataset) and getattr(values.factory, "mlzo_replayable", False) is not True:
            raise UnsupportedFeatureError("Multiple stream workers require factory.mlzo_replayable = True.")
        batched = native_loader
        for operation in self._operations:
            options = dict(operation.options)
            if operation.name in {"map", "transform"}:
                if operation.name == "transform" and batched:
                    raise ConfigurationError("transform() acts on samples; use map() for batches.")
                values = mapped(values, options["function"])
            elif operation.name == "filter":
                if batched:
                    raise ConfigurationError("filter() acts on samples, not existing batches.")
                values = FilteredIterable(values, options["function"])
            elif operation.name == "batch":
                sampler = options["sampler"]
                if sampler is not None and not map_style(values):
                    raise UnsupportedFeatureError("A sampler requires an indexable Dataset, not an iterable or filtered source.")
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)
                if options["shuffle"]:
                    if not map_style(values):
                        raise UnsupportedFeatureError("Batch shuffle requires an indexable finite source; use shuffle(buffer=...) for streams.")
                    values = OrderedDataset(values, 42 + epoch)
                collate = options["collate_fn"]
                if collate is None:
                    collate = collate_normalized if options["normalize"] else collate_default
                values = DataLoader(
                    values if isinstance(values, Dataset) else IterableView(values),
                    batch_size=options["size"], drop_last=options["drop_last"],
                    sampler=sampler, collate_fn=collate, num_workers=workers,
                    pin_memory=options["pin_memory"],
                    generator=torch.Generator().manual_seed(epoch % (2**64)),
                    **({"prefetch_factor": prefetch} if prefetch is not None else {}),
                )
                batched = True
            elif operation.name == "shuffle":
                if options["buffer_size"] is None and not finite(values):
                    raise UnsupportedFeatureError("Full shuffling requires finite input. Set buffer for streams.")
                if map_style(values) and options["buffer_size"] is None:
                    values = OrderedDataset(values, options["seed"] + epoch)
                else:
                    values = ShuffledIterable(values, options["seed"] + epoch, options["buffer_size"])
            elif operation.name == "normalize":
                values = mapped(values, _Normalize(**options))
            elif operation.name == "tokenize":
                values = mapped(values, _Tokenize(**options))
            else:
                raise UnsupportedFeatureError(f"Data operation {operation.name!r} is not supported.")
        if not batched and workers:
            return DataLoader(
                values, batch_size=None, collate_fn=_identity, num_workers=workers,
                generator=torch.Generator().manual_seed(epoch % (2**64)),
                **({"prefetch_factor": prefetch} if prefetch is not None else {}),
            )
        return DatasetIterable(values) if map_style(values) else values

    def __iter__(self) -> Iterator:
        return iter(self.build())

    def __len__(self) -> int:
        if self.source.kind not in {"native", "memory", "paired", "indexed"}:
            raise TypeError("This pipeline's length is available only after building its source.")
        if any(operation.name == "filter" for operation in self._operations):
            raise TypeError("A filtered pipeline has unknown length until consumed.")
        count = _length(materialize_source(self.source))
        for operation in self._operations:
            if operation.name == "batch":
                options = dict(operation.options)
                if options["sampler"] is not None:
                    count = _length(options["sampler"])
                    if count is None:
                        raise TypeError("This sampler does not provide a length.")
                if count is None:
                    raise TypeError("This data source does not provide a length.")
                size = options["size"]
                count = count // size if options["drop_last"] else (count + size - 1) // size
        if count is None:
            raise TypeError("This data source does not provide a length.")
        return count

    def to_spec(self, encode: Callable[[Any], Any]) -> dict[str, Any]:
        """Encode version 2 configuration. The one-argument encoder handles values.

        A remote encoder should reject nonimportable callables and unsupported
        native objects. Neither factories nor map functions are invoked here.
        """
        return {
            "version": 2,
            "source": {
                "kind": self.source.kind,
                "value": encode(self.source.value),
                "options": {name: encode(value) for name, value in self.source.options},
            },
            "operations": [
                {
                    "name": operation.name,
                    "options": {name: encode(value) for name, value in operation.options},
                }
                for operation in self._operations
            ],
            "loader": {
                "workers": self._num_workers,
                "prefetch_factor": self._prefetch_factor,
                "configured": self._loader_configured,
            },
        }

    @classmethod
    def from_spec(cls, spec: Mapping, decode: Callable[[Any], Any]) -> DataPipeline:
        """Decode without building. Version 1 keeps legacy batch shapes and shuffle order."""
        if (
            not isinstance(spec, Mapping)
            or set(spec) != {"version", "source", "operations", "loader"}
        ):
            raise ConfigurationError("A data pipeline specification needs version, source, operations, and loader.")
        if type(spec["version"]) is not int or spec["version"] not in {1, 2}:
            raise UnsupportedFeatureError("Unsupported data pipeline specification version.")
        source = spec["source"]
        if (
            not isinstance(source, Mapping)
            or set(source) != {"kind", "value", "options"}
        ):
            raise ConfigurationError("Invalid data source specification.")
        if not isinstance(source["options"], Mapping):
            raise ConfigurationError("Data source options must be a mapping.")
        if not isinstance(source["kind"], str):
            raise ConfigurationError("Data source kind must be a string.")
        source_value = decode(source["value"])
        source_options = {name: decode(value) for name, value in source["options"].items()}
        new_sources = {"paired": {"targets"}, "indexed": {"length"}, "stream": set()}
        if source["kind"] in new_sources:
            if set(source_options) != new_sources[source["kind"]]:
                raise ConfigurationError(f"Invalid options for data source {source['kind']!r}.")
            if source["kind"] == "paired":
                result = data(source_value, **source_options)
            elif source["kind"] == "indexed":
                result = data(get=source_value, **source_options)
            else:
                result = data(stream=source_value)
        elif source["kind"] == "images":
            if set(source_options) != {"labels", "size", "mode"}:
                raise ConfigurationError("Image specifications need labels, size, and mode.")
            result = cls(source_value).images(**source_options)
        else:
            result = cls(source_value)
            if source["kind"] != result.source.kind or source_options:
                raise ConfigurationError("Data source kind or options do not match the decoded value.")
        operations = spec["operations"]
        if not isinstance(operations, list):
            raise ConfigurationError("Data operations must be an ordered list.")
        required = {
            "map": {"function"},
            "transform": {"function"},
            "filter": {"function"},
            "shuffle": {"seed", "buffer_size"},
            "batch": {"size", "drop_last", "collate_fn"} if spec["version"] == 1 else {
                "size", "drop_last", "collate_fn", "shuffle", "workers", "prefetch", "sampler", "pin_memory", "normalize"
            },
            "normalize": {"mean", "std", "dim", "key"},
            "tokenize": {"tokenizer", "key"},
        }
        for operation in operations:
            if (
                not isinstance(operation, Mapping)
                or set(operation) != {"name", "options"}
            ):
                raise ConfigurationError("Invalid data operation specification.")
            name = operation["name"]
            if not isinstance(name, str) or name not in required:
                raise UnsupportedFeatureError(f"Data operation {name!r} is not supported.")
            options = operation["options"]
            if not isinstance(options, Mapping) or set(options) != required[name]:
                raise ConfigurationError(f"Invalid options for data operation {name!r}.")
            decoded = {key: decode(value) for key, value in options.items()}
            if spec["version"] == 1 and name == "batch":
                decoded["normalize"] = False
            result = getattr(result, name)(**decoded)
        loader = spec["loader"]
        if (
            not isinstance(loader, Mapping)
            or set(loader) != {"workers", "prefetch_factor", "configured"}
        ):
            raise ConfigurationError("Invalid loader specification.")
        if not isinstance(loader["configured"], bool):
            raise ConfigurationError("Loader configured flag must be a boolean.")
        if loader["workers"] is not None:
            _integer(loader["workers"], "worker count")
        elif spec["version"] == 1:
            raise ConfigurationError("Version 1 worker count must be an integer.")
        if loader["prefetch_factor"] is not None:
            _integer(loader["prefetch_factor"], "prefetch_factor", minimum=1)
        if loader["configured"]:
            if loader["workers"] is not None:
                result = result.workers(loader["workers"])
            if loader["prefetch_factor"] is not None:
                result = result.prefetch(loader["prefetch_factor"])
        elif loader["workers"] not in {None, 0} or loader["prefetch_factor"] is not None:
            raise ConfigurationError("Unconfigured loader specifications cannot include worker options.")
        return result

    def local_paths(self) -> tuple[Path, ...]:
        """List explicit source paths as absolute Paths, without reading them.

        Paths captured inside factories or callables cannot be discovered.
        """
        if self.source.kind in {"path", "images"} and not is_s3(self.source.value):
            return (Path(self.source.value).expanduser().absolute(),)
        return ()

    def rewrite_local_paths(self, rewrite: Callable[[Path], str | Path]) -> DataPipeline:
        """Return a pipeline with explicit paths replaced by rewrite(absolute_path)."""
        paths = self.local_paths()
        if not paths:
            return self
        path = path_string(rewrite(paths[0]))
        if self.source.kind == "images" and is_s3(path):
            raise UnsupportedFeatureError("Image directories must remain local worker paths.")
        result = describe_source(path)
        return replace(self, source=Source(self.source.kind, result.value, self.source.options))


_MISSING = object()


def data(
    source: Any = None,
    targets: Any = _MISSING,
    *,
    get: Callable[[int], Any] | None = None,
    length: int | None = None,
    stream: Callable[[], Iterable] | None = None,
) -> DataPipeline:
    """Describe one source, input/target columns, an index function, or a stream.

    get and length must be supplied together. stream creates a fresh iterator
    for each pass. These modes cannot be combined. A single sequence of pairs
    remains a sequence of samples; use two arguments to describe columns.
    Existing pipelines are returned unchanged and native loaders retain their
    identity on build. No sample function or stream factory is invoked here.
    """
    indexed = get is not None or length is not None
    if indexed or stream is not None:
        if source is not None or targets is not _MISSING or (indexed and stream is not None):
            raise ConfigurationError("Choose only one data source mode: source/targets, get/length, or stream.")
        if indexed:
            if get is None or length is None:
                raise ConfigurationError("An indexed data source requires both get and length.")
            return DataPipeline(indexed_source(get, length))
        return DataPipeline(stream_source(stream))
    if source is None:
        raise ConfigurationError("data() requires a source, get/length, or stream.")
    if targets is not _MISSING:
        return DataPipeline(paired_source(source, targets))
    return source if isinstance(source, DataPipeline) else DataPipeline(source)


def images(
    path: str | Path,
    size: Sequence[int] | None = None,
    *,
    labels: str,
    mode: str = "RGB",
) -> DataPipeline:
    """Describe a local image directory with an explicit label convention."""
    return data(path).images(size=size, labels=labels, mode=mode)
