"""Native dataset adapters used by the ordered pipeline builder."""

from __future__ import annotations

import copy
import random
import sys
from collections.abc import Mapping, MutableMapping
from itertools import islice
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, default_collate, get_worker_info

from mlzo.errors import ConfigurationError
from mlzo.transforms.tensor import Tensor


def length(value):
    try:
        return len(value)
    except (TypeError, NotImplementedError):
        return None


def map_style(value):
    return isinstance(value, Dataset) and not isinstance(value, IterableDataset)


def finite(value):
    return getattr(value, "finite", False) or length(value) is not None


def batch_length(source, size, drop_last, workers):
    """Count native batches without reading samples or guessing worker shards."""
    if workers > 1 and not map_style(source):
        if not isinstance(source, IterableView):
            raise TypeError("Native iterable worker shard lengths are unknown.")
        counts = source.shard_lengths(workers)
    else:
        counts = [len(source)]
    if size is None:
        return sum(counts)
    return sum(count // size if drop_last else (count + size - 1) // size for count in counts)


class PipelineLoader(DataLoader):
    """Native loading with exact lengths for MLZO's known iterable shards."""

    def __len__(self):
        if isinstance(self.dataset, IterableDataset):
            return batch_length(self.dataset, self.batch_size, self.drop_last, self.num_workers)
        return super().__len__()


class MappedDataset(Dataset):
    def __init__(self, source, function):
        self.source, self.function = source, function

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        return self.function(self.source[index])


class OrderedDataset(Dataset):
    """Permute only indices, leaving expensive sample reads to the loader."""

    def __init__(self, source, seed):
        self.source = source
        # Keep the persisted v1 shuffle order while permuting indices rather
        # than materializing samples. Native DataLoader still owns fetching.
        self.indices = list(range(len(source)))
        random.Random(seed).shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.source[self.indices[index]]


class IterableView(IterableDataset):
    def __init__(self, source):
        self.source = source
        self.finite = finite(source)

    def __iter__(self):
        if map_style(self.source):
            worker = get_worker_info()
            start, step = (worker.id, worker.num_workers) if worker else (0, 1)
            return (self.source[index] for index in range(start, len(self.source), step))
        # Native IterableDatasets own their sharding. StreamDataset does its own
        # striding. Never stride either a second time here.
        return iter(self.source)

    def __len__(self):
        return len(self.source)

    def shard_lengths(self, workers):
        if map_style(self.source):
            count, extra = divmod(len(self.source), workers)
            return [count + (worker < extra) for worker in range(workers)]
        if isinstance(self.source, IterableView):
            return self.source.shard_lengths(workers)
        raise TypeError("Native iterable worker shard lengths are unknown.")


class MappedIterable(IterableView):
    def __init__(self, source, function):
        super().__init__(source)
        self.function = function

    def __iter__(self):
        return map(self.function, super().__iter__())


class FilteredIterable(IterableView):
    def __init__(self, source, predicate):
        super().__init__(source)
        self.predicate = predicate

    def __len__(self):
        raise TypeError("A filtered source has unknown length until consumed.")

    def shard_lengths(self, workers):
        raise TypeError("A filtered source has unknown worker shard lengths.")

    def __iter__(self):
        for sample in super().__iter__():
            result = self.predicate(sample)
            if isinstance(result, torch.Tensor):
                if result.ndim != 0 or result.dtype != torch.bool:
                    raise ConfigurationError("filter() requires a scalar boolean predicate result.")
                result = result.item()
            # NumPy boolean scalars implement __bool__, but integers and arrays
            # must not pass as predicates merely because bool(value) works.
            numpy = sys.modules.get("numpy")
            if numpy is not None and isinstance(result, numpy.bool_):
                result = bool(result)
            if not isinstance(result, bool):
                raise ConfigurationError("filter() requires a scalar boolean predicate result.")
            if result:
                yield sample


class ShuffledIterable(IterableView):
    def __init__(self, source, seed, buffer_size):
        super().__init__(source)
        self.seed, self.buffer_size = seed, buffer_size

    def __iter__(self):
        generator = random.Random(self.seed)
        iterator = super().__iter__()
        if self.buffer_size is None:
            values = list(iterator)
            generator.shuffle(values)
            yield from values
            return
        buffer = list(islice(iterator, self.buffer_size))
        for item in iterator:
            index = generator.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = item
        generator.shuffle(buffer)
        yield from buffer


def mapped(source, function):
    return MappedDataset(source, function) if map_style(source) else MappedIterable(source, function)


def _map_leaves(value: Any, function) -> Any:
    if isinstance(value, Mapping):
        items = {key: _map_leaves(item, function) for key, item in value.items()}
        try:
            if isinstance(value, MutableMapping):
                result = copy.copy(value)
                result.update(items)
                return result
            return type(value)(items)
        except TypeError:
            return items
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(_map_leaves(item, function) for item in value))
    if isinstance(value, tuple):
        return tuple(_map_leaves(item, function) for item in value)
    if isinstance(value, list):
        return [_map_leaves(item, function) for item in value]
    return function(value)


def _normalize_leaf(value):
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.reshape(1, 1)
        if value.ndim == 1:
            return value.unsqueeze(-1)
    return value


def collate_default(samples):
    """Unwrap sample leaves without copying, detaching, or changing their rank."""
    native = _map_leaves(samples, lambda value: value.raw if isinstance(value, Tensor) else value)
    return default_collate(native)


def collate_normalized(samples):
    return _map_leaves(collate_default(samples), _normalize_leaf)
