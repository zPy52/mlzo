"""Native dataset adapters used by the ordered pipeline builder."""

from __future__ import annotations

import random
from collections.abc import Mapping
from itertools import islice
from typing import Any

import torch
from torch.utils.data import Dataset, IterableDataset, default_collate, get_worker_info

from mlzo.errors import ConfigurationError


def length(value):
    try:
        return len(value)
    except (TypeError, NotImplementedError):
        return None


def map_style(value):
    return isinstance(value, Dataset) and not isinstance(value, IterableDataset)


def finite(value):
    return getattr(value, "finite", False) or length(value) is not None


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

    def __iter__(self):
        for sample in super().__iter__():
            result = self.predicate(sample)
            if isinstance(result, torch.Tensor):
                if result.ndim != 0 or result.dtype != torch.bool:
                    raise ConfigurationError("filter() requires a scalar boolean predicate result.")
                result = result.item()
            # NumPy boolean scalars implement __bool__, but integers and arrays
            # must not pass as predicates merely because bool(value) works.
            if type(result).__module__ == "numpy" and type(result).__name__ == "bool_":
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


def normalize_batch(value: Any) -> Any:
    """Shape native collated leaves, never unbatched samples or custom output."""
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.reshape(1, 1)
        if value.ndim == 1:
            return value.unsqueeze(-1)
        return value
    if isinstance(value, Mapping):
        return {key: normalize_batch(item) for key, item in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(normalize_batch(item) for item in value))
    if isinstance(value, tuple):
        return tuple(normalize_batch(item) for item in value)
    if isinstance(value, list):
        return [normalize_batch(item) for item in value]
    return value


def collate_normalized(samples):
    return normalize_batch(default_collate(samples))
