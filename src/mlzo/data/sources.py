"""Source adapters. Construction and serialization never read a source."""

from __future__ import annotations

import inspect
import io
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit

import torch
from torch.utils.data import DataLoader, Dataset

from mlzo.errors import ConfigurationError, UnsupportedFeatureError


FILE_SUFFIXES = {".json", ".jsonl", ".pt", ".pth"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Source:
    kind: str
    value: Any
    options: tuple[tuple[str, Any], ...] = ()


def path_string(value: Any) -> str:
    path = os.fspath(value)
    if not isinstance(path, str) or not path:
        raise ConfigurationError("A data path must be a nonempty string or Path.")
    scheme = urlsplit(path).scheme
    if scheme and scheme != "s3":
        raise UnsupportedFeatureError(f"Data URI scheme {scheme!r} is not supported.")
    return path


def is_s3(path: str) -> bool:
    return urlsplit(path).scheme == "s3"


def validate_s3_file(path: str) -> None:
    parts = urlsplit(path)
    if (
        not parts.netloc
        or not parts.path.lstrip("/")
        or parts.path.endswith("/")
        or Path(parts.path).suffix.lower() not in FILE_SUFFIXES
        or parts.query
        or parts.fragment
    ):
        raise UnsupportedFeatureError(
            "S3 data must name one .json, .jsonl, .pt, or .pth object. "
            "Arbitrary S3 prefixes and image directories are not supported."
        )


def describe_source(value: Any) -> Source:
    if isinstance(value, Source):
        return value
    if isinstance(value, (str, os.PathLike)):
        path = path_string(value)
        if is_s3(path):
            validate_s3_file(path)
        return Source("path", path)
    if isinstance(value, (DataLoader, Dataset)):
        return Source("native", value)
    if callable(value):
        try:
            inspect.signature(value).bind()
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                "A dataset factory must accept zero arguments. Use a named factory "
                "in an importable module for remote runs."
            ) from exc
        return Source("factory", value)
    if isinstance(value, (torch.Tensor, Mapping, Sequence, Iterable)):
        return Source("memory", value)
    raise UnsupportedFeatureError(
        "Data sources must be native datasets/loaders, iterable collections, "
        "zero-argument factories, or supported file paths."
    )


class SequenceDataset(Dataset):
    def __init__(self, values: Sequence | torch.Tensor):
        if isinstance(values, torch.Tensor) and values.ndim == 0:
            raise ConfigurationError("A tensor data source needs a leading sample dimension.")
        self.values = values

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> Any:
        return self.values[index]


class ColumnsDataset(Dataset):
    """A mapping of field names to equally sized tensors or sequences."""

    def __init__(self, columns: Mapping, *, tensors_only: bool = False):
        if not columns or any(not isinstance(key, str) for key in columns):
            raise ConfigurationError("Column data needs a nonempty mapping with string keys.")
        lengths = []
        for value in columns.values():
            if isinstance(value, torch.Tensor):
                if value.ndim == 0:
                    raise ConfigurationError("Each tensor column needs a sample dimension.")
            elif tensors_only or not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise ConfigurationError(
                    "Columns must be tensors or sequences; tensor files require tensor columns."
                )
            lengths.append(len(value))
        if len(set(lengths)) != 1:
            raise ConfigurationError("Every data column must have the same number of samples.")
        self.columns = columns
        self.length = lengths[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {key: value[index] for key, value in self.columns.items()}


@dataclass(frozen=True)
class DatasetIterable:
    dataset: Dataset

    def __iter__(self) -> Iterator:
        # Dataset.__getitem__ is not required to raise IndexError past its length.
        for index in range(len(self.dataset)):
            yield self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)


def adapt_values(value: Any) -> Any:
    if isinstance(value, (DataLoader, Dataset)):
        return value
    if isinstance(value, torch.Tensor):
        return SequenceDataset(value)
    if isinstance(value, Mapping):
        return ColumnsDataset(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return SequenceDataset(value)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        return value
    raise UnsupportedFeatureError("The dataset factory did not return a supported data source.")


@contextmanager
def open_binary(path: str) -> Iterator[Any]:
    if not is_s3(path):
        with Path(path).expanduser().open("rb") as handle:
            yield handle
        return
    validate_s3_file(path)
    try:
        import boto3
    except ImportError as exc:
        raise UnsupportedFeatureError(
            "Reading S3 data requires the optional boto3 dependency. Install boto3 on the worker."
        ) from exc
    parts = urlsplit(path)
    body = boto3.client("s3").get_object(Bucket=parts.netloc, Key=parts.path.lstrip("/"))["Body"]
    try:
        yield body
    finally:
        body.close()


@dataclass(frozen=True)
class JsonLines:
    path: str

    def __iter__(self) -> Iterator[dict[str, Any]]:
        with open_binary(self.path) as stream:
            # Both local files and botocore StreamingBody support line iteration.
            lines = stream.iter_lines() if is_s3(self.path) else stream
            for number, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeError) as exc:
                    raise ConfigurationError(f"Invalid JSON on line {number} of {self.path!r}.") from exc
                if not isinstance(record, dict):
                    raise ConfigurationError(
                        f"JSONL line {number} must be an object with named fields."
                    )
                yield record


def file_dataset(path: str) -> Any:
    suffix = Path(urlsplit(path).path).suffix.lower()
    if suffix not in FILE_SUFFIXES:
        raise UnsupportedFeatureError(
            "Supported data files are .json, .jsonl, .pt, and .pth. "
            "For a local image directory, call .images(labels=...)."
        )
    if suffix == ".jsonl":
        return JsonLines(path)
    with open_binary(path) as stream:
        if suffix == ".json":
            try:
                values = json.load(stream)
            except (ValueError, UnicodeError) as exc:
                raise ConfigurationError(f"Invalid JSON data file {path!r}.") from exc
            if not isinstance(values, list) or any(not isinstance(item, dict) for item in values):
                raise ConfigurationError("A JSON data file must contain an array of record objects.")
            return SequenceDataset(values)
        # Use a seekable buffer for S3. Never permit arbitrary pickle objects.
        handle = io.BytesIO(stream.read()) if is_s3(path) else stream
        values = torch.load(handle, map_location="cpu", weights_only=True)
    if isinstance(values, torch.Tensor):
        return SequenceDataset(values)
    if isinstance(values, Mapping):
        return ColumnsDataset(values, tensors_only=True)
    raise ConfigurationError(
        "A tensor data file must contain a tensor or a mapping of string names to tensors."
    )


class ImageDirectory(Dataset):
    """Sorted local images, with explicit labels or no labels, decoded on access."""

    def __init__(self, path: str, *, labels: str, size: tuple[int, int] | None, mode: str):
        root = Path(path).expanduser()
        if not root.is_dir():
            raise ConfigurationError(f"Image source {path!r} must be a local directory.")
        self.size = size
        self.mode = mode
        self.labels = labels
        self.classes: tuple[str, ...] = ()
        self.samples: list[tuple[Path, int | None]] = []
        if labels == "unlabeled":
            self.samples = [(file, None) for file in self._files(root)]
        else:
            if any(file.is_file() and file.suffix.lower() in IMAGE_SUFFIXES for file in root.iterdir()):
                raise ConfigurationError(
                    "Class-subdirectory images must be inside root/<class>/; "
                    "the root contains images without an explicit class."
                )
            directories = sorted(
                (folder for folder in root.iterdir() if folder.is_dir() and not folder.name.startswith(".")),
                key=lambda folder: folder.name,
            )
            self.classes = tuple(folder.name for folder in directories)
            for index, folder in enumerate(directories):
                self.samples.extend((file, index) for file in self._files(folder))
        if not self.samples:
            raise ConfigurationError(f"No supported image files found in {path!r}.")

    @staticmethod
    def _files(root: Path) -> list[Path]:
        return sorted(
            (
                path for path in root.rglob("*")
                if path.is_file()
                and path.suffix.lower() in IMAGE_SUFFIXES
                and not any(part.startswith(".") for part in path.relative_to(root).parts)
            ),
            key=lambda path: path.relative_to(root).as_posix(),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Any:
        try:
            from PIL import Image
        except ImportError as exc:
            raise UnsupportedFeatureError(
                "Image directory decoding requires the optional Pillow dependency. Install Pillow on the worker."
            ) from exc
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert(self.mode)
            if self.size is not None:
                image = image.resize((self.size[1], self.size[0]), resample=Image.Resampling.BILINEAR)
            channels = 3 if self.mode == "RGB" else 1
            values = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
            values = values.reshape(image.height, image.width, channels).permute(2, 0, 1)
            values = values.to(dtype=torch.float32).div(255)
        return values if self.labels == "unlabeled" else (values, label)


def materialize_source(source: Source) -> Any:
    if source.kind == "path":
        return file_dataset(source.value)
    if source.kind == "images":
        return ImageDirectory(source.value, **dict(source.options))
    if source.kind == "factory":
        return adapt_values(source.value())
    if source.kind in {"native", "memory"}:
        return adapt_values(source.value)
    raise ConfigurationError(f"Unknown data source kind {source.kind!r}.")
