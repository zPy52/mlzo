---
title: Data pipelines
description: Public MLZO 0.1.0a1 API signatures and behavior for data pipelines.
---

Read the [data pipelines guide](../../../guides/data/) for examples and usage constraints. Signatures and declared fields below come from MLZO `0.1.0a1` source. Type names follow their defining Python modules.

## data

```python
mlzo.data(
    source: Any = None,
    targets: Any = _MISSING,
    *,
    get: Callable[[int], Any] | None = None,
    length: int | None = None,
    stream: Callable[[], Iterable] | None = None,
) -> DataPipeline
```

Describe one source, input/target columns, an index function, or a stream.

get and length must be supplied together. stream creates a fresh iterator
for each pass. These modes cannot be combined. A single sequence of pairs
remains a sequence of samples; use two arguments to describe columns.
Existing pipelines are returned unchanged and native loaders retain their
identity on build. No sample function or stream factory is invoked here.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/data/pipeline.py#L535).

## images

```python
mlzo.images(
    path: str | Path,
    size: Sequence[int] | None = None,
    *,
    labels: str,
    mode: str = 'RGB',
) -> DataPipeline
```

Describe a local image directory with an explicit label convention.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/data/pipeline.py#L567).

## DataPipeline

Declared fields:

```python
source: Any
```

Immutable ordered pipeline configuration. Construction and serialization do not call sample functions or read data. A build opens sources; iteration requests samples. Source objects remain caller-owned.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/data/pipeline.py#L191).

### DataPipeline.map

```python
DataPipeline.map(function: Callable[[Any], Any]) -> DataPipeline
```

Apply a local callable to each current item, including batches.

### DataPipeline.shuffle

```python
DataPipeline.shuffle(seed: int, *, buffer_size: int | None=None) -> DataPipeline
```

Shuffle with seed + epoch. Streams require a positive buffer_size.

Full shuffle consumes the finite upstream sequence when iteration starts.
Buffered shuffle uses bounded memory and is not a uniform full shuffle.

### DataPipeline.batch

```python
DataPipeline.batch(size: int, drop_last: bool=False, collate_fn: Callable | None=None) -> DataPipeline
```

Group current items with torch default_collate or an explicit callable.

### DataPipeline.workers

```python
DataPipeline.workers(count: int, *, prefetch_factor: int | None=None) -> DataPipeline
```

Read indexable source samples in CPU workers, preserving source order.

Ordered pipeline operations run in the calling process. Generic streams
and native DataLoaders cannot be repartitioned by this alpha adapter.

### DataPipeline.prefetch

```python
DataPipeline.prefetch(factor: int) -> DataPipeline
```

Set the number of source items prefetched per worker. Requires workers > 0.

### DataPipeline.images

```python
DataPipeline.images(size: Sequence[int] | None=None, *, labels: str, mode: str='RGB') -> DataPipeline
```

Decode a local directory to float32 CHW tensors in [0, 1].

labels must be 'unlabeled' or 'class_subdirectories'. The latter returns
(image, integer_class) using sorted immediate directory names. size is
(height, width); mode is RGB or L. Pillow is imported only when decoding.

### DataPipeline.normalize

```python
DataPipeline.normalize(
    mean: Real | Sequence[Real] | str,
    std: Real | Sequence[Real] | None = None,
    *,
    dim: int = -3,
    key: str | None = None,
) -> DataPipeline
```

Normalize a floating tensor, a pair's input, or an explicit record field.

dim defaults to the image channel dimension. Use dim=-1 for features.
normalize('imagenet') selects RGB ImageNet mean and standard deviation.

### DataPipeline.tokenize

```python
DataPipeline.tokenize(tokenizer: Callable, *, key: str | None=None) -> DataPipeline
```

Apply the caller's tokenizer to each item or one explicit record field.

No vocabulary, padding, truncation, or attention mask is inferred. Use a
custom batch collate_fn for padding.

### DataPipeline.build

```python
DataPipeline.build(epoch: int=0) -> Iterable
```

Rebuild on the execution worker, with shuffle seeds offset by epoch.

Reiterable sources start a new pass. A supplied one-shot iterator keeps
its own consumed state. No exact cursor or restart guarantee is implied.

### DataPipeline.to_spec

```python
DataPipeline.to_spec(encode: Callable[[Any], Any]) -> dict[str, Any]
```

Encode configuration only. The one-argument encoder handles values.

A remote encoder should reject nonimportable callables and unsupported
native objects. Neither factories nor map functions are invoked here.

### DataPipeline.from_spec

```python
DataPipeline.from_spec(spec: Mapping, decode: Callable[[Any], Any]) -> DataPipeline
```

Reconstruct configuration with a one-argument decoder, without building.

### DataPipeline.local_paths

```python
DataPipeline.local_paths() -> tuple[Path, ...]
```

List explicit source paths as absolute Paths, without reading them.

Paths captured inside factories or callables cannot be discovered.

### DataPipeline.rewrite_local_paths

```python
DataPipeline.rewrite_local_paths(rewrite: Callable[[Path], str | Path]) -> DataPipeline
```

Return a pipeline with explicit paths replaced by rewrite(absolute_path).
