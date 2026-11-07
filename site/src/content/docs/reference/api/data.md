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

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/data/pipeline.py#L576).

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

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/data/pipeline.py#L608).

## DataPipeline

Declared fields:

```python
source: Any
```

Immutable ordered pipeline configuration. Construction and serialization do not call sample functions or read data. A build opens sources; iteration requests samples. Source objects remain caller-owned.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/data/pipeline.py#L129).

### DataPipeline.map

```python
DataPipeline.map(function: Callable[[Any], Any]) -> DataPipeline
```

Apply a local callable to each current item, including batches.

### DataPipeline.transform

```python
DataPipeline.transform(function: Callable[[Any], Any]) -> DataPipeline
```

Transform individual samples. After batch(), use map() instead.

### DataPipeline.filter

```python
DataPipeline.filter(function: Callable[[Any], bool]) -> DataPipeline
```

Lazily retain samples with a scalar boolean predicate; length is unknown.

### DataPipeline.shuffle

```python
DataPipeline.shuffle(
    seed: int = 42,
    *,
    buffer: int | None = None,
    buffer_size: int | None = None,
) -> DataPipeline
```

Shuffle with seed + epoch. Streams need a bounded buffer.

Full indexable shuffles permute indices without reading samples. Finite
iterable shuffles buffer samples during iteration. Buffered shuffle is
bounded and is not a uniform full permutation. buffer_size is legacy.

### DataPipeline.batch

```python
DataPipeline.batch(
    size: int,
    drop_last: bool = False,
    collate_fn: Callable | None = None,
    *,
    shuffle: bool = False,
    workers: int | None = None,
    prefetch: int | None = None,
    combine: Callable | None = None,
    sampler: Any = None,
    pin_memory: bool = False,
    normalize: bool = True,
) -> DataPipeline
```

Build batches with native DataLoader workers and collation.

Default collation unwraps MLZO tensor leaves at their existing shape,
then shapes scalar/1D batch leaves to at least two axes after stacking.
normalize=False preserves native collated shapes. Custom combine or
collate_fn receives original samples and preserves its exact output.
Worker prefetch counts batches per worker. Construct a native DataLoader
for advanced options such as batch_sampler, then pass it to data() unchanged.

### DataPipeline.workers

```python
DataPipeline.workers(count: int, *, prefetch_factor: int | None=None) -> DataPipeline
```

Set native loading workers. Operations before batching run there.

### DataPipeline.prefetch

```python
DataPipeline.prefetch(factor: int) -> DataPipeline
```

Set native prefetch_factor, counting batches per worker.

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

Construct native loading lazily; shuffle uses seed + epoch.

Iterable workers batch their own shards, following native PyTorch
semantics, including drop_last per worker. Partial epochs have no exact
cursor replay guarantee. Generic native IterableDatasets own sharding.

### DataPipeline.to_spec

```python
DataPipeline.to_spec(encode: Callable[[Any], Any]) -> dict[str, Any]
```

Encode version 2 configuration. The one-argument encoder handles values.

A remote encoder should reject nonimportable callables and unsupported
native objects. Neither factories nor map functions are invoked here.

### DataPipeline.from_spec

```python
DataPipeline.from_spec(spec: Mapping, decode: Callable[[Any], Any]) -> DataPipeline
```

Decode without building. Version 1 keeps legacy batch shapes and shuffle order.

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
