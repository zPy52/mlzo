---
title: Data pipelines
description: Build ordered, deferred data pipelines for native datasets, files, images, and S3 objects.
---

`mlzo.data(source)` records an immutable pipeline configuration. Each chained method returns a new pipeline. Construction does not call factories, read files, decode images, contact S3, or iterate data. Native source objects remain caller-owned and can still be mutable.

## Declare operations in order

```python
import mlzo
import torch

records = [{"x": torch.tensor([float(i)]), "y": i % 2} for i in range(12)]
pipeline = mlzo.data(records).shuffle(seed=7).batch(4)
for batch in pipeline.build(epoch=0):
    print(batch["x"].shape)
```

`build(epoch=0)` opens the source for execution. A factory is called once per build with no arguments. `iter(pipeline)` builds epoch zero. A map after a batch receives a batch, and a shuffle after a batch shuffles complete batches. Default batching uses PyTorch's `default_collate`.

## Choose a source

| Source | Notes |
| --- | --- |
| Dataset or IterableDataset | Native PyTorch source semantics |
| DataLoader | Without extra configuration, `build()` returns it unchanged. Further operations consume its existing batches |
| Tensor | Iterates along the leading sample dimension |
| Mapping | String field names with equally sized tensor or sequence values |
| Sequence | Local records |
| Other iterable | Local use; one-shot iterators remain consumed |
| Zero-argument factory | Creates a fresh source at build time |
| Path | One of the explicit supported file formats below |

Remote runs need an importable factory or a transportable file/pipeline recipe. Arbitrary native datasets and iterators are not automatically serialized.

## Read files

| Suffix | Required contents | Read behavior |
| --- | --- | --- |
| `.json` | An array of record objects | Parses into memory at build |
| `.jsonl` | One record object per nonblank line | Streams records during iteration |
| `.pt`, `.pth` | A tensor or nonempty string-to-tensor mapping with equal leading dimensions | Loads on CPU using `weights_only=True` |

```python
torch.save({"x": torch.randn(16, 4), "y": torch.zeros(16, dtype=torch.long)}, "samples.pt")
pipeline = mlzo.data("samples.pt").batch(4)
```

Tensor mappings produce field-based records. Built-in supervised steps require an `(inputs, targets)` pair, so map records to a pair or use a custom `step(model, batch)`. A list-valued JSON field remains a Python list. Map it to a tensor explicitly when tensor collation is needed.

For S3, install `.[s3]` and name one object, such as `mlzo.data("s3://bucket/train.jsonl")`. The adapter uses boto3's normal credential resolution when opening the object. Prefix listing, format guessing, and S3 image directories are unsupported.

## Decode images

```python
pipeline = (
    mlzo.data("images/")
    .images(size=(224, 224), labels="class_subdirectories")
    .normalize("imagenet")
    .shuffle(seed=7)
    .batch(32)
)
```

Install `.[vision]`. Images become float32 tensors in `[0, 1]` with shape `[channels, height, width]`. Size uses `(height, width)` with bilinear resizing. Modes are `RGB` and `L`.

Choose the label mode explicitly. `unlabeled` returns tensors from sorted recursive paths. `class_subdirectories` returns `(image, class_index)` using sorted immediate class directory names; files directly under the root are an error. Hidden paths are excluded. JPEG, PNG, BMP, GIF, TIFF, and WebP are supported; animated files use the first frame.

## Normalize and tokenize

`normalize(mean, std, dim=-3, key=None)` transforms a floating tensor or the input of a tensor/target pair without mutating it. For records, choose a field with `key="image"`. Use `dim=-1` for feature vectors. Statistics match the selected dimension or contain one broadcast value. The `imagenet` preset expects three-channel RGB.

`tokenize(tokenizer, key=None)` calls your tokenizer on each item or a selected record field. MLZO selects no vocabulary, padding, or masks. Supply `batch(..., collate_fn=...)` for those decisions.

## Shuffle and load in workers

`shuffle(seed=7)` buffers the complete upstream source and requires a known finite length. For JSONL and streams, use `shuffle(seed=7, buffer_size=1024)`. Buffered shuffling bounds memory but does not promise a uniform full permutation. The epoch seed is `seed + epoch`.

`workers(count, prefetch_factor=None)` loads indexable source samples in CPU worker processes, preserving order. Pipeline transforms execute in the caller. JSON arrays, tensors, local images, sequences, and map-style datasets support this adapter. JSONL, IterableDataset, arbitrary iterables, and existing DataLoaders cannot be repartitioned by it.

`prefetch(factor)` requires positive workers at build time. Follow PyTorch's guarded process-entrypoint rules on platforms that spawn processes.

## Resume and transport

Reiterable sources can restart an epoch. One-shot iterators cannot resume. There is no exact arbitrary cursor, native sampler, or prefetch replay guarantee.

`to_spec(encode)` and `DataPipeline.from_spec(spec, decode)` convert version-one configuration without building it. `local_paths()` identifies explicit local paths; `rewrite_local_paths(rewrite)` returns rewritten configuration without file I/O. Paths hidden inside factories cannot be discovered automatically. See [remote submission](../../remote/submission/) for staging behavior and the [API reference](../../reference/api/data/) for exact method signatures.
