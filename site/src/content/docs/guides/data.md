---
title: Data pipelines
description: Declare paired, indexed, or streaming sources and configure ordered native worker batching.
---

`ml.data(...)` records immutable, ordered pipeline configuration. Each chained
method returns a new pipeline. Construction and serialization do not invoke
factories, call `get`, iterate samples, read files, decode images, or contact S3.
They can validate metadata such as paired column lengths. Supplied native objects
and collection contents remain caller-owned and may still be mutable.

## Choose a source

```python
import mlzo as ml
import torch

X = torch.arange(48, dtype=torch.float32).reshape(12, 4)
y = torch.arange(12) % 2
train = ml.data(X, y).batch(4)

# No Dataset subclass is needed.
def get_sample(i):
    return X[i], y[i]

indexed = ml.data(get=get_sample, length=len(X)).batch(4)

def generate():
    for i in range(len(X)):
        yield X[i], y[i]

streamed = ml.data(stream=generate).batch(4)
```

| Form | Contract |
| --- | --- |
| `data(X, y)` | Equally sized tensor, NumPy, or sequence columns, paired by leading sample index |
| `data(get=fn, length=N)` | A synchronous index function and a nonnegative integer length; supply both |
| `data(stream=factory)` | A synchronous zero-argument factory returning a fresh finite or infinite iterable per pass |
| `data(dataset)` | Native `Dataset` or `IterableDataset` |
| `data(loader)` | Existing native `DataLoader`; a no-operation build returns that exact loader |
| `data(source)` | A tensor, mapping of equally sized string-keyed columns, sequence, or other local iterable |
| `data(factory)` | A zero-argument factory returning a supported source at build time |
| `data(path)` | An explicit supported file or image-directory configuration |

MLZO unwraps tensor columns without reshaping them. A **single sequence of sample
pairs stays row-major**: `data([(x0, y0), (x1, y1)])` describes two samples, not
two columns. Use two positional arguments to declare columns. Source modes are
exclusive; mixing `get`/`length`, `stream`, and a positional source raises an error.

`build(epoch=0)` opens the configured source for execution. Ordinary source
factories run at build; stream factories create iterators for each pass, and
`get(i)` runs only when a sample is requested. JSON arrays and tensor files read
at build, whereas JSONL and sample decoding read during iteration. Building a
pipeline is therefore different from consuming samples. `iter(pipeline)` builds
epoch zero.

Remote runs require importable functions in the submitted project or a
transportable file/pipeline recipe. Local lambdas and closures can be convenient,
but arbitrary native objects and iterators are not automatically transportable.
Paths hidden inside factories remain the author's responsibility.

## Transform samples or current items

```python
def augment(sample):
    features, target = sample
    return features + 1, target

def is_valid(sample):
    return bool(torch.isfinite(sample[0]).all())

def summarize(batch):
    features, targets = batch
    return {"mean": features.mean(dim=0), "targets": targets}

pipeline = (
    ml.data(get=get_sample, length=len(X))
    .transform(augment)
    .filter(is_valid)
    .shuffle(seed=42, buffer=64)
    .batch(4)
    .map(summarize)
)
```

Use `transform(fn)` on individual samples before `batch`. `map(fn)` transforms
the current items, so a map after batching receives a batch. Each function takes
one complete item, including its tuple or mapping structure. Filtering is lazy
and gives the result unknown length. A shuffle after batching shuffles batches.
Pipeline order is part of the configuration; methods do not mutate earlier pipelines.

## Batch with native tensor outputs

```python
features, targets = next(iter(train))
assert isinstance(features, torch.Tensor)
assert isinstance(targets, torch.Tensor)
assert features.shape == (4, 4)
assert targets.shape == (4, 1)

native_shapes = ml.data(X, y).batch(4, normalize=False)
assert next(iter(native_shapes))[1].shape == (4,)
```

Default batching collates samples using native PyTorch rules, unwraps MLZO tensor
leaves, and then normalizes scalar/one-dimensional tensor leaves in the completed
batch. Outputs remain **native `torch.Tensor` leaves**. Feature vectors stack to
`(B, F)`, and scalar labels stack to `(B, 1)`. Samples are not normalized to
`(F, 1)` before stacking. Higher-rank batch leaves keep their shape.

`batch(size, normalize=False)` retains native collated shapes. A custom
`combine(samples)` controls the exact output shape; MLZO does not normalize its
result. `collate_fn` remains a legacy alias for `combine`. Supply only one.

```python
def combine(samples):
    features, targets = zip(*samples)
    return torch.stack(features), torch.stack(targets)

custom = ml.data(X, y).batch(4, combine=combine)
assert next(iter(custom))[1].shape == (4,)
```

## Shuffle, workers, and native loader options

```python
# Put this in an importable script with a guarded process entrypoint.
if __name__ == "__main__":
    parallel = ml.data(get=get_sample, length=len(X)).batch(
        4, shuffle=True, workers=4, prefetch=2,
        drop_last=True, pin_memory=False,
    )
```

Batch configuration accepts `shuffle`, `workers`, `prefetch`, `combine`, `sampler`,
`drop_last`, `pin_memory`, and `normalize`. Without worker configuration, loading
runs in the caller. `workers` uses native DataLoader worker processes for batching;
`prefetch=2` means two batches per worker, not two source samples. Positive prefetch
requires positive workers. Standalone `.workers(count)` and `.prefetch(count)`
remain available, as does the older `workers(..., prefetch_factor=...)` spelling.
On spawn platforms, worker functions and factories must be importable and pickleable,
and the script must guard process startup.

`shuffle(seed=42)` requires known finite upstream length for a full permutation.
For streams and filtered sources, use `shuffle(seed=42, buffer=1024)`.
`buffer_size` remains an alias for `buffer`; supply only one. Buffer shuffle uses
bounded memory and does not promise a uniform full permutation. Each epoch uses
`seed + epoch`; repeated builds reproduce that order only when the upstream
items and their order also agree. Native samplers and random user transforms
retain their own randomness rules.

A native `sampler` can select indexable samples in `batch(..., sampler=...)`.
Combining a sampler with shuffle is an error. For full DataLoader configuration,
including `batch_sampler` and other advanced native options, construct the loader
first:

```python
loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X, y), batch_size=4,
    num_workers=0, pin_memory=False,
)
adapted = ml.data(loader)
assert adapted.build() is loader
```

MLZO does not silently reconfigure an existing loader. Its sampler, batch size,
collation, and workers remain native settings. Downstream item operations consume
its existing batches; choose `loader.dataset` instead when MLZO should own batching.
The no-operation adapter also preserves the loader's native label shapes.

## Stream safely

A finite stream can complete an epoch. An infinite stream needs a finite training
step budget, such as `run.fit(steps=100, on="cpu")`; an epoch target alone cannot
finish an infinite pass. Unknown-length streams and filters require a shuffle buffer.

With multiple workers, `data(stream=factory)` requires an importable deterministic
factory with the explicit attribute `factory.mlzo_replayable = True`. Each worker
replays that sequence and takes a strided shard. This prevents duplicate output
samples across workers only when every replay produces the same sequence.
It can repeat source reads, so do not make this promise for a live, destructive,
or independently random source.

```python
# Define this factory at module scope in an importable project file.
def replayable_samples():
    for i in range(100):
        yield torch.tensor([float(i)]), i % 2

replayable_samples.mlzo_replayable = True
```

Without that promise, do not replicate a `stream` factory across workers.
Native `IterableDataset` authors are responsible for safe worker sharding in
their own iterator. There is no distributed sharding or exact cursor-resume
contract. One-shot iterators remain consumed and cannot resume.

## File formats

`data(path)` supports these explicit formats, locally or as one S3 object.

| Suffix | Contents | Loading behavior |
| --- | --- | --- |
| `.json` | An array of record objects, such as `[{"x": [1, 2], "y": 0}]` | Parsed in memory at build |
| `.jsonl` | One record object per nonblank line, such as `{"text": "hello", "label": 0}` | Streamed on iteration |
| `.pt`, `.pth` | One tensor, or a nonempty mapping of string names to tensors with equal leading dimensions | Loaded on CPU at build with `torch.load(weights_only=True)` |

For example, `torch.save({"x": inputs, "y": targets}, "samples.pt")` produces
records with `x` and `y` fields. Built-in supervised steps expect a pair, so map
these records to `(inputs, targets)` or use a custom `step(model, batch)`.
No pickle object loading or arbitrary serialized datasets are supported. A
list-valued JSON field remains a Python list; map it to a tensor explicitly if
the desired collation needs a tensor.

S3 paths must name a specific object with a supported suffix, for example
`s3://bucket/train.jsonl`. Prefix listing and format guessing are unsupported.
The S3 adapter imports `boto3` only when a file is opened and uses its normal
credential resolution. An omitted `boto3` dependency raises
`UnsupportedFeatureError` on use.

## Images and normalization

```python
pipeline = (
    ml.data("images/")
    .images(size=(224, 224), labels="class_subdirectories")
    .normalize("imagenet")
    .shuffle(seed=42)
    .batch(32)
)
```

Install `.[vision]` for Pillow, which MLZO imports only on sample access.
`images(path, ..., labels=...)` is also available from `mlzo.data`. Decoding
returns float32 tensors in `[0, 1]`, with shape `(channels, height, width)`.
`size` is `(height, width)` and applies bilinear resizing. Modes are `RGB` and
`L`. Supported file extensions are JPEG, PNG, BMP, GIF, TIFF, and WebP. Animated
files use the first frame.

The label mode is required:

- `labels="unlabeled"` recursively reads sorted image paths and returns tensors.
- `labels="class_subdirectories"` reads `root/<class>/...`, assigns integer
  labels from sorted immediate class directory names, and returns
  `(image, class_index)`. Images directly inside the root cause an error.

Hidden paths are excluded. No labels are inferred from filenames. S3 image
directories are unsupported.

`normalize(mean, std, dim=-3, key=None)` operates on a floating tensor or the
input of a `(tensor, target)` pair. For a record, specify an explicit field such
as `key="image"`. Statistics must match the selected dimension, or contain one
value to broadcast across it. Use `dim=-1` for feature vectors. ImageNet means
and standard deviations are for three-channel RGB input. Operations preserve
targets and do not mutate input tensors.

`tokenize(tokenizer, key=None)` calls the supplied tokenizer on an item, or
replaces one explicit record field with its result. No vocabulary or padding is
selected. Supply a custom `batch(..., combine=...)` for padding and masks.

## Resume and transport

Reiterable sources can restart an epoch. Partial epochs restart on training
resume; sampler, worker, and prefetch cursors are not restored exactly.

`to_spec(encode)` and `DataPipeline.from_spec(spec, decode)` describe pipeline
configuration without building it. Source values and individual operation/source
options pass through the supplied codec; structural metadata and loader flags
use ordinary values. A remote codec must reject unsupported native objects and
nonimportable callables. Indexed and streaming sources transport their
importable callables rather than sample contents. `local_paths()` identifies
explicit local paths, and `rewrite_local_paths(rewrite)` returns rewritten
configuration without file I/O. Paths hidden inside factories cannot be
identified automatically.

See [remote submission](../../remote/submission/) for source staging and the
[API reference](../../reference/api/data/) for generated signatures.
