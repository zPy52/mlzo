# Data pipelines

`data(source)` creates immutable configuration. Chain methods return new
pipelines. Creating or serializing a pipeline does not invoke its factory,
iterate its data, read files, decode images, or contact S3. Supplied collection
contents and native objects remain caller-owned and may still be mutable.

```python
from mlzo.data import data

pipeline = data(records).map(transform).shuffle(seed=42).batch(32)
for batch in pipeline.build(epoch=1):
    ...
```

`build(epoch=0)` constructs the source on the execution worker. Each factory
is called once per build, with no arguments. `iter(pipeline)` builds epoch zero.
Map, normalization, tokenization, and batching run lazily in their declared
order. A map after batching receives a batch. A shuffle after batching shuffles
whole batches. Batching uses `torch.utils.data.default_collate` unless the caller
supplies `collate_fn`.

Sources can be native PyTorch `Dataset`, `IterableDataset`, or `DataLoader`
objects, sequences of records, tensors with a leading sample dimension, or
mappings of string field names to equally sized tensors/sequences. Other
iterables are supported locally. A factory must return one of these source
types. Use a named zero-argument factory in an importable project module for
remote runs. Arbitrary iterators and native objects are not automatically
transportable.

A native `DataLoader` without additional pipeline configuration is returned
unchanged by `build()`. Added operations consume its existing batches; they do
not change its sampler or batch size. Configure its workers directly. Providing
its dataset instead permits pipeline worker configuration.

## File formats

`data(path)` supports these explicit formats, locally or as one S3 object.

| Suffix | Contents | Loading behavior |
| --- | --- | --- |
| `.json` | An array of record objects, such as `[{"x": [1, 2], "y": 0}]` | Parsed in memory at build |
| `.jsonl` | One record object per nonblank line, such as `{"text": "hello", "label": 0}` | Streamed on iteration |
| `.pt`, `.pth` | One tensor, or a nonempty mapping of string names to tensors with equal leading dimensions | Loaded on CPU at build with `torch.load(weights_only=True)` |

For example, `torch.save({"x": inputs, "y": targets}, "samples.pt")` produces
records with `x` and `y` fields. No pickle object loading or arbitrary serialized
datasets are supported. A list-valued JSON field remains a Python list; map it
to a tensor explicitly if the desired collation needs a tensor.

S3 paths must name a specific object with a supported suffix, for example
`s3://bucket/train.jsonl`. Prefix listing and format guessing are unsupported.
The S3 adapter imports `boto3` only when a file is opened and uses its normal
credential resolution. An omitted `boto3` dependency raises
`UnsupportedFeatureError` on use.

## Images and normalization

```python
pipeline = (
    data("images/")
    .images(size=(224, 224), labels="class_subdirectories")
    .normalize("imagenet")
    .shuffle(seed=42)
    .batch(32)
)
```

Image decoding requires optional Pillow, imported only on sample access.
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
selected. Supply a custom `batch(..., collate_fn=...)` for padding and masks.

## Shuffling, workers, and restart limits

`shuffle(seed)` buffers the entire upstream sequence and requires a known
finite length. For JSONL or other streams use
`shuffle(seed, buffer_size=1024)`. Buffered shuffling uses bounded memory and
does not promise a uniform full permutation. Each shuffle uses `seed + epoch`.
Rebuilding with the same epoch reproduces that permutation when the upstream
items and their order agree. Native sampler randomness, source randomness, and
arbitrary user transforms retain their own semantics.

`workers(count, prefetch_factor=None)` reads indexable source samples in CPU
workers while preserving their order. Pipeline operations run in the calling
process, so their order is retained. JSON arrays, tensor files, local image
directories, sequences, and map-style datasets support workers. JSONL,
`IterableDataset`, arbitrary iterables, and preconfigured native loaders cannot
be repartitioned by this adapter. Unsupported configurations raise
`UnsupportedFeatureError`.

`prefetch(factor)` sets the number of source items prefetched per worker and
requires a positive worker count at build time. It does not create an
asynchronous iterator when workers are zero. Follow native PyTorch process
entrypoint rules when using workers on platforms that spawn processes.

Reiterable sources can start another pass. A supplied one-shot iterator stays
consumed; a factory can create a fresh iterator at each build. There is no exact
cursor resume guarantee for arbitrary iterators, native loaders, or prefetched
samples. No distributed sharding, caching, or GPU preprocessing is implemented.

## Transport and artifact staging

`pipeline.to_spec(encode)` and `DataPipeline.from_spec(spec, decode)` use version
1 configuration dictionaries. Each encoder/decoder receives exactly one value
per call. Source values and individual operation/source options go through the
codec. Structural metadata and loader flags are plain primitives. A remote
codec must reject unsupported native objects and nonimportable callables.
Neither direction builds the source or calls a dataset factory.

`local_paths()` returns a tuple of absolute `pathlib.Path` objects for explicit
local file/directory sources. `rewrite_local_paths(rewrite)` returns new
configuration after calling `rewrite(absolute_path)` once. It performs no file
I/O. Paths hidden in factories or transforms cannot be discovered or rewritten;
those callables are responsible for worker-accessible locations.
