---
title: Tensor and data use cases
description: Copy small examples for tensor operations, paired data, custom samples, streams, and native DataLoaders.
---

Use `import mlzo as ml` for the examples below. After [installing MLZO](../installation/),
run the snippets in order in one Python process. They use synthetic CPU data
and require no optional dependencies. The worker example is a separate script.

## Keep dimensions while calculating

```python
import mlzo as ml
import torch
from torch import nn

values = ml.tensor([[1, 2], [3, 4]]).astype("float32").on("cpu")
row_means = values.mean(axis=1)
centered = values - row_means
assert row_means.shape == (2, 1)
assert centered.raw.tolist() == [[-0.5, 0.5], [-0.5, 0.5]]
assert values[0].shape == (1, 2)
assert values[:, 0].shape == (2, 1)

column = ml.tensor([1., 2., 3.])
assert column.shape == (3, 1)
vector = column.remove_dim(1)
assert vector.shape == (3,)
assert vector.add_dim(0).shape == (1, 3)
assert column.transpose().shape == (1, 3)
```

`ml.tensor` returns an MLZO wrapper. Scalars become `(1, 1)` and vectors become
`(N, 1)`. Basic integer indexing and reductions keep dimensions, so a row mean
broadcasts across its row. Use `keepdims=False` when reducing should remove an
axis, or `normalize=False` when creating a tensor should preserve its input rank.
Explicit `reshape`, `flatten`, and `remove_dim` keep their requested shapes.
Integer and boolean means use PyTorch's default floating dtype.

## Reduce by name and call PyTorch

```python
sequence = ml.tensor(
    torch.arange(24, dtype=torch.float32).reshape(2, 3, 4),
    axes=("batch", "time", "feature"),
)
pooled = sequence.mean("time", keepdims=False)
assert pooled.axes == ("batch", "feature")
assert pooled.shape == (2, 4)
assert sequence.align("time", "batch", "feature").shape == (3, 2, 4)

linear = nn.Linear(4, 2)
logits = linear(pooled.raw)
assert logits.shape == (2, 2)

native = torch.tensor([1., 2., 3.], requires_grad=True)
wrapped = ml.wrap(native)
assert wrapped.raw is native
assert wrapped.shape == (3,)
(wrapped * wrapped).sum().backward()
torch.testing.assert_close(native.grad, 2 * native.detach())
```

Axis names match the final tensor shape. Use `.raw` for direct `nn.Module`
calls, arbitrary `torch.*` functions, and other native APIs. `ml.wrap(native)`
preserves the native tensor's shape, storage, and autograd graph. MLZO training
unwraps inputs itself. See [tensors and transforms](../../guides/transforms/)
for named broadcasting, advanced indexing, and migration details.

## Train from feature and label columns

```python
generator = torch.Generator().manual_seed(7)
X = torch.randn(12, 4, generator=generator)
y = (X.sum(dim=1) > 0).long()

source = ml.data(X, y)
batches = source.shuffle(seed=7).batch(4)
features, labels = next(iter(batches))
assert isinstance(features, torch.Tensor)
assert features.shape == (4, 4)
assert labels.shape == (4, 1)

run = ml.train(ml.Lazy(nn.Linear, 4, 2), batches,
               loss="cross_entropy", seed=7)
history = run.fit(epochs=1, on="cpu")
assert history.counters["iteration"] == 3
```

`data(X, y)` pairs equally sized columns by sample index. Pipeline methods
return new configurations, so `source` remains unbatched. Construction, `train`,
and `plan` do not read samples or construct lazy models. Iteration and `fit`
execute the configured work.

Default batching returns native PyTorch tensors. Feature vectors stack to
`(B, F)` and scalar labels to `(B, 1)`. MLZO's `loss="cross_entropy"` adapter
accepts these integral column labels for `(B, C)` logits. For direct PyTorch
loss calls, retain native label shapes:

```python
native_batches = ml.data(X, y).batch(4, normalize=False)
features, labels = next(iter(native_batches))
assert labels.shape == (4,)
loss = nn.CrossEntropyLoss()(nn.Linear(4, 2)(features), labels)
assert loss.ndim == 0
```

## Load samples with a function

```python
def get_sample(i):
    return ml.wrap(X[i]), y[i]

indexed = ml.data(get=get_sample, length=len(X)).batch(4)
features, labels = next(iter(indexed))
assert features.shape == (4, 4)
assert labels.shape == (4, 1)
assert len(indexed) == 3
```

Supply `get` and `length` together. MLZO calls `get(i)` when it needs a sample;
you do not need a `Dataset` subclass. `ml.wrap(X[i])` keeps each feature vector
at `(F,)`. Constructing it with `ml.tensor(X[i])` would instead make it `(F, 1)`
and produce a batch of shape `(B, F, 1)`.

## Transform samples and process batches

```python
def scale(sample):
    features, label = sample
    return features / 10, label

def is_finite(sample):
    return bool(torch.isfinite(sample[0]).all())

def as_record(batch):
    features, labels = batch
    return {"features": features, "labels": labels}

processed = (
    ml.data(X, y)
    .transform(scale)
    .filter(is_finite)
    .shuffle(seed=7, buffer=8)
    .batch(4)
    .map(as_record)
)
record = next(iter(processed))
assert record["features"].shape == (4, 4)
assert record["labels"].shape == (4, 1)
```

`transform` takes one sample before batching. `map` takes the current item,
which is a batch here. Each function receives the complete pair or record.
Filtering makes the length unknown even if every sample passes. Use a shuffle
buffer after filtering and `fit(steps=...)` for training. This record output
also needs a custom `step(model, batch)` to select its fields; built-in
supervised steps consume an input/target pair.

## Train from a stream

```python
def samples():
    for i in range(len(X)):
        yield X[i], y[i]

streamed = ml.data(stream=samples).shuffle(seed=7, buffer=8).batch(4)
stream_run = ml.train(ml.Lazy(nn.Linear, 4, 2), streamed,
                      loss="cross_entropy", seed=7)
stream_history = stream_run.fit(steps=5, on="cpu")
assert stream_history.counters["iteration"] == 5
```

Pass a zero-argument factory that returns a fresh iterable for each pass. This
finite stream restarts after its first three batches to reach five steps.
An infinite stream uses the same API. Both need a finite step budget because
MLZO does not know their batch count. Buffered shuffle holds at most the
configured number of samples and does not promise a uniform full permutation.

This example loads in the caller. Multiworker stream factories require an
importable deterministic factory marked `mlzo_replayable=True`; each worker
replays it and takes a strided shard. See [stream safety](../../guides/data/#stream-safely)
before using workers with a stream.

## Pad variable-length samples

```python
sequences = [torch.tensor([1, 2]), torch.tensor([3, 4, 5])]

def pad_samples(samples):
    return nn.utils.rnn.pad_sequence(samples, batch_first=True,
                                      padding_value=0)

padded = ml.data(sequences).batch(2, combine=pad_samples)
assert next(iter(padded)).tolist() == [[1, 2, 0], [3, 4, 5]]
```

`combine` receives the original samples and controls the exact batch output.
MLZO leaves its shapes unchanged. If samples contain MLZO wrappers, unwrap
them inside the function before calling native PyTorch operations.

## Configure workers or keep an existing loader

Save this complete example as `worker_data.py`, then run `python worker_data.py`:

```python no-exec
import mlzo as ml
import torch

def get_sample(i):
    return torch.full((4,), float(i)), i % 2

if __name__ == "__main__":
    batches = ml.data(get=get_sample, length=12).batch(
        4, shuffle=True, workers=2, prefetch=2,
    )
    sample_count = 0
    batch_count = 0
    for features, labels in batches:
        assert features.shape == (4, 4)
        assert labels.shape == (4, 1)
        sample_count += len(features)
        batch_count += 1
    assert (sample_count, batch_count) == (12, 3)
    print(f"Loaded {sample_count} samples in {batch_count} batches")
```

Worker functions must be importable and pickleable on spawn platforms. Keep
them at module scope and guard the entrypoint as shown. `prefetch=2` queues two
batches per worker and requires positive `workers`. These are local loading
processes; they do not distribute training across machines.

For native options such as `batch_sampler`, configure a PyTorch `DataLoader`
and pass it to MLZO:

```python
dataset = torch.utils.data.TensorDataset(X, y)
sampler = torch.utils.data.BatchSampler(
    torch.utils.data.SequentialSampler(dataset), batch_size=4, drop_last=False,
)
loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
adapted = ml.data(loader)
assert adapted.build() is loader
assert next(iter(adapted))[1].shape == (4,)
```

A no-operation adapter returns the original loader with its existing batches,
collation, and native label shapes. Use `ml.data(loader.dataset)` when MLZO
should configure batching. See the [data pipeline guide](../../guides/data/)
for samplers, file sources, images, and remote transport requirements.
