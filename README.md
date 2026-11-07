# MLZO

MLZO `0.1.0a1` is a Python alpha for deferred PyTorch training. A run can execute on one local device or one enrolled remote worker. Planning describes the run before constructing lazy models or opening factory-backed data.

Read the [documentation](https://zpy52.github.io/mlzo/) for installation, tutorials,
training and remote-worker guides, the Python API, and the CLI reference.
The open-source [documentation site](site/README.md) also runs locally.

## Install locally

Use Python 3.12 or newer in an activated virtual environment, from this directory:

```sh
python -m pip install -e .
# Optional image decoding, S3 datasets, and Gymnasium environments:
python -m pip install '.[vision,s3,rl]'
```

PyTorch is required. Pillow, boto3, and Gymnasium are optional and load only when used. The Intel macOS dependency marker keeps NumPy below 2 for the legacy PyTorch wheel. The declared dependency range has not been validated across platforms or versions. MLZO uses the [MIT license](LICENSE). The package has not been published; see [publishing](docs/publishing.md).

## A small CPU run

This synthetic example does not download data:

```python
import torch
from torch import nn
import mlzo
x = torch.randn(64, 4)
y = (x.sum(dim=1) > 0).long()
batches = mlzo.data(x, y).batch(16)
run = mlzo.train(mlzo.Lazy(nn.Linear, 4, 2), batches, loss="cross_entropy", seed=7)
print(run.plan(on="cpu").explain())
history = run.fit(epochs=2, on="cpu")
```

`Lazy` records its factory and arguments. `train()` and `plan()` do not call that factory or consume batches. `fit()` materializes the model and binds its optimizer. Models you construct eagerly have already allocated their weights.

Save training state, evaluate, and export a separate inference artifact:

```python
snapshot = run.checkpoint(to="outputs/checkpoints")
print(run.evaluate(batches))
artifact = run.export(to="outputs/classifier")
with mlzo.deploy(artifact, device="cpu") as predictor:
    print(predictor.predict(x[:2]))
```

Checkpoints support local full-state storage and epoch-restart resume. Inference artifacts contain weights and, when available, an importable construction recipe. Use a new export directory for changed weights.

## Tensors and data sources

`mlzo.tensor(values)` returns a lightweight `Tensor` wrapper. Scalars become
`(1, 1)` and vectors become `(N, 1)` by default. Reductions preserve dimensions,
and basic integer indexing uses singleton slices. Explicit shape operations
keep their requested shape. Use `normalize=False` to retain a native input
shape, `mlzo.wrap(native)` to preserve an existing tensor's shape and identity,
and `.raw` for direct PyTorch calls. Optional axis names produce `NamedTensor`.
See [tensor conventions and migration](src/mlzo/transforms/README.md).

```python
vector = mlzo.tensor([1., 2., 3.], normalize=False)
assert vector.shape == (3,)
native = vector.raw
matrix = mlzo.tensor([[1, 2], [3, 4]])
assert matrix.mean(axis=1).raw.tolist() == [[1.5], [3.5]]
indexed = mlzo.data(get=lambda i: (x[i], y[i]), length=len(x)).batch(16)
```

Integer and boolean `mean()` inputs use `torch.get_default_dtype()` when `dtype`
is omitted or `None`. Construction and sums retain their native dtype. Explicit
mean dtypes and floating/complex inputs follow PyTorch; `.raw.mean()` stays native.

`data(x, y)` pairs columns without a dataset subclass. `get=...` with `length=...`
defines indexed samples; `stream=...` supplies a fresh iterable per pass.
Pipelines also accept native datasets/loaders, sequences, paths, and factories.
`transform` operates on samples before batching; `map` operates on current items,
including batches. Default batches contain native tensor leaves, with feature
vectors stacked as `(B, F)` and scalar labels as `(B, 1)`. Use
`batch(..., normalize=False)` for native batch shapes. See [data details](src/mlzo/data/README.md)
for workers, custom collation, streams, and advanced DataLoader options.

Buffered shuffle of indexed data uses native worker-local batches. Pipeline and
built-loader lengths count each worker's partial batch and `drop_last` behavior.
Unknown-length streams, filters, and native iterable sources with unknown worker
shards require `fit(steps=...)`; epoch-only training requires a known batch count.

## Examples and commands

After installation, run these from the repository root:

```sh
python examples/local_classifier.py
python examples/explicit_updates.py
mlzo --help
python -m mlzo --version
```

For the full test suite, install the test dependencies first:

```sh
python -m pip install -e '.[test]'
python -m pytest -q
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and package checks.

The [remote project example](examples/remote_project/README.md) has an importable model and data factory, plus a guarded submit script. [Usage](docs/usage.md) covers the Python API, coordinator setup, and CLI. `--json` produces structured successes and errors; enrollment secrets go through protected token files or environment variables.

This alpha has no DDP, FSDP, pipeline execution, managed provisioning, S3 checkpoints, or remote inference. GPU and TLS execution paths remain unvalidated. Remote execution is for trusted, single-owner code and is not a hostile-code sandbox. Read the [support boundaries](docs/alpha-support.md) before remote use.

The original [plan.md](plan.md) describes the future product. It is not a list of capabilities available in this release. Changes are recorded in [CHANGELOG.md](CHANGELOG.md).

For isolated changes, CPU tests, PR evidence, and the owner's review process,
see [Contributing](CONTRIBUTING.md).
