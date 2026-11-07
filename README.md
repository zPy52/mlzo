# MLZO

MLZO `0.1.0a1` is a Python alpha for deferred PyTorch training. A run can execute on one local device or one enrolled remote worker. Planning describes the run before constructing lazy models or opening factory-backed data.

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
samples = torch.utils.data.TensorDataset(x, y)
batches = mlzo.data(samples).batch(16)
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

## Examples and commands

After installation, run these from the repository root:

```sh
python examples/local_classifier.py
python examples/explicit_updates.py
mlzo --help
python -m mlzo --version
```

The [remote project example](examples/remote_project/README.md) has an importable model and data factory, plus a guarded submit script. [Usage](docs/usage.md) covers the Python API, coordinator setup, and CLI. `--json` produces structured successes and errors; enrollment secrets go through protected token files or environment variables.

This alpha has no DDP, FSDP, pipeline execution, managed provisioning, S3 checkpoints, or remote inference. GPU and TLS execution paths remain unvalidated. Remote execution is for trusted, single-owner code and is not a hostile-code sandbox. Read the [support boundaries](docs/alpha-support.md) before remote use.

The original [plan.md](plan.md) describes the future product. It is not a list of capabilities available in this release. Changes are recorded in [CHANGELOG.md](CHANGELOG.md).
