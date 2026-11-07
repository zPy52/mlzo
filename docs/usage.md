# Using the Python alpha

Install from the repository with `python -m pip install -e .` in a Python 3.12+ virtual environment. Optional extras are `vision` for Pillow, `s3` for boto3 dataset reads, and `rl` for Gymnasium environment IDs. The [README](../README.md) has a ten-line CPU quickstart.

## Describe, plan, execute

```python
import mlzo
from torch import nn

model = mlzo.Lazy(nn.Linear, 4, 2)
# `samples` is an existing dataset or an importable zero-argument factory.
batches = mlzo.data(samples).shuffle(seed=42).batch(16)
run = mlzo.train(model, batches, loss="cross_entropy",
                 optimizer=mlzo.opt("adamw", lr=1e-3), seed=42)
print(run.plan(on="cpu").explain())
history = run.fit(epochs=2, on="cpu")
print(run.status().to_dict())
```

`train()` records configuration. A plan does not construct lazy models, call training/data factories, consume batches, or run a forward pass. It can inspect device availability and query a remote cluster. Memory estimates remain unknown when they require execution. `Lazy` does not defer objects you have already constructed in its arguments; put whole-model construction in a factory when that matters.

`fit()` accepts a total epoch target or a total iteration-step target. Steps count calls to the algorithm, which may differ from optimizer updates. Local execution is synchronous. `wait=False` is supported for durable remote submissions. `parallel="auto"` or `"single"` uses one worker/device; distributed strategies raise an unsupported error.

Built-in supervised losses include `cross_entropy` and `mse`. For other batch structures, pass `step(model, batch)` instead of `loss=`. Subclass `Training`, implement `step()`, and store resumable tensor/primitive state in `self.state` for more control. A scalar loss or `Step(loss=...)` requests automatic optimization. Calling `self.update(name, loss)` performs that named update immediately; then return metrics without an automatic loss. See [explicit_updates.py](../examples/explicit_updates.py).

Optimizer specs support stable parameter selectors such as `mlzo.opt("sgd", params="encoder", lr=0.01)`. `mlzo.schedules` offers constant, linear, cosine, and exponential schedules, which advance on successful optimizer updates. Explicit named updates require `accumulate=1`. `grad`, `vmap`, `jacrev`, other transforms, and `jit` preserve their PyTorch semantics.

## Tensor shapes and native boundaries

`mlzo.tensor(values)` returns `Tensor`, or `NamedTensor` with `axes=...`.
Construction uses `torch.as_tensor`; scalars normalize to `(1, 1)`, and vectors
to `(N, 1)`. `normalize=False` preserves the input rank. `mlzo.wrap(native)`
retains the native tensor's shape and identity, and `.raw` exposes it explicitly.
Reductions default to `keepdims=True`; wholly basic integer indexing keeps
singleton dimensions. Advanced/boolean indices use whole-index native semantics.
Explicit reshape, flatten, and singleton removal keep their result shape.

```python
column = mlzo.tensor([1., 2., 3.])
vector = mlzo.tensor([1., 2., 3.], axes=("feature",), normalize=False)
assert column.shape == (3, 1)
assert vector.shape == (3,)
native_vector = vector.raw
```

Axis names must match the final normalized shape. The wrapper exposes a limited
set of methods and has no transparent torch dispatch. Native modules, optimizers,
function transforms, and compiled regions use `.raw`. MLZO training and deployment
unwrap leaves at their own boundaries without reshaping; `fit(on=...)` transfers
batches. Code that expected the previous native default can use
`mlzo.tensor(values, normalize=False).raw`. Named reductions that previously
dropped axes now need `keepdims=False`. See [tensor details and migration](../src/mlzo/transforms/README.md).

The supervised `loss="cross_entropy"` adapter, also used for supported
`nn.CrossEntropyLoss` instances passed as `loss=`, squeezes only integral `(B, 1)`
class targets for `(B, C)` logits. Dense and probability targets keep their shapes,
and dtypes are not coerced. Custom losses and direct native loss calls keep the
supplied target shape; use `target.squeeze(1)` explicitly in a custom classification
step when needed. There is no separate `mlzo.loss` constructor.

## Data, checkpoints, and inference

`data(X, y)` pairs equally sized columns. `data(get=fn, length=N)` defines
indexed samples without a subclass, and `data(stream=factory)` defines a fresh
finite or infinite iterable per pass. Existing datasets/loaders, sequences,
paths, and zero-argument source factories remain supported. A single sequence
of sample pairs stays row-major. Construction does not call user sample functions
or consume data; build opens the source, and iteration requests samples.

`transform(fn)` operates on samples before batching, while `map(fn)` operates
on the current items, including batches. `filter(fn)` is lazy and makes length
unknown. Default batching produces native tensor leaves, normalizing only after
collation: feature vectors become `(B, F)`, and scalar labels become `(B, 1)`.
Use `batch(..., normalize=False)` for native shapes or `combine=fn` for exact
custom shapes. The legacy `collate_fn` spelling remains supported.

```python
import torch

X = torch.randn(16, 4)
y = (X.sum(dim=1) > 0).long()
train_data = mlzo.data(X, y).batch(4)
classifier = mlzo.train(mlzo.Lazy(nn.Linear, 4, 2), train_data,
                        loss="cross_entropy")
classifier.fit(epochs=1, on="cpu")
```

`batch(64, shuffle=True, workers=4, prefetch=2)` uses native worker batching;
prefetch counts batches per worker. `sampler` conflicts with shuffle. Construct a
native `DataLoader` first and pass it to `data(loader)` for advanced options such
as `batch_sampler`. A no-operation build returns the original loader; MLZO does
not silently reconfigure it. Buffered shuffle accepts `buffer` and the legacy
`buffer_size` alias, uses `seed + epoch`, and is bounded rather than a uniform
full permutation. Multiworker stream factories need an importable deterministic
factory with `mlzo_replayable=True` for strided sharding. Native IterableDataset
authors own worker sharding. Infinite streams need a step budget.

Local JSON/JSONL and restricted tensor files remain supported. Image directories
require explicit label mode and the `vision` extra. S3 reads require one object
path and the `s3` extra. See [data details](../src/mlzo/data/README.md) for loading
boundaries and examples.

```python
policy = mlzo.checkpoint("outputs/checkpoints", every=100, keep_last=3)
# Configure periodic saves with train(..., checkpoint=policy).
snapshot = run.checkpoint(to=policy.to)
print(run.evaluate(batches))
artifact = run.export(to="outputs/classifier")
model = mlzo.load_artifact(artifact, device="cpu")
with mlzo.deploy(artifact, device="cpu") as predictor:
    predictions = predictor.predict(inputs)
```

Checkpoint intervals count optimizer update events and save at safe boundaries. Use separate roots per run because retention applies to the whole root. Checkpoints contain restricted full training state with checksums and an atomic manifest. S3 checkpoints and per-block checkpoints are unsupported.

Restore with a new compatible run and `new_run.fit(epochs=3, on="cpu", resume=snapshot)`. A checkpoint directory also works. `resume="latest"` requires a configured checkpoint policy. Model, optimizer, scheduler, RNG, algorithm state, and counters are restored; a partial data epoch restarts. Exact sampler/worker cursor replay is not provided, and one-shot iterators cannot resume. Keep the same algorithm, optimizer ownership, precision, and compatible schedule configuration.

Inference exports are separate from resumable checkpoints. Export directories are immutable; choose a fresh destination for changed weights. A stored importable recipe reconstructs the model. A weights-only artifact requires a compatible zero-argument `factory=` when loading or deploying. Factory code is trusted executable Python. Deployment runs synchronously in the local process; remote deployment is unsupported. See [inference details](../src/mlzo/inference/README.md).

`mlzo.dqn` supplies a small step-driven DQN recipe. Environment IDs load Gymnasium only at execution. Its replay/target state can resume, but the environment resets and an unfinished episode is discarded. Read [the recipe limits](../src/mlzo/recipes/README.md) before use.

## Start a coordinator and worker

The CLI and Python API call the same cluster and worker services. These commands demonstrate loopback development on one machine. In the first terminal:

```sh
mlzo coordinator serve --directory .mlzo/coordinator --host 127.0.0.1 --port 8787
```

The service persists SQLite state and credentials in that directory. It creates mode-0600 `admin.token` and `submitter.token` files without printing their contents. `--token-file` and `--submitter-token-file` select other paths. Existing files take precedence; otherwise `MLZO_TOKEN` and `MLZO_SUBMITTER_TOKEN` supply credentials, or the service generates them. Keep the directory for restarts.

In a second terminal, configure the admin client, export an invitation, and start a CPU worker:

```sh
mlzo cluster create demo --endpoint http://127.0.0.1:8787 --token-file .mlzo/coordinator/admin.token
mlzo cluster invite demo --token-file .mlzo/worker-invite.token
mlzo worker join --token-file .mlzo/worker-invite.token --directory .mlzo/worker --device cpu
mlzo worker serve --directory .mlzo/worker
```

`cluster create` saves configuration; it does not start a coordinator or provision hardware. Invitations expire after 900 seconds and allow one join by default. Transfer an invitation file privately to enroll another machine. `worker join` also accepts `MLZO_INVITE_TOKEN`, and `--serve` starts serving after enrollment. No positional secret is accepted. Invitation output requires a new `--token-file` path or explicit `--show-token`; only the latter prints its secret. Invitation representations and public dictionaries omit the secret.

Named client configurations default to `~/.mlzo/clusters`; worker identity defaults to `~/.mlzo/worker`. `MLZO_HOME` or `--home` changes that root. `--directory` gives a worker an explicit persistent identity/job directory and should be reused for serve, status, and leave. Multiple services must not share one worker identity. Use `submitter.token` for a training-only client configuration and keep admin credentials separate.

For a different machine, use a reachable HTTPS endpoint and start the coordinator with `--cert-file server.pem --key-file server.key`. Set `--endpoint` when its advertised origin differs from the bind address. Private CAs require an explicit trusted `--ca-file` on each client and worker. Invitation CA metadata does not establish trust. HTTP is restricted to loopback. GPU and TLS paths have not been validated for this release.

The remote supervisor requires POSIX process groups and file locking. Jobs run in worker-side subprocesses and virtual environments, using the worker's installed MLZO/PyTorch. Project dependencies may require installation and network access on the worker. The [synthetic remote project](../examples/remote_project/README.md) declares no extra dependencies and does not download data. This service executes trusted single-owner code and is not a hostile-code sandbox.

## Submit and reconnect

Use importable constructors and data factories in the project snapshot:

```python
cluster = mlzo.cluster("demo")
run.fit(epochs=2, on=cluster, code="path/to/project", wait=False)
saved_id = run.id
attached = mlzo.run(saved_id, on=mlzo.cluster("demo"))
print(attached.status().to_dict())
history = attached.wait()
```

`code=` is a directory. Source archives exclude common credential, environment, and output paths and are bounded to 100 MiB. Explicit pipeline file sources can be staged. Paths hidden inside factories remain the author's responsibility. Arbitrary pickle, closures, eager model objects, and custom `Training` instances cannot be submitted. Use an importable zero-argument training factory for custom algorithms. Keep the assigned worker service and its persistent job directory available after completion if you need an export.

Run the guarded example with `python examples/remote_project/submit.py --cluster demo --wait`. Once you have its run ID, the following commands use saved client configuration:

```sh
mlzo cluster status demo
mlzo cluster workers demo
mlzo run status RUN_ID --cluster demo --json
mlzo run events RUN_ID --cluster demo --after 0
mlzo run checkpoint RUN_ID --cluster demo
mlzo run export RUN_ID --cluster demo --to /absolute/worker/path/artifact
mlzo run cancel RUN_ID --cluster demo
mlzo checkpoint list outputs/checkpoints
```

Remote manual checkpoints require a completed run. Without `--to`, the command returns its retained checkpoint reference. With `--to`, checkpoint/export destinations and returned paths belong to the assigned worker. These commands do not download artifacts. Remote resume and remote evaluation are unsupported. `mlzo run resume RUN_ID --cluster demo` returns an explicit error.

Use `mlzo cluster drain demo WORKER_ID`, `remove demo WORKER_ID`, or `shutdown demo` to administer membership. Shutdown drains and stops worker services while leaving the coordinator running; interrupt its foreground command to stop the coordinator. `mlzo worker leave --directory .mlzo/worker` is graceful by default, with `--no-graceful` requesting cancellation. Scaling is unsupported. Worker status reports persisted local metadata and the current Python object's service state; use `cluster workers` for coordinator membership and liveness, since a fresh CLI object cannot inspect another process's in-memory state.

`mlzo` and `python -m mlzo` are equivalent. `--json` is accepted anywhere and wraps output as `{"ok": true, "result": ...}` or `{"ok": false, "error": ...}`. Services emit a startup record before blocking. Error text is deliberately limited and remote error details are omitted to avoid exposing credentials. `--help` lists exact options. See [alpha support](alpha-support.md) for the remaining boundaries.
