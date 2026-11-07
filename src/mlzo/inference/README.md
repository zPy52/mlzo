# Local inference in 0.1.0a1

`export_model(model, to=path, recipe=None, block=None)` writes the artifact to
`path` itself, using a sibling staging directory followed by atomic rename.
The API never changes a committed directory. Re-exporting the identical
manifest and weights reuses the directory; different or invalid existing
contents cause an error. Local writers use a fail-fast exclusive sibling lock.
After a crashed export, confirm the writer has stopped before removing its
`.mlzo-export.lock` or `.staging-*` directory. Atomic rename must be supported by
the destination filesystem. Export should run at a completed training boundary.

The version-one `mlzo.inference` manifest contains an artifact identity, the
MLZO/Python/PyTorch versions, the weights SHA256 and size, the reconstruction
mode, and the selected block path when applicable. `weights.pt` contains only a
restricted CPU copy of `model.state_dict()`, including state_dict version
metadata and restricted extra state. No optimizer, training checkpoint, or
Python model object is saved. Tensor subclasses, custom tensor attributes,
uninitialized parameters, meta tensors, and custom extra-state objects fail
explicitly.

Recipes come from the supplied root recipe or `mlzo.model.get_recipe(model)`.
The factory must have an importable `FactoryRef`. Arguments support `None`,
booleans, integers, finite floats, strings, lists, tuples, and dictionaries with
string keys. Tagged container encoding preserves tuples and dictionary order.
A recipe that exists but cannot be represented or imported fails explicitly;
it is never silently dropped. An eager model with no recipe is labelled
`weights_only`, and loading it requires `factory=`. Export never constructs an
unresolved `Lazy`. `block="encoder.layers.0"` selects registered submodules and
uses only that block's recipe, even when the caller supplies a root recipe.

`load_artifact(reference, device="cpu", factory=None)` accepts an `ArtifactRef`,
its dictionary representation, or a local directory path. It validates the
manifest, verifies a private copy of the weights, and loads that exact copy with
`torch.load(weights_only=True)`. It requires exact state_dict keys and uses
`load_state_dict(strict=True)`. The result is a materialized model in eval mode.
`factory=` overrides a stored recipe and must be a zero-argument callable
returning a matching `torch.nn.Module` or `Lazy`. Stored recipe factories are
trusted Python code and run on load; checksum validation does not authenticate
an artifact or sandbox its constructor. Training checkpoint directories are
not inference artifacts.

`deploy(reference, device="cpu", factory=None)` returns a local `Deployment`.
Its `predict(*args, **kwargs)` moves tensors in dictionaries, lists, tuples,
and namedtuples to the configured device, then calls the model in eval mode
under `torch.inference_mode()`. Other input values pass through. The model
validates input shapes and types. Outputs retain their ordinary types and
device. `status()` reports local model readiness and artifact identity, and
`shutdown()` is idempotent. Prediction and context entry after shutdown raise
`MLZOError`. The deployment also supports `with deploy(...) as local:`.
`on=` with any non-None target raises `UnsupportedFeatureError`; there is no
inference server or remote deployment in this alpha. Validation, corruption,
and reconstruction errors use `ConfigurationError`.
