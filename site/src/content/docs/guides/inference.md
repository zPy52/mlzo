---
title: Export and inference
description: Export immutable model artifacts and run synchronous local predictions.
---

## Export after training

```python
artifact = run.export(to="outputs/classifier-v1")
model = mlzo.load_artifact(artifact, device="cpu")
```

The destination is the artifact directory itself. Export stores a restricted CPU copy of `state_dict()` in `weights.pt` and a version-one manifest with identity, version metadata, checksum, size, and reconstruction information. It does not save optimizer state or a Python model object.

Use a fresh destination for changed weights. An identical export can reuse its directory, but different or invalid existing contents raise an error. Writers use an exclusive sibling lock, a staging directory, and atomic rename. The destination filesystem must support these operations. Export requires a completed training boundary.

## Reconstruct the model

A `Lazy` root with an importable factory and supported primitive arguments can store its construction recipe. `nn.Linear` from the quickstart is one such recipe. An eager model with no recipe produces a `weights_only` artifact and needs a compatible factory:

```python
from torch import nn

model = mlzo.load_artifact("outputs/eager-classifier", device="cpu",
                           factory=lambda: nn.Linear(4, 2))
```

This example assumes an existing artifact for that architecture. `factory=` is a zero-argument callable and can override a stored recipe. The loaded model uses strict state-dictionary key matching and enters evaluation mode. Checksum verification detects corruption but does not authenticate a source or sandbox the factory's Python code.

Unresolved lazy models, uninitialized or meta tensors, custom tensor attributes/subclasses, and arbitrary extra-state objects fail explicitly. A recipe that exists but cannot be represented is rejected rather than silently omitted.

## Make predictions

```python
with mlzo.deploy(artifact, device="cpu") as predictor:
    logits = predictor.predict(inputs[:2])
    print(predictor.status())
```

`deploy()` creates a local synchronous `Deployment`. It unwraps MLZO `Tensor`
and `NamedTensor` leaves in dictionaries, lists, tuples, and namedtuples, without
reshaping them. It moves native tensors to the configured device and calls the
model under `torch.inference_mode()`. Other values pass through. Outputs retain
their ordinary types and device; the model validates shapes and dtypes.

For example, `predictor.predict(mlzo.wrap(native_inputs))` preserves the input
shape. `mlzo.tensor(vector)` constructs a column by default; use
`mlzo.tensor(vector, normalize=False)` when the model expects a vector.
Direct calls to the model returned by `load_artifact` need `.raw` explicitly,
such as `model(wrapped_inputs.raw)`. Native modules do not dispatch transparently
through the wrapper. See [tensor boundaries](../transforms/).

`shutdown()` releases the model and is idempotent. The context manager calls it on exit. Further predictions or context entry raise `MLZOError`. A non-`None` `on=` target raises `UnsupportedFeatureError`; this release has no inference server or remote inference deployment.

## Export one block

`run.export(to=..., block="encoder.layers.0")` selects a registered submodule. The block needs its own recipe for automatic reconstruction; the root recipe does not describe a different submodule. You can instead load a weights-only block with an explicit compatible factory.

`mlzo.export_model(model, to=..., recipe=None, block=None)` supports direct model export outside `Run`. Use it at a completed update boundary. See the [inference reference](../../reference/api/inference/).
