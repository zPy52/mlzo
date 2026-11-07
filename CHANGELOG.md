# Changelog

## Unreleased

- `tensor()` now returns `Tensor` by default and `NamedTensor` with axes.
  Scalars and vectors normalize to matrices, reductions keep dimensions, and
  basic integer indexing keeps singleton dimensions. `normalize=False`,
  `wrap()`, and `.raw` provide explicit shape and native-tensor boundaries.
  See [migration examples](src/mlzo/transforms/README.md#migrate-from-the-previous-native-default).
- Declarative paired columns, `get`/`length`, and `stream` data sources.
  Sample transforms, lazy filters, configurable worker batching, custom combine
  functions, and native DataLoader escape hatches.
- Training and local prediction unwrap MLZO tensor leaves without reshaping.
  The supervised cross-entropy adapter accepts integral `(B, 1)` class targets
  for `(B, C)` logits; custom losses keep their target shapes.

## 0.1.0a1

First Python alpha, distributed under the MIT license and prepared for later publication.

- Deferred model construction, immutable data pipelines, single-device planning, and PyTorch function transforms.
- Supervised training and custom `Training` algorithms with explicit named updates, scalar history, evaluation, and local full-state checkpoints.
- Authenticated coordinator and POSIX worker services for trusted, single-worker remote jobs, source snapshots, durable run status, and reconnect handles.
- Immutable local inference artifacts and synchronous local prediction.
- A basic DQN recipe with bounded replay and partial environment resume.
- An argparse CLI, typed package marker, CPU examples, and publishing instructions.

Validation is intentionally limited to basic compilation, import, packaging, and tiny smoke checks. There is no comprehensive test suite, CI, GPU/TLS validation, or dependency compatibility matrix. See [alpha support](docs/alpha-support.md) for unsupported features and resume limits.
