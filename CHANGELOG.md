# Changelog

## 0.1.0a1

First Python alpha, distributed under the MIT license and prepared for later publication.

- Deferred model construction, immutable data pipelines, single-device planning, and PyTorch function transforms.
- Supervised training and custom `Training` algorithms with explicit named updates, scalar history, evaluation, and local full-state checkpoints.
- Authenticated coordinator and POSIX worker services for trusted, single-worker remote jobs, source snapshots, durable run status, and reconnect handles.
- Immutable local inference artifacts and synchronous local prediction.
- A basic DQN recipe with bounded replay and partial environment resume.
- An argparse CLI, typed package marker, CPU examples, and publishing instructions.

Validation is intentionally limited to basic compilation, import, packaging, and tiny smoke checks. There is no comprehensive test suite, CI, GPU/TLS validation, or dependency compatibility matrix. See [alpha support](docs/alpha-support.md) for unsupported features and resume limits.
