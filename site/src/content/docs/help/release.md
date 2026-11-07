---
title: Release 0.1.0a1
description: The first MLZO Python alpha, its implemented capabilities, and current verification scope.
---

MLZO `0.1.0a1` is the first Python alpha, distributed under the [MIT license](https://github.com/zPy52/mlzo/blob/main/LICENSE). Install it from source; the package has not been published to PyPI. This documentation does not change the package version or announce a beta release.

## Available in the alpha

- Deferred model construction, immutable ordered data pipelines, and single-device plans.
- Supervised training, custom `Training` algorithms, named updates, bounded scalar history, and local evaluation.
- Full-state local checkpoints with integrity verification, retention, and compatible epoch-restart resume.
- Authenticated coordinator and POSIX worker services for trusted single-worker jobs, source snapshots, durable status, and reconnect handles.
- Immutable inference artifacts and synchronous local prediction.
- A small discrete-action DQN recipe with replay/target state and partial environment resume.
- PyTorch function transform aliases, explicit named tensors, and a structured CLI.

## Current evidence

The repository now includes CPU behavioral tests and a package-building CI workflow. This expands the original release's tiny smoke checks. The tests compare supported training behavior with raw PyTorch, verify persistence and credential rules, and run the worker path using local subprocesses and loopback networking.

These results do not certify GPUs, TLS, multi-host operation, distributed backpropagation, cloud configuration, automatic recovery, or production readiness. See the [support matrix](../support/) for the full boundaries.

## Documentation scope

The guides and reference cover the implemented alpha. API signatures and CLI help come from source, and the site build indexes the documentation for local search. The historical `alpha-contract.md` and `plan.md` remain in the repository as design records; this site uses current implementation and support boundaries for user instructions.
