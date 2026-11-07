---
title: Alpha support matrix
description: What MLZO 0.1.0a1 implements, what it rejects, and what its tests do not establish.
---

This site describes `0.1.0a1`. The repository's `plan.md` is a future design, not an implemented capability list. Unsupported requests should fail explicitly.

| Area | Implemented | Limits |
| --- | --- | --- |
| Training | One local device or one remote worker, supervised/custom steps, named updates | No DDP, FSDP, pipeline/tensor parallelism, or multi-worker run |
| Planning | Deferred factories, device/precision decisions, placement checks | No forward pass, memory probing, automatic partitioning |
| Data | Native local sources, factories, JSON/JSONL, restricted tensor files, images, optional S3 objects | No distributed sharding or exact arbitrary cursor replay |
| Checkpoints | Full local snapshots, integrity checks, atomic manifests, retention, compatible restore | No S3, sharded/per-block state, generation fencing, automatic recovery |
| Resume | Model/optimizer/scheduler/RNG/algorithm state and counters | Partial epochs restart; one-shot iterators and remote resume unsupported |
| Cluster | Authenticated coordinator, expiring invitations, scoped identities, source snapshots, events, reconnect/cancel | Trusted single-owner code; one job/device per worker; no provisioning or automatic worker-failure recovery |
| Inference | Immutable local artifacts, strict loading, local synchronous prediction | No remote inference server or autoscaling; weights-only artifacts need a factory |
| DQN | Bounded replay, target synchronization, discrete vector observations | Partial environment reset on resume; no vector environments, PPO, actor-critic, continuous actions, distributed rollouts |
| Transforms | Direct PyTorch aliases, explicit named tensor operations | NamedTensor is not a general torch dispatcher or transform pytree |

## Platform and validation

The package declares Python `>=3.12` and PyTorch `>=2.2.2,<3`. These ranges are not a tested compatibility matrix. Intel macOS uses a NumPy-below-2 marker for its legacy PyTorch wheel. Worker supervision requires POSIX process groups and file locking.

The current repository has a CPU test suite and GitHub Actions workflow. Tests cover deferred construction, numerical comparisons against raw PyTorch, weighted uneven accumulation, checkpoint integrity and epoch-boundary resume, failed-step cleanup, enrollment and credential scope, CLI output, and single-worker loopback execution with reconnect and subsequent jobs.

These checks use synthetic local data. They do not validate multi-host training or backpropagation, GPUs, mixed precision, OOM recovery, TLS deployments, IAM/VPC configuration, or production readiness. A plan's validation label is a metadata result and does not replace execution evidence.

## Operational scope

Remote execution runs trusted Python code and dependency installation. It is not a hostile-code sandbox or a multi-tenant service. Keep credentials and runtime state out of submitted projects and repositories. Checksums detect corruption; they do not authenticate an artifact's author or make its factory safe to execute.

For practical setup, read [remote execution](../../remote/setup/). For a failure, see [troubleshooting](../troubleshooting/).
