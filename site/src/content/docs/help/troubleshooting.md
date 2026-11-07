---
title: Troubleshooting
description: Diagnose unsupported features, shape errors, failed runs, checkpoint incompatibility, and remote setup.
---

## The model is still unmaterialized

Calling an unresolved `Lazy` module raises `TrainingError`. Planning intentionally leaves it unresolved. Prepare it through `run.fit()` or `run.evaluate()`, or call `mlzo.materialize(model, "cpu")`. A failed construction retains the error; fix the factory and create a new recipe.

## A loss or batch is rejected

Built-in supervised training expects `(inputs, targets)`, not a field mapping or arbitrary object. Use a mapping transform or `step(model, batch)` for another structure. Return a scalar differentiable loss for automatic updates. Put only scalar values in metrics and give a finite nonnegative detached normalization weight.

For cross entropy, the usual class-index target is integer-valued and the model emits logits. Tensor shape and dtype rules follow PyTorch.

## A factory cannot be transported or exported

Move the factory to a named definition in an importable project module. Closures, lambdas, and `__main__` definitions cannot be reconstructed by a remote worker or stored recipe. Local training allows broader callables, but that does not make them exportable. For an eager model without a recipe, export weights and supply a compatible zero-argument `factory=` at load time.

## A run failed and will not continue

A failed step can follow already completed parameter updates. Ordinary `fit()` cannot pretend those updates never happened. Resume a committed checkpoint with compatible configuration or construct a fresh run. Pending-gradient cleanup does not restore the previous model weights.

Check `run.status().to_dict()` and remote events for the terminal state. CLI errors intentionally omit remote details that could reveal credentials.

## Resume rejects my checkpoint

Keep the model, algorithm, optimizer ownership, schedule definition/horizon, and precision compatible. A changed inferred schedule horizon can also invalidate continued training. Use a checkpoint root for `resume="latest"` only with a configured policy.

A training checkpoint cannot be loaded as an inference artifact. A one-shot data iterator cannot resume. A partially completed data epoch restarts rather than replaying an exact cursor. Do not edit manifests or bypass corruption checks to force a restore.

## An export directory already exists

Exports are immutable. Choose a new destination for changed weights or reconstruction information. After a crashed export, confirm that its writer has stopped before removing an owned stale `.mlzo-export.lock` or `.staging-*` directory. Do not remove another process's lock.

## A worker is absent or not receiving work

Use `mlzo cluster workers NAME` for coordinator membership and liveness. `worker status` in a fresh process cannot read another object's in-memory service state. Confirm that the worker service uses its enrolled persistent directory, is not drained, has a compatible device, and has the project dependencies available.

An expired or already-used invitation needs a fresh invitation file. Join persists a scoped identity; do not share one identity directory between services. `cluster create` saves configuration and does not start any service.

## HTTPS or device selection fails

HTTP is allowed only on loopback. Other hosts require HTTPS and a coordinator certificate/key pair. A private CA must be trusted explicitly on each client and worker; invitation metadata does not establish that trust. Logical GPU IDs belong to the target worker. Multiple GPU placements do not enable distributed training.

## An optional adapter is unavailable

Install `.[vision]` for image decoding, `.[s3]` for S3 object reads, or `.[rl]` for Gymnasium environment IDs. Optional dependencies load on use. Adding an extra does not enable a capability listed as unsupported, such as S3 checkpoints or vector DQN environments.

## Exception types

`MLZOError` is the base operational exception. `ConfigurationError` also subclasses `ValueError`. `UnsupportedFeatureError` subclasses `ConfigurationError` and `NotImplementedError`. Training, checkpoint, and cluster operations have `TrainingError`, `CheckpointError`, and `ClusterError` types. See the [exception reference](../../reference/api/errors/).
