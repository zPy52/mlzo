---
title: Exceptions
description: Public MLZO 0.1.0a1 API signatures and behavior for exceptions.
---

Read the [exceptions guide](../../../help/troubleshooting/) for examples and usage constraints. Signatures and declared fields below come from MLZO `0.1.0a1` source. Type names follow their defining Python modules.

## MLZOError

Base error for an MLZO operation.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/errors.py#L4).

## ConfigurationError

A configuration cannot be executed as requested.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/errors.py#L8).

## UnsupportedFeatureError

A requested capability is outside the supported alpha subset.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/errors.py#L12).

## TrainingError

An accepted training run failed.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/errors.py#L16).

## CheckpointError

A checkpoint could not be saved, verified, or restored.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/errors.py#L20).

## ClusterError

A cluster operation failed.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/errors.py#L24).
