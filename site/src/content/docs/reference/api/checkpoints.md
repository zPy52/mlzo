---
title: Checkpoints
description: Public MLZO 0.1.0a1 API signatures and behavior for checkpoints.
---

Read the [checkpoints guide](../../../guides/checkpoints/) for examples and usage constraints. Signatures and declared fields below come from MLZO `0.1.0a1` source. Type names follow their defining Python modules.

## checkpoint

```python
mlzo.checkpoint(
    to: str | os.PathLike[str],
    every: int = 1000,
    keep_last: int | None = 3,
    mode: str = 'full',
) -> CheckpointPolicy
```

Create a validated checkpoint policy without allocating model state.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/checkpoint/policy.py#L56).

## CheckpointPolicy

Declared fields:

```python
to: str | os.PathLike[str]
every: int = 1000
keep_last: int | None = 3
mode: str = 'full'
```

Configure full, single-worker snapshots at an execution-engine interval.

`keep_last=None` disables retention. Retention applies to the whole local
destination, so use a separate destination for each run. The execution
engine counts successful optimizer update events for `every` and decides
which boundaries are safe for a complete snapshot; this storage module
does not schedule training work.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/checkpoint/policy.py#L16).

### CheckpointPolicy.to_dict

```python
CheckpointPolicy.to_dict() -> dict[str, Any]
```

Returns a dictionary representation. Public invitation and cluster representations omit credentials.

### CheckpointPolicy.from_dict

```python
CheckpointPolicy.from_dict(value: Mapping[str, Any]) -> CheckpointPolicy
```

Reconstructs this value from its dictionary representation, applying its validation rules.

## CheckpointRef

Declared fields:

```python
path: str | os.PathLike[str]
snapshot_id: str
run_id: str | None = None
created_at: str | None = None
metadata: dict[str, Any] = field(default_factory=dict)
```

A committed snapshot directory and its primitive-only catalog data.

`path` names a snapshot directory, not its destination root. Metadata in
the reference is informational; loading reads and validates the manifest.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/checkpoint/manifest.py#L24).

### CheckpointRef.to_dict

```python
CheckpointRef.to_dict() -> dict[str, Any]
```

Returns a dictionary representation. Public invitation and cluster representations omit credentials.

### CheckpointRef.from_dict

```python
CheckpointRef.from_dict(value: Mapping[str, Any]) -> CheckpointRef
```

Reconstructs this value from its dictionary representation, applying its validation rules.

## save_snapshot

```python
mlzo.save_snapshot(
    state: dict[Any, Any],
    *,
    to: str | os.PathLike[str],
    run_id: str,
    metadata: dict[str, Any] | None = None,
    keep_last: int | None = None,
) -> CheckpointRef
```

Save one complete full snapshot at a caller-selected safe boundary.

State consists of tensors, None/bool/int/float/str, dict/OrderedDict, list,
and tuple values. Integer dictionary keys and Python RNG tuples survive
unchanged. Tensors are detached and copied to CPU; loading returns CPU
state. OrderedDict's PyTorch `_metadata` is preserved. Custom objects,
NumPy arrays, tensor attributes, and cyclic containers must be converted
explicitly. Manifest metadata must be JSON data with string keys.

This alpha supports local single-worker generation-zero snapshots only.
Use one writer and a separate destination per run. Committed directories
are never rewritten. Publication fsyncs state first, publishes the manifest
atomically last, then atomically replaces latest.json. Retention runs only
after successful latest publication and always keeps this new snapshot.
A failed publication may leave an unadvertised committed directory, and a
process crash may leave a lock requiring manual cleanup. There is no reader
pinning against concurrent retention and no distributed writer fencing.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/checkpoint/snapshot.py#L113).

## load_snapshot

```python
mlzo.load_snapshot(reference: Reference) -> dict[Any, Any]
```

Verify format, exact size, and SHA256 before restricted CPU loading.

Accept a CheckpointRef, its dictionary form, a committed snapshot directory,
manifest.json, a destination root, or its latest.json pointer. A bad latest
pointer or state file raises; loading never selects an older snapshot.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/checkpoint/snapshot.py#L229).

## list_snapshots

```python
mlzo.list_snapshots(to: str | os.PathLike[str]) -> list[CheckpointRef]
```

List committed snapshots newest first, without importing torch.

Validates manifests and state sizes, skips incomplete directories, and
raises on corrupt committed entries. A missing destination returns [].
Listing does not use or repair latest.json and does not hash all state files.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/checkpoint/snapshot.py#L274).
