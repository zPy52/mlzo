---
title: Clusters and workers
description: Public MLZO 0.1.0a1 API signatures and behavior for clusters and workers.
---

Read the [clusters and workers guide](../../../remote/setup/) for examples and usage constraints. Signatures and declared fields below come from MLZO `0.1.0a1` source. Type names follow their defining Python modules.

## cluster

```python
mlzo.cluster(
    name: str,
    *,
    endpoint: str | None = None,
    token: str | None = None,
    ca_file: str | None = None,
    home: str | os.PathLike | None = None,
) -> Cluster
```

Create/load a private named configuration. This never starts a network service.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/cluster/client.py#L202).

## Cluster

```python
mlzo.Cluster(
    name: str,
    *,
    endpoint: str | None = None,
    token: str | None = None,
    ca_file: str | None = None,
    home: str | os.PathLike | None = None,
)
```

A control client, without network or hardware work during construction.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/cluster/client.py#L51).

### Cluster.to_dict

```python
Cluster.to_dict() -> dict
```

Returns a dictionary representation. Public invitation and cluster representations omit credentials.

### Cluster.status

```python
Cluster.status() -> dict
```

Queries coordinator status using this client's credential.

### Cluster.workers

```python
Cluster.workers() -> list[dict]
```

Returns enrolled-worker membership and liveness reported by the coordinator.

### Cluster.invite

```python
Cluster.invite(*, expires_in: float=900, max_joins: int=1) -> Invitation
```

Creates an expiring worker enrollment invitation. Defaults to 900 seconds and one join; requires admin authority.

### Cluster.drain

```python
Cluster.drain(worker_id: str) -> dict
```

Stops new assignments to the named worker.

### Cluster.remove

```python
Cluster.remove(worker_id: str) -> dict
```

Revokes the named worker's membership.

### Cluster.shutdown

```python
Cluster.shutdown() -> dict
```

Drain enrolled services and ask them to stop after existing work finishes.

### Cluster.scale

```python
Cluster.scale(count: int, **kwargs: Any) -> None
```

Raises `UnsupportedFeatureError`. Managed provisioning and scaling are unavailable; enroll existing machines explicitly.

### Cluster.submit

```python
Cluster.submit(
    run: Any,
    *,
    epochs: int | None = None,
    steps: int | None = None,
    parallel: Any = 'auto',
    code: str | os.PathLike | None = None,
    resume: Any = None,
    request_id: str | None = None,
    max_runtime: float = 3600,
) -> str
```

Snapshots a trusted importable project and returns its durable run ID. One worker executes the run. Remote resume is rejected.

### Cluster.get_run

```python
Cluster.get_run(run_id: str) -> dict
```

Returns the durable coordinator record for this run ID.

### Cluster.events

```python
Cluster.events(run_id: str, after: int=0) -> list[dict]
```

Returns scalar events after the supplied sequence cursor.

### Cluster.wait

```python
Cluster.wait(run_id: str, *, timeout: float | None=None, poll_interval: float=0.5) -> dict
```

Polls until a terminal record or the optional timeout. Returns the record; use `Run.wait()` for History and failure exceptions.

### Cluster.cancel

```python
Cluster.cancel(run_id: str) -> dict
```

Requests cancellation of the run.

### Cluster.command_status

```python
Cluster.command_status(command_id: str) -> dict
```

Queries a previously submitted worker artifact command by command ID.

### Cluster.checkpoint_run

```python
Cluster.checkpoint_run(run_id: str, to: Any=None) -> Any
```

Returns a completed run's retained reference or requests a checkpoint at a worker-local destination. Does not download it.

### Cluster.export_run

```python
Cluster.export_run(run_id: str, to: Any, block: str | None=None) -> Any
```

Requests a completed run's export on its assigned worker. The service and job directory must remain available.

## Invitation

An enrollment capability. Only encode() exposes the transferable secret.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/cluster/client.py#L21).

### Invitation.to_dict

```python
Invitation.to_dict() -> dict
```

Returns a dictionary representation. Public invitation and cluster representations omit credentials.

### Invitation.encode

```python
Invitation.encode() -> str
```

Returns the transferable secret invitation envelope. Treat the result as a credential.

### Invitation.decode

```python
Invitation.decode(value: Any) -> 'Invitation'
```

Validates an encoded invitation envelope or accepts an existing invitation. It does not establish CA trust.

## run

```python
mlzo.run(run_id: str, *, on: Cluster) -> Any
```

Reconnect through the training module's normal remote Run delegation.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/cluster/client.py#L219).

## Coordinator

```python
mlzo.Coordinator(
    directory: str | os.PathLike,
    *,
    host: str = '127.0.0.1',
    port: int = 0,
    token: str | None = None,
    submitter_token: str | None = None,
    cert_file: str | None = None,
    key_file: str | None = None,
    ca_file: str | None = None,
    endpoint: str | None = None,
    max_runtime: float = 3600,
)
```

Explicit HTTPS service; HTTP is restricted to loopback development.

`start()` and context-manager entry start a background server. `serve()`
blocks for CLI use. `shutdown()` stops this server, while Cluster.shutdown
requests graceful worker-service termination and keeps the coordinator up.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/cluster/coordinator.py#L332).

### Coordinator.token

```python
Coordinator.token: str  # read-only property
```

Returns the admin secret. Do not print or store it in project source.

### Coordinator.submitter_token

```python
Coordinator.submitter_token: str  # read-only property
```

Returns the submitter secret. It is distinct from the admin credential.

### Coordinator.endpoint

```python
Coordinator.endpoint: str  # read-only property
```

Returns the advertised origin or the bound endpoint after startup.

### Coordinator.serve

```python
Coordinator.serve() -> None
```

Serves in the current thread until shutdown.

### Coordinator.start

```python
Coordinator.start() -> 'Coordinator'
```

Starts the coordinator in a background thread and returns it. Context-manager entry also starts it.

### Coordinator.shutdown

```python
Coordinator.shutdown() -> None
```

Stops this coordinator. `Cluster.shutdown()` instead requests worker service termination.

## Worker

```python
mlzo.Worker(
    directory: str | os.PathLike | None = None,
    *,
    home: str | os.PathLike | None = None,
    ca_file: str | None = None,
)
```

A single job at a time on one device. serve() is explicitly blocking.

[View source](https://github.com/zPy52/mlzo/blob/main/src/mlzo/cluster/worker.py#L66).

### Worker.join

```python
Worker.join(
    invite: Invitation | str,
    *,
    directory: str | os.PathLike | None = None,
    home: str | os.PathLike | None = None,
    device: str | None = None,
    ca_file: str | None = None,
) -> 'Worker'
```

Enrols using an Invitation or encoded invitation, persists scoped identity, and returns a worker. Provide explicit CA trust for private TLS.

### Worker.status

```python
Worker.status() -> dict
```

Reports persisted local metadata and this object's service state. Query `Cluster.workers()` for coordinator membership and liveness.

### Worker.stop

```python
Worker.stop(*, graceful: bool=True) -> None
```

Requests this worker object's service stop, optionally allowing current work to finish.

### Worker.leave

```python
Worker.leave(graceful: bool=True) -> dict
```

Leaves coordinator membership, gracefully by default, and returns the result.

### Worker.start

```python
Worker.start(*, poll_interval: float=0.5) -> 'Worker'
```

Starts this worker service in a background thread and returns the worker.

### Worker.serve

```python
Worker.serve(*, poll_interval: float=0.5) -> None
```

Runs the polling service until stopped. One job/device executes at a time and the directory is exclusively locked.

## worker module

`mlzo.worker` is the worker module. `mlzo.worker.join` delegates to `Worker.join` and returns a worker. The module also exposes the `Worker` class.

```python
mlzo.worker.join(invite: Invitation | str, **kwargs: Any) -> Worker
```
