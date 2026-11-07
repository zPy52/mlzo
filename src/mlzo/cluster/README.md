# Single-worker remote alpha

The coordinator accepts JSON recipes and immutable source archives. An enrolled
worker launches the normal `Run.fit` in a job subprocess. The coordinator never
receives model tensors. This service executes trusted project code under the
worker user's account. It is not a sandbox.

## Python and CLI integration

All services below are available from `mlzo.cluster`. The package also exposes
the `worker` module, whose `join` function delegates to `Worker.join`.

- `cluster(name, *, endpoint=None, token=None, ca_file=None, home=None) -> Cluster`
  creates or loads `home/clusters/<name>.json`. Home defaults to `MLZO_HOME`,
  then `~/.mlzo`. Directories use mode 0700 and JSON files mode 0600. Creation
  starts no server. An unconfigured name is allowed, but service calls require
  an endpoint and credential. `Cluster.to_dict()` and repr omit credentials.
- `Cluster.status() -> dict`, `workers() -> list[dict]`,
  `invite(*, expires_in=900, max_joins=1) -> Invitation`,
  `drain(worker_id) -> dict`, `remove(worker_id) -> dict`, `shutdown() -> dict`.
  `scale(count, **kwargs)` raises `UnsupportedFeatureError` for managed capacity,
  including AWS. The planned `cluster.aws(...)` spelling and
  `mlzo.providers.aws(...)` also raise this error without contacting AWS.
- `Invitation.encode() -> str` explicitly exposes the transferable capability.
  Use that value for a CLI invitation output. `repr` and `to_dict()` redact its
  secret. `Invitation.decode(value)` accepts an encoded string or Invitation.
- `Coordinator(directory, *, host='127.0.0.1', port=0, token=None,
  submitter_token=None, cert_file=None, key_file=None, ca_file=None,
  endpoint=None, max_runtime=3600)`. `serve()` blocks. `start()` and context
  manager entry start a background server. `shutdown()` or `stop()` stops the
  server. `endpoint`, `token`, and `submitter_token` are explicit properties for
  configuring clients. Generated credentials persist in the private directory.
  The admin credential can submit; the distinct submitter credential cannot
  enroll, drain, remove, or stop workers. Worker credentials only access worker
  operations and archives assigned to that worker.
- `Worker.join(invite, *, directory=None, home=None, device=None,
  ca_file=None) -> Worker` enrolls and saves a scoped identity. Its default
  directory is `<MLZO_HOME>/worker`. `Worker(directory=None, *, home=None,
  ca_file=None)` reloads that identity. Use separate directories for separate
  identities. Inventory is gathered at join/serve, never import. A selected
  `device='cpu'` does not query GPU availability.
- `Worker.serve(*, poll_interval=0.5)` blocks. `start(...) -> Worker` is optional
  background mode. `status()` and `to_dict()` return local, redacted status.
  `stop(*, graceful=True)` requests local termination. `leave(graceful=True)`
  also asks the coordinator to stop this identity. Graceful stop finishes its
  active job. Forced stop requests safe cancellation and escalates to signals.
- `Cluster.submit(run, *, epochs=None, steps=None, parallel='auto', code=None,
  resume=None, request_id=None, max_runtime=3600) -> str` accepts exactly one
  positive integer budget. The coordinator caps runtime at its own configured
  limit. `code` is a project directory. Without it, submission searches upward
  from cwd for `pyproject.toml`; ambiguity raises a configuration error.
- `Cluster.get_run(id) -> dict`, `events(id, after=0) -> list[dict]`,
  `wait(id, *, timeout=None, poll_interval=0.5) -> dict`, and `cancel(id) -> dict`.
  The terminal dict contains `state`, `history`, and `counters` for `Run.wait`.
  `run(id, *, on=cluster) -> Run` reconnects using the existing training handle.
- `Cluster.checkpoint_run(id, to=None) -> CheckpointRef` returns the completed
  internal snapshot metadata. With `to`, it queues a worker-side snapshot copy.
  `export_run(id, to, block=None) -> ArtifactRef` reconstructs the completed
  model in its job venv, restores restricted checkpoint weights, and exports
  there. These manual operations reject active and failed jobs. A configured
  checkpoint policy still writes during normal training.
  `command_status(command_id) -> dict` inspects an operation that outlived the
  default 120-second client wait. Its pending-command error includes the ID.

```python
from mlzo.cluster import Coordinator, Worker, cluster

with Coordinator('/private/coordinator-state') as service:
    target = cluster('development', endpoint=service.endpoint,
                     token=service.token, home='/private/client-state')
    worker = Worker.join(target.invite(), directory='/private/worker-state',
                         device='cpu').start()
    try:
        # Existing Run with a root Lazy and importable data/model factories.
        learning.fit(steps=1, on=target, code='/path/to/project', wait=False)
        history = learning.wait()
        reference = learning.export(to='exports/final')
    finally:
        worker.leave(graceful=True)
```

Keep the worker service running for later checkpoint copies and exports. Relative
remote output paths resolve in the worker's extracted `jobs/<run-id>/project`
directory. Absolute paths also belong to the worker. Returning a reference does
not download weights to the submitting laptop. The current API has no download,
remote resume-submission, remote evaluation, or remote inference service.

## Transport and execution

Nonloopback servers require a TLS certificate and key. Supply a reachable
advertised `endpoint` when binding a wildcard interface or using a public DNS
name. Clients use normal certificate/hostname verification. A private CA needs
an explicit readable `ca_file` on each client/worker machine. The optional
invitation CA path is only an informational trust reference; it never disables
verification. Unencrypted HTTP is restricted to loopback development. Redirects
and environment proxy configuration are disabled to keep bearer credentials on
the configured origin.

Requests and replies have socket deadlines and size limits. JSON is at most
9 MiB, archives at most 100 MiB compressed and expanded, events 16 KiB each,
and results 512 KiB. At most 16 concurrent coordinator handlers run. Histories
and event spools retain bounded scalar records; oversized records are trimmed.
Event sequence gaps indicate dropped records. Metrics are reporting summaries,
not a complete per-iteration audit log. Native process/pip logs are suppressed;
preparation failures report the phase and exit code without environment dumps.

`serialize_run`, declared data discovery and remote planning happen before
acceptance. Submission stages declared data using the artifact's `data_mapping`,
then serializes the remapped recipe. The archive digest is verified before
worker extraction. Workers cache uploaded archives by digest. Specs, archive
identity, assignment, generation zero, events, results and membership persist in
SQLite. Matching run/request retries return the accepted ID; changed recipe,
archive or budget conflicts. Accepted plans and placement remain pinned on retry.

Each job gets a private venv and sanitized environment, including a separate
HOME and MLZO_HOME. Its only inherited import paths point to the worker's MLZO
and PyTorch runtime site-packages. This policy also makes other packages in that
worker runtime site-packages visible. Declared dependencies install into the
job venv with pip. Packaged projects install there with dependency resolution
enabled, so build-generated requirements also apply. Simple importable `.py` projects need neither dependencies nor
packaging metadata. Pip/build commands use argument arrays. They never change
the supervisor environment. Native builds and index/network availability are
worker responsibilities; there is no cross-platform environment copying or
lockfile solver. Runtime/preparation metadata persist with each job.

The supervisor retains a private process group and durable launch intent before
starting user code. Coordinator outages do not stop training. Events and the
terminal result spool to the worker filesystem for retry. A restarted supervisor
reconciles its existing process identity; ambiguous interrupted launches fail
rather than rerunning the job. A service lock prevents two supervisors sharing
one identity directory. The alpha supervisor requires POSIX. Job processes also
have a finite runtime alarm. Cancellation checks safe iteration boundaries and
escalates to process-group termination after a grace period.

Drain stops new claims and permits active work to finish. Cluster shutdown asks
all enrolled services to drain and exit; it never powers off hosts or deletes
resources. Remove revokes membership and marks unfinished assigned work failed,
while warning that a disconnected process may continue until it observes
revocation. A stale heartbeat reports a disconnected worker while retaining its
job ownership. There is no automatic reassignment, retry, generation recovery,
leader election, multiworker collective execution, or managed cloud provisioning.
