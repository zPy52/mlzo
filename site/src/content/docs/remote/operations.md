---
title: Operate runs and workers
description: Inspect events, cancel jobs, retrieve worker-local artifact references, and stop services.
---

## Inspect a run

```sh
mlzo run status RUN_ID --cluster demo --json
mlzo run events RUN_ID --cluster demo --after 0
```

Replace `RUN_ID` with a submitted run ID. JSON results use `{"ok": true, "result": ...}`. Failures use `{"ok": false, "error": ...}` and a nonzero exit status. Remote error details are deliberately limited to avoid credential exposure.

The coordinator tracks submitted, assigned/preparing, running, and terminal job records. Use `run.status()` for state and counters and `run.wait()` for successful history. Events carry scalar reports, not live tensor graphs.

## Cancel

```sh
mlzo run cancel RUN_ID --cluster demo
```

Python callers can use `run.cancel()` or `cluster.cancel(run_id)`. Cancellation requests termination of the worker job; inspect the terminal record afterward. The worker supervisor bounds jobs and manages owned process groups. A worker failure does not provide automatic recovery or transparent rescheduling of training.

## Checkpoint and export

```sh
mlzo run checkpoint RUN_ID --cluster demo
mlzo run checkpoint RUN_ID --cluster demo --to /absolute/worker/path/checkpoints
mlzo run export RUN_ID --cluster demo --to /absolute/worker/path/classifier
```

These commands require a completed run. Without `--to`, checkpoint returns the retained reference. Explicit destinations and returned paths belong to the assigned worker. The commands do not copy files back to the client. Export requires the worker service and persisted job directory to remain available. Choose a fresh export destination for changed weights.

`mlzo run resume RUN_ID --cluster demo` returns an explicit unsupported error. Remote evaluation and remote inference deployment are also unavailable.

## Manage membership

```sh
mlzo cluster workers demo
mlzo cluster drain demo WORKER_ID
mlzo cluster remove demo WORKER_ID
```

Drain stops new assignments to a worker. Remove revokes its membership. Use the coordinator's worker listing for membership and liveness; `worker status` reads persisted local metadata and the state of its current Python object, so a new CLI object cannot inspect another process's in-memory service state.

```sh
mlzo worker status --directory .mlzo/worker
mlzo worker leave --directory .mlzo/worker
```

Leave is graceful by default. `--no-graceful` requests cancellation. Reuse the same persistent worker directory for these commands.

## Stop services

`mlzo cluster shutdown demo` drains enrolled services and requests their stop after current work. It leaves the coordinator running. Interrupt the coordinator's foreground command to stop that service. In Python, `Coordinator.shutdown()` stops the coordinator itself; it has a different purpose from `Cluster.shutdown()`.

Managed scaling and provisioning are unsupported. `cluster scale` and provider-shaped helpers reject the operation explicitly. See the [CLI reference](../../reference/cli/) for all options.
