---
title: Set up a remote worker
description: Start the coordinator, create credentials, and enroll one CPU worker on loopback.
---

This walkthrough runs a coordinator and worker on the same machine. It exercises the remote execution path without cloud resources. A job executes trusted project code on one worker and one device. The worker supervisor requires POSIX process groups and file locking.

Install MLZO and PyTorch in the environment that runs the worker. Use separate persistent coordinator and worker directories.

## Start the coordinator

In terminal one, from the repository root:

```sh
mlzo coordinator serve --directory .mlzo/coordinator --host 127.0.0.1 --port 8787
```

The foreground service persists SQLite state and credentials. It creates private `admin.token` and `submitter.token` files without printing their contents. Keep the directory across restarts.

Existing token files take precedence. If absent, `MLZO_TOKEN` and `MLZO_SUBMITTER_TOKEN` can supply their respective credentials; otherwise the service generates them. `--token-file` and `--submitter-token-file` choose other paths.

## Configure a client and create an invitation

In terminal two:

```sh
mlzo cluster create demo --endpoint http://127.0.0.1:8787 --token-file .mlzo/coordinator/admin.token
mlzo cluster invite demo --token-file .mlzo/worker-invite.token
```

`cluster create` saves a client configuration. It does not start a coordinator or provision a machine. The admin client can invite and manage workers. Use the submitter credential for a separate training client when administration is unnecessary.

An invitation expires after 900 seconds and permits one join by default. Its output path must be new. Keep the file private. `--show-token` explicitly prints a secret invitation; avoid it for routine setup. Public invitation dictionaries and representations omit the secret.

## Enroll and serve

```sh
mlzo worker join --token-file .mlzo/worker-invite.token --directory .mlzo/worker --device cpu
mlzo worker serve --directory .mlzo/worker
```

`worker join --serve` combines the last two steps. The worker persists its scoped identity and job directories; reuse its directory for serve, status, and leave. Do not run two services with one identity. Invitation input can also come from `MLZO_INVITE_TOKEN`; no positional secret argument is accepted.

In a third terminal, inspect membership:

```sh
mlzo cluster status demo
mlzo cluster workers demo
```

Named cluster configurations default to `~/.mlzo/clusters`; the default worker root is `~/.mlzo/worker`. `MLZO_HOME` or command-specific `--home` changes that root. An explicit worker `--directory` chooses its identity location directly.

## Across machines

HTTP is restricted to loopback. A nonloopback coordinator requires `--cert-file` and `--key-file` and a reachable HTTPS endpoint. `--endpoint` supplies the advertised origin if it differs from the bind address. For a private CA, configure an explicitly trusted `--ca-file` on every client and worker. Invitation CA metadata does not establish trust.

TLS and GPU execution paths remain unvalidated in this alpha. Loopback tests do not demonstrate connectivity, certificate configuration, or training across hosts. Jobs execute trusted single-owner code and are not sandboxed against hostile submissions. MLZO does not configure a VPC, firewall, IAM role, or cloud machine.

Continue with [submitting a project](../submission/).
