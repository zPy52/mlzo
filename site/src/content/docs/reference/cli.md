---
title: Command-line interface
description: Exact MLZO command help generated from the implemented argument parser.
---

`mlzo` and `python -m mlzo` are equivalent. `--json` is accepted anywhere and wraps successes as `{"ok": true, "result": ...}` and errors as `{"ok": false, "error": ...}`. Services emit a startup record before blocking. Errors return a nonzero exit code and omit sensitive details.

Use [remote setup](../../remote/setup/) for a complete command sequence. Artifact paths returned by remote checkpoint/export commands belong to the assigned worker. `cluster scale` and `run resume` exist to return explicit unsupported errors.

Token input uses protected files or environment variables. Invitation output needs a new `--token-file` path unless you explicitly choose `--show-token`.

## mlzo

```text
usage: mlzo [-h] [--version] [--json]
            {cluster,coordinator,worker,run,checkpoint} ...

MLZO Python alpha: local training and one-worker remote execution.

positional arguments:
  {cluster,coordinator,worker,run,checkpoint}
    cluster             configure or administer a coordinator
    coordinator         run a persistent coordinator
    worker              enroll or serve with a persistent worker identity
    run                 inspect a remote run through a saved cluster
    checkpoint          inspect local checkpoint storage

options:
  -h, --help            show this help message and exit
  --version             print the package version
  --json                emit JSON success/errors; accepted anywhere in the
                        command
```

### mlzo cluster

```text
usage: mlzo cluster [-h]
                    {create,status,workers,invite,drain,remove,shutdown,scale}
                    ...

positional arguments:
  {create,status,workers,invite,drain,remove,shutdown,scale}
    create              save a named client configuration; does not provision
                        a machine
    status              read coordinator status
    workers             list enrolled workers
    invite              export a short-lived worker enrollment invitation
    drain               stop assigning work to a worker
    remove              revoke a worker
    shutdown            drain and stop worker services; keep the coordinator
                        running
    scale               request scaling; unsupported in this alpha

options:
  -h, --help            show this help message and exit
```

#### mlzo cluster create

```text
usage: mlzo cluster create [-h] [--home HOME] --endpoint ENDPOINT
                           [--token-file TOKEN_FILE] [--ca-file CA_FILE]
                           name

save a named client configuration; does not provision a machine

positional arguments:
  name                  saved cluster name

options:
  -h, --help            show this help message and exit
  --home HOME           configuration root; default MLZO_HOME or ~/.mlzo
  --endpoint ENDPOINT   verified HTTPS origin, or loopback HTTP
  --token-file TOKEN_FILE
                        admin or submitter token file; otherwise MLZO_TOKEN
  --ca-file CA_FILE     trusted private CA certificate
```

#### mlzo cluster status

```text
usage: mlzo cluster status [-h] [--home HOME] name

read coordinator status

positional arguments:
  name         saved cluster name

options:
  -h, --help   show this help message and exit
  --home HOME  configuration root; default MLZO_HOME or ~/.mlzo
```

#### mlzo cluster workers

```text
usage: mlzo cluster workers [-h] [--home HOME] name

list enrolled workers

positional arguments:
  name         saved cluster name

options:
  -h, --help   show this help message and exit
  --home HOME  configuration root; default MLZO_HOME or ~/.mlzo
```

#### mlzo cluster invite

```text
usage: mlzo cluster invite [-h] [--home HOME]
                           (--token-file TOKEN_FILE | --show-token)
                           [--expires-in EXPIRES_IN] [--max-joins MAX_JOINS]
                           name

export a short-lived worker enrollment invitation

positional arguments:
  name                  saved cluster name

options:
  -h, --help            show this help message and exit
  --home HOME           configuration root; default MLZO_HOME or ~/.mlzo
  --token-file TOKEN_FILE
                        create a new mode-0600 invitation file
  --show-token          explicitly print the secret invitation
  --expires-in EXPIRES_IN
                        invitation lifetime in seconds
  --max-joins MAX_JOINS
```

#### mlzo cluster drain

```text
usage: mlzo cluster drain [-h] [--home HOME] name worker_id

stop assigning work to a worker

positional arguments:
  name         saved cluster name
  worker_id

options:
  -h, --help   show this help message and exit
  --home HOME  configuration root; default MLZO_HOME or ~/.mlzo
```

#### mlzo cluster remove

```text
usage: mlzo cluster remove [-h] [--home HOME] name worker_id

revoke a worker

positional arguments:
  name         saved cluster name
  worker_id

options:
  -h, --help   show this help message and exit
  --home HOME  configuration root; default MLZO_HOME or ~/.mlzo
```

#### mlzo cluster shutdown

```text
usage: mlzo cluster shutdown [-h] [--home HOME] name

drain and stop worker services; keep the coordinator running

positional arguments:
  name         saved cluster name

options:
  -h, --help   show this help message and exit
  --home HOME  configuration root; default MLZO_HOME or ~/.mlzo
```

#### mlzo cluster scale

```text
usage: mlzo cluster scale [-h] [--home HOME] name count

request scaling; unsupported in this alpha

positional arguments:
  name         saved cluster name
  count

options:
  -h, --help   show this help message and exit
  --home HOME  configuration root; default MLZO_HOME or ~/.mlzo
```

### mlzo coordinator

```text
usage: mlzo coordinator [-h] {serve} ...

positional arguments:
  {serve}
    serve     serve until interrupted

options:
  -h, --help  show this help message and exit
```

#### mlzo coordinator serve

```text
usage: mlzo coordinator serve [-h] --directory DIRECTORY [--host HOST]
                              [--port PORT] [--endpoint ENDPOINT]
                              [--cert-file CERT_FILE] [--key-file KEY_FILE]
                              [--ca-file CA_FILE] [--token-file TOKEN_FILE]
                              [--submitter-token-file SUBMITTER_TOKEN_FILE]
                              [--max-runtime MAX_RUNTIME]

options:
  -h, --help            show this help message and exit
  --directory DIRECTORY
                        private persistent coordinator state
  --host HOST
  --port PORT
  --endpoint ENDPOINT   advertised origin when different from the bind address
  --cert-file CERT_FILE
                        server TLS certificate
  --key-file KEY_FILE   server TLS private key
  --ca-file CA_FILE     CA path advertised to clients as informational
                        metadata
  --token-file TOKEN_FILE
                        read/create admin token file; default
                        DIRECTORY/admin.token
  --submitter-token-file SUBMITTER_TOKEN_FILE
                        read/create submitter token file; default
                        DIRECTORY/submitter.token
  --max-runtime MAX_RUNTIME
                        maximum job runtime in seconds
```

### mlzo worker

```text
usage: mlzo worker [-h] {join,serve,status,leave} ...

positional arguments:
  {join,serve,status,leave}

options:
  -h, --help            show this help message and exit
```

#### mlzo worker join

```text
usage: mlzo worker join [-h] [--directory DIRECTORY] [--home HOME]
                        [--ca-file CA_FILE] [--token-file TOKEN_FILE]
                        [--device DEVICE] [--serve]
                        [--poll-interval POLL_INTERVAL]

options:
  -h, --help            show this help message and exit
  --directory DIRECTORY
                        persistent identity directory; default
                        MLZO_HOME/worker or ~/.mlzo/worker
  --home HOME           configuration root; default MLZO_HOME or ~/.mlzo
  --ca-file CA_FILE     explicit trusted private CA certificate
  --token-file TOKEN_FILE
                        invitation file; otherwise MLZO_INVITE_TOKEN
  --device DEVICE       worker device, for example cpu or cuda:0
  --serve               serve immediately after enrollment
  --poll-interval POLL_INTERVAL
```

#### mlzo worker serve

```text
usage: mlzo worker serve [-h] [--directory DIRECTORY] [--home HOME]
                         [--ca-file CA_FILE] [--poll-interval POLL_INTERVAL]

options:
  -h, --help            show this help message and exit
  --directory DIRECTORY
                        persistent identity directory; default
                        MLZO_HOME/worker or ~/.mlzo/worker
  --home HOME           configuration root; default MLZO_HOME or ~/.mlzo
  --ca-file CA_FILE     explicit trusted private CA certificate
  --poll-interval POLL_INTERVAL
```

#### mlzo worker status

```text
usage: mlzo worker status [-h] [--directory DIRECTORY] [--home HOME]
                          [--ca-file CA_FILE]

options:
  -h, --help            show this help message and exit
  --directory DIRECTORY
                        persistent identity directory; default
                        MLZO_HOME/worker or ~/.mlzo/worker
  --home HOME           configuration root; default MLZO_HOME or ~/.mlzo
  --ca-file CA_FILE     explicit trusted private CA certificate
```

#### mlzo worker leave

```text
usage: mlzo worker leave [-h] [--directory DIRECTORY] [--home HOME]
                         [--ca-file CA_FILE] [--graceful | --no-graceful]

options:
  -h, --help            show this help message and exit
  --directory DIRECTORY
                        persistent identity directory; default
                        MLZO_HOME/worker or ~/.mlzo/worker
  --home HOME           configuration root; default MLZO_HOME or ~/.mlzo
  --ca-file CA_FILE     explicit trusted private CA certificate
  --graceful, --no-graceful
                        finish current work before leaving; --no-graceful
                        requests cancellation
```

### mlzo run

```text
usage: mlzo run [-h] {status,events,cancel,checkpoint,export,resume} ...

positional arguments:
  {status,events,cancel,checkpoint,export,resume}

options:
  -h, --help            show this help message and exit
```

#### mlzo run status

```text
usage: mlzo run status [-h] --cluster CLUSTER [--home HOME] run_id

positional arguments:
  run_id

options:
  -h, --help         show this help message and exit
  --cluster CLUSTER  saved cluster name
  --home HOME        configuration root; default MLZO_HOME or ~/.mlzo
```

#### mlzo run events

```text
usage: mlzo run events [-h] --cluster CLUSTER [--home HOME] [--after AFTER]
                       run_id

positional arguments:
  run_id

options:
  -h, --help         show this help message and exit
  --cluster CLUSTER  saved cluster name
  --home HOME        configuration root; default MLZO_HOME or ~/.mlzo
  --after AFTER      event sequence cursor
```

#### mlzo run cancel

```text
usage: mlzo run cancel [-h] --cluster CLUSTER [--home HOME] run_id

positional arguments:
  run_id

options:
  -h, --help         show this help message and exit
  --cluster CLUSTER  saved cluster name
  --home HOME        configuration root; default MLZO_HOME or ~/.mlzo
```

#### mlzo run checkpoint

```text
usage: mlzo run checkpoint [-h] --cluster CLUSTER [--home HOME] [--to TO]
                           run_id

positional arguments:
  run_id

options:
  -h, --help         show this help message and exit
  --cluster CLUSTER  saved cluster name
  --home HOME        configuration root; default MLZO_HOME or ~/.mlzo
  --to TO            destination on the assigned worker filesystem
```

#### mlzo run export

```text
usage: mlzo run export [-h] --cluster CLUSTER [--home HOME] --to TO
                       [--block BLOCK]
                       run_id

positional arguments:
  run_id

options:
  -h, --help         show this help message and exit
  --cluster CLUSTER  saved cluster name
  --home HOME        configuration root; default MLZO_HOME or ~/.mlzo
  --to TO            destination on the assigned worker filesystem
  --block BLOCK      registered model submodule path
```

#### mlzo run resume

```text
usage: mlzo run resume [-h] --cluster CLUSTER [--home HOME] run_id

positional arguments:
  run_id

options:
  -h, --help         show this help message and exit
  --cluster CLUSTER  saved cluster name
  --home HOME        configuration root; default MLZO_HOME or ~/.mlzo
```

### mlzo checkpoint

```text
usage: mlzo checkpoint [-h] {list} ...

positional arguments:
  {list}
    list      list committed snapshots at a local root

options:
  -h, --help  show this help message and exit
```

#### mlzo checkpoint list

```text
usage: mlzo checkpoint list [-h] location

positional arguments:
  location

options:
  -h, --help  show this help message and exit
```
