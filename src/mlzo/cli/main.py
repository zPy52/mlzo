"""Small argparse interface to the same cluster and worker APIs used by Python."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from functools import partial
import json
import os
from pathlib import Path
import signal
import stat
import sys
import threading
from typing import Any

from mlzo._version import __version__
from mlzo.errors import ConfigurationError, MLZOError, UnsupportedFeatureError


class _UsageError(Exception):
    """Parser errors omit argument values, which might contain credentials."""


class _CommandError(Exception):
    """Only fixed, credential-free messages belong in this exception."""


def _emit(value: Any, *, json_output: bool = False) -> None:
    result = {"ok": True, "result": _public(value)} if json_output else _public(value)
    print(json.dumps(result, indent=None if json_output else 2, allow_nan=False), flush=True)


def _public(value: Any) -> Any:
    # Prefer public conversions before dataclass conversion: Invitation.to_dict()
    # deliberately leaves out its secret envelope field.
    if callable(getattr(value, "to_dict", None)):
        return _public(value.to_dict())
    if is_dataclass(value):
        return _public(asdict(value))
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            name = str(key).lower()
            if name in {"token", "secret", "authorization", "credential", "password"} or name.endswith("_token"):
                result[key] = "[redacted]"
            elif name == "error" and item is not None:
                result[key] = {"message": "Error details omitted to protect credentials."}
            else:
                result[key] = _public(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_public(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


class _Parser(argparse.ArgumentParser):
    def __init__(self, *args: Any, json_output: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.json_output = json_output

    def error(self, message: str) -> None:
        raise _UsageError("Invalid command arguments. Run this command with --help.")

    def print_help(self, file: Any = None) -> None:
        if self.json_output:
            _emit({"help": self.format_help()}, json_output=True)
        else:
            super().print_help(file)


def _parser(json_output: bool) -> _Parser:
    parser_type = partial(_Parser, json_output=json_output)
    parser = parser_type(prog="mlzo", description="MLZO Python alpha: local training and one-worker remote execution.")
    parser.add_argument("--version", action="store_true", help="print the package version")
    parser.add_argument("--json", action="store_true", help="emit JSON success/errors; accepted anywhere in the command")
    commands = parser.add_subparsers(dest="command", parser_class=parser_type)

    group = commands.add_parser("cluster", help="configure or administer a coordinator")
    actions = group.add_subparsers(dest="action", required=True, parser_class=parser_type)
    for name, help_text in {
        "create": "save a named client configuration; does not provision a machine",
        "status": "read coordinator status", "workers": "list enrolled workers",
        "invite": "export a short-lived worker enrollment invitation",
        "drain": "stop assigning work to a worker", "remove": "revoke a worker",
        "shutdown": "drain and stop worker services; keep the coordinator running",
        "scale": "request scaling; unsupported in this alpha",
    }.items():
        command = actions.add_parser(name, help=help_text, description=help_text)
        command.add_argument("name", help="saved cluster name")
        command.add_argument("--home", help="configuration root; default MLZO_HOME or ~/.mlzo")
        if name == "create":
            command.add_argument("--endpoint", required=True, help="verified HTTPS origin, or loopback HTTP")
            command.add_argument("--token-file", help="admin or submitter token file; otherwise MLZO_TOKEN")
            command.add_argument("--ca-file", help="trusted private CA certificate")
        elif name == "invite":
            output = command.add_mutually_exclusive_group(required=True)
            output.add_argument("--token-file", help="create a new mode-0600 invitation file")
            output.add_argument("--show-token", action="store_true", help="explicitly print the secret invitation")
            command.add_argument("--expires-in", type=float, default=900, help="invitation lifetime in seconds")
            command.add_argument("--max-joins", type=int, default=1)
        elif name in {"drain", "remove"}:
            command.add_argument("worker_id")
        elif name == "scale":
            command.add_argument("count", type=int)

    group = commands.add_parser("coordinator", help="run a persistent coordinator")
    actions = group.add_subparsers(dest="action", required=True, parser_class=parser_type)
    command = actions.add_parser("serve", help="serve until interrupted")
    command.add_argument("--directory", required=True, help="private persistent coordinator state")
    command.add_argument("--host", default="127.0.0.1")
    command.add_argument("--port", type=int, default=8787)
    command.add_argument("--endpoint", help="advertised origin when different from the bind address")
    command.add_argument("--cert-file", help="server TLS certificate")
    command.add_argument("--key-file", help="server TLS private key")
    command.add_argument("--ca-file", help="CA path advertised to clients as informational metadata")
    command.add_argument("--token-file", help="read/create admin token file; default DIRECTORY/admin.token")
    command.add_argument("--submitter-token-file", help="read/create submitter token file; default DIRECTORY/submitter.token")
    command.add_argument("--max-runtime", type=float, default=3600, help="maximum job runtime in seconds")

    group = commands.add_parser("worker", help="enroll or serve with a persistent worker identity")
    actions = group.add_subparsers(dest="action", required=True, parser_class=parser_type)
    for name in ("join", "serve", "status", "leave"):
        command = actions.add_parser(name)
        command.add_argument("--directory", help="persistent identity directory; default MLZO_HOME/worker or ~/.mlzo/worker")
        command.add_argument("--home", help="configuration root; default MLZO_HOME or ~/.mlzo")
        command.add_argument("--ca-file", help="explicit trusted private CA certificate")
        if name == "join":
            command.add_argument("--token-file", help="invitation file; otherwise MLZO_INVITE_TOKEN")
            command.add_argument("--device", help="worker device, for example cpu or cuda:0")
            command.add_argument("--serve", action="store_true", help="serve immediately after enrollment")
        if name in {"join", "serve"}:
            command.add_argument("--poll-interval", type=float, default=0.5)
        if name == "leave":
            command.add_argument("--graceful", action=argparse.BooleanOptionalAction, default=True,
                                 help="finish current work before leaving; --no-graceful requests cancellation")

    group = commands.add_parser("run", help="inspect a remote run through a saved cluster")
    actions = group.add_subparsers(dest="action", required=True, parser_class=parser_type)
    for name in ("status", "events", "cancel", "checkpoint", "export", "resume"):
        command = actions.add_parser(name)
        command.add_argument("run_id")
        command.add_argument("--cluster", required=True, help="saved cluster name")
        command.add_argument("--home", help="configuration root; default MLZO_HOME or ~/.mlzo")
        if name == "events":
            command.add_argument("--after", type=int, default=0, help="event sequence cursor")
        if name in {"checkpoint", "export"}:
            command.add_argument("--to", required=name == "export", help="destination on the assigned worker filesystem")
        if name == "export":
            command.add_argument("--block", help="registered model submodule path")

    group = commands.add_parser("checkpoint", help="inspect local checkpoint storage")
    actions = group.add_subparsers(dest="action", required=True, parser_class=parser_type)
    command = actions.add_parser("list", help="list committed snapshots at a local root")
    command.add_argument("location")
    return parser


def _read_secret(path: str | Path | None, env: str, *, required: bool = False,
                 missing_ok: bool = False) -> str | None:
    value = None
    if path is not None:
        token_path = Path(path).expanduser()
        try:
            descriptor = os.open(token_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except FileNotFoundError:
            if not missing_ok:
                raise _CommandError("The token file does not exist.") from None
        else:
            with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise _CommandError("The token file must be a regular file.")
                value = stream.read(32769).strip()
    if value is None:
        value = os.environ.get(env)
    if value is not None and (not value or len(value) > 32768 or any(c.isspace() for c in value)):
        raise _CommandError("The credential must be a nonempty token without whitespace.")
    if value is None and required:
        raise _CommandError(f"Supply --token-file or set {env}.")
    return value


def _write_secret(path: str | Path, value: str) -> str:
    target = Path(path).expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except FileExistsError:
        raise _CommandError("The token output file already exists. Choose a new path.") from None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(value + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return str(target)


def _cluster(args: argparse.Namespace) -> Any:
    from mlzo.cluster.client import cluster

    if args.action == "create":
        token = _read_secret(args.token_file, "MLZO_TOKEN", required=True)
        return cluster(args.name, endpoint=args.endpoint, token=token, ca_file=args.ca_file, home=args.home).to_dict()
    client = cluster(args.name, home=args.home)
    if args.action == "status":
        return client.status()
    if args.action == "workers":
        return client.workers()
    if args.action == "invite":
        invitation = client.invite(expires_in=args.expires_in, max_joins=args.max_joins)
        if args.show_token:
            # Explicit opt-in is the only path allowed to emit this secret.
            result = {"invitation": invitation.encode(), "metadata": invitation.to_dict()}
        else:
            result = {"token_file": _write_secret(args.token_file, invitation.encode()), "metadata": invitation.to_dict()}
        return result
    if args.action == "drain":
        return client.drain(args.worker_id)
    if args.action == "remove":
        return client.remove(args.worker_id)
    if args.action == "shutdown":
        return client.shutdown()
    return client.scale(args.count)


_SERVED = object()


def _coordinator(args: argparse.Namespace, json_output: bool) -> Any:
    from mlzo.cluster.coordinator import Coordinator

    directory = Path(args.directory).expanduser()
    token_path = Path(args.token_file).expanduser() if args.token_file else directory / "admin.token"
    submitter_path = Path(args.submitter_token_file).expanduser() if args.submitter_token_file else directory / "submitter.token"
    token = _read_secret(token_path, "MLZO_TOKEN", missing_ok=True)
    submitter_token = _read_secret(submitter_path, "MLZO_SUBMITTER_TOKEN", missing_ok=True)
    if token_path.absolute() == submitter_path.absolute():
        raise _CommandError("Admin and submitter token files must be different.")
    service = Coordinator(directory, host=args.host, port=args.port, token=token,
                          submitter_token=submitter_token, cert_file=args.cert_file,
                          key_file=args.key_file, ca_file=args.ca_file, endpoint=args.endpoint,
                          max_runtime=args.max_runtime)
    # Coordinator also persists credentials in its private state. Standalone
    # token files let clients enroll without copying the entire credentials file.
    if not token_path.exists():
        _write_secret(token_path, service.token)
    if not submitter_path.exists():
        _write_secret(submitter_path, service.submitter_token)
    previous = signal.getsignal(signal.SIGTERM)

    def stop(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        service.start()
        _emit({"state": "serving", "endpoint": service.endpoint, "directory": str(directory.absolute()),
               "token_file": str(token_path.absolute()), "submitter_token_file": str(submitter_path.absolute())},
              json_output=json_output)
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        service.shutdown()
        signal.signal(signal.SIGTERM, previous)
    return _SERVED


def _worker(args: argparse.Namespace, json_output: bool) -> Any:
    from mlzo.cluster.worker import Worker

    if args.action == "join":
        invite = _read_secret(args.token_file, "MLZO_INVITE_TOKEN", required=True)
        instance = Worker.join(invite, directory=args.directory, home=args.home,
                               device=args.device, ca_file=args.ca_file)
    else:
        instance = Worker(args.directory, home=args.home, ca_file=args.ca_file)
    if args.action == "leave":
        return instance.leave(graceful=args.graceful)
    if args.action == "serve" or (args.action == "join" and args.serve):
        _emit({"state": "starting", "id": instance.id, "directory": str(instance.directory)}, json_output=json_output)
        try:
            instance.serve(poll_interval=args.poll_interval)
        except KeyboardInterrupt:
            instance.leave(graceful=True)
        return _SERVED
    return instance.status()


def _run(args: argparse.Namespace) -> Any:
    from mlzo.cluster.client import cluster

    if args.action == "resume":
        raise _CommandError("Remote resume is unsupported in MLZO 0.1.0a1. Local Run.fit(resume=...) restarts the saved epoch.")
    client = cluster(args.cluster, home=args.home)
    if args.action == "status":
        return client.get_run(args.run_id)
    if args.action == "events":
        return client.events(args.run_id, after=args.after)
    if args.action == "cancel":
        return client.cancel(args.run_id)
    if args.action == "checkpoint":
        return client.checkpoint_run(args.run_id, to=args.to)
    return client.export_run(args.run_id, to=args.to, block=args.block)


def _error(exc: Exception, json_output: bool) -> int:
    # Exceptions can include user code, URLs, or service responses. Never emit
    # arbitrary exception text, even after attempted string redaction.
    if isinstance(exc, (_UsageError, _CommandError)):
        kind, message = "ConfigurationError", str(exc)
    elif isinstance(exc, UnsupportedFeatureError):
        kind, message = "UnsupportedFeatureError", "This operation is unsupported in MLZO 0.1.0a1. See docs/alpha-support.md."
    elif isinstance(exc, ConfigurationError):
        kind, message = "ConfigurationError", "Invalid configuration. Check the command options and credential files."
    elif isinstance(exc, MLZOError):
        kind, message = type(exc).__name__, "The operation failed. Check the coordinator, worker, and saved configuration."
    elif isinstance(exc, OSError):
        kind, message = "OSError", "A file or service could not be accessed. Check paths, permissions, and service availability."
    else:
        kind, message = "OperationError", "The operation failed. Error details are omitted to protect credentials."
    if json_output:
        print(json.dumps({"ok": False, "error": {"type": kind, "message": message}}), flush=True)
    else:
        print(f"mlzo: {kind}: {message}", file=sys.stderr)
    return 2 if isinstance(exc, (_UsageError, _CommandError, ConfigurationError)) else 1


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    json_output = "--json" in arguments
    # A single global flag works before or after nested command names.
    arguments = [value for value in arguments if value != "--json"]
    try:
        parser = _parser(json_output)
        args = parser.parse_args(arguments)
        if args.version:
            if json_output:
                _emit({"version": __version__}, json_output=True)
            else:
                print(f"mlzo {__version__}")
            return 0
        if args.command is None:
            parser.print_help()
            return 0
        if args.command == "cluster":
            result = _cluster(args)
        elif args.command == "coordinator":
            result = _coordinator(args, json_output)
        elif args.command == "worker":
            result = _worker(args, json_output)
        elif args.command == "run":
            result = _run(args)
        else:
            from mlzo.checkpoint import list_snapshots
            result = list_snapshots(args.location)
        if result is not _SERVED:
            _emit(result, json_output=json_output)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        return _error(exc, json_output)


if __name__ == "__main__":
    raise SystemExit(main())
