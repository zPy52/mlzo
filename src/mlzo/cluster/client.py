"""Cluster configuration, durable submission and reconnect services for Python/CLI."""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any
import uuid

from mlzo.errors import ClusterError, ConfigurationError, TrainingError, UnsupportedFeatureError
from ._common import (MAX_ARCHIVE, TERMINAL, Transport, atomic_json, bounded_json,
                      budget, endpoint_url, identifier, mlzo_home, positive_number,
                      private_dir, read_json)


@dataclass(frozen=True, repr=False)
class Invitation:
    """An enrollment capability. Only encode() exposes the transferable secret."""
    _envelope: dict = field(repr=False)

    def __repr__(self) -> str:
        return f'Invitation(endpoint={self._envelope.get("endpoint")!r}, version=1)'

    def to_dict(self) -> dict:
        return {key: value for key, value in self._envelope.items() if key != 'secret'}

    def encode(self) -> str:
        return 'mlzo-invite-v1.' + base64.urlsafe_b64encode(bounded_json(self._envelope, 16 * 1024)).decode('ascii')

    @classmethod
    def decode(cls, value: Any) -> 'Invitation':
        if isinstance(value, cls):
            return value
        try:
            if not isinstance(value, str) or not value.startswith('mlzo-invite-v1.') or len(value) > 24000:
                raise ValueError()
            payload = json.loads(base64.b64decode(value.split('.', 1)[1], altchars=b'-_', validate=True))
            if (not isinstance(payload, dict) or payload.get('version') != 1
                    or not isinstance(payload.get('secret'), str) or not 16 <= len(payload['secret']) <= 256):
                raise ValueError()
            endpoint_url(payload['endpoint'])
        except (ValueError, TypeError, KeyError, UnicodeError):
            raise ConfigurationError('Invalid or unsupported MLZO invitation envelope.') from None
        return cls(payload)


class Cluster:
    """A control client, without network or hardware work during construction."""
    def __init__(self, name: str, *, endpoint: str | None = None, token: str | None = None,
                 ca_file: str | None = None, home: str | os.PathLike | None = None) -> None:
        self.name = identifier(name)
        self.endpoint = endpoint_url(endpoint) if endpoint is not None else None
        if token is not None and (not isinstance(token, str) or not token or len(token) > 256 or any(c.isspace() for c in token)):
            raise ConfigurationError('Invalid cluster credential.')
        self._token = token
        self.ca_file = str(ca_file) if ca_file is not None else None
        self.home = str(home) if home is not None else None
        self._transport: Transport | None = None

    def __repr__(self) -> str:
        return f'Cluster(name={self.name!r}, endpoint={self.endpoint!r})'

    def to_dict(self) -> dict:
        return {'name': self.name, 'endpoint': self.endpoint, 'ca_file': self.ca_file,
                'configured': self.endpoint is not None and self._token is not None}

    def _api(self) -> Transport:
        if self.endpoint is None or self._token is None:
            raise ConfigurationError('Cluster has no coordinator credentials. Configure endpoint= and token=; cluster creation does not start a server.')
        if self._transport is None:
            self._transport = Transport(self.endpoint, self._token, self.ca_file)
        return self._transport

    def status(self) -> dict:
        return self._api().request('GET', '/v1/status')

    def workers(self) -> list[dict]:
        return self._api().request('GET', '/v1/workers')['workers']

    def invite(self, *, expires_in: float = 900, max_joins: int = 1) -> Invitation:
        positive_number(expires_in, 'expires_in', 7 * 86400)
        return Invitation(self._api().request('POST', '/v1/invite', {'expires_in': expires_in, 'max_joins': max_joins}))

    def drain(self, worker_id: str) -> dict:
        return self._api().request('POST', f'/v1/workers/{identifier(worker_id)}/drain', {})

    def remove(self, worker_id: str) -> dict:
        return self._api().request('POST', f'/v1/workers/{identifier(worker_id)}/remove', {})

    def shutdown(self) -> dict:
        """Drain enrolled services and ask them to stop after existing work finishes."""
        return self._api().request('POST', '/v1/shutdown', {})

    def scale(self, count: int, **kwargs: Any) -> None:
        raise UnsupportedFeatureError('Managed provisioning and scaling, including AWS, are unavailable in MLZO 0.1.0a1. Enroll an existing machine with Worker.join().')

    def submit(self, run: Any, *, epochs: int | None = None, steps: int | None = None,
               parallel: Any = 'auto', code: str | os.PathLike | None = None,
               resume: Any = None, request_id: str | None = None,
               max_runtime: float = 3600) -> str:
        from .artifacts import snapshot_project
        from .serialization import collect_data_paths, serialize_run
        from .planning_helpers import placement_for
        budget(epochs, steps)
        if resume is not None:
            raise UnsupportedFeatureError('Remote resume submission is deferred. Completed snapshots are retained on the assigned worker.')
        positive_number(max_runtime, 'max_runtime', 7 * 86400)
        # Validate serialization and planning before upload or durable acceptance.
        serialize_run(run)
        plan = run.plan(on=self, parallel=parallel)
        project = _project_directory(code)
        paths = collect_data_paths(run)
        with tempfile.TemporaryDirectory(prefix='mlzo-submit-') as temporary:
            artifact = snapshot_project(project, destination=Path(temporary) / 'project.tar.gz', data_paths=paths, max_bytes=MAX_ARCHIVE)
            spec = serialize_run(run, data_paths=artifact.data_mapping)
            size = Path(artifact.path).stat().st_size
            if size > MAX_ARCHIVE:
                raise ConfigurationError('Project archive exceeds the 100 MiB alpha limit.')
            self._api().request('PUT', '/v1/archives/' + artifact.digest, raw=Path(artifact.path).read_bytes())
            value = {'request_id': identifier(request_id or run.id), 'spec': spec, 'archive': artifact.digest,
                     'fit': {'epochs': epochs, 'steps': steps, 'parallel': parallel, 'resume': None, 'max_runtime': max_runtime},
                     'placement': placement_for(run, plan), 'plan': plan.to_dict()}
            return self._api().request('POST', '/v1/runs', value)['id']

    def get_run(self, run_id: str) -> dict:
        return self._api().request('GET', '/v1/runs/' + identifier(run_id))

    def events(self, run_id: str, after: int = 0) -> list[dict]:
        if type(after) is not int or after < 0:
            raise ConfigurationError('after must be a nonnegative sequence number.')
        return self._api().request('GET', f'/v1/runs/{identifier(run_id)}/events?after={after}')['events']

    def wait(self, run_id: str, *, timeout: float | None = None, poll_interval: float = 0.5) -> dict:
        positive_number(poll_interval, 'poll_interval', 60)
        if timeout is not None:
            positive_number(timeout, 'timeout', 7 * 86400)
        start = time.monotonic()
        while True:
            value = self.get_run(run_id)
            if value['state'] in TERMINAL:
                return value
            if timeout is not None and time.monotonic() - start >= timeout:
                raise ClusterError('Timed out waiting for the remote run. The job remains assigned; reconnect to inspect it.')
            time.sleep(poll_interval)

    def cancel(self, run_id: str) -> dict:
        return self._api().request('POST', f'/v1/runs/{identifier(run_id)}/cancel', {})

    def _command(self, run_id: str, spec: dict, *, timeout: float = 120) -> dict:
        positive_number(timeout, 'timeout', 7 * 86400)
        value = self._api().request('POST', f'/v1/runs/{identifier(run_id)}/command', {'request_id': uuid.uuid4().hex, 'spec': spec})
        start = time.monotonic()
        while value['state'] not in {'completed', 'failed'}:
            if time.monotonic() - start > timeout:
                raise ClusterError(f'Remote command {value["id"]} is pending. Keep the assigned worker service running and inspect command_status().')
            time.sleep(0.25)
            value = self.command_status(value['id'])
        result = value['result']
        if value['state'] == 'failed':
            error = result.get('error', {})
            if error.get('type') == 'UnsupportedFeatureError':
                raise UnsupportedFeatureError(error.get('message', 'Remote operation is unsupported.'))
            raise TrainingError(error.get('message', 'Remote command failed.'))
        return result['reference']

    def command_status(self, command_id: str) -> dict:
        return self._api().request('GET', '/v1/commands/' + identifier(command_id))

    def checkpoint_run(self, run_id: str, to: Any = None) -> Any:
        from mlzo.checkpoint import CheckpointRef
        value = self.get_run(run_id)
        if value['state'] != 'completed':
            raise UnsupportedFeatureError('Remote manual checkpoint requires a completed job; use a checkpoint policy while training.')
        if to is None:
            if not value.get('checkpoint'):
                raise ClusterError('The completed job has no internal checkpoint.')
            return CheckpointRef.from_dict(value['checkpoint'])
        return CheckpointRef.from_dict(self._command(run_id, {'type': 'checkpoint', 'to': os.fspath(to)}))

    def export_run(self, run_id: str, to: Any, block: str | None = None) -> Any:
        from mlzo.inference import ArtifactRef
        return ArtifactRef.from_dict(self._command(run_id, {'type': 'export', 'to': os.fspath(to), 'block': block}))


def _project_directory(code: str | os.PathLike | None) -> Path:
    if code is not None:
        path = Path(code).expanduser().absolute()
        if not path.is_dir():
            raise ConfigurationError('code= must be an explicit project directory, not a script file.')
        return path
    current = Path.cwd()
    for directory in (current, *current.parents):
        if (directory / 'pyproject.toml').is_file():
            return directory
    raise ConfigurationError('Cannot infer the project root. Pass code=<project directory> or run inside a project with pyproject.toml.')


def cluster(name: str, *, endpoint: str | None = None, token: str | None = None,
            ca_file: str | None = None, home: str | os.PathLike | None = None) -> Cluster:
    """Create/load a private named configuration. This never starts a network service."""
    identifier(name)
    directory = private_dir(mlzo_home(home) / 'clusters')
    path = directory / (name + '.json')
    if path.is_symlink():
        raise ConfigurationError('Cluster configuration must not be a symlink.')
    stored = read_json(path) if path.exists() else {}
    result = Cluster(name, endpoint=endpoint if endpoint is not None else stored.get('endpoint'),
                     token=token if token is not None else stored.get('token'),
                     ca_file=ca_file if ca_file is not None else stored.get('ca_file'), home=home)
    atomic_json(path, {'version': 1, 'name': name, 'endpoint': result.endpoint,
                       'token': result._token, 'ca_file': result.ca_file})
    return result


def run(run_id: str, *, on: Cluster) -> Any:
    """Reconnect through the training module's normal remote Run delegation."""
    from mlzo.training import Run
    if not isinstance(on, Cluster):
        raise ConfigurationError('run(id, on=...) requires a Cluster.')
    status = on.get_run(identifier(run_id))
    result = Run(None)
    result.id, result._remote_cluster, result._state = run_id, on, status['state']
    return result


# Keep the planned factory spelling explicit without provisioning-shaped stubs.
from mlzo.providers import aws as _aws
cluster.aws = _aws
