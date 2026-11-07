"""Authenticated single-owner coordinator. Application code never runs here."""
from __future__ import annotations

import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import socket
import ssl
import tempfile
import threading
from typing import Any
from urllib.parse import parse_qs, urlsplit

from mlzo.errors import ClusterError, ConfigurationError, UnsupportedFeatureError
from ._common import (APIError, MAX_ARCHIVE, MAX_JSON, MAX_RESULT, TIMEOUT,
                      atomic_json, bounded_json, budget, digest_value, identifier,
                      loopback, positive_number, private_dir, read_json, reject_json_constant)
from ._store import Store


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    allow_reuse_address = True
    request_queue_size = 16

    def __init__(self, address: tuple, handler: type, owner: 'Coordinator', context: ssl.SSLContext | None):
        self.owner, self.context = owner, context
        self.slots = threading.BoundedSemaphore(16)
        if ':' in address[0]:
            self.address_family = socket.AF_INET6
        super().__init__(address, handler)

    def get_request(self) -> tuple:
        connection, address = super().get_request()
        connection.settimeout(TIMEOUT)
        if self.context is not None:
            try:
                connection = self.context.wrap_socket(connection, server_side=True)
            except BaseException:
                connection.close()
                raise
        return connection, address

    def process_request(self, request: socket.socket, address: tuple) -> None:
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request: socket.socket, address: tuple) -> None:
        try:
            super().process_request_thread(request, address)
        finally:
            self.slots.release()

    def handle_error(self, request: Any, client_address: Any) -> None:
        # The base implementation prints requests and tracebacks. Credentials
        # and application errors must not leak into coordinator logs.
        return


class _Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.0'
    server_version = 'MLZO/0.1'

    def log_message(self, *args: Any) -> None:
        return

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        self._reply(code, {'error': {'type': 'ClusterError', 'message': 'Invalid or unsupported HTTP request.'}})

    def _reply(self, status: int, value: Any, *, binary: bool = False) -> None:
        data = value if binary else bounded_json(value)
        self.send_response(status)
        self.send_header('Content-Type', 'application/octet-stream' if binary else 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(data)

    def _length(self, limit: int) -> int:
        values = self.headers.get_all('Content-Length', [])
        if self.headers.get('Transfer-Encoding') or len(values) != 1:
            raise APIError('Exactly one Content-Length is required; chunked requests are unsupported.', status=411)
        try:
            size = int(values[0])
        except ValueError:
            raise APIError('Invalid Content-Length.') from None
        if not 0 <= size <= limit:
            raise APIError('Request exceeds the size limit.', status=413)
        return size

    def _body(self) -> dict:
        if self.headers.get_content_type() != 'application/json':
            raise APIError('Expected application/json.', status=415)
        size = self._length(MAX_JSON)
        data = self.rfile.read(size)
        if len(data) != size:
            raise APIError('Incomplete request body.')
        try:
            value = json.loads(data, parse_constant=reject_json_constant)
            if not isinstance(value, dict):
                raise ValueError()
        except (ValueError, UnicodeError, RecursionError):
            raise APIError('Request must contain a JSON object.') from None
        return value

    def _auth(self, scope: str) -> str | None:
        header = self.headers.get('Authorization', '')
        if not header.startswith('Bearer ') or len(header) > 512:
            raise APIError('Authentication is required.', status=401)
        token = header[7:]
        owner = self.server.owner
        if scope in {'admin', 'submit'}:
            if hmac.compare_digest(token, owner._token):
                return None
            if scope == 'submit' and hmac.compare_digest(token, owner._submitter_token):
                return None
            raise APIError('This credential does not authorize this operation.', status=403)
        return owner._store.worker_auth(token)

    def _upload(self, digest: str) -> dict:
        self._auth('submit')
        digest = digest_value(digest)
        if self.headers.get_content_type() != 'application/octet-stream':
            raise APIError('Expected application/octet-stream.', status=415)
        remaining = size = self._length(MAX_ARCHIVE)
        owner = self.server.owner
        descriptor, temporary = tempfile.mkstemp(prefix='.upload-', dir=owner._archives)
        checksum = hashlib.sha256()
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise APIError('Incomplete project archive.')
                    remaining -= len(chunk)
                    checksum.update(chunk)
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            if checksum.hexdigest() != digest:
                raise APIError('Project archive digest mismatch.')
            target = owner._archives / digest
            if target.exists():
                if target.stat().st_size != size:
                    raise APIError('Immutable archive metadata mismatch.', status=409)
            else:
                os.replace(temporary, target)
            owner._store.archive(digest, size)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return {'digest': digest, 'size_bytes': size}

    def _archive_request(self, digest: str) -> tuple[Any, bool]:
        owner, store = self.server.owner, self.server.owner._store
        digest = digest_value(digest)
        if self.command == 'PUT':
            return self._upload(digest), False
        if self.command == 'GET':
            worker_id = self._auth('worker')
            store.require_archive_assignment(worker_id, digest)
            target = owner._archives / digest
            if not target.exists():
                raise APIError('Assigned archive is missing.', status=404)
            if target.stat().st_size > MAX_ARCHIVE:
                raise APIError('Stored archive exceeds the size limit.', status=413)
            data = target.read_bytes()
            if len(data) > MAX_ARCHIVE:
                raise APIError('Stored archive exceeds the size limit.', status=413)
            return data, True
        raise APIError('Unknown API endpoint.', status=404)

    def _worker_request(self, path: list[str]) -> dict:
        owner, store = self.server.owner, self.server.owner._store
        worker_id = self._auth('worker')
        if self.command != 'POST' or len(path) != 3:
            raise APIError('Unknown worker endpoint.', status=404)
        payload = self._body()
        if path[2] == 'heartbeat':
            inventory = payload.get('inventory')
            if inventory is not None:
                owner._inventory(inventory)
            return store.heartbeat(worker_id, inventory)
        if path[2] == 'claim':
            known = payload.get('known_run')
            if known is not None:
                identifier(known)
            return store.claim(worker_id, known)
        if path[2] == 'report':
            owner._report(payload)
            return store.report(worker_id, payload)
        if path[2] == 'drain':
            return store.manage(worker_id, 'drain')
        if path[2] == 'leave':
            store.leave(worker_id)
            return {'state': 'leaving'}
        if path[2] == 'command':
            return {'command': store.claim_command(worker_id)}
        if path[2] == 'command-result':
            bounded_json(payload['result'], MAX_RESULT)
            return store.report_command(worker_id, payload['id'], payload['result'])
        raise APIError('Unknown API endpoint.', status=404)

    def _run_request(self, path: list[str], query: str) -> dict:
        store = self.server.owner._store
        self._auth('submit')
        run_id = identifier(path[2])
        if len(path) == 3 and self.command == 'GET':
            return store.get_run(run_id)
        if len(path) == 4:
            action = path[3]
            if action == 'events' and self.command == 'GET':
                after = int(parse_qs(query).get('after', ['0'])[0])
                if after < 0:
                    raise ConfigurationError('after must be nonnegative.')
                return {'events': store.events(run_id, after)}
            if action == 'cancel' and self.command == 'POST':
                self._body()
                return store.cancel(run_id)
            if action == 'command' and self.command == 'POST':
                value = self._body()
                spec = value['spec']
                if not isinstance(spec, dict) or spec.get('type') not in {'checkpoint', 'export'}:
                    raise ConfigurationError('Unknown completed-job command.')
                for field in ('to', 'block'):
                    if spec.get(field) is not None and (
                        not isinstance(spec[field], str)
                        or len(spec[field]) > 4096
                        or '\x00' in spec[field]
                    ):
                        raise ConfigurationError('Invalid remote path or block.')
                if spec['type'] == 'export' and not spec.get('to'):
                    raise ConfigurationError('Remote export requires to=.')
                return store.command(run_id, spec, value['request_id'])
        raise APIError('Unknown API endpoint.', status=404)

    def _route(self) -> tuple[Any, bool]:
        owner, store = self.server.owner, self.server.owner._store
        parsed = urlsplit(self.path)
        path = parsed.path.strip('/').split('/')
        if not path or path[0] != 'v1':
            raise APIError('Unknown API endpoint.', status=404)
        if path[:2] == ['v1', 'archives'] and len(path) == 3:
            return self._archive_request(path[2])
        if self.command == 'POST' and path == ['v1', 'enroll']:
            payload = self._body()
            if not isinstance(payload.get('secret'), str) or len(payload['secret']) > 256:
                raise APIError('Invalid enrollment secret.', status=401)
            owner._inventory(payload.get('inventory'))
            return store.enroll(payload['secret'], payload['request_id'], payload['inventory']), False
        if path[:2] == ['v1', 'worker']:
            return self._worker_request(path), False
        if path[:2] == ['v1', 'commands'] and len(path) == 3 and self.command == 'GET':
            self._auth('submit')
            return store.command_status(path[2]), False
        if path == ['v1', 'status'] and self.command == 'GET':
            self._auth('submit')
            counts = store.run_counts()
            return {
                'state': 'ready',
                'generation': 0,
                'workers': store.workers(),
                'runs': counts,
                'capabilities': ['single_worker', 'completed_checkpoint', 'completed_export'],
            }, False
        if path == ['v1', 'workers'] and self.command == 'GET':
            self._auth('submit')
            return {'workers': store.workers()}, False
        if path == ['v1', 'invite'] and self.command == 'POST':
            self._auth('admin')
            payload = self._body()
            expires = positive_number(payload.get('expires_in', 900), 'expires_in', 7 * 86400)
            joins = payload.get('max_joins', 1)
            if type(joins) is not int or not 1 <= joins <= 256:
                raise ConfigurationError('max_joins must be an integer between 1 and 256.')
            return {'version': 1, 'endpoint': owner.endpoint,
                    'secret': store.invite(expires, joins), 'expires_in': expires,
                    'max_joins': joins, 'ca_file': owner.ca_file}, False
        if len(path) == 4 and path[:2] == ['v1', 'workers'] and self.command == 'POST':
            self._auth('admin')
            self._body()
            return store.manage(path[2], path[3]), False
        if path == ['v1', 'shutdown'] and self.command == 'POST':
            self._auth('admin')
            self._body()
            return store.stop_all(), False
        if path == ['v1', 'runs'] and self.command == 'POST':
            self._auth('submit')
            payload = self._body()
            owner._submission(payload)
            return {'id': store.submit(payload)}, False
        if len(path) >= 3 and path[:2] == ['v1', 'runs']:
            return self._run_request(path, parsed.query), False
        raise APIError('Unknown API endpoint.', status=404)

    def _handle(self) -> None:
        try:
            value, binary = self._route()
            self._reply(200, value, binary=binary)
        except APIError as exc:
            self._reply(exc.status, {'error': {'type': exc.code, 'message': str(exc)}})
        except UnsupportedFeatureError as exc:
            self._reply(422, {'error': {'type': 'UnsupportedFeatureError', 'message': str(exc)}})
        except ConfigurationError as exc:
            self._reply(400, {'error': {'type': 'ConfigurationError', 'message': str(exc)}})
        except (KeyError, TypeError, ValueError, OverflowError, RecursionError):
            self._reply(400, {'error': {'type': 'ConfigurationError', 'message': 'Malformed control request.'}})
        except (OSError, TimeoutError):
            # A disconnected peer cannot receive an error response.
            try:
                self._reply(408, {'error': {'type': 'ClusterError', 'message': 'Request timed out or storage is unavailable.'}})
            except OSError:
                pass
        except Exception:
            self._reply(500, {'error': {'type': 'ClusterError', 'message': 'Coordinator operation failed.'}})

    do_GET = do_POST = do_PUT = _handle


class Coordinator:
    """Explicit HTTPS service; HTTP is restricted to loopback development.

    ``start()`` and context-manager entry start a background server. ``serve()``
    blocks for CLI use. ``shutdown()`` stops this server, while Cluster.shutdown
    requests graceful worker-service termination and keeps the coordinator up.
    """
    def __init__(self, directory: str | os.PathLike, *, host: str = '127.0.0.1', port: int = 0,
                 token: str | None = None, submitter_token: str | None = None,
                 cert_file: str | None = None, key_file: str | None = None,
                 ca_file: str | None = None, endpoint: str | None = None,
                 max_runtime: float = 3600) -> None:
        if not loopback(host) and not (cert_file and key_file):
            raise ConfigurationError('Nonloopback coordinators require a TLS certificate and key.')
        if bool(cert_file) != bool(key_file):
            raise ConfigurationError('Provide both cert_file and key_file.')
        if type(port) is not int or not 0 <= port <= 65535:
            raise ConfigurationError('port must be between 0 and 65535.')
        self.directory = private_dir(Path(directory))
        credentials = self.directory / 'credentials.json'
        stored = read_json(credentials) if credentials.exists() else {}
        self._token = token or stored.get('admin_token') or secrets.token_urlsafe(32)
        self._submitter_token = submitter_token or stored.get('submitter_token') or secrets.token_urlsafe(32)
        if self._token == self._submitter_token:
            raise ConfigurationError('Admin and submitter credentials must be distinct.')
        for credential in (self._token, self._submitter_token):
            if not isinstance(credential, str) or not credential or len(credential) > 256 or any(c.isspace() for c in credential):
                raise ConfigurationError('Tokens must be nonempty strings without whitespace, at most 256 characters.')
        enrollment_key = stored.get('enrollment_key') or secrets.token_hex(32)
        atomic_json(credentials, {'admin_token': self._token, 'submitter_token': self._submitter_token, 'enrollment_key': enrollment_key})
        self._store = Store(self.directory / 'coordinator.sqlite3', enrollment_key)
        self._archives = private_dir(self.directory / 'archives')
        self.host, self.port = host, port
        self.cert_file, self.key_file, self.ca_file = cert_file, key_file, str(ca_file) if ca_file else None
        self._advertised = endpoint
        if endpoint is not None:
            from ._common import endpoint_url
            endpoint_url(endpoint)
            if cert_file and not endpoint.startswith('https://'):
                raise ConfigurationError('A TLS server must advertise an HTTPS endpoint.')
        self.max_runtime = positive_number(max_runtime, 'max_runtime', 7 * 86400)
        self._server: _Server | None = None
        self._thread: threading.Thread | None = None
        self._serving = threading.Event()

    def __repr__(self) -> str:
        return f'Coordinator(host={self.host!r}, port={self.port})'

    @property
    def token(self) -> str:
        """Explicit access for configuring an admin client; never included in repr."""
        return self._token

    @property
    def submitter_token(self) -> str:
        return self._submitter_token

    @property
    def endpoint(self) -> str:
        if self._advertised:
            return self._advertised
        host = '127.0.0.1' if self.host == '0.0.0.0' else self.host
        if ':' in host:
            host = '[' + host + ']'
        return f'{"https" if self.cert_file else "http"}://{host}:{self.port}'

    def _bind(self) -> None:
        if self._server is not None:
            return
        context = None
        if self.cert_file:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.load_cert_chain(self.cert_file, self.key_file)
        self._server = _Server((self.host, self.port), _Handler, self, context)
        self.port = self._server.server_port

    def serve(self) -> None:
        self._bind()
        if self._serving.is_set():
            raise ClusterError('Coordinator is already serving.')
        self._serving.set()
        try:
            self._server.serve_forever(poll_interval=0.2)
        finally:
            self._serving.clear()

    def start(self) -> 'Coordinator':
        if self._thread is not None and self._thread.is_alive():
            return self
        self._bind()
        self._thread = threading.Thread(target=self.serve, name='mlzo-coordinator', daemon=True)
        self._thread.start()
        self._serving.wait(timeout=2)
        return self

    def shutdown(self) -> None:
        if self._server is not None:
            if self._serving.is_set():
                self._server.shutdown()
            self._server.server_close()
            if self._thread is not None and self._thread is not threading.current_thread():
                self._thread.join(timeout=5)
            self._server = None

    stop = shutdown

    def __enter__(self) -> 'Coordinator':
        return self.start()

    def __exit__(self, *args: Any) -> None:
        self.shutdown()

    @staticmethod
    def _inventory(value: Any) -> None:
        if not isinstance(value, dict) or not isinstance(value.get('devices'), list) or not 1 <= len(value['devices']) <= 64:
            raise ConfigurationError('Worker inventory needs 1-64 device records.')
        bounded_json(value, 64 * 1024)
        for item in value['devices']:
            if not isinstance(item, dict) or item.get('type') not in {'cpu', 'cuda', 'mps'} or not isinstance(item.get('device'), str):
                raise ConfigurationError('Invalid worker device inventory.')

    def _submission(self, value: dict) -> None:
        from .serialization import validate_run_spec
        validate_run_spec(value['spec'])
        digest_value(value['archive'])
        identifier(value['request_id'])
        fit = value['fit']
        budget(fit.get('epochs'), fit.get('steps'))
        if fit.get('parallel') not in {'auto', 'single', 'none', None}:
            raise UnsupportedFeatureError('Only single-worker single-device execution is supported.')
        fit['max_runtime'] = min(positive_number(fit.get('max_runtime', self.max_runtime), 'max_runtime', 7 * 86400), self.max_runtime)
        if fit.get('resume') is not None:
            raise UnsupportedFeatureError('Remote submission resume is deferred; completed snapshots remain on the worker.')
        placement = value['placement']
        if not isinstance(placement, dict) or placement.get('precision') not in {'auto', 'fp32', 'fp16', 'bf16'}:
            raise ConfigurationError('Invalid single-worker placement request.')
        if placement.get('worker_id') is not None:
            identifier(placement['worker_id'])
        if placement.get('device') is not None:
            from .planning_helpers import validate_device
            validate_device(placement['device'])
        plan = value['plan']
        if not isinstance(plan, dict) or plan.get('strategy') != 'single':
            raise ConfigurationError('A validated single-device plan is required before acceptance.')

    @staticmethod
    def _report(value: dict) -> None:
        identifier(value['id'])
        events = value.get('events', [])
        if not isinstance(events, list) or len(events) > 100:
            raise ConfigurationError('Report at most 100 scalar events at once.')
        for event in events:
            bounded_json(event, 16 * 1024)
        if value.get('result') is not None:
            bounded_json(value['result'], MAX_RESULT)
