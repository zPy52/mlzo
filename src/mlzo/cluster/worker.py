"""Persistent scoped worker, polling supervisor and credential-free subprocesses."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Iterator
import uuid

from mlzo.errors import ClusterError, ConfigurationError, UnsupportedFeatureError
from ._common import (APIError, MAX_ARCHIVE, MAX_RESULT, Transport, atomic_json,
                      identifier, mlzo_home, positive_number, private_dir, read_json)
from .client import Invitation
from ._launcher import child_environment
from ._reports import error_value, write_result


def _directory(directory: str | os.PathLike | None, home: str | os.PathLike | None) -> Path:
    return private_dir(Path(directory) if directory is not None else mlzo_home(home) / 'worker')


def _inventory(device: str | None) -> dict:
    import platform
    import torch
    from mlzo._version import __version__
    from .planning_helpers import validate_device
    if device is not None:
        device = validate_device(device)
    if device == 'cpu':
        devices, notes = [{'device': 'cpu', 'type': 'cpu', 'bf16': None}], []
    else:
        from mlzo.planning.devices import local_inventory
        devices, notes = local_inventory()
        if device is not None:
            devices = [item for item in devices if item['device'] == device]
            if not devices:
                raise ConfigurationError('The selected worker device is unavailable on this machine.')
    return {'devices': devices, 'versions': {'python': platform.python_version(), 'torch': str(torch.__version__),
            'cuda': torch.version.cuda, 'mlzo': __version__}, 'cpu_count': os.cpu_count(), 'notes': notes[:16]}


@contextmanager
def _service_lock(directory: Path) -> Iterator[None]:
    try:
        import fcntl
    except ImportError:
        raise UnsupportedFeatureError('The alpha worker supervisor requires POSIX process groups and file locking.') from None
    descriptor = os.open(directory / 'service.lock', os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ClusterError('A worker service is already using this identity directory.') from None
        yield
    finally:
        os.close(descriptor)


class Worker:
    """A single job at a time on one device. serve() is explicitly blocking."""
    def __init__(self, directory: str | os.PathLike | None = None, *, home: str | os.PathLike | None = None,
                 ca_file: str | None = None) -> None:
        self.directory = _directory(directory, home)
        config_path = self.directory / 'identity.json'
        if not config_path.exists():
            raise ConfigurationError('No enrolled worker identity here. Call Worker.join(invite, directory=...) first.')
        if config_path.is_symlink():
            raise ConfigurationError('Worker identity must not be a symlink.')
        config = read_json(config_path)
        self.id, self.endpoint = identifier(config['id']), config['endpoint']
        self.device = config.get('device')
        self._config = config
        self._transport = Transport(self.endpoint, config['token'], ca_file or config.get('ca_file'))
        self._stop_requested = threading.Event()
        self._force_stop = threading.Event()
        self._serving = threading.Event()
        self._thread: threading.Thread | None = None
        self._child: subprocess.Popen | None = None
        self._last_error: str | None = None
        self._last_contact: float | None = None
        self._draining = False
        self._revoked = False
        private_dir(self.directory / 'jobs')
        private_dir(self.directory / 'cache')
        private_dir(self.directory / 'commands')

    def __repr__(self) -> str:
        return f'Worker(id={self.id!r}, endpoint={self.endpoint!r}, device={self.device!r})'

    @classmethod
    def join(cls, invite: Invitation | str, *, directory: str | os.PathLike | None = None,
             home: str | os.PathLike | None = None, device: str | None = None,
             ca_file: str | None = None) -> 'Worker':
        invitation = Invitation.decode(invite)
        envelope = invitation._envelope
        target = _directory(directory, home)
        identity = target / 'identity.json'
        if identity.exists():
            worker = cls(target, ca_file=ca_file)
            if worker.endpoint != envelope['endpoint']:
                raise ConfigurationError('This directory is enrolled with a different coordinator.')
            if device is not None and worker.device != device:
                raise ConfigurationError('This identity has another selected device. Use a separate worker directory.')
            return worker
        pending_path = target / 'enrollment.json'
        if pending_path.exists():
            pending = read_json(pending_path)
            if pending['invite_digest'] != hashlib.sha256(invitation.encode().encode()).hexdigest():
                raise ConfigurationError('This worker directory has a pending enrollment with a different invitation.')
            inventory = pending['inventory']
            if pending.get('device') != device:
                raise ConfigurationError('Retry enrollment with the same selected device.')
        else:
            inventory = _inventory(device)
            pending = {'request_id': uuid.uuid4().hex, 'inventory': inventory, 'device': device,
                       'invite_digest': hashlib.sha256(invitation.encode().encode()).hexdigest()}
            atomic_json(pending_path, pending)
        # An invitation's CA path is informational. It never disables normal
        # verification or injects a certificate. Operators explicitly supply a
        # readable trusted ca_file on this machine when a private CA is needed.
        transport = Transport(envelope['endpoint'], ca_file=ca_file)
        result = transport.request('POST', '/v1/enroll', {'secret': envelope['secret'],
                                   'request_id': pending['request_id'], 'inventory': inventory})
        atomic_json(identity, {'version': 1, 'id': result['id'], 'token': result['token'],
                    'endpoint': envelope['endpoint'], 'device': device, 'ca_file': str(ca_file) if ca_file else None,
                    'inventory': inventory})
        pending_path.unlink(missing_ok=True)
        return cls(target)

    def status(self) -> dict:
        active_path = self.directory / 'active.json'
        active = read_json(active_path) if active_path.exists() else None
        serving = self._serving.is_set()
        if not serving and (self.directory / 'service.lock').exists():
            try:
                import fcntl
                with (self.directory / 'service.lock').open('rb') as lock:
                    try:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        serving = True
            except (OSError, ImportError):
                pass
        persisted_path = self.directory / 'service-state.json'
        persisted = read_json(persisted_path) if not self._serving.is_set() and persisted_path.exists() else {}
        return {'id': self.id, 'endpoint': self.endpoint, 'device': self.device,
                'state': 'serving' if serving else 'stopped',
                'draining': self._draining or self._stop_requested.is_set() or persisted.get('draining', False),
                'active': {'kind': active['kind'], 'id': active['id']} if active else None,
                'last_contact': self._last_contact or persisted.get('last_contact'),
                'error': self._last_error or persisted.get('error'),
                'inventory': persisted.get('inventory', self._config.get('inventory'))}

    to_dict = status

    def stop(self, *, graceful: bool = True) -> None:
        self._draining = True
        self._stop_requested.set()
        if not graceful:
            self._force_stop.set()
        if not self._serving.is_set() and not graceful:
            active_path = self.directory / 'active.json'
            if active_path.exists():
                self._request_cancel(read_json(active_path), 'Worker stopped by its owner.')

    def leave(self, graceful: bool = True) -> dict:
        self.stop(graceful=graceful)
        try:
            self._transport.request('POST', '/v1/worker/leave', {})
        except ClusterError:
            self._last_error = 'Coordinator unavailable during leave; local stop remains requested.'
        return self.status()

    def start(self, *, poll_interval: float = 0.5) -> 'Worker':
        if self._thread is not None and self._thread.is_alive():
            return self
        self._thread = threading.Thread(target=self.serve, kwargs={'poll_interval': poll_interval},
                                        name='mlzo-worker-' + self.id[:8], daemon=True)
        self._thread.start()
        return self

    @staticmethod
    def _alive(active: dict, directory: Path) -> bool:
        child_path = directory / 'child.json'
        if not child_path.exists():
            # Popen has returned but the interpreter may still be starting.
            child = active
        else:
            child = read_json(child_path)
        pid = child.get('pid')
        if type(pid) is not int or pid < 1 or child.get('nonce') != active['nonce']:
            return False
        try:
            os.kill(pid, 0)
            output = subprocess.run(['ps', '-ww', '-p', str(pid), '-o', 'args='], capture_output=True,
                                    text=True, check=False, timeout=3).stdout
            return active['nonce'] in output and ('mlzo.cluster._launcher' in output or 'mlzo.cluster._command' in output)
        except (OSError, subprocess.SubprocessError):
            return False

    def _spawn(self, active: dict, directory: Path, argv: list[str], cwd: Path) -> None:
        # Persist an intent before creating the process. Ambiguous interrupted
        # launches fail explicitly on restart instead of replaying user code.
        atomic_json(directory / 'launch.json', active)
        atomic_json(self.directory / 'active.json', active)
        environment = child_environment(directory)
        # Only MLZO's installation/source root is added for the preparation
        # interpreter, never the parent's arbitrary PYTHONPATH or secrets.
        environment['PYTHONPATH'] = str(Path(__file__).resolve().parents[2])
        try:
            self._child = subprocess.Popen(argv, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            active['pid'] = self._child.pid
            atomic_json(self.directory / 'active.json', active)
        except BaseException as exc:
            if active['kind'] == 'run':
                write_result(directory, {'state': 'failed', 'history': None, 'counters': {}, 'error': error_value(exc, 'launch')})
            else:
                atomic_json(directory / 'result.json', {'error': error_value(exc, 'command_launch')})

    def _launch_run(self, job: dict) -> None:
        directory = private_dir(self.directory / 'jobs' / identifier(job['id']))
        existing = directory / 'launch.json'
        if existing.exists():
            active = read_json(existing)
            child_path = directory / 'child.json'
            if child_path.exists():
                active['pid'] = read_json(child_path)['pid']
            atomic_json(self.directory / 'active.json', active)
            return
        archive = self.directory / 'cache' / job['archive']
        if not archive.exists():
            raw = self._transport.request('GET', '/v1/archives/' + job['archive'], binary=True)
            if hashlib.sha256(raw).hexdigest() != job['archive']:
                raise ClusterError('Downloaded archive digest mismatch.')
            temporary = archive.with_suffix('.partial')
            with temporary.open('wb') as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, archive)
        if archive.stat().st_size > MAX_ARCHIVE:
            raise ClusterError('Cached archive exceeds the alpha size limit.')
        assignment = {**job, 'archive_path': str(archive),
                      'worker_runtime': self._config.get('inventory', {}).get('versions', {})}
        atomic_json(directory / 'assignment.json', assignment)
        active = {'kind': 'run', 'id': job['id'], 'run_id': job['id'], 'pid': None,
                  'nonce': secrets.token_hex(24), 'started': time.time(), 'max_runtime': job['fit']['max_runtime']}
        self._spawn(active, directory, [sys.executable, '-m', 'mlzo.cluster._launcher', str(directory), active['nonce']], directory)

    def _launch_command(self, command: dict) -> None:
        directory = private_dir(self.directory / 'commands' / identifier(command['id']))
        existing = directory / 'launch.json'
        if existing.exists():
            active = read_json(existing)
            child_path = directory / 'child.json'
            if child_path.exists():
                active['pid'] = read_json(child_path)['pid']
            atomic_json(self.directory / 'active.json', active)
            return
        job = self.directory / 'jobs' / identifier(command['run_id'])
        atomic_json(directory / 'request.json', command)
        python = job / 'venv' / 'bin' / 'python'
        active = {'kind': 'command', 'id': command['id'], 'run_id': command['run_id'], 'pid': None,
                  'nonce': secrets.token_hex(24), 'started': time.time(), 'max_runtime': 300}
        self._spawn(active, directory, [str(python), '-m', 'mlzo.cluster._command', str(job), str(directory), active['nonce']], job / 'project')

    def _request_cancel(self, active: dict, reason: str) -> None:
        directory = self.directory / ('jobs' if active['kind'] == 'run' else 'commands') / active['id']
        marker = directory / 'cancel.json'
        if not marker.exists():
            atomic_json(marker, {'requested_at': time.time(), 'reason': reason, 'timeout': reason.startswith('Runtime')})
        value = read_json(marker)
        elapsed = time.time() - value['requested_at']
        if self._alive(active, directory) and elapsed >= 5:
            try:
                os.killpg(active['pid'], signal.SIGKILL if elapsed >= 10 else signal.SIGTERM)
            except ProcessLookupError:
                pass

    def _monitor(self) -> bool:
        active_path = self.directory / 'active.json'
        if not active_path.exists():
            return False
        active = read_json(active_path)
        directory = self.directory / ('jobs' if active['kind'] == 'run' else 'commands') / active['id']
        child_path = directory / 'child.json'
        if child_path.exists():
            child = read_json(child_path)
            if child.get('nonce') == active['nonce'] and child.get('pid') != active.get('pid'):
                active['pid'] = child['pid']
                atomic_json(active_path, active)
        if self._child is not None:
            self._child.poll()  # Reap children before checking identity via ps.
        alive = self._alive(active, directory)
        # Allow interpreter startup time even if its nonce marker is not written yet.
        if not alive and active.get('pid') and time.time() - active['started'] < 3:
            alive = True
        if alive and (self._force_stop.is_set() or time.time() - active['started'] >= active['max_runtime']):
            self._request_cancel(active, 'Worker stopped by its owner.' if self._force_stop.is_set() else 'Runtime budget exceeded.')
        if (directory / 'cancel.json').exists() and alive:
            self._request_cancel(active, read_json(directory / 'cancel.json')['reason'])
        result_path = directory / 'result.json'
        if not alive and not result_path.exists():
            marker = read_json(directory / 'cancel.json') if (directory / 'cancel.json').exists() else {}
            error = {'type': 'TimeoutError' if marker.get('timeout') else 'WorkerProcessError',
                     'message': marker.get('reason', 'Job process stopped or launch was interrupted. No automatic replay.'), 'phase': 'supervisor'}
            if active['kind'] == 'run':
                write_result(directory, {'state': 'cancelled' if marker and not marker.get('timeout') else 'failed',
                                        'history': None, 'counters': {}, 'error': error})
            else:
                atomic_json(result_path, {'error': error})
        if self._revoked:
            # Retain the unacknowledged report for inspection, but stop the
            # service once its revoked job has actually stopped.
            return alive
        if active['kind'] == 'command':
            if not alive and result_path.exists():
                self._transport.request('POST', '/v1/worker/command-result', {'id': active['id'], 'result': read_json(result_path, MAX_RESULT)})
                active_path.unlink(missing_ok=True)
                self._child = None
                return False
            return True
        events_path, ack_path = directory / 'events.json', directory / 'ack.json'
        events = read_json(events_path).get('events', []) if events_path.exists() else []
        ack = read_json(ack_path)['seq'] if ack_path.exists() else 0
        pending = [event for event in events if event['seq'] > ack]
        report = {'id': active['id'], 'events': pending[:100], 'state': 'running' if events_path.exists() else 'preparing'}
        terminal = not alive and result_path.exists() and len(pending) <= 100
        if terminal:
            report['result'] = read_json(result_path, MAX_RESULT)
            marker = read_json(directory / 'cancel.json') if (directory / 'cancel.json').exists() else {}
            if marker.get('timeout'):
                report['result']['state'] = 'failed'
                report['result']['error'] = {'type': 'TimeoutError', 'message': marker['reason'], 'phase': 'supervisor'}
                if isinstance(report['result'].get('history'), dict):
                    report['result']['history']['state'] = 'failed'
        response = self._transport.request('POST', '/v1/worker/report', report)
        atomic_json(ack_path, {'seq': response['ack']})
        if terminal:
            active_path.unlink(missing_ok=True)
            self._child = None
            return False
        return True

    def serve(self, *, poll_interval: float = 0.5) -> None:
        positive_number(poll_interval, 'poll_interval', 60)
        with _service_lock(self.directory):
            if self._serving.is_set():
                raise ClusterError('This worker is already serving.')
            self._serving.set()
            old_signals = {}
            if threading.current_thread() is threading.main_thread():
                def stopping(signum: int, frame: Any) -> None:
                    self.stop(graceful=signum == signal.SIGTERM)
                for signum in (signal.SIGINT, signal.SIGTERM):
                    old_signals[signum] = signal.signal(signum, stopping)
            try:
                inventory = _inventory(self.device)
                self._config['inventory'] = inventory
                while True:
                    active_path = self.directory / 'active.json'
                    try:
                        active = self._monitor()
                        if self._revoked:
                            if not active:
                                break
                            time.sleep(poll_interval)
                            continue
                        heartbeat = self._transport.request('POST', '/v1/worker/heartbeat', {'inventory': inventory})
                        self._last_contact, self._last_error = time.time(), None
                        self._draining = bool(heartbeat['drain']) or self._stop_requested.is_set()
                        if heartbeat['stop']:
                            self._stop_requested.set()
                        assigned = heartbeat.get('active_run')
                        if active and assigned and assigned.get('cancel'):
                            self._request_cancel(read_json(active_path), 'Cancelled by submitter.')
                        if not active and self._stop_requested.is_set():
                            break
                        if not active:
                            command = self._transport.request('POST', '/v1/worker/command', {})['command']
                            if command:
                                self._launch_command(command)
                            elif not self._draining:
                                response = self._transport.request('POST', '/v1/worker/claim', {'known_run': None})
                                if response.get('job'):
                                    self._launch_run(response['job'])
                        if not active_path.exists() and self._stop_requested.is_set():
                            break
                    except APIError as exc:
                        if exc.status in {401, 403}:
                            self._last_error = 'Worker membership was revoked or authorization failed.'
                            self._revoked = True
                            self._stop_requested.set()
                            self._force_stop.set()
                            if not active_path.exists():
                                break
                        else:
                            self._last_error = 'Coordinator rejected a worker operation; local work remains owned.'
                    except ClusterError:
                        self._last_error = 'Coordinator unavailable; local computation and reports remain on disk.'
                        if self._stop_requested.is_set() and not active_path.exists():
                            break
                    except (OSError, ValueError) as exc:
                        self._last_error = f'Worker preparation/control error: {type(exc).__name__}.'
                    atomic_json(self.directory / 'service-state.json', {
                        'last_contact': self._last_contact, 'error': self._last_error,
                        'draining': self._draining or self._stop_requested.is_set(), 'inventory': inventory})
                    time.sleep(poll_interval)
            finally:
                self._serving.clear()
                for signum, handler in old_signals.items():
                    signal.signal(signum, handler)


def join(invite: Invitation | str, **kwargs: Any) -> Worker:
    return Worker.join(invite, **kwargs)
