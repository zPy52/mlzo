"""One-owner SQLite journal. No liveness timeout changes job ownership."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import hmac
import json
from pathlib import Path
import secrets
import sqlite3
import threading
import time
from typing import Any, Iterator
import uuid

from mlzo.errors import ConfigurationError, UnsupportedFeatureError
from ._common import (APIError, MAX_EVENT, MAX_EVENTS, MAX_RESULT, TERMINAL,
                      bounded_json, identifier, secret_hash)


def _dump(value: Any, limit: int = MAX_RESULT) -> str:
    return bounded_json(value, limit).decode('ascii')


def _load(value: str | None, default: Any = None) -> Any:
    return json.loads(value) if value is not None else default


class Store:
    def __init__(self, path: Path, enrollment_key: str) -> None:
        path.touch(mode=0o600, exist_ok=True)
        path.chmod(0o600)
        self._key = enrollment_key.encode()
        self._lock = threading.RLock()
        self.db = sqlite3.connect(path, timeout=15, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS workers (
                id TEXT PRIMARY KEY, token_hash TEXT UNIQUE NOT NULL,
                inventory TEXT NOT NULL, last_seen REAL NOT NULL,
                drain INTEGER NOT NULL DEFAULT 0, revoked INTEGER NOT NULL DEFAULT 0,
                stop INTEGER NOT NULL DEFAULT 0, active_run TEXT, error TEXT);
            CREATE TABLE IF NOT EXISTS invites (
                digest TEXT PRIMARY KEY, expires REAL NOT NULL,
                max_joins INTEGER NOT NULL, joins INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS enrollments (
                request_id TEXT PRIMARY KEY, invite_digest TEXT NOT NULL,
                worker_id TEXT NOT NULL, inventory TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS archives (
                digest TEXT PRIMARY KEY, size INTEGER NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, request_id TEXT UNIQUE NOT NULL, fingerprint TEXT NOT NULL,
                spec TEXT NOT NULL, archive TEXT NOT NULL, fit TEXT NOT NULL,
                placement TEXT NOT NULL, plan TEXT NOT NULL,
                state TEXT NOT NULL, worker_id TEXT, device TEXT, generation INTEGER NOT NULL DEFAULT 0,
                created REAL NOT NULL, updated REAL NOT NULL,
                cancel INTEGER NOT NULL DEFAULT 0, counters TEXT, result TEXT);
            CREATE TABLE IF NOT EXISTS events (
                run_id TEXT NOT NULL, seq INTEGER NOT NULL, payload TEXT NOT NULL,
                PRIMARY KEY(run_id, seq));
            CREATE TABLE IF NOT EXISTS commands (
                id TEXT PRIMARY KEY, run_id TEXT NOT NULL, worker_id TEXT NOT NULL,
                spec TEXT NOT NULL, state TEXT NOT NULL, result TEXT, created REAL NOT NULL);
        ''')

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self.db.execute('BEGIN IMMEDIATE')
            try:
                yield self.db
            except BaseException:
                self.db.execute('ROLLBACK')
                raise
            else:
                self.db.execute('COMMIT')

    def worker_auth(self, token: str) -> str:
        with self._lock:
            row = self.db.execute('SELECT id,revoked FROM workers WHERE token_hash=?', (secret_hash(token),)).fetchone()
        if row is None or row['revoked']:
            raise APIError('Worker credentials are invalid or revoked.', status=401)
        return row['id']

    def invite(self, expires_in: float, max_joins: int) -> str:
        secret = secrets.token_urlsafe(32)
        with self.transaction() as db:
            db.execute('INSERT INTO invites(digest,expires,max_joins) VALUES(?,?,?)',
                       (secret_hash(secret), time.time() + expires_in, max_joins))
        return secret

    def enroll(self, secret: str, request_id: str, inventory: dict) -> dict:
        request_id = identifier(request_id)
        digest, inventory_json = secret_hash(secret), _dump(inventory, 64 * 1024)
        with self.transaction() as db:
            previous = db.execute('SELECT * FROM enrollments WHERE request_id=?', (request_id,)).fetchone()
            if previous is not None:
                if previous['invite_digest'] != digest or previous['inventory'] != inventory_json:
                    raise APIError('Enrollment request ID is already bound to another request.', status=409)
                worker_id = previous['worker_id']
                row = db.execute('SELECT revoked FROM workers WHERE id=?', (worker_id,)).fetchone()
                if row['revoked']:
                    raise APIError('This worker enrollment was revoked.', status=401)
            else:
                invitation = db.execute('SELECT * FROM invites WHERE digest=?', (digest,)).fetchone()
                if invitation is None or invitation['expires'] <= time.time() or invitation['joins'] >= invitation['max_joins']:
                    raise APIError('Invitation is invalid, expired, or exhausted.', status=401)
                worker_id = uuid.uuid4().hex
                token = self._worker_token(request_id, digest)
                db.execute('INSERT INTO workers(id,token_hash,inventory,last_seen) VALUES(?,?,?,?)',
                           (worker_id, secret_hash(token), inventory_json, time.time()))
                db.execute('INSERT INTO enrollments VALUES(?,?,?,?)', (request_id, digest, worker_id, inventory_json))
                db.execute('UPDATE invites SET joins=joins+1 WHERE digest=?', (digest,))
        return {'id': worker_id, 'token': self._worker_token(request_id, digest), 'generation': 0}

    def _worker_token(self, request_id: str, digest: str) -> str:
        return hmac.new(self._key, (request_id + ':' + digest).encode(), hashlib.sha256).hexdigest()

    def workers(self) -> list[dict]:
        with self._lock:
            rows = self.db.execute('SELECT * FROM workers ORDER BY id LIMIT 256').fetchall()
        values = []
        for row in rows:
            connected = time.time() - row['last_seen'] < 30
            state = ('removed' if row['revoked'] else 'stopped' if row['stop'] and not row['active_run']
                     else 'offline' if not connected else 'draining' if row['drain'] else 'busy' if row['active_run'] else 'idle')
            values.append({'id': row['id'], 'state': state, 'connected': connected,
                           'draining': bool(row['drain']), 'active_run': row['active_run'],
                           'last_seen': row['last_seen'], 'inventory': _load(row['inventory']),
                           'error': row['error']})
        return values

    def manage(self, worker_id: str, action: str) -> dict:
        worker_id = identifier(worker_id)
        with self.transaction() as db:
            row = db.execute('SELECT id FROM workers WHERE id=?', (worker_id,)).fetchone()
            if row is None:
                raise APIError('Worker does not exist.', status=404)
            if action == 'drain':
                db.execute('UPDATE workers SET drain=1 WHERE id=?', (worker_id,))
            elif action == 'remove':
                db.execute('UPDATE workers SET drain=1,revoked=1 WHERE id=?', (worker_id,))
                # Never transfer ownership. A removed worker may still be computing while offline.
                failure = _dump({'state': 'failed', 'history': None, 'counters': {},
                                 'error': {'type': 'WorkerRemoved', 'message': 'Assigned worker was removed. Its disconnected process may continue until it observes revocation.'}})
                db.execute("UPDATE runs SET state='failed',result=?,updated=? WHERE worker_id=? AND state NOT IN ('completed','failed','cancelled')",
                           (failure, time.time(), worker_id))
            else:
                raise ConfigurationError('Unknown membership operation.')
        return next(worker for worker in self.workers() if worker['id'] == worker_id)

    def stop_all(self) -> dict:
        with self.transaction() as db:
            db.execute('UPDATE workers SET drain=1,stop=1 WHERE revoked=0')
        return {'state': 'shutdown_requested', 'workers': self.workers(), 'graceful': True}

    def heartbeat(self, worker_id: str, inventory: dict | None = None, error: str | None = None) -> dict:
        with self.transaction() as db:
            row = db.execute('SELECT * FROM workers WHERE id=? AND revoked=0', (worker_id,)).fetchone()
            if row is None:
                raise APIError('Worker credentials were revoked.', status=401)
            db.execute('UPDATE workers SET last_seen=?,error=? WHERE id=?', (time.time(), error, worker_id))
            if inventory is not None:
                db.execute('UPDATE workers SET inventory=? WHERE id=?', (_dump(inventory, 64 * 1024), worker_id))
            active = db.execute('SELECT id,state,cancel FROM runs WHERE id=?', (row['active_run'],)).fetchone()
        return {'drain': bool(row['drain']), 'stop': bool(row['stop']),
                'active_run': dict(active) if active is not None else None}

    def archive(self, digest: str, size: int) -> None:
        with self.transaction() as db:
            db.execute('INSERT OR IGNORE INTO archives VALUES(?,?,?)', (digest, size, time.time()))

    def submit(self, payload: dict) -> str:
        spec, archive = payload['spec'], payload['archive']
        run_id, request_id = identifier(spec['id']), identifier(payload['request_id'])
        fit, placement, plan = payload['fit'], payload['placement'], payload['plan']
        # Inventory and derived plans can change between retries. The accepted
        # recipe, archive and budget are immutable; retain the first assignment
        # policy when a submitter retries the same job ID.
        fingerprint = secret_hash(_dump({k: payload[k] for k in ('spec', 'archive', 'fit')}, 9 * 1024 * 1024))
        with self.transaction() as db:
            row = db.execute('SELECT id,fingerprint FROM runs WHERE id=? OR request_id=?', (run_id, request_id)).fetchone()
            if row is not None:
                if row['id'] != run_id or row['fingerprint'] != fingerprint:
                    raise APIError('Submission ID already refers to a different immutable run.', status=409)
                return row['id']
            if db.execute('SELECT digest FROM archives WHERE digest=?', (archive,)).fetchone() is None:
                raise APIError('Upload the content-addressed project archive before submitting.', status=409)
            now = time.time()
            db.execute('''INSERT INTO runs(id,request_id,fingerprint,spec,archive,fit,placement,plan,state,created,updated)
                          VALUES(?,?,?,?,?,?,?,?,?,?,?)''',
                       (run_id, request_id, fingerprint, _dump(spec, 9 * 1024 * 1024), archive,
                        _dump(fit), _dump(placement), _dump(plan), 'queued', now, now))
        return run_id

    def _assignment(self, row: sqlite3.Row) -> dict:
        return {'id': row['id'], 'spec': _load(row['spec']), 'archive': row['archive'],
                'fit': _load(row['fit']), 'device': row['device'], 'generation': row['generation']}

    def claim(self, worker_id: str, known_run: str | None) -> dict:
        with self.transaction() as db:
            worker = db.execute('SELECT * FROM workers WHERE id=? AND revoked=0', (worker_id,)).fetchone()
            if worker is None:
                raise APIError('Worker credentials were revoked.', status=401)
            active = worker['active_run']
            if active:
                row = db.execute('SELECT * FROM runs WHERE id=?', (active,)).fetchone()
                # Recover delivery of an assignment only to its existing persistent owner.
                return {'job': self._assignment(row), 'existing': True}
            if known_run is not None:
                raise APIError('Worker has an unresolved local job; refusing another claim.', status=409)
            if worker['drain'] or worker['stop']:
                return {'job': None}
            inventory = _load(worker['inventory'])
            devices = inventory.get('devices', [])
            for row in db.execute("SELECT * FROM runs WHERE state='queued' ORDER BY created LIMIT 256").fetchall():
                placement = _load(row['placement'])
                if placement.get('worker_id') not in {None, worker_id}:
                    continue
                desired = placement.get('device')
                suitable = [item for item in devices if isinstance(item, dict) and (desired is None or item.get('device') == desired)]
                precision = placement.get('precision', 'auto')
                suitable = [item for item in suitable if not (
                    (precision == 'fp16' and item.get('type') != 'cuda') or
                    (precision == 'bf16' and (item.get('type') == 'mps' or item.get('bf16') is False)))]
                if not suitable:
                    continue
                selected = min(suitable, key=lambda item: {'cuda': 0, 'mps': 1, 'cpu': 2}.get(item.get('type'), 3))
                now = time.time()
                db.execute("UPDATE runs SET state='preparing',worker_id=?,device=?,updated=? WHERE id=? AND state='queued'",
                           (worker_id, selected['device'], now, row['id']))
                db.execute('UPDATE workers SET active_run=? WHERE id=?', (row['id'], worker_id))
                row = db.execute('SELECT * FROM runs WHERE id=?', (row['id'],)).fetchone()
                return {'job': self._assignment(row), 'existing': False}
        return {'job': None}

    def owns(self, worker_id: str, run_id: str) -> sqlite3.Row:
        row = self.db.execute('SELECT * FROM runs WHERE id=? AND worker_id=?', (run_id, worker_id)).fetchone()
        if row is None:
            raise APIError('Run is not assigned to this worker.', status=403)
        return row

    def report(self, worker_id: str, payload: dict) -> dict:
        run_id = identifier(payload['id'])
        with self.transaction() as db:
            row = self.owns(worker_id, run_id)
            for event in payload.get('events', []):
                seq = event.get('seq')
                if type(seq) is not int or seq < 1:
                    raise ConfigurationError('Event sequence must be a positive integer.')
                event_json = _dump(event, MAX_EVENT)
                db.execute('INSERT OR IGNORE INTO events VALUES(?,?,?)', (run_id, seq, event_json))
            db.execute('''DELETE FROM events WHERE run_id=? AND seq NOT IN
                          (SELECT seq FROM events WHERE run_id=? ORDER BY seq DESC LIMIT ?)''',
                       (run_id, run_id, MAX_EVENTS))
            if row['state'] not in TERMINAL:
                result = payload.get('result')
                if result is not None:
                    if result.get('state') not in TERMINAL:
                        raise ConfigurationError('Worker result must be terminal.')
                    db.execute('UPDATE runs SET state=?,result=?,counters=?,updated=? WHERE id=?',
                               (result['state'], _dump(result), _dump(result.get('counters', {})), time.time(), run_id))
                    db.execute('UPDATE workers SET active_run=NULL WHERE id=? AND active_run=?', (worker_id, run_id))
                elif payload.get('state') == 'running':
                    db.execute("UPDATE runs SET state='running',updated=? WHERE id=?", (time.time(), run_id))
                events = payload.get('events', [])
                if events and result is None:
                    db.execute('UPDATE runs SET counters=?,updated=? WHERE id=?',
                               (_dump(events[-1].get('counters', {})), time.time(), run_id))
            maximum = db.execute('SELECT MAX(seq) FROM events WHERE run_id=?', (run_id,)).fetchone()[0] or 0
        return {'ack': maximum, 'terminal': payload.get('result') is not None}

    def get_run(self, run_id: str) -> dict:
        with self._lock:
            row = self.db.execute('SELECT * FROM runs WHERE id=?', (identifier(run_id),)).fetchone()
            if row is None:
                raise APIError('Run does not exist.', status=404)
            worker = self.db.execute('SELECT * FROM workers WHERE id=?', (row['worker_id'],)).fetchone()
        result = dict(_load(row['result'], {}))
        result.update({'id': row['id'], 'state': row['state'], 'worker_id': row['worker_id'],
                       'device': row['device'], 'generation': row['generation'],
                       'created_at': row['created'], 'updated_at': row['updated'],
                       'counters': _load(row['counters'], {}), 'cancel_requested': bool(row['cancel']),
                       'archive_digest': row['archive'], 'placement': _load(row['placement'])})
        if row['state'] == 'queued':
            result['warning'] = 'Waiting for an idle enrolled worker matching the requested device and precision.'
        if worker is not None:
            result['worker_connected'] = time.time() - worker['last_seen'] < 30 and not worker['revoked']
            result['worker_removed'] = bool(worker['revoked'])
            if not result['worker_connected'] and row['state'] not in TERMINAL:
                result['warning'] = 'Assigned worker is disconnected or removed. Ownership is retained; no automatic retry.'
        return result

    def events(self, run_id: str, after: int) -> list[dict]:
        self.get_run(run_id)
        with self._lock:
            rows = self.db.execute('SELECT payload FROM events WHERE run_id=? AND seq>? ORDER BY seq LIMIT 100', (run_id, after)).fetchall()
        return [_load(row['payload']) for row in rows]

    def cancel(self, run_id: str) -> dict:
        with self.transaction() as db:
            row = db.execute('SELECT state FROM runs WHERE id=?', (identifier(run_id),)).fetchone()
            if row is None:
                raise APIError('Run does not exist.', status=404)
            if row['state'] == 'queued':
                db.execute("UPDATE runs SET cancel=1,state='cancelled',updated=?,result=? WHERE id=?",
                           (time.time(), _dump({'state': 'cancelled', 'history': None, 'counters': {}}), run_id))
            elif row['state'] not in TERMINAL:
                db.execute('UPDATE runs SET cancel=1,updated=? WHERE id=?', (time.time(), run_id))
        return self.get_run(run_id)

    def command(self, run_id: str, spec: dict, request_id: str) -> dict:
        with self.transaction() as db:
            row = db.execute('SELECT * FROM runs WHERE id=?', (identifier(run_id),)).fetchone()
            if row is None:
                raise APIError('Run does not exist.', status=404)
            if row['state'] != 'completed':
                raise UnsupportedFeatureError('Manual remote checkpoint/export requires a completed job. Use a checkpoint policy while training.')
            if not _load(row['result'], {}).get('checkpoint'):
                raise UnsupportedFeatureError('This job has no completed internal snapshot.')
            existing = db.execute('SELECT * FROM commands WHERE id=?', (identifier(request_id),)).fetchone()
            encoded = _dump(spec)
            if existing is not None:
                if existing['run_id'] != run_id or existing['spec'] != encoded:
                    raise APIError('Command request ID already belongs to another command.', status=409)
            else:
                db.execute('INSERT INTO commands VALUES(?,?,?,?,?,?,?)',
                           (request_id, run_id, row['worker_id'], encoded, 'queued', None, time.time()))
        return self.command_status(request_id)

    def command_status(self, command_id: str) -> dict:
        with self._lock:
            row = self.db.execute('SELECT * FROM commands WHERE id=?', (identifier(command_id),)).fetchone()
        if row is None:
            raise APIError('Command does not exist.', status=404)
        return {'id': row['id'], 'run_id': row['run_id'], 'state': row['state'], 'result': _load(row['result'])}

    def claim_command(self, worker_id: str) -> dict | None:
        with self.transaction() as db:
            row = db.execute("SELECT * FROM commands WHERE worker_id=? AND state IN ('running','queued') ORDER BY created LIMIT 1", (worker_id,)).fetchone()
            if row is None:
                return None
            db.execute("UPDATE commands SET state='running' WHERE id=?", (row['id'],))
            return {'id': row['id'], 'run_id': row['run_id'], 'spec': _load(row['spec'])}

    def report_command(self, worker_id: str, command_id: str, result: dict) -> dict:
        with self.transaction() as db:
            row = db.execute('SELECT * FROM commands WHERE id=? AND worker_id=?', (identifier(command_id), worker_id)).fetchone()
            if row is None:
                raise APIError('Command is not owned by this worker.', status=403)
            if row['state'] not in {'completed', 'failed'}:
                db.execute('UPDATE commands SET state=?,result=? WHERE id=?',
                           ('failed' if result.get('error') else 'completed', _dump(result), command_id))
        return {'ack': True}
