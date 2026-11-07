"""Bounded wire values, private local state, and control-plane transport."""
from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import ssl
import tempfile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, ProxyHandler, Request, build_opener

from mlzo.errors import ClusterError, ConfigurationError, UnsupportedFeatureError

MAX_JSON = 9 * 1024 * 1024
MAX_ARCHIVE = 100 * 1024 * 1024
MAX_RESULT = 512 * 1024
MAX_EVENT = 16 * 1024
MAX_EVENTS = 1000
TIMEOUT = 10.0
TERMINAL = frozenset({'completed', 'failed', 'cancelled'})
_IDENTIFIER = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z')
_DIGEST = re.compile(r'[0-9a-f]{64}\Z')


class APIError(ClusterError):
    def __init__(self, message: str, *, status: int = 400, code: str = 'ClusterError') -> None:
        super().__init__(message)
        self.status, self.code = status, code


def identifier(value: Any) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value) or value in {'.', '..'}:
        raise ConfigurationError('IDs must be 1-128 portable letters, digits, dots, underscores or hyphens.')
    return value


def digest_value(value: Any) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ConfigurationError('Expected a SHA-256 content digest.')
    return value


def bounded_json(value: Any, limit: int = MAX_JSON) -> bytes:
    try:
        data = json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(',', ':'), sort_keys=True).encode('ascii')
    except (TypeError, ValueError, RecursionError, OverflowError):
        raise ConfigurationError('Control messages must be finite, bounded JSON values.') from None
    if len(data) > limit:
        raise ConfigurationError('Control message exceeds its size limit.')
    return data


def read_json(path: Path, limit: int = MAX_JSON) -> Any:
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ConfigurationError('Local metadata exceeds its size limit.')
    try:
        return json.loads(data, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, UnicodeError, RecursionError):
        raise ConfigurationError('Invalid JSON metadata.') from None


def private_dir(path: Path) -> Path:
    path = path.expanduser().absolute()
    if path.is_symlink():
        raise ConfigurationError('Private state directories must not be symlinks.')
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    return path


def atomic_json(path: Path, value: Any, limit: int = MAX_JSON) -> None:
    data = bounded_json(value, limit)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix='.mlzo-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def mlzo_home(home: str | os.PathLike | None = None) -> Path:
    return private_dir(Path(home or os.environ.get('MLZO_HOME') or Path.home() / '.mlzo'))


def loopback(host: str) -> bool:
    if host.lower() == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def endpoint_url(value: Any) -> str:
    if not isinstance(value, str):
        raise ConfigurationError('A coordinator endpoint is required.')
    try:
        url = urlsplit(value)
        port = url.port
    except ValueError:
        raise ConfigurationError('Invalid coordinator endpoint.') from None
    if (url.scheme not in {'http', 'https'} or not url.hostname or url.username is not None
            or url.password is not None or url.query or url.fragment or url.path not in {'', '/'}
            or port == 0):
        raise ConfigurationError('Endpoint must be an HTTP(S) origin without credentials, query or path.')
    if url.scheme == 'http' and not loopback(url.hostname):
        raise ConfigurationError('Unencrypted HTTP is allowed only on loopback; use verified HTTPS remotely.')
    return value.rstrip('/')


def secret_hash(secret: str) -> str:
    return hashlib.sha256(secret.encode('utf-8')).hexdigest()


def positive_number(value: Any, name: str, maximum: float) -> float:
    if type(value) not in {int, float} or not math.isfinite(value) or not 0 < value <= maximum:
        raise ConfigurationError(f'{name} must be positive and at most {maximum}.')
    return float(value)


def budget(epochs: Any, steps: Any) -> None:
    if (epochs is None) == (steps is None):
        raise ConfigurationError('Remote jobs require exactly one finite epochs= or steps= budget.')
    for value in (epochs, steps):
        if value is not None and (type(value) is not int or not 0 < value <= 10**9):
            raise ConfigurationError('Remote epochs/steps must be positive integers at most 1000000000.')


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


class Transport:
    """No redirects and no certificate-verification bypass; credentials stay private."""
    def __init__(self, endpoint: str, token: str | None = None, ca_file: str | None = None,
                 timeout: float = TIMEOUT) -> None:
        self.endpoint = endpoint_url(endpoint)
        if token is not None and (not isinstance(token, str) or not token or any(c.isspace() for c in token)):
            raise ConfigurationError('A nonempty control-plane token is required.')
        self._token = token
        self.ca_file = str(ca_file) if ca_file is not None else None
        self.timeout = positive_number(timeout, 'timeout', 120)
        context = ssl.create_default_context(cafile=self.ca_file)
        self._opener = build_opener(ProxyHandler({}), _NoRedirect(), HTTPSHandler(context=context))

    def __repr__(self) -> str:
        return f'Transport(endpoint={self.endpoint!r})'

    def request(self, method: str, path: str, value: Any = None, *, raw: bytes | None = None,
                binary: bool = False) -> Any:
        data = raw if raw is not None else (bounded_json(value) if value is not None else None)
        headers = {'Accept': 'application/octet-stream' if binary else 'application/json'}
        if self._token is not None:
            headers['Authorization'] = 'Bearer ' + self._token
        if data is not None:
            headers['Content-Type'] = 'application/octet-stream' if raw is not None else 'application/json'
        request = Request(self.endpoint + path, data=data, headers=headers, method=method)
        limit = MAX_ARCHIVE if binary else MAX_JSON
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                result = response.read(limit + 1)
        except HTTPError as exc:
            body = exc.read(min(MAX_JSON, 64 * 1024) + 1)
            try:
                error = json.loads(body)['error']
                code, message = error['type'], error['message']
                if not isinstance(message, str):
                    raise ValueError()
            except (ValueError, KeyError, TypeError):
                code, message = 'ClusterError', f'Coordinator rejected request with HTTP {exc.code}.'
            if code == 'UnsupportedFeatureError':
                raise UnsupportedFeatureError(message) from None
            if code == 'ConfigurationError':
                raise ConfigurationError(message) from None
            raise APIError(message, status=exc.code, code=code) from None
        except (URLError, OSError, TimeoutError):
            raise ClusterError('Coordinator connection failed; verify reachability and TLS trust.') from None
        if len(result) > limit:
            raise ClusterError('Coordinator response exceeds the transport limit.')
        if binary:
            return result
        try:
            return json.loads(result, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        except (ValueError, UnicodeError, RecursionError):
            raise ClusterError('Coordinator returned invalid JSON.') from None
