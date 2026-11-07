"""Bounded, deterministic source archives for trusted remote Python projects.

Only allowlisted source/configuration files enter a default snapshot. Local
datasets require explicit ``data_paths`` and are stored under ``.mlzo/data``.
Archives contain no host wheels or virtual environments. Extraction validates
the complete archive into private staging before publishing a project directory;
it neither imports the project nor installs its declared dependencies.
"""

from __future__ import annotations

import fnmatch
import gzip
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import tarfile
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable

from mlzo._version import __version__
from mlzo.errors import ClusterError

__all__ = ["ProjectArtifact", "snapshot_project", "extract_project"]

PROJECT_FORMAT = "mlzo.project"
PROJECT_VERSION = 1
MANIFEST_NAME = ".mlzo/manifest.json"
DATA_DIRECTORY = ".mlzo/data"
DEFAULT_MAX_BYTES = 100 * 1024 * 1024
_MAX_FILES = 10_000
_MAX_MANIFEST = 4 * 1024 * 1024
_MAX_DEPENDENCIES = 1024
_DEPENDENCY_FORMAT = "name-version-v1"
_CHUNK = 64 * 1024
_HEX = re.compile(r"[0-9a-f]{64}\Z")
# A deliberately small pip-compatible subset. No URLs, markers, includes,
# editable installs, local paths, command-line options, or index credentials.
_REQUIREMENT = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*"
    r"(?:\[[A-Za-z0-9][A-Za-z0-9._-]*(?:\s*,\s*[A-Za-z0-9][A-Za-z0-9._-]*)*\])?"
    r"(?:\s*(?:===|==|!=|~=|<=|>=|<|>)\s*[A-Za-z0-9][A-Za-z0-9.*+!_-]*"
    r"(?:\s*,\s*(?:===|==|!=|~=|<=|>=|<|>)\s*[A-Za-z0-9][A-Za-z0-9.*+!_-]*)*)?\Z"
)
_SOURCE_SUFFIXES = {
    ".py", ".pyi", ".pyx", ".pxd", ".c", ".cc", ".cpp", ".h", ".hpp",
    ".cu", ".cuh", ".rs", ".md", ".rst", ".toml", ".yaml", ".yml", ".ini", ".cfg",
}
_SOURCE_NAMES = {
    "makefile", "cmakelists.txt", "dockerfile", "pipfile", "readme", "readme.txt",
    "license", "license.txt", "copying", "notice", "notice.txt", ".gitignore",
    ".mlzoignore", ".dockerignore", ".python-version", "manifest.in",
}
_DATA_SUFFIXES = {
    ".pt", ".pth", ".json", ".jsonl", ".jpg", ".jpeg", ".png", ".bmp",
    ".gif", ".tif", ".tiff", ".webp",
}
_SKIP_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules", "__pycache__",
    ".cache", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".nox",
    "build", "dist", "htmlcov", "checkpoints", "checkpoint", "outputs", "output",
    "runs", "wandb", "mlruns", ".mlzo",
}
_ROOT_DATA_DIRS = {"data", "dataset", "datasets", "models", "weights"}
_CREDENTIAL_DIRS = {".aws", ".ssh", ".azure", ".kube", ".gnupg", ".config"}
_SECRET_NAMES = {
    ".env", ".netrc", ".npmrc", ".pypirc", "id_rsa", "id_dsa", "id_ecdsa",
    "id_ed25519", "authorized_keys", "known_hosts",
}
_SECRET_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".crt", ".cer", ".jks", ".keystore"}


@dataclass(frozen=True)
class ProjectArtifact:
    path: Path
    digest: str
    size_bytes: int
    manifest: dict[str, Any]
    data_mapping: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON metadata. The archive bytes remain at ``path``."""
        return {"path": str(self.path), "digest": self.digest, "size_bytes": self.size_bytes,
                "manifest": json.loads(json.dumps(self.manifest)), "data_mapping": dict(self.data_mapping)}


@dataclass(frozen=True)
class _Input:
    local: Path
    relative: str
    kind: str
    signature: tuple[int, ...]


@dataclass(frozen=True)
class _Member:
    name: str
    offset: int
    size: int
    mode: int


def _limit(value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise ClusterError("max_bytes must be a positive integer.")
    return value


def _safe_name(value: Any) -> str:
    if (
        type(value) is not str or not value or "\\" in value or ":" in value
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
        or any(part in {"", ".", ".."} or part.endswith((" ", ".")) for part in value.split("/"))
        or PurePosixPath(value).is_absolute()
    ):
        raise ClusterError(f"Unsafe archive path {value!r}.")
    try:
        # The format is strictly POSIX ustar: no PAX/GNU path or size overrides.
        tarfile.TarInfo(value).tobuf(format=tarfile.USTAR_FORMAT, encoding="utf-8", errors="strict")
    except (ValueError, UnicodeError) as exc:
        raise ClusterError(f"Archive path {value!r} does not fit portable POSIX ustar; shorten it.") from exc
    return value


def _secret(path: PurePosixPath) -> bool:
    parts = tuple(part.lower() for part in path.parts)
    name = parts[-1]
    return (
        any(part in _CREDENTIAL_DIRS for part in parts)
        or name in _SECRET_NAMES or name.startswith(".env.")
        or name.endswith((".env", ".env.local"))
        or PurePosixPath(name).suffix in _SECRET_SUFFIXES
        or any(word in name for word in ("credential", "secret", "private_key", "service_account", "service-account"))
        or name.startswith(("token.", "tokens.", "id_rsa.", "id_ed25519."))
    )


def _source_allowed(relative: str) -> bool:
    path = PurePosixPath(relative)
    name = path.name.lower()
    if _secret(path) or any(
        part.startswith(".") or part.lower() in _SKIP_DIRS or part.lower().endswith(".egg-info")
        for part in path.parts[:-1]
    ):
        return False
    if len(path.parts) > 1 and path.parts[0].lower() in _ROOT_DATA_DIRS:
        # These names also commonly identify importable Python packages.
        # Keep their code without implicitly including dataset/model assets.
        return path.suffix.lower() in {".py", ".pyi"}
    if name.startswith(".") and name not in _SOURCE_NAMES:
        return False
    if path.suffix.lower() in _SOURCE_SUFFIXES or name in _SOURCE_NAMES:
        return True
    if fnmatch.fnmatch(name, "requirements*.txt"):
        return True
    return path.suffix.lower() == ".json" and (
        name in {"config.json", "settings.json", "package.json"} or name.endswith(".config.json")
        or any(part.lower() in {"config", "configs"} for part in path.parts[:-1])
    )


def _signature(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _regular(path: Path) -> os.stat_result:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ClusterError(f"Only regular files are supported: {path}.")
    return info


def _open_regular(path: Path) -> BinaryIO:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ClusterError(f"Only regular files are supported: {path}.")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _ignore_rules(root: Path) -> tuple[str, ...]:
    path = root / ".mlzoignore"
    if not path.exists() or path.is_symlink():
        return ()
    if _regular(path).st_size > 64 * 1024:
        raise ClusterError(".mlzoignore exceeds 64 KiB.")
    with _open_regular(path) as stream:
        content = stream.read(64 * 1024 + 1).decode("utf-8")
    if len(content.encode("utf-8")) > 64 * 1024:
        raise ClusterError(".mlzoignore changed or exceeds 64 KiB.")
    rules = tuple(line.strip() for line in content.splitlines() if line.strip() and not line.lstrip().startswith("#"))
    if len(rules) > 1024 or any(rule.startswith("!") for rule in rules):
        raise ClusterError(".mlzoignore supports up to 1024 exclusion globs; negated re-inclusion rules are unsupported.")
    return rules


def _ignored(relative: str, rules: tuple[str, ...]) -> bool:
    parts = PurePosixPath(relative).parts
    for rule in rules:
        anchored = rule.startswith("/")
        pattern = rule.lstrip("/").rstrip("/")
        if not pattern:
            continue
        if not anchored and "/" not in pattern:
            if any(fnmatch.fnmatchcase(part, pattern) for part in parts):
                return True
        elif fnmatch.fnmatchcase(relative, pattern) or any(
            fnmatch.fnmatchcase("/".join(parts[:index]), pattern) for index in range(1, len(parts))
        ):
            return True
    return False


def _walk(root: Path, *, data: bool, rules: tuple[str, ...] = ()) -> Iterable[Path]:
    seen = 0

    def scan_error(error: OSError) -> None:
        raise ClusterError(f"Cannot scan project directory: {error}") from error

    for current, directories, files in os.walk(root, topdown=True, followlinks=False, onerror=scan_error):
        here = Path(current)
        kept = []
        for name in sorted(directories):
            seen += 1
            candidate = here / name
            relative = candidate.relative_to(root).as_posix()
            if candidate.is_symlink():
                if data:
                    raise ClusterError(f"Declared data contains a symlink: {candidate}.")
                continue
            if (
                name.lower() in _SKIP_DIRS or name.lower() in _CREDENTIAL_DIRS
                or name.startswith(".") or name.lower().endswith(".egg-info")
                or (not data and _ignored(relative, rules))
            ):
                continue
            kept.append(name)
        directories[:] = kept
        for name in sorted(files):
            seen += 1
            candidate = here / name
            relative = candidate.relative_to(root).as_posix()
            if candidate.is_symlink():
                if data:
                    raise ClusterError(f"Declared data contains a symlink: {candidate}.")
                continue
            if not stat.S_ISREG(candidate.lstat().st_mode):
                if data:
                    raise ClusterError(f"Declared data contains a non-regular file: {candidate}.")
                continue
            if data:
                if _secret(PurePosixPath(relative)):
                    raise ClusterError(f"Declared data contains a credential-like file: {candidate}.")
                if candidate.suffix.lower() in _DATA_SUFFIXES and not name.startswith("."):
                    yield candidate
            elif _source_allowed(relative) and not _ignored(relative, rules):
                yield candidate
        if seen > _MAX_FILES * 10:
            raise ClusterError("Project scan exceeds 100000 entries; narrow code= or add .mlzoignore exclusions.")


def _inputs(root: Path, data_paths: tuple[Path, ...], max_bytes: int) -> tuple[list[_Input], dict[str, str]]:
    result: list[_Input] = []
    mapping: dict[str, str] = {}
    names: set[str] = set()
    total = 0

    def add(local: Path, relative: str, kind: str) -> None:
        nonlocal total
        _safe_name(relative)
        folded = relative.casefold()
        if folded in names:
            raise ClusterError(f"Duplicate or case-colliding archive path {relative!r}.")
        names.add(folded)
        info = _regular(local)
        total += info.st_size
        if total > max_bytes or len(result) >= _MAX_FILES:
            raise ClusterError(f"Project exceeds {max_bytes} bytes or {_MAX_FILES} files.")
        result.append(_Input(local, relative, kind, _signature(info)))

    rules = _ignore_rules(root)
    for local in _walk(root, data=False, rules=rules):
        add(local, local.relative_to(root).as_posix(), "source")
    if not result:
        raise ClusterError("Project has no selected source/build/configuration files; choose a Python project directory.")
    for index, original in enumerate(data_paths):
        if original.is_symlink():
            raise ClusterError(f"Declared data paths cannot be symlinks: {original}.")
        if _secret(PurePosixPath(original.as_posix())) or any(
            part.lower() in {".git", ".hg", ".svn", ".venv", "venv", "node_modules"}
            for part in original.parts
        ):
            raise ClusterError(f"Credential-like paths cannot be declared as data: {original}.")
        relative = _safe_name(f"{DATA_DIRECTORY}/{index:04d}/{original.name}")
        mapping[str(original)] = relative
        if original.is_dir():
            count = len(result)
            for local in _walk(original, data=True):
                add(local, f"{relative}/{local.relative_to(original).as_posix()}", "data")
            if len(result) == count:
                raise ClusterError(f"Declared data directory contains no supported data files: {original}.")
        else:
            if original.suffix.lower() not in _DATA_SUFFIXES:
                raise ClusterError(f"Unsupported declared data file {original}; use tensor, JSON/JSONL, or image files.")
            add(original, relative, "data")
    return sorted(result, key=lambda item: item.relative), mapping


def _copy_input(item: _Input, destination: Path, remaining: int) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    checksum = hashlib.sha256()
    size = 0
    with _open_regular(item.local) as source, destination.open("xb") as target:
        if _signature(os.fstat(source.fileno())) != item.signature:
            raise ClusterError(f"Source changed while staging {item.local}; retry the snapshot.")
        while chunk := source.read(_CHUNK):
            size += len(chunk)
            if size > remaining or size > item.signature[3]:
                raise ClusterError(f"Source changed or exceeded the snapshot limit: {item.local}.")
            checksum.update(chunk)
            target.write(chunk)
        if size != item.signature[3] or _signature(os.fstat(source.fileno())) != item.signature:
            raise ClusterError(f"Source changed while staging {item.local}; retry the snapshot.")
    mode = 0o755 if item.signature[2] & 0o111 else 0o644
    return {"path": item.relative, "size_bytes": size, "sha256": checksum.hexdigest(), "mode": mode, "kind": item.kind}


def _requirements(values: Any) -> list[str]:
    if type(values) is not list or len(values) > _MAX_DEPENDENCIES:
        raise ClusterError("Project dependencies must be a list of at most 1024 name/version requirements.")
    for requirement in values:
        if type(requirement) is not str or len(requirement) > 512 or not _REQUIREMENT.fullmatch(requirement):
            raise ClusterError(
                f"Unsupported dependency {requirement!r}. Remote {_DEPENDENCY_FORMAT} dependencies allow "
                "package names, extras, and version constraints only; prepare URLs, markers, local paths, and indexes on the worker."
            )
    return list(values)


def _dependencies(project: Path) -> tuple[list[str], str | None]:
    pyproject = project / "pyproject.toml"
    requirements = project / "requirements.txt"
    if pyproject.is_file():
        if pyproject.stat().st_size > 1024 * 1024:
            raise ClusterError("pyproject.toml exceeds the 1 MiB dependency metadata limit.")
        with pyproject.open("rb") as handle:
            document = tomllib.load(handle)
        project_info = document.get("project", {})
        if not isinstance(project_info, dict):
            raise ClusterError("pyproject.toml [project] must be a table.")
        if "dependencies" in project_info:
            return _requirements(project_info["dependencies"]), "pyproject.toml"
    if requirements.is_file():
        if requirements.stat().st_size > 1024 * 1024:
            raise ClusterError("requirements.txt exceeds the 1 MiB dependency metadata limit.")
        lines = []
        for line in requirements.read_text(encoding="utf-8").splitlines():
            line = line.split(" #", 1)[0].strip()
            if line and not line.startswith("#"):
                lines.append(line)
        return _requirements(lines), "requirements.txt"
    return [], None


class _BoundedWriter:
    def __init__(self, stream: BinaryIO, limit: int) -> None:
        self.stream = stream
        self.limit = limit
        self.size = 0

    def write(self, content: bytes) -> int:
        self.size += len(content)
        if self.size > self.limit:
            raise ClusterError(f"Expanded archive, including tar headers, exceeds {self.limit} bytes.")
        return self.stream.write(content)

    def tell(self) -> int:
        return self.size


def _hash_file(path: Path, max_bytes: int) -> tuple[str, int]:
    checksum = hashlib.sha256()
    size = 0
    with _open_regular(path) as handle:
        while chunk := handle.read(_CHUNK):
            size += len(chunk)
            if size > max_bytes:
                raise ClusterError(f"Archive exceeds {max_bytes} bytes.")
            checksum.update(chunk)
    return checksum.hexdigest(), size


def _archive(project: Path, files: list[dict], destination: Path, max_bytes: int) -> None:
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as zipped:
            bounded = _BoundedWriter(zipped, max_bytes)
            with tarfile.open(fileobj=bounded, mode="w|", format=tarfile.USTAR_FORMAT, encoding="utf-8", errors="strict") as archive:
                entries = {entry["path"]: entry for entry in files}
                entries[MANIFEST_NAME] = {"size_bytes": (project / MANIFEST_NAME).stat().st_size, "mode": 0o644}
                for relative, entry in sorted(entries.items()):
                    info = tarfile.TarInfo(relative)
                    info.size = entry["size_bytes"]
                    info.mode = entry["mode"]
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    with (project / relative).open("rb") as stream:
                        archive.addfile(info, stream)
        raw.flush()
        os.fsync(raw.fileno())


def snapshot_project(
    path: str | os.PathLike[str], *, destination: str | os.PathLike[str] | None = None,
    data_paths: Iterable[str | os.PathLike[str]] = (), max_bytes: int = DEFAULT_MAX_BYTES,
) -> ProjectArtifact:
    """Stage a stable source snapshot and atomically publish a read-only tar.gz.

    ``destination`` may be an archive filename ending in ``.tar.gz`` or a
    directory. By default a private system temp directory holds the archive.
    An existing different artifact is never overwritten.

    .mlzoignore supports exclusion globs and comments, without re-inclusion.
    Explicit data directories include only tensor, JSON/JSONL, and image files.
    Credentials, links, environments, outputs, and host wheels are excluded.
    """
    limit = _limit(max_bytes)
    try:
        root = Path(path).expanduser().absolute()
        if root.is_symlink() or not root.is_dir():
            raise ClusterError("snapshot_project requires a real project directory, not a symlink.")
        if isinstance(data_paths, (str, bytes, os.PathLike)):
            raise ClusterError("data_paths must be an iterable of explicitly declared files/directories.")
        declared = tuple(sorted({Path(item).expanduser().absolute() for item in data_paths}, key=str))
        inputs, mapping = _inputs(root, declared, limit)
        with tempfile.TemporaryDirectory(prefix=".mlzo-stage-") as temporary:
            staging = Path(temporary) / "project"
            staging.mkdir()
            files = []
            total = 0
            for item in inputs:
                entry = _copy_input(item, staging / item.relative, limit - total)
                files.append(entry)
                total += entry["size_bytes"]
            current, current_mapping = _inputs(root, declared, limit)
            if current != inputs or current_mapping != mapping:
                raise ClusterError("Project changed while staging; retry the snapshot from a stable source tree.")
            dependencies, dependency_source = _dependencies(staging)
            manifest = {
                "format": PROJECT_FORMAT, "version": PROJECT_VERSION, "mlzo_version": __version__,
                "runtime": {"python": platform.python_version(), "implementation": platform.python_implementation(),
                            "platform": platform.system(), "machine": platform.machine()},
                "files": files, "size_bytes": total, "data_mapping": mapping,
                "dependency_format": _DEPENDENCY_FORMAT, "dependency_source": dependency_source,
                "dependencies": dependencies,
            }
            encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
            if len(encoded) > _MAX_MANIFEST or total + len(encoded) > limit:
                raise ClusterError("Project manifest or expanded content exceeds the snapshot limit.")
            (staging / MANIFEST_NAME).parent.mkdir(parents=True, exist_ok=True)
            (staging / MANIFEST_NAME).write_bytes(encoded)
            bundle = Path(temporary) / "project.tar.gz"
            _archive(staging, files, bundle, limit)
            digest, size = _hash_file(bundle, limit)
            selected = Path(destination).expanduser().absolute() if destination is not None else Path(tempfile.mkdtemp(prefix="mlzo-artifacts-"))
            if selected.is_symlink():
                raise ClusterError("Artifact destination cannot be a symlink.")
            target = selected / f"{digest}.tar.gz" if selected.is_dir() or not selected.name.endswith(".tar.gz") else selected
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            descriptor, incoming = tempfile.mkstemp(prefix=".mlzo-artifact-", suffix=".tmp", dir=target.parent)
            incoming_path = Path(incoming)
            try:
                with os.fdopen(descriptor, "wb") as output, bundle.open("rb") as source:
                    shutil.copyfileobj(source, output, _CHUNK)
                    output.flush()
                    os.fsync(output.fileno())
                incoming_path.chmod(0o444)
                try:
                    os.link(incoming_path, target)
                except FileExistsError:
                    if target.is_symlink() or _hash_file(target, limit) != (digest, size):
                        raise ClusterError(f"Artifact destination already contains different content: {target}.")
            finally:
                incoming_path.unlink(missing_ok=True)
            return ProjectArtifact(target, digest, size, manifest, dict(mapping))
    except ClusterError:
        raise
    except (OSError, ValueError, TypeError, UnicodeError, tarfile.TarError) as exc:
        raise ClusterError(f"Cannot snapshot project: {exc}") from exc


def _stage_archive(archive: Path, staging: Path, expected: str | None, limit: int) -> Path:
    if expected is not None and (type(expected) is not str or not _HEX.fullmatch(expected)):
        raise ClusterError("Expected artifact digest must be a lowercase SHA-256 hex string.")
    copied = staging / "input.tar.gz"
    checksum = hashlib.sha256()
    size = 0
    with _open_regular(archive) as source, copied.open("xb") as output:
        while chunk := source.read(_CHUNK):
            size += len(chunk)
            if size > limit:
                raise ClusterError(f"Compressed archive exceeds {limit} bytes.")
            checksum.update(chunk)
            output.write(chunk)
    if expected is not None and checksum.hexdigest() != expected:
        raise ClusterError("Project archive SHA-256 digest mismatch.")
    unpacked = staging / "input.tar"
    total = 0
    # Gzip CRC/truncation checks run to EOF. Concatenated streams cannot bypass
    # the expanded-byte bound or the strict tar trailer checks below.
    with gzip.open(copied, "rb") as source, unpacked.open("xb") as target:
        while chunk := source.read(min(_CHUNK, limit - total + 1)):
            total += len(chunk)
            if total > limit:
                raise ClusterError(f"Expanded archive exceeds {limit} bytes.")
            target.write(chunk)
    return unpacked


def _members(archive: Path, limit: int) -> list[_Member]:
    members = []
    names: set[str] = set()
    size = archive.stat().st_size
    if size > limit or size % 512:
        raise ClusterError("Invalid or oversized POSIX tar stream.")
    with archive.open("rb") as source:
        while True:
            header = source.read(512)
            if len(header) != 512:
                raise ClusterError("Project tar is truncated or lacks its end marker.")
            if header == bytes(512):
                if source.read(512) != bytes(512):
                    raise ClusterError("Project tar requires two zero end-of-archive blocks.")
                while remainder := source.read(_CHUNK):
                    if any(remainder):
                        raise ClusterError("Project tar has trailing entries or nonzero data after its end marker.")
                return members
            try:
                info = tarfile.TarInfo.frombuf(header, encoding="utf-8", errors="strict")
            except (tarfile.HeaderError, UnicodeError) as exc:
                raise ClusterError("Invalid project tar header or checksum.") from exc
            if header[257:263] != b"ustar\x00" or header[263:265] != b"00":
                raise ClusterError("Only MLZO POSIX ustar archives are supported; zip, GNU, and PAX formats are unsupported.")
            name = _safe_name(info.name)
            folded = name.casefold()
            if folded in names:
                raise ClusterError(f"Duplicate archive member {name!r}.")
            names.add(folded)
            if len(members) >= _MAX_FILES + 1:
                raise ClusterError("Project archive contains too many entries.")
            if info.type not in {tarfile.REGTYPE, tarfile.AREGTYPE} or info.linkname:
                raise ClusterError(f"Archive member {name!r} is not a regular file; links and special entries are forbidden.")
            if info.size < 0 or info.mode not in {0o644, 0o755}:
                raise ClusterError(f"Invalid archive size or permissions for {name!r}.")
            offset = source.tell()
            padded = (info.size + 511) // 512 * 512
            if offset + padded > size:
                raise ClusterError(f"Truncated or oversized archive member {name!r}.")
            members.append(_Member(name, offset, info.size, info.mode))
            source.seek(offset + info.size)
            if any(source.read(padded - info.size)):
                raise ClusterError(f"Nonzero tar padding for {name!r}.")


def _json_object(pairs: list[tuple[str, Any]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ClusterError(f"Duplicate manifest key {key!r}.")
        result[key] = value
    return result


def _manifest(archive: Path, members: list[_Member], limit: int) -> dict[str, Any]:
    by_name = {member.name: member for member in members}
    entry = by_name.get(MANIFEST_NAME)
    if entry is None or entry.size > _MAX_MANIFEST:
        raise ClusterError("Project archive needs a bounded .mlzo/manifest.json.")
    with archive.open("rb") as source:
        source.seek(entry.offset)
        manifest = json.loads(source.read(entry.size), object_pairs_hook=_json_object)
    expected = {"format", "version", "mlzo_version", "runtime", "files", "size_bytes", "data_mapping",
                "dependency_format", "dependency_source", "dependencies"}
    if type(manifest) is not dict or set(manifest) != expected:
        raise ClusterError("Invalid project manifest fields.")
    if manifest["format"] != PROJECT_FORMAT or type(manifest["version"]) is not int or manifest["version"] != PROJECT_VERSION:
        raise ClusterError("Unsupported project manifest format/version.")
    if type(manifest["mlzo_version"]) is not str or len(manifest["mlzo_version"]) > 128:
        raise ClusterError("Invalid package version metadata.")
    runtime = manifest["runtime"]
    if type(runtime) is not dict or set(runtime) != {"python", "implementation", "platform", "machine"}:
        raise ClusterError("Invalid project runtime metadata.")
    if any(type(value) is not str or len(value) > 128 for value in runtime.values()):
        raise ClusterError("Invalid project runtime metadata values.")
    if manifest["dependency_format"] != _DEPENDENCY_FORMAT:
        raise ClusterError("Unsupported dependency declaration format.")
    _requirements(manifest["dependencies"])
    dependency_source = manifest["dependency_source"]
    if dependency_source is not None and dependency_source not in {"pyproject.toml", "requirements.txt"}:
        raise ClusterError("Invalid dependency source.")
    if dependency_source is None and manifest["dependencies"]:
        raise ClusterError("Dependencies require an allowlisted project declaration.")
    if type(manifest["files"]) is not list or len(manifest["files"]) > _MAX_FILES:
        raise ClusterError("Invalid project file inventory.")
    listed: set[str] = set()
    total = 0
    for item in manifest["files"]:
        if type(item) is not dict or set(item) != {"path", "size_bytes", "sha256", "mode", "kind"}:
            raise ClusterError("Invalid project file metadata.")
        name = _safe_name(item["path"])
        if name in listed or name == MANIFEST_NAME:
            raise ClusterError(f"Duplicate manifest file {name!r}.")
        listed.add(name)
        actual = by_name.get(name)
        if actual is None or type(item["size_bytes"]) is not int or item["size_bytes"] != actual.size:
            raise ClusterError(f"Missing file or size mismatch for {name!r}.")
        if type(item["mode"]) is not int or item["mode"] != actual.mode:
            raise ClusterError(f"Permission mismatch for {name!r}.")
        if type(item["sha256"]) is not str or not _HEX.fullmatch(item["sha256"]):
            raise ClusterError(f"Invalid file checksum for {name!r}.")
        if name.startswith(DATA_DIRECTORY + "/"):
            if item["kind"] != "data" or PurePosixPath(name).suffix.lower() not in _DATA_SUFFIXES or _secret(PurePosixPath(name)):
                raise ClusterError(f"Invalid declared data entry {name!r}.")
        elif item["kind"] != "source" or not _source_allowed(name):
            raise ClusterError(f"Entry {name!r} is not allowlisted project source.")
        total += actual.size
    if listed | {MANIFEST_NAME} != set(by_name):
        raise ClusterError("Archive entries do not match the project manifest inventory.")
    if type(manifest["size_bytes"]) is not int or manifest["size_bytes"] != total or total + entry.size > limit:
        raise ClusterError("Project manifest total size mismatch or expanded-byte limit exceeded.")
    mapping = manifest["data_mapping"]
    if type(mapping) is not dict or len(mapping) > _MAX_FILES:
        raise ClusterError("Invalid project data mapping.")
    mapped: set[str] = set()
    for original, relative in mapping.items():
        if type(original) is not str or not original.startswith("/") or "\x00" in original:
            raise ClusterError("Original data mapping paths must be absolute paths.")
        relative = _safe_name(relative)
        if not relative.startswith(DATA_DIRECTORY + "/") or relative in mapped:
            raise ClusterError("Invalid or duplicate extracted data mapping path.")
        mapped.add(relative)
        if not any(name == relative or name.startswith(relative + "/") for name in listed):
            raise ClusterError(f"Data mapping {relative!r} has no corresponding files.")
    for item in manifest["files"]:
        if item["kind"] == "data" and not any(item["path"] == root or item["path"].startswith(root + "/") for root in mapped):
            raise ClusterError(f"Data entry {item['path']!r} is not covered by an explicit declaration.")
    return manifest


def extract_project(
    archive: str | os.PathLike[str] | ProjectArtifact, destination: str | os.PathLike[str], *,
    digest: str | None = None, max_bytes: int = DEFAULT_MAX_BYTES,
) -> dict[str, Any]:
    """Verify all archive metadata and file hashes, then publish the project.

    The destination must be absent or an empty directory. ``max_bytes`` bounds
    both compressed bytes and the complete expanded tar stream. Member headers,
    rather than manifest-provided paths, determine staging writes. Archives are
    trusted user source code; checksums establish integrity, not authorship.
    """
    limit = _limit(max_bytes)
    try:
        if isinstance(archive, ProjectArtifact):
            digest = archive.digest if digest is None else digest
            archive = archive.path
        source = Path(archive).expanduser().absolute()
        target = Path(destination).expanduser().absolute()
        if target.is_symlink() or (target.exists() and (not target.is_dir() or any(target.iterdir()))):
            raise ClusterError("Extraction destination must be absent or an empty real directory.")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".mlzo-extract-", dir=target.parent) as temporary:
            staging = Path(temporary)
            unpacked = _stage_archive(source, staging, digest, limit)
            members = _members(unpacked, limit)
            manifest = _manifest(unpacked, members, limit)
            project = staging / "project"
            project.mkdir()
            checksums = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
            with unpacked.open("rb") as stream:
                for member in members:
                    # Only the independently checked header name is used here.
                    path = project.joinpath(*PurePosixPath(member.name).parts)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    checksum = hashlib.sha256()
                    stream.seek(member.offset)
                    remaining = member.size
                    with path.open("xb") as output:
                        while remaining:
                            chunk = stream.read(min(_CHUNK, remaining))
                            if not chunk:
                                raise ClusterError(f"Truncated member {member.name!r}.")
                            remaining -= len(chunk)
                            checksum.update(chunk)
                            output.write(chunk)
                    if member.name != MANIFEST_NAME and checksum.hexdigest() != checksums[member.name]:
                        raise ClusterError(f"Project file SHA-256 mismatch for {member.name!r}.")
                    path.chmod(member.mode)
            dependencies, dependency_source = _dependencies(project)
            if dependencies != manifest["dependencies"] or dependency_source != manifest["dependency_source"]:
                raise ClusterError("Project dependency metadata differs from its source declaration.")
            if target.is_symlink():
                raise ClusterError("Extraction destination became a symlink.")
            removed_empty = False
            if target.exists():
                # rmdir refuses a directory populated since the initial check.
                target.rmdir()
                removed_empty = True
            try:
                project.rename(target)
            except BaseException:
                if removed_empty and not target.exists():
                    target.mkdir()
                raise
            return manifest
    except ClusterError:
        raise
    except (OSError, ValueError, TypeError, UnicodeError, EOFError, RecursionError, tarfile.TarError) as exc:
        raise ClusterError(f"Cannot verify/extract project archive: {exc}") from exc
