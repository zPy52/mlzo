"""Versioned local inference artifacts with restricted weights and recipes.

Committed directories are immutable through this API. Publication uses a sibling
staging directory and an exclusive lock for cooperating local writers. A crashed
writer may leave a staging directory or lock; remove those only after confirming
the writer has stopped. The destination filesystem must support atomic rename.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import stat
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import torch
from torch import nn

from mlzo._version import __version__
from mlzo.errors import ConfigurationError, UnsupportedFeatureError
from mlzo.model import FactoryRef, Lazy, get_recipe, materialize, module_paths

from ._recipe import decode_arguments, encode_recipe
from ._state import state_dict_copy

FORMAT = "mlzo.inference"
FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
WEIGHTS_NAME = "weights.pt"
MAX_JSON_BYTES = 1024 * 1024
CHUNK_BYTES = 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _local_path(value: Any) -> Path:
    try:
        path = os.fspath(value)
    except TypeError as exc:
        raise ConfigurationError("An inference artifact path must be a local string or PathLike.") from exc
    if type(path) is not str or not path.strip() or "\x00" in path:
        raise ConfigurationError("An inference artifact path must be a nonempty local string.")
    if "://" in path:
        raise UnsupportedFeatureError("Inference artifacts support local filesystem paths only in MLZO 0.1.0a1.")
    return Path(os.path.abspath(os.path.expanduser(path)))


def _json_bytes(value: Any) -> bytes:
    try:
        data = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ConfigurationError("Inference metadata must contain finite JSON values with no cycles.") from exc
    if len(data) > MAX_JSON_BYTES:
        raise ConfigurationError("Inference manifest/metadata exceeds the 1 MiB limit.")
    return data


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """A local artifact directory and a copy of its informational manifest.

    Loading always validates the on-disk manifest and weights. When present,
    metadata's artifact_id also pins the reference to that artifact identity.
    """

    path: str | os.PathLike[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", str(_local_path(self.path)))
        if type(self.metadata) is not dict:
            raise ConfigurationError("ArtifactRef metadata must be a JSON dictionary.")
        object.__setattr__(self, "metadata", json.loads(_json_bytes(self.metadata)))

    @property
    def artifact_id(self) -> str | None:
        return self.metadata.get("artifact_id")

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "metadata": json.loads(_json_bytes(self.metadata))}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactRef:
        if not isinstance(value, Mapping) or "path" not in value or set(value) - {"path", "metadata"}:
            raise ConfigurationError("An ArtifactRef needs path and optional metadata fields.")
        return cls(**dict(value))


def _reference(value: Any) -> ArtifactRef:
    if isinstance(value, ArtifactRef):
        return value
    if isinstance(value, Mapping):
        return ArtifactRef.from_dict(value)
    return ArtifactRef(value)


def _block_path(value: Any) -> str:
    if type(value) is not str or not value or any(not part for part in value.split(".")) or "/" in value or "\\" in value:
        raise ConfigurationError("block must be a nonempty canonical registered-module path, such as 'encoder.layers.0'.")
    return value


def _select(model: nn.Module, block: str | None) -> nn.Module:
    if not isinstance(model, nn.Module):
        raise ConfigurationError("export_model() requires a torch.nn.Module.")
    selected = model
    if block is not None:
        for part in _block_path(block).split("."):
            if isinstance(selected, Lazy) and selected.state != "ready":
                raise ConfigurationError("Export never constructs Lazy modules. Materialize the model before selecting a block.")
            child = selected._modules.get(part)
            if child is None:
                raise ConfigurationError(f"No registered module at canonical block path {block!r}.")
            selected = child
    for path, node in module_paths(selected):
        if isinstance(node, Lazy) and node.state != "ready":
            raise ConfigurationError(
                f"Cannot export unresolved Lazy module at {path or '<root>'!r}; "
                "materialize it through training or mlzo.materialize() first."
            )
    return selected


def _artifact_id(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(_json_bytes({key: value for key, value in manifest.items() if key != "artifact_id"})).hexdigest()


def _validate_manifest(value: Any) -> dict[str, Any]:
    if type(value) is not dict or value.get("format") != FORMAT:
        raise ConfigurationError("Not an MLZO inference artifact. Training checkpoints cannot be loaded for inference directly.")
    if type(value.get("format_version")) is not int or value["format_version"] != FORMAT_VERSION:
        raise UnsupportedFeatureError("Unsupported MLZO inference manifest version.")
    fields = {"format", "format_version", "artifact_id", "kind", "block", "reconstruction", "recipe", "versions", "weights"}
    if set(value) != fields:
        raise ConfigurationError("Inference manifest has missing or unknown fields.")
    if value["kind"] == "model":
        if value["block"] is not None:
            raise ConfigurationError("A model artifact must not declare a block path.")
    elif value["kind"] == "block":
        _block_path(value["block"])
    else:
        raise ConfigurationError("Inference artifact kind must be 'model' or 'block'.")
    if value["reconstruction"] == "recipe":
        decode_arguments(value["recipe"])
    elif value["reconstruction"] != "weights_only" or value["recipe"] is not None:
        raise ConfigurationError("Inference reconstruction must declare a recipe or explicit weights_only mode.")
    versions = value["versions"]
    if type(versions) is not dict or set(versions) != {"mlzo", "python", "torch"} or any(type(item) is not str or not item for item in versions.values()):
        raise ConfigurationError("Inference manifest must identify MLZO, Python, and PyTorch versions.")
    weights = value["weights"]
    if type(weights) is not dict or set(weights) != {"file", "format", "sha256", "size_bytes"}:
        raise ConfigurationError("Inference manifest must describe one state_dict weights file.")
    if weights["file"] != WEIGHTS_NAME or weights["format"] != "torch.state_dict":
        raise ConfigurationError("Inference weights must use the restricted weights.pt state_dict format.")
    if type(weights["size_bytes"]) is not int or weights["size_bytes"] <= 0:
        raise ConfigurationError("Invalid inference weights size_bytes.")
    for name, digest in (("weights SHA256", weights["sha256"]), ("artifact_id", value["artifact_id"])):
        if type(digest) is not str or _SHA256.fullmatch(digest) is None:
            raise ConfigurationError(f"Invalid inference {name}.")
    if value["artifact_id"] != _artifact_id(value):
        raise ConfigurationError("Inference manifest does not match its artifact_id.")
    return value


@contextmanager
def _open_regular(path: Path) -> Iterator[BinaryIO]:
    if path.is_symlink():
        raise ConfigurationError(f"Inference artifact files must not be symlinks: {path}.")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ConfigurationError(f"Expected a regular inference artifact file: {path}.")
        stream = os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise
    with stream:
        yield stream


def _read_manifest(reference: ArtifactRef) -> dict[str, Any]:
    root = Path(reference.path)
    if root.is_symlink() or not root.is_dir():
        raise ConfigurationError(f"Expected a committed local inference artifact directory: {root}.")

    def pairs(entries: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in entries:
            if key in result:
                raise ConfigurationError(f"Duplicate JSON field {key!r} in inference manifest.")
            result[key] = value
        return result

    try:
        with _open_regular(root / MANIFEST_NAME) as stream:
            if os.fstat(stream.fileno()).st_size > MAX_JSON_BYTES:
                raise ConfigurationError("Inference manifest exceeds the 1 MiB limit.")
            data = stream.read(MAX_JSON_BYTES + 1)
        if len(data) > MAX_JSON_BYTES:
            raise ConfigurationError("Inference manifest exceeds the 1 MiB limit.")
        manifest = _validate_manifest(json.loads(data, object_pairs_hook=pairs))
    except (OSError, ValueError, RecursionError) as exc:
        if isinstance(exc, ConfigurationError):
            raise
        raise ConfigurationError(f"Cannot read a valid inference manifest at {root}.") from exc
    if reference.artifact_id is not None and reference.artifact_id != manifest["artifact_id"]:
        raise ConfigurationError("ArtifactRef identity does not match the inference manifest.")
    return manifest


@contextmanager
def _verified_weights(root: Path, weights: dict[str, Any]) -> Iterator[BinaryIO]:
    # Load precisely the bytes we verified, even if the original file changes.
    with tempfile.SpooledTemporaryFile(max_size=8 * CHUNK_BYTES, mode="w+b") as verified:
        digest = hashlib.sha256()
        size = 0
        with _open_regular(root / WEIGHTS_NAME) as source:
            if os.fstat(source.fileno()).st_size != weights["size_bytes"]:
                raise ConfigurationError("Inference weights size does not match the manifest.")
            while chunk := source.read(min(CHUNK_BYTES, weights["size_bytes"] - size + 1)):
                size += len(chunk)
                if size > weights["size_bytes"]:
                    raise ConfigurationError("Inference weights grew while reading.")
                digest.update(chunk)
                verified.write(chunk)
        if size != weights["size_bytes"] or digest.hexdigest() != weights["sha256"]:
            raise ConfigurationError("Inference weights SHA256/size verification failed; refusing to load corrupted weights.")
        verified.seek(0)
        yield verified


def _fsync_directory(path: Path) -> None:
    if os.name == "posix":
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


@contextmanager
def _writer(destination: Path) -> Iterator[None]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.parent / f".{destination.name}.mlzo-export.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ConfigurationError(f"Inference destination is locked by a writer: {lock}. Confirm the writer stopped before removing a stale lock.") from exc
    identity = os.fstat(descriptor)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write(f"MLZO local inference export; pid={os.getpid()}\n")
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        current = lock.lstat()
        if (current.st_dev, current.st_ino) != (identity.st_dev, identity.st_ino):
            raise ConfigurationError("Inference writer lock was replaced; refusing to remove another writer's lock.")
        lock.unlink()


def export_model(model: nn.Module, *, to: Any, recipe: Any = None, block: str | None = None) -> ArtifactRef:
    """Export trained state to an immutable local inference directory.

    The destination names the artifact itself. An identical existing artifact
    can be reused; a different or invalid destination is never overwritten.
    Block selection uses only that block's recipe, ignoring the root recipe.
    A known recipe that is non-importable or has unsupported arguments fails
    explicitly. An eager model without a recipe is labelled weights_only and
    needs a zero-argument factory at load time. Export never constructs Lazy
    modules, changes training mode, or serializes optimizer state.
    """
    selected = _select(model, block)
    source_recipe = get_recipe(selected) if block is not None or recipe is None else recipe
    encoded = None if source_recipe is None else encode_recipe(source_recipe)
    if encoded is not None:
        decode_arguments(encoded)
    state = state_dict_copy(selected.state_dict(), freeze=True)
    destination = _local_path(to)
    staging: Path | None = None
    try:
        with _writer(destination):
            try:
                staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
                with (staging / WEIGHTS_NAME).open("xb") as stream:
                    torch.save(state, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                digest = hashlib.sha256()
                with (staging / WEIGHTS_NAME).open("rb") as stream:
                    while chunk := stream.read(CHUNK_BYTES):
                        digest.update(chunk)
                manifest = {
                    "format": FORMAT,
                    "format_version": FORMAT_VERSION,
                    "kind": "block" if block is not None else "model",
                    "block": block,
                    "reconstruction": "weights_only" if encoded is None else "recipe",
                    "recipe": encoded,
                    "versions": {"mlzo": __version__, "python": platform.python_version(), "torch": str(torch.__version__)},
                    "weights": {"file": WEIGHTS_NAME, "format": "torch.state_dict", "sha256": digest.hexdigest(), "size_bytes": (staging / WEIGHTS_NAME).stat().st_size},
                }
                manifest["artifact_id"] = _artifact_id(manifest)
                _validate_manifest(manifest)
                with (staging / MANIFEST_NAME).open("xb") as stream:
                    stream.write(_json_bytes(manifest))
                    stream.flush()
                    os.fsync(stream.fileno())
                if destination.exists() or destination.is_symlink():
                    existing = _read_manifest(ArtifactRef(destination))
                    if existing != manifest:
                        raise ConfigurationError(f"Inference destination already contains a different artifact: {destination}. Choose a new path.")
                    with _verified_weights(destination, existing["weights"]):
                        pass
                    return ArtifactRef(destination, existing)
                _fsync_directory(staging)
                os.rename(staging, destination)
                staging = None
                _fsync_directory(destination.parent)
                return ArtifactRef(destination, manifest)
            finally:
                if staging is not None:
                    shutil.rmtree(staging)
    except OSError as exc:
        raise ConfigurationError(f"Cannot publish local inference artifact at {destination}: {exc}") from exc


def _load_artifact(reference: Any, *, device: str | torch.device, factory: Any) -> tuple[nn.Module, ArtifactRef]:
    ref = _reference(reference)
    manifest = _read_manifest(ref)
    try:
        with _verified_weights(Path(ref.path), manifest["weights"]) as stream:
            try:
                state = torch.load(stream, map_location="cpu", weights_only=True)
            except Exception as exc:
                raise ConfigurationError(
                    "Restricted inference weights loading failed. Use a compatible PyTorch version "
                    "and a state_dict artifact; unsafe pickle loading is never attempted."
                ) from exc
    except OSError as exc:
        raise ConfigurationError(f"Cannot read inference weights at {ref.path}.") from exc
    state = state_dict_copy(state, freeze=False)
    if factory is not None:
        constructor = factory.resolve() if isinstance(factory, FactoryRef) else factory
        args, kwargs = (), {}
    elif manifest["recipe"] is not None:
        args, kwargs = decode_arguments(manifest["recipe"])
        constructor = FactoryRef.from_dict(manifest["recipe"]["factory"]).resolve()
    else:
        raise ConfigurationError("This is a weights-only inference artifact. Supply factory= with a zero-argument callable returning the matching torch.nn.Module or Lazy.")
    if not callable(constructor) or isinstance(constructor, nn.Module):
        raise ConfigurationError("factory must be a constructor callable, not an existing model instance.")
    try:
        model = constructor(*args, **kwargs)
        if not isinstance(model, nn.Module):
            raise ConfigurationError("The inference factory must return a torch.nn.Module or Lazy module.")
        model = materialize(model, device)
        expected_keys = set(model.state_dict())
        if set(state) != expected_keys:
            missing = sorted(expected_keys - set(state))
            unexpected = sorted(set(state) - expected_keys)
            raise ConfigurationError(f"Inference state_dict keys do not match the factory model. Missing: {missing}; unexpected: {unexpected}.")
        model.load_state_dict(state, strict=True)
        model.eval()
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(f"Cannot reconstruct inference model on {device!s}: {exc}") from exc
    return model, ArtifactRef(ref.path, manifest)


def load_artifact(reference: Any, *, device: str | torch.device = "cpu", factory: Any = None) -> nn.Module:
    """Verify an inference artifact and return its materialized model in eval mode.

    ``factory`` overrides a stored recipe and is called with no arguments. It
    can return a Lazy model; materialize resolves it on the requested device.
    Stored recipes import trusted user code, but weights always use the
    restricted ``torch.load(weights_only=True)`` loader with strict model keys.
    Training checkpoints and incomplete export directories are rejected.
    """
    return _load_artifact(reference, device=device, factory=factory)[0]
