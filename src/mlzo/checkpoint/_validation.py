"""Validation shared by policy, references, and the on-disk format."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

from mlzo.errors import CheckpointError, ConfigurationError, UnsupportedFeatureError

MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_DEPTH = 128
SNAPSHOT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def destination(value: str | os.PathLike[str]) -> str:
    try:
        result = os.fspath(value)
    except TypeError:
        raise ConfigurationError("Checkpoint destination must be a path or URI string.") from None
    if not isinstance(result, str) or not result.strip() or "\x00" in result:
        raise ConfigurationError("Checkpoint destination must be a nonempty path or URI string.")
    if "://" in result:
        if result.startswith("s3://"):
            raise UnsupportedFeatureError(
                "S3 checkpoints are not implemented in MLZO 0.1.0a1. "
                "Use a local filesystem path on the execution worker."
            )
        raise UnsupportedFeatureError(
            "Checkpoint URI storage is not implemented in this alpha. Use a local filesystem path."
        )
    return result


def local_path(value: str | os.PathLike[str]) -> Path:
    return Path(destination(value)).expanduser().absolute()


def retention(value: int | None) -> None:
    if value is not None and (type(value) is not int or value <= 0):
        raise ConfigurationError("keep_last must be a positive integer or None to disable retention.")


def snapshot_id(value: Any) -> str:
    if type(value) is not str or SNAPSHOT_ID.fullmatch(value) is None:
        raise CheckpointError("snapshot_id must be a safe, nonempty snapshot identifier.")
    return value


def key_path(path: str, key: Any, index: int) -> str:
    """Describe a key without calling repr on an unsupported object."""
    if type(key) is str:
        label = json.dumps(key[:96], ensure_ascii=True)
        return f"{path}[{label}]"
    if type(key) in (int, bool, float) or key is None:
        # Extremely large integer keys need not be converted to decimal for an error path.
        if type(key) is int and key.bit_length() > 256:
            return f"{path}[<integer key #{index}>]"
        return f"{path}[{key}]"
    return f"{path}[<key #{index}>]"


def json_value(value: Any, *, path: str = "metadata") -> Any:
    """Copy JSON data, reject custom objects, and report their exact container path."""
    active: set[int] = set()

    def visit(item: Any, current: str, depth: int) -> Any:
        if depth > MAX_DEPTH:
            raise CheckpointError(f"{current}: metadata nesting exceeds {MAX_DEPTH} levels.")
        if item is None or type(item) in (str, bool, int):
            return item
        if type(item) is float:
            if not math.isfinite(item):
                raise CheckpointError(f"{current}: metadata numbers must be finite JSON numbers.")
            return item
        if type(item) not in (dict, list, tuple):
            raise CheckpointError(
                f"{current}: metadata supports only JSON scalar values, string-keyed dictionaries, "
                "and lists or tuples. Convert custom objects explicitly."
            )
        identity = id(item)
        if identity in active:
            raise CheckpointError(f"{current}: cyclic metadata cannot be checkpointed.")
        active.add(identity)
        try:
            if type(item) is dict:
                copied = {}
                for index, (key, child) in enumerate(item.items()):
                    child_path = key_path(current, key, index)
                    if type(key) is not str:
                        raise CheckpointError(f"{child_path}: metadata dictionary keys must be strings.")
                    copied[key] = visit(child, child_path, depth + 1)
                return copied
            return [visit(child, f"{current}[{index}]", depth + 1) for index, child in enumerate(item)]
        finally:
            active.remove(identity)

    return visit(value, path, 0)


def metadata_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if type(value) is not dict:
        raise CheckpointError("metadata: expected a string-keyed dictionary of JSON values.")
    copied = json_value(value)
    # Check size and JSON representability now, before creating any snapshot files.
    json_bytes(copied, path="metadata")
    return copied


def json_bytes(value: Any, *, path: str) -> bytes:
    try:
        result = json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise CheckpointError(f"{path}: cannot encode the validated checkpoint JSON data.") from None
    if len(result) > MAX_JSON_BYTES:
        raise CheckpointError(f"{path}: checkpoint JSON exceeds the {MAX_JSON_BYTES}-byte limit.")
    return result


def parse_json(data: bytes, *, path: str) -> dict[str, Any]:
    if len(data) > MAX_JSON_BYTES:
        raise CheckpointError(f"{path}: checkpoint JSON exceeds the size limit.")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CheckpointError(f"{path}: duplicate JSON object keys are invalid.")
            result[key] = value
        return result

    def reject_constant(_: str) -> None:
        raise CheckpointError(f"{path}: nonfinite JSON numbers are invalid.")

    try:
        value = json.loads(data, object_pairs_hook=unique_object, parse_constant=reject_constant)
    except (ValueError, UnicodeError, RecursionError):
        raise CheckpointError(f"{path}: invalid checkpoint JSON.") from None
    if type(value) is not dict:
        raise CheckpointError(f"{path}: expected a JSON object.")
    return value
