"""Device metadata queried only when planning is explicitly requested."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from mlzo.errors import ConfigurationError


def device_name(value: Any) -> str:
    if not isinstance(value, (str, torch.device)):
        raise ConfigurationError("device must be 'cpu', 'mps', 'cuda', or 'cuda:<logical GPU ID>'.")
    try:
        device = torch.device(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ConfigurationError(f"Invalid device {value!r}; use cpu, mps, or cuda:<logical GPU ID>.") from exc
    if device.type not in {"cpu", "mps", "cuda"}:
        raise ConfigurationError(f"Device type {device.type!r} is unsupported; use cpu, mps, or cuda.")
    if device.type != "cuda" and device.index not in {None, 0}:
        raise ConfigurationError(f"Indexed {device.type} devices are unsupported; use {device.type!r}.")
    return str(device) if device.type == "cuda" else device.type


def local_inventory() -> tuple[list[dict[str, Any]], list[str]]:
    devices: list[dict[str, Any]] = [{"device": "cpu", "type": "cpu", "bf16": None}]
    notes: list[str] = []
    try:
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                devices.append({
                    "device": f"cuda:{index}", "type": "cuda", "index": index,
                    "logical_gpu": index, "name": props.name,
                    "total_memory_bytes": props.total_memory,
                    "compute_capability": [props.major, props.minor],
                    "bf16": props.major >= 8 and int((torch.version.cuda or "0").split(".")[0]) >= 11,
                })
    except (RuntimeError, OSError, AssertionError) as exc:
        notes.append(f"CUDA inventory could not be fully queried: {exc}")
    mps = getattr(torch.backends, "mps", None)
    try:
        if mps is not None and mps.is_available():
            devices.append({"device": "mps", "type": "mps", "bf16": False})
    except (RuntimeError, OSError) as exc:
        notes.append(f"MPS inventory could not be queried: {exc}")
    return devices, notes


def _record(raw: Any, worker: str | None, default_type: str | None = None) -> dict[str, Any] | None:
    if isinstance(raw, str):
        try:
            name = device_name(raw)
        except ConfigurationError:
            return None
        raw = {"device": name}
    if not isinstance(raw, Mapping) or raw.get("available") is False:
        return None
    kind = raw.get("type", raw.get("kind", default_type))
    name = raw.get("device")
    if kind == "gpu":
        kind = "cuda"
    if isinstance(name, str):
        try:
            name = device_name(name)
        except ConfigurationError:
            return None
        kind = name.split(":")[0]
    if not isinstance(kind, str) or kind not in {"cuda", "cpu", "mps"}:
        return None
    index = raw.get("index", raw.get("id", 0))
    if isinstance(name, str) and ":" in name:
        index = int(name.split(":")[1])
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        index = 0
    bf16 = raw.get("bf16", raw.get("supports_bf16"))
    capability = raw.get("compute_capability", raw.get("capability"))
    if bf16 is None and isinstance(capability, (list, tuple)) and capability:
        try:
            bf16 = int(capability[0]) >= 8
        except (ValueError, TypeError):
            pass
    record: dict[str, Any] = {
        "device": f"cuda:{index}" if kind == "cuda" else kind,
        "type": kind, "index": index, "worker": worker,
        "bf16": bf16 if isinstance(bf16, bool) else None,
    }
    for key in ("name", "total_memory_bytes", "compute_capability"):
        if key in raw and isinstance(raw[key], (str, int, float, list, tuple)):
            record[key] = raw[key]
    logical = raw.get("logical_gpu", raw.get("logical_id"))
    if isinstance(logical, int) and not isinstance(logical, bool) and logical >= 0:
        record["logical_gpu"] = logical
    return record


def _worker_inventory(worker: Mapping[str, Any]) -> list[dict[str, Any]]:
    state = worker.get("state", worker.get("status"))
    if isinstance(state, str) and state in {"offline", "failed", "removed", "drained"}:
        return []
    worker_id = worker.get("id", worker.get("worker_id"))
    worker_id = str(worker_id) if worker_id is not None else None
    metadata = worker.get("metadata", {})
    hardware = {**worker, **metadata} if isinstance(metadata, Mapping) else dict(worker)
    for key in ("inventory", "hardware", "capabilities"):
        extra = hardware.get(key)
        if isinstance(extra, Mapping):
            hardware.update(extra)
    raw_devices = hardware.get("devices")
    default_type = None
    if raw_devices is None:
        raw_devices = hardware.get("gpus")
        default_type = "cuda"
    if isinstance(raw_devices, Mapping):
        raw_devices = list(raw_devices.values())
    if isinstance(raw_devices, int) and not isinstance(raw_devices, bool):
        raw_devices = [{"type": "cuda", "index": index} for index in range(max(0, raw_devices))]
    if not isinstance(raw_devices, (list, tuple)):
        count = hardware.get("cuda_device_count", hardware.get("gpu_count"))
        if isinstance(count, int) and not isinstance(count, bool):
            raw_devices = [{"type": "cuda", "index": index} for index in range(max(0, count))]
        elif isinstance(hardware.get("device"), str):
            raw_devices = [hardware["device"]]
        else:
            raw_devices = []
    records = []
    for raw in raw_devices:
        if isinstance(raw, int) and not isinstance(raw, bool) and default_type == "cuda":
            raw = {"type": "cuda", "index": raw}
        record = _record(raw, worker_id, default_type)
        if record is not None:
            records.append(record)
    return records


def remote_inventory(cluster: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """Read advisory worker metadata without consulting coordinator hardware."""
    notes: list[str] = []
    devices: list[dict[str, Any]] = []
    for method_name in ("workers", "status"):
        method = getattr(cluster, method_name, None)
        if not callable(method):
            continue
        try:
            payload = method()
        except Exception as exc:
            notes.append(f"Remote {method_name} metadata unavailable: {type(exc).__name__}: {exc}")
            continue
        if isinstance(payload, Mapping):
            workers = payload.get("workers")
            if isinstance(workers, Mapping):
                workers = [dict(value, worker_id=key) for key, value in workers.items() if isinstance(value, Mapping)]
            if not isinstance(workers, (list, tuple)):
                workers = [payload]
        elif isinstance(payload, (list, tuple)):
            workers = payload
        else:
            workers = []
        for worker in workers:
            if isinstance(worker, Mapping):
                devices.extend(_worker_inventory(worker))
        if devices:
            break
    next_gpu = 0
    for record in devices:
        if record["type"] == "cuda":
            logical = record.get("logical_gpu", next_gpu)
            record["worker_device"] = record["device"]
            record["device"] = f"cuda:{logical}"
            record["logical_gpu"] = logical
            next_gpu = max(next_gpu, logical + 1)
    if not devices:
        notes.append("Remote inventory has no usable device metadata; the worker must resolve the device.")
    return devices, notes
