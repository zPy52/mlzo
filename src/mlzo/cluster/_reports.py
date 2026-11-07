"""Bounded disk reporting shared by the supervisor and execution process."""
from __future__ import annotations
from collections import deque
import math
from pathlib import Path
from typing import Any
from ._common import MAX_EVENT, MAX_EVENTS, MAX_RESULT, atomic_json, bounded_json


def scalar_value(value: Any, depth: int = 0) -> Any:
    if depth > 5:
        return None
    if value is None or type(value) in {bool, int}:
        return value
    if type(value) is float:
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value[:1024]
    if isinstance(value, dict):
        return {str(k)[:128]: scalar_value(v, depth + 1) for k, v in list(value.items())[:64]}
    if isinstance(value, (list, tuple)):
        return [scalar_value(v, depth + 1) for v in value[:32]]
    return None


def error_value(exc: BaseException, phase: str) -> dict:
    # Child processes receive no control credentials. Do not attach traceback,
    # environment, command argv or local worker configuration to error reports.
    return {'type': type(exc).__name__, 'message': str(exc)[:2048], 'phase': phase}


def write_result(directory: Path, result: dict) -> None:
    history = result.get('history')
    if isinstance(history, dict):
        records = history.get('records', [])
        for record in records:
            record['metrics'] = scalar_value(record.get('metrics', {}))
        while records:
            try:
                bounded_json(result, MAX_RESULT)
                break
            except ValueError:
                removed = max(1, len(records) // 2)
                del records[:removed]
                history['dropped'] = history.get('dropped', 0) + removed
    atomic_json(directory / 'result.json', result, MAX_RESULT)


class Spool:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.events: deque = deque(maxlen=MAX_EVENTS)
        self.sequence = 0
        self.dropped = 0

    def emit(self, event: dict) -> None:
        self.sequence += 1
        value = scalar_value(event)
        value['seq'] = self.sequence
        try:
            bounded_json(value, MAX_EVENT)
        except ValueError:
            value = {'seq': self.sequence, 'type': str(event.get('type', 'event'))[:64], 'truncated': True}
        if len(self.events) == MAX_EVENTS:
            self.dropped += 1
        self.events.append(value)
        while True:
            try:
                atomic_json(self.directory / 'events.json', {'events': list(self.events), 'dropped': self.dropped})
                break
            except ValueError:
                self.events.popleft()
                self.dropped += 1
