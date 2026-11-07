"""Translate advisory logical GPU plans to one worker's local device."""
from __future__ import annotations
import re
from typing import Any
from mlzo.errors import ConfigurationError


def validate_device(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r'cpu|mps|cuda(?::[0-9]+)?', value):
        raise ConfigurationError('Worker device must be cpu, mps, cuda, or cuda:<index>.')
    return 'cuda:0' if value == 'cuda' else value


def placement_for(run: Any, plan: Any) -> dict:
    explicit = run.config.get('device') is not None or bool(plan.placements)
    desired = plan.device if explicit else None
    worker_id = None
    if explicit:
        for device in plan.inventory:
            if device.get('device') == desired:
                worker_id = device.get('worker_id', device.get('worker'))
                desired = device.get('worker_device', desired)
                break
        if desired != 'auto':
            desired = validate_device(desired)
    return {'worker_id': worker_id, 'device': desired, 'precision': run.config.get('precision', 'auto')}
