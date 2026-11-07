"""Private execution entrypoint. Calls the normal Run.fit, never a second loop."""
from __future__ import annotations

import os
from pathlib import Path
import platform
import signal
import sys
from typing import Any

from ._common import atomic_json, read_json
from ._reports import Spool, error_value, write_result


def _project_imports(project: Path) -> None:
    # Simple importable scripts need no packaging metadata. src-layout projects
    # are also installed when they declare a build/project configuration.
    sys.path[:0] = [str(project), str(project / 'src')]


def main(argv: list[str] | None = None) -> int:
    import torch
    from mlzo._version import __version__
    from .serialization import deserialize_run
    directory = Path((sys.argv[1:] if argv is None else argv)[0]).absolute()
    assignment = read_json(directory / 'assignment.json')
    project = directory / 'project'
    os.chdir(project)
    _project_imports(project)
    spool = Spool(directory)
    versions = {'mlzo': __version__, 'python': platform.python_version(), 'torch': str(torch.__version__),
                'cuda': torch.version.cuda, 'device': assignment['device']}
    atomic_json(directory / 'runtime.json', versions)
    run: Any = None
    phase = 'deserialization'

    def cancelled(signum: int, frame: Any) -> None:
        if run is not None and run._engine is not None and run._state == 'running':
            run._engine._cancel_requested = True
        else:
            raise InterruptedError('Job cancelled before training started.')

    signal.signal(signal.SIGTERM, cancelled)
    signal.signal(signal.SIGINT, cancelled)
    try:
        run = deserialize_run(assignment['spec'])
        # Logical remote GPU IDs may have mapped to a different local index.
        # The coordinator pins that mapping. Never erase a Lazy GPU annotation.
        configured = run.config.get('device')
        if configured is not None:
            from .planning_helpers import validate_device
            original = validate_device(str(configured))
            assigned = assignment['device']
            if original != assigned and not (original.startswith('cuda:') and assigned.startswith('cuda:')):
                raise ValueError('Assigned device conflicts with the submitted explicit device.')
            run.config['device'] = assigned
        # A single worker has one local logical namespace. Conflicting explicit
        # Lazy GPU placement is rejected by the normal planner below.
        phase = 'planning'
        plan = run.plan(on=assignment['device'], parallel=assignment['fit']['parallel'])
        atomic_json(directory / 'worker-plan.json', plan.to_dict())

        def report(event: dict) -> None:
            spool.emit(event)
            if (directory / 'cancel.json').exists() and run._engine is not None:
                run._engine._cancel_requested = True

        run._event_callback = report
        if (directory / 'cancel.json').exists():
            raise InterruptedError('Job cancelled before training started.')
        phase = 'training'
        history = run.fit(epochs=assignment['fit']['epochs'], steps=assignment['fit']['steps'],
                          on=assignment['device'], parallel=assignment['fit']['parallel'])
        phase = 'final_checkpoint'
        reference = run.checkpoint(to=directory / 'internal-checkpoint')
        write_result(directory, {'state': history.state, 'history': history.to_dict(), 'counters': dict(history.counters),
                      'checkpoint': reference.to_dict(), 'runtime': versions, 'error': None})
        return 0
    except BaseException as exc:
        counters = run._engine.counters() if run is not None and run._engine is not None else {}
        is_cancelled = (directory / 'cancel.json').exists() and isinstance(exc, (InterruptedError, KeyboardInterrupt))
        write_result(directory, {'state': 'cancelled' if is_cancelled else 'failed', 'history': None,
                      'counters': counters, 'runtime': versions, 'error': error_value(exc, phase)})
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
