"""Completed-job operations in the original job venv and project directory."""
from __future__ import annotations
import os
from pathlib import Path
import sys
from ._common import atomic_json, read_json
from ._reports import error_value


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    job, directory, nonce = Path(arguments[0]).absolute(), Path(arguments[1]).absolute(), arguments[2]
    atomic_json(directory / 'child.json', {'pid': os.getpid(), 'nonce': nonce})
    try:
        from mlzo.checkpoint import load_snapshot, save_snapshot
        from ._job import _project_imports
        from .serialization import deserialize_run
        assignment = read_json(job / 'assignment.json')
        result = read_json(job / 'result.json')
        command = read_json(directory / 'request.json')['spec']
        project = job / 'project'
        os.chdir(project)
        _project_imports(project)
        state = load_snapshot(result['checkpoint'])
        if command['type'] == 'checkpoint':
            reference = save_snapshot(state, to=command['to'], run_id=assignment['id'],
                                      metadata={'remote': True, 'worker_job_directory': str(project)})
        else:
            run = deserialize_run(assignment['spec'])
            # Reuse the checked worker-side device configuration from execution.
            if run.config.get('device') is not None:
                run.config['device'] = assignment['device']
            engine = run._prepare(on=assignment['device'], parallel=assignment['fit']['parallel'])
            engine.model.load_state_dict(state['model'])
            reference = run.export(to=command['to'], block=command.get('block'))
        atomic_json(directory / 'result.json', {'reference': reference.to_dict(), 'error': None})
        return 0
    except BaseException as exc:
        atomic_json(directory / 'result.json', {'error': error_value(exc, 'completed_command')})
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
