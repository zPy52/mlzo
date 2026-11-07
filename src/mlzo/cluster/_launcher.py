"""Credential-free preparation subprocess; creates a private job venv."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import tomllib
import venv
from typing import Any

from ._common import MAX_ARCHIVE, atomic_json, bounded_json, read_json
from ._reports import error_value, write_result


def child_environment(directory: Path) -> dict[str, str]:
    home = directory / 'runtime-home'
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = directory / 'tmp'
    temporary.mkdir(exist_ok=True, mode=0o700)
    environment = {'PATH': os.defpath, 'HOME': str(home), 'TMPDIR': str(temporary),
                   'TEMP': str(temporary), 'TMP': str(temporary), 'LANG': 'C.UTF-8',
                   'PYTHONNOUSERSITE': '1', 'PYTHONUNBUFFERED': '1',
                   'MLZO_HOME': str(home / '.mlzo'), 'PIP_CONFIG_FILE': os.devnull,
                   'PIP_DISABLE_PIP_VERSION_CHECK': '1', 'PIP_NO_INPUT': '1'}
    # Runtime paths and device visibility are the only inherited settings.
    for name in ('SYSTEMROOT', 'WINDIR', 'CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH'):
        if name in os.environ:
            environment[name] = os.environ[name]
    return environment


def _run(argv: list[str], *, directory: Path, environment: dict, deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Job preparation exceeded its runtime budget.')
    # pip output may contain credentials from index configuration or build
    # scripts. It is not part of the coordinator's scalar protocol.
    completed = subprocess.run(argv, cwd=directory, env=environment, stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=remaining, check=False)
    if completed.returncode:
        raise RuntimeError(f'Worker dependency/project installation failed with exit code {completed.returncode}. Inspect the project dependency declarations on the worker.')


def _prepare(directory: Path, assignment: dict, deadline: float) -> tuple[Path, Path]:
    from .artifacts import extract_project
    project = directory / 'project'
    manifest = extract_project(assignment['archive_path'], project, digest=assignment['archive'], max_bytes=MAX_ARCHIVE)
    atomic_json(directory / 'project-manifest.json', manifest)
    dependencies = manifest.get('dependencies', [])
    if not isinstance(dependencies, list) or any(not isinstance(item, str) or item.startswith('-') for item in dependencies):
        raise ValueError('Unsupported project dependency declarations.')
    environment_dir = directory / 'venv'
    # Writes and pip installs are confined to this venv. The worker's installed
    # PyTorch and MLZO are inherited through path references, never copied from
    # a submitter's virtualenv.
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment_dir)
    python = environment_dir / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    site_candidates = list(environment_dir.glob('lib/python*/site-packages'))
    if not site_candidates:
        raise RuntimeError('Cannot find job virtualenv site-packages.')
    runtime_roots = [str(Path(__file__).resolve().parents[2])]
    torch_spec = importlib.util.find_spec('torch')
    if torch_spec is None or not torch_spec.submodule_search_locations:
        raise RuntimeError('The worker needs an installed PyTorch runtime before serving jobs.')
    runtime_roots.append(str(Path(next(iter(torch_spec.submodule_search_locations))).parent))
    (site_candidates[0] / 'mlzo-worker-runtime.pth').write_text('\n'.join(dict.fromkeys(runtime_roots)) + '\n', encoding='utf-8')
    environment = child_environment(directory)
    if dependencies:
        _run([str(python), '-m', 'pip', 'install', '--no-input', *dependencies], directory=project, environment=environment, deadline=deadline)
    metadata = {}
    if (project / 'pyproject.toml').is_file():
        with (project / 'pyproject.toml').open('rb') as stream:
            metadata = tomllib.load(stream)
    install_project = bool(metadata.get('build-system') or metadata.get('project', {}).get('name')
                           or (project / 'setup.py').is_file() or (project / 'setup.cfg').is_file())
    if install_project:
        # Resolve the project's own metadata too. Legacy setup.py and dynamic
        # project metadata may declare dependencies beyond the static manifest;
        # --no-deps would silently skip those requirements.
        _run([str(python), '-m', 'pip', 'install', '--no-input', str(project)],
             directory=project, environment=environment, deadline=deadline)
    if (directory / 'cancel.json').exists():
        raise InterruptedError('Job cancelled during preparation.')
    atomic_json(directory / 'preparation.json', {'state': 'ready', 'dependencies': dependencies,
                'installed_project': install_project, 'runtime_policy': 'inherit-worker-pytorch',
                'python': sys.version.split()[0], 'archive_digest': assignment['archive']})
    return python, project


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    directory, nonce = Path(arguments[0]).absolute(), arguments[1]
    atomic_json(directory / 'child.json', {'pid': os.getpid(), 'nonce': nonce})
    assignment = read_json(directory / 'assignment.json')
    deadline = time.monotonic() + assignment['fit']['max_runtime']
    child: subprocess.Popen | None = None
    phase = 'preparation'

    def interrupted(signum: int, frame: Any) -> None:
        raise InterruptedError('Job process received a termination signal.')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    def timed_out(signum: int, frame: Any) -> None:
        raise TimeoutError('Job exceeded its finite runtime budget.')
    signal.signal(signal.SIGALRM, timed_out)
    signal.setitimer(signal.ITIMER_REAL, assignment['fit']['max_runtime'])
    try:
        python, project = _prepare(directory, assignment, deadline)
        phase = 'execution'
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Job exceeded its runtime budget during preparation.')
        child = subprocess.Popen([str(python), '-m', 'mlzo.cluster._job', str(directory)],
                                 cwd=project, env=child_environment(directory), stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        child.wait(timeout=remaining)
        if not (directory / 'result.json').is_file():
            raise RuntimeError(f'Job subprocess exited with code {child.returncode} without a terminal result.')
        return child.returncode or 0
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=3)
        timeout = isinstance(exc, (TimeoutError, subprocess.TimeoutExpired))
        if not (directory / 'result.json').exists() or timeout:
            cancelled = (directory / 'cancel.json').exists() and not timeout
            result = {'state': 'cancelled' if cancelled else 'failed', 'history': None, 'counters': {},
                      'runtime': assignment.get('worker_runtime', {}),
                      'error': error_value(TimeoutError('Job exceeded its finite runtime budget.') if timeout else exc, phase)}
            write_result(directory, result)
            atomic_json(directory / 'preparation.json', {'state': 'failed', 'error': result['error']})
        # Clean up any native loader/build children in this job's process group.
        # The launcher always runs in a private session made by Worker.
        if os.name == 'posix':
            os.killpg(os.getpgrp(), signal.SIGKILL)
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    raise SystemExit(main())
