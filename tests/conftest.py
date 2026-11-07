from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch


@pytest.fixture
def tmp_path(request):
    # Each run cleans only its own state, including when pytest runs overlap.
    root = request.config.cache.mkdir("mlzo-runtime")
    with TemporaryDirectory(prefix="case-", dir=root) as directory:
        yield Path(directory)


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("MLZO_HOME", str(tmp_path / "mlzo-home"))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    for name in ("MLZO_TOKEN", "MLZO_SUBMITTER_TOKEN", "MLZO_INVITE_TOKEN"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
