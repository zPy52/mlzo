"""Same-host HTTP and subprocess coverage of the single-worker contract."""

import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest
import torch

import mlzo
from mlzo.checkpoint import load_snapshot
from mlzo.cluster import Cluster, Coordinator, Worker
from mlzo.errors import ClusterError, TrainingError


pytestmark = pytest.mark.skipif(os.name != "posix", reason="The alpha worker requires POSIX process groups.")


@pytest.fixture
def coordinator(tmp_path):
    service = Coordinator(tmp_path / "coordinator", max_runtime=30)
    try:
        with service:
            yield service, Cluster("local", endpoint=service.endpoint, token=service.token)
    finally:
        # shutdown() closes the listener; the alpha has no public store-close API.
        service._store.db.close()


def test_enrollment_is_single_use_and_credentials_are_scoped(coordinator, tmp_path):
    service, admin = coordinator
    submitter = Cluster("submitter", endpoint=service.endpoint, token=service.submitter_token)
    assert submitter.status()["state"] == "ready"
    with pytest.raises(ClusterError):
        submitter.invite()
    invitation = admin.invite()
    worker = Worker.join(invitation, directory=tmp_path / "worker", device="cpu")
    with pytest.raises(ClusterError):
        Worker.join(invitation, directory=tmp_path / "other-worker", device="cpu")
    identity = json.loads((worker.directory / "identity.json").read_text())
    worker_client = Cluster("worker", endpoint=service.endpoint, token=identity["token"])
    with pytest.raises(ClusterError):
        worker_client.workers()
    assert [entry["id"] for entry in admin.workers()] == [worker.id]


def _kill_owned_jobs(directory):
    # Launchers create their own process groups. If the supervisor cannot stop,
    # only kill a group whose PID and nonce still match this fixture's records.
    records = []
    active = directory / "active.json"
    if active.exists():
        records.append(json.loads(active.read_text()))
    records.extend(json.loads(path.read_text()) for path in directory.glob("jobs/*/child.json"))
    for record in records:
        pid, nonce = record.get("pid"), record.get("nonce")
        if not isinstance(pid, int) or not isinstance(nonce, str):
            continue
        command = subprocess.run(["ps", "-ww", "-p", str(pid), "-o", "args="],
                                 capture_output=True, text=True, timeout=3, check=False).stdout
        try:
            if nonce in command and "mlzo.cluster._launcher" in command and os.getpgid(pid) == pid:
                os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.fixture
def remote_worker(coordinator, tmp_path, monkeypatch):
    _, client = coordinator
    worker = Worker.join(client.invite(), directory=tmp_path / "worker", device="cpu")
    project = Path(__file__).parent / "fixtures" / "remote_project"
    monkeypatch.syspath_prepend(str(project))
    module = importlib.import_module("tiny_project")
    with (tmp_path / "worker.log").open("w+") as log:
        process = subprocess.Popen(
            [sys.executable, "-c", "from mlzo.cluster import Worker; import sys; "
             "Worker(sys.argv[1]).serve(poll_interval=0.05)", str(worker.directory)],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            yield client, worker, project, module
        finally:
            try:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            finally:
                _kill_owned_jobs(worker.directory)
                if process.returncode not in (0, None):
                    log.seek(0)
                    print(log.read())
                sys.modules.pop("tiny_project", None)


def _run(project, factory):
    return mlzo.train(mlzo.Lazy(factory), mlzo.data(project.samples).batch(2),
                      loss="mse", optimizer="sgd", lr=0.03, seed=13, precision="fp32", matmul="strict")


def test_remote_training_and_reconnect_match_local_run(remote_worker, tmp_path):
    client, worker, code, project = remote_worker
    local = _run(project, project.model)
    expected = local.fit(epochs=2, on="cpu")
    submitted = _run(project, project.model).fit(epochs=2, on=client, code=code, wait=False)
    result = client.wait(submitted.id, timeout=50, poll_interval=0.05)
    assert result["state"] == "completed", result
    assert result["worker_id"] == worker.id
    actual = mlzo.run(submitted.id, on=client).wait()
    assert actual.counters == expected.counters
    assert actual.metrics["loss"] == pytest.approx(expected.metrics["loss"], rel=1e-6, abs=1e-7)
    # Both worker and test share this host's filesystem. This is not a remote
    # checkpoint download API or evidence of inter-machine gradient exchange.
    remote_state = load_snapshot(client.checkpoint_run(submitted.id))
    local_state = load_snapshot(local.checkpoint(to=tmp_path / "local-checkpoint"))
    torch.testing.assert_close(remote_state["model"], local_state["model"], rtol=1e-6, atol=1e-7)
    events = client.events(submitted.id)
    assert any(event["type"] == "completed" for event in events)
    assert client.events(submitted.id, after=events[-1]["seq"]) == []


def test_remote_failure_is_terminal_and_worker_accepts_next_job(remote_worker):
    client, worker, code, project = remote_worker
    broken = _run(project, project.broken_model).fit(epochs=1, on=client, code=code, wait=False)
    result = client.wait(broken.id, timeout=50, poll_interval=0.05)
    assert result["state"] == "failed", result
    assert result["error"]["phase"] == "training"
    assert "deliberate remote construction failure" in result["error"]["message"]
    with pytest.raises(TrainingError):
        broken.wait()
    successor = _run(project, project.model).fit(epochs=1, on=client, code=code, wait=False)
    result = client.wait(successor.id, timeout=50, poll_interval=0.05)
    assert result["state"] == "completed", result
    assert result["worker_id"] == worker.id
    assert result["counters"]["optimizer_steps"] == {"default": 2}
