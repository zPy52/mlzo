"""Exercise the implemented single-worker workflow from plan.md sections 19-23."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

import mlzo
from mlzo.cluster import Coordinator, Worker
from mlzo.errors import ClusterError


@unittest.skipUnless(os.name == "posix", "The alpha worker supervisor requires POSIX.")
class WorkerWorkflow(unittest.TestCase):
    def test_worker_training_reconnect_checkpoint_and_export_match_local(self):
        project = Path(__file__).resolve().parents[1] / "examples" / "remote_project"
        with patch.object(sys, "path", [str(project), *sys.path]):
            from toy_project.data import samples

            def learning():
                return mlzo.train(
                    mlzo.Lazy(nn.Linear, 4, 2),
                    mlzo.data(samples).shuffle(seed=17).batch(16),
                    loss="cross_entropy",
                    optimizer=mlzo.opt("sgd", lr=0.1),
                    seed=17,
                    precision="fp32",
                )

            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                service = Coordinator(root / "coordinator", max_runtime=120).start()
                worker = None
                try:
                    cluster = mlzo.cluster(
                        "verification", endpoint=service.endpoint,
                        token=service.token, home=root / "client",
                    )
                    invite = cluster.invite(max_joins=1)
                    worker = Worker.join(invite, directory=root / "worker", device="cpu")
                    retry = Worker.join(invite, directory=root / "worker", device="cpu")
                    self.assertEqual(retry.id, worker.id)
                    with self.assertRaises(ClusterError):
                        Worker.join(invite, directory=root / "second-worker", device="cpu")
                    worker.start(poll_interval=0.1)

                    remote = learning()
                    remote.fit(steps=3, on=cluster, code=str(project), wait=False)
                    self.assertIsNone(remote._engine)
                    self.assertEqual(remote.algorithm_input.state, "unmaterialized")
                    result = cluster.wait(remote.id, timeout=110, poll_interval=0.1)
                    self.assertEqual(result["state"], "completed", result.get("error"))
                    reconnected = mlzo.run(remote.id, on=cluster)
                    history = reconnected.wait()
                    self.assertEqual(history.counters["iteration"], 3)
                    self.assertTrue(cluster.events(remote.id))

                    actual = mlzo.load_snapshot(reconnected.checkpoint())
                    reference = learning()
                    reference.fit(steps=3, on="cpu")
                    for name, value in reference._engine.model.state_dict().items():
                        torch.testing.assert_close(actual["model"][name], value, rtol=0, atol=0)
                    self.assertEqual(actual["counters"], reference.status().counters)

                    cluster.drain(worker.id)
                    copied = reconnected.checkpoint(to=root / "copied-checkpoint")
                    restored = mlzo.load_snapshot(copied)
                    for name, value in actual["model"].items():
                        torch.testing.assert_close(restored["model"][name], value, rtol=0, atol=0)
                    artifact = reconnected.export(to=root / "exported-model")
                    with mlzo.deploy(artifact, device="cpu") as deployment:
                        inputs = torch.ones(2, 4)
                        with torch.no_grad():
                            expected = reference._engine.model(inputs)
                        torch.testing.assert_close(deployment.predict(inputs), expected, rtol=0, atol=0)
                    self.assertTrue(cluster.workers()[0]["draining"])
                finally:
                    if worker is not None:
                        worker.leave(graceful=False)
                        if worker._thread is not None:
                            worker._thread.join(timeout=20)
                            self.assertFalse(worker._thread.is_alive(), "Temporary worker did not stop")
                    service.shutdown()
                    service._store.db.close()


if __name__ == "__main__":
    unittest.main()
