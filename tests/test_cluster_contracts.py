"""Real loopback requests protect the journal and role boundaries in plan.md section 19."""

import hashlib
from http.client import HTTPConnection
import json
import tempfile
import unittest
from pathlib import Path

from torch import nn

import mlzo
from mlzo.cluster import Coordinator, Worker
from mlzo.cluster._common import APIError, Transport
from mlzo.cluster.serialization import serialize_run


class ClusterContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.service = Coordinator(self.root / "coordinator").start()
        self.admin = Transport(self.service.endpoint, self.service.token)
        self.submitter = Transport(self.service.endpoint, self.service.submitter_token)
        self.cluster = mlzo.cluster("test", endpoint=self.service.endpoint,
                                    token=self.service.token, home=self.root / "client")

    def tearDown(self):
        self.service.shutdown()
        self.service._store.db.close()
        self.temporary.cleanup()

    def enroll(self, name):
        return Worker.join(self.cluster.invite(), directory=self.root / name, device="cpu")

    def submit(self):
        archive = b"archive bytes stay outside JSON control messages"
        digest = hashlib.sha256(archive).hexdigest()
        self.submitter.request("PUT", "/v1/archives/" + digest, raw=archive)
        run = mlzo.train(mlzo.Lazy(nn.Linear, 2, 1), [[1, 2]], loss="mse")
        payload = {
            "request_id": run.id,
            "spec": serialize_run(run),
            "archive": digest,
            "fit": {"epochs": None, "steps": 1, "parallel": "single", "resume": None},
            "placement": {"worker_id": None, "device": "cpu", "precision": "fp32"},
            "plan": run.plan(on="cpu").to_dict(),
        }
        accepted = self.submitter.request("POST", "/v1/runs", payload)
        self.assertEqual(self.submitter.request("POST", "/v1/runs", payload), accepted)
        return accepted["id"], digest, archive

    def test_status_and_membership_roles(self):
        self.assertEqual(self.submitter.request("GET", "/v1/status")["runs"], {})
        for transport, method, path, value, status in [
            (Transport(self.service.endpoint), "GET", "/v1/status", None, 401),
            (self.submitter, "POST", "/v1/invite", {}, 403),
        ]:
            with self.subTest(path=path), self.assertRaises(APIError) as caught:
                transport.request(method, path, value)
            self.assertEqual(caught.exception.status, status)
        worker = self.enroll("worker")
        reply = worker._transport.request("POST", "/v1/worker/leave", {})
        self.assertEqual(reply, {"state": "leaving"})
        state = self.admin.request("GET", "/v1/workers")["workers"][0]
        self.assertEqual(state["state"], "stopped")
        self.assertTrue(state["draining"])

    def test_archive_access_requires_assignment_to_that_worker(self):
        owner, outsider = self.enroll("owner"), self.enroll("outsider")
        run_id, digest, archive = self.submit()
        self.assertEqual(self.submitter.request("GET", "/v1/status")["runs"], {"queued": 1})
        job = owner._transport.request("POST", "/v1/worker/claim", {"known_run": None})["job"]
        self.assertEqual(job["id"], run_id)
        self.assertEqual(owner._transport.request("GET", "/v1/archives/" + digest, binary=True), archive)
        with self.assertRaises(APIError) as caught:
            outsider._transport.request("GET", "/v1/archives/" + digest, binary=True)
        self.assertEqual(caught.exception.status, 403)
        self.assertEqual(self.submitter.request("GET", "/v1/status")["runs"], {"preparing": 1})

    def test_malformed_routes_preserve_auth_validation_and_body_precedence(self):
        worker = self.enroll("worker")
        admin, credential = self.service.token, worker._transport._token
        unknown = "Unknown API endpoint."
        invalid_id = "IDs must be 1-128 portable letters, digits, dots, underscores or hyphens."
        cases = [
            (None, "POST", "/unknown", b"[", 404, unknown),
            (None, "GET", "/v1/archives/bad", b"[", 400, "Expected a SHA-256 content digest."),
            (None, "PUT", "/v1/archives/bad", b"[", 400, "Expected a SHA-256 content digest."),
            (None, "POST", "/v1/archives/" + "a" * 64, b"[", 404, unknown),
            (None, "GET", "/v1/worker/unknown", b"[", 401, "Authentication is required."),
            (admin, "POST", "/v1/worker/unknown", b"[", 401, "Worker credentials are invalid or revoked."),
            (credential, "GET", "/v1/worker/unknown", b"[", 404, "Unknown worker endpoint."),
            (credential, "POST", "/v1/worker/unknown/extra", b"[", 404, "Unknown worker endpoint."),
            (credential, "POST", "/v1/worker/unknown", b"[", 400, "Request must contain a JSON object."),
            (credential, "POST", "/v1/worker/unknown", b"{}", 404, unknown),
            (None, "POST", "/v1/runs", b"[", 401, "Authentication is required."),
            (None, "GET", "/v1/runs/..", b"[", 401, "Authentication is required."),
            (admin, "PUT", "/v1/runs/../unknown", b"[", 400, invalid_id),
            (admin, "POST", "/v1/runs/valid/unknown", b"[", 404, unknown),
            (admin, "GET", "/v1/runs/valid/events?after=-1", b"[", 400, "after must be nonnegative."),
            (admin, "POST", "/v1/runs/valid/cancel", b"[", 400, "Request must contain a JSON object."),
        ]
        for token, method, path, body, status, message in cases:
            with self.subTest(method=method, path=path, authenticated=token is not None):
                connection = HTTPConnection(self.service.host, self.service.port, timeout=5)
                headers = {"Content-Type": "application/json"}
                if token is not None:
                    headers["Authorization"] = "Bearer " + token
                try:
                    connection.request(method, path, body, headers)
                    response = connection.getresponse()
                    self.assertEqual(response.status, status)
                    self.assertEqual(json.loads(response.read())["error"]["message"], message)
                finally:
                    connection.close()

    def test_reports_commands_and_cancel_keep_durable_state(self):
        worker = self.enroll("worker")
        run_id, _, _ = self.submit()
        worker._transport.request("POST", "/v1/worker/claim", {"known_run": None})
        event = {"seq": 1, "type": "iteration", "counters": {"iteration": 1}}
        result = {"state": "completed", "history": None, "counters": {"iteration": 1},
                  "checkpoint": {"path": "worker-owned-snapshot"}, "error": None}
        report = {"id": run_id, "events": [event], "result": result}
        for _ in range(2):
            self.assertEqual(worker._transport.request("POST", "/v1/worker/report", report),
                             {"ack": 1, "terminal": True})
        events = self.submitter.request("GET", f"/v1/runs/{run_id}/events?after=0")["events"]
        self.assertEqual(events, [event])
        self.assertEqual(self.submitter.request("GET", "/v1/status")["runs"], {"completed": 1})
        request = {"request_id": "export-request", "spec": {"type": "export", "to": "model"}}
        command = self.submitter.request("POST", f"/v1/runs/{run_id}/command", request)
        self.assertEqual(command["state"], "queued")
        self.assertEqual(worker._transport.request("POST", "/v1/worker/command", {})["command"]["id"], command["id"])
        completed = {"reference": {"path": "model"}, "error": None}
        worker._transport.request("POST", "/v1/worker/command-result", {"id": command["id"], "result": completed})
        self.assertEqual(self.submitter.request("GET", "/v1/commands/" + command["id"])["result"], completed)
        queued_id, _, _ = self.submit()
        cancelled = self.submitter.request("POST", f"/v1/runs/{queued_id}/cancel", {})
        self.assertEqual(cancelled["state"], "cancelled")


if __name__ == "__main__":
    unittest.main()
