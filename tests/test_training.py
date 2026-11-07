from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import mlzo
from mlzo.checkpoint import load_snapshot, read_manifest
from mlzo.errors import CheckpointError, TrainingError


def test_planning_defers_model_and_data_until_fit():
    calls = []

    def model_factory():
        calls.append("model")
        return nn.Linear(2, 1)

    def data_factory():
        calls.append("data")
        return [(torch.ones(2), torch.ones(1))]

    run = mlzo.train(mlzo.Lazy(model_factory), mlzo.data(data_factory).batch(1), loss="mse")
    assert run.plan(on="cpu").strategy == "single"
    assert calls == []
    history = run.fit(epochs=1, on="cpu")
    assert calls == ["model", "data"]
    assert history.counters["optimizer_steps"] == {"default": 1}


def test_accumulated_loss_gradients_and_updates_match_pytorch():
    torch.manual_seed(17)
    model = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))
    reference = deepcopy(model)
    x = torch.tensor([[1.0, 0.5], [-0.5, 0.3], [0.1, -0.2], [0.7, 0.8], [-0.3, 0.4]])
    y = torch.tensor([[0.4], [-0.2], [0.8], [0.1], [-0.5]])
    # Unequal microbatches and a final short accumulation group exercise both
    # sample weighting and the epoch-boundary flush.
    batches = [(x[:2], y[:2]), (x[2:3], y[2:3]), (x[3:], y[3:])]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.07, momentum=0.9)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.07, momentum=0.9)
    observed_gradients = []

    def capture_gradients(optimizer, args, kwargs):
        observed_gradients.append([p.grad.detach().clone() for p in model.parameters()])

    expected_losses, expected_gradients = [], []
    for group in (batches[:2], batches[2:]):
        with torch.no_grad():
            expected_losses.extend(F.mse_loss(reference(a), b).item() for a, b in group)
        reference_optimizer.zero_grad()
        inputs, targets = (torch.cat(values) for values in zip(*group))
        F.mse_loss(reference(inputs), targets).backward()
        expected_gradients.append([p.grad.detach().clone() for p in reference.parameters()])
        reference_optimizer.step()

    with optimizer.register_step_pre_hook(capture_gradients):
        history = mlzo.train(model, batches, loss="mse", optimizer=optimizer,
                             accumulate=2, precision="fp32", matmul="strict").fit(epochs=1, on="cpu")

    assert history.metrics["loss"] == pytest.approx(expected_losses, rel=1e-5, abs=1e-7)
    assert len(observed_gradients) == 2
    for actual, expected in zip(observed_gradients, expected_gradients):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(model.state_dict(), reference.state_dict(), rtol=1e-5, atol=1e-7)
    assert history.counters["optimizer_steps"] == {"default": 2}


def test_epoch_boundary_resume_restores_optimizer_schedule_and_rng(tmp_path):
    batches = [(torch.tensor([[0.2, 0.4], [-0.5, 0.3]]), torch.tensor([[0.7], [-0.1]]))]

    def model_factory():
        return nn.Sequential(nn.Linear(2, 4), nn.Dropout(0.25), nn.Linear(4, 1))

    def make_run(seed):
        return mlzo.train(
            mlzo.Lazy(model_factory),
            batches, loss="mse", optimizer=mlzo.opt("adamw", lr=0.03),
            schedule=mlzo.cosine(total_steps=4), seed=seed, precision="fp32", matmul="strict",
        )

    uninterrupted = make_run(11)
    expected = uninterrupted.fit(epochs=4, on="cpu")
    expected_state = load_snapshot(uninterrupted.checkpoint(to=tmp_path / "expected"))
    interrupted = make_run(11)
    interrupted.fit(epochs=2, on="cpu")
    saved = interrupted.checkpoint(to=tmp_path / "saved")
    resumed = make_run(99)
    actual = resumed.fit(epochs=4, on="cpu", resume=saved)
    actual_state = load_snapshot(resumed.checkpoint(to=tmp_path / "actual"))

    assert actual.counters == expected.counters
    assert actual.metrics["loss"] == pytest.approx(expected.metrics["loss"][2:], rel=1e-6)
    for field in ("model", "optimizers"):
        torch.testing.assert_close(actual_state[field], expected_state[field], rtol=0, atol=0)
    assert actual_state["schedulers"] == expected_state["schedulers"]
    torch.testing.assert_close(actual_state["rng"]["torch"], expected_state["rng"]["torch"], rtol=0, atol=0)
    for key in expected_state["rng"].keys() - {"torch"}:
        assert actual_state["rng"][key] == expected_state["rng"][key]

    # Keep the file size intact so the content checksum, rather than a size
    # check or deserialization error, must reject the damaged snapshot.
    manifest = read_manifest(saved)
    payload = tmp_path / "saved" / saved.snapshot_id / manifest["state"]["file"]
    content = bytearray(payload.read_bytes())
    content[len(content) // 2] ^= 1
    payload.write_bytes(content)
    with pytest.raises(CheckpointError, match="SHA256"):
        load_snapshot(saved)


def test_failed_step_clears_gradients_and_refuses_unsafe_checkpoint(tmp_path):
    torch.manual_seed(3)
    model = nn.Linear(2, 1)
    original = deepcopy(model.state_dict())
    calls = 0

    def step(model, batch):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("deliberate failure with pending gradients")
        return F.mse_loss(model(batch[0]), batch[1])

    batch = (torch.ones(1, 2), torch.ones(1, 1))
    run = mlzo.train(model, [batch, batch], step=step, accumulate=2)
    with pytest.raises(TrainingError, match="deliberate failure"):
        run.fit(epochs=1, on="cpu")
    assert run.status().state == "failed"
    assert run.status().counters["optimizer_steps"] == {"default": 0}
    assert all(p.grad is None for p in model.parameters())
    torch.testing.assert_close(model.state_dict(), original, rtol=0, atol=0)
    with pytest.raises(TrainingError):
        run.checkpoint(to=tmp_path / "unsafe")
    assert not (tmp_path / "unsafe").exists()
