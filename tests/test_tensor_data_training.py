"""Native PyTorch comparisons at MLZO's tensor/data execution boundaries."""

from collections import OrderedDict, namedtuple
from copy import deepcopy
from dataclasses import dataclass

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import mlzo as ml
from mlzo.errors import ConfigurationError
from mlzo.training.adapters import builtin_loss
from mlzo.training.runtime import restricted_copy


@pytest.mark.parametrize("kind", ["single", "paired", "mapping"])
def test_wrapped_columns_index_native_samples(kind):
    x = ml.tensor([[1., 2.], [3., 4.], [5., 6.]])
    y = ml.tensor([0, 1, 0])
    if kind == "single":
        batch = next(iter(ml.data(x).batch(2).build()))
        torch.testing.assert_close(batch, x.raw[:2])
    elif kind == "paired":
        batch = next(iter(ml.data(x, y).batch(2).build()))
        torch.testing.assert_close(batch, (x.raw[:2], y.raw[:2]))
    else:
        batch = next(iter(ml.data({"x": x, "y": y}).batch(2).build()))
        torch.testing.assert_close(batch, {"x": x.raw[:2], "y": y.raw[:2]})


@pytest.mark.parametrize("loss", [
    "cross_entropy",
    nn.CrossEntropyLoss(),
    nn.CrossEntropyLoss(weight=torch.tensor([0.5, 2., 3.], dtype=torch.float64),
                        ignore_index=-7, label_smoothing=0.2),
    nn.CrossEntropyLoss(weight=torch.tensor([0.5, 2., 3.], dtype=torch.float64),
                        ignore_index=-7, label_smoothing=0.2, reduction="sum"),
])
def test_column_class_indices_match_native_cross_entropy(loss):
    logits = torch.tensor([[0.2, 1.1, -0.3], [0.1, 0.5, 0.7], [1., -1., 0.2]],
                          dtype=torch.float64, requires_grad=True)
    reference = logits.detach().clone().requires_grad_()
    ignore = getattr(loss, "ignore_index", -100)
    targets = torch.tensor([[1], [ignore], [0]])
    native_loss = nn.CrossEntropyLoss() if isinstance(loss, str) else loss
    expected = native_loss(reference, targets[:, 0])
    actual = builtin_loss(logits, targets, loss)
    torch.testing.assert_close(actual.loss, expected)
    actual.loss.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, reference.grad)
    if native_loss.reduction == "sum":
        assert actual.weight == 1
    elif native_loss.weight is None:
        assert actual.weight == 2
    else:
        assert actual.weight == 2.5


def test_uint8_class_indices_match_native_weighted_cross_entropy():
    logits = torch.tensor([[0.2, 1.1, -0.3], [0.1, 0.5, 0.7]], requires_grad=True)
    reference = logits.detach().clone().requires_grad_()
    target = torch.tensor([[1], [0]], dtype=torch.uint8)
    loss = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 2., 3.]))
    expected = loss(reference, target[:, 0])
    actual = builtin_loss(logits, target, loss)
    assert actual.weight == 2.5
    torch.testing.assert_close(actual.loss, expected)
    actual.loss.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, reference.grad)


@pytest.mark.parametrize("source", ["direct", "columns"])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_wrapped_classification_training_matches_unequal_native_accumulation(source, reduction):
    torch.manual_seed(21)
    model = nn.Linear(2, 3).double()
    reference = deepcopy(model)
    x = torch.linspace(-1, 1, 14, dtype=torch.float64).reshape(7, 2)
    y = torch.tensor([0, -7, 2, 1, 0, 2, 1])
    loss = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 2., 3.], dtype=torch.float64),
                               ignore_index=-7, label_smoothing=0.15, reduction=reduction)
    sizes = (2, 3, 2) if source == "direct" else (3, 3, 1)
    native_batches = list(zip(x.split(sizes), y.split(sizes)))
    if source == "direct":
        data = [(ml.wrap(a), ml.tensor(b)) for a, b in native_batches]
    else:
        data = ml.data(ml.tensor(x), ml.tensor(y)).batch(3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.04, momentum=0.7)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.04, momentum=0.7)
    expected_losses, expected_gradients, actual_gradients = [], [], []
    for group in (native_batches[:2], native_batches[2:]):
        with torch.no_grad():
            expected_losses.extend(loss(reference(a), b).item() for a, b in group)
        reference_optimizer.zero_grad()
        a, b = (torch.cat(values) for values in zip(*group))
        group_loss = loss(reference(a), b)
        # Sum reductions retain the existing mean-of-microbatch-sums policy.
        if reduction == "sum":
            group_loss = group_loss / len(group)
        group_loss.backward()
        expected_gradients.append([p.grad.clone() for p in reference.parameters()])
        reference_optimizer.step()

    def capture(optimizer, args, kwargs):
        actual_gradients.append([p.grad.clone() for p in model.parameters()])

    run = ml.train(model, data, loss=loss, optimizer=optimizer, accumulate=2,
                   precision="fp32", matmul="strict")
    with optimizer.register_step_pre_hook(capture):
        history = run.fit(epochs=1, on="cpu")
    assert history.metrics["loss"] == pytest.approx(expected_losses, rel=1e-12)
    torch.testing.assert_close(actual_gradients, expected_gradients, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(model.state_dict(), reference.state_dict(), rtol=1e-12, atol=1e-12)
    assert history.counters["optimizer_steps"] == {"default": 2}


Pair = namedtuple("Pair", "target metadata")


@dataclass(frozen=True)
class Payload:
    values: list
    labels: Pair


@pytest.mark.parametrize("style", ["functional", "class"])
def test_callbacks_receive_native_nested_leaves_with_autograd(style):
    torch.manual_seed(9)
    model = nn.Linear(2, 1).double()
    reference = deepcopy(model)
    leaf = torch.tensor([[1., 2.], [3., 1.]], dtype=torch.float64, requires_grad=True)
    reference_leaf = leaf.detach().clone().requires_grad_()
    inputs = leaf * 2
    target = torch.tensor([[0.2], [0.7]], dtype=torch.float64)
    scalar = torch.tensor(3.)
    vector = torch.arange(2.)
    batch = OrderedDict(payload=Payload(
        [ml.wrap(inputs), (ml.wrap(scalar), ml.wrap(vector))],
        Pair(ml.NamedTensor(target, ("batch", "output")), "kept")))

    def step(module, received):
        assert isinstance(received, OrderedDict)
        payload = received["payload"]
        assert isinstance(payload, Payload)
        assert isinstance(payload.values, list)
        assert isinstance(payload.values[1], tuple)
        assert isinstance(payload.labels, Pair)
        assert payload.labels.metadata == "kept"
        assert payload.values[0] is inputs
        assert payload.values[1][0] is scalar
        assert payload.values[1][1] is vector
        assert payload.labels.target is target
        return F.mse_loss(module(payload.values[0]), payload.labels.target)

    class Algorithm(ml.Training):
        def step(self, batch):
            return step(self.model, batch)

        def evaluate(self, batch):
            return step(self.model, batch)

    options = {"step": step} if style == "functional" else {}
    run = ml.train(model if style == "functional" else Algorithm(model), [batch],
                   optimizer=ml.opt("sgd", lr=0.03), **options)
    expected_loss = F.mse_loss(reference(reference_leaf * 2), target)
    expected_loss.backward()
    torch.optim.SGD(reference.parameters(), lr=0.03).step()
    history = run.fit(epochs=1, on="cpu")
    assert history.metrics["loss"] == pytest.approx([expected_loss.item()])
    torch.testing.assert_close(leaf.grad, reference_leaf.grad)
    torch.testing.assert_close(model.state_dict(), reference.state_dict())
    expected_evaluation = F.mse_loss(reference(inputs), target).item()
    evaluation = run.evaluate([batch], **({"step": step} if style == "functional" else {}))
    assert evaluation["loss"] == pytest.approx(expected_evaluation)


@pytest.mark.parametrize("kind", ["dense", "probability", "single_class_probability"])
def test_other_cross_entropy_target_shapes_keep_native_semantics(kind):
    torch.manual_seed(2)
    if kind == "dense":
        logits = torch.randn(2, 3, 1, 4, dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[[0, 1, 2, 1]], [[2, 1, 0, 0]]])
    else:
        classes = 1 if kind == "single_class_probability" else 3
        logits = torch.randn(2, classes, dtype=torch.float64, requires_grad=True)
        target = torch.randn(2, classes, dtype=torch.float64).softmax(1)
    reference = logits.detach().clone().requires_grad_()
    expected = F.cross_entropy(reference, target, label_smoothing=0.1)
    actual = builtin_loss(logits, target, nn.CrossEntropyLoss(label_smoothing=0.1))
    torch.testing.assert_close(actual.loss, expected)
    assert actual.weight == (target.numel() if kind == "dense" else 2)
    actual.loss.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, reference.grad)


@pytest.mark.parametrize("target", [
    torch.tensor([[[0]], [[1]]]),
    torch.tensor([[0.], [1.]]),
    torch.tensor([[0], [1]], dtype=torch.int32),
    torch.tensor([[False], [True]]),
])
def test_invalid_classification_targets_are_not_coerced(target):
    logits = torch.randn(2, 3)
    # int32 class indices still fail after the narrow shape adaptation because
    # MLZO does not cast targets to the dtype expected by native cross entropy.
    native_target = target[:, 0] if target.dtype == torch.int32 else target
    with pytest.raises(RuntimeError):
        F.cross_entropy(logits, native_target)
    with pytest.raises(RuntimeError):
        builtin_loss(logits, target, "cross_entropy")


def test_all_ignored_column_labels_skip_updates():
    model = nn.Linear(2, 3)
    before = deepcopy(model.state_dict())
    run = ml.train(model, [(ml.tensor([[1., 2.], [3., 4.]]), ml.tensor([-7, -7]))],
                   loss=nn.CrossEntropyLoss(ignore_index=-7, weight=torch.tensor([1., 2., 3.])))
    history = run.fit(epochs=1, on="cpu")
    assert history.metrics["loss"] == [0.]
    assert history.counters["optimizer_steps"] == {"default": 0}
    assert history.counters["skipped_updates"] == {"default": 1}
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)


@pytest.mark.parametrize("loss", ["mse", "l1", "bce", "bce_with_logits"])
def test_regression_and_binary_column_targets_keep_shape(loss):
    raw = torch.tensor([[0.3], [0.6]], requires_grad=True)
    target = torch.tensor([[0.1], [0.9]])
    native = {"mse": F.mse_loss, "l1": F.l1_loss,
              "bce": F.binary_cross_entropy, "bce_with_logits": F.binary_cross_entropy_with_logits}[loss]
    actual = builtin_loss(raw, target, loss)
    torch.testing.assert_close(actual.loss, native(raw, target))
    assert actual.weight == 2


def test_mse_keeps_native_broadcasting_for_explicit_mismatched_shapes():
    prediction = torch.tensor([[0.3], [0.6]], requires_grad=True)
    target = torch.tensor([0.1, 0.9])
    with pytest.warns(UserWarning, match="target size"):
        actual = builtin_loss(prediction, target, "mse")
    with pytest.warns(UserWarning, match="target size"):
        expected = F.mse_loss(prediction, target)
    torch.testing.assert_close(actual.loss, expected)
    assert actual.weight == 4


def test_custom_loss_receives_original_column_target_and_native_calls_stay_native():
    model = nn.Linear(2, 3)
    x, y = ml.tensor([[0.2, 0.8], [0.3, -0.4]]), ml.tensor([1, 0])
    with pytest.raises(RuntimeError):
        F.cross_entropy(model(x.raw), y.raw)

    def loss(prediction, target):
        assert target is y.raw
        assert target.shape == (2, 1)
        return F.cross_entropy(prediction, target[:, 0])

    history = ml.train(model, [(x, y)], loss=loss).fit(epochs=1, on="cpu")
    assert history.counters["optimizer_steps"] == {"default": 1}


def test_wrappers_are_still_rejected_in_checkpoint_state():
    with pytest.raises(ConfigurationError, match="Unsupported checkpoint object"):
        restricted_copy({"wrapped": ml.tensor([1., 2.])})


def test_deployment_unwraps_exported_model_inputs(tmp_path):
    torch.manual_seed(4)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)

        def forward(self, inputs, *, extra):
            return self.linear(inputs["values"][0]) + extra[0]

    model = Model().eval()
    x, offset = torch.randn(3, 2), torch.tensor(0.4)
    artifact = ml.export_model(model, to=tmp_path / "export")
    with ml.deploy(artifact, factory=Model, device="cpu") as deployment:
        expected = model({"values": [x]}, extra=(offset,))
        actual = deployment.predict({"values": [ml.tensor(x)]}, extra=(ml.wrap(offset),))
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (3, 1)
    assert not actual.requires_grad
