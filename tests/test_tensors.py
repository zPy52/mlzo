"""Tensor behavior compared with native PyTorch, including the explicit raw boundary."""

import operator
import platform

import numpy as np
import pytest
import torch

import mlzo as ml
from mlzo.errors import ConfigurationError


@pytest.mark.parametrize("shape, expected", [((), (1, 1)), ((0,), (0, 1)), ((3,), (3, 1)), ((2, 3), (2, 3)), ((2, 1, 3), (2, 1, 3))])
def test_construction_normalizes_only_at_the_boundary(shape, expected):
    raw = torch.zeros(shape)
    wrapped = ml.tensor(raw)
    assert isinstance(wrapped, ml.Tensor)
    assert not isinstance(wrapped, torch.Tensor)
    assert wrapped.shape == expected
    assert ml.tensor(raw, normalize=False).shape == shape
    assert ml.wrap(raw).raw is raw
    assert ml.transforms.Tensor is ml.Tensor
    assert ml.transforms.wrap is ml.wrap
    if raw.numel():
        assert wrapped.raw.data_ptr() == raw.data_ptr()


def test_conversion_aliases_numpy_and_preserves_graphs_and_names():
    array = np.arange(3, dtype=np.float32)
    wrapped = ml.tensor(array)
    wrapped.raw[0, 0] = 7
    assert array[0] == 7
    leaf = torch.arange(3, dtype=torch.float64, requires_grad=True)
    value = ml.tensor(leaf * 2, dtype="float32", device="cpu")
    ml.tensor(value).sum().backward()
    torch.testing.assert_close(leaf.grad, torch.full_like(leaf, 2))
    assert ml.wrap(value) is value
    for original in (leaf, ml.wrap(leaf), leaf * 2):
        with pytest.raises(ConfigurationError, match="detach"):
            ml.tensor(original, requires_grad=False)
        with pytest.raises(ConfigurationError, match="detach"):
            ml.tensor(original, dtype="int64", requires_grad=False)
    assert not ml.tensor(value.detach(), requires_grad=False).requires_grad
    assert ml.tensor([1.0], requires_grad=True).requires_grad
    named = ml.tensor([[1.0, 2.0]], axes=("batch", "features"))
    assert isinstance(named, ml.Tensor)
    assert ml.tensor(named).axes == named.axes
    assert ml.tensor(named, dtype="float64").dtype == torch.float64
    with pytest.raises(ConfigurationError, match="normalize=False"):
        ml.tensor([1, 2], axes=("items",))
    assert ml.tensor([1, 2], axes=("items",), normalize=False).axes == ("items",)
    assert ml.tensor([1, 2], axes=("items", "value")).shape == (2, 1)


@pytest.mark.parametrize("values", [2.0, [1.0, 2.0], [[1.0, 2.0]]])
def test_new_gradient_tensor_exposes_a_native_leaf(values):
    x = ml.tensor(values, requires_grad=True)
    assert x.raw.is_leaf
    (x ** 2).sum().backward()
    torch.testing.assert_close(x.raw.grad, 2 * x.raw)
    before = x.raw.detach().clone()
    torch.optim.SGD([x.raw], lr=0.1).step()
    torch.testing.assert_close(x.raw, before * 0.8)


def test_invalid_names_do_not_enable_gradients_on_the_input():
    raw = torch.ones(2, 3)
    with pytest.raises(ConfigurationError):
        ml.tensor(raw, axes=("one",), requires_grad=True)
    assert not raw.requires_grad


def test_views_clones_and_explicit_detach_match_native_autograd():
    raw = torch.arange(24.0).reshape(2, 3, 4).requires_grad_()
    reference = raw.detach().clone().requires_grad_()
    x = ml.wrap(raw)[:, -1, 1].reorder(2, 0, 1).reshape(2).clone()
    expected = reference[:, -1:, 1:2].permute(2, 0, 1).reshape(2).clone()
    x.backward(ml.wrap(torch.tensor([2.0, 3.0])))
    expected.backward(torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(raw.grad, reference.grad, rtol=0, atol=0)
    detached = x.detach()
    assert not detached.requires_grad
    assert detached.raw.data_ptr() == x.raw.data_ptr()


def test_scalar_vector_empty_and_named_scalar_shape_edges():
    scalar = ml.tensor(4.0, normalize=False)
    assert scalar[...].shape == ()
    assert scalar[None].shape == (1,)
    assert scalar.sum().shape == ()
    assert scalar.reorder(()).shape == ()
    with pytest.raises(IndexError):
        scalar[0]
    vector = ml.tensor([1, 2, 3], normalize=False)
    assert vector[-1].shape == (1,)
    empty = ml.tensor([])
    assert empty.shape == (0, 1)
    assert empty.sum().shape == (1, 1)
    with pytest.raises(IndexError):
        empty[0]
    named = ml.tensor(4.0, axes=(), normalize=False)
    assert named.transpose(0, 0).axes == ()
    assert named.sum(keepdims=False).axes == ()
    assert named.reorder(()).axes == ()


def test_named_augmented_arithmetic_aligns_names_and_refuses_unknown_axes():
    raw = torch.arange(6.0).reshape(2, 3)
    x = ml.tensor(raw, axes=("batch", "features"))
    y = ml.tensor(torch.arange(6.0).reshape(3, 2), axes=("features", "batch"))
    expected = raw + y.raw.T
    x += y
    assert x.raw is raw
    torch.testing.assert_close(raw, expected)
    with pytest.raises(ConfigurationError, match="add axes"):
        x += ml.tensor([1.0, 2.0], axes=("other",), normalize=False)
    unnamed = ml.wrap(torch.zeros(2, 3))
    with pytest.raises(ConfigurationError, match="named"):
        unnamed += x


def test_explicit_shape_operations_and_conversions_match_pytorch():
    raw = torch.arange(24).reshape(2, 3, 4)
    x = ml.wrap(raw).astype("float64").on("cpu")
    assert x.dtype == torch.float64
    assert x.device == torch.device("cpu")
    torch.testing.assert_close(x.transpose().raw, raw.double().transpose(-2, -1))
    torch.testing.assert_close(x.transpose(0, 2).raw, raw.double().transpose(0, 2))
    torch.testing.assert_close(x.reorder(2, 0, 1).raw, raw.double().permute(2, 0, 1))
    torch.testing.assert_close(x.reorder((2, 0, 1)).raw, raw.double().permute(2, 0, 1))
    assert x.add_dim(-1).remove_dim(-1).shape == raw.shape
    with pytest.raises(ConfigurationError, match="size 1"):
        x.remove_dim(0)
    assert ml.tensor([1, 2, 3]).remove_dim(1).shape == (3,)
    assert x.reshape(24).shape == (24,)
    assert x.reshape((24,)).shape == (24,)
    assert ml.tensor(3).reshape(()).shape == ()
    assert x.flatten().shape == (24,)
    torch.testing.assert_close(x.flatten(1, 2).raw, raw.double().flatten(1, 2))
    assert x.to(dtype=torch.float32).dtype == torch.float32
    assert x.astype(torch.float32).dtype == torch.float32
    with pytest.raises(ConfigurationError, match="dtype"):
        x.astype("not_a_dtype")


@pytest.mark.parametrize("method", ["mean", "sum"])
@pytest.mark.parametrize("axis", [None, 0, -1, (0, -1), []])
def test_reductions_keep_dimensions_by_default(method, axis):
    raw = torch.arange(24.0).reshape(2, 3, 4)
    x = ml.wrap(raw)
    expected = getattr(raw, method)(dim=axis, keepdim=True)
    torch.testing.assert_close(getattr(x, method)(axis=axis).raw, expected)
    torch.testing.assert_close(getattr(x, method)(dim=axis, keepdim=False).raw,
                               getattr(raw, method)(dim=axis, keepdim=False))
    assert getattr(x, method)().shape == (1, 1, 1)
    assert getattr(x, method)(keepdims=False).shape == ()
    assert getattr(x, method)(0, False).shape == (3, 4)
    assert getattr(x, method)(axis=0, dtype="float64").dtype == torch.float64
    with pytest.raises(ConfigurationError, match="axis.*dim"):
        getattr(x, method)(axis=0, dim=1)
    with pytest.raises(ConfigurationError, match="keepdim"):
        getattr(x, method)(keepdims=True, keepdim=False)


@pytest.mark.parametrize("index, native_index", [
    (0, (slice(0, 1),)),
    (-1, (slice(1, 2),)),
    ((1, -1, 0), (slice(1, 2), slice(2, 3), slice(0, 1))),
    ((Ellipsis, -1), (Ellipsis, slice(3, 4))),
    ((None, -1, Ellipsis, 0), (None, slice(1, 2), Ellipsis, slice(0, 1))),
    ((1, None, -2), (slice(1, 2), None, slice(1, 2))),
    ((Ellipsis, None), (Ellipsis, None)),
    ((), ()),
    ((slice(None), slice(1, 3)), (slice(None), slice(1, 3))),
    (np.int64(-1), (slice(1, 2),)),
])
def test_basic_integer_indexing_retains_each_dimension(index, native_index):
    raw = torch.arange(24).reshape(2, 3, 4)
    actual = ml.wrap(raw)[index]
    torch.testing.assert_close(actual.raw, raw[native_index])
    assert actual.raw.data_ptr() == raw[native_index].data_ptr()


@pytest.mark.parametrize("index", [2, -3, (Ellipsis, 4), (0, 0, 0, 0)])
def test_invalid_basic_indices_do_not_become_empty_slices(index):
    with pytest.raises(IndexError):
        ml.wrap(torch.zeros(2, 3, 4))[index]


def test_repeated_ellipsis_is_left_to_native_pytorch():
    raw = torch.arange(24).reshape(2, 3, 4)
    torch.testing.assert_close(ml.wrap(raw)[..., ...].raw, raw[..., ...])


@pytest.mark.parametrize("index", [
    ([0, 1], 1),
    (1, [0, 2]),
    torch.tensor([True, False]),
    torch.tensor(1),
    (True, 1),
    False,
    (torch.tensor([0, 1]), slice(None), torch.tensor([2, 3])),
])
def test_advanced_and_boolean_indexing_preserves_native_semantics(index):
    raw = torch.arange(24).reshape(2, 3, 4)
    torch.testing.assert_close(ml.wrap(raw)[index].raw, raw[index])


def test_assignment_and_explicit_inplace_operations_preserve_aliases_and_grad_rules():
    raw = torch.zeros(2, 3)
    x = ml.wrap(raw)
    x[:, -1] = ml.tensor([7.0, 8.0])
    torch.testing.assert_close(raw, torch.tensor([[0.0, 0.0, 7.0], [0.0, 0.0, 8.0]]))
    x[[0, 1], 0] = ml.wrap(torch.tensor([1.0, 2.0]))
    x += 2
    assert x.raw is raw
    torch.testing.assert_close(raw, torch.tensor([[3.0, 2.0, 9.0], [4.0, 2.0, 10.0]]))
    view = x[0]
    view.raw.mul_(2)
    torch.testing.assert_close(raw[0], torch.tensor([6.0, 4.0, 18.0]))
    with pytest.raises(AttributeError):
        x.add_(1)
    leaf = ml.tensor([1.0], requires_grad=True)
    with pytest.raises(RuntimeError):
        leaf += 1
    with pytest.raises(IndexError):
        x[2] = 0
    clone = x.clone()
    clone.raw.zero_()
    assert torch.count_nonzero(raw) > 0
    assert clone.raw.data_ptr() != raw.data_ptr()


@pytest.mark.parametrize("operation", [operator.iadd, operator.isub, operator.imul, operator.itruediv, operator.ifloordiv, operator.ipow])
def test_augmented_arithmetic_mutates_the_native_storage(operation):
    raw = torch.tensor([[4.0, 9.0]])
    expected = raw.clone()
    wrapped = ml.wrap(raw)
    assert operation(wrapped, ml.wrap(torch.tensor(2.0))) is wrapped
    operation(expected, 2)
    torch.testing.assert_close(raw, expected)


@pytest.mark.parametrize("operation", [operator.add, operator.sub, operator.mul, operator.truediv, operator.floordiv, operator.pow, operator.matmul])
def test_arithmetic_matches_native_in_both_orders(operation):
    raw = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    other = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    for left, right in [(ml.wrap(raw), ml.wrap(other)), (ml.wrap(raw), other), (raw, ml.wrap(other))]:
        torch.testing.assert_close(operation(left, right).raw, operation(raw, other))
    if operation is not operator.matmul:
        torch.testing.assert_close(operation(ml.wrap(raw), 2).raw, operation(raw, 2))
        torch.testing.assert_close(operation(2, ml.wrap(raw)).raw, operation(2, raw))
    torch.testing.assert_close((-ml.wrap(raw)).raw, -raw)


def test_matmul_method_uses_positional_pytorch_semantics():
    left = torch.arange(6.0).reshape(2, 3)
    right = torch.arange(12.0).reshape(3, 4)
    for other in (right, ml.wrap(right)):
        torch.testing.assert_close(ml.wrap(left).matmul(other).raw, left.matmul(right))
    named = ml.tensor(left, axes=("rows", "inner"))
    result = named.matmul(ml.tensor(right, axes=("inner", "columns")))
    assert type(result) is ml.Tensor
    torch.testing.assert_close(result.raw, left @ right)


def test_named_operations_retain_alignment_broadcasting_and_gradients():
    raw = torch.arange(24.0).reshape(2, 3, 4).requires_grad_()
    x = ml.tensor(raw, axes=("batch", "time", "features"))
    y = ml.tensor(torch.arange(4.0), axes=("features",), normalize=False)
    actual = (x + y).align("features", "batch", "time").flatten(("features", "batch"), into="items")
    expected = (raw + y.raw).permute(2, 0, 1).flatten(0, 1)
    assert actual.axes == ("items", "time")
    torch.testing.assert_close(actual.raw, expected)
    assert x.mean("time").axes == x.axes
    assert x.mean("time").shape == (2, 1, 4)
    assert x.sum(axis=-1, keepdims=False).axes == ("batch", "time")
    assert x.mean(("batch", "time"), keepdim=False).axes == ("features",)
    torch.testing.assert_close(x.sum("batch", "features").raw, raw.sum((0, 2), keepdim=True))
    actual.mean().backward()
    torch.testing.assert_close(raw.grad, torch.full_like(raw, 1 / 24))
    assert x.transpose().axes == ("batch", "features", "time")
    assert x.reorder(2, 0, 1).axes == ("features", "batch", "time")
    assert x[:, -1].axes == x.axes
    assert type(x[None]) is ml.Tensor
    assert type(x[[0]]) is ml.Tensor
    assert type(x.reshape(24)) is ml.Tensor
    assert type(x.add_dim(0)) is ml.Tensor
    assert x.mean("time").remove_dim(1).axes == ("batch", "features")
    assert x.astype("float64").axes == x.axes
    assert x.detach().axes == x.axes
    with pytest.raises(ConfigurationError, match="unnamed nonscalar"):
        x + ml.wrap(torch.ones(4))
    with pytest.raises(ConfigurationError):
        x.mean("unknown")


def test_native_nn_optimizer_and_func_use_raw_without_gradient_changes():
    model = torch.nn.Linear(2, 1, dtype=torch.float64)
    reference = torch.nn.Linear(2, 1, dtype=torch.float64)
    reference.load_state_dict(model.state_dict())
    raw = torch.tensor([[0.2, 0.4], [-0.5, 0.3]], dtype=torch.float64, requires_grad=True)
    ref_input = raw.detach().clone().requires_grad_()
    x = ml.wrap(raw)
    output = ml.wrap(model(x.raw))
    loss = (output ** 2).mean()
    loss.backward()
    reference(ref_input).square().mean().backward()
    torch.testing.assert_close(raw.grad, ref_input.grad, rtol=0, atol=0)
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    ref_optim = torch.optim.SGD(reference.parameters(), lr=0.1)
    optim.step()
    ref_optim.step()
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(ml.grad(lambda v: v.square().sum())(x.raw), 2 * raw)
    torch.testing.assert_close(ml.vmap(lambda v: v * 2)(x.raw), 2 * raw)


def test_compile_uses_native_tensor_boundary():
    raw = torch.randn(2, 3)
    if platform.system() == "Darwin" and platform.machine() == "x86_64":
        with pytest.raises(RuntimeError, match="not supported"):
            ml.jit(lambda v: v.sin() + 1, backend="eager")(ml.wrap(raw).raw)
        pytest.skip("PyTorch 2.2.2 does not support torch.compile on Intel macOS")
    compiled = ml.jit(lambda v: v.sin() + 1, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(ml.wrap(raw).raw), raw.sin() + 1)
