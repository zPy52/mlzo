"""Behavioral contracts for declarative data source adapters."""

from copy import deepcopy
from itertools import count, islice

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset

import mlzo as ml
from mlzo.cluster.serialization import deserialize_run, serialize_run
from mlzo.data.sources import materialize_source
from mlzo.errors import ConfigurationError, UnsupportedFeatureError


def get_sample(index):
    return index * 2


def finite_stream():
    return iter(range(9))


finite_stream.mlzo_replayable = True


def unshardable_stream():
    return iter(range(9))


def infinite_stream():
    return count()


async def async_source(index=0):
    return index


def forbidden_get(index):
    raise AssertionError("get executed before iteration")


def forbidden_stream():
    raise AssertionError("stream executed before iteration")


def test_paired_tensors_preserve_sample_shapes_and_autograd():
    inputs = torch.arange(6.0).reshape(3, 2).requires_grad_()
    targets = torch.arange(3)
    pipeline = ml.data(inputs, targets)
    assert len(pipeline) == 3
    dataset = materialize_source(pipeline.source)
    assert isinstance(dataset, Dataset)
    samples = list(pipeline)
    assert [tuple(x.shape) for x, _ in samples] == [(2,), (2,), (2,)]
    assert [tuple(y.shape) for _, y in samples] == [(), (), ()]
    torch.testing.assert_close(torch.stack([x for x, _ in samples]), inputs)
    torch.testing.assert_close(torch.stack([y for _, y in samples]), targets)
    sum(x.sum() for x, _ in samples).backward()
    torch.testing.assert_close(inputs.grad, torch.ones_like(inputs))


def test_paired_sequences_and_empty_sources():
    assert list(ml.data([[1, 2], [3, 4]], [0, 1])) == [([1, 2], 0), ([3, 4], 1)]
    empty = ml.data(torch.empty(0, 2), [])
    assert len(empty) == 0
    assert list(empty) == []


@pytest.mark.parametrize("inputs,targets", [
    (torch.ones(3, 2), torch.ones(2)),
    (torch.tensor(1), [0]),
    ([1], torch.tensor(0)),
    ([1], None),
    ((x for x in range(2)), [0, 1]),
    ({"x": [1]}, [0]),
    ("ab", [0, 1]),
])
def test_invalid_paired_columns_fail_at_configuration(inputs, targets):
    with pytest.raises(ConfigurationError):
        ml.data(inputs, targets)


def test_numpy_arrays_are_indexable_columns_and_samples_without_conversion():
    np = pytest.importorskip("numpy")
    inputs = np.arange(6).reshape(3, 2)
    targets = np.arange(3)
    pipeline = ml.data(inputs, targets)
    assert len(pipeline) == 3
    x, y = next(iter(pipeline))
    assert np.shares_memory(x, inputs)
    assert x.shape == (2,)
    assert y.shape == ()
    assert isinstance(materialize_source(ml.data(inputs).source), Dataset)
    records = ml.data({"x": inputs, "y": targets})
    np.testing.assert_array_equal(next(iter(records))["x"], inputs[0])
    with pytest.raises(ConfigurationError):
        ml.data(np.array(1), targets)
    with pytest.raises(ConfigurationError):
        ml.data(inputs, targets[:2])


def test_single_sequence_of_pairs_is_row_major():
    samples = (([1, 2], 0), ([3, 4], 1))
    assert list(ml.data(samples)) == list(samples)


def test_named_tensor_columns_are_unwrapped_before_sample_indexing():
    inputs = ml.tensor([[1.0, 2.0], [3.0, 4.0]], axes=("samples", "features"))
    targets = ml.tensor([0, 1], axes=("samples",), normalize=False)
    x, y = next(iter(ml.data(inputs, targets)))
    torch.testing.assert_close(x, inputs.raw[0])
    torch.testing.assert_close(y, targets.raw[0])


def test_indexed_source_is_lazy_through_plan_length_spec_and_build():
    calls = []

    def get(index):
        calls.append(index)
        return index * 2

    pipeline = ml.data(get=get, length=3)
    ml.train(ml.Lazy(nn.Linear, 1, 1), pipeline, loss="mse").plan(on="cpu")
    assert len(pipeline) == 3
    restored = ml.DataPipeline.from_spec(pipeline.to_spec(lambda value: value), lambda value: value)
    dataset = materialize_source(restored.source)
    assert isinstance(dataset, Dataset)
    assert not isinstance(dataset, IterableDataset)
    assert len(dataset) == 3
    values = restored.build(epoch=2)
    assert calls == []
    assert list(values) == [0, 2, 4]
    assert calls == [0, 1, 2]
    assert len(pipeline.batch(2)) == 2
    assert len(pipeline.batch(2, drop_last=True)) == 1


def test_indexed_source_enforces_finite_bounds_even_if_get_does_not():
    dataset = materialize_source(ml.data(get=get_sample, length=2).source)
    assert list(dataset) == [0, 2]
    for index in (-1, 2, 100):
        with pytest.raises(IndexError):
            dataset[index]
    with pytest.raises(TypeError):
        dataset[0.5]
    empty = ml.data(get=get_sample, length=0)
    assert len(empty) == 0
    assert list(empty) == []


@pytest.mark.parametrize("kwargs", [
    {}, {"get": get_sample}, {"length": 2}, {"get": None, "length": 2},
    {"get": 3, "length": 2}, {"get": lambda: 0, "length": 2},
    {"get": lambda index, required: 0, "length": 2},
    {"get": lambda *, index: 0, "length": 2},
    {"get": get_sample, "length": -1}, {"get": get_sample, "length": True},
    {"get": get_sample, "length": 2.0}, {"get": get_sample, "length": float("inf")},
    {"get": get_sample, "length": 2**100},
    {"source": [1], "get": get_sample, "length": 1},
    {"targets": [1], "get": get_sample, "length": 1},
    {"stream": finite_stream, "get": get_sample, "length": 2},
    {"stream": finite_stream, "length": 2},
    {"source": [1], "stream": finite_stream},
    {"targets": [1], "stream": finite_stream},
    {"stream": iter([1])}, {"stream": lambda required: iter([required])},
    {"source": async_source}, {"get": async_source, "length": 2},
    {"stream": async_source},
])
def test_invalid_or_ambiguous_modes_raise_configuration_errors(kwargs):
    with pytest.raises(ConfigurationError):
        ml.data(**kwargs)


def test_stream_factory_restarts_per_iteration_and_epoch_without_early_calls():
    calls = []

    def stream():
        calls.append("factory")
        return iter([1, 2, 3])

    pipeline = ml.data(stream=stream)
    ml.train(ml.Lazy(nn.Linear, 1, 1), pipeline, loss="mse").plan(on="cpu")
    restored = ml.DataPipeline.from_spec(pipeline.to_spec(lambda value: value), lambda value: value)
    dataset = materialize_source(restored.source)
    assert isinstance(dataset, IterableDataset)
    built = restored.build(epoch=3)
    with pytest.raises(TypeError):
        len(pipeline)
    assert calls == []
    assert list(built) == [1, 2, 3]
    assert list(built) == [1, 2, 3]
    assert list(restored.build(epoch=4)) == [1, 2, 3]
    assert calls == ["factory"] * 3


def test_infinite_stream_consumes_only_requested_samples():
    assert list(islice(ml.data(stream=infinite_stream), 5)) == list(range(5))


def test_invalid_stream_return_fails_only_on_iteration():
    pipeline = ml.data(stream=lambda: 42)
    built = pipeline.build()
    with pytest.raises(ConfigurationError, match="iterable"):
        iter(built)


@pytest.mark.parametrize("pipeline", [
    lambda: ml.data(get=forbidden_get, length=2),
    lambda: ml.data(stream=forbidden_stream),
])
def test_remote_roundtrip_and_build_never_execute_sample_functions(pipeline):
    run = ml.train(ml.Lazy(nn.Linear, 1, 1), pipeline(), loss="mse")
    run.plan(on="cpu")
    restored = deserialize_run(serialize_run(run))
    restored.plan(on="cpu")
    restored.data.build()


def test_stream_replayable_factory_shards_without_duplicates_in_native_workers():
    dataset = materialize_source(ml.data(stream=finite_stream).source)
    loader = DataLoader(dataset, batch_size=None, num_workers=2, multiprocessing_context="spawn")
    assert list(loader) == list(range(9))


def test_stream_workers_reject_factories_without_replayability_declaration():
    dataset = materialize_source(ml.data(stream=unshardable_stream).source)
    loader = DataLoader(dataset, batch_size=None, num_workers=2, multiprocessing_context="spawn")
    with pytest.raises(UnsupportedFeatureError, match="mlzo_replayable"):
        list(loader)


@pytest.mark.parametrize("pipeline", [
    lambda: ml.data(get=get_sample, length=3),
    lambda: ml.data(stream=finite_stream),
    lambda: ml.data([1, 2], [3, 4]),
])
def test_new_source_specs_roundtrip_through_existing_remote_codec(pipeline):
    run = ml.train(ml.Lazy(nn.Linear, 1, 1), pipeline(), loss="mse")
    spec = serialize_run(run)
    restored = deserialize_run(spec)
    assert serialize_run(restored) == spec
    assert list(restored.data) == list(run.data)


@pytest.mark.parametrize("pipeline", [
    lambda: ml.data(get=lambda index: index, length=2),
    lambda: ml.data(stream=lambda: iter([1, 2])),
    lambda: ml.data(torch.ones(2, 1), torch.zeros(2)),
    lambda: ml.data(DataLoader([1, 2])),
])
def test_remote_codec_still_rejects_local_functions_and_opaque_objects(pipeline):
    run = ml.train(ml.Lazy(nn.Linear, 1, 1), pipeline(), loss="mse")
    with pytest.raises(ConfigurationError):
        serialize_run(run)


def test_existing_loader_pipeline_iterator_factory_and_path_behavior(tmp_path):
    loader = DataLoader([1, 2, 3], batch_size=2)
    pipeline = ml.data(loader)
    assert ml.data(pipeline) is pipeline
    assert pipeline.build() is loader
    iterator = iter([1, 2, 3])
    one_shot = ml.data(iterator)
    assert next(iter(one_shot)) == 1
    assert list(one_shot) == [2, 3]
    assert list(one_shot) == []
    assert list(ml.data(finite_stream)) == list(range(9))
    path = tmp_path / "later.json"
    deferred = ml.data(path)
    spec = deferred.to_spec(lambda value: value)
    restored = ml.DataPipeline.from_spec(spec, lambda value: value)
    path.write_text('[{"x": 1}]')
    assert list(restored) == [{"x": 1}]


def test_new_source_specs_validate_options_and_callable_signatures():
    indexed = ml.data(get=get_sample, length=2).to_spec(lambda value: value)
    stream = ml.data(stream=finite_stream).to_spec(lambda value: value)
    paired = ml.data([1, 2], [3, 4]).to_spec(lambda value: value)
    malformed = []
    for original in (indexed, stream, paired):
        extra = deepcopy(original)
        extra["source"]["options"]["unknown"] = True
        malformed.append(extra)
    invalid_length = deepcopy(indexed)
    invalid_length["source"]["options"]["length"] = -1
    malformed.append(invalid_length)
    invalid_get = deepcopy(indexed)
    invalid_get["source"]["value"] = finite_stream
    malformed.append(invalid_get)
    mismatch = deepcopy(paired)
    mismatch["source"]["options"]["targets"] = [3]
    malformed.append(mismatch)
    for spec in malformed:
        with pytest.raises(ConfigurationError):
            ml.DataPipeline.from_spec(spec, lambda value: value)
