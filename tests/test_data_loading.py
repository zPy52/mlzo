"""Behavioral checks for native loading and ordered pipeline operations."""

import itertools
import os
import random
from collections import OrderedDict, namedtuple
from tempfile import TemporaryDirectory

import pytest
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, Sampler, get_worker_info

import mlzo as ml
from mlzo.errors import ConfigurationError, UnsupportedFeatureError


@pytest.fixture
def worker_tmp(isolated_runtime, monkeypatch):
    # macOS torch_shm_manager uses a Unix socket under TMPDIR. The full
    # worktree's pytest runtime path exceeds that socket's path limit.
    with TemporaryDirectory(prefix="mlzo-loader-", dir="/tmp") as directory:
        monkeypatch.setenv("TMPDIR", directory)
        yield


def index_sample(index):
    return torch.tensor([index, index + 1]), index


def twice(sample):
    return sample * 2


def positive(sample):
    return sample > 0


def finite_stream():
    yield from range(11)


finite_stream.mlzo_replayable = True


def infinite_stream():
    yield from itertools.count()


infinite_stream.mlzo_replayable = True


def forbidden_stream():
    raise AssertionError("factory was called")


def worker_sample(index):
    return {"value": index, "pid": os.getpid()}


def worker_transform(sample):
    return {**sample, "transform_pid": os.getpid()}


def worker_wrap(sample):
    return {key: ml.wrap(torch.tensor(value)) for key, value in worker_transform(sample).items()}


def worker_combine(samples):
    return {"samples": samples, "combine_pid": os.getpid()}


def failing_sample(index):
    raise RuntimeError("loading failed deliberately")


def regression_sample(index):
    return torch.tensor([float(index)]), torch.tensor([0.])


class SizedShardedNative(IterableDataset):
    def __len__(self):
        return 6

    def __iter__(self):
        worker = get_worker_info()
        start, step = (worker.id, worker.num_workers) if worker else (0, 1)
        for index in range(start, 6, step):
            yield regression_sample(index)


@pytest.mark.parametrize("count,drop_last,expected_sizes", [
    (6, False, [2, 2, 1, 1]),
    (10, False, [3, 3, 2, 2]),
    (10, True, []),
    (14, True, [4, 4]),
])
def test_buffered_worker_lengths_match_native_partial_batches(worker_tmp, count, drop_last, expected_sizes):
    pipeline = ml.data(get=regression_sample, length=count).shuffle(buffer=2).batch(
        4, workers=4, prefetch=2, drop_last=drop_last,
    )
    loader = pipeline.build()
    iterator = iter(loader)
    processes = list(iterator._workers)
    try:
        batches = list(iterator)
    finally:
        iterator._shutdown_workers()
    assert all(not process.is_alive() for process in processes)
    assert [len(x) for x, _ in batches] == expected_sizes
    assert len(pipeline) == len(loader) == len(batches)
    if not drop_last:
        assert sorted(value for x, _ in batches for value in x[:, 0].tolist()) == list(range(count))


def test_buffered_worker_epoch_covers_samples_and_infers_accumulated_horizon(worker_tmp):
    pipeline = ml.data(get=regression_sample, length=6).shuffle(buffer=2).batch(4, workers=4)
    seen, horizons = [], []

    def step(model, batch):
        x, y = batch
        seen.extend(x[:, 0].tolist())
        return (model(x) - y).square().mean()

    def schedule(update, horizon):
        horizons.append(horizon)
        return 1.

    history = ml.train(torch.nn.Linear(1, 1), pipeline, step=step, accumulate=3,
                       schedule=schedule).fit(epochs=1, on="cpu")
    assert sorted(seen) == list(range(6))
    assert history.counters["iteration"] == 4
    assert history.counters["epoch"] == 1
    assert history.counters["optimizer_steps"] == {"default": 2}
    assert set(horizons) == {2}


def test_native_worker_shards_have_unknown_batch_count(worker_tmp):
    pipeline = ml.data(SizedShardedNative()).shuffle(buffer=2).batch(4, workers=4)
    with pytest.raises(TypeError, match="length|shard"):
        len(pipeline)
    loader = pipeline.build()
    with pytest.raises(TypeError, match="length|shard"):
        len(loader)
    iterator = iter(loader)
    processes = list(iterator._workers)
    try:
        assert sorted(value for x, _ in iterator for value in x[:, 0].tolist()) == list(range(6))
    finally:
        iterator._shutdown_workers()
    assert all(not process.is_alive() for process in processes)


def test_unknown_finite_length_requires_steps_and_explicit_accumulation_horizon():
    pipeline = ml.data(get=regression_sample, length=3).filter(lambda pair: True).batch(1)
    with pytest.raises(ConfigurationError, match="fit\\(steps"):
        ml.train(torch.nn.Linear(1, 1), pipeline, loss="mse").fit(epochs=1, on="cpu")
    with pytest.raises(ConfigurationError, match="explicit total_steps"):
        ml.train(torch.nn.Linear(1, 1), pipeline, loss="mse", accumulate=2,
                 schedule=ml.linear()).fit(steps=6, on="cpu")
    history = ml.train(torch.nn.Linear(1, 1), pipeline, loss="mse", accumulate=2,
                       schedule=ml.linear(total_steps=4)).fit(steps=6, on="cpu")
    assert history.counters["optimizer_steps"] == {"default": 4}


def test_worker_lengths_preserve_provenance_without_reading_samples():
    def forbidden(*args):
        raise AssertionError("Length calculation must not read or transform samples")

    pipeline = ml.data(get=forbidden, length=6).shuffle(buffer=2).transform(forbidden).batch(
        4, workers=4, combine=forbidden,
    ).map(forbidden)
    assert len(pipeline) == len(pipeline.build()) == 4
    filtered = ml.data(get=forbidden, length=6).filter(forbidden).shuffle(buffer=2).batch(4, workers=4)
    for value in (filtered, filtered.build()):
        with pytest.raises(TypeError):
            len(value)


class EpochSampler(Sampler):
    def __init__(self):
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        return iter([self.epoch % 5, 4, 4])

    def __len__(self):
        return 3


class Lengthless(Dataset):
    def __getitem__(self, index):
        return index


def test_transform_filter_map_order_and_lazy_configuration():
    calls = []

    def get(index):
        calls.append(index)
        return index

    base = ml.data(get=get, length=6)
    pipeline = base.transform(twice).filter(positive).shuffle().batch(2).map(lambda x: x + 1)
    spec = pipeline.to_spec(lambda value: value)
    restored = ml.DataPipeline.from_spec(spec, lambda value: value)
    assert calls == []
    with pytest.raises(TypeError):
        len(pipeline)
    output = torch.cat(list(restored)).flatten().tolist()
    assert sorted(output) == [3, 5, 7, 9, 11]
    assert calls == list(range(6))
    assert len(base) == 6
    with pytest.raises(ConfigurationError, match="sample|batch"):
        base.batch(2).transform(twice)


def test_default_batch_normalization_is_only_at_output():
    x, y = next(iter(ml.data(get=index_sample, length=3).batch(2)))
    assert x.shape == (2, 2)
    assert y.shape == (2, 1)
    x, y = next(iter(ml.data(get=index_sample, length=3).batch(2, normalize=False)))
    assert x.shape == (2, 2)
    assert y.shape == (2,)
    batch = next(iter(ml.data([{"scalar": 1, "nested": (torch.ones(3), 2)}]).batch(1)))
    assert batch["scalar"].shape == (1, 1)
    assert batch["nested"][0].shape == (1, 3)
    assert batch["nested"][1].shape == (1, 1)


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("source", ["get", "stream", "transform"])
def test_default_collation_unwraps_nested_samples_without_changing_sample_shapes(source, normalize):
    record = namedtuple("Record", "features label")
    leaf = torch.arange(6., requires_grad=True)
    samples = [OrderedDict(item=record(
        ml.wrap((leaf * 2).reshape(3, 2)[index]),
        ml.tensor(index, normalize=False, axes=()),
    ), nested=[ml.wrap(torch.ones(1, 2))],
        column=ml.tensor([index, index + 1])) for index in range(3)]
    # Mixed native/wrapped leaves follow the same collation rules.
    samples[1]["item"] = record(samples[1]["item"].features.raw, samples[1]["item"].label)
    if source == "get":
        pipeline = ml.data(get=samples.__getitem__, length=len(samples))
    elif source == "stream":
        pipeline = ml.data(stream=lambda: iter(samples))
    else:
        pipeline = ml.data(range(len(samples))).transform(samples.__getitem__)
    batch = next(iter(pipeline.batch(3, normalize=normalize)))
    assert isinstance(batch, OrderedDict)
    assert isinstance(batch["item"], record)
    assert isinstance(batch["nested"], list)
    assert batch["item"].features.shape == (3, 2)
    assert batch["item"].label.shape == ((3, 1) if normalize else (3,))
    assert batch["nested"][0].shape == (3, 1, 2)
    assert batch["column"].shape == (3, 2, 1)
    torch.testing.assert_close(batch["item"].features, (leaf * 2).reshape(3, 2))
    batch["item"].features.sum().backward()
    torch.testing.assert_close(leaf.grad, torch.full_like(leaf, 2))


def test_custom_combine_receives_original_wrappers_and_preserves_output():
    samples = [ml.wrap(torch.tensor([1., 2.])), ml.tensor(3.)]
    output = {"scalar": ml.wrap(torch.tensor(4.)), "samples": samples}

    def combine(received):
        assert all(actual is expected for actual, expected in zip(received, samples, strict=True))
        return output

    assert next(iter(ml.data(samples).batch(2, combine=combine))) is output


def test_default_wrapped_sample_collation_runs_in_native_workers(worker_tmp):
    pipeline = ml.data(get=worker_sample, length=5).transform(worker_wrap).batch(2, workers=2)
    batches = list(pipeline)
    assert torch.cat([batch["value"] for batch in batches]).flatten().tolist() == list(range(5))
    for batch in batches:
        assert batch["pid"].shape == (len(batch["value"]), 1)
        assert (batch["pid"] != os.getpid()).all()
        torch.testing.assert_close(batch["pid"], batch["transform_pid"])


def test_custom_collation_shapes_win_and_aliases_reject_conflicts():
    def combine(samples):
        return torch.tensor(samples)

    assert next(iter(ml.data([1, 2]).batch(2, combine=combine))).shape == (2,)
    assert next(iter(ml.data([1, 2]).batch(2, collate_fn=combine))).shape == (2,)
    with pytest.raises(ConfigurationError, match="combine|collate"):
        ml.data([1]).batch(1, combine=combine, collate_fn=combine)
    with pytest.raises(ConfigurationError, match="buffer"):
        ml.data([1]).shuffle(buffer=2, buffer_size=3)


def test_shuffle_epochs_index_order_and_no_sample_materialization():
    calls = []

    def get(index):
        calls.append(index)
        return index

    pipeline = ml.data(get=get, length=30).shuffle()
    built = pipeline.build(2)
    assert calls == []
    first = next(iter(built))
    assert calls == [first]
    assert list(pipeline.build(2)) == list(pipeline.build(2))
    assert list(pipeline.build(2)) != list(pipeline.build(3))
    assert sorted(pipeline) == list(range(30))
    assert list(ml.data(range(30)).shuffle(buffer=4)) == list(ml.data(range(30)).shuffle(buffer_size=4))


def test_filter_scalar_predicates_and_streams():
    import numpy as np

    assert list(ml.data(torch.arange(5)).filter(lambda x: x >= 2)) == [2, 3, 4]
    assert list(ml.data([0, 1, 2]).filter(lambda x: np.bool_(x > 0))) == [1, 2]
    assert list(itertools.islice(ml.data(stream=infinite_stream).filter(lambda x: x % 2 == 0), 5)) == [0, 2, 4, 6, 8]
    for invalid in [lambda x: torch.tensor([True, False]), lambda x: 1, lambda x: "yes",
                    lambda x: np.int64(1), lambda x: np.array(True), lambda x: np.array([True])]:
        with pytest.raises(ConfigurationError, match="boolean|scalar"):
            list(ml.data([1]).filter(invalid))
    bounded = ml.data(stream=infinite_stream).shuffle(buffer=5)
    assert list(itertools.islice(bounded, 10)) == list(itertools.islice(bounded, 10))
    with pytest.raises(UnsupportedFeatureError, match="finite|buffer"):
        ml.data(stream=infinite_stream).shuffle().build()


def test_native_sampler_order_replacement_lengths_and_epoch():
    replacement = torch.utils.data.WeightedRandomSampler([0, 1, 0], 5, replacement=True)
    repeated = ml.data(range(3)).batch(2, sampler=replacement)
    assert len(repeated) == 3
    assert torch.cat(list(repeated)).flatten().tolist() == [1] * 5
    sampler = EpochSampler()
    pipeline = ml.data(range(5)).batch(2, sampler=sampler)
    assert len(pipeline) == 2
    assert torch.cat(list(pipeline.build(2))).flatten().tolist() == [2, 4, 4]
    assert sampler.epoch == 2
    assert len(ml.data(range(5)).batch(2, sampler=sampler, drop_last=True)) == 1
    for pipeline in [ml.data(range(5)), ml.data(range(5)).shuffle()]:
        with pytest.raises(ConfigurationError, match="sampler|shuffle"):
            pipeline.batch(2, sampler=sampler, shuffle=True)
    with pytest.raises(ConfigurationError, match="sampler|shuffle"):
        ml.data(range(5)).shuffle().batch(2, sampler=sampler)
    with pytest.raises(UnsupportedFeatureError, match="sampler|iterable"):
        ml.data(stream=finite_stream).batch(2, sampler=sampler).build()


def test_native_loader_identity_mapping_and_advanced_escape_hatch():
    loader = DataLoader(list(range(6)), batch_sampler=[[4, 1], [3]], collate_fn=torch.tensor)
    assert ml.data(loader).build() is loader
    assert [x.tolist() for x in ml.data(loader).map(twice)] == [[8, 2], [6]]
    assert next(iter(ml.data(loader))).shape == (2,)
    for pipeline in [ml.data(loader).workers(0), ml.data(loader).prefetch(2), ml.data(loader).batch(2)]:
        with pytest.raises(UnsupportedFeatureError, match="DataLoader|native loader"):
            pipeline.build()


def test_batch_empty_partial_drop_last_pin_and_lengthless():
    assert list(ml.data([]).batch(3, shuffle=True)) == []
    assert len(ml.data([]).batch(3)) == 0
    assert list(ml.data([0, 0]).filter(positive).shuffle().batch(2)) == []
    assert [len(x) for x in ml.data(range(5)).batch(2)] == [2, 2, 1]
    assert [len(x) for x in ml.data(range(5)).batch(2, drop_last=True)] == [2, 2]
    loader = ml.data(range(5)).batch(2, pin_memory=True).build()
    assert isinstance(loader, DataLoader) and loader.pin_memory
    assert torch.cat(list(loader)).flatten().tolist() == list(range(5))
    with pytest.raises(UnsupportedFeatureError, match="length|__len__"):
        ml.data(Lengthless()).batch(2).build()


def test_real_workers_execute_get_transform_and_combine_with_native_prefetch(worker_tmp):
    pipeline = ml.data(get=worker_sample, length=5).transform(worker_transform).batch(2, workers=2, prefetch=3, combine=worker_combine)
    loader = pipeline.build()
    assert isinstance(loader, DataLoader)
    assert loader.batch_size == 2 and loader.prefetch_factor == 3
    batches = list(loader)
    assert [x["value"] for b in batches for x in b["samples"]] == list(range(5))
    assert len({b["combine_pid"] for b in batches}) == 2
    for batch in batches:
        assert batch["combine_pid"] != os.getpid()
        assert all(s["pid"] == s["transform_pid"] == batch["combine_pid"] for s in batch["samples"])
    legacy = ml.data(range(5)).batch(2).workers(2, prefetch_factor=3).build()
    assert legacy.batch_size == 2 and legacy.prefetch_factor == 3
    assert torch.cat(list(legacy)).flatten().tolist() == list(range(5))


def test_multiworker_streams_have_no_duplicates_or_omissions_and_cleanup(worker_tmp):
    finite = ml.data(stream=finite_stream).transform(twice).batch(2, workers=2)
    assert sorted(torch.cat(list(finite)).flatten().tolist()) == list(range(0, 22, 2))
    loader = ml.data(stream=infinite_stream).batch(1, workers=2, prefetch=1).build()
    iterator = iter(loader)
    processes = list(iterator._workers)
    try:
        assert sorted(int(next(iterator).item()) for _ in range(20)) == list(range(20))
    finally:
        iterator._shutdown_workers()
    assert all(not process.is_alive() for process in processes)
    with pytest.raises(UnsupportedFeatureError, match="replayable"):
        ml.data(stream=forbidden_stream).batch(2, workers=2).build()


def test_worker_failure_propagates_and_can_cleanup(worker_tmp):
    loader = ml.data(get=failing_sample, length=3).batch(2, workers=1).build()
    iterator = iter(loader)
    processes = list(iterator._workers)
    try:
        with pytest.raises(RuntimeError, match="loading failed deliberately"):
            next(iterator)
    finally:
        iterator._shutdown_workers()
    assert all(not process.is_alive() for process in processes)


@pytest.mark.parametrize("configure", [
    lambda p: p.batch(2, workers=-1),
    lambda p: p.batch(2, prefetch=2),
    lambda p: p.batch(2, workers=2, prefetch=0),
    lambda p: p.batch(2, shuffle=1),
    lambda p: p.batch(2, normalize=1),
    lambda p: p.batch(2, pin_memory=1),
    lambda p: p.batch(2, workers=2).workers(1),
    lambda p: p.workers(1).batch(2, workers=2),
    lambda p: p.batch(2, workers=1, prefetch=2).prefetch(3),
])
def test_invalid_loader_options_are_clear(configure):
    with pytest.raises(ConfigurationError):
        configure(ml.data(range(4))).build()


def test_version_two_roundtrip_and_version_one_shape_compatibility():
    pipeline = ml.data(get=index_sample, length=7).transform(lambda pair: pair).filter(lambda pair: pair[1] > 0).shuffle(seed=17).batch(3, normalize=False)
    spec = pipeline.to_spec(lambda value: value)
    assert spec["version"] == 2
    restored = ml.DataPipeline.from_spec(spec, lambda value: value)
    for left, right in zip(pipeline.build(2), restored.build(2), strict=True):
        assert all(torch.equal(a, b) for a, b in zip(left, right))
    old = ml.data([1, 2, 3]).to_spec(lambda value: value)
    old["version"] = 1
    old["loader"]["workers"] = 0
    old["operations"] = [{"name": "batch", "options": {"size": 2, "drop_last": False, "collate_fn": None}}]
    assert next(iter(ml.DataPipeline.from_spec(old, lambda value: value))).shape == (2,)


def test_version_one_shuffle_keeps_persisted_order():
    spec = ml.data(range(12)).to_spec(lambda value: value)
    spec["version"] = 1
    spec["loader"]["workers"] = 0
    spec["operations"] = [{"name": "shuffle", "options": {"seed": 9, "buffer_size": None}}]
    expected = list(range(12))
    random.Random(11).shuffle(expected)
    restored = ml.DataPipeline.from_spec(spec, lambda value: value)
    assert list(restored.build(2)) == expected
    restored_again = ml.DataPipeline.from_spec(restored.to_spec(lambda value: value), lambda value: value)
    assert list(restored_again.build(2)) == expected


class ShardedNative(IterableDataset):
    def __iter__(self):
        worker = get_worker_info()
        start, step = (worker.id, worker.num_workers) if worker else (0, 1)
        yield from range(start, 7, step)


def even(value):
    return value % 2 == 0


def test_native_iterable_sharding_and_filter_shuffle_in_real_workers(worker_tmp):
    native = ml.data(ShardedNative()).map(twice).batch(2, workers=2)
    assert sorted(torch.cat(list(native)).flatten().tolist()) == list(range(0, 14, 2))
    stream = ml.data(stream=finite_stream).filter(even).shuffle(buffer=3).batch(2, workers=2)
    assert sorted(torch.cat(list(stream)).flatten().tolist()) == list(range(0, 11, 2))
    indexed = ml.data(range(11)).filter(even).shuffle().batch(2, workers=2)
    assert sorted(torch.cat(list(indexed)).flatten().tolist()) == list(range(0, 11, 2))


def test_batch_shuffle_epochs_and_numeric_normalization_are_distinct():
    pipeline = ml.data(torch.arange(30, dtype=torch.float32).reshape(10, 3)).normalize(1, 2, dim=-1).batch(4, shuffle=True)
    first = torch.cat(list(pipeline.build(2)))
    assert first.shape == (10, 3)
    assert torch.equal(first, torch.cat(list(pipeline.build(2))))
    assert not torch.equal(first, torch.cat(list(pipeline.build(3))))
    assert torch.equal(first[:, 0].sort().values, (torch.arange(0, 30, 3) - 1) / 2)


def test_stream_shuffle_reads_only_a_bounded_prefix_and_filtered_length_is_unknown():
    reads = []

    def stream():
        for value in itertools.count():
            reads.append(value)
            yield value

    pipeline = ml.data(stream=stream).filter(even).shuffle(buffer=3).batch(2)
    built = pipeline.build()
    assert reads == []
    with pytest.raises(TypeError):
        len(built)
    batch = next(iter(built))
    assert batch.shape == (2, 1)
    assert len(reads) <= 10


def test_namedtuple_default_collation_and_custom_scalar_keep_shapes():
    from collections import namedtuple

    record = namedtuple("Record", "features label")
    output = next(iter(ml.data([record(torch.ones(4), 1)]).batch(1)))
    assert isinstance(output, record)
    assert output.features.shape == (1, 4)
    assert output.label.shape == (1, 1)
    output = next(iter(ml.data([1, 2]).batch(2, combine=lambda samples: torch.tensor(3))))
    assert output.shape == ()


def test_spec_encoder_keeps_cluster_callable_and_native_restrictions():
    from mlzo.cluster.serialization import serialize_run

    for pipeline in [ml.data(range(4)).transform(lambda x: x), ml.data(range(4)).filter(lambda x: True), ml.data(range(4)).batch(2, sampler=EpochSampler())]:
        with pytest.raises((ConfigurationError, UnsupportedFeatureError)):
            serialize_run(ml.train(ml.Lazy(torch.nn.Linear, 1, 1), pipeline, loss="mse"))


def label_positive(sample):
    return sample[1] > 0


def test_remote_recipe_roundtrip_runs_new_loading_operations():
    from mlzo.cluster.serialization import serialize_run, deserialize_run

    pipeline = ml.data(get=index_sample, length=7).filter(label_positive).shuffle(17).batch(3)
    run = ml.train(ml.Lazy(torch.nn.Linear, 2, 1), pipeline, loss="mse")
    spec = serialize_run(run)
    restored = deserialize_run(spec)
    assert serialize_run(restored) == spec
    for left, right in zip(pipeline.build(2), restored.data.build(2), strict=True):
        assert all(torch.equal(a, b) for a, b in zip(left, right))


def test_explicit_sampler_can_bound_a_lengthless_native_dataset():
    pipeline = ml.data(Lengthless()).batch(2, sampler=[3, 1, 3])
    assert len(pipeline) == 2
    assert torch.cat(list(pipeline)).flatten().tolist() == [3, 1, 3]
