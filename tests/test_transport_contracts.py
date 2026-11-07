"""Recipes, identities, and artifact integrity from the implemented plan contracts."""

import os
import tempfile
import unittest
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

import mlzo
from mlzo.cluster.serialization import deserialize_run, serialize_run, validate_run_spec
from mlzo.errors import CheckpointError, ConfigurationError
from mlzo.inference._recipe import decode_arguments, encode_recipe
from mlzo.inference._state import state_dict_copy


class TransportContracts(unittest.TestCase):
    def test_shared_lazy_recipes_round_trip_without_materializing(self):
        shared = mlzo.Lazy(nn.Linear, 2, 2)
        run = mlzo.train(mlzo.Lazy(nn.Sequential, shared, shared), [[1, 2]], loss="mse")
        spec = serialize_run(run)
        restored = deserialize_run(spec)
        self.assertIs(restored.algorithm_input.args[0], restored.algorithm_input.args[1])
        self.assertEqual(restored.algorithm_input.state, "unmaterialized")
        self.assertEqual(shared.state, "unmaterialized")
        self.assertEqual(serialize_run(restored), spec)

    def test_wire_validation_does_not_import_application_factories(self):
        run = mlzo.train(mlzo.Lazy(nn.Linear, 2, 1), [[1, 2]], loss="mse")
        spec = serialize_run(run)
        spec["algorithm_input"]["factory"]["module"] = "mlzo_nonexistent_application"
        validate_run_spec(spec)
        with self.assertRaises(ConfigurationError):
            deserialize_run(spec)

    def test_lazy_cycle_is_rejected_before_reusing_its_recipe_id(self):
        model = mlzo.Lazy(nn.Sequential)
        model.args = (model,)
        run = mlzo.train(model, [], loss="mse")
        with self.assertRaises(ConfigurationError) as caught:
            serialize_run(run)
        self.assertEqual(
            str(caught.exception),
            "run.algorithm_input.args[0]: Cyclic recipes cannot be transported; "
            "put construction in an importable factory.",
        )

    def test_validation_requires_complete_and_unique_lazy_recipes(self):
        model = mlzo.Lazy(nn.Sequential, mlzo.Lazy(nn.Linear, 2, 1))
        spec = serialize_run(mlzo.train(model, [], loss="mse"))
        incomplete = deepcopy(spec)
        incomplete["algorithm_input"]["args"]["items"][0] = {"__mlzo__": "lazy_ref", "id": "l0"}
        duplicate = deepcopy(spec)
        duplicate["algorithm_input"]["args"]["items"][0]["id"] = "l0"
        cases = (
            (incomplete, "run.algorithm_input.args[0]: Lazy references must refer to an earlier complete recipe."),
            (duplicate, "run.algorithm_input: Duplicate nested Lazy recipe identity."),
        )
        for malformed, message in cases:
            with self.subTest(message=message), self.assertRaises(ConfigurationError) as caught:
                validate_run_spec(malformed)
            self.assertEqual(str(caught.exception), message)

    def test_local_factory_error_keeps_its_recipe_path(self):
        run = mlzo.train(mlzo.Lazy(lambda: nn.Linear(2, 1)), [[1, 2]], loss="mse")
        with self.assertRaisesRegex(ConfigurationError, r"run.algorithm_input.factory"):
            serialize_run(run)

    def test_data_paths_resolve_in_the_worker_project(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            original = root / "records.jsonl"
            original.write_text('{"x": 1}\n')
            batches = mlzo.data(original).map(dict).shuffle(seed=3).batch(2)
            run = mlzo.train(mlzo.Lazy(nn.Linear, 2, 1), batches, loss="mse")
            relative = ".mlzo/data/0000/records.jsonl"
            spec = serialize_run(run, data_paths={original: relative})
            previous = Path.cwd()
            try:
                os.chdir(root)
                restored = deserialize_run(spec)
            finally:
                os.chdir(previous)
            self.assertEqual(restored.data.local_paths(), (root / relative,))
            self.assertEqual([operation.name for operation in restored.data._operations], ["map", "shuffle", "batch"])

    def test_inference_arguments_escape_user_tags_and_preserve_tuples(self):
        arguments = (2, 1, {"type": "tuple", "items": [("x", {"type": "dict"})]})
        keywords = {"options": {"nested": [None, True, 0.5, ("a", "b")]}}
        encoded = encode_recipe({"factory": nn.Linear, "args": arguments, "kwargs": keywords})
        self.assertEqual(decode_arguments(encoded), (arguments, keywords))
        with self.assertRaises(ConfigurationError):
            encode_recipe({"factory": nn.Linear, "args": (float("nan"),), "kwargs": {}})
        cyclic = []
        cyclic.append(cyclic)
        with self.assertRaises(ConfigurationError):
            encode_recipe({"factory": nn.Linear, "args": (cyclic,), "kwargs": {}})

    def test_inference_state_copy_preserves_aliases_metadata_and_detaches(self):
        tensor = torch.tensor([1.0, 2.0], requires_grad=True)
        shared = [tensor]
        state = OrderedDict(weight=tensor, alias=tensor, children=(shared, shared))
        state._metadata = {"shared": shared}
        for freeze in (True, False):
            with self.subTest(freeze=freeze):
                copied = state_dict_copy(state, freeze=freeze)
                self.assertIs(type(copied), OrderedDict)
                self.assertEqual(list(copied), list(state))
                self.assertIs(copied["weight"], copied["alias"])
                self.assertIs(copied["weight"], copied["children"][0][0])
                self.assertIs(copied["children"][0], copied["children"][1])
                self.assertIs(copied._metadata["shared"], copied["children"][0])
                self.assertIsNot(copied["children"][0], shared)
                self.assertFalse(copied["weight"].requires_grad)
                self.assertTrue(tensor.requires_grad)
                torch.testing.assert_close(copied["weight"], tensor)
                self.assertEqual(copied["weight"].data_ptr() == tensor.data_ptr(), not freeze)

    def test_inference_state_copy_rejects_container_and_metadata_cycles(self):
        cyclic = []
        cyclic.append(cyclic)
        metadata_cycle = OrderedDict(weight=torch.ones(1))
        metadata_cycle._metadata = {"root": metadata_cycle}
        for state in ({"cycle": cyclic}, metadata_cycle):
            for freeze in (True, False):
                with self.subTest(freeze=freeze, metadata=type(state) is OrderedDict):
                    with self.assertRaises(ConfigurationError) as caught:
                        state_dict_copy(state, freeze=freeze)
                    self.assertEqual(
                        str(caught.exception),
                        "Cyclic state_dict values are unsupported for inference export.",
                    )

    def test_planning_is_deferred_and_materialization_preserves_aliases(self):
        calls = []

        def model_factory():
            calls.append("model")
            return nn.Linear(2, 2)

        def data_factory():
            calls.append("data")
            return []

        shared = mlzo.Lazy(model_factory)
        model = nn.Sequential(shared, shared)
        run = mlzo.train(model, mlzo.data(data_factory), loss="mse")
        run.plan(on="cpu")
        self.assertEqual(calls, [])
        concrete = mlzo.materialize(model, "cpu")
        self.assertIs(concrete[0], concrete[1])
        self.assertIs(mlzo.materialize(model, "cpu"), concrete)
        self.assertEqual(calls, ["model"])

    def test_checkpoint_corruption_is_rejected_before_state_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = mlzo.save_snapshot({"weight": torch.ones(2)}, to=directory, run_id="integrity")
            state = Path(reference.path) / "state.pt"
            content = bytearray(state.read_bytes())
            content[-1] ^= 1
            state.write_bytes(content)
            with self.assertRaisesRegex(CheckpointError, "SHA256"):
                mlzo.load_snapshot(reference)

    def test_named_axes_preserve_the_pytorch_gradient(self):
        values = torch.arange(6, dtype=torch.float64).reshape(2, 3).requires_grad_()
        reference = values.detach().clone().requires_grad_()
        named = mlzo.tensor(values, axes=("batch", "features"))
        named.align("features", "batch").mean("batch").raw.square().sum().backward()
        reference.transpose(0, 1).mean(dim=1).square().sum().backward()
        torch.testing.assert_close(values.grad, reference.grad, rtol=0, atol=0)
        self.assertIs(mlzo.grad, torch.func.grad)


if __name__ == "__main__":
    unittest.main()
