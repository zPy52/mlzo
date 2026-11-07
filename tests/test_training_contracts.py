"""CPU references for the implemented training contracts in plan.md sections 15-16."""

import copy
import functools
import inspect
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import mlzo
from mlzo.errors import ConfigurationError, UnsupportedFeatureError
from mlzo.planning import pipeline
from mlzo.training.schedules import Schedule


class SupervisedTraining(mlzo.Training):
    def step(self, batch):
        inputs, targets = batch
        return F.mse_loss(self.model(inputs), targets)


class TrainingContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.model = nn.Linear(3, 2).double()
        self.inputs = torch.arange(18, dtype=torch.float64).reshape(6, 3) / 10
        self.targets = torch.linspace(-1, 1, 12, dtype=torch.float64).reshape(6, 2)
        self.batches = list(zip(self.inputs.split(2), self.targets.split(2)))

    def assert_state_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_state_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for item, reference in zip(actual, expected):
                self.assert_state_equal(item, reference)
        else:
            self.assertEqual(actual, expected)

    def test_three_training_styles_match_pytorch_adamw(self):
        reference = copy.deepcopy(self.model)
        optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)
        for inputs, targets in self.batches:
            optimizer.zero_grad(set_to_none=True)
            F.mse_loss(reference(inputs), targets).backward()
            optimizer.step()
        for style in ("declarative", "functional", "class"):
            with self.subTest(style=style):
                model = copy.deepcopy(self.model)
                options = {}
                algorithm = model
                if style == "declarative":
                    options["loss"] = "mse"
                elif style == "functional":
                    options["step"] = lambda module, batch: F.mse_loss(module(batch[0]), batch[1])
                else:
                    algorithm = SupervisedTraining(model)
                run = mlzo.train(algorithm, self.batches, optimizer=mlzo.opt("adamw", lr=0.01), **options)
                run.fit(epochs=1, on="cpu")
                self.assert_state_equal(model.state_dict(), reference.state_dict())
                self.assert_state_equal(run._engine.optimizers["default"].state_dict(), optimizer.state_dict())

    def test_unequal_accumulation_and_short_window_match_full_batches(self):
        reference = copy.deepcopy(self.model)
        optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        for inputs, targets in ((self.inputs[:4], self.targets[:4]), (self.inputs[4:], self.targets[4:])):
            optimizer.zero_grad(set_to_none=True)
            F.mse_loss(reference(inputs), targets).backward()
            optimizer.step()
        batches = [(self.inputs[:1], self.targets[:1]), (self.inputs[1:4], self.targets[1:4]),
                   (self.inputs[4:], self.targets[4:])]
        for weighted_step in (False, True):
            with self.subTest(weighted_step=weighted_step):
                model = copy.deepcopy(self.model)
                if weighted_step:
                    def step(module, batch):
                        inputs, targets = batch
                        return mlzo.Step(loss=F.mse_loss(module(inputs), targets), weight=targets.numel())
                    options = {"step": step}
                else:
                    options = {"loss": "mse"}
                run = mlzo.train(model, batches, optimizer=mlzo.opt("sgd", lr=0.1), accumulate=2, **options)
                history = run.fit(epochs=1, on="cpu")
                for actual, expected in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                self.assertEqual(history.counters["optimizer_steps"], {"default": 2})

    def test_zero_weight_skips_optimizer_and_scheduler(self):
        run = mlzo.train(copy.deepcopy(self.model), [(self.inputs[:2], torch.full((2,), -100))],
                         loss="cross_entropy", optimizer=mlzo.opt("sgd", lr=0.1),
                         schedule=mlzo.linear(total_steps=2))
        history = run.fit(epochs=1, on="cpu")
        self.assert_state_equal(run._engine.model.state_dict(), self.model.state_dict())
        self.assertEqual(history.counters["optimizer_steps"], {"default": 0})
        self.assertEqual(history.counters["skipped_updates"], {"default": 1})
        self.assertEqual(run._engine.schedulers["default"].successful, 0)

    def test_schedule_forms_keep_update_indexing(self):
        cases = [
            (None, [2, 2, 2, 2]),
            (mlzo.linear(warmup=2, total_steps=4), [1, 2, 1, 0]),
            (mlzo.cosine(total_steps=4), [2, 1.5, 0.5, 0]),
            (mlzo.exponential(gamma=0.5), [2, 1, 0.5, 0.25]),
            (lambda index: 1 / index, [2, 1, 2 / 3, 0.5]),
            (lambda index, horizon: (horizon - index + 1) / horizon, [2, 1.5, 1, 0.5]),
            (functools.partial(torch.optim.lr_scheduler.StepLR, step_size=1, gamma=0.5), [2, 1, 0.5, 0.25]),
        ]
        for source, expected in cases:
            with self.subTest(source=source):
                parameter = nn.Parameter(torch.ones(1))
                optimizer = torch.optim.SGD([parameter], lr=2)
                schedule = Schedule(optimizer, source, 4)
                for rate in expected:
                    self.assertAlmostEqual(optimizer.param_groups[0]["lr"], rate)
                    parameter.grad = torch.zeros_like(parameter)
                    optimizer.step()
                    schedule.advance()

    def test_schedule_inspects_callable_signature_once(self):
        for multiplier in (lambda index: 1 / index, lambda index, horizon: 1 / index):
            class Multiplier:
                inspections = 0

                @property
                def __signature__(self):
                    self.inspections += 1
                    return inspect.signature(multiplier)

                def __call__(self, *args):
                    return multiplier(*args)

            with self.subTest(multiplier=multiplier):
                source = Multiplier()
                parameter = nn.Parameter(torch.ones(1))
                optimizer = torch.optim.SGD([parameter], lr=2)
                schedule = Schedule(optimizer, source, 4)
                self.assertEqual(source.inspections, 1)
                self.assertEqual(optimizer.param_groups[0]["lr"], 2)
                optimizer.step()
                schedule.advance()
                self.assertEqual(optimizer.param_groups[0]["lr"], 1)
                self.assertEqual(source.inspections, 1)

    def test_schedule_rejections_preserve_validation_order(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=2)
        other = torch.optim.SGD(copy.deepcopy(self.model).parameters(), lr=2)
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau
        cases = [
            (mlzo.linear(), None, ConfigurationError, "Linear/cosine schedules require"),
            (mlzo.linear(warmup=3, total_steps=2), None, ConfigurationError, "warmup cannot exceed"),
            (lambda: 1, None, ConfigurationError, "Callable multipliers require"),
            (lambda: 1, 4, ConfigurationError, "Schedule must accept"),
            (object(), 4, ConfigurationError, "schedule requires"),
            (torch.optim.lr_scheduler.StepLR(other, step_size=1), 4, ConfigurationError, "bound optimizer"),
            (lambda opt: object(), 4, ConfigurationError, "bound optimizer"),
            (lambda opt: type("InvalidScheduler", (), {"optimizer": opt})(),
             4, ConfigurationError, "must return a PyTorch scheduler"),
            (plateau(other), 4, UnsupportedFeatureError, "Metric-driven schedulers"),
            (plateau, None, UnsupportedFeatureError, "Metric-driven schedulers"),
        ]
        for source, horizon, error, message in cases:
            with self.subTest(source=source), self.assertRaisesRegex(error, message):
                Schedule(optimizer, source, horizon)

    def test_schedule_restore_keeps_next_rate_and_definition_and_horizon_checks(self):
        factory = functools.partial(torch.optim.lr_scheduler.StepLR, step_size=1, gamma=0.5)
        sources = [mlzo.linear(total_steps=4), mlzo.cosine(), lambda index: 1 / index,
                   lambda index, horizon: 1 / index, factory, "native_instance"]
        for source in sources:
            with self.subTest(source=source):
                optimizers = [torch.optim.SGD([nn.Parameter(torch.ones(1))], lr=2) for _ in range(2)]
                schedules = [Schedule(opt, factory(opt) if source == "native_instance" else source, 4)
                             for opt in optimizers]
                original, restored = schedules
                for _ in range(2):
                    optimizers[0].step()
                    original.advance()
                restored.load_state_dict(original.state_dict())
                self.assert_state_equal(restored.state_dict(), original.state_dict())
                for opt, schedule in zip(optimizers, schedules):
                    opt.step()
                    schedule.advance()
                self.assert_state_equal(restored.state_dict(), original.state_dict())
                for field, value, message in (("definition", {}, "definition differs"),
                                              ("horizon", 5, "horizon differs"),
                                              ("inferred_horizon", not restored.inferred_horizon, "horizon differs")):
                    if field == "horizon" and source in (factory, "native_instance"):
                        continue
                    state = restored.state_dict()
                    with self.assertRaisesRegex(ConfigurationError, message):
                        restored.load_state_dict({**state, field: value})
                    self.assert_state_equal(restored.state_dict(), state)

    def test_evaluation_restores_mixed_modes_after_exception(self):
        model = nn.Sequential(copy.deepcopy(self.model), nn.Dropout())
        run = mlzo.train(model, self.batches, loss="mse")
        run.fit(steps=1, on="cpu")
        model[1].eval()
        modes = [module.training for module in model.modules()]

        def fail(module, batch):
            self.assertFalse(any(child.training for child in module.modules()))
            self.assertFalse(torch.is_grad_enabled())
            raise RuntimeError("evaluator failed")

        with self.assertRaisesRegex(RuntimeError, "evaluator failed"):
            run.evaluate(self.batches, step=fail)
        self.assertEqual([module.training for module in model.modules()], modes)

    def test_checkpoint_resume_matches_uninterrupted_weights_optimizer_and_rng(self):
        initial = nn.Sequential(copy.deepcopy(self.model), nn.Dropout(0.2))

        def learning():
            return mlzo.train(copy.deepcopy(initial), self.batches, loss="mse", seed=17,
                              optimizer=mlzo.opt("adamw", lr=0.01), schedule=mlzo.cosine(total_steps=6))

        continuous = learning()
        continuous.fit(epochs=2, on="cpu")
        interrupted = learning()
        interrupted.fit(epochs=1, on="cpu")
        with tempfile.TemporaryDirectory() as directory:
            snapshot = interrupted.checkpoint(to=Path(directory) / "state")
            resumed = learning()
            resumed.fit(epochs=2, on="cpu", resume=snapshot)
        self.assert_state_equal(resumed._engine.model.state_dict(), continuous._engine.model.state_dict())
        self.assert_state_equal(resumed._engine.optimizers["default"].state_dict(),
                                continuous._engine.optimizers["default"].state_dict())
        self.assertEqual(resumed.status().counters, continuous.status().counters)
        self.assert_state_equal(resumed._engine.schedulers["default"].state_dict(),
                                continuous._engine.schedulers["default"].state_dict())

    def test_named_updates_complete_before_the_next_forward(self):
        class Alternating(mlzo.Training):
            step_driven = True

            def step(self, batch):
                self.update("d", self.model["d"].weight.square().sum())
                self.state = float(self.model["d"].weight.detach().item())
                self.update("g", (self.model["g"].weight * self.model["d"].weight.detach()).square().sum())

        model = nn.ModuleDict({name: nn.Linear(1, 1, bias=False).double() for name in ("d", "g")})
        for parameter in model.parameters():
            nn.init.ones_(parameter)
        algorithm = Alternating(model)
        run = mlzo.train(algorithm, optimizer={name: mlzo.opt("sgd", params=name, lr=0.1) for name in model})
        history = run.fit(steps=1, on="cpu")
        self.assertAlmostEqual(algorithm.state, 0.8)
        self.assertAlmostEqual(model["g"].weight.item(), 0.872)
        self.assertEqual(history.counters["optimizer_steps"], {"d": 1, "g": 1})
        self.assertEqual(history.counters["iteration"], 1)

    def test_deferred_features_still_fail_explicitly(self):
        run = mlzo.train(copy.deepcopy(self.model), self.batches, loss="mse")
        actions = [lambda: run.plan(parallel="ddp"), lambda: run.plan(parallel=pipeline(stages=2)),
                   lambda: mlzo.checkpoint("s3://bucket/state"), lambda: mlzo.deploy(None, on=object()),
                   lambda: mlzo.cluster.aws()]
        for action in actions:
            with self.subTest(action=action), self.assertRaises(UnsupportedFeatureError):
                action()

    def test_stop_flushes_the_short_window_before_checkpointing(self):
        calls = 0

        def step(module, batch):
            nonlocal calls
            calls += 1
            inputs, targets = batch
            return mlzo.Step(loss=F.mse_loss(module(inputs), targets),
                             weight=targets.numel(), stop=calls == 2)

        with tempfile.TemporaryDirectory() as directory:
            run = mlzo.train(copy.deepcopy(self.model), self.batches, step=step, accumulate=3,
                             checkpoint=mlzo.checkpoint(directory, every=1))
            history = run.fit(steps=5, on="cpu")
            self.assertEqual(history.counters["iteration"], 2)
            self.assertEqual(history.counters["optimizer_steps"], {"default": 1})
            snapshots = mlzo.list_snapshots(directory)
            self.assertEqual(len(snapshots), 1)
            self.assertEqual(mlzo.load_snapshot(snapshots[0])["counters"], history.counters)

    def test_cancel_flushes_accumulation_before_the_terminal_event(self):
        run = mlzo.train(copy.deepcopy(self.model), self.batches, loss="mse", accumulate=3)
        events = []

        def report(event):
            events.append(event)
            if event["type"] == "iteration":
                run.cancel()

        run._event_callback = report
        history = run.fit(steps=5, on="cpu")
        self.assertEqual(history.state, "cancelled")
        self.assertEqual(history.counters["iteration"], 1)
        self.assertEqual(history.counters["optimizer_steps"], {"default": 1})
        self.assertEqual([event["type"] for event in events], ["prepared", "running", "iteration", "cancelled"])
        self.assertEqual(events[-2]["counters"]["update_events"], 0)
        self.assertEqual(events[-1]["counters"]["update_events"], 1)

    def test_double_update_failure_preserves_completed_updates_and_blocks_snapshot(self):
        class InvalidUpdate(mlzo.Training):
            step_driven = True

            def step(self, batch):
                self.update("default", self.model.weight.square().sum())
                return self.model.weight.square().sum()

        run = mlzo.train(InvalidUpdate(copy.deepcopy(self.model)))
        with self.assertRaisesRegex(mlzo.TrainingError, "also returned an automatic loss"):
            run.fit(steps=1, on="cpu")
        self.assertEqual(run.status().state, "failed")
        self.assertEqual(run.status().counters["iteration"], 0)
        self.assertEqual(run.status().counters["optimizer_steps"], {"default": 1})
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(mlzo.TrainingError, "failed iteration"):
                run.checkpoint(to=directory)


if __name__ == "__main__":
    unittest.main()
