"""Use a named optimizer explicitly. Run: python examples/explicit_updates.py."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

import mlzo


class ClassifierTraining(mlzo.Training[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self) -> None:
        super().__init__(mlzo.Lazy(nn.Linear, 4, 2))
        self.state = {"examples_seen": 0}

    def step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> mlzo.Step:
        x, y = batch
        # Default batches have column labels; native cross entropy needs (B,).
        y = y.squeeze(1)
        loss = F.cross_entropy(self.model(x), y)
        self.update("classifier", loss)
        self.state["examples_seen"] += len(x)
        # Returning a second optimizable loss would request an invalid double update.
        return mlzo.Step(metrics={"loss": loss.detach()})

    def evaluate(self, batch: tuple[torch.Tensor, torch.Tensor]) -> mlzo.Step:
        x, y = batch
        y = y.squeeze(1)
        return mlzo.Step(loss=F.cross_entropy(self.model(x), y))


def main() -> None:
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(32, 4, generator=generator)
    y = (x[:, 0] > 0).long()
    batches = mlzo.data(x, y).batch(8)
    algorithm = ClassifierTraining()
    run = mlzo.train(algorithm, batches,
                     optimizer={"classifier": mlzo.opt("sgd", lr=0.1)},
                     precision="fp32", seed=11)
    run.fit(steps=8, on="cpu")
    print(run.status().to_dict())
    print(run.evaluate(batches))
    print(algorithm.state)


if __name__ == "__main__":
    main()
