"""A synthetic CPU classifier. Run: python examples/local_classifier.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import TensorDataset

import mlzo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/local-classifier"))
    args = parser.parse_args()

    generator = torch.Generator().manual_seed(7)
    x = torch.randn(64, 4, generator=generator)
    y = (x.sum(dim=1) > 0).long()
    batches = mlzo.data(TensorDataset(x, y)).shuffle(seed=7).batch(16)
    run = mlzo.train(
        mlzo.Lazy(nn.Linear, 4, 2), batches, loss="cross_entropy",
        optimizer=mlzo.opt("sgd", lr=0.1), precision="fp32", seed=7,
        checkpoint=mlzo.checkpoint(args.output / "checkpoints", every=4),
    )
    print(run.plan(on="cpu").explain())
    run.fit(epochs=2, on="cpu")
    print("Status:", run.status().to_dict())
    print("Evaluation:", run.evaluate(batches))
    print("Checkpoint:", run.checkpoint().to_dict())
    artifact = run.export(to=args.output / "artifact")
    print("Artifact:", artifact.to_dict())
    with mlzo.deploy(artifact, device="cpu") as predictor:
        print("Predicted classes:", predictor.predict(x[:4]).argmax(dim=1).tolist())


if __name__ == "__main__":
    main()
