"""Construction runs on the assigned worker, after planning."""

from torch import Tensor, nn


class Classifier(nn.Module):
    def __init__(self, features: int = 4, classes: int = 2) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(features, 8), nn.ReLU(), nn.Linear(8, classes))

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
