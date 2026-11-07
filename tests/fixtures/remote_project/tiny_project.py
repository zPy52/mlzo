"""Importable, dependency-free submission project for the loopback test."""

import torch
from torch import nn


def model():
    return nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))


def samples():
    x = torch.tensor([[0.2, 0.4], [-0.5, 0.3], [0.8, -0.2]])
    y = torch.tensor([[0.7], [-0.1], [0.4]])
    return torch.utils.data.TensorDataset(x, y)


def broken_model():
    raise RuntimeError("deliberate remote construction failure")
