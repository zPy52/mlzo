"""An importable, zero-argument factory with no file or network dependency."""

import torch
from torch.utils.data import TensorDataset


def samples() -> TensorDataset:
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(64, 4, generator=generator)
    y = (x.sum(dim=1) > 0).long()
    return TensorDataset(x, y)
