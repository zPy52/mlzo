"""Declarative and functional algorithms share the class-based execution path."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from mlzo.errors import ConfigurationError

from .types import Step, Training


def _sum(value: torch.Tensor) -> torch.Tensor:
    return value.sum(dtype=torch.float32) if value.dtype in {torch.float16, torch.bfloat16} else value.sum()


def builtin_loss(prediction: torch.Tensor, target: torch.Tensor, loss: Any) -> Step | None:
    """Return a numerator-derived mean and the actual PyTorch denominator.

    Only built-in cross entropy adapts integral class indices from (B, 1) to
    (B,) for plain (B, C) logits. Dense targets, probability targets and other
    losses keep their shapes and dtypes. Custom losses bypass this policy.
    """
    reduction = "mean" if isinstance(loss, str) else getattr(loss, "reduction", None)
    if isinstance(loss, (str, nn.CrossEntropyLoss)) and (not isinstance(loss, str) or loss == "cross_entropy"):
        if (
            prediction.ndim == 2
            and target.shape == (prediction.shape[0], 1)
            and target.dtype in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
        ):
            target = target.squeeze(1)
        weight = None if isinstance(loss, str) else loss.weight
        ignore_index = -100 if isinstance(loss, str) else loss.ignore_index
        smoothing = 0.0 if isinstance(loss, str) else loss.label_smoothing
        if target.is_floating_point() and target.shape == prediction.shape:
            denominator = target.numel() / (prediction.shape[1] if prediction.ndim > 1 else prediction.shape[0])
        else:
            valid = target != ignore_index
            if weight is None:
                denominator = float(valid.sum().item())
            else:
                indices = target[valid]
                # uint8 indexes act as masks. Cast only this weight lookup;
                # cross_entropy still receives the caller's target dtype.
                if indices.dtype == torch.uint8:
                    indices = indices.long()
                denominator = float(_sum(weight[indices].detach()).item())
        if denominator == 0 and reduction in {"mean", "sum"}:
            return Step(loss=prediction.reshape(-1)[:0].sum(), weight=0.0)
        numerator = F.cross_entropy(
            prediction,
            target,
            weight=weight,
            ignore_index=ignore_index,
            reduction="sum",
            label_smoothing=smoothing,
        )
    elif isinstance(loss, nn.MSELoss) or (isinstance(loss, str) and loss == "mse"):
        raw = F.mse_loss(prediction, target, reduction="none")
        numerator, denominator = _sum(raw), float(raw.numel())
    elif isinstance(loss, nn.L1Loss) or (isinstance(loss, str) and loss == "l1"):
        raw = F.l1_loss(prediction, target, reduction="none")
        numerator, denominator = _sum(raw), float(raw.numel())
    elif isinstance(loss, nn.BCELoss) or (isinstance(loss, str) and loss == "bce"):
        raw = F.binary_cross_entropy(
            prediction,
            target,
            weight=getattr(loss, "weight", None),
            reduction="none",
        )
        numerator, denominator = _sum(raw), float(raw.numel())
    elif isinstance(loss, nn.BCEWithLogitsLoss) or (isinstance(loss, str) and loss in {"bce_with_logits", "bce_logits"}):
        raw = F.binary_cross_entropy_with_logits(
            prediction,
            target,
            weight=getattr(loss, "weight", None),
            pos_weight=getattr(loss, "pos_weight", None),
            reduction="none",
        )
        numerator, denominator = _sum(raw), float(raw.numel())
    else:
        return None
    if denominator == 0 and reduction in {"mean", "sum"}:
        return Step(loss=numerator, weight=0.0)
    if reduction == "sum":
        return Step(loss=numerator, weight=1.0)
    if reduction != "mean":
        raise ConfigurationError("Built-in losses require reduction='mean' or 'sum'; use a custom scalar step for reduction='none'.")
    if denominator == 0:
        return Step(loss=numerator * 0.0, weight=0.0)
    return Step(loss=numerator / denominator, weight=denominator)


class DeclarativeTraining(Training[Any]):
    automatic = True

    def __init__(self, model: nn.Module, loss: Any) -> None:
        super().__init__(model)
        self.loss = loss

    def step(self, batch: Any) -> torch.Tensor | Step:
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise ConfigurationError("Declarative training expects (inputs, targets); use step= for other batch structures.")
        inputs, targets = batch
        prediction = self.model(inputs)
        result = builtin_loss(prediction, targets, self.loss)
        if result is not None:
            return result
        if not callable(self.loss):
            raise ConfigurationError(f"Unknown loss {self.loss!r}.")
        return self.loss(prediction, targets)

    def evaluate(self, batch: Any) -> torch.Tensor | Step:
        return self.step(batch)


class FunctionalTraining(Training[Any]):
    automatic = True

    def __init__(self, model: nn.Module, function: Any) -> None:
        super().__init__(model)
        self.function = function

    def step(self, batch: Any) -> torch.Tensor | Step | None:
        return self.function(self.model, batch)
