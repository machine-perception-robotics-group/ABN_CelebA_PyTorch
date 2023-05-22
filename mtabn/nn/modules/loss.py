#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
from torch.nn import Module
from .. import functional as F

from torch import Tensor
from typing import Optional


__all__ = ['BFLossWithLogits', 'WeightedBFLossWithLogits']


class BFLossWithLogits(Module):
    """Binary Focal Loss with Logits"""

    def __init__(self, alpha: float = -1, gamma: float = 2, reduction: str = 'mean') -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_focal_loss_with_logits(input, target,
                                               self.alpha,
                                               self.gamma,
                                               self.reduction)


class WeightedBFLossWithLogits(Module):
    """Weighted Binary Focal Loss with Logits"""

    def __init__(self, freq_hist: Tensor, alpha: float = -1, gamma: float = 2, reduction: str = 'mean') -> None:
        super().__init__()

        _weight = torch.exp(-freq_hist)
        _weight = _weight * (_weight.size(0) / torch.sum(_weight))
        self.register_buffer('weight', _weight)
        self.weight: Optional[Tensor]

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.weighted_binary_focal_loss_with_logits(input, target,
                                                        self.weight,
                                                        self.alpha,
                                                        self.gamma,
                                                        self.reduction)
