#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn.functional as F


def binary_focal_loss_with_logits(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = -1,
        gamma: float = 2,
        reduction: str = "none",
    ) -> torch.Tensor:

    """
    The original implementations are as follows:
        https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
        https://pytorch.org/vision/0.15/_modules/torchvision/ops/focal_loss.html#sigmoid_focal_loss
    """

    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )

    return loss


def weighted_binary_focal_loss_with_logits(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weight: torch.Tensor,
        alpha: float = -1,
        gamma: float = 2,
        reduction: str = "none",
    ) -> torch.Tensor:

    """
    This is re-implementation of the following paper's loss function:
        N. Sarafianos, et al., "Deep Imbalanced Attribute Classification using Visual Attention Aggregation," in ECCV, 2017.

    This implementation is based on the following loss functions:
        https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
        https://pytorch.org/vision/0.15/_modules/torchvision/ops/focal_loss.html#sigmoid_focal_loss
    """

    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    loss = torch.mul(torch.exp(-weight), loss)

    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )

    return loss
