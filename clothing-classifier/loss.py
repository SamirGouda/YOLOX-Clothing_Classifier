#! /usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math
from itertools import permutations

class CrossEntropyLoss(nn.Module):
    def __init__(self, reduction: str='none') -> None:
        super().__init__()

        # reduction='none' gives loss per sample
        self.loss_func = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor,
                weights: Optional[torch.Tensor] = None):
        if weights is not None:
            raise NotImplementedError
        loss = self.loss_func(outputs, targets)

        return loss


class LogSoftmaxWrapper(nn.Module):
    """
    Arguments
    ---------
    Returns
    ---------
    loss : torch.Tensor : Learning loss
    predictions : torch.Tensor : Log probabilities
    Example
    -------
    >>> outputs = torch.tensor([ [1., -1.], [-1., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> outputs = outputs.unsqueeze(1)
    >>> targets = torch.tensor([ [0], [1], [0], [1] ])
    >>> log_prob = LogSoftmaxWrapper(nn.Identity())
    >>> loss = log_prob(outputs, targets)
    >>> 0 <= loss < 1
    tensor(True)
    >>> log_prob = LogSoftmaxWrapper(AngularMargin(margin=0.2, scale=32))
    >>> loss = log_prob(outputs, targets)
    >>> 0 <= loss < 1
    tensor(True)
    >>> outputs = torch.tensor([ [1., -1.], [-1., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> log_prob = LogSoftmaxWrapper(AdditiveAngularMargin(margin=0.3, scale=32))
    >>> loss = log_prob(outputs, targets)
    >>> 0 <= loss < 1
    tensor(True)
    """

    def __init__(self, loss_fn):
        super(LogSoftmaxWrapper, self).__init__()
        self.loss_fn = loss_fn
        self.criterion = torch.nn.KLDivLoss(reduction="sum")

    def forward(self, outputs, targets):
        """
        Arguments
        ---------
        outputs : torch.Tensor
            Network output tensor, of shape
            [batch, 1, outdim].
        targets : torch.Tensor
            Target tensor, of shape [batch, 1].

        Returns
        -------
        loss: torch.Tensor
            Loss for current examples.
        """
        outputs = outputs.squeeze(1)
        # targets = targets.squeeze(1)
        targets = F.one_hot(targets.long(), outputs.shape[1]).float()
        try:
            predictions = self.loss_fn(outputs, targets)
        except TypeError:
            predictions = self.loss_fn(outputs)

        predictions = F.log_softmax(predictions, dim=1)
        loss = self.criterion(predictions, targets) / targets.sum()
        return loss
