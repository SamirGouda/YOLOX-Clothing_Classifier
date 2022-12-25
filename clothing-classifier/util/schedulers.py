"""
Schedulers for updating hyperparameters (such as learning rate).
"""

import math
import torch


class LinearScheduler:
    """Scheduler with linear annealing technique.

    The learning rate linearly decays over the specified number of epochs.

    Arguments
    ---------
    initial_value : float
        The value upon initialization.
    final_value : float
        The value used when the epoch count reaches ``epoch_count - 1``.
    epoch_count : int
        Number of epochs.

    Example
    -------
    >>> scheduler = LinearScheduler(1.0, 0.0, 4)
    >>> scheduler(current_epoch=1)
    (1.0, 0.666...)
    >>> scheduler(current_epoch=2)
    (0.666..., 0.333...)
    >>> scheduler(current_epoch=3)
    (0.333..., 0.0)
    >>> scheduler(current_epoch=4)
    (0.0, 0.0)
    """

    def __init__(self, initial_value, final_value, epoch_count):
        self.value_at_epoch = torch.linspace(
            initial_value, final_value, steps=epoch_count
        ).tolist()

    def __call__(self, current_epoch):
        """Returns the current and new value for the hyperparameter.

        Arguments
        ---------
        current_epoch : int
            Number of times the dataset has been iterated.
        """
        old_index = max(0, current_epoch - 1)
        index = min(current_epoch, len(self.value_at_epoch) - 1)
        return self.value_at_epoch[old_index], self.value_at_epoch[index]


