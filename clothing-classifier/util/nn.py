#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from tracemalloc import is_tracing

from numpy import pad
sys.path.insert(0, '/media/asr9/HDD/kateb/scripts/pytorch')
import os
from pathlib import Path
from typing import List
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from processing.transformations import Fbank
import math
from typing import Tuple, List

class Linear(torch.nn.Module):
    """Computes a linear transformation y = wx + b.

    Arguments
    ---------
    n_neurons : int
        It is the number of output neurons (i.e, the dimensionality of the
        output).
    input_shape: tuple
        It is the shape of the input tensor.
    input_size: int
        Size of the input tensor.
    bias : bool
        If True, the additive bias b is adopted.
    combine_dims : bool
        If True and the input is 4D, combine 3rd and 4th dimensions of input.

    Example
    -------
    >>> inputs = torch.rand(10, 50, 40)
    >>> lin_t = Linear(input_shape=(10, 50, 40), n_neurons=100)
    >>> output = lin_t(inputs)
    >>> output.shape
    torch.Size([10, 50, 100])
    """

    def __init__(
        self,
        n_neurons,
        input_shape=None,
        input_size=None,
        bias=True,
        combine_dims=False,
    ):
        super().__init__()
        self.combine_dims = combine_dims

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size")

        if input_size is None:
            input_size = input_shape[-1]
            if len(input_shape) == 4 and self.combine_dims:
                input_size = input_shape[2] * input_shape[3]

        # Weights are initialized following pytorch approach
        self.w = nn.Linear(input_size, n_neurons, bias=bias)

    def forward(self, x):
        """Returns the linear transformation of input tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input to transform linearly.
        """
        if x.ndim == 4 and self.combine_dims:
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        wx = self.w(x)

        return wx

class Conv1d(nn.Module):
    """This function implements 1d convolution.

    Arguments
    ---------
    out_channels : int
        It is the number of output channels.
    kernel_size : int
        Kernel size of the convolutional filters.
    input_shape : tuple
        The shape of the input. Alternatively use ``in_channels``.
    in_channels : int
        The number of input channels. Alternatively use ``input_shape``.
    stride : int
        Stride factor of the convolutional filters. When the stride factor > 1,
        a decimation in time is performed.
    dilation : int
        Dilation factor of the convolutional filters.
    padding : str
        (same, valid, causal). If "valid", no padding is performed.
        If "same" and stride is 1, output shape is the same as the input shape.
        "causal" results in causal (dilated) convolutions.
    groups: int
        Number of blocked connections from input channels to output channels.
    padding_mode : str
        This flag specifies the type of padding. See torch.nn documentation
        for more information.
    skip_transpose : bool
        If False, uses batch x time x channel convention of speechbrain.
        If True, uses batch x channel x time convention.

    Example
    -------
    >>> inp_tensor = torch.rand([10, 40, 16])
    >>> cnn_1d = Conv1d(
    ...     input_shape=inp_tensor.shape, out_channels=8, kernel_size=5
    ... )
    >>> out_tensor = cnn_1d(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 40, 8])
    """

    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        input_shape: Tuple[int]=None,
        in_channels: int=None,
        stride: int=1,
        dilation: int=1,
        padding: str="same",
        groups: int=1,
        bias: bool=True,
        padding_mode: str="reflect",
        skip_transpose: bool=False,
    ):
        super().__init__()
        self.kernel_size: int = kernel_size
        self.stride: int = stride
        self.dilation: int = dilation
        self.padding: str = padding
        self.padding_mode: str = padding_mode
        self.unsqueeze: bool = False
        self.skip_transpose: bool = skip_transpose
        if input_shape is None and in_channels is None:
            raise ValueError("Must provide one of input_shape or in_channels")
        
        if in_channels is None:
            in_channels = self._check_input_shape(input_shape)

        self.conv: nn.Module = nn.Conv1d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            padding=0,
            groups=groups,
            bias=bias,
        )
        self.stride: torch.Tensor = torch.ones(1, dtype=torch.int)

    def forward(self, x: torch.Tensor):
        """Returns the output of the convolution.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            input to convolve. 2d or 4d tensors are expected.
        """

        if not self.skip_transpose:
            x = x.transpose(1, -1)

        if self.unsqueeze:
            x = x.unsqueeze(1)

        if self.padding == "same":
            x = self._manage_padding(
                x, self.kernel_size, self.dilation, self.stride
            )

        elif self.padding == "causal":
            num_pad = (self.kernel_size - 1) * self.dilation
            x = F.pad(x, (num_pad, 0))

        elif self.padding == "valid":
            pass

        else:
            raise ValueError(
                "Padding must be 'same', 'valid' or 'causal'. Got "
                + self.padding
            )
        wx = self.conv(x)

        if self.unsqueeze:
            wx = wx.squeeze(1)

        if not self.skip_transpose:
            wx = wx.transpose(1, -1)

        return wx

    def _manage_padding(
        self, x: torch.Tensor, kernel_size: int, dilation: int, stride: torch.Tensor,
    ):
        """This function performs zero-padding on the time axis
        such that their lengths is unchanged after the convolution.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.
        kernel_size : int
            Size of kernel.
        dilation : int
            Dilation used.
        stride : int
            Stride.
        """

        # Detecting input shape
        L_in: int = x.shape[-1]

        # Time padding
        # for torch.jit.trace
        # padding: torch.Tensor = self._get_padding_elem(L_in, stride, kernel_size, dilation).to(x.device).detach()
        # for torch.jit.script
        padding: int = int(self._get_padding_elem(L_in, stride, kernel_size, dilation).to(x.device).detach().item())
        
        # padding: List[torch.Tensor] = [padding, padding]
        padding: List[int] = [padding, padding]
        
        # Applying padding
        x = F.pad(x, padding, mode=self.padding_mode)
        
        return x

    def _get_padding_elem(self, L_in: int, stride: torch.Tensor, kernel_size: int, dilation: int):
        """This function computes the number of elements to add for zero-padding.

        Arguments
        ---------
        L_in : int
        stride: int
        kernel_size : int
        dilation : int
        """
        L_out: torch.Tensor = torch.floor(((L_in - dilation * (kernel_size - 1) - 1)) / stride + 1) 
                
        padding: torch.Tensor = torch.floor((L_in - L_out) / 2).to(dtype=torch.int)
  
        return padding

    # @torch.jit.script
    # def _get_L_out(self, input_: torch.Tensor):
    #     L_in = input_.shape[-1]
    #     L_out = (
    #                 torch.floor(torch.tensor((L_in - self.dilation * (self.kernel_size - 1) - 1)) / self.stride) + 1
    #             )
    #     return L_in, L_out

    def _check_input_shape(self, shape: tuple):
        """Checks the input shape and returns the number of input channels.
        """

        if len(shape) == 2:
            self.unsqueeze: bool = True
            in_channels: int = 1
        elif self.skip_transpose:
            in_channels: int = shape[1]
        elif len(shape) == 3:
            in_channels: int = shape[2]
        else:
            raise ValueError(
                "conv1d expects 2d, 3d inputs. Got " + str(len(shape))
            )

        # Kernel size must be odd
        if self.kernel_size % 2 == 0:
            raise ValueError(
                "The field kernel size must be an odd number. Got %s."
                % (self.kernel_size)
            )
        return in_channels

@torch.jit.script
def _get_padding_from_tensor(padding: torch.Tensor):
    tuple_padding = [padding.detach(), padding.detach()]
    return tuple_padding


class BatchNorm1d(nn.Module):
    """Applies 1d batch normalization to the input tensor.

    Arguments
    ---------
    input_shape : tuple
        The expected shape of the input. Alternatively, use ``input_size``.
    input_size : int
        The expected size of the input. Alternatively, use ``input_shape``.
    eps : float
        This value is added to std deviation estimation to improve the numerical
        stability.
    momentum : float
        It is a value used for the running_mean and running_var computation.
    affine : bool
        When set to True, the affine parameters are learned.
    track_running_stats : bool
        When set to True, this module tracks the running mean and variance,
        and when set to False, this module does not track such statistics.
    combine_batch_time : bool
        When true, it combines batch an time axis.


    Example
    -------
    >>> input = torch.randn(100, 10)
    >>> norm = BatchNorm1d(input_shape=input.shape)
    >>> output = norm(input)
    >>> output.shape
    torch.Size([100, 10])
    """

    def __init__(
        self,
        input_shape: tuple=None,
        input_size: int=None,
        eps: float=1e-05,
        momentum: float=0.1,
        affine: bool=True,
        track_running_stats: bool=True,
        combine_batch_time: bool=False,
        skip_transpose: bool=False,
    ):
        super().__init__()
        self.combine_batch_time: bool = combine_batch_time
        self.skip_transpose: bool = skip_transpose

        if input_size is None and skip_transpose:
            input_size: int = input_shape[1]
        elif input_size is None:
            input_size: int = input_shape[-1]

        self.norm: nn.Module = nn.BatchNorm1d(
            input_size,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, x: torch.Tensor):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, [channels])
            input to normalize. 2d or 3d tensors are expected in input
            4d tensors can be used when combine_dims=True.
        """
        shape_or = x.shape
        if self.combine_batch_time:
            if x.ndim == 3:
                x = x.reshape(shape_or[0] * shape_or[1], shape_or[2])
            else:
                x = x.reshape(
                    shape_or[0] * shape_or[1], shape_or[3], shape_or[2]
                )

        elif not self.skip_transpose:
            x = x.transpose(-1, 1)

        x_n: torch.Tensor = self.norm(x)

        if self.combine_batch_time:
            x_n = x_n.reshape(shape_or)
        elif not self.skip_transpose:
            x_n = x_n.transpose(1, -1)

        return x_n