"""Library implementing normalization.

"""
import torch
import torch.nn as nn


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
        input_shape=None,
        input_size=None,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        combine_batch_time=False,
        skip_transpose=False,
    ):
        super().__init__()
        self.combine_batch_time = combine_batch_time
        self.skip_transpose = skip_transpose

        if input_size is None and skip_transpose:
            input_size = input_shape[1]
        elif input_size is None:
            input_size = input_shape[-1]

        self.norm = nn.BatchNorm1d(
            input_size,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, x):
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

        x_n = self.norm(x)

        if self.combine_batch_time:
            x_n = x_n.reshape(shape_or)
        elif not self.skip_transpose:
            x_n = x_n.transpose(1, -1)

        return x_n


class BatchNorm2d(nn.Module):
    """Applies 2d batch normalization to the input tensor.

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

    Example
    -------
    >>> input = torch.randn(100, 10, 5, 20)
    >>> norm = BatchNorm2d(input_shape=input.shape)
    >>> output = norm(input)
    >>> output.shape
    torch.Size([100, 10, 5, 20])
    """

    def __init__(
        self,
        input_shape=None,
        input_size=None,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super().__init__()

        if input_shape is None and input_size is None:
            raise ValueError("Expected input_shape or input_size as input")

        if input_size is None:
            input_size = input_shape[-1]

        self.norm = nn.BatchNorm2d(
            input_size,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, x):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel1, channel2)
            input to normalize. 4d tensors are expected.
        """
        x = x.transpose(-1, 1)
        x_n = self.norm(x)
        x_n = x_n.transpose(1, -1)

        return x_n


class LayerNorm(nn.Module):
    """Applies layer normalization to the input tensor.

    Arguments
    ---------
    input_shape : tuple
        The expected shape of the input.
    eps : float
        This value is added to std deviation estimation to improve the numerical
        stability.
    elementwise_affine : bool
        If True, this module has learnable per-element affine parameters
        initialized to ones (for weights) and zeros (for biases).

    Example
    -------
    >>> input = torch.randn(100, 101, 128)
    >>> norm = LayerNorm(input_shape=input.shape)
    >>> output = norm(input)
    >>> output.shape
    torch.Size([100, 101, 128])
    """

    def __init__(
        self,
        input_size=None,
        input_shape=None,
        eps=1e-05,
        elementwise_affine=True,
    ):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if input_shape is not None:
            input_size = input_shape[2:]

        self.norm = torch.nn.LayerNorm(
            input_size,
            eps=self.eps,
            elementwise_affine=self.elementwise_affine,
        )

    def forward(self, x):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channels)
            input to normalize. 3d or 4d tensors are expected.
        """
        return self.norm(x)


class InstanceNorm1d(nn.Module):
    """Applies 1d instance normalization to the input tensor.

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
    track_running_stats : bool
        When set to True, this module tracks the running mean and variance,
        and when set to False, this module does not track such statistics.
    affine : bool
        A boolean value that when set to True, this module has learnable
        affine parameters, initialized the same way as done for
        batch normalization. Default: False.

    Example
    -------
    >>> input = torch.randn(100, 10, 20)
    >>> norm = InstanceNorm1d(input_shape=input.shape)
    >>> output = norm(input)
    >>> output.shape
    torch.Size([100, 10, 20])
    """

    def __init__(
        self,
        input_shape=None,
        input_size=None,
        eps=1e-05,
        momentum=0.1,
        track_running_stats=True,
        affine=False,
    ):
        super().__init__()

        if input_shape is None and input_size is None:
            raise ValueError("Expected input_shape or input_size as input")

        if input_size is None:
            input_size = input_shape[-1]

        self.norm = nn.InstanceNorm1d(
            input_size,
            eps=eps,
            momentum=momentum,
            track_running_stats=track_running_stats,
            affine=affine,
        )

    def forward(self, x):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channels)
            input to normalize. 3d tensors are expected.
        """
        x = x.transpose(-1, 1)
        x_n = self.norm(x)
        x_n = x_n.transpose(1, -1)

        return x_n


class InstanceNorm2d(nn.Module):
    """Applies 2d instance normalization to the input tensor.

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
    track_running_stats : bool
        When set to True, this module tracks the running mean and variance,
        and when set to False, this module does not track such statistics.
    affine : bool
        A boolean value that when set to True, this module has learnable
        affine parameters, initialized the same way as done for
        batch normalization. Default: False.

    Example
    -------
    >>> input = torch.randn(100, 10, 20, 2)
    >>> norm = InstanceNorm2d(input_shape=input.shape)
    >>> output = norm(input)
    >>> output.shape
    torch.Size([100, 10, 20, 2])
    """

    def __init__(
        self,
        input_shape=None,
        input_size=None,
        eps=1e-05,
        momentum=0.1,
        track_running_stats=True,
        affine=False,
    ):
        super().__init__()

        if input_shape is None and input_size is None:
            raise ValueError("Expected input_shape or input_size as input")

        if input_size is None:
            input_size = input_shape[-1]

        self.norm = nn.InstanceNorm2d(
            input_size,
            eps=eps,
            momentum=momentum,
            track_running_stats=track_running_stats,
            affine=affine,
        )

    def forward(self, x):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel1, channel2)
            input to normalize. 4d tensors are expected.
        """
        x = x.transpose(-1, 1)
        x_n = self.norm(x)
        x_n = x_n.transpose(1, -1)

        return x_n


class GroupNorm(nn.Module):
    """Applies group normalization to the input tensor.

    Arguments
    ---------
    input_shape : tuple
        The expected shape of the input. Alternatively, use ``input_size``.
    input_size : int
        The expected size of the input. Alternatively, use ``input_shape``.
    num_groups : int
        Number of groups to separate the channels into.
    eps : float
        This value is added to std deviation estimation to improve the numerical
        stability.
    affine : bool
         A boolean value that when set to True, this module has learnable per-channel
         affine parameters initialized to ones (for weights) and zeros (for biases).
    Example
    -------
    >>> input = torch.randn(100, 101, 128)
    >>> norm = GroupNorm(input_size=128, num_groups=128)
    >>> output = norm(input)
    >>> output.shape
    torch.Size([100, 101, 128])
    """

    def __init__(
        self,
        input_shape=None,
        input_size=None,
        num_groups=None,
        eps=1e-05,
        affine=True,
    ):
        super().__init__()
        self.eps = eps
        self.affine = affine

        if input_shape is None and input_size is None:
            raise ValueError("Expected input_shape or input_size as input")

        if num_groups is None:
            raise ValueError("Expected num_groups as input")

        if input_shape is not None:
            input_size = input_shape[-1]

        self.norm = torch.nn.GroupNorm(
            num_groups, input_size, eps=self.eps, affine=self.affine,
        )

    def forward(self, x):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channels)
            input to normalize. 3d or 4d tensors are expected.
        """
        x = x.transpose(-1, 1)
        x_n = self.norm(x)
        x_n = x_n.transpose(1, -1)

        return x_n


class GlobalLayerNorm(nn.Module):
    """Calculate Global Layer Normalization.

    Arguments
    ---------
       dim : (int or list or torch.Size)
           Input shape from an expected input of size.
       eps : float
           A value added to the denominator for numerical stability.
       elementwise_affine : bool
          A boolean value that when set to True,
          this module has learnable per-element affine parameters
          initialized to ones (for weights) and zeros (for biases).

    Example
    -------
    >>> x = torch.randn(5, 10, 20)
    >>> GLN = GlobalLayerNorm(10, 3)
    >>> x_norm = GLN(x)
    """

    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """Returns the normalized tensor.

        Arguments
        ---------
        x : torch.Tensor
            Tensor of size [N, C, K, S] or [N, C, L].
        """
        # x = N x C x K x S or N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x K x S
        # gln: mean,var N x 1 x 1
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x - mean) ** 2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = (
                    self.weight * (x - mean) / torch.sqrt(var + self.eps)
                    + self.bias
                )
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)

        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x - mean) ** 2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = (
                    self.weight * (x - mean) / torch.sqrt(var + self.eps)
                    + self.bias
                )
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    """Calculate Cumulative Layer Normalization.

       Arguments
       ---------
       dim : int
        Dimension that you want to normalize.
       elementwise_affine : True
        Learnable per-element affine parameters.

    Example
    -------
    >>> x = torch.randn(5, 10, 20)
    >>> CLN = CumulativeLayerNorm(10)
    >>> x_norm = CLN(x)
    """

    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine, eps=1e-8
        )

    def forward(self, x):
        """Returns the normalized tensor.

        Arguments
        ---------
        x : torch.Tensor
            Tensor size [N, C, K, S] or [N, C, L]
        """
        # x: N x C x K x S or N x C x L
        # N x K x S x C
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
            # N x K x S x C == only channel norm
            x = super().forward(x)
            # N x C x K x S
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            x = torch.transpose(x, 1, 2)
            # N x L x C == only channel norm
            x = super().forward(x)
            # N x C x L
            x = torch.transpose(x, 1, 2)
        return x