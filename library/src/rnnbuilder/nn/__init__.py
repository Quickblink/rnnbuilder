""" This module provides factories for standard torch modules. Documentation and signatures are directly copied from
PyTorch and copyright applies accordingly.
"""
import torch as _torch
from ..base import ModuleFactory as _ModuleFactory
from ..custom._modules import _StatelessWrapper
from ..custom._factories import _NonRecurrentFactory
from ..base._utils import _flatten_shape
from typing import Optional as _Optional

class Linear(_ModuleFactory):
    r"""Applies a linear transformation to the incoming data: \(y = xA^T + b\)

    Args:
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
    """


    def __init__(self, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.out_features = out_features
        self.bias = bias

    def _assemble_module(self, in_shape, unrolled):
        f_shape = _flatten_shape(in_shape)
        return _StatelessWrapper(f_shape, (self.out_features,), _torch.nn.Linear(f_shape[0], self.out_features, self.bias))

    def _shape_change(self, in_shape):
        return (self.out_features,)


class Conv2d(_NonRecurrentFactory):
    r"""Applies a 2D convolution over an input signal composed of several input
    planes.


    Args:
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
    """
    def __init__(
        self,
        out_channels: int,
        kernel_size: _torch.nn.common_types._size_2_t,
        stride: _torch.nn.common_types._size_2_t = 1,
        padding: _torch.nn.common_types._size_2_t = 0,
        dilation: _torch.nn.common_types._size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'
    ):
        super().__init__((lambda in_shape, *args: _torch.nn.Conv2d(in_shape[0], *args)), False, 'auto',
                         out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

class ReLU(_ModuleFactory):
    r"""Applies the rectified linear unit function element-wise:

    .. math::
        \text{ReLU}(x) = (x)^+ = \max(0, x)

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def _assemble_module(self, in_shape, unrolled):
        return _StatelessWrapper(in_shape, in_shape, _torch.nn.ReLU(inplace=self.inplace))


class Sigmoid(_ModuleFactory):
    r"""Applies the element-wise function:

    .. math::
        \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}
    """

    def _assemble_module(self, in_shape, unrolled):
        return _StatelessWrapper(in_shape, in_shape, _torch.nn.Sigmoid())


class Tanh(_ModuleFactory):
    r"""Applies the element-wise function:

    .. math::
        \text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}
    """

    def _assemble_module(self, in_shape, unrolled):
        return _StatelessWrapper(in_shape, in_shape, _torch.nn.Tanh())



class BatchNorm2d(_ModuleFactory):
    r"""Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    [Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift](https://arxiv.org/abs/1502.03167) .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and \(\gamma\) and \(\beta\) are learnable parameter vectors
    of size `C` (where `C` is the input size). By default, the elements of \(\gamma\) are set
    to 1 and the elements of \(\beta\) are set to 0. The standard-deviation is calculated
    via the biased estimator, equivalent to `torch.var(input, unbiased=False)`.


    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial Batch Normalization.

    Args:
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers `running_mean` and `running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``
    """

    def __init__(self, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        self.args = eps, momentum, affine, track_running_stats

    def _assemble_module(self, in_shape, unrolled):
        return _StatelessWrapper(in_shape, in_shape, _torch.nn.BatchNorm2d(in_shape[0], *self.args))


class MaxPool2d(_NonRecurrentFactory):
    r"""Applies a 2D max pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size \((N, C, H, W)\),
    output \((N, C, H_{out}, W_{out})\) and `kernel_size` \((kH, kW)\)
    can be precisely described as:

    .. math::
        \begin{aligned}
            out(N_i, C_j, h, w) ={} & \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                    & \text{input}(N_i, C_j, \text{stride[0]} \times h + m,
                                                   \text{stride[1]} \times w + n)
        \end{aligned}

    If `padding` is non-zero, then the input is implicitly zero-padded on both sides
    for `padding` number of points. `dilation` controls the spacing between the kernel points.

    The parameters can either be:

    - a single ``int`` -- in which case the same value is used for the height and width dimension
    - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
      and the second `int` for the width dimension

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is `kernel_size`
        padding: implicit zero padding to be added on both sides
        dilation: a parameter that controls the stride of elements in the window

    """

    def __init__(self, kernel_size: _torch.nn.common_types._size_any_t, stride: _Optional[_torch.nn.common_types._size_any_t] = None,
                 padding: _torch.nn.common_types._size_any_t = 0, dilation: _torch.nn.common_types._size_any_t = 1) -> None:
        super().__init__((lambda in_shape, *args: _torch.nn.MaxPool2d(*args)), False, 'auto',
                         kernel_size, stride, padding, dilation)