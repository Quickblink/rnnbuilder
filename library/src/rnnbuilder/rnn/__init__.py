import torch as _torch
from ..custom._factories import _RecurrentFactory
from ._modules import _LSTMModule, _TempConvPlus2dModule
from torch.nn.modules.utils import _pair


class LSTM(_RecurrentFactory):
    r"""Applies a multi-layer long short-term memory (LSTM) RNN to an input
    sequence.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}

    where \(h_t\) is the hidden state at time `t`, \(c_t\) is the cell
    state at time `t`, \(x_t\) is the input at time `t`, \(h_{t-1}\)
    is the hidden state of the layer at time `t-1` or the initial hidden
    state at time `0`, and \(i_t\), \(f_t\), \(g_t\),
    \(o_t\) are the input, forget, cell, and output gates, respectively.
    \(\sigma\) is the sigmoid function, and \(\odot\) is the Hadamard product.

    In a multilayer LSTM, the input \(x^{(l)}_t\) of the \(l\) -th layer
    (\(l >= 2\)) is the hidden state \(h^{(l-1)}_t\) of the previous layer multiplied by
    dropout \(\delta^{(l-1)}_t\) where each \(\delta^{(l-1)}_t\) is a Bernoulli random
    variable which is \(0\) with probability `dropout`.

    Args:
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two LSTMs together to form a `stacked LSTM`,
            with the second LSTM taking in outputs of the first LSTM and
            computing the final results. Default: 1
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            LSTM layer except the last layer, with dropout probability equal to
            `dropout`. Default: 0

    """
    def __init__(self, hidden_size: int,
                 num_layers: int = 1, bias: bool = True,
                 dropout: float = 0.) -> None:
        super().__init__(_LSTMModule, 'flatten', False, True, hidden_size, num_layers, bias, dropout)



class TempConvPlus2d(_RecurrentFactory):
    """A 2d-convolution combined with a convolution over the time dimension.
    Useful for e.g. frame stacking in Q-Learning.

    Args:
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        time_kernel_size (int): same but for time dimension
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        time_stride (int): same but for time dimension
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        time_padding (int): same but for time dimension
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        time_dilation (int): same but for time dimension
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
    """
    def __init__(
        self,
        out_channels: int,
        kernel_size: _torch.nn.common_types._size_2_t,
        time_kernel_size: _torch.nn.common_types._size_1_t,
        stride: _torch.nn.common_types._size_2_t = 1,
        padding: _torch.nn.common_types._size_2_t = 0,
        dilation: _torch.nn.common_types._size_2_t = 1,
        time_stride: _torch.nn.common_types._size_1_t = 1,
        time_padding: _torch.nn.common_types._size_1_t = 0,
        time_dilation: _torch.nn.common_types._size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'
    ):
        super().__init__(_TempConvPlus2dModule, False, False, False, out_channels=out_channels,
                         kernel_size=(time_kernel_size,)+tuple(_pair(kernel_size)),
                         stride=(time_stride,) + tuple(_pair(stride)),
                         padding=(time_padding,) + tuple(_pair(padding)),
                         dilation=(time_dilation,) + tuple(_pair(dilation)),
                         groups=groups, bias=bias, padding_mode=padding_mode)