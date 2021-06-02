from ..custom import CustomModule
import torch
import torch.nn as nn


class _LSTMModule(CustomModule):
    def __init__(self, hidden_size: int,
                 num_layers: int = 1, bias: bool = True,
                 dropout: float = 0.) -> None:
        super().__init__()
        self.args = {'hidden_size': hidden_size,
                     'num_layers': num_layers,
                     'bias': bias,
                     'dropout': dropout}


    def enter_in_shape(self, in_shape):
        self.lstm = nn.LSTM(in_shape[0], **self.args)

    def get_out_shape(self, in_shape):
        return (self.args['hidden_size'],)

    def get_initial_state(self, batch_size):
        h = torch.zeros(self.lstm.get_expected_hidden_size(None, [batch_size]), device=self.device).transpose(0,1)
        return h, h.clone()

    def forward(self, x, h):
        h = (h[0].transpose(0,1), h[1].transpose(0,1))
        x, h = self.lstm(x, h)
        h = (h[0].transpose(0,1), h[1].transpose(0,1))
        return x, h


class _TempConvPlus2dModule(CustomModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.inner = None

    def enter_in_shape(self, in_shape):
        self.inner = nn.Conv3d(in_channels=in_shape[0], **self.kwargs)

    def get_out_shape(self, in_shape):
        values = {k: torch.tensor(self.kwargs[k][1:], dtype=torch.float) for k in
                  ['kernel_size', 'stride', 'padding', 'dilation']}
        rest_shape = torch.floor((torch.tensor(in_shape[1:], dtype=torch.float) + 2 * values['padding'] - values[
            'dilation'] * (values['kernel_size'] - 1) - 1) / values['stride'] + 1).long()
        return (self.kwargs['out_channels'],) + tuple(rest_shape)

    def forward(self, x, _):
        x = x.permute(1, 2, 0, 3, 4)
        x = self.inner(x)
        return x.permute(2, 0, 1, 3, 4), ()