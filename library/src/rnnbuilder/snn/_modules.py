from ..custom import CustomModule
import torch
from torch import nn
from math import exp




class _NoResetNeuron(CustomModule):
    def __init__(self, tau, spike_function):
        super().__init__()
        self.beta = exp(-1 / tau)
        self.spike_fn = spike_function

    def get_out_shape(self, in_shape):
        return in_shape

    def enter_in_shape(self, in_shape):
        self.initial_mem = nn.Parameter(torch.zeros(in_shape), requires_grad=True)
        self.in_size = in_shape[0]


    def get_initial_state(self, batch_size):
        return {
            'mem': self.initial_mem.expand([batch_size, self.in_size]),
        }

    def get_initial_output(self, full_shape):
        return self.spike_fn(self.initial_mem.expand(full_shape) - 1)

    def forward(self, x, h):
        new_h = {}
        new_h['mem'] = self.beta * h['mem'] + x
        spikes = self.spike_fn(new_h['mem'] - 1)
        return spikes, new_h


class _LIFNeuron(_NoResetNeuron):
    def __init__(self, tau, spike_function):
        super().__init__(tau, spike_function)

    def forward(self, x, h):
        out, new_h = super().forward(x, h)
        new_h['mem'] = new_h['mem'] - out
        return out, new_h


class _DiscontinuousNeuron(nn.Module):
    def __init__(self, spike_function, threshold):
        super().__init__()
        self.spike_fn = spike_function
        self.threshold = threshold

    def forward(self, x):
        return self.spike_fn(x-self.threshold)


class _CooldownNeuron(_NoResetNeuron):
    def __init__(self, tau, spike_function):
        super().__init__(tau, spike_function)


    def enter_in_shape(self, in_shape):
        super().enter_in_shape(in_shape)
        self.register_buffer('sgn', torch.ones([self.in_size], requires_grad=False))
        self.sgn[(self.in_size//2):] *= -1

    def get_initial_output(self, full_shape):
        return (self.sgn < 0).float().expand(full_shape)

    def forward(self, x, h):
        new_h = {}
        new_h['mem'] = self.beta * h['mem'] + (1 - self.beta) * torch.exp(x)
        spikes = self.spike_fn(self.sgn * (h['mem'] - 1))
        return spikes, new_h


class _AdaptiveNeuron(_NoResetNeuron):
    def __init__(self, tau, spike_function, tau_thr, gamma):
        super().__init__(tau, spike_function)
        self.beta_thr = exp(-1 / tau_thr)
        self.gamma = gamma


    def get_initial_state(self, batch_size):
        h = super().get_initial_state(batch_size)
        h['rel_thresh'] = torch.zeros([batch_size, self.in_size], device=self.initial_mem.device)
        return h

    def forward(self, x, h):
        new_h = {}
        threshold = 1 + h['rel_thresh']
        spikes = self.spike_fn((h['mem'] - threshold)/threshold)
        new_h['mem'] = self.beta * h['mem'] + x - (spikes * threshold)
        new_h['rel_thresh'] = self.beta_thr * h['rel_thresh'] + self.gamma * spikes
        return spikes, new_h
