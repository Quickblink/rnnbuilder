import torch
from .modules import LIFNeuron, NoResetNeuron, CooldownNeuron, AdaptiveNeuron, DiscontinuousNeuron
from ..custom._factories import RecurrentFactory
from ..base import ModuleFactory
from ..custom._modules import StatelessWrapper




def spike_linear(gradient_factor=0.3):
    """Linear surrogate gradient function. Gradient is multiplied with `gradient_factor` at the threshold and falls
    linearly on both sides reaching 0 at 0 and \(2thr\) respectively.

    Args:
        gradient_factor: a multiplicator < 1 to add stability to backpropagation. Default: 0.3
    """
    class SpikeLinear(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return (input > 0).float()

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            return grad_output * torch.max(torch.zeros([1], device=input.device), 1 - torch.abs(input)) * gradient_factor

    return SpikeLinear.apply


class LIF(RecurrentFactory):
    r"""Standard Leaky-Integrate-and-Fire neuron.

    Args:
        tau: time constant of membrane potential. Default: 5
        spike_function: Surrogate gradient function used for backpropagation. Default: spike_linear()
    """
    def __init__(self, tau: float = 5,
                 spike_function=spike_linear()):
        super().__init__(LIFNeuron, 'flatten', True, True, tau, spike_function)

class NoReset(RecurrentFactory):
    r"""LIF neuron without the reset mechanism. Has improved memory capabilities. (See my master's thesis for details)

    Args:
        tau: time constant of membrane potential. Proportional to the average memory retention. Default: 5
        spike_function: Surrogate gradient function used for backpropagation. Default: spike_linear()
    """
    def __init__(self, tau: float = 5,
                 spike_function=spike_linear()):
        super().__init__(NoResetNeuron, 'flatten', True, True, tau, spike_function)

class Cooldown(RecurrentFactory):
    r"""NoReset neuron with additional exponential input transformation. Suitable to enable long-term memory.
     (See my master's thesis for details)

    Args:
        tau: time constant of membrane potential. Proportional to the average memory retention. Default: 5
        spike_function: Surrogate gradient function used for backpropagation. Default: spike_linear()
    """
    def __init__(self, tau: float = 5,
                 spike_function=spike_linear()):
        super().__init__(LIFNeuron, 'flatten', True, True, tau, spike_function)

class Adaptive(RecurrentFactory):
    r"""LIF neuron with adaptive threshold as presented in "Bellec et al., 2018: Long short-term memory and
    learning-to-learn in networks of spiking neurons".

    Args:
        tau: time constant of membrane potential. Default: 5
        spike_function: Surrogate gradient function used for backpropagation. Default: spike_linear()
        tau_thr: time constant of threshold. Proportional to the average memory retention. Default: 5
        gamma: scaling factor of threshold increase. Does not directly influence memory capabilities. Default: 0.25
    """
    def __init__(self, tau: float = 5,
                 spike_function=spike_linear(),
                 tau_thr: float = 5,
                 gamma: float = 0.25):
        super().__init__(LIFNeuron, 'flatten', True, True, tau, spike_function, tau_thr, gamma)

class Discontinuous(ModuleFactory):
    """Discontinuous spiking neuron. Essentially just the spike function without the persistent membrane potential.
    Equivalent to a LIF neuron with \(tau=0\).

    """
    def __init__(self, spike_function=spike_linear(), threshold: float = 1):
        self.args = spike_function, threshold

    def _assemble_module(self, in_shape, unrolled):
        return StatelessWrapper(in_shape, in_shape, DiscontinuousNeuron(*self.args))
