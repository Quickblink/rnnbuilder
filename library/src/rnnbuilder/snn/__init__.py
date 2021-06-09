import torch as _torch
from ._modules import _LIFNeuron, _NoResetNeuron, _CooldownNeuron, _AdaptiveNeuron, _DiscontinuousNeuron
from ..custom._factories import _RecurrentFactory
from ..base import ModuleFactory as _ModuleFactory
from ..custom._modules import _StatelessWrapper




class SpikeLinear:
    """Linear surrogate gradient function. Gradient is multiplied with `gradient_factor` at the threshold and falls
    linearly on both sides reaching 0 at 0 and \(2thr\) respectively.

    Args:
        gradient_factor: a multiplicator < 1 to add stability to backpropagation. Default: 0.3
    """
    class _Inner(_torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, gradient_factor):
            ctx.save_for_backward(input)
            ctx.gradient_factor = gradient_factor
            return (input > 0).float()

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            return grad_output * _torch.max(_torch.zeros([1], device=input.device),
                                            1 - _torch.abs(input)) * ctx.gradient_factor, None

    def __init__(self, gradient_factor=0.3):
        self.gradient_factor = gradient_factor

    def __call__(self, x):
        return SpikeLinear._Inner.apply(x, self.gradient_factor)





class LIF(_RecurrentFactory):
    r"""Standard Leaky-Integrate-and-Fire neuron.

    Args:
        tau: time constant of membrane potential. Default: 5
        spike_function: Surrogate gradient function used for backpropagation. Default: spike_linear()
    """
    def __init__(self, tau: float = 5,
                 spike_function=SpikeLinear()):
        super().__init__(_LIFNeuron, 'flatten', True, True, tau, spike_function)

class NoReset(_RecurrentFactory):
    r"""LIF neuron without the reset mechanism. Has improved memory capabilities. (See my master's thesis for details)

    Args:
        tau: time constant of membrane potential. Proportional to the average memory retention. Default: 5
        spike_function: Surrogate gradient function used for backpropagation. Default: spike_linear()
    """
    def __init__(self, tau: float = 5,
                 spike_function=SpikeLinear()):
        super().__init__(_NoResetNeuron, 'flatten', True, True, tau, spike_function)

class Cooldown(_RecurrentFactory):
    r"""NoReset neuron with additional exponential input transformation. Suitable to enable long-term memory.
     (See my master's thesis for details)

    Args:
        tau: time constant of membrane potential. Proportional to the average memory retention. Default: 5
        spike_function: Surrogate gradient function used for backpropagation. Default: spike_linear()
    """
    def __init__(self, tau: float = 5,
                 spike_function=SpikeLinear()):
        super().__init__(_CooldownNeuron, 'flatten', True, True, tau, spike_function)

class Adaptive(_RecurrentFactory):
    r"""LIF neuron with adaptive threshold as presented in "Bellec et al., 2018: Long short-term memory and
    learning-to-learn in networks of spiking neurons".

    Args:
        tau: time constant of membrane potential. Default: 5
        spike_function: Surrogate gradient function used for backpropagation. Default: spike_linear()
        tau_thr: time constant of threshold. Proportional to the average memory retention. Default: 5
        gamma: scaling factor of threshold increase. Does not directly influence memory capabilities. Default: 0.25
    """
    def __init__(self, tau: float = 5,
                 spike_function=SpikeLinear(),
                 tau_thr: float = 5,
                 gamma: float = 0.25):
        super().__init__(_AdaptiveNeuron, 'flatten', True, True, tau, spike_function, tau_thr, gamma)

class Discontinuous(_ModuleFactory):
    """Discontinuous spiking neuron. Essentially just the spike function without the persistent membrane potential.
    Equivalent to a LIF neuron with \(tau=0\).

    """
    def __init__(self, spike_function=SpikeLinear(), threshold: float = 1):
        self.args = spike_function, threshold

    def _assemble_module(self, in_shape, unrolled):
        return _StatelessWrapper(in_shape, in_shape, _DiscontinuousNeuron(*self.args))
