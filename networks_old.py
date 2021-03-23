import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
from threading import Condition
from library.src.rnnbuilder.base.utils import StateContainerNew

class BellecSpike(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * torch.max(torch.zeros([1], device=input.device), 1 - torch.abs(input)) * 0.3



class SuperSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).
    """

    scale = 2.0#2.0#100.0  # controls steepness of surrogate gradient #TODO: make this a config parameter

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use the
        ctx.save_for_backward method.
        """
        #print(input[0,0].item())
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the
        surrogate gradient of the loss with respect to the input.
        Here we use the normalized negative part of a fast sigmoid
        as this was done in Zenke & Ganguli (2018).
        """
        #print(grad_output[0,0].item())
        input, = ctx.saved_tensors
        #clamped = torch.clamp(grad_output, -1e-3, 1e-3)
        out = grad_output / (SuperSpike.scale * torch.abs(input) + 1.0) ** 2

        #out = torch.clamp((grad_output / (SuperSpike.scale * torch.abs(input) + 1.0) ** 2), -1, 1)
        return out #torch.where((out == 0), torch.ones([1]) * 0.001, out)


def flatten(shape):
    out = 1
    for s in shape:
        out *= s
    return out


def magic_flatten(tensor, args):
    args = sorted(args, key=(lambda arg: tensor.names.index(arg)))
    namedshape = []
    combined_name = ''
    for arg in args:
        combined_name += arg
        namedshape.append((arg, tensor.shape[tensor.names.index(arg)]))
    return tensor.flatten(args, combined_name), (combined_name, namedshape)




class LinearModule(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.out_size = out_size

    def get_initial_state(self, batch_size):
        return ()

    def forward(self, x, h):
        return self.linear(x), ()


class FlattenerModule(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.in_size = size
        self.out_size = flatten(size)

    def get_initial_state(self, batch_size):
        return ()

    def forward(self, x, h):
        return x.flatten(('C', 'H', 'W'), 'data'), ()



class NetworkModule(nn.Module):
    def __init__(self, in_shape, *paths):
        super().__init__()
        sizes = {'input': in_shape}
        self.recurrent_layers = set()
        #print(paths)
        for path in paths:
            for input in path.inputs:
                if input not in sizes:
                    self.recurrent_layers.add(input)
            sizes[path.name] = path.deduce_size(sizes)
        self.order = []
        self.paths = nn.ModuleDict()
        self.inputs = {}
        for path in paths:
            self.paths[path.name] = nn.ModuleList()
            self.order.append(path.name)
            self.inputs[path.name] = path.inputs
            cur_size = sizes[path.inputs[0]] if len(path.inputs) == 1 else torch.tensor([sizes[x] for x in path.inputs]).sum().item()
            for factory in path.strand:
                new_module = factory(cur_size)
                self.paths[path.name].append(new_module)
                cur_size = new_module.out_size
        self.out_size = sizes['output']
        self.in_size = sizes['input']

    def forward(self, inp, h):
        state, spikes = h
        new_state = {}
        results = {'input': inp, **spikes}
        for name in self.order:
            if len(self.inputs[name]) > 1:
                inputs = []
                for p in self.inputs[name]:
                    inputs.append(results[p])
                x = torch.cat(inputs, dim=-1)
            else:
                x = results[self.inputs[name][0]]
            new_state[name] = []
            for i, module in enumerate(self.paths[name]):
                x, h = module(x, state[name][i])
                new_state[name].append(h)
            results[name] = x
        new_spikes = {x: results[x] for x in self.recurrent_layers}
        return results['output'], (new_state, new_spikes)

    def get_initial_state(self, batch_size):
        state = {name: [module.get_initial_state(batch_size) for module in path] for name, path in self.paths.items()}
        spikes = {x: self.paths[x][-1].get_initial_output(batch_size) for x in self.recurrent_layers}
        return state, spikes

class ConvModule(nn.Module):
    def __init__(self, conv, neuron):
        super().__init__()
        self.conv = conv
        self.neuron = neuron
        self.out_size = conv.out_shape

    def forward(self, inp, h):
        x = self.conv(inp)
        x = x.align_to('time', 'batch', 'C', 'H', 'W')
        x, named_shape = magic_flatten(x, ('C', 'H', 'W'))
        x, n_h = self.neuron(x, h)
        x = x.unflatten(*named_shape)
        return x, n_h

    def get_initial_state(self, batch_size):
        return self.neuron.get_initial_state(batch_size)



class ConvWrapper2(nn.Module):
    defaults = {
        'stride': 1,
        'padding': 0,
        'dilation': 1
    }
    def __init__(self, in_shape, params):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_shape[0], **params)
        self.out_shape = self.calc_shape(in_shape, params) #TODO: static call correct?

    @staticmethod
    def calc_shape(in_shape, params):
        values = {**ConvWrapper2.defaults, **params}
        for k in values:
            if not isinstance(values[k], (tuple, list)):
                values[k] = torch.tensor([values[k], values[k]], dtype=torch.float)
            else:
                values[k] = torch.tensor(values[k], dtype=torch.float)
        rest_shape = torch.floor((torch.tensor(in_shape[1:], dtype=torch.float) + 2 * values['padding'] - values['dilation'] * (values['kernel_size'] - 1)-1) / values['stride'] + 1).long()
        return tuple([params['out_channels']] + rest_shape.tolist())


    def forward(self, x):
        x = x.align_to('time', 'batch', 'C', 'H', 'W')
        x, named_shape = magic_flatten(x, ('time', 'batch'))
        x = x.rename(None)
        x = self.conv(x)
        x = x.refine_names('timebatch', 'C', 'H', 'W')
        x = x.unflatten(*named_shape)
        return x

class ConvWrapper3d(nn.Module):
    defaults = {
        'kernel_size': 1,
        'stride': 1,
        'padding': 0,
        'dilation': 1
    }
    def __init__(self, in_shape, params):
        super().__init__()
        self.out_shape, changed_params = self.calc_shape(in_shape, params)
        self.conv = nn.Conv3d(in_channels=in_shape[0], **changed_params)

    @staticmethod
    def calc_shape(in_shape, params):
        depth_params = {**ConvWrapper3d.defaults}
        params = {**ConvWrapper3d.defaults, **params} #TODO: correct defaults??? was conwrapper2 before
        values = {}
        if 'frame_stack' in params:
            depth_params['kernel_size'] = params['frame_stack']
            del params['frame_stack']
        for k in depth_params:
            if isinstance(params[k], tuple) and len(params[k]) == 2:
                params[k] = (depth_params[k], params[k][0], params[k][2])
            else:
                params[k] = (depth_params[k], params[k], params[k])
            values[k] = torch.tensor(params[k][1:], dtype=torch.float)
        rest_shape = torch.floor((torch.tensor(in_shape[1:], dtype=torch.float) + 2 * values['padding'] - values['dilation'] * (values['kernel_size'] - 1)-1) / values['stride'] + 1).long()
        return tuple([params['out_channels']] + rest_shape.tolist()), params


    def forward(self, x):
        x = x.align_to('batch', 'C', 'time', 'H', 'W')
        x = x.rename(None)
        x = self.conv(x)
        x = x.refine_names('batch', 'C', 'time', 'H', 'W')
        return x



class SequenceWrapperFull(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.in_size = model.in_size
        self.out_size = model.out_size
        self.is_logging = False
        self.full_state = False

    def make_log(self, length, first_entry):
        if type(first_entry) is dict:
            new_log = {}
            for k, v in first_entry.items():
                new_log[k] = self.make_log(length, v)
        elif type(first_entry) is torch.Tensor:
            new_log = torch.empty(length + first_entry.shape, device=first_entry.device)
        else:
            raise Exception('Unknown type in logging!')
        return new_log

    def enter_log(self, log, entry, t):
        if type(log) is dict:
            for k in log:
                self.enter_log(log[k], entry[k], t)
        elif type(log) is torch.Tensor:
            log[t] = entry.rename(None)
        else:
            raise Exception('Unknown type in logging!')

    def forward(self, inp, h):
        self.model.is_logging = self.is_logging
        if self.is_logging:
            model_out = self.model(inp[0], h)
            out1, h, first_entry = model_out[0], model_out[1], model_out[2] if len(model_out) > 2 else model_out[0]
            log = self.make_log(inp.shape[:1], first_entry)
            self.enter_log(log, first_entry, 0)
        else:
            out1, h = self.model(inp[0], h)
        output = torch.empty(inp.shape[:1]+out1.shape, device=inp.device, names=('time', *out1.names))
        if self.full_state:
            state = StateContainerNew(h, inp.shape[:2], inp.device)
            for container, entry in state.transfer(h):
                container[0] = entry.rename(None)
        output[0] = out1.rename(None)
        for t in range(1, inp.shape[0]):
            model_out = self.model(inp[t], h)
            output[t], h = model_out[0].rename(None), model_out[1]
            if self.is_logging:
                self.enter_log(log, model_out[2] if len(model_out) > 2 else model_out[0], t)
            if self.full_state:
                for container, entry in state.transfer(h):
                    container[t] = entry.rename(None)
        out_state = state.state if self.full_state else h
        if self.is_logging:
            return output, out_state, log
        else:
            return output, out_state

    def get_initial_state(self, batch_size):
        return self.model.get_initial_state(batch_size)



class BaseNeuron(nn.Module):
    def __init__(self, size, _):
        super().__init__()
        self.in_size = size
        self.out_size = size
        self.register_buffer('device_zero', torch.zeros(1, requires_grad=False))

    def get_initial_state(self, batch_size):
        return ()

    def get_initial_output(self, batch_size):
        return self.device_zero.expand([batch_size, self.in_size])

    def forward(self, x, h):
        return x, ()

class Multiplicator(BaseNeuron):
    def __init__(self, size, factor):
        super().__init__(size, None)
        self.factor = factor

    def forward(self, x, h):
        return x*self.factor, ()


class Dueling(BaseNeuron):
    def forward(self, x, h):
        val_out = x[..., :1]
        adv_out = x[..., 1:]
        return val_out.expand(*x.shape[:-1],self.out_size) + (adv_out - adv_out.mean(dim=-1, keepdim=True).expand(*x.shape[:-1],self.out_size)), ()


class LSTMWrapper(BaseNeuron):
    def __init__(self, in_size, out_size):
        super().__init__(in_size, None)
        self.out_size = out_size
        self.lstm = nn.LSTM(in_size, out_size)
        #with torch.no_grad():
        #    self.lstm.bias_hh_l0[512:1024] += -10000

    def get_initial_state(self, batch_size):
        h = torch.zeros(self.lstm.get_expected_hidden_size(None, [batch_size]), device=self.device_zero.device).transpose(0,1)
        return h, h.clone()

    def forward(self, x, h):
        #x = x.align_to('time', 'batch', 'data')
        x = x.rename(None)
        h = (h[0].transpose(0,1), h[1].transpose(0,1))
        x, h = self.lstm(x, h)
        h = (h[0].transpose(0,1), h[1].transpose(0,1))
        x = x.refine_names('time', 'batch', 'data')
        return x, h


class LSTMWrapperOneStep(BaseNeuron):
    def __init__(self, in_size, out_size):
        super().__init__(in_size, None)
        self.out_size = out_size
        self.lstm = nn.LSTM(in_size, out_size)
        #with torch.no_grad():
        #    self.lstm.bias_hh_l0[512:1024] += -10000

    def get_initial_state(self, batch_size):
        h = torch.zeros(self.lstm.get_expected_hidden_size(None, [batch_size]), device=self.device_zero.device).transpose(0,1)
        return h, h.clone()

    def forward(self, x, h):
        #x = x.align_to('time', 'batch', 'data')
        x = x.rename(None)
        h = (h[0].transpose(0,1), h[1].transpose(0,1))
        x, h = self.lstm(x.unsqueeze(0), h)
        h = (h[0].transpose(0,1), h[1].transpose(0,1))
        x = x.squeeze(0).refine_names('batch', 'data')
        return x, h


class ReLuWrapper(BaseNeuron):
    def __init__(self, size, _):
        super().__init__(size, None)

    def forward(self, x, h):
        return F.relu(x), ()


class TanhWrapper(BaseNeuron):
    def __init__(self, size, _):
        super().__init__(size, None)

    def forward(self, x, h):
        return torch.tanh(x), ()


class MeanModule(BaseNeuron):
    def __init__(self, size, last_index):
        super().__init__(size, None)
        self.last_index = last_index

    def forward(self, x, h):
        return x[self.last_index:].mean(dim=0), ()



class NoResetNeuron(BaseNeuron):
    def __init__(self, size, params):
        super().__init__(size, None)
        self.beta = params['BETA']
        if params['1-beta'] == 'improved':
            self.factor = (1 - self.beta ** 2) ** (0.5)
        elif params['1-beta']:
            self.factor = (1-self.beta)
        else:
            self.factor = 1
        if params['SPIKE_FN'] == 'bellec':
            self.spike_fn = BellecSpike.apply
        else:
            self.spike_fn = SuperSpike.apply
        self.initial_mem = nn.Parameter(torch.zeros([size]), requires_grad=True)
        self.target_var = 1
        self.est_rate = 0.5


    def get_initial_state(self, batch_size):
        return {
            'mem': self.initial_mem.expand([batch_size, self.in_size]),
        }

    def get_initial_output(self, batch_size):
        return self.spike_fn(self.initial_mem.expand([batch_size, self.in_size]) - 1)

    def forward(self, x, h):
        new_h = {}
        new_h['mem'] = self.beta * h['mem'] + self.factor * x
        spikes = self.spike_fn(new_h['mem'].refine_names(*x.names) - 1)
        #print(x.names, new_h['mem'].names, h['mem'].names, spikes.names)
        return spikes, new_h




class CooldownNeuron(NoResetNeuron):
    def __init__(self, size, params):
        super().__init__(size, params)
        self.offset = params['OFFSET']
        self.elu = torch.nn.ELU()
        self.register_buffer('sgn', torch.ones([size], requires_grad=False))
        self.sgn[(size//2):] *= -1

    def get_initial_output(self, batch_size):
        return (self.sgn < 0).float().expand([batch_size, self.in_size])

    def forward(self, x, h):
        new_h = {}
        new_h['mem'] = self.beta * h['mem'] + self.elu(x.rename(None)-self.offset) + 1
        spikes = self.spike_fn(self.sgn * (h['mem'] - 1))
        return spikes, new_h



class LIFNeuron(NoResetNeuron):
    def __init__(self, size, params):
        super().__init__(size, params)
        self.est_rate = 0.06


    def forward(self, x, h):
        out, new_h = super().forward(x, h)
        new_h['mem'] = new_h['mem'] - out#.detach()#TODO:remove
        return out, new_h


class AdaptiveNeuron(NoResetNeuron):
    def __init__(self, size, params):
        super().__init__(size, params)
        self.decay = params['ADAPDECAY']
        self.scale = params['ADAPSCALE']
        self.est_rate = 0.06


    def get_initial_state(self, batch_size):
        h = super().get_initial_state(batch_size)
        h['rel_thresh'] = torch.zeros([batch_size, self.in_size], device=self.initial_mem.device)
        return h

    def forward(self, x, h):
        new_h = {}
        new_h['rel_thresh'] = self.decay * h['rel_thresh'] + (1-self.decay) * h['spikes']
        threshold = 1 + new_h['rel_thresh'] * self.scale
        new_h['spikes'] = self.spike_fn((h['mem'] - threshold)/threshold)
        new_h['mem'] = self.beta * h['mem'] + self.factor * x - (new_h['spikes'] * threshold)#.detach()
        return new_h['spikes'], new_h



class DiscontinuousNeuron(BaseNeuron):
    def __init__(self, size, params):
        super().__init__(size, None)
        if params['SPIKE_FN'] == 'bellec':
            self.spike_fn = BellecSpike.apply
        else:
            self.spike_fn = SuperSpike.apply
        self.threshold = params['THRESHOLD']

    def get_initial_state(self, batch_size):
        return ()

    def forward(self, x, h):
        return self.spike_fn(x-self.threshold), () #had no -1 before


class OutputNeuron(NoResetNeuron):
    def __init__(self, size, params):
        super().__init__(size, params)

    def forward(self, x, h):
        _, new_h = super().forward(x, h)
        return new_h['mem'], new_h