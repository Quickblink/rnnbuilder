import torch
import torch.nn as nn
from utils import StateContainerNew



class ModuleBase(nn.Module):
    def __init__(self, in_shape):
        super().__init__()
        self.in_shape = in_shape

    def get_initial_state(self, batch_size):
        return ()



class StatelessWrapper(ModuleBase):
    def __init__(self, in_shape, inner, out_shape):
        super().__init__(in_shape)
        self.inner = inner
        self.out_shape = out_shape

    def forward(self, x, h):
        time = x.shape[0]
        batch = x.shape[1]
        x = x.reshape((time*batch,)+self.in_shape)
        x = self.inner(x)
        return x.reshape((time, batch)+self.out_shape), ()

    def get_initial_output(self, batch_size):
        return torch.zeros((1, batch_size)+self.out_shape)



class SequentialModule(ModuleBase):
    def __init__(self, in_shape, modules):
        super().__init__(in_shape)
        self.mlist = nn.ModuleList(modules)

    def forward(self, inp, hidden_state):
        new_state = []
        x = inp
        for i, module in enumerate(self.mlist):
            x, h = module(x, hidden_state[i])
            new_state.append(h)
        return x, new_state

    def get_initial_state(self, batch_size):
        return [module.get_initial_state(batch_size) for module in self.mlist]

    def get_initial_output(self, batch_size):
        self.mlist[-1].get_initial_output(batch_size)

class OuterNetworkModule(ModuleBase):
    def __init__(self, in_shape, order, inputs, layers: nn.ModuleDict, cycle_outputs, placeholders_rev):
        super().__init__(in_shape)
        self.order = order
        self.layers = layers
        self.inputs = inputs
        self.placeholders_rev = placeholders_rev  # maps from placeholder names to layer names

        self.cycle_outputs = cycle_outputs

    #TODO: full_state output
    def _run_cycle(self, results, cycle_name, h):
        time = results[next(iter(self.inputs[cycle_name]))].shape[0]
        inner_results = {name: results[name][0].unsqueeze(0) for name in self.inputs[cycle_name]}
        inner_results, h = self.layers[cycle_name](inner_results, h)
        for name in self.cycle_outputs[cycle_name]:
            results[name] = torch.empty((time,)+inner_results[name].shape, device=inner_results[name].device)
            results[name][0] = inner_results[name]
        for t in range(1, time):
            inner_results = {name: results[name][t].unsqueeze(0) for name in self.inputs[cycle_name]}
            inner_results, h = self.layers[cycle_name](inner_results, h)
            for name in self.cycle_outputs[cycle_name]:
                results[name][t] = inner_results[name]

    def forward(self, inp, hidden_state):
        state, recurrent_outputs = hidden_state
        new_state = {}
        new_recurrent_outputs = {}
        results = {'input': inp}
        for layer_name in self.order:
            if layer_name in self.cycle_outputs:
                self._run_cycle(results, layer_name, state[layer_name])
            else:
                if len(self.inputs[layer_name]) > 1:
                    inputs = [results[p].reshape(results[p].shape[:2]+(-1,)) for p in self.inputs[layer_name]]
                    x = torch.cat(inputs, dim=-1)
                else:
                    x = results[next(iter(self.inputs[layer_name]))]
                results[layer_name], new_state[layer_name] = self.layers[layer_name](x, state[layer_name])
                if layer_name in self.placeholders_rev:
                    ph = self.placeholders_rev[layer_name]
                    new_recurrent_outputs[ph] = results[layer_name][-1:]
                    results[ph] = torch.cat((recurrent_outputs[ph], results[layer_name][:-1]), dim=0)
        return results['output'], (new_state, new_recurrent_outputs)

    def get_initial_state(self, batch_size):
        state = {layer_name: layer.get_initial_state(batch_size) for layer_name, layer in self.layers.items()}
        recurrent_outputs = {ph_name: self.layers[layer_name].get_initial_output(batch_size) for layer_name, ph_name in self.placeholders_rev.items()}
        return state, recurrent_outputs


    def get_initial_output(self, batch_size):
        self.mlist[-1].get_initial_output(batch_size)

class InnerNetworkModule(ModuleBase):
    def __init__(self, in_shape, order, inputs, layers: nn.ModuleDict, recurrent_layers):
        super().__init__(in_shape)
        self.order = order
        self.layers = layers
        self.inputs = inputs
        self.recurrent_layers = recurrent_layers  # maps from placeholder names to layer names


    def forward(self, inp, hidden_state):
        state, recurrent_outputs = hidden_state
        new_state = {}
        results = {'input': inp, **recurrent_outputs}
        for layer_name in self.order:
            if len(self.inputs[layer_name]) > 1:
                inputs = [results[p].reshape(results[p].shape[:2]+(-1,)) for p in self.inputs[layer_name]]
                x = torch.cat(inputs, dim=-1)
            else:
                x = results[self.inputs[layer_name][0]]
            results[layer_name], new_state[layer_name] = self.layers[layer_name](x, state[layer_name])
        new_recurrent_outputs = {ph_name: results[layer_name] for ph_name, layer_name in self.recurrent_layers.items()}
        return results['output'], (new_state, new_recurrent_outputs)

    def get_initial_state(self, batch_size):
        state = {layer_name: layer.get_initial_state(batch_size) for layer_name, layer in self.layers.items()}
        recurrent_outputs = {ph_name: self.layers[layer_name].get_initial_output(batch_size) for ph_name, layer_name in self.recurrent_layers.items()}
        return state, recurrent_outputs


    def get_initial_output(self, batch_size):
        self.mlist[-1].get_initial_output(batch_size)

class NestedNetworkModule(InnerNetworkModule):
    def forward(self, results, hidden_state):
        state, recurrent_outputs = hidden_state
        new_state = {}
        results = {**recurrent_outputs, **results}
        for layer_name in self.order:
            if len(self.inputs[layer_name]) > 1:
                inputs = [results[p].reshape(results[p].shape[:2]+(-1,)) for p in self.inputs[layer_name]]
                x = torch.cat(inputs, dim=-1)
            else:
                x = results[self.inputs[layer_name][0]]
            results[layer_name], new_state[layer_name] = self.layers[layer_name](x, state[layer_name])
        new_recurrent_outputs = {ph_name: results[layer_name] for ph_name, layer_name in self.recurrent_layers.items()}
        return results, (new_state, new_recurrent_outputs)


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