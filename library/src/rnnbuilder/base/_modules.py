import torch

from ._utils import StateContainerNew

class _ModuleBase(torch.nn.Module):
    def __init__(self, in_shape):
        super().__init__()
        self.out_shape = in_shape
        self.in_shape = in_shape

    def get_initial_state(self, batch_size):
        return ()

    def forward(self, x, h):
        return x, h

    def get_initial_output(self, batch_size):
        return torch.zeros((batch_size,) + self.out_shape, device=self.device)


class _SequentialModule(_ModuleBase):
    def __init__(self, in_shape, modules):
        super().__init__(in_shape)
        self.mlist = torch.nn.ModuleList(modules)

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
        return self.mlist[-1].get_initial_output(batch_size)

class _OuterNetworkModule(_ModuleBase):
    def __init__(self, in_shape, order, inputs, layers: torch.nn.ModuleDict, cycle_outputs, placeholders_rev, input_modes):
        super().__init__(in_shape)
        self.order = order
        self.layers = layers
        self.inputs = inputs
        self.placeholders_rev = placeholders_rev  # maps from placeholder names to layer names
        self._full_state = False
        self.cycle_outputs = cycle_outputs
        self.input_modes = input_modes


    def _run_cycle(self, results, cycle_name, h):
        time, batch = results[next(iter(self.inputs[cycle_name]))].shape[:2]
        inner_results = {name: results[name][0].unsqueeze(0) for name in self.inputs[cycle_name]}
        inner_results, h = self.layers[cycle_name](inner_results, h)
        if self._full_state:
            state = StateContainerNew(h, (time, batch), self.device)
            for container, entry in state.transfer(h):
                container[0] = entry
        for name in self.cycle_outputs[cycle_name]:
            results[name] = torch.empty((time,)+inner_results[name].shape[1:], device=inner_results[name].device)
            results[name][0] = inner_results[name][0]
        for t in range(1, time):
            inner_results = {name: results[name][t].unsqueeze(0) for name in self.inputs[cycle_name]}
            inner_results, h = self.layers[cycle_name](inner_results, h)
            for name in self.cycle_outputs[cycle_name]:
                results[name][t] = inner_results[name][0]
            if self._full_state:
                for container, entry in state.transfer(h):
                    container[t] = entry
        out_state = state.state if self._full_state else h
        return out_state

    def forward(self, inp, hidden_state):
        state, recurrent_outputs = hidden_state
        new_state = {}
        new_recurrent_outputs = {}
        results = {'input': inp}
        for layer_name in self.order:
            if layer_name in self.cycle_outputs:
                new_state[layer_name] = self._run_cycle(results, layer_name, state[layer_name])
            else:
                if len(self.inputs[layer_name]) > 1:
                    inputs = [results[p].reshape(results[p].shape[:2]+(-1,)) for p in self.inputs[layer_name]]
                    if self.input_modes[layer_name] == 'stack':
                        x = torch.cat(inputs, dim=-1)
                    elif self.input_modes[layer_name] == 'sum':
                        x = sum(inputs)
                    else:
                        x = inputs
                else:
                    x = results[self.inputs[layer_name][0]]
                results[layer_name], new_state[layer_name] = self.layers[layer_name](x, state[layer_name])
                if layer_name in self.placeholders_rev:
                    ph = self.placeholders_rev[layer_name]
                    new_recurrent_outputs[ph] = results[layer_name] if self._full_state else results[layer_name][-1]
                    results[ph] = torch.cat((recurrent_outputs[ph].unsqueeze(0), results[layer_name][:-1]), dim=0)
        return results['output'], (new_state, new_recurrent_outputs)

    def get_initial_state(self, batch_size):
        state = {layer_name: layer.get_initial_state(batch_size) for layer_name, layer in self.layers.items()}
        recurrent_outputs = {ph_name: self.layers[layer_name].get_initial_output(batch_size) for layer_name, ph_name in self.placeholders_rev.items()}
        return state, recurrent_outputs


    def get_initial_output(self, batch_size):
        return self.layers['output'].get_initial_output(batch_size)

class _InnerNetworkModule(_ModuleBase):
    def __init__(self, in_shape, order, inputs, layers: torch.nn.ModuleDict, recurrent_layers, intial_values, input_modes):
        super().__init__(in_shape)
        self.order = order
        self.layers = layers
        self.inputs = inputs
        self.recurrent_layers = recurrent_layers  # maps from placeholder names to layer names
        self.initial_values = intial_values # maps from placeholder names to initial_value functions
        self.input_modes = input_modes


    def forward(self, inp, hidden_state):
        state, recurrent_outputs = hidden_state
        new_state = {}
        results = {'input': inp, **{k: v.unsqueeze(0) for k, v in recurrent_outputs.items()}}
        for layer_name in self.order:
            if len(self.inputs[layer_name]) > 1:
                inputs = [results[p].reshape(results[p].shape[:2]+(-1,)) for p in self.inputs[layer_name]]
                if self.input_modes[layer_name] == 'stack':
                    x = torch.cat(inputs, dim=-1)
                elif self.input_modes[layer_name] == 'sum':
                    x = sum(inputs)
                else:
                    x = inputs
            else:
                x = results[self.inputs[layer_name][0]]
            results[layer_name], new_state[layer_name] = self.layers[layer_name](x, state[layer_name])
        new_recurrent_outputs = {ph_name: results[layer_name].squeeze(0) for ph_name, layer_name in self.recurrent_layers.items()}
        return results['output'], (new_state, new_recurrent_outputs)

    def get_initial_state(self, batch_size):
        state = {layer_name: layer.get_initial_state(batch_size) for layer_name, layer in self.layers.items()}
        recurrent_outputs = {ph_name: self.layers[layer_name].get_initial_output(batch_size) for ph_name, layer_name in self.recurrent_layers.items()}
        for ph_name, value_func in self.initial_values:
            recurrent_outputs[ph_name] = value_func(recurrent_outputs[ph_name].shape)
        return state, recurrent_outputs


    def get_initial_output(self, batch_size):
        return self.layers['output'].get_initial_output(batch_size)

class _NestedNetworkModule(_InnerNetworkModule):
    def forward(self, results, hidden_state):
        state, recurrent_outputs = hidden_state
        new_state = {}
        results = {**{k: v.unsqueeze(0) for k, v in recurrent_outputs.items()}, **results}
        for layer_name in self.order:
            if len(self.inputs[layer_name]) > 1:
                inputs = [results[p].reshape(results[p].shape[:2]+(-1,)) for p in self.inputs[layer_name]]
                if self.input_modes[layer_name] == 'stack':
                    x = torch.cat(inputs, dim=-1)
                elif self.input_modes[layer_name] == 'sum':
                    x = sum(inputs)
                else:
                    x = inputs
            else:
                x = results[self.inputs[layer_name][0]]
            results[layer_name], new_state[layer_name] = self.layers[layer_name](x, state[layer_name])
        new_recurrent_outputs = {ph_name: results[layer_name].squeeze(0) for ph_name, layer_name in self.recurrent_layers.items()}
        return results, (new_state, new_recurrent_outputs)