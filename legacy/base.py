import torch
import torch.nn as nn
from utils import StateContainerNew

'''tensor format
Time Modes:
    -single/unrolled
    -time before batch
    -time after batch
    -time in batch
Unrolled/single flag not necessary, read time_size
Data Format:
    -simple
    -complex
    -any
Compression flag


make data flattening a module feature for most modules, only permanently flatten through linear

'''


class ShapeInfo:
    def __init__(self):
        self.time_first = True
        self.time_in_batch = False
        self.batch_size = 1
        self.time_size = 1
        #self.data_dims = () # just simple size tuple?
        #self.data_compressed = False

    def _get_current_time_mode(self):
        return 'tib' if self.time_in_batch else ('tbb' if self.time_size else 'tab')

    def _get_time_and_batch(self):
        return [self.batch_size*self.time_size] if self.time_in_batch else ([self.time_size, self.batch_size] if self.time_first else [self.batch_size, self.time_size])

    def prepare_data(self, tensor, time_modes, data_dims):
        do_transpose = False
        #time dim stuff
        if self._get_current_time_mode() in time_modes:
            target_shape = self._get_time_and_batch()
        elif 'tib' in time_modes:
            target_shape = [self.batch_size*self.time_size]
            self.time_in_batch = True
        else:
            self.time_in_batch = False
            if 'tbb' in time_modes and self.time_first:
                target_shape = [self.time_size, self.batch_size]
            elif 'tab' in time_modes and not self.time_first:
                target_shape = [self.batch_size, self.time_size]
            else:
                target_shape = {'tbb': [self.time_size, self.batch_size], 'tab': [self.batch_size, self.time_size]}[time_modes[0]]
                do_transpose = True
                self.time_first = not self.time_first


        tensor = tensor.reshape(target_shape+data_dims)
        if do_transpose:
            tensor = tensor.transpose(0, 1)

        return tensor

    def change_time(self, new_time):
        self.time_size = new_time


class ModuleBase(nn.Module):
    def __init__(self, in_shape):
        super().__init__()
        self.in_shape = in_shape
        #self.out_shape = in_shape

    #def get_out_shape(self):
       # return self.out_shape

    def get_initial_state(self, batch_size):
        return ()

    #def prepare_data(self, data, meta_dims):
        #return data.reshape(meta_dims[1:]+self.in_shape)


class StatelessWrapper(ModuleBase):
    def __init__(self, inner):
        super().__init__(inner.in_shape)
        self.inner = inner

    def forward(self, x, h):
        return self.inner(x.reshape((x.shape[0]*x.shape[1], )+self.in_shape)), ()




class SequentialModule(ModuleBase):
    def __init__(self, in_shape, modules):
        super().__init__(in_shape)
        self.mlist = nn.ModuleList(modules)
        self.out_shape = self.mlist[-1].get_out_shape()

    def forward(self, inp, hidden_state, inp_info):
        new_state = []
        x = inp
        for i, module in enumerate(self.mlist):
            x, h, inp_info = module(x, hidden_state[i], inp_info)
            new_state.append(h)
        return x, new_state, inp_info

    def get_initial_state(self, batch_size):
        return [module.get_initial_state(batch_size) for module in self.mlist]

    def get_initial_output(self, batch_size):
        self.mlist[-1].get_initial_output(batch_size)

class InnerNetworkModule(ModuleBase):
    def __init__(self, in_shape, order, inputs, layers: nn.ModuleDict):
        super().__init__(in_shape)
        self.order = order
        self.layers = layers
        self.inputs = inputs
        self.out_shape = layers['output'].get_out_shape() #TODO: what if no output layer?

    def forward(self, inp, hidden_state, meta_dims, results=None):
        results = results or {'input': inp}

#class CycleUnroller(ModuleBase):



class OuterNetworkModule(ModuleBase):
    def __init__(self, in_shape, order, inputs, layers: nn.ModuleDict, cycles):
        super().__init__(in_shape)
        self.order = order
        self.layers = layers
        self.cycles = cycles
        self.inputs = inputs
        self.out_shape = layers['output'].get_out_shape()

        self.cycle_inputs = []
        self.cycle_outputs = []

    def run_cycle(self, results, cycle_name):
        inner_results = {name: results[name]}


    def forward(self, inp, hidden_state, meta_dims):
        new_state = {}
        results = {'input': inp}
        result_meta_dims = {'input': meta_dims}
        for layer_name in self.order:
            #TODO: execute cycles
            if len(self.inputs[layer_name]) > 1:
                inputs = [results[p].reshape(result_meta_dims[layer_name]+(-1,)) for p in self.inputs[layer_name]]
                x = torch.cat(inputs, dim=-1)
            else:
                x = results[self.inputs[layer_name][0]]
            x, h, inp_info = module(x, hidden_state[i], inp_info)
            new_state.append(h)
        return x, new_state, inp_info

    #TODO: remove list for single inputs


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