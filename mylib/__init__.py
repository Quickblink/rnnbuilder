from torch import nn
import torch
from typing import Union, Callable, Literal, Optional, Type
from .base.modules import InnerNetworkModule, OuterNetworkModule, NestedNetworkModule, SequentialModule
from .base.factories import ModuleFactory
from .base.utils import shape_sum

class Sequential(ModuleFactory):
    def __init__(self, *module_facs):
        super().__init__()
        self.module_facs = module_facs


    def _shape_change(self, in_shape):
        cur_shape = in_shape
        for factory in self.module_facs:
            cur_shape = factory._shape_change(cur_shape)
        return cur_shape

    def _assemble_module(self, in_shape, unrolled):
        mlist = []
        cur_shape = in_shape
        for factory in self.module_facs:
            new_module = factory._assemble_module(cur_shape, unrolled)
            cur_shape = factory._shape_change(cur_shape)
            mlist.append(new_module)
        return SequentialModule(in_shape, mlist)

class LayerInput:
    _mode = 'stack'


class LayerBase(LayerInput):
    def __init__(self):
        self._registered = False

    def sum(self, *layers: 'LayerBase'):
        if not self._registered:
            raise Exception('This layer is not part of a network yet. Chaining is not allowed.')
        return Sum(self, *layers)

    def stack(self, *layers: 'LayerBase'):
        if not self._registered:
            raise Exception('This layer is not part of a network yet. Chaining is not allowed.')
        return Stack(self, *layers)

    def apply(self, *module_facs, placeholder=None):
        if not self._registered:
            raise Exception('This layer is not part of a network yet. Chaining is not allowed.')
        return Layer(self, module_facs, placeholder=placeholder)

class InputBase(LayerInput):
    def __init__(self, *layers: LayerBase):
        self.layers = layers

    def apply(self, *module_facs, placeholder=None):
        return Layer(self, module_facs, placeholder=placeholder)

class Stack(InputBase):
    pass

class Sum(InputBase):
    _mode = 'sum'


class Placeholder(LayerBase):
    inputs = set()
    def __init__(self, initial_value : Callable[[tuple], torch.Tensor] = None):
        super().__init__()
        self.initial_value = initial_value
        self._layer = None

    def _get_layer(self):
        if self._layer is None:
            raise Exception('Placeholder not assigned to layer.')
        return self._layer


class Layer(LayerBase):
    def __init__(self, input: LayerInput, factory, placeholder=None):
        super().__init__()
        if isinstance(input, InputBase):
            self.inputs = input.layers
        else:
            self.inputs = (input,)
        self.mode = input._mode
        if not isinstance(factory, ModuleFactory):
            self.factory = factory[0] if len(factory) == 1 else Sequential(*factory)
        else:
            self.factory = factory
        self.placeholder = placeholder
        if placeholder:
            placeholder._layer = self


class Network(ModuleFactory):
    def __init__(self):
        super().__init__() # TODO: check setattr calls in this
        self._layers = {}
        self._ph = {}
        self._reverse_laph = {}
        self._og_order = []
        self.input = Placeholder()
        del self._ph['input']

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if not isinstance(value, LayerBase):
            return
        if key in self._ph:
            if isinstance(value, Placeholder) or (value.placeholder and value.placeholder is not self._ph[key]):
                raise Exception('Two placeholders given for a layer.')
            self._ph[key]._layer = value
            self._ph[key +'_ph'] = self._ph[key]
            self._reverse_laph[self._ph[key]] = key +'_ph'
            del self._ph[key]
        if key in self._layers:
            raise Exception('Overwriting layers with other layers is not allowed.')
        if isinstance(value, Layer):
            for input in value.inputs:
                if input not in self._reverse_laph:
                    raise Exception('Input is not in network.')
            self._layers[key] = value
            self._og_order.append(key)
        if isinstance(value, Placeholder):
            self._ph[key] = value
        self._reverse_laph[value] = key


    def _compute_shapes(self, in_shape):
        ph_lookup = {ph_name: self._reverse_laph[ph._get_layer()] for ph_name, ph in self._ph.items()}
        input_names = {layer_name: {self._reverse_laph[inp_lay] for inp_lay in layer.inputs} for layer_name, layer in self._layers.items()}
        input_no_ph = {layer_name: {(ph_lookup[inp] if inp in ph_lookup else inp) for inp in inputs} for layer_name, inputs in input_names.items()}
        shapes = {'input': in_shape, **{layer: None for layer in self._layers}}
        layers = set(self._layers)
        while layers:
            found_new = None
            for layer_name in layers:
                inputs = input_no_ph[layer_name]
                inp_shape = shapes[next(iter(inputs))] if len(inputs) == 1 else shape_sum \
                    ([shapes[layer_name] for layer_name in inputs])
                shapes[layer_name] = self._layers[layer_name].factory._shape_change(inp_shape)
                if shapes[layer_name] is not None:
                    found_new = layer_name
                    break
            layers.remove(found_new)
        input_shapes = {layer_name: shapes[next(iter(input_no_ph[layer_name]))] if len(input_no_ph[layer_name]) == 1
        else shape_sum([shapes[layer_name] for layer_name in input_no_ph[layer_name]]) for layer_name in self._layers}
        return shapes, input_shapes

    def _shape_change(self, in_shape):
        shapes, _ = self._compute_shapes(in_shape)
        return shapes['output']

    def _compute_execution_order(self, input_no_ph, input_names, ph_rev):

        # Find reachables
        reachable = {}
        for path, inp_list in input_no_ph.items():
            visited = {*inp_list}
            todo = {*inp_list}
            while todo:
                todo.update(set(input_no_ph[todo.pop()]).difference(visited))
                visited.update(todo)
            reachable[path] = visited

        i = 0
        cycles_dict = {}
        full_cycles_dict = {}
        in_cycles = set()
        for path, reach in reachable.items():
            if path in in_cycles:
                continue
            cycle_with = []
            for other in reach:
                if path in reachable[other]:
                    cycle_with.append(other)
            if cycle_with:
                cycle_with.sort(key=self._og_order.index)
                cycles_dict[f'c{i}'] = cycle_with
                full_cycles_dict[f'c{i}'] = set(cycle_with)
                for layer_name in cycle_with:
                    if layer_name in ph_rev:
                        full_cycles_dict[f'c{i}'].add(ph_rev[layer_name])
                i += 1
                in_cycles.update(cycle_with)
        remaining = set(self._og_order).difference(in_cycles)

        requirements = {**input_no_ph}
        new_inputs = {layer_name: input_names[layer_name] for layer_name in remaining}
        for c_name, cycle in cycles_dict.items():
            req = set()
            req2 = set()
            for path in cycle:
                req.update(input_no_ph[path])
                req2.update(input_names[path])
            requirements[c_name] = req.difference(cycle)
            new_inputs[c_name] = req.difference(full_cycles_dict[c_name])


        nodes = remaining.union(cycles_dict.keys())
        computed_order = []
        done = {'input'}
        while nodes:
            for node in nodes:
                if requirements[node].issubset(done):
                    done.add(node)
                    if node in cycles_dict:
                        done.update(cycles_dict[node])
                    computed_order.append(node)
            nodes.difference_update(done)

        req_remain = set.union(*([input_names[layer_name] for layer_name in remaining]+[{'output'}]))
        cycle_outputs = {c_name: req_remain.intersection(cycle) for c_name, cycle in full_cycles_dict.items()}

        return computed_order, remaining, cycles_dict, new_inputs, cycle_outputs



    def _assemble_module(self, in_shape, unrolled):
        if not 'output' in self._layers:
            raise Exception('Output layer missing.')
        initial_values = {}
        for ph_name, ph in self._ph.items():
            if ph.initial_value:
                initial_values[ph_name] = ph.initial_value
        out_shapes, in_shapes = self._compute_shapes(in_shape)
        ph_lookup = {ph_name: self._reverse_laph[ph._get_layer()] for ph_name, ph in self._ph.items()}
        input_names = {layer_name: {self._reverse_laph[inp_lay] for inp_lay in layer.inputs} for layer_name, layer in self._layers.items()}
        input_no_ph = {layer_name: {(ph_lookup[inp] if inp in ph_lookup else inp) for inp in inputs} for layer_name, inputs in input_names.items()}
        input_no_ph['input'] = set()
        input_modes = {layer_name: layer.mode for layer_name, layer in self._layers.items()}
        if not unrolled:
            module_dict = nn.ModuleDict()
            ph_rev = {layer_name: ph_name for ph_name, layer_name in ph_lookup.items()}
            outer_order, outer_layers, cycles_layers, new_inputs, cycles_outputs = self._compute_execution_order \
                (input_no_ph, input_names, ph_rev)
            for name, cycle in cycles_layers.items():
                recurent = {ph_rev[layer_name]: layer_name for layer_name in cycle}
                init_values = {name: initial_values[name] for name in set(recurent).intersection(initial_values)}
                module_dict_inner = nn.ModuleDict()
                for layer in cycle:
                    module_dict_inner[layer] = self._layers[layer].factory._assemble_module(in_shapes[layer], True)
                inputs = {layer: input_names[layer] for layer in cycle}
                module_dict[name] = NestedNetworkModule(in_shape, cycle, inputs, module_dict_inner, recurent, init_values, input_modes)
            for name in outer_layers:
                module_dict[name] = self._layers[name].factory._assemble_module(in_shapes[name], False)
            outer_ph_rev = {layer_name: ph_rev[layer_name] for layer_name in set(outer_order).intersection(ph_rev.keys())}
            return OuterNetworkModule(in_shape, outer_order, new_inputs, module_dict, cycles_outputs, outer_ph_rev, input_modes)
        else:
            module_dict = nn.ModuleDict()
            for layer in self._og_order:
                module_dict[layer] = self._layers[layer].factory._assemble_module(in_shapes[layer], True)
            inputs = {layer: input_names[layer] for layer in self._og_order}
            return InnerNetworkModule(in_shape, self._og_order, inputs, module_dict, ph_lookup, initial_values, input_modes)
