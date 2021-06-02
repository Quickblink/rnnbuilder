"""
rnnbuilder is a library for building PyTorch models in a modular way.

##Modules and Factories
Instead of building networks from modules directly, this library uses separate factory objects. They are used as
specification for a network or sub-network and can be invoked to yield independent models from that specification.

Classic PyTorch code:
```
input_shape = (10,)
linear = torch.nn.Linear(in_features=10,out_features=10)
model = torch.nn.Sequential(linear, linear)

input = torch.rand(input_shape)
output = model(input)
```
The above code applies a matrix multiplication with a *single weight matrix* twice to the input.

rnnbuilder **(not equivalent)**:
```
input_shape = (7,)
linear = rnnbuilder.nn.Linear(out_features=10)
sequential = rnnbuilder.Sequential(linear, linear)
model = sequential.make_model(input_shape)

input = torch.rand(input_shape)
output = model(input)
```
This code does not reuse any layers. Instead it will initialize a model with two separate linear layers, both with an
output size of 10. The first has in_features=7 and the second in_features=10. The model is generated with an additional
call to the outermost factory, in this case 'sequential'.

The use of factory classes replaces the need to define classes for every architecture and offers lazy initialization
without the need to precalculate the size of inputs (more useful for convolutions and recurrent layers).

##Network class
The `Network` class is the main tool to build powerful (and recurrent) architectures. Take a look at the documentation
to gain an overview of what is possible.

###Example: LSTM
![LSTM computational graph](img/rnnbuilder_graph_3.svg)
The above computational graph shows a typical 1-layer LSTM identical in function to `torch.nn.LSTM`. The blue boxes show
the `Layer`s of the `Network`, the green circles present the operations associated to these layers and the red boxes
represent the values of the corresponding layers at the previous time step. They are represented in code by
`Placeholder`s. Here is the whole network built using rnnbuilder:

```
hidden_size = 32
n = rnnbuilder.Network()

n.output = rnnbuilder.Placeholder()
n.h_and_i = n.input.stack(n.output)

n.i = n.h_and_i.apply(Linear(hidden_size), Sigmoid())
n.f = n.h_and_i.apply(Linear(hidden_size), Sigmoid())
n.o = n.h_and_i.apply(Linear(hidden_size), Sigmoid())
n.g = n.h_and_i.apply(Linear(hidden_size), Tanh())

n.c = rnnbuilder.Placeholder()
n.c_1 = n.f.append(n.c).apply(Hadamard())
n.c_2 = n.i.append(n.g).apply(Hadamard())

n.c = n.c_1.sum(n.c_2)
n.tan_c = n.c.apply(Tanh())
n.output = n.o.append(n.tan_c).apply(Hadamard())
```

##Modules
rnnbuilder provides a number of factories for standard torch modules under `rnnbuilder.nn`. These have identical
signatures to the `torch.nn` modules (without in_features or in_channels). Additionally, some specific factories for RNN
and SNN modules are provided under `rnnbuilder.rnn` and `rnnbuilder.snn`.

If you need to use any other modules, `rnnbuilder.custom` provides a base class and registration methods to retrieve
corresponding factory classes.



"""
import torch as _torch
from typing import Union as _Union, Callable as _Callable, Iterable as _Iterable
from .base._modules import _InnerNetworkModule, _OuterNetworkModule, _NestedNetworkModule, _SequentialModule
from .base import ModuleFactory as _ModuleFactory, LayerInput as _LayerInput, LayerBase as _LayerBase, InputBase as _InputBase
from .base._utils import _shape_sum, _any
from . import custom, rnn, snn, nn




class Sequential(_ModuleFactory):
    """Equivalent to `torch.nn.Sequential`. Accepts multiple `ModuleFactory` or an iterable of `ModuleFactory`s.
    The corresponding modules are executed sequentially and associated state is managed automatically."""

    def __init__(self, *module_factory: _Union[_ModuleFactory, _Iterable[_ModuleFactory]]):
        super().__init__()
        self.module_facs = module_factory if all([isinstance(m, _ModuleFactory) for m in module_factory]) \
            else list(module_factory[0])

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
        return _SequentialModule(in_shape, mlist)


class Stack(_InputBase):
    """Stacks inputs along the first data dimension using `torch.cat`. Is used as an input to `Layer` in `Network`"""
    pass


class Sum(_InputBase):
    """Adds up inputs element-wise. Is used as an input to `Layer` in `Network`"""
    _mode = 'sum'

class List(_InputBase):
    """Provides multiple inputs to the next `Layer` in `Network` as a list. Only works with specific factories."""
    _mode = 'list'


class Placeholder(_LayerBase):
    """Can be assigned as an attribute to a `Network` to represent a layer output of the previous time step. Needs to be
    linked to a `Layer` by either overwriting the attribute with a `Layer` or handing it directly to the `Layer` as an
    optional initialization parameter.

    Args:
        initial_value: (optional) function that returns the initial output value used for the first step /
            initial state. Per default, outputs are initialized to zero unless overwritten by the `Layer`'s class."""
    _inputs = set()

    def __init__(self, initial_value: _Callable[[tuple], _torch.Tensor] = None):
        super().__init__()
        self.initial_value = initial_value
        self._layer = None

    def _get_layer(self):
        if self._layer is None:
            raise Exception('Placeholder not assigned to layer.')
        return self._layer


class Layer(_LayerBase):
    """Can be assigned as attribute to a `Network` as its main building block. When assigned to an attribute holding a
    `Placeholder`, they are linked automatically.

    Args:
        input: The input to the module(s) in this layer. Given a network n, can be one of n.input (the input to the
            network), n.some_layer, n.some_placeholder or an input aggregation (`Sum` or `Stack`).
        factory: Either a single `ModuleFactory` or an iterable of `ModuleFactory`s that is automatically wrapped by a
            `Sequential`.
        placeholder: (optional) A `Placeholder` to link to this layer. This can be useful if the placeholder is still
            required after layer definition (prohibiting overwriting the attribute as described above)

    """
    def __init__(self, input: _LayerInput, factory: _Union[_ModuleFactory, _Iterable[_ModuleFactory]],
                 placeholder: Placeholder = None):
        super().__init__()
        if isinstance(input, _InputBase):
            self._inputs = input.layers
        else:
            self._inputs = (input,)
        self.mode = input._mode
        if not isinstance(factory, _ModuleFactory):
            factory = list(factory)
            self.factory = factory[0] if len(factory) == 1 else Sequential(*factory)
        else:
            self.factory = factory
        self.placeholder = placeholder
        if placeholder:
            placeholder._layer = self


class Network(_ModuleFactory):
    """Main class for dynamic network building. After initialization, `Layer`s and `Placeholder`s can be added in the
    form of attributes. The input to the network is available as an attribute 'input' and the the user is required to
    define a layer called 'output'. There are multiple supported methods to assemble a layer.

    Example:
        ```
        >>>n = Network()
        ...n.first_layer_prev = Placeholder()
        ...n.first_layer = Layer(Stack(n.input, n.first_layer_prev), some_factory1, placeholder=n.first_layer_prev)
        ...n.second_layer = n.first_layer_prev.stack(n.input).apply(some_factory2)
        ...n.output = n.second_layer.apply(some_factory3)
        ```

    The assembled object is a `ModuleFactory` that can be used like any other. It automatically detects cycles and
    unrolls execution where necessary.
    """

    def __init__(self):
        super().__init__()
        self._layers = {}
        self._ph = {}
        self._reverse_laph = {}
        self._og_order = []
        self.input = Placeholder()
        del self._ph['input']

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if not isinstance(value, _LayerInput):
            return
        if isinstance(value, _InputBase):
            value = value.apply(_ModuleFactory())
        value._registered = True
        if key in self._ph:
            if isinstance(value, Placeholder) or (value.placeholder and value.placeholder is not self._ph[key]):
                raise Exception('Two placeholders given for a layer.')
            self._ph[key]._layer = value
            self._ph[key + '_ph'] = self._ph[key]
            self._reverse_laph[self._ph[key]] = key + '_ph'
            del self._ph[key]
        if key in self._layers:
            raise Exception('Overwriting layers with other layers is not allowed.')
        if isinstance(value, Layer):
            for input in value._inputs:
                if input not in self._reverse_laph:
                    raise Exception('Input is not in network.')
            self._layers[key] = value
            self._og_order.append(key)
        if isinstance(value, Placeholder):
            self._ph[key] = value
        self._reverse_laph[value] = key

    def _calc_input_shape(self, shapes, input_mode): #TODO: make only list and move rest in operations
        if input_mode == 'sum':
            if not _any(shapes):
                return None
            shape = _any(shapes)
            for other in shapes:
                if other and other != shape:
                    raise Exception('Shape mismatch in sum of inputs.')
            return shape
        elif input_mode == 'stack':
            if not all(shapes):
                return None
            dims = max([len(x) for x in shapes])
            reference = (1,)*(dims-len(shapes[0]))+(shapes[0])
            first_dim = reference[0]
            test = reference[1:]
            for shape in shapes[1:]:
                shape = (1,)*(dims-len(shape))+(shape)
                if not shape[1:] == test:
                    return _shape_sum(shapes)
                first_dim += shape[0]
            return (first_dim,)+test
        elif input_mode == 'list':
            return shapes
        else:
            raise Exception('input mode not implemented, please report this')


    def _compute_shapes(self, in_shape, input_modes):
        ph_lookup = {ph_name: self._reverse_laph[ph._get_layer()] for ph_name, ph in self._ph.items()}
        input_names = {layer_name: {self._reverse_laph[inp_lay] for inp_lay in layer._inputs} for layer_name, layer in
                       self._layers.items()}
        input_no_ph = {layer_name: {(ph_lookup[inp] if inp in ph_lookup else inp) for inp in inputs} for
                       layer_name, inputs in input_names.items()}
        shapes = {'input': in_shape, **{layer: None for layer in self._layers}}
        layers = set(self._layers)
        while layers:
            found_new = None
            for layer_name in layers:
                inputs = input_no_ph[layer_name]
                inp_shape = self._calc_input_shape([shapes[layer_name] for layer_name in inputs], input_modes[layer_name])
                shapes[layer_name] = self._layers[layer_name].factory._shape_change(inp_shape)
                if shapes[layer_name] is not None:
                    found_new = layer_name
                    break
            layers.remove(found_new)
        input_shapes = {layer_name: self._calc_input_shape([shapes[input_name] for input_name
                        in input_no_ph[layer_name]], input_modes[layer_name]) for layer_name in self._layers}
        return shapes, input_shapes

    def _shape_change(self, in_shape):
        input_modes = {layer_name: layer.mode for layer_name, layer in self._layers.items()}
        shapes, _ = self._compute_shapes(in_shape, input_modes)
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

        req_remain = set.union(*([set(input_names[layer_name]) for layer_name in remaining] + [{'output'}]))
        cycle_outputs = {c_name: req_remain.intersection(cycle) for c_name, cycle in full_cycles_dict.items()}

        return computed_order, remaining, cycles_dict, new_inputs, cycle_outputs

    def _assemble_module(self, in_shape, unrolled):
        if not 'output' in self._layers:
            raise Exception('Output layer missing.')
        initial_values = {}
        for ph_name, ph in self._ph.items():
            if ph.initial_value:
                initial_values[ph_name] = ph.initial_value
        input_modes = {layer_name: layer.mode for layer_name, layer in self._layers.items()}
        out_shapes, in_shapes = self._compute_shapes(in_shape, input_modes)
        ph_lookup = {ph_name: self._reverse_laph[ph._get_layer()] for ph_name, ph in self._ph.items()}
        input_names = {layer_name: [self._reverse_laph[inp_lay] for inp_lay in layer._inputs] for layer_name, layer in
                       self._layers.items()}
        input_no_ph = {layer_name: {(ph_lookup[inp] if inp in ph_lookup else inp) for inp in inputs} for
                       layer_name, inputs in input_names.items()}
        input_no_ph['input'] = set()
        if not unrolled:
            module_dict = _torch.nn.ModuleDict()
            ph_rev = {layer_name: ph_name for ph_name, layer_name in ph_lookup.items()}
            outer_order, outer_layers, cycles_layers, new_inputs, cycles_outputs = self._compute_execution_order \
                (input_no_ph, input_names, ph_rev)
            for name, cycle in cycles_layers.items():
                recurent = {ph_rev[layer_name]: layer_name for layer_name in set(cycle).intersection(ph_rev)}
                init_values = {name: initial_values[name] for name in set(recurent).intersection(initial_values)}
                module_dict_inner = _torch.nn.ModuleDict()
                for layer in cycle:
                    module_dict_inner[layer] = self._layers[layer].factory._assemble_module(in_shapes[layer], True)
                inputs = {layer: input_names[layer] for layer in cycle}
                module_dict[name] = _NestedNetworkModule(in_shape, cycle, inputs, module_dict_inner, recurent,
                                                         init_values, input_modes)
            for name in outer_layers:
                module_dict[name] = self._layers[name].factory._assemble_module(in_shapes[name], False)
            outer_ph_rev = {layer_name: ph_rev[layer_name] for layer_name in
                            set(outer_order).intersection(ph_rev.keys())}
            return _OuterNetworkModule(in_shape, outer_order, new_inputs, module_dict, cycles_outputs, outer_ph_rev,
                                       input_modes)
        else:
            module_dict = _torch.nn.ModuleDict()
            for layer in self._og_order:
                module_dict[layer] = self._layers[layer].factory._assemble_module(in_shapes[layer], True)
            inputs = {layer: input_names[layer] for layer in self._og_order}
            return _InnerNetworkModule(in_shape, self._og_order, inputs, module_dict, ph_lookup, initial_values,
                                       input_modes)
